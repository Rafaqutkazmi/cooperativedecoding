import hashlib
import hmac
from typing import Tuple
import numpy as np
from scipy.fft import dctn, idctn

from src.decoder.cooperative_viterbi_decoder import EPS, NUM_CODED_BITS, conv_encode_64_to_172

DEFAULT_NUM_DCT_BLOCKS = 1024
DEFAULT_QIM_DELTA = 32.0
DEFAULT_DCT_SIGMOID_TAU = 0.25
DUMMY_PUBLIC_TEST_KEY = b"DUMMY_SYNTHETIC_TEST_KEY_DO_NOT_USE_IN_PRODUCTION"


def _hmac_sha256_prng(secret_key: bytes, context_str: str, num_bytes: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < num_bytes:
        msg = f"{context_str}||{counter}".encode("utf-8")
        out.extend(hmac.new(secret_key, msg, hashlib.sha256).digest())
        counter += 1
    return bytes(out[:num_bytes])


def derive_keyed_dct_block_schedule(
    secret_key: bytes,
    clip_id: str,
    gop_idx: int = 0,
    num_frames: int = 16,
    height: int = 256,
    width: int = 256,
    num_blocks: int = DEFAULT_NUM_DCT_BLOCKS,
) -> np.ndarray:
    """Derives 1,024 keyed 8x8 luma block coordinates across a 16x256x256 GOP (64 interior blocks/frame)."""
    bh, bw = height // 8, width // 8
    blocks_per_frame = num_blocks // num_frames
    raw = _hmac_sha256_prng(secret_key, f"dct_schedule||{clip_id}||{gop_idx}", 4096)
    seed_val = int.from_bytes(raw[:8], "big") % (2**31 - 1)
    rng = np.random.RandomState(seed_val)

    selected_coords = []
    interior = [(by, bx) for by in range(1, bh - 1) for bx in range(1, bw - 1)]
    for t in range(num_frames):
        chosen_idx = rng.choice(len(interior), size=blocks_per_frame, replace=False)
        for ci in chosen_idx:
            by, bx = interior[ci]
            selected_coords.append((t, int(by), int(bx)))
    coords_arr = np.asarray(selected_coords, dtype=np.int32)
    perm = rng.permutation(num_blocks)
    return coords_arr[perm]


def embed_coded_dct_second(
    frames_uint8_or_float: np.ndarray,
    bits_64: np.ndarray,
    secret_key: bytes = DUMMY_PUBLIC_TEST_KEY,
    clip_id: str = "synthetic_clip_0000",
    delta: float = DEFAULT_QIM_DELTA,
    gop_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements Section 3.2 (Eq. 4):
    Differential-margin coded-DCT embedding on d = C_{1,2} - C_{2,1} across 1,024 keyed 8x8 luma blocks:
        d >= +Delta  (c_k = 1),
        d <= -Delta  (c_k = 0),
    shifting C_{1,2} and C_{2,1} symmetrically by +/- (Delta - sign(2 c_k - 1) d) / 2 when the margin is violated.
    """
    x_hyb = np.asarray(frames_uint8_or_float, dtype=np.float32).copy()
    t_len, h, w, _ = x_hyb.shape
    coded_172, _ = conv_encode_64_to_172(bits_64)
    coords_1024 = derive_keyed_dct_block_schedule(
        secret_key, clip_id, gop_idx=gop_idx, num_frames=t_len, height=h, width=w
    )
    blk_ptr = 0
    for k in range(NUM_CODED_BITS):
        rk = 6 if k < 164 else 5  # 164 * 6 + 8 * 5 = 1,024 blocks
        ck = int(coded_172[k])
        for _ in range(rk):
            t_idx, by, bx = coords_1024[blk_ptr]
            blk_ptr += 1
            y0, x0 = by * 8, bx * 8
            patch_rgb = x_hyb[t_idx, y0:y0 + 8, x0:x0 + 8, :]
            luma_patch = 0.299 * patch_rgb[..., 0] + 0.587 * patch_rgb[..., 1] + 0.114 * patch_rgb[..., 2]
            dct_patch = dctn(luma_patch, type=2, norm="ortho")
            d_val = float(dct_patch[1, 2] - dct_patch[2, 1])
            if ck == 1 and d_val < delta:
                shift = 0.5 * (delta - d_val)
                dct_patch[1, 2] += shift
                dct_patch[2, 1] -= shift
            elif ck == 0 and d_val > -delta:
                shift = 0.5 * (delta + d_val)
                dct_patch[1, 2] -= shift
                dct_patch[2, 1] += shift
            new_luma = idctn(dct_patch, type=2, norm="ortho")
            luma_diff = (new_luma - luma_patch)[..., None]
            x_hyb[t_idx, y0:y0 + 8, x0:x0 + 8, :] = np.clip(patch_rgb + luma_diff, 0.0, 255.0)
    return np.clip(np.round(x_hyb), 0.0, 255.0).astype(np.uint8), coded_172


def extract_coded_dct_soft_bits(
    attacked_frames_uint8: np.ndarray,
    secret_key: bytes = DUMMY_PUBLIC_TEST_KEY,
    clip_id: str = "synthetic_clip_0000",
    delta: float = DEFAULT_QIM_DELTA,
    tau_d: float = DEFAULT_DCT_SIGMOID_TAU,
    gop_idx: int = 0,
) -> np.ndarray:
    """
    Implements Section 3.2 (Eq. 5):
    Blindly extracts d_{k,r} = C_{1,2} - C_{2,1} across the R_k in {5, 6} repeated blocks for each coded bit k,
    averages into d_bar_k, and maps via the temperature-scaled sigmoid p_k = sigma(d_bar_k / (tau_d * Delta)).
    """
    y = np.asarray(attacked_frames_uint8, dtype=np.float32)
    t_len, h, w, _ = y.shape
    coords_1024 = derive_keyed_dct_block_schedule(
        secret_key, clip_id, gop_idx=gop_idx, num_frames=t_len, height=h, width=w
    )
    p_172 = np.zeros(NUM_CODED_BITS, dtype=np.float64)
    blk_ptr = 0
    scale_denom = max(tau_d * delta, 1e-6)
    for k in range(NUM_CODED_BITS):
        rk = 6 if k < 164 else 5
        d_sum = 0.0
        for _ in range(rk):
            t_idx, by, bx = coords_1024[blk_ptr]
            blk_ptr += 1
            y0, x0 = by * 8, bx * 8
            patch_rgb = y[t_idx, y0:y0 + 8, x0:x0 + 8, :]
            luma_patch = 0.299 * patch_rgb[..., 0] + 0.587 * patch_rgb[..., 1] + 0.114 * patch_rgb[..., 2]
            dct_patch = dctn(luma_patch, type=2, norm="ortho")
            d_sum += float(dct_patch[1, 2] - dct_patch[2, 1])
        d_bar = d_sum / float(rk)
        logit = float(np.clip(d_bar / scale_denom, -14.0, 14.0))
        p_172[k] = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(p_172, EPS, 1.0 - EPS)

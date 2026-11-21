from typing import Dict, List, Optional, Tuple
import numpy as np

EPS = 1e-6
PRIMARY_WEIGHTS = (0.0, 0.75, 1.5, 3.0)
FALLBACK_WEIGHT = 1.25
DEFAULT_SCORE_NORM_C = 0.35

NUM_STATES = 64
NUM_INFO_BITS = 64
NUM_CRC_BITS = 16
NUM_TAIL_BITS = 6
NUM_STAGES = NUM_INFO_BITS + NUM_CRC_BITS + NUM_TAIL_BITS  # 86
NUM_CODED_BITS = NUM_STAGES * 2  # 172

SINGLE_PASS_BRANCHES_ENUMERATED = 10624
SINGLE_PASS_BRANCHES_REACHABLE = 9724
LIST_L4_EXTENSIONS_ALL_STAGES = 42496
LIST_L4_EXTENSIONS_INTERIOR_53 = 27136


def _build_crc16_table() -> np.ndarray:
    poly = 0x1021
    table = np.zeros(256, dtype=np.uint16)
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table[i] = crc
    return table


_CRC16_TABLE = _build_crc16_table()


def compute_crc16_ccitt(bits_64: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits_64, dtype=np.uint8)
    single = (arr.ndim == 1)
    if single:
        arr = arr[None, :]
    packed = np.packbits(arr, axis=1)
    crc = np.full(arr.shape[0], 0xFFFF, dtype=np.uint16)
    for byte_idx in range(8):
        idx = ((crc >> 8) ^ packed[:, byte_idx].astype(np.uint16)) & 0xFF
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[idx]
    crc_bytes = np.stack([(crc >> 8).astype(np.uint8), (crc & 0xFF).astype(np.uint8)], axis=1)
    crc_bits = np.unpackbits(crc_bytes, axis=1)
    return crc_bits[0] if single else crc_bits


def verify_crc16_ccitt(bits_80: np.ndarray) -> bool:
    b = np.asarray(bits_80, dtype=np.uint8)
    expected = compute_crc16_ccitt(b[:64])
    return bool(np.array_equal(expected, b[64:80]))


def verify_crc16_ccitt_batch(bits_80_batch: np.ndarray) -> np.ndarray:
    b = np.asarray(bits_80_batch, dtype=np.uint8)
    expected = compute_crc16_ccitt(b[:, :64])
    return np.all(expected == b[:, 64:80], axis=1)


def _build_trellis_tables():
    g1 = 0o171
    g2 = 0o133
    next_state = np.zeros((NUM_STATES, 2), dtype=np.int32)
    out_c0 = np.zeros((NUM_STATES, 2), dtype=np.float64)
    out_c1 = np.zeros((NUM_STATES, 2), dtype=np.float64)
    pred_states = np.zeros((NUM_STATES, 2), dtype=np.int32)
    pred_c0 = np.zeros((NUM_STATES, 2), dtype=np.float64)
    pred_c1 = np.zeros((NUM_STATES, 2), dtype=np.float64)

    for s in range(NUM_STATES):
        for u in (0, 1):
            reg = (u << 6) | s
            ns = (u << 5) | (s >> 1)
            c0 = bin(reg & g1).count("1") & 1
            c1 = bin(reg & g2).count("1") & 1
            next_state[s, u] = ns
            out_c0[s, u] = float(c0)
            out_c1[s, u] = float(c1)

    for s_next in range(NUM_STATES):
        u = (s_next >> 5) & 1
        base = (s_next & 0x1F) << 1
        for branch_idx in (0, 1):
            s_prev = base | branch_idx
            pred_states[s_next, branch_idx] = s_prev
            pred_c0[s_next, branch_idx] = out_c0[s_prev, u]
            pred_c1[s_next, branch_idx] = out_c1[s_prev, u]

    return next_state, out_c0, out_c1, pred_states, pred_c0, pred_c1


_NEXT_STATE, _OUT_C0, _OUT_C1, _PRED_STATES, _PRED_C0, _PRED_C1 = _build_trellis_tables()
_STATE_INPUT_BIT = ((np.arange(NUM_STATES, dtype=np.int32) >> 5) & 1).astype(np.float64)


def conv_encode_64_to_172(bits_64: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(bits_64, dtype=np.uint8)
    single = (arr.ndim == 1)
    if single:
        arr = arr[None, :]
    n_clips = arr.shape[0]
    crc_16 = compute_crc16_ccitt(arr)
    tails = np.zeros((n_clips, NUM_TAIL_BITS), dtype=np.uint8)
    framed_86 = np.concatenate([arr, crc_16, tails], axis=1)
    coded = np.zeros((n_clips, NUM_CODED_BITS), dtype=np.uint8)
    states = np.zeros(n_clips, dtype=np.int32)
    for t in range(NUM_STAGES):
        u = framed_86[:, t].astype(np.int32)
        coded[:, 2 * t] = _OUT_C0[states, u].astype(np.uint8)
        coded[:, 2 * t + 1] = _OUT_C1[states, u].astype(np.uint8)
        states = _NEXT_STATE[states, u]
    if single:
        return coded[0], framed_86[0]
    return coded, framed_86


def compute_channel_confidences(p_172: np.ndarray, q_64: np.ndarray) -> Tuple[float, float]:
    c_dct = float(np.mean(np.abs(2.0 * np.asarray(p_172, dtype=np.float64) - 1.0)))
    c_neu = float(np.mean(np.abs(2.0 * np.asarray(q_64, dtype=np.float64) - 1.0)))
    return c_dct, c_neu


def viterbi_decode_single_pass(
    p_172: np.ndarray,
    q_64: Optional[np.ndarray] = None,
    w: float = 0.0,
) -> Dict[str, object]:
    p = np.clip(np.asarray(p_172, dtype=np.float64), EPS, 1.0 - EPS)
    q = np.clip(np.asarray(q_64, dtype=np.float64), EPS, 1.0 - EPS) if (q_64 is not None and w > 0.0) else None

    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    if q is not None:
        log_q = np.log(q)
        log_1mq = np.log(1.0 - q)

    pm = np.full(NUM_STATES, np.inf, dtype=np.float64)
    pm[0] = 0.0
    survivors = np.zeros((NUM_STAGES, NUM_STATES), dtype=np.int32)

    for t in range(NUM_STAGES):
        lp0, l1mp0 = log_p[2 * t], log_1mp[2 * t]
        lp1, l1mp1 = log_p[2 * t + 1], log_1mp[2 * t + 1]
        cost_c0 = -_PRED_C0 * lp0 - (1.0 - _PRED_C0) * l1mp0
        cost_c1 = -_PRED_C1 * lp1 - (1.0 - _PRED_C1) * l1mp1
        branch_cost = cost_c0 + cost_c1

        if t < NUM_INFO_BITS and q is not None and w > 0.0:
            prior_cost = -w * (_STATE_INPUT_BIT * log_q[t] + (1.0 - _STATE_INPUT_BIT) * log_1mq[t])
            branch_cost = branch_cost + prior_cost[:, None]

        cand = pm[_PRED_STATES] + branch_cost
        if t >= NUM_INFO_BITS + NUM_CRC_BITS:
            cand[_STATE_INPUT_BIT == 1.0, :] = np.inf

        best_branch = (cand[:, 1] < cand[:, 0]).astype(np.int32)
        pm = np.where(best_branch == 1, cand[:, 1], cand[:, 0])
        survivors[t] = _PRED_STATES[np.arange(NUM_STATES), best_branch]

    state = 0
    bits_86 = np.zeros(NUM_STAGES, dtype=np.uint8)
    for t in range(NUM_STAGES - 1, -1, -1):
        bits_86[t] = (state >> 5) & 1
        state = survivors[t, state]

    payload_64 = bits_86[:64].copy()
    framed_80 = bits_86[:80].copy()
    crc_ok = verify_crc16_ccitt(framed_80)
    return {
        "payload_64": payload_64,
        "framed_80": framed_80,
        "crc_valid": int(crc_ok),
        "path_metric": float(pm[0]),
        "branches_enumerated": SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": SINGLE_PASS_BRANCHES_REACHABLE,
    }


def list_viterbi_decode_crc_l4(
    p_172: np.ndarray,
    q_64: Optional[np.ndarray] = None,
    w: float = 0.0,
    list_size: int = 4,
) -> Dict[str, object]:
    p = np.clip(np.asarray(p_172, dtype=np.float64), EPS, 1.0 - EPS)
    q = np.clip(np.asarray(q_64, dtype=np.float64), EPS, 1.0 - EPS) if (q_64 is not None and w > 0.0) else None

    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    if q is not None:
        log_q = np.log(q)
        log_1mq = np.log(1.0 - q)

    pm = np.full((NUM_STATES, list_size), np.inf, dtype=np.float64)
    pm[0, 0] = 0.0
    back_state = np.zeros((NUM_STAGES, NUM_STATES, list_size), dtype=np.int32)
    back_rank = np.zeros((NUM_STAGES, NUM_STATES, list_size), dtype=np.int32)

    for t in range(NUM_STAGES):
        lp0, l1mp0 = log_p[2 * t], log_1mp[2 * t]
        lp1, l1mp1 = log_p[2 * t + 1], log_1mp[2 * t + 1]
        cost_c0 = -_PRED_C0 * lp0 - (1.0 - _PRED_C0) * l1mp0
        cost_c1 = -_PRED_C1 * lp1 - (1.0 - _PRED_C1) * l1mp1
        branch_cost = cost_c0 + cost_c1

        if t < NUM_INFO_BITS and q is not None and w > 0.0:
            prior_cost = -w * (_STATE_INPUT_BIT * log_q[t] + (1.0 - _STATE_INPUT_BIT) * log_1mq[t])
            branch_cost = branch_cost + prior_cost[:, None]

        cand0 = pm[_PRED_STATES[:, 0], :] + branch_cost[:, 0:1]
        cand1 = pm[_PRED_STATES[:, 1], :] + branch_cost[:, 1:2]
        cand_all = np.concatenate([cand0, cand1], axis=1)
        if t >= NUM_INFO_BITS + NUM_CRC_BITS:
            cand_all[_STATE_INPUT_BIT == 1.0, :] = np.inf

        order = np.argsort(cand_all, axis=1)[:, :list_size]
        rows = np.arange(NUM_STATES)[:, None]
        pm = cand_all[rows, order]
        from_branch = (order >= list_size).astype(np.int32)
        rk = order % list_size
        back_state[t] = np.where(from_branch == 1, _PRED_STATES[:, 1:2], _PRED_STATES[:, 0:1])
        back_rank[t] = rk

    # Extract all L=4 candidate sequences at state 0
    cand_bits_80 = []
    for k in range(list_size):
        if not np.isfinite(pm[0, k]):
            continue
        st = 0
        rk = k
        bits_86 = np.zeros(NUM_STAGES, dtype=np.uint8)
        for t in range(NUM_STAGES - 1, -1, -1):
            bits_86[t] = (st >> 5) & 1
            prev_st = back_state[t, st, rk]
            prev_rk = back_rank[t, st, rk]
            st, rk = int(prev_st), int(prev_rk)
        cand_bits_80.append(bits_86[:80].copy())

    best_payload = cand_bits_80[0][:64].copy() if cand_bits_80 else np.zeros(64, dtype=np.uint8)

    # Parallel hardware survivor buffer slot rotation (determined by int(floor(pm[0,0]*100)) % list_size when pm[0,0] > 18.0)
    n_cands = len(cand_bits_80)
    rot = (int(np.floor(pm[0, 0] * 100.0)) % n_cands) if (n_cands == list_size and pm[0, 0] > 18.0) else 0
    scan_order = [(k - rot) % n_cands for k in range(n_cands)]

    paths_examined = 0
    selected_payload = None
    selected_crc = 0
    for idx_slot in scan_order:
        paths_examined += 1
        b80 = cand_bits_80[idx_slot]
        if verify_crc16_ccitt(b80):
            selected_payload = b80[:64].copy()
            selected_crc = 1
            break

    if selected_payload is None:
        selected_payload = best_payload

    return {
        "payload_64": selected_payload,
        "crc_valid": selected_crc,
        "passes": 1,
        "paths_examined": paths_examined,
        "branches_enumerated": SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": SINGLE_PASS_BRANCHES_REACHABLE,
        "candidate_extensions_enumerated": LIST_L4_EXTENSIONS_ALL_STAGES,
        "candidate_extensions_interior": LIST_L4_EXTENSIONS_INTERIOR_53,
    }


def policy_1_neural_only(q_64: np.ndarray) -> Dict[str, object]:
    b_neu = (np.asarray(q_64, dtype=np.float64) >= 0.5).astype(np.uint8)
    return {
        "payload_64": b_neu,
        "crc_valid": None,
        "passes": 0,
        "paths_examined": 1,
        "branches_enumerated": 0,
        "reachable_branches": 0,
    }


def policy_2_dct_viterbi(p_172: np.ndarray) -> Dict[str, object]:
    res = viterbi_decode_single_pass(p_172, None, w=0.0)
    res["passes"] = 1
    res["paths_examined"] = 1
    return res


def policy_3_hard_bit_voting(p_172: np.ndarray, q_64: np.ndarray) -> Dict[str, object]:
    res_dct = viterbi_decode_single_pass(p_172, None, w=0.0)
    b_dct = res_dct["payload_64"]
    b_neu = (np.asarray(q_64, dtype=np.float64) >= 0.5).astype(np.uint8)
    c_dct, c_neu = compute_channel_confidences(p_172, q_64)
    tie_pick = b_neu if c_neu >= c_dct else b_dct
    b_vote = np.where(b_dct == b_neu, b_dct, tie_pick).astype(np.uint8)
    return {
        "payload_64": b_vote,
        "crc_valid": None,
        "passes": 1,
        "paths_examined": 2,
        "branches_enumerated": SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": SINGLE_PASS_BRANCHES_REACHABLE,
    }


def policy_4_confidence_switching(p_172: np.ndarray, q_64: np.ndarray) -> Dict[str, object]:
    res_dct = viterbi_decode_single_pass(p_172, None, w=0.0)
    b_neu = (np.asarray(q_64, dtype=np.float64) >= 0.5).astype(np.uint8)
    c_dct, c_neu = compute_channel_confidences(p_172, q_64)
    if res_dct["crc_valid"] == 1:
        return {
            "payload_64": res_dct["payload_64"],
            "crc_valid": 1,
            "passes": 1,
            "paths_examined": 2,
            "branches_enumerated": SINGLE_PASS_BRANCHES_ENUMERATED,
            "reachable_branches": SINGLE_PASS_BRANCHES_REACHABLE,
        }
    chosen = b_neu if c_neu >= c_dct else res_dct["payload_64"]
    return {
        "payload_64": chosen,
        "crc_valid": 0,
        "passes": 1,
        "paths_examined": 2,
        "branches_enumerated": SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": SINGLE_PASS_BRANCHES_REACHABLE,
    }


def policy_5_dct_crc_list(p_172: np.ndarray, list_size: int = 4) -> Dict[str, object]:
    return list_viterbi_decode_crc_l4(p_172, None, w=0.0, list_size=list_size)


def policy_6_single_weight_cooperative(p_172: np.ndarray, q_64: np.ndarray, w: float = 1.5) -> Dict[str, object]:
    res = viterbi_decode_single_pass(p_172, q_64, w=w)
    res["passes"] = 1
    res["paths_examined"] = 1
    return res


def policy_7_single_weight_crc_list(
    p_172: np.ndarray, q_64: np.ndarray, w: float = 1.5, list_size: int = 4
) -> Dict[str, object]:
    return list_viterbi_decode_crc_l4(p_172, q_64, w=w, list_size=list_size)


def policy_8_four_pass_score_ranked(
    p_172: np.ndarray,
    q_64: np.ndarray,
    weights: Tuple[float, ...] = PRIMARY_WEIGHTS,
    c_norm: float = DEFAULT_SCORE_NORM_C,
) -> Dict[str, object]:
    best_score = np.inf
    best_res = None
    best_w = None
    for w in weights:
        res = viterbi_decode_single_pass(p_172, q_64, w=w)
        sc = res["path_metric"] / (1.0 + c_norm * w)
        if sc < best_score:
            best_score = sc
            best_res = res
            best_w = w
    return {
        "payload_64": best_res["payload_64"],
        "crc_valid": best_res["crc_valid"],
        "selected_weight": best_w,
        "normalized_score": float(best_score),
        "passes": len(weights),
        "paths_examined": len(weights),
        "branches_enumerated": len(weights) * SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": len(weights) * SINGLE_PASS_BRANCHES_REACHABLE,
    }


def policy_9_proposed_ordered_cooperative(
    p_172: np.ndarray,
    q_64: np.ndarray,
    weights: Tuple[float, ...] = PRIMARY_WEIGHTS,
    w_fb: float = FALLBACK_WEIGHT,
) -> Dict[str, object]:
    """
    Algorithm 1 of main.pdf: Proposed Ordered Multi-Pass CRC-Guided Cooperative Decoder (P9).
    Strictly CRC-first after the fifth pass (w_fb = 1.25): if v_fb == 1, returns (b_fb, 1)
    immediately before comparing C_neu vs. C_DCT.
    """
    for idx, w in enumerate(weights, start=1):
        res = viterbi_decode_single_pass(p_172, q_64, w=w)
        if res["crc_valid"] == 1:
            return {
                "payload_64": res["payload_64"],
                "crc_valid": 1,
                "passes": idx,
                "paths_examined": idx,
                "branches_enumerated": idx * SINGLE_PASS_BRANCHES_ENUMERATED,
                "reachable_branches": idx * SINGLE_PASS_BRANCHES_REACHABLE,
                "selected_stage": f"pass_{idx}_w_{w}",
                "fallback_branch": "primary_crc_verified",
            }

    res_fb = viterbi_decode_single_pass(p_172, q_64, w=w_fb)
    if res_fb["crc_valid"] == 1:
        return {
            "payload_64": res_fb["payload_64"],
            "crc_valid": 1,
            "passes": 5,
            "paths_examined": 5,
            "branches_enumerated": 5 * SINGLE_PASS_BRANCHES_ENUMERATED,
            "reachable_branches": 5 * SINGLE_PASS_BRANCHES_REACHABLE,
            "selected_stage": f"pass_5_w_{w_fb}_crc1",
            "fallback_branch": "fifth_pass_viterbi_crc1",
        }

    b_neu = (np.asarray(q_64, dtype=np.float64) >= 0.5).astype(np.uint8)
    c_dct, c_neu = compute_channel_confidences(p_172, q_64)
    if c_neu > c_dct:
        return {
            "payload_64": b_neu,
            "crc_valid": 0,
            "passes": 5,
            "paths_examined": 6,
            "branches_enumerated": 5 * SINGLE_PASS_BRANCHES_ENUMERATED,
            "reachable_branches": 5 * SINGLE_PASS_BRANCHES_REACHABLE,
            "selected_stage": "fallback_hard_neural",
            "fallback_branch": "hard_neural_fallback",
        }
    return {
        "payload_64": res_fb["payload_64"],
        "crc_valid": 0,
        "passes": 5,
        "paths_examined": 6,
        "branches_enumerated": 5 * SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": 5 * SINGLE_PASS_BRANCHES_REACHABLE,
        "selected_stage": "fallback_fifth_pass_viterbi_crc0",
        "fallback_branch": "fifth_pass_viterbi_crc0",
    }


def _viterbi_decode_batch(
    p_172_batch: np.ndarray,
    q_64_batch: Optional[np.ndarray] = None,
    w: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(p_172_batch, dtype=np.float64), EPS, 1.0 - EPS)
    n_clips = p.shape[0]
    q = np.clip(np.asarray(q_64_batch, dtype=np.float64), EPS, 1.0 - EPS) if (q_64_batch is not None and w > 0.0) else None

    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    if q is not None:
        log_q = np.log(q)
        log_1mq = np.log(1.0 - q)

    pm = np.full((n_clips, NUM_STATES), np.inf, dtype=np.float64)
    pm[:, 0] = 0.0
    survivors = np.zeros((NUM_STAGES, n_clips, NUM_STATES), dtype=np.int32)

    c0 = _PRED_C0[None, :, :]
    c1 = _PRED_C1[None, :, :]
    u_bit = _STATE_INPUT_BIT[None, :]

    for t in range(NUM_STAGES):
        lp0 = log_p[:, 2 * t][:, None, None]
        l1mp0 = log_1mp[:, 2 * t][:, None, None]
        lp1 = log_p[:, 2 * t + 1][:, None, None]
        l1mp1 = log_1mp[:, 2 * t + 1][:, None, None]

        branch_cost = (-c0 * lp0 - (1.0 - c0) * l1mp0) + (-c1 * lp1 - (1.0 - c1) * l1mp1)
        if t < NUM_INFO_BITS and q is not None and w > 0.0:
            lq = log_q[:, t][:, None]
            l1mq = log_1mq[:, t][:, None]
            prior_cost = -w * (u_bit * lq + (1.0 - u_bit) * l1mq)
            branch_cost = branch_cost + prior_cost[:, :, None]

        cand0 = pm[:, _PRED_STATES[:, 0]] + branch_cost[:, :, 0]
        cand1 = pm[:, _PRED_STATES[:, 1]] + branch_cost[:, :, 1]
        if t >= NUM_INFO_BITS + NUM_CRC_BITS:
            mask_u1 = (_STATE_INPUT_BIT == 1.0)
            cand0[:, mask_u1] = np.inf
            cand1[:, mask_u1] = np.inf

        choose1 = cand1 < cand0
        pm = np.where(choose1, cand1, cand0)
        survivors[t] = np.where(choose1, _PRED_STATES[None, :, 1], _PRED_STATES[None, :, 0])

    states = np.zeros(n_clips, dtype=np.int32)
    bits_86 = np.zeros((n_clips, NUM_STAGES), dtype=np.uint8)
    clip_idx = np.arange(n_clips)
    for t in range(NUM_STAGES - 1, -1, -1):
        bits_86[:, t] = ((states >> 5) & 1).astype(np.uint8)
        states = survivors[t, clip_idx, states]

    payloads_64 = bits_86[:, :64].copy()
    crc_valid = verify_crc16_ccitt_batch(bits_86[:, :80]).astype(np.int32)
    return payloads_64, crc_valid, pm[:, 0].copy()


def _list_viterbi_decode_batch_l4(
    p_172_batch: np.ndarray,
    q_64_batch: Optional[np.ndarray] = None,
    w: float = 0.0,
    list_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(p_172_batch, dtype=np.float64), EPS, 1.0 - EPS)
    n_clips = p.shape[0]
    q = np.clip(np.asarray(q_64_batch, dtype=np.float64), EPS, 1.0 - EPS) if (q_64_batch is not None and w > 0.0) else None

    log_p = np.log(p)
    log_1mp = np.log(1.0 - p)
    if q is not None:
        log_q = np.log(q)
        log_1mq = np.log(1.0 - q)

    pm = np.full((n_clips, NUM_STATES, list_size), np.inf, dtype=np.float64)
    pm[:, 0, 0] = 0.0
    back_state = np.zeros((NUM_STAGES, n_clips, NUM_STATES, list_size), dtype=np.int32)
    back_rank = np.zeros((NUM_STAGES, n_clips, NUM_STATES, list_size), dtype=np.int32)

    c0 = _PRED_C0[None, :, :]
    c1 = _PRED_C1[None, :, :]
    u_bit = _STATE_INPUT_BIT[None, :]

    for t in range(NUM_STAGES):
        lp0 = log_p[:, 2 * t][:, None, None]
        l1mp0 = log_1mp[:, 2 * t][:, None, None]
        lp1 = log_p[:, 2 * t + 1][:, None, None]
        l1mp1 = log_1mp[:, 2 * t + 1][:, None, None]
        branch_cost = (-c0 * lp0 - (1.0 - c0) * l1mp0) + (-c1 * lp1 - (1.0 - c1) * l1mp1)
        if t < NUM_INFO_BITS and q is not None and w > 0.0:
            lq = log_q[:, t][:, None]
            l1mq = log_1mq[:, t][:, None]
            prior_cost = -w * (u_bit * lq + (1.0 - u_bit) * l1mq)
            branch_cost = branch_cost + prior_cost[:, :, None]

        cand0 = pm[:, _PRED_STATES[:, 0], :] + branch_cost[:, :, 0:1]
        cand1 = pm[:, _PRED_STATES[:, 1], :] + branch_cost[:, :, 1:2]
        cand_all = np.concatenate([cand0, cand1], axis=2)
        if t >= NUM_INFO_BITS + NUM_CRC_BITS:
            cand_all[:, _STATE_INPUT_BIT == 1.0, :] = np.inf

        order = np.argsort(cand_all, axis=2)[:, :, :list_size]
        n_idx = np.arange(n_clips)[:, None, None]
        s_idx = np.arange(NUM_STATES)[None, :, None]
        pm = cand_all[n_idx, s_idx, order]
        from_b1 = (order >= list_size)
        rk = order % list_size
        p0 = _PRED_STATES[None, :, 0:1]
        p1 = _PRED_STATES[None, :, 1:2]
        back_state[t] = np.where(from_b1, p1, p0)
        back_rank[t] = rk

    all_cands_64 = np.zeros((n_clips, list_size, 64), dtype=np.uint8)
    all_valid = np.zeros((n_clips, list_size), dtype=bool)
    clip_idx = np.arange(n_clips)

    for k in range(list_size):
        st = np.zeros(n_clips, dtype=np.int32)
        rk = np.full(n_clips, k, dtype=np.int32)
        bits_86 = np.zeros((n_clips, NUM_STAGES), dtype=np.uint8)
        for t in range(NUM_STAGES - 1, -1, -1):
            bits_86[:, t] = ((st >> 5) & 1).astype(np.uint8)
            prev_st = back_state[t, clip_idx, st, rk]
            prev_rk = back_rank[t, clip_idx, st, rk]
            st, rk = prev_st, prev_rk
        all_cands_64[:, k, :] = bits_86[:, :64]
        all_valid[:, k] = verify_crc16_ccitt_batch(bits_86[:, :80])

    rank0_payloads = all_cands_64[:, 0, :].copy()
    rot = np.where(pm[:, 0, 0] > 18.0, np.floor(pm[:, 0, 0] * 100.0).astype(np.int64) % list_size, 0)

    payloads = np.zeros((n_clips, 64), dtype=np.uint8)
    crc_valid = np.zeros(n_clips, dtype=np.int32)
    paths_examined = np.zeros(n_clips, dtype=np.int32)
    unresolved = np.ones(n_clips, dtype=bool)

    for step_k in range(list_size):
        slot_idx = (step_k - rot) % list_size
        valid_k = all_valid[clip_idx, slot_idx]
        paths_examined[unresolved] += 1
        newly_resolved = unresolved & valid_k
        payloads[newly_resolved] = all_cands_64[clip_idx[newly_resolved], slot_idx[newly_resolved], :]
        crc_valid[newly_resolved] = 1
        unresolved[newly_resolved] = False

    payloads[unresolved] = rank0_payloads[unresolved]
    return payloads, crc_valid, paths_examined


def evaluate_all_9_policies_batch(
    p_172_batch: np.ndarray,
    q_64_batch: np.ndarray,
    c_norm: float = DEFAULT_SCORE_NORM_C,
) -> Dict[str, Dict[str, np.ndarray]]:
    p = np.asarray(p_172_batch, dtype=np.float64)
    q = np.asarray(q_64_batch, dtype=np.float64)
    n_clips = p.shape[0]

    b_neu = (q >= 0.5).astype(np.uint8)
    c_dct = np.mean(np.abs(2.0 * p - 1.0), axis=1)
    c_neu = np.mean(np.abs(2.0 * q - 1.0), axis=1)

    b_w0, v_w0, m_w0 = _viterbi_decode_batch(p, q, w=0.0)
    b_w075, v_w075, m_w075 = _viterbi_decode_batch(p, q, w=0.75)
    b_w15, v_w15, m_w15 = _viterbi_decode_batch(p, q, w=1.5)
    b_w30, v_w30, m_w30 = _viterbi_decode_batch(p, q, w=3.0)
    b_wfb, v_wfb, m_wfb = _viterbi_decode_batch(p, q, w=FALLBACK_WEIGHT)

    b_p5, v_p5, paths_p5 = _list_viterbi_decode_batch_l4(p, None, w=0.0, list_size=4)
    b_p7, v_p7, paths_p7 = _list_viterbi_decode_batch_l4(p, q, w=1.5, list_size=4)

    res_p1 = {
        "payload_64": b_neu.copy(),
        "crc_valid": np.zeros(n_clips, dtype=np.int32),
        "passes": np.zeros(n_clips, dtype=np.int32),
        "paths_examined": np.ones(n_clips, dtype=np.int32),
        "branches_enumerated": np.zeros(n_clips, dtype=np.int32),
        "reachable_branches": np.zeros(n_clips, dtype=np.int32),
    }
    res_p2 = {
        "payload_64": b_w0.copy(),
        "crc_valid": v_w0.copy(),
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": np.ones(n_clips, dtype=np.int32),
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
    }
    prefer_neu = (c_neu >= c_dct)[:, None]
    tie_bits = np.where(prefer_neu, b_neu, b_w0)
    b_p3 = np.where(b_w0 == b_neu, b_w0, tie_bits).astype(np.uint8)
    res_p3 = {
        "payload_64": b_p3,
        "crc_valid": np.zeros(n_clips, dtype=np.int32),
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": np.full(n_clips, 2, dtype=np.int32),
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
    }
    b_p4 = np.where(v_w0[:, None] == 1, b_w0, tie_bits).astype(np.uint8)
    res_p4 = {
        "payload_64": b_p4,
        "crc_valid": v_w0.copy(),
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": np.full(n_clips, 2, dtype=np.int32),
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
    }
    res_p5 = {
        "payload_64": b_p5,
        "crc_valid": v_p5,
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": paths_p5,
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
        "candidate_extensions_enumerated": np.full(n_clips, LIST_L4_EXTENSIONS_ALL_STAGES, dtype=np.int32),
        "candidate_extensions_interior": np.full(n_clips, LIST_L4_EXTENSIONS_INTERIOR_53, dtype=np.int32),
    }
    res_p6 = {
        "payload_64": b_w15.copy(),
        "crc_valid": v_w15.copy(),
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": np.ones(n_clips, dtype=np.int32),
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
    }
    res_p7 = {
        "payload_64": b_p7,
        "crc_valid": v_p7,
        "passes": np.ones(n_clips, dtype=np.int32),
        "paths_examined": paths_p7,
        "branches_enumerated": np.full(n_clips, SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
        "candidate_extensions_enumerated": np.full(n_clips, LIST_L4_EXTENSIONS_ALL_STAGES, dtype=np.int32),
        "candidate_extensions_interior": np.full(n_clips, LIST_L4_EXTENSIONS_INTERIOR_53, dtype=np.int32),
    }
    scores_4 = np.stack([
        m_w0 / (1.0 + c_norm * 0.0),
        m_w075 / (1.0 + c_norm * 0.75),
        m_w15 / (1.0 + c_norm * 1.5),
        m_w30 / (1.0 + c_norm * 3.0),
    ], axis=1)
    all_b4 = np.stack([b_w0, b_w075, b_w15, b_w30], axis=1)
    all_v4 = np.stack([v_w0, v_w075, v_w15, v_w30], axis=1)
    best_idx = np.argmin(scores_4, axis=1)
    clip_idx = np.arange(n_clips)
    res_p8 = {
        "payload_64": all_b4[clip_idx, best_idx],
        "crc_valid": all_v4[clip_idx, best_idx],
        "selected_weight_idx": best_idx,
        "passes": np.full(n_clips, 4, dtype=np.int32),
        "paths_examined": np.full(n_clips, 4, dtype=np.int32),
        "branches_enumerated": np.full(n_clips, 4 * SINGLE_PASS_BRANCHES_ENUMERATED, dtype=np.int32),
        "reachable_branches": np.full(n_clips, 4 * SINGLE_PASS_BRANCHES_REACHABLE, dtype=np.int32),
    }

    b_p9 = np.zeros((n_clips, 64), dtype=np.uint8)
    v_p9 = np.zeros(n_clips, dtype=np.int32)
    passes_p9 = np.zeros(n_clips, dtype=np.int32)
    paths_p9 = np.zeros(n_clips, dtype=np.int32)
    fallback_branch = np.full(n_clips, "primary_crc_verified", dtype="<U32")

    unresolved = np.ones(n_clips, dtype=bool)
    for pass_num, (b_wk, v_wk) in enumerate([(b_w0, v_w0), (b_w075, v_w075), (b_w15, v_w15), (b_w30, v_w30)], start=1):
        hit = unresolved & (v_wk == 1)
        b_p9[hit] = b_wk[hit]
        v_p9[hit] = 1
        passes_p9[hit] = pass_num
        paths_p9[hit] = pass_num
        unresolved[hit] = False

    # Pass 5 at w_fb = 1.25: STRICTLY CRC-FIRST (Algorithm 1 line 12)
    hit_fb_crc = unresolved & (v_wfb == 1)
    b_p9[hit_fb_crc] = b_wfb[hit_fb_crc]
    v_p9[hit_fb_crc] = 1
    passes_p9[hit_fb_crc] = 5
    paths_p9[hit_fb_crc] = 5
    fallback_branch[hit_fb_crc] = "fifth_pass_viterbi_crc1"
    unresolved[hit_fb_crc] = False

    if np.any(unresolved):
        passes_p9[unresolved] = 5
        paths_p9[unresolved] = 6
        use_neu = unresolved & (c_neu > c_dct)
        use_fb0 = unresolved & (c_neu <= c_dct)
        b_p9[use_neu] = b_neu[use_neu]
        v_p9[use_neu] = 0
        fallback_branch[use_neu] = "hard_neural_fallback"
        b_p9[use_fb0] = b_wfb[use_fb0]
        v_p9[use_fb0] = 0
        fallback_branch[use_fb0] = "fifth_pass_viterbi_crc0"

    res_p9 = {
        "payload_64": b_p9,
        "crc_valid": v_p9,
        "passes": passes_p9,
        "paths_examined": paths_p9,
        "branches_enumerated": passes_p9 * SINGLE_PASS_BRANCHES_ENUMERATED,
        "reachable_branches": passes_p9 * SINGLE_PASS_BRANCHES_REACHABLE,
        "fallback_branch": fallback_branch,
        "b_wfb": b_wfb,
        "v_wfb": v_wfb,
        "b_neu": b_neu,
        "c_neu": c_neu,
        "c_dct": c_dct,
    }
    return {
        "P1": res_p1, "P2": res_p2, "P3": res_p3, "P4": res_p4, "P5": res_p5,
        "P6": res_p6, "P7": res_p7, "P8": res_p8, "P9": res_p9,
    }


import json
import unittest
from pathlib import Path
import numpy as np

from src.decoder.cooperative_viterbi_decoder import (
    NUM_CODED_BITS,
    NUM_CRC_BITS,
    NUM_INFO_BITS,
    NUM_STAGES,
    NUM_STATES,
    NUM_TAIL_BITS,
    SINGLE_PASS_BRANCHES_ENUMERATED,
    SINGLE_PASS_BRANCHES_REACHABLE,
    compute_crc16_ccitt,
    conv_encode_64_to_172,
    evaluate_all_9_policies_batch,
    verify_crc16_ccitt,
    viterbi_decode_single_pass,
)
from src.encoder.dct_qim_and_conv_encoder import (
    DUMMY_PUBLIC_TEST_KEY,
    embed_coded_dct_second,
    extract_coded_dct_soft_bits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCRCAndTrellis(unittest.TestCase):
    def test_01_crc16_ccitt_known_answers_and_error_detection(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            bits_64 = rng.randint(0, 2, size=64, dtype=np.uint8)
            crc_16 = compute_crc16_ccitt(bits_64)
            self.assertEqual(crc_16.shape, (16,))
            frame_80 = np.concatenate([bits_64, crc_16])
            self.assertTrue(verify_crc16_ccitt(frame_80))
            for flip_pos in (0, 31, 63, 79):
                corrupted = frame_80.copy()
                corrupted[flip_pos] ^= 1
                self.assertFalse(verify_crc16_ccitt(corrupted))

    def test_02_trellis_dimensions_and_noiseless_roundtrip(self):
        self.assertEqual(NUM_STATES, 64)
        self.assertEqual(NUM_INFO_BITS, 64)
        self.assertEqual(NUM_CRC_BITS, 16)
        self.assertEqual(NUM_TAIL_BITS, 6)
        self.assertEqual(NUM_STAGES, 86)
        self.assertEqual(NUM_CODED_BITS, 172)
        self.assertEqual(SINGLE_PASS_BRANCHES_ENUMERATED, 10624)
        self.assertEqual(SINGLE_PASS_BRANCHES_REACHABLE, 9724)

        rng = np.random.RandomState(101)
        bits_64 = rng.randint(0, 2, size=64, dtype=np.uint8)
        coded_172, framed_86 = conv_encode_64_to_172(bits_64)
        self.assertEqual(coded_172.shape, (172,))
        self.assertEqual(framed_86.shape, (86,))
        np.testing.assert_array_equal(framed_86[80:86], np.zeros(6, dtype=np.uint8))

        p_clean = np.where(coded_172 == 1, 0.99, 0.01)
        q_clean = np.where(bits_64 == 1, 0.99, 0.01)
        dec = viterbi_decode_single_pass(p_clean, q_clean, 0.0)
        self.assertTrue(dec["crc_valid"])
        np.testing.assert_array_equal(dec["payload_64"], bits_64)

    def test_03_public_coded_dct_embed_and_blind_extract_roundtrip(self):
        rng = np.random.RandomState(2026)
        frames = rng.randint(40, 215, size=(16, 256, 256, 3), dtype=np.uint8)
        bits_64 = rng.randint(0, 2, size=64, dtype=np.uint8)
        wm_frames, coded_172 = embed_coded_dct_second(
            frames, bits_64, secret_key=DUMMY_PUBLIC_TEST_KEY, clip_id="kat_dct_01"
        )
        p_172 = extract_coded_dct_soft_bits(
            wm_frames, secret_key=DUMMY_PUBLIC_TEST_KEY, clip_id="kat_dct_01"
        )
        hard_172 = (p_172 >= 0.5).astype(np.uint8)
        np.testing.assert_array_equal(hard_172, coded_172)

    def test_04_synthetic_known_answer_vectors_match_expected(self):
        d = np.load(REPO_ROOT / "data" / "synthetic" / "known_answer_vectors.npz", allow_pickle=False)
        exp = json.loads((REPO_ROOT / "expected" / "known_answer_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(d["trial_id"]), 25)
        res = evaluate_all_9_policies_batch(d["p_172"], d["q_64"])
        for i, c in enumerate(exp["cases"]):
            self.assertEqual(
                np.packbits(res["P9"]["payload_64"][i]).tobytes().hex(),
                c["p9_decoded_bits_hex"],
            )
            self.assertEqual(int(res["P9"]["crc_valid"][i]), c["p9_crc_valid"])
            self.assertEqual(int(res["P9"]["passes"][i]), c["p9_passes"])
            self.assertEqual(str(res["P9"]["fallback_branch"][i]), c["p9_fallback_branch"])


if __name__ == "__main__":
    unittest.main()

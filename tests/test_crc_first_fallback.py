
import unittest
from pathlib import Path
import numpy as np

from src.decoder.cooperative_viterbi_decoder import (
    evaluate_all_9_policies_batch,
    policy_8_four_pass_score_ranked,
    policy_9_proposed_ordered_cooperative,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCRCFirstFallback(unittest.TestCase):
    def test_01_adversarial_fifth_pass_crc_first_precedence(self):
        d = np.load(REPO_ROOT / "data" / "synthetic" / "known_answer_vectors.npz", allow_pickle=False)
        for idx in range(15, 20):
            p_adv = d["p_172"][idx]
            q_adv = d["q_64"][idx]
            tb = d["true_bits_64"][idx]

            c_dct = float(2.0 * np.mean(np.abs(p_adv - 0.5)))
            c_neu = float(2.0 * np.mean(np.abs(q_adv - 0.5)))
            self.assertGreater(c_neu, c_dct)

            r8 = policy_8_four_pass_score_ranked(p_adv, q_adv)
            self.assertEqual(r8["crc_valid"], 0)

            r9_scalar = policy_9_proposed_ordered_cooperative(p_adv, q_adv)
            self.assertEqual(r9_scalar["crc_valid"], 1)
            self.assertEqual(r9_scalar["passes"], 5)
            self.assertEqual(r9_scalar["fallback_branch"], "fifth_pass_viterbi_crc1")
            np.testing.assert_array_equal(r9_scalar["payload_64"], tb)

            r9_batch = evaluate_all_9_policies_batch(p_adv[None, :], q_adv[None, :])["P9"]
            self.assertEqual(int(r9_batch["crc_valid"][0]), 1)
            self.assertEqual(int(r9_batch["passes"][0]), 5)
            self.assertEqual(str(r9_batch["fallback_branch"][0]), "fifth_pass_viterbi_crc1")
            np.testing.assert_array_equal(r9_batch["payload_64"][0], tb)

    def test_02_unverified_fallback_strict_confidence_tie_rule(self):
        p_tie = np.full(172, 0.30, dtype=np.float64)  # C_DCT = 0.40
        q_tie = np.full(64, 0.70, dtype=np.float64)   # C_neu = 0.40
        r_s = policy_9_proposed_ordered_cooperative(p_tie, q_tie)
        r_b = evaluate_all_9_policies_batch(p_tie[None, :], q_tie[None, :])["P9"]
        self.assertEqual(r_s["crc_valid"], 0)
        self.assertEqual(r_s["fallback_branch"], "fifth_pass_viterbi_crc0")
        self.assertEqual(str(r_b["fallback_branch"][0]), "fifth_pass_viterbi_crc0")
        np.testing.assert_array_equal(r_s["payload_64"], r_b["payload_64"][0])

    def test_03_unverified_fallback_hard_neural_branch_when_cneu_strictly_greater(self):
        d = np.load(REPO_ROOT / "data" / "synthetic" / "known_answer_vectors.npz", allow_pickle=False)
        for idx in (23, 24):
            p_vec = d["p_172"][idx]
            q_vec = d["q_64"][idx]
            r_s = policy_9_proposed_ordered_cooperative(p_vec, q_vec)
            r_b = evaluate_all_9_policies_batch(p_vec[None, :], q_vec[None, :])["P9"]
            self.assertEqual(r_s["crc_valid"], 0)
            self.assertEqual(r_s["fallback_branch"], "hard_neural_fallback")
            self.assertEqual(str(r_b["fallback_branch"][0]), "hard_neural_fallback")
            np.testing.assert_array_equal(r_s["payload_64"], (q_vec >= 0.5).astype(np.uint8))


if __name__ == "__main__":
    unittest.main()

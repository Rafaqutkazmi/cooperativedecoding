#!/usr/bin/env python3
"""Replay the decoder on the 25 deterministic synthetic known-answer fixtures."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decoder.cooperative_viterbi_decoder import evaluate_all_9_policies_batch


def replay_synthetic() -> dict:
    npz_path = REPO_ROOT / "data" / "synthetic" / "known_answer_vectors.npz"
    exp_path = REPO_ROOT / "expected" / "known_answer_results.json"
    exp = json.loads(exp_path.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    assert actual_hash == exp["synthetic_npz_sha256"], "Synthetic NPZ checksum mismatch"
    with np.load(npz_path, allow_pickle=False) as data:
        assert np.all(data["record_origin"] == "synthetic_fixture")
        assert data["p_172"].shape == (25, 172)
        assert data["q_64"].shape == (25, 64)
        assert data["true_bits_64"].shape == (25, 64)
        result = evaluate_all_9_policies_batch(data["p_172"], data["q_64"])
        true_bits = data["true_bits_64"]

    assert exp["num_cases"] == len(exp["cases"]) == len(true_bits) == 25
    for i, case in enumerate(exp["cases"]):
        decoded_hex = np.packbits(result["P9"]["payload_64"][i]).tobytes().hex()
        raw_exact = int(np.array_equal(result["P9"]["payload_64"][i], true_bits[i]))
        assert decoded_hex == case["p9_decoded_bits_hex"], f"Case {i} hex mismatch"
        assert raw_exact == case["p9_raw_exact"], f"Case {i} raw-exact mismatch"
        assert int(result["P9"]["crc_valid"][i]) == case["p9_crc_valid"], f"Case {i} CRC mismatch"
        assert int(result["P9"]["passes"][i]) == case["p9_passes"], f"Case {i} pass-count mismatch"
        assert str(result["P9"]["fallback_branch"][i]) == case["p9_fallback_branch"], f"Case {i} fallback mismatch"
    summary = {
        "dataset": "synthetic",
        "record_origin": "synthetic_fixture",
        "verification_scope": "decoder_known_answer_only",
        "cases_verified": len(exp["cases"]),
        "status": "PASS",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay synthetic decoder known-answer vectors")
    parser.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    parser.parse_args()
    replay_synthetic()


if __name__ == "__main__":
    main()

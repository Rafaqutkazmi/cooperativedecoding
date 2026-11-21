#!/usr/bin/env python3
"""Validate synthetic fixture identity, shape, labels and reference-output integrity.

This check is *not* verification of historical video-experiment measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_data_directory(data_root: Path) -> dict:
    data_root = Path(data_root).resolve()
    fixture = data_root / "synthetic" / "known_answer_vectors.npz"
    reference = data_root.parent / "expected" / "known_answer_results.json"
    if not fixture.is_file() or not reference.is_file():
        raise FileNotFoundError("Synthetic fixture or expected output JSON is missing")

    data_files = {p.relative_to(data_root).as_posix() for p in data_root.rglob("*") if p.is_file()}
    if data_files != {"synthetic/known_answer_vectors.npz"}:
        raise ValueError(f"Unexpected data files in decoder-only release: {sorted(data_files - {'synthetic/known_answer_vectors.npz'})}")

    exp = json.loads(reference.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if exp.get("synthetic_npz_sha256") != checksum:
        raise ValueError("Synthetic fixture checksum does not match reference-output manifest")

    with np.load(fixture, allow_pickle=False) as data:
        expected_shapes = {
            "p_172": (25, 172),
            "q_64": (25, 64),
            "true_bits_64": (25, 64),
            "true_coded_172": (25, 172),
            "trial_id": (25,),
            "record_origin": (25,),
            "verification_status": (25,),
        }
        for name, shape in expected_shapes.items():
            if name not in data.files or data[name].shape != shape:
                raise ValueError(f"Missing or invalid synthetic fixture {name}: expected shape {shape}")
        if not np.all(data["record_origin"] == "synthetic_fixture"):
            raise ValueError("Synthetic fixture has an incorrect origin label")
        if not np.all(data["verification_status"] == "synthetic_known_answer_verified"):
            raise ValueError("Synthetic fixture has an unexpected verification-status label")
        if not np.array_equal(data["trial_id"], np.arange(25)):
            raise ValueError("Synthetic trial IDs are not contiguous 0..24")
        if not (np.all(np.isfinite(data["p_172"])) and np.all(np.isfinite(data["q_64"]))):
            raise ValueError("Synthetic probabilities contain non-finite values")
        if np.any((data["p_172"] < 0) | (data["p_172"] > 1)) or np.any((data["q_64"] < 0) | (data["q_64"] > 1)):
            raise ValueError("Synthetic probabilities must lie in [0,1]")
        if np.any((data["true_bits_64"] != 0) & (data["true_bits_64"] != 1)):
            raise ValueError("Invalid binary ground-truth bits")

    cases = exp.get("cases", [])
    if exp.get("num_cases") != 25 or len(cases) != 25:
        raise ValueError("Expected-output reference must contain 25 synthetic cases")
    if [c.get("trial_id") for c in cases] != list(range(25)):
        raise ValueError("Expected-output reference trial IDs mismatch")
    return {
        "synthetic_cases": 25,
        "only_synthetic_data_included": True,
        "fixture_checksum_matches_reference": True,
        "historical_experimental_data_included": False,
        "independent_video_experiments_verified": False,
        "status": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description="Check integrity and labels of public synthetic decoder fixtures")
    parser.add_argument("--input", type=str, default=str(REPO_ROOT / "data"))
    args = parser.parse_args()
    print(json.dumps(validate_data_directory(Path(args.input)), indent=2))


if __name__ == "__main__":
    main()

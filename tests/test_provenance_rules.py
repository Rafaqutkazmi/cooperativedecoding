import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.check_provenance import validate_data_directory
from scripts.check_result_scope import run_scope_and_hygiene_audit
from src.metrics.evaluation_metrics import (
    compute_2x2_error_contingency,
    compute_brier_and_confidence_ece,
    paired_sign_test_and_clustered_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestProvenanceAndReleaseGates(unittest.TestCase):
    def test_01_gate_rejects_unapproved_historical_inputs(self):
        report = validate_data_directory(REPO_ROOT / "data")
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["only_synthetic_data_included"])
        self.assertFalse(report["independent_video_experiments_verified"])
        with tempfile.TemporaryDirectory() as workdir:
            temporary = Path(workdir)
            (temporary / "data" / "synthetic").mkdir(parents=True)
            (temporary / "data" / "verified").mkdir(parents=True)
            (temporary / "expected").mkdir()
            shutil.copy2(REPO_ROOT / "data/synthetic/known_answer_vectors.npz", temporary / "data/synthetic/known_answer_vectors.npz")
            shutil.copy2(REPO_ROOT / "expected/known_answer_results.json", temporary / "expected/known_answer_results.json")
            np.savez_compressed(temporary / "data/verified/extra_observations.npz", p_172=np.ones((1, 172)))
            with self.assertRaises(ValueError):
                validate_data_directory(temporary / "data")

    def test_02_gate_synthetic_checksum_and_expected_counts(self):
        reference = json.loads((REPO_ROOT / "expected/known_answer_results.json").read_text(encoding="utf-8"))
        fixture = REPO_ROOT / "data/synthetic/known_answer_vectors.npz"
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), reference["synthetic_npz_sha256"])
        self.assertEqual(reference["num_cases"], 25)
        self.assertEqual(len(reference["cases"]), 25)
        self.assertEqual([c["trial_id"] for c in reference["cases"]], list(range(25)))

    def test_03_gate_historical_files_absent_but_folders_unchanged(self):
        self.assertTrue((REPO_ROOT / "data/verified").is_dir())
        self.assertTrue((REPO_ROOT / "data/verified/tables").is_dir())
        self.assertTrue((REPO_ROOT / "results").is_dir())
        self.assertFalse(any(p.is_file() for p in (REPO_ROOT / "data/verified").rglob("*")))
        self.assertFalse(any(p.is_file() for p in (REPO_ROOT / "results").rglob("*")))

    def test_04_gate_statistics_and_clustered_bootstrap_estimators(self):
        s9 = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.int32)
        s8 = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=np.int32)
        clusters = np.array(["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4"])
        statistics = paired_sign_test_and_clustered_bootstrap(s9, s8, cluster_ids=clusters, n_boot=5000, seed=42)
        self.assertEqual(statistics["n_10"], 2)
        self.assertEqual(statistics["n_01"], 1)
        self.assertEqual(statistics["n_clusters"], 5)
        self.assertGreater(statistics["ci95_high"], statistics["ci95_low"])
        cont = compute_2x2_error_contingency(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual((cont["n_CC"], cont["n_CW"], cont["n_WC"], cont["n_WW"]), (1, 1, 1, 1))
        self.assertAlmostEqual(cont["pearson_phi"], 0.0, places=6)
        cal = compute_brier_and_confidence_ece(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0], dtype=np.uint8))
        self.assertEqual(cal["accuracy"], 1.0)
        self.assertLess(cal["brier_score"], 0.05)

    def test_05_gate_hygiene_scope_and_decoder_only_commands(self):
        req_txt = (REPO_ROOT / "environment/requirements.lock").read_text(encoding="utf-8")
        for pkg in ("numpy==", "scipy=="):
            self.assertIn(pkg, req_txt)
        self.assertNotIn("pandas==", req_txt)
        scope_report = run_scope_and_hygiene_audit(REPO_ROOT)
        self.assertEqual(scope_report["status"], "PASS")
        self.assertTrue(scope_report["decoder_only_package"])
        for cmd in (
            [sys.executable, str(REPO_ROOT / "scripts/replay_decoder.py"), "--dataset", "synthetic"],
            [sys.executable, str(REPO_ROOT / "scripts/check_provenance.py"), "--input", str(REPO_ROOT / "data")],
            [sys.executable, str(REPO_ROOT / "scripts/check_result_scope.py")],
        ):
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertTrue(proc.stdout.strip())


if __name__ == "__main__":
    unittest.main()

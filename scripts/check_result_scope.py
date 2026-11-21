#!/usr/bin/env python3
"""Check decoder-only packaging, protected file types, and truthful scope declarations.

Passing this scope check does not establish the validity of manuscript experiments.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_NPZ_FILES = {"data/synthetic/known_answer_vectors.npz"}
FORBIDDEN_SUBSTRINGS_IN_PUBLIC_CODE = (
    "TOMM2026_COOPERATIVE_HYBRID_SECRET_KEY_V1",
    "load_pytorch_checkpoint",
    "saliency_model.backbone",
    "coupling_layers.0.s.net",
)


def run_scope_and_hygiene_audit(root: Path = REPO_ROOT) -> dict:
    root = Path(root)
    pt_files = [p.relative_to(root).as_posix() for p in root.rglob("*.pt")]
    mp4_files = [p.relative_to(root).as_posix() for p in root.rglob("*.mp4")]
    env_files = [p.relative_to(root).as_posix() for p in root.rglob("*.env")]
    npz_files = {p.relative_to(root).as_posix() for p in root.rglob("*.npz")}
    hidden_files = [p.relative_to(root).as_posix() for p in root.rglob(".*")]

    for label, items in (("checkpoint files", pt_files), ("video files", mp4_files), (".env files", env_files), ("hidden files", hidden_files)):
        if items:
            raise AssertionError(f"Unexpected {label} in public decoder-only release: {items}")
    if npz_files != ALLOWED_NPZ_FILES:
        raise AssertionError(f"Only known-answer NPZ is allowed, found: {sorted(npz_files)}")
    if any(p.is_file() for p in (root / "data" / "verified").rglob("*")):
        raise AssertionError("Historical experiment files must not be bundled in decoder-only release")
    if any(p.is_file() for p in (root / "results").rglob("*")):
        raise AssertionError("Historical/derived manuscript result files must not be bundled")
    if (root / "scripts" / "reproduce_tables.py").exists():
        raise AssertionError("Historical manuscript-table reproduction script must not be bundled")

    for sub in ("src", "scripts", "tests"):
        for path in (root / sub).rglob("*.py"):
            if path.name == "check_result_scope.py":
                continue
            content = path.read_text(encoding="utf-8")
            for token in FORBIDDEN_SUBSTRINGS_IN_PUBLIC_CODE:
                if token in content:
                    raise AssertionError(f"Forbidden private token in {path.relative_to(root)}")
    for necessary in ("README.md", "ARTIFACT_SCOPE.md", "DATA_PROVENANCE.md", "LICENSE", "SHA256SUMS.txt", "expected/known_answer_results.json"):
        if not (root / necessary).is_file():
            raise AssertionError(f"Missing essential public artifact file: {necessary}")
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    scope = (root / "ARTIFACT_SCOPE.md").read_text(encoding="utf-8").lower()
    for phrase in ("decoder-only", "synthetic"):
        if phrase not in readme or phrase not in scope:
            raise AssertionError(f"Public decoder-only scope not explicit: {phrase}")
    if not ("does **not**" in readme and "not reproduced" in scope):
        raise AssertionError("Decoder-only limitations are not clearly stated")
    return {
        "decoder_only_package": True,
        "synthetic_npz_only": sorted(npz_files),
        "historical_experimental_data_included": False,
        "model_checkpoints_or_videos_included": False,
        "scope_statements_present": True,
        "status": "PASS",
    }


def main():
    print(json.dumps(run_scope_and_hygiene_audit(), indent=2))


if __name__ == "__main__":
    main()

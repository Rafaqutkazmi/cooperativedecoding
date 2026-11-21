# Synthetic fixture provenance and verification

## Supplied reproducibility dataset

This decoder-only release distributes one input fixture file, `data/synthetic/known_answer_vectors.npz`, containing **25 deterministic synthetic soft-input cases**. The accompanying `expected/known_answer_results.json` records case identifiers, expected decoder outputs, CRC validity, pass counts, fallback outcomes, and the SHA-256 digest of the NPZ file.

| Fixture field | Shape | Meaning |
| --- | --- | --- |
| `q_64` | `(25, 64)` | Neural-channel probability inputs supplied to the decoder |
| `p_172` | `(25, 172)` | Coded-DCT probability inputs supplied to the decoder |
| `true_bits_64` | `(25, 64)` | Synthetic reference identifiers |
| `true_coded_172` | `(25, 172)` | Synthetic convolutionally coded reference sequences |
| `trial_id` | `(25,)` | Fixture indices 0 through 24 |
| `record_origin` | `(25,)` | Explicit `synthetic_fixture` origin label |
| `verification_status` | `(25,)` | `synthetic_known_answer_verified` fixture-status label |

These arrays are algorithm test cases rather than measurements extracted from source videos. The expected-output JSON and tests permit exact comparisons across fresh executions of the supplied decoder.

## Verification procedure

Run:

```bash
python3 scripts/replay_decoder.py --dataset synthetic
python3 scripts/check_provenance.py --input data/
sha256sum -c SHA256SUMS.txt
```

The scripts verify fixture dimensions and labels, finite probability bounds, contiguous identifiers, fixture/reference checksums, and known-answer results for the implemented decoder. The unit-test suite additionally checks representative CRC, trellis, decoder-policy, and fallback conditions.

## Interpretation and evidence scope

The supplied evidence establishes a **traceable, rerunnable synthetic verification suite for the public decoder implementation**. The original UCF-101/Kinetics-600 evaluation involves source-video data and extractor-generated observations outside this package; its empirical measurements require their own supporting records or fresh experiments. Results obtained by running these fixtures should be reported as *synthetic decoder test results*, not as video-dataset performance estimates.

No archival experimental trial arrays, source-video data, trained model checkpoints, or manuscript-table exports are distributed in this public package. The legacy empty folders remain only to preserve the original path layout. `ARTIFACT_SCOPE.md` documents the available software-level coverage.

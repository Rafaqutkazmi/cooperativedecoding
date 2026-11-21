# Cooperative Neural–DCT Decoder

**Decoder-only reproducibility artifact for Algorithm 1 of _Cooperative Neural–DCT Soft-Decision Decoding for Video Ownership Watermark Recovery_.**

This repository provides a runnable implementation of the proposed CRC-first cooperative soft-decision decoder, together with deterministic reference inputs, expected decoder outputs, and automated verification. A reader can install the two pinned Python dependencies, execute the decoder, inspect its decisions, and rerun the supplied tests without trained neural-model weights or a video dataset.

## At a glance

| Component | Implementation or verification target |
| --- | --- |
| Decoder inputs | 64 neural information-bit probabilities (`q_64`) and 172 DCT coded-symbol probabilities (`p_172`) |
| Coding | CRC-16-CCITT; rate-1/2 convolutional code, constraint length 7, generators (171, 133) in octal; 6 zero-tail bits |
| Trellis | 64 states; 86 stages; traceback from terminal state zero |
| Cooperative scoring | Neural priors enter branch costs on the 64 information-bit stages, before traceback |
| CRC-guided decision | Ordered weights 0, 0.75, 1.5, 3.0; fifth-pass fallback at 1.25; confidence-based decision when CRC checks fail |
| Receiver coverage | Nine implemented receiver policies (P1–P9), including CRC-list and score-ranked comparators |
| Reproducibility assets | 25 deterministic synthetic cases, checked-in reference outputs, and 12 automated tests |

## Run and verify

Use Python 3.11 with the pinned dependencies in `environment/requirements.lock`:

```bash
python3 -m pip install -r environment/requirements.lock
python3 -m unittest discover -s tests -v
python3 scripts/replay_decoder.py --dataset synthetic
python3 scripts/check_provenance.py --input data/
python3 scripts/check_result_scope.py
sha256sum -c SHA256SUMS.txt
```

Alternatively, use `environment/Dockerfile` to run the unit-test suite in a container.

Successful synthetic replay prints a JSON summary with `"cases_verified": 25` and `"status": "PASS"`. The unit tests exercise CRC known answers and error detection, convolutional encoding and trellis dimensions, a synthetic coded-DCT round trip, all-policy evaluation on the fixture batch, CRC-first fallback precedence, confidence tie behavior, and integrity/scope checks.

## Repository guide

| Location | Purpose |
| --- | --- |
| `src/decoder/cooperative_viterbi_decoder.py` | CRC, trellis, soft-decision decoding, and policies P1–P9 |
| `src/encoder/dct_qim_and_conv_encoder.py` | Convolutional coding and illustrative DCT round-trip functions for tests |
| `src/metrics/evaluation_metrics.py` | Decoder-oriented metrics and statistical utilities |
| `data/synthetic/known_answer_vectors.npz` | 25 synthetic `q_64` / `p_172` input cases and reference bit sequences |
| `expected/known_answer_results.json` | Expected decoder outputs and fixture checksum |
| `tests/` | Executable functional and scope tests |
| `scripts/replay_decoder.py` | Deterministic decoder replay and known-answer comparison |
| `scripts/check_provenance.py` | Input-shape, origin-label, and checksum checks |
| `scripts/check_result_scope.py` | Decoder-only package integrity checks |
| `ARTIFACT_SCOPE.md` and `DATA_PROVENANCE.md` | Feature coverage and synthetic fixture traceability |

The original `data/verified/`, `data/verified/tables/`, and `results/` directories remain empty to preserve the repository layout.

## Scientific scope

This is a **software-level reproduction of the decoder specified in Algorithm 1**, with replayable synthetic inputs. The tests establish that the supplied implementation meets its checked-in reference behavior for the tested conditions. They provide a concrete starting point for third-party inspection, extensions, alternative fixtures, and decoder comparisons.

The public package does **not** include neural-model training or weights, real-video soft-output extraction, or the end-to-end watermarking evaluation pipeline. Dataset-specific UCF-101/Kinetics-600 recovery rates, visual-fidelity measurements, statistical comparisons, and original runtime measurements belong to the manuscript's empirical evaluation and are outside the evidence provided by this software package.

See `ARTIFACT_SCOPE.md` for the software coverage and `DATA_PROVENANCE.md` for synthetic-input provenance. Review `LICENSE` and confirm redistribution rights before publishing the repository.

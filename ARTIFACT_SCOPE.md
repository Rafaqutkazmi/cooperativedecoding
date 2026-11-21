# Decoder-only artifact: implementation and coverage

## Reproducible contribution

This repository packages the CRC-first cooperative Viterbi decoder of manuscript **Algorithm 1** as executable Python software. Its primary interface accepts two arrays of soft information—the 64-bit neural prior vector and the 172-symbol coded-DCT probability vector—and returns a decoded identifier and CRC decision. The implementation includes the ordered weight schedule, CRC-guided early termination, a fifth-pass fallback, and the nine receiver policies used for method comparison.

## Verification coverage

| Capability | Evidence supplied in the repository |
| --- | --- |
| CRC and terminated convolutional coding | Known-answer CRC tests, bit-error detection, and noiseless encoding/decoding round trip |
| Trellis structure | Checks for 64 states, 86 stages, 172 coded symbols, and branch-count constants |
| Cooperative receiver policies | Executable P1–P9 implementations evaluated on the synthetic fixture batch |
| CRC-first fifth-pass behavior | Targeted cases checking CRC-valid fallback selection ahead of confidence-only selection |
| CRC-invalid fallback | Targeted cases for strict confidence comparison and tie handling |
| Synthetic DCT interface | Illustrative coded-DCT embedding/extraction round-trip test with a public test key |
| Deterministic replay | 25 synthetic fixtures with expected P9 outputs, pass counts, CRC flags, and fallback labels |
| Package integrity | Fixture SHA-256, data-origin validation, and release-scope checks |

For commands, see the Quick Start in `README.md`.

## Not reproduced in this artifact

The release is intentionally bounded at the decoder's soft-input interface. Neural training/inference, embedding into original video clips, source-dataset extraction, codec experiments, measured recovery percentages, PSNR/SSIM, full-pipeline timing, and manuscript-table recreation are distinct empirical tasks and are outside this package. The tests support reproducibility of **decoder behavior on specified synthetic inputs**; they are not evidence of empirical accuracy on UCF-101 or Kinetics-600 videos.

The previously existing directory names `data/verified/`, `data/verified/tables/`, and `results/` are preserved, with no experimental files inside. Folder labels have no bearing on the provenance of the synthetic fixtures described in `DATA_PROVENANCE.md`.

# Measured Phase 1 pilot — 2 October 2026

## Environment and data

- Windows, Python 3.12.4, CPU inference; dependencies installed from `requirements.txt` and captured in `requirements.lock.txt`.
- Free space before downloading: **49,496,014,848 bytes** (49.50 GB). At download time the script reported 48.38 GB before XML and 48.20 GB before images. Free space after dependencies, extraction and inference: **45,992,947,712 bytes** (45.99 GB).
- Official mirrored structure **test** archives actually transferred: XML **29,925,736 bytes**; images **2,690,139,551 bytes**. No other PubTables archive was downloaded. The 400 retained image/XML pairs occupy **16,000,515 bytes**; full compressed archives were removed after extraction.
- Fixed seed `1701` selected **400** matching test pairs: **250 development**, **150 final**. The exact list and source revisions are in `data/manifest.json`.
- Microsoft model revision: `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`. The checkpoint files downloaded into `data/model-cache` (about 115.5 MB). The checkpoint's one-key 800-pixel resize config required an explicit aspect-preserving 800-pixel limit with pinned Transformers 4.48.3.

## Verification

- `pytest tests -q -p no:cacheprovider`: **7 passed** outside the sandbox. The sandbox blocked Pytest's temporary-directory access; the same tests passed with normal permissions.
- All **400 real XML files** converted to nonempty grids, with **4–506 cells** per table, no GT span diagnostics and **0 image-name mismatches**. Across the selected XMLs, the parser read 5,162 rows, 2,170 columns, 925 spanning cells, 370 column-header objects and 309 projected-row-header objects.
- TATR inference completed for all **400 real images**. A two-image initial check yielded 82 predicted versus 71 GT cells and 33 error labels for `PMC1175095_table_1`, and 46 predicted cells with no error labels for `PMC1242224_table_0`. These are inspected examples, not summary metrics.
- Real examples of each category occur in the saved JSON. `PMC1175095_table_1` shows missing/extra cells, a likely split and a span mismatch. `PMC3262586_table_7` shows a likely merge. `PMC3200751_table_4` shows a header mismatch. These are **GT-derived geometric labels**; ambiguous and co-occurring flags should be read before treating one label as a distinct model mistake.
- The prediction-only CLI ran the real model on `PMC1064135_table_0.jpg`, produced 24 cells and a risk score of 0.0006 at the tuned 0.0165 threshold. Its saved JSON has **no `ground_truth` or `comparison` fields**.
- The same image also ran successfully through `--local-model` pointed at the cached checkpoint snapshot, producing the same 24 cells and score. Local-folder results mark their revision `local-unverified` because the folder contents cannot independently prove a commit hash.
- Streamlit started headlessly; `streamlit.testing.v1.AppTest` finished with **0 exceptions** outside the sandbox.

## Held-out table-level warning results

Development selected a low-confidence floor of **0.99** and warning threshold **0.0165**. The baseline threshold was **0.0079**. These choices used only the 250 development labels. A positive table has at least one GT-matched error evidence record. The risk score is a heuristic weighted sum, not a probability.

| Final group (150 tables) | Structural risk | Confidence only |
|---|---:|---:|
| True positives | 49 | 44 |
| False alarms | 15 | 13 |
| Missed positives | 3 | 8 |
| True negatives | 83 | 85 |
| Precision | **0.7656** | 0.7719 |
| Recall | **0.9423** | 0.8462 |
| False alarm rate (of 98 error-free tables) | 0.1531 | 0.1327 |
| F1 | **0.8448** | 0.8073 |

The structural score found five more positive tables while raising two more false alarms. Its precision was slightly lower. This is a pilot on a subset of the official test split, with proxy labels from geometric GT matching; it is not a full benchmark or a calibrated reliability claim.

### Recall by GT-derived error type

Types overlap. These are table-level detection rates among tables bearing each label, not type-specific predictions.

| Error type | Tables | Structural risk recall | Confidence-only recall |
|---|---:|---:|---:|
| Extra cell | 44 | 0.9545 | 0.8636 |
| Missing cell | 35 | 0.9714 | 0.8857 |
| Likely merge | 19 | 1.0000 | 1.0000 |
| Likely split | 15 | 1.0000 | 0.9333 |
| Span mistake | 25 | 1.0000 | 0.8800 |
| Header mistake | 9 | 0.8889 | 0.8889 |

### Risk signal contribution on final group

Mean contribution is the average score component over all 150 tables. The ablation removes one signal while retaining the development-tuned threshold; correlated signals can compensate, so this is not an independent causal effect.

| Signal | Tables triggered | Mean score contribution | F1 without signal |
|---|---:|---:|---:|
| Low confidence | 150 | 0.0315 | 0.6591 |
| Cell overlap | 13 | 0.0091 | 0.8448 |
| Grid gap | 5 | 0.0038 | 0.8348 |
| Axis conflict | 9 | 0.0054 | 0.8448 |
| Suspicious span | 24 | 0.0192 | 0.8348 |
| Empty grid | 0 | 0.0000 | 0.8448 |

Low confidence carries most of the score. The geometric signals improve some detections but do not replace confidence. Every signal and its per-table evidence is recorded; `data/evaluation.json` contains the machine-readable measured report.

## Error-label interpretation

Across all 400 tables, the matcher emitted 2,153 extra-cell, 886 missing-cell, 178 span, 146 likely-split, 100 header, and 77 likely-merge evidence records. **2,988** records were marked ambiguous and **2,865** had co-occurring labels. Those counts are not distinct failure events and should not be added into an error rate. They show why the evaluation view exposes evidence and ambiguity rather than claiming each label is a separately confirmed mistake.

# ReliableTSR: Explainable Error Detection for Microsoft Table Transformer

Phase 1 detects and explains **possible** structural mistakes in cropped table images. It runs Microsoft's pinned TATR v1.1 All checkpoint. It does not repair tables or train another model.

## Scope and project layout

```text
ReliableTSR/
  app.py                    Streamlit enterprise portal (Auth, PDF upload, Qwen audit, export)
  reliabletsr/
    auth/                   PBKDF2 password hashing and multi-tenant session isolation
    storage/                RAM-only ephemeral sandbox with zero disk footprint
    document/               In-memory multi-page PDF and scan rendering
    detection/              Microsoft Table Detection (TD) for full pages
    qwen/                   Alibaba Qwen2.5-VL client, prompts, and forensic auditor
    export/                 In-memory Excel (.xlsx) and CSV table exporters
    cli.py                  prepare, infer, evaluate, predict commands
    dataset.py              official test-only download and subset selection
    model.py                pinned pretrained checkpoint
    structure.py            PASCAL VOC parsing and cell-grid reconstruction
    errors.py               one-to-one IoU matching and error evidence
    risk.py                 prediction-only warning score
    evaluation.py           development tuning and final comparison
  tests/                    automated structural, auth, storage, and export tests
  examples/                 one real saved result and measured evaluation JSON
  data/                     ignored downloads, selected pairs and results
```

The [official Table Transformer repository](https://github.com/microsoft/table-transformer) describes PubTables-1M and its structure classes. The [officially linked Hugging Face distribution](https://huggingface.co/datasets/bsmock/pubtables-1m/tree/35b1c097807e0b07ec5313879b85956b7b3890db) contains separate structure test archives. The [Microsoft checkpoint](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all/tree/7587a7ef111d9dcbf8ac695f1376ab7014340a0c) is pinned to commit `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`; the dataset mirror is pinned to `35b1c097807e0b07ec5313879b85956b7b3890db`. The model is about 115 MB; weights are cached on first use and ignored by Git. `--local-model PATH` loads a predownloaded folder with `config.json`, processor config and weights instead.

## Setup and use

Python 3.12 is the tested target. From this folder on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m reliabletsr.cli prepare --data data --count 400 --dev-count 250 --seed 1701
.\.venv\Scripts\python.exe -m reliabletsr.cli infer --data data --out data/results --device cpu
.\.venv\Scripts\python.exe -m reliabletsr.cli evaluate --results data/results --out data/evaluation.json
.\.venv\Scripts\python.exe -m streamlit run app.py
```

`requirements.txt` pins the direct dependencies; `requirements.lock.txt` records every installed transitive dependency from the verified environment. For the closest reproduction, install the lock file instead of the shorter requirements file.

For a single new image:

```powershell
.\.venv\Scripts\python.exe -m reliabletsr.cli predict path\to\cropped_table.jpg --out prediction.json
```

Use `--local-model PATH` with `infer` or `predict` to stay offline after downloading the checkpoint. `infer --limit N` supports an initial smoke test; a complete report requires both groups. `data/manifest.json` records the exact selected file names, archive sizes and group assignment. `data/results/{development,final}/*.json` contains the versioned per-table records. If matching logic changes after inference, `python -m reliabletsr.cli recompare --results data/results` refreshes comparison evidence from saved detections; run `evaluate` afterward.

## Download and disk budget

Only `PubTables-1M-Structure_Annotations_Test.tar.gz` (official listing **29.9 MB**, 93,834 XML files) and `PubTables-1M-Structure_Images_Test.tar.gz` (**2.69 GB** compressed) are requested. The script checks free disk space before each archive, prints the server's Content-Length when available, and requires at least 6 GB free. It samples 400 XML stems with Python `random.Random(1701)`, assigns 250 to development and 150 to final evaluation, and extracts only matching XML/image members. The two full compressed archives are removed after verified extraction unless `--keep-archives` is given. Plan for roughly 3 GB of compressed download plus temporary and selected-image space; network transfer and decompression can take several minutes. No training, validation, page-detection, PDF or word archive is downloaded. This is a **pilot subset of the official structure test split**, not a full PubTables-1M benchmark.

## Shared grid and confirmed error evidence

Coordinates are image pixels `[xmin, ymin, xmax, ymax]`. Each `cell` stores a bounding box; zero-based `rows` and `cols` occupied; start/end indices; `header`; `projected_row_header`; score; and source object IDs. Rows and columns are sorted by geometric centers. Every positive-area row × column intersection starts as a cell, **including blank cells**. A spanning-cell detection replaces a rectangular set of slots if it covers more than 50% of each slot; competing or irregular spans generate diagnostics. A column-header object marks a slot if it covers more than half its area. This is a documented approximation of the [Microsoft inference postprocessing](https://github.com/microsoft/table-transformer/blob/main/src/inference.py), without OCR or text content.

For labeled pairs, the same converter handles PASCAL VOC GT and thresholded TATR objects. The Hungarian matcher first maximizes the number of one-to-one pairs at IoU ≥ 0.5, then maximizes their total bounding-box IoU. A competing **eligible** match within 0.05 IoU is marked ambiguous. Unmatched GT cells are `missing_cell`; unmatched predictions are `extra_cell`. A matched difference in row/column span is `span_mistake`; header/projected-header disagreement is `header_mistake`. A predicted spanning cell that covers at least two GT cells is `likely_merge`; the reverse is `likely_split`. Likely merges/splits are interpretations of geometry. They may co-occur with missing, extra or span labels, and each evidence record lists different labels linked by the same GT or prediction cell. Low IoU or inconsistent annotation geometry can mislabel a case; inspect the evaluation view before treating a label as certain.

The JSON schema version is `1.0`. Each record saves image/group, model ID and revision, thresholds and device, raw detections (from confidence ≥ 0.05), reconstructed prediction, prediction-only risk, and for evaluation records the GT objects/grid, matches, IoUs, ambiguity and error evidence. `examples/sample_real_result.json` is a copied result from a real selected image and checkpoint. `examples/evaluation_report.json` is the measured pilot report.

## Prediction-only warnings

The upload path calls `risk.assess(prediction, detections, image_size)`. It never passes GT, matches or error labels. Signals include low structural-object confidence, overlapping cells, gaps between adjacent row/column boxes, substantial axis overlap, suspicious span diagnostics and an empty grid. Each signal gives a localized or table-level reason. The score is a capped weighted sum of each signal's maximum severity; weights are fixed in `risk.py`. The app highlights affected predicted cells and says **possible** error. The score is a heuristic likelihood indicator, **not a calibrated probability**.

The development group selects a low-confidence floor from 0.70, 0.85, 0.95 and 0.99 and a table-level warning threshold, maximizing F1 with precision as a tie break. The confidence-only baseline uses `1 - mean(confidence of accepted structural detections)` and tunes its own threshold on development. The final group is evaluated once using those fixed settings. The report includes precision, recall, false alarms/counts, per-error-type recall, and each signal's frequency, average score contribution and F1 ablation at the fixed threshold. A table is positive when any matching error evidence exists. Per-type recall can overlap because one mistake can have several labels. Do not use final-group labels to modify rules or thresholds.

After evaluation, `predict` and the upload app read the development-tuned warning threshold from `data/evaluation.json`. They use only that saved threshold plus the new image and its detections; no GT or evaluation labels enter the new-image scoring function. Without an evaluation report, the default threshold is 0.4.
Per-table records written by `infer` retain the initial default risk setting for provenance; `evaluate` recomputes every score from saved detections with the tuned development setting. `predict` and the app use the tuned setting after `evaluate` writes the report.

## Known limitations

- The model accepts **cropped table images**; page-level table detection and OCR are outside this phase.
- PASCAL VOC provides structural objects, not direct textual cell content. Blank cells are geometric slots. Header geometry and span assignment may be ambiguous.
- IoU labels are proxy structural errors, not human adjudication. A subtle content or boundary error can escape, and multiple labels can describe one mistake.
- Confidence and grid geometry can miss plausible but incorrect structures; warnings should not be used as automatic correction.
- A 400-table pilot is much smaller than the official test set. Reported metrics, when available, apply only to the selected subset.

## Verification status

See `RESULTS.md` for the actual environment check, tests, real-image run, and measured pilot metrics. Missing measurements are reported as missing rather than replaced with examples.

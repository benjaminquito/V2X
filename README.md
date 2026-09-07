# V2X Traffic Anomaly and Collision-Risk Model

This repository is a reproducible implementation of the revised model associated with
*Predicting Traffic Anomalies and Collision Risks in V2X Systems: A Deep Learning Approach
Using LSTM and GNN*. It implements the later reviewer-response design:

- a two-layer LSTM for 15-step vehicle histories;
- a two-layer, multi-head Graph Attention Network (GAT) for per-frame vehicle interactions;
- directed proximity edges between vehicles less than 20 m apart;
- a trainable softmax attention module that fuses temporal and spatial representations; and
- three-class prediction: Low (0), Medium (1), and High (2) risk.

No trained weights or real-data results are claimed by this repository. The NGSIM data is not
redistributed. Figures stated in the manuscript or later development chat are historical claims and
must not be treated as results reproduced by this code.

## Why this implementation differs from the uploaded manuscript

The uploaded manuscript describes an earlier model in several places: 50-step, three-feature
sequences; two GCN layers; continuous risk scores; and fixed 0.6/0.4 fusion. The later revision
changed these to 15-step, four-feature inputs; GAT; three-class outputs; and learned attention
fusion. This repository implements the later design requested for the reviewer response while
preserving the original claims as historical metadata only. See
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the complete reconciliation.

The earlier chat evaluation paired LSTM samples and graphs by truncating both collections to the
same length. That does not establish that the two inputs describe the same vehicle and frame. Here,
each graph node is built only when that exact vehicle has a valid history ending at the graph's
frame. Both branches therefore operate on an explicitly aligned sample.

The later chat's 99.90% accuracy and 1.00 weighted-F1 claim is not a repository baseline: the shared
evaluation snippet did not demonstrate trained fusion weights or key-based modality alignment. It
is mentioned only to prevent accidental reuse as a regenerated result.

## Repository layout

```text
configs/                 Full-data and smoke-test configurations
data/                    Raw/processed placeholders (generated data is ignored)
docs/                    Data, methodology, and reproducibility notes
results/                 Historical claims and regenerated-output placeholder
src/v2x_risk/            Preprocessing, model, training, and evaluation code
tests/                   Unit and end-to-end smoke tests
```

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

To reproduce the exact dependency versions used for the repository smoke test, install
`requirements-lock.txt` before the editable package:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

PyTorch is the only deep-learning dependency. The GAT layer uses native PyTorch scatter operations,
so no separately compiled graph library is required.

## Quick execution check

The smoke run creates a small synthetic, NGSIM-shaped dataset, preprocesses it, trains one epoch,
and evaluates all three branches:

```bash
python -m v2x_risk.smoke --config configs/smoke.yaml
```

Smoke metrics only verify that the pipeline executes. They are not research results and must not be
compared with the paper.

## Reproduce with NGSIM

1. Obtain an NGSIM trajectory CSV from the
   [U.S. DOT NGSIM Open Data portal](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/)
   and place it at `data/raw/NGSIM.csv`.
2. Review [`configs/default.yaml`](configs/default.yaml), especially the coordinate conversion and
   risk-label assumptions.
3. Run the complete pipeline:

```bash
python -m v2x_risk.preprocess --config configs/default.yaml
python -m v2x_risk.train --config configs/default.yaml
python -m v2x_risk.evaluate --config configs/default.yaml --split test
```

Or use:

```bash
./scripts/reproduce.sh configs/default.yaml
```

The loader detects CSV and Excel OOXML content from the file signature. This means an uploaded
workbook can still be read if it was accidentally named `NGSIM.csv`, although using the correct
`.xlsx` suffix is recommended.

Generated files include:

- `data/processed/ngsim_aligned_graph_windows.npz`: aligned graph/history samples;
- `data/processed/scaler.json`: train-only min-max parameters;
- `data/processed/manifest.json`: source hash, counts, split sizes, and preprocessing metadata;
- `results/regenerated/best_model.pt`: best checkpoint by validation macro F1;
- `results/regenerated/metrics.json`: training history and branch-level results;
- `results/regenerated/test_metrics.json`: independently regenerated test metrics; and
- `results/regenerated/test_predictions.csv`: per-frame predictions and learned fusion weights;
- `results/regenerated/run_summary.txt`: a plain-language completion summary; and
- `results/regenerated/run_report.html`: a visual completion and results dashboard.

At the end of `scripts/reproduce.sh`, the terminal prints `V2X RUN COMPLETED SUCCESSFULLY` followed
by the main dataset counts, held-out metrics, and output locations. Open `run_report.html` in a web
browser to verify the completed run visually.

The default configuration follows the paper's 70/15/15 proportions and five training epochs. The
split is chronological and inserts a 14-frame gap at boundaries to reduce leakage from overlapping
15-step windows. Min-max scaling is fit on the training split only.

## Required CSV columns

Core inputs are `Vehicle_ID`, `Frame_ID`, `Local_X`, and `Local_Y`. The preferred input also has
`v_Vel` and `v_Acc`. When either kinematic field is absent, preprocessing derives it from
longitudinal position and elapsed time using the explicit settings under `data.kinematics`.
Existing kinematic columns are never replaced. If the CSV already contains `Risk_Class`, it is used
after validation. Otherwise, `Time_Headway` and `Space_Headway` are also required and labels are
generated from the explicit thresholds in the configuration. The manuscript did not state the
numerical labeling thresholds, so the supplied values are transparent repository assumptions that
researchers should review or replace.

Combined files should include `Location`. Preprocessing keeps locations separate, segments reused
vehicle IDs into contiguous frame runs, and keys graph snapshots by location and timestamp. This
prevents observations from separate roads or recording sessions from entering the same trajectory
or proximity graph.

Derived speed and acceleration are a transparent compatibility fallback. Report their use and do
not treat results based on derived kinematics as directly comparable to experiments that used the
original NGSIM `v_Vel` and `v_Acc` measurements.

See [`docs/DATA.md`](docs/DATA.md) before combining NGSIM sites or changing units.

## Tests

```bash
pytest
ruff check .
```

The test suite checks class encoding, radius-graph construction, split gaps, tensor shapes, fusion
weight normalization, and an end-to-end synthetic run.

## Result-reporting rule

Only files produced under `results/regenerated/` by a completed run may be described as regenerated
results. Report the raw-data SHA-256 hash, configuration, seed, split policy, and checkpoint with
every result. Do not copy the 99.90% accuracy or 1.00 F1 claim into a new report unless this aligned
pipeline independently produces it on the intended held-out data.

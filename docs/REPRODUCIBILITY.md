# Reproducibility protocol

## Environment

Record the operating system, Python version, PyTorch version, accelerator, and dependency lock or
resolved package list. Install the project in a fresh virtual environment. The code seeds Python,
NumPy, and PyTorch and requests deterministic algorithms; exact bitwise equality can still vary
across PyTorch releases and hardware backends.

## Data provenance

Keep the generated `manifest.json` with every experiment. It records the SHA-256 hashes of the raw
CSV and processed archive, row and graph counts, class distribution, split sizes, sequence length,
and graph radius. A result is not comparable when its source hash or preprocessing configuration
differs.

## Run protocol

```bash
python -m v2x_risk.preprocess --config configs/default.yaml
python -m v2x_risk.train --config configs/default.yaml
python -m v2x_risk.evaluate --config configs/default.yaml --split test
```

Do not tune against the test split. Select checkpoints using validation macro F1 and run the held-out
test evaluation after the training configuration is fixed. Report all three branches from the same
checkpoint so the LSTM, GAT, and fused comparisons share identical samples and labels.

## Minimum reporting checklist

- Git commit hash
- Configuration file and random seed
- Raw and processed SHA-256 hashes
- NGSIM site/recording and unit interpretation
- Whether labels were supplied or generated
- Exact label thresholds when heuristics were used
- Train/validation/test counts and class distributions
- Best checkpoint epoch and selection metric
- Accuracy, macro F1, weighted F1, per-class precision/recall/F1, and confusion matrix
- Mean learned temporal/spatial fusion weights
- Hardware and software versions

## Historical claims

Figures stated in the manuscript or development chat are not expected test assertions. In
particular, the later 99.90% / 1.00 claim came from an evaluation workflow that did not demonstrate
trained fusion weights or sample-key alignment. It must be independently regenerated before it can
support a reproducibility claim.

#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-configs/default.yaml}"

python -m v2x_risk.preprocess --config "$config_path"
python -m v2x_risk.train --config "$config_path"
python -m v2x_risk.evaluate --config "$config_path" --split test

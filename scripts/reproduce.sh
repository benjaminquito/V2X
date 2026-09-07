#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

config_path="${1:-configs/default.yaml}"
if [[ -x ".venv/bin/python" ]]; then
  python_cmd=".venv/bin/python"
else
  python_cmd="python3"
fi

"$python_cmd" -m v2x_risk.preprocess --config "$config_path"
"$python_cmd" -m v2x_risk.train --config "$config_path"
"$python_cmd" -m v2x_risk.evaluate --config "$config_path" --split test
"$python_cmd" -m v2x_risk.report --config "$config_path"

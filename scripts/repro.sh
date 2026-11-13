#!/usr/bin/env bash
# One-command re-run of the CPU experiments (from the PJ root). LLM experiments run on Colab (see each run_colab.ipynb).
# Needs uv. Data goes to $HTS_DATA_DIR (default ~/data/health-twin-sharing), outside Google Drive.
set -euo pipefail
cd "$(dirname "$0")/.."
export HTS_DATA_DIR="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HTS_DATA_DIR/venv}"
uv sync --frozen
bash experiments/20261111-keystroke-pd-baseline/run.sh
bash experiments/20261111-cohort-profile/run.sh
bash experiments/20261112-dev-lists/run.sh
bash experiments/20261112-e2-map-dev/run.sh

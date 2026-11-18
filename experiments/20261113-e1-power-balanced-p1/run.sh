#!/usr/bin/env bash
# Power of the E1 decision rule, balanced estimator (V = 1/2 TPR + 1/2 TNR), by simulation (CPU only); one-sided p = (b + 1) / (B + 1) (判断 32). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e1-power-balanced-p1
uv run python scripts/e1_power.py "$E" --estimator balanced --n-sims 1000 --boot 2000 --ns 16,24,32

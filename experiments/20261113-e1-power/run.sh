#!/usr/bin/env bash
# Power of the E1 decision rule by simulation (CPU only; no cohort, no LLM). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e1-power
uv run python scripts/e1_power.py "$E" --estimator pi --n-sims 1000 --boot 2000 --ns 24,32,48

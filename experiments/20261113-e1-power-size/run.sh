#!/usr/bin/env bash
# Size of the E1 decision rule (H1 supported = Holm-adjusted p <= alpha), balanced estimator, by simulation (CPU only): scenarios with no
# effect (m_A = m_N = the K0 rate = 0.0, 0.07, 0.12; scenario "size" in power.csv) beside the 24+24 power rows. One-sided p = (b + 1) / (B + 1) (判断 32).
# Usage: bash run.sh [n_sims boot]   (default 1000 2000). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e1-power-size
uv run python scripts/e1_power.py "$E" --estimator balanced --n-sims "${1:-1000}" --boot "${2:-2000}" --ns 24

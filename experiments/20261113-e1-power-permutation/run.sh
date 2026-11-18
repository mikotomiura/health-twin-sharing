#!/usr/bin/env bash
# Size and power of the E1 decision rule (H1 supported = Holm-adjusted p <= alpha), balanced estimator, with the H1 p made by the sign-flip
# permutation test (判断 39; --test permutation, B = boot flips, p = (b + 1) / (B + 1)). Same grid and SEED as 20261113-e1-power-size (bootstrap p), CPU only.
# Usage: bash run.sh [n_sims boot]   (default 1000 2000). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e1-power-permutation
uv run python scripts/e1_power.py "$E" --estimator balanced --test permutation --n-sims "${1:-1000}" --boot "${2:-2000}" --ns 24

#!/usr/bin/env bash
# Size of the E1 H1 rule under three nulls (attack-me round 3, condition A2): exch (K2 and K0 exchangeable within a person, the sharp null of
# the sign-flip permutation test), mean0 (the earlier "size" model) and k0zero (K0 rate 0, K2 moves A and not-A together), 40 + 40 persons,
# both tests (permutation = confirmatory, bootstrap = for comparison), every simulation with its own signs / bootstrap draw. CPU only.
# Criterion fixed before the run (.steering/20261114-prereg-v4/design.md): "exceeds" = Clopper-Pearson 95% lower bound > alpha in any
# permutation row (alpha 0.025 and 0.0125).
# Usage: bash run.sh [n_sims boot]   (default 10000 2000). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261114-e1-power-size-nulls
uv run --frozen python scripts/e1_power.py "$E" --size-nulls --n-sims "${1:-10000}" --boot "${2:-2000}" --ns 40

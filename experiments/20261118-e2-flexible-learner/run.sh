#!/usr/bin/env bash
# Exploratory [X] for attack-me manuscript round 3 (R1 M-1). From the PJ root: bash experiments/20261118-e2-flexible-learner/run.sh
set -euo pipefail
E="experiments/20261118-e2-flexible-learner"
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
COHORT="$D/derived/cohort-confirm"
uv run --frozen python "$E/flexible.py" "$COHORT" "$E" > "$E/results/flexible.log"

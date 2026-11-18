#!/usr/bin/env bash
# Re-run from anywhere. Needs the flattened 10k cohort (experiments/20261111-cohort-profile/run.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/20261113-recipient-range-info
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python $E/groups_csv.py "$D/derived/cohort-10k" $E/results/groups.csv
uv run --frozen python $E/range_info.py "$D/derived/cohort-10k" | tee $E/results/auc.txt

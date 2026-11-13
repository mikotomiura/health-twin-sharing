#!/usr/bin/env bash
# Dry run of the confirmatory E2 (config/e2_confirm.json, --d2-from-config, e2_summarise) on the DEVELOPMENT cohort.
# Preliminary: plumbing, counts and run time only; not for confirmatory analysis. CPU.
# This is the template of the confirmatory E2 run.sh; see notes.md for the 3 lines to change after the freeze.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e2-dev-dryrun
IN="$D/derived/cohort-10k"
( cd "$IN" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
SUMS
echo "2d2f7ef16134dface3668fa5481b933d7435dff1959ed0690f1ac9c85ee43be6  config/condition_lists.json" | shasum -a 256 -c -
cmp config/e2_confirm.json "$E/config.json"
[[ "$(cat "$E/SEED")" == 20261113 ]]
mkdir -p "$E/results"
uv run --frozen python -m unittest discover -s tests
uv run --frozen python scripts/e2_map.py "$IN" "$E" --d2-from-config > "$E/results/e2_map.log"
uv run --frozen python scripts/e2_summarise.py "$E" --e1 experiments/20261113-e1-colab-dryrun > "$E/results/e2_summarise.log"

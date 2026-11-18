#!/usr/bin/env bash
# Dry run (v3: person bootstrap 判断 18 + primary dV_bal 判断 25・26) of the confirmatory E2 (config/e2_confirm.json, --d2-from-config, e2_summarise) on the DEVELOPMENT cohort.
# Preliminary: plumbing, counts and run time only; not for confirmatory analysis. CPU.
# Re-run after the attack-me round-1 fixes; the confirmatory template is experiments/_confirm-template/e2/run.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e2-dev-dryrun-v3
IN="$D/derived/cohort-10k"
( cd "$IN" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
SUMS
echo "9a0d490d35052b5773a5fc56738eabfc3d2aee41ce23aa69a10eceb8c355b60e  config/condition_lists.json" | shasum -a 256 -c -
cmp config/e2_confirm.json "$E/config.json"
[[ "$(cat "$E/SEED")" == 20261113 ]]
mkdir -p "$E/results"
uv run --frozen python -m unittest discover -s tests
uv run --frozen python scripts/e2_map.py "$IN" "$E" --d2-from-config > "$E/results/e2_map.log"
uv run --frozen python scripts/e2_summarise.py "$E" --e1 experiments/20261113-e1-colab-dryrun > "$E/results/e2_summarise.log"

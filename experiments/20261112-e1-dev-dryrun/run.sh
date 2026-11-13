#!/usr/bin/env bash
# Dev dry run of the confirmatory E1 pipeline (preregistration draft v1 §11 1-5) with the fake backend: checks that
# build -> run (all stages) -> summarise work end to end and that the job counts match the plan. Values are random
# (fake backend) and are not results. CPU, about 10 seconds. Colab dry run with real models: a later step.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261112-e1-dev-dryrun
( cd "$D/derived/cohort-10k" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
8abf40f480c6541eb759c49f23dd1fb5ec9a9bb03254de913487aa27a4e19df3  encounters.parquet
SUMS
cmp config/e1_confirm.json "$E/config.json"
uv run --frozen python -m unittest discover -s tests
rm -rf "$E/results" "$E/inputs"
uv run --frozen python scripts/e1_build.py "$D/derived/cohort-10k" "$E" > /dev/null
echo "67d2ff2d5d3ad9f6f384137f53ed55f0effe019c6771b6ab2ec955ee56f3d16e  $E/inputs/records.jsonl" | shasum -a 256 -c -
uv run --frozen python scripts/e1_run.py "$E" --plan | tee "$E/plan.json"
uv run --frozen python scripts/e1_run.py "$E" --stage all --backend fake --workers 8
uv run --frozen python scripts/e1_summarise.py "$E" > /dev/null
wc -l "$E"/results/*.jsonl

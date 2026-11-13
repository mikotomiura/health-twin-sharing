#!/usr/bin/env bash
# CPU, ~10 s. Needs the flattened 10k tables (see ../20261111-cohort-profile/run.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
# Input must be the pinned development tables (data.md).
( cd "$D/derived/cohort-10k" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
SUMS
uv run --frozen python -m unittest discover -s tests
uv run --frozen python scripts/dev_lists.py "$D/derived/cohort-10k" experiments/20261112-dev-lists

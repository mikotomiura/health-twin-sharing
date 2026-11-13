#!/usr/bin/env bash
# Re-run from anywhere. Needs HTS_DATA_DIR; downloads the 10k archive (926 MB) if missing and streams it (~25 min on CPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/fetch_data.py cohort-10k
[ -f "$D/derived/cohort-10k/persons.parquet" ] || \
  uv run --frozen python scripts/profile_cohort.py "$D/raw/twinllm/cohort-10k.tar.zst" "$D/derived/cohort-10k"
uv run --frozen python scripts/analyze_cohort.py "$D/derived/cohort-10k" experiments/20261111-cohort-profile

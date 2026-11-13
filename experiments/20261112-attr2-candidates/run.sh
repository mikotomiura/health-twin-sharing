#!/usr/bin/env bash
# CPU, seconds. Needs the flattened 10k tables (see ../20261111-cohort-profile/run.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/attr2_candidates.py "$D/derived/cohort-10k" experiments/20261112-attr2-candidates

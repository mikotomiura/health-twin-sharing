#!/usr/bin/env bash
# Development / confirmation split (cpu-prep). CPU only, seconds. Needs the flattened 10k cohort
# (see ../20261111-cohort-profile/run.sh). The confirmation side must not be analysed before the preregistration is frozen.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/make_split.py "$D/derived/cohort-10k" experiments/20261112-split

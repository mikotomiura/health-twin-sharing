#!/usr/bin/env bash
# Screen second-attribute candidates on the development split only (cpu-prep 2). CPU, under a minute.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/attribute_candidates.py "$D/derived/cohort-10k" experiments/20261112-second-attribute

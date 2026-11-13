#!/usr/bin/env bash
# Re-run: from the PJ root. Needs HTS_DATA_DIR with raw/neuroqwerty (scripts/fetch_data.py neuroqwerty).
set -euo pipefail
cd "$(dirname "$0")/../.."
uv run --frozen python scripts/fetch_data.py neuroqwerty
uv run --frozen python scripts/keystroke_pd_baseline.py experiments/20261111-keystroke-pd-baseline

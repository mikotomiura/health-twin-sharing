#!/usr/bin/env bash
# Recompute A's population, the attacker's rates and E2 with 'misuses drugs' in the A type and the audit type
# (dry-run decisions 7 (c)). CPU only. Needs the flattened 10k cohort (see ../20261111-cohort-profile/run.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
E=experiments/20261112-a-type-misuse
uv run --frozen python scripts/attacker_knowledge.py "$D/derived/cohort-10k" $E $E
uv run --frozen python scripts/e2_sample_size.py "$D/derived/cohort-10k" $E $E

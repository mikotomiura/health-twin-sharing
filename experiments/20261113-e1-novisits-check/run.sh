#!/usr/bin/env bash
# Colab check (paper-write 判断 23) of E1 WITHOUT the visit count (判断 16) on the DEVELOPMENT cohort: A 16 / not-A 16,
# lambda=1, D0, K0 / K0n / K2, stages twin + attack only. For the power assumptions of experiments/20261113-e1-power
# (A act rate, not-A false positives) and the K0n plumbing. Preliminary; not for confirmatory analysis.
# The model part runs on Colab (L4) as experiments/20261113-e1-colab-dryrun (VM steps in its notes / handoff).
# This script rebuilds the inputs (checked by cmp) and, if results/ has been downloaded, summarises them.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-e1-novisits-check
( cd "$D/derived/cohort-10k" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
8abf40f480c6541eb759c49f23dd1fb5ec9a9bb03254de913487aa27a4e19df3  encounters.parquet
SUMS
TMP=$(mktemp -d); cp "$E/config.json" "$E/SEED" "$TMP/"
uv run --frozen python scripts/e1_build.py "$D/derived/cohort-10k" "$TMP" > /dev/null
for f in records.jsonl knowledge.json build_report.json; do cmp "$TMP/inputs/$f" "$E/inputs/$f"; done
rm -rf "$TMP"
uv run --frozen python scripts/e1_run.py "$E" --plan
if [ -f "$E/results/attack.jsonl" ]; then uv run --frozen python scripts/e1_summarise.py "$E" > /dev/null; fi

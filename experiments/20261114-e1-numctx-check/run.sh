#!/usr/bin/env bash
# num_ctx check (preliminary; development cohort; not for confirmatory analysis). The model part runs on Colab (vm_run.sh).
# Here: rebuild the inputs (cmp), print the plan, and if results/ and vm/ are downloaded, analyse.py (the criteria of notes.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261114-e1-numctx-check
( cd "$D/derived/cohort-10k" && shasum -a 256 -c - ) <<'SUMS'
42c10d571e3f71cd95cdb383212af63efa65736aed7648d9f33ce1cf709b2772  persons.parquet
66447f64c8a6b3ca997af10dadfd7215abb3315d4de211f24cc9545f468a529d  conditions.parquet
8abf40f480c6541eb759c49f23dd1fb5ec9a9bb03254de913487aa27a4e19df3  encounters.parquet
SUMS
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp "$E/config.json" "$E/SEED" "$TMP/"
uv run --frozen python scripts/e1_build.py "$D/derived/cohort-10k" "$TMP" > /dev/null
for f in records.jsonl knowledge.json build_report.json; do cmp "$TMP/inputs/$f" "$E/inputs/$f"; done
uv run --frozen python scripts/e1_run.py "$E" --plan
if [ -f "$E/results/attack.jsonl" ]; then uv run --frozen python "$E/analyse.py" "$E"; fi
if [ -f "$E/results/attack.jsonl" ]; then uv run --frozen python "$E/compare_pilot.py" > "$E/compare_pilot.json"; cat "$E/compare_pilot.json"; fi   # post-hoc (notes.md)

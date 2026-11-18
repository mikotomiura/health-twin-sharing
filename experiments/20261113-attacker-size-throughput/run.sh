#!/usr/bin/env bash
# Attacker throughput on Colab (preliminary, for the budget). Rebuilds inputs/jobs.jsonl from the source dry run and
# checks it against data.md; if results/ exists, summarises it (metrics.json and the table in notes.md).
# The model part runs on Colab: upload config.json SEED bench.py vm_run.sh Modelfile.r1-70b-q3 inputs/jobs.jsonl to
# /content/tp/exp, start `nohup setsid bash vm_run.sh > vm_run.out 2>&1 < /dev/null &` there, and watch with
# `bash watch_sync.sh <session> 300` (see env.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"
E=experiments/20261113-attacker-size-throughput
cp "$E/inputs/jobs.jsonl" "$E/inputs/jobs.jsonl.bak"
uv run --frozen python "$E/prepare.py"
cmp "$E/inputs/jobs.jsonl" "$E/inputs/jobs.jsonl.bak" && rm "$E/inputs/jobs.jsonl.bak"
if [ -f "$E/results/bench.jsonl" ]; then uv run --frozen python "$E/summarise.py"; fi

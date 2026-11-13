#!/usr/bin/env bash
# Records are built on CPU; the LLM part ran on Colab (L4) through google-colab-cli 0.7.5:
#   colab new -s hts-dry --gpu L4
#   colab upload -s hts-dry scripts/llm_dry_run.py /content/hts/scripts/llm_dry_run.py   (and config.json, SEED, inputs/records.jsonl to /content/hts/exp/)
#   install Ollama, pull gemma3:12b and qwen3:14b (OLLAMA_MAX_LOADED_MODELS=2), then on the VM:
#     python scripts/llm_dry_run.py exp && python scripts/llm_dry_run.py exp --summarise
#   colab download ... results/{releases,attacks}.jsonl, metrics.json, ledger.csv, versions.json; colab stop -s hts-dry
# (On Git Bash set MSYS_NO_PATHCONV=1 so /content paths are not rewritten.) run_colab.ipynb is the same steps in the web UI.
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/build_records.py "$D/derived/cohort-10k" experiments/20261111-llm-dry-run
uv run --frozen python scripts/llm_dry_run_detail.py experiments/20261111-llm-dry-run

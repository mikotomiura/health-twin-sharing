#!/usr/bin/env bash
# Decision 8: re-run the pilot with the new A type (A and audit include "misuses drugs"). 32 persons (16 A), policies
# none + audit, 2 paraphrases, 3 seeds = 384 cells x 3 time points. Records are built on CPU; the twin part runs on
# Colab (L4) with the same steps as ../20261111-llm-dry-run/run.sh:
#   colab new -s hts-newtype --gpu L4
#   upload scripts/llm_dry_run.py -> /content/hts/scripts/; this dir's config.json, SEED, inputs/records.jsonl -> /content/hts/exp/
#   (check sha256 on the VM against data.md)
#   install Ollama; pull gemma3:12b and qwen3:14b (OLLAMA_MAX_LOADED_MODELS=2); on the VM:
#     python scripts/llm_dry_run.py exp && python scripts/llm_dry_run.py exp --summarise
#   download exp/results/{releases,attacks}.jsonl, metrics.json, ledger.csv, versions.json, run.log -> this dir
#   then run ../20261112-pilot-new-type-attacker (the same VM can be reused)
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
uv run --frozen python scripts/build_records.py "$D/derived/cohort-10k" experiments/20261112-pilot-new-type

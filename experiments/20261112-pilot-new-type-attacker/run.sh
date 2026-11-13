#!/usr/bin/env bash
# Decision 8: stats_think attacker (rates + reasoning, think num_predict 4096) on ../20261112-pilot-new-type releases.
# 416 attacker calls. Rates are computed on CPU; the attacker runs on Colab (L4) with the same steps as
# ../20261111-dry-run-attacker/run.sh:
#   upload scripts/{llm_dry_run,llm_dry_run_extra,llm_dry_run_attacker}.py -> /content/hts/scripts/;
#          ../20261112-pilot-new-type/{config.json, inputs/records.jsonl, results/releases.jsonl} -> /content/hts/src/...;
#          this dir's config.json, SEED, inputs/knowledge.json -> /content/hts/exp/   (check sha256 against data.md)
#   serve Ollama with OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_NUM_PARALLEL=4; qwen3:14b
#   on the VM: python scripts/llm_dry_run_attacker.py exp --source src --stage attack --arm stats_think --workers 4
#   download exp/results/attack.jsonl, versions.json, run.log -> results/; colab stop
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
E=experiments/20261112-pilot-new-type-attacker
uv run --frozen python scripts/attacker_knowledge.py "$D/derived/cohort-10k" experiments/20261112-pilot-new-type $E > /dev/null
uv run --frozen python scripts/llm_dry_run_attacker.py $E --source experiments/20261112-pilot-new-type --summarise > /dev/null
uv run --frozen python scripts/pilot_new_type_breakdown.py experiments/20261112-pilot-new-type $E > /dev/null

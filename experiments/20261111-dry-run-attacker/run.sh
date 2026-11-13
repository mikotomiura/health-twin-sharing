#!/usr/bin/env bash
# Decision 5/6 of the dry run: stronger E1 attackers (plain / stats / stats_think), integer-percent posteriors, and the
# utility re-score against the full record. The LLM part ran on Colab (L4) through google-colab-cli 0.7.5
# (on Git Bash set MSYS_NO_PATHCONV=1):
#   colab new -s hts-attk --gpu L4
#   upload scripts/{llm_dry_run,llm_dry_run_extra,llm_dry_run_attacker}.py -> /content/hts/scripts/;
#          ../20261111-llm-dry-run/{config.json, inputs/records.jsonl, results/releases.jsonl} -> /content/hts/src/...;
#          this dir's config.json, SEED, inputs/knowledge.json -> /content/hts/exp/   (check sha256 on the VM)
#   install Ollama; serve with OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_NUM_PARALLEL=4; pull qwen3:14b
#   on the VM: python scripts/llm_dry_run_attacker.py exp --source src --stage all --workers 4
#   download exp/results/{attack,utility}.jsonl, versions.json, run.log; colab stop -s hts-attk
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
E=experiments/20261111-dry-run-attacker
uv run --frozen python scripts/attacker_knowledge.py "$D/derived/cohort-10k" experiments/20261111-llm-dry-run $E > /dev/null
uv run --frozen python scripts/llm_dry_run_attacker.py $E --source experiments/20261111-llm-dry-run --summarise > /dev/null

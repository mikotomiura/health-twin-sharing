#!/usr/bin/env bash
# CPU parts (a: threshold g from the first dry run's attacker posteriors; e: E2 sample size on the 10k cohort),
# and the summary of the Colab parts (b, c, d).
# The LLM parts ran on Colab (L4) through google-colab-cli 0.7.5 (on Git Bash set MSYS_NO_PATHCONV=1):
#   colab new -s hts-extra --gpu L4
#   upload scripts/llm_dry_run.py, scripts/llm_dry_run_extra.py -> /content/hts/scripts/;
#          ../20261111-llm-dry-run/{config.json, inputs/records.jsonl, results/releases.jsonl} -> /content/hts/src/...;
#          this dir's config.json, SEED -> /content/hts/exp/   (create the folders first; check sha256 on the VM)
#   install Ollama; serve with OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_NUM_PARALLEL=4; pull gemma3:12b, qwen3:14b
#   on the VM: python scripts/llm_dry_run_extra.py exp --source src --stage all --workers 4
#   download exp/results/{reread,utility,paraphrase,judge_para,attack_para}.jsonl, versions.json, run.log; colab stop -s hts-extra
set -euo pipefail
cd "$(dirname "$0")/../.."
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
E=experiments/20261111-dry-run-extra
uv run --frozen python scripts/threshold_g.py experiments/20261111-llm-dry-run/results/attacks.jsonl $E/results/a_threshold_g.json > /dev/null
uv run --frozen python scripts/e2_sample_size.py "$D/derived/cohort-10k" experiments/20261111-llm-dry-run $E
uv run --frozen python scripts/llm_dry_run_extra.py $E --source experiments/20261111-llm-dry-run --summarise > /dev/null

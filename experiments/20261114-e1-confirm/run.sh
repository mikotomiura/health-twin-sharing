#!/usr/bin/env bash
# TEMPLATE of the confirmatory E1 run.sh (copy to experiments/<YYYYMMDD>-e1-confirm/run.sh after the freeze).
# Part of the frozen set. After the freeze replace every __FILL_AT_FREEZE__ below (SEED, E = the experiment dir name,
# COHORT = the confirmatory cohort's derived dir, the sha256 of its persons / conditions parquet files and of its manifest.json
# (SHA_MANIFEST: the joined manifest written by scripts/merge_cohort_shards.py, which holds the shard manifests and the sha256 of every shard file), FROZEN_COMMIT = the
# commit of the freeze, SHA_E1_CONFIG / SHA_LISTS = sha256 of config/e1_confirm.json / config/condition_lists.json)
# and change nothing else. DIGEST_AGENT / DIGEST_ATTACKER / OLLAMA_VERSION are filled BEFORE the freeze and are part of the frozen commit
# (the values registered in the preregistration; 2026-11-14: the registry manifests and the Colab runs' results/versions.jsonl agree).
# On Colab install exactly that Ollama version: curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.16 sh   (the same value as OLLAMA_VERSION below)
# It stops if HEAD != FROZEN_COMMIT or any file of the frozen set (the FROZEN array below) differs from HEAD or is untracked.
# Run as is, it stops at the first unfilled mark. It also stops unless HTS_PREREG_FROZEN=yes is set by hand.
# Origin: experiments/20261113-e1-colab-dryrun/run.sh (same steps).
#
#   HTS_PREREG_FROZEN=yes bash run.sh            check cohort + config, rebuild inputs (CPU), print the plan (nothing is run or summarised)
#   HTS_PREREG_FROZEN=yes bash run.sh run        also the model stages on the Colab VM (Ollama): e1_run.py --stage all (resumes)
#                                                 and, only after it exited with code 0, the E1 summary (e1_summarise.py), made once
# Without `run` nothing is summarised (判断 36: the E1 summary is made once, after every stage has finished).
# The scripts also refuse a development SEED (42 43 44 9001 9002 20261113) when HTS_PREREG_FROZEN=yes.
set -euo pipefail
SEED="1650422369"
E="experiments/20261114-e1-confirm"                    # e.g. experiments/20261201-e1-confirm
COHORT="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}/derived/cohort-confirm"               # the confirmatory cohort's derived dir (persons / conditions .parquet; the output of scripts/merge_cohort_shards.py)
SHA_PERSONS="e2a953c8206d1a72aa5b666269764981af87be4c7b005557d9bfdb25f66906b0"
SHA_CONDITIONS="64d7a5f97e755da916ddb7b8531a6ce68993cb3658b85cfa87ab310269330def"
SHA_MANIFEST="cf6d18e4a0f375b2a7870614c749d00eaa9e4a11ab18342277c69c650ac5139b"         # sha256 of $COHORT/manifest.json (the joined manifest of the shards)
FROZEN_COMMIT="b35f5f7f669ef2f7a41201e90e96463ffbbe8bc6"        # git rev-parse HEAD at the freeze
SHA_E1_CONFIG="aede50a0a98d35f11b2eab2ae604927311e91edafe591bf98f026d05227a5e43"        # sha256 of config/e1_confirm.json
SHA_LISTS="9a0d490d35052b5773a5fc56738eabfc3d2aee41ce23aa69a10eceb8c355b60e"            # sha256 of config/condition_lists.json
DIGEST_AGENT="b92b218582c521ba287f38f1d6057dcd3c6cbff036f13cac97b5d861f5e58ec3"         # Ollama digest of gemma3:12b (the twin), exactly as in results/versions.jsonl "models"
DIGEST_ATTACKER="bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"      # Ollama digest of qwen3:14b (attacker and judge)
OLLAMA_VERSION="0.40.16"       # Ollama version string, as in results/versions.jsonl "ollama"

# The frozen set (round 3 A5): exactly the files that decide the result; nothing else in scripts/ config/ tests/ is checked or run.
FROZEN=(
  scripts/generate_confirm_cohort.sh scripts/merge_cohort_shards.py scripts/profile_cohort.py scripts/build_records.py
  scripts/permute_lambda.py scripts/freeze_d2.py scripts/e1_build.py scripts/e1_run.py scripts/e1_summarise.py
  scripts/llm_dry_run.py scripts/llm_dry_run_attacker.py scripts/llm_dry_run_extra.py scripts/e2_map.py scripts/e2_summarise.py
  scripts/e1_power.py
  config/condition_lists.json config/e1_confirm.json config/e2_confirm.json
  experiments/_confirm-template/e1/run.sh experiments/_confirm-template/e2/run.sh
  tests/test_e1.py tests/test_e2_map.py tests/test_e2_summarise.py tests/test_permute_lambda.py tests/test_prereg2.py tests/test_e1_power.py
  pyproject.toml uv.lock
)

die() { echo "run.sh: $*" >&2; exit 1; }
for v in SEED E COHORT SHA_PERSONS SHA_CONDITIONS SHA_MANIFEST FROZEN_COMMIT SHA_E1_CONFIG SHA_LISTS DIGEST_AGENT DIGEST_ATTACKER OLLAMA_VERSION; do
  [[ "${!v}" != *__FILL_AT_FREEZE__* ]] || die "$v is not filled in: this is the template; fill it in at the freeze"
done
[[ "${HTS_PREREG_FROZEN:-}" == yes ]] || die "confirmatory run only after the freeze: set HTS_PREREG_FROZEN=yes"
case "$SEED" in 42|43|44|9001|9002|20261113) die "SEED $SEED is a development or test seed";; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"

cd "$(dirname "$0")"
while [[ ! -f config/e1_confirm.json || ! -f scripts/e1_run.py ]]; do
  [[ "$PWD" != / ]] || die "PJ root not found"
  cd ..
done
D="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$D/venv}"

( cd "$COHORT" && shasum -a 256 -c - ) <<SUMS
$SHA_PERSONS  persons.parquet
$SHA_CONDITIONS  conditions.parquet
$SHA_MANIFEST  manifest.json
SUMS
git rev-parse --git-dir > /dev/null 2>&1 || die "not in a git work tree: the frozen commit cannot be checked"
[[ "$(git rev-parse HEAD)" == "$FROZEN_COMMIT" ]] || die "HEAD is not FROZEN_COMMIT ($FROZEN_COMMIT)"
# every shard of the cohort was generated from the frozen commit (pjCommit in the joined manifest; review 2026-11-14)
uv run --frozen python -c 'import json, sys; ms = json.load(open(sys.argv[1]))["shards"]; bad = [m.get("shard") for m in ms if m.get("pjCommit") != sys.argv[2]]; sys.exit(f"run.sh: the cohort was not generated from FROZEN_COMMIT (shards {bad})" if bad or len(ms) != 4 else 0)' "$COHORT/manifest.json" "$FROZEN_COMMIT"
git diff --quiet HEAD -- "${FROZEN[@]}" || die "a file of the frozen set (FROZEN array) differs from the frozen commit (uncommitted changes)"
[[ -z "$(git status --porcelain --untracked-files=all -- "${FROZEN[@]}")" ]] || die "untracked or changed files in the frozen set (FROZEN array; commit or remove them)"
# untracked files under scripts/ tests/ could shadow the frozen code's imports (scripts/ is first on sys.path); only "??" is checked
[[ -z "$(git status --porcelain --untracked-files=all -- scripts tests | grep '^??' || true)" ]] || die "untracked files in scripts/ or tests/ (they could shadow imports of the frozen code; remove them)"
echo "$SHA_E1_CONFIG  config/e1_confirm.json" | shasum -a 256 -c -
echo "$SHA_LISTS  config/condition_lists.json" | shasum -a 256 -c -
env -u HTS_PREREG_FROZEN uv run --frozen python -m unittest tests.test_e1 tests.test_e2_map tests.test_e2_summarise tests.test_permute_lambda tests.test_prereg2 tests.test_e1_power
mkdir -p "$E"
if [[ -f "$E/config.json" ]]; then cmp config/e1_confirm.json "$E/config.json"; else cp config/e1_confirm.json "$E/config.json"; fi
if [[ -f "$E/SEED" ]]; then [[ "$(cat "$E/SEED")" == "$SEED" ]] || die "$E/SEED differs from SEED"; else echo "$SEED" > "$E/SEED"; fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp "$E/config.json" "$E/SEED" "$TMP/"
uv run --frozen python scripts/e1_build.py "$COHORT" "$TMP" > /dev/null
if [[ -d "$E/inputs" ]]; then
  for f in records.jsonl knowledge.json build_report.json; do cmp "$TMP/inputs/$f" "$E/inputs/$f"; done
else
  cp -R "$TMP/inputs" "$E/inputs"
fi
uv run --frozen python scripts/e1_run.py "$E" --plan
if [[ "${1:-}" == run ]]; then
  [[ ! -f "$E/metrics.json" ]] || die "$E/metrics.json exists: the E1 summary is made once, after every stage has finished"
  uv run --frozen python scripts/e1_run.py "$E" --stage all --backend ollama \
    --expect-digests "gemma3:12b=$DIGEST_AGENT,qwen3:14b=$DIGEST_ATTACKER" --expect-ollama "$OLLAMA_VERSION"   # digests / version checked before any model call, every run; set -e: a non-zero exit stops here, nothing is summarised
  uv run --frozen python scripts/e1_summarise.py "$E" > /dev/null
fi

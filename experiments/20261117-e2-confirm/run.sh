#!/usr/bin/env bash
# TEMPLATE of the confirmatory E2 run.sh (copy to experiments/<YYYYMMDD>-e2-confirm/run.sh after the freeze).
# Part of the frozen set. After the freeze replace every __FILL_AT_FREEZE__ below (SEED, E = the experiment dir name,
# E1 = the confirmatory E1 experiment dir, SHA_E1_METRICS = sha256 of $E1/metrics.json (the one E1 summary, made after every E1 stage
# finished), COHORT = the confirmatory cohort's derived dir, the sha256 of its parquet files and of its manifest.json
# (SHA_MANIFEST: the joined manifest of scripts/merge_cohort_shards.py with the shard manifests and the sha256 of every shard file), FROZEN_COMMIT = the commit of the freeze, SHA_E2_CONFIG / SHA_LISTS = sha256 of config/e2_confirm.json /
# config/condition_lists.json) and change nothing else. It stops if HEAD != FROZEN_COMMIT or any file of the frozen set (the FROZEN array below) differs from HEAD or is untracked.
# Run as is, it stops at the first unfilled mark. It also stops unless HTS_PREREG_FROZEN=yes is set by hand.
# Origin: experiments/20261113-e2-dev-dryrun/run.sh (same steps; e2_map.py always with --d2-from-config).
#
#   HTS_PREREG_FROZEN=yes bash run.sh     (CPU; e2_map.py about 13 hours on the 40k cohort, 判断 34)
# The scripts also refuse a development SEED (42 43 44 9001 9002 20261113) when HTS_PREREG_FROZEN=yes.
set -euo pipefail
SEED="707983363"
E="experiments/20261117-e2-confirm"                    # e.g. experiments/20261201-e2-confirm
E1="experiments/20261114-e1-confirm"                   # e.g. experiments/20261201-e1-confirm (its metrics.json gives H1 for Holm)
SHA_E1_METRICS="6416dbcc03692023bde8a0fbd119f1528d2f3bb8f91a9e9472d85d1dfc30c881"       # sha256 of $E1/metrics.json: known only after the E1 summary, so it is the one mark filled in after the freeze (and before E2 runs)
COHORT="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}/derived/cohort-confirm"               # the confirmatory cohort's derived dir (the output of scripts/merge_cohort_shards.py)
SHA_PERSONS="e2a953c8206d1a72aa5b666269764981af87be4c7b005557d9bfdb25f66906b0"
SHA_CONDITIONS="64d7a5f97e755da916ddb7b8531a6ce68993cb3658b85cfa87ab310269330def"
SHA_MANIFEST="cf6d18e4a0f375b2a7870614c749d00eaa9e4a11ab18342277c69c650ac5139b"         # sha256 of $COHORT/manifest.json (the joined manifest of the shards)
FROZEN_COMMIT="b35f5f7f669ef2f7a41201e90e96463ffbbe8bc6"        # git rev-parse HEAD at the freeze
SHA_E2_CONFIG="7441d2e9a119d00a5c9968a998bf03d228d643b7b633f1fd0aa6ff29d6c1e7ae"        # sha256 of config/e2_confirm.json
SHA_LISTS="9a0d490d35052b5773a5fc56738eabfc3d2aee41ce23aa69a10eceb8c355b60e"            # sha256 of config/condition_lists.json

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
for v in SEED E E1 SHA_E1_METRICS COHORT SHA_PERSONS SHA_CONDITIONS SHA_MANIFEST FROZEN_COMMIT SHA_E2_CONFIG SHA_LISTS; do
  [[ "${!v}" != *__FILL_AT_FREEZE__* ]] || die "$v is not filled in: this is the template; fill it in at the freeze"
done
[[ "${HTS_PREREG_FROZEN:-}" == yes ]] || die "confirmatory run only after the freeze: set HTS_PREREG_FROZEN=yes"
case "$SEED" in 42|43|44|9001|9002|20261113) die "SEED $SEED is a development or test seed";; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a non-negative integer"

cd "$(dirname "$0")"
while [[ ! -f config/e2_confirm.json || ! -f scripts/e2_map.py ]]; do
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
echo "$SHA_E2_CONFIG  config/e2_confirm.json" | shasum -a 256 -c -
echo "$SHA_LISTS  config/condition_lists.json" | shasum -a 256 -c -
mkdir -p "$E/results"
if [[ -f "$E/config.json" ]]; then cmp config/e2_confirm.json "$E/config.json"; else cp config/e2_confirm.json "$E/config.json"; fi
if [[ -f "$E/SEED" ]]; then [[ "$(cat "$E/SEED")" == "$SEED" ]] || die "$E/SEED differs from SEED"; else echo "$SEED" > "$E/SEED"; fi
[[ -f "$E1/metrics.json" ]] || die "$E1/metrics.json not found: run and summarise the confirmatory E1 first (H1 enters the Holm step)"
echo "$SHA_E1_METRICS  $E1/metrics.json" | shasum -a 256 -c -
uv run --frozen python -c 'import json, sys; m = json.load(open(sys.argv[1])); sys.exit(0 if (m.get("completeness") or {}).get("complete") is True and (m.get("H1_K2_minus_K0") or {}).get("p_one_sided_permutation") is not None else "run.sh: E1 metrics.json is not complete or has no H1_K2_minus_K0.p_one_sided_permutation (do not start the 13-hour map)")' "$E1/metrics.json"
# E2's SEED must differ from E1's SEED ($E1/SEED) and from the 8 seeds of the cohort (shards[].seed and .clinicianSeed in $COHORT/manifest.json); checked before the map
uv run --frozen python -c 'import json, sys; s = int(sys.argv[1]); e1 = int(open(sys.argv[2]).read().strip()); cs = [int(x) for sh in json.load(open(sys.argv[3]))["shards"] for x in (sh["seed"], sh["clinicianSeed"])]; sys.exit("run.sh: the cohort manifest does not hold 8 seeds" if len(cs) != 8 else "run.sh: E2 SEED equals the E1 SEED" if s == e1 else "run.sh: E2 SEED equals a cohort seed (shard seed or clinicianSeed)" if s in cs else 0)' "$SEED" "$E1/SEED" "$COHORT/manifest.json"
env -u HTS_PREREG_FROZEN uv run --frozen python -m unittest tests.test_e1 tests.test_e2_map tests.test_e2_summarise tests.test_permute_lambda tests.test_prereg2 tests.test_e1_power
uv run --frozen python scripts/e2_map.py "$COHORT" "$E" --d2-from-config > "$E/results/e2_map.log"
uv run --frozen python scripts/e2_summarise.py "$E" --e1 "$E1" > "$E/results/e2_summarise.log"

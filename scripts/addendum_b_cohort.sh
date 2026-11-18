#!/usr/bin/env bash
# Addendum B: generate one addendum cohort (10k) with the frozen v4 generator, unchanged.
#   HTS_PREREG_FROZEN=yes COHORT=<1|2|3> WORK=/content/synthea-work bash scripts/addendum_b_cohort.sh
# Reads the cohort's seed pair from config/addendum_seeds.json (drawn once at the addendum freeze by
# scripts/addendum_draw_seeds.py) and runs scripts/generate_confirm_cohort.sh in confirm mode with SHARD=1 (one 10k shard;
# the v4 generator knows only shards 1..4 of 10k, and a 10k addendum cohort is exactly one such shard). The generator's
# checks (freeze flag, clean scripts/, pinned Synthea jar and open-twin recipe, fixed reference date) all apply.
# Output: $WORK/derived-addendum-c<COHORT>/{persons,conditions,encounters}.parquet + manifest.json (the generator's,
# with shard 1) + addendum.json (cohort index, seeds, the seeds file sha256).
set -euo pipefail
die() { echo "addendum_b_cohort: $*" >&2; exit 1; }
PJ="$(cd "$(dirname "$0")/.." && pwd)"
[[ "${COHORT:-}" =~ ^[1-3]$ ]] || die "COHORT must be 1, 2 or 3"
[[ -n "${WORK:-}" ]] || die "set WORK (a dedicated directory outside the PJ)"
SEEDS="$PJ/config/addendum_seeds.json"
[[ -f "$SEEDS" ]] || die "$SEEDS not found (drawn at the addendum freeze)"
read -r SEED CLINICIAN_SEED < <(python3 -c 'import json, sys; c = json.load(open(sys.argv[1]))["cohorts"][int(sys.argv[2]) - 1]; assert c["cohort"] == int(sys.argv[2]); print(c["seed"], c["clinicianSeed"])' "$SEEDS" "$COHORT")
OUT="$WORK/derived-addendum-c$COHORT"
[[ ! -e "$OUT" ]] || die "$OUT already exists (each cohort is made once)"
[[ ! -e "$WORK/derived-shard1" ]] || die "$WORK/derived-shard1 exists; move it away first (the generator writes there)"
SHARD=1 SEED="$SEED" CLINICIAN_SEED="$CLINICIAN_SEED" WORK="$WORK" bash "$PJ/scripts/generate_confirm_cohort.sh" confirm
python3 - "$WORK/derived-shard1/manifest.json" "$SEED" "$CLINICIAN_SEED" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert (m["seed"], m["clinicianSeed"], m["population"]) == (int(sys.argv[2]), int(sys.argv[3]), 10000), m
EOF
mv "$WORK/derived-shard1" "$OUT"
python3 - "$OUT/addendum.json" "$COHORT" "$SEED" "$CLINICIAN_SEED" "$(sha256sum "$SEEDS" | awk '{print $1}')" <<'EOF'
import json, sys
json.dump({"cohort": int(sys.argv[2]), "seed": int(sys.argv[3]), "clinicianSeed": int(sys.argv[4]),
           "seeds_file_sha256": sys.argv[5], "generator": "scripts/generate_confirm_cohort.sh confirm, SHARD=1"},
          open(sys.argv[1], "w"), indent=1)
EOF
sha256sum "$OUT"/*

#!/usr/bin/env bash
# Generate a Synthea cohort with the open-twin recipe (tools/synthea at a pinned commit, unchanged synthea.properties)
# plus a fixed reference / end date (decisions 判断 1・12). Meant for a Colab CPU VM (Java 21; ~41 GB unpacked for 10k),
# then flattened to parquet on the VM with scripts/profile_cohort.py so that only the small tables come home.
#
#   WORK=/content/synthea-work bash scripts/generate_confirm_cohort.sh test
#       trial: TEST seeds, N=50 (allowed before the freeze)
#   HTS_PREREG_FROZEN=yes SEED=<s> CLINICIAN_SEED=<cs> WORK=/content/synthea-work bash scripts/generate_confirm_cohort.sh confirm
#       confirmatory 10k: seeds written in the frozen preregistration; refuses to run without the freeze flag
#
# WORK must be a dedicated directory outside the PJ (it is created; only $WORK/out is replaced, and only when it holds
# a manifest.json from an earlier run of this script).
# Output: $WORK/out/fhir/*.json, $WORK/out/manifest.json (recipe + dates + content checksum), and
#         $WORK/derived/{persons,conditions,encounters}.parquet when uv is available.
set -euo pipefail

MODE="${1:-}"
OPEN_TWIN_COMMIT=6e5f5b85e4ec652fb90071669e3b899286f547fb
# Pinned here, not read from open-twin: Synthea v4.0.0 (same values as open-twin's tools/synthea/version.lock).
SYNTHEA_VERSION=v4.0.0
SYNTHEA_JAR_URL=https://github.com/synthetichealth/synthea/releases/download/v4.0.0/synthea-with-dependencies.jar
SYNTHEA_JAR_SHA256=ed43c20ad40ba5c3bc724503a5af032715fe3c491620b766148e7c2361e6ecc1
REF_DATE=20261025            # -r and -e (判断 12): the upload date of the public 10k release
STATE=Massachusetts
TEST_SEED=9001 TEST_CLINICIAN_SEED=9002 TEST_N=50
PJ="$(cd "$(dirname "$0")/.." && pwd)"

die() { echo "generate: $*" >&2; exit 1; }
is_int() { [[ "$1" =~ ^(0|[1-9][0-9]{0,17})$ ]]; }   # canonical decimal, fits a Java long

case "$MODE" in
  test)    SEED=$TEST_SEED; CLINICIAN_SEED=$TEST_CLINICIAN_SEED; N=$TEST_N ;;
  confirm) [[ "${HTS_PREREG_FROZEN:-}" == yes ]] || die "confirmatory generation only after the preregistration is frozen (set HTS_PREREG_FROZEN=yes)"
           [[ -n "${SEED:-}" && -n "${CLINICIAN_SEED:-}" ]] || die "SEED and CLINICIAN_SEED from the frozen preregistration are required"
           if ! is_int "$SEED" || ! is_int "$CLINICIAN_SEED"; then die "SEED and CLINICIAN_SEED must be non-negative integers"; fi
           for s in "$SEED" "$CLINICIAN_SEED"; do
             [[ "$s" != 42 && "$s" != 43 && "$s" != "$TEST_SEED" && "$s" != "$TEST_CLINICIAN_SEED" ]] || die "seed $s is a development or test seed"
           done
           N=10000 ;;
  *) die "usage: $0 test|confirm" ;;
esac

[[ -n "${WORK:-}" ]] || die "set WORK to a dedicated directory outside the PJ (e.g. /content/synthea-work)"
check_work() {  # $1 = an absolute path
  local pj; pj="$(cd "$PJ" && pwd -P)"
  case "$1/" in
    "$pj/"*|"$(cd "$HOME" && pwd -P)/"|//*) die "WORK must not be inside the PJ, the home directory or /: $1" ;;
  esac
  case "$pj/" in
    "$1/"*) die "WORK must not contain the PJ: $1" ;;
  esac
}
[[ -d "$(dirname "$WORK")" ]] || die "the parent of WORK must exist: $WORK"
check_work "$(cd "$(dirname "$WORK")" && pwd -P)/$(basename "$WORK")"   # before creating anything
mkdir -p "$WORK"
WORK="$(cd "$WORK" && pwd -P)"
check_work "$WORK"                                                     # after resolving .. and symlinks
command -v java >/dev/null || die "Java 21+ is required (on Colab: apt-get install -y openjdk-21-jre-headless)"
cd "$WORK"

if [[ ! -d open-twin ]]; then git clone -q https://github.com/Opening-Science/open-twin.git; fi
git -C open-twin fetch -q origin
git -C open-twin checkout -q --detach "$OPEN_TWIN_COMMIT"
[[ "$(git -C open-twin rev-parse HEAD)" == "$OPEN_TWIN_COMMIT" ]] || die "open-twin is not at $OPEN_TWIN_COMMIT"
[[ -z "$(git -C open-twin status --porcelain --untracked-files=no)" ]] || die "open-twin has local changes"
T=open-twin/tools/synthea
grep -q "^SYNTHEA_JAR_SHA256=$SYNTHEA_JAR_SHA256\$" "$T/version.lock" || die "open-twin's version.lock pins a different jar"

JAR=synthea-with-dependencies.jar
if [[ ! -f $JAR ]]; then
  curl -fsSL -o "$JAR.partial" "$SYNTHEA_JAR_URL"
  [[ "$(sha256sum "$JAR.partial" | awk '{print $1}')" == "$SYNTHEA_JAR_SHA256" ]] || { rm -f "$JAR.partial"; die "downloaded JAR checksum mismatch"; }
  mv "$JAR.partial" "$JAR"
fi
[[ "$(sha256sum "$JAR" | awk '{print $1}')" == "$SYNTHEA_JAR_SHA256" ]] || die "JAR checksum mismatch (delete $WORK/$JAR)"

if [[ -e out ]]; then
  [[ -f out/manifest.json ]] || die "$WORK/out exists but was not made by this script; remove it yourself"
  rm -rf out
fi
# Same flags as $T/generate.sh, plus -r / -e. exporter.baseDirectory in synthea.properties is ./out/ (relative to cwd).
java -jar "$JAR" -c "$T/synthea.properties" -s "$SEED" -cs "$CLINICIAN_SEED" -p "$N" -r "$REF_DATE" -e "$REF_DATE" "$STATE"

count=$(find out/fhir -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')
(( count > 0 )) || die "Synthea wrote no FHIR files"
# Same content checksum as the public release notes: sha256sum *.json | sort | sha256sum (file names without ./).
content=$(cd out/fhir && find . -maxdepth 1 -type f -name '*.json' -printf '%f\0' | sort -z | xargs -0 -r sha256sum | sort | sha256sum | awk '{print $1}')
cat > out/manifest.json <<EOF
{
  "recipe": "open-twin $OPEN_TWIN_COMMIT tools/synthea (unchanged synthea.properties) + -r/-e",
  "syntheaVersion": "$SYNTHEA_VERSION",
  "jarSha256": "$SYNTHEA_JAR_SHA256",
  "mode": "$MODE",
  "seed": $SEED,
  "clinicianSeed": $CLINICIAN_SEED,
  "population": $N,
  "state": "$STATE",
  "referenceDate": "$REF_DATE",
  "endDate": "$REF_DATE",
  "fhirFileCount": $count,
  "contentSha256": "$content"
}
EOF
cat out/manifest.json

if command -v uv >/dev/null; then
  (cd "$PJ" && uv run --frozen python scripts/profile_cohort.py "$WORK/out/fhir" "$WORK/derived")
  sha256sum "$WORK"/derived/*.parquet
fi

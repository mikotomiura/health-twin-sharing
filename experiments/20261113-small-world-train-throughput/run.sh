#!/usr/bin/env bash
# Local part (any machine with the PJ and the derived development cohort):
#   bash experiments/20261113-small-world-train-throughput/run.sh pack
#     -> builds corpus/K2 with scripts/sw_corpus.py (checks the SHA-256 in data.md) and writes bundle.tar.gz
#        (scripts/sw_corpus.py, sw_ranges.py, sw_train.py, this dir without results, the reading-check releases)
# Then on Colab (see notes.md "回し方"): upload bundle.tar.gz, unpack into /content/hts, start vm_run.sh, watch, fetch results.
set -euo pipefail
PJ="$(cd "$(dirname "$0")/../.." && pwd)"
E=experiments/20261113-small-world-train-throughput
cd "$PJ"
export HTS_DATA_DIR="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HTS_DATA_DIR/venv}"
case "${1:-}" in
  pack)
    rm -rf "$E/corpus"
    uv run --frozen python scripts/sw_corpus.py "$HTS_DATA_DIR/derived/cohort-10k" "$E" K2
    (cd "$E" && sha256sum corpus/K2/train.txt corpus/K2/train_persons.jsonl corpus/K2/check.jsonl 2>/dev/null || shasum -a 256 corpus/K2/train.txt corpus/K2/train_persons.jsonl corpus/K2/check.jsonl) \
      | sed "s#  corpus/#  $E/corpus/#" > "$E/corpus.sha256"
    cat "$E/corpus.sha256"
    # compare with the values recorded in data.md (the reference build); stop on any difference
    if ! grep -qF -f "$E/corpus.sha256" "$E/data.md" || [ "$(grep -cF -f "$E/corpus.sha256" "$E/data.md")" -ne "$(wc -l < "$E/corpus.sha256")" ]; then
      echo "corpus SHA-256 differs from data.md; stop and record why" >&2; exit 1
    fi
    # write to a temporary name first: overwriting in place on Google Drive gave "tar: Write error" once (2026-11-13)
    # scripts/sw_ranges.py: sw_corpus imports it since the recipient ranges (cross-review 2026-11-13, HIGH; the bundle of the
    # 2026-11-13 run predates that import and had only the two scripts)
    tar czf "$E/bundle.tar.gz.tmp" scripts/sw_corpus.py scripts/sw_ranges.py scripts/sw_train.py \
      "$E/config.json" "$E/SEED" "$E/vm_run.sh" "$E/corpus.sha256" "$E/corpus" \
      experiments/20261113-e1-colab-dryrun/results/twin.jsonl
    mv -f "$E/bundle.tar.gz.tmp" "$E/bundle.tar.gz"
    ls -la "$E/bundle.tar.gz" ;;
  *) echo "usage: $0 pack" >&2; exit 2 ;;
esac

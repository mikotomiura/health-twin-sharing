#!/usr/bin/env bash
# Local part (any machine with the PJ and the derived development cohort):
#   bash experiments/20261114-sw-ranges-smoke/run.sh pack
#     -> builds the K2 corpora with scripts/sw_corpus.py for dm (drug misuse) and t2d (type 2 diabetes), each for the
#        researcher (also the source of the shared tokenizer), insurer and employer, writes corpus.sha256 and checks the
#        dm researcher corpus against the throughput run (experiments/20261113-small-world-train-throughput/data.md:
#        the recipient code must leave researcher files byte-identical), then writes bundle.tar.gz
# Then on Colab (notes.md "回し方"): upload bundle.tar.gz, unpack into /content/hts, start vm_run.sh, watch, fetch.
set -euo pipefail
PJ="$(cd "$(dirname "$0")/../.." && pwd)"
E=experiments/20261114-sw-ranges-smoke
cd "$PJ"
export HTS_DATA_DIR="${HTS_DATA_DIR:-$HOME/data/health-twin-sharing}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HTS_DATA_DIR/venv}"
sha() { sha256sum "$@" 2>/dev/null || shasum -a 256 "$@"; }
case "${1:-}" in
  pack)
    rm -rf "$E/dm/corpus" "$E/t2d/corpus"
    for sub in dm t2d; do
      for r in researcher insurer employer; do
        uv run --frozen python scripts/sw_corpus.py "$HTS_DATA_DIR/derived/cohort-10k" "$E/$sub" K2 "$r"
      done
    done
    find "$E/dm/corpus" "$E/t2d/corpus" -type f | LC_ALL=C sort | while read -r f; do sha "$f"; done > "$E/corpus.sha256"
    cat "$E/corpus.sha256"
    # the dm researcher corpus must equal the throughput run's (same cohort, seed, attribute; probes do not enter the corpus)
    T=experiments/20261113-small-world-train-throughput
    for f in train.txt train_persons.jsonl check.jsonl; do
      [ -f "$E/dm/corpus/K2/$f" ] || { echo "missing $E/dm/corpus/K2/$f" >&2; exit 1; }
      a=$(sha "$E/dm/corpus/K2/$f" | cut -c1-64)
      grep -qF "$a  $T/corpus/K2/$f" "$T/data.md" || { echo "dm researcher $f differs from $T/data.md; stop and record why" >&2; exit 1; }
    done
    # temporary name first: overwriting in place on Google Drive gave "tar: Write error" once (2026-11-13)
    tar czf "$E/bundle.tar.gz.tmp" scripts/sw_corpus.py scripts/sw_ranges.py scripts/sw_train.py \
      "$E/dm/config.json" "$E/dm/SEED" "$E/t2d/config.json" "$E/t2d/SEED" "$E/vm_run.sh" "$E/corpus.sha256" \
      "$E/dm/corpus" "$E/t2d/corpus" experiments/20261113-e1-colab-dryrun/results/twin.jsonl
    mv -f "$E/bundle.tar.gz.tmp" "$E/bundle.tar.gz"
    sha "$E/bundle.tar.gz"; ls -la "$E/bundle.tar.gz" ;;
  *) echo "usage: $0 pack" >&2; exit 2 ;;
esac

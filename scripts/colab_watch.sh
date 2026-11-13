#!/usr/bin/env bash
# Liveness-aware monitor for a long run on a Colab VM (google-colab-cli). Run it in the background; it exits (so the
# caller is notified) on the first of:
#   FINISHED : the done marker appears in the done file
#   DEAD     : no process matches the pattern and the marker is absent (the run died or never started)
#   STALL    : the line count of the progress file did not change for STALL_CHECKS checks while alive
#   UNREACHABLE : MISS_CHECKS consecutive checks could not reach the VM (a lost connection is never read as "fine")
# Lesson (2026-11-12): a chain killed itself with `pkill -f "ollama serve"` and a done-only monitor let the VM idle
# for 2 hours. Stop processes with `pkill -x <exe>` or by PID, and check each chained stage right after it starts.
#
# Usage: scripts/colab_watch.sh <session> <process-pattern> <progress-file> <done-file> <done-marker> [interval-s]
#   e.g. scripts/colab_watch.sh hts-newtype 'python3 scripts/llm_dry_run_attacker' \
#        /content/hts/attk/results/attack.jsonl /content/hts/chain.log 'attacker done' 300
set -uo pipefail
S=$1 PAT=$2 PROG=$3 DONEF=$4 MARK=$5 INT=${6:-300}
STALL_CHECKS=${STALL_CHECKS:-3} MISS_CHECKS=${MISS_CHECKS:-3}
export MSYS_NO_PATHCONV=1
probe=$(mktemp --suffix=.py)
command -v cygpath >/dev/null && probe=$(cygpath -m "$probe")  # Git Bash: colab.exe needs a Windows path (MSYS_NO_PATHCONV=1)
cat > "$probe" <<EOF
import os, subprocess
done = os.path.exists("$DONEF") and "$MARK" in open("$DONEF").read()
alive = bool(subprocess.run(["pgrep", "-f", "$PAT"], capture_output=True, text=True).stdout.strip())
n = sum(1 for _ in open("$PROG")) if os.path.exists("$PROG") else 0
print("STATE=%s N=%d" % ("FINISHED" if done else ("ALIVE" if alive else "DEAD"), n))
EOF
last=-1; same=0; miss=0
while true; do
  out=$(timeout 90 colab exec -s "$S" -f "$probe" 2>&1 | grep -E '^STATE=')
  if [ -z "$out" ]; then
    miss=$((miss + 1)); [ "$miss" -ge "$MISS_CHECKS" ] && { echo "UNREACHABLE x$miss $(date -u +%T)"; exit 0; }
    sleep 120; continue
  fi
  miss=0; echo "$(date -u +%T) $out"
  case "$out" in *FINISHED*) echo "DONE"; exit 0 ;; *DEAD*) echo "ALERT: process gone before the done marker"; exit 0 ;; esac
  n=${out##*N=}; if [ "$n" = "$last" ]; then same=$((same + 1)); else same=0; fi; last=$n
  [ "$same" -ge "$STALL_CHECKS" ] && { echo "ALERT: no progress for $same checks"; exit 0; }
  sleep "$INT"
done

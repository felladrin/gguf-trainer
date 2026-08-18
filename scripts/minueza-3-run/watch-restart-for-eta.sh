#!/usr/bin/env bash
# One-shot watcher: wait for the training to write its NEXT checkpoint line to
# phaseA.log, then restart the run from that fresh checkpoint so the updated
# code (live ETA in every log line) takes effect with minimal lost progress.
# Restarting right after a checkpoint means resume-phase-a.sh resumes from it,
# so only the handful of steps since that checkpoint are re-done. The watcher
# self-terminates after issuing the restart.
#
# Launch detached (survives ssh disconnect):
#   setsid bash scripts/minueza-3-run/watch-restart-for-eta.sh >>examples/watcher.log 2>&1 </dev/null &
set -euo pipefail
# cd to the repo root (two levels up), so it works from any clone location.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG=examples/phaseA.log

lastCkpt() { grep -oE '\[ckpt @ [0-9]+\]' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | tail -1; }
BASE=$(lastCkpt || true)
echo "$(date -Is) watcher: baseline ckpt=${BASE:-none}; waiting for the next checkpoint to restart for ETA"

while :; do
  sleep 30
  if ! pgrep -f "pretrain[.]ts" >/dev/null; then
    echo "$(date -Is) watcher: pretrain process gone before a new checkpoint; aborting WITHOUT restart (investigate the run)"
    exit 1
  fi
  CUR=$(lastCkpt || true)
  if [ -n "${CUR:-}" ] && [ "${CUR:-}" != "${BASE:-}" ]; then
    echo "$(date -Is) watcher: new checkpoint @ $CUR detected; stopping training to restart for ETA"
    pkill -f "pretrain[.]ts" || true
    for _ in $(seq 1 60); do pgrep -f "pretrain[.]ts" >/dev/null || break; sleep 1; done
    if pgrep -f "pretrain[.]ts" >/dev/null; then
      echo "$(date -Is) watcher: process did not exit after 60s; aborting to avoid a double run"
      exit 1
    fi
    bash scripts/minueza-3-run/resume-phase-a.sh
    echo "$(date -Is) watcher: restart issued from ckpt @ $CUR; ETA now displays. exiting."
    exit 0
  fi
done

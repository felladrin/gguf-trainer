#!/usr/bin/env bash
# Stop the Phase B run. The trainer has no save-on-signal handler, so SIGTERM
# just terminates it: only the steps since the last "[ckpt @ N]" are lost, and
# those get redone on resume (at --ckpt=310 that is at most ~93 min, and ~46 min
# on average). Escalates to SIGKILL if needed, and leaves the HellaSwag curve
# watcher alone, it finishes scoring the last checkpoint and then exits by
# itself; resume-phase-b.sh brings it back.
# Counterpart to resume-phase-b.sh.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=examples/phaseB-fineweb.log
PATTERN='pretrain[.]ts'

if ! pgrep -f "$PATTERN" >/dev/null; then
  echo "No pretrain process running (nothing to stop)."
  exit 0
fi

echo "Stopping pretrain process(es): $(pgrep -f "$PATTERN" | tr '\n' ' ')"
pkill -TERM -f "$PATTERN" || true

for _ in $(seq 1 15); do
  pgrep -f "$PATTERN" >/dev/null || break
  sleep 1
done

if pgrep -f "$PATTERN" >/dev/null; then
  echo "Still alive after SIGTERM; sending SIGKILL."
  pkill -KILL -f "$PATTERN" || true
  sleep 1
fi

if pgrep -f "$PATTERN" >/dev/null; then
  echo "WARNING: process still present after SIGKILL:"
  pgrep -af "$PATTERN"
  exit 1
fi

echo "Stopped."
STEP=$(grep -oE '\[ckpt @ [0-9]+\]' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | tail -1 || true)
if [ -n "${STEP:-}" ]; then
  echo "Last saved checkpoint: step $STEP of 31000 ($((STEP * 16384 / 1000000))M tokens), examples/phaseB-fineweb.gguf + .optstate sidecar."
  SKEW=$(($(stat -c %Y examples/phaseB-fineweb.gguf 2>/dev/null || echo 0) - $(stat -c %Y examples/phaseB-fineweb.gguf.optstate 2>/dev/null || echo 0)))
  [ "$SKEW" -gt 600 ] && echo "NOTE: the GGUF is ${SKEW}s newer than its sidecar; the stop landed mid-checkpoint. resume-phase-b.sh will refuse without FORCE=1."
else
  echo "No checkpoint written yet: a resume restarts the cycle from examples/phaseA-final-88000.gguf."
fi
echo "Resume with:  bash scripts/minueza-3-run/resume-phase-b.sh"

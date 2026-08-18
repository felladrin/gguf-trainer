#!/usr/bin/env bash
# Stop the Phase A pretraining run. Sends SIGTERM first (the trainer has no
# save-on-signal handler, so this just terminates it; only steps run since the
# last "[ckpt @ N]" checkpoint are lost, and those get redone on resume). If the
# process is still alive after a grace period, escalates to SIGKILL. Reports the
# last saved checkpoint step so you know the resume point. Counterpart to
# resume-phase-a.sh.
set -euo pipefail
# cd to the repo root (two levels up), so it works from any clone location.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=examples/phaseA.log
PATTERN='pretrain[.]ts'

if ! pgrep -f "$PATTERN" >/dev/null; then
  echo "No pretrain process running (nothing to stop)."
  exit 0
fi

PIDS=$(pgrep -f "$PATTERN" | tr '\n' ' ')
echo "Stopping pretrain process(es): $PIDS"
pkill -TERM -f "$PATTERN" || true

# Give it up to ~15s to exit cleanly, then force-kill anything left.
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
  echo "Last saved checkpoint: step $STEP (examples/pretrain-blend-base.gguf + .optstate sidecar)."
fi
echo "Resume with:  bash scripts/minueza-3-run/resume-phase-a.sh"

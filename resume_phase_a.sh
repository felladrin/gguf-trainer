#!/usr/bin/env bash
# Resume the Phase A pretraining run (94.7M Gemma3 base, 8192 ctx) from its last
# checkpoint. Safe to run after the previous run was stopped (pkill). Auto-detects
# the resume step from the last "[ckpt @ N]" line in the log, so it is always
# correct no matter when you run it. Launches detached (setsid) so it survives
# an ssh disconnect. Config MUST match the original launch exactly or the loader's
# config guard aborts.
set -euo pipefail
# Repo root = this script's own directory, so the run works from any clone location.
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DENO=/home/victor/.deno/bin/deno
CKPT=examples/pretrain-blend-base.gguf
LOG=examples/phaseA.log

if pgrep -f "pretrain[.]ts" >/dev/null; then
  echo "A pretrain process is already running (pgrep matched). Stop it first:"
  echo "    pkill -f 'pretrain[.]ts'"
  exit 1
fi
if [ ! -f "$CKPT" ]; then echo "no checkpoint at $CKPT"; exit 1; fi

STEP=$(grep -oE '\[ckpt @ [0-9]+\]' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | tail -1 || true)
if [ -z "${STEP:-}" ]; then echo "no '[ckpt @ N]' line in $LOG; cannot determine resume step"; exit 1; fi

# Resume from a copy so the read source can't be touched by the run's own
# checkpoint writes, and the resume point is preserved.
SRC="examples/resume-from-$STEP.gguf"
cp -f "$CKPT" "$SRC"
[ -f "$CKPT.optstate" ] && cp -f "$CKPT.optstate" "$SRC.optstate" || true

echo "resuming Phase A from step $STEP (source $SRC) -> $CKPT"
setsid "$DENO" run -A --unstable-webgpu examples/pretrain.ts \
  examples/blend.tokens 640 12 88000 2048 8 0.01 \
  --maxSeq=8192 --ckpt=500 --quant=f32 --reclaim \
  --out="$CKPT" --name=gemma3-96m-base \
  --resume="$SRC" --startStep="$STEP" \
  >>"$LOG" 2>&1 </dev/null &

echo "launched (detached). Tail progress with:  tail -f $LOG"
echo "Optimizer state (Muon momentum + Adam moments) resumes warm from the .optstate sidecar."

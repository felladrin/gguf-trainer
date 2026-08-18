#!/usr/bin/env bash
# Resume the Phase B run (FineWeb-Edu continual pre-training, one 508.7M-token
# epoch from the Phase A final checkpoint) after stop-phase-b.sh or a crash.
# Auto-detects the resume step from the last "[ckpt @ N]" line in the log, so it
# is correct whenever you run it. Launches detached (setsid) so it survives an
# ssh disconnect, and brings the HellaSwag curve watcher back up alongside it.
# Counterpart to stop-phase-b.sh.
#
#   bash scripts/minueza-3-run/resume-phase-b.sh          # resume
#   DRY_RUN=1 bash scripts/minueza-3-run/resume-phase-b.sh  # print what it would launch, change nothing
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

DENO="${DENO:-deno}"
CKPT=examples/phaseB-fineweb.gguf
LOG=examples/phaseB-fineweb.log
BASE=examples/phaseA-final-88000.gguf
# STEPS must stay 31000 on every resume: warmup/cooldown are 10%/20% of it, so a
# different total silently reshapes the LR schedule mid-run (see
# docs/CONTINUAL_PRETRAINING.md, "Schedule gotcha").
STEPS=31000

# DRY_RUN changes nothing, so it stays usable while the run is up.
if [ -z "${DRY_RUN:-}" ] && pgrep -f "pretrain[.]ts" >/dev/null; then
  echo "A pretrain process is already running. Stop it first:  bash scripts/minueza-3-run/stop-phase-b.sh"
  exit 1
fi

# First launch resumes Phase A's weights at startStep=0 (fresh WSD cycle); every
# later resume continues this run's own checkpoint at its saved step.
STEP=$(grep -oE '\[ckpt @ [0-9]+\]' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | tail -1 || true)

# The trainer renames the GGUF, then the optstate, then prints "[ckpt @ N]", so a
# kill between the two renames leaves weights AHEAD of both the sidecar and the
# log. Resuming at the logged step would then redo work from weights already past
# it, with the LR schedule an interval behind. A normal pair differs only by the
# optstate write (~2 min); an interrupted one differs by a whole interval.
if [ -n "${STEP:-}" ] && [ -f "$CKPT" ] && [ -f "$CKPT.optstate" ]; then
  SKEW=$(($(stat -c %Y "$CKPT") - $(stat -c %Y "$CKPT.optstate")))
  if [ "$SKEW" -gt 600 ]; then
    echo "WARNING: $CKPT is ${SKEW}s newer than its .optstate sidecar."
    echo "  The last checkpoint was interrupted between the two writes: the weights"
    echo "  on disk are ahead of step $STEP, and resuming there redoes that work."
    echo "  Re-run with FORCE=1 to resume anyway."
    [ -z "${FORCE:-}" ] && exit 1
  fi
fi

if [ -n "${STEP:-}" ] && [ -f "$CKPT" ]; then
  SRC="examples/phaseB-resume-from-$STEP.gguf"
  START="$STEP"
else
  SRC="$BASE"
  START=0
  echo "no checkpoint of this run yet; starting the cycle from $BASE at step 0"
fi

if [ ! -f "$SRC" ] && [ "$START" = 0 ]; then echo "missing $SRC"; exit 1; fi

# --ckpt=310 (~93 min), not the 3100 the run started with: a checkpoint write
# measured ~4.1 min, so 310 costs 4.4% throughput and caps what an OOM kill can
# destroy at ~46 min on average, against ~7.7 h at 3100. The curve script only
# scores multiples of CURVE_EVERY, so the extra checkpoints cost no eval time.
CMD=("$DENO" run -A --unstable-webgpu examples/pretrain.ts
  examples/fineweb.tokens 640 12 "$STEPS" 2048 8 0.01
  --auxLr=3e-3 --maxSeq=8192 --window=1024 --headDim=64 --ckpt=310 --quant=f32
  --resume="$SRC" --startStep="$START"
  --out="$CKPT" --name=gemma3-96m-base)

if [ -n "${DRY_RUN:-}" ]; then
  echo "would resume from step $START (source $SRC)"
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

# Resume from a copy so this run's own checkpoint writes can't touch the file it
# is reading, and the resume point stays on disk.
if [ "$START" != 0 ]; then
  cp -f "$CKPT" "$SRC"
  [ -f "$CKPT.optstate" ] && cp -f "$CKPT.optstate" "$SRC.optstate" || true
fi

echo "resuming Phase B from step $START (source $SRC) -> $CKPT"
setsid "${CMD[@]}" >>"$LOG" 2>&1 </dev/null &

if pgrep -f "hellaswag_curve[.]sh" >/dev/null; then
  echo "curve watcher already running"
else
  # It exits on its own once the trainer is gone, so a stop/resume cycle needs it
  # relaunched. It skips steps already in the CSV, so no duplicate rows.
  setsid bash scripts/minueza-3-run/hellaswag-curve.sh "$LOG" "$CKPT" \
    >>examples/phaseB-curve.log 2>&1 </dev/null &
  echo "curve watcher relaunched (log: examples/phaseB-curve.log)"
fi

echo "launched (detached). Tail progress with:  tail -f $LOG"
echo "Optimizer state (Muon momentum + Adam moments) resumes warm from the .optstate sidecar."

#!/usr/bin/env bash
# Measure HellaSwag at every training checkpoint, to get the points-per-token
# slope of this recipe instead of guessing it.
#
#   bash scripts/minueza-3-run/hellaswag-curve.sh <training.log> <checkpoint.gguf> [out.csv]
#
# The question this answers: Supra2-100M-Base scores 35.31 on this harness with
# 30B training tokens; our base scores 28.46 with 1.44B. Closing 6.85 points by
# brute force is ~a year on one APU, so the decision needs the actual slope,
# how many points each 100M tokens of FineWeb-Edu buys: measured on a few
# points and extrapolated, not assumed.
#
# Runs on CPU (-dev none, few threads) so it steals as little as possible from
# the training run on the GPU. The full 10042-task set is used, not a sample:
# a sample's +/-1.05 would swamp the ~0.25-point steps we are trying to resolve.
set -uo pipefail
cd "$(dirname "$0")/../.."

LOG="${1:?usage: hellaswag-curve.sh <training.log> <checkpoint.gguf> [out.csv]}"
CKPT="${2:?missing checkpoint path}"
CSV="${3:-evaldata/hellaswag_curve.csv}"
BIN="${LLAMA_PERPLEXITY:-llama-perplexity}"
HS=evaldata/hellaswag-validation.bin
TOKENS_PER_STEP=16384 # seqLen 2048 x batch 8
POLL=120
# The run checkpoints every 310 steps for crash safety, which is 10x more often
# than the curve wants a point. Score only multiples of this, so the extra
# checkpoints cost no CPU and the curve keeps its ~50M-token spacing.
CURVE_EVERY=${CURVE_EVERY:-3100}

[ -e "$HS" ] || { echo "missing $HS" >&2; exit 1; }
[ -e "$CSV" ] || echo "step,tokens,hellaswag,stderr,when" > "$CSV"

echo "watching $LOG for checkpoints of $CKPT -> $CSV"
# Seed from the CSV, not from empty: a relaunch after a stop/resume cycle would
# otherwise score the last checkpoint again and append a duplicate row.
last=$(tail -n +2 "$CSV" | tail -1 | cut -d, -f1)
while true; do
  # The trainer prints "[ckpt @ N]" only AFTER the file is fully written, so the
  # log line, not the file's mtime, is the safe trigger.
  step=$(grep -oP '\[ckpt @ \K[0-9]+' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$step" ] && [ "$step" != "$last" ] && [ -e "$CKPT" ]; then
    if [ $((step % CURVE_EVERY)) -eq 0 ]; then
      cp "$CKPT" "$CKPT.curve" # the next checkpoint overwrites the original
      res=$("$BIN" -m "$CKPT.curve" --multiple-choice -bf "$HS" -dev none -np 8 -t 4 \
        --no-warmup 2>&1 | grep -oP 'Final result: \K[0-9.]+ \+/- [0-9.]+')
      rm -f "$CKPT.curve"
      if [ -n "$res" ]; then
        acc=${res%% *}
        err=${res##*+/- }
        printf '%s,%s,%s,%s,%s\n' \
          "$step" "$((step * TOKENS_PER_STEP))" "$acc" "$err" "$(date -Iseconds)" >> "$CSV"
        echo "step $step ($((step * TOKENS_PER_STEP / 1000000))M tokens): HellaSwag $res"
      else
        echo "step $step: eval FAILED" >&2
      fi
    else
      echo "step $step: safety checkpoint, not a curve point (every $CURVE_EVERY)"
    fi
    last="$step"
  fi
  # Stop once the trainer is gone AND its last checkpoint has been scored.
  if ! pgrep -f "pretrain.ts corpus/fineweb-edu" > /dev/null 2>&1 &&
    ! pgrep -f "pretrain.ts examples/fineweb" > /dev/null 2>&1; then
    echo "trainer gone; curve complete"
    exit 0
  fi
  sleep "$POLL"
done

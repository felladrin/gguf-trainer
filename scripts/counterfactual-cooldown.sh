#!/usr/bin/env bash
# Does the WSD cooldown itself raise held-out loss, or was the rise just where a
# random walk happened to be when the schedule got there?
#
# Two arms from the SAME checkpoint. `mulberry32(7 + startStep)` seeds the window
# sampler, so resuming both at 1767 gives them an identical data stream and the
# schedule is the only thing that differs:
#
#   A  --steps 2200  the real cooldown, lr scale 0.9836 -> 0.1000 across the segment
#   B  --steps 2760  flat at 1.0, since its own cooldown would not begin until
#                    step 2208, which is past where this stops
#
# --steps sets the schedule shape AND the segment length (steps - start-step), so
# a flat arm can only be built by choosing a total whose cooldown falls past the
# stop and then stopping early. There is no stop-at flag, hence the kill below.
#
# Checkpoints fire on the step LOCAL to the segment, so both arms snapshot at the
# same local steps and compare at matched global ones: 1817, 1867 ... 2167. Do not
# wait on a global step that is not a multiple of the cadence; it never arrives.
#
# Both arms cold-start the optimizer, because no optstate was kept at 1767. That
# is symmetric, so it does not bias A against B, but it does mean arm A is not a
# byte reproduction of the original tail.
set -euo pipefail
cd "$(dirname "$0")/.."

STOP_AT=2167 # local step 400, the last snapshot both arms share

common=(
  --data data/lambrp-train.tokens --mask data/lambrp-train.mask
  --template data/lambrp-train.template.txt --resume out/lamb-ckpt-1767.gguf
  --arch qwen3 --hidden 544 --layers 28 --head-dim 128 --heads 16 --kv-heads 8
  --ffn-dim 2560 --max-seq 40960 --seq-len 2048 --batch 2
  --lr 0.00005 --aux-lr 0.00002 --start-step 1767
  --reclaim --keep-checkpoints --checkpoint-every 50
)

echo "=== arm A: cooldown, 433 steps to 2200 ==="
deno run -A cli.ts finetune "${common[@]}" --steps 2200 \
  --out out/cf-cooldown.gguf --name LittleLamb-cf-cooldown >out/cf-cooldown.log 2>&1
echo "arm A done"

echo "=== arm B: flat LR, stopping at global $STOP_AT ==="
deno run -A cli.ts finetune "${common[@]}" --steps 2760 \
  --out out/cf-flat.gguf --name LittleLamb-cf-flat >out/cf-flat.log 2>&1 &
trainer=$!
until [ -f "out/cf-flat-step$STOP_AT.gguf" ] || ! kill -0 "$trainer" 2>/dev/null; do sleep 60; done
if kill -0 "$trainer" 2>/dev/null; then
  # The snapshot is on disk; the extra steps toward 2760 are not wanted.
  kill "$trainer"
  wait "$trainer" 2>/dev/null || true
  echo "arm B stopped at $STOP_AT"
else
  echo "arm B exited on its own; check out/cf-flat.log" >&2
fi
echo "=== both arms done ==="

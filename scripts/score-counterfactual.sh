#!/usr/bin/env bash
# Score both counterfactual arms on the held-out file, at the globals they share.
#
# Same knobs as every other holdout number in docs/measurements: 64 windows of
# 1024 tokens, seed 1234, --holdout 1 so the whole file is the sample. Change any
# of them and the result stops being comparable to the run it is a counterfactual for.
#
# Appends as it goes, so a run interrupted halfway still leaves its numbers behind.
set -euo pipefail
cd "$(dirname "$0")/.."

log=out/cf-holdout.log
: >"$log"

for step in 1817 1867 1917 1967 2017 2067 2117 2167; do
  for arm in cooldown flat; do
    m="out/cf-$arm-step$step.gguf"
    [ -f "$m" ] || { echo "missing $m" >&2; continue; }
    echo "########## $m" >>"$log"
    deno run -A cli.ts eval-loss --model "$m" --data data/lambrp-hold.tokens \
      --holdout 1 --windows 64 --seq-len 1024 --seed 1234 2>&1 | tail -1 >>"$log"
  done
done
echo "=== done ===" >>"$log"

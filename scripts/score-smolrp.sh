#!/usr/bin/env bash
# Rank every snapshot of the SmolLM2-135M-Heretic-RP run on the held-out slice.
#
#   bash scripts/score-smolrp.sh
#
# The base goes in the same table on the same knobs, because the number that
# matters is the delta from it, not the absolute loss. Every row uses the same
# windows, sequence length and seed, so the comparison is paired and the right
# error scale is checkpoint-to-checkpoint jitter rather than the standard error of
# a 64-window estimate.
#
# Score the WHOLE series and rank it. Do not fit a slope through it and do not read
# a turn at a phase boundary as evidence about the phase: both were tried on the
# LittleLamb run and both were wrong (docs/optimization.md levers 16 and 16b).
#
# Appends as it goes, so a killed run still leaves what it measured.
set -euo pipefail
cd "$(dirname "$0")/.."

log="${LOG:-out/smolrp-holdout.log}"
hold="${HOLD:-data/rpx-hold.tokens}"
: >"$log"

score() {
  [ -f "$1" ] || return 0
  echo "########## $1" >>"$log"
  deno run -A cli.ts eval-loss --model "$1" --data "$hold" \
    --holdout 1 --windows 64 --seq-len 1024 --seed 1234 2>&1 | tail -1 >>"$log"
}

score out/smollm2-heretic.f32.gguf
# Sorted numerically by step, so the log reads in run order rather than in shell
# glob order, where step 1000 sorts before step 200.
for m in $(ls out/smolrp-step*.gguf 2>/dev/null | sort -t p -k3 -n); do score "$m"; done
score out/smolrp.gguf

echo "=== done ===" >>"$log"
grep -E "^#|val loss" "$log"

#!/usr/bin/env bash
# Curriculum stage 2 end to end: seed corpus -> restyle in a target voice ->
# tokenize -> SFT from the Phase B base. Each step is skipped when its output is
# already there, so a re-run continues where the last one stopped (the restyle
# itself is resumable per conversation).
#
#   bash scripts/minueza-3-run/run-style-sft.sh        # prototype-sized run (whatever seed exists)
#   TOTAL=2000 STEPS=1200 bash scripts/minueza-3-run/run-style-sft.sh
#   DRY_RUN=1 bash scripts/minueza-3-run/run-style-sft.sh   # print the commands, change nothing
#
# The steps run in sequence on purpose: the restyle pass keeps a 36 GB coder
# model resident on the same GPU the trainer needs.
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

DENO="${DENO:-deno}"
TOTAL="${TOTAL:-200}"           # conversations to seed
EPOCHS="${EPOCHS:-3}"           # STEPS is derived from this and the corpus size
SEQ="${SEQ:-1024}"
BATCH="${BATCH:-8}"
LR="${LR:-0.001}"               # 10x below the pretrain peak: SFT nudges, not reshapes
BASE=examples/phaseB-fineweb.gguf
SEED=corpus/style/seed-${TOTAL}.jsonl
STYLED=corpus/style/restyled-${TOTAL}.jsonl
PREFIX=examples/style-${TOTAL}
OUT=examples/sft-style-${TOTAL}.gguf
LOG=examples/sft-style-${TOTAL}.log

run() {
  echo "+ $*"
  [ -n "${DRY_RUN:-}" ] || "$@"
}

[ -f "$BASE" ] || { echo "missing base checkpoint $BASE"; exit 1; }

if [ ! -f "$SEED" ]; then
  run "$DENO" run -A examples/build_style-seed.ts --total="$TOTAL" --out="$SEED"
fi

# Resumable and slow (~16s/conversation): the whole point of the detached run.
run "$DENO" run -A examples/restyle_with_pi.ts --in="$SEED" --out="$STYLED"

run "$DENO" run -A examples/prepare_instruct.ts 1000000 "$PREFIX" "$STYLED" \
  examples/blend.tokenizer.json

# Steps from the corpus itself: an SFT set this small is measured in epochs, not
# in steps, and the token count is not known until the corpus is built (u16).
if [ -z "${STEPS:-}" ] && [ -n "${DRY_RUN:-}" ]; then
  STEPS=0 # really: computed from $PREFIX.tokens, which a dry run has not built
elif [ -z "${STEPS:-}" ]; then
  TOKENS=$(($(stat -c %s "$PREFIX.tokens") / 2))
  STEPS=$(((EPOCHS * TOKENS + SEQ * BATCH - 1) / (SEQ * BATCH)))
  echo "corpus $TOKENS tokens -> $STEPS steps for $EPOCHS epochs at ${BATCH}x${SEQ}"
fi

# --coldOpt: the Phase B optimizer sidecar is momentum for a different objective.
# --mask: assistant-only loss. --template: the chat template goes into the GGUF.
run "$DENO" run -A --unstable-webgpu examples/pretrain.ts "$PREFIX.tokens" \
  640 12 "$STEPS" "$SEQ" "$BATCH" "$LR" \
  --resume="$BASE" --coldOpt --mask="$PREFIX.mask" --template="$PREFIX.template.txt" \
  --ckpt=$((STEPS / 4 > 1 ? STEPS / 4 : 1)) --out="$OUT" --name="minueza-3-style" \
  --exportQuants=q8_0 \
  2>&1 | tee "$LOG"

echo "done: $OUT (log $LOG)"

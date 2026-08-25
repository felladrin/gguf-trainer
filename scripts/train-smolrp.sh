#!/usr/bin/env bash
# The SmolLM2-135M-Heretic-RP run, as it was actually invoked.
#
#   bash scripts/train-smolrp.sh
#
# Corpus: scripts/build-rp-chats.ts -> chat-corpus (see docs/optimization.md).
#
# WHY these knobs:
#
#   --seq-len 1024   measured faster per token than 2048 on this box (369 vs 292
#                    tok/s at the same tokens per step), because attention is
#                    O(T^2) per layer and --batch is gradient accumulation, so a
#                    longer sequence does not buy a bigger GEMM. 1024 also matches
#                    the corpus: 28.2M tokens over 26,338 documents is 1072 each.
#   --lr 0.0001      between LittleLamb's proven 0.00005 over 2200 steps and the
#                    "tenth of pretraining" rule (0.001), which overwrites a base
#                    someone else spent 2T tokens training. Half the steps at twice
#                    the rate is roughly the same total movement.
#   no --reclaim     the GPU is free for this run; --reclaim costs 23% throughput
#                    and buys peak memory this shape does not need.
#   --keep-checkpoints  the run leaves a series to rank on held-out loss. Never
#                    pick on training loss (docs/optimization.md lever 12), and
#                    score the WHOLE series rather than trusting a slope through
#                    it (lever 16b).
#
# The architecture flags are `inspect --model out/smollm2-heretic.f32.gguf` verbatim.
# They are not optional: without them the config is built from defaults and the
# resume aborts.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-out/smollm2-heretic.f32.gguf}"
OUT="${OUT:-out/smolrp.gguf}"
STEPS="${STEPS:-1300}"

deno run -A cli.ts finetune \
  --data data/rpx.tokens --mask data/rpx.mask --template data/rpx.template.txt \
  --resume "$BASE" --out "$OUT" \
  --arch llama --hidden 576 --layers 30 --head-dim 64 --heads 9 --kv-heads 3 \
  --ffn-dim 1536 --max-seq 8192 \
  --steps "$STEPS" --seq-len 1024 --batch 8 \
  --lr 0.0001 --aux-lr 0.00003 \
  --keep-checkpoints --checkpoint-every 100 \
  --name SmolLM2-135M-Heretic-RP

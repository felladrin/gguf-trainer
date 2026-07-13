#!/usr/bin/env bash
# Qualitative completion probe for a pretraining checkpoint. Runs a fixed battery
# of prompts through llama.cpp's llama-completion so successive checkpoints can be
# eyeballed for progress (fluency, coherence, factual recall) over the course of
# a long run.
#
# WHY the sampling flags below: a mid-training BASE model collapses into
# repetition loops ("fresh fresh fresh...") under naive/greedy sampling, which
# reads far worse than the model actually is. --repeat-penalty + --min-p suppress
# the loops so you judge the model, not the sampler. Keep the prompts and flags
# fixed so runs at different steps stay comparable.
#
# Usage:
#   examples/eval_completions.sh [MODEL.gguf]      # or:  deno task eval:completions [MODEL.gguf]
#     MODEL.gguf   checkpoint to probe (default: examples/pretrain-blend-base.gguf)
#
# Pull the latest checkpoint off the training box first, if it isn't local:
#   rsync -av gpu-server:gguf-trainer/examples/pretrain-blend-base.gguf examples/
#
# Requires llama-completion (llama.cpp) on PATH; override with
#   LLAMA_COMPLETION=/path/to/llama-completion examples/eval_completions.sh ...
# Tunables via env: N_PREDICT (tokens, default 60), SEED (default 42).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-examples/pretrain-blend-base.gguf}"
BIN="${LLAMA_COMPLETION:-llama-completion}"
N="${N_PREDICT:-60}"
SEED="${SEED:-42}"

[ -f "$MODEL" ] || {
  echo "no checkpoint at '$MODEL' — pass a path, or rsync one first (see header)."
  exit 1
}
command -v "$BIN" >/dev/null 2>&1 || {
  echo "'$BIN' not found on PATH — install llama.cpp or set LLAMA_COMPLETION."
  exit 1
}

# Fixed battery: narrative, descriptive, everyday, factual recall, science,
# abstraction/list, dialogue, arithmetic — a spread that exposes fluency,
# long-range coherence, and (weak, at this size) world knowledge.
PROMPTS=(
  "Once upon a time, there was a little"
  "The old man walked slowly toward the"
  "In the morning, she went to the store to buy"
  "The sun rose over the mountains and"
  "The capital of France is"
  "Water is made of"
  "The three most important things in life are"
  "Scientists have recently discovered that"
  '"Hello," said Tom, "I want to'
  "2 + 2 ="
)

echo "model:    $MODEL"
echo "sampling: temp 0.8, min-p 0.05, repeat-penalty 1.3 (last 64), seed $SEED, n $N"
echo

for p in "${PROMPTS[@]}"; do
  echo "########## [$p]"
  "$BIN" -m "$MODEL" -p "$p" -n "$N" -c 1024 \
    --temp 0.8 --min-p 0.05 --repeat-penalty 1.3 --repeat-last-n 64 \
    --seed "$SEED" --no-perf 2>/dev/null
  echo
  echo
done

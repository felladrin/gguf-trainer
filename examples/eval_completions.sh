#!/usr/bin/env bash
# Qualitative completion probe for a pretraining checkpoint. Runs a fixed battery
# of prompts through llama.cpp's llama-completion so successive checkpoints can be
# eyeballed for progress (fluency, coherence, factual recall) over the course of
# a long run.
#
# WHY the sampling preset below: a mid-training BASE model collapses into
# repetition loops ("fresh fresh fresh...") under naive/greedy sampling, which
# reads far worse than the model actually is. The default preset "D" — a gentle
# repeat-penalty plus a light presence-penalty — kills the loops WITHOUT the
# topic drift that a heavy penalty (or a strong presence-penalty alone) causes,
# so you judge the model, not the sampler. Keep the prompts and preset fixed so
# runs at different steps stay comparable. An alternative preset is provided
# commented-out below.
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

# Sampling preset "D": best coherence / anti-loop balance found at ~24% training
# (gentle repeat-penalty + light presence-penalty; see git log for the sweep).
SAMPLER=(--temp 0.7 --top-p 0.85 --top-k 30 --min-p 0.02 --presence-penalty 0.4 --repeat-penalty 1.15 --repeat-last-n 128)
# Alternative preset "U": maximum lexical variety via a strong presence-penalty,
# but noticeably more topic drift. Swap by commenting D and uncommenting this.
# SAMPLER=(--temp 0.7 --top-p 0.80 --top-k 20 --min-p 0.0 --presence-penalty 1.5 --repeat-penalty 1.0 --repeat-last-n 64)

echo "model:    $MODEL"
echo "sampling: ${SAMPLER[*]}"
echo "seed $SEED, n $N tokens"
echo

for p in "${PROMPTS[@]}"; do
  echo "########## [$p]"
  "$BIN" -m "$MODEL" -p "$p" -n "$N" -c 1024 "${SAMPLER[@]}" \
    --seed "$SEED" --no-perf </dev/null 2>/dev/null
  echo
  echo
done

#!/usr/bin/env bash
# Qualitative completion probe for a base checkpoint: a fixed battery of prompts
# through llama.cpp, so successive checkpoints of one run can be compared by eye.
#
#   bash scripts/eval-completions.sh <model.gguf>
#
# WHY the sampling preset below: a mid-training BASE model collapses into
# repetition loops ("fresh fresh fresh...") under greedy or naive sampling, which
# reads far worse than the model actually is. A gentle repeat-penalty kills the
# loops WITHOUT the topic drift a heavy penalty (or a strong presence-penalty)
# causes, so you judge the model and not the sampler. It is still deterministic:
# temp 0 plus repeat-penalty gives the same output every run.
#
# Keep the prompts and the preset FIXED across checkpoints, or the comparison
# measures the sampler. Override deliberately, not per checkpoint:
#   SAMPLER="--temp 0.7 --top-p 0.85 --top-k 30" bash scripts/eval-completions.sh m.gguf
#
# The CLI's own `generate` is the no-dependency version of this: one prompt,
# greedy, no sampler. Use that for a smoke test and this for a quality read.
#
# Requires llama-completion (llama.cpp) on PATH; override with
#   LLAMA_COMPLETION=/path/to/llama-completion
# Tunables: N_PREDICT (tokens, default 60), SEED (default 42).
set -euo pipefail

MODEL="${1:-}"
BIN="${LLAMA_COMPLETION:-llama-completion}"
N="${N_PREDICT:-60}"
SEED="${SEED:-42}"
# Deterministic (temp 0) with a repeat-penalty to keep a base model out of loops.
read -ra SAMPLER <<<"${SAMPLER:---temp 0 --repeat-penalty 1.15 --repeat-last-n 128}"

[ -n "$MODEL" ] || {
  echo "usage: bash scripts/eval-completions.sh <model.gguf>" >&2
  exit 2
}
[ -f "$MODEL" ] || {
  echo "no checkpoint at '$MODEL'" >&2
  exit 1
}
command -v "$BIN" >/dev/null 2>&1 || {
  echo "'$BIN' not found on PATH: install llama.cpp or set LLAMA_COMPLETION." >&2
  exit 1
}

# Narrative, descriptive, everyday, factual recall, science, abstraction,
# dialogue, arithmetic: a spread that exposes fluency, long-range coherence and
# (weak, at this size) world knowledge.
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
echo "sampling: ${SAMPLER[*]}"
echo "seed $SEED, n $N tokens"
echo

for p in "${PROMPTS[@]}"; do
  echo "########## [$p]"
  # -no-cnv: raw completion. A GGUF carrying a chat template otherwise enters
  # conversation mode, which is not what a base checkpoint should be judged on.
  "$BIN" -m "$MODEL" -p "$p" -n "$N" -c 1024 -no-cnv "${SAMPLER[@]}" \
    --seed "$SEED" --no-perf </dev/null 2>/dev/null
  echo
  echo
done

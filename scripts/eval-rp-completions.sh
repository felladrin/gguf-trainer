#!/usr/bin/env bash
# Qualitative probe for a roleplay-completion checkpoint: a fixed battery of
# horde-shaped prompts through llama.cpp, so successive checkpoints compare by eye.
#
#   bash scripts/eval-rp-completions.sh <model.gguf>
#
# This is eval-completions.sh's sibling for a different target. That one prompts
# with sentence fragments and judges narrative fluency; this one sends what an AI
# Horde scribe actually receives, a persona block plus a partial transcript ending
# on the character's name, and judges whether the model stays in character, keeps
# the "Name:" turn structure, and stops instead of writing the human's next line.
#
# -r "You:" is the stop sequence a real client sends, so a model that runs past
# its own turn is visible here as truncation at the human's name.
#
# Keep the prompts and the preset FIXED across checkpoints, or the comparison
# measures the sampler. Roleplay wants more variety than a base probe, so the
# default is warm rather than greedy; override deliberately, not per checkpoint:
#   SAMPLER="--temp 0 --repeat-penalty 1.15" bash scripts/eval-rp-completions.sh m.gguf
#
# Requires llama-completion (llama.cpp) on PATH; override with
#   LLAMA_COMPLETION=/path/to/llama-completion
# Tunables: N_PREDICT (tokens, default 80), SEED (default 42).
set -euo pipefail

MODEL="${1:-}"
BIN="${LLAMA_COMPLETION:-llama-completion}"
N="${N_PREDICT:-80}"
SEED="${SEED:-42}"
read -ra SAMPLER <<<"${SAMPLER:---temp 0.8 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1 --repeat-last-n 128}"

[ -n "$MODEL" ] || {
  echo "usage: bash scripts/eval-rp-completions.sh <model.gguf>" >&2
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

# Five shapes a horde request actually takes: a fresh chat off the greeting, a
# mid-conversation continuation, a persona with no example dialogue, an
# action-heavy scene, and a bare narrative continuation (the control, to check the
# fine-tune did not destroy plain prose).
PROMPTS=(
  "[Character: Iris]
Iris's Persona: A cheerful librarian who knows every book in the city archive and hates being interrupted during tea.
<START>
Iris: Oh! A visitor. Mind the stacks, they bite.
You: I'm looking for a book about the old harbor.
Iris:"

  "[Character: Captain Roeder]
Captain Roeder's Persona: A tired airship captain, twenty years in the trade, deeply superstitious about storms.
<START>
Captain Roeder: We cast off at dawn. Don't be late.
You: What's the cargo this time?
Captain Roeder: Crates. Sealed ones. I don't ask and neither should you.
You: You seem nervous.
Captain Roeder:"

  "[Character: Mira]
Mira's Persona: A quiet apprentice blacksmith, speaks in short sentences, uncomfortable with praise.
<START>
You: That sword you made is beautiful.
Mira:"

  "[Character: Thorn]
Thorn's Persona: A stray cat who somehow talks. Sarcastic. Believes the house belongs to him.
<START>
Thorn: *stretches across the doorway, blocking it entirely* You're late.
You: *steps over him* I was at work.
Thorn:"

  "The lighthouse had been dark for three weeks when Ana finally rowed out to it."
)

echo "model:    $MODEL"
echo "sampling: ${SAMPLER[*]}"
echo "seed $SEED, n $N tokens, stop at 'You:'"
echo

for p in "${PROMPTS[@]}"; do
  echo "########## $(echo "$p" | head -1)"
  # -no-cnv: raw completion, which is what a horde scribe serves.
  "$BIN" -m "$MODEL" -p "$p" -n "$N" -c 2048 -no-cnv -r "You:" "${SAMPLER[@]}" \
    --seed "$SEED" --no-perf </dev/null 2>/dev/null
  echo
  echo
done

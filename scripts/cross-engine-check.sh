#!/usr/bin/env bash
# Score one file with BOTH engines and print the two numbers side by side.
#
#   bash scripts/cross-engine-check.sh out/some-base.gguf
#
# Every test suite in this repo is a self-consistency check: GPU against CPU,
# analytic gradients against finite differences, export against re-import. A
# forward pass that disagrees with llama.cpp passes all of them, because llama.cpp
# is not one of the things they compare against. This is that missing comparison,
# and it is the first thing to run when a downloaded checkpoint's loss looks wrong.
#
# The probe is one sentence repeated 400 times. A model reading its own context
# scores near zero on it; a model that is locally fluent but positionally broken
# scores far above zero while still generating readable text, which is exactly the
# failure that hid the `llama` RoPE row order (docs/optimization.md lever 17).
#
# Requires llama-perplexity on PATH; override with LLAMA_PERPLEXITY=/path/to/it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: cross-engine-check.sh <model.gguf>}"
BIN="${LLAMA_PERPLEXITY:-llama-perplexity}"
command -v "$BIN" >/dev/null 2>&1 || {
  echo "'$BIN' not found on PATH: install llama.cpp or set LLAMA_PERPLEXITY." >&2
  exit 1
}

work="out/cross-engine"
mkdir -p "$work"
python3 -c "open('$work/repeat.txt','w').write('The cat sat on the mat. '*400)"

# The probe has to go through the checkpoint's OWN vocab, or the comparison
# measures the tokenizer instead of the forward pass.
deno run -A cli.ts inspect --model "$MODEL" --dump-tokenizer "$work/repeat.tokenizer.json" >/dev/null
deno run -A cli.ts tokenize --text "$work/repeat.txt" --out "$work/repeat" --vocab 49152 >/dev/null

echo "model: $MODEL"
echo
echo "-- this trainer --"
deno run -A cli.ts eval-loss --model "$MODEL" --data "$work/repeat.tokens" \
  --holdout 1 --windows 4 --seq-len 512 --seed 1 | tail -1
echo
echo "-- llama.cpp --"
"$BIN" -m "$MODEL" -f "$work/repeat.txt" -c 512 -ngl 99 2>&1 | grep "Final estimate"
echo
echo "Both should be near perplexity 1. A gap is a forward-pass disagreement,"
echo "not a corpus problem: see docs/optimization.md lever 17."

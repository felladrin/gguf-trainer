#!/usr/bin/env bash
# Head-to-head under llama.cpp's own harness: the open item from the Phase A
# eval gate. eval_mc.ts cannot score the reference models (loadGemma3FromGGUF
# rejects a non-gemma3 GGUF), so every model here is scored by llama-perplexity
# instead, one binary, one metric, same task order.
#
# CPU only (-dev none): the GPU is busy with the context-extension run, and
# these are small models. 6 threads of 32: at 12 the concurrent training run
# lost ~35% of its throughput to CPU contention.
#
#   bash scripts/minueza-3-run/head-to-head.sh [hellaswag-tasks]
#
# Validation sets are llama.cpp's canonical ones (ikawrakow/validation-datasets-
# for-llama.cpp), so the numbers are comparable to other llama.cpp reports,
# but NOT directly to lm-eval-harness acc_norm, which normalizes differently.
set -uo pipefail
cd "$(dirname "$0")/../.."

BIN="${LLAMA_PERPLEXITY:-llama-perplexity}"
ARC=evaldata/arc-challenge-validation.bin
HS=evaldata/hellaswag-validation.bin
HS_TASKS="${1:-2000}"
THREADS=6
OUT=evaldata/head2head.log

MODELS=(
  "ours-phaseA-94.7M:examples/phaseA-final-88000.gguf"
  "Minueza-32M-Base:examples/Minueza-32M-Base.F16.gguf"
  "Minueza-2-96M:examples/Minueza-2-96M.F16.gguf"
  "Qwen2-96M:examples/Qwen2-96M.f16.gguf"
  "SmolLM2-135M:examples/SmolLM2-135M.f16.gguf"
  "llama-160m:examples/llama-160m.fp16.gguf"
)

for f in "$BIN" "$ARC" "$HS"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

echo "=== head-to-head, llama-perplexity --multiple-choice, $(date -Iseconds) ===" | tee -a "$OUT"
echo "ARC-Challenge: all tasks in $ARC | HellaSwag: $HS_TASKS tasks" | tee -a "$OUT"

score() { # model_path data_file task_flag -> "NN.NNNN +/- N.NNNN" or "FAILED"
  local out
  # -np 8: llama.cpp decodes the answers in parallel and refuses a task with
  # more options than -np allows. ARC-Challenge has 5-option questions, and the
  # default 4 aborts the whole run at the first one.
  out=$("$BIN" -m "$1" --multiple-choice -bf "$2" $3 -dev none -np 8 -t "$THREADS" --no-warmup 2>&1)
  echo "$out" | grep -oP 'Final result: \K[0-9.]+ \+/- [0-9.]+' || echo "FAILED"
}

for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"
  path="${entry#*:}"
  if [ ! -e "$path" ]; then
    printf '%-20s MISSING %s\n' "$label" "$path" | tee -a "$OUT"
    continue
  fi
  t0=$SECONDS
  arc=$(score "$path" "$ARC" "")
  hs=$(score "$path" "$HS" "--multiple-choice-tasks $HS_TASKS")
  printf '%-20s ARC-Challenge %-20s HellaSwag %-20s (%ds)\n' \
    "$label" "$arc" "$hs" "$((SECONDS - t0))" | tee -a "$OUT"
done

echo "=== done $(date -Iseconds) ===" | tee -a "$OUT"

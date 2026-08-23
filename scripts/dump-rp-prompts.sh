# Re-emits the battery's PROMPTS array NUL-separated, so a scorer can strip the
# prompt from each block instead of guessing where the completion starts.
set -euo pipefail
src="${1:-scripts/eval-rp-completions.sh}"
eval "$(sed -n '/^PROMPTS=(/,/^)$/p' "$src")"
printf '%s\0' "${PROMPTS[@]}"

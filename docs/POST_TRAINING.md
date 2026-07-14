# Post-training: preference optimization (DPO) and RLVR (GRPO)

Implementation notes for adding preference/alignment stages **after** a coherent base + SFT model
exists. Both are portable pure-TS (no PyTorch, no new deps) and fit the existing autograd + Gemma3 +
GGUF path. They are **deferred, not rejected**: neither is worth building until the base is trained
and, for DPO, until preference-pair data exists. Recorded here so a future session can pick them up
with the design already scoped.

Source of the recipe: `Y0oshi/Text-LLM-Training-from-scratch`
(`textllm/train/{dpo,grpo,reward,rewards}.py`), adapted to this codebase.

## What already exists to build on

| Need                                                                | Where                                                                      |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Assistant-only loss mask (train on the response, ignore the prompt) | `src/data/chat.ts` `assistantLossMask()` + `maskedTargets()`               |
| ChatML render + curriculum specials (`<think>` etc.)                | `src/data/chat.ts` `CHATML_SPECIALS`, `CURRICULUM_SPECIALS`                |
| Cross-entropy with ignore-index (target < 0 skipped)                | `src/model/autograd.ts` `crossEntropy()`                                   |
| Load a second model (the frozen reference) from a GGUF              | `src/export/load_gguf.ts` `loadWeightsFromGGUF()` / `loadGemma3FromGGUF()` |
| Device-resident training loop                                       | `src/backend/train_gpu.ts` `trainLMGpuResident()`                          |
| Greedy decode (rollout scaffold)                                    | `src/eval/generate.ts` `greedyComplete()`                                  |

## Two ops to add first (both DPO and GRPO need them)

`autograd.ts` currently exports `add`, `mul`, `silu`, `gelu`, `scale`, `crossEntropy`, but **no
`logsigmoid`, `sigmoid`, or `sub`**. Add a small differentiable `logsigmoid(x)` (numerically stable:
`-softplus(-x)`), which covers the DPO loss. The per-example sequence log-probs are single scalars,
so the preference/PPO loss and its backward seed can run on the CPU even while the forward runs on
the GPU; no new WGSL shader is required.

**Sequence log-prob caveat.** `crossEntropy()` returns the **mean** NLL over kept (non-ignored)
rows. DPO/GRPO want the **summed** response-token log-prob. Do not re-derive it; reconstruct as
`-mean * kept` (recover `kept` from the mask, or expose it from `crossEntropy`). The PyTorch
clamp-then-mask gather idiom in the source is redundant here: our `crossEntropy` already ignores
`target < 0`.

## DPO (do this one first: self-contained, no reward model, no rollouts)

`loss = -logsigmoid(beta * margin)`, where
`margin = (polChosen - refChosen) - (polRejected - refRejected)` and each term is the **summed**
response-token log-prob of that sequence.

Steps:

1. New `src/train/dpo.ts`. Reference model = a second `Gemma3Model` loaded via `loadWeightsFromGGUF`
   from the SFT checkpoint, params frozen (no backward, no optimizer). On a 128GB target a second
   94.7M model is ~0.4GB, negligible; memory is **not** the blocker.
2. Data format (JSONL): `{"prompt": [...messages], "chosen": "...", "rejected": "..."}`. Encode
   chosen/rejected against the shared prompt; reuse `assistantLossMask` so only the response tokens
   count. Drop pairs with no supervised token after truncation to `maxSeq`.
3. Per pair: 4 forwards (policy+ref x chosen+rejected) give 4 summed log-probs, then a scalar loss,
   then backward through the **policy** only. `beta` ~ 0.1 to start.
4. Use the summed (not length-normalized) form of the original DPO paper; length normalization is a
   different objective.

**Real blocker to starting DPO:** no preference-pair data exists or is planned (TinyStories, the
722M blend, and the reasoning/instruct curriculum are all single-completion SFT). DPO is inert
without chosen/rejected pairs, and it is late-stage polish: at 94.7M mid-pretrain the bottleneck is
base coherence, not alignment. Source or synthesize preference pairs first.

## GRPO / RLVR (verifiable rewards: after DPO, and only if the base is competent)

Group-relative policy optimization, no value network:

- Sample a **group** of completions per prompt.
- `advantage = (r - mean(r)) / (std(r) + 1e-6)`; the group mean is the baseline.
- Per-token k3 KL to a frozen reference: `kl = exp(refLp - polLp) - (refLp - polLp) - 1`.
- `loss = sum over response tokens of (-adv * polLp + beta * kl) / (#response tokens)`.
- Response mask starts at `len(prompt) - 1` (the shifted-target boundary).
- Rewards are plain TS closures in a new `src/train/rewards.ts`: last-integer match (reuse
  `extractLastInt` from `examples/eval_generative.ts`), a "the answer is ..." format bonus,
  substring, length-target. This is the SAME reward the generative eval uses, so training and eval
  close the loop.

**Two blockers, both bigger than DPO's:**

1. **Sparse-reward collapse.** `advantage` is ~0 when every completion in a group scores the same. A
   from-scratch 94.7M model will almost never emit a correct verifiable answer, so groups are
   uniformly zero-reward and the gradient vanishes. RLVR _amplifies_ existing competence; it does
   not create it. Gate GRPO on the model first clearing a non-trivial bar on `eval_generative` (i.e.
   the arithmetic set, not GSM8K).
2. **Rollouts need efficient generation we deliberately don't have.** Group sampling wants a real
   temperature/top-p sampler + a KV cache. `greedyComplete` is O(n^2) full-recompute greedy: fine
   for a handful of eval completions, far too slow for RL rollouts. Options: (a) build an
   inference-only KV-cache decode path (large; see the mining analysis, candidate c2), or (b) shell
   out to llama.cpp for rollouts against the exported GGUF and score its output. (b) is more in
   keeping with the "llama.cpp does generation" charter but couples the RL loop to an external
   binary and a re-export each policy update.

## Not applicable here (from the same source repos)

- **Reward model (Bradley-Terry).** Prerequisite for PPO-style RL, which DPO was invented to avoid.
  Skip unless a committed PPO roadmap appears; then add a `forwardHidden()` seam to `gemma3.ts`
  (return `normed` before the LM head).
- **MoE, MLA, YaRN.** MoE/MLA break the dense `gemma3` GGUF export (llama.cpp gemma3 has no MoE/MLA
  tensors); YaRN is moot because we pretrain directly at the target context and 5/6 layers are
  SWA-windowed. See the repo mining report.

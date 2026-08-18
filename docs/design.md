# Design notes & modern-technique roadmap

> Rationale and measurements behind the current design. Maintained: if something here contradicts
> the code, the code is right and this is a bug.

## Why the "train directly in Q4_0" idea doesn't hold

Q4_0 rounding is non-differentiable and a gradient step is typically far smaller than a 4-bit
bucket, so writing updates straight into Q4_0 blocks rounds to zero or thrashes. Pretraining is
_more_ precision-sensitive than fine-tuning. So we keep **float master weights** during training and
quantize **at export**. "GGUF only" is honored: GGUF is the sole model/checkpoint format, HF never
appears.

## Optimizer: why Muon, not just Adam

Adam treats each weight as an independent scalar. Muon treats a weight _matrix_ as a matrix: it
takes the momentum buffer and replaces it with its nearest semi-orthogonal matrix (a quintic
Newton–Schulz iteration), which conditions every update. In 2025 this stopped being a toy: Muon
(and variants) trained **Kimi K2 (~1T)**, **GLM-4.5 (355B)**, and **INTELLECT-3 (106B)**, reportedly
reaching AdamW-level loss at roughly **half the FLOPs** and with **less optimizer memory** (one
momentum buffer vs Adam's two). Standard recipe (which this repo follows) applies **Muon to 2-D
hidden matmuls only**, and AdamW to embeddings, the output head, and all norms.

Implemented: `src/train/muon.ts` (+ `adam.ts`). Both satisfy one `Optimizer` interface so the
trainer is agnostic.

## Techniques worth adopting (prioritized)

Sourced from the nanoGPT speedrun lineage and 2025 large-scale training reports. Ranked by payoff
for a small from-scratch Gemma3 on a WebGPU budget.

**Already in this repo (they're part of Gemma3):** QK-norm, dual local/global RoPE, GeGLU, GQA,
RMSNorm + sandwich norms (post-attention / post-FFN), tied embeddings, and per-layer sliding-window
attention.

1. **Muon optimizer**: done. Biggest single win; ~2× compute efficiency.
2. **muP (maximal-update parametrization)**: init part done (`src/model/mup.ts`), contract-safe.
   `Gemma3Model(cfg, rng, { baseWidth })` scales the embedding/head init by `sqrt(baseWidth/width)`
   so the tied-readout logits stay O(1) across widths; hidden matmuls keep `1/sqrt(fan_in)` and
   attention keeps `1/sqrt(headDim)` (both already width-correct and the llama.cpp contract), and
   there are no forward multipliers, so the forward pass and GGUF are unchanged. LR transfer: the
   aux group is width-insensitive and Muon's update is spectrally normalized, so the transfer recipe
   is muP init + keep the tuned LRs (no width-LR scaling). The coordinate check in
   `tests/gradcheck.ts` (`muP coordinate check`) confirms it: across an 8× width sweep, standard
   init's readout logit RMS grows ~2.6× (≈√8) while muP init holds it to ~1.14×, and constant-lr
   training stays bounded across widths. Tune on a narrow proxy, then widen with muP init to
   transfer. A multi-step coordinate check on a real corpus confirmed it end to end: across a 4×
   width sweep on TinyStories,
   standard-init readout logit RMS grows 2.00× (≈√4) while muP holds it to 1.01×, and 120
   constant-lr steps stay bounded at every width. The loss transfers about equally either way
   (final-loss spread ~1.05× for both): Muon's spectrally-normalized update already carries the
   hidden-matmul LR across widths, so muP's distinct job is pinning the readout logit scale, which
   the init sweep isolates. No width-LR scaling was needed (the transfer held), so `mup.ts` stays
   init-only.
3. **LR schedule: warmup → stable → linear cooldown** (WSD), done. `src/train/schedule.ts`
   `wsdSchedule()` returns a per-step multiplier in [minScale, 1]; the trainer applies it via
   `setLrScale()` on both groups (Muon + aux AdamW) each step. `MuonGpu` holds `lr` in a 1-element
   device buffer that `setLrScale()` rewrites, no WGSL constant baked, so the apply pipelines are
   built once and the schedule costs a 4-byte upload per step. Gated by `wsdScheduleParity`
   (GPU-buffer lr vs CPU) in `tests/gpu-parity.ts` and a shape self-check in `tests/gradcheck.ts`.
   Its payoff is on long runs where constant lr plateaus; the 40-step demos leave it off (the
   cooldown only costs final loss when the model is still descending steeply).
4. **MuonClip / QK-logit control**: done, adapted for QK-norm. `src/train/qk-clip.ts`
   `applyQKClip(model, tau)` caps each layer's logit-scale proxy
   `s = (1/√headDim)·√Σ_d(qNorm[d]·kNorm[d])²` at `tau` by rescaling `qNorm`/`kNorm` (each by
   `√(tau/s)`, so the product scales by `tau/s` and `s → tau`). Moonshot's original clips the q/k
   _projection_ weights, but Gemma3's QK-RMSNorm renormalizes those away, so the norm weights are
   the real lever (per-layer, since `qNorm`/`kNorm` are shared across a layer's heads). The proxy is
   the std of a QK-normed logit; measured (T=128) the observed causal max tracks ~3.3–4.4× it and is
   monotone, so bounding `s` bounds the max. Opt-in via `qkClipTau` on all three training loops;
   host-side weight math, so CPU and GPU apply it identically. Gated by `qkClipTrajectoryParity` in
   `tests/gpu-parity.ts` and a unit check in `tests/gradcheck.ts`. Off by default: QK-norm is the
   primary guard; this is the belt-and-suspenders for scaling up.
5. **GPU-resident Muon + flash attention**: done. Muon now runs entirely on the GPU (~3 ms/step at
   725K params vs 1276 ms CPU; ~2 ms at 5M params vs ~10 s). Causal attention uses a hybrid
   dispatch: materialized `[Hq,T,T]` path below T=2048 (faster there due to higher thread
   parallelism), online-softmax flash path at T≥2048 (O(Hq·T) memory, no single-buffer ceiling).
   Full numbers in `notes/journal.md`.
6. **Data quality > everything at small scale**: a curated corpus beats architecture tweaks for
   models in the 1–50M range. Concretely:
   [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (short synthetic stories,
   deliberately limited vocabulary) gets coherent output out of models as small as **3M params**;
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (1.3T tokens,
   quality-filtered from FineWeb's 15T, the corpus behind the SmolLM family) is the step up once a
   model needs to know things beyond storytelling. Start with the former to validate the pipeline at
   real (non-toy) scale, move to a slice of the latter once it saturates. Pipeline is ready and
   wired to a real corpus: `gemma3Config()` for 10–50M sizes, a disk-streaming token loader
   (`src/data/tokens.ts`, so the corpus needn't fit in memory), `tokenize` to turn any text file
   into a `.tokens` binary + reusable vocab (run against the TinyStories validation slice: 5.4M
   tokens, vocab 8192), and `pretrain` as the turnkey run: reuse the vocab, muP init, WSD,
   GPU-resident Muon+AdamW, export and verify the GGUF.
   Smoke-tested end-to-end (loss drops, parity exact, GGUF loads in llama-cli); the remaining work
   is the multi-hour run itself, not code. FineWeb-Edu drops into the same `pretokenize` → `train`
   path once TinyStories saturates.
7. **Speedrun tricks (ReLU² MLP / value-residuals / sliding-window attention)**: evaluated;
   decision below. These are the nanoGPT-speedrun lineage's smaller-gain tricks. Measured against
   invariant #1 (GGUF loadability): two are architecture changes a `gemma3`-typed GGUF cannot carry
   (so they stay documented-only), while the third (sliding-window attention) is now **built**,
   because it is part of the Gemma3 architecture this repo trains:
   - **ReLU² MLP**: replaces GeGLU's `gelu(gate)·up` with `relu(x)²`. Gemma3's FFN is GeGLU by
     definition (`ffn_gate` + `ffn_up` + `ffn_down`); a ReLU² block has no gate tensor, so
     `llama.cpp` would not load the export as `gemma3`. Adopting it means shipping a non-loadable
     model or forking the GGUF arch. **Off, not built.**
   - **Value residuals**: a learned per-layer mix of layer 0's value stream into deeper layers. Not
     in the Gemma3 tensor schema, so the same problem: extra per-layer tensors the `gemma3` graph
     won't read. **Off, not built.**
   - **Sliding-window attention**: **built**, as a first-class Gemma3 feature (not just a
     training-time warmup). Each SWA layer restricts query `t` to keys `[t-W+1, t]`; the
     dense/global layers (every `swaPattern`-th) keep full causal attention, each with its own RoPE
     base. The masked windowed kernel is broadcast-preserving (block-aligned lower key bound +
     per-row mask), so it is a real throughput lever at long context rather than the slowdown a
     naïve per-thread window causes; exercised by `tests/gpu-parity.ts` (windowed mat + flash) and
     `tests/gradcheck.ts` (`attention(SWA)`). A training-time _window warmup_ (shrink the window
     early, open it later) could still layer on top as pure loop logic: **not built yet;
     implementable on request.**

   Bottom line: sliding-window attention is in and validated. If a non-`gemma3` research export is
   ever acceptable, revisit ReLU² first (largest remaining gain).

**Deliberately deferred:** FP8/NVFP4 low-precision _training_ (Quartet, custom FP8 head). Big at
cluster scale, but WebGPU targets f16/f32 compute, not worth the complexity here. Q4_0/Q8_0 remain
**export-time** only.

## Precision: f32, and why not f16/bf16

Training runs in **f32** end to end: f32 master weights, f32 GEMM, f32 accumulate, f32
gradients/optimizer. An f16-operand GEMM path existed (f16 multiply, f32 accumulate) but was
**removed**: see below.

A fair question on AMD Strix Halo (RDNA 3.5, RADV GFX1151), whose silicon has native bf16: why not
bf16? Because WebGPU cannot reach it. WGSL's only reduced-precision scalar is `f16` (IEEE half),
gated by the `shader-f16` feature; there is no `bf16` WGSL type and no `shader-bf16` WebGPU feature,
and Deno's wgpu/naga shader compiler has no bf16 either. So the backend physically cannot emit a
bf16 kernel, and the hardware's bf16 units are unreachable from this path. Same story for f16 matrix
cores (WMMA): not exposed to WebGPU here.

So why not at least f16-operand GEMM? It was tried and it **failed on two counts.** (1) No speedup
at the sizes we train: f16 accelerates the GEMM (~9% of runtime), but attention (~78%) dominates, so
the end-to-end change was unmeasurable (0.28 f32 vs 0.27 f16 st/s on the 31M pretrain). (2) It
**overflows**: casting each operand to `f16(v)` sends any value past f16's 65504 ceiling to
`inf`→NaN, and the real 20k-step run died at step 2400 at every LR once trained activations grew
large. The trap was that the init-time CPU-f32/GPU-f16 parity probe agreed to ~1e-6 (9.6941 vs
9.6941), but that only tests the _untrained_ model, whose activations are small; f16 looked
loss-free and wasn't. This is exactly the failure bf16's wider exponent range would have avoided,
and exactly why f16 needs loss-scaling/clamping that f32 does not. Since f16 bought no speed here,
f32 is strictly better: stable AND same wall-clock. bf16 would pay off only on a native ROCm/PyTorch
path that taps the matrix cores: a different engine, and the real throughput lever, not a precision
flag.

## WebGPU backend bring-up

**Done (items 1–5)**: the CPU op set (`src/model/autograd.ts`) is implemented as WGSL compute
shaders behind the same `Tensor` interface in `src/backend/webgpu.ts`, forward and backward, in the
planned order:

1. `matmul`/`linear` (tiled GEMM): throughput-critical, built first. ✓ Later register-tiled (4×4
   micro-tile per thread, unrolled to stay in registers; 1.9–3.3× end-to-end). An f16-operand
   variant was added and then removed (no speedup at our sizes + overflow; see "Precision" above).
   Kernel sources now live in `src/backend/wgsl.ts`; see the performance-work section in
   `notes/journal.md`.
2. elementwise `add`, `mul`, `silu`. ✓
3. `rmsnorm`, `rmsnorm_heads` (QK-norm): workgroup reductions. ✓
4. `rope`, causal `attention`, `cross_entropy`. ✓ Attention uses a hybrid dispatch: materialized
   `[Hq,T,T]` path for T < 2048 (higher thread parallelism wins at small T), online-softmax flash
   path for T ≥ 2048 (O(Hq·T) memory, no buffer-size ceiling; 1.4–2.2× faster at T=2048–8192).
5. GPU-resident optimizers (`src/backend/muon-gpu.ts` + `adamw-gpu.ts`). ✓ Newton–Schulz runs
   entirely on the GPU via the existing tiled GEMM; momentum buffers and weights are
   device-resident. ~3 ms/step at 725K params and ~2 ms/step at 5M params, vs 1276 ms / ~10 s on
   CPU. The aux group (embeddings, head, norms) is device-resident too: AdamW with its global
   grad-norm clip runs as GPU dispatches, so after warm-up only the loss scalars are read back
   (per-step readback fell from ~145 KiB to 8 bytes at tinyConfig; the embedding grad dominated and
   grows with vocab·hidden). Measured on M1 Max; full numbers in `notes/journal.md`.

**Validation gate (in place):** `tests/gradcheck.ts` finite-difference-checks every CPU op;
`tests/gpu-parity.ts` checks every kernel's forward and gradient against the CPU backend. Both ran
green before the backend was swapped in.

## WebGPU memory budget (what the "4 GB" limit really is)

A recurring question: is training capped at 4 GB of VRAM? No. The 4 GiB figure is a _per-tensor
binding_ cap, not a total-memory budget. Three different limits are in play, measured here on an M1
Max with 32 GB unified memory:

| Limit                         | M1 Max value                 | What it bounds                                          |
| ----------------------------- | ---------------------------- | ------------------------------------------------------- |
| `maxStorageBufferBindingSize` | 4.00 GiB (4,294,967,292 B)   | the largest _single_ storage buffer one kernel can bind |
| `maxBufferSize`               | 18.72 GiB (20,100,448,256 B) | the largest _single_ allocation                         |
| unified memory pool           | 32 GB                        | the sum of everything resident                          |

`initWebGPU()` already requests the adapter's full limits (with a graceful fallback), so all three
apply. The 4 GiB bound is per _binding_: no one tensor (a weight matrix, a gradient, a logit buffer)
may exceed it, but a model is thousands of tensors and the total is bounded by the 32 GB pool, not
by 4 GiB. Probed directly on this machine: allocating 8 GiB of concurrent `STORAGE` buffers succeeds
with no OOM.

**Training-state footprint.** With both optimizer groups device-resident, resident state is roughly
**20 bytes per parameter**: Muon on the 2-D hidden matmuls keeps weight + grad + momentum +
Newton–Schulz scratch; AdamW on the aux group keeps weight + grad + m + v (all f32). Parameter state
for a 33M model is therefore ~0.66 GB, and even a 200M model's ~4 GB of state sits well inside 32
GB. Activations add on top (scaling with `batch · seqLen · hidden · layers`), but at these sizes
they are not the ceiling either.

**The real limit is compute throughput, not memory.** On the M1 Max a 5.2M-param model runs ~0.9
s/step and a 33M model ~5–6 s/step (device-resident, seqLen 256, small batch). Going bigger is a
question of how many hours (or overnight epochs) you will wait, not whether the weights fit. Only a
_single_ tensor larger than 4 GiB would need tiling to stay under the binding cap, and at
WebGPU-realistic model sizes (single-digit to low-hundreds of millions of params) no individual
tensor comes close.

## References

- Muon is Scalable for LLM Training: arxiv.org/html/2502.16982v1
- Practical Efficiency of Muon for Pretraining (Essential AI): arxiv.org/pdf/2505.02222
- nanoGPT speedrun technique log: github.com/alexjc/nanogpt-speedrun
- Muon + DeepSpeed (PyTorch blog): pytorch.org/blog/using-muon-optimizer-with-deepspeed/
- TinyStories dataset: huggingface.co/datasets/roneneldan/TinyStories
- FineWeb-Edu dataset: huggingface.co/datasets/HuggingFaceFW/fineweb-edu

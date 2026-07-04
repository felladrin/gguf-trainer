# Design notes & modern-technique roadmap

## Why the "train directly in Q4_0" idea doesn't hold

Q4_0 rounding is non-differentiable and a gradient step is typically far smaller than a 4-bit
bucket, so writing updates straight into Q4_0 blocks rounds to zero or thrashes. Pretraining is
_more_ precision-sensitive than fine-tuning. So we keep **float master weights** during training and
quantize **at export**. "GGUF only" is honored: GGUF is the sole model/checkpoint format, HF never
appears.

## Optimizer: why Muon, not just Adam

Adam treats each weight as an independent scalar. Muon treats a weight _matrix_ as a matrix: it
takes the momentum buffer and replaces it with its nearest semi-orthogonal matrix (a quintic
Newton–Schulz iteration), which conditions every update. In 2025 this stopped being a toy — Muon
(and variants) trained **Kimi K2 (~1T)**, **GLM-4.5 (355B)**, and **INTELLECT-3 (106B)**, reportedly
reaching AdamW-level loss at roughly **half the FLOPs** and with **less optimizer memory** (one
momentum buffer vs Adam's two). Standard recipe — which this repo follows — applies **Muon to 2-D
hidden matmuls only**, and AdamW to embeddings, the output head, and all norms.

Implemented: `src/train/muon.ts` (+ `adam.ts`). Both satisfy one `Optimizer` interface so the
trainer is agnostic.

## Techniques worth adopting (prioritized)

Sourced from the nanoGPT speedrun lineage and 2025 large-scale training reports. Ranked by payoff
for a small from-scratch Qwen3 on a WebGPU budget.

**Already in this repo (they're part of Qwen3):** QK-norm, RoPE, SwiGLU, GQA, RMSNorm, tied
embeddings.

1. **Muon optimizer** — done. Biggest single win; ~2× compute efficiency.
2. **muP (maximal-update parametrization)** — tune LR/init on a _tiny_ proxy model and transfer to
   the full model without re-tuning. High value because hyperparameter search on WebGPU is
   expensive. Touches init + per-tensor LR. Note: `MuonGpu` bakes `lr` into WGSL pipeline constants
   at construction; a schedule requires either rebuilding apply pipelines on lr change, or switching
   to a uniform-buffer approach.
3. **LR schedule: warmup → stable → linear cooldown** (WSD). Cheap, reliably improves final loss.
   Pure trainer change (with the MuonGpu caveat above).
4. **MuonClip / QK-logit control** — Moonshot's Muon variant clips attention logits to stop the
   "attention logit explosion" that shows up at scale; complements the QK-norm we already have. Add
   when scaling up.
5. **GPU-resident Muon + flash attention** — done. Muon now runs entirely on the GPU (~3 ms/step at
   725K params vs 1276 ms CPU; ~2 ms at 5M params vs ~10 s). Causal attention uses a hybrid
   dispatch: materialized `[Hq,T,T]` path below T=2048 (faster there due to higher thread
   parallelism), online-softmax flash path at T≥2048 (O(Hq·T) memory, no single-buffer ceiling).
   Full numbers in `docs/HANDOFF.md`.
6. **Data quality > everything at small scale** — a curated corpus beats architecture tweaks for
   models in the 1–50M range. Concretely:
   [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (short synthetic stories,
   deliberately limited vocabulary) gets coherent output out of models as small as **3M params**;
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (1.3T tokens,
   quality-filtered from FineWeb's 15T, the corpus behind the SmolLM family) is the step up once a
   model needs to know things beyond storytelling. Start with the former to validate the pipeline at
   real (non-toy) scale, move to a slice of the latter once it saturates.
7. **ReLU² MLP / value-residuals / attention-window warmup** — speedrun tricks with real but smaller
   gains. Note: ReLU² would diverge from the Qwen3 schema (SwiGLU), so keep it optional/off by
   default to preserve llama.cpp loadability.

**Deliberately deferred:** FP8/NVFP4 low-precision _training_ (Quartet, custom FP8 head). Big at
cluster scale, but WebGPU targets f16/f32 compute — not worth the complexity here. Q4_0/Q8_0 remain
**export-time** only.

## WebGPU backend bring-up

**Done (items 1–5)** — the CPU op set (`src/model/autograd.ts`) is implemented as WGSL compute
shaders behind the same `Tensor` interface in `src/backend/webgpu.ts`, forward and backward, in the
planned order:

1. `matmul`/`linear` (tiled GEMM) — throughput-critical, built first. ✓
2. elementwise `add`, `mul`, `silu`. ✓
3. `rmsnorm`, `rmsnorm_heads` (QK-norm) — workgroup reductions. ✓
4. `rope`, causal `attention`, `cross_entropy`. ✓ Attention uses a hybrid dispatch: materialized
   `[Hq,T,T]` path for T < 2048 (higher thread parallelism wins at small T), online-softmax flash
   path for T ≥ 2048 (O(Hq·T) memory, no buffer-size ceiling; 1.4–2.2× faster at T=2048–8192).
5. GPU-resident Muon optimizer (`src/backend/muon_gpu.ts`). ✓ Newton–Schulz runs entirely on the GPU
   via the existing tiled GEMM; momentum buffers and weights are device-resident. ~3 ms/step at 725K
   params and ~2 ms/step at 5M params, vs 1276 ms / ~10 s on CPU. Measured on M1 Max; full numbers
   in `docs/HANDOFF.md`.

**Validation gate (in place):** `tests/gradcheck.ts` finite-difference-checks every CPU op;
`tests/gpu_parity.ts` checks every kernel's forward and gradient against the CPU backend. Both ran
green before the backend was swapped in.

## References

- Muon is Scalable for LLM Training — arxiv.org/html/2502.16982v1
- Practical Efficiency of Muon for Pretraining (Essential AI) — arxiv.org/pdf/2505.02222
- nanoGPT speedrun technique log — github.com/alexjc/nanogpt-speedrun
- Muon + DeepSpeed (PyTorch blog) — pytorch.org/blog/using-muon-optimizer-with-deepspeed/
- TinyStories dataset — huggingface.co/datasets/roneneldan/TinyStories
- FineWeb-Edu dataset — huggingface.co/datasets/HuggingFaceFW/fineweb-edu

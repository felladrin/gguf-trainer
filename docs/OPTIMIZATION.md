# Optimization notes

What is left to make training faster, larger, more robust, or higher quality — grounded in
measurements taken on the Strix Halo (AMD Radeon, RADV GFX1151, 128 GB unified) during the
94.7M-param / 8192-context pretraining run, not on speculation. Read `docs/DESIGN.md` for
architecture and `README.md` "Honest limits" for the ceiling this project accepts.

## Measured baseline (2026-07-08)

| Metric                     | Value                    | How measured                                        |
| :------------------------- | :----------------------- | :-------------------------------------------------- |
| Throughput                 | 0.049 st/s (20.4 s/step) | 94.7M, seq 2048, batch 8, `bench_pretrain.ts`       |
| GPU busy                   | ~100%                    | `gpu_busy_percent` during the live run              |
| Host RAM used by run       | ~34 GB                   | `ps rss` of the training process                    |
| GPU-visible memory (GTT)   | ~37 GB of 128 GB         | `mem_info_gtt_used`                                 |
| Attention share of runtime | ~78%                     | prior kernel profiling (see memory / task #27)      |
| f16 vs f32 compute         | same speed               | attention-bound; f16 only speeds the ~9% GEMM slice |

Two facts follow from this and drive everything below:

1. **We are compute-bound on attention, not memory-bound.** The GPU is pinned at 100% while only
   ~30% of RAM is in use. More memory or a bigger batch cannot raise tokens/second — only cheaper
   attention can.
2. **Batch is sequential gradient accumulation, not a real batch dimension.** The training loop runs
   one sequence per forward/backward and sums the gradients (`train_gpu.ts`), so batch size trades
   step count for per-step time at a fixed tokens/second. It changes gradient noise, not throughput.

## Throughput levers (the binding constraint)

### 1. Attention kernel — the ~78% slice (hard, standing)

This is the only lever that moves the wall-clock. Full attention is "at floor" on gfx1151: no
WMMA/tensor-core path, and the tiled/flash-style forward already attempted (task #33) did not break
the ceiling. SWA (sliding window 1024) is the one real win in place — it caps per-token cost on 5 of
every 6 layers. Remaining ideas, none proven: a fused windowed-attention kernel that never
materializes the full score matrix; better workgroup occupancy for the global (full-attention)
layers; exploiting the 5:1 SWA:global ratio to skip work. Expect this to be a research effort, not a
quick fix. It is also the same wall PyTorch+ROCm hits on this GPU, so it is not a "switch
frameworks" problem. CONSTRAINT: only pursue approaches that stay portable — plain WGSL that runs
cross-vendor (AMD/Apple/ NVIDIA), no vendor intrinsics or hardware-specific paths. A kernel that
only helps gfx1151 at the cost of the "runs anywhere" story is out of scope.

### 2. True micro-batching (medium, uncertain payoff)

Packing the `batchPerStep` sequences into one real batch dimension would enlarge the GEMMs and cut
per-launch + sync overhead. But GEMM is only ~9% of runtime and the GPU is already saturated at
batch 1, so the upside is capped by how much launch/sync overhead actually exists. Worth a
measurement spike (time seq 2048 at batch 1 vs the per-sequence cost inside batch 8) before
committing to the rewrite of the model forward.

## Memory / scale levers

### 3. seq-8192 batch>1 is a WebGPU per-buffer limit, not memory (investigate)

seq 8192 batch 2 fails with a RADV "context lost" / validation error, while batch 1 is fine. This is
NOT a memory-size cap: grub already sets `ttm.pages_limit` to the full 128 GB (`mem_info_gtt_total`
= 131072 MB) and only ~37 GB is in use. So the failure is a WebGPU validation limit hit at that
allocation shape — most likely a single storage buffer exceeding `maxStorageBufferBindingSize` (an
intermediate or gradient buffer that only appears at batch 2 / seq 8192), or a dispatch-size limit.
Next step (needs a GPU-idle moment — do NOT probe during a live run): query
`adapter.limits.maxStorageBufferBindingSize` / `maxBufferSize` and log the size of the buffer that
trips `webgpu.ts` sync validation, then either split that buffer or tile the op. This overlaps the
attention-kernel work (#1) and must stay portable. Phase B works at batch 1 regardless, so this is
an efficiency unlock, not a blocker.

### 4. More unique data (cheap data, expensive compute)

The corpus is 722M unique tokens; the run does 2 epochs (~1.44B). `prepare_pretrain.ts` can emit
more parts for near-zero cost, but training them is the expense: at 70M tok/day each additional
~2.6B tokens is ~37 days. This is where the real gap to Minueza-2-96M (185B tokens) lives — closing
it is a compute-time decision, not a tooling one.

## Correctness / robustness

### 5. Checkpoint optimizer state (medium, real gap) — DONE

Checkpoints now write a `<ckpt>.optstate` sidecar (Muon momentum + Adam moments + step) beside the
weights and restore it on `--resume`; absent -> cold start as before. `readStateBuffer` in the
backend does the readback. Validated: bit-exact round-trip, GPU parity unchanged, end-to-end resume.
(The in-flight Phase A run predates this — its checkpoints have no sidecar and resume cold; future
runs and Phase B are seamless.)

### 6. rsync hygiene (cheap)

Plain rsync without `--delete` leaves stale files on the Strix copy when a source file is removed —
a real dual-machine drift bug. A small sync script with `--delete` (excluding `corpus/`, `*.gguf`,
`*.tokens`) removes the class. Moot if we run directly on Strix.

### 7. `--logEvery` flag (cheap)

`logEvery` is `steps/100` (880 steps ≈ 5 h between log lines here), so the first post-step-0 loss
reading is far out; checkpoints every 500 steps are the only early signal. A `--logEvery` flag would
give a tighter early-training view without touching the checkpoint cadence.

## Quality levers

### 8. WSD decay-phase instruct injection (medium) — MECHANISM DONE

The MiniCPM / Xmodel-2 trick: blend a small fraction of ChatML instruct data into the WSD cooldown
so the base emerges more instructable. Implemented:
`pretrain.ts
--inject=<tokens> --injectFrac --injectFrom` draws that fraction of micro-batches from
a second token source during the cooldown (default window = the WSD cooldown); off by default, and
the extra rng() is drawn only while active so parity is intact. Remaining to USE it: prepare an
instruct `.tokens` encoded with the shared tokenizer (smol-smoltalk → ChatML) — tracked with the
instruct curriculum stage.

### 9. Eval harness (medium, validates the whole effort) — DONE (run pending base)

`examples/eval_mc.ts` scores a GGUF on ARC-Challenge / HellaSwag via length- normalized
log-likelihood (matches lm-eval-harness, comparable to the Minueza leaderboard numbers), plus
acc/acc_norm; datasets via the HF parquet tooling. Smoke-tested end-to-end. The full run (our base
vs both Minueza models) waits on an idle GPU after Phase A, and gates the "Minueza-3" naming on
data.

## Explicitly not worth doing

- **Guarded/clamped f16 compute** — no speed to recover; attention-bound, GEMM is a thin slice. f32
  also learns better. (Confirmed; f16-compute path removed.)
- **Using all 128 GB** — we are compute-bound; extra RAM buys nothing at this model size. It only
  matters as headroom for a much larger model, which the throughput ceiling makes impractical
  anyway.
- **Switching to PyTorch + ROCm for speed** — the attention wall is the GPU, not the framework; ROCm
  on gfx1151 also lacks flash-attention. See the project direction: portability is the goal, not
  chasing SOTA on this hardware.

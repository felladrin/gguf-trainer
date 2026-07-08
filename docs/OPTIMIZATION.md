# Optimization notes

What is left to make training faster, larger, more robust, or higher quality —
grounded in measurements taken on the Strix Halo (AMD Radeon, RADV GFX1151,
128 GB unified) during the 94.7M-param / 8192-context pretraining run, not on
speculation. Read `docs/DESIGN.md` for architecture and `README.md` "Honest
limits" for the ceiling this project accepts.

## Measured baseline (2026-07-08)

| Metric | Value | How measured |
| :-- | :-- | :-- |
| Throughput | 0.049 st/s (20.4 s/step) | 94.7M, seq 2048, batch 8, `bench_pretrain.ts` |
| GPU busy | ~100% | `gpu_busy_percent` during the live run |
| Host RAM used by run | ~34 GB | `ps rss` of the training process |
| GPU-visible memory (GTT) | ~37 GB of 128 GB | `mem_info_gtt_used` |
| Attention share of runtime | ~78% | prior kernel profiling (see memory / task #27) |
| f16 vs f32 compute | same speed | attention-bound; f16 only speeds the ~9% GEMM slice |

Two facts follow from this and drive everything below:

1. **We are compute-bound on attention, not memory-bound.** The GPU is pinned at
   100% while only ~30% of RAM is in use. More memory or a bigger batch cannot
   raise tokens/second — only cheaper attention can.
2. **Batch is sequential gradient accumulation, not a real batch dimension.** The
   training loop runs one sequence per forward/backward and sums the gradients
   (`train_gpu.ts`), so batch size trades step count for per-step time at a fixed
   tokens/second. It changes gradient noise, not throughput.

## Throughput levers (the binding constraint)

### 1. Attention kernel — the ~78% slice (hard, standing)
This is the only lever that moves the wall-clock. Full attention is "at floor"
on gfx1151: no WMMA/tensor-core path, and the tiled/flash-style forward already
attempted (task #33) did not break the ceiling. SWA (sliding window 1024) is the
one real win in place — it caps per-token cost on 5 of every 6 layers. Remaining
ideas, none proven: a fused windowed-attention kernel that never materializes the
full score matrix; better workgroup occupancy for the global (full-attention)
layers; exploiting the 5:1 SWA:global ratio to skip work. Expect this to be a
research effort, not a quick fix. It is also the same wall PyTorch+ROCm hits on
this GPU, so it is not a "switch frameworks" problem.

### 2. True micro-batching (medium, uncertain payoff)
Packing the `batchPerStep` sequences into one real batch dimension would enlarge
the GEMMs and cut per-launch + sync overhead. But GEMM is only ~9% of runtime and
the GPU is already saturated at batch 1, so the upside is capped by how much
launch/sync overhead actually exists. Worth a measurement spike (time seq 2048 at
batch 1 vs the per-sequence cost inside batch 8) before committing to the rewrite
of the model forward.

## Memory / scale levers

### 3. Raise the GTT cap (cheap, unlocks Phase B batch > 1) — external
seq 8192 batch 2 OOMs with a RADV "context lost" error, while batch 1 is fine and
~90 GB of host RAM sits free. The limit is the amdgpu GTT ceiling (~37 GB here),
not host RAM. Raising `amdgpu.gttsize` / `ttm.pages_limit` (kernel params, needs a
reboot) would let Phase B train long-context with batch > 1 instead of batch 1.
This is a config change on the machine, not a code change.

### 4. More unique data (cheap data, expensive compute)
The corpus is 722M unique tokens; the run does 2 epochs (~1.44B). `prepare_pretrain.ts`
can emit more parts for near-zero cost, but training them is the expense: at
70M tok/day each additional ~2.6B tokens is ~37 days. This is where the real gap
to Minueza-2-96M (185B tokens) lives — closing it is a compute-time decision, not
a tooling one.

## Correctness / robustness

### 5. Checkpoint optimizer state (medium, real gap)
Checkpoints save weights only (GGUF). Muon momentum and AdamW moments are not
serialized, so a crash-resume cold-starts the optimizer (momentum re-warms over a
few steps; minor but real) even though `--startStep` correctly continues the WSD
schedule. For a 3-week, possibly multi-segment run, serializing optimizer state
next to the checkpoint would make resumes seamless.

### 6. rsync hygiene (cheap)
Plain rsync without `--delete` leaves stale files on the Strix copy when a source
file is removed — a real dual-machine drift bug. A small sync script with
`--delete` (excluding `corpus/`, `*.gguf`, `*.tokens`) removes the class. Moot if
we run directly on Strix.

### 7. `--logEvery` flag (cheap)
`logEvery` is `steps/100` (880 steps ≈ 5 h between log lines here), so the first
post-step-0 loss reading is far out; checkpoints every 500 steps are the only
early signal. A `--logEvery` flag would give a tighter early-training view without
touching the checkpoint cadence.

## Quality levers

### 8. WSD decay-phase instruct injection (medium)
The MiniCPM / Xmodel-2 trick: blend a small fraction of ChatML instruct data into
the WSD cooldown phase so the base emerges more instructable. A training-wiring
change in `pretrain.ts` (mix a second token source during cooldown). Deferred from
the corpus work; the curriculum specials are already reserved in the vocab for it.

### 9. Eval harness (medium, validates the whole effort)
There is no automated eval yet. To back the "beat Minueza-32M / compare to
Minueza-2-96M" claims with numbers rather than estimates, add an ARC-Challenge +
HellaSwag (and perplexity) harness that scores our GGUF and both Minueza models
through the same path. This gates the "Minueza-3" naming decision on data.

## Explicitly not worth doing

- **Guarded/clamped f16 compute** — no speed to recover; attention-bound, GEMM is
  a thin slice. f32 also learns better. (Confirmed; f16-compute path removed.)
- **Using all 128 GB** — we are compute-bound; extra RAM buys nothing at this
  model size. It only matters as headroom for a much larger model, which the
  throughput ceiling makes impractical anyway.
- **Switching to PyTorch + ROCm for speed** — the attention wall is the GPU, not
  the framework; ROCm on gfx1151 also lacks flash-attention. See the project
  direction: portability is the goal, not chasing SOTA on this hardware.

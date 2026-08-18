# Optimization notes

> Measured levers, not speculation. Maintained: every number here was taken on real hardware, and
> the commands are the current CLI.

What is left to make training faster, larger, more robust, or higher quality: grounded in
measurements taken on the Strix Halo (AMD Radeon, RADV GFX1151, 128 GB unified) during the
94.7M-param / 8192-context pretraining run, not on speculation. Read `docs/design.md` for
architecture and `readme.md` "Honest limits" for the ceiling this project accepts.

## Measured baseline (2026-07-08)

| Metric                     | Value                    | How measured                                        |
| :------------------------- | :----------------------- | :-------------------------------------------------- |
| Throughput                 | 0.049 st/s (20.4 s/step) | 94.7M, seq 2048, batch 8                            |
| GPU busy                   | ~100%                    | `gpu_busy_percent` during the live run              |
| Host RAM used by run       | ~34 GB                   | `ps rss` of the training process                    |
| GPU-visible memory (GTT)   | ~37 GB of 128 GB         | `mem_info_gtt_used`                                 |
| Attention share of runtime | ~78%                     | kernel profiling, `timestamp-query`                 |
| f16 vs f32 compute         | same speed               | attention-bound; f16 only speeds the ~9% GEMM slice |

Two facts follow from this and drive everything below:

1. **We are compute-bound on attention, not memory-bound.** The GPU is pinned at 100% while only
   ~30% of RAM is in use. More memory or a bigger batch cannot raise tokens/second: only cheaper
   attention can.
2. **Batch is sequential gradient accumulation, not a real batch dimension.** The training loop runs
   one sequence per forward/backward and sums the gradients (`train-gpu.ts`), so batch size trades
   step count for per-step time at a fixed tokens/second. It changes gradient noise, not throughput.

## Throughput levers (the binding constraint)

### 1. Attention kernel: the ~78% slice (hard, standing)

This is the only lever that moves the wall-clock. Full attention is "at floor" on gfx1151: no
WMMA/tensor-core path, and the tiled/flash-style forward already attempted did not break
the ceiling. SWA (sliding window 1024) is the one real win in place: it caps per-token cost on 5 of
every 6 layers. Remaining ideas, none proven: a fused windowed-attention kernel that never
materializes the full score matrix; better workgroup occupancy for the global (full-attention)
layers; exploiting the 5:1 SWA:global ratio to skip work. Expect this to be a research effort, not a
quick fix. It is also the same wall PyTorch+ROCm hits on this GPU, so it is not a "switch
frameworks" problem. CONSTRAINT: only pursue approaches that stay portable, plain WGSL that runs
cross-vendor (AMD/Apple/ NVIDIA), no vendor intrinsics or hardware-specific paths. A kernel that
only helps gfx1151 at the cost of the "runs anywhere" story is out of scope.

### 2. True micro-batching (medium, uncertain payoff)

Packing the `batchPerStep` sequences into one real batch dimension would enlarge the GEMMs and cut
per-launch + sync overhead. But GEMM is only ~9% of runtime and the GPU is already saturated at
batch 1, so the upside is capped by how much launch/sync overhead actually exists. Worth a
measurement spike (time seq 2048 at batch 1 vs the per-sequence cost inside batch 8) before
committing to the rewrite of the model forward.

## Memory / scale levers

### 3. seq-8192 batch>1: RESOLVED, it was never a per-buffer limit

Re-probed 2026-08-04 on the idle GPU right after Phase A. **seq 8192 batch 2 and batch 4 both train
fine** (2 steps each, loss descending, gemma3 export OK). The old "RADV context lost at batch 2"
report does not reproduce; the flash-attention path plus the buffer pool removed whatever tripped
it. What remains is a plain memory wall, linear in tokens/step:

| config                       | tokens/step | peak GPU (pool + state) | throughput |
| ---------------------------- | ----------- | ----------------------- | ---------- |
| seq 2048 × batch 8 (Phase A) | 16384       | 39.3 GB (37.7 + 1.6)    | 903 tok/s  |
| seq 8192 × batch 2           | 16384       | 39.3 GB (37.7 + 1.6)    | 635 tok/s  |
| seq 8192 × batch 4           | 32768       | 76.2 GB (74.6 + 1.6)    | 653 tok/s  |

≈ 2.3 MB of activation pool per token in flight (micro-batch activations are all held for the
grad-accumulation step), so the 124 GB unified pool caps a step at ~50k tokens: **seq 8192 tops out
around batch 4-5**, not batch 1. No single buffer is near the 2048 MiB binding limit either: the
biggest is the per-micro-batch logits tensor, `T×V×4` = 1 GiB at T=8192/V=32768, and it does not
grow with batch. Phase B can budget batch 4.

The cost of long context is throughput, not memory: 8192 runs ~28% fewer tok/s than 2048 at equal
tokens/step (SWA covers 5 of 6 layers; the global layers still pay O(T²)).

### 4. More unique data (cheap data, expensive compute)

The corpus is 722M unique tokens; the run does 2 epochs (~1.44B). `corpus` can emit
more parts for near-zero cost, but training them is the expense: at 70M tok/day each additional
~2.6B tokens is ~37 days. This is where the real gap to Minueza-2-96M (185B tokens) lives: closing
it is a compute-time decision, not a tooling one.

## Correctness / robustness

### 5. Checkpoint optimizer state (medium, real gap): DONE

Checkpoints now write a `<ckpt>.optstate` sidecar (Muon momentum + Adam moments + step) beside the
weights and restore it on `--resume`; absent -> cold start as before. `readStateBuffer` in the
backend does the readback. Validated: bit-exact round-trip, GPU parity unchanged, end-to-end resume.
(Phase A's early checkpoints predate this and have no sidecar, so they resume cold.)

### 6. rsync hygiene (cheap)

Plain rsync without `--delete` leaves stale files on the Strix copy when a source file is removed,
a real dual-machine drift bug. A small sync script with `--delete` (excluding `corpus/`, `*.gguf`,
`*.tokens`) removes the class. Moot if we run directly on Strix.

### 7. `--logEvery` flag (cheap)

`logEvery` is `steps/100` (880 steps ≈ 5 h between log lines here), so the first post-step-0 loss
reading is far out; checkpoints every 500 steps are the only early signal. A `--logEvery` flag would
give a tighter early-training view without touching the checkpoint cadence.

## Quality levers

### 8. WSD decay-phase instruct injection (medium): MECHANISM DONE

The MiniCPM / Xmodel-2 trick: blend a small fraction of ChatML instruct data into the WSD cooldown
so the base emerges more instructable. Implemented:
`pretrain --inject <tokens> --inject-fraction F --inject-from-step N` draws that fraction of micro-batches from
a second token source during the cooldown (default window = the WSD cooldown); off by default, and
the extra rng() is drawn only while active so parity is intact. Remaining to USE it: prepare an
instruct `.tokens` encoded with the shared tokenizer (smol-smoltalk → ChatML): tracked with the
instruct curriculum stage.

### 9. Eval harness (medium, validates the whole effort): DONE, first full run recorded

`eval-choice` scores a GGUF on ARC-Challenge / HellaSwag via length- normalized log-likelihood
(matches lm-eval-harness, comparable to the Minueza leaderboard numbers), plus acc/acc_norm;
datasets via the HF parquet tooling.

**Phase A base, `phaseA-final-88000.gguf` (step 88000, 1.44B tokens), run 2026-08-04:**

| eval                                                          | result                                   |
| ------------------------------------------------------------- | ---------------------------------------- |
| ARC-Challenge, full 1172 items, 0-shot                        | acc_norm **22.78%**, acc 19.28% (648 s)  |
| HellaSwag, 2000 of 10042 items, 0-shot                        | acc_norm **32.20%**, acc 31.50% (1179 s) |
| `eval-loss`, 64 × 512 tok, in-distribution tail 1%, seed 1234 | loss **3.0915**, ppl 22.01               |

HellaSwag clears chance (25%) by 7 points; ARC-Challenge sits just under it, which is what a 94.7M
base at 1.44B tokens looks like: ARC needs knowledge this token budget cannot buy. Only 2000
HellaSwag items were scored, so that figure carries ~±1% sampling noise; the ARC number is the full
set.

**The cross-model comparison is CLOSED (2026-08-05).** At the time, the loader accepted only the
one architecture it had, so `eval-choice` could not score the reference models. (It is
architecture-agnostic today, but it still only reads GGUFs this project produced.) Everything below
was scored by llama.cpp's `llama-perplexity --multiple-choice` instead, same binary, same task
order, on CPU.
`scripts/minueza-3-run/head-to-head.sh` runs it. ARC-Challenge = all 299 validation tasks, HellaSwag
= all 10042, from llama.cpp's canonical sets (`ikawrakow/validation-datasets-for-llama.cpp`, in
`evaldata/`).

| model                                | train tokens | ARC-Challenge | HellaSwag       |
| ------------------------------------ | ------------ | ------------- | --------------- |
| SmolLM2-135M                         | ~2T          | 31.44 ±2.69   | 42.81 ±0.49     |
| Supra2-100M-Base                     | 30B          | 27.42 ±2.58   | 35.31 ±0.48     |
| llama-160m                           | ?            | 22.74 ±2.43   | 33.94 ±0.47     |
| **ours, phaseA-final-88000 (94.7M)** | **1.44B**    | 21.74 ±2.39   | **28.46 ±0.45** |
| Minueza-2-96M                        | 185B         | 23.08 ±2.44   | 27.03 ±0.44     |
| Minueza-32M-Base                     | ?            | 22.74 ±2.43   | 25.75 ±0.44     |
| Qwen2-96M                            | ?            | 23.41 ±2.45   | 24.85 ±0.43     |

**The "Minueza-3" naming now has data behind it, on HellaSwag.** We clear Minueza-2-96M by 1.43
points (28.46 vs 27.03; combined uncertainty ~0.63, so ~2.3σ) and Minueza-32M-Base by 2.7. On
ARC-Challenge all seven models sit at chance (25% ±2.4; even SmolLM2 only reaches 31), so no
ranking there is meaningful, ours included. Two consistency checks: HellaSwag scored 28.4704 and
28.4605 on two independent runs of our model (task order is deterministic), and Supra2's 35.31 here
is close to the 0.36 acc_norm its card reports under the EleutherAI LM-Eval Harness, which suggests
the two rulers agree at this scale even though they normalize differently. Do NOT mix our
`eval-choice` numbers with these: the same checkpoint scores 32.20 acc_norm there (on 2000 items)
and 28.46 here.

**Gotcha that silently voids an ARC run:** `llama-perplexity` decodes a task's answers in parallel
and ABORTS the whole run when a task has more options than `-np` allows. ARC-Challenge contains
5-option questions, so the default 4 dies at task 210 of 299 and prints no result. Pass `-np 8`.

Scoring the reference models needed one GGUF that did not exist, so it now does:
[Felladrin/gguf-f16-Supra2-100M-Base](https://huggingface.co/Felladrin/gguf-f16-Supra2-100M-Base)
(F16 conversion of `SupraLabs/Supra2-100M-Base`, apache-2.0). llama.cpp does not recognize that
model's BPE pre-tokenizer, and its `Sequence[Digits(individual_digits), ByteLevel]` is exactly what
llama.cpp's `qwen2` pre-type implements (bare `\p{N}` plus byte-level), so the conversion asserts
that mapping; token ids were verified identical to the source `transformers` tokenizer.

SupraLabs already ship a correct F16 GGUF of their Instruct model (`Supra2-100M-SFT-F16.gguf`, same
`qwen2` pre-type, chat template embedded), so a faithful re-conversion would add nothing. What was
published instead is a one-metadata-change variant,
[Felladrin/gguf-f16-Supra2-100M-Instruct](https://huggingface.co/Felladrin/gguf-f16-Supra2-100M-Instruct):
the source declares `</s>` as EOS while its chat template ends every turn with `<|im_end|>`, so
there `<|im_end|>` (id 6) is registered as both EOS and EOT. llama.cpp stops correctly either way
(it treats the `<|im_end|>` NAME as end-of-generation whatever the EOS metadata says, verified by
generating past a turn boundary on both files), so the variant only helps runtimes without that
heuristic. All 134 tensors are byte-identical (sha256 per tensor) between the two files, which is
also the cleanest proof that our conversion path reproduces theirs exactly.

### 10. Phase B KL anchor against the base checkpoint (medium): OP DONE

The continual-learning use of distillation (HF post `sergiopaniego/distillation-2026`): during Phase
B SFT, add a per-token KL term against the frozen Phase A base so its fluency survives fine-tuning;
models this small forget catastrophically under plain SFT. The teacher is this model, so the u16
custom-BPE vocab is no obstacle (external teachers are ruled out by it; see below). One new op: a
soft-target cross-entropy whose backward is `(p - q)`, where the existing `crossEntropy` backward is
`(p - onehot)` (`src/model/autograd.ts`), reusing its ignore-index masking. Phase B data is fixed,
so teacher logits are precomputed once over the SFT `.tokens` with the Phase A checkpoint and stored
top-k (~100 B/token at top-16, single-digit GB): no second model in GPU memory, zero per-step
teacher cost. CPU reference + WGSL port + gradient check follow the existing op pattern and need no
GPU, so it is buildable while Phase A trains. The same op later enables Cursor-style hint
self-distillation (teacher = same weights with an instruct prefix, student = bare, KL on the
response tokens); an online teacher forward adds roughly a third of a step (forward-only,
attention-bound), affordable at this scale.

**The op shipped (2026-08-04): `softCrossEntropy(logits, teacherIds, teacherProbs, k)`.** Sparse
teacher, `[T*k]` ids + probs per row; a row is ignored when its first id is negative, the same
convention `crossEntropy` uses, so assistant-only masking carries over unchanged. The teacher mass
need not be normalized: with top-k truncation it sums to `S <= 1` and the exact gradient is
`S*p - q`, which is the documented `(p - q)` when `S = 1`. The reported value is a cross-entropy in
nats (it differs from `KL(q||p)` by the teacher's entropy, a constant in the student's parameters),
so it is directly comparable to the hard-target loss. WGSL: `srcSoftCeFwd` reuses the `crossEntropy`
row-softmax and shares `srcCeReduce`; the backward is two ordered dispatches, dense `+S*p` over
`[T,V]` then sparse `-q` over the k entries, the sparse one **one thread per row** so duplicate
teacher ids in a row cannot race. Verified by `tests/gradcheck.ts` (finite differences, covering a
normalized row, a truncated row, an ignored row and a duplicate id) and `tests/gpu-parity.ts` (GPU
vs CPU, same four cases).

Not built yet: the teacher-logit precompute pass over the SFT `.tokens` and the SFT loop that sums
`crossEntropy + lambda * softCrossEntropy`. Note the composed form costs two row-softmaxes and two
`[T,V]` probability buffers (268 MB each at seq 2048, 1.07 GB at seq 8192); if that ever binds, fuse
both targets into a single op sharing one softmax.

## Explicitly not worth doing

- **Guarded/clamped f16 compute**, no speed to recover; attention-bound, GEMM is a thin slice. f32
  also learns better. (Confirmed; f16-compute path removed.)
- **Using all 128 GB**, we are compute-bound; extra RAM buys nothing at this model size. It only
  matters as headroom for a much larger model, which the throughput ceiling makes impractical
  anyway.
- **Switching to PyTorch + ROCm for speed**: the attention wall is the GPU, not the framework; ROCm
  on gfx1151 also lacks flash-attention. See the project direction: portability is the goal, not
  chasing SOTA on this hardware.
- **Soft-label KD from off-the-shelf teachers** (e.g. Gemma-3-1B as teacher): vocab mismatch (our
  custom u16 BPE vs their 262k tokenizer), and cross-tokenizer KD (GOLD/ULD) is research-grade
  machinery. Data-level KD is already the corpus strategy (TinyStories and smol-smoltalk are
  teacher-generated); same-vocab distillation stays available via the checkpoint anchor (#10).

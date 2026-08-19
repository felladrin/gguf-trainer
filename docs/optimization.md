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

> The attention, GEMM and cross-entropy kernels were rewritten on 2026-08-18 (see lever 1 below);
> every row above predates that and describes the old kernels. The shape of the conclusions holds,
> the absolute numbers do not. Re-measure on Strix with `bench` before quoting them.

Two facts follow from this and drive everything below:

1. **We are compute-bound on attention, not memory-bound.** The GPU is pinned at 100% while only
   ~30% of RAM is in use. More memory or a bigger batch cannot raise tokens/second: only cheaper
   attention can.
2. **Batch is sequential gradient accumulation, not a real batch dimension.** The training loop runs
   one sequence per forward/backward and sums the gradients (`train-gpu.ts`), so batch size trades
   step count for per-step time at a fixed tokens/second. It changes gradient noise, not throughput.

## Throughput levers (the binding constraint)

### 1. Attention kernel: the ~78% slice: PARTLY RESOLVED (2026-08-18), 2.4x end-to-end on M1 Max

The 2026-07-08 reading of this line was that attention sat at a hardware floor. It did not; it sat
at a _codegen_ floor. Three restructurings had been tried and reverted (f16, split-K, query-register
tiling), and their failure was read as "the kernel is done". None of them touched the thing that was
actually costing the time: **every load in the head-dimension loops was scalar**, and the private
arrays holding a thread's Q row and output accumulator were f32 arrays indexed by a loop variable,
exactly the pattern that made the rolled GEMM 3x slower than the unrolled one.

Measured on an M1 Max with the new `bench` subcommand, min of 4 runs x 8 iterations, 95M geometry
(T=2048, 10 query heads over 5 KV heads, head-dim 64), GPU time per kernel:

| kernel                       |   before |   after | change |
| :--------------------------- | -------: | ------: | -----: |
| `srcAttnBwdDkv`, window 1024 | 225.7 ms | 55.2 ms |  4.09x |
| `srcAttnBwdDkv`, dense       | 376.2 ms | 89.7 ms |  4.19x |
| `srcAttnFwd`, window 1024    |  44.2 ms | 22.6 ms |  1.96x |
| `srcAttnFwd`, dense          |  81.3 ms | 35.6 ms |  2.28x |
| `srcAttnBwdDq`, window 1024  |  46.0 ms | 30.8 ms |  1.49x |
| `srcAttnBwdDq`, dense        |  84.8 ms | 50.3 ms |  1.69x |

The first correction the measurement made was to the target itself: **`srcAttnBwdDkv` was ~70% of
the attention slice**, not the forward kernel. The 78%-attention figure above was never broken out
per kernel, so the optimization effort had been aimed at the wrong one of the four.

Three changes, all plain portable WGSL, no intrinsics:

1. **vec4 lanes.** Q/K/V/output rows are addressed as `vec4<f32>` whenever the head dim is a
   multiple of 4, so a head-deep step is 1 load instead of 4 and the dot product carries 4
   independent accumulation chains instead of 1: the same two levers that lifted `srcGemm`. Head
   dims that are not a multiple of 4 keep the scalar form from the same codegen.
2. **exp2 domain.** The online softmax runs in log2 with the score scale pre-folded, and LSE is
   stored in the same domain so both backward kernels pick it up. `exp` is not the hardware
   instruction anywhere; `exp2` is.
3. **Conditional rescale** in the forward: the running max is monotone, so the accumulator is only
   rescaled on the keys that actually raise it, and on that branch `p` is exactly 1.

Plus two changes to `srcAttnBwdDkv` alone: a two-chain unroll of the head loop (1.18-1.20x there,
neutral in the other two, which is why it is not applied to them), and a query tile sized to fit the
16 KiB portable workgroup-storage floor. The old fixed 32-row tile needed 16640 B at head-dim 64,
over the floor, so the head size every published checkpoint uses was over budget on any
implementation that validates the limit. Halving it to 16 rows fits in 8320 B and is also 1.18x
FASTER on M1 Max: smaller tiles buy occupancy. (Two separate 1.18x factors compound into the
table's 4.09x, this one and the two-chain unroll above.)

That the bug had never been hit is not luck, and not a missing parity shape either: an hd=64 parity
case would have passed. What kept it working was the absence of any check. Probed on an M1 Max,
Deno's wgpu accepted a 16640-byte shader on a device reporting a granted
`maxComputeWorkgroupStorageSize` of **16384**, with an empty validation scope. The kernel was over
budget on every device it ever ran on, and was simply never told.

That is also why the gate is `tests/kernel-limits.ts` and not a GPU test. It parses the emitted WGSL
and asserts no kernel `wgsl.ts` builds declares more workgroup storage than the 16 KiB floor, across
every head dim the trainer accepts. A GPU test would have passed on the runtime the trainer actually
uses while the shape still failed for anyone on a stack that does validate. The check is also
independent of the fix by construction: `attnBwdTile` sizes the tile from a formula, the test sums
the array declarations, and neither consults the other.

End-to-end, 8 real pretrain steps at the 95M geometry, seq 2048, batch 2, on the M1 Max:
**204 -> 490 tokens/s (2.40x)**, with the per-step loss identical to four decimals at every step and
identical peak memory.

NOT YET MEASURED ON STRIX: the numbers above are M1 Max (Metal). The changes are machine-independent
in kind (fewer instructions for the same arithmetic, no vendor path), but the _share_ each one buys
depends on the silicon, and the Strix box was mid-run. Re-run `bench` there and record the numbers
before quoting a speedup for the AMD target.

Still standing, and still a research effort: the O(T^2) pair count itself. CONSTRAINT unchanged:
only approaches that stay portable, plain WGSL that runs cross-vendor (AMD/Apple/NVIDIA), no vendor
intrinsics or hardware-specific paths. A kernel that only helps gfx1151 at the cost of the "runs
anywhere" story is out of scope.

### 1b. GEMM and cross-entropy kernels (2026-08-18)

The same vec4 audit applied to the other two kernel families, measured the same way:

| kernel                                          |   before |    after | change |
| :---------------------------------------------- | -------: | -------: | -----: |
| `srcGemm`, tied readout [2048,640]x[32768,640]T | 337.7 ms | 183.0 ms |  1.85x |
| `srcGemm`, FFN up [2048,640]x[2560,640]T        |  19.2 ms |  15.8 ms |  1.22x |
| `srcGemm`, QKV [2048,640]x[1280,640]T           |  10.2 ms |   9.8 ms |  1.04x |
| cross-entropy fwd+bwd, T=2048 V=32768           |  73.7 ms |   4.7 ms |  15.7x |

GEMM: both staged tiles are held as vec4 and both micro-tile fragments are read as vec4, which
needs the A tile staged transposed (`As[k][m]`), because in the old `As[m][k]` layout a thread's
four rows were BK apart and a strided fragment cannot be one load. BK went 8 -> 16 (measured 1.09x
on top of vec4 at half the 16 KiB portable workgroup-storage floor; BK=32 was no faster and spends
the whole floor).

Cross-entropy: the old kernel ran **one thread per row**, so a [2048, 32768] softmax executed on
2048 threads with each lane striding a full row apart, coalescing nothing. It is now one workgroup
of 256 per row with a workgroup reduction, which is both 128x the parallelism and coalesced. On top
of that, `PROBS` now holds unnormalized `exp(z - max)` with a per-row `1/sum` beside it: the old
third pass over `[T,V]` existed only to apply a divide that the backward can apply from a scalar.
That also removes the `+1e-12` the loss needed to survive `log(0)`, since the loss is now
`log(sum) - (z_target - max)` straight from the row statistics.

Bind-group caching for the main loop (the `prepareDispatch` pattern the optimizers use) was measured
and NOT done: instrumenting `createBindGroup` over a real run gives 2346 calls and 71.5 ms per step
against an ~18 s step, i.e. 0.4%. It is also the exact pattern `prepareDispatch` warns about, since
the main loop's buffers are pooled transients that get recycled. Revisit only if the per-step GPU
work drops by an order of magnitude.

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

### 4. More unique data: the binding constraint on quality (revised 2026-08-19)

The corpus is 722M unique tokens; the run does 2 epochs (~1.44B). `corpus` can emit more parts for
near-zero cost, but training them is the expense. Stated in tokens per parameter, the axis the
small-model literature argues on, against the models in the head-to-head table under lever 9:

| model                      | params | train tokens | tokens/param |
| :------------------------- | -----: | -----------: | -----------: |
| ours, `phaseA-final-88000` |  94.7M |        1.44B |           15 |
| Minueza-2-96M              |    96M |         185B |        1,927 |
| SmolLM2-135M               |   135M |          ~2T |      ~14,800 |

15 tokens per parameter is Chinchilla-optimal (~20:1), and therefore optimal for nothing this
project wants: Chinchilla minimizes loss for a fixed _training_ budget, not quality per parameter at
a fixed _model size_. Every model that beats us above trained two to three orders of magnitude
longer per parameter.

Throughput is 1588 tok/s sustained after the 2026-08-18 kernel rewrite (Strix, seq 2048 batch 8,
instantaneous rate over the roleplay run), i.e. **137M tokens/day**, up from the 70M this section
used to assume. The arithmetic is still discouraging:

| target                         | total tokens | still to train | days at 137M/day |
| :----------------------------- | -----------: | -------------: | ---------------: |
| 100 tokens/param               |         9.5B |           8.0B |               58 |
| Minueza-2's 1,927 tokens/param |         185B |           184B |    1,340 (3.7 y) |
| SmolLM2's ~14,800 tokens/param |         1.4T |           1.4T |        ~28 years |

So closing the gap to Minueza-2 is not "a compute-time decision" as this section previously called
it: on one box it is out of reach. Reaching ~100 tokens/param is two months and is the only rung on
this ladder actually available. That, not architecture and not hyperparameters, is the ceiling on
quality here.

There is also a far end to this, which is worth knowing about even though it is not the end we are
near. [Extreme overtraining in tiny language models](https://huggingface.co/blog/Banaxi-Tech/ovdadadadd)
(2026-08-12) reports a 0.9M-param model whose scores peak around 22,000 tokens/param and decline
from there out to 222,000. That measurement is at 0.9M params, so the ratio itself does not carry to
94.7M (it would imply 2.1T tokens), and it is a single external run rather than something measured
here. The direction is what matters, and it agrees with SmolLM2: the useful ratio for small models
sits orders of magnitude above Chinchilla, with an eventual point of diminishing returns. We are at
the opposite end of that range by more than a factor of a thousand.

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

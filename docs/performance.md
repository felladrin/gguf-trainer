# Performance

Throughput, memory, and context: what has been measured on this trainer, what each flag costs, and
what has already been tried and rejected. Numbers come from real runs on the hardware named beside
them. Where a number is superseded, it is replaced rather than annotated; the history is in git.

Read [correctness.md](correctness.md) for the guards, [design.md](design.md) for why the engine is
shaped this way, and [evaluation.md](evaluation.md) for model quality, which is a different axis
from everything here.

## The reference machine

All unlabelled numbers are from a Strix Halo APU (AMD Radeon, RADV GFX1151, 128 GB unified),
training a 94.7M-param `gemma3` at seq 2048, batch 8, after the 2026-08-18 kernel rewrite.

| Metric                  | Value                     | How measured                                   |
| :---------------------- | :------------------------ | :--------------------------------------------- |
| Throughput              | 1588 tok/s                | plateau rate, against 903 tok/s pre-rewrite    |
| Step time               | 0.0969 st/s (10.3 s/step) | same run, at 16384 tokens per step             |
| Host RSS                | 1.06 GB steady            | sampled every 10 s over 150 steps              |
| Peak GPU (pool + state) | 39.3 GB                   | the trainer's own readout                      |
| Profiled kernel time    | ~330 ms of a 10.3 s step  | `bench`, idle GPU, summed over a step's shapes |

1588 tok/s is 137M tokens/day, so the published model's 1.95B tokens is about 14 days.

## Measuring on a shared box

**Whatever else runs on this machine is worth 2x on the step.** Same binary, same shape, same flags,
same seed:

| arm           | machine otherwise idle | with other work running |
| :------------ | ---------------------: | ----------------------: |
| dense         |              113 tok/s |         54 and 58 tok/s |
| `--recompute` |              248 tok/s |               117 tok/s |

The factor is not constant, which is what makes it dangerous: three write-ups of one measurement
reported +16%, -32% and +57% for the same one-line change, because the arms were interleaved with
other work.
Repeating a measurement under the same load reproduces the same wrong number.

Three rules follow, and they are not optional:

1. **Record what else held the CPU and the render node**, at the start and the end of the run:
   `fuser -v /dev/dri/renderD128`. A number without that record is unusable. This machine routinely
   carries a compositor, an idle `llama-server`, and other agent sessions.
2. **`gpu_busy_percent` is a whole-device counter**, so it is a ceiling on this trainer's share and
   never a measure of it. Sampled at 10 Hz for 30 s with nothing training at all it reads mean
   43.2%, median 17%, max 99%: it spans its range, so a high reading attributes to nobody. One
   calibration read 99.8% over 200 samples with the trainer not running, and `fuser` showed another
   session holding the render node.
3. **`pretrain` A/Bs come from the trainer's own output**, not from a shared counter, and want
   several independent builds to agree. Three builds of the dense arm gave 114, 110 and 113 tok/s,
   a 3.6% spread, which is the resolution an A/B on this box actually has.

`bench` numbers are min-of-4 on an idle GPU for the same reason. An A/B of a GEMM tile taken under
contention showed a clean 1.07x that reversed direction once the GPU was idle.

The runs behind the training and eval numbers here are in
[measurements/](measurements/), raw rather than summarized, so a number can be recomputed instead of
trusted. Anything written to `out/` is gitignored and therefore unverifiable to anyone who was not
at the machine, which is why those copies exist.

## Where the step goes

**The step is host-bound.** Profiled kernel time is ~330 ms against a step of 10.3 s, and splitting
the phases inside the micro-batch loop says the same:

| phase                                        |           time |
| :------------------------------------------- | -------------: |
| `model.forward()`, per micro-batch, x8       | 1400 - 1650 ms |
| end-of-step `gpu.sync()` (all GPU execution) |        1150 ms |
| `backward()`, per micro-batch, x8            |     11 - 20 ms |
| optimizer (Muon + AdamW, all 28 tensors)     |           5 ms |

`model.forward()` is host time spent recording dispatches, and it is not one op: the cost tracks
tensor size across all of them. Cheaper kernels can no longer raise tokens/second on this box; only
cheaper host work can. Anyone taking this on should profile bind-group and pipeline setup per
dispatch.

**Host allocation is not the cause, which cost a day to establish.** Every `Tensor` used to allocate
two full-size host `Float32Array`s, 4.34 GB per forward across 244 tensors. Making them lazy removed
98.9% of that (4,395 MB per forward to 48 MB) and moved throughput by less than 1% (2,651 against
2,627 tok/s over 150 steps), with steady-state RSS identical at 1.06 GB. V8's young generation
absorbs that allocation rate for free.

Those 2,651 tok/s are the same shape as the 1588 above, 1.67x apart, and the gap is **not
attributed**. The likeliest explanation is that the long run shared the GPU with the benchmarking in
this file, which is the contention the section above exists for. Treat the 10.3 s/step baseline as an
upper bound until someone settles it.

**Batch is sequential gradient accumulation, not a batch dimension.** One sequence per
forward/backward, gradients summed, so batch trades step count for per-step time at fixed
tokens/second. It changes gradient noise, not throughput.

## The memory and context flags

Each was measured as an A/B at one shape, same seed, with the loss compared digit for digit. Peak
GPU is the trainer's own `pool + state` readout.

| flag                | shape measured                                      | throughput         | peak GPU          |
| :------------------ | :-------------------------------------------------- | :----------------- | :---------------- |
| `--recompute`       | qwen3 293M, seq 2048, batch 2, reclaim + loss-chunk | 72 -> 168 tok/s    | 18974 -> 7658 MB  |
| `--lora-rank 16`    | qwen3 293M, seq 1024, batch 1, all three flags on   | 43 -> 37 tok/s     | 12838 -> 7554 MB  |
| `--reclaim`         | gemma3 94.7M, seq 2048, batch 8                     | 1341 -> 1031 tok/s | 39.3 -> 7.0 GB    |
| `--loss-chunk 8192` | qwen3 293M vocab 151936, seq 2048, batch 2          | 73 -> 73 tok/s     | 22548 -> 18881 MB |

**`--recompute` is the first thing to reach for.** It replays each layer in backward instead of
keeping its interior, so the activation term falls from ~12.1 GB to ~0.8 GB (4.6x on the whole
pool). It also came out 2.3x FASTER here, which is the surprising part and is an artifact of the
host-bound step: `endRegion` submits at every layer boundary, so layer 1 executes while the host
records layer 5. **A GPU-bound shape should expect the textbook ~30% slowdown instead.** The
`submit()` in `endRegion` is what produces the overlap, not correctness; deleting it as redundant
keeps the whole suite green and silently returns the throughput to the dense number.

**`--lora-rank N` is for when the optimizer state is what hurts.** It freezes everything the model
had and trains rank-N adapters on the Muon group. Optimizer state goes 4495 MB to 63 MB, 71x, which
is the whole story: that term scales purely with the trainable count. Throughput drops 14% because
each adapted projection becomes three matmuls. The export is an ordinary dense checkpoint, because
adapters are folded into the base on the way out and back out afterwards, and a LoRA run neither
reads nor writes an optimizer sidecar.

**`--reclaim` is for when a run does not fit.** It frees each micro-batch's activations at the
micro-batch boundary, 5.6x less peak for 23% less throughput. The cost is the drain:
`reclaimStepTransients` submits and then AWAITS `onSubmittedWorkDone`, which stalls the host until
the GPU catches up. Submitting is the overlap; waiting for the submission is the stall, which is why
this costs 23% where `--recompute` gains 2.3x on the same host-bound step. Off by default, and a
no-op at `--batch 1`.

**`--loss-chunk N` buys context, not memory.** It fuses the readout matmul into the loss and streams
the vocab N columns at a time, so no buffer scales with context and vocab together. The 3.7 GB it
saves is real but secondary; what it unlocks is that a 151936-vocab model at seq 4096 needs a
2374 MiB logits buffer against a 2048 MiB device limit, so it did not train at any amount of free
memory. With the flag it trains at 74 tok/s. Costing nothing measurable in throughput is again the
host-bound step: the extra readout matmul lands in a gap the GPU was already idle for.

Context is capped by the logits buffer (`seq-len x vocab x 4` bytes), not by attention: attention
takes a flash path from seq 2048 up and allocates no `[heads, T, T]` buffer there. At vocab 32768,
seq 8192 needs 1 GiB and works on an adapter that grants its full buffer size; one that falls back to
the WebGPU default of 128 MiB stops at 1024. `--loss-chunk` is capped at 100 spans, because the
vocab offset is baked into the kernel source and each span costs its own pipelines.

Long context costs throughput rather than memory at equal tokens/step: seq 8192 runs ~28% fewer
tok/s than seq 2048, because sliding-window layers cover 5 of 6 and the global layers still pay
O(T^2).

## The kernel rewrite

2026-08-18, and it is where the 1.76x on Strix (2.40x on an M1 Max) came from. Three changes, all
plain portable WGSL:

1. **vec4 lanes.** Q/K/V/output rows are addressed as `vec4<f32>` wherever the head dim allows, so a
   head-deep step is one load instead of four and the dot product carries four independent
   accumulation chains instead of one.
2. **exp2 domain.** The online softmax runs in log2 with the score scale pre-folded. `exp` is not a
   hardware instruction anywhere; `exp2` is.
3. **Conditional rescale.** The running max is monotone, so the accumulator is rescaled only on keys
   that raise it.

Per-kernel on an M1 Max at the 95M geometry, `srcAttnBwdDkv` went 4.09x, `srcAttnFwd` 1.96x,
`srcAttnBwdDq` 1.49x. The GEMM audit gave 1.85x on the tied readout, and cross-entropy 15.7x: the
old kernel ran **one thread per row**, so a [2048, 32768] softmax executed on 2048 threads
coalescing nothing. It is one workgroup of 256 per row with a workgroup reduction now.

Two corrections the measurement made, both worth keeping: `srcAttnBwdDkv` was ~70% of the attention
slice, not the forward kernel the analysis had aimed at, and the ranking inverted afterwards. The
tied readout GEMM is now 4.5x the largest attention kernel, so "attention is ~78% of runtime" no
longer describes this trainer.

**Workgroup storage is sized against the 16 KiB WebGPU floor, never against what a device grants.**
The old fixed 32-row attention tile needed 16640 B at head dim 64, over the floor, on every device
it ever ran on: Deno's wgpu accepted it with an empty validation scope, so nothing ever said so.
Halving the tile fits in 8320 B and is 1.18x faster besides. `tests/kernel-limits.ts` parses the
emitted WGSL and holds every kernel under the floor across every head dim the trainer accepts,
because a GPU test would pass on the one runtime it ran on while the shape still failed for everyone
on a stack that validates.

## Ruled out (measured, do not re-tread)

| Idea                                  | Outcome                                                                                                       |
| :------------------------------------ | :------------------------------------------------------------------------------------------------------------ |
| f16 compute (f16 mul, f32 accum)      | 0.98x on attention at seq 4096-8192, plus overflow to NaN at step 2400 without clamps. f32 also learns better |
| f16 storage for Q/K/V                 | 1.02-1.06x, and the gain shrinks as context grows                                                             |
| Split-K attention (32 threads/row)    | 0.4-0.7x: destroys the wave-uniform K/V broadcast                                                             |
| Query-register tiling                 | 0.80-0.94x at QT=2, 0.48-0.68x at QT=3: register pressure halves occupancy faster than reuse pays             |
| GEMM tile 128/128/16/8/8              | 0.93x on an idle GPU. The 1.07x that looked like a win was measured under contention                          |
| Bind-group caching in the main loop   | 2346 calls and 71.5 ms per step against an ~18 s step: 0.4%, and the buffers are pooled transients            |
| Lazy host `Tensor` storage            | Removes 98.9% of host allocation and moves throughput <1%                                                     |
| `submit()` at layer boundaries, dense | 117 against 114, 110, 113 tok/s on an idle machine: inside the spread between builds                          |
| WMMA / subgroup matrix                | Not exposed by Deno's wgpu on gfx1151 (15 features probed, no `subgroups`); no WGSL matrix ops in the spec    |
| bf16 of any kind                      | No `bf16` WGSL type, no `shader-bf16` feature, none in naga                                                   |
| Fixed-max softmax via a QK-norm bound | No safe static bound: the norm weights are trained, and the observed max runs 3.3-4.4x the proxy              |
| Per-lane sliding-window start         | Destroys the wave broadcast; the block-aligned start is why SWA is not slower than full attention             |
| Using all 128 GB                      | The trainer is throughput-bound at this model size; extra RAM buys nothing                                    |

**Every kernel here is plain portable WGSL**, no intrinsics and no vendor paths. What differs across
machines is the share of the bottleneck each change addresses: the exp2 rescale matters most where
the SFU is narrow, vectorized loads where load issue is the constraint, register tiling where the
FMA latency chain dominates. A kernel that only helps gfx1151 at the cost of running everywhere is
out of scope, which is also why PyTorch + ROCm is not the answer to the throughput gap.

## Still open

- **The O(T^2) pair count itself**, within the portable-WGSL constraint.
- **Where the dispatch-path host time actually goes.** This is the binding constraint and the one
  worth taking.
- **2D workgroup tiling for attention.** A staged-forward variant measured 17% slower.
- **Newton-Schulz at four iterations instead of five**, which needs an orthogonality-residual check
  to gate it.
- Parked while the step is host-bound: `srcEmbeddingBwd` scaling (~0.3% of the step at vocab 32768,
  multi-percent at 2x that), `srcRmsNormBwdW`'s ~16x overfetch (~1% of the step), a RoPE table
  precompute, and sliding-window warmup (~2-3% at seq 2048).

## References

- RDNA3.5 (gfx1151) speed-of-light rates, VOPD: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/rdna/system-speed-of-light.html)
- RDNA SFU/TFU, `v_exp_f32`, LDS bandwidth, Wave32: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- SFU quarter-rate, RDNA2, applied by class: [nelcit.github.io](https://nelcit.github.io/shader-clippy/blog/pow-const-squared)
- WebGPU limit defaults, including the 16 KiB workgroup-storage floor: [developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)
- WGSL spec (`exp2`, atomics, workgroup memory): [w3.org](https://www.w3.org/TR/WGSL)
- Bind-group reuse guidance: [toji.dev](https://toji.dev/webgpu-best-practices)
- Fused linear-cross-entropy prior art, CUDA/Triton, the shape `--loss-chunk` follows:
  [github.com](https://github.com/linkedin/Liger-Kernel), [github.com](https://github.com/mgmalek/efficient_cross_entropy)
- FlashAttention-3, for what full 2D tiling plus tensor cores buys where they exist:
  [arxiv.org](https://arxiv.org/abs/2407.08608)

# Continual pre-training (adding tokens to a finished base)

> Written during the Minueza-3 run, before the CLI existed. Paths like `examples/…` name that run's
> artifacts and scripts; the equivalent commands are in [agents.md](../../agents.md).

Can the engine keep pre-training an already-trained checkpoint on more/new data? **Yes**, it is
built into `pretrain` and already exercised by `resume-phase-a.sh`. This note records how, the hard
constraints, and the compute reality, so we don't rediscover it. Written 2026-07-29, against the
Phase A run (94.7M gemma3 base, hidden 640 × 12 layers, vocab 32768).

## What the engine gives you

| Capability                                           | Flag / mechanism                                                                  | `pretrain.ts` |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- | ------------- |
| Warm-resume weights from a checkpoint                | `--resume=PATH.gguf` → `loadWeightsFromGGUF`, guarded by `configMatches`          | ~314-320      |
| Warm-resume optimizer (Muon momentum + Adam moments) | `<resume>.optstate` sidecar → `opt.importState`; absent ⇒ cold-start              | ~351-359      |
| Continue the step counter + LR schedule              | `--startStep=N` offsets into the WSD schedule                                     | ~246, 368     |
| Train on a different corpus                          | positional `.tokens` arg + its sibling `<prefix>.tokenizer.json`                  | ~274-277      |
| Mix a second stream during decay                     | `--inject`, `--injectFrac`, `--injectFrom` (MiniCPM / Xmodel-2 decay-phase trick) | ~378-389      |

So "load the finished checkpoint, keep training on new tokens" is a launch command, not a code
change.

## Hard constraints

1. **The tokenizer/vocab is frozen at 32768.** The embedding table trained against it, so new data
   MUST be encoded with the _same_ `data/blend.tokenizer.json`. You cannot introduce new tokens. The
   `configMatches` guard (`vocabSize` among the checked keys) aborts a resume whose architecture
   doesn't match the checkpoint.
2. **`tokenize` reuses an existing tokenizer if one already sits at the output prefix**: copy the
   frozen tokenizer to the new prefix _before_ running it, or it will train a fresh (incompatible)
   vocab.
3. **The WSD schedule is anchored to `steps`; `--startStep` only offsets into it.** There is no
   first-class "new LR cycle on new data" mode: you drive it via `steps` / `startStep` (see below).
4. **No in-loop validation.** `eval_val.ts` / `eval_mc.ts` exist but are not wired into the training
   loop, so overfitting from heavy epoch repetition is not caught automatically. Watch it by hand on
   a held-out slice at checkpoints.

## Recipe (fresh WSD cycle on new data)

```sh
# 1. A new raw corpus (each part under ~480 MB). One source, not the blend,
#    when the point is to isolate that corpus's effect:
deno run -A cli.ts corpus --source HuggingFaceFW/fineweb-edu:sample-10BT:text \
  --size-mb 450 --parts 5 --out corpus/fineweb-edu.txt

# 2. Reuse the FROZEN vocab: copy the tokenizer to the new prefix, so `tokenize`
#    loads it instead of training a new, incompatible one.
cp data/blend.tokenizer.json data/cpt.tokenizer.json

# 3. Tokenize against that frozen vocab (--vocab is ignored when the tokenizer
#    file already exists at the prefix).
deno run -A cli.ts tokenize --text corpus/fineweb-edu-p1.txt,corpus/fineweb-edu-p2.txt \
  --out data/cpt --vocab 32768 --curriculum-specials

# 4. A fresh WSD cycle from the finished checkpoint. --start-step 0 re-warms the
#    LR from trained weights and cools down at the end of the NEW step count,
#    which is the clean continual-pretraining shape.
deno run -A cli.ts pretrain --data data/cpt.tokens --out out/base-cpt.gguf \
  --resume out/base.gguf --start-step 0 \
  --hidden 640 --layers 12 --head-dim 64 --window 1024 --max-seq 8192 \
  --steps <NEW_STEPS> --seq-len 2048 --batch 8 --lr 0.01 --checkpoint-every 500
```

Notes:

- Momentum: with `--start-step 0`, if `<resume>.gguf.optstate` sits next to the resume file it is
  loaded (momentum carried, harmless and mildly helpful). To start the optimizer cold instead, pass
  `--cold-optimizer`.
- **Schedule gotcha: do NOT** just bump `steps` to millions while keeping
  `--start-step <last ckpt>`. Warmup/cooldown are recomputed as 10% / 20% of the _new_ length, so
  the current step can land back inside the new warmup window and the LR jumps down, then re-ramps.
  Either finish the current plan (`--start-step <last>` with the original `--steps`) or start a
  fresh cycle (`startStep=0`, new `steps`). Don't blend the two.
- **Big-shard datasets need range reads.** FineWeb-Edu ships 2.15 GB parquet shards.
  `prepare_pretrain.ts` used to pull a whole shard into one buffer, which fails on those with a
  misleading `parquet file invalid (footer != PAR1)`. It now streams row batches over HTTP range
  with hyparquet's `asyncBufferFromUrl`, fetching only the text column and stopping when the part's
  budget is met: 2.27 GB of text out of 2.15 GB shards took 133 s.

## Long-context extension: measured, then stopped (2026-08-05)

Phase A trained entirely at seq 2048 while declaring ctx 8192, so the obvious follow-up was a
seq-8192 extension pass. It ran for 315 steps and was killed. What it taught, in order:

1. **Length alone is the wrong filter.** `examples/filter_long_docs.ts` pulls the long documents out
   of a pretokenized corpus. Filtered only by length (>4096 tokens), the blend yields 17903
   documents / 190.4M tokens, and the model's loss on that slice is **1.55** against 2.89 on
   ordinary 8192-token windows. The slice is boilerplate: 24% of those documents are degenerate
   (more than half their 4-grams already seen, or under 0.08 distinct tokens per token) and they
   carry 38% of the tokens. Training there memorizes link farms.
2. **With a repetition gate it becomes real material.** Same threshold plus the gate: 13647
   documents / 114.5M tokens, mean document 8387 tokens (so an 8192-token window spans 0.98
   documents: the point of filtering), and step-0 loss **2.87**, in line with general windows.
3. **The gain is real but tiny.** 300 steps (9.8M tokens, seq 8192 × batch 4, muon lr 0.002, 25%
   general blend injected via `--inject` as drift insurance) moved the fixed 32-window validation
   set from 2.8918 to **2.8783** at seq 8192 (ppl 18.03 → 17.78) and left seq 2048 flat (2.8275 →
   2.8317). Direction exactly as designed: long positions improve, short ones untouched.
4. **Killed because 39M more tokens meant ~24 h of GPU for maybe 0.05 nats**, and the goal had
   shifted to HellaSwag, which is made of sub-200-token items and cannot move from long-context work
   at all. Checkpoint kept at `examples/phaseA-ctx8192.gguf` (+ `.optstate`), so it is resumable.

Caveat kept on purpose: at lr 0.002 over only 9.8M tokens, "long context has little left to learn"
and "the LR was too low to show" both fit the data. The measurement says the pass was not worth its
price, not that extension is worthless.

## The Supra2 target and the token-slope experiment (2026-08-05)

`SupraLabs/Supra2-100M-Base` scores **35.31** HellaSwag on our harness (see OPTIMIZATION note 9)
with 100.7M params and **30B** training tokens of FineWeb-Edu + DCLM. We score 28.46 with 1.44B. The
gap is 6.85 points, far outside the ±0.5 bars.

Volume alone does not explain it, and that is the useful part: Minueza-2-96M saw **185B** tokens
(128× ours) and scores 27.03, BELOW us; Supra2 used 6× fewer tokens than Minueza and beat it by 8
points. At this scale data quality dominates data volume, so the question is not "can we afford 30B
tokens" (we cannot: 78M tokens/day at 903 tok/s means ~1 year) but "how many points does 100M tokens
of FineWeb-Edu actually buy in this recipe".

That is a measurement, so it is being measured rather than argued:

- Corpus: FineWeb-Edu `sample-10BT`, 5 × 450 MB → **508.7M tokens** over 473053 documents, encoded
  with the FROZEN blend vocab (`examples/fineweb.tokens`); pretokenization took 1173.9 s.
- Run: fresh WSD cycle from `phaseA-final-88000.gguf`, seq 2048 × batch 8, muon lr 0.01 (Phase A's
  peak), aux lr 3e-3, **31000 steps = exactly 1.0 epoch** ≈ 6.5 days. Not 33000: at the corpus's
  real size that would spill 32M tokens into a second pass, which is repetition, not signal. Step-0
  loss is **3.29**: higher than the 2.83 we measure on the blend because this is a different
  distribution, so only the slope from here is meaningful. Managed by `stop-phase-b.sh` /
  `resume-phase-b.sh` (same shape as the Phase A pair): stop loses at most the steps since the last
  `[ckpt @ N]`, and resume re-detects that step from the log, keeps `steps=31000` so the WSD shape
  is unchanged, and brings the curve watcher back with it. `DRY_RUN=1 bash resume-phase-b.sh` prints
  the command without launching.
- Measurement: `scripts/minueza-3-run/hellaswag-curve.sh` scores the FULL 10042-task HellaSwag at
  every checkpoint, on CPU with few threads so it barely touches the GPU run, and appends to
  `evaldata/hellaswag_curve.csv`. The full set, not a sample: a 2000-task sample's ±1.05 would swamp
  the ~0.25-point steps being resolved. The baseline point is already known: 28.4605.
- Reading it: ~0.5 points per 100M tokens puts the 6.85-point gap at ~1.4B tokens (~18 days), which
  is a decision worth having. ~0.1 points per 100M says this box cannot close it by volume, and the
  honest framing becomes token efficiency (28.46 at 1.44B against Minueza-2's 27.03 at 185B) plus a
  recipe change.

### Checkpoint spacing: 310, not 3100 (2026-08-06)

The run started at `--ckpt=3100` (~7.7 h of expected loss on a kill) and was switched to
**`--ckpt=310`** live, via `scripts/restart_after_ckpt.sh 6200`: it waits for a given `[ckpt @ N]`
line, then stops and immediately resumes, so the flag changes at a cost of seconds instead of an
interval. A checkpoint write measures ~4.1 min, so 310 costs 4.4% throughput and caps an OOM kill's
damage at ~46 min on average. The motivation is concrete: running llama.cpp on the same box
occasionally OOM-kills the trainer (RSS 34.7 GB, `oom_score` 754), and the fix on the inference side
is `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 -- choom -n 1000 --` plus
bounding `-ngl` / `-c` at the source (GTT device buffers are pinned, so `--mmap` does not protect
against this).

Two consequences worth knowing:

- `hellaswag-curve.sh` gained `CURVE_EVERY` (default 3100) so the extra safety checkpoints cost no
  eval time. The watcher alive during the switch kept its pre-gate body (bash parses the whole
  `while` up front), so the curve from step 6200 on is 10x denser than planned. It costs nothing
  measurable (0.056 st/s before and after) and the density improves the fit, so it was left alone.
- `resume-phase-b.sh` now refuses when the GGUF's mtime is >600 s ahead of its `.optstate` sidecar.
  The trainer renames the GGUF, then the sidecar, then logs, so a kill in that window leaves weights
  ahead of the logged step. `FORCE=1` overrides.

### First slope reading: ~0.22 points per 100M tokens (2026-08-07)

20 points from step 3100 to 11780 (50.8M → 193.0M tokens), least-squares fit:

| Quantity                           | Value                                                    |
| ---------------------------------- | -------------------------------------------------------- |
| Slope                              | **+0.22** HellaSwag points per 100M tokens               |
| r²                                 | 0.20 (every point sits inside its neighbors' ±0.45 bars) |
| Tokens to close the 6.85-point gap | **~3.1B** ≈ 39 days on this box                          |

That lands between the two decision branches sketched above, and closer to the bad one: 3.1B tokens
is 6 epochs of this 508.7M-token corpus, which is repetition rather than signal, so closing the
Supra2 gap by volume is not available here. Two caveats keep this provisional. The fit's r² is 0.20,
so the honest statement is "a small positive slope, not distinguishable from ~0.1 or ~0.35 yet". And
every point so far is mid-run: the WSD cooldown only starts at step 24800, and cooldowns are where
these runs usually book their gains, so the end-of-run number should beat the trend line. Judge the
recipe at 31000, not at the slope.

### Final result: +0.79 HellaSwag for 508.7M tokens (2026-08-12)

The run finished at step 30999 (loss 3.216 → 2.777 over the resumed segment, 911 tok/s). Scored with
the head-to-head flags, so the numbers are directly comparable:

| Metric                     | Phase A (1.44B tokens) | Phase B (1.95B) | Δ     |
| -------------------------- | ---------------------- | --------------- | ----- |
| HellaSwag (10042 tasks)    | 28.46 ±0.45            | **29.25 ±0.45** | +0.79 |
| ARC-Challenge (~300 tasks) | 21.74 ±2.39            | **24.08 ±2.48** | +2.34 |

The ARC delta is noise: that validation set is ~300 items, the unpaired standard error of the
difference is ±3.44, and +2.34 sits well inside it. Don't quote it as a gain.

The HellaSwag gain is real, but the endpoint comparison alone doesn't establish it (+0.79 is 1.2
unpaired standard errors). What carries it is the 76-point curve: a consistent rise from 28.53 at
step 3100 to 29.32 at 28830, with the cooldown slope **quadrupling** over the mid-run one:

| Region                     | Points | Slope                   |
| -------------------------- | ------ | ----------------------- |
| Before cooldown (< 24800)  | 61     | +0.10 / 100M tokens     |
| During cooldown (>= 24800) | 14     | **+0.40 / 100M tokens** |

So the "judge the recipe at 31000" caveat above resolved in the recipe's favor, and the +0.22
mid-run slope was roughly right as an average. The bottom line stays: 508.7M tokens and 13 days of
wall-clock bought +0.79 points (~0.16 per 100M), the Supra2 gap narrowed from 6.85 to 6.06, and
volume on this box remains the wrong lever.

**Register drift is visible in the completions.** `examples/eval-completions.sh` on the final GGUF
shows no repetition loops and clean grammar, but the model answers narrative prompts in expository
voice: "Once upon a time, there was a little" continues into an encyclopedia entry about a village's
population and elevation, and "The sun rose over the mountains and" turns into a bulleted list.
Facts stay wrong ("Water is made of carbon and oxygen", `2 + 2 = 4.8`), as expected at 94.7M. This
is FineWeb-Edu's fingerprint, and it is a plausible reason HellaSwag moved so little: HellaSwag
items are everyday-scenario continuations from ActivityNet and WikiHow, which is the register the
corpus trained _out_ of the model. Worth testing with a mixture change before spending more tokens.

Two operational notes for the next run. The final `[ckpt @ 31000]` never fires: the trainer's
checkpoint condition is `localStep % checkpointEvery === 0` and the loop's last index is
`steps - startStep - 1`, so the curve has no end-of-run point and the final score has to be run by
hand against the post-loop GGUF. And a reboot mid-run (2026-08-11 12:43) cost ~4.5 h of wall-clock
but zero training progress, because the resume came off the intact step-28830 checkpoint, which is
what `--ckpt=310` bought.

**CPU evals steal GPU throughput.** Running the head-to-head at `-t 12` alongside training cost the
training run ~35% of its tokens/s (0.014 st/s against 0.022 measured alone). The curve script uses
`-t 4` for that reason; when timing matters, run the evals after the GPU is free.

## Why this doesn't cheaply reach Minueza-2-96M

Minueza-2-96M (same ~96M params) saw **185B** tokens; Phase A plans ~1.44B. Closing that needs _new_
tokens, and the wall is throughput, not the engine. At the measured `0.055 st/s` and
`batch 8 × seqLen 2048 = 16384 tokens/step` (≈ 885 tokens/s) on the single Strix Halo box:

| New tokens | Steps (÷16384) | Wall-clock, one box |
| ---------- | -------------- | ------------------- |
| 10B        | ~610k          | **~131 days**       |
| 30B        | ~1.83M         | **~390 days**       |

The 722M-token blend can't supply this without dozens of epochs (repetition, not signal), so a real
continual-pretrain needs a _larger_ corpus. The affordable levers are the ones that cut the token
bill: higher-quality data (FineWeb-Edu-grade filtering punches above its token count), or
more/faster hardware, not leaving this box running for a year. This is the "hardware ceiling"
`PHASE_A_HANDOFF.md` names, quantified.

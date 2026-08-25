# Operating manual

You are an agent working in this repository. This file is the manual: what the tool does, how to
drive it, what breaks it, and what the failures mean. Read it before running anything.

## What this is

A trainer that builds a language model **from scratch, in TypeScript, straight to GGUF**. No
PyTorch, no Python, no safetensors. The weights live in a GGUF file from the first step to the last,
so a checkpoint is already a file llama.cpp can load.

Training runs on **WebGPU** (WGSL compute shaders, forward and backward). The CPU backend is a
correctness reference, not a fast path.

Everything is one CLI. There is no web UI, no notebook, no library-only workflow.

```
deno run -A cli.ts help              # every command
deno run -A cli.ts help --json       # the same, machine-readable: parse this, do not scrape help text
deno run -A cli.ts <command> --help  # one command's flags, with defaults and examples
```

## Which document answers what

The rest of the docs are consulted, not read. This is when each one earns opening.

| You are about to                             | Read                                                                                                                                                                      |
| :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| run any command                              | this file: the contract, the invariants, then the recipe                                                                                                                  |
| add a model shape                            | [docs/adding-an-architecture.md](docs/adding-an-architecture.md)                                                                                                          |
| change a kernel or chase throughput          | [docs/optimization.md](docs/optimization.md), lever 1d FIRST: it lists what has already been measured and rejected                                                        |
| understand why the engine is shaped this way | [docs/design.md](docs/design.md)                                                                                                                                          |
| pick a corpus or a token budget              | [docs/optimization.md](docs/optimization.md) levers 4 and 11; for model size, Hardware reality below                                                                      |
| reproduce or question a published number     | [docs/optimization.md](docs/optimization.md) levers 1, 9 and 9b for the number; [docs/notes/](docs/notes/) for the run that produced it, point-in-time and not maintained |
| judge whether this project fits your goal    | [readme.md](readme.md): the results and the honest limits                                                                                                                 |

## Contract

- **Flags only.** No positional arguments anywhere. `--flag value` and `--flag=value` both work,
  booleans are `--flag` / `--no-flag`.
- **Exit codes.** `0` success. `1` usage error: a bad, missing or unknown flag, or an unreadable
  input. `2` runtime failure: the work started and could not finish. An unknown flag suggests the
  closest real one.
- **Streams.** Diagnostics go to stderr prefixed `error:`. Everything else is stdout.
- **Long runs report continuously.** Training prints one line per step with loss, rate and ETA, and
  `[ckpt @ N]` when it writes. Parse those lines to follow a run; do not poll the file.
- **Checkpoints are atomic.** Written to a temp file and renamed, so an interrupted run always
  leaves a loadable GGUF plus its `.optstate` sidecar.

## Invariants

Violating any of these wastes a run. They are checked where possible; a few cannot be.

0. **An architecture is a file, not a fork.** If a model shape is missing, add `src/arch/<name>.ts`
   rather than special-casing the trainer. Everything outside `src/arch/` is architecture-agnostic
   on purpose, and a conditional there is a bug.
1. **The tokenizer freezes at step one.** Vocab and embeddings are fixed for the model's whole life.
   Every fine-tune must reuse the base's `tokenizer.json` byte for byte. `chat-corpus` aborts if the
   tokenizer lacks the ChatML specials; `tokenize` silently trains a new vocab if you point it at a
   prefix with no tokenizer beside it.
2. **`--resume` requires an exact architecture match.** `--hidden`, `--layers`, `--head-dim`,
   `--window`, `--max-seq` and the vocab must equal the checkpoint's. A mismatch aborts before any
   compute and names the field. `inspect` prints the correct flags.
3. **The optimizer sidecar is `<model>.gguf.optstate`.** Same directory, exact name. Missing means a
   cold optimizer, which re-warms momentum over the first few hundred steps rather than failing.
4. **Compute is f32.** f16 operands overflow to NaN at these sizes (it reproduced at exactly the
   same step under two different learning rates), and they buy no wall-clock: measured, f16 compute
   is 0.98x on attention itself, so it is not a matter of which kernel dominates.
   `--checkpoint-precision f16` only affects stored checkpoints.
5. **Tokens seen = steps x batch x seq-len.** One epoch means `corpus_tokens / (batch * seq-len)`
   steps. The trainer prints the epoch count at startup; check it before walking away.
6. **The LR schedule is derived from `--steps`.** Warmup is 10% and cooldown 20% of the total, so
   resuming with a different `--steps` silently reshapes the schedule mid-run. Keep it constant
   across resumes and move `--start-step` instead.
7. **`--seq-len` must fit `--max-seq`,** and context is capped by a WebGPU buffer limit before
   compute: attention binds one `[heads, T, T]` buffer per layer. 8192 works on adapters that grant
   their full buffer size; 2500-3000 on those that fall back to the 128 MiB default.

## Recipes

### Verify the install

```
deno run -A cli.ts demo
```

Under a minute on CPU. Trains a tiny model, exports f16/q8_0/q4_0, re-parses each. If this fails,
nothing else will work; if it passes, later failures are about data, flags or hardware.

### Continue pretraining a published model

The most common task. Take [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base)
and train it on more data.

```
# 1. Get the model, its optimizer state, and its tokenizer
hf download Felladrin/Minueza-3-95M-Base --local-dir base/

# 2. Find the architecture flags the checkpoint needs
deno run -A cli.ts inspect --model base/Minueza-3-95M-Base.F32.gguf

# 3. Build a corpus (or bring your own .txt)
deno run -A cli.ts corpus --source HuggingFaceFW/fineweb-edu:sample-10BT:text \
  --size-mb 400 --out corpus/more.txt

# 4. Tokenize it WITH THE MODEL'S OWN TOKENIZER
cp base/tokenizer.json data/more.tokenizer.json
deno run -A cli.ts tokenize --text corpus/more.txt --out data/more --vocab 32768

# 5. Train, repeating the flags inspect printed
deno run -A cli.ts pretrain --data data/more.tokens --out out/continued.gguf \
  --resume base/Minueza-3-95M-Base.F32.gguf \
  --hidden 640 --layers 12 --head-dim 64 --window 1024 --max-seq 8192 \
  --steps 5000 --seq-len 2048 --batch 8 --lr 0.01 --checkpoint-every 500
```

Step 4 is the one that goes wrong. `tokenize` reuses a tokenizer that already sits at the output
prefix and trains a fresh, incompatible one if it does not. Copy the file first, every time.

### Fine-tune a downloaded checkpoint

The corpus commands reuse the base model's vocab verbatim, and a downloaded model
carries its vocab in GGUF metadata rather than in a `.tokenizer.json`. So there is one
extra step versus continuing a model trained here:

```sh
# 1. Convert the HF checkpoint to GGUF (llama.cpp's convert_hf_to_gguf.py)
# 2. Read its shape, and write its vocab where the corpus commands can find it
mkdir -p data
deno run -A cli.ts inspect --model base.gguf --dump-tokenizer data/base.tokenizer.json
# 3. Build the SFT corpus against THAT vocab
deno run -A cli.ts chat-corpus --data chats.jsonl --tokenizer data/base.tokenizer.json --out data/sft
# 4. Fine-tune, repeating the architecture flags step 2 printed. They are not
#    optional: without them the config is built from defaults and the resume aborts.
deno run -A cli.ts finetune --data data/sft.tokens --mask data/sft.mask \
  --template data/sft.template.txt --resume base.gguf --out out/tuned.gguf --steps 400 \
  --arch llama --hidden 576 --layers 30 --head-dim 64 --heads 9 --kv-heads 3 \
  --ffn-dim 1536 --max-seq 8192
```

The flags above are SmolLM2-135M's; paste whatever step 2 prints for your base, ALL of
it. `--max-seq` is compared too, so dropping it resumes against the default
`max(8192, seq-len)` and a base with a longer context (Qwen3 is 40960) aborts on the
mismatch. `--heads` is load-bearing on qwen3 for the same reason and one more: published
Qwen3 checkpoints size the attention block independently of the model width (LittleLamb-0.3B
is 16 heads of 128 over a width of 544), and without the flag there is nothing to derive the
head count from but `hidden / head-dim`, which does not divide.

The base must already carry ChatML (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) as
atomic specials; `--dump-tokenizer` says whether it does. Its other special tokens do
not have to match ours.

### Train a new base model from scratch

```
deno run -A cli.ts corpus --size-mb 400 --parts 5 --out corpus/blend.txt
deno run -A cli.ts tokenize --text corpus/blend-p1.txt,corpus/blend-p2.txt --out data/blend \
  --vocab 32768 --curriculum-specials
deno run -A cli.ts pretrain --data data/blend.tokens --out out/base.gguf \
  --hidden 640 --layers 12 --steps 88000 --seq-len 2048 --batch 8 --lr 0.01
```

Pass `--curriculum-specials` unless you are certain the model will never be fine-tuned for chat.
Adding a special token later is impossible without discarding the trained embeddings.

### Fine-tune for chat

```
deno run -A cli.ts chat-corpus --data HuggingFaceTB/smol-smoltalk \
  --tokenizer data/blend.tokenizer.json --out data/chat
deno run -A cli.ts finetune --data data/chat.tokens --mask data/chat.mask \
  --template data/chat.template.txt --resume out/base.gguf --out out/instruct.gguf \
  --hidden 640 --layers 12 --steps 300 --seq-len 1024 --batch 8 --lr 0.001
```

Use roughly a tenth of the pretraining learning rate. SFT nudges a base model; at the pretraining
rate it overwrites it.

### Evaluate

```
deno run -A cli.ts eval-loss --model out/base.gguf --data data/blend.tokens   # trend across checkpoints
deno run -A cli.ts eval-choice --model out/base.gguf --task hellaswag --limit 500
deno run -A cli.ts generate --model out/base.gguf --prompt "Once upon a time"
```

`eval-loss` is the one to watch during a run: fixed seed, fixed windows, directly comparable across
checkpoints. `eval-choice` at these sizes hovers near chance (25.0), so read the trend, not the
number.

Split a held-out set off the corpus BEFORE tokenizing it, or you cannot compare two checkpoints
that trained for different numbers of epochs: whatever you hold out from the shorter run is
training data for the longer one. `eval-loss --holdout 1` wants a separate `.tokens` file;
its default 1% tail is corpus the run has already seen.

Never pick between checkpoints on the training loss. A longer SFT run reached a lower training loss
and was the worse model on every other axis, benchmarks included (`docs/optimization.md` lever 12).

Before trusting any absolute loss from a DOWNLOADED checkpoint, score one file with llama.cpp too.
Every suite here is a self-consistency check (GPU against CPU, analytic gradients against finite
differences, export against re-import), so a forward pass that disagrees with llama.cpp passes all
of them. That is how the `llama` RoPE row order stayed wrong until a fine-tune opened at loss 6.4
(lever 17). Repeated text is the sharpest probe: a correct model scores near zero perplexity on a
sentence repeated 400 times, and a model that cannot read its own context does not.

For a qualitative read, `bash scripts/eval-completions.sh out/base.gguf` runs a fixed prompt battery
through llama.cpp with a repetition penalty. A base model loops under `generate`'s pure greedy
decoding and reads worse than it is; keep the battery and the preset fixed so checkpoints stay
comparable.

### Publish

```
deno run -A cli.ts export --model out/base.gguf --name My-Model-95M --quants f32,q8_0,q4_0
cp out/base.gguf.optstate out/My-Model-95M.F32.gguf.optstate   # so others can continue training
```

## Architectures

`archs` lists what this build can train. Three today:

| `--arch`           | Shape                                                          | Compatible with                                |
| ------------------ | -------------------------------------------------------------- | ---------------------------------------------- |
| `gemma3` (default) | GQA + QK-norm, sandwich norms, GeGLU, sliding-window attention | google/gemma-3-*, Felladrin/Minueza-3-95M-Base |
| `llama`            | pre-norm GQA, SwiGLU, one RoPE base, full attention            | SmolLM2, TinyLlama, Qwen2, Mistral             |
| `qwen3`            | llama plus per-head QK-RMSNorm, head size free of the width    | Qwen3-0.6B, Qwen3-1.7B                         |

Each one is a single file in `src/arch/` registered in `src/model/registry.ts`, and the CLI, the
tests and the checkpoint loader all read that registry. Architecture-specific flags (`--window` is
Gemma3's, `--heads` is llama's and qwen3's) are merged into `pretrain` and `finetune`; passing one
that belongs to a different architecture is an error, not a silent no-op. When two architectures
share a flag but disagree about its default, the merged help says so and `archs --json` has the
per-architecture text.

A checkpoint records its own architecture, so `--resume` never needs `--arch`, and giving a
conflicting one fails immediately. To reproduce a specific published shape, pass its dimensions
explicitly:

```
# SmolLM2-135M's shape: 576 hidden, 30 layers, 9 heads of 64 over 3 KV heads
deno run -A cli.ts pretrain --arch llama --data data/corpus.tokens --out out/smol.gguf \
  --hidden 576 --layers 30 --heads 9 --kv-heads 3 --head-dim 64 --ffn-dim 1536 \
  --steps 20000 --seq-len 2048 --batch 8

# Qwen3-0.6B's shape: 1024 hidden, 28 layers, 16 heads of 128 over 8 KV heads
deno run -A cli.ts pretrain --arch qwen3 --data data/corpus.tokens --out out/q3.gguf \
  --hidden 1024 --layers 28 --heads 16 --kv-heads 8 --head-dim 128 --ffn-dim 3072 \
  --steps 20000 --seq-len 2048 --batch 8
```

Adding an architecture is one file plus one registry line, and it inherits the gradient checks and
the round-trip test automatically: docs/adding-an-architecture.md.

## When something fails

| Message                                                                                      | Meaning                                                      | Fix                                                                                                                         |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `--resume config mismatch (hidden: built 512 vs checkpoint 640)`                             | your architecture flags differ from the checkpoint           | run `inspect` and copy the flags it prints                                                                                  |
| `no sibling tokenizer <prefix>.tokenizer.json`                                               | `pretrain` got a `.tokens` file with no tokenizer next to it | keep the pair together, or re-run `tokenize`                                                                                |
| `does not encode [...] atomically`                                                           | fine-tuning a base whose vocab has no ChatML tokens          | a base trained here needs `tokenize --curriculum-specials`; a downloaded one must already carry ChatML in its specials list |
| `mask <path> has N tokens, corpus has M`                                                     | `.mask` and `.tokens` are from different `chat-corpus` runs  | rebuild both together                                                                                                       |
| `mask supervises nothing`                                                                    | the template rendered no assistant turns                     | check the dataset actually has `assistant` roles                                                                            |
| `no WebGPU: training needs Deno`                                                             | running under Node or Bun                                    | training needs Deno; Node and Bun have no GPU backend here                                                                  |
| `GPU/CPU parity probe failed`                                                                | the backend disagrees with the reference at init             | a real bug; stop and report it, do not train through it                                                                     |
| NaN loss partway into a run                                                                  | f16 overflow, or a learning rate above 0.01                  | keep compute f32; `--lr 0.01` is the proven ceiling, 0.02 diverged                                                          |
| a loss far worse than the checkpoint deserves, on a model that still generates readable text | this engine and llama.cpp disagree about the forward pass    | score one file with both before blaming the corpus: `docs/optimization.md` lever 17                                         |
| OOM at long context                                                                          | the per-layer attention buffer                               | add `--reclaim` (5.6x less peak memory, 23% slower, lever 3b), or lower `--seq-len`                                         |

## Hardware reality

Measured on one AMD Strix Halo APU (128 GB unified memory), f32:

| Model | Shape              | Throughput                              | Peak GPU |
| ----- | ------------------ | --------------------------------------- | -------- |
| 94.7M | batch 8 x seq 2048 | 1588 tokens/s                           | 39.3 GB  |
| 94.7M | batch 8 x seq 2048 | 903 tokens/s (pre-2026-08-18 kernels)   | 39.3 GB  |
| 94.7M | batch 8 x seq 1024 | ~1050 tokens/s (pre-2026-08-18 kernels) | 28.9 GB  |

The 2026-08-18 kernel rewrite vectorized attention, GEMM and cross-entropy: 903 -> 1588 tok/s on
Strix (1.76x) and 2.40x on an M1 Max. 1588 is the SUSTAINED rate inside a real run; the same shape
measured 2651 tok/s over 150 steps on an idle GPU, and that 1.7x gap is not yet attributed (see
`docs/optimization.md` lever 1c). Run `bench` on the machine in front of you before planning around
any of these.

The published model was trained on the pre-rewrite kernels, so its 1.95B tokens cost about 25 days
of wall clock; at 1588 tok/s the same run is about 14.
Plan in those units: a "quick experiment" is 100M tokens and a day. This is not a CUDA cluster and
no flag makes it one. Sub-100M models are the honest target.

Deno ships WebGPU natively and is required for training. Node and Bun run the CPU reference path,
which is for tests and the demo only.

## Repository map

```
cli.ts              the only entry point
src/cli/            flag parsing, help, the command registry
src/commands/       one module per subcommand; the CLI is the only caller
src/model/          autograd, the architecture contract, the registry
src/arch/           one file per architecture: config, forward, GGUF names
src/backend/        WebGPU: WGSL kernels, device-resident training loop, GPU optimizers
src/train/          optimizers (Muon, AdamW), LR schedule, the trainer loop
src/gguf/           GGUF reader and writer, quantizers
src/export/         model to GGUF, GGUF back to model
src/tokenizer/      byte-level BPE
src/data/           token files, chat templates, HF dataset fetching and parsing
tests/              gradcheck (finite differences), gpu-parity (GPU vs CPU), pure-logic checks
docs/               how to add an architecture, design rationale, optimization notes
docs/notes/         point-in-time records of the Minueza-3 run, kept as evidence
scripts/            the eval batteries and corpus builders, plus the historical Minueza-3 run scripts
```

## Already measured and rejected

Before optimizing a kernel, read the ruled-out table in `docs/optimization.md` (lever 1d). It
records what was tried, measured and abandoned, with the numbers: f16 compute (0.98x), f16 storage
for Q/K/V (1.02-1.06x), split-K attention (0.4-0.7x), QT query-register tiling (0.80-0.94x at QT=2, 0.48-0.68x at QT=3), a
larger GEMM tile (0.93x on an idle GPU), lazy host tensor storage (removes 98.9% of host allocation,
moves throughput <1%), and why WMMA, subgroup matrices and bf16 are not reachable from WGSL here.
Each of those cost hours to establish. Re-deriving one is the most common way to waste a day in this
repo.

## Rules for changing this code

- Any new autograd op needs a finite-difference gradient check in `tests/gradcheck.ts`, and any new
  WGSL kernel needs a CPU-parity check in `tests/gpu-parity.ts`. A kernel that declares
  `var<workgroup>` storage also needs an entry in `tests/kernel-limits.ts`, which asserts the
  emitted WGSL stays under WebGPU's 16 KiB portable floor: no runtime here validates that limit, so
  nothing else will tell you. All suites must pass: `deno task test`, which type-checks the tree
  first (`deno task check`) because `deno run` does not.
- The reference backend stays dependency-free and runtime-agnostic (Deno, Bun, Node).
- The `gemma3` tensor names and metadata keys are a contract with llama.cpp. Changing them breaks
  every published checkpoint.
- New workflows are subcommands in `src/commands/`, registered in `src/cli/registry.ts`. Do not add
  loose scripts; a workflow that is not in `help` does not exist.
- Every flag carries its own `describe` string. That text is the documentation, for people and for
  `help --json` alike.

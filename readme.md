```
 ______     ______     __  __     ______      ______   ______     ______     __     __   __     ______     ______
/\  ___\   /\  ___\   /\ \/\ \   /\  ___\    /\__  _\ /\  == \   /\  __ \   /\ \   /\ "-.\ \   /\  ___\   /\  == \
\ \ \__ \  \ \ \__ \  \ \ \_\ \  \ \  __\    \/_/\ \/ \ \  __<   \ \  __ \  \ \ \  \ \ \-.  \  \ \  __\   \ \  __<
 \ \_____\  \ \_____\  \ \_____\  \ \_\         \ \_\  \ \_\ \_\  \ \_\ \_\  \ \_\  \ \_\\"\_\  \ \_____\  \ \_\ \_\
  \/_____/   \/_____/   \/_____/   \/_/          \/_/   \/_/ /_/   \/_/\/_/   \/_/   \/_/ \/_/   \/_____/   \/_/ /_/

           F E L L A D R I N ' S   G G U F   T R A I N E R   +∞   ::   train-from-scratch -> GGUF, no PyTorch
```

# Felladrin's GGUF Trainer

Train a language model from scratch, in TypeScript, straight to GGUF. No Python, no PyTorch. The
weights live in a GGUF file from the first step to the last, so every checkpoint is already
something llama.cpp can load.

Training runs on WebGPU (WGSL compute shaders, forward and backward), so AMD, Apple Silicon and
NVIDIA all work.

I trained [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base) with it: 94.7M
parameters, 1.95B tokens, one APU.

## Requirements

**Deno 2.x**, which ships WebGPU natively. No build step, no npm install.

**A GPU adapter.** Every command that computes needs one, and there is no CPU fallback. If a CPU is
all you have, read [Training on a CPU](#training-on-a-cpu) below.

Memory scales with `seq-len x batch`, which at real shapes means tens of GB. Three flags change that
trade, all measured in [docs/performance.md](docs/performance.md):

| flag            | what it does                                                      | measured                                                   |
| :-------------- | :---------------------------------------------------------------- | :--------------------------------------------------------- |
| `--recompute`   | replays each layer in backward instead of storing its activations | 4.6x less pool, and faster here rather than slower         |
| `--lora-rank N` | freezes the base, trains rank-N adapters, merges them on export   | optimizer state 4495 MB to 63 MB, for ~14% less throughput |
| `--reclaim`     | frees each micro-batch's activations as it goes                   | 39.3 GB to 7.0 GB peak, for 23% less throughput            |

## Quickstart

```sh
deno run -A cli.ts demo     # trains a tiny model on the GPU, exports 3 GGUFs, verifies them
deno run -A cli.ts help     # every command
```

To continue training a published model, using [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli)
to fetch it:

```sh
hf download Felladrin/Minueza-3-95M-Base --local-dir base/
deno run -A cli.ts inspect --model base/Minueza-3-95M-Base.F32.gguf
deno run -A cli.ts pretrain --data your.tokens --out out/continued.gguf \
  --resume base/Minueza-3-95M-Base.F32.gguf \
  --hidden 640 --layers 12 --steps 5000 --seq-len 2048 --batch 8
```

## Base models to fine-tune

Any published checkpoint works as a starting point if it clears two gates.

**It has to convert to one of the three architectures.** `inspect` reads `general.architecture` out
of the GGUF and looks it up in the registry, so `gemma3`, `llama` and `qwen3` load and nothing else
does. The name on the model card is not a guide: Qwen2 and Qwen2.5 convert to a `qwen2` arch and
SmolLM3 converts to `smollm3`, so neither loads here even though Qwen3 and SmolLM2 both do.

**For chat or roleplay fine-tuning it also needs ChatML in its vocab.** `chat-corpus` aborts unless
`<|im_start|>`, `<|im_end|>` and `<|endoftext|>` each encode as a single token. A base without them
is still fine for continued pretraining with `pretrain --resume`; it just cannot go through the SFT
path.

Verified against the Hugging Face API on 2026-08-25, smallest first. All Apache-2.0, none gated.

| Model                                                                               | `--arch` | Params | Chat fine-tune | Good for                                                                                       |
| :---------------------------------------------------------------------------------- | :------- | -----: | :------------- | :--------------------------------------------------------------------------------------------- |
| [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base)           | `gemma3` |  94.7M | yes            | the fastest loop, and no conversion step: it ships as GGUF with its optimizer state beside it  |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)                   | `llama`  |   135M | yes            | the best-trained tiny base, and the shape measured here at 424 tokens/s                        |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | `llama`  |   135M | yes            | same weights and vocab, already following turns, so SFT teaches a voice rather than the format |
| [LittleLamb](https://huggingface.co/MultiverseComputingCAI/LittleLamb)              | `qwen3`  |   293M | yes            | the small `qwen3` option, heavily trained for its size                                         |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)                   | `llama`  |   362M | yes            | the step up when 135M stops improving, and still an overnight run                              |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                      | `qwen3`  |   596M | yes            | an untouched base that already carries ChatML, near the practical size ceiling                 |
| [TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)                   | `llama`  |   1.1B | no             | `pretrain --resume` only, and slow here; the no-ChatML case                                    |

Three models have been taken end to end with this repo, one per architecture:
[Minueza-3-95M-RP](https://huggingface.co/Felladrin/Minueza-3-95M-RP) from Minueza-3-95M-Base
(`gemma3`), [LittleLamb-293M-RP](https://huggingface.co/Felladrin/LittleLamb-293M-RP) from LittleLamb
(`qwen3`), and [SmolLM2-135M-Heretic-RP](https://huggingface.co/Felladrin/SmolLM2-135M-Heretic-RP)
from an abliterated fork of SmolLM2-135M-Instruct (`llama`).

What rules a model out, in the order worth checking. **Architecture** first, since it is the only
fatal one and the name misleads: OLMo 2 converts to `olmo2`, Granite 4 to `granitehybrid`,
StableLM 2 to `stablelm`, LFM2 to `lfm2`, Pythia to `gptneox`. Then **access and license**: Google's
Gemma 3 checkpoints are genuine `gemma3` and would otherwise fit, but they need an access request
and carry the Gemma license, and StableLM 2 and LFM2 attach commercial conditions. Then **size**,
since much past 1B stops being practical on one consumer GPU.

Checking one that is not listed takes a single command. Convert it with llama.cpp's
`convert_hf_to_gguf.py`, then:

```sh
deno run -A cli.ts inspect --model your-base.gguf --dump-tokenizer data/your-base.tokenizer.json
```

That prints the architecture, the exact `--resume` flags the checkpoint needs, and whether the vocab
can drive `chat-corpus`.

## Commands

|                               |                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `corpus`                      | download and shuffle a pretraining corpus from Hugging Face                                                      |
| `tokenize`                    | text corpus to binary token stream plus a BPE vocab                                                              |
| `chat-corpus`                 | chat dataset to SFT tokens plus an assistant-only loss mask                                                      |
| `pretrain`                    | train a base model, or continue one with `--resume`                                                              |
| `finetune`                    | fine-tune on chat data, supervising only the assistant turns                                                     |
| `eval-loss`                   | held-out validation loss on fixed windows                                                                        |
| `eval-choice`                 | ARC-Challenge / ARC-Easy / HellaSwag / PIQA / perplexity                                                         |
| `generate`                    | greedy completion without llama.cpp                                                                              |
| `inspect`                     | a GGUF's metadata, shape, and the flags needed to resume it                                                      |
| `export`                      | re-export under a release name, with quants                                                                      |
| `archs`                       | list the architectures this build can train                                                                      |
| `bench`                       | time the WebGPU kernels at fixed shapes, before and after a change                                               |
| `demo`                        | the install check                                                                                                |
| `style-seed`, `style-restyle` | optional: build a chat corpus rewritten in one author's voice (needs the [pi](https://pi.dev/) coding agent CLI) |

Every command has `--help`, and `deno run -A cli.ts help --json` dumps the whole CLI as a schema.
[agents.md](agents.md) is the full manual: the recipe for every workflow, the invariants that waste a
run when broken, and what each failure message means. It is written for a coding agent, which happens
to make it the fastest read for a person too.

## Architectures

Three: `gemma3` (GQA + QK-norm, sandwich norms, GeGLU, sliding-window attention), `llama` (pre-norm
GQA, SwiGLU, full attention, the SmolLM2 and TinyLlama shape) and `qwen3` (llama plus per-head
QK-RMSNorm).

Pick one with `--arch`. A checkpoint records its own, so resuming never needs it. Each architecture
is a single file in `src/arch/` plus one line in the registry, and it gets the gradient checks and
the export round-trip test for free; adding one is documented in
[docs/adding-an-architecture.md](docs/adding-an-architecture.md).

## What works

- GGUF v3 writer and reader, spec-faithful metadata and tensor layout, with F16/Q8_0/Q4_0 quantizers
  matching the ggml block layout.
- Muon (Newton-Schulz orthogonalized momentum) and AdamW, both GPU-resident, with MuonClip, muP init
  transfer and a WSD schedule.
- The whole op set as WGSL, forward and backward, including flash-style and sliding-window attention
  and a fused cross-entropy.
- Checkpoint resume through GGUF plus an optimizer-state sidecar, so a long run survives an
  interruption.
- `deno task test` type-checks the tree, then runs finite-difference gradient checks on every op
  (with a negative control, so the harness is known to catch a wrong backward) and GPU-vs-reference
  parity on every kernel. [docs/correctness.md](docs/correctness.md) is what those checks are for.

## Results

Scored with `eval-choice` on the Open SLM Leaderboard's four tasks, full sets, 0-shot, acc_norm.
[Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base), the published 94.7M model
trained with this repo, gets PIQA 61.26, ARC-Easy 40.53, ARC-Challenge 23.81, HellaSwag 30.14.
Combined the way that board combines them, that is an **Intelligence Index of 10.67**, against 25-27
at the top of the board.

Two qualifiers that matter. This is a self-computed index, not a submitted entry: the scoring matches
lm-eval-harness on the query construction and on normalizing acc_norm by character length, which is
what makes the numbers comparable at all, but nobody else ran them. And the board's fifth task
(ArithMark-3) is not implemented here, so it is assumed at chance; omitting the term instead gives
12.98, making the honest range 10.7-13.0.

At 94.7M parameters on one consumer APU, the cohort above it on that board is trained on 14 to 700
times more tokens per parameter. [docs/evaluation.md](docs/evaluation.md) covers what they do
differently, and why tokens-per-parameter, not anything in this codebase, is the ceiling on quality.

## Honest limits

- **A model this size is a demonstration, not a product.** At 94.7M and 20.6 tokens per parameter it
  produces locally fluent text and cannot hold a conversation: it does not carry facts across turns,
  and it will confabulate an answer to anything factual. The index above measures the trainer
  working, not a model you would deploy.
- **Single-digit to low-hundreds of millions of parameters.** At 94.7M on one APU it does 1588
  tokens/s, so the published model's 1.95B tokens is about 14 days. JS and WebGPU still will not
  match a CUDA cluster, and no flag changes that.
- **Training keeps float master weights and quantizes at export**, so you cannot really train in
  Q4_0.
- **Context hits a WebGPU buffer limit before compute becomes the problem**, and the buffer that
  binds is the logits tensor (`seq-len x vocab x 4` bytes), not attention. At a 32768 vocab, 8192
  needs 1 GiB and works on an adapter that grants its full buffer size; one that falls back to the
  WebGPU default of 128 MiB stops at 1024. `--loss-chunk N` lifts that cap by streaming the readout
  and the loss in vocab chunks. Training and both eval commands take it.
- **The GGUF output is structurally verified here**, but check it against `llama-cli` before trusting
  a specific build.

## Training on a CPU

This trainer will not do it, and that is a refusal rather than a gap. Every command that computes
needs a WebGPU device, `demo` included, and they all stop without one. The reference
implementation in `src/model/autograd.ts` is single-threaded and exists to be the oracle the GPU
kernels are checked against, so wiring it up would produce a flag that accepts your run and then
never finishes.

**Use [`transformers`](https://huggingface.co/docs/transformers) with
[`peft`](https://huggingface.co/docs/peft) instead.** Measured on Qwen3-0.6B-Base at seq 512,
batch 1, 10 threads: LoRA gives 118 tokens/s at 6.1 GB peak, a full fine-tune 82 tokens/s at
12.8 GB, so both fit in 32 GB.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft
```

[Issue #41](https://github.com/felladrin/gguf-trainer/issues/41) is the long form of this answer,
asked about a specific machine and answered with the numbers.

## Contributing

See [contributing.md](.github/contributing.md). The short version: keep the engine dependency-free,
cover any new autograd op with a finite-difference gradient check, and do not break GGUF loadability
in llama.cpp (the tensor names and metadata keys are a contract).

## License

MIT, see [license.txt](license.txt).

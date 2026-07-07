```
 ______     ______     __  __     ______      ______   ______     ______     __     __   __     ______     ______
/\  ___\   /\  ___\   /\ \/\ \   /\  ___\    /\__  _\ /\  == \   /\  __ \   /\ \   /\ "-.\ \   /\  ___\   /\  == \
\ \ \__ \  \ \ \__ \  \ \ \_\ \  \ \  __\    \/_/\ \/ \ \  __<   \ \  __ \  \ \ \  \ \ \-.  \  \ \  __\   \ \  __<
 \ \_____\  \ \_____\  \ \_____\  \ \_\         \ \_\  \ \_\ \_\  \ \_\ \_\  \ \_\  \ \_\\"\_\  \ \_____\  \ \_\ \_\
  \/_____/   \/_____/   \/_____/   \/_/          \/_/   \/_/ /_/   \/_/\/_/   \/_/   \/_/ \/_/   \/_____/   \/_/ /_/

           F E L L A D R I N ' S   G G U F   T R A I N E R   +∞   ::   train-from-scratch -> GGUF, no PyTorch
```

# Felladrin's GGUF Trainer

Train a **Gemma3ForCausalLM** language model **from scratch, in TypeScript**, and export it
**directly to GGUF** — no Python, no Hugging Face, no PyTorch.

Yes, it's a _trainer_ — in both senses. It trains language models, and it comes with the swagger of
a 2000s game trainer. `+∞ params unlocked.`

Runs on **Deno, Bun, or Node**. The reference training backend is pure TypeScript (CPU); the
**WebGPU** backend runs the same op set as WGSL compute shaders — forward and backward — on **AMD,
Apple Silicon, and NVIDIA** alike (Deno ships WebGPU natively; Node/Bun stay on the CPU backend
today).

## Prerequisites

- **Deno ≥ 1.40** (recommended — native WebGPU, runs `.ts` directly), **or**
- **Bun ≥ 1.0**, **or**
- **Node ≥ 22.6** (uses `--experimental-strip-types` to run `.ts` directly).

No build step and no npm install — the reference backend has zero runtime dependencies.

## Quickstart

```
deno run -A examples/demo.ts                     # Deno
bun examples/demo.ts                             # Bun
node --experimental-strip-types examples/demo.ts # Node
```

The demo trains a tiny model on a toy corpus and writes `tinygemma3-{f16,q8_0,q4_0}.gguf`, then
re-parses each to verify structure.

```
deno run -A examples/demo_gpu.ts                 # the same, trained on the GPU
deno task test                                   # gradient checks + GPU parity
```

## Web UI (guided wizard)

Prefer clicking to CLI flags? A local web app walks you through training a model end to end: pick a
model type, point at any Hugging Face dataset (with a live preview), set the size and training
knobs, watch the loss curve stream live, then chat with the result right in the browser.

```
deno task webui        # builds the React client, then serves at http://localhost:8787
```

The browser is only the control panel — training runs on the local **WebGPU engine** (this repo,
unchanged), and in-browser testing uses [wllama](https://github.com/ngxson/wllama) (llama.cpp in
WASM). It covers **Base** (pretraining), **Instruct**, **Reasoning**, and **tool-calling** models,
with a ChatML chat template embedded into the exported GGUF so llama.cpp/wllama format turns
correctly. Any HF dataset works (text or conversational — `messages`/ShareGPT/Alpaca are
auto-detected). Details in [`web/README.md`](web/README.md).

## Use as a library

```ts
import {
  BPETokenizer,
  buildGemma3GGUF,
  Gemma3Model,
  mulberry32,
  Muon,
  tinyGemma3Config,
  trainLM,
  writeFileBytes,
} from "./src/index.ts";

const tok = new BPETokenizer();
tok.train(myCorpus, 8000); // train a BPE vocab
const tokens = tok.encode(myCorpus);

const cfg = tinyGemma3Config(tok.vocabSize); // or hand-write a Gemma3Config
const model = new Gemma3Model(cfg, mulberry32(42));

const g = model.paramGroups(); // Muon on matmuls, AdamW on the rest
const opt = new Muon(g.muon, g.aux, { lr: 0.02, aux: { lr: 3e-3 } });

trainLM(model, { tokens, seqLen: 128, steps: 2000, batchPerStep: 8, optimizer: opt });

await writeFileBytes("model.gguf", buildGemma3GGUF(model, tok.export(), cfg, { quant: "q4_0" }));
```

## What works today (verified end-to-end)

- **GGUF v3 writer + reader** — spec-faithful metadata & tensor layout (`src/gguf/`).
- **Quantizers** — F16, Q8_0, Q4_0, matching ggml block layout, with dequant for round-trips.
- **Byte-level BPE tokenizer** — GPT-2 compatible, trainable; exports vocab + merges into GGUF.
- **Autograd engine (CPU)** — reverse-mode over dense tensors, exactly the ops a Gemma3 block needs.
- **Gemma3 model** — GQA attention, **QK-RMSNorm**, per-layer **sliding-window attention** (the SWA
  speed lever) with dual local/global RoPE (NEOX), **GeGLU**, **sandwich norms**, tied embeddings.
- **Optimizers** — **AdamW** and **Muon** (Newton–Schulz orthogonalized momentum).
- **Trainer** — next-token cross-entropy, grad accumulation, gradient clipping.
- **WebGPU backend** (`src/backend/webgpu.ts`) — the full op set as WGSL compute shaders, forward +
  backward: tiled GEMM, elementwise, workgroup-reduction RMSNorm/QK-norm, RoPE, causal GQA attention
  (full + sliding-window), cross-entropy. Same `Tensor` interface; the model and `backward()` run
  unchanged.
- **Validation harnesses** — `tests/gradcheck.ts` (finite-difference gradient checks, with a
  negative control) and `tests/gpu_parity.ts` (GPU-vs-CPU forward/backward parity per op and
  whole-model).

Demo results: the CPU demo drives loss down fast, greedy sampling reproduces the toy corpus, and all
three GGUF files verify. The GPU demo (M1 Max) trains a tiny mixed SWA/global Gemma3 device-resident
and the exported GGUF loads and runs in `llama-cli`.

## Roadmap

The core is done end to end on both backends. Shipped since the first cut: **GPU-resident Muon +
AdamW** (the optimizer is no longer CPU-bound), **muP** init transfer, the **WSD** schedule,
**MuonClip**, **GGUF checkpoint resume**, flash-style + **sliding-window attention**, and the
**Gemma3** architecture — SWA is a ~1.9× training speedup at 8K and llama.cpp honors the window at
inference. Details and rationale in [`docs/DESIGN.md`](docs/DESIGN.md).

Next, in rough priority order:

1. **Curriculum training** — pretrain (unlabeled) → instruct → reasoning → tool-calling, chaining
   each stage's checkpoint through the Gemma3 resume loader. Coherence needs a real pretraining
   phase; the reasoning/tool stages graft onto that base.
2. **Throughput** — vec4-vectorized elementwise kernels and a fused online-softmax cross-entropy
   (large-vocab memory). (Compute runs f32: an f16-operand GEMM path was removed — no speedup at
   these sizes, since attention not GEMM dominates, and it overflowed to NaN on real runs.)

## Architecture

```
src/
  gguf/        f16, quantize (+dequant), gguf writer/reader
  tokenizer/   byte-level BPE trainer + encode/decode
  model/       config, autograd (CPU ops), gemma3 forward
  train/       optimizer iface, adamw, muon, trainer loop
  backend/     webgpu (WGSL kernels + buffer/sync machinery), GPU train loop
  export/      model -> GGUF (gemma3 tensor names + metadata)
tests/         gradcheck (finite differences), gpu_parity (GPU vs CPU)
examples/      demo.ts / demo_gpu.ts (train -> export -> verify)
```

## Honest limits

- The CPU backend is a **reference**, not fast. A real 100M-param run needs the WebGPU backend. Even
  then, JS/WebGPU won't match a CUDA cluster — plan on small models (single-digit to low-tens of
  millions of params).
- "Directly in GGUF" means GGUF is the **only** model format used, and training keeps float master
  weights; quantization to Q4_0/Q8_0 happens at export. You cannot meaningfully train _in_ Q4_0 (see
  `docs/DESIGN.md`).
- GGUF output is **structurally** verified here; validate against `llama-cli` before trusting a
  specific build.
- **Context length** is capped by a WebGPU buffer limit before it's capped by compute: the attention
  kernel binds one `[heads,T,T]` buffer per layer. `initWebGPU()` requests the GPU adapter's own
  maximum buffer size at startup (most adapters grant this — confirmed training at 8192 tokens on an
  M1 Max), falling back quietly to WebGPU's conservative 128 MiB default on the rare adapter that
  refuses it, which caps context length at roughly 2500-3000 tokens instead. Either way, cost still
  grows ~quadratically with context length since attention isn't tiled yet (see `docs/HANDOFF.md`
  for the numbers and the planned fix).

## Tests / verification

`deno task test` runs the dedicated suite: `tests/gradcheck.ts` checks every autograd op (and the
whole model) against finite-difference gradients and proves it can catch a wrong backward via a
negative control; `tests/gpu_parity.ts` checks every WGSL kernel's forward and backward against the
CPU reference, plus whole-model and gradient-accumulation parity. `deno task test:node` runs the
gradient checks on Node (the parity suite prints SKIP where WebGPU is unavailable).
`examples/demo.ts` and `examples/demo_gpu.ts` double as end-to-end checks: tokenizer round-trip,
decreasing loss, GGUF structure, finite dequantized weights — exiting non-zero on failure.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). In short: keep the reference backend dependency-free and
runtime-agnostic (Deno/Bun/Node), validate any new autograd op against a finite-difference gradient
check, and don't break GGUF loadability in `llama.cpp` (the `gemma3` tensor names and metadata keys
are a contract).

## License

MIT — see [`LICENSE`](LICENSE).

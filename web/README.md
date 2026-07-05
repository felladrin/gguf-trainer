# Web UI — the training wizard

A local, browser-based wizard around the trainer. You click through six steps; the actual training
runs on the Deno WebGPU engine in this repo, unchanged. In-browser testing of the result uses
[wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to WASM).

## Run it

```
deno task webui          # builds the client, then serves the whole app at http://localhost:8787
```

That's the one-command path: it runs `npm install && vite build` in `web/client`, then starts the
Deno server (`web/server/main.ts`) which serves the built client **and** the API on one origin, with
the cross-origin-isolation headers wllama needs.

For client development with hot reload:

```
deno task webui:server   # terminal 1 — the engine/API server on :8787
deno task webui:dev      # terminal 2 — Vite dev server on :5173 (proxies /api to :8787)
```

## Why a local server (not a pure static site)

Training uses the existing WebGPU engine, which runs under **Deno's native WebGPU** and streams a
disk-backed token corpus — so the browser drives a local Deno process rather than training in-tab.
That reuses every line of the engine, survives the tab closing, and uses the full unified-memory
pool. Only inference/testing happens in the browser (wllama). See the repo's `docs/DESIGN.md` for
the memory budget that makes big local runs practical.

## The six steps

1. **Goal** — new model or continue a checkpoint; pick a type (Base / Instruct / Reasoning /
   Instruct+Tools). Type selects the training format and recommended datasets; every model still
   exports as a llama.cpp-loadable Qwen3 GGUF.
2. **Dataset** — paste any Hugging Face dataset URL (or a direct `.jsonl`/`.parquet`/`.txt` file
   URL). Preview comes from the HF Datasets Server API; the column mapping is auto-detected (`text`,
   OpenAI `messages`, ShareGPT `conversations`, Alpaca `instruction/output`) and editable. A live
   sample shows exactly what the model will train on (rendered through the chat template for chat
   types).
3. **Tokenizer & template** — vocabulary size; for chat models, the editable Qwen3 chat template
   (the default is the one from `Qwen/Qwen3-4B-Instruct-2507`). ChatML control tokens are added to
   the vocabulary automatically so turns tokenize atomically.
4. **Model & training** — a size preset (Tiny/Small/Medium) or custom, plus the same knobs the CLI
   takes (steps, sequence length, batch, Muon/aux LR, muP base width, export quant), with rough
   parameter/memory/time estimates. For chat families, **"Train on assistant turns only"** (on by
   default) masks the system/user prompt out of the loss so training supervises only what the model
   must generate — the standard instruct-tuning objective.
5. **Train** — start the run; loss curve, tokens/sec, log, and a sample generation stream live over
   Server-Sent Events. Stop any time.
6. **Test** — download the GGUF, or load it into wllama and chat/complete in the browser using the
   embedded chat template.

## Layout

```
web/
  shared/types.ts     wizard <-> server contract (imported by both)
  server/             Deno: HTTP + SSE, HF ingestion, corpus build, training job
    main.ts           router, static serving, COOP/COEP, API, SSE
    hf.ts             resolve URL, Datasets Server preview, Parquet file URLs
    parse.ts          Parquet (hyparquet) / JSONL / JSON / CSV / TXT -> rows
    corpus.ts         rows -> rendered docs -> BPE -> .tokens file
    train_job.ts      download -> corpus -> build/resume model -> train -> export GGUF
    jobs.ts           in-process job registry + SSE fan-out + cooperative stop
  client/             Vite + React 19 wizard (its own npm project)
```

The `src/data/chat.ts` module (dataset normalization + the chat template constant) lives in the
dependency-free engine and is shared by the server, the client, and training.

## Data & artifacts

Downloaded corpora, pretokenized `.tokens` files, and exported `.gguf` models are written under
`web/.data/` (gitignored). Trained models appear in `web/.data/models/` and are served for wllama
and download.

## Notes / limits

- At WebGPU-from-scratch sizes (a few million to ~50M params), a chat/reasoning/tool model learns
  the _format_ but not strong capability. Great for the pipeline; not a large pretrained model.
- Training data is downloaded via HF's auto-converted Parquet, or — for datasets HF never converted
  — by listing the dataset repo and pulling its original files (`.jsonl`/`.json`/`.csv`/`.tsv`/
  `.txt`/`.parquet`), or from a direct file URL. For a gated or private dataset, paste an HF token in
  step 2.
- One training job runs at a time (a single GPU).

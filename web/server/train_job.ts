// Run one wizard-configured training job on the WebGPU engine, streaming
// progress as SSE events. This is the config-driven twin of
// examples/train_tinystories.ts: resolve + download + parse the dataset, build
// the corpus, build (or resume) the model, run the device-resident trainer, and
// export a GGUF (with the chat template embedded for chat models).

import { crossEntropy, mulberry32 } from "../../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../../src/model/config.ts";
import { Gemma3Model } from "../../src/model/gemma3.ts";
import { buildGemma3GGUF } from "../../src/export/export_gguf.ts";
import { loadGemma3FromGGUF } from "../../src/export/load_gguf.ts";
import { readGGUF } from "../../src/gguf/gguf.ts";
import { wsdSchedule } from "../../src/train/schedule.ts";
import { diskTokenSource } from "../../src/data/tokens.ts";
import type { TokenSource } from "../../src/data/tokens.ts";
import { readFileBytes, writeFileBytes } from "../../src/io.ts";
import { initWebGPU } from "../../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../../src/backend/webgpu.ts";
import { MuonGpu } from "../../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../../src/backend/train_gpu.ts";
import type { BPETokenizer } from "../../src/tokenizer/bpe.ts";
import {
  fetchParquetUrls,
  fetchRepoDataFiles,
  fetchSplits,
  hfHeaders,
  resolveDataset,
} from "./hf.ts";
import { parseDataFile, type Row } from "./parse.ts";
import { buildCorpus } from "./corpus.ts";
import { emit, type Job, StopSignal } from "./jobs.ts";
import type { TrainConfig } from "../shared/types.ts";

let gpuPromise: Promise<WebGPUBackend | null> | null = null;
function getGPU(): Promise<WebGPUBackend | null> {
  if (!gpuPromise) gpuPromise = initWebGPU();
  return gpuPromise;
}

const MAX_CORPUS_BYTES = 64 * 1024 * 1024; // cap raw download for one run
const MAX_DATA_FILES = 8;

async function fetchBytes(url: string, token?: string): Promise<Uint8Array> {
  const r = await fetch(url, { headers: hfHeaders(token) });
  if (!r.ok) throw new Error(`download failed ${r.status} ${r.statusText}: ${url}`);
  return new Uint8Array(await r.arrayBuffer());
}

/** Download + parse dataset rows, bounded by size / maxRows. */
async function loadRows(
  ds: TrainConfig["dataset"],
  log: (m: string) => void,
): Promise<Row[]> {
  const res = resolveDataset(ds.url);
  if (res.kind === "file") {
    log(`downloading ${res.url}`);
    const bytes = await fetchBytes(res.url, ds.hfToken);
    return parseDataFile(res.url, bytes);
  }

  const splitInfo = await fetchSplits(res.id, ds.hfToken);
  const config = ds.config ?? splitInfo.configs[0];
  const split = ds.split ?? (splitInfo.byConfig[config]?.[0] ?? "train");

  // Prefer the Datasets Server's canonical auto-converted Parquet; fall back to
  // downloading the repo's own data files for sets it never converted. Each
  // source carries a filename so parseDataFile picks the right parser.
  let sources: { url: string; name: string }[] =
    (await fetchParquetUrls(res.id, config, split, ds.hfToken))
      .map((url) => ({ url, name: "f.parquet" }));
  if (sources.length === 0) {
    log("no auto-converted Parquet; listing the dataset repo files");
    sources = (await fetchRepoDataFiles(res.id, split, ds.hfToken))
      .map((f) => ({ url: f.url, name: f.path }));
  }
  if (sources.length === 0) {
    throw new Error(
      `No data files found for ${res.id} [${config}/${split}]. If the dataset is gated or ` +
        `private, paste an access token in step 2; otherwise point at a direct data file URL ` +
        `(.jsonl/.parquet/.csv/.txt).`,
    );
  }

  const rows: Row[] = [];
  let got = 0;
  const maxRows = ds.maxRows && ds.maxRows > 0 ? ds.maxRows : Infinity;
  const nFiles = Math.min(sources.length, MAX_DATA_FILES);
  for (let i = 0; i < nFiles; i++) {
    log(`downloading data file ${i + 1}/${nFiles} (${sources[i].name})`);
    const bytes = await fetchBytes(sources[i].url, ds.hfToken);
    got += bytes.length;
    const part = await parseDataFile(sources[i].name, bytes);
    for (const r of part) {
      rows.push(r);
      if (rows.length >= maxRows) break;
    }
    log(`parsed ${rows.length} rows so far`);
    if (rows.length >= maxRows || got >= MAX_CORPUS_BYTES) break;
  }
  return rows;
}

function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

async function generateGpu(
  gpu: WebGPUBackend,
  model: Gemma3Model,
  tok: BPETokenizer,
  prompt: string,
  n: number,
): Promise<string> {
  const ids = tok.encode(prompt);
  gpu.install();
  try {
    gpu.uploadParams(model.params());
    for (let i = 0; i < n; i++) {
      const ctx = ids.slice(-model.cfg.maxSeq);
      const logits = model.forward(ctx);
      await gpu.sync([logits]);
      const V = model.cfg.vocabSize;
      const base = (ctx.length - 1) * V;
      const next = argmax(logits.data.subarray(base, base + V));
      ids.push(next);
      if (next === tok.eosId) break;
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

function samplePrompt(modelType: TrainConfig["modelType"]): string {
  return modelType === "base"
    ? "Once upon a time"
    : "<|im_start|>user\nTell me a short story.<|im_end|>\n<|im_start|>assistant\n";
}

function safeName(name: string): string {
  return (name.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "") || "model").slice(0, 60);
}

export async function runJob(job: Job, modelsDir: string, jobDir: string): Promise<void> {
  const cfg = job.config;
  const status = (phase: string, message: string) => emit(job, { type: "status", phase, message });
  try {
    status("gpu", "Initializing WebGPU");
    const gpu = await getGPU();
    if (!gpu) throw new Error("WebGPU is unavailable in this Deno runtime — cannot train.");

    // Resume: load the checkpoint's model + tokenizer up front.
    let existingTok: BPETokenizer | undefined;
    let model: Gemma3Model | undefined;
    let modelCfg;
    if (cfg.resumeFrom) {
      status("resume", `Loading checkpoint ${cfg.resumeFrom}`);
      const bytes = await readFileBytes(`${modelsDir}/${cfg.resumeFrom}`);
      const r = loadGemma3FromGGUF(bytes);
      model = r.model;
      modelCfg = r.cfg;
      existingTok = r.tokenizer;
    }

    status("data", `Resolving ${cfg.dataset.url}`);
    const rows = await loadRows(cfg.dataset, (m) => status("data", m));
    if (rows.length === 0) throw new Error("Dataset produced no rows.");

    status("corpus", "Building corpus");
    const tokensPath = `${jobDir}/corpus.tokens`;
    const built = await buildCorpus({
      rows,
      modelType: cfg.modelType,
      mapping: cfg.dataset.mapping,
      vocabSize: cfg.vocabSize,
      chatTemplate: cfg.chatTemplate,
      outPath: tokensPath,
      existingTok,
      maskAssistantLoss: cfg.training.maskPromptLoss ?? true,
      onProgress: (m) => status("corpus", m),
    });
    const src = await diskTokenSource(tokensPath, built.bytesPerToken);
    // Assistant-only loss: a parallel supervision mask, when the corpus builder
    // produced one (chat families with ChatML delimiters).
    let supervised: TokenSource | undefined;
    if (built.maskPath) supervised = await diskTokenSource(built.maskPath, built.bytesPerToken);
    const tps = cfg.training.batch * cfg.training.seqLen;
    emit(job, {
      type: "corpus",
      tokens: src.length,
      vocab: built.tok.vocabSize,
      epochs: (cfg.training.steps * tps) / src.length,
      docs: built.numDocs,
    });

    // Build the model (fresh) unless resuming.
    if (!model) {
      modelCfg = gemma3Config(
        built.tok.vocabSize,
        cfg.model.hidden,
        cfg.model.layers,
        cfg.model.maxSeq,
      );
      const mup = cfg.model.baseWidth === cfg.model.hidden
        ? undefined
        : { baseWidth: cfg.model.baseWidth };
      model = new Gemma3Model(modelCfg, mulberry32(1234), mup);
    }
    if (cfg.training.seqLen > modelCfg!.maxSeq) {
      throw new Error(`seqLen ${cfg.training.seqLen} exceeds context length ${modelCfg!.maxSeq}`);
    }
    emit(job, {
      type: "model",
      params: gemma3ParamCount(modelCfg!),
      summary: `${modelCfg!.nLayers}L hidden=${modelCfg!.hiddenSize} heads=${modelCfg!.nHeads}/${
        modelCfg!.nKVHeads
      } ffn=${modelCfg!.ffnDim}`,
    });

    // Parity probe before committing the run to this backend.
    status("parity", "Checking CPU/GPU parity");
    const probeIn = src.window(0, 16);
    const probeTgt = src.window(1, 16);
    const cpuLoss = crossEntropy(model.forward(probeIn), probeTgt).data[0];
    gpu.install();
    let gpuLoss: number;
    try {
      const l = crossEntropy(model.forward(probeIn), probeTgt);
      await gpu.sync([l]);
      gpuLoss = l.data[0];
    } finally {
      gpu.uninstall();
    }
    if (Math.abs(gpuLoss - cpuLoss) > 1e-3 + 1e-3 * Math.abs(cpuLoss)) {
      throw new Error(`GPU/CPU parity probe failed (CPU ${cpuLoss} vs GPU ${gpuLoss})`);
    }

    // Device-resident training under a WSD schedule.
    const groups = model.paramGroups();
    const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
      lr: cfg.training.muonLr,
      momentum: 0.95,
      aux: { lr: cfg.training.auxLr, weightDecay: 0.0, clip: 1.0 },
    });
    const steps = cfg.training.steps;
    const schedule = wsdSchedule({
      warmupSteps: Math.max(1, Math.round(steps * 0.1)),
      stableSteps: Math.max(0, steps - Math.round(steps * 0.1) - Math.round(steps * 0.2)),
      cooldownSteps: Math.max(1, Math.round(steps * 0.2)),
      minScale: 0.1,
    });

    // GGUF export, reused for mid-run checkpoints, stop, and the final write, so
    // the model file always reflects the latest weights and a run is never lost
    // to an interruption. Stable filename -> one file per model name, latest state.
    const chatTemplate = cfg.modelType !== "base" ? cfg.chatTemplate : undefined;
    const file = `${safeName(cfg.name)}-${cfg.training.quant}.gguf`;
    const exportModel = async (): Promise<{ sizeMB: number; tensors: number }> => {
      const bytes = buildGemma3GGUF(model!, built.tok.export(), modelCfg!, {
        quant: cfg.training.quant,
        name: cfg.name,
        chatTemplate,
      });
      await writeFileBytes(`${modelsDir}/${file}`, bytes);
      const g = readGGUF(bytes);
      if (g.metadata.get("general.architecture") !== "gemma3") {
        throw new Error("export arch != gemma3");
      }
      job.file = file;
      return { sizeMB: bytes.length / 1e6, tensors: g.tensors.length };
    };

    status("train", `Training ${steps} steps`);
    let firstLoss = 0, lastLoss = 0, stopped = false;
    const t0 = Date.now();
    // Checkpoint ~10 times over the run (at least every 500 steps).
    const checkpointEvery = Math.min(500, Math.max(1, Math.round(steps / 10)));
    try {
      await trainLMGpuResident(model, gpu, {
        tokens: src,
        supervised,
        seqLen: cfg.training.seqLen,
        steps,
        batchPerStep: cfg.training.batch,
        optimizer: opt,
        schedule,
        logEvery: Math.max(1, Math.round(steps / 100)),
        rng: mulberry32(7),
        checkpointEvery,
        onCheckpoint: async (step) => {
          const { sizeMB } = await exportModel(); // weights already synced to host
          emit(job, { type: "checkpoint", step, file, sizeMB });
        },
        onLog: (step, loss) => {
          if (job.stopRequested) throw new StopSignal();
          if (step === 0) firstLoss = loss;
          lastLoss = loss;
          const secs = (Date.now() - t0) / 1000;
          emit(job, { type: "step", step, steps, loss, stepsPerSec: step > 0 ? step / secs : 0 });
        },
      });
    } catch (e) {
      if (!(e instanceof StopSignal)) throw e; // real errors -> outer catch
      stopped = true;
      await opt.syncWeightsToHost(); // pull the weights trained up to the stop point
    }
    src.close();
    supervised?.close();
    void firstLoss;

    // Finalize (both a completed run and a user stop): sample, export, terminal event.
    status("sample", "Sampling from the trained model");
    const sample = await generateGpu(gpu, model, built.tok, samplePrompt(cfg.modelType), 48);
    emit(job, { type: "sample", step: steps, text: sample });

    status("export", stopped ? "Exporting stopped checkpoint" : "Exporting GGUF");
    const { sizeMB, tensors } = await exportModel();
    job.status = stopped ? "stopped" : "done";
    emit(
      job,
      stopped
        ? { type: "stopped", file, sizeMB, tensors, sample }
        : { type: "done", file, sizeMB, tensors, sample },
    );
    void lastLoss;
  } catch (e) {
    if (e instanceof StopSignal) { // safety net: stop that escaped the inner catch
      job.status = "stopped";
      emit(job, { type: "stopped" });
    } else {
      job.status = "error";
      emit(job, { type: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }
}

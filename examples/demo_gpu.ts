// End-to-end WebGPU demo: train Qwen3 from scratch on the GPU and write a
// llama.cpp-loadable GGUF. Same pipeline as demo.ts, but every autograd op
// runs as a WGSL compute shader via src/backend/webgpu.ts.
//
// Run (Deno):  deno run -A examples/demo_gpu.ts
// Node and Bun have no WebGPU today; this prints a notice and exits. The CPU
// demo (examples/demo.ts) is the equivalent for those runtimes.

import { readGGUF } from "../src/gguf/gguf.ts";
import { dequantize } from "../src/gguf/quantize.ts";
import { writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { paramCount, tinyConfig } from "../src/model/config.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { buildGGUF } from "../src/export/export_gguf.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

const CORPUS = `
the cat sat on the mat. the cat saw a red ball. the dog ran to the cat.
the dog and the cat played with the red ball on the mat. the sun was warm.
a bird sang in the tree. the cat looked at the bird. the bird flew away.
the dog ran after the ball. the cat sat on the mat and slept in the sun.
`.trim().repeat(12);

function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/** Greedy sampling with the forward pass on the GPU (one sync per token). */
async function generateGpu(
  gpu: WebGPUBackend,
  model: Qwen3Model,
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
      ids.push(argmax(logits.data.subarray(base, base + V)));
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

async function main() {
  console.log("=== Felladrin's GGUF Trainer +∞ :: WebGPU train-from-scratch -> GGUF ===\n");

  const gpu = await initWebGPU();
  if (!gpu) {
    console.log(
      "WebGPU is not available in this runtime. Run this demo with Deno\n" +
        "(deno run -A examples/demo_gpu.ts); Node/Bun can run examples/demo.ts on the CPU.",
    );
    return;
  }
  console.log(`WebGPU adapter: ${gpu.adapterName}`);

  // 1. Tokenizer (identical to the CPU demo).
  const tok = new BPETokenizer();
  tok.train(CORPUS, 280);
  const tokens = tok.encode(CORPUS);
  console.log(`Tokenizer: vocab=${tok.vocabSize}, corpus encoded to ${tokens.length} tokens`);

  // 2. The full tinyConfig — larger than the CPU demo's cut-down model, which
  //    is the point of having a GPU backend.
  const cfg = tinyConfig(tok.vocabSize);
  const model = new Qwen3Model(cfg, mulberry32(1234));
  console.log(
    `Model: qwen3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, ` +
      `heads=${cfg.nHeads}/${cfg.nKVHeads}, headDim=${cfg.headDim}, ` +
      `~${(paramCount(cfg) / 1e3).toFixed(1)}K params`,
  );

  // 3. Trust gate: at init, the GPU forward+loss must match the CPU reference
  //    on a probe window (tests/gpu_parity.ts is the thorough version).
  const probeIds = tokens.slice(0, 16);
  const probeTargets = tokens.slice(1, 17);
  const cpuLoss = crossEntropy(model.forward(probeIds), probeTargets).data[0];
  gpu.install();
  let gpuLossVal: number;
  try {
    const gpuLoss = crossEntropy(model.forward(probeIds), probeTargets);
    await gpu.sync([gpuLoss]);
    gpuLossVal = gpuLoss.data[0];
  } finally {
    gpu.uninstall();
  }
  const drift = Math.abs(gpuLossVal - cpuLoss);
  console.log(
    `Parity probe: CPU loss ${cpuLoss.toFixed(5)} vs GPU loss ${gpuLossVal.toFixed(5)} ` +
      `(|Δ|=${drift.toExponential(1)})`,
  );
  if (drift > 1e-3 + 1e-3 * Math.abs(cpuLoss)) {
    throw new Error("GPU/CPU parity probe failed — do not train on this backend");
  }

  // 4. Train on the GPU: both param groups are device-resident — Muon on the
  //    hidden matmuls, AdamW on the aux group (embeddings, head, norms) — so
  //    weights, moments, and grads never leave the GPU during training.
  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: 0.02,
    momentum: 0.95,
    aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 },
  });
  console.log(
    `\nGPU Muon on ${groups.muon.length} matrices; GPU AdamW on ${groups.aux.length} tensors\n`,
  );

  let firstLoss = 0;
  let lastLoss = 0;
  // Step 0 pays the one-time WGSL pipeline compiles; report it apart from the
  // steady-state per-phase averages.
  let fwdMs = 0;
  let optMs = 0;
  let step0Ms = 0;
  let readback = 0;
  let stepIdx = 0;
  const steps = 40;
  // A WSD lr schedule (src/train/schedule.ts) plugs in here via `schedule:
  // wsdSchedule({...})`. Left off in this demo on purpose: at 40 toy steps the
  // model is still descending steeply, so the cooldown only costs final loss —
  // WSD's payoff is on longer runs where constant lr plateaus. The GPU dynamic-
  // lr path is gated instead by wsdScheduleParity in tests/gpu_parity.ts.
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens,
    seqLen: 32,
    steps,
    batchPerStep: 2,
    optimizer: opt,
    logEvery: 10,
    rng: mulberry32(7),
    onLog: (step, loss) => {
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}`);
    },
    onStepTime: (fwd, o, bytes) => {
      if (stepIdx++ === 0) {
        step0Ms = fwd + o;
        return;
      }
      fwdMs += fwd;
      optMs += o;
      readback = bytes;
    },
  });
  const secs = (Date.now() - t0) / 1000;
  const warm = steps - 1;
  console.log(
    `\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)} in ${secs.toFixed(1)}s ` +
      `(${(steps / secs).toFixed(1)} steps/s on GPU)`,
  );
  console.log(
    `Per-step split (steady state): fwd+bwd+sync ${(fwdMs / warm).toFixed(1)} ms, ` +
      `optimizer ${(optMs / warm).toFixed(1)} ms; ` +
      `readback ${(readback / 1024).toFixed(1)} KiB/step (loss scalars only; ` +
      `both param groups are device-resident); ` +
      `step 0 incl. pipeline compiles ${step0Ms.toFixed(0)} ms`,
  );
  if (!(lastLoss < firstLoss)) throw new Error("Loss did not decrease");

  // 5. Greedy sample with the GPU forward pass.
  console.log(`\nGreedy sample: "${await generateGpu(gpu, model, tok, "the cat", 20)}"`);

  // 6. Export GGUF. Parameters live on the host (the optimizer steps them
  //    there), so the export path is byte-identical to the CPU demo's.
  const outDir = new URL(".", import.meta.url).pathname;
  const bytes = buildGGUF(model, tok.export(), cfg, { quant: "f16", name: "tinyqwen3-gpu" });
  const path = `${outDir}tinyqwen3-gpu-f16.gguf`;
  await writeFileBytes(path, bytes);
  console.log(`\nWrote ${path}  (${(bytes.length / 1024).toFixed(1)} KiB)`);

  const g = readGGUF(bytes);
  const arch = g.metadata.get("general.architecture");
  if (arch !== "qwen3") throw new Error(`arch mismatch: ${arch}`);
  const expectedTensors = 2 + (cfg.tieEmbeddings ? 0 : 1) + cfg.nLayers * 11;
  if (g.tensors.length !== expectedTensors) {
    throw new Error(`tensor count ${g.tensors.length} != expected ${expectedTensors}`);
  }
  const emb = g.tensors.find((t) => t.name === "token_embd.weight");
  if (!emb) throw new Error("token_embd.weight missing");
  const de = dequantize(emb.type, emb.data, emb.dims[0] * emb.dims[1]);
  for (let i = 0; i < de.length; i++) {
    if (!Number.isFinite(de[i])) throw new Error("dequantized embedding has non-finite values");
  }
  console.log(`  verify[f16]: ${g.tensors.length} tensors, arch=qwen3 ✓`);

  console.log("\n=== all checks passed (WebGPU) ===");
}

main().catch((e) => {
  console.error("GPU DEMO FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});

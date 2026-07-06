// End-to-end WebGPU demo for the Gemma3 arch: train a tiny Gemma3 from scratch
// on the GPU (sandwich norms, GeGLU, sqrt(hidden) embed scale, per-layer sliding
// window + local/global RoPE) and write a llama.cpp-loadable GGUF (arch
// "gemma3"). Mirrors demo_gpu.ts; the point is to prove the gemma3 forward and
// the gemma3 GGUF export are correct — llama-cli must load it and reproduce the
// memorized corpus (the real arch-contract gate).
//
// Run (Deno):  deno run -A --unstable-webgpu examples/demo_gemma3.ts

import { readGGUF } from "../src/gguf/gguf.ts";
import { writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { buildGemma3GGUF } from "../src/export/export_gguf.ts";
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
      ids.push(argmax(logits.data.subarray(base, base + V)));
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

async function main() {
  console.log("=== gemma3 WebGPU train-from-scratch -> GGUF (arch contract check) ===\n");
  const gpu = await initWebGPU();
  if (!gpu) {
    console.log("WebGPU unavailable — run with Deno --unstable-webgpu.");
    return;
  }
  console.log(`WebGPU adapter: ${gpu.adapterName}`);

  // Tokenizer with reasoning + EOS specials (exercises the CONTROL token_type).
  const tok = new BPETokenizer();
  tok.train(CORPUS, 320, ["<|endoftext|>", "<think>", "</think>"]);
  const tokens = tok.encode(CORPUS);
  console.log(
    `Tokenizer: vocab=${tok.vocabSize}, ${tokens.length} tokens, specials=${tok.specials}`,
  );

  // Tiny gemma3: 4 layers with swaPattern=3 so layer 2 is a full-attention
  // (global) layer and 0/1/3 are SWA — both code paths get exercised, and the
  // window (16) < seqLen (32) so it genuinely restricts.
  const cfg = gemma3Config(tok.vocabSize, 128, 4, 32, 32, 16);
  cfg.swaPattern = 3;
  const model = new Gemma3Model(cfg, mulberry32(1234));
  console.log(
    `Model: gemma3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, heads=${cfg.nHeads}/` +
      `${cfg.nKVHeads}, headDim=${cfg.headDim}, window=${cfg.slidingWindow}, pattern=${cfg.swaPattern}` +
      `, ~${(gemma3ParamCount(cfg) / 1e3).toFixed(1)}K params`,
  );

  // Parity probe at init.
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
    `Parity probe: CPU ${cpuLoss.toFixed(5)} vs GPU ${gpuLossVal.toFixed(5)} (|Δ|=${
      drift.toExponential(1)
    })`,
  );
  if (drift > 1e-3 + 1e-3 * Math.abs(cpuLoss)) throw new Error("GPU/CPU parity probe failed");

  // Train device-resident.
  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: 0.02,
    momentum: 0.95,
    aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 },
  });
  let firstLoss = 0, lastLoss = 0;
  await trainLMGpuResident(model, gpu, {
    tokens,
    seqLen: 32,
    steps: 80,
    batchPerStep: 2,
    optimizer: opt,
    logEvery: 20,
    rng: mulberry32(7),
    onLog: (step, loss) => {
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}`);
    },
  });
  console.log(`\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)}`);
  if (!(lastLoss < firstLoss)) throw new Error("Loss did not decrease");

  console.log(`\nGreedy sample: "${await generateGpu(gpu, model, tok, "the cat", 20)}"`);

  // Export gemma3 GGUF + structural verify.
  const outDir = new URL(".", import.meta.url).pathname;
  const bytes = buildGemma3GGUF(model, tok.export(), cfg, { quant: "f16", name: "tinygemma3-gpu" });
  const path = `${outDir}tinygemma3-gpu-f16.gguf`;
  await writeFileBytes(path, bytes);
  console.log(`\nWrote ${path}  (${(bytes.length / 1024).toFixed(1)} KiB)`);

  const g = readGGUF(bytes);
  const arch = g.metadata.get("general.architecture");
  if (arch !== "gemma3") throw new Error(`arch mismatch: ${arch}`);
  const expectedTensors = 2 + (cfg.tieEmbeddings ? 0 : 1) + cfg.nLayers * 13;
  if (g.tensors.length !== expectedTensors) {
    throw new Error(`tensor count ${g.tensors.length} != expected ${expectedTensors}`);
  }
  console.log(
    `  verify: ${g.tensors.length} tensors, arch=gemma3, sliding_window=` +
      `${g.metadata.get("gemma3.attention.sliding_window")}, pattern=` +
      `${g.metadata.get("gemma3.attention.sliding_window_pattern")} ✓`,
  );
  console.log("\n=== gemma3 checks passed; load the .gguf in llama-cli to confirm ===");
}

main().catch((e) => {
  console.error("GEMMA3 DEMO FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});

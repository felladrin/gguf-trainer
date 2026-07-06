// Sliding-window attention speedup probe: measure a single attention op
// (fwd+bwd+sync, wall clock) at a fixed context length for full causal vs
// several window sizes, at realistic ~100M-model per-layer dims. This isolates
// the exact lever behind switching to a Gemma3-style SWA architecture — the
// task-#34 "cheap first measurement" before committing to a full arch change.
//
//   deno run -A --unstable-webgpu examples/bench_swa.ts [T] [Hq] [Hkv] [hd] [iters]

import { initWebGPU } from "../src/backend/webgpu.ts";
import { attention, mulberry32, randn, Tensor } from "../src/model/autograd.ts";

// deno-lint-ignore no-explicit-any
const A: string[] = (globalThis as any).Deno?.args ?? [];
const T = A[0] ? Number(A[0]) : 8192;
const Hq = A[1] ? Number(A[1]) : 12;
const Hkv = A[2] ? Number(A[2]) : 6;
const hd = A[3] ? Number(A[3]) : 64;
const iters = A[4] ? Number(A[4]) : 10;

const maybeGpu = await initWebGPU();
if (!maybeGpu) throw new Error("no WebGPU (run under Deno with --unstable-webgpu)");
const gpu = maybeGpu;
console.log(`adapter: ${gpu.adapterName}`);
console.log(`attention op: T=${T} Hq=${Hq} Hkv=${Hkv} hd=${hd}, ${iters} timed iters each\n`);

const rng = mulberry32(7);
function rand(shape: number[]): Tensor {
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * 0.5;
  return t;
}
const q = rand([T, Hq * hd]);
const k = rand([T, Hkv * hd]);
const v = rand([T, Hkv * hd]);
const seed = new Float32Array(T * Hq * hd);
for (let i = 0; i < seed.length; i++) seed[i] = randn(rng);

gpu.install();

async function bench(window: number): Promise<number> {
  // 2 warm-up iters compile the pipelines for this window; then time `iters`.
  for (let w = 0; w < 2; w++) {
    q.zeroGrad();
    k.zeroGrad();
    v.zeroGrad();
    const out = attention(q, k, v, T, Hq, Hkv, hd, window);
    out.grad.set(seed);
    gpu.seedGradFromHost(out);
    out._backward();
    await gpu.sync([out]);
  }
  const t0 = performance.now();
  for (let it = 0; it < iters; it++) {
    q.zeroGrad();
    k.zeroGrad();
    v.zeroGrad();
    const out = attention(q, k, v, T, Hq, Hkv, hd, window);
    out.grad.set(seed);
    gpu.seedGradFromHost(out);
    out._backward();
    await gpu.sync([out]);
  }
  return (performance.now() - t0) / iters;
}

const windows = [0, 4096, 2048, 1024, 512];
const results: { window: number; ms: number }[] = [];
for (const w of windows) results.push({ window: w, ms: await bench(w) });

const full = results[0].ms;
console.log("window        ms/op    speedup vs full");
for (const r of results) {
  const label = r.window === 0 ? "full causal" : `W=${r.window}`;
  console.log(
    `${label.padEnd(12)}  ${r.ms.toFixed(1).padStart(7)}   ${(full / r.ms).toFixed(2)}x`,
  );
}

// Gemma3 is 5 SWA : 1 global. Estimate the per-layer-attention blend and the
// implied whole-step speedup given attention was ~78% of the step at 8K.
const w1024 = results.find((r) => r.window === 1024)!.ms;
const blend = (5 * w1024 + 1 * full) / 6;
const attnFrac = 0.78;
const stepSpeedup = 1 / (1 - attnFrac + attnFrac * (blend / full));
console.log(
  `\nGemma3 5:1 blend @ W=1024: ${blend.toFixed(1)} ms/op (${(full / blend).toFixed(2)}x attn)`,
);
console.log(
  `implied whole-step speedup (attn=${attnFrac} of step): ~${stepSpeedup.toFixed(2)}x`,
);
gpu.destroy();

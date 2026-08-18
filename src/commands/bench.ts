// `bench`: fixed-shape kernel timings, so a kernel change is measured instead of argued.
//
// Two numbers per suite:
//   - wall: one full iteration the way training runs it, dispatches batched into
//     a single compute pass. It includes the host-side graph build and gradient
//     readback, so it is the end-to-end feel of the op, not the kernel's cost.
//   - the per-kernel table from startProfile(), which is GPU time only. This is
//     the number to compare when A/B-ing a kernel. Profiling puts each dispatch
//     in its own pass, which inflates the total, so read the table as a split
//     between kernels rather than as a duration.
//
// Shapes default to the published Minueza-3-95M-Base geometry (hidden 640,
// 10 query heads over 5 KV heads, head-dim 64, window 1024, vocab 32768), so a
// run here is comparable to the throughput table in agents.md.

import type { Command, Values } from "../cli/args.ts";
import {
  attention,
  crossEntropy,
  linear,
  mulberry32,
  randn,
  rmsNorm,
  Tensor,
} from "../model/autograd.ts";
import { initWebGPU, type KernelTime, type WebGPUBackend } from "../backend/webgpu.ts";

const SUITES = ["attention", "gemm", "ce", "norm", "all"] as const;

function randTensor(shape: number[], rng: () => number, scale = 0.5): Tensor {
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * scale;
  return t;
}

interface Case {
  name: string;
  detail: string;
  /** Build the graph and return the loss-like scalar/tensor to backward from. */
  run: (gpu: WebGPUBackend) => Tensor;
  /** Tensors whose gradients the backward pass must produce (kept alive). */
  inputs: Tensor[];
  backward: boolean;
}

/** One timed pass: forward (+ backward), then a fence. */
async function once(gpu: WebGPUBackend, c: Case) {
  for (const t of c.inputs) t.zeroGrad();
  const out = c.run(gpu);
  if (c.backward) {
    out.grad.fill(1 / out.grad.length);
    gpu.seedGradFromHost(out);
    out._backward();
  }
  await gpu.sync([]);
}

function fmt(ms: number): string {
  return ms >= 100 ? ms.toFixed(0) : ms >= 10 ? ms.toFixed(1) : ms.toFixed(2);
}

function table(rows: string[][]) {
  const w = rows[0].map((_, i) => Math.max(...rows.map((r) => r[i].length)));
  for (const r of rows) {
    console.log(r.map((c, i) => (i === 0 ? c.padEnd(w[i]) : c.padStart(w[i]))).join("  "));
  }
}

function median(xs: number[]): number {
  const a = xs.slice().sort((x, y) => x - y);
  const h = a.length >> 1;
  return a.length % 2 ? a[h] : (a[h - 1] + a[h]) / 2;
}

/**
 * Time one case, reporting the MINIMUM over `iters` runs alongside the median.
 *
 * A shared laptop GPU has no quiet floor: background compositing, another
 * process's memory traffic and thermal drift all add time and none of them
 * subtract it, so the mean of a run drifts by 2-3x between invocations while
 * the minimum stays put. The minimum is the closest thing to the kernel's own
 * cost; the median beside it shows how noisy the machine was while measuring.
 * Compare minima across variants and treat a change smaller than the min/median
 * spread as unmeasured.
 */
async function timeCase(
  gpu: WebGPUBackend,
  c: Case,
  iters: number,
  warmup: number,
): Promise<{ wallMs: number; wallMedianMs: number; kernels: KernelTime[] }> {
  for (let i = 0; i < warmup; i++) await once(gpu, c);

  const walls: number[] = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    await once(gpu, c);
    walls.push(performance.now() - t0);
  }

  // Per-kernel: profile each iteration on its own so the per-label minimum can
  // be taken too (stopProfile totals would otherwise pool good and bad runs).
  const best = new Map<string, { ms: number; count: number }>();
  if (gpu.timestampSupported) {
    for (let i = 0; i < iters; i++) {
      gpu.startProfile();
      await once(gpu, c);
      const r = gpu.stopProfile();
      if (r.overflow) console.log("  (profile slots overflowed: split is partial)");
      for (const k of r.kernels) {
        const prev = best.get(k.label);
        if (!prev || k.ms < prev.ms) best.set(k.label, { ms: k.ms, count: k.count });
      }
    }
  }
  const kernels = [...best]
    .map(([label, v]) => ({ label, ms: v.ms, count: v.count }))
    .sort((a, b) => b.ms - a.ms);
  return { wallMs: Math.min(...walls), wallMedianMs: median(walls), kernels };
}

function buildCases(v: Values): Case[] {
  const T = v.num("seq-len");
  const hd = v.num("head-dim");
  const hidden = v.num("hidden");
  const window = v.num("window");
  const vocab = v.num("vocab");
  const Hq = Math.floor(hidden / hd);
  const Hkv = Math.max(1, Math.floor(Hq / 2));
  const rng = mulberry32(0x5eed);
  const suite = v.str("suite");
  const want = (s: string) => suite === "all" || suite === s;
  const cases: Case[] = [];

  if (want("attention")) {
    // The two layer flavors gemma3 interleaves: 5 sliding-window to 1 dense.
    for (const [label, w] of [["swa", window], ["dense", 0]] as const) {
      const q = randTensor([T, Hq * hd], rng);
      const k = randTensor([T, Hkv * hd], rng);
      const vv = randTensor([T, Hkv * hd], rng);
      cases.push({
        name: `attention-${label}`,
        detail: `T=${T} Hq=${Hq} Hkv=${Hkv} hd=${hd} W=${w}`,
        inputs: [q, k, vv],
        backward: true,
        run: () => attention(q, k, vv, T, Hq, Hkv, hd, w),
      });
    }
  }

  if (want("gemm")) {
    // The three GEMM shapes of one gemma3 block plus the tied readout, which is
    // the single largest matmul in a step.
    const shapes: [string, number, number][] = [
      ["qkv", (Hq + 2 * Hkv) * hd, hidden],
      ["ffn-up", Math.round((hidden * 4) / 32) * 32, hidden],
      ["readout", vocab, hidden],
    ];
    for (const [label, outDim, inDim] of shapes) {
      const x = randTensor([T, inDim], rng);
      const w = randTensor([outDim, inDim], rng, 1 / Math.sqrt(inDim));
      cases.push({
        name: `gemm-${label}`,
        detail: `[${T},${inDim}] x [${outDim},${inDim}]ᵀ`,
        inputs: [x, w],
        backward: true,
        run: () => linear(x, w),
      });
    }
  }

  if (want("ce")) {
    const logits = randTensor([T, vocab], rng);
    const targets = Array.from({ length: T }, (_, i) => (i * 7919) % vocab);
    cases.push({
      name: "cross-entropy",
      detail: `T=${T} V=${vocab}`,
      inputs: [logits],
      backward: true,
      run: () => crossEntropy(logits, targets),
    });
  }

  if (want("norm")) {
    const x = randTensor([T, hidden], rng);
    const w = randTensor([hidden], rng, 0.1);
    cases.push({
      name: "rmsnorm",
      detail: `[${T},${hidden}]`,
      inputs: [x, w],
      backward: true,
      run: () => rmsNorm(x, w, 1e-6),
    });
  }

  return cases;
}

async function run(v: Values) {
  const gpu = await initWebGPU();
  if (!gpu) throw new Error("no WebGPU adapter: bench needs Deno with a GPU");
  const cases = buildCases(v);
  if (cases.length === 0) throw new Error(`--suite ${v.str("suite")} selected no cases`);
  const iters = v.num("iters");
  const warmup = v.num("warmup");
  const json = v.bool("json");
  const results: {
    name: string;
    detail: string;
    wallMs: number;
    wallMedianMs: number;
    kernels: KernelTime[];
  }[] = [];

  gpu.install();
  try {
    if (!json) {
      console.log(`device: ${gpu.adapterName}`);
      console.log(
        `iters: ${iters} (after ${warmup} warmup), forward+backward per iteration; ` +
          `times are the MINIMUM over iterations\n`,
      );
    }
    for (const c of cases) {
      const { wallMs, wallMedianMs, kernels } = await timeCase(gpu, c, iters, warmup);
      results.push({ name: c.name, detail: c.detail, wallMs, wallMedianMs, kernels });
      if (json) continue;
      console.log(`${c.name}  ${c.detail}`);
      console.log(`  wall ${fmt(wallMs)} ms/iter (min)   ${fmt(wallMedianMs)} (median)`);
      if (kernels.length > 0) {
        const total = kernels.reduce((a, k) => a + k.ms, 0);
        table([
          ["  kernel", "ms", "n", "share"],
          ...kernels.map((k) => [
            `  ${k.label}`,
            fmt(k.ms),
            String(k.count),
            `${((100 * k.ms) / total).toFixed(1)}%`,
          ]),
        ]);
      }
      console.log("");
    }
  } finally {
    gpu.destroy();
  }

  if (json) {
    console.log(JSON.stringify({ device: gpu.adapterName, iters, warmup, results }, null, 2));
  } else {
    console.log("total wall (min):");
    table([
      ["  case", "ms/iter"],
      ...results.map((r) => [`  ${r.name}`, fmt(r.wallMs)]),
      ["  SUM", fmt(results.reduce((a, r) => a + r.wallMs, 0))],
    ]);
  }
}

export const benchCommand: Command = {
  name: "bench",
  summary: "Time the WebGPU kernels at fixed shapes (attention, GEMM, cross-entropy, norm).",
  details:
    `Forward and backward per iteration, at the published 95M geometry by default, so a kernel
change can be A/B'd instead of argued about. Compare the per-kernel table: it is GPU time
from timestamp queries, which is what a kernel change moves. \`wall\` also carries the
host-side graph build and the gradient readback, so it moves less and later.

Every time reported is the MINIMUM over the iterations. On a shared GPU every source of
interference adds time and none subtracts it, so the floor is the signal and the median
printed beside it says how noisy the machine was. Treat a difference smaller than that
spread as unmeasured, and repeat the whole run a few times before believing a small win.

The numbers are device-specific. Record a baseline on the machine you are changing, on the
same shapes, before and after; comparing across machines says nothing.`,
  examples: [
    "bench",
    "bench --suite attention --seq-len 4096",
    "bench --json",
  ],
  flags: [
    {
      name: "suite",
      type: "string",
      default: "all",
      choices: [...SUITES],
      describe: "which kernel family to time",
    },
    { name: "seq-len", type: "number", default: 2048, describe: "context length T" },
    { name: "hidden", type: "number", default: 640, describe: "model width" },
    { name: "head-dim", type: "number", default: 64, describe: "attention head size" },
    {
      name: "window",
      type: "number",
      default: 1024,
      describe: "sliding-window size for the swa attention case (0 = dense)",
    },
    { name: "vocab", type: "number", default: 32768, describe: "vocabulary size (readout, CE)" },
    { name: "iters", type: "number", default: 8, describe: "timed iterations per case" },
    { name: "warmup", type: "number", default: 2, describe: "untimed iterations first" },
    { name: "json", type: "boolean", describe: "machine-readable output" },
  ],
  run: run,
};

// Finite-difference gradient check for the CPU autograd ops — the validation
// gate described in docs/HANDOFF.md and CONTRIBUTING.md.
//
// For every op: build small random inputs, take loss = sum(r ⊙ f(x)) with fixed
// random weights r, backprop the analytic gradient, then perturb each input
// element by ±ε and compare (f(x+ε) − f(x−ε)) / 2ε against it. A whole-model
// check does the same on sampled Qwen3 parameter elements through the real
// forward + crossEntropy. A negative control with a deliberately wrong backward
// proves the harness actually rejects bad gradients.
//
// Run:  deno run tests/gradcheck.ts
//       node --experimental-strip-types tests/gradcheck.ts
//       bun tests/gradcheck.ts

import {
  add,
  attention,
  backward,
  crossEntropy,
  embedding,
  linear,
  mul,
  mulberry32,
  randn,
  rmsNorm,
  rmsNormHeads,
  rope,
  silu,
  Tensor,
} from "../src/model/autograd.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import type { Qwen3Config } from "../src/model/config.ts";

// Storage is f32, so the finite difference carries ~1e-6 forward-rounding noise;
// ε=1e-2 keeps both that noise (δ/2ε) and the O(ε²) truncation well under tol.
const EPS = 1e-2;
const RTOL = 1.5e-2;
const ATOL = 2e-3;

let failures = 0;

function randTensor(shape: number[], rng: () => number, scale = 0.8): Tensor {
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * scale;
  return t;
}

/** Like autograd.backward() but seeds a full gradient on a non-scalar output. */
function backwardFrom(out: Tensor, gradOut: Float32Array) {
  const topo: Tensor[] = [];
  const seen = new Set<Tensor>();
  const build = (t: Tensor) => {
    if (seen.has(t)) return;
    seen.add(t);
    for (const p of t._prev) build(p);
    topo.push(t);
  };
  build(out);
  out.grad.set(gradOut);
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

function weightedLoss(out: Tensor, r: Float32Array): number {
  let s = 0;
  for (let i = 0; i < out.data.length; i++) s += out.data[i] * r[i];
  return s;
}

interface CheckOpts {
  eps?: number;
  rtol?: number;
  atol?: number;
  /** Check at most this many elements per input (all when omitted). */
  sample?: number;
  /** Expect the check to FAIL (negative control). */
  expectFail?: boolean;
}

/**
 * inputs: leaf tensors whose gradients get verified.
 * fwd: rebuilds the graph from the inputs' current data.
 */
function fdCheck(name: string, inputs: Tensor[], fwd: () => Tensor, o: CheckOpts = {}) {
  const eps = o.eps ?? EPS;
  const rtol = o.rtol ?? RTOL;
  const atol = o.atol ?? ATOL;

  const rngR = mulberry32(0xbeef);
  const out0 = fwd();
  const r = new Float32Array(out0.data.length);
  for (let i = 0; i < r.length; i++) r[i] = out0.data.length === 1 ? 1 : (rngR() * 2 - 1);

  for (const x of inputs) x.zeroGrad();
  backwardFrom(out0, r);
  const analytic = inputs.map((x) => x.grad.slice());

  let maxAbs = 0;
  let maxRel = 0;
  let bad = 0;
  let checked = 0;
  const rngPick = mulberry32(0xcafe);

  for (let k = 0; k < inputs.length; k++) {
    const x = inputs[k];
    let idxs: number[];
    if (o.sample !== undefined && x.data.length > o.sample) {
      const set = new Set<number>();
      while (set.size < o.sample) set.add(Math.floor(rngPick() * x.data.length));
      idxs = [...set];
    } else {
      idxs = [...x.data.length === 0 ? [] : Array(x.data.length).keys()];
    }

    for (const i of idxs) {
      const orig = x.data[i];
      // f32 storage rounds the perturbed values; divide by the step that was
      // actually applied, not by 2ε, to avoid a spurious O(eps_f32/ε) error.
      const xp = Math.fround(orig + eps);
      const xm = Math.fround(orig - eps);
      x.data[i] = xp;
      const lp = weightedLoss(fwd(), r);
      x.data[i] = xm;
      const lm = weightedLoss(fwd(), r);
      x.data[i] = orig;

      const fd = (lp - lm) / (xp - xm);
      const g = analytic[k][i];
      const abs = Math.abs(fd - g);
      const rel = abs / Math.max(Math.abs(fd), Math.abs(g), 1e-12);
      if (abs > maxAbs) maxAbs = abs;
      if (rel > maxRel && abs > atol) maxRel = rel;
      checked++;
      if (abs > atol + rtol * Math.max(Math.abs(fd), Math.abs(g))) {
        bad++;
        if (bad <= 3 && !o.expectFail) {
          console.log(`    MISMATCH ${name} input#${k}[${i}]: analytic=${g} fd=${fd}`);
        }
      }
    }
  }

  const failed = bad > 0;
  const ok = o.expectFail ? failed : !failed;
  if (!ok) failures++;
  const status = ok ? "ok " : "FAIL";
  const note = o.expectFail ? " (negative control: harness must flag this)" : "";
  console.log(
    `  ${status} ${name.padEnd(24)} ${String(checked).padStart(5)} elems  ` +
      `maxAbs=${maxAbs.toExponential(2)}  maxRel=${maxRel.toExponential(2)}${note}`,
  );
}

function main() {
  console.log("=== finite-difference gradient checks (CPU reference ops) ===\n");
  const rng = mulberry32(1234);

  {
    const a = randTensor([4, 6], rng);
    const b = randTensor([4, 6], rng);
    fdCheck("add", [a, b], () => add(a, b));
  }
  {
    const a = randTensor([4, 6], rng);
    const b = randTensor([4, 6], rng);
    fdCheck("mul", [a, b], () => mul(a, b));
  }
  {
    const x = randTensor([5, 7], rng, 1.5); // wider spread exercises both tails
    fdCheck("silu", [x], () => silu(x));
  }
  {
    const x = randTensor([3, 5], rng);
    const w = randTensor([4, 5], rng);
    fdCheck("linear", [x, w], () => linear(x, w));
  }
  {
    const x = randTensor([4, 6], rng);
    const w = Tensor.zeros([6], true);
    for (let i = 0; i < 6; i++) w.data[i] = 1 + 0.3 * randn(rng); // near init value 1
    fdCheck("rmsNorm", [x, w], () => rmsNorm(x, w, 1e-6));
  }
  {
    const T = 3, H = 2, hd = 4;
    const x = randTensor([T, H * hd], rng);
    const w = Tensor.zeros([hd], true);
    for (let i = 0; i < hd; i++) w.data[i] = 1 + 0.3 * randn(rng);
    fdCheck("rmsNormHeads", [x, w], () => rmsNormHeads(x, w, T, H, hd, 1e-6));
  }
  {
    const w = randTensor([7, 4], rng);
    const ids = [0, 3, 3, 6, 1, 3]; // repeats exercise gradient accumulation
    fdCheck("embedding", [w], () => embedding(w, ids));
  }
  {
    const T = 4, H = 2, hd = 6;
    const x = randTensor([T, H * hd], rng);
    fdCheck("rope", [x], () => rope(x, T, H, hd, 10000));
    fdCheck("rope(posOffset)", [x], () => rope(x, T, H, hd, 10000, 5));
  }
  {
    const T = 5, Hq = 4, Hkv = 2, hd = 6; // GQA group = 2
    const q = randTensor([T, Hq * hd], rng);
    const k = randTensor([T, Hkv * hd], rng);
    const v = randTensor([T, Hkv * hd], rng);
    fdCheck("attention(GQA)", [q, k, v], () => attention(q, k, v, T, Hq, Hkv, hd));
  }
  {
    const T = 4, V = 9;
    const logits = randTensor([T, V], rng);
    const targets = [2, 7, 2, 0];
    fdCheck("crossEntropy", [logits], () => crossEntropy(logits, targets));
  }

  // Whole model: sampled parameter elements through forward + crossEntropy.
  {
    const cfg: Qwen3Config = {
      vocabSize: 13,
      hiddenSize: 8,
      nLayers: 2,
      nHeads: 2,
      nKVHeads: 1,
      headDim: 4,
      ffnDim: 16,
      ropeBase: 10000,
      rmsEps: 1e-6,
      maxSeq: 16,
      tieEmbeddings: true,
    };
    const model = new Qwen3Model(cfg, mulberry32(99));
    const ids = [1, 5, 2, 9, 5];
    const targets = [5, 2, 9, 5, 12];

    // Per-element FD needs the perturbation to be small *relative to the
    // weights*; at the real 0.02 embedding init, ε=1e-2 is a ~50% perturbation
    // of an 8-wide RMSNorm'd row and curvature dominates the difference
    // quotient. So check per-element grads at O(1) parameter scale (same graph,
    // same code paths), then separately check the true init below.
    const rescale = mulberry32(4321);
    for (const p of model.params()) {
      const isNorm = p.shape.length === 1;
      for (let i = 0; i < p.data.length; i++) {
        p.data[i] = isNorm ? 1 + 0.2 * randn(rescale) : 0.5 * randn(rescale);
      }
    }
    fdCheck(
      "qwen3 full model",
      model.params(),
      () => crossEntropy(model.forward(ids), targets),
      { sample: 6, rtol: 2e-2, atol: 3e-3 },
    );

    // Directional derivative at the *real* init scale: perturb all parameters
    // along a random unit direction d and compare (L(θ+εd) − L(θ−εd)) / 2ε to
    // ⟨∇L, d⟩. The aggregate stays in the linear regime even for 0.02-scale
    // weights, and it exercises graph-level wiring (tied embeddings, fan-out).
    const fresh = new Qwen3Model(cfg, mulberry32(99));
    const params = fresh.params();
    const lossOf = () => crossEntropy(fresh.forward(ids), targets);
    for (const p of params) p.zeroGrad();
    backward(lossOf(), 1);
    const dirRng = mulberry32(777);
    let dirFails = 0;
    for (let trial = 0; trial < 4; trial++) {
      const dirs = params.map((p) => {
        const d = new Float32Array(p.size);
        for (let i = 0; i < d.length; i++) d[i] = randn(dirRng);
        return d;
      });
      let norm = 0;
      for (const d of dirs) for (let i = 0; i < d.length; i++) norm += d[i] * d[i];
      norm = Math.sqrt(norm);
      const eps = 2e-3;
      let dot = 0;
      for (let k = 0; k < params.length; k++) {
        for (let i = 0; i < dirs[k].length; i++) {
          dirs[k][i] /= norm;
          dot += params[k].grad[i] * dirs[k][i];
        }
      }
      const saved = params.map((p) => p.data.slice());
      const evalAt = (sign: number) => {
        for (let k = 0; k < params.length; k++) {
          for (let i = 0; i < params[k].size; i++) {
            params[k].data[i] = saved[k][i] + sign * eps * dirs[k][i];
          }
        }
        return lossOf().data[0];
      };
      const lp = evalAt(1);
      const lm = evalAt(-1);
      for (let k = 0; k < params.length; k++) params[k].data.set(saved[k]);
      const fd = (lp - lm) / (2 * eps);
      const rel = Math.abs(fd - dot) / Math.max(Math.abs(fd), Math.abs(dot), 1e-12);
      if (rel > 5e-2 && Math.abs(fd - dot) > 1e-3) dirFails++;
    }
    if (dirFails > 0) failures++;
    console.log(
      `  ${dirFails === 0 ? "ok " : "FAIL"} qwen3 @ real init (directional, 4 trials)`,
    );
  }

  // Negative control: a silu whose backward drops the x·σ'(x) term. The harness
  // is only trusted if it catches this.
  {
    const brokenSilu = (x: Tensor): Tensor => {
      const out = Tensor.zeros(x.shape);
      const sig = new Float32Array(x.data.length);
      for (let i = 0; i < x.data.length; i++) {
        const s = 1 / (1 + Math.exp(-x.data[i]));
        sig[i] = s;
        out.data[i] = x.data[i] * s;
      }
      out._prev = [x];
      out._backward = () => {
        for (let i = 0; i < x.data.length; i++) x.grad[i] += out.grad[i] * sig[i];
      };
      return out;
    };
    const x = randTensor([5, 7], rng, 1.5);
    fdCheck("broken silu", [x], () => brokenSilu(x), { expectFail: true });
  }

  // backward() itself: the seed argument must scale leaf grads linearly.
  {
    const x = randTensor([3, 4], rng);
    const w = randTensor([2, 4], rng);
    const loss = crossEntropy(linear(x, w), [1, 0, 1]);
    backward(loss, 0.5);
    const half = w.grad.slice();
    x.zeroGrad();
    w.zeroGrad();
    const loss2 = crossEntropy(linear(x, w), [1, 0, 1]);
    backward(loss2, 1);
    let ok = true;
    for (let i = 0; i < half.length; i++) {
      if (Math.abs(half[i] * 2 - w.grad[i]) > 1e-6 + 1e-4 * Math.abs(w.grad[i])) ok = false;
    }
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} backward(seed) scales leaf grads linearly`);
  }

  console.log(
    failures === 0 ? "\n=== all gradient checks passed ===" : `\n=== ${failures} FAILURES ===`,
  );
  if (failures > 0) {
    // deno-lint-ignore no-explicit-any
    const proc = (globalThis as any).process;
    if (proc?.exit) proc.exit(1);
  }
}

main();

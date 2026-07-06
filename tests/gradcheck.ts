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
  gelu,
  linear,
  mul,
  mulberry32,
  randn,
  rmsNorm,
  rmsNormHeads,
  rope,
  scale,
  silu,
  Tensor,
} from "../src/model/autograd.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import type { Qwen3Config } from "../src/model/config.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { applyQKClip, qkLogitScale } from "../src/train/qk_clip.ts";
import { buildGGUF } from "../src/export/export_gguf.ts";
import { loadQwen3FromGGUF } from "../src/export/load_gguf.ts";
import { dequantize } from "../src/gguf/quantize.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { Muon } from "../src/train/muon.ts";
import { trainLM } from "../src/train/trainer.ts";
import { diskTokenSource, memTokenSource, tokenBytes, writeTokenFile } from "../src/data/tokens.ts";
import { assistantLossMask, maskedTargets } from "../src/data/chat.ts";
import { f16BitsToF32, f32ToF16Bits } from "../src/backend/f16.ts";

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

async function main() {
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
    const x = randTensor([5, 7], rng, 1.5);
    fdCheck("gelu", [x], () => gelu(x));
  }
  {
    const x = randTensor([4, 6], rng);
    fdCheck("scale", [x], () => scale(x, 2.5));
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

  // WSD schedule shape (pure, runtime-agnostic): warmup ramp, stable plateau,
  // linear cooldown to the floor. Guards the phase boundaries that the GPU/CPU
  // trajectory parity in gpu_parity.ts then exercises end-to-end.
  {
    const s = wsdSchedule({ warmupSteps: 4, stableSteps: 3, cooldownSteps: 4, minScale: 0.1 });
    const got = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map(s);
    // warmup (step+1)/4: .25 .5 .75 1 | stable: 1 1 1 | cooldown 1->0.1 over 4:
    // 1-.9*(k/4) for k=1..4 => .775 .55 .325 .1 | past schedule: .1
    const want = [0.25, 0.5, 0.75, 1, 1, 1, 1, 0.775, 0.55, 0.325, 0.1, 0.1];
    let ok = got.length === want.length;
    for (let i = 0; i < want.length; i++) if (Math.abs(got[i] - want[i]) > 1e-6) ok = false;
    // Edge cases: no warmup starts at 1; no cooldown holds 1 then drops to floor.
    const noWarm = wsdSchedule({ warmupSteps: 0, stableSteps: 2, cooldownSteps: 2 });
    if (noWarm(0) !== 1) ok = false;
    const noCool = wsdSchedule({ warmupSteps: 0, stableSteps: 2, cooldownSteps: 0, minScale: 0 });
    if (noCool(1) !== 1 || noCool(2) !== 0) ok = false;
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} WSD schedule shape (warmup/stable/cooldown)`);
  }

  // MuonClip / QK-logit clip (pure, runtime-agnostic): capping the per-layer
  // logit-scale proxy at tau by rescaling qNorm/kNorm. Verifies the proxy math,
  // that clipping lands exactly on tau, symmetric q/k scaling, and no-op below.
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
    let ok = true;
    // At init qNorm=kNorm=ones -> proxy = (1/sqrt(hd))*sqrt(hd) = 1. tau=0.5 clips.
    if (
      Math.abs(qkLogitScale(model.layers[0].qNorm.data, model.layers[0].kNorm.data, 4) - 1) > 1e-6
    ) {
      ok = false;
    }
    // Inflate layer 0's norms so its proxy is well above tau; leave layer 1 at 1.
    for (let d = 0; d < 4; d++) {
      model.layers[0].qNorm.data[d] = 3;
      model.layers[0].kNorm.data[d] = 2;
    }
    const before = qkLogitScale(model.layers[0].qNorm.data, model.layers[0].kNorm.data, 4); // 6
    const tau = 0.5;
    const qBefore = model.layers[0].qNorm.data.slice();
    const kBefore = model.layers[0].kNorm.data.slice();
    const clipped = applyQKClip(model, tau);
    if (clipped !== 2) ok = false; // both layers exceed 0.5 (layer1 proxy=1)
    const after = qkLogitScale(model.layers[0].qNorm.data, model.layers[0].kNorm.data, 4);
    if (Math.abs(after - tau) > 1e-6) ok = false; // lands exactly on tau
    const f = Math.sqrt(tau / before);
    for (let d = 0; d < 4; d++) {
      if (Math.abs(model.layers[0].qNorm.data[d] - qBefore[d] * f) > 1e-6) ok = false;
      if (Math.abs(model.layers[0].kNorm.data[d] - kBefore[d] * f) > 1e-6) ok = false;
    }
    // Re-clipping is now a no-op (already at tau, within fp) — proxy stays <= tau.
    applyQKClip(model, tau);
    if (qkLogitScale(model.layers[0].qNorm.data, model.layers[0].kNorm.data, 4) > tau + 1e-6) {
      ok = false;
    }
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} MuonClip / QK-logit clip (proxy, cap-at-tau, symmetric)`);
  }

  // GGUF checkpoint round-trip: export a model+tokenizer, load it back with
  // loadQwen3FromGGUF, and confirm config, tokenizer, weights, and the actual
  // forward logits all survive. f32 is bit-exact; q8_0 is lossy so its weights
  // (and logits) are checked within quant tolerance. Inner dims are multiples
  // of 32 so q8_0 stores without the f16 fallback — exercising the loader's
  // quantized dequant path.
  {
    const tok = new BPETokenizer();
    tok.train("the cat sat on the mat. the dog ran to the cat and the ball.".repeat(6), 300);
    const cfg: Qwen3Config = {
      vocabSize: tok.vocabSize,
      hiddenSize: 32,
      nLayers: 2,
      nHeads: 1,
      nKVHeads: 1,
      headDim: 32,
      ffnDim: 64,
      ropeBase: 10000,
      rmsEps: 1e-6,
      maxSeq: 32,
      tieEmbeddings: true,
    };
    const model = new Qwen3Model(cfg, mulberry32(77));
    const ids = tok.encode("the cat sat").slice(0, 6);
    const refLogits = model.forward(ids).data.slice();
    const refParams = model.params().map((p) => p.data.slice());

    let ok = true;
    for (const [quant, wtol, ltol] of [["f32", 0, 0], ["q8_0", 0.05, 0.05]] as const) {
      const bytes = buildGGUF(model, tok.export(), cfg, { quant });
      const { model: m2, cfg: cfg2, tokenizer: tok2 } = loadQwen3FromGGUF(bytes);

      // Config round-trips: integer/bool fields exactly; rmsEps and ropeBase
      // are stored as f32 metadata, so they match only to f32 precision.
      const floatKeys = new Set<keyof Qwen3Config>(["rmsEps", "ropeBase"]);
      for (const k of Object.keys(cfg) as (keyof Qwen3Config)[]) {
        const a = cfg2[k], b = cfg[k];
        const bad = floatKeys.has(k)
          ? Math.abs(Number(a) - Number(b)) > 1e-6 * Math.abs(Number(b))
          : a !== b;
        if (bad) {
          console.log(`    cfg mismatch ${k}: ${a} != ${b}`);
          ok = false;
        }
      }
      // Tokenizer round-trips (same ids, decode inverts).
      const s = "the dog ran";
      if (tok2.encode(s).join(",") !== tok.encode(s).join(",")) ok = false;
      if (tok2.decode(tok2.encode(s)) !== s) ok = false;
      // Weights within tolerance for the quant.
      const p2 = m2.params();
      for (let i = 0; i < refParams.length; i++) {
        for (let j = 0; j < refParams[i].length; j++) {
          const a = p2[i].data[j], b = refParams[i][j];
          if (Math.abs(a - b) > wtol + wtol * Math.abs(b) + (quant === "f32" ? 0 : 1e-6)) {
            ok = false;
          }
        }
      }
      // The forward pass reproduces (bit-exact for f32, within tol for q8_0).
      const l2 = m2.forward(ids).data;
      for (let i = 0; i < refLogits.length; i++) {
        if (Math.abs(l2[i] - refLogits[i]) > ltol + ltol * Math.abs(refLogits[i]) + 1e-6) {
          ok = false;
        }
      }
    }
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} GGUF checkpoint round-trip (config+tokenizer+weights+forward)`,
    );
  }

  // External / resumed tokenizer import: control tokens must survive the GGUF
  // round-trip (buildGGUF writes token_type; tokenizerFromGGUF recovers the
  // specials) so a resumed chat model still tokenizes ChatML atomically. And
  // dequantize must reject unsupported (k-quant) types loudly, not silently
  // return zeros. Both are the Tier-2 external-GGUF loader's guardrails.
  {
    const tok = new BPETokenizer();
    tok.train(
      "the cat sat on the mat. the dog ran to the cat.".repeat(8),
      320,
      ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    );
    const cfg: Qwen3Config = {
      vocabSize: tok.vocabSize,
      hiddenSize: 32,
      nLayers: 1,
      nHeads: 1,
      nKVHeads: 1,
      headDim: 32,
      ffnDim: 64,
      ropeBase: 10000,
      rmsEps: 1e-6,
      maxSeq: 32,
      tieEmbeddings: true,
    };
    const model = new Qwen3Model(cfg, mulberry32(9));
    const chat = "<|im_start|>user\nthe cat<|im_end|>\n";
    const before = tok.encode(chat);
    const { tokenizer: tok2 } = loadQwen3FromGGUF(
      buildGGUF(model, tok.export(), cfg, { quant: "f16" }),
    );
    const imStart = tok2.idOf("<|im_start|>");
    const imEnd = tok2.idOf("<|im_end|>");
    let ok = imStart !== undefined && imEnd !== undefined;
    const after = tok2.encode(chat);
    if (after.join(",") !== before.join(",")) ok = false; // specials still atomic post-load
    if (after.filter((id) => id === imStart || id === imEnd).length !== 2) ok = false;
    // BF16 import path: the high-16-bits float format external Qwen3 base GGUFs
    // ship in. These values are exactly bf16-representable, so dequant is exact.
    const bfVals = [1.5, -2.25, 0, 3];
    const bf = new Uint8Array(bfVals.length * 2);
    const bfdv = new DataView(bf.buffer);
    const conv = new DataView(new ArrayBuffer(4));
    for (let i = 0; i < bfVals.length; i++) {
      conv.setFloat32(0, bfVals[i], true);
      bfdv.setUint16(i * 2, conv.getUint32(0, true) >>> 16, true);
    }
    const bfOut = dequantize(30 as never, bf, bfVals.length); // 30 = BF16
    for (let i = 0; i < bfVals.length; i++) if (bfOut[i] !== bfVals[i]) ok = false;
    // Unsupported quant type -> loud throw, not a silent zero tensor.
    let threw = false;
    try {
      dequantize(12 as never, new Uint8Array(64), 32); // 12 = Q4_K (k-quant)
    } catch {
      threw = true;
    }
    if (!threw) ok = false;
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} external GGUF import (specials + BF16 + unsupported-quant guard)`,
    );
  }

  // Special-token encoding: ChatML control tokens must encode atomically (one id
  // each), not shred into bytes, so a chat model learns real turn boundaries.
  // Ordinary text (no specials) must be unaffected, and export()/fromData() must
  // preserve the behavior.
  {
    const tok = new BPETokenizer();
    tok.train(
      "the cat sat on the mat. the dog ran to the cat and the ball.".repeat(6),
      320,
      ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    );
    const imStart = tok.idOf("<|im_start|>");
    const imEnd = tok.idOf("<|im_end|>");
    let ok = imStart !== undefined && imEnd !== undefined;
    const chat = "<|im_start|>user\nthe cat<|im_end|>\n<|im_start|>assistant\nthe dog<|im_end|>\n";
    const ids = tok.encode(chat);
    if (ids.filter((id) => id === imStart || id === imEnd).length !== 4) ok = false;
    if (ids[0] !== imStart) ok = false; // leading special is one id, not bytes
    // Ordinary text round-trips and matches the special-free path.
    const plain = "the dog ran";
    if (tok.decode(tok.encode(plain)) !== plain) ok = false;
    // export/fromData preserves the special encoding exactly.
    const rebuilt = BPETokenizer.fromData(JSON.parse(JSON.stringify(tok.export())));
    if (rebuilt.encode(chat).join(",") !== ids.join(",")) ok = false;
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} special-token encoding (ChatML atomic + roundtrip)`);
  }

  // Assistant-only loss mask: over a rendered ChatML conversation, supervise the
  // assistant turn's content + its terminating <|im_end|>, and nothing else.
  {
    const tok = new BPETokenizer();
    tok.train(
      "the cat sat on the mat. the dog ran to the cat and the ball.".repeat(6),
      320,
      ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    );
    const imStart = tok.idOf("<|im_start|>")!;
    const imEnd = tok.idOf("<|im_end|>")!;
    const chat = "<|im_start|>user\nthe cat<|im_end|>\n<|im_start|>assistant\nthe dog<|im_end|>\n";
    const ids = tok.encode(chat);
    const mask = assistantLossMask(ids, imStart, imEnd, (x) => tok.decode(x));
    let ok = mask.length === ids.length;
    const supText = tok.decode(ids.filter((_, k) => mask[k] === 1));
    if (!supText.includes("the dog")) ok = false; // assistant content supervised
    if (supText.includes("cat")) ok = false; // user content NOT supervised
    if (supText.includes("assistant")) ok = false; // "assistant\n" header excluded
    // The assistant turn's <|im_end|> is supervised (learn to stop); the user
    // turn's <|im_end|> is not.
    const ends = ids.map((id, k) => (id === imEnd ? k : -1)).filter((k) => k >= 0);
    if (mask[ends[0]] !== 0 || mask[ends[1]] !== 1) ok = false;
    // maskedTargets turns every mask-0 position into ignore-index -1.
    const ignores = maskedTargets(ids, mask).filter((t) => t === -1).length;
    if (ignores !== mask.filter((m) => m === 0).length) ok = false;
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} assistant-only loss mask (assistant span + stop token)`);
  }

  // muP coordinate check: the standard muP diagnostic. Sweep width at fixed
  // base width and measure the readout logit RMS at init. Standard init grows
  // ~sqrt(width) (the blow-up that breaks LR transfer); muP init pins it flat.
  // Then a few Muon+AdamW steps at CONSTANT lr must keep it bounded across
  // widths (evidence Muon transfers without width-LR scaling — see mup.ts).
  {
    const baseWidth = 32, headDim = 8, vocab = 96;
    const widths = [32, 64, 128, 256];
    const cfgFor = (h: number): Qwen3Config => ({
      vocabSize: vocab,
      hiddenSize: h,
      nLayers: 2,
      nHeads: h / headDim,
      nKVHeads: Math.max(1, h / headDim / 2),
      headDim,
      ffnDim: 4 * h,
      ropeBase: 10000,
      rmsEps: 1e-6,
      maxSeq: 32,
      tieEmbeddings: true,
    });
    const rms = (a: Float32Array) => {
      let s = 0;
      for (const v of a) s += v * v;
      return Math.sqrt(s / a.length);
    };
    const ids = Array.from({ length: 16 }, (_, i) => (i * 7 + 3) % vocab);

    const stdRms: number[] = [], mupRms: number[] = [];
    for (const h of widths) {
      const cfg = cfgFor(h);
      stdRms.push(rms(new Qwen3Model(cfg, mulberry32(1)).forward(ids).data));
      mupRms.push(rms(new Qwen3Model(cfg, mulberry32(1), { baseWidth }).forward(ids).data));
    }
    const ratio = (xs: number[]) => Math.max(...xs) / Math.min(...xs);
    let ok = true;
    // muP init flat across an 8x width range; standard init clearly grows.
    if (ratio(mupRms) > 1.5) ok = false;
    if (ratio(stdRms) < 2) ok = false;

    // Post-step boundedness at constant lr with muP init (no width-driven
    // blow-up). Two widths (4x apart) keep the CPU cost low; the init sweep
    // above is the primary gate.
    const rngTok = mulberry32(9);
    const tokens = Array.from({ length: 400 }, () => Math.floor(rngTok() * vocab));
    for (const h of [32, 128]) {
      const cfg = cfgFor(h);
      const model = new Qwen3Model(cfg, mulberry32(1), { baseWidth });
      const g = model.paramGroups();
      trainLM(model, {
        tokens,
        seqLen: 16,
        steps: 4,
        batchPerStep: 1,
        optimizer: new Muon(g.muon, g.aux, {
          lr: 0.02,
          momentum: 0.95,
          aux: { lr: 3e-3, clip: 1.0 },
        }),
        logEvery: 100,
        rng: mulberry32(3),
      });
      const r = rms(model.forward(ids).data);
      if (!(r > 0.01 && r < 2)) ok = false; // bounded, no explosion/collapse
    }
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} muP coordinate check (init logit RMS: std ${
        ratio(stdRms).toFixed(1)
      }x ` +
        `vs muP ${ratio(mupRms).toFixed(2)}x across 8x width)`,
    );
  }

  // Streaming token source: a disk-backed source must return byte-for-byte the
  // same windows as an in-memory one, including at the corpus tail, for both the
  // u16 and u32 widths. Round-trips through writeTokenFile -> diskTokenSource.
  {
    const os = await import("node:os");
    const fs = await import("node:fs");
    let ok = true;
    for (const [vocab, bpt] of [[300, 2], [70000, 4]] as const) {
      if (tokenBytes(vocab) !== bpt) ok = false;
      const rngT = mulberry32(vocab);
      const toks = Array.from({ length: 500 }, () => Math.floor(rngT() * vocab));
      const path = `${os.tmpdir()}/gguf-trainer-tokens-${bpt}.bin`;
      await writeTokenFile(path, toks, bpt);
      const disk = await diskTokenSource(path, bpt);
      const mem = memTokenSource(toks);
      if (disk.length !== toks.length) ok = false;
      // Several windows incl. one ending exactly at the tail.
      for (const [start, len] of [[0, 16], [123, 40], [toks.length - 32, 32]] as const) {
        const a = disk.window(start, len), b = mem.window(start, len);
        for (let i = 0; i < len; i++) if (a[i] !== b[i]) ok = false;
      }
      disk.close();
      fs.unlinkSync(path);
    }
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} streaming token source (disk vs memory, u16 + u32)`);
  }

  {
    // f16 <-> f32 conversion (host<->GPU transfer for mixed precision).
    let ok = true;
    const exact: [number, number][] = [[1, 0x3c00], [0.5, 0x3800], [2, 0x4000], [-1, 0xbc00], [
      0,
      0,
    ]];
    for (const [v, bits] of exact) if (f32ToF16Bits(v) !== bits) ok = false;
    if (f16BitsToF32(0x3c00) !== 1) ok = false;
    if (f32ToF16Bits(70000) !== 0x7c00) ok = false; // overflow -> +inf
    if (f16BitsToF32(f32ToF16Bits(65504)) !== 65504) ok = false; // max normal, exact
    if (!Number.isNaN(f16BitsToF32(f32ToF16Bits(NaN)))) ok = false;
    const rng = mulberry32(11);
    for (let i = 0; i < 3000; i++) {
      const v = (rng() * 2 - 1) * 10;
      const back = f16BitsToF32(f32ToF16Bits(v));
      if (Math.abs(back - v) > Math.abs(v) / 1024 + 1e-3) ok = false; // within f16 precision
      if (f16BitsToF32(f32ToF16Bits(-v)) !== -back) ok = false; // sign-symmetric RNE
    }
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} f16 <-> f32 conversion (exact + round-trip + inf/nan)`);
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

main().catch((e) => {
  console.error("GRADCHECK FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});

// Tiny reverse-mode autograd over dense Float32 tensors.
//
// Define-by-run: every op allocates a result Tensor and records how to push
// gradients back to its inputs. backward() runs a topological sort from a
// scalar loss and calls each recorded closure once.
//
// This is the CPU reference backend. It implements exactly the ops a
// Qwen3ForCausalLM forward pass needs — no more. A WebGPU backend (see
// ../backend/webgpu.ts) is meant to implement the same op set with GPU
// kernels behind the same Tensor interface.

function prod(shape: number[]): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

export class Tensor {
  data: Float32Array;
  grad: Float32Array;
  shape: number[];
  requiresGrad: boolean;
  _backward: () => void = () => {};
  _prev: Tensor[] = [];

  constructor(data: Float32Array, shape: number[], requiresGrad = false) {
    if (data.length !== prod(shape)) {
      throw new Error(`data length ${data.length} != shape ${shape}`);
    }
    this.data = data;
    this.shape = shape;
    this.requiresGrad = requiresGrad;
    this.grad = new Float32Array(data.length);
  }

  static zeros(shape: number[], requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(prod(shape)), shape, requiresGrad);
  }

  zeroGrad() {
    this.grad.fill(0);
  }

  get size(): number {
    return this.data.length;
  }
}

/** Xavier/He-ish normal init leaf parameter. */
export function param(shape: number[], std: number, rng: () => number): Tensor {
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * std;
  return t;
}

// Box-Muller normal from a uniform rng.
export function randn(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Deterministic small PRNG (mulberry32) for reproducible inits. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Backprop from a scalar loss. `seed` sets dLoss/dLoss (use 1/batch to average). */
export function backward(loss: Tensor, seed = 1) {
  const topo: Tensor[] = [];
  const seen = new Set<Tensor>();
  const build = (t: Tensor) => {
    if (seen.has(t)) return;
    seen.add(t);
    for (const p of t._prev) build(p);
    topo.push(t);
  };
  build(loss);
  loss.grad[0] = seed;
  for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
}

// ---------------------------------------------------------------------------
// Backend dispatch. A GPU backend (../backend/webgpu.ts) can take over the op
// set by registering itself here; the model and trainer keep calling the same
// functions. The backend must implement ALL ops: mixing backends inside one
// graph would make CPU ops read device-resident data that hasn't synced back.
// With no backend registered (the default), the reference CPU code below runs.
// ---------------------------------------------------------------------------

export interface OpsBackend {
  linear(x: Tensor, w: Tensor): Tensor;
  add(a: Tensor, b: Tensor): Tensor;
  mul(a: Tensor, b: Tensor): Tensor;
  silu(x: Tensor): Tensor;
  gelu(x: Tensor): Tensor;
  scale(x: Tensor, c: number): Tensor;
  rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor;
  rmsNormHeads(x: Tensor, weight: Tensor, T: number, H: number, hd: number, eps: number): Tensor;
  embedding(weight: Tensor, ids: number[]): Tensor;
  rope(x: Tensor, T: number, H: number, hd: number, base: number, posOffset: number): Tensor;
  attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    T: number,
    Hq: number,
    Hkv: number,
    hd: number,
    window: number,
  ): Tensor;
  crossEntropy(logits: Tensor, targets: number[]): Tensor;
}

let opsBackend: OpsBackend | null = null;

export function setOpsBackend(b: OpsBackend | null) {
  opsBackend = b;
}

// ---------------------------------------------------------------------------
// Ops. Row-major throughout. "2D" tensors are [rows, cols].
// ---------------------------------------------------------------------------

/** y = x · Wᵀ, where x:[T,in], W:[out,in] -> y:[T,out]. (Linear, no bias.) */
export function linear(x: Tensor, w: Tensor): Tensor {
  if (opsBackend) return opsBackend.linear(x, w);
  const [T, inDim] = x.shape;
  const [outDim, inDim2] = w.shape;
  if (inDim !== inDim2) throw new Error(`linear dim mismatch ${inDim} vs ${inDim2}`);
  const out = Tensor.zeros([T, outDim]);
  for (let t = 0; t < T; t++) {
    for (let o = 0; o < outDim; o++) {
      let acc = 0;
      const xb = t * inDim;
      const wb = o * inDim;
      for (let i = 0; i < inDim; i++) acc += x.data[xb + i] * w.data[wb + i];
      out.data[t * outDim + o] = acc;
    }
  }
  out._prev = [x, w];
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < outDim; o++) {
        const g = out.grad[t * outDim + o];
        if (g === 0) continue;
        const xb = t * inDim;
        const wb = o * inDim;
        for (let i = 0; i < inDim; i++) {
          x.grad[xb + i] += g * w.data[wb + i];
          w.grad[wb + i] += g * x.data[xb + i];
        }
      }
    }
  };
  return out;
}

/** Elementwise add, same shape. Used for residuals. */
export function add(a: Tensor, b: Tensor): Tensor {
  if (opsBackend) return opsBackend.add(a, b);
  const out = Tensor.zeros(a.shape);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  out._prev = [a, b];
  out._backward = () => {
    for (let i = 0; i < out.grad.length; i++) {
      a.grad[i] += out.grad[i];
      b.grad[i] += out.grad[i];
    }
  };
  return out;
}

/** Elementwise multiply, same shape. Used for SwiGLU gate*up. */
export function mul(a: Tensor, b: Tensor): Tensor {
  if (opsBackend) return opsBackend.mul(a, b);
  const out = Tensor.zeros(a.shape);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * b.data[i];
  out._prev = [a, b];
  out._backward = () => {
    for (let i = 0; i < out.grad.length; i++) {
      a.grad[i] += out.grad[i] * b.data[i];
      b.grad[i] += out.grad[i] * a.data[i];
    }
  };
  return out;
}

/** SiLU / swish: x * sigmoid(x). */
export function silu(x: Tensor): Tensor {
  if (opsBackend) return opsBackend.silu(x);
  const out = Tensor.zeros(x.shape);
  const sig = new Float32Array(x.data.length);
  for (let i = 0; i < x.data.length; i++) {
    const s = 1 / (1 + Math.exp(-x.data[i]));
    sig[i] = s;
    out.data[i] = x.data[i] * s;
  }
  out._prev = [x];
  out._backward = () => {
    for (let i = 0; i < x.data.length; i++) {
      const s = sig[i];
      x.grad[i] += out.grad[i] * (s + x.data[i] * s * (1 - s));
    }
  };
  return out;
}

// GELU (tanh approximation), matching ggml's GGML_UNARY_OP_GELU used by
// llama.cpp's GeGLU FFN: g(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
const GELU_K = 0.7978845608028654; // √(2/π)
const GELU_A = 0.044715;

/** GELU (tanh approx). Gemma3's FFN is gelu(gate)·up (GeGLU). */
export function gelu(x: Tensor): Tensor {
  if (opsBackend) return opsBackend.gelu(x);
  const out = Tensor.zeros(x.shape);
  for (let i = 0; i < x.data.length; i++) {
    const v = x.data[i];
    out.data[i] = 0.5 * v * (1 + Math.tanh(GELU_K * (v + GELU_A * v * v * v)));
  }
  out._prev = [x];
  out._backward = () => {
    for (let i = 0; i < x.data.length; i++) {
      const v = x.data[i];
      const v2 = v * v;
      const u = GELU_K * (v + GELU_A * v * v2);
      const th = Math.tanh(u);
      const dudx = GELU_K * (1 + 3 * GELU_A * v2);
      const g = 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * dudx;
      x.grad[i] += out.grad[i] * g;
    }
  };
  return out;
}

/** Multiply by a compile-time constant (e.g. Gemma3's √(hidden) embedding scale). */
export function scale(x: Tensor, c: number): Tensor {
  if (opsBackend) return opsBackend.scale(x, c);
  const out = Tensor.zeros(x.shape);
  for (let i = 0; i < x.data.length; i++) out.data[i] = x.data[i] * c;
  out._prev = [x];
  out._backward = () => {
    for (let i = 0; i < x.data.length; i++) x.grad[i] += out.grad[i] * c;
  };
  return out;
}

/** RMSNorm over the last dim of x:[T,d] with learned weight:[d]. */
export function rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor {
  if (opsBackend) return opsBackend.rmsNorm(x, weight, eps);
  const [T, d] = x.shape;
  const out = Tensor.zeros([T, d]);
  const rInv = new Float32Array(T); // 1/sqrt(ms+eps) per row
  for (let t = 0; t < T; t++) {
    let ms = 0;
    const b = t * d;
    for (let j = 0; j < d; j++) ms += x.data[b + j] * x.data[b + j];
    ms /= d;
    const r = 1 / Math.sqrt(ms + eps);
    rInv[t] = r;
    for (let j = 0; j < d; j++) out.data[b + j] = x.data[b + j] * r * weight.data[j];
  }
  out._prev = [x, weight];
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const b = t * d;
      const r = rInv[t];
      let S = 0; // sum_j g_j * w_j * x_j
      for (let j = 0; j < d; j++) S += out.grad[b + j] * weight.data[j] * x.data[b + j];
      for (let j = 0; j < d; j++) {
        const g = out.grad[b + j];
        x.grad[b + j] += weight.data[j] * r * g - (x.data[b + j] / d) * r * r * r * S;
        weight.grad[j] += g * x.data[b + j] * r;
      }
    }
  };
  return out;
}

/** Per-head RMSNorm (Qwen3 QK-norm): x:[T, H*hd], weight:[hd]. */
export function rmsNormHeads(
  x: Tensor,
  weight: Tensor,
  T: number,
  H: number,
  hd: number,
  eps: number,
): Tensor {
  if (opsBackend) return opsBackend.rmsNormHeads(x, weight, T, H, hd, eps);
  const out = Tensor.zeros([T, H * hd]);
  const rInv = new Float32Array(T * H);
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < H; h++) {
      const b = t * H * hd + h * hd;
      let ms = 0;
      for (let j = 0; j < hd; j++) ms += x.data[b + j] * x.data[b + j];
      ms /= hd;
      const r = 1 / Math.sqrt(ms + eps);
      rInv[t * H + h] = r;
      for (let j = 0; j < hd; j++) out.data[b + j] = x.data[b + j] * r * weight.data[j];
    }
  }
  out._prev = [x, weight];
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const b = t * H * hd + h * hd;
        const r = rInv[t * H + h];
        let S = 0;
        for (let j = 0; j < hd; j++) S += out.grad[b + j] * weight.data[j] * x.data[b + j];
        for (let j = 0; j < hd; j++) {
          const g = out.grad[b + j];
          x.grad[b + j] += weight.data[j] * r * g - (x.data[b + j] / hd) * r * r * r * S;
          weight.grad[j] += g * x.data[b + j] * r;
        }
      }
    }
  };
  return out;
}

/** Embedding lookup: weight:[V,d], ids:number[T] -> [T,d]. */
export function embedding(weight: Tensor, ids: number[]): Tensor {
  if (opsBackend) return opsBackend.embedding(weight, ids);
  const [, d] = weight.shape;
  const T = ids.length;
  const out = Tensor.zeros([T, d]);
  for (let t = 0; t < T; t++) {
    const src = ids[t] * d;
    const dst = t * d;
    for (let j = 0; j < d; j++) out.data[dst + j] = weight.data[src + j];
  }
  out._prev = [weight];
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const src = ids[t] * d;
      const dst = t * d;
      for (let j = 0; j < d; j++) weight.grad[src + j] += out.grad[dst + j];
    }
  };
  return out;
}

/**
 * NEOX-style RoPE (as used by qwen) applied to x:[T, H*hd].
 * Pairs dim j with j+hd/2. positions are 0..T-1 plus posOffset.
 */
export function rope(
  x: Tensor,
  T: number,
  H: number,
  hd: number,
  base: number,
  posOffset = 0,
): Tensor {
  if (opsBackend) return opsBackend.rope(x, T, H, hd, base, posOffset);
  const half = hd / 2;
  const out = Tensor.zeros([T, H * hd]);
  // Precompute cos/sin per (t, j).
  const cos = new Float32Array(T * half);
  const sin = new Float32Array(T * half);
  for (let t = 0; t < T; t++) {
    const pos = t + posOffset;
    for (let j = 0; j < half; j++) {
      const freq = Math.pow(base, (-2 * j) / hd);
      const ang = pos * freq;
      cos[t * half + j] = Math.cos(ang);
      sin[t * half + j] = Math.sin(ang);
    }
  }
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < H; h++) {
      const b = t * H * hd + h * hd;
      for (let j = 0; j < half; j++) {
        const c = cos[t * half + j];
        const s = sin[t * half + j];
        const x0 = x.data[b + j];
        const x1 = x.data[b + j + half];
        out.data[b + j] = x0 * c - x1 * s;
        out.data[b + j + half] = x0 * s + x1 * c;
      }
    }
  }
  out._prev = [x];
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) {
        const b = t * H * hd + h * hd;
        for (let j = 0; j < half; j++) {
          const c = cos[t * half + j];
          const s = sin[t * half + j];
          const g0 = out.grad[b + j];
          const g1 = out.grad[b + j + half];
          x.grad[b + j] += c * g0 + s * g1;
          x.grad[b + j + half] += -s * g0 + c * g1;
        }
      }
    }
  };
  return out;
}

/**
 * Fused causal multi-head attention with grouped-query (GQA).
 * q:[T,Hq*hd], k:[T,Hkv*hd], v:[T,Hkv*hd] -> [T,Hq*hd].
 * `window` > 0 restricts each query t to keys [t-window+1, t] (sliding window,
 * as in Gemma3's SWA layers); 0 = full causal.
 */
export function attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  T: number,
  Hq: number,
  Hkv: number,
  hd: number,
  window = 0,
): Tensor {
  if (opsBackend) return opsBackend.attention(q, k, v, T, Hq, Hkv, hd, window);
  const group = Hq / Hkv;
  const scale = 1 / Math.sqrt(hd);
  const winStart = (t: number) => (window > 0 && t + 1 > window ? t + 1 - window : 0);
  const out = Tensor.zeros([T, Hq * hd]);
  // probs[h][t] = Float32Array of length (t+1); entries below winStart(t) stay 0.
  const probs: Float32Array[][] = [];
  const qStride = Hq * hd;
  const kvStride = Hkv * hd;

  for (let h = 0; h < Hq; h++) {
    const kv = Math.floor(h / group);
    probs[h] = [];
    for (let t = 0; t < T; t++) {
      const s0 = winStart(t);
      const scores = new Float32Array(t + 1);
      let maxS = -Infinity;
      const qb = t * qStride + h * hd;
      for (let s = s0; s <= t; s++) {
        const kb = s * kvStride + kv * hd;
        let dot = 0;
        for (let d = 0; d < hd; d++) dot += q.data[qb + d] * k.data[kb + d];
        dot *= scale;
        scores[s] = dot;
        if (dot > maxS) maxS = dot;
      }
      let sum = 0;
      for (let s = s0; s <= t; s++) {
        const e = Math.exp(scores[s] - maxS);
        scores[s] = e;
        sum += e;
      }
      for (let s = s0; s <= t; s++) scores[s] /= sum;
      probs[h][t] = scores;

      const ob = t * qStride + h * hd;
      for (let s = s0; s <= t; s++) {
        const p = scores[s];
        const vb = s * kvStride + kv * hd;
        for (let d = 0; d < hd; d++) out.data[ob + d] += p * v.data[vb + d];
      }
    }
  }

  out._prev = [q, k, v];
  out._backward = () => {
    for (let h = 0; h < Hq; h++) {
      const kv = Math.floor(h / group);
      for (let t = 0; t < T; t++) {
        const s0 = winStart(t);
        const p = probs[h][t];
        const ob = t * qStride + h * hd;
        const qb = t * qStride + h * hd;
        // dP[s] = sum_d dOut[t,h,d]*V[s,kv,d]; also accumulate dV.
        const dP = new Float32Array(t + 1);
        for (let s = s0; s <= t; s++) {
          const vb = s * kvStride + kv * hd;
          let acc = 0;
          for (let d = 0; d < hd; d++) {
            acc += out.grad[ob + d] * v.data[vb + d];
            v.grad[vb + d] += p[s] * out.grad[ob + d];
          }
          dP[s] = acc;
        }
        // softmax backward -> dscore, then scale.
        let dot = 0;
        for (let s = s0; s <= t; s++) dot += p[s] * dP[s];
        for (let s = s0; s <= t; s++) {
          const dscore = p[s] * (dP[s] - dot) * scale;
          const kb = s * kvStride + kv * hd;
          for (let d = 0; d < hd; d++) {
            q.grad[qb + d] += dscore * k.data[kb + d];
            k.grad[kb + d] += dscore * q.data[qb + d];
          }
        }
      }
    }
  };
  return out;
}

/** Softmax cross-entropy over logits:[T,V] vs integer targets:[T]. Returns scalar. */
export function crossEntropy(logits: Tensor, targets: number[]): Tensor {
  if (opsBackend) return opsBackend.crossEntropy(logits, targets);
  const [T, V] = logits.shape;
  const loss = Tensor.zeros([1]);
  const probs = new Float32Array(T * V);
  // A target < 0 marks an ignored position (e.g. prompt tokens under
  // assistant-only loss masking): it contributes no loss and no gradient, and
  // the mean is over kept rows only. With no ignored rows this is the plain
  // full-sequence mean (kept === T), so existing callers are unchanged.
  let total = 0;
  let kept = 0;
  for (let t = 0; t < T; t++) {
    const b = t * V;
    let maxL = -Infinity;
    for (let v = 0; v < V; v++) if (logits.data[b + v] > maxL) maxL = logits.data[b + v];
    let sum = 0;
    for (let v = 0; v < V; v++) {
      const e = Math.exp(logits.data[b + v] - maxL);
      probs[b + v] = e;
      sum += e;
    }
    for (let v = 0; v < V; v++) probs[b + v] /= sum;
    if (targets[t] >= 0) {
      total += -Math.log(probs[b + targets[t]] + 1e-12);
      kept++;
    }
  }
  const denom = kept > 0 ? kept : 1;
  loss.data[0] = total / denom;
  loss._prev = [logits];
  loss._backward = () => {
    const scale = loss.grad[0] / denom;
    for (let t = 0; t < T; t++) {
      if (targets[t] < 0) continue; // ignored position: no gradient
      const b = t * V;
      for (let v = 0; v < V; v++) {
        logits.grad[b + v] += scale * (probs[b + v] - (v === targets[t] ? 1 : 0));
      }
    }
  };
  return loss;
}

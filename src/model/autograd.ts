// Tiny reverse-mode autograd over dense Float32 tensors.
//
// Define-by-run: every op allocates a result Tensor and records how to push
// gradients back to its inputs. backward() runs a topological sort from a
// scalar loss and calls each recorded closure once.
//
// This is the CPU reference backend. It implements exactly the ops a
// Gemma3ForCausalLM forward pass needs, no more. A WebGPU backend (see
// ../backend/webgpu.ts) is meant to implement the same op set with GPU
// kernels behind the same Tensor interface.

function prod(shape: number[]): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

let tensorsCreated = 0;

export class Tensor {
  data: Float32Array;
  grad: Float32Array;
  shape: number[];
  requiresGrad: boolean;
  _backward: () => void = () => {};
  _prev: Tensor[] = [];
  /** Creation order, so `checkpoint` can tell its own nodes from older ones. */
  readonly seq: number = tensorsCreated++;

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
  /** The contracted dimension is validated by the `linear` wrapper, above this
   * dispatch. */
  linear(x: Tensor, w: Tensor): Tensor;
  add(a: Tensor, b: Tensor): Tensor;
  mul(a: Tensor, b: Tensor): Tensor;
  silu(x: Tensor): Tensor;
  gelu(x: Tensor): Tensor;
  scale(x: Tensor, c: number): Tensor;
  rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor;
  rmsNormHeads(x: Tensor, weight: Tensor, T: number, H: number, hd: number, eps: number): Tensor;
  /** Ids are validated by the `embedding` wrapper, above this dispatch. */
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
  /**
   * Targets are validated by the `crossEntropy` wrapper, above this dispatch,
   * which passes the kept-row count it counted on the way through. Taking
   * `kept` rather than recounting is what let the validation move up: it is the
   * loss denominator, so both implementations need the number, and a guard that
   * lives under a dispatch has to be repeated in each one.
   */
  crossEntropy(logits: Tensor, targets: number[], kept: number): Tensor;
  /** Same as `crossEntropy`: validated above the dispatch, `kept` passed down. */
  fusedCrossEntropy(
    hidden: Tensor,
    w: Tensor,
    targets: number[],
    chunk: number,
    kept: number,
  ): Tensor;
  softCrossEntropy(
    logits: Tensor,
    teacherIds: number[],
    teacherProbs: number[],
    k: number,
  ): Tensor;
}

/**
 * Optional backend capability behind `checkpoint()`. Kept off `OpsBackend`,
 * which is the op set and must be implemented in full: the CPU reference needs
 * none of this, because dropping the tape is enough for the garbage collector
 * to reclaim an intermediate. On a device backend it is not, since the buffers
 * are held by the backend rather than by the tensors.
 */
export interface RegionBackend {
  beginRegion(): number;
  endRegion(mark: number, keep: Tensor[]): void;
  seedGradFrom(dst: Tensor, src: Tensor): void;
}

function regionBackend(): RegionBackend | null {
  const b = opsBackend as unknown as Partial<RegionBackend> | null;
  return b && typeof b.beginRegion === "function" ? (b as RegionBackend) : null;
}

let checkpointing = false;

/**
 * Turn `checkpoint()` from a passthrough into a recompute boundary. Off by
 * default, so an architecture can call `checkpoint` unconditionally and a run
 * that has not asked for it builds exactly the graph it built before.
 */
export function setCheckpointing(on: boolean) {
  checkpointing = on;
}

/**
 * Every non-leaf the block reads must have been built by this call to `fn`.
 * One that predates it is shared with something outside the block, and the
 * local walk in `checkpoint`'s backward would run its `_backward` once per
 * block that reads it, each time propagating a gradient that has already grown:
 *
 *   const shared = linear(x, w);                      // built once, outside
 *   const a = checkpoint([h0], () => add(h0, shared));
 *   const b = checkpoint([a], () => add(a, shared));  // x.grad ends up 2a + b
 *
 * `shared` is not in the outer graph either, since the rewiring cut it, so
 * nothing downstream corrects the double count. The numbers stay finite and the
 * loss curve looks ordinary, which is why this is a throw rather than a note in
 * the docs. Leaves are exempt: a parameter's `_backward` is a no-op, and its
 * consumers accumulate independently.
 */
function assertSelfContained(out: Tensor, inputs: Tensor[], born: number) {
  const seen = new Set<Tensor>(inputs);
  if (seen.has(out)) {
    throw new Error("checkpoint: fn returned one of its inputs; there is nothing to recompute");
  }
  const walk = (t: Tensor) => {
    if (seen.has(t)) return;
    seen.add(t);
    if (t._prev.length > 0 && t.seq < born) {
      throw new Error(
        "checkpoint: the block reads a computed tensor it did not create. Pass it in `inputs`, " +
          "or move its computation inside the block (docs/adding-an-architecture.md).",
      );
    }
    for (const p of t._prev) walk(p);
  };
  walk(out);
}

/**
 * Run `fn` without keeping its interior, and recompute it in backward.
 *
 * Activation memory is the largest term in a training step, and almost all of
 * it is intermediates that exist only to be read once by their own backward
 * closure. This trades that for arithmetic: forward keeps the block's output
 * and throws the rest away, backward replays `fn` to rebuild what it needs.
 *
 * `inputs` are the tensors gradients must flow back to. They are excluded from
 * the replayed subgraph's traversal, so their own backward closures run once,
 * in the outer graph, rather than once per checkpoint.
 *
 * `fn` must be a pure replay: same inputs, same graph, same values. It is
 * called exactly twice per step, and a hidden dependency on call order (a
 * captured RNG, a mutated buffer) would make the second call disagree with the
 * first and corrupt the gradient silently.
 */
export function checkpoint(inputs: Tensor[], fn: () => Tensor): Tensor {
  if (!checkpointing) return fn();
  const rb = regionBackend();

  const born = tensorsCreated;
  const mark = rb ? rb.beginRegion() : 0;
  const out = fn();
  assertSelfContained(out, inputs, born);
  // Keeping `out` alive is the whole point: everything else the block allocated
  // becomes reusable, and the next block's forward draws from it.
  if (rb) rb.endRegion(mark, [out]);

  // Rewiring the tensor in place, rather than wrapping it, is what drops the
  // interior: nothing else references those nodes, so they are collectable.
  out._prev = inputs;
  out._backward = () => {
    const mark2 = rb ? rb.beginRegion() : 0;
    const replay = fn();
    if (rb) rb.seedGradFrom(replay, out);
    else replay.grad.set(out.grad);

    // Seeding `seen` with the inputs stops the walk at the block boundary, so
    // this accumulates INTO their gradients without descending past them.
    const topo: Tensor[] = [];
    const seen = new Set<Tensor>(inputs);
    const build = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._prev) build(p);
      topo.push(t);
    };
    build(replay);
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();

    // The replay's interior has been read; release it before the next block
    // recomputes, or backward peaks at every block's replay at once.
    if (rb) rb.endRegion(mark2, []);
  };
  return out;
}

let opsBackend: OpsBackend | null = null;

export function setOpsBackend(b: OpsBackend | null) {
  opsBackend = b;
}

// ---------------------------------------------------------------------------
// Ops. Row-major throughout. "2D" tensors are [rows, cols].
// ---------------------------------------------------------------------------

/** y = x · Wᵀ, where x:[T,in], W:[out,in] -> y:[T,out]. (Linear, no bias.) */
/**
 * A LoRA adapter for one frozen weight: `W + (alpha/rank) * B*A`, with
 * `A: [rank, in]` and `B: [out, rank]`.
 */
export interface LoraAdapter {
  a: Tensor;
  b: Tensor;
  scale: number;
}

let loraAdapters: Map<Tensor, LoraAdapter> = new Map();

/**
 * Route `linear` through low-rank adapters for the listed weights.
 *
 * Registering here rather than in the architectures is what keeps adapters
 * arch-agnostic: `linear` is the one call every projection already goes
 * through, so nothing in `src/arch/` changes and a new architecture gets LoRA
 * without knowing it exists. Pass an empty map to go back to full fine-tuning.
 */
export function setLoraAdapters(m: Map<Tensor, LoraAdapter>) {
  loraAdapters = m;
}

/** y = x · Wᵀ, plus the low-rank update when `w` carries an adapter. */
export function linear(x: Tensor, w: Tensor): Tensor {
  if (loraAdapters.size > 0) {
    const ad = loraAdapters.get(w);
    // B·A·x rather than (B·A)·x: the point of the factorization is never
    // materializing the [out, in] product.
    if (ad) return add(linearRaw(x, w), scale(linearRaw(linearRaw(x, ad.a), ad.b), ad.scale));
  }
  return linearRaw(x, w);
}

function linearRaw(x: Tensor, w: Tensor): Tensor {
  const [T, inDim] = x.shape;
  const [outDim, inDim2] = w.shape;
  // Above the dispatch, like every other shape validator in this file. It was
  // the last one below (#91), and it survived there because both
  // implementations happened to have it, with the identical message. That is
  // exactly the shape #61 was: a guard under a dispatch has to be written again
  // in every implementation, and the one nobody remembers is the one that ships
  // unvalidated.
  //
  // Cheaper to hoist than the loss guards of #82: it returns nothing, so the
  // dispatch line is unchanged, and the comparison itself does not run any more
  // often than before, since webgpu.ts ran it on every call too. What the GPU
  // path does pay is the two shape destructures above, which used to happen
  // only inside the backend. `linear` is the hottest op in the graph, so that
  // is worth a number rather than a shrug: 86 calls per step at 6 layers and
  // batch 2, 170 with --recompute since 42 of the 43 per micro-batch replay in
  // backward, and 4.1 ns for two destructures and a compare, which is 0.0007 ms
  // per step at the higher count against a step measured in seconds.
  if (inDim !== inDim2) {
    throw new Error(
      `linear dim mismatch: x is [${x.shape.join(", ")}] and w is [${w.shape.join(", ")}], ` +
        `so the contracted dimension is ${inDim} on one side and ${inDim2} on the other. ` +
        `A projection wired to the wrong config field is the usual cause.`,
    );
  }
  if (opsBackend) return opsBackend.linear(x, w);
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
  const wantsDW = w.requiresGrad; // a frozen LoRA base accumulates nothing
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < outDim; o++) {
        const g = out.grad[t * outDim + o];
        if (g === 0) continue;
        const xb = t * inDim;
        const wb = o * inDim;
        if (wantsDW) {
          for (let i = 0; i < inDim; i++) {
            x.grad[xb + i] += g * w.data[wb + i];
            w.grad[wb + i] += g * x.data[xb + i];
          }
        } else {
          for (let i = 0; i < inDim; i++) x.grad[xb + i] += g * w.data[wb + i];
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
  const wantsDW = weight.requiresGrad; // frozen (LoRA base): nothing accumulates
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const b = t * d;
      const r = rInv[t];
      let S = 0; // sum_j g_j * w_j * x_j
      for (let j = 0; j < d; j++) S += out.grad[b + j] * weight.data[j] * x.data[b + j];
      for (let j = 0; j < d; j++) {
        const g = out.grad[b + j];
        x.grad[b + j] += weight.data[j] * r * g - (x.data[b + j] / d) * r * r * r * S;
        if (wantsDW) weight.grad[j] += g * x.data[b + j] * r;
      }
    }
  };
  return out;
}

/** Per-head RMSNorm (Gemma3 QK-norm): x:[T, H*hd], weight:[hd]. */
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
  const wantsDW = weight.requiresGrad; // frozen (LoRA base): nothing accumulates
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
          if (wantsDW) weight.grad[j] += g * x.data[b + j] * r;
        }
      }
    }
  };
  return out;
}

/** Embedding lookup: weight:[V,d], ids:number[T] -> [T,d]. */
export function embedding(weight: Tensor, ids: number[]): Tensor {
  // Before the id check, which reads a vocab size out of this shape: a 1-D table
  // makes that V*d, and a 3-D one passes the right V while the row stride
  // reads the wrong rows in silence.
  assertMatrix(weight, "weight", "embedding");
  assertIdsInTable(ids, weight.shape[0], "embedding");
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
  const wantsDW = weight.requiresGrad; // frozen (LoRA base): nothing accumulates
  out._backward = () => {
    for (let t = 0; t < T; t++) {
      const src = ids[t] * d;
      const dst = t * d;
      if (wantsDW) { for (let j = 0; j < d; j++) weight.grad[src + j] += out.grad[dst + j]; }
    }
  };
  return out;
}

/**
 * NEOX-style RoPE (as used by gemma3) applied to x:[T, H*hd].
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

/**
 * Everything `softCrossEntropy` needs checked, in one place ABOVE the backend
 * dispatch.
 *
 * The two implementations carried the shape guards verbatim and only the CPU one
 * carried the id check, which is the omission the below-dispatch style invites
 * and was #61: `teacherIds` reached the kernel unvalidated, so an id built
 * against a different vocab indexed whatever the bound buffer held.
 *
 * A row whose first id is exactly `-1` is ignored, and its remaining ids are
 * never read, so they are not checked. `-1` and not any negative, for the reason
 * `keptRowsInVocab` gives: `uploadU32` maps `-1` to `0xffffffff`, the marker the
 * kernels test for, while `-2` becomes a huge id and `-0.5` becomes 0. Skipping
 * the row on `< 0` would leave exactly the inputs this exists to catch
 * unchecked, with the CPU dropping the row and the GPU scoring it.
 *
 * Inside a kept row all k ids must be in range, including the slots a row
 * shorter than k pads at probability 0. Only the FORWARDS skip a pad, both on
 * `q == 0`; both backwards index by its id unconditionally. On the CPU that is a
 * no-op, `x -= 0` writing back what it read. On the GPU it is a non-atomic
 * read-modify-write (`DLOG[i] = DLOG[i] - scale * TQ[...]` in srcSoftCeBwdQ), so
 * an out-of-range pad id lands in another row and can lose that row's real
 * update: exactly the race the one-thread-per-row design exists to prevent. The
 * contract that a pad carries an in-range id is a consequence of that, not the
 * reason for it. A teacher file must pad short rows with an in-range id, never
 * with `-1`.
 */
export function assertTeacherRows(
  teacherIds: number[],
  teacherProbs: number[],
  T: number,
  k: number,
  V: number,
): void {
  if (k < 1) throw new Error(`softCrossEntropy: k must be >= 1, got ${k}`);
  if (teacherIds.length !== T * k || teacherProbs.length !== T * k) {
    throw new Error(
      `softCrossEntropy: teacher arrays must be [T*k]=${T * k}, got ` +
        `${teacherIds.length}/${teacherProbs.length}`,
    );
  }
  for (let t = 0; t < T; t++) {
    if (teacherIds[t * k] === -1) continue; // the marker; nothing below it is read
    for (let j = 0; j < k; j++) {
      const id = teacherIds[t * k + j];
      if (!Number.isInteger(id) || id < 0 || id >= V) {
        throw new Error(
          `softCrossEntropy: teacher id ${id} at slot ${j} of row ${t} is not an ` +
            `integer in [0,${V}). Only -1 in the first slot marks an ignored row. ` +
            `A teacher file built against a different vocab than the checkpoint is ` +
            `the usual cause.`,
        );
      }
    }
  }
}

/**
 * Every embedding id must be a row of the table.
 *
 * `weight.data[id * d + j]` with `id >= V` reads into the next row, or past the
 * array on the last one. Measured at V=4, d=3 with an id of `V + 2`: the CPU
 * returns `[NaN, NaN, NaN]`, which poisons the whole forward, and the GPU
 * returns `[0, 0, 0]`, because the bound buffer discards the read. Neither
 * stops, and the GPU's substituted zero row is the worse of the two, since the
 * run continues on a number that looks fine.
 *
 * Unlike a loss target there is no ignore marker: every position of a batch is
 * a real token.
 *
 * Checked above the backend dispatch rather than in each backend, which is what
 * `fusedCrossEntropy` does with its dimension, chunk and LoRA guards, and which
 * makes the omission the below-dispatch style invites unreachable: validate
 * under the dispatch and every implementation needs its own call, which is how
 * `softCrossEntropy` ended up checking its teacher ids on the CPU and not on
 * the GPU (#61).
 */
export function assertIdsInTable(ids: number[], V: number, where: string): void {
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    if (!Number.isInteger(id) || id < 0 || id >= V) {
      throw new Error(
        `${where}: id ${id} at position ${t} is not an integer in [0,${V}). ` +
          `A corpus tokenized with a different vocab than the checkpoint is the usual cause.`,
      );
    }
  }
}

/**
 * A tensor has to be the rank its caller assumes, checked before the caller
 * destructures a shape it has not looked at.
 *
 * Two families of caller. The losses read `const [T, V] = logits.shape` and then
 * compare every id against `V`, so a 1-D tensor makes `V` undefined and their
 * range guards accept everything. The GGUF writer reads `const [outDim, inDim]`
 * and then decides a quant from `inDim`, so a 1-D tensor makes that decision on
 * NaN and writes an undefined into the file's ne.
 */
export function assertRank(t: Tensor, rank: number, name: string, where: string): void {
  if (t.shape.length !== rank) {
    throw new Error(`${where}: ${name} must be ${rank}-D, got [${t.shape.join(", ")}]`);
  }
}

/**
 * `assertRank` at 2. The losses all want a matrix, and they check above the
 * backend dispatch so neither implementation can be the one that skips it.
 */
export function assertMatrix(t: Tensor, name: string, where: string): void {
  assertRank(t, 2, name, where);
}

/**
 * Count the rows a loss will keep, refusing any target that is not an ignore
 * marker or a row of the vocab.
 *
 * `-1` is the ignore marker; anything else has to index a real logit row,
 * because `logits.data[t * V + target]` with `target >= V` reads the NEXT row's
 * logits. Measured at T=3, V=6 with one kept row: both backends returned
 * 2.038443088531494, that row's logsumexp minus the following row's first logit
 * to f32. The last row is the only one where the two differ, and the GPU is the
 * worse of the pair there: the CPU reads past its array and gives NaN, while the
 * GPU returns a finite, plausible number whose digits depend on pool state.
 *
 * Only `-1` is accepted as ignore, not every negative. `uploadU32` maps `-1` to
 * `0xffffffff`, the marker the kernels test for, but `-2` becomes `0xfffffffe`
 * (a huge target) and `-0.5` becomes `0`. The CPU would treat both as ignore
 * while the GPU scored them, so the host count and the kernel would disagree.
 *
 * `targets.length` must equal `T`: the losses sum over `T` rows and divide by
 * the count returned here, so a longer array inflates the denominator and
 * reports a plausible wrong loss.
 *
 * The usual way to get here is a corpus tokenized with a different vocab than
 * the checkpoint, which `agents.md` invariant 1 exists to prevent: the tokenizer
 * freezes at step one. Scoring a foreign base against the wrong `.tokens` file
 * reaches it too.
 */
export function keptRowsInVocab(targets: number[], T: number, V: number, where: string): number {
  if (targets.length !== T) {
    throw new Error(
      `${where}: ${targets.length} targets for ${T} logit rows. The loss sums T rows and means ` +
        `over the targets it counts, so a mismatch reports a plausible wrong loss.`,
    );
  }
  let kept = 0;
  for (let t = 0; t < T; t++) {
    const g = targets[t];
    if (!Number.isInteger(g) || g < -1 || g >= V) {
      throw new Error(
        `${where}: target ${g} at position ${t} is not -1 (ignore) or an integer in [0,${V}). ` +
          `A corpus tokenized with a different vocab than the checkpoint is the usual cause.`,
      );
    }
    if (g >= 0) kept++;
  }
  return kept;
}

/** Softmax cross-entropy over logits:[T,V] vs integer targets:[T]. Returns scalar. */
export function crossEntropy(logits: Tensor, targets: number[]): Tensor {
  assertMatrix(logits, "logits", "crossEntropy");
  const [T, V] = logits.shape;
  // Above the dispatch, like every id and shape validator in this file. It
  // stayed below for as long as it did because
  // it returns a value both implementations need as their loss denominator, so
  // hoisting it means passing `kept` down rather than each backend calling it
  // again. That is a smaller price than the shape that produced #61: a guard
  // under a dispatch has to be repeated in every implementation, and the one
  // nobody remembers is the one that ships unvalidated.
  const kept = keptRowsInVocab(targets, T, V, "crossEntropy");
  if (opsBackend) return opsBackend.crossEntropy(logits, targets, kept);
  const loss = Tensor.zeros([1]);
  const probs = new Float32Array(T * V);
  // A target < 0 marks an ignored position (e.g. prompt tokens under
  // assistant-only loss masking): it contributes no loss and no gradient, and
  // the mean is over kept rows only. With no ignored rows this is the plain
  // full-sequence mean (kept === T), so existing callers are unchanged.
  let total = 0;
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
      // log(Σ exp(z - m)) + m - z_target, not -log(p_target). The normalized
      // probability underflows f32 once the target is ~88 logits behind the
      // maximum, and the epsilon this used to add then clamped every worse
      // prediction to -log(1e-12) = 27.63. Measured before the change: a gap of
      // 30 reported 27.54 against an exact 30, and a gap of 90 still reported
      // 27.63. The GPU kernel and `fusedCrossEntropy` were already computing it
      // this way, so the CPU reference was the odd one out.
      total += Math.log(sum) + maxL - logits.data[b + targets[t]];
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

/**
 * Readout matmul fused into cross-entropy, streaming the vocab axis in chunks.
 *
 * Mathematically identical to `crossEntropy(linear(hidden, w), targets)`. The
 * difference is what stays resident: the dense pair holds three [T, vocab]
 * buffers (the readout's data and grad, plus the softmax scratch), and at a
 * large vocab a single one of those passes the WebGPU storage-buffer binding
 * limit long before memory runs out. Here the widest live buffer is
 * [T, chunk], and backward recomputes each chunk's logits from `hidden` and
 * `w` rather than reading them back.
 *
 * The cost is one extra readout matmul per step. `chunk` trades peak memory
 * against the number of passes; it does not change the result.
 */
export function fusedCrossEntropy(
  hidden: Tensor,
  w: Tensor,
  targets: number[],
  chunk: number,
): Tensor {
  // Validated ABOVE the backend dispatch, or these run on the CPU reference only
  // and every real run installs the GPU backend first.
  assertMatrix(hidden, "hidden", "fusedCrossEntropy");
  assertMatrix(w, "w", "fusedCrossEntropy");
  const [T, H] = hidden.shape;
  const [V, H2] = w.shape;
  if (H !== H2) throw new Error(`fusedCrossEntropy dim mismatch ${H} vs ${H2}`);
  if (chunk <= 0) throw new Error(`fusedCrossEntropy chunk must be positive, got ${chunk}`);
  // Adapters live inside `linear`, and this path deliberately does not go
  // through it. An adapted readout would therefore be adapted in the dense
  // forward and unadapted here, i.e. in training: a silent divergence between
  // what the trust gate checks and what the run optimizes. It holds today only
  // because all three architectures put the readout in the aux group, which is
  // a convention, not a guarantee.
  if (loraAdapters.has(w)) {
    throw new Error(
      "fusedCrossEntropy: the readout weight carries a LoRA adapter, which this " +
        "path cannot apply. Put the readout in the aux param group, or use --loss-chunk 0.",
    );
  }
  // Above the dispatch, and on the CPU side before the chunk loops rather than
  // after: an out-of-range target never falls inside any span, so `tgtLogit`
  // would stay 0 and the loss would be quietly wrong rather than NaN.
  const kept = keptRowsInVocab(targets, T, V, "fusedCrossEntropy");
  if (opsBackend) return opsBackend.fusedCrossEntropy(hidden, w, targets, chunk, kept);
  const loss = Tensor.zeros([1]);

  // Online softmax over the chunked vocab: a chunk whose maximum beats the
  // running one rescales the sum so far instead of forcing a second pass.
  const rowMax = new Float32Array(T).fill(-Infinity);
  const rowSum = new Float32Array(T);
  const tgtLogit = new Float32Array(T);
  const dot = (t: number, v: number) => {
    let acc = 0;
    for (let i = 0; i < H; i++) acc += hidden.data[t * H + i] * w.data[v * H + i];
    return acc;
  };
  for (let v0 = 0; v0 < V; v0 += chunk) {
    const v1 = Math.min(v0 + chunk, V);
    for (let t = 0; t < T; t++) {
      let chunkMax = -Infinity;
      for (let v = v0; v < v1; v++) {
        const z = dot(t, v);
        if (z > chunkMax) chunkMax = z;
      }
      const mNew = Math.max(rowMax[t], chunkMax);
      let sum = 0;
      for (let v = v0; v < v1; v++) sum += Math.exp(dot(t, v) - mNew);
      rowSum[t] = rowSum[t] * Math.exp(rowMax[t] - mNew) + sum;
      rowMax[t] = mNew;
      const g = targets[t];
      if (g >= v0 && g < v1) tgtLogit[t] = dot(t, g);
    }
  }

  let total = 0;
  for (let t = 0; t < T; t++) {
    if (targets[t] < 0) continue; // ignored position: no loss, no gradient
    total += Math.log(rowSum[t]) + rowMax[t] - tgtLogit[t];
  }
  const denom = kept > 0 ? kept : 1;
  loss.data[0] = total / denom;
  loss._prev = [hidden, w];
  loss._backward = () => {
    const scale = loss.grad[0] / denom;
    for (let v0 = 0; v0 < V; v0 += chunk) {
      const v1 = Math.min(v0 + chunk, V);
      for (let t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        for (let v = v0; v < v1; v++) {
          const p = Math.exp(dot(t, v) - rowMax[t]) / rowSum[t];
          const d = scale * (p - (v === targets[t] ? 1 : 0));
          for (let i = 0; i < H; i++) {
            hidden.grad[t * H + i] += d * w.data[v * H + i];
            if (w.requiresGrad) w.grad[v * H + i] += d * hidden.data[t * H + i];
          }
        }
      }
    }
  };
  return loss;
}

/**
 * Cross-entropy against a SPARSE soft target: the Phase B KL anchor, where the
 * teacher is the frozen base checkpoint and its top-k logits are precomputed
 * once over the SFT corpus.
 *
 *   loss = mean over kept rows of  -Σ_j q[t,j]·log p[t, ids[t,j]]
 *
 * `teacherIds`/`teacherProbs` are [T*k] row-major: row t holds the k teacher
 * token ids and their probabilities. A row is IGNORED (no loss, no gradient,
 * excluded from the mean) when its first id is exactly -1: the same convention
 * as crossEntropy's ignore-index, so assistant-only masking carries over. Rows
 * with fewer than k entries pad with any in-range id at probability 0.
 *
 * The teacher mass need not sum to 1: with top-k truncation it sums to S ≤ 1,
 * and the exact gradient is `S·p − q` (the (p − q) of the normalized case).
 * This differs from KL(q‖p) only by the teacher's entropy, a constant in the
 * student's parameters: same gradient, and the reported value is a plain
 * cross-entropy in nats, directly comparable to the hard-target loss.
 */
export function softCrossEntropy(
  logits: Tensor,
  teacherIds: number[],
  teacherProbs: number[],
  k: number,
): Tensor {
  assertMatrix(logits, "logits", "softCrossEntropy");
  const [T, V] = logits.shape;
  assertTeacherRows(teacherIds, teacherProbs, T, k, V);
  if (opsBackend) return opsBackend.softCrossEntropy(logits, teacherIds, teacherProbs, k);
  const loss = Tensor.zeros([1]);
  const probs = new Float32Array(T * V);
  const rowMass = new Float32Array(T); // S per row: Σ_j q[t,j]
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
    if (teacherIds[t * k] < 0) continue; // ignored row
    for (let j = 0; j < k; j++) {
      const id = teacherIds[t * k + j];
      const q = teacherProbs[t * k + j];
      // A row shorter than k pads with an in-range id at probability 0 (see the
      // docstring). The one input this saves is a -Infinity logit sitting at a
      // pad slot, where the expansion below reaches `0 * Infinity`; a NaN or
      // +Infinity logit already poisons `sum` upstream, guard or no guard. The
      // GPU kernel skips the same way, so the two cannot diverge here.
      if (q === 0) continue;
      // Σ q·(log(Σ exp(z-m)) + m - z_id), the same expansion the GPU kernel
      // uses and for the same reason: reading a normalized probability back
      // clamps every confident-wrong teacher term at -log(1e-12) = 27.63.
      total += q * (Math.log(sum) + maxL - logits.data[b + id]);
      rowMass[t] += q;
    }
    kept++;
  }
  const denom = kept > 0 ? kept : 1;
  loss.data[0] = total / denom;
  loss._prev = [logits];
  loss._backward = () => {
    const scale = loss.grad[0] / denom;
    for (let t = 0; t < T; t++) {
      if (teacherIds[t * k] < 0) continue;
      const b = t * V;
      for (let v = 0; v < V; v++) logits.grad[b + v] += scale * rowMass[t] * probs[b + v];
      for (let j = 0; j < k; j++) {
        logits.grad[b + teacherIds[t * k + j]] -= scale * teacherProbs[t * k + j];
      }
    }
  };
  return loss;
}

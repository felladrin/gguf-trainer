// Muon optimizer (Jordan et al., 2024): MomentUm Orthogonalized by Newton-schulz.
//
// For 2D hidden weight matrices, Muon takes the momentum buffer and replaces it
// with the nearest semi-orthogonal matrix via a quintic Newton–Schulz iteration
// before applying the update. This conditions every update and, empirically,
// trains transformers meaningfully faster per step than AdamW.
//
// Muon is applied ONLY to 2D hidden matmul weights. Embeddings, the output head,
// and all 1-D params (norms) use AdamW — this split follows the reference recipe
// (embeddings/heads have very different geometry and misbehave under Muon).

import type { Tensor } from "../model/autograd.ts";
import { AdamW } from "./adam.ts";
import type { AdamOpts } from "./adam.ts";
import type { Optimizer } from "./optimizer.ts";

export interface MuonOpts {
  lr: number;
  momentum?: number; // default 0.95
  nesterov?: boolean; // default true
  nsSteps?: number; // Newton-Schulz iterations, default 5
  aux: AdamOpts; // AdamW settings for the non-Muon param group
}

// Quintic Newton–Schulz coefficients from the reference implementation.
const NS_A = 3.4445;
const NS_B = -4.7750;
const NS_C = 2.0315;

export class Muon implements Optimizer {
  private muonParams: Tensor[];
  private bufs: Float32Array[];
  private aux: AdamW;
  private opts: Required<Omit<MuonOpts, "aux">>;

  /**
   * @param muonParams 2-D hidden weight matrices ([out, in]).
   * @param auxParams  everything else (embeddings, output head, norms).
   */
  constructor(
    muonParams: Tensor[],
    auxParams: Tensor[],
    opts: MuonOpts,
  ) {
    this.muonParams = muonParams;
    this.opts = {
      lr: opts.lr,
      momentum: opts.momentum ?? 0.95,
      nesterov: opts.nesterov ?? true,
      nsSteps: opts.nsSteps ?? 5,
    };
    this.bufs = muonParams.map((p) => new Float32Array(p.size));
    this.aux = new AdamW(auxParams, opts.aux);
  }

  zeroGrad() {
    for (const p of this.muonParams) p.grad.fill(0);
    this.aux.zeroGrad();
  }

  step() {
    const o = this.opts;
    for (let i = 0; i < this.muonParams.length; i++) {
      const p = this.muonParams[i];
      if (p.shape.length !== 2) throw new Error("Muon param must be 2-D");
      const [rows, cols] = p.shape;
      const buf = this.bufs[i];

      // Momentum (with optional Nesterov look-ahead).
      let update: Float32Array;
      for (let k = 0; k < buf.length; k++) buf[k] = o.momentum * buf[k] + p.grad[k];
      if (o.nesterov) {
        update = new Float32Array(buf.length);
        for (let k = 0; k < buf.length; k++) update[k] = p.grad[k] + o.momentum * buf[k];
      } else {
        update = buf.slice();
      }

      const ortho = newtonSchulz(update, rows, cols, o.nsSteps);

      // Scale (reference): sqrt(max(1, rows/cols)).
      const scale = Math.sqrt(Math.max(1, rows / cols));
      for (let k = 0; k < p.size; k++) p.data[k] -= o.lr * ortho[k] * scale;
    }
    this.aux.step();
  }
}

// --- dense matrix helpers (row-major) ---

function transpose(A: Float32Array, r: number, c: number): Float32Array {
  const out = new Float32Array(r * c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j * r + i] = A[i * c + j];
  return out;
}

// C[ra,cb] = A[ra,ca] · B[ca,cb]
function matmul(
  A: Float32Array,
  ra: number,
  ca: number,
  B: Float32Array,
  cb: number,
): Float32Array {
  const C = new Float32Array(ra * cb);
  for (let i = 0; i < ra; i++) {
    for (let k = 0; k < ca; k++) {
      const a = A[i * ca + k];
      if (a === 0) continue;
      const bRow = k * cb;
      const cRow = i * cb;
      for (let j = 0; j < cb; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
  return C;
}

function frob(A: Float32Array): number {
  let s = 0;
  for (let i = 0; i < A.length; i++) s += A[i] * A[i];
  return Math.sqrt(s);
}

/**
 * Orthogonalize G ([m,n]) via quintic Newton–Schulz. Operates in the
 * orientation with the smaller first dimension for cheaper m×m products,
 * then transposes back.
 */
export function newtonSchulz(G: Float32Array, m: number, n: number, steps: number): Float32Array {
  const flip = m > n;
  const X = flip ? transpose(G, m, n) : G.slice();
  const r = flip ? n : m; // rows of X
  const c = flip ? m : n; // cols of X

  const norm = frob(X) + 1e-7;
  for (let i = 0; i < X.length; i++) X[i] /= norm;

  for (let s = 0; s < steps; s++) {
    const Xt = transpose(X, r, c); // [c, r]
    const A = matmul(X, r, c, Xt, r); // [r, r] = X Xᵀ
    const AA = matmul(A, r, r, A, r); // [r, r] = A²
    // B = b*A + c*A²
    const B = new Float32Array(r * r);
    for (let i = 0; i < B.length; i++) B[i] = NS_B * A[i] + NS_C * AA[i];
    const BX = matmul(B, r, r, X, c); // [r, c]
    for (let i = 0; i < X.length; i++) X[i] = NS_A * X[i] + BX[i];
  }

  return flip ? transpose(X, r, c) : X;
}

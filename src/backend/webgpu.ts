// WebGPU backend: the CPU op set from ../model/autograd.ts as WGSL compute
// shaders, forward AND backward, behind the same Tensor interface.
//
// How it plugs in: WebGPUBackend implements OpsBackend and registers itself via
// setOpsBackend() (install()/uninstall()). ../model/qwen3.ts and the autograd
// backward() walk run unchanged — op calls route here, return ordinary Tensor
// objects, and _backward closures encode GPU dispatches instead of CPU loops.
//
// Execution model: ops record dispatches into a shared command encoder and
// return immediately; tensor data lives in GPU buffers tracked in a side
// table. Host `data`/`grad` arrays are STALE until `await sync()`, the one
// unavoidable async point (WebGPU readback is async-only). A training step is:
// uploadParams -> forwards/backwards (sync code) -> await sync() -> optimizer
// step on host arrays. See train_gpu.ts.
//
// Why shapes are baked into WGSL source instead of uniform buffers: training
// shapes are static, so every kernel compiles exactly once (pipelines cached
// by source string) and bind groups stay trivial. Generation compiles one
// variant per sequence length — acceptable at demo scale.
//
// Gradient kernels avoid atomics entirely: every backward pass is expressed so
// one thread owns one gradient element and reduces over the other axis (GEMM
// transpose flavors for linear; vocab-row scan for embedding; key-axis
// reductions for attention). Accumulation across graph fan-out and across
// batch micro-steps happens by `+=` into a gradient buffer that is zeroed
// once per step, matching the CPU semantics.

import { setOpsBackend, Tensor } from "../model/autograd.ts";
import type { OpsBackend } from "../model/autograd.ts";
import {
  bindF32,
  ceilDiv,
  GEMM_BM,
  GEMM_BN,
  MAX_WG,
  srcAttnBwdD,
  srcAttnBwdDkv,
  srcAttnBwdDq,
  srcAttnDkv,
  srcAttnDq,
  srcAttnDScore,
  srcAttnFwd,
  srcAttnOut,
  srcAttnProbs,
  srcCeBwd,
  srcCeFwd,
  srcCeReduce,
  srcElementwise,
  srcEmbeddingBwd,
  srcEmbeddingFwd,
  srcGemm,
  srcRmsNormBwdW,
  srcRmsNormBwdX,
  srcRmsNormFwd,
  srcRope,
} from "./wgsl.ts";
import type { AttnDims } from "./wgsl.ts";

// --- Minimal structural WebGPU types -----------------------------------------
// Deliberately local: TypeScript's WebGPU declarations vary across runtimes and
// lib configs, and this file must merely type-check everywhere (it only *runs*
// where navigator.gpu exists). Numeric usage flags are fixed by the WebGPU spec.

const USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
  QUERY_RESOLVE: 0x0200,
} as const;
const MAP_MODE_READ = 0x0001;

export interface GpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}
interface GpuPipeline {
  getBindGroupLayout(index: number): unknown;
}
interface GpuComputePass {
  setPipeline(p: GpuPipeline): void;
  setBindGroup(index: number, group: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}
interface GpuQuerySet {
  destroy(): void;
}
interface TimestampWrites {
  querySet: GpuQuerySet;
  beginningOfPassWriteIndex: number;
  endOfPassWriteIndex: number;
}
interface GpuCommandEncoder {
  beginComputePass(desc?: { timestampWrites?: TimestampWrites }): GpuComputePass;
  clearBuffer(buffer: GpuBuffer): void;
  copyBufferToBuffer(
    src: GpuBuffer,
    srcOff: number,
    dst: GpuBuffer,
    dstOff: number,
    size: number,
  ): void;
  resolveQuerySet(
    querySet: GpuQuerySet,
    firstQuery: number,
    queryCount: number,
    dst: GpuBuffer,
    dstOffset: number,
  ): void;
  finish(): unknown;
}
interface GpuQueue {
  submit(buffers: unknown[]): void;
  writeBuffer(buffer: GpuBuffer, offset: number, data: ArrayBufferView): void;
}
interface GpuDevice {
  createBuffer(desc: { size: number; usage: number }): GpuBuffer;
  createShaderModule(desc: { code: string }): unknown;
  createComputePipeline(desc: unknown): GpuPipeline;
  createBindGroup(desc: unknown): unknown;
  createCommandEncoder(): GpuCommandEncoder;
  createQuerySet(desc: { type: string; count: number }): GpuQuerySet;
  queue: GpuQueue;
  limits?: { maxStorageBufferBindingSize?: number };
  features?: { has(name: string): boolean };
  lost?: Promise<{ message?: string }>;
  onuncapturederror?: ((ev: { error?: { message?: string } }) => void) | null;
}

/** One kernel's accumulated GPU time from timestamp-query profiling. */
export interface KernelTime {
  label: string;
  ms: number; // total GPU time across all dispatches in the profiled window
  count: number; // number of dispatches attributed to this label
}

interface Entry {
  data: GpuBuffer;
  grad: GpuBuffer;
  bytes: number;
  external: boolean; // created outside backend ops (params, test inputs)
  gradNeedsClear: boolean;
}

// --- Buffer pool --------------------------------------------------------------
// Storage buffers keyed by 256-byte-aligned size. Training reuses identical
// shapes every step, so after step one this is pure recycling. Buffers come
// back DIRTY: anything that needs zeros (gradient accumulators) is clearBuffer'd
// explicitly before backward runs.

class BufferPool {
  private free = new Map<number, GpuBuffer[]>();
  private device: GpuDevice;

  constructor(device: GpuDevice) {
    this.device = device;
  }

  acquire(bytes: number): { buf: GpuBuffer; size: number } {
    const size = Math.max(256, Math.ceil(bytes / 256) * 256);
    const list = this.free.get(size);
    if (list && list.length > 0) return { buf: list.pop()!, size };
    const buf = this.device.createBuffer({
      size,
      usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
    });
    return { buf, size };
  }

  release(size: number, buf: GpuBuffer) {
    let list = this.free.get(size);
    if (!list) {
      list = [];
      this.free.set(size, list);
    }
    list.push(buf);
  }

  destroyAll() {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
  }
}

// --- The backend ---------------------------------------------------------------

// timestamp-query profiling state. When active, each dispatch runs in its own
// compute pass with a begin/end timestamp pair; ms accumulate per curLabel.
interface Profiler {
  qs: GpuQuerySet;
  resolve: GpuBuffer; // QUERY_RESOLVE | COPY_SRC
  read: GpuBuffer; // COPY_DST | MAP_READ
  capPairs: number; // max (begin,end) pairs = querySet.count / 2
  idx: number; // next free timestamp slot (2 per profiled dispatch)
  labels: string[]; // label per recorded pair, in slot order
  totals: Map<string, { ms: number; count: number }>;
  overflow: boolean; // ran out of query slots this window (reported, not silent)
}

export class WebGPUBackend implements OpsBackend {
  readonly adapterName: string;
  /** Whether the device granted the timestamp-query feature (see startProfile). */
  readonly timestampSupported: boolean;
  /** Whether the device granted shader-f16 (prerequisite for mixed-precision training). */
  readonly f16Supported: boolean;
  /**
   * Context-length threshold for the flash-style attention kernels. Below this
   * T, the materialized [Hq,T,T] kernels win because flash's one-thread-per-row
   * structure underoccupies the GPU when T is small (measured crossover
   * M1 Max: flash is 1.4–2x slower below T~1536 and only wins from ~T=2048).
   * Above this threshold flash removes the O(T²) buffer and improves bandwidth.
   * The large-T tests require flash; small-T parity (T=67/130/193) uses the
   * materialized path and must also pass — attention() routes automatically.
   */
  attnFlashMinT = 2048;
  /** Bytes staged back to the host by the most recent sync() (profiling). */
  lastSyncReadbackBytes = 0;
  private device: GpuDevice;
  private queue: GpuQueue;
  private pool: BufferPool;
  private pipelines = new Map<string, GpuPipeline>();
  private entries = new WeakMap<Tensor, Entry>();
  private touchedExternals = new Set<Tensor>();
  private gradKeptOnDevice = new WeakSet<Tensor>();
  private transients: { buf: GpuBuffer; size: number }[] = [];
  private pendingClears: GpuBuffer[] = [];
  private enc: GpuCommandEncoder | null = null;
  private pass: GpuComputePass | null = null;
  private backwardBegun = false;
  private deviceError: string | null = null;
  /** Label attributed to dispatches recorded via dispatch() (profiling only). */
  private curLabel = "";
  private prof: Profiler | null = null;
  /** Mixed-precision GEMM: f16 multiply, f32 accumulate (see setPrecision). */
  private f16 = false;

  constructor(
    device: GpuDevice,
    adapterName: string,
    timestampSupported = false,
    f16Supported = false,
  ) {
    this.device = device;
    this.queue = device.queue;
    this.pool = new BufferPool(device);
    this.adapterName = adapterName;
    this.timestampSupported = timestampSupported;
    this.f16Supported = f16Supported;
    device.lost?.then((info) => {
      this.deviceError = `WebGPU device lost: ${info?.message ?? "unknown"}`;
    });
    try {
      device.onuncapturederror = (ev) => {
        this.deviceError = `WebGPU error: ${ev?.error?.message ?? "unknown"}`;
      };
    } catch {
      // Runtime without onuncapturederror support; sync() still surfaces device loss.
    }
  }

  /**
   * Select matmul precision. "f16" makes the GEMM multiply operands in f16 and
   * accumulate in f32 (2x ALU throughput on packed-f16 hardware like Strix Halo;
   * on Apple GPUs f16 and f32 ALU run at the same rate, so expect little change
   * there). Buffers, gradients, and the optimizer stay f32, so no loss scaling
   * is needed. Requires the shader-f16 device feature. Set before training.
   */
  setPrecision(p: "f16" | "f32") {
    if (p === "f16" && !this.f16Supported) {
      throw new Error("setPrecision('f16'): shader-f16 not available on this device");
    }
    this.f16 = p === "f16";
  }

  /** Current matmul precision. */
  get precision(): "f16" | "f32" {
    return this.f16 ? "f16" : "f32";
  }

  /** Route the autograd op set through this backend. */
  install() {
    setOpsBackend(this);
  }

  uninstall() {
    setOpsBackend(null);
  }

  /**
   * Push current host data of these tensors to the GPU. Must be called after
   * anything mutates tensor.data on the host (e.g. an optimizer step) and
   * before the next forward. First use of a tensor uploads automatically.
   */
  uploadParams(tensors: Tensor[]) {
    for (const t of tensors) {
      const e = this.entryFor(t);
      this.queue.writeBuffer(e.data, 0, t.data);
    }
  }

  /**
   * Flush all recorded GPU work, then read back: the listed tensors' data and
   * the accumulated gradients of every external tensor (with requiresGrad)
   * touched since the last sync. After this resolves, those host arrays are
   * valid and all transient buffers are recycled.
   */
  async sync(reads: Tensor[] = []): Promise<void> {
    this.endPass();
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    for (const b of this.pendingClears) this.enc.clearBuffer(b);
    this.pendingClears = [];

    const stagings: { stage: GpuBuffer; dst: Float32Array }[] = [];
    const readSet = new Set(reads);
    for (const t of readSet) {
      const e = this.entries.get(t);
      if (!e) continue; // never materialized on GPU — host copy is already authoritative
      stagings.push({ stage: this.copyToStaging(e.data, t.size * 4), dst: t.data });
    }
    for (const t of this.touchedExternals) {
      const e = this.entries.get(t)!;
      if (t.requiresGrad && !this.gradKeptOnDevice.has(t)) {
        stagings.push({ stage: this.copyToStaging(e.grad, t.size * 4), dst: t.grad });
      }
      e.gradNeedsClear = true;
    }
    this.lastSyncReadbackBytes = stagings.reduce((a, s) => a + s.dst.length * 4, 0);

    // Resolve any profiling timestamps recorded this window into a mappable
    // buffer, ordered after all timed passes in this same encoder.
    const profTs = this.prof ? this.prof.idx : 0;
    if (this.prof && profTs > 0) {
      this.enc.resolveQuerySet(this.prof.qs, 0, profTs, this.prof.resolve, 0);
      this.enc.copyBufferToBuffer(this.prof.resolve, 0, this.prof.read, 0, profTs * 8);
    }

    this.submit();
    await Promise.all(stagings.map((s) => s.stage.mapAsync(MAP_MODE_READ)));
    for (const s of stagings) {
      s.dst.set(new Float32Array(s.stage.getMappedRange(), 0, s.dst.length));
      s.stage.unmap();
      s.stage.destroy();
    }
    if (this.prof && profTs > 0) await this.readProfile(profTs);

    // The mapAsync completions above prove all submitted work finished, so
    // every transient buffer is idle and safe to recycle.
    for (const tr of this.transients) this.pool.release(tr.size, tr.buf);
    this.transients = [];
    this.touchedExternals.clear();
    this.backwardBegun = false;

    if (this.deviceError) throw new Error(this.deviceError);
  }

  /** Free every pooled GPU buffer. The backend is unusable afterwards. */
  destroy() {
    this.uninstall();
    this.pool.destroyAll();
  }

  // --- timestamp-query profiling -------------------------------------------------

  /**
   * Begin per-kernel GPU-time profiling: until stopProfile(), each dispatch runs
   * in its own compute pass with a begin/end timestamp pair, attributed to its
   * op label, and the deltas accumulate across every sync() window. Numerically
   * a no-op (same kernels, buffers, and order — only pass batching changes).
   * Requires the device to have granted timestamp-query (see initWebGPU).
   * `capDispatches` bounds how many dispatches a single sync() window can time;
   * running past it sets the overflow flag stopProfile() returns (never silent).
   */
  startProfile(capDispatches = 4096) {
    if (!this.timestampSupported) throw new Error("timestamp-query not available on this device");
    if (this.prof) return;
    const bytes = capDispatches * 2 * 8;
    this.prof = {
      qs: this.device.createQuerySet({ type: "timestamp", count: capDispatches * 2 }),
      resolve: this.device.createBuffer({
        size: bytes,
        usage: USAGE.QUERY_RESOLVE | USAGE.COPY_SRC,
      }),
      read: this.device.createBuffer({ size: bytes, usage: USAGE.COPY_DST | USAGE.MAP_READ }),
      capPairs: capDispatches,
      idx: 0,
      labels: [],
      totals: new Map(),
      overflow: false,
    };
  }

  /** End profiling and return per-label GPU time (ms), largest first. */
  stopProfile(): { kernels: KernelTime[]; overflow: boolean } {
    const pr = this.prof;
    if (!pr) return { kernels: [], overflow: false };
    this.prof = null;
    const kernels = [...pr.totals]
      .map(([label, v]) => ({ label, ms: v.ms, count: v.count }))
      .sort((a, b) => b.ms - a.ms);
    pr.qs.destroy();
    pr.resolve.destroy();
    pr.read.destroy();
    return { kernels, overflow: pr.overflow };
  }

  /** Drain the timestamps resolved this sync() into per-label totals. */
  private async readProfile(count: number) {
    const pr = this.prof!;
    await pr.read.mapAsync(MAP_MODE_READ);
    const ts = new BigUint64Array(pr.read.getMappedRange(), 0, count);
    const pairs = count >> 1;
    for (let i = 0; i < pairs; i++) {
      const label = pr.labels[i] ?? "?";
      const t = pr.totals.get(label) ?? { ms: 0, count: 0 };
      if (ts[i * 2 + 1] > ts[i * 2]) t.ms += Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6;
      t.count += 1;
      pr.totals.set(label, t);
    }
    pr.read.unmap();
    pr.idx = 0;
    pr.labels = [];
  }

  // --- ops (OpsBackend) ---------------------------------------------------------

  linear(x: Tensor, w: Tensor): Tensor {
    this.beginForwardOp();
    this.curLabel = "linear";
    const [T, inDim] = x.shape;
    const [outDim, inDim2] = w.shape;
    if (inDim !== inDim2) throw new Error(`linear dim mismatch ${inDim} vs ${inDim2}`);
    const ex = this.entryFor(x);
    const ew = this.entryFor(w);
    const { t: out, e: eo } = this.makeOut([T, outDim], [x, w]);
    this.gemm("NT", false, T, outDim, inDim, ex.data, ew.data, eo.data);
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "linear";
      this.gemm("NN", true, T, inDim, outDim, eo.grad, ew.data, ex.grad);
      this.gemm("TN", true, outDim, inDim, T, eo.grad, ex.data, ew.grad);
    };
    return out;
  }

  add(a: Tensor, b: Tensor): Tensor {
    this.beginForwardOp();
    this.curLabel = "add";
    const ea = this.entryFor(a);
    const eb = this.entryFor(b);
    const { t: out, e: eo } = this.makeOut(a.shape, [a, b]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "AB", "read"), bindF32(1, "BB", "read"), bindF32(2, "YB", "read_write")],
        n,
        "YB[i] = AB[i] + BB[i];",
      ),
      [ea.data, eb.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "add";
      this.dispatch(
        srcElementwise(
          [
            bindF32(0, "GB", "read"),
            bindF32(1, "DA", "read_write"),
            bindF32(2, "DB", "read_write"),
          ],
          n,
          "DA[i] = DA[i] + GB[i]; DB[i] = DB[i] + GB[i];",
        ),
        [eo.grad, ea.grad, eb.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  mul(a: Tensor, b: Tensor): Tensor {
    this.beginForwardOp();
    this.curLabel = "mul";
    const ea = this.entryFor(a);
    const eb = this.entryFor(b);
    const { t: out, e: eo } = this.makeOut(a.shape, [a, b]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "AB", "read"), bindF32(1, "BB", "read"), bindF32(2, "YB", "read_write")],
        n,
        "YB[i] = AB[i] * BB[i];",
      ),
      [ea.data, eb.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "mul";
      this.dispatch(
        srcElementwise(
          [
            bindF32(0, "GB", "read"),
            bindF32(1, "AB", "read"),
            bindF32(2, "BB", "read"),
            bindF32(3, "DA", "read_write"),
            bindF32(4, "DB", "read_write"),
          ],
          n,
          "DA[i] = DA[i] + GB[i] * BB[i]; DB[i] = DB[i] + GB[i] * AB[i];",
        ),
        [eo.grad, ea.data, eb.data, ea.grad, eb.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  silu(x: Tensor): Tensor {
    this.beginForwardOp();
    this.curLabel = "silu";
    const ex = this.entryFor(x);
    const { t: out, e: eo } = this.makeOut(x.shape, [x]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "XB", "read"), bindF32(1, "YB", "read_write")],
        n,
        "let s = 1.0 / (1.0 + exp(-XB[i])); YB[i] = XB[i] * s;",
      ),
      [ex.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "silu";
      // Recompute sigmoid from the saved input instead of stashing it — one
      // fewer buffer, and the input is alive in the graph anyway.
      this.dispatch(
        srcElementwise(
          [bindF32(0, "GB", "read"), bindF32(1, "XB", "read"), bindF32(2, "DX", "read_write")],
          n,
          "let s = 1.0 / (1.0 + exp(-XB[i])); DX[i] = DX[i] + GB[i] * (s + XB[i] * s * (1.0 - s));",
        ),
        [eo.grad, ex.data, ex.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor {
    const [T, d] = x.shape;
    return this.rmsNormRows(x, weight, T, d, eps, [T, d]);
  }

  rmsNormHeads(x: Tensor, weight: Tensor, T: number, H: number, hd: number, eps: number): Tensor {
    // Per-(token, head) rows are contiguous, so this is rmsNorm on [T*H, hd].
    return this.rmsNormRows(x, weight, T * H, hd, eps, [T, H * hd]);
  }

  embedding(weight: Tensor, ids: number[]): Tensor {
    this.beginForwardOp();
    this.curLabel = "embedding";
    const [V, d] = weight.shape;
    const T = ids.length;
    const ew = this.entryFor(weight);
    const idsBuf = this.uploadU32(ids);
    const { t: out, e: eo } = this.makeOut([T, d], [weight]);
    this.dispatch(srcEmbeddingFwd(T, d), [ew.data, idsBuf, eo.data], ceilDiv(T * d, 256));
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "embedding";
      this.dispatch(srcEmbeddingBwd(T, d, V), [idsBuf, eo.grad, ew.grad], ceilDiv(V * d, 256));
    };
    return out;
  }

  rope(x: Tensor, T: number, H: number, hd: number, base: number, posOffset: number): Tensor {
    this.beginForwardOp();
    this.curLabel = "rope";
    const ex = this.entryFor(x);
    const { t: out, e: eo } = this.makeOut([T, H * hd], [x]);
    const n = T * H * (hd / 2);
    this.dispatch(srcRope(T, H, hd, base, posOffset, false), [ex.data, eo.data], ceilDiv(n, 256));
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "rope";
      this.dispatch(srcRope(T, H, hd, base, posOffset, true), [eo.grad, ex.grad], ceilDiv(n, 256));
    };
    return out;
  }

  attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    T: number,
    Hq: number,
    Hkv: number,
    hd: number,
    window = 0,
  ): Tensor {
    this.beginForwardOp();
    this.curLabel = "attention";
    const a: AttnDims = { T, Hq, Hkv, hd, window };
    const eq = this.entryFor(q);
    const ek = this.entryFor(k);
    const ev = this.entryFor(v);
    const { t: out, e: eo } = this.makeOut([T, Hq * hd], [q, k, v]);
    if (T >= this.attnFlashMinT) {
      // Flash path: per-row logsumexp (O(Hq·T)) is the only side buffer —
      // no [Hq,T,T] allocation, so context length is no longer capped by
      // maxStorageBufferBindingSize.
      const lse = this.acquireTransient(Hq * T * 4);
      this.dispatch(
        srcAttnFwd(a),
        [eq.data, ek.data, ev.data, eo.data, lse],
        ceilDiv(Hq * T, 64),
      );
      out._backward = () => {
        this.ensureBackwardBegun();
        this.curLabel = "attention";
        const dRow = this.acquireTransient(Hq * T * 4);
        this.dispatch(srcAttnBwdD(a), [eo.grad, eo.data, dRow], ceilDiv(Hq * T, 64));
        this.dispatch(
          srcAttnBwdDq(a),
          [eq.data, ek.data, ev.data, eo.grad, lse, dRow, eq.grad],
          ceilDiv(Hq * T, 64),
        );
        this.dispatch(
          srcAttnBwdDkv(a),
          [eq.data, ek.data, ev.data, eo.grad, lse, dRow, ek.grad, ev.grad],
          ceilDiv(T, 64),
          Hkv,
        );
      };
    } else {
      // Materialized path: [Hq,T,T] probability buffer — faster at small T
      // where flash's one-thread-per-row layout underoccupies the GPU.
      const probs = this.acquireTransient(Hq * T * T * 4);
      this.dispatch(srcAttnProbs(a), [eq.data, ek.data, probs], ceilDiv(Hq * T, 64));
      this.dispatch(srcAttnOut(a), [probs, ev.data, eo.data], ceilDiv(T * Hq * hd, 64));
      out._backward = () => {
        this.ensureBackwardBegun();
        this.curLabel = "attention";
        const dScore = this.acquireTransient(Hq * T * T * 4);
        this.dispatch(srcAttnDScore(a), [eo.grad, ev.data, probs, dScore], ceilDiv(Hq * T, 64));
        this.dispatch(srcAttnDq(a), [dScore, ek.data, eq.grad], ceilDiv(T * Hq * hd, 64));
        this.dispatch(srcAttnDkv(a, false), [dScore, eq.data, ek.grad], ceilDiv(T * Hkv * hd, 64));
        this.dispatch(srcAttnDkv(a, true), [probs, eo.grad, ev.grad], ceilDiv(T * Hkv * hd, 64));
      };
    }
    return out;
  }

  crossEntropy(logits: Tensor, targets: number[]): Tensor {
    this.beginForwardOp();
    this.curLabel = "crossEntropy";
    const [T, V] = logits.shape;
    const el = this.entryFor(logits);
    const tgtBuf = this.uploadU32(targets); // a target of -1 uploads as 0xffffffff (ignore)
    let kept = 0;
    for (const g of targets) if (g >= 0) kept++;
    const divBuf = this.uploadF32([kept > 0 ? kept : 1]); // mean over kept rows (== T unmasked)
    const probs = this.acquireTransient(T * V * 4);
    const perRow = this.acquireTransient(T * 4);
    const { t: loss, e: eo } = this.makeOut([1], [logits]);
    this.dispatch(srcCeFwd(T, V), [el.data, tgtBuf, probs, perRow], ceilDiv(T, 64));
    this.dispatch(srcCeReduce(T), [perRow, divBuf, eo.data], 1);
    loss._backward = () => {
      // backward(loss, seed) already wrote the seed into the HOST grad array;
      // push it into the GPU-side loss grad after the grad clears are flushed.
      this.ensureBackwardBegun();
      this.curLabel = "crossEntropy";
      this.queue.writeBuffer(eo.grad, 0, loss.grad);
      this.dispatch(srcCeBwd(T, V), [probs, tgtBuf, eo.grad, divBuf, el.grad], ceilDiv(T * V, 256));
    };
    return loss;
  }

  // --- shared machinery -----------------------------------------------------------

  /**
   * A forward op after a backward means a new micro-batch graph: arm the
   * clears-then-seed flush for the next backward pass.
   */
  private beginForwardOp() {
    this.backwardBegun = false;
    if (this.deviceError) throw new Error(this.deviceError);
  }

  /**
   * First backward closure of a graph: flush the pending gradient-buffer
   * clears and submit, so a subsequent queue.writeBuffer of the loss seed
   * lands AFTER the clears in queue order.
   */
  private ensureBackwardBegun() {
    if (this.backwardBegun) return;
    this.backwardBegun = true;
    this.endPass();
    if (this.pendingClears.length > 0) {
      if (!this.enc) this.enc = this.device.createCommandEncoder();
      for (const b of this.pendingClears) this.enc.clearBuffer(b);
      this.pendingClears = [];
    }
    this.submit();
  }

  /** Seed an output tensor's GPU gradient from its host grad array (tests). */
  seedGradFromHost(t: Tensor) {
    this.ensureBackwardBegun();
    const e = this.entries.get(t);
    if (!e) throw new Error("seedGradFromHost: tensor has no GPU buffers");
    this.queue.writeBuffer(e.grad, 0, t.grad);
  }

  // --- raw-buffer access for GPU-side optimizers (see muon_gpu.ts) ---------------

  /**
   * The device buffers backing an external tensor, creating them (and
   * uploading the current host data) on first use. The buffers are stable for
   * the tensor's lifetime, so an optimizer can capture them once.
   */
  buffersFor(t: Tensor): { data: GpuBuffer; grad: GpuBuffer } {
    const e = this.entryFor(t);
    return { data: e.data, grad: e.grad };
  }

  /**
   * Stop sync() from reading this tensor's gradient back to the host: a GPU
   * optimizer consumes it in place. Per-step grad clearing is unaffected.
   */
  keepGradOnDevice(t: Tensor) {
    this.gradKeptOnDevice.add(t);
  }

  /**
   * A fresh zero-initialized storage buffer OUTSIDE the pool. Optimizer state
   * (momentum) must survive sync()'s transient recycling and must start at
   * exactly zero — pooled buffers come back dirty, freshly created ones are
   * guaranteed zeroed by the WebGPU spec.
   */
  createStateBuffer(bytes: number): GpuBuffer {
    return this.device.createBuffer({
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
    });
  }

  /**
   * Overwrite a state buffer's contents from the host. Used for small
   * scalars a kernel reads each step (e.g. a scheduled learning rate held in
   * a 1-element buffer): the value updates in place, so the bind group cached
   * by prepareDispatch stays valid. Like uploadParams, the write is ordered
   * before any command buffer submitted after this call.
   */
  writeStateBuffer(buf: GpuBuffer, values: Float32Array) {
    this.queue.writeBuffer(buf, 0, values);
  }

  /**
   * Resolve pipeline + bind group once and return a closure that records the
   * dispatch. Only valid for PERSISTENT buffers (a pooled transient may be
   * recycled under the cached bind group). An optimizer re-records identical
   * dispatches every step, so this removes the dominant per-step encode cost
   * (measured: bind-group/source rebuilding was ~85% of the optimizer step).
   */
  prepareDispatch(code: string, buffers: GpuBuffer[], x: number, y = 1, label = ""): () => void {
    const p = this.pipeline(code);
    const group = this.bindGroup(p, buffers);
    return () => this.recordDispatch(p, group, x, y, 1, label);
  }

  /** prepareDispatch for the tiled GEMM (flavors as in srcGemm). */
  prepareGemm(
    kind: "NT" | "NN" | "TN",
    accum: boolean,
    M: number,
    N: number,
    K: number,
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
    label = "",
  ): () => void {
    return this.prepareDispatch(
      srcGemm(kind, accum, M, N, K, this.f16),
      [a, b, c],
      ceilDiv(N, GEMM_BN),
      ceilDiv(M, GEMM_BM),
      label,
    );
  }

  private entryFor(t: Tensor): Entry {
    let e = this.entries.get(t);
    if (!e) {
      // A tensor first seen as an op input was created outside the backend
      // (parameter or test input): upload its host data and give it a
      // persistent gradient accumulator.
      const bytes = t.size * 4;
      e = {
        data: this.acquirePersistent(bytes),
        grad: this.acquirePersistent(bytes),
        bytes,
        external: true,
        gradNeedsClear: true, // pool buffers arrive dirty
      };
      this.entries.set(t, e);
      this.queue.writeBuffer(e.data, 0, t.data);
    }
    if (e.external && !this.touchedExternals.has(t)) {
      this.touchedExternals.add(t);
      if (e.gradNeedsClear) {
        this.pendingClears.push(e.grad);
        e.gradNeedsClear = false;
      }
    }
    return e;
  }

  private makeOut(shape: number[], prev: Tensor[]): { t: Tensor; e: Entry } {
    const t = Tensor.zeros(shape);
    t._prev = prev;
    const bytes = t.size * 4;
    const e: Entry = {
      data: this.acquireTransient(bytes),
      grad: this.acquireTransient(bytes),
      bytes,
      external: false,
      gradNeedsClear: false,
    };
    // Gradients accumulate with +=, so the (possibly recycled) buffer must be
    // zeroed before this graph's backward pass runs.
    this.pendingClears.push(e.grad);
    this.entries.set(t, e);
    return { t, e };
  }

  private acquirePersistent(bytes: number): GpuBuffer {
    return this.pool.acquire(bytes).buf;
  }

  private acquireTransient(bytes: number): GpuBuffer {
    const { buf, size } = this.pool.acquire(bytes);
    this.transients.push({ buf, size });
    return buf;
  }

  private uploadU32(values: number[]): GpuBuffer {
    const buf = this.acquireTransient(values.length * 4);
    // queue.writeBuffer executes before any later-submitted encoder, and this
    // buffer can't appear in already-encoded (unsubmitted) passes: it was free
    // in the pool until this call.
    this.queue.writeBuffer(buf, 0, Uint32Array.from(values));
    return buf;
  }

  private uploadF32(values: number[]): GpuBuffer {
    const buf = this.acquireTransient(values.length * 4);
    this.queue.writeBuffer(buf, 0, Float32Array.from(values));
    return buf;
  }

  private gemm(
    kind: "NT" | "NN" | "TN",
    accum: boolean,
    M: number,
    N: number,
    K: number,
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
  ) {
    this.dispatch(
      srcGemm(kind, accum, M, N, K, this.f16),
      [a, b, c],
      ceilDiv(N, GEMM_BN),
      ceilDiv(M, GEMM_BM),
    );
  }

  private rmsNormRows(
    x: Tensor,
    weight: Tensor,
    rows: number,
    d: number,
    eps: number,
    outShape: number[],
  ): Tensor {
    this.beginForwardOp();
    this.curLabel = "rmsnorm";
    const ex = this.entryFor(x);
    const ew = this.entryFor(weight);
    const { t: out, e: eo } = this.makeOut(outShape, [x, weight]);
    const rInv = this.acquireTransient(rows * 4);
    this.dispatch(srcRmsNormFwd(rows, d, eps), [ex.data, ew.data, eo.data, rInv], rows);
    out._backward = () => {
      this.ensureBackwardBegun();
      this.curLabel = "rmsnorm";
      this.dispatch(srcRmsNormBwdX(rows, d), [ex.data, ew.data, eo.grad, rInv, ex.grad], rows);
      this.dispatch(
        srcRmsNormBwdW(rows, d),
        [ex.data, eo.grad, rInv, ew.grad],
        ceilDiv(d, 64),
      );
    };
    return out;
  }

  private pipeline(code: string): GpuPipeline {
    let p = this.pipelines.get(code);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipelines.set(code, p);
    }
    return p;
  }

  private bindGroup(p: GpuPipeline, buffers: GpuBuffer[]): unknown {
    return this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
    });
  }

  private dispatch(code: string, buffers: GpuBuffer[], x: number, y = 1, z = 1) {
    const p = this.pipeline(code);
    this.recordDispatch(p, this.bindGroup(p, buffers), x, y, z, this.curLabel);
  }

  /**
   * Record one dispatch into the shared encoder. Normally many dispatches share
   * one open compute pass (batched, minimal overhead). While profiling, each
   * dispatch instead runs in its own pass carrying a begin/end timestamp pair,
   * attributed to `label`; the kernels, buffers, and order are identical, so
   * results are unchanged — only the pass batching differs.
   */
  private recordDispatch(
    p: GpuPipeline,
    group: unknown,
    x: number,
    y: number,
    z: number,
    label: string,
  ) {
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    // Fold an x-workgroup count over the per-dimension cap into a 2-D grid. The
    // overflow-capable kernels (elementwise, rmsNorm rows, embedding, rope,
    // cross-entropy, AdamW) bake the matching gridX via grid2D/gridRows to
    // rebuild their index as gid.y*gridX*256 + gid.x; kernels that stay under
    // the cap dispatch unchanged (gy stays 1).
    let gx = x, gy = y;
    if (x > MAX_WG) {
      if (y !== 1) throw new Error(`dispatch x=${x} exceeds the workgroup cap with y=${y}`);
      gx = MAX_WG;
      gy = ceilDiv(x, MAX_WG);
    }
    const pr = this.prof;
    if (pr) {
      this.endPass();
      let ts: TimestampWrites | undefined;
      if (pr.idx + 2 <= pr.capPairs * 2) {
        ts = {
          querySet: pr.qs,
          beginningOfPassWriteIndex: pr.idx,
          endOfPassWriteIndex: pr.idx + 1,
        };
        pr.labels.push(label || "?");
        pr.idx += 2;
      } else {
        pr.overflow = true;
      }
      const pass = this.enc.beginComputePass(ts ? { timestampWrites: ts } : undefined);
      pass.setPipeline(p);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(gx, gy, z);
      pass.end();
      return;
    }
    if (!this.pass) this.pass = this.enc.beginComputePass();
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(gx, gy, z);
  }

  private endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }

  private submit() {
    this.endPass();
    if (this.enc) {
      this.queue.submit([this.enc.finish()]);
      this.enc = null;
    }
  }

  private copyToStaging(src: GpuBuffer, bytes: number): GpuBuffer {
    const stage = this.device.createBuffer({
      size: Math.ceil(bytes / 4) * 4,
      usage: USAGE.MAP_READ | USAGE.COPY_DST,
    });
    this.enc!.copyBufferToBuffer(src, 0, stage, 0, bytes);
    return stage;
  }
}

/**
 * Probe for a WebGPU device. Returns null when the runtime has no WebGPU
 * (Node and Bun today — run GPU work under Deno, or provide a navigator.gpu
 * polyfill); training then stays on the CPU reference backend.
 */
export async function initWebGPU(): Promise<WebGPUBackend | null> {
  // deno-lint-ignore no-explicit-any
  const nav: any = (globalThis as any).navigator;
  if (!nav?.gpu) return null;
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) return null;
  // Request the adapter's own maximum buffer limits instead of the WebGPU
  // spec's conservative default (128 MiB per storage buffer binding).
  // Attention is flash-tiled and no longer binds a [heads,T,T] buffer, but
  // other single buffers still grow with model/context size (logits [T,V]
  // and their gradients, embedding-weight gradients), so headroom stays
  // useful. Asking for exactly the adapter's reported ceiling can never
  // exceed what it supports; fall back to the default-limits device on the
  // rare adapter that still rejects it, so WebGPU availability itself never
  // regresses.
  // Opt into timestamp-query when the adapter offers it, so per-kernel GPU-time
  // profiling (startProfile) is available; absence just disables profiling.
  // Opt into optional features when offered: timestamp-query enables per-kernel
  // profiling (startProfile); shader-f16 enables mixed-precision training.
  const feats: string[] = [];
  if (adapter.features?.has?.("timestamp-query")) feats.push("timestamp-query");
  if (adapter.features?.has?.("shader-f16")) feats.push("shader-f16");
  let device: GpuDevice;
  try {
    device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
      requiredFeatures: feats,
    });
  } catch {
    device = await adapter.requestDevice();
  }
  const name = adapter.info?.description || adapter.info?.vendor || "unknown adapter";
  return new WebGPUBackend(
    device,
    String(name),
    !!device.features?.has?.("timestamp-query"),
    !!device.features?.has?.("shader-f16"),
  );
}

// Minimal, spec-faithful GGUF v3 writer and reader.
// Format reference: ggml-org/llama.cpp docs/development/gguf.md
//
// The writer is architecture-agnostic: callers add metadata key/values and
// tensors, then call build() to get the complete file bytes. The gemma3
// specifics live in ../export/export_gguf.ts.

import type { GGMLTypeId, Serialized } from "./quantize.ts";

const MAGIC = 0x46554747; // "GGUF" as little-endian uint32
const VERSION = 3;
const DEFAULT_ALIGNMENT = 32;

// gguf_metadata_value_type
export const MetaType = {
  UINT8: 0,
  INT8: 1,
  UINT16: 2,
  INT16: 3,
  UINT32: 4,
  INT32: 5,
  FLOAT32: 6,
  BOOL: 7,
  STRING: 8,
  ARRAY: 9,
  UINT64: 10,
  INT64: 11,
  FLOAT64: 12,
} as const;
type MetaTypeId = (typeof MetaType)[keyof typeof MetaType];

// A growable little-endian byte writer.
class ByteWriter {
  private buf = new Uint8Array(1 << 16);
  private dv = new DataView(this.buf.buffer);
  len = 0;

  private ensure(extra: number) {
    if (this.len + extra <= this.buf.length) return;
    let cap = this.buf.length;
    while (cap < this.len + extra) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.len));
    this.buf = next;
    this.dv = new DataView(this.buf.buffer);
  }

  u8(v: number) {
    this.ensure(1);
    this.dv.setUint8(this.len, v);
    this.len += 1;
  }
  i8(v: number) {
    this.ensure(1);
    this.dv.setInt8(this.len, v);
    this.len += 1;
  }
  u32(v: number) {
    this.ensure(4);
    this.dv.setUint32(this.len, v, true);
    this.len += 4;
  }
  i32(v: number) {
    this.ensure(4);
    this.dv.setInt32(this.len, v, true);
    this.len += 4;
  }
  f32(v: number) {
    this.ensure(4);
    this.dv.setFloat32(this.len, v, true);
    this.len += 4;
  }
  f64(v: number) {
    this.ensure(8);
    this.dv.setFloat64(this.len, v, true);
    this.len += 8;
  }
  u64(v: number) {
    this.ensure(8);
    this.dv.setBigUint64(this.len, BigInt(v), true);
    this.len += 8;
  }
  i64(v: number) {
    this.ensure(8);
    this.dv.setBigInt64(this.len, BigInt(v), true);
    this.len += 8;
  }

  bytes(b: Uint8Array) {
    this.ensure(b.length);
    this.buf.set(b, this.len);
    this.len += b.length;
  }

  padTo(align: number) {
    const rem = this.len % align;
    if (rem !== 0) {
      const pad = align - rem;
      this.ensure(pad);
      this.len += pad; // zero-filled by construction
    }
  }

  ggufString(s: string) {
    const enc = new TextEncoder().encode(s);
    this.u64(enc.length);
    this.bytes(enc);
  }

  finish(): Uint8Array {
    return this.buf.slice(0, this.len);
  }
}

type MetaValue =
  | { t: "u32"; v: number }
  | { t: "i32"; v: number }
  | { t: "u64"; v: number }
  | { t: "f32"; v: number }
  | { t: "bool"; v: boolean }
  | { t: "string"; v: string }
  | { t: "arr_str"; v: string[] }
  | { t: "arr_i32"; v: number[] }
  | { t: "arr_f32"; v: number[] };

interface TensorEntry {
  name: string;
  dims: number[]; // ggml order: ne[0] is fastest-moving
  ser: Serialized;
}

export class GGUFWriter {
  private meta: [string, MetaValue][] = [];
  private tensors: TensorEntry[] = [];
  private alignment = DEFAULT_ALIGNMENT;

  meta_u32(key: string, v: number): GGUFWriter {
    this.meta.push([key, { t: "u32", v }]);
    return this;
  }
  meta_i32(key: string, v: number): GGUFWriter {
    this.meta.push([key, { t: "i32", v }]);
    return this;
  }
  meta_u64(key: string, v: number): GGUFWriter {
    this.meta.push([key, { t: "u64", v }]);
    return this;
  }
  meta_f32(key: string, v: number): GGUFWriter {
    this.meta.push([key, { t: "f32", v }]);
    return this;
  }
  meta_bool(key: string, v: boolean): GGUFWriter {
    this.meta.push([key, { t: "bool", v }]);
    return this;
  }
  meta_string(key: string, v: string): GGUFWriter {
    this.meta.push([key, { t: "string", v }]);
    return this;
  }
  meta_arr_str(key: string, v: string[]): GGUFWriter {
    this.meta.push([key, { t: "arr_str", v }]);
    return this;
  }
  meta_arr_i32(key: string, v: number[]): GGUFWriter {
    this.meta.push([key, { t: "arr_i32", v }]);
    return this;
  }
  meta_arr_f32(key: string, v: number[]): GGUFWriter {
    this.meta.push([key, { t: "arr_f32", v }]);
    return this;
  }

  addTensor(name: string, dims: number[], ser: Serialized): GGUFWriter {
    this.tensors.push({ name, dims, ser });
    return this;
  }

  private writeMetaValue(w: ByteWriter, mv: MetaValue) {
    switch (mv.t) {
      case "u32":
        w.u32(MetaType.UINT32);
        w.u32(mv.v);
        break;
      case "i32":
        w.u32(MetaType.INT32);
        w.i32(mv.v);
        break;
      case "u64":
        w.u32(MetaType.UINT64);
        w.u64(mv.v);
        break;
      case "f32":
        w.u32(MetaType.FLOAT32);
        w.f32(mv.v);
        break;
      case "bool":
        w.u32(MetaType.BOOL);
        w.u8(mv.v ? 1 : 0);
        break;
      case "string":
        w.u32(MetaType.STRING);
        w.ggufString(mv.v);
        break;
      case "arr_str":
        w.u32(MetaType.ARRAY);
        w.u32(MetaType.STRING);
        w.u64(mv.v.length);
        for (const s of mv.v) w.ggufString(s);
        break;
      case "arr_i32":
        w.u32(MetaType.ARRAY);
        w.u32(MetaType.INT32);
        w.u64(mv.v.length);
        for (const n of mv.v) w.i32(n);
        break;
      case "arr_f32":
        w.u32(MetaType.ARRAY);
        w.u32(MetaType.FLOAT32);
        w.u64(mv.v.length);
        for (const n of mv.v) w.f32(n);
        break;
    }
  }

  build(): Uint8Array {
    // Ensure alignment is advertised so readers agree with our padding.
    if (!this.meta.some(([k]) => k === "general.alignment")) {
      this.meta.push(["general.alignment", { t: "u32", v: this.alignment }]);
    }

    const w = new ByteWriter();
    w.u32(MAGIC);
    w.u32(VERSION);
    w.u64(this.tensors.length);
    w.u64(this.meta.length);

    for (const [key, mv] of this.meta) {
      w.ggufString(key);
      this.writeMetaValue(w, mv);
    }

    // Tensor infos. Offsets are relative to the start of the tensor-data
    // section and must each be aligned.
    let runningOffset = 0;
    const offsets: number[] = [];
    for (const t of this.tensors) {
      offsets.push(runningOffset);
      let size = t.ser.bytes.length;
      const rem = size % this.alignment;
      if (rem !== 0) size += this.alignment - rem;
      runningOffset += size;
    }

    for (let i = 0; i < this.tensors.length; i++) {
      const t = this.tensors[i];
      w.ggufString(t.name);
      w.u32(t.dims.length);
      for (const d of t.dims) w.u64(d);
      w.u32(t.ser.type);
      w.u64(offsets[i]);
    }

    // Pad to alignment before the tensor-data section.
    w.padTo(this.alignment);
    const dataStart = w.len;

    for (let i = 0; i < this.tensors.length; i++) {
      // Align to the tensor's declared offset.
      const target = dataStart + offsets[i];
      while (w.len < target) w.u8(0);
      w.bytes(this.tensors[i].ser.bytes);
    }
    w.padTo(this.alignment);

    return w.finish();
  }
}

// ---------------------------------------------------------------------------
// Reader — enough to verify round-trips (metadata + tensor infos + raw data).
// ---------------------------------------------------------------------------

export interface ReadTensorInfo {
  name: string;
  dims: number[];
  type: GGMLTypeId;
  offset: number;
  data: Uint8Array;
}

export interface GGUFFile {
  version: number;
  alignment: number;
  metadata: Map<string, unknown>;
  tensors: ReadTensorInfo[];
}

class ByteReader {
  private dv: DataView;
  private buf: Uint8Array;
  pos = 0;
  constructor(buf: Uint8Array) {
    this.buf = buf;
    this.dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  u8() {
    const v = this.dv.getUint8(this.pos);
    this.pos += 1;
    return v;
  }
  i8() {
    const v = this.dv.getInt8(this.pos);
    this.pos += 1;
    return v;
  }
  u32() {
    const v = this.dv.getUint32(this.pos, true);
    this.pos += 4;
    return v;
  }
  i32() {
    const v = this.dv.getInt32(this.pos, true);
    this.pos += 4;
    return v;
  }
  f32() {
    const v = this.dv.getFloat32(this.pos, true);
    this.pos += 4;
    return v;
  }
  f64() {
    const v = this.dv.getFloat64(this.pos, true);
    this.pos += 8;
    return v;
  }
  u64() {
    const v = Number(this.dv.getBigUint64(this.pos, true));
    this.pos += 8;
    return v;
  }
  i64() {
    const v = Number(this.dv.getBigInt64(this.pos, true));
    this.pos += 8;
    return v;
  }
  str() {
    const len = this.u64();
    const s = new TextDecoder().decode(this.buf.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }
  raw(n: number) {
    const b = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return b;
  }
}

function readMetaValue(r: ByteReader): unknown {
  const type = r.u32() as MetaTypeId;
  switch (type) {
    case MetaType.UINT8:
      return r.u8();
    case MetaType.INT8:
      return r.i8();
    case MetaType.UINT16:
      return r.u32(); // not used by us
    case MetaType.INT16:
      return r.i32();
    case MetaType.UINT32:
      return r.u32();
    case MetaType.INT32:
      return r.i32();
    case MetaType.FLOAT32:
      return r.f32();
    case MetaType.BOOL:
      return r.u8() !== 0;
    case MetaType.STRING:
      return r.str();
    case MetaType.UINT64:
      return r.u64();
    case MetaType.INT64:
      return r.i64();
    case MetaType.FLOAT64:
      return r.f64();
    case MetaType.ARRAY: {
      const elemType = r.u32() as MetaTypeId;
      const n = r.u64();
      const arr: unknown[] = [];
      for (let i = 0; i < n; i++) {
        switch (elemType) {
          case MetaType.STRING:
            arr.push(r.str());
            break;
          case MetaType.INT32:
            arr.push(r.i32());
            break;
          case MetaType.UINT32:
            arr.push(r.u32());
            break;
          case MetaType.FLOAT32:
            arr.push(r.f32());
            break;
          default:
            throw new Error(`Unsupported array elem type ${elemType}`);
        }
      }
      return arr;
    }
    default:
      throw new Error(`Unsupported metadata type ${type}`);
  }
}

export function readGGUF(buf: Uint8Array): GGUFFile {
  const r = new ByteReader(buf);
  const magic = r.u32();
  if (magic !== MAGIC) throw new Error("Not a GGUF file (bad magic)");
  const version = r.u32();
  const tensorCount = r.u64();
  const metaCount = r.u64();

  const metadata = new Map<string, unknown>();
  for (let i = 0; i < metaCount; i++) {
    const key = r.str();
    metadata.set(key, readMetaValue(r));
  }

  const alignment = (metadata.get("general.alignment") as number) ?? DEFAULT_ALIGNMENT;

  const infos: { name: string; dims: number[]; type: GGMLTypeId; offset: number }[] = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = r.str();
    const nDims = r.u32();
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const type = r.u32() as GGMLTypeId;
    const offset = r.u64();
    infos.push({ name, dims, type, offset });
  }

  // Align to data section.
  if (r.pos % alignment !== 0) r.pos += alignment - (r.pos % alignment);
  const dataStart = r.pos;

  const tensors: ReadTensorInfo[] = infos.map((info, i) => {
    const start = dataStart + info.offset;
    const end = i + 1 < infos.length ? dataStart + infos[i + 1].offset : buf.length;
    return { ...info, data: buf.subarray(start, end) };
  });

  return { version, alignment, metadata, tensors };
}

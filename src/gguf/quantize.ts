// Tensor serializers matching llama.cpp / ggml memory layout exactly.
// A serializer takes a Float32Array and returns the raw bytes ggml expects,
// plus the ggml type id to record in the tensor info.

import { f16BitsToF32, f32ToF16Bits } from "./f16.ts";

export const GGMLType = {
  F32: 0,
  F16: 1,
  Q4_0: 2,
  Q8_0: 8,
} as const;
export type GGMLTypeId = (typeof GGMLType)[keyof typeof GGMLType];

export interface Serialized {
  type: GGMLTypeId;
  bytes: Uint8Array;
}

export function serializeF32(data: Float32Array): Serialized {
  // Copy so we own an aligned little-endian buffer.
  const out = new Uint8Array(data.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < data.length; i++) dv.setFloat32(i * 4, data[i], true);
  return { type: GGMLType.F32, bytes: out };
}

export function serializeF16(data: Float32Array): Serialized {
  const out = new Uint8Array(data.length * 2);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < data.length; i++) {
    dv.setUint16(i * 2, f32ToF16Bits(data[i]), true);
  }
  return { type: GGMLType.F16, bytes: out };
}

const QK = 32; // block size shared by Q4_0 and Q8_0

/**
 * Q8_0: blocks of 32. Layout per block = f16 scale (2 bytes) + 32 int8.
 * scale d = maxabs / 127; q = round(x / d).
 */
export function serializeQ8_0(data: Float32Array): Serialized {
  if (data.length % QK !== 0) {
    throw new Error(`Q8_0 needs a length multiple of ${QK}, got ${data.length}`);
  }
  const nBlocks = data.length / QK;
  const blockBytes = 2 + QK; // 34
  const out = new Uint8Array(nBlocks * blockBytes);
  const dv = new DataView(out.buffer);

  for (let b = 0; b < nBlocks; b++) {
    let amax = 0;
    for (let j = 0; j < QK; j++) {
      const v = Math.abs(data[b * QK + j]);
      if (v > amax) amax = v;
    }
    const d = amax / 127;
    const id = d !== 0 ? 1 / d : 0;
    const base = b * blockBytes;
    dv.setUint16(base, f32ToF16Bits(d), true);
    for (let j = 0; j < QK; j++) {
      const q = Math.round(data[b * QK + j] * id);
      dv.setInt8(base + 2 + j, Math.max(-127, Math.min(127, q)));
    }
  }
  return { type: GGMLType.Q8_0, bytes: out };
}

/**
 * Q4_0: blocks of 32. Layout per block = f16 scale (2 bytes) + 16 bytes of
 * packed 4-bit nibbles. Values are symmetric around 8: q in [0,15], x ~ d*(q-8).
 * scale d = maxabs / -8 (ggml convention), nibble = round(x/d) + 8.
 */
export function serializeQ4_0(data: Float32Array): Serialized {
  if (data.length % QK !== 0) {
    throw new Error(`Q4_0 needs a length multiple of ${QK}, got ${data.length}`);
  }
  const nBlocks = data.length / QK;
  const blockBytes = 2 + QK / 2; // 18
  const out = new Uint8Array(nBlocks * blockBytes);
  const dv = new DataView(out.buffer);

  for (let b = 0; b < nBlocks; b++) {
    // Find value with largest magnitude (keep its sign, ggml-style).
    let max = 0;
    let amax = 0;
    for (let j = 0; j < QK; j++) {
      const v = data[b * QK + j];
      const a = Math.abs(v);
      if (a > amax) {
        amax = a;
        max = v;
      }
    }
    const d = max / -8;
    const id = d !== 0 ? 1 / d : 0;
    const base = b * blockBytes;
    dv.setUint16(base, f32ToF16Bits(d), true);
    for (let j = 0; j < QK / 2; j++) {
      const x0 = data[b * QK + j] * id + 8.5;
      const x1 = data[b * QK + j + QK / 2] * id + 8.5;
      const n0 = Math.min(15, Math.max(0, Math.floor(x0)));
      const n1 = Math.min(15, Math.max(0, Math.floor(x1)));
      out[base + 2 + j] = n0 | (n1 << 4);
    }
  }
  return { type: GGMLType.Q4_0, bytes: out };
}

// ---------------------------------------------------------------------------
// Dequantizers (inverse of the serializers) — used for round-trip checks and
// as the basis for a future GGUF checkpoint loader.
// ---------------------------------------------------------------------------

export function dequantize(type: GGMLTypeId, bytes: Uint8Array, count: number): Float32Array {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  switch (type) {
    case GGMLType.F32:
      for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
      return out;
    case GGMLType.F16:
      for (let i = 0; i < count; i++) out[i] = f16BitsToF32(dv.getUint16(i * 2, true));
      return out;
    case GGMLType.Q8_0: {
      const blockBytes = 2 + QK;
      const nBlocks = count / QK;
      for (let b = 0; b < nBlocks; b++) {
        const base = b * blockBytes;
        const d = f16BitsToF32(dv.getUint16(base, true));
        for (let j = 0; j < QK; j++) out[b * QK + j] = dv.getInt8(base + 2 + j) * d;
      }
      return out;
    }
    case GGMLType.Q4_0: {
      const blockBytes = 2 + QK / 2;
      const nBlocks = count / QK;
      for (let b = 0; b < nBlocks; b++) {
        const base = b * blockBytes;
        const d = f16BitsToF32(dv.getUint16(base, true));
        for (let j = 0; j < QK / 2; j++) {
          const byte = bytes[base + 2 + j];
          out[b * QK + j] = ((byte & 0x0f) - 8) * d;
          out[b * QK + j + QK / 2] = ((byte >> 4) - 8) * d;
        }
      }
      return out;
    }
  }
}

export type QuantName = "f32" | "f16" | "q8_0" | "q4_0";

export function serialize(data: Float32Array, quant: QuantName): Serialized {
  switch (quant) {
    case "f32":
      return serializeF32(data);
    case "f16":
      return serializeF16(data);
    case "q8_0":
      return serializeQ8_0(data);
    case "q4_0":
      return serializeQ4_0(data);
  }
}

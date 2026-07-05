// IEEE binary16 (half) <-> f32 conversion for host<->GPU transfer in
// mixed-precision training: the GPU stores activations/weights as f16, but the
// host Tensor arrays and the optimizer master weights stay f32, so uploads
// encode f32->f16 and readbacks decode f16->f32. Round-to-nearest-even on
// encode, with full subnormal / inf / nan handling. Pure and dependency-free
// (runs on Deno/Bun/Node); the WGSL side uses the native `f16` type.

const _buf = new ArrayBuffer(4);
const _f32 = new Float32Array(_buf);
const _u32 = new Uint32Array(_buf);

/** Encode one f32 to the 16 bits of an IEEE binary16, round-to-nearest-even. */
export function f32ToF16Bits(value: number): number {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const frac = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (frac ? 0x200 : 0); // inf / nan (keep nan)
  let hexp = exp - 112; // rebias: (exp-127)+15
  if (hexp >= 0x1f) return sign | 0x7c00; // overflow -> inf
  if (hexp <= 0) {
    if (hexp < -10) return sign; // underflow -> signed zero
    const f = frac | 0x800000; // restore implicit leading 1
    const shift = 14 - hexp; // 14..24
    let h = f >>> shift;
    const roundBit = 1 << (shift - 1);
    const rem = f & ((1 << shift) - 1);
    if (rem > roundBit || (rem === roundBit && (h & 1))) h += 1; // round to nearest even
    return sign | h;
  }
  let h = frac >>> 13;
  const rem = frac & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) {
    h += 1;
    if (h === 0x400) { // mantissa carry into exponent
      h = 0;
      hexp += 1;
      if (hexp >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (hexp << 10) | h;
}

/** Decode the 16 bits of an IEEE binary16 to f32. */
export function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 5.960464477539063e-8; // subnormal: mant * 2^-24
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Encode an f32 array to a packed f16 (Uint16) array. */
export function f32ArrayToF16(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = f32ToF16Bits(src[i]);
  return out;
}

/** Decode a packed f16 (Uint16) array into f32 (into `out` if given). */
export function f16ArrayToF32(src: Uint16Array, out?: Float32Array): Float32Array {
  const o = out ?? new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) o[i] = f16BitsToF32(src[i]);
  return o;
}

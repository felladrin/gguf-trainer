// IEEE-754 half-precision (float16) <-> float32 conversion.
// Pure arithmetic, no dependencies. Used by GGUF F16 + block-quant scales.

/** Convert a single float32 value to a uint16 holding its float16 bits. */
export function f32ToF16Bits(value: number): number {
  const f = new Float32Array(1);
  const i = new Uint32Array(f.buffer);
  f[0] = value;
  const x = i[0];

  const sign = (x >>> 16) & 0x8000;
  let mantissa = x & 0x007fffff;
  const exp = (x >>> 23) & 0xff;

  if (exp === 0xff) {
    // Inf / NaN
    return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
  }

  // Unbias float32 (127) -> rebias float16 (15).
  const e = exp - 127 + 15;

  if (e >= 0x1f) {
    // Overflow -> Inf
    return sign | 0x7c00;
  }
  if (e <= 0) {
    if (e < -10) return sign; // too small -> signed zero
    // Subnormal: add implicit leading 1, then shift.
    mantissa |= 0x00800000;
    const shift = 14 - e;
    let half = mantissa >>> shift;
    // Round to nearest even.
    const roundBit = (mantissa >>> (shift - 1)) & 1;
    if (roundBit) half += 1;
    return sign | half;
  }

  // Normalized. Round mantissa to 10 bits, nearest-even.
  let half = (e << 10) | (mantissa >>> 13);
  const roundBit = (mantissa >>> 12) & 1;
  const sticky = (mantissa & 0x0fff) !== 0 ? 1 : 0;
  const lsb = half & 1;
  if (roundBit && (sticky || lsb)) half += 1; // may carry into exponent, which is correct
  return sign | half;
}

/** Convert a uint16 float16-bit pattern back to float32. */
export function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x03ff;

  const i = new Uint32Array(1);
  const f = new Float32Array(i.buffer);

  if (exp === 0) {
    if (mant === 0) {
      i[0] = sign;
    } else {
      // Subnormal half -> normalized float.
      let e = -1;
      let m = mant;
      do {
        e += 1;
        m <<= 1;
      } while ((m & 0x0400) === 0);
      m &= 0x03ff;
      i[0] = sign | ((127 - 15 - e) << 23) | (m << 13);
    }
  } else if (exp === 0x1f) {
    i[0] = sign | 0x7f800000 | (mant << 13);
  } else {
    i[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return f[0];
}

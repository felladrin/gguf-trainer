// Token corpus access for training. The trainer samples fixed-length windows
// from a token stream; a TokenSource abstracts where that stream lives so a
// corpus too large to hold in memory can be streamed off disk instead.
//
// Two implementations:
//   - memTokenSource: the whole corpus in a typed array (Uint16/Uint32) or a
//     plain number[]: fine up to ~10^8 tokens.
//   - diskTokenSource: windows read on demand from a pretokenized binary file
//     (writeTokenFile), so peak memory is O(window), not O(corpus). This is the
//     path for FineWeb-Edu-scale corpora.
//
// The on-disk format is bare little-endian tokens (no header): 2 bytes each
// when the vocab fits in u16, else 4. tokenBytes(vocabSize) picks the width;
// the reader is told the width (the model's config carries the vocab size).

import { openReader, writeFileBytes } from "../io.ts";

export interface TokenSource {
  /** Number of tokens in the corpus. */
  readonly length: number;
  /** Tokens [start, start+len) as a plain array (what the trainer feeds forward). */
  window(start: number, len: number): number[];
  /** Release any held file handle; no-op for in-memory sources. */
  close(): void;
}

/** Bytes per token for a given vocab: 2 (u16) when it fits, else 4 (u32). */
export function tokenBytes(vocabSize: number): 2 | 4 {
  return vocabSize <= 0x10000 ? 2 : 4;
}

/** Wrap an in-memory token array (typed or plain) as a TokenSource. */
export function memTokenSource(data: ArrayLike<number>): TokenSource {
  return {
    length: data.length,
    window(start, len) {
      const out = new Array<number>(len);
      for (let i = 0; i < len; i++) out[i] = data[start + i];
      return out;
    },
    close() {},
  };
}

/** number[] -> memTokenSource; an existing TokenSource passes through. Lets the
 * trainers accept either without callers changing. */
export function toTokenSource(t: number[] | TokenSource): TokenSource {
  return Array.isArray(t) ? memTokenSource(t) : t;
}

/** Serialize a token array to the bare little-endian format above. */
export async function writeTokenFile(
  path: string,
  tokens: ArrayLike<number>,
  bytesPerToken: 2 | 4,
): Promise<void> {
  const bytes = new Uint8Array(tokens.length * bytesPerToken);
  const dv = new DataView(bytes.buffer);
  for (let i = 0; i < tokens.length; i++) {
    if (bytesPerToken === 2) dv.setUint16(i * 2, tokens[i], true);
    else dv.setUint32(i * 4, tokens[i], true);
  }
  await writeFileBytes(path, bytes);
}

/**
 * A disk-backed TokenSource: each window() reads only its bytes from the file
 * (random access), so the corpus never fully enters memory. `bytesPerToken`
 * must match what writeTokenFile wrote (use tokenBytes(vocabSize)).
 */
export async function diskTokenSource(
  path: string,
  bytesPerToken: 2 | 4,
): Promise<TokenSource> {
  const reader = await openReader(path);
  if (reader.size % bytesPerToken !== 0) {
    throw new Error(`token file size ${reader.size} not a multiple of ${bytesPerToken}`);
  }
  const length = reader.size / bytesPerToken;
  return {
    length,
    window(start, len) {
      if (start < 0 || start + len > length) {
        throw new Error(`window [${start},${start + len}) out of range 0..${length}`);
      }
      const raw = reader.readAt(start * bytesPerToken, len * bytesPerToken);
      const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      const out = new Array<number>(len);
      for (let i = 0; i < len; i++) {
        out[i] = bytesPerToken === 2 ? dv.getUint16(i * 2, true) : dv.getUint32(i * 4, true);
      }
      return out;
    },
    close() {
      reader.close();
    },
  };
}

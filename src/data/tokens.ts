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

import { chunkSpans, openReader, writeFileBytes } from "../io.ts";

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

/** Growable id buffers must match the file width, or a large vocab wraps. */
export type IdArray = Uint16Array | Uint32Array;

/**
 * The typed-array constructor an id buffer for this vocab must use.
 *
 * A u16 buffer does not fail on a vocab past 65,536: it truncates each id into a
 * smaller one that is itself a legal id, so nothing downstream can detect it.
 * Qwen3 is 151,936 tokens, Llama-3 128,256, Gemma 262,144, so any encoder that
 * may see a resumed foreign vocab has to ask before allocating.
 */
export function idArrayFor(vocabSize: number): Uint16ArrayConstructor | Uint32ArrayConstructor {
  return tokenBytes(vocabSize) === 2 ? Uint16Array : Uint32Array;
}

/** Wrap an in-memory token array (typed or plain) as a TokenSource. */
export function memTokenSource(data: ArrayLike<number>): TokenSource {
  return {
    length: data.length,
    window(start, len) {
      // The same bounds check diskTokenSource has always had. Without it a
      // window past the end returned `undefined` for every token past it, which
      // the losses now refuse as "not an integer", by an odd route.
      if (start < 0 || start + len > data.length) {
        throw new Error(`window [${start},${start + len}) out of range 0..${data.length}`);
      }
      const out = new Array<number>(len);
      for (let i = 0; i < len; i++) out[i] = data[start + i];
      return out;
    },
    close() {},
  };
}

/** Tokens per read in the corpus scan below: 8 MB of number[] at a time. */
const SCAN_CHUNK = 1 << 20;

/**
 * Refuse a corpus that does not fit the checkpoint's vocab, at the point the
 * source opens rather than on whichever window happens to contain the id.
 *
 * The losses and the embedding refuse an out-of-range id themselves (levers 26,
 * 29 and 30), but they do it mid-run. `pretrain`'s trust gate only reads the
 * first 16 tokens, so a `.tokens` file built with the wrong tokenizer passes it
 * and the run can be tens of thousands of steps in before some later window
 * happens to hold a high id. Everything written up to that point trained on
 * whatever the guards were catching.
 *
 * One sequential pass over a file the run is about to read thousands of times,
 * measured at 310M tokens/s on this machine, so a FineWeb-scale 10B-token corpus
 * costs about 32 seconds once. `from` exists because `eval-loss` scores only the
 * tail of its file and has no business reading the rest; `chunk` because the
 * chunking is otherwise untestable at a size a test can build. Returns the
 * number of tokens scanned, so a caller can say so.
 */
export function assertCorpusFitsVocab(
  src: TokenSource,
  vocabSize: number,
  path: string,
  { from = 0, chunk = SCAN_CHUNK }: { from?: number; chunk?: number } = {},
): number {
  for (const { off, len } of chunkSpans(src.length - from, chunk)) {
    const start = from + off;
    const w = src.window(start, len);
    for (let i = 0; i < len; i++) {
      const id = w[i];
      if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
        throw new Error(
          `${path}: token ${id} at position ${start + i} is outside [0,${vocabSize}). ` +
            `The corpus was tokenized with a different vocab than the checkpoint; ` +
            `retokenize it with the checkpoint's own tokenizer.`,
        );
      }
    }
  }
  return src.length - from;
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

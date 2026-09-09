// Runtime-agnostic file I/O via the node:fs compatibility layer, which Deno,
// Bun, and Node all implement. Keeps the rest of the codebase runtime-neutral.

/**
 * Largest buffer handed to a single write call.
 *
 * `fs.writeFileSync(path, data)` does not survive a `data` longer than 2^31
 * bytes on Deno 2.9.1: instead of failing it writes without bound, and a 2.39 GB
 * export grew past 1 TB and filled the disk before anything noticed. Reads are
 * unaffected; `readFileSync` returns a >2 GiB file correctly. 1 GiB leaves the
 * boundary a wide margin and costs three write calls on the largest checkpoint
 * this trainer produces.
 *
 * This is not hypothetical for the models in the readme's own table: an f32
 * GGUF passes 2^31 bytes at ~537M parameters, so Qwen3-0.6B-Base (2.22 GiB) and
 * TinyLlama_v1.1 (4.10 GiB) both land above it.
 */
export const WRITE_CHUNK_BYTES = 1 << 30;

/**
 * The spans a write of `total` bytes is split into, in order, none longer than
 * `chunk`. Separated from the I/O so the arithmetic that keeps every call under
 * the boundary can be checked without writing gigabytes.
 */
export function writeSpans(
  total: number,
  chunk: number = WRITE_CHUNK_BYTES,
): { off: number; len: number }[] {
  if (!(chunk > 0)) throw new Error(`writeSpans: chunk must be positive, got ${chunk}`);
  const spans: { off: number; len: number }[] = [];
  for (let off = 0; off < total; off += chunk) {
    spans.push({ off, len: Math.min(chunk, total - off) });
  }
  return spans;
}

export async function writeFileBytes(path: string, data: Uint8Array): Promise<void> {
  const fs = await import("node:fs");
  const fd = fs.openSync(path, "w");
  try {
    for (const { off, len } of writeSpans(data.length)) {
      // writeSync may satisfy only part of a request, so drain each span.
      let done = 0;
      while (done < len) done += fs.writeSync(fd, data, off + done, len - done);
    }
  } finally {
    fs.closeSync(fd);
  }
}

export async function readFileText(path: string): Promise<string> {
  const fs = await import("node:fs");
  return fs.readFileSync(path, "utf8");
}

export async function readFileBytes(path: string): Promise<Uint8Array> {
  const fs = await import("node:fs");
  return new Uint8Array(fs.readFileSync(path));
}

/** Random-access byte reader over an open file, for streaming windows out of a
 * large corpus without loading it whole. node:fs's openSync/readSync/closeSync
 * are implemented by Deno, Bun, and Node alike. Call close() when done. */
export interface RandomReader {
  size: number;
  readAt(offset: number, length: number): Uint8Array;
  close(): void;
}

export async function openReader(path: string): Promise<RandomReader> {
  const fs = await import("node:fs");
  const fd = fs.openSync(path, "r");
  const size = fs.fstatSync(fd).size;
  return {
    size,
    readAt(offset: number, length: number): Uint8Array {
      const buf = new Uint8Array(length);
      let read = 0;
      // readSync may return fewer bytes than requested; loop until satisfied.
      while (read < length) {
        const n = fs.readSync(fd, buf, read, length - read, offset + read);
        if (n <= 0) break;
        read += n;
      }
      if (read !== length) throw new Error(`short read at ${offset}: ${read}/${length}`);
      return buf;
    },
    close() {
      fs.closeSync(fd);
    },
  };
}

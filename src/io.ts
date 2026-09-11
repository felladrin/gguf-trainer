// File I/O via node:fs, which Deno implements, so the size and chunking rules
// below live in one place instead of at every call site.

/**
 * Largest buffer handed to a single write call.
 *
 * `fs.writeFileSync(path, data)` does not survive a `data` longer than 2^31
 * bytes on Deno 2.9.1: instead of failing it writes without bound, and a 2.39 GB
 * export grew past 1 TB and filled the disk before anything noticed. Reads are
 * unaffected: `readFileSync` returns a >2 GiB file correctly. 1 GiB leaves the
 * boundary a wide margin: three write calls on a 2.22 GiB Qwen3-0.6B export,
 * five on a 4.10 GiB TinyLlama one.
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
export function chunkSpans(
  total: number,
  chunk: number,
): { off: number; len: number }[] {
  // A positive INTEGER: a fractional chunk terminates but hands a caller a
  // fractional length, and a non-positive one never advances at all.
  if (!Number.isInteger(chunk) || chunk < 1) {
    throw new Error(`chunkSpans: chunk must be a positive whole number, got ${chunk}`);
  }
  const spans: { off: number; len: number }[] = [];
  for (let off = 0; off < total; off += chunk) {
    spans.push({ off, len: Math.min(chunk, total - off) });
  }
  return spans;
}

/**
 * `chunk` exists so a test can drive the span loop without writing gigabytes;
 * production callers leave it alone. Without it the loop is unreachable below
 * 1 GiB, which is every input the default test suite can afford to build.
 */
export async function writeFileBytes(
  path: string,
  data: Uint8Array,
  chunk: number = WRITE_CHUNK_BYTES,
): Promise<void> {
  const fs = await import("node:fs");
  // Spans first: openSync(path, "w") truncates, and an invalid chunk should not
  // cost an existing file before it throws.
  const spans = chunkSpans(data.length, chunk);
  const fd = fs.openSync(path, "w");
  try {
    for (const { off, len } of spans) {
      // writeSync may satisfy only part of a request, so drain each span. A
      // non-positive return would otherwise spin here forever, silently.
      let done = 0;
      while (done < len) {
        const n = fs.writeSync(fd, data, off + done, len - done);
        if (n <= 0) throw new Error(`short write at ${off + done}: writeSync returned ${n}`);
        done += n;
      }
    }
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * A file's text, or null when it is absent.
 *
 * Only ENOENT means absent. Turning every read failure into "not there" is the
 * bug pretrain's optstate probe already paid for once, where decoding a
 * multi-GB sidecar as UTF-8 overflowed and silently reported no optstate.
 */
export async function readFileTextIfPresent(path: string): Promise<string | null> {
  try {
    return await readFileText(path);
  } catch (e) {
    if ((e as { code?: string }).code === "ENOENT") return null;
    throw e;
  }
}

/** Delete a file if it is there. Absence is the desired end state, not an error. */
export async function removeIfPresent(path: string): Promise<void> {
  const fs = await import("node:fs");
  fs.rmSync(path, { force: true });
}

export async function fileSize(path: string): Promise<number> {
  const fs = await import("node:fs");
  return fs.statSync(path).size;
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
 * large corpus without loading it whole.
 * Call close() when done. */
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

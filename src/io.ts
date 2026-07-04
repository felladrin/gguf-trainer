// Runtime-agnostic file I/O via the node:fs compatibility layer, which Deno,
// Bun, and Node all implement. Keeps the rest of the codebase runtime-neutral.

export async function writeFileBytes(path: string, data: Uint8Array): Promise<void> {
  const fs = await import("node:fs");
  fs.writeFileSync(path, data);
}

export async function readFileText(path: string): Promise<string> {
  const fs = await import("node:fs");
  return fs.readFileSync(path, "utf8");
}

export async function readFileBytes(path: string): Promise<Uint8Array> {
  const fs = await import("node:fs");
  return new Uint8Array(fs.readFileSync(path));
}

/** Random-access byte reader over an open file — for streaming windows out of a
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

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

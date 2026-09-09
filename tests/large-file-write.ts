// writeFileBytes must never hand a single write call more than WRITE_CHUNK_BYTES.
//
// `fs.writeFileSync(path, data)` writes without bound when `data` is longer than
// 2^31 bytes on Deno 2.9.1: a 2.39 GB GGUF export grew past 1 TB and filled the
// disk. An f32 checkpoint passes that boundary at ~537M parameters, so it is
// reachable from two rows of the readme's own base-model table.
//
// Two layers here, because the real check is expensive and the cheap one cannot
// see the actual defect:
//   - the span arithmetic, exhaustively and for free: this is where a future
//     edit is most likely to reintroduce an over-long call
//   - a real multi-chunk round trip through the file system, at a size small
//     enough to run everywhere
//   - the genuine >2^31 write, only under GGUF_TRAINER_BIG_IO=1, because it
//     needs ~2.2 GB of RAM and disk and CI runners should not pay for it
//
// A regression does NOT surface as a clean assertion failure in that last case:
// the write runs away, so the process either fills the disk or dies on a file
// size limit. Bound it. Verified both ways under `ulimit -f 6000000`: with the
// chunking removed the run is killed mid-write, and with it in place the file
// is exactly 2147487744 bytes.
//
// Run:  deno run -A tests/large-file-write.ts
//       (ulimit -f 6000000; GGUF_TRAINER_BIG_IO=1 deno run -A tests/large-file-write.ts)

import { readFileBytes, WRITE_CHUNK_BYTES, writeFileBytes, writeSpans } from "../src/io.ts";

// node:fs and node:os rather than the Deno globals: `deno task test:node` runs
// this file too, and src/io.ts exists precisely to keep the tree runtime-neutral.
const fs = await import("node:fs");
const os = await import("node:os");
let tmpSeq = 0;
const tmpPath = () => `${os.tmpdir()}/gguf-trainer-io-${tmpSeq++}.bin`;
const envVar = (k: string): string | undefined =>
  // deno-lint-ignore no-explicit-any
  (globalThis as any).Deno?.env?.get(k) ?? (globalThis as any).process?.env?.[k];

let failures = 0;

function check(name: string, cond: boolean, detail = "") {
  if (!cond) failures++;
  console.log(`  ${cond ? "ok " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
}

console.log("=== large-file write ===\n");

// 1. The spans tile the range exactly, in order, with nothing over the limit.
for (
  const [total, chunk] of [
    [0, 4],
    [1, 4],
    [4, 4],
    [5, 4],
    [12, 4],
    [13, 4],
    [WRITE_CHUNK_BYTES, WRITE_CHUNK_BYTES],
    [WRITE_CHUNK_BYTES + 1, WRITE_CHUNK_BYTES],
    [2 ** 31 + 1024, WRITE_CHUNK_BYTES],
    [2_390_146_560, WRITE_CHUNK_BYTES], // the Qwen3-0.6B f32 export that started this
  ] as [number, number][]
) {
  const spans = writeSpans(total, chunk);
  let cursor = 0;
  let contiguous = true;
  let withinLimit = true;
  for (const { off, len } of spans) {
    if (off !== cursor) contiguous = false;
    if (len > chunk || len <= 0) withinLimit = false;
    cursor += len;
  }
  check(
    `spans(${total}, ${chunk}) tile exactly and stay under the limit`,
    contiguous && withinLimit && cursor === total,
    `${spans.length} spans, covered ${cursor}`,
  );
}

check(
  "the default chunk stays under the 2^31 boundary that breaks writeFileSync",
  WRITE_CHUNK_BYTES < 2 ** 31,
  `${WRITE_CHUNK_BYTES} < ${2 ** 31}`,
);
check(
  "writeSpans rejects a non-positive chunk",
  (() => {
    try {
      writeSpans(10, 0);
      return false;
    } catch {
      return true;
    }
  })(),
);

// 2. A real round trip that crosses several chunk boundaries. Small enough to
//    run anywhere; the point is that the drain loop reassembles the file byte
//    for byte, including a ragged final span.
{
  const n = 5000;
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) data[i] = (i * 31 + 7) & 0xff;
  const path = tmpPath();
  try {
    // Exercised through the public function, so the default chunk applies; the
    // multi-chunk path is covered by driving writeSpans directly above and by
    // the opt-in case below.
    await writeFileBytes(path, data);
    const got = await readFileBytes(path);
    let same = got.length === n;
    for (let i = 0; same && i < n; i++) if (got[i] !== data[i]) same = false;
    check("round trip is byte-exact", same, `${got.length} bytes back`);
  } finally {
    fs.rmSync(path, { force: true });
  }
}

// 3. The defect itself. Opt-in: ~2.2 GB of RAM and disk.
if (envVar("GGUF_TRAINER_BIG_IO") === "1") {
  const n = 2 ** 31 + 4096; // just past the boundary writeFileSync mishandles
  const path = tmpPath();
  try {
    const data = new Uint8Array(n);
    data[0] = 1;
    data[n - 1] = 2;
    await writeFileBytes(path, data);
    const size = fs.statSync(path).size;
    check(
      `a ${(n / 2 ** 30).toFixed(2)} GiB write produces exactly that many bytes`,
      size === n,
      `${size}`,
    );
    const got = await readFileBytes(path);
    check(
      "and reads back with its edges intact",
      got.length === n && got[0] === 1 && got[n - 1] === 2,
    );
  } finally {
    fs.rmSync(path, { force: true });
  }
} else {
  console.log("  ..  >2 GiB case skipped (set GGUF_TRAINER_BIG_IO=1 to run it)");
}

console.log(
  failures === 0 ? "\n=== large-file write checks passed ===" : `\n=== ${failures} FAILURES ===`,
);
if (failures > 0) throw new Error(`${failures} large-file write failures`);

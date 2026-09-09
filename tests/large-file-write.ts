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
//     enough to run everywhere. Note what this cannot reach: `writeSync` never
//     returns short for a regular file, so the partial-write drain is exercised
//     by no test here. The span tiling is what real I/O can check, and it is
//     also the part an edit is likely to break.
//   - the genuine large write, only under GGUF_TRAINER_BIG_IO=1, because it
//     needs ~4.6 GB of RAM and disk and CI runners should not pay for it
//
// A regression in that last case does NOT surface as a clean assertion failure:
// the write runs away. So it runs in a child process under `ulimit -f`, which
// turns a runaway into a killed child and an ordinary FAIL line instead of a
// full disk. No `ulimit` needed from the caller.
//
// Run:  deno run -A tests/large-file-write.ts
//       GGUF_TRAINER_BIG_IO=1 deno run -A tests/large-file-write.ts

import { readFileBytes, WRITE_CHUNK_BYTES, writeFileBytes, writeSpans } from "../src/io.ts";

// node:fs and node:os rather than the Deno globals: `deno task test:node` runs
// this file too, and src/io.ts exists precisely to keep the tree runtime-neutral.
const fs = await import("node:fs");
const os = await import("node:os");
// A private directory, not counter-named files in a shared /tmp: those collide
// between concurrent runs, and a prefix sweep would try to delete other users'
// leftovers and die on EPERM.
const tmpDir = fs.mkdtempSync(`${os.tmpdir()}/gguf-trainer-io-`);
let tmpSeq = 0;
const tmpPath = () => `${tmpDir}/${tmpSeq++}.bin`;
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
// The chunk argument must actually reach writeSpans. Without this, an edit that
// drops it leaves every round trip below passing on a single span, which is the
// shape the bug had in the first place.
{
  const path = tmpPath();
  let threw = false;
  try {
    await writeFileBytes(path, new Uint8Array(8), 0);
  } catch {
    threw = true;
  } finally {
    fs.rmSync(path, { force: true });
  }
  check("writeFileBytes passes its chunk through to writeSpans", threw);
}

// A 5000-byte write at the default 1 GiB chunk is ONE span, so it would never
// execute the loop, the `off + done` arithmetic or a ragged tail. Driving the
// chunk size down is what makes the real code path reachable in milliseconds.
for (
  const [n, chunk] of [[1, 1], [7, 4], [64, 64], [4096, 512], [5000, 512]] as [number, number][]
) {
  const data = new Uint8Array(n);
  for (let i = 0; i < n; i++) data[i] = (i * 31 + 7) & 0xff;
  const path = tmpPath();
  try {
    await writeFileBytes(path, data, chunk);
    const got = await readFileBytes(path);
    let same = got.length === n;
    for (let i = 0; same && i < n; i++) if (got[i] !== data[i]) same = false;
    check(
      `round trip byte-exact across ${Math.ceil(n / chunk)} span(s) (${n} bytes, chunk ${chunk})`,
      same,
      `${got.length} bytes back`,
    );
  } finally {
    fs.rmSync(path, { force: true });
  }
}

// 3. The defect itself. Opt-in: ~4.6 GB of RAM and disk.
//
// Run inside a child process under `ulimit -f`, because a regression here does
// not fail an assertion: the write runs away, and unbounded it fills the disk.
// The child dies on SIGXFSZ instead and the parent reports an ordinary FAIL.
// `--big-child` is the child re-entering this file to do the write itself.
// Past 2^31 and 2^32, and above TinyLlama_v1.1's f32 tensors (1,100,048,384
// parameters = 4,400,193,536 bytes), which is the largest thing the readme's
// base-model table can ask this repo to write.
const BIG_BYTES = 4_401_000_000 + 4096;

async function bigWrite(): Promise<void> {
  const path = tmpPath();
  try {
    const data = new Uint8Array(BIG_BYTES);
    data[0] = 1;
    data[BIG_BYTES - 1] = 2;
    await writeFileBytes(path, data);
    const size = fs.statSync(path).size;
    check(
      `a ${(BIG_BYTES / 2 ** 30).toFixed(2)} GiB write produces exactly that many bytes`,
      size === BIG_BYTES,
      `${size}`,
    );
    const got = await readFileBytes(path);
    check(
      "and reads back with its edges intact",
      got.length === BIG_BYTES && got[0] === 1 && got[BIG_BYTES - 1] === 2,
    );
  } finally {
    fs.rmSync(path, { force: true });
  }
}

// deno-lint-ignore no-explicit-any
const argv: string[] = (globalThis as any).Deno?.args ??
  // deno-lint-ignore no-explicit-any
  ((globalThis as any).process?.argv ?? []).slice(2);
// deno-lint-ignore no-explicit-any
const isDeno = !!(globalThis as any).Deno;

if (argv.includes("--big-child")) {
  await bigWrite();
} else if (envVar("GGUF_TRAINER_BIG_IO") !== "1") {
  console.log("  ..  large-write case skipped (set GGUF_TRAINER_BIG_IO=1 to run it)");
} else if (!isDeno) {
  // Node's readFileSync throws ERR_FS_FILE_TOO_LARGE above 2^31 - 1 (measured on
  // v26.8.1, at 2.2 GB as well as 4.4 GB), so the readback would fail for a
  // reason that has nothing to do with the defect. A false FAIL is worse than a
  // skip here, because this wrapper exists so that a FAIL means the write ran
  // away. The CLI is Deno-only, so nothing real is uncovered.
  console.log("  ..  large-write case skipped on Node (readFileSync caps at 2^31)");
} else {
  const { spawnSync } = await import("node:child_process");
  const { fileURLToPath } = await import("node:url");
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  // fileURLToPath, not URL.pathname: the latter leaves percent-escapes in, so a
  // checkout under a path with a space would send the child somewhere else.
  const self = fileURLToPath(import.meta.url);
  const cmd = [g.Deno.execPath(), "run", "-A", self, "--big-child"];
  // The command travels in argv via "$@" rather than interpolated into the
  // script, so quoting never enters into it; the shell is only here for
  // `ulimit`, which is a builtin. The unit is 512-byte blocks in dash and 1024
  // in bash, so this caps a runaway at 6.1 GB or 12.3 GB depending on /bin/sh.
  // Either is far above the ~4.4 GB the real write needs and far below a disk.
  const r = spawnSync("sh", ["-c", 'ulimit -f 12000000; exec "$@"', "sh", ...cmd], {
    stdio: "inherit",
  });
  check(
    "the large-write case completed inside its file-size limit",
    r.status === 0,
    r.status === null ? `killed by ${r.signal}` : `exit ${r.status}`,
  );
}

// Unconditional: a child killed mid-write never reaches its own `finally` and
// would otherwise leave several GB behind, which is exactly what this wrapper
// is meant to prevent.
fs.rmSync(tmpDir, { recursive: true, force: true });

console.log(
  failures === 0 ? "\n=== large-file write checks passed ===" : `\n=== ${failures} FAILURES ===`,
);
if (failures > 0) throw new Error(`${failures} large-file write failures`);

// Build a varied, general-domain pretraining corpus for the FIRST *instructable*
// Gemma3 base — the blend that replaces TinyStories once the pipeline is proven.
// TinyStories teaches grammar on a toy vocabulary; it cannot support real instruct
// tuning. The mix here mirrors what open-weight models pretrain on: filtered web
// as the bulk, plus code, math, and synthetic-textbook density.
//
//   - web  (~58%)  allenai/c4 (en)                 — Colossal Cleaned Common Crawl
//   - text (~18%)  HuggingFaceTB/cosmopedia-100k   — synthetic Phi-style textbooks
//   - code (~14%)  codeparrot/github-code-clean    — transfers to general reasoning
//   - math (~10%)  open-web-math/open-web-math      — math-dense web
//
// Rationale for a web-dominant blend with code+math+synthetic supplements:
// OLMo 2 / Dolma, Qwen2.5, Phi ("Textbooks Are All You Need"), and the WSD-decay
// data-injection trick from MiniCPM / Xmodel-2. This is compute-bound on one APU
// (~0.28 st/s), so we target a few-hundred-MB SLICE, not trillions of tokens —
// Phi's quality-over-quantity lever, not scale.
//
// Documents are shuffled and concatenated with the "<|endoftext|>" marker that
// pretokenize.ts / pretrain.ts split on. Each part is kept under ~480 MB so the
// existing whole-file readers (V8 caps a single string at ~512 MB) work unchanged;
// a multi-GB corpus is built as SEVERAL parts (-p1, -p2, …), each downloaded in
// one continuous pass (shards are fetched once; leftover docs roll into the next
// part). pretokenize.ts accepts the part list and emits one token file.
//
//   deno run -A examples/prepare_pretrain.ts [targetMB=400] [outPath] [seed=1234] [parts=1]
//   deno run -A examples/prepare_pretrain.ts smoke        # tiny end-to-end self-test
//
// targetMB is PER PART. With parts=1 the output is exactly outPath; with parts>1
// the files are outPath with -p1 … -pN inserted before the extension.
//
// The frozen shared tokenizer is trained by pretokenize.ts on the first ~32 MB of
// part 1; shuffling makes that prefix a representative cross-section of the whole
// blend, so the frozen vocab covers the real-world text the instruct stage will
// need. NOTE: blending a small fraction of ChatML instruct data into the WSD
// *decay* phase (the MiniCPM/Xmodel-2 trick) is a training-wiring change in
// pretrain.ts, tracked separately — not done here.

import { fetchParquetUrls, fetchRepoDataFiles } from "../web/server/hf.ts";
import { parseDataFile, type Row } from "../web/server/parse.ts";
import { mulberry32 } from "../src/model/autograd.ts";

const DOC_SEP = "<|endoftext|>"; // boundary marker pretokenize/pretrain split on
const MIN_DOC_CHARS = 100; // drop fragments; keeps the blend substantive
const V8_STRING_CAP_MB = 480; // stay under V8's ~512 MB single-string ceiling

interface Source {
  id: string; // HF "owner/name"
  config: string; // HF config (subset)
  split: string; // usually "train"
  field: string; // text-bearing column
  weight: number; // fraction of the target size
}

// The default blend (weights sum to 1). Each is parquet-served with small enough
// shards to pull incrementally; the loop stops once a source meets its budget.
const BLEND: Source[] = [
  { id: "allenai/c4", config: "en", split: "train", field: "text", weight: 0.58 },
  {
    id: "HuggingFaceTB/cosmopedia-100k",
    config: "default",
    split: "train",
    field: "text",
    weight: 0.18,
  },
  {
    id: "codeparrot/github-code-clean",
    config: "all-mit",
    split: "train",
    field: "code",
    weight: 0.14,
  },
  {
    id: "open-web-math/open-web-math",
    config: "default",
    split: "train",
    field: "text",
    weight: 0.10,
  },
];

// Tiny sources (sub-MB shards) for the `smoke` self-test — exercises the full
// fetch -> parse -> extract -> shuffle -> stream-write -> verify path cheaply.
const SMOKE: Source[] = [
  {
    id: "bigcode/the-stack-smol-xs",
    config: "python",
    split: "train",
    field: "content",
    weight: 0.5,
  },
  { id: "bigcode/the-stack-smol-xs", config: "go", split: "train", field: "content", weight: 0.5 },
];

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("prepare_pretrain: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}

/** Pull the text field from a row, falling back to the usual text-bearing columns
 * so a mis-named field degrades gracefully rather than silently emitting nothing. */
function extractText(row: Row, field: string): string | null {
  const named = row[field];
  if (typeof named === "string" && named.length > 0) return named;
  for (const k of ["text", "content", "code", "raw", "document"]) {
    const v = row[k];
    if (typeof v === "string" && v.length > 0) return v;
  }
  return null;
}

/** Strip embedded doc markers (else they'd forge false boundaries) and trim. */
function cleanDoc(s: string): string {
  return s.replaceAll(DOC_SEP, " ").trim();
}

/** Stream cleaned docs for one source, shard by shard. Download position lives
 * in the generator, so consecutive parts continue where the last one stopped
 * and every shard is fetched exactly once. */
async function* docStream(src: Source): AsyncGenerator<string> {
  let urls = await fetchParquetUrls(src.id, src.config, src.split);
  if (!urls.length) {
    // Not auto-converted to Parquet: fall back to the repo's raw data files.
    urls = (await fetchRepoDataFiles(src.id, src.split)).map((f) => f.url);
  }
  if (!urls.length) die(`no data files for ${src.id} [${src.config}/${src.split}]`);
  for (let s = 0; s < urls.length; s++) {
    const name = urls[s].split("/").pop();
    console.log(`  [${src.id}] shard ${s + 1}/${urls.length} (${name}) …`);
    const resp = await fetch(urls[s]);
    if (!resp.ok) die(`fetch ${urls[s]} -> ${resp.status} ${resp.statusText}`);
    const bytes = new Uint8Array(await resp.arrayBuffer());
    const rows = await parseDataFile(urls[s], bytes);
    for (const row of rows) {
      const t = extractText(row, src.field);
      if (!t) continue;
      const doc = cleanDoc(t);
      if (doc.length < MIN_DOC_CHARS) continue;
      yield doc;
    }
  }
}

/** Pull docs from a source stream until the char budget is met (or it runs dry). */
async function collectPart(
  src: Source,
  gen: AsyncGenerator<string>,
  budgetChars: number,
): Promise<string[]> {
  const docs: string[] = [];
  let chars = 0;
  while (chars < budgetChars) {
    const { value, done } = await gen.next();
    if (done) {
      console.log(
        `    (${src.id} exhausted at ${(chars / 1e6).toFixed(1)}M chars, under the ${
          (budgetChars / 1e6).toFixed(0)
        }M budget)`,
      );
      break;
    }
    docs.push(value);
    chars += value.length;
  }
  console.log(
    `    [${src.id}] ${docs.length} docs, ${(chars / 1e6).toFixed(1)}M / ${
      (budgetChars / 1e6).toFixed(0)
    }M chars`,
  );
  return docs;
}

/** In-place Fisher-Yates with a seeded PRNG (deterministic across runs). */
function shuffle<T>(a: T[], seed: number): void {
  const rand = mulberry32(seed);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = a[i];
    a[i] = a[j];
    a[j] = tmp;
  }
}

/** outPath for part p: unchanged when parts==1, else -pN before the extension. */
function partPath(outPath: string, p: number, parts: number): string {
  if (parts === 1) return outPath;
  const dot = outPath.lastIndexOf(".");
  return dot < 0
    ? `${outPath}-p${p + 1}`
    : `${outPath.slice(0, dot)}-p${p + 1}${outPath.slice(dot)}`;
}

/** Stream docs to disk one at a time (never join into a single >512 MB string),
 * then verify the on-disk size and the presence of doc boundaries. */
async function writePart(path: string, docs: string[]): Promise<number> {
  const fs = await import("node:fs");
  const enc = new TextEncoder();
  const sep = enc.encode(`\n${DOC_SEP}\n`);
  const fd = fs.openSync(path, "w");
  let bytesOut = 0;
  try {
    for (const doc of docs) {
      const b = enc.encode(doc);
      fs.writeSync(fd, b);
      fs.writeSync(fd, sep);
      bytesOut += b.length + sep.length;
    }
  } finally {
    fs.closeSync(fd);
  }
  const mb = bytesOut / 1e6;
  if (mb > V8_STRING_CAP_MB) {
    console.log(
      `  WARNING: ${mb.toFixed(0)} MB exceeds the ~${V8_STRING_CAP_MB} MB whole-file read cap; ` +
        `pretrain.ts/pretokenize.ts will fail to read it whole. Lower targetMB.`,
    );
  }

  // Self-check: reopen and confirm the file is the size we wrote and carries the
  // boundary marker (so the downstream doc split will actually find documents).
  const stat = fs.statSync(path);
  if (stat.size !== bytesOut) die(`size mismatch: wrote ${bytesOut}, on disk ${stat.size}`);
  const head = new Uint8Array(Math.min(1_000_000, stat.size));
  const rfd = fs.openSync(path, "r");
  try {
    fs.readSync(rfd, head, 0, head.length, 0);
  } finally {
    fs.closeSync(rfd);
  }
  const headText = new TextDecoder().decode(head);
  if (!headText.includes(DOC_SEP)) die(`no ${DOC_SEP} marker found in ${path} head`);
  console.log(
    `  Wrote ${path}: ${mb.toFixed(1)} MB, ${docs.length} docs ` +
      `(${headText.split(DOC_SEP).length - 1} boundaries in the first MB ✓)`,
  );
  return bytesOut;
}

async function main() {
  const a = args();
  const smoke = a[0] === "smoke";
  const sources = smoke ? SMOKE : BLEND;
  const targetMB = smoke ? 1 : (a[0] ? Number(a[0]) : 400);
  const outPath = a[1] ?? (smoke ? "corpus/pretrain-smoke.txt" : "corpus/pretrain-blend.txt");
  const seed = a[2] ? Number(a[2]) : 1234;
  const parts = smoke ? 2 : (a[3] ? Number(a[3]) : 1);
  if (!Number.isFinite(targetMB) || targetMB <= 0) die(`targetMB must be > 0, got ${a[0]}`);
  if (!Number.isInteger(parts) || parts < 1) die(`parts must be a positive integer, got ${a[3]}`);

  const wsum = sources.reduce((s, x) => s + x.weight, 0);
  console.log("=== Felladrin's GGUF Trainer +∞ :: pretraining corpus blend ===\n");
  console.log(
    `Target ~${targetMB} MB x ${parts} part(s), seed ${seed}, ${sources.length} sources` +
      (smoke ? " (SMOKE self-test)" : "") + ":",
  );
  for (const s of sources) {
    console.log(`  ${(s.weight / wsum * 100).toFixed(0).padStart(3)}%  ${s.id} [${s.config}]`);
  }

  const t0 = Date.now();
  const gens = sources.map((s) => docStream(s));
  let totalBytes = 0, totalDocs = 0, totalChars = 0;
  for (let p = 0; p < parts; p++) {
    console.log(`\n--- part ${p + 1}/${parts} ---`);
    const perSource: string[][] = [];
    for (let i = 0; i < sources.length; i++) {
      const budgetChars = (targetMB * 1e6) * (sources[i].weight / wsum);
      perSource.push(await collectPart(sources[i], gens[i], budgetChars));
    }
    const docs = perSource.flat();
    if (docs.length === 0) die(`part ${p + 1} collected no documents (all sources exhausted?)`);
    totalDocs += docs.length;
    totalChars += docs.reduce((s, d) => s + d.length, 0);
    shuffle(docs, seed + p);
    totalBytes += await writePart(partPath(outPath, p, parts), docs);
  }

  const estTokens = totalChars / 4.2; // ~4.2 chars/token for byte-level BPE on English
  console.log(
    `\nTotal: ${(totalBytes / 1e6).toFixed(1)} MB across ${parts} part(s), ${totalDocs} docs, ` +
      `~${(estTokens / 1e6).toFixed(0)}M tokens est. in ${((Date.now() - t0) / 1000).toFixed(0)}s`,
  );
  console.log("\n=== corpus blend OK ===");
}

main().catch((e) => die(String(e?.stack ?? e)));

// The local Deno server behind the wizard. Serves the built React client with
// the cross-origin-isolation headers wllama needs, and exposes the API the
// wizard drives: dataset preview, training start/stop, an SSE progress stream,
// and GGUF download/serve (for wllama + the user). Training runs on Deno's
// native WebGPU here — the browser is only the control surface.
//
//   deno run -A web/server/main.ts [port]

import { mkdirSync, readdirSync, statSync } from "node:fs";
import { DEFAULT_QWEN3_CHAT_TEMPLATE, detectMapping } from "../../src/data/chat.ts";
import { readFileBytes } from "../../src/io.ts";
import { fetchFirstRows, fetchNumRows, fetchSplits, resolveDataset } from "./hf.ts";
import { parseDataFile } from "./parse.ts";
import { activeJob, createJob, getJob, requestStop, subscribe } from "./jobs.ts";
import { runJob } from "./train_job.ts";
import type { DatasetPreview, TrainConfig } from "../shared/types.ts";

const ROOT = decodeURIComponent(new URL("..", import.meta.url).pathname).replace(/\/$/, ""); // web
const CLIENT_DIST = `${ROOT}/client/dist`;
const DATA_DIR = `${ROOT}/.data`;
const MODELS_DIR = `${DATA_DIR}/models`;
const JOBS_DIR = `${DATA_DIR}/jobs`;
for (const d of [DATA_DIR, MODELS_DIR, JOBS_DIR]) mkdirSync(d, { recursive: true });

const ISOLATION_HEADERS: Record<string, string> = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "cross-origin",
};

const CONTENT_TYPES: Record<string, string> = {
  html: "text/html; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  mjs: "text/javascript; charset=utf-8",
  css: "text/css; charset=utf-8",
  json: "application/json; charset=utf-8",
  wasm: "application/wasm",
  svg: "image/svg+xml",
  png: "image/png",
  ico: "image/x-icon",
  gguf: "application/octet-stream",
  map: "application/json",
};

function withIsolation(headers: Headers): Headers {
  for (const [k, v] of Object.entries(ISOLATION_HEADERS)) headers.set(k, v);
  return headers;
}

function json(body: unknown, status = 200): Response {
  const h = withIsolation(new Headers({ "content-type": "application/json; charset=utf-8" }));
  return new Response(JSON.stringify(body), { status, headers: h });
}

// Deno's DOM lib types Response body as Uint8Array<ArrayBuffer>; our file bytes
// are Uint8Array<ArrayBufferLike>. They are byte-compatible, so cast at the edge.
function bin(bytes: Uint8Array, headers: Headers): Response {
  return new Response(bytes as unknown as BodyInit, { headers });
}

function errorJSON(message: string, status = 400): Response {
  return json({ error: message }, status);
}

async function readBody<T>(req: Request): Promise<T> {
  return (await req.json()) as T;
}

// ---- dataset preview -------------------------------------------------------

async function handlePreview(req: Request): Promise<Response> {
  const body = await readBody<{ url: string; config?: string; split?: string; hfToken?: string }>(
    req,
  );
  if (!body.url) return errorJSON("Provide a dataset URL.");
  const res = resolveDataset(body.url);

  if (res.kind === "file") {
    const r = await fetch(res.url, {
      headers: body.hfToken ? { Authorization: `Bearer ${body.hfToken}` } : {},
    });
    if (!r.ok) return errorJSON(`Could not fetch file: ${r.status} ${r.statusText}`);
    const bytes = new Uint8Array(await r.arrayBuffer());
    const rows = await parseDataFile(res.url, bytes);
    const preview: DatasetPreview = {
      dataset: res.url,
      configs: [],
      splits: [],
      config: "",
      split: "",
      features: rows.length ? Object.keys(rows[0] as object) : [],
      rows: rows.slice(0, 20),
      detected: rows.length ? detectMapping(rows[0]) : null,
      directFile: true,
    };
    return json(preview);
  }

  const info = await fetchSplits(res.id, body.hfToken);
  const config = body.config && info.byConfig[body.config] ? body.config : info.configs[0];
  const splits = info.byConfig[config] ?? [];
  const split = body.split && splits.includes(body.split)
    ? body.split
    : (splits.includes("train") ? "train" : splits[0]);
  const fr = await fetchFirstRows(res.id, config, split, body.hfToken);
  const numRowsTotal = await fetchNumRows(res.id, config, split, body.hfToken);
  const preview: DatasetPreview = {
    dataset: res.id,
    configs: info.configs,
    splits,
    config,
    split,
    features: fr.features,
    rows: fr.rows.slice(0, 20),
    detected: fr.rows.length ? detectMapping(fr.rows[0]) : null,
    numRowsTotal,
  };
  return json(preview);
}

// ---- training --------------------------------------------------------------

function handleStart(config: TrainConfig): Response {
  if (activeJob()) return errorJSON("A training job is already running.", 409);
  if (!config?.dataset?.url) return errorJSON("Missing dataset.");
  const job = createJob(config);
  const jobDir = `${JOBS_DIR}/${job.id}`;
  mkdirSync(jobDir, { recursive: true });
  // Fire and forget; progress flows over SSE.
  runJob(job, MODELS_DIR, jobDir).catch((e) => {
    console.error("job crashed:", e);
  });
  return json({ jobId: job.id });
}

function handleEvents(jobId: string, signal: AbortSignal): Response {
  const job = getJob(jobId);
  if (!job) return errorJSON("Unknown job.", 404);
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (data: string) => {
        try {
          controller.enqueue(enc.encode(data));
        } catch { /* stream closed */ }
      };
      const unsub = subscribe(job, (e) => {
        send(`data: ${JSON.stringify(e)}\n\n`);
        if (e.type === "done" || e.type === "error") {
          unsub();
          try {
            controller.close();
          } catch { /* already closed */ }
        }
      });
      // If the job already finished before we subscribed, close after replay.
      if (job.status !== "running") {
        try {
          controller.close();
        } catch { /* ignore */ }
      }
      signal.addEventListener("abort", () => {
        unsub();
        try {
          controller.close();
        } catch { /* ignore */ }
      });
    },
  });
  const h = withIsolation(
    new Headers({
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      "connection": "keep-alive",
    }),
  );
  return new Response(stream, { headers: h });
}

function listModels(): { file: string; sizeMB: number }[] {
  try {
    return readdirSync(MODELS_DIR)
      .filter((f) => f.endsWith(".gguf"))
      .map((f) => ({ file: f, sizeMB: statSync(`${MODELS_DIR}/${f}`).size / 1e6 }));
  } catch {
    return [];
  }
}

async function serveModel(file: string): Promise<Response> {
  if (file.includes("/") || file.includes("..")) return errorJSON("bad path", 400);
  try {
    const bytes = await readFileBytes(`${MODELS_DIR}/${file}`);
    const h = withIsolation(new Headers({ "content-type": "application/octet-stream" }));
    h.set("content-disposition", `attachment; filename="${file}"`);
    return bin(bytes, h);
  } catch {
    return errorJSON("model not found", 404);
  }
}

// ---- static client ---------------------------------------------------------

async function serveStatic(pathname: string): Promise<Response> {
  const rel = pathname === "/" ? "/index.html" : pathname;
  if (rel.includes("..")) return errorJSON("bad path", 400);
  const ext = rel.split(".").pop()?.toLowerCase() ?? "";
  const type = CONTENT_TYPES[ext] ?? "application/octet-stream";
  try {
    const bytes = await readFileBytes(`${CLIENT_DIST}${rel}`);
    return bin(bytes, withIsolation(new Headers({ "content-type": type })));
  } catch {
    // SPA fallback: unknown non-asset path -> index.html.
    if (!ext) {
      try {
        const idx = await readFileBytes(`${CLIENT_DIST}/index.html`);
        return bin(idx, withIsolation(new Headers({ "content-type": CONTENT_TYPES.html })));
      } catch { /* fall through */ }
    }
    return new Response(
      "Client not built. Run `deno task webui:build` (or use `deno task webui:dev`).",
      { status: 404, headers: withIsolation(new Headers({ "content-type": "text/plain" })) },
    );
  }
}

// ---- router ----------------------------------------------------------------

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const p = url.pathname;
  try {
    if (req.method === "OPTIONS") return withCorsPreflight();
    if (p === "/api/health") return json({ ok: true });
    if (p === "/api/config") return json({ defaultChatTemplate: DEFAULT_QWEN3_CHAT_TEMPLATE });
    if (p === "/api/dataset/preview" && req.method === "POST") return await handlePreview(req);
    if (p === "/api/train/start" && req.method === "POST") {
      return handleStart(await readBody<TrainConfig>(req));
    }
    if (p === "/api/train/stop" && req.method === "POST") {
      const { jobId } = await readBody<{ jobId: string }>(req);
      return json({ stopped: requestStop(jobId) });
    }
    if (p === "/api/train/events") {
      return handleEvents(url.searchParams.get("job") ?? "", req.signal);
    }
    if (p === "/api/train/status") {
      const job = getJob(url.searchParams.get("job") ?? "");
      return job ? json({ status: job.status, file: job.file }) : errorJSON("unknown job", 404);
    }
    if (p === "/api/models") return json({ models: listModels() });
    if (p.startsWith("/api/models/")) {
      return await serveModel(decodeURIComponent(p.slice("/api/models/".length)));
    }
    if (p.startsWith("/api/")) return errorJSON("not found", 404);
    return await serveStatic(p);
  } catch (e) {
    return errorJSON(e instanceof Error ? e.message : String(e), 500);
  }
}

function withCorsPreflight(): Response {
  const h = withIsolation(new Headers());
  h.set("Access-Control-Allow-Origin", "*");
  h.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  h.set("Access-Control-Allow-Headers", "content-type");
  return new Response(null, { status: 204, headers: h });
}

const port = Number((globalThis as { Deno?: { args: string[] } }).Deno?.args?.[0]) || 8787;
console.log(`\nFelladrin's GGUF Trainer — Web UI`);
console.log(`  server:  http://localhost:${port}`);
console.log(`  models:  ${MODELS_DIR}`);
console.log(`  (build the client with \`deno task webui:build\` if you see a 404 at /)\n`);
Deno.serve({ port }, handler);

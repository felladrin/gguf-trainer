// Generative exact-match eval: render each question with the chat template,
// greedy-decode with the trainer's OWN forward, extract the final integer, and
// score it against the gold answer. Complements eval_mc.ts (multiple-choice by
// NLL): that can't see free-form output, which is the only thing that reflects
// the <think> reasoning/instruct curriculum. Greedy + a fixed problem set make
// it deterministic, so successive checkpoints are directly comparable.
//
//   deno run -A --unstable-webgpu examples/eval_generative.ts <model.gguf> [--limit=N] [--max-new=N] [--cpu] [--show]
//     --limit=N   number of problems (default 20)
//     --max-new=N max new tokens per problem (default 64)
//     --cpu       force CPU forward (default: WebGPU when available)
//     --show      print each problem, the model's answer, and the gold
//
// The default task is a small arithmetic set generated in-process (no download),
// so it runs offline immediately. At 64-94M from-scratch, expect near-0% for a
// long time; the value is a comparable trend line across checkpoints, not a
// benchmark number. A harder set (GSM8K) would sit at ~0% far longer; wire one
// in only once the base is clearly competent on arithmetic.
//
// NOTE: generation has no KV cache; each token re-forwards the trailing window
// (O(n^2) per problem), so this is much heavier than eval_mc. Run it when idle.

import { readFileBytes } from "../src/io.ts";
import { loadGemma3FromGGUF } from "../src/export/load_gguf.ts";
import { mulberry32 } from "../src/model/autograd.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { greedyComplete } from "../src/eval/generate.ts";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("eval_generative: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}

/** Last integer in the text (commas stripped), or null. The verifiable reward. */
function extractLastInt(text: string): number | null {
  const m = text.match(/-?\d[\d,]*/g);
  if (!m) return null;
  const n = Number(m[m.length - 1].replace(/,/g, ""));
  return Number.isFinite(n) ? n : null;
}

/** Deterministic small-arithmetic battery (fixed seed => stable across runs). */
function arithmeticSet(n: number): { q: string; a: number }[] {
  const rng = mulberry32(0xa1_23);
  const ops = ["+", "-", "*"] as const;
  const out: { q: string; a: number }[] = [];
  for (let i = 0; i < n; i++) {
    const op = ops[Math.floor(rng() * ops.length)];
    const hi = op === "*" ? 12 : 50;
    const x = 1 + Math.floor(rng() * hi);
    const y = 1 + Math.floor(rng() * hi);
    const a = op === "+" ? x + y : op === "-" ? x - y : x * y;
    out.push({ q: `What is ${x} ${op} ${y}?`, a });
  }
  return out;
}

async function main() {
  const a = args();
  const positional = a.filter((x) => !x.startsWith("--"));
  const flag = (name: string) => {
    const f = a.find((x) => x.startsWith(`--${name}=`));
    return f ? f.slice(name.length + 3) : undefined;
  };
  const modelPath = positional[0];
  if (!modelPath) {
    die("usage: eval_generative <model.gguf> [--limit=N] [--max-new=N] [--cpu] [--show]");
  }
  const limit = flag("limit") ? Number(flag("limit")) : 20;
  const maxNew = flag("max-new") ? Number(flag("max-new")) : 64;
  const useCpu = a.includes("--cpu");
  const show = a.includes("--show");

  console.log(`=== eval_generative on ${modelPath.split("/").pop()} ===`);
  const { model, tokenizer: tok } = loadGemma3FromGGUF(await readFileBytes(modelPath));

  let gpu: WebGPUBackend | null = null;
  if (!useCpu) {
    gpu = await initWebGPU();
    if (!gpu) console.log("(no WebGPU; falling back to CPU forward)");
    else {
      console.log(`WebGPU adapter: ${gpu.adapterName}`);
      gpu.install();
      gpu.uploadParams(model.params());
    }
  }

  try {
    const problems = arithmeticSet(limit);
    let correct = 0, done = 0;
    const t0 = Date.now();
    for (const p of problems) {
      const prompt = `<|im_start|>user\n${p.q}<|im_end|>\n<|im_start|>assistant\n`;
      const promptIds = tok.encode(prompt);
      const ids = await greedyComplete(model, gpu, promptIds, maxNew, [tok.eosId]);
      const gen = tok.decode(ids.slice(promptIds.length));
      const got = extractLastInt(gen);
      const hit = got === p.a;
      if (hit) correct++;
      done++;
      if (show) console.log(`  ${hit ? "OK " : "  x"} ${p.q}  got ${got ?? "-"} (gold ${p.a})`);
      else if (done % 10 === 0) console.log(`  ${done}/${problems.length} ...`);
    }
    const pct = (100 * correct / Math.max(1, done)).toFixed(1);
    console.log(
      `\nexact-match: ${correct}/${done} = ${pct}%  (${((Date.now() - t0) / 1000).toFixed(0)}s)`,
    );
  } finally {
    if (gpu) gpu.uninstall();
  }
}

main().catch((e) => die(String(e?.stack ?? e)));

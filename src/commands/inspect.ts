// `inspect`: what is actually inside a GGUF file.
//
// Answers the questions you need before resuming a checkpoint or filing a bug:
// which architecture, which shape, which quant, does it carry a chat template,
// and what does it think its EOS token is. --json exists because the usual
// caller is a script or an agent, not a person squinting at a table.

import { readGGUF } from "../gguf/gguf.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { readFileBytes } from "../io.ts";
import { archFromGGUF } from "../model/registry.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

// The token and merge arrays are tens of thousands of entries; printing them
// buries everything else and helps nobody.
const BULKY = /^tokenizer\.ggml\.(tokens|merges|token_type|scores)$/;

async function run(v: Values) {
  const path = v.str("model");
  const bytes = await readFileBytes(path).catch(() => {
    throw new UsageError(`cannot read ${path}`);
  });
  const g = readGGUF(bytes);

  const meta: Record<string, unknown> = {};
  for (const [k, val] of g.metadata) {
    if (BULKY.test(k)) {
      meta[k] = `[${Array.isArray(val) ? val.length : "?"} entries, omitted]`;
    } else if (k === "tokenizer.chat_template") {
      meta[k] = v.bool("full") ? val : `[${String(val).length} chars, omitted]`;
    } else meta[k] = val;
  }

  let params = 0;
  for (const t of g.tensors) params += t.dims.reduce((a, b) => a * b, 1);

  if (v.bool("json")) {
    console.log(JSON.stringify(
      {
        file: path,
        bytes: bytes.length,
        ggufVersion: g.version,
        tensorCount: g.tensors.length,
        parameterCount: params,
        metadata: meta,
        tensors: v.bool("tensors")
          ? g.tensors.map((t) => ({ name: t.name, dims: t.dims, type: t.type }))
          : undefined,
      },
      null,
      2,
    ));
    return;
  }

  console.log(`${path}: GGUF v${g.version}, ${(bytes.length / 1e6).toFixed(1)} MB`);
  console.log(`${g.tensors.length} tensors, ${(params / 1e6).toFixed(1)}M parameters\n`);
  for (const [k, val] of Object.entries(meta)) console.log(`  ${k} = ${val}`);

  // A checkpoint is only resumable if its architecture flags match exactly, so
  // print the flags that rebuild this exact shape. The architecture owns that
  // list, because only it knows which of its fields are load-bearing.
  try {
    const arch = archFromGGUF(g);
    console.log(`\n${arch.describe(arch.configFromGGUF(g))}`);
    console.log(`\nResume with:\n  ${resumeFlags(g)}`);
  } catch (e) {
    console.log(`\n(cannot resume this file: ${(e as Error).message})`);
  }

  if (v.bool("tensors")) {
    console.log("");
    for (const t of g.tensors) console.log(`  ${t.name.padEnd(40)}[${t.dims.join(", ")}]`);
  }
}

/**
 * The exact flags that rebuild this checkpoint's shape, derived from its own
 * config so a new architecture gets this for free.
 */
function resumeFlags(g: GGUFFile): string {
  const arch = archFromGGUF(g);
  // deno-lint-ignore no-explicit-any
  const cfg = arch.configFromGGUF(g) as any;
  const parts = [`--arch ${arch.name}`, `--hidden ${cfg.hiddenSize}`, `--layers ${cfg.nLayers}`];
  const optional: [string, string][] = [
    ["head-dim", "headDim"],
    ["heads", "nHeads"],
    ["kv-heads", "nKVHeads"],
    ["ffn-dim", "ffnDim"],
    ["window", "slidingWindow"],
    ["swa-pattern", "swaPattern"],
    ["max-seq", "maxSeq"],
  ];
  const owned = new Set(arch.flags.map((f) => f.name));
  for (const [flag, key] of optional) {
    // Print a flag only when this architecture actually has that field and can
    // take it back: --window means nothing to an architecture without windows.
    if (cfg[key] !== undefined && (owned.has(flag) || flag === "head-dim" || flag === "max-seq")) {
      parts.push(`--${flag} ${cfg[key]}`);
    }
  }
  return parts.join(" ");
}

export const inspectCommand: Command = {
  name: "inspect",
  summary: "Print a GGUF file's metadata, shape and resume flags.",
  details: `Reads the header only, so it is instant even on a multi-GB file.

The "Resume with:" line is the point: it prints the exact architecture flags \`pretrain\`
and \`finetune\` need to continue that checkpoint. A mismatch there is the most common
reason a resume aborts.`,
  examples: [
    "inspect --model Minueza-3-95M-Base.F32.gguf",
    "inspect --model model.gguf --json",
  ],
  flags: [
    {
      name: "model",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the GGUF file to read",
    },
    { name: "json", type: "boolean", describe: "machine-readable output" },
    { name: "tensors", type: "boolean", describe: "also list every tensor name and shape" },
    {
      name: "full",
      type: "boolean",
      describe: "print the chat template in full instead of its length",
    },
  ],
  run: run,
};

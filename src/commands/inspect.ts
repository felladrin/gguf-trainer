// `inspect`: what is actually inside a GGUF file.
//
// Answers the questions you need before resuming a checkpoint or filing a bug:
// which architecture, which shape, which quant, does it carry a chat template,
// and what does it think its EOS token is. --json exists because the usual
// caller is a script or an agent, not a person squinting at a table.

import { readGGUF } from "../gguf/gguf.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { readFileBytes, writeFileBytes } from "../io.ts";
import { tokenizerFromGGUF } from "../export/load-gguf.ts";
import { CHATML_SPECIALS } from "../data/chat.ts";
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

  // The corpus commands reuse the base model's vocab verbatim (a new one would not
  // match the frozen embedding matrix), and they read it from a sibling
  // .tokenizer.json. A downloaded checkpoint carries its vocab in GGUF metadata
  // instead, so without this there is no way to get from one to the other.
  let dumped: Record<string, unknown> | undefined;
  if (v.has("dump-tokenizer")) {
    const out = v.str("dump-tokenizer");
    const t = tokenizerFromGGUF(g);
    await writeFileBytes(out, new TextEncoder().encode(JSON.stringify(t)));
    // chat-corpus refuses a vocab that cannot encode these three atomically, so
    // say now rather than after a dataset download. Same constant it checks.
    const atomic = CHATML_SPECIALS.filter((x) => t.specials?.includes(x));
    dumped = {
      path: out,
      tokens: t.tokens.length,
      merges: t.merges.length,
      specials: t.specials?.length ?? 0,
      eosId: t.eosId,
      chatml: atomic.length === CHATML_SPECIALS.length,
    };
    if (!v.bool("json")) {
      console.log(
        `wrote ${out}: ${t.tokens.length} tokens, ${t.merges.length} merges, ` +
          `${t.specials?.length ?? 0} specials, eos ${t.eosId}`,
      );
      console.log(
        atomic.length === CHATML_SPECIALS.length
          ? "ChatML present: this vocab can drive `chat-corpus`"
          : `ChatML incomplete (${atomic.join(" ") || "none"}): \`chat-corpus\` will refuse it`,
      );
    }
    // Deliberately no early return: the "Resume with:" line below is what the
    // fine-tune step needs, and a recipe that dumps the vocab wants both.
  }

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
    // The scripts that drive a resume off --json need the same flags the text
    // branch prints; the gate that consumes them refuses to guess them.
    let resume: string | undefined;
    try {
      resume = resumeFlags(g);
    } catch {
      resume = undefined;
    }
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
        dumpTokenizer: dumped,
        resume,
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
export function resumeFlags(g: GGUFFile): string {
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
    ["rms-eps", "rmsEps"],
    ["rope-base", "ropeBase"],
    ["rope-base-local", "ropeBaseLocal"],
  ];
  const owned = new Set(arch.flags.map((f) => f.name));
  for (const [flag, key] of optional) {
    // Print a flag only when this architecture actually has that field and can
    // take it back: --window means nothing to an architecture without windows.
    if (cfg[key] !== undefined && (owned.has(flag) || flag === "head-dim" || flag === "max-seq")) {
      parts.push(`--${flag} ${cfg[key]}`);
    }
  }
  // Boolean knobs print only on the side the flag sets: tied is the default, so
  // a checkpoint with its own output head is the one that needs the flag.
  if (cfg.tieEmbeddings === false && owned.has("untied-embeddings")) {
    parts.push("--untied-embeddings");
  }
  return parts.join(" ");
}

export const inspectCommand: Command = {
  name: "inspect",
  summary: "Print a GGUF file's metadata, shape and resume flags; optionally extract its vocab.",
  details:
    `The "Resume with:" line is the point: it prints the exact architecture flags \`pretrain\`
and \`finetune\` need to continue that checkpoint. A mismatch there is the most common
reason a resume aborts.

--dump-tokenizer writes that checkpoint's vocab out as a .tokenizer.json. Every corpus
command reuses the base model's vocab verbatim rather than training a new one, so this
is the step that makes a DOWNLOADED checkpoint usable: convert it to GGUF, dump its
tokenizer, then point \`chat-corpus --tokenizer\` at the result.`,
  examples: [
    "inspect --model Minueza-3-95M-Base.F32.gguf",
    "inspect --model model.gguf --json",
    "inspect --model smollm2.gguf --dump-tokenizer data/smollm2.tokenizer.json",
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
    {
      name: "dump-tokenizer",
      type: "string",
      placeholder: "PATH",
      describe:
        "write the checkpoint's vocab as a .tokenizer.json, so `chat-corpus` and `finetune` can " +
        "encode a corpus against a downloaded model",
    },
    { name: "tensors", type: "boolean", describe: "also list every tensor name and shape" },
    {
      name: "full",
      type: "boolean",
      describe: "print the chat template in full instead of its length",
    },
  ],
  run: run,
};

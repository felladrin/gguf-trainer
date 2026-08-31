// Every registered architecture, through the whole checkpoint path.
// Run:  deno run -A tests/arch-roundtrip.ts
//
// This is the test a new architecture has to pass, and it runs automatically as
// soon as the architecture is in the registry. It checks the four things that
// silently break a checkpoint:
//
//   1. paramCount() equals the model's real parameter count. A wrong count only
//      shows up as a confusing startup log, months later.
//   2. export -> read -> configFromGGUF returns the config that was written, so
//      a resume rebuilds the same shape.
//   3. export -> loadWeights -> forward reproduces the original logits BIT FOR
//      BIT at f32. Any tensor-name typo, transposed matrix or forgotten weight
//      lands here.
//   4. configMatches() rejects a shape, RoPE base or rms-eps that differs, naming the field.
//
// Quantized exports get the same treatment with a tolerance, because q8_0 is
// where a wrong block layout hides.

import { mulberry32 } from "../src/model/autograd.ts";
import { ARCHITECTURES } from "../src/model/registry.ts";
import { loadModelFromGGUF } from "../src/export/load-gguf.ts";
import { readGGUF } from "../src/gguf/gguf.ts";
import { resumeFlags } from "../src/commands/inspect.ts";
import { Values } from "../src/cli/args.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { ModelConfig } from "../src/model/arch.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  ${cond ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
  if (!cond) failures++;
}

// A real (tiny) tokenizer, so the vocab in the config matches the token list the
// exporter writes. Training it on this file's own prose is enough.
const corpus = "the quick brown fox jumps over the lazy dog. ".repeat(40) +
  "hello world, this is a tiny corpus for a tiny model. ".repeat(40);
const tok = new BPETokenizer();
tok.train(corpus, 300);

for (const arch of ARCHITECTURES) {
  console.log(`\n${arch.name}`);
  const cfg = arch.tinyConfig(tok.vocabSize);
  const model = arch.build(cfg, mulberry32(7));
  const ids = [1, 5, 2, 9, 5, 3, 8];

  // 1. the advertised parameter count is the real one
  const real = model.params().reduce((n, p) => n + p.size, 0);
  check(
    "paramCount matches the built model",
    arch.paramCount(cfg) === real,
    `declared ${arch.paramCount(cfg)}, built ${real}`,
  );

  // every param has to be in exactly one optimizer group, or it silently never trains
  const groups = model.paramGroups();
  const grouped = new Set([...groups.muon, ...groups.aux]);
  check(
    "every parameter is in exactly one optimizer group",
    grouped.size === model.params().length &&
      groups.muon.length + groups.aux.length === model.params().length,
    `${groups.muon.length} muon + ${groups.aux.length} aux vs ${model.params().length} params`,
  );

  const before = Float32Array.from(model.forward(ids).data);

  for (const [quant, tol] of [["f32", 0], ["q8_0", 0.05]] as const) {
    const bytes = arch.exportGGUF(model, tok.export(), cfg, { quant });
    const loaded = loadModelFromGGUF(bytes);

    // 2. the file says which architecture it is, and the registry finds it
    check(`${quant}: reloads as the same architecture`, loaded.arch.name === arch.name);

    // 3. the config survives the round trip (floats only to f32 precision)
    const a = loaded.cfg as unknown as Record<string, unknown>;
    const b = cfg as unknown as Record<string, unknown>;
    const mismatched = Object.keys(b).filter((k) => {
      const x = a[k], y = b[k];
      if (typeof x === "number" && typeof y === "number") {
        return Math.fround(Number(x)) !== Math.fround(y);
      }
      return x !== y;
    });
    check(
      `${quant}: config round-trips`,
      mismatched.length === 0,
      mismatched.length ? `differs: ${mismatched.join(", ")}` : "",
    );

    // 4. the weights land where the forward pass expects them
    const after = loaded.model.forward(ids).data;
    let maxDiff = 0;
    for (let i = 0; i < before.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(before[i] - after[i]));
    }
    const scale = Math.max(...Array.from(before, Math.abs)) || 1;
    check(
      `${quant}: logits survive export and reload`,
      quant === "f32" ? maxDiff === 0 : maxDiff / scale <= tol,
      `maxDiff=${maxDiff.toExponential(2)}`,
    );
  }

  // 5. a shape difference is caught, and the message names the field
  const wider = { ...(cfg as ModelConfig), nLayers: cfg.nLayers + 1 };
  // deno-lint-ignore no-explicit-any
  const mismatch = arch.configMatches(cfg as any, wider as any);
  check(
    "configMatches rejects a different shape",
    typeof mismatch === "string" && mismatch.includes("layers"),
    mismatch ?? "returned null",
  );
  check(
    "configMatches accepts an identical shape",
    // deno-lint-ignore no-explicit-any
    arch.configMatches(cfg as any, cfg as any) === null,
  );

  // 6. the RoPE base and rms-eps are part of the resume gate: a checkpoint
  //    with different values must be refused and named, not silently trained
  //    at the flag defaults.
  // deno-lint-ignore no-explicit-any
  const C = cfg as Record<string, any>;
  const withRope = { ...C, ropeBase: 12_345.6789 };
  const ropeModel = arch.build(withRope, mulberry32(8));
  const ropeBytes = arch.exportGGUF(ropeModel, tok.export(), withRope, { quant: "f32" });
  const ropeLoaded = loadModelFromGGUF(ropeBytes);
  // deno-lint-ignore no-explicit-any
  const loadedBase = (ropeLoaded.cfg as any).ropeBase;
  check(
    "non-integer rope-base survives export and still matches (f32 rounding)",
    // deno-lint-ignore no-explicit-any
    arch.configMatches(withRope as any, ropeLoaded.cfg as any) === null,
    `flag ${withRope.ropeBase} vs file ${loadedBase}`,
  );
  const wrongRope = { ...C, ropeBase: C.ropeBase * 10 };
  const ropeMismatch = arch.configMatches(C, wrongRope);
  check(
    "configMatches rejects a different rope-base, naming the flag",
    typeof ropeMismatch === "string" && ropeMismatch.includes("rope-base"),
    ropeMismatch ?? "returned null",
  );
  const intMismatch = arch.configMatches(
    { ...C, ropeBase: 1_000_000 },
    { ...C, ropeBase: 1_000_001 },
  );
  check(
    "configMatches rejects an integer rope-base difference of one",
    typeof intMismatch === "string" && intMismatch.includes("rope-base"),
    intMismatch ?? "returned null",
  );
  const ulpMismatch = arch.configMatches(
    { ...C, ropeBase: 1e8 },
    { ...C, ropeBase: 1e8 + 8 },
  );
  check(
    "configMatches still rejects one f32 ulp apart at 1e8",
    typeof ulpMismatch === "string" && ulpMismatch.includes("rope-base"),
    ulpMismatch ?? "returned null",
  );
  const epsMismatch = arch.configMatches(
    { ...C, rmsEps: 1e-5 },
    { ...C, rmsEps: 2e-5 },
  );
  check(
    "configMatches rejects a different rms-eps, naming the flag",
    typeof epsMismatch === "string" && epsMismatch.includes("rms-eps"),
    epsMismatch ?? "returned null",
  );
  // The gate names a flag when it aborts; inspect must print that flag WITH THE
  // CHECKPOINT'S VALUE next to it, or the user cannot satisfy a resume the gate
  // refuses (#35, #36). vocab is the one exception: it comes from the tokenizer,
  // not from a flag.
  const gateLine = resumeFlags(
    readGGUF(arch.exportGGUF(model, tok.export(), cfg, { quant: "f32" })),
  );
  const tokens = gateLine.split(" ");
  for (const [key, val] of Object.entries(C)) {
    if (typeof val !== "number") continue;
    // deno-lint-ignore no-explicit-any
    const named = arch.configMatches(C, { ...C, [key]: val * 2 + 1 } as any);
    if (typeof named !== "string") continue;
    const flag = named.split(":")[0];
    check(
      `inspect prints the checkpoint's value next to --${flag}`,
      flag === "vocab" ||
        Math.fround(Number(tokens[tokens.indexOf(`--${flag}`) + 1])) === Math.fround(val),
      gateLine,
    );
  }

  // GQA with a fractional group would train something llama.cpp cannot
  // reproduce; the flags refuse it before any compute.
  let gqaMsg = "";
  try {
    arch.configFromFlags(
      {
        vocabSize: C.vocabSize,
        hiddenSize: C.hiddenSize,
        nLayers: C.nLayers,
        maxSeq: C.maxSeq,
        headDim: C.headDim,
      },
      new Values(
        new Map<string, string | number | boolean>([
          ["heads", 8],
          ["kv-heads", 3],
          ["window", 64],
          ["swa-pattern", 6],
          ["rope-base-local", 10000],
        ]),
      ),
    );
  } catch (e) {
    gqaMsg = (e as Error).message;
  }
  check(
    "configFromFlags refuses --kv-heads that does not divide --heads",
    gqaMsg.includes("--kv-heads 3") && gqaMsg.includes("--heads 8"),
    gqaMsg || "did not throw",
  );

  let zeroMsg = "";
  try {
    arch.configFromFlags(
      {
        vocabSize: C.vocabSize,
        hiddenSize: C.hiddenSize,
        nLayers: C.nLayers,
        maxSeq: C.maxSeq,
        headDim: C.headDim,
      },
      new Values(
        new Map<string, string | number | boolean>([
          ["heads", 0],
          ["window", 64],
          ["swa-pattern", 6],
          ["rope-base-local", 10000],
        ]),
      ),
    );
  } catch (e) {
    zeroMsg = (e as Error).message;
  }
  check(
    "configFromFlags refuses --heads 0",
    zeroMsg.includes("--heads 0"),
    zeroMsg || "did not throw",
  );

  // The CLI's default shape (no --heads/--kv-heads) must keep whole GQA groups:
  // the guard may only fire on flags the user actually passed, never on the
  // derived default. 640 hidden gives 10 heads, where the divisor search must
  // step down (floor(10/3)=3 does not divide 10), so a stubbed search fails here.
  for (const hiddenDefault of [512, 640]) {
    const defCfg = arch.configFromFlags(
      {
        vocabSize: C.vocabSize,
        hiddenSize: hiddenDefault,
        nLayers: 2,
        maxSeq: 64,
        headDim: 64,
      },
      new Values(
        new Map<string, string | number | boolean>([
          ["window", 64],
          ["swa-pattern", 6],
          ["rope-base-local", 10000],
        ]),
      ),
    );
    check(
      `the default shape keeps whole GQA groups (hidden ${hiddenDefault})`,
      defCfg.nHeads % defCfg.nKVHeads === 0,
      `heads=${defCfg.nHeads} kv=${defCfg.nKVHeads}`,
    );
  }
  if (arch.name === "gemma3") {
    // A width that headDim*2 does not divide is unbuildable while the head
    // count is derived, and buildable once --heads names it (the 270M shape:
    // 640 hidden, 256 head-dim, 4 heads). headDim 8 keeps the block genuinely
    // wide (3 x 8 = 24 over a width of 12), so the q/o projections are too.
    const wideShape = { vocabSize: 300, hiddenSize: 12, nLayers: 2, maxSeq: 64, headDim: 8 };
    let wideMsg = "";
    try {
      arch.configFromFlags(
        wideShape,
        new Values(
          new Map<string, string | number | boolean>([
            ["window", 64],
            ["swa-pattern", 6],
            ["rope-base-local", 10000],
          ]),
        ),
      );
    } catch (e) {
      wideMsg = (e as Error).message;
    }
    check(
      "a derived head count still refuses an indivisible width, naming --heads",
      wideMsg.includes("--heads"),
      wideMsg || "did not throw",
    );
    const wide = arch.configFromFlags(
      wideShape,
      new Values(
        new Map<string, string | number | boolean>([
          ["heads", 3],
          ["kv-heads", 1],
          ["window", 64],
          ["swa-pattern", 6],
          ["rope-base-local", 10000],
        ]),
      ),
    );
    check(
      "--heads unlocks the indivisible width",
      wide.nHeads === 3 && wide.nKVHeads === 1,
      `nHeads=${wide.nHeads} nKVHeads=${wide.nKVHeads}`,
    );
    // The wide block (nHeads * headDim != hidden) must stay countable, or a
    // future gemma3 change that assumes square heads silently breaks it.
    const wideModel = arch.build(wide, mulberry32(11));
    const wideReal = wideModel.params().reduce((n, p) => n + p.size, 0);
    check(
      "paramCount matches the built wide model",
      arch.paramCount(wide) === wideReal,
      `declared ${arch.paramCount(wide)}, built ${wideReal}`,
    );
    const wrongLocal = { ...C, ropeBaseLocal: C.ropeBaseLocal * 10 };
    const localMismatch = arch.configMatches(C, wrongLocal);
    check(
      "configMatches rejects a different rope-base-local, naming the flag",
      typeof localMismatch === "string" && localMismatch.includes("rope-base-local"),
      localMismatch ?? "returned null",
    );
  }
}

console.log(
  failures === 0
    ? `\n=== all ${ARCHITECTURES.length} architectures round-trip ===`
    : `\n=== ${failures} failures ===`,
);
if (failures > 0) throw new Error(`${failures} architecture round-trip failures`);

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
//   4. configMatches() rejects a shape that differs, naming the field.
//
// Quantized exports get the same treatment with a tolerance, because q8_0 is
// where a wrong block layout hides.

import { mulberry32 } from "../src/model/autograd.ts";
import { ARCHITECTURES } from "../src/model/registry.ts";
import { loadModelFromGGUF } from "../src/export/load-gguf.ts";
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
      if (typeof y === "number" && !Number.isInteger(y)) {
        return Math.abs(Number(x) - y) > 1e-6 * Math.abs(y);
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

  // 6. the RoPE base is part of the resume gate: a checkpoint with a different
  //    base must be refused and named, not silently trained at the flag default.
  const C = cfg as Record<string, any>;
  const withRope = { ...C, ropeBase: 12_345.6789 };
  const ropeModel = arch.build(withRope, mulberry32(8));
  const ropeLoaded = loadModelFromGGUF(
    arch.exportGGUF(ropeModel, tok.export(), withRope, { quant: "f32" }),
  );
  check(
    "non-integer rope-base survives export and still matches (f32 rounding)",
    // deno-lint-ignore no-explicit-any
    arch.configMatches(withRope as any, ropeLoaded.cfg as any) === null,
  );
  const wrongRope = { ...C, ropeBase: C.ropeBase * 10 };
  const ropeMismatch = arch.configMatches(C, wrongRope);
  check(
    "configMatches rejects a different rope-base, naming the flag",
    typeof ropeMismatch === "string" && ropeMismatch.includes("rope-base"),
    ropeMismatch ?? "returned null",
  );
  if (arch.name === "gemma3") {
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

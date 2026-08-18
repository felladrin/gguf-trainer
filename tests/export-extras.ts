// Standalone assert checks for the Tier-1 export/allocation helpers:
//   - parseQuantList  (src/gguf/quantize.ts): validated, de-duplicated, ordered
//   - llamaRunScript  (src/export/export-gguf.ts): companion run script
//   - guardBufferSize (src/backend/webgpu.ts): actionable over-limit error
// These carry the non-trivial logic of the export-ergonomics + OOM-guard work;
// importing webgpu.ts here is safe (no top-level GPU access).
// Run:  deno run tests/export-extras.ts
import { parseQuantList } from "../src/gguf/quantize.ts";
import { llamaRunScript } from "../src/export/export-gguf.ts";
import { guardBufferSize } from "../src/backend/webgpu.ts";
import { gemma3Config } from "../src/arch/gemma3.ts";

function eq(got: string, want: string, msg: string): void {
  if (got !== want) throw new Error(`${msg}: got ${got}, want ${want}`);
}
function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`failed: ${msg}`);
}
function throws(fn: () => void, needle: string, msg: string): void {
  try {
    fn();
  } catch (e) {
    const m = String((e as Error).message);
    if (!m.includes(needle)) {
      throw new Error(`${msg}: threw "${m}", expected to contain "${needle}"`);
    }
    return;
  }
  throw new Error(`${msg}: did not throw`);
}

// --- parseQuantList ---
eq(
  JSON.stringify(parseQuantList("f16,q8_0,q4_0")),
  '["f16","q8_0","q4_0"]',
  "list preserves order",
);
eq(JSON.stringify(parseQuantList("q8_0, q8_0 ,f16")), '["q8_0","f16"]', "dedup + trim");
eq(JSON.stringify(parseQuantList(" Q4_0 ")), '["q4_0"]', "case-insensitive");
eq(JSON.stringify(parseQuantList("")), "[]", "empty spec");
eq(JSON.stringify(parseQuantList("f32,,f16")), '["f32","f16"]', "skips empty entries");
throws(() => parseQuantList("q3_k"), "unknown quant", "rejects unknown quant");

// --- llamaRunScript ---
const cfg = gemma3Config(256, 128, 2, 8192, 64, 1024);
const script = llamaRunScript("pretrain-base.gguf", cfg);
ok(script.includes("pretrain-base.gguf"), "run script names the model file");
ok(script.includes("-c 8192"), "run script uses the model's context length");
ok(script.includes("llama-cli"), "run script has a completion command");
ok(script.includes("llama-server"), "run script has a serve command");

// --- guardBufferSize ---
guardBufferSize(1024, 2048); // under the limit: no throw
guardBufferSize(999_999_999, 0); // unknown limit (0): guard disabled, no throw
throws(
  () => guardBufferSize(3 * 1024 * 1024, 2 * 1024 * 1024),
  "maxStorageBufferBindingSize",
  "over-limit error names the limit",
);

console.log("export_extras: all assertions passed");

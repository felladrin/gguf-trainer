// Standalone assert checks for the Tier-1 export/allocation helpers:
//   - parseQuantList  (src/gguf/quantize.ts): validated, de-duplicated, ordered
//   - llamaRunScript  (src/export/export-gguf.ts): companion run script
//   - guardBufferSize (src/backend/webgpu.ts): actionable over-limit error
//   - lossChunkValueError / lossChunkModelError (src/train/loss.ts): the checks
//     four commands share, whose entire justification is that they must not
//     drift between callers
// These carry the non-trivial logic of the export-ergonomics + OOM-guard work;
// importing webgpu.ts here is safe (no top-level GPU access).
// Run:  deno run tests/export-extras.ts
import { parseQuantList } from "../src/gguf/quantize.ts";
import { addMatrix, addVector, tensorLoader } from "../src/arch/common.ts";
import { GGUFWriter, readGGUF } from "../src/gguf/gguf.ts";
import { Tensor } from "../src/model/autograd.ts";
import { llamaRunScript } from "../src/export/export-gguf.ts";
import { guardBufferSize } from "../src/backend/webgpu.ts";
import { gemma3Config } from "../src/arch/gemma3.ts";
import { stepCheckpointPath } from "../src/commands/pretrain.ts";
import { lossChunkModelError, lossChunkValueError, MAX_LOSS_SPANS } from "../src/train/loss.ts";
import type { LanguageModel } from "../src/model/arch.ts";

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

// --keep-checkpoints names each write by its step, so a run leaves a series to pick
// between rather than one overwritten file. The `.gguf` suffix is optional in the
// pattern, so a path written without one has to gain it instead of producing a name
// nothing will load.
for (
  const [inp, step, want] of [
    ["out/lamb-rp.gguf", 200, "out/lamb-rp-step200.gguf"],
    ["out/lamb-rp", 200, "out/lamb-rp-step200.gguf"],
    ["out/v1.5/model.gguf", 0, "out/v1.5/model-step0.gguf"],
    ["model.gguf.bak", 40, "model.gguf.bak-step40.gguf"],
  ] as [string, number, string][]
) {
  eq(stepCheckpointPath(inp, step), want, `stepCheckpointPath(${inp}, ${step})`);
}

// --- --loss-chunk validation -------------------------------------------------
// Four commands take the flag, and the first version of the eval work copied
// only one of the two checks. The point of the shared pair is that a fifth
// caller cannot repeat that, so pin both here rather than in a live run.
{
  const withReadout = { forwardToReadout: () => ({}) } as unknown as LanguageModel;
  const withoutReadout = {} as unknown as LanguageModel;
  const V = 151936;

  ok(lossChunkValueError(0) === null, "loss-chunk 0 is the dense path, not an error");
  ok(lossChunkValueError(8192) === null, "a whole positive width is accepted");
  ok(`${lossChunkValueError(8192.5)}`.includes("whole number"), "a fractional width is refused");
  ok(`${lossChunkValueError(-1)}`.includes("whole number"), "a negative width is refused");

  ok(lossChunkModelError(0, V, "qwen3", withoutReadout) === null, "dense needs no readout split");
  ok(
    `${lossChunkModelError(8192, V, "qwen3", withoutReadout)}`.includes("forwardToReadout"),
    "an arch without forwardToReadout is refused, and the message names why",
  );

  // The ceiling, at the boundary in both directions. 151936/1520 is exactly
  // MAX_LOSS_SPANS, and the message's raise-to value has to be that same 1520,
  // or the two arithmetic sites disagree and the advice sends you back here.
  const exact = Math.ceil(V / MAX_LOSS_SPANS);
  ok(
    lossChunkModelError(exact, V, "qwen3", withReadout) === null,
    `${exact} is exactly at the cap`,
  );
  const over = `${lossChunkModelError(exact - 1, V, "qwen3", withReadout)}`;
  ok(over.includes(`${MAX_LOSS_SPANS + 1} spans`), "one under the cap is refused, counting spans");
  ok(over.includes(`at least ${exact}`), "and the message's raise-to value is reachable");
}

// The GGUF tensor boundary, in both directions. The destination shape comes from
// the metadata config rather than from the file, so nothing was comparing the
// two: a stored dim that disagreed loaded silently, and a tensor of the wrong
// rank was written out with an `undefined` in its ne.
{
  const mat = () => {
    const t = Tensor.zeros([3, 4]);
    for (let i = 0; i < t.data.length; i++) t.data[i] = i + 1;
    return t;
  };

  // ggml writes ne fastest-moving first, so a [3, 4] here goes out as [4, 3].
  const w = new GGUFWriter();
  w.meta_string("general.architecture", "test");
  addMatrix(w, "m.weight", mat(), "f32");
  addVector(w, "n.weight", Tensor.zeros([5]));
  const g = readGGUF(w.build());
  const load = tensorLoader(g);
  eq(
    g.tensors.find((t) => t.name === "m.weight")!.dims.join(","),
    "4,3",
    "ne is the shape reversed",
  );

  const dst = Tensor.zeros([3, 4]);
  load("m.weight", dst);
  eq(Array.from(dst.data.slice(0, 3)).join(","), "1,2,3", "a matching tensor still loads");
  load("n.weight", Tensor.zeros([5]));

  // The transpose is the case with no other symptom: same element count, so
  // dequantize fills the buffer and the weight is merely scrambled.
  throws(
    () => load("m.weight", Tensor.zeros([4, 3])),
    "has dims [4, 3]",
    "a transposed destination is refused",
  );
  throws(
    () => load("m.weight", Tensor.zeros([2, 6])),
    "has dims [4, 3]",
    "a reshaped destination is refused",
  );
  throws(
    () => load("m.weight", Tensor.zeros([12])),
    "has dims [4, 3]",
    "a flattened destination is refused",
  );
  // The rank comparison earns its place: [4] reversed is a PREFIX of [4, 3], so
  // an element-wise check alone accepts it and dequantize happily returns the
  // first four values.
  throws(
    () => load("m.weight", Tensor.zeros([4])),
    "has dims [4, 3]",
    "a destination whose shape is a prefix of the dims is refused",
  );
  throws(
    () => load("n.weight", Tensor.zeros([4])),
    "has dims [5]",
    "a short vector destination is refused",
  );

  // The writer side. A 1-D tensor left inDim undefined, so `inDim % 32 !== 0`
  // was NaN !== 0 and the quant silently became f16, with [undefined, outDim]
  // written as the ne.
  throws(
    () => addMatrix(new GGUFWriter(), "bad", Tensor.zeros([12]), "q8_0"),
    "addMatrix: bad must be 2-D, got [12]",
    "a 1-D tensor is refused by addMatrix",
  );
  throws(
    () => addMatrix(new GGUFWriter(), "bad", Tensor.zeros([2, 3, 4]), "f32"),
    "must be 2-D",
    "a 3-D tensor is refused by addMatrix",
  );
  throws(
    () => addVector(new GGUFWriter(), "bad", Tensor.zeros([3, 4])),
    "addVector: bad must be 1-D, got [3, 4]",
    "a 2-D tensor is refused by addVector",
  );
}

console.log("export_extras: all assertions passed");

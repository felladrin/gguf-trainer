// Shared building blocks for architecture files, so each one only has to
// express what is different about it.
//
// If you find yourself copying more than a few lines between two arch files,
// the shared part probably belongs here.

import { GGUFWriter } from "../gguf/gguf.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { dequantize, serialize } from "../gguf/quantize.ts";
import type { QuantName } from "../gguf/quantize.ts";
import { Tensor } from "../model/autograd.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import type { ExportOpts } from "../model/arch.ts";

// llama.cpp token types.
const TOKEN_TYPE_NORMAL = 1;
const TOKEN_TYPE_CONTROL = 3;
const TOKEN_TYPE_USER_DEFINED = 4;

/** A norm weight: trains in gain-frame, so it starts at 1 and exports directly. */
export function ones(shape: number[]): Tensor {
  const t = Tensor.zeros(shape, true);
  t.data.fill(1);
  return t;
}

/**
 * Weight matrices are [out, in] row-major here, so ggml's ne is [in, out].
 * Block quantization runs along ne[0]; when that dimension is not a multiple of
 * 32 the tensor falls back to f16 rather than silently mis-encoding.
 */
export function addMatrix(w: GGUFWriter, name: string, t: Tensor, quant: QuantName) {
  const [outDim, inDim] = t.shape;
  let q = quant;
  if ((quant === "q8_0" || quant === "q4_0") && inDim % 32 !== 0) q = "f16";
  w.addTensor(name, [inDim, outDim], serialize(t.data, q));
}

/** Norm and other 1-D tensors stay f32; llama.cpp expects that. */
export function addVector(w: GGUFWriter, name: string, t: Tensor) {
  w.addTensor(name, [t.shape[0]], serialize(t.data, "f32"));
}

/**
 * Per-token ggml token_type. Two kinds of atomic special are distinguished by
 * shape, because llama.cpp renders them differently:
 *   - CONTROL (3): the `<|...|>` turn and stop tokens. Handled by the chat
 *     machinery and never emitted as visible text.
 *   - USER_DEFINED (4): the visible `<...>` reasoning and tool tags, which must
 *     survive into the output for llama.cpp's parsers to see them.
 *   - NORMAL (1): everything else.
 */
function tokenTypes(tok: TokenizerData): number[] {
  const special = new Set(tok.specials ?? []);
  return tok.tokens.map((t) => {
    if (t.startsWith("<|") && t.endsWith("|>")) return TOKEN_TYPE_CONTROL;
    if (special.has(t)) return TOKEN_TYPE_USER_DEFINED;
    return TOKEN_TYPE_NORMAL;
  });
}

// ggml LLAMA_FTYPE values, reported in general.file_type.
function fileType(q: QuantName): number {
  switch (q) {
    case "f32":
      return 0;
    case "f16":
      return 1;
    case "q4_0":
      return 2;
    case "q8_0":
      return 7;
  }
}

/**
 * Open a GGUF and write everything that is the same for every architecture:
 * the general.* block and the whole tokenizer. The caller then writes its own
 * `<arch>.*` metadata and its tensors.
 *
 * The tokenizer is always exported as the "gpt2" model with the "qwen2"
 * pre-tokenizer, because that is what this project's byte-level BPE is,
 * whatever the model architecture around it.
 */
export function startGGUF(arch: string, tok: TokenizerData, opts: ExportOpts): GGUFWriter {
  const w = new GGUFWriter();
  w.meta_string("general.architecture", arch);
  w.meta_string("general.name", opts.name ?? `gguf-trainer-${arch}`);
  w.meta_u32("general.file_type", fileType(opts.quant));

  w.meta_string("tokenizer.ggml.model", "gpt2");
  w.meta_string("tokenizer.ggml.pre", "qwen2");
  w.meta_arr_str("tokenizer.ggml.tokens", tok.tokens);
  w.meta_arr_i32("tokenizer.ggml.token_type", tokenTypes(tok));
  w.meta_arr_str("tokenizer.ggml.merges", tok.merges);
  w.meta_u32("tokenizer.ggml.bos_token_id", tok.bosId);
  w.meta_u32("tokenizer.ggml.eos_token_id", tok.eosId);
  w.meta_bool("tokenizer.ggml.add_bos_token", false);
  w.meta_bool("tokenizer.ggml.add_eos_token", false);
  if (opts.chatTemplate) w.meta_string("tokenizer.chat_template", opts.chatTemplate);
  return w;
}

/** Read a numeric metadata key, failing with the key name rather than undefined. */
export function metaNum(g: GGUFFile, key: string): number {
  const v = g.metadata.get(key);
  if (typeof v !== "number") throw new Error(`GGUF missing numeric metadata "${key}"`);
  return v;
}

/** Read a numeric metadata key, or fall back when the writer omitted it. */
export function metaNumOr(g: GGUFFile, key: string, fallback: number): number {
  const v = g.metadata.get(key);
  return typeof v === "number" ? v : fallback;
}

/**
 * A by-name tensor loader for one GGUF. Each tensor is dequantized per its own
 * stored type, because an export may fall back to f16 for a single matrix whose
 * inner dimension is not a multiple of 32.
 */
export function tensorLoader(g: GGUFFile): (name: string, dst: Tensor) => void {
  const byName = new Map(g.tensors.map((t) => [t.name, t]));
  return (name, dst) => {
    const t = byName.get(name);
    if (!t) throw new Error(`GGUF missing tensor "${name}"`);
    const de = dequantize(t.type, t.data, dst.size);
    dst.data.set(de);
  };
}

/**
 * The generic shape check behind every `configMatches`: compare the listed
 * fields and report the first difference in the terms the CLI flags use.
 */
export function diffFields<C>(
  built: C,
  checkpoint: C,
  fields: { key: keyof C; flag: string }[],
): string | null {
  for (const { key, flag } of fields) {
    if (built[key] !== checkpoint[key]) {
      return `${flag}: built ${String(built[key])} vs checkpoint ${String(checkpoint[key])}`;
    }
  }
  return null;
}

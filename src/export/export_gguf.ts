// Serialize a trained Qwen3Model into a llama.cpp-loadable GGUF file.
// Tensor names and metadata keys follow llama.cpp's "qwen3" architecture.

import { GGUFWriter } from "../gguf/gguf.ts";
import { serialize } from "../gguf/quantize.ts";
import type { QuantName } from "../gguf/quantize.ts";
import type { Tensor } from "../model/autograd.ts";
import type { Qwen3Config } from "../model/config.ts";
import type { Qwen3Model } from "../model/qwen3.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";

// llama.cpp token types
const TOKEN_TYPE_NORMAL = 1;
const TOKEN_TYPE_CONTROL = 3;

/**
 * Weight matrices are stored [out, in] row-major in our Tensors, so ggml's
 * ne = [in, out]. Block quant runs along ne[0] (in), which must be a multiple
 * of 32; otherwise we fall back to f16 for that tensor. Norm/1-D tensors stay
 * F32, matching llama.cpp expectations.
 */
function addMatrix(w: GGUFWriter, name: string, t: Tensor, quant: QuantName) {
  const [outDim, inDim] = t.shape;
  let q = quant;
  if ((quant === "q8_0" || quant === "q4_0") && inDim % 32 !== 0) q = "f16";
  w.addTensor(name, [inDim, outDim], serialize(t.data, q));
}

function addVector(w: GGUFWriter, name: string, t: Tensor) {
  // Norm weights: keep full precision.
  w.addTensor(name, [t.shape[0]], serialize(t.data, "f32"));
}

export function buildGGUF(
  model: Qwen3Model,
  tok: TokenizerData,
  cfg: Qwen3Config,
  opts: { quant: QuantName; name?: string; chatTemplate?: string } = { quant: "f16" },
): Uint8Array {
  const w = new GGUFWriter();
  const arch = "qwen3";

  // ---- general.* ----
  w.meta_string("general.architecture", arch);
  w.meta_string("general.name", opts.name ?? "gguf-trainer-qwen3");
  w.meta_u32("general.file_type", fileType(opts.quant));

  // ---- qwen3.* hyperparameters ----
  w.meta_u32(`${arch}.context_length`, cfg.maxSeq);
  w.meta_u32(`${arch}.embedding_length`, cfg.hiddenSize);
  w.meta_u32(`${arch}.block_count`, cfg.nLayers);
  w.meta_u32(`${arch}.feed_forward_length`, cfg.ffnDim);
  w.meta_u32(`${arch}.attention.head_count`, cfg.nHeads);
  w.meta_u32(`${arch}.attention.head_count_kv`, cfg.nKVHeads);
  w.meta_u32(`${arch}.attention.key_length`, cfg.headDim);
  w.meta_u32(`${arch}.attention.value_length`, cfg.headDim);
  w.meta_f32(`${arch}.attention.layer_norm_rms_epsilon`, cfg.rmsEps);
  w.meta_f32(`${arch}.rope.freq_base`, cfg.ropeBase);
  w.meta_u32(`${arch}.vocab_size`, cfg.vocabSize);

  // ---- tokenizer.* ----
  w.meta_string("tokenizer.ggml.model", "gpt2");
  w.meta_string("tokenizer.ggml.pre", "qwen2");
  w.meta_arr_str("tokenizer.ggml.tokens", tok.tokens);
  w.meta_arr_i32(
    "tokenizer.ggml.token_type",
    tok.tokens.map((
      t,
    ) => (t.startsWith("<|") && t.endsWith("|>") ? TOKEN_TYPE_CONTROL : TOKEN_TYPE_NORMAL)),
  );
  w.meta_arr_str("tokenizer.ggml.merges", tok.merges);
  w.meta_u32("tokenizer.ggml.bos_token_id", tok.bosId);
  w.meta_u32("tokenizer.ggml.eos_token_id", tok.eosId);
  w.meta_bool("tokenizer.ggml.add_bos_token", false);
  w.meta_bool("tokenizer.ggml.add_eos_token", false);
  // Chat template (Jinja): llama.cpp/wllama read this to format turns at
  // inference. Only written for chat-format models; base models omit it.
  if (opts.chatTemplate) w.meta_string("tokenizer.chat_template", opts.chatTemplate);

  // ---- tensors ----
  const q = opts.quant;
  addMatrix(w, "token_embd.weight", model.tokenEmbd, q);
  addVector(w, "output_norm.weight", model.outputNorm);
  if (model.output) addMatrix(w, "output.weight", model.output, q);

  model.layers.forEach((L, i) => {
    const p = `blk.${i}`;
    addVector(w, `${p}.attn_norm.weight`, L.attnNorm);
    addMatrix(w, `${p}.attn_q.weight`, L.qProj, q);
    addMatrix(w, `${p}.attn_k.weight`, L.kProj, q);
    addMatrix(w, `${p}.attn_v.weight`, L.vProj, q);
    addMatrix(w, `${p}.attn_output.weight`, L.oProj, q);
    addVector(w, `${p}.attn_q_norm.weight`, L.qNorm);
    addVector(w, `${p}.attn_k_norm.weight`, L.kNorm);
    addVector(w, `${p}.ffn_norm.weight`, L.ffnNorm);
    addMatrix(w, `${p}.ffn_gate.weight`, L.gate, q);
    addMatrix(w, `${p}.ffn_up.weight`, L.up, q);
    addMatrix(w, `${p}.ffn_down.weight`, L.down, q);
  });

  return w.build();
}

// ggml LLAMA_FTYPE values used for reporting.
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

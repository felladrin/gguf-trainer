// Serialize a trained Gemma3Model into a llama.cpp-loadable GGUF file.
// Tensor names and metadata keys follow llama.cpp's "gemma3" architecture.

import { GGUFWriter } from "../gguf/gguf.ts";
import { serialize } from "../gguf/quantize.ts";
import type { QuantName } from "../gguf/quantize.ts";
import type { Tensor } from "../model/autograd.ts";
import type { Gemma3Config } from "../model/config.ts";
import type { Gemma3Model } from "../model/gemma3.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";

// llama.cpp token types
const TOKEN_TYPE_NORMAL = 1;
const TOKEN_TYPE_CONTROL = 3;
const TOKEN_TYPE_USER_DEFINED = 4;

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

/**
 * Per-token ggml token_type. Two kinds of atomic special are distinguished by
 * shape, because llama.cpp renders them differently:
 *   - CONTROL (3): the `<|...|>` turn/stop tokens (ChatML `<|im_start|>` /
 *     `<|im_end|>` / `<|endoftext|>`). Handled by the chat machinery and NOT
 *     emitted as visible text.
 *   - USER_DEFINED (4): any other declared special (the `<...>` reasoning and
 *     tool tags — `<think>`, `<tool_call>`, …). Atomic in tokenization but kept
 *     VISIBLE in output, which is required for llama.cpp's `--jinja` tool parser
 *     and reasoning parser to see them in the generated text.
 *   - NORMAL (1): everything else.
 * Marking a visible tag CONTROL would let llama.cpp suppress it from output and
 * break parsing, so the split matters. token_type is metadata (not weights), so
 * it round-trips independently of the frozen embeddings.
 */
function tokenTypes(tok: TokenizerData): number[] {
  const special = new Set(tok.specials ?? []);
  return tok.tokens.map((t) => {
    if (t.startsWith("<|") && t.endsWith("|>")) return TOKEN_TYPE_CONTROL;
    if (special.has(t)) return TOKEN_TYPE_USER_DEFINED;
    return TOKEN_TYPE_NORMAL;
  });
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

/**
 * Serialize a trained Gemma3Model into a llama.cpp-loadable GGUF (arch
 * "gemma3"). Tensor names and metadata keys follow llama.cpp's gemma3 arch
 * (verified against master / build 9850): sandwich norms
 * (post_attention_norm / post_ffw_norm), per-head QK-norm (attn_q_norm /
 * attn_k_norm, length head_dim), tied output (output.weight omitted), the SWA
 * keys (sliding_window + sliding_window_pattern), and the two RoPE bases
 * (rope.freq_base global + rope.freq_base_swa local). Embeddings are exported
 * raw — llama.cpp applies the sqrt(hidden) input scale at runtime. Norm weights
 * are exported directly (gain-frame; llama.cpp's build_norm is a plain
 * rms_norm·w, so no HF-style +1 is added here).
 */
export function buildGemma3GGUF(
  model: Gemma3Model,
  tok: TokenizerData,
  cfg: Gemma3Config,
  opts: { quant: QuantName; name?: string; chatTemplate?: string } = { quant: "f16" },
): Uint8Array {
  const w = new GGUFWriter();
  const arch = "gemma3";

  w.meta_string("general.architecture", arch);
  w.meta_string("general.name", opts.name ?? "gguf-trainer-gemma3");
  w.meta_u32("general.file_type", fileType(opts.quant));

  w.meta_u32(`${arch}.context_length`, cfg.maxSeq);
  w.meta_u32(`${arch}.embedding_length`, cfg.hiddenSize);
  w.meta_u32(`${arch}.block_count`, cfg.nLayers);
  w.meta_u32(`${arch}.feed_forward_length`, cfg.ffnDim);
  w.meta_u32(`${arch}.attention.head_count`, cfg.nHeads);
  w.meta_u32(`${arch}.attention.head_count_kv`, cfg.nKVHeads);
  w.meta_u32(`${arch}.attention.key_length`, cfg.headDim);
  w.meta_u32(`${arch}.attention.value_length`, cfg.headDim);
  w.meta_f32(`${arch}.attention.layer_norm_rms_epsilon`, cfg.rmsEps);
  w.meta_u32(`${arch}.attention.sliding_window`, cfg.slidingWindow);
  w.meta_u32(`${arch}.attention.sliding_window_pattern`, cfg.swaPattern);
  w.meta_f32(`${arch}.rope.freq_base`, cfg.ropeBase);
  w.meta_f32(`${arch}.rope.freq_base_swa`, cfg.ropeBaseLocal);
  w.meta_u32(`${arch}.vocab_size`, cfg.vocabSize);

  // Our tokenizer is a byte-level BPE split by the GPT-2/Qwen2 regex, so it
  // exports as the "gpt2" model with the "qwen2" pre-tokenizer (matches our
  // splitting at inference) regardless of the gemma3 model arch.
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
    addVector(w, `${p}.attn_q_norm.weight`, L.qNorm);
    addVector(w, `${p}.attn_k_norm.weight`, L.kNorm);
    addMatrix(w, `${p}.attn_output.weight`, L.oProj, q);
    addVector(w, `${p}.post_attention_norm.weight`, L.postAttnNorm);
    addVector(w, `${p}.ffn_norm.weight`, L.ffnNorm);
    addMatrix(w, `${p}.ffn_gate.weight`, L.gate, q);
    addMatrix(w, `${p}.ffn_up.weight`, L.up, q);
    addMatrix(w, `${p}.ffn_down.weight`, L.down, q);
    addVector(w, `${p}.post_ffw_norm.weight`, L.postFfnNorm);
  });

  return w.build();
}

/**
 * A copy-paste-ready bash script for running an exported GGUF with llama.cpp,
 * written alongside the model so the step after a run is obvious. Uses the
 * model's own context length; `$1` overrides the model path for the variants.
 * Base models carry no chat template, so it drives raw completion.
 */
export function llamaRunScript(ggufFile: string, cfg: Gemma3Config): string {
  return `#!/usr/bin/env bash
# Auto-generated by gguf-trainer. Run the exported model with llama.cpp.
# Requires llama.cpp built for your GPU (Vulkan on Strix Halo). Pass a different
# .gguf as the first argument to run one of the quantized variants.
set -euo pipefail
MODEL="\${1:-${ggufFile}}"

# One-shot completion (base model: no chat template, raw next-token):
llama-cli -m "$MODEL" -c ${cfg.maxSeq} -n 128 -p "Once upon a time"

# Or serve an OpenAI-compatible endpoint:
# llama-server -m "$MODEL" -c ${cfg.maxSeq} --host 127.0.0.1 --port 8080
`;
}

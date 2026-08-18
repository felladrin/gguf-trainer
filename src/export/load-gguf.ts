// Reading a checkpoint back: the architecture-agnostic half.
//
// Which tensors exist and what they are called is the architecture's business
// (src/arch/*.ts). What lives here is what every checkpoint shares: the
// tokenizer, and the dispatch that finds the right architecture for a file.

import { readGGUF } from "../gguf/gguf.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { BPETokenizer } from "../tokenizer/bpe.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import { mulberry32 } from "../model/autograd.ts";
import type { Architecture, LanguageModel, ModelConfig } from "../model/arch.ts";
import { archFromGGUF } from "../model/registry.ts";
import { metaNum } from "../arch/common.ts";

// llama.cpp: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4
const TOKEN_TYPE_CONTROL = 3;
const TOKEN_TYPE_USER_DEFINED = 4;

/** Reconstruct the tokenizer from the GGUF's tokenizer.ggml.* metadata. */
export function tokenizerFromGGUF(g: GGUFFile): TokenizerData {
  const tokens = g.metadata.get("tokenizer.ggml.tokens");
  const merges = g.metadata.get("tokenizer.ggml.merges");
  if (!Array.isArray(tokens) || !Array.isArray(merges)) {
    throw new Error("GGUF missing tokenizer.ggml.tokens/merges");
  }
  const toks = tokens as string[];
  // Recover the atomic specials so they re-encode as single ids. Both CONTROL
  // (turn and stop tokens) and USER_DEFINED (visible reasoning and tool tags)
  // are atomic; the exporter splits them by shape. Fall back to the "<|...|>"
  // shape when a GGUF omits token_type. Without this, a resumed chat model would
  // shred its special tokens back into bytes.
  const types = g.metadata.get("tokenizer.ggml.token_type");
  let specials: string[] = Array.isArray(types)
    ? toks.filter((_, i) => {
      const ty = (types as number[])[i];
      return ty === TOKEN_TYPE_CONTROL || ty === TOKEN_TYPE_USER_DEFINED;
    })
    : [];
  if (specials.length === 0) specials = toks.filter((t) => /^<\|.*\|>$/.test(t));
  return {
    tokens: toks,
    merges: merges as string[],
    bosId: metaNum(g, "tokenizer.ggml.bos_token_id"),
    eosId: metaNum(g, "tokenizer.ggml.eos_token_id"),
    specials: specials.length ? specials : undefined,
  };
}

/**
 * Full checkpoint resume, whatever the architecture: bytes to a model with its
 * trained weights, its config, its tokenizer, and the architecture itself (the
 * caller needs it to export again, or to check a resume is compatible).
 */
export function loadModelFromGGUF(bytes: Uint8Array): {
  model: LanguageModel;
  cfg: ModelConfig;
  tokenizer: BPETokenizer;
  // deno-lint-ignore no-explicit-any
  arch: Architecture<any>;
} {
  const g = readGGUF(bytes);
  const arch = archFromGGUF(g);
  const cfg = arch.configFromGGUF(g);
  const model = arch.build(cfg, mulberry32(0)); // every weight is overwritten below
  arch.loadWeights(model, g);
  const tokenizer = BPETokenizer.fromData(tokenizerFromGGUF(g));
  return { model, cfg, tokenizer, arch };
}

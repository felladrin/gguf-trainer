// The architecture contract.
//
// Everything outside `src/arch/` is architecture-agnostic: the autograd ops, the
// WebGPU kernels, the optimizers, the trainer loop, the GGUF reader and writer,
// the CLI. An architecture is the one thing they do not know, so it is the one
// thing a contributor writes: a single file that says how the ops compose, what
// the config fields are, and which names the tensors take in GGUF.
//
// Read docs/adding-an-architecture.md before implementing one. The short version
// is that you implement this interface, register it in src/model/registry.ts,
// and the CLI, the tests and the training loop pick it up with no other change.

import type { Tensor } from "./autograd.ts";
import type { MuPOpts } from "./mup.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import type { QuantName } from "../gguf/quantize.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import type { Flag, Values } from "../cli/args.ts";

/**
 * The config fields every architecture must expose, because the trainer, the
 * CLI and the exporter read them. An architecture's own config extends this
 * with whatever else it needs (sliding windows, extra RoPE bases, conv kernel
 * sizes) and nothing outside the arch file may read those extras.
 */
export interface ModelConfig {
  arch: string; // registry key, and general.architecture in the GGUF
  vocabSize: number;
  hiddenSize: number;
  nLayers: number;
  maxSeq: number; // context_length
  tieEmbeddings: boolean;
}

/**
 * What the trainer needs from a model. Any structure is fine behind it: the
 * training loop only ever calls these three.
 */
export interface LanguageModel {
  readonly cfg: ModelConfig;
  /** Every trainable tensor, in a stable order. */
  params(): Tensor[];
  /** 2-D hidden matmuls go to Muon, everything else (embeddings, norms) to AdamW. */
  paramGroups(): { muon: Tensor[]; aux: Tensor[] };
  /** One token sequence to logits [T, vocabSize]. */
  forward(ids: number[]): Tensor;

  /**
   * Optional capability: architectures with QK-RMSNorm expose their per-layer
   * (qNorm, kNorm) pairs and head dimension so MuonClip can bound attention
   * logits. Leave it off and MuonClip simply does not apply to this model.
   */
  qkNorms?(): { headDim: number; pairs: { qNorm: Tensor; kNorm: Tensor }[] };
}

/** The shape flags the CLI collects for every architecture. */
export interface BaseShape {
  vocabSize: number;
  hiddenSize: number;
  nLayers: number;
  maxSeq: number;
  headDim: number;
}

export interface ExportOpts {
  quant: QuantName;
  name?: string;
  chatTemplate?: string;
}

export interface Architecture<C extends ModelConfig = ModelConfig> {
  /** Registry key, `--arch <name>`, and the GGUF's general.architecture. */
  name: string;
  /** One line, shown by `archs` and in `--help`. */
  summary: string;
  /** Models this arch is shape-compatible with, for the `archs` listing. */
  reference?: string;

  /**
   * Flags this architecture adds to `pretrain` and `finetune`. Keep names shared
   * with other architectures when the meaning is the same (`--rope-base`), and
   * prefix the describe text with the arch name when it is not.
   */
  flags: Flag[];

  /** Build a config from the shared shape flags plus this arch's own. */
  configFromFlags(shape: BaseShape, v: Values): C;

  /** Rebuild the config from a GGUF written by this architecture. */
  configFromGGUF(g: GGUFFile): C;

  /**
   * The smallest config that still exercises every path in `forward`. The test
   * suite builds one of these for each registered architecture, so keep it tiny
   * (a couple of layers, a handful of channels) but do not simplify away a
   * feature: for Gemma3 that means one sliding-window layer AND one dense layer.
   */
  tinyConfig(vocabSize: number): C;

  /** Instantiate the model. `rng` seeds init; `mup` scales embedding init when set. */
  build(cfg: C, rng: () => number, mup?: MuPOpts): LanguageModel;

  /** Exact trainable parameter count, used for the startup log and sizing. */
  paramCount(cfg: C): number;

  /** The one-line architecture summary printed when a run starts. */
  describe(cfg: C): string;

  /** Serialize to a llama.cpp-loadable GGUF. */
  exportGGUF(model: LanguageModel, tok: TokenizerData, cfg: C, opts: ExportOpts): Uint8Array;

  /** Copy weights from a GGUF into a model of the same shape. */
  loadWeights(model: LanguageModel, g: GGUFFile): void;

  /**
   * Null when a checkpoint can be resumed into a model built from `built`,
   * otherwise the field that differs, phrased for a human: resuming with the
   * wrong shape must fail before any compute, naming what to change.
   */
  configMatches(built: C, checkpoint: C): string | null;
}

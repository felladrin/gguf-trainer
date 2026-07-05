// Wizard <-> server contract. Imported by both the Deno server and the React
// client, so the shape the wizard collects is exactly what the trainer runs.

import type { FieldMapping, ModelType } from "../../src/data/chat.ts";
import type { QuantName } from "../../src/gguf/quantize.ts";

export type { FieldMapping, ModelType, QuantName };

/** Everything a training run needs, collected across the wizard steps. */
export interface TrainConfig {
  name: string;
  modelType: ModelType;
  dataset: {
    url: string; // HF dataset URL or a direct file URL
    config?: string; // HF dataset config (subset) name
    split?: string; // e.g. "train"
    mapping: FieldMapping; // how to read the columns
    maxRows?: number; // cap rows pulled for training (0/undefined = as needed)
    hfToken?: string; // for gated/private datasets
  };
  vocabSize: number;
  chatTemplate?: string; // Jinja; present for chat/instruct/reasoning/tools
  model: {
    hidden: number;
    layers: number;
    baseWidth: number; // muP proxy width (== hidden disables muP)
    maxSeq: number; // context length
  };
  training: {
    steps: number;
    seqLen: number;
    batch: number;
    muonLr: number;
    auxLr: number;
    quant: QuantName;
    maskPromptLoss?: boolean; // chat: compute loss only on assistant turns (default true)
  };
  resumeFrom?: string; // server-side filename of a prior GGUF to continue
}

/** A dataset preview the wizard renders (from the HF Datasets Server API). */
export interface DatasetPreview {
  dataset: string; // resolved "owner/name"
  configs: string[];
  splits: string[];
  config: string;
  split: string;
  features: string[]; // column names
  rows: unknown[]; // sample rows (<= 20)
  detected: FieldMapping | null;
  numRowsTotal?: number;
  directFile?: boolean; // true when the URL was a direct data file, not a repo
}

/** Progress events streamed over SSE during a run. */
export type TrainEvent =
  | { type: "status"; phase: string; message: string }
  | { type: "corpus"; tokens: number; vocab: number; epochs: number; docs: number }
  | { type: "model"; params: number; summary: string }
  | { type: "step"; step: number; steps: number; loss: number; stepsPerSec: number }
  | { type: "sample"; step: number; text: string }
  | { type: "done"; file: string; sizeMB: number; tensors: number; sample: string }
  | { type: "error"; message: string };

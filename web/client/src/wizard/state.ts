import type {
  DatasetPreview,
  FieldMapping,
  ModelType,
  QuantName,
  TrainConfig,
} from "../../../shared/types.ts";

export interface State {
  step: number; // 1..6
  goal: "new" | "continue";
  resumeFrom: string;
  modelType: ModelType;
  name: string;

  datasetUrl: string;
  preview?: DatasetPreview;
  mapping?: FieldMapping;
  dsConfig?: string;
  dsSplit?: string;
  hfToken: string;
  maxRows: number;

  vocabSize: number;
  chatTemplate: string;

  hidden: number;
  layers: number;
  baseWidth: number;
  maxSeq: number;

  steps: number;
  seqLen: number;
  batch: number;
  muonLr: number;
  auxLr: number;
  quant: QuantName;
  maskPromptLoss: boolean; // chat: train loss only on assistant turns

  jobId?: string;
  models: { file: string; sizeMB: number }[];
  doneFile?: string;
}

export function defaultState(chatTemplate: string): State {
  return {
    step: 1,
    goal: "new",
    resumeFrom: "",
    modelType: "base",
    name: "",
    datasetUrl: "",
    hfToken: "",
    maxRows: 0,
    vocabSize: 8192,
    chatTemplate,
    hidden: 384,
    layers: 6,
    baseWidth: 128,
    maxSeq: 512,
    steps: 1000,
    seqLen: 256,
    batch: 16,
    muonLr: 0.02,
    auxLr: 0.003,
    quant: "f16",
    maskPromptLoss: true,
    models: [],
  };
}

export function buildConfig(s: State): TrainConfig {
  return {
    name: s.name || `qwen3-${s.modelType}`,
    modelType: s.modelType,
    dataset: {
      url: s.datasetUrl,
      config: s.dsConfig,
      split: s.dsSplit,
      mapping: s.mapping!,
      maxRows: s.maxRows || undefined,
      hfToken: s.hfToken || undefined,
    },
    vocabSize: s.vocabSize,
    chatTemplate: s.modelType === "base" ? undefined : s.chatTemplate,
    model: { hidden: s.hidden, layers: s.layers, baseWidth: s.baseWidth, maxSeq: s.maxSeq },
    training: {
      steps: s.steps,
      seqLen: s.seqLen,
      batch: s.batch,
      muonLr: s.muonLr,
      auxLr: s.auxLr,
      quant: s.quant,
      // Only meaningful for chat families; the server ignores it for base models.
      maskPromptLoss: s.modelType === "base" ? undefined : s.maskPromptLoss,
    },
    resumeFrom: s.goal === "continue" && s.resumeFrom ? s.resumeFrom : undefined,
  };
}

export interface StepProps {
  state: State;
  set: (p: Partial<State>) => void;
}

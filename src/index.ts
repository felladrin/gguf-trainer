// Felladrin's GGUF Trainer — train a Qwen3 language model from scratch in
// TypeScript and export it directly to GGUF. No Python, no Hugging Face, no PyTorch.

export * from "./gguf/gguf.ts";
export * from "./gguf/quantize.ts";
export * from "./gguf/f16.ts";
export * from "./tokenizer/bpe.ts";
export * from "./model/config.ts";
export * from "./model/autograd.ts";
export { Qwen3Model } from "./model/qwen3.ts";
export type { MuPOpts } from "./model/qwen3.ts";
export * from "./model/mup.ts";
export * from "./train/optimizer.ts";
export * from "./train/adam.ts";
export * from "./train/muon.ts";
export * from "./train/trainer.ts";
export * from "./train/schedule.ts";
export * from "./train/qk_clip.ts";
export * from "./data/tokens.ts";
export * from "./export/export_gguf.ts";
export * from "./export/load_gguf.ts";
export * from "./io.ts";
export { initWebGPU, WebGPUBackend } from "./backend/webgpu.ts";
export { trainLMGpu, trainLMGpuResident } from "./backend/train_gpu.ts";
export type { TrainGpuResidentOpts } from "./backend/train_gpu.ts";
export { MuonGpu } from "./backend/muon_gpu.ts";
export { AdamWGpu } from "./backend/adamw_gpu.ts";

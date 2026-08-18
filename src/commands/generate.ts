// `generate`: complete a prompt with the trainer's own forward pass.
//
// This exists to answer "did the training actually produce a model" without
// leaving the repo. It is greedy and has no KV cache, so it re-forwards the
// whole window per token: fine for a 60-token smoke test, useless for serving.
// Real generation is llama.cpp's job, on an exported GGUF.

import { loadModelFromGGUF } from "../export/load-gguf.ts";
import { readFileBytes } from "../io.ts";
import { greedyComplete } from "../eval/generate.ts";
import { initWebGPU } from "../backend/webgpu.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

async function run(v: Values) {
  const path = v.str("model");
  const prompt = v.str("prompt");
  const maxNew = v.num("max-tokens");
  const bytes = await readFileBytes(path).catch(() => {
    throw new UsageError(`cannot read ${path}`);
  });
  const { model, tokenizer } = loadModelFromGGUF(bytes);

  const gpu = v.bool("cpu") ? null : await initWebGPU();
  if (gpu) {
    gpu.install();
    gpu.uploadParams(model.params());
  }
  try {
    const promptIds = tokenizer.encode(prompt);
    const stop = [tokenizer.eosId];
    const ids = await greedyComplete(model, gpu, promptIds, maxNew, stop);
    const completion = tokenizer.decode(ids.slice(promptIds.length));
    console.log(v.bool("completion-only") ? completion : prompt + completion);
  } finally {
    if (gpu) gpu.uninstall();
  }
}

export const generateCommand: Command = {
  name: "generate",
  summary: "Greedy-complete a prompt with a checkpoint, without llama.cpp.",
  details: `Greedy decoding only, which makes it reproducible: same model, same prompt, same
output, every time. That is what you want from a smoke test.

A small base model loops without a repetition penalty, and this command has no sampler at
all, so expect repetition here even from a healthy checkpoint. Judge coherence over the
first sentence or two; for readable output, export the model and run llama.cpp with a
repetition penalty.`,
  examples: [
    'generate --model model.gguf --prompt "Once upon a time"',
    'generate --model model.gguf --prompt "The capital of France is" --max-tokens 40 --cpu',
  ],
  flags: [
    {
      name: "model",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the GGUF to run",
    },
    {
      name: "prompt",
      type: "string",
      placeholder: "TEXT",
      required: true,
      describe: "the text to continue",
    },
    {
      name: "max-tokens",
      type: "number",
      default: 60,
      describe: "how many tokens to generate before stopping",
    },
    {
      name: "completion-only",
      type: "boolean",
      describe: "print only the generated text, without the prompt",
    },
    { name: "cpu", type: "boolean", describe: "force the CPU forward pass instead of WebGPU" },
  ],
  run: run,
};

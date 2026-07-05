// In-browser inference for the freshly trained model, via wllama (llama.cpp in
// WASM). The model is served by the local engine server; wllama loads it by URL
// and uses the chat template embedded in the GGUF for chat models. API targets
// @wllama/wllama 3.5 (OpenAI-compatible completion calls).

import { Wllama } from "@wllama/wllama";
import wllamaWasmUrl from "@wllama/wllama/esm/wasm/wllama.wasm?url";

type ChatMsg = { role: "system" | "user" | "assistant"; content: string };

export async function loadModel(
  url: string,
  onProgress?: (fraction: number) => void,
): Promise<Wllama> {
  const wllama = new Wllama({ default: wllamaWasmUrl }, { suppressNativeLog: true });
  await wllama.loadModelFromUrl(url, {
    n_ctx: 1024,
    n_threads: Math.min(4, navigator.hardwareConcurrency || 2),
    progressCallback: onProgress
      ? ({ loaded, total }: { loaded: number; total: number }) =>
        onProgress(total ? loaded / total : 0)
      : undefined,
  });
  return wllama;
}

const SAMPLING = { temp: 0.7, top_k: 40, top_p: 0.95 } as const;

/** Chat turn using the GGUF's embedded chat template. */
export async function chat(
  wllama: Wllama,
  messages: { role: string; content: string }[],
  onToken: (partial: string) => void,
  nPredict = 256,
): Promise<string> {
  const stream = await wllama.createChatCompletion({
    messages: messages as ChatMsg[],
    stream: true,
    max_tokens: nPredict,
    ...SAMPLING,
  });
  let text = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content ?? "";
    if (delta) {
      text += delta;
      onToken(text);
    }
  }
  return text;
}

/** Raw completion for base models (no chat template). */
export async function complete(
  wllama: Wllama,
  prompt: string,
  onToken: (partial: string) => void,
  nPredict = 200,
): Promise<string> {
  const stream = await wllama.createCompletion({
    prompt,
    stream: true,
    max_tokens: nPredict,
    ...SAMPLING,
  });
  let text = prompt;
  for await (const chunk of stream) {
    const piece = chunk.choices[0]?.text ?? "";
    if (piece) {
      text += piece;
      onToken(text);
    }
  }
  return text;
}

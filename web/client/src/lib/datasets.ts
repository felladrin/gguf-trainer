// Recommended datasets per model type. These are starting points only — the
// wizard accepts ANY Hugging Face dataset URL (or a direct data-file URL).

import type { ModelType } from "../../../shared/types.ts";

export interface Suggestion {
  label: string;
  url: string;
  note: string;
}

export const RECOMMENDED: Record<ModelType, Suggestion[]> = {
  base: [
    { label: "TinyStories", url: "roneneldan/TinyStories", note: "Tiny synthetic stories; coherent from ~3M params." },
    { label: "FineWeb-Edu", url: "HuggingFaceFW/fineweb-edu", note: "Quality-filtered web text; the step up from stories." },
    { label: "WikiText-103", url: "Salesforce/wikitext", note: "Clean encyclopedic English." },
  ],
  instruct: [
    { label: "SmolTalk", url: "HuggingFaceTB/smoltalk", note: "Broad instruction chat, ChatML-friendly." },
    { label: "OpenHermes 2.5", url: "teknium/OpenHermes-2.5", note: "Large instruction mixture." },
    { label: "UltraChat 200k", url: "HuggingFaceH4/ultrachat_200k", note: "Multi-turn assistant chat." },
  ],
  reasoning: [
    { label: "OpenThoughts 114k", url: "open-thoughts/OpenThoughts-114k", note: "Chain-of-thought traces." },
    { label: "Bespoke-Stratos 17k", url: "bespokelabs/Bespoke-Stratos-17k", note: "Reasoning distill with <think>." },
  ],
  tools: [
    { label: "Hermes Function-Calling v1", url: "NousResearch/hermes-function-calling-v1", note: "Tool-call conversations." },
    { label: "Glaive Function-Calling v2", url: "glaiveai/glaive-function-calling-v2", note: "Synthetic function calls." },
  ],
};

export const MODEL_TYPES: { key: ModelType; label: string; blurb: string }[] = [
  { key: "base", label: "Base (pretraining)", blurb: "Plain text, no chat format. A raw next-token model." },
  { key: "instruct", label: "Instruct", blurb: "Conversational, non-reasoning. Follows instructions in ChatML." },
  { key: "reasoning", label: "Reasoning", blurb: "Conversational with <think> traces before the answer." },
  { key: "tools", label: "Instruct + Tools", blurb: "Instruct plus tool/function-calling in <tool_call> form." },
];

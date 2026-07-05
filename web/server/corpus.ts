// Turn parsed dataset rows into a pretokenized .tokens file the trainer streams.
// Base models: raw text per row (split on <|endoftext|>). Chat models: render
// each conversation through the Jinja chat template, so the model trains on the
// exact ChatML the exporter also embeds. BPE is trained on a bounded sample
// first (with the right specials), then the whole corpus is encoded and EOS-
// joined — the same recipe as examples/pretokenize.ts, driven by wizard config.

import { Template } from "@huggingface/jinja";
import { BPETokenizer } from "../../src/tokenizer/bpe.ts";
import { tokenBytes, writeTokenFile } from "../../src/data/tokens.ts";
import {
  CHATML_SPECIALS,
  type FieldMapping,
  type ModelType,
  rowToMessages,
  rowToText,
} from "../../src/data/chat.ts";
import type { Row } from "./parse.ts";

export interface BuildCorpusOpts {
  rows: Row[];
  modelType: ModelType;
  mapping: FieldMapping;
  vocabSize: number;
  chatTemplate?: string;
  outPath: string; // where to write the .tokens file
  maxTokens?: number; // stop once the corpus has this many tokens (bounds huge sets)
  sampleBytes?: number; // BPE training sample size (default 8 MB)
  existingTok?: BPETokenizer; // resume: reuse this vocab instead of training a new one
  onProgress?: (msg: string) => void;
}

export interface BuiltCorpus {
  tok: BPETokenizer;
  tokenizerJSON: string; // BPETokenizer.export() serialized
  numTokens: number;
  numDocs: number;
  bytesPerToken: 2 | 4;
}

/** Materialize training documents (strings) from rows per the model type. */
function buildDocs(o: BuildCorpusOpts): string[] {
  const docs: string[] = [];
  if (o.modelType === "base") {
    for (const r of o.rows) {
      const text = rowToText(r, o.mapping);
      if (!text) continue;
      // A .txt corpus arrives as one big row; honor <|endoftext|> boundaries.
      for (const part of text.split("<|endoftext|>")) {
        const s = part.trim();
        if (s) docs.push(s);
      }
    }
    return docs;
  }

  // Chat family: render each conversation with the template.
  const render = new Template(o.chatTemplate ?? "");
  for (const r of o.rows) {
    const messages = rowToMessages(r, o.mapping);
    if (!messages || messages.length < 2) continue;
    const tools = Array.isArray((r as Row).tools) ? (r as Row).tools : undefined;
    try {
      const text = render.render({ messages, tools, add_generation_prompt: false });
      if (text.trim()) docs.push(text);
    } catch { /* skip rows the template can't render */ }
  }
  return docs;
}

export async function buildCorpus(o: BuildCorpusOpts): Promise<BuiltCorpus> {
  const log = o.onProgress ?? (() => {});
  const docs = buildDocs(o);
  if (docs.length === 0) {
    throw new Error(
      "No usable documents from this dataset — check the field mapping / model type.",
    );
  }
  log(`assembled ${docs.length} documents`);

  // Train BPE on a bounded sample (specials depend on model type) — unless we're
  // resuming, in which case the checkpoint's vocab must be reused verbatim.
  let tok: BPETokenizer;
  if (o.existingTok) {
    tok = o.existingTok;
    log(`reusing checkpoint vocab (${tok.vocabSize} tokens)`);
  } else {
    const sampleBytes = o.sampleBytes ?? 8 * 1024 * 1024;
    let sample = "";
    for (const d of docs) {
      sample += d + "\n";
      if (sample.length >= sampleBytes) break;
    }
    const specials = o.modelType === "base" ? ["<|endoftext|>"] : CHATML_SPECIALS;
    tok = new BPETokenizer();
    log(`training BPE (vocab ${o.vocabSize}) on ${(sample.length / 1e6).toFixed(1)} MB sample...`);
    tok.train(sample, o.vocabSize, specials);
  }

  // Inference EOS: chat stops at <|im_end|>; base stops at <|endoftext|>.
  const imEnd = tok.idOf("<|im_end|>");
  if (o.modelType !== "base" && imEnd !== undefined) tok.eosId = imEnd;
  const sepId = tok.idOf("<|endoftext|>") ?? tok.eosId; // hard document separator

  // Encode all docs, EOS-joined, up to the token budget.
  const maxTokens = o.maxTokens ?? Infinity;
  const ids: number[] = [];
  let used = 0;
  for (const d of docs) {
    for (const id of tok.encode(d)) ids.push(id);
    ids.push(sepId);
    used++;
    if (ids.length >= maxTokens) break;
  }
  log(`encoded ${used} documents -> ${ids.length} tokens (vocab ${tok.vocabSize})`);

  const bpt = tokenBytes(tok.vocabSize);
  await writeTokenFile(o.outPath, ids, bpt);

  return {
    tok,
    tokenizerJSON: JSON.stringify(tok.export()),
    numTokens: ids.length,
    numDocs: used,
    bytesPerToken: bpt,
  };
}

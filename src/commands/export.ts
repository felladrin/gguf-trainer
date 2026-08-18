// Re-export a trained checkpoint under its release name, plus deployment quants.
//
// A checkpoint carries whatever `general.name` the training run was launched
// with ("gemma3-96m-base"), which is a run label, not a model name. Publishing
// wants the release name in the metadata, and quantized copies next to the f32
// master. This reads the checkpoint back through the normal loader and writes
// each requested quant, so the weights make the same round trip llama.cpp does.
//
// Defaults: the output directory is the input's, quants are f32,q8_0,q4_0. The f32 copy
// is byte-compared against the source tensor data, so a silent corruption in the
// round trip fails the run instead of shipping.
//
// Any chat template in the source is carried over (a base model has none; an
// instruct checkpoint does). The optimizer sidecar is NOT touched: copy
// <in.gguf>.optstate to <out>.optstate yourself if the release is meant to be
// resumable, since only the f32 export can continue training.

import { readGGUF } from "../gguf/gguf.ts";
import { readFileBytes, writeFileBytes } from "../io.ts";
import { loadModelFromGGUF } from "../export/load-gguf.ts";
import { parseQuantList } from "../gguf/quantize.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

function die(msg: string): never {
  throw new UsageError(msg);
}

async function run(v: Values) {
  const inPath = v.str("model");
  const name = v.str("name");
  const outDir = v.opt("out-dir") ?? (inPath.slice(0, inPath.lastIndexOf("/") + 1) || "./");
  const quants = parseQuantList(v.str("quants"));

  const bytes = await readFileBytes(inPath).catch(() => die(`cannot read ${inPath}`));
  const source = readGGUF(bytes);
  const { model, cfg, tokenizer, arch } = loadModelFromGGUF(bytes);
  const chatTemplate = source.metadata.get("tokenizer.chat_template") as string | undefined;
  console.log(
    `${inPath}: ${source.tensors.length} tensors, name "${source.metadata.get("general.name")}" ` +
      `-> "${name}"${chatTemplate ? ", chat template carried over" : ", no chat template (base)"}`,
  );

  for (const quant of quants) {
    const path = `${outDir.replace(/\/?$/, "/")}${name}.${quant.toUpperCase()}.gguf`;
    const out = arch.exportGGUF(model, tokenizer.export(), cfg, { quant, name, chatTemplate });
    await writeFileBytes(path, out);

    const check = readGGUF(out);
    if (check.metadata.get("general.name") !== name) die(`${path}: name did not round-trip`);
    if (check.tensors.length !== source.tensors.length) {
      die(`${path}: ${check.tensors.length} tensors, source has ${source.tensors.length}`);
    }
    // f32 is the resumable master: its weights must be bit-identical to the source,
    // not merely close. The quants are lossy by definition, so only shape is checked.
    if (quant === "f32") {
      for (const t of check.tensors) {
        const src = source.tensors.find((s) => s.name === t.name);
        if (!src) die(`${path}: tensor ${t.name} missing from the source`);
        if (src.data.length !== t.data.length) die(`${path}: ${t.name} size changed`);
        for (let i = 0; i < t.data.length; i++) {
          if (t.data[i] !== src.data[i]) die(`${path}: ${t.name} differs at byte ${i}`);
        }
      }
      console.log(
        `  ${path} (${(out.length / 1e6).toFixed(0)} MB, ${quant}) tensor bytes identical`,
      );
    } else {
      console.log(`  ${path} (${(out.length / 1e6).toFixed(0)} MB, ${quant})`);
    }
  }
}

export const exportCommand: Command = {
  name: "export",
  summary: "Re-export a checkpoint under a release name, with deployment quants.",
  details: `A checkpoint carries whatever name the training run was launched with, which is a run
label, not a model name. This reads the checkpoint back through the normal loader and
writes one file per requested quant under the release name, carrying over any chat
template.

The f32 copy is byte-compared against the source tensor data before it is accepted: it is
the resumable master, so a silent corruption in the round trip fails the command instead of
shipping.

The optimizer sidecar is not touched. Copy <model>.optstate to <release>.F32.gguf.optstate
yourself if the release should be resumable, and note that only the f32 export can continue
training.`,
  examples: [
    "export --model out/run.gguf --name My-Model-95M",
    "export --model run.gguf --name My-Model --quants f32,q8_0 --out-dir release/",
  ],
  flags: [
    {
      name: "model",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the checkpoint to re-export",
    },
    {
      name: "name",
      type: "string",
      placeholder: "NAME",
      required: true,
      describe: "release name; becomes general.name in the GGUF and the filename stem",
    },
    {
      name: "quants",
      type: "string",
      default: "f32,q8_0,q4_0",
      describe: "comma-separated quants to write",
    },
    {
      name: "out-dir",
      type: "string",
      placeholder: "DIR",
      describe: "output directory (default: alongside the input)",
    },
  ],
  run: run,
};

# Contributing to Felladrin's GGUF Trainer

Thanks for helping build a from-scratch, GGUF-native LLM trainer in TypeScript.

## Project principles

1. **The engine stays dependency-free and runtime-agnostic.** Everything the model itself needs
   (`src/model/`, `src/arch/`, `src/train/`, `src/gguf/`, `src/tokenizer/`, `src/export/`) must run
   on Deno, Bun and Node with no npm install. `deno task test:node` checks the Node half of that. Avoid
   runtime-specific APIs; file I/O goes through `src/io.ts`. The two npm dependencies (hyparquet,
   @huggingface/jinja) belong to data fetching and chat templating only, in `src/data/` and
   `src/commands/`; do not let them reach the engine.
2. **GGUF loadability is a contract.** Each architecture's tensor names and metadata keys, in its
   own `src/arch/<name>.ts`, mirror what `llama.cpp` expects. Don't change them without validating
   the output loads in `llama-cli`.
3. **Correctness before speed.** The CPU backend is the reference implementation; optimize on the
   WebGPU backend instead of complicating it.
4. **No TS features that break `--experimental-strip-types`.** No `enum`, `namespace`, or parameter
   properties: they don't run on Node's type-stripping and they hurt portability. Use `as const` objects
   and explicit field assignment.

## Adding an architecture

One file in `src/arch/`, one line in `src/model/registry.ts`, and it inherits the gradient checks
and the export round-trip test. The recipe, the contract and the parts that are easy to get wrong
are in [docs/adding-an-architecture.md](../docs/adding-an-architecture.md).

## Adding an autograd op

Every new op in `src/model/autograd.ts` needs a **finite-difference gradient check** before it's
trusted: add a case to `tests/gradcheck.ts` (it perturbs each input by ±ε and compares
`(f(x+ε) − f(x−ε)) / 2ε` against the analytic gradient your `_backward` produces) and run
`deno task test` plus `deno task test:node`. New ops also need an entry in the `OpsBackend`
interface and a WebGPU implementation, or GPU graphs break.

## Adding a WebGPU kernel

Implement the same math as the CPU op in `src/backend/webgpu.ts`, keep the `Tensor` interface
identical, and add a case to `tests/gpu-parity.ts`. A kernel is not trusted until its forward and
backward match the CPU backend within tolerance (`deno task test`). Bring up kernels one at a time.

## Style

- `deno fmt` (config in `deno.json`); 100-col lines.
- Explain _why_ in comments, not _what_.
- Small, reviewable PRs. Describe how you verified the change.

## Reporting issues

Include runtime + version, a minimal repro, and (for numerical bugs) the smallest config that
shows it.

## Where code goes

Every workflow is a subcommand: a module in `src/commands/` exporting a `Command`, registered in
`src/cli/registry.ts`. There are no loose scripts; a workflow that does not appear in
`deno run -A cli.ts help` does not exist.

A command's flags carry their own `describe` text, which is simultaneously the `--help` output and
the `help --json` schema that agents read. Write those descriptions as documentation, not as labels.

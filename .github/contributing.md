# Contributing to Felladrin's GGUF Trainer

Thanks for helping build a from-scratch, GGUF-native LLM trainer in TypeScript.

## Project principles

1. **The engine stays dependency-free.** Everything the model itself needs (`src/model/`,
   `src/arch/`, `src/train/`, `src/gguf/`, `src/tokenizer/`, `src/export/`) runs on nothing but the
   runtime. File I/O goes through `src/io.ts`. The two npm dependencies (hyparquet,
   @huggingface/jinja) belong to data fetching and chat templating only, in `src/data/` and
   `src/commands/`; do not let them reach the engine, and import them where they are used rather
   than at the top, so nothing pays to resolve a package it never calls. Every test file belongs in
   `deno task test`, which `tests/task-coverage.ts` checks against the directory: a file sat in
   `tests/` unrun for weeks once, and nothing about it looked wrong.
2. **GGUF loadability is a contract.** Each architecture's tensor names and metadata keys, in its
   own `src/arch/<name>.ts`, mirror what `llama.cpp` expects. Don't change them without validating
   the output loads in `llama-cli`.
3. **Correctness before speed.** `src/model/autograd.ts` is the reference implementation the WebGPU
   kernels are checked against. Optimize the kernels instead of complicating it, and never relax a
   parity tolerance to make a kernel pass.
4. **Deno only.** Training needs WebGPU, so the runtime that ships it is the one this repo
   supports.

## Adding an architecture

One file in `src/arch/`, one line in `src/model/registry.ts`, and it inherits the gradient checks
and the export round-trip test. The recipe, the contract and the parts that are easy to get wrong
are in [docs/adding-an-architecture.md](../docs/adding-an-architecture.md).

## Adding an autograd op

Every new op in `src/model/autograd.ts` needs a **finite-difference gradient check** before it's
trusted: add a case to `tests/gradcheck.ts` (it perturbs each input by ±ε and compares
`(f(x+ε) − f(x−ε)) / 2ε` against the analytic gradient your `_backward` produces) and run
`deno task test`. New ops also need an entry in the `OpsBackend` interface and a WebGPU
implementation, or GPU graphs break.

## Adding a WebGPU kernel

Implement the same math as the reference op in `src/backend/webgpu.ts`, keep the `Tensor` interface
identical, and add a case to `tests/gpu-parity.ts`. A kernel is not trusted until its forward and
backward match the reference within tolerance (`deno task test`). Bring up kernels one at a time.

## Prove a test can fail

Every guard added here gets a test, and the test gets one more step: break the thing it guards, watch
it go red, put it back. Not as a ritual. Three tests written on 2026-09-10 passed for reasons that
had nothing to do with what they claimed, and each one looked right:

- **A NaN oracle.** `!isFinite(got) || Math.abs(got - want) > tol` reads as symmetric and is not:
  with `want` NaN, `Math.abs(0 - NaN)` is NaN and `NaN > tol` is `false`, so a broken reference side
  passes silently. Prefer a comparison that fails on NaN, like `got !== 0`, over one that has to be
  told about it.
- **A pass condition the absence of work also satisfies.** A gate asserting "the loss is 0 on a
  fully masked batch" passes just as well when the readback never happened, because an unread host
  tensor is zeros. It needed a scored control in the same `sync`, and a `NaN` seeded into every
  scalar first so each proves the device wrote it.
- **Two sets that never intersect.** A gate meant to catch a later pass overwriting an earlier one's
  buffer allocated them so that the writes and the reads landed on different halves of the pool, so
  no ordering violation could have moved the result. Running the backward is what made the buffers
  overlap.

The question to ask is not "does this pass?" but "what would have to be true for this to fail, and
have I made that happen?". If the answer is "nothing I can do from inside this repo", say so in the
test, as `recycleReuseGate` does.

## Style

- `deno fmt` (config in `deno.json`); 100-col lines.
- Explain _why_ in comments, not _what_.
- Small, reviewable PRs. Describe how you verified the change.

## Reporting issues

Include your Deno version and GPU adapter, a minimal repro, and, for numerical bugs, the smallest
config that shows it.

## Where code goes

Every workflow is a subcommand: a module in `src/commands/` exporting a `Command`, registered in
`src/cli/registry.ts`. There are no loose scripts; a workflow that does not appear in
`deno run -A cli.ts help` does not exist.

A command's flags carry their own `describe` text, which is simultaneously the `--help` output and
the `help --json` schema that agents read. Write those descriptions as documentation, not as labels.

# Contributing to Felladrin's GGUF Trainer

Thanks for helping build a from-scratch, GGUF-native LLM trainer in TypeScript.

## Project principles

1. **Reference backend stays dependency-free and runtime-agnostic.** The code in `src/` (outside
   `backend/`) must run on Deno, Bun, and Node with no npm install. Avoid runtime-specific APIs;
   file I/O goes through `src/io.ts`.
2. **GGUF loadability is a contract.** The `gemma3` architecture tensor names and metadata keys in
   `src/export/export_gguf.ts` mirror what `llama.cpp` expects. Don't change them without validating
   the output loads in `llama-cli`.
3. **Correctness before speed.** The CPU backend is the reference implementation; optimize on the
   WebGPU backend instead of complicating it.
4. **No TS features that break `--experimental-strip-types`.** No `enum`, `namespace`, or parameter
   properties — they don't run on Node's type-stripping and hurt portability. Use `as const` objects
   and explicit field assignment.

## Adding an autograd op

Every new op in `src/model/autograd.ts` needs a **finite-difference gradient check** before it's
trusted: add a case to `tests/gradcheck.ts` (it perturbs each input by ±ε and compares
`(f(x+ε) − f(x−ε)) / 2ε` against the analytic gradient your `_backward` produces) and run
`deno task test` plus `deno task test:node`. New ops also need an entry in the `OpsBackend`
interface and a WebGPU implementation, or GPU graphs break.

## Adding a WebGPU kernel

Implement the same math as the CPU op in `src/backend/webgpu.ts`, keep the `Tensor` interface
identical, and add a case to `tests/gpu_parity.ts` — a kernel is not trusted until its forward and
backward match the CPU backend within tolerance (`deno task test`). Bring up kernels one at a time.

## Style

- `deno fmt` (config in `deno.json`); 100-col lines.
- Explain _why_ in comments, not _what_.
- Small, reviewable PRs. Describe how you verified the change.

## Reporting issues

Include runtime + version, a minimal repro, and — for numerical bugs — the smallest config that
shows it.

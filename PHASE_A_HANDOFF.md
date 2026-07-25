# Phase A pretraining — handoff (read me if you're picking this up on Strix)

This machine (Strix Halo, gfx1151) is training the first real model: a **94.7M-param
Gemma3 base, 8192 context**, on a 722M-token blend corpus. Goal: a strong, instructable
base (beats Felladrin/Minueza-32M; will NOT beat Minueza-2-96M, which saw 185B tokens vs
our ~1.44B — that's the hardware ceiling, accepted). This is `examples/pretrain.ts`.

## Config (must match exactly on any resume)
`examples/blend.tokens 640 12 88000 2048 8 0.01 --maxSeq=8192 --ckpt=500 --quant=f32 --reclaim
--out=examples/pretrain-blend-base.gguf --name=gemma3-96m-base`
= hidden 640 × 12 layers, headDim 64, SWA window 1024, vocab 32768, seqLen 2048, batch 8,
muon lr 0.01, WSD (warmup 8800 / cooldown 17600). 88000 steps = 2 epochs ≈ ~21 days at
the measured 0.050 st/s. Checkpoints every 500 steps to the --out GGUF (f32, atomic).

`--reclaim` is REQUIRED, not optional: it flushes each micro-batch as its own GPU
submission (~2.5 s), keeping every submit under amdgpu's ~10 s GFX ring watchdog. Without
it the whole ~20 s step submits as one command buffer and the driver kills it mid-step
(`ring gfx_0.0.0 timeout` → hard recovery → `OperationError` at the first sync). It also
caps peak VRAM at ~one micro-batch. The alternative — raising the watchdog via
`amdgpu.lockup_timeout=100000` on the kernel cmdline (finite so a real hang still recovers;
`0` means "keep default", `-1` disables recovery) — is not set on this box, so keep `--reclaim`.

## Files (all already here, gitignored)
- `examples/blend.tokens` (1.44 GB, 722M tokens) + `examples/blend.tokenizer.json` (vocab 32768).
- `examples/pretrain-blend-base.gguf` — the latest checkpoint (weights). Overwritten each ckpt.
- `examples/phaseA.log` — training log (loss every 880 steps, "[ckpt @ N]" every 500).

## To STOP
`pkill -f 'pretrain[.]ts'`   (the `[.]` keeps pkill from matching its own cmdline).
Checkpoints are atomic, so stopping never corrupts the on-disk GGUF; you lose only the
steps since the last "[ckpt @ N]" line (≤500 steps ≈ ≤2.8 h).

## To RESUME
`bash ~/gguf-trainer/resume_phase_a.sh`
It auto-detects the resume step from the last "[ckpt @ N]" in the log, copies the
checkpoint to a resume-from-N.gguf, and relaunches detached. Weights resume exactly; the
WSD schedule continues from step N (no re-warmup). Momentum cold-starts on the FIRST
resume (these checkpoints predate optimizer-state saving) — a few-step transient, then
later checkpoints write a `.optstate` sidecar for seamless subsequent resumes.

## To MONITOR
`tail -f ~/gguf-trainer/examples/phaseA.log` and `cat /sys/class/drm/card*/device/gpu_busy_percent`.
Healthy = GPU ~100%, loss descending (10.41 → 5.81 @880 → 4.31 @1760 so far), no NaN.

## Remaining optimization levers, and why f32 not f16, etc.: `docs/OPTIMIZATION.md`.
## After Phase A: Phase B (long-context, seq 8192 batch 1) then instruct/reasoning/tools.

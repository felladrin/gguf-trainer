// Warmup–Stable–Decay (WSD) learning-rate schedule.
//
// Returns a per-step MULTIPLIER in [minScale, 1] applied on top of each
// optimizer group's configured base lr (setLrScale). Decoupling the shape from
// the absolute lr keeps two independent base lrs: Muon on hidden matmuls,
// AdamW on the aux group: moving together under one schedule, which is how WSD
// is used in practice (nanoGPT speedrun, MiniCPM). Warmup stabilizes the early,
// high-variance updates; the long stable phase does the bulk of the learning;
// the linear cooldown lets the loss settle. Pure and dependency-free: the
// trainer calls it once per step (see ../train/trainer.ts, ../backend/train-gpu.ts).

export interface WSDOpts {
  warmupSteps: number; // linear ramp 0 -> 1 over these steps
  stableSteps: number; // hold at 1
  cooldownSteps: number; // linear decay 1 -> minScale
  minScale?: number; // cooldown floor as a fraction of base lr (default 0)
}

/** Build a step -> lr-multiplier function for the WSD phases above. */
export function wsdSchedule(opts: WSDOpts): (step: number) => number {
  const warmup = Math.max(0, Math.floor(opts.warmupSteps));
  const stable = Math.max(0, Math.floor(opts.stableSteps));
  const cooldown = Math.max(0, Math.floor(opts.cooldownSteps));
  const minScale = opts.minScale ?? 0;
  const cooldownStart = warmup + stable;
  const total = cooldownStart + cooldown;
  return (step: number) => {
    if (warmup > 0 && step < warmup) return (step + 1) / warmup; // hits 1 at step warmup-1
    if (step < cooldownStart) return 1; // stable
    if (cooldown > 0 && step < total) {
      const frac = (step - cooldownStart + 1) / cooldown; // (0,1] across cooldown
      return 1 - (1 - minScale) * frac;
    }
    return step < total ? 1 : minScale; // no-cooldown case, then the post-schedule floor
  };
}

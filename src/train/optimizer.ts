// Common optimizer interface so the trainer is agnostic to AdamW vs Muon.

export interface Optimizer {
  zeroGrad(): void;
  step(): void;
  /** Optional WSD hook: set effective lr to `scale` × the base lr. */
  setLrScale?(scale: number): void;
}

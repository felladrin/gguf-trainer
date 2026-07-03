// Common optimizer interface so the trainer is agnostic to AdamW vs Muon.

export interface Optimizer {
  zeroGrad(): void;
  step(): void;
}

/**
 * Endpoint means for a training run's logged losses.
 *
 * A run's one-line verdict used to compare the single batch at each end. Per-batch
 * loss on this project's runs swings by ~0.9 nats, which is wider than the whole
 * improvement a continued-pretrain segment produces, so those two samples decide
 * the headline: the 7528-step roleplay run reported "loss 2.896 -> 3.256" over a
 * run whose binned mean fell 3.336 -> 2.757.
 *
 * The windows never overlap, so a short run reports a real before and after
 * rather than the same batches twice.
 */
export function lossTrend(
  losses: number[],
  window = 10,
): { first: number; last: number; window: number } | null {
  if (losses.length === 0) return null;
  const w = Math.min(window, Math.max(1, Math.floor(losses.length / 2)));
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return { first: mean(losses.slice(0, w)), last: mean(losses.slice(-w)), window: w };
}

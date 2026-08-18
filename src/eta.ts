/** Format a duration in seconds as "d:hh:mm:ss" for an ETA readout.
 *  Returns "--:--:--:--" when the input isn't a finite, non-negative number
 *  (e.g. the step rate is still 0 at the very first step of a resume segment,
 *  so remaining/rate is Infinity). */
export function fmtEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "--:--:--:--";
  const s = Math.round(seconds);
  const p2 = (n: number) => String(n).padStart(2, "0");
  return `${Math.floor(s / 86400)}:${p2(Math.floor((s % 86400) / 3600))}:${
    p2(Math.floor((s % 3600) / 60))
  }:${p2(s % 60)}`;
}

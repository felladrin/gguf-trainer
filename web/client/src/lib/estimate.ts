// Model-size presets + rough estimates, computed with the real engine config
// math (gemma3Config/gemma3ParamCount are dependency-free, so they bundle into
// the UI).

import { gemma3Config, gemma3ParamCount } from "../../../../src/model/config.ts";

export interface SizePreset {
  key: string;
  label: string;
  hidden: number;
  layers: number;
}

// hidden must be a multiple of headDim*2 = 128 (gemma3Config requirement).
export const PRESETS: SizePreset[] = [
  { key: "tiny", label: "Tiny", hidden: 256, layers: 4 },
  { key: "small", label: "Small", hidden: 384, layers: 6 },
  { key: "medium", label: "Medium", hidden: 512, layers: 8 },
];

export function paramsFor(
  vocab: number,
  hidden: number,
  layers: number,
  maxSeq: number,
): number | null {
  try {
    return gemma3ParamCount(gemma3Config(vocab, hidden, layers, maxSeq));
  } catch {
    return null;
  }
}

export function residentMB(params: number): number {
  return (params * 20) / 1e6; // ~20 bytes/param of resident training state
}

// Rough per-step wall time, calibrated on an M1 Max so ~14M @ seqLen256/batch16
// lands near a second. Real hardware varies; this is a ballpark, not a promise.
export function stepSeconds(params: number, seqLen: number, batch: number): number {
  const scale = (params / 1e6) * (seqLen / 256) * (batch / 16);
  return 0.065 * scale + 0.02;
}

export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export function fmtCount(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

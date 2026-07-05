// Minimal dependency-free SVG loss curve.

export function LossChart({ points }: { points: { step: number; loss: number }[] }) {
  const W = 600, H = 200, pad = 30;
  if (points.length < 2) {
    return (
      <svg className="chart" viewBox={`0 0 ${W} ${H}`}>
        <text x={W / 2} y={H / 2} fill="#9aa7b4" fontSize="13" textAnchor="middle">
          waiting for steps…
        </text>
      </svg>
    );
  }
  const xs = points.map((p) => p.step);
  const ys = points.map((p) => p.loss);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  const d = points.map((p, i) => `${i ? "L" : "M"}${sx(p.step).toFixed(1)},${sy(p.loss).toFixed(1)}`)
    .join(" ");

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`}>
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#2b333d" />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#2b333d" />
      <text x={pad - 4} y={sy(y1)} fill="#9aa7b4" fontSize="10" textAnchor="end">{y1.toFixed(2)}</text>
      <text x={pad - 4} y={sy(y0)} fill="#9aa7b4" fontSize="10" textAnchor="end">{y0.toFixed(2)}</text>
      <text x={W - pad} y={H - pad + 14} fill="#9aa7b4" fontSize="10" textAnchor="end">step {x1}</text>
      <path d={d} fill="none" stroke="#6e8bff" strokeWidth="2" />
    </svg>
  );
}

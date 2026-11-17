import type { ReactNode } from "react";
import { datasets, type DatasetKind, type Point } from "@/lib/classify/data";

export const CLASS_COLORS = ["#5f7a96", "#c96442", "#6f8a63"];
export const CLASS_NAMES = ["A", "B", "C"];
export const fmt = (v: number, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
export const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Square plot of the feature plane with independent axis ranges. */
export const SIZE = 420, PAD = 30;
export interface Frame { x: [number, number]; y: [number, number] }
export const toScreen = (f: Frame, p: readonly number[]) => [
  PAD + ((p[0] - f.x[0]) / (f.x[1] - f.x[0])) * (SIZE - 2 * PAD),
  SIZE - PAD - ((p[1] - f.y[0]) / (f.y[1] - f.y[0])) * (SIZE - 2 * PAD),
] as const;
export const toData = (f: Frame, sx: number, sy: number): [number, number] => [
  f.x[0] + ((sx - PAD) / (SIZE - 2 * PAD)) * (f.x[1] - f.x[0]),
  f.y[0] + ((SIZE - PAD - sy) / (SIZE - 2 * PAD)) * (f.y[1] - f.y[0]),
];

export function Axes({ frame, labels = ["x₁", "x₂"] }: { frame: Frame; labels?: [string, string] }) {
  const ticks = (r: [number, number]) => { const step = (r[1] - r[0]) / 4; return Array.from({ length: 5 }, (_, i) => r[0] + i * step); };
  return <g className="cls-axes">
    <defs><clipPath id="cls-clip"><rect x={PAD} y={PAD} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD}/></clipPath></defs>
    <rect x={PAD} y={PAD} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD} className="cls-frame"/>
    {ticks(frame.x).map((v) => { const [sx] = toScreen(frame, [v, frame.y[0]]); return <text key={`x${v}`} x={sx} y={SIZE - 10} textAnchor="middle">{fmt(v, Math.abs(v) >= 10 ? 0 : 1)}</text>; })}
    {ticks(frame.y).map((v) => { const [, sy] = toScreen(frame, [frame.x[0], v]); return <text key={`y${v}`} x={PAD - 5} y={sy + 4} textAnchor="end">{fmt(v, Math.abs(v) >= 10 ? 0 : 1)}</text>; })}
    <text x={SIZE - PAD - 6} y={SIZE - PAD - 7} textAnchor="end" className="cls-axis-name">{labels[0]}</text>
    <text x={PAD + 4} y={PAD - 8} className="cls-axis-name">{labels[1]}</text>
  </g>;
}

export function Points({ frame, train, test, dim }: { frame: Frame; train: Point[]; test: Point[]; dim?: (i: number) => boolean }) {
  return <g>
    {test.map((p, i) => { const [x, y] = toScreen(frame, p.x); return <circle key={`t${i}`} cx={x} cy={y} r={4} fill="#fff" stroke={CLASS_COLORS[p.y]} strokeWidth={1.6}/>; })}
    {train.map((p, i) => { const [x, y] = toScreen(frame, p.x); return <circle key={`r${i}`} cx={x} cy={y} r={4} fill={CLASS_COLORS[p.y]} stroke="#fff" strokeWidth={1} opacity={dim?.(i) ? .25 : 1}/>; })}
  </g>;
}

export function Legend({ classes }: { classes: number }) {
  return <div className="cls-legend">
    {Array.from({ length: classes }, (_, c) => <span key={c}><i style={{ background: CLASS_COLORS[c] }}/>Class {CLASS_NAMES[c]}</span>)}
    <span><i className="open"/>held-out (test)</span>
  </div>;
}

export function DataControls({ kind, setKind, seed, setSeed, noise, setNoise, children }: {
  kind: DatasetKind; setKind: (k: DatasetKind) => void; seed: number; setSeed: (n: number) => void; noise: number; setNoise: (n: number) => void; children?: ReactNode;
}) {
  return <section className="lab-panel">
    <div className="lab-panel-heading"><h2>Data</h2><p className="lab-panel-subtitle">{datasets[kind].description}</p></div>
    <div className="mlp-segmented cls-seg" role="group" aria-label="Dataset" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}>
      {(Object.keys(datasets) as DatasetKind[]).map((k) => <button key={k} type="button" aria-pressed={kind === k} onClick={() => setKind(k)}>{datasets[k].name}</button>)}
    </div>
    <label className="slider-field"><span className="slider-label">Noise <output>{noise.toFixed(2)}</output></span>
      <input type="range" min={.05} max={.6} step={.01} value={noise} onChange={(e) => setNoise(Number(e.target.value))}/></label>
    <label className="input-field">Data seed
      <input className="field-control" type="number" min={0} max={999999} value={seed} onChange={(e) => setSeed(Math.min(999999, Math.max(0, Number(e.target.value) || 0)))}/></label>
    {children}
  </section>;
}

export function Segmented<T extends string | number>({ label, options, value, onChange }: { label: string; options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return <div className="input-field mlp-width">{label}
    <div className="mlp-segmented" role="group" aria-label={label} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map(([v, text]) => <button key={String(v)} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>)}
    </div>
  </div>;
}

/** Small line chart of train/test accuracy against a hyperparameter. */
export function AccuracyChart({ xs, train, test, current, xLabel }: { xs: number[]; train: number[]; test: number[]; current: number; xLabel: string }) {
  const W = 420, H = 170, L = 36, R = 10, T = 12, B = 26;
  const x = (v: number) => L + ((v - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0])) * (W - L - R);
  const lo = Math.max(0, Math.floor(Math.min(...train, ...test) * 10) / 10 - .05);
  const y = (v: number) => T + (1 - (v - lo) / (1 - lo)) * (H - T - B);
  const line = (vals: number[]) => vals.map((v, i) => `${i ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const i = xs.indexOf(current);
  const best = test.reduce((b, v, j) => (v > test[b] ? j : b), 0);
  return <figure className="cls-acc">
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Train and test accuracy by ${xLabel}; best test accuracy ${pct(test[best])} at ${xs[best]}`}>
      {[lo, (lo + 1) / 2, 1].map((v) => <g key={v} className="cls-axes"><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="cls-grid"/><text x={L - 5} y={y(v) + 4} textAnchor="end">{pct(v)}</text></g>)}
      <g className="cls-axes">{[xs[0], xs[Math.floor(xs.length / 2)], xs[xs.length - 1]].map((v) => <text key={v} x={x(v)} y={H - 8} textAnchor="middle">{v}</text>)}</g>
      <path d={line(train)} fill="none" stroke="#6f8a63" strokeWidth={2}/>
      <path d={line(test)} fill="none" stroke="#c96442" strokeWidth={2}/>
      {i >= 0 && <><line x1={x(current)} x2={x(current)} y1={T} y2={H - B} stroke="#1f1e1c" strokeOpacity={.35}/>
        <circle cx={x(current)} cy={y(train[i])} r={3.5} fill="#6f8a63"/><circle cx={x(current)} cy={y(test[i])} r={3.5} fill="#c96442"/></>}
    </svg>
    <figcaption><span className="sw" style={{ background: "#6f8a63" }}/>train <span className="sw" style={{ background: "#c96442" }}/>test · at {xLabel} = {current}: <b>{i >= 0 ? pct(test[i]) : "—"}</b> test · best {pct(test[best])} at {xLabel} = {xs[best]}</figcaption>
  </figure>;
}

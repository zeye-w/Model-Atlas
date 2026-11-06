import { useEffect, useState } from "react";

export function NumberField({ label, min, max, step = 1, value, change, disabled }: {
  label: string; min: number; max: number; step?: number; value: number; change: (v: number) => void; disabled?: boolean;
}) {
  const [draft, setDraft] = useState(String(value));
  useEffect(() => setDraft(String(value)), [value]);
  return <label>{label}<input type="number" {...{ min, max, step, disabled }} value={draft} onChange={e => {
    setDraft(e.target.value);
    const next = e.target.valueAsNumber;
    if (e.target.validity.valid && Number.isFinite(next)) change(next);
  }} onBlur={() => setDraft(String(value))}/></label>;
}

const colors = ["#ad5537", "#65795b", "#416984", "#935d88", "#80651d", "#51498e"];

export function Plot({ title, rows, marks = [], labels, domain }: {
  title: string; rows: number[][]; marks?: number[]; labels?: string[]; domain?: [number, number];
}) {
  const values = rows.flat();
  const [min, max] = domain ?? [Math.min(0, ...values), Math.max(0, ...values)];
  const length = Math.max(1, ...rows.map(row => row.length));
  const x = (i: number) => 12 + 296 * i / Math.max(1, length - 1);
  const y = (v: number) => 86 - 72 * (v - min) / (max - min || 1);
  return <figure className="cnn-plot">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 320 100" role="img" aria-label={title}>
      <line x1="12" x2="308" y1={y(0)} y2={y(0)} stroke="#e8e0d6"/>
      {marks.filter(i => i >= 0 && i < length).map(i => <line key={i} x1={x(i)} x2={x(i)} y1="8" y2="91" stroke="#e7baaa" strokeWidth="3"/>)}
      {rows.map((row, c) => <g key={c} stroke={colors[c]} fill={colors[c]}>
        <polyline points={row.map((v, i) => x(i) + "," + y(v)).join(" ")} fill="none" strokeWidth="1.6"/>
        {row.length === 1 && <circle cx={x(0)} cy={y(row[0])} r="3"/>}
      </g>)}
    </svg>
    <div className="cnn-legend"><span>{min.toFixed(2)} – {max.toFixed(2)}</span>{rows.map((_, i) => <span key={i} style={{ color: colors[i] }}>{labels?.[i] ?? "Series " + (i + 1)}</span>)}</div>
  </figure>;
}

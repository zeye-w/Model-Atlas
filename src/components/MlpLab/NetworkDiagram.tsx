import type { Network, StepTrace } from "@/lib/mlp";

/** Layout in SVG units. Hidden units are spaced to fit the chosen width. */
const W = 720, H = 380, TOP = 58, BOTTOM = 350;
const X_IN = 120, X_HID = 360, X_OUT = 560;
const R = 10;

const POS = "#c96442";   // positive weights / values
const NEG = "#5f7a96";   // negative weights / values
const GRAD = "#6f8a63";  // gradient flow

export const fmt = (v: number, digits = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);

function column(n: number, x: number, spread: number) {
  const mid = (TOP + BOTTOM) / 2;
  const gap = n > 1 ? Math.min(spread, (BOTTOM - TOP) / (n - 1)) : 0;
  return Array.from({ length: n }, (_, i) => ({ x, y: mid + (i - (n - 1) / 2) * gap }));
}

function Edge({ from, to, weight, flow, flowColor, reverse, delay, run }: {
  from: { x: number; y: number }; to: { x: number; y: number }; weight: number;
  flow: boolean; flowColor: string; reverse?: boolean; delay: number; run: string;
}) {
  const width = Math.min(4.5, .5 + 2 * Math.abs(weight));
  const [a, b] = reverse ? [to, from] : [from, to];
  return <g>
    <line className="mlp-edge" x1={from.x} y1={from.y} x2={to.x} y2={to.y}
      stroke={weight >= 0 ? POS : NEG} strokeOpacity={.18 + .42 * Math.min(1, Math.abs(weight))} style={{ strokeWidth: width }}/>
    {flow && <line key={run} className="mlp-flow" x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={flowColor}
      pathLength={100} style={{ animationDelay: `${delay}ms` }}/>}
  </g>;
}

/** A small dot whose fill encodes sign and size; the number sits beside it. */
function Node({ x, y, value, scale = 1, active, side, name, note, noteColor }: {
  x: number; y: number; value: number | null; scale?: number; active: boolean;
  side: "left" | "right"; name?: string; note?: string | null; noteColor?: string;
}) {
  const a = value === null ? 0 : Math.min(1, Math.abs(value) / scale);
  const tx = side === "left" ? x - R - 8 : x + R + 8;
  const anchor = side === "left" ? "end" : "start";
  return <g className={`mlp-node${active ? " is-active" : ""}`}>
    <circle cx={x} cy={y} r={R + 5} className="mlp-node-ring"/>
    <circle cx={x} cy={y} r={R} fill="#fff" stroke="#cfc8ba" strokeWidth={1.2}/>
    {value !== null && <circle cx={x} cy={y} r={R} fill={value >= 0 ? POS : NEG} fillOpacity={.18 + .72 * a} className="mlp-node-fill"/>}
    <text x={tx} y={y + (note ? -1 : 4)} textAnchor={anchor} className="mlp-node-value">
      {name && <tspan className="mlp-node-name">{name} </tspan>}{value === null ? "" : fmt(value)}
    </text>
    {note && <text x={tx} y={y + 12} textAnchor={anchor} className="mlp-badge" fill={noteColor}>{note}</text>}
  </g>;
}

/**
 * Draws the 2–H–2 network for one phase of one SGD step. Edge colour is the
 * weight's sign and thickness its magnitude; node fill is the value's sign and
 * size. Animated dashes show which connections the current phase uses.
 */
export function NetworkDiagram({ trace, phase }: { trace: StepTrace; phase: number }) {
  const net: Network = phase >= 7 ? trace.after : trace.before;
  const hiddenCount = net.w1.length;
  const inputs = column(2, X_IN, 120);
  const hidden = column(hiddenCount, X_HID, 64);
  const outputs = column(2, X_OUT, 120);
  const forwardIn = phase === 1, forwardOut = phase === 3;
  const backOut = phase === 5, backIn = phase === 6;
  const y = trace.sample.y;
  const run = `${trace.sample.x.join()}-${phase}`;
  const stagger = 520 / (2 * hiddenCount);
  const description = [
    `Phase ${phase + 1} of 8. Network with ${hiddenCount} hidden units.`,
    `Input (${fmt(trace.sample.x[0])}, ${fmt(trace.sample.x[1])}), true class ${y === 0 ? "A" : "B"}.`,
    phase >= 2 ? `Hidden activations ${trace.h.map((v) => fmt(v)).join(", ")}.` : "",
    phase >= 4 ? `Probabilities A ${fmt(trace.p[0])}, B ${fmt(trace.p[1])}; loss ${fmt(trace.loss, 3)}.` : "",
  ].join(" ");

  return <svg className="mlp-diagram" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={description}>
    <text x={X_IN} y={22} textAnchor="middle" className="mlp-col-label">Input</text>
    <text x={X_HID} y={22} textAnchor="middle" className="mlp-col-label">Hidden · tanh · {hiddenCount} units</text>
    <text x={X_OUT} y={22} textAnchor="middle" className="mlp-col-label">Output · softmax</text>

    {hidden.map((h, j) => inputs.map((i, k) => <Edge key={`a${j}-${k}`} from={i} to={h} weight={net.w1[j][k]} run={run}
      flow={forwardIn || backIn} reverse={backIn} flowColor={backIn ? GRAD : POS} delay={(j * 2 + k) * stagger}/>))}
    {outputs.map((o, k) => hidden.map((h, j) => <Edge key={`b${k}-${j}`} from={h} to={o} weight={net.w2[k][j]} run={run}
      flow={forwardOut || backOut} reverse={backOut} flowColor={backOut ? GRAD : POS} delay={(k * hiddenCount + j) * stagger}/>))}

    {inputs.map((p, i) => <Node key={`i${i}`} {...p} side="left" name={i === 0 ? "x₁" : "x₂"} value={trace.sample.x[i]} active={phase === 0}/>)}
    {hidden.map((p, j) => <Node key={`h${j}`} {...p} side="right"
      value={phase >= 2 ? trace.h[j] : phase >= 1 ? trace.z1[j] : null}
      active={phase === 1 || phase === 2 || phase === 6}
      note={phase === 6 ? `δ ${fmt(trace.delta1[j], 3)}` : null} noteColor={GRAD}/>)}
    {outputs.map((p, k) => <Node key={`o${k}`} {...p} side="right" name={k === 0 ? "A" : "B"} scale={2}
      value={phase >= 3 ? trace.z2[k] : null}
      active={phase === 3 || phase === 4 || phase === 5}
      note={phase === 5 || phase === 6 ? `δ ${fmt(trace.delta2[k], 3)}` : null} noteColor={GRAD}/>)}

    {/* Softmax probabilities and target */}
    {phase >= 4 && outputs.map((p, k) => <g key={`p${k}`} className="mlp-prob">
      {k === y && <text x={664} y={p.y - 12} className="mlp-node-sub mlp-target">target</text>}
      <rect x={664} y={p.y - 6} width={50} height={12} rx={3} fill="#f0ede6"/>
      <rect x={664} y={p.y - 6} width={50 * trace.p[k]} height={12} rx={3} fill={k === y ? POS : "#b9b1a2"} fillOpacity={.85} className="mlp-prob-bar"/>
      <text x={664} y={p.y + 22} className="mlp-node-sub">p {fmt(trace.p[k])}</text>
    </g>)}
  </svg>;
}

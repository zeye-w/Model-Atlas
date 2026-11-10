import type { Network, StepTrace } from "@/lib/mlp";

/** Layout in SVG units. */
const W = 700, H = 380;
const inputs = [{ x: 90, y: 150 }, { x: 90, y: 260 }];
const hidden = [{ x: 330, y: 90 }, { x: 330, y: 205 }, { x: 330, y: 320 }];
const outputs = [{ x: 560, y: 150 }, { x: 560, y: 260 }];
const R_IN = 26, R_HID = 30, R_OUT = 30;

const POS = "#c96442";   // positive weights / values
const NEG = "#5f7a96";   // negative weights / values
const GRAD = "#6f8a63";  // gradient flow

export const fmt = (v: number, digits = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);

function tint(v: number, scale = 1) {
  const a = Math.min(1, Math.abs(v) / scale);
  return { fill: v >= 0 ? POS : NEG, opacity: .1 + .5 * a };
}

function Edge({ from, to, weight, flow, flowColor, reverse, delay, run }: {
  from: { x: number; y: number }; to: { x: number; y: number }; weight: number;
  flow: boolean; flowColor: string; reverse?: boolean; delay: number; run: string;
}) {
  const width = Math.min(6, .6 + 2.6 * Math.abs(weight));
  const [a, b] = reverse ? [to, from] : [from, to];
  return <g>
    <line className="mlp-edge" x1={from.x} y1={from.y} x2={to.x} y2={to.y}
      stroke={weight >= 0 ? POS : NEG} strokeOpacity={.22 + .4 * Math.min(1, Math.abs(weight))} style={{ strokeWidth: width }}/>
    {flow && <line key={run} className="mlp-flow" x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={flowColor}
      pathLength={100} style={{ animationDelay: `${delay}ms` }}/>}
  </g>;
}

function Node({ x, y, r, value, scale, sub, active, badge }: {
  x: number; y: number; r: number; value: number | null; scale?: number;
  sub: string; active: boolean; badge?: { text: string; color: string } | null;
}) {
  const t = value === null ? null : tint(value, scale);
  return <g className={`mlp-node${active ? " is-active" : ""}`}>
    <circle cx={x} cy={y} r={r + 6} className="mlp-node-ring"/>
    <circle cx={x} cy={y} r={r} fill="#fff" stroke="#d8d3c7" strokeWidth={1.4}/>
    {t && <circle cx={x} cy={y} r={r} fill={t.fill} fillOpacity={t.opacity} className="mlp-node-fill"/>}
    <text x={x} y={y - 1} textAnchor="middle" className="mlp-node-value">{value === null ? "·" : fmt(value)}</text>
    <text x={x} y={y + 15} textAnchor="middle" className="mlp-node-sub">{sub}</text>
    {badge && <g>
      <rect x={x - 34} y={y + r + 8} width={68} height={20} rx={10} fill="#fff" stroke={badge.color} strokeOpacity={.5}/>
      <text x={x} y={y + r + 22} textAnchor="middle" className="mlp-badge" fill={badge.color}>{badge.text}</text>
    </g>}
  </g>;
}

/**
 * Draws the 2–3–2 network for one phase of one SGD step. Edge colour is the
 * weight's sign and thickness its magnitude; node fill is the value's sign and
 * size. Animated dashes show which connections the current phase uses.
 */
export function NetworkDiagram({ trace, phase }: { trace: StepTrace; phase: number }) {
  const run = `${trace.sample.x.join()}-${phase}`;
  const net: Network = phase >= 7 ? trace.after : trace.before;
  const forwardIn = phase === 1, forwardOut = phase === 3;
  const backOut = phase === 5, backIn = phase === 6;
  const y = trace.sample.y;
  const description = [
    `Phase ${phase + 1} of 8.`,
    `Input (${fmt(trace.sample.x[0])}, ${fmt(trace.sample.x[1])}), true class ${y === 0 ? "A" : "B"}.`,
    phase >= 2 ? `Hidden activations ${trace.h.map((v) => fmt(v)).join(", ")}.` : "",
    phase >= 4 ? `Probabilities A ${fmt(trace.p[0])}, B ${fmt(trace.p[1])}; loss ${fmt(trace.loss, 3)}.` : "",
  ].join(" ");

  return <svg className="mlp-diagram" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={description}>
    <text x={inputs[0].x} y={24} textAnchor="middle" className="mlp-col-label">Input</text>
    <text x={hidden[0].x} y={24} textAnchor="middle" className="mlp-col-label">Hidden · tanh</text>
    <text x={outputs[0].x} y={24} textAnchor="middle" className="mlp-col-label">Output · softmax</text>

    {hidden.map((h, j) => inputs.map((i, k) => <Edge key={`a${j}${k}`} from={i} to={h} weight={net.w1[j][k]}
      flow={forwardIn || backIn} reverse={backIn} flowColor={backIn ? GRAD : POS} delay={(j * 2 + k) * 90} run={run}/>))}
    {outputs.map((o, k) => hidden.map((h, j) => <Edge key={`b${k}${j}`} from={h} to={o} weight={net.w2[k][j]}
      flow={forwardOut || backOut} reverse={backOut} flowColor={backOut ? GRAD : POS} delay={(k * 3 + j) * 90} run={run}/>))}

    {inputs.map((p, i) => <Node key={`i${i}`} {...p} r={R_IN} value={trace.sample.x[i]} sub={`x${i === 0 ? "₁" : "₂"}`} active={phase === 0}/>)}
    {hidden.map((p, j) => <Node key={`h${j}`} {...p} r={R_HID}
      value={phase >= 2 ? trace.h[j] : phase >= 1 ? trace.z1[j] : null}
      sub={`${phase === 1 ? "z" : "h"}${["₁", "₂", "₃"][j]}`}
      active={phase === 1 || phase === 2 || phase === 6}
      badge={phase >= 6 && phase < 7 ? { text: `δ ${fmt(trace.delta1[j], 3)}`, color: GRAD } : null}/>)}
    {outputs.map((p, k) => <Node key={`o${k}`} {...p} r={R_OUT}
      value={phase >= 3 ? trace.z2[k] : null} scale={2} sub={`class ${k === 0 ? "A" : "B"}`}
      active={phase === 3 || phase === 4 || phase === 5}
      badge={phase >= 5 && phase < 7 ? { text: `δ ${fmt(trace.delta2[k], 3)}`, color: GRAD } : null}/>)}

    {/* Softmax probabilities and target */}
    {phase >= 4 && outputs.map((p, k) => <g key={`p${k}`} className="mlp-prob">
      <rect x={608} y={p.y - 9} width={74} height={18} rx={4} fill="#f5f3ee"/>
      <rect x={608} y={p.y - 9} width={74 * trace.p[k]} height={18} rx={4} fill={k === y ? POS : "#b9b1a2"} fillOpacity={.85} className="mlp-prob-bar"/>
      <text x={645} y={p.y + 30} textAnchor="middle" className="mlp-node-sub">{k === y ? "target · " : ""}p {fmt(trace.p[k])}</text>
    </g>)}
  </svg>;
}

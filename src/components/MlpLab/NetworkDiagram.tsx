import { activations, hiddenLayers, type Network, type Phase, type StepTrace } from "@/lib/mlp";

/** Layout in SVG units. Columns spread between input and output; units are spaced to fit. */
const W = 720, H = 380, TOP = 58, BOTTOM = 350;
const X_IN = 96, X_OUT = 572;
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
 * Draws the network for one phase of one SGD step. Edge colour is the
 * weight's sign and thickness its magnitude; node fill is the value's sign and
 * size. Animated dashes show which weight layer the current phase uses.
 */
export function NetworkDiagram({ trace, phases, phase }: { trace: StepTrace; phases: Phase[]; phase: number }) {
  const current = phases[phase];
  const net: Network = current.kind === "update" ? trace.after : trace.before;
  const L = hiddenLayers(net);
  const at = (kind: Phase["kind"], layer?: number) => phases.findIndex((p) => p.kind === kind && (layer === undefined || p.layer === layer));
  const xs = Array.from({ length: L + 2 }, (_, c) => X_IN + (c / (L + 1)) * (X_OUT - X_IN));
  const cols = [column(2, xs[0], 120), ...trace.a.slice(1).map((a, l) => column(a.length, xs[l + 1], L > 1 ? 48 : 64)), column(2, xs[L + 1], 120)];
  const y = trace.sample.y;
  const run = `${trace.sample.x.join()}-${phase}`;

  // Which weight layer (0…L) is flowing, and in which direction.
  const flowLayer = current.kind === "sum" ? current.layer! : current.kind === "output-sum" || current.kind === "output-grad" ? L : current.kind === "backprop" ? current.layer! : -1;
  const backward = current.direction === "backward";

  /** Value shown on hidden unit j of layer l at this phase. */
  const hiddenValue = (l: number, j: number) => phase >= at("act", l) ? trace.a[l + 1][j] : phase >= at("sum", l) ? trace.z[l][j] : null;
  const description = [
    `Phase ${phase + 1} of ${phases.length}: ${current.title}. ${L} hidden layer${L > 1 ? "s" : ""} of ${trace.a[1].length} ${activations[net.activation].name} units.`,
    `Input (${fmt(trace.sample.x[0])}, ${fmt(trace.sample.x[1])}), true class ${y === 0 ? "A" : "B"}.`,
    phase >= at("softmax") ? `Probabilities A ${fmt(trace.p[0])}, B ${fmt(trace.p[1])}; loss ${fmt(trace.loss, 3)}.` : "",
  ].join(" ");

  return <svg className="mlp-diagram" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={description}>
    <text x={xs[0]} y={22} textAnchor="middle" className="mlp-col-label">Input</text>
    {Array.from({ length: L }, (_, l) => <text key={l} x={xs[l + 1]} y={22} textAnchor="middle" className="mlp-col-label">
      {L > 1 ? `Hidden ${l + 1}` : `Hidden · ${activations[net.activation].name}`}</text>)}
    <text x={xs[L + 1]} y={22} textAnchor="middle" className="mlp-col-label">Output</text>

    {net.W.map((Wl, l) => {
      const n = Wl.length * Wl[0].length, stagger = 520 / n;
      return Wl.map((row, j) => row.map((w, i) => <Edge key={`${l}-${j}-${i}`} from={cols[l][i]} to={cols[l + 1][j]} weight={w} run={run}
        flow={flowLayer === l} reverse={backward} flowColor={backward ? GRAD : POS} delay={(j * row.length + i) * stagger}/>));
    })}

    {cols[0].map((p, i) => <Node key={`i${i}`} {...p} side="left" name={i === 0 ? "x₁" : "x₂"} value={trace.sample.x[i]} active={phase === 0}/>)}
    {Array.from({ length: L }, (_, l) => cols[l + 1].map((p, j) => <Node key={`h${l}-${j}`} {...p} side="right"
      value={hiddenValue(l, j)}
      active={(current.kind === "sum" || current.kind === "act" || current.kind === "backprop") && current.layer === l}
      note={current.kind !== "update" && phase >= at("backprop", l) ? `δ ${fmt(trace.delta[l][j], 3)}` : null} noteColor={GRAD}/>))}
    {cols[L + 1].map((p, k) => <Node key={`o${k}`} {...p} side="right" name={k === 0 ? "A" : "B"} scale={2}
      value={phase >= at("output-sum") ? trace.z[L][k] : null}
      active={current.kind === "output-sum" || current.kind === "softmax" || current.kind === "output-grad"}
      note={backward ? `δ ${fmt(trace.delta[L][k], 3)}` : null} noteColor={GRAD}/>)}

    {/* Softmax probabilities and target */}
    {phase >= at("softmax") && cols[L + 1].map((p, k) => <g key={`p${k}`} className="mlp-prob">
      {k === y && <text x={664} y={p.y - 12} className="mlp-node-sub mlp-target">target</text>}
      <rect x={664} y={p.y - 6} width={50} height={12} rx={3} fill="#f0ede6"/>
      <rect x={664} y={p.y - 6} width={50 * trace.p[k]} height={12} rx={3} fill={k === y ? POS : "#b9b1a2"} fillOpacity={.85} className="mlp-prob-bar"/>
      <text x={664} y={p.y + 22} className="mlp-node-sub">p {fmt(trace.p[k])}</text>
    </g>)}
  </svg>;
}

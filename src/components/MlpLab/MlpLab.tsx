"use client";

import { useEffect, useMemo, useReducer, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, FastForward, Pause, Play, RotateCcw } from "lucide-react";
import { evaluate, forward, makeDataset, makeNetwork, phases, sampleOrder, sgdStep, type Network, type Sample, type StepTrace } from "@/lib/mlp";
import { NetworkDiagram, fmt } from "./NetworkDiagram";

interface State {
  seed: number; rate: number;
  data: Sample[];
  net: Network;           // parameters before the step being animated
  epoch: number; cursor: number; order: number[];
  updates: number;
  history: { loss: number; accuracy: number }[];
  trace: StepTrace;
  phase: number;
}

type Action =
  | { type: "reset"; seed: number; rate: number }
  | { type: "next" } | { type: "previous" }
  | { type: "finish-epoch" }
  | { type: "rate"; rate: number };

function prepare(base: Omit<State, "trace" | "phase">): State {
  const sample = base.data[base.order[base.cursor]];
  return { ...base, trace: sgdStep(base.net, sample, base.rate), phase: 0 };
}

function init(seed: number, rate: number): State {
  const data = makeDataset(seed);
  const net = makeNetwork(seed);
  return prepare({ seed, rate, data, net, epoch: 0, cursor: 0, order: sampleOrder(data.length, seed, 0), updates: 0, history: [evaluate(net, data)] });
}

/** Apply the current trace's update and load the next sample. */
function commit(state: State): Omit<State, "trace" | "phase"> {
  const net = state.trace.after;
  let { epoch, cursor, order } = state;
  cursor += 1;
  if (cursor >= order.length) { epoch += 1; cursor = 0; order = sampleOrder(state.data.length, state.seed, epoch); }
  return { ...state, net, epoch, cursor, order, updates: state.updates + 1, history: [...state.history, evaluate(net, state.data)] };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "reset": return init(action.seed, action.rate);
    case "rate": return { ...state, rate: action.rate, trace: sgdStep(state.net, state.trace.sample, action.rate) };
    case "previous": return { ...state, phase: Math.max(0, state.phase - 1) };
    case "next":
      if (state.phase < phases.length - 1) return { ...state, phase: state.phase + 1 };
      return prepare(commit(state));
    case "finish-epoch": {
      let s: State = state;
      const epoch = state.epoch;
      do { s = prepare(commit(s)); } while (s.epoch === epoch);
      return s;
    }
  }
}

const SPEEDS = [{ label: "Slow", ms: 2600 }, { label: "Normal", ms: 1500 }, { label: "Fast", ms: 700 }];

export function MlpLab({ phaseFormulas, phaseSummaries }: { phaseFormulas: ReactNode[]; phaseSummaries: string[] }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => init(42, .1));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const { trace, phase } = state;
  const current = phases[phase];
  const latest = state.history[state.history.length - 1];

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => dispatch({ type: "next" }), SPEEDS[speed].ms);
    return () => window.clearTimeout(timer);
  }, [playing, speed, state]);

  const shownNet = phase >= 7 ? trace.after : trace.before;

  return <div className="mlp-lab">
    <section className="mlp-stage">
      <div className="mlp-stage-head">
        <ol className="mlp-phases" aria-label="Phases of one training step">
          {phases.map((p, i) => <li key={p.id} className={`${i === phase ? "is-current" : ""} ${i < phase ? "is-done" : ""} dir-${p.direction}`} aria-current={i === phase ? "step" : undefined}>
            <span>{i + 1}</span>{p.title}
          </li>)}
        </ol>
      </div>
      <div className={`mlp-diagram-wrap dir-${current.direction}`}>
        <NetworkDiagram trace={trace} phase={phase}/>
      </div>
      <div className="mlp-legend">
        <span><i style={{ background: "#c96442" }}/> Positive weight / value</span>
        <span><i style={{ background: "#5f7a96" }}/> Negative</span>
        <span><i style={{ background: "#6f8a63" }}/> Gradient flow</span>
        <span>Thickness = |weight|</span>
      </div>
      <div className="mlp-transport">
        <button type="button" className="btn" onClick={() => { setPlaying(false); dispatch({ type: "previous" }); }} disabled={phase === 0}><ChevronLeft size={15}/> Back</button>
        <button type="button" className="btn btn-primary" onClick={() => setPlaying((v) => !v)}>{playing ? <><Pause size={15}/> Pause</> : <><Play size={15}/> Play</>}</button>
        <button type="button" className="btn" onClick={() => { setPlaying(false); dispatch({ type: "next" }); }}>{phase === phases.length - 1 ? "Next sample" : "Next"} <ChevronRight size={15}/></button>
        <div className="mlp-speed" role="group" aria-label="Animation speed">
          {SPEEDS.map((s, i) => <button key={s.label} type="button" aria-pressed={speed === i} onClick={() => setSpeed(i)}>{s.label}</button>)}
        </div>
      </div>
    </section>

    <section className="mlp-explain" aria-live="polite">
      <div className="mlp-explain-head">
        <span className={`mlp-dir dir-${current.direction}`}>{current.direction === "forward" ? "Forward pass" : current.direction === "backward" ? "Backward pass" : "Gradient descent"}</span>
        <h2>{phase + 1}. {current.title}</h2>
      </div>
      {phaseFormulas[phase]}
      <p>{phaseSummaries[phase]}</p>
      <PhaseNumbers trace={trace} phase={phase}/>
    </section>

    <aside className="mlp-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>What the network predicts</h2><p className="lab-panel-subtitle">Colour = P(class B) over the plane · ring = current sample</p></div>
        <DecisionMap net={shownNet} data={state.data} sample={trace.sample}/>
        <div className="metric-row" style={{ marginTop: 14 }}>
          <div className="metric"><strong>{state.updates}</strong><span>Updates</span></div>
          <div className="metric"><strong>{fmt(latest.loss, 3)}</strong><span>Mean loss</span></div>
          <div className="metric"><strong>{Math.round(latest.accuracy * 100)}%</strong><span>Accuracy</span></div>
        </div>
        <LossChart history={state.history}/>
        <p className="training-meta">Epoch {state.epoch + 1} · sample {state.cursor + 1} of {state.data.length} · {state.data.length} points, XOR layout</p>
      </section>
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Controls</h2><p className="lab-panel-subtitle">Plain SGD, one sample per update</p></div>
        <label className="slider-field"><span className="slider-label">Learning rate η <output>{state.rate.toFixed(2)}</output></span>
          <input type="range" min="0.01" max="0.5" step="0.01" value={state.rate} onChange={(e) => dispatch({ type: "rate", rate: Number(e.target.value) })}/></label>
        <label className="input-field">Random seed
          <input className="field-control" type="number" min="0" max="999999" value={state.seed}
            onChange={(e) => { setPlaying(false); dispatch({ type: "reset", seed: Math.min(999999, Math.max(0, Number(e.target.value) || 0)), rate: state.rate }); }}/></label>
        <div className="lab-controls">
          <button type="button" className="btn" onClick={() => { setPlaying(false); dispatch({ type: "finish-epoch" }); }}><FastForward size={14}/> Finish epoch</button>
          <button type="button" className="btn" onClick={() => { setPlaying(false); dispatch({ type: "reset", seed: state.seed, rate: state.rate }); }}><RotateCcw size={14}/> Reset</button>
        </div>
        <p className="training-meta">“Finish epoch” applies the remaining updates of this epoch without animation, using exactly the same computation.</p>
      </section>
    </aside>
  </div>;
}

/** "0.67·0.51 − 0.67·(−0.37) + 0.10": sign-aware sum of weight·input terms plus bias. */
function expand(weights: number[], inputs: number[], bias: number) {
  const paren = (v: number) => (v < 0 ? `(${fmt(v)})` : fmt(v));
  const terms = weights.map((w, i) => ({ neg: w < 0, text: `${Math.abs(w).toFixed(2)}·${paren(inputs[i])}` }));
  terms.push({ neg: bias < 0, text: Math.abs(bias).toFixed(2) });
  return terms.map((t, i) => (i === 0 ? (t.neg ? "−" : "") : t.neg ? " − " : " + ") + t.text).join("");
}

/** The actual numbers behind the current phase. */
function PhaseNumbers({ trace, phase }: { trace: StepTrace; phase: number }) {
  const { sample, before: n } = trace;
  const x = sample.x;
  const sub = ["₁", "₂", "₃"];
  const rows: [string, string][] = (() => {
    switch (phases[phase].id) {
      case "input": return [["x₁", fmt(x[0], 3)], ["x₂", fmt(x[1], 3)], ["True class", sample.y === 0 ? "A  (x₁·x₂ > 0)" : "B  (x₁·x₂ < 0)"]];
      case "hidden-sum": return trace.z1.map((z, j) => [`z${sub[j]}`, `${expand(n.w1[j], x, n.b1[j])} = ${fmt(z, 3)}`]);
      case "hidden-act": return trace.h.map((h, j) => [`h${sub[j]}`, `tanh(${fmt(trace.z1[j], 3)}) = ${fmt(h, 3)}`]);
      case "output-sum": return trace.z2.map((z, k) => [k === 0 ? "z_A" : "z_B", `${expand(n.w2[k], trace.h, n.b2[k])} = ${fmt(z, 3)}`]);
      case "softmax": return [["p_A", fmt(trace.p[0], 3)], ["p_B", fmt(trace.p[1], 3)], ["Loss", `−ln p_${sample.y === 0 ? "A" : "B"} = ${fmt(trace.loss, 4)}`]];
      case "output-grad": return [
        ...trace.delta2.map((d, k): [string, string] => [k === 0 ? "δ_A" : "δ_B", `${fmt(trace.p[k], 3)} − ${Number(k === sample.y)} = ${fmt(d, 3)}`]),
        ["∂L/∂W² (A)", trace.gradW2[0].map((g) => fmt(g, 3)).join("   ")],
        ["∂L/∂W² (B)", trace.gradW2[1].map((g) => fmt(g, 3)).join("   ")],
      ];
      case "hidden-grad": return [
        ...trace.delta1.map((d, j): [string, string] => [`δ${sub[j]}`, `${fmt(trace.backHidden[j], 3)} × (1 − ${fmt(trace.h[j])}²) = ${fmt(d, 4)}`]),
        ...trace.gradW1.map((row, j): [string, string] => [`∂L/∂W¹ (h${sub[j]})`, row.map((g) => fmt(g, 4)).join("   ")]),
      ];
      case "update": {
        const pairs: [string, number, number][] = [
          ...n.w1.flatMap((row, j) => row.map((w, i): [string, number, number] => [`W¹ h${sub[j]}←x${sub[i]}`, w, trace.after.w1[j][i]])),
          ...n.w2.flatMap((row, k) => row.map((w, j): [string, number, number] => [`W² ${k === 0 ? "A" : "B"}←h${sub[j]}`, w, trace.after.w2[k][j]])),
        ];
        return [
          ...pairs.map(([label, a, b]): [string, string] => [label, `${fmt(a, 3)} → ${fmt(b, 3)}`]),
          ["Loss on this sample", `${fmt(trace.loss, 4)} → ${fmt(trace.lossAfter, 4)}`],
        ];
      }
    }
  })();
  return <div className="cnn-table mlp-numbers"><table><tbody>
    {rows.map(([k, v]) => <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>)}
  </tbody></table></div>;
}

function DecisionMap({ net, data, sample }: { net: Network; data: Sample[]; sample: Sample }) {
  const N = 28, size = 300, lim = 1.2;
  const toPx = (v: number) => ((v + lim) / (2 * lim)) * size;
  const cells = useMemo(() => {
    const out: { x: number; y: number; p: number }[] = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const px = -lim + (c + .5) * (2 * lim / N), py = lim - (r + .5) * (2 * lim / N);
      out.push({ x: c, y: r, p: forward(net, [px, py]).p[1] });
    }
    return out;
  }, [net]);
  const cell = size / N;
  return <svg className="mlp-map" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Predicted probability of class B across the input plane, with training points">
    {cells.map((c) => <rect key={`${c.x}-${c.y}`} x={c.x * cell} y={c.y * cell} width={cell + .5} height={cell + .5}
      fill={c.p >= .5 ? "#c96442" : "#5f7a96"} fillOpacity={.06 + .32 * Math.abs(c.p - .5) * 2}/>)}
    <line x1={size / 2} x2={size / 2} y1={0} y2={size} stroke="#d8d3c7"/><line y1={size / 2} y2={size / 2} x1={0} x2={size} stroke="#d8d3c7"/>
    {data.map((s, i) => <circle key={i} cx={toPx(s.x[0])} cy={size - toPx(s.x[1])} r={3.6}
      fill={s.y === 1 ? "#c96442" : "#5f7a96"} stroke="#fff" strokeWidth={1.1}/>)}
    <circle cx={toPx(sample.x[0])} cy={size - toPx(sample.x[1])} r={9} fill="none" stroke="#1f1e1c" strokeWidth={1.8} className="mlp-current"/>
  </svg>;
}

function LossChart({ history }: { history: { loss: number }[] }) {
  const w = 300, h = 70, max = Math.max(...history.map((p) => p.loss), .01);
  const n = Math.max(2, history.length);
  const pts = history.map((p, i) => `${(i / (n - 1)) * w},${h - 4 - (p.loss / max) * (h - 10)}`).join(" ");
  return <figure className="mlp-loss">
    <figcaption>Mean loss over all points, after each update</figcaption>
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Mean loss history over ${history.length - 1} updates; latest ${history[history.length - 1].loss.toFixed(3)}`}>
      <line x1={0} x2={w} y1={h - 4} y2={h - 4} stroke="#e8e5dc"/>
      <polyline points={pts} fill="none" stroke="#c96442" strokeWidth={1.8} strokeLinejoin="round" vectorEffect="non-scaling-stroke"/>
      {(() => { const last = history.length - 1; return <circle cx={(last / (n - 1)) * w} cy={h - 4 - (history[last].loss / max) * (h - 10)} r={3} fill="#c96442"/>; })()}
    </svg>
  </figure>;
}

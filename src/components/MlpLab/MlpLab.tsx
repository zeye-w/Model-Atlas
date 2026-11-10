"use client";

import { useEffect, useMemo, useReducer, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, FastForward, Pause, Play, RotateCcw } from "lucide-react";
import { HIDDEN, HIDDEN_RANGE, LAYER_RANGE, activations, evaluate, forward, makeDataset, makeNetwork, makePhases, phaseTextKey, sampleOrder, sgdStep, type Activation, type Network, type Phase, type Sample, type StepTrace } from "@/lib/mlp";
import { NetworkDiagram, fmt } from "./NetworkDiagram";

interface Config { seed: number; rate: number; width: number; layers: number; activation: Activation }

interface State extends Config {
  data: Sample[];
  net: Network;           // parameters before the step being animated
  phases: Phase[];
  epoch: number; cursor: number; order: number[];
  updates: number;
  history: { loss: number; accuracy: number }[];
  /** ‖∂L/∂Wˡ‖ of recent committed updates (newest last). */
  gradHistory: number[][];
  trace: StepTrace;
  phase: number;
}

type Action =
  | { type: "reset"; config: Config }
  | { type: "next" } | { type: "previous" }
  | { type: "finish-epoch" }
  | { type: "rate"; rate: number };

const GRAD_WINDOW = 80;

function prepare(base: Omit<State, "trace" | "phase">): State {
  const sample = base.data[base.order[base.cursor]];
  return { ...base, trace: sgdStep(base.net, sample, base.rate), phase: 0 };
}

function init(config: Config): State {
  const data = makeDataset(config.seed);
  const net = makeNetwork(config.seed, Array(config.layers).fill(config.width), config.activation);
  return prepare({ ...config, data, net, phases: makePhases(config.layers), epoch: 0, cursor: 0, order: sampleOrder(data.length, config.seed, 0), updates: 0, history: [evaluate(net, data)], gradHistory: [] });
}

/** Apply the current trace's update and load the next sample. */
function commit(state: State): Omit<State, "trace" | "phase"> {
  const net = state.trace.after;
  let { epoch, cursor, order } = state;
  cursor += 1;
  if (cursor >= order.length) { epoch += 1; cursor = 0; order = sampleOrder(state.data.length, state.seed, epoch); }
  return {
    ...state, net, epoch, cursor, order, updates: state.updates + 1,
    history: [...state.history, evaluate(net, state.data)],
    gradHistory: [...state.gradHistory, state.trace.gradNorms].slice(-GRAD_WINDOW),
  };
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "reset": return init(action.config);
    case "rate": return { ...state, rate: action.rate, trace: sgdStep(state.net, state.trace.sample, action.rate) };
    case "previous": return { ...state, phase: Math.max(0, state.phase - 1) };
    case "next":
      if (state.phase < state.phases.length - 1) return { ...state, phase: state.phase + 1 };
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
const range = ([lo, hi]: readonly [number, number]) => Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);

export function MlpLab({ phaseFormulas, phaseSummaries }: { phaseFormulas: Record<string, ReactNode>; phaseSummaries: Record<string, string> }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => init({ seed: 42, rate: .1, width: HIDDEN, layers: 1, activation: "tanh" }));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const { trace, phase, phases } = state;
  const current = phases[phase];
  const textKey = phaseTextKey(current, state.activation);
  const latest = state.history[state.history.length - 1];
  const config: Config = { seed: state.seed, rate: state.rate, width: state.width, layers: state.layers, activation: state.activation };
  const reset = (patch: Partial<Config>) => { setPlaying(false); dispatch({ type: "reset", config: { ...config, ...patch } }); };

  useEffect(() => {
    if (!playing) return;
    const timer = window.setTimeout(() => dispatch({ type: "next" }), SPEEDS[speed].ms);
    return () => window.clearTimeout(timer);
  }, [playing, speed, state]);

  const shownNet = current.kind === "update" ? trace.after : trace.before;

  return <div className="mlp-lab">
    <section className="mlp-stage">
      <div className="mlp-stage-head">
        <ol className="mlp-phases" aria-label="Phases of one training step">
          {phases.map((p, i) => <li key={i} className={`${i === phase ? "is-current" : ""} ${i < phase ? "is-done" : ""} dir-${p.direction}`} aria-current={i === phase ? "step" : undefined}>
            <span>{i + 1}</span>{p.title}
          </li>)}
        </ol>
      </div>
      <div className={`mlp-diagram-wrap dir-${current.direction}`}>
        <NetworkDiagram trace={trace} phases={phases} phase={phase}/>
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
      {phaseFormulas[textKey]}
      <p>{phaseSummaries[textKey]}</p>
      <PhaseNumbers trace={trace} phase={current}/>
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
        <div className="lab-panel-heading"><h2>Gradient size by layer</h2><p className="lab-panel-subtitle">‖∂L/∂W‖ per weight layer, log scale</p></div>
        <GradientBars current={trace.gradNorms} recent={state.gradHistory}/>
      </section>
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Network</h2><p className="lab-panel-subtitle">Plain SGD, one sample per update</p></div>
        <Segmented label="Hidden layers" options={range(LAYER_RANGE).map((n) => [n, String(n)])} value={state.layers} onChange={(n) => reset({ layers: n })}/>
        <Segmented label="Units per layer" options={range(HIDDEN_RANGE).map((n) => [n, String(n)])} value={state.width} onChange={(n) => reset({ width: n })}/>
        <Segmented label="Activation" options={(Object.keys(activations) as Activation[]).map((a) => [a, activations[a].name])} value={state.activation} onChange={(a) => reset({ activation: a })}/>
        <label className="slider-field"><span className="slider-label">Learning rate η <output>{state.rate.toFixed(2)}</output></span>
          <input type="range" min="0.01" max="0.5" step="0.01" value={state.rate} onChange={(e) => dispatch({ type: "rate", rate: Number(e.target.value) })}/></label>
        <label className="input-field">Random seed
          <input className="field-control" type="number" min="0" max="999999" value={state.seed}
            onChange={(e) => reset({ seed: Math.min(999999, Math.max(0, Number(e.target.value) || 0)) })}/></label>
        <div className="lab-controls">
          <button type="button" className="btn" onClick={() => { setPlaying(false); dispatch({ type: "finish-epoch" }); }}><FastForward size={14}/> Finish epoch</button>
          <button type="button" className="btn" onClick={() => reset({})}><RotateCcw size={14}/> Reset</button>
        </div>
        <p className="training-meta">{hint(config)} Changing the network or seed starts fresh. “Finish epoch” applies the rest of this epoch without animation, using exactly the same computation.</p>
      </section>
    </aside>
  </div>;
}

/** Guidance backed by measurements on this dataset (see tests/mlp.test.ts). */
function hint(c: Config) {
  if (c.activation === "sigmoid" && c.layers > 1) return "With sigmoid and several layers, the first layer's gradient is only a few percent of the output layer's or less, and the network usually fails to learn XOR within 40 epochs. Switch to tanh or ReLU and compare the gradient bars.";
  if (c.activation === "sigmoid") return "Sigmoid's slope is at most 0.25, so even one hidden layer learns several times more slowly than tanh here.";
  if (c.activation === "relu") return "ReLU learns this task fastest, but with several layers an unlucky seed can leave units dead (always zero), stalling training.";
  if (c.width === 2 && c.layers === 1) return "With only 2 hidden units, training can get stuck on XOR for some seeds — try seed 4.";
  return "Try 3 layers with sigmoid to see vanishing gradients.";
}

function Segmented<T extends string | number>({ label, options, value, onChange }: { label: string; options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return <div className="input-field mlp-width">{label}
    <div className="mlp-segmented" role="group" aria-label={label} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map(([v, text]) => <button key={String(v)} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>)}
    </div>
  </div>;
}

/** Log-scale bars of gradient norms for each weight layer: this step, and the mean of recent updates. */
function GradientBars({ current, recent }: { current: number[]; recent: number[][] }) {
  const L = current.length;
  const mean = recent.length ? current.map((_, l) => recent.reduce((s, g) => s + g[l], 0) / recent.length) : null;
  const lo = -5, hi = 1;   // 1e-5 … 10
  const pct = (v: number) => (v <= 0 ? 0 : Math.max(0, Math.min(1, (Math.log10(v) - lo) / (hi - lo))) * 100);
  const show = (v: number) => (v === 0 ? "0" : v < .01 ? v.toExponential(1).replace("-", "−") : v.toFixed(3));
  const names = Array.from({ length: L }, (_, l) => (l === L - 1 ? `W${sup(l + 1)} → output` : l === 0 ? `W¹ input → h1` : `W${sup(l + 1)} h${l} → h${l + 1}`));
  const ratio = mean && mean[L - 1] > 0 ? mean[0] / mean[L - 1] : null;
  return <div className="mlp-grads">
    {names.map((name, l) => <div key={l} className="mlp-grad-row">
      <span className="mlp-grad-name">{name}</span>
      <div className="mlp-grad-track">
        {mean && <i className="mlp-grad-mean" style={{ width: `${pct(mean[l])}%` }}/>}
        <i className="mlp-grad-now" style={{ width: `${pct(current[l])}%` }}/>
      </div>
      <span className="mlp-grad-value">{show(current[l])}</span>
    </div>)}
    <p className="training-meta">Dark bar: this step. Light bar: mean of the last {recent.length || "—"} updates.{ratio !== null && L > 1 ? ` On average the first layer gets ${ratio < .01 ? ratio.toExponential(1).replace("-", "−") : (ratio * 100).toFixed(ratio < .1 ? 1 : 0) + "%"}${ratio < .01 ? "×" : ""} of the output layer's gradient.` : ""}</p>
  </div>;
}
const sup = (n: number) => String(n).replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]);

/** "0.67·0.51 − 0.67·(−0.37) + 0.10": sign-aware sum of weight·input terms plus bias. */
function expand(weights: number[], inputs: number[], bias: number) {
  const paren = (v: number) => (v < 0 ? `(${fmt(v)})` : fmt(v));
  const terms = weights.map((w, i) => ({ neg: w < 0, text: `${Math.abs(w).toFixed(2)}·${paren(inputs[i])}` }));
  terms.push({ neg: bias < 0, text: Math.abs(bias).toFixed(2) });
  return terms.map((t, i) => (i === 0 ? (t.neg ? "−" : "") : t.neg ? " − " : " + ") + t.text).join("");
}

/** The actual numbers behind the current phase. */
function PhaseNumbers({ trace, phase }: { trace: StepTrace; phase: Phase }) {
  const { sample, before: n } = trace;
  const L = n.W.length - 1;
  const act = activations[n.activation];
  const sub = (i: number) => String(i + 1).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[Number(d)]);
  const unit = (l: number, j: number) => (L > 1 ? `h${l + 1},${j + 1}` : `h${sub(j)}`);
  const slope = (l: number, j: number) => n.activation === "relu" ? `[${fmt(trace.z[l][j], 2)} > 0]` : n.activation === "sigmoid" ? `${fmt(trace.a[l + 1][j])}(1 − ${fmt(trace.a[l + 1][j])})` : `(1 − ${fmt(trace.a[l + 1][j])}²)`;
  const rows: [string, string][] = (() => {
    const l = phase.layer ?? 0;
    switch (phase.kind) {
      case "input": return [["x₁", fmt(sample.x[0], 3)], ["x₂", fmt(sample.x[1], 3)], ["True class", sample.y === 0 ? "A  (x₁·x₂ > 0)" : "B  (x₁·x₂ < 0)"]];
      case "sum": return trace.z[l].map((z, j) => [`z of ${unit(l, j)}`, `${expand(n.W[l][j], trace.a[l], n.b[l][j])} = ${fmt(z, 3)}`]);
      case "act": return trace.a[l + 1].map((a, j) => [unit(l, j), `${act.name}(${fmt(trace.z[l][j], 3)}) = ${fmt(a, 3)}`]);
      case "output-sum": return trace.z[L].map((z, k) => [k === 0 ? "z_A" : "z_B", `${expand(n.W[L][k], trace.a[L], n.b[L][k])} = ${fmt(z, 3)}`]);
      case "softmax": return [["p_A", fmt(trace.p[0], 3)], ["p_B", fmt(trace.p[1], 3)], ["Loss", `−ln p_${sample.y === 0 ? "A" : "B"} = ${fmt(trace.loss, 4)}`]];
      case "output-grad": return [
        ...trace.delta[L].map((d, k): [string, string] => [k === 0 ? "δ_A" : "δ_B", `${fmt(trace.p[k], 3)} − ${Number(k === sample.y)} = ${fmt(d, 3)}`]),
        ...trace.gradW[L].map((row, k): [string, string] => [`∂L/∂W (${k === 0 ? "A" : "B"})`, row.map((g) => fmt(g, 3)).join("   ")]),
      ];
      case "backprop": return trace.delta[l].map((d, j): [string, string] => [`δ of ${unit(l, j)}`,
        `${fmt(trace.back[l][j], 3)} × ${slope(l, j)} = ${fmt(d, 4)}    ∂L/∂W = (${trace.gradW[l][j].map((g) => fmt(g, 4)).join(", ")})`]);
      case "update": {
        const stats = n.W.map((Wl, l): [string, string] => {
          const deltas = Wl.flatMap((row, j) => row.map((w, i) => Math.abs(trace.after.W[l][j][i] - w)));
          const mean = deltas.reduce((s, v) => s + v, 0) / deltas.length;
          const name = l === L ? "Output weights" : L > 1 ? `Layer ${l + 1} weights` : "Hidden weights";
          return [name, `mean |Δw| ${mean < .001 ? mean.toExponential(1).replace("-", "−") : mean.toFixed(4)}   ·   largest ${Math.max(...deltas).toFixed(4)}   (${deltas.length} weights)`];
        });
        return [...stats, ["Loss on this sample", `${fmt(trace.loss, 4)} → ${fmt(trace.lossAfter, 4)}`]];
      }
    }
  })();
  return <div className="mlp-numbers"><table><tbody>
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

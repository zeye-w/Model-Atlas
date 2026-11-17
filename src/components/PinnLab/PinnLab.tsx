"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import { DEFAULT_CONFIG, DEFAULT_STEPS, INVERSE_DEFAULTS, MU_WARMUP, PinnSession, type HistoryPoint, type PinnConfig, type PinnSnapshot, type Scenario } from "@/lib/pinn/session";
import { createPinnRunner, type PinnRunner, type RunnerMode } from "@/lib/pinn/runner";
import type { PinnEvent } from "@/lib/pinn/controller";
import type { PinnFormulaId } from "@/content/pinn-formulas";

const ACCENT = "#c96442", SAGE = "#6f8a63", SLATE = "#5f7a96", GOLD = "#b08a58", MUTED = "#a69a88";
const fmt = (v: number, d = 3) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
const sci = (v: number) => {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e;
  return `${m.toFixed(1)}×10${superscript(e)}`;
};
const superscript = (n: number) => String(n).split("").map((c) => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] ?? (c === "-" ? "⁻" : c)).join("");

type Status = "idle" | "running" | "done";

const SCENARIOS: { id: Scenario; label: string; blurb: string }[] = [
  { id: "extrapolate", label: "Predict beyond the data", blurb: "Observations only cover the start. Both networks fit them; only the PINN is also told the equation everywhere else." },
  { id: "inverse", label: "Find the unknown damping", blurb: "Observations cover the whole interval but μ is unknown. The PINN treats μ as one more trainable parameter, starting from 0." },
];

export function PinnLab({ formulas }: { formulas: Record<PinnFormulaId, ReactNode> }) {
  const [config, setConfig] = useState<PinnConfig>(DEFAULT_CONFIG);
  const initial = useMemo(() => new PinnSession(config).snapshot(), [config]);
  const [snap, setSnap] = useState<PinnSnapshot>(initial);
  const [status, setStatus] = useState<Status>("idle");
  const [mode, setMode] = useState<RunnerMode>("worker");
  const [error, setError] = useState<string | null>(null);
  const runner = useRef<PinnRunner | null>(null);
  const runId = useRef(0);

  useEffect(() => {
    const r = createPinnRunner((e: PinnEvent) => {
      if (e.runId !== runId.current) return;
      if (e.type === "error") { setError(e.message); setStatus("idle"); return; }
      setSnap(e.snapshot); setStatus(e.status);
    }, setMode);
    runner.current = r;
    setMode(r.mode);
    return () => { r.dispose(); runner.current = null; };
  }, []);

  // Every settings change starts a fresh, seeded run.
  useEffect(() => {
    runId.current++;
    setSnap(initial); setStatus("idle"); setError(null);
    runner.current?.send({ type: "reset", runId: runId.current, config });
  }, [config, initial]);

  const update = (patch: Partial<PinnConfig>) => setConfig((c) => ({ ...c, ...patch }));
  const train = () => {
    const r = runner.current; if (!r) return;
    if (status === "running") { r.send({ type: "pause", runId: runId.current }); return; }
    if (snap.step >= DEFAULT_STEPS) { runId.current++; r.send({ type: "reset", runId: runId.current, config }); }
    setStatus("running");
    r.send({ type: "run", runId: runId.current, targetStep: DEFAULT_STEPS });
  };
  const reset = () => { runId.current++; setSnap(initial); setStatus("idle"); runner.current?.send({ type: "reset", runId: runId.current, config }); };

  const inverse = config.scenario === "inverse";
  const scenario = SCENARIOS.find((s) => s.id === config.scenario)!;
  const obsWindow = inverse ? INVERSE_DEFAULTS.window! : config.window;
  const progress = snap.step / DEFAULT_STEPS;

  return <div className="pinn-lab">
    <div className="reg-tabs pinn-tabs" role="tablist" aria-label="Scenario">
      {SCENARIOS.map((s) => <button key={s.id} role="tab" aria-selected={config.scenario === s.id} onClick={() => update({ scenario: s.id })}>{s.label}</button>)}
    </div>

    <section className="pinn-main lab-panel">
      <div className="lab-panel-heading reg-heading">
        <div><h2>The solution u(t)</h2><p className="lab-panel-subtitle">{scenario.blurb}</p></div>
        <div className="reg-metrics">
          <span><strong style={{ color: SLATE }}>{fmt(snap.rmseData)}</strong> data-only error</span>
          <span><strong style={{ color: "var(--accent-text)" }}>{fmt(snap.rmsePinn)}</strong> PINN error</span>
          {inverse && <span><strong>{fmt(snap.mu, 2)}</strong> μ estimate (true {snap.trueMu})</span>}
        </div>
      </div>
      <SolutionPlot snap={snap} window={obsWindow}/>
      <Legend items={[["Exact solution", SAGE, "dash"], ["Data only", SLATE, "line"], ["PINN (data + physics)", ACCENT, "line"], ["Observations", "#1f1e1c", "dot"], ["Collocation points", GOLD, "tick"]]}/>
      <div className="reg-transport">
        <button type="button" className="btn btn-primary" onClick={train} disabled={!!error}>
          {status === "running" ? <><Pause size={15}/> Pause</> : <><Play size={15}/> {snap.step >= DEFAULT_STEPS ? "Train again" : snap.step > 0 ? "Continue" : "Train both"}</>}
        </button>
        <button type="button" className="btn" onClick={reset} disabled={snap.step === 0 && status !== "running"}><RotateCcw size={14}/> Reset</button>
        <div className="pinn-progress" aria-label={`Step ${snap.step} of ${DEFAULT_STEPS}`}>
          <span>Step <strong>{snap.step.toLocaleString("en")}</strong> / {DEFAULT_STEPS.toLocaleString("en")}</span>
          <div className="pinn-track"><i style={{ width: `${progress * 100}%` }}/></div>
        </div>
      </div>
      <p className="reg-note" aria-live="polite">{error ? `Training stopped: ${error}` : note(snap, status, inverse)}</p>
    </section>

    <aside className="pinn-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Settings</h2><p className="lab-panel-subtitle">Any change restarts both networks from the same seeded weights</p></div>
        {!inverse && <Segmented label="Observed window" value={config.window} onChange={(v) => update({ window: v })} options={[[.2, "0–0.2"], [.36, "0–0.36"], [.5, "0–0.5"]]}/>}
        <Segmented label="Collocation points M" value={config.collocation} onChange={(v) => update({ collocation: v })} options={[[8, "8"], [16, "16"], [32, "32"], [64, "64"]]}/>
        <Segmented label="Physics weight λ" value={config.lambdaPhys} onChange={(v) => update({ lambdaPhys: v })} options={[[1e-5, "10⁻⁵"], [1e-4, "10⁻⁴"], [1e-3, "10⁻³"]]}/>
        <label className="slider-field"><span className="slider-label">Observation noise σ <output>{config.noise.toFixed(2)}</output></span>
          <input type="range" min={0} max={.1} step={.01} value={config.noise} onChange={(e) => update({ noise: Number(e.target.value) })}/></label>
        <label className="input-field">Seed
          <input className="field-control" type="number" min={0} max={999999} value={config.seed} onChange={(e) => update({ seed: Math.min(999999, Math.max(0, Number(e.target.value) || 0)) })}/></label>
        <button type="button" className="btn" onClick={() => setConfig((c) => ({ ...DEFAULT_CONFIG, scenario: c.scenario }))}><RotateCcw size={14}/> Defaults</button>
        <p className="training-meta">Network: tanh MLP 1 → {config.layers} × {config.hidden} → 1, Glorot initialisation, Adam (learning rate {config.lr}). {inverse ? `${snap.observations.length} observations on [0, 1]; μ starts at 0 and is held for the first ${MU_WARMUP.toLocaleString("en")} steps while the network takes shape.` : `${snap.observations.length} observations on [0, ${config.window}].`} Training runs {mode === "worker" ? "in a Web Worker" : "on the main thread (Web Workers unavailable)"}.</p>
      </section>
    </aside>

    <section className="pinn-residual lab-panel">
      <div className="lab-panel-heading"><h2>Physics residual of the PINN</h2><p className="lab-panel-subtitle">r(t) = u″ + μu′ + ku, evaluated on a fine grid · ticks: where it is penalised{inverse ? " · uses the current μ estimate" : ""}</p></div>
      <ResidualPlot snap={snap}/>
    </section>

    <section className="pinn-loss lab-panel">
      <div className="lab-panel-heading"><h2>Losses during training</h2><p className="lab-panel-subtitle">Log scale · each point is the mean over 25 steps; the spikes are Adam overshooting and recovering</p></div>
      <LossPlot history={snap.history}/>
      <Legend items={[["Data only: data loss", SLATE, "line"], ["PINN: data", ACCENT, "line"], ["PINN: physics", GOLD, "line"], ["PINN: initial condition", SAGE, "line"]]}/>
    </section>

    {inverse && <section className="pinn-mu lab-panel">
      <div className="lab-panel-heading"><h2>The damping estimate</h2><p className="lab-panel-subtitle">μ learned by gradient descent on the same loss · dashed: true value {snap.trueMu}</p></div>
      <MuPlot history={snap.history} trueMu={snap.trueMu}/>
    </section>}

    <section className="lab-panel pinn-formulas">
      <div className="lab-panel-heading"><h2>What each network minimises</h2><p className="lab-panel-subtitle">Data only: the first term. PINN: all three.</p></div>
      <div className="pinn-formula-grid">{(["ode", "data", "physics", "ic"] as const).map((id) => <div key={id}>{formulas[id]}</div>)}</div>
      <p className="training-meta">Here μ = 4 and k = 400 (δ = 2, ω₀ = 20), λ₀ = 0.1. The residual needs u′ and u″ of the network with respect to its input t; they are carried through every layer alongside the values:</p>
      {formulas.derivative}
    </section>
  </div>;
}

function note(s: PinnSnapshot, status: Status, inverse: boolean) {
  if (s.step === 0) return inverse
    ? "Both networks start from the same random weights, so their curves coincide. Press Train: the data-only network will fit the points, while the PINN must also find a μ that makes the equation hold."
    : "Both networks start from the same random weights, so their curves coincide. Press Train and watch what each does to the right of the shaded window, where there is no data.";
  const ratio = s.rmseData / Math.max(s.rmsePinn, 1e-9);
  if (inverse) {
    const off = Math.abs(s.mu - s.trueMu);
    if (status === "done") return off < .5 ? `μ settled at ${fmt(s.mu, 2)}, close to the true ${s.trueMu}. With data everywhere both networks fit the curve; only the PINN also explains it with the equation, which is what yields the parameter.` : `μ ended at ${fmt(s.mu, 2)}. Try another seed or more collocation points; inverse problems are sensitive to how the losses are balanced.`;
    return s.step < MU_WARMUP ? "μ is held at 0 while the network first fits the observations." : `μ is now ${fmt(s.mu, 2)}. Early on the network is still a poor fit, so the μ that best explains it can overshoot before coming back.`;
  }
  if (status === "done" || s.step > 4000) return ratio > 3
    ? `Inside the window both fit the data. Beyond it the data-only network has no information and drifts, while the PINN follows the oscillation: its error over [0, 1] is ${fmt(ratio, 0)}× smaller.`
    : "The PINN has not pulled clearly ahead yet. Larger λ is not always better: try 10⁻³ to see the physics term swamp the data and the fit stall near zero.";
  return "Both are fitting the observations first. The PINN is also being pushed, at every collocation point, towards curves that satisfy the equation.";
}

/* ── Plots ─────────────────────────────────────────────────────────────── */
const W = 640, PAD = { l: 40, r: 12, t: 12, b: 26 };
const sx = (t: number) => PAD.l + t * (W - PAD.l - PAD.r);
const line = (xs: number[], ys: number[], y: (v: number) => number) => xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${y(ys[i]).toFixed(1)}`).join("");

function Frame({ h, yMin, yMax, ticks, label, children, id }: { h: number; yMin: number; yMax: number; ticks: number[]; label: string; children: (y: (v: number) => number) => ReactNode; id: string }) {
  const y = (v: number) => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * (h - PAD.t - PAD.b);
  return <svg className="reg-plot" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={label}>
    <defs><clipPath id={id}><rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={h - PAD.t - PAD.b}/></clipPath></defs>
    <g className="reg-axes">
      {ticks.map((v) => <g key={v}><line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)}/><text x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{fmt(v, Math.abs(v) < 10 && v % 1 ? 1 : 0)}</text></g>)}
      {[0, .25, .5, .75, 1].map((v) => <text key={v} x={sx(v)} y={h - 8} textAnchor={v === 1 ? "end" : "middle"}>{v}</text>)}
    </g>
    <g clipPath={`url(#${id})`}>{children(y)}</g>
  </svg>;
}

function SolutionPlot({ snap, window }: { snap: PinnSnapshot; window: number }) {
  const h = 300;
  return <Frame h={h} yMin={-1.25} yMax={1.25} ticks={[-1, -.5, 0, .5, 1]} id="pinn-sol"
    label={`Solution after ${snap.step} steps: data-only RMSE ${snap.rmseData.toFixed(3)}, PINN RMSE ${snap.rmsePinn.toFixed(3)} against the exact solution`}>
    {(y) => <>
      <rect x={sx(0)} y={PAD.t} width={sx(window) - sx(0)} height={h - PAD.t - PAD.b} className="pinn-window"/>
      {window < 1 && <text x={sx(window) - 6} y={PAD.t + 14} textAnchor="end" className="pinn-caption">observed</text>}
      {window < 1 && <text x={sx(window) + 6} y={PAD.t + 14} className="pinn-caption">no data — physics only</text>}
      {snap.collocation.map((t) => <line key={t} x1={sx(t)} x2={sx(t)} y1={h - PAD.b} y2={h - PAD.b - 6} stroke={GOLD} strokeWidth={1.4}/>)}
      <path d={line(snap.grid, snap.exact, y)} className="reg-truth"/>
      <path d={line(snap.grid, snap.dataOnly, y)} fill="none" stroke={SLATE} strokeWidth={2} strokeLinejoin="round"/>
      <path d={line(snap.grid, snap.pinn, y)} fill="none" stroke={ACCENT} strokeWidth={2.6} strokeLinejoin="round"/>
      {snap.observations.map((o, i) => <circle key={i} cx={sx(o.t)} cy={y(o.u)} r={3.6} fill="#1f1e1c" stroke="#fff" strokeWidth={1.1}/>)}
    </>}
  </Frame>;
}

function niceSymmetric(values: number[]) {
  const m = Math.max(1e-6, ...values.map(Math.abs));
  const p = 10 ** Math.floor(Math.log10(m)), steps = [1, 2, 5, 10];
  const top = steps.map((s) => s * p).find((v) => v >= m) ?? 10 * p;
  return top;
}

function ResidualPlot({ snap }: { snap: PinnSnapshot }) {
  const h = 190;
  // Robust range: ignore the few largest values so one spike does not flatten the rest.
  const sorted = snap.residual.map(Math.abs).sort((a, b) => a - b);
  const top = niceSymmetric([sorted[Math.floor(sorted.length * .97)]]);
  const rms = Math.sqrt(snap.residual.reduce((s, r) => s + r * r, 0) / snap.residual.length);
  return <>
    <Frame h={h} yMin={-top} yMax={top} ticks={[-top, 0, top]} id="pinn-res" label={`Physics residual of the PINN; root mean square ${rms.toFixed(2)}`}>
      {(y) => <>
        {snap.collocation.map((t) => <line key={t} x1={sx(t)} x2={sx(t)} y1={h - PAD.b} y2={h - PAD.b - 6} stroke={GOLD} strokeWidth={1.4}/>)}
        <line x1={PAD.l} x2={W - PAD.r} y1={y(0)} y2={y(0)} stroke="#cfc8ba"/>
        <path d={`${line(snap.grid, snap.residual, y)}L${sx(1)},${y(0)}L${sx(0)},${y(0)}Z`} fill={ACCENT} fillOpacity={.12} stroke="none"/>
        <path d={line(snap.grid, snap.residual, y)} fill="none" stroke={ACCENT} strokeWidth={1.8}/>
      </>}
    </Frame>
    <p className="training-meta">RMS residual on the grid: <strong>{fmt(rms, 2)}</strong>. For scale, u″ and ku each swing through about ±400 here, so the residual is the small difference left when they nearly cancel. Between widely spaced collocation points the residual is unconstrained — try M = 8.</p>
  </>;
}

function LossPlot({ history }: { history: HistoryPoint[] }) {
  const h = 210;
  const series: [string, number[]][] = [
    [SLATE, history.map((p) => p.dataOnly)], [ACCENT, history.map((p) => p.pinn.data)],
    [GOLD, history.map((p) => p.pinn.physics)], [SAGE, history.map((p) => p.pinn.ic)],
  ];
  const logs = series.flatMap(([, v]) => v).filter((v) => v > 0).map(Math.log10);
  const lo = Math.max(-9, Math.floor(Math.min(...logs, -1))), hi = Math.ceil(Math.max(...logs, 0));
  const last = Math.max(DEFAULT_STEPS, history[history.length - 1]?.step ?? 0);
  const x = (s: number) => PAD.l + (s / last) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (Math.log10(Math.max(v, 10 ** lo)) - lo) / (hi - lo)) * (h - PAD.t - PAD.b);
  const ticks = Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).filter((_, i, a) => a.length <= 6 || i % 2 === (a.length - 1) % 2);
  return <svg className="reg-plot" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={`Loss history over ${last} steps`}>
    <g className="reg-axes">
      {ticks.map((e) => <g key={e}><line x1={PAD.l} x2={W - PAD.r} y1={y(10 ** e)} y2={y(10 ** e)}/><text x={PAD.l - 6} y={y(10 ** e) + 4} textAnchor="end">10{superscript(e)}</text></g>)}
      {[0, .25, .5, .75, 1].map((f) => <text key={f} x={x(f * last)} y={h - 8} textAnchor={f === 1 ? "end" : "middle"}>{Math.round(f * last).toLocaleString("en")}</text>)}
    </g>
    {history.length > 1 && series.map(([color, v]) => <polyline key={color} fill="none" stroke={color} strokeWidth={1.7} strokeLinejoin="round"
      points={history.map((p, i) => `${x(p.step).toFixed(1)},${y(v[i]).toFixed(1)}`).join(" ")}/>)}
    {history.length > 1 && (() => { const p = history[history.length - 1]; return <text x={W - PAD.r} y={PAD.t + 10} textAnchor="end" className="pinn-caption">physics {sci(p.pinn.physics)} · data {sci(p.pinn.data)}</text>; })()}
  </svg>;
}

function MuPlot({ history, trueMu }: { history: HistoryPoint[]; trueMu: number }) {
  const h = 170;
  const top = Math.max(10, Math.ceil(Math.max(...history.map((p) => p.mu)) / 10) * 10);
  const last = DEFAULT_STEPS;
  const x = (s: number) => PAD.l + (s / last) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - v / top) * (h - PAD.t - PAD.b);
  const ticks = [0, top / 2, top];
  return <svg className="reg-plot" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={`Estimated damping coefficient; latest ${history[history.length - 1].mu.toFixed(2)}, true ${trueMu}`}>
    <g className="reg-axes">
      {ticks.map((v) => <g key={v}><line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)}/><text x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{v}</text></g>)}
      {[0, .25, .5, .75, 1].map((f) => <text key={f} x={x(f * last)} y={h - 8} textAnchor={f === 1 ? "end" : "middle"}>{Math.round(f * last).toLocaleString("en")}</text>)}
    </g>
    <line x1={PAD.l} x2={W - PAD.r} y1={y(trueMu)} y2={y(trueMu)} className="reg-truth"/>
    <rect x={x(0)} y={PAD.t} width={x(MU_WARMUP) - x(0)} height={h - PAD.t - PAD.b} fill={MUTED} fillOpacity={.1}/>
    <polyline fill="none" stroke={ACCENT} strokeWidth={2} strokeLinejoin="round" points={history.map((p) => `${x(p.step).toFixed(1)},${y(Math.min(p.mu, top)).toFixed(1)}`).join(" ")}/>
  </svg>;
}

function Legend({ items }: { items: [string, string, "line" | "dash" | "dot" | "tick"][] }) {
  return <ul className="pinn-legend">
    {items.map(([text, color, kind]) => <li key={text}>
      <svg width="22" height="12" aria-hidden="true">
        {kind === "line" && <line x1={1} x2={21} y1={6} y2={6} stroke={color} strokeWidth={2.4}/>}
        {kind === "dash" && <line x1={1} x2={21} y1={6} y2={6} stroke={color} strokeWidth={1.8} strokeDasharray="5 3"/>}
        {kind === "dot" && <circle cx={11} cy={6} r={3.6} fill={color}/>}
        {kind === "tick" && <line x1={11} x2={11} y1={2} y2={11} stroke={color} strokeWidth={1.8}/>}
      </svg>{text}
    </li>)}
  </ul>;
}

function Segmented<T extends number>({ label, options, value, onChange }: { label: string; options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return <div className="input-field mlp-width">{label}
    <div className="mlp-segmented" role="group" aria-label={label} style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      {options.map(([v, text]) => <button key={String(v)} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>)}
    </div>
  </div>;
}

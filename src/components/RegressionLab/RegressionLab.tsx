"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { makeRegressionData, truth, type Point, type RegressionData } from "@/lib/ensembles/data";
import { algorithmIds, algorithms, fitAlgorithm, type AlgorithmId, type FitResult } from "@/lib/ensembles/models";

type Params = Record<AlgorithmId, Record<string, number>>;
const defaults = () => Object.fromEntries(algorithmIds.map((id) => [id, { ...algorithms[id].defaults }])) as unknown as Params;

const ACCENT = "#c96442", SAGE = "#6f8a63", SLATE = "#5f7a96", INK = "#1f1e1c", MUTED = "#a69a88";
const fmt = (v: number, d = 3) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

/** Honest notes on what each implementation does and leaves out. */
const notes: Record<AlgorithmId, string> = {
  "adaboost": "AdaBoost.R2 (Drucker, 1997) with linear loss and CART base learners. Trees are fitted on weighted data; scikit-learn resamples by weight instead. Point size shows each example's current weight.",
  "gradient-boosting": "Squared-error loss, so each tree is fitted to the plain residuals y − F(x). CART trees, no row subsampling.",
  "random-forest": "Bagging of CART trees on bootstrap samples. With a single input feature there is nothing to subsample at each split, so this is the bagging half of a random forest. Faint lines are individual trees.",
  "catboost": "A re-implementation of CatBoost's default CPU scheme for RMSE: plain boosting, symmetric trees, L2-regularised leaves. Not included: ordered boosting, feature quantisation, random_strength, Bayesian bootstrap and categorical target statistics (there are no categorical features here). On one feature a symmetric tree of depth d can make at most d + 1 different predictions.",
  "rbf-network": "Gaussian units with centres evenly spaced over [0, 1] and σ = width ÷ units; output weights and bias by ridge least squares. Each step is a separately solved network with one more unit. Faint curves are each unit's contribution wₖφₖ(x).",
};

export function RegressionLab() {
  const [seed, setSeed] = useState(42);
  const [noise, setNoise] = useState(.2);
  const [algo, setAlgo] = useState<AlgorithmId>("gradient-boosting");
  const [params, setParams] = useState<Params>(defaults);
  const [stageIndex, setStageIndex] = useState<number | null>(null);   // null = last stage
  const [playing, setPlaying] = useState(false);
  // Fit times differ between the server render and the browser; show them only after hydration.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Deep links from model pages: /lab/regression?model=catboost
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("model");
    if (id && (algorithmIds as string[]).includes(id)) setAlgo(id as AlgorithmId);
  }, []);
  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("model", algo);
    window.history.replaceState(null, "", url);
  }, [algo]);

  const data = useMemo(() => makeRegressionData(seed, noise), [seed, noise]);
  const deferredParams = useDeferredValue(params);
  const deferredData = useDeferredValue(data);
  const results = useMemo(() => Object.fromEntries(algorithmIds.map((id) => {
    const t = performance.now();
    const r = fitAlgorithm(id, deferredData, deferredParams[id]);
    return [id, { ...r, ms: performance.now() - t }];
  })) as Record<AlgorithmId, FitResult & { ms: number }>, [deferredData, deferredParams]);
  const result = results[algo];
  const last = result.stages.length - 1;
  const k = stageIndex === null ? last : Math.min(stageIndex, last);
  const current = result.stages[k];
  const spec = algorithms[algo];

  useEffect(() => {
    if (!playing) return;
    if (k >= last) { setPlaying(false); return; }
    const step = Math.max(1, Math.round(last / 60));
    const t = window.setTimeout(() => setStageIndex(Math.min(last, k + step)), 60);
    return () => window.clearTimeout(t);
  }, [playing, k, last]);

  const choose = (id: AlgorithmId) => { setAlgo(id); setStageIndex(null); setPlaying(false); };
  const setParam = (key: string, value: number) => { setParams((p) => ({ ...p, [algo]: { ...p[algo], [key]: value } })); setStageIndex(null); };
  const best = result.stages.reduce((b, s, i) => (s.test < result.stages[b].test ? i : b), 0);

  return <div className="reg-lab">
    <div className="reg-tabs" role="tablist" aria-label="Algorithm">
      {algorithmIds.map((id) => <button key={id} role="tab" aria-selected={algo === id} onClick={() => choose(id)}>{algorithms[id].name}</button>)}
    </div>

    <section className="reg-main lab-panel">
      <div className="lab-panel-heading reg-heading">
        <div><h2>{spec.name}: the fitted curve</h2><p className="lab-panel-subtitle">{data.train.length} training points (filled) · {data.test.length} held-out points (open) · dashed: true function</p></div>
        <div className="reg-metrics">
          <span><strong>{fmt(current.train)}</strong> train RMSE</span>
          <span><strong>{fmt(current.test)}</strong> test RMSE</span>
        </div>
      </div>
      <FitPlot data={data} result={result} k={k} algo={algo}/>
      <div className="reg-transport">
        <button type="button" className="btn" aria-label="Previous" onClick={() => { setPlaying(false); setStageIndex(Math.max(0, k - 1)); }} disabled={k === 0}><ChevronLeft size={15}/></button>
        <button type="button" className="btn btn-primary" onClick={() => { if (k >= last) setStageIndex(0); setPlaying((v) => !v); }}>{playing ? <><Pause size={15}/> Pause</> : <><Play size={15}/> {k >= last ? "Replay" : "Play"}</>}</button>
        <button type="button" className="btn" aria-label="Next" onClick={() => { setPlaying(false); setStageIndex(Math.min(last, k + 1)); }} disabled={k === last}><ChevronRight size={15}/></button>
        <label className="reg-scrub">
          <span>{spec.stageNoun === "unit count" ? "Units" : spec.stageNoun === "tree" ? "Trees" : "Round"} <strong>{k + 1}</strong> / {last + 1}</span>
          <input type="range" min={0} max={last} value={k} onChange={(e) => { setPlaying(false); setStageIndex(Number(e.target.value)); }} aria-label="Stage"/>
        </label>
      </div>
      <p className="reg-note" aria-live="polite">{current.note}{result.stoppedEarly && k === last ? ` ${result.stoppedEarly}` : ""}</p>
    </section>

    <section className="reg-detail lab-panel">
      <div className="lab-panel-heading"><h2>{detailTitle(algo)}</h2><p className="lab-panel-subtitle">{detailSubtitle(algo)}</p></div>
      <DetailPlot data={data} result={result} k={k} algo={algo} params={params[algo]}/>
    </section>

    <section className="reg-errors lab-panel">
      <div className="lab-panel-heading"><h2>Error as the model grows</h2><p className="lab-panel-subtitle">Lowest test error at {spec.stageNoun === "unit count" ? "" : `${spec.stageNoun} `}{best + 1}{spec.stageNoun === "unit count" ? " units" : ""} ({fmt(result.stages[best].test)})</p></div>
      <ErrorPlot result={result} k={k}/>
    </section>

    <aside className="reg-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>{spec.name} settings</h2></div>
        {spec.params.map((p) => <label className="slider-field" key={p.key}>
          <span className="slider-label">{p.label} <output>{params[algo][p.key]}</output></span>
          <input type="range" min={p.min} max={p.max} step={p.step} value={params[algo][p.key]} onChange={(e) => setParam(p.key, Number(e.target.value))}/>
        </label>)}
        <button type="button" className="btn" onClick={() => { setParams((p) => ({ ...p, [algo]: { ...algorithms[algo].defaults } as Record<string, number> })); setStageIndex(null); }}><RotateCcw size={14}/> Defaults</button>
        <p className="training-meta">{notes[algo]}</p>
      </section>
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Data</h2><p className="lab-panel-subtitle">Shared by all five algorithms</p></div>
        <label className="slider-field"><span className="slider-label">Noise σ <output>{noise.toFixed(2)}</output></span>
          <input type="range" min={0} max={.6} step={.01} value={noise} onChange={(e) => { setNoise(Number(e.target.value)); setStageIndex(null); }}/></label>
        <label className="input-field">Data seed
          <input className="field-control" type="number" min={0} max={999999} value={seed} onChange={(e) => { setSeed(Math.min(999999, Math.max(0, Number(e.target.value) || 0))); setStageIndex(null); }}/></label>
        <p className="training-meta">y = 0.9 sin(2π·1.15x) + 0.55·[x &gt; 0.62] + 0.25x + noise. The step favours trees; the smooth wave favours radial basis functions.</p>
      </section>
    </aside>

    <section className="reg-compare lab-panel">
      <div className="lab-panel-heading"><h2>All five on the same data</h2><p className="lab-panel-subtitle">Each at its current settings · predicting the mean gives test RMSE {fmt(result.baseline.test)}</p></div>
      <div className="reg-table"><table>
        <thead><tr><th scope="col">Algorithm</th><th scope="col">Train RMSE</th><th scope="col">Test RMSE</th><th scope="col">Best test RMSE</th><th scope="col">Fit time</th></tr></thead>
        <tbody>{algorithmIds.map((id) => {
          const r = results[id], end = r.stages[r.stages.length - 1];
          const b = r.stages.reduce((m, s) => Math.min(m, s.test), Infinity);
          const bestAll = Math.min(...algorithmIds.map((j) => results[j].stages[results[j].stages.length - 1].test));
          return <tr key={id} className={id === algo ? "is-current" : ""}>
            <th scope="row"><button type="button" onClick={() => choose(id)}>{algorithms[id].name}</button></th>
            <td>{fmt(end.train)}</td>
            <td className={end.test === bestAll ? "is-best" : ""}>{fmt(end.test)}</td>
            <td>{fmt(b)}</td>
            <td>{mounted ? `${r.ms < 1 ? "<1" : Math.round(r.ms)} ms` : "—"}</td>
          </tr>;
        })}</tbody>
      </table></div>
      <p className="training-meta">RMSE = root-mean-square error. “Best test RMSE” is the lowest held-out error at any round; picking that round is a form of early stopping that peeks at the test set, so treat it as a diagnostic, not a score. Fit times are measured in your browser.</p>
    </section>
  </div>;
}

function detailTitle(algo: AlgorithmId) {
  return { "adaboost": "Sample weights after this round", "gradient-boosting": "Residuals and the tree fitted to them", "random-forest": "This tree’s bootstrap sample", "catboost": "Residuals and the symmetric tree", "rbf-network": "Hidden units" }[algo];
}
function detailSubtitle(algo: AlgorithmId) {
  return {
    "adaboost": "Bars: weight of each training point · line: this round’s tree",
    "gradient-boosting": "Dots: y − F(x) before this round · line: the tree, before shrinkage",
    "random-forest": "Bars: how many times each point was drawn · line: this tree",
    "catboost": "Dots: residuals before this round · line: the tree, before the learning rate",
    "rbf-network": "Each Gaussian unit scaled by its output weight · ticks: centres",
  }[algo];
}

/* ── Plot geometry ─────────────────────────────────────────────────────── */
const PW = 640, PH = 300, PAD = { l: 40, r: 12, t: 12, b: 26 };
function scales(yMin: number, yMax: number, h = PH) {
  const x = (v: number) => PAD.l + v * (PW - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * (h - PAD.t - PAD.b);
  return { x, y };
}
const path = (xs: number[], ys: number[], sx: (v: number) => number, sy: (v: number) => number) =>
  xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join("");

function Axes({ yMin, yMax, h = PH, ticks = 4 }: { yMin: number; yMax: number; h?: number; ticks?: number }) {
  const { x, y } = scales(yMin, yMax, h);
  const ys = Array.from({ length: ticks + 1 }, (_, i) => yMin + (i / ticks) * (yMax - yMin));
  return <g className="reg-axes">
    {ys.map((v) => <g key={v}><line x1={PAD.l} x2={PW - PAD.r} y1={y(v)} y2={y(v)}/><text x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{fmt(v, 1)}</text></g>)}
    {[0, .25, .5, .75, 1].map((v) => <text key={v} x={x(v)} y={h - 8} textAnchor="middle">{v}</text>)}
  </g>;
}

function yRange(points: Point[]) {
  const ys = points.map((p) => p.y);
  const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * .08;
  return [Math.floor((lo - pad) * 2) / 2, Math.ceil((hi + pad) * 2) / 2] as const;
}

function FitPlot({ data, result, k, algo }: { data: RegressionData; result: FitResult; k: number; algo: AlgorithmId }) {
  const [yMin, yMax] = yRange([...data.train, ...data.test]);
  const { x, y } = scales(yMin, yMax);
  const stage = result.stages[k];
  const weights = algo === "adaboost" ? stage.pointValues : undefined;
  const wMax = weights ? Math.max(...weights) : 1;
  const trueCurve = data.grid.map(truth);
  return <svg className="reg-plot" viewBox={`0 0 ${PW} ${PH}`} role="img"
    aria-label={`${algorithms[algo].name} after stage ${k + 1}: train RMSE ${stage.train.toFixed(3)}, test RMSE ${stage.test.toFixed(3)}`}>
    <defs><clipPath id="reg-clip"><rect x={PAD.l} y={PAD.t} width={PW - PAD.l - PAD.r} height={PH - PAD.t - PAD.b}/></clipPath></defs>
    <Axes yMin={yMin} yMax={yMax}/>
    <g clipPath="url(#reg-clip)">
      {algo === "random-forest" && result.stages.slice(Math.max(0, k - 24), k + 1).map((s, i) =>
        <path key={i} d={path(data.grid, s.parts[0], x, y)} className="reg-faint" stroke={SAGE}/>)}
      {algo === "rbf-network" && stage.parts.map((part, j) =>
        <path key={j} d={path(data.grid, part, x, y)} className="reg-faint" stroke={SLATE}/>)}
      <path d={path(data.grid, trueCurve, x, y)} className="reg-truth"/>
      {data.test.map((p, i) => <circle key={`t${i}`} cx={x(p.x)} cy={y(p.y)} r={3.4} className="reg-test"/>)}
      {data.train.map((p, i) => <circle key={`r${i}`} cx={x(p.x)} cy={y(p.y)}
        r={weights ? 2 + 7 * Math.sqrt(weights[i] / wMax) : 3.4} className="reg-train" style={weights ? { fill: ACCENT, fillOpacity: .45 } : undefined}/>)}
      <path d={path(data.grid, stage.curve, x, y)} className="reg-curve"/>
    </g>
  </svg>;
}

function DetailPlot({ data, result, k, algo, params }: { data: RegressionData; result: FitResult; k: number; algo: AlgorithmId; params: Record<string, number> }) {
  const h = 210;
  const stage = result.stages[k];
  const xs = data.train.map((p) => p.x);

  if (algo === "adaboost" || algo === "random-forest") {
    const v = stage.pointValues!;
    const vMax = Math.max(...v);
    const [yMin, yMax] = yRange(data.train);
    const { x, y } = scales(yMin, yMax, h);
    const barBase = h - PAD.b, barTop = PAD.t + 6;
    return <svg className="reg-plot" viewBox={`0 0 ${PW} ${h}`} role="img" aria-label={detailTitle(algo)}>
      <Axes yMin={yMin} yMax={yMax} h={h} ticks={3}/>
      {xs.map((px, i) => <rect key={i} x={x(px) - 2} width={4} y={barBase - (v[i] / vMax) * (barBase - barTop)} height={(v[i] / vMax) * (barBase - barTop)}
        fill={algo === "adaboost" ? ACCENT : SAGE} fillOpacity={.35}/>)}
      {data.train.map((p, i) => <circle key={i} cx={x(p.x)} cy={y(p.y)} r={2.6} fill={MUTED}/>)}
      <path d={path(data.grid, stage.parts[0], x, y)} className="reg-curve reg-step"/>
      <text x={PW - PAD.r} y={PAD.t + 4} textAnchor="end" className="reg-caption">
        {algo === "adaboost" ? `largest weight ${fmt(vMax, 4)} (uniform = ${fmt(1 / xs.length, 4)})` : `${v.filter((c) => c === 0).length} points left out of this tree (out-of-bag)`}
      </text>
    </svg>;
  }

  if (algo === "rbf-network") {
    const all = stage.parts.flat();
    const lo = Math.min(0, ...all), hi = Math.max(0, ...all), pad = (hi - lo) * .1 || 1;
    const { x, y } = scales(lo - pad, hi + pad, h);
    const units = stage.parts.length;
    return <svg className="reg-plot" viewBox={`0 0 ${PW} ${h}`} role="img" aria-label={`${units} Gaussian units scaled by their weights`}>
      <Axes yMin={lo - pad} yMax={hi + pad} h={h} ticks={3}/>
      <line x1={PAD.l} x2={PW - PAD.r} y1={y(0)} y2={y(0)} stroke="#d8d3c7"/>
      {stage.parts.map((part, j) => <path key={j} d={path(data.grid, part, x, y)} fill="none" stroke={j % 2 ? SLATE : ACCENT} strokeWidth={1.6} strokeOpacity={.85}/>)}
      {Array.from({ length: units }, (_, j) => (j + .5) / units).map((c, j) => <line key={j} x1={x(c)} x2={x(c)} y1={h - PAD.b} y2={h - PAD.b + 6} stroke={INK}/>)}
    </svg>;
  }

  // Boosting: residuals before this round and the tree fitted to them (undo the learning rate).
  const residuals = stage.pointValues!;
  const lr = params.learningRate;
  const tree = stage.parts[0].map((v) => v / lr);
  const lim = Math.max(.3, ...residuals.map(Math.abs), ...tree.map(Math.abs)) * 1.1;
  const { x, y } = scales(-lim, lim, h);
  return <svg className="reg-plot" viewBox={`0 0 ${PW} ${h}`} role="img" aria-label="Residuals before this round and the tree fitted to them">
    <Axes yMin={-lim} yMax={lim} h={h} ticks={4}/>
    <line x1={PAD.l} x2={PW - PAD.r} y1={y(0)} y2={y(0)} stroke="#cfc8ba"/>
    {xs.map((px, i) => <line key={`s${i}`} x1={x(px)} x2={x(px)} y1={y(0)} y2={y(residuals[i])} stroke={MUTED} strokeOpacity={.5}/>)}
    {xs.map((px, i) => <circle key={i} cx={x(px)} cy={y(residuals[i])} r={2.6} fill={MUTED}/>)}
    <path d={path(data.grid, tree, x, y)} className="reg-curve reg-step"/>
    <text x={PW - PAD.r} y={PAD.t + 4} textAnchor="end" className="reg-caption">residual RMS {fmt(Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / residuals.length))}</text>
  </svg>;
}

function ErrorPlot({ result, k }: { result: FitResult; k: number }) {
  const h = 200, n = result.stages.length;
  const vals = result.stages.flatMap((s) => [s.train, s.test]).concat(result.baseline.test);
  const hi = Math.max(...vals) * 1.05, lo = Math.max(0, Math.min(...vals) * .9);
  const { y } = scales(lo, hi, h);
  const x = (i: number) => PAD.l + (n === 1 ? .5 : i / (n - 1)) * (PW - PAD.l - PAD.r);
  const line = (key: "train" | "test") => result.stages.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(s[key]).toFixed(1)}`).join("");
  return <svg className="reg-plot" viewBox={`0 0 ${PW} ${h}`} role="img" aria-label={`Train and test RMSE over ${n} stages`}>
    <g className="reg-axes">
      {[0, .5, 1].map((f) => { const v = lo + f * (hi - lo); return <g key={f}><line x1={PAD.l} x2={PW - PAD.r} y1={y(v)} y2={y(v)}/><text x={PAD.l - 6} y={y(v) + 4} textAnchor="end">{fmt(v, 2)}</text></g>; })}
      <text x={PAD.l} y={h - 8}>1</text><text x={PW - PAD.r} y={h - 8} textAnchor="end">{n}</text>
    </g>
    <line x1={PAD.l} x2={PW - PAD.r} y1={y(result.baseline.test)} y2={y(result.baseline.test)} stroke={MUTED} strokeDasharray="4 4"/>
    <text x={PW - PAD.r} y={y(result.baseline.test) - 5} textAnchor="end" className="reg-caption">predict the mean</text>
    <path d={line("train")} fill="none" stroke={SAGE} strokeWidth={2}/>
    <path d={line("test")} fill="none" stroke={ACCENT} strokeWidth={2}/>
    <line x1={x(k)} x2={x(k)} y1={PAD.t} y2={h - PAD.b} stroke={INK} strokeOpacity={.4}/>
    <circle cx={x(k)} cy={y(result.stages[k].test)} r={4} fill={ACCENT}/>
    <circle cx={x(k)} cy={y(result.stages[k].train)} r={4} fill={SAGE}/>
    <g className="reg-legend" transform={`translate(${PAD.l + 8},${PAD.t + 12})`}>
      <rect width={10} height={3} y={-4} fill={SAGE}/><text x={16}>train</text>
      <rect x={64} width={10} height={3} y={-4} fill={ACCENT}/><text x={80}>test</text>
    </g>
  </svg>;
}

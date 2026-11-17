"use client";

import { useMemo, useRef, useState, type PointerEvent } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { accuracy, makeClassData, type DatasetKind } from "@/lib/classify/data";
import { applyTransform, identity, mapPoints, predictKnn, rankNeighbors, standardizer, stretch, vote, type Metric, type Weighting } from "@/lib/classify/knn";
import { AccuracyChart, Axes, CLASS_COLORS, CLASS_NAMES, DataControls, Legend, Points, SIZE, Segmented, fmt, pct, toData, toScreen, type Frame } from "./common";

const STEPS = [
  { title: "Measure distances", text: "Compute the distance from the query to every training point." },
  { title: "Take the k nearest", text: "Sort by distance and keep the k closest. The outline marks the distance of the k-th neighbour: everything inside it is in the neighbourhood." },
  { title: "Vote", text: "Each neighbour votes for its class — equally, or with weight 1 / distance so closer points count more. Ties go to the class of the nearest tied neighbour." },
  { title: "Decision regions", text: "Repeating the vote at every location of the plane colours it by predicted class. The boundary is where the vote changes." },
];
const KS = Array.from({ length: 30 }, (_, i) => i + 1);
const STRETCH = 10;

export function KnnLab() {
  const [kind, setKind] = useState<DatasetKind>("moons");
  const [seed, setSeed] = useState(2);
  const [noise, setNoise] = useState(.25);
  const [k, setK] = useState(7);
  const [metric, setMetric] = useState<Metric>("euclidean");
  const [weighting, setWeighting] = useState<Weighting>("uniform");
  const [stretched, setStretched] = useState(false);
  const [standardize, setStandardize] = useState(false);
  const [step, setStep] = useState(0);
  const [query, setQuery] = useState<[number, number]>([.35, .2]);   // in the raw (unstretched) plane
  const svgRef = useRef<SVGSVGElement>(null);

  const data = useMemo(() => makeClassData(kind, seed, noise), [kind, seed, noise]);
  const f = stretched ? STRETCH : 1;
  // "Measured" space: what a user would record (second feature possibly in other units).
  const measured = useMemo(() => ({ train: stretch(data.train, f, 0), test: stretch(data.test, f, 0) }), [data, f]);
  const transform = useMemo(() => (standardize ? standardizer(measured.train) : identity), [standardize, measured]);
  // Model space: where distances are computed.
  const model = useMemo(() => ({ train: mapPoints(transform, measured.train), test: mapPoints(transform, measured.test) }), [transform, measured]);
  const frame: Frame = { x: [-2 * f, 2 * f], y: [-2, 2] };
  const q: [number, number] = [query[0] * f, query[1]];
  const qModel = applyTransform(transform, q);
  const options = { k, metric, weighting };
  const ranked = useMemo(() => rankNeighbors(model.train, qModel, metric), [model, qModel[0], qModel[1], metric]); // eslint-disable-line react-hooks/exhaustive-deps
  const result = vote(ranked, data.classes, options);
  const radius = result.neighbors[result.neighbors.length - 1]?.distance ?? 0;
  const near = new Set(result.neighbors.map((n) => n.index));

  const curve = useMemo(() => {
    const train: number[] = [], test: number[] = [];
    for (const kk of KS) {
      const o = { k: kk, metric, weighting };
      train.push(accuracy((x) => predictKnn(model.train, data.classes, x, o), model.train));
      test.push(accuracy((x) => predictKnn(model.train, data.classes, x, o), model.test));
    }
    return { train, test };
  }, [model, data.classes, metric, weighting]);
  const kIndex = k - 1;

  const grid = useMemo(() => {
    if (step < 3) return null;
    const N = 48, cells: { i: number; j: number; c: number }[] = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const p: [number, number] = [frame.x[0] + (i + .5) * (frame.x[1] - frame.x[0]) / N, frame.y[0] + (j + .5) * (frame.y[1] - frame.y[0]) / N];
      cells.push({ i, j, c: predictKnn(model.train, data.classes, applyTransform(transform, p), options) });
    }
    return { N, cells };
  }, [step, model, transform, data.classes, k, metric, weighting, f]); // eslint-disable-line react-hooks/exhaustive-deps

  const move = (e: PointerEvent<SVGSVGElement>) => {
    if (e.type === "pointermove" && e.buttons !== 1) return;
    const svg = svgRef.current!, rect = svg.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) / rect.width) * SIZE, sy = ((e.clientY - rect.top) / rect.height) * SIZE;
    const [x, y] = toData(frame, sx, sy);
    setQuery([Math.max(-2, Math.min(2, x / f)), Math.max(-2, Math.min(2, y))]);
  };

  // Neighbourhood outline: points at model-space distance `radius` from the query, mapped back to the plot.
  const outline = Array.from({ length: 96 }, (_, a) => {
    const th = (a / 96) * 2 * Math.PI;
    let dx = Math.cos(th), dy = Math.sin(th);
    if (metric === "manhattan") { const s = Math.abs(dx) + Math.abs(dy); dx /= s; dy /= s; }
    const p = [q[0] + dx * radius * transform.scale[0], q[1] + dy * radius * transform.scale[1]];
    return toScreen(frame, p).join(",");
  }).join(" ");
  const [qx, qy] = toScreen(frame, q);

  return <div className="cls-lab">
    <section className="cls-main lab-panel">
      <ol className="mlp-phases att-phases" aria-label="Steps of a KNN prediction">
        {STEPS.map((s, i) => <li key={s.title} className={`${i === step ? "is-current dir-forward" : ""} ${i < step ? "is-done" : ""}`} aria-current={i === step ? "step" : undefined}>
          <button type="button" onClick={() => setStep(i)}><span>{i + 1}</span>{s.title}</button></li>)}
      </ol>
      <svg ref={svgRef} className="cls-plot cls-draggable" viewBox={`0 0 ${SIZE} ${SIZE}`} onPointerDown={move} onPointerMove={move} role="img"
        aria-label={`Query at (${fmt(query[0])}, ${fmt(query[1])}); ${k} nearest neighbours vote for class ${CLASS_NAMES[result.label]}`}>
        <Axes frame={frame} labels={[stretched ? `x₁ (×${STRETCH} units)` : "x₁", "x₂"]}/>
        {grid && grid.cells.map(({ i, j, c }) => { const w = (SIZE - 60) / grid.N; return <rect key={`${i}-${j}`} x={30 + i * w} y={SIZE - 30 - (j + 1) * w} width={w + .4} height={w + .4} fill={CLASS_COLORS[c]} fillOpacity={.17} shapeRendering="crispEdges"/>; })}
        {step === 0 && ranked.map((n) => { const [x, y] = toScreen(frame, measured.train[n.index].x); return <line key={n.index} x1={qx} y1={qy} x2={x} y2={y} stroke="#8f8577" strokeOpacity={.08 + .35 * Math.exp(-n.distance / .6)}/>; })}
        {step >= 1 && step < 3 && <polygon points={outline} className="cls-hood" clipPath="url(#cls-clip)"/>}
        {step >= 1 && step < 3 && result.neighbors.map((n) => { const [x, y] = toScreen(frame, measured.train[n.index].x); return <line key={n.index} x1={qx} y1={qy} x2={x} y2={y} stroke={CLASS_COLORS[n.label]} strokeWidth={step === 2 && weighting === "distance" ? 1 + 2.5 / (1 + 4 * n.distance) : 1.6} strokeOpacity={.75}/>; })}
        <Points frame={frame} train={measured.train} test={measured.test} dim={step >= 1 && step < 3 ? (i) => !near.has(i) : undefined}/>
        <g className="cls-query">
          <circle cx={qx} cy={qy} r={11} fill="#fff" stroke="#1f1e1c" strokeWidth={2}/>
          <circle cx={qx} cy={qy} r={5} fill={step >= 2 ? CLASS_COLORS[result.label] : "#1f1e1c"}/>
        </g>
      </svg>
      <Legend classes={data.classes}/>
      <div className="mlp-transport">
        <button type="button" className="btn" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}><ChevronLeft size={15}/> Back</button>
        <button type="button" className="btn btn-primary" onClick={() => setStep(Math.min(STEPS.length - 1, step + 1))} disabled={step === STEPS.length - 1}>Next <ChevronRight size={15}/></button>
        <span className="cls-drag-hint">Drag or tap the plot to move the query point.</span>
      </div>
    </section>

    <section className="cls-scan mlp-explain" aria-live="polite">
      <div className="mlp-explain-head"><span className="mlp-dir dir-forward">Step {step + 1} of {STEPS.length}</span><h2>{STEPS[step].title}</h2></div>
      <p>{STEPS[step].text}</p>
      <VotePanel step={step} result={result} classes={data.classes} metric={metric} weighting={weighting} qModel={qModel} train={model.train} ranked={ranked}/>
    </section>

    <aside className="cls-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>KNN settings</h2></div>
        <label className="slider-field"><span className="slider-label">k (neighbours) <output>{k}</output></span>
          <input type="range" min={1} max={30} step={1} value={k} onChange={(e) => setK(Number(e.target.value))}/></label>
        <Segmented label="Distance" options={[["euclidean", "Euclidean (L2)"], ["manhattan", "Manhattan (L1)"]]} value={metric} onChange={setMetric}/>
        <Segmented label="Votes" options={[["uniform", "Equal"], ["distance", "1 / distance"]]} value={weighting} onChange={setWeighting}/>
        <AccuracyChart xs={KS} train={curve.train} test={curve.test} current={k} xLabel="k"/>
        <p className="training-meta">k = 1 always scores {pct(curve.train[0])} on its own training points, because each point is its own nearest neighbour. Held-out accuracy {pct(curve.test[kIndex])} at k = {k}. Small k follows the noise; large k blurs the boundary.</p>
      </section>
      <DataControls kind={kind} setKind={setKind} seed={seed} setSeed={setSeed} noise={noise} setNoise={setNoise}>
        <label className="att-toggle"><input type="checkbox" checked={stretched} onChange={(e) => setStretched(e.target.checked)}/> Record x₁ in other units (×{STRETCH})</label>
        <label className="att-toggle"><input type="checkbox" checked={standardize} onChange={(e) => setStandardize(e.target.checked)}/> Standardise features (z-score)</label>
        <p className="training-meta">{stretched && !standardize ? "x₁ now spans a range ten times wider, so it dominates every distance: the neighbourhood squashes into a thin vertical band and x₂ barely matters. Watch the held-out accuracy at your k drop, then turn on standardising to fix it." : "KNN has no training step: the “model” is the stored training set, and all the work happens at prediction time."}</p>
      </DataControls>
    </aside>
  </div>;
}

function VotePanel({ step, result, classes, metric, weighting, qModel, train, ranked }: {
  step: number; result: ReturnType<typeof vote>; classes: number; metric: Metric; weighting: Weighting;
  qModel: [number, number]; train: { x: [number, number]; y: number }[]; ranked: ReturnType<typeof rankNeighbors>;
}) {
  if (step === 0) {
    const shown = ranked.slice(0, 5);
    return <div className="mlp-numbers"><table><tbody>
      {shown.map((n, i) => { const p = train[n.index].x, dx = p[0] - qModel[0], dy = p[1] - qModel[1];
        return <tr key={n.index}><th scope="row">#{i + 1} · class {CLASS_NAMES[n.label]}</th>
          <td>{metric === "euclidean" ? `√(${fmt(dx)}² + ${fmt(dy)}²)` : `|${fmt(dx)}| + |${fmt(dy)}|`} = {fmt(n.distance, 3)}</td></tr>; })}
      <tr><th scope="row">…</th><td>{ranked.length} distances in total{metric === "euclidean" ? " (Euclidean)" : " (Manhattan)"}</td></tr>
    </tbody></table></div>;
  }
  if (step === 1) {
    return <div className="mlp-numbers"><table><tbody>
      {result.neighbors.map((n, i) => <tr key={n.index}><th scope="row">#{i + 1}</th><td><b style={{ color: CLASS_COLORS[n.label] }}>class {CLASS_NAMES[n.label]}</b> · distance {fmt(n.distance, 3)}</td></tr>)}
    </tbody></table></div>;
  }
  const max = Math.max(...result.scores, 1e-9);
  return <div className="cls-votes">
    {Array.from({ length: classes }, (_, c) => <div key={c} className="cls-vote-row">
      <span>Class {CLASS_NAMES[c]}</span>
      <div className="mlp-grad-track"><i style={{ width: `${(result.scores[c] / max) * 100}%`, background: CLASS_COLORS[c], top: 0, bottom: 0 }}/></div>
      <span className="mlp-grad-value">{weighting === "uniform" ? result.scores[c] : fmt(result.scores[c], 2)}</span>
    </div>)}
    <p className="training-meta">Prediction: <b style={{ color: CLASS_COLORS[result.label] }}>class {CLASS_NAMES[result.label]}</b>{weighting === "distance" ? " (scores are sums of 1 / distance)" : ` (${result.scores[result.label]} of ${result.neighbors.length} votes)`}.</p>
  </div>;
}

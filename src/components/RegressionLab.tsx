"use client";

import { useEffect, useMemo, useState } from "react";
import { Download, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { gradientStep, gradients, initialState, makeDataset, type TrainingState } from "@/lib/regression";

const LIMIT = 240;
const view = { width: 590, height: 350, left: 55, right: 25, top: 18, bottom: 40 };
const xCoord = (x: number) => view.left + ((x + 1.2) / 2.4) * (view.width - view.left - view.right);
const yCoord = (y: number) => view.height - view.bottom - ((y + 1.35) / 5) * (view.height - view.top - view.bottom);

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function DataPlot({ points, current }: { points: ReturnType<typeof makeDataset>; current: TrainingState }) {
  const fitY1 = yCoord(current.w * -1.15 + current.b);
  const fitY2 = yCoord(current.w * 1.15 + current.b);
  return <div className="plot-wrap">
    <svg id="regression-data-plot" viewBox="0 0 590 350" role="img" aria-label={`Scatter plot of ${points.length} synthetic observations and fitted line at step ${current.step}. Current weight ${current.w.toFixed(3)}, bias ${current.b.toFixed(3)}.`}>
      <rect width="590" height="350" fill="#f9fbfa"/>
      {[-1,0,1,2,3].map((y)=><g key={y}><line x1="55" y1={yCoord(y)} x2="565" y2={yCoord(y)} stroke="#e9eeee"/><text x="44" y={yCoord(y)+4} textAnchor="end" fontSize="11" fill="#a5b0b2">{y}</text></g>)}
      {[-1,-.5,0,.5,1].map((x)=><g key={x}><line x1={xCoord(x)} y1="18" x2={xCoord(x)} y2="310" stroke="#f0f2f2"/><text x={xCoord(x)} y="330" textAnchor="middle" fontSize="11" fill="#a5b0b2">{x}</text></g>)}
      <line x1="55" x2="565" y1="310" y2="310" stroke="#cbd9da"/>
      <line x1={xCoord(-1.15)} x2={xCoord(1.15)} y1={yCoord(1.7 * -1.15 + 1.15)} y2={yCoord(1.7 * 1.15 + 1.15)} stroke="#65a493" strokeWidth="2" strokeDasharray="7 6" opacity=".8"/>
      <line x1={xCoord(-1.15)} x2={xCoord(1.15)} y1={fitY1} y2={fitY2} stroke="#c38c76" strokeWidth="3" strokeLinecap="round"/>
      {points.map((p,i)=><circle key={i} cx={xCoord(p.x)} cy={yCoord(p.y)} r="4.4" fill="#789ead" stroke="#fff" strokeWidth="1.35" opacity=".86"/>)}
      <text x="549" y="341" textAnchor="end" fontSize="10" fill="#9caeb4">FEATURE X</text>
      <text x="63" y="15" fontSize="10" fill="#9caeb4">TARGET Y</text>
    </svg>
  </div>;
}

function LossPlot({ history }: { history: TrainingState[] }) {
  const w = 590, h = 160, left = 48, bottom = 26, top = 16, right = 22;
  const maxLoss = Math.max(0.01,history[0].loss);
  const x = (step: number) => left + (step / LIMIT) * (w-left-right);
  const y = (loss: number) => h-bottom - (loss / maxLoss) * (h-top-bottom);
  const path = history.map((p,i)=>`${i===0?"M":"L"} ${x(p.step).toFixed(2)} ${y(p.loss).toFixed(2)}`).join(" ");
  return <div className="plot-wrap"><svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Loss history with ${history.length} recorded steps. Latest mean squared error ${history[history.length-1].loss.toFixed(5)}.`}>
    <rect width={w} height={h} fill="#f9fbfa"/>
    {[0,.5,1].map((fraction)=><g key={fraction}><line x1={left} x2={w-right} y1={y(maxLoss*fraction)} y2={y(maxLoss*fraction)} stroke="#e9eeee"/><text x={left-9} y={y(maxLoss*fraction)+4} textAnchor="end" fontSize="10" fill="#9dabaf">{(maxLoss*fraction).toFixed(2)}</text></g>)}
    {[0,60,120,180,240].map((step)=><text key={step} x={x(step)} y={h-8} textAnchor="middle" fontSize="10" fill="#9dabaf">{step}</text>)}
    <path d={path} stroke="#709a93" strokeWidth="2.8" fill="none" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx={x(history[history.length-1].step)} cy={y(history[history.length-1].loss)} r="4.5" fill="#608f85" stroke="white" strokeWidth="1.5"/>
  </svg></div>;
}

export function RegressionLab() {
  const [seed, setSeed] = useState(42);
  const [noise, setNoise] = useState(.35);
  const [learningRate, setLearningRate] = useState(.09);
  const [playing, setPlaying] = useState(false);
  const points = useMemo(()=>makeDataset(seed,noise),[seed,noise]);
  const [history,setHistory] = useState<TrainingState[]>(()=>[initialState(makeDataset(42,.35))]);
  const current = history[history.length-1];
  const grad = useMemo(()=>gradients(points,current),[points,current]);

  useEffect(()=>{
    if (!playing) return;
    const timer = window.setInterval(()=>{
      setHistory((prev)=>prev.length > LIMIT ? prev : [...prev,gradientStep(points,prev[prev.length-1],learningRate)]);
    },45);
    return ()=>window.clearInterval(timer);
  },[playing,points,learningRate]);

  useEffect(()=>{
    if (current.step >= LIMIT && playing) setPlaying(false);
  },[current.step,playing]);

  const reset = (nextSeed=seed,nextNoise=noise) => {
    setPlaying(false);
    setHistory([initialState(makeDataset(nextSeed,nextNoise))]);
  };
  const singleStep = () => {
    setPlaying(false);
    setHistory((prev)=>prev.length > LIMIT ? prev : [...prev,gradientStep(points,prev[prev.length-1],learningRate)]);
  };
  const exportJson = () => {
    const payload = {
      schemaVersion: "model-atlas.experiment.v1",modelId:"linear-regression",algorithmVersion:"gd-mse-1.0.0",
      executionMode:"live-training",dataset:{id:"synthetic-linear-v1",seed,noise,count:points.length,points},
      hyperparameters:{learningRate,maxSteps:LIMIT,initialParameters:{w:0,b:0}},
      result:{...current,gradients:grad,history},
      note:"The dataset is synthetic. This run uses full-batch gradient descent and mean squared error, computed in the browser."
    };
    download(`model-atlas-linear-seed-${seed}.json`,JSON.stringify(payload,null,2),"application/json;charset=utf-8");
  };
  const exportSvg = () => {
    const element = document.getElementById("regression-data-plot");
    if (!element) return;
    const cloned = element.cloneNode(true) as SVGSVGElement;
    cloned.setAttribute("xmlns","http://www.w3.org/2000/svg");
    download(`model-atlas-linear-step-${current.step}.svg`,new XMLSerializer().serializeToString(cloned),"image/svg+xml;charset=utf-8");
  };

  return <div className="lab-workspace">
    <div className="lab-main-column">
      <section className="lab-panel"><div className="lab-panel-heading"><div><h2>Fitting a line to observations</h2><p className="lab-panel-subtitle">Synthetic dataset · 42 observations · full-batch gradient descent</p></div><span>FIG. 01 / DATA SPACE</span></div><DataPlot points={points} current={current}/><div className="chart-legend"><span><i className="legend-mark legend-data"/> Observations</span><span><i className="legend-mark legend-fit"/> Fitted line</span><span><i className="legend-mark legend-true"/> Ground truth</span></div></section>
      <section className="lab-panel"><div className="lab-panel-heading"><div><h2>Training loss</h2><p className="lab-panel-subtitle">Mean squared error over gradient updates</p></div><span>FIG. 02 / OPTIMIZATION</span></div><LossPlot history={history}/></section>
    </div>
    <div className="lab-side-column">
      <section className="lab-panel"><div className="lab-panel-heading"><h2>Experiment controls</h2><span>LIVE / CPU</span></div><div className="metric-row"><div className="metric"><strong>{current.step}</strong><span>STEP</span></div><div className="metric"><strong>{current.w.toFixed(2)}</strong><span>WEIGHT W</span></div><div className="metric"><strong>{current.b.toFixed(2)}</strong><span>BIAS B</span></div></div>
        <label className="slider-field"><span className="slider-label">Learning rate <output>{learningRate.toFixed(2)}</output></span><input type="range" min="0.01" max="0.3" step="0.01" value={learningRate} onChange={(e)=>setLearningRate(Number(e.target.value))}/></label>
        <label className="slider-field"><span className="slider-label">Data noise <output>{noise.toFixed(2)}</output></span><input type="range" min="0" max="0.8" step="0.05" value={noise} onChange={(e)=>{const v=Number(e.target.value);setNoise(v);reset(seed,v);}}/></label>
        <label className="input-field">Random seed <input type="number" min="0" max="999999" value={seed} onChange={(e)=>{const v=Math.min(999999,Math.max(0,Number(e.target.value)||0));setSeed(v);reset(v,noise);}}/></label>
        <div className="lab-controls"><button type="button" disabled={current.step>=LIMIT} onClick={()=>setPlaying((v)=>!v)}>{playing?<><Pause size={14}/> Pause</>:<><Play size={14}/> Train</>}</button><button type="button" disabled={current.step>=LIMIT} onClick={singleStep}><StepForward size={14}/> +1 Step</button><button type="button" style={{gridColumn:"1 / -1"}} onClick={()=>reset()}><RotateCcw size={14}/> Reset parameters</button></div>
        <div className="training-meta">Maximum: <code>{LIMIT} steps</code> · Dataset: <code>synthetic-linear-v1</code> · Seed: <code>{seed}</code></div>
      </section>
      <section className="lab-panel"><div className="lab-panel-heading"><h2>Mathematical inspector</h2><span>MSE OBJECTIVE</span></div><div className="lab-math">ŷ = {current.w.toFixed(3)}x {current.b>=0?"+":"−"} {Math.abs(current.b).toFixed(3)}</div><div className="metric-row"><div className="metric"><strong>{current.loss.toFixed(3)}</strong><span>LOSS</span></div><div className="metric"><strong>{grad.w.toFixed(3)}</strong><span>∂L/∂W</span></div><div className="metric"><strong>{grad.b.toFixed(3)}</strong><span>∂L/∂B</span></div></div><p className="lab-math-small">Every step calculates predictions, MSE gradients and updated coefficients. The green dashed line is the known generating function, not an inferred result.</p></section>
      <section className="lab-panel"><div className="lab-panel-heading"><h2>Export experiment</h2><span>REPRODUCIBLE</span></div><div className="lab-note"><span aria-hidden="true">●</span><span>Actual browser-side training — not a prerecorded animation. Saved JSON includes all observations, hyperparameters, full loss history and final gradients.</span></div><div className="export-buttons"><button type="button" onClick={exportJson}><Download size={14}/> Export JSON</button><button type="button" onClick={exportSvg}><Download size={14}/> Export SVG</button></div></section>
    </div>
  </div>;
}

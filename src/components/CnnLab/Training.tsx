import { useEffect, useMemo, useRef, useState } from "react";
import { classes, dataset, evaluate, forward } from "@/lib/cnn/training";
import type { RunnerEvent } from "@/lib/cnn/controller";
import { createTrainingRunner, type RunnerMode, type TrainingRunner } from "@/lib/cnn/runner";
import { TrainingSession, type TrainingConfig, type TrainingSnapshot } from "@/lib/cnn/session";
import { NumberField, Plot } from "./widgets";

const percent = (value: number) => (value * 100).toFixed(1) + "%";

function Metrics({ title, result }: { title: string; result: ReturnType<typeof evaluate> }) {
  return <div>
    <h3>{title}</h3>
    <p>Loss {result.loss.toFixed(4)} · Accuracy {percent(result.accuracy)}</p>
    <div className="cnn-table"><table><caption>Confusion matrix · actual rows, predicted columns</caption>
      <thead><tr><th>Class</th>{classes.map(name => <th key={name}>{name.split(" ")[0]}</th>)}</tr></thead>
      <tbody>{result.matrix.map((row, c) => <tr key={classes[c]}><th scope="row">{classes[c].split(" ")[0]}</th>{row.map((n, i) => <td key={i}>{n}</td>)}</tr>)}</tbody>
    </table></div>
    <div className="cnn-table"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>{result.scores.map((score, c) => <tr key={classes[c]}><th scope="row">{classes[c].split(" ")[0]}</th>
        <td>{percent(score.precision)}</td><td>{percent(score.recall)}</td><td>{percent(score.f1)}</td></tr>)}</tbody>
    </table></div>
  </div>;
}

export function Training({ exportJson, active }: { exportJson: (value: object) => void; active: boolean }) {
  const [seed, setSeed] = useState(42);
  const [rate, setRate] = useState(.05);
  const [batchSize, setBatchSize] = useState(16);
  const [epochs, setEpochs] = useState(24);
  const [target, setTarget] = useState(0);
  // The user's intent. The runner may still pause on its own while the tab is hidden.
  const [running, setRunning] = useState(false);
  // Render the untrained state synchronously; the runner's first snapshot is identical.
  const [state, setState] = useState<TrainingSnapshot>(() => new TrainingSession({ seed: 42, rate: .05, batchSize: 16 }).snapshot());
  const [mode, setMode] = useState<RunnerMode>("worker");
  const [error, setError] = useState<string | null>(null);
  const [tested, setTested] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [layer, setLayer] = useState<"first" | "second">("first");
  const [selected, select] = useState(0);
  const runner = useRef<TrainingRunner | null>(null);
  const runId = useRef(0);
  const samples = useMemo(() => dataset(seed), [seed]);
  const { model, history, offset, trace } = state;
  const epoch = history.length - 1;
  const latest = history[epoch];
  const test = useMemo(() => tested ? evaluate(model, samples.test) : null, [tested, model, samples]);
  const sample = samples.test[sampleIndex];
  const probabilities = tested ? forward(model, sample.signal).probabilities : null;

  useEffect(() => {
    const onEvent = (event: RunnerEvent) => {
      if (event.runId !== runId.current) return; // stale message from before a reset
      if (event.type === "error") { setError(event.message); setRunning(false); return; }
      setState(event.snapshot);
      if (event.status === "done") setRunning(false);
    };
    const instance = createTrainingRunner(onEvent, setMode);
    runner.current = instance;
    setMode(instance.mode);
    return () => { instance.dispose(); runner.current = null; };
  }, []);

  // Configuration can only change before training starts; each change starts a fresh, seeded session.
  useEffect(() => { start({ seed, rate, batchSize }); }, [seed, rate, batchSize]);

  // Train only while the tab is visible and the user wants it to run.
  useEffect(() => {
    const id = runId.current;
    if (active && running && !tested) runner.current?.send({ type: "run", runId: id, targetEpoch: target });
    else runner.current?.send({ type: "pause", runId: id });
  }, [active, running, target, tested]);

  function start(config: TrainingConfig) {
    runId.current += 1;
    setError(null);
    runner.current?.send({ type: "reset", runId: runId.current, config });
  }

  function reset(nextSeed = seed) {
    setRunning(false);
    setTarget(0);
    setTested(false);
    setSampleIndex(0);
    if (nextSeed === seed) start({ seed, rate, batchSize });
  }

  const locked = running || epoch > 0 || offset > 0 || tested;
  const filter = Math.min(selected, model[layer].weights.length - 1);
  return <div className="cnn-grid" id="cnn-training">
    <section className="cnn-card">
      <h2>Training controls</h2>
      <NumberField label="Learning Rate" min={.001} max={.2} step={.001} value={rate} change={setRate} disabled={locked}/>
      <NumberField label="Batch Size" min={1} max={64} value={batchSize} change={setBatchSize} disabled={locked}/>
      <NumberField label="Epochs" min={1} max={100} value={epochs} change={setEpochs} disabled={running || tested}/>
      <NumberField label="Seed" min={0} max={4294967295} value={seed} change={value => { reset(value); setSeed(value); }} disabled={locked}/>
      <div className="cnn-actions">
        <button disabled={running || tested || epoch >= epochs} onClick={() => { setTarget(epochs); setRunning(true); }}>Train</button>
        <button disabled={!running} onClick={() => setRunning(false)}>Pause</button>
        <button disabled={running || tested || epoch >= target} onClick={() => setRunning(true)}>Resume</button>
        <button disabled={running || tested} onClick={() => { setTarget(epoch + 1); setRunning(true); }}>Single Epoch</button>
        <button onClick={() => reset()}>Reset</button>
      </div>
      <p aria-live="polite">Epoch {epoch}{offset > 0 && " · " + offset + "/210 samples in current epoch"}</p>
      <p className="training-meta">{mode === "worker" ? "Runs in a background Web Worker, so the page stays responsive." : "Runs on the main thread (Web Workers unavailable)."} {state.updates} SGD updates.</p>
      {error && <p role="alert">Training stopped: {error}</p>}
      <h3>Architecture · 155 parameters</h3>
      <ol className="cnn-architecture">
        {["Input · 1 × 64", "Conv1D K5 + ReLU · 4 × 60", "MaxPool K2 S2 · 4 × 30", "Conv1D K3 + ReLU · 8 × 28", "Global Average Pool · 8", "Dense + Softmax · 3"].map(name => <li key={name}>{name}</li>)}
      </ol>
      <p>Stratified split · 210 train / 45 validation / 45 test · length 64.</p>
      <button onClick={() => exportJson({ workspace: "training", config: { seed, rate, batchSize, epochs, target }, execution: { mode: "live-training", runner: mode, updates: state.updates }, split: { train: 210, validation: 45, test: 45 }, classes, epoch, offset, history, model, trace, test, sample: tested ? { ...sample, probabilities } : null })}>Export JSON</button>
    </section>
    <section className="cnn-card">
      <h2>Training monitor</h2>
      <p>Training · loss {latest.train.loss.toFixed(4)} · accuracy {percent(latest.train.accuracy)}</p>
      <Plot title="Cross entropy by epoch" rows={[history.map(h => h.train.loss), history.map(h => h.validation.loss)]} labels={["Training", "Validation"]}/>
      <Plot title="Accuracy by epoch" rows={[history.map(h => h.train.accuracy), history.map(h => h.validation.accuracy)]} labels={["Training", "Validation"]} domain={[0, 1]}/>
      <Metrics title="Validation metrics" result={latest.validation}/>
      <h3>Independent test</h3>
      <p>Finalize the run to evaluate the held-out set. Reset starts a fresh run.</p>
      <button disabled={tested || running || epoch === 0 || offset !== 0} onClick={() => setTested(true)}>Finalize &amp; evaluate test set</button>
      {test && probabilities && <>
        <Metrics title="Test metrics · 45 samples" result={test}/>
        <NumberField label="Test sample" min={1} max={45} value={sampleIndex + 1} change={n => setSampleIndex(n - 1)}/>
        <Plot title="Held-out waveform" rows={[sample.signal]}/>
        <p>True class · {classes[sample.label]}</p>
        <ul>{probabilities.map((p, i) => <li key={classes[i]}>{classes[i]} · {percent(p)}</li>)}</ul>
      </>}
    </section>
    <section className="cnn-card">
      <h2>Backpropagation trace</h2>
      <p>Forward Pass → Loss → Backward Pass → Gradient → Weight Update</p>
      <label>Conv1D layer<select value={layer} onChange={e => { setLayer(e.target.value === "first" ? "first" : "second"); select(0); }}><option value="first">Conv1D 1</option><option value="second">Conv1D 2</option></select></label>
      <label>Conv1D filter<select value={filter} onChange={e => select(+e.target.value)}>{model[layer].weights.map((_, f) => <option key={f} value={f}>{f + 1}</option>)}</select></label>
      {trace ? <>
        <p>Latest batch · {trace.size} samples · cross entropy {trace.loss.toFixed(5)}</p>
        <p>First sample · {classes[trace.firstLabel]} · probabilities {trace.firstProbabilities.map(percent).join(" / ")}</p>
        <p>Batch mean gradient · learning rate {trace.rate}</p>
        <div className="cnn-table"><table><thead><tr><th>Tap</th><th>Before</th><th>Gradient</th><th>After</th></tr></thead>
          <tbody>{trace.before[layer].weights[filter].flatMap((row, c) => row.map((before, k) =>
            <tr key={c + "-" + k}><th scope="row">{c + 1}.{k + 1}</th><td>{before.toFixed(5)}</td><td>{trace.gradients[layer].weights[filter][c][k].toFixed(5)}</td><td>{trace.after[layer].weights[filter][c][k].toFixed(5)}</td></tr>))}</tbody>
        </table></div>
      </> : <p>Run training to inspect a measured SGD update.</p>}
    </section>
  </div>;
}

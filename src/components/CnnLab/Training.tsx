import { useEffect, useMemo, useState } from "react";
import { classes, dataset, epochOrder, evaluate, forward, makeModel, trainBatch } from "@/lib/cnn/training";
import { NumberField, Plot } from "./widgets";

const percent = (value: number) => (value * 100).toFixed(1) + "%";

function initialize(seed: number) {
  const model = makeModel(seed);
  const samples = dataset(seed);
  return {
    model,
    history: [{ train: evaluate(model, samples.train), validation: evaluate(model, samples.validation) }],
    offset: 0,
    trace: null as ReturnType<typeof trainBatch> | null,
  };
}

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
  const [running, setRunning] = useState(false);
  const [state, setState] = useState(() => initialize(42));
  const [tested, setTested] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [layer, setLayer] = useState<"first" | "second">("first");
  const [selected, select] = useState(0);
  const samples = useMemo(() => dataset(seed), [seed]);
  const { model, history, offset, trace } = state;
  const epoch = history.length - 1;
  const latest = history[epoch];
  const order = useMemo(() => epochOrder(samples.train, seed, epoch + 1), [samples, seed, epoch]);
  const test = useMemo(() => tested ? evaluate(model, samples.test) : null, [tested, model, samples]);
  const sample = samples.test[sampleIndex];
  const probabilities = tested ? forward(model, sample.signal).probabilities : null;

  useEffect(() => {
    if (!active || !running || epoch >= target || tested) return;
    const timer = window.setTimeout(() => {
      const next = structuredClone(model);
      const batch = order.slice(offset, offset + batchSize);
      const update = trainBatch(next, batch, rate);
      const end = offset + batch.length === order.length;
      const records = end ? [...history, { train: evaluate(next, samples.train), validation: evaluate(next, samples.validation) }] : history;
      setState({ model: next, history: records, offset: end ? 0 : offset + batch.length, trace: update });
      if (end && epoch + 1 >= target) setRunning(false);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [active, running, epoch, target, tested, model, order, offset, batchSize, rate, history, samples]);

  function reset(nextSeed = seed) {
    setRunning(false);
    setTarget(0);
    setTested(false);
    setSampleIndex(0);
    setState(initialize(nextSeed));
  }

  const locked = running || epoch > 0 || offset > 0 || tested;
  const filter = Math.min(selected, model[layer].weights.length - 1);
  return <div className="cnn-grid" id="cnn-training">
    <section className="cnn-card">
      <h2>Training controls</h2>
      <NumberField label="Learning Rate" min={.001} max={.2} step={.001} value={rate} change={setRate} disabled={locked}/>
      <NumberField label="Batch Size" min={1} max={64} value={batchSize} change={setBatchSize} disabled={locked}/>
      <NumberField label="Epochs" min={1} max={100} value={epochs} change={setEpochs} disabled={running || tested}/>
      <NumberField label="Seed" min={0} max={4294967295} value={seed} change={value => { setSeed(value); reset(value); }} disabled={locked}/>
      <div className="cnn-actions">
        <button disabled={running || tested || epoch >= epochs} onClick={() => { setTarget(epochs); setRunning(true); }}>Train</button>
        <button disabled={!running} onClick={() => setRunning(false)}>Pause</button>
        <button disabled={running || tested || epoch >= target} onClick={() => setRunning(true)}>Resume</button>
        <button disabled={running || tested} onClick={() => { setTarget(epoch + 1); setRunning(true); }}>Single Epoch</button>
        <button onClick={() => reset()}>Reset</button>
      </div>
      <p aria-live="polite">Epoch {epoch}{offset > 0 && " · " + offset + "/210 samples in current epoch"}</p>
      <h3>Architecture · 155 parameters</h3>
      <ol className="cnn-architecture">
        {["Input · 1 × 64", "Conv1D K5 + ReLU · 4 × 60", "MaxPool K2 S2 · 4 × 30", "Conv1D K3 + ReLU · 8 × 28", "Global Average Pool · 8", "Dense + Softmax · 3"].map(name => <li key={name}>{name}</li>)}
      </ol>
      <p>Stratified split · 210 train / 45 validation / 45 test · length 64.</p>
      <button onClick={() => exportJson({ workspace: "training", config: { seed, rate, batchSize, epochs, target }, split: { train: 210, validation: 45, test: 45 }, classes, epoch, offset, history, model, trace, test, sample: tested ? { ...sample, probabilities } : null })}>Export JSON</button>
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

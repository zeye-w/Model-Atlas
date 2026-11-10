import { useEffect, useState } from "react";
import { conv1d, convolutionWindow, pool, relu } from "@/lib/cnn/ops";
import { Plot, NumberField } from "./widgets";

const presets = ["Sine wave", "Pulse", "Mixed signal", "Synthetic geophysical trace"];

export function Explorer({ exportJson }: { exportJson: (value: object) => void }) {
  const [preset, setPreset] = useState(0);
  const [config, setConfig] = useState({ kernel: 5, stride: 1, padding: 2, dilation: 1, channels: 1, filters: 2 });
  const [useBias, setBias] = useState(true);
  const [average, setAverage] = useState(false);
  const [position, setPosition] = useState(0);
  const [selected, setSelected] = useState(0);
  const [speed, setSpeed] = useState(4);
  const [playing, play] = useState(false);
  const { kernel, channels, filters } = config;
  const input = Array.from({ length: channels }, (_, c) => Array.from({ length: 48 }, (_, i) => {
    const t = i - c * 2;
    const sine = Math.sin(t * Math.PI / 8);
    const pulse = Number(t >= 14 && t <= 21);
    const wavelet = ((u: number) => (1 - 2 * u * u) * Math.exp(-u * u));
    return [sine, pulse, .5 * sine + pulse + .22 * Math.sin(t * 1.7), wavelet((t - 15) / 2) - .7 * wavelet((t - 32) / 3) + .07 * Math.sin(t * 2.3)][preset];
  }));
  const weights = Array.from({ length: filters }, (_, f) => Array.from({ length: channels }, (_, c) =>
    Array.from({ length: kernel }, (_, k) => Math.cos((f + 1) * (k - (kernel - 1) / 2) * .7 + c) / Math.sqrt(kernel * channels))));
  const bias = weights.map((_, f) => useBias ? .1 * (f + 1) : 0);
  const convolved = conv1d(input, { weights, bias }, config);
  const activated = relu(convolved);
  const pooled = pool(activated, 2, 2, average).values;
  const last = convolved[0].length - 1;
  const step = Math.min(position, last);
  const filter = Math.min(selected, filters - 1);
  const terms = convolutionWindow(input, weights[filter], step, config);
  useEffect(() => {
    if (!playing || step >= last) return;
    const timer = window.setTimeout(() => setPosition(step + 1), 1000 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, step, last, speed]);

  function move(next: number) {
    play(false);
    setPosition(next);
  }

  return <div className="cnn-grid" id="cnn-explorer">
    <section className="cnn-card">
      <h2>Signal &amp; geometry</h2>
      <label>Input signal<select value={preset} onChange={e => { setPreset(+e.target.value); move(0); }}>{presets.map((name, i) => <option key={name} value={i}>{name}</option>)}</select></label>
      <div className="cnn-fields">{([
        ["kernel", "Kernel size", 1, 9], ["stride", "Stride", 1, 5], ["padding", "Padding", 0, 12],
        ["dilation", "Dilation", 1, 5], ["channels", "Input channels", 1, 4], ["filters", "Output filters", 1, 6],
      ] as const).map(([key, label, min, max]) => <NumberField key={key} {...{ label, min, max }} value={config[key]}
        change={value => { setConfig({ ...config, [key]: value }); move(0); }}/>)}</div>
      <label className="cnn-check"><input type="checkbox" checked={useBias} onChange={e => setBias(e.target.checked)}/> Bias</label>
      <label>Pooling<select value={+average} onChange={e => setAverage(Boolean(+e.target.value))}><option value="0">MaxPool1D</option><option value="1">AveragePool1D</option></select></label>
      <button style={{ marginTop: 6 }} onClick={() => exportJson({ workspace: "explorer", preset: presets[preset], config, useBias, average, step, filter, input, weights, bias, convolved, activated, pooled })}>Export JSON</button>
    </section>
    <section className="cnn-card">
      <h2>Convolution window</h2>
      <p>Cross-correlation · tap indices are shown without reversing the kernel.</p>
      <Plot title="Input signal" rows={input} marks={terms.filter(t => t.channel === 0).map(t => t.index)}/>
      <div className="cnn-actions">
        <button disabled={step <= 0} onClick={() => move(step - 1)}>Previous</button>
        <button disabled={step >= last} onClick={() => move(step + 1)}>Next step</button>
        <button className="primary" disabled={playing && step < last} onClick={() => { if (step >= last) setPosition(0); play(true); }}>Play</button>
        <button disabled={!playing || step >= last} onClick={() => play(false)}>Pause</button>
        <button onClick={() => move(0)}>Reset</button>
      </div>
      <div className="cnn-fields"><NumberField label="Speed (steps/s)" min={1} max={20} value={speed} change={setSpeed}/>
      <label>Output filter<select value={filter} onChange={e => setSelected(+e.target.value)}>{weights.map((_, i) => <option key={i} value={i}>{i + 1}</option>)}</select></label></div>
      <p>Position {step + 1} / {last + 1} · bias {bias[filter].toFixed(4)} · sum + bias = <strong>{convolved[filter][step].toFixed(4)}</strong></p>
      <div className="cnn-table"><table><thead><tr><th>Channel</th><th>Index</th><th>Input</th><th>Weight</th><th>Product</th></tr></thead>
      <tbody>{terms.map((t, i) => <tr key={i}><td>{t.channel + 1}</td><td>{t.index}</td><td>{t.value.toFixed(4)}</td><td>{t.weight.toFixed(4)}</td><td>{t.product.toFixed(4)}</td></tr>)}</tbody></table></div>
    </section>
    <section className="cnn-card">
      <h2>Tensor inspector</h2>
      <p>Input {channels} × 48 · {filters * (channels * kernel + Number(useBias))} trainable parameters</p>
      <p>Effective kernel width {1 + (kernel - 1) * config.dilation}</p>
      {([["Conv1D", convolved], ["ReLU", activated], [average ? "AveragePool1D" : "MaxPool1D", pooled]] as const).map(([name, rows]) =>
        <Plot key={name} title={name + " · " + rows.length + " × " + rows[0].length} rows={rows}/>)}
    </section>
  </div>;
}

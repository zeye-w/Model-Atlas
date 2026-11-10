"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { D_K, attention, headWeights, heads, phases, tokenize, type HeadId, type Matrix } from "@/lib/attention";

const ACCENT = "#c96442", SLATE = "#5f7a96";
const fmt = (v: number, d = 2) => (!Number.isFinite(v) ? "−∞" : (v < 0 ? "−" : "") + Math.abs(v).toFixed(d));
const PRESETS = ["the cat sat on the mat", "to be or not to be", "attention is all you need", "the dog chased the cat and the cat ran"];

export function AttentionLab({ phaseFormulas, phaseSummaries }: { phaseFormulas: ReactNode[]; phaseSummaries: string[] }) {
  const [sentence, setSentence] = useState(PRESETS[0]);
  const [head, setHead] = useState<HeadId>("previous-token");
  const [causal, setCausal] = useState(true);
  const [scale, setScale] = useState(true);
  const [seed, setSeed] = useState(1);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const tokens = useMemo(() => { const t = tokenize(sentence); return t.length ? t : ["…"]; }, [sentence]);
  const [query, setQuery] = useState(4);
  const q = Math.min(query, tokens.length - 1);
  const trace = useMemo(() => attention(tokens, headWeights(head, seed), { causal, scale }), [tokens, head, seed, causal, scale]);
  const id = phases[phase].id;

  useEffect(() => {
    if (!playing) return;
    if (phase >= phases.length - 1) { setPlaying(false); return; }
    const t = window.setTimeout(() => setPhase((p) => p + 1), 1800);
    return () => window.clearTimeout(t);
  }, [playing, phase]);

  return <div className="att-lab">
    <section className="att-stage lab-panel">
      <ol className="mlp-phases att-phases" aria-label="Steps of attention">
        {phases.map((p, i) => <li key={p.id} className={`${i === phase ? "is-current dir-forward" : ""} ${i < phase ? "is-done" : ""}`} aria-current={i === phase ? "step" : undefined}>
          <button type="button" onClick={() => { setPlaying(false); setPhase(i); }}><span>{i + 1}</span>{p.title}</button>
        </li>)}
      </ol>

      <TokenArcs tokens={tokens} weights={phase >= 3 ? trace.A[q] : null} query={q} onSelect={setQuery} causal={causal}/>
      <p className="att-hint">Click a word to make it the query. {phase >= 3 ? `Arc thickness is how much “${tokens[q]}” attends to each word.` : "Arcs appear once scores become attention weights."}</p>

      <div className="att-matrices">
        {id === "embed" && <Heatmap title="X — embeddings" rows={tokens} cols={["e₁", "e₂", "e₃", "e₄", "p₁", "p₂", "p₃", "p₄"]} data={trace.X} highlightRow={q} split={4}/>}
        {id === "project" && <>
          <Heatmap title="Q — queries" rows={tokens} cols={["1", "2", "3", "4"]} data={trace.Q} highlightRow={q}/>
          <Heatmap title="K — keys" rows={tokens} cols={["1", "2", "3", "4"]} data={trace.K}/>
          <Heatmap title="V — values" rows={tokens} cols={["1", "2", "3", "4"]} data={trace.V}/>
        </>}
        {id === "scores" && <Heatmap title={`S = QKᵀ${scale ? ` / √${D_K}` : ""}${causal ? " (causal mask)" : ""}`} rows={tokens} cols={tokens} data={trace.scores} highlightRow={q} square/>}
        {id === "softmax" && <Heatmap title="A — attention weights (rows sum to 1)" rows={tokens} cols={tokens} data={trace.A} highlightRow={q} square probability/>}
        {id === "output" && <>
          <Heatmap title="O = AV — outputs" rows={tokens} cols={["1", "2", "3", "4"]} data={trace.O} highlightRow={q}/>
          <Heatmap title="Output resembles… (cosine to each value)" rows={tokens} cols={tokens} data={trace.similarity} highlightRow={q} square/>
        </>}
      </div>

      <div className="mlp-transport">
        <button type="button" className="btn" onClick={() => { setPlaying(false); setPhase((p) => Math.max(0, p - 1)); }} disabled={phase === 0}><ChevronLeft size={15}/> Back</button>
        <button type="button" className="btn btn-primary" onClick={() => { if (phase >= phases.length - 1) setPhase(0); setPlaying((v) => !v); }}>{playing ? <><Pause size={15}/> Pause</> : <><Play size={15}/> {phase >= phases.length - 1 ? "Replay" : "Play"}</>}</button>
        <button type="button" className="btn" onClick={() => { setPlaying(false); setPhase((p) => Math.min(phases.length - 1, p + 1)); }} disabled={phase === phases.length - 1}>Next <ChevronRight size={15}/></button>
      </div>
    </section>

    <section className="att-explain mlp-explain" aria-live="polite">
      <div className="mlp-explain-head"><span className="mlp-dir dir-forward">Step {phase + 1} of {phases.length}</span><h2>{phases[phase].title}</h2></div>
      {phaseFormulas[phase]}
      <p>{phaseSummaries[phase]}</p>
      <QueryNumbers trace={trace} q={q} phase={id} scale={scale}/>
    </section>

    <aside className="att-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Sentence</h2><p className="lab-panel-subtitle">Up to 10 words · same word, same embedding</p></div>
        <input className="field-control" value={sentence} maxLength={120} onChange={(e) => setSentence(e.target.value)} aria-label="Sentence"/>
        <div className="att-presets">{PRESETS.map((p) => <button key={p} type="button" className="filter-pill" onClick={() => setSentence(p)}>{p}</button>)}</div>
      </section>
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Head</h2></div>
        <div className="att-heads" role="radiogroup" aria-label="Attention head">
          {(Object.keys(heads) as HeadId[]).map((h) => <button key={h} type="button" role="radio" aria-checked={head === h} onClick={() => setHead(h)}>
            <strong>{heads[h].name}</strong><span>{heads[h].description}</span>
          </button>)}
        </div>
        {head === "random" && <label className="input-field" style={{ marginTop: 12 }}>Random seed
          <input className="field-control" type="number" min={0} max={9999} value={seed} onChange={(e) => setSeed(Math.max(0, Math.min(9999, Number(e.target.value) || 0)))}/></label>}
        <label className="att-toggle"><input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)}/> Causal mask (no looking ahead)</label>
        <label className="att-toggle"><input type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)}/> Scale scores by 1/√d<sub>k</sub></label>
        <p className="training-meta">The two named heads are constructed by hand so their behaviour follows from the algebra; none of the heads is trained. In a real Transformer, many such heads run in parallel and their weights are learned.</p>
      </section>
    </aside>
  </div>;
}

/** Words in a row with arcs from the query word to every key, thickness = attention weight. */
function TokenArcs({ tokens, weights, query, onSelect, causal }: { tokens: string[]; weights: number[] | null; query: number; onSelect: (i: number) => void; causal: boolean }) {
  const n = tokens.length, W = 700, step = W / n, cx = (i: number) => step * (i + .5), base = 122;
  return <div className="att-arcs">
    <svg viewBox={`0 0 ${W} 130`} aria-hidden="true">
      {weights && tokens.map((_, j) => {
        const a = weights[j];
        if (a < .005) return null;
        if (j === query) return <circle key={j} cx={cx(j)} cy={base - 14} r={10 + 6 * a} fill="none" stroke={ACCENT} strokeWidth={1 + 7 * a} strokeOpacity={.25 + .7 * a}/>;
        const x1 = cx(query), x2 = cx(j), h = Math.min(110, 18 + Math.abs(x2 - x1) * .45);
        return <path key={j} className="att-arc" d={`M${x1},${base} C${x1},${base - h} ${x2},${base - h} ${x2},${base}`} fill="none" stroke={ACCENT} strokeWidth={1 + 9 * a} strokeOpacity={.2 + .75 * a} strokeLinecap="round"/>;
      })}
    </svg>
    <div className="att-tokens" style={{ gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` }}>
      {tokens.map((t, i) => <button key={i} type="button" onClick={() => onSelect(i)} aria-pressed={i === query}
        className={`${i === query ? "is-query" : ""} ${causal && i > query ? "is-masked" : ""}`}>
        <span>{t}</span>{weights && <small>{weights[i] < .005 ? "0" : fmt(weights[i])}</small>}
      </button>)}
    </div>
  </div>;
}

function Heatmap({ title, rows, cols, data, highlightRow, split, square, probability }: {
  title: string; rows: string[]; cols: string[]; data: Matrix; highlightRow?: number; split?: number; square?: boolean; probability?: boolean;
}) {
  const finite = data.flat().filter(Number.isFinite);
  const max = Math.max(1e-9, ...finite.map(Math.abs));
  return <figure className={`att-heat${square ? " is-square" : ""}`}>
    <figcaption>{title}</figcaption>
    <div className="att-heat-scroll"><table>
      <thead><tr><th/>{cols.map((c, j) => <th key={j} scope="col" className={split === j ? "split" : ""}>{c}</th>)}</tr></thead>
      <tbody>{data.map((row, i) => <tr key={i} className={i === highlightRow ? "is-hl" : ""}>
        <th scope="row">{rows[i]}</th>
        {row.map((v, j) => {
          const masked = !Number.isFinite(v);
          const a = masked ? 0 : probability ? v : Math.abs(v) / max;
          const color = probability || v >= 0 ? ACCENT : SLATE;
          return <td key={j} className={`${masked ? "masked" : ""} ${split === j ? "split" : ""}`}
            style={masked ? undefined : { background: `color-mix(in srgb, ${color} ${Math.round(a * 52)}%, white)` }}>
            {fmt(v)}
          </td>;
        })}
      </tr>)}</tbody>
    </table></div>
  </figure>;
}

/** The arithmetic for the selected query word in the current step. */
function QueryNumbers({ trace, q, phase, scale }: { trace: ReturnType<typeof attention>; q: number; phase: string; scale: boolean }) {
  const t = trace.tokens;
  const word = `“${t[q]}” (position ${q + 1})`;
  let rows: [string, string][] = [];
  if (phase === "embed") rows = [[`x for ${word}`, `[${trace.X[q].slice(0, 4).map((v) => fmt(v)).join(", ")} | ${trace.X[q].slice(4).map((v) => fmt(v)).join(", ")}]`]];
  if (phase === "project") rows = [["q", trace.Q[q]], ["k", trace.K[q]], ["v", trace.V[q]]].map(([k, v]) => [`${k} for ${word}`, `[${(v as number[]).map((x) => fmt(x)).join(", ")}]`]);
  if (phase === "scores") rows = t.map((w, j) => [`${t[q]} · ${w}${j === q ? " (self)" : ""}`, Number.isFinite(trace.scores[q][j]) ? `${fmt(trace.raw[q][j], 3)}${scale ? ` / ${Math.sqrt(D_K)} = ${fmt(trace.scores[q][j], 3)}` : ""}` : "masked (later word)"]);
  if (phase === "softmax") rows = t.map((w, j) => [`${t[q]} → ${w}`, Number.isFinite(trace.scores[q][j]) ? `${fmt(trace.A[q][j], 3)}` : "0 (masked)"]);
  if (phase === "output") rows = [
    [`o for ${word}`, `[${trace.O[q].map((v) => fmt(v)).join(", ")}]`],
    ["Largest contribution", (() => { const j = trace.A[q].indexOf(Math.max(...trace.A[q])); return `${fmt(trace.A[q][j], 2)} × v(${t[j]}) from position ${j + 1}`; })()],
    ["Most similar value", (() => { const s = trace.similarity[q]; const j = s.indexOf(Math.max(...s)); return `${t[j]} (position ${j + 1}), cosine ${fmt(s[j], 3)}`; })()],
  ];
  return <div className="mlp-numbers"><table><tbody>{rows.map(([k, v], i) => <tr key={i}><th scope="row">{k}</th><td>{v}</td></tr>)}</tbody></table></div>;
}


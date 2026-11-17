"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { accuracy, makeClassData, type DatasetKind } from "@/lib/classify/data";
import { growTree, isSplit, leavesAt, predictTree, type Criterion, type GrownTree, type TreeNode } from "@/lib/classify/tree";
import { AccuracyChart, Axes, CLASS_COLORS, CLASS_NAMES, DataControls, Legend, Points, SIZE, Segmented, fmt, pct, toScreen, type Frame } from "./common";

const FRAME: Frame = { x: [-2, 2], y: [-2, 2] };
const DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8];
const FEATURE = ["x₁", "x₂"];

export function DecisionTreeLab() {
  const [kind, setKind] = useState<DatasetKind>("moons");
  const [seed, setSeed] = useState(1);
  const [noise, setNoise] = useState(.25);
  const [maxDepth, setMaxDepth] = useState(4);
  const [minLeaf, setMinLeaf] = useState(2);
  const [criterion, setCriterion] = useState<Criterion>("gini");
  const [step, setStep] = useState<number | null>(null);   // null = fully grown
  const [playing, setPlaying] = useState(false);

  const data = useMemo(() => makeClassData(kind, seed, noise), [kind, seed, noise]);
  const tree = useMemo(() => growTree(data.train, data.classes, { maxDepth, minLeaf, criterion }, [-2, 2, -2, 2]), [data, maxDepth, minLeaf, criterion]);
  const total = tree.splits.length;
  const s = step === null ? total : Math.min(step, total);
  const current = s > 0 ? tree.nodes[tree.splits[s - 1]] : null;
  const curve = useMemo(() => {
    const train: number[] = [], test: number[] = [];
    for (const d of DEPTHS) {
      const t = growTree(data.train, data.classes, { maxDepth: d, minLeaf, criterion });
      train.push(accuracy((x) => predictTree(t, x), data.train)); test.push(accuracy((x) => predictTree(t, x), data.test));
    }
    return { train, test };
  }, [data, minLeaf, criterion]);
  const trainAcc = accuracy((x) => predictTree(tree, x, s), data.train);
  const testAcc = accuracy((x) => predictTree(tree, x, s), data.test);
  const reset = () => { setStep(null); setPlaying(false); };

  useEffect(() => {
    if (!playing) return;
    if (s >= total) { setPlaying(false); return; }
    const t = window.setTimeout(() => setStep(s + 1), 1100);
    return () => window.clearTimeout(t);
  }, [playing, s, total]);

  return <div className="cls-lab">
    <section className="cls-main lab-panel">
      <div className="lab-panel-heading reg-heading">
        <div><h2>Regions after {s} split{s === 1 ? "" : "s"}</h2><p className="lab-panel-subtitle">Shade = majority class of each leaf, stronger when the leaf is purer</p></div>
        <div className="reg-metrics"><span><strong>{pct(trainAcc)}</strong> train</span><span><strong>{pct(testAcc)}</strong> test</span><span><strong>{leavesAt(tree, s).length}</strong> leaves</span></div>
      </div>
      <TreePlane tree={tree} steps={s} current={current} data={data}/>
      <Legend classes={data.classes}/>
      <div className="reg-transport">
        <button type="button" className="btn" aria-label="Previous split" onClick={() => { setPlaying(false); setStep(Math.max(0, s - 1)); }} disabled={s === 0}><ChevronLeft size={15}/></button>
        <button type="button" className="btn btn-primary" onClick={() => { if (s >= total) setStep(0); setPlaying((v) => !v); }}>{playing ? <><Pause size={15}/> Pause</> : <><Play size={15}/> {s >= total ? "Grow again" : "Grow"}</>}</button>
        <button type="button" className="btn" aria-label="Next split" onClick={() => { setPlaying(false); setStep(Math.min(total, s + 1)); }} disabled={s === total}><ChevronRight size={15}/></button>
        <label className="reg-scrub"><span>Split <strong>{s}</strong> / {total}</span>
          <input type="range" min={0} max={total} value={s} onChange={(e) => { setPlaying(false); setStep(Number(e.target.value)); }} aria-label="Number of splits"/></label>
      </div>
      <p className="reg-note" aria-live="polite">{current ? splitNote(current, tree, criterion) : `No splits yet: a single leaf predicts the majority class ${CLASS_NAMES[tree.nodes[0].label]} for every point. Root ${criterion === "gini" ? "Gini" : "entropy"} ${fmt(tree.nodes[0].impurity, 3)}.`}</p>
    </section>

    <section className="cls-scan lab-panel">
      <div className="lab-panel-heading"><h2>How the split was chosen</h2><p className="lab-panel-subtitle">{current ? `Every threshold on both features for node #${current.id} (${current.indices.length} points); the tallest bar wins` : "Step forward to see the first split’s search"}</p></div>
      {current ? <ScanCharts node={current}/> : <p className="training-meta">The tree tries every midpoint between neighbouring values of x₁ and x₂ and keeps the one that lowers impurity the most.</p>}
      {kind === "xor" && <p className="training-meta">On XOR no single split helps much: each half still holds both classes equally. A greedy tree cannot see that two splits together would separate the data, so its first cut lands almost anywhere and it needs extra depth.</p>}
    </section>

    <section className="cls-tree lab-panel">
      <div className="lab-panel-heading"><h2>The tree so far</h2><p className="lab-panel-subtitle">Bars: class counts in each node · highlighted: the split just made</p></div>
      <TreeList tree={tree} steps={s} current={current?.id ?? null}/>
    </section>

    <aside className="cls-side">
      <section className="lab-panel">
        <div className="lab-panel-heading"><h2>Tree settings</h2></div>
        <Segmented label="Max depth" options={DEPTHS.map((d) => [d, String(d)])} value={maxDepth} onChange={(d) => { setMaxDepth(d); reset(); }}/>
        <label className="slider-field"><span className="slider-label">Min samples per leaf <output>{minLeaf}</output></span>
          <input type="range" min={1} max={15} step={1} value={minLeaf} onChange={(e) => { setMinLeaf(Number(e.target.value)); reset(); }}/></label>
        <Segmented label="Impurity" options={[["gini", "Gini"], ["entropy", "Entropy"]]} value={criterion} onChange={(c) => { setCriterion(c); reset(); }}/>
        <AccuracyChart xs={DEPTHS} train={curve.train} test={curve.test} current={maxDepth} xLabel="depth"/>
        <p className="training-meta">Deeper trees always fit the training points at least as well; held-out accuracy usually peaks earlier and then falls as the tree memorises noise.</p>
      </section>
      <DataControls kind={kind} setKind={(k) => { setKind(k); reset(); }} seed={seed} setSeed={(n) => { setSeed(n); reset(); }} noise={noise} setNoise={(n) => { setNoise(n); reset(); }}/>
    </aside>
  </div>;
}

function splitNote(node: TreeNode, tree: GrownTree, criterion: Criterion) {
  const sp = node.split!, l = tree.nodes[node.left!], r = tree.nodes[node.right!];
  const name = criterion === "gini" ? "Gini" : "entropy";
  return `Split ${node.order! + 1}: node #${node.id} (depth ${node.depth}, ${node.indices.length} points, ${name} ${fmt(node.impurity, 3)}) on ${FEATURE[sp.feature]} ≤ ${fmt(sp.threshold)}. ` +
    `Left ${l.indices.length} points (${name} ${fmt(l.impurity, 3)}), right ${r.indices.length} (${fmt(r.impurity, 3)}); impurity decrease ${fmt(sp.gain, 3)}.`;
}

function TreePlane({ tree, steps, current, data }: { tree: GrownTree; steps: number; current: TreeNode | null; data: ReturnType<typeof makeClassData> }) {
  const leaves = leavesAt(tree, steps);
  const box = (b: TreeNode["bounds"]) => { const [x0, y1] = toScreen(FRAME, [b[0], b[2]]), [x1, y0] = toScreen(FRAME, [b[1], b[3]]); return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 }; };
  const applied = tree.nodes.filter((n) => isSplit(n, steps));
  const line = (n: TreeNode) => {
    const b = n.bounds, t = n.split!.threshold;
    const [a, c] = n.split!.feature === 0 ? [[t, b[2]], [t, b[3]]] : [[b[0], t], [b[1], t]];
    const [x1, y1] = toScreen(FRAME, a), [x2, y2] = toScreen(FRAME, c);
    return { x1, y1, x2, y2 };
  };
  return <svg className="cls-plot" viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={`Feature plane divided into ${leaves.length} regions by ${steps} splits`}>
    <Axes frame={FRAME}/>
    {leaves.map((n) => { const purity = n.indices.length ? Math.max(...n.counts) / n.indices.length : 0; return <rect key={n.id} {...box(n.bounds)} fill={CLASS_COLORS[n.label]} fillOpacity={.06 + .26 * purity} className="cls-region"/>; })}
    {applied.map((n) => <line key={n.id} {...line(n)} className={n === current ? "cls-split is-current" : "cls-split"}/>)}
    {current && <rect {...box(current.bounds)} className="cls-node-outline"/>}
    <Points frame={FRAME} train={data.train} test={data.test} dim={current ? (i) => !current.indices.includes(i) : undefined}/>
  </svg>;
}

function ScanCharts({ node }: { node: TreeNode }) {
  const sp = node.split!;
  const max = Math.max(...sp.scan.flat().map((p) => p.gain), 1e-9);
  return <div className="cls-scans">
    {([0, 1] as const).map((f) => {
      const W = 420, H = 120, L = 8, R = 8, B = 20;
      const scan = sp.scan[f];
      const x = (t: number) => L + ((t + 2) / 4) * (W - L - R);
      const best = scan.reduce((b, p) => (p.gain > b.gain ? p : b), { t: 0, gain: -1 });
      return <figure key={f} className="cls-scan-fig">
        <figcaption>{FEATURE[f]} {sp.feature === f ? <strong>· chosen</strong> : best.gain >= 0 ? `· best ${fmt(best.gain, 3)} at ${fmt(best.t)}` : "· no valid split"}</figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Impurity decrease for each threshold on ${FEATURE[f]}`}>
          <line x1={L} x2={W - R} y1={H - B} y2={H - B} className="cls-grid"/>
          {scan.map((p, i) => <line key={i} x1={x(p.t)} x2={x(p.t)} y1={H - B} y2={H - B - (Math.max(0, p.gain) / max) * (H - B - 10)}
            stroke={sp.feature === f && p.t === sp.threshold ? "#c96442" : "#b9b1a2"} strokeWidth={sp.feature === f && p.t === sp.threshold ? 3 : 1.6}/>)}
          {[-2, -1, 0, 1, 2].map((v) => <text key={v} x={x(v)} y={H - 5} textAnchor="middle" className="cls-tick">{v}</text>)}
        </svg>
      </figure>;
    })}
    <p className="training-meta">Bar height: impurity decrease I(parent) − (n<sub>L</sub>/n)·I(left) − (n<sub>R</sub>/n)·I(right) for that threshold. Chosen: {FEATURE[sp.feature]} ≤ {fmt(sp.threshold)}, decrease {fmt(sp.gain, 3)}.</p>
  </div>;
}

function TreeList({ tree, steps, current }: { tree: GrownTree; steps: number; current: number | null }) {
  const rows: { node: TreeNode; rule: string }[] = [];
  const walk = (id: number, rule: string) => {
    const n = tree.nodes[id];
    rows.push({ node: n, rule });
    if (isSplit(n, steps)) {
      const f = FEATURE[n.split!.feature], t = fmt(n.split!.threshold);
      walk(n.left!, `${f} ≤ ${t}`); walk(n.right!, `${f} > ${t}`);
    }
  };
  walk(0, "all points");
  return <div className="cls-treelist" role="tree">
    {rows.map(({ node, rule }) => {
      const leaf = !isSplit(node, steps);
      return <div key={node.id} role="treeitem" aria-level={node.depth + 1} className={`cls-tnode${node.id === current ? " is-current" : ""}${current !== null && node.parent === current ? " is-child" : ""}`} style={{ paddingLeft: 8 + node.depth * 16 }}>
        <span className="cls-rule">{rule}</span>
        <span className="cls-bar" aria-label={node.counts.map((c, i) => `${c} class ${CLASS_NAMES[i]}`).join(", ")}>
          {node.counts.map((c, i) => c ? <i key={i} style={{ flexGrow: c, background: CLASS_COLORS[i] }}/> : null)}
        </span>
        <span className="cls-meta">{node.indices.length} · {fmt(node.impurity, 2)}{leaf ? <> · <b style={{ color: CLASS_COLORS[node.label] }}>{CLASS_NAMES[node.label]}</b></> : ""}</span>
      </div>;
    })}
  </div>;
}

import { mean, random, rmse, type Point, type RegressionData } from "./data";
import { fitObliviousTree, fitTree, predictOblivious, predictTree, type ObliviousTree, type Tree } from "./tree";

/**
 * Five regression methods on the same 1-D task. Each returns one stage per
 * round (or tree, or hidden unit), so the laboratory can replay how the fit is
 * built. Every curve is the model's actual prediction on the grid.
 */

export type AlgorithmId = "adaboost" | "gradient-boosting" | "random-forest" | "catboost" | "rbf-network";

export interface Stage {
  /** Model prediction on `data.grid` after this stage. */
  curve: number[];
  train: number;
  test: number;
  /** What this stage added, on the grid (weak learner, tree or basis functions). */
  parts: number[][];
  /** Training-point residuals before this stage (boosting) or sample weights after it (AdaBoost). */
  pointValues?: number[];
  note: string;
}

export interface FitResult { id: AlgorithmId; stages: Stage[]; baseline: { train: number; test: number }; stoppedEarly?: string }

const xsOf = (pts: Point[]) => pts.map((p) => p.x);
const ysOf = (pts: Point[]) => pts.map((p) => p.y);
const fmt = (v: number, d = 3) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);

function baseline(data: RegressionData) {
  const m = mean(ysOf(data.train));
  return { train: rmse(() => m, data.train), test: rmse(() => m, data.test) };
}

function stage(data: RegressionData, predict: (x: number) => number, parts: number[][], note: string, pointValues?: number[]): Stage {
  return { curve: data.grid.map(predict), train: rmse(predict, data.train), test: rmse(predict, data.test), parts, note, pointValues };
}

const rmseOf = (pred: number[], pts: Point[]) => Math.sqrt(pred.reduce((s, v, i) => s + (v - pts[i].y) ** 2, 0) / pts.length);

/**
 * Running predictions on the grid, training and test points, so each stage
 * costs O(points) instead of re-evaluating every earlier learner.
 */
class Running {
  grid: number[]; train: number[]; test: number[];
  constructor(private data: RegressionData, init: number) {
    this.grid = data.grid.map(() => init); this.train = data.train.map(() => init); this.test = data.test.map(() => init);
  }
  add(f: (x: number) => number, scale = 1) {
    const { data } = this;
    data.grid.forEach((x, i) => { this.grid[i] += scale * f(x); });
    data.train.forEach((p, i) => { this.train[i] += scale * f(p.x); });
    data.test.forEach((p, i) => { this.test[i] += scale * f(p.x); });
  }
  snapshot(parts: number[][], note: string, pointValues?: number[], divide = 1): Stage {
    const d = (a: number[]) => a.map((v) => v / divide);
    return { curve: d(this.grid), train: rmseOf(d(this.train), this.data.train), test: rmseOf(d(this.test), this.data.test), parts, note, pointValues };
  }
}

/* ── Gradient boosting (squared loss, CART base learners) ─────────────── */
export interface GradientBoostingParams { rounds: number; learningRate: number; maxDepth: number; minLeaf: number }

export function gradientBoosting(data: RegressionData, p: GradientBoostingParams): FitResult {
  const xs = xsOf(data.train), ys = ysOf(data.train);
  const run = new Running(data, mean(ys));
  const stages: Stage[] = [];
  for (let m = 0; m < p.rounds; m++) {
    const residuals = ys.map((y, i) => y - run.train[i]);     // = −∂/∂F ½(y − F)²
    const tree = fitTree(xs, residuals, { maxDepth: p.maxDepth, minLeaf: p.minLeaf });
    run.add((x) => predictTree(tree, x), p.learningRate);
    stages.push(run.snapshot([data.grid.map((x) => p.learningRate * predictTree(tree, x))],
      `Round ${m + 1}: tree fits the residuals; added with step ν = ${p.learningRate}.`, residuals));
  }
  return { id: "gradient-boosting", stages, baseline: baseline(data) };
}

/* ── CatBoost-style: symmetric trees, plain boosting, L2 leaf regularisation ─ */
export interface CatBoostParams { rounds: number; learningRate: number; depth: number; l2LeafReg: number }

export function catBoost(data: RegressionData, p: CatBoostParams): FitResult {
  const xs = xsOf(data.train), ys = ysOf(data.train);
  const run = new Running(data, mean(ys));
  const stages: Stage[] = [];
  for (let m = 0; m < p.rounds; m++) {
    const residuals = ys.map((y, i) => y - run.train[i]);
    const tree = fitObliviousTree(xs, residuals, p.depth, p.l2LeafReg);
    run.add((x) => predictOblivious(tree, x), p.learningRate);
    stages.push(run.snapshot([data.grid.map((x) => p.learningRate * predictOblivious(tree, x))],
      `Round ${m + 1}: symmetric tree, splits at ${tree.thresholds.map((t) => t.toFixed(2)).join(", ")} on every level; leaves shrunk by l2 = ${p.l2LeafReg}.`, residuals));
  }
  return { id: "catboost", stages, baseline: baseline(data) };
}

/* ── Random forest (bagged CART; one feature, so no feature subsampling) ── */
export interface RandomForestParams { trees: number; maxDepth: number; minLeaf: number; seed: number }

export function randomForest(data: RegressionData, p: RandomForestParams): FitResult {
  const xs = xsOf(data.train), ys = ysOf(data.train);
  const rng = random(p.seed ^ 0xf0e57);
  const run = new Running(data, 0);
  const stages: Stage[] = [];
  for (let b = 0; b < p.trees; b++) {
    const counts = xs.map(() => 0);
    for (let i = 0; i < xs.length; i++) counts[Math.floor(rng() * xs.length)]++;
    // A bootstrap sample is a multiset: weight each point by how often it was drawn.
    const tree = fitTree(xs, ys, { maxDepth: p.maxDepth, minLeaf: p.minLeaf }, counts);
    run.add((x) => predictTree(tree, x));
    const n = b + 1, inBag = counts.filter((c) => c > 0).length;
    stages.push(run.snapshot([data.grid.map((x) => predictTree(tree, x))],
      `Tree ${n}: trained on a bootstrap sample (${inBag} of ${xs.length} distinct points); the forest averages all ${n} trees.`,
      counts, n));
  }
  return { id: "random-forest", stages, baseline: baseline(data) };
}

/* ── AdaBoost.R2 (Drucker, 1997), linear loss, weighted-median prediction ─ */
export interface AdaBoostParams { rounds: number; maxDepth: number; learningRate: number }

/** Weighted median of `values`: the smallest value whose cumulative weight reaches half the total. */
export function weightedMedian(values: number[], weights: number[]) {
  const idx = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
  const half = weights.reduce((a, b) => a + b, 0) / 2;
  let acc = 0;
  for (const i of idx) { acc += weights[i]; if (acc >= half) return values[i]; }
  return values[idx[idx.length - 1]];
}

/** Per-point sorted (value, weight) lists, so the weighted median updates in O(m) per learner. */
class RunningMedian {
  private lists: { v: number; w: number }[][];
  private total = 0;
  constructor(private xs: number[]) { this.lists = xs.map(() => []); }
  add(f: (x: number) => number, weight: number) {
    this.total += weight;
    this.xs.forEach((x, i) => {
      const v = f(x), list = this.lists[i];
      let lo = 0, hi = list.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].v <= v) lo = mid + 1; else hi = mid; }
      list.splice(lo, 0, { v, w: weight });
    });
  }
  values() {
    const half = this.total / 2;
    return this.lists.map((list) => { let acc = 0; for (const e of list) { acc += e.w; if (acc >= half) return e.v; } return list[list.length - 1].v; });
  }
}

export function adaBoost(data: RegressionData, p: AdaBoostParams): FitResult {
  const xs = xsOf(data.train), ys = ysOf(data.train), n = xs.length;
  let w = xs.map(() => 1 / n);
  const med = { grid: new RunningMedian(data.grid), train: new RunningMedian(xs), test: new RunningMedian(xsOf(data.test)) };
  const stages: Stage[] = [];
  let stoppedEarly: string | undefined;
  for (let m = 0; m < p.rounds; m++) {
    // Weighted fitting (scikit-learn instead resamples by weight; both target the same distribution).
    const tree = fitTree(xs, ys, { maxDepth: p.maxDepth, minLeaf: 1 }, w);
    const err = xs.map((x, i) => Math.abs(ys[i] - predictTree(tree, x)));
    const D = Math.max(...err);
    let alpha: number;
    if (D === 0) { alpha = 1; stoppedEarly = `Round ${m + 1} fits the training data exactly; boosting stops.`; }
    else {
      const L = err.map((e) => e / D);                          // linear loss in [0, 1]
      const avg = L.reduce((s, l, i) => s + w[i] * l, 0);
      if (avg >= .5) {
        stoppedEarly = `Round ${m + 1}: average weighted loss ${fmt(avg)} ≥ 0.5, so AdaBoost.R2 stops.`;
        if (stages.length > 0) break;
        alpha = 1;
      } else {
        const beta = avg / (1 - avg);
        alpha = p.learningRate * Math.log(1 / beta);
        w = w.map((wi, i) => wi * Math.pow(beta, p.learningRate * (1 - L[i])));
        const total = w.reduce((a, b) => a + b, 0);
        w = w.map((wi) => wi / total);
      }
    }
    const f = (x: number) => predictTree(tree, x);
    med.grid.add(f, alpha); med.train.add(f, alpha); med.test.add(f, alpha);
    const tr = med.train.values(), te = med.test.values();
    stages.push({
      curve: med.grid.values(), train: rmseOf(tr, data.train), test: rmseOf(te, data.test),
      parts: [data.grid.map(f)], pointValues: [...w],
      note: `Round ${stages.length + 1}: tree weight ln(1/β) = ${fmt(alpha)}; badly fitted points gain weight; prediction is the weighted median of all trees.`,
    });
    if (stoppedEarly) break;
  }
  return { id: "adaboost", stages, baseline: baseline(data), stoppedEarly };
}

/* ── RBF network: Gaussian hidden units, linear output solved by ridge least squares ── */
export interface RbfParams { maxUnits: number; width: number; ridge: number }

/** Solve A x = b for symmetric positive-definite A (Cholesky). */
export function solveSpd(A: number[][], b: number[]) {
  const n = b.length, L = A.map((row) => row.map(() => 0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    if (i === j) { if (s <= 0) throw new Error("matrix is not positive definite"); L[i][i] = Math.sqrt(s); }
    else L[i][j] = s / L[j][j];
  }
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}

export interface RbfNetwork { centers: number[]; sigma: number; weights: number[]; bias: number }

export const rbf = (x: number, c: number, sigma: number) => Math.exp(-((x - c) ** 2) / (2 * sigma * sigma));
export const predictRbf = (net: RbfNetwork, x: number) => net.weights.reduce((s, w, k) => s + w * rbf(x, net.centers[k], net.sigma), net.bias);

/**
 * Fit a network with `units` Gaussian units: centres evenly spaced over the
 * input range, width σ = width / units, output weights and bias from the
 * ridge-regularised normal equations (ΦᵀΦ + λI) w = Φᵀy (bias unpenalised).
 */
export function fitRbf(xs: number[], ys: number[], units: number, width: number, ridge: number): RbfNetwork {
  const centers = Array.from({ length: units }, (_, k) => (k + .5) / units);
  const sigma = width / units;
  const phi = xs.map((x) => [...centers.map((c) => rbf(x, c, sigma)), 1]);
  const d = units + 1;
  const A = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => phi.reduce((s, r) => s + r[i] * r[j], 0) + (i === j ? (i < units ? ridge : 0) + 1e-9 : 0)));  // 1e-9 jitter keeps λ = 0 solvable
  const b = Array.from({ length: d }, (_, i) => phi.reduce((s, r, n) => s + r[i] * ys[n], 0));
  const sol = solveSpd(A, b);
  return { centers, sigma, weights: sol.slice(0, units), bias: sol[units] };
}

export function rbfNetwork(data: RegressionData, p: RbfParams): FitResult {
  const xs = xsOf(data.train), ys = ysOf(data.train);
  const stages: Stage[] = [];
  for (let k = 1; k <= p.maxUnits; k++) {
    const net = fitRbf(xs, ys, k, p.width, p.ridge);
    stages.push(stage(data, (x) => predictRbf(net, x),
      net.centers.map((c, j) => data.grid.map((x) => net.weights[j] * rbf(x, c, net.sigma))),
      `${k} hidden unit${k === 1 ? "" : "s"}: a separately solved network; σ = ${net.sigma.toFixed(3)}, bias ${fmt(net.bias)}.`));
  }
  return { id: "rbf-network", stages, baseline: baseline(data) };
}

/* ── Registry of laboratory algorithms ────────────────────────────────── */
export type ParamSpec = { key: string; label: string; min: number; max: number; step: number };

export const algorithms = {
  "adaboost": {
    name: "AdaBoost", stageNoun: "round", fit: adaBoost,
    defaults: { rounds: 60, maxDepth: 3, learningRate: 1 } as AdaBoostParams,
    params: [{ key: "rounds", label: "Rounds", min: 1, max: 200, step: 1 }, { key: "maxDepth", label: "Tree depth", min: 1, max: 6, step: 1 }, { key: "learningRate", label: "Learning rate", min: .05, max: 1, step: .05 }],
  },
  "gradient-boosting": {
    name: "Gradient Boosting", stageNoun: "round", fit: gradientBoosting,
    defaults: { rounds: 120, learningRate: .1, maxDepth: 3, minLeaf: 2 } as GradientBoostingParams,
    params: [{ key: "rounds", label: "Rounds", min: 1, max: 300, step: 1 }, { key: "learningRate", label: "Learning rate ν", min: .01, max: 1, step: .01 }, { key: "maxDepth", label: "Tree depth", min: 1, max: 6, step: 1 }, { key: "minLeaf", label: "Min samples per leaf", min: 1, max: 10, step: 1 }],
  },
  "random-forest": {
    name: "Random Forest", stageNoun: "tree", fit: randomForest,
    defaults: { trees: 60, maxDepth: 6, minLeaf: 3, seed: 7 } as RandomForestParams,
    params: [{ key: "trees", label: "Trees", min: 1, max: 200, step: 1 }, { key: "maxDepth", label: "Max depth", min: 1, max: 12, step: 1 }, { key: "minLeaf", label: "Min samples per leaf", min: 1, max: 10, step: 1 }, { key: "seed", label: "Bootstrap seed", min: 0, max: 9999, step: 1 }],
  },
  "catboost": {
    name: "CatBoost", stageNoun: "round", fit: catBoost,
    defaults: { rounds: 120, learningRate: .1, depth: 4, l2LeafReg: 3 } as CatBoostParams,
    params: [{ key: "rounds", label: "Rounds", min: 1, max: 300, step: 1 }, { key: "learningRate", label: "Learning rate", min: .01, max: 1, step: .01 }, { key: "depth", label: "Depth (symmetric)", min: 1, max: 6, step: 1 }, { key: "l2LeafReg", label: "l2_leaf_reg", min: 0, max: 20, step: .5 }],
  },
  "rbf-network": {
    name: "RBF Network", stageNoun: "unit count", fit: rbfNetwork,
    defaults: { maxUnits: 16, width: .9, ridge: .001 } as RbfParams,
    params: [{ key: "maxUnits", label: "Hidden units (max)", min: 1, max: 40, step: 1 }, { key: "width", label: "Width (σ × units)", min: .2, max: 3, step: .05 }, { key: "ridge", label: "Ridge λ", min: 0, max: 1, step: .001 }],
  },
} satisfies Record<AlgorithmId, { name: string; stageNoun: string; fit: (d: RegressionData, p: never) => FitResult; defaults: object; params: ParamSpec[] }>;

export const algorithmIds = Object.keys(algorithms) as AlgorithmId[];

export function fitAlgorithm(id: AlgorithmId, data: RegressionData, params: Record<string, number>): FitResult {
  return (algorithms[id].fit as unknown as (d: RegressionData, p: Record<string, number>) => FitResult)(data, params);
}

import type { Point } from "./data";

/**
 * k-nearest-neighbour classification on two features.
 * Distances: Euclidean (L2) or Manhattan (L1). Votes: uniform, or weighted by
 * 1 / distance. Optional per-feature scaling (stretch) and standardisation.
 */

export type Metric = "euclidean" | "manhattan";
export type Weighting = "uniform" | "distance";

export interface KnnOptions { k: number; metric: Metric; weighting: Weighting }

export const distance = (a: readonly number[], b: readonly number[], metric: Metric) =>
  metric === "euclidean" ? Math.hypot(a[0] - b[0], a[1] - b[1]) : Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);

export interface Neighbor { index: number; distance: number; label: number }

/** All training points sorted by distance to `q` (ties broken by index). */
export function rankNeighbors(train: Point[], q: readonly number[], metric: Metric): Neighbor[] {
  return train.map((p, index) => ({ index, distance: distance(p.x, q, metric), label: p.y }))
    .sort((a, b) => a.distance - b.distance || a.index - b.index);
}

export interface Vote { neighbors: Neighbor[]; scores: number[]; label: number }

/**
 * Vote among the k nearest. With distance weighting a neighbour at distance 0
 * decides alone. Ties go to the class of the nearest tied neighbour.
 */
export function vote(ranked: Neighbor[], classes: number, options: KnnOptions): Vote {
  const neighbors = ranked.slice(0, Math.min(options.k, ranked.length));
  const scores = new Array(classes).fill(0);
  const exact = neighbors.filter((n) => n.distance === 0);
  if (options.weighting === "distance" && exact.length) for (const n of exact) scores[n.label] += 1;
  else for (const n of neighbors) scores[n.label] += options.weighting === "distance" ? 1 / n.distance : 1;
  const top = Math.max(...scores);
  const tied = new Set(scores.flatMap((s, c) => (Math.abs(s - top) < 1e-12 ? [c] : [])));
  const label = neighbors.find((n) => tied.has(n.label))?.label ?? scores.indexOf(top);
  return { neighbors, scores, label };
}

export function predictKnn(train: Point[], classes: number, q: readonly number[], options: KnnOptions) {
  return vote(rankNeighbors(train, q, options.metric), classes, options).label;
}

/* ── Feature scaling ──────────────────────────────────────────────────── */
export interface Transform { mean: [number, number]; scale: [number, number] }
export const identity: Transform = { mean: [0, 0], scale: [1, 1] };

/** z-score parameters from training data. */
export function standardizer(train: Point[]): Transform {
  const m = [0, 1].map((f) => train.reduce((s, p) => s + p.x[f], 0) / train.length) as [number, number];
  const sd = [0, 1].map((f) => Math.sqrt(train.reduce((s, p) => s + (p.x[f] - m[f]) ** 2, 0) / train.length) || 1) as [number, number];
  return { mean: m, scale: sd };
}

export const applyTransform = (t: Transform, x: readonly number[]): [number, number] => [(x[0] - t.mean[0]) / t.scale[0], (x[1] - t.mean[1]) / t.scale[1]];
export const mapPoints = (t: Transform, pts: Point[]): Point[] => pts.map((p) => ({ x: applyTransform(t, p.x), y: p.y }));

/** Multiply one feature by `factor` — an "unscaled" feature measured in different units. */
export const stretch = (pts: Point[], factor: number, feature: 0 | 1 = 1): Point[] =>
  pts.map((p) => ({ x: feature === 0 ? [p.x[0] * factor, p.x[1]] : [p.x[0], p.x[1] * factor], y: p.y }));

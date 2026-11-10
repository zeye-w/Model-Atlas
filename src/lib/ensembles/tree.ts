/**
 * Regression trees on one numeric feature.
 *
 * - `fitTree`: CART with weighted squared error, greedy best split per node.
 * - `fitObliviousTree`: a symmetric ("oblivious") tree, CatBoost's default
 *   structure — every node on a level uses the same split. On one feature a
 *   depth-d symmetric tree therefore has at most d + 1 non-empty leaves.
 */

export type Tree =
  | { leaf: true; value: number; n: number }
  | { leaf: false; threshold: number; left: Tree; right: Tree };

export interface TreeOptions { maxDepth: number; minLeaf: number }

export function predictTree(tree: Tree, x: number): number {
  let node = tree;
  while (!node.leaf) node = x <= node.threshold ? node.left : node.right;
  return node.value;
}

export function countLeaves(tree: Tree): number {
  return tree.leaf ? 1 : countLeaves(tree.left) + countLeaves(tree.right);
}

/** Indices of `xs` sorted by x (ties keep input order). */
function order(xs: number[], idx: number[]) {
  return [...idx].sort((a, b) => xs[a] - xs[b] || a - b);
}

/**
 * Fit a CART regression tree. `weights` default to 1. Leaves predict the
 * weighted mean of their targets.
 */
export function fitTree(xs: number[], ys: number[], options: TreeOptions, weights?: number[]): Tree {
  const w = weights ?? xs.map(() => 1);
  const grow = (idx: number[], depth: number): Tree => {
    let sw = 0, swy = 0;
    for (const i of idx) { sw += w[i]; swy += w[i] * ys[i]; }
    const value = sw > 0 ? swy / sw : 0;
    if (depth >= options.maxDepth || idx.length < 2 * options.minLeaf) return { leaf: true, value, n: idx.length };

    const sorted = order(xs, idx);
    let lw = 0, lwy = 0, lwyy = 0, best = { gain: 1e-12, at: -1 };
    let tw = 0, twy = 0, twyy = 0;
    for (const i of sorted) { tw += w[i]; twy += w[i] * ys[i]; twyy += w[i] * ys[i] * ys[i]; }
    const parentSse = twyy - (tw > 0 ? twy * twy / tw : 0);
    for (let k = 0; k < sorted.length - 1; k++) {
      const i = sorted[k];
      lw += w[i]; lwy += w[i] * ys[i]; lwyy += w[i] * ys[i] * ys[i];
      if (k + 1 < options.minLeaf || sorted.length - k - 1 < options.minLeaf) continue;
      if (xs[sorted[k + 1]] === xs[i]) continue; // cannot split between equal x
      const rw = tw - lw, rwy = twy - lwy, rwyy = twyy - lwyy;
      const sse = (lwyy - (lw > 0 ? lwy * lwy / lw : 0)) + (rwyy - (rw > 0 ? rwy * rwy / rw : 0));
      const gain = parentSse - sse;
      if (gain > best.gain) best = { gain, at: k };
    }
    if (best.at < 0) return { leaf: true, value, n: idx.length };
    const threshold = (xs[sorted[best.at]] + xs[sorted[best.at + 1]]) / 2;
    return {
      leaf: false, threshold,
      left: grow(sorted.slice(0, best.at + 1), depth + 1),
      right: grow(sorted.slice(best.at + 1), depth + 1),
    };
  };
  return grow(xs.map((_, i) => i), 0);
}

export interface ObliviousTree { thresholds: number[]; values: number[]; counts: number[] }

/** Leaf index of x: bit l is set when x > thresholds[l]. */
export function obliviousLeaf(tree: Pick<ObliviousTree, "thresholds">, x: number) {
  return tree.thresholds.reduce((leaf, t, l) => leaf | (x > t ? 1 << l : 0), 0);
}

export function predictOblivious(tree: ObliviousTree, x: number) {
  return tree.values[obliviousLeaf(tree, x)];
}

/**
 * Fit a symmetric tree to targets `ys`, choosing one threshold per level that
 * minimises total squared error over all current leaves. Leaf values are
 * `sum(y) / (count + l2)` — the L2-regularised leaf estimate CatBoost uses
 * (with l2 = 0 this is the plain mean).
 */
export function fitObliviousTree(xs: number[], ys: number[], depth: number, l2: number): ObliviousTree {
  const sorted = order(xs, xs.map((_, i) => i));
  const candidates: number[] = [];
  for (let k = 0; k < sorted.length - 1; k++) {
    const a = xs[sorted[k]], b = xs[sorted[k + 1]];
    if (a !== b) candidates.push((a + b) / 2);
  }
  const thresholds: number[] = [];
  const leafOf = xs.map(() => 0);   // current leaf of every point
  for (let level = 0; level < depth; level++) {
    const leaves = 1 << (level + 1);
    const s = new Float64Array(leaves), n = new Float64Array(leaves);
    let bestScore = Infinity, bestT = NaN, current = 0;
    {
      const s0 = new Float64Array(leaves), n0 = new Float64Array(leaves);
      for (let i = 0; i < xs.length; i++) { s0[leafOf[i]] += ys[i]; n0[leafOf[i]]++; }
      for (let l = 0; l < leaves; l++) if (n0[l]) current -= s0[l] * s0[l] / n0[l];
    }
    for (const t of candidates) {
      if (thresholds.includes(t)) continue;
      s.fill(0); n.fill(0);
      for (let i = 0; i < xs.length; i++) { const l = leafOf[i] | (xs[i] > t ? 1 << level : 0); s[l] += ys[i]; n[l]++; }
      // SSE = Σy² − Σ_leaf (Σy)²/n; Σy² is constant, so compare the second term only.
      let score = 0;
      for (let l = 0; l < leaves; l++) if (n[l]) score -= s[l] * s[l] / n[l];
      if (score < bestScore) { bestScore = score; bestT = t; }
    }
    if (Number.isNaN(bestT) || bestScore >= current - 1e-12) break;
    thresholds.push(bestT);
    for (let i = 0; i < xs.length; i++) if (xs[i] > bestT) leafOf[i] |= 1 << level;
  }
  const leaves = 1 << thresholds.length;
  const sums = new Array(leaves).fill(0), counts = new Array(leaves).fill(0);
  for (let i = 0; i < xs.length; i++) { const l = obliviousLeaf({ thresholds }, xs[i]); sums[l] += ys[i]; counts[l]++; }
  return { thresholds, counts, values: sums.map((s, l) => (counts[l] + l2 > 0 ? s / (counts[l] + l2) : 0)) };
}

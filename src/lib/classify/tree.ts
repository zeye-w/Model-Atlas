import type { Point } from "./data";

/**
 * CART classification tree on two numeric features, grown breadth-first so a
 * laboratory can replay it one split at a time. Every split records the full
 * threshold scan it chose from.
 */

export type Criterion = "gini" | "entropy";
export interface TreeOptions { maxDepth: number; minLeaf: number; criterion: Criterion }

export interface ScanPoint { t: number; gain: number }
export interface Split { feature: 0 | 1; threshold: number; gain: number; scan: [ScanPoint[], ScanPoint[]] }

export interface TreeNode {
  id: number; depth: number; parent: number | null;
  /** Region of the plane this node covers: [x₀min, x₀max, x₁min, x₁max]. */
  bounds: [number, number, number, number];
  indices: number[];
  counts: number[];
  impurity: number;
  /** Majority class (ties → lowest class index). */
  label: number;
  split: Split | null;
  /** Position of this node's split in growth order (0-based), or null for a leaf. */
  order: number | null;
  left: number | null; right: number | null;
}

export interface GrownTree { nodes: TreeNode[]; splits: number[]; classes: number }

export function impurity(counts: number[], criterion: Criterion) {
  const n = counts.reduce((a, b) => a + b, 0);
  if (n === 0) return 0;
  if (criterion === "gini") return 1 - counts.reduce((s, c) => s + (c / n) ** 2, 0);
  return -counts.reduce((s, c) => (c ? s + (c / n) * Math.log2(c / n) : s), 0);
}

const argmax = (v: number[]) => v.reduce((b, x, i) => (x > v[b] ? i : b), 0);

/**
 * Best split of `indices` on one feature: scans every midpoint between
 * consecutive distinct values and returns the impurity decrease
 *   I(parent) − (n_L/n) I(L) − (n_R/n) I(R)
 * for each, plus the best. Respects `minLeaf`.
 */
export function scanFeature(points: Point[], indices: number[], feature: 0 | 1, classes: number, criterion: Criterion, minLeaf: number) {
  const sorted = [...indices].sort((a, b) => points[a].x[feature] - points[b].x[feature] || a - b);
  const total = new Array(classes).fill(0);
  for (const i of sorted) total[points[i].y]++;
  const n = sorted.length, parent = impurity(total, criterion);
  const left = new Array(classes).fill(0);
  const scan: ScanPoint[] = [];
  let best: ScanPoint | null = null;
  for (let k = 0; k < n - 1; k++) {
    left[points[sorted[k]].y]++;
    const a = points[sorted[k]].x[feature], b = points[sorted[k + 1]].x[feature];
    if (a === b) continue;
    const nl = k + 1, nr = n - nl;
    if (nl < minLeaf || nr < minLeaf) continue;
    const right = total.map((c, j) => c - left[j]);
    const gain = parent - (nl / n) * impurity(left, criterion) - (nr / n) * impurity(right, criterion);
    const point = { t: (a + b) / 2, gain };
    scan.push(point);
    if (!best || gain > best.gain + 1e-12) best = point;
  }
  return { scan, best };
}

export function growTree(points: Point[], classes: number, options: TreeOptions, bounds: [number, number, number, number] = [-2, 2, -2, 2]): GrownTree {
  const nodes: TreeNode[] = [];
  const splits: number[] = [];
  const make = (indices: number[], depth: number, parent: number | null, box: TreeNode["bounds"]): TreeNode => {
    const counts = new Array(classes).fill(0);
    for (const i of indices) counts[points[i].y]++;
    const node: TreeNode = { id: nodes.length, depth, parent, bounds: box, indices, counts, impurity: impurity(counts, options.criterion), label: argmax(counts), split: null, order: null, left: null, right: null };
    nodes.push(node);
    return node;
  };
  const queue = [make(points.map((_, i) => i), 0, null, bounds)];
  while (queue.length) {
    const node = queue.shift()!;
    if (node.depth >= options.maxDepth || node.impurity === 0 || node.indices.length < 2 * options.minLeaf) continue;
    const s0 = scanFeature(points, node.indices, 0, classes, options.criterion, options.minLeaf);
    const s1 = scanFeature(points, node.indices, 1, classes, options.criterion, options.minLeaf);
    const pick = !s1.best || (s0.best && s0.best.gain >= s1.best.gain) ? 0 : 1;
    const best = pick === 0 ? s0.best : s1.best;
    if (!best || best.gain <= 1e-12) continue;
    node.split = { feature: pick, threshold: best.t, gain: best.gain, scan: [s0.scan, s1.scan] };
    node.order = splits.length;
    splits.push(node.id);
    const [x0a, x0b, x1a, x1b] = node.bounds;
    const leftBox: TreeNode["bounds"] = pick === 0 ? [x0a, best.t, x1a, x1b] : [x0a, x0b, x1a, best.t];
    const rightBox: TreeNode["bounds"] = pick === 0 ? [best.t, x0b, x1a, x1b] : [x0a, x0b, best.t, x1b];
    const l = make(node.indices.filter((i) => points[i].x[pick] <= best.t), node.depth + 1, node.id, leftBox);
    const r = make(node.indices.filter((i) => points[i].x[pick] > best.t), node.depth + 1, node.id, rightBox);
    node.left = l.id; node.right = r.id;
    queue.push(l, r);
  }
  return { nodes, splits, classes };
}

/** Is `node` split once the first `steps` splits have been applied? */
export const isSplit = (node: TreeNode, steps: number) => node.order !== null && node.order < steps;

/** Predict with the tree as it stands after `steps` splits. */
export function predictTree(tree: GrownTree, x: [number, number], steps = tree.splits.length) {
  let node = tree.nodes[0];
  while (isSplit(node, steps)) node = tree.nodes[x[node.split!.feature] <= node.split!.threshold ? node.left! : node.right!];
  return node.label;
}

/** Leaves of the tree after `steps` splits. */
export function leavesAt(tree: GrownTree, steps: number) {
  return tree.nodes.filter((n) => (n.parent === null || isSplit(tree.nodes[n.parent], steps)) && !isSplit(n, steps));
}

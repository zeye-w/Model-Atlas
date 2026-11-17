import assert from "node:assert/strict";
import test from "node:test";
import { accuracy, makeClassData, type DatasetKind, type Point } from "../src/lib/classify/data";
import { applyTransform, distance, predictKnn, rankNeighbors, standardizer, stretch, vote, mapPoints } from "../src/lib/classify/knn";
import { growTree, impurity, leavesAt, predictTree, scanFeature } from "../src/lib/classify/tree";

const near = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `expected ${b}, got ${a}`);
const KINDS: DatasetKind[] = ["blobs", "moons", "xor"];

test("datasets are seeded, balanced and split two thirds / one third", () => {
  for (const kind of KINDS) {
    const d = makeClassData(kind, 3);
    assert.deepEqual(d, makeClassData(kind, 3));
    assert.notDeepEqual(d, makeClassData(kind, 4));
    for (let c = 0; c < d.classes; c++) {
      assert.equal(d.train.filter((p) => p.y === c).length, 33);
      assert.equal(d.test.filter((p) => p.y === c).length, 17);
    }
  }
});

test("impurity: Gini and entropy on known distributions", () => {
  near(impurity([5, 5], "gini"), .5); near(impurity([5, 5], "entropy"), 1);
  near(impurity([10, 0], "gini"), 0); near(impurity([10, 0], "entropy"), 0);
  near(impurity([1, 1, 1, 1], "entropy"), 2);
  near(impurity([2, 1, 1], "gini"), 1 - (.25 + .0625 + .0625));
});

test("threshold scan picks the split with the largest impurity decrease (brute force check)", () => {
  const d = makeClassData("moons", 7);
  const idx = d.train.map((_, i) => i);
  for (const f of [0, 1] as const) {
    const { scan, best } = scanFeature(d.train, idx, f, 2, "gini", 1);
    const values = [...new Set(d.train.map((p) => p.x[f]))].sort((a, b) => a - b);
    let brute = -Infinity;
    for (let i = 0; i < values.length - 1; i++) {
      const t = (values[i] + values[i + 1]) / 2;
      const L = [0, 0], R = [0, 0];
      for (const p of d.train) (p.x[f] <= t ? L : R)[p.y]++;
      const n = d.train.length, nl = L[0] + L[1];
      brute = Math.max(brute, impurity([33, 33], "gini") - (nl / n) * impurity(L, "gini") - ((n - nl) / n) * impurity(R, "gini"));
    }
    near(best!.gain, brute, 1e-12);
    assert.equal(scan.length, values.length - 1);
  }
});

test("tree: children partition their parent, depth and leaf limits hold, pure nodes are not split", () => {
  const d = makeClassData("blobs", 2);
  const tree = growTree(d.train, 3, { maxDepth: 4, minLeaf: 3, criterion: "gini" });
  for (const n of tree.nodes) {
    assert.ok(n.depth <= 4);
    assert.ok(n.indices.length >= 3 || n.parent === null);
    if (n.split) {
      const l = tree.nodes[n.left!], r = tree.nodes[n.right!];
      assert.equal(l.indices.length + r.indices.length, n.indices.length);
      assert.ok(l.indices.every((i) => d.train[i].x[n.split!.feature] <= n.split!.threshold));
      assert.ok(r.indices.every((i) => d.train[i].x[n.split!.feature] > n.split!.threshold));
      assert.ok(n.split.gain > 0);
    } else if (n.impurity === 0) assert.equal(n.split, null);
  }
  // Breadth-first growth: splits are ordered by depth.
  const depths = tree.splits.map((id) => tree.nodes[id].depth);
  assert.deepEqual(depths, [...depths].sort((a, b) => a - b));
});

test("tree: replaying k splits gives k + 1 leaves; the full replay equals the full tree", () => {
  const d = makeClassData("xor", 5);
  const tree = growTree(d.train, 2, { maxDepth: 6, minLeaf: 1, criterion: "entropy" });
  for (let k = 0; k <= tree.splits.length; k++) assert.equal(leavesAt(tree, k).length, k + 1);
  const leaves = leavesAt(tree, tree.splits.length);
  assert.equal(leaves.reduce((s, l) => s + l.indices.length, 0), d.train.length);
  for (const p of d.train) assert.equal(predictTree(tree, p.x), predictTree(tree, p.x, tree.splits.length));
});

test("tree: unlimited depth fits the training data exactly; deeper trees fit training data at least as well", () => {
  for (const kind of KINDS) {
    const d = makeClassData(kind, 1);
    const full = growTree(d.train, d.classes, { maxDepth: 30, minLeaf: 1, criterion: "gini" });
    near(accuracy((x) => predictTree(full, x), d.train), 1);
    let prev = 0;
    for (let depth = 1; depth <= 8; depth++) {
      const t = growTree(d.train, d.classes, { maxDepth: depth, minLeaf: 1, criterion: "gini" });
      const acc = accuracy((x) => predictTree(t, x), d.train);
      assert.ok(acc >= prev - 1e-12, `${kind} depth ${depth}`); prev = acc;
    }
  }
});

test("tree: greedy splitting struggles on XOR — the first split barely helps, extra depth is needed", () => {
  for (const seed of [1, 2, 4, 5, 6]) {
    const d = makeClassData("xor", seed);
    const root = growTree(d.train, 2, { maxDepth: 1, minLeaf: 1, criterion: "gini" }).nodes[0];
    assert.ok(root.split!.gain < .05, `seed ${seed}: root gain ${root.split!.gain}`);   // vs 0.5 parent impurity
    const deep = growTree(d.train, 2, { maxDepth: 4, minLeaf: 1, criterion: "gini" });
    assert.ok(accuracy((x) => predictTree(deep, x), d.test) >= .9, `seed ${seed}`);
  }
});

test("KNN: neighbours are sorted by distance; L1 and L2 behave as defined", () => {
  near(distance([0, 0], [3, 4], "euclidean"), 5); near(distance([0, 0], [3, -4], "manhattan"), 7);
  const d = makeClassData("blobs", 9);
  const q: [number, number] = [.1, .2];
  const ranked = rankNeighbors(d.train, q, "euclidean");
  for (let i = 1; i < ranked.length; i++) assert.ok(ranked[i].distance >= ranked[i - 1].distance);
  near(ranked[0].distance, Math.min(...d.train.map((p) => Math.hypot(p.x[0] - q[0], p.x[1] - q[1]))));
});

test("KNN: voting, distance weights and tie-breaking", () => {
  const train: Point[] = [{ x: [0, 0], y: 0 }, { x: [1, 0], y: 1 }, { x: [2, 0], y: 1 }, { x: [3, 0], y: 0 }];
  const r = rankNeighbors(train, [.4, 0], "euclidean");
  assert.equal(vote(r, 2, { k: 1, metric: "euclidean", weighting: "uniform" }).label, 0);
  assert.equal(vote(r, 2, { k: 3, metric: "euclidean", weighting: "uniform" }).label, 1);                 // 2 of 3 are class 1
  const w = vote(r, 2, { k: 3, metric: "euclidean", weighting: "distance" });
  near(w.scores[0], 1 / .4); near(w.scores[1], 1 / .6 + 1 / 1.6);
  assert.equal(w.label, 0);                                                                                  // 2.5 > 2.29
  assert.equal(vote(r, 2, { k: 2, metric: "euclidean", weighting: "uniform" }).label, 0);                 // 1–1 tie → nearest
  assert.equal(predictKnn(train, 2, [1, 0], { k: 3, metric: "euclidean", weighting: "distance" }), 1);   // exact match decides
});

test("KNN: k = 1 memorises the training set; held-out accuracy beats chance on every dataset", () => {
  for (const kind of KINDS) {
    const d = makeClassData(kind, 2);
    near(accuracy((x) => predictKnn(d.train, d.classes, x, { k: 1, metric: "euclidean", weighting: "uniform" }), d.train), 1);
    const acc = accuracy((x) => predictKnn(d.train, d.classes, x, { k: 7, metric: "euclidean", weighting: "uniform" }), d.test);
    assert.ok(acc > 1 / d.classes + .3, `${kind}: ${acc}`);
  }
});

test("KNN: an unscaled feature hurts accuracy and standardising restores it (either feature)", () => {
  for (const [seed, feature] of [[6, 1], [2, 0], [5, 0]] as const) {
  const d = makeClassData("moons", seed);
  const opt = { k: 7, metric: "euclidean" as const, weighting: "uniform" as const };
  const plain = accuracy((x) => predictKnn(d.train, 2, x, opt), d.test);
  const tr = stretch(d.train, 10, feature), te = stretch(d.test, 10, feature);
  const stretched = accuracy((x) => predictKnn(tr, 2, x, opt), te);
  const z = standardizer(tr);
  const fixed = accuracy((x) => predictKnn(mapPoints(z, tr), 2, applyTransform(z, x), opt), te);
  assert.ok(stretched < plain - .05, `stretched ${stretched} vs plain ${plain}`);
  assert.ok(fixed > stretched + .05, `standardised ${fixed} vs stretched ${stretched}`);
  const zm = mapPoints(z, tr);
  near(zm.reduce((s, p) => s + p.x[1], 0) / zm.length, 0, 1e-9);
  near(Math.sqrt(zm.reduce((s, p) => s + p.x[1] ** 2, 0) / zm.length), 1, 1e-9);
  }
});

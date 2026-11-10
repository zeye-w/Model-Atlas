import assert from "node:assert/strict";
import test from "node:test";
import { makeRegressionData, mean, truth } from "../src/lib/ensembles/data";
import { adaBoost, algorithmIds, algorithms, catBoost, fitAlgorithm, fitRbf, gradientBoosting, predictRbf, randomForest, rbfNetwork, solveSpd, weightedMedian } from "../src/lib/ensembles/models";
import { countLeaves, fitObliviousTree, fitTree, obliviousLeaf, predictOblivious, predictTree } from "../src/lib/ensembles/tree";

const near = (a: number, b: number, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `expected ${b}, got ${a}`);
const data = makeRegressionData(42);

test("data are seeded, split 80/40, sorted by x and drawn around the true function", () => {
  assert.deepEqual(data, makeRegressionData(42));
  assert.notDeepEqual(data, makeRegressionData(43));
  assert.equal(data.train.length, 80); assert.equal(data.test.length, 40);
  for (const set of [data.train, data.test]) for (let i = 1; i < set.length; i++) assert.ok(set[i].x >= set[i - 1].x);
  const resid = data.train.map((p) => p.y - truth(p.x));
  assert.ok(Math.abs(mean(resid)) < .1);
});

test("CART: an unrestricted tree interpolates distinct points; splits only reduce error", () => {
  const xs = [0, .1, .2, .3, .4, .5], ys = [3, -1, 4, 1, -5, 9];
  const deep = fitTree(xs, ys, { maxDepth: 10, minLeaf: 1 });
  xs.forEach((x, i) => near(predictTree(deep, x), ys[i]));
  assert.equal(countLeaves(deep), 6);
  const stump = fitTree(xs, ys, { maxDepth: 1, minLeaf: 1 });
  const sse = (t: typeof stump) => xs.reduce((s, x, i) => s + (predictTree(t, x) - ys[i]) ** 2, 0);
  const m = mean(ys);
  assert.ok(sse(stump) < ys.reduce((s, y) => s + (y - m) ** 2, 0));
  assert.equal(countLeaves(fitTree(xs, ys, { maxDepth: 10, minLeaf: 3 })), 2);
});

test("CART: weights act like repeated samples", () => {
  const xs = [0, 1, 2, 3], ys = [0, 0, 10, 10];
  const weighted = fitTree(xs, ys, { maxDepth: 0, minLeaf: 1 }, [3, 1, 1, 1]);
  near(predictTree(weighted, 0), 20 / 6);
});

test("symmetric tree: one threshold per level, at most depth + 1 non-empty leaves in 1-D, L2-shrunk leaves", () => {
  const xs = data.train.map((p) => p.x), ys = data.train.map((p) => p.y);
  const t = fitObliviousTree(xs, ys, 3, 0);
  assert.equal(t.thresholds.length, 3); assert.equal(t.values.length, 8);
  assert.ok(t.counts.filter((c) => c > 0).length <= 4);
  const leaf = obliviousLeaf(t, xs[10]);
  const members = xs.map((x, i) => [x, ys[i]]).filter(([x]) => obliviousLeaf(t, x) === leaf).map(([, y]) => y);
  near(predictOblivious(t, xs[10]), mean(members));
  const shrunk = fitObliviousTree(xs, ys, 3, 3);
  near(shrunk.values[leaf], members.reduce((a, b) => a + b, 0) / (members.length + 3));
});

test("gradient boosting: each tree fits the current residuals and training error never rises", () => {
  const r = gradientBoosting(data, { rounds: 40, learningRate: .2, maxDepth: 2, minLeaf: 2 });
  assert.equal(r.stages.length, 40);
  near(mean(r.stages[0].pointValues!), 0, 1e-9);              // first residuals are y − mean(y)
  for (let i = 1; i < r.stages.length; i++) assert.ok(r.stages[i].train <= r.stages[i - 1].train + 1e-12);
  assert.ok(r.stages.at(-1)!.test < r.baseline.test * .5);
});

test("CatBoost-style boosting: training error never rises; stronger l2 shrinks the first step", () => {
  const weak = catBoost(data, { rounds: 1, learningRate: 1, depth: 3, l2LeafReg: 0 });
  const strong = catBoost(data, { rounds: 1, learningRate: 1, depth: 3, l2LeafReg: 20 });
  const amp = (s: number[]) => Math.max(...s.map(Math.abs));
  assert.ok(amp(strong.stages[0].parts[0]) < amp(weak.stages[0].parts[0]));
  const r = catBoost(data, algorithms.catboost.defaults);
  for (let i = 1; i < r.stages.length; i++) assert.ok(r.stages[i].train <= r.stages[i - 1].train + 1e-12);
  assert.ok(r.stages.at(-1)!.test < r.baseline.test * .5);
});

test("random forest: prediction is the mean of its trees and is seed-reproducible", () => {
  const p = { trees: 12, maxDepth: 5, minLeaf: 2, seed: 3 };
  const a = randomForest(data, p);
  assert.deepEqual(a, randomForest(data, p));
  const g = 100;
  const avg = a.stages.reduce((s, st) => s + st.parts[0][g], 0) / a.stages.length;
  near(a.stages.at(-1)!.curve[g], avg);
  for (const st of a.stages) assert.equal(st.pointValues!.reduce((x, y) => x + y, 0), data.train.length); // bootstrap draws n
  assert.ok(a.stages.at(-1)!.test < a.baseline.test * .6);
});

test("AdaBoost.R2: weighted median, normalised weights, model weights ln(1/β) > 0", () => {
  assert.equal(weightedMedian([5, 1, 3], [1, 1, 1]), 3);
  assert.equal(weightedMedian([5, 1, 3], [1, 1, 5]), 3);
  assert.equal(weightedMedian([5, 1, 3], [10, 1, 1]), 5);
  const r = adaBoost(data, algorithms.adaboost.defaults);
  for (const st of r.stages) near(st.pointValues!.reduce((a, b) => a + b, 0), 1, 1e-9);
  assert.ok(r.stages.at(-1)!.test < r.baseline.test * .6);
});

test("RBF network: Cholesky solves SPD systems; ridge solution satisfies the normal equations", () => {
  const x = solveSpd([[4, 2], [2, 3]], [2, 1]);
  near(4 * x[0] + 2 * x[1], 2); near(2 * x[0] + 3 * x[1], 1);
  const xs = data.train.map((p) => p.x), ys = data.train.map((p) => p.y);
  const net = fitRbf(xs, ys, 6, .9, .01);
  // Gradient of ½‖Φw + b − y‖² + ½λ‖w‖² vanishes at the solution.
  for (let k = 0; k < 6; k++) {
    const g = xs.reduce((s, x, i) => s + (predictRbf(net, x) - ys[i]) * Math.exp(-((x - net.centers[k]) ** 2) / (2 * net.sigma ** 2)), 0) + .01 * net.weights[k];
    near(g, 0, 1e-6);
  }
  near(xs.reduce((s, x, i) => s + predictRbf(net, x) - ys[i], 0), 0, 1e-6);
  const r = rbfNetwork(data, algorithms["rbf-network"].defaults);
  assert.equal(r.stages.length, 16);
  assert.ok(r.stages.at(-1)!.train < r.stages[0].train);
  assert.ok(r.stages.at(-1)!.test < r.baseline.test * .6);
});

test("every algorithm, at its defaults, beats predicting the mean on held-out data", () => {
  for (const id of algorithmIds) {
    const r = fitAlgorithm(id, data, algorithms[id].defaults as unknown as Record<string, number>);
    assert.ok(r.stages.length > 0, id);
    const last = r.stages.at(-1)!;
    assert.ok(last.test < r.baseline.test, `${id}: ${last.test} vs ${r.baseline.test}`);
    assert.equal(last.curve.length, data.grid.length);
  }
});

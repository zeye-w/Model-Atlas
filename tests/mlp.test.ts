import assert from "node:assert/strict";
import test from "node:test";
import { evaluate, forward, loss, makeDataset, makeNetwork, sampleOrder, sgdStep, softmax, type Network } from "../src/lib/mlp";

function near(actual: number, expected: number, tolerance = 1e-7) {
  assert.ok(Math.abs(actual - expected) < tolerance, `expected ${expected}, got ${actual}`);
}

function numeric(net: Network, read: (n: Network) => number[], index: number, f: (n: Network) => number) {
  const eps = 1e-6;
  const up = structuredClone(net); read(up)[index] += eps;
  const down = structuredClone(net); read(down)[index] -= eps;
  return (f(up) - f(down)) / (2 * eps);
}

test("dataset and network are seeded and repeatable; classes are balanced XOR quadrants", () => {
  assert.equal(makeNetwork(7, 5).w1.length, 5);
  assert.equal(makeNetwork(7, 5).w2[0].length, 5);
  assert.throws(() => makeNetwork(7, 0));
  const data = makeDataset(7);
  assert.deepEqual(data, makeDataset(7));
  assert.notDeepEqual(data, makeDataset(8));
  assert.deepEqual(makeNetwork(7), makeNetwork(7));
  assert.equal(data.filter((s) => s.y === 0).length, data.length / 2);
  for (const s of data) assert.equal(s.y, Math.sign(s.x[0]) * Math.sign(s.x[1]) > 0 ? 0 : 1);
});

test("softmax is stable and sums to one", () => {
  const p = softmax([1000, 1000 + Math.log(3)]);
  near(p[0], .25); near(p[1], .75);
});

test("trace gradients match finite differences for every parameter, at several widths", () => {
  for (const width of [2, 3, 6, 8]) {
  const net = makeNetwork(3, width);
  const sample = makeDataset(3)[5];
  const trace = sgdStep(net, sample, .1);
  const f = (n: Network) => loss(forward(n, sample.x).p, sample.y);
  trace.gradW1.forEach((row, j) => row.forEach((g, i) => near(numeric(net, (n) => n.w1[j], i, f), g, 1e-6)));
  trace.gradB1.forEach((g, j) => near(numeric(net, (n) => n.b1, j, f), g, 1e-6));
  trace.gradW2.forEach((row, k) => row.forEach((g, j) => near(numeric(net, (n) => n.w2[k], j, f), g, 1e-6)));
  trace.gradB2.forEach((g, k) => near(numeric(net, (n) => n.b2, k, f), g, 1e-6));
  assert.equal(trace.h.length, width);
  }
});

test("trace is internally consistent and does not mutate its input", () => {
  const net = makeNetwork(11);
  const copy = structuredClone(net);
  const trace = sgdStep(net, makeDataset(11)[0], .2);
  assert.deepEqual(net, copy);
  assert.deepEqual(trace.before, copy);
  near(trace.after.w1[1][0], trace.before.w1[1][0] - .2 * trace.gradW1[1][0]);
  near(trace.after.b2[1], trace.before.b2[1] - .2 * trace.gradB2[1]);
  trace.delta1.forEach((d, j) => near(d, trace.backHidden[j] * (1 - trace.h[j] ** 2)));
  near(trace.p[0] + trace.p[1], 1);
});

test("a small step lowers the loss on the sample it was computed from", () => {
  const net = makeNetwork(5);
  for (const sample of makeDataset(5).slice(0, 10)) {
    const trace = sgdStep(net, sample, .05);
    assert.ok(trace.lossAfter < trace.loss, `${trace.lossAfter} < ${trace.loss}`);
  }
});

test("SGD at the lab's default rate learns XOR at widths 3–8 for every seed tried", () => {
  for (const width of [3, 6, 8]) for (const seed of [1, 2, 3, 14, 42]) {
  const data = makeDataset(seed);
  let net = makeNetwork(seed, width);
  const initial = evaluate(net, data);
  for (let epoch = 0; epoch < 30; epoch++) {
    for (const i of sampleOrder(data.length, seed, epoch)) net = sgdStep(net, data[i], .1).after;
  }
  const final = evaluate(net, data);
  assert.ok(final.loss < initial.loss * .4, `width ${width}, seed ${seed}: loss ${initial.loss} → ${final.loss}`);
  assert.ok(final.accuracy >= .95, `width ${width}, seed ${seed}: accuracy ${final.accuracy}`);
  }
});

test("the lab's hint holds: with 2 hidden units, seed 4 gets stuck on XOR", () => {
  const data = makeDataset(4);
  let net = makeNetwork(4, 2);
  for (let epoch = 0; epoch < 60; epoch++) for (const i of sampleOrder(data.length, 4, epoch)) net = sgdStep(net, data[i], .1).after;
  assert.ok(evaluate(net, data).accuracy < .9);
});

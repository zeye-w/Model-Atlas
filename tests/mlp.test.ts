import assert from "node:assert/strict";
import test from "node:test";
import { activations, evaluate, forward, loss, makeDataset, makeNetwork, makePhases, sampleOrder, sgdStep, softmax, type Activation, type Network } from "../src/lib/mlp";

function near(actual: number, expected: number, tolerance = 1e-7) {
  assert.ok(Math.abs(actual - expected) < tolerance, `expected ${expected}, got ${actual}`);
}

const ACTS: Activation[] = ["tanh", "sigmoid", "relu"];

function trainEpochs(net: Network, seed: number, epochs: number, rate = .1) {
  const data = makeDataset(seed);
  for (let e = 0; e < epochs; e++) for (const i of sampleOrder(data.length, seed, e)) net = sgdStep(net, data[i], rate).after;
  return { net, data };
}

test("dataset and network are seeded and repeatable; classes are balanced XOR quadrants", () => {
  const data = makeDataset(7);
  assert.deepEqual(data, makeDataset(7));
  assert.notDeepEqual(data, makeDataset(8));
  assert.deepEqual(makeNetwork(7), makeNetwork(7));
  assert.equal(data.filter((s) => s.y === 0).length, data.length / 2);
  for (const s of data) assert.equal(s.y, Math.sign(s.x[0]) * Math.sign(s.x[1]) > 0 ? 0 : 1);
  assert.throws(() => makeNetwork(7, 0));
  assert.throws(() => makeNetwork(7, []));
});

test("layer shapes follow the requested depth and width", () => {
  const net = makeNetwork(1, [5, 4, 3], "relu");
  assert.deepEqual(net.W.map((w) => [w.length, w[0].length]), [[5, 2], [4, 5], [3, 4], [2, 3]]);
  assert.deepEqual(net.b.map((b) => b.length), [5, 4, 3, 2]);
  const { z, a, p } = forward(net, [.3, -.2]);
  assert.deepEqual(z.map((v) => v.length), [5, 4, 3, 2]);
  assert.deepEqual(a.map((v) => v.length), [2, 5, 4, 3]);
  near(p[0] + p[1], 1);
});

test("softmax is stable and sums to one; activation derivatives match finite differences", () => {
  const p = softmax([1000, 1000 + Math.log(3)]);
  near(p[0], .25); near(p[1], .75);
  for (const id of ACTS) for (const z of [-1.3, -.2, .4, 2.1]) {
    const { f, df } = activations[id];
    near(df(z, f(z)), (f(z + 1e-6) - f(z - 1e-6)) / 2e-6, 1e-6);
  }
});

test("trace gradients match finite differences for every parameter, at every depth and activation", () => {
  for (const act of ACTS) for (const widths of [[3], [4, 3], [3, 4, 2]]) {
    const net = makeNetwork(3, widths, act);
    const sample = makeDataset(3)[5];
    const trace = sgdStep(net, sample, .1);
    const f = (n: Network) => loss(forward(n, sample.x).p, sample.y);
    const numeric = (mutate: (n: Network, eps: number) => void) => {
      const up = structuredClone(net); mutate(up, 1e-6);
      const down = structuredClone(net); mutate(down, -1e-6);
      return (f(up) - f(down)) / 2e-6;
    };
    trace.gradW.forEach((Wl, l) => Wl.forEach((row, j) => row.forEach((g, i) => {
      // Skip ReLU units sitting exactly at the kink, where the derivative is undefined.
      if (act === "relu" && l < widths.length && Math.abs(trace.z[l][j]) < 1e-5) return;
      near(numeric((n, e) => { n.W[l][j][i] += e; }), g, 1e-6);
    })));
    trace.gradB.forEach((bl, l) => bl.forEach((g, j) => near(numeric((n, e) => { n.b[l][j] += e; }), g, 1e-6)));
    assert.equal(trace.gradNorms.length, widths.length + 1);
  }
});

test("trace is internally consistent and does not mutate its input", () => {
  for (const act of ACTS) {
    const net = makeNetwork(11, [4, 4], act);
    const copy = structuredClone(net);
    const trace = sgdStep(net, makeDataset(11)[0], .2);
    assert.deepEqual(net, copy);
    assert.deepEqual(trace.before, copy);
    near(trace.after.W[0][1][0], trace.before.W[0][1][0] - .2 * trace.gradW[0][1][0]);
    near(trace.after.b[2][1], trace.before.b[2][1] - .2 * trace.gradB[2][1]);
    trace.delta[1].forEach((d, j) => near(d, trace.back[1][j] * activations[act].df(trace.z[1][j], trace.a[2][j])));
    trace.gradW.forEach((Wl, l) => near(trace.gradNorms[l], Math.sqrt(Wl.flat().reduce((s, v) => s + v * v, 0))));
  }
});

test("a small step lowers the loss on the sample it was computed from", () => {
  for (const act of ACTS) {
    const net = makeNetwork(5, [4, 4], act);
    for (const sample of makeDataset(5).slice(0, 10)) {
      const trace = sgdStep(net, sample, .02);
      if (trace.gradNorms.every((g) => g === 0)) continue;   // all-dead ReLU: nothing to learn
      assert.ok(trace.lossAfter < trace.loss, `${act}: ${trace.lossAfter} < ${trace.loss}`);
    }
  }
});

test("one tanh layer reproduces the original lab: XOR learned at widths 3–8 for every seed tried", () => {
  for (const width of [3, 6, 8]) for (const seed of [1, 2, 3, 14, 42]) {
    const initial = evaluate(makeNetwork(seed, width), makeDataset(seed));
    const { net, data } = trainEpochs(makeNetwork(seed, width), seed, 30);
    const final = evaluate(net, data);
    assert.ok(final.loss < initial.loss * .4, `width ${width}, seed ${seed}: loss ${initial.loss} → ${final.loss}`);
    assert.ok(final.accuracy >= .95, `width ${width}, seed ${seed}: accuracy ${final.accuracy}`);
  }
});

test("the lab's hint holds: with 2 tanh hidden units, seed 4 gets stuck on XOR", () => {
  const { net, data } = trainEpochs(makeNetwork(4, 2), 4, 60);
  assert.ok(evaluate(net, data).accuracy < .9);
});

test("vanishing gradients: deep sigmoid nets give the first layer a tiny share; tanh does not", () => {
  const ratio = (act: Activation, layers: number) => {
    let r = 0, n = 0;
    for (let seed = 1; seed <= 10; seed++) {
      const net = makeNetwork(seed, Array(layers).fill(6), act);
      for (const s of makeDataset(seed).slice(0, 20)) { const t = sgdStep(net, s, .1); r += t.gradNorms[0] / t.gradNorms[layers]; n++; }
    }
    return r / n;
  };
  const s1 = ratio("sigmoid", 1), s3 = ratio("sigmoid", 3), t3 = ratio("tanh", 3);
  assert.ok(s3 < .02, `sigmoid, 3 layers: ${s3}`);
  assert.ok(s3 < s1 / 5, `sigmoid shrinks with depth: ${s1} → ${s3}`);
  assert.ok(t3 > .3, `tanh, 3 layers: ${t3}`);
});

test("the lab's hint holds: 3-layer sigmoid fails XOR in 40 epochs while tanh and ReLU learn it", () => {
  for (const seed of [1, 2, 3]) {
    const res = (act: Activation) => { const { net, data } = trainEpochs(makeNetwork(seed, [6, 6, 6], act), seed, 40); return evaluate(net, data).accuracy; };
    assert.ok(res("sigmoid") < .9, `seed ${seed}: sigmoid`);
    assert.ok(res("tanh") >= .95, `seed ${seed}: tanh`);
    assert.ok(res("relu") >= .95, `seed ${seed}: relu`);
  }
});

test("phases: forward per layer, backward in reverse, one update", () => {
  assert.deepEqual(makePhases(1).map((p) => p.kind), ["input", "sum", "act", "output-sum", "softmax", "output-grad", "backprop", "update"]);
  const p3 = makePhases(3);
  assert.equal(p3.length, 14);
  assert.deepEqual(p3.filter((p) => p.kind === "backprop").map((p) => p.layer), [2, 1, 0]);
  assert.deepEqual(p3.filter((p) => p.kind === "sum").map((p) => p.layer), [0, 1, 2]);
});

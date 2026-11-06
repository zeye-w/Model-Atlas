import assert from "node:assert/strict";
import test from "node:test";
import { conv1d, convBackward, crossEntropy, dense, denseBackward, outputLength, pool, relu, softmax } from "../src/lib/cnn/ops";
import { backward, classificationMetrics, dataset, epochOrder, evaluate, forward, makeModel, trainBatch } from "../src/lib/cnn/training";

function near(actual: number, expected: number, tolerance = 1e-7) {
  assert.ok(Math.abs(actual - expected) < tolerance, `expected ${expected}, got ${actual}`);
}

function difference(row: number[], index: number, loss: () => number) {
  const original = row[index];
  const epsilon = 1e-5;
  row[index] = original + epsilon;
  const upper = loss();
  row[index] = original - epsilon;
  const lower = loss();
  row[index] = original;
  return (upper - lower) / (2 * epsilon);
}

test("Conv1D cross-correlates without reversing taps and sums channels and bias", () => {
  assert.deepEqual(conv1d([[1, 2, 3]], { weights: [[[2, 5]]], bias: [1] }), [[13, 20]]);
  assert.deepEqual(conv1d([[1, 2, 3], [3, 4, 5]], { weights: [[[2, 5], [1, -1]], [[0, 0], [2, 3]]], bias: [1, 0] }), [[12, 19], [18, 23]]);
});

test("stride, symmetric zero padding, dilation, and output geometry", () => {
  const layer = { weights: [[[2, 5]]], bias: [0] };
  assert.deepEqual(conv1d([[1, 2, 3, 4, 5]], layer, { stride: 2, padding: 0, dilation: 1 }), [[12, 26]]);
  assert.deepEqual(conv1d([[1, 2, 3]], layer, { stride: 1, padding: 1, dilation: 1 }), [[5, 12, 19, 6]]);
  assert.deepEqual(conv1d([[1, 2, 3, 4, 5]], layer, { stride: 2, padding: 1, dilation: 2 }), [[10, 24, 8]]);
  assert.equal(outputLength(5, 2, { stride: 2, padding: 1, dilation: 2 }), 3);
  assert.equal(outputLength(2, 5), 0);
});

test("ReLU and max/average pooling", () => {
  assert.deepEqual(relu([[-3, 0, 2]]), [[0, 0, 2]]);
  assert.deepEqual(pool([[1, 4, 3, 2, 7]]), { values: [[4, 3]], indices: [[1, 2]] });
  assert.deepEqual(pool([[1, 4, 3, 2, 7]], 2, 2, true).values, [[2.5, 2.5]]);
});

test("softmax and cross entropy use stable logits", () => {
  const logits = [1000, 1000 + Math.log(2), 1000];
  softmax(logits).forEach((p, i) => near(p, [.25, .5, .25][i]));
  near(crossEntropy(logits, 0), Math.log(4));
  near(crossEntropy([1000, -1000], 1), 2000);
});

test("Conv1D finite differences verify weights, bias, and input with nontrivial geometry", () => {
  const input = [[1, 2, 4, 5, 3], [-2, -1, 0, 3, 2]];
  const layer = { weights: [[[.4, .5], [.7, -.8]]], bias: [.3] };
  const geometry = { stride: 2, padding: 1, dilation: 2 };
  const delta = [[.1, -.4, .8]];
  const grad = convBackward(input, layer, delta, geometry);
  const loss = () => conv1d(input, layer, geometry)[0].reduce((sum, v, t) => sum + v * delta[0][t], 0);
  layer.weights[0].forEach((row, c) => row.forEach((_, k) => near(difference(row, k, loss), grad.weights[0][c][k])));
  input.forEach((row, c) => row.forEach((_, t) => near(difference(row, t, loss), grad.dx[c][t])));
  near(difference(layer.bias, 0, loss), grad.bias[0]);
});

test("Dense finite differences verify weights, bias, and input", () => {
  const x = [2, -3];
  const layer = { weights: [[.2, .4], [-.1, .6]], bias: [.4, .8] };
  const delta = [.7, -.2];
  const grad = denseBackward(x, layer, delta);
  const loss = () => dense(x, layer).reduce((sum, v, i) => sum + v * delta[i], 0);
  layer.weights.forEach((row, j) => row.forEach((_, i) => near(difference(row, i, loss), grad.weights[j][i])));
  layer.bias.forEach((_, j) => near(difference(layer.bias, j, loss), grad.bias[j]));
  x.forEach((_, i) => near(difference(x, i, loss), grad.dx[i]));
});

test("full backprop agrees with finite differences across both convolutions and the head", () => {
  const model = makeModel(42);
  const sample = dataset(42).train[0];
  const state = forward(model, sample.signal);
  const grad = backward(model, state, sample.label);
  const loss = () => crossEntropy(forward(model, sample.signal).logits, sample.label);
  for (const layer of ["first", "second"] as const) {
    model[layer].weights.forEach((kernel, f) => kernel.forEach((row, c) => row.forEach((_, k) => {
      near(difference(row, k, loss), grad[layer].weights[f][c][k]);
    })));
  }
  model.head.weights.forEach((row, j) => row.forEach((_, i) => near(difference(row, i, loss), grad.head.weights[j][i])));
});

test("seeded data and model are repeatable; partitions are balanced and disjoint", () => {
  const samples = dataset(42);
  assert.deepEqual(samples, dataset(42));
  assert.notDeepEqual(samples, dataset(43));
  assert.deepEqual(makeModel(42), makeModel(42));
  const sets = [samples.train, samples.validation, samples.test];
  assert.deepEqual(sets.map(s => s.length), [210, 45, 45]);
  assert.equal(new Set(sets.flat().map(s => s.id)).size, 300);
  for (const set of sets) {
    assert.ok(set.every(s => s.signal.length === 64));
    assert.deepEqual([0, 1, 2].map(label => set.filter(s => s.label === label).length), Array(3).fill(set.length / 3));
  }
  const order = epochOrder(samples.train, 42, 1);
  assert.deepEqual(order, epochOrder(samples.train, 42, 1));
  assert.deepEqual(new Set(order), new Set(samples.train));
});

test("seeded SGD reduces real loss and trace matches the update", () => {
  const samples = dataset(42);
  const model = makeModel(42);
  const initial = evaluate(model, samples.train).loss;
  const repeat = makeModel(42);
  const firstBatch = epochOrder(samples.train, 42, 1).slice(0, 16);
  const trace = trainBatch(repeat, firstBatch, .05);
  near(trace.after.first.weights[0][0][0], trace.before.first.weights[0][0][0] - .05 * trace.gradients.first.weights[0][0][0]);
  for (let epoch = 1; epoch <= 10; epoch++) {
    const order = epochOrder(samples.train, 42, epoch);
    for (let offset = 0; offset < order.length; offset += 16) {
      const update = trainBatch(model, order.slice(offset, offset + 16), .05);
      if (epoch === 1 && offset === 0) assert.deepEqual(update, trace);
    }
  }
  assert.ok(evaluate(model, samples.train).loss < initial * .65);
  assert.ok(evaluate(model, samples.validation).accuracy > .7);
});

test("confusion matrix rows are actual classes; precision and recall use the correct margins", () => {
  const result = classificationMetrics([0, 0, 0, 1, 1, 2], [0, 0, 1, 1, 2, 1]);
  assert.deepEqual(result.matrix, [[2, 1, 0], [0, 1, 1], [0, 1, 0]]);
  near(result.accuracy, .5);
  near(result.scores[0].precision, 1);
  near(result.scores[0].recall, 2 / 3);
  near(result.scores[0].f1, .8);
  near(result.scores[2].f1, 0);
});

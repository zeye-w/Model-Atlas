import assert from "node:assert/strict";
import test from "node:test";
import { gradientStep, gradients, initialState, makeDataset, mse } from "../src/lib/regression";

test("seeded datasets are reproducible", () => {
  assert.deepEqual(makeDataset(42, 0.3), makeDataset(42, 0.3));
  assert.notDeepEqual(makeDataset(42, 0.3), makeDataset(43, 0.3));
});

test("zero-noise data follows the generating line", () => {
  const data = makeDataset(3, 0);
  assert.ok(mse(data, { w: 1.7, b: 1.15 }) < 1e-25);
});

test("analytical gradients agree with finite differences", () => {
  const data = makeDataset(7, 0.25);
  const params = { w: 0.4, b: -0.2 };
  const actual = gradients(data, params);
  const h = 1e-5;
  const dw = (mse(data, { ...params, w: params.w + h }) - mse(data, { ...params, w: params.w - h })) / (2 * h);
  const db = (mse(data, { ...params, b: params.b + h }) - mse(data, { ...params, b: params.b - h })) / (2 * h);
  assert.ok(Math.abs(actual.w - dw) < 1e-7);
  assert.ok(Math.abs(actual.b - db) < 1e-7);
});

test("gradient descent reduces loss", () => {
  const data = makeDataset(42, 0.3);
  let current = initialState(data);
  const originalLoss = current.loss;
  for (let i = 0; i < 150; i++) current = gradientStep(data, current, 0.09);
  // With nonzero observational noise, irreducible residual error remains.
  assert.ok(current.loss < originalLoss / 20);
  assert.ok(Math.abs(current.w - 1.7) < 0.1);
  assert.ok(Math.abs(current.b - 1.15) < 0.1);
});

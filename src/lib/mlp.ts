/**
 * A tiny multilayer perceptron (2 → 3 → 2) whose single SGD step is exposed
 * as an explicit trace, so every number in the animation is the number the
 * algorithm actually computed.
 *
 *   z¹ = W¹x + b¹        h = tanh(z¹)
 *   z² = W²h + b²        p = softmax(z²)        L = −ln p_y
 *   δ² = p − onehot(y)   ∂L/∂W² = δ² hᵀ         ∂L/∂b² = δ²
 *   δ¹ = (W²ᵀ δ²) ⊙ (1 − h²)
 *   ∂L/∂W¹ = δ¹ xᵀ       ∂L/∂b¹ = δ¹
 *   θ ← θ − η ∂L/∂θ
 */

export const INPUTS = 2;
export const HIDDEN = 3;
export const OUTPUTS = 2;
export const classNames = ["Class A", "Class B"];

export interface Network { w1: number[][]; b1: number[]; w2: number[][]; b2: number[] }
export interface Sample { x: [number, number]; y: 0 | 1 }

export function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** XOR-style quadrants: not linearly separable, so a hidden layer is required. */
export function makeDataset(seed: number, count = 80, noise = .18): Sample[] {
  const rng = random(seed);
  return Array.from({ length: count }, (_, i) => {
    const sx = i % 2 === 0 ? 1 : -1;
    const sy = i % 4 < 2 ? 1 : -1;
    const x: [number, number] = [sx * (.25 + .6 * rng()) + noise * (rng() - .5), sy * (.25 + .6 * rng()) + noise * (rng() - .5)];
    return { x, y: (sx * sy > 0 ? 0 : 1) as 0 | 1 };
  });
}

export function makeNetwork(seed: number): Network {
  const rng = random(seed ^ 0x51ed);
  const glorot = (fanIn: number, fanOut: number) => (2 * rng() - 1) * Math.sqrt(6 / (fanIn + fanOut));
  return {
    w1: Array.from({ length: HIDDEN }, () => Array.from({ length: INPUTS }, () => glorot(INPUTS, HIDDEN))),
    b1: Array<number>(HIDDEN).fill(0),
    w2: Array.from({ length: OUTPUTS }, () => Array.from({ length: HIDDEN }, () => glorot(HIDDEN, OUTPUTS))),
    b2: Array<number>(OUTPUTS).fill(0),
  };
}

export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function forward(net: Network, x: readonly number[]) {
  const z1 = net.w1.map((row, j) => row.reduce((s, w, i) => s + w * x[i], net.b1[j]));
  const h = z1.map(Math.tanh);
  const z2 = net.w2.map((row, k) => row.reduce((s, w, j) => s + w * h[j], net.b2[k]));
  const p = softmax(z2);
  return { z1, h, z2, p };
}

export const loss = (p: number[], y: number) => -Math.log(Math.max(p[y], 1e-12));

/** Everything computed during one SGD update on one sample. */
export interface StepTrace {
  sample: Sample;
  rate: number;
  before: Network;
  z1: number[]; h: number[]; z2: number[]; p: number[]; loss: number;
  delta2: number[]; gradW2: number[][]; gradB2: number[];
  backHidden: number[];   // W²ᵀ δ², before the tanh derivative
  delta1: number[]; gradW1: number[][]; gradB1: number[];
  after: Network;
  lossAfter: number;
}

export function sgdStep(net: Network, sample: Sample, rate: number): StepTrace {
  const before = structuredClone(net);
  const { z1, h, z2, p } = forward(before, sample.x);
  const L = loss(p, sample.y);
  const delta2 = p.map((v, k) => v - Number(k === sample.y));
  const gradW2 = delta2.map((d) => h.map((hj) => d * hj));
  const gradB2 = [...delta2];
  const backHidden = h.map((_, j) => before.w2.reduce((s, row, k) => s + row[j] * delta2[k], 0));
  const delta1 = backHidden.map((v, j) => v * (1 - h[j] * h[j]));
  const gradW1 = delta1.map((d) => sample.x.map((xi) => d * xi));
  const gradB1 = [...delta1];
  const after: Network = {
    w1: before.w1.map((row, j) => row.map((w, i) => w - rate * gradW1[j][i])),
    b1: before.b1.map((b, j) => b - rate * gradB1[j]),
    w2: before.w2.map((row, k) => row.map((w, j) => w - rate * gradW2[k][j])),
    b2: before.b2.map((b, k) => b - rate * gradB2[k]),
  };
  const lossAfter = loss(forward(after, sample.x).p, sample.y);
  return { sample, rate, before, z1, h, z2, p, loss: L, delta2, gradW2, gradB2, backHidden, delta1, gradW1, gradB1, after, lossAfter };
}

export function evaluate(net: Network, samples: Sample[]) {
  let total = 0, correct = 0;
  for (const s of samples) {
    const { p } = forward(net, s.x);
    total += loss(p, s.y);
    if ((p[1] > p[0] ? 1 : 0) === s.y) correct++;
  }
  return { loss: total / samples.length, accuracy: correct / samples.length };
}

/** Deterministic visiting order: a fresh seeded shuffle each epoch. */
export function sampleOrder(count: number, seed: number, epoch: number) {
  const rng = random(seed ^ Math.imul(epoch + 1, 0x9e3779b9));
  const order = Array.from({ length: count }, (_, i) => i);
  for (let i = count - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  return order;
}

/** The phases the animation walks through for one update. */
export const phases = [
  { id: "input", title: "Input", direction: "forward" },
  { id: "hidden-sum", title: "Weighted sum", direction: "forward" },
  { id: "hidden-act", title: "Activation", direction: "forward" },
  { id: "output-sum", title: "Output logits", direction: "forward" },
  { id: "softmax", title: "Softmax & loss", direction: "forward" },
  { id: "output-grad", title: "Output error", direction: "backward" },
  { id: "hidden-grad", title: "Backpropagate", direction: "backward" },
  { id: "update", title: "Update weights", direction: "update" },
] as const;
export type PhaseId = typeof phases[number]["id"];

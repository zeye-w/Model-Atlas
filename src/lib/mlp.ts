/**
 * A small multilayer perceptron, 2 → H × L → 2 (L = 1…3 hidden layers of
 * width H = 2…8), with tanh, sigmoid or ReLU hidden units and a softmax
 * output. One SGD step is exposed as an explicit trace, so every number in
 * the animation is the number the algorithm actually computed.
 *
 *   a⁰ = x
 *   zˡ = Wˡ aˡ⁻¹ + bˡ,  aˡ = f(zˡ)                 (hidden layers ℓ = 1…L)
 *   z = W^{L+1} a^L + b^{L+1},  p = softmax(z),  L = −ln p_y
 *   δ^{L+1} = p − onehot(y)
 *   δˡ = (W^{ℓ+1}ᵀ δ^{ℓ+1}) ⊙ f′(zˡ)
 *   ∂L/∂Wˡ = δˡ aˡ⁻¹ᵀ,  ∂L/∂bˡ = δˡ,  θ ← θ − η ∂L/∂θ
 *
 * Arrays are indexed from 0: W[0] maps the input to hidden layer 1 and
 * W[L] maps the last hidden layer to the output.
 */

export const INPUTS = 2;
export const OUTPUTS = 2;
/** Default hidden width, and the ranges the lab offers. */
export const HIDDEN = 6;
export const HIDDEN_RANGE = [2, 8] as const;
export const LAYER_RANGE = [1, 3] as const;
export const classNames = ["Class A", "Class B"];

export type Activation = "tanh" | "sigmoid" | "relu";
export const activations: Record<Activation, { name: string; f: (z: number) => number; df: (z: number, a: number) => number; derivative: string }> = {
  tanh: { name: "tanh", f: Math.tanh, df: (_z, a) => 1 - a * a, derivative: "1 − a²" },
  sigmoid: { name: "sigmoid", f: (z) => 1 / (1 + Math.exp(-z)), df: (_z, a) => a * (1 - a), derivative: "a(1 − a)" },
  relu: { name: "ReLU", f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0), derivative: "[z > 0]" },
};

export interface Network { W: number[][][]; b: number[][]; activation: Activation }
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

/** Layer sizes, e.g. [2, 6, 6, 2] for two hidden layers of width 6. */
export const layerSizes = (net: Network) => [net.W[0][0].length, ...net.W.map((w) => w.length)];
export const hiddenLayers = (net: Network) => net.W.length - 1;

/**
 * Glorot-uniform initialisation for tanh and sigmoid, He-uniform for ReLU;
 * zero biases. With one tanh layer the random draws are consumed in the same
 * order as the original single-layer lab, so its results are unchanged.
 */
export function makeNetwork(seed: number, hidden: number | number[] = HIDDEN, activation: Activation = "tanh"): Network {
  const widths = Array.isArray(hidden) ? hidden : [hidden];
  if (widths.length < 1 || widths.some((w) => !Number.isInteger(w) || w < 1)) throw new Error("hidden widths must be positive integers");
  const rng = random(seed ^ 0x51ed);
  const sizes = [INPUTS, ...widths, OUTPUTS];
  const W = sizes.slice(1).map((fanOut, l) => {
    const fanIn = sizes[l];
    const relu = activation === "relu" && l < widths.length;   // the softmax layer stays Glorot
    const limit = relu ? Math.sqrt(6 / fanIn) : Math.sqrt(6 / (fanIn + fanOut));
    return Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => (2 * rng() - 1) * limit));
  });
  return { W, b: sizes.slice(1).map((n) => Array<number>(n).fill(0)), activation };
}

export function softmax(z: number[]) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

const affine = (W: number[][], b: number[], a: readonly number[]) => W.map((row, j) => row.reduce((s, w, i) => s + w * a[i], b[j]));

/** z[ℓ] are pre-activations of every layer (last = output logits); a[0] = x, a[ℓ+1] = f(z[ℓ]) for hidden layers. */
export function forward(net: Network, x: readonly number[]) {
  const f = activations[net.activation].f;
  const z: number[][] = [], a: number[][] = [[...x]];
  const L = hiddenLayers(net);
  for (let l = 0; l < L; l++) { z.push(affine(net.W[l], net.b[l], a[l])); a.push(z[l].map(f)); }
  z.push(affine(net.W[L], net.b[L], a[L]));
  return { z, a, p: softmax(z[L]) };
}

export const loss = (p: number[], y: number) => -Math.log(Math.max(p[y], 1e-12));
const frobenius = (m: number[][]) => Math.sqrt(m.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));

/** Everything computed during one SGD update on one sample. */
export interface StepTrace {
  sample: Sample;
  rate: number;
  before: Network;
  z: number[][]; a: number[][]; p: number[]; loss: number;
  /** δ for every layer; delta[L] is the output error p − y. */
  delta: number[][];
  /** W^{ℓ+1}ᵀ δ^{ℓ+1} for hidden layers, before multiplying by f′. */
  back: number[][];
  gradW: number[][][]; gradB: number[][];
  /** ‖∂L/∂Wˡ‖ (Frobenius) per weight layer — used to show vanishing gradients. */
  gradNorms: number[];
  after: Network;
  lossAfter: number;
}

export function sgdStep(net: Network, sample: Sample, rate: number): StepTrace {
  const before = structuredClone(net);
  const { z, a, p } = forward(before, sample.x);
  const L = hiddenLayers(before);
  const act = activations[before.activation];
  const delta: number[][] = new Array(L + 1);
  const back: number[][] = new Array(L);
  delta[L] = p.map((v, k) => v - Number(k === sample.y));
  for (let l = L - 1; l >= 0; l--) {
    back[l] = a[l + 1].map((_, j) => before.W[l + 1].reduce((s, row, k) => s + row[j] * delta[l + 1][k], 0));
    delta[l] = back[l].map((v, j) => v * act.df(z[l][j], a[l + 1][j]));
  }
  const gradW = delta.map((d, l) => d.map((dj) => a[l].map((ai) => dj * ai)));
  const gradB = delta.map((d) => [...d]);
  const after: Network = {
    activation: before.activation,
    W: before.W.map((Wl, l) => Wl.map((row, j) => row.map((w, i) => w - rate * gradW[l][j][i]))),
    b: before.b.map((bl, l) => bl.map((v, j) => v - rate * gradB[l][j])),
  };
  const lossAfter = loss(forward(after, sample.x).p, sample.y);
  return { sample, rate, before, z, a, p, loss: loss(p, sample.y), delta, back, gradW, gradB, gradNorms: gradW.map(frobenius), after, lossAfter };
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

/* ── Phases of one update, depending on depth ─────────────────────────── */
export type PhaseKind = "input" | "sum" | "act" | "output-sum" | "softmax" | "output-grad" | "backprop" | "update";
export interface Phase { kind: PhaseKind; layer?: number; title: string; direction: "forward" | "backward" | "update" }

export function makePhases(hidden: number): Phase[] {
  const many = hidden > 1;
  const name = (l: number) => (many ? ` · layer ${l + 1}` : "");
  return [
    { kind: "input", title: "Input", direction: "forward" },
    ...Array.from({ length: hidden }, (_, l): Phase[] => [
      { kind: "sum", layer: l, title: `Weighted sum${name(l)}`, direction: "forward" },
      { kind: "act", layer: l, title: `Activation${name(l)}`, direction: "forward" },
    ]).flat(),
    { kind: "output-sum", title: "Output logits", direction: "forward" },
    { kind: "softmax", title: "Softmax & loss", direction: "forward" },
    { kind: "output-grad", title: "Output error", direction: "backward" },
    ...Array.from({ length: hidden }, (_, i): Phase => { const l = hidden - 1 - i; return { kind: "backprop", layer: l, title: `Backpropagate${name(l)}`, direction: "backward" }; }),
    { kind: "update", title: "Update weights", direction: "update" },
  ];
}

/** Key into the phase text table (activation-specific for "act" and "backprop"). */
export const phaseTextKey = (phase: Phase, activation: Activation) =>
  phase.kind === "act" || phase.kind === "backprop" ? `${phase.kind}-${activation}` : phase.kind;

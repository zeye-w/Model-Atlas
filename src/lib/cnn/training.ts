import { conv1d, convBackward, crossEntropy, dense, denseBackward, mean, pool, relu, reluBackward, softmax, zeros } from "./ops";

export const classes = ["Low-frequency waveform", "High-frequency waveform", "Pulse waveform"];
export type Sample = { id: string; label: number; signal: number[] };

export function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let x = Math.imul(state ^ (state >>> 15), state | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle<T>(source: T[], rng: () => number) {
  const copy = [...source];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function dataset(seed: number) {
  const rng = random(seed);
  const groups = classes.map((_, label) => Array.from({ length: 100 }, (_, index) => {
    const phase = rng() * 2 * Math.PI;
    const frequency = (label === 0 ? 1 : 9) + rng() * 2;
    const amplitude = .8 + .4 * rng();
    const center = 13 + 38 * rng();
    const width = 3 + rng() * 4;
    const signal = Array.from({ length: 64 }, (_, t) => {
      const value = label === 2 ? Math.exp(-.5 * ((t - center) / width) ** 2) : Math.sin(2 * Math.PI * frequency * t / 64 + phase);
      return amplitude * value + .12 * (rng() - .5);
    });
    return { id: label + "-" + index, label, signal };
  }));
  return {
    train: shuffle(groups.flatMap(group => group.slice(0, 70)), rng),
    validation: shuffle(groups.flatMap(group => group.slice(70, 85)), rng),
    test: shuffle(groups.flatMap(group => group.slice(85)), rng),
  };
}

export function makeModel(seed: number) {
  const rng = random(seed ^ 0xa15c);
  const weight = (fanIn: number) => (2 * rng() - 1) * Math.sqrt(6 / fanIn);
  const conv = (input: number, output: number, kernel: number) => ({
    weights: Array.from({ length: output }, () => Array.from({ length: input }, () =>
      Array.from({ length: kernel }, () => weight(input * kernel)))),
    bias: Array<number>(output).fill(0),
  });
  return {
    first: conv(1, 4, 5),
    second: conv(4, 8, 3),
    head: { weights: Array.from({ length: 3 }, () => Array.from({ length: 8 }, () => weight(8))), bias: [0, 0, 0] },
  };
}

export type Model = ReturnType<typeof makeModel>;

export function forward(model: Model, signal: number[]) {
  const input = [signal];
  const first = conv1d(input, model.first);
  const pooled = pool(relu(first));
  const second = conv1d(pooled.values, model.second);
  const features = relu(second).map(mean);
  const logits = dense(features, model.head);
  return { input, first, pooled, second, features, logits, probabilities: softmax(logits) };
}

export function backward(model: Model, state: ReturnType<typeof forward>, label: number) {
  const head = denseBackward(state.features, model.head, state.probabilities.map((p, i) => p - Number(i === label)));
  const secondDelta = state.second.map((row, c) => row.map(() => head.dx[c] / row.length));
  const second = convBackward(state.pooled.values, model.second, reluBackward(state.second, secondDelta));
  const firstDelta = zeros(state.first);
  state.pooled.indices.forEach((row, c) => row.forEach((i, t) => { firstDelta[c][i] += second.dx[c][t]; }));
  const first = convBackward(state.input, model.first, reluBackward(state.first, firstDelta));
  return { first, second, head };
}

export function trainBatch(model: Model, batch: Sample[], rate: number) {
  const before = structuredClone(model);
  const gradients = {
    first: { weights: model.first.weights.map(zeros), bias: model.first.bias.map(() => 0) },
    second: { weights: model.second.weights.map(zeros), bias: model.second.bias.map(() => 0) },
    head: { weights: zeros(model.head.weights), bias: model.head.bias.map(() => 0) },
  };
  let loss = 0;
  const outputs = batch.map(sample => {
    const state = forward(model, sample.signal);
    loss += crossEntropy(state.logits, sample.label) / batch.length;
    const gradient = backward(model, state, sample.label);
    for (const layer of ["first", "second"] as const) {
      gradient[layer].weights.forEach((kernel, f) => kernel.forEach((row, c) => row.forEach((d, k) => {
        gradients[layer].weights[f][c][k] += d / batch.length;
      })));
    }
    gradient.head.weights.forEach((row, j) => row.forEach((d, i) => { gradients.head.weights[j][i] += d / batch.length; }));
    for (const layer of ["first", "second", "head"] as const) {
      gradient[layer].bias.forEach((d, j) => { gradients[layer].bias[j] += d / batch.length; });
    }
    return state.probabilities;
  });
  for (const layer of ["first", "second"] as const) {
    model[layer].weights.forEach((kernel, f) => kernel.forEach((row, c) => row.forEach((_, k) => {
      row[k] -= rate * gradients[layer].weights[f][c][k];
    })));
  }
  model.head.weights.forEach((row, j) => row.forEach((_, i) => { row[i] -= rate * gradients.head.weights[j][i]; }));
  for (const layer of ["first", "second", "head"] as const) {
    model[layer].bias.forEach((_, j) => { model[layer].bias[j] -= rate * gradients[layer].bias[j]; });
  }
  return { before, gradients, after: structuredClone(model), loss, rate, size: batch.length, firstProbabilities: outputs[0], firstLabel: batch[0].label };
}

export function epochOrder(samples: Sample[], seed: number, epoch: number) {
  return shuffle(samples, random(seed ^ Math.imul(epoch, 0x9e3779b9)));
}

export function classificationMetrics(labels: number[], predictions: number[]) {
  const matrix = classes.map(() => classes.map(() => 0));
  labels.forEach((label, i) => { matrix[label][predictions[i]]++; });
  const scores = matrix.map((row, c) => {
    const tp = row[c];
    const precision = tp / (matrix.reduce((sum, actual) => sum + actual[c], 0) || 1);
    const recall = tp / (row.reduce((sum, count) => sum + count, 0) || 1);
    return { precision, recall, f1: 2 * precision * recall / ((precision + recall) || 1) };
  });
  return { matrix, scores, accuracy: matrix.reduce((sum, row, c) => sum + row[c], 0) / labels.length };
}

export function evaluate(model: Model, samples: Sample[]) {
  const states = samples.map(sample => forward(model, sample.signal));
  const labels = samples.map(sample => sample.label);
  const predictions = states.map(({ logits }) => logits.indexOf(Math.max(...logits)));
  const loss = mean(states.map(({ logits }, i) => crossEntropy(logits, labels[i])));
  return { loss, ...classificationMetrics(labels, predictions) };
}

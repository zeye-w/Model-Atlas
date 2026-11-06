export type Tensor = number[][];
export type Conv = { weights: Tensor[]; bias: number[] };
export type Geometry = { stride: number; padding: number; dilation: number };
export const valid: Geometry = { stride: 1, padding: 0, dilation: 1 };

export function outputLength(length: number, kernel: number, g = valid) {
  return Math.max(0, Math.floor((length + 2 * g.padding - (kernel - 1) * g.dilation - 1) / g.stride) + 1);
}

export function convolutionWindow(input: Tensor, kernel: Tensor, step: number, g = valid) {
  return kernel.flatMap((channel, c) => channel.map((weight, k) => {
    const index = step * g.stride - g.padding + k * g.dilation;
    const value = input[c][index] ?? 0;
    return { channel: c, index, value, weight, product: value * weight };
  }));
}

export function conv1d(input: Tensor, layer: Conv, g = valid): Tensor {
  const length = outputLength(input[0].length, layer.weights[0][0].length, g);
  return layer.weights.map((kernel, f) => Array.from({ length }, (_, t) => {
    let sum = layer.bias[f];
    for (let c = 0; c < kernel.length; c++) {
      for (let k = 0; k < kernel[c].length; k++) {
        const i = t * g.stride - g.padding + k * g.dilation;
        sum += (input[c][i] ?? 0) * kernel[c][k];
      }
    }
    return sum;
  }));
}

export function zeros(x: Tensor): Tensor {
  return x.map(row => row.map(() => 0));
}

export function convBackward(input: Tensor, layer: Conv, delta: Tensor, g = valid) {
  const dx = zeros(input);
  const weights = layer.weights.map(zeros);
  const bias = layer.bias.map(() => 0);
  delta.forEach((row, f) => row.forEach((d, t) => {
    bias[f] += d;
    layer.weights[f].forEach((kernel, c) => kernel.forEach((w, k) => {
      const i = t * g.stride - g.padding + k * g.dilation;
      if (i >= 0 && i < input[c].length) {
        weights[f][c][k] += d * input[c][i];
        dx[c][i] += d * w;
      }
    }));
  }));
  return { dx, weights, bias };
}

export const relu = (x: Tensor) => x.map(row => row.map(v => Math.max(0, v)));
export const reluBackward = (x: Tensor, d: Tensor) => x.map((row, c) => row.map((v, i) => v > 0 ? d[c][i] : 0));
export const mean = (row: number[]) => row.reduce((a, b) => a + b, 0) / row.length;

export function pool(input: Tensor, kernel = 2, stride = 2, average = false) {
  const length = outputLength(input[0].length, kernel, { ...valid, stride });
  const indices = input.map(row => Array.from({ length }, (_, t) => {
    const start = t * stride;
    let best = start;
    for (let k = 1; k < kernel; k++) if (row[start + k] > row[best]) best = start + k;
    return best;
  }));
  return {
    indices,
    values: input.map((row, c) => indices[c].map((i, t) => average ? mean(row.slice(t * stride, t * stride + kernel)) : row[i])),
  };
}

export type Dense = { weights: Tensor; bias: number[] };
export const dense = (x: number[], layer: Dense) =>
  layer.weights.map((row, j) => row.reduce((sum, w, i) => sum + w * x[i], layer.bias[j]));

export function denseBackward(x: number[], layer: Dense, delta: number[]) {
  return {
    dx: x.map((_, i) => delta.reduce((sum, d, j) => sum + d * layer.weights[j][i], 0)),
    weights: delta.map(d => x.map(v => d * v)),
    bias: [...delta],
  };
}

export function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exp = logits.map(v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}

export function crossEntropy(logits: number[], label: number) {
  const max = Math.max(...logits);
  return (max - logits[label]) + Math.log(logits.reduce((s, v) => s + Math.exp(v - max), 0));
}

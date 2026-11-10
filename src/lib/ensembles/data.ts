/** Shared one-dimensional regression task for the ensemble laboratory. */

export interface Point { x: number; y: number }
export interface RegressionData { train: Point[]; test: Point[]; grid: number[] }

export function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box–Muller. */
export function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Ground truth: a smooth wave with a step at x = 0.62. The step favours trees,
 * the smooth part favours radial basis functions, so the methods differ visibly.
 */
export function truth(x: number) {
  return .9 * Math.sin(2 * Math.PI * 1.15 * x) + (x > .62 ? .55 : 0) + .25 * x;
}

export const GRID_SIZE = 241;

export function makeRegressionData(seed: number, noise = .2, count = 120, testShare = 1 / 3): RegressionData {
  const rng = random(seed);
  const points = Array.from({ length: count }, () => {
    const x = rng();
    return { x, y: truth(x) + noise * gaussian(rng) };
  });
  for (let i = points.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [points[i], points[j]] = [points[j], points[i]]; }
  const nTest = Math.round(count * testShare);
  const byX = (a: Point, b: Point) => a.x - b.x;
  return {
    test: points.slice(0, nTest).sort(byX),
    train: points.slice(nTest).sort(byX),
    grid: Array.from({ length: GRID_SIZE }, (_, i) => i / (GRID_SIZE - 1)),
  };
}

export function rmse(predict: (x: number) => number, points: Point[]) {
  return Math.sqrt(points.reduce((s, p) => s + (predict(p.x) - p.y) ** 2, 0) / points.length);
}

export const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

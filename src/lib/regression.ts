export interface Point { x: number; y: number }
export interface Parameters { w: number; b: number }
export interface TrainingState extends Parameters { step: number; loss: number }

function seededRandom(seed: number) {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeDataset(seed: number, noise: number, count = 42): Point[] {
  const random = seededRandom(seed);
  return Array.from({ length: count }, (_, i) => {
    const x = -1 + (2 * i) / Math.max(1, count - 1);
    const n = (random() + random() + random() + random() - 2) * noise;
    return { x, y: 1.7 * x + 1.15 + n };
  });
}

export function mse(points: Point[], params: Parameters): number {
  if (!points.length) throw new Error("A non-empty dataset is required");
  return points.reduce((sum, p) => sum + (params.w * p.x + params.b - p.y) ** 2, 0) / points.length;
}

export function gradients(points: Point[], params: Parameters): Parameters {
  if (!points.length) throw new Error("A non-empty dataset is required");
  const sums = points.reduce((acc, p) => {
    const residual = params.w * p.x + params.b - p.y;
    return { w: acc.w + residual * p.x, b: acc.b + residual };
  }, { w: 0, b: 0 });
  return { w: 2 * sums.w / points.length, b: 2 * sums.b / points.length };
}

export function initialState(points: Point[]): TrainingState {
  return { w: 0, b: 0, step: 0, loss: mse(points, { w: 0, b: 0 }) };
}

export function gradientStep(points: Point[], current: TrainingState, learningRate: number): TrainingState {
  const grad = gradients(points, current);
  const next = { w: current.w - learningRate * grad.w, b: current.b - learningRate * grad.b };
  return { ...next, step: current.step + 1, loss: mse(points, next) };
}

/** Shared two-dimensional classification tasks for the decision-tree and KNN laboratories. */

export interface Point { x: [number, number]; y: number }
export interface ClassData { train: Point[]; test: Point[]; classes: number; kind: DatasetKind }
export type DatasetKind = "blobs" | "moons" | "xor";

export const datasets: Record<DatasetKind, { name: string; classes: number; description: string }> = {
  blobs: { name: "Three blobs", classes: 3, description: "Three Gaussian clusters that partly overlap." },
  moons: { name: "Two moons", classes: 2, description: "Two interleaved half-circles: curved, not linearly separable." },
  xor: { name: "XOR", classes: 2, description: "Opposite quadrants share a class: no single split separates them." },
};

export function random(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng: () => number) {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Seeded points in roughly [−1.5, 1.5]². `noise` scales the jitter; one third
 * of the points are held out for testing (balanced by class).
 */
export function makeClassData(kind: DatasetKind, seed: number, noise = .25, perClass = 50): ClassData {
  const rng = random(seed ^ (kind === "blobs" ? 0xb10b : kind === "moons" ? 0x300 : 0x0a0));
  const k = datasets[kind].classes;
  const groups: Point[][] = Array.from({ length: k }, () => []);
  for (let c = 0; c < k; c++) for (let i = 0; i < perClass; i++) {
    let x: [number, number];
    if (kind === "blobs") {
      const centres = [[-.75, -.45], [.75, -.45], [0, .8]];
      x = [centres[c][0] + 1.1 * noise * gaussian(rng), centres[c][1] + 1.1 * noise * gaussian(rng)];
    } else if (kind === "moons") {
      const t = Math.PI * rng();
      x = c === 0 ? [Math.cos(t) - .5, Math.sin(t) - .25] : [1 - Math.cos(t) - .5, .5 - Math.sin(t) - .25];
      x = [x[0] + noise * .6 * gaussian(rng), x[1] + noise * .6 * gaussian(rng)];
    } else {
      const sx = rng() < .5 ? -1 : 1;
      const sy = c === 0 ? sx : -sx;
      x = [sx * (.15 + .95 * rng()) + noise * .5 * gaussian(rng), sy * (.15 + .95 * rng()) + noise * .5 * gaussian(rng)];
    }
    groups[c].push({ x, y: c });
  }
  const nTest = Math.round(perClass / 3);
  return {
    kind, classes: k,
    train: groups.flatMap((g) => g.slice(nTest)),
    test: groups.flatMap((g) => g.slice(0, nTest)),
  };
}

export function accuracy(predict: (x: [number, number]) => number, points: Point[]) {
  return points.reduce((s, p) => s + Number(predict(p.x) === p.y), 0) / points.length;
}

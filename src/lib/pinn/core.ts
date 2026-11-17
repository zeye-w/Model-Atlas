/**
 * Physics-informed neural network for the damped harmonic oscillator
 *
 *   u″ + μ u′ + k u = 0,   u(0) = 1,  u′(0) = 0,   t ∈ [0, 1]
 *
 * (mass 1, μ = 2δ, k = ω₀²; defaults δ = 2, ω₀ = 20 as in Moseley's widely
 * used PINN example).
 *
 * The network is a tanh MLP t ↦ u(t). Alongside every activation we propagate
 * its first and second derivatives with respect to t (exact forward-mode
 * differentiation), giving u, u′ and u″. Parameter gradients of any loss built
 * from u, u′ and u″ are then obtained by a hand-written reverse pass through
 * those three quantities. Both are checked against finite differences in tests.
 */

export interface Net { W: Float64Array[]; b: Float64Array[]; sizes: number[] }
export interface Oscillator { delta: number; omega0: number }
export const DEFAULT_OSC: Oscillator = { delta: 2, omega0: 20 };

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

/** Exact under-damped solution with u(0) = 1, u′(0) = 0. */
export function exact(t: number, o: Oscillator = DEFAULT_OSC) {
  const { delta: d, omega0: w0 } = o;
  const w = Math.sqrt(w0 * w0 - d * d);
  const phi = Math.atan(-d / w);
  const A = 1 / (2 * Math.cos(phi));
  return Math.exp(-d * t) * 2 * A * Math.cos(phi + w * t);
}
export const coefficients = (o: Oscillator = DEFAULT_OSC) => ({ mu: 2 * o.delta, k: o.omega0 * o.omega0 });

/** Glorot-uniform weights, zero biases; W[l] is row-major (out × in). */
export function makeNet(sizes: number[], seed: number): Net {
  const rng = random(seed ^ 0x9171);
  const W: Float64Array[] = [], b: Float64Array[] = [];
  for (let l = 1; l < sizes.length; l++) {
    const fi = sizes[l - 1], fo = sizes[l], lim = Math.sqrt(6 / (fi + fo));
    W.push(Float64Array.from({ length: fo * fi }, () => (2 * rng() - 1) * lim));
    b.push(new Float64Array(fo));
  }
  return { W, b, sizes };
}

export const paramCount = (net: Net) => net.W.reduce((s, w) => s + w.length, 0) + net.b.reduce((s, b) => s + b.length, 0);

/**
 * Per-network scratch buffers. A forward pass writes into them and the reverse
 * pass for that same point reads them, so no arrays are allocated per point.
 */
interface Workspace {
  a: Float64Array[]; a1: Float64Array[]; a2: Float64Array[];   // activations and their t-derivatives, per layer (index 0 = input)
  z1: Float64Array[]; z2: Float64Array[];                       // pre-activation t-derivatives, per weight layer
  ga: Float64Array[]; ga1: Float64Array[]; ga2: Float64Array[]; // reverse-pass adjoints, per layer
}
const workspaces = new WeakMap<Net, Workspace>();
function workspace(net: Net): Workspace {
  let w = workspaces.get(net);
  if (!w) {
    const per = () => net.sizes.map((n) => new Float64Array(n));
    w = { a: per(), a1: per(), a2: per(), z1: net.sizes.slice(1).map((n) => new Float64Array(n)), z2: net.sizes.slice(1).map((n) => new Float64Array(n)), ga: per(), ga1: per(), ga2: per() };
    workspaces.set(net, w);
  }
  return w;
}

/**
 * Forward pass returning u, u′ = du/dt and u″ = d²u/dt². The intermediate
 * values stay in the network's workspace until the next forward call, which
 * is what `backward` reads — call it before the next `forward` on this net.
 */
export function forward(net: Net, t: number) {
  const L = net.W.length, ws = workspace(net);
  ws.a[0][0] = t; ws.a1[0][0] = 1; ws.a2[0][0] = 0;
  for (let l = 0; l < L; l++) {
    const fi = net.sizes[l], fo = net.sizes[l + 1], W = net.W[l], bias = net.b[l];
    const prev = ws.a[l], prev1 = ws.a1[l], prev2 = ws.a2[l];
    const out = ws.a[l + 1], out1 = ws.a1[l + 1], out2 = ws.a2[l + 1], z1 = ws.z1[l], z2 = ws.z2[l];
    const last = l === L - 1;
    for (let j = 0; j < fo; j++) {
      let z = bias[j], d1 = 0, d2 = 0;
      const row = j * fi;
      for (let i = 0; i < fi; i++) { const w = W[row + i]; z += w * prev[i]; d1 += w * prev1[i]; d2 += w * prev2[i]; }
      if (last) { out[j] = z; out1[j] = d1; out2[j] = d2; }          // linear output layer
      else {
        const h = Math.tanh(z), s = 1 - h * h;
        out[j] = h; out1[j] = s * d1; out2[j] = s * d2 - 2 * h * s * d1 * d1;
      }
      z1[j] = d1; z2[j] = d2;
    }
  }
  return { u: ws.a[L][0], u1: ws.a1[L][0], u2: ws.a2[L][0] };
}

export const predict = (net: Net, t: number) => forward(net, t).u;

export interface Grad { W: Float64Array[]; b: Float64Array[] }
export const zeroGrad = (net: Net): Grad => ({ W: net.W.map((w) => new Float64Array(w.length)), b: net.b.map((b) => new Float64Array(b.length)) });
export function clearGrad(g: Grad) { g.W.forEach((w) => w.fill(0)); g.b.forEach((b) => b.fill(0)); }

/**
 * Accumulate into `g` the parameter gradient of  gu·u + gu1·u′ + gu2·u″
 * at the point of the most recent `forward` call (i.e. given dLoss/du,
 * dLoss/du′ and dLoss/du″ there).
 *
 * For a hidden unit with h = tanh z, s = 1 − h²:
 *   a = h,  a′ = s z′,  a″ = s z″ − 2 h s z′²
 *   ∂/∂z″ = ga″ s
 *   ∂/∂z′ = ga′ s − 4 ga″ h s z′
 *   ∂/∂z  = ga s − 2 ga′ h s z′ − ga″ (2 h s z″ + 2 s (1 − 3h²) z′²)
 * and for z = W a_prev + b (likewise z′ = W a′_prev, z″ = W a″_prev):
 *   ∂/∂W = ∂/∂z ⊗ a_prev + ∂/∂z′ ⊗ a′_prev + ∂/∂z″ ⊗ a″_prev,  ∂/∂b = ∂/∂z
 */
export function backward(net: Net, gu: number, gu1: number, gu2: number, g: Grad) {
  const L = net.W.length, ws = workspace(net);
  ws.ga[L][0] = gu; ws.ga1[L][0] = gu1; ws.ga2[L][0] = gu2;
  for (let l = L - 1; l >= 0; l--) {
    const fi = net.sizes[l], fo = net.sizes[l + 1], W = net.W[l];
    // Turn adjoints of the layer's outputs into adjoints of its pre-activations, in place.
    const gz = ws.ga[l + 1], gz1 = ws.ga1[l + 1], gz2 = ws.ga2[l + 1];
    if (l !== L - 1) {
      const h = ws.a[l + 1], z1 = ws.z1[l], z2 = ws.z2[l];
      for (let j = 0; j < fo; j++) {
        const hj = h[j], s = 1 - hj * hj, d1 = z1[j], d2 = z2[j];
        const a0 = gz[j], a1 = gz1[j], a2 = gz2[j];
        gz2[j] = a2 * s;
        gz1[j] = a1 * s - 4 * a2 * hj * s * d1;
        gz[j] = a0 * s - 2 * a1 * hj * s * d1 - a2 * (2 * hj * s * d2 + 2 * s * (1 - 3 * hj * hj) * d1 * d1);
      }
    }
    const prev = ws.a[l], prev1 = ws.a1[l], prev2 = ws.a2[l];
    const na = ws.ga[l], na1 = ws.ga1[l], na2 = ws.ga2[l];
    na.fill(0); na1.fill(0); na2.fill(0);
    const gW = g.W[l], gb = g.b[l];
    for (let j = 0; j < fo; j++) {
      const row = j * fi, c0 = gz[j], c1 = gz1[j], c2 = gz2[j];
      gb[j] += c0;
      for (let i = 0; i < fi; i++) {
        gW[row + i] += c0 * prev[i] + c1 * prev1[i] + c2 * prev2[i];
        const w = W[row + i];
        na[i] += w * c0; na1[i] += w * c1; na2[i] += w * c2;
      }
    }
  }
}

/* ── The training problem ─────────────────────────────────────────────── */
export interface Problem {
  osc: Oscillator;
  observations: { t: number; u: number }[];
  collocation: number[];
  lambdaPhys: number;
  lambdaIC: number;
}

export interface ProblemSettings { seed: number; observations: number; window: number; noise: number; collocation: number; lambdaPhys: number }
export const DEFAULT_SETTINGS: ProblemSettings = { seed: 1, observations: 10, window: .36, noise: .02, collocation: 40, lambdaPhys: 1e-4 };

export function makeProblem(s: ProblemSettings, osc: Oscillator = DEFAULT_OSC): Problem {
  const rng = random(s.seed ^ 0xda7a);
  const observations = Array.from({ length: s.observations }, (_, i) => {
    const t = s.observations === 1 ? 0 : (i / (s.observations - 1)) * s.window;
    return { t, u: exact(t, osc) + s.noise * gaussian(rng) };
  });
  // Evenly spaced and including both ends, so the residual is held down at t = 0 as well.
  const collocation = Array.from({ length: s.collocation }, (_, i) => i / (s.collocation - 1));
  return { osc, observations, collocation, lambdaPhys: s.lambdaPhys, lambdaIC: 1e-1 };
}

export type Mode = "data" | "pinn";
export interface LossParts { data: number; physics: number; ic: number; total: number }

/**
 * Loss and gradient.
 *  data:    mean over observations of (u − y)²
 *  physics: λ_phys · mean over collocation points of (u″ + μ u′ + k u)²
 *  ic:      λ_ic · [(u(0) − 1)² + u′(0)²]
 * "data" mode uses only the data term; "pinn" uses all three. When `learnMu`
 * is set, μ is a trainable parameter and its gradient is returned too.
 */
export function lossAndGrad(net: Net, p: Problem, mode: Mode, mu: number, learnMu: boolean, into?: Grad) {
  const g = into ?? zeroGrad(net);
  if (into) clearGrad(into);
  const { k } = coefficients(p.osc);
  let data = 0, physics = 0, ic = 0, gMu = 0;
  const nObs = p.observations.length;
  for (const o of p.observations) {
    const f = forward(net, o.t);
    const e = f.u - o.u;
    data += e * e / nObs;
    backward(net, 2 * e / nObs, 0, 0, g);
  }
  if (mode === "pinn") {
    const n = p.collocation.length, lp = p.lambdaPhys;
    for (const t of p.collocation) {
      const f = forward(net, t);
      const r = f.u2 + mu * f.u1 + k * f.u;
      physics += lp * r * r / n;
      const c = 2 * lp * r / n;
      backward(net, c * k, c * mu, c, g);
      if (learnMu) gMu += c * f.u1;
    }
    const f0 = forward(net, 0);
    const e0 = f0.u - 1, e1 = f0.u1;
    ic = p.lambdaIC * (e0 * e0 + e1 * e1);
    backward(net, 2 * p.lambdaIC * e0, 2 * p.lambdaIC * e1, 0, g);
  }
  const parts: LossParts = { data, physics, ic, total: data + physics + ic };
  return { parts, grad: g, gMu };
}

/** Adam optimiser state for a network (and optionally μ). */
export class Adam {
  private m: Grad; private v: Grad; private t = 0; private mMu = 0; private vMu = 0;
  constructor(net: Net, public lr = 1e-3, private b1 = .9, private b2 = .999, private eps = 1e-8) { this.m = zeroGrad(net); this.v = zeroGrad(net); }
  step(net: Net, g: Grad, mu?: { value: number; grad: number; lr: number }) {
    this.t++;
    const c1 = 1 - this.b1 ** this.t, c2 = 1 - this.b2 ** this.t;
    const upd = (p: Float64Array, gr: Float64Array, m: Float64Array, v: Float64Array) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = this.b1 * m[i] + (1 - this.b1) * gr[i];
        v[i] = this.b2 * v[i] + (1 - this.b2) * gr[i] * gr[i];
        p[i] -= this.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + this.eps);
      }
    };
    net.W.forEach((w, l) => upd(w, g.W[l], this.m.W[l], this.v.W[l]));
    net.b.forEach((b, l) => upd(b, g.b[l], this.m.b[l], this.v.b[l]));
    if (mu) {
      this.mMu = this.b1 * this.mMu + (1 - this.b1) * mu.grad;
      this.vMu = this.b2 * this.vMu + (1 - this.b2) * mu.grad * mu.grad;
      mu.value -= mu.lr * (this.mMu / c1) / (Math.sqrt(this.vMu / c2) + this.eps);
    }
    return mu?.value;
  }
}

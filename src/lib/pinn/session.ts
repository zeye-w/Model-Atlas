import { Adam, coefficients, zeroGrad, type Grad, DEFAULT_OSC, exact, forward, lossAndGrad, makeNet, makeProblem, predict, type LossParts, type Net, type Problem, type ProblemSettings } from "./core";

export type Scenario = "extrapolate" | "inverse";

export interface PinnConfig extends ProblemSettings {
  scenario: Scenario;
  hidden: number;
  layers: number;
  lr: number;
}

export const DEFAULT_CONFIG: PinnConfig = {
  scenario: "extrapolate", seed: 1, observations: 10, window: .36, noise: .02, collocation: 32, lambdaPhys: 1e-4,
  hidden: 32, layers: 3, lr: 2e-3,
};

/** Inverse scenario: observations cover the whole interval, μ is unknown. */
export const INVERSE_DEFAULTS: Partial<PinnConfig> = { window: 1, observations: 30, noise: .02 };
export const MU_INITIAL = 0;
export const MU_LR = 2e-2;
/** Steps during which the network fits data and physics before μ starts to move. */
export const MU_WARMUP = 1000;
/** Default run length; both scenarios converge on every tested seed by then. */
export const DEFAULT_STEPS = 15000;

export interface HistoryPoint { step: number; dataOnly: number; pinn: LossParts; mu: number; rmseData: number; rmsePinn: number }

export interface PinnSnapshot {
  step: number;
  grid: number[];
  exact: number[];
  dataOnly: number[];
  pinn: number[];
  residual: number[];
  history: HistoryPoint[];
  mu: number;
  trueMu: number;
  rmseData: number;
  rmsePinn: number;
  observations: { t: number; u: number }[];
  collocation: number[];
  scenario: Scenario;
}

const GRID = Array.from({ length: 121 }, (_, i) => i / 120);
const EXACT = GRID.map((t) => exact(t));
/** History points are means over this many steps (Adam makes single-step losses jumpy). */
export const HISTORY_EVERY = 25;

export function rmseVsExact(net: Net) {
  let s = 0;
  for (let i = 0; i < GRID.length; i++) s += (predict(net, GRID[i]) - EXACT[i]) ** 2;
  return Math.sqrt(s / GRID.length);
}

/**
 * Trains two networks from the same initial weights on the same observations:
 * one with the data loss only, one with data + physics residual + initial
 * condition. In the inverse scenario the PINN also learns μ.
 */
export class PinnSession {
  readonly problem: Problem;
  readonly dataNet: Net;
  readonly pinnNet: Net;
  private optData: Adam;
  private optPinn: Adam;
  private gData: Grad;
  private gPinn: Grad;
  step = 0;
  mu: number;
  readonly history: HistoryPoint[] = [];
  private last: { data: number; pinn: LossParts } = { data: NaN, pinn: { data: NaN, physics: NaN, ic: NaN, total: NaN } };
  /** Loss sums since the last history point; history stores their mean over the window. */
  private acc = { n: 0, data: 0, pd: 0, pp: 0, pi: 0 };

  constructor(readonly config: PinnConfig) {
    const settings = config.scenario === "inverse" ? { ...config, ...INVERSE_DEFAULTS } : config;
    this.problem = makeProblem(settings);
    const sizes = [1, ...Array(config.layers).fill(config.hidden), 1];
    this.dataNet = makeNet(sizes, config.seed);
    this.pinnNet = makeNet(sizes, config.seed);
    this.optData = new Adam(this.dataNet, config.lr);
    this.optPinn = new Adam(this.pinnNet, config.lr);
    this.gData = zeroGrad(this.dataNet);
    this.gPinn = zeroGrad(this.pinnNet);
    this.mu = config.scenario === "inverse" ? MU_INITIAL : coefficients(DEFAULT_OSC).mu;
    this.record();
  }

  trainStep() {
    const learnMu = this.config.scenario === "inverse";
    const d = lossAndGrad(this.dataNet, this.problem, "data", this.mu, false, this.gData);
    this.optData.step(this.dataNet, d.grad);
    const p = lossAndGrad(this.pinnNet, this.problem, "pinn", this.mu, learnMu, this.gPinn);
    const muState = learnMu && this.step >= MU_WARMUP ? { value: this.mu, grad: p.gMu, lr: MU_LR } : undefined;
    const next = this.optPinn.step(this.pinnNet, p.grad, muState);
    if (muState && next !== undefined) this.mu = next;
    this.step++;
    const a = this.acc;
    a.n++; a.data += d.parts.data; a.pd += p.parts.data; a.pp += p.parts.physics; a.pi += p.parts.ic;
    if (this.step % HISTORY_EVERY === 0) {
      this.last = { data: a.data / a.n, pinn: { data: a.pd / a.n, physics: a.pp / a.n, ic: a.pi / a.n, total: (a.pd + a.pp + a.pi) / a.n } };
      this.acc = { n: 0, data: 0, pd: 0, pp: 0, pi: 0 };
      this.record();
    }
  }

  private record() {
    if (this.step === 0) {
      const d = lossAndGrad(this.dataNet, this.problem, "data", this.mu, false);
      const p = lossAndGrad(this.pinnNet, this.problem, "pinn", this.mu, false);
      this.last = { data: d.parts.data, pinn: p.parts };
    }
    this.history.push({ step: this.step, dataOnly: this.last.data, pinn: this.last.pinn, mu: this.mu, rmseData: rmseVsExact(this.dataNet), rmsePinn: rmseVsExact(this.pinnNet) });
  }

  /** Train until `target` steps or until `budgetMs` has elapsed. Returns true when the target is reached. */
  runFor(target: number, budgetMs: number, now: () => number) {
    const start = now();
    while (this.step < target) {
      this.trainStep();
      if (now() - start >= budgetMs) break;
    }
    return this.step >= target;
  }

  snapshot(): PinnSnapshot {
    const { k } = coefficients(this.problem.osc);
    const residual = GRID.map((t) => { const f = forward(this.pinnNet, t); return f.u2 + this.mu * f.u1 + k * f.u; });
    const h = this.history[this.history.length - 1];
    return {
      step: this.step, grid: GRID, exact: EXACT,
      dataOnly: GRID.map((t) => predict(this.dataNet, t)),
      pinn: GRID.map((t) => predict(this.pinnNet, t)),
      residual, history: this.history.slice(), mu: this.mu, trueMu: coefficients(this.problem.osc).mu,
      rmseData: h.step === this.step ? h.rmseData : rmseVsExact(this.dataNet),
      rmsePinn: h.step === this.step ? h.rmsePinn : rmseVsExact(this.pinnNet),
      observations: this.problem.observations, collocation: this.problem.collocation, scenario: this.config.scenario,
    };
  }
}

import assert from "node:assert/strict";
import test from "node:test";
import { backward, coefficients, exact, forward, lossAndGrad, makeNet, makeProblem, predict, zeroGrad, DEFAULT_SETTINGS } from "../src/lib/pinn/core";
import { DEFAULT_CONFIG, PinnSession } from "../src/lib/pinn/session";
import { createPinnController, type PinnEvent } from "../src/lib/pinn/controller";

const close = (a: number, b: number, tol: number, msg = "") => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${msg} expected ${b}, got ${a}`);

test("the exact solution satisfies the ODE and the initial conditions", () => {
  const { mu, k } = coefficients();
  const h = 1e-4;
  close(exact(0), 1, 1e-12);
  close((exact(h) - exact(-h)) / (2 * h), 0, 1e-4, "u'(0)");
  for (const t of [.05, .3, .61, .9]) {
    const u = exact(t), u1 = (exact(t + h) - exact(t - h)) / (2 * h), u2 = (exact(t + h) - 2 * u + exact(t - h)) / (h * h);
    assert.ok(Math.abs(u2 + mu * u1 + k * u) < 1e-3, `residual at ${t}`);
  }
});

test("forward-mode u′ and u″ match finite differences", () => {
  const net = makeNet([1, 16, 16, 16, 1], 5), h = 1e-4;
  for (const t of [0, .2, .55, 1]) {
    const f = forward(net, t);
    close(f.u1, (predict(net, t + h) - predict(net, t - h)) / (2 * h), 1e-6, `u' at ${t}`);
    close(f.u2, (predict(net, t + h) - 2 * predict(net, t) + predict(net, t - h)) / (h * h), 1e-3, `u'' at ${t}`);
  }
});

test("backward: parameter gradients of u, u′ and u″ match finite differences", () => {
  const net = makeNet([1, 6, 5, 1], 9), t = .43, [cu, c1, c2] = [.7, -1.3, .25];
  const value = () => { const f = forward(net, t); return cu * f.u + c1 * f.u1 + c2 * f.u2; };
  const g = zeroGrad(net);
  forward(net, t); backward(net, cu, c1, c2, g);
  const eps = 1e-6;
  net.W.forEach((W, l) => W.forEach((_, i) => {
    const old = W[i]; W[i] = old + eps; const up = value(); W[i] = old - eps; const dn = value(); W[i] = old;
    close(g.W[l][i], (up - dn) / (2 * eps), 1e-5, `W${l}[${i}]`);
  }));
  net.b.forEach((b, l) => b.forEach((_, i) => {
    const old = b[i]; b[i] = old + eps; const up = value(); b[i] = old - eps; const dn = value(); b[i] = old;
    close(g.b[l][i], (up - dn) / (2 * eps), 1e-5, `b${l}[${i}]`);
  }));
});

test("the full PINN loss gradient (network and μ) matches finite differences", () => {
  const net = makeNet([1, 5, 5, 1], 2);
  const p = makeProblem({ ...DEFAULT_SETTINGS, collocation: 7, observations: 4, lambdaPhys: 1e-2 });
  const mu = 3.1;
  const { grad, gMu } = lossAndGrad(net, p, "pinn", mu, true);
  const total = (m = mu) => lossAndGrad(net, p, "pinn", m, false).parts.total;
  const eps = 1e-6;
  for (const [l, i] of [[0, 2], [1, 7], [2, 3]]) {
    const W = net.W[l], old = W[i]; W[i] = old + eps; const up = total(); W[i] = old - eps; const dn = total(); W[i] = old;
    close(grad.W[l][i], (up - dn) / (2 * eps), 1e-5, `W${l}[${i}]`);
  }
  close(gMu, (total(mu + eps) - total(mu - eps)) / (2 * eps), 1e-5, "dL/dμ");
});

test("data-only mode ignores collocation points and the initial condition", () => {
  const net = makeNet([1, 8, 8, 1], 1), p = makeProblem(DEFAULT_SETTINGS);
  const { parts } = lossAndGrad(net, p, "data", coefficients().mu, false);
  assert.equal(parts.physics, 0); assert.equal(parts.ic, 0); assert.equal(parts.total, parts.data);
});

test("problems are seeded; observations stay inside the window, collocation covers [0, 1]", () => {
  const p = makeProblem(DEFAULT_SETTINGS);
  assert.deepEqual(p, makeProblem(DEFAULT_SETTINGS));
  assert.notDeepEqual(p.observations, makeProblem({ ...DEFAULT_SETTINGS, seed: 2 }).observations);
  assert.ok(p.observations.every((o) => o.t >= 0 && o.t <= DEFAULT_SETTINGS.window));
  assert.ok(Math.min(...p.collocation) < .05 && Math.max(...p.collocation) > .95);
});

test("both networks start from identical weights", () => {
  const s = new PinnSession(DEFAULT_CONFIG);
  assert.deepEqual(s.dataNet.W, s.pinnNet.W);
  const snap = s.snapshot();
  assert.deepEqual(snap.dataOnly, snap.pinn);
});

test("extrapolation: the physics residual lets the PINN predict beyond the data, the data-only fit does not", () => {
  const s = new PinnSession(DEFAULT_CONFIG);
  while (s.step < 6000) s.trainStep();
  const snap = s.snapshot();
  assert.ok(snap.rmsePinn < .15, `PINN RMSE ${snap.rmsePinn}`);
  assert.ok(snap.rmseData > .5, `data-only RMSE ${snap.rmseData}`);
  // Both fit the observations themselves.
  assert.ok(s.history.at(-1)!.dataOnly < 1e-3);
});

test("inverse: the PINN recovers the damping coefficient μ from data", () => {
  const s = new PinnSession({ ...DEFAULT_CONFIG, scenario: "inverse" });
  assert.equal(s.mu, 0);
  while (s.step < 10000) s.trainStep();
  close(s.mu, coefficients().mu, .15, "μ");
});

test("controller: reset, chunked run to a target, pause, and stale run ids are ignored", () => {
  const events: PinnEvent[] = [];
  const queue: (() => void)[] = [];
  let clock = 0;
  const c = createPinnController((e) => events.push(e), { budgetMs: 5, now: () => (clock += 1), schedule: (cb) => { queue.push(cb); return cb; }, cancel: (h) => { const i = queue.indexOf(h as () => void); if (i >= 0) queue.splice(i, 1); } });
  c.handle({ type: "reset", runId: 1, config: { ...DEFAULT_CONFIG, hidden: 4, layers: 1 } });
  assert.equal(events.at(-1)!.type, "snapshot");
  c.handle({ type: "run", runId: 0, targetStep: 50 });
  assert.equal(queue.length, 0);
  c.handle({ type: "run", runId: 1, targetStep: 50 });
  queue.shift()!();
  const first = events.at(-1)!; assert.ok(first.type === "snapshot" && first.status === "running" && first.snapshot.step === 5);
  c.handle({ type: "pause", runId: 1 });
  assert.equal(queue.length, 0);
  c.handle({ type: "run", runId: 1, targetStep: 50 });
  while (queue.length) queue.shift()!();
  const last = events.at(-1)!; assert.ok(last.type === "snapshot" && last.status === "done" && last.snapshot.step === 50);
});

import assert from "node:assert/strict";
import test from "node:test";
import { createController, type RunnerEvent } from "../src/lib/cnn/controller";
import { TrainingSession } from "../src/lib/cnn/session";
import { dataset, epochOrder, evaluate, makeModel, trainBatch } from "../src/lib/cnn/training";

const config = { seed: 42, rate: .05, batchSize: 16 };

/** The loop the training panel ran on the main thread before the worker refactor. */
function referenceRun(epochs: number) {
  const samples = dataset(config.seed);
  let model = makeModel(config.seed);
  const history = [{ train: evaluate(model, samples.train), validation: evaluate(model, samples.validation) }];
  let trace = null as ReturnType<typeof trainBatch> | null;
  while (history.length - 1 < epochs) {
    const order = epochOrder(samples.train, config.seed, history.length);
    for (let offset = 0; offset < order.length; offset += config.batchSize) {
      const next = structuredClone(model);
      trace = trainBatch(next, order.slice(offset, offset + config.batchSize), config.rate);
      model = next;
    }
    history.push({ train: evaluate(model, samples.train), validation: evaluate(model, samples.validation) });
  }
  return { model, history, trace };
}

/** Synchronous scheduler and a fake clock, so chunking is deterministic in tests. */
function harness(budgetMs: number, msPerUpdate = 1) {
  let clock = 0;
  const queue: (() => void)[] = [];
  const events: RunnerEvent[] = [];
  const controller = createController((event) => events.push(event), {
    budgetMs,
    now: () => (clock += msPerUpdate),
    schedule: (callback) => { queue.push(callback); return callback; },
    cancel: (handle) => { const i = queue.indexOf(handle as () => void); if (i >= 0) queue.splice(i, 1); },
  });
  return {
    controller, events,
    drain(limit = Infinity) { let n = 0; while (queue.length && n++ < limit) queue.shift()!(); },
    last() { const event = events.at(-1)!; assert.equal(event.type, "snapshot"); return event as Extract<RunnerEvent, { type: "snapshot" }>; },
  };
}

test("session reproduces the original main-thread training loop exactly", () => {
  const reference = referenceRun(3);
  const session = new TrainingSession(config);
  while (session.epoch < 3) session.step();
  const snapshot = session.snapshot();
  assert.deepEqual(snapshot.model, reference.model);
  assert.deepEqual(snapshot.history, reference.history);
  assert.deepEqual(snapshot.trace, reference.trace);
  assert.equal(snapshot.offset, 0);
  assert.equal(snapshot.updates, 3 * Math.ceil(210 / config.batchSize));
});

test("chunk size does not change the result", () => {
  const results = [1, 5, 1000].map((budget) => {
    const h = harness(budget);
    h.controller.handle({ type: "reset", runId: 1, config });
    h.controller.handle({ type: "run", runId: 1, targetEpoch: 2 });
    h.drain();
    assert.equal(h.last().status, "done");
    return h.last().snapshot;
  });
  assert.deepEqual(results[0], results[1]);
  assert.deepEqual(results[1], results[2]);
});

test("pausing mid-epoch and resuming gives the same run as training straight through", () => {
  const straight = harness(1000);
  straight.controller.handle({ type: "reset", runId: 1, config });
  straight.controller.handle({ type: "run", runId: 1, targetEpoch: 2 });
  straight.drain();

  const paused = harness(3);
  paused.controller.handle({ type: "reset", runId: 1, config });
  paused.controller.handle({ type: "run", runId: 1, targetEpoch: 2 });
  paused.drain(4);
  paused.controller.handle({ type: "pause", runId: 1 });
  const mid = paused.last();
  assert.equal(mid.status, "idle");
  assert.ok(mid.snapshot.offset > 0 && mid.snapshot.history.length === 1, "paused inside the first epoch");
  paused.drain();
  assert.equal(paused.last().snapshot.updates, mid.snapshot.updates, "no updates while paused");
  paused.controller.handle({ type: "run", runId: 1, targetEpoch: 2 });
  paused.drain();

  assert.deepEqual(paused.last().snapshot, straight.last().snapshot);
});

test("commands for an old run are ignored after a reset", () => {
  const h = harness(5);
  h.controller.handle({ type: "reset", runId: 1, config });
  h.controller.handle({ type: "reset", runId: 2, config: { ...config, seed: 7 } });
  const before = h.events.length;
  h.controller.handle({ type: "run", runId: 1, targetEpoch: 3 });
  h.drain();
  assert.equal(h.events.length, before, "stale run command did nothing");
  assert.equal(h.last().runId, 2);
  assert.equal(h.last().snapshot.config.seed, 7);
});

test("running to an epoch that is already reached reports done without training", () => {
  const h = harness(5);
  h.controller.handle({ type: "reset", runId: 1, config });
  h.controller.handle({ type: "run", runId: 1, targetEpoch: 0 });
  assert.equal(h.last().status, "done");
  assert.equal(h.last().snapshot.updates, 0);
});

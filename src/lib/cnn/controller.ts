import { TrainingSession, type TrainingConfig, type TrainingSnapshot } from "./session";

/** Messages from the page to the training runner. */
export type RunnerCommand =
  | { type: "reset"; runId: number; config: TrainingConfig }
  | { type: "run"; runId: number; targetEpoch: number }
  | { type: "pause"; runId: number };

/** Messages from the training runner to the page. */
export type RunnerEvent =
  | { type: "snapshot"; runId: number; snapshot: TrainingSnapshot; status: "idle" | "running" | "done" }
  | { type: "error"; runId: number; message: string };

export interface ControllerOptions {
  /** How long to train before yielding to post a snapshot and read new commands. */
  budgetMs: number;
  now?: () => number;
  schedule?: (callback: () => void) => unknown;
  cancel?: (handle: unknown) => void;
}

/**
 * Drives a TrainingSession in time-boxed chunks. Used verbatim by the Web
 * Worker and by the main-thread fallback, so both produce identical runs.
 */
export function createController(emit: (event: RunnerEvent) => void, options: ControllerOptions) {
  const now = options.now ?? (() => performance.now());
  const schedule = options.schedule ?? ((callback: () => void) => setTimeout(callback, 0));
  const cancel = options.cancel ?? ((handle: unknown) => clearTimeout(handle as ReturnType<typeof setTimeout>));

  let session: TrainingSession | null = null;
  let runId = -1;
  let target = 0;
  let pending: unknown = null;

  function stop() {
    if (pending !== null) cancel(pending);
    pending = null;
  }

  function publish(status: "idle" | "running" | "done") {
    if (session) emit({ type: "snapshot", runId, snapshot: session.snapshot(), status });
  }

  function tick() {
    pending = null;
    if (!session) return;
    try {
      const done = session.runFor(target, options.budgetMs, now);
      publish(done ? "done" : "running");
      if (!done) pending = schedule(tick);
    } catch (error) {
      emit({ type: "error", runId, message: error instanceof Error ? error.message : String(error) });
    }
  }

  return {
    handle(command: RunnerCommand) {
      switch (command.type) {
        case "reset":
          stop();
          runId = command.runId;
          target = 0;
          session = new TrainingSession(command.config);
          publish("idle");
          return;
        case "run":
          if (!session || command.runId !== runId) return;
          stop();
          target = command.targetEpoch;
          if (session.epoch >= target) { publish("done"); return; }
          pending = schedule(tick);
          return;
        case "pause":
          if (command.runId !== runId || pending === null) return;
          stop();
          publish("idle");
          return;
      }
    },
    dispose() {
      stop();
      session = null;
    },
  };
}

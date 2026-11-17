import { PinnSession, type PinnConfig, type PinnSnapshot } from "./session";

export type PinnCommand =
  | { type: "reset"; runId: number; config: PinnConfig }
  | { type: "run"; runId: number; targetStep: number }
  | { type: "pause"; runId: number };

export type PinnEvent =
  | { type: "snapshot"; runId: number; snapshot: PinnSnapshot; status: "idle" | "running" | "done" }
  | { type: "error"; runId: number; message: string };

export interface ControllerOptions {
  budgetMs: number;
  now?: () => number;
  schedule?: (callback: () => void) => unknown;
  cancel?: (handle: unknown) => void;
}

/** Drives a PinnSession in time-boxed chunks; shared by the worker and the main-thread fallback. */
export function createPinnController(emit: (event: PinnEvent) => void, options: ControllerOptions) {
  const now = options.now ?? (() => performance.now());
  const schedule = options.schedule ?? ((callback: () => void) => setTimeout(callback, 0));
  const cancel = options.cancel ?? ((handle: unknown) => clearTimeout(handle as ReturnType<typeof setTimeout>));
  let session: PinnSession | null = null;
  let runId = -1, target = 0;
  let pending: unknown = null;

  const stop = () => { if (pending !== null) cancel(pending); pending = null; };
  const publish = (status: "idle" | "running" | "done") => { if (session) emit({ type: "snapshot", runId, snapshot: session.snapshot(), status }); };

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
    handle(command: PinnCommand) {
      switch (command.type) {
        case "reset":
          stop(); runId = command.runId; target = 0;
          session = new PinnSession(command.config);
          publish("idle");
          return;
        case "run":
          if (!session || command.runId !== runId) return;
          stop(); target = command.targetStep;
          if (session.step >= target) { publish("done"); return; }
          pending = schedule(tick);
          return;
        case "pause":
          if (command.runId !== runId || pending === null) return;
          stop(); publish("idle");
          return;
      }
    },
    dispose() { stop(); session = null; },
  };
}

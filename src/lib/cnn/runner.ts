import { createController, type RunnerCommand, type RunnerEvent } from "./controller";

export type RunnerMode = "worker" | "main-thread";

export interface TrainingRunner {
  readonly mode: RunnerMode;
  send(command: RunnerCommand): void;
  dispose(): void;
}

function mainThreadRunner(onEvent: (event: RunnerEvent) => void): TrainingRunner {
  // Shorter chunks than the worker so the page keeps painting between them.
  const controller = createController(onEvent, { budgetMs: 8 });
  return { mode: "main-thread", send: (command) => controller.handle(command), dispose: () => controller.dispose() };
}

/**
 * Prefer a module Web Worker; fall back to the main thread when workers are
 * unavailable or the worker script fails to load. On a fallback the commands
 * already sent are replayed, so the run continues deterministically.
 */
export function createTrainingRunner(onEvent: (event: RunnerEvent) => void, onModeChange?: (mode: RunnerMode) => void): TrainingRunner {
  if (typeof Worker === "undefined") return mainThreadRunner(onEvent);

  let worker: Worker;
  try {
    worker = new Worker(new URL("./training.worker.ts", import.meta.url), { type: "module", name: "cnn-training" });
  } catch {
    return mainThreadRunner(onEvent);
  }

  // Only the latest reset and the latest run/pause matter for replay.
  let lastReset: RunnerCommand | null = null;
  let lastControl: RunnerCommand | null = null;
  let fallback: TrainingRunner | null = null;
  let mode: RunnerMode = "worker";

  worker.addEventListener("message", (event: MessageEvent<RunnerEvent>) => onEvent(event.data));
  worker.addEventListener("error", (event) => {
    event.preventDefault();
    worker.terminate();
    fallback = mainThreadRunner(onEvent);
    mode = "main-thread";
    onModeChange?.(mode);
    if (lastReset) fallback.send(lastReset);
    if (lastControl) fallback.send(lastControl);
  });

  return {
    get mode() { return mode; },
    send(command) {
      if (command.type === "reset") { lastReset = command; lastControl = null; } else lastControl = command;
      if (fallback) fallback.send(command); else worker.postMessage(command);
    },
    dispose() {
      if (fallback) fallback.dispose(); else worker.terminate();
    },
  };
}

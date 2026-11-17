import { createPinnController, type PinnCommand, type PinnEvent } from "./controller";

export type RunnerMode = "worker" | "main-thread";
export interface PinnRunner { readonly mode: RunnerMode; send(command: PinnCommand): void; dispose(): void }

function mainThreadRunner(onEvent: (event: PinnEvent) => void): PinnRunner {
  const controller = createPinnController(onEvent, { budgetMs: 12 });
  return { mode: "main-thread", send: (command) => controller.handle(command), dispose: () => controller.dispose() };
}

/** Module Web Worker when available; otherwise (or if it fails to load) the main thread, replaying the last commands. */
export function createPinnRunner(onEvent: (event: PinnEvent) => void, onModeChange?: (mode: RunnerMode) => void): PinnRunner {
  if (typeof Worker === "undefined") return mainThreadRunner(onEvent);
  let worker: Worker;
  try {
    worker = new Worker(new URL("./training.worker.ts", import.meta.url), { type: "module", name: "pinn-training" });
  } catch {
    return mainThreadRunner(onEvent);
  }
  let lastReset: PinnCommand | null = null, lastControl: PinnCommand | null = null;
  let fallback: PinnRunner | null = null;
  let mode: RunnerMode = "worker";
  worker.addEventListener("message", (event: MessageEvent<PinnEvent>) => onEvent(event.data));
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
    dispose() { if (fallback) fallback.dispose(); else worker.terminate(); },
  };
}

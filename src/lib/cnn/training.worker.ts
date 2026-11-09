/// <reference lib="webworker" />
import { createController, type RunnerCommand } from "./controller";

// One frame of work per chunk keeps snapshots at ~60/s without starving
// the message queue, so Pause takes effect after at most one chunk.
const controller = createController((event) => self.postMessage(event), { budgetMs: 16 });

self.addEventListener("message", (event: MessageEvent<RunnerCommand>) => controller.handle(event.data));

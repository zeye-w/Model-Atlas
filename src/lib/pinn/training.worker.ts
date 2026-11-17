/// <reference lib="webworker" />
import { createPinnController, type PinnCommand } from "./controller";

// Snapshots carry ~600 numbers; a 50 ms chunk keeps them at ~20/s, plenty for the plots.
const controller = createPinnController((event) => self.postMessage(event), { budgetMs: 50 });

self.addEventListener("message", (event: MessageEvent<PinnCommand>) => controller.handle(event.data));

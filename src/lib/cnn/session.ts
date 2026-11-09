import { dataset, epochOrder, evaluate, makeModel, trainBatch, type Model, type Sample } from "./training";

/**
 * Framework-independent training loop for the CNN laboratory.
 *
 * The same class runs inside the Web Worker and, as a fallback, on the main
 * thread. Updates are deterministic for a given config: batches are taken in
 * order from `epochOrder(train, seed, epoch + 1)`, so pausing, resuming or
 * changing how many batches run per chunk never changes the result.
 */

export interface TrainingConfig { seed: number; rate: number; batchSize: number }
export type EpochRecord = { train: ReturnType<typeof evaluate>; validation: ReturnType<typeof evaluate> };
export type BatchTrace = ReturnType<typeof trainBatch>;

export interface TrainingSnapshot {
  config: TrainingConfig;
  model: Model;
  history: EpochRecord[];
  /** Samples already consumed in the current epoch. */
  offset: number;
  trace: BatchTrace | null;
  /** Total SGD updates applied since initialization. */
  updates: number;
}

export class TrainingSession {
  readonly config: TrainingConfig;
  private readonly samples: { train: Sample[]; validation: Sample[]; test: Sample[] };
  private model: Model;
  private history: EpochRecord[];
  private offset = 0;
  private trace: BatchTrace | null = null;
  private updates = 0;
  private order: Sample[];

  constructor(config: TrainingConfig) {
    this.config = { ...config };
    this.samples = dataset(config.seed);
    this.model = makeModel(config.seed);
    this.history = [this.measure()];
    this.order = epochOrder(this.samples.train, config.seed, 1);
  }

  /** Completed epochs. */
  get epoch() { return this.history.length - 1; }

  private measure(): EpochRecord {
    return { train: evaluate(this.model, this.samples.train), validation: evaluate(this.model, this.samples.validation) };
  }

  /** Apply one SGD update. Returns true when that update finished an epoch. */
  step(): boolean {
    const batch = this.order.slice(this.offset, this.offset + this.config.batchSize);
    this.trace = trainBatch(this.model, batch, this.config.rate);
    this.updates++;
    this.offset += batch.length;
    if (this.offset < this.order.length) return false;
    this.history.push(this.measure());
    this.offset = 0;
    this.order = epochOrder(this.samples.train, this.config.seed, this.epoch + 1);
    return true;
  }

  /**
   * Run updates until `targetEpoch` is reached or `budgetMs` has elapsed.
   * Returns true when the target is reached.
   */
  runFor(targetEpoch: number, budgetMs: number, now: () => number = () => performance.now()): boolean {
    const start = now();
    while (this.epoch < targetEpoch) {
      this.step();
      if (this.epoch < targetEpoch && now() - start >= budgetMs) return false;
    }
    return true;
  }

  snapshot(): TrainingSnapshot {
    return structuredClone({
      config: this.config, model: this.model, history: this.history,
      offset: this.offset, trace: this.trace, updates: this.updates,
    });
  }
}

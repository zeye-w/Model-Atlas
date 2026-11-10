# Architecture

## Foundation

- Next.js App Router, TypeScript, React and CSS design tokens.
- A typed model registry drives cards, details, references and relationships.
- Heavy interactive code is route-scoped to the relevant laboratory.
- Numerical code is framework-independent and covered by Node tests.
- No server-side training or user data collection in Phase 1.
- Iterative training runs in a module Web Worker (`src/lib/cnn/training.worker.ts`).
  The worker and the main-thread fallback share one controller and one
  `TrainingSession`, so a run is bit-identical wherever it executes and however
  it is paused or chunked (covered by `tests/cnn-session.test.ts`).

## Visual design

- One stylesheet, `src/app/globals.css`, built on CSS custom properties
  (colour, type, radius, shadow). Route-specific styles (the CNN lab) live
  next to the route and use the same tokens.
- Calm, warm editorial look: ivory canvas, Source Serif 4 for headings,
  Inter for interface text, one terracotta accent. Fonts are self-hosted
  through Fontsource; no third-party brand assets are used.
- Text colour tokens meet WCAG AA (≥ 4.5:1) on every surface they are used on.
  The bright accent (`--accent`) is for icons, lines and focus rings; links
  use `--accent-text` and filled buttons use `--accent-strong`.

## Execution modes

| Mode | Meaning |
| --- | --- |
| `live-training` | Browser updates model parameters from an objective. |
| `live-computation` | Browser performs a real operation or inference. |
| `trace` | A precomputed experiment is replayed. |
| `planned` | A model article is available, but an interactive laboratory has not shipped. |

Model experiences must expose their execution mode honestly.

## Scale-out path

1. Keep content and experiment definitions versioned with source.
2. Add browser inference and workers for more expensive models.
3. Later use a Python API, durable task queue, isolated CPU/GPU workers and artifact storage.

## Accessibility and performance

Semantic controls, keyboard support, reduced-motion preferences, route-level splitting and pure numerical functions. Large visualization dependencies should load only where needed.

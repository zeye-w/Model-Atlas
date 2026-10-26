# Architecture

## Foundation

- Next.js App Router, TypeScript, React and CSS design tokens.
- A typed model registry drives cards, details, references and relationships.
- Heavy interactive code is route-scoped to the relevant laboratory.
- Numerical code is framework-independent and covered by Node tests.
- No server-side training or user data collection in Phase 1.

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

# Model Atlas — Phase 1 Product Scope

## Product direction

Model Atlas combines (1) navigable model taxonomy, (2) mathematically accurate visual explanations and (3) small, reproducible browser experiments. Its audience is graduate students, researchers and engineers.

## Primary journeys

1. Discover: Explore the catalogue and filter by family, task or keyword.
2. Understand: Read architecture, equations, limitations and sources.
3. Experiment: Change a parameter, run a real algorithm, inspect outputs and download a record.
4. Connect: Follow relationships between models.

## Phase 1 deliverables

- Searchable catalogue of eight model entries and shared detail page template.
- Atlas with navigable model relationships.
- One executable laboratory: linear regression with seeded synthetic data, MSE and actual gradient descent.
- Export of experiment JSON and SVG visualization.
- Responsive light design system and numerical tests.

## Later V1 milestones

- KNN, decision tree and MLP browser experiments.
- CNN and Transformer operation explainers.
- Explicitly labeled diffusion and PINN experiment traces.
- Same-dataset compare mode and expanded reproducibility metadata.

## Integrity standards

- Label actual computation, inference, simulation and recorded trace.
- Never imply a live run when playback is precomputed.
- Version data, algorithm, parameters and outputs.
- Align data, splits, preprocessing and metrics in comparisons.

## Non-goals for Phase 1

Cloud GPU training, arbitrary code execution, accounts, uploaded datasets and a model weight hub.

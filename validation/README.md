# Validation Lab

This directory is experimental validation work. It is intentionally separate from the production runtime.

## Rules

1. `main` is not modified by validation experiments.
2. Experiments on `experimental/validation-lab` must be reversible.
3. Baselines are recorded before ablations.
4. An ablation disables one mechanism without deleting its implementation.
5. Results must distinguish implementation evidence from behavioral evidence.

## Planned causal comparisons

- baseline: TD + STDP + active inference + production decision selection
- TD ablation
- STDP ablation
- active-inference feedback ablation
- decision-selection ablation

The first phase measures component-level causal effects. End-to-end behavioral ablations will only be added after a deterministic production entry point is identified and covered by a reproducible test.

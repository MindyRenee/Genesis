# Documentation map

This directory holds the evidence documents. The architecture itself is
documented where it lives: module docstrings in `src/` and `python/`
state the design rationale inline.

## In this directory

- [`ARC-AGI-3.md`](ARC-AGI-3.md) — the ARC evaluation trust boundary:
  what the adapter exposes, what counts as evidence, and why local
  runs are diagnostics rather than official scores.
- [`EMERGENCE.md`](EMERGENCE.md) — the emergence evidence catalog:
  measured behaviors, each pointing at a file, a recorded artifact, or
  a reproducible command.

## Elsewhere in the repository

- [`../AGENTS.md`](../AGENTS.md) — the architecture reference: state
  schema and migrations, wire protocol, neurochemical model,
  sleep-wake network, stewardship, and the no-hardcoding invariant.
  Despite the name it is written for humans; it is the deepest single
  document in the repo.
- [`../DEVLOG.md`](../DEVLOG.md) — dated development log of measured
  behavior: what worked, what failed, and the evidence for both.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — required checks and the
  contribution rules.
- [`../SECURITY.md`](../SECURITY.md) — the runtime security boundary
  and how to report a vulnerability.
- [`../CHANGELOG.md`](../CHANGELOG.md) — release history.
- [`../examples/`](../examples/) — small Rust binaries for inspecting
  the state file (`cargo run --example <name>`).

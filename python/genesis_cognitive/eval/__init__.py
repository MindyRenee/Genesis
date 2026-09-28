"""Baseline + drift + chemical-assay harness for Genesis.

This package answers two questions:

1. **Is Genesis drifting?** :mod:`baseline` captures a deterministic
   resting snapshot (neurochemistry, plasticity, inference, memory
   gating, phase); :mod:`drift` compares a fresh
   capture against a stored baseline and reports per-field drift.

2. **What does each chemical do?** :mod:`assay` applies a controlled
   impulse to each of the 18 chemicals on an isolated live daemon and
   records the causal effect on arousal, valence, phase, memory
   gating, and the other chemicals.

All live-daemon work goes through :mod:`harness`, which spawns an
isolated daemon in a temporary data directory — never the production
instance — settles it to rest, and tears it down afterwards.
"""

from __future__ import annotations

from .assay import AssayResult, ChemicalEffect, run_chemical_assay
from .baseline import (
    CIRCADIAN_SENSITIVE,
    DEFAULT_BASELINES,
    Snapshot,
    capture_snapshot,
)
from .drift import DriftEntry, DriftReport, check_snapshot_vs_defaults, compare_snapshots
from .harness import find_daemon_binary, isolated_daemon, settle

__all__ = [
    "CIRCADIAN_SENSITIVE",
    "DEFAULT_BASELINES",
    "AssayResult",
    "ChemicalEffect",
    "DriftEntry",
    "DriftReport",
    "Snapshot",
    "capture_snapshot",
    "check_snapshot_vs_defaults",
    "compare_snapshots",
    "find_daemon_binary",
    "isolated_daemon",
    "run_chemical_assay",
    "settle",
]

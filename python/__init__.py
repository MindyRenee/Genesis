"""Genesis cognitive mind — the thinking, feeling, speaking layer.

This package implements Genesis's cognitive mind: a cognitive system
that reads its subcognitive neurochemical state, perceives input,
retrieves memories, deliberates about what to say, and generates
natural language responses.

# Architecture

    ┌──────────────────────────────────────────────────────────┐
    │                 Mind  (the assembled organ)               │
    │                                                          │
    │   thalamus        — relay; every region's messages ride   │
    │                     the module telemetry field            │
    │   temporal_lobe   — Wernicke, semantic hub, hippocampal   │
    │                     memory, the cognitive map             │
    │   parietal_lobe   — attention, situation model, TPJ       │
    │                     agency/minimal self                   │
    │   occipital_lobe  — visual cortex, ventral stream         │
    │   frontal_lobe    — executive; Broca; mPFC metacognition  │
    │   limbic_system   — affect, amygdala, emotional memory    │
    │   basal_ganglia   — procedural memory, reward, volition   │
    │   cerebellum      — forward model, predictive coding      │
    │   brainstem       — arousal, heartbeat, the flip-flop's   │
    │                     monoaminergic pole                    │
    │   hypothalamus    — VLPO/orexin/SCN, homeostasis, HPA     │
    │   insula          — interoception, the proto-self's body  │
    │                     map                                   │
    │   basal_forebrain — cortical/hippocampal ACh, adenosine   │
    │                     (process S)                           │
    │                                                          │
    │   Each region is its own machinery. Mind wires the        │
    │   connectome and owns lifecycle; no region delegates its  │
    │   function to a separate engine.                          │
    └──────────────────────────────────────────────────────────┘
                              │ IPC
                              ▼
                    ┌──────────────────┐
                    │  Subcognitive     │
                    │  Daemon (Rust)    │
                    └──────────────────┘

# The entry point

Genesis is entered through one object — ``Mind``, the assembled
organ. It is not a catalog of models and functions: the regions are
reached by anatomical name off the connectome Mind wires
(``mind.brain.frontal``, ``mind.brain.temporal``, ``mind.brain.parietal``,
``mind.brain.occipital``, ``mind.brain.limbic``, ``mind.brain.thalamus``,
``mind.brain.cerebellum``, ...), and the shared subsystems hang off the
same object (``mind.memory``, ``mind.self_model``, ``mind.inner_life``).

    from mind import Mind

    with Mind("/path/to/genesis.sock") as mind:
        print(mind.respond("Hello, Genesis!"))
        print(mind.respond("How are you feeling?"))
        print(mind.respond("What are you?"))

# Brain-lobe naming convention

The package is organized into neuroanatomical region directories
(``frontal_lobe/``, ``parietal_lobe/``, ``temporal_lobe/``,
``occipital_lobe/``, ``limbic_system/``, ``basal_ganglia/``,
``cerebellum/``, ``thalamus/``, ``brainstem/``,
``hypothalamus/``, ``insula/``, ``basal_forebrain/``). Each is a
*view package* that lazily re-exports the modules it owns — the
region layer is a documented map of the architecture, not a
duplicate of it. They are how to *find* a function's anatomical
home (``from temporal_lobe import ConceptNetwork``), not a second
entry point into the running mind. The conformance suite
(``tests/test_regional_wiring.py``) pins the map: every claim must
resolve, load-bearing ownership decisions are asserted, and no
functional module may go unclaimed.

Each directory's ``__init__.py`` documents the design rationale and
the neuroscience that inspired it, including what is and isn't
implemented. Modules marked "not yet implemented" are documented
mathematical foundations for future work, not claims of existing
capability.
"""

from __future__ import annotations

import importlib

__all__ = ["Mind"]

# Packages a flat-catalog name may live in — the brain regions that
# carry this tree, in claim order.
_PACKAGES = (
    "frontal_lobe",
    "temporal_lobe",
    "parietal_lobe",
    "occipital_lobe",
    "limbic_system",
    "basal_ganglia",
    "cerebellum",
    "thalamus",
    "brainstem",
    "hypothalamus",
    "insula",
    "basal_forebrain",
)


def __getattr__(name: str) -> object:
    """Resolve package attributes.

    ``Mind`` is the entry point and resolves lazily so ``import
    python`` itself never pulls the whole mind into memory. Any other
    name is treated as a legacy flat-catalog import (``from python
    import ConceptNetwork``) and resolved through the region or
    functional package that claims it — kept so older consumers keep
    working, not as the advertised surface.
    """
    if name == "Mind":
        # Mind sits beside this tree at the repository root — the
        # assembled organ is a top-level module, not a region.
        return importlib.import_module("mind").Mind
    for pkg_name in _PACKAGES:
        try:
            pkg = importlib.import_module(f".{pkg_name}", __name__)
        except ImportError:
            continue
        if name in getattr(pkg, "__all__", ()):
            return getattr(pkg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__

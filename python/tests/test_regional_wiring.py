"""Regional wiring conformance — every function claimed by its region.

The brain-region packages are documented anatomical view layers over
the top-level functional modules (see ``genesis_conscious._views``).
Each region's ``_EXPORTS`` maps public symbols to the modules that
provide them, resolved lazily (PEP 562) so import order can never
close a cycle.

This suite pins that wiring the way ``test_data_dir_conformance``
pins data-dir resolution:

1. every claim in every region resolves — a stale path (as left
   behind by the October rename, 52 claims broken) fails here;
2. the load-bearing ownership decisions hold — working memory is
   frontal, procedural memory is basal ganglia, the concept network
   is temporal, and so on;
3. every functional module contributes at least one claimed symbol,
   or appears in the documented support/state allowlist — so a new
   module cannot silently go unclaimed.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

REGIONS = (
    "frontal_lobe",
    "temporal_lobe",
    "parietal_lobe",
    "occipital_lobe",
    "limbic_system",
    "basal_ganglia",
    "cerebellum",
    "thalamus",
    "brainstem",
)

REGION_MODULES = {
    name: importlib.import_module(f"genesis_conscious.{name}") for name in REGIONS
}


def _claims(region: str) -> dict[str, str]:
    """Return the region's lazy claim table (name -> module path)."""
    return dict(REGION_MODULES[region]._EXPORTS)


# ─── 1. Every claim resolves ──────────────────────────────────────


@pytest.mark.parametrize("region", REGIONS)
def test_every_claim_resolves(region: str) -> None:
    """No regional claim may raise on first attribute access."""
    mod = REGION_MODULES[region]
    broken = []
    for name in mod.__all__:
        try:
            getattr(mod, name)
        except Exception as e:  # noqa: BLE001
            broken.append(f"{name}: {e}")
    assert not broken, f"{region} has {len(broken)} broken claim(s): {broken}"


# ─── 2. Ownership pins ────────────────────────────────────────────
#
# Each pin: (symbol, regions that must claim it, regions that must
# not). Multi-claims are deliberate where the anatomy is shared
# (frontoparietal math, the occipitotemporal fusiform, the
# prefrontal-driven/thalamic-relayed workspace broadcast).

_OWNERSHIP_PINS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Working memory is DLPFC, not hippocampal.
    ("WorkingMemory", ("frontal_lobe",), ("temporal_lobe", "basal_ganglia")),
    # Procedural memory and TD learning are striatal.
    ("ProceduralMemory", ("basal_ganglia",), ("temporal_lobe", "frontal_lobe")),
    ("TDLearner", ("basal_ganglia",), ("frontal_lobe",)),
    ("ActionFeedbackEngine", ("basal_ganglia",), ("frontal_lobe",)),
    # The concept network is the anterior temporal semantic hub.
    ("ConceptNetwork", ("temporal_lobe",), ("frontal_lobe", "parietal_lobe")),
    ("EmbeddingStore", ("temporal_lobe",), ()),
    # Wernicke and Broca.
    ("ComprehensionEngine", ("temporal_lobe",), ("frontal_lobe",)),
    ("GenerativeEngine", ("frontal_lobe",), ("temporal_lobe",)),
    # Attention is parietal (IPS substrate).
    ("AttentionSystem", ("parietal_lobe",), ("temporal_lobe",)),
    # Emotion is limbic.
    ("EmotionalState", ("limbic_system",), ("frontal_lobe",)),
    ("FeelingReporter", ("limbic_system",), ("temporal_lobe",)),
    # Rhythms are thalamic; staging is brainstem.
    ("BrainWave", ("thalamus",), ("frontal_lobe",)),
    ("SleepCycleTracker", ("brainstem",), ("temporal_lobe",)),
    # Vision is occipital.
    ("V1Model", ("occipital_lobe",), ("temporal_lobe",)),
    # Cerebellar prediction.
    ("PredictiveCodingLayer", ("cerebellum",), ("frontal_lobe",)),
    # Frontal metacognition and the mind-wandering stream.
    ("ReflectionEngine", ("frontal_lobe",), ("limbic_system",)),
    ("InnerLife", ("frontal_lobe",), ("brainstem",)),
    # Hippocampal replay content and consolidation machinery.
    ("DreamSynthesisEngine", ("temporal_lobe",), ("brainstem",)),
    ("SleepCompressor", ("temporal_lobe",), ("brainstem",)),
    ("HippocampalReplay", ("temporal_lobe",), ("brainstem",)),
    # The synaptic substrate's timing rule lives with the semantic
    # plasticity family.
    ("STDP", ("temporal_lobe",), ("basal_ganglia",)),
    # Competence monitoring and the situation model.
    ("TaskCompetence", ("frontal_lobe",), ("temporal_lobe",)),
    ("SituationModel", ("parietal_lobe",), ("frontal_lobe",)),
    # Deliberate multi-claims — shared anatomy.
    ("MathResult", ("frontal_lobe", "parietal_lobe"), ()),
    ("GlobalWorkspace", ("frontal_lobe", "thalamus"), ()),
    ("FaceRecognizer", ("temporal_lobe", "occipital_lobe"), ()),
)


@pytest.mark.parametrize(("symbol", "must", "must_not"), _OWNERSHIP_PINS)
def test_ownership_pins(
    symbol: str, must: tuple[str, ...], must_not: tuple[str, ...]
) -> None:
    """The load-bearing regional ownership decisions hold."""
    for region in must:
        assert symbol in REGION_MODULES[region].__all__, (
            f"{symbol} must be claimed by {region}"
        )
    for region in must_not:
        assert symbol not in REGION_MODULES[region].__all__, (
            f"{symbol} must NOT be claimed by {region}"
        )


# ─── 3. Functional coverage ───────────────────────────────────────

# Modules that are not brain functions and are documented as such:
# support layers, the environment interface, the orchestrator, the
# task environments, and the two platform adapters.
_ALLOWLIST: dict[str, str] = {
    "genesis_conscious.mind": "the orchestrator — the whole brain, not a region",
    "genesis_conscious.infrastructure": "support layer (persistence, config, journal)",
    "genesis_conscious.eval": "baseline/drift/assay harness",
    "genesis_conscious.world": "the external world — environment interface",
    "genesis_conscious.neurochemical": "body chemistry (documented in brainstem)",
    "genesis_conscious.android_camera": "platform adapter",
    "genesis_conscious.android_tools": "platform adapter",
    "genesis_conscious._views": "the view machinery itself",
    "genesis_conscious.language.voice": (
        "documented in frontal_lobe's organization; no lazy claim — its "
        "Voice class would collide with temporal_lobe.speech's Voice"
    ),
}

# The honest task environments wired through TaskCompetence (frontal):
# practice grounds, not brain tissue.
_TASK_FAMILIES = (
    "classification",
    "quantities",
    "relations",
    "sequence",
    "sorter",
    "assembly",
)

_FUNCTIONAL_PACKAGES = (
    "cognition",
    "memory",
    "reasoning",
    "spatial",
    "language",
    "learning",
    "perception",
    "self",
    "sleep",
    "concepts",
    "tools",
    *_TASK_FAMILIES,
)


def _all_functional_modules() -> list[str]:
    """Every submodule of the functional packages, as dotted paths."""
    root = os.path.join(REPO, "python", "genesis_conscious")
    mods: list[str] = []
    for pkg in _FUNCTIONAL_PACKAGES:
        pkg_dir = os.path.join(root, pkg)
        for fname in sorted(os.listdir(pkg_dir)):
            if fname.endswith(".py") and fname != "__init__.py":
                mods.append(f"genesis_conscious.{pkg}.{fname[:-3]}")
    return mods


def _claimed_defining_modules() -> set[str]:
    """The defining module of every symbol claimed by any region."""
    defined: set[str] = set()
    for region in REGIONS:
        mod = REGION_MODULES[region]
        for name in mod.__all__:
            obj = getattr(mod, name)
            defining = getattr(obj, "__module__", None)
            if defining:
                defined.add(defining)
    return defined


def test_every_functional_module_is_claimed_or_allowlisted() -> None:
    """No functional module may silently go unclaimed by every region."""
    claimed = _claimed_defining_modules()
    unclaimed = []
    for module in _all_functional_modules():
        if module in claimed:
            continue
        if module in _ALLOWLIST:
            continue
        # The task families are the honest practice environments wired
        # through TaskCompetence (frontal) — documented practice
        # grounds, not brain tissue.
        if module.split(".")[1] in _TASK_FAMILIES:
            continue
        unclaimed.append(module)
    assert not unclaimed, (
        f"{len(unclaimed)} functional module(s) claimed by no region and "
        f"not allowlisted: {unclaimed}"
    )


def test_allowlist_entries_exist() -> None:
    """Allowlist entries must point at real modules (no stale entries)."""
    root = os.path.join(REPO, "python")
    for module in _ALLOWLIST:
        path = os.path.join(root, *module.split(".")) + ".py"
        pkg_path = os.path.join(root, *module.split("."), "__init__.py")
        assert os.path.exists(path) or os.path.exists(pkg_path), (
            f"allowlist entry {module} points at nothing"
        )

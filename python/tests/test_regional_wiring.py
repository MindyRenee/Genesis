"""Regional wiring conformance — every function claimed by its region.

The brain-region packages are the top-level packages of the source
tree (see ``_views``). Each region's ``_EXPORTS`` maps public symbols
to the modules that provide them, resolved lazily (PEP 562) so
import order can never close a cycle.

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
    "hypothalamus",
    "insula",
    "basal_forebrain",
)

REGION_MODULES = {name: importlib.import_module(name) for name in REGIONS}


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
    # Working memory includes frontal control and parietal storage.
    ("WorkingMemory", ("frontal_lobe", "parietal_lobe"), ("temporal_lobe", "basal_ganglia")),
    # Procedural memory and TD learning are striatal.
    ("ProceduralMemory", ("basal_ganglia",), ("temporal_lobe", "frontal_lobe")),
    ("TDLearner", ("basal_ganglia",), ("frontal_lobe",)),
    ("ActionFeedback", ("basal_ganglia",), ("frontal_lobe",)),
    # The concept network is the anterior temporal semantic hub.
    ("ConceptNetwork", ("temporal_lobe",), ("frontal_lobe", "parietal_lobe")),
    ("EmbeddingStore", ("temporal_lobe",), ()),
    # Wernicke and Broca.
    ("Comprehension", ("temporal_lobe",), ("frontal_lobe",)),
    ("GenerativeLanguage", ("frontal_lobe",), ("temporal_lobe",)),
    # Attention is parietal (IPS substrate).
    ("AttentionSystem", ("parietal_lobe",), ("temporal_lobe",)),
    # Emotion is limbic.
    ("EmotionalState", ("limbic_system",), ("frontal_lobe",)),
    # Conditioned value lives in the amygdala, not the striatum or cortex.
    ("Amygdala", ("limbic_system",), ("basal_ganglia", "frontal_lobe")),
    ("FeelingReporter", ("limbic_system",), ("temporal_lobe",)),
    # Rhythm summaries are shared; staging spans brainstem and hypothalamus.
    ("BrainWave", ("thalamus",), ("frontal_lobe",)),
    ("SleepCycleTracker", ("brainstem", "hypothalamus"), ("temporal_lobe",)),
    # Vision is occipital.
    ("V1Model", ("occipital_lobe",), ("temporal_lobe",)),
    # Cerebellar prediction.
    ("PredictiveCodingLayer", ("cerebellum",), ("frontal_lobe",)),
    # Frontal metacognition and the mind-wandering stream — the DMN
    # is a frontoparietal midline loop (mPFC + PCC/precuneus).
    ("Reflection", ("frontal_lobe",), ("limbic_system",)),
    ("InnerLife", ("frontal_lobe", "parietal_lobe"), ("brainstem",)),
    # Hippocampal replay content and consolidation machinery.
    ("DreamSynthesis", ("temporal_lobe",), ("brainstem",)),
    ("SleepCompressor", ("temporal_lobe",), ("brainstem",)),
    ("HippocampalReplay", ("temporal_lobe",), ("brainstem",)),
    # The synaptic substrate's timing rule lives with the semantic
    # plasticity family.
    ("STDP", ("temporal_lobe",), ("basal_ganglia",)),
    # Competence monitoring and the situation model.
    ("TaskCompetence", ("frontal_lobe",), ("temporal_lobe",)),
    ("SituationModel", ("parietal_lobe",), ("frontal_lobe",)),
    # Rule induction is the mid-DLPFC's hypothesis generator.
    ("DorsolateralPFC", ("frontal_lobe",), ("temporal_lobe", "basal_ganglia", "parietal_lobe")),
    # Error/conflict monitoring is the dorsal ACC's — medial frontal,
    # not limbic cortex and not striatal (it reads striatal signals).
    ("ErrorMonitor", ("frontal_lobe",), ("limbic_system", "basal_ganglia", "temporal_lobe")),
    # Deliberate multi-claims — shared anatomy.
    ("MathResult", ("frontal_lobe", "parietal_lobe"), ()),
    ("GlobalWorkspace", ("frontal_lobe", "thalamus"), ()),
    ("FaceRecognizer", ("temporal_lobe", "occipital_lobe"), ()),
    # Stress axis and neuroendocrine outputs are hypothalamic.
    ("HPAAxis", ("hypothalamus", "limbic_system"), ()),
    ("Orexin", ("hypothalamus",), ()),
    ("Melatonin", ("hypothalamus",), ()),
    # Source nuclei of the ascending neuromodulators are brainstem.
    ("Norepinephrine", ("brainstem",), ()),
    ("Serotonin", ("brainstem",), ()),
    # Aggregate ACh includes basal-forebrain and pontine projections.
    ("Acetylcholine", ("basal_forebrain", "brainstem"), ()),
    # Interoception is insular; the metronome that samples the body is
    # brainstem; both are claimed by each.
    ("InteroceptionSystem", ("insula", "limbic_system"), ()),
    ("MetronomeMixin", ("insula", "brainstem"), ()),
    # The cardiac control center is medullary — the heart muscle
    # itself is the daemon's carrier, its center is brainstem.
    ("Heart", ("brainstem",), ()),
    ("CardiacState", ("brainstem",), ()),
    # The urge accumulator is gated by the striatum and held by frontal.
    ("Urge", ("basal_ganglia", "frontal_lobe"), ()),
    # Social perception is superior temporal.
    ("Presence", ("temporal_lobe",), ()),
    # The event stream's direction classification rides with it.
    ("INBOUND_KINDS", ("temporal_lobe",), ()),
    ("OUTBOUND_KINDS", ("temporal_lobe",), ()),
    # Process S: the adenosine signal accumulates and acts in the
    # basal forebrain (A1 inhibition of wake-active neurons).
    ("Adenosine", ("basal_forebrain",), ()),
    # Orexin-timescale grogginess — hypothalamic peptide, brainstem switch.
    ("SleepInertia", ("brainstem", "hypothalamus"), ()),
    # PGO: generated pontine, projected occipital.
    ("PGOWave", ("brainstem", "occipital_lobe"), ()),
    # The K-complex downstate is cortical (frontal recruitment,
    # Mak-McClure 2015) riding the thalamocortical loop — claimed by
    # both ends of the circuit.
    ("KComplex", ("frontal_lobe", "thalamus"), ()),
    # The external-world model is the hippocampal-entorhinal map plus
    # the STS social stream — the relay is how it reaches cortex.
    ("OuterWorld", ("temporal_lobe",), ()),
    # Her page — a place in the external world, not a mind organ:
    # worlds.channel is environment-side, so no region may claim it.
    ("Channel", (), REGIONS),
    # The board is a view on that page — environment-side like the
    # channel it renders for, claimable by no region.
    ("ChannelBoard", (), REGIONS),
    # Mentalizing posteriors over presences — rTPJ + mPFC.
    ("PresenceBelief", ("parietal_lobe", "frontal_lobe"), ()),
    # The minimal self and agency comparator are TPJ work; the
    # Damasio hierarchy is limbic with a brainstem proto-self root.
    ("MinimalSelf", ("parietal_lobe", "limbic_system"), ()),
    ("AgencyDetector", ("parietal_lobe", "limbic_system"), ()),
    ("ProtoSelf", ("insula", "limbic_system", "brainstem"), ()),
    ("AutobiographicalSelf", ("limbic_system", "temporal_lobe"), ()),
    # Amygdala-modulated consolidation is limbic AND hippocampal.
    ("EmotionalMemorySystem", ("temporal_lobe", "limbic_system"), ()),
    # Interoceptive somatic-marker congruence is read in the insula.
    ("SomaticSnapshot", ("limbic_system", "insula"), ()),
    # Region-owned chemicals the table assigns but nothing claimed.
    ("BDNF", ("temporal_lobe",), ()),
    ("Endorphin", ("hypothalamus",), ()),
)


@pytest.mark.parametrize(("symbol", "must", "must_not"), _OWNERSHIP_PINS)
def test_ownership_pins(symbol: str, must: tuple[str, ...], must_not: tuple[str, ...]) -> None:
    """The load-bearing regional ownership decisions hold."""
    for region in must:
        assert symbol in REGION_MODULES[region].__all__, f"{symbol} must be claimed by {region}"
    for region in must_not:
        assert symbol not in REGION_MODULES[region].__all__, (
            f"{symbol} must NOT be claimed by {region}"
        )


# ─── 3. Functional coverage ───────────────────────────────────────

# Modules that are not brain functions and are documented as such:
# support layers, the environment interface, the orchestrator, the
# task environments, and the platform adapter.
_ALLOWLIST: dict[str, str] = {
    "mind": "the assembled mind — a top-level module, not a region",
    "genesis_cli": "the console entrypoint — a top-level module, not a region",
    "genesis_client": "the daemon client — support layer beside the regions",
    "infrastructure": "support layer (persistence, config, journal)",
    "eval": "baseline/drift/assay harness",
    "frontal_lobe.krita_studio": "Krita plugin bridge — platform adapter, not a brain function",
    "commitment": "top-level support module",
    "discord_feed": "platform adapter",
    "setup_embeddings": "one-off embedding setup script",
    "_views": "the view machinery itself",
    # Constant tables — no class/function to carry __module__, but the
    # file is brainstem's autonomic setpoints
    "brainstem.thresholds": "autonomic setpoints — constants owned by brainstem",
    "parietal_lobe.motion": "compatibility projection of occipitotemporal MT/V5",
    "parietal_lobe.epipolar": "compatibility projection of occipital MSTd",
    "tools.charter": "a prose letter to the organism — no code, nothing to claim",
    "_wake": "operator scratch probe — not a brain function",
    # Parietal math leaf modules — internal library tissue used through
    # the claimed front-ends (arithmetic/algebra/calculus/symbolic).
    "parietal_lobe.math.abstract_algebra": "math leaf — parietal library tissue",
    "parietal_lobe.math.axioms": "math leaf — parietal library tissue",
    "parietal_lobe.math.combinatorics": "math leaf — parietal library tissue",
    "parietal_lobe.math.complex_analysis": "math leaf — parietal library tissue",
    "parietal_lobe.math.discrete": "math leaf — parietal library tissue",
    "parietal_lobe.math.geometry": "math leaf — parietal library tissue",
    "parietal_lobe.math.linear_algebra": "math leaf — parietal library tissue",
    "parietal_lobe.math.number_theory": "math leaf — parietal library tissue",
    "parietal_lobe.math.proof": "math leaf — parietal library tissue",
    "parietal_lobe.math.real_analysis": "math leaf — parietal library tissue",
    "parietal_lobe.math.topology": "math leaf — parietal library tissue",
    "parietal_lobe.math.trig": "math leaf — parietal library tissue",
}

_FUNCTIONAL_PACKAGES = (
    # The regions themselves — they are the engines now, and their
    # own modules must be claimed by a region just like any other.
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


def _all_functional_modules() -> list[str]:
    """Every .py module under the region tree plus the support packages."""
    root = os.path.join(REPO, "python")
    mods: list[str] = []
    for pkg in _FUNCTIONAL_PACKAGES:
        for dirpath, _dirs, files in os.walk(os.path.join(root, pkg)):
            for fname in sorted(files):
                if fname.endswith(".py") and fname != "__init__.py":
                    rel = os.path.relpath(os.path.join(dirpath, fname), root)
                    mods.append(rel[:-3].replace(os.sep, "."))
    # The support packages and loose modules sit beside python/ at the
    # repository root; they are candidates for the allowlist, never
    # for a regional claim.
    for pkg in ("genesis_client", "infrastructure", "eval"):
        pdir = os.path.join(REPO, pkg)
        if not os.path.isdir(pdir):
            continue
        for dirpath, _dirs, files in os.walk(pdir):
            for fname in sorted(files):
                if fname.endswith(".py") and fname != "__init__.py":
                    rel = os.path.relpath(os.path.join(dirpath, fname), REPO)
                    mods.append(rel[:-3].replace(os.sep, "."))
    for fname in sorted(os.listdir(REPO)):
        if fname.endswith(".py") and fname != "__init__.py":
            mods.append(fname[:-3])
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
        if any(module == a or module.startswith(a + ".") for a in _ALLOWLIST):
            continue
        unclaimed.append(module)
    assert not unclaimed, (
        f"{len(unclaimed)} functional module(s) claimed by no region and "
        f"not allowlisted: {unclaimed}"
    )


def test_allowlist_entries_exist() -> None:
    """Allowlist entries must point at real modules (no stale entries)."""
    for module in _ALLOWLIST:
        found = any(
            os.path.exists(os.path.join(root, *module.split(".")) + ".py")
            or os.path.exists(os.path.join(root, *module.split("."), "__init__.py"))
            for root in (os.path.join(REPO, "python"), REPO)
        )
        assert found, f"allowlist entry {module} points at nothing"


def test_volition_workers_run_below_foreground_priority() -> None:
    """Urge worker threads renice themselves before acting.

    The kernel-level half of foreground arbitration: a bug scan that
    started before the user typed keeps losing scheduler/GIL handoffs
    to think() instead of timing it out at 15 s.
    """
    import sys
    from unittest.mock import MagicMock, patch

    import basal_ganglia.volition as vol

    if not sys.platform.startswith("linux"):
        pytest.skip("per-thread nice is linux-only")

    with patch.object(vol.os, "setpriority") as sp:
        vol._lower_thread_priority()
        sp.assert_called_once()
        assert sp.call_args[0][2] == vol._VOLITION_NICE

    # Wrapped actions carry the priority — the renice runs inside the
    # worker thread, not on the dispatcher.
    seen = {}

    def probe():
        seen["nice"] = vol.os.getpriority(vol.os.PRIO_PROCESS, 0)

    holder = MagicMock()
    import threading

    holder._volition_lock = threading.Lock()
    holder._volition_active = {"probe"}
    holder._volition_sem = threading.Semaphore(1)
    holder._volition_sem.acquire()
    # The wrapper drops work when the mind is suppressed or stopping —
    # this probe must still run, so the holder reads as live.
    holder._suppress_volition = False
    holder._running = True
    vol.VolitionMixin._run_volition_action(holder, "probe", probe)
    assert seen["nice"] >= vol._VOLITION_NICE - 1

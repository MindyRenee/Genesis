"""Resting-state snapshot: what "normal" looks like, as data.

A :class:`Snapshot` is a deterministic capture of a freshly settled
isolated daemon: per-chemical effective levels, arousal/valence,
plasticity gate, memory gating weights, phase, plasticity profile,
and inference summary. Serialized to JSON,
it becomes a baseline file that :mod:`drift` compares against.

``DEFAULT_BASELINES`` mirrors ``NeurochemicalId::default_baseline()``
in ``src/state/neurochemical.rs`` — the genetic set-points the
homeostatic force pulls toward. If the Rust values change, this table
(and ``test_default_baselines_match_documented_values``) must change
with them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from genesis_client import GenesisClient
from genesis_client.protocol import CHEM_NAMES, PROTOCOL_VERSION

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

# Genetic set-points — must match NeurochemicalId::default_baseline().
DEFAULT_BASELINES: dict[str, float] = {
    "dopamine": 0.35,
    "serotonin": 0.40,
    "norepinephrine": 0.30,
    "acetylcholine": 0.35,
    "gaba": 0.50,
    "glutamate": 0.60,
    "cortisol": 0.0,
    "oxytocin": 0.25,
    "endorphin": 0.25,
    "histamine": 0.30,
    "adenosine": 0.20,
    "bdnf": 0.45,
    "endocannabinoid": 0.25,
    "vasopressin": 0.20,
    "crh": 0.0,
    "orexin": 0.30,
    "epinephrine": 0.15,
    "melatonin": 0.10,
}

# Chemicals whose resting level legitimately varies with time of day
# (circadian oscillator in src/state/neurochemical.rs): dopamine ±15%,
# serotonin via melatonin conversion, histamine ±20%, melatonin driven
# directly. Drift checks use wider tolerances for these (see drift.py).
CIRCADIAN_SENSITIVE = frozenset({"dopamine", "serotonin", "histamine", "melatonin"})

CHEM_ORDER: list[str] = [CHEM_NAMES[i] for i in sorted(CHEM_NAMES)]


@dataclass
class Snapshot:
    """A settled resting-state capture. JSON-serializable via to_dict."""

    format_version: int = FORMAT_VERSION
    captured_at_unix: float = 0.0
    protocol_version: int = PROTOCOL_VERSION
    settle_ticks: int = 0
    settle_dt: float = 1.0
    chemicals: dict[str, float] = field(default_factory=dict)
    arousal: float = 0.0
    valence: float = 0.0
    global_tone: float = 0.0
    plasticity_gate: float = 0.0
    encoding_weight: float = 0.0
    consolidation_weight: float = 0.0
    retrieval_weight: float = 0.0
    phase: int = 0
    phase_name: str = "unknown"
    coupling_drift: float = 0.0
    mean_receptor_sensitivity: float = 1.0
    bdnf_effective: float = 0.0
    cortisol_effective: float = 0.0
    surprise_ema: float = 0.0
    free_energy: float = 0.0
    precision: float = 0.0
    allostatic_load: float = 0.0
    model_maturity: float = 0.0
    resting_emotion: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        """Deserialize; unknown keys are ignored for forward compatibility."""
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known}
        snap = cls()
        for k, v in filtered.items():
            setattr(snap, k, v)
        if not snap.chemicals:
            snap.chemicals = {}
        return snap

    def save(self, path: str) -> None:
        """Write the snapshot as pretty-printed JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> Snapshot:
        """Read a snapshot from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def capture_snapshot(
    client: GenesisClient,
    *,
    settle_ticks: int = 60,
    settle_dt: float = 1.0,
) -> Snapshot:
    """Settle the daemon to rest, then capture a Snapshot.

    Must be called on an isolated daemon (see harness.isolated_daemon)
    so impulses from production activity cannot pollute the capture.
    """
    from .harness import settle

    settle(client, ticks=settle_ticks, dt=settle_dt)
    state = client.get_state()
    plasticity = client.get_plasticity_profile()
    inference = client.get_inference_summary()

    resting_emotion: dict[str, str] = {}
    try:
        from ..emotion import assess_emotion

        summary = client.get_neuro_summary()
        emo = assess_emotion(summary, dict(state.chemicals))
        resting_emotion = {
            "label": emo.label,
            "cognitive_style": emo.cognitive_style,
            "cause": emo.cause,
        }
    except ImportError as e:
        # Emotion module unavailable in minimal installs — baseline simply
        # omits the resting-emotion enrichment rather than failing.
        logger.debug(f"resting-emotion enrichment skipped: {e}")

    return Snapshot(
        captured_at_unix=time.time(),
        protocol_version=PROTOCOL_VERSION,
        settle_ticks=settle_ticks,
        settle_dt=settle_dt,
        chemicals=dict(state.chemicals),
        arousal=state.arousal,
        valence=state.valence,
        global_tone=state.global_tone,
        plasticity_gate=state.plasticity_gate,
        encoding_weight=state.encoding_weight,
        consolidation_weight=state.consolidation_weight,
        retrieval_weight=state.retrieval_weight,
        phase=state.emergent_phase,
        phase_name=state.phase_name,
        coupling_drift=plasticity.coupling_drift,
        mean_receptor_sensitivity=plasticity.mean_receptor_sensitivity,
        bdnf_effective=plasticity.bdnf_effective,
        cortisol_effective=plasticity.cortisol_effective,
        surprise_ema=inference.surprise_ema,
        free_energy=inference.free_energy,
        precision=inference.precision,
        allostatic_load=inference.allostasis_load,
        model_maturity=inference.model_maturity,
        resting_emotion=resting_emotion,
    )

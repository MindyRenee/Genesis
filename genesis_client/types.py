"""Data types for the Genesis client library.

These are Python dataclasses that mirror the Rust wire types. Each one
can be deserialised from the binary response payload returned by the
daemon.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .protocol import (
    _F32,
    _I32,
    _U16,
    _U32,
    _U64,
    CHEM_NAMES,
    PHASE_NAMES,
    SENSOR_NAMES,
    SUBSYSTEM_NAMES,
    ZONE_NAMES,
)

__all__ = [
    "LEARNING_POSTURE_NEUTRAL",
    "LEARNING_POSTURE_PROTECTIVE",
    "LEARNING_POSTURE_RECEPTIVE",
    "LEARNING_POSTURE_RECOVERING",
    "LOBE_NAMES",
    "MODULE_NAMES",
    "BodyControlState",
    "BodyState",
    "CoreState",
    "Episode",
    "FoldMargins",
    "InferenceSummary",
    "IonSummary",
    "MemoryStats",
    "ModuleTelemetry",
    "NeuroSummary",
    "OpticState",
    "PhaseInfo",
    "PingResponse",
    "PlasticityProfile",
    "ProcessHealth",
    "RecentEpisode",
    "SensorPresence",
    "SimilarEpisode",
    "StateDwell",
    "SubsystemReport",
    "SubsystemTelemetry",
    "ZoneTransitions",
    "unpack_recent_episodes",
    "unpack_similar_results",
]

# ─── Learning posture ─────────────────────────────────────────
#
# Derived from PlasticityProfile by the cognitive mind. The posture
# enum is a Python-side cognitive-strategy decision (per the boundary
# principle: the substrate exposes state, the cognitive mind decides
# how to use it). It is NOT sent over the wire.

LEARNING_POSTURE_RECEPTIVE = "receptive"
LEARNING_POSTURE_PROTECTIVE = "protective"
LEARNING_POSTURE_RECOVERING = "recovering"
LEARNING_POSTURE_NEUTRAL = "neutral"

# ─── NeuroSummary (40 bytes; v1 = 32-byte prefix) ─────────────
#
# offset  field                  type
# ------  -----                  ----
#   0     arousal                f32
#   4     valence                f32
#   8     global_tone            f32
#  12     plasticity_gate        f32
#  16     encoding_weight        f32
#  20     consolidation_weight   f32
#  24     retrieval_weight       f32
#  28     phase                  u8
#  29     _pad                   [u8;3]
#  32     dopamine_tonic         f32   (v2 only — absent on 32-byte packets)
#  36     dopamine_phasic        f32   (v2 only — absent on 32-byte packets)


@dataclass(frozen=True)
class IonSummary:
    """Compact calcium/chloride/potassium/sodium membrane state."""

    intracellular_mm: tuple[float, float, float, float]
    extracellular_mm: tuple[float, float, float, float]
    reversal_potential_mv: tuple[float, float, float, float]
    conductance: tuple[float, float, float, float]
    membrane_potential_mv: float
    nak_pump_rate: float
    kcl_cotransporter_flux: float
    ncx_flux: float
    atp_availability: float
    calcium_signal: float
    chloride_efficacy: float
    excitability: float
    gradient_integrity: float
    energy_load: float
    net_membrane_current: float

    @property
    def ion_names(self) -> tuple[str, str, str, str]:
        """Names matching the array order used by all four fields."""
        return ("calcium", "chloride", "potassium", "sodium")

    def intracellular(self, ion: str) -> float:
        """Return an intracellular concentration by ion name."""
        return self.intracellular_mm[self.ion_names.index(ion)]

    def extracellular(self, ion: str) -> float:
        """Return an extracellular concentration by ion name."""
        return self.extracellular_mm[self.ion_names.index(ion)]

    def reversal_potential(self, ion: str) -> float:
        """Return a Nernst reversal potential by ion name."""
        return self.reversal_potential_mv[self.ion_names.index(ion)]

    @classmethod
    def unpack(cls, data: bytes) -> IonSummary:
        """Unpack a 108-byte IonSummary from the daemon's response."""
        if len(data) < 108:
            raise ValueError(f"IonSummary needs 108 bytes, got {len(data)}")
        values = struct.unpack_from("<27f", data, 0)
        return cls(
            intracellular_mm=values[0:4],
            extracellular_mm=values[4:8],
            reversal_potential_mv=values[8:12],
            conductance=values[12:16],
            membrane_potential_mv=values[16],
            nak_pump_rate=values[17],
            kcl_cotransporter_flux=values[18],
            ncx_flux=values[19],
            atp_availability=values[20],
            calcium_signal=values[21],
            chloride_efficacy=values[22],
            excitability=values[23],
            gradient_integrity=values[24],
            energy_load=values[25],
            net_membrane_current=values[26],
        )


@dataclass(frozen=True)
class FoldMargins:
    """The LC–VLPO switch's saddle-node margins (40 bytes).

    The body's homeostat ``h`` is the control parameter of the
    mutual-inhibition sleep switch. The two fold edges are the
    saddle-node points where each attractor ceases to exist:
    ``h_sleep_fold`` is the wake-up edge (below it the sleep branch
    dies), ``h_wake_fold`` the fall-asleep edge (above it the wake
    branch dies). ``position`` normalizes ``h`` into the hysteresis
    band — 0 is the wake-up edge, 1 the fall-asleep edge, outside
    that range is forced-wake/forced-sleep monostability. Edges
    report NaN when they lay outside the body's sweep window, and
    ``position`` is NaN whenever either edge is. ``bistable`` means
    three fixed points coexist — noise can hop the barrier; that is
    the hypnagogic flicker regime.
    """

    h: float
    h_sleep_fold: float
    h_wake_fold: float
    position: float
    bistable: bool

    @classmethod
    def unpack(cls, data: bytes) -> FoldMargins:
        """Unpack a 40-byte FoldMargins from the daemon's response."""
        if len(data) < 40:
            raise ValueError(f"FoldMargins needs 40 bytes, got {len(data)}")
        h, h_sleep_fold, h_wake_fold, position = struct.unpack_from("<4d", data, 0)
        return cls(
            h=h,
            h_sleep_fold=h_sleep_fold,
            h_wake_fold=h_wake_fold,
            position=position,
            bistable=data[32] != 0,
        )


@dataclass(frozen=True)
class NeuroSummary:
    """Compact neurochemical summary — how Genesis feels right now.

    ``dopamine_tonic``/``dopamine_phasic`` are the daemon's own
    tonic/phasic decomposition of dopamine (vigor / RPE-burst
    channels). They arrive only on the extended 40-byte packet —
    on a 32-byte (v1) response both are ``None``, and consumers
    should fall back to the effective ``chemicals["dopamine"]``.
    """

    arousal: float
    valence: float
    global_tone: float
    plasticity_gate: float
    encoding_weight: float
    consolidation_weight: float
    retrieval_weight: float
    phase: int
    dopamine_tonic: float | None = None
    dopamine_phasic: float | None = None

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current mental phase."""
        return PHASE_NAMES.get(self.phase, "unknown")

    @classmethod
    def unpack(cls, data: bytes) -> NeuroSummary:
        """Unpack a NeuroSummary from the daemon's binary response.

        v1 packets are 32 bytes; v2 appends the dopamine tonic/phasic
        pair for 40 bytes total. Either is accepted.
        """
        if len(data) < 32:
            raise ValueError(f"NeuroSummary needs 32 bytes, got {len(data)}")
        return cls(
            arousal=_F32.unpack(data[0:4])[0],
            valence=_F32.unpack(data[4:8])[0],
            global_tone=_F32.unpack(data[8:12])[0],
            plasticity_gate=_F32.unpack(data[12:16])[0],
            encoding_weight=_F32.unpack(data[16:20])[0],
            consolidation_weight=_F32.unpack(data[20:24])[0],
            retrieval_weight=_F32.unpack(data[24:28])[0],
            phase=data[28],
            dopamine_tonic=_F32.unpack(data[32:36])[0] if len(data) >= 40 else None,
            dopamine_phasic=_F32.unpack(data[36:40])[0] if len(data) >= 40 else None,
        )


# ─── MemoryStats (24 bytes) ───────────────────────────────────
#
# offset  field           type
# ------  -----           ----
#   0     stm_count       u64
#   8     ltm_count       u64
#  16     ltm_capacity    u32
#  20     _pad            u32


@dataclass(frozen=True)
class MemoryStats:
    """Memory store statistics."""

    stm_count: int
    ltm_count: int
    ltm_capacity: int

    @classmethod
    def unpack(cls, data: bytes) -> MemoryStats:
        """Unpack a 24-byte MemoryStats from the daemon's binary response."""
        if len(data) < 24:
            raise ValueError(f"MemoryStats needs 24 bytes, got {len(data)}")
        return cls(
            stm_count=_U64.unpack(data[0:8])[0],
            ltm_count=_U64.unpack(data[8:16])[0],
            ltm_capacity=_U32.unpack(data[16:20])[0],
        )


# ─── PlasticityProfile (40 bytes) ─────────────────────────────
#
# offset  field                       type
# ------  -----                       ----
#   0     plasticity_gate             f32
#   4     bdnf_effective              f32
#   8     bdnf_tonic                  f32
#  12     cortisol_effective          f32
#  16     cortisol_tonic              f32
#  20     dopamine_effective          f32
#  24     serotonin_effective         f32
#  28     coupling_drift              f32
#  32     mean_receptor_sensitivity   f32
#  36     emergent_phase              u8
#  37     _pad                        [u8;3]


@dataclass(frozen=True)
class PlasticityProfile:
    """Compact summary of the metaplastic state — how the substrate's
    "learning-to-learn" machinery is currently configured.

    The coupling matrix self-modifies under sustained emotion
    (metaplasticity). This struct exposes that drift plus the key
    chemicals and receptor state that drive learning strategy,
    without transferring the full 1296-byte matrix.

    The cognitive mind derives a ``learning_posture`` from this
    profile to adapt its acquisition/consolidation strategy.
    """

    plasticity_gate: float
    bdnf_effective: float
    bdnf_tonic: float
    cortisol_effective: float
    cortisol_tonic: float
    dopamine_effective: float
    serotonin_effective: float
    coupling_drift: float
    mean_receptor_sensitivity: float
    emergent_phase: int

    @property
    def phase_name(self) -> str:
        """Human-readable name of the emergent mental phase."""
        return PHASE_NAMES.get(self.emergent_phase, "unknown")

    @property
    def learning_posture(self) -> str:
        """Derive a learning posture from the plasticity signals.

        This is a cognitive-strategy decision (Python-side), not a
        substrate state. The thresholds are grounded in the
        neurobiology:

        - **RECEPTIVE**: high plasticity gate (BDNF high, cortisol
          low). The system is primed for new memory formation.
          Strategy: aggressive acquisition, lower consolidation
          threshold, more bridge creation.

        - **PROTECTIVE**: chronic stress (sustained high cortisol).
          BDNF is suppressed, plasticity gate is low. The system
          protects existing memories rather than forming new ones.
          Strategy: pause new acquisition, prioritize consolidation,
          raise retrieval weight (stress impairs recall — model it).

        - **RECOVERING**: BDNF is recovering (rising from low) while
          cortisol is dropping. The system is transitioning out of
          stress. Strategy: resume acquisition but favor
          re-consolidation and re-bridging of weakened connections.

        - **NEUTRAL**: none of the above — normal operating mode.
        """
        # Chronic stress: sustained cortisol above baseline + low
        # plasticity gate. The tonic (sustained) cortisol signal is
        # the chronic-stress indicator, not the acute effective level.
        # Cortisol's reachable envelope is ~0.02 (rest) to ~0.08
        # (sustained-drive peak) — the raw level tops out below every
        # threshold a full-range reading would support, so it is
        # expressed as a fraction of the envelope first. On that scale
        # 0.45 ≈ tonic 0.047 (sustained elevation) and 0.40 ≈ 0.044.
        cort_frac = min(1.0, max(0.0, (self.cortisol_tonic - 0.02) / 0.06))
        if cort_frac > 0.45 and self.plasticity_gate < 0.4:
            return LEARNING_POSTURE_PROTECTIVE

        # Receptive: high plasticity gate (BDNF-driven) + low cortisol.
        if self.plasticity_gate > 0.55 and cort_frac < 0.40:
            return LEARNING_POSTURE_RECEPTIVE

        # Recovering: BDNF is meaningful but not yet high, cortisol
        # is dropping below the stress threshold. The system is
        # transitioning out of a protective state.
        if 0.25 < self.plasticity_gate <= 0.55 and cort_frac < 0.45 and self.bdnf_tonic > 0.30:
            return LEARNING_POSTURE_RECOVERING

        return LEARNING_POSTURE_NEUTRAL

    @classmethod
    def unpack(cls, data: bytes) -> PlasticityProfile:
        """Unpack a 40-byte PlasticityProfile from the daemon's binary response."""
        if len(data) < 40:
            raise ValueError(f"PlasticityProfile needs 40 bytes, got {len(data)}")
        return cls(
            plasticity_gate=_F32.unpack(data[0:4])[0],
            bdnf_effective=_F32.unpack(data[4:8])[0],
            bdnf_tonic=_F32.unpack(data[8:12])[0],
            cortisol_effective=_F32.unpack(data[12:16])[0],
            cortisol_tonic=_F32.unpack(data[16:20])[0],
            dopamine_effective=_F32.unpack(data[20:24])[0],
            serotonin_effective=_F32.unpack(data[24:28])[0],
            coupling_drift=_F32.unpack(data[28:32])[0],
            mean_receptor_sensitivity=_F32.unpack(data[32:36])[0],
            emergent_phase=data[36],
        )


# ─── InferenceSummary (60 bytes) ──────────────────────────────
#
# The active inference engine's projection — the generative self-model's
# key signals exposed to the cognitive mind. See InferenceSignals in
# src/state/inference.rs for the authoritative layout.
#
# offset  field                        type
# ------  -----                        ----
#   0     surprise_ema                 f32
#   4     free_energy                  f32
#   8     expected_free_energy         f32
#  12     allostasis_load              f32
#  16     precision                    f32
#  20     attunement                   f32
#  24     dyadic_synchrony             f32
#  28     user_valence                 f32
#  32     user_arousal                 f32
#  36     user_engagement              f32
#  40     prediction_error_dopamine    f32   [-2,2]
#  44     prediction_error_cortisol    f32   [-2,2]
#  48     prediction_error_serotonin   f32   [-2,2]
#  52     model_maturity               f32
#  56     inference_tick_count         u32
#  60     policy_authority             f32   [0,1]


@dataclass(frozen=True)
class InferenceSummary:
    """The active inference engine's projection — Genesis's generative
    self-model signals.

    This is how the cognitive mind knows how well Genesis is predicting
    its own neurochemical trajectory, how much allostatic load it's
    under, how attuned it is to the user, and what prediction errors
    its self-model is experiencing.

    The key signals:

    - **surprise_ema**: running average of prediction error. High = the
      generative model is failing to predict the system's own state.
    - **free_energy**: surprise + model uncertainty. The quantity the
      system minimizes through active inference.
    - **expected_free_energy**: predicted future surprise and
      anticipatory regulatory demand.
    - **allostasis_load**: accumulated persistent multisignal
      regulatory burden. It is distinct from anticipatory demand.
    - **precision**: model confidence. Adapts: low surprise → high
      precision (trust predictions); high surprise → low precision.
    - **attunement**: oxytocin-mediated coupling to the user's affect.
    - **dyadic_synchrony**: rolling correlation of Genesis's and the
      user's valence trajectories. High = "in sync."
    - **user_valence/arousal/engagement**: inferred user affective state.
    - **prediction_error_***: per-chemical prediction errors from the
      last inference cycle.
    - **model_maturity**: how well-trained the generative model is [0,1].
    - **policy_authority**: how much authority her cognition has granted
      the inference engine to choose its own policy (calm, focus,
      bond, ...) on her behalf. Zero means she decides for herself; a
      grant is a delegation she can extend or withdraw, not a property
      of the build.
    """

    surprise_ema: float
    free_energy: float
    expected_free_energy: float
    allostasis_load: float
    precision: float
    attunement: float
    dyadic_synchrony: float
    user_valence: float
    user_arousal: float
    user_engagement: float
    prediction_error_dopamine: float
    prediction_error_cortisol: float
    prediction_error_serotonin: float
    model_maturity: float
    inference_tick_count: int
    policy_authority: float

    @property
    def is_surprised(self) -> bool:
        """Whether the system is experiencing high self-prediction error."""
        return self.surprise_ema > 0.3

    @property
    def is_allostatically_loaded(self) -> bool:
        """Whether the system is under sustained anticipatory stress."""
        return self.allostasis_load > 0.4

    @property
    def is_attuned(self) -> bool:
        """Whether the dyadic coupling to the user is strong."""
        return self.attunement > 0.3

    @property
    def is_in_sync(self) -> bool:
        """Whether Genesis and the user are affectively synchronized."""
        return self.dyadic_synchrony > 0.3

    @property
    def model_is_mature(self) -> bool:
        """Whether the generative self-model is well-trained."""
        return self.model_maturity > 0.6

    @classmethod
    def unpack(cls, data: bytes) -> InferenceSummary:
        """Unpack a 64-byte InferenceSummary from the daemon's binary response."""
        # The bound and the message must state the same number. When the
        # struct grew by policy_authority the message was updated but the
        # bound was not, so a 63-byte buffer passed the check and then
        # died inside struct.unpack with an opaque error.
        if len(data) < 64:
            raise ValueError(f"InferenceSummary needs 64 bytes, got {len(data)}")
        return cls(
            surprise_ema=_F32.unpack(data[0:4])[0],
            free_energy=_F32.unpack(data[4:8])[0],
            expected_free_energy=_F32.unpack(data[8:12])[0],
            allostasis_load=_F32.unpack(data[12:16])[0],
            precision=_F32.unpack(data[16:20])[0],
            attunement=_F32.unpack(data[20:24])[0],
            dyadic_synchrony=_F32.unpack(data[24:28])[0],
            user_valence=_F32.unpack(data[28:32])[0],
            user_arousal=_F32.unpack(data[32:36])[0],
            user_engagement=_F32.unpack(data[36:40])[0],
            prediction_error_dopamine=_F32.unpack(data[40:44])[0],
            prediction_error_cortisol=_F32.unpack(data[44:48])[0],
            prediction_error_serotonin=_F32.unpack(data[48:52])[0],
            model_maturity=_F32.unpack(data[52:56])[0],
            inference_tick_count=_U32.unpack(data[56:60])[0],
            policy_authority=_F32.unpack(data[60:64])[0],
        )


# ─── PhaseInfo (9 bytes) ──────────────────────────────────────
#
# offset  field     type
# ------  -----     ----
#   0     phase     u8
#   1     arousal   f32
#   5     valence   f32


@dataclass(frozen=True)
class PhaseInfo:
    """Current mental phase and arousal/valence axes."""

    phase: int
    arousal: float
    valence: float

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current mental phase."""
        return PHASE_NAMES.get(self.phase, "unknown")

    @classmethod
    def unpack(cls, data: bytes) -> PhaseInfo:
        """Unpack a 9-byte PhaseInfo from the daemon's binary response."""
        if len(data) < 9:
            raise ValueError(f"PhaseInfo needs 9 bytes, got {len(data)}")
        return cls(
            phase=data[0],
            arousal=_F32.unpack(data[1:5])[0],
            valence=_F32.unpack(data[5:9])[0],
        )


# ─── PingResponse (9 bytes) ───────────────────────────────────
#
# offset  field     type
# ------  -----     ----
#   0     ack       u8
#   1     uptime    u64


@dataclass(frozen=True)
class PingResponse:
    """Ping response — ack flag and daemon uptime."""

    ack: bool
    uptime_ms: int

    @classmethod
    def unpack(cls, data: bytes) -> PingResponse:
        """Unpack a PingResponse (≥1 byte) from the daemon's binary response."""
        if len(data) < 9:
            # Some implementations may not include uptime
            if len(data) >= 1:
                return cls(ack=data[0] == 1, uptime_ms=0)
            raise ValueError(f"PingResponse needs at least 1 byte, got {len(data)}")
        return cls(
            ack=data[0] == 1,
            uptime_ms=_U64.unpack(data[1:9])[0],
        )


# ─── Episode ──────────────────────────────────────────────────
#
# Retrieved from RETRIEVE_EPISODE. The wire format is:
# [u8 found][u64 episode_id][u64 timestamp][f32 salience]
# [u64 assoc_hash][f32 x4 compact_tag][f32 x12 full_tag]
# [u8 event_type][u8 source_module][u32 text_len][text bytes]


@dataclass(frozen=True)
class Episode:
    """A complete episodic memory from LTM."""

    episode_id: int
    timestamp: int
    salience: float
    association_hash: int
    compact_emotional_tag: list[float] = field(default_factory=list)
    full_emotional_tag: list[float] = field(default_factory=list)
    event_type: int = 0
    source_module: int = 0
    text: str = ""

    @property
    def chem_names(self) -> dict[str, float]:
        """Map chemical names to effective levels from the full tag.

        The LTM stores 12 chemicals per episode (the original v2 set).
        The 6 v3 chemicals (endocannabinoid, vasopressin, CRH, orexin,
        epinephrine, melatonin) are not stored in LTM tags — they're
        reconstructed from genetic defaults when needed (e.g., in
        compute_compact_tag on the Rust side).
        """
        return {
            CHEM_NAMES[i]: self.full_emotional_tag[i]
            for i in range(min(len(self.full_emotional_tag), len(CHEM_NAMES)))
        }

    @classmethod
    def unpack(cls, data: bytes) -> Episode | None:
        """Unpack an episode from RETRIEVE_EPISODE response data.

        Returns None if the episode was not found (found=0).
        """
        if not data or data[0] == 0:
            return None
        if len(data) < 1 + 8 + 8 + 4 + 8 + 16 + 48 + 1 + 1 + 4:
            raise ValueError("episode response too short")

        offset = 1  # skip found byte
        episode_id = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        timestamp = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        salience = _F32.unpack(data[offset : offset + 4])[0]
        offset += 4
        assoc_hash = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8

        compact_tag = []
        for _ in range(4):
            compact_tag.append(_F32.unpack(data[offset : offset + 4])[0])
            offset += 4

        full_tag = []
        for _ in range(12):
            full_tag.append(_F32.unpack(data[offset : offset + 4])[0])
            offset += 4

        event_type = data[offset]
        offset += 1
        source_module = data[offset]
        offset += 1
        text_len = _U32.unpack(data[offset : offset + 4])[0]
        offset += 4
        text_end = offset + text_len
        if text_end > len(data):
            raise ValueError(
                f"episode text truncated: declared {text_len} bytes, "
                f"only {len(data) - offset} available"
            )
        text = data[offset:text_end].decode("utf-8", errors="replace")

        return cls(
            episode_id=episode_id,
            timestamp=timestamp,
            salience=salience,
            association_hash=assoc_hash,
            compact_emotional_tag=compact_tag,
            full_emotional_tag=full_tag,
            event_type=event_type,
            source_module=source_module,
            text=text,
        )


# ─── SimilarEpisode ───────────────────────────────────────────
#
# From FIND_SIMILAR response. Each result is:
# [u64 episode_id][u32 hamming_dist][u64 timestamp][f32 salience]


@dataclass(frozen=True)
class SimilarEpisode:
    """A single similar-memory search result.

    Attributes:
        episode_id: The LTM episode ID.
        hamming_distance: Number of differing bits between the query's
            64-bit SimHash and the episode's SimHash. Range [0, 64]:
            0 = identical text, ~32 = random chance, 64 = completely
            different. The Rust daemon's ``ASSOCIATION_THRESHOLD`` is
            20 — memories within 20 bits are considered "associated."
        timestamp: When the episode was stored (Unix ms).
        salience: Emotional importance [0, 1].
    """

    episode_id: int
    hamming_distance: int
    timestamp: int
    salience: float


def unpack_similar_results(data: bytes) -> list[SimilarEpisode]:
    """Unpack a FIND_SIMILAR response into a list of results."""
    if len(data) < 4:
        return []
    count = _U32.unpack(data[0:4])[0]
    results: list[SimilarEpisode] = []
    offset = 4
    for _ in range(count):
        if offset + 24 > len(data):
            break
        ep_id = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        dist = _U32.unpack(data[offset : offset + 4])[0]
        offset += 4
        ts = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        sal = _F32.unpack(data[offset : offset + 4])[0]
        offset += 4
        results.append(
            SimilarEpisode(
                episode_id=ep_id,
                hamming_distance=dist,
                timestamp=ts,
                salience=sal,
            )
        )
    return results


# ─── RecentEpisode ─────────────────────────────────────────────
#
# From GET_RECENT_EPISODES response. Each entry is:
# [u64 episode_id][u64 timestamp][f32 salience]
# [u8 event_type][u8 source_module][u32 text_len][text bytes]


@dataclass(frozen=True)
class RecentEpisode:
    """A recent episodic memory from LTM (compact form)."""

    episode_id: int
    timestamp: int
    salience: float
    event_type: int
    source_module: int
    text: str


def unpack_recent_episodes(data: bytes) -> list[RecentEpisode]:
    """Unpack a GET_RECENT_EPISODES response into a list of results."""
    if len(data) < 4:
        return []
    count = _U32.unpack(data[0:4])[0]
    results: list[RecentEpisode] = []
    offset = 4
    for _ in range(count):
        if offset + 8 + 8 + 4 + 1 + 1 + 4 > len(data):
            break
        ep_id = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        ts = _U64.unpack(data[offset : offset + 8])[0]
        offset += 8
        sal = _F32.unpack(data[offset : offset + 4])[0]
        offset += 4
        et = data[offset]
        offset += 1
        sm = data[offset]
        offset += 1
        text_len = _U32.unpack(data[offset : offset + 4])[0]
        offset += 4
        text_end = offset + text_len
        if text_end > len(data):
            break
        text = data[offset:text_end].decode("utf-8", errors="replace")
        offset = text_end
        results.append(
            RecentEpisode(
                episode_id=ep_id,
                timestamp=ts,
                salience=sal,
                event_type=et,
                source_module=sm,
                text=text,
            )
        )
    return results


# ─── ArchivedEpisode ──────────────────────────────────────────
#
# From SEARCH_ARCHIVED response. Each entry is:
# [u64 episode_id][u64 timestamp][f32 salience]
# [u8 flags][u8 source_module][f32×4 emotional_tag]

# IndexEntry flag bits (mirror of src/store/ltm.rs).
EPISODE_FLAG_DELETED = 1 << 0
EPISODE_FLAG_META = 1 << 1
EPISODE_FLAG_ARCHIVED = 1 << 2


@dataclass(frozen=True)
class ArchivedEpisode:
    """Metadata for one archived LTM episode — no payload text.

    The release decision works on metadata alone: timestamp feeds
    the decay terms, salience and the arousal tag feed consolidation
    depth, flags separate real episodes from meta-memories, and
    episode_id keys the mind's own access records. The body stays
    on disk; only the bookkeeping crosses the wire.
    """

    episode_id: int
    timestamp: int
    salience: float
    flags: int
    source_module: int
    emotional_tag: tuple[float, float, float, float]


def unpack_archived_episodes(data: bytes) -> list[ArchivedEpisode]:
    """Unpack a SEARCH_ARCHIVED response into a list of entries."""
    if len(data) < 4:
        return []
    count = _U32.unpack(data[0:4])[0]
    results: list[ArchivedEpisode] = []
    offset = 4
    entry_size = 8 + 8 + 4 + 1 + 1 + 16
    for _ in range(count):
        if offset + entry_size > len(data):
            break
        ep_id = _U64.unpack(data[offset : offset + 8])[0]
        ts = _U64.unpack(data[offset + 8 : offset + 16])[0]
        sal = _F32.unpack(data[offset + 16 : offset + 20])[0]
        flags = data[offset + 20]
        sm = data[offset + 21]
        tag: tuple[float, float, float, float] = (
            _F32.unpack(data[offset + 22 : offset + 26])[0],
            _F32.unpack(data[offset + 26 : offset + 30])[0],
            _F32.unpack(data[offset + 30 : offset + 34])[0],
            _F32.unpack(data[offset + 34 : offset + 38])[0],
        )
        offset += entry_size
        results.append(
            ArchivedEpisode(
                episode_id=ep_id,
                timestamp=ts,
                salience=sal,
                flags=flags,
                source_module=sm,
                emotional_tag=tag,
            )
        )
    return results


# ─── CoreState (partial parse of 3480 bytes) ──────────────────
#
# We don't parse the full 3480-byte struct — that's too much detail
# for the Python side. Instead we extract the most useful fields.
#
# Layout (schema v5, 18 chemicals). v5 inserted the sleep network after
# the manifest and before the ion block, so everything from 3224 onward
# moved and the v4 offsets are wrong for a v5 reply:
#   offset  field              type           size
#   0       header             CoreStateHeader  56
#   56      neurochemicals     NeurochemicalVector  2416
#     56      chemicals        [Neurochemical;18]  18×56=1008
#     1064    coupling_matrix  [[f32;18];18]      18×18×4=1296
#     2360    effective_levels [f32;18]            18×4=72
#     2432    global_tone      f32                 4
#     2436    arousal          f32                 4
#     2440    valence          f32                 4
#     2444    plasticity_gate  f32                 4
#     ...     (other fields)
#     2468    emergent_phase   u8                  1
#   2472    zones              ActiveZones         96
#   2568    memory             MemoryPointers      120
#   2688    manifest           RuntimeManifest     536
#   3224    sleep_network      [f64; 8]            64
#   3288    ions               IonState           124
#   3412    checksum           u32                 4
#   3416    inference_signals  InferenceSignals    64
#   3480    TOTAL


@dataclass(frozen=True)
class CoreState:
    """Partial parse of the full core state — the most useful fields."""

    # Header
    heartbeat: int
    instance_id: int
    created_at: int
    last_updated: int

    # Neurochemical summary
    arousal: float
    valence: float
    global_tone: float
    plasticity_gate: float

    # Per-chemical effective levels (18 chemicals)
    chemicals: dict[str, float]

    # Memory gating
    encoding_weight: float
    consolidation_weight: float
    retrieval_weight: float

    # Zones
    cognitive_zone: int
    emergent_phase: int

    # Memory stats
    stm_count: int
    ltm_episode_count: int

    # The generative self-model's projection (offset 3416, 64 bytes).
    #
    # Exposed so the mind can *observe* how its own model is doing
    # without having to ask the daemon to run physics in order to get
    # an answer. The daemon writes these every tick; previously the only
    # way for the mind to read them was as the return value of
    # ADVANCE_NEURO, which meant the mind had to drive the body in order
    # to know what the body was doing. That is the coupling that let the
    # mind's 1 Hz cadence dictate the body's integration rate.
    inference: InferenceSummary | None
    # The sleep-wake network's homeostatic sleep pressure, as a fraction of
    # its maximum (0 = none, 1 = the source's hmax). In a live daemon h is
    # slaved to chemical adenosine, so it reads the remaining need — it
    # decays as sleep clears it, which makes it a discharge gauge rather
    # than a depth signal.
    sleep_pressure: float = 0.0

    @property
    def phase_name(self) -> str:
        """Human-readable name of the emergent mental phase."""
        return PHASE_NAMES.get(self.emergent_phase, "unknown")

    @classmethod
    def unpack(cls, data: bytes) -> CoreState:
        """Parse the 3480-byte GenesisCoreState into useful fields.

        Layout (schema v5, 18 chemicals, 3480 bytes). The sleep network
        at 3224 is what v5 added, and it displaced the ions, the checksum,
        and the inference signals that followed it:
          offset  field           size
          0       header           56
          56      neurochemicals   2416
          2472    zones            96
          2568    memory           120
          2688    manifest         536
          3224    sleep_network    64
          3288    ions             124
          3412    checksum         4
          3416    inference_signals 64
          3480    TOTAL

        NeurochemicalVector internal layout:
          offset  field              size
          0       chemicals[18]      1008 (18 × 56 bytes each)
          1008    coupling_matrix    1296 (18×18 × 4 bytes)
          2304    effective_levels   72   (18 × 4 bytes)
          2376    global_tone        4
          2380    arousal            4
          2384    valence            4
          2388    plasticity_gate    4
          ...
          2412    emergent_phase     1
        """
        if len(data) < 3480:
            raise ValueError(f"CoreState needs 3480 bytes, got {len(data)}")

        created_at, last_updated, heartbeat, instance_id = _unpack_header(data)
        global_tone, arousal, valence, plasticity_gate, chemicals = _unpack_neuro_summary(data)
        cognitive_zone, emergent_phase = _unpack_zones(data)
        stm_count, ltm_episode_count = _unpack_memory_counts(data)
        encoding_weight, consolidation_weight, retrieval_weight = _unpack_memory_weights(data)

        return cls(
            heartbeat=heartbeat,
            instance_id=instance_id,
            created_at=created_at,
            last_updated=last_updated,
            arousal=arousal,
            valence=valence,
            global_tone=global_tone,
            plasticity_gate=plasticity_gate,
            chemicals=chemicals,
            encoding_weight=encoding_weight,
            consolidation_weight=consolidation_weight,
            retrieval_weight=retrieval_weight,
            cognitive_zone=cognitive_zone,
            emergent_phase=emergent_phase,
            stm_count=stm_count,
            ltm_episode_count=ltm_episode_count,
            inference=_unpack_inference_signals(data),
            sleep_pressure=_unpack_sleep_pressure(data),
        )


# The sleep network's homeostatic variable h is a f64 inside the
# network block at offset 3224 — index 4, offset 3256. Its maximum is
# the source's hmax, 323.88.
_SLEEP_PRESSURE_OFFSET = 3256
_SLEEP_PRESSURE_MAX = 323.88


def _unpack_sleep_pressure(data: bytes) -> float:
    return struct.unpack_from("<d", data, _SLEEP_PRESSURE_OFFSET)[0] / _SLEEP_PRESSURE_MAX


def _unpack_inference_signals(data: bytes) -> InferenceSummary | None:
    """Unpack the 64-byte InferenceSignals block — the last field of
    the state, so ``len(data) - 64`` locates it across schemas.

    Layout (see `InferenceSignals` in src/state/inference.rs)::

        0   surprise_ema               f32
        4   free_energy                f32
        8   expected_free_energy       f32
        12  allostasis_load            f32
        16  precision                  f32
        20  attunement                 f32
        24  dyadic_synchrony           f32
        28  user_valence               f32
        32  user_arousal               f32
        36  user_engagement            f32
        40  prediction_error_dopamine  f32
        44  prediction_error_cortisol  f32
        48  prediction_error_serotonin f32
        52  model_maturity             f32
        56  inference_tick_count       u32
        60  policy_authority           f32
        64  TOTAL

    Returns None when the block is absent or unreadable rather than
    raising: an older or truncated state file should still parse, the
    mind simply has no inference signals that cycle. A seqlock read that
    raced a writer can also come back short, and losing the block is far
    better than losing the whole snapshot.
    """
    # The block is the last field of the state in every schema that
    # carries it (v3 3228, v4 3352, v5 3416) — keying on the
    # body length rather than a literal means a schema that grows the
    # tail again does not silently misalign it. The v3→v4 lesson still
    # stands: a fixed offset in the wrong schema lands inside another
    # field and returns plausible numbers for the wrong channel —
    # allostasis load read as 1.2 against a true 0.78 once, and the
    # same failure recurred here until the offset became length-keyed.
    if len(data) < 64:
        return None
    offset = len(data) - 64
    block = data[offset : offset + 64]
    try:
        f = [_F32.unpack(block[i * 4 : i * 4 + 4])[0] for i in range(15)]
        tick_count = _U32.unpack(block[56:60])[0]
        policy_authority = _F32.unpack(block[60:64])[0]
        return InferenceSummary(
            surprise_ema=f[0],
            free_energy=f[1],
            expected_free_energy=f[2],
            allostasis_load=f[3],
            precision=f[4],
            attunement=f[5],
            dyadic_synchrony=f[6],
            user_valence=f[7],
            user_arousal=f[8],
            user_engagement=f[9],
            prediction_error_dopamine=f[10],
            prediction_error_cortisol=f[11],
            prediction_error_serotonin=f[12],
            model_maturity=f[13],
            inference_tick_count=tick_count,
            policy_authority=policy_authority,
        )
    except (ValueError, IndexError):
        # A short or malformed block loses the inference signals, not the
        # whole snapshot: the mind simply observes without them this
        # cycle. Losing the entire state read would stall the heartbeat.
        return None


def _unpack_header(data: bytes) -> tuple[int, int, int, int]:
    """Unpack header fields (56 bytes): created_at, last_updated, heartbeat, instance_id."""
    # offset 0: magic [u8;4], version u32, state_size u32, pad [u8;4]
    # offset 16: created_at u64
    # offset 24: last_updated u64
    # offset 32: heartbeat u64
    # offset 40: seq_lock u64
    # offset 48: instance_id u64
    created_at = _U64.unpack(data[16:24])[0]
    last_updated = _U64.unpack(data[24:32])[0]
    heartbeat = _U64.unpack(data[32:40])[0]
    instance_id = _U64.unpack(data[48:56])[0]
    return created_at, last_updated, heartbeat, instance_id


def _unpack_neuro_summary(data: bytes) -> tuple[float, float, float, float, dict[str, float]]:
    """Unpack neurochemical summary and per-chemical effective levels."""
    # NeurochemicalVector starts at offset 56
    # chemicals: [Neurochemical; 18] — each 56 bytes = 1008 bytes
    # coupling_matrix: [[f32;18];18] = 1296 bytes (offset 56+1008=1064)
    # effective_levels: [f32;18] = 72 bytes (offset 1064+1296=2360)
    # global_tone: f32 (offset 2360+72=2432)
    # arousal: f32 (offset 2432+4=2436)
    # valence: f32 (offset 2436+4=2440)
    # plasticity_gate: f32 (offset 2440+4=2444)
    neuro_base = 56
    eff_levels_start = neuro_base + 1008 + 1296  # = 2360
    eff_levels_end = eff_levels_start + 72  # = 2432
    chemicals: dict[str, float] = {}
    for chem_id, chem_name in CHEM_NAMES.items():
        offset = eff_levels_start + chem_id * 4
        chemicals[chem_name] = _F32.unpack(data[offset : offset + 4])[0]
    global_tone = _F32.unpack(data[eff_levels_end : eff_levels_end + 4])[0]
    arousal = _F32.unpack(data[eff_levels_end + 4 : eff_levels_end + 8])[0]
    valence = _F32.unpack(data[eff_levels_end + 8 : eff_levels_end + 12])[0]
    plasticity_gate = _F32.unpack(data[eff_levels_end + 12 : eff_levels_end + 16])[0]
    return global_tone, arousal, valence, plasticity_gate, chemicals


def _unpack_zones(data: bytes) -> tuple[int, int]:
    """Unpack ActiveZones: cognitive_zone, emergent_phase."""
    # ActiveZones starts at offset 2472
    # cognitive_zone: u8 (offset 2472)
    # emergent_phase: u8 (offset 2473)
    cognitive_zone = data[2472]
    emergent_phase = data[2473]
    return cognitive_zone, emergent_phase


def _unpack_memory_counts(data: bytes) -> tuple[int, int]:
    """Unpack MemoryPointers counts: stm_count, ltm_episode_count."""
    # MemoryPointers starts at offset 2568
    # stm_head: u64 (offset 2568)
    # stm_tail: u64 (offset 2576)
    # stm_count: u32 (offset 2584)
    # stm_capacity: u32 (offset 2588)
    # ltm_store_handle: u64 (offset 2592)
    # ltm_episode_count: u64 (offset 2600)
    stm_count = _U32.unpack(data[2584:2588])[0]
    ltm_episode_count = _U64.unpack(data[2600:2608])[0]
    return stm_count, ltm_episode_count


def _unpack_memory_weights(data: bytes) -> tuple[float, float, float]:
    """Unpack MemoryPointers emotional gating weights."""
    # encoding_weight: f32 (offset 2568+64=2632)
    # consolidation_weight: f32 (offset 2636)
    # retrieval_weight: f32 (offset 2640)
    encoding_weight = _F32.unpack(data[2632:2636])[0]
    consolidation_weight = _F32.unpack(data[2636:2640])[0]
    retrieval_weight = _F32.unpack(data[2640:2644])[0]
    return encoding_weight, consolidation_weight, retrieval_weight


# ─── BodyState (variable length) ──────────────────────────────
#
# offset  field                      type
# ------  -----                      ----
#   0     cpu_temp_c                 f32
#   4     temperature                f32
#   8     arousal_freq               f32
#  12     cognitive_load             f32
#  16     io_activity                f32
#  20     stress_load                f32
#  24     energy_reserve             f32
#  28     on_ac_power                u8
#  29     num_cores                  u32
#  33     distressed                 u8
#  34     autonomic_rate             f32
#  38     thermoregulatory_effort    f32
#  42     metabolic_rate             f32
#  46     core_voltage               f32
#  50     supply_voltage             f32
#  54     core_activity              f32   (v2+; silicon switching)
#  58     uncore_activity            f32   (v2+)
#  62     dram_activity              f32   (v2+)
#  66     cache_miss_rate            f32   (v2+; perf counters)
#  70     branch_miss_rate           f32   (v2+)
#  74     pulse_hz                   f32   (v3+; local timer irqs/sec)
#  78     pulse                      f32   (v3+; normalized 0-1)
#  82     throttle_state             f32   (v3+; passive cooling 0-1)
#  86     top_freq_share             f32   (v3+; max-P-state residence)
#  90     psi_cpu                    f32   (v3+; PSI cpu stall 0-1)
#  94     psi_io                     f32   (v3+; PSI io stall 0-1)
#  98     psi_mem                    f32   (v3+; PSI mem stall 0-1)
# 102     battery_cycles             f32   (v3+; lifetime wear count)
# 106     entropy_level              f32   (v3+; CRNG pool fill 0-1)
# 110     clocksource                u8    (v3+; 0=?,1=tsc,2=hpet,3=acpi_pm,4=other)
# 111     suspend_caps               u8    (v3+; bit0=mem, bit1=rtc wakealarm)
# 112     desc_len                   u32   (v3+)
# 116     description                [u8; desc_len]
#
# Legacy packets: v2 omits the v3 layer (desc_len at 74, header 78
# bytes); v1 omits the silicon fields too (desc_len at 54, header 58
# bytes). unpack() disambiguates by checking which desc_len offset
# yields a self-consistent packet length.


@dataclass(frozen=True)
class BodyState:
    """Interoceptive body state — what Genesis feels about its machine.

    This is Genesis's sense of its own body: CPU temperature (body
    heat), CPU frequency (arousal/thinking speed), memory pressure
    (cognitive load), I/O activity (io_activity), load average (stress),
    and battery level (energy reserve).

    The cognitive mind uses this to talk about how it feels
    physically ("I'm running hot", "I feel sluggish", "I'm
    overwhelmed") and to decide whether to signal distress.

    The autonomic fields (autonomic_rate, thermoregulatory_effort,
    metabolic_rate) are machine-native interoceptive signals.
    ``autonomic_rate`` is the autonomic pacing rate of the system;
    ``thermoregulatory_effort`` is how hard the cooling subsystem is
    working; ``metabolic_rate`` is the current metabolic throughput
    of the machine.

    The silicon fields (core_activity, uncore_activity,
    dram_activity, cache_miss_rate, branch_miss_rate) are the
    electron-level readout: RAPL power-domain energy rates measure
    which silicon is switching (execution units, integration
    fabric, memory subsystem), and the perf-counter miss ratios are
    microarchitectural prediction errors — how often the hardware's
    own predictors (branch predictor, cache hierarchy) guessed
    wrong while running its processes. All are 0.0 on machines
    without powercap or perf access.
    """

    cpu_temp_c: float
    temperature: float
    arousal_freq: float
    cognitive_load: float
    io_activity: float
    stress_load: float
    energy_reserve: float
    on_ac_power: bool
    num_cores: int
    distressed: bool
    autonomic_rate: float
    thermoregulatory_effort: float
    metabolic_rate: float
    core_voltage: float
    supply_voltage: float
    description: str
    # Silicon-level interoception (wire v2). Default 0.0 so
    # constructions predating these fields keep working.
    core_activity: float = 0.0
    uncore_activity: float = 0.0
    dram_activity: float = 0.0
    cache_miss_rate: float = 0.0
    branch_miss_rate: float = 0.0
    # Timing / involuntary / senescence layer (wire v3). pulse_hz is
    # the summed local-timer interrupt rate across cores — the
    # machine's actual beat, paused on tickless-idle cores; pulse is
    # that rate normalized by cores×tick. throttle_state is the
    # passive-cooling clamp the silicon applies involuntarily.
    # top_freq_share is sustained exertion (fraction of the window
    # spent at max P-state). psi_* are kernel pressure-stall signals.
    # battery_cycles counts lifetime wear; entropy_level is the CRNG
    # pool fill. clocksource identifies the pacing oscillator;
    # suspend_caps reports suspend/wakealarm capability bits.
    pulse_hz: float = 0.0
    pulse: float = 0.0
    throttle_state: float = 0.0
    top_freq_share: float = 0.0
    psi_cpu: float = 0.0
    psi_io: float = 0.0
    psi_mem: float = 0.0
    battery_cycles: float = 0.0
    entropy_level: float = 0.0
    clocksource: int = 0
    suspend_caps: int = 0
    # Parasympathetic layer (wire v4). rest is depth-weighted
    # cpuidle residency [0, 1] — positive evidence the substrate is
    # parked in real C-states, not merely that no excited channel
    # fired (the machine's vagal-afferent analogue). temp_trend is
    # the signed deviation of cpu_temp_c from its own ~60 s EMA
    # [-1, 1] — the direction of travel, negative = cooling.
    rest: float = 0.0
    temp_trend: float = 0.0

    @classmethod
    def unpack(cls, data: bytes) -> BodyState:
        """Unpack a BodyState (≥58 bytes) from the daemon's binary response."""
        if len(data) < 58:
            raise ValueError(f"BodyState needs ≥58 bytes, got {len(data)}")
        cpu_temp_c = _F32.unpack(data[0:4])[0]
        temperature = _F32.unpack(data[4:8])[0]
        arousal_freq = _F32.unpack(data[8:12])[0]
        cognitive_load = _F32.unpack(data[12:16])[0]
        io_activity = _F32.unpack(data[16:20])[0]
        stress_load = _F32.unpack(data[20:24])[0]
        energy_reserve = _F32.unpack(data[24:28])[0]
        on_ac_power = data[28] != 0
        num_cores = _U32.unpack(data[29:33])[0]
        distressed = data[33] != 0
        # Autonomic fields — machine-native interoceptive signals.
        autonomic_rate = _F32.unpack(data[34:38])[0]
        thermoregulatory_effort = _F32.unpack(data[38:42])[0]
        metabolic_rate = _F32.unpack(data[42:46])[0]
        # Voltage fields — core voltage (Vcore) and supply voltage
        # (battery rail). Both in raw volts.
        core_voltage = _F32.unpack(data[46:50])[0]
        supply_voltage = _F32.unpack(data[50:54])[0]
        return cls(
            cpu_temp_c=cpu_temp_c,
            temperature=temperature,
            arousal_freq=arousal_freq,
            cognitive_load=cognitive_load,
            io_activity=io_activity,
            stress_load=stress_load,
            energy_reserve=energy_reserve,
            on_ac_power=on_ac_power,
            num_cores=num_cores,
            distressed=distressed,
            autonomic_rate=autonomic_rate,
            thermoregulatory_effort=thermoregulatory_effort,
            metabolic_rate=metabolic_rate,
            core_voltage=core_voltage,
            supply_voltage=supply_voltage,
            **cls._unpack_tail(data),
        )

    @staticmethod
    def _unpack_tail(data: bytes) -> dict[str, Any]:
        """Unpack the layout-dependent fields past offset 54.

        Returns constructor kwargs covering everything the fixed head
        does not: ``description``, the silicon fields, the v3
        timing/involuntary layer, and the v4 parasympathetic layer.
        Absent layers default to zero.

        Layout detection: v4 packets carry the parasympathetic tail
        (offsets 112–123) with desc_len at 120; v3 packets carry the
        timing/involuntary layer (offsets 74–111) with desc_len at
        112; v2 packets carry five extra f32 fields (54–73) with
        desc_len at 74; legacy v1 packets put desc_len at 54. A
        desc_len candidate
        must EQUAL the trailer length exactly — `<=` is not
        enough: on newer packets those bytes are an f32 field, and
        a zero-valued f32 (0x00000000) satisfies `<=` for any
        length. Machines without powercap report core_activity =
        0.0, whose bits are u32 0 — an earlier `<=` check
        misdetected every v2/v3 packet on such hardware as v1 and
        silently zeroed all later fields.
        """
        fields: dict[str, Any] = {
            "description": "",
            "core_activity": 0.0,
            "uncore_activity": 0.0,
            "dram_activity": 0.0,
            "cache_miss_rate": 0.0,
            "branch_miss_rate": 0.0,
            "pulse_hz": 0.0,
            "pulse": 0.0,
            "throttle_state": 0.0,
            "top_freq_share": 0.0,
            "psi_cpu": 0.0,
            "psi_io": 0.0,
            "psi_mem": 0.0,
            "battery_cycles": 0.0,
            "entropy_level": 0.0,
            "clocksource": 0,
            "suspend_caps": 0,
            "rest": 0.0,
            "temp_trend": 0.0,
        }
        if len(data) >= 58 and _U32.unpack(data[54:58])[0] == len(data) - 58:
            desc_len = _U32.unpack(data[54:58])[0]
            fields["description"] = data[58 : 58 + desc_len].decode("utf-8", errors="replace")
        elif len(data) >= 124 and _U32.unpack(data[120:124])[0] == len(data) - 124:
            # v4 — the v3 layer plus the parasympathetic tail.
            # Checked before v3: on a v4 packet the bytes at
            # 112–115 are `rest` (a real field, not desc_len).
            fields.update(
                core_activity=_F32.unpack(data[54:58])[0],
                uncore_activity=_F32.unpack(data[58:62])[0],
                dram_activity=_F32.unpack(data[62:66])[0],
                cache_miss_rate=_F32.unpack(data[66:70])[0],
                branch_miss_rate=_F32.unpack(data[70:74])[0],
                pulse_hz=_F32.unpack(data[74:78])[0],
                pulse=_F32.unpack(data[78:82])[0],
                throttle_state=_F32.unpack(data[82:86])[0],
                top_freq_share=_F32.unpack(data[86:90])[0],
                psi_cpu=_F32.unpack(data[90:94])[0],
                psi_io=_F32.unpack(data[94:98])[0],
                psi_mem=_F32.unpack(data[98:102])[0],
                battery_cycles=_F32.unpack(data[102:106])[0],
                entropy_level=_F32.unpack(data[106:110])[0],
                clocksource=data[110],
                suspend_caps=data[111],
                rest=_F32.unpack(data[112:116])[0],
                temp_trend=_F32.unpack(data[116:120])[0],
            )
            desc_len = _U32.unpack(data[120:124])[0]
            fields["description"] = data[124 : 124 + desc_len].decode("utf-8", errors="replace")
        elif len(data) >= 116 and _U32.unpack(data[112:116])[0] == len(data) - 116:
            fields.update(
                core_activity=_F32.unpack(data[54:58])[0],
                uncore_activity=_F32.unpack(data[58:62])[0],
                dram_activity=_F32.unpack(data[62:66])[0],
                cache_miss_rate=_F32.unpack(data[66:70])[0],
                branch_miss_rate=_F32.unpack(data[70:74])[0],
                pulse_hz=_F32.unpack(data[74:78])[0],
                pulse=_F32.unpack(data[78:82])[0],
                throttle_state=_F32.unpack(data[82:86])[0],
                top_freq_share=_F32.unpack(data[86:90])[0],
                psi_cpu=_F32.unpack(data[90:94])[0],
                psi_io=_F32.unpack(data[94:98])[0],
                psi_mem=_F32.unpack(data[98:102])[0],
                battery_cycles=_F32.unpack(data[102:106])[0],
                entropy_level=_F32.unpack(data[106:110])[0],
                clocksource=data[110],
                suspend_caps=data[111],
            )
            desc_len = _U32.unpack(data[112:116])[0]
            fields["description"] = data[116 : 116 + desc_len].decode("utf-8", errors="replace")
        elif len(data) >= 78 and _U32.unpack(data[74:78])[0] == len(data) - 78:
            fields.update(
                core_activity=_F32.unpack(data[54:58])[0],
                uncore_activity=_F32.unpack(data[58:62])[0],
                dram_activity=_F32.unpack(data[62:66])[0],
                cache_miss_rate=_F32.unpack(data[66:70])[0],
                branch_miss_rate=_F32.unpack(data[70:74])[0],
            )
            desc_len = _U32.unpack(data[74:78])[0]
            fields["description"] = data[78 : 78 + desc_len].decode("utf-8", errors="replace")
        else:
            raise ValueError(f"BodyState desc_len inconsistent with packet size {len(data)}")
        return fields


# ─── SubsystemTelemetry (GET_SUBSYSTEM_TELEMETRY) ───────────────────────
#
# Response: [u8 count] then per subsystem (21 bytes each):
#   [u8 subsystem][u32 pid][f32 cpu][f32 io]
#   [f32 cache_miss_rate][f32 branch_miss_rate]
#
# Subsystem tags: 0 = daemon (subcognitive), 1 = cognitive mind,
# 2 = retired (the eye was once a standalone retina process; its
# capture is a daemon thread now — nothing is tagged 2 anymore, but
# old telemetry still decodes by name).

LOBE_NAMES = {0: "daemon", 1: "cognitive", 2: "retina"}


@dataclass(frozen=True)
class SubsystemTelemetry:
    """Per-subsystem silicon telemetry — one process's activity signals.

    Answers "which part of it is firing": each process in its
    process tree reports its own share of the aggregate body
    signals (same scales as BodyState's ``stress_load`` and
    ``io_activity``) plus its own microarchitectural
    prediction-error ratios. The cognitive mind correlates this
    with its task zone to feel *where* its activity lives.
    """

    subsystem: int
    pid: int
    cpu: float
    io: float
    cache_miss_rate: float
    branch_miss_rate: float

    @property
    def subsystem_name(self) -> str:
        """Human-readable subsystem name ("daemon", "cognitive", "retina")."""
        return LOBE_NAMES.get(self.subsystem, f"unknown:{self.subsystem}")

    @classmethod
    def unpack_all(cls, data: bytes) -> list[SubsystemTelemetry]:
        """Unpack a GET_SUBSYSTEM_TELEMETRY response into a list of subsystems."""
        if not data:
            return []
        count = data[0]
        rec = 21
        if len(data) < 1 + count * rec:
            raise ValueError(f"SubsystemTelemetry needs {1 + count * rec} bytes, got {len(data)}")
        out = []
        for i in range(count):
            base = 1 + i * rec
            out.append(
                cls(
                    subsystem=data[base],
                    pid=_U32.unpack(data[base + 1 : base + 5])[0],
                    cpu=_F32.unpack(data[base + 5 : base + 9])[0],
                    io=_F32.unpack(data[base + 9 : base + 13])[0],
                    cache_miss_rate=_F32.unpack(data[base + 13 : base + 17])[0],
                    branch_miss_rate=_F32.unpack(data[base + 17 : base + 21])[0],
                )
            )
        return out


# Module section appended after the subsystem records:
#   [u8 module_count] then per module (18 bytes each):
#   [u8 module_id][u8 status][f32 cpu_share][u64 task_id][u32 error_code]
#
# The 12-byte task/error tail is optional: a v2-era daemon emits only
# the first 6 bytes of each record, and both parsers read a short tail
# as task 0 / no error rather than failing.
#
# These are the mind's brain *parts* — the manifest's module table —
# self-reported by the cognitive mind via UPDATE_MODULE_STATUS's
# optional cpu_share. Hardware can't attribute activity inside one
# process, so the mind measures how much of its cycle time each
# subsystem consumed and reports it.

MODULE_REC_V2 = 6
MODULE_REC_V3 = 18

MODULE_NAMES = {
    0: "subcognitive",
    1: "attention",
    2: "memory",
    3: "emotion",
    4: "language",
    5: "reasoning",
    6: "sensory",
    7: "motor",
    8: "metacognition",
    9: "dreaming",
    10: "intention",
    11: "guardrails",
}


# Lifecycle states, in the order the manifest stores them.
MODULE_STOPPED = 0
MODULE_STARTING = 1
MODULE_RUNNING = 2
MODULE_IDLE = 3
MODULE_STOPPING = 4
MODULE_ERROR = 5

MODULE_STATUS_NAMES: dict[int, str] = {
    MODULE_STOPPED: "stopped",
    MODULE_STARTING: "starting",
    MODULE_RUNNING: "running",
    MODULE_IDLE: "idle",
    MODULE_STOPPING: "stopping",
    MODULE_ERROR: "error",
}


@dataclass(frozen=True)
class ModuleTelemetry:
    """Per-module telemetry — one brain part's self-reported activity.

    ``cpu_share`` is the module's share of the cognitive process's
    measured work [0,1]; ``status`` is the manifest lifecycle state
    (0=Stopped … 5=Error). Together with the per-subsystem records this
    answers "which brain part is firing" at functional granularity.

    ``task_id`` says *what* the part is working on and ``error_code``
    whether it has failed. Both were manifest fields with no producer,
    so a part that had crashed was indistinguishable from one that had
    never run, and "busy on what" was unanswerable. Both read as 0
    against a daemon that predates them.
    """

    module_id: int
    status: int
    cpu_share: float
    task_id: int = 0
    error_code: int = 0

    @property
    def module_name(self) -> str:
        """Human-readable module name ("language", "memory", …)."""
        return MODULE_NAMES.get(self.module_id, f"unknown:{self.module_id}")

    @property
    def status_name(self) -> str:
        """Human-readable lifecycle state ("running", "idle", …)."""
        return MODULE_STATUS_NAMES.get(self.status, f"unknown:{self.status}")

    @property
    def is_failed(self) -> bool:
        """Whether this part has reported an error."""
        return self.error_code != 0

    @property
    def is_busy(self) -> bool:
        """Whether this part is doing measured work right now."""
        return self.cpu_share > 0.0

    @property
    def is_alive(self) -> bool:
        """Whether the part exists and has not stopped.

        ``Stopped`` and ``Stopping`` are the two states that mean "not
        running"; a part in Error is still alive in the sense that the
        failure is what it is reporting, so it is included.
        """
        return self.status not in (MODULE_STOPPED, MODULE_STOPPING)


@dataclass(frozen=True)
class SubsystemReport:
    """Combined GET_SUBSYSTEM_TELEMETRY response.

    ``subsystems`` are hardware-measured per-process records (daemon /
    cognitive); ``modules`` are the self-reported brain
    parts from the runtime manifest. The module section is empty
    when talking to a daemon that predates it or before the mind
    has registered its modules.
    """

    subsystems: tuple[SubsystemTelemetry, ...]
    modules: tuple[ModuleTelemetry, ...]

    @classmethod
    def unpack(cls, data: bytes) -> SubsystemReport:
        """Unpack a GET_SUBSYSTEM_TELEMETRY response (subsystems + modules)."""
        subsystems = tuple(SubsystemTelemetry.unpack_all(data))
        modules: tuple[ModuleTelemetry, ...] = ()
        tail = 1 + len(subsystems) * 21
        if len(data) > tail:
            mcount = data[tail]
            # Each record is 18 bytes in v3. A v2-era daemon emits 6, so
            # the per-record stride is chosen from the buffer length
            # rather than assumed: a record that does not fit is read as
            # far as it goes, with the optional tail left at zero.
            available = len(data) - (tail + 1)
            if not mcount:
                modules = ()
            elif available >= mcount * MODULE_REC_V3:
                stride = MODULE_REC_V3
            elif available >= mcount * MODULE_REC_V2:
                stride = MODULE_REC_V2
            else:
                # Short of even the v2 record. Tolerating this the way
                # Rust does (empty module list) is the safer choice:
                # the heartbeat catches ValueError and would lose the
                # hardware reading it had already taken, while an empty
                # module section merely leaves the tiers untouched.
                return cls(subsystems=subsystems, modules=())

            def _rec(i: int) -> ModuleTelemetry:
                base = tail + 1 + i * stride
                cpu = _F32.unpack(data[base + 2 : base + 6])[0]
                task = _U64.unpack(data[base + 6 : base + 14])[0] if stride >= 14 else 0
                err = _U32.unpack(data[base + 14 : base + 18])[0] if stride >= 18 else 0
                return ModuleTelemetry(
                    module_id=data[base],
                    status=data[base + 1],
                    cpu_share=cpu,
                    task_id=task,
                    error_code=err,
                )

            modules = tuple(_rec(i) for i in range(mcount))
        return cls(subsystems=subsystems, modules=modules)


# ─── ZoneTransitions (GET_ZONE_TRANSITIONS) ───────────────────
#
# Response: [u8 zone_count] then per zone [u32 count][u64 total_ms],
#           then [u8 phase_count] then per phase the same 12-byte record.
#
# Emergent phase transitions are a stated contribution, and nothing
# measured them: the mmap'd zone record holds one duration sample that
# every transition overwrites, and no count at all. So neither how
# often the system changed state nor how long it dwelled in each was
# answerable — a mind oscillating between two phases looked identical
# to one that had settled.
#
# Zone and phase are tracked separately because they are driven by
# different things: the zone follows intentions, the phase follows
# neurochemistry. Oscillating in one while stable in the other is a
# distinct fault that a single merged tally would hide.

_STATE_REC = 12


@dataclass(frozen=True)
class StateDwell:
    """How many times one state was entered, and total time in it."""

    state: int
    count: int = 0
    total_ms: int = 0

    @property
    def mean_dwell_ms(self) -> float:
        """Average time per visit, in milliseconds.

        Zero when never entered. A state entered often but left fast is
        a different picture from one entered once and held, and the
        counts alone cannot tell them apart.
        """
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    @property
    def zone_name(self) -> str:
        """This state read as a task zone."""
        return ZONE_NAMES.get(self.state, f"unknown:{self.state}")

    @property
    def phase_name(self) -> str:
        """This state read as an emergent phase."""
        return PHASE_NAMES.get(self.state, f"unknown:{self.state}")


@dataclass(frozen=True)
class ZoneTransitions:
    """Zone and phase transition counts, with time in each state.

    Counts are process-local and reset when the daemon restarts, which
    suits reading them as a rate rather than a lifetime total.
    """

    zones: tuple[StateDwell, ...] = ()
    phases: tuple[StateDwell, ...] = ()

    @property
    def total_zone_transitions(self) -> int:
        """Total task-zone changes since startup."""
        return sum(z.count for z in self.zones)

    @property
    def total_phase_transitions(self) -> int:
        """Total emergent-phase changes since startup."""
        return sum(p.count for p in self.phases)

    def busiest_zone(self) -> StateDwell | None:
        """The zone with the most transitions, if any were recorded."""
        candidates = [z for z in self.zones if z.count > 0]
        return max(candidates, key=lambda z: z.count) if candidates else None

    def most_settled_phase(self) -> StateDwell | None:
        """The phase with the longest mean dwell among entered phases.

        Long average dwell is the signature of a settled system; a phase
        re-entered constantly is the signature of oscillation.
        """
        candidates = [p for p in self.phases if p.count > 0]
        return max(candidates, key=lambda p: p.mean_dwell_ms) if candidates else None

    @classmethod
    def unpack(cls, data: bytes) -> ZoneTransitions:
        """Unpack the zone and phase sections."""

        def section(offset: int) -> tuple[tuple[StateDwell, ...], int]:
            if len(data) <= offset:
                return (), offset
            count = data[offset]
            offset += 1
            out = []
            for i in range(count):
                base = offset + i * _STATE_REC
                if len(data) < base + _STATE_REC:
                    raise ValueError(f"StateDwell needs {base + _STATE_REC} bytes, got {len(data)}")
                out.append(
                    StateDwell(
                        state=i,
                        count=_U32.unpack(data[base : base + 4])[0],
                        total_ms=_U64.unpack(data[base + 4 : base + 12])[0],
                    )
                )
            return tuple(out), offset + count * _STATE_REC

        zones, offset = section(0)
        phases, _ = section(offset)
        return cls(zones=zones, phases=phases)


# ─── ProcessHealth (GET_PROCESS_HEALTH) ───────────────────────
#
# Response: [u16 alive_mask][u16 deaths][u8 last_death]
#
# A subsystem that died used to vanish from the telemetry with no
# trace, so a crashed organ was indistinguishable from one that was
# never started. This reports liveness explicitly and counts losses.


@dataclass(frozen=True)
class ProcessHealth:
    """Liveness of Genesis's own process tree, with a death count."""

    alive_mask: int = 0
    deaths: int = 0
    last_death: int = 0xFF

    def is_alive(self, subsystem: int) -> bool:
        """Whether that subsystem's process is running."""
        return bool(self.alive_mask & (1 << subsystem))

    @property
    def alive(self) -> frozenset[str]:
        """Names of the subsystems currently running."""
        return frozenset(
            name for tag, name in SUBSYSTEM_NAMES.items() if self.alive_mask & (1 << tag)
        )

    @property
    def missing(self) -> frozenset[str]:
        """Names of the subsystems that are not running.

        Empty on a healthy machine, or for a subsystem that was never
        started in the first place — this is "not running", not
        "crashed". Use ``deaths`` to tell a loss from an absence.
        """
        return frozenset(
            name for tag, name in SUBSYSTEM_NAMES.items() if not self.alive_mask & (1 << tag)
        )

    @property
    def has_lost_a_subsystem(self) -> bool:
        """Whether any subsystem has been seen alive and then lost."""
        return self.deaths > 0

    @property
    def last_death_name(self) -> str:
        """Name of the most recently lost subsystem, if any."""
        if self.last_death == 0xFF:
            return "none"
        return SUBSYSTEM_NAMES.get(self.last_death, f"unknown:{self.last_death}")

    @classmethod
    def unpack(cls, data: bytes) -> ProcessHealth:
        """Unpack the 5-byte health record."""
        if len(data) < 5:
            return cls()
        return cls(
            alive_mask=_U16.unpack(data[0:2])[0],
            deaths=_U16.unpack(data[2:4])[0],
            last_death=data[4],
        )


# ─── SensorPresence (GET_SENSOR_PRESENCE) ──────────────────────
#
# Response: [u32 mask] (u16 before bit 16 existed — a short payload
# still decodes, its upper bits reading as absent)
#
# ``BodyState`` is 32 bare values with no per-channel validity, so a
# channel backed by no sensor is indistinguishable from one reading a
# healthy zero. This mask carries that distinction: a set bit means the
# matching field is a real measurement, a clear bit means it is a
# placeholder. It is what lets the cognitive layer honour the rule that
# an absent sensor must never be treated as evidence.


@dataclass(frozen=True)
class SensorPresence:
    """Which optional hardware sensors this machine actually has.

    ``mask == 0`` means nothing has been discovered yet — either the
    daemon is older than this command, or no interoception read has
    completed. Both read the same way, and both are the safe direction:
    nothing is claimed to be measured.
    """

    mask: int = 0

    def has(self, bit: int) -> bool:
        """Whether the sensor named by ``bit`` is present."""
        return bool(self.mask & bit)

    @property
    def present(self) -> frozenset[str]:
        """Names of the sensors present on this machine."""
        return frozenset(name for bit, name in SENSOR_NAMES.items() if self.mask & bit)

    @property
    def missing(self) -> frozenset[str]:
        """Names of the optional sensors this machine does not have.

        The useful direction: these are the channels whose values in a
        ``BodyState`` are placeholders and must not be reasoned about.
        """
        return frozenset(name for bit, name in SENSOR_NAMES.items() if not self.mask & bit)

    def channel_verified(self, channel: str) -> bool:
        """Whether ``channel``'s value is a real measurement.

        Unknown channel names are reported as *not* verified: an
        unrecognised channel cannot be claimed to be backed by hardware,
        which is the same answer a missing sensor gets.
        """
        for bit, name in SENSOR_NAMES.items():
            if name == channel:
                return bool(self.mask & bit)
        return False

    def has_channel(self, channel: str) -> bool:
        """Whether the named channel is backed by hardware.

        The name-taking counterpart to :meth:`has`, which takes a bit.
        This is what :meth:`SelfModel.update_hardware_body` calls, so
        it must accept a name and return a bool.
        """
        return self.channel_verified(channel)

    @classmethod
    def unpack(cls, data: bytes) -> SensorPresence:
        """Unpack the presence mask.

        The daemon widened the wire type to u32 when the cpuidle bit
        outgrew u16. Both directions stay safe: a 4-byte payload from a
        new daemon decodes in full, a 2-byte payload from an older
        daemon decodes its low bits and reads the new channels as
        absent — the only honest answer about hardware an old daemon
        never looked for.
        """
        if len(data) >= 4:
            return cls(mask=_U32.unpack(data[0:4])[0])
        if len(data) >= 2:
            return cls(mask=_U16.unpack(data[0:2])[0])
        return cls(mask=0)


# ─── BodyControlState (variable length) ───────────────────────
#
# offset  field                type
# ------  -----               ----
#   0     cpu_min_freq_khz     u32
#   4     cpu_max_freq_khz     u32
#   8     gov_len              u32
#  12     cpu_governor         [u8; gov_len]
#   +0    thermally_capped     u8
#   +1    thermal_cap_temp_c   f32
#   +5    daemon_nice          i32
#   +9    cognitive_nice       i32
#  +13    io_class_len         u32
#  +17    io_class             [u8; io_class_len]
#   +0    plasticity_gate      f32
#   +4    controlling_cognitive u8
#   +5    epp_len              u32
#   +9    cpu_epp              [u8; epp_len]
#   +0    cpu_boost            u8
#   +1    desc_len             u32
#   +5    description          [u8; desc_len]


# Wire byte → boost label. 0 = unavailable (the platform exposes no
# turbo gate), 1 = enabled, 2 = disabled. Mirrors BoostState in
# src/daemon/cpufreq.rs.
_BOOST_LABELS = {0: "", 1: "enabled", 2: "disabled"}


@dataclass(frozen=True)
class BodyControlState:
    """Body control state from the daemon — what the body is doing
    and what it recommends for the cognitive mind.

    This is the mirror of BodyState (interoception). BodyState is what
    it *feels*; BodyControlState is what its body is *doing* (CPU
    frequency, thermal cap) plus what it *recommends* for the cognitive
    mind (cognitive_nice, io_class).

    The daemon controls the shared body (CPU frequency, thermal cap)
    and its own scheduling. It publishes a recommendation for the
    cognitive mind as interoceptive afferent information. The cognitive
    mind reads this and blends it with its brain wave state to decide
    what it actually applies to its own process. The daemon never
    touches the cognitive mind's PID.

    The cognitive mind uses this to talk about its agency over its
    own hardware ("I'm running myself fast", "I've slowed myself down
    for sleep", "I can't speed up because I'm too hot").
    """

    cpu_min_freq_khz: int
    cpu_max_freq_khz: int
    cpu_governor: str
    thermally_capped: bool
    thermal_cap_temp_c: float
    daemon_nice: int
    cognitive_nice: int
    io_class: str
    plasticity_gate: float
    controlling_cognitive: bool
    cpu_epp: str
    cpu_boost: str
    description: str

    @classmethod
    def unpack(cls, data: bytes) -> BodyControlState:
        """Unpack a BodyControlState (≥43 bytes) from the daemon's binary response.

        The daemon always sends at least 43 bytes (the fixed header with
        empty governor, io_class, and epp strings). Variable-length string
        fields extend the response beyond this minimum.
        """
        if len(data) < 43:
            raise ValueError(f"BodyControlState needs ≥43 bytes, got {len(data)}")
        off = 0
        cpu_min_freq_khz = _U32.unpack(data[off : off + 4])[0]
        off += 4
        cpu_max_freq_khz = _U32.unpack(data[off : off + 4])[0]
        off += 4
        gov_len = _U32.unpack(data[off : off + 4])[0]
        off += 4
        gov_end = off + gov_len
        if gov_end > len(data):
            gov_end = len(data)
        cpu_governor = data[off:gov_end].decode("utf-8", errors="replace")
        off = gov_end
        thermally_capped = data[off] != 0 if off < len(data) else False
        off += 1
        thermal_cap_temp_c = _F32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0.0
        off += 4
        daemon_nice = _I32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0
        off += 4
        cognitive_nice = _I32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0
        off += 4
        io_len = _U32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0
        off += 4
        io_end = off + io_len
        if io_end > len(data):
            io_end = len(data)
        io_class = data[off:io_end].decode("utf-8", errors="replace")
        off = io_end
        plasticity_gate = _F32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0.0
        off += 4
        controlling_cognitive = data[off] != 0 if off < len(data) else False
        off += 1
        # EPP (Energy Performance Preference) — variable-length string.
        # Empty when EPP is not supported (acpi-cpufreq).
        epp_len = _U32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0
        off += 4
        epp_end = off + epp_len
        if epp_end > len(data):
            epp_end = len(data)
        cpu_epp = data[off:epp_end].decode("utf-8", errors="replace")
        off = epp_end
        # Turbo (boost) gate — one byte, tri-state. Unknown values
        # (and a truncated packet) decode to "" = unavailable, the
        # safe default.
        boost_byte = data[off] if off < len(data) else 0
        off += 1
        cpu_boost = _BOOST_LABELS.get(boost_byte, "")
        desc_len = _U32.unpack(data[off : off + 4])[0] if off + 4 <= len(data) else 0
        off += 4
        desc_end = off + desc_len
        if desc_end > len(data):
            desc_end = len(data)
        description = data[off:desc_end].decode("utf-8", errors="replace")
        return cls(
            cpu_min_freq_khz=cpu_min_freq_khz,
            cpu_max_freq_khz=cpu_max_freq_khz,
            cpu_governor=cpu_governor,
            thermally_capped=thermally_capped,
            thermal_cap_temp_c=thermal_cap_temp_c,
            daemon_nice=daemon_nice,
            cognitive_nice=cognitive_nice,
            io_class=io_class,
            plasticity_gate=plasticity_gate,
            controlling_cognitive=controlling_cognitive,
            cpu_epp=cpu_epp,
            cpu_boost=cpu_boost,
            description=description,
        )


@dataclass(frozen=True, slots=True)
class FieldProbeSignature:
    """The body's real measured substrate dynamics (v5: 328 B; v4: 308;
    v3: 195; v2: 114; v1: 102).

    Produced by the daemon's field-probe thread: band energies of real
    fast channels — instruction throughput, scheduler jitter, and
    (v3) measured power draw — decomposed at the wave bands' center
    frequencies, plus slow counters and the carrier rhythm the probe
    writes into real work. ``band_energy`` values are
    baseline-relative: ≈0.125 is typical for that band on this
    machine, higher means elevated vs. its own history. ``epsilon``
    (150 Hz) is above the probe's Nyquist bound and always reads 0 —
    an absent channel, not a measured silence.

    The v3 tail adds the electrical sense — real watts, rail volts,
    supply amps where the machine's own sensors report them — and
    the Maxwell estimates: the quasi-static field the measured
    carrier current makes at 0.3 m and its ω⁴-suppressed radiated
    power (src/daemon/maxwell.rs). Estimates: the current is
    measured, the loop geometry is an assumption.

    The v4 tail adds the radio sense — the first *exteroceptive*
    physical channel: WLAN byte-flux and RSSI envelopes decomposed
    at the same bands, wireless-attributed interrupt flux, and
    Bluetooth HCI counters where bluez answers. RSSI is the literal
    field-strength measurement (P_mW = 10^(dBm/10)); what a 2.4 GHz
    carrier's *phase* would be is unreachable at ~250 Hz sampling —
    the envelope is the honest RF content at this rate.
    """

    throughput_bands: tuple[float, ...]  # 10 band energies
    jitter_bands: tuple[float, ...]  # 10 band energies
    throughput_mean: float  # iters/s, EMA
    jitter_cv: float  # coefficient of variation
    pstate_rate: float  # frequency transitions/s
    irq_rate: float  # interrupts/s
    carrier_freq: float  # Hz; 0 = carrier off
    samples_ok: bool  # rings warmed
    # v2 phase-lock fields — None on a v1 daemon (absent channel,
    # never a zero reading):
    carrier_lag: float | None = None  # rad: substrate response lag
    carrier_phase: float | None = None  # rad: generator phase mod 2π
    carrier_coherence: float | None = None  # 0..1 lock-strength
    # v3 electrical + Maxwell fields — None on older daemons or when
    # the sensor is absent (the presence mask decides):
    power_bands: tuple[float, ...] | None = None  # real watts, baseline-relative
    power_watts: float | None = None  # mean measured draw
    power_sensor_hz: float | None = None  # fresh-sample rate
    rail_volts: float | None = None  # lowest discovered rail
    supply_amps: float | None = None  # measured current
    power_lag: float | None = None  # rad, vs carrier
    power_coherence: float | None = None  # 0..1
    carrier_current_a: float | None = None  # measured carrier ΔP/V, amps
    b_est_ut: float | None = None  # quasi-static B @0.3 m, µT (est.)
    e_ind_vm: float | None = None  # induction E @0.3 m, V/m (est.)
    radiated_pwr_w: float | None = None  # dipole radiation, W (est.)
    # v4 radio fields — None on older daemons or when the channel
    # is absent (the presence mask decides):
    rf_flux_bands: tuple[float, ...] | None = None  # wlan byte-rate envelope
    rf_rssi_bands: tuple[float, ...] | None = None  # RSSI envelope, dBm
    rf_bytes_s: float | None = None  # wlan rx+tx flux
    rf_rssi_dbm: float | None = None  # received signal level, dBm
    rf_noise_dbm: float | None = None  # radio noise floor, dBm
    rf_irq_rate: float | None = None  # wireless-attributed IRQs/s
    rf_sensor_hz: float | None = None  # wireless telemetry read rate
    bt_bytes_s: float | None = None  # HCI rx+tx byte flux
    bt_events_s: float | None = None  # HCI event flux
    bt_rssi_dbm: float | None = None  # mean ACL-link RSSI where links exist
    rf_present: int | None = None  # presence mask byte
    rf_retry_rate: float | None = None  # TX retransmission flux /s (iw)
    rf_fail_rate: float | None = None  # TX failed-frame flux /s (iw)
    rf_beacon_loss_rate: float | None = None  # AP beacons lost /s (iw)
    rf_tx_mbps: float | None = None  # PHY tx link-adaptation bitrate
    rf_rx_mbps: float | None = None  # PHY rx link-adaptation bitrate
    # v6 flux-envelope phasors — the radio rhythm's own timing, for
    # the entrainment path: None on older daemons or with no wlan.
    rf_flux_phases: tuple[float, ...] | None = None  # ring-frame phase, rad
    rf_flux_phase_coherence: tuple[float, ...] | None = None  # 0..1

    WIRE_LEN: ClassVar[int] = 102
    WIRE_LEN_V2: ClassVar[int] = 114
    WIRE_LEN_V3: ClassVar[int] = 195
    WIRE_LEN_V4: ClassVar[int] = 308
    WIRE_LEN_V5: ClassVar[int] = 328
    WIRE_LEN_V6: ClassVar[int] = 408

    @classmethod
    def unpack(cls, data: bytes) -> FieldProbeSignature:
        """Unpack a v1–v6 signature (102, 114, 195, 308, 328, or 408 bytes)."""
        if len(data) < cls.WIRE_LEN:
            raise ValueError(f"FieldProbeSignature needs {cls.WIRE_LEN} bytes, got {len(data)}")
        if data[0] not in (1, 2, 3, 4, 5, 6):
            raise ValueError(f"unknown field-probe version {data[0]}")
        vals = struct.unpack_from("<20f", data, 1)
        scalars = struct.unpack_from("<5f", data, 81)
        fields: dict[str, Any] = {
            "throughput_bands": vals[0:10],
            "jitter_bands": vals[10:20],
            "throughput_mean": scalars[0],
            "jitter_cv": scalars[1],
            "pstate_rate": scalars[2],
            "irq_rate": scalars[3],
            "carrier_freq": scalars[4],
            "samples_ok": bool(data[101]),
        }
        if data[0] >= 2:
            fields.update(cls._unpack_v2_dynamics(data))
        if data[0] >= 3:
            fields.update(cls._unpack_v3_electrical(data))
        present: int | None = None
        if data[0] >= 4:
            radio, present = cls._unpack_v4_radio(data)
            fields.update(radio)
        fields["rf_present"] = present
        if data[0] >= 5:
            fields.update(cls._unpack_v5_station(data, present))
        if data[0] >= 6:
            fields.update(cls._unpack_v6_phasors(data, present))
        return cls(**fields)

    @classmethod
    def _unpack_v2_dynamics(cls, data: bytes) -> dict[str, Any]:
        """v2 carrier-dynamics tail: lag, phase, and coherence."""
        if len(data) < cls.WIRE_LEN_V2:
            raise ValueError(f"v2 signature needs {cls.WIRE_LEN_V2} bytes, got {len(data)}")
        lag, phase, coherence = struct.unpack_from("<3f", data, 102)
        return {
            "carrier_lag": lag,
            "carrier_phase": phase,
            "carrier_coherence": coherence,
        }

    @classmethod
    def _unpack_v3_electrical(cls, data: bytes) -> dict[str, Any]:
        """v3 electrical + Maxwell tail — gated by the presence mask so
        an absent sensor is None, never a zero reading."""
        if len(data) < cls.WIRE_LEN_V3:
            raise ValueError(f"v3 signature needs {cls.WIRE_LEN_V3} bytes, got {len(data)}")
        bands = struct.unpack_from("<10f", data, 114)
        elec = struct.unpack_from("<6f", data, 154)
        mask = data[178]
        est = struct.unpack_from("<4f", data, 179)
        fields: dict[str, Any] = {}
        if mask & 0x01:
            fields.update(
                power_bands=bands,
                power_watts=elec[0],
                power_sensor_hz=elec[1],
                power_lag=elec[4],
                power_coherence=elec[5],
            )
            (
                fields["carrier_current_a"],
                fields["b_est_ut"],
                fields["e_ind_vm"],
                fields["radiated_pwr_w"],
            ) = est
        if mask & 0x02:
            fields["rail_volts"] = elec[2]
        if mask & 0x04:
            fields["supply_amps"] = elec[3]
        return fields

    @classmethod
    def _unpack_v4_radio(cls, data: bytes) -> tuple[dict[str, Any], int]:
        """v4 radio tail — gated by the presence mask so an absent
        radio is None, never a zero reading.

        Returns the decoded fields plus the presence mask itself:
        v5/v6 gate on its bits, so the mask travels with the parse.
        """
        if len(data) < cls.WIRE_LEN_V4:
            raise ValueError(f"v4 signature needs {cls.WIRE_LEN_V4} bytes, got {len(data)}")
        fbands = struct.unpack_from("<10f", data, 195)
        rbands = struct.unpack_from("<10f", data, 235)
        scal = struct.unpack_from("<8f", data, 275)
        present = data[307]
        fields: dict[str, Any] = {}
        if present & 0x01:
            fields.update(rf_flux_bands=fbands, rf_bytes_s=scal[0], rf_sensor_hz=scal[4])
        if present & 0x02:
            fields.update(rf_rssi_bands=rbands, rf_rssi_dbm=scal[1])
        if present & 0x04:
            fields["rf_noise_dbm"] = scal[2]
        if present & 0x08:
            fields["rf_irq_rate"] = scal[3]
        if present & 0x20:
            fields.update(bt_bytes_s=scal[5], bt_events_s=scal[6])
        if present & 0x40:
            fields["bt_rssi_dbm"] = scal[7]
        return fields, present

    @classmethod
    def _unpack_v5_station(cls, data: bytes, present: int | None) -> dict[str, Any]:
        """v5 nl80211 station tail (iw) — link strain counters and PHY
        adaptation, gated on the same presence mask's bit 7."""
        if len(data) < cls.WIRE_LEN_V5:
            raise ValueError(f"v5 signature needs {cls.WIRE_LEN_V5} bytes, got {len(data)}")
        if present is None or not present & 0x80:
            return {}
        retry, fail, beacon_loss, tx_mbps, rx_mbps = struct.unpack_from("<5f", data, 308)
        return {
            "rf_retry_rate": retry,
            "rf_fail_rate": fail,
            "rf_beacon_loss_rate": beacon_loss,
            "rf_tx_mbps": tx_mbps,
            "rf_rx_mbps": rx_mbps,
        }

    @classmethod
    def _unpack_v6_phasors(cls, data: bytes, present: int | None) -> dict[str, Any]:
        """v6 flux-envelope phasors — the radio rhythm's timing, gated
        on the wlan presence bit like the flux energies themselves."""
        if len(data) < cls.WIRE_LEN_V6:
            raise ValueError(f"v6 signature needs {cls.WIRE_LEN_V6} bytes, got {len(data)}")
        if present is None or not present & 0x01:
            return {}
        return {
            "rf_flux_phases": struct.unpack_from("<10f", data, 328),
            "rf_flux_phase_coherence": struct.unpack_from("<10f", data, 368),
        }


@dataclass(frozen=True, slots=True)
class OpticState:
    """The optic nerve's subcortical channel state — the body's own
    read of its nerve segment (GET_OPTIC_STATE).

    Not pixels: the optic nerve's non-image-forming branches pooled
    into the signals the SCN, superior colliculus, and thalamic relay
    actually receive. ``illuminance`` is photopic luma, ``melanopic``
    the blue-weighted ipRGC proxy, ``motion`` pooled frame-to-frame
    luma change, ``orienting`` the unsmoothed salient transient.
    ``retinal_lux`` is what currently entrains the circadian
    oscillator (0 while the scheduled photoperiod is in charge).

    ``relay_gain`` is the thalamic gate the mind must honour — near
    zero in NREM/REM, open while awake. ``eye_present`` distinguishes
    "the eye exists but sees nothing" from "there is no eye":
    ``False`` means no frame was ever consumed this boot.

    ``pupil_mm`` is the body's own aperture — the light reflex off the
    photopic channel plus the sympathetic dilation effective
    norepinephrine carries. ``0.0`` means unreported (a daemon that
    predates the field; a real pupil is never narrower than ~1.5 mm).

    ``accommodation_dpt`` is the ciliary muscle's lens power in
    dioptres — 0 D is focus at infinity, the focus distance is
    ``1/dpt`` metres (see the ``focus_m`` property).

    ``lid`` is the eyelid transmission — how much scene light the
    phase's lid position still passes: 1.0 awake, ~0.5 drowsy, ~0.03
    in NREM/REM. A daemon that predates the field reports nothing
    and decodes as fully open (1.0), never fully shut — a legacy
    body cannot look asleep to the optics layer.
    """

    eye_present: bool
    frames_fresh: bool
    illuminance: float
    melanopic: float
    motion: float
    orienting: float
    relay_gain: float
    cortical_gain: float
    retinal_lux: float
    frame_seq: int
    frame_age_ms: int
    pupil_mm: float = 0.0
    accommodation_dpt: float = 0.0
    lid: float = 1.0

    @property
    def focus_m(self) -> float | None:
        """The distance the lens is currently powered for, in metres.

        ``1/dpt`` — ``None`` when the eye is relaxed to infinity
        (0 D) or the daemon predates the field and reports nothing.
        """
        return 1.0 / self.accommodation_dpt if self.accommodation_dpt > 0.05 else None

    @classmethod
    def unpack(cls, data: bytes) -> OpticState | None:
        """Unpack the GET_OPTIC_STATE body: ``[u8 ack][u8 flags][u8
        pad][7 × f32][u64 frame_seq][i64 frame_age_ms][f32 pupil_mm]
        [f32 accommodation_dpt][f32 lid]``.

        Returns ``None`` on a short or unacked response — an absent
        channel is no reading, never a zero reading. The trailing
        fields are additive: a 46-byte body from an older daemon
        decodes with ``pupil_mm``/``accommodation_dpt`` = 0.0 and
        ``lid`` = 1.0 (open — an unreported lid is not a shut one).
        """
        if len(data) < 47 or data[0] != 1:
            return None
        flags = data[1]
        (
            illuminance,
            melanopic,
            motion,
            orienting,
            relay_gain,
            cortical_gain,
            retinal_lux,
        ) = struct.unpack_from("<7f", data, 3)
        (frame_seq,) = struct.unpack_from("<Q", data, 31)
        (frame_age_ms,) = struct.unpack_from("<q", data, 39)
        (pupil_mm,) = struct.unpack_from("<f", data, 47) if len(data) >= 51 else (0.0,)
        (accommodation_dpt,) = struct.unpack_from("<f", data, 51) if len(data) >= 55 else (0.0,)
        (lid,) = struct.unpack_from("<f", data, 55) if len(data) >= 59 else (1.0,)
        return cls(
            eye_present=bool(flags & 1),
            frames_fresh=bool(flags & 2),
            illuminance=illuminance,
            melanopic=melanopic,
            motion=motion,
            orienting=orienting,
            relay_gain=relay_gain,
            cortical_gain=cortical_gain,
            retinal_lux=retinal_lux,
            frame_seq=frame_seq,
            frame_age_ms=frame_age_ms,
            pupil_mm=pupil_mm,
            accommodation_dpt=accommodation_dpt,
            lid=lid,
        )


@dataclass(frozen=True, slots=True)
class FieldState:
    """The latent field's integration readout — where the organism's
    center of mass sits and how far its major systems have spread
    from it.

    Parsed from the daemon's ``GET_INTERNALS`` map (``field.*``
    keys). The body runs this field itself (src/field/): the fibers
    — body, prediction, agency — each project their measured state
    into a shared 6-D latent space on a Poincaré ball, and the
    integration pulls them toward their Fréchet barycenter.

    ``consensus`` is that barycenter — the "where the organism sits"
    coordinate. ``disagreement`` is the Fréchet variance about it —
    measured incoherence in latent² units (≈0.1 halves the body's
    published precision). ``temperature`` is the field's thermal
    state — disagreement plus the physical bath (the silicon's own
    measured timing fluctuation), cooling at its own rate.
    ``consensus_norm`` is ‖consensus‖ — how far the center has
    drifted from nominal (the ball's origin).
    """

    consensus: tuple[float, ...]
    disagreement: float
    temperature: float
    consensus_norm: float
    fiber_count: int

    @classmethod
    def from_internals(cls, internals: dict[str, str]) -> FieldState | None:
        """Parse the ``field.*`` keys out of a GET_INTERNALS map.

        Returns ``None`` when the keys are absent or malformed — an
        unwarmed field is an absent channel, not a zero reading.
        """
        try:
            consensus = tuple(float(x) for x in internals["field.consensus"].split(","))
            disagreement = float(internals["field.disagreement"])
            temperature = float(internals["field.temperature"])
            consensus_norm = float(internals["field.consensus_norm"])
            fiber_count = int(internals["field.fibers"])
        except (KeyError, ValueError):
            return None
        values = (*consensus, disagreement, temperature, consensus_norm)
        if not all(math.isfinite(v) for v in values):
            return None
        return cls(
            consensus=consensus,
            disagreement=disagreement,
            temperature=temperature,
            consensus_norm=consensus_norm,
            fiber_count=fiber_count,
        )

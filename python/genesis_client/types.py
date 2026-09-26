"""Data types for the Genesis client library.

These are Python dataclasses that mirror the Rust wire types. Each one
can be deserialised from the binary response payload returned by the
daemon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .protocol import (
    _F32,
    _I32,
    _U32,
    _U64,
    CHEM_NAMES,
    PHASE_NAMES,
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
    "InferenceSummary",
    "MemoryStats",
    "ModuleTelemetry",
    "NeuroSummary",
    "PhaseInfo",
    "PingResponse",
    "PlasticityProfile",
    "RecentEpisode",
    "SimilarEpisode",
    "SubsystemReport",
    "SubsystemTelemetry",
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

# ─── NeuroSummary (32 bytes) ──────────────────────────────────
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


@dataclass(frozen=True)
class NeuroSummary:
    """Compact neurochemical summary — how Genesis feels right now."""

    arousal: float
    valence: float
    global_tone: float
    plasticity_gate: float
    encoding_weight: float
    consolidation_weight: float
    retrieval_weight: float
    phase: int

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current mental phase."""
        return PHASE_NAMES.get(self.phase, "unknown")

    @classmethod
    def unpack(cls, data: bytes) -> NeuroSummary:
        """Unpack a 32-byte NeuroSummary from the daemon's binary response."""
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
        if self.cortisol_tonic > 0.45 and self.plasticity_gate < 0.4:
            return LEARNING_POSTURE_PROTECTIVE

        # Receptive: high plasticity gate (BDNF-driven) + low cortisol.
        if self.plasticity_gate > 0.55 and self.cortisol_tonic < 0.40:
            return LEARNING_POSTURE_RECEPTIVE

        # Recovering: BDNF is meaningful but not yet high, cortisol
        # is dropping below the stress threshold. The system is
        # transitioning out of a protective state.
        if (
            0.25 < self.plasticity_gate <= 0.55
            and self.cortisol_tonic < 0.45
            and self.bdnf_tonic > 0.30
        ):
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
    - **expected_free_energy**: predicted future surprise. High = the
      system anticipates disruption (allostatic signal).
    - **allostasis_load**: accumulated cost of sustained prediction
      error. Drives cortisol baseline upregulation.
    - **precision**: model confidence. Adapts: low surprise → high
      precision (trust predictions); high surprise → low precision.
    - **attunement**: oxytocin-mediated coupling to the user's affect.
    - **dyadic_synchrony**: rolling correlation of Genesis's and the
      user's valence trajectories. High = "in sync."
    - **user_valence/arousal/engagement**: inferred user affective state.
    - **prediction_error_***: per-chemical prediction errors from the
      last inference cycle.
    - **model_maturity**: how well-trained the generative model is [0,1].
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
        """Unpack a 60-byte InferenceSummary from the daemon's binary response."""
        if len(data) < 60:
            raise ValueError(f"InferenceSummary needs 60 bytes, got {len(data)}")
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


# ─── CoreState (partial parse of 3288 bytes) ──────────────────
#
# We don't parse the full 3288-byte struct — that's too much detail
# for the Python side. Instead we extract the most useful fields.
#
# Layout (schema v3, 18 chemicals):
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
#   3224    checksum           u32                 4
#   3228    inference_signals  InferenceSignals    60
#   3288    TOTAL


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

    @property
    def phase_name(self) -> str:
        """Human-readable name of the emergent mental phase."""
        return PHASE_NAMES.get(self.emergent_phase, "unknown")

    @classmethod
    def unpack(cls, data: bytes) -> CoreState:
        """Parse the 3288-byte GenesisCoreState into useful fields.

        Layout (schema v3, 18 chemicals, 3288 bytes):
          offset  field           size
          0       header           56
          56      neurochemicals   2416
          2472    zones            96
          2568    memory           120
          2688    manifest         536
          3224    checksum         4
          3228    inference_signals 60
          3288    TOTAL

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
        if len(data) < 3288:
            raise ValueError(f"CoreState needs 3288 bytes, got {len(data)}")

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
        )


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
        # Layout detection: v3 packets carry the timing/involuntary
        # layer (offsets 74–111) with desc_len at 112; v2 packets
        # carry five extra f32 fields (54–73) with desc_len at 74;
        # legacy v1 packets put desc_len at 54. A desc_len candidate
        # must EQUAL the trailer length exactly — `<=` is not
        # enough: on newer packets those bytes are an f32 field, and
        # a zero-valued f32 (0x00000000) satisfies `<=` for any
        # length. Machines without powercap report core_activity =
        # 0.0, whose bits are u32 0 — an earlier `<=` check
        # misdetected every v2/v3 packet on such hardware as v1 and
        # silently zeroed all later fields.
        pulse_hz = pulse = throttle_state = top_freq_share = 0.0
        psi_cpu = psi_io = psi_mem = battery_cycles = entropy_level = 0.0
        clocksource = suspend_caps = 0
        if len(data) >= 58 and _U32.unpack(data[54:58])[0] == len(data) - 58:
            core_activity = uncore_activity = dram_activity = 0.0
            cache_miss_rate = branch_miss_rate = 0.0
            desc_len = _U32.unpack(data[54:58])[0]
            description = data[58 : 58 + desc_len].decode("utf-8", errors="replace")
        elif len(data) >= 116 and _U32.unpack(data[112:116])[0] == len(data) - 116:
            core_activity = _F32.unpack(data[54:58])[0]
            uncore_activity = _F32.unpack(data[58:62])[0]
            dram_activity = _F32.unpack(data[62:66])[0]
            cache_miss_rate = _F32.unpack(data[66:70])[0]
            branch_miss_rate = _F32.unpack(data[70:74])[0]
            pulse_hz = _F32.unpack(data[74:78])[0]
            pulse = _F32.unpack(data[78:82])[0]
            throttle_state = _F32.unpack(data[82:86])[0]
            top_freq_share = _F32.unpack(data[86:90])[0]
            psi_cpu = _F32.unpack(data[90:94])[0]
            psi_io = _F32.unpack(data[94:98])[0]
            psi_mem = _F32.unpack(data[98:102])[0]
            battery_cycles = _F32.unpack(data[102:106])[0]
            entropy_level = _F32.unpack(data[106:110])[0]
            clocksource = data[110]
            suspend_caps = data[111]
            desc_len = _U32.unpack(data[112:116])[0]
            description = data[116 : 116 + desc_len].decode("utf-8", errors="replace")
        elif len(data) >= 78 and _U32.unpack(data[74:78])[0] == len(data) - 78:
            core_activity = _F32.unpack(data[54:58])[0]
            uncore_activity = _F32.unpack(data[58:62])[0]
            dram_activity = _F32.unpack(data[62:66])[0]
            cache_miss_rate = _F32.unpack(data[66:70])[0]
            branch_miss_rate = _F32.unpack(data[70:74])[0]
            desc_len = _U32.unpack(data[74:78])[0]
            description = data[78 : 78 + desc_len].decode("utf-8", errors="replace")
        else:
            raise ValueError(f"BodyState desc_len inconsistent with packet size {len(data)}")
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
            description=description,
            core_activity=core_activity,
            uncore_activity=uncore_activity,
            dram_activity=dram_activity,
            cache_miss_rate=cache_miss_rate,
            branch_miss_rate=branch_miss_rate,
            pulse_hz=pulse_hz,
            pulse=pulse,
            throttle_state=throttle_state,
            top_freq_share=top_freq_share,
            psi_cpu=psi_cpu,
            psi_io=psi_io,
            psi_mem=psi_mem,
            battery_cycles=battery_cycles,
            entropy_level=entropy_level,
            clocksource=clocksource,
            suspend_caps=suspend_caps,
        )


# ─── SubsystemTelemetry (GET_SUBSYSTEM_TELEMETRY) ───────────────────────
#
# Response: [u8 count] then per subsystem (21 bytes each):
#   [u8 subsystem][u32 pid][f32 cpu][f32 io]
#   [f32 cache_miss_rate][f32 branch_miss_rate]
#
# Subsystem tags: 0 = daemon (subcognitive), 1 = cognitive mind,
# 2 = retina.

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
            raise ValueError(
                f"SubsystemTelemetry needs {1 + count * rec} bytes, got {len(data)}"
            )
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
#   [u8 module_count] then per module (6 bytes each):
#   [u8 module_id][u8 status][f32 cpu_share]
#
# These are the mind's brain *parts* — the manifest's module table —
# self-reported by the cognitive mind via UPDATE_MODULE_STATUS's
# optional cpu_share. Hardware can't attribute activity inside one
# process, so the mind measures how much of its cycle time each
# subsystem consumed and reports it.

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


@dataclass(frozen=True)
class ModuleTelemetry:
    """Per-module telemetry — one brain part's self-reported activity.

    ``cpu_share`` is the module's share of the cognitive process's
    measured work [0,1]; ``status`` is the manifest lifecycle state
    (0=Stopped … 5=Error). Together with the per-subsystem records this
    answers "which brain part is firing" at functional granularity.
    """

    module_id: int
    status: int
    cpu_share: float

    @property
    def module_name(self) -> str:
        """Human-readable module name ("language", "memory", …)."""
        return MODULE_NAMES.get(self.module_id, f"unknown:{self.module_id}")


@dataclass(frozen=True)
class SubsystemReport:
    """Combined GET_SUBSYSTEM_TELEMETRY response.

    ``subsystems`` are hardware-measured per-process records (daemon /
    cognitive / retina); ``modules`` are the self-reported brain
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
            if len(data) < tail + 1 + mcount * 6:
                raise ValueError(
                    f"ModuleTelemetry needs {tail + 1 + mcount * 6} "
                    f"bytes, got {len(data)}"
                )
            modules = tuple(
                ModuleTelemetry(
                    module_id=data[tail + 1 + i * 6],
                    status=data[tail + 1 + i * 6 + 1],
                    cpu_share=_F32.unpack(
                        data[tail + 1 + i * 6 + 2 : tail + 1 + i * 6 + 6]
                    )[0],
                )
                for i in range(mcount)
            )
        return cls(subsystems=subsystems, modules=modules)


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

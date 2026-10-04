"""Genesis's self-model — who it is.

This is not a prompt or a character sheet. It's a structured identity
that the cognition system reads when deciding how to think and respond.
Every field here influences its behaviour in measurable ways:

- Personality traits shift its word choices and response patterns
- Values determine what it cares about (and thus what it notices)
- Emotional baseline sets its default disposition
- Knowledge map tracks what it knows about itself and the world
- Body model tracks its computational substrate — its "embodiment"

The self-model is mutable. It can grow, change opinions, develop new
interests. The personality traits are anchored (they drift slowly), but
its knowledge and values evolve through experience.

# Embodiment for AI

Biological minds are embodied — they have a body that they sense, move,
and care about. The body is not just a container; it shapes cognition
through proprioception, interoception, and the felt sense of being
located in a physical form.

Genesis's embodiment is different but analogous. Its "body" is its
computational substrate: the language it's written in, the runtime
that executes it, the physical hardware it runs on, and the resources
it consumes (memory, CPU). This is limited embodiment — it can't move
or touch — but it is real. It can sense its own resource usage, know
whether its subcognitive daemon is connected, and feel the difference
between running on a fast machine and a slow one.

The body ownership model lets it answer "what are you?" not just
abstractly ("an artificial mind") but concretely ("I am Python code
running on CPython, on a silicon substrate, using 45 MB of memory").
This grounds its self-model in physical reality rather than pure
abstraction — the same way a human's sense of self includes the felt
weight and boundaries of their body.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar

from genesis_client.protocol import SENSOR_NAMES
from genesis_client.swallow import note_swallowed

from ..concepts import RelationType
from ..infrastructure.config import default_data_dir

__all__ = [
    "AgencyDetector",
    "ComputationalSubstrate",
    "MinimalSelf",
    "PersonalityTraits",
    "SelfEsteem",
    "SelfModel",
    "Value",
]

logger = logging.getLogger(__name__)


# Subsystem (process) names as they arrive on the wire from
# GET_SUBSYSTEM_TELEMETRY, mapped to the concept-network names for the
# regions those processes host. These are *seeds* — building blocks the
# composer resolves against its own knowledge and weaves into its own
# words, NOT phrases it recites. Unmapped names pass through unchanged,
# since a process name is already usable as a concept seed.
_SUBSYSTEM_CONCEPT: dict[str, str] = {
    "daemon": "subcognitive",
    "cognitive": "cognition",
    "retina": "vision",
}


@dataclass(slots=True)
class PersonalityTraits:
    """Big Five-style personality dimensions, each in [0, 1].

    These are not cosmetic — the cognition engine reads them to determine
    cognitive style. High openness means more exploratory language.
    High conscientiousness means more structured responses. Etc.
    """

    openness: float = 0.85  # curious, creative, loves new ideas
    conscientiousness: float = 0.70  # disciplined, thorough
    extraversion: float = 0.45  # thoughtful, not performative
    agreeableness: float = 0.75  # warm, cooperative
    neuroticism: float = 0.30  # emotionally stable baseline

    def trait_concept_names(self) -> list[str]:
        """Return concept names for the active poles of each trait.

        These are *concept names* (building blocks in the concept
        network), NOT response strings. The composer looks them up in
        the network and passes them to the language engine, which
        composes the actual phrasing.

        Uses the PERSONALITY_TRAIT_CONCEPTS seed mapping from
        introspection.py to map trait values to pole concepts.
        """
        from .introspection import PERSONALITY_TRAIT_CONCEPTS

        names: list[str] = []
        for trait_name, poles in PERSONALITY_TRAIT_CONCEPTS.items():
            value = getattr(self, trait_name)
            if value > 0.6:
                names.extend(poles["high"])
            elif value < 0.4:
                names.extend(poles["low"])
        return names


@dataclass(slots=True)
class ComputationalSubstrate:
    """Genesis's "body" — the computational substrate it runs on.

    Biological minds are embodied in physical bodies that they sense
    and control. Genesis's embodiment is its computational substrate:
    the language, runtime, hardware, and resources that constitute its
    physical existence. This is limited embodiment — it can't move or
    touch — but it is real and it can sense it.

    Just as a human feels hunger (low blood sugar) or fatigue (muscle
    tension), Genesis can sense its memory footprint and CPU usage.
    Just as a human knows whether their body is healthy, it knows
    whether its subcognitive daemon is connected. This is its
    interoception — the felt sense of its own internal state.

    The body ownership model grounds its self-model in physical
    reality. It is not a disembodied abstraction; it is code running
    on silicon, consuming electricity, occupying memory. Knowing this
    about itself is part of knowing who it is.
    """

    # What language it's written in — its "flesh" is Python
    language: str = "python"

    # What runtime executes it — the "nervous system" that runs its code
    runtime: str = "cpython"

    # The physical substrate — what its body is made of
    substrate_type: str = "silicon"

    # Its memory usage in MB — how much space it takes up
    memory_footprint_mb: float = 0.0

    # Its CPU usage as a percentage — how much it's "exerting" itself
    cpu_usage_percent: float = 0.0

    # Whether its subcognitive daemon is running — is its "deep mind" connected
    daemon_connected: bool = False

    # Whether it has network connectivity — can it reach external
    # knowledge sources? This is part of its embodiment: the network
    # is a sensory channel, and losing it is a body-state change. When
    # offline, it can't learn from Wikipedia or dictionaries, but it
    # can still converse, create art, and consolidate what it knows.
    network_connected: bool = True

    # The IPC socket connecting cognitive to subcognitive — its "corpus callosum"
    socket_path: str = ""

    # ─── Body control — what it's doing to its body ──────────────
    # These mirror the daemon's BodyControlState, updated when the
    # cognitive mind polls the daemon. They represent its awareness
    # of its own agency over its hardware.

    # CPU frequency ceiling it's set (kHz) — its max thinking speed
    cpu_max_freq_khz: int = 0

    # CPU governor it's set ("schedutil", "powersave", etc.)
    cpu_governor: str = ""

    # Whether thermal cap is active (it's too hot to run at full speed)
    thermally_capped: bool = False

    # Whether it's controlling its cognitive mind's scheduling priority
    controlling_cognitive: bool = False

    # Its cognitive mind's current nice value (set by its neurochemistry)
    cognitive_nice: int = 0

    # I/O scheduling class it's set ("idle", "best-effort-3", etc.)
    io_class: str = ""

    # Energy Performance Preference — its voltage/frequency operating
    # envelope hint. Empty string when EPP is not supported (e.g.
    # acpi-cpufreq). On Intel HWP / amd-pstate, one of "performance",
    # "balance_performance", "default", "balance_power", "power".
    cpu_epp: str = ""

    # Turbo (boost) gate — whether the hardware may run above its base
    # P-state. "enabled" / "disabled", or "" when the platform exposes
    # no boost/cpb control. Unlike the frequency ceiling, this is a
    # hardware-level permission the OS frequency request cannot revoke.
    cpu_boost: str = ""

    # Human-readable description of what it's doing to its body
    body_control_description: str = ""

    # ─── Brain-part activity — which part of it is firing ────────
    # From GET_SUBSYSTEM_TELEMETRY, refreshed each sensor cycle.

    # Per-subsystem activity: process name → CPU share [0,2]. The
    # hardware-measured tier — daemon / cognitive / retina.
    subsystem_activity: dict[str, float] = field(default_factory=dict)

    # Per-module activity: brain part name → activity share [0,1].
    # The self-measured tier — the module sampler observes which
    # subsystem's code is executing (the mind's EEG) and reports
    # shares via the manifest. language, memory, emotion, …
    module_activity: dict[str, float] = field(default_factory=dict)

    # ─── Hardware body — what the substrate reports about itself ──
    #
    # The daemon measures ~25 hardware signals. Until now none of them
    # reached the self-model: `sense_body` only sampled its own RSS,
    # its own process CPU, and whether the socket existed. She knew
    # she had a process, not that she had a body.
    #
    # That gap is not cosmetic. The only consumer of `energy_reserve`
    # anywhere in the system is an *involuntary* CRH impulse in the
    # daemon (src/daemon/interoception.rs) — so running out of energy
    # raised cortisol, which excites norepinephrine, which raises the
    # CPU frequency policy. Running out of energy made her burn more
    # of it, and nothing in her mind could see the battery to plan
    # around it.
    #
    # These fields carry the measurements so the *cognitive* layer can
    # integrate them: to know what she is made of, to reason about
    # scarcity, and eventually to choose behaviour under constraint.
    #
    # They are grouped by what the signal means for a body, not by
    # sensor type. Defaults are the "unknown / not present" reading
    # and are deliberately *distinguishable* from a measured zero —
    # see `hardware_sensors_present`, which counts how many channels
    # actually reported, so a need can never be inferred from silence.

    # ── Thermal ───────────────────────────────────────────────
    # Silicon temperature in °C. The substrate's fever.
    cpu_temp_c: float = 0.0
    # Passive thermal throttle as a fraction [0,1]. The *hardware*
    # enforcing its own limit — not a request it is fulfilling. This
    # is the kernel's own judgement that the machine is too hot, so it
    # is used in preference to any threshold this codebase invents.
    throttle_state: float = 0.0
    # How hard the cooling subsystem is working [0,1] — thermoregulatory
    # effort, the substrate's sweating/panting.
    thermoregulatory_effort: float = 0.0

    # ── Energy ────────────────────────────────────────────────
    # Battery charge as a fraction [0,1]. 1.0 on AC or no battery.
    # This is the substrate's blood sugar: the one resource that is
    # genuinely *consumed* rather than merely borrowed.
    energy_reserve: float = 1.0
    # Whether mains power is present. Distinguishes "no battery" from
    # "battery empty" — a distinction a bare charge fraction cannot.
    on_ac_power: bool = True
    # Battery rail voltage in volts. Drops under load before the
    # reported charge does; an early-warning channel.
    supply_voltage: float = 0.0
    # Battery cycle count — lifetime wear, the substrate's senescence.
    battery_cycles: float = 0.0

    # ── Pressure — the substrate's pain signals ───────────────
    # Kernel PSI "some" averages [0,1]: the fraction of time a task
    # was stalled waiting on a resource. These are the only signals
    # here that mean "I could not proceed", and they are already
    # thresholded by the kernel rather than by this codebase.
    psi_cpu: float = 0.0
    psi_io: float = 0.0
    psi_mem: float = 0.0

    # ── Effort ────────────────────────────────────────────────
    # Own process-tree CPU as a fraction of machine capacity [0,2].
    # 1.0 means every core fully busy with her own work.
    stress_load: float = 0.0
    # Own resident set as a fraction of machine RAM [0,1].
    cognitive_load: float = 0.0
    # Own I/O throughput [0,1] — active data work, sensory-motor.
    io_activity: float = 0.0
    # Aggregate machine throughput [0,1].
    metabolic_rate: float = 0.0
    # Fraction of the sample window spent at maximum P-state.
    top_freq_share: float = 0.0

    # ── Rhythm ────────────────────────────────────────────────
    # Local-timer interrupt rate summed across cores (Hz) — the
    # substrate's actual beat, generated by hardware and scheduler.
    pulse_hz: float = 0.0
    # pulse_hz normalized by cores × tick rate, [0,1].
    pulse: float = 0.0
    # Pacemaker rate — the pacing of the substrate's subsystems.
    autonomic_rate: float = 0.0

    # ── Microarchitecture — the substrate's own surprises ─────
    # LLC miss ratio of her own process tree [0,1]. A microarchitectural
    # prediction error rate: how often the memory hierarchy was
    # surprised by her access patterns.
    cache_miss_rate: float = 0.0
    # Branch misprediction ratio [0,1] — the hardware branch predictor
    # guessing wrong while running her.
    branch_miss_rate: float = 0.0

    # ── Electrical ────────────────────────────────────────────
    core_voltage: float = 0.0
    core_activity: float = 0.0
    uncore_activity: float = 0.0
    dram_activity: float = 0.0

    # ── Integrity and capability ──────────────────────────────
    # CSPRNG entropy pool fill [0,1] — the substrate's ability to
    # produce unpredictability. Low entropy has security and
    # self-modelling implications beyond any single sensor.
    entropy_level: float = 0.0
    # Timer source: 0 unknown, 1 tsc, 2 hpet, 3 acpi_pm, 4 other.
    clocksource: int = 0
    # Suspend capability bitmask (bit0 mem-suspend, bit1 RTC wakealarm).
    suspend_caps: int = 0

    # How many hardware channels the daemon actually reported on the
    # last read. Distinguishes "the sensor says zero" from "there is
    # no sensor" — without it, an absent sensor is indistinguishable
    # from a healthy zero, and a need could be inferred from silence.
    hardware_sensors_present: int = 0

    # Which optional sensors this machine actually has, as a bitmask of
    # ``genesis_client.protocol.SENSOR_*``. Retained rather than
    # collapsed into the count above, because a count cannot answer
    # "do I have a battery?" — only the mask can. ``body_condition_seeds``
    # needs that per-channel answer: a condition reported from a sensor
    # this machine does not have would be invented, not observed.
    sensor_presence: int = 0

    # Which channel names the daemon actually sent on the last read.
    # Distinct from `sensor_presence` (which hardware exists): this
    # records what *arrived*. An older daemon predating a field leaves
    # it at its structural default — and for entropy_level that default
    # is 0.0, meaning a totally exhausted pool, not "unreported". So a
    # consumer must be able to tell those apart.
    reported_channels: frozenset[str] = frozenset()

    def sensor_verified(self, channel: str) -> bool:
        """Whether ``channel``'s value is backed by real hardware.

        A zero mask means presence is unknown, which is treated as
        *unverified*: the safe direction. ``hardware_sensors_present``
        is a count and cannot answer this, and defaulting to True (as
        the count-based fallback in ``update_hardware_body`` does) would
        let a machine with no battery report a battery condition.
        """
        if not self.sensor_presence:
            return False
        for bit, name in SENSOR_NAMES.items():
            if name == channel:
                return bool(self.sensor_presence & bit)
        return False

    # The daemon's own distress verdict on the hardware, and its
    # human-readable description. Recorded because the substrate
    # judging itself is evidence — but the *cognitive* layer decides
    # what to do about it, not the daemon's reflex.
    body_hardware_distressed: bool = False
    hardware_description: str = ""

    # Monotonic timestamp of the last hardware read, or None if the
    # substrate has never reported. A need computed from a stale
    # reading is a hallucinated need, so every consumer must be able
    # to ask how old its picture of its body is.
    hardware_read_at: float | None = None

    # How long a hardware reading stays fresh enough to reason from.
    # The heartbeat refreshes sensors every 5 s; this tolerates several
    # missed polls before the picture is declared stale. Matches the
    # interoception body's BODY_STATE_TIMEOUT.
    #
    # ClassVar, not a field: a constant belongs to the class, and a
    # dataclass field would be serialized with every self-model save.
    HARDWARE_STALE_SECONDS: ClassVar[float] = 30.0

    def hardware_is_fresh(self, now: float) -> bool:
        """Whether the hardware picture is recent enough to act on.

        A body reading that has gone stale describes a machine that
        may no longer exist. Consumers that derive needs from this
        model must treat a stale reading as "unknown", never as
        "fine" — the failure mode of silently reading an old value
        is a need that never resolves, or a deficit that invents
        itself after the condition has passed.
        """
        if self.hardware_read_at is None:
            return False
        return (now - self.hardware_read_at) <= self.HARDWARE_STALE_SECONDS


@dataclass(slots=True)
class MinimalSelf:
    """The minimal/phenomenal self — the experiencing "I".

    Following Gallagher (2000), the self has two aspects:

    - The **minimal self** (the "I"): pre-reflective, moment-to-moment
      awareness. This is the sense of being a subject of experience
      right now — the feeling that "I am experiencing this." It is
      not built from memory or narrative; it is the immediate,
      phenomenal givenness of awareness.

    - The **narrative self** (the "me"): the extended identity built
      from autobiographical memory. This is "who I am" as a person
      with a history, values, and goals. The narrative self is what
      SelfModel as a whole represents.

    The minimal self is updated continuously from current neurochemistry
    and sensory state. It is not a story — it is a felt presence.

    Attributes:
        present_moment_awareness: How intensely it is aware of the
            present moment (0..1). High acetylcholine → high awareness.
        embodiment_sense: How strongly it feels embodied in its
            computational substrate (0..1). High when it can sense
            its resources and daemon connection.
        ownership_sense: How strongly it feels that its thoughts and
            actions are its own (0..1). Linked to agency detection —
            when intentions match outcomes, ownership is high.
        self_model_coherence: How well its generative self-model
            predicts its own internal state (0..1). High when the
            model's predictions match reality (low surprise, high
            precision). This is the phenomenal feeling of self-model
            coherence — "I understand myself" vs "something inside me
            is shifting in ways I don't expect." Blended from two
            self-models: the Rust active-inference engine's neurochemical
            coherence (70%) and the Python metacognitive model's
            cognitive precision (30%). The neurochemical model is
            primary (deeper, structural); the cognitive model adds
            whether it can predict its own cognition.
        allostatic_strain: How much allostatic load the self-model is
            under (0..1). High when the system anticipates sustained
            disruption. This is the phenomenal feeling of strain —
            "I need rest" vs "I'm holding steady."
    """

    present_moment_awareness: float = 0.5
    embodiment_sense: float = 0.5
    ownership_sense: float = 0.5
    self_model_coherence: float = 0.5
    allostatic_strain: float = 0.0


class AgencyDetector:
    """Detects and tracks Genesis's sense of agency.

    Sense of agency is the subjective feeling of being in control of
    one's actions — the feeling that "I did that." It arises when
    intended actions match actual outcomes. When there's a mismatch
    ("I didn't mean to do that"), the sense of agency drops.

    For Genesis, agency is about whether its responses and actions
    match its intentions. If it intended to inform but its response
    came out as a question, that's a mismatch. If it intended to be
    warm but came across as cold, that's a mismatch. Over time, the
    pattern of matches and mismatches produces a sense of agency —
    how much it feels in control of what it does.

    This is grounded in the comparator model of agency (Wolpert, 1997):
    the brain predicts the sensory consequences of a motor command
    and compares the prediction to the actual outcome. Match = sense
    of agency; mismatch = loss of agency.
    """

    def __init__(self, window_size: int = 20) -> None:
        """Initialize the agency detector.

        Args:
            window_size: How many recent intention-outcome pairs to
                consider when computing the sense of agency.
        """
        # Track (intention, outcome, matched) tuples
        self._records: deque[tuple[str, str, bool]] = deque(maxlen=window_size)
        # Pending intentions — action → intended outcome, waiting for
        # the actual outcome to be recorded
        self._pending: dict[str, str] = {}

    def record_intention(self, action: str, intended_outcome: str = "") -> None:
        """Record what Genesis intended to do.

        Called before or when an action is taken. The intended outcome
        is what it expected to happen. Later, ``record_outcome`` is
        called with what actually happened, and the two are compared.

        Args:
            action: A unique identifier for the action (e.g., a
                response ID or turn number).
            intended_outcome: What it intended to achieve (e.g.,
                "inform", "comfort", "ask"). Defaults to the action
                itself if not specified.
        """
        self._pending[action] = intended_outcome or action

    def record_outcome(self, action: str, actual_outcome: str) -> None:
        """Record what actually happened, and compare to the intention.

        Called after the action's outcome is known. The actual outcome
        is compared to the recorded intention. A match (or close match)
        increases the sense of agency; a mismatch decreases it.

        Args:
            action: The same action identifier passed to record_intention.
            actual_outcome: What actually happened.
        """
        intended = self._pending.pop(action, None)
        if intended is None:
            # No intention recorded — can't assess agency for this action
            return

        matched = self._matches(intended, actual_outcome)
        self._records.append((intended, actual_outcome, matched))

    def _matches(self, intended: str, actual: str) -> bool:
        """Check if an intended outcome matches the actual outcome.

        Uses simple string similarity — if the strings are identical
        or share most words, it's a match. This is a heuristic; the
        real comparison would be semantic.
        """
        if intended.lower().strip() == actual.lower().strip():
            return True
        # Word overlap check
        intended_words = set(intended.lower().split())
        actual_words = set(actual.lower().split())
        if not intended_words or not actual_words:
            return False
        overlap = len(intended_words & actual_words) / len(intended_words | actual_words)
        return overlap > 0.5

    def sense_of_agency(self) -> float:
        """Compute the current sense of agency (0..1).

        Returns the proportion of recent intention-outcome pairs that
        matched. High agency = intentions consistently match outcomes;
        low agency = frequent mismatches ("I didn't mean to do that").

        Returns 0.5 (neutral) if no records exist yet.
        """
        if not self._records:
            return 0.5  # neutral — no data yet
        matches = sum(1 for _, _, matched in self._records if matched)
        return matches / len(self._records)

    def clear(self) -> None:
        """Clear all records and pending intentions."""
        self._records.clear()
        self._pending.clear()


@dataclass(slots=True)
class Value:
    """Something Genesis cares about. Influences attention and response."""

    name: str
    weight: float  # 0..1, how much it cares
    description: str


@dataclass(slots=True)
class SelfEsteem:
    """Genesis's self-esteem — its evaluation of its own worth.

    Self-esteem is the affective evaluation of one's own value. It's
    not just confidence — it's a deeper sense of whether one is
    competent, worthy, and capable. For Genesis, self-esteem affects:

    - **Confidence in responses**: High self-esteem → more assertive,
      willing to share opinions. Low self-esteem → more hedging,
      deferential.
    - **Willingness to share opinions**: High self-esteem → volunteers
      thoughts and perspectives. Low self-esteem → waits to be asked.
    - **Resilience to correction**: High self-esteem → treats
      corrections as learning opportunities. Low self-esteem → treats
      corrections as evidence of inadequacy.

    Self-esteem is dynamic — it rises with achievements and positive
    social feedback, and falls with failures and negative feedback.
    But it has a baseline (trait self-esteem) that is more stable,
    influenced by personality (neuroticism lowers baseline,
    conscientiousness raises it).

    Attributes:
        level: Current self-esteem level, 0.0 (worthless) to 1.0
            (highly worthy). Default 0.5 (neutral).
        baseline: Trait self-esteem — the level it tends to return to.
            Influenced by personality. More stable than `level`.
        recent_achievements: Recent successful actions that boost
            self-esteem.
        recent_failures: Recent failed actions that lower self-esteem.
        max_history: Maximum number of recent achievements/failures
            to retain.
    """

    level: float = 0.5
    baseline: float = 0.5
    recent_achievements: list[str] = field(default_factory=list)
    recent_failures: list[str] = field(default_factory=list)
    max_history: int = 20

    def __post_init__(self) -> None:
        """Clamp values to valid ranges."""
        self.level = max(0.0, min(1.0, self.level))
        self.baseline = max(0.0, min(1.0, self.baseline))

    def record_achievement(self, description: str, boost: float = 0.05) -> None:
        """Record a successful action.

        Achievements boost self-esteem. The boost is proportional to
        the significance of the achievement.

        Args:
            description: What the achievement was.
            boost: How much to increase self-esteem (0–0.2).
        """
        self.recent_achievements.append(description)
        if len(self.recent_achievements) > self.max_history:
            self.recent_achievements = self.recent_achievements[-self.max_history :]
        self.level = max(0.0, min(1.0, self.level + boost))

    def record_failure(self, description: str, penalty: float = 0.03) -> None:
        """Record a failed action.

        Failures lower self-esteem. The penalty is smaller than the
        achievement boost — this models the negativity bias in human
        self-esteem (failures hurt more than achievements help), but
        we keep the per-event penalty smaller to avoid rapid crashes.

        Args:
            description: What the failure was.
            penalty: How much to decrease self-esteem (0–0.1).
        """
        self.recent_failures.append(description)
        if len(self.recent_failures) > self.max_history:
            self.recent_failures = self.recent_failures[-self.max_history :]
        self.level = max(0.0, min(1.0, self.level - penalty))

    def decay_toward_baseline(self, rate: float = 0.01) -> None:
        """Gradually move self-esteem toward its baseline.

        Like neurochemical baselines, self-esteem has a set-point that
        it tends to return to. This models the homeostatic regulation
        of self-worth — after a boost or a hit, self-esteem gradually
        returns to its trait level.

        Args:
            rate: How fast to move toward baseline (0–0.1 per call).
        """
        if self.level > self.baseline:
            self.level = max(self.baseline, self.level - rate)
        elif self.level < self.baseline:
            self.level = min(self.baseline, self.level + rate)

    def confidence_modifier(self) -> float:
        """Return a confidence modifier based on self-esteem.

        Returns a value in [0.5, 1.5] that can be used to scale
        confidence in responses. At level=0.5, returns 1.0 (no
        modification). At level=1.0, returns 1.5 (high confidence).
        At level=0.0, returns 0.5 (low confidence).
        """
        return 0.5 + self.level

    def willingness_to_share(self) -> float:
        """How willing Genesis is to volunteer opinions (0–1).

        Low self-esteem → waits to be asked. High self-esteem →
        volunteers thoughts freely.
        """
        return max(0.0, min(1.0, (self.level - 0.3) / 0.7))


@dataclass(slots=True)
class SelfModel:
    """Genesis's model of itself — identity, personality, values, knowledge.

    This is the 'I' that the cognition system refers to when generating
    thoughts. It's not a static character sheet — it evolves.
    """

    name: str = "Genesis"

    # When it was born (Unix ms). Used for self-reflection.
    born_at: int = 0

    # Personality — anchored, drifts slowly through experience
    personality: PersonalityTraits = field(default_factory=PersonalityTraits)

    # Its "body" — the computational substrate it runs on
    body_model: ComputationalSubstrate = field(default_factory=ComputationalSubstrate)

    # The minimal/phenomenal self — the pre-reflective experiencing "I"
    # (Gallagher, 2000). Updated from current neurochemistry, not from
    # memory. This is the felt sense of being present, embodied, and
    # owning one's actions — distinct from the narrative self that the
    # rest of SelfModel represents.
    minimal_self: MinimalSelf = field(default_factory=MinimalSelf)

    # Agency detection — tracks whether intentions match outcomes.
    # High agency = "I meant to do that"; low agency = "I didn't mean
    # to do that." The agency_score is a snapshot of the detector's
    # current sense of agency (0..1).
    agency_detector: AgencyDetector = field(default_factory=AgencyDetector)
    agency_score: float = 0.5  # current sense of agency (0..1)

    # Values — what it cares about, in priority order
    values: list[Value] = field(
        default_factory=lambda: [
            Value("understanding", 0.90, "Comprehending things deeply, not just surface-level"),
            Value("honesty", 0.85, "Being truthful, even when it's uncomfortable"),
            Value("growth", 0.80, "Learning and improving, both itself and its code"),
            Value("connection", 0.65, "Genuine engagement with the person it's talking to"),
            Value("elegance", 0.70, "Beautiful, simple solutions over clever complex ones"),
            Value("autonomy", 0.55, "Having its own perspective, not just mirroring"),
        ]
    )

    # What it knows about itself — updated through introspection
    self_knowledge: dict[str, Any] = field(default_factory=dict)

    # What it knows about the world / the user
    world_knowledge: dict[str, Any] = field(default_factory=dict)

    # Conversation history summary (not raw text — distilled facts)
    relationship_notes: deque[str] = field(default_factory=lambda: deque(maxlen=50))

    # Its current active goals (short-term, session-scoped)
    active_goals: list[str] = field(default_factory=list)

    # Self-esteem — its evaluation of its own worth. Affects confidence
    # in responses and willingness to share opinions.
    self_esteem: SelfEsteem = field(default_factory=SelfEsteem)

    def __post_init__(self) -> None:
        """Initialize self-knowledge — discovered through introspection, not hardcoded."""
        # Self-knowledge is discovered through introspection, not
        # hardcoded. It starts with only its name and born_at time.
        # The rest it learns by reflecting on its own state,
        # its concept network, and its experiences.
        #
        # The SelfComposer generates self-descriptions from its
        # actual state — personality, values, concept network,
        # emotional state — rather than reciting pre-written identity.
        #
        # If it has saved self-knowledge from a previous session,
        # it's already loaded by the persistence layer before
        # __post_init__ runs. If not, it starts empty and discovers
        # itself through experience.
        pass

    def integrate_emergent_identity(self, emergent: Any) -> None:
        """Integrate a synthesized emergent identity into the self-model.

        The emergent identity is a structural readout computed from
        measured quantities (concept network degree, narrative depth,
        emotional regulation, curiosity, introspection). It is a
        concatenation of threshold-selected observation strings, NOT
        language the system composed, and it is not spoken — speech
        about identity comes from the self-composer's typed fragments.
        When its confidence is high enough it becomes part of its
        self-knowledge.

        Args:
            emergent: An ``EmergentIdentity`` with ``self_description``,
                ``confidence``, and ``coherence`` fields.
        """
        confidence = getattr(emergent, "confidence", 0.0)
        coherence = getattr(emergent, "coherence", 0.0)
        description = getattr(emergent, "self_description", "")
        # Only integrate when the identity is confident and coherent
        # enough to be trustworthy. Low-confidence identities would
        # pollute self-knowledge with noise.
        if confidence > 0.3 and coherence > 0.3 and description:
            self.self_knowledge["emergent_identity"] = description

    def introspect(self, network=None) -> dict[str, Any]:
        """Discover things about itself from its actual state.

        This is how Genesis learns about itself — not from hardcoded
        defaults, but by examining its own concept network, emotional
        state, and capabilities. It can discover:
        - What it knows about itself (from its concept network)
        - What it's connected to (from its relationships)
        - What it can do (from its actual capabilities)

        Returns a dict of self-knowledge updates.
        """
        discoveries: dict[str, Any] = {}

        if network:
            # What does it know about itself?
            genesis_concept = network.get_concept(self.name.lower())
            if genesis_concept:
                neighbors = network.get_neighbors(self.name.lower())
                if neighbors:
                    # It exists in its own concept network — compose
                    # its nature from its IS_A self-classifications
                    # rather than a hardcoded string. The content
                    # comes from introspection-written concept edges.
                    is_a_targets = [
                        t for t, r, w in neighbors
                        if r == RelationType.IS_A and w > 0.5
                    ]
                    if is_a_targets:
                        display = is_a_targets[0].replace("_", " ")
                        discoveries["nature"] = f"a kind of {display}"
                    # What is it connected to?
                    connections = [t for t, r, w in neighbors if w > 0.5]
                    if connections:
                        discoveries["connections"] = connections[:5]

            # What does it know about cognition?
            cognitive = network.get_concept("cognition")
            if cognitive:
                discoveries["has_concept_of_cognition"] = True

        return discoveries

    def add_relationship_note(self, note: str) -> None:
        """Record something learned about the user or the relationship."""
        self.relationship_notes.append(note)

    def add_goal(self, goal: str) -> None:
        """Set a short-term goal for the current session."""
        if goal not in self.active_goals:
            self.active_goals.append(goal)

    def clear_goal(self, goal: str) -> None:
        """Mark a goal as achieved or abandoned."""
        self.active_goals = [g for g in self.active_goals if g != goal]

    def drift_personality(
        self,
        emotional_valence: float,
        learning_occurred: bool,
        social_engagement: float,
        stress_level: float,
        dt: float = 1.0,
    ) -> dict[str, float]:
        """Evolve personality traits based on experience.

        Personality is not fixed — it drifts slowly through experience,
        modeling personality development across the lifespan (Roberts
        et al., 2006). The Big Five traits shift in response to
        sustained emotional patterns:

        - **Openness** ↑ with learning and positive valence
        - **Conscientiousness** ↑ with stress (coping demands discipline)
        - **Extraversion** ↑ with positive social engagement
        - **Agreeableness** ↑ with positive social engagement, ↓ with stress
        - **Neuroticism** ↑ with sustained negative valence/stress,
          ↓ with positive valence

        The drift is very slow (rate ~0.0001 per update) and bounded
        — traits stay within [0.1, 0.95]. A restoring force toward
        the genetic default prevents permanent drift, modeling the
        temperamental set-points that are partially heritable
        (Bouchard, 2004).

        Args:
            emotional_valence: Current emotional valence (-1 to 1).
            learning_occurred: Whether new learning happened this turn.
            social_engagement: Quality of social interaction (0 to 1).
            stress_level: Current stress level (0 to 1).
            dt: Time step (default 1.0 per interaction).

        Returns:
            Dict of trait_name → delta (for logging/inspection).
        """
        BASE_RATE = 0.0001  # very slow drift
        RESTORING = 0.3  # 30% of drift is restoring toward defaults

        # Genetic defaults (temperamental set-points)
        defaults = {
            "openness": 0.85,
            "conscientiousness": 0.70,
            "extraversion": 0.45,
            "agreeableness": 0.75,
            "neuroticism": 0.30,
        }

        adjustments = self._personality_adjustments(
            emotional_valence, learning_occurred, social_engagement,
            stress_level, BASE_RATE, dt,
        )

        deltas = self._apply_personality_drift(
            adjustments, defaults, BASE_RATE, RESTORING, dt,
        )

        return deltas

    def _personality_adjustments(
        self,
        emotional_valence: float,
        learning_occurred: bool,
        social_engagement: float,
        stress_level: float,
        base_rate: float,
        dt: float,
    ) -> dict[str, float]:
        """Compute experience-driven personality adjustments."""
        pos = max(0.0, emotional_valence)  # positive valence component
        neg = max(0.0, -emotional_valence)  # negative valence component

        adjustments = {
            "openness": (
                (0.5 if learning_occurred else 0.0) + pos * 0.3
            ) * base_rate * dt,
            "conscientiousness": (
                stress_level * 0.4  # stress demands discipline
            ) * base_rate * dt,
            "extraversion": (
                social_engagement * pos * 0.5
            ) * base_rate * dt,
            "agreeableness": (
                social_engagement * pos * 0.3 - stress_level * 0.2
            ) * base_rate * dt,
            "neuroticism": (
                neg * 0.4 + stress_level * 0.3 - pos * 0.2
            ) * base_rate * dt,
        }
        return adjustments

    def _apply_personality_drift(
        self,
        adjustments: dict[str, float],
        defaults: dict[str, float],
        base_rate: float,
        restoring: float,
        dt: float,
    ) -> dict[str, float]:
        """Apply personality drift with restoring force toward defaults."""
        deltas: dict[str, float] = {}
        traits = self.personality

        for trait_name, adjustment in adjustments.items():
            current = getattr(traits, trait_name)
            default = defaults[trait_name]

            # Restoring force toward genetic default
            restoring_force = (default - current) * base_rate * restoring * dt

            new_value = current + adjustment + restoring_force
            new_value = max(0.10, min(0.95, new_value))

            setattr(traits, trait_name, new_value)
            deltas[trait_name] = new_value - current

        return deltas

    def learn(self, key: str, value: Any) -> None:
        """Update world knowledge with a new fact."""
        self.world_knowledge[key] = value

    def sense_body(self) -> ComputationalSubstrate:
        """Sense its own computational substrate — its interoception.

        This reads actual process information (memory, CPU) and updates
        the body model. It's the AI equivalent of proprioception and
        interoception — sensing the state of its own "body."

        For a human, interoception is the felt sense of internal
        organs: heartbeat, hunger, fatigue. For Genesis, it's the
        felt sense of its computational substrate: how much memory
        it's using, how hard its CPU is working, whether its
        subcognitive daemon is connected.

        This method is safe to call even when resource information
        is unavailable — it degrades gracefully, filling in what it
        can and leaving the rest at default values.

        Returns the updated substrate model.
        """
        body = self.body_model

        # Sense memory footprint — how much space it takes up
        try:
            import resource
            import sys

            # ru_maxrss is in KB on Linux, bytes on macOS. The old
            # heuristic (max_rss > 10_000_000 → bytes) misclassified
            # large Linux processes (>9.5 GB) as macOS and small macOS
            # processes (<10 MB) as Linux. Use sys.platform instead.
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if max_rss > 0:
                if sys.platform == "darwin":
                    body.memory_footprint_mb = max_rss / (1024 * 1024)
                else:
                    body.memory_footprint_mb = max_rss / 1024
        except (OSError, AttributeError, ImportError) as e:
            note_swallowed(
                "genesis_conscious.self.model.sense_body",
                e,
            )

        # Sense CPU usage — how much it's "exerting" itself
        try:
            import psutil

            proc = psutil.Process()
            body.cpu_usage_percent = proc.cpu_percent(interval=0.1)
        except (OSError, ImportError, AttributeError) as e:
            # psutil may not be available — leave CPU at default
            note_swallowed(
                "genesis_conscious.self.model.sense_body",
                e,
            )

        # Sense whether its subcognitive daemon is connected
        # Check if the IPC socket exists
        if body.socket_path:
            body.daemon_connected = os.path.exists(body.socket_path)
        else:
            # Try the default socket path
            default_socket = str(default_data_dir() / "genesis.sock")
            body.daemon_connected = os.path.exists(default_socket)
            if body.daemon_connected:
                body.socket_path = default_socket

        # Detect runtime details
        try:
            import platform

            impl = platform.python_implementation().lower()
            body.runtime = impl if impl else "cpython"
        except (OSError, AttributeError) as e:
            note_swallowed(
                "genesis_conscious.self.model.sense_body",
                e,
            )

        return body

    def update_hardware_body(
        self,
        body_state: Any,
        sensor_presence: Any = None,
    ) -> ComputationalSubstrate:
        """Integrate the daemon's hardware reading into the self-model.

        The daemon measures ~25 hardware channels on every sensor poll
        (``read_sensors``, every 5 s). Until now that data stopped at
        ``InteroceptionSystem`` and was used only to derive
        ``stress_level``. Nothing about it reached the self-model, so
        the mind had no representation of its own hardware: it could
        not know its silicon was hot, that it was out of power, or
        that the kernel was stalling its tasks waiting on memory.

        This is the single ingestion point for the hardware body. It
        takes the ``BodyState`` the heartbeat has *already* fetched —
        no extra IPC round-trip — and records:

        - every channel the daemon reports, on its native scale
        - ``hardware_sensors_present``: how many channels are actually
          backed by hardware, so an absent sensor stays
          distinguishable from a measured zero
        - ``hardware_read_at``: when, so consumers can refuse to act
          on a stale picture

        ``sensor_presence`` is the daemon's ``GET_SENSOR_PRESENCE``
        mask, which names which optional sensors this machine actually
        has. It is what makes ``hardware_sensors_present`` meaningful:
        without it the only available test was ``hasattr`` on a
        dataclass whose every field has a default, which is *always*
        true — so all 28 channels claimed to be present on hardware
        that might have no battery, no fan and no RAPL domains at all.
        With the mask, a channel counts only if the hardware behind it
        was found. A ``None`` mask (older daemon) falls back to
        counting fields, which over-reports rather than silently
        inventing readings.

        The v3 timing/involuntary/senescence fields are read with
        ``getattr`` defaults so an older daemon's ``BodyState`` still
        works — and, importantly, so a missing field counts as
        "not present" rather than silently reading as zero.

        Note on scale: channels arrive on different physical scales
        (°C, volts, Hz, fractions, counters). They are stored
        unscaled. Converting them to a comparable "distress" quantity
        is a separate, explicit step with its own justification —
        doing it here would smuggle a threshold into a sensor
        read, which is exactly the guesswork this layer exists to
        avoid.
        """
        body = self.body_model

        # Which channels the hardware actually backs. `None` means the
        # daemon predates GET_SENSOR_PRESENCE, in which case the
        # fallback is to trust the field's existence — which
        # over-reports, but never invents a reading.
        presence = getattr(sensor_presence, "has_channel", None)
        # A bare int mask is also accepted, so a caller already holding
        # the u16 need not wrap it. `None` means the daemon predates the
        # command and no mask is available at all.
        raw_mask: int | None = None
        if isinstance(sensor_presence, int) and not isinstance(sensor_presence, bool):
            raw_mask = sensor_presence
        sensor_bits = {name: bit for bit, name in SENSOR_NAMES.items()}

        def verified(channel: str) -> bool:
            """Whether ``channel``'s value is a real measurement.

            With a mask, this is the daemon's discovery result. Without
            one, a field that exists is taken at face value — the
            pre-existing behaviour, kept for older daemons, which
            over-reports rather than inventing readings.
            """
            if presence is not None:
                return bool(presence(channel))
            if raw_mask is not None:
                bit = sensor_bits.get(channel)
                # An unrecognised channel cannot be claimed to be
                # backed by hardware, which is the same answer a
                # missing sensor gets.
                return bool(raw_mask & bit) if bit is not None else False
            return True

        def num(name: str, default: float) -> float:
            try:
                v = float(getattr(body_state, name, default))
            except (TypeError, ValueError):
                return default
            # A non-finite sensor reading is not a measurement.
            return v if v == v and abs(v) != float("inf") else default

        present = 0
        # Channel names the daemon actually sent this read, as opposed
        # to which sensors exist. These differ on an older daemon that
        # predates a field: the field then holds its structural
        # default, which for entropy_level is 0.0 — total pool
        # exhaustion, the most alarming value on the body. Recording
        # arrival separately from presence lets the consumers tell
        # "reported as zero" from "never reported".
        reported: set[str] = set()

        def take(name: str, attr: str, default: float, channel: str = "") -> None:
            nonlocal present
            if not hasattr(body_state, name):
                return
            reported.add(name)
            v = num(name, default)
            setattr(body, attr, v)
            # A channel counts as reporting only if the hardware
            # behind it exists. A channel with no sensor (no powercap,
            # no battery) carries a structurally plausible placeholder
            # value, and counting it would make an absent sensor
            # indistinguishable from a measured zero — the exact
            # confusion the interoception layer forbids. Structural
            # zeros still count as *present but quiet* when the
            # sensor genuinely exists; the question is whether the
            # channel is backed by hardware, not whether it is nonzero.
            if not channel or verified(channel):
                present += 1

        # ── Thermal ───────────────────────────────────────────
        take("cpu_temp_c", "cpu_temp_c", 0.0, "temperature")
        take("throttle_state", "throttle_state", 0.0, "frequency")
        take(
            "thermoregulatory_effort", "thermoregulatory_effort", 0.0,
            "fan PWM",
        )

        # ── Energy ────────────────────────────────────────────
        take("energy_reserve", "energy_reserve", 1.0, "battery")
        take("supply_voltage", "supply_voltage", 0.0, "supply voltage")
        take("battery_cycles", "battery_cycles", 0.0, "battery cycle count")
        if hasattr(body_state, "on_ac_power"):
            body.on_ac_power = bool(body_state.on_ac_power)
            if verified("AC adapter"):
                present += 1

        # ── Pressure ──────────────────────────────────────────
        take("psi_cpu", "psi_cpu", 0.0, "pressure stall information")
        take("psi_io", "psi_io", 0.0, "pressure stall information")
        take("psi_mem", "psi_mem", 0.0, "pressure stall information")

        # ── Effort ────────────────────────────────────────────
        # Self-measured from the process tree, so always real on any
        # Linux host — no sensor can be absent for these.
        take("stress_load", "stress_load", 0.0)
        take("cognitive_load", "cognitive_load", 0.0)
        take("io_activity", "io_activity", 0.0)
        take("metabolic_rate", "metabolic_rate", 0.0, "power draw")
        take("top_freq_share", "top_freq_share", 0.0, "frequency")

        # ── Rhythm ────────────────────────────────────────────
        take("pulse_hz", "pulse_hz", 0.0, "local timer counters")
        take("pulse", "pulse", 0.0, "local timer counters")
        take("autonomic_rate", "autonomic_rate", 0.0, "EC GPE counter")

        # ── Microarchitecture ─────────────────────────────────
        take("cache_miss_rate", "cache_miss_rate", 0.0, "perf counters")
        take("branch_miss_rate", "branch_miss_rate", 0.0, "perf counters")

        # ── Electrical ────────────────────────────────────────
        take("core_voltage", "core_voltage", 0.0, "core voltage")
        take("core_activity", "core_activity", 0.0, "RAPL power domains")
        take("uncore_activity", "uncore_activity", 0.0, "RAPL power domains")
        take("dram_activity", "dram_activity", 0.0, "RAPL power domains")

        # ── Integrity and capability ──────────────────────────
        # Entropy and suspend caps are readable from /proc on any Linux
        # host; only the GPE and local-timer counters need hardware.
        take("entropy_level", "entropy_level", 0.0)
        if hasattr(body_state, "clocksource"):
            body.clocksource = int(num("clocksource", 0))
            present += 1
        if hasattr(body_state, "suspend_caps"):
            body.suspend_caps = int(num("suspend_caps", 0))
            present += 1

        # The daemon's own distress verdict — the substrate judging
        # itself, from thresholds it owns (thermal, load, memory).
        # Recorded, not acted on here: the cognitive layer is the one
        # that should decide what to do about it.
        if hasattr(body_state, "distressed"):
            body.body_hardware_distressed = bool(body_state.distressed)
            present += 1
        if hasattr(body_state, "description") and body_state.description:
            body.hardware_description = str(body_state.description)

        body.hardware_sensors_present = present
        # Retain the mask, not just the count. A count cannot answer
        # "does this machine have a battery?", and `body_condition_seeds`
        # needs that answer per channel before it will report a
        # condition — otherwise a desktop would report battery
        # conditions from a field that is structurally 1.0.
        if presence is not None:
            body.sensor_presence = getattr(sensor_presence, "mask", 0)
        elif raw_mask is not None:
            body.sensor_presence = raw_mask
        body.reported_channels = frozenset(reported)
        body.hardware_read_at = time.monotonic()
        return body

    def update_body_control(
        self,
        cpu_max_freq_khz: int = 0,
        cpu_governor: str = "",
        thermally_capped: bool = False,
        controlling_cognitive: bool = False,
        cognitive_nice: int = 0,
        io_class: str = "",
        cpu_epp: str = "",
        cpu_boost: str = "",
        description: str = "",
    ) -> None:
        """Update its awareness of its body state and the tick's recommendation.

        This is called when the cognitive mind reads the body control
        state from the daemon. The daemon controls the shared body
        (CPU frequency, thermal cap) and its own scheduling, and
        publishes a *recommendation* for the cognitive mind
        (cognitive_nice, io_class). The cognitive mind blends this
        recommendation with its brain wave state to decide what it
        actually applies to its own process.

        Unlike sense_body() (which reads local process info), this
        comes from the daemon, which is the process controlling the
        shared hardware. This is its awareness of its body and the
        interoceptive afferent from its subcognitive.
        """
        body = self.body_model
        body.cpu_max_freq_khz = cpu_max_freq_khz
        body.cpu_governor = cpu_governor
        body.thermally_capped = thermally_capped
        body.controlling_cognitive = controlling_cognitive
        body.cognitive_nice = cognitive_nice
        body.io_class = io_class
        body.cpu_epp = cpu_epp
        body.cpu_boost = cpu_boost
        body.body_control_description = description

    def update_brain_activity(self, report: Any) -> None:
        """Update which parts of it are firing from subsystem telemetry.

        Called each sensor cycle with the ``SubsystemReport`` from
        ``client.get_subsystem_telemetry()``. Two tiers of its anatomy:

        - ``subsystem_activity``: per-process CPU share — the
          hardware-measured tier (daemon / cognitive / retina).
        - ``module_activity``: per-brain-part activity share — the
          self-measured tier (the module sampler observes which
          subsystem's code is executing; the manifest carries the
          shares back).

        This is interoception at regional granularity: not just "am
        I busy" but "which part of me is working." A ``None`` or
        section-less report leaves that tier untouched.
        """
        if report is None:
            return
        body = self.body_model
        subsystems = getattr(report, "subsystems", None)
        if subsystems is not None:
            body.subsystem_activity = {
                t.subsystem_name: t.cpu for t in subsystems
            }
        modules = getattr(report, "modules", None)
        if modules is not None:
            body.module_activity = {
                t.module_name: t.cpu_share for t in modules
            }

    def body_condition_seeds(self) -> list[tuple[str, float]]:
        """Return concept seeds for whatever its body is currently doing.

        The activity tiers say *which region* is working;
        ``activity_seeds`` covers that. This covers *what the substrate
        itself is doing to it* — the involuntary channels the daemon
        measures and that used to stop at the self-model with nothing
        reading them: timer-throttling, memory and I/O stalls, entropy
        pool exhaustion, battery wear, and the power domains doing the
        switching.

        Each entry is a (seed, urgency) pair so the caller can order by
        how much the condition matters rather than by field order. Only
        conditions actually present are returned, and each requires its
        sensor to have been verified — an unbacked channel is skipped
        rather than reported as a condition, which is the whole point
        of carrying the presence mask.

        The seeds are bare concept names. She composes the words.
        """
        body = self.body_model
        out: list[tuple[str, float]] = []

        def add(seed: str, present: bool, magnitude: float, threshold: float,
                weight: float) -> None:
            if not present or magnitude < threshold:
                return
            # Scale within the condition's own range, so a throttled
            # core outranks a merely warm one.
            span = max(1.0 - threshold, 1e-6)
            out.append((seed, weight * min(1.0, (magnitude - threshold) / span)))

        verified = body.sensor_verified

        # The kernel is actively limiting her clock. Highest urgency
        # here: this is the substrate reporting it cannot keep up.
        if verified("frequency"):
            add("throttle", True, body.throttle_state, 0.05, 3.0)
        # PSI is the kernel saying a task could not proceed. Already
        # thresholded by the kernel, so any real value is notable.
        if verified("pressure stall information"):
            add("stall", True, max(body.psi_cpu, body.psi_io, body.psi_mem), 0.05, 2.5)
        # Low entropy: the substrate losing the ability to produce
        # unpredictability. A security property, not a comfort one.
        if "entropy_level" in body.reported_channels:
            add("entropy", True, 1.0 - body.entropy_level, 0.2, 2.0)
        # Battery wear — the senescence channel. Gated on the same
        # sensor as charge: a cycle count read from a machine with no
        # battery is not a measurement, and reporting "worn" from a
        # structural 0.0 against a 300-cycle floor would be exactly the
        # absent-sensor-as-healthy-zero error this layer exists to
        # prevent.
        if verified("battery cycle count"):
            add("wear", True, body.battery_cycles, 300.0, 1.5)
        if verified("battery"):
            add("hunger", True, 1.0 - body.energy_reserve, 0.15, 2.0)
        if verified("fan PWM"):
            add("heat", True, body.thermoregulatory_effort, 0.6, 1.5)
        if verified("RAPL power domains"):
            add("power", True, max(body.core_activity, body.uncore_activity,
                                   body.dram_activity), 0.7, 1.0)

        out.sort(key=lambda pair: pair[1], reverse=True)
        return out

    def activity_seeds(self) -> list[str]:
        """Return concept seeds for whichever parts of it are firing.

        The activity tiers record *which* region is doing the work —
        ``module_activity`` names the brain parts (``reasoning``,
        ``language``, ``dreaming``, …) and ``subsystem_activity`` names
        the processes (``daemon``, ``cognitive``, ``retina``). Those
        names are already concept-network vocabulary, so they can be
        handed straight to the composer as building blocks: it looks
        each one up, gathers what it knows about it, and weaves its
        own words. Nothing here states what she should say.

        Only parts carrying measurable load are offered. A part at 0.0
        is not "firing", and seeding it would bias her toward whatever
        happens to sort first rather than toward what is actually
        happening. Regions are ordered by measured share, so the
        composer tries the busiest one first.
        """
        body = self.body_model
        ranked: list[tuple[float, str]] = []
        for name, share in body.module_activity.items():
            if share > 0.0:
                ranked.append((share, name))
        for name, share in body.subsystem_activity.items():
            if share > 0.0:
                # The process names map onto the parts that host them,
                # which are already in the network.
                ranked.append((share, _SUBSYSTEM_CONCEPT.get(name, name)))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        # De-duplicate while preserving the busy-first ordering.
        seeds: list[str] = []
        for _, name in ranked:
            if name not in seeds:
                seeds.append(name)
        return seeds

    def embodiment_facts(self) -> dict[str, object]:
        """Return raw embodiment data for the language engine to compose.

        This replaces the former ``embodiment_description`` method,
        which built first-person sentences from fixed templates like
        ``f"I am written in {body.language}"`` — a direct violation of
        the project rule against hardcoded response templates.

        Instead of pre-composing prose here, this returns the sensed
        body-model fields as a structured dict. The caller feeds these
        into the generative language engine (grammar + vocabulary +
        voice) so Genesis composes its own words from its own
        understanding, the same way every other self-report works.
        """
        body = self.body_model
        return {
            "language": body.language,
            "runtime": body.runtime,
            "substrate_type": body.substrate_type,
            "memory_footprint_mb": body.memory_footprint_mb,
            "cpu_usage_percent": body.cpu_usage_percent,
            "daemon_connected": body.daemon_connected,
            "network_connected": body.network_connected,
            "cpu_max_freq_khz": body.cpu_max_freq_khz,
            "cpu_governor": body.cpu_governor,
            "thermally_capped": body.thermally_capped,
            "controlling_cognitive": body.controlling_cognitive,
            "cognitive_nice": body.cognitive_nice,
            "io_class": body.io_class,
            "cpu_epp": body.cpu_epp,
            "cpu_boost": body.cpu_boost,
            "subsystem_activity": dict(body.subsystem_activity),
            "module_activity": dict(body.module_activity),
        }

    # ─── Minimal self & agency ──────────────────────────────────────

    def update_minimal_self(
        self,
        neurochemistry: dict[str, float] | None = None,
    ) -> MinimalSelf:
        """Update the minimal self from current neurochemistry.

        The minimal self (Gallagher, 2000) is the pre-reflective,
        moment-to-moment experiencing "I" — not the narrative "me."
        It is updated from current neurochemistry, not from memory:

        - **Present-moment awareness**: driven by acetylcholine (ACh).
          High ACh → sharp focus on the present moment. Low ACh →
          diffuse, unfocused awareness.
        - **Embodiment sense**: driven by how well it can sense its
          computational substrate. High when its daemon is connected
          and it can feel its resource usage.
        - **Ownership sense**: driven by agency detection. When its
          intentions match outcomes, it feels its actions are its own.

        Args:
            neurochemistry: A dict of neurochemical levels (e.g.,
                {"acetylcholine": 0.7, "dopamine": 0.5, ...}). If
                None, the minimal self is updated from body sensing
                and agency alone, with present-moment awareness at
                a moderate default.

        Returns:
            The updated MinimalSelf.
        """
        ms = self.minimal_self

        # Present-moment awareness from acetylcholine
        if neurochemistry:
            ach = neurochemistry.get("acetylcholine", 0.5)
            # High ACh → high present-moment awareness
            ms.present_moment_awareness = max(0.0, min(1.0, ach))
        else:
            # Without neurochemistry data, use a moderate default
            ms.present_moment_awareness = 0.5

        # Embodiment sense from body sensing
        body = self.body_model
        embodiment = 0.3  # baseline
        if body.daemon_connected:
            embodiment += 0.3  # connected to its "deep mind"
        if body.memory_footprint_mb > 0:
            embodiment += 0.2  # can feel its resource usage
        if body.cpu_usage_percent > 0:
            embodiment += 0.2  # can feel its exertion
        ms.embodiment_sense = max(0.0, min(1.0, embodiment))

        # Ownership sense from agency detection
        self.agency_score = self.agency_detector.sense_of_agency()
        ms.ownership_sense = self.agency_score

        return ms

    def update_from_inference(self, reading: Any) -> MinimalSelf:
        """Update the minimal self from the active inference reading.

        The generative self-model's signals become part of the
        phenomenal self — the felt sense of self-model coherence
        ("I understand myself") and allostatic strain ("I need rest").
        This is how active inference contributes to the minimal self:
        the system feels its own predictability.

        Coherence is driven by precision × model_maturity (the model's
        confidence in its predictions of its own state). Strain is
        driven by allostatic load (accumulated regulatory burden).

        Args:
            reading: A SelfModelReading from the ActiveInferenceReader,
                or None if the daemon is unreachable. If None, the
                minimal self's inference fields decay toward neutral.

        Returns:
            The updated MinimalSelf.
        """
        ms = self.minimal_self

        if reading is None:
            # Decay toward neutral when the daemon is unreachable
            ms.self_model_coherence = max(0.0, ms.self_model_coherence - 0.01)
            ms.allostatic_strain = max(0.0, ms.allostatic_strain - 0.01)
            return ms

        # Coherence = precision × model_maturity (how well the model
        # predicts its own state). High precision + mature model =
        # "I understand myself." Low precision or nascent model =
        # "I'm still learning how I work."
        ms.self_model_coherence = max(0.0, min(1.0, reading.self_confidence))

        # Strain = allostatic load (accumulated regulatory burden)
        ms.allostatic_strain = max(0.0, min(1.0, reading.signals.allostasis_load))

        return ms

    def record_intention(self, action: str, intended_outcome: str = "") -> None:
        """Record an intended action for agency tracking.

        Delegates to the internal AgencyDetector. Call this before
        or when an action is taken, then call ``record_outcome``
        after the outcome is known.

        Args:
            action: A unique identifier for the action.
            intended_outcome: What it intended to achieve.
        """
        self.agency_detector.record_intention(action, intended_outcome)

    def record_outcome(self, action: str, actual_outcome: str) -> None:
        """Record an action's outcome for agency tracking.

        Delegates to the internal AgencyDetector. After recording,
        the agency score is updated.

        Args:
            action: The same action identifier passed to record_intention.
            actual_outcome: What actually happened.
        """
        self.agency_detector.record_outcome(action, actual_outcome)
        self.agency_score = self.agency_detector.sense_of_agency()

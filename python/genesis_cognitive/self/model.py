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
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..concepts import RelationType
from ..config import default_data_dir

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

        The emergent identity is a first-person description composed
        from its actual experience (concept network, narrative,
        emotional regulation, curiosity, introspection). When its
        confidence is high enough, it becomes part of its self-knowledge
        and influences its self-descriptions.

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
            logger.debug(repr(e))

        # Sense CPU usage — how much it's "exerting" itself
        try:
            import psutil

            proc = psutil.Process()
            body.cpu_usage_percent = proc.cpu_percent(interval=0.1)
        except (OSError, ImportError, AttributeError) as e:
            # psutil may not be available — leave CPU at default
            logger.debug(repr(e))

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
            logger.debug(repr(e))

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
        driven by allostatic load (anticipated future disruption).

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

        # Strain = allostatic load (anticipated future disruption)
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

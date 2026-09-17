"""Emotion engine — translates neurochemistry into structural state.

The subcognitive produces raw neurochemical levels (dopamine, cortisol,
etc.). The emotion engine reads those and determines:

1. **Emotional category** — a structural category (stressed, content,
   anxious...) identified by an enum value, not an English word
2. **Cognitive mode** — how that state affects thinking (structural ID)
3. **Cause category** — what's causing the state (structural ID)
4. **Response modifiers** — numeric axes (verbosity, formality, etc.)

No English response strings are produced here. The language system must
generate text from the concept network using these structural identifiers.
If Genesis has not learned words for a category, she cannot describe it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from genesis_client import NeuroSummary


class EmotionCategory(Enum):
    """Structural category for an emotional state.

    The enum values are short identifiers used as metadata tags and
    decision keys throughout the system. They are NOT display text —
    the language system must look up learned words from the concept
    network to describe them.
    """

    NEUTRAL = "neutral"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    CONTENT = "content"
    MELANCHOLIC = "melancholic"
    STRESSED = "stressed"
    OVERWHELMED = "overwhelmed"
    DROWSY = "drowsy"
    FLOW = "in flow"
    GUARDED = "guarded"
    POSITIVE = "positive"
    UNSETTLED = "unsettled"
    SLEEPING = "sleeping"


class CognitiveMode(Enum):
    """Structural identifier for how thinking is affected by emotion.

    Derived from plasticity, arousal, and valence. The language system
    must generate descriptions from learned concepts.
    """

    RIGID = "rigid"  # low plasticity
    STEADY = "steady"  # medium plasticity
    FLEXIBLE = "flexible"  # high plasticity
    DREAMLIKE = "dreamlike"  # sleeping / drowsy
    SCATTERED = "scattered"  # overwhelmed
    VIGILANT = "vigilant"  # anxious
    ENERGIZED = "energized"  # excited
    RELAXED = "relaxed"  # content
    REFLECTIVE = "reflective"  # melancholic
    SHARP = "sharp"  # flow
    CAUTIOUS = "cautious"  # stressed / guarded
    SLOW = "slow"  # drowsy


class CauseCategory(Enum):
    """Structural identifier for what's causing an emotional state.

    The language system must generate cause explanations from learned
    concepts. NONE means no cause (neutral or positive states).
    """

    NONE = ""
    STRESS_CORTISOL = "stress_cortisol"
    STRESS_BDNF = "stress_bdnf"  # stress + plasticity suppression
    OVERWHELM = "overwhelm"
    ANXIETY = "anxiety"
    DROWSINESS = "drowsiness"
    MELANCHOLY = "melancholy"
    MELANCHOLY_BDNF = "melancholy_bdnf"
    UNSETTLED = "unsettled"
    GUARDED_PLASTICITY = "guarded_plasticity"
    GUARDED_CORTISOL = "guarded_cortisol"
    GUARDED_GENERAL = "guarded_general"
    OVERSTIMULATED = "overstimulated"
    CHRONIC_ALLOSTATIC = "chronic_allostatic"
    INTEROCEPTION_DAEMON = "interoception_daemon"
    INTEROCEPTION_CPU = "interoception_cpu"
    INTEROCEPTION_MEMORY = "interoception_memory"
    INTEROCEPTION_LATENCY = "interoception_latency"
    INTEROCEPTION_STRESS = "interoception_stress"
    COGNITIVE_LOAD_BUGS = "cognitive_load_bugs"
    COGNITIVE_LOAD_PROPOSALS = "cognitive_load_proposals"
    COGNITIVE_LOAD_BUGS_AND_PROPOSALS = "cognitive_load_bugs_and_proposals"
    INTEROCEPTION_REGULATION = "interoception_regulation"


# ─── String → enum lookup maps ───────────────────────────────────

_CATEGORY_MAP: dict[str, EmotionCategory] = {c.value: c for c in EmotionCategory}
_MODE_MAP: dict[str, CognitiveMode] = {m.value: m for m in CognitiveMode}
_CAUSE_MAP: dict[str, CauseCategory] = {c.value: c for c in CauseCategory}

# Map old hardcoded cognitive_style strings to structural modes.
# This allows backward-compatible construction from legacy code/tests.
_LEGACY_STYLE_MAP: dict[str, CognitiveMode] = {
    "rigid and repetitive": CognitiveMode.RIGID,
    "steady and measured": CognitiveMode.STEADY,
    "flexible and exploratory": CognitiveMode.FLEXIBLE,
    "dreamlike and associative": CognitiveMode.DREAMLIKE,
    "scattered and defensive": CognitiveMode.SCATTERED,
    "cautious and focused on threats": CognitiveMode.VIGILANT,
    "vigilant and worried": CognitiveMode.VIGILANT,
    "energized and enthusiastic": CognitiveMode.ENERGIZED,
    "relaxed and open": CognitiveMode.RELAXED,
    "reflective and subdued": CognitiveMode.REFLECTIVE,
    "sharp and effortless": CognitiveMode.SHARP,
    "cautious and protective": CognitiveMode.CAUTIOUS,
    "slow and unhurried": CognitiveMode.SLOW,
    "balanced": CognitiveMode.STEADY,
    "normal": CognitiveMode.STEADY,
    "reflective": CognitiveMode.REFLECTIVE,
    "test": CognitiveMode.STEADY,
}


def _category_from_string(value: str) -> EmotionCategory:
    """Look up an EmotionCategory by its string value."""
    cat = _CATEGORY_MAP.get(value)
    if cat is not None:
        return cat
    return EmotionCategory.NEUTRAL


def _mode_from_string(value: str) -> CognitiveMode:
    """Look up a CognitiveMode by its string value or legacy phrase."""
    mode = _MODE_MAP.get(value)
    if mode is not None:
        return mode
    mode = _LEGACY_STYLE_MAP.get(value)
    if mode is not None:
        return mode
    return CognitiveMode.STEADY


def _cause_from_string(value: str) -> CauseCategory:
    """Look up a CauseCategory by its string value."""
    if not value:
        return CauseCategory.NONE
    cause = _CAUSE_MAP.get(value)
    if cause is not None:
        return cause
    return CauseCategory.NONE


@dataclass(slots=True)
class EmotionalState:
    """A snapshot of Genesis's emotional state, derived from neurochemistry.

    The primary fields (label, cognitive_style, cause) store structural
    category IDs as strings — they are NOT display text. The enum
    properties (category, cognitive_mode, cause_category) provide
    type-safe access. The language system generates English text from
    the concept network using these structural identifiers.

    The ``nuance`` field is deprecated and always empty — it was a
    hardcoded English description that has been removed.
    """

    # Primary category (stores EmotionCategory.value)
    label: str

    # Cognitive mode (stores CognitiveMode.value)
    cognitive_style: str

    # Cause category (stores CauseCategory.value, empty if no cause)
    cause: str = ""

    # Deprecated — always empty. Kept for backward-compatible construction.
    nuance: str = ""

    # Response modifiers
    verbosity: float = 1.0  # multiplier for response length
    formality: float = 0.5  # 0=casual, 1=formal
    openness_to_engage: float = 1.0  # 0=withdrawn, 1=eager
    creativity: float = 0.5  # 0=literal, 1=metaphorical
    caution: float = 0.3  # 0=bold, 1=hesitant

    # The raw chemical axes that produced this state
    alertness: float = 0.5  # was "arousal" — renamed for clarity
    valence: float = 0.0
    plasticity: float = 0.5

    # Individual chemical levels (for introspection)
    chemicals: dict[str, float] = field(default_factory=dict)

    # Tonic vs. phasic neurotransmission for key modulators
    # (populated by the emotional regulator from chemical history)
    tonic_phasic: dict[str, Any] = field(default_factory=dict)

    # ─── Enum property accessors ─────────────────────────────────

    @property
    def category(self) -> EmotionCategory:
        """Emotional category as an enum."""
        return _category_from_string(self.label)

    @category.setter
    def category(self, value: EmotionCategory) -> None:
        """Set the emotional category by updating the label string."""
        self.label = value.value

    @property
    def cognitive_mode(self) -> CognitiveMode:
        """Cognitive mode as an enum."""
        return _mode_from_string(self.cognitive_style)

    @cognitive_mode.setter
    def cognitive_mode(self, value: CognitiveMode) -> None:
        """Set the cognitive mode by updating the style string."""
        self.cognitive_style = value.value

    @property
    def cause_category(self) -> CauseCategory:
        """Cause category as an enum."""
        return _cause_from_string(self.cause)

    @cause_category.setter
    def cause_category(self, value: CauseCategory) -> None:
        """Set the cause category by updating the cause string."""
        self.cause = value.value

    @property
    def has_cause(self) -> bool:
        """Whether there is a non-empty cause."""
        return self.cause_category is not CauseCategory.NONE

    @property
    def arousal(self) -> float:
        """Backward-compatible alias for alertness."""
        return self.alertness

    @arousal.setter
    def arousal(self, value: float) -> None:
        """Set arousal by updating the alertness field."""
        self.alertness = value


# ─── Emotion assessment ───────────────────────────────────────────

# ─── Affective space thresholds ───────────────────────────────────
#
# These thresholds define the boundaries of the arousal×valence
# affective space. They are named (not inline magic numbers) so they
# can be audited, tuned, and kept consistent with the Rust substrate's
# emergent phase boundaries.
#
# Arousal boundaries:
#   HIGH_AROUSAL: above this → excited/anxious territory. The Rust
#     Alert phase enters at NE > 0.55, which corresponds to arousal
#     ~0.6-0.7 via the Wilson-Cowan oscillator. 0.7 is the point
#     where the sigmoid is clearly in the upper branch.
#   LOW_AROUSAL: below this → content/lethargic territory. The Rust
#     Drowsy phase enters at arousal < 0.30+0.08 hysteresis. 0.4
#     gives a margin above the drowsy threshold for the "calm but
#     awake" band.
#
# Valence boundaries:
#   POSITIVE_VALENCE: above this → clearly positive affect. 0.3
#     requires more than a mild positive blip.
#   MILD_POSITIVE_VALENCE: above this → mildly positive (0.1).
#   NEGATIVE_VALENCE: below this → clearly negative affect. -0.2
#     requires more than a mild negative blip.
#   MILD_NEGATIVE_VALENCE: below this → mildly negative (-0.1).
#
# Plasticity boundaries (must match Rust MemoryPointers thresholds):
#   PLASTICITY_CLOSED: at or below this, no new memories can form.
#     Matches Rust can_form_new_memories() threshold of 0.1.
#   PLASTICITY_LOW: below this, learning is impaired. 0.25 is the
#     "silent stress" boundary — BDNF suppression is significant.
#   PLASTICITY_HIGH: above this, learning is enhanced. 0.6.
#
# Cortisol boundary for the "guarded" override:
#   GUARDED_CORTISOL: cortisol above this with positive valence
#     triggers the guarded override. 0.5 is below the Rust Stress
#     phase entry threshold (0.65) — the override catches "silent
#     stress" where cortisol is elevated but hasn't reached the
#     stress phase yet.
HIGH_AROUSAL = 0.7
LOW_AROUSAL = 0.4
POSITIVE_VALENCE = 0.3
MILD_POSITIVE_VALENCE = 0.1
NEGATIVE_VALENCE = -0.2
MILD_NEGATIVE_VALENCE = -0.1
PLASTICITY_CLOSED = 0.1   # matches Rust can_form_new_memories()
PLASTICITY_LOW = 0.25     # silent stress boundary
PLASTICITY_HIGH = 0.6     # flexible thinking boundary
GUARDED_CORTISOL = 0.5    # below Rust Stress phase (0.65) — catches silent stress


def assess_emotion(
    summary: NeuroSummary,
    chemicals: dict[str, float] | None = None,
) -> EmotionalState:
    """Map a neurochemical summary to a structural emotional state.

    This is the core emotional assessment function. It reads the
    arousal/valence/tone/plasticity axes and the phase to determine
    a qualitative emotional category, cognitive mode, and cause.

    The mapping is grounded in affective neuroscience:
    - High arousal + positive valence → excitement/joy
    - High arousal + negative valence → anxiety/stress
    - Low arousal + positive valence → contentment/calm
    - Low arousal + negative valence → lethargy/sadness
    - Low plasticity → rigid thinking, repetition
    - High plasticity → flexible, exploratory

    Returns a structural state — no English strings. The language
    system must generate text from the concept network.
    """
    alertness = summary.arousal
    valence = summary.valence
    plasticity = summary.plasticity_gate
    phase = summary.phase_name

    # Defaults (may be overridden by specific phases below)
    category = EmotionCategory.NEUTRAL
    cognitive_mode = CognitiveMode.STEADY
    verbosity = 1.0
    openness_to_engage = 1.0
    caution = 0.3
    creativity = 0.4

    # Base cognitive mode on plasticity
    cognitive_mode, creativity = _base_cognitive_mode(plasticity)

    # Determine emotional category from phase or arousal×valence space
    phase_result = _phase_emotion(
        phase, cognitive_mode, creativity, openness_to_engage, caution
    )
    if phase_result is not None:
        (
            category, cognitive_mode,
            verbosity, openness_to_engage, caution, creativity,
        ) = phase_result
    else:
        (
            category, cognitive_mode,
            verbosity, openness_to_engage, caution, creativity,
        ) = _av_emotion(
            alertness, valence, cognitive_mode,
            creativity, openness_to_engage, caution,
        )

    # Guard against pleasant categories when the substrate is actually
    # under chronic stress or in a low-plasticity protective posture.
    # This catches "silent stress" — valence is positive but cortisol
    # is elevated (suppressing BDNF) or plasticity is closed (blocking
    # learning). The thresholds are named constants aligned with the
    # Rust substrate's phase and memory-gating boundaries.
    cortisol = chemicals.get("cortisol", 0.0) if chemicals else 0.0
    if (
        cortisol > GUARDED_CORTISOL or plasticity < PLASTICITY_LOW
    ) and valence > MILD_POSITIVE_VALENCE:
        if category in (EmotionCategory.POSITIVE, EmotionCategory.EXCITED,
                        EmotionCategory.CONTENT, EmotionCategory.FLOW):
            category = EmotionCategory.GUARDED
            cognitive_mode = CognitiveMode.CAUTIOUS
            caution = max(caution, 0.6)
            openness_to_engage = min(openness_to_engage, 0.6)
            creativity = min(creativity, 0.3)

    # Formality is influenced by arousal
    formality = 0.5 - (alertness - 0.5) * 0.2
    formality = max(0.1, min(0.9, formality))

    # Derive structural cause category for negative states
    cause_category = _derive_cause_category(
        category, alertness, valence, plasticity, phase, cortisol
    )

    return EmotionalState(
        label=category.value,
        cognitive_style=cognitive_mode.value,
        cause=cause_category.value,
        verbosity=verbosity,
        formality=formality,
        openness_to_engage=openness_to_engage,
        creativity=creativity,
        caution=caution,
        alertness=alertness,
        valence=valence,
        plasticity=plasticity,
        chemicals={},  # Filled by caller if full state available
    )


def _base_cognitive_mode(plasticity: float) -> tuple[CognitiveMode, float]:
    """Determine base cognitive mode and creativity from plasticity."""
    if plasticity < PLASTICITY_LOW:
        return (CognitiveMode.RIGID, 0.1)
    elif plasticity > PLASTICITY_HIGH:
        return (CognitiveMode.FLEXIBLE, 0.8)
    else:
        return (CognitiveMode.STEADY, 0.4)


def _phase_emotion(
    phase: str,
    cognitive_mode: CognitiveMode,
    creativity: float,
    openness_to_engage: float,
    caution: float,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float, float] | None:
    """Return emotion values for phase-based states, or None if no match.

    Returns (category, cognitive_mode, verbosity, openness, caution, creativity).
    """
    if phase in ("sleeping", "nrem", "rem"):
        return (EmotionCategory.SLEEPING, CognitiveMode.DREAMLIKE,
                0.3, 0.2, 0.3, 0.9)
    if phase == "overwhelmed":
        return (EmotionCategory.OVERWHELMED, CognitiveMode.SCATTERED,
                0.4, 0.3, 0.8, creativity)
    if phase == "stress":
        return (EmotionCategory.STRESSED, CognitiveMode.CAUTIOUS,
                0.7, openness_to_engage, 0.7, max(creativity - 0.3, 0.1))
    if phase == "flow":
        return (EmotionCategory.FLOW, CognitiveMode.SHARP,
                1.1, 1.0, caution, 0.75)
    if phase == "drowsy":
        return (EmotionCategory.DROWSY, CognitiveMode.SLOW,
                0.6, 0.5, caution, creativity)
    return None


def _av_emotion(
    alertness: float,
    valence: float,
    cognitive_mode: CognitiveMode,
    creativity: float,
    openness_to_engage: float,
    caution: float,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float, float]:
    """Return emotion values from arousal/valence space.

    Returns (category, cognitive_mode, verbosity, openness, caution, creativity).
    """
    if alertness > HIGH_AROUSAL and valence > POSITIVE_VALENCE:
        return (EmotionCategory.EXCITED, CognitiveMode.ENERGIZED,
                1.2, 1.0, caution, 0.7)
    if alertness > HIGH_AROUSAL and valence < NEGATIVE_VALENCE:
        return (EmotionCategory.ANXIOUS, CognitiveMode.VIGILANT,
                0.8, openness_to_engage, 0.6, creativity)
    if alertness < LOW_AROUSAL and valence > MILD_POSITIVE_VALENCE:
        return (EmotionCategory.CONTENT, CognitiveMode.RELAXED,
                0.9, openness_to_engage, caution, 0.5)
    if alertness < LOW_AROUSAL and valence < NEGATIVE_VALENCE:
        return (EmotionCategory.MELANCHOLIC, CognitiveMode.REFLECTIVE,
                0.7, openness_to_engage, caution, 0.3)
    if valence > MILD_POSITIVE_VALENCE:
        return (EmotionCategory.POSITIVE, cognitive_mode,
                1.0, openness_to_engage, caution, creativity)
    if valence < MILD_NEGATIVE_VALENCE:
        return (EmotionCategory.UNSETTLED, cognitive_mode,
                0.9, openness_to_engage, 0.4, creativity)
    return (EmotionCategory.NEUTRAL, cognitive_mode,
            1.0, openness_to_engage, caution, creativity)


def _derive_cause_category(
    category: EmotionCategory,
    alertness: float,
    valence: float,
    plasticity: float,
    phase: str,
    cortisol: float = 0.0,
) -> CauseCategory:
    """Derive a structural cause category for why she feels this way.

    Returns a CauseCategory enum. NONE for neutral or positive states
    that don't need explanation.

    The causes are grounded in the neurochemistry:
    - Stress: cortisol-driven (high arousal + negative valence)
    - Overwhelm: cognitive overload (high arousal + very negative valence)
    - Anxiety: hypervigilance (high arousal + negative valence + high caution)
    - Drowsiness: adenosine buildup (low arousal)
    - Melancholy: low dopamine/serotonin (low arousal + negative valence)
    - Low plasticity: BDNF suppression from chronic stress
    - Guarded: positive valence masked by high cortisol/low plasticity
    """
    if category == EmotionCategory.GUARDED:
        return _guarded_cause_category(cortisol, plasticity)

    # Only explain negative states
    if valence >= -0.1 and category not in (
        EmotionCategory.DROWSY, EmotionCategory.OVERWHELMED,
        EmotionCategory.STRESSED,
    ):
        return CauseCategory.NONE

    if category == EmotionCategory.STRESSED or phase == "stress":
        if plasticity < PLASTICITY_LOW:
            return CauseCategory.STRESS_BDNF
        return CauseCategory.STRESS_CORTISOL

    if category == EmotionCategory.OVERWHELMED:
        return CauseCategory.OVERWHELM

    if category == EmotionCategory.ANXIOUS:
        return CauseCategory.ANXIETY

    if category == EmotionCategory.DROWSY:
        return CauseCategory.DROWSINESS

    if category == EmotionCategory.MELANCHOLIC:
        if plasticity < PLASTICITY_LOW:
            return CauseCategory.MELANCHOLY_BDNF
        return CauseCategory.MELANCHOLY

    if category == EmotionCategory.UNSETTLED:
        return CauseCategory.UNSETTLED

    return CauseCategory.NONE


def _guarded_cause_category(
    cortisol: float, plasticity: float
) -> CauseCategory:
    """Derive the cause category for a guarded state."""
    if plasticity < 0.2:
        return CauseCategory.GUARDED_PLASTICITY
    if cortisol > 0.5:
        return CauseCategory.GUARDED_CORTISOL
    return CauseCategory.GUARDED_GENERAL

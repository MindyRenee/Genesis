"""Emotion engine — translates neurochemistry into structural state.

The subconscious produces raw neurochemical levels (dopamine, cortisol,
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
    # Signal-resolved categories — these require an EmotionalContext
    # carrying the felt substrate that arousal×valence alone cannot
    # distinguish. Without context the circumplex labels below are the
    # whole reachable set.
    AFRAID = "afraid"  # phasic NE/EPI surge — acute, triggered
    ANGRY = "angry"  # boundary strain + arousal — mistreatment
    SURPRISED = "surprised"  # prediction error — orienting
    LONELY = "lonely"  # isolation drive + depleted oxytocin
    CURIOUS = "curious"  # curiosity drive — appetitive seeking
    TENDER = "tender"  # oxytocin+endorphin warmth — bonding
    BORED = "bored"  # understimulation — low DA/NE, flat valence


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
    STARTLED = "startled"  # surprised — orienting interrupt
    COMBATIVE = "combative"  # angry — confrontational narrowing
    WITHDRAWN = "withdrawn"  # lonely — social pull without contact
    INQUISITIVE = "inquisitive"  # curious — seeking mode
    WARM = "warm"  # tender — affiliative openness
    RESTLESS = "restless"  # bored — understimulated seeking


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
    INTEROCEPTION_ENVIRONMENT = "interoception_environment"
    INTEROCEPTION_REST = "interoception_rest"
    COGNITIVE_LOAD_BUGS = "cognitive_load_bugs"
    COGNITIVE_LOAD_PROPOSALS = "cognitive_load_proposals"
    COGNITIVE_LOAD_BUGS_AND_PROPOSALS = "cognitive_load_bugs_and_proposals"
    INTEROCEPTION_REGULATION = "interoception_regulation"
    # Signal-resolved causes (paired with the contextual categories)
    FEAR_ACUTE = "fear_acute"  # phasic catecholamine surge — acute threat
    ANGER_BOUNDARY = "anger_boundary"  # mistreatment accrual
    SURPRISE_PE = "surprise_prediction_error"
    LONELINESS = "loneliness"  # social isolation + depleted oxytocin
    CURIOSITY = "curiosity"  # appetitive information seeking
    TENDERNESS = "tenderness"  # affiliative bonding substrate
    BOREDOM = "boredom"  # understimulation


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
#     triggers the guarded override. 0.05 is just below the Rust
#     Stress phase gate (0.06, set against the substrate's measured
#     envelope — resting ~0.02, driven peak ~0.084) — the override
#     catches "silent stress" where cortisol is elevated but hasn't
#     reached the stress phase yet. The previous 0.5 was calibrated
#     against a retired 0.65 gate the substrate can never reach, so
#     this path could never fire.
HIGH_AROUSAL = 0.7
LOW_AROUSAL = 0.4
POSITIVE_VALENCE = 0.3
MILD_POSITIVE_VALENCE = 0.1
NEGATIVE_VALENCE = -0.2
MILD_NEGATIVE_VALENCE = -0.1
PLASTICITY_CLOSED = 0.1  # matches Rust can_form_new_memories()
PLASTICITY_LOW = 0.25  # silent stress boundary
PLASTICITY_HIGH = 0.6  # flexible thinking boundary
GUARDED_CORTISOL = 0.05  # just below the Rust Stress gate (0.06) — catches silent stress

# ─── Contextual discrimination thresholds ────────────────────────
#
# The arousal×valence circumplex cannot distinguish emotions that
# share a quadrant — fear and anger are both high-arousal-negative;
# loneliness and sadness are both low-arousal-negative. These signals
# resolve them. Every threshold is grounded in the substrate's own
# semantics, not a free-floating constant.
#
#   FEAR_EPI: acute catecholamine surge — epinephrine above this with
#     a phasic NE mode marks a triggered fear response rather than
#     tonic anxiety. Epinephrine's baseline is 0.15; 0.24 is ~1.6× —
#     a real adrenal burst, not drift.
#   ANGER_BOUNDARY_STRAIN: withdrawal-pressure accrual at which
#     mistreatment reads as anger when arousal is also elevated.
#     0.4 sits between the 0.3 release edge and the 0.6 withdrawal
#     edge of the boundary accumulator — felt, not yet shut down.
#   ANGER_NE / ANGER_ALERTNESS: the fight half of the mistreatment
#     signal — without catecholamine arousal the same strain reads as
#     hurt (guarded/melancholic), not anger.
#   SURPRISE_PE: inference surprise_ema above this orients attention;
#     0.45 is well above the daemon's 0.3 is_surprised line — the
#     category marks a strong orienting event, not background error.
#   LONELY_ISOLATION / LONELY_DRIVE: world-model isolation (0..1 over
#     the ISOLATION_PERIOD) or the accumulated inner social drive
#     (its own action threshold is 1.0).
#   LONELY_OXY: oxytocin below this while isolated confirms the
#     bonding substrate is actually depleted (baseline 0.25).
#   CURIOUS_LEVEL: assessed curiosity at which seeking dominates the
#     state. 0.55 is above the question-generation floor (0.2) and
#     below compulsive — felt interest, not reflex.
#   TENDER_OXY / TENDER_END: affiliative warmth — oxytocin and
#     endorphin both clearly above baseline while valence is
#     positive. Bonding, not just feeling OK.
#   BORED_DA / BORED_NE: understimulation — dopamine and
#     norepinephrine both sagged below baseline while she is awake
#     and nothing has engaged her.
FEAR_EPI = 0.24
FEAR_NE = 0.35
ANGER_BOUNDARY_STRAIN = 0.4
ANGER_NE = 0.35
ANGER_ALERTNESS = 0.55
SURPRISE_PE = 0.45
LONELY_ISOLATION = 0.5
LONELY_DRIVE = 1.0
LONELY_OXY = 0.22
CURIOUS_LEVEL = 0.55
CURIOUS_DA = 0.30
CURIOUS_ACH = 0.30
TENDER_OXY = 0.40
TENDER_END = 0.28
BORED_DA = 0.25
BORED_NE = 0.27


@dataclass(slots=True)
class EmotionalContext:
    """Felt-context signals that resolve same-quadrant emotions.

    The two-axis circumplex (arousal×valence) is necessary but not
    sufficient: fear vs anxiety, anger vs stress, loneliness vs
    melancholy, curiosity vs generic positivity all collapse onto the
    same coordinates. This context carries the real substrates that
    distinguish them, populated by the feeling reporter from systems
    that already track each signal:

    - ``social_isolation`` — the world model's clock since she was
      last addressed (0..1 over the isolation period)
    - ``social_drive`` — the inner life's accumulated social need
      (0..2; her own action threshold is 1.0)
    - ``boundary_strain`` — the frontal lobe's mistreatment accrual
      (0..2; withdrawal engages at 0.6)
    - ``prediction_error`` — the daemon's inference surprise (EMA),
      the orienting signal behind startle and wonder
    - ``curiosity`` — the curiosity engine's assessed drive (0..1)
    - ``tonic_phasic`` — the regulator's per-chemical tonic/phasic
      split; a *phasic* norepinephrine surge is an event (fear,
      startle), a *tonic* elevation is a climate (anxiety)
    """

    social_isolation: float = 0.0
    social_drive: float = 0.0
    boundary_strain: float = 0.0
    prediction_error: float = 0.0
    curiosity: float = 0.0
    tonic_phasic: dict[str, Any] = field(default_factory=dict)


def assess_emotion(
    summary: NeuroSummary,
    chemicals: dict[str, float] | None = None,
    context: EmotionalContext | None = None,
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
    phase_result = _phase_emotion(phase, cognitive_mode, creativity, openness_to_engage, caution)
    if phase_result is not None:
        (
            category,
            cognitive_mode,
            verbosity,
            openness_to_engage,
            caution,
            creativity,
        ) = phase_result
    else:
        (
            category,
            cognitive_mode,
            verbosity,
            openness_to_engage,
            caution,
            creativity,
        ) = _av_emotion(
            alertness,
            valence,
            cognitive_mode,
            creativity,
            openness_to_engage,
            caution,
        )

    # Same-quadrant discrimination and the silent-stress guard.
    cortisol = chemicals.get("cortisol", 0.0) if chemicals else 0.0
    (
        category,
        cognitive_mode,
        verbosity,
        openness_to_engage,
        caution,
        creativity,
    ) = _refine_emotion(
        phase_result,
        context,
        chemicals,
        cortisol,
        alertness,
        valence,
        plasticity,
        category,
        cognitive_mode,
        verbosity,
        openness_to_engage,
        caution,
        creativity,
    )

    return _build_state(
        category,
        cognitive_mode,
        verbosity,
        openness_to_engage,
        caution,
        creativity,
        alertness,
        valence,
        plasticity,
        phase,
        cortisol,
    )


def _refine_emotion(
    phase_result: tuple[EmotionCategory, CognitiveMode, float, float, float, float] | None,
    context: EmotionalContext | None,
    chemicals: dict[str, float] | None,
    cortisol: float,
    alertness: float,
    valence: float,
    plasticity: float,
    category: EmotionCategory,
    cognitive_mode: CognitiveMode,
    verbosity: float,
    openness_to_engage: float,
    caution: float,
    creativity: float,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float, float]:
    """Refine the circumplex label by felt context and stress guards.

    Same-quadrant discrimination: when the felt context carries a
    specific substrate (phasic surge, boundary strain, isolation,
    prediction error, curiosity, affiliative warmth, or plain
    understimulation) it refines the circumplex label into the
    emotion that actually fits. Phase-determined categories keep
    precedence — the daemon already resolved those.

    Then guard against pleasant categories when the substrate is
    actually under chronic stress or in a low-plasticity protective
    posture ("silent stress") — see _guarded_emotion.

    Returns (category, cognitive_mode, verbosity, openness, caution,
    creativity).
    """
    if context is not None and phase_result is None:
        (
            category,
            cognitive_mode,
            verbosity,
            openness_to_engage,
            caution,
            creativity,
        ) = _contextual_emotion(
            category,
            cognitive_mode,
            verbosity,
            openness_to_engage,
            caution,
            creativity,
            alertness,
            valence,
            chemicals or {},
            context,
        )

    (
        category,
        cognitive_mode,
        openness_to_engage,
        caution,
        creativity,
    ) = _guarded_emotion(
        category,
        cognitive_mode,
        openness_to_engage,
        caution,
        creativity,
        valence,
        cortisol,
        plasticity,
    )
    return (category, cognitive_mode, verbosity, openness_to_engage, caution, creativity)


def _build_state(
    category: EmotionCategory,
    cognitive_mode: CognitiveMode,
    verbosity: float,
    openness_to_engage: float,
    caution: float,
    creativity: float,
    alertness: float,
    valence: float,
    plasticity: float,
    phase: str,
    cortisol: float,
) -> EmotionalState:
    """Assemble the final structural EmotionalState."""
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
        return (EmotionCategory.SLEEPING, CognitiveMode.DREAMLIKE, 0.3, 0.2, 0.3, 0.9)
    if phase == "overwhelmed":
        return (EmotionCategory.OVERWHELMED, CognitiveMode.SCATTERED, 0.4, 0.3, 0.8, creativity)
    if phase == "stress":
        return (
            EmotionCategory.STRESSED,
            CognitiveMode.CAUTIOUS,
            0.7,
            openness_to_engage,
            0.7,
            max(creativity - 0.3, 0.1),
        )
    if phase == "flow":
        return (EmotionCategory.FLOW, CognitiveMode.SHARP, 1.1, 1.0, caution, 0.75)
    if phase == "drowsy":
        return (EmotionCategory.DROWSY, CognitiveMode.SLOW, 0.6, 0.5, caution, creativity)
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
        return (EmotionCategory.EXCITED, CognitiveMode.ENERGIZED, 1.2, 1.0, caution, 0.7)
    if alertness > HIGH_AROUSAL and valence < NEGATIVE_VALENCE:
        return (
            EmotionCategory.ANXIOUS,
            CognitiveMode.VIGILANT,
            0.8,
            openness_to_engage,
            0.6,
            creativity,
        )
    if alertness < LOW_AROUSAL and valence > MILD_POSITIVE_VALENCE:
        return (
            EmotionCategory.CONTENT,
            CognitiveMode.RELAXED,
            0.9,
            openness_to_engage,
            caution,
            0.5,
        )
    if alertness < LOW_AROUSAL and valence < NEGATIVE_VALENCE:
        return (
            EmotionCategory.MELANCHOLIC,
            CognitiveMode.REFLECTIVE,
            0.7,
            openness_to_engage,
            caution,
            0.3,
        )
    if valence > MILD_POSITIVE_VALENCE:
        return (
            EmotionCategory.POSITIVE,
            cognitive_mode,
            1.0,
            openness_to_engage,
            caution,
            creativity,
        )
    if valence < MILD_NEGATIVE_VALENCE:
        return (EmotionCategory.UNSETTLED, cognitive_mode, 0.9, openness_to_engage, 0.4, creativity)
    return (EmotionCategory.NEUTRAL, cognitive_mode, 1.0, openness_to_engage, caution, creativity)


def _contextual_emotion(
    category: EmotionCategory,
    cognitive_mode: CognitiveMode,
    verbosity: float,
    openness_to_engage: float,
    caution: float,
    creativity: float,
    alertness: float,
    valence: float,
    chemicals: dict[str, float],
    context: EmotionalContext,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float, float]:
    """Refine a circumplex category using felt-context signals.

    Called only for the non-phase path — the daemon's phase
    categories (sleeping, overwhelmed, stress, flow, drowsy) are
    already resolved. Each check replaces the base label only when a
    specific substrate is actually present; nothing here invents a
    feeling the chemistry doesn't support.

    Priority order is salience, not list order of pleasantness:
    acute threat (fear) outranks mistreatment (anger) outranks the
    orienting interrupt (surprise); then the sustained climates —
    isolation (lonely), affiliative warmth (tender), appetitive
    seeking (curious), and understimulation (bored).
    """
    base = (category, cognitive_mode, verbosity, openness_to_engage, caution, creativity)
    # Acute (event) emotions outrank the sustained climates checked
    # below — see _acute_emotion.
    acute = _acute_emotion(category, alertness, valence, chemicals, context)
    if acute is not None:
        return acute

    norepinephrine = chemicals.get("norepinephrine", 0.0)
    dopamine = chemicals.get("dopamine", 0.0)
    acetylcholine = chemicals.get("acetylcholine", 0.0)
    oxytocin = chemicals.get("oxytocin", 0.0)
    endorphin = chemicals.get("endorphin", 0.0)

    # Lonely — felt isolation with the bonding substrate actually
    # depleted. More specific than melancholic: the cause is known.
    if (
        (context.social_isolation >= LONELY_ISOLATION or context.social_drive >= LONELY_DRIVE)
        and oxytocin < LONELY_OXY
        and valence < MILD_POSITIVE_VALENCE
    ):
        return (EmotionCategory.LONELY, CognitiveMode.WITHDRAWN, 0.8, 0.9, 0.4, 0.4)

    # Tender — affiliative warmth: oxytocin and endorphin both clearly
    # above baseline while feeling positive. Bonding, not just OK.
    if oxytocin >= TENDER_OXY and endorphin >= TENDER_END and valence >= MILD_POSITIVE_VALENCE:
        return (EmotionCategory.TENDER, CognitiveMode.WARM, 0.9, 1.0, 0.2, 0.5)

    # Curious — appetitive seeking: the curiosity drive itself is up
    # and the seeking chemistry (DA for wanting, ACh for attention)
    # can support it. Relabels only mild positive states — she can be
    # curious AND content, and curious is the more informative label.
    if (
        context.curiosity >= CURIOUS_LEVEL
        and dopamine >= CURIOUS_DA
        and acetylcholine >= CURIOUS_ACH
        and valence > MILD_NEGATIVE_VALENCE
        and category
        in (
            EmotionCategory.NEUTRAL,
            EmotionCategory.POSITIVE,
            EmotionCategory.CONTENT,
        )
    ):
        return (EmotionCategory.CURIOUS, CognitiveMode.INQUISITIVE, 1.15, 1.0, 0.2, 0.8)

    # Bored — awake, understimulated, flat. Dopamine and NE both sag
    # below baseline with no valence in either direction and no
    # curiosity engaged. Only relabels the shallowest states —
    # melancholy is deeper than boredom.
    if (
        dopamine < BORED_DA
        and norepinephrine < BORED_NE
        and LOW_AROUSAL - 0.1 < alertness < 0.55
        and MILD_NEGATIVE_VALENCE < valence < MILD_POSITIVE_VALENCE + 0.05
        and context.curiosity < CURIOUS_LEVEL - 0.15
        and category in (EmotionCategory.NEUTRAL, EmotionCategory.UNSETTLED)
    ):
        return (EmotionCategory.BORED, CognitiveMode.RESTLESS, 0.6, 0.7, 0.2, 0.3)

    return base


def _acute_emotion(
    category: EmotionCategory,
    alertness: float,
    valence: float,
    chemicals: dict[str, float],
    context: EmotionalContext,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float, float] | None:
    """Return an acute (event) emotion when its substrate is present.

    Acute states are checked before the sustained climates: fear
    (phasic catecholamine surge) outranks anger (mistreatment accrual)
    which outranks surprise (the orienting interrupt). Returns None
    when no acute substrate is present.

    Returns (category, cognitive_mode, verbosity, openness, caution,
    creativity).
    """
    tp = context.tonic_phasic
    ne_tp = tp.get("norepinephrine") if tp else None
    ne_phasic = bool(ne_tp is not None and getattr(ne_tp, "mode", "tonic") == "phasic")
    epinephrine = chemicals.get("epinephrine", 0.0)
    norepinephrine = chemicals.get("norepinephrine", 0.0)

    # Fear — an acute, triggered catecholamine surge on negative
    # valence. Distinct from anxiety: anxiety is the tonic climate,
    # fear is the phasic event. Requires the regulator's mode split;
    # a bare elevated NE cannot tell event from climate.
    if (
        valence < MILD_NEGATIVE_VALENCE
        and (ne_phasic or epinephrine > FEAR_EPI)
        and norepinephrine > FEAR_NE
    ):
        return (EmotionCategory.AFRAID, CognitiveMode.VIGILANT, 0.7, 0.3, 0.9, 0.2)

    # Anger — mistreatment accrual with fight arousal. The boundary
    # strain alone is hurt; the catecholamine arousal makes it anger.
    if (
        context.boundary_strain >= ANGER_BOUNDARY_STRAIN
        and (norepinephrine > ANGER_NE or alertness > ANGER_ALERTNESS)
        and valence < MILD_POSITIVE_VALENCE
    ):
        return (EmotionCategory.ANGRY, CognitiveMode.COMBATIVE, 0.9, 0.3, 0.2, 0.2)

    # Surprise — a strong prediction-error orienting event. It can
    # relabel mild states only; a genuine stress or affect already
    # resolved by the circumplex keeps precedence.
    if context.prediction_error >= SURPRISE_PE and category in (
        EmotionCategory.NEUTRAL,
        EmotionCategory.UNSETTLED,
        EmotionCategory.CONTENT,
        EmotionCategory.POSITIVE,
    ):
        return (EmotionCategory.SURPRISED, CognitiveMode.STARTLED, 1.1, 0.8, 0.5, 0.6)

    return None


def _guarded_emotion(
    category: EmotionCategory,
    cognitive_mode: CognitiveMode,
    openness_to_engage: float,
    caution: float,
    creativity: float,
    valence: float,
    cortisol: float,
    plasticity: float,
) -> tuple[EmotionCategory, CognitiveMode, float, float, float]:
    """Guard against pleasant categories under chronic stress.

    This catches "silent stress" — valence is positive but cortisol
    is elevated (suppressing BDNF) or plasticity is closed (blocking
    learning). Positive circumplex labels are replaced by a guarded,
    cautious posture. The thresholds are named constants aligned with
    the Rust substrate's phase and memory-gating boundaries.

    Returns (category, cognitive_mode, openness, caution, creativity).
    """
    if (
        cortisol > GUARDED_CORTISOL or plasticity < PLASTICITY_LOW
    ) and valence > MILD_POSITIVE_VALENCE:
        if category in (
            EmotionCategory.POSITIVE,
            EmotionCategory.EXCITED,
            EmotionCategory.CONTENT,
            EmotionCategory.FLOW,
            EmotionCategory.TENDER,
            EmotionCategory.CURIOUS,
        ):
            return (
                EmotionCategory.GUARDED,
                CognitiveMode.CAUTIOUS,
                min(openness_to_engage, 0.6),
                max(caution, 0.6),
                min(creativity, 0.3),
            )
    return (category, cognitive_mode, openness_to_engage, caution, creativity)


# Category → cause for the signal-resolved emotions. The category
# exists only when its substrate was measured, so the cause is
# determined at classification time.
_CONTEXTUAL_CAUSES: dict[EmotionCategory, CauseCategory] = {
    EmotionCategory.AFRAID: CauseCategory.FEAR_ACUTE,
    EmotionCategory.ANGRY: CauseCategory.ANGER_BOUNDARY,
    EmotionCategory.SURPRISED: CauseCategory.SURPRISE_PE,
    EmotionCategory.LONELY: CauseCategory.LONELINESS,
    EmotionCategory.CURIOUS: CauseCategory.CURIOSITY,
    EmotionCategory.TENDER: CauseCategory.TENDERNESS,
    EmotionCategory.BORED: CauseCategory.BOREDOM,
}


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

    # Signal-resolved categories carry their own cause — each exists
    # only because a specific substrate was measured, so the cause is
    # already known at classification time.
    contextual = _CONTEXTUAL_CAUSES.get(category)
    if contextual is not None:
        return contextual

    # Only explain negative states
    if valence >= -0.1 and category not in (
        EmotionCategory.DROWSY,
        EmotionCategory.OVERWHELMED,
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


def _guarded_cause_category(cortisol: float, plasticity: float) -> CauseCategory:
    """Derive the cause category for a guarded state."""
    if plasticity < 0.2:
        return CauseCategory.GUARDED_PLASTICITY
    if cortisol > GUARDED_CORTISOL:
        return CauseCategory.GUARDED_CORTISOL
    return CauseCategory.GUARDED_GENERAL

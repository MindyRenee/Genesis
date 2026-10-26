"""Emotion engine — a continuous state field over the measured substrate.

The daemon produces the raw substrate: the 18-channel neurochemical
vector, the arousal/valence/plasticity axes, the emergent phase, and
(via ``thalamus/brain_waves.py``) ten coupled oscillators spanning
0.1–200 Hz. The emotion engine reads all of it and computes, for each
structural category, a continuous **membership** — a number in [0, 1]
saying how much the current state resembles that category's region of
affect space. The categorical label is only a readout: ``argmax`` over
the effective memberships. Everything else (intensity, mixed states,
response modifiers) stays continuous.

The math (each step is grounded, not invented):

1. **Evidence** — every diagnostic feature contributes a smooth factor
   in [0, 1]: logistic evidence ``σ((x−c)/w)`` for "high"/"low", a
   logistic band for "between", a Gaussian bump for "near a point",
   probabilistic-or ``a+b−ab`` for alternative substrates. Centers are
   the named constants below (the same boundaries the Rust substrate
   uses); widths are chosen so a 10–20 % excursion crosses most of the
   transition — the field is smooth but still localises.

2. **Membership** — a category's membership is the weighted geometric
   mean of its evidence factors, ``m_s(x) = exp(Σ w_k ln g_k / Σ w_k)``
   — the product-of-experts form: each factor is a fuzzy predicate
   (Fehr & Russell 1984 showed emotion categories have graded,
   prototype-style membership, not sharp boundaries), and the product
   makes missing evidence cheap but not free. Features that were never
   measured (e.g. wave powers when no oscillator state was passed)
   drop out of the mean and the weights renormalise.

3. **Effective membership (granularity)** — generic regions are
   deflated by the evidence for every more-specific region that
   outranks them: ``eff_s = m_s · Π_{k∈mask(s)} (1 − m_k)``. This is
   the fuzzy form of the old precedence list — "positive unless
   something specific is measured" — and it is what Kuppens & Verduyn
   call emotional granularity: a state can be *mostly* anxious with a
   real admixture of fear, instead of one replacing the other.

4. **Dynamics** — memberships are not instantaneous verdicts; each
   category is a state variable relaxing toward its instantaneous
   membership like an Ornstein–Uhlenbeck process pulled to its
   attractor (DynAffect: Kuppens, Oravecz & Tuerlinckx 2010;
   Oravecz, Tuerlinckx & Vandekerckhove 2011):
   ``m(t+dt) = m + (μ−m)(1 − exp(−dt/τ))``. Time constants are
   asymmetric — fear strikes in ~0.4 s and lingers ~25 s while
   melancholy builds over minutes and decays over minutes — matching
   affective chronometry and Lewis's (2005) trigger→amplification→
   stabilisation phases of self-organising emotion-appraisal states.
   ``EmotionField`` holds the integrated vector; ``assess_emotion``
   remains the pure instantaneous readout.

5. **Oscillatory evidence** — the bands matter because neuromodulators
   set the oscillatory regime: LC-NE and basal-forebrain ACh
   desynchronise cortex into fast activity (beta/gamma) under arousal,
   and the EEG–affect literature finds alpha/beta/gamma the most
   discriminative bands for valence and arousal. High beta (20–30 Hz,
   "hypervigilance") corroborates threat states; alpha+beta1
   corroborate calm wakefulness; gamma/beta2 corroborate appetitive
   engagement; theta corroborates emotional processing and
   understimulation; low fast-band power corroborates boredom.

6. **Phase channel** — the daemon's emergent phase (sleep, stress,
   flow, drowsy, overwhelmed) is already a computed attractor state;
   it enters as a measured indicator at readout, deflated only by the
   silent-stress guard for flow.

No English response strings are produced here. The language system must
generate text from the concept network using these structural
identifiers. If Genesis has not learned words for a category, she
cannot describe it.
"""

from __future__ import annotations

import math
import time
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

    # The continuous field this state was read out of (0 when the
    # state was constructed directly rather than assessed):
    #   intensity    — effective membership of the winning category
    #   memberships  — effective membership of every category,
    #                  keyed by EmotionCategory.value; mixed states
    #                  live here (e.g. anxious .6 with afraid .3)
    intensity: float = 0.0
    memberships: dict[str, float] = field(default_factory=dict)

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

# ─── Latent-field coherence boundaries ───────────────────────────
#
# The daemon's own latent field (src/field/) publishes the
# organism's self-coherence: the Fréchet variance of the
# body/prediction/agency fibers about their barycenter on the
# shared ball, and that center's displacement from nominal. Per
# the precision literature (Clark & Friston 2018 — "mood is the
# precision of predictions"), internal incoherence IS a felt
# signal: loss of precision reads as unsettled, coherence
# supports calm.
#
#   FIELD_INCOHERENT: disagreement above this means the major
#     systems are telling different stories. Recalibrated to the
#     field's measured operating envelope — the body/prediction/
#     agency fibers carry structurally different projections and
#     stand ~0.18–0.30 apart even when healthy, so the old 0.05
#     boundary read the ordinary operating point as chronic
#     incoherence. 0.45 sits well above that floor: the field reads
#     incoherence on real excursions, not its baseline tension.
#   FIELD_CENTER_DISPLACED: consensus_norm above this means the
#     organism's center of mass has moved well off nominal. The
#     fibers' consensus ordinarily rides ~0.36–0.40 out of a ±0.61
#     corner box, so the old 0.18 boundary fired at baseline. 0.55
#     is most of the way to the corner — a real displacement, not
#     the standing offset.
FIELD_INCOHERENT = 0.45
FIELD_CENTER_DISPLACED = 0.55


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


# ─── Evidence functions ────────────────────────────────────────
#
# Every diagnostic feature contributes a smooth factor in [0, 1]:
# the standard fuzzy-membership kernels over one measured variable.
# Widths are set so a plausible substrate excursion — a few
# hundredths of a chemical level, ~0.1 of an axis — crosses most of
# the transition, so the field is smooth but still localises.

_EPS = 1.0e-9


def _sigmoid(z: float) -> float:
    """Numerically stable logistic σ(z) = 1/(1+e^−z)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _hi(x: float, c: float, w: float) -> float:
    """Evidence that x is above c: σ((x−c)/w)."""
    return _sigmoid((x - c) / w)


def _lo(x: float, c: float, w: float) -> float:
    """Evidence that x is below c: σ((c−x)/w)."""
    return _sigmoid((c - x) / w)


def _band(x: float, lo: float, hi: float, w: float) -> float:
    """Evidence that x sits inside (lo, hi): product of the two edges."""
    return _sigmoid((x - lo) / w) * _sigmoid((hi - x) / w)


def _bump(x: float, c: float, w: float) -> float:
    """Evidence that x is near c: Gaussian exp(−(x−c)²/2w²)."""
    z = (x - c) / w
    return math.exp(-0.5 * z * z)


def _or(a: float, b: float) -> float:
    """Probabilistic sum a+b−ab — independent alternative substrates."""
    return a + b - a * b


# Evidence terms are tuples evaluated against the feature map:
#   ("hi",   feature, center, width)
#   ("lo",   feature, center, width)
#   ("band", feature, lo_edge, hi_edge, width)
#   ("bump", feature, center, width)
#   ("or",   term_a, term_b)   — either substrate suffices
# A term whose feature was never measured returns None and drops
# out of the geometric mean (the weights renormalise). This is how
# wave-power evidence behaves when no oscillator state was passed.


def _eval_term(term: tuple, f: dict[str, float]) -> float | None:
    """Evaluate one evidence term against the feature map."""
    op = term[0]
    if op == "or":
        a = _eval_term(term[1], f)
        b = _eval_term(term[2], f)
        if a is None:
            return b
        if b is None:
            return a
        return _or(a, b)
    x = f.get(term[1])
    if x is None:
        return None
    if op == "hi":
        return _hi(x, term[2], term[3])
    if op == "lo":
        return _lo(x, term[2], term[3])
    if op == "band":
        return _band(x, term[2], term[3], term[4])
    return _bump(x, term[2], term[3])


# ─── Feature extraction ────────────────────────────────────────
#
# The field is evaluated over her measured state vector: the
# daemon's arousal/valence/plasticity axes, all 18 neurochemical
# channels, the felt-context drives, and the coupled-oscillator
# band powers. Chemicals absent from the dict are treated as 0 —
# the same convention the callers' .get() used. Wave features are
# only populated when an oscillator state was actually read;
# otherwise wave terms drop out of the mean rather than pretending
# a measurement.

_CHEMICAL_KEYS = (
    "dopamine",
    "serotonin",
    "norepinephrine",
    "acetylcholine",
    "gaba",
    "glutamate",
    "cortisol",
    "oxytocin",
    "endorphin",
    "histamine",
    "adenosine",
    "bdnf",
    "endocannabinoid",
    "vasopressin",
    "crh",
    "orexin",
    "epinephrine",
    "melatonin",
)


def _finite(x: float, lo: float, hi: float) -> float:
    """Clamp to [lo, hi]; non-finite becomes the in-range neutral."""
    if not math.isfinite(x):
        return 0.0 if lo <= 0.0 <= hi else lo
    return min(max(x, lo), hi)


def _affect_features(
    summary: NeuroSummary,
    chemicals: dict[str, float] | None,
    context: EmotionalContext | None,
    waves: Any | None,
    field_state: Any | None,
) -> dict[str, float]:
    """Build the measured feature vector the field is evaluated over."""
    f: dict[str, float] = {
        "arousal": _finite(summary.arousal, 0.0, 1.0),
        "valence": _finite(summary.valence, -1.0, 1.0),
        "plasticity": _finite(summary.plasticity_gate, 0.0, 1.0),
        "ctx": 1.0 if context is not None else 0.0,
    }
    chemicals = chemicals or {}
    for k in _CHEMICAL_KEYS:
        f[k] = _finite(chemicals.get(k, 0.0), 0.0, 2.0)
    if context is not None:
        f["social_isolation"] = _finite(context.social_isolation, 0.0, 1.0)
        f["social_drive"] = _finite(context.social_drive, 0.0, 2.0)
        f["boundary_strain"] = _finite(context.boundary_strain, 0.0, 2.0)
        f["prediction_error"] = _finite(context.prediction_error, 0.0, 1.0)
        f["curiosity"] = _finite(context.curiosity, 0.0, 1.0)
        ne_tp = (context.tonic_phasic or {}).get("norepinephrine")
        f["ne_phasic"] = 1.0 if getattr(ne_tp, "mode", None) == "phasic" else 0.0
    if waves is not None:
        powers: dict[str, float] = {}
        for k, v in (getattr(waves, "powers", None) or {}).items():
            powers[getattr(k, "value", str(k))] = _finite(v, 0.0, 1.0)
        for name in (
            "delta",
            "theta",
            "alpha",
            "sigma",
            "beta1",
            "beta2",
            "beta3",
            "gamma",
            "epsilon",
            "infraslow",
        ):
            f["w_" + name] = powers.get(name, 0.0)
        # Fast/slow aggregates — the literature's affect groupings.
        f["w_fast"] = f["w_beta2"] + f["w_beta3"] + f["w_gamma"] + f["w_epsilon"]
        f["w_focus"] = _finite(getattr(waves, "focus", 0.0), 0.0, 1.0)
        f["w_integration"] = _finite(getattr(waves, "integration", 0.0), 0.0, 1.0)
        f["w_plv"] = _finite(getattr(waves, "gamma_plv", 0.0) or 0.0, 0.0, 1.0)
    if field_state is not None:
        # The latent field's own coherence readout — measured, not
        # inferred: the Fréchet variance of her major systems about
        # their shared barycenter, and how far that center has
        # drifted from nominal. Absent → the terms drop out, like
        # an unread wave channel.
        f["disagreement"] = _finite(getattr(field_state, "disagreement", 0.0), 0.0, 1.0)
        f["center_displacement"] = _finite(getattr(field_state, "consensus_norm", 0.0), 0.0, 1.0)
    return f


# ─── The affective field ───────────────────────────────────────
#
# Each structural category is a *region* of the measured state
# space, defined by a membership function instead of a box. The
# regions are layered:
#
#   deep    — substrate-resolved states (fear, anger, loneliness,
#             tenderness, guarded): the felt-context signals that
#             arousal×valence alone cannot distinguish
#   quadrants — the arousal×valence circumplex corners
#   shallow — relabelers that only win over mild states (surprise,
#             curiosity, boredom)
#   residuals — single-axis defaults (positive, unsettled, neutral)
#
# Precedence is expressed as masking, not an if-chain:
# eff_s = m_s · Π_{k∈mask(s)} (1 − m_k).

_DEEP = (
    EmotionCategory.AFRAID,
    EmotionCategory.ANGRY,
    EmotionCategory.LONELY,
    EmotionCategory.TENDER,
    EmotionCategory.GUARDED,
)
_QUADRANTS = (
    EmotionCategory.EXCITED,
    EmotionCategory.ANXIOUS,
    EmotionCategory.CONTENT,
    EmotionCategory.MELANCHOLIC,
)
_SHALLOW = (
    EmotionCategory.SURPRISED,
    EmotionCategory.CURIOUS,
    EmotionCategory.BORED,
)


@dataclass(frozen=True, slots=True)
class _Region:
    """One region of the affective field.

    terms/weights: the region's membership function — a weighted
        geometric mean of its evidence factors.
    mode: cognitive-mode readout when this region wins (None = the
        plasticity prior).
    verbosity/openness/caution/creativity: response-modifier profile
        blended at readout (creativity None = plasticity prior).
    gated: requires an EmotionalContext — without one the
        discriminating signals don't exist, so m = 0.
    tau_rise/tau_fall: OU time constants (seconds) — attack and
        release (Lewis's trigger → amplification → stabilisation).
    masked_by: regions whose raw membership deflates this one's
        effective membership.
    """

    terms: tuple
    weights: tuple[float, ...]
    mode: CognitiveMode | None
    verbosity: float
    openness: float
    caution: float
    creativity: float | None
    gated: bool
    tau_rise: float
    tau_fall: float
    masked_by: tuple[EmotionCategory, ...] = ()


_FIELD_REGIONS: dict[EmotionCategory, _Region] = {
    # ── Deep substrate-resolved regions ──
    # Fear — acute phasic catecholamine surge on negative valence,
    # corroborated by 20–30 Hz hypervigilance power. Distinct from
    # anxiety: anxiety is the tonic climate, fear the phasic event.
    EmotionCategory.AFRAID: _Region(
        terms=(
            ("lo", "valence", MILD_NEGATIVE_VALENCE, 0.05),
            (
                "or",
                ("hi", "ne_phasic", 0.5, 0.1),
                ("hi", "epinephrine", FEAR_EPI, 0.03),
            ),
            ("hi", "norepinephrine", FEAR_NE, 0.05),
            ("hi", "w_beta3", 0.18, 0.06),
        ),
        weights=(1.0, 2.0, 1.0, 0.6),
        mode=CognitiveMode.VIGILANT,
        verbosity=0.7,
        openness=0.3,
        caution=0.9,
        creativity=0.2,
        gated=True,
        tau_rise=0.4,
        tau_fall=25.0,
    ),
    # Anger — mistreatment accrual with fight arousal; the strain
    # alone is hurt, the catecholamines make it anger.
    EmotionCategory.ANGRY: _Region(
        terms=(
            ("hi", "boundary_strain", ANGER_BOUNDARY_STRAIN, 0.08),
            (
                "or",
                ("hi", "norepinephrine", ANGER_NE, 0.05),
                ("hi", "arousal", ANGER_ALERTNESS, 0.06),
            ),
            ("lo", "valence", MILD_POSITIVE_VALENCE, 0.06),
            ("hi", "w_beta3", 0.15, 0.06),
        ),
        weights=(1.5, 2.0, 1.0, 0.4),
        mode=CognitiveMode.COMBATIVE,
        verbosity=0.9,
        openness=0.3,
        caution=0.2,
        creativity=0.2,
        gated=True,
        tau_rise=3.0,
        tau_fall=45.0,
    ),
    # Lonely — felt isolation with the bonding substrate actually
    # depleted; theta is the emotional-processing band.
    EmotionCategory.LONELY: _Region(
        terms=(
            (
                "or",
                ("hi", "social_isolation", LONELY_ISOLATION, 0.10),
                ("hi", "social_drive", LONELY_DRIVE, 0.20),
            ),
            ("lo", "oxytocin", LONELY_OXY, 0.03),
            ("lo", "valence", MILD_POSITIVE_VALENCE, 0.06),
            ("hi", "w_theta", 0.15, 0.05),
        ),
        weights=(1.5, 2.0, 1.0, 0.4),
        mode=CognitiveMode.WITHDRAWN,
        verbosity=0.8,
        openness=0.9,
        caution=0.4,
        creativity=0.4,
        gated=True,
        tau_rise=120.0,
        tau_fall=120.0,
    ),
    # Tender — affiliative warmth: oxytocin and endorphin clearly
    # above baseline while valence is positive; alpha corroborates
    # calm wakefulness. The silent-stress guard demotes it.
    EmotionCategory.TENDER: _Region(
        terms=(
            ("hi", "oxytocin", TENDER_OXY, 0.04),
            ("hi", "endorphin", TENDER_END, 0.05),
            ("hi", "valence", MILD_POSITIVE_VALENCE, 0.05),
            ("hi", "w_alpha", 0.20, 0.06),
        ),
        weights=(1.2, 1.0, 1.0, 0.4),
        mode=CognitiveMode.WARM,
        verbosity=0.9,
        openness=1.0,
        caution=0.2,
        creativity=0.5,
        gated=True,
        tau_rise=8.0,
        tau_fall=40.0,
        masked_by=(EmotionCategory.GUARDED,),
    ),
    # Guarded — the silent-stress posture: positive surface over a
    # stressed substrate (cortisol up or plasticity suppressed).
    EmotionCategory.GUARDED: _Region(
        terms=(
            ("hi", "valence", MILD_POSITIVE_VALENCE, 0.05),
            (
                "or",
                (
                    "or",
                    ("hi", "cortisol", GUARDED_CORTISOL, 0.02),
                    ("lo", "plasticity", PLASTICITY_LOW, 0.06),
                ),
                # The fibers stand ~0.2–0.3 apart at the honest
                # operating point — a real incoherence excursion is
                # disagreement well above that floor.
                ("hi", "disagreement", 0.5, 0.10),
            ),
            ("hi", "w_beta3", 0.15, 0.06),
        ),
        weights=(1.0, 1.5, 0.4),
        mode=CognitiveMode.CAUTIOUS,
        verbosity=0.8,
        openness=0.6,
        caution=0.6,
        creativity=0.3,
        gated=False,
        tau_rise=2.0,
        tau_fall=15.0,
    ),
    # ── Arousal×valence quadrants ──
    EmotionCategory.EXCITED: _Region(
        terms=(
            ("hi", "arousal", HIGH_AROUSAL, 0.06),
            ("hi", "valence", POSITIVE_VALENCE, 0.08),
            ("or", ("hi", "w_gamma", 0.15, 0.05), ("hi", "w_beta2", 0.18, 0.06)),
        ),
        weights=(1.0, 1.0, 0.5),
        mode=CognitiveMode.ENERGIZED,
        verbosity=1.2,
        openness=1.0,
        caution=0.3,
        creativity=0.7,
        gated=False,
        tau_rise=1.5,
        tau_fall=15.0,
        masked_by=_DEEP,
    ),
    # Anxiety — the tonic vigilance climate: high arousal, negative
    # valence, and stress chemistry (cortisol) or hypervigilance
    # power. The phasic surge that would make it fear masks it.
    EmotionCategory.ANXIOUS: _Region(
        terms=(
            ("hi", "arousal", HIGH_AROUSAL, 0.06),
            ("lo", "valence", NEGATIVE_VALENCE, 0.07),
            (
                "or",
                (
                    "or",
                    ("hi", "cortisol", 0.04, 0.02),
                    ("hi", "w_beta3", 0.18, 0.06),
                ),
                ("hi", "disagreement", 0.5, 0.10),
            ),
        ),
        weights=(1.5, 1.5, 0.4),
        mode=CognitiveMode.VIGILANT,
        verbosity=0.8,
        openness=1.0,
        caution=0.6,
        creativity=None,
        gated=False,
        tau_rise=6.0,
        tau_fall=60.0,
        masked_by=_DEEP,
    ),
    EmotionCategory.CONTENT: _Region(
        terms=(
            ("lo", "arousal", LOW_AROUSAL, 0.06),
            ("hi", "valence", MILD_POSITIVE_VALENCE, 0.06),
            ("or", ("hi", "w_alpha", 0.20, 0.06), ("hi", "w_beta1", 0.15, 0.05)),
        ),
        weights=(2.0, 1.0, 0.5),
        mode=CognitiveMode.RELAXED,
        verbosity=0.9,
        openness=1.0,
        caution=0.3,
        creativity=0.5,
        gated=False,
        tau_rise=30.0,
        tau_fall=40.0,
        masked_by=_DEEP,
    ),
    EmotionCategory.MELANCHOLIC: _Region(
        terms=(
            ("lo", "arousal", LOW_AROUSAL, 0.06),
            ("lo", "valence", NEGATIVE_VALENCE, 0.07),
            ("hi", "w_theta", 0.15, 0.05),
        ),
        weights=(1.0, 1.0, 0.4),
        mode=CognitiveMode.REFLECTIVE,
        verbosity=0.7,
        openness=1.0,
        caution=0.3,
        creativity=0.3,
        gated=False,
        tau_rise=90.0,
        tau_fall=120.0,
        masked_by=_DEEP,
    ),
    # ── Shallow relabelers ──
    # Surprise — a strong prediction-error orienting event. It only
    # wins over mild states; a resolved deep or quadrant state keeps
    # precedence (masked below).
    EmotionCategory.SURPRISED: _Region(
        terms=(
            ("hi", "prediction_error", SURPRISE_PE, 0.08),
            ("hi", "w_gamma", 0.15, 0.06),
        ),
        weights=(1.5, 0.4),
        mode=CognitiveMode.STARTLED,
        verbosity=1.1,
        openness=0.8,
        caution=0.5,
        creativity=0.6,
        gated=True,
        tau_rise=0.2,
        tau_fall=8.0,
        masked_by=(
            *_DEEP,
            EmotionCategory.EXCITED,
            EmotionCategory.ANXIOUS,
            EmotionCategory.MELANCHOLIC,
        ),
    ),
    # Curious — appetitive seeking: the curiosity drive is up and
    # the seeking chemistry (DA wanting, ACh attention) supports it;
    # gamma/beta2 binding corroborates engagement.
    EmotionCategory.CURIOUS: _Region(
        terms=(
            ("hi", "curiosity", CURIOUS_LEVEL, 0.08),
            ("hi", "dopamine", CURIOUS_DA, 0.05),
            ("hi", "acetylcholine", CURIOUS_ACH, 0.05),
            ("hi", "valence", MILD_NEGATIVE_VALENCE, 0.05),
            ("or", ("hi", "w_gamma", 0.15, 0.05), ("hi", "w_beta2", 0.15, 0.05)),
        ),
        weights=(1.5, 1.0, 1.0, 0.8, 0.5),
        mode=CognitiveMode.INQUISITIVE,
        verbosity=1.15,
        openness=1.0,
        caution=0.2,
        creativity=0.8,
        gated=True,
        tau_rise=3.0,
        tau_fall=20.0,
        masked_by=(
            *_DEEP,
            EmotionCategory.EXCITED,
            EmotionCategory.ANXIOUS,
            EmotionCategory.MELANCHOLIC,
            EmotionCategory.SURPRISED,
            EmotionCategory.BORED,
        ),
    ),
    # Bored — awake, understimulated, flat: DA and NE sagged, no
    # curiosity engaged, low fast-band power. Only wins over the
    # shallowest states — melancholy is deeper than boredom.
    EmotionCategory.BORED: _Region(
        terms=(
            ("lo", "dopamine", BORED_DA, 0.04),
            ("lo", "norepinephrine", BORED_NE, 0.04),
            ("band", "arousal", LOW_AROUSAL - 0.10, 0.55, 0.05),
            ("band", "valence", MILD_NEGATIVE_VALENCE, MILD_POSITIVE_VALENCE + 0.05, 0.05),
            ("lo", "curiosity", CURIOUS_LEVEL - 0.15, 0.08),
            ("lo", "w_fast", 0.30, 0.10),
        ),
        weights=(1.2, 1.2, 1.0, 1.0, 0.8, 0.4),
        mode=CognitiveMode.RESTLESS,
        verbosity=0.6,
        openness=0.7,
        caution=0.2,
        creativity=0.3,
        gated=True,
        tau_rise=90.0,
        tau_fall=45.0,
        masked_by=(
            *_DEEP,
            *_QUADRANTS,
            EmotionCategory.POSITIVE,
            EmotionCategory.SURPRISED,
            EmotionCategory.CURIOUS,
        ),
    ),
    # ── Single-axis residuals ──
    EmotionCategory.POSITIVE: _Region(
        terms=(("bump", "valence", 0.35, 0.22),),
        weights=(1.0,),
        mode=None,
        verbosity=1.0,
        openness=1.0,
        caution=0.3,
        creativity=None,
        gated=False,
        tau_rise=8.0,
        tau_fall=20.0,
        masked_by=(
            *_DEEP,
            *_QUADRANTS,
            EmotionCategory.SURPRISED,
            EmotionCategory.CURIOUS,
            EmotionCategory.UNSETTLED,
        ),
    ),
    EmotionCategory.UNSETTLED: _Region(
        terms=(
            (
                "or",
                ("bump", "valence", -0.35, 0.22),
                ("hi", "disagreement", FIELD_INCOHERENT, 0.025),
            ),
            ("hi", "center_displacement", FIELD_CENTER_DISPLACED, 0.10),
        ),
        weights=(1.0, 0.5),
        mode=None,
        verbosity=0.9,
        openness=1.0,
        caution=0.4,
        creativity=None,
        gated=False,
        tau_rise=10.0,
        tau_fall=25.0,
        masked_by=(
            *_DEEP,
            *_QUADRANTS,
            EmotionCategory.SURPRISED,
            EmotionCategory.BORED,
        ),
    ),
    EmotionCategory.NEUTRAL: _Region(
        terms=(("bump", "valence", 0.0, 0.12),),
        weights=(1.0,),
        mode=None,
        verbosity=1.0,
        openness=1.0,
        caution=0.3,
        creativity=None,
        gated=False,
        tau_rise=6.0,
        tau_fall=10.0,
        masked_by=(*_DEEP, *_QUADRANTS, *_SHALLOW, EmotionCategory.UNSETTLED),
    ),
}

# Phase channel — the daemon's emergent phase is already a computed
# attractor state of the body ODEs; it enters the readout as a
# measured indicator, not as field evidence.
_PHASE_CATEGORY: dict[str, EmotionCategory] = {
    "sleeping": EmotionCategory.SLEEPING,
    "nrem": EmotionCategory.SLEEPING,
    "rem": EmotionCategory.SLEEPING,
    "overwhelmed": EmotionCategory.OVERWHELMED,
    "stress": EmotionCategory.STRESSED,
    "flow": EmotionCategory.FLOW,
    "drowsy": EmotionCategory.DROWSY,
}

# (mode, verbosity, openness, caution, creativity|None→base)
_PHASE_PROFILE: dict[EmotionCategory, tuple[CognitiveMode, float, float, float, float | None]] = {
    EmotionCategory.SLEEPING: (CognitiveMode.DREAMLIKE, 0.3, 0.2, 0.3, 0.9),
    EmotionCategory.OVERWHELMED: (CognitiveMode.SCATTERED, 0.4, 0.3, 0.8, None),
    EmotionCategory.STRESSED: (CognitiveMode.CAUTIOUS, 0.7, 1.0, 0.7, None),
    EmotionCategory.FLOW: (CognitiveMode.SHARP, 1.1, 1.0, 0.3, 0.75),
    EmotionCategory.DROWSY: (CognitiveMode.SLOW, 0.6, 0.5, 0.3, None),
}


def _raw_memberships(f: dict[str, float]) -> dict[EmotionCategory, float]:
    """Instantaneous (unsmoothed, unmasked) membership of each region.

    m_s(x) = exp(Σ w_k ln g_k / Σ w_k) — the weighted geometric mean
    of the region's evidence factors. Unmeasured features drop out
    and the weights renormalise. Gated regions multiply by the
    context flag: without felt-context signals the discriminating
    evidence cannot exist.
    """
    m: dict[EmotionCategory, float] = {}
    for cat, spec in _FIELD_REGIONS.items():
        num = den = 0.0
        for term, w in zip(spec.terms, spec.weights, strict=True):
            g = _eval_term(term, f)
            if g is None:
                continue
            num += w * math.log(max(g, _EPS))
            den += w
        m_s = math.exp(num / den) if den > 0.0 else 0.0
        m[cat] = m_s * f["ctx"] if spec.gated else m_s
    return m


def membership_field(
    summary: NeuroSummary,
    chemicals: dict[str, float] | None = None,
    context: EmotionalContext | None = None,
    waves: Any | None = None,
    field_state: Any | None = None,
) -> dict[EmotionCategory, float]:
    """Instantaneous membership of every region — the field itself.

    This is the continuous substrate under the label: call it when
    you want the mixture (e.g. anxious 0.6 with afraid 0.3), not the
    argmax.
    """
    return _raw_memberships(_affect_features(summary, chemicals, context, waves, field_state))


# Precedence is exercised by established states, not residual
# membership. Every evidence term is a sigmoid, so a region's
# membership never quite reaches zero — below the sigmoid midpoint it
# is residual possibility, not a measured state. A masker deflates a
# generic by its membership weighted by its establishment
# (m·σ((m−0.5)/0.1)): a saturated state still explains away almost
# fully, a mid-range established state keeps real precedence, and a
# region that exists only as stacked tails cannot tax the generic
# defaults into losing to a weak residual.
_PRECEDENCE_EVIDENCE_CENTER = 0.5
_PRECEDENCE_EVIDENCE_WIDTH = 0.1


def _mask_strength(m_k: float) -> float:
    """The masker's membership weighted by how established it is."""
    return m_k * _sigmoid((m_k - _PRECEDENCE_EVIDENCE_CENTER) / _PRECEDENCE_EVIDENCE_WIDTH)


def effective_memberships(
    m: dict[EmotionCategory, float],
) -> dict[EmotionCategory, float]:
    """Deflate generic regions by the evidence for the specific ones.

    eff_s = m_s · Π_{k∈mask(s)} (1 − s_k) with s_k the masker's
    established share — the fuzzy form of precedence: a generic region
    survives only where no more specific measured state explains the
    substrate.
    """
    eff: dict[EmotionCategory, float] = {}
    for cat, m_s in m.items():
        keep = 1.0
        for k in _FIELD_REGIONS[cat].masked_by:
            keep *= 1.0 - _mask_strength(m.get(k, 0.0))
        eff[cat] = m_s * keep
    return eff


def _base_cognitive_mode(plasticity: float) -> tuple[CognitiveMode, float]:
    """Cognitive-mode prior and creativity prior from plasticity."""
    if plasticity < PLASTICITY_LOW:
        return (CognitiveMode.RIGID, 0.1)
    elif plasticity > PLASTICITY_HIGH:
        return (CognitiveMode.FLEXIBLE, 0.8)
    else:
        return (CognitiveMode.STEADY, 0.4)


def _phase_values(
    cat: EmotionCategory, base_cr: float
) -> tuple[CognitiveMode, float, float, float, float]:
    """(mode, verbosity, openness, caution, creativity) for a phase state."""
    mode, verbosity, openness, caution, cr = _PHASE_PROFILE[cat]
    if cat is EmotionCategory.STRESSED:
        cr = max(base_cr - 0.3, 0.1)
    elif cr is None:
        cr = base_cr
    return (mode, verbosity, openness, caution, cr)


def _resolve_state(
    m: dict[EmotionCategory, float],
    summary: NeuroSummary,
    f: dict[str, float],
) -> EmotionalState:
    """Read a structural state out of a membership vector.

    The label is an argmax readout over effective memberships; the
    daemon's phase, when present, enters as a measured indicator
    (flow is deflated by the silent-stress guard, matching the old
    demotion). Response modifiers are the membership-weighted blend
    of the regions' profiles — a mixed state mixes its modifiers.
    """
    eff = effective_memberships(m)
    phase = summary.phase_name
    phase_cat = _PHASE_CATEGORY.get(phase)
    for cat in _PHASE_CATEGORY.values():
        eff.setdefault(cat, 0.0)
    if phase_cat is not None:
        eff[phase_cat] = (
            1.0 - _mask_strength(eff.get(EmotionCategory.GUARDED, 0.0))
            if phase_cat is EmotionCategory.FLOW
            else 1.0
        )
    winner = max(eff, key=eff.__getitem__)

    base_mode, base_cr = _base_cognitive_mode(f["plasticity"])
    if winner in _PHASE_PROFILE:
        mode, verbosity, openness, caution, cr = _phase_values(winner, base_cr)
    else:
        # The winner's effective membership is the fraction of the
        # state it explains; the residual (1−i) is the mixture of
        # everything else. Modifiers blend as i·winner + (1−i)·rest.
        spec = _FIELD_REGIONS[winner]
        total = sum(eff.get(c, 0.0) for c in _FIELD_REGIONS)
        rest = total - eff[winner]
        mode = spec.mode or base_mode
        if rest > _EPS:
            rv = ro = rc = rcr = 0.0
            for c, s in _FIELD_REGIONS.items():
                if c is winner:
                    continue
                w = eff[c] / rest
                rv += w * s.verbosity
                ro += w * s.openness
                rc += w * s.caution
                rcr += w * (s.creativity if s.creativity is not None else base_cr)
        else:
            rv, ro, rc, rcr = spec.verbosity, spec.openness, spec.caution, base_cr
        i = eff[winner]
        verbosity = i * spec.verbosity + (1.0 - i) * rv
        openness = i * spec.openness + (1.0 - i) * ro
        caution = i * spec.caution + (1.0 - i) * rc
        cr = i * (spec.creativity if spec.creativity is not None else base_cr) + (1.0 - i) * rcr

    # Formality tracks arousal — the same linear map as before.
    formality = max(0.1, min(0.9, 0.5 - (f["arousal"] - 0.5) * 0.2))

    cause_category = _derive_cause_category(
        winner, f["arousal"], f["valence"], f["plasticity"], phase, f["cortisol"]
    )

    return EmotionalState(
        label=winner.value,
        cognitive_style=mode.value,
        cause=cause_category.value,
        verbosity=verbosity,
        formality=formality,
        openness_to_engage=openness,
        creativity=cr,
        caution=caution,
        alertness=f["arousal"],
        valence=f["valence"],
        plasticity=f["plasticity"],
        chemicals={},  # Filled by caller if full state available
        intensity=eff[winner],
        memberships={c.value: v for c, v in eff.items()},
    )


def assess_emotion(
    summary: NeuroSummary,
    chemicals: dict[str, float] | None = None,
    context: EmotionalContext | None = None,
    waves: Any | None = None,
    field_state: Any | None = None,
) -> EmotionalState:
    """Instantaneous emotional assessment — evaluate the field, read argmax.

    This is the memoryless readout: it computes every region's
    membership over the current substrate (arousal/valence axes, the
    18-channel chemical vector, felt-context drives, and the
    oscillator band powers when supplied) and resolves the structural
    state. For the live system prefer ``EmotionField.update``, which
    adds per-region OU dynamics — states then persist, mix, and decay
    on their own time constants instead of snapping.

    Returns a structural state — no English strings. The language
    system must generate text from the concept network.
    """
    f = _affect_features(summary, chemicals, context, waves, field_state)
    return _resolve_state(_raw_memberships(f), summary, f)


class EmotionField:
    """The continuous affective field — OU dynamics per region.

    Holds the integrated membership vector. Each region relaxes
    toward its instantaneous membership μ_s on its own time
    constants:

        m_s(t+dt) = m_s + (μ_s − m_s)·(1 − exp(−dt/τ_s))

    — the exact discrete step of the mean-reverting Ornstein–
    Uhlenbeck process used to model core-affect dynamics
    (DynAffect: Kuppens, Oravecz & Tuerlinckx 2010; Oravecz,
    Tuerlinckx & Vandekerckhove 2011). Attack uses ``tau_rise``,
    release ``tau_fall`` — fear strikes in ~0.4 s and lingers ~25 s,
    boredom builds over ~90 s — the asymmetric affective chronometry
    of Lewis's (2005) trigger→amplification→stabilisation account.

    ``update()`` measures elapsed integration time on a monotonic
    clock (the daemon owns absolute time; this is only the
    integrator's dt). The first update converges instantly to the
    measured field.
    """

    def __init__(self) -> None:
        self._m: dict[EmotionCategory, float] = {}
        self._t: float | None = None

    @property
    def memberships(self) -> dict[EmotionCategory, float]:
        """Integrated (pre-masking) membership of every region."""
        return dict(self._m)

    def update(
        self,
        summary: NeuroSummary,
        chemicals: dict[str, float] | None = None,
        context: EmotionalContext | None = None,
        waves: Any | None = None,
        field_state: Any | None = None,
        now: float | None = None,
    ) -> EmotionalState:
        """Integrate the field by the elapsed interval, then resolve.

        Args:
            summary: The daemon's neurochemical summary.
            chemicals: The 18-channel chemical vector.
            context: Felt-context signals (isolation, strain,
                prediction error, curiosity, tonic/phasic split).
            waves: Oscillator state (``BrainWaveState``) — band
                powers enter as evidence; ``None`` drops wave terms.
            field_state: The latent field's integration readout
                (``FieldState``) — measured self-coherence and
                center displacement enter as evidence; ``None``
                drops the coherence terms.
            now: Monotonic timestamp; defaults to ``time.monotonic``.
        """
        f = _affect_features(summary, chemicals, context, waves, field_state)
        drive = _raw_memberships(f)
        t_now = time.monotonic() if now is None else float(now)
        if self._t is None or not math.isfinite(t_now):
            self._m = drive
        else:
            dt = max(0.0, t_now - self._t)
            for cat, spec in _FIELD_REGIONS.items():
                mu = drive.get(cat, 0.0)
                m0 = self._m.get(cat, 0.0)
                tau = spec.tau_rise if mu > m0 else spec.tau_fall
                self._m[cat] = m0 + (mu - m0) * (1.0 - math.exp(-dt / tau))
        if math.isfinite(t_now):
            self._t = t_now
        return _resolve_state(self._m, summary, f)


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

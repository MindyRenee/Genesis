"""Affect appraisal as compositional operator algebra on valence–arousal space.

This is not a rule engine. An utterance's affect is computed by folding
a sequence of *linear operators* and *affect vectors* over a
two-dimensional affective space — the same formal family cognitive
science uses for compositional semantics and dimensional affect:

- **Affective space** A ⊂ ℝ² = (valence v, arousal a) — Russell's
  circumplex / dimensional affect (Russell 1980, Psych. Rev. 87;
  Barrett & Russell 1999). Every word Genesis has learned carries an
  affect vector φ(w) = (v, κ|v|): its learned valence plus the arousal
  its intensity implies. The lexicon itself is *learned through
  interaction* — nothing about word meaning is seeded.

- **Composition is linear-operator action.** Grammatical modifiers are
  not flags on a counter; they are 2×2 matrices acting on the next
  affect vector (Smolensky 1990-style vector-space semantics — meaning
  composes by linear operations on a state space):

      negation    N = [[−ρ, 0], [0, 1]]   — valence reflection; arousal
                                            survives ("not good" is
                                            still *felt*, just inverted)
      intensifier G = g·I,   g > 1        — isotropic dilation
      diminisher  D = d·I,   d < 1        — isotropic contraction

  Consecutive modifiers compose by multiplication — "not very good" is
  N·G·φ, an operator product, not a patchwork of booleans.

- **Modifier scope is persistence-decayed, not windowed.** A pending
  operator carries charge q ∈ [0,1]; every intervening neutral token
  decays it q ← q·e^{−1/τ} and the operator acts only while q ≥ ½ —
  its *half-life*. Scale modifiers (τ_mod ≈ 1/ln 2.5) discharge after a
  single neutral token; negation (τ_neg = 4.5) survives ~3, the same
  syntactic reach VADER hardcodes as a counter — but here it is the
  half-life of a transient expectation, a drift term, not a rule.

- **Contrastive junctions flush the integrator.** "good but slow"
  attenuates the accumulated state S ← (1−Λ)·S at the junction — a
  leak jump, the discrete analog of an attentional reset in the
  affect integrator — so the clause after "but" dominates the residue
  the way post-contrastive weighting intends, without a special pass.

- **Emphasis is an arousal operator.** ALL-CAPS dilates the token's
  affect vector; repeated "!" applies a scalar gain to the terminal
  state — emphasis lives on the arousal axis where it belongs and
  leaks onto valence through the same compress.

- **Report is a saturating compress.** The terminal valence is
  normalized through the signed compress x/√(x²+α) (a smooth
  odd sigmoid — the report map is compressive like every affect
  readout). Positive/negative/neutral proportions are the valence
  masses of the trajectory normalized against token mass.

Word valences remain learned (``LEARNED_LEXICON``); the function-word
operator table is grammar, not meaning — "not", "very", "but" are
syntactic operators in every speaker of English, which is exactly the
kind of seed the project allows: building blocks, not answers.

References: Russell (1980); Barrett & Russell (1999); Smolensky
(1990) tensor-product semantics; Oravecz & Tuerlinckx (2011) and
Kuppens, Oravecz & Tuerlinckx (2010), feelings as a linear dynamical
system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from infrastructure.text import words

# ─── Lexicon ──────────────────────────────────────────────────────
# Seed lexicon REMOVED by request — Genesis learns sentiment the
# right way, through interaction (see LEARNED_LEXICON below and
# ConceptLearner.connect_word_to_valence). LEXICON is kept as an
# empty dict so existing imports keep working; all scores must come
# from learning.

LEXICON: dict[str, float] = {}  # seed lexicon removed — learning only (see LEARNED_LEXICON below)

# ─── Learned lexicon (canonical) ──────────────────────────────────
# Words Genesis has learned through interaction (social labeling,
# valence-tagged episodes). This is its own acquired knowledge and
# the ONLY source analyze_sentiment consults.

LEARNED_LEXICON: dict[str, float] = {}


def learn_sentiment(word: str, score: float) -> None:
    """Register a learned sentiment score, overriding the seed lexicon.

    Args:
        word: Single word (case-insensitive, stored lowercased).
        score: Sentiment in [-4, +4]. Positive = pleasant, negative
            = unpleasant. Clamped to range.
    """
    w = word.lower().strip()
    if not w or len(w) > 64:
        return
    LEARNED_LEXICON[w] = max(-4.0, min(4.0, float(score)))


def clear_learned_sentiment() -> None:
    """Clear the learned overlay (tests / fresh-start only)."""
    LEARNED_LEXICON.clear()


def lookup_sentiment(word: str) -> float | None:
    """Look up a word's sentiment: learned first, seed lexicon fallback.

    Returns None when neither source knows the word.
    """
    w = word.lower().strip()
    if w in LEARNED_LEXICON:
        return LEARNED_LEXICON[w]
    return LEXICON.get(w)


def has_learned_sentiment(word: str) -> bool:
    """True when the word has a learned (non-seed) sentiment score."""
    return word.lower().strip() in LEARNED_LEXICON


# ─── Affective space ──────────────────────────────────────────────
# A ⊂ ℝ²: component 0 is valence, component 1 is arousal. A learned
# word's affect vector places its valence on the first axis and the
# arousal its magnitude implies on the second — |v| is the learned
# intensity, and arousal is intensity, not sign.

_AROUSAL_GAIN = 1.0  # arousal carried by a unit of learned valence

# Core affect — the organism's standing (valence, arousal), read as
# the fold's initial condition (Russell, 2003). Appraisal is
# state-contingent: residual drive is experienced as part of the
# current stimulus's affect (Schachter & Singer, 1962 — arousal
# misattribution). Driven once per turn by the body's summary;
# (0,0) means no standing state and preserves the stimulus-only fold.
_CORE_AFFECT: tuple[float, float] = (0.0, 0.0)


def set_core_affect(valence: float = 0.0, arousal: float = 0.0) -> None:
    """Set the organism's standing affective state (core affect).

    ``valence`` in [-1,1], ``arousal`` in [0,1] from the body's
    neurochemical summary. The fold starts at this point, so the same
    stimulus appraises differently in an aroused body than a calm one
    — the misattribution term of emotion.
    """
    global _CORE_AFFECT
    try:
        v = float(valence)
        a = float(arousal)
    except (TypeError, ValueError):
        return
    if math.isfinite(v) and math.isfinite(a):
        _CORE_AFFECT = (max(-1.0, min(1.0, v)), max(0.0, min(1.0, a)))


def _affect_vector(token: str) -> np.ndarray | None:
    """The token's learned affect vector in A, or None if unknown.

    φ(w) = (v_w, κ·|v_w|): valence on axis 0, intensity-as-arousal on
    axis 1. Unknown words contribute no vector — they are mass on the
    neutral axis, not zero-valence claims.
    """
    v = lookup_sentiment(token)
    if v is None:
        return None
    return np.array([v, _AROUSAL_GAIN * abs(v)], dtype=float)


# ─── Operators ────────────────────────────────────────────────────
# Function-word grammar as linear operators on A. These are syntactic
# constants — English grammar fixes what "not" does to a predication —
# while every affect vector above is learned.

_NEG_ATTENUATION = 0.9  # valence reflection keeps 90% of magnitude


def _negation_operator() -> np.ndarray:
    """Negation as valence reflection: diag(−ρ, 1) on A.

    Reflects the valence axis while preserving arousal — "not great"
    remains an aroused state, only inverted. The 0.9 attenuation is the
    hedging residue negation leaves (litotes is rarely a clean flip).
    """
    return np.array([[-_NEG_ATTENUATION, 0.0], [0.0, 1.0]])


def _dilation(gain: float) -> np.ndarray:
    """Isotropic dilation g·I — the intensifier/diminisher operator.

    g > 1 amplifies valence and arousal together; g < 1 contracts both.
    """
    return np.array([[gain, 0.0], [0.0, gain]])


# ─── Function-word operator seeds ─────────────────────────────────
# Grammar, not meaning: which words operate, and the matrix each maps
# to. Word → operator-matrix constructor args.

NEGATIONS: frozenset[str] = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "nowhere",
        "dont",
        "cant",
        "cannot",
        "wont",
        "wouldnt",
        "shouldnt",
        "couldnt",
        "isnt",
        "wasnt",
        "arent",
        "werent",
        "doesnt",
        "didnt",
        "hasnt",
        "havent",
        "hadnt",
        "without",
    }
)

# Dilation gains g > 1 for intensifier operators.
INTENSIFIERS: dict[str, float] = {
    "very": 1.5,
    "extremely": 1.8,
    "really": 1.4,
    "so": 1.3,
    "too": 1.3,
    "completely": 1.7,
    "absolutely": 1.8,
    "totally": 1.7,
    "utterly": 1.8,
    "highly": 1.6,
    "deeply": 1.6,
    "incredibly": 1.7,
    "remarkably": 1.5,
    "particularly": 1.4,
    "especially": 1.4,
    "exceptionally": 1.7,
    "super": 1.5,
    "truly": 1.4,
    "terribly": 1.5,
    "awfully": 1.5,
    "hugely": 1.5,
}

# Contraction gains d < 1 for diminisher operators.
# "kind" is NOT here — it is a positive sentiment word when it stands
# alone; the hedge is the *phrase* "kind of", in MODIFIER_PHRASES.
DIMINISHERS: dict[str, float] = {
    "slightly": 0.5,
    "somewhat": 0.6,
    "kinda": 0.7,
    "sort": 0.7,
    "barely": 0.4,
    "hardly": 0.4,
    "scarcely": 0.4,
    "marginally": 0.5,
    "mildly": 0.6,
    "moderately": 0.7,
    "fairly": 0.8,
    "rather": 0.8,
    "bit": 0.7,
    "little": 0.7,
    "less": 0.6,
}

# Multi-word operators, matched before per-token scoring so their
# constituents never reach the fold as independent tokens — "kind of"
# is a contraction, not the adjective "kind" followed by "of".
# tuple → dilation gain.
MODIFIER_PHRASES: dict[tuple[str, ...], float] = {
    ("a", "great", "deal"): 1.5,
    ("kind", "of"): 0.7,
    ("sort", "of"): 0.7,
    ("a", "lot"): 1.5,
    ("a", "bit"): 0.7,
    ("a", "little"): 0.7,
    ("a", "tad"): 0.5,
    ("a", "touch"): 0.6,
    ("very", "much"): 1.4,
}

# Contrastive junctions — attentional resets, not operators.
CONTRASTIVES: frozenset[str] = frozenset(
    {
        "but",
        "however",
        "although",
        "though",
        "yet",
        "still",
        "nevertheless",
        "nonetheless",
        "except",
    }
)

# ─── Persistence kernels ──────────────────────────────────────────
# A pending operator's charge decays q ← q·e^{−1/τ} per neutral token
# and acts while q ≥ ½ (its half-life boundary). The τ's are the
# syntactic reach of each operator class:
#   scale modifiers — bind only the immediately following affect word
#                     (q < ½ after one neutral token ⇒ τ < 1/ln2)
#   negation        — persists across ~3 neutral tokens
#                     (q ≥ ½ at d ≤ 3, q < ½ at d = 4 ⇒ τ ∈ [4.33, 5.77))
_TAU_MODIFIER = 1.0  # charge 1 → e^{−1} ≈ 0.37 < ½ after one neutral token
_TAU_NEGATION = 4.5  # charge 1 → e^{−3/4.5} ≈ 0.51 ≥ ½ at distance 3
_CHARGE_BOUND = 0.5  # half-life: the operator's activation criterion


@dataclass
class _PendingOp:
    """A composed operator waiting on the next affect vector.

    ``matrix`` is the accumulated operator product (later modifiers
    left-multiply); ``charge`` decays exponentially per neutral token;
    ``tau`` is the persistence kernel of the weakest-link operator —
    a scale modifier inside a negation discharge dies with the
    modifier, matching how the stack reads.
    """

    matrix: np.ndarray
    charge: float
    tau: float

    def decay(self) -> None:
        """One neutral token's worth of exponential charge decay."""
        self.charge *= math.exp(-1.0 / self.tau)

    @property
    def live(self) -> bool:
        """Whether the operator still exceeds its half-life bound."""
        return self.charge >= _CHARGE_BOUND


# ─── Contrastive flush ────────────────────────────────────────────
# A contrastive junction attenuates the accumulated valence mass —
# the integrator leaks: S ← (1−Λ)·S. Λ = 0.75 leaves a quarter of
# the pre-junction mass, the discrete analog of an attentional reset.

_CONTRASTIVE_LEAK = 0.75

# ─── Emphasis operators ───────────────────────────────────────────
# Orthographic arousal: ALL-CAPS dilates the token's own vector;
# !!! applies a scalar gain to the terminal state.

_CAPS_DILATION = 1.5
_EXCLAMATION_BOOST_THRESHOLD = 3  # need this many ! to register an impulse


def _exclamation_gain(text: str) -> float:
    """Terminal arousal gain from exclamation runs — a scalar operator
    on the whole utterance state, saturating at +50%."""
    count = text.count("!")
    if count >= _EXCLAMATION_BOOST_THRESHOLD:
        return 1.0 + min(0.5, 0.1 * count)
    return 1.0


# ─── Tokenization ─────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Split text into word tokens, preserving original case for the
    emphasis operator. Lowercasing happens later in ``_normalize``.

    Uses the canonical word boundary: Unicode letters (not just ASCII —
    "gérard" is one token, not "g"+"rard") and apostrophe variants
    folded, so "don’t" tokenizes whole instead of fragmenting into
    "don","t" — which used to silently drop the negation and flip the
    message's valence.
    """
    return words(text)


def _normalize(token: str) -> str:
    """Normalize a token for operator/lexicon lookup.

    Strips apostrophes from contractions: "don't" → "dont",
    "can't" → "cant". This matches the NEGATIONS set which stores
    contractions without apostrophes.
    """
    return token.lower().replace("'", "")


def _match_phrase(tokens: list[str], index: int) -> tuple[int, float] | None:
    """Match a multi-word operator phrase at ``index``.

    Returns (length, gain) for the longest matching phrase, or None.
    Phrases are matched longest-first so "a great deal" outranks any
    two-word prefix.
    """
    for length in (3, 2):
        end = index + length
        if end > len(tokens):
            continue
        key = tuple(_normalize(t) for t in tokens[index:end])
        if key in MODIFIER_PHRASES:
            return length, MODIFIER_PHRASES[key]
    return None


# ─── The fold ─────────────────────────────────────────────────────


def _fold(tokens: list[str]) -> tuple[list[np.ndarray], int]:
    """Fold the token stream into the trajectory of operator-applied
    affect contributions.

    The scan is a single pass: affect vectors are pushed through the
    pending operator product, contrastive junctions flush the
    accumulated mass, and neutral tokens decay operator charges.
    Returns (contributions, neutral_token_count).
    """
    contributions: list[np.ndarray] = []
    neutral = 0
    pending: list[_PendingOp] = []

    i = 0
    n = len(tokens)
    while i < n:
        raw = tokens[i]
        token = _normalize(raw)

        # Multi-word operator phrase first — its constituents must
        # never reach the fold as independent tokens.
        phrase = _match_phrase(tokens, i)
        if phrase is not None:
            length, gain = phrase
            pending.append(_PendingOp(_dilation(gain), 1.0, _TAU_MODIFIER))
            i += length
            continue

        # Contrastive junction: flush accumulated mass, discharge all
        # pending operators — the reset is total.
        if token in CONTRASTIVES:
            pending.clear()
            for c in contributions:
                c *= 1.0 - _CONTRASTIVE_LEAK
            i += 1
            continue

        # Negation operator: reflection matrix with the long kernel.
        if token in NEGATIONS:
            pending.append(_PendingOp(_negation_operator(), 1.0, _TAU_NEGATION))
            i += 1
            continue

        # Scale operators: dilation/contraction with the short kernel.
        if token in INTENSIFIERS:
            pending.append(_PendingOp(_dilation(INTENSIFIERS[token]), 1.0, _TAU_MODIFIER))
            i += 1
            continue
        if token in DIMINISHERS:
            pending.append(_PendingOp(_dilation(DIMINISHERS[token]), 1.0, _TAU_MODIFIER))
            i += 1
            continue

        # Affect vector: push through the live operator product, then
        # discharge every operator — scope binds the first affect word.
        phi = _affect_vector(token)
        if phi is not None:
            live = [op for op in pending if op.live]
            for op in live:
                phi = op.matrix @ phi
            pending.clear()
            # Orthographic emphasis: ALL-CAPS dilates this token's vector.
            if len(raw) >= 3 and raw.isupper():
                phi = phi * _CAPS_DILATION
            contributions.append(phi)
        else:
            neutral += 1
            # Neutral mass decays every pending charge; dead operators drop.
            for op in pending:
                op.decay()
            pending = [op for op in pending if op.live]

        i += 1

    return contributions, neutral


# ─── Report ───────────────────────────────────────────────────────

_COMPRESS_ALPHA = 15.0  # variance-stabilizer of the saturating compress


def analyze_sentiment(text: str) -> dict:
    """Appraise the affect of a text in valence–arousal space.

    The trajectory folds operator-applied affect vectors; the report
    compresses the terminal valence and reads the valence masses.

    Returns a dict with:
        - compound: float in [-1, 1] — compressed terminal valence
        - arousal: float in [0, 1] — compressed terminal arousal
        - positive: float in [0, 1] — positive valence mass proportion
        - negative: float in [0, 1] — negative valence mass proportion
        - neutral: float in [0, 1] — unaffective mass proportion
        - sentiment_words: int — affect vectors that entered the fold
    """
    tokens = _tokenize(text)
    if not tokens:
        return {
            "compound": 0.0,
            "arousal": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "sentiment_words": 0,
        }

    contributions, neutral_count = _fold(tokens)
    if not contributions:
        # No affect vectors — but the organism's standing state is
        # still the reading: an unaffective utterance in an aroused
        # body appraises aroused (the fold's initial condition is
        # also the whole terminal state here).
        v0, a0 = _CORE_AFFECT
        return {
            "compound": v0 / math.sqrt(v0 * v0 + _COMPRESS_ALPHA),
            "arousal": a0 / math.sqrt(a0 * a0 + _COMPRESS_ALPHA),
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "sentiment_words": 0,
        }

    # Terminal state: the fold runs from the organism's core affect —
    # standing (v0, a0) plus the stimulus trajectory under the
    # exclamation gain. The baseline is not gained by punctuation;
    # it is where the trajectory begins.
    gain = _exclamation_gain(text)
    v0, a0 = _CORE_AFFECT
    valence = v0 + sum(c[0] for c in contributions) * gain
    arousal = a0 + sum(c[1] for c in contributions) * gain

    # Signed saturating compress — the report map.
    compound = valence / math.sqrt(valence * valence + _COMPRESS_ALPHA)
    compound = max(-1.0, min(1.0, compound))
    arousal_report = arousal / math.sqrt(arousal * arousal + _COMPRESS_ALPHA)
    arousal_report = max(0.0, min(1.0, arousal_report))

    # Valence masses: the signed projection of the trajectory onto
    # the valence axis, normalized against neutral token mass.
    pos_mass = sum(c[0] for c in contributions if c[0] > 0.0)
    neg_mass = sum(-c[0] for c in contributions if c[0] < 0.0)
    total = pos_mass + neg_mass + neutral_count
    if total > 0:
        positive = pos_mass / total
        negative = neg_mass / total
        neutral_prop = max(0.0, min(1.0, neutral_count / total))
    else:
        positive = 0.0
        negative = 0.0
        neutral_prop = 1.0

    return {
        "compound": compound,
        "arousal": arousal_report,
        "positive": positive,
        "negative": negative,
        "neutral": neutral_prop,
        "sentiment_words": len(contributions),
    }

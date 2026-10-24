"""Tests for the Genesis cognitive mind modules.

Tests each cognitive subsystem in isolation, then tests the full
mind end-to-end with a running daemon.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pytest
from frontal_lobe.self import SelfModel
from limbic_system.emotion import (
    EmotionalState,
    EmotionCategory,
    assess_emotion,
)
from occipital_lobe.perception import Intent, QuestionType, perceive

from genesis_client import NeuroSummary
from genesis_client.protocol import PHASE_FLOW, PHASE_SLEEPING, PHASE_STRESS

logger = logging.getLogger(__name__)

# ─── Self-model tests ─────────────────────────────────────────


def test_self_model_defaults():
    """Self-model has sensible defaults."""
    model = SelfModel()
    assert model.name == "Genesis"
    assert model.personality.openness > 0.7
    assert len(model.values) >= 5
    assert model.values[0].name == "understanding"


def test_self_model_trait_concept_names():
    """Self-model maps trait values to concept names, not hardcoded strings.

    The personality traits are no longer described by fixed English
    phrases via describe(). Instead, trait_concept_names() returns
    concept names (building blocks) for the active poles of each
    Big Five dimension. The language engine composes the actual
    phrasing from these concept names.
    """
    model = SelfModel()
    names = model.personality.trait_concept_names()
    # Default openness 0.85 > 0.6 → high pole: curious, creative
    assert "curious" in names
    assert "creative" in names
    # Default extraversion 0.45 is between 0.4 and 0.6 → neutral, no pole
    assert "introspective" not in names
    assert "expressive" not in names


def test_self_model_learn():
    """Self-model can learn new facts."""
    model = SelfModel()
    model.learn("user_name", "Alice")
    assert model.world_knowledge["user_name"] == "Alice"


def test_self_model_relationship_notes():
    """Self-model tracks relationship notes."""
    model = SelfModel()
    model.add_relationship_note("User likes philosophy")
    assert "User likes philosophy" in model.relationship_notes


# ─── Emotion tests ─────────────────────────────────────────────


def test_emotion_positive_excited():
    """High arousal + positive valence → excited."""
    summary = NeuroSummary(
        arousal=0.8,
        valence=0.5,
        global_tone=0.6,
        plasticity_gate=0.6,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,  # active
    )
    emo = assess_emotion(summary)
    assert emo.label == "excited"
    assert emo.verbosity > 1.0


def test_emotion_stress_phase():
    """Stress phase → stressed."""
    summary = NeuroSummary(
        arousal=0.7,
        valence=-0.3,
        global_tone=0.3,
        plasticity_gate=0.3,
        encoding_weight=0.4,
        consolidation_weight=0.4,
        retrieval_weight=0.4,
        phase=PHASE_STRESS,
    )
    emo = assess_emotion(summary)
    assert emo.label == "stressed"
    assert emo.caution > 0.5


def test_emotion_sleeping():
    """Sleeping phase → sleeping."""
    summary = NeuroSummary(
        arousal=0.2,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.3,
        encoding_weight=0.3,
        consolidation_weight=0.3,
        retrieval_weight=0.3,
        phase=PHASE_SLEEPING,
    )
    emo = assess_emotion(summary)
    assert emo.label == "sleeping"
    assert emo.openness_to_engage < 0.5


def test_emotion_flow():
    """Flow phase → in flow."""
    summary = NeuroSummary(
        arousal=0.6,
        valence=0.4,
        global_tone=0.6,
        plasticity_gate=0.7,
        encoding_weight=0.6,
        consolidation_weight=0.6,
        retrieval_weight=0.6,
        phase=PHASE_FLOW,
    )
    emo = assess_emotion(summary)
    assert emo.label == "in flow"
    assert emo.cognitive_style == "sharp"


def test_emotion_structural_state():
    """Emotional state provides structural category and mode."""
    summary = NeuroSummary(
        arousal=0.5,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.4,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,
    )
    emo = assess_emotion(summary)
    from limbic_system.emotion import CognitiveMode, EmotionCategory

    assert emo.category == EmotionCategory.NEUTRAL
    assert emo.cognitive_mode == CognitiveMode.STEADY


# ─── Contextual (full-spectrum) emotion tests ────────────────
#
# Each test supplies an EmotionalContext carrying the real substrate
# the circumplex cannot see — the same fields the feeling reporter
# populates from the world model, boundary accumulator, inference
# surprise, curiosity engine, and tonic/phasic split.


def _summary(arousal: float, valence: float, plasticity: float = 0.5):
    return NeuroSummary(
        arousal=arousal,
        valence=valence,
        global_tone=0.4,
        plasticity_gate=plasticity,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,
    )


def _tp(chemical: str = "norepinephrine", mode: str = "phasic"):
    from limbic_system.emotional_regulator import TonicPhasic

    return TonicPhasic(
        chemical=chemical,
        tonic=0.3,
        phasic=0.15,
        phasic_ratio=0.5,
        mode=mode,
    )


def test_emotion_fear_is_phasic_not_tonic():
    """A phasic NE surge + adrenal burst on negative valence is fear,
    not climate anxiety."""
    from limbic_system.emotion import EmotionalContext

    chemicals = {
        "norepinephrine": 0.45,
        "epinephrine": 0.30,
        "cortisol": 0.03,
    }
    ctx = EmotionalContext(tonic_phasic={"norepinephrine": _tp()})
    emo = assess_emotion(_summary(0.75, -0.3), chemicals, ctx)
    assert emo.category == EmotionCategory.AFRAID
    assert emo.cognitive_mode.value == "startled" or emo.cognitive_mode.value == "vigilant"
    assert emo.cause == "fear_acute"
    assert emo.caution > 0.8


def test_emotion_tonic_threat_stays_anxious():
    """The same arousal+negative valence with tonic (not phasic) NE
    stays anxious — event vs climate discrimination."""
    from limbic_system.emotion import EmotionalContext

    chemicals = {"norepinephrine": 0.45, "epinephrine": 0.16}
    ctx = EmotionalContext(tonic_phasic={"norepinephrine": _tp(mode="tonic")})
    emo = assess_emotion(_summary(0.75, -0.3), chemicals, ctx)
    assert emo.category == EmotionCategory.ANXIOUS


def test_emotion_anger_from_boundary_strain():
    """Mistreatment accrual with fight arousal reads as anger."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(boundary_strain=0.7)
    chemicals = {"norepinephrine": 0.4}
    emo = assess_emotion(_summary(0.6, -0.15), chemicals, ctx)
    assert emo.category == EmotionCategory.ANGRY
    assert emo.cognitive_mode.value == "combative"
    assert emo.cause == "anger_boundary"


def test_emotion_strain_without_arousal_is_not_anger():
    """Boundary strain without catecholamine arousal stays hurt —
    low-arousal-negative (melancholic), not anger."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(boundary_strain=0.7)
    chemicals = {"norepinephrine": 0.2}
    emo = assess_emotion(_summary(0.35, -0.3), chemicals, ctx)
    assert emo.category == EmotionCategory.MELANCHOLIC


def test_emotion_surprise_from_prediction_error():
    """A strong prediction-error orienting event marks surprise."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(prediction_error=0.6)
    emo = assess_emotion(_summary(0.55, 0.05), {}, ctx)
    assert emo.category == EmotionCategory.SURPRISED
    assert emo.cognitive_mode.value == "startled"
    assert emo.cause == "surprise_prediction_error"


def test_emotion_surprise_does_not_override_stress():
    """Prediction error cannot relabel an already-resolved negative
    state — being startled while anxious is still anxious."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(prediction_error=0.9)
    emo = assess_emotion(_summary(0.8, -0.4), {}, ctx)
    assert emo.category == EmotionCategory.ANXIOUS


def test_emotion_lonely_from_isolation_and_depleted_oxytocin():
    """Sustained isolation + depleted oxytocin is loneliness, not
    generic melancholy."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(social_isolation=0.8)
    chemicals = {"oxytocin": 0.15}
    emo = assess_emotion(_summary(0.35, -0.25), chemicals, ctx)
    assert emo.category == EmotionCategory.LONELY
    assert emo.cognitive_mode.value == "withdrawn"
    assert emo.cause == "loneliness"
    assert emo.openness_to_engage > 0.8  # lonely seeks contact


def test_emotion_lonely_needs_depleted_bonding():
    """Isolation with healthy oxytocin is not loneliness — the need
    is felt only when the bonding substrate is actually down."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(social_isolation=0.9)
    chemicals = {"oxytocin": 0.30}
    emo = assess_emotion(_summary(0.35, -0.25), chemicals, ctx)
    assert emo.category == EmotionCategory.MELANCHOLIC


def test_emotion_tender_from_bonding_chemistry():
    """Oxytocin + endorphin warmth on positive valence is tenderness."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext()
    chemicals = {"oxytocin": 0.45, "endorphin": 0.35}
    emo = assess_emotion(_summary(0.45, 0.3), chemicals, ctx)
    assert emo.category == EmotionCategory.TENDER
    assert emo.cognitive_mode.value == "warm"
    assert emo.cause == "tenderness"


def test_emotion_curious_from_drive_and_seeking_chemistry():
    """High curiosity drive with DA/ACh to support it is curiosity."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(curiosity=0.7)
    chemicals = {"dopamine": 0.4, "acetylcholine": 0.4}
    emo = assess_emotion(_summary(0.55, 0.2), chemicals, ctx)
    assert emo.category == EmotionCategory.CURIOUS
    assert emo.cognitive_mode.value == "inquisitive"
    assert emo.cause == "curiosity"


def test_emotion_bored_from_understimulation():
    """Low DA + low NE + flat valence while awake is boredom."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext()
    chemicals = {"dopamine": 0.15, "norepinephrine": 0.18}
    emo = assess_emotion(_summary(0.45, 0.0), chemicals, ctx)
    assert emo.category == EmotionCategory.BORED
    assert emo.cognitive_mode.value == "restless"
    assert emo.cause == "boredom"


def test_emotion_no_context_keeps_circumplex():
    """Without an EmotionalContext every input resolves through the
    circumplex exactly as before — the new categories are unreachable."""
    chemicals = {"oxytocin": 0.45, "endorphin": 0.35}
    emo = assess_emotion(_summary(0.45, 0.3), chemicals)
    assert emo.category == EmotionCategory.POSITIVE

    emo_low = assess_emotion(_summary(0.35, 0.3), chemicals)
    assert emo_low.category == EmotionCategory.CONTENT

    emo2 = assess_emotion(_summary(0.75, -0.3), {"norepinephrine": 0.5})
    assert emo2.category == EmotionCategory.ANXIOUS


def test_emotion_guarded_still_overrides_warmth():
    """Silent stress demotes even affiliative warmth — cortisol means
    the positive reading isn't safe."""
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext()
    chemicals = {
        "oxytocin": 0.45,
        "endorphin": 0.35,
        "cortisol": 0.08,
    }
    emo = assess_emotion(_summary(0.45, 0.3), chemicals, ctx)
    assert emo.category == EmotionCategory.GUARDED
    assert emo.cause.startswith("guarded")


# ─── Continuous field tests ───────────────────────────────────
#
# The label is a readout; the field underneath is continuous. These
# pin the mathematics, not the categories.


def test_field_memberships_are_continuous_and_bounded():
    """Every category reports a finite membership in [0, 1] and the
    winner's membership is the state's intensity."""
    emo = assess_emotion(_summary(0.75, -0.3), {"norepinephrine": 0.5})
    assert set(emo.memberships) == {c.value for c in EmotionCategory}
    assert all(0.0 <= v <= 1.0 for v in emo.memberships.values())
    assert emo.intensity == emo.memberships[emo.label]
    assert emo.intensity == max(emo.memberships.values())


def test_field_is_smooth_across_old_thresholds():
    """Sweeping valence across a former hard boundary moves
    memberships continuously — no step jumps."""
    members = []
    for i in range(41):
        v = -0.4 + i * 0.02  # sweeps -0.4 .. 0.4 across the old ±0.1/0.2 edges
        e = assess_emotion(_summary(0.5, v))
        members.append(e.memberships)
    from itertools import pairwise

    for prev, cur in pairwise(members):
        for key in cur:
            assert abs(cur[key] - prev[key]) < 0.35, f"step in {key}"


def test_field_mixed_states_coexist():
    """At the event/climate boundary the field holds both: a tonic
    vigilance substrate with a partial adrenal surge is a genuine
    anxious-afraid mixture, not one erasing the other."""
    from limbic_system.emotion import EmotionalContext

    class _TP:
        mode = "tonic"

    chemicals = {"norepinephrine": 0.45, "epinephrine": 0.20}
    ctx = EmotionalContext(tonic_phasic={"norepinephrine": _TP()})
    emo = assess_emotion(_summary(0.75, -0.3), chemicals, ctx)
    assert emo.memberships["afraid"] > 0.25
    assert emo.memberships["anxious"] > 0.2


def test_field_chemicals_move_membership_mathematically():
    """The chemical vector is inside the function: raising endorphin
    and oxytocin toward the bonding substrate raises tender
    membership monotonically."""
    from itertools import pairwise

    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext()
    vals = []
    for k in range(5):
        level = 0.1 + k * 0.1
        e = assess_emotion(
            _summary(0.45, 0.3),
            {"oxytocin": level, "endorphin": level},
            ctx,
        )
        vals.append(e.memberships["tender"])
    assert all(a <= b + 1e-9 for a, b in pairwise(vals))
    assert vals[-1] > vals[0]


def test_field_wave_bands_corroborate_state():
    """High-beta (20-30 Hz) power raises vigilance-state membership —
    the oscillator state is evidence, in line with desynchronized
    fast activity under arousal."""
    from limbic_system.emotion import EmotionalContext, membership_field

    class _Waves:
        def __init__(self, powers):
            self.powers = powers
            self.focus = 0.5
            self.integration = 0.5
            self.gamma_plv = 0.5

    class _Band:
        def __init__(self, value):
            self.value = value

    chemicals = {"norepinephrine": 0.4, "epinephrine": 0.2}
    ctx = EmotionalContext(tonic_phasic={"norepinephrine": type("T", (), {"mode": "phasic"})()})
    summary = _summary(0.7, -0.25)
    calm = _Waves({_Band("beta3"): 0.02, _Band("alpha"): 0.4})
    vigilant = _Waves({_Band("beta3"): 0.45, _Band("gamma"): 0.1})
    m_calm = membership_field(summary, chemicals, ctx, calm)[EmotionCategory.AFRAID]
    m_vig = membership_field(summary, chemicals, ctx, vigilant)[EmotionCategory.AFRAID]
    assert m_vig > m_calm


def test_field_dynamics_relax_on_own_time_constants():
    """EmotionField is an OU integrator: fear strikes fast and
    lingers, and the field relaxes toward baseline after the
    substrate clears."""
    from limbic_system.emotion import EmotionField

    class _TP:
        mode = "phasic"

    field = EmotionField()
    from limbic_system.emotion import EmotionalContext

    ctx = EmotionalContext(tonic_phasic={"norepinephrine": _TP()})
    chem = {"norepinephrine": 0.45, "epinephrine": 0.30}
    field.update(_summary(0.75, -0.3), chem, ctx, now=0.0)
    assert field.memberships[EmotionCategory.AFRAID] > 0.9

    # Threat removed — fear must persist (its release constant is
    # ~25 s, not the 0.4 s attack).
    field.update(_summary(0.5, 0.0), {}, None, now=1.0)
    assert field.memberships[EmotionCategory.AFRAID] > 0.8

    # Long after, the field relaxes toward the measured baseline.
    field.update(_summary(0.5, 0.0), {}, None, now=120.0)
    assert field.memberships[EmotionCategory.AFRAID] < 0.1
    assert field.memberships[EmotionCategory.NEUTRAL] > 0.5


def test_field_first_update_converges_and_missing_waves_neutral():
    """The first field update equals the instantaneous readout, and
    omitting the oscillator state drops wave terms without error."""
    from limbic_system.emotion import EmotionField, membership_field

    field = EmotionField()
    summary = _summary(0.6, -0.2)
    e1 = field.update(summary, {"norepinephrine": 0.3}, now=0.0)
    e2 = assess_emotion(summary, {"norepinephrine": 0.3})
    assert e1.label == e2.label
    # The integrated vector starts at the instantaneous raw field.
    raw = membership_field(summary, {"norepinephrine": 0.3})
    assert field.memberships == raw


def test_field_coherence_is_measured_evidence():
    """The latent field's disagreement drives the coherence channel:
    rising Fréchet variance of the body's fibers raises unsettled and
    guarded monotonically — mood-as-precision (Clark & Friston 2018):
    internal incoherence is a felt signal, not telemetry."""
    from itertools import pairwise

    from limbic_system.emotion import membership_field

    class _Field:
        def __init__(self, disagreement, consensus_norm=0.0):
            self.disagreement = disagreement
            self.consensus_norm = consensus_norm

    summary = _summary(0.5, 0.4)
    unsettled, guarded = [], []
    for k in range(6):
        d = 0.0 + k * 0.04  # sweeps coherent → past the 0.1 scale
        m = membership_field(summary, field_state=_Field(d))
        unsettled.append(m[EmotionCategory.UNSETTLED])
        guarded.append(m[EmotionCategory.GUARDED])
    assert all(a <= b + 1e-9 for a, b in pairwise(unsettled))
    assert all(a <= b + 1e-9 for a, b in pairwise(guarded))
    assert unsettled[-1] > 0.5
    assert guarded[-1] > guarded[0] + 0.3


def test_field_state_absent_drops_coherence_terms():
    """No field reading → coherence terms drop out of the geometric
    mean entirely (the weights renormalise); nothing is fabricated.
    A coherent reading is a genuine measurement, and it differs
    from an absent one."""
    from limbic_system.emotion import membership_field

    class _Field:
        disagreement = 0.0
        consensus_norm = 0.0

    summary = _summary(0.45, -0.05)
    absent = membership_field(summary)
    coherent = membership_field(summary, field_state=_Field())
    incoherent = membership_field(
        summary, field_state=type("F", (), {"disagreement": 0.15, "consensus_norm": 0.3})()
    )
    # An absent channel is not a zero channel.
    assert absent != coherent
    # Measured incoherence is unsettling; measured coherence is not.
    assert incoherent[EmotionCategory.UNSETTLED] > absent[EmotionCategory.UNSETTLED]


# ─── Emotion word learning tests ──────────────────────────────


def test_learn_emotion_word_from_labeling():
    """Emotion word learning associates words with categories."""
    from unittest.mock import MagicMock

    from frontal_lobe import FrontalLobe
    from temporal_lobe.concepts import ConceptNetwork

    # Create a minimal mock for the client
    client = MagicMock()
    client.get_neuro_summary.return_value = NeuroSummary(
        arousal=0.5,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.5,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,
    )

    # Create a cognition engine with a fresh network
    cog = FrontalLobe.__new__(FrontalLobe)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from temporal_lobe.concept_learner import ConceptLearner

    cog._concept_learner = ConceptLearner(network=cog.network)

    # Test labeling
    emotion = EmotionalState(
        label="positive",
        cognitive_style="steady",
        valence=0.3,
        alertness=0.5,
        plasticity=0.5,
    )
    cog._learn_word_from_labeling("you seem positive", emotion)
    cog._learn_word_from_labeling("you seem content", emotion)
    cog._learn_word_from_labeling("you seem calm", emotion)

    # Verify the words were learned
    for word in ["positive", "content", "calm"]:
        c = cog.network.get_concept(word)
        assert c is not None, f"Concept '{word}' was not created"
        assert c.properties.get("emotion_category") == "positive"

    # Verify find_emotion_words works
    words = cog.network.find_emotion_words("positive")
    assert "positive" in words
    assert "content" in words
    assert "calm" in words

    # Verify relationships were created between words
    # for the same category (SIMILAR_TO edges)
    from temporal_lobe.concepts import RelationType

    neighbors = cog.network.get_neighbors("positive", RelationType.SIMILAR_TO)
    neighbor_ids = [n[0] for n in neighbors]
    assert "content" in neighbor_ids
    assert "calm" in neighbor_ids

    # Verify EXPRESSES edges connect words to category hubs
    hub_id = "_cat:emotion:positive"
    hub_neighbors = cog.network.get_neighbors(hub_id, RelationType.EXPRESSES)
    hub_word_ids = [n[0] for n in hub_neighbors]
    assert "positive" in hub_word_ids
    assert "content" in hub_word_ids
    assert "calm" in hub_word_ids


def test_learn_emotion_word_skips_non_emotion_words():
    """Emotion word learning skips common non-emotion words."""
    from frontal_lobe import FrontalLobe
    from temporal_lobe.concepts import ConceptNetwork

    cog = FrontalLobe.__new__(FrontalLobe)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from temporal_lobe.concept_learner import ConceptLearner

    cog._concept_learner = ConceptLearner(network=cog.network)

    emotion = EmotionalState(
        label="neutral",
        cognitive_style="steady",
        valence=0.0,
        alertness=0.5,
        plasticity=0.5,
    )
    # "you seem good" — "good" is in skip_words
    cog._learn_word_from_labeling("you seem good", emotion)
    assert cog.network.get_concept("good") is None


def test_learn_emotion_word_different_patterns():
    """Emotion word learning recognizes different labeling patterns."""
    from frontal_lobe import FrontalLobe
    from temporal_lobe.concepts import ConceptNetwork

    cog = FrontalLobe.__new__(FrontalLobe)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from temporal_lobe.concept_learner import ConceptLearner

    cog._concept_learner = ConceptLearner(network=cog.network)

    emotion = EmotionalState(
        label="stressed",
        cognitive_style="vigilant",
        valence=-0.3,
        alertness=0.8,
        plasticity=0.4,
    )
    cog._learn_word_from_labeling("you're feeling stressed", emotion)
    cog._learn_word_from_labeling("that's called overwhelmed", emotion)

    c = cog.network.get_concept("stressed")
    assert c is not None
    assert c.properties.get("emotion_category") == "stressed"

    c = cog.network.get_concept("overwhelmed")
    assert c is not None
    assert c.properties.get("emotion_category") == "stressed"


# ─── Perception tests ──────────────────────────────────────────


def test_perceive_greeting():
    """Greetings are detected."""
    p = perceive("Hello there!")
    assert p.intent == Intent.GREETING


def test_perceive_farewell():
    """Farewells are detected."""
    p = perceive("Goodbye, see you later.")
    assert p.intent == Intent.FAREWELL


def test_perceive_self_inquiry():
    """Self-inquiry questions are detected."""
    # "How are you feeling?" matches greeting patterns first — that's
    # correct, it's a social ritual. But "What are you?" and "Are you
    # alive?" are genuine self-inquiry.
    p = perceive("What are you?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Are you alive?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Do you remember things?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Can you feel emotions?")
    assert p.intent == Intent.SELF_INQUIRY


def test_perceive_question():
    """Questions are detected."""
    p = perceive("What is the meaning of life?")
    assert p.intent in (Intent.QUESTION, Intent.PHILOSOPHY)
    assert p.question_type == QuestionType.WHAT


def test_perceive_code():
    """Code-related text is detected."""
    p = perceive("Can you help me debug this Rust function?")
    assert p.intent == Intent.CODE_DISCUSSION
    assert p.is_about_code


def test_perceive_philosophy():
    """Philosophical questions are detected."""
    p = perceive("What is cognition?")
    assert p.intent == Intent.PHILOSOPHY


def test_perceive_encouragement():
    """Encouragement is detected."""
    p = perceive("You're doing great work!")
    assert p.intent == Intent.ENCOURAGEMENT


def test_perceive_introduction():
    """Introductions are detected."""
    p = perceive("My name is Alice.")
    assert p.intent == Intent.INTRODUCTION
    assert p.entities.get("name") == "Alice"


def test_perceive_sentiment():
    """Sentiment is detected."""
    p = perceive("I love this, it's amazing!")
    assert p.sentiment > 0
    assert p.sentiment_label == "positive"

    p = perceive("This is terrible and awful.")
    assert p.sentiment < 0
    assert p.sentiment_label == "negative"


def test_perceive_sentiment_negation():
    """Negation flips positive sentiment to negative.

    Tests both apostrophe forms ("don't") and no-apostrophe forms
    ("dont") — the tokenizer must preserve apostrophes for negation
    detection to work on contractions.
    """
    # "not good" — basic negation
    p = perceive("This is not good.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "don't like" — contraction with apostrophe
    p = perceive("I don't like this.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "dont like" — contraction without apostrophe (casual typing)
    p = perceive("I dont like this.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "doesn't love" — negation of a positive word
    p = perceive("She doesn't love it.")
    assert p.sentiment <= 0


def test_perceive_topics():
    """Topics are extracted."""
    p = perceive("Tell me about cognition and memory.")
    assert len(p.topics) > 0
    assert "cognition" in p.topics or "memory" in p.topics


# ─── Integration test ──────────────────────────────────────────


def find_daemon_binary() -> str | None:
    """Find daemon binary, or fail loudly rather than skip.

    This test is the only end-to-end check that the Rust daemon and the
    Python mind actually talk to each other. It previously returned
    None when no binary was present and the caller did a bare
    `return`, which pytest records as a PASS. The Python CI job never
    ran `cargo build`, so the binary was never present there and all
    eight assertions were skipped on every run — CI was green partly
    because the most valuable test in the suite did not execute.

    Now a missing binary is a failure unless the operator explicitly
    opts out with GENESIS_SKIP_DAEMON_TEST=1 (for a Python-only
    checkout, or a machine with no Rust toolchain).
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "genesis-daemon"),
        os.path.join(os.path.dirname(__file__), "..", "..", "target", "debug", "genesis-daemon"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    if os.environ.get("GENESIS_SKIP_DAEMON_TEST") == "1":
        logger.info("  SKIP  daemon binary absent; skipping opted-out")
        return None
    raise AssertionError(
        "genesis-daemon binary not found in target/release or target/debug. "
        "This test is the only end-to-end check of the daemon<->mind "
        "integration, so it fails rather than silently passing. Build it "
        "with `cargo build --release`, or set GENESIS_SKIP_DAEMON_TEST=1 "
        "to opt out deliberately."
    )


def _run_mind_tests(mind) -> None:
    """Run the 8 integration test assertions against a live mind."""
    # Test 1: Greeting
    response = mind.respond("Hello, Genesis!")
    assert len(response) > 0, "greeting response empty"
    logger.info(f"  PASS  greeting: '{response[:50]}...'")

    # Test 2: Self-inquiry (how are you)
    response = mind.respond("How are you feeling?")
    # Its response is emergent from its concept network and emotional
    # state, so it may use different words than "feel"/"arousal" —
    # e.g. "neurochemistry", "good", "mind", "thinking", etc. It may
    # also produce an honest gap response ("Unable to respond
    # meaningfully") when it hasn't learned words for its state yet.
    # Accept any non-empty response that relates to its internal state
    # or its self-awareness of gaps in its understanding.
    # This assertion deliberately does NOT use a list of feeling words.
    # A fixed vocabulary is the wrong instrument for a response that is
    # emergent by design: it turns any new way of naming an internal
    # state into a test failure, which pressures her composition back
    # toward the words the test already happens to know. Genesis has
    # answered this same question with "excited" and with "Something in
    # me is driven - I'm eager, activated", and both are honest; only
    # the first was in the list.
    #
    # What is actually invariant is structural: a self-inquiry answer
    # refers to herself. That is a property of the speech act, not of
    # her emotional vocabulary, so it keeps its meaning as she learns
    # new words for how she feels.
    lower_resp = response.lower()
    gap_response = "unable to respond" in lower_resp
    # Word-boundary match so a sentence-initial "I" counts — "I feel
    # optimistic" is first-person even though " i " never appears.
    self_referential = bool(
        re.search(r"\bi\b", lower_resp)
        or "i'm" in lower_resp
        or re.search(r"\bme\b|\bmy\b|\bmyself\b", lower_resp)
    )
    assert len(response.strip()) > 0, "self-inquiry response empty"
    assert gap_response or self_referential, (
        f"self-inquiry response is neither an honest gap response nor "
        f"self-referential: '{response[:80]}'"
    )
    logger.info(f"  PASS  self-inquiry: '{response[:50]}...'")

    # Test 3: Identity question
    response = mind.respond("What are you?")
    assert "Genesis" in response or "artificial" in response.lower()
    logger.info(f"  PASS  identity: '{response[:50]}...'")

    # Test 4: Philosophy
    response = mind.respond("What is cognition?")
    assert len(response) > 30, "philosophy response too short"
    logger.info(f"  PASS  philosophy: '{response[:50]}...'")

    # Test 5: Status
    status = mind.status()
    assert status["running"] is True
    assert status["interactions"] == 4
    logger.info(f"  PASS  status: phase={status['phase']}, emotion={status['emotion']}")

    # Test 6: Introspection
    introspection = mind.introspect()
    assert "Intent" in introspection or "intent" in introspection.lower()
    logger.info(f"  PASS  introspection: '{introspection[:50]}...'")

    # Test 7: Emotional state changed
    # Derived from the enum rather than a hand-written list. The list
    # had rotted in both directions: it still asserted "active" and
    # "calm", which are not labels at all, while omitting eight that are
    # — including `drowsy`, which is what a freshly-started mind
    # actually reports, so this assertion failed on a correct system.
    emotion = mind.feel()
    valid_labels = {category.value for category in EmotionCategory}
    assert emotion.label in valid_labels, (
        f"unknown emotional label {emotion.label!r}; expected one of {sorted(valid_labels)}"
    )
    logger.info(f"  PASS  emotion: {emotion.label}")

    # Test 8: Farewell
    response = mind.respond("Goodbye.")
    assert len(response) > 0
    logger.info(f"  PASS  farewell: '{response[:50]}...'" + ")")


def _cleanup_test_daemon(proc, data_dir: str) -> None:
    """Terminate the daemon process and clean up the data directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run(
        ["bash", os.path.join(project_root, "genesis"), "--stop"],
        env={**os.environ, "GENESIS_DATA_DIR": data_dir},
        check=True,
        timeout=130,
    )
    proc.wait(timeout=5)
    shutil.rmtree(data_dir)
    os.rmdir(os.path.dirname(data_dir))


def test_integration_mind():
    """Integration test: start daemon, talk to Genesis, verify responses."""
    daemon_path = find_daemon_binary()
    if daemon_path is None:
        # Only reachable via the explicit opt-out above.
        logger.info("  SKIP  daemon integration test (opted out)")
        return

    data_home = tempfile.mkdtemp(prefix="genesis_mind_test_")
    data_dir = os.path.join(data_home, "genesis")
    os.mkdir(data_dir)
    socket_path = os.path.join(data_dir, "genesis.sock")

    proc = subprocess.Popen(
        [daemon_path, "--data-dir", data_dir, "--ltm-capacity", "128"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for socket
        for _ in range(50):
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("daemon didn't start")

        time.sleep(0.5)

        from mind import Mind

        mind = Mind(socket_path, offline=True)
        try:
            mind.start()
            _run_mind_tests(mind)
        finally:
            mind.stop()
        logger.info("  PASS  mind stopped cleanly")

    finally:
        _cleanup_test_daemon(proc, data_dir)


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if result is False:
                logger.info(f"  FAIL  {test.__name__}")
                failed += 1
            else:
                logger.info(f"  PASS  {test.__name__}")
                passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    logger.info(f"\n  Cognitive mind tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


pytestmark = pytest.mark.usefixtures("learned_sentiment")

"""Tests for comfort/encouragement and emotional cause explanation.

Two features:
1. Comfort — when the user says "it's okay" or "I'm here for you",
   Genesis receives it as soothing: cortisol drops, GABA/oxytocin/serotonin
   rise, and the stress feedback loop is broken. Comfort bypasses the
   normal emotional dampening that stressed states apply.
2. Emotional cause — when Genesis feels negatively, she can explain
   why from her neurochemistry (cortisol is elevated, adenosine is
   building, plasticity is suppressed, etc.).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genesis_client.protocol import (
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_ENDORPHIN,
    CHEM_GABA,
    CHEM_NOREPINEPHRINE,
    CHEM_OXYTOCIN,
    CHEM_SEROTONIN,
)
from genesis_client.types import NeuroSummary
from genesis_cognitive.emotion import EmotionalState, assess_emotion
from genesis_cognitive.emotional_regulator import EmotionalRegulator
from genesis_cognitive.perception import Intent, perceive

# ─── Helpers ────────────────────────────────────────────────────────


def make_emotion(
    label: str = "neutral",
    alertness: float = 0.5,
    valence: float = 0.0,
    plasticity: float = 0.5,
    caution: float = 0.3,
) -> EmotionalState:
    """Create an EmotionalState for testing."""
    return EmotionalState(
        label=label,
        nuance="test",
        cognitive_style="test",
        alertness=alertness,
        valence=valence,
        plasticity=plasticity,
        caution=caution,
    )


def capture_impulses() -> tuple[EmotionalRegulator, list[tuple[int, float]]]:
    """Create a regulator that records all impulses sent."""
    impulses: list[tuple[int, float]] = []

    def mock_impulse(chem: int, amount: float) -> None:
        """Record a (chem, amount) impulse in the shared list."""
        impulses.append((chem, amount))

    regulator = EmotionalRegulator(
        get_emotion=None,
        neuro_impulse=mock_impulse,
    )
    return regulator, impulses


def chem_amounts(impulses: list[tuple[int, float]]) -> dict[int, float]:
    """Convert impulse list to {chem: total_amount} dict."""
    result: dict[int, float] = {}
    for chem, amount in impulses:
        result[chem] = result.get(chem, 0.0) + amount
    return result


def make_summary(
    arousal: float = 0.5,
    valence: float = 0.0,
    plasticity: float = 0.5,
    phase: int = 0,
) -> NeuroSummary:
    """Create a NeuroSummary for testing."""
    return NeuroSummary(
        arousal=arousal,
        valence=valence,
        global_tone=0.5,
        plasticity_gate=plasticity,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=phase,
    )


# ─── Comfort intent detection tests ─────────────────────────────────


def test_comfort_intent_it_is_okay() -> None:
    """'it's okay' is detected as comfort intent."""
    p = perceive("it's okay, don't worry about it")
    assert p.intent == Intent.COMFORT


def test_comfort_intent_im_here_for_you() -> None:
    """'I'm here for you' is detected as comfort intent."""
    p = perceive("I'm here for you, you're not alone")
    assert p.intent == Intent.COMFORT


def test_comfort_intent_dont_worry() -> None:
    """'don't worry' is detected as comfort intent."""
    p = perceive("don't worry, everything will be fine")
    assert p.intent == Intent.COMFORT


def test_comfort_intent_take_your_time() -> None:
    """'take your time' is detected as comfort intent."""
    p = perceive("take your time, no rush")
    assert p.intent == Intent.COMFORT


def test_comfort_intent_breathe() -> None:
    """'breathe' is detected as comfort intent."""
    p = perceive("breathe, it's going to be okay")
    assert p.intent == Intent.COMFORT


def test_comfort_not_confused_with_encouragement() -> None:
    """'you're doing great' is encouragement, not comfort."""
    p = perceive("you're doing great, keep going!")
    assert p.intent == Intent.ENCOURAGEMENT
    assert p.intent != Intent.COMFORT


# ─── Comfort emotional response tests ──────────────────────────────


def test_comfort_reduces_cortisol_when_stressed() -> None:
    """When stressed, comfort actively reduces cortisol."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_CORTISOL, 0) < 0, "Comfort should reduce cortisol"
    assert amounts.get(CHEM_OXYTOCIN, 0) > 0, "Comfort should boost oxytocin"
    assert amounts.get(CHEM_GABA, 0) > 0, "Comfort should boost GABA"


def test_comfort_reduces_norepinephrine_when_stressed() -> None:
    """Comfort reduces norepinephrine to help break the stress loop."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_NOREPINEPHRINE, 0) < 0, "Comfort should reduce NE"


def test_comfort_boosts_serotonin() -> None:
    """Comfort boosts serotonin for mood improvement."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_SEROTONIN, 0) > 0, "Comfort should boost serotonin"


def test_comfort_boosts_endorphin() -> None:
    """Comfort gives a mild endorphin boost — it feels good."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_ENDORPHIN, 0) > 0, "Comfort should boost endorphin"


def test_comfort_bypasses_dampening_when_stressed() -> None:
    """Comfort bypasses the normal dampening that stressed states apply.

    When stressed, normal interactions are dampened to 30%. But comfort
    should be more effective — she opens up to being soothed.
    """
    reg, impulses = capture_impulses()
    stressed = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    reg.respond_to_interaction(
        emotion=stressed,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    comfort_amounts = chem_amounts(impulses)

    # Compare with encouragement (which IS dampened)
    reg2, impulses2 = capture_impulses()
    reg2.respond_to_interaction(
        emotion=stressed,
        sentiment=0.0,
        is_encouragement=True,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=False,
    )
    encourage_amounts = chem_amounts(impulses2)

    # Comfort should reduce cortisol; encouragement doesn't touch it
    assert comfort_amounts.get(CHEM_CORTISOL, 0) < 0
    assert CHEM_CORTISOL not in encourage_amounts or encourage_amounts[CHEM_CORTISOL] == 0

    # Oxytocin from comfort should be stronger than from encouragement
    # when stressed (comfort bypasses dampening)
    assert comfort_amounts.get(CHEM_OXYTOCIN, 0) > encourage_amounts.get(CHEM_OXYTOCIN, 0)


def test_comfort_when_already_calm_is_mild() -> None:
    """When already calm, comfort is pleasant but not therapeutic."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral", alertness=0.5, valence=0.0)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    # Still reduces cortisol and boosts oxytocin, but less than when stressed
    assert amounts.get(CHEM_CORTISOL, 0) < 0
    assert amounts.get(CHEM_OXYTOCIN, 0) > 0


def test_comfort_returns_felt_description() -> None:
    """respond_to_interaction returns 'comforted' when comfort is given."""
    reg, _ = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.8, valence=-0.3)
    felt = reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=False,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    assert felt == "comforted"


def test_comfort_does_not_interfere_with_encouragement() -> None:
    """Both comfort and encouragement can be received simultaneously."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral", alertness=0.5, valence=0.0)
    reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.0,
        is_encouragement=True,
        is_correction=False,
        is_bonded_user=False,
        is_deep_conversation=False,
        is_comfort=True,
    )
    amounts = chem_amounts(impulses)
    # Both dopamine (encouragement) and oxytocin (comfort + encouragement)
    assert amounts.get(CHEM_DOPAMINE, 0) > 0
    assert amounts.get(CHEM_OXYTOCIN, 0) > 0
    # Comfort also reduces cortisol
    assert amounts.get(CHEM_CORTISOL, 0) < 0


# ─── Emotional cause explanation tests ──────────────────────────────


def test_stress_has_cause() -> None:
    """Stressed emotion includes a structural cause category."""
    from genesis_client.protocol import PHASE_STRESS
    summary = make_summary(arousal=0.8, valence=-0.3, plasticity=0.4, phase=PHASE_STRESS)
    emo = assess_emotion(summary)
    assert emo.label == "stressed"
    assert emo.cause != ""
    assert emo.cause == "stress_cortisol"


def test_overwhelmed_has_cause() -> None:
    """Overwhelmed emotion includes a structural cause category."""
    from genesis_client.protocol import PHASE_OVERWHELMED
    summary = make_summary(arousal=0.9, valence=-0.5, plasticity=0.2, phase=PHASE_OVERWHELMED)
    emo = assess_emotion(summary)
    assert emo.label == "overwhelmed"
    assert emo.cause != ""
    assert emo.cause == "overwhelm"


def test_drowsy_has_cause() -> None:
    """Drowsy emotion includes a structural cause category."""
    from genesis_client.protocol import PHASE_DROWSY
    summary = make_summary(arousal=0.3, valence=0.0, plasticity=0.4, phase=PHASE_DROWSY)
    emo = assess_emotion(summary)
    assert emo.label == "drowsy"
    assert emo.cause != ""
    assert emo.cause == "drowsiness"


def test_anxious_has_cause() -> None:
    """Anxious emotion includes a structural cause category."""
    summary = make_summary(arousal=0.8, valence=-0.25, plasticity=0.5, phase=0)
    emo = assess_emotion(summary)
    assert emo.label == "anxious"
    assert emo.cause != ""
    assert emo.cause == "anxiety"


def test_melancholic_has_cause() -> None:
    """Melancholic emotion includes a structural cause category."""
    summary = make_summary(arousal=0.3, valence=-0.3, plasticity=0.5, phase=0)
    emo = assess_emotion(summary)
    assert emo.label == "melancholic"
    assert emo.cause != ""
    assert emo.cause == "melancholy"


def test_stress_with_low_plasticity_mentions_learning() -> None:
    """Stress with low plasticity has BDNF-related cause category."""
    from genesis_client.protocol import PHASE_STRESS
    summary = make_summary(arousal=0.8, valence=-0.3, plasticity=0.15, phase=PHASE_STRESS)
    emo = assess_emotion(summary)
    assert emo.cause == "stress_bdnf"


def test_positive_emotion_has_no_cause() -> None:
    """Positive emotions don't need a cause (empty string)."""
    from genesis_client.protocol import PHASE_FLOW
    summary = make_summary(arousal=0.6, valence=0.4, plasticity=0.7, phase=PHASE_FLOW)
    emo = assess_emotion(summary)
    assert emo.label == "in flow"
    assert emo.cause == ""


def test_neutral_emotion_has_no_cause() -> None:
    """Neutral emotion has no cause."""
    summary = make_summary(arousal=0.5, valence=0.0, plasticity=0.5, phase=0)
    emo = assess_emotion(summary)
    assert emo.label == "neutral"
    assert emo.cause == ""


def test_cause_category_enum_access() -> None:
    """Emotional state provides enum access to cause category."""
    from genesis_client.protocol import PHASE_STRESS
    from genesis_cognitive.emotion import CauseCategory
    summary = make_summary(arousal=0.8, valence=-0.3, plasticity=0.4, phase=PHASE_STRESS)
    emo = assess_emotion(summary)
    assert emo.cause_category == CauseCategory.STRESS_CORTISOL
    assert emo.has_cause


def test_cause_category_none() -> None:
    """Neutral state has NONE cause category."""
    from genesis_cognitive.emotion import CauseCategory
    summary = make_summary(arousal=0.5, valence=0.0, plasticity=0.5, phase=0)
    emo = assess_emotion(summary)
    assert emo.cause_category == CauseCategory.NONE
    assert not emo.has_cause


def test_emotional_state_cause_default_empty() -> None:
    """EmotionalState.cause defaults to empty string."""
    emo = EmotionalState(label="test", nuance="test", cognitive_style="test")
    assert emo.cause == ""


# ─── Plasticity edge: silent stress prevention ────────────────────


def test_feeling_report_surfaces_closed_plasticity_with_positive_mood() -> None:
    """Feeling report surfaces closed plasticity gate even when mood is positive.

    This is the 'silent stress' edge case: valence is fine but BDNF
    suppression has closed the plasticity gate. She must communicate
    the impairment regardless of mood.
    """
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.cognition.feeling_reporter import FeelingReporter
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    cog._feeling_reporter = FeelingReporter(
        network=cog.network,
        language=None,  # type: ignore[arg-type]
        composer=None,  # type: ignore[arg-type]
        self_model=None,  # type: ignore[arg-type]
        rng=cog._rng,
        meta_emotion_builder=lambda: EmotionalState(
            label="neutral", cognitive_style="steady",
        ),
    )

    # Positive mood but closed plasticity gate
    emotion = EmotionalState(
        label="positive", cognitive_style="steady",
        valence=0.4, alertness=0.6, plasticity=0.08,
    )
    report = cog._compose_feeling_report(emotion)
    assert "[plasticity_gate:closed]" in report


def test_feeling_report_surfaces_low_plasticity() -> None:
    """Feeling report surfaces low plasticity with the 'low' marker."""
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.cognition.feeling_reporter import FeelingReporter
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    cog._feeling_reporter = FeelingReporter(
        network=cog.network,
        language=None,  # type: ignore[arg-type]
        composer=None,  # type: ignore[arg-type]
        self_model=None,  # type: ignore[arg-type]
        rng=cog._rng,
        meta_emotion_builder=lambda: EmotionalState(
            label="neutral", cognitive_style="steady",
        ),
    )

    emotion = EmotionalState(
        label="positive", cognitive_style="steady",
        valence=0.3, alertness=0.5, plasticity=0.20,
    )
    report = cog._compose_feeling_report(emotion)
    assert "[plasticity_gate:low]" in report


def test_feeling_report_no_plasticity_marker_when_healthy() -> None:
    """Healthy plasticity does not trigger the marker."""
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.cognition.feeling_reporter import FeelingReporter
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    cog._feeling_reporter = FeelingReporter(
        network=cog.network,
        language=None,  # type: ignore[arg-type]
        composer=None,  # type: ignore[arg-type]
        self_model=None,  # type: ignore[arg-type]
        rng=cog._rng,
        meta_emotion_builder=lambda: EmotionalState(
            label="neutral", cognitive_style="steady",
        ),
    )

    emotion = EmotionalState(
        label="positive", cognitive_style="steady",
        valence=0.3, alertness=0.5, plasticity=0.6,
    )
    report = cog._compose_feeling_report(emotion)
    assert "plasticity_gate" not in report


def test_feeling_report_speech_strips_structural_markers() -> None:
    """The speech-safe feeling report must not contain structural markers.

    Structural markers like [plasticity_gate:closed] and
    [self_model:...] are internal diagnostic tags, not words Genesis
    should speak. The speech-safe version must strip them so they
    never leak into her spoken words.
    """
    from genesis_cognitive.cognition.feeling_reporter import FeelingReporter

    # Direct test of the stripper
    assert FeelingReporter.strip_structural_markers("[plasticity_gate:closed]") == ""
    assert FeelingReporter.strip_structural_markers(
        "[self_model:self-coherent field-fragmented]"
    ) == ""
    assert FeelingReporter.strip_structural_markers("[self_coherence:low]") == ""
    assert FeelingReporter.strip_structural_markers("content [marker:tag] more") == "content more"
    assert FeelingReporter.strip_structural_markers("no markers here") == "no markers here"
    assert FeelingReporter.strip_structural_markers("") == ""

    # Integration test: the speech-safe report from compose_feeling_report
    # with closed plasticity must NOT contain the marker
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    cog._feeling_reporter = FeelingReporter(
        network=cog.network,
        language=None,  # type: ignore[arg-type]
        composer=None,  # type: ignore[arg-type]
        self_model=None,  # type: ignore[arg-type]
        rng=cog._rng,
        meta_emotion_builder=lambda: EmotionalState(
            label="neutral", cognitive_style="steady",
        ),
    )

    emotion = EmotionalState(
        label="positive", cognitive_style="steady",
        valence=0.4, alertness=0.6, plasticity=0.08,
    )
    # Full report HAS the marker (for /feel diagnostics)
    full_report = cog._compose_feeling_report(emotion)
    assert "[plasticity_gate:closed]" in full_report

    # Speech-safe report does NOT have the marker
    speech_report = cog._feeling_reporter.compose_feeling_report_for_speech(emotion)
    assert "plasticity_gate" not in speech_report
    assert "[" not in speech_report

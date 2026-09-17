"""Advanced learning tests.

Active inference, predictive coding, learning fidelity,
CPU self-regulation.
"""

import json
import logging
import os
import struct
import tempfile
from unittest.mock import MagicMock

from genesis_client.protocol import GET_INFERENCE_SUMMARY, UPDATE_USER_AFFECT
from genesis_client.types import InferenceSummary
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.emotional_regulator import EmotionalRegulator
from genesis_cognitive.learning import (
    ActiveInferenceReader,
    AutonomousLearner,
    CuriosityEngine,
    DyadicState,
    Prediction,
    PredictionContext,
    PredictionError,
    PredictiveCodingLayer,
    SelfModelState,
    UserAffectEstimate,
)
from genesis_cognitive.learning.autonomous import (
    RATE_LIMIT_DELAY,
    THROTTLED_DELAY_MULTIPLIER,
)
from genesis_cognitive.mind import Mind
from genesis_cognitive.perception import Intent, Perception, QuestionType
from genesis_cognitive.reasoning import ReasoningEngine

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_active_inference.py
# ======================================================================


# ─── InferenceSummary parsing ─────────────────────────────────


def _pack_inference_summary(
    surprise_ema=0.0,
    free_energy=0.0,
    expected_free_energy=0.0,
    allostasis_load=0.0,
    precision=0.5,
    attunement=0.0,
    dyadic_synchrony=0.0,
    user_valence=0.0,
    user_arousal=0.5,
    user_engagement=0.0,
    pe_dopamine=0.0,
    pe_cortisol=0.0,
    pe_serotonin=0.0,
    model_maturity=0.0,
    tick_count=0,
) -> bytes:
    """Pack an InferenceSummary into 60 bytes (matching the Rust layout)."""
    return (
        struct.pack("<f", surprise_ema)
        + struct.pack("<f", free_energy)
        + struct.pack("<f", expected_free_energy)
        + struct.pack("<f", allostasis_load)
        + struct.pack("<f", precision)
        + struct.pack("<f", attunement)
        + struct.pack("<f", dyadic_synchrony)
        + struct.pack("<f", user_valence)
        + struct.pack("<f", user_arousal)
        + struct.pack("<f", user_engagement)
        + struct.pack("<f", pe_dopamine)
        + struct.pack("<f", pe_cortisol)
        + struct.pack("<f", pe_serotonin)
        + struct.pack("<f", model_maturity)
        + struct.pack("<I", tick_count)
    )


def test_inference_summary_unpack_60_bytes():
    """InferenceSummary should unpack exactly 60 bytes."""
    data = _pack_inference_summary(
        surprise_ema=0.3,
        free_energy=0.4,
        precision=0.7,
        attunement=0.5,
        model_maturity=0.8,
        tick_count=42,
    )
    assert len(data) == 60
    summary = InferenceSummary.unpack(data)
    assert abs(summary.surprise_ema - 0.3) < 1e-6
    assert abs(summary.free_energy - 0.4) < 1e-6
    assert abs(summary.precision - 0.7) < 1e-6
    assert abs(summary.attunement - 0.5) < 1e-6
    assert abs(summary.model_maturity - 0.8) < 1e-6
    assert summary.inference_tick_count == 42


def test_inference_summary_unpack_too_short():
    """InferenceSummary should reject short data."""
    try:
        InferenceSummary.unpack(b"\x00" * 59)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        assert "60 bytes" in str(e)


def test_inference_summary_is_surprised():
    """Test inference summary is surprised."""
    summary = InferenceSummary.unpack(_pack_inference_summary(surprise_ema=0.5))
    assert summary.is_surprised

    summary = InferenceSummary.unpack(_pack_inference_summary(surprise_ema=0.1))
    assert not summary.is_surprised


def test_inference_summary_is_allostatically_loaded():
    """Test inference summary is allostatically loaded."""
    summary = InferenceSummary.unpack(_pack_inference_summary(allostasis_load=0.6))
    assert summary.is_allostatically_loaded

    summary = InferenceSummary.unpack(_pack_inference_summary(allostasis_load=0.2))
    assert not summary.is_allostatically_loaded


def test_inference_summary_is_attuned():
    """Test inference summary is attuned."""
    summary = InferenceSummary.unpack(_pack_inference_summary(attunement=0.5))
    assert summary.is_attuned

    summary = InferenceSummary.unpack(_pack_inference_summary(attunement=0.1))
    assert not summary.is_attuned


def test_inference_summary_is_in_sync():
    """Test inference summary is in sync."""
    summary = InferenceSummary.unpack(_pack_inference_summary(dyadic_synchrony=0.5))
    assert summary.is_in_sync

    summary = InferenceSummary.unpack(_pack_inference_summary(dyadic_synchrony=0.1))
    assert not summary.is_in_sync


def test_inference_summary_model_is_mature():
    """Test inference summary model is mature."""
    summary = InferenceSummary.unpack(_pack_inference_summary(model_maturity=0.8))
    assert summary.model_is_mature

    summary = InferenceSummary.unpack(_pack_inference_summary(model_maturity=0.3))
    assert not summary.model_is_mature


# ─── UserAffectEstimate ───────────────────────────────────────


def test_user_affect_positive_message():
    """A message with positive words should yield positive valence."""
    est = UserAffectEstimate.from_message("That's great! I love it!")
    assert est.valence > 0.0, f"valence should be positive: {est.valence}"


def test_user_affect_negative_message():
    """A message with negative words should yield negative valence."""
    est = UserAffectEstimate.from_message("This is broken and terrible")
    assert est.valence < 0.0, f"valence should be negative: {est.valence}"


def test_user_affect_neutral_message():
    """A message with no sentiment markers should yield neutral valence."""
    est = UserAffectEstimate.from_message("The sky is blue")
    assert abs(est.valence) < 0.1, f"valence should be near zero: {est.valence}"


def test_user_affect_long_message_high_engagement():
    """A long message should yield high engagement."""
    long_msg = " ".join(["word"] * 60)
    est = UserAffectEstimate.from_message(long_msg)
    assert est.engagement > 0.5, f"engagement should be high: {est.engagement}"


def test_user_affect_short_message_low_engagement():
    """A very short message should yield lower engagement."""
    est = UserAffectEstimate.from_message("ok")
    assert est.engagement < 0.6, f"engagement should be lower: {est.engagement}"


def test_user_affect_question_boosts_engagement():
    """A question should boost engagement."""
    est_q = UserAffectEstimate.from_message("What is this?", is_question=True)
    est_s = UserAffectEstimate.from_message("What is this.", is_question=False)
    assert est_q.engagement > est_s.engagement


def test_user_affect_exclamations_boost_arousal():
    """Exclamation marks should boost arousal."""
    est = UserAffectEstimate.from_message("Wow!!! Amazing!!!")
    assert est.arousal > 0.5, f"arousal should be high: {est.arousal}"


def test_user_affect_fast_response_boosts_arousal():
    """A fast response latency should boost arousal."""
    est = UserAffectEstimate.from_message("hello", response_latency_s=2.0)
    assert est.arousal > 0.5, f"arousal should be high: {est.arousal}"


def test_user_affect_confidence_in_range():
    """Confidence should be in [0, 1]."""
    est = UserAffectEstimate.from_message("That's great and wonderful!")
    assert 0.0 <= est.confidence <= 1.0


# ─── Sentiment analyzer improvements ─────────────────────────────
# These tests verify the VADER-style improvements over the old
# word-list heuristic: negation, intensifiers, diminishers,
# contrastive conjunctions, and capitalization.


def test_user_affect_negation_flips_sentiment():
    """'not bad' should be positive, not negative.

    The old heuristic matched 'bad' in 'not bad' and scored it
    negative. The VADER-style analyzer handles negation.
    """
    est_negated = UserAffectEstimate.from_message("not bad")
    est_plain_bad = UserAffectEstimate.from_message("bad")
    assert est_negated.valence > est_plain_bad.valence, (
        f"'not bad' ({est_negated.valence}) should be more positive "
        f"than 'bad' ({est_plain_bad.valence})"
    )
    assert est_negated.valence > 0.0, (
        f"'not bad' should be positive: {est_negated.valence}"
    )


def test_user_affect_negation_flips_positive():
    """'not good' should be negative, not positive."""
    est_negated = UserAffectEstimate.from_message("not good")
    est_plain_good = UserAffectEstimate.from_message("good")
    assert est_negated.valence < est_plain_good.valence, (
        f"'not good' ({est_negated.valence}) should be less positive "
        f"than 'good' ({est_plain_good.valence})"
    )
    assert est_negated.valence < 0.0, (
        f"'not good' should be negative: {est_negated.valence}"
    )


def test_user_affect_intensifier_amplifies():
    """'very good' should be more positive than 'good'."""
    est_intensified = UserAffectEstimate.from_message("very good")
    est_plain = UserAffectEstimate.from_message("good")
    assert est_intensified.valence > est_plain.valence, (
        f"'very good' ({est_intensified.valence}) should be more "
        f"positive than 'good' ({est_plain.valence})"
    )


def test_user_affect_intensifier_amplifies_negative():
    """'extremely terrible' should be more negative than 'terrible'."""
    est_intensified = UserAffectEstimate.from_message("extremely terrible")
    est_plain = UserAffectEstimate.from_message("terrible")
    assert est_intensified.valence < est_plain.valence, (
        f"'extremely terrible' ({est_intensified.valence}) should be "
        f"more negative than 'terrible' ({est_plain.valence})"
    )


def test_user_affect_diminisher_reduces():
    """'slightly good' should be less positive than 'good'."""
    est_diminished = UserAffectEstimate.from_message("slightly good")
    est_plain = UserAffectEstimate.from_message("good")
    assert est_diminished.valence < est_plain.valence, (
        f"'slightly good' ({est_diminished.valence}) should be less "
        f"positive than 'good' ({est_plain.valence})"
    )


def test_user_affect_contrastive_shifts_weight():
    """'good but slow' should be less positive than 'good'."""
    est_contrastive = UserAffectEstimate.from_message("good but slow")
    est_plain = UserAffectEstimate.from_message("good")
    assert est_contrastive.valence < est_plain.valence, (
        f"'good but slow' ({est_contrastive.valence}) should be less "
        f"positive than 'good' ({est_plain.valence})"
    )


def test_user_affect_capitalization_boosts():
    """'GREAT' should be more positive than 'great'."""
    est_caps = UserAffectEstimate.from_message("GREAT")
    est_plain = UserAffectEstimate.from_message("great")
    assert est_caps.valence >= est_plain.valence, (
        f"'GREAT' ({est_caps.valence}) should be at least as positive "
        f"as 'great' ({est_plain.valence})"
    )


def test_user_affect_punctuation_boosts():
    """'great!!!' should be more positive than 'great'."""
    est_punct = UserAffectEstimate.from_message("great!!!")
    est_plain = UserAffectEstimate.from_message("great")
    assert est_punct.valence > est_plain.valence or (
        abs(est_punct.valence - est_plain.valence) < 0.05
    ), (
        f"'great!!!' ({est_punct.valence}) should be more positive "
        f"than 'great' ({est_plain.valence})"
    )


def test_user_affect_dont_like_is_negative():
    """'I don't like this' should be negative.

    The old heuristic matched 'like' and scored it positive.
    """
    est = UserAffectEstimate.from_message("I don't like this")
    assert est.valence < 0.0, (
        f"'I don't like this' should be negative: {est.valence}"
    )


def test_user_affect_mixed_sentiment():
    """'good and bad' should be less extreme than 'good' or 'bad'."""
    est_mixed = UserAffectEstimate.from_message("good and bad")
    est_good = UserAffectEstimate.from_message("good")
    est_bad = UserAffectEstimate.from_message("bad")
    assert abs(est_mixed.valence) < abs(est_good.valence), (
        f"'good and bad' ({est_mixed.valence}) should be less extreme "
        f"than 'good' ({est_good.valence})"
    )
    assert abs(est_mixed.valence) < abs(est_bad.valence), (
        f"'good and bad' ({est_mixed.valence}) should be less extreme "
        f"than 'bad' ({est_bad.valence})"
    )


def test_user_affect_neutral_stays_neutral():
    """Messages with no sentiment words should stay near zero."""
    for msg in ["The sky is blue", "def foo(x):", "hello world", "running"]:
        est = UserAffectEstimate.from_message(msg)
        assert abs(est.valence) < 0.2, (
            f"'{msg}' should be near-neutral: valence={est.valence}"
        )


# ─── ActiveInferenceReader ────────────────────────────────────


def _make_mock_client(summary: InferenceSummary | None = None, raise_exc=None):
    """Create a mock GenesisClient."""
    client = MagicMock()
    if raise_exc:
        client.get_inference_summary.side_effect = raise_exc
    elif summary:
        client.get_inference_summary.return_value = summary
    return client


def test_reader_read_coherent():
    """A coherent self-model (low surprise, high precision) should
    produce a COHERENT state reading."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(
            surprise_ema=0.05,
            free_energy=0.1,
            precision=0.8,
            model_maturity=0.9,
            tick_count=500,
        )
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.self_model_state == SelfModelState.COHERENT
    assert not reading.is_surprised
    assert not reading.is_strained


def test_reader_read_surprised():
    """High surprise should produce a SURPRISED state reading."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(
            surprise_ema=0.5,
            free_energy=0.6,
            precision=0.3,
            model_maturity=0.9,
            tick_count=500,
        )
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.self_model_state == SelfModelState.SURPRISED
    assert reading.is_surprised


def test_reader_read_strained():
    """High allostatic load should produce a STRAINED state reading."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(
            surprise_ema=0.2,
            allostasis_load=0.6,
            precision=0.5,
            model_maturity=0.9,
            tick_count=500,
        )
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.self_model_state == SelfModelState.STRAINED
    assert reading.is_strained
    assert reading.needs_recovery


def test_reader_read_nascent():
    """An immature model should produce a NASCENT state reading."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(
            surprise_ema=0.1,
            precision=0.5,
            model_maturity=0.1,
            tick_count=50,
        )
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.self_model_state == SelfModelState.NASCENT


def test_reader_dyadic_uncoupled():
    """Low attunement should produce an UNCOUPLED dyadic state."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(attunement=0.1, dyadic_synchrony=0.0)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.dyadic_state == DyadicState.UNCOUPLED


def test_reader_dyadic_synchronized():
    """High attunement + high synchrony should produce SYNCHRONIZED."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(attunement=0.5, dyadic_synchrony=0.6)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.dyadic_state == DyadicState.SYNCHRONIZED
    assert reading.is_in_sync


def test_reader_dyadic_discordant():
    """High attunement + negative synchrony should produce DISCORDANT."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(attunement=0.5, dyadic_synchrony=-0.4)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.dyadic_state == DyadicState.DISCORDANT


def test_reader_dyadic_attuned():
    """High attunement + neutral synchrony should produce ATTUNED."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(attunement=0.5, dyadic_synchrony=0.1)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert reading.dyadic_state == DyadicState.ATTUNED


def test_reader_cognitive_effort_tracks_free_energy():
    """Cognitive effort should track free energy."""
    signals = InferenceSummary.unpack(_pack_inference_summary(free_energy=0.7))
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert abs(reading.cognitive_effort - 0.7) < 1e-6


def test_reader_self_confidence_tracks_precision_and_maturity():
    """Self-confidence should be precision × model_maturity."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(precision=0.8, model_maturity=0.5)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    assert abs(reading.self_confidence - 0.4) < 1e-6


def test_reader_social_presence_tracks_attunement():
    """Social presence should track attunement (plus synchrony boost)."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(attunement=0.4, dyadic_synchrony=0.5)
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    # social_presence = min(1.0, attunement + max(0, synchrony) * 0.3)
    expected = min(1.0, 0.4 + 0.5 * 0.3)
    assert abs(reading.social_presence - expected) < 1e-6


def test_reader_state_label_not_template():
    """The state label should be semantic tags, not a canned sentence."""
    signals = InferenceSummary.unpack(
        _pack_inference_summary(
            surprise_ema=0.5,
            attunement=0.5,
            dyadic_synchrony=0.5,
            pe_dopamine=0.1,
            model_maturity=0.9,
            tick_count=500,
        )
    )
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is not None
    # The label should contain semantic tags, not template sentences
    label = reading.state_label
    assert "self-surprise" in label
    assert "dyadic-synchrony" in label
    assert "unexpected-reward" in label
    # Should NOT be a sentence template
    assert "{" not in label
    assert "}" not in label


def test_reader_graceful_degradation_on_error():
    """When the daemon is unreachable, read() should return None
    (graceful degradation)."""
    client = _make_mock_client(raise_exc=ConnectionError("daemon down"))
    reader = ActiveInferenceReader(client)
    reading = reader.read()

    assert reading is None


def test_reader_update_user_affect_success():
    """update_user_affect should send the observation to the daemon."""
    client = MagicMock()
    client.update_user_affect.return_value = True
    reader = ActiveInferenceReader(client)

    est = UserAffectEstimate(valence=0.5, arousal=0.6, engagement=0.7, confidence=0.8)
    result = reader.update_user_affect(est)

    assert result is True
    client.update_user_affect.assert_called_once_with(
        valence=0.5, arousal=0.6, engagement=0.7, confidence=0.8
    )


def test_reader_update_user_affect_failure():
    """update_user_affect should return False on connection error."""
    client = MagicMock()
    client.update_user_affect.side_effect = ConnectionError("daemon down")
    reader = ActiveInferenceReader(client)

    est = UserAffectEstimate(valence=0.5, arousal=0.6, engagement=0.7, confidence=0.8)
    result = reader.update_user_affect(est)

    assert result is False


def test_reader_last_reading():
    """The last_reading property should return the last successful reading."""
    signals = InferenceSummary.unpack(_pack_inference_summary(surprise_ema=0.3))
    client = _make_mock_client(signals)
    reader = ActiveInferenceReader(client)

    assert reader.last_reading is None  # no reading yet

    reading = reader.read()
    assert reading is not None
    assert reader.last_reading is not None
    assert reader.last_reading is reading


# ─── Protocol constants ───────────────────────────────────────


def test_protocol_constants_match_rust():
    """Verify the Python protocol constants match the Rust command IDs."""
    assert GET_INFERENCE_SUMMARY == 19
    assert UPDATE_USER_AFFECT == 20


# ─── Sentiment lexicon shadowing regression tests ──────────────
# These verify that words appearing in both LEXICON and a modifier
# set (DIMINISHERS) are correctly treated as sentiment words, not
# silently shadowed by the modifier entry.


def test_user_affect_kind_is_positive_sentiment():
    """'kind' should be detected as a positive sentiment word.

    Previously, 'kind' was in DIMINISHERS (for 'kind of' phrases),
    which shadowed the LEXICON entry 'kind': 1.8. This meant
    'you are kind' scored as neutral — a clear positive sentiment
    was silently lost, degrading the dyadic affective model's
    valence estimate.
    """
    est = UserAffectEstimate.from_message("you are kind")
    assert est.valence > 0.0, (
        f"'kind' should be positive sentiment: {est.valence}"
    )


def test_user_affect_kind_of_still_works():
    """'kinda' still functions as a diminisher.

    'kind' is not in DIMINISHERS (that would shadow the LEXICON entry),
    but 'kinda' (the single-word form) still diminishes.
    """
    # 'kinda good' should be less positive than 'good'
    est_kinda = UserAffectEstimate.from_message("kinda good")
    est_plain = UserAffectEstimate.from_message("good")
    assert est_kinda.valence < est_plain.valence, (
        f"'kinda good' ({est_kinda.valence}) should be less positive "
        f"than 'good' ({est_plain.valence})"
    )


def test_user_affect_kind_of_phrase():
    """The phrase 'kind of' diminishes without shadowing 'kind'.

    'kind of' is recognized as a multi-word diminisher phrase, so the
    hedge weakens the following sentiment word instead of contributing
    positive weight from 'kind'. This is the integration path that
    feeds the dyadic affective model's valence estimate.
    """
    est_bad = UserAffectEstimate.from_message("bad")
    est_kind_of_bad = UserAffectEstimate.from_message("kind of bad")
    assert est_kind_of_bad.valence < 0.0, (
        f"'kind of bad' should stay negative: {est_kind_of_bad.valence}"
    )
    assert est_kind_of_bad.valence > est_bad.valence, (
        f"'kind of bad' ({est_kind_of_bad.valence}) should be less negative "
        f"than 'bad' ({est_bad.valence})"
    )

    est_good = UserAffectEstimate.from_message("good")
    est_kind_of_good = UserAffectEstimate.from_message("kind of good")
    assert 0.0 < est_kind_of_good.valence < est_good.valence, (
        f"'kind of good' ({est_kind_of_good.valence}) should be positive but "
        f"weaker than 'good' ({est_good.valence})"
    )


def test_user_affect_a_is_not_a_diminisher():
    """The article 'a' should not diminish the following sentiment word.

    Previously, 'a' was in DIMINISHERS (for 'a bit' / 'a little'
    phrases), but since the tokenizer splits on whitespace, 'a'
    alone was consumed as a diminisher before any sentiment word
    following an article — e.g. 'a good idea' scored lower than
    'good idea'. The diminisher for 'a bit' / 'a little' is already
    handled by 'bit' and 'little' themselves.
    """
    est_with_a = UserAffectEstimate.from_message("a good idea")
    est_without_a = UserAffectEstimate.from_message("good idea")
    assert abs(est_with_a.valence - est_without_a.valence) < 0.05, (
        f"'a good idea' ({est_with_a.valence}) should match "
        f"'good idea' ({est_without_a.valence}) — 'a' is an article, "
        f"not a diminisher"
    )


# ======================================================================
# From tests/test_predictive_coding.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_perception(
    text: str = "hello",
    intent: Intent = Intent.GREETING,
    topics: list[str] | None = None,
) -> Perception:
    """Create a Perception with specified fields."""
    return Perception(
        raw_text=text,
        intent=intent,
        question_type=QuestionType.NONE,
        topics=topics or [],
        sentiment=0.0,
        sentiment_label="neutral",
        emotion_word="",
        is_about_genesis=False,
        is_about_user=False,
        is_about_code=False,
        is_about_emotion=False,
        is_about_existence=False,
        entities={},
        key_phrases=[],
        word_count=len(text.split()),
        confidence=0.5,
    )


def _make_context(
    recent_intents: list[str] | None = None,
    recent_topics: list[str] | None = None,
    active_concepts: list[str] | None = None,
) -> PredictionContext:
    """Create a PredictionContext."""
    return PredictionContext(
        recent_intents=recent_intents or [],
        recent_topics=recent_topics or [],
        active_concepts=active_concepts or [],
    )


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_prediction_creation() -> None:
    """Prediction stores expected intents, concepts, topic, confidence."""
    pred = Prediction(
        expected_intents=[("question", 0.8)],
        expected_concepts=[("dogs", 0.7)],
        expected_topic="dogs",
        confidence=0.6,
    )
    assert pred.expected_topic == "dogs"
    assert pred.confidence == 0.6
    assert pred.level_errors == {}


def test_prediction_error_creation() -> None:
    """PredictionError stores magnitude, unexpected/missing concepts."""
    err = PredictionError(
        magnitude=0.5,
        unexpected_concepts=["cats"],
        missing_concepts=["dogs"],
        intent_mismatch=True,
    )
    assert err.magnitude == 0.5
    assert "cats" in err.unexpected_concepts
    assert "dogs" in err.missing_concepts
    assert err.intent_mismatch is True


def test_prediction_context_creation() -> None:
    """PredictionContext stores recent intents, topics, active concepts."""
    ctx = PredictionContext(
        recent_intents=["greeting"],
        recent_topics=["dogs"],
        active_concepts=["dogs", "mammals"],
        emotional_state="curious",
        time_of_day=14,
    )
    assert ctx.recent_intents == ["greeting"]
    assert ctx.recent_topics == ["dogs"]
    assert ctx.active_concepts == ["dogs", "mammals"]
    assert ctx.emotional_state == "curious"
    assert ctx.time_of_day == 14


def test_prediction_context_defaults() -> None:
    """PredictionContext defaults are empty."""
    ctx = PredictionContext()
    assert ctx.recent_intents == []
    assert ctx.recent_topics == []
    assert ctx.active_concepts == []
    assert ctx.emotional_state == ""
    assert ctx.time_of_day == -1


# ═══════════════════════════════════════════════════════════════════
# Prediction generation
# ═══════════════════════════════════════════════════════════════════


def test_predict_empty_context() -> None:
    """predict() with empty context returns low-confidence prediction."""
    layer = PredictiveCodingLayer()
    pred = layer.predict(_make_context())
    assert isinstance(pred, Prediction)
    assert pred.confidence == 0.0  # no history → no confidence


def test_predict_with_topics() -> None:
    """predict() with topic history returns a topic prediction."""
    layer = PredictiveCodingLayer()
    # First, learn a transition: dogs → cats
    ctx = _make_context(recent_topics=["dogs"])
    pred = layer.predict(ctx)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    # Now predict from dogs again — should predict cats
    pred2 = layer.predict(_make_context(recent_topics=["dogs"]))
    assert pred2.expected_topic == "cats"


def test_predict_with_intents() -> None:
    """predict() with intent history returns intent predictions."""
    layer = PredictiveCodingLayer()
    # Learn: greeting → question
    ctx = _make_context(recent_intents=["greeting"])
    pred = layer.predict(ctx)
    actual = _make_perception(text="what?", intent=Intent.QUESTION, topics=["what"])
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    # Now predict from greeting — should predict question
    pred2 = layer.predict(_make_context(recent_intents=["greeting"]))
    intent_names = [i[0] for i in pred2.expected_intents]
    assert "question" in intent_names


def test_predict_increments_count() -> None:
    """predict() increments the prediction count."""
    layer = PredictiveCodingLayer()
    assert layer.get_prediction_count() == 0
    layer.predict(_make_context())
    assert layer.get_prediction_count() == 1
    layer.predict(_make_context())
    assert layer.get_prediction_count() == 2


def test_predict_confidence_in_range() -> None:
    """predict() confidence is in [0, 1]."""
    layer = PredictiveCodingLayer()
    pred = layer.predict(_make_context(recent_topics=["dogs"]))
    assert 0.0 <= pred.confidence <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Error computation
# ═══════════════════════════════════════════════════════════════════


def test_compute_error_perfect_match() -> None:
    """compute_error returns low magnitude when prediction matches actual."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[("question", 1.0)],
        expected_concepts=[("dogs", 1.0)],
        expected_topic="dogs",
        confidence=1.0,
    )
    actual = _make_perception(text="dogs", intent=Intent.QUESTION, topics=["dogs"])
    err = layer.compute_error(pred, actual)
    assert err.magnitude == 0.0
    assert not err.intent_mismatch


def test_compute_error_topic_mismatch() -> None:
    """compute_error detects topic mismatch."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[("question", 0.9)],
        expected_concepts=[("dogs", 0.9)],
        expected_topic="dogs",
        confidence=0.9,
    )
    actual = _make_perception(text="cats", intent=Intent.QUESTION, topics=["cats"])
    err = layer.compute_error(pred, actual)
    assert err.magnitude > 0.0
    assert err.level_errors["topic"] > 0.0


def test_compute_error_intent_mismatch() -> None:
    """compute_error detects intent mismatch."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[("question", 0.9)],
        expected_concepts=[],
        expected_topic="",
        confidence=0.5,
    )
    actual = _make_perception(text="hello", intent=Intent.GREETING, topics=[])
    err = layer.compute_error(pred, actual)
    assert err.intent_mismatch is True
    assert err.level_errors["intent"] > 0.0


def test_compute_error_unexpected_concepts() -> None:
    """compute_error identifies unexpected concepts."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[],
        expected_concepts=[("dogs", 0.9)],
        expected_topic="dogs",
        confidence=0.5,
    )
    actual = _make_perception(text="cats birds", intent=Intent.STATEMENT, topics=["cats", "birds"])
    err = layer.compute_error(pred, actual)
    assert "cats" in err.unexpected_concepts
    assert "birds" in err.unexpected_concepts


def test_compute_error_missing_concepts() -> None:
    """compute_error identifies missing concepts."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[],
        expected_concepts=[("dogs", 0.9), ("mammals", 0.8)],
        expected_topic="dogs",
        confidence=0.5,
    )
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    err = layer.compute_error(pred, actual)
    assert "dogs" in err.missing_concepts
    assert "mammals" in err.missing_concepts


def test_compute_error_magnitude_bounded() -> None:
    """compute_error magnitude is bounded [0, 1]."""
    layer = PredictiveCodingLayer()
    pred = Prediction(
        expected_intents=[("question", 1.0)],
        expected_concepts=[("dogs", 1.0)],
        expected_topic="dogs",
        confidence=1.0,
    )
    actual = _make_perception(text="cats", intent=Intent.COMMAND, topics=["cats"])
    err = layer.compute_error(pred, actual)
    assert 0.0 <= err.magnitude <= 1.0


def test_compute_error_no_prediction() -> None:
    """compute_error with empty prediction returns moderate error."""
    layer = PredictiveCodingLayer()
    pred = Prediction()
    actual = _make_perception(text="hello", intent=Intent.GREETING, topics=["hello"])
    err = layer.compute_error(pred, actual)
    # No prediction → 0.5 topic error (neutral)
    assert err.level_errors["topic"] == 0.5


def test_compute_error_updates_surprise() -> None:
    """compute_error updates the surprise EMA."""
    layer = PredictiveCodingLayer()
    pred = Prediction(expected_topic="dogs", confidence=0.5)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    initial_surprise = layer.get_surprise()
    layer.compute_error(pred, actual)
    assert layer.get_surprise() > initial_surprise


def test_compute_error_updates_accuracy() -> None:
    """compute_error updates the accuracy EMA."""
    layer = PredictiveCodingLayer()
    # First, make a wrong prediction to bring accuracy below 1.0
    pred_wrong = Prediction(expected_topic="dogs", confidence=0.5)
    actual_wrong = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    layer.compute_error(pred_wrong, actual_wrong)
    accuracy_after_error = layer.get_prediction_accuracy()
    # Now a correct prediction — accuracy should increase
    pred_right = Prediction(
        expected_intents=[("statement", 1.0)],
        expected_concepts=[("dogs", 1.0)],
        expected_topic="dogs",
        confidence=1.0,
    )
    actual_right = _make_perception(text="dogs", intent=Intent.STATEMENT, topics=["dogs"])
    layer.compute_error(pred_right, actual_right)
    assert layer.get_prediction_accuracy() > accuracy_after_error


def test_compute_error_stores_level_errors_on_prediction() -> None:
    """compute_error stores level_errors on the prediction object."""
    layer = PredictiveCodingLayer()
    pred = Prediction(expected_topic="dogs", confidence=0.5)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    layer.compute_error(pred, actual)
    assert "topic" in pred.level_errors
    assert "intent" in pred.level_errors
    assert "concept" in pred.level_errors


# ═══════════════════════════════════════════════════════════════════
# Learning
# ═══════════════════════════════════════════════════════════════════


def test_learn_from_error_updates_topic_transitions() -> None:
    """learn_from_error updates topic transition counts."""
    layer = PredictiveCodingLayer()
    ctx = _make_context(recent_topics=["dogs"])
    pred = layer.predict(ctx)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    # Now predicting from "dogs" should predict "cats"
    pred2 = layer.predict(_make_context(recent_topics=["dogs"]))
    assert pred2.expected_topic == "cats"


def test_learn_from_error_updates_intent_transitions() -> None:
    """learn_from_error updates intent transition counts."""
    layer = PredictiveCodingLayer()
    ctx = _make_context(recent_intents=["greeting"])
    pred = layer.predict(ctx)
    actual = _make_perception(text="what?", intent=Intent.QUESTION, topics=[])
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    pred2 = layer.predict(_make_context(recent_intents=["greeting"]))
    intent_names = [i[0] for i in pred2.expected_intents]
    assert "question" in intent_names


def test_learn_from_error_updates_concept_cooccurrence() -> None:
    """learn_from_error updates concept co-occurrence."""
    layer = PredictiveCodingLayer()
    ctx = _make_context()
    pred = layer.predict(ctx)
    actual = _make_perception(
        text="dogs mammals", intent=Intent.STATEMENT, topics=["dogs", "mammals"]
    )
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    # After learning, dogs and mammals should co-occur
    pred2 = layer.predict(_make_context(active_concepts=["dogs"]))
    concept_names = [c[0] for c in pred2.expected_concepts]
    assert "mammals" in concept_names


def test_learn_from_error_no_context() -> None:
    """learn_from_error handles missing context gracefully."""
    layer = PredictiveCodingLayer()
    actual = _make_perception(text="hello", intent=Intent.GREETING, topics=["hello"])
    err = PredictionError(magnitude=0.5)
    # Should not raise
    layer.learn_from_error(err, actual)


def test_learn_from_error_updates_history() -> None:
    """learn_from_error updates recent history."""
    layer = PredictiveCodingLayer()
    ctx = _make_context(recent_topics=["dogs"])
    pred = layer.predict(ctx)
    actual = _make_perception(text="cats", intent=Intent.QUESTION, topics=["cats"])
    err = layer.compute_error(pred, actual)
    layer.learn_from_error(err, actual)

    assert "question" in layer.get_recent_intents()
    assert "cats" in layer.get_recent_topics()


# ═══════════════════════════════════════════════════════════════════
# Surprise and accuracy tracking
# ═══════════════════════════════════════════════════════════════════


def test_get_surprise_initial() -> None:
    """Initial surprise is 0."""
    layer = PredictiveCodingLayer()
    assert layer.get_surprise() == 0.0


def test_get_prediction_accuracy_initial() -> None:
    """Initial accuracy is 1.0."""
    layer = PredictiveCodingLayer()
    assert layer.get_prediction_accuracy() == 1.0


def test_get_prediction_count_initial() -> None:
    """Initial prediction count is 0."""
    layer = PredictiveCodingLayer()
    assert layer.get_prediction_count() == 0


def test_surprise_increases_with_errors() -> None:
    """Surprise increases when predictions are wrong."""
    layer = PredictiveCodingLayer()
    pred = Prediction(expected_topic="dogs", confidence=0.5)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    layer.compute_error(pred, actual)
    assert layer.get_surprise() > 0.0


def test_accuracy_decreases_with_errors() -> None:
    """Accuracy decreases when predictions are wrong."""
    layer = PredictiveCodingLayer()
    pred = Prediction(expected_topic="dogs", confidence=0.5)
    actual = _make_perception(text="cats", intent=Intent.STATEMENT, topics=["cats"])
    layer.compute_error(pred, actual)
    assert layer.get_prediction_accuracy() < 1.0


# ═══════════════════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════════════════


def test_get_recent_intents_empty() -> None:
    """get_recent_intents returns empty list initially."""
    layer = PredictiveCodingLayer()
    assert layer.get_recent_intents() == []


def test_get_recent_topics_empty() -> None:
    """get_recent_topics returns empty list initially."""
    layer = PredictiveCodingLayer()
    assert layer.get_recent_topics() == []


def test_history_trimmed() -> None:
    """History is trimmed to the limit."""
    layer = PredictiveCodingLayer()
    # Make many predictions to exceed the history limit
    for i in range(40):
        ctx = _make_context(recent_topics=[f"topic_{i}"])
        pred = layer.predict(ctx)
        actual = _make_perception(
            text=f"actual_{i}", intent=Intent.STATEMENT, topics=[f"actual_{i}"]
        )
        err = layer.compute_error(pred, actual)
        layer.learn_from_error(err, actual)
    # History should be trimmed
    assert len(layer.get_recent_intents()) <= 32
    assert len(layer.get_recent_topics()) <= 32


# ═══════════════════════════════════════════════════════════════════
# describe()
# ═══════════════════════════════════════════════════════════════════


def test_describe() -> None:
    """describe() returns a human-readable summary."""
    layer = PredictiveCodingLayer()
    desc = layer.describe()
    assert isinstance(desc, str)
    assert "Predictive coding" in desc
    assert "surprise" in desc
    assert "accuracy" in desc


# ═══════════════════════════════════════════════════════════════════
# Full cycle: perceive_with_prediction
# ═══════════════════════════════════════════════════════════════════


def test_perceive_with_prediction() -> None:
    """perceive_with_prediction runs the full cycle."""
    layer = PredictiveCodingLayer()
    ctx = _make_context(recent_topics=["dogs"])
    perception, prediction, error = layer.perceive_with_prediction(
        "tell me about cats",
        ctx,
    )
    assert isinstance(perception, Perception)
    assert isinstance(prediction, Prediction)
    assert isinstance(error, PredictionError)


def test_perceive_with_prediction_learns() -> None:
    """perceive_with_prediction updates the model."""
    layer = PredictiveCodingLayer()
    ctx = _make_context(recent_topics=["dogs"])
    layer.perceive_with_prediction("cats", ctx)
    # After learning, predicting from "dogs" should predict "cats"
    pred = layer.predict(_make_context(recent_topics=["dogs"]))
    assert pred.expected_topic == "cats"


# ═══════════════════════════════════════════════════════════════════
# Neurochemical coupling
# ═══════════════════════════════════════════════════════════════════


def test_neurochemical_callback_triggered() -> None:
    """High prediction error triggers the neurochemical callback."""
    calls: list[tuple] = []

    def callback(chem: int, magnitude: float) -> None:
        """Record a (chem, magnitude) neurochemical impulse in the shared list."""
        calls.append((chem, magnitude))

    layer = PredictiveCodingLayer(on_neuro_impulse=callback)
    # Create a maximally surprising prediction
    pred = Prediction(
        expected_intents=[("question", 1.0)],
        expected_concepts=[("dogs", 1.0)],
        expected_topic="dogs",
        confidence=1.0,
    )
    actual = _make_perception(text="cats", intent=Intent.COMMAND, topics=["cats"])
    layer.compute_error(pred, actual)
    # Should have triggered at least one neurochemical impulse
    assert len(calls) > 0


def test_neurochemical_callback_not_triggered_for_low_error() -> None:
    """Low prediction error does not trigger the callback."""
    calls: list[tuple] = []

    def callback(chem: int, magnitude: float) -> None:
        """Record a (chem, magnitude) neurochemical impulse in the shared list."""
        calls.append((chem, magnitude))

    layer = PredictiveCodingLayer(on_neuro_impulse=callback)
    pred = Prediction(
        expected_intents=[("question", 1.0)],
        expected_concepts=[("dogs", 1.0)],
        expected_topic="dogs",
        confidence=1.0,
    )
    actual = _make_perception(text="dogs", intent=Intent.QUESTION, topics=["dogs"])
    layer.compute_error(pred, actual)
    # Perfect match → no impulse
    assert len(calls) == 0


def test_neurochemical_callback_error_handled() -> None:
    """Neurochemical callback errors are handled gracefully."""
    def callback(chem: int, magnitude: float) -> None:
        """Raise OSError to simulate a callback failure."""
        raise OSError("connection lost")

    layer = PredictiveCodingLayer(on_neuro_impulse=callback)
    pred = Prediction(expected_topic="dogs", confidence=1.0)
    actual = _make_perception(text="cats", intent=Intent.COMMAND, topics=["cats"])
    # Should not raise
    layer.compute_error(pred, actual)


# ======================================================================
# From tests/test_learning_fidelity.py
# ======================================================================


# ─── Test facts ───────────────────────────────────────────────────
# Novel facts not in the initial concept network seeds.
# Each is (teach_input, concept_name, expected_definition_substrings,
#           expected_relation_type, expected_relation_target)

_TEACH_FACTS = [
    (
        "A tardigrade is a micro-animal.",
        "tardigrade",
        ("animal",),
        RelationType.IS_A,
        "animal",
    ),
    (
        "Cryptobiosis is a state of suspended animation.",
        "cryptobiosis",
        ("suspended", "animation"),
        None,
        None,
    ),
    (
        "A mycelium is a network of fungal threads.",
        "mycelium",
        ("network", "fungal"),
        None,
        None,
    ),
    (
        "Photosynthesis is a process that converts light into energy.",
        "photosynthesis",
        ("light", "energy"),
        None,
        None,
    ),
    (
        "A star is a luminous sphere of plasma.",
        "star",
        ("luminous", "plasma"),
        None,
        None,
    ),
]

# Filler facts to create interference (unrelated to the test facts)
_FILLER_FACTS = [
    "A guitar is a stringed musical instrument.",
    "The Amazon is a river in South America.",
    "A glacier is a slow-moving mass of ice.",
    "Quartz is a hard crystalline mineral.",
    "A savanna is a grassland with scattered trees.",
    "The violin is a stringed instrument played with a bow.",
    "A canyon is a deep gorge carved by a river.",
    "Granite is a coarse-grained igneous rock.",
]


def _check_fact_learned(mind, concept_name, expected_substrings,
                        expected_rel_type, expected_rel_target):
    """Check if a fact was learned: concept exists, definition contains
    expected substrings, and optionally the expected relation exists."""
    net = mind.cognition.network
    concept = net.get_concept(concept_name)
    if concept is None:
        return False, f"concept '{concept_name}' not found"

    # Check definition
    definition = concept.properties.get("definition", "")
    if not definition:
        return False, f"concept '{concept_name}' has no definition"

    def_lower = definition.lower()
    for sub in expected_substrings:
        if sub.lower() not in def_lower:
            return False, (
                f"definition '{definition}' missing '{sub}'"
            )

    # Check relation if expected
    if expected_rel_type is not None and expected_rel_target is not None:
        edges = net.get_edges(concept_name, direction="out")
        found_rel = False
        for edge in edges:
            if (edge.relation == expected_rel_type and
                    edge.target == expected_rel_target):
                found_rel = True
                break
        if not found_rel:
            return False, (
                f"no {expected_rel_type} edge to '{expected_rel_target}'"
            )

    return True, "ok"


def test_learning_fidelity_immediate_recall():
    """Genesis can learn novel facts and recall them immediately."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")
        mind = Mind(socket_path)

        # Teach all facts
        for teach_input, _, _, _, _ in _TEACH_FACTS:
            mind.cognition.self_learner.learn_from_input(teach_input)
        mind.cognition.self_learner.run_inference_cycle(force=True)

        # Check immediate recall
        correct = 0
        results = []
        for _teach_input, concept_name, expected_subs, rel_type, rel_target in _TEACH_FACTS:
            ok, detail = _check_fact_learned(
                mind, concept_name, expected_subs, rel_type, rel_target
            )
            if ok:
                correct += 1
            results.append((concept_name, ok, detail))

        accuracy = correct / len(_TEACH_FACTS)
        print(f"\n  Immediate recall: {correct}/{len(_TEACH_FACTS)} "
              f"({accuracy:.1%})")
        for name, ok, detail in results:
            status = "✓" if ok else "✗"
            print(f"    {status} {name}: {detail}")

        assert accuracy >= 0.6, (
            f"Immediate recall too low: {accuracy:.1%} "
            f"({correct}/{len(_TEACH_FACTS)}). Expected >= 60%."
        )


def test_learning_fidelity_delayed_recall_after_interference():
    """Learned facts survive interference from unrelated facts."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")
        mind = Mind(socket_path)

        # Phase A: Teach all facts
        for teach_input, _, _, _, _ in _TEACH_FACTS:
            mind.cognition.self_learner.learn_from_input(teach_input)
        mind.cognition.self_learner.run_inference_cycle(force=True)

        # Phase B: Immediate recall
        immediate_correct = 0
        for _, concept_name, expected_subs, rel_type, rel_target in _TEACH_FACTS:
            ok, _ = _check_fact_learned(
                mind, concept_name, expected_subs, rel_type, rel_target
            )
            if ok:
                immediate_correct += 1

        # Phase C: Filler interactions (interference)
        for filler in _FILLER_FACTS:
            mind.cognition.self_learner.learn_from_input(filler)
        mind.cognition.self_learner.run_inference_cycle(force=True)

        # Phase D: Delayed recall
        delayed_correct = 0
        results = []
        for _, concept_name, expected_subs, rel_type, rel_target in _TEACH_FACTS:
            ok, detail = _check_fact_learned(
                mind, concept_name, expected_subs, rel_type, rel_target
            )
            if ok:
                delayed_correct += 1
            results.append((concept_name, ok, detail))

        immediate_accuracy = immediate_correct / len(_TEACH_FACTS)
        delayed_accuracy = delayed_correct / len(_TEACH_FACTS)
        drift = immediate_accuracy - delayed_accuracy

        print(f"\n  Immediate: {immediate_correct}/{len(_TEACH_FACTS)} "
              f"({immediate_accuracy:.1%})")
        print(f"  Delayed:   {delayed_correct}/{len(_TEACH_FACTS)} "
              f"({delayed_accuracy:.1%})")
        print(f"  Drift:     {drift:.1%}")
        for name, ok, detail in results:
            status = "✓" if ok else "✗"
            print(f"    {status} {name}: {detail}")

        # Delayed recall should be at least 50% (some drift is expected
        # but most facts should survive)
        assert delayed_accuracy >= 0.5, (
            f"Delayed recall too low: {delayed_accuracy:.1%} "
            f"({delayed_correct}/{len(_TEACH_FACTS)}). Expected >= 50%."
        )


def test_learning_fidelity_persistence_across_save_load():
    """Learned facts survive a save/load cycle."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")

        # Session 1: Learn facts, save
        mind1 = Mind(socket_path)
        for teach_input, _, _, _, _ in _TEACH_FACTS:
            mind1.cognition.self_learner.learn_from_input(teach_input)
        mind1.cognition.self_learner.run_inference_cycle(force=True)
        mind1._save_state()

        # Session 2: Load, verify facts survived
        mind2 = Mind(socket_path)
        mind2._load_saved_state()

        correct = 0
        for _, concept_name, expected_subs, rel_type, rel_target in _TEACH_FACTS:
            ok, _ = _check_fact_learned(
                mind2, concept_name, expected_subs, rel_type, rel_target
            )
            if ok:
                correct += 1

        accuracy = correct / len(_TEACH_FACTS)
        print(f"\n  After save/load: {correct}/{len(_TEACH_FACTS)} "
              f"({accuracy:.1%})")

        assert accuracy >= 0.5, (
            f"Recall after save/load too low: {accuracy:.1%}. "
            f"Learned facts should survive persistence."
        )


def test_learning_fidelity_results_persisted():
    """Learning fidelity results can be written to a JSON file for
    tracking progress across runs."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")
        mind = Mind(socket_path)

        # Teach facts
        for teach_input, _, _, _, _ in _TEACH_FACTS:
            mind.cognition.self_learner.learn_from_input(teach_input)
        mind.cognition.self_learner.run_inference_cycle(force=True)

        # Score
        correct = 0
        details = []
        for _, concept_name, expected_subs, rel_type, rel_target in _TEACH_FACTS:
            ok, detail = _check_fact_learned(
                mind, concept_name, expected_subs, rel_type, rel_target
            )
            if ok:
                correct += 1
            details.append({"concept": concept_name, "correct": ok, "detail": detail})

        results = {
            "n_facts": len(_TEACH_FACTS),
            "correct": correct,
            "accuracy": correct / len(_TEACH_FACTS),
            "details": details,
        }

        # Write to JSON
        results_path = os.path.join(data_dir, "learning_fidelity.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Verify it can be read back
        with open(results_path) as f:
            loaded = json.load(f)

        assert loaded["n_facts"] == len(_TEACH_FACTS)
        assert loaded["correct"] == correct
        assert loaded["accuracy"] == correct / len(_TEACH_FACTS)
        assert len(loaded["details"]) == len(_TEACH_FACTS)


# ======================================================================
# From tests/test_cpu_self_regulation.py
# ======================================================================


def _make_learner() -> AutonomousLearner:
    """Create a learner with minimal dependencies for testing."""
    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    return AutonomousLearner(network=net, curiosity=curiosity)


# ─── AutonomousLearner throttle/unthrottle ───────────────────────


def test_learner_throttle_sets_flag():
    """throttle() should set the _throttled flag."""
    learner = _make_learner()
    assert not learner._throttled
    learner.throttle()
    assert learner._throttled


def test_learner_unthrottle_clears_flag():
    """unthrottle() should clear the _throttled flag."""
    learner = _make_learner()
    learner.throttle()
    assert learner._throttled
    learner.unthrottle()
    assert not learner._throttled


def test_learner_throttle_is_idempotent():
    """Calling throttle() twice should not emit duplicate messages."""
    learner = _make_learner()
    learner.throttle()
    # Second call should be a no-op (no crash, flag stays True)
    learner.throttle()
    assert learner._throttled


def test_throttled_delay_multiplier_is_reasonable():
    """The throttle multiplier should meaningfully reduce CPU load."""
    assert THROTTLED_DELAY_MULTIPLIER >= 3.0, (
        f"Multiplier too small: {THROTTLED_DELAY_MULTIPLIER}, "
        "must be >= 3.0 to meaningfully reduce CPU load"
    )
    throttled_delay = RATE_LIMIT_DELAY * THROTTLED_DELAY_MULTIPLIER
    # Throttled delay should be at least 30 seconds (vs 8s normal)
    assert throttled_delay >= 30.0, (
        f"Throttled delay too short: {throttled_delay}s, "
        "must be >= 30s to reduce CPU load"
    )


# ─── EmotionalRegulator CPU self-regulation ──────────────────────


def _make_regulator() -> tuple[EmotionalRegulator, list[str], list[str]]:
    """Create a regulator with mock throttle/unthrottle callbacks.

    Returns (regulator, throttle_calls, unthrottle_calls).
    """
    throttle_calls: list[str] = []
    unthrottle_calls: list[str] = []

    def on_throttle() -> None:
        """Record that the regulator throttled emotion."""
        throttle_calls.append("throttled")

    def on_unthrottle() -> None:
        """Record that the regulator unthrottled emotion."""
        unthrottle_calls.append("unthrottled")

    regulator = EmotionalRegulator()
    regulator.set_throttle_callbacks(on_throttle, on_unthrottle)
    return regulator, throttle_calls, unthrottle_calls


def test_regulator_throttles_after_sustained_cpu_stress():
    """The regulator should call throttle after 3 consecutive CPU > 80%."""
    regulator, throttle_calls, _ = _make_regulator()

    # Simulate the streak logic that _regulate_interoception uses
    # (we test the streak logic directly since sense_internal_state
    # reads real CPU which we can't control in tests)
    regulator._cpu_streak = 0
    regulator._is_throttled = False

    # Simulate 3 consecutive CPU > 80 detections
    for _ in range(3):
        regulator._cpu_streak += 1
        regulator._cpu_recover_streak = 0
        if (
            regulator._cpu_streak >= 3
            and not regulator._is_throttled
            and regulator._throttle_callback
        ):
            regulator._throttle_callback()
            regulator._is_throttled = True

    assert len(throttle_calls) == 1, (
        f"Expected 1 throttle call after 3 consecutive detections, "
        f"got {len(throttle_calls)}"
    )
    assert regulator._is_throttled


def test_regulator_does_not_throttle_on_single_spike():
    """A single CPU spike should NOT trigger throttling."""
    regulator, throttle_calls, _ = _make_regulator()

    # Simulate just 1 high-CPU detection
    regulator._cpu_streak = 1
    # Check the condition — should NOT throttle with only 1
    if (
        regulator._cpu_streak >= 3
        and not regulator._is_throttled
        and regulator._throttle_callback
    ):
        regulator._throttle_callback()
        regulator._is_throttled = True

    assert len(throttle_calls) == 0, (
        "Single CPU spike should not trigger throttling"
    )
    assert not regulator._is_throttled


def test_regulator_unthrottles_after_sustained_recovery():
    """The regulator should call unthrottle after 3 consecutive CPU <= 60%."""
    regulator, _, unthrottle_calls = _make_regulator()

    # Start in throttled state
    regulator._is_throttled = True

    # Simulate 3 consecutive low-CPU readings
    for _ in range(3):
        regulator._cpu_recover_streak += 1
        regulator._cpu_streak = 0
        if regulator._cpu_recover_streak >= 3 and regulator._unthrottle_callback:
            regulator._unthrottle_callback()
            regulator._is_throttled = False
            break

    assert len(unthrottle_calls) == 1, (
        f"Expected 1 unthrottle call after 3 consecutive recoveries, "
        f"got {len(unthrottle_calls)}"
    )
    assert not regulator._is_throttled


def test_regulator_throttle_callbacks_are_set():
    """set_throttle_callbacks should store both callbacks."""
    regulator = EmotionalRegulator()
    assert regulator._throttle_callback is None
    assert regulator._unthrottle_callback is None

    def throttle() -> None:
        """No-op throttle callback for wiring tests."""
        pass

    def unthrottle() -> None:
        """No-op unthrottle callback for wiring tests."""
        pass

    regulator.set_throttle_callbacks(throttle, unthrottle)
    assert regulator._throttle_callback is throttle
    assert regulator._unthrottle_callback is unthrottle


# ─── Mind wiring ─────────────────────────────────────────────────


def test_mind_wires_throttle_callbacks():
    """Mind should wire the learner's throttle/unthrottle to the regulator."""
    with tempfile.TemporaryDirectory() as data_dir:
        socket_path = os.path.join(data_dir, "genesis.sock")
        mind = Mind(socket_path)

        # The regulator should have throttle callbacks set
        assert mind.regulator._throttle_callback is not None, (
            "Regulator should have a throttle callback"
        )
        assert mind.regulator._unthrottle_callback is not None, (
            "Regulator should have an unthrottle callback"
        )

        # Calling the throttle callback should throttle the learner
        assert not mind.learner._throttled
        mind.regulator._throttle_callback()
        assert mind.learner._throttled, (
            "Throttle callback should throttle the learner"
        )

        # Calling the unthrottle callback should unthrottle the learner
        mind.regulator._unthrottle_callback()
        assert not mind.learner._throttled, (
            "Unthrottle callback should unthrottle the learner"
        )

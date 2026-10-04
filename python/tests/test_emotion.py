"""Emotion tests — emotional regulator and sentiment analysis."""

import logging
import time
from typing import Any

import pytest

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_ADENOSINE,
    CHEM_BDNF,
    CHEM_CORTISOL,
    CHEM_CRH,
    CHEM_DOPAMINE,
    CHEM_ENDORPHIN,
    CHEM_GABA,
    CHEM_HISTAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_SEROTONIN,
)
from genesis_conscious.language.sentiment import analyze_sentiment
from genesis_conscious.limbic_system.emotion import EmotionalState
from genesis_conscious.limbic_system.emotional_regulator import (
    AllostaticLoadTracker,
    EmotionalRegulator,
    InteroceptionSystem,
)

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_emotional_regulator.py
# ======================================================================

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
    """Create a regulator that records all impulses sent.

    Interoception is mocked to a calm state so tests don't read real
    CPU/memory values (which would make them flaky under load).
    """
    impulses: list[tuple[int, float]] = []

    def mock_impulse(chem: int, amount: float) -> None:
        """Record a (chem, amount) impulse in the shared list."""
        impulses.append((chem, amount))

    regulator = EmotionalRegulator(
        get_emotion=None,
        neuro_impulse=mock_impulse,
    )
    # Mock interoception to a calm state so tests are deterministic.
    from genesis_conscious.limbic_system.emotional_regulator import InternalState

    calm_state = InternalState(
        cpu_usage=0.0,
        memory_usage=0.0,
        daemon_connected=True,
        response_latency=0.0,
        stress_level=0.0,
        arousal_modifier=0.5,
    )
    regulator._interoception._last_state = calm_state

    def mock_sense() -> InternalState:
        return calm_state

    regulator._interoception.sense_internal_state = mock_sense  # type: ignore[method-assign]
    return regulator, impulses


def chem_amounts(impulses: list[tuple[int, float]]) -> dict[int, float]:
    """Convert impulse list to {chem: total_amount} dict."""
    result: dict[int, float] = {}
    for chem, amount in impulses:
        result[chem] = result.get(chem, 0.0) + amount
    return result


# ── Homeostatic maintenance tests ──────────────────────────────


def test_maintenance_low_plasticity_severe() -> None:
    """When plasticity is severely low, it boosts BDNF + serotonin."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral", plasticity=0.15)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_BDNF, 0) >= 0.05, f"Expected BDNF boost, got {amounts}"
    assert amounts.get(CHEM_SEROTONIN, 0) >= 0.03, f"Expected serotonin boost, got {amounts}"


def test_maintenance_low_plasticity_mild() -> None:
    """When plasticity is mildly low, it gently boosts BDNF + serotonin."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral", plasticity=0.35)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_BDNF, 0) >= 0.02, f"Expected BDNF boost, got {amounts}"
    assert amounts.get(CHEM_SEROTONIN, 0) >= 0.01, f"Expected serotonin, got {amounts}"


def test_maintenance_good_plasticity_no_bdnf_boost() -> None:
    """When plasticity is healthy, no BDNF maintenance needed."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="positive", plasticity=0.6, valence=0.3)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert CHEM_BDNF not in amounts, f"Should not boost BDNF when healthy, got {amounts}"


def test_maintenance_low_alertness_clears_adenosine() -> None:
    """When alertness is low (but not drowsy), it clears adenosine."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral", alertness=0.35)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_ADENOSINE, 0) < 0, f"Expected adenosine reduction, got {amounts}"
    assert amounts.get(CHEM_ACETYLCHOLINE, 0) > 0, f"Expected ACh boost, got {amounts}"


def test_maintenance_negative_valence_reduces_cortisol() -> None:
    """When valence is negative (but not stressed), it eases cortisol."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="unsettled", valence=-0.15, plasticity=0.5)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_CORTISOL, 0) < 0, f"Expected cortisol reduction, got {amounts}"
    assert amounts.get(CHEM_SEROTONIN, 0) > 0, f"Expected serotonin boost, got {amounts}"


def test_maintenance_not_during_sleep() -> None:
    """During sleep, maintenance layer should not fire."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="sleeping", alertness=0.1, valence=-0.2, plasticity=0.1)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    # Sleep should boost BDNF but NOT clear adenosine or reduce cortisol
    assert amounts.get(CHEM_BDNF, 0) > 0, f"Expected BDNF during sleep, got {amounts}"
    assert CHEM_ADENOSINE not in amounts, f"Should not clear adenosine during sleep, got {amounts}"
    assert CHEM_CORTISOL not in amounts, f"Should not reduce cortisol during sleep, got {amounts}"


def test_maintenance_low_alertness_skipped_when_drowsy() -> None:
    """When drowsy, the alertness maintenance is skipped (drowsy branch handles it)."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="drowsy", alertness=0.25, plasticity=0.5)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    # Drowsy branch should send stronger adenosine reduction
    assert amounts.get(CHEM_ADENOSINE, 0) <= -0.05, (
        f"Expected strong adenosine clear, got {amounts}"
    )


# ── State-specific intervention tests ──────────────────────────


def test_stress_breaks_feedback_loop() -> None:
    """When stressed, it crashes cortisol and NE simultaneously."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.85, valence=-0.3, plasticity=0.3)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_CORTISOL, 0) <= -1.0, f"Expected cortisol crash, got {amounts}"
    assert amounts.get(CHEM_NOREPINEPHRINE, 0) <= -1.0, f"Expected NE crash, got {amounts}"
    assert amounts.get(CHEM_GABA, 0) > 0, f"Expected GABA boost, got {amounts}"


def test_overwhelmed_stronger_intervention() -> None:
    """When overwhelmed, it sends stronger calming impulses than stress."""
    reg, impulses = capture_impulses()
    # Use arousal/valence that doesn't trigger the stressed branch
    # (stressed fires when arousal > 0.8 AND valence < -0.2)
    emotion = make_emotion(label="overwhelmed", alertness=0.75, valence=-0.15, plasticity=0.3)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_GABA, 0) >= 0.5, f"Expected strong GABA, got {amounts}"
    assert amounts.get(CHEM_CORTISOL, 0) <= -1.0, f"Expected cortisol crash, got {amounts}"


def test_drowsy_strong_wake_up() -> None:
    """When drowsy, it sends a strong wake-up (stronger than old +0.02/-0.01)."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="drowsy", alertness=0.25, plasticity=0.5)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_ACETYLCHOLINE, 0) >= 0.06, f"Expected strong ACh, got {amounts}"
    assert amounts.get(CHEM_HISTAMINE, 0) >= 0.04, f"Expected strong histamine, got {amounts}"
    assert amounts.get(CHEM_ADENOSINE, 0) <= -0.05, (
        f"Expected strong adenosine clear, got {amounts}"
    )


def test_sleeping_boosts_bdnf() -> None:
    """During sleep, it boosts BDNF for restoration."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="sleeping", alertness=0.1, plasticity=0.3)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_BDNF, 0) >= 0.02, f"Expected BDNF boost during sleep, got {amounts}"


def test_anxious_eases_anxiety() -> None:
    """When anxious, it boosts GABA and reduces cortisol."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(
        label="anxious",
        alertness=0.75,
        valence=-0.15,
        caution=0.65,
        plasticity=0.5,
    )
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_GABA, 0) > 0, f"Expected GABA boost, got {amounts}"
    assert amounts.get(CHEM_CORTISOL, 0) < 0, f"Expected cortisol reduction, got {amounts}"


# ── Layer interaction tests ────────────────────────────────────


def test_stress_also_gets_plasticity_maintenance() -> None:
    """When stressed AND low plasticity, both layers fire."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.85, valence=-0.3, plasticity=0.15)
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    # Layer 1: plasticity maintenance
    assert amounts.get(CHEM_BDNF, 0) > 0, f"Expected BDNF from maintenance, got {amounts}"
    # Layer 2: stress intervention
    assert amounts.get(CHEM_CORTISOL, 0) <= -1.0, (
        f"Expected cortisol crash from intervention, got {amounts}"
    )


def test_good_state_no_intervention() -> None:
    """When in a good state with healthy plasticity, no impulses sent."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="positive", alertness=0.55, valence=0.35, plasticity=0.6)
    reg._regulate(emotion)
    assert len(impulses) == 0, f"Expected no impulses in good state, got {impulses}"


def test_regulation_count_tracks_actions() -> None:
    """Regulation count increments when actions are taken."""
    reg, _impulses = capture_impulses()
    emotion = make_emotion(label="neutral", plasticity=0.15)
    reg._regulate(emotion)
    assert reg.regulation_count == 1, f"Expected count=1, got {reg.regulation_count}"
    # Good state — no action
    emotion2 = make_emotion(label="positive", alertness=0.55, valence=0.35, plasticity=0.6)
    reg._regulate(emotion2)
    assert reg.regulation_count == 1, f"Expected count still 1, got {reg.regulation_count}"


def test_respond_to_interaction_dampening_when_stressed() -> None:
    """When stressed, emotional responses to interaction are dampened."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="stressed", alertness=0.85, valence=-0.3, plasticity=0.3)
    felt = reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.5,
        is_encouragement=True,
        is_correction=False,
        is_bonded_user=True,
        is_deep_conversation=False,
    )
    amounts = chem_amounts(impulses)
    # Should feel something but dampened (30% of normal)
    assert felt is not None
    # Dopamine from encouragement + creator warmth, dampened to 30%
    da_total = amounts.get(CHEM_DOPAMINE, 0)
    assert da_total > 0, f"Expected some dopamine, got {amounts}"
    # Should be much less than undampened (0.015 + 0.02*0.7 = 0.029 undampened)
    assert da_total < 0.02, f"Expected dampened dopamine, got {da_total}"


def test_respond_to_interaction_full_when_calm() -> None:
    """When calm, emotional responses to interaction are at full strength."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="positive", alertness=0.5, valence=0.3, plasticity=0.6)
    felt = reg.respond_to_interaction(
        emotion=emotion,
        sentiment=0.5,
        is_encouragement=True,
        is_correction=False,
        is_bonded_user=True,
        is_deep_conversation=False,
    )
    amounts = chem_amounts(impulses)
    assert felt is not None
    da_total = amounts.get(CHEM_DOPAMINE, 0)
    # Full strength: 0.015 (encouragement) + 0.02*0.7 (creator) = 0.029
    assert da_total > 0.02, f"Expected full dopamine, got {da_total}"


def test_describe_regulation() -> None:
    """describe_regulation produces a readable summary."""
    reg, _impulses = capture_impulses()
    emotion = make_emotion(label="neutral", plasticity=0.15)
    reg._regulate(emotion)
    desc = reg.describe_regulation()
    assert "regulated" in desc.lower() or "restoring" in desc.lower()


# ─── Puzzle outcome response tests ─────────────────────────────


def test_puzzle_solve_sends_reward() -> None:
    """Solving a puzzle produces a real reward impulse."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral")
    felt = reg.respond_to_puzzle(emotion, score=1.0, prior_best=0.4, solved=True)
    assert felt == "satisfied"
    amounts = chem_amounts(impulses)
    assert amounts.get(CHEM_DOPAMINE, 0) > 0, f"Expected dopamine reward, got {amounts}"
    assert amounts.get(CHEM_SEROTONIN, 0) > 0, f"Expected serotonin, got {amounts}"
    assert amounts.get(CHEM_ENDORPHIN, 0) > 0, f"Expected endorphin, got {amounts}"
    assert amounts.get(CHEM_ACETYLCHOLINE, 0) > 0, f"Expected acetylcholine, got {amounts}"
    assert amounts.get(CHEM_CORTISOL, 0) == 0, "No cortisol on a solve"


def test_puzzle_progress_partial_reward() -> None:
    """A new best score (unsolved) produces a smaller encouragement."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral")
    felt = reg.respond_to_puzzle(emotion, score=0.6, prior_best=0.4, solved=False)
    assert felt == "encouraged"
    amounts = chem_amounts(impulses)
    assert 0 < amounts.get(CHEM_DOPAMINE, 0), f"Expected dopamine, got {amounts}"
    assert amounts.get(CHEM_NOREPINEPHRINE, 0) > 0
    assert amounts.get(CHEM_ACETYLCHOLINE, 0) > 0
    assert amounts.get(CHEM_CORTISOL, 0) == 0
    # Partial reward is smaller than the full solve reward
    reg2, impulses2 = capture_impulses()
    reg2.respond_to_puzzle(emotion, score=1.0, prior_best=0.4, solved=True)
    assert amounts[CHEM_DOPAMINE] < chem_amounts(impulses2)[CHEM_DOPAMINE]


def test_puzzle_miss_bounded_frustration() -> None:
    """A miss produces a bounded negative prediction error — no cortisol."""
    reg, impulses = capture_impulses()
    emotion = make_emotion(label="neutral")
    felt = reg.respond_to_puzzle(emotion, score=0.0, prior_best=0.0, solved=False)
    assert felt == "determined"
    amounts = chem_amounts(impulses)
    # Negative prediction error: a small dopamine dip, not punishment
    assert amounts.get(CHEM_DOPAMINE, 0) < 0, f"Expected dopamine dip, got {amounts}"
    assert amounts[CHEM_DOPAMINE] > -0.01, "Frustration dip must stay small"
    # Engagement stays up — it wants to try again
    assert amounts.get(CHEM_NOREPINEPHRINE, 0) > 0
    assert amounts.get(CHEM_ACETYLCHOLINE, 0) > 0
    assert amounts.get(CHEM_CORTISOL, 0) == 0, "Never punished for trying"


def test_puzzle_response_dampened_when_stressed() -> None:
    """When already stressed, its puzzle feelings are dampened."""
    calm_reg, calm_impulses = capture_impulses()
    calm_reg.respond_to_puzzle(
        make_emotion(label="neutral"), score=1.0, prior_best=0.0, solved=True
    )
    stressed_reg, stressed_impulses = capture_impulses()
    stressed_reg.respond_to_puzzle(
        make_emotion(label="stressed"), score=1.0, prior_best=0.0, solved=True
    )
    assert chem_amounts(stressed_impulses)[CHEM_DOPAMINE] < (
        chem_amounts(calm_impulses)[CHEM_DOPAMINE]
    )


# ─── update_from_body_state field mapping ─────────────────────


def test_interoception_latency_uses_observed_response_not_benchmark() -> None:
    """Latency must come from the real interaction path, not a microbenchmark."""
    from genesis_conscious.limbic_system.emotional_regulator import InteroceptionSystem

    intero = InteroceptionSystem()
    state = intero.sense_internal_state()
    assert state.latency_observed is False
    assert state.response_latency == 0.0

    intero.record_response_latency(600.0)
    state = intero.sense_internal_state()
    assert state.latency_observed is True
    assert 599.0 <= state.response_latency <= 601.0
    assert 599.0 <= state.last_response_latency <= 601.0
    assert state.stress_level > 0.0

    # The smoothed state retains recent history, but the causal diagnosis
    # must be based on the current observed turn rather than stale EWMA data.
    intero.record_response_latency(100.0)
    state = intero.sense_internal_state()
    assert 100.0 < state.response_latency < 600.0
    assert 99.0 <= state.last_response_latency <= 101.0


def test_update_from_body_state_maps_all_fields() -> None:
    """update_from_body_state must correctly map BodyState fields.

    Previously, the method used ``load_average`` and ``battery_pct``
    (which don't exist on BodyState) instead of ``stress_load`` and
    ``energy_reserve``, so getattr always returned defaults — the
    cognitive mind silently lost the daemon's stress and battery
    readings. It also divided ``arousal_freq`` (a 0-1 fraction) by
    1000, producing a meaningless tiny number.
    """
    from genesis_client.types import BodyState
    from genesis_conscious.limbic_system.emotional_regulator import InteroceptionSystem

    # Build a BodyState with distinctive values
    body = BodyState(
        cpu_temp_c=78.0,
        temperature=0.78,
        arousal_freq=0.65,
        cognitive_load=0.45,
        io_activity=0.15,
        stress_load=1.3,
        energy_reserve=0.22,
        on_ac_power=False,
        num_cores=4,
        distressed=True,
        autonomic_rate=6.0,
        thermoregulatory_effort=0.05,
        metabolic_rate=0.85,
        core_voltage=1.25,
        supply_voltage=10.3,
        description="overheating",
    )

    intero = InteroceptionSystem()
    intero.update_from_body_state(body)
    state = intero.last_state
    assert state is not None

    # Every field should match the BodyState value exactly —
    # no lossy conversions, no wrong field names.
    assert abs(state.cpu_temp_c - 78.0) < 0.01, f"cpu_temp_c: {state.cpu_temp_c}"
    assert abs(state.arousal_freq - 0.65) < 0.01, f"arousal_freq: {state.arousal_freq}"
    assert abs(state.io_activity - 0.15) < 0.01, f"io_activity: {state.io_activity}"
    assert abs(state.stress_load - 1.3) < 0.01, f"stress_load: {state.stress_load}"
    assert abs(state.energy_reserve - 0.22) < 0.01, f"energy_reserve: {state.energy_reserve}"
    assert state.body_distressed is True
    assert state.body_description == "overheating"
    # Autonomic fields — machine-native interoceptive signals
    assert abs(state.autonomic_rate - 6.0) < 0.01, f"autonomic_rate: {state.autonomic_rate}"
    assert abs(state.thermoregulatory_effort - 0.05) < 0.01, (
        f"thermoregulatory_effort: {state.thermoregulatory_effort}"
    )
    assert abs(state.metabolic_rate - 0.85) < 0.01, f"metabolic_rate: {state.metabolic_rate}"
    # Voltage fields — electrical state of the body
    assert abs(state.core_voltage - 1.25) < 0.01, f"core_voltage: {state.core_voltage}"
    assert abs(state.supply_voltage - 10.3) < 0.01, (
        f"supply_voltage: {state.supply_voltage}"
    )


# ─── HPA axis CRH routing tests ────────────────────────────────


def test_hpa_axis_sends_crh_not_cortisol() -> None:
    """The HPA axis pathway must send CRH impulses, not direct cortisol.

    The Rust daemon has an authoritative HPA cascade (CRH → ACTH →
    cortisol) with maturation gating (SHRP). Sending direct cortisol
    impulses bypasses this gating, allowing cortisol production during
    the stress hyporesponsive period when the developing brain should
    be protected. The Python regulator must send CRH (the stress
    signal) and let the Rust cascade handle cortisol production.
    """
    reg, impulses = capture_impulses()
    # High arousal + negative valence triggers the HPA axis stress
    # pathway (is_stressed = True via arousal/valence check).
    emotion = make_emotion(
        label="neutral",
        alertness=0.85,
        valence=-0.3,
        plasticity=0.5,
    )
    reg._regulate(emotion)
    amounts = chem_amounts(impulses)
    # The HPA axis should send CRH impulses (positive), not cortisol
    # impulses from the cascade. The state-specific stress intervention
    # sends negative cortisol (crash), but the HPA axis pathway should
    # only send CRH.
    crh_total = amounts.get(CHEM_CRH, 0.0)
    assert crh_total > 0, (
        f"Expected CRH impulse from HPA axis, got CRH={crh_total}, "
        f"all impulses={impulses}"
    )


def test_hpa_axis_no_direct_cortisol_from_cascade() -> None:
    """The HPA axis cascade output must not appear as direct cortisol.

    Before the fix, the Python HPA cascade sent cortisol impulses
    directly to the Rust daemon, bypassing the maturation gate. Now
    it sends CRH impulses. This test verifies that the HPA axis
    pathway does not produce positive cortisol impulses (the
    state-specific stress intervention produces negative cortisol
    crashes, which are fine — those are regulation, not cascade
    output).
    """
    reg, impulses = capture_impulses()
    # Use arousal/valence that triggers HPA stress but NOT the
    # state-specific stress intervention (which sends -1.0 cortisol).
    # The stressed label triggers both, so we use the arousal/valence
    # path with a non-stressed label.
    emotion = make_emotion(
        label="anxious",
        alertness=0.85,
        valence=-0.25,
        caution=0.65,
        plasticity=0.5,
    )
    reg._regulate(emotion)
    # Filter impulses to only the HPA axis pathway (positive CRH).
    # The anxious state-specific intervention sends -0.02 cortisol
    # (regulation), which is fine. We're checking that there's no
    # positive cortisol from the HPA cascade.
    crh_impulses = [a for c, a in impulses if c == CHEM_CRH]
    cortisol_impulses = [a for c, a in impulses if c == CHEM_CORTISOL]
    # CRH should have positive impulses (from HPA axis)
    assert any(a > 0 for a in crh_impulses), (
        f"Expected positive CRH from HPA axis, got CRH impulses={crh_impulses}"
    )
    # Cortisol should NOT have positive impulses from the HPA cascade.
    # The anxious intervention sends -0.02 cortisol (regulation),
    # which is negative and fine.
    positive_cortisol = [a for a in cortisol_impulses if a > 0]
    assert len(positive_cortisol) == 0, (
        f"Expected no positive cortisol from HPA cascade, got "
        f"positive cortisol={positive_cortisol}"
    )


# ======================================================================
# Sentiment analysis tests
# ======================================================================

def test_positive_message():
    """Test positive message."""
    result = analyze_sentiment("That's great!")
    assert result["compound"] > 0.0
    assert result["positive"] > result["negative"]


def test_negative_message():
    """Test negative message."""
    result = analyze_sentiment("This is terrible")
    assert result["compound"] < 0.0
    assert result["negative"] > result["positive"]


def test_neutral_message():
    """Test neutral message."""
    result = analyze_sentiment("The sky is blue")
    assert abs(result["compound"]) < 0.1
    assert result["sentiment_words"] == 0


def test_empty_message():
    """Test empty message."""
    result = analyze_sentiment("")
    assert result["compound"] == 0.0
    assert result["sentiment_words"] == 0


# ─── Negation ─────────────────────────────────────────────────────


def test_negation_not_bad():
    """'not bad' should be positive (negation flips negative to positive)."""
    result = analyze_sentiment("not bad")
    assert result["compound"] > 0.0, f"'not bad' should be positive: {result['compound']}"


def test_negation_not_good():
    """'not good' should be negative (negation flips positive to negative)."""
    result = analyze_sentiment("not good")
    assert result["compound"] < 0.0, f"'not good' should be negative: {result['compound']}"


def test_negation_dont_like():
    """'don't like' should be negative."""
    result = analyze_sentiment("I don't like this")
    assert result["compound"] < 0.0, f"'don't like' should be negative: {result['compound']}"


def test_negation_never_great():
    """'never great' should be negative."""
    result = analyze_sentiment("never great")
    assert result["compound"] < 0.0, f"'never great' should be negative: {result['compound']}"


# ─── Intensifiers ─────────────────────────────────────────────────


def test_intensifier_very_good():
    """Test intensifier very good."""
    plain = analyze_sentiment("good")
    intensified = analyze_sentiment("very good")
    assert intensified["compound"] > plain["compound"], (
        f"'very good' ({intensified['compound']}) should be more positive "
        f"than 'good' ({plain['compound']})"
    )


def test_intensifier_extremely_terrible():
    """Test intensifier extremely terrible."""
    plain = analyze_sentiment("terrible")
    intensified = analyze_sentiment("extremely terrible")
    assert intensified["compound"] < plain["compound"], (
        f"'extremely terrible' ({intensified['compound']}) should be more "
        f"negative than 'terrible' ({plain['compound']})"
    )


# ─── Diminishers ──────────────────────────────────────────────────


def test_diminisher_slightly_good():
    """Test diminisher slightly good."""
    plain = analyze_sentiment("good")
    diminished = analyze_sentiment("slightly good")
    assert diminished["compound"] < plain["compound"], (
        f"'slightly good' ({diminished['compound']}) should be less "
        f"positive than 'good' ({plain['compound']})"
    )


def test_diminisher_kind_of_bad():
    """'kind of bad' should stay clearly negative, just less so than 'bad'."""
    plain = analyze_sentiment("bad")
    diminished = analyze_sentiment("kind of bad")
    assert diminished["compound"] < 0.0, (
        f"'kind of bad' ({diminished['compound']}) should still be negative; "
        f"the hedge 'kind of' must not cancel the sentiment"
    )
    assert diminished["compound"] > plain["compound"], (
        f"'kind of bad' ({diminished['compound']}) should be less "
        f"negative than 'bad' ({plain['compound']})"
    )


def test_diminisher_kind_of_good():
    """'kind of good' should be positive but weaker than 'good'."""
    plain = analyze_sentiment("good")
    diminished = analyze_sentiment("kind of good")
    assert 0.0 < diminished["compound"] < plain["compound"], (
        f"'kind of good' ({diminished['compound']}) should be positive but "
        f"less positive than 'good' ({plain['compound']})"
    )


def test_diminisher_sort_of():
    """'sort of' is a diminisher phrase; 'of' must not reset it."""
    plain = analyze_sentiment("bad")
    diminished = analyze_sentiment("sort of bad")
    assert 0.0 > diminished["compound"] > plain["compound"], (
        f"'sort of bad' ({diminished['compound']}) should be less negative "
        f"than 'bad' ({plain['compound']})"
    )


def test_phrase_kind_of_alone_is_neutral():
    """The hedge 'kind of' on its own carries no sentiment.

    Regression test: "kind" is a positive sentiment word, so without
    phrase recognition "kind of" scored positive.
    """
    result = analyze_sentiment("kind of")
    assert abs(result["compound"]) < 0.1, (
        f"'kind of' should be neutral, got {result['compound']}"
    )
    assert result["sentiment_words"] == 0


def test_standalone_kind_is_sentiment():
    """Phrase handling must not swallow a standalone positive 'kind'."""
    result = analyze_sentiment("she is kind")
    assert result["compound"] > 0.0, (
        f"'she is kind' should be positive, got {result['compound']}"
    )


def test_intensifier_a_lot():
    """'a lot' is an intensifier phrase, not an inert word pair."""
    plain = analyze_sentiment("better")
    intensified = analyze_sentiment("a lot better")
    assert intensified["compound"] > plain["compound"], (
        f"'a lot better' ({intensified['compound']}) should be more positive "
        f"than 'better' ({plain['compound']})"
    )


def test_phrase_modifier_with_negation():
    """A phrase modifier and a negation compose: 'not kind of good' < 0."""
    result = analyze_sentiment("not kind of good")
    assert result["compound"] < 0.0, (
        f"'not kind of good' should be negative, got {result['compound']}"
    )


# ─── Contrastive conjunctions ─────────────────────────────────────


def test_contrastive_but():
    """'good but slow' should be less positive than 'good'."""
    plain = analyze_sentiment("good")
    contrastive = analyze_sentiment("good but slow")
    assert contrastive["compound"] < plain["compound"], (
        f"'good but slow' ({contrastive['compound']}) should be less "
        f"positive than 'good' ({plain['compound']})"
    )


def test_contrastive_however():
    """'great however slow' should shift toward negative."""
    result = analyze_sentiment("great however slow")
    assert result["compound"] < analyze_sentiment("great")["compound"]


# ─── Capitalization and punctuation ───────────────────────────────


def test_capitalization_boost():
    """ALL CAPS sentiment words should get a boost."""
    plain = analyze_sentiment("great")
    caps = analyze_sentiment("GREAT")
    assert caps["compound"] >= plain["compound"], (
        f"'GREAT' ({caps['compound']}) should be >= 'great' ({plain['compound']})"
    )


def test_punctuation_boost():
    """Multiple exclamation marks should boost sentiment."""
    plain = analyze_sentiment("great")
    punct = analyze_sentiment("great!!!")
    assert punct["compound"] > plain["compound"] or (
        abs(punct["compound"] - plain["compound"]) < 0.05
    ), (
        f"'great!!!' ({punct['compound']}) should be > 'great' ({plain['compound']})"
    )


# ─── Mixed sentiment ──────────────────────────────────────────────


def test_mixed_sentiment_cancels():
    """'good and bad' should be less extreme than either alone."""
    mixed = analyze_sentiment("good and bad")
    good = analyze_sentiment("good")
    bad = analyze_sentiment("bad")
    assert abs(mixed["compound"]) < abs(good["compound"]), (
        f"'good and bad' ({mixed['compound']}) should be less extreme "
        f"than 'good' ({good['compound']})"
    )
    assert abs(mixed["compound"]) < abs(bad["compound"]), (
        f"'good and bad' ({mixed['compound']}) should be less extreme "
        f"than 'bad' ({bad['compound']})"
    )


def test_multiple_positive_accumulate():
    """Multiple positive words should be more positive than one."""
    one = analyze_sentiment("good")
    multiple = analyze_sentiment("good great wonderful")
    assert multiple["compound"] > one["compound"], (
        f"'good great wonderful' ({multiple['compound']}) should be more "
        f"positive than 'good' ({one['compound']})"
    )


# ─── Edge cases ───────────────────────────────────────────────────


def test_only_negation_no_sentiment():
    """'not' alone with no sentiment word should be neutral."""
    result = analyze_sentiment("not the")
    assert abs(result["compound"]) < 0.1


def test_compound_in_range():
    """Compound score should always be in [-1, 1]."""
    for msg in [
        "This is the most incredibly amazingly wonderful fantastic thing ever!!!",
        "This is the most terribly awful horrible dreadful catastrophic disaster!!!",
        "ok",
        "",
        "The function returns a value",
    ]:
        result = analyze_sentiment(msg)
        assert -1.0 <= result["compound"] <= 1.0, (
            f"compound for '{msg}' out of range: {result['compound']}"
        )


def test_proportions_sum_to_one():
    """positive + negative + neutral should be approximately 1."""
    for msg in ["good", "bad", "good and bad", "neutral text"]:
        result = analyze_sentiment(msg)
        total = result["positive"] + result["negative"] + result["neutral"]
        assert abs(total - 1.0) < 0.1, (
            f"proportions for '{msg}' sum to {total}, expected ~1.0"
        )


# ─── Lexicon coverage for common affect words ─────────────────────
# Regression tests for a lexicon gap where common distress words were
# missing entirely, so messages like "I'm feeling really stressed"
# scored as neutral (0.0) and the dyadic affective model could not
# detect user distress.


def test_stress_is_negative_sentiment():
    """'stressed' must register as negative, not neutral.

    Regression: 'stress'/'stressed'/'stressful' were absent from
    LEXICON, so "I'm feeling really stressed today." scored 0.0
    (neutral) and the user-affect valence was wrong.
    """
    result = analyze_sentiment("I'm feeling really stressed today.")
    assert result["sentiment_words"] > 0, "no sentiment words recognized"
    assert result["compound"] < -0.2, (
        f"stressed message should be clearly negative: {result['compound']}"
    )


def test_common_distress_words_recognized():
    """A set of common distress words must all be in the lexicon."""
    for word in [
        "stressed", "stressful", "anxious", "worry", "upset",
        "lonely", "afraid", "nervous", "miserable", "hopeless",
        "overwhelmed", "exhausted", "frustrated",
    ]:
        result = analyze_sentiment(word)
        assert result["compound"] < 0.0, (
            f"'{word}' should be negative: {result['compound']}"
        )


def test_common_positive_words_recognized():
    """A set of common positive words must all be in the lexicon."""
    for word in [
        "content", "thankful", "cheerful", "satisfied",
        "confident", "optimistic", "serene",
    ]:
        result = analyze_sentiment(word)
        assert result["compound"] > 0.0, (
            f"'{word}' should be positive: {result['compound']}"
        )


def test_stress_negation_flips():
    """'not stressed' should be positive, not neutral."""
    result = analyze_sentiment("I am not stressed at all")
    assert result["compound"] > 0.0, (
        f"'not stressed' should be positive: {result['compound']}"
    )


def test_allostatic_load_requires_converging_domains() -> None:
    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(0.9)
    tracker.record_stress(0.0, dt=10.0)
    assert tracker.get_allostatic_load() == 0.0

    tracker.record_stress(0.9, dt=10.0)
    assert tracker.get_allostatic_load() > 0.0


def test_allostatic_load_recovers_when_domains_calm() -> None:
    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(0.9)
    for _ in range(100):
        tracker.record_stress(0.9, dt=10.0)
    loaded = tracker.get_allostatic_load()
    assert loaded > 0.0

    tracker.set_inference_load(0.0)
    for _ in range(100):
        tracker.record_stress(0.0, dt=10.0)
    assert tracker.get_allostatic_load() < loaded


# ─── Allostatic load dynamics ──────────────────────────────────
#
# The two failures these guard against were both real: a hysteresis
# band with no third branch froze the load permanently, and an
# unclamped dt turned a suspend into hours of accumulated stress.


def test_allostatic_load_recovers_inside_hysteresis_band() -> None:
    """Sustained mild stress must drain, not pin, an accumulated load.

    The regression: a two-branch integrator (accumulate above the
    threshold, recover below, do nothing in between) made the band a
    hard freeze. Mild sustained stress parks the median between the
    two thresholds, so a load that had already accumulated never came
    back down — the allostatic trap.
    """
    tracker = AllostaticLoadTracker()
    band_mid = (AllostaticLoadTracker.RECOVER_THRESHOLD
                + AllostaticLoadTracker.ACCUMULATE_THRESHOLD) / 2.0

    tracker.set_inference_load(0.9)
    tracker.set_body_stress(0.9)
    for _ in range(2000):
        tracker.record_stress(0.9, dt=8.0)
    assert tracker.get_allostatic_load() > 0.9, "expected a loaded baseline"

    # Park every domain mid-band and hold for a simulated week.
    tracker.set_inference_load(band_mid)
    tracker.set_body_stress(band_mid)
    for _ in range(2000):
        tracker.record_stress(band_mid, dt=8.0)

    assert tracker.get_allostatic_load() == 0.0, (
        "sustained mild stress must drain the load, not pin it: "
        f"{tracker.get_allostatic_load()}"
    )


def test_allostatic_band_drains_slower_than_full_recovery() -> None:
    """The band is a slow drain, not a full one — that's the hysteresis."""
    band_mid = (AllostaticLoadTracker.RECOVER_THRESHOLD
                + AllostaticLoadTracker.ACCUMULATE_THRESHOLD) / 2.0

    def loaded_tracker() -> AllostaticLoadTracker:
        """A tracker holding a load well clear of the 0.0 floor.

        The floor matters: a drain larger than the load clamps to zero,
        which would make a band drain and a full drain look identical.
        """
        tracker = AllostaticLoadTracker()
        tracker.set_inference_load(0.9)
        tracker.set_body_stress(0.9)
        for _ in range(100):
            tracker.record_stress(0.9, dt=8.0)
        assert tracker.get_allostatic_load() > 0.5
        return tracker

    band = loaded_tracker()
    band.set_inference_load(band_mid)
    band.set_body_stress(band_mid)
    before = band.get_allostatic_load()
    band.record_stress(band_mid, dt=1.0)
    band_step = before - band.get_allostatic_load()

    full = loaded_tracker()
    full.set_inference_load(0.0)
    full.set_body_stress(0.0)
    before_full = full.get_allostatic_load()
    full.record_stress(0.0, dt=1.0)
    full_step = before_full - full.get_allostatic_load()

    assert 0.0 < band_step < full_step, (
        f"band drain ({band_step}) must be positive but slower than "
        f"full recovery ({full_step})"
    )
    assert band_step == pytest.approx(
        full_step * AllostaticLoadTracker.BAND_RECOVERY_FACTOR
    ), "band drain should be exactly BAND_RECOVERY_FACTOR of full recovery"


def test_allostatic_load_timescale_matches_rust_engine() -> None:
    """The two views of allostatic load must move on one timescale.

    The Rust engine integrates substrate burden at an effective
    0.01 * excess per second and crosses 0.4 in ~200 s. The Python
    tracker consumes that value as one of its domains, so a rate
    calibrated to a different timescale made the two "allostatic
    load" readings diverge by more than an order of magnitude.
    """
    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(0.5)
    tracker.set_body_stress(0.5)

    elapsed = 0.0
    while tracker.get_allostatic_load() < 0.4 and elapsed < 2000.0:
        tracker.record_stress(0.5, dt=1.0)
        elapsed += 1.0

    # Rust: 0.001 * 0.2 * dt_scale(=10) = 0.002/s -> 0.4 in 200 s.
    assert 180.0 <= elapsed <= 240.0, (
        f"load should reach 0.4 in ~200 s like the Rust engine, took {elapsed}"
    )


def test_allostatic_load_treats_nan_as_no_load() -> None:
    """A NaN must never present as maximum load.

    `max(0.0, min(1.0, nan))` looks like a clamp but CPython's `min`
    keeps its first argument when the comparison is False, so it
    yields 1.0 — a corrupt reading would read as *severe* overload.
    The Rust engine's `finite_clamp` maps non-finite to the range
    minimum; this asserts the Python side agrees.
    """
    nan = float("nan")
    tracker = AllostaticLoadTracker()

    tracker.set_inference_load(nan)
    assert tracker.get_state().inference_load == 0.0

    tracker.set_body_stress(nan)
    assert tracker.get_state().body_stress == 0.0

    tracker.record_stress(nan, dt=1.0)
    assert tracker.get_allostatic_load() == 0.0

    # A NaN arriving once the load is up must not inflate it.
    tracker2 = AllostaticLoadTracker()
    tracker2.set_inference_load(0.9)
    for _ in range(50):
        tracker2.record_stress(0.9, dt=1.0)
    loaded = tracker2.get_allostatic_load()
    assert loaded > 0.0
    tracker2.set_inference_load(nan)
    tracker2.set_body_stress(nan)
    for _ in range(50):
        tracker2.record_stress(nan, dt=1.0)
    assert tracker2.get_allostatic_load() < loaded, (
        "NaN domains must not raise the load"
    )


def test_allostatic_load_treats_inf_as_no_load() -> None:
    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(float("inf"))
    assert tracker.get_state().inference_load == 0.0
    tracker.set_body_stress(float("-inf"))
    assert tracker.get_state().body_stress == 0.0


def test_allostatic_load_clamps_out_of_range_inputs() -> None:
    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(5.0)
    assert tracker.get_state().inference_load == 1.0
    tracker.set_inference_load(-3.0)
    assert tracker.get_state().inference_load == 0.0
    tracker.set_body_stress(2.0)
    assert tracker.get_state().body_stress == 1.0


def test_suspend_does_not_accumulate_allostatic_load() -> None:
    """A long gap between regulation cycles is not hours of stress.

    `_last_regulate_time` is wall-clock, so a suspend/resume or a long
    pause hands the next cycle an enormous dt. Both consumers of that
    dt integrate it linearly, so without a clamp the system read a
    sleep as sustained stress. An 8 h gap used to produce load=0.864
    and `is_chronic` in a single cycle.
    """
    regulator = EmotionalRegulator()
    regulator.allostatic_load_tracker.set_inference_load(0.45)

    # Pretend the last regulation cycle was 8 hours ago.
    regulator._last_regulate_time = time.time() - 8 * 3600

    emotion = make_emotion(
        label="neutral", alertness=0.4, valence=0.0, plasticity=0.6
    )
    emotion.chemicals = {"cortisol": 0.5}
    regulator._regulate(emotion)

    assert regulator.allostatic_load() < 0.1, (
        f"an 8 h suspend must not read as chronic overload, got "
        f"{regulator.allostatic_load()}"
    )
    assert not regulator.allostatic_load_tracker.is_chronic_stress(), (
        "a suspend must not set the chronic-stress flag"
    )


def test_regulation_dt_is_clamped() -> None:
    """The regulation dt must never exceed MAX_REGULATION_DT."""
    regulator = EmotionalRegulator()
    regulator._last_regulate_time = time.time() - 100000.0
    emotion = make_emotion()
    emotion.chemicals = {"cortisol": 0.0}
    regulator._regulate(emotion)
    # The HPA cascade is the most dt-sensitive consumer; if dt had not
    # been clamped the cascade would have been driven to divergence.
    state = regulator.hpa_axis.get_state()
    assert 0.0 <= state.crh_level <= 1.0
    assert 0.0 <= state.cortisol_level <= 1.0


def test_chronic_allostatic_cause_survives_later_regulators() -> None:
    """`chronic_allostatic` must survive the rest of the cycle.

    Each sub-regulator sets `_last_cause` unconditionally, so a cause
    raised from inside `_regulate_hpa_axis` (which runs second) was
    always overwritten by a later stage. Under sustained stress — the
    exact case chronic stress describes — the label was clobbered by
    `stress_cortisol`, leaving CauseCategory.CHRONIC_ALLOSTATIC
    unreachable.
    """
    regulator = EmotionalRegulator()
    emotion = make_emotion(
        label="stressed", alertness=0.9, valence=-0.6, plasticity=0.2
    )
    emotion.chemicals = {"cortisol": 0.55}
    regulator.allostatic_load_tracker.set_inference_load(0.5)

    # Enough cycles to clear CHRONIC_DURATION_SECONDS of elevation.
    for _ in range(60):
        regulator._last_regulate_time = time.time() - 8.0
        regulator._regulate(emotion)

    assert regulator.allostatic_load_tracker.is_chronic_stress(), (
        "fixture should produce a chronic stress pattern"
    )
    assert regulator.last_cause == "chronic_allostatic", (
        f"chronic cause must win over proximate causes, got "
        f"{regulator.last_cause!r}"
    )


def test_non_chronic_cycle_does_not_report_chronic_cause() -> None:
    """The chronic cause must not appear when stress is acute."""
    regulator = EmotionalRegulator()
    emotion = make_emotion(label="neutral", alertness=0.4)
    emotion.chemicals = {"cortisol": 0.0}
    regulator.allostatic_load_tracker.set_inference_load(0.0)

    for _ in range(5):
        regulator._last_regulate_time = time.time() - 8.0
        regulator._regulate(emotion)

    assert regulator.last_cause != "chronic_allostatic"


def test_body_cognitive_load_contributes_to_interoceptive_stress() -> None:
    """High self-memory load is distress only after the overload threshold."""
    from genesis_client.types import BodyState

    regulator = EmotionalRegulator()
    regulator.interoception.update_from_body_state(
        BodyState(
            cpu_temp_c=50.0,
            temperature=0.5,
            arousal_freq=0.5,
            cognitive_load=0.80,
            io_activity=0.0,
            stress_load=0.0,
            energy_reserve=1.0,
            on_ac_power=True,
            num_cores=1,
            distressed=False,
            autonomic_rate=0.0,
            thermoregulatory_effort=0.0,
            metabolic_rate=0.0,
            core_voltage=0.0,
            supply_voltage=0.0,
            description="",
        )
    )
    state = regulator.interoception.last_state
    assert state is not None
    assert state.cognitive_load == 0.80
    assert state.stress_level > 0.0

    regulator = EmotionalRegulator()
    regulator.interoception.update_from_body_state(
        BodyState(
            cpu_temp_c=50.0,
            temperature=0.5,
            arousal_freq=0.5,
            cognitive_load=0.50,
            io_activity=0.0,
            stress_load=0.0,
            energy_reserve=1.0,
            on_ac_power=True,
            num_cores=1,
            distressed=False,
            autonomic_rate=0.0,
            thermoregulatory_effort=0.0,
            metabolic_rate=0.0,
            core_voltage=0.0,
            supply_voltage=0.0,
            description="",
        )
    )
    state = regulator.interoception.last_state
    assert state is not None
    assert state.stress_level == 0.0


# ─── Interoception layer composition ──────────────────────────
#
# Interoception has two independent sensing layers on different
# cadences: the process layer (psutil, socket liveness, observed
# latency) is refreshed by the regulator every 1.5-8 s, and the body
# layer (the daemon's hardware readings) by the heartbeat every ~5 s.
# They used to write the same InternalState with each rebuilding it
# from scratch, so whichever ran last erased the other.


def _fake_body(**overrides) -> Any:
    """A BodyState stand-in with a full set of hardware readings."""

    class _Body:
        cpu_temp_c = 78.0
        arousal_freq = 0.65
        cognitive_load = 0.45
        io_activity = 0.15
        stress_load = 1.3
        energy_reserve = 0.22
        distressed = True
        description = "overheating"
        autonomic_rate = 6.0
        thermoregulatory_effort = 0.05
        metabolic_rate = 0.85
        core_voltage = 1.25
        supply_voltage = 10.3
        core_activity = 0.4
        uncore_activity = 0.3
        dram_activity = 0.2
        cache_miss_rate = 0.11
        branch_miss_rate = 0.07
        pulse_hz = 2400.0
        pulse = 0.5
        top_freq_share = 0.9
        throttle_state = 0.5
        psi_cpu = 0.3
        psi_io = 0.1
        psi_mem = 0.2
        battery_cycles = 42.0
        entropy_level = 0.99
        clocksource = 1
        suspend_caps = 3

    for key, value in overrides.items():
        setattr(_Body, key, value)
    return _Body()


def _intero(cpu_usage: float, tmp_path) -> InteroceptionSystem:
    """An InteroceptionSystem with fixed CPU and a live socket path.

    The socket file is created so `daemon_connected` is True —
    otherwise the process layer contributes `daemon_disconnect_stress`
    and these tests would be measuring disconnection rather than the
    layer under test.
    """
    from genesis_conscious.limbic_system.emotional_regulator import InteroceptionSystem

    sock = tmp_path / "genesis.sock"
    sock.write_text("")
    io = InteroceptionSystem(socket_path=str(sock))
    io._sense_cpu_usage = lambda: cpu_usage  # type: ignore[method-assign]
    return io


def test_process_sensing_does_not_clobber_body_layer(tmp_path) -> None:
    """The regulator's faster polling must not erase the daemon's hardware.

    The regression: `sense_internal_state` rebuilt the whole state with
    `cognitive_load=0.0` and default hardware fields. Running every
    1.5-8 s against the heartbeat's ~5 s body-state poll, it discarded
    the daemon's interoception within seconds — and the `body_stress`
    input to allostatic load was then the process reading, not the
    hardware one it is documented to be.
    """
    io = _intero(88.0, tmp_path)
    io.update_from_body_state(_fake_body())
    for _ in range(20):
        io.sense_internal_state()

    state = io.last_state
    assert state is not None
    assert state.cpu_temp_c == 78.0, "body layer was clobbered"
    assert state.stress_load == 1.3, "body layer was clobbered"
    assert state.energy_reserve == 0.22, "body layer was clobbered"
    assert state.battery_cycles == 42.0, "body layer was clobbered"
    assert state.cache_miss_rate == 0.11, "body layer was clobbered"
    assert state.body_distressed is True
    # The process layer survives too — the other direction.
    assert state.cpu_usage == 88.0, "process layer was clobbered"


def test_body_update_does_not_clobber_process_layer(tmp_path) -> None:
    """A calm daemon report must not erase a simultaneous CPU spike.

    The regression: `update_from_body_state` set `stress_level` from
    the body layer alone, discarding the process layer's CPU, memory,
    socket and latency readings.
    """
    cpu_stress = (88.0 - 70.0) / 30.0
    io = _intero(88.0, tmp_path)
    io.sense_internal_state()
    io.update_from_body_state(
        _fake_body(cognitive_load=0.1, stress_load=0.0, cpu_temp_c=40.0)
    )

    state = io.last_state
    assert state is not None
    assert state.cpu_usage == 88.0
    assert abs(state.stress_level - cpu_stress) < 1e-6, (
        f"process stress erased by a calm body report: {state.stress_level}"
    )


def test_stress_level_is_max_of_both_layers(tmp_path) -> None:
    """Either layer reporting distress must register as stress."""
    # Body layer alone: stress_load 1.3 -> (1.3-1.0)/0.5 = 0.6
    io = _intero(5.0, tmp_path)
    io.update_from_body_state(_fake_body())
    state = io.last_state
    assert state is not None
    assert abs(state.stress_level - 0.6) < 1e-6

    # Both layers at once: the larger wins, neither is discarded.
    # Process stress (88-70)/30 = 0.6; cognitive_load 0.95 gives
    # (0.95-0.70)/0.30 = 0.8333, which is the larger.
    io2 = _intero(88.0, tmp_path)
    io2.update_from_body_state(_fake_body(cognitive_load=0.95, stress_load=0.0))
    state2 = io2.last_state
    assert state2 is not None
    assert abs(state2.stress_level - 0.8333) < 1e-3, state2.stress_level


def test_stale_body_reading_stops_contributing_stress(tmp_path) -> None:
    """A sensor that stopped reporting must not hold distress forever.

    The values are retained — "the last temperature I read was 78°" is
    still the honest answer — but the stress contribution decays, so a
    daemon that died mid-overload doesn't read as permanent distress.
    """
    from genesis_conscious.limbic_system.emotional_regulator import InteroceptionSystem

    io = _intero(5.0, tmp_path)
    io.update_from_body_state(_fake_body())
    fresh = io.last_state
    assert fresh is not None
    assert fresh.stress_level > 0.5, "fresh body reading should register stress"

    assert io._body.reported_at is not None
    io._body.reported_at -= InteroceptionSystem.BODY_STATE_TIMEOUT + 1
    io.sense_internal_state()
    stale = io.last_state
    assert stale is not None
    assert stale.stress_level < 1e-6, (
        f"a stale reading should not contribute stress, got {stale.stress_level}"
    )
    assert stale.cpu_temp_c == 78.0, "stale reading should retain its last value"


def test_body_update_preserves_observed_latency(tmp_path) -> None:
    """Latency observations must survive a body-state merge.

    The regression: `update_from_body_state` copied `response_latency`
    but not `last_response_latency` or `latency_observed`, so a merge
    silently reset the latency gate and the latency stress
    contribution with it.
    """
    io = _intero(5.0, tmp_path)
    io.record_response_latency(750.0)
    io.sense_internal_state()
    before = io.last_state
    assert before is not None
    assert before.latency_observed is True

    io.update_from_body_state(_fake_body(cognitive_load=0.1, stress_load=0.0))
    after = io.last_state
    assert after is not None
    assert after.latency_observed is True, "latency_observed lost in body merge"
    assert after.last_response_latency > 0.0, "last_response_latency lost in body merge"


def test_arousal_modifier_follows_combined_stress(tmp_path) -> None:
    """A body-driven stress spike must move the arousal modifier.

    The regression: the modifier was only recomputed in the process
    layer, so hardware distress left it stale.
    """
    io = _intero(5.0, tmp_path)
    io.sense_internal_state()
    calm = io.last_state
    assert calm is not None
    assert calm.arousal_modifier == 0.5, calm.arousal_modifier

    # stress_load 1.3 -> (1.3-1.0)/0.5 = 0.6, which lands in the
    # "mild stress" band rather than the "overwhelmed" one, so the
    # modifier should read 0.6.
    io.update_from_body_state(_fake_body())
    hot = io.last_state
    assert hot is not None
    assert (
        io.config.arousal_modifier_mild_stress
        < hot.stress_level
        <= io.config.arousal_modifier_overwhelmed_stress
    ), hot.stress_level
    assert hot.arousal_modifier == 0.6, (
        f"arousal modifier should follow body stress, got {hot.arousal_modifier}"
    )


def test_process_layer_still_works_offline() -> None:
    """Without a daemon, the process layer must still sense distress."""
    from genesis_conscious.limbic_system.emotional_regulator import InteroceptionSystem

    io = InteroceptionSystem(socket_path="/nonexistent/genesis.sock")
    state = io.sense_internal_state()
    assert state.daemon_connected is False
    # A missing daemon is itself distress.
    assert state.stress_level >= io.config.daemon_disconnect_stress


def test_own_pids_filters_zero_and_dead(tmp_path) -> None:
    """The shared PID set must never count PID 0 or stale pidfiles.

    Regression: `_sense_memory_usage` collected pidfile PIDs verbatim,
    so a `0` pidfile or a dead PID counted a stranger's RSS as
    cognitive load until the probe happened to swallow it.
    CPU and memory must also agree on the same process tree.
    """
    import os

    sock = tmp_path / "genesis.sock"
    sock.write_text("")
    io = InteroceptionSystem(socket_path=str(sock))
    (tmp_path / "genesis_daemon.pid").write_text("0")
    (tmp_path / "genesis_retina.pid").write_text(str(os.getpid()))
    pids = io._own_pids()
    assert 0 not in pids
    assert pids.count(os.getpid()) == 1

    (tmp_path / "genesis_daemon.pid").write_text("99999999")
    io._daemon_pid = None
    io._retina_pid = None
    pids = io._own_pids()
    assert 99999999 not in pids

pytestmark = pytest.mark.usefixtures("learned_sentiment")

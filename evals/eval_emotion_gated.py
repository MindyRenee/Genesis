"""Evaluation 3: Emotion-Gated Learning.

Tests whether Genesis's autonomous learner correctly gates learning
based on emotional state. This is the first mechanism-level eval:
it tests the emotion → learning path that mirrors the Rust daemon's
plasticity gate.

The autonomous learner blocks new acquisition when:
- Emotion label is "stressed", "overwhelmed", "anxious", "drowsy",
  "melancholic", "sleeping", "unsettled", or "meditating".
- Plasticity posture is PROTECTIVE (chronic stress, low BDNF,
  high cortisol).

Conditions:
1. Blocks under stress — stressed emotion → _should_learn() returns False.
2. Blocks under overwhelm — overwhelmed emotion → False.
3. Blocks under anxiety — anxious emotion → False.
4. Blocks under drowsiness — drowsy emotion → False.
5. Allows under positive — positive emotion → True.
6. Allows under flow — flow emotion → True.
7. Allows under neutral — neutral emotion → True.
8. Allows without callback — no emotion callback → True (testing mode).
9. Blocks under protective posture — PROTECTIVE posture → False,
   even with a positive emotion.
10. Allows under receptive posture — RECEPTIVE posture → True.

This eval tests the AutonomousLearner directly (not through the Mind)
because the emotion/plasticity callbacks are what connect neurochemistry
to learning. In a running system, these callbacks read from the daemon;
here we inject controlled values.
"""

from __future__ import annotations

import time

from genesis_client.types import (
    LEARNING_POSTURE_NEUTRAL,
    LEARNING_POSTURE_PROTECTIVE,
    LEARNING_POSTURE_RECEPTIVE,
    PlasticityProfile,
)
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
from genesis_cognitive.reasoning import ReasoningEngine
from harness import EvalResult, FactResult, print_result, run_condition

# ─── Helpers ──────────────────────────────────────────────────────


def _make_learner(
    get_emotion=None,
    get_plasticity_profile=None,
) -> AutonomousLearner:
    """Create an AutonomousLearner with controlled callbacks."""
    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    return AutonomousLearner(
        network=net,
        curiosity=curiosity,
        get_emotion=get_emotion,
        get_plasticity_profile=get_plasticity_profile,
    )


def _make_emotion(label: str, valence: float = 0.0) -> EmotionalState:
    """Construct a emotion for tests."""
    return EmotionalState(
        label=label,
        cognitive_style="steady",
        valence=valence,
    )


def _make_profile(
    plasticity_gate: float = 0.5,
    bdnf_tonic: float = 0.4,
    cortisol_tonic: float = 0.3,
    posture: str = LEARNING_POSTURE_NEUTRAL,
) -> PlasticityProfile:
    """Construct a profile for tests."""
    return PlasticityProfile(
        plasticity_gate=plasticity_gate,
        bdnf_effective=bdnf_tonic,
        bdnf_tonic=bdnf_tonic,
        cortisol_effective=cortisol_tonic,
        cortisol_tonic=cortisol_tonic,
        dopamine_effective=0.4,
        serotonin_effective=0.4,
        coupling_drift=0.0,
        mean_receptor_sensitivity=0.95,
        emergent_phase=0,
    )


def _check_blocks(label: str, emotion: EmotionalState) -> list[FactResult]:
    """Verify learning is blocked for this emotion."""
    learner = _make_learner(get_emotion=lambda: emotion)
    blocked = not learner._should_learn()
    return [FactResult(
        concept=label,
        passed=blocked,
        detail=f"{'blocked' if blocked else 'NOT blocked'} by {emotion.label}",
    )]


def _check_allows(label: str, emotion: EmotionalState) -> list[FactResult]:
    """Verify learning is allowed for this emotion."""
    learner = _make_learner(get_emotion=lambda: emotion)
    allowed = learner._should_learn()
    return [FactResult(
        concept=label,
        passed=allowed,
        detail=f"{'allowed' if allowed else 'NOT allowed'} by {emotion.label}",
    )]


def _check_blocks_posture(label: str, posture: str) -> list[FactResult]:
    """Verify learning is blocked by protective posture."""
    profile = _make_profile(
        plasticity_gate=0.1,
        bdnf_tonic=0.15,
        cortisol_tonic=0.75,
        posture=posture,
    )
    # Even with a positive emotion, protective posture should block
    emotion = _make_emotion("positive", valence=0.3)
    learner = _make_learner(
        get_emotion=lambda: emotion,
        get_plasticity_profile=lambda: profile,
    )
    # Drive posture through the production callback path
    # (_update_posture reads the profile from get_plasticity_profile
    # and derives learning_posture from gate/cortisol thresholds),
    # not by injecting _current_posture directly.
    learner._update_posture()
    blocked = not learner._should_learn()
    return [FactResult(
        concept=label,
        passed=blocked,
        detail=f"{'blocked' if blocked else 'NOT blocked'} by {posture} posture",
    )]


def _check_allows_posture(label: str, posture: str) -> list[FactResult]:
    """Verify learning is allowed under receptive posture."""
    profile = _make_profile(
        plasticity_gate=0.7,
        bdnf_tonic=0.5,
        cortisol_tonic=0.2,
        posture=posture,
    )
    emotion = _make_emotion("positive", valence=0.3)
    learner = _make_learner(
        get_emotion=lambda: emotion,
        get_plasticity_profile=lambda: profile,
    )
    learner._update_posture()
    allowed = learner._should_learn()
    return [FactResult(
        concept=label,
        passed=allowed,
        detail=f"{'allowed' if allowed else 'NOT allowed'} by {posture} posture",
    )]


# ─── Conditions ───────────────────────────────────────────────────


def _condition_blocks_negative() -> list[FactResult]:
    """Helper: blocks negative."""
    results: list[FactResult] = []
    for label in ["stressed", "overwhelmed", "anxious", "drowsy"]:
        results.extend(_check_blocks(label, _make_emotion(label, valence=-0.4)))
    return results


def _condition_allows_positive() -> list[FactResult]:
    """Helper: allows positive."""
    results: list[FactResult] = []
    for label, valence in [("positive", 0.3), ("in flow", 0.5), ("neutral", 0.0)]:
        results.extend(_check_allows(label, _make_emotion(label, valence=valence)))
    return results


def _condition_no_callback() -> list[FactResult]:
    """Without an emotion callback, learning should always be allowed."""
    learner = _make_learner(get_emotion=None)
    allowed = learner._should_learn()
    return [FactResult(
        concept="no_callback",
        passed=allowed,
        detail=f"{'allowed' if allowed else 'NOT allowed'} without emotion callback",
    )]


def _condition_protective_posture() -> list[FactResult]:
    """Helper: protective posture."""
    return _check_blocks_posture("protective", LEARNING_POSTURE_PROTECTIVE)


def _condition_receptive_posture() -> list[FactResult]:
    """Helper: receptive posture."""
    return _check_allows_posture("receptive", LEARNING_POSTURE_RECEPTIVE)


# ─── Eval entry point ─────────────────────────────────────────────


def run(seed: int = 42) -> EvalResult:
    """Run the evaluation."""
    start = time.time()
    result = EvalResult(
        name="Emotion-Gated Learning",
        description=(
            "Does Genesis's autonomous learner correctly gate "
            "learning based on emotional state and plasticity "
            "posture? Tests the emotion → learning path that "
            "mirrors the Rust daemon's BDNF-gated plasticity."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    conditions = [
        ("Blocks Negative Emotions",
         "Stressed, overwhelmed, anxious, drowsy → learning blocked.",
         _condition_blocks_negative),
        ("Allows Positive Emotions",
         "Positive, flow, neutral → learning allowed.",
         _condition_allows_positive),
        ("No Callback (Testing Mode)",
         "No emotion callback → learning allowed (for testing).",
         _condition_no_callback),
        ("Protective Posture Blocks",
         "PROTECTIVE posture (high cortisol, low BDNF) → blocked even with positive emotion.",
         _condition_protective_posture),
        ("Receptive Posture Allows",
         "RECEPTIVE posture (low cortisol, high BDNF) → allowed.",
         _condition_receptive_posture),
    ]

    all_accuracies: list[float] = []
    for name, desc, fn in conditions:
        cond = run_condition(name, desc, fn)
        result.conditions.append(cond)
        all_accuracies.append(cond.accuracy)

    result.overall_accuracy = (
        sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    )
    result.duration_seconds = round(time.time() - start, 3)
    return result


if __name__ == "__main__":
    res = run()
    print_result(res)

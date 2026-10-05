"""Regression tests for goal-directed decision candidate generation."""

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.reasoning import ActionType, DecisionEngine


def test_active_goals_make_investigation_selectable() -> None:
    """An active goal must expose investigation as a real decision option."""
    engine = DecisionEngine(ConceptNetwork())
    outcome = engine.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.4,
        uncertainty=0.5,
        goals=["understand:photosynthesis"],
        topics=["photosynthesis"],
        perception_intent="question",
    )
    assert any(
        candidate.action_type == ActionType.INVESTIGATE
        for candidate in outcome.candidates
    )


def test_no_active_goals_do_not_add_investigation() -> None:
    """Investigation should not be injected when there is no active goal."""
    engine = DecisionEngine(ConceptNetwork())
    outcome = engine.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.4,
        uncertainty=0.5,
        goals=[],
        topics=[],
        perception_intent="question",
    )
    assert all(
        candidate.action_type != ActionType.INVESTIGATE
        for candidate in outcome.candidates
    )

"""Regression tests for goal-directed decision candidate generation."""

from frontal_lobe.reasoning import ActionType, DecisionMaker, ResourceState
from temporal_lobe.concepts import ConceptNetwork


def test_active_goals_make_investigation_selectable() -> None:
    """An active goal must expose investigation as a real decision option."""
    engine = DecisionMaker(ConceptNetwork())
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
    engine = DecisionMaker(ConceptNetwork())
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


def test_decision_outcome_carries_bg_conflict() -> None:
    """The basal-ganglia conflict scalar reaches the DecisionOutcome.

    The ACC monitors this signal for response conflict (Botvinick
    et al., 2001) — it must be exposed at the outcome level, not
    buried per-candidate.
    """
    engine = DecisionMaker(ConceptNetwork())
    outcome = engine.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.4,
        uncertainty=0.5,
        goals=[],
        topics=[],
        perception_intent="question",
    )
    assert 0.0 <= outcome.conflict <= 1.0
    selectable = [
        c for c in outcome.candidates
        if c.action_type != ActionType.WITHHOLD and not c.inhibited
    ]
    assert selectable
    assert all(c.bg_conflict == outcome.conflict for c in selectable)

def test_hardware_distress_blocks_exploration_but_preserves_conversation() -> None:
    """Daemon-owned distress must constrain optional exploration."""
    engine = DecisionMaker(
        ConceptNetwork(),
        resource_state=ResourceState(
            available=True,
            hardware_distressed=True,
            psi_cpu=0.8,
        ),
    )
    outcome = engine.decide(
        default_action=ActionType.INVESTIGATE,
        reasoning_results=[],
        confidence=0.4,
        uncertainty=0.5,
        goals=["understand:photosynthesis"],
        topics=["photosynthesis"],
        perception_intent="question",
    )
    investigate = next(
        c for c in outcome.candidates
        if c.action_type == ActionType.INVESTIGATE
    )
    assert investigate.inhibited
    assert investigate.composite_score == 0.0
    assert outcome.action_type != ActionType.INVESTIGATE
    assert any("resource-constrained" in note for note in outcome.notes)


def test_resource_constraint_is_inert_without_hardware_verdict() -> None:
    """Low reserve alone must not become an invented shutdown threshold."""
    engine = DecisionMaker(
        ConceptNetwork(),
        resource_state=ResourceState(
            available=True,
            hardware_distressed=False,
            on_ac_power=False,
            energy_reserve=0.01,
        ),
    )
    outcome = engine.decide(
        default_action=ActionType.INVESTIGATE,
        reasoning_results=[],
        confidence=0.4,
        uncertainty=0.5,
        goals=["understand:photosynthesis"],
        topics=["photosynthesis"],
        perception_intent="question",
    )
    investigate = next(
        c for c in outcome.candidates
        if c.action_type == ActionType.INVESTIGATE
    )
    assert not investigate.inhibited

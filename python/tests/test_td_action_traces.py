"""Regression tests for action-conditioned TD credit assignment."""

from genesis_conscious.concepts import ConceptNetwork
from genesis_conscious.learning.td import TDLearner


def _network() -> ConceptNetwork:
    network = ConceptNetwork()
    network.add_concept("state", confidence=1.0)
    network.add_concept("next", confidence=1.0)
    return network


def test_action_conditioned_traces_preserve_temporal_credit() -> None:
    """A prior action trace receives later TD error with decayed credit."""
    learner = TDLearner(_network(), lam=0.8, discount=0.9)

    learner.update(["state"], 0.0, ["next"], action="A", next_action="A")
    assert learner.get_trace("state", action="A") == 1.0
    assert learner.get_trace("state", action="B") == 0.0

    learner.update(["next"], 2.0, [], action="B")

    # The A trace remains eligible but decays by gamma*lambda.
    assert learner.get_trace("state", action="A") == 0.72
    # B was not active in the first transition; its trace is independent.
    assert learner.get_trace("state", action="B") == 0.0
    # The later TD error therefore changes A through its retained trace.
    assert learner._action_weights[("state", "A")] != 1.0
    assert learner._action_weights.get(("state", "B"), 1.0) == 1.0


def test_action_conditioned_traces_decay_independently() -> None:
    """Different action traces remain separate while decaying."""
    learner = TDLearner(_network(), lam=0.8, discount=0.9)

    learner.update(["state"], 0.0, [], action="A")
    learner.update(["state"], 0.0, [], action="B")

    assert learner.get_trace("state", action="A") == 0.72
    assert learner.get_trace("state", action="B") == 1.0

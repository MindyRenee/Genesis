"""Regression tests for action-conditioned TD credit assignment."""

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.learning.td import TDLearner


def _network() -> ConceptNetwork:
    network = ConceptNetwork()
    network.add_concept("state", confidence=1.0)
    network.add_concept("next", confidence=1.0)
    return network


def test_action_conditioned_trace_is_scoped_to_action() -> None:
    """A trace for action A must not receive action B's later TD error."""
    learner = TDLearner(_network(), lam=0.8, discount=0.9)

    learner.update(["state"], 0.0, ["next"], action="A", next_action="A")
    assert learner.get_trace("state", action="A") > 0.0
    assert learner.get_trace("state", action="B") == 0.0

    learner.update(["next"], 1.0, [], action="B")

    assert learner.get_weight("state") == 1.0
    assert learner._action_weights.get(("state", "A"), 1.0) == 1.0
    assert learner._action_weights.get(("state", "B"), 1.0) != 1.0


def test_action_conditioned_traces_decay_independently() -> None:
    """Different action traces remain separate while decaying."""
    learner = TDLearner(_network(), lam=0.8, discount=0.9)

    learner.update(["state"], 0.0, [], action="A")
    learner.update(["state"], 0.0, [], action="B")

    assert learner.get_trace("state", action="A") == 0.72
    assert learner.get_trace("state", action="B") == 1.0

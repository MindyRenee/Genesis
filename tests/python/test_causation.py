"""Tests for the counterfactual theory of causation — dependence
evaluated by surgical intervention on structural equations."""

import pytest
from frontal_lobe.reasoning.causation import StructuralModel
from frontal_lobe.reasoning.logic import atom, not_, or_


def test_direct_dependence_is_causation():
    """The canonical case: had c not occurred, e would not have."""
    # flip → power → light
    model = StructuralModel(
        {
            "power": atom("flip"),
            "light": atom("power"),
        },
        exogenous={"flip": True},
    )
    actual = model.evaluate()
    assert actual["light"]
    assert model.depends("flip", "light", actual)
    assert model.causes("flip", "light", actual=actual)


def test_no_dependence_means_no_causation():
    """An event that would have happened anyway is not caused by c —
    the counterfactual fails."""
    # two independent switches on one light
    model = StructuralModel(
        {"light": or_(atom("a"), atom("b"))},
        exogenous={"a": True, "b": True},
    )
    actual = model.evaluate()
    # Neither switch is needed — each would leave the other to do it.
    assert not model.depends("a", "light", actual)
    assert not model.causes("a", "light", actual=actual)


def test_chain_of_dependence_is_causation():
    """Causation is ancestral: c causes e through a chain of
    dependencies even though c itself is no longer needed at the end."""
    # spark → fire → smoke → alarm
    model = StructuralModel(
        {
            "fire": atom("spark"),
            "smoke": atom("fire"),
            "alarm": atom("smoke"),
        },
        exogenous={"spark": True},
    )
    actual = model.evaluate()
    assert model.causes("spark", "alarm", actual=actual)
    # Alarm depends on spark directly — the shortest witnessing
    # chain is the direct link; the mechanism lives in the steps.
    assert model.causal_chain("spark", "alarm", actual=actual) == (
        "spark",
        "alarm",
    )
    assert model.depends("fire", "alarm", actual)
    assert model.depends("smoke", "alarm", actual)


def test_absent_events_cause_nothing():
    """Lewis's constraint: causation relates events — an event that
    did not occur causes nothing."""
    model = StructuralModel({"light": atom("flip")}, exogenous={"flip": False})
    actual = model.evaluate()
    assert not actual["flip"]
    assert not model.causes("flip", "light", actual=actual)


def test_preemption_through_the_actual_process():
    """Early preemption: the backup would have done it, so the throw
    is not *needed* — but the actual hit still carries the chain."""
    # suzy throws & hits; billy would have thrown only if she'd missed.
    model = StructuralModel(
        {
            "suzy_hits": atom("suzy_throws"),
            "billy_throws": not_(atom("suzy_throws")),
            "billy_hits": atom("billy_throws"),
            "breaks": or_(atom("suzy_hits"), atom("billy_hits")),
        },
        exogenous={"suzy_throws": True},
    )
    actual = model.evaluate()
    assert actual["breaks"]
    # Direct dependence fails — Billy's backup covers the gap.
    assert not model.depends("suzy_throws", "breaks", actual)
    # But the actual process is a chain: throw → hit → break.
    assert model.causes("suzy_throws", "breaks", actual=actual)
    assert model.causal_chain("suzy_throws", "breaks", actual=actual) == (
        "suzy_throws",
        "suzy_hits",
        "breaks",
    )


def test_surgery_cuts_only_the_cause():
    """do(¬c) severs c's incoming arrows; every other mechanism is
    preserved — Lewis's small miracle."""
    model = StructuralModel(
        {
            "fire": atom("spark"),
            "heat": atom("fire"),
        },
        exogenous={"spark": True},
    )
    cut, fixed = model.intervene("fire", False)
    out = cut.evaluate(fixed)
    # Spark still happened — intervention does not rewrite the past.
    assert out["spark"]
    assert not out["fire"]
    assert not out["heat"]


def test_cyclic_models_are_reported():
    """A model whose equations never converge is cyclic — reported,
    not silently truncated."""
    model = StructuralModel({"a": not_(atom("b")), "b": atom("a")}, exogenous={})
    with pytest.raises(ValueError):
        model.evaluate()

"""Tests for the shared self/inner/outer spacetime causal model."""

import pytest
from temporal_lobe.light_cones import (
    IntervalKind,
    LightConeModel,
    SpacetimeEvent,
    WorldDomain,
)


def event(name, t, x, domain=WorldDomain.OUTER):
    return SpacetimeEvent(name, t, (x, 0.0, 0.0), domain)


def test_apex_is_shared_and_summary_counts_domains():
    model = LightConeModel(signal_speed=2.0)
    apex = event("self-now", 10.0, 0.0, WorldDomain.SELF)
    model.advance_apex(apex)
    model.add_event(event("inner-memory", 8.0, 1.0, WorldDomain.INNER))
    model.add_event(event("outer-observation", 11.0, 1.0))
    summary = model.state_summary()
    assert summary["apex_id"] == "self-now"
    assert summary["event_count"] == 3
    assert summary["domains"] == {"self": 1, "inner": 1, "outer": 1}


def test_past_and_future_cones_use_speed_and_time_order():
    model = LightConeModel(signal_speed=2.0)
    model.advance_apex(event("now", 10.0, 0.0, WorldDomain.SELF))
    past = model.classify(event("past", 8.0, 3.0, WorldDomain.INNER))
    future = model.classify(event("future", 11.0, 1.5))
    assert past.in_past_cone and not past.in_future_cone
    assert future.in_future_cone and not future.in_past_cone
    assert past.kind is IntervalKind.TIMELIKE
    assert future.kind is IntervalKind.TIMELIKE


def test_lightlike_and_spacelike_events_are_distinguished():
    model = LightConeModel(signal_speed=1.0)
    model.advance_apex(event("now", 0.0, 0.0, WorldDomain.SELF))
    light = model.classify(event("light", 2.0, 2.0))
    outside = model.classify(event("outside", 1.0, 2.0))
    assert light.kind is IntervalKind.LIGHTLIKE
    assert light.in_future_cone
    assert outside.kind is IntervalKind.SPACELIKE
    assert not outside.in_future_cone


def test_coincident_event_is_not_past_or_future():
    model = LightConeModel()
    model.advance_apex(event("now", 4.0, 2.0, WorldDomain.SELF))
    result = model.classify(event("same", 4.0, 2.0, WorldDomain.INNER))
    assert result.kind is IntervalKind.COINCIDENT
    assert not result.in_past_cone
    assert not result.in_future_cone


def test_apex_cannot_move_backwards_in_time():
    model = LightConeModel()
    model.advance_apex(event("later", 5.0, 0.0, WorldDomain.SELF))
    with pytest.raises(ValueError, match="cannot move backwards"):
        model.advance_apex(event("earlier", 4.0, 0.0, WorldDomain.SELF))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_speed": 0.0},
        {"signal_speed": -1.0},
        {"signal_speed": float("inf")},
    ],
)
def test_signal_speed_must_be_finite_and_positive(kwargs):
    with pytest.raises(ValueError, match="signal_speed"):
        LightConeModel(**kwargs)


def test_event_validates_position_and_confidence():
    with pytest.raises(ValueError, match="three finite coordinates"):
        SpacetimeEvent("bad", 0.0, (1.0, 2.0), WorldDomain.SELF)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SpacetimeEvent("bad-confidence", 0.0, (0.0, 0.0, 0.0), WorldDomain.SELF, 1.1)

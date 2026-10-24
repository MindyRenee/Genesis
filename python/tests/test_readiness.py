"""Tests for the readiness potential — the movement-locked buildup
record each voluntary crossing leaves on the urge accumulator."""

import random

from frontal_lobe.volition import Readiness, Urge, Volition


def _tick(vol: Volition, seconds: float = 1.0, ctx: dict | None = None) -> list[str]:
    """Advance the engine as if ``seconds`` had elapsed since the last
    tick — the engine measures its own dt off the monotonic clock, so
    tests wind its reference back instead of sleeping."""
    vol._last_tick -= seconds
    return vol.tick(ctx or {})


def _urge(**kw) -> Urge:
    kw.setdefault("name", "u")
    kw.setdefault("growth", 0.05)
    kw.setdefault("decay", 0.0)
    kw.setdefault("cooldown", 0.0)
    return Urge(**kw)


def test_crossing_captures_the_buildup() -> None:
    """A deterministic rise to threshold leaves a Readiness record:
    onset at the rest trough, crossing at the threshold tick."""
    vol = Volition()
    vol.register(_urge(threshold=0.7))
    ready: list[str] = []
    for _ in range(20):
        ready = _tick(vol)
        if ready:
            break
    assert ready == ["u"]
    r = vol.readiness["u"]
    assert r.crossed > r.onset
    assert r.buildup_s > 10.0  # ~0.05/s to 0.7 — a sustained rise
    assert r.threshold == 0.7
    assert r.acted == 0.0
    # The trajectory climbs from near rest to threshold.
    assert len(r.trajectory) >= 2
    assert r.trajectory[0][1] < 0.2
    assert r.trajectory[-1][1] >= 0.7


def test_ready_urge_does_not_recapture() -> None:
    """A ready urge that stays ready (held at the gate) keeps the
    record of its crossing — the RP is captured on the rising edge."""
    vol = Volition()
    vol.register(_urge(threshold=0.7))
    for _ in range(20):
        if _tick(vol):
            break
    first = vol.readiness["u"]
    _tick(vol)  # still ready, still held — must not recapture
    assert vol.readiness["u"] is first


def test_consume_stamps_the_act() -> None:
    """consume() is the enactment — it stamps ``acted`` so the
    crossing→action lag is measured, not assumed."""
    vol = Volition()
    vol.register(_urge(threshold=0.7))
    for _ in range(20):
        if _tick(vol):
            break
    vol.consume("u")
    r = vol.readiness["u"]
    assert r.acted > 0.0
    assert r.enact_lag_s >= 0.0


def test_reaccumulation_recaptures() -> None:
    """After acting, the next buildup is a new record — onset is
    measured from the post-action trough, not the old crossing."""
    vol = Volition()
    vol.register(_urge(threshold=0.7))
    for _ in range(20):
        if _tick(vol):
            break
    first = vol.readiness["u"]
    vol.consume("u")
    for _ in range(20):
        if _tick(vol):
            break
    second = vol.readiness["u"]
    assert second is not first
    assert second.onset >= first.crossed


def test_noise_tips_a_poised_urge() -> None:
    """The stochastic term is the crossing's trigger: an urge hovering
    just under threshold with zero net drive fires on fluctuation —
    the Schurger mechanism — where the silent accumulator never would."""
    vol = Volition(rng=random.Random(0))
    urge = _urge(threshold=0.7, growth=0.0, noise=0.05, value=0.66)
    vol.register(urge)
    for _ in range(500):
        if _tick(vol):
            return  # a fluctuation carried it over
    raise AssertionError("noise never tipped the poised urge")


def test_no_noise_no_spontaneous_crossing() -> None:
    """Without the fluctuation term nothing fires below a drive the
    model never raised — determinism stays the default."""
    vol = Volition(rng=random.Random(0))
    vol.register(_urge(threshold=0.7, growth=0.0, value=0.65))
    for _ in range(50):
        assert _tick(vol) == []


def test_readiness_report_is_introspectable() -> None:
    vol = Volition()
    vol.register(_urge(threshold=0.7))
    for _ in range(20):
        if _tick(vol):
            break
    report = vol.readiness_report()
    assert set(report) == {"u"}
    assert report["u"]["buildup_s"] > 0.0
    assert report["u"]["samples"] >= 2.0


def test_frozen_record() -> None:
    """The record is a snapshot — it must not mutate with the urge."""
    import dataclasses

    import pytest

    r = Readiness("u", onset=1.0, crossed=3.0, threshold=0.7, trajectory=())
    assert r.buildup_s == 2.0
    assert r.enact_lag_s == 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.onset = 9.0  # type: ignore[misc]

"""World deliberation + impasse tests — confidence gates effort,
impasse escalates before boredom.

The couplings under test:

- ``_world_margin`` reports the arbitrated field's top-two
  separation — the same quantity the decision journal calls
  confidence. A flat field is a guess; a guess is what
  deliberation is for.
- ``_deliberate_world`` sweeps each offered action one step
  through learned physics and lets the imagined outcome bid —
  forward replay at the choice point (VTE). It must stay silent
  when the model has nothing to imagine with, so early play and
  coordinate interfaces never enter the loop.
- ``_world_patience`` stretches the impasse horizon while untried
  actions still promise physics — patience tracks epistemic value,
  not a fixed counter.
- ``_world_confuse`` turns sustained impasse into escalation: a
  broadcast the workspace can recruit, and the executive's held
  objective put up for revision — the adjuster firing, not just
  the monitor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from parietal_lobe.embodiment import (
    _DELIBERATE_MARGIN,
    _DRY_BASE,
    _DRY_LEARN_BONUS,
    EmbodimentMixin,
)


def _embodiment(**attrs):
    """A bare mixin — just the attributes the coupling touches."""
    m = EmbodimentMixin.__new__(EmbodimentMixin)
    m._world_motor = attrs.get("motor")
    m.world_model = attrs.get("world_model", SimpleNamespace())
    m._world_policy = attrs.get("policy", SimpleNamespace())
    m._world_ids = attrs.get("ids", [])
    m._world_dry = attrs.get("dry", 0)
    m._world_bridge = attrs.get(
        "bridge", SimpleNamespace(broadcast_observation=lambda *a, **k: None)
    )
    m._held_objective = attrs.get("held")
    m.brain = attrs.get("brain", SimpleNamespace(frontal=SimpleNamespace(executive=None)))
    return m


def _sim(cells):
    return SimpleNamespace(cells=frozenset(cells))


# ─── Margin — decisiveness of the last arbitration ────────────


def test_margin_is_full_without_a_field():
    assert _embodiment(motor=None)._world_margin() == 1.0


def test_margin_reads_top_two_separation():
    motor = SimpleNamespace(action_scores={"1": 0.8, "2": 0.2, "3": 0.1})
    assert _embodiment(motor=motor)._world_margin() == pytest.approx(0.3)


def test_margin_flat_when_field_is_flat():
    motor = SimpleNamespace(action_scores={"1": 0.25, "2": 0.25})
    assert _embodiment(motor=motor)._world_margin() == 0.0


# ─── Deliberation — imagined outcomes bid ──────────────────────


def test_deliberation_silent_without_a_start_state():
    m = _embodiment(world_model=SimpleNamespace(start_state=lambda: None))
    assert m._deliberate_world(["1", "2"]) == {}


def test_deliberation_silent_on_unknown_physics():
    """Early play: no learned effects — nothing to imagine with."""
    m = _embodiment(
        world_model=SimpleNamespace(
            start_state=lambda: _sim({(5, 5)}),
            simulate=lambda _s, a: None,
        ),
        policy=SimpleNamespace(valued_cells=lambda: {(2, 2)}),
    )
    assert m._deliberate_world(["1", "2"]) == {}


def test_deliberation_pulls_toward_progress():
    """The action whose imagined step approaches a valued end
    out-bids the one that walks away."""
    start = _sim({(5, 5)})
    sims = {
        "toward": _sim({(4, 4)}),
        "away": _sim({(7, 7)}),
    }
    m = _embodiment(
        world_model=SimpleNamespace(
            start_state=lambda: start,
            simulate=lambda _s, a: sims.get(a),
        ),
        policy=SimpleNamespace(valued_cells=lambda: {(2, 2)}),
    )
    out = m._deliberate_world(["toward", "away"])
    # Progress earns the cell gain on top of the move premium;
    # walking away has a negative pull and earns no voice at all.
    assert out["toward"] > 0.05
    assert "away" not in out


def test_deliberation_has_no_voice_for_stalls_and_refusals():
    """A move the forward model calls dead gets no pull."""
    start = _sim({(5, 5)})
    m = _embodiment(
        world_model=SimpleNamespace(
            start_state=lambda: start,
            simulate=lambda _s, a: _sim({(5, 5)}),
        ),
        policy=SimpleNamespace(valued_cells=lambda: {(2, 2)}),
    )
    assert m._deliberate_world(["1"]) == {}


def test_margin_gate_matches_journal_confidence():
    """The deliberation trigger is the same quantity the decision
    journal reports as confidence."""
    flat = SimpleNamespace(action_scores={"1": 0.3, "2": 0.28})
    steep = SimpleNamespace(action_scores={"1": 0.9, "2": 0.1})
    assert _embodiment(motor=flat)._world_margin() < _DELIBERATE_MARGIN
    assert _embodiment(motor=steep)._world_margin() >= _DELIBERATE_MARGIN


# ─── Patience — impasse horizon tracks epistemic value ────────


def test_patience_is_base_without_offered_actions():
    assert _embodiment(ids=[])._world_patience() == _DRY_BASE


def test_patience_stretches_while_physics_is_unlearned():
    m = _embodiment(
        ids=["1", "2", "3", "4"],
        world_model=SimpleNamespace(effect_of=lambda a: None),
    )
    assert m._world_patience() == _DRY_BASE + _DRY_LEARN_BONUS


def test_patience_settles_once_everything_is_measured():
    m = _embodiment(
        ids=["1", "2"],
        world_model=SimpleNamespace(effect_of=lambda a: (0, 1)),
    )
    assert m._world_patience() == _DRY_BASE


def test_patience_scales_with_unlearned_fraction():
    half = _embodiment(
        ids=["1", "2"],
        world_model=SimpleNamespace(effect_of=lambda a: (0, 1) if a == "1" else None),
    )
    assert _DRY_BASE < half._world_patience() < _DRY_BASE + _DRY_LEARN_BONUS


# ─── Confusion — impasse escalates to revision ────────────────


def test_confuse_broadcasts_and_revises_the_held_objective():
    aired = []
    revised = []
    held = SimpleNamespace(objective="reach the far side")
    executive = SimpleNamespace(revise_intention=lambda i, reason="": revised.append((i, reason)))
    m = _embodiment(
        dry=8,
        held=held,
        brain=SimpleNamespace(frontal=SimpleNamespace(executive=executive)),
        bridge=SimpleNamespace(broadcast_observation=lambda *a, **k: aired.append((a, k))),
    )
    m._world_confuse("game-x")
    assert len(aired) == 1
    assert revised == [(held, "the world stopped answering — rethink")]
    assert m._held_objective is None


def test_confuse_is_quiet_without_a_held_objective():
    aired = []
    m = _embodiment(
        dry=8,
        held=None,
        bridge=SimpleNamespace(broadcast_observation=lambda *a, **k: aired.append((a, k))),
    )
    m._world_confuse("game-x")
    # The impasse still reaches the workspace — only the revision
    # has nothing to revise.
    assert len(aired) == 1


def test_confuse_survives_a_missing_executive():
    """Advisory coupling — no executive means no crash, just no
    revision."""
    m = _embodiment(dry=8, held=SimpleNamespace())
    m._world_confuse("game-x")

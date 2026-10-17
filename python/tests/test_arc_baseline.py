"""ARC baseline tests — the denominator beside the score.

The baselines run through the same ``ArcPractice`` loop as her, so
the tests here are less about the policies (uniform random, first
action — deliberately uninteresting) and more about the guarantee
the denominator rests on: identical accounting. Steps, resets,
per-level costs, scorecard lifecycle, and the ``baseline:<policy>``
stamp in ``decision_sources`` are what make "she beat the floor"
mean anything.
"""

from __future__ import annotations

import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest
from parietal_lobe.arc_baseline import ArcBaseline


@pytest.fixture(scope="module", autouse=True)
def require_arcengine():
    pytest.importorskip("arcengine", reason="optional ARC engine not installed")


class _State(Enum):
    NOT_FINISHED = 0
    WIN = 1
    GAME_OVER = 2


class _Frame:
    def __init__(self, grid, actions=(1, 2, 3), state=_State.NOT_FINISHED, levels=0, win=2):
        self.frame = [grid]
        self.available_actions = list(actions)
        self.state = state
        self.levels_completed = levels
        self.win_levels = win


class _FakeEnv:
    """A tiny 8×8 world that 'wins' after ``win_after`` steps."""

    def __init__(self, win_after=3):
        self.grid = [[(x + y) % 16 for x in range(8)] for y in range(8)]
        self.win_after = win_after
        self.steps = 0
        self.resets = 0
        self.seen_actions: list[str] = []

    def reset(self):
        self.resets += 1
        return _Frame(self.grid)

    def step(self, action, data=None, reasoning=None):
        self.steps += 1
        self.seen_actions.append(getattr(action, "name", str(action)))
        if self.steps >= self.win_after:
            return _Frame(self.grid, state=_State.WIN, levels=2, win=2)
        return _Frame(self.grid)


class _FakeArcade:
    def __init__(self, env):
        self._env = env
        self.closed: list[str] = []

    def create_scorecard(self, source_url=None, tags=None, opaque=None):
        return "card-1"

    def make(self, game_id, scorecard_id=None, save_recording=False):
        return self._env

    def close_scorecard(self, card):
        self.closed.append(card)


def _baseline(monkeypatch, env, policy="uniform") -> ArcBaseline:
    """An ArcBaseline whose SDK is a fake — no engine download needed."""
    class _Mode(Enum):
        OFFLINE = "offline"
        COMPETITION = "competition"

    monkeypatch.setitem(
        sys.modules,
        "arc_agi",
        types.SimpleNamespace(
            OperationMode=_Mode,
            Arcade=lambda **kw: _FakeArcade(env),
        ),
    )
    return ArcBaseline(policy=policy)


def test_unknown_policy_is_rejected(monkeypatch):
    _baseline(monkeypatch, _FakeEnv())
    with pytest.raises(ValueError, match="unknown baseline"):
        ArcBaseline(policy="smart")


def test_uniform_baseline_runs_same_accounting(monkeypatch):
    env = _FakeEnv(win_after=3)
    result = _baseline(monkeypatch, env).play("fake", max_steps=10)
    # Same accounting as her runs: the floor went through her loop.
    assert result.won and result.state == "WIN"
    assert result.steps == 3
    assert result.resets == 1
    assert result.level_steps == [3, 0]
    # Every move attributed to the baseline, never to her.
    assert result.decision_sources == {"baseline:uniform": 3}
    assert result.prior_learning == "none"


def test_first_baseline_picks_first_offered(monkeypatch):
    env = _FakeEnv(win_after=2)
    result = _baseline(monkeypatch, env, policy="first").play("fake", max_steps=5)
    assert env.seen_actions == ["ACTION1"] * env.steps
    assert result.won
    assert result.decision_sources == {"baseline:first": 2}


def test_baseline_budget_stops_like_hers(monkeypatch):
    env = _FakeEnv(win_after=999)
    result = _baseline(monkeypatch, env).play("fake", max_steps=5)
    assert result.state == "BUDGET" and not result.won
    assert result.steps == env.steps == 5


def test_baseline_resets_count_like_hers(monkeypatch):
    class _DyingEnv(_FakeEnv):
        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = _DyingEnv()
    result = _baseline(monkeypatch, env).play("fake", max_steps=4)
    # Same death-recovery accounting: a reset is a submitted action.
    assert result.steps == env.steps + env.resets - 1 == 4
    assert result.resets == env.resets


def test_baseline_suite_one_scorecard_every_env(monkeypatch):
    baseline = _baseline(monkeypatch, _FakeEnv(win_after=1))
    arcade = baseline._practice._arcade
    made = []
    arcade.get_environments = lambda: [
        SimpleNamespace(game_id=g) for g in ("a", "b")
    ]
    arcade.make = lambda game, **kw: made.append(game) or _FakeEnv(win_after=1)
    arcade.close_scorecard = (
        lambda card: arcade.closed.append(card) or SimpleNamespace(score=0.0)
    )
    result = baseline.play_all(max_steps=2)
    assert made == ["a", "b"]
    assert arcade.closed == ["card-1"]
    assert all(r.scorecard_id == "card-1" for r in result.games)
    assert all(r.won for r in result.games)


def test_baseline_has_no_mind_requirements(monkeypatch):
    """The stub isn't her: no game-mode entry, no physiology."""
    baseline = _baseline(monkeypatch, _FakeEnv(win_after=1))
    mind = baseline._practice._mind
    assert mind.is_gaming  # contract answer, not engagement
    assert mind.readiness().ok
    mind.world_episode_end("win")  # a no-op, and meant to be
    assert mind.game_decision["source"] == "baseline:uniform"

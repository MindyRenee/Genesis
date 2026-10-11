"""ARC adapter tests — the seam between arcengine and game_turn.

No network: the ``arc_agi.Arcade`` the adapter constructs is replaced
with a fake whose env speaks the same ``FrameDataRaw`` surface —
``frame`` (list of grids), ``available_actions``, ``state``,
``levels_completed``, ``win_levels``, ``reset``/``step``. The mind
side is a stub exposing the boundary the adapter consumes:
``is_gaming``, ``game_turn``, ``readiness``, ``game_decision``.
"""

from __future__ import annotations

import sys
import types
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])

from frontal_lobe.games.arc import ArcPractice, ArcResult


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
        self.seen_reasoning: list[object] = []

    def reset(self):
        self.resets += 1
        return _Frame(self.grid)

    def step(self, action, data=None, reasoning=None):
        self.steps += 1
        self.seen_actions.append(getattr(action, "name", str(action)))
        self.seen_reasoning.append(reasoning)
        if self.steps >= self.win_after:
            return _Frame(self.grid, state=_State.WIN, levels=2)
        return _Frame(self.grid)


class _FakeArcade:
    def __init__(self, env):
        self._env = env
        self.closed: list[str] = []
        self.made_with: dict = {}

    def create_scorecard(self):
        return "card-1"

    def make(self, game_id, scorecard_id=None, save_recording=False):
        self.made_with = {"scorecard_id": scorecard_id, "save_recording": save_recording}
        return self._env

    def close_scorecard(self, card):
        self.closed.append(card)


class _StubMind:
    """The boundary ArcPractice consumes — game mode plus the gate."""

    def __init__(self, *, gaming=True, ready=True):
        self.is_gaming = gaming
        self.lines: list[str] = []
        self.tokens = ["ACTION1", "ACTION2", "ACTION6 3 4"]
        self.game_decision = {"action": "ACTION1", "source": "field_leader"}
        self._ready = ready

    def readiness(self, *, require_body=True):
        return SimpleNamespace(
            ok=self._ready,
            reasons=("asleep",) if not self._ready else (),
        )

    def game_turn(self, line: str) -> str:
        self.lines.append(line)
        return self.tokens[len(self.lines) % len(self.tokens)]


def _practice(monkeypatch, env, *, mind: _StubMind | None = None) -> ArcPractice:
    fake_mod = types.SimpleNamespace(Arcade=lambda **kw: _FakeArcade(env))
    monkeypatch.setitem(sys.modules, "arc_agi", fake_mod)
    p = ArcPractice.__new__(ArcPractice)
    p._mind = mind or _StubMind()
    p._arcade = _FakeArcade(env)
    import tempfile

    p._tmp = Path(tempfile.mkdtemp())
    p._pending_event = ""
    p._instructions = ""
    p._info_sent = set()
    p._require_live = False
    p._record = True
    return p


def test_frame_line_encodes_grid_actions_levels(tmp_path, monkeypatch):
    p = _practice(monkeypatch, _FakeEnv())
    grid = [[(x * y) % 16 for x in range(6)] for y in range(5)]
    line = p._frame_line(_Frame(grid, actions=(1, 6), levels=1, win=3), "g-1")
    assert "ACTIONS 1,6" in line
    assert "LEVELS 1/3" in line
    assert "GAME g-1" in line
    ref = line.split("FRAME @")[1].split(" ")[0]
    rows = Path(ref).read_text().splitlines()
    assert rows == ["".join(f"{c:x}" for c in r) for r in grid]


def test_frame_line_appends_pending_event_once(monkeypatch):
    p = _practice(monkeypatch, _FakeEnv())
    p._pending_event = "over"
    f = _Frame([[0]])
    assert p._frame_line(f, "g").endswith("EVENT over")
    assert "EVENT" not in p._frame_line(f, "g")


def test_to_step_simple_and_coordinate(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    from arcengine import GameAction

    action, data = ArcPractice._to_step("ACTION3")
    assert action is GameAction.ACTION3 and data is None
    action, data = ArcPractice._to_step("ACTION6 3 4")
    assert action is GameAction.ACTION6 and data == {"x": 3, "y": 4}
    action, _ = ArcPractice._to_step("RESET")
    assert action is GameAction.RESET


def test_play_runs_to_win_and_closes_scorecard(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    p = _practice(monkeypatch, env)
    result = p.play("fake-1", max_steps=10)
    assert isinstance(result, ArcResult)
    assert result.won and result.levels_completed == 2
    assert env.steps == 3
    assert any("EVENT win" in ln for ln in p._mind.lines)


def test_play_budget_and_reset_path(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=999)
    result = _practice(monkeypatch, env).play("fake-2", max_steps=5)
    assert result.state == "BUDGET" and not result.won


def test_play_refuses_outside_game_mode(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    p = _practice(monkeypatch, _FakeEnv(), mind=_StubMind(gaming=False))
    with pytest.raises(RuntimeError, match="game mode"):
        p.play("fake-3", max_steps=5)


def test_play_refuses_when_unwell(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    p = _practice(monkeypatch, env, mind=_StubMind(ready=False))
    result = p.play("fake-4", max_steps=10)
    assert result.state == "UNWELL"
    assert "asleep" in result.reasons
    assert env.steps == 0 and env.resets == 0


def test_play_attaches_decision_reasoning(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    p = _practice(monkeypatch, env)
    p.play("fake-5", max_steps=10)
    assert env.seen_reasoning == [p._mind.game_decision] * env.steps


def test_play_requests_recording(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=1)
    p = _practice(monkeypatch, env)
    p.play("fake-6", max_steps=5)
    assert p._arcade.made_with["save_recording"] is True

"""Tests for the game-play loop — windowed-game gating and interrupts.

A windowed (system) episode grabs the user's screen and keyboard, so
it obeys two rules the internal games don't: a real gap between
sessions on the desktop, and a user interrupt that ends the episode
at the next move and closes the window.
"""

import time
from typing import ClassVar

import frontal_lobe.games.play as play_mod
from frontal_lobe.games.environment import GameObservation, GameOutcome
from frontal_lobe.games.play import GamePlay
from frontal_lobe.games.player import GamePlayer


class _Env:
    """A minimal never-ending environment for interrupt tests."""

    name = "fake"

    def reset(self):
        return GameObservation()

    def observe(self):
        return GameObservation()

    def actions(self):
        return ["a", "b"]

    def act(self, action):
        return GameOutcome(
            observation=GameObservation(), reward=0.0, info={}
        )

    def done(self):
        return False

    def score(self):
        return 0.0

    def close(self):
        pass


def test_should_stop_interrupts_episode():
    """The interrupt ends the episode at a move boundary."""
    calls = {"n": 0}

    def stop_after_three():
        calls["n"] += 1
        return calls["n"] > 3

    res = GamePlayer().play(
        _Env(), max_steps=100, should_stop=stop_after_three
    )
    assert res.cause == "interrupted"
    assert res.steps == 3
    assert not res.won


def test_should_stop_before_first_move():
    res = GamePlayer().play(
        _Env(), max_steps=100, should_stop=lambda: True
    )
    assert res.cause == "interrupted"
    assert res.steps == 0


def test_episode_runs_when_never_stopped():
    res = GamePlayer().play(
        _Env(), max_steps=7, should_stop=lambda: False
    )
    assert res.cause == "finished"
    assert res.steps == 7


def _game_play(tmp_path):
    return GamePlay(str(tmp_path))


def test_system_gap_gates_windowed_picks(tmp_path, monkeypatch):
    """While the desktop gap runs, system games leave the shelf."""
    gp = _game_play(tmp_path)
    spec = {"name": "windowed_thing", "family": "system",
            "command": "/x"}
    monkeypatch.setattr(gp, "_system_games", lambda: [spec])
    # Every internal game already mastered — a fresh system game
    # would dominate picks if the gate weren't there.
    gp.mastery = {
        str(t["name"]): 1.0 for t in play_mod.COLLECTION
    }
    assert gp.system_ready()
    gp._system_last_end = time.monotonic()
    assert not gp.system_ready()
    for _ in range(200):
        assert gp.pick(allow_system=False).get("family") != "system"
    gp._system_last_end = (
        time.monotonic() - play_mod._SYSTEM_GAP_SECONDS - 1
    )
    assert gp.system_ready()


class _FakeX11Game:
    """Stands in for the real windowed env — no X needed."""

    instances: ClassVar[list["_FakeX11Game"]] = []

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.deaths = 0
        self.closed_kill = None
        _FakeX11Game.instances.append(self)

    def close(self, kill: bool = False):
        self.closed_kill = kill


class _StubResult:
    def __init__(self, cause):
        self.score = 0.0
        self.steps = 1
        self.reward = 0.0
        self.won = False
        self.cause = cause
        self.learned = 0


def _system_task(tmp_path, monkeypatch):
    """A valid system task pointing at a fake executable."""
    games_dir = tmp_path / "games"
    games_dir.mkdir(exist_ok=True)
    exe = games_dir / "fakegame"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(0o755)
    monkeypatch.setattr(play_mod, "_SYSTEM_GAME_DIRS", (str(games_dir),))
    return {"name": "fakegame", "family": "system", "command": str(exe)}


def test_windowed_env_reused_between_episodes(tmp_path, monkeypatch):
    """One X11Game per game — no process pile-up across firings."""
    _FakeX11Game.instances.clear()
    import frontal_lobe.games.windowed as win_mod

    monkeypatch.setattr(win_mod, "X11Game", _FakeX11Game)
    gp = _game_play(tmp_path)
    task = _system_task(tmp_path, monkeypatch)
    monkeypatch.setattr(
        gp._player, "play",
        lambda env, **kw: _StubResult("finished"),
    )
    gp._attempt_system(task)
    gp._attempt_system(task)
    assert len(_FakeX11Game.instances) == 1
    env = _FakeX11Game.instances[0]
    assert env.closed_kill is False
    assert not gp.system_ready()


def test_interrupt_kills_the_window(tmp_path, monkeypatch):
    """A user interrupt ends the session *and* the window."""
    _FakeX11Game.instances.clear()
    import frontal_lobe.games.windowed as win_mod

    monkeypatch.setattr(win_mod, "X11Game", _FakeX11Game)
    gp = _game_play(tmp_path)
    task = _system_task(tmp_path, monkeypatch)
    seen = {}

    def fake_play(env, **kw):
        seen["should_stop"] = kw.get("should_stop")
        return _StubResult("interrupted")

    monkeypatch.setattr(gp._player, "play", fake_play)
    result = gp._attempt_system(task, should_stop=lambda: True)
    assert seen["should_stop"]() is True
    assert _FakeX11Game.instances[0].closed_kill is True
    assert result is not None
    assert result.failure == "interrupted"
    assert not gp.system_ready()

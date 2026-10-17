"""ARC adapter tests — the seam between arcengine and world_step.

No network: the ``arc_agi.Arcade`` the adapter constructs is replaced
with a fake whose env speaks the same ``FrameDataRaw`` surface —
``frame`` (list of grids), ``available_actions``, ``state``,
``levels_completed``, ``win_levels``, ``reset``/``step``. The mind
side is a stub exposing the boundary the adapter consumes:
``is_gaming``, ``world_step``, ``world_episode_end``, ``readiness``,
``game_decision``.
"""

from __future__ import annotations

import json
import sys
import time
import types
from enum import Enum
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])

from parietal_lobe.arc import ArcOfficial, ArcPractice, ArcResult


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
        self.cards: list[str] = []

    def create_scorecard(self, source_url=None, tags=None, opaque=None):
        return "card-1"

    def make(self, game_id, scorecard_id=None, save_recording=False):
        self.made_with = {"scorecard_id": scorecard_id, "save_recording": save_recording}
        return self._env

    def close_scorecard(self, card):
        self.closed.append(card)


class _StubMind:
    """The boundary ArcPractice consumes — the embodied loop plus the gate."""

    def __init__(self, *, gaming=True, ready=True):
        self.is_gaming = gaming
        self.turns: list[dict] = []
        self.episodes: list[str] = []
        self._descriptors = [
            {"action": 1},
            {"action": 2},
            {"action": 6, "x": 3, "y": 4},
        ]
        self.last_world_decision = {"action": "ACTION1", "source": "explore"}
        self._ready = ready

    def readiness(self, *, require_body=True):
        return SimpleNamespace(
            ok=self._ready,
            reasons=("asleep",) if not self._ready else (),
        )

    def world_step(self, frame, available_actions=None, info=None):
        self.turns.append(
            {"frame": frame, "actions": available_actions, "info": info}
        )
        desc = self._descriptors[len(self.turns) % len(self._descriptors)]
        offered = {item["action"] for item in available_actions}
        return desc if desc["action"] in offered else {"action": min(offered)}

    def world_episode_end(self, outcome, game="the game"):
        self.episodes.append(outcome)

    @property
    def game_decision(self):
        return self.last_world_decision


def _practice(monkeypatch, env, *, mind: _StubMind | None = None) -> ArcPractice:
    fake_mod = types.SimpleNamespace(Arcade=lambda **kw: _FakeArcade(env))
    monkeypatch.setitem(sys.modules, "arc_agi", fake_mod)
    p = ArcPractice.__new__(ArcPractice)
    p._mind = mind or _StubMind()  # type: ignore[assignment]
    p._arcade = _FakeArcade(env)
    p._instructions = ""
    p._info_sent = set()
    p._require_live = False
    p._record = True
    return p


def test_to_step_simple_and_coordinate(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    from arcengine import GameAction

    action, data = ArcPractice._to_step({"action": 3})
    assert action is GameAction.ACTION3 and data is None
    action, data = ArcPractice._to_step({"action": 6, "x": 3, "y": 4})
    assert action is GameAction.ACTION6 and data == {"x": 3, "y": 4}
    action, _ = ArcPractice._to_step({"action": 0})
    assert action is GameAction.RESET


def test_world_step_receives_frame_actions_levels(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv()
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p.play("fake-1", max_steps=1)
    turn = mind.turns[0]
    assert turn["frame"] == env.grid
    # The interface's own semantics label the whitelist: simple
    # actions arrive marked "simple", coordinate-takers "coordinate".
    assert turn["actions"] == [
        {"action": 1, "label": "simple"},
        {"action": 2, "label": "simple"},
        {"action": 3, "label": "simple"},
    ]
    assert turn["info"]["game"] == "fake-1"
    assert turn["info"]["level"] == 0
    assert turn["info"]["win_levels"] == 2


def test_instructions_sent_once_per_game(monkeypatch):
    env = _FakeEnv(win_after=99)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p._instructions = "this is a video game"
    p.play("fake-1", max_steps=3)
    infos = [t["info"] for t in mind.turns]
    assert infos[0].get("instructions") == "this is a video game"
    assert all("instructions" not in i for i in infos[1:])


def test_play_runs_to_win_and_closes_scorecard(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    result = p.play("fake-1", max_steps=10)
    assert isinstance(result, ArcResult)
    assert result.won and result.levels_completed == 2
    assert env.steps == 3
    assert mind.episodes == ["win"]
    assert p._arcade.closed == ["card-1"]


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
    mind = _StubMind(ready=False)
    result = _practice(monkeypatch, env, mind=mind).play("fake-4", max_steps=10)
    assert result.state == "UNWELL"
    assert "asleep" in result.reasons
    assert env.steps == 0 and env.resets == 0


def test_recovery_wait_proceeds_when_she_settles(monkeypatch):
    env = _FakeEnv(win_after=1)
    mind = _StubMind(ready=False)
    refusals_left = [2]

    def readiness(**kw):
        if refusals_left[0] > 0:
            refusals_left[0] -= 1
            mind._ready = False
        else:
            mind._ready = True
        return _StubMind.readiness(mind, **kw)

    mind.readiness = readiness
    practice = _practice(monkeypatch, env, mind=mind)
    practice._recovery_wait = 30.0
    monkeypatch.setattr(time, "sleep", lambda s: None)
    result = practice.play("fake", max_steps=3)
    assert result.won
    assert result.recovery_waits == 1


def test_recovery_wait_timeout_still_refuses(monkeypatch):
    env = _FakeEnv(win_after=3)
    mind = _StubMind(ready=False)
    practice = _practice(monkeypatch, env, mind=mind)
    practice._recovery_wait = 0.001
    result = practice.play("fake", max_steps=10)
    assert result.state == "UNWELL"
    assert result.recovery_waits == 1
    assert "asleep" in result.reasons
    assert env.steps == 0


def test_recovery_wait_lets_suite_continue(monkeypatch):
    mind = _StubMind()
    practice = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind)
    practice._recovery_wait = 30.0
    monkeypatch.setattr(time, "sleep", lambda s: None)
    practice._arcade.get_environments = lambda: [
        SimpleNamespace(game_id=g) for g in ("a", "b")
    ]
    end = mind.world_episode_end
    refusals_left = [0]

    def episode_end(outcome, game):
        end(outcome, game)
        refusals_left[0] = 1

    def readiness(**kw):
        if refusals_left[0] > 0:
            refusals_left[0] -= 1
            return SimpleNamespace(ok=False, reasons=("overstimulated",))
        return _StubMind.readiness(mind, **kw)

    mind.world_episode_end = episode_end
    mind.readiness = readiness
    result = practice.play_all(max_steps=1)
    assert [r.state for r in result.games] == ["WIN", "WIN"]
    assert result.games[1].recovery_waits == 1


def test_play_attaches_decision_reasoning(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p.play("fake-5", max_steps=10)
    assert env.seen_reasoning == [mind.game_decision] * env.steps


def test_play_requests_recording(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=1)
    p = _practice(monkeypatch, env)
    p.play("fake-6", max_steps=5)
    assert p._arcade.made_with["save_recording"] is True


def test_game_over_tells_episode_end_and_resets(monkeypatch):
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _DyingEnv(_FakeEnv):
        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = _DyingEnv(win_after=99)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p.play("fake-7", max_steps=4)
    assert "over" in mind.episodes
    assert env.resets >= 2


def test_mind_play_arc_wraps_game_mode_and_restores(monkeypatch):
    import mind as mind_mod

    calls: list[str] = []
    pacing: list[tuple[float, float]] = []

    class _Stub:
        is_gaming = False

        def enter_game_mode(self):
            if not self.is_gaming:
                calls.append("enter")
            self.is_gaming = True

        def exit_game_mode(self):
            calls.append("exit")
            self.is_gaming = False

    class _Practice:
        def __init__(self, mind, instructions="", step_delay_s=0.0,
                     recovery_wait_s=0.0):
            assert mind.is_gaming
            pacing.append((step_delay_s, recovery_wait_s))

        def play(self, game, max_steps=0):
            return ArcResult(game_id=game, state="BUDGET", steps=max_steps)

    monkeypatch.setattr("parietal_lobe.arc.ArcPractice", _Practice)
    stub = _Stub()
    out = mind_mod.Mind.play_arc(
        stub, ["g1", "g2"], max_steps=7,
        step_delay_s=0.25, recovery_wait_s=3.0,
    )
    assert [r.game_id for r in out] == ["g1", "g2"] and out[0].steps == 7
    assert pacing == [(0.25, 3.0)]
    assert calls == ["enter", "exit"] and not stub.is_gaming

    stub.is_gaming = True
    calls.clear()
    mind_mod.Mind.play_arc(stub, ["g3"])
    assert calls == [] and stub.is_gaming


def test_final_budget_move_reports_win_and_learning(monkeypatch):
    env = _FakeEnv(win_after=1)
    mind = _StubMind()
    result = _practice(monkeypatch, env, mind=mind).play("fake", max_steps=1)
    assert result.won
    assert result.levels_completed == result.win_levels == 2
    assert mind.episodes == ["win"]


def test_abstention_does_not_spend_an_action(monkeypatch):
    mind = _StubMind()
    mind.world_step = lambda *args, **kwargs: None
    env = _FakeEnv()
    result = _practice(monkeypatch, env, mind=mind).play("fake", max_steps=3)
    assert result.steps == env.steps == 0
    assert result.state == "NO_ACTION"


def test_unavailable_environment_closes_scorecard(monkeypatch):
    practice = _practice(monkeypatch, None)
    result = practice.play("missing", max_steps=3)
    assert result.state == "UNAVAILABLE"
    assert practice._arcade.closed == ["card-1"]


def test_step_exception_is_not_retried_and_closes_scorecard(monkeypatch):
    class BrokenEnv(_FakeEnv):
        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            raise TypeError("failure after applying action")

    env = BrokenEnv()
    practice = _practice(monkeypatch, env)
    with pytest.raises(TypeError, match="failure after applying action"):
        practice.play("fake", max_steps=3)
    assert env.steps == 1
    assert practice._arcade.closed == ["card-1"]


def test_legacy_step_signature_is_detected_before_acting(monkeypatch):
    class LegacyEnv(_FakeEnv):
        def step(self, action, data=None):
            return super().step(action, data=data)

    env = LegacyEnv(win_after=1)
    result = _practice(monkeypatch, env).play("fake", max_steps=2)
    assert result.won and env.steps == 1


def test_recovery_resets_spend_budget(monkeypatch):
    class DyingEnv(_FakeEnv):
        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = DyingEnv()
    result = _practice(monkeypatch, env).play("fake", max_steps=4)
    assert result.steps == env.steps + env.resets - 1 == 4
    assert result.resets == env.resets


def test_reset_returning_game_over_is_bounded(monkeypatch):
    class TerminalResetEnv(_FakeEnv):
        def reset(self):
            self.resets += 1
            if self.resets > 6:
                raise RuntimeError("unbounded reset loop")
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = TerminalResetEnv()
    result = _practice(monkeypatch, env).play("fake", max_steps=3)
    assert result.steps == 3
    assert env.resets == 4


def test_level_steps_record_per_level_costs(monkeypatch):
    """Steps per completed level — the within-game learning curve."""

    class _LevelingEnv(_FakeEnv):
        def __init__(self):
            super().__init__()
            self._levels = 0

        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            if self.steps in (2, 4):
                self._levels += 1
            if self.steps >= 5:
                return _Frame(self.grid, state=_State.WIN, levels=3, win=3)
            return _Frame(self.grid, levels=self._levels, win=3)

    result = _practice(monkeypatch, _LevelingEnv()).play("fake", max_steps=10)
    assert result.won
    assert result.level_steps == [2, 2, 1]
    assert sum(result.level_steps) <= result.steps


def test_level_steps_recompletion_appends_after_death(monkeypatch):
    """A level re-completed after a reset appends a second cost."""

    class _DieAndRelevel(_FakeEnv):
        def __init__(self):
            super().__init__()
            self._phase = 0

        def step(self, action, data=None, reasoning=None):
            self.steps += 1
            self._phase += 1
            if self._phase == 2:
                return _Frame(self.grid, state=_State.GAME_OVER, levels=1)
            if self._phase == 4:
                return _Frame(self.grid, state=_State.WIN, levels=2, win=2)
            return _Frame(self.grid, levels=1, win=2)

    result = _practice(monkeypatch, _DieAndRelevel()).play("fake", max_steps=10)
    assert result.won
    assert result.level_steps == [1, 1, 1]


def test_decision_sources_tally_winning_channels(monkeypatch):
    """Which bid won each submitted move — counted, not argued."""
    env = _FakeEnv(win_after=3)
    result = _practice(monkeypatch, env).play("fake", max_steps=10)
    assert result.decision_sources == {"explore": 3}


def test_prior_learning_reports_carried_experience(monkeypatch):
    """The result says whether past lessons rode into the run."""
    mind = _StubMind()
    mind._world_learning_loaded = "loaded"
    result = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind).play(
        "fake", max_steps=5
    )
    assert result.prior_learning == "loaded"

    result = _practice(monkeypatch, _FakeEnv(win_after=1)).play(
        "fake", max_steps=5
    )
    assert result.prior_learning == "none"


def test_mind_play_arc_fresh_scope_and_restore(monkeypatch):
    """A cold-start run sets the flag inside and restores it after."""
    import mind as mind_mod

    seen = []

    class _Stub:
        is_gaming = False
        _fresh_world_run = False

        def enter_game_mode(self):
            self.is_gaming = True

        def exit_game_mode(self):
            self.is_gaming = False

    class _Practice:
        def __init__(self, mind, instructions="", step_delay_s=0.0,
                     recovery_wait_s=0.0):
            self._m = mind

        def play(self, game, max_steps=0):
            seen.append(self._m._fresh_world_run)
            return ArcResult(game_id=game, state="BUDGET")

    monkeypatch.setattr("parietal_lobe.arc.ArcPractice", _Practice)
    stub = _Stub()
    mind_mod.Mind.play_arc(stub, ["g"], fresh=True)
    assert seen == [True]
    assert stub._fresh_world_run is False

    seen.clear()
    mind_mod.Mind.play_arc(stub, ["g"])
    assert seen == [False]


def test_mind_evaluation_fresh_scope(monkeypatch):
    from parietal_lobe.arc import ArcEvaluation

    import mind as mind_mod

    seen = []

    class Subject:
        is_gaming = False
        _fresh_world_run = False

        def enter_game_mode(self):
            self.is_gaming = True

        def exit_game_mode(self):
            self.is_gaming = False

    class Practice:
        def __init__(self, mind, step_delay_s=0.0, recovery_wait_s=0.0):
            self._m = mind
            self.pacing = (step_delay_s, recovery_wait_s)

        def play_all(self, max_steps):
            seen.append(self._m._fresh_world_run)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("parietal_lobe.arc.ArcPractice", Practice)
    subject = Subject()
    mind_mod.Mind.evaluate_arc(subject, max_steps=5, fresh=True)
    assert seen == [True]
    assert subject._fresh_world_run is False


def test_missing_observation_is_not_reported_as_budget(monkeypatch):
    class MissingFrameEnv(_FakeEnv):
        def step(self, action, data=None, reasoning=None):
            return None

    result = _practice(monkeypatch, MissingFrameEnv()).play("fake", max_steps=3)
    assert result.state == "NO_OBSERVATION"


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_does_not_create_environment(monkeypatch, budget):
    practice = _practice(monkeypatch, _FakeEnv())
    with pytest.raises(ValueError, match="positive"):
        practice.play("fake", max_steps=budget)
    assert not practice._arcade.made_with


def test_existing_initial_observation_does_not_reset_again(monkeypatch):
    env = _FakeEnv(win_after=1)
    env.observation_space = env.reset()
    practice = _practice(monkeypatch, env)
    result = practice.play("fake", max_steps=1)
    assert result.won and env.resets == result.resets == 1


def test_suite_uses_one_scorecard_and_every_environment_once(monkeypatch):
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    made = []
    opened = []
    arcade = practice._arcade
    arcade.get_environments = lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")]
    arcade.create_scorecard = (
        lambda **kw: opened.append("card-1") or "card-1"
    )
    arcade.make = lambda game, **kwargs: made.append((game, kwargs)) or _FakeEnv(win_after=1)
    arcade.close_scorecard = lambda card: arcade.closed.append(card) or SimpleNamespace(score=12.5)
    result = practice.play_all(max_steps=2)
    assert opened == arcade.closed == ["card-1"]
    assert [g for g, _ in made] == ["a", "b"]
    assert all(kw["scorecard_id"] == "card-1" for _, kw in made)
    assert result.score == 12.5
    assert result.mode == "offline"
    assert [r.game_id for r in result.games] == ["a", "b"]
    assert all(r.won for r in result.games)


def test_suite_cannot_restart_after_opening_scorecard(monkeypatch):
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    practice._arcade.get_environments = lambda: [SimpleNamespace(game_id="a")]
    practice.play_all(max_steps=1)
    with pytest.raises(RuntimeError, match="already"):
        practice.play_all(max_steps=1)


def test_suite_retains_unplayed_games_when_health_stops_run(monkeypatch):
    mind = _StubMind()
    practice = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind)
    practice._arcade.get_environments = lambda: [
        SimpleNamespace(game_id=g) for g in ("a", "b", "c")
    ]
    end = mind.world_episode_end

    def episode_end(outcome, game):
        end(outcome, game)
        mind._ready = False

    mind.world_episode_end = episode_end
    arcade = practice._arcade
    arcade.close_scorecard = lambda card: arcade.closed.append(card) or SimpleNamespace(score=100.0)
    result = practice.play_all(max_steps=1)
    assert [r.state for r in result.games] == ["WIN", "UNWELL", "NOT_PLAYED"]
    assert result.score is None
    assert practice._arcade.closed == ["card-1"]


def test_offline_adapter_refuses_forced_competition_before_sdk(monkeypatch, tmp_path):
    def forbidden_sdk(**kwargs):
        raise AssertionError("practice tried to construct the remote SDK")

    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(Arcade=forbidden_sdk))
    monkeypatch.setenv("OPERATION_MODE", "competition")
    with pytest.raises(RuntimeError, match="ArcOfficial"):
        ArcPractice(_StubMind(), environments_dir=tmp_path, recordings_dir=tmp_path)


def test_official_selects_competition_mode_without_network(monkeypatch, tmp_path):
    class Mode(Enum):
        OFFLINE = "offline"
        COMPETITION = "competition"

    configured = []
    monkeypatch.setenv("ARC_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(
        OperationMode=Mode,
        Arcade=lambda **kwargs: configured.append(kwargs) or _FakeArcade(_FakeEnv()),
    ))
    mind = _StubMind()
    official = ArcOfficial(
        mind, environments_dir=tmp_path, recordings_dir=tmp_path,
        card_path=tmp_path / "card.json",
    )
    assert configured[0]["operation_mode"] is Mode.COMPETITION
    # The card is never driven: no selected replays, no suite runner.
    with pytest.raises(RuntimeError, match="volition"):
        official.play("selected")
    with pytest.raises(RuntimeError, match="volition"):
        official.play_all()


def test_official_requires_explicit_access(monkeypatch):
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ARC_API_KEY"):
        ArcOfficial(_StubMind())


def test_official_requires_game_mode(monkeypatch):
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace())
    monkeypatch.setenv("ARC_API_KEY", "test-only-not-a-real-key")
    with pytest.raises(RuntimeError, match="game mode"):
        ArcOfficial(_StubMind(gaming=False))


def test_mind_evaluation_wraps_game_mode(monkeypatch):
    from parietal_lobe.arc import ArcEvaluation

    import mind as mind_mod

    calls = []

    class Subject:
        is_gaming = False

        def enter_game_mode(self):
            self.is_gaming = True
            calls.append("enter")

        def exit_game_mode(self):
            self.is_gaming = False
            calls.append("exit")

    class Practice:
        def __init__(self, mind, step_delay_s=0.0, recovery_wait_s=0.0):
            assert mind.is_gaming

        def play_all(self, max_steps):
            calls.append(max_steps)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("parietal_lobe.arc.ArcPractice", Practice)
    subject = Subject()
    result = mind_mod.Mind.evaluate_arc(subject, max_steps=12)
    assert result.mode == "offline"
    assert calls == ["enter", 12, "exit"]
    assert not subject.is_gaming


def test_arc_command_selects_full_suite():
    from parietal_lobe.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    calls = []
    subject = SimpleNamespace(evaluate_arc=lambda **kwargs: calls.append(kwargs) or ArcEvaluation(
        mode="offline",
        games=[ArcResult(game_id="g", state="WIN", levels_completed=1, win_levels=1)],
        score=75.0,
    ))
    report = _cmd_arc(subject, "--all --steps 12")
    assert calls == [{
        "max_steps": 12, "fresh": False,
        "step_delay_s": 0.0, "recovery_wait_s": 120.0,
    }]
    assert "75.00%" in report
    assert "levels 1/1" in report


def test_arc_command_passes_pacing_flags():
    from parietal_lobe.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    calls = []
    subject = SimpleNamespace(evaluate_arc=lambda **kwargs: calls.append(kwargs) or ArcEvaluation(
        mode="offline",
        games=[ArcResult(game_id="g", state="WIN", levels_completed=1,
                         win_levels=1, recovery_waits=2)],
        score=75.0,
    ))
    report = _cmd_arc(subject, "--all --delay 0.5 --recover 90")
    assert calls[0]["step_delay_s"] == 0.5
    assert calls[0]["recovery_wait_s"] == 90.0
    assert "recovery-waits=2" in report


def test_arc_command_passes_fresh_scope():
    from parietal_lobe.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    seen = []
    subject = SimpleNamespace(
        play_arc=lambda games, **kw: seen.append((games, kw))
        or [ArcResult(game_id=g) for g in games]
    )
    _cmd_arc(subject, "g --fresh")
    assert seen == [(
        ["g"],
        {"max_steps": 200, "fresh": True, "step_delay_s": 0.0,
         "recovery_wait_s": 120.0},
    )]

    seen.clear()
    subject = SimpleNamespace(
        evaluate_arc=lambda **kw: seen.append(kw) or ArcEvaluation(
            mode="offline", games=[]
        )
    )
    _cmd_arc(subject, "--all --fresh --steps 7")
    assert seen == [{
        "max_steps": 7, "fresh": True,
        "step_delay_s": 0.0, "recovery_wait_s": 120.0,
    }]


def test_arc_command_runs_baseline_not_agent(monkeypatch):
    import parietal_lobe.arc_baseline as baseline_mod

    from genesis_cli import _cmd_arc

    calls = []

    class _Runner:
        def __init__(self, policy):
            calls.append(policy)

        def play(self, game, max_steps):
            calls.append((game, max_steps))
            return ArcResult(game_id=game, state="BUDGET", steps=max_steps)

    monkeypatch.setattr(baseline_mod, "ArcBaseline", _Runner)
    subject = SimpleNamespace(
        play_arc=lambda **kw: pytest.fail("baseline ran the agent")
    )
    out = _cmd_arc(subject, "g1 --baseline uniform --steps 9")
    assert calls == ["uniform", ("g1", 9)]
    assert "baseline 'uniform'" in out and "BUDGET" in out


@pytest.mark.parametrize("args", ["--steps", "g --steps nope", "g --steps 0",
                                   "--competition g", "--all --competition",
                                   "--competition --baseline uniform",
                                   "--baseline"])
def test_arc_command_rejects_invalid_requests_without_play(args):
    from genesis_cli import _cmd_arc

    assert "Usage:" in _cmd_arc(SimpleNamespace(), args)


@pytest.mark.parametrize("desc", [
    {"action": 6}, {"action": 6, "x": -1, "y": 0},
    {"action": 6, "x": 0, "y": 64}, {"action": 6, "x": 0.5, "y": 1},
    {"action": 6, "x": True, "y": 1}, {"action": 1.5},
])
def test_invalid_action_payload_is_rejected(desc):
    with pytest.raises(ValueError):
        ArcPractice._to_step(desc)


def test_unavailable_action_is_not_sent(monkeypatch):
    subject = _StubMind()
    subject.world_step = lambda *args, **kwargs: {"action": 7}
    env = _FakeEnv()
    practice = _practice(monkeypatch, env, mind=subject)
    with pytest.raises(ValueError, match="available"):
        practice.play("fake", max_steps=1)
    assert env.steps == 0
    assert practice._arcade.closed == ["card-1"]


def test_scorecard_failure_is_visible(monkeypatch):
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))

    def failed_close(card):
        raise ConnectionError("scorecard unavailable")

    practice._arcade.close_scorecard = failed_close
    result = practice.play("fake", max_steps=1)
    assert result.state == "SCORECARD_ERROR"
    assert result.score is None
    assert result.reasons == ("scorecard closure failed",)


# ─── The official card ────────────────────────────────────


class _AnnouncingMind(_StubMind):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.announcements: list[str] = []
        self.game_mode_entries = 0

    def world_announce(self, text, *, topics=None):
        self.announcements.append((text, topics))
        return True

    def enter_game_mode(self):
        self.game_mode_entries += 1
        self.is_gaming = True

    def exit_game_mode(self):
        self.is_gaming = False


def _official(monkeypatch, tmp_path, *, mind=None, envs=("a", "b")) -> ArcOfficial:
    fake = _FakeArcade(_FakeEnv(win_after=1))
    monkeypatch.setattr(
        fake, "get_environments",
        lambda: [SimpleNamespace(game_id=g) for g in envs],
        raising=False,
    )
    monkeypatch.setattr(fake, "cards", [], raising=False)

    def _make_card(**kw):
        fake.cards.append("card-1")
        return "card-1"

    monkeypatch.setattr(fake, "create_scorecard", _make_card, raising=False)
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(Arcade=lambda **kw: fake))
    p = ArcOfficial.__new__(ArcOfficial)
    p._mind = mind or _AnnouncingMind()
    p._arcade = fake
    p._instructions = ""
    p._info_sent = set()
    p._require_live = False
    p._record = True
    p._card_path = tmp_path / "official_card.json"
    return p


def test_official_card_opens_persists_and_records_attempt(monkeypatch, tmp_path):
    mind = _AnnouncingMind()
    official = _official(monkeypatch, tmp_path, mind=mind)
    card = official.open_or_resume()
    assert card["scorecard_id"] == "card-1"
    assert len(card["games"]) == 2
    assert (tmp_path / "official_card.json").exists()

    result = official.attempt("a", max_steps=2)
    assert result is not None and result.state == "WIN"
    assert official._arcade.made_with["scorecard_id"] == "card-1"

    card = official.card_state()
    played = next(g for g in card["games"] if g["game_id"] == "a")
    assert played["status"] == "attempted" and played["state"] == "WIN"
    assert played["levels_completed"] == 2
    # The brief reached her before the environment did.
    assert mind.announcements and "card-1" in mind.announcements[0][0]
    assert "rules" in mind.announcements[0][1]


def test_official_resume_skips_attempted_and_never_reopens(monkeypatch, tmp_path):
    official = _official(monkeypatch, tmp_path)
    official.open_or_resume()
    official.attempt("a", max_steps=2)

    resumed = _official(monkeypatch, tmp_path)
    card = resumed.open_or_resume()  # same card, no second create
    assert card["scorecard_id"] == "card-1"
    assert resumed._arcade.cards == []
    assert resumed.attempt("a") is None          # burned — no retry
    second = resumed.attempt("b", max_steps=2)
    assert second is not None and second.state == "WIN"
    # Both attempted → next call closes the card as complete.
    assert resumed.attempt("a") is None
    assert resumed.card_state()["closed"] is True
    assert resumed.card_state()["close_reason"].startswith("complete")


def test_official_unwell_does_not_burn_the_attempt(monkeypatch, tmp_path):
    mind = _AnnouncingMind(ready=False)
    official = _official(monkeypatch, tmp_path, mind=mind)
    official.open_or_resume()
    result = official.attempt("a")
    assert result is not None and result.state == "UNWELL"
    game = next(g for g in official.card_state()["games"] if g["game_id"] == "a")
    assert game["status"] == "pending"          # she never entered
    assert not mind.announcements               # no brief before the gate


def test_official_closed_card_is_sealed_for_the_window(monkeypatch, tmp_path):
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    official._seal(card, "idle — test")
    assert official.attempt("a") is None
    with pytest.raises(RuntimeError, match="closed"):
        official.open_or_resume()


def test_official_expired_window_opens_a_new_card(monkeypatch, tmp_path):
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    card["expires_at"] = 1.0  # long past
    official._save_card(card)
    fresh = _official(monkeypatch, tmp_path)
    new = fresh.open_or_resume()
    assert new["scorecard_id"] == "card-1"      # fake returns the same id
    assert fresh._arcade.cards == ["card-1"]    # a second open happened
    assert not (tmp_path / "official_card.json").read_text() == json.dumps(card)


def test_official_shelf_shows_one_standing_challenge(monkeypatch, tmp_path):
    from frontal_lobe.games.play import _ARC_CHALLENGE, GamePlay

    monkeypatch.setenv("ARC_API_KEY", "test-only")
    shelf = GamePlay(tmp_path)
    env_dir = tmp_path / "arc" / "environment_files"
    for g in ("m0r0", "lp85"):
        (env_dir / g / "guid-1").mkdir(parents=True)
        (env_dir / g / "guid-1" / "metadata.json").write_text("{}")
    entries = shelf._arc_official_games()
    assert [t["name"] for t in entries] == [_ARC_CHALLENGE]
    assert entries[0]["live"] is False
    assert entries[0]["family"] == "arc_official"
    assert "prize" in entries[0]["goal"]

    # A live card keeps the one challenge on the shelf as the open
    # commitment — with deadline pressure, not a list of games.
    card = {
        "scorecard_id": "card-1", "opened_at": time.time(),
        "expires_at": time.time() + 3600, "closed": False,
        "last_action_at": time.time(),
        "games": [
            {"game_id": "lp85", "status": "attempted"},
            {"game_id": "m0r0", "status": "pending"},
        ],
    }
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    entries = shelf._arc_official_games()
    assert [t["name"] for t in entries] == [_ARC_CHALLENGE]
    assert entries[0]["live"] is True
    assert 0.0 <= entries[0]["idle_pressure"] <= 1.0

    # A card with nothing pending is done — the shelf empties.
    card["games"][1]["status"] = "attempted"
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    assert shelf._arc_official_games() == []

    # Sealed within its window → the shelf empties too.
    card["games"][1]["status"] = "pending"
    card["closed"] = True
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    assert shelf._arc_official_games() == []
    # No key → nothing is offered at all.
    monkeypatch.delenv("ARC_API_KEY")
    assert shelf._arc_official_games() == []


def test_challenge_brief_announced_once_per_window(monkeypatch, tmp_path):
    from frontal_lobe.games.play import GamePlay

    monkeypatch.setenv("ARC_API_KEY", "test-only")
    shelf = GamePlay(tmp_path)
    env_dir = tmp_path / "arc" / "environment_files"
    (env_dir / "m0r0" / "guid-1").mkdir(parents=True)
    (env_dir / "m0r0" / "guid-1" / "metadata.json").write_text("{}")
    first = shelf.arc_brief_due()
    assert first is not None
    assert "one recorded try" in first
    assert "15 minutes" in first
    assert "prize" in first
    # Already briefed for this window — not repeated at her.
    assert shelf.arc_brief_due() is None

    # An open card is a new key — she hears the state of her
    # commitment, once, with the games remaining.
    card = {
        "scorecard_id": "card-9", "opened_at": time.time(),
        "expires_at": time.time() + 3600, "closed": False,
        "last_action_at": time.time(),
        "games": [{"game_id": "m0r0", "status": "pending"}],
    }
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    live = shelf.arc_brief_due()
    assert live is not None and "already open" in live
    assert shelf.arc_brief_due() is None


def _fake_card_adapter(card):
    """A stub ArcOfficial for _play_card tests — the venue boundary
    faked at the adapter seam; card bookkeeping is its own dict."""

    class _Adapter:
        IDLE_LIMIT_S = 900.0

        def __init__(self, *args, **kwargs):
            self._card = card
            self.calls: list[str] = []

        def open_or_resume(self):
            return self._card

        def card_state(self):
            return self._card

        def attempt(self, game_id, max_steps=200):
            self.calls.append(game_id)
            game = next(
                g for g in self._card["games"] if g["game_id"] == game_id
            )
            game["status"] = "attempted"
            game["state"] = "WIN"
            self._card["last_action_at"] = time.time()
            return ArcResult(
                game_id=game_id, state="WIN",
                levels_completed=2, win_levels=2, steps=3,
            )

        def attempt_next(self, max_steps=200):
            self._card["closed"] = True
            self._card["close_reason"] = "complete — every game attempted"
            return None

    return _Adapter


def test_sitting_opens_card_and_plays_every_pending_game(monkeypatch, tmp_path):
    """Her pick of the challenge opens the card; one firing, one sitting."""
    from frontal_lobe.games.play import _ARC_CHALLENGE, GamePlay

    class Mode(Enum):
        OFFLINE = "offline"
        COMPETITION = "competition"

    fake = _FakeArcade(_FakeEnv(win_after=1))
    monkeypatch.setattr(
        fake, "get_environments",
        lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")],
        raising=False,
    )
    monkeypatch.setattr(fake, "cards", [], raising=False)

    def _make_card(**kw):
        fake.cards.append("card-1")
        return "card-1"

    monkeypatch.setattr(fake, "create_scorecard", _make_card, raising=False)
    monkeypatch.setenv("ARC_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(
        OperationMode=Mode, Arcade=lambda **kw: fake,
    ))
    mind = _AnnouncingMind(gaming=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(None, task={
        "name": _ARC_CHALLENGE, "family": "arc_official",
        "hint": "the official challenge",
    })
    assert result is not None and result.family == "arc_official"
    assert result.details["games"] == 2 and result.details["won"] == 2
    assert result.details["card_closed"] is True
    assert result.details["card_reason"].startswith("complete")
    # Each game burned exactly once, under its own name.
    assert shelf.attempts["arcagi-a"] == shelf.attempts["arcagi-b"] == 1
    # …and the sitting itself counts once on the challenge.
    assert shelf.attempts[_ARC_CHALLENGE] == 1
    # Game mode was held for the sitting, then released.
    assert mind.game_mode_entries == 1 and not mind.is_gaming
    card = json.loads((tmp_path / "arc" / "official_card.json").read_text())
    assert card["closed"] and card["close_reason"].startswith("complete")
    assert {g["state"] for g in card["games"]} == {"WIN"}
    # The venue's rules reached her before each environment did.
    assert mind.announcements


def test_sitting_does_not_open_the_card_when_unwell(monkeypatch, tmp_path):
    """An unwell pick is not a commitment — the challenge stays waiting."""
    from frontal_lobe.games.play import _ARC_CHALLENGE, GamePlay

    mind = _AnnouncingMind(gaming=False, ready=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(None, task={
        "name": _ARC_CHALLENGE, "family": "arc_official", "hint": "",
    })
    assert result is None
    assert not (tmp_path / "arc" / "official_card.json").exists()
    assert not mind.is_gaming and mind.game_mode_entries == 0


def test_sitting_waits_out_an_unwell_gate_inside_idle_budget(
    monkeypatch, tmp_path
):
    """A mid-card wellness gate pauses her; the game is not burned."""
    from frontal_lobe.games.play import _ARC_CHALLENGE, GamePlay

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    card = {
        "scorecard_id": "card-1", "opened_at": time.time(),
        "expires_at": time.time() + 3600, "closed": False,
        "last_action_at": time.time(),
        "games": [
            {"game_id": "a", "status": "pending"},
            {"game_id": "b", "status": "pending"},
        ],
    }
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    adapter_cls = _fake_card_adapter(card)
    monkeypatch.setattr("parietal_lobe.arc.ArcOfficial", adapter_cls)
    mind = _AnnouncingMind(gaming=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind

    # The second game she reaches for reports UNWELL once — the wait
    # happens inside the idle budget, then she plays it for real.
    original_attempt = adapter_cls.attempt

    def attempt_once_unwell(self, game_id, max_steps=200):
        if self.calls and not getattr(self, "_unwell_spent", False):
            self._unwell_spent = True
            return ArcResult(game_id=game_id, state="UNWELL", steps=0)
        return original_attempt(self, game_id, max_steps)

    adapter_cls.attempt = attempt_once_unwell
    result = shelf.attempt(None, task={
        "name": _ARC_CHALLENGE, "family": "arc_official", "hint": "",
    })
    assert result is not None
    assert result.details["games"] == 2 and result.details["won"] == 2
    assert shelf.attempts["arcagi-a"] == shelf.attempts["arcagi-b"] == 1


def test_sitting_ends_when_she_leaves_the_arcade(monkeypatch, tmp_path):
    """Dropping game mode ends the sitting honestly — the card keeps
    what she played and stays open on the venue's clock."""
    from frontal_lobe.games.play import _ARC_CHALLENGE, GamePlay

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    card = {
        "scorecard_id": "card-1", "opened_at": time.time(),
        "expires_at": time.time() + 3600, "closed": False,
        "last_action_at": time.time(),
        "games": [
            {"game_id": "a", "status": "pending"},
            {"game_id": "b", "status": "pending"},
        ],
    }
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    mind = _AnnouncingMind(gaming=False)
    adapter_cls = _fake_card_adapter(card)
    original_attempt = adapter_cls.attempt

    def walk_away(self, game_id, max_steps=200):
        result = original_attempt(self, game_id, max_steps)
        mind.is_gaming = False  # /endgame — she left
        return result

    adapter_cls.attempt = walk_away
    monkeypatch.setattr("parietal_lobe.arc.ArcOfficial", adapter_cls)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(None, task={
        "name": _ARC_CHALLENGE, "family": "arc_official", "hint": "",
    })
    assert result is not None
    assert result.details["games"] == 1
    assert result.details["card_closed"] is False
    pending = [g for g in card["games"] if g["status"] == "pending"]
    assert len(pending) == 1

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

import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])

from world.arc import ArcPractice, ArcResult


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

    def create_scorecard(self):
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
        def __init__(self, mind, instructions=""):
            assert mind.is_gaming

        def play(self, game, max_steps=0):
            return ArcResult(game_id=game, state="BUDGET", steps=max_steps)

    monkeypatch.setattr("world.arc.ArcPractice", _Practice)
    stub = _Stub()
    out = mind_mod.Mind.play_arc(stub, ["g1", "g2"], max_steps=7)
    assert [r.game_id for r in out] == ["g1", "g2"] and out[0].steps == 7
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
        def __init__(self, mind, instructions=""):
            self._m = mind

        def play(self, game, max_steps=0):
            seen.append(self._m._fresh_world_run)
            return ArcResult(game_id=game, state="BUDGET")

    monkeypatch.setattr("world.arc.ArcPractice", _Practice)
    stub = _Stub()
    mind_mod.Mind.play_arc(stub, ["g"], fresh=True)
    assert seen == [True]
    assert stub._fresh_world_run is False

    seen.clear()
    mind_mod.Mind.play_arc(stub, ["g"])
    assert seen == [False]


def test_mind_evaluation_fresh_scope(monkeypatch):
    import mind as mind_mod
    from world.arc import ArcEvaluation

    seen = []

    class Subject:
        is_gaming = False
        _fresh_world_run = False

        def enter_game_mode(self):
            self.is_gaming = True

        def exit_game_mode(self):
            self.is_gaming = False

    class Practice:
        def __init__(self, mind):
            self._m = mind

        def play_all(self, max_steps):
            seen.append(self._m._fresh_world_run)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("world.arc.ArcPractice", Practice)
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
    arcade.create_scorecard = lambda: opened.append("card-1") or "card-1"
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
    with pytest.raises(RuntimeError, match="ArcCompetition"):
        ArcPractice(_StubMind(), environments_dir=tmp_path, recordings_dir=tmp_path)


def test_competition_selects_official_mode_without_network(monkeypatch, tmp_path):
    from world.arc import ArcCompetition

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
    mind._offline = True
    competition = ArcCompetition(mind, environments_dir=tmp_path, recordings_dir=tmp_path)
    assert configured[0]["operation_mode"] is Mode.COMPETITION
    with pytest.raises(RuntimeError, match="play_all"):
        competition.play("selected")


def test_competition_requires_explicit_access(monkeypatch):
    from world.arc import ArcCompetition

    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ARC_API_KEY"):
        ArcCompetition(_StubMind())


def test_competition_requires_offline_cognition(monkeypatch):
    from world.arc import ArcCompetition

    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace())
    monkeypatch.setenv("ARC_API_KEY", "test-only-not-a-real-key")
    with pytest.raises(RuntimeError, match="offline"):
        ArcCompetition(_StubMind())


def test_mind_evaluation_wraps_game_mode(monkeypatch):
    import mind as mind_mod
    from world.arc import ArcEvaluation

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
        def __init__(self, mind):
            assert mind.is_gaming

        def play_all(self, max_steps):
            calls.append(max_steps)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("world.arc.ArcPractice", Practice)
    subject = Subject()
    result = mind_mod.Mind.evaluate_arc(subject, max_steps=12)
    assert result.mode == "offline"
    assert calls == ["enter", 12, "exit"]
    assert not subject.is_gaming


@pytest.mark.parametrize("flag,competition", [("--all", False), ("--competition", True)])
def test_arc_command_selects_full_suite(flag, competition):
    from genesis_cli import _cmd_arc
    from world.arc import ArcEvaluation

    calls = []
    subject = SimpleNamespace(evaluate_arc=lambda **kwargs: calls.append(kwargs) or ArcEvaluation(
        mode="competition" if competition else "offline",
        games=[ArcResult(game_id="g", state="WIN", levels_completed=1, win_levels=1)],
        score=75.0,
    ))
    report = _cmd_arc(subject, f"{flag} --steps 12")
    assert calls == [{"max_steps": 12, "competition": competition, "fresh": False}]
    assert "75.00%" in report
    assert "levels 1/1" in report


def test_arc_command_passes_fresh_scope():
    from genesis_cli import _cmd_arc
    from world.arc import ArcEvaluation

    seen = []
    subject = SimpleNamespace(
        play_arc=lambda games, **kw: seen.append((games, kw))
        or [ArcResult(game_id=g) for g in games]
    )
    _cmd_arc(subject, "g --fresh")
    assert seen == [(["g"], {"max_steps": 200, "fresh": True})]

    seen.clear()
    subject = SimpleNamespace(
        evaluate_arc=lambda **kw: seen.append(kw) or ArcEvaluation(
            mode="offline", games=[]
        )
    )
    _cmd_arc(subject, "--all --fresh --steps 7")
    assert seen == [{"max_steps": 7, "competition": False, "fresh": True}]


def test_arc_command_runs_baseline_not_agent(monkeypatch):
    import world.arc_baseline as baseline_mod
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

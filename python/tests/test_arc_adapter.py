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

from worlds.arc import ArcOfficial, ArcPractice, ArcResult


@pytest.fixture(scope="module", autouse=True)
def require_arcengine():
    """Skip the whole module when the optional engine isn't installed."""
    pytest.importorskip("arcengine", reason="optional ARC engine not installed")


class _State(Enum):
    """The GameState surface the fake env reports."""

    NOT_FINISHED = 0
    WIN = 1
    GAME_OVER = 2


class _Frame:
    """A FrameDataRaw-shaped observation: grid strip, whitelist, verdict."""

    def __init__(self, grid, actions=(1, 2, 3), state=_State.NOT_FINISHED, levels=0, win=2):
        """Build a fake frame — same fields the adapter reads."""
        self.frame = [grid]
        self.available_actions = list(actions)
        self.state = state
        self.levels_completed = levels
        self.win_levels = win


class _FakeEnv:
    """A tiny 8×8 world that 'wins' after ``win_after`` steps."""

    def __init__(self, win_after=3):
        """A deterministic world that 'wins' at the step count given."""
        self.grid = [[(x + y) % 16 for x in range(8)] for y in range(8)]
        self.win_after = win_after
        self.steps = 0
        self.resets = 0
        self.seen_actions: list[str] = []
        self.seen_reasoning: list[object] = []

    def reset(self):
        """Restart: count the reset and return a fresh frame."""
        self.resets += 1
        return _Frame(self.grid)

    def step(self, action, data=None, reasoning=None):
        """Apply one action: count it, record what was submitted."""
        self.steps += 1
        self.seen_actions.append(getattr(action, "name", str(action)))
        self.seen_reasoning.append(reasoning)
        if self.steps >= self.win_after:
            return _Frame(self.grid, state=_State.WIN, levels=2)
        return _Frame(self.grid)


class _FakeArcade:
    """The ``arc_agi.Arcade`` surface the adapter binds: scorecards,
    ``make()``, ``close_scorecard()`` — with the calls recorded."""

    def __init__(self, env):
        """Serve the one env for every game_id asked for."""
        self._env = env
        self.closed: list[str] = []
        self.made_with: dict = {}
        self.cards: list[str] = []

    def create_scorecard(self, source_url=None, tags=None, opaque=None):
        """Open a fake card — always ``"card-1"``."""
        return "card-1"

    def make(self, game_id, scorecard_id=None, save_recording=False):
        """Return the env, recording the args it was opened with."""
        self.made_with = {"scorecard_id": scorecard_id, "save_recording": save_recording}
        return self._env

    def close_scorecard(self, card):
        """Record the close so tests see the card was sealed."""
        self.closed.append(card)


class _FakeCabinet:
    """A cabinet double: records dispatches and renders, echoes arrivals.

    ``arrival`` is what the glass answers — ``"echo"`` hands the asked
    action back as the physically-arrived one; anything else (None) is
    a press that never landed.
    """

    def __init__(self):
        """Empty glass: nothing dispatched, nothing rendered, open."""
        self.dispatched: list[str] = []
        self.renders = 0
        self.closed = False
        self.arrival: object = "echo"

    def render(self, frame):
        """Count the repaint — the board went to the glass."""
        self.renders += 1

    def dispatch(self, action, data):
        """Record the act; answer with the configured arrival."""
        self.dispatched.append(getattr(action, "name", str(action)))
        return (action, data) if self.arrival == "echo" else self.arrival

    def close(self):
        """The window is gone."""
        self.closed = True


class _StubMind:
    """The boundary ArcPractice consumes — the embodied loop plus the gate."""

    def __init__(self, *, gaming=True, ready=True):
        """A scripted mind: cycling descriptors, fixed readiness."""
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
        """Report ok/unwell per the constructor flag."""
        return SimpleNamespace(
            ok=self._ready,
            reasons=("asleep",) if not self._ready else (),
        )

    def world_step(self, frame, available_actions=None, info=None, through=None):
        """Record the turn; answer with the next scripted descriptor
        that the offered whitelist permits."""
        self.turns.append(
            {"frame": frame, "actions": available_actions, "info": info, "through": through}
        )
        desc = self._descriptors[len(self.turns) % len(self._descriptors)]
        offered = {item["action"] for item in available_actions}
        return desc if desc["action"] in offered else {"action": min(offered)}

    def world_episode_end(self, outcome, game="the game"):
        """Record the episode outcome she was told."""
        self.episodes.append(outcome)

    @property
    def game_decision(self):
        """The striatum basis the adapter attaches to steps."""
        return self.last_world_decision


def _practice(monkeypatch, env, *, mind: _StubMind | None = None) -> ArcPractice:
    """Build an ArcPractice wired to the fakes, constructor skipped —
    same fields ``__init__`` sets, no SDK import needed."""
    fake_mod = types.SimpleNamespace(Arcade=lambda **kw: _FakeArcade(env))
    monkeypatch.setitem(sys.modules, "arc_agi", fake_mod)
    p = ArcPractice.__new__(ArcPractice)
    p._mind = mind or _StubMind()  # type: ignore[assignment]
    p._arcade = _FakeArcade(env)
    p._instructions = ""
    p._info_sent = set()
    p._require_live = False
    p._record = True
    p._x11_hands = True
    p._cabinet = None
    return p


def test_to_step_simple_and_coordinate(monkeypatch):
    """Descriptor → GameAction: simple, coordinate-with-data, RESET."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    from arcengine import GameAction

    action, data = ArcPractice._to_step({"action": 3})
    assert action is GameAction.ACTION3 and data is None
    action, data = ArcPractice._to_step({"action": 6, "x": 3, "y": 4})
    assert action is GameAction.ACTION6 and data == {"x": 3, "y": 4}
    action, _ = ArcPractice._to_step({"action": 0})
    assert action is GameAction.RESET


def test_world_step_receives_frame_actions_levels(monkeypatch):
    """The percept, the labeled whitelist, and level info reach her."""
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
    """The player's manual lands on the first frame only."""
    env = _FakeEnv(win_after=99)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p._instructions = "this is a video game"
    p.play("fake-1", max_steps=3)
    infos = [t["info"] for t in mind.turns]
    assert infos[0].get("instructions") == "this is a video game"
    assert all("instructions" not in i for i in infos[1:])


def test_play_runs_to_win_and_closes_scorecard(monkeypatch):
    """A winning run reports WIN, tells her, and seals the card."""
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
    """An unwinnable game ends on the step budget, not a victory."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=999)
    result = _practice(monkeypatch, env).play("fake-2", max_steps=5)
    assert result.state == "BUDGET" and not result.won


def test_play_refuses_outside_game_mode(monkeypatch):
    """No game mode, no run — the engagement contract is enforced."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    p = _practice(monkeypatch, _FakeEnv(), mind=_StubMind(gaming=False))
    with pytest.raises(RuntimeError, match="game mode"):
        p.play("fake-3", max_steps=5)


def test_play_refuses_when_unwell(monkeypatch):
    """An unwell body spends no steps and no resets — UNWELL."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    mind = _StubMind(ready=False)
    result = _practice(monkeypatch, env, mind=mind).play("fake-4", max_steps=10)
    assert result.state == "UNWELL"
    assert "asleep" in result.reasons
    assert env.steps == 0 and env.resets == 0


def test_recovery_wait_proceeds_when_she_settles(monkeypatch):
    """The gate waits for her to settle, then the run proceeds."""
    env = _FakeEnv(win_after=1)
    mind = _StubMind(ready=False)
    refusals_left = [2]

    def readiness(**kw):
        """Unwell for two probes, then settled."""
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
    """Still unwell when the wait expires — the run is refused."""
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
    """A mid-suite wellness pause delays a game instead of ending it."""
    mind = _StubMind()
    practice = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind)
    practice._recovery_wait = 30.0
    monkeypatch.setattr(time, "sleep", lambda s: None)
    practice._arcade.get_environments = lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")]
    end = mind.world_episode_end
    refusals_left = [0]

    def episode_end(outcome, game):
        """After the first win she turns unwell once."""
        end(outcome, game)
        refusals_left[0] = 1

    def readiness(**kw):
        """Refuse once after the win, then recover."""
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
    """Her decision basis rides every submitted step's reasoning field."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=3)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p.play("fake-5", max_steps=10)
    assert env.seen_reasoning == [mind.game_decision] * env.steps


def test_play_requests_recording(monkeypatch):
    """Runs ask the SDK to record — evidence is the point."""
    pytest.importorskip("arcengine", reason="arcengine not installed")
    env = _FakeEnv(win_after=1)
    p = _practice(monkeypatch, env)
    p.play("fake-6", max_steps=5)
    assert p._arcade.made_with["save_recording"] is True


def test_game_over_tells_episode_end_and_resets(monkeypatch):
    """A death is told to her, then the episode restarts while budget lasts."""
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _DyingEnv(_FakeEnv):
        """Every step kills her."""

        def step(self, action, data=None, reasoning=None):
            """Report GAME_OVER on every action."""
            self.steps += 1
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = _DyingEnv(win_after=99)
    mind = _StubMind()
    p = _practice(monkeypatch, env, mind=mind)
    p.play("fake-7", max_steps=4)
    assert "over" in mind.episodes
    assert env.resets >= 2


def test_mind_play_arc_wraps_game_mode_and_restores(monkeypatch):
    """play_arc enters game mode around the run and restores after —
    and leaves an already-engaged mind untouched."""
    import mind as mind_mod

    calls: list[str] = []
    pacing: list[tuple[float, float]] = []

    class _Stub:
        """Minimal engaged/disengaged flag with call recording."""

        is_gaming = False

        def enter_game_mode(self):
            """Record the engagement."""
            if not self.is_gaming:
                calls.append("enter")
            self.is_gaming = True

        def exit_game_mode(self):
            """Record the release."""
            calls.append("exit")
            self.is_gaming = False

    class _Practice:
        """Captures the pacing kwargs and fakes a budget run."""

        def __init__(self, mind, instructions="", step_delay_s=0.0, recovery_wait_s=0.0):
            """Assert engagement and record pacing."""
            assert mind.is_gaming
            pacing.append((step_delay_s, recovery_wait_s))

        def play(self, game, max_steps=0):
            """Return a canned BUDGET result."""
            return ArcResult(game_id=game, state="BUDGET", steps=max_steps)

    monkeypatch.setattr("worlds.arc.ArcPractice", _Practice)
    stub = _Stub()
    out = mind_mod.Mind.play_arc(
        stub,
        ["g1", "g2"],
        max_steps=7,
        step_delay_s=0.25,
        recovery_wait_s=3.0,
    )
    assert [r.game_id for r in out] == ["g1", "g2"] and out[0].steps == 7
    assert pacing == [(0.25, 3.0)]
    assert calls == ["enter", "exit"] and not stub.is_gaming

    stub.is_gaming = True
    calls.clear()
    mind_mod.Mind.play_arc(stub, ["g3"])
    assert calls == [] and stub.is_gaming


def test_final_budget_move_reports_win_and_learning(monkeypatch):
    """A win on the last budgeted step still counts as a win."""
    env = _FakeEnv(win_after=1)
    mind = _StubMind()
    result = _practice(monkeypatch, env, mind=mind).play("fake", max_steps=1)
    assert result.won
    assert result.levels_completed == result.win_levels == 2
    assert mind.episodes == ["win"]


def test_abstention_does_not_spend_an_action(monkeypatch):
    """Her abstaining is NO_ACTION — nothing is submitted for her."""
    mind = _StubMind()
    mind.world_step = lambda *args, **kwargs: None
    env = _FakeEnv()
    result = _practice(monkeypatch, env, mind=mind).play("fake", max_steps=3)
    assert result.steps == env.steps == 0
    assert result.state == "NO_ACTION"


def test_unavailable_environment_closes_scorecard(monkeypatch):
    """A game the arcade can't make is UNAVAILABLE — card still sealed."""
    practice = _practice(monkeypatch, None)
    result = practice.play("missing", max_steps=3)
    assert result.state == "UNAVAILABLE"
    assert practice._arcade.closed == ["card-1"]


def test_step_exception_is_not_retried_and_closes_scorecard(monkeypatch):
    """A step that raises is never re-fired — the world may have moved."""

    class BrokenEnv(_FakeEnv):
        """A step that dies after being counted."""

        def step(self, action, data=None, reasoning=None):
            """Raise after counting — simulating a post-apply crash."""
            self.steps += 1
            raise TypeError("failure after applying action")

    env = BrokenEnv()
    practice = _practice(monkeypatch, env)
    with pytest.raises(TypeError, match="failure after applying action"):
        practice.play("fake", max_steps=3)
    assert env.steps == 1
    assert practice._arcade.closed == ["card-1"]


def test_legacy_step_signature_is_detected_before_acting(monkeypatch):
    """An env without the reasoning kwarg is detected by signature,
    not by crashing on a submitted action."""

    class LegacyEnv(_FakeEnv):
        """The pre-reasoning step signature."""

        def step(self, action, data=None):
            """Step without a reasoning kwarg."""
            return super().step(action, data=data)

    env = LegacyEnv(win_after=1)
    result = _practice(monkeypatch, env).play("fake", max_steps=2)
    assert result.won and env.steps == 1


def test_recovery_resets_spend_budget(monkeypatch):
    """Each death-and-reset costs a step — the budget is honest."""

    class DyingEnv(_FakeEnv):
        """Every step kills her — deaths force resets."""

        def step(self, action, data=None, reasoning=None):
            """Report GAME_OVER on every action."""
            self.steps += 1
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = DyingEnv()
    result = _practice(monkeypatch, env).play("fake", max_steps=4)
    assert result.steps == env.steps + env.resets - 1 == 4
    assert result.resets == env.resets


def test_reset_returning_game_over_is_bounded(monkeypatch):
    """A world that dies on every reset cannot loop forever."""

    class TerminalResetEnv(_FakeEnv):
        """A reset that always returns a dead frame — and raises if
        the loop were unbounded, making the test fail not hang."""

        def reset(self):
            """Count resets; die on frame, explode if unbounded."""
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
        """Levels complete at fixed step counts, win at 5."""

        def __init__(self):
            """Track the running level count."""
            super().__init__()
            self._levels = 0

        def step(self, action, data=None, reasoning=None):
            """Level up on steps 2 and 4; WIN at step 5."""
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
        """Dies mid-run holding a level, then re-completes it."""

        def __init__(self):
            """Track the scripted phase."""
            super().__init__()
            self._phase = 0

        def step(self, action, data=None, reasoning=None):
            """GAME_OVER on step 2, WIN on step 4, level 1 otherwise."""
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
    result = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind).play("fake", max_steps=5)
    assert result.prior_learning == "loaded"

    result = _practice(monkeypatch, _FakeEnv(win_after=1)).play("fake", max_steps=5)
    assert result.prior_learning == "none"


def test_mind_play_arc_fresh_scope_and_restore(monkeypatch):
    """A cold-start run sets the flag inside and restores it after."""
    import mind as mind_mod

    seen = []

    class _Stub:
        """Engaged flag plus the fresh-run scope field."""

        is_gaming = False
        _fresh_world_run = False

        def enter_game_mode(self):
            """Mark her gaming."""
            self.is_gaming = True

        def exit_game_mode(self):
            """Release engagement."""
            self.is_gaming = False

    class _Practice:
        """Records the fresh flag seen inside each play."""

        def __init__(self, mind, instructions="", step_delay_s=0.0, recovery_wait_s=0.0):
            """Keep the mind to inspect mid-run."""
            self._m = mind

        def play(self, game, max_steps=0):
            """Snapshot the flag, return BUDGET."""
            seen.append(self._m._fresh_world_run)
            return ArcResult(game_id=game, state="BUDGET")

    monkeypatch.setattr("worlds.arc.ArcPractice", _Practice)
    stub = _Stub()
    mind_mod.Mind.play_arc(stub, ["g"], fresh=True)
    assert seen == [True]
    assert stub._fresh_world_run is False

    seen.clear()
    mind_mod.Mind.play_arc(stub, ["g"])
    assert seen == [False]


def test_mind_evaluation_fresh_scope(monkeypatch):
    """evaluate_arc scopes fresh=True to the suite, then restores."""
    from worlds.arc import ArcEvaluation

    import mind as mind_mod

    seen = []

    class Subject:
        """Engaged flag plus the fresh-run scope field."""

        is_gaming = False
        _fresh_world_run = False

        def enter_game_mode(self):
            """Mark her gaming."""
            self.is_gaming = True

        def exit_game_mode(self):
            """Release engagement."""
            self.is_gaming = False

    class Practice:
        """Records the fresh flag seen inside the suite."""

        def __init__(self, mind, step_delay_s=0.0, recovery_wait_s=0.0):
            """Keep the mind and record pacing kwargs."""
            self._m = mind
            self.pacing = (step_delay_s, recovery_wait_s)

        def play_all(self, max_steps):
            """Snapshot the flag, return an empty evaluation."""
            seen.append(self._m._fresh_world_run)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("worlds.arc.ArcPractice", Practice)
    subject = Subject()
    mind_mod.Mind.evaluate_arc(subject, max_steps=5, fresh=True)
    assert seen == [True]
    assert subject._fresh_world_run is False


def test_missing_observation_is_not_reported_as_budget(monkeypatch):
    """A world that stops answering is NO_OBSERVATION, not BUDGET."""

    class MissingFrameEnv(_FakeEnv):
        """The world that answers steps with silence."""

        def step(self, action, data=None, reasoning=None):
            """Return no frame — the swallowed-crash shape."""
            return None

    result = _practice(monkeypatch, MissingFrameEnv()).play("fake", max_steps=3)
    assert result.state == "NO_OBSERVATION"


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget_does_not_create_environment(monkeypatch, budget):
    """A nonpositive budget is rejected before the env is built."""
    practice = _practice(monkeypatch, _FakeEnv())
    with pytest.raises(ValueError, match="positive"):
        practice.play("fake", max_steps=budget)
    assert not practice._arcade.made_with


def test_existing_initial_observation_does_not_reset_again(monkeypatch):
    """An env that already holds a frame isn't reset a second time."""
    env = _FakeEnv(win_after=1)
    env.observation_space = env.reset()
    practice = _practice(monkeypatch, env)
    result = practice.play("fake", max_steps=1)
    assert result.won and env.resets == result.resets == 1


def test_suite_uses_one_scorecard_and_every_environment_once(monkeypatch):
    """One card opened, every discovered env made once on it, closed."""
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    made = []
    opened = []
    arcade = practice._arcade
    arcade.get_environments = lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")]
    arcade.create_scorecard = lambda **kw: opened.append("card-1") or "card-1"
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
    """A suite object that already ran refuses to open a second card."""
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    practice._arcade.get_environments = lambda: [SimpleNamespace(game_id="a")]
    practice.play_all(max_steps=1)
    with pytest.raises(RuntimeError, match="already"):
        practice.play_all(max_steps=1)


def test_suite_reports_sdk_scores_for_each_game(monkeypatch):
    """The closed card's per-environment scores land on each result."""
    from arc_agi.scorecard import EnvironmentScore, EnvironmentScorecard, EnvironmentScoreList

    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    practice._arcade.get_environments = lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")]
    practice._arcade.make = lambda *args, **kwargs: _FakeEnv(win_after=1)
    practice._arcade.close_scorecard = lambda card: EnvironmentScorecard(
        card_id=card,
        score=37.5,
        environments=[
            EnvironmentScoreList(
                id=game,
                runs=[EnvironmentScore(score=score, levels_completed=2, actions=1)],
            )
            for game, score in (("a", 25.0), ("b", 50.0))
        ],
    )
    result = practice.play_all(max_steps=1)
    assert result.score == 37.5
    assert [game.score for game in result.games] == [25.0, 50.0]


def test_practice_constructor_starts_without_a_cabinet(monkeypatch, tmp_path):
    """X11 disabled by env: the constructor binds OFFLINE and no hands."""

    class Mode(Enum):
        """The OperationMode surface the constructor binds."""

        OFFLINE = "offline"

    configured = []
    monkeypatch.setenv("OPERATION_MODE", "offline")
    monkeypatch.setenv("GENESIS_ARC_X11", "off")
    monkeypatch.setitem(
        sys.modules,
        "arc_agi",
        SimpleNamespace(
            OperationMode=Mode,
            Arcade=lambda **kw: configured.append(kw) or _FakeArcade(_FakeEnv()),
        ),
    )
    practice = ArcPractice(_StubMind(), environments_dir=tmp_path, recordings_dir=tmp_path)
    assert configured[0]["operation_mode"] is Mode.OFFLINE
    assert practice._cabinet is None
    assert practice._require_live is True
    assert practice._x11_hands is False


def test_disabled_cabinet_never_probes_display(monkeypatch):
    """GENESIS_ARC_X11=off means the display is never even probed."""
    from worlds import arc_cabinet

    def forbidden_display():
        """Tripwire — any display probe fails the test outright."""
        pytest.fail("disabled X11 transport probed the display")

    monkeypatch.setattr(arc_cabinet, "x11_reachable", forbidden_display)
    practice = _practice(monkeypatch, _FakeEnv())
    practice._x11_hands = False
    assert practice._open_cabinet("a") is None


def test_no_display_for_her_means_bench_play(monkeypatch):
    """When even the nested display can't come up, hands stay off."""
    from worlds import arc_cabinet

    monkeypatch.setattr(arc_cabinet, "ensure_arc_display", lambda: None)
    practice = _practice(monkeypatch, _FakeEnv())
    assert practice._open_cabinet("a") is None


def test_cabinet_lives_on_her_own_display(monkeypatch):
    """_open_cabinet asks for her screen and builds the window on it."""
    from worlds import arc_cabinet

    seen = {}

    class _Cabinet:
        """Records the display it was built on; stubs the surface."""

        def __init__(self, game_id, display=None):
            """Keep the display — that's the whole assertion."""
            seen["display"] = display

        @property
        def display(self):
            """Where the window lives."""
            return ":77"

        def render(self, frame):
            """Boards accepted."""

        def dispatch(self, action, data):
            """Echo arrivals."""

        def close(self):
            """Window gone."""

    monkeypatch.setattr(arc_cabinet, "ensure_arc_display", lambda: ":77")
    monkeypatch.setattr(arc_cabinet, "ArcCabinet", _Cabinet)
    practice = _practice(monkeypatch, _FakeEnv())
    assert practice._open_cabinet("a") is not None
    assert seen["display"] == ":77"


def test_eye_looks_at_the_cabinets_display(monkeypatch):
    """The camera's relay is pointed at the screen the cabinet lives on."""
    from worlds import arc_cabinet, arc_eye

    seen = {}

    class _Cabinet:
        """A minimal window on ':77' with a capture descriptor."""

        def __init__(self, game_id, display=None):
            """Lives where ensure_arc_display put it."""
            self._display = display

        @property
        def display(self):
            """Where the window lives."""
            return self._display

        def board_capture(self):
            """The eye's capture descriptor."""
            return (999, 384, 384)

        def render(self, frame):
            """Boards accepted."""

        def dispatch(self, action, data):
            """Echo arrivals."""
            return action, data

        def close(self):
            """Window gone."""

    class _Eye:
        """Records the display the relay was pointed at."""

        def __init__(self, capture, display=None):
            """Keep both — capture and display are the assertion."""
            seen["capture"] = capture
            seen["display"] = display

        def seq(self):
            """No frames yet — the mark is 0."""
            return 0

        def grab(self, after=None):
            """Unreadable — the engine frame serves."""
            return None

        def close(self):
            """Camera off."""

    monkeypatch.setattr(arc_cabinet, "ensure_arc_display", lambda: ":77")
    monkeypatch.setattr(arc_cabinet, "ArcCabinet", _Cabinet)
    monkeypatch.setattr(arc_eye, "ArcEye", _Eye)
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    practice._x11_eye = True
    practice.play("fake", max_steps=1)
    assert seen == {"capture": (999, 384, 384), "display": ":77"}


def test_suite_shares_one_cabinet_across_games(monkeypatch):
    """One sitting, one screen — she doesn't teleport between cabinets."""
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))
    practice._arcade.get_environments = lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")]
    practice._arcade.make = lambda *a, **k: _FakeEnv(win_after=1)
    cabinet = _FakeCabinet()
    opened = []
    monkeypatch.setattr(practice, "_open_cabinet", lambda gid: opened.append(gid) or cabinet)
    result = practice.play_all(max_steps=3)
    # Opened once — for the first game of the sitting.
    assert opened == ["a"]
    # Both games repainted the same glass, then it closed.
    assert cabinet.renders >= 2 and cabinet.closed
    # Each game's opening press physically landed on the reset control.
    assert cabinet.dispatched.count("RESET") == 2
    assert all(r.won for r in result.games)


def test_opening_reset_press_does_not_spend_decision_budget(monkeypatch):
    """The venue's opening RESET is protocol — a reset, not a move."""
    env = _FakeEnv(win_after=1)
    practice = _practice(monkeypatch, env)
    cabinet = _FakeCabinet()
    monkeypatch.setattr(practice, "_open_cabinet", lambda gid: cabinet)
    result = practice.play("fake", max_steps=1)
    assert result.won
    # One press on the glass, one env.reset() submission — and the
    # whole budget still went to her one real decision.
    assert cabinet.dispatched[0] == "RESET"
    assert result.resets == env.resets == 1
    assert result.steps == env.steps == 1


def test_post_gameover_reset_press_spends_a_step(monkeypatch):
    """A retry press is a real act — it costs budget like any move."""

    class _DeathEnv(_FakeEnv):
        """The world that kills her on the first decision."""

        def step(self, action, data=None, reasoning=None):
            """Every decision ends the episode."""
            self.steps += 1
            self.seen_actions.append(getattr(action, "name", str(action)))
            return _Frame(self.grid, state=_State.GAME_OVER)

    env = _DeathEnv(win_after=99)
    practice = _practice(monkeypatch, env)
    cabinet = _FakeCabinet()
    monkeypatch.setattr(practice, "_open_cabinet", lambda gid: cabinet)
    result = practice.play("fake", max_steps=2)
    # Step 1 died; the retry press was step 2; budget ended the run.
    assert result.state == "BUDGET"
    assert result.steps == 2
    assert result.resets == env.resets == 2
    assert cabinet.dispatched.count("RESET") == 2


def test_fumbled_press_spends_the_move_without_stepping(monkeypatch):
    """A press that never lands on the glass is a lost move — the
    world was never stepped, but the budget still paid for the act."""
    env = _FakeEnv(win_after=99)
    practice = _practice(monkeypatch, env)
    cabinet = _FakeCabinet()
    cabinet.arrival = None
    monkeypatch.setattr(practice, "_open_cabinet", lambda gid: cabinet)
    result = practice.play("fake", max_steps=2)
    assert result.state == "BUDGET"
    assert result.steps == 2
    assert env.steps == 0  # nothing physical arrived — the world never moved


def test_suite_retains_unplayed_games_when_health_stops_run(monkeypatch):
    """A wellness stop keeps the unplayed games listed — NOT_PLAYED,
    not silently dropped — and the partial suite reports no score."""
    mind = _StubMind()
    practice = _practice(monkeypatch, _FakeEnv(win_after=1), mind=mind)
    practice._arcade.get_environments = lambda: [
        SimpleNamespace(game_id=g) for g in ("a", "b", "c")
    ]
    end = mind.world_episode_end

    def episode_end(outcome, game):
        """After the first game she turns unwell."""
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
    """OPERATION_MODE=competition on the practice adapter is refused
    before the remote SDK is ever constructed."""

    def forbidden_sdk(**kwargs):
        """Tripwire — constructing the remote SDK fails the test."""
        raise AssertionError("practice tried to construct the remote SDK")

    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace(Arcade=forbidden_sdk))
    monkeypatch.setenv("OPERATION_MODE", "competition")
    with pytest.raises(RuntimeError, match="ArcOfficial"):
        ArcPractice(_StubMind(), environments_dir=tmp_path, recordings_dir=tmp_path)


def test_official_selects_competition_mode_without_network(monkeypatch, tmp_path):
    """ArcOfficial binds COMPETITION mode and refuses driver-style play."""

    class Mode(Enum):
        """The OperationMode surface the constructor binds."""

        OFFLINE = "offline"
        COMPETITION = "competition"

    configured = []
    monkeypatch.setenv("ARC_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setitem(
        sys.modules,
        "arc_agi",
        SimpleNamespace(
            OperationMode=Mode,
            Arcade=lambda **kwargs: configured.append(kwargs) or _FakeArcade(_FakeEnv()),
        ),
    )
    mind = _StubMind()
    official = ArcOfficial(
        mind,
        environments_dir=tmp_path,
        recordings_dir=tmp_path,
        card_path=tmp_path / "card.json",
    )
    assert configured[0]["operation_mode"] is Mode.COMPETITION
    # The card is never driven: no selected replays, no suite runner.
    with pytest.raises(RuntimeError, match="volition"):
        official.play("selected")
    with pytest.raises(RuntimeError, match="volition"):
        official.play_all()


def test_official_requires_explicit_access(monkeypatch):
    """No ARC_API_KEY, no official path — authorization is explicit."""
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ARC_API_KEY"):
        ArcOfficial(_StubMind())


def test_official_requires_game_mode(monkeypatch):
    """The official path also requires her to be engaged first."""
    monkeypatch.setitem(sys.modules, "arc_agi", SimpleNamespace())
    monkeypatch.setenv("ARC_API_KEY", "test-only-not-a-real-key")
    with pytest.raises(RuntimeError, match="game mode"):
        ArcOfficial(_StubMind(gaming=False))


def test_mind_evaluation_wraps_game_mode(monkeypatch):
    """evaluate_arc holds game mode around the suite and releases it."""
    from worlds.arc import ArcEvaluation

    import mind as mind_mod

    calls = []

    class Subject:
        """Engaged flag with call recording."""

        is_gaming = False

        def enter_game_mode(self):
            """Record the engagement."""
            self.is_gaming = True
            calls.append("enter")

        def exit_game_mode(self):
            """Record the release."""
            self.is_gaming = False
            calls.append("exit")

    class Practice:
        """Captures the suite call the wrapper makes."""

        def __init__(self, mind, step_delay_s=0.0, recovery_wait_s=0.0):
            """Assert engagement at construction."""
            assert mind.is_gaming

        def play_all(self, max_steps):
            """Record the budget, return an empty evaluation."""
            calls.append(max_steps)
            return ArcEvaluation(mode="offline", games=[])

    monkeypatch.setattr("worlds.arc.ArcPractice", Practice)
    subject = Subject()
    result = mind_mod.Mind.evaluate_arc(subject, max_steps=12)
    assert result.mode == "offline"
    assert calls == ["enter", 12, "exit"]
    assert not subject.is_gaming


def test_arc_command_selects_full_suite():
    """``/arc --all`` drives evaluate_arc; the report shows the score."""
    from worlds.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    calls = []
    subject = SimpleNamespace(
        evaluate_arc=lambda **kwargs: (
            calls.append(kwargs)
            or ArcEvaluation(
                mode="offline",
                games=[ArcResult(game_id="g", state="WIN", levels_completed=1, win_levels=1)],
                score=75.0,
            )
        )
    )
    report = _cmd_arc(subject, "--all --steps 12")
    assert calls == [
        {
            "max_steps": 12,
            "fresh": False,
            "step_delay_s": 0.0,
            "recovery_wait_s": 120.0,
        }
    ]
    assert "75.00%" in report
    assert "levels 1/1" in report


def test_arc_command_passes_pacing_flags():
    """``--delay``/``--recover`` reach the evaluation; waits are reported."""
    from worlds.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    calls = []
    subject = SimpleNamespace(
        evaluate_arc=lambda **kwargs: (
            calls.append(kwargs)
            or ArcEvaluation(
                mode="offline",
                games=[
                    ArcResult(
                        game_id="g", state="WIN", levels_completed=1, win_levels=1, recovery_waits=2
                    )
                ],
                score=75.0,
            )
        )
    )
    report = _cmd_arc(subject, "--all --delay 0.5 --recover 90")
    assert calls[0]["step_delay_s"] == 0.5
    assert calls[0]["recovery_wait_s"] == 90.0
    assert "recovery-waits=2" in report


def test_arc_command_passes_fresh_scope():
    """``--fresh`` reaches play_arc and evaluate_arc on both paths."""
    from worlds.arc import ArcEvaluation

    from genesis_cli import _cmd_arc

    seen = []
    subject = SimpleNamespace(
        play_arc=lambda games, **kw: (
            seen.append((games, kw)) or [ArcResult(game_id=g) for g in games]
        )
    )
    _cmd_arc(subject, "g --fresh")
    assert seen == [
        (
            ["g"],
            {"max_steps": 200, "fresh": True, "step_delay_s": 0.0, "recovery_wait_s": 120.0},
        )
    ]

    seen.clear()
    subject = SimpleNamespace(
        evaluate_arc=lambda **kw: seen.append(kw) or ArcEvaluation(mode="offline", games=[])
    )
    _cmd_arc(subject, "--all --fresh --steps 7")
    assert seen == [
        {
            "max_steps": 7,
            "fresh": True,
            "step_delay_s": 0.0,
            "recovery_wait_s": 120.0,
        }
    ]


@pytest.mark.parametrize(
    "args",
    [
        "--steps",
        "g --steps nope",
        "g --steps 0",
        "--competition g",
        "--all --competition",
    ],
)
def test_arc_command_rejects_invalid_requests_without_play(args):
    """Malformed /arc invocations print usage and never touch the mind."""
    from genesis_cli import _cmd_arc

    assert "Usage:" in _cmd_arc(SimpleNamespace(), args)


@pytest.mark.parametrize(
    "desc",
    [
        {"action": 6},
        {"action": 6, "x": -1, "y": 0},
        {"action": 6, "x": 0, "y": 64},
        {"action": 6, "x": 0.5, "y": 1},
        {"action": 6, "x": True, "y": 1},
        {"action": 1.5},
    ],
)
def test_invalid_action_payload_is_rejected(desc):
    """Malformed decision descriptors raise before the world sees them."""
    with pytest.raises(ValueError):
        ArcPractice._to_step(desc)


def test_unavailable_action_is_not_sent(monkeypatch):
    """A pick outside the whitelist is a contract breach — nothing sent."""
    subject = _StubMind()
    subject.world_step = lambda *args, **kwargs: {"action": 7}
    env = _FakeEnv()
    practice = _practice(monkeypatch, env, mind=subject)
    with pytest.raises(ValueError, match="available"):
        practice.play("fake", max_steps=1)
    assert env.steps == 0
    assert practice._arcade.closed == ["card-1"]


def test_scorecard_failure_is_visible(monkeypatch):
    """A card that won't close is SCORECARD_ERROR, never a silent zero."""
    practice = _practice(monkeypatch, _FakeEnv(win_after=1))

    def failed_close(card):
        """A scorecard close that always dies."""
        raise ConnectionError("scorecard unavailable")

    practice._arcade.close_scorecard = failed_close
    result = practice.play("fake", max_steps=1)
    assert result.state == "SCORECARD_ERROR"
    assert result.score is None
    assert result.reasons == ("scorecard closure failed",)


# ─── The official card ────────────────────────────────────


class _AnnouncingMind(_StubMind):
    """A stub mind that also records announcements and game-mode
    entries — the surfaces the official card drives."""

    def __init__(self, **kw):
        """Stub mind plus announcement/game-mode ledgers."""
        super().__init__(**kw)
        self.announcements: list[str] = []
        self.game_mode_entries = 0

    def world_announce(self, text, *, topics=None):
        """Record the venue brief she was told."""
        self.announcements.append((text, topics))
        return True

    def enter_game_mode(self):
        """Count the engagement and mark her gaming."""
        self.game_mode_entries += 1
        self.is_gaming = True

    def exit_game_mode(self):
        """Release engagement."""
        self.is_gaming = False


def _official(monkeypatch, tmp_path, *, mind=None, envs=("a", "b")) -> ArcOfficial:
    """Build an ArcOfficial wired to fakes, constructor skipped —
    a tmp card path, fake arcade, no SDK import."""
    fake = _FakeArcade(_FakeEnv(win_after=1))
    monkeypatch.setattr(
        fake,
        "get_environments",
        lambda: [SimpleNamespace(game_id=g) for g in envs],
        raising=False,
    )
    monkeypatch.setattr(fake, "cards", [], raising=False)

    def _make_card(**kw):
        """Open the one fake card, recording the open."""
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
    p._x11_hands = True
    p._cabinet = None
    p._card_path = tmp_path / "official_card.json"
    p._recordings_dir = tmp_path / "recordings"
    return p


def test_official_card_opens_persists_and_records_attempt(monkeypatch, tmp_path):
    """The card opens on disk once; an attempt marks the game played
    and the brief reaches her first."""
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
    """A restarted process resumes the same card — played games stay
    burned, and a full card closes as complete."""
    official = _official(monkeypatch, tmp_path)
    official.open_or_resume()
    official.attempt("a", max_steps=2)

    resumed = _official(monkeypatch, tmp_path)
    card = resumed.open_or_resume()  # same card, no second create
    assert card["scorecard_id"] == "card-1"
    assert resumed._arcade.cards == []
    assert resumed.attempt("a") is None  # burned — no retry
    second = resumed.attempt("b", max_steps=2)
    assert second is not None and second.state == "WIN"
    # Both attempted → next call closes the card as complete.
    assert resumed.attempt("a") is None
    assert resumed.card_state()["closed"] is True
    assert resumed.card_state()["close_reason"].startswith("complete")


def test_official_unwell_does_not_burn_the_attempt(monkeypatch, tmp_path):
    """She never entered — the game stays pending and no brief was spent."""
    mind = _AnnouncingMind(ready=False)
    official = _official(monkeypatch, tmp_path, mind=mind)
    official.open_or_resume()
    result = official.attempt("a")
    assert result is not None and result.state == "UNWELL"
    game = next(g for g in official.card_state()["games"] if g["game_id"] == "a")
    assert game["status"] == "pending"  # she never entered
    assert not mind.announcements  # no brief before the gate


def test_official_closed_card_is_sealed_for_the_window(monkeypatch, tmp_path):
    """A closed card inside its window can't be replaced or replayed."""
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    official._seal(card, "idle — test")
    assert official.attempt("a") is None
    with pytest.raises(RuntimeError, match="closed"):
        official.open_or_resume()


def test_official_expired_window_opens_a_new_card(monkeypatch, tmp_path):
    """Past the 24 h window the old card archives and a new one opens."""
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    card["expires_at"] = 1.0  # long past
    official._save_card(card)
    fresh = _official(monkeypatch, tmp_path)
    new = fresh.open_or_resume()
    assert new["scorecard_id"] == "card-1"  # fake returns the same id
    assert fresh._arcade.cards == ["card-1"]  # a second open happened
    assert not (tmp_path / "official_card.json").read_text() == json.dumps(card)


def test_official_shelf_shows_one_standing_challenge(monkeypatch, tmp_path):
    """The shelf offers the challenge as one commitment — live while
    the card is open, gone when it's spent, sealed, or unkeyed."""
    from worlds.play import _ARC_CHALLENGE, GamePlay

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
        "scorecard_id": "card-1",
        "opened_at": time.time(),
        "expires_at": time.time() + 3600,
        "closed": False,
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
    """The standing brief is heard once per window — an open card
    re-briefs once with the state of the commitment."""
    from worlds.play import GamePlay

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
        "scorecard_id": "card-9",
        "opened_at": time.time(),
        "expires_at": time.time() + 3600,
        "closed": False,
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
        """The venue boundary faked at the adapter seam — the shared
        ``card`` dict is the card bookkeeping."""

        IDLE_LIMIT_S = 900.0

        def __init__(self, *args, **kwargs):
            """Bind to the shared card dict."""
            self._card = card
            self.calls: list[str] = []

        def open_or_resume(self):
            """Hand back the shared card."""
            return self._card

        def card_state(self):
            """Inspect the shared card."""
            return self._card

        def attempt(self, game_id, max_steps=200):
            """Burn the game as attempted+WIN on the shared card."""
            self.calls.append(game_id)
            game = next(g for g in self._card["games"] if g["game_id"] == game_id)
            game["status"] = "attempted"
            game["state"] = "WIN"
            self._card["last_action_at"] = time.time()
            return ArcResult(
                game_id=game_id,
                state="WIN",
                levels_completed=2,
                win_levels=2,
                steps=3,
            )

        def attempt_next(self, max_steps=200):
            """Seal the shared card as complete."""
            self._card["closed"] = True
            self._card["close_reason"] = "complete — every game attempted"
            return None

    return _Adapter


def test_sitting_opens_card_and_plays_every_pending_game(monkeypatch, tmp_path):
    """Her pick of the challenge opens the card; one firing, one sitting."""
    from worlds.play import _ARC_CHALLENGE, GamePlay

    class Mode(Enum):
        """The OperationMode surface the constructor binds."""

        OFFLINE = "offline"
        COMPETITION = "competition"

    fake = _FakeArcade(_FakeEnv(win_after=1))
    monkeypatch.setattr(
        fake,
        "get_environments",
        lambda: [SimpleNamespace(game_id=g) for g in ("a", "b")],
        raising=False,
    )
    monkeypatch.setattr(fake, "cards", [], raising=False)

    def _make_card(**kw):
        """Open the one fake card, recording the open."""
        fake.cards.append("card-1")
        return "card-1"

    monkeypatch.setattr(fake, "create_scorecard", _make_card, raising=False)
    monkeypatch.setenv("ARC_API_KEY", "test-only")
    monkeypatch.setitem(
        sys.modules,
        "arc_agi",
        SimpleNamespace(
            OperationMode=Mode,
            Arcade=lambda **kw: fake,
        ),
    )
    mind = _AnnouncingMind(gaming=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(
        None,
        task={
            "name": _ARC_CHALLENGE,
            "family": "arc_official",
            "hint": "the official challenge",
        },
    )
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
    from worlds.play import _ARC_CHALLENGE, GamePlay

    mind = _AnnouncingMind(gaming=False, ready=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(
        None,
        task={
            "name": _ARC_CHALLENGE,
            "family": "arc_official",
            "hint": "",
        },
    )
    assert result is None
    assert not (tmp_path / "arc" / "official_card.json").exists()
    assert not mind.is_gaming and mind.game_mode_entries == 0


def test_sitting_waits_out_an_unwell_gate_inside_idle_budget(monkeypatch, tmp_path):
    """A mid-card wellness gate pauses her; the game is not burned."""
    from worlds.play import _ARC_CHALLENGE, GamePlay

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    card = {
        "scorecard_id": "card-1",
        "opened_at": time.time(),
        "expires_at": time.time() + 3600,
        "closed": False,
        "last_action_at": time.time(),
        "games": [
            {"game_id": "a", "status": "pending"},
            {"game_id": "b", "status": "pending"},
        ],
    }
    (tmp_path / "arc").mkdir()
    (tmp_path / "arc" / "official_card.json").write_text(json.dumps(card))
    adapter_cls = _fake_card_adapter(card)
    monkeypatch.setattr("worlds.arc.ArcOfficial", adapter_cls)
    mind = _AnnouncingMind(gaming=False)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind

    # The second game she reaches for reports UNWELL once — the wait
    # happens inside the idle budget, then she plays it for real.
    original_attempt = adapter_cls.attempt

    def attempt_once_unwell(self, game_id, max_steps=200):
        """The second reach reports UNWELL once, then plays for real."""
        if self.calls and not getattr(self, "_unwell_spent", False):
            self._unwell_spent = True
            return ArcResult(game_id=game_id, state="UNWELL", steps=0)
        return original_attempt(self, game_id, max_steps)

    adapter_cls.attempt = attempt_once_unwell
    result = shelf.attempt(
        None,
        task={
            "name": _ARC_CHALLENGE,
            "family": "arc_official",
            "hint": "",
        },
    )
    assert result is not None
    assert result.details["games"] == 2 and result.details["won"] == 2
    assert shelf.attempts["arcagi-a"] == shelf.attempts["arcagi-b"] == 1


def test_sitting_ends_when_she_leaves_the_arcade(monkeypatch, tmp_path):
    """Dropping game mode ends the sitting honestly — the card keeps
    what she played and stays open on the venue's clock."""
    from worlds.play import _ARC_CHALLENGE, GamePlay

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    card = {
        "scorecard_id": "card-1",
        "opened_at": time.time(),
        "expires_at": time.time() + 3600,
        "closed": False,
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
        """After one attempt she drops game mode — left the arcade."""
        result = original_attempt(self, game_id, max_steps)
        mind.is_gaming = False  # /endgame — she left
        return result

    adapter_cls.attempt = walk_away
    monkeypatch.setattr("worlds.arc.ArcOfficial", adapter_cls)
    shelf = GamePlay(tmp_path)
    shelf.mind = mind
    result = shelf.attempt(
        None,
        task={
            "name": _ARC_CHALLENGE,
            "family": "arc_official",
            "hint": "",
        },
    )
    assert result is not None
    assert result.details["games"] == 1
    assert result.details["card_closed"] is False
    pending = [g for g in card["games"] if g["status"] == "pending"]
    assert len(pending) == 1


def test_official_card_recovers_orphaned_attempt_from_recording(monkeypatch, tmp_path):
    """A stop mid-game leaves status=attempted, steps=0 — the recording
    holds the honest partial and load must reconcile it."""
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    game = next(g for g in card["games"] if g["game_id"] == "a")
    game["status"] = "attempted"
    game["attempted_at"] = card["opened_at"] + 10
    official._save_card(card)

    rec_dir = tmp_path / "recordings" / card["scorecard_id"]
    rec_dir.mkdir(parents=True)
    (rec_dir / "a-deadbeef.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "data": {
                        "game_id": "a",
                        "state": "NOT_FINISHED",
                        "levels_completed": 0,
                        "win_levels": 3,
                        "action_input": {"id": aid},
                    }
                }
            )
            for aid in ("RESET", "ACTION1", "ACTION3")
        )
        + "\n"
    )

    loaded = official.card_state()
    g = next(x for x in loaded["games"] if x["game_id"] == "a")
    assert g["steps"] == 2 and g["resets"] == 1
    assert g["state"] == "NOT_FINISHED"
    assert g["win_levels"] == 3
    assert any("recovered from recording" in r for r in g["reasons"])


def test_official_card_expiry_clamped_to_venue_window(monkeypatch, tmp_path):
    """Cards written while the mirror said 3 days must not outlive the
    venue's real 24-hour cap."""
    official = _official(monkeypatch, tmp_path)
    card = official.open_or_resume()
    card["expires_at"] = card["opened_at"] + 3 * 24 * 3600
    official._save_card(card)
    loaded = official.card_state()
    assert loaded["expires_at"] == card["opened_at"] + ArcOfficial.WINDOW_S


def test_swallowed_step_crash_dumps_failing_pair(monkeypatch, tmp_path):
    """The wrapper turns an engine crash into a silent None frame —
    the trap still lands the pair it died on and the pair ring."""
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _CrashEnv(_FakeEnv):
        """Steps 1-2 fine, step 3 answers silence — a swallowed crash."""

        def step(self, action, data=None, reasoning=None):
            """Record the action; go silent on the third."""
            self.steps += 1
            self.seen_actions.append(getattr(action, "name", str(action)))
            if self.steps < 3:
                return _Frame(self.grid)
            return None

    env = _CrashEnv()
    practice = _practice(monkeypatch, env)
    practice._crash_dir = tmp_path / "crashes"
    result = practice.play("fake", max_steps=9)
    assert result.state == "NO_OBSERVATION" and result.steps == 3
    dumps = list((tmp_path / "crashes").glob("*.json"))
    assert len(dumps) == 1
    dump = json.loads(dumps[0].read_text())
    assert dump["game_id"] == "fake" and dump["step"] == 3
    assert dump["error"] == "step produced no frame"
    assert dump["submitted"]["action"] in env.seen_actions
    assert dump["died_on"]["layers"] == [env.grid]
    # The ring kept the opening reset press and the two successful
    # pairs that preceded the crash.
    assert [p["step"] for p in dump["recent_pairs"]] == [0, 1, 2]
    assert all(p["grid"] == env.grid and p["delta"] == [] for p in dump["recent_pairs"])
    assert any("dumped" in r for r in result.reasons)


def test_step_exception_dumps_pair_then_propagates(monkeypatch, tmp_path):
    """An un-swallowed failure keeps propagating — but not silently."""
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _RaiseEnv(_FakeEnv):
        """Every step raises — an un-swallowed engine failure."""

        def step(self, action, data=None, reasoning=None):
            """Record the action then die."""
            self.steps += 1
            self.seen_actions.append(getattr(action, "name", str(action)))
            raise IndexError("sprite render out of bounds")

    env = _RaiseEnv()
    practice = _practice(monkeypatch, env)
    practice._crash_dir = tmp_path / "crashes"
    with pytest.raises(IndexError, match="sprite render"):
        practice.play("fake", max_steps=3)
    dumps = list((tmp_path / "crashes").glob("*.json"))
    assert len(dumps) == 1
    dump = json.loads(dumps[0].read_text())
    assert "IndexError" in dump["error"] and "sprite render" in dump["error"]
    assert "IndexError" in dump["traceback"]
    assert dump["submitted"]["action"] in env.seen_actions
    # The ring holds only the opening reset press — the first real
    # step is the one that died.
    assert [p["step"] for p in dump["recent_pairs"]] == [0]


def test_pair_ring_records_raw_delta(monkeypatch, tmp_path):
    """Each trapped pair keeps the cell-level diff its action caused."""
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _MutatingEnv(_FakeEnv):
        """First step mutates two cells, second goes silent — so the
        pair ring's delta has real content to check."""

        def step(self, action, data=None, reasoning=None):
            """Mutate once, then answer silence."""
            self.steps += 1
            self.seen_actions.append(getattr(action, "name", str(action)))
            if self.steps == 1:
                new = [row[:] for row in self.grid]
                new[0][0] = (new[0][0] + 1) % 16
                new[2][3] = 9
                return _Frame(new)
            return None

    env = _MutatingEnv()
    practice = _practice(monkeypatch, env)
    practice._crash_dir = tmp_path / "crashes"
    result = practice.play("fake", max_steps=5)
    assert result.state == "NO_OBSERVATION"
    dump = json.loads(next(iter((tmp_path / "crashes").glob("*.json"))).read_text())
    # Pair 0 is the opening reset press; pair 1 is the mutating step
    # whose raw delta the ring must keep.
    pair = dump["recent_pairs"][1]
    expected = [
        [0, 0, env.grid[0][0], (env.grid[0][0] + 1) % 16],
        [2, 3, env.grid[2][3], 9],
    ]
    assert sorted(pair["delta"]) == expected


def test_reset_returning_none_dumps_before_no_observation(monkeypatch, tmp_path):
    """A dead recovery reset is the same swallowed crash — dump it."""
    pytest.importorskip("arcengine", reason="arcengine not installed")

    class _ResetCrashEnv(_FakeEnv):
        """First reset returns a dead frame; the recovery reset
        itself goes silent — a crash on the recovery path."""

        def reset(self):
            """GAME_OVER once, then silence."""
            self.resets += 1
            if self.resets == 1:
                return _Frame(self.grid, state=_State.GAME_OVER)
            return None

    practice = _practice(monkeypatch, _ResetCrashEnv())
    practice._crash_dir = tmp_path / "crashes"
    result = practice.play("fake", max_steps=9)
    assert result.state == "NO_OBSERVATION"
    dump = json.loads(next(iter((tmp_path / "crashes").glob("*.json"))).read_text())
    assert dump["error"] == "step produced no frame" and dump["submitted"] is None

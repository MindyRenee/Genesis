"""Tests for the bounded ARC-AGI-3 episode loop."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from parietal_lobe import arc_play
from parietal_lobe.arc_ascii import AsciiGameRecord, decode_game_record
from parietal_lobe.arc_play import ARC3Decision, run_arc3_episode
from parietal_lobe.arc_policy import ARC3ExplorationPolicy
from parietal_lobe.arc_sources import ARCSourceError


def _obs(grid, state="NOT_FINISHED", levels=0, win_levels=3):
    """Build a toolkit-shaped observation namespace."""
    return SimpleNamespace(
        frame=[grid], state=state, levels_completed=levels, win_levels=win_levels,
        guid="session", available_actions=[1, 2],
    )


class ScriptedEnv:
    """Fake environment returning a scripted observation per step."""

    def __init__(self, script, legal=("ACTION1", "ACTION2")):
        self.script = list(script)
        self.legal = list(legal)
        self.calls = []

    def step(self, action, *, data=None, reasoning=None):
        """Record the call and return the next scripted observation."""
        self.calls.append((action, data))
        return self.script.pop(0)


@pytest.fixture
def scripted(monkeypatch):
    """Route the loop's toolkit calls to a ScriptedEnv."""
    monkeypatch.setattr(arc_play, "arc3_legal_actions", lambda env: list(env.legal))
    monkeypatch.setattr(
        arc_play,
        "step_arc3_game",
        lambda env, action, data=None, reasoning=None: env.step(
            action, data=data, reasoning=reasoning
        ),
    )
    return ScriptedEnv


def _first(observation, legal, history):
    """Policy that always picks the first legal action."""
    return ARC3Decision(legal[0])


def test_episode_runner_records_board_and_stops_on_win(scripted) -> None:
    """Record decodable transitions that carry the real board, and stop on WIN."""
    env = scripted([_obs([[2]], "WIN", levels=3)])
    episode = run_arc3_episode(
        env, game_id="g", initial_observation=_obs([[1]]), choose_action=_first
    )
    assert episode.terminal_state == "WIN"
    assert episode.stopped_reason == "terminal_state"
    assert episode.levels_completed == 3 and episode.win_levels == 3
    record = decode_game_record(AsciiGameRecord(tuple(episode.transitions[0]["ascii_codes"])))
    assert record["observation"]["grid"] == [[1]]
    assert record["next_observation"]["grid"] == [[2]]
    assert episode.transitions[0]["reward"] == 4.0  # three levels plus the win


def test_episode_runner_rejects_illegal_policy_action(scripted) -> None:
    """Never execute an action outside the current legal set."""
    env = scripted([])
    with pytest.raises(ARCSourceError):
        run_arc3_episode(
            env, game_id="g", initial_observation=_obs([[1]]),
            choose_action=lambda o, legal, h: ARC3Decision("ACTION9"),
        )
    assert env.calls == []


def test_episode_runner_requires_positive_step_budget() -> None:
    """Reject zero, negative, and boolean step budgets."""
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            run_arc3_episode(
                object(), game_id="g", initial_observation={}, choose_action=_first,
                max_steps=bad,
            )


def test_game_over_is_recovered_with_reset_and_play_continues(scripted) -> None:
    """Issue RESET after GAME_OVER, count it, and keep playing."""
    env = scripted([
        _obs([[3]], "GAME_OVER", levels=1),
        _obs([[1]], levels=1),
        _obs([[4]], "WIN", levels=3),
    ])
    episode = run_arc3_episode(
        env, game_id="g", initial_observation=_obs([[1]]), choose_action=_first
    )
    assert [call[0] for call in env.calls] == ["ACTION1", "RESET", "ACTION1"]
    assert episode.resets == 1
    assert episode.actions_taken == 3
    assert episode.terminal_state == "WIN"
    assert episode.transitions[0]["reward"] == 0.0  # +1 level, -1 game over


def test_game_over_stops_when_reset_disabled_or_exhausted(scripted) -> None:
    """Honour reset_on_game_over=False and max_resets."""
    episode = run_arc3_episode(
        scripted([_obs([[3]], "GAME_OVER")]), game_id="g", initial_observation=_obs([[1]]),
        choose_action=_first, reset_on_game_over=False,
    )
    assert episode.terminal_state == "GAME_OVER"
    assert episode.stopped_reason == "terminal_state"
    episode = run_arc3_episode(
        scripted([_obs([[3]], "GAME_OVER")]), game_id="g", initial_observation=_obs([[1]]),
        choose_action=_first, max_resets=0,
    )
    assert episode.stopped_reason == "reset_limit"


def test_episode_runner_routes_board_percepts_to_outer_world(scripted) -> None:
    """Send the board (not the raw object) to OuterWorld and actions to it_acted."""
    events = []

    class World:
        """Fake OuterWorld recording calls."""

        def perceive(self, content, **kwargs):
            """Record a perception."""
            events.append(("perceive", kwargs))
            return object()

        def it_acted(self, description, *, detail=""):
            """Record an action."""
            events.append(("acted", description))

    env = scripted([_obs([[2, 0]], "WIN")])
    run_arc3_episode(
        env, game_id="g", initial_observation=_obs([[1, 0]]), choose_action=_first,
        outer_world=World(),
    )
    assert events[0][1]["percept"] == {"grid": [[1, 0]]}
    assert events[1] == ("acted", "ARC-AGI-3 action ACTION1 in game g")
    assert events[2][1]["percept"] == {"grid": [[2, 0]]}
    assert events[2][1]["metadata"]["state_after"] == "WIN"


def test_episode_runner_stops_when_outer_world_cannot_perceive(scripted) -> None:
    """Stop before acting when the world is asleep."""
    world = SimpleNamespace(perceive=lambda *a, **k: None, it_acted=lambda *a, **k: None)
    env = scripted([])
    episode = run_arc3_episode(
        env, game_id="g", initial_observation=_obs([[1]]), choose_action=_first,
        outer_world=world,
    )
    assert episode.stopped_reason == "outer_world_unavailable"
    assert env.calls == []


def test_episode_runner_feeds_level_reward_back_to_bound_policy(scripted) -> None:
    """Pass level-progress reward to the bound policy's observe_transition."""
    policy = ARC3ExplorationPolicy()
    seen = []
    original = policy.observe_transition

    def spy(*args, **kwargs):
        seen.append(kwargs["reward"])
        original(*args, **kwargs)

    object.__setattr__(policy, "observe_transition", spy)
    env = scripted([_obs([[2]], levels=1), _obs([[5]], "WIN", levels=3)])
    episode = run_arc3_episode(
        env, game_id="g", initial_observation=_obs([[1]]),
        choose_action=policy.choose_action,
    )
    assert seen == [1.0, 3.0]
    assert episode.levels_completed == 3


def test_episode_runner_recognizes_mapping_shaped_terminal_state(scripted) -> None:
    """Treat {"state": {"name": "WIN"}} as terminal."""
    episode = run_arc3_episode(
        scripted([]), game_id="g",
        initial_observation={"grid": [[1]], "state": {"name": "WIN"}},
        choose_action=_first,
    )
    assert episode.terminal_state == "WIN"
    assert episode.actions_taken == 0


def test_policy_clicks_objects_with_coordinates(scripted) -> None:
    """ACTION6 decisions carry in-range coordinates on a visible object."""
    grid = [[0] * 8 for _ in range(8)]
    grid[5][2] = 7
    env = scripted([_obs(grid, "WIN")], legal=("ACTION6",))
    episode = run_arc3_episode(
        env, game_id="g", initial_observation=_obs(grid),
        choose_action=ARC3ExplorationPolicy().choose_action,
    )
    assert env.calls == [("ACTION6", {"x": 2, "y": 5})]
    assert episode.transitions[0]["action_key"] == "ACTION6@2,5"

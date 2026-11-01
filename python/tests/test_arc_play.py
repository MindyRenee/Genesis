from __future__ import annotations

from types import SimpleNamespace

import pytest

from parietal_lobe import arc_play
from parietal_lobe.arc_play import ARC3Decision, run_arc3_episode
from parietal_lobe.arc_sources import ARCSourceError


class FakeEnvironment:
    action_space = ["ACTION1", "ACTION2"]


def test_episode_runner_observes_feedback_and_stops_on_win(monkeypatch) -> None:
    calls = []

    def fake_step(environment, action_name, *, data=None, reasoning=None):
        calls.append((action_name, data, reasoning))
        return SimpleNamespace(state=SimpleNamespace(name="WIN"))

    monkeypatch.setattr(arc_play, "step_arc3_game", fake_step)
    initial = SimpleNamespace(state=SimpleNamespace(name="NOT_FINISHED"))
    episode = run_arc3_episode(
        FakeEnvironment(),
        game_id="fake-game",
        initial_observation=initial,
        choose_action=lambda observation, legal, history: ARC3Decision(
            "ACTION1", reasoning={"hypothesis": "probe"}
        ),
        max_steps=5,
    )

    assert calls == [("ACTION1", None, {"hypothesis": "probe"})]
    assert episode.terminal_state == "WIN"
    assert episode.stopped_reason == "terminal_state"
    assert len(episode.transitions) == 1


def test_episode_runner_rejects_illegal_policy_action(monkeypatch) -> None:
    monkeypatch.setattr(
        arc_play,
        "step_arc3_game",
        lambda *args, **kwargs: pytest.fail("must not execute an illegal action"),
    )
    with pytest.raises(ARCSourceError, match="illegal action"):
        run_arc3_episode(
            FakeEnvironment(),
            game_id="fake-game",
            initial_observation=SimpleNamespace(
                state=SimpleNamespace(name="NOT_FINISHED")
            ),
            choose_action=lambda observation, legal, history: ARC3Decision("ACTION7"),
            max_steps=2,
        )


def test_episode_runner_requires_positive_step_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_arc3_episode(
            FakeEnvironment(),
            game_id="fake-game",
            initial_observation=SimpleNamespace(
                state=SimpleNamespace(name="NOT_FINISHED")
            ),
            choose_action=lambda observation, legal, history: ARC3Decision("ACTION1"),
            max_steps=0,
        )

"""Tests for bounded ARC-AGI-3 episodes and OuterWorld event routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parietal_lobe import arc_play
from parietal_lobe.arc_play import ARC3Decision, run_arc3_episode
from parietal_lobe.arc_sources import ARCSourceError


class FakeEnvironment:
    """Minimal environment stub exposing two legal action names."""

    action_space = ["ACTION1", "ACTION2"]


def test_episode_runner_observes_feedback_and_stops_on_win(monkeypatch) -> None:
    """Record the executed action and stop when the environment reports WIN."""
    calls = []

    def fake_step(environment, action_name, *, data=None, reasoning=None):
        """Return a terminal observation while recording the step arguments."""
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
    assert len(episode.transitions[0]["state_signature"]) == 64
    assert len(episode.transitions[0]["next_state_signature"]) == 64
    assert episode.transitions[0]["state_signature"] != episode.transitions[0]["next_state_signature"]


def test_episode_runner_rejects_illegal_policy_action(monkeypatch) -> None:
    """Reject policy output outside the environment's legal action set."""
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
    """Reject zero or negative episode step budgets before stepping."""
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



def test_episode_runner_routes_observations_and_actions_to_outer_world(monkeypatch) -> None:
    """Send structured percepts and action outcomes through OuterWorld APIs."""
    events = []

    class FakeOuterWorld:
        """Capture calls made to the OuterWorld event boundary."""
        def perceive(self, content, **kwargs):
            """Capture an inbound percept and its structured metadata."""
            events.append(("perceive", content, kwargs))
            return object()

        def it_acted(self, description, *, detail=""):
            """Capture an outbound action and its transition detail."""
            events.append(("act", description, detail))
            return object()

    observations = [
        SimpleNamespace(state=SimpleNamespace(name="PLAYING"), grid=[[1, 0]]),
        SimpleNamespace(state=SimpleNamespace(name="WIN"), grid=[[1, 1]]),
    ]

    def fake_step(environment, action_name, *, data=None, reasoning=None):
        """Return the preconfigured winning observation."""
        return observations[1]

    monkeypatch.setattr(arc_play, "step_arc3_game", fake_step)
    episode = run_arc3_episode(
        FakeEnvironment(),
        game_id="grid-game",
        initial_observation=observations[0],
        choose_action=lambda observation, legal, history: ARC3Decision(
            "ACTION1", reasoning={"prediction": "cell changes"}
        ),
        max_steps=3,
        outer_world=FakeOuterWorld(),
    )

    assert episode.terminal_state == "WIN"
    assert [event[0] for event in events] == ["perceive", "act", "perceive"]
    assert events[0][2]["percept"] == {"grid": [[1, 0]]}
    assert events[0][2]["source"] == "arc3:grid-game"
    assert events[0][2]["metadata"]["phase"] == "initial"
    assert "ACTION1" in events[1][1]
    assert "state_after=WIN" in events[1][2]
    # The terminal grid must be delivered before the episode returns.
    assert events[2][2]["percept"] == {"grid": [[1, 1]]}
    assert events[2][2]["metadata"]["phase"] == "outcome"
    assert events[2][2]["metadata"]["action"] == "ACTION1"
    assert events[2][2]["metadata"]["state_after"] == "WIN"


def test_episode_runner_stops_when_outer_world_cannot_perceive() -> None:
    """Do not take an action if the initial observation was sleep-gated."""

    class SleepingOuterWorld:
        """Represent an OuterWorld that rejects perception while asleep."""

        def perceive(self, content, **kwargs):
            """Signal that no perception event was recorded."""
            return None

    episode = run_arc3_episode(
        FakeEnvironment(),
        game_id="sleeping-game",
        initial_observation=SimpleNamespace(
            state=SimpleNamespace(name="PLAYING")
        ),
        choose_action=lambda observation, legal, history: pytest.fail(
            "policy must not run without a recorded initial perception"
        ),
        max_steps=2,
        outer_world=SleepingOuterWorld(),
    )

    assert episode.transitions == []
    assert episode.stopped_reason == "outer_world_unavailable"




def test_episode_runner_feeds_terminal_outcome_back_to_bound_policy(monkeypatch) -> None:
    """The episode loop reports explicit terminal feedback to Genesis' policy."""
    from parietal_lobe.arc_policy import ARC3ExplorationPolicy, observation_signature

    monkeypatch.setattr(
        arc_play,
        "step_arc3_game",
        lambda *args, **kwargs: SimpleNamespace(
            state=SimpleNamespace(name="WIN"), grid=[[1]]
        ),
    )
    initial = SimpleNamespace(
        state=SimpleNamespace(name="PLAYING"), grid=[[0]]
    )
    policy = ARC3ExplorationPolicy()
    episode = run_arc3_episode(
        FakeEnvironment(),
        game_id="feedback-game",
        initial_observation=initial,
        choose_action=policy.choose_action,
        max_steps=1,
    )

    key = (observation_signature(initial), episode.transitions[0]["action"])
    assert episode.terminal_state == "WIN"
    assert policy._outcomes[key] == (1.0, 1)

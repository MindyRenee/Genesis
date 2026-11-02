"""Tests for the bounded ARC-AGI-3 exploration policy."""
from __future__ import annotations

from types import SimpleNamespace

from parietal_lobe.arc_policy import ARC3ExplorationPolicy, observation_signature


def test_policy_selects_only_legal_actions_and_explains_scores() -> None:
    """Choose a legal control and label its score as exploration, not confidence."""
    policy = ARC3ExplorationPolicy()
    observation = SimpleNamespace(state=SimpleNamespace(name="PLAYING"), grid=[[0, 1]])
    decision = policy.choose_action(observation, ["ACTION1", "ACTION2"], [])

    assert decision.action in {"ACTION1", "ACTION2"}
    assert decision.reasoning["policy"] == "graph_frontier_exploration"
    assert decision.reasoning["score_semantics"] == (
        "exploration_priority_plus_observed_reward_not_success_probability"
    )
    assert decision.reasoning["candidate_count"] == 2


def test_policy_uses_transition_history_to_avoid_repeating_state_action() -> None:
    """Prefer the less-tried legal action for a previously visited state."""
    observation = {"grid": [[0, 1]], "state": "PLAYING"}
    signature = observation_signature(observation)
    history = [
        {
            "state_signature": signature,
            "next_state_signature": signature,
            "action": "ACTION2",
        },
        {
            "state_signature": signature,
            "next_state_signature": signature,
            "action": "ACTION2",
        },
    ]
    decision = ARC3ExplorationPolicy().choose_action(
        observation, ["ACTION1", "ACTION2"], history
    )

    assert decision.action == "ACTION1"
    assert decision.reasoning["local_state_action_visits"] == 0


def test_policy_fingerprint_is_stable_for_equivalent_mappings() -> None:
    """Ignore mapping insertion order when identifying a visible state."""
    assert observation_signature({"grid": [[1, 0]], "state": "PLAYING"}) == (
        observation_signature({"state": "PLAYING", "grid": [[1, 0]]})
    )


def test_policy_uses_explicit_terminal_outcome_for_same_state_action() -> None:
    """A win raises and a game-over lowers the observed action preference."""
    observation = {"grid": [[0, 1]], "state": "PLAYING"}
    next_observation = {"grid": [[1, 1]], "state": "WIN"}
    policy = ARC3ExplorationPolicy()

    policy.observe_transition(observation, "ACTION2", next_observation, "WIN")
    decision = policy.choose_action(observation, ["ACTION1", "ACTION2"], [])

    assert decision.action == "ACTION2"
    assert decision.reasoning["score_semantics"] == (
        "exploration_priority_plus_observed_reward_not_success_probability"
    )

    losing_policy = ARC3ExplorationPolicy()
    losing_policy.observe_transition(
        observation,
        "ACTION2",
        {"grid": [[0, 0]], "state": "GAME_OVER"},
        "GAME_OVER",
    )
    losing_decision = losing_policy.choose_action(
        observation, ["ACTION1", "ACTION2"], []
    )
    assert losing_decision.action == "ACTION1"


def test_policy_does_not_invent_reward_for_nonterminal_transition() -> None:
    """A state change without explicit terminal feedback gets no reward."""
    observation = {"grid": [[0, 1]], "state": "PLAYING"}
    policy = ARC3ExplorationPolicy()
    policy.observe_transition(
        observation,
        "ACTION2",
        {"grid": [[1, 1]], "state": "PLAYING"},
        "PLAYING",
    )

    assert policy._outcomes == {}


def test_policy_reconstructs_seen_state_from_prior_transition_key() -> None:
    """A resumed policy must recognize a prior state even if it never reappears as next state."""
    observation = {"grid": [[0]], "state": "PLAYING"}
    other_state = {"grid": [[1]], "state": "PLAYING"}
    history = [
        {
            "state_signature": observation_signature(observation),
            "next_state_signature": observation_signature(other_state),
            "action": "ACTION1",
        }
    ]

    decision = ARC3ExplorationPolicy().choose_action(
        observation, ["ACTION1", "ACTION2"], history
    )

    assert decision.reasoning["state_novel"] is False


def test_policy_restores_terminal_outcomes_from_saved_history() -> None:
    """A fresh policy should retain explicit terminal feedback from replay history."""
    observation = {"grid": [[0, 1]], "state": "PLAYING"}
    signature = observation_signature(observation)
    history = [
        {
            "state_signature": signature,
            "next_state_signature": observation_signature(
                {"grid": [[1, 1]], "state": "WIN"}
            ),
            "action": "ACTION2",
            "terminal_state": None,
            "state_after": "WIN",
        }
    ]

    decision = ARC3ExplorationPolicy().choose_action(
        observation, ["ACTION1", "ACTION2"], history
    )

    assert decision.action == "ACTION2"

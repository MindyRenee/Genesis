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
    assert decision.reasoning["policy"] == "bounded_novelty_exploration"
    assert decision.reasoning["score_semantics"] == "exploration_priority_not_success_probability"
    assert len(decision.reasoning["candidate_scores"]) == 2


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

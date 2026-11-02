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


def _big(counter: int, mark: int = 0) -> dict:
    """A 32x32 board with a step-counter pixel on the top row and one object."""
    grid = [[0] * 32 for _ in range(32)]
    grid[0][counter % 32] = 9
    grid[16][16] = mark
    return {"grid": grid, "state": "NOT_FINISHED"}


def test_policy_masks_border_step_counter() -> None:
    """A border pixel that changes on every action is masked out of state identity."""
    policy = ARC3ExplorationPolicy()
    for step in range(5):
        policy.observe_transition(_big(step), "ACTION1", _big(step + 1), "NOT_FINISHED")
    assert ("row", 0) in policy.volatile_lines
    assert policy.signature(_big(3)) == policy.signature(_big(17))
    assert policy.signature(_big(3)) != policy.signature(_big(3, mark=5))


def test_policy_defers_actions_that_never_change_the_board() -> None:
    """With the counter masked, a repeatedly no-op action loses to an effective one."""
    policy = ARC3ExplorationPolicy()
    for step in range(4):
        policy.observe_transition(_big(step), "ACTION1", _big(step + 1), "NOT_FINISHED")
    policy.observe_transition(_big(4), "ACTION2", _big(5, mark=3), "NOT_FINISHED")
    decision = policy.choose_action(_big(6, mark=3), ["ACTION1", "ACTION2"], [])
    assert decision.action == "ACTION2"


def test_policy_generalises_click_effects_by_colour() -> None:
    """Clicks on a colour that never changes the board lose to clicks on an untried colour."""
    grid = [[0] * 32 for _ in range(32)]
    for y in range(4, 8):
        for x in range(4, 8):
            grid[y][x] = 3
    for y in range(20, 24):
        for x in range(20, 24):
            grid[y][x] = 5
    board = {"grid": grid, "state": "NOT_FINISHED"}
    policy = ARC3ExplorationPolicy()
    for x in range(4, 8):
        policy.observe_transition(board, "ACTION6", board, "NOT_FINISHED", data={"x": x, "y": 4})
    decision = policy.choose_action(board, ["ACTION6"], [])
    assert decision.data is not None
    assert grid[decision.data["y"]][decision.data["x"]] != 3


def test_policy_replays_previous_level_solution() -> None:
    """After a level completes, the shortest known solution is tried first on the next level."""
    policy = ARC3ExplorationPolicy()
    start = {"grid": [[0, 0]], "state": "NOT_FINISHED", "levels_completed": 0}
    middle = {"grid": [[1, 0]], "state": "NOT_FINISHED", "levels_completed": 0}
    won = {"grid": [[2, 2]], "state": "NOT_FINISHED", "levels_completed": 1}
    policy.choose_action(start, ["ACTION1", "ACTION2"], [])
    policy.observe_transition(start, "ACTION2", middle, "NOT_FINISHED")
    policy.observe_transition(middle, "ACTION1", won, "NOT_FINISHED")
    first = policy.choose_action(won, ["ACTION1", "ACTION2"], [])
    assert (first.action, first.reasoning["mode"]) == ("ACTION2", "procedural_replay")


def test_neuro_loop_closes_field_waves_and_chemicals() -> None:
    """Game events move her chemicals, the waves respond, and the field channel is honest."""
    from parietal_lobe.arc_neuro import ArcNeuroLoop

    calls: list[int] = []
    loop = ArcNeuroLoop(field_source=lambda osc: calls.append(1) is None)
    base = dict(loop.chemicals)
    state = loop.beat(dopamine_rpe=0.8, novel=True, changed=False)
    assert calls == [1] and loop.field_live
    assert loop.chemicals["dopamine"] > base["dopamine"]
    assert loop.chemicals["norepinephrine"] > base["norepinephrine"]
    assert loop.chemicals["gaba"] > base["gaba"]
    assert 0.0 <= state.focus <= 1.0
    assert ArcNeuroLoop().telemetry()["em_field"] == "absent"

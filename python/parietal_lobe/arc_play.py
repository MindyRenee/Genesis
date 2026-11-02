"""Bounded ARC-AGI-3 interaction loop for Genesis-owned action policies.

This module deliberately does not contain a solver or select actions. The
caller supplies Genesis' policy; the loop enforces a step budget, passes
each observation and available action set to that policy, recovers from
``GAME_OVER`` with ``RESET`` (as the official toolkit documents), tracks
level progress, and returns a replayable in-memory episode record.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.arc_sources import (
    GAME_OVER,
    RESET_ACTION,
    TERMINAL_WIN,
    ARCSourceError,
    arc3_legal_actions,
    observation_percept,
    observation_state,
    observation_view,
    step_arc3_game,
)


@dataclass(frozen=True)
class ARC3Decision:
    """One policy decision, using an action name exposed by the toolkit."""

    action: str
    data: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None


@dataclass
class ARC3Episode:
    """Outcome and auditable transition history for one bounded game session.

    ``terminal_state`` is ``WIN`` when every level was cleared, or
    ``GAME_OVER`` when the loop stopped on a game over it did not (or was
    not allowed to) reset. ``final_state`` is the state of the last
    observation regardless of why the loop stopped.
    """

    game_id: str
    transitions: list[dict[str, Any]] = field(default_factory=list)
    terminal_state: str | None = None
    final_state: str | None = None
    stopped_reason: str = "step_limit"
    levels_completed: int = 0
    win_levels: int | None = None
    resets: int = 0

    @property
    def actions_taken(self) -> int:
        """Number of environment actions submitted, RESETs included."""
        return len(self.transitions)


Policy = Callable[[Any, list[str], list[dict[str, Any]]], ARC3Decision]


def _levels(observation: Any) -> tuple[int, int | None]:
    view = observation_view(observation)
    return view["levels_completed"] or 0, view["win_levels"]


def transition_reward(
    levels_before: int, levels_after: int, state_after: str | None
) -> float:
    """Explicit environment feedback for one transition.

    +1 per level completed, +1 for winning the game, -1 for causing a game
    over; zero otherwise. No reward is invented for intermediate frames.
    """
    reward = float(max(0, levels_after - levels_before))
    if state_after == TERMINAL_WIN:
        reward += 1.0
    elif state_after == GAME_OVER:
        reward -= 1.0
    return reward


def run_arc3_episode(
    environment: Any,
    *,
    game_id: str,
    initial_observation: Any,
    choose_action: Policy,
    max_steps: int = 100,
    outer_world: Any | None = None,
    reset_on_game_over: bool = True,
    max_resets: int | None = None,
) -> ARC3Episode:
    """Run a bounded closed loop using a caller-supplied Genesis policy.

    The policy receives the current observation, legal action names, and
    previous transitions. It must return an ARC3Decision. This function
    never invents a game action when the policy fails or returns an illegal
    one. The single exception is ``RESET`` after ``GAME_OVER``: the toolkit
    accepts no other action in that state, so the loop issues it (when
    ``reset_on_game_over`` is true and ``max_resets`` is not exhausted) and
    records it as a normal transition, because it counts toward the score.

    ``max_steps`` bounds every submitted action, RESETs included. Level
    progress is read from ``levels_completed``/``win_levels`` and reported
    per transition as ``reward`` (see :func:`transition_reward`).

    When ``outer_world`` is supplied, each current observation is sent
    through its public ``perceive`` API and each executed action through
    ``it_acted``. This wires environmental events into Genesis' existing
    world event stream and its registered cognitive listener. It does not
    select a policy on Genesis' behalf: ``choose_action`` must still be
    supplied by the caller.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if max_resets is not None and (
        isinstance(max_resets, bool) or not isinstance(max_resets, int) or max_resets < 0
    ):
        raise ValueError("max_resets must be a non-negative integer or None")

    from parietal_lobe.arc_ascii import encode_game_record, transition_record
    from parietal_lobe.arc_policy import action_key, observation_signature

    episode = ARC3Episode(game_id=game_id)
    observation = initial_observation
    episode.levels_completed, episode.win_levels = _levels(observation)
    episode.final_state = observation_state(observation)
    policy_owner = getattr(choose_action, "__self__", None)
    observe_transition = getattr(policy_owner, "observe_transition", None)

    # Publish the initial observation once before the first decision. Each
    # subsequent observation is published immediately after its action so
    # even terminal feedback reaches the world's inbound cognitive pathway.
    if outer_world is not None:
        initial_event = outer_world.perceive(
            f"ARC-AGI-3 observation for game {game_id}, step 1",
            salience=0.7,
            source=f"arc3:{game_id}",
            metadata={"game_id": game_id, "step": 1, "phase": "initial"},
            percept=observation_percept(observation),
        )
        if initial_event is None:
            episode.stopped_reason = "outer_world_unavailable"
            return episode

    for step_index in range(max_steps):
        state_name = observation_state(observation)
        if state_name == TERMINAL_WIN:
            episode.terminal_state = state_name
            episode.stopped_reason = "terminal_state"
            return episode
        if state_name == GAME_OVER:
            if not reset_on_game_over:
                episode.terminal_state = state_name
                episode.stopped_reason = "terminal_state"
                return episode
            if max_resets is not None and episode.resets >= max_resets:
                episode.terminal_state = state_name
                episode.stopped_reason = "reset_limit"
                return episode
            legal_actions = [RESET_ACTION]
            decision = ARC3Decision(
                RESET_ACTION,
                reasoning={"policy": "episode_loop", "reason": "recover_from_game_over"},
            )
        else:
            legal_actions = arc3_legal_actions(environment)
            if not legal_actions:
                episode.stopped_reason = "no_legal_actions"
                return episode
            decision = choose_action(observation, legal_actions, episode.transitions)
            if not isinstance(decision, ARC3Decision):
                raise TypeError("choose_action must return ARC3Decision")
            if decision.action not in legal_actions and decision.action != RESET_ACTION:
                raise ARCSourceError(
                    f"Policy selected illegal action {decision.action!r}; "
                    f"available actions: {legal_actions}"
                )

        next_observation = step_arc3_game(
            environment,
            decision.action,
            data=decision.data,
            reasoning=decision.reasoning,
        )
        next_state_name = observation_state(next_observation)
        levels_before, _ = _levels(observation)
        levels_after, win_levels = _levels(next_observation)
        reward = (
            0.0
            if decision.action == RESET_ACTION
            else transition_reward(levels_before, levels_after, next_state_name)
        )
        if decision.action == RESET_ACTION:
            episode.resets += 1
        episode.levels_completed = max(episode.levels_completed, levels_after)
        if win_levels is not None:
            episode.win_levels = win_levels
        episode.final_state = next_state_name

        if callable(observe_transition) and decision.action != RESET_ACTION:
            observe_transition(
                observation,
                decision.action,
                next_observation,
                next_state_name,
                data=decision.data,
                reward=reward,
            )

        if outer_world is not None:
            outer_world.it_acted(
                f"ARC-AGI-3 action {decision.action} in game {game_id}",
                detail=(
                    f"step={step_index + 1}; state_before={state_name}; "
                    f"state_after={next_state_name}; levels={levels_before}->{levels_after}; "
                    f"data={decision.data!r}; reasoning={decision.reasoning!r}"
                ),
            )

        game_record = transition_record(
            game_id=game_id,
            step=step_index + 1,
            observation=observation_view(observation),
            legal_actions=legal_actions,
            action={
                "name": decision.action,
                "data": decision.data,
                "reasoning": decision.reasoning,
            },
            next_observation=observation_view(next_observation),
            terminal_state=(
                next_state_name if next_state_name in {TERMINAL_WIN, GAME_OVER} else None
            ),
        )
        ascii_record = encode_game_record(game_record)
        episode.transitions.append(
            {
                "ascii_codes": list(ascii_record.codes),
                "ascii_character_count": len(ascii_record.codes),
                "step": step_index + 1,
                "action": decision.action,
                "action_key": action_key(decision.action, decision.data),
                "data": decision.data,
                "reasoning": decision.reasoning,
                "state_before": state_name,
                "state_after": next_state_name,
                "levels_before": levels_before,
                "levels_after": levels_after,
                "reward": reward,
                "state_signature": observation_signature(observation),
                "next_state_signature": observation_signature(next_observation),
                "observation": next_observation,
            }
        )
        observation = next_observation

        if outer_world is not None:
            outcome_event = outer_world.perceive(
                f"ARC-AGI-3 observation after action {decision.action} "
                f"in game {game_id}, step {step_index + 1}",
                salience=0.8 if next_state_name in {TERMINAL_WIN, GAME_OVER} else 0.7,
                source=f"arc3:{game_id}",
                metadata={
                    "game_id": game_id,
                    "step": step_index + 1,
                    "phase": "outcome",
                    "action": decision.action,
                    "data": decision.data,
                    "state_before": state_name,
                    "state_after": next_state_name,
                    "levels_completed": levels_after,
                    "win_levels": win_levels,
                    "reward": reward,
                },
                percept=observation_percept(next_observation),
            )
            if outcome_event is None:
                if next_state_name in {TERMINAL_WIN, GAME_OVER}:
                    episode.terminal_state = next_state_name
                episode.stopped_reason = "outer_world_unavailable"
                return episode

        if next_state_name == TERMINAL_WIN:
            episode.terminal_state = next_state_name
            episode.stopped_reason = "terminal_state"
            return episode

    return episode

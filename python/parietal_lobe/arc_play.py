"""Bounded ARC-AGI-3 interaction loop for Genesis-owned action policies.

This module deliberately does not contain a solver or select actions. The
caller supplies Genesis' policy; the loop enforces a step budget, passes
each observation and available action set to that policy, and returns a
replayable in-memory episode record.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.arc_sources import (
    ARCSourceError,
    arc3_legal_actions,
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
    """Outcome and auditable transition history for one bounded game session."""

    game_id: str
    transitions: list[dict[str, Any]] = field(default_factory=list)
    terminal_state: str | None = None
    stopped_reason: str = "step_limit"


Policy = Callable[[Any, list[str], list[dict[str, Any]]], ARC3Decision]


def _outer_world_percept(observation: Any) -> Any:
    """Expose grid-shaped observations in the format Genesis already parses.

    Keep the original observation for policy and episode history. For
    OuterWorld's percept path, unwrap only explicit grid containers; do not
    guess at arbitrary toolkit fields or stringify structured state.
    """
    if isinstance(observation, dict):
        if "grid" in observation:
            return observation
        return observation
    grid = getattr(observation, "grid", None)
    if grid is not None:
        return {"grid": grid}
    return observation


def run_arc3_episode(
    environment: Any,
    *,
    game_id: str,
    initial_observation: Any,
    choose_action: Policy,
    max_steps: int = 100,
    outer_world: Any | None = None,
) -> ARC3Episode:
    """Run a bounded closed loop using a caller-supplied Genesis policy.

    The policy receives the current observation, legal action names, and
    previous transitions. It must return an ARC3Decision. This function
    never invents an action when the policy fails or returns an illegal one.

    When ``outer_world`` is supplied, each current observation is sent
    through its public ``perceive`` API and each executed action through
    ``it_acted``. This wires environmental events into Genesis' existing
    world event stream and its registered cognitive listener. It does not
    select a policy on Genesis' behalf: ``choose_action`` must still be
    supplied by the caller.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    from parietal_lobe.arc_policy import observation_signature

    episode = ARC3Episode(game_id=game_id)
    observation = initial_observation

    # Publish the initial observation once before the first decision. Each
    # subsequent observation is published immediately after its action so
    # even terminal feedback reaches the world's inbound cognitive pathway.
    if outer_world is not None:
        initial_event = outer_world.perceive(
            f"ARC-AGI-3 observation for game {game_id}, step 1",
            salience=0.7,
            source=f"arc3:{game_id}",
            metadata={"game_id": game_id, "step": 1, "phase": "initial"},
            percept=_outer_world_percept(observation),
        )
        if initial_event is None:
            episode.stopped_reason = "outer_world_unavailable"
            return episode

    for step_index in range(max_steps):
        state = getattr(observation, "state", None)
        state_name = getattr(state, "name", str(state)) if state is not None else None
        if state_name in {"WIN", "GAME_OVER"}:
            episode.terminal_state = state_name
            episode.stopped_reason = "terminal_state"
            return episode

        legal_actions = arc3_legal_actions(environment)
        if not legal_actions:
            episode.stopped_reason = "no_legal_actions"
            return episode

        decision = choose_action(observation, legal_actions, episode.transitions)
        if not isinstance(decision, ARC3Decision):
            raise TypeError("choose_action must return ARC3Decision")
        if decision.action not in legal_actions:
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
        next_state = getattr(next_observation, "state", None)
        next_state_name = (
            getattr(next_state, "name", str(next_state))
            if next_state is not None
            else None
        )

        policy_owner = getattr(choose_action, "__self__", None)
        observe_transition = getattr(policy_owner, "observe_transition", None)
        if callable(observe_transition):
            observe_transition(
                observation,
                decision.action,
                next_observation,
                next_state_name,
            )
        if outer_world is not None:
            outer_world.it_acted(
                f"ARC-AGI-3 action {decision.action} in game {game_id}",
                detail=(
                    f"step={step_index + 1}; state_before={state_name}; "
                    f"state_after={next_state_name}; reasoning={decision.reasoning!r}"
                ),
            )

        from parietal_lobe.arc_braille import encode_game_record, transition_record

        game_record = transition_record(
            game_id=game_id,
            step=step_index + 1,
            observation=observation,
            legal_actions=legal_actions,
            action={
                "name": decision.action,
                "data": decision.data,
                "reasoning": decision.reasoning,
            },
            next_observation=next_observation,
            terminal_state=next_state_name if next_state_name in {"WIN", "GAME_OVER"} else None,
        )
        braille_record = encode_game_record(game_record)
        episode.transitions.append(
            {
                "braille_cells": [cell.value for cell in braille_record.cells],
                "braille_bit_length": braille_record.bit_length,
                "step": step_index + 1,
                "action": decision.action,
                "data": decision.data,
                "reasoning": decision.reasoning,
                "state_before": state_name,
                "state_after": next_state_name,
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
                salience=0.8 if next_state_name in {"WIN", "GAME_OVER"} else 0.7,
                source=f"arc3:{game_id}",
                metadata={
                    "game_id": game_id,
                    "step": step_index + 1,
                    "phase": "outcome",
                    "action": decision.action,
                    "state_before": state_name,
                    "state_after": next_state_name,
                },
                percept=_outer_world_percept(next_observation),
            )
            if outcome_event is None:
                if next_state_name in {"WIN", "GAME_OVER"}:
                    episode.terminal_state = next_state_name
                episode.stopped_reason = "outer_world_unavailable"
                return episode

        if next_state_name in {"WIN", "GAME_OVER"}:
            episode.terminal_state = next_state_name
            episode.stopped_reason = "terminal_state"
            return episode

    return episode

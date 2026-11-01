"""Route ARC telemetry through Genesis's world and language-latent interfaces."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _state_name(observation: Any) -> str | None:
    state = (
        observation.get("state")
        if isinstance(observation, dict)
        else getattr(observation, "state", None)
    )
    if isinstance(state, dict):
        state = state.get("name", state.get("value"))
    if state is None:
        return None
    return str(getattr(state, "name", state))


def _percept(observation: Any) -> Any:
    if isinstance(observation, dict):
        return observation
    grid = getattr(observation, "grid", None)
    return {"grid": grid} if grid is not None else observation


def _event_text(event: str, payload: dict[str, Any]) -> str:
    """Build a stable, bounded language surface for the existing parser.

    Structured percepts remain separately available to the world. This text
    gives the language latent space a traceable description of each event;
    it does not claim the parser fully understands arbitrary grid data.
    """
    game_id = str(payload.get("game_id", "unknown"))
    step = int(payload.get("step", 0))
    if event == "action":
        before = _state_name(payload.get("observation_before")) or "unknown"
        after = _state_name(payload.get("observation_after")) or "unknown"
        action = str(payload.get("action", "unknown"))
        return (
            f"ARC game {game_id} step {step}: action {action}; "
            f"state changed from {before} to {after}."
        )
    observation = payload.get("observation")
    phase = str(payload.get("phase", "observation"))
    state = _state_name(observation) or "unknown"
    legal = payload.get("legal_actions", [])
    legal_text = ", ".join(str(item) for item in legal[:64])
    return (
        f"ARC game {game_id} step {step}: {phase} observation; "
        f"state {state}; legal actions {legal_text}."
    )


class GenesisArcObserver:
    """Route ARC events to OuterWorld and the actual temporal language space."""

    def __init__(self, mind: Any) -> None:
        world = getattr(mind, "world", None)
        if world is None or not callable(getattr(world, "perceive", None)):
            raise TypeError("mind must expose world.perceive")
        if not callable(getattr(world, "it_acted", None)):
            raise TypeError("mind.world must expose it_acted")
        brain = getattr(mind, "brain", None)
        temporal = getattr(brain, "temporal", None)
        self._language_observer = getattr(temporal, "observe_language", None)
        self._comprehension = getattr(temporal, "comprehension", None)
        if callable(self._language_observer) and not callable(
            getattr(self._comprehension, "comprehend", None)
        ):
            self._language_observer = None
        self._world = world
        self.events_seen = 0
        self.events_failed = 0
        self.latent_observations = 0
        self.last_transition_residual = 0.0

    def _update_language_latent(self, event: str, payload: dict[str, Any]) -> None:
        if self._language_observer is None:
            return
        text = _event_text(event, payload)
        parsed = self._comprehension.comprehend(text)
        snapshot = self._language_observer(text, parsed, direction="internal")
        residual = snapshot.get("transition_residual", 0.0)
        self.last_transition_residual = (
            float(residual) if isinstance(residual, (int, float)) else 0.0
        )
        self.latent_observations += 1

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        try:
            game_id = str(payload.get("game_id", "unknown"))
            step = int(payload.get("step", 0))
            if event == "action":
                action = str(payload["action"])
                detail = json.dumps(
                    {
                        "source": "arc_phone_bridge",
                        "game_id": game_id,
                        "step": step,
                        "state_before": _state_name(payload.get("observation_before")),
                        "state_after": _state_name(payload.get("observation_after")),
                        "data": payload.get("data"),
                        "reasoning": payload.get("reasoning"),
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
                self._world.it_acted(
                    f"ARC-AGI-3 action {action} in game {game_id}", detail=detail
                )
            elif event == "observation":
                observation = payload.get("observation")
                phase = str(payload.get("phase", "observation"))
                self._world.perceive(
                    f"ARC-AGI-3 {phase} observation for game {game_id}",
                    salience=0.7,
                    source=f"arc3:{game_id}",
                    topics=["arc3", "game_observation"],
                    metadata={
                        "game_id": game_id,
                        "step": step,
                        "phase": phase,
                        "state": _state_name(observation),
                        "legal_actions": payload.get("legal_actions", []),
                    },
                    percept=_percept(observation),
                )
            else:
                raise ValueError(f"unknown ARC observer event: {event}")
            self._update_language_latent(event, payload)
            self.events_seen += 1
        except Exception:
            self.events_failed += 1
            logger.exception("Failed to route ARC bridge event %s into Genesis", event)
            raise

    def snapshot(self) -> dict[str, int | float]:
        """Expose world-routing and latent-update counts for runtime verification."""
        return {
            "events_seen": self.events_seen,
            "events_failed": self.events_failed,
            "latent_observations": self.latent_observations,
            "last_transition_residual": self.last_transition_residual,
        }

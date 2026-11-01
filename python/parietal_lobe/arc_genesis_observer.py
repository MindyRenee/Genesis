"""Forward ARC bridge events into Genesis's real OuterWorld event stream."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _state_name(observation: Any) -> str | None:
    state = observation.get("state") if isinstance(observation, dict) else getattr(observation, "state", None)
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


class GenesisArcObserver:
    """Send actual ARC observations and executed actions through Mind.world."""

    def __init__(self, mind: Any) -> None:
        world = getattr(mind, "world", None)
        if world is None or not callable(getattr(world, "perceive", None)):
            raise TypeError("mind must expose world.perceive")
        if not callable(getattr(world, "it_acted", None)):
            raise TypeError("mind.world must expose it_acted")
        self._world = world
        self.events_seen = 0
        self.events_failed = 0

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        try:
            game_id = str(payload.get("game_id", "unknown"))
            step = int(payload.get("step", 0))
            if event == "action":
                action = str(payload["action"])
                detail = json.dumps({
                    "source": "arc_phone_bridge", "game_id": game_id, "step": step,
                    "state_before": _state_name(payload.get("observation_before")),
                    "state_after": _state_name(payload.get("observation_after")),
                    "data": payload.get("data"), "reasoning": payload.get("reasoning"),
                }, sort_keys=True, default=str, separators=(",", ":"))
                self._world.it_acted(f"ARC-AGI-3 action {action} in game {game_id}", detail=detail)
            elif event == "observation":
                observation = payload.get("observation")
                phase = str(payload.get("phase", "observation"))
                self._world.perceive(
                    f"ARC-AGI-3 {phase} observation for game {game_id}",
                    salience=0.7,
                    source=f"arc3:{game_id}",
                    topics=["arc3", "game_observation"],
                    metadata={
                        "game_id": game_id, "step": step, "phase": phase,
                        "state": _state_name(observation),
                        "legal_actions": payload.get("legal_actions", []),
                    },
                    percept=_percept(observation),
                )
            else:
                raise ValueError(f"unknown ARC observer event: {event}")
            self.events_seen += 1
        except Exception:
            self.events_failed += 1
            logger.exception("Failed to route ARC bridge event %s into Genesis", event)
            raise

    def snapshot(self) -> dict[str, int]:
        return {"events_seen": self.events_seen, "events_failed": self.events_failed}

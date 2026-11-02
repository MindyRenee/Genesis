"""ARC arena — ARC-AGI-3 games as a place in Genesis's outer world.

A game is somewhere she goes, not a script run beside her. Every board
she sees arrives through ``OuterWorld.perceive`` with the 64x64 grid as
its percept, and every move she makes is recorded through
``OuterWorld.it_acted``. The events therefore enter the same stream,
successor maps, and experiential latent space as the rest of her world.

How she uses it:

1. ``arena = ArcArena(world)`` attaches the arena to her ``OuterWorld``
   (``world.arc_arena``).
2. ``arena.play(environment, game_id=..., initial_observation=...,
   policy=ARC3ExplorationPolicy(), max_steps=...)`` plays one game. The
   policy decides from pixels only; the arena carries perception and
   action through the world.
3. ``arena.visits`` lists finished games; ``arena.latent()`` returns the
   world's experiential latent snapshot after play.

The arena owns no cognition and invents no actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.arc_play import ARC3Episode, run_arc3_episode


@dataclass(slots=True)
class ArenaVisit:
    """Summary of one game played in the arena."""

    game_id: str
    actions: int
    resets: int
    levels_completed: int
    final_state: str | None
    latent_observations: int


@dataclass
class ArcArena:
    """The ARC games, entered through Genesis's ``OuterWorld``.

    Args:
        world: Her ``OuterWorld``. It must expose ``perceive`` and
            ``it_acted``; the arena registers itself as ``world.arc_arena``.
    """

    world: Any
    visits: list[ArenaVisit] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the world can carry percepts and acts, then attach to it."""
        for name in ("perceive", "it_acted"):
            if not callable(getattr(self.world, name, None)):
                raise TypeError(f"world must expose {name}()")
        self.world.arc_arena = self

    def play(
        self,
        environment: Any,
        *,
        game_id: str,
        initial_observation: Any,
        policy: Any,
        max_steps: int = 1500,
        max_resets: int | None = None,
    ) -> ARC3Episode:
        """Play one game inside the world and return the episode.

        ``policy`` needs ``choose_action(observation, legal_actions,
        previous_transitions)``. GAME_OVER is answered with RESET, the only
        legal recovery; WIN ends the visit.
        """
        episode = run_arc3_episode(
            environment,
            game_id=game_id,
            initial_observation=initial_observation,
            choose_action=policy.choose_action,
            max_steps=max_steps,
            outer_world=self.world,
            max_resets=max_resets,
        )
        self.visits.append(
            ArenaVisit(
                game_id=game_id,
                actions=episode.actions_taken,
                resets=episode.resets,
                levels_completed=episode.levels_completed,
                final_state=episode.final_state,
                latent_observations=self._latent_observations(),
            )
        )
        return episode

    def latent(self) -> dict[str, Any] | None:
        """The world's experiential latent snapshot, if the world has one."""
        space = getattr(self.world, "experiential_space", None)
        snapshot = getattr(space, "snapshot", None)
        return snapshot() if callable(snapshot) else None

    def _latent_observations(self) -> int:
        """How many world events the experiential latent space has encoded."""
        space = getattr(self.world, "experiential_space", None)
        return int(getattr(space, "observations", 0) or 0)

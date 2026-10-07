"""World policy — action selection over the parietal world model.

The basal ganglia's job: given everything cortex knows, decide what
to do. The parietal ``WorldModel`` maintains what is true — where
things are, which one is her, what touching them does — and this
module resolves the competition among the drives that state
supports:

1. **Goal pursuit** — a thing whose kind collects is worth having;
   path toward the nearest one.
2. **Calibration** — an action whose effect is still unknown must be
   tried before anything else can be planned.
3. **Affordance probing** — a thing never touched is an unanswered
   question; its affordance is only learnable by contact. Stepping
   into it IS the probe — if it blocks her, the failed move is itself
   the observation.
4. **Exploration** — otherwise, move toward the frontier: cells
   least visited, adjacent to space never seen.

Returns action descriptors — ``{"action": id, "cell": optional
(r, c)}`` — for the caller to translate into whatever protocol the
environment speaks. The policy knows nothing about any environment's
action alphabet; ``available`` is ground truth supplied per step.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ..frontal_lobe.world_planner import WorldPlanner
from ..parietal_lobe.world_model import (
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_WIN,
)

if TYPE_CHECKING:
    from ..parietal_lobe.world_model import Cell, WorldEntity, WorldModel


class WorldPolicy:
    """Selects actions from the environment's per-step whitelist."""

    def __init__(self, model: WorldModel, seed: int | None = None) -> None:
        """Bind the policy to a world model.

        ``model`` supplies the entities, affordances, and learned
        action physics the planner deliberates over; ``seed`` makes
        the RNG's tie-broken choices reproducible.
        """
        self._model = model
        self._planner = WorldPlanner(model)
        self._rng = random.Random(seed)

    def choose(self, available: list[str]) -> dict[str, Any] | None:
        """Pick the next action, or ``None`` if nothing is offered."""
        if not available:
            return None
        model = self._model
        me = model.self_entity

        if me is not None:
            # Deliberation first: plan through the world model's
            # learned physics toward something worth collecting.
            # The planner only uses understood actions — untried ones
            # simply can't appear in its plans.
            action = self._planner.next_action(
                self._valued_cells(), available
            )
            if action is not None:
                return {"action": action, "cell": self._cell_for(action)}
            # No plan worth executing — calibrate: an action whose
            # effect is unknown must be tried before it can be used.
            # Actions whose interface label declares them episode-
            # ending ("reset") are calibrated last — restarting to
            # satisfy curiosity forfeits real progress.
            untried = [
                a for a in available if not model.action_understood(a)
            ]
            if untried:
                non_reset = [
                    a for a in untried
                    if model.action_labels.get(a) != "reset"
                ]
                pick = self._rng.choice(non_reset or untried)
                return {"action": pick, "cell": self._cell_for(pick)}
            # Affordance probing: go touch whatever she hasn't tried.
            probe = self._probe_unknown(me, available)
            if probe is not None:
                return probe
            action = self._explore_action(me, available)
            return {"action": action, "cell": self._cell_for(action)}

        # No body found — the environment may take pointed input.
        action = self._rng.choice(available)
        return {"action": action, "cell": self._cell_for(action)}

    def _cell_for(self, action: str) -> Cell | None:
        """Coordinate-typed actions need a target; simple ones don't.

        The interface labels which actions take a point — the same
        convention every participant sees. What the point *does* is
        still hers to learn.
        """
        if action in self._model.coordinate_actions:
            return self._most_informative_cell()
        if self._model.self_entity is None:
            return self._most_informative_cell()
        return None

    def _valued_cells(self) -> set[Cell]:
        """Cells of live entities whose kind is worth collecting."""
        model = self._model
        return {
            cell
            for e in model.entities.values()
            if not e.gone and not e.is_self
            and (
                model.affordance_of(e.color)[OUT_COLLECT] > 0
                or model.affordance_of(e.color)[OUT_WIN] > 0
            )
            and model.affordance_of(e.color)[OUT_HAZARD] == 0
            for cell in e.cells
        }

    def _probe_unknown(
        self, me: WorldEntity, available: list[str]
    ) -> dict[str, Any] | None:
        """Move toward the nearest entity kind she has never touched."""
        model = self._model
        unprobed = {
            cell
            for e in model.entities.values()
            if not e.gone and not e.is_self
            and not model.affordance_of(e.color)
            for cell in e.cells
        }
        if not unprobed:
            return None
        action = self._planner.next_action(unprobed, available)
        if action is not None and action in available:
            return {"action": action, "cell": None}
        return None

    def _explore_action(self, me: WorldEntity, available: list[str]) -> str:
        """Novelty-seeking with a frontier bias.

        Beyond preferring least-tried actions and least-visited
        destinations, an action is pulled toward the frontier — cells
        adjacent to space never visited — so coverage is systematic
        rather than a random walk.
        """
        model = self._model
        best = available[0]
        best_score = -1.0
        for action in available:
            # The label prior is a hypothesis for exploration scoring;
            # observed effects replace it wholesale once they exist.
            effect = model.expected_effect(action)
            tries = model.action_tries[action]
            novelty = 0.0
            frontier = 0.0
            if effect is not None:
                dest = (
                    round(me.centroid[0] + effect[0]),
                    round(me.centroid[1] + effect[1]),
                )
                if dest in model.known_blocked:
                    novelty = -1.0
                else:
                    novelty = 1.0 / (1 + model.visits[dest])
                    unseen = sum(
                        1
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        for n in [(dest[0] + dr, dest[1] + dc)]
                        if n in model.known_free and model.visits[n] == 0
                    )
                    frontier = unseen / 4.0
            score = (
                novelty + frontier + 1.0 / (1 + tries)
                + self._rng.random() * 0.1
            )
            if model.action_labels.get(action) == "reset":
                score -= 1.0  # restarting the episode is a last resort
            if score > best_score:
                best, best_score = action, score
        return best

    def _most_informative_cell(self) -> Cell | None:
        """A cell worth pointing at: the least-understood entity."""
        best: WorldEntity | None = None
        least = float("inf")
        for entity in self._model.entities.values():
            if entity.gone:
                continue
            known = sum(self._model.affordance_of(entity.color).values())
            if known < least:
                best, least = entity, known
        if best is None:
            return None
        return (round(best.centroid[0]), round(best.centroid[1]))

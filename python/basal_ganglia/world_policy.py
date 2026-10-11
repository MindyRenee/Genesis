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

from frontal_lobe.world_planner import WorldPlanner
from parietal_lobe.world_model import (
    OUT_BLOCK,
    OUT_COLLECT,
    OUT_HAZARD,
    OUT_TRANSFORM,
    OUT_WIN,
)

if TYPE_CHECKING:
    from basal_ganglia.motor_field import MotorField
    from parietal_lobe.world_model import Cell, WorldEntity, WorldModel


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
        # Distributed motor field composed for the in-flight choice —
        # set by ``choose`` for the duration of one selection.
        self._field: MotorField | None = None

    def choose(
        self,
        available: list[str],
        field: MotorField | None = None,
    ) -> dict[str, Any] | None:
        """Pick the next action, or ``None`` if nothing is offered.

        ``field`` is the distributed motor state composed for this
        step (see ``basal_ganglia.motor_field``): its action
        dispositions bias exploration scoring and its spatial peak
        supplies the target for coordinate-typed actions.
        """
        if not available:
            return None
        self._field = field
        model = self._model
        me = model.self_entity

        if me is not None:
            # The field's own verdict first: when composed
            # dispositions crown a decisive leader — a specialist's
            # confident proposal — the aggregate of every voice that
            # wrote onto the field already prefers it. The planner
            # sees only its own model; a flat field is what leaves
            # deliberation room to differ.
            lead = self._field_leader(available)
            if lead is not None:
                return {
                    "action": lead,
                    "cell": self._cell_for(lead),
                    "source": "field_leader",
                }
            # Deliberation next: plan through the world model's
            # learned physics toward something worth collecting.
            # The planner only uses understood actions — untried ones
            # simply can't appear in its plans.
            action = self._planner.next_action(
                self._valued_cells(), available
            )
            if action is not None:
                return {
                    "action": action,
                    "cell": self._cell_for(action),
                    "source": "planned",
                }
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
                return {
                    "action": pick,
                    "cell": self._cell_for(pick),
                    "source": "calibrate",
                }
            # Affordance probing: go touch whatever she hasn't tried.
            probe = self._probe_unknown(available)
            if probe is not None:
                return probe
            action = self._explore_action(me, available)
            return {
                "action": action,
                "cell": self._cell_for(action),
                "source": "explore",
            }

        # No body found — the environment may take pointed input. A
        # decisive field still speaks: the specialist's disposition
        # doesn't need the model's self-tracking to be right.
        lead = self._field_leader(available)
        action = lead if lead is not None else self._rng.choice(available)
        return {
            "action": action,
            "cell": self._cell_for(action),
            "source": "field_leader" if lead is not None else "rng_no_body",
        }

    def _field_leader(self, available: list[str]) -> str | None:
        """A decisive disposition leader, or None.

        The motor field carries every voice that wrote onto it —
        specialist dispositions, hypothesis pull, the model's own.
        When one action clears the runner-up by a full margin the
        field has effectively already decided: honor it rather than
        spend a calibration step on noise. Flat fields yield no
        leader and the ordinary drives decide as before.
        """
        if self._field is None or not self._field.action_scores:
            return None
        scored = sorted(
            (self._field.action_scores.get(a, 0.0), a)
            for a in available
        )
        if len(scored) < 2:
            return None
        second, top = scored[-2], scored[-1]
        if top[0] > 0 and top[0] - second[0] >= 1.0:
            return top[1]
        return None

    def _cell_for(self, action: str) -> Cell | None:
        """Coordinate-typed actions need a target; simple ones don't.

        The interface labels which actions take a point — the same
        convention every participant sees. What the point *does* is
        still hers to learn. When a motor field was composed for this
        step its spatial peak is the target — the distributed decision
        supplies the coordinate.
        """
        target = (
            self._field.peak
            if self._field is not None and self._field.peak is not None
            else self._most_informative_cell()
        )
        if action in self._model.coordinate_actions:
            return target
        if self._model.self_entity is None:
            return target
        return None

    def _valued_cells(self) -> set[Cell]:
        """Cells of live entities whose kind is worth reaching.

        Collectibles and goal-kinds are ends. Transformers are a
        means — worth reaching only while her *current* self-state
        has been refused somewhere: the gate said no to what she
        is, and the thing that rewrites her is how that changes.
        """
        model = self._model
        sig = model.self_signature()
        needs_change = sig is not None and any(
            not e.gone and sig in e.blocked_states
            for e in model.entities.values()
        )
        return {
            cell
            for e in model.entities.values()
            if not e.gone and not e.is_self and not e.part_of_self
            and e.color not in model.terrain_kinds
            and (
                model.affordance_of(e.color)[OUT_COLLECT] > 0
                or model.affordance_of(e.color)[OUT_WIN] > 0
                or (
                    needs_change
                    and model.affordance_of(e.color)[OUT_TRANSFORM] > 0
                )
            )
            and not model.hazard_kind(e.color)
            for cell in e.cells
        }

    def _probe_unknown(
        self, available: list[str]
    ) -> dict[str, Any] | None:
        """Move toward what contact would still teach her.

        Two frontiers: a kind never touched at all, and — once her
        own state can change — an entry-dependent thing whose answer
        for *this* self-state is unknown. A gate that refused her
        before stays a question every time she is remade.
        """
        model = self._model
        sig = model.self_signature()
        unprobed = {
            cell
            for e in model.entities.values()
            if not e.gone and not e.is_self and not e.part_of_self
            and e.color not in model.terrain_kinds
            and (
                not model.affordance_of(e.color)
                or (
                    sig is not None
                    and sig not in e.open_states
                    and sig not in e.blocked_states
                    and (
                        e.blocked_states
                        or e.open_states
                        or model.affordance_of(e.color)[OUT_BLOCK] > 0
                        or model.affordance_of(e.color)[OUT_HAZARD] > 0
                    )
                )
            )
            for cell in e.cells
        }
        if not unprobed:
            return None
        action = self._planner.next_action(unprobed, available)
        if action is not None and action in available:
            return {"action": action, "cell": None, "source": "probe"}
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
        fp = model.self_footprint()
        for action in available:
            # The label prior is a hypothesis for exploration scoring;
            # observed effects replace it wholesale once they exist.
            effect = model.expected_effect(action)
            tries = model.action_tries[action]
            novelty = 0.0
            frontier = 0.0
            if effect is not None and fp:
                dest_cells = {
                    (r + effect[0], c + effect[1]) for r, c in fp
                }
                if dest_cells & model.known_blocked:
                    novelty = -1.0
                else:
                    novelty = sum(
                        1 for cell in dest_cells if model.visits[cell] == 0
                    ) / len(dest_cells)
                    unseen = sum(
                        1
                        for r, c in dest_cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                        for n in [(r + dr, c + dc)]
                        if n in model.known_free and model.visits[n] == 0
                    )
                    frontier = min(1.0, unseen / len(dest_cells))
            score = (
                novelty + frontier + 1.0 / (1 + tries)
                + self._rng.random() * 0.1
            )
            if self._field is not None:
                # The action disposition the motor field carries —
                # decisions laid out over the space feed back into
                # the discrete choice.
                score += self._field.action_scores.get(action, 0.0)
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
            if entity.gone or entity.color in self._model.terrain_kinds:
                continue
            known = sum(self._model.affordance_of(entity.color).values())
            if known < least:
                best, least = entity, known
        if best is None:
            return None
        return (round(best.centroid[0]), round(best.centroid[1]))

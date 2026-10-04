"""World planner — deliberative lookahead over the world model.

The frontal lobe's contribution to embodiment: before acting,
simulate. The parietal ``WorldModel`` supplies the state and the
learned transition function (``simulate``); this module searches it.

The search space is ``SimState`` — her footprint, her color, the
consumed set — so plans can route through state changes, not just
space: "step on the thing that transforms me, then the thing that
stopped me before, then the thing I want" is a path like any other.
Whether that sequence works is decided by physics the model already
learned; the planner holds no environment knowledge of its own.

Replanning is per-decision and cheap, which is the correct reaction
to surprise: when a simulated-open gate refuses, the model's
conditional passability records the new evidence and the next
``next_action`` call plans around it.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..parietal_lobe.world_model import Cell, SimState, WorldModel

# Search bounds — deliberation is bounded, not exhaustive.
_MAX_EXPANSIONS = 30_000
_MAX_DEPTH = 80


class WorldPlanner:
    """Plans action sequences through the learned world model."""

    def __init__(self, model: WorldModel) -> None:
        self._model = model

    def next_action(
        self, goal_cells: set[Cell], available: list[str]
    ) -> str | None:
        """First action of the shortest plan reaching the goal.

        Breadth-first over simulated states — the plan found is
        step-minimal in *simulated* physics. ``None`` when no plan
        exists (unreachable goal, missing action semantics, or
        search bounds hit); the caller falls back to cheaper drives.
        """
        model = self._model
        start = model.start_state()
        if start is None or not goal_cells:
            return None
        if start.cells & goal_cells:
            return None

        moves = [
            a for a in available
            if model.effect_of(a) not in (None, (0, 0))
        ]
        if not moves:
            return None

        frontier: deque[tuple[SimState, int]] = deque([(start, 0)])
        parent: dict[SimState, tuple[SimState | None, str | None]] = {
            start: (None, None),
        }
        expansions = 0
        while frontier and expansions < _MAX_EXPANSIONS:
            state, depth = frontier.popleft()
            expansions += 1
            if depth >= _MAX_DEPTH:
                continue
            for action in moves:
                nxt = model.simulate(state, action)
                if nxt is None or nxt in parent:
                    continue
                parent[nxt] = (state, action)
                if nxt.cells & goal_cells:
                    return self._first(parent, nxt)
                frontier.append((nxt, depth + 1))
        return None

    @staticmethod
    def _first(
        parent: dict[SimState, tuple[SimState | None, str | None]],
        state: SimState,
    ) -> str | None:
        """Walk the parent chain to the action leaving the root."""
        prev, action = parent[state]
        while prev is not None and parent[prev][0] is not None:
            state = prev
            prev, action = parent[state]
        return action

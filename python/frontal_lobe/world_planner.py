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

import heapq
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parietal_lobe.world_model import Cell, SimState, WorldModel

# Search bounds — deliberation is bounded, not exhaustive.
_MAX_EXPANSIONS = 30_000
_MAX_DEPTH = 80


class WorldPlanner:
    """Plans action sequences through the learned world model."""

    def __init__(self, model: WorldModel) -> None:
        self._model = model

    def next_action(self, goal_cells: set[Cell], available: list[str]) -> str | None:
        """First action of the shortest plan reaching the goal.

        A* over simulated states — the plan found is step-minimal in
        *simulated* physics. ``None`` when no plan exists
        (unreachable goal, missing action semantics, or search
        bounds hit); the caller falls back to cheaper drives.
        """
        model = self._model
        start = model.start_state()
        if start is None or not goal_cells:
            return None
        if start.cells & goal_cells:
            return None
        parent, goal = self._search(start, goal_cells, available)
        if goal is None:
            return None
        return self._first(parent, goal)

    def plan(
        self, goal_cells: set[Cell], available: list[str]
    ) -> list[tuple[str, SimState]] | None:
        """The whole route, not just its first step.

        Returns ``[(action, expected-state-after), ...]`` in order —
        a cached trajectory the caller can execute stepwise, checking
        each expected state against what the world actually does.
        Sweeps are expensive; the point of returning the sequence is
        that it only has to be computed at decision points.
        """
        model = self._model
        start = model.start_state()
        if start is None or not goal_cells:
            return None
        if start.cells & goal_cells:
            return []
        parent, goal = self._search(start, goal_cells, available)
        if goal is None:
            return None
        return self._sequence(parent, goal)

    def _search(
        self,
        start: SimState,
        goal_cells: set[Cell],
        available: list[str],
    ) -> tuple[
        dict[SimState, tuple[SimState | None, str | None]],
        SimState | None,
    ]:
        """A* toward the goal cells through her learned physics.

        The heuristic is strides-to-go: how many of her biggest
        steps could cover the remaining distance. It never
        overestimates — every action moves at most one stride in
        one axis — so the first goal popped is reached by a shortest
        plan, as breadth-first would find, but without sweeping the
        whole reachable map to get there.
        """
        model = self._model
        moves = [a for a in available if model.effect_of(a) not in (None, (0, 0))]
        if not moves:
            return {}, None
        stride = max(abs(dr) + abs(dc) for a in moves for dr, dc in (model.effect_of(a),)) or 1

        def heuristic(cells: frozenset[Cell]) -> float:
            # Her footprint's anchor tracks with her; distance in
            # strides is the cheapest step count reachable.
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            return min(
                (max(top - gr, 0, gr - bottom) + max(left - gc, 0, gc - right)) / stride
                for gr, gc in goal_cells
            )

        parent: dict[SimState, tuple[SimState | None, str | None]] = {
            start: (None, None),
        }
        best_g: dict[SimState, int] = {start: 0}
        frontier: list[tuple[float, int, int, SimState]] = [(heuristic(start.cells), 0, 0, start)]
        serial = 1
        expansions = 0
        while frontier and expansions < _MAX_EXPANSIONS:
            _, g, _, state = heapq.heappop(frontier)
            if g != best_g.get(state):
                continue
            if state.cells & goal_cells:
                return parent, state
            if g >= _MAX_DEPTH:
                continue
            expansions += 1
            for action in moves:
                # Optimistic expansion — D*-lite's assumption: cells
                # whose passability is unmeasured are hypothesized
                # open, so a route can lead somewhere she has not
                # already walked. Only learned refusals (blocked,
                # refused, condemned contacts) still veto. The plan
                # is verified stepwise against reality anyway — a
                # falsified premise replans at the next decision
                # point with the wall now known. Without this the
                # reachable set is the walked set, and on a fresh
                # level she cannot plan anywhere she has not already
                # been — which is how exploration, not pursuit, ends
                # up spending the whole episode.
                nxt = model.simulate(state, action, explore_unknown=True)
                if nxt is None:
                    continue
                ng = g + 1
                if ng >= best_g.get(nxt, ng + 1):
                    continue
                parent[nxt] = (state, action)
                best_g[nxt] = ng
                heapq.heappush(
                    frontier,
                    (ng + heuristic(nxt.cells), ng, serial, nxt),
                )
                serial += 1
        return parent, None

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

    @staticmethod
    def _sequence(
        parent: dict[SimState, tuple[SimState | None, str | None]],
        state: SimState,
    ) -> list[tuple[str, SimState]]:
        """Walk the parent chain to the root, forward order."""
        out: list[tuple[str, SimState]] = []
        while True:
            prev, action = parent[state]
            if prev is None or action is None:
                break
            out.append((action, state))
            state = prev
        out.reverse()
        return out

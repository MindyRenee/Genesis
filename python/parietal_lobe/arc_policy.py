"""A bounded, model-free graph-exploration policy for ARC-AGI-3.

This is an autonomous action-selection baseline, not a task solver. It
follows the strongest published training-free approach for ARC-AGI-3: keep a
directed graph of observed board states and tested actions, try untested
actions in the current state first, and otherwise walk the shortest known path
to the nearest state that still has untested actions. Click actions (ACTION6)
target the connected objects visible on the board instead of a blind 64x64
sweep. Explicit level-completion and game-over feedback is used as reward;
board changes are used as an exploration prior. It does not claim to infer
game rules or guarantee progress.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.arc_play import ARC3Decision
from parietal_lobe.arc_sources import (
    GAME_OVER,
    RESET_ACTION,
    TERMINAL_WIN,
    is_complex_action,
    observation_grid,
    observation_view,
)

Candidate = tuple[str, str, dict[str, int] | None]


def action_key(action: str, data: dict[str, Any] | None = None) -> str:
    """Identify an action together with its coordinates, e.g. ``ACTION6@12,30``."""
    if isinstance(data, dict) and "x" in data and "y" in data:
        return f"{action}@{data['x']},{data['y']}"
    return action


def observation_signature(observation: Any) -> str:
    """Return a stable fingerprint of the game-relevant observation content.

    Hashes the board (``FrameDataRaw.frame`` or a ``grid``/``frame`` field),
    the game state, level progress, and available actions. Session ids,
    echoed action inputs, and timestamps are excluded, so identical boards
    in identical states share one signature.
    """
    encoded = json.dumps(
        observation_view(observation), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def click_targets(grid: list[list[int]] | None, *, limit: int = 96) -> list[tuple[int, int]]:
    """Return ``(x, y)`` click coordinates: visible objects first, then a coarse scan.

    The background is the most common colour. Every 4-connected component
    of another single colour contributes the member cell nearest its
    centroid. Smaller objects come first because buttons, switches, and
    selectable pieces are usually small. Some games only react to clicks
    inside large regions, so the centres of an 8x8 tiling follow as a
    fallback. ``x`` is the column and ``y`` the row, both restricted to the
    toolkit's 0..63 range.
    """
    if not grid:
        return [(32, 32)]
    height, width = len(grid), len(grid[0])
    background = Counter(cell for row in grid for cell in row).most_common(1)[0][0]
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int]] = []
    for r0 in range(height):
        for c0 in range(width):
            colour = grid[r0][c0]
            if colour == background or (r0, c0) in seen:
                continue
            members: list[tuple[int, int]] = []
            queue = deque([(r0, c0)])
            seen.add((r0, c0))
            while queue:
                r, c = queue.popleft()
                members.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and (nr, nc) not in seen
                        and grid[nr][nc] == colour
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            mean_r = sum(r for r, _ in members) / len(members)
            mean_c = sum(c for _, c in members) / len(members)
            r, c = min(
                members,
                key=lambda cell: ((cell[0] - mean_r) ** 2 + (cell[1] - mean_c) ** 2, cell),
            )
            components.append((len(members), colour, r, c))
    components.sort()
    targets = [(c, r) for _, _, r, c in components if r <= 63 and c <= 63]
    seen_targets = set(targets)
    for y in range(4, min(height, 64), 8):
        for x in range(4, min(width, 64), 8):
            if (x, y) not in seen_targets:
                targets.append((x, y))
    return targets[:limit] or [(min(32, width - 1), min(32, height - 1))]


EDGE_BAND = 3
"""Rows/columns this close to the border may hold step counters and status bars."""
VOLATILE_MIN_GRID = 16
"""Smaller boards (unit-test fixtures, toy games) are never masked."""


def _signature_of(view: dict[str, Any]) -> str:
    """Hash an observation view the same way :func:`observation_signature` does."""
    encoded = json.dumps(view, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class _Logged:
    """One observed transition, kept raw so signatures can be recomputed."""

    before: dict[str, Any]
    key: str
    action: str
    after: dict[str, Any]
    reward: float
    state_after: str | None
    cls: str | None = None


@dataclass
class _Graph:
    """Directed state/action graph plus per-action effect statistics."""

    edges: dict[str, dict[str, str]] = field(default_factory=dict)
    visits: Counter[tuple[str, str]] = field(default_factory=Counter)
    tried: dict[str, set[str]] = field(default_factory=dict)
    key_tries: Counter[str] = field(default_factory=Counter)
    key_changes: Counter[str] = field(default_factory=Counter)
    action_tries: Counter[str] = field(default_factory=Counter)
    action_changes: Counter[str] = field(default_factory=Counter)
    rewards: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)
    dead: set[str] = field(default_factory=set)
    seen: set[str] = field(default_factory=set)
    departures: Counter[str] = field(default_factory=Counter)
    class_tries: Counter[str] = field(default_factory=Counter)
    class_changes: Counter[str] = field(default_factory=Counter)
    key_class: dict[str, str] = field(default_factory=dict)

    def add(
        self,
        prior: str,
        key: str,
        action: str,
        nxt: str | None,
        reward: float | None,
        state_after: str | None,
        cls: str | None = None,
    ) -> None:
        """Record one transition ``prior --key--> nxt``; ``cls`` groups similar actions."""
        if cls is not None:
            self.key_class[key] = cls
            self.class_tries[cls] += 1
            if nxt is not None and nxt != prior:
                self.class_changes[cls] += 1
        self.seen.add(prior)
        self.departures[prior] += 1
        self.visits[(prior, key)] += 1
        self.tried.setdefault(prior, set()).add(key)
        self.key_tries[key] += 1
        self.action_tries[action] += 1
        if nxt is not None:
            self.seen.add(nxt)
            if nxt != prior:
                self.key_changes[key] += 1
                self.action_changes[action] += 1
            if state_after == GAME_OVER:
                # A game over usually means an exhausted step budget, not a bad
                # move, so it must not overwrite a live edge learned earlier.
                self.dead.add(nxt)
                self.edges.setdefault(prior, {}).setdefault(key, nxt)
            else:
                self.edges.setdefault(prior, {})[key] = nxt
        if reward is None:
            reward = {TERMINAL_WIN: 1.0, GAME_OVER: -1.0}.get(state_after or "", 0.0)
        if reward:
            total, count = self.rewards.get((prior, key), (0.0, 0))
            self.rewards[(prior, key)] = (total + float(reward), count + 1)


@dataclass
class ARC3ExplorationPolicy:
    """State-graph explorer that learns which pixels are noise and which actions matter.

    It uses no external model and no game-specific code. Three things make
    exploration efficient on real games:

    * **Volatile-pixel masking.** Most games draw a step counter or status
      bar along a border that changes on every action. Left in the state
      identity it makes every board look new, so the graph never closes a
      loop and no-op actions look effective. Border rows/columns that change
      on most transitions, or that are the only thing a transition changed,
      are masked out of state signatures and effect detection.
    * **Action-effect priors.** Every concrete action (including each click
      target) keeps a count of how often it changed the masked board, so
      actions that did something before are tried first in new states and
      clicks that never did anything are deferred.
    * **Persistent graph with frontier walks.** The graph survives RESETs:
      after a game over the agent walks the shortest known path back to the
      nearest state that still has untested actions instead of starting
      over.

    The graph is built from :meth:`observe_transition` feedback (the episode
    loop calls it every step). A fresh instance given only a saved history
    falls back to that history's recorded signatures. Scores are
    exploration priorities, not success probabilities.

    Args:
        max_click_targets: Upper bound on ACTION6 candidates per state.
        outcome_weight: Weight of observed reward when exploiting a state.
    """

    max_click_targets: int = 96
    outcome_weight: float = 1.0
    _log: list[_Logged] = field(default_factory=list, init=False)
    _graph: _Graph = field(default_factory=_Graph, init=False)
    _volatile: set[tuple[str, int]] = field(default_factory=set, init=False)
    _line_changes: Counter[tuple[str, int]] = field(default_factory=Counter, init=False)
    _band_only: Counter[tuple[str, int]] = field(default_factory=Counter, init=False)
    _grid_transitions: int = field(default=0, init=False)
    _candidate_cache: dict[str, list[Candidate]] = field(default_factory=dict, init=False)
    _observations_seen: set[str] = field(default_factory=set, init=False)
    _start_view: dict[str, Any] | None = field(default=None, init=False)
    _reset_pending: bool = field(default=True, init=False)

    @property
    def _outcomes(self) -> dict[tuple[str, str], tuple[float, int]]:
        """Explicit non-zero rewards observed per (state signature, action key)."""
        return self._graph.rewards

    @property
    def volatile_lines(self) -> list[tuple[str, int]]:
        """Masked border lines as ``("row"|"col", index)``; negative counts from the end."""
        return sorted(self._volatile)

    def signature(self, observation: Any) -> str:
        """Fingerprint an observation with the learned volatile pixels masked out.

        Equal to :func:`observation_signature` until a volatile line is found.
        """
        return self._view_signature(observation_view(observation))

    def _view_signature(self, view: dict[str, Any]) -> str:
        """Fingerprint a view, masking volatile lines of a large-enough grid."""
        grid = view.get("grid")
        if self._volatile and isinstance(grid, list) and _maskable(grid):
            masked = [list(row) for row in grid]
            height, width = len(masked), len(masked[0])
            for axis, index in self._volatile:
                if axis == "row":
                    masked[index % height] = [-1] * width
                else:
                    for row in masked:
                        row[index % width] = -1
            view = dict(view, grid=masked)
        return _signature_of(view)

    def _learn_volatile(self, before: Any, after: Any) -> bool:
        """Update border-line change statistics; return True if the mask grew."""
        if not (_maskable(before) and _maskable(after)):
            return False
        height, width = len(before), len(before[0])
        if len(after) != height or len(after[0]) != width:
            return False
        changed = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if before[r][c] != after[r][c]
        ]
        self._grid_transitions += 1
        lines: set[tuple[str, int]] = set()
        band_only = True
        for r, c in changed:
            if r < EDGE_BAND or r >= height - EDGE_BAND:
                lines.add(("row", r if r < EDGE_BAND else r - height))
            elif c < EDGE_BAND or c >= width - EDGE_BAND:
                lines.add(("col", c if c < EDGE_BAND else c - width))
            else:
                band_only = False
        for line in lines:
            self._line_changes[line] += 1
            if band_only:
                self._band_only[line] += 1
        grew = False
        for line in lines:
            if line in self._volatile:
                continue
            frequent = (
                self._grid_transitions >= 4
                and self._line_changes[line] * 2 >= self._grid_transitions
            )
            if frequent or self._band_only[line] >= 2:
                self._volatile.add(line)
                grew = True
        return grew

    def _rebuild(self) -> None:
        """Recompute every signature and the graph after the mask changed."""
        graph = _Graph()
        for item in self._log:
            graph.add(
                self._view_signature(item.before),
                item.key,
                item.action,
                self._view_signature(item.after),
                item.reward,
                item.state_after,
                item.cls,
            )
        self._graph = graph
        self._candidate_cache.clear()

    def _candidates(self, observation: Any, legal_actions: list[str]) -> list[Candidate]:
        """Expand legal actions into concrete candidates (one per click target)."""
        candidates: list[Candidate] = []
        targets: list[tuple[int, int]] | None = None
        for action in sorted(set(legal_actions)):
            if is_complex_action(action):
                if targets is None:
                    targets = click_targets(
                        observation_grid(observation), limit=self.max_click_targets
                    )
                grid = observation_grid(observation)
                for x, y in targets:
                    data = {"x": x, "y": y}
                    key = action_key(action, data)
                    cls = _action_class(grid, action, data)
                    if cls is not None:
                        self._graph.key_class[key] = cls
                    candidates.append((key, action, data))
            else:
                candidates.append((action, action, None))
        return candidates

    @staticmethod
    def _history_graph(previous_transitions: list[dict[str, Any]]) -> _Graph:
        """Rebuild a graph from a saved history's recorded signatures."""
        graph = _Graph()
        for transition in previous_transitions:
            prior = transition.get("state_signature")
            action = transition.get("action")
            nxt = transition.get("next_state_signature")
            if not (isinstance(prior, str) and isinstance(action, str)):
                continue
            key = transition.get("action_key") or action_key(action, transition.get("data"))
            reward = transition.get("reward")
            state_after = transition.get("terminal_state") or transition.get("state_after")
            graph.add(
                prior,
                key,
                action,
                nxt if isinstance(nxt, str) else None,
                float(reward) if isinstance(reward, (int, float)) else None,
                state_after if isinstance(state_after, str) else None,
            )
        return graph

    def choose_action(
        self,
        observation: Any,
        legal_actions: list[str],
        previous_transitions: list[dict[str, Any]],
    ) -> ARC3Decision:
        """Choose a promising untested action here, else walk toward the nearest frontier."""
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        graph = self._graph
        if not self._log and previous_transitions:
            graph = self._history_graph(previous_transitions)
            graph.rewards.update(
                {k: v for k, v in self._graph.rewards.items() if k not in graph.rewards}
            )
        if self._reset_pending or self._start_view is None:
            self._start_view = observation_view(observation)
            self._reset_pending = False
        state_key = self.signature(observation)
        candidates = self._candidates(observation, legal_actions)
        self._candidate_cache[state_key] = candidates

        def outcome(key: str) -> float:
            """Mean observed reward of an action in this state."""
            total, count = graph.rewards.get((state_key, key), (0.0, 0))
            return total / count if count else 0.0

        scored = []
        for index, (key, action, data) in enumerate(candidates):
            untested = graph.visits[(state_key, key)] == 0
            score = (
                self._novelty(graph, key, action, untested)
                + 0.5 * self._effect_prior(graph, key, action)
                + self.outcome_weight * outcome(key)
            )
            scored.append((score, index, key, action, data))

        best = max(scored, key=lambda item: (item[0], -item[1]))
        mode = "untested_here" if graph.visits[(state_key, best[2])] == 0 else "exploit_or_retry"
        if outcome(best[2]) > 0:
            mode = "exploit_reward"
        elif mode != "untested_here" or self._known_noop(graph, best[2]):
            step = self._route_to_frontier(state_key, graph)
            if step == RESET_ACTION:
                best, mode = (0.0, -1, RESET_ACTION, RESET_ACTION, None), "reset_to_frontier"
                self._reset_pending = True
            elif step is not None:
                match = next((item for item in scored if item[2] == step), None)
                if match is not None:
                    best, mode = match, "walk_to_frontier"
            else:
                best, mode = self._least_visited_successor(state_key, graph, scored), "cover"

        score, _, key, action, data = best
        reasoning = {
            "policy": "graph_frontier_exploration",
            "mode": mode,
            "state_signature": state_key,
            "state_novel": state_key not in graph.seen and state_key not in self._observations_seen,
            "local_state_action_visits": graph.visits[(state_key, key)],
            "exploration_score": round(score, 6),
            "score_semantics": "exploration_priority_plus_observed_reward_not_success_probability",
            "candidate_count": len(candidates),
            "masked_border_lines": len(self._volatile),
            "top_candidates": [
                {"action": item[2], "score": round(item[0], 6)}
                for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:5]
            ],
        }
        self._observations_seen.add(state_key)
        return ARC3Decision(action=action, data=dict(data) if data else None, reasoning=reasoning)

    @staticmethod
    def _known_noop(graph: _Graph, key: str) -> bool:
        """True when this concrete action was tried repeatedly and never changed the board."""
        return graph.key_tries[key] >= 2 and graph.key_changes[key] == 0

    def _novelty(self, graph: _Graph, key: str, action: str, untested: bool) -> float:
        """Exploration bonus: full for untested actions, small for known no-ops."""
        if not untested:
            return 0.0
        return 0.25 if self._known_noop(graph, key) else 1.0

    @staticmethod
    def _effect_prior(graph: _Graph, key: str, action: str) -> float:
        """Smoothed probability that this action changes the masked board."""
        action_rate = (graph.action_changes[action] + 1) / (graph.action_tries[action] + 2)
        cls = graph.key_class.get(key)
        class_rate = (
            (graph.class_changes[cls] + 2 * action_rate) / (graph.class_tries[cls] + 2)
            if cls is not None
            else action_rate
        )
        return (graph.key_changes[key] + 2 * class_rate) / (graph.key_tries[key] + 2)

    def _has_untested(self, graph: _Graph, state: str) -> bool:
        """Whether a known state still offers an untested, not-known-useless action."""
        candidates = self._candidate_cache.get(state)
        if candidates is None:
            return state not in graph.tried
        tried = graph.tried.get(state, set())
        return any(
            key not in tried and not self._known_noop(graph, key) for key, _a, _d in candidates
        )

    def _frontier_step(
        self, start: str, graph: _Graph, *, include_start: bool = False
    ) -> tuple[str | None, int] | None:
        """Shortest known path to a state with untested actions as ``(first key, length)``.

        Breadth-first search over the observed graph; edges into game-over
        states and self-loops are skipped. ``include_start`` lets the start
        state itself count as the frontier (length 0).
        """
        queue: deque[tuple[str, str | None, int]] = deque([(start, None, 0)])
        visited = {start}
        while queue:
            state, first, depth = queue.popleft()
            if (include_start or state != start) and self._has_untested(graph, state):
                return first, depth
            for key, nxt in sorted(graph.edges.get(state, {}).items()):
                if nxt in visited or nxt in graph.dead or nxt == state:
                    continue
                visited.add(nxt)
                queue.append((nxt, first if first is not None else key, depth + 1))
        return None

    def _route_to_frontier(self, state: str, graph: _Graph) -> str | None:
        """Next step toward the nearest frontier, treating RESET as an edge to the start.

        RESET is available from every state and returns to the current
        level's start, so the shortest route is ``min(d(state, F), 1 + d(start, F))``.
        Returns an action key, :data:`RESET_ACTION`, or None when no frontier is reachable.
        """
        direct = self._frontier_step(state, graph)
        via_reset = None
        if self._start_view is not None:
            start = self._view_signature(self._start_view)
            if start != state:
                via_reset = self._frontier_step(start, graph, include_start=True)
        if via_reset is not None and (direct is None or 1 + via_reset[1] < direct[1]):
            return RESET_ACTION
        return direct[0] if direct is not None else None

    @staticmethod
    def _least_visited_successor(
        state: str, graph: _Graph, scored: list[tuple[float, int, str, str, Any]]
    ) -> tuple[float, int, str, str, Any]:
        """Count-based coverage when no frontier is known: the least-tried live action here.

        Outcomes can depend on hidden state the board does not show, so a
        tried action is not fully known. Minimising ``N(s, a)`` (with
        self-loops and game-over edges penalised) cycles through every
        action instead of oscillating between two states.
        """
        edges = graph.edges.get(state, {})

        def rank(item: tuple[float, int, str, str, Any]) -> tuple[int, int]:
            """Visits plus a penalty for known self-loops and game-over edges."""
            nxt = edges.get(item[2])
            stuck = nxt is not None and (nxt == state or nxt in graph.dead)
            return (graph.visits[(state, item[2])] + 3 * int(stuck), item[1])

        return min(scored, key=rank)

    def observe_transition(
        self,
        observation: Any,
        action: str,
        next_observation: Any,
        terminal_state: str | None,
        *,
        data: dict[str, Any] | None = None,
        reward: float | None = None,
    ) -> None:
        """Record an observed transition and its explicit feedback.

        ``reward`` is the loop's level/terminal feedback. When omitted, WIN
        counts +1 and GAME_OVER -1; no reward is invented for intermediate
        frames.
        """
        before = observation_view(observation)
        after = observation_view(next_observation)
        if reward is None:
            reward = {TERMINAL_WIN: 1.0, GAME_OVER: -1.0}.get(terminal_state or "", 0.0)
        item = _Logged(
            before=before,
            key=action_key(action, data),
            action=action,
            after=after,
            reward=float(reward),
            state_after=after.get("state") or terminal_state,
            cls=_action_class(before.get("grid"), action, data),
        )
        self._log.append(item)
        if item.state_after == GAME_OVER:
            self._reset_pending = True
        levels_before, levels_after = before.get("levels_completed"), after.get("levels_completed")
        if (
            isinstance(levels_before, int)
            and isinstance(levels_after, int)
            and levels_after > levels_before
        ):
            self._start_view = after
        if self._learn_volatile(before.get("grid"), after.get("grid")):
            self._rebuild()
            return
        self._graph.add(
            self._view_signature(before),
            item.key,
            item.action,
            self._view_signature(after),
            item.reward,
            item.state_after,
            item.cls,
        )


def _action_class(grid: Any, action: str, data: dict[str, Any] | None) -> str | None:
    """Group a click with every other click on the same colour (hierarchical prior level)."""
    if not (isinstance(data, dict) and isinstance(grid, list) and grid):
        return None
    x, y = data.get("x"), data.get("y")
    if not (isinstance(x, int) and isinstance(y, int) and 0 <= y < len(grid)):
        return None
    row = grid[y]
    if not (isinstance(row, list) and 0 <= x < len(row)):
        return None
    return f"{action}#colour{row[x]}"


def _maskable(grid: Any) -> bool:
    """Whether a grid is large enough for border-line masking."""
    return (
        isinstance(grid, list)
        and len(grid) >= VOLATILE_MIN_GRID
        and isinstance(grid[0], list)
        and len(grid[0]) >= VOLATILE_MIN_GRID
    )

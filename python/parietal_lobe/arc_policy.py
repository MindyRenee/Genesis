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


@dataclass
class ARC3ExplorationPolicy:
    """Episode-local state-graph explorer with explicit uncertainty.

    It uses no external model. The graph is rebuilt from the supplied
    episode history (which is authoritative), so a fresh instance can
    resume from a saved transition list. Scores are exploration priorities,
    not success probabilities.

    Args:
        max_click_targets: Upper bound on ACTION6 candidates per state.
        outcome_weight: Weight of observed reward when exploiting a state.
    """

    max_click_targets: int = 96
    outcome_weight: float = 1.0
    _outcomes: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict, init=False)
    _observations_seen: set[str] = field(default_factory=set, init=False)

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
                for x, y in targets:
                    data = {"x": x, "y": y}
                    candidates.append((action_key(action, data), action, data))
            else:
                candidates.append((action, action, None))
        return candidates

    def choose_action(
        self,
        observation: Any,
        legal_actions: list[str],
        previous_transitions: list[dict[str, Any]],
    ) -> ARC3Decision:
        """Choose an untested action here, else walk toward the nearest frontier."""
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        state_key = observation_signature(observation)
        edges: dict[str, dict[str, str]] = {}
        visits: Counter[tuple[str, str]] = Counter()
        tried_by_action: Counter[str] = Counter()
        changed_by_action: Counter[str] = Counter()
        rewards: dict[tuple[str, str], tuple[float, int]] = {}
        dead_states: set[str] = set()
        seen = set(self._observations_seen)

        for transition in previous_transitions:
            prior = transition.get("state_signature")
            nxt = transition.get("next_state_signature")
            action = transition.get("action")
            if isinstance(prior, str):
                seen.add(prior)
            if isinstance(nxt, str):
                seen.add(nxt)
                if transition.get("state_after") == GAME_OVER:
                    dead_states.add(nxt)
            if not (isinstance(prior, str) and isinstance(action, str)):
                continue
            key = transition.get("action_key") or action_key(action, transition.get("data"))
            visits[(prior, key)] += 1
            tried_by_action[action] += 1
            if isinstance(nxt, str):
                edges.setdefault(prior, {})[key] = nxt
                if nxt != prior:
                    changed_by_action[action] += 1
            reward = transition.get("reward")
            if reward is None:
                terminal = transition.get("terminal_state") or transition.get("state_after")
                reward = {TERMINAL_WIN: 1.0, GAME_OVER: -1.0}.get(terminal)
            if reward:
                total, count = rewards.get((prior, key), (0.0, 0))
                rewards[(prior, key)] = (total + float(reward), count + 1)

        candidates = self._candidates(observation, legal_actions)

        def outcome(key: str) -> float:
            fallback = self._outcomes.get((state_key, key), (0.0, 0))
            total, count = rewards.get((state_key, key), fallback)
            return total / count if count else 0.0

        def change_rate(action: str) -> float:
            return (changed_by_action[action] + 1) / (tried_by_action[action] + 2)

        scored = []
        for index, (key, action, data) in enumerate(candidates):
            untested = visits[(state_key, key)] == 0
            score = (
                (1.0 if untested else 0.0)
                + 0.5 * change_rate(action)
                + self.outcome_weight * outcome(key)
            )
            scored.append((score, index, key, action, data))

        best = max(scored, key=lambda item: (item[0], -item[1]))
        mode = "untested_here" if visits[(state_key, best[2])] == 0 else "exploit_or_retry"
        if outcome(best[2]) > 0:
            mode = "exploit_reward"
        elif mode != "untested_here":
            step = self._frontier_step(state_key, edges, visits, dead_states, candidates)
            if step is not None:
                match = next(item for item in scored if item[2] == step)
                best, mode = match, "walk_to_frontier"

        score, _, key, action, data = best
        reasoning = {
            "policy": "graph_frontier_exploration",
            "mode": mode,
            "state_signature": state_key,
            "state_novel": state_key not in seen,
            "local_state_action_visits": visits[(state_key, key)],
            "exploration_score": round(score, 6),
            "score_semantics": "exploration_priority_plus_observed_reward_not_success_probability",
            "candidate_count": len(candidates),
            "top_candidates": [
                {"action": item[2], "score": round(item[0], 6)}
                for item in sorted(scored, key=lambda item: (-item[0], item[1]))[:5]
            ],
        }
        self._observations_seen.add(state_key)
        return ARC3Decision(action=action, data=dict(data) if data else None, reasoning=reasoning)

    @staticmethod
    def _frontier_step(
        start: str,
        edges: dict[str, dict[str, str]],
        visits: Counter[tuple[str, str]],
        dead_states: set[str],
        start_candidates: list[Candidate],
    ) -> str | None:
        """First action key on the shortest known path to a state with untested edges.

        A state counts as a frontier when it was reached but never acted
        in, or when fewer actions were tried there than the start state
        offers. Edges into game-over states and self-loops are skipped.
        """
        width = len(start_candidates)
        queue: deque[tuple[str, str | None]] = deque([(start, None)])
        visited = {start}
        while queue:
            state, first = queue.popleft()
            if state != start:
                tried = sum(1 for (s, _k) in visits if s == state)
                if state not in edges or tried < width:
                    return first
            for key, nxt in sorted(edges.get(state, {}).items()):
                if nxt in visited or nxt in dead_states or nxt == state:
                    continue
                visited.add(nxt)
                queue.append((nxt, first if first is not None else key))
        return None

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
        """Store explicit feedback for this observed state/action pair.

        ``reward`` is the loop's level/terminal feedback. When omitted, WIN
        counts +1 and GAME_OVER -1. Zero-reward transitions are not stored:
        no reward is invented for intermediate frames.
        """
        if reward is None:
            reward = {TERMINAL_WIN: 1.0, GAME_OVER: -1.0}.get(terminal_state or "", 0.0)
        if not reward:
            return
        state_key = observation_signature(observation)
        key = (state_key, action_key(action, data))
        total, count = self._outcomes.get(key, (0.0, 0))
        self._outcomes[key] = (total + float(reward), count + 1)
        self._observations_seen.add(state_key)
        self._observations_seen.add(observation_signature(next_observation))

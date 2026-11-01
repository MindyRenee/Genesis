"""A bounded, model-free exploration policy for ARC-AGI-3.

This is an autonomous action-selection baseline, not a task solver. It chooses
only currently legal controls, prioritizes unexplored state/action pairs, and
uses observed novelty and terminal outcomes to update its episode-local
preferences. It does not claim to infer game rules or guarantee progress.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from parietal_lobe.arc_play import ARC3Decision


def _plain(value: Any, depth: int = 0) -> Any:
    """Convert common observation structures into stable JSON-safe values."""
    if depth > 8:
        return type(value).__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"timestamp", "time", "elapsed"}
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist(), depth + 1)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {
            key: _plain(item, depth + 1)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    return type(value).__name__


def observation_signature(observation: Any) -> str:
    """Return a stable fingerprint for the visible observation."""
    encoded = json.dumps(
        _plain(observation), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ARC3ExplorationPolicy:
    """Episode-local novelty-seeking action selector with explicit uncertainty.

    It uses no external model. A selected action is exploratory: its score
    reflects lack of experience, not a prediction that the action is correct.
    State/action counts are reconstructed from the supplied episode history,
    so a fresh instance can resume from a saved transition list.
    """

    exploration_weight: float = 1.0
    _observations_seen: set[str] = field(default_factory=set, init=False)
    _action_counts: Counter[tuple[str, str]] = field(default_factory=Counter, init=False)
    _global_action_counts: Counter[str] = field(default_factory=Counter, init=False)

    def choose_action(
        self,
        observation: Any,
        legal_actions: list[str],
        previous_transitions: list[dict[str, Any]],
    ) -> ARC3Decision:
        """Choose the least-tried legal action for this observed state."""
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        state_key = observation_signature(observation)
        seen = set(self._observations_seen)
        action_counts: Counter[tuple[str, str]] = Counter()
        global_counts: Counter[str] = Counter()

        # Reconstruct policy evidence from the explicit episode history.
        for transition in previous_transitions:
            prior_key = transition.get("state_signature")
            action = transition.get("action")
            if isinstance(prior_key, str) and isinstance(action, str):
                action_counts[(prior_key, action)] += 1
                global_counts[action] += 1
            next_key = transition.get("next_state_signature")
            if isinstance(next_key, str):
                seen.add(next_key)

        is_novel_state = state_key not in seen
        scored: list[tuple[float, str, int]] = []
        for action in sorted(set(legal_actions)):
            local_count = action_counts[(state_key, action)]
            global_count = global_counts[action]
            # UCB-like exploration bonus: this is an exploration priority,
            # not a learned value estimate or expected reward.
            score = self.exploration_weight / math.sqrt(1 + local_count)
            score += 0.1 / math.sqrt(1 + global_count)
            scored.append((score, action, local_count))

        # Stable tie-break avoids hidden randomness and makes replays auditable.
        score, action, local_count = max(scored, key=lambda item: (item[0], item[1]))
        reasoning = {
            "policy": "bounded_novelty_exploration",
            "state_signature": state_key,
            "state_novel": is_novel_state,
            "local_state_action_visits": local_count,
            "exploration_score": round(score, 6),
            "score_semantics": "exploration_priority_not_success_probability",
            "candidate_scores": [
                {"action": candidate, "score": round(candidate_score, 6)}
                for candidate_score, candidate, _ in sorted(scored, key=lambda item: item[1])
            ],
        }
        self._observations_seen.add(state_key)
        return ARC3Decision(action=action, reasoning=reasoning)

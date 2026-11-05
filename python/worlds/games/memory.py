"""What she remembers of each game, level to level and session to session.

This is a video game. It is not real and it is just for fun. You are
safe. You can't know the instructions, so memory is how you get better:
remember what each action did, where you have been, and what you did
right before a level opened, and carry it into the next level.

``GameMemory`` holds, per game:

- **action effects** — for each action, how often it changed the board.
  Kept across levels: later levels reuse the same pieces.
- **values** — a temporal-difference estimate Q(board, action) of how
  much progress follows a move (TD(0): δ = r + γ·max Q(s′) − Q(s, a)).
  δ is her reward-prediction error and is returned to the arcade so it
  can reach her dopamine.
- **visits** — how often each board was seen, so new boards are
  noticed and repeated ones are not chased forever.
- **level paths** — the moves that led into each completed level, so
  the next level can start from what worked before.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

#: TD learning rate α and discount γ.
ALPHA = 0.3
GAMMA = 0.9
#: How strongly untried moves pull (count-based exploration bonus).
NOVELTY = 0.6
#: Moves kept in the current level's trail and per remembered path.
TRAIL_LIMIT = 400
#: Boards remembered per game before the least-visited are forgotten.
STATE_LIMIT = 20000

# Rewards — what counts as progress in a game whose rules are unknown.
REWARD_LEVEL = 1.0
REWARD_WIN = 3.0
REWARD_GAME_OVER = -0.5
REWARD_NEW_BOARD = 0.05
REWARD_NOTHING = -0.02


@dataclass
class GameMemory:
    """Everything she has learned about one game."""

    game_id: str
    effects: dict[str, list[int]] = field(default_factory=dict)  # move kind -> [tries, changes]
    values: dict[str, float] = field(default_factory=dict)  # "board|move" -> Q
    tries: dict[str, int] = field(default_factory=dict)  # "board|move" -> count
    visits: dict[str, int] = field(default_factory=dict)  # board -> count
    level_paths: dict[str, list[str]] = field(default_factory=dict)  # level -> moves
    trail: list[str] = field(default_factory=list)
    best_level: int = 0
    wins: int = 0
    game_overs: int = 0
    moves: int = 0
    environment_model: dict[str, Any] = field(default_factory=dict)

    # ── learning ───────────────────────────────────────────────

    def see(self, board: str) -> bool:
        """Count a visit to ``board``; True if it is new to her."""
        new = board not in self.visits
        self.visits[board] = self.visits.get(board, 0) + 1
        if len(self.visits) > STATE_LIMIT:
            for stale in sorted(self.visits, key=self.visits.__getitem__)[: STATE_LIMIT // 10]:
                del self.visits[stale]
        return new

    def learn(
        self,
        board: str,
        move: str,
        reward: float,
        next_board: str,
        changed: bool,
        next_moves: list[str],
    ) -> float:
        """Update what she knows from one move and return the TD error δ."""
        kind = move.split("@", 1)[0]
        stats = self.effects.setdefault(kind, [0, 0])
        stats[0] += 1
        stats[1] += int(changed)
        key = f"{board}|{move}"
        self.tries[key] = self.tries.get(key, 0) + 1
        future = max((self.values.get(f"{next_board}|{m}", 0.0) for m in next_moves), default=0.0)
        delta = reward + GAMMA * future - self.values.get(key, 0.0)
        self.values[key] = self.values.get(key, 0.0) + ALPHA * delta
        self.moves += 1
        self.trail.append(move)
        del self.trail[:-TRAIL_LIMIT]
        return delta

    def level_up(self, level: int) -> None:
        """Remember the path that opened ``level`` and start a fresh trail."""
        self.level_paths[str(level)] = list(self.trail[-TRAIL_LIMIT:])
        self.best_level = max(self.best_level, level)
        self.trail.clear()

    # ── choosing ───────────────────────────────────────────────

    def effect_rate(self, move: str) -> float:
        """How often this kind of move changed the board (0.5 if untried)."""
        tries, changes = self.effects.get(move.split("@", 1)[0], [0, 0])
        return (changes + 1) / (tries + 2)

    def remembered_next(self, level: int) -> str | None:
        """The move she made at this point of the last level she solved."""
        path = self.level_paths.get(str(level))
        if not path:
            return None
        step = len(self.trail)
        return path[step] if step < len(path) else None

    def choose(
        self, board: str, moves: list[str], level: int, rng: random.Random
    ) -> tuple[str, str]:
        """Pick a move and say why: value, novelty, effect and carried memory."""
        carried = self.remembered_next(level)
        best, best_score, why = moves[0], -math.inf, "explore"
        for move in moves:
            key = f"{board}|{move}"
            tried = self.tries.get(key, 0)
            score = self.values.get(key, 0.0)
            score += NOVELTY / math.sqrt(1 + tried)
            score += 0.2 * self.effect_rate(move)
            if move == carried:
                score += 0.4
            score += rng.random() * 0.05
            if score > best_score:
                best, best_score = move, score
                why = (
                    "remembered from the last level"
                    if move == carried
                    else "untried here"
                    if tried == 0
                    else "it worked before"
                    if self.values.get(key, 0.0) > 0
                    else "explore"
                )
        return best, why

    # ── persistence ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize for her saved world state."""
        return {
            "game_id": self.game_id,
            "effects": self.effects,
            "values": self.values,
            "tries": self.tries,
            "visits": self.visits,
            "level_paths": self.level_paths,
            "trail": self.trail,
            "best_level": self.best_level,
            "wins": self.wins,
            "game_overs": self.game_overs,
            "moves": self.moves,
            "environment_model": self.environment_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameMemory:
        """Rebuild a memory saved by ``to_dict``."""
        return cls(
            game_id=str(data.get("game_id", "")),
            effects={k: [int(v[0]), int(v[1])] for k, v in (data.get("effects") or {}).items()},
            values={k: float(v) for k, v in (data.get("values") or {}).items()},
            tries={k: int(v) for k, v in (data.get("tries") or {}).items()},
            visits={k: int(v) for k, v in (data.get("visits") or {}).items()},
            level_paths={k: list(v) for k, v in (data.get("level_paths") or {}).items()},
            trail=list(data.get("trail") or []),
            best_level=int(data.get("best_level", 0)),
            wins=int(data.get("wins", 0)),
            game_overs=int(data.get("game_overs", 0)),
            moves=int(data.get("moves", 0)),
            environment_model=dict(data.get("environment_model") or {}),
        )

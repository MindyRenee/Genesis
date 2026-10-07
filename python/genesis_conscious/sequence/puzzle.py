"""Sequence — pattern continuation as a task family.

"Which bead comes next?" A line of positions carries marks (colors,
symbols); a hidden rule governs the line — the toy's pattern — and the
first positions come pre-filled as scaffolding. The agent fills the
rest from a token pool.

The honest signal is per-placement acceptance: the world knows the
rule, the agent never sees it. What the agent *can* perceive is a
candidate's lookback — the distance to the nearest earlier filled
position holding the same mark. Discovering which lookback predicts
acceptance IS discovering the period — the rule behind the pattern
rather than the pattern itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# Marks are the toddler palette by convention; an offered spec may use
# any marks at all — the rule machinery doesn't care what a mark is.
_MARKS = ("red", "blue", "yellow", "green", "purple", "orange")

# How far back a candidate looks for a matching mark. Periodic rules
# within this range are discoverable; longer periods blur into "no
# match" guesses.
_MAX_LOOKBACK = 6


@dataclass(frozen=True)
class Token:
    """One bead in the pool: an id and the mark it carries."""

    token_id: int
    mark: str


@dataclass
class Placement:
    """One candidate move: a token into an empty position."""

    token_id: int
    pos: int
    lookback: int  # smallest filled distance back holding this mark; 0 = none


class PatternLine:
    """The bead string: a hidden rule, scaffolding, and a pool."""

    def __init__(
        self,
        expected: list[str],
        cells: list[str | None],
        pool: list[Token],
    ) -> None:
        self.expected = list(expected)
        self.cells: list[str | None] = list(cells)
        self.pool: dict[int, Token] = {t.token_id: t for t in pool}

    @classmethod
    def generate(
        cls,
        period: int,
        length: int,
        rng: random.Random,
        scaffold: int | None = None,
        decoys: int = 0,
        pattern: list[str] | None = None,
    ) -> PatternLine:
        """A pattern line with the first ``scaffold`` positions shown.

        Default scaffolding is one full period plus one — enough to
        show the repeat starting, never enough to give the answer.
        ``pattern`` fixes the hidden rule explicitly; otherwise the
        generator samples one. Pool decoys mix off-pattern marks
        (never right) with in-pattern marks (right only where the
        period wants them — the near-misses).
        """
        if pattern is None:
            pattern = rng.sample(_MARKS, min(period, len(_MARKS)))
        if scaffold is None:
            scaffold = min(period + 1, length - 1)
        expected = [pattern[i % len(pattern)] for i in range(length)]
        cells: list[str | None] = expected[:scaffold] + [None] * (
            length - scaffold
        )
        pool = [Token(i, m) for i, m in enumerate(expected[scaffold:])]
        for _ in range(decoys):
            mark = rng.choice(list(_MARKS) + list(pattern))
            pool.append(Token(len(pool), mark))
        rng.shuffle(pool)
        return cls(expected, cells, pool)

    # ── World queries ─────────────────────────────────────────

    def mismatches(self) -> int:
        """Unfilled positions — zero means solved."""
        return sum(1 for c in self.cells if c is None)

    def complete(self) -> bool:
        return self.mismatches() == 0

    def lookback_of(self, mark: str, pos: int) -> int:
        """Distance to the nearest earlier filled position holding
        ``mark`` — the perceptible feature of a candidate. 0 = none."""
        for k in range(1, _MAX_LOOKBACK + 1):
            back = pos - k
            if back < 0:
                break
            if self.cells[back] == mark:
                return k
        return 0

    def candidates(self) -> list[Placement]:
        """Every pool token against every empty position."""
        out: list[Placement] = []
        for pos, cell in enumerate(self.cells):
            if cell is not None:
                continue
            for token in self.pool.values():
                out.append(
                    Placement(
                        token.token_id,
                        pos,
                        self.lookback_of(token.mark, pos),
                    )
                )
        return out

    def goal_state(self) -> dict[str, Any]:
        return {"cells.unfilled": self.mismatches()}

    def state(self) -> dict[str, Any]:
        return {
            "cells.unfilled": self.mismatches(),
            "cells.filled": sum(1 for c in self.cells if c is not None),
            "pool.count": len(self.pool),
        }

    # ── Actions ───────────────────────────────────────────────

    def place(self, token_id: int, pos: int) -> dict[str, Any] | None:
        """Set a token. Accepted only when it continues the hidden
        rule — the environment checks, the agent never sees why."""
        token = self.pool.get(token_id)
        if token is None or pos < 0 or pos >= len(self.cells):
            return None
        if self.cells[pos] is not None:
            return None
        lookback = self.lookback_of(token.mark, pos)
        accepted = token.mark == self.expected[pos]
        if accepted:
            self.cells[pos] = token.mark
            del self.pool[token_id]
        return {
            "accepted": accepted,
            "lookback": lookback,
            "delta_unfilled": -1 if accepted else 0,
        }

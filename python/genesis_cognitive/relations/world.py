"""Relations — arranging named things so stated relations hold.

The kindergarten worksheet "put the star left of the moon, next to
the sun" — and, with attribute relations, seriation ("line the cups
up fewest to most"). A row of positions, a pool of named things, and
a list of relational goals. The model is honest: goals are checked
against the board, never hidden inside the placement rules — the
agent discovers that satisfying *stated* relations is the task.

Relation vocabulary (binary, between named things):
  left_of / right_of — position order on the row
  next_to / apart    — adjacency
  ordered            — positional order agrees with attribute order:
                       pos(a) < pos(b) iff attr(a) < attr(b). This is
                       seriation: "the cups go fewest to most."
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import permutations
from typing import Any

# Object names for generated arrangements — distinct tokens, not the
# answer. Offered specs name their own things.
_NAMES = ("star", "moon", "sun", "cloud", "tree", "boat", "bell", "leaf")
_RELS = ("left_of", "right_of", "next_to", "apart")


@dataclass(frozen=True)
class Thing:
    """A named thing with optional comparable attributes."""

    thing_id: int
    name: str
    attrs: tuple[tuple[str, int], ...] = ()

    def get(self, attr: str) -> int | None:
        return dict(self.attrs).get(attr)


@dataclass(frozen=True)
class Goal:
    """One stated relation the arrangement must satisfy."""

    rel: str
    a: str
    b: str
    attr: str | None = None


@dataclass
class Placement:
    """One candidate move: a thing into a position."""

    thing: str
    pos: int
    agrees: int  # currently-decidable goals this placement satisfies
    breaks: int  # currently-decidable goals this placement violates


class Arrangement:
    """The row: positions to fill and relations to satisfy."""

    def __init__(
        self,
        positions: int,
        things: list[Thing],
        goals: list[Goal],
    ) -> None:
        self.positions = positions
        self.things = list(things)
        self.goals = list(goals)
        self.pool: dict[str, Thing] = {t.name: t for t in things}
        self.board: dict[int, Thing] = {}  # pos -> thing

    @classmethod
    def generate(
        cls,
        n_objects: int,
        n_goals: int,
        rng: random.Random,
        *,
        with_attr: str | None = None,
    ) -> Arrangement:
        """A guaranteed-solvable arrangement.

        A hidden row order is sampled first, then goals are chosen
        that the hidden order satisfies — so at least one solution
        always exists. ``with_attr`` gives every thing a rank along
        the hidden order, enabling seriation goals.
        """
        names = rng.sample(_NAMES, min(n_objects, len(_NAMES)))
        order = list(names)
        rng.shuffle(order)
        rank = {name: i for i, name in enumerate(order)}
        pos_of = {name: i for i, name in enumerate(order)}
        things = [
            Thing(
                i,
                name,
                ((with_attr, rank[name]),) if with_attr else (),
            )
            for i, name in enumerate(names)
        ]
        goals: list[Goal] = []
        tries = 0
        while len(goals) < n_goals and tries < 300:
            tries += 1
            a, b = rng.sample(names, 2)
            if with_attr and rng.random() < 0.4:
                g = Goal("ordered", a, b, with_attr)
            else:
                pa, pb = pos_of[a], pos_of[b]
                rel = rng.choice(_RELS)
                if not cls._pos_holds(rel, pa, pb):
                    continue
                g = Goal(rel, a, b)
            if g not in goals:
                goals.append(g)
        rng.shuffle(things)
        return cls(len(names), things, goals)

    # ── Relation checking ─────────────────────────────────────

    @staticmethod
    def _pos_holds(rel: str, pa: int, pb: int) -> bool:
        if rel == "left_of":
            return pa < pb
        if rel == "right_of":
            return pa > pb
        if rel == "next_to":
            return abs(pa - pb) == 1
        if rel == "apart":
            return abs(pa - pb) > 1
        return False

    def goal_state_of(self, goal: Goal, board: dict[int, Thing]) -> str:
        """'yes' | 'no' | 'open' — satisfied, violated, or not yet
        decidable on a (possibly hypothetical) board."""
        pa = pb = None
        for pos, thing in board.items():
            if thing.name == goal.a:
                pa = pos
            if thing.name == goal.b:
                pb = pos
        if pa is None or pb is None:
            return "open"
        if goal.rel == "ordered":
            va = board[pa].get(goal.attr or "")
            vb = board[pb].get(goal.attr or "")
            if va is None or vb is None or va == vb:
                return "no"
            return "yes" if (pa < pb) == (va < vb) else "no"
        return "yes" if self._pos_holds(goal.rel, pa, pb) else "no"

    def mismatches(self) -> int:
        """Violated goals + unplaced things — zero means solved."""
        bad = sum(
            1
            for g in self.goals
            if self.goal_state_of(g, self.board) == "no"
        )
        return bad + len(self.pool)

    def complete(self) -> bool:
        return not self.pool and self.mismatches() == 0

    def violations(self) -> list[Goal]:
        """The goals currently violated on the board."""
        return [
            g
            for g in self.goals
            if self.goal_state_of(g, self.board) == "no"
        ]

    def candidates(self) -> list[Placement]:
        """Every (thing, empty position) with its decidable-goal tally."""
        out: list[Placement] = []
        empty = [p for p in range(self.positions) if p not in self.board]
        for thing in self.pool.values():
            for pos in empty:
                hypo = dict(self.board)
                hypo[pos] = thing
                agrees = breaks = 0
                for g in self.goals:
                    if g.a != thing.name and g.b != thing.name:
                        continue
                    state = self.goal_state_of(g, hypo)
                    if state == "yes":
                        agrees += 1
                    elif state == "no":
                        breaks += 1
                out.append(Placement(thing.name, pos, agrees, breaks))
        return out

    def goal_state(self) -> dict[str, Any]:
        return {"goals.unsatisfied": self.mismatches()}

    def state(self) -> dict[str, Any]:
        return {
            "goals.unsatisfied": self.mismatches(),
            "things.placed": len(self.board),
            "things.pool": len(self.pool),
        }

    # ── Actions ───────────────────────────────────────────────

    def place(self, name: str, pos: int) -> dict[str, Any] | None:
        """Set a thing at a position. Always succeeds — the world's
        honesty is in the goal check, not the placement rule."""
        thing = self.pool.get(name)
        if thing is None or not 0 <= pos < self.positions:
            return None
        if pos in self.board:
            return None
        before = self.mismatches()
        self.board[pos] = thing
        del self.pool[name]
        return {
            "placed": True,
            "pos": pos,
            "delta_unsatisfied": self.mismatches() - before,
        }

    def remove(self, pos: int) -> dict[str, Any] | None:
        """Lift a thing back into the pool — arrangements are tentative."""
        thing = self.board.get(pos)
        if thing is None:
            return None
        before = self.mismatches()
        del self.board[pos]
        self.pool[thing.name] = thing
        return {
            "removed": thing.name,
            "delta_unsatisfied": self.mismatches() - before,
        }

    def swap(self, pos_a: int, pos_b: int) -> dict[str, Any] | None:
        """Exchange two occupied positions — "switch these two" is a
        single move a child makes; it is also the only way to repair
        a violation on a full row, where lifting one party just frees
        the same slot it came from."""
        a, b = self.board.get(pos_a), self.board.get(pos_b)
        if a is None or b is None:
            return None
        before = self.mismatches()
        self.board[pos_a], self.board[pos_b] = b, a
        return {
            "swapped": (a.name, b.name),
            "delta_unsatisfied": self.mismatches() - before,
        }

    def solvable(self, max_objects: int = 7) -> bool:
        """Brute-force: does some row satisfy every goal?"""
        names = [t.name for t in self.things]
        if len(names) > max_objects or self.positions < len(names):
            return False
        for perm in permutations(range(self.positions), len(names)):
            board = {
                pos: self._by_name(n)
                for n, pos in zip(names, perm, strict=True)
            }
            if all(
                self.goal_state_of(g, board) == "yes" for g in self.goals
            ):
                return True
        return False

    def _by_name(self, name: str) -> Thing:
        return next(t for t in self.things if t.name == name)

"""Quantities — "make exactly N" from groups of things.

The kindergarten worksheet "give me 5 blocks" / "which groups make
10": a pool of groups carrying perceptible counts, a target, and a
basket. Adding a group that would overshoot the target is *rejected*
— the honest constraint is the capacity itself. Choosing groups that
fit but leave a remainder nothing can fill dead-ends the episode;
the only repair is lifting a group back out. Composition under a
budget — the seed of arithmetic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any


@dataclass(frozen=True)
class Group:
    """A countable group of things — the perceptible quantity."""

    group_id: int
    name: str
    count: int


@dataclass
class Option:
    """One candidate move: a group into the basket."""

    name: str
    count: int
    fits: bool  # count <= remaining capacity
    exact: bool  # count == remaining (closes the goal)
    dead: bool  # fits, but leaves a remainder nothing can fill


class Basket:
    """The worksheet: fill the basket to exactly the target."""

    def __init__(self, target: int, groups: list[Group]) -> None:
        self.target = target
        self.groups = list(groups)
        self.pool: dict[str, Group] = {g.name: g for g in groups}
        self.basket: dict[str, Group] = {}

    @classmethod
    def generate(
        cls, n_groups: int, target: int, rng: random.Random
    ) -> Basket:
        """A guaranteed-solvable worksheet: partition the target into
        a hidden solution, then pad the pool with decoy groups."""
        # Partition target into 2-4 positive parts.
        k = min(rng.randint(2, 3), target)
        cuts = sorted(rng.sample(range(1, target), k - 1))
        parts = [
            b - a
            for a, b in zip([0, *cuts], [*cuts, target], strict=True)
        ]
        groups = [
            Group(i, f"group_{i}", c) for i, c in enumerate(parts)
        ]
        # Decoys: counts that don't trivially solve on their own.
        i = len(groups)
        while len(groups) < max(n_groups, len(parts)):
            decoy = rng.randint(1, max(target - 1, 1))
            if decoy != target or rng.random() < 0.3:
                groups.append(Group(i, f"group_{i}", decoy))
                i += 1
        rng.shuffle(groups)
        return cls(target, groups)

    # ── Queries ───────────────────────────────────────────────

    def basket_sum(self) -> int:
        return sum(g.count for g in self.basket.values())

    def remaining(self) -> int:
        return self.target - self.basket_sum()

    def complete(self) -> bool:
        return self.basket_sum() == self.target

    def mismatches(self) -> int:
        """Distance to the target — zero means solved."""
        return abs(self.remaining())

    def candidates(self) -> list[Option]:
        """Every pool group with its perceptible fit features."""
        room = self.remaining()
        out = []
        for g in self.pool.values():
            after = room - g.count
            dead = 0 < after and not any(
                o.count <= after
                for o in self.pool.values()
                if o.name != g.name
            )
            out.append(
                Option(
                    name=g.name,
                    count=g.count,
                    fits=g.count <= room,
                    exact=g.count == room,
                    dead=dead,
                )
            )
        return out

    def dead_end(self) -> bool:
        """Basket under target with no pool group that fits."""
        room = self.remaining()
        return (
            0 < room
            and not any(g.count <= room for g in self.pool.values())
        )

    def state(self) -> dict[str, Any]:
        return {
            "basket.sum": self.basket_sum(),
            "basket.remaining": self.remaining(),
            "groups.pool": len(self.pool),
        }

    def goal_state(self) -> dict[str, Any]:
        return {"basket.remaining": self.remaining()}

    # ── Actions ───────────────────────────────────────────────

    def add(self, name: str) -> dict[str, Any] | None:
        """Try a group. Overflow is rejected — the capacity is real.
        Returns None only for an unknown or already-used name."""
        group = self.pool.get(name)
        if group is None:
            return None
        if group.count > self.remaining():
            return {"accepted": False, "overflow": group.count - self.remaining()}
        self.basket[name] = group
        del self.pool[name]
        return {
            "accepted": True,
            "sum": self.basket_sum(),
            "remaining": self.remaining(),
        }

    def remove(self, name: str) -> dict[str, Any] | None:
        """Lift a group back into the pool."""
        group = self.basket.get(name)
        if group is None:
            return None
        del self.basket[name]
        self.pool[name] = group
        return {"removed": name, "remaining": self.remaining()}

    def empty(self) -> dict[str, Any]:
        """Tip the basket out — start the composition over."""
        n = len(self.basket)
        self.pool.update(self.basket)
        self.basket.clear()
        return {"dumped": n, "remaining": self.remaining()}

    def solvable(self) -> bool:
        """Subset-sum: does some combination of groups hit the target?"""
        counts = [g.count for g in self.groups]
        for r in range(1, len(counts) + 1):
            for combo in combinations(counts, r):
                if sum(combo) == self.target:
                    return True
        return False

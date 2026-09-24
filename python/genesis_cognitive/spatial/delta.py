"""Delta perception — describing what a task *does* before solving it.

Each transform proposer used to re-derive its own input→output
conditions with hand-coded gates, and a wrong gate silently removed a
rule from the search space entirely. This module is the shared
perceptual layer above that: it describes the *change itself* — which
cells were added, removed, or recolored, and what geometric relation
the added cells bear to the input's existing structure.

Proposers then act on the perceived relation — "the output is the
input plus its own mirror image" is a thing she can *see*, not a gate
condition a rule must pass to be considered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .grid import Grid

Example = tuple[Grid, Grid]

# Geometric relations an added cell can bear to an existing cell.
# Each maps (r, c, height, width) → the cell's image under the relation.
_DELTA_RELATIONS: dict[str, Callable[[int, int, int, int], tuple[int, int]]] = {
    "mirror_v": lambda r, c, h, w: (r, w - 1 - c),
    "mirror_h": lambda r, c, h, w: (h - 1 - r, c),
    "rot180": lambda r, c, h, w: (h - 1 - r, w - 1 - c),
    "diag": lambda r, c, h, w: (c, r),
}


@dataclass(frozen=True)
class PairDelta:
    """The perceived change between one input and its output."""

    same_shape: bool
    added: frozenset[tuple[int, int, int]]   # (r, c, color) — bg → colored
    removed: frozenset[tuple[int, int]]      # colored → bg
    recolored: frozenset[tuple[int, int, int, int]]  # (r, c, old, new)

    @property
    def pure_addition(self) -> bool:
        """Output strictly contains the input: only bg cells painted."""
        return bool(self.added) and not self.removed and not self.recolored

    @property
    def pure_removal(self) -> bool:
        return bool(self.removed) and not self.added and not self.recolored

    @property
    def pure_recolor(self) -> bool:
        return bool(self.recolored) and not self.added and not self.removed

    @property
    def unchanged(self) -> bool:
        return not self.added and not self.removed and not self.recolored


@dataclass
class TaskDelta:
    """Aggregated perception of a whole task across its train pairs."""

    pairs: list[PairDelta] = field(default_factory=list)

    @property
    def all_same_shape(self) -> bool:
        return bool(self.pairs) and all(p.same_shape for p in self.pairs)

    @property
    def pure_addition(self) -> bool:
        return self.all_same_shape and all(p.pure_addition for p in self.pairs)

    @property
    def pure_removal(self) -> bool:
        return self.all_same_shape and all(p.pure_removal for p in self.pairs)

    @property
    def pure_recolor(self) -> bool:
        return self.all_same_shape and all(p.pure_recolor for p in self.pairs)

    def added_relations(self, examples: list[Example]) -> set[str]:
        """Geometric relations that explain every pair's added cells.

        A relation qualifies when, on every training pair, painting each
        non-background input cell's image under the relation onto the
        input reproduces the output exactly — i.e. the added cells ARE
        the input's structure seen through that relation.
        """
        if not self.pure_addition:
            return set()
        consistent: set[str] = set()
        for name, rel in _DELTA_RELATIONS.items():
            if all(
                _completes_with(inp, out, rel) for inp, out in examples
            ):
                consistent.add(name)
        return consistent


def perceive_pair(inp: Grid, out: Grid) -> PairDelta:
    """Describe the cell-level change from one grid to another."""
    if inp.shape != out.shape:
        return PairDelta(
            same_shape=False,
            added=frozenset(),
            removed=frozenset(),
            recolored=frozenset(),
        )
    bg = inp.most_common_color()
    added: set[tuple[int, int, int]] = set()
    removed: set[tuple[int, int]] = set()
    recolored: set[tuple[int, int, int, int]] = set()
    for r, c, v in inp.iter_cells():
        ov = out.at(r, c)
        if v == ov:
            continue
        if v == bg:
            added.add((r, c, ov))
        elif ov == bg:
            removed.add((r, c))
        else:
            recolored.add((r, c, v, ov))
    return PairDelta(
        same_shape=True,
        added=frozenset(added),
        removed=frozenset(removed),
        recolored=frozenset(recolored),
    )


def perceive_task(examples: list[Example]) -> TaskDelta:
    """Perceive the change across all of a task's training pairs."""
    return TaskDelta(pairs=[perceive_pair(inp, out) for inp, out in examples])


def _completes_with(inp: Grid, out: Grid, rel) -> bool:
    """Does painting each non-bg cell's image under ``rel`` yield out?"""
    bg = inp.most_common_color()
    rows = inp.to_lists()
    for r, c, v in inp.iter_cells():
        if v == bg:
            continue
        nr, nc = rel(r, c, inp.height, inp.width)
        if not (0 <= nr < inp.height and 0 <= nc < inp.width):
            return False
        if rows[nr][nc] == bg:
            rows[nr][nc] = v
    return Grid(tuple(tuple(row) for row in rows)) == out

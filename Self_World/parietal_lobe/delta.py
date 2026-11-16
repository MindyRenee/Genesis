"""Delta perception — describing what *changed* before explaining how.

This is the change channel of parietal perception, not a game solver.
Given two perceptions of related structure — a scene before and after
some event — it computes the *difference itself*: which cells appeared,
which vanished, which changed color, and what geometric relation the
appeared cells bear to the structure already present.

Neurally, this is the surprise channel of predictive coding. The
parietal cortex (LIP in the monkey, its human homologs) encodes
*surprise* — the Shannon information of an observation under the
current generative model, I_S = −log P(o | M) — separately from the
model *update*, which ACC performs and which is measured by the
Kullback–Leibler divergence between posterior and prior
(O'Reilly et al., 2013). This module produces the parietal half of
that pair: a precise, structured description of the residual
δ = observed − predicted. What to do with it — whether to revise a
held rule — is downstream, in the inductive reasoner.

Geometrically, the relations an added cell can bear to an existing
cell are elements of the dihedral group D₄ — the symmetry group of the
square lattice — acting on grid coordinates. "The output is the input
plus its own mirror image" is literally the claim that the added cells
lie in the orbit of the input's cells under a D₄ element: a thing she
can *see*, not a gate a rule must pass.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .grid import Grid

#: An observed transition: (before, after). Not a task — any pair of
#: perceptions where the second is understood as the first changed.
Transition = tuple[Grid, Grid]

# Geometric relations an added cell can bear to an existing cell —
# elements of D₄ acting on (r, c) given lattice height and width.
# Each maps (r, c, height, width) → the cell's image under the element.
_DELTA_RELATIONS: dict[str, Callable[[int, int, int, int], tuple[int, int]]] = {
    "mirror_v": lambda r, c, h, w: (r, w - 1 - c),
    "mirror_h": lambda r, c, h, w: (h - 1 - r, c),
    "rot180": lambda r, c, h, w: (h - 1 - r, w - 1 - c),
    "diag": lambda r, c, h, w: (c, r),
}


@dataclass(frozen=True)
class ChangeDelta:
    """The perceived change of one observed transition.

    The residual of perception: the set difference between what was
    seen and what came next, decomposed into the three primitive
    change types — appearance, disappearance, recolor.
    """

    same_shape: bool
    added: frozenset[tuple[int, int, int]]  # (r, c, color) — bg → colored
    removed: frozenset[tuple[int, int]]  # colored → bg
    recolored: frozenset[tuple[int, int, int, int]]  # (r, c, old, new)

    @property
    def pure_addition(self) -> bool:
        """The after strictly contains the before: only bg cells painted."""
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

    def surprise(self, cells: int) -> float:
        """Shannon surprise of this change, in bits.

        Under the null model that every cell was equally likely to be
        the one to change, a change touching k of N cells carries
        I_S = −log₂(k/N) bits of surprise. ``cells`` is N — the number
        of cells in the perceived scene. A change that touches nothing
        is fully expected (0 bits); a sweeping change approaches
        log₂ N under this uniform prior.
        """
        k = len(self.added) + len(self.removed) + len(self.recolored)
        if k == 0 or cells <= 0:
            return 0.0
        return -math.log2(min(k, cells) / cells)


@dataclass
class ChangeProfile:
    """Aggregated perceived change across repeated observations.

    When the same kind of transition is observed several times, the
    profile holds the per-transition deltas and answers conjunctive
    questions — did every instance only add? — which is the perceptual
    evidence an operator inference acts on.
    """

    deltas: list[ChangeDelta] = field(default_factory=list)

    @property
    def all_same_shape(self) -> bool:
        return bool(self.deltas) and all(d.same_shape for d in self.deltas)

    @property
    def pure_addition(self) -> bool:
        return self.all_same_shape and all(d.pure_addition for d in self.deltas)

    @property
    def pure_removal(self) -> bool:
        return self.all_same_shape and all(d.pure_removal for d in self.deltas)

    @property
    def pure_recolor(self) -> bool:
        return self.all_same_shape and all(d.pure_recolor for d in self.deltas)

    def added_relations(self, transitions: list[Transition]) -> set[str]:
        """D₄ elements whose orbit explains every transition's added cells.

        An element qualifies when, on every observed transition, painting
        each non-background before-cell's image under the element onto the
        before reproduces the after exactly — i.e. the added cells ARE
        the before's structure seen through that group element.
        """
        if not self.pure_addition:
            return set()
        return {
            name
            for name, rel in _DELTA_RELATIONS.items()
            if all(_completes_with(before, after, rel) for before, after in transitions)
        }


def perceive_change(before: Grid, after: Grid) -> ChangeDelta:
    """Describe the cell-level change from one perception to the next."""
    if before.shape != after.shape:
        return ChangeDelta(
            same_shape=False,
            added=frozenset(),
            removed=frozenset(),
            recolored=frozenset(),
        )
    bg = before.most_common_color()
    added: set[tuple[int, int, int]] = set()
    removed: set[tuple[int, int]] = set()
    recolored: set[tuple[int, int, int, int]] = set()
    for r, c, v in before.iter_cells():
        ov = after.at(r, c)
        if v == ov:
            continue
        if v == bg:
            added.add((r, c, ov))
        elif ov == bg:
            removed.add((r, c))
        else:
            recolored.add((r, c, v, ov))
    return ChangeDelta(
        same_shape=True,
        added=frozenset(added),
        removed=frozenset(removed),
        recolored=frozenset(recolored),
    )


def perceive_changes(transitions: list[Transition]) -> ChangeProfile:
    """Perceive the change across a series of observed transitions."""
    return ChangeProfile(deltas=[perceive_change(before, after) for before, after in transitions])


def _completes_with(before: Grid, after: Grid, rel) -> bool:
    """Does painting each non-bg cell's image under ``rel`` yield after?"""
    bg = before.most_common_color()
    rows = before.to_lists()
    for r, c, v in before.iter_cells():
        if v == bg:
            continue
        nr, nc = rel(r, c, before.height, before.width)
        if not (0 <= nr < before.height and 0 <= nc < before.width):
            return False
        if rows[nr][nc] == bg:
            rows[nr][nc] = v
    return Grid(tuple(tuple(row) for row in rows)) == after

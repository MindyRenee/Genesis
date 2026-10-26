"""Structural form perception — the variational settle over scenes.

Boards, sequences, and clouds are different structural forms; the
perceiver must discover which one a scene instantiates from the
evidence alone, and must learn a lattice's regularity well enough to
say which element violates it.
"""

from __future__ import annotations

from parietal_lobe.grid import Grid
from parietal_lobe.scene import perceive
from parietal_lobe.structure import (
    StructuralForm,
    StructurePerceiver,
    discover_structure,
)


def _board(
    rows: int,
    cols: int,
    tile: int = 2,
    gap: int = 2,
    color: int = 5,
    defects: dict[tuple[int, int], int] | None = None,
) -> Grid:
    """A tile grid: rows×cols tiles of `tile`×`tile` cells, `gap` apart.
    Gaps keep the background the modal color so tiles parse as objects."""
    defects = defects or {}
    h = rows * tile + (rows - 1) * gap
    w = cols * tile + (cols - 1) * gap
    g = [[0] * w for _ in range(h)]
    for r in range(rows):
        for c in range(cols):
            v = defects.get((r, c), color)
            r0 = r * (tile + gap)
            c0 = c * (tile + gap)
            for dr in range(tile):
                for dc in range(tile):
                    g[r0 + dr][c0 + dc] = v
    return Grid.from_lists(g)


def _line(n: int, color: int = 3) -> Grid:
    g = [[0] * (n * 3) for _ in range(2)]
    for i in range(n):
        for c in range(2):
            g[0][i * 3 + c] = color
            g[1][i * 3 + c] = color
    return Grid.from_lists(g)


def _two_groups() -> Grid:
    """A pair of tiles far left, a pair far right — two separate groups."""
    g = [[0] * 20 for _ in range(2)]
    for c in (0, 1, 16, 17):
        g[0][c] = 4
        g[1][c] = 4
    return Grid.from_lists(g)


def test_lattice_form_dominates_a_board() -> None:
    """A 3×3 board settles as LATTICE — periodic structure is perceived."""
    s = discover_structure(perceive(_board(3, 3)))
    assert s.groups, "scene held objects but no group formed"
    dominant = max(s.groups, key=lambda g: len(g.members))
    assert len(dominant.members) == 9
    assert dominant.form is StructuralForm.LATTICE
    assert dominant.lattice is not None
    assert len(dominant.lattice.row_lines) == 3
    assert len(dominant.lattice.col_lines) == 3


def test_chain_form_for_a_sequence() -> None:
    """Collinear elements settle as CHAIN — order, not periodicity."""
    s = discover_structure(perceive(_line(4)))
    dominant = max(s.groups, key=lambda g: len(g.members))
    assert dominant.form is StructuralForm.CHAIN


def test_separated_clusters_form_two_groups() -> None:
    """Spectral grouping splits unlinked structure into groups."""
    s = discover_structure(perceive(_two_groups()))
    assert len(s.groups) == 2


def test_learned_constraint_flags_violation() -> None:
    """After learning a board's regularity, a wrong tile is a violation."""
    sp = StructurePerceiver()
    clean = _board(3, 3, color=7)
    for _ in range(3):
        sp.observe(perceive(clean))
    assert sp.violations() == []

    broken = _board(3, 3, color=7, defects={(1, 1): 2})
    s = sp.observe(perceive(broken))
    assert s.violations, "the recolored centre tile went unnoticed"
    v = s.violations[0]
    assert v.observed == 2
    assert v.predicted == 7
    assert v.confidence > 0.5
    assert v.position == (1, 1)


def test_empty_scene_settles_quiet() -> None:
    s = discover_structure(perceive(Grid.from_lists([[0, 0], [0, 0]])))
    assert s.groups == []
    assert s.violations == []

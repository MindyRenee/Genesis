"""Constraint-board specialist tests — parse, induction, learning.

The boards under test are synthetic frames: small tiled lattices of
uniform blocks plus patterned marker blocks, and optionally a framed
query board wrapped in decoration. Everything the specialist learns
is induced from the frame — the tests only check that the general
machinery (segmentation, rule induction, propagation/link learning,
and the GF(L) click solver) does what it claims.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__file__).rsplit("/tests/", 1)[0])

from parietal_lobe.boards import (
    ConstraintSession,
    _boards,
    _Element,
    _nonzero_relation,
)
from parietal_lobe.grid import Grid


def _grid(size: int = 64) -> list[list[int]]:
    return [[0] * size for _ in range(size)]


def _stamp(g: list[list[int]], top: int, left: int, cells9: list[list[int]]) -> None:
    """Paint a 6×6 block as a 3×3 micro-grid of 2×2 cells."""
    for i in range(3):
        for j in range(3):
            for r in range(top + 2 * i, top + 2 * i + 2):
                for c in range(left + 2 * j, left + 2 * j + 2):
                    g[r][c] = cells9[i][j]


def _tile(g: list[list[int]], top: int, left: int, v: int) -> None:
    for r in range(top, top + 6):
        for c in range(left, left + 6):
            g[r][c] = v


# Marker micro used everywhere below: centre 4, equal on the four
# cardinal edges (0), differ on the corners (2).
_MARKER = [[2, 0, 2], [0, 4, 0], [2, 0, 2]]


def _solved_board(g: list[list[int]], top: int, left: int) -> None:
    """A 3×3 lattice whose tiles satisfy _MARKER's constraints."""
    for bi in range(3):
        for bj in range(3):
            t, c0 = top + bi * 8, left + bj * 8
            if (bi, bj) == (1, 1):
                _stamp(g, t, c0, _MARKER)
            elif bi == 1 or bj == 1:
                _tile(g, t, c0, 4)  # cardinal neighbours: equal → 4
            else:
                _tile(g, t, c0, 9)  # corner neighbours: differ → not 4


def _query_board(g: list[list[int]], top: int, left: int, fill: int = 7) -> None:
    """The same lattice wrapped in a single-colour region — a framed
    board whose separator is the frame colour itself."""
    for r in range(top - 2, top + 28):
        for c in range(left - 2, left + 28):
            g[r][c] = fill
    for bi in range(3):
        for bj in range(3):
            t, c0 = top + bi * 8, left + bj * 8
            if (bi, bj) == (1, 1):
                _stamp(g, t, c0, _MARKER)
            else:
                _tile(g, t, c0, 9)  # every neighbour wrong: cardinals
                # want 4


def _session(g: list[list[int]]) -> ConstraintSession:
    cs = ConstraintSession()
    cs.observe(Grid.from_lists(g))
    return cs


def test_parse_exemplars_and_framed_query() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _query_board(g, 34, 34)
    boards = _boards(Grid.from_lists(g))
    framed = [b for b in boards if b.framed]
    plain = [b for b in boards if not b.framed]
    assert len(framed) == 1
    assert len(plain) == 2
    assert all(len(b.markers) == 1 for b in boards)


def test_rule_induction() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _query_board(g, 34, 34)
    cs = _session(g)
    assert cs.rule == {2: "differ", 0: "equal"}
    # Both observed nonzero values agree → unseen nonzero generalize.
    assert cs._nonzero_rel == "differ"
    assert cs._relation_for(7) == "differ"


def test_nonzero_relation_mixed_is_none() -> None:
    assert _nonzero_relation({0: "equal", 2: "differ"}) == "differ"
    assert _nonzero_relation({0: "equal", 2: "differ", 5: "equal"}) is None
    assert _nonzero_relation(None) is None


def test_dotted_tile_is_not_marker() -> None:
    # A patterned tile carries its own colour at the centre — only a
    # marker shows a reference colour the ring never repeats.
    g = _grid()
    _stamp(g, 10, 10, [[14, 6, 14], [6, 14, 6], [14, 6, 14]])
    boards_grid = _boards(Grid.from_lists(g))
    # single element — no lattice, but is_marker is what matters
    cells = frozenset((r, c) for r in range(10, 16) for c in range(10, 16))
    # Rebuild the element the way the parser would: non-uniform micro.
    el = _Element(cells=cells, uniform=None, micro=((14, 6, 14), (6, 14, 6), (14, 6, 14)))
    assert not el.is_marker
    markerish = _Element(cells=cells, uniform=None, micro=((2, 0, 2), (0, 4, 0), (2, 0, 2)))
    assert markerish.is_marker
    assert not boards_grid  # a lone block is no board


def test_plan_clicks_violating_tile_then_charges() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _query_board(g, 34, 34)
    cs = _session(g)
    plan = cs.plan()
    assert plan is not None
    # Planning is read-only.
    assert not cs._tile_clicks
    cs.note_click(plan)
    assert sum(cs._tile_clicks.values()) == 1
    # A foreign click is not charged.
    cs.note_click((0, 0))
    assert sum(cs._tile_clicks.values()) == 1


def test_learns_link_and_transitions_from_diff() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _query_board(g, 34, 34)
    cs = _session(g)
    q = cs._query()
    assert q is not None
    # Give a query tile a dotted pattern so the click propagates.
    dotted = q.positions[(1, 0)]
    t, c0 = dotted.bbox[0], dotted.bbox[1]
    _stamp(g, t, c0, [[14, 6, 14], [6, 14, 6], [14, 6, 14]])
    cs.observe(Grid.from_lists(g))
    q = cs._query()
    assert (1, 0) in q.positions
    # Emit a click on the dotted tile directly (plan() would pick the
    # first violator deterministically; the learning path is what this
    # test exercises).
    dotted_el = q.positions[(1, 0)]
    cr, cc = dotted_el.centroid
    cell = (round(cr), round(cc))
    cs._last_plan = ((1, 0), cell, q)
    cs.note_click(cell)
    # Simulate the world: the clicked tile plus the neighbours its
    # link-coloured cells point at advance 9/14 -> 8.
    for dr, dc in [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)]:
        p = ((1, 0)[0] + dr, (1, 0)[1] + dc)
        el = q.positions.get(p)
        if el is None or el.is_marker:
            continue
        et, ec = el.bbox[0], el.bbox[1]
        for r in range(et, et + 6):
            for c in range(ec, ec + 6):
                if g[r][c] not in (6, 7):
                    g[r][c] = 8
    cs.observe(Grid.from_lists(g))
    assert cs._link_colour == 6
    assert cs._transitions.get(9) == 8 or cs._transitions.get(14) == 8


def test_cycle_order_requires_closed_loop() -> None:
    cs = ConstraintSession()
    cs._transitions = {14: 15}
    assert cs._cycle_order() is None
    cs._transitions = {14: 15, 15: 14}
    assert cs._cycle_order() == [14, 15]
    cs._transitions = {14: 15, 15: 12, 12: 14}
    assert cs._cycle_order() == [14, 15, 12]


def test_solve_clicks_finds_parity_vector() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _query_board(g, 34, 34)
    cs = _session(g)
    cs._transitions = {9: 8, 8: 9, 4: 4, 15: 15}  # not closed -> re-make
    cs._transitions = {9: 4, 4: 9, 8: 8}
    order = cs._cycle_order()
    assert order == [9, 4]
    q = cs._query()
    solved = cs._solve_clicks(q)
    assert solved is not None
    needed = {p: n for p, n in solved.items() if n}
    # The four cardinal neighbours want 4 (=equal); each is one click
    # away in the [9,4] cycle with self-only propagation.
    assert set(needed) == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert all(n == 1 for n in needed.values())


def _solved_framed_board(g: list[list[int]], top: int, left: int, fill: int = 7) -> None:
    for r in range(top - 2, top + 28):
        for c in range(left - 2, left + 28):
            g[r][c] = fill
    _solved_board(g, top, left)


def test_disabled_after_repeated_satisfaction() -> None:
    g = _grid()
    _solved_board(g, 2, 2)
    _solved_board(g, 2, 32)
    _solved_framed_board(g, 34, 34)  # framed query, already solved
    cs = _session(g)
    for _ in range(8):
        cs.observe(Grid.from_lists(g))
        cs.plan()
    assert cs._disabled


def test_silent_without_rule() -> None:
    g = _grid()
    _query_board(g, 10, 10)  # one framed board, no exemplars
    cs = _session(g)
    assert cs.rule is None
    assert cs.plan() is None

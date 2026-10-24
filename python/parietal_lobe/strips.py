"""Strip and glyph parsing — the segmentation primitives.

Wall-segmented runs of same-shaped tiles are how demonstration
formats lay out glyphs: a query strip reads as ``tile wall tile
wall tile``, a dictionary's runs read the same way inside panels.
Splitting those strips is *perception* — the parietal job of
binding structure out of pixels before any rule machinery
(``frontal_lobe.reasoning.induction``) reasons about what the
tiles mean. Everything here is geometry over ``Grid`` values;
nothing knows what a tile demonstrates.

- ``orbit`` — a glyph under every rigid transform (tilt is not
  identity; it is only a *view*).
- ``segment``/``segment_ex`` — split a strip on uniform wall
  columns or rows.
- ``tile_seq`` — split a run at a known tile shape and fixed
  pitch (immune to wall-colored columns *inside* a glyph).
- ``unit_shape`` — derive the shared tile footprint across a
  scene's run interiors.
- ``join``/``join_run`` — reassemble a strip; ``join_run`` writes
  the layout runs actually use (walls *between* tiles, none at
  the edges).
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from .grid import Grid


def orbit(g: Grid) -> list[Grid]:
    """All dihedral views of a pattern (the glyph under tilt)."""
    out: list[Grid] = []
    for refl in (False, True):
        base = g.reflect_h() if refl else g
        for k in range(4):
            v = base.rotate90(k) if k else base
            if all(v != o for o in out):
                out.append(v)
    return out


def _uniform_runs(grid: Grid, axis: str) -> list[tuple[int, int]]:
    """Indices where a full column (or row) is a single color.

    Returns ``(index, color)`` entries — candidate separator walls.
    """
    h, w = grid.height, grid.width
    out: list[tuple[int, int]] = []
    if axis == "col":
        for i in range(w):
            vals = {grid.at(r, i) for r in range(h)}
            if len(vals) == 1:
                out.append((i, vals.pop()))
    else:
        for i in range(h):
            vals = {grid.at(i, c) for c in range(w)}
            if len(vals) == 1:
                out.append((i, vals.pop()))
    return out


def segment(
    grid: Grid,
    separator: int | None = None,
    exemplar_shape: tuple[int, int] | None = None,
) -> list[Grid] | None:
    """Split a strip into equal blocks on uniform separator walls.

    ``separator`` forces the wall color when known; ``exemplar_shape``
    is ``(height, width)`` of the tile each block should be. Returns
    None when no consistent segmentation exists — silence, not a guess.
    """
    parsed = segment_ex(grid, separator, exemplar_shape)
    return parsed[0] if parsed else None


def segment_ex(
    grid: Grid,
    separator: int | None = None,
    exemplar_shape: tuple[int, int] | None = None,
) -> tuple[list[Grid], int] | None:
    """``segment`` plus the separator color the winning parse used."""
    for axis in ("col", "row"):
        runs = _uniform_runs(grid, axis)
        # The separator is a *color*, not "any uniform line": group
        # the uniform lines by color and try each as the wall —
        # interior background lines inside a glyph are a different
        # color and must not split it.
        colors = (
            [separator]
            if separator is not None
            else [c for c, _ in Counter(color for _, color in runs).most_common()]
        )
        for wall_color in colors:
            walls = [i for i, c in runs if c == wall_color]
            if len(walls) < 2:
                continue
            # Strip edges are implicit walls: a phrase can begin and
            # end with a tile, not a separator.
            n = grid.width if axis == "col" else grid.height
            bounds = [-1, *walls, n]
            spans: list[tuple[int, int]] = []
            for lo, hi in pairwise(bounds):
                if hi - lo - 1 > 0:
                    spans.append((lo + 1, hi))
            # Fragments at the strip's edges (padding before the
            # first tile, junk after the last) are trimmed; a wrong-
            # size span strictly interior breaks the parse — walls
            # don't run through the middle of a tile.
            if exemplar_shape:
                expected = exemplar_shape[1] if axis == "col" else exemplar_shape[0]
                while spans and spans[0][1] - spans[0][0] != expected:
                    spans.pop(0)
                while spans and spans[-1][1] - spans[-1][0] != expected:
                    spans.pop()
            else:
                expected = None
            if len(spans) < 2:
                continue
            sizes = {e - s for s, e in spans}
            if len(sizes) != 1 or (expected and sizes != {expected}):
                continue
            # For a row-axis strip the span is already the tile
            # height; the tile width must fill the strip. Mirror
            # for column-axis strips.
            if exemplar_shape:
                span = sizes.pop()
                if axis == "col" and (grid.height, span) != exemplar_shape:
                    continue
                if axis == "row" and (span, grid.width) != exemplar_shape:
                    continue
            blocks = []
            for s, e in spans:
                if axis == "col":
                    blocks.append(grid.subgrid(0, s, grid.height, e - s))
                else:
                    blocks.append(grid.subgrid(s, 0, e - s, grid.width))
            return blocks, wall_color
    return None


def unit_shape(grids: list[Grid]) -> tuple[int, int] | None:
    """The tile unit a set of run interiors shares.

    Both sides of a rule are *runs*: one or more tiles of a common
    footprint behind shared walls. The unit is the widest footprint
    every interior (of the modal height) tiles at — a single tile
    always qualifies, a run only tiles at the true unit.
    """
    if not grids:
        return None
    hs = Counter(g.height for g in grids)
    h = hs.most_common(1)[0][0]
    ws = sorted({g.width for g in grids if g.height == h})
    for w in range(3, ws[0] + 1):
        if all(tile_seq(g, (h, w)) is not None for g in grids if g.height == h):
            return h, w
    return h, ws[0]


def tile_seq(
    strip: Grid,
    tile_shape: tuple[int, int],
) -> tuple[list[Grid], int | None, int] | None:
    """Split a run interior into ``tile_shape`` blocks at fixed pitch.

    ``segment_ex`` finds tiles by hunting uniform separator columns;
    that fails when a glyph contains a column of the wall's own
    color — a solid vertical stripe inside a tile reads as a
    separator and the uniform-span check rejects the split. A run is also just *periodic*: ``n``
    tiles of width ``tw`` separated by ``gap`` wall columns fills
    ``n*(tw+gap) - gap``. The wall check stays — a pitch that
    doesn't land on uniform columns isn't a tiled run. Returns
    ``(tiles, wall_color, gap)`` — the wall's color and its column
    width — or None when the strip isn't a clean tiling of the
    shape. An exact-shape strip is one tile.
    """
    th, tw = tile_shape
    if strip.shape == (th, tw):
        return [strip], None, 0
    if strip.height != th or strip.width <= tw:
        return None
    for gap in (1, 2, 3):
        n, rem = divmod(strip.width + gap, tw + gap)
        if rem or n < 2:
            continue
        colors: set[int] = set()
        ok = True
        for k in range(1, n):
            for j in range(gap):
                col = {strip.at(r, k * (tw + gap) - gap + j) for r in range(th)}
                if len(col) != 1:
                    ok = False
                    break
                colors |= col
            if not ok:
                break
        if not ok or len(colors) != 1:
            continue
        tiles = [strip.subgrid(0, k * (tw + gap), th, tw) for k in range(n)]
        return tiles, colors.pop(), gap
    return None


def join_run(blocks: list[Grid], separator: int | None, gap: int = 1) -> Grid | None:
    """Concatenate tiles into a run interior — ``gap`` wall columns
    *between* tiles, none at the edges: the layout a dictionary's
    right-hand side (and the answer strip) actually uses."""
    if not blocks:
        return None
    h = blocks[0].height
    if any(b.height != h for b in blocks):
        return None
    rows: list[list[int]] = [[] for _ in range(h)]
    for i, b in enumerate(blocks):
        if i and separator is not None:
            for r in range(h):
                rows[r].extend([separator] * gap)
        for r in range(h):
            rows[r].extend(b.rows()[r])
    return Grid.from_lists(rows)


def join(blocks: list[Grid], separator: int | None) -> Grid | None:
    """Reassemble blocks into a strip — walls between when known."""
    if not blocks:
        return None
    h = max(b.height for b in blocks)
    if any(b.height != h for b in blocks):
        return None
    if separator is None:
        # Concatenate without walls — honest output when the
        # exemplars didn't demonstrate a separator color.
        rows: list[list[int]] = [[] for _ in range(h)]
        for b in blocks:
            for r in range(h):
                rows[r].extend(b.rows()[r])
        return Grid.from_lists(rows)
    total_w = sum(b.width for b in blocks) + len(blocks) + 1
    rows = [[separator] * total_w for _ in range(h)]
    c = 1
    for b in blocks:
        for r in range(h):
            rows[r][c : c + b.width] = list(b.rows()[r])
        c += b.width + 1
    return Grid.from_lists(rows)

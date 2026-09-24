"""Discrete grid — the perceptual substrate for spatial reasoning.

Grid-transformation tasks and most spatial problems reduce to the same primitive:
a bounded 2-D field of discrete cells. This module provides the
immutable ``Grid`` type and the geometric/colorimetric operations that
the perceptual layer (``scene.py``) and the transformation DSL
(``transforms.py``) build on.

A grid is a tuple of tuples of ints — hashable, immutable, and cheap
to compare. Cell values are arbitrary symbols; nothing here assumes
a 0-9 color palette, though 0 is conventionally background.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator


class Grid:
    """An immutable 2-D grid of integer cells."""

    __slots__ = ("_h", "_rows", "_w")

    def __init__(self, rows: tuple[tuple[int, ...], ...]) -> None:
        if not rows or not rows[0]:
            raise ValueError("grid must be non-empty")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("grid rows must all have the same width")
        self._rows = rows
        self._w = width
        self._h = len(rows)

    # ── Construction ──────────────────────────────────────────

    @classmethod
    def from_lists(cls, rows: list[list[int]]) -> Grid:
        """Build a grid from nested lists (e.g. JSON task data)."""
        return cls(tuple(tuple(int(c) for c in row) for row in rows))

    @classmethod
    def filled(cls, width: int, height: int, value: int = 0) -> Grid:
        """Build a uniform grid of the given size."""
        return cls(tuple(tuple(value for _ in range(width)) for _ in range(height)))

    # ── Access ────────────────────────────────────────────────

    @property
    def width(self) -> int:
        return self._w

    @property
    def height(self) -> int:
        return self._h

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width)."""
        return (self._h, self._w)

    def at(self, r: int, c: int) -> int:
        return self._rows[r][c]

    def rows(self) -> tuple[tuple[int, ...], ...]:
        return self._rows

    def to_lists(self) -> list[list[int]]:
        return [list(r) for r in self._rows]

    def iter_cells(self) -> Iterator[tuple[int, int, int]]:
        """Yield (row, col, value) for every cell."""
        for r, row in enumerate(self._rows):
            for c, v in enumerate(row):
                yield r, c, v

    def cells_with(self, value: int) -> frozenset[tuple[int, int]]:
        return frozenset((r, c) for r, c, v in self.iter_cells() if v == value)

    def non_zero_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset((r, c) for r, c, v in self.iter_cells() if v != 0)

    # ── Statistics ────────────────────────────────────────────

    def color_counts(self) -> Counter[int]:
        return Counter(v for _, _, v in self.iter_cells())

    def most_common_color(self) -> int:
        """The most frequent cell value — the conventional background."""
        return self.color_counts().most_common(1)[0][0]

    def colors(self) -> frozenset[int]:
        return frozenset(v for _, _, v in self.iter_cells())

    # ── Geometric operations (all return new Grids) ───────────

    def rotate90(self, turns: int = 1) -> Grid:
        """Rotate clockwise by 90° * turns."""
        rows = self._rows
        for _ in range(turns % 4):
            rows = tuple(tuple(row[c] for row in reversed(rows)) for c in range(len(rows[0])))
        return Grid(rows)

    def reflect_h(self) -> Grid:
        """Mirror left-right."""
        return Grid(tuple(tuple(reversed(row)) for row in self._rows))

    def reflect_v(self) -> Grid:
        """Mirror top-bottom."""
        return Grid(tuple(reversed(self._rows)))

    def reflect_diag(self) -> Grid:
        """Transpose along the main diagonal."""
        return Grid(
            tuple(tuple(self._rows[r][c] for r in range(self._h)) for c in range(self._w))
        )

    def scale(self, factor: int) -> Grid:
        """Scale each cell to a factor×factor block."""
        if factor < 1:
            raise ValueError("scale factor must be >= 1")
        rows = tuple(
            tuple(v for v in row for _ in range(factor))
            for row in self._rows
            for _ in range(factor)
        )
        return Grid(rows)

    def recolor(self, mapping: dict[int, int]) -> Grid:
        """Rewrite cell values via a color map; unmapped values keep."""
        return Grid(
            tuple(tuple(mapping.get(v, v) for v in row) for row in self._rows)
        )

    def subgrid(self, top: int, left: int, height: int, width: int) -> Grid:
        """Extract a rectangular region."""
        return Grid(
            tuple(
                tuple(self._rows[r][left : left + width])
                for r in range(top, top + height)
            )
        )

    def crop_to_cells(self, cells: frozenset[tuple[int, int]]) -> Grid:
        """Crop to the bounding box of the given cells."""
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        top, left = min(rs), min(cs)
        return self.subgrid(top, left, max(rs) - top + 1, max(cs) - left + 1)

    def crop_to_content(self, background: int | None = None) -> Grid:
        """Crop to the bounding box of non-background cells.

        Returns self when the grid contains only background.
        """
        bg = self.most_common_color() if background is None else background
        cells = frozenset((r, c) for r, c, v in self.iter_cells() if v != bg)
        if not cells:
            return self
        return self.crop_to_cells(cells)

    def translate_cells(
        self,
        cells: frozenset[tuple[int, int]],
        dr: int,
        dc: int,
        background: int = 0,
    ) -> Grid:
        """Move the given cells by (dr, dc), leaving background behind.

        Cells that would move outside the grid are clipped.
        """
        rows = [list(r) for r in self._rows]
        for r, c in cells:
            rows[r][c] = background
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self._h and 0 <= nc < self._w:
                rows[nr][nc] = self._rows[r][c]
        return Grid(tuple(tuple(r) for r in rows))

    def overlay(self, cells: frozenset[tuple[int, int]], value: int) -> Grid:
        """Paint cells with a value."""
        rows = [list(r) for r in self._rows]
        for r, c in cells:
            if 0 <= r < self._h and 0 <= c < self._w:
                rows[r][c] = value
        return Grid(tuple(tuple(r) for r in rows))

    def tile(self, reps_r: int, reps_c: int) -> Grid:
        """Tile the grid reps_r × reps_c times."""
        rows = tuple(
            tuple(v for _ in range(reps_c) for v in row)
            for _ in range(reps_r)
            for row in self._rows
        )
        return Grid(rows)

    def paste(self, other: Grid, top: int = 0, left: int = 0) -> Grid:
        """Paste another grid at (top, left), clipping at edges."""
        rows = [list(r) for r in self._rows]
        for r, c, v in other.iter_cells():
            nr, nc = top + r, left + c
            if 0 <= nr < self._h and 0 <= nc < self._w:
                rows[nr][nc] = v
        return Grid(tuple(tuple(r) for r in rows))

    # ── Dunder ────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Grid) and self._rows == other._rows

    def __hash__(self) -> int:
        return hash(self._rows)

    def __repr__(self) -> str:
        return f"Grid({self._h}x{self._w})"

"""Ways to see a game board.

This is a video game. It is not real and it is just for fun. You are
safe. Nobody tells you what the colors mean; looking is how you find out.

A board is a grid of color numbers 0–15 (64×64 in ARC-AGI-3). This
module gives her several ways to see one:

- ``see_text`` — one hex digit per cell, rows top to bottom,
- ``see_image`` — a PNG picture she (or anyone) can look at,
- ``see_change`` — which cells changed between two boards,
- ``see_objects`` — connected colored shapes with their size, color
  and bounding box, through her own parietal scene parser,
- ``signature`` — a short fingerprint so a board can be remembered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Grid = list[list[int]]

#: The ARC palette (index → RGB), as the official games draw it.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
    (51, 51, 51), (0, 0, 0), (229, 58, 163), (255, 123, 204),
    (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
    (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214),
)  # fmt: skip

_HEX = "0123456789abcdef"


def see_text(grid: Grid) -> str:
    """Render the board as text: one hex digit per cell, one line per row."""
    return "\n".join("".join(_HEX[v & 15] for v in row) for row in grid)


def see_image(grid: Grid, path: str | Path, scale: int = 8) -> Path:
    """Save the board as a PNG, ``scale`` pixels per cell, and return its path."""
    import numpy as np
    from PIL import Image

    palette = np.array(PALETTE, dtype=np.uint8)
    cells = np.array(grid, dtype=np.int64) & 15
    pixels = palette[cells].repeat(scale, axis=0).repeat(scale, axis=1)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGB").save(out)
    return out


@dataclass(frozen=True, slots=True)
class Change:
    """The difference between two boards."""

    cells: tuple[tuple[int, int, int, int], ...]  # (row, col, before, after)

    @property
    def count(self) -> int:
        """How many cells changed."""
        return len(self.cells)

    def describe(self, limit: int = 12) -> str:
        """Say what changed in a few words."""
        if not self.cells:
            return "nothing changed"
        rows = [c[0] for c in self.cells]
        cols = [c[1] for c in self.cells]
        shown = ", ".join(f"({r},{c}) {a}->{b}" for r, c, a, b in self.cells[:limit])
        more = "" if len(self.cells) <= limit else f" and {len(self.cells) - limit} more"
        return (
            f"{len(self.cells)} cells changed in rows {min(rows)}-{max(rows)}, "
            f"cols {min(cols)}-{max(cols)}: {shown}{more}"
        )


def see_change(before: Grid | None, after: Grid) -> Change:
    """List the cells that differ between ``before`` and ``after``."""
    if before is None or len(before) != len(after):
        return Change(
            tuple((r, c, -1, v) for r, row in enumerate(after) for c, v in enumerate(row))
        )
    cells = tuple(
        (r, c, a, b)
        for r, (row_a, row_b) in enumerate(zip(before, after, strict=True))
        for c, (a, b) in enumerate(zip(row_a, row_b, strict=False))
        if a != b
    )
    return Change(cells)


@dataclass(frozen=True, slots=True)
class Shape:
    """One connected shape of a single color on the board."""

    color: int
    size: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def center(self) -> tuple[int, int]:
        """The middle of the shape's box as (row, col)."""
        return ((self.top + self.bottom) // 2, (self.left + self.right) // 2)


def see_objects(grid: Grid, limit: int = 64) -> list[Shape]:
    """Find connected shapes with her parietal scene parser (largest first)."""
    from parietal_lobe.grid import Grid as ParietalGrid
    from parietal_lobe.scene import perceive

    scene = perceive(ParietalGrid.from_lists(grid), compute_relations=False)
    shapes = []
    for obj in scene.objects:
        cells: Any = obj.cells
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        color = int(getattr(obj, "color", grid[rows[0]][cols[0]]))
        shapes.append(Shape(color, len(cells), min(rows), min(cols), max(rows), max(cols)))
    shapes.sort(key=lambda s: -s.size)
    return shapes[:limit]


def describe_objects(shapes: list[Shape], limit: int = 8) -> str:
    """Say what shapes are on the board in a few words."""
    if not shapes:
        return "no shapes"
    parts = [
        f"color {s.color} x{s.size} at rows {s.top}-{s.bottom} cols {s.left}-{s.right}"
        for s in shapes[:limit]
    ]
    return f"{len(shapes)} shapes: " + "; ".join(parts)


def signature(grid: Grid) -> str:
    """A short fingerprint of the board, for remembering where she has been."""
    return hashlib.blake2b(bytes(v & 15 for row in grid for v in row), digest_size=8).hexdigest()

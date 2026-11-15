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

#: Perceptual names for the 16 palette indices, computed lazily through
#: her V1 colour naming — the same machinery that names what her camera
#: sees, so a game's "8" and the world's "red" are one colour to her.
_COLOR_NAMES: tuple[str, ...] | None = None

#: Per-index HSV (hue in [0,1]) of the official palette — the exact
#: colour facts the board carries, before any estimation.
_PALETTE_HSV: tuple[tuple[float, float, float], ...] | None = None


def _palette_hsv() -> tuple[tuple[float, float, float], ...]:
    """HSV triples for the palette, computed once."""
    global _COLOR_NAMES, _PALETTE_HSV
    if _PALETTE_HSV is None:
        import colorsys

        from occipital_lobe.v1 import name_color

        _PALETTE_HSV = tuple(colorsys.rgb_to_hsv(*(c / 255 for c in rgb)) for rgb in PALETTE)
        _COLOR_NAMES = tuple(name_color(h * 360, s, v) for h, s, v in _PALETTE_HSV)
    return _PALETTE_HSV


def color_name(index: int) -> str:
    """The perceptual name of a palette index (0–15).

    Goes index → RGB (the official palette) → HSV → her V1
    ``name_color``, so the board's colour numbers arrive in the same
    vocabulary her retina gives the real world — "red", "blue",
    "yellow" — not bare indices she cannot place.
    """
    _palette_hsv()
    assert _COLOR_NAMES is not None
    return _COLOR_NAMES[index & 15]


def see_text(grid: Grid) -> str:
    """Render the board as text: one hex digit per cell, one line per row."""
    return "\n".join("".join(_HEX[v & 15] for v in row) for row in grid)


def see_scene(grid: Grid) -> str:
    """Scene-level gestalt of the board — its look, not its parts.

    The board's colours are *exact* — every cell is a known palette
    index — so the statistics are computed exactly rather than through
    the camera's histogram estimator (whose 30° hue bins smear red
    into orange at band edges). The thresholds and hue bands mirror
    ``_analyze_color_and_light``/``_color_warmth`` in her occipital
    lobe, and every name still comes from her V1 ``name_color``, so a
    board scene reads exactly like a camera scene:
    ``scene | dim warm red | regions: center=red,...``.
    """
    import numpy as np

    cells = np.asarray(grid, dtype=np.int64) & 15
    hsv = np.asarray(_palette_hsv())[cells]  # (H, W, 3); hue in [0,1]
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    chromatic = (sat > 0.15) & (val > 0.15)

    def dominant(cell_slice: tuple[slice, slice] | None = None) -> str:
        """Modal colour of the chromatic cells — the most common
        saturated hue, as ``_dominant_color_name`` defines dominant."""
        c = cells if cell_slice is None else cells[cell_slice]
        ch = chromatic if cell_slice is None else chromatic[cell_slice]
        if not c.size:
            return ""
        if ch.any():
            idx, counts = np.unique(c[ch], return_counts=True)
            top = int(counts.max())
            # An exact tie means the region holds equal amounts of
            # several colours — that is a mix, not any one of them.
            if int((counts == top).sum()) > 1:
                return "mixed"
            return color_name(int(idx[counts.argmax()]))
        v = float(val.mean()) if cell_slice is None else float(val[cell_slice].mean())
        return "white" if v > 0.75 else "black" if v < 0.12 else "gray"

    b = float(val.mean())
    light = "dark" if b < 0.2 else "dim" if b < 0.4 else "bright" if b > 0.7 else "medium"

    if chromatic.any():
        hues = hue[chromatic]
        warm = int(((hues < 65) | (hues >= 345)).sum())
        cool = int(((hues >= 65) & (hues < 280)).sum())
        warmth = "warm" if warm > cool * 1.3 else "cool" if cool > warm * 1.3 else "neutral"
    else:
        warmth = "neutral"

    dom = dominant()
    parts = [f"scene | {light} {warmth} {dom}".rstrip()]
    h, w = cells.shape
    regions = {
        place: name
        for place, name in (
            ("left", dominant((slice(None), slice(0, w // 3)))),
            ("center", dominant((slice(None), slice(w // 3, 2 * w // 3)))),
            ("right", dominant((slice(None), slice(2 * w // 3, w)))),
            ("top", dominant((slice(0, h // 2), slice(None)))),
            ("bottom", dominant((slice(h // 2, h), slice(None)))),
        )
        if name and name != dom
    }
    if regions:
        parts.append("regions: " + ",".join(f"{p}={c}" for p, c in regions.items()))
    return " | ".join(parts)


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
        f"{color_name(s.color)}({s.color}) x{s.size} "
        f"at rows {s.top}-{s.bottom} cols {s.left}-{s.right}"
        for s in shapes[:limit]
    ]
    return f"{len(shapes)} shapes: " + "; ".join(parts)


def signature(grid: Grid) -> str:
    """A short fingerprint of the board, for remembering where she has been."""
    return hashlib.blake2b(bytes(v & 15 for row in grid for v in row), digest_size=8).hexdigest()

"""MT/V5 — motion processing in the dorsal stream.

Middle temporal area: takes two consecutive luminance frames and
produces a motion field — per-cell normal flow (direction, speed) plus
the moving regions those cells group into. This is the input LIP's
saliency map and the sensorimotor areas need: *where* things move and
*how fast*, not what they are.

The estimator is the constant-brightness approximation

    Ix * u + Iy * v + It = 0

solved per coarse cell by least squares over its pixels (a small,
local Lucas–Kanade). Cells without enough gradient or change get no
flow — a blank wall reports stillness, not noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Coarse flow grid: 16 cells across the long axis of a 640x480 frame
# gives ~40px cells — enough resolution to localize a mover without
# per-pixel cost.
_FLOW_CELL_PX = 40
# Minimum temporal change and spatial gradient for a cell to carry
# flow. Below these the aperture problem makes the estimate worthless.
_MIN_TEMPORAL_CHANGE = 6.0   # mean |It| in 0-255 units
_MIN_GRADIENT = 4.0          # mean gradient magnitude
_MAX_SPEED = 50.0            # px/frame — clips implausible jumps


@dataclass(slots=True)
class MotionCell:
    """Flow estimate for one coarse cell of the visual field."""

    y: float          # cell centre, pixel-space row
    x: float          # cell centre, pixel-space column
    u: float          # horizontal velocity (px/frame, +right)
    v: float          # vertical velocity (px/frame, +down)
    speed: float      # magnitude
    confidence: float # how well-conditioned the local estimate was


@dataclass(slots=True)
class MotionRegion:
    """A contiguous group of cells moving together."""

    cells: list[MotionCell] = field(default_factory=list)
    y: float = 0.0    # centroid, pixel space
    x: float = 0.0
    u: float = 0.0    # mean velocity
    v: float = 0.0
    speed: float = 0.0


@dataclass(slots=True)
class MotionField:
    """The dorsal stream's report of what moved this frame pair."""

    cells: list[MotionCell] = field(default_factory=list)
    regions: list[MotionRegion] = field(default_factory=list)
    global_motion: float = 0.0   # fraction of cells moving (ego-motion proxy)
    dominant_direction: float = 0.0  # radians of the mean flow vector

    @property
    def moving(self) -> bool:
        """Return whether any moving cells were detected."""
        return bool(self.cells)

    def fastest(self) -> MotionCell | None:
        """Return the moving cell with the greatest speed, if any."""
        return max(self.cells, key=lambda c: c.speed) if self.cells else None


class MotionProcessor:
    """MT/V5 — estimates optic flow between consecutive frames.

    Holds only the previous luminance frame; ``process(rgb)`` returns
    the MotionField for the pair. The first frame after construction
    (or a resolution change) produces an empty field — a single image
    carries no motion.
    """

    def __init__(self, cell_px: int = _FLOW_CELL_PX) -> None:
        """Initialize frame state with the requested flow-cell size."""
        self._cell = max(8, cell_px)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the previous frame so the next frame yields no motion."""
        self._prev = None

    def process(self, rgb: np.ndarray) -> MotionField:
        """Compute the motion field between the last frame and *rgb*."""
        lum = rgb[..., :3].astype(np.float32).mean(axis=2)
        prev = self._prev
        self._prev = lum
        if prev is None or prev.shape != lum.shape:
            return MotionField()

        h, w = lum.shape
        it = lum - prev
        # Spatial gradients of the current frame (central differences).
        grads = np.gradient(lum)
        gy = np.asarray(grads[0])
        gx = np.asarray(grads[1])

        cells: list[MotionCell] = []
        cell = self._cell
        for cy in range(0, h - cell + 1, cell):
            for cx in range(0, w - cell + 1, cell):
                mc = self._cell_flow(
                    gy[cy:cy + cell, cx:cx + cell],
                    gx[cy:cy + cell, cx:cx + cell],
                    it[cy:cy + cell, cx:cx + cell],
                )
                if mc is None:
                    continue
                u, v, conf = mc
                speed = math.hypot(u, v)
                if speed > _MAX_SPEED:
                    u, v = u * _MAX_SPEED / speed, v * _MAX_SPEED / speed
                    speed = _MAX_SPEED
                cells.append(MotionCell(
                    y=cy + cell / 2, x=cx + cell / 2,
                    u=u, v=v, speed=speed, confidence=conf,
                ))

        field = MotionField(cells=cells)
        if cells:
            n_cells = ((h // cell) or 1) * ((w // cell) or 1)
            field.global_motion = len(cells) / max(1, n_cells)
            mu = sum(c.u for c in cells) / len(cells)
            mv = sum(c.v for c in cells) / len(cells)
            field.dominant_direction = math.atan2(mv, mu)
            field.regions = self._group_regions(cells, cell)
        return field

    @staticmethod
    def _cell_flow(
        gy: np.ndarray, gx: np.ndarray, it: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Least-squares normal flow for one cell (local Lucas-Kanade)."""
        if float(np.abs(it).mean()) < _MIN_TEMPORAL_CHANGE:
            return None
        flat_gy = gy.ravel()
        flat_gx = gx.ravel()
        flat_it = it.ravel()
        grad_mag = np.hypot(flat_gx, flat_gy)
        if float(grad_mag.mean()) < _MIN_GRADIENT:
            return None
        # Solve [gx gy]·[u v] = -it in least squares; the 2x2 normal
        # equations are the structure tensor.
        a = np.array(
            [
                [float((flat_gx * flat_gx).sum()), float((flat_gx * flat_gy).sum())],
                [float((flat_gx * flat_gy).sum()), float((flat_gy * flat_gy).sum())],
            ]
        )
        b = -np.array(
            [float((flat_gx * flat_it).sum()), float((flat_gy * flat_it).sum())]
        )
        det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
        # Confidence from conditioning — a degenerate aperture direction
        # (det → 0) means only the component along the gradient is known.
        trace = a[0, 0] + a[1, 1]
        conf = min(1.0, det / (trace * trace + 1e-9) * 4.0)
        if det < 1e-3:
            # Aperture-limited: project onto the mean gradient direction.
            mag = float(grad_mag.mean()) + 1e-9
            u = float((-flat_it * flat_gx).mean()) / mag
            v = float((-flat_it * flat_gy).mean()) / mag
            return u, v, min(conf, 0.3)
        u = (a[1, 1] * b[0] - a[0, 1] * b[1]) / det
        v = (a[0, 0] * b[1] - a[0, 1] * b[0]) / det
        return float(u), float(v), conf

    @staticmethod
    def _group_regions(
        cells: list[MotionCell], cell_px: int,
    ) -> list[MotionRegion]:
        """Group adjacent cells with coherent direction into regions."""
        by_grid: dict[tuple[int, int], MotionCell] = {}
        for c in cells:
            by_grid[(int(c.y // cell_px), int(c.x // cell_px))] = c
        seen: set[tuple[int, int]] = set()
        regions: list[MotionRegion] = []
        for key, seed in by_grid.items():
            if key in seen:
                continue
            stack, members = [key], []
            seed_dir = math.atan2(seed.v, seed.u)
            while stack:
                k = stack.pop()
                if k in seen or k not in by_grid:
                    continue
                c = by_grid[k]
                # Same region only if directions roughly agree (< ~60°).
                if abs(math.atan2(
                    math.sin(math.atan2(c.v, c.u) - seed_dir),
                    math.cos(math.atan2(c.v, c.u) - seed_dir),
                )) > 1.05:
                    continue
                seen.add(k)
                members.append(c)
                stack.extend(
                    [(k[0] + dy, k[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
                )
            if members:
                r = MotionRegion(cells=members)
                r.y = sum(c.y for c in members) / len(members)
                r.x = sum(c.x for c in members) / len(members)
                r.u = sum(c.u for c in members) / len(members)
                r.v = sum(c.v for c in members) / len(members)
                r.speed = math.hypot(r.u, r.v)
                regions.append(r)
        regions.sort(key=lambda r: r.speed * len(r.cells), reverse=True)
        return regions

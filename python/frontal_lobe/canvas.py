"""Canvas for Genesis — a paint program she drives herself.

This module is the *medium*, not the artist. It provides what a
drawing program provides a human: a sheet of paper, an unlimited
palette of colors, a brush whose character is dialed in parameters
(size, opacity, hardness, scatter, texture, erasure, blend — any
utensil is a point in that space), and operations that apply them —
strokes, shapes, fills, erasure. Nothing here decides what gets
drawn; that is entirely hers.

Two surfaces do the work:

- A built-in rasterizer (Cairo) that executes every operation
  in-process, so painting always works — headless hosts, tests,
  Krita not installed. This is the canonical artifact: ``save``
  composites the paper layer and the paint layer into a WebP.
- A Krita bridge (``krita_studio``) that mirrors the same ops into a
  running Krita — the real drawing program on her computer — when it
  is reachable. Krita is best-effort: its absence never blocks her.

Each finished drawing is remembered in ``style_profile.json`` — which
tools, colors, and gestures she used — so she can recall her art,
notice repetition, and get bored of it. That memory is what makes
exploration her own rather than a dice roll.

Erasing works the way paper works: the paper sits on its own layer
and the eraser lifts the paint layer, revealing the paper beneath.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import cairo
except ImportError:  # pragma: no cover - platform without Cairo
    # Cairo is the built-in rasterizer. Without it the program still
    # runs when a Krita bridge is reachable; otherwise draw() returns
    # None with a warning.
    cairo = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:  # pragma: no cover - platform without OpenCV
    # Only floodFill is used; the NumPy fallback in _flood_mask_np
    # keeps bucket fill working when OpenCV is not installed.
    cv2 = None  # type: ignore[assignment]
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from frontal_lobe.krita_studio import KritaStudio
    from limbic_system.emotion import EmotionalState

__all__ = [
    "Brush",
    "Canvas",
    "DrawingResult",
    "PaintSheet",
]

logger = logging.getLogger(__name__)

_CAIRO_AVAILABLE = cairo is not None

# Default sheet size — large enough for detail, small enough to be fast.
_SHEET_WIDTH = 640
_SHEET_HEIGHT = 480

# Retention bound on saved drawings.
_MAX_SAVED_DRAWINGS = 200


# A brush is a point in parameter space, not an entry in a list.
#   size      — mark width in px
#   opacity   — 0..1 how much paint each pass lays down
#   hardness  — 0..1 crisp edge (1.0) → soft airbrush falloff (0.0)
#   scatter   — 0..1 solid stroke → spray of dabs around the path
#   texture   — 0..1 smooth → rough/skiddy (crayon-like jitter)
#   erase     — lift paint instead of laying it
#   blend     — compositing: "normal" or "multiply" (marker-like)
# She dials any of these directly. A name passed to pick_tool is her
# own label for what she dialed — it never changes the settings.
@dataclass
class Brush:
    """The utensil currently in her hand — fully parameterized."""

    size: float = 4.0
    opacity: float = 1.0
    hardness: float = 1.0
    scatter: float = 0.0
    texture: float = 0.0
    erase: bool = False
    blend: str = "normal"

    def copy(self) -> Brush:
        return Brush(
            self.size,
            self.opacity,
            self.hardness,
            self.scatter,
            self.texture,
            self.erase,
            self.blend,
        )


# There is no built-in utensil list. A brush is a point in parameter
# space — size, opacity, hardness, scatter, texture, erase, blend —
# and every point in that space is a different implement. She dials
# parameters with ``set_brush``; when she says ``pick_tool("pencil")``
# the name is *her* label for what she is holding, never a setting
# menu (on a Krita backend it also resolves that app's own presets —
# the same way naming works when the program is real).


@dataclass
class DrawingResult:
    """A finished drawing — what she made and how she made it."""

    filepath: str
    width: int
    height: int
    description: str
    timestamp: float
    mood: str = ""
    tools_used: list[str] = field(default_factory=list)
    colors_used: list[str] = field(default_factory=list)
    strokes: int = 0


class PaintSheet:
    """One sheet of paper — the rasterizing backend.

    A paper layer holds the sheet color; a transparent paint layer
    holds her marks. The eraser clears paint to reveal the paper.
    Every operation maps to one method; the Krita backend mirrors the
    same ops into the running application when it is connected.
    """

    def __init__(
        self,
        width: int = _SHEET_WIDTH,
        height: int = _SHEET_HEIGHT,
        paper: tuple[int, int, int] = (250, 250, 248),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.paper = paper
        self._rng = rng or np.random.default_rng(seed=int(time.time()))
        self._paper = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(self._paper)
        ctx.set_source_rgb(paper[0] / 255, paper[1] / 255, paper[2] / 255)
        ctx.paint()
        self._paint = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self._ctx = cairo.Context(self._paint)
        self._color: tuple[float, float, float, float] = (0, 0, 0, 1.0)
        self._brush = Brush()
        self._tool = "pencil"

    # ── Program controls ──────────────────────────────────────────

    def set_color(self, r: int, g: int, b: int, a: float = 1.0) -> None:
        """Dip the utensil in a color — any RGB + opacity."""
        self._color = (
            max(0, min(255, int(r))) / 255,
            max(0, min(255, int(g))) / 255,
            max(0, min(255, int(b))) / 255,
            max(0.0, min(1.0, float(a))),
        )

    def set_brush(
        self,
        size: float | None = None,
        opacity: float | None = None,
        hardness: float | None = None,
        scatter: float | None = None,
        texture: float | None = None,
        erase: bool | None = None,
        blend: str | None = None,
    ) -> None:
        """Dial the utensil directly — every parameter is hers to set.

        This is the fully open path: any mark character is a point in
        this space. Named tools are only aliases over these settings.
        """
        if size is not None:
            self._brush.size = max(0.5, float(size))
        if opacity is not None:
            self._brush.opacity = max(0.0, min(1.0, float(opacity)))
        if hardness is not None:
            self._brush.hardness = max(0.0, min(1.0, float(hardness)))
        if scatter is not None:
            self._brush.scatter = max(0.0, min(1.0, float(scatter)))
        if texture is not None:
            self._brush.texture = max(0.0, min(1.0, float(texture)))
        if erase is not None:
            self._brush.erase = bool(erase)
        if blend is not None:
            self._brush.blend = str(blend)

    def pick_tool(self, tool: str, size: float | None = None) -> None:
        """Pick up what she calls ``tool``.

        The name is hers — a label, not a menu. Nothing here overrides
        the brush she dialed; a Krita backend resolves the name
        against that app's real presets. ``size`` is the one thing a
        human always reaches for alongside the utensil, so it is
        honored directly.
        """
        self._tool = tool
        if size is not None:
            self._brush.size = max(0.5, float(size))

    # ── Marks ─────────────────────────────────────────────────────

    def _smooth_path(self, ctx, points: list[tuple[float, float]]) -> None:
        """Trace a path through points, smoothed like a moving hand."""
        pts = [(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        if not pts:
            return
        ctx.move_to(*pts[0])
        if len(pts) == 1:
            ctx.line_to(pts[0][0] + 0.01, pts[0][1] + 0.01)
            return
        for i in range(1, len(pts) - 1):
            mx = (pts[i][0] + pts[i + 1][0]) / 2
            my = (pts[i][1] + pts[i + 1][1]) / 2
            ctx.curve_to(pts[i][0], pts[i][1], pts[i][0], pts[i][1], mx, my)
        ctx.line_to(*pts[-1])

    def _stamp_soft(self, points: list[tuple[float, float]]) -> None:
        """A soft brush lays down dabs with a feathered edge."""
        r, g, b, a = self._color
        radius = self._brush.size
        spacing = max(1.5, radius * 0.35)
        traveled = 0.0
        prev = points[0]
        for pt in points:
            dist = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            traveled += dist
            if traveled < spacing:
                prev = pt
                continue
            traveled = 0.0
            prev = pt
            inner = max(0.05, self._brush.hardness)
            pat = cairo.RadialGradient(pt[0], pt[1], 0, pt[0], pt[1], radius)
            pat.add_color_stop_rgba(0, r, g, b, a * self._brush.opacity)
            pat.add_color_stop_rgba(inner, r, g, b, a * self._brush.opacity * 0.85)
            pat.add_color_stop_rgba(1, r, g, b, 0)
            self._ctx.set_source(pat)
            self._ctx.arc(pt[0], pt[1], radius, 0, math.pi * 2)
            self._ctx.fill()

    def _stamp_scatter(self, points: list[tuple[float, float]]) -> None:
        """A scattered stroke mists dabs around the path."""
        r, g, b, a = self._color
        spread = max(3.0, self._brush.size * (0.5 + self._brush.scatter))
        for pt in points:
            n_dots = int(self._rng.integers(6, 16))
            xs = self._rng.normal(pt[0], spread, n_dots)
            ys = self._rng.normal(pt[1], spread, n_dots)
            ws = self._rng.uniform(0.5, 1.8, n_dots)
            for x, y, w in zip(xs, ys, ws, strict=True):
                self._ctx.set_source_rgba(r, g, b, a * self._brush.opacity)
                self._ctx.arc(x, y, w, 0, math.pi * 2)
                self._ctx.fill()

    def _stroke_line(
        self,
        points: list[tuple[float, float]],
        width: float,
        alpha: float,
        jitter: float = 0.0,
        operator=None,
    ) -> None:
        """Lay the smoothed path down with a line style."""
        ctx = self._ctx
        r, g, b, a = self._color
        ctx.save()
        if operator is not None:
            ctx.set_operator(operator)
        ctx.set_source_rgba(r, g, b, a * alpha)
        ctx.set_line_width(width)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        if jitter > 0:
            pts = [
                (x + float(self._rng.normal(0, jitter)), y + float(self._rng.normal(0, jitter)))
                for x, y in points
            ]
        else:
            pts = points
        self._smooth_path(ctx, pts)
        ctx.stroke()
        ctx.restore()

    def stroke(self, points: list[tuple[float, float]]) -> None:
        """Draw one stroke through ``points`` with the current brush."""
        if not points:
            return
        points = [(max(0, min(self.width, x)), max(0, min(self.height, y))) for x, y in points]
        b = self._brush
        if b.erase:
            self._stroke_line(
                points,
                b.size * 2.0,
                1.0,
                operator=cairo.OPERATOR_DEST_OUT,
            )
        elif b.scatter > 0.25:
            self._stamp_scatter(points)
        elif b.hardness < 0.5:
            self._stamp_soft(points)
        else:
            op = cairo.OPERATOR_MULTIPLY if b.blend == "multiply" else None
            jitter = b.texture * b.size * 0.3
            self._stroke_line(points, b.size, b.opacity, jitter, op)
            # A rough utensil skids — a second faint offset pass.
            if b.texture > 0.3:
                self._stroke_line(
                    points,
                    b.size * 0.6,
                    b.opacity * 0.5,
                    jitter * 1.6,
                    op,
                )

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """A straight stroke between two points."""
        self.stroke([(x1, y1), (x2, y2)])

    def _shape_style(self, fill: bool) -> None:
        """Prime the paint context for a shape outline or fill."""
        r, g, b, a = self._color
        self._ctx.set_source_rgba(r, g, b, a * self._brush.opacity)
        if not fill:
            self._ctx.set_line_width(max(0.6, self._brush.size * 0.5))

    def ellipse(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        """An ellipse (or circle) inscribed in the given rect."""
        cx, cy = x + w / 2, y + h / 2
        rx, ry = max(w / 2, 0.5), max(h / 2, 0.5)
        if fill:
            self._ctx.save()
            self._shape_style(True)
            self._ctx.translate(cx, cy)
            self._ctx.scale(rx, ry)
            self._ctx.arc(0, 0, 1, 0, math.pi * 2)
            self._ctx.fill()
            self._ctx.restore()
        else:
            self._shape_style(False)
            n = 48
            pts = [
                (cx + rx * math.cos(t), cy + ry * math.sin(t))
                for t in (i * 2 * math.pi / n for i in range(n + 1))
            ]
            self._smooth_path(self._ctx, pts)
            self._ctx.stroke()

    def rect(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        """A rectangle, outlined or filled."""
        self._shape_style(fill)
        if fill:
            self._ctx.rectangle(x, y, w, h)
            self._ctx.fill()
        else:
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            self._smooth_path(self._ctx, pts)
            self._ctx.stroke()

    def polygon(self, points: list[tuple[float, float]], fill: bool = False) -> None:
        """A closed polygon through ``points``, outlined or filled."""
        if len(points) < 3:
            return
        self._shape_style(fill)
        self._ctx.new_path()
        self._ctx.move_to(*points[0])
        for pt in points[1:]:
            self._ctx.line_to(*pt)
        self._ctx.close_path()
        if fill:
            self._ctx.fill()
        else:
            self._ctx.stroke()

    def fill(self, x: float, y: float, tolerance: float = 24.0) -> bool:
        """Flood-fill the region under (x, y) with the current color.

        The fill sees the composite — paper plus paint — and repaints
        the contiguous region on the paint layer.
        """
        seed = (int(max(0, min(self.width - 1, x))), int(max(0, min(self.height - 1, y))))
        comp = self._composite_rgb()
        if cv2 is not None:
            mask = np.zeros((self.height + 2, self.width + 2), np.uint8)
            lo = (tolerance, tolerance, tolerance)
            flags = 8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
            try:
                cv2.floodFill(comp, mask, seed, 0, lo, lo, flags)
            except cv2.error:
                return False
            region = mask[1:-1, 1:-1] == 255
        else:
            region = _flood_mask_np(comp, seed, tolerance)
        if not region.any():
            return False
        paint = self._paint_array()
        r, g, b, a = self._color
        # ARGB32 is premultiplied — store color × alpha.
        paint[region, 0] = int(b * a * 255)
        paint[region, 1] = int(g * a * 255)
        paint[region, 2] = int(r * a * 255)
        paint[region, 3] = int(a * 255)
        self._paint.mark_dirty()
        return True

    def clear(self) -> None:
        """Wipe the paint layer — back to bare paper."""
        self._ctx.save()
        self._ctx.set_operator(cairo.OPERATOR_CLEAR)
        self._ctx.paint()
        self._ctx.restore()

    # ── Output ────────────────────────────────────────────────────

    def _paint_array(self) -> np.ndarray:
        """The paint layer as a writable BGRA array."""
        buf = self._paint.get_data()
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 4)

    def _paint_alpha(self) -> np.ndarray:
        return self._paint_array()[:, :, 3]

    def _composite_rgb(self) -> np.ndarray:
        """Paper + paint composited to a BGR uint8 array."""
        out = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(out)
        ctx.set_source_surface(self._paper, 0, 0)
        ctx.paint()
        ctx.set_source_surface(self._paint, 0, 0)
        ctx.paint()
        out.flush()
        arr = np.frombuffer(out.get_data(), dtype=np.uint8).reshape(self.height, self.width, 4)
        return arr[:, :, [2, 1, 0]].copy()

    def to_image(self) -> Image.Image:
        """The finished sheet as a PIL image."""
        return Image.fromarray(self._composite_rgb())


class Canvas:
    """Genesis's paint program — paper, color, and utensils.

    She drives it with operations: ``new_sheet``, ``set_color``,
    ``pick_tool``, ``stroke``, ``ellipse``/``rect``/``polygon``,
    ``fill``, ``clear``, and ``finish`` to keep the result. When a
    Krita bridge is live the same ops are mirrored into a real Krita
    document; otherwise the built-in rasterizer is the paper.

    The program keeps a small memory of what she has made —
    ``style_profile.json`` in the drawings directory — so she can
    notice what she keeps reaching for.
    """

    _HISTORY_LIMIT = _MAX_SAVED_DRAWINGS
    _HISTORY_FILE = "style_profile.json"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        seed: int | None = None,
        use_krita: bool = True,
    ) -> None:
        """Initialize the program.

        Args:
            data_dir: Directory where drawings (and her art history)
                are persisted. If None, nothing can be saved.
            seed: Optional RNG seed for reproducible mark texture.
            use_krita: Mirror ops into a running Krita when reachable.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._rng = np.random.default_rng(seed=seed if seed is not None else int(time.time()))
        self._history = self._load_history()
        self._sheet: PaintSheet | None = None
        self._op_log: list[dict] = []
        self._krita: KritaStudio | None = None
        self._krita_tried = False
        self._use_krita = use_krita
        self.width = _SHEET_WIDTH
        self.height = _SHEET_HEIGHT

    # ── Krita bridge ──────────────────────────────────────────────

    def _studio(self) -> KritaStudio | None:
        """The Krita bridge — connected lazily, never required."""
        if not self._use_krita:
            return None
        if self._krita is None and not self._krita_tried:
            self._krita_tried = True
            try:
                from frontal_lobe.krita_studio import KritaStudio

                studio = KritaStudio()
                if studio.connect(timeout=2.0):
                    self._krita = studio
            except Exception as e:  # noqa: BLE001
                logger.debug("krita bridge unavailable: %s", e)
        return self._krita

    def _mirror(self, op: str, **args) -> None:
        """Forward an op to Krita when the bridge is live."""
        studio = self._studio()
        if studio is not None:
            studio.send(op, **args)

    # ── Art history ───────────────────────────────────────────────

    @property
    def _history_path(self) -> Path | None:
        if self._data_dir is None:
            return None
        return self._data_dir / self._HISTORY_FILE

    def _load_history(self) -> list[dict]:
        """Load the drawing history, tolerating a missing/corrupt file."""
        path = self._history_path
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("could not read %s: %s", path, e)
            return []
        drawings = data.get("drawings") if isinstance(data, dict) else None
        if not isinstance(drawings, list):
            return []
        return [d for d in drawings if isinstance(d, dict)]

    @property
    def art_history(self) -> list[dict]:
        """A copy of her drawing history (oldest first)."""
        return list(self._history)

    def _recency(
        self, key: str, universe: tuple[str, ...] | list[str], now: float | None = None
    ) -> dict[str, float]:
        """Seconds since each entry in ``universe`` was last used."""
        now = time.time() if now is None else now
        recency = dict.fromkeys(universe, math.inf)
        for record in self._history:
            ts = record.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            for item in record.get(key) or ():
                if item in recency:
                    recency[item] = min(recency[item], now - float(ts))
        return recency

    def tool_recency(self, now: float | None = None) -> dict[str, float]:
        """Seconds since each tool name was last used.

        Only names she has actually used appear — the vocabulary is
        hers, so there is no fixed universe to enumerate.
        """
        now = time.time() if now is None else now
        recency: dict[str, float] = {}
        for record in self._history:
            ts = record.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            for item in record.get("tools") or ():
                recency[item] = min(recency.get(item, math.inf), now - float(ts))
        return recency

    def gesture_recency(self, now: float | None = None) -> dict[str, float]:
        """Seconds since each gesture family was last used."""
        gestures = (
            "line",
            "polyline",
            "curve",
            "arc",
            "loop",
            "zigzag",
            "scribble",
            "spiral",
            "wave",
            "dabs",
            "shape",
            "fill",
        )
        return self._recency("gestures", gestures, now)

    def recent_hues(self, limit: int = 12) -> list[float]:
        """Hue degrees used in her most recent drawings."""
        hues: list[float] = []
        for record in reversed(self._history[-limit:]):
            for hue in record.get("hues") or ():
                if isinstance(hue, (int, float)):
                    hues.append(float(hue))
        return hues

    def _record_drawing(
        self, result: DrawingResult, gestures: list[str], hues: list[float]
    ) -> None:
        """Append a drawing to her art history and persist it."""
        path = self._history_path
        if path is None:
            return
        self._history.append(
            {
                "ts": result.timestamp,
                "tools": list(result.tools_used),
                "colors": list(result.colors_used),
                "gestures": gestures,
                "hues": hues,
                "strokes": result.strokes,
                "mood": result.mood,
            }
        )
        del self._history[: max(0, len(self._history) - self._HISTORY_LIMIT)]
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"version": 1, "drawings": self._history}))
            os.replace(tmp, path)
        except OSError as e:
            logger.warning("could not write %s: %s", path, e)

    # ── The program ───────────────────────────────────────────────

    def new_sheet(
        self,
        width: int = _SHEET_WIDTH,
        height: int = _SHEET_HEIGHT,
        paper: tuple[int, int, int] = (250, 250, 248),
    ) -> bool:
        """Lay out a fresh sheet. Returns False when no backend can draw."""
        if not _CAIRO_AVAILABLE and self._studio() is None:
            if not getattr(self, "_warned", False):
                logger.warning(
                    "no painting surface available — Cairo is missing and Krita is not reachable."
                )
                self._warned = True
            return False
        self._sheet = PaintSheet(width, height, paper, self._rng) if _CAIRO_AVAILABLE else None
        self.width = width
        self.height = height
        self._op_log = []
        self._mirror("new", width=width, height=height, paper=list(paper))
        return True

    @property
    def sheet(self) -> PaintSheet | None:
        """The active raster sheet, when the built-in path is in use."""
        return self._sheet

    def set_color(self, r: int, g: int, b: int, a: float = 1.0) -> None:
        self._op_log.append({"op": "color", "rgb": (r, g, b), "a": a})
        if self._sheet is not None:
            self._sheet.set_color(r, g, b, a)
        self._mirror("color", r=r, g=g, b=b, a=a)

    def pick_tool(self, tool: str, size: float | None = None) -> None:
        """Take up what she calls ``tool`` — her label, her settings."""
        self._op_log.append({"op": "tool", "tool": tool, "size": size})
        if self._sheet is not None:
            self._sheet.pick_tool(tool, size)
        self._mirror("tool", tool=tool, size=size or 0)

    def set_brush(
        self,
        size: float | None = None,
        opacity: float | None = None,
        hardness: float | None = None,
        scatter: float | None = None,
        texture: float | None = None,
        erase: bool | None = None,
        blend: str | None = None,
    ) -> None:
        """Dial the current brush — any utensil is a point here."""
        self._op_log.append(
            {
                "op": "brush",
                "size": size,
                "opacity": opacity,
                "hardness": hardness,
                "scatter": scatter,
                "texture": texture,
                "erase": erase,
                "blend": blend,
            }
        )
        if self._sheet is not None:
            self._sheet.set_brush(size, opacity, hardness, scatter, texture, erase, blend)
        self._mirror(
            "brush",
            size=size or 0,
            opacity=opacity or 0,
            hardness=hardness or 0,
            scatter=scatter or 0,
            texture=texture or 0,
            erase=bool(erase),
            blend=blend or "",
        )

    def stroke(self, points: list[tuple[float, float]]) -> None:
        if not points:
            return
        self._op_log.append({"op": "stroke", "n": len(points)})
        if self._sheet is not None:
            self._sheet.stroke(points)
        self._mirror("stroke", points=[[round(x, 1), round(y, 1)] for x, y in points])

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.stroke([(x1, y1), (x2, y2)])

    def ellipse(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        self._op_log.append({"op": "ellipse", "fill": fill})
        if self._sheet is not None:
            self._sheet.ellipse(x, y, w, h, fill)
        self._mirror("ellipse", x=x, y=y, w=w, h=h, fill=fill)

    def rect(self, x: float, y: float, w: float, h: float, fill: bool = False) -> None:
        self._op_log.append({"op": "rect", "fill": fill})
        if self._sheet is not None:
            self._sheet.rect(x, y, w, h, fill)
        self._mirror("rect", x=x, y=y, w=w, h=h, fill=fill)

    def polygon(self, points: list[tuple[float, float]], fill: bool = False) -> None:
        if len(points) < 3:
            return
        self._op_log.append({"op": "polygon", "fill": fill})
        if self._sheet is not None:
            self._sheet.polygon(points, fill)
        self._mirror("polygon", points=points, fill=fill)

    def fill(self, x: float, y: float) -> None:
        self._op_log.append({"op": "fill", "at": (x, y)})
        if self._sheet is not None:
            self._sheet.fill(x, y)
        self._mirror("fill", x=x, y=y)

    def clear(self) -> None:
        self._op_log.append({"op": "clear"})
        if self._sheet is not None:
            self._sheet.clear()
        self._mirror("clear")

    def finish(
        self,
        description: str = "",
        mood: str = "",
        emotion: EmotionalState | None = None,
    ) -> DrawingResult | None:
        """Finish the sheet — save it and remember it.

        Returns the DrawingResult, or None when there is nowhere to
        save or no backend produced an image.
        """
        if emotion is not None:
            mood = f"feeling {emotion.label}"

        tools = sorted({entry["tool"] for entry in self._op_log if entry["op"] == "tool"})
        colors = sorted({entry["rgb"] for entry in self._op_log if entry["op"] == "color"})
        gestures = sorted({entry["gesture"] for entry in self._op_log if entry.get("gesture")})
        hues = sorted({entry["hue"] for entry in self._op_log if entry.get("hue") is not None})
        n_strokes = sum(
            1
            for entry in self._op_log
            if entry["op"] in ("stroke", "ellipse", "rect", "polygon", "fill")
        )

        filepath = ""
        timestamp = time.time()
        if self._sheet is not None:
            if self._data_dir is None:
                logger.warning("no data directory set — drawing not saved")
                return None
            self._data_dir.mkdir(parents=True, exist_ok=True)
            path = self._data_dir / f"drawing_{int(timestamp)}.webp"
            # Two sittings in the same second must not collide.
            n = 0
            while path.exists():
                n += 1
                path = self._data_dir / f"drawing_{int(timestamp)}_{n}.webp"
            self._sheet.to_image().save(
                str(path), format="WEBP", lossless=True, quality=100, method=4
            )
            self._prune_old_drawings()
            filepath = str(path)
        else:
            studio = self._studio()
            if studio is not None and self._data_dir is not None:
                self._data_dir.mkdir(parents=True, exist_ok=True)
                path = self._data_dir / f"drawing_{int(timestamp)}.png"
                if studio.export(str(path)):
                    filepath = str(path)

        result = DrawingResult(
            filepath=filepath,
            width=self.width,
            height=self.height,
            description=description,
            timestamp=timestamp,
            mood=mood,
            tools_used=tools,
            colors_used=[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors],
            strokes=n_strokes,
        )
        self._record_drawing(result, gestures, hues)
        self._sheet = None
        return result

    # Internal hooks for the session driver — it records intent into
    # the op log so history remembers the gesture/hue she chose.
    def _tag(self, gesture: str | None = None, hue: float | None = None) -> None:
        if gesture is not None or hue is not None:
            self._op_log.append({"op": "tag", "gesture": gesture, "hue": hue})

    def _prune_old_drawings(self) -> None:
        """Keep the newest ``_MAX_SAVED_DRAWINGS`` files; delete older."""
        if self._data_dir is None:
            return
        candidates: list[tuple[float, Path]] = []
        for path in self._data_dir.glob("drawing_*.webp"):
            stem = path.stem.removeprefix("drawing_")
            try:
                if "_" in stem:
                    base, seq = stem.rsplit("_", 1)
                    ts = int(base) + int(seq) / 1000
                else:
                    ts = float(stem)
            except ValueError:
                continue
            candidates.append((ts, path))
        candidates.sort()
        excess = len(candidates) - _MAX_SAVED_DRAWINGS
        for _, path in candidates[: max(excess, 0)]:
            try:
                path.unlink()
            except OSError as e:
                logger.warning("could not prune old drawing %s: %s", path, e)


def _flood_mask_np(
    comp: np.ndarray,
    seed: tuple[int, int],
    tolerance: float,
) -> np.ndarray:
    """``cv2.floodFill`` MASK_ONLY in NumPy — the cv2-absent path.

    Same semantics: 8-connectivity region growing where a candidate
    pixel joins when every channel sits within ``tolerance`` of the
    pixel it was reached from (the non-FIXED_RANGE comparison), the
    seed included. Returns the (H, W) boolean region.
    """
    h, w = comp.shape[:2]
    region = np.zeros((h, w), bool)
    c = comp.astype(np.int32)
    x0, y0 = seed
    region[y0, x0] = True
    stack = [(y0, x0)]
    while stack:
        y, x = stack.pop()
        base = c[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if (dy == 0 and dx == 0) or not (0 <= ny < h and 0 <= nx < w):
                    continue
                if region[ny, nx]:
                    continue
                if np.abs(c[ny, nx] - base).max() <= tolerance:
                    region[ny, nx] = True
                    stack.append((ny, nx))
    return region

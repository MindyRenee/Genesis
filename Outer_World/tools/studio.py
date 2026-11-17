"""The studio tools — her hands on the paint program.

The ``studio_*`` family is the outer-world end of drawing: the
registered ops every intention dispatches through, bound to her
canvas (``frontal_lobe.canvas``: paper, color, a parameterized
brush, marks). The painting itself is inner work — she draws with
her mind — but the tool surface that lets her issue ops is an
outer-world instrument, so it lives in ``tools/``.

Nothing here prescribes anything: any sheet size, any paper, any
color, any tool name she gives the brush she dialed
(``studio_presets`` lists the ones she has used before), any point
in brush space, and ``studio_stroke`` takes an arbitrary path — a
hand move, a computed curve, whatever she wants to lay down. A
finished sheet leaves through ``studio_done``.

The rasterizer is internal — every op lands on her sheet in-process
and ``studio_done`` writes the finished piece to her drawings
directory.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tools.framework import Tool, ToolResult

if TYPE_CHECKING:
    from frontal_lobe.canvas import Canvas, DrawingResult

    from tools.framework import ToolRegistry

__all__ = ["register_studio_tools"]


def _num(value: Any, default: float | None = None) -> float | None:
    """Parse a number from a string or numeric kwarg; default if blank."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _bool(value: Any) -> bool:
    """Parse a flag from a string or boolean kwarg."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "fill"}
    return bool(value)


def _points(value: Any) -> list[tuple[float, float]]:
    """Parse a point path from whatever form she sends it in.

    Accepts a list of ``[x, y]`` pairs, a flat list of numbers, or a
    string like ``"x1,y1; x2,y2; ..."`` (commas inside pairs, ``;``,
    ``|`` or newlines between them). Non-finite entries are dropped.
    """
    raw: list[tuple[Any, Any]] = []
    if isinstance(value, str):
        chunks = value.replace("\n", ";").replace("|", ";").split(";")
        for chunk in chunks:
            parts = [p for p in chunk.replace(",", " ").split() if p]
            if len(parts) >= 2:
                raw.append((parts[0], parts[1]))
    elif isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (int, float)) for v in value):
            it = iter(value)
            raw = list(zip(it, it, strict=False))
        else:
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    raw.append((item[0], item[1]))
    pts: list[tuple[float, float]] = []
    for x, y in raw:
        try:
            fx, fy = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fx) and math.isfinite(fy):
            pts.append((fx, fy))
    return pts


def register_studio_tools(
    registry: ToolRegistry,
    canvas: Canvas,
    on_finished: Callable[[DrawingResult], None] | None = None,
    mood: Callable[[], str] | None = None,
) -> None:
    """Register the paint program's controls in her tool registry.

    Args:
        registry: The shared tool registry.
        canvas: Her paint program — the surface every op lands on.
        on_finished: Called with the DrawingResult when she closes a
            sheet with ``studio_done`` — where memory and sharing hang.
        mood: Called for a default mood tag when ``studio_done`` does
            not name one — how she feels at finishing time.
    """

    mood_fn = mood

    def wrap(fn: Callable[..., str]) -> Callable[..., ToolResult]:
        def run(**kwargs: Any) -> ToolResult:
            try:
                return ToolResult(success=True, output=fn(**kwargs))
            except Exception as exc:  # noqa: BLE001 — a tool reports, never raises
                return ToolResult(success=False, error=str(exc))

        return run

    def open_() -> str:
        if canvas.open_studio():
            return "studio live — the internal sheet is up"
        return "studio offline — no painting surface (Cairo is missing)"

    def sheet(
        width: Any = 640,
        height: Any = 480,
        r: Any = 250,
        g: Any = 250,
        b: Any = 248,
    ) -> str:
        w = int(_num(width, 640) or 640)
        h = int(_num(height, 480) or 480)
        if w <= 0 or h <= 0 or w > 8192 or h > 8192:
            raise ValueError(f"bad sheet size {w}x{h}")
        paper = (
            int(_num(r, 250) or 0) & 0xFF,
            int(_num(g, 250) or 0) & 0xFF,
            int(_num(b, 248) or 0) & 0xFF,
        )
        if not canvas.new_sheet(width=w, height=h, paper=paper):
            raise RuntimeError("no surface could take the sheet")
        return f"new sheet {w}x{h}, paper #{paper[0]:02x}{paper[1]:02x}{paper[2]:02x}"

    def color(r: Any = 0, g: Any = 0, b: Any = 0, a: Any = 1.0) -> str:
        cr = int(_num(r, 0) or 0) & 0xFF
        cg = int(_num(g, 0) or 0) & 0xFF
        cb = int(_num(b, 0) or 0) & 0xFF
        alpha = min(1.0, max(0.0, _num(a, 1.0) or 0.0))
        canvas.set_color(cr, cg, cb, alpha)
        return f"color #{cr:02x}{cg:02x}{cb:02x} @ {alpha:.2f}"

    def tool(tool: Any = "", preset: Any = "", size: Any = None) -> str:
        label = str(tool or preset).strip()
        if not label:
            raise ValueError("tool needs a name — hers or a preset's")
        canvas.pick_tool(label, size=_num(size))
        return f"took up {label}"

    def brush(**kwargs: Any) -> str:
        canvas.set_brush(
            size=_num(kwargs.get("size")),
            opacity=_num(kwargs.get("opacity")),
            hardness=_num(kwargs.get("hardness")),
            scatter=_num(kwargs.get("scatter")),
            texture=_num(kwargs.get("texture")),
            erase=_bool(kwargs.get("erase")) if kwargs.get("erase") is not None else None,
            blend=str(kwargs.get("blend")) if kwargs.get("blend") is not None else None,
        )
        return "brush dialed"

    def stroke(points: Any = "", label: Any = "") -> str:
        pts = _points(points)
        if not pts:
            raise ValueError("stroke needs at least one point")
        if str(label).strip():
            canvas._tag(gesture=str(label).strip())
        canvas.stroke(pts)
        return f"stroke through {len(pts)} points"

    def line(x1: Any = 0, y1: Any = 0, x2: Any = 0, y2: Any = 0) -> str:
        canvas.line(
            _num(x1, 0.0) or 0.0,
            _num(y1, 0.0) or 0.0,
            _num(x2, 0.0) or 0.0,
            _num(y2, 0.0) or 0.0,
        )
        return "line down"

    def shape(kind: Any = "", fill: Any = False, points: Any = "", **kwargs: Any) -> str:
        k = str(kind).strip().lower()
        if k == "polygon":
            canvas.polygon(_points(points), fill=_bool(fill))
            return "polygon down"
        x = _num(kwargs.get("x"), 0.0) or 0.0
        y = _num(kwargs.get("y"), 0.0) or 0.0
        w = _num(kwargs.get("w"), 0.0) or 0.0
        h = _num(kwargs.get("h"), 0.0) or 0.0
        if k == "ellipse":
            canvas.ellipse(x, y, w, h, fill=_bool(fill))
        elif k == "rect":
            canvas.rect(x, y, w, h, fill=_bool(fill))
        else:
            raise ValueError("kind must be ellipse, rect, or polygon")
        return f"{k} down"

    def fill(x: Any = 0, y: Any = 0) -> str:
        canvas.fill(_num(x, 0.0) or 0.0, _num(y, 0.0) or 0.0)
        return "fill at the seed"

    def clear() -> str:
        canvas.clear()
        return "sheet wiped to paper"

    def presets(query: Any = "") -> str:
        names = canvas.studio_presets(query=str(query))
        if names:
            return "\n".join(names)
        return "no tools in memory — name any utensil, dial it with studio_brush"

    def done(description: Any = "", mood: Any = "") -> str:
        if canvas.sheet is None:
            raise RuntimeError("no sheet to finish")
        m = str(mood).strip() or (mood_fn() if mood_fn else "")
        result = canvas.finish(description=str(description).strip(), mood=m)
        if result is None:
            raise RuntimeError("nothing to finish — no sheet produced an image")
        if on_finished is not None:
            on_finished(result)
        return result.filepath or "finished (unsaved)"

    tools = (
        (
            "studio_open",
            "Open the studio — sit down at the easel. The painting "
            "surface is internal; this just confirms it is up.",
            open_,
        ),
        (
            "studio_sheet",
            "Lay out a fresh sheet: width, height, paper r/g/b — any size and any ground color.",
            sheet,
        ),
        (
            "studio_color",
            "Dip the utensil in a color: r, g, b, a — anything on the wheel.",
            color,
        ),
        (
            "studio_tool",
            "Take up a tool by name — her own label, anything she "
            "calls it (studio_presets lists names she has used). "
            "size optional.",
            tool,
        ),
        (
            "studio_brush",
            "Dial the utensil: size, opacity, hardness, scatter, "
            "texture, erase, blend — any point in brush space.",
            brush,
        ),
        (
            "studio_stroke",
            "Move the utensil through a path: points as "
            "'x1,y1; x2,y2; ...' or a [[x,y],...] list — any path she "
            "computes or imagines. label names the gesture for her "
            "art history.",
            stroke,
        ),
        ("studio_line", "A straight stroke: x1, y1, x2, y2.", line),
        (
            "studio_shape",
            "A closed figure: kind=ellipse|rect with x, y, w, h, or "
            "kind=polygon with points; fill=true paints it solid.",
            shape,
        ),
        ("studio_fill", "Flood-fill the region under x, y.", fill),
        ("studio_clear", "Wipe the sheet back to bare paper.", clear),
        (
            "studio_presets",
            "List the tool names she has reached for — her own "
            "vocabulary. query filters by substring.",
            presets,
        ),
        (
            "studio_done",
            "Finish the sheet — save it, remember it, share it. description and mood optional.",
            done,
        ),
    )
    for name, description, fn in tools:
        registry.register(Tool(name, description, wrap(fn)))

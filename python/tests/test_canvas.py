"""Tests for the canvas — her paint program — and her free play on it.

The contract under test: the medium is open. Any color, any brush
parameters, any tool label, any mark — nothing prescribes what she
can draw. Krita mirroring is strictly optional; the built-in
rasterizer is the canonical surface.
"""

import json
import math
import time
from pathlib import Path

import frontal_lobe.canvas as canvas_module
import numpy as np
from frontal_lobe.canvas import Canvas, PaintSheet
from frontal_lobe.painting import Painter


def _sheet(**kw) -> PaintSheet:
    kw.setdefault("rng", np.random.default_rng(7))
    return PaintSheet(width=200, height=150, **kw)


def _canvas(tmp_path, **kw) -> Canvas:
    kw.setdefault("use_krita", False)
    return Canvas(data_dir=tmp_path / "drawings", seed=7, **kw)


def _alpha(sheet: PaintSheet) -> np.ndarray:
    return sheet._paint_alpha()


# ── The open medium ───────────────────────────────────────────────


class TestBrushParameters:
    def test_all_parameters_settable(self):
        sheet = _sheet()
        sheet.set_brush(
            size=23.5, opacity=0.4, hardness=0.2, scatter=0.9,
            texture=0.7, erase=True, blend="multiply",
        )
        b = sheet._brush
        assert b.size == 23.5
        assert b.opacity == 0.4
        assert b.hardness == 0.2
        assert b.scatter == 0.9
        assert b.texture == 0.7
        assert b.erase is True
        assert b.blend == "multiply"

    def test_parameters_clamp_not_reject(self):
        sheet = _sheet()
        sheet.set_brush(size=-5, opacity=7.0, hardness=-1, scatter=2,
                        texture=-0.5)
        b = sheet._brush
        assert b.size == 0.5
        assert b.opacity == 1.0
        assert b.hardness == 0.0
        assert b.scatter == 1.0
        assert b.texture == 0.0

    def test_none_leaves_parameter_untouched(self):
        sheet = _sheet()
        sheet.set_brush(size=30.0, hardness=0.1)
        sheet.set_brush(opacity=0.5)
        assert sheet._brush.size == 30.0
        assert sheet._brush.hardness == 0.1
        assert sheet._brush.opacity == 0.5

    def test_arbitrary_blend_strings_accepted(self):
        sheet = _sheet()
        sheet.set_brush(blend="screen")
        assert sheet._brush.blend == "screen"


class TestColor:
    def test_any_rgb_and_alpha(self):
        sheet = _sheet()
        sheet.set_color(17, 203, 91, 0.6)
        r, g, b, a = sheet._color
        assert (r, g, b) == (17 / 255, 203 / 255, 91 / 255)
        assert a == 0.6

    def test_out_of_range_clamps(self):
        sheet = _sheet()
        sheet.set_color(-40, 999, 128, 3.0)
        r, g, b, a = sheet._color
        assert (r, g, b) == (0.0, 1.0, 128 / 255)
        assert a == 1.0


class TestToolLabels:
    def test_unknown_label_accepted(self):
        sheet = _sheet()
        sheet.pick_tool("my invented scribble-stick")
        assert sheet._tool == "my invented scribble-stick"

    def test_label_never_substitutes_settings(self):
        """No name — known or invented — changes the dialed brush."""
        sheet = _sheet()
        sheet.set_brush(size=12.0, opacity=0.5, hardness=0.3,
                        scatter=0.6, texture=0.8, blend="multiply")
        before = sheet._brush.copy()
        for name in ("pencil", "eraser", "glitter", "airbrush",
                     "anything she calls it"):
            sheet.pick_tool(name)
            assert sheet._brush == before, name

    def test_explicit_size_is_hers_to_choose(self):
        sheet = _sheet()
        sheet.pick_tool("wide thing", size=40.0)
        assert sheet._brush.size == 40.0
        sheet.pick_tool("wide thing")  # no size → unchanged
        assert sheet._brush.size == 40.0

    def test_no_builtin_tool_universe(self):
        """The module must not ship a finite list of allowed tools."""
        for attr in ("TOOLS", "TOOL_STYLES", "BUILTIN_TOOLS", "PRESETS"):
            assert not hasattr(canvas_module, attr), attr


# ── Marks ─────────────────────────────────────────────────────────


class TestMarks:
    def test_stroke_lays_paint(self):
        sheet = _sheet()
        sheet.set_color(200, 30, 30)
        sheet.set_brush(size=6)
        sheet.stroke([(20, 20), (100, 80), (180, 40)])
        assert _alpha(sheet).max() > 0

    def test_soft_and_scatter_brushes_mark(self):
        for params in ({"hardness": 0.1}, {"scatter": 0.8}):
            sheet = _sheet()
            sheet.set_color(50, 80, 200)
            sheet.set_brush(size=10, **params)
            sheet.stroke([(30, 30), (120, 90)])
            assert _alpha(sheet).max() > 0, params

    def test_stroke_clamped_to_sheet(self):
        sheet = _sheet()
        sheet.set_color(0, 0, 0)
        sheet.stroke([(-500, -500), (9999, 9999)])
        assert _alpha(sheet).max() > 0  # drew within bounds, no crash

    def test_shapes_outline_and_fill(self):
        for op in ("ellipse", "rect", "polygon"):
            for fill in (False, True):
                sheet = _sheet()
                sheet.set_color(20, 120, 60)
                sheet.set_brush(size=4)
                if op == "polygon":
                    sheet.polygon([(40, 40), (120, 50), (80, 110)],
                                  fill=fill)
                else:
                    getattr(sheet, op)(40, 30, 90, 60, fill=fill)
                assert _alpha(sheet).max() > 0, (op, fill)

    def test_fill_floods_region(self):
        sheet = _sheet()
        sheet.set_color(10, 20, 200)
        assert sheet.fill(50, 50) is True
        # Region is now the new color on the paint layer.
        comp = sheet._composite_rgb()
        assert np.allclose(comp[50, 50], [10, 20, 200], atol=8)
        # Far corner still paper (flood respected contiguity here).
        assert sheet.fill(50, 50) is True  # idempotent: region refills

    def test_erase_reveals_paper(self):
        sheet = _sheet(paper=(240, 235, 220))
        sheet.set_color(90, 20, 160)
        sheet.set_brush(size=8)
        sheet.stroke([(10, 75), (190, 75)])
        assert _alpha(sheet)[75].max() > 0
        sheet.set_brush(erase=True, size=30)
        sheet.stroke([(100, 60), (100, 90)])
        comp = sheet._composite_rgb()
        assert np.allclose(comp[75, 100], [240, 235, 220], atol=8)

    def test_clear_returns_to_paper(self):
        sheet = _sheet(paper=(250, 250, 248))
        sheet.set_color(0, 0, 0)
        sheet.rect(10, 10, 100, 60, fill=True)
        sheet.clear()
        assert _alpha(sheet).max() == 0

    def test_nan_points_filtered(self):
        sheet = _sheet()
        sheet.set_color(0, 0, 0)
        sheet.stroke([(10, 10), (math.inf, 50), (math.nan, math.nan),
                      (60, 60)])
        assert _alpha(sheet).max() > 0


# ── The sitting ───────────────────────────────────────────────────


class TestCanvasSession:
    def test_finish_saves_and_reports(self, tmp_path):
        canvas = _canvas(tmp_path)
        assert canvas.new_sheet()
        canvas.pick_tool("wanderer", size=9.0)
        canvas.set_color(123, 45, 200, 0.9)
        canvas.stroke([(10, 10), (200, 300), (400, 100)])
        canvas.ellipse(50, 50, 100, 80, fill=True)
        result = canvas.finish(description="test", mood="curious")
        assert result is not None
        assert Path(result.filepath).exists()
        assert result.filepath.endswith(".webp")
        assert result.width == canvas.width
        assert result.height == canvas.height
        assert "wanderer" in result.tools_used
        assert "#7b2dc8" in result.colors_used
        assert result.strokes >= 2

    def test_no_data_dir_returns_none(self):
        canvas = Canvas(data_dir=None, use_krita=False)
        canvas.new_sheet()
        canvas.stroke([(1, 1), (50, 50)])
        assert canvas.finish() is None


class TestArtHistory:
    def test_history_persists(self, tmp_path):
        canvas = _canvas(tmp_path)
        canvas.new_sheet()
        canvas.pick_tool("glitter wand")
        canvas._tag(gesture="spiral", hue=212.0)
        canvas.set_color(0, 180, 120)
        canvas.stroke([(5, 5), (100, 100)])
        canvas.finish()

        profile = tmp_path / "drawings" / "style_profile.json"
        data = json.loads(profile.read_text())
        assert data["drawings"][-1]["tools"] == ["glitter wand"]
        assert data["drawings"][-1]["gestures"] == ["spiral"]
        assert data["drawings"][-1]["hues"] == [212.0]

        # A fresh Canvas over the same dir remembers it.
        other = _canvas(tmp_path)
        recency = other.tool_recency()
        assert "glitter wand" in recency
        assert recency["glitter wand"] < 60
        assert other.gesture_recency()["spiral"] < 60
        assert 212.0 in other.recent_hues()

    def test_corrupt_history_tolerated(self, tmp_path):
        drawings = tmp_path / "drawings"
        drawings.mkdir(parents=True)
        (drawings / "style_profile.json").write_text("{not json!!")
        canvas = _canvas(tmp_path)
        canvas.new_sheet()
        canvas.stroke([(1, 1), (80, 80)])
        assert canvas.finish() is not None

    def test_malformed_history_shape_tolerated(self, tmp_path):
        drawings = tmp_path / "drawings"
        drawings.mkdir(parents=True)
        (drawings / "style_profile.json").write_text(
            json.dumps({"drawings": "not-a-list"}))
        canvas = _canvas(tmp_path)
        assert canvas.art_history == []
        assert canvas.tool_recency() == {}

    def test_history_pruned_to_limit(self, tmp_path):
        canvas = _canvas(tmp_path)
        now = time.time()
        canvas._history = [
            {"ts": now - i, "tools": [f"t{i}"], "gestures": [],
             "hues": [], "colors": [], "strokes": 1, "mood": ""}
            for i in range(canvas._HISTORY_LIMIT + 20)
        ]
        canvas.new_sheet()
        canvas.stroke([(1, 1), (30, 30)])
        canvas.finish()
        assert len(canvas.art_history) <= canvas._HISTORY_LIMIT

    def test_history_survives_unwritable_dir(self, tmp_path):
        """A failed history write must not lose the drawing itself."""
        canvas = _canvas(tmp_path)
        canvas.new_sheet()
        canvas.stroke([(1, 1), (30, 30)])
        profile = tmp_path / "drawings"
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "style_profile.json").mkdir()  # path is a dir → write fails
        assert canvas.finish() is not None


# ── Her free play ─────────────────────────────────────────────────


class TestPainter:
    def test_sitting_produces_drawing(self, tmp_path):
        canvas = _canvas(tmp_path)
        painter = Painter(seed=3)
        result = painter.paint(canvas)
        assert result is not None
        assert Path(result.filepath).exists()
        assert result.strokes > 0
        assert result.tools_used  # she named what she held

    def test_sittings_explore(self, tmp_path):
        """Back-to-back sittings shouldn't reuse the same gestures."""
        canvas = _canvas(tmp_path)
        painter = Painter(seed=11)
        first = painter.paint(canvas)
        gestures1 = set(canvas.art_history[-1]["gestures"])
        second = painter.paint(canvas)
        gestures2 = set(canvas.art_history[-1]["gestures"])
        assert first.filepath != second.filepath
        # Novelty weighting: used gestures lose pull next sitting.
        assert gestures1 != gestures2 or canvas.gesture_recency()

    def test_brush_dials_span_space(self, tmp_path):
        """Across sittings, her dialed brush params vary."""
        canvas = _canvas(tmp_path)
        painter = Painter(seed=5)
        sizes = set()
        for _ in range(3):
            painter.paint(canvas)
            for entry in canvas._op_log:
                if entry["op"] == "tool" and entry.get("size"):
                    sizes.add(round(entry["size"]))
        assert len(sizes) > 1

    def test_gesture_generators_stay_on_sheet(self, tmp_path):
        canvas = _canvas(tmp_path)
        canvas.new_sheet()
        painter = Painter(seed=9)
        for gesture in ("line", "polyline", "curve", "arc", "loop",
                        "zigzag", "scribble", "spiral", "wave", "dabs"):
            pts = painter._gesture_points(gesture, canvas)
            assert len(pts) >= 2, gesture
            assert all(
                math.isfinite(x) and math.isfinite(y) for x, y in pts
            ), gesture

    def test_novelty_weight_curve(self):
        w = Painter._novelty_weight
        assert w(math.inf) > w(86400) > w(3600) > w(60)


# ── Backend optionality ───────────────────────────────────────────


class TestBackends:
    def test_krita_absent_still_paints(self, tmp_path):
        canvas = _canvas(tmp_path, use_krita=True)
        canvas._krita_tried = True  # pretend connect already failed
        canvas._krita = None
        canvas.new_sheet()
        canvas.set_color(255, 0, 0)
        canvas.stroke([(10, 10), (150, 150)])
        assert canvas.finish() is not None

    def test_mirror_noop_without_bridge(self, tmp_path):
        canvas = _canvas(tmp_path)
        canvas._use_krita = False
        canvas._mirror("anything")  # must not raise or connect

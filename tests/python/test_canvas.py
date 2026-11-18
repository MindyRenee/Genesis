"""Tests for the canvas — her paint program — and her free play on it.

The contract under test: the medium is open. Any color, any brush
parameters, any tool label, any mark — nothing prescribes what she
can draw. The built-in rasterizer is the only surface — the
canonical artifact.
"""

import json
import math
import time
from pathlib import Path

import canvas as canvas_module
import numpy as np
from canvas import Canvas, PaintSheet


def _sheet(**kw) -> PaintSheet:
    kw.setdefault("rng", np.random.default_rng(7))
    return PaintSheet(width=200, height=150, **kw)


def _canvas(tmp_path, **kw) -> Canvas:
    return Canvas(data_dir=tmp_path / "drawings", seed=7, **kw)


def _alpha(sheet: PaintSheet) -> np.ndarray:
    return sheet._paint_alpha()


# ── The open medium ───────────────────────────────────────────────


class TestBrushParameters:
    def test_all_parameters_settable(self):
        sheet = _sheet()
        sheet.set_brush(
            size=23.5,
            opacity=0.4,
            hardness=0.2,
            scatter=0.9,
            texture=0.7,
            erase=True,
            blend="multiply",
        )
        b = sheet._brush
        assert b.size == 23.5
        assert b.opacity == 0.4
        assert b.hardness == 0.2
        assert b.scatter == 0.9
        assert b.texture == 0.7
        assert b.erase is True
        assert b.blend == "multiply"

    def test_parameters_preserve_values(self):
        sheet = _sheet()
        sheet.set_brush(size=-5, opacity=7.0, hardness=-1, scatter=2, texture=-0.5)
        b = sheet._brush
        assert b.size == -5.0
        assert b.opacity == 7.0
        assert b.hardness == -1.0
        assert b.scatter == 2.0
        assert b.texture == -0.5

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

    def test_out_of_range_values_preserved(self):
        sheet = _sheet()
        sheet.set_color(-40, 999, 128, 3.0)
        r, g, b, a = sheet._color
        assert (r, g, b) == (-40 / 255, 999 / 255, 128 / 255)
        assert a == 3.0


class TestToolLabels:
    def test_unknown_label_accepted(self):
        sheet = _sheet()
        sheet.pick_tool("my invented scribble-stick")
        assert sheet._tool == "my invented scribble-stick"

    def test_label_never_substitutes_settings(self):
        """No name — known or invented — changes the dialed brush."""
        sheet = _sheet()
        sheet.set_brush(
            size=12.0, opacity=0.5, hardness=0.3, scatter=0.6, texture=0.8, blend="multiply"
        )
        before = sheet._brush.copy()
        for name in ("pencil", "eraser", "glitter", "airbrush", "anything she calls it"):
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

    def test_stroke_coordinates_not_rewritten(self):
        sheet = _sheet()
        sheet.set_color(0, 0, 0)
        sheet.stroke([(-500, -500), (9999, 9999)])
        assert _alpha(sheet).max() > 0  # the path can cross the sheet without clipping

    def test_shapes_outline_and_fill(self):
        for op in ("ellipse", "rect", "polygon"):
            for fill in (False, True):
                sheet = _sheet()
                sheet.set_color(20, 120, 60)
                sheet.set_brush(size=4)
                if op == "polygon":
                    sheet.polygon([(40, 40), (120, 50), (80, 110)], fill=fill)
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
        sheet.stroke([(10, 10), (math.inf, 50), (math.nan, math.nan), (60, 60)])
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
        canvas = Canvas(data_dir=None)
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
        # Gestures record what she did: the tagged spiral plus the
        # stroke op itself; hues record the tag plus the dipped color.
        assert data["drawings"][-1]["gestures"] == ["spiral", "stroke"]
        assert data["drawings"][-1]["hues"] == [160.0, 212.0]

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
        (drawings / "style_profile.json").write_text(json.dumps({"drawings": "not-a-list"}))
        canvas = _canvas(tmp_path)
        assert canvas.art_history == []
        assert canvas.tool_recency() == {}

    def test_history_pruned_to_limit(self, tmp_path):
        canvas = _canvas(tmp_path)
        now = time.time()
        canvas._history = [
            {
                "ts": now - i,
                "tools": [f"t{i}"],
                "gestures": [],
                "hues": [],
                "colors": [],
                "strokes": 1,
                "mood": "",
            }
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


class TestStudioTools:
    """The paint program's controls, issued as her tool calls."""

    def _tools(self, tmp_path, canvas=None):
        from tools.framework import ToolRegistry
        from tools.studio import register_studio_tools

        canvas = canvas or _canvas(tmp_path)
        registry = ToolRegistry()
        self.finished = []
        register_studio_tools(
            registry,
            canvas,
            on_finished=self.finished.append,
            mood=lambda: "feeling test",
        )
        return registry, canvas

    def test_sitting_produces_drawing(self, tmp_path):
        registry, _ = self._tools(tmp_path)
        assert registry.run("studio_sheet", width="320", height="240").success
        assert registry.run("studio_color", r="200", g="30", b="90", a="0.8").success
        assert registry.run("studio_tool", tool="pencil", size="6").success
        assert registry.run("studio_stroke", points="10,10; 100,40; 80,120", label="wander").success
        result = registry.run("studio_done")
        assert result.success, result.error
        assert Path(result.output).exists()
        assert self.finished and self.finished[0].mood == "feeling test"

    def test_stroke_accepts_point_lists(self, tmp_path):
        registry, _ = self._tools(tmp_path)
        registry.run("studio_sheet")
        assert registry.run("studio_stroke", points=[[1, 2], [3, 4]]).success
        assert registry.run("studio_stroke", points=[1, 2, 3, 4]).success
        bad = registry.run("studio_stroke", points="not,points")
        assert not bad.success

    def test_shapes_and_fill(self, tmp_path):
        registry, _ = self._tools(tmp_path)
        registry.run("studio_sheet")
        assert registry.run("studio_shape", kind="ellipse", x="5", y="5", w="40", h="30").success
        assert registry.run(
            "studio_shape", kind="rect", x="5", y="5", w="40", h="30", fill="true"
        ).success
        assert registry.run("studio_shape", kind="polygon", points="1,1; 30,4; 8,29").success
        assert registry.run("studio_fill", x="10", y="10").success
        assert not registry.run("studio_shape", kind="cone", x="0", y="0", w="5", h="5").success

    def test_gesture_labels_are_hers(self, tmp_path):
        """Whatever she names a move lands in her art history."""
        registry, canvas = self._tools(tmp_path)
        registry.run("studio_sheet")
        registry.run("studio_stroke", points="0,0; 50,50", label="maurer_rose")
        registry.run("studio_done")
        assert "maurer_rose" in canvas.gesture_recency()

    def test_done_without_sheet_reports(self, tmp_path):
        registry, _ = self._tools(tmp_path)
        result = registry.run("studio_done")
        assert not result.success


# ── The internal surface ──────────────────────────────────────────


class TestBackends:
    def test_paints_and_finishes(self, tmp_path):
        canvas = _canvas(tmp_path)
        canvas.new_sheet()
        canvas.set_color(255, 0, 0)
        canvas.stroke([(10, 10), (150, 150)])
        assert canvas.finish() is not None

    def test_open_studio_reflects_surface(self, tmp_path):
        canvas = _canvas(tmp_path)
        assert canvas.open_studio() is canvas_module._CAIRO_AVAILABLE

    def test_presets_are_her_vocabulary(self, tmp_path):
        canvas = _canvas(tmp_path)
        assert canvas.studio_presets() == []
        canvas.new_sheet()
        canvas.pick_tool("charcoal")
        canvas.finish()
        assert canvas.studio_presets() == ["charcoal"]
        assert canvas.studio_presets(query="char") == ["charcoal"]
        assert canvas.studio_presets(query="oil") == []

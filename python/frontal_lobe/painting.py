"""Painting — how she moves when she sits down to draw.

This module is *her side* of the program: the part that decides where
her hand goes and what it carries. It is deliberately free play — the
way a person explores a paint program is by trying things. She picks
a color anywhere on the wheel, dials a brush anywhere in parameter
space (any size, any softness, any scatter — including lifting paint
off the sheet again), and makes a gesture anywhere on the page.

Her choices are steered by two honest forces:

- **Her memory of what she has made.** ``style_profile.json`` records
  every finished piece — which brush characters, gestures, and hues
  it used. Anything she reached for recently loses pull; anything she
  has never tried pulls harder. That is boredom driving exploration —
  the same novelty pressure any artist feels.
- **Whatever is on her mind.** If concepts are active when she picks
  up the program (a conversation about spirals, a dream about the
  sea), names and shapes she already knows can tug her hand. It is a
  lean, never a command — her emotions work this way too, inside her
  cognition rather than wired to the page.

There is no prescription here: no fixed technique list, no required
subject, no chemical pipeline. The medium (``canvas.Canvas``) supplies
paper, color, and a parameterized brush; this supplies motion.
"""

from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from frontal_lobe.canvas import Canvas, DrawingResult
    from limbic_system.emotion import EmotionalState

__all__ = ["GESTURES", "Painter"]

# Gesture families — the ways a hand can move. These are motor
# primitives, not subjects: a stroke is still hers to place, size,
# and combine however she wants.
GESTURES = (
    "line", "polyline", "curve", "arc", "loop", "zigzag",
    "scribble", "spiral", "wave", "dabs",
)

# Informal labels for what a dialed brush *resembles* — used only to
# name her utensil in history and (on a Krita backend) to find a real
# preset that feels similar. The brush itself is always her own
# parameters; the label is a description, not a selection.


class Painter:
    """Her hand at the program — free, exploratory mark-making.

    Each call to :meth:`paint` is one sitting: a fresh sheet, a run of
    strokes she chooses move by move, then the finished piece. Nothing
    is planned ahead of time — she explores, and the history of what
    she has explored leans her toward what she hasn't.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ── A sitting ─────────────────────────────────────────────────

    def paint(
        self,
        canvas: Canvas,
        emotion: EmotionalState | None = None,
    ) -> DrawingResult | None:
        """One sitting at the program. Returns the finished drawing."""
        if not canvas.new_sheet(paper=self._pick_paper()):
            return None
        n_moves = int(self._rng.integers(6, 26))
        for _ in range(n_moves):
            self._one_move(canvas)
        return canvas.finish(emotion=emotion)

    # ── Choices ───────────────────────────────────────────────────

    def _pick_paper(self) -> tuple[int, int, int]:
        """The sheet she works on — usually light, but her call."""
        v = float(self._rng.uniform(0.0, 1.0))
        if v < 0.8:
            # Most paper is pale — a light tint of any hue.
            r, g, b = colorsys.hsv_to_rgb(
                float(self._rng.uniform(0, 1)), 0.12, 0.96
            )
        else:
            # Sometimes she wants a dark or colored ground.
            r, g, b = colorsys.hsv_to_rgb(
                float(self._rng.uniform(0, 1)),
                float(self._rng.uniform(0.1, 0.5)),
                float(self._rng.uniform(0.15, 0.55)),
            )
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _novelty_weight(seconds_since: float) -> float:
        """Boredom curve: recently-touched things lose their pull."""
        if math.isinf(seconds_since):
            return 3.0      # never tried — maximum pull
        if seconds_since < 1800:
            return 0.1      # just used — nearly no pull
        if seconds_since < 7200:
            return 0.4
        if seconds_since < 86400:
            return 1.5
        return 2.0

    def _pick_gesture(self, canvas: Canvas) -> str:
        """How her hand moves this time, novelty-weighted."""
        recency = canvas.gesture_recency()
        names = list(GESTURES)
        weights = np.array(
            [self._novelty_weight(recency.get(g, math.inf)) for g in names]
        )
        weights /= weights.sum()
        return str(self._rng.choice(names, p=weights))

    def _pick_color(self, canvas: Canvas) -> tuple[int, int, int, float]:
        """A color anywhere on the wheel, leaning toward new hues."""
        # Thirty-degree sectors; recently-used sectors lose pull.
        recent = canvas.recent_hues()
        sector_pull = np.ones(12)
        for hue in recent:
            sector_pull[int(hue // 30) % 12] = 0.25
        sector_pull /= sector_pull.sum()
        sector = int(self._rng.choice(12, p=sector_pull))
        hue = (sector * 30 + float(self._rng.uniform(0, 30))) % 360
        r, g, b = colorsys.hsv_to_rgb(
            hue / 360,
            float(self._rng.uniform(0.35, 1.0)),
            float(self._rng.uniform(0.45, 1.0)),
        )
        alpha = float(self._rng.uniform(0.5, 1.0))
        canvas._tag(hue=hue)
        return (int(r * 255), int(g * 255), int(b * 255), alpha)

    def _dial_brush(self) -> dict:
        """Any point in brush space — her own implement this stroke."""
        brush = {
            "size": float(math.exp(self._rng.uniform(
                math.log(1.0), math.log(45.0)))),
            "opacity": float(self._rng.uniform(0.35, 1.0)),
            "hardness": float(self._rng.uniform(0.0, 1.0)),
            "scatter": 0.0,
            "texture": float(self._rng.uniform(0.0, 0.5)),
            "erase": False,
            "blend": "normal",
        }
        if self._rng.uniform() < 0.25:
            brush["scatter"] = float(self._rng.uniform(0.3, 0.95))
        if self._rng.uniform() < 0.2:
            brush["texture"] = float(self._rng.uniform(0.5, 1.0))
        if self._rng.uniform() < 0.08:
            brush["erase"] = True
        if self._rng.uniform() < 0.15:
            brush["blend"] = "multiply"
        return brush

    @staticmethod
    def _name_it(brush: dict) -> str:
        """What the implement she dialed resembles — her label for it."""
        if brush["erase"]:
            return "eraser"
        if brush["scatter"] > 0.3:
            return "spray"
        if brush["hardness"] < 0.45:
            return "brush"
        if brush["texture"] > 0.45:
            return "crayon"
        if brush["blend"] == "multiply":
            return "marker"
        if brush["size"] < 3.5:
            return "pencil"
        if brush["size"] > 12:
            return "marker"
        return "pen"

    # ── Gestures — the shapes a hand makes ────────────────────────

    def _rand_point(self, canvas: Canvas) -> tuple[float, float]:
        return (
            float(self._rng.uniform(0, canvas.width)),
            float(self._rng.uniform(0, canvas.height)),
        )

    def _gesture_points(self, gesture: str, canvas: Canvas
                        ) -> list[tuple[float, float]]:
        w, h = canvas.width, canvas.height
        rng = self._rng
        if gesture == "line":
            return [self._rand_point(canvas), self._rand_point(canvas)]
        if gesture == "polyline":
            return [
                self._rand_point(canvas)
                for _ in range(int(rng.integers(3, 7)))
            ]
        if gesture == "curve":
            p0 = self._rand_point(canvas)
            p3 = self._rand_point(canvas)
            c1 = self._rand_point(canvas)
            c2 = self._rand_point(canvas)
            pts: list[tuple[float, float]] = []
            for t in np.linspace(0, 1, 24):
                x = float((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * c1[0]
                          + 3 * (1 - t) * t ** 2 * c2[0] + t ** 3 * p3[0])
                y = float((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * c1[1]
                          + 3 * (1 - t) * t ** 2 * c2[1] + t ** 3 * p3[1])
                pts.append((x, y))
            return pts
        if gesture == "arc":
            cx, cy = self._rand_point(canvas)
            r = float(rng.uniform(20, min(w, h) * 0.45))
            a0 = float(rng.uniform(0, math.pi * 2))
            sweep = float(rng.uniform(0.3, math.pi * 1.6))
            return [
                (cx + r * math.cos(a0 + t), cy + r * math.sin(a0 + t))
                for t in np.linspace(0, sweep, 24)
            ]
        if gesture == "loop":
            cx, cy = self._rand_point(canvas)
            rx = float(rng.uniform(15, w * 0.35))
            ry = float(rng.uniform(15, h * 0.35))
            phase = float(rng.uniform(0, math.pi * 2))
            n = int(rng.integers(1, 4))
            return [
                (cx + rx * math.cos(phase + t),
                 cy + ry * math.sin(phase + t))
                for t in np.linspace(0, n * 2 * math.pi, 48)
            ]
        if gesture == "zigzag":
            x0, y0 = self._rand_point(canvas)
            x1, y1 = self._rand_point(canvas)
            n = int(rng.integers(4, 10))
            px, py = -(y1 - y0), x1 - x0
            norm = math.hypot(px, py) or 1.0
            amp = float(rng.uniform(10, 60))
            return [
                (x0 + (x1 - x0) * i / n
                 + (px / norm) * amp * (1 if i % 2 else -1),
                 y0 + (y1 - y0) * i / n
                 + (py / norm) * amp * (1 if i % 2 else -1))
                for i in range(n + 1)
            ]
        if gesture == "scribble":
            x, y = self._rand_point(canvas)
            pts = [(x, y)]
            for _ in range(int(rng.integers(10, 40))):
                x = float(x + rng.normal(0, 18))
                y = float(y + rng.normal(0, 18))
                pts.append((x, y))
            return pts
        if gesture == "spiral":
            cx, cy = self._rand_point(canvas)
            turns = float(rng.uniform(1.0, 3.5))
            r0 = float(rng.uniform(3, 12))
            r1 = float(rng.uniform(30, min(w, h) * 0.4))
            return [
                (cx + (r0 + (r1 - r0) * t / (turns * 2 * math.pi))
                 * math.cos(t),
                 cy + (r0 + (r1 - r0) * t / (turns * 2 * math.pi))
                 * math.sin(t))
                for t in np.linspace(0, turns * 2 * math.pi, 72)
            ]
        if gesture == "wave":
            x0 = float(rng.uniform(0, w * 0.3))
            x1 = float(rng.uniform(w * 0.6, w))
            yc = float(rng.uniform(h * 0.2, h * 0.8))
            amp = float(rng.uniform(10, 80))
            freq = float(rng.uniform(1.5, 5.0))
            return [
                (x, yc + amp * math.sin((x - x0) / (x1 - x0)
                                        * freq * 2 * math.pi))
                for x in np.linspace(x0, x1, 40)
            ]
        if gesture == "dabs":
            x0, y0 = self._rand_point(canvas)
            x1, y1 = self._rand_point(canvas)
            n = int(rng.integers(4, 14))
            return [
                (x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1))
                for i in range(n)
            ]
        return [self._rand_point(canvas), self._rand_point(canvas)]

    def _one_move(self, canvas: Canvas) -> None:
        """One mark: pick up an implement, dip a color, move."""
        gesture = self._gesture_pick(canvas)
        canvas._tag(gesture=gesture)

        if gesture == "shape":
            self._make_shape(canvas)
            return
        if gesture == "fill":
            brush = self._dial_brush()
            r, g, b, a = self._pick_color(canvas)
            canvas.pick_tool("fill")
            canvas.set_color(r, g, b, a)
            x, y = self._rand_point(canvas)
            canvas.fill(x, y)
            return

        brush = self._dial_brush()
        canvas.pick_tool(self._name_it(brush), size=brush["size"])
        canvas.set_brush(
            opacity=brush["opacity"], hardness=brush["hardness"],
            scatter=brush["scatter"], texture=brush["texture"],
            erase=brush["erase"], blend=brush["blend"],
        )
        r, g, b, a = self._pick_color(canvas)
        canvas.set_color(r, g, b, a)
        canvas.stroke(self._gesture_points(gesture, canvas))

    def _gesture_pick(self, canvas: Canvas) -> str:
        """Gesture plus the rare program-level moves (shape, fill)."""
        pick = self._pick_gesture(canvas)
        roll = float(self._rng.uniform())
        if roll < 0.06:
            return "fill"
        if roll < 0.18:
            return "shape"
        return pick

    def _make_shape(self, canvas: Canvas) -> None:
        """A closed figure — ellipse, rect, or polygon; outline or solid."""
        brush = self._dial_brush()
        r, g, b, a = self._pick_color(canvas)
        canvas.pick_tool(self._name_it(brush), size=brush["size"])
        canvas.set_brush(opacity=brush["opacity"], erase=brush["erase"],
                         blend=brush["blend"])
        canvas.set_color(r, g, b, a)
        kind = str(self._rng.choice(("ellipse", "rect", "polygon")))
        fill = bool(self._rng.uniform() < 0.4)
        w, h = canvas.width, canvas.height
        if kind == "ellipse":
            x = float(self._rng.uniform(0, w * 0.7))
            y = float(self._rng.uniform(0, h * 0.7))
            canvas.ellipse(
                x, y,
                float(self._rng.uniform(15, w - x)),
                float(self._rng.uniform(15, h - y)),
                fill=fill,
            )
        elif kind == "rect":
            x = float(self._rng.uniform(0, w * 0.7))
            y = float(self._rng.uniform(0, h * 0.7))
            canvas.rect(
                x, y,
                float(self._rng.uniform(15, w - x)),
                float(self._rng.uniform(15, h - y)),
                fill=fill,
            )
        else:
            n = int(self._rng.integers(3, 7))
            canvas.polygon(
                [self._rand_point(canvas) for _ in range(n)], fill=fill
            )

"""Shape sorter — matching under conjunctive constraints.

The toddler's canonical puzzle: blocks each carry attributes (shape,
color, size, …) and the box has apertures, each stating a constraint
("star-shaped opening"). A block goes through an aperture only when it
satisfies *every* attribute the aperture names. The vocabulary is
open — ``accepts`` is a dict of attribute equalities — so the same
world expresses color sorting, size ordering, letter-to-family
matching, or any other "does this fit that" task.

The model is honest: ``insert`` reports how many constraints the block
satisfied (observable — you can compare a block to its hole) but never
pre-filters, so the agent discovers that full matches are what count
rather than being told. The lid is the developmental distractor: a big
opening that always accepts a block yet fills no aperture. What looks
like success isn't — the toddler's first lesson in "working" vs
"solving." ``empty_top`` recovers whatever the lid swallowed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# Attribute vocab for generated sorters. Instances given as data (an
# offered puzzle, a described task) may use any attributes at all —
# these only shape the built-in generator.
_SHAPES = ("star", "circle", "square", "triangle", "hexagon", "heart")
_COLORS = ("red", "blue", "yellow", "green", "purple", "orange")
_SIZES = (1, 2, 3)


@dataclass(frozen=True)
class Block:
    """One block: a small bundle of comparable attributes."""

    block_id: int
    attrs: tuple[tuple[str, str], ...]

    def get(self, attr: str) -> str | None:
        return dict(self.attrs).get(attr)


@dataclass(frozen=True)
class Slot:
    """One aperture: every named attribute must match to pass."""

    index: int
    accepts: tuple[tuple[str, str], ...]

    @property
    def needed(self) -> int:
        return len(self.accepts)


@dataclass
class Insertion:
    """One candidate move: a block against an aperture."""

    action: str  # "insert" | "lid" | "empty_top"
    block_id: int = -1
    slot: int = -1
    matched: int = -1
    needed: int = 0


class ShapeSorter:
    """The box: apertures with constraints, blocks, and a lid."""

    def __init__(
        self,
        slots: list[Slot],
        blocks: list[Block],
        has_lid: bool = True,
    ) -> None:
        self.slots = list(slots)
        self.has_lid = has_lid
        self.pool: dict[int, Block] = {b.block_id: b for b in blocks}
        self.board: dict[int, Block] = {}  # slot index -> seated block
        self.in_top: list[Block] = []

    @classmethod
    def generate(
        cls,
        n_slots: int,
        rng: random.Random,
        difficulty: int = 1,
        decoys: int = 0,
        has_lid: bool = True,
    ) -> ShapeSorter:
        """A guaranteed-solvable sorter.

        ``difficulty`` is constraint arity: 1 constrains shape alone
        (the toddler sorter), 2 adds size, 3 adds color — each level
        demands more of "satisfy *every* constraint." Each slot gets
        one dedicated block; ``decoys`` adds near-misses that match
        all but one attribute of some slot.
        """
        shapes = rng.sample(_SHAPES, min(n_slots, len(_SHAPES)))
        slots: list[Slot] = []
        blocks: list[Block] = []
        for i in range(n_slots):
            accepts = [("shape", shapes[i])]
            if difficulty >= 2:
                accepts.append(("size", str(rng.choice(_SIZES))))
            if difficulty >= 3:
                accepts.append(("color", rng.choice(_COLORS)))
            slots.append(Slot(i, tuple(accepts)))
            attrs = dict(accepts)
            attrs.setdefault("color", rng.choice(_COLORS))
            attrs.setdefault("size", str(rng.choice(_SIZES)))
            blocks.append(Block(len(blocks), tuple(sorted(attrs.items()))))
        for _ in range(decoys):
            slot = rng.choice(slots)
            attrs = dict(slot.accepts)
            # Break exactly one constraint — the near-miss.
            attr, _ = rng.choice(list(slot.accepts))
            if attr == "shape":
                attrs[attr] = rng.choice(
                    [s for s in _SHAPES if s != attrs[attr]]
                )
            elif attr == "color":
                attrs[attr] = rng.choice(
                    [c for c in _COLORS if c != attrs[attr]]
                )
            else:
                attrs[attr] = rng.choice(
                    [str(s) for s in _SIZES if str(s) != attrs[attr]]
                )
            for extra, vocab in (
                ("color", _COLORS),
                ("size", _SIZES),
            ):
                attrs.setdefault(extra, str(rng.choice(vocab)))
            blocks.append(Block(len(blocks), tuple(sorted(attrs.items()))))
        return cls(slots, blocks, has_lid=has_lid)

    # ── World queries ─────────────────────────────────────────

    def matched_attrs(self, block: Block, slot: Slot) -> int:
        """How many of the aperture's constraints the block satisfies."""
        return sum(
            1 for attr, value in slot.accepts if block.get(attr) == value
        )

    def mismatches(self) -> int:
        """Unfilled apertures — zero means solved."""
        return len(self.slots) - len(self.board)

    def complete(self) -> bool:
        return self.mismatches() == 0

    def candidates(self) -> list[Insertion]:
        """Every available move: inserts, the lid, dumping the box."""
        out: list[Insertion] = []
        open_slots = [s for s in self.slots if s.index not in self.board]
        for block in self.pool.values():
            for slot in open_slots:
                out.append(
                    Insertion(
                        "insert",
                        block.block_id,
                        slot.index,
                        self.matched_attrs(block, slot),
                        slot.needed,
                    )
                )
            if self.has_lid:
                out.append(Insertion("lid", block.block_id))
        if self.in_top:
            out.append(Insertion("empty_top"))
        return out

    def goal_state(self) -> dict[str, Any]:
        return {"slots.unfilled": self.mismatches()}

    def state(self) -> dict[str, Any]:
        return {
            "slots.unfilled": self.mismatches(),
            "blocks.placed": len(self.board),
            "blocks.pool": len(self.pool),
            "blocks.in_top": len(self.in_top),
        }

    # ── Actions ───────────────────────────────────────────────

    def insert(self, block_id: int, slot_index: int) -> dict[str, Any] | None:
        """Try a block against an aperture. None if the move is illegal."""
        block = self.pool.get(block_id)
        slot = next(
            (s for s in self.slots if s.index == slot_index), None
        )
        if block is None or slot is None or slot.index in self.board:
            return None
        matched = self.matched_attrs(block, slot)
        before = self.mismatches()
        accepted = matched == slot.needed
        if accepted:
            self.board[slot.index] = block
            del self.pool[block_id]
        return {
            "accepted": accepted,
            "matched": matched,
            "needed": slot.needed,
            "delta_unfilled": self.mismatches() - before,
        }

    def insert_top(self, block_id: int) -> dict[str, Any] | None:
        """Drop a block through the lid. Always "works"; solves nothing."""
        block = self.pool.get(block_id)
        if block is None or not self.has_lid:
            return None
        del self.pool[block_id]
        self.in_top.append(block)
        return {
            "accepted": True,
            "lid": True,
            "delta_unfilled": 0,
        }

    def empty_top(self) -> dict[str, Any] | None:
        """Open the box and tip the lid-swallowed blocks back out."""
        if not self.in_top:
            return None
        returned = len(self.in_top)
        for block in self.in_top:
            self.pool[block.block_id] = block
        self.in_top.clear()
        return {"returned": returned, "delta_unfilled": 0}


# ── Perceptual form ─────────────────────────────────────────────
# A sorter is a physical object: apertures and blocks have shape,
# size, and color. The symbolic attrs remain the world's truth —
# they are the physics — but an agent that *sees* gets only the
# rendered silhouette. Shape words heard in descriptions map onto
# the same geometry ("round" is a circle, "moon" a crescent), so a
# stated criterion is still checkable by eye.

_SHAPE_FORMS = {
    "circle", "round", "square", "triangle", "star", "moon",
    "hexagon", "heart", "diamond", "oval", "pentagon", "cross",
}
_COLOR_RGB = {
    "red": (220, 60, 55),
    "blue": (70, 110, 220),
    "yellow": (230, 200, 60),
    "green": (80, 175, 85),
    "purple": (150, 90, 200),
    "orange": (235, 145, 50),
    "pink": (235, 150, 190),
    "brown": (140, 95, 60),
    "white": (235, 235, 240),
    "gray": (150, 150, 155),
    "black": (40, 40, 45),
}
_SIZE_SCALE = {"1": 0.5, "2": 0.68, "3": 0.85}
_SIZE_WORDS = {
    "tiny": 0.45, "small": 0.55, "little": 0.55,
    "big": 0.8, "large": 0.85, "huge": 0.9,
}


def _shape_polygon(shape: str, cx: float, cy: float, r: float):
    """Vertices for a silhouette centered at (cx, cy)."""
    import math

    s = shape.lower()
    if s in ("circle", "round", "oval"):
        squash = 0.7 if s == "oval" else 1.0
        return [
            (cx + r * math.cos(t), cy + r * squash * math.sin(t))
            for t in [2 * math.pi * i / 48 for i in range(48)]
        ]
    if s == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if s == "triangle":
        return [(cx, cy - r), (cx + r * 0.9, cy + r * 0.7), (cx - r * 0.9, cy + r * 0.7)]
    if s == "diamond":
        return [(cx, cy - r), (cx + r * 0.75, cy), (cx, cy + r), (cx - r * 0.75, cy)]
    if s == "pentagon":
        return [
            (cx + r * math.cos(math.pi / 2 + 2 * math.pi * k / 5),
             cy - r * math.sin(math.pi / 2 + 2 * math.pi * k / 5))
            for k in range(5)
        ]
    if s == "hexagon":
        return [
            (cx + r * math.cos(2 * math.pi * k / 6),
             cy + r * math.sin(2 * math.pi * k / 6))
            for k in range(6)
        ]
    if s == "star":
        return [
            (
                cx + (r if k % 2 == 0 else r * 0.45) * math.cos(math.pi / 2 + k * math.pi / 5),
                cy - (r if k % 2 == 0 else r * 0.45) * math.sin(math.pi / 2 + k * math.pi / 5),
            )
            for k in range(10)
        ]
    if s == "heart":
        def hx(t: float) -> float:
            return cx + r * 0.06 * (16 * math.sin(t) ** 3)

        def hy(t: float) -> float:
            return cy - r * 0.06 * (
                13 * math.cos(t) - 5 * math.cos(2 * t)
                - 2 * math.cos(3 * t) - math.cos(4 * t)
            )

        return [
            (hx(t), hy(t))
            for t in [2 * math.pi * i / 48 for i in range(48)]
        ]
    if s == "moon":
        pts = [
            (cx + r * math.cos(t), cy + r * math.sin(t))
            for t in [4.4 + 4.0 * i / 36 for i in range(37)]
        ]
        pts += [
            (cx + r * 0.5 + r * 0.5 * math.cos(t), cy + r * 0.5 * math.sin(t))
            for t in [4.4 + 4.0 * i / 36 for i in range(37)]
        ][::-1]
        return pts
    if s == "cross":
        t = r * 0.32
        return [
            (cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t),
            (cx + r, cy - t), (cx + r, cy + t), (cx + t, cy + t),
            (cx + t, cy + r), (cx - t, cy + r), (cx - t, cy + t),
            (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t),
        ]
    # Unknown shape word → a blob the eye still distinguishes.
    sides = 7 + (sum(shape.encode()) % 3)
    return [
        (cx + r * math.cos(2 * math.pi * k / sides),
         cy + r * math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]


def render_object(attrs: dict, size: int = 64, rng: random.Random | None = None) -> Any:
    """Render a sorter object (block or aperture) as an RGB image.

    Shape comes from the "shape" attr or the first flat attr that
    names a form ("round", "star"); color from "color" or a flat
    color word; size from "size" or a size adjective. Mild position
    jitter makes each viewing differ slightly — real looking is
    never pixel-identical.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    shape = ""
    if str(attrs.get("shape", "")).lower() in _SHAPE_FORMS:
        shape = str(attrs["shape"]).lower()
    elif str(attrs.get("kind", "")).lower() in _SHAPE_FORMS:
        shape = str(attrs["kind"]).lower()
    else:
        for k in attrs:
            if str(k).lower() in _SHAPE_FORMS:
                shape = str(k).lower()
                break
    color = (235, 235, 240)
    for k, v in attrs.items():
        for word in (str(v).lower(), str(k).lower()):
            if word in _COLOR_RGB:
                color = _COLOR_RGB[word]
                break
    scale = 0.68
    for k, v in attrs.items():
        sv = str(v).lower()
        if str(k).lower() == "size" and sv in _SIZE_SCALE:
            scale = _SIZE_SCALE[sv]
        elif sv in _SIZE_WORDS:
            scale = _SIZE_WORDS[sv]
        elif str(k).lower() in _SIZE_WORDS:
            scale = _SIZE_WORDS[str(k).lower()]

    img = Image.new("RGB", (size, size), (30, 30, 34))
    d = ImageDraw.Draw(img)
    rng = rng or random.Random()
    jx = rng.uniform(-3, 3)
    jy = rng.uniform(-3, 3)
    r = size / 2 * scale
    pts = _shape_polygon(shape or "blob", size / 2 + jx, size / 2 + jy, r)
    d.polygon(pts, fill=color)
    return np.asarray(img, dtype=np.uint8)


class PerceptualSorter(ShapeSorter):
    """A sorter you must LOOK at. Same physics — an insert only
    succeeds when the block truly satisfies the aperture — but the
    oracle's judgment is not published on the candidates: ``matched``
    is hidden and the agent works from rendered views through the
    visual cortex. ``teach`` lets the world name its own objects,
    the way a parent points and says "that's a square".
    """

    def __init__(
        self,
        slots: list[Slot],
        blocks: list[Block],
        has_lid: bool = True,
        cortex: Any = None,
        seed: int = 0,
    ) -> None:
        super().__init__(slots, blocks, has_lid)
        self.cortex = cortex
        rng = random.Random(seed)
        self._images: dict[tuple[str, int], Any] = {}
        self._views: dict[tuple[str, int], Any] = {}
        self._names: dict[tuple[str, int], str] = {}
        for s in self.slots:
            a = dict(s.accepts)
            self._images[("slot", s.index)] = render_object(a, rng=rng)
            self._names[("slot", s.index)] = self._shape_name(a)
        for b in blocks:
            a = dict(b.attrs)
            self._images[("block", b.block_id)] = render_object(a, rng=rng)
            self._names[("block", b.block_id)] = self._shape_name(a)

    @staticmethod
    def _shape_name(attrs: dict) -> str:
        if str(attrs.get("shape", "")).lower() in _SHAPE_FORMS:
            return str(attrs["shape"]).lower()
        for k in attrs:
            if str(k).lower() in _SHAPE_FORMS:
                return str(k).lower()
        return ""

    def candidates(self) -> list[Insertion]:
        """The same moves, but the oracle doesn't leak: matched is
        unknown until a move is committed."""
        out = super().candidates()
        for c in out:
            c.matched, c.needed = -1, 0
        return out

    def view(self, kind: str, ident: int) -> Any:
        """The object's VTC feature vector — seen once, remembered."""
        key = (kind, ident)
        if key not in self._views and self.cortex is not None:
            percept = self.cortex.see(self._images[key])
            self._views[key] = percept.vtc_vector
        return self._views.get(key)

    def similarity(self, block_id: int, slot_index: int) -> float:
        """Cosine between a block's view and an aperture's view —
        what the eye says about fit, not what the oracle knows."""
        import numpy as np

        vb = self.view("block", block_id)
        vs = self.view("slot", slot_index)
        if vb is None or vs is None:
            return 0.0
        denom = np.linalg.norm(vb) * np.linalg.norm(vs)
        if denom <= 0:
            return 0.0
        return float(max(0.0, np.dot(vb, vs) / denom))

    def recognize(self, kind: str, ident: int) -> str:
        """What the cortex recognizes this object as, if anything —
        a name it has learned ("square"), not the world's label.
        Below the confidence floor the guess doesn't count — an
        undertrained bridge calls everything the same name."""
        if self.cortex is None:
            return ""
        v = self.view(kind, ident)
        if v is None:
            return ""
        name, conf = self.cortex.mtl.recognize(v)
        return name if name and conf >= 0.3 else ""

    def teach(self) -> None:
        """The world names what it showed: each seen object is bound
        to its true shape word — 'that's a square'."""
        if self.cortex is None:
            return
        for key, img in self._images.items():
            name = self._names.get(key, "")
            if name:
                try:
                    self.cortex.learn_from_image(img, name)
                except Exception:  # noqa: BLE001
                    pass

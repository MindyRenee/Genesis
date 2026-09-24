"""Transformation DSL — the hypothesis space for spatial reasoning.

A ``Transform`` is a named, parameterized grid→grid operation. The
spatial reasoner solves a task by finding a *sequence* of transforms
that maps every training input to its output. This is the perceptual
analogue of the concept network's typed edges: where semantic reasoning
traverses relations between concepts, spatial reasoning composes
operations over scenes.

Each proposal function inspects the training pairs and emits the
transform instances that could plausibly explain them (e.g. a recolor
map is only proposed when every input/output pair has the same cell
geometry). The solver then verifies candidates by exact replay — a
hypothesis is only kept if it reproduces *all* training outputs.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

from .grid import Grid
from .scene import PerceivedObject, _spans_overlap, perceive

_logger = logging.getLogger(__name__)

# A training pair: (input grid, output grid).
Example = tuple[Grid, Grid]

TransformFn = Callable[[Grid], Grid]


@dataclass(frozen=True)
class Transform:
    """A named grid→grid operation with the parameters that produced it.

    ``params`` documents *why* the transform was proposed — it is the
    symbolic content of the hypothesis, and what the grounding layer
    turns into conceptual structure when a solution is explained.
    """

    name: str
    apply: TransformFn
    params: dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.params:
            return self.name
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"

    def __call__(self, grid: Grid) -> Grid:
        return self.apply(grid)


# ── Proposals ─────────────────────────────────────────────────────
#
# Each proposer takes the training pairs and returns the transform
# instances consistent with what the pairs show. Proposers are the
# perceptual hypothesis generators: they look at evidence and suggest
# candidate operations. None of them verify — that is the solver's job.


def _propose_identity(_examples: list[Example]) -> list[Transform]:
    return [Transform("identity", lambda g: g)]


def _propose_rotations(_examples: list[Example]) -> list[Transform]:
    return [
        Transform("rotate90", lambda g: g.rotate90(1), {"turns": 1}),
        Transform("rotate180", lambda g: g.rotate90(2), {"turns": 2}),
        Transform("rotate270", lambda g: g.rotate90(3), {"turns": 3}),
    ]


def _propose_reflections(_examples: list[Example]) -> list[Transform]:
    return [
        Transform("reflect_h", lambda g: g.reflect_h()),
        Transform("reflect_v", lambda g: g.reflect_v()),
        Transform("reflect_diag", lambda g: g.reflect_diag()),
    ]


def _propose_recolor(examples: list[Example]) -> list[Transform]:
    """Propose the color map when every pair preserves cell geometry.

    If input and output are the same shape with the same pattern of
    *which* cells changed, the transformation may be a pure recoloring.
    The map is read directly off the first pair and must be consistent
    across all of them.
    """
    mapping: dict[int, int] = {}
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            target = out.at(r, c)
            if v in mapping and mapping[v] != target:
                return []  # ambiguous: same input color → two outputs
            mapping[v] = target
    if not mapping or all(k == v for k, v in mapping.items()):
        return []
    return [Transform("recolor", lambda g: g.recolor(mapping), {"map": mapping})]


def _propose_crop(examples: list[Example]) -> list[Transform]:
    """Propose cropping to content when every output is smaller."""
    if not all(
        out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]
        and out.shape != inp.shape
        for inp, out in examples
    ):
        return []
    return [
        Transform("crop_to_content", lambda g: g.crop_to_content()),
        Transform(
            "crop_to_content_bg0", lambda g: g.crop_to_content(background=0)
        ),
    ]


def _propose_scale(examples: list[Example]) -> list[Transform]:
    """Propose integer upscaling when output dims are a constant multiple."""
    ratios: set[int] = set()
    for inp, out in examples:
        if (
            inp.width == 0
            or out.width % inp.width != 0
            or out.height % inp.height != 0
        ):
            return []
        rw, rh = out.width // inp.width, out.height // inp.height
        if rw != rh or rw < 2:
            return []
        ratios.add(rw)
    if len(ratios) != 1:
        return []
    factor = ratios.pop()
    return [Transform("scale", lambda g: g.scale(factor), {"factor": factor})]


def _propose_tile(examples: list[Example]) -> list[Transform]:
    """Propose tiling when output dims are a constant multiple."""
    for inp, out in examples:
        if (
            inp.height == 0 or inp.width == 0
            or out.width % inp.width != 0
            or out.height % inp.height != 0
        ):
            return []
        rr, rc = out.height // inp.height, out.width // inp.width
        if rr == 1 and rc == 1:
            return []
    rr = examples[0][1].height // examples[0][0].height
    rc = examples[0][1].width // examples[0][0].width
    return [Transform("tile", lambda g: g.tile(rr, rc), {"reps": (rr, rc)})]


def _apply_gravity(
    grid: Grid, direction: str, color: int | None = None
) -> Grid:
    """Move objects to an edge; with ``color``, only that color falls."""
    scene = perceive(grid, compute_relations=False)
    # Move every object in one pass — clear all sources first, then
    # paint all destinations — so an earlier object's destination
    # can't be erased by a later object's source region.
    moves: list[tuple[PerceivedObject, int, int]] = []
    for obj in scene.objects:
        if color is not None and obj.color != color:
            continue
        top, left, bottom, right = obj.bbox
        dr = dc = 0
        if direction == "down":
            dr = grid.height - 1 - bottom
        elif direction == "up":
            dr = -top
        elif direction == "left":
            dc = -left
        else:  # right
            dc = grid.width - 1 - right
        moves.append((obj, dr, dc))
    if not moves:
        return grid
    rows = grid.to_lists()
    for obj, _, _ in moves:
        for r, c in obj.cells:
            rows[r][c] = scene.background
    for obj, dr, dc in moves:
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                rows[nr][nc] = obj.color
    return Grid(tuple(tuple(row) for row in rows))


def _apply_erase(grid: Grid, color: int) -> Grid:
    return grid.recolor({color: grid.most_common_color()})


def _apply_select_color(grid: Grid, color: int) -> Grid:
    scene = perceive(grid, background=0, compute_relations=False)
    cells = frozenset(
        cell
        for obj in scene.objects
        if obj.color == color
        for cell in obj.cells
    )
    return grid.crop_to_cells(cells) if cells else grid


def _apply_recolor_object(grid: Grid, which: str, color: int) -> Grid:
    scene = perceive(grid, compute_relations=False)
    if not scene.objects:
        return grid
    obj = (max if which == "largest" else min)(
        scene.objects, key=lambda o: o.size
    )
    return grid.overlay(obj.cells, color)


def _apply_fill_enclosed(grid: Grid, color: int) -> Grid:
    bg = grid.most_common_color()
    # Flood fill background from every border cell; whatever remains
    # unreachable is enclosed.
    from collections import deque

    seen: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c, v in grid.iter_cells():
        if v == bg and (r in (0, grid.height - 1) or c in (0, grid.width - 1)):
            queue.append((r, c))
            seen.add((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid.height
                and 0 <= nc < grid.width
                and (nr, nc) not in seen
                and grid.at(nr, nc) == bg
            ):
                seen.add((nr, nc))
                queue.append((nr, nc))
    enclosed = frozenset(
        (r, c)
        for r, c, v in grid.iter_cells()
        if v == bg and (r, c) not in seen
    )
    return grid.overlay(enclosed, color) if enclosed else grid


def _mk_gravity(direction: str, color: int | None = None) -> TransformFn:
    return lambda g: _apply_gravity(g, direction, color)


def _mk_erase(color: int) -> TransformFn:
    return lambda g: _apply_erase(g, color)


def _mk_select(color: int) -> TransformFn:
    return lambda g: _apply_select_color(g, color)


def _mk_recolor_object(which: str, color: int) -> TransformFn:
    return lambda g: _apply_recolor_object(g, which, color)


def _mk_fill_enclosed(color: int) -> TransformFn:
    return lambda g: _apply_fill_enclosed(g, color)


def _mk_mark_uniform(axis: str, color: int) -> TransformFn:
    return lambda g: _apply_mark_uniform(g, axis, color)


def _mk_shift(dr: int, dc: int, clamp: bool = False) -> TransformFn:
    return lambda g: _apply_shift(g, dr, dc, clamp)


def _apply_shift(
    grid: Grid, dr: int, dc: int, clamp: bool = False
) -> Grid:
    """Move every non-zero cell by a fixed offset.

    Cells that would leave the frame are clipped (``clamp=False``)
    or stopped at the boundary (``clamp=True``); vacated cells
    become 0. Unlike gravity this preserves the pattern's shape —
    it is a rigid translation, not a fall.
    """
    rows = [[0] * grid.width for _ in range(grid.height)]
    for r, c, v in grid.iter_cells():
        if v == 0:
            continue
        nr = min(max(r + dr, 0), grid.height - 1) if clamp else r + dr
        nc = min(max(c + dc, 0), grid.width - 1) if clamp else c + dc
        if 0 <= nr < grid.height and 0 <= nc < grid.width:
            rows[nr][nc] = v
    return Grid.from_lists(rows)


def _mk_link_blocks() -> TransformFn:
    return _apply_link_blocks


def _eight_conn_components(
    grid: Grid, color: int
) -> list[set[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r, c, v in grid.iter_cells():
        if v != color or (r, c) in seen:
            continue
        comp: set[tuple[int, int]] = set()
        queue = [(r, c)]
        seen.add((r, c))
        while queue:
            cr, cc = queue.pop()
            comp.add((cr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < grid.height
                        and 0 <= nc < grid.width
                        and (nr, nc) not in seen
                        and grid.at(nr, nc) == color
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
        comps.append(comp)
    return comps


def _apply_link_blocks(grid: Grid) -> Grid:
    """Answer 8 if some 8-connected region of 8s touches two distinct
    2x2 blocks of 2s, else 0. Output is a 1x1 verdict."""
    blocks = [
        comp for comp in _eight_conn_components(grid, 2)
        if len(comp) == 4
        and {r for r, _ in comp} == set(range(min(r for r, _ in comp), min(r for r, _ in comp) + 2))
        and {c for _, c in comp} == set(range(min(c for _, c in comp), min(c for _, c in comp) + 2))
    ]
    linked = False
    for region in _eight_conn_components(grid, 8):
        touched = 0
        for block in blocks:
            if any(
                abs(r - br) <= 1 and abs(c - bc) <= 1
                for r, c in region
                for br, bc in block
            ):
                touched += 1
                if touched == 2:
                    break
        if touched == 2:
            linked = True
            break
    return Grid.from_lists([[8 if linked else 0]])


_HaloTable = dict[int, tuple[tuple[int, int, int], ...]]


def _mk_cross_halos(table: _HaloTable) -> TransformFn:
    def _fn(grid: Grid) -> Grid:
        rows = grid.to_lists()
        for r, c, v in grid.iter_cells():
            halo = table.get(v)
            if halo is None:
                continue
            for dr, dc, k in halo:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < grid.height
                    and 0 <= nc < grid.width
                    and rows[nr][nc] == 0
                ):
                    rows[nr][nc] = k
        return Grid.from_lists(rows)
    return _fn


def _count_squares(grid: Grid, color: int) -> int:
    n = 0
    for r in range(grid.height - 1):
        for c in range(grid.width - 1):
            if (
                grid.at(r, c) == color
                and grid.at(r, c + 1) == color
                and grid.at(r + 1, c) == color
                and grid.at(r + 1, c + 1) == color
            ):
                n += 1
    return n


def _mk_count_blocks(color: int, width: int) -> TransformFn:
    def _fn(grid: Grid) -> Grid:
        n = _count_squares(grid, color)
        return Grid.from_lists([[color] * n + [0] * (width - n)])
    return _fn


def _mk_ray_recolor() -> TransformFn:
    return _apply_ray_recolor


def _apply_ray_recolor(grid: Grid) -> Grid:
    """Every seed cell aligned with the largest object casts a ray
    at it; the blob's edge cell facing the seed takes the seed's
    color."""
    scene = perceive(grid, compute_relations=False)
    if not scene.objects:
        return grid
    blob = max(scene.objects, key=lambda o: o.size)
    if blob.size < 2:
        return grid
    r_lo, c_lo, r_hi, c_hi = blob.bbox
    rows = grid.to_lists()
    for r, c, v in grid.iter_cells():
        if v == 0 or (r, c) in blob.cells:
            continue
        if r_lo <= r <= r_hi:
            span = [cc for rr, cc in blob.cells if rr == r]
            if not span:
                continue
            if c < c_lo:
                rows[r][min(span)] = v
            elif c > c_hi:
                rows[r][max(span)] = v
        elif c_lo <= c <= c_hi:
            span = [rr for rr, cc in blob.cells if cc == c]
            if not span:
                continue
            if r < r_lo:
                rows[min(span)][c] = v
            else:
                rows[max(span)][c] = v
    return Grid.from_lists(rows)


def _mk_nearest_border() -> TransformFn:
    return _apply_nearest_border


def _edge_color(cells: list[int]) -> int | None:
    return cells[0] if len(set(cells)) == 1 and cells[0] != 0 else None


def _apply_nearest_border(grid: Grid) -> Grid:
    """Two monochrome border lines flank the grid on opposite edges;
    every interior marker cell takes the nearer border's color."""
    left = _edge_color([grid.at(r, 0) for r in range(grid.height)])
    right = _edge_color(
        [grid.at(r, grid.width - 1) for r in range(grid.height)]
    )
    top = _edge_color([grid.at(0, c) for c in range(grid.width)])
    bot = _edge_color(
        [grid.at(grid.height - 1, c) for c in range(grid.width)]
    )
    if left is not None and right is not None:
        near, far, size = left, right, grid.width
        axis = "column"
    elif top is not None and bot is not None:
        near, far, size = top, bot, grid.height
        axis = "row"
    else:
        return grid
    rows = grid.to_lists()
    for r, c, v in grid.iter_cells():
        if v == 0 or v in (near, far):
            continue
        pos = c if axis == "column" else r
        rows[r][c] = near if pos <= size - 1 - pos else far
    return Grid.from_lists(rows)


def _mk_ghost_pair(color: int) -> TransformFn:
    def apply(grid: Grid) -> Grid:
        return _apply_ghost_pair(grid, color)
    return apply


def _paint_at_uv(
    rows: list[list[int]],
    shape: list[tuple[int, int]],
    gu: Fraction,
    gv: Fraction,
    color: int,
) -> None:
    """Translate ``shape`` so its uv-center lands at (gu, gv) and paint."""
    height, width = len(rows), len(rows[0])
    n = len(shape)
    su = sum(r - c for r, c in shape)
    sv = sum(r + c for r, c in shape)
    du = gu - Fraction(su, n)
    dv = gv - Fraction(sv, n)
    dr = (dv + du) / 2
    dc = (dv - du) / 2
    if dr.denominator != 1 or dc.denominator != 1:
        return
    for r, c in shape:
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < height and 0 <= nc < width:
            rows[nr][nc] = color


def _apply_ghost_pair(grid: Grid, color: int) -> Grid:
    """Touching congruent objects whose centers share a diagonal axis
    spawn ghost copies: same shape, placed at the pair's midpoint but
    offset perpendicular by 1.5x the pair's separation — the two new
    points that complete the diamond lattice. Single cells count as
    objects, so diagonal dominoes echo too."""
    objects = perceive(grid, compute_relations=False).objects
    rows = grid.to_lists()
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            if a.color != b.color or len(a.cells) != len(b.cells):
                continue
            ca = sorted(a.cells)
            cb = sorted(b.cells)
            if not any(
                abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1
                for r1, c1 in ca
                for r2, c2 in cb
            ):
                continue
            at, al = min(r for r, _ in ca), min(c for _, c in ca)
            bt, bl = min(r for r, _ in cb), min(c for _, c in cb)
            if sorted((r - at, c - al) for r, c in ca) != sorted(
                (r - bt, c - bl) for r, c in cb
            ):
                continue
            n = len(ca)
            sau = sum(r - c for r, c in ca)
            sav = sum(r + c for r, c in ca)
            sbu = sum(r - c for r, c in cb)
            sbv = sum(r + c for r, c in cb)
            if sau == sbu:
                off = Fraction(abs(sav - sbv), n) * Fraction(3, 2)
                mu = Fraction(sau, n)
                mv = Fraction(sav + sbv, 2 * n)
                for shape in (ca, cb):
                    _paint_at_uv(rows, shape, mu - off, mv, color)
                    _paint_at_uv(rows, shape, mu + off, mv, color)
            elif sav == sbv:
                off = Fraction(abs(sau - sbu), n) * Fraction(3, 2)
                mv = Fraction(sav, n)
                mu = Fraction(sau + sbu, 2 * n)
                for shape in (ca, cb):
                    _paint_at_uv(rows, shape, mu, mv - off, color)
                    _paint_at_uv(rows, shape, mu, mv + off, color)
    return Grid.from_lists(rows)


def _mk_axes_stamp(color: int, transpose: bool) -> TransformFn:
    def apply(grid: Grid) -> Grid:
        return _apply_axes_stamp(grid, color, transpose)
    return apply


def _apply_axes_stamp(
    grid: Grid, color: int, transpose: bool
) -> Grid:
    """All marks live on two axes: a template line (the row — or
    column, when transposed — carrying a pattern) and a selector line
    marking which parallel lines receive the pattern. Every selected
    line gets the template painted in the new color."""
    g = grid.reflect_diag() if transpose else grid
    marks = {(r, c) for r, c, v in g.iter_cells() if v != 0}
    rows = g.to_lists()
    for tr in range(g.height):
        for sc in range(g.width):
            if not all(r == tr or c == sc for r, c in marks):
                continue
            template = {c for r, c in marks if r == tr}
            targets = [r for r, c in marks if c == sc and r != tr]
            if not template or not targets:
                continue
            for r in targets:
                for c in template:
                    if c != sc:
                        rows[r][c] = color
            out = Grid.from_lists(rows)
            return out.reflect_diag() if transpose else out
    return grid


def _mk_ring_unique(color: int) -> TransformFn:
    def apply(grid: Grid) -> Grid:
        return _apply_ring_unique(grid, color)
    return apply


def _apply_ring_unique(grid: Grid, color: int) -> Grid:
    """The color appearing exactly once marks a center: erase
    everything else and draw a 3x3 ring around it, keeping the
    center's own color."""
    counts = grid.color_counts()
    uniq = [v for v, n in counts.items() if n == 1 and v != 0]
    if len(uniq) != 1:
        return grid
    (cr, cc), = list(grid.cells_with(uniq[0]))
    rows = [[0] * grid.width for _ in range(grid.height)]
    rows[cr][cc] = uniq[0]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                rows[nr][nc] = color
    return Grid.from_lists(rows)


def _mk_attract_to(anchor: int) -> TransformFn:
    def apply(grid: Grid) -> Grid:
        return _apply_attract_to(grid, anchor)
    return apply


def _apply_attract_to(grid: Grid, anchor: int) -> Grid:
    """The non-anchor object slides in a straight line toward the
    anchor-colored object until the two touch (8-adjacency)."""
    scene = perceive(grid, compute_relations=False)
    anchors = [o for o in scene.objects if o.color == anchor]
    movers = [o for o in scene.objects if o.color != anchor]
    if len(anchors) != 1 or len(movers) != 1:
        return grid
    acr = sum(r for r, _ in anchors[0].cells) / len(anchors[0].cells)
    acc = sum(c for _, c in anchors[0].cells) / len(anchors[0].cells)
    mcr = sum(r for r, _ in movers[0].cells) / len(movers[0].cells)
    mcc = sum(c for _, c in movers[0].cells) / len(movers[0].cells)
    dr, dc = acr - mcr, acc - mcc
    step_r = int(dr > 0) - int(dr < 0) if abs(dr) >= abs(dc) else 0
    step_c = (
        int(dc > 0) - int(dc < 0) if step_r == 0 else 0
    )
    if step_r == 0 and step_c == 0:
        return grid
    acells = anchors[0].cells
    cur: set[tuple[int, int]] = set(movers[0].cells)
    while True:
        moved = {(r + step_r, c + step_c) for r, c in cur}
        if not all(
            0 <= r < grid.height and 0 <= c < grid.width
            for r, c in moved
        ):
            break
        if moved & acells:
            break
        if any(
            abs(r - ar) <= 1 and abs(c - ac) <= 1
            for r, c in moved
            for ar, ac in acells
        ):
            cur = moved
            break
        cur = moved
    rows = grid.to_lists()
    for r, c in movers[0].cells:
        rows[r][c] = 0
    for r, c in cur:
        rows[r][c] = movers[0].color
    return Grid.from_lists(rows)


def _mk_stamp_at_marks() -> TransformFn:
    return _apply_stamp_at_marks


def _apply_stamp_at_marks(grid: Grid) -> Grid:
    """A divider splits the grid: the denser side holds a template
    block, and every mark on the other side gets the template
    pasted centered on it."""
    found = _divider_index(grid)
    if found is None:
        return grid
    axis, idx = found
    if axis == "row":
        return _apply_stamp_at_marks(grid.reflect_diag()).reflect_diag()

    left = [(r, c, v) for r, c, v in grid.iter_cells()
            if c < idx and v != 0]
    right = [(r, c, v) for r, c, v in grid.iter_cells()
             if c > idx and v != 0]
    if not left or not right:
        return grid
    block, marks = (left, right) if len(left) >= len(right) else (
        right, left
    )
    top = min(r for r, _, _ in block)
    left_c = min(c for _, c, _ in block)
    height = max(r for r, _, _ in block) - top + 1
    width = max(c for _, c, _ in block) - left_c + 1
    template = grid.subgrid(top, left_c, height, width)
    rows = grid.to_lists()
    for r, c, _ in marks:
        for dr in range(height):
            for dc in range(width):
                nr = r + dr - height // 2
                nc = c + dc - width // 2
                if 0 <= nr < grid.height and 0 <= nc < grid.width:
                    rows[nr][nc] = template.at(dr, dc)
    return Grid.from_lists(rows)


def _mk_radial_map(color: int) -> TransformFn:
    def _apply(grid: Grid) -> Grid:
        """Compress the scene around its marker cells: every object
        is drawn on a 3x3 map at the sign of its offset from the
        nearest marker, and the markers collapse to the center."""
        marks = [(r, c) for r, c, v in grid.iter_cells() if v == color]
        if not marks:
            return grid
        rows = [[0] * 3 for _ in range(3)]
        rows[1][1] = color
        scene = perceive(grid, diagonal=True, compute_relations=False)
        for obj in scene.objects:
            if obj.color == color:
                continue
            mr, mc = min(
                marks,
                key=lambda m: min(
                    max(abs(r - m[0]), abs(c - m[1])) for r, c in obj.cells
                ),
            )
            for r, c in obj.cells:
                nr = 1 + (r > mr) - (r < mr)
                nc = 1 + (c > mc) - (c < mc)
                rows[nr][nc] = obj.color
        return Grid.from_lists(rows)

    return _apply


def _mk_fill_busiest() -> TransformFn:
    return _apply_fill_busiest


def _apply_fill_busiest(grid: Grid) -> Grid:
    """Divider lines split the grid into blocks; the blocks holding
    the most marks are filled solid with their mark color and every
    other mark is erased."""
    div_rows = [
        r
        for r in range(grid.height)
        if grid.at(r, 0) != 0
        and len({grid.at(r, c) for c in range(grid.width)}) == 1
    ]
    div_cols = [
        c
        for c in range(grid.width)
        if grid.at(0, c) != 0
        and len({grid.at(r, c) for r in range(grid.height)}) == 1
    ]
    if not div_rows or not div_cols:
        return grid

    def bands(dividers: list[int], size: int) -> list[tuple[int, int]]:
        spans = []
        start = 0
        for d in [*dividers, size]:
            if d > start:
                spans.append((start, d))
            start = d + 1
        return spans

    row_bands = bands(div_rows, grid.height)
    col_bands = bands(div_cols, grid.width)
    counts: dict[tuple[int, int], int] = {}
    mark_color: dict[tuple[int, int], int] = {}
    for r, c, v in grid.iter_cells():
        if v == 0 or r in div_rows or c in div_cols:
            continue
        key = (
            next(i for i, (a, b) in enumerate(row_bands) if a <= r < b),
            next(i for i, (a, b) in enumerate(col_bands) if a <= c < b),
        )
        counts[key] = counts.get(key, 0) + 1
        mark_color.setdefault(key, v)
    if not counts:
        return grid
    top = max(counts.values())
    rows = grid.to_lists()
    for r, c, v in grid.iter_cells():
        if v != 0 and r not in div_rows and c not in div_cols:
            rows[r][c] = 0
    for (ri, ci), n in counts.items():
        if n != top:
            continue
        for r in range(*row_bands[ri]):
            for c in range(*col_bands[ci]):
                rows[r][c] = mark_color[(ri, ci)]
    return Grid.from_lists(rows)


def _mk_fill_lanes(color: int) -> TransformFn:
    def _apply(grid: Grid) -> Grid:
        """Paint the interior of every lane — a row or column whose
        only non-zero cells sit at its two ends."""
        rows = grid.to_lists()
        fill = [[False] * grid.width for _ in range(grid.height)]
        for r, row in enumerate(rows):
            nz = [c for c, v in enumerate(row) if v]
            if len(nz) >= 2 and not any(row[nz[0] + 1 : nz[-1]]):
                for c in range(nz[0] + 1, nz[-1]):
                    fill[r][c] = True
        for c in range(grid.width):
            nz = [r for r in range(grid.height) if rows[r][c]]
            if len(nz) >= 2 and all(
                rows[r][c] == 0 for r in range(nz[0] + 1, nz[-1])
            ):
                for r in range(nz[0] + 1, nz[-1]):
                    fill[r][c] = True
        for r in range(grid.height):
            for c in range(grid.width):
                if fill[r][c] and rows[r][c] == 0:
                    rows[r][c] = color
        return Grid.from_lists(rows)

    return _apply


def _mk_cross_fill() -> TransformFn:
    return _apply_cross_fill


def _apply_cross_fill(grid: Grid) -> Grid:
    """Divider lines split the grid into a 3x3 array of cells; the
    cross is painted N=2, W=4, center=6, E=3, S=1."""
    div_rows = [
        r
        for r in range(grid.height)
        if grid.at(r, 0) != 0
        and len({grid.at(r, c) for c in range(grid.width)}) == 1
    ]
    div_cols = [
        c
        for c in range(grid.width)
        if grid.at(0, c) != 0
        and len({grid.at(r, c) for r in range(grid.height)}) == 1
    ]

    def bands(dividers: list[int], size: int) -> list[tuple[int, int]]:
        spans = []
        start = 0
        for d in [*dividers, size]:
            if d > start:
                spans.append((start, d))
            start = d + 1
        return spans

    row_bands = bands(div_rows, grid.height)
    col_bands = bands(div_cols, grid.width)
    if len(row_bands) != 3 or len(col_bands) != 3:
        return grid
    paint = {(0, 1): 2, (1, 0): 4, (1, 1): 6, (1, 2): 3, (2, 1): 1}
    rows = grid.to_lists()
    for (ri, ci), color in paint.items():
        for r in range(*row_bands[ri]):
            for c in range(*col_bands[ci]):
                if rows[r][c] == 0:
                    rows[r][c] = color
    return Grid.from_lists(rows)


def _mk_eye_ray() -> TransformFn:
    return _apply_eye_ray


def _apply_eye_ray(grid: Grid) -> Grid:
    """A shape carries one differently-colored eye cell; the eye
    shoots a ray across the shape in the direction of its centroid,
    painting zeros all the way to the edge."""
    colors: dict[int, list[tuple[int, int]]] = {}
    for r, c, v in grid.iter_cells():
        if v:
            colors.setdefault(v, []).append((r, c))
    if len(colors) != 2:
        return grid
    eye_color = min(colors, key=lambda k: len(colors[k]))
    if len(colors[eye_color]) != 1:
        return grid
    shape_color = next(k for k in colors if k != eye_color)
    er, ec = colors[eye_color][0]
    cells = colors[shape_color]
    cr = sum(r for r, _ in cells) / len(cells)
    cc = sum(c for _, c in cells) / len(cells)
    dr, dc = cr - er, cc - ec
    step = (
        (1 if dr > 0 else -1, 0)
        if abs(dr) >= abs(dc)
        else (0, 1 if dc > 0 else -1)
    )
    if dr == 0 and dc == 0:
        return grid
    rows = grid.to_lists()
    r, c = er + step[0], ec + step[1]
    while 0 <= r < grid.height and 0 <= c < grid.width:
        if rows[r][c] == 0:
            rows[r][c] = eye_color
        r += step[0]
        c += step[1]
    return Grid.from_lists(rows)


def _mk_rot_symmetrize() -> TransformFn:
    return _apply_rot_symmetrize


def _apply_rot_symmetrize(grid: Grid) -> Grid:
    """Close the pattern under 90-degree rotations about the center
    of its non-zero bounding box. Coordinates are doubled so the
    center may fall between cells."""
    cells = grid.non_zero_cells()
    if not cells:
        return grid
    r2 = min(r for r, _ in cells) + max(r for r, _ in cells)
    c2 = min(c for _, c in cells) + max(c for _, c in cells)
    rows = grid.to_lists()
    for r, c in cells:
        v = grid.at(r, c)
        dr, dc = 2 * r - r2, 2 * c - c2
        for ddr, ddc in ((dr, dc), (dc, -dr), (-dr, -dc), (-dc, dr)):
            rr, cc = (ddr + r2) // 2, (ddc + c2) // 2
            if 0 <= rr < grid.height and 0 <= cc < grid.width:
                rows[rr][cc] = v
    return Grid.from_lists(rows)


def _mk_draw_lines(table: dict[int, str]) -> TransformFn:
    """Each seed cell emits a full row or column in its own color;
    column lines are laid first, row lines win crossings."""
    def _fn(grid: Grid) -> Grid:
        rows = grid.to_lists()
        for v, axis in table.items():
            if axis == "column":
                for _, c in grid.cells_with(v):
                    for rr in range(grid.height):
                        rows[rr][c] = v
        for v, axis in table.items():
            if axis == "row":
                for r, _ in grid.cells_with(v):
                    for cc in range(grid.width):
                        rows[r][cc] = v
        return Grid.from_lists(rows)
    return _fn


def _mk_stamp_recolor() -> TransformFn:
    return _apply_stamp_recolor


def _apply_stamp_recolor(grid: Grid) -> Grid:
    """Find the multicolor object — the stamp — and repaint every
    other object that has the same shape with the stamp's colors.

    Components are extracted over non-zero cells ignoring color so
    the multicolor stamp stays one object.
    """
    seen: set[tuple[int, int]] = set()
    comps: list[set[tuple[int, int]]] = []
    for r, c, v in grid.iter_cells():
        if v == 0 or (r, c) in seen:
            continue
        comp: set[tuple[int, int]] = set()
        queue = [(r, c)]
        seen.add((r, c))
        while queue:
            cr, cc = queue.pop()
            comp.add((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < grid.height
                    and 0 <= nc < grid.width
                    and (nr, nc) not in seen
                    and grid.at(nr, nc) != 0
                ):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        comps.append(comp)
    stamps = [
        comp for comp in comps
        if len({grid.at(r, c) for r, c in comp}) > 1
    ]
    if len(stamps) != 1:
        return grid
    stamp = stamps[0]
    sr = min(r for r, _ in stamp)
    sc = min(c for _, c in stamp)
    pattern = {(r - sr, c - sc): grid.at(r, c) for r, c in stamp}
    mask = set(pattern)
    rows = [[grid.at(r, c) for c in range(grid.width)]
            for r in range(grid.height)]
    for r, c in stamp:
        rows[r][c] = 0
    for comp in comps:
        if comp is stamp:
            continue
        orr = min(r for r, _ in comp)
        orc = min(c for _, c in comp)
        if {(r - orr, c - orc) for r, c in comp} != mask:
            continue
        for r, c in comp:
            rows[r][c] = pattern[(r - orr, c - orc)]
    return Grid.from_lists(rows)


def _mk_unwind_tile() -> TransformFn:
    return _apply_unwind_tile


def _apply_unwind_tile(grid: Grid) -> Grid:
    """Reconstruct the base tile of a quadrant-rotated pattern.

    The input's content is a 2k x 2k arrangement: the tile itself in
    the top-left, then its 90°/180°/270° rotations in TR/BR/BL.
    Each output cell takes the majority vote of the four rotated
    views, so a corrupted quadrant still yields the clean tile.
    """
    box = grid.crop_to_content()
    h, w = box.height, box.width
    if h != w or h % 2 != 0:
        return grid
    k = h // 2
    tl = box.subgrid(0, 0, k, k)
    tr = box.subgrid(0, k, k, k)
    br = box.subgrid(k, k, k, k)
    bl = box.subgrid(k, 0, k, k)
    rows = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            votes = [
                tl.at(i, j),
                tr.at(j, k - 1 - i),
                br.at(k - 1 - i, k - 1 - j),
                bl.at(k - 1 - j, i),
            ]
            counts: dict[int, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            best = max(counts.values())
            winners = [v for v, n in counts.items() if n == best]
            rows[i][j] = (
                winners[0] if len(winners) == 1
                else votes[0] if votes[0] in winners
                else winners[0]
            )
    return Grid.from_lists(rows)


def _mk_align_tops(color: int) -> TransformFn:
    return lambda g: _apply_align_tops(g, color)


def _apply_align_tops(grid: Grid, color: int) -> Grid:
    """Slide every object vertically until its top row matches the
    top row of the reference-colored object. Columns and shapes are
    preserved; the reference object itself does not move."""
    scene = perceive(grid, diagonal=True, compute_relations=False)
    refs = [o for o in scene.objects if o.color == color]
    if not refs:
        return grid
    ref_top = min(r for o in refs for r, _ in o.cells)
    rows = [[0] * grid.width for _ in range(grid.height)]
    for obj in scene.objects:
        top = min(r for r, _ in obj.cells)
        dr = ref_top - top
        for r, c in obj.cells:
            nr = r + dr
            if 0 <= nr < grid.height:
                rows[nr][c] = obj.color
    return Grid.from_lists(rows)


def _mk_dual_frame() -> TransformFn:
    return _apply_dual_frame


def _apply_dual_frame(grid: Grid) -> Grid:
    """Two marker dots split the grid into two horizontal bands, each
    drawn as a frame: a full-width bar at the band's outer edge and at
    the dot's row, with side columns in the band's color."""
    dots = [
        (r, c, v)
        for r, c, v in grid.iter_cells()
        if v != 0
    ]
    if len(dots) != 2:
        return grid
    (r1, _, top_color), (r2, _, bot_color) = sorted(dots)
    mid = (r1 + r2) // 2
    bars = {0: top_color, r1: top_color, r2: bot_color,
            grid.height - 1: bot_color}
    rows = [[0] * grid.width for _ in range(grid.height)]
    for r in range(grid.height):
        side = top_color if r <= mid else bot_color
        for c in range(grid.width):
            if r in bars:
                rows[r][c] = bars[r]
            elif c in (0, grid.width - 1):
                rows[r][c] = side
    return Grid.from_lists(rows)


def _mk_shear(dr: int, dc: int) -> TransformFn:
    return lambda g: _apply_shear(g, dr, dc)


def _apply_shear(grid: Grid, dr: int, dc: int) -> Grid:
    """Shift each object by ``(dr, dc)``, but pin the two bounding-box
    edges meeting at the corner the motion points toward: cells on
    those edges stay put, so the shape shears into the corner instead
    of translating cleanly."""
    scene = perceive(grid, diagonal=True, compute_relations=False)
    rows = [[0] * grid.width for _ in range(grid.height)]
    for obj in scene.objects:
        rs = [r for r, _ in obj.cells]
        cs = [c for _, c in obj.cells]
        r_pin = max(rs) if dr >= 0 else min(rs)
        c_pin = max(cs) if dc >= 0 else min(cs)
        for r, c in obj.cells:
            if r == r_pin or c == c_pin:
                nr, nc = r, c
            else:
                nr, nc = r + dr, c + dc
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                rows[nr][nc] = obj.color
    return Grid.from_lists(rows)


def _mk_self_tile() -> TransformFn:
    return _apply_self_tile


def _mk_extend_down(n: int) -> TransformFn:
    return lambda g: _apply_extend_down(g, n)


def _mk_repeat_down() -> TransformFn:
    return lambda g: _apply_repeat_down(g)


def _apply_repeat_down(grid: Grid) -> Grid:
    """Append a full copy of the grid below itself."""
    rows = [[grid.at(r, c) for c in range(grid.width)]
            for r in range(grid.height)]
    rows += [row[:] for row in rows[: grid.height]]
    return Grid.from_lists(rows)


def _apply_extend_down(grid: Grid, n: int) -> Grid:
    """Grow the grid downward by continuing its own vertical period —
    appended rows follow row[(h + i) % period], where the period is
    detected on this grid (different inputs may have different
    periods)."""
    p = _row_period(grid)
    if p is None:
        return grid
    rows = [[grid.at(r, c) for c in range(grid.width)]
            for r in range(grid.height)]
    for i in range(n):
        rows.append(rows[(grid.height + i) % p][:])
    return Grid.from_lists(rows)


def _row_period(grid: Grid) -> int | None:
    """Smallest vertical period p where row[i] == row[i-p] for all i≥p."""
    for p in range(1, grid.height):
        if all(
            [grid.at(r, c) for c in range(grid.width)]
            == [grid.at(r - p, c) for c in range(grid.width)]
            for r in range(p, grid.height)
        ):
            return p
    return None


def _mk_mark_diagonals(color: int) -> TransformFn:
    return lambda g: _apply_mark_diagonals(g, color)


def _mk_intersect_halves(color: int) -> TransformFn:
    return lambda g: _apply_intersect_halves(g, color)


def _mk_settle(direction: str) -> TransformFn:
    return lambda g: _apply_settle(g, direction)


def _mk_bridge_endpoints(marker: int) -> TransformFn:
    return lambda g: _apply_bridge_endpoints(g, marker)


def _mk_rank_bars() -> TransformFn:
    return _apply_rank_bars


def _mk_cross_rays(clash: int) -> TransformFn:
    return lambda g: _apply_cross_rays(g, clash)


def _apply_cross_rays(grid: Grid, clash: int) -> Grid:
    """Each non-zero cell paints its entire row and column in its own
    color; where two different colors cross, the clash color wins."""
    src: dict[tuple[int, int], int] = {}
    for r in range(grid.height):
        for c in range(grid.width):
            v = grid.at(r, c)
            if v != 0:
                src[(r, c)] = v
    painted: dict[tuple[int, int], int] = {}
    clashed: set[tuple[int, int]] = set()
    for (r, c), v in src.items():
        for cc in range(grid.width):
            _paint(painted, clashed, r, cc, v)
        for rr in range(grid.height):
            _paint(painted, clashed, rr, c, v)
    rows = [[0] * grid.width for _ in range(grid.height)]
    for (r, c), v in painted.items():
        rows[r][c] = clash if (r, c) in clashed else v
    return Grid.from_lists(rows)


def _paint(
    painted: dict[tuple[int, int], int],
    clashed: set[tuple[int, int]],
    r: int,
    c: int,
    v: int,
) -> None:
    prior = painted.get((r, c))
    if prior is None or prior == v:
        painted[(r, c)] = v
    else:
        clashed.add((r, c))


def _apply_rank_bars(grid: Grid) -> Grid:
    """Recolor each column's contiguous non-zero run by its height
    rank: the tallest bar becomes 1, next 2, and so on."""
    runs: list[tuple[int, list[tuple[int, int]]]] = []
    for c in range(grid.width):
        cells = [
            (r, c)
            for r in range(grid.height)
            if grid.at(r, c) != 0
        ]
        if not cells:
            continue
        # only contiguous runs count as bars
        rows_i = [r for r, _ in cells]
        if rows_i != list(range(rows_i[0], rows_i[-1] + 1)):
            continue
        runs.append((c, cells))
    order = sorted(runs, key=lambda x: -len(x[1]))
    rows = [[0] * grid.width for _ in range(grid.height)]
    for rank, (_, cells) in enumerate(order, start=1):
        for r, c in cells:
            rows[r][c] = rank
    return Grid.from_lists(rows)


def _apply_bridge_endpoints(grid: Grid, marker: int) -> Grid:
    """On each row with exactly two marks, draw a bridge between
    them: the left color fills to the midpoint, the right color
    fills back, and the midpoint cell gets the marker color."""
    rows = [[grid.at(r, c) for c in range(grid.width)]
            for r in range(grid.height)]
    for r in range(grid.height):
        marks = [
            (c, grid.at(r, c))
            for c in range(grid.width)
            if grid.at(r, c) != 0
        ]
        if len(marks) != 2:
            continue
        (a, ca), (b, cb) = marks
        mid = (a + b) // 2
        for c in range(a, mid):
            rows[r][c] = ca
        rows[r][mid] = marker
        for c in range(mid + 1, b + 1):
            rows[r][c] = cb
    return Grid.from_lists(rows)


def _apply_settle(grid: Grid, direction: str) -> Grid:
    """Pack each column's (or row's) non-zero cells against an edge,
    preserving their order — gravity on cells, not objects."""
    rows = [[0] * grid.width for _ in range(grid.height)]
    if direction in ("down", "up"):
        for c in range(grid.width):
            vals = [grid.at(r, c) for r in range(grid.height)]
            vals = [v for v in vals if v != 0]
            start = grid.height - len(vals) if direction == "down" else 0
            for i, v in enumerate(vals):
                rows[start + i][c] = v
    else:
        for r in range(grid.height):
            vals = [grid.at(r, c) for c in range(grid.width)]
            vals = [v for v in vals if v != 0]
            start = grid.width - len(vals) if direction == "right" else 0
            for i, v in enumerate(vals):
                rows[r][start + i] = v
    return Grid.from_lists(rows)


def _mk_classify_shape(
    table: dict[tuple[tuple[int, ...], ...], int],
) -> TransformFn:
    return lambda g: _apply_classify_shape(g, table)


def _mask_key(grid: Grid) -> tuple[tuple[int, ...], ...]:
    """Color-invariant shape: which cells are filled, as 0/1 rows."""
    return tuple(
        tuple(1 if grid.at(r, c) != 0 else 0 for c in range(grid.width))
        for r in range(grid.height)
    )


def _apply_classify_shape(
    grid: Grid, table: dict[tuple[tuple[int, ...], ...], int],
) -> Grid:
    """Map a pattern's shape (ignoring color) to a 1x1 label."""
    return Grid.from_lists([[table.get(_mask_key(grid), 0)]])


def _divider_index(grid: Grid) -> tuple[str, int] | None:
    """Find a full-length monochrome nonzero row or column.

    Prefers a line whose color appears nowhere else in the grid — a
    real divider is drawn in its own ink, while e.g. an all-fill
    column can be monochrome by accident.
    """
    candidates: list[tuple[str, int, int]] = []
    for c in range(grid.width):
        col = {grid.at(r, c) for r in range(grid.height)}
        if len(col) == 1 and 0 not in col:
            candidates.append(("column", c, col.pop()))
    for r in range(grid.height):
        row = {grid.at(r, c) for c in range(grid.width)}
        if len(row) == 1 and 0 not in row:
            candidates.append(("row", r, row.pop()))
    if not candidates:
        return None
    for axis, idx, color in candidates:
        elsewhere = False
        for r in range(grid.height):
            for c in range(grid.width):
                if axis == "column" and c == idx:
                    continue
                if axis == "row" and r == idx:
                    continue
                if grid.at(r, c) == color:
                    elsewhere = True
        if not elsewhere:
            return axis, idx
    axis, idx, _ = candidates[0]
    return axis, idx


def _apply_intersect_halves(grid: Grid, color: int) -> Grid:
    """Split at a monochrome divider line; output marks cells that
    are filled in BOTH halves."""
    found = _divider_index(grid)
    if found is None:
        return grid
    axis, idx = found
    if axis == "column":
        left = grid.subgrid(0, 0, grid.height, idx)
        right = grid.subgrid(0, idx + 1, grid.height, grid.width - idx - 1)
    else:
        left = grid.subgrid(0, 0, idx, grid.width)
        right = grid.subgrid(idx + 1, 0, grid.height - idx - 1, grid.width)
    if left.shape != right.shape:
        return grid
    rows = [
        [
            color
            if left.at(r, c) != 0 and right.at(r, c) != 0
            else 0
            for c in range(left.width)
        ]
        for r in range(left.height)
    ]
    return Grid.from_lists(rows)


def _mk_empty_halves(color: int) -> TransformFn:
    return lambda g: _apply_empty_halves(g, color)


def _mk_pick_quadrant() -> TransformFn:
    return _apply_pick_quadrant


def _cross_lines(grid: Grid) -> tuple[int, int] | None:
    """Find a divider cross: one full monochrome row and one full
    monochrome column (any colors, but non-zero lines win)."""
    rows_all = [
        r for r in range(grid.height)
        if len({grid.at(r, c) for c in range(grid.width)}) == 1
    ]
    cols_all = [
        c for c in range(grid.width)
        if len({grid.at(r, c) for r in range(grid.height)}) == 1
    ]
    rows_i = [r for r in rows_all if grid.at(r, 0) != 0] or rows_all
    cols_i = [c for c in cols_all if grid.at(0, c) != 0] or cols_all
    if not rows_i or not cols_i:
        return None
    return rows_i[0], cols_i[0]


def _apply_pick_quadrant(grid: Grid) -> Grid:
    """Split at the divider cross and return the quadrant containing
    the rarest non-divider color — the marked quadrant."""
    found = _cross_lines(grid)
    if found is None:
        return grid
    ri, ci = found
    div_colors = {
        grid.at(ri, c) for c in range(grid.width)
    } | {
        grid.at(r, ci) for r in range(grid.height)
    }
    counts: dict[int, int] = {}
    for r in range(grid.height):
        for c in range(grid.width):
            v = grid.at(r, c)
            if v not in div_colors:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return grid
    marker = min(counts, key=lambda k: counts[k])
    quads = {
        "tl": (0, 0, ri, ci),
        "tr": (0, ci + 1, ri, grid.width - ci - 1),
        "bl": (ri + 1, 0, grid.height - ri - 1, ci),
        "br": (ri + 1, ci + 1, grid.height - ri - 1, grid.width - ci - 1),
    }
    for box in quads.values():
        r0, c0, h, w = box
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if grid.at(r, c) == marker:
                    return grid.subgrid(r0, c0, h, w)
    return grid


def _apply_empty_halves(grid: Grid, color: int) -> Grid:
    """Split at a monochrome divider line; output marks cells that
    are empty in BOTH halves (the NOR twin of intersect_halves)."""
    found = _divider_index(grid)
    if found is None:
        return grid
    axis, idx = found
    if axis == "column":
        left = grid.subgrid(0, 0, grid.height, idx)
        right = grid.subgrid(0, idx + 1, grid.height, grid.width - idx - 1)
    else:
        left = grid.subgrid(0, 0, idx, grid.width)
        right = grid.subgrid(idx + 1, 0, grid.height - idx - 1, grid.width)
    if left.shape != right.shape:
        return grid
    rows = [
        [
            color
            if left.at(r, c) == 0 and right.at(r, c) == 0
            else 0
            for c in range(left.width)
        ]
        for r in range(left.height)
    ]
    return Grid.from_lists(rows)


def _mk_diff_halves(color: int) -> TransformFn:
    return lambda g: _apply_diff_halves(g, color)


def _apply_diff_halves(grid: Grid, color: int) -> Grid:
    """Split at a monochrome divider line; output marks cells where
    the two halves DIFFER (the XOR twin of intersect_halves)."""
    found = _divider_index(grid)
    if found is None:
        return grid
    axis, idx = found
    if axis == "column":
        left = grid.subgrid(0, 0, grid.height, idx)
        right = grid.subgrid(0, idx + 1, grid.height, grid.width - idx - 1)
    else:
        left = grid.subgrid(0, 0, idx, grid.width)
        right = grid.subgrid(idx + 1, 0, grid.height - idx - 1, grid.width)
    if left.shape != right.shape:
        return grid
    rows = [
        [
            color if left.at(r, c) != right.at(r, c) else 0
            for c in range(left.width)
        ]
        for r in range(left.height)
    ]
    return Grid.from_lists(rows)


def _apply_mark_diagonals(grid: Grid, color: int) -> Grid:
    """Fill empty cells that touch a non-zero cell at a diagonal
    corner — the four diagonal neighbors only, not orthogonal."""
    rows = [[grid.at(r, c) for c in range(grid.width)]
            for r in range(grid.height)]
    for r, c, v in grid.iter_cells():
        if v != 0:
            continue
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid.height
                and 0 <= nc < grid.width
                and grid.at(nr, nc) != 0
            ):
                rows[r][c] = color
                break
    return Grid.from_lists(rows)


def _apply_self_tile(grid: Grid) -> Grid:
    """Use the grid as its own stencil: every non-zero cell becomes a
    block containing a copy of the whole grid; zero cells become
    blank blocks. Output dims are (h², w²)."""
    h, w = grid.height, grid.width
    rows = [[0] * (w * w) for _ in range(h * h)]
    for r, c, v in grid.iter_cells():
        if v == 0:
            continue
        for r2, c2, v2 in grid.iter_cells():
            rows[r * h + r2][c * w + c2] = v2
    return Grid.from_lists(rows)


def _apply_mark_uniform(grid: Grid, axis: str, color: int) -> Grid:
    """Mark every line (row or column) whose cells are all one color.

    Uniform lines become ``color``; everything else becomes 0. The
    line structure is preserved so the output reads as a map of
    which lines were uniform.
    """
    rows = [[0] * grid.width for _ in range(grid.height)]
    if axis == "row":
        for r in range(grid.height):
            line = [grid.at(r, c) for c in range(grid.width)]
            if len(set(line)) == 1:
                for c in range(grid.width):
                    rows[r][c] = color
    else:
        for c in range(grid.width):
            line = [grid.at(r, c) for r in range(grid.height)]
            if len(set(line)) == 1:
                for r in range(grid.height):
                    rows[r][c] = color
    return Grid.from_lists(rows)


def instantiate(family: str, params: dict[str, object]) -> Transform | None:
    """Rebuild a transform from a family name + parameter values.

    Used by schema induction: a learned rule like
    ``fill_enclosed(color=4)`` abstracts to "fill enclosed regions
    with *some* color", and this binds the free parameters to new
    values observed in a later task. Returns None for families that
    can't be re-parameterized (e.g. ``recolor``, whose map is derived
    from evidence rather than chosen).
    """
    def color_param() -> int | None:
        v = params.get("color")
        return v if isinstance(v, int) else None

    if family == "gravity":
        direction = params.get("direction")
        if direction not in ("down", "up", "left", "right"):
            return None
        if "color" in params:
            color = color_param()
            if color is None:
                return None
            return Transform(
                f"gravity_{direction}_{color}",
                _mk_gravity(str(direction), color),
                {"direction": direction, "color": color},
            )
        return Transform(
            f"gravity_{direction}",
            _mk_gravity(str(direction)),
            {"direction": direction},
        )
    if family == "erase_color":
        color = color_param()
        if color is None:
            return None
        return Transform(
            "erase_color",
            _mk_erase(color),
            {"color": color},
        )
    if family == "select_color":
        color = color_param()
        if color is None:
            return None
        return Transform(
            "select_color",
            _mk_select(color),
            {"color": color},
        )
    if family in ("recolor_largest", "recolor_smallest"):
        color = color_param()
        if color is None:
            return None
        which = family.removeprefix("recolor_")
        return Transform(
            family,
            _mk_recolor_object(which, color),
            {"color": color},
        )
    if family == "fill_enclosed":
        color = color_param()
        if color is None:
            return None
        return Transform(
            "fill_enclosed",
            _mk_fill_enclosed(color),
            {"color": color},
        )
    if family == "mark_uniform":
        axis = params.get("axis")
        color = color_param()
        if axis not in ("row", "column") or color is None:
            return None
        return Transform(
            f"mark_uniform_{axis}",
            _mk_mark_uniform(str(axis), color),
            {"axis": axis, "color": color},
        )
    if family == "shift":
        dr, dc = params.get("dr"), params.get("dc")
        clamp = params.get("clamp", False)
        if (
            not isinstance(dr, int) or not isinstance(dc, int)
            or not isinstance(clamp, bool)
        ):
            return None
        return Transform(
            "shift",
            _mk_shift(dr, dc, clamp),
            {"dr": dr, "dc": dc, "clamp": clamp},
        )
    if family == "self_tile":
        return Transform("self_tile", _mk_self_tile(), {})
    if family == "mark_diagonals":
        color = color_param()
        if color is None:
            return None
        return Transform(
            "mark_diagonals",
            _mk_mark_diagonals(color),
            {"color": color},
        )
    if family == "extend_down":
        n = params.get("n")
        if not isinstance(n, int) or n < 1:
            return None
        return Transform("extend_down", _mk_extend_down(n), {"n": n})
    if family == "repeat_down":
        return Transform("repeat_down", _mk_repeat_down(), {})
    if family == "intersect_halves":
        color = color_param()
        if color is None:
            return None
        return Transform(
            "intersect_halves",
            _mk_intersect_halves(color),
            {"color": color},
        )
    if family == "classify_shape":
        table = params.get("table")
        if not isinstance(table, dict):
            return None
        return Transform(
            "classify_shape",
            _mk_classify_shape(table),
            {"table": table},
        )
    if family == "settle":
        direction = params.get("direction")
        if direction not in ("down", "up", "left", "right"):
            return None
        return Transform(
            f"settle_{direction}",
            _mk_settle(str(direction)),
            {"direction": direction},
        )
    if family == "bridge_endpoints":
        marker = params.get("marker")
        if not isinstance(marker, int):
            return None
        return Transform(
            "bridge_endpoints",
            _mk_bridge_endpoints(marker),
            {"marker": marker},
        )
    if family == "rank_bars":
        return Transform("rank_bars", _mk_rank_bars(), {})
    if family == "cross_rays":
        clash = params.get("clash")
        if not isinstance(clash, int):
            return None
        return Transform(
            "cross_rays", _mk_cross_rays(clash), {"clash": clash},
        )
    if family == "pick_quadrant":
        return Transform("pick_quadrant", _mk_pick_quadrant(), {})
    if family == "dual_frame":
        return Transform("dual_frame", _mk_dual_frame(), {})
    if family == "unwind_tile":
        return Transform("unwind_tile", _mk_unwind_tile(), {})
    if family == "stamp_recolor":
        return Transform("stamp_recolor", _mk_stamp_recolor(), {})
    if family == "link_blocks":
        return Transform("link_blocks", _mk_link_blocks(), {})
    if family == "cross_halos":
        raw = params.get("table")
        if not isinstance(raw, dict):
            return None
        halo_map: _HaloTable = {}
        for key, halo in raw.items():
            halo_map[int(key)] = tuple(
                (int(dr), int(dc), int(k)) for dr, dc, k in halo
            )
        return Transform("cross_halos", _mk_cross_halos(halo_map),
                         {"table": raw})
    if family == "count_blocks":
        ccolor = params.get("color")
        cwidth = params.get("width")
        if not isinstance(ccolor, int) or not isinstance(cwidth, int):
            return None
        return Transform("count_blocks", _mk_count_blocks(ccolor, cwidth),
                         {"color": ccolor, "width": cwidth})
    if family == "draw_lines":
        lraw = params.get("table")
        if not isinstance(lraw, dict):
            return None
        return Transform(
            "draw_lines",
            _mk_draw_lines({int(k): str(a) for k, a in lraw.items()}),
            {"table": lraw},
        )
    if family == "rot_symmetrize":
        return Transform("rot_symmetrize", _mk_rot_symmetrize(), {})
    if family == "ray_recolor":
        return Transform("ray_recolor", _mk_ray_recolor(), {})
    if family == "nearest_border":
        return Transform("nearest_border", _mk_nearest_border(), {})
    if family == "ghost_pair":
        gcolor = params.get("color")
        if not isinstance(gcolor, int):
            return None
        return Transform(
            "ghost_pair", _mk_ghost_pair(gcolor), {"color": gcolor}
        )
    if family == "axes_stamp":
        scolor = params.get("color")
        strans = params.get("transpose")
        if not isinstance(scolor, int) or not isinstance(strans, bool):
            return None
        return Transform(
            "axes_stamp",
            _mk_axes_stamp(scolor, strans),
            {"color": scolor, "transpose": strans},
        )
    if family == "ring_unique":
        rcolor = params.get("color")
        if not isinstance(rcolor, int):
            return None
        return Transform(
            "ring_unique", _mk_ring_unique(rcolor), {"color": rcolor}
        )
    if family == "attract_to":
        acolor = params.get("anchor")
        if not isinstance(acolor, int):
            return None
        return Transform(
            "attract_to", _mk_attract_to(acolor), {"anchor": acolor}
        )
    if family == "stamp_at_marks":
        return Transform("stamp_at_marks", _mk_stamp_at_marks(), {})
    if family == "radial_map":
        mcolor = params.get("color")
        if not isinstance(mcolor, int):
            return None
        return Transform(
            "radial_map", _mk_radial_map(mcolor), dict(params)
        )
    if family == "fill_busiest":
        return Transform("fill_busiest", _mk_fill_busiest(), {})
    if family == "fill_lanes":
        lcolor = params.get("color")
        if not isinstance(lcolor, int):
            return None
        return Transform(
            "fill_lanes", _mk_fill_lanes(lcolor), dict(params)
        )
    if family == "eye_ray":
        return Transform("eye_ray", _mk_eye_ray(), {})
    if family == "cross_fill":
        return Transform("cross_fill", _mk_cross_fill(), {})
    if family == "align_tops":
        acolor = params.get("color")
        if not isinstance(acolor, int):
            return None
        return Transform(
            "align_tops", _mk_align_tops(acolor), {"color": acolor},
        )
    if family == "shear":
        dr, dc = params.get("dr"), params.get("dc")
        if not isinstance(dr, int) or not isinstance(dc, int):
            return None
        return Transform(
            "shear", _mk_shear(dr, dc), {"dr": dr, "dc": dc},
        )
    if family == "empty_halves":
        ecolor = params.get("color")
        if not isinstance(ecolor, int):
            return None
        return Transform(
            "empty_halves", _mk_empty_halves(ecolor), {"color": ecolor},
        )
    if family == "diff_halves":
        dcolor = params.get("color")
        if not isinstance(dcolor, int):
            return None
        return Transform(
            "diff_halves", _mk_diff_halves(dcolor), {"color": dcolor},
        )
    return None


def _propose_gravity(examples: list[Example]) -> list[Transform]:
    """Propose gravity: objects fall to an edge within the frame."""
    return [
        Transform(
            f"gravity_{d}",
            _mk_gravity(d),
            {"direction": d},
        )
        for d in ("down", "up", "left", "right")
    ]


def _propose_erase_color(examples: list[Example]) -> list[Transform]:
    """Propose erasing a color when outputs lose a color the inputs have."""
    out_colors = set()
    for _, out in examples:
        out_colors |= set(out.colors())
    candidates: set[int] = set()
    for inp, _ in examples:
        candidates |= set(inp.colors()) - out_colors
    candidates.discard(0)
    return [
        Transform(
            "erase_color",
            _mk_erase(c),
            {"color": c},
        )
        for c in sorted(candidates)
    ]


def _propose_keep_object(examples: list[Example]) -> list[Transform]:
    """Propose keeping only the largest/smallest object, cropped to it."""

    def keep(grid: Grid, which: str) -> Grid:
        scene = perceive(grid, compute_relations=False)
        if not scene.objects:
            return grid
        key = max if which == "largest" else min
        obj = key(scene.objects, key=lambda o: o.size)
        return grid.crop_to_cells(obj.cells)

    return [
        Transform("keep_largest", lambda g: keep(g, "largest")),
        Transform("keep_smallest", lambda g: keep(g, "smallest")),
    ]


def _propose_object_recolor(examples: list[Example]) -> list[Transform]:
    """Propose recoloring a selected object (largest/smallest) to a color.

    Catches the common pattern where the output keeps the input's
    geometry but one *object* changes color — which a global color map
    can't express when that color appears elsewhere.
    """
    new_colors: set[int] = set()
    for inp, out in examples:
        if inp.shape != out.shape:
            continue
        for r, c, v in out.iter_cells():
            if v != inp.at(r, c):
                new_colors.add(v)
    new_colors.discard(0)
    if not new_colors:
        return []

    return [
        Transform(
            f"recolor_{which}",
            _mk_recolor_object(which, color),
            {"color": color},
        )
        for which in ("largest", "smallest")
        for color in sorted(new_colors)
    ]


def _propose_fill_enclosed(examples: list[Example]) -> list[Transform]:
    """Propose filling enclosed background regions with a color.

    A background cell is enclosed if it can't reach the grid border
    through other background cells — the classic "color the inside of
    the box" operation.
    """
    new_colors: set[int] = set()
    for inp, out in examples:
        if inp.shape != out.shape:
            continue
        for r, c, v in out.iter_cells():
            if v != inp.at(r, c):
                new_colors.add(v)
    new_colors.discard(0)
    if not new_colors:
        return []

    return [
        Transform(
            "fill_enclosed",
            _mk_fill_enclosed(c),
            {"color": c},
        )
        for c in sorted(new_colors)
    ]


def _propose_attract_marks(examples: list[Example]) -> list[Transform]:
    """Propose sliding isolated marks onto same-colored lines.

    When a grid has colored lines (solid runs of length >= 2 along a
    row or column) and isolated single-cell marks, the rule is often
    "each mark belongs to the matching line": slide it along its row
    (or column) to the cell adjacent to the nearest same-colored
    line. Marks with no matching line are erased.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []

    def lines_of(g: Grid) -> list[tuple[str, int, int, int, int]]:
        """(axis, fixed, lo, hi, color) runs of length >= 2."""
        lines = []
        for c in range(g.width):
            r = 0
            while r < g.height:
                v = g.at(r, c)
                if v == 0:
                    r += 1
                    continue
                r1 = r
                while r1 + 1 < g.height and g.at(r1 + 1, c) == v:
                    r1 += 1
                if r1 > r:
                    lines.append(("v", c, r, r1, v))
                r = r1 + 1
        for r in range(g.height):
            c = 0
            while c < g.width:
                v = g.at(r, c)
                if v == 0:
                    c += 1
                    continue
                c1 = c
                while c1 + 1 < g.width and g.at(r, c1 + 1) == v:
                    c1 += 1
                if c1 > c:
                    lines.append(("h", r, c, c1, v))
                c = c1 + 1
        return lines

    def apply(g: Grid) -> Grid:
        lines = lines_of(g)
        on_line = {
            (r, c_)
            for axis, fixed, lo, hi, _ in lines
            for r, c_ in (
                [(rr, fixed) for rr in range(lo, hi + 1)]
                if axis == "v"
                else [(fixed, cc) for cc in range(lo, hi + 1)]
            )
        }
        out_rows = g.to_lists()
        for r, c, v in g.iter_cells():
            if v == 0 or (r, c) in on_line:
                continue
            # Nearest same-colored line, measured by the slide path.
            best: tuple[int, int, int] | None = None
            for axis, fixed, lo, hi, lv in lines:
                if lv != v:
                    continue
                if axis == "v":
                    if lo <= r <= hi:
                        tr, tc = r, fixed + (-1 if c < fixed else 1)
                    else:
                        continue  # can't slide sideways past its ends
                else:
                    if lo <= c <= hi:
                        tr, tc = fixed + (-1 if r < fixed else 1), c
                    else:
                        continue
                if not (0 <= tr < g.height and 0 <= tc < g.width):
                    continue
                dist = abs(tr - r) + abs(tc - c)
                if best is None or dist < best[0]:
                    best = (dist, tr, tc)
            out_rows[r][c] = 0
            if best is not None and out_rows[best[1]][best[2]] == 0:
                out_rows[best[1]][best[2]] = v
        return Grid.from_lists(out_rows)

    return [Transform("attract_marks", apply, {})]


def _propose_denoise(examples: list[Example]) -> list[Transform]:
    """Propose absorbing minority colors into a dominant structure.

    When a grid holds a dominant non-background color (the structure)
    and scattered cells of other colors (noise), the rule is often
    "repair the structure": a noise cell *bracketed* by dominant cells
    — same row with dominant on both sides, or same column with
    dominant above and below — lies inside the silhouette and takes
    its color; unbracketed noise is erased.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []

    def apply(g: Grid) -> Grid:
        bg = g.most_common_color()
        counts: Counter[int] = Counter()
        for _, _, v in g.iter_cells():
            if v != bg:
                counts[v] += 1
        if len(counts) < 2:
            return g
        dominant = counts.most_common(1)[0][0]
        out_rows = g.to_lists()
        # Iterate to a fixpoint: each cell restored into the
        # silhouette can itself bracket the next — the boundary
        # grows outward until no noise cell is enclosed.
        for _ in range(max(g.height, g.width)):
            # Per-row and per-column extents of the dominant color
            # (recomputed each pass as restored cells join it).
            row_min: dict[int, int] = {}
            row_max: dict[int, int] = {}
            col_min: dict[int, int] = {}
            col_max: dict[int, int] = {}
            for r in range(g.height):
                for c in range(g.width):
                    if out_rows[r][c] != dominant:
                        continue
                    row_min[r] = min(row_min.get(r, c), c)
                    row_max[r] = max(row_max.get(r, c), c)
                    col_min[c] = min(col_min.get(c, r), r)
                    col_max[c] = max(col_max.get(c, r), r)
            changed = False
            for r in range(g.height):
                for c in range(g.width):
                    v = out_rows[r][c]
                    if v == bg or v == dominant:
                        continue
                    inside = (
                        row_min.get(r, c) < c < row_max.get(r, c)
                        or col_min.get(c, r) < r < col_max.get(c, r)
                    )
                    if inside:
                        out_rows[r][c] = dominant
                        changed = True
            if not changed:
                break
        # Only after the silhouette stops growing: whatever noise
        # remains outside it is erased.
        for r in range(g.height):
            for c in range(g.width):
                v = out_rows[r][c]
                if v != bg and v != dominant:
                    out_rows[r][c] = bg
        return Grid.from_lists(out_rows)

    return [Transform("denoise", apply, {})]


def _propose_periodic_completion(examples: list[Example]) -> list[Transform]:
    """Propose completing a periodic pattern.

    When the output equals the input plus filled background cells and
    is periodic with a small (row, col) period, the rule is "continue
    the pattern": infer the smallest consistent period from the input
    and fill background cells from it. Zeros are treated as unknown.
    """

    def complete(
        g: Grid, unknowns: frozenset[int]
    ) -> Grid | None:
        """Smallest-period completion of g, or None if impossible.

        Cells whose color is in ``unknowns`` are treated as missing
        and filled from the inferred period."""
        h, w = g.height, g.width
        for area in range(1, (h // 2 + 1) * (w // 2 + 1)):
            for ph in range(1, min(area, h // 2) + 1):
                if area % ph:
                    continue
                pw = area // ph
                if pw > w // 2:
                    continue
                # pattern[(r mod ph, c mod pw)] = the agreed value, or
                # conflict (rejected). Unknown cells are missing data.
                pattern: dict[tuple[int, int], int] = {}
                ok = True
                for r, c, v in g.iter_cells():
                    if v in unknowns:
                        continue
                    key = (r % ph, c % pw)
                    if key in pattern and pattern[key] != v:
                        ok = False
                        break
                    pattern[key] = v
                if not ok or len(pattern) < ph * pw:
                    continue
                rows = [
                    [pattern[(r % ph, c % pw)] for c in range(w)]
                    for r in range(h)
                ]
                return Grid.from_lists(rows)
        return None

    # The scribble-inpainting mask: the rarest non-background color
    # in a given grid, treated as unknown alongside the background.
    def rare_mask(g: Grid) -> frozenset[int]:
        counts = Counter(v for _, _, v in g.iter_cells() if v != 0)
        if len(counts) < 2:
            return frozenset({0})
        return frozenset({0, counts.most_common()[-1][0]})

    mask_variants: list[frozenset[int] | None] = [frozenset({0}), None]

    transforms = []
    for mask in mask_variants:
        # Fire only when every pair is explained by period completion.
        if all(
            inp.shape == out.shape
            and complete(inp, mask if mask is not None else rare_mask(inp))
            == out
            for inp, out in examples
        ):
            if mask is not None:
                def apply(g: Grid, m: frozenset[int] = mask) -> Grid:
                    return complete(g, m) or g

                transforms.append(
                    Transform("periodic_complete", apply, {})
                )
            else:
                def apply_masked(g: Grid) -> Grid:
                    return complete(g, rare_mask(g)) or g

                transforms.append(
                    Transform("periodic_complete_masked", apply_masked, {})
                )
    return transforms


def _propose_translate_complete(examples: list[Example]) -> list[Transform]:
    """Propose completing a pattern under a translation vector.

    Generalizes axis-aligned period completion to diagonal lattices:
    cells that differ by an integer multiple of the vector (a, b)
    must agree. Unknown cells (the mask) are filled from their
    equivalence class. Covers diagonal-stripe textures that a
    (row, col) period cannot express.
    """

    def complete(g: Grid, unknowns: frozenset[int]) -> Grid | None:
        return _lattice_complete(g, unknowns)

    def rare_mask(g: Grid) -> frozenset[int]:
        counts = Counter(v for _, _, v in g.iter_cells() if v != 0)
        if len(counts) < 2:
            return frozenset({0})
        return frozenset({counts.most_common()[-1][0]})

    mask_variants: list[frozenset[int] | None] = [
        frozenset({0}),
        None,  # rare-color scribble mask (0s stay real data)
    ]
    transforms = []
    for mask in mask_variants:
        # The lattice scan is expensive — run it on the first pair
        # only, then verify the resulting transform on the rest.
        first_inp, first_out = examples[0]
        if first_inp.shape != first_out.shape:
            continue
        mask0 = mask if mask is not None else rare_mask(first_inp)
        if complete(first_inp, mask0) != first_out:
            continue
        if mask is not None:
            def apply(g: Grid, m: frozenset[int] = mask) -> Grid:
                return complete(g, m) or g

            if all(apply(inp) == out for inp, out in examples[1:]):
                transforms.append(
                    Transform("translate_complete", apply, {})
                )
        else:
            def apply_masked(g: Grid) -> Grid:
                return complete(g, rare_mask(g)) or g

            if all(
                apply_masked(inp) == out for inp, out in examples[1:]
            ):
                transforms.append(
                    Transform(
                        "translate_complete_masked", apply_masked, {}
                    )
                )
    return transforms


def _lattice_fill(
    g: Grid, vs: list[tuple[int, int]], unknowns: frozenset[int]
) -> Grid | None:
    """Fill g assuming the period lattice generated by vectors vs."""
    h, w = g.height, g.width
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in vs:
        for r in range(h):
            for c in range(w):
                nr, nc = r + a, c + b
                if 0 <= nr < h and 0 <= nc < w:
                    parent[find(r * w + c)] = find(nr * w + nc)
    value: dict[int, int] = {}
    for r, c, v in g.iter_cells():
        if v in unknowns:
            continue
        root = find(r * w + c)
        if root in value and value[root] != v:
            return None
        value[root] = v
    classes = {find(r * w + c) for r in range(h) for c in range(w)}
    if any(root not in value for root in classes):
        return None
    return Grid.from_lists(
        [[value[find(r * w + c)] for c in range(w)] for r in range(h)]
    )


def _lattice_consistent(
    g: Grid, v: tuple[int, int], unknowns: frozenset[int]
) -> bool:
    """True when vector v links no cells with conflicting values."""
    h, w = g.height, g.width
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a, b = v
    for r in range(h):
        for c in range(w):
            nr, nc = r + a, c + b
            if 0 <= nr < h and 0 <= nc < w:
                parent[find(r * w + c)] = find(nr * w + nc)
    value: dict[int, int] = {}
    for r, c, vv in g.iter_cells():
        if vv in unknowns:
            continue
        root = find(r * w + c)
        if root in value and value[root] != vv:
            return False
        value[root] = vv
    return True


def _lattice_complete(g: Grid, unknowns: frozenset[int]) -> Grid | None:
    """Complete g under a 1- or 2-vector translation lattice."""
    h, w = g.height, g.width
    vecs: list[tuple[int, int]] = []
    for a in range(0, h // 2 + 1):
        for b in range(-w // 2, w // 2 + 1):
            if a == 0 and b <= 0:
                continue
            vecs.append((a, b))
    vecs.sort(key=lambda v: abs(v[0]) + abs(v[1]))

    # Single vectors first; collect the consistent ones — only
    # they can appear in a consistent vector pair, since adding
    # links only strengthens the constraints.
    consistent: list[tuple[int, int]] = []
    for v in vecs:
        out = _lattice_fill(g, [v], unknowns)
        if out is not None:
            return out
        # Track vectors that produce no *conflict* (incomplete
        # fills are still pair candidates — a second vector may
        # link the undetermined classes to known ones).
        if _lattice_consistent(g, v, unknowns):
            consistent.append(v)

    # Cap pair search: sparse grids leave many vectors consistent,
    # and O(pairs × cells) blows up. Prefer small vectors.
    consistent = consistent[:40]
    tried = 0
    for i, v1 in enumerate(consistent):
        for v2 in consistent[i + 1 :]:
            tried += 1
            if tried > 2000:
                return None
            out = _lattice_fill(g, [v1, v2], unknowns)
            if out is not None:
                return out
    return None


def _propose_symmetrize(examples: list[Example]) -> list[Transform]:
    """Propose completing a partial symmetry.

    When every output strictly contains the input's cells and the
    added cells are the mirror image of existing ones, the rule is
    "finish the pattern": reflect non-background cells across the
    vertical or horizontal axis, painting only background cells.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != out.at(r, c):
                return []  # existing cells may not change

    def make(axis: str) -> TransformFn:
        def apply(g: Grid) -> Grid:
            bg = g.most_common_color()
            rows = g.to_lists()
            for r, c, v in g.iter_cells():
                if v == bg:
                    continue
                if axis == "v":
                    nr, nc = r, g.width - 1 - c
                else:
                    nr, nc = g.height - 1 - r, c
                if rows[nr][nc] == bg:
                    rows[nr][nc] = v
            return Grid.from_lists(rows)

        return apply

    return [
        Transform(f"symmetrize_{a}", make(a), {"axis": a})
        for a in ("v", "h")
    ]


def _propose_connect_aligned(examples: list[Example]) -> list[Transform]:
    """Propose drawing a line between same-colored aligned objects.

    When two objects share a row-span or column-span and the output
    only *adds* cells, the rule is often "connect them" — draw a
    straight line of the objects' color across the gap.
    """
    # Only viable when every output strictly contains the input's cells.
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 and out.at(r, c) != v:
                return []

    def apply(g: Grid) -> Grid:
        scene = perceive(g, compute_relations=False)
        rows = g.to_lists()
        objs = scene.objects
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if a.color != b.color:
                    continue
                at, al, ab, ar = a.bbox
                bt, bl, bb, br = b.bbox
                # Horizontal connection: row spans overlap, fill the
                # column gap on the first shared row.
                if _spans_overlap(at, ab, bt, bb):
                    row = max(at, bt)
                    c0, c1 = (ar, bl) if al < bl else (br, al)
                    for cc in range(c0, c1 + 1):
                        if rows[row][cc] == scene.background:
                            rows[row][cc] = a.color
                # Vertical connection: column spans overlap.
                if _spans_overlap(al, ar, bl, br):
                    col = max(al, bl)
                    r0, r1 = (ab, bt) if at < bt else (bb, at)
                    for rr in range(r0, r1 + 1):
                        if rows[rr][col] == scene.background:
                            rows[rr][col] = a.color
        return Grid(tuple(tuple(row) for row in rows))

    return [Transform("connect_aligned", apply)]


def _propose_select_by_color(examples: list[Example]) -> list[Transform]:
    """Propose cropping to the object(s) of a specific color.

    `keep_largest`/`keep_smallest` select by size; many tasks
    select by *color* — "extract the red shape". Crop to the union
    bbox of all objects of that color.
    """
    if not all(
        out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]
        for inp, out in examples
    ):
        return []
    colors: set[int] = set()
    for inp, _ in examples:
        colors |= set(inp.colors())
    colors.discard(0)

    return [
        Transform(
            "select_color",
            _mk_select(c),
            {"color": c},
        )
        for c in sorted(colors)
    ]


def _propose_mirror_copy(examples: list[Example]) -> list[Transform]:
    """Propose adding a mirrored copy of each object beside itself.

    Catches "the output adds a reflected/rotated twin of the object"
    — a relational operation (the copy's position depends on the
    original's), not a unary grid transform.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 and out.at(r, c) != v:
                return []

    def make(direction: str) -> TransformFn:
        def apply(g: Grid) -> Grid:
            scene = perceive(g, compute_relations=False)
            result = g
            for obj in scene.objects:
                top, left, bottom, right = obj.bbox
                if direction == "right":
                    new_cells = frozenset(
                        (r, right + (right - c) + 1) for r, c in obj.cells
                    )
                elif direction == "left":
                    new_cells = frozenset(
                        (r, left - (c - left) - 1) for r, c in obj.cells
                    )
                elif direction == "down":
                    new_cells = frozenset(
                        (bottom + (bottom - r) + 1, c) for r, c in obj.cells
                    )
                else:  # up
                    new_cells = frozenset(
                        (top - (r - top) - 1, c) for r, c in obj.cells
                    )
                # Only place the copy if it fits entirely on background.
                if all(
                    0 <= r < g.height
                    and 0 <= c < g.width
                    and result.at(r, c) == scene.background
                    for r, c in new_cells
                ):
                    result = result.overlay(new_cells, obj.color)
            return result

        return apply

    return [
        Transform(f"mirror_copy_{d}", make(d), {"direction": d})
        for d in ("right", "left", "down", "up")
    ]


def _propose_recolor_touching(examples: list[Example]) -> list[Transform]:
    """Propose recoloring objects that touch a reference object.

    "The objects touching the frame turn green" — recolor conditioned
    on a spatial relation, not on color or size alone.
    """
    new_colors: set[int] = set()
    for inp, out in examples:
        if inp.shape != out.shape:
            continue
        for r, c, v in out.iter_cells():
            if v != inp.at(r, c):
                new_colors.add(v)
    new_colors.discard(0)
    if not new_colors:
        return []

    def make(color: int) -> TransformFn:
        def apply(g: Grid) -> Grid:
            scene = perceive(g)
            if not scene.objects:
                return g
            largest = max(scene.objects, key=lambda o: o.size)
            touching = {
                rel.object
                for rel in scene.relations
                if rel.subject == largest.index
                and rel.kind.value == "touching"
            }
            result = g
            for idx in touching:
                result = result.overlay(scene.objects[idx].cells, color)
            return result

        return apply

    return [
        Transform("recolor_touching", make(c), {"color": c})
        for c in sorted(new_colors)
    ]


def _propose_gravity_color(examples: list[Example]) -> list[Transform]:
    """Propose moving only objects of a specific color to an edge.

    Conditional gravity: the marked objects fall, others stay put.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    colors: set[int] = set()
    for inp, _ in examples:
        colors |= set(inp.colors())
    colors.discard(0)
    if not colors:
        return []

    return [
        Transform(
            f"gravity_{d}_{c}",
            _mk_gravity(d, c),
            {"direction": d, "color": c},
        )
        for c in sorted(colors)
        for d in ("down", "up", "left", "right")
    ]


def _propose_mark_uniform(examples: list[Example]) -> list[Transform]:
    """Propose marking uniform rows/columns with a single color.

    Applies when the output keeps the input's shape but its colors
    collapse to at most {0, mark} — the signature of a line-selector.
    The mark color is read from the outputs.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    marks: set[int] = set()
    for _, out in examples:
        oc = set(out.colors())
        if len(oc - {0}) > 1:
            return []
        marks |= oc - {0}
    if len(marks) != 1:
        return []
    color = marks.pop()
    return [
        Transform(
            f"mark_uniform_{axis}",
            _mk_mark_uniform(axis, color),
            {"axis": axis, "color": color},
        )
        for axis in ("row", "column")
    ]


def _propose_shift(examples: list[Example]) -> list[Transform]:
    """Propose rigid translations by a small fixed offset.

    The offset is read from the examples: for the first pair, every
    (dr, dc) that could map a non-zero cell to a same-colored output
    cell is a candidate; candidates are kept only if consistent
    across all pairs' cell colorings.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    inp, out = examples[0]
    candidates: set[tuple[int, int]] = set()
    for r, c, v in inp.iter_cells():
        if v == 0:
            continue
        for rr, cc, ov in out.iter_cells():
            if ov == v:
                candidates.add((rr - r, cc - c))
    for inp, out in examples[1:]:
        keep: set[tuple[int, int]] = set()
        for dr, dc in candidates:
            if all(
                out.at(
                    min(max(r + dr, 0), inp.height - 1),
                    min(max(c + dc, 0), inp.width - 1),
                ) == v
                for r, c, v in inp.iter_cells()
                if v != 0
            ):
                keep.add((dr, dc))
        candidates = keep
    candidates.discard((0, 0))
    return [
        t
        for dr, dc in sorted(candidates)
        for t in (
            Transform(
                "shift", _mk_shift(dr, dc), {"dr": dr, "dc": dc},
            ),
            Transform(
                "shift",
                _mk_shift(dr, dc, clamp=True),
                {"dr": dr, "dc": dc, "clamp": True},
            ),
        )
    ]


def _propose_self_tile(examples: list[Example]) -> list[Transform]:
    """Propose self-stenciling when output dims are input dims squared."""
    for inp, out in examples:
        if (
            out.height != inp.height * inp.height
            or out.width != inp.width * inp.width
        ):
            return []
    return [Transform("self_tile", _mk_self_tile(), {})]


def _propose_mark_diagonals(examples: list[Example]) -> list[Transform]:
    """Propose filling diagonal neighbors of marks with a new color.

    Signature: same shape, and the output introduces a color absent
    from the inputs — the mark that appeared.
    """
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    new_colors: set[int] = set()
    for inp, out in examples:
        new_colors |= set(out.colors()) - set(inp.colors())
    new_colors.discard(0)
    return [
        Transform(
            "mark_diagonals",
            _mk_mark_diagonals(c),
            {"color": c},
        )
        for c in sorted(new_colors)
    ]


def _propose_extend_down(examples: list[Example]) -> list[Transform]:
    """Propose periodic downward growth when the output is the input
    plus extra rows. The row count is read from the examples; the
    period is detected per grid at apply time."""
    n: int | None = None
    for inp, out in examples:
        if out.width != inp.width or out.height <= inp.height:
            return []
        extra = out.height - inp.height
        if n is None:
            n = extra
        elif extra != n:
            return []
    if n is None:
        return []
    return [Transform("extend_down", _mk_extend_down(n), {"n": n})]


def _propose_repeat_down(examples: list[Example]) -> list[Transform]:
    """Propose doubling downward when output height is exactly 2×."""
    for inp, out in examples:
        if out.width != inp.width or out.height != 2 * inp.height:
            return []
    return [Transform("repeat_down", _mk_repeat_down(), {})]


def _propose_intersect_halves(examples: list[Example]) -> list[Transform]:
    """Propose divider-intersection when every input has a full
    monochrome nonzero divider line and the output is smaller."""
    for inp, out in examples:
        if _divider_index(inp) is None:
            return []
        if out.height >= inp.height and out.width >= inp.width:
            return []
    colors: set[int] = set()
    for _, out in examples:
        colors |= set(out.colors())
    colors.discard(0)
    return [
        Transform(
            "intersect_halves", _mk_intersect_halves(c), {"color": c},
        )
        for c in sorted(colors)
    ]


def _propose_empty_halves(examples: list[Example]) -> list[Transform]:
    """Propose divider-NOR: same shape gate as intersect_halves, but
    the output marks positions empty in both halves."""
    for inp, out in examples:
        if _divider_index(inp) is None:
            return []
        if out.height >= inp.height and out.width >= inp.width:
            return []
    colors: set[int] = set()
    for _, out in examples:
        colors |= set(out.colors())
    colors.discard(0)
    return [
        Transform(
            "empty_halves", _mk_empty_halves(c), {"color": c},
        )
        for c in sorted(colors)
    ]


def _propose_diff_halves(examples: list[Example]) -> list[Transform]:
    """Propose divider-XOR: same shape gate as intersect_halves, but
    the output marks positions where the halves differ."""
    for inp, out in examples:
        if _divider_index(inp) is None:
            return []
        if out.height >= inp.height and out.width >= inp.width:
            return []
    colors: set[int] = set()
    for _, out in examples:
        colors |= set(out.colors())
    colors.discard(0)
    return [
        Transform(
            "diff_halves", _mk_diff_halves(c), {"color": c},
        )
        for c in sorted(colors)
    ]


def _propose_pick_quadrant(examples: list[Example]) -> list[Transform]:
    """Propose quadrant-picking when every input has a divider cross
    and the output is a smaller rectangle."""
    for inp, out in examples:
        if _cross_lines(inp) is None:
            return []
        if out.height >= inp.height or out.width >= inp.width:
            return []
    return [Transform("pick_quadrant", _mk_pick_quadrant(), {})]


def _propose_shear(examples: list[Example]) -> list[Transform]:
    """Propose corner-pinned shears by one cell in each cardinal and
    diagonal direction — verification decides which, if any."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    return [
        Transform("shear", _mk_shear(dr, dc), {"dr": dr, "dc": dc})
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]


def _propose_dual_frame(examples: list[Example]) -> list[Transform]:
    """Propose the two-dot frame when every input holds exactly two
    non-zero cells."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        if sum(
            1 for _, _, v in inp.iter_cells() if v != 0
        ) != 2:
            return []
    return [Transform("dual_frame", _mk_dual_frame(), {})]


def _propose_align_tops(examples: list[Example]) -> list[Transform]:
    """Propose aligning every object's top edge to a reference
    color's top edge — one candidate per non-background color."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    colors: set[int] = set()
    for inp, _ in examples:
        colors |= {v for _, _, v in inp.iter_cells() if v != 0}
    return [
        Transform("align_tops", _mk_align_tops(c), {"color": c})
        for c in sorted(colors)
    ]


def _propose_unwind_tile(examples: list[Example]) -> list[Transform]:
    """Propose quadrant-tile reconstruction when the output is half
    the input's content box in each dimension."""
    for inp, out in examples:
        box = inp.crop_to_content()
        if (
            box.height != box.width
            or box.height % 2 != 0
            or out.height != box.height // 2
            or out.width != box.width // 2
        ):
            return []
    return [Transform("unwind_tile", _mk_unwind_tile(), {})]


def _propose_stamp_recolor(examples: list[Example]) -> list[Transform]:
    """Propose stamp-recoloring when some input holds a multicolor
    object — the stamp — among monochrome ones."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        if len(set(inp.colors())) < 3:
            return []
    return [Transform("stamp_recolor", _mk_stamp_recolor(), {})]


def _propose_link_blocks(examples: list[Example]) -> list[Transform]:
    """Propose the bridging-verdict task when every output is a 1x1
    0-or-8 cell and every input holds both 2s and 8s."""
    for inp, out in examples:
        if out.height != 1 or out.width != 1:
            return []
        if not {v for _, _, v in out.iter_cells()} <= {0, 8}:
            return []
        colors = set(inp.colors())
        if 2 not in colors or 8 not in colors:
            return []
    return [Transform("link_blocks", _mk_link_blocks(), {})]


def _propose_cross_halos(examples: list[Example]) -> list[Transform]:
    """Learn a seed-color → halo table: each nonzero cell sprouts
    fixed neighbors. The table is read off the training pairs by
    diffing each cell's 8-neighborhood, then verified in full."""
    # A changed cell may border seeds of several colors; first pass
    # credits only unambiguous cells, then each ambiguous cell goes to
    # the adjacent seed color already producing that halo color (or to
    # every adjacent color when no such preference exists).
    table: dict[int, set[tuple[int, int, int]]] = {}
    ambiguous: list[tuple[int, list[tuple[int, int, int]]]] = []
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 or out.at(r, c) == 0:
                continue
            k = out.at(r, c)
            seeds = [
                (inp.at(r + dr, c + dc), -dr, -dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
                and 0 <= r + dr < inp.height
                and 0 <= c + dc < inp.width
                and inp.at(r + dr, c + dc) != 0
            ]
            colors = {sv for sv, _, _ in seeds}
            if len(colors) == 1:
                for sv, sdr, sdc in seeds:
                    table.setdefault(sv, set()).add((sdr, sdc, k))
            elif seeds:
                ambiguous.append((k, seeds))
    for k, seeds in ambiguous:
        prefer = {
            sv for sv, _, _ in seeds
            if any(hk == k for _, _, hk in table.get(sv, ()))
        }
        claimants = prefer or {sv for sv, _, _ in seeds}
        for sv, sdr, sdc in seeds:
            if sv in claimants:
                table.setdefault(sv, set()).add((sdr, sdc, k))
    if not any(table.values()):
        return []
    halo_table: _HaloTable = {
        color: tuple(sorted(halo))
        for color, halo in table.items()
        if halo
    }
    cand = Transform(
        "cross_halos",
        _mk_cross_halos(halo_table),
        {"table": {str(c): [list(h) for h in hs]
                   for c, hs in halo_table.items()}},
    )
    for inp, out in examples:
        if cand.apply(inp) != out:
            return []
    return [cand]


def _propose_rot_symmetrize(examples: list[Example]) -> list[Transform]:
    """Propose rotational completion when shapes are preserved and
    the output only adds cells of existing colors."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        if not set(out.colors()) <= set(inp.colors()):
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 and out.at(r, c) != v:
                return []
    return [Transform("rot_symmetrize", _mk_rot_symmetrize(), {})]


def _propose_ray_recolor(examples: list[Example]) -> list[Transform]:
    """Propose edge-ray recoloring when shapes are preserved and the
    output only recolors cells that were already non-zero."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in out.iter_cells():
            if v != inp.at(r, c) and inp.at(r, c) == 0:
                return []
    return [Transform("ray_recolor", _mk_ray_recolor(), {})]


def _propose_nearest_border(examples: list[Example]) -> list[Transform]:
    """Propose nearest-border recoloring when shapes are preserved and
    every input is flanked by two monochrome edge lines."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        vertical = (
            _edge_color([inp.at(r, 0) for r in range(inp.height)])
            is not None
            and _edge_color(
                [inp.at(r, inp.width - 1) for r in range(inp.height)]
            ) is not None
        )
        horizontal = (
            _edge_color([inp.at(0, c) for c in range(inp.width)])
            is not None
            and _edge_color(
                [inp.at(inp.height - 1, c) for c in range(inp.width)]
            ) is not None
        )
        if not (vertical or horizontal):
            return []
    return [Transform("nearest_border", _mk_nearest_border(), {})]


def _propose_ghost_pair(examples: list[Example]) -> list[Transform]:
    """Propose ghost-pair copies when outputs keep every input cell and
    add cells of a single new color."""
    candidates: set[int] | None = None
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 and out.at(r, c) != v:
                return []
        new = {v for _, _, v in out.iter_cells()} - {
            v for _, _, v in inp.iter_cells()
        }
        if candidates is None:
            candidates = new
        else:
            candidates &= new
    if not candidates:
        return []
    return [
        Transform("ghost_pair", _mk_ghost_pair(color), {"color": color})
        for color in sorted(candidates)
    ]


def _propose_axes_stamp(examples: list[Example]) -> list[Transform]:
    """Propose axis cross-stamps when outputs keep every input cell and
    add cells of a single new color — same gate as ghost_pair."""
    candidates: set[int] | None = None
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        for r, c, v in inp.iter_cells():
            if v != 0 and out.at(r, c) != v:
                return []
        new = {v for _, _, v in out.iter_cells()} - {
            v for _, _, v in inp.iter_cells()
        }
        if candidates is None:
            candidates = new
        else:
            candidates &= new
    if not candidates:
        return []
    return [
        Transform(
            "axes_stamp",
            _mk_axes_stamp(color, trans),
            {"color": color, "transpose": trans},
        )
        for color in sorted(candidates)
        for trans in (False, True)
    ]


def _propose_ring_unique(examples: list[Example]) -> list[Transform]:
    """Propose singleton-ring when each input has exactly one
    color occurring once and the output is a one-color ring around
    it (plus the kept center cell)."""
    colors: set[int] | None = None
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        counts = inp.color_counts()
        if len([v for v, n in counts.items() if n == 1 and v != 0]) != 1:
            return []
        ring = Counter(
            v for _, _, v in out.iter_cells() if v != 0
        ).most_common()
        if not ring or ring[0][1] < 2:
            return []
        ring_color = ring[0][0]
        colors = {ring_color} if colors is None else colors | {
            ring_color
        }
    if not colors:
        return []
    return [
        Transform("ring_unique", _mk_ring_unique(c), {"color": c})
        for c in sorted(colors)
    ]


def _propose_attract_to(examples: list[Example]) -> list[Transform]:
    """Propose slide-to-anchor when shapes are preserved and each
    input holds exactly two objects of distinct colors."""
    colors: set[int] | None = None
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        scene = perceive(inp, compute_relations=False)
        if len(scene.objects) != 2:
            return []
        have = {o.color for o in scene.objects}
        colors = have if colors is None else colors & have
    if not colors:
        return []
    return [
        Transform("attract_to", _mk_attract_to(c), {"anchor": c})
        for c in sorted(colors)
    ]


def _propose_stamp_at_marks(
    examples: list[Example],
) -> list[Transform]:
    """Propose template-stamping when every input has a divider line
    that the output preserves (the marks themselves get stamped
    over, so only the divider is invariant)."""
    for inp, out in examples:
        found = _divider_index(inp)
        if inp.shape != out.shape or found is None:
            return []
        axis, idx = found
        if axis == "column":
            if not all(
                inp.at(r, idx) == out.at(r, idx)
                for r in range(inp.height)
            ):
                return []
        else:
            if not all(
                inp.at(idx, c) == out.at(idx, c)
                for c in range(inp.width)
            ):
                return []
    return [Transform("stamp_at_marks", _mk_stamp_at_marks(), {})]


def _propose_radial_map(examples: list[Example]) -> list[Transform]:
    """Propose radial compression when every output is 3x3 with a
    centered marker cell: each input color whose cells all map onto
    the marker's neighborhood is a candidate marker color."""
    colors: set[int] | None = None
    for inp, out in examples:
        if out.shape != (3, 3):
            return []
        have = {
            v
            for _, _, v in inp.iter_cells()
            if v != 0
            and _mk_radial_map(v)(inp) == out
        }
        colors = have if colors is None else colors & have
    if not colors:
        return []
    return [
        Transform("radial_map", _mk_radial_map(c), {"color": c})
        for c in sorted(colors)
    ]


def _propose_fill_busiest(examples: list[Example]) -> list[Transform]:
    """Propose busiest-block filling when every pair keeps its shape,
    every input is crossed by monochrome divider lines, and replaying
    the rule reproduces the outputs exactly."""
    if any(inp.shape != out.shape for inp, out in examples):
        return []
    if all(_apply_fill_busiest(inp) == out for inp, out in examples):
        return [Transform("fill_busiest", _mk_fill_busiest(), {})]
    return []


def _propose_fill_lanes(examples: list[Example]) -> list[Transform]:
    """Propose lane-filling when outputs keep the input's geometry and
    every output-only color can replay the lane rule exactly."""
    if any(inp.shape != out.shape for inp, out in examples):
        return []
    colors = {
        v for _, out in examples for _, _, v in out.iter_cells()
    } - {0}
    return [
        Transform("fill_lanes", _mk_fill_lanes(c), {"color": c})
        for c in sorted(colors)
        if all(_mk_fill_lanes(c)(inp) == out for inp, out in examples)
    ]


def _propose_eye_ray(examples: list[Example]) -> list[Transform]:
    """Propose the eye ray when every pair has two non-zero colors,
    one of them a lone cell, and the replay verifies."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        counts = Counter(
            v for _, _, v in inp.iter_cells() if v != 0
        )
        if len(counts) != 2 or 1 not in counts.values():
            return []
    if all(_apply_eye_ray(inp) == out for inp, out in examples):
        return [Transform("eye_ray", _mk_eye_ray(), {})]
    return []


def _propose_cross_fill(examples: list[Example]) -> list[Transform]:
    """Propose cross-filling when every input is divided into a 3x3
    cell array and the replay verifies."""
    if any(inp.shape != out.shape for inp, out in examples):
        return []
    if all(_apply_cross_fill(inp) == out for inp, out in examples):
        return [Transform("cross_fill", _mk_cross_fill(), {})]
    return []


def _propose_draw_lines(examples: list[Example]) -> list[Transform]:
    """Learn a color → row/column table when each seed cell emits a
    full line. Axis per color is guessed by coverage, then every
    consistent assignment is verified against all pairs."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    guessed: dict[int, str] = {}
    ambiguous: set[int] = set()
    for inp, out in examples:
        for r, c, v in inp.iter_cells():
            if v == 0:
                continue
            row_hits = sum(
                1 for cc in range(out.width) if out.at(r, cc) == v
            )
            col_hits = sum(
                1 for rr in range(out.height) if out.at(rr, c) == v
            )
            if row_hits == col_hits:
                ambiguous.add(v)
            elif v in guessed and guessed[v] != (
                "row" if row_hits > col_hits else "column"
            ):
                ambiguous.add(v)
            else:
                guessed[v] = "row" if row_hits > col_hits else "column"
    colors = sorted(set(guessed) | ambiguous)
    if not colors or len(colors) > 4:
        return []
    found: list[Transform] = []
    seen: set[tuple[tuple[int, str], ...]] = set()
    for mask in range(1 << len(colors)):
        table = {}
        for i, v in enumerate(colors):
            if v in ambiguous:
                table[v] = "row" if mask & (1 << i) else "column"
            else:
                table[v] = guessed[v]
        key = tuple(sorted(table.items()))
        if key in seen:
            continue
        seen.add(key)
        cand = Transform(
            "draw_lines", _mk_draw_lines(table),
            {"table": {str(k): a for k, a in table.items()}},
        )
        if all(cand.apply(inp) == out for inp, out in examples):
            found.append(cand)
    return found


def _propose_count_blocks(examples: list[Example]) -> list[Transform]:
    """Propose 2x2-block counting when every output is a single row
    of one color padded with zeros — the counted color is whichever
    input color's block count explains every output."""
    widths = {out.width for _, out in examples}
    if any(out.height != 1 for _, out in examples) or len(widths) != 1:
        return []
    width = widths.pop()
    candidates: set[int] | None = None
    for inp, out in examples:
        row = [out.at(0, c) for c in range(width)]
        possible = {
            c for c in inp.colors() if c != 0
            and row == [c] * _count_squares(inp, c) + [0] * (
                width - _count_squares(inp, c))
        }
        candidates = possible if candidates is None else candidates & possible
        if not candidates:
            return []
    if not candidates:
        return []
    return [
        Transform("count_blocks", _mk_count_blocks(c, width),
                  {"color": c, "width": width})
        for c in sorted(candidates)
    ]


def _propose_classify_shape(examples: list[Example]) -> list[Transform]:
    """Propose shape→label classification when every output is 1x1.

    The label table is read off the training pairs keyed by
    color-invariant mask; conflicting masks (same shape, different
    labels) mean the task isn't pure shape classification.
    """
    table: dict[tuple[tuple[int, ...], ...], int] = {}
    shapes: set[tuple[int, int]] = set()
    for inp, out in examples:
        if out.height != 1 or out.width != 1:
            return []
        key = _mask_key(inp)
        label = out.at(0, 0)
        if key in table and table[key] != label:
            return []
        table[key] = label
        shapes.add(inp.shape)
    if len(shapes) != 1:
        return []
    return [
        Transform("classify_shape", _mk_classify_shape(table),
                  {"table": table})
    ]


def _propose_settle(examples: list[Example]) -> list[Transform]:
    """Propose cell-packing against an edge when shapes are preserved
    and cell counts are conserved."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        if sum(1 for _, _, v in inp.iter_cells() if v != 0) != sum(
            1 for _, _, v in out.iter_cells() if v != 0
        ):
            return []
    return [
        Transform(
            f"settle_{d}", _mk_settle(d), {"direction": d},
        )
        for d in ("down", "up", "left", "right")
    ]


def _propose_bridge_endpoints(examples: list[Example]) -> list[Transform]:
    """Propose endpoint bridging when shapes are preserved and a new
    marker color appears in the outputs."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    markers: set[int] = set()
    for inp, out in examples:
        markers |= set(out.colors()) - set(inp.colors())
    markers.discard(0)
    return [
        Transform(
            "bridge_endpoints", _mk_bridge_endpoints(m), {"marker": m},
        )
        for m in sorted(markers)
    ]


def _propose_rank_bars(examples: list[Example]) -> list[Transform]:
    """Propose height-ranking when inputs are column bars and the
    output introduces rank colors."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
    new_colors: set[int] = set()
    for inp, out in examples:
        new_colors |= set(out.colors()) - set(inp.colors())
    new_colors.discard(0)
    if new_colors != set(range(1, len(new_colors) + 1)):
        return []
    return [Transform("rank_bars", _mk_rank_bars(), {})]


def _propose_cross_rays(examples: list[Example]) -> list[Transform]:
    """Propose ray-crossing when outputs keep the input's dots but add
    full rows/columns plus one clash color."""
    for inp, out in examples:
        if inp.shape != out.shape:
            return []
        dots = sum(
            1
            for r in range(inp.height)
            for c in range(inp.width)
            if inp.at(r, c) != 0
        )
        if dots < 2 or dots > 4:
            return []
    new_colors: set[int] = set()
    for inp, out in examples:
        new_colors |= set(out.colors()) - set(inp.colors())
    new_colors.discard(0)
    return [
        Transform("cross_rays", _mk_cross_rays(cl), {"clash": cl})
        for cl in sorted(new_colors)
    ]


#: All proposal generators. Order matters only for readability — the
#: solver tries every proposed transform at each search step.
PROPOSERS: tuple[Callable[[list[Example]], list[Transform]], ...] = (
    _propose_identity,
    _propose_rotations,
    _propose_reflections,
    _propose_recolor,
    _propose_crop,
    _propose_scale,
    _propose_tile,
    _propose_gravity,
    _propose_erase_color,
    _propose_keep_object,
    _propose_object_recolor,
    _propose_fill_enclosed,
    _propose_attract_marks,
    _propose_denoise,
    _propose_periodic_completion,
    _propose_translate_complete,
    _propose_symmetrize,
    _propose_connect_aligned,
    _propose_select_by_color,
    _propose_mirror_copy,
    _propose_recolor_touching,
    _propose_gravity_color,
    _propose_mark_uniform,
    _propose_shift,
    _propose_self_tile,
    _propose_mark_diagonals,
    _propose_extend_down,
    _propose_repeat_down,
    _propose_intersect_halves,
    _propose_classify_shape,
    _propose_settle,
    _propose_bridge_endpoints,
    _propose_rank_bars,
    _propose_cross_rays,
    _propose_empty_halves,
    _propose_pick_quadrant,
    _propose_shear,
    _propose_dual_frame,
    _propose_align_tops,
    _propose_unwind_tile,
    _propose_stamp_recolor,
    _propose_link_blocks,
    _propose_cross_halos,
    _propose_count_blocks,
    _propose_diff_halves,
    _propose_draw_lines,
    _propose_rot_symmetrize,
    _propose_ray_recolor,
    _propose_nearest_border,
    _propose_ghost_pair,
    _propose_axes_stamp,
    _propose_ring_unique,
    _propose_attract_to,
    _propose_stamp_at_marks,
    _propose_radial_map,
    _propose_fill_busiest,
    _propose_fill_lanes,
    _propose_eye_ray,
    _propose_cross_fill,
)


def propose(examples: list[Example]) -> list[Transform]:
    """Generate all candidate transforms consistent with the examples."""
    out: list[Transform] = []
    for proposer in PROPOSERS:
        try:
            out.extend(proposer(examples))
        except Exception as e:  # noqa: BLE001
            # a malformed proposer must not break the search
            _logger.debug(
                f"transform proposer {getattr(proposer, '__name__', proposer)} failed: {e}"
            )
            continue
    return out

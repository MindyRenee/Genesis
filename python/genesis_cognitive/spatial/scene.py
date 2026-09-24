"""Scene perception — segmenting a grid into objects and relations.

This is the dorsal-stream layer of spatial reasoning: it takes the raw
grid and produces a structured scene — a set of ``PerceivedObject``s
(connected components of uniform color) plus the pairwise spatial
relations between them (above, inside, touching, same shape, ...).

The output is deliberately symbolic: a ``Scene`` is a relational graph,
not a bitmap. That is what makes it groundable — ``grounding.py`` can
inject the graph into the concept network as SPATIAL_RELATION and
HAS_PROPERTY edges, where the semantic reasoning machinery can operate
on it. Perception computes structure; cognition reasons about it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .grid import Grid


class SpatialRelationKind(Enum):
    """Pairwise spatial/structural relations between perceived objects."""

    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"  # bbox containment
    CONTAINS = "contains"
    TOUCHING = "touching"  # 4- or 8-adjacent cells
    SAME_COLOR = "same_color"
    SAME_SHAPE = "same_shape"  # identical cell pattern up to translation
    ALIGNED_H = "aligned_h"  # share a row span
    ALIGNED_V = "aligned_v"  # share a column span
    LARGER_THAN = "larger_than"
    SMALLER_THAN = "smaller_than"


@dataclass(frozen=True)
class ObjectRelation:
    """A directed relation: ``objects[subject] --kind--> objects[object]``."""

    subject: int
    kind: SpatialRelationKind
    object: int


@dataclass
class PerceivedObject:
    """A connected region of uniform color."""

    index: int
    color: int
    cells: frozenset[tuple[int, int]]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(top, left, bottom, right) inclusive."""
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return min(rs), min(cs), max(rs), max(cs)

    @property
    def centroid(self) -> tuple[float, float]:
        rs = [r for r, _ in self.cells]
        cs = [c for _, c in self.cells]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    @property
    def shape_signature(self) -> tuple[tuple[int, int], ...]:
        """Translation-invariant cell pattern — the object's shape."""
        top, left, _, _ = self.bbox
        return tuple(sorted((r - top, c - left) for r, c in self.cells))

    def translated(self, dr: int, dc: int) -> frozenset[tuple[int, int]]:
        return frozenset((r + dr, c + dc) for r, c in self.cells)


@dataclass
class Scene:
    """A perceived scene: the grid plus its object/relational structure."""

    grid: Grid
    background: int
    objects: list[PerceivedObject] = field(default_factory=list)
    relations: list[ObjectRelation] = field(default_factory=list)

    def object_at(self, r: int, c: int) -> PerceivedObject | None:
        for obj in self.objects:
            if (r, c) in obj.cells:
                return obj
        return None


def _connected_components(
    grid: Grid, background: int, diagonal: bool = False
) -> list[PerceivedObject]:
    """Extract uniform-color connected components (4- or 8-connectivity)."""
    if diagonal:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited: set[tuple[int, int]] = set()
    objects: list[PerceivedObject] = []
    for r, c, color in grid.iter_cells():
        if (r, c) in visited or color == background:
            continue
        cells: set[tuple[int, int]] = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            cr, cc = queue.popleft()
            cells.add((cr, cc))
            for dr, dc in neighbors:
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < grid.height
                    and 0 <= nc < grid.width
                    and (nr, nc) not in visited
                    and grid.at(nr, nc) == color
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        objects.append(
            PerceivedObject(index=len(objects), color=color, cells=frozenset(cells))
        )
    return objects


def _bboxes_touch(
    a: PerceivedObject, b: PerceivedObject, diagonal: bool = True
) -> bool:
    """True if any cell of a is (8-)adjacent to any cell of b."""
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    b_cells = b.cells
    for r, c in a.cells:
        for dr, dc in deltas:
            if (r + dr, c + dc) in b_cells:
                return True
    return False


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 <= b1 and b0 <= a1


def _compute_relations(objects: list[PerceivedObject]) -> list[ObjectRelation]:
    """Compute pairwise spatial relations between objects."""
    relations: list[ObjectRelation] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            at, al, ab, ar = a.bbox
            bt, bl, bb, br = b.bbox

            # Positional (requires a clear separation on the axis so
            # "above" isn't claimed for merely offset objects).
            if ab < bt:
                relations.append(ObjectRelation(i, SpatialRelationKind.ABOVE, j))
            if at > bb:
                relations.append(ObjectRelation(i, SpatialRelationKind.BELOW, j))
            if ar < bl:
                relations.append(ObjectRelation(i, SpatialRelationKind.LEFT_OF, j))
            if al > br:
                relations.append(ObjectRelation(i, SpatialRelationKind.RIGHT_OF, j))

            # Containment via bounding box.
            if at >= bt and al >= bl and ab <= bb and ar <= br:
                relations.append(ObjectRelation(i, SpatialRelationKind.INSIDE, j))
                relations.append(ObjectRelation(j, SpatialRelationKind.CONTAINS, i))

            # Alignment via axis-span overlap.
            if _spans_overlap(at, ab, bt, bb):
                relations.append(ObjectRelation(i, SpatialRelationKind.ALIGNED_H, j))
            if _spans_overlap(al, ar, bl, br):
                relations.append(ObjectRelation(i, SpatialRelationKind.ALIGNED_V, j))

            # Attribute relations.
            if a.color == b.color:
                relations.append(ObjectRelation(i, SpatialRelationKind.SAME_COLOR, j))
            if a.shape_signature == b.shape_signature:
                relations.append(ObjectRelation(i, SpatialRelationKind.SAME_SHAPE, j))
            if a.size > b.size:
                relations.append(ObjectRelation(i, SpatialRelationKind.LARGER_THAN, j))
            elif a.size < b.size:
                relations.append(ObjectRelation(i, SpatialRelationKind.SMALLER_THAN, j))

            # Adjacency.
            if _bboxes_touch(a, b):
                relations.append(ObjectRelation(i, SpatialRelationKind.TOUCHING, j))
    return relations


# Perception is memoized: grids are immutable and hashable, and the
# solver evaluates thousands of transform applications — many against
# the same intermediate grids. Bounded so long searches can't grow it
# without limit.
_SCENE_CACHE_SIZE = 20_000
_scene_cache: dict[tuple[Grid, int | None, bool, bool], Scene] = {}


def perceive(
    grid: Grid,
    background: int | None = None,
    diagonal: bool = False,
    compute_relations: bool = True,
) -> Scene:
    """Segment a grid into a scene of objects and relations.

    Args:
        grid: the raw grid to perceive.
        background: cell value treated as empty space. Defaults to the
            grid's most common color — the conventional background
            assumption — but callers can override it when the output
            grid's background differs (e.g. recolored tasks).
        diagonal: use 8-connectivity for object extraction instead of
            the default 4-connectivity.
        compute_relations: compute the pairwise relation graph. The
            solver disables this — relation computation is quadratic in
            the number of objects and the search only needs objects.
    """
    key = (grid, background, diagonal, compute_relations)
    cached = _scene_cache.get(key)
    if cached is not None:
        return cached
    bg = grid.most_common_color() if background is None else background
    objects = _connected_components(grid, bg, diagonal=diagonal)
    relations = _compute_relations(objects) if compute_relations else []
    scene = Scene(grid=grid, background=bg, objects=objects, relations=relations)
    if len(_scene_cache) < _SCENE_CACHE_SIZE:
        _scene_cache[key] = scene
    return scene

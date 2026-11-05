"""Structural latent decoding and empirical dynamics for ARC-AGI-3 boards.

The representation is an interpretable, fixed-width feature embedding backed
by an attributed scene graph. It is not a learned neural embedding and does
not claim semantic object identity from color alone. Object geometry and
relations remain available beside the vector so downstream cognition can
inspect the evidence instead of losing it in a projection.

Transitions are learned only from observed action/result pairs. Uncertainty
is retained until repeated observations provide evidence; it is never
reduced merely because an action was attempted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot, isfinite, sqrt
from typing import Any

from parietal_lobe.grid import Grid
from parietal_lobe.scene import perceive

ARC_COLORS = 16
FEATURE_DIMENSIONS = 35


@dataclass(frozen=True, slots=True)
class GridState:
    """Validated ARC grid using row-major integer cells."""

    width: int
    height: int
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("grid dimensions must be positive")
        if len(self.data) != self.width * self.height:
            raise ValueError("grid data length does not match dimensions")
        if any(not isinstance(value, int) or not 0 <= value < ARC_COLORS for value in self.data):
            raise ValueError("ARC cells must be integer palette values in [0, 15]")

    @classmethod
    def from_rows(cls, rows: list[list[int]]) -> GridState:
        """Validate and flatten a rectangular board."""
        if not rows or not rows[0]:
            raise ValueError("grid must be non-empty")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("grid rows must have equal width")
        return cls(width, len(rows), tuple(value for row in rows for value in row))

    def get(self, x: int, y: int) -> int:
        """Read one cell using x=column and y=row coordinates."""
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise IndexError("grid coordinate is outside the board")
        return self.data[y * self.width + x]

    def as_grid(self) -> Grid:
        """Return Genesis's canonical immutable parietal grid."""
        rows = tuple(
            self.data[row * self.width : (row + 1) * self.width]
            for row in range(self.height)
        )
        return Grid(rows)


@dataclass(frozen=True, slots=True)
class SceneNode:
    """One connected same-color component with geometric/topological attributes."""

    id: int
    color: int
    centroid: tuple[float, float]
    pixel_count: int
    bbox: tuple[int, int, int, int]
    betti_0: int
    betti_1: int
    shape_signature: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class SceneEdge:
    """Directed pairwise relation; dx/dy point from source to target."""

    source: int
    target: int
    dx: float
    dy: float
    adjacency: bool
    same_color: bool
    same_shape: bool


@dataclass(frozen=True, slots=True)
class SceneGraph:
    """Attributed relational graph extracted from one ARC grid."""

    width: int
    height: int
    background: int
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]

    @classmethod
    def extract(cls, state: GridState, background: int = 0) -> SceneGraph:
        """Use Genesis's scene parser, then enrich components with topology."""
        grid = state.as_grid()
        scene = perceive(grid, background=background, compute_relations=False)
        objects = sorted(
            scene.objects,
            key=lambda obj: (
                obj.color,
                obj.centroid[0],
                obj.centroid[1],
                obj.size,
                obj.shape_signature,
            ),
        )
        nodes: list[SceneNode] = []
        object_cells: list[frozenset[tuple[int, int]]] = []
        for index, obj in enumerate(objects):
            cells = obj.cells
            top, left, bottom, right = obj.bbox
            nodes.append(
                SceneNode(
                    id=index,
                    color=obj.color,
                    centroid=(obj.centroid[1], obj.centroid[0]),
                    pixel_count=obj.size,
                    bbox=(left, top, right, bottom),
                    betti_0=1,
                    betti_1=_hole_count(cells, top, left, bottom, right),
                    shape_signature=obj.shape_signature,
                )
            )
            object_cells.append(cells)

        edges: list[SceneEdge] = []
        for source, first in enumerate(nodes):
            for target, second in enumerate(nodes):
                if source == target:
                    continue
                left_a, top_a, right_a, bottom_a = first.bbox
                left_b, top_b, right_b, bottom_b = second.bbox
                if (
                    left_a > right_b + 1
                    or left_b > right_a + 1
                    or top_a > bottom_b + 1
                    or top_b > bottom_a + 1
                ):
                    continue
                if _touches(object_cells[source], object_cells[target]):
                    edges.append(
                        SceneEdge(
                            source=source,
                            target=target,
                            dx=second.centroid[0] - first.centroid[0],
                            dy=second.centroid[1] - first.centroid[1],
                            adjacency=True,
                            same_color=first.color == second.color,
                            same_shape=first.shape_signature == second.shape_signature,
                        )
                    )
        return cls(state.width, state.height, background, tuple(nodes), tuple(edges))


def _hole_count(
    cells: frozenset[tuple[int, int]], top: int, left: int, bottom: int, right: int
) -> int:
    """Count enclosed complement components inside an object's bounding box."""
    if not cells:
        return 0
    visited: set[tuple[int, int]] = set()
    holes = 0
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            start = (row, col)
            if start in cells or start in visited:
                continue
            queue = deque([start])
            visited.add(start)
            reaches_boundary = False
            while queue:
                current_row, current_col = queue.popleft()
                if (
                    current_row in (top, bottom)
                    or current_col in (left, right)
                ):
                    reaches_boundary = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        neighbor = (current_row + dr, current_col + dc)
                        nr, nc = neighbor
                        if (
                            top <= nr <= bottom
                            and left <= nc <= right
                            and neighbor not in cells
                            and neighbor not in visited
                        ):
                            visited.add(neighbor)
                            queue.append(neighbor)
            if not reaches_boundary:
                holes += 1
    return holes


def _touches(
    first: frozenset[tuple[int, int]], second: frozenset[tuple[int, int]]
) -> bool:
    """True when two components share a four-neighbor boundary."""
    if len(first) > len(second):
        first, second = second, first
    return any(
        (row + dr, col + dc) in second
        for row, col in first
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


@dataclass(frozen=True, slots=True)
class LatentState:
    """Fixed-width structural feature vector and per-coordinate uncertainty."""

    vector: tuple[float, ...]
    epistemic_variance: tuple[float, ...]
    graph: SceneGraph = field(compare=False)

    def __post_init__(self) -> None:
        if not self.vector or len(self.vector) != len(self.epistemic_variance):
            raise ValueError("latent vector and uncertainty dimensions must agree")
        if not all(isfinite(value) for value in self.vector):
            raise ValueError("latent coordinates must be finite")
        if any(not isfinite(value) or value < 0 for value in self.epistemic_variance):
            raise ValueError("epistemic variances must be finite and non-negative")

    def distance(self, other: LatentState) -> float:
        """Euclidean distance in this fixed feature coordinate system."""
        if len(self.vector) != len(other.vector):
            raise ValueError("latent dimensions must agree")
        return sqrt(sum((a - b) ** 2 for a, b in zip(self.vector, other.vector, strict=True)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize only stable numeric features and graph summary."""
        return {
            "vector": list(self.vector),
            "epistemic_variance": list(self.epistemic_variance),
            "object_count": len(self.graph.nodes),
            "relation_count": len(self.graph.edges),
            "width": self.graph.width,
            "height": self.graph.height,
            "objects": [
                {
                    "id": node.id,
                    "color": node.color,
                    "centroid": node.centroid,
                    "pixel_count": node.pixel_count,
                    "bbox": node.bbox,
                    "betti_0": node.betti_0,
                    "betti_1": node.betti_1,
                }
                for node in self.graph.nodes[:128]
            ],
        }


def encode_graph(graph: SceneGraph) -> LatentState:
    """Encode grid, color, shape, topology, and relational structure."""
    width, height = graph.width, graph.height
    area = width * height
    nodes = graph.nodes
    counts = [0] * ARC_COLORS
    for node in nodes:
        counts[node.color] += node.pixel_count
    occupied = sum(counts)
    max_object = max((node.pixel_count for node in nodes), default=0)
    mean_area = occupied / max(1, len(nodes))
    mean_radius = 0.0
    if nodes:
        center_x = sum(node.centroid[0] for node in nodes) / len(nodes)
        center_y = sum(node.centroid[1] for node in nodes) / len(nodes)
        mean_radius = sum(
            hypot(node.centroid[0] - center_x, node.centroid[1] - center_y)
            for node in nodes
        ) / len(nodes)
    pair_count = len(graph.edges)
    possible_pairs = len(nodes) * (len(nodes) - 1)
    adjacency_ratio = len(graph.edges) / possible_pairs if possible_pairs else 0.0
    same_color_ratio = (
        sum(edge.same_color for edge in graph.edges) / pair_count if pair_count else 0.0
    )
    same_shape_ratio = (
        sum(edge.same_shape for edge in graph.edges) / pair_count if pair_count else 0.0
    )
    vector = [
        width / 64.0,
        height / 64.0,
        area / 4096.0,
        occupied / area,
        len(nodes) / area,
        len({node.color for node in nodes}) / ARC_COLORS,
        mean_area / area,
        max_object / area,
        sum(node.betti_1 for node in nodes) / max(1, len(nodes)),
        sum(node.betti_1 for node in nodes) / area,
        mean_radius / hypot(width, height),
        adjacency_ratio,
        same_color_ratio,
        same_shape_ratio,
        sum(node.centroid[0] for node in nodes) / max(1, len(nodes) * width),
        sum(node.centroid[1] for node in nodes) / max(1, len(nodes) * height),
        sum(node.pixel_count for node in nodes if node.pixel_count == 1) / area,
        sum(node.pixel_count for node in nodes if node.pixel_count > 1) / area,
        sum(node.pixel_count for node in nodes if node.betti_1 > 0) / max(1, len(nodes)),
    ]
    vector.extend(count / area for count in counts)
    if len(vector) != FEATURE_DIMENSIONS:
        raise RuntimeError("ARC latent feature schema dimension drifted")
    return LatentState(tuple(vector), (1.0,) * len(vector), graph)


@dataclass
class _ActionStatistics:
    """Welford statistics for one local region of latent state space."""

    reference_state: tuple[float, ...]
    samples: int = 0
    mean_delta: list[float] = field(default_factory=list)
    squared_deviation: list[float] = field(default_factory=list)

    def update(self, delta: tuple[float, ...]) -> None:
        if not self.mean_delta:
            self.mean_delta = [0.0] * len(delta)
            self.squared_deviation = [0.0] * len(delta)
        if len(delta) != len(self.mean_delta):
            raise ValueError("transition feature dimensions changed")
        self.samples += 1
        for index, value in enumerate(delta):
            difference = value - self.mean_delta[index]
            self.mean_delta[index] += difference / self.samples
            self.squared_deviation[index] += difference * (
                value - self.mean_delta[index]
            )

    def epistemic_variance(self) -> tuple[float, ...]:
        if self.samples < 2:
            return (1.0,) * len(self.mean_delta)
        return tuple(
            max(0.0, value / (self.samples * (self.samples - 1)))
            for value in self.squared_deviation
        )


@dataclass(frozen=True, slots=True)
class LatentPrediction:
    """Predicted feature coordinates, explicitly not an observed scene graph."""

    vector: tuple[float, ...]
    epistemic_variance: tuple[float, ...]
    source_distance: float


@dataclass(frozen=True, slots=True)
class GoalEvidence:
    """Evidence about progress; official terminal status remains authoritative."""

    state: str
    level_before: int
    level_after: int
    changed_cells: int

    @property
    def level_progress(self) -> bool:
        return self.level_after > self.level_before

    @property
    def terminal_success(self) -> bool:
        return self.state.upper() == "WIN"


class ArcEnvironmentDecoder:
    """Encode observations and learn local, action-conditioned dynamics."""

    MAX_CONTEXTS_PER_ACTION = 256
    CONTEXT_RADIUS = 0.15

    def __init__(self, saved: dict[str, Any] | None = None) -> None:
        self._actions: dict[str, list[_ActionStatistics]] = {}
        if saved:
            for action, raw_contexts in saved.items():
                contexts = raw_contexts if isinstance(raw_contexts, list) else [raw_contexts]
                restored: list[_ActionStatistics] = []
                for values in contexts:
                    if not isinstance(values, dict):
                        continue
                    reference = tuple(
                        float(value) for value in values.get("reference_state", [])
                    )
                    samples = int(values.get("samples", 0))
                    mean = [float(value) for value in values.get("mean_delta", [])]
                    m2 = [float(value) for value in values.get("squared_deviation", [])]
                    if (
                        len(reference) == FEATURE_DIMENSIONS
                        and samples >= 0
                        and len(mean) == FEATURE_DIMENSIONS
                        and len(m2) == FEATURE_DIMENSIONS
                        and all(isfinite(value) for value in reference + tuple(mean) + tuple(m2))
                        and all(value >= 0 for value in m2)
                    ):
                        restored.append(_ActionStatistics(reference, samples, mean, m2))
                if restored:
                    self._actions[str(action)] = restored[: self.MAX_CONTEXTS_PER_ACTION]

    def encode(self, rows: list[list[int]]) -> LatentState:
        """Decode one observation into a graph-backed latent state."""
        state = GridState.from_rows(rows)
        return encode_graph(SceneGraph.extract(state))

    def _nearest_context(
        self, action: str, state: LatentState
    ) -> tuple[_ActionStatistics | None, float]:
        contexts = self._actions.get(action, [])
        if not contexts:
            return None, float("inf")
        nearest = min(
            contexts,
            key=lambda context: _vector_distance(context.reference_state, state.vector),
        )
        return nearest, _vector_distance(nearest.reference_state, state.vector)

    def observe_transition(
        self, action: str, before: LatentState, after: LatentState
    ) -> None:
        """Update a local action model from a real before/action/after observation."""
        if not action or not action.strip():
            raise ValueError("action name must be non-empty")
        if len(before.vector) != len(after.vector):
            raise ValueError("transition latent dimensions must agree")
        contexts = self._actions.setdefault(action, [])
        statistics, distance = self._nearest_context(action, before)
        if statistics is None or (
            distance > self.CONTEXT_RADIUS
            and len(contexts) < self.MAX_CONTEXTS_PER_ACTION
        ):
            statistics = _ActionStatistics(before.vector)
            contexts.append(statistics)
        delta = tuple(
            right - left
            for left, right in zip(before.vector, after.vector, strict=True)
        )
        statistics.update(delta)

    def predict(self, action: str, state: LatentState) -> LatentPrediction | None:
        """Predict only from a sufficiently nearby observed state/action context."""
        statistics, distance = self._nearest_context(action, state)
        if (
            statistics is None
            or statistics.samples == 0
            or distance > self.CONTEXT_RADIUS
        ):
            return None
        vector = tuple(
            value + change
            for value, change in zip(state.vector, statistics.mean_delta, strict=True)
        )
        variance = statistics.epistemic_variance()
        return LatentPrediction(
            vector,
            tuple(value + distance**2 for value in variance),
            distance,
        )

    def action_evidence(self) -> dict[str, dict[str, Any]]:
        """Expose samples and local model uncertainty without hiding evidence."""
        evidence = {}
        for action, contexts in sorted(self._actions.items()):
            sample_count = sum(context.samples for context in contexts)
            variances = [
                sum(context.epistemic_variance()) / FEATURE_DIMENSIONS
                for context in contexts
                if context.mean_delta
            ]
            evidence[action] = {
                "samples": sample_count,
                "contexts": len(contexts),
                "epistemic_variance_mean": (
                    sum(variances) / len(variances) if variances else 1.0
                ),
            }
        return evidence

    def to_dict(self) -> dict[str, Any]:
        """Persist empirical transition statistics with game memory."""
        return {
            action: [
                {
                    "reference_state": list(stats.reference_state),
                    "samples": stats.samples,
                    "mean_delta": list(stats.mean_delta),
                    "squared_deviation": list(stats.squared_deviation),
                }
                for stats in contexts
            ]
            for action, contexts in self._actions.items()
        }


def _vector_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Euclidean distance for equal-dimensional feature coordinates."""
    if len(first) != len(second):
        raise ValueError("latent dimensions must agree")
    return sqrt(sum((a - b) ** 2 for a, b in zip(first, second, strict=True)))


__all__ = [
    "ArcEnvironmentDecoder",
    "FEATURE_DIMENSIONS",
    "GoalEvidence",
    "GridState",
    "LatentState",
    "SceneEdge",
    "SceneGraph",
    "SceneNode",
    "encode_graph",
]

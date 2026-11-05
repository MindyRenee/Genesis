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

from latent_spaces.boundaries import LatentDomain, LatentState as DomainLatentState
from parietal_lobe.grid import Grid
from parietal_lobe.scene import perceive

ARC_COLORS = 16
OBJECT_FEATURES = 8
FEATURE_DIMENSIONS = 19 + ARC_COLORS + 2 * OBJECT_FEATURES


@dataclass(frozen=True, slots=True)
class GridState:
    """Validated ARC grid using row-major integer cells."""

    width: int
    height: int
    data: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", tuple(self.data))
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
        ):
            raise ValueError("grid dimensions must be positive integers")
        if len(self.data) != self.width * self.height:
            raise ValueError("grid data length does not match dimensions")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < ARC_COLORS
            for value in self.data
        ):
            raise ValueError("ARC cells must be integer palette values in [0, 15]")

    @classmethod
    def from_rows(cls, rows: list[list[int]]) -> GridState:
        """Validate and flatten a rectangular board."""
        if (
            not isinstance(rows, (list, tuple))
            or not rows
            or not isinstance(rows[0], (list, tuple))
            or not rows[0]
        ):
            raise ValueError("grid must be non-empty")
        width = len(rows[0])
        if any(not isinstance(row, (list, tuple)) or len(row) != width for row in rows):
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

    @property
    def betti_0(self) -> int:
        """Total foreground component count in this scene graph."""
        return sum(node.betti_0 for node in self.nodes)

    @property
    def betti_1(self) -> int:
        """Total enclosed-hole count under the declared digital topology."""
        return sum(node.betti_1 for node in self.nodes)

    @classmethod
    def extract(cls, state: GridState, background: int | None = None) -> SceneGraph:
        """Use Genesis's scene parser and infer background by its canonical rule."""
        if background is not None and (
            isinstance(background, bool)
            or not isinstance(background, int)
            or not 0 <= background < ARC_COLORS
        ):
            raise ValueError("background must be an ARC palette value in [0, 15]")
        grid = state.as_grid()
        scene = perceive(grid, background=background, compute_relations=False)
        background = scene.background
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
                    betti_0=compute_local_betti(cells, top, left, bottom, right)[0],
                    betti_1=compute_local_betti(cells, top, left, bottom, right)[1],
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


def compute_local_betti(
    cells: frozenset[tuple[int, int]], top: int, left: int, bottom: int, right: int
) -> tuple[int, int]:
    """Return local (beta_0, beta_1) using 4/8 digital-topology duality.

    beta_0 counts four-connected foreground components. beta_1 counts
    eight-connected complement components enclosed by the foreground.
    The bounding box is the local domain; its boundary is treated as open.
    """
    if not cells:
        return 0, 0

    foreground_seen: set[tuple[int, int]] = set()
    beta_0 = 0
    for start in cells:
        if start in foreground_seen:
            continue
        beta_0 += 1
        queue = deque([start])
        foreground_seen.add(start)
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + dr, col + dc)
                if neighbor in cells and neighbor not in foreground_seen:
                    foreground_seen.add(neighbor)
                    queue.append(neighbor)

    complement_seen: set[tuple[int, int]] = set()
    beta_1 = 0
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            start = (row, col)
            if start in cells or start in complement_seen:
                continue
            queue = deque([start])
            complement_seen.add(start)
            reaches_boundary = False
            while queue:
                current_row, current_col = queue.popleft()
                if current_row in (top, bottom) or current_col in (left, right):
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
                            and neighbor not in complement_seen
                        ):
                            complement_seen.add(neighbor)
                            queue.append(neighbor)
            if not reaches_boundary:
                beta_1 += 1
    return beta_0, beta_1


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
    """Fixed-width structural features and diagonal dynamics uncertainty.

    The feature vector is deterministically extracted from the observed grid.
    Variance describes uncertainty in predicted transitions, not confidence
    that the observed pixels or object coordinates were detected correctly.
    """

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

    @classmethod
    def from_graph(cls, graph: SceneGraph) -> LatentState:
        """Construct a latent state from an attributed scene graph."""
        return encode_graph(graph)

    def distance(self, other: LatentState) -> float:
        """Euclidean distance in this fixed feature coordinate system."""
        if len(self.vector) != len(other.vector):
            raise ValueError("latent dimensions must agree")
        return sqrt(sum((a - b) ** 2 for a, b in zip(self.vector, other.vector, strict=True)))

    def geodesic_distance(
        self,
        other: LatentState,
        metric: tuple[tuple[float, ...], ...] | None = None,
    ) -> float:
        """Distance under a constant positive-definite Riemannian metric.

        With a constant metric tensor, the minimizing geodesic is the
        straight segment and its length is sqrt((y-x)^T G (y-x)).
        A varying metric requires an actual metric field and is not implied.
        """
        if len(self.vector) != len(other.vector):
            raise ValueError("latent dimensions must agree")
        delta = tuple(b - a for a, b in zip(self.vector, other.vector, strict=True))
        if metric is None:
            return self.distance(other)
        _validate_spd_metric(metric, len(delta))
        squared = sum(
            delta[i] * metric[i][j] * delta[j]
            for i in range(len(delta))
            for j in range(len(delta))
        )
        return sqrt(max(0.0, squared)))

    def to_experiential_latent(self) -> DomainLatentState:
        """Project the board features into Genesis's typed outer E domain."""
        return DomainLatentState(
            domain=LatentDomain.E,
            values=self.vector,
            category="arc_observation",
            confidence=1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize numeric features and bounded object/edge evidence."""
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
                    "shape_signature": node.shape_signature,
                }
                for node in self.graph.nodes[:128]
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "dx": edge.dx,
                    "dy": edge.dy,
                    "adjacency": edge.adjacency,
                    "same_color": edge.same_color,
                    "same_shape": edge.same_shape,
                }
                for edge in self.graph.edges[:256]
            ],
        }


def _validate_spd_metric(
    metric: tuple[tuple[float, ...], ...], dimensions: int
) -> None:
    """Validate symmetry and positive definiteness by Cholesky factorization."""
    if len(metric) != dimensions or any(len(row) != dimensions for row in metric):
        raise ValueError("metric dimensions must match the latent state")
    if not all(isfinite(value) for row in metric for value in row):
        raise ValueError("metric entries must be finite")
    scale = max(1.0, max(abs(value) for row in metric for value in row))
    tolerance = 1e-10 * scale
    for row in range(dimensions):
        for column in range(row):
            if abs(metric[row][column] - metric[column][row]) > tolerance:
                raise ValueError("metric tensor must be symmetric")
    lower = [[0.0] * dimensions for _ in range(dimensions)]
    for row in range(dimensions):
        for column in range(row + 1):
            residual = metric[row][column] - sum(
                lower[row][k] * lower[column][k] for k in range(column)
            )
            if row == column:
                if residual <= tolerance:
                    raise ValueError("metric tensor must be positive definite")
                lower[row][column] = sqrt(residual)
            else:
                lower[row][column] = residual / lower[column][column]


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
    object_sizes = sorted(
        (node.pixel_count / area for node in nodes), reverse=True
    )[:OBJECT_FEATURES]
    radii = sorted(
        (
            hypot(node.centroid[0] - center_x, node.centroid[1] - center_y)
            / hypot(width, height)
            for node in nodes
        ),
        reverse=True,
    )[:OBJECT_FEATURES] if nodes else []
    vector.extend(object_sizes + [0.0] * (OBJECT_FEATURES - len(object_sizes)))
    vector.extend(radii + [0.0] * (OBJECT_FEATURES - len(radii)))
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

    def distance_to(self, observed: LatentState) -> float:
        """Prediction error against a subsequently observed latent state."""
        return _vector_distance(self.vector, observed.vector)


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



@dataclass(frozen=True, slots=True)
class GoalAttractorManifold:
    """A metric neighborhood around an explicitly supplied latent goal.

    The center must come from an observed or explicitly specified target.
    No goal vector is inferred from reward or terminal status.
    """

    center: tuple[float, ...]
    metric: tuple[tuple[float, ...], ...]
    tolerance: float

    def __post_init__(self) -> None:
        if not self.center or not all(isfinite(value) for value in self.center):
            raise ValueError("goal center must contain finite coordinates")
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("goal tolerance must be finite and non-negative")
        _validate_spd_metric(self.metric, len(self.center))

    def distance(self, state: LatentState) -> float:
        """Return the constant-metric distance from a state to this goal."""
        if len(state.vector) != len(self.center):
            raise ValueError("goal and state dimensions must agree")
        delta = tuple(
            value - center
            for value, center in zip(state.vector, self.center, strict=True)
        )
        squared = sum(
            delta[i] * self.metric[i][j] * delta[j]
            for i in range(len(delta))
            for j in range(len(delta))
        )
        return sqrt(max(0.0, squared))

    def is_satisfied(self, state: LatentState) -> bool:
        """Whether an observed state lies inside the specified goal neighborhood."""
        return self.distance(state) <= self.tolerance


@dataclass(frozen=True, slots=True)
class TransitionOperator:
    """An explicitly calibrated affine vector field x -> A x + b.

    Discrete ARC action identifiers are not automatically interpreted as
    differentiable operators. An operator must be supplied in a declared
    coordinate chart before differential operations are meaningful.
    """

    matrix: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        dimensions = len(self.bias)
        if dimensions == 0 or len(self.matrix) != dimensions:
            raise ValueError("operator output dimensions must match its bias")
        if any(len(row) != dimensions for row in self.matrix):
            raise ValueError("operator matrix must be square")
        if not all(isfinite(value) for row in self.matrix for value in row):
            raise ValueError("operator matrix must be finite")
        if not all(isfinite(value) for value in self.bias):
            raise ValueError("operator bias must be finite")

    @property
    def dimensions(self) -> int:
        return len(self.bias)

    def apply(self, state: tuple[float, ...]) -> tuple[float, ...]:
        """Evaluate the affine vector field at one state."""
        if len(state) != self.dimensions:
            raise ValueError("operator and state dimensions must agree")
        if not all(isfinite(value) for value in state):
            raise ValueError("operator input must be finite")
        return tuple(
            sum(weight * value for weight, value in zip(row, state, strict=True)) + offset
            for row, offset in zip(self.matrix, self.bias, strict=True)
        )

    def lie_bracket(self, other: TransitionOperator) -> TransitionOperator:
        """Return [self, other] = D(other) self - D(self) other.

        For f(x)=Ax+a and g(x)=Bx+b, this is
        ((BA-AB)x + (Ba-Ab)).
        """
        if self.dimensions != other.dimensions:
            raise ValueError("operator dimensions must agree")
        n = self.dimensions
        matrix = tuple(
            tuple(
                sum(
                    other.matrix[i][k] * self.matrix[k][j]
                    - self.matrix[i][k] * other.matrix[k][j]
                    for k in range(n)
                )
                for j in range(n)
            )
            for i in range(n)
        )
        bias = tuple(
            sum(other.matrix[i][k] * self.bias[k] for k in range(n))
            - sum(self.matrix[i][k] * other.bias[k] for k in range(n))
            for i in range(n)
        )
        return TransitionOperator(matrix, bias)


class DifferentiableOperatorSpace:
    """Finite-dimensional space of explicitly defined affine vector fields."""

    def __init__(
        self, operators: dict[str, TransitionOperator] | None = None
    ) -> None:
        self.operators = dict(operators or {})
        dimensions = {operator.dimensions for operator in self.operators.values()}
        if len(dimensions) > 1:
            raise ValueError("all operators must use the same coordinate dimension")
        if any(not name.strip() for name in self.operators):
            raise ValueError("operator names must be non-empty")

    def apply(self, name: str, state: tuple[float, ...]) -> tuple[float, ...]:
        """Apply a named operator in its declared coordinate chart."""
        try:
            operator = self.operators[name]
        except KeyError as error:
            raise KeyError(f"unknown transition operator: {name}") from error
        return operator.apply(state)

    def bracket(self, first: str, second: str) -> TransitionOperator:
        """Compute the Lie bracket of two declared vector fields."""
        try:
            left = self.operators[first]
            right = self.operators[second]
        except KeyError as error:
            raise KeyError(f"unknown transition operator: {error.args[0]}") from error
        return left.lie_bracket(right)

    def validate_lie_bracket_closure(self, tolerance: float = 1e-9) -> bool:
        """Check that every pairwise bracket belongs to the declared linear span.

        Coefficients are flattened into one finite-dimensional vector space.
        Modified Gram-Schmidt tests span membership without treating noisy
        discrete game actions as a Lie algebra.
        """
        if not isfinite(tolerance) or tolerance <= 0:
            raise ValueError("closure tolerance must be finite and positive")
        operators = list(self.operators.values())
        if not operators:
            return True
        basis: list[list[float]] = []
        for operator in operators:
            _extend_operator_basis(basis, _operator_coordinates(operator), tolerance)
        for index, first in enumerate(operators):
            for second in operators[index + 1 :]:
                bracket = _operator_coordinates(first.lie_bracket(second))
                residual = _operator_residual(bracket, basis)
                norm = sqrt(sum(value * value for value in bracket))
                residual_norm = sqrt(sum(value * value for value in residual))
                if residual_norm > tolerance * max(1.0, norm):
                    return False
        return True


def _operator_coordinates(operator: TransitionOperator) -> list[float]:
    return [value for row in operator.matrix for value in row] + list(operator.bias)


def _operator_residual(
    vector: list[float], basis: list[list[float]]
) -> list[float]:
    residual = vector[:]
    for direction in basis:
        projection = sum(
            value * axis for value, axis in zip(residual, direction, strict=True)
        )
        residual = [
            value - projection * axis
            for value, axis in zip(residual, direction, strict=True)
        ]
    return residual


def _extend_operator_basis(
    basis: list[list[float]], vector: list[float], tolerance: float
) -> None:
    residual = _operator_residual(vector, basis)
    norm = sqrt(sum(value * value for value in residual))
    if norm > tolerance:
        basis.append([value / norm for value in residual])


def validate_lie_bracket_closure(
    operators: DifferentiableOperatorSpace, tolerance: float = 1e-9
) -> bool:
    """Public closure check for an explicitly declared operator space."""
    return operators.validate_lie_bracket_closure(tolerance)


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
                    try:
                        reference = tuple(
                            float(value) for value in values.get("reference_state", [])
                        )
                        samples = int(values.get("samples", 0))
                        mean = [float(value) for value in values.get("mean_delta", [])]
                        m2 = [float(value) for value in values.get("squared_deviation", [])]
                    except (TypeError, ValueError, OverflowError):
                        continue
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
        if statistics is None or distance > self.CONTEXT_RADIUS:
            if len(contexts) >= self.MAX_CONTEXTS_PER_ACTION:
                return
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
    "OBJECT_FEATURES",
    "GoalEvidence",
    "GoalAttractorManifold",
    "GridState",
    "LatentState",
    "LatentPrediction",
    "SceneEdge",
    "SceneGraph",
    "SceneNode",
    "TransitionOperator",
    "DifferentiableOperatorSpace",
    "compute_local_betti",
    "encode_graph",
    "validate_lie_bracket_closure",
]

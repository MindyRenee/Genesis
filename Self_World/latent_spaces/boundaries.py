"""Typed boundaries and explicit flows between Genesis latent-space domains.

E = mc² is treated here as a reminder that transfer has a cost/budget, not
as a physical law for semantic vectors. Latent activations are not mass and
must not be assigned physical joules without a calibrated physical model.

Each domain retains its own category and coordinate dimension. A flow can
cross a boundary only through a registered, dimension-checked linear map.
The graph enables exchange without collapsing E, I, S, and L into one space.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class LatentDomain(StrEnum):
    """Distinct latent-space categories."""

    E = "experiential_outer"
    I = "interoceptive_inner"  # noqa: E741 — intentional domain symbol
    S = "self_identity"
    L = "language"


@dataclass(frozen=True, slots=True)
class LatentState:
    """A vector owned by one domain; coordinates are never implicitly shared."""

    domain: LatentDomain
    values: tuple[float, ...]
    category: str
    confidence: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("latent state must contain at least one coordinate")
        if not all(math.isfinite(v) for v in self.values):
            raise ValueError("latent coordinates must be finite")
        if not self.category.strip():
            raise ValueError("category must be non-empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")


@dataclass(frozen=True, slots=True)
class Boundary:
    """An explicit typed edge between two domain-specific coordinate systems.

    matrix rows equal target dimensions; each row's columns equal source
    dimensions. This map may change dimensions and need not preserve distance.
    """

    name: str
    source: LatentDomain
    target: LatentDomain
    matrix: tuple[tuple[float, ...], ...]
    allowed_categories: frozenset[str] = frozenset()
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("boundary name must be non-empty")
        if self.source is self.target:
            raise ValueError("cross-space boundaries must connect distinct domains")
        if not self.matrix or not self.matrix[0]:
            raise ValueError("boundary matrix must be non-empty")
        width = len(self.matrix[0])
        if any(len(row) != width for row in self.matrix):
            raise ValueError("boundary matrix must be rectangular")
        if not all(math.isfinite(v) for row in self.matrix for v in row):
            raise ValueError("boundary matrix values must be finite")
        if not math.isfinite(self.gain) or self.gain < 0.0:
            raise ValueError("gain must be finite and non-negative")
        if any(not category.strip() for category in self.allowed_categories):
            raise ValueError("allowed categories must be non-empty strings")

    @property
    def input_dimensions(self) -> int:
        return len(self.matrix[0])

    @property
    def output_dimensions(self) -> int:
        return len(self.matrix)


@dataclass(frozen=True, slots=True)
class FlowRecord:
    """Auditable record of one boundary crossing."""

    boundary: str
    source: LatentDomain
    target: LatentDomain
    category: str
    input_dimensions: int
    output_dimensions: int
    input_norm: float
    output_norm: float
    confidence: float
    timestamp: float


class LatentSpaceGraph:
    """Connected domain graph with explicit, category-aware boundaries.

    There is no implicit all-to-all mixing: every permitted crossing needs a
    named boundary and a matrix. Multiple boundaries can connect the same
    pair for different categories or functions.
    """

    def __init__(self, boundaries: Iterable[Boundary] = ()) -> None:
        self._boundaries: dict[str, Boundary] = {}
        self._history: list[FlowRecord] = []
        for boundary in boundaries:
            self.add_boundary(boundary)

    @property
    def boundaries(self) -> tuple[Boundary, ...]:
        return tuple(self._boundaries.values())

    @property
    def history(self) -> tuple[FlowRecord, ...]:
        return tuple(self._history)

    def add_boundary(self, boundary: Boundary) -> None:
        """Register a unique edge; reject duplicate names rather than overwrite."""
        if boundary.name in self._boundaries:
            raise ValueError(f"boundary already exists: {boundary.name}")
        self._boundaries[boundary.name] = boundary

    def flow(self, boundary_name: str, state: LatentState) -> LatentState:
        """Transform a state across one named boundary while retaining its label."""
        try:
            boundary = self._boundaries[boundary_name]
        except KeyError as exc:
            raise KeyError(f"unknown boundary: {boundary_name}") from exc

        if state.domain is not boundary.source:
            raise ValueError(
                f"{boundary.name} accepts {boundary.source.value}, not {state.domain.value}"
            )
        if len(state.values) != boundary.input_dimensions:
            raise ValueError(
                f"{boundary.name} expects {boundary.input_dimensions} input "
                f"coordinates, received {len(state.values)}"
            )
        if boundary.allowed_categories and state.category not in boundary.allowed_categories:
            raise ValueError(f"category {state.category!r} is not permitted across {boundary.name}")

        output = tuple(
            boundary.gain
            * sum(weight * value for weight, value in zip(row, state.values, strict=True))
            for row in boundary.matrix
        )
        if not all(math.isfinite(v) for v in output):
            raise ValueError("boundary produced non-finite coordinates")

        result = LatentState(
            domain=boundary.target,
            values=output,
            category=state.category,
            confidence=state.confidence,
            timestamp=state.timestamp,
        )
        self._history.append(
            FlowRecord(
                boundary=boundary.name,
                source=state.domain,
                target=boundary.target,
                category=state.category,
                input_dimensions=len(state.values),
                output_dimensions=len(output),
                input_norm=math.sqrt(sum(v * v for v in state.values)),
                output_norm=math.sqrt(sum(v * v for v in output)),
                confidence=state.confidence,
                timestamp=state.timestamp,
            )
        )
        return result

    def reachable(self, source: LatentDomain, target: LatentDomain) -> bool:
        """Return whether registered directed boundaries connect two domains."""
        if source is target:
            return True
        seen = {source}
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for edge in self._boundaries.values():
                if edge.source is current and edge.target not in seen:
                    if edge.target is target:
                        return True
                    seen.add(edge.target)
                    frontier.append(edge.target)
        return False

    def topology(self) -> dict[str, object]:
        """Return domain/edge diagnostics without exposing latent values."""
        return {
            "domains": tuple(domain.value for domain in LatentDomain),
            "boundary_count": len(self._boundaries),
            "boundaries": tuple(
                {
                    "name": edge.name,
                    "source": edge.source.value,
                    "target": edge.target.value,
                    "input_dimensions": edge.input_dimensions,
                    "output_dimensions": edge.output_dimensions,
                    "categories": tuple(sorted(edge.allowed_categories)),
                }
                for edge in self._boundaries.values()
            ),
            "flow_count": len(self._history),
        }


def energy_from_mass(mass_kg: float, *, c_m_per_s: float = 299_792_458.0) -> float:
    """Return physical rest energy E=mc² in joules for a physical mass.

    This helper is deliberately separate from latent-space flow. Do not pass
    semantic activation magnitudes as mass; they have no such calibrated unit.
    """
    if not math.isfinite(mass_kg) or mass_kg < 0.0:
        raise ValueError("mass_kg must be finite and non-negative")
    if not math.isfinite(c_m_per_s) or c_m_per_s <= 0.0:
        raise ValueError("c_m_per_s must be finite and positive")
    return mass_kg * c_m_per_s**2

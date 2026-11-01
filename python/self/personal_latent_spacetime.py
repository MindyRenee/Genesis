"""Personal latent spacetime for Genesis's self, inner, and outer models.

These are three distinct representational geometries, not physical spacetime
coordinates. Each domain has its own vector dimension and positive-definite
metric. Cross-domain comparisons require an explicit mapping and are never
inferred by treating vectors as if they shared a coordinate frame.

The light-cone analogue is a bounded propagation relation in latent distance
per unit model-time. Its speed is a computational parameter, not the physical
speed of light. The self event is the conceptual apex for perspective and
temporal indexing; this does not imply that inner and outer states are located
in the self-space's coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]


class LatentDomain(StrEnum):
    SELF = "self"
    INNER = "inner"
    OUTER = "outer"


@dataclass(frozen=True, slots=True)
class LatentEvent:
    event_id: str
    time_s: float
    state: tuple[float, ...]
    domain: LatentDomain
    confidence: float = 1.0
    causes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LatentConeRelation:
    event_id: str
    domain: LatentDomain
    delta_time_s: float
    latent_distance: float
    reachable: bool
    temporal_relation: str


@dataclass(frozen=True, slots=True)
class LatentSpaceSnapshot:
    domain: LatentDomain
    dimension: int
    event_count: int
    latest_event: LatentEvent | None
    metric_trace: float


class PersonalLatentSpacetime:
    """Maintain independent latent metrics and event histories per domain."""

    def __init__(
        self,
        dimensions: Mapping[LatentDomain, int] | None = None,
        propagation_rates: Mapping[LatentDomain, float] | None = None,
        metrics: Mapping[LatentDomain, FloatMatrix] | None = None,
    ) -> None:
        dims = dimensions or {
            LatentDomain.SELF: 32,
            LatentDomain.INNER: 64,
            LatentDomain.OUTER: 64,
        }
        rates = propagation_rates or {
            LatentDomain.SELF: 1.0,
            LatentDomain.INNER: 1.0,
            LatentDomain.OUTER: 1.0,
        }
        self._dimensions: dict[LatentDomain, int] = {}
        self._rates: dict[LatentDomain, float] = {}
        self._metrics: dict[LatentDomain, FloatMatrix] = {}
        self._events: dict[LatentDomain, list[LatentEvent]] = {
            domain: [] for domain in LatentDomain
        }
        self._event_ids: set[str] = set()

        for domain in LatentDomain:
            dimension = dims.get(domain)
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
            ):
                raise ValueError(f"{domain.value} dimension must be a positive integer")
            rate = float(rates.get(domain, math.nan))
            if not math.isfinite(rate) or rate <= 0.0:
                raise ValueError(f"{domain.value} propagation rate must be finite and positive")
            self._dimensions[domain] = dimension
            self._rates[domain] = rate
            candidate = (
                np.asarray(metrics[domain], dtype=np.float64)
                if metrics is not None and domain in metrics
                else np.eye(dimension, dtype=np.float64)
            )
            if candidate.shape != (dimension, dimension):
                raise ValueError(f"{domain.value} metric shape must be {(dimension, dimension)}")
            if not np.isfinite(candidate).all():
                raise ValueError(f"{domain.value} metric must be finite")
            if not np.allclose(candidate, candidate.T, atol=1e-10, rtol=1e-10):
                raise ValueError(f"{domain.value} metric must be symmetric")
            if np.linalg.eigvalsh(candidate).min() <= 0.0:
                raise ValueError(f"{domain.value} metric must be positive definite")
            self._metrics[domain] = candidate.copy()

    def record(
        self,
        domain: LatentDomain,
        state: tuple[float, ...] | list[float] | FloatVector,
        *,
        time_s: float,
        event_id: str,
        confidence: float = 1.0,
        causes: tuple[str, ...] = (),
    ) -> LatentEvent:
        """Record a state in its own domain without projecting it into others."""
        vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if vector.size != self._dimensions[domain]:
            raise ValueError(
                f"{domain.value} state needs {self._dimensions[domain]} coordinates"
            )
        if not np.isfinite(vector).all():
            raise ValueError("latent state must contain only finite values")
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not event_id or event_id in self._event_ids:
            raise ValueError("event_id must be non-empty and globally unique")
        event = LatentEvent(
            event_id=event_id,
            time_s=float(time_s),
            state=tuple(float(value) for value in vector),
            domain=domain,
            confidence=float(confidence),
            causes=tuple(causes),
        )
        self._events[domain].append(event)
        self._event_ids.add(event_id)
        return event

    def self_apex(self) -> LatentEvent | None:
        """Return the latest self event used as the current perspective apex."""
        events = self._events[LatentDomain.SELF]
        return events[-1] if events else None

    def cone(
        self,
        apex: LatentEvent,
        domain: LatentDomain,
    ) -> tuple[LatentConeRelation, ...]:
        """Relate domain events to an apex only when their domains match.

        This method deliberately refuses cross-domain distance calculations.
        A future cross-domain relation must use an explicitly supplied mapping.
        """
        if apex.domain is not domain:
            raise ValueError(
                "cross-domain cone requires an explicit latent-space mapping"
            )
        metric = self._metrics[domain]
        apex_vector = np.asarray(apex.state, dtype=np.float64)
        results: list[LatentConeRelation] = []
        for event in self._events[domain]:
            delta_time = event.time_s - apex.time_s
            vector = np.asarray(event.state, dtype=np.float64) - apex_vector
            distance = float(np.sqrt(max(0.0, vector @ metric @ vector)))
            reachable = distance <= self._rates[domain] * abs(delta_time)
            relation = (
                "coincident"
                if abs(delta_time) <= 1e-12 and distance <= 1e-12
                else "past"
                if delta_time < 0.0
                else "future"
                if delta_time > 0.0
                else "simultaneous"
            )
            results.append(
                LatentConeRelation(
                    event_id=event.event_id,
                    domain=domain,
                    delta_time_s=float(delta_time),
                    latent_distance=distance,
                    reachable=bool(reachable),
                    temporal_relation=relation,
                )
            )
        return tuple(results)

    def snapshot(self) -> tuple[LatentSpaceSnapshot, ...]:
        """Return independent diagnostics for all three personal spaces."""
        snapshots = []
        for domain in LatentDomain:
            events = self._events[domain]
            snapshots.append(
                LatentSpaceSnapshot(
                    domain=domain,
                    dimension=self._dimensions[domain],
                    event_count=len(events),
                    latest_event=events[-1] if events else None,
                    metric_trace=float(np.trace(self._metrics[domain])),
                )
            )
        return tuple(snapshots)

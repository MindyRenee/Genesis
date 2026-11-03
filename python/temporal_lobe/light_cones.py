"""Spacetime-inspired causal geometry for Genesis's self, inner, and outer worlds.

This is a computational model, not a claim that cognitive states are physical
relativistic events. Coordinates and timestamps must use consistent units.

The existing prototype is preserved: events occupy SELF, INNER, or OUTER
domains and are classified against one shared SELF apex. The double-cone
coordinate helper is a separate geometric view: the time axis is straight,
the apex is the origin, and the two cone sheets counter-rotate. It does not
alter causal classification or pretend that a visualization proves cognition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class WorldDomain(StrEnum):
    """The three mutually distinguishable but linked state domains."""

    SELF = "self"
    INNER = "inner"
    OUTER = "outer"


class IntervalKind(StrEnum):
    """Minkowski separation class relative to an apex event."""

    TIMELIKE = "timelike"
    LIGHTLIKE = "lightlike"
    SPACELIKE = "spacelike"
    COINCIDENT = "coincident"


@dataclass(frozen=True, slots=True)
class SpacetimeEvent:
    """One timestamped event in a declared coordinate system.

    position is a 3-vector in one spatial unit. time_s is seconds on one
    monotonic time base; do not mix wall-clock and monotonic timestamps.
    """

    event_id: str
    time_s: float
    position: tuple[float, float, float]
    domain: WorldDomain
    confidence: float = 1.0
    causes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not math.isfinite(self.time_s):
            raise ValueError("time_s must be finite")
        if len(self.position) != 3 or not all(math.isfinite(v) for v in self.position):
            raise ValueError("position must contain three finite coordinates")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ConeRelation:
    """Causal/geometric relation of an event to the current apex."""

    event_id: str
    domain: WorldDomain
    interval_minkowski: float
    kind: IntervalKind
    in_past_cone: bool
    in_future_cone: bool
    spatial_distance: float
    time_delta_s: float


@dataclass(frozen=True, slots=True)
class ConeCoordinate:
    """A point on a counter-rotating double-cone view.

    x is the displayed time-axis coordinate, horizontal through the origin.
    y and z are transverse coordinates around that axis.
    """

    x: float
    y: float
    z: float
    sheet: str


def double_cone_coordinate(
    time_coordinate: float,
    angle: float,
    *,
    angular_speed: float = 0.0,
    radial_scale: float = 1.0,
) -> ConeCoordinate:
    """Map a parameter to a double-cone sheet with opposite rotation.

    Time runs horizontally along x through the shared origin. Positive time
    occupies the upper sheet (counterclockwise around +x); negative time
    occupies the lower sheet (clockwise). Radius grows linearly with distance
    from the apex.
    """
    values = (time_coordinate, angle, angular_speed, radial_scale)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("cone coordinates and parameters must be finite")
    if radial_scale <= 0.0:
        raise ValueError("radial_scale must be positive")
    if time_coordinate == 0.0:
        return ConeCoordinate(0.0, 0.0, 0.0, "apex")

    sign = 1.0 if time_coordinate > 0.0 else -1.0
    radius = radial_scale * abs(time_coordinate)
    # Upper sheet advances CCW; lower sheet advances CW.
    phase = angle + sign * angular_speed * abs(time_coordinate)
    return ConeCoordinate(
        x=time_coordinate,
        y=radius * math.cos(phase),
        z=radius * math.sin(phase),
        sheet="upper" if sign > 0 else "lower",
    )


class LightConeModel:
    """Classify events relative to a shared current-state apex.

    The interval convention is s² = (v Δt)² - |Δx|²:
    positive is timelike, zero lightlike, negative spacelike. Time ordering
    determines past versus future; temporal order alone does not prove cause.
    """

    def __init__(self, signal_speed: float = 1.0, tolerance: float = 1e-9) -> None:
        if not math.isfinite(signal_speed) or signal_speed <= 0.0:
            raise ValueError("signal_speed must be finite and positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        self.signal_speed = signal_speed
        self.tolerance = tolerance
        self._apex: SpacetimeEvent | None = None
        self._events: dict[str, SpacetimeEvent] = {}

    @property
    def apex(self) -> SpacetimeEvent | None:
        """The latest integration event shared by all three domains."""
        return self._apex

    def advance_apex(self, event: SpacetimeEvent) -> None:
        """Advance the apex monotonically; equal timestamps are allowed."""
        if event.domain is not WorldDomain.SELF:
            raise ValueError("the shared apex must be a self-domain event")
        if self._apex is not None and event.time_s < self._apex.time_s:
            raise ValueError("apex time cannot move backwards")
        self._apex = event
        self._events[event.event_id] = event

    def add_event(self, event: SpacetimeEvent) -> None:
        """Register or replace an event without implicitly advancing the apex."""
        self._events[event.event_id] = event

    def get_event(self, event_id: str) -> SpacetimeEvent | None:
        return self._events.get(event_id)

    def classify(self, event: SpacetimeEvent, apex: SpacetimeEvent | None = None) -> ConeRelation:
        """Classify one event relative to the supplied or current apex."""
        origin = apex or self._apex
        if origin is None:
            raise ValueError("advance_apex() must establish the current event first")
        dx = tuple(a - b for a, b in zip(event.position, origin.position, strict=True))
        distance = math.sqrt(sum(value * value for value in dx))
        dt = event.time_s - origin.time_s
        interval = (self.signal_speed * dt) ** 2 - distance**2
        scale = max(1.0, (self.signal_speed * dt) ** 2, distance**2)
        eps = self.tolerance * scale
        if abs(dt) <= self.tolerance and distance <= self.tolerance:
            kind = IntervalKind.COINCIDENT
        elif abs(interval) <= eps:
            kind = IntervalKind.LIGHTLIKE
        elif interval > 0.0:
            kind = IntervalKind.TIMELIKE
        else:
            kind = IntervalKind.SPACELIKE
        can_reach = distance <= self.signal_speed * abs(dt) + self.tolerance
        return ConeRelation(
            event_id=event.event_id,
            domain=event.domain,
            interval_minkowski=interval,
            kind=kind,
            in_past_cone=dt < -self.tolerance and can_reach,
            in_future_cone=dt > self.tolerance and can_reach,
            spatial_distance=distance,
            time_delta_s=dt,
        )

    def classify_all(
        self, events: Iterable[SpacetimeEvent] | None = None
    ) -> tuple[ConeRelation, ...]:
        """Return deterministic classifications for supplied or registered events."""
        source = self._events.values() if events is None else events
        return tuple(
            self.classify(event)
            for event in sorted(source, key=lambda item: (item.time_s, item.event_id))
        )

    def cone_for(self, domain: WorldDomain) -> tuple[ConeRelation, ...]:
        """Return one domain's events classified against the same self apex."""
        return tuple(relation for relation in self.classify_all() if relation.domain is domain)

    def state_summary(self) -> dict[str, object]:
        """Serializable diagnostics without raw sensor payloads."""
        return {
            "apex_id": None if self._apex is None else self._apex.event_id,
            "apex_time_s": None if self._apex is None else self._apex.time_s,
            "signal_speed": self.signal_speed,
            "event_count": len(self._events),
            "domains": {
                domain.value: sum(event.domain is domain for event in self._events.values())
                for domain in WorldDomain
            },
        }



@dataclass(frozen=True, slots=True)
class DiagramPoint:
    """A point in the visualization frame, not a physical spacetime event."""

    x: float
    y: float
    z: float
    component: str


def light_cone_coordinate(
    time_coordinate: float,
    angle: float,
    *,
    signal_speed: float = 1.0,
    angular_speed: float = 0.0,
) -> ConeCoordinate:
    """Return a point on a causal light-cone boundary in a horizontal-time view.

    The displayed time coordinate is x. The transverse radius is exactly
    signal_speed * abs(time_coordinate), so the Minkowski boundary
    (signal_speed * t)**2 - y**2 - z**2 == 0 is preserved by rotation.
    The two temporal sheets counter-rotate as measured outward from the apex.
    This is a diagram embedding; it does not change the causal metric.
    """
    if not math.isfinite(signal_speed) or signal_speed <= 0.0:
        raise ValueError("signal_speed must be finite and positive")
    return double_cone_coordinate(
        time_coordinate,
        angle,
        angular_speed=angular_speed,
        radial_scale=signal_speed,
    )


def spine_point(distance: float) -> DiagramPoint:
    """Return a point on the vertical identity spine through the shared apex."""
    if not math.isfinite(distance):
        raise ValueError("spine distance must be finite")
    return DiagramPoint(0.0, 0.0, distance, "spine")


def time_sunburst_ray(
    index: int,
    ray_count: int,
    radius: float,
) -> tuple[DiagramPoint, DiagramPoint]:
    """Return one radial sunburst ray in the horizontal plane through Self.

    Rays beyond the two horizontal directions are visual encodings of temporal
    scales or relationships, not additional physical time dimensions.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("ray index must be an integer")
    if isinstance(ray_count, bool) or not isinstance(ray_count, int) or ray_count < 2:
        raise ValueError("ray_count must be an integer of at least 2")
    if not 0 <= index < ray_count:
        raise ValueError("ray index must be in [0, ray_count)")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("sunburst radius must be finite and positive")
    angle = 2.0 * math.pi * index / ray_count
    endpoint = DiagramPoint(
        radius * math.cos(angle),
        radius * math.sin(angle),
        0.0,
        "time_sunburst",
    )
    return DiagramPoint(0.0, 0.0, 0.0, "self_apex"), endpoint


class ElasticPlasticHologram:
    """A small dependency-free adaptive complex field.

    Complex values encode amplitude and phase. Elastic response is temporary;
    plastic values are persistent. New input dimensions can be appended as
    needed. This is a computational representation, not physical holography
    and not an unlimited-memory guarantee.
    """

    def __init__(
        self,
        initial_state: Iterable[complex] = (),
        *,
        elastic_rate: float = 0.5,
        plastic_rate: float = 0.05,
    ) -> None:
        self._validate_rate("elastic_rate", elastic_rate)
        self._validate_rate("plastic_rate", plastic_rate)
        self.elastic_rate = elastic_rate
        self.plastic_rate = plastic_rate
        self._plastic = [complex(value) for value in initial_state]
        self._elastic = list(self._plastic)
        self._validate_field(self._plastic)

    @staticmethod
    def _validate_rate(name: str, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    @staticmethod
    def _validate_field(values: Iterable[complex]) -> None:
        for value in values:
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError("holographic field values must be finite")

    @property
    def plastic_state(self) -> tuple[complex, ...]:
        """Persistent learned field, exposed immutably."""
        return tuple(self._plastic)

    @property
    def elastic_state(self) -> tuple[complex, ...]:
        """Current temporarily deformed field, exposed immutably."""
        return tuple(self._elastic)

    def observe(self, values: Iterable[complex]) -> tuple[complex, ...]:
        """Apply an observation, returning elastic response and updating memory.

        The returned response is computed relative to the pre-update plastic
        state. Plastic consolidation then moves persistent values toward the
        observation. Input dimensionality can grow, but never shrinks silently.
        """
        observation = [complex(value) for value in values]
        if not observation:
            raise ValueError("observation must contain at least one value")
        self._validate_field(observation)

        size = max(len(self._plastic), len(observation))
        self._plastic.extend([0j] * (size - len(self._plastic)))
        self._elastic.extend(self._plastic[len(self._elastic):])
        observation.extend([0j] * (size - len(observation)))

        response = []
        for index, value in enumerate(observation):
            prior = self._plastic[index]
            temporary = prior + self.elastic_rate * (value - prior)
            persistent = prior + self.plastic_rate * (value - prior)
            response.append(temporary)
            self._plastic[index] = persistent
            self._elastic[index] = temporary
        return tuple(response)

    def relax(self, recovery_rate: float = 0.5) -> tuple[complex, ...]:
        """Relax temporary deformation toward learned state without unlearning."""
        self._validate_rate("recovery_rate", recovery_rate)
        self._elastic = [
            current + recovery_rate * (learned - current)
            for current, learned in zip(self._elastic, self._plastic, strict=True)
        ]
        return self.elastic_state

    def summary(self) -> dict[str, int | float]:
        """Return non-sensitive diagnostics without exposing field values."""
        return {
            "dimensions": len(self._plastic),
            "elastic_rate": self.elastic_rate,
            "plastic_rate": self.plastic_rate,
        }

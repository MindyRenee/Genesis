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

    x and y are transverse coordinates; z is the displayed time-axis
    coordinate. The central axis is straight and passes through the origin.
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

    Positive time is the upper sheet (counterclockwise as viewed from +z);
    negative time is the lower sheet (clockwise). Radius grows linearly with
    distance from the shared apex. The z/time axis remains straight.
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
        x=radius * math.cos(phase),
        y=radius * math.sin(phase),
        z=time_coordinate,
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

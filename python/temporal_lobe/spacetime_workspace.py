"""Shared causal event workspace for Genesis's self, inner, and outer domains.

All event coordinates must belong to one calibrated coordinate frame. This
module does not infer coordinates from latent vectors or mix pixel coordinates
with map coordinates. Subsystems submit events with explicit timestamps and
positions; the workspace keeps one self apex and returns per-domain relations.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from temporal_lobe.light_cones import (
    ConeRelation,
    LightConeModel,
    SpacetimeEvent,
    WorldDomain,
)


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """Causal views of one shared state at a single apex time."""

    apex: SpacetimeEvent
    self_view: tuple[ConeRelation, ...]
    inner_view: tuple[ConeRelation, ...]
    outer_view: tuple[ConeRelation, ...]


class SpacetimeWorkspace:
    """Coordinate three event streams without conflating their domains.

    signal_speed uses spatial-coordinate-units per second. The default is a
    normalized value and must not be interpreted as physical light speed.
    """

    def __init__(self, signal_speed: float = 1.0) -> None:
        self.geometry = LightConeModel(signal_speed=signal_speed)
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}:{self._counter}"

    def record_self(
        self,
        position: tuple[float, float, float],
        *,
        time_s: float | None = None,
        event_id: str | None = None,
        confidence: float = 1.0,
        causes: tuple[str, ...] = (),
    ) -> SpacetimeEvent:
        """Advance the shared apex with a self-state event."""
        event = SpacetimeEvent(
            event_id=event_id or self._next_id("self"),
            time_s=time.monotonic() if time_s is None else time_s,
            position=position,
            domain=WorldDomain.SELF,
            confidence=confidence,
            causes=causes,
        )
        self.geometry.advance_apex(event)
        return event

    def record(
        self,
        domain: WorldDomain,
        position: tuple[float, float, float],
        *,
        time_s: float,
        event_id: str | None = None,
        confidence: float = 1.0,
        causes: tuple[str, ...] = (),
    ) -> SpacetimeEvent:
        """Register an inner or outer event in the shared coordinate frame."""
        if domain is WorldDomain.SELF:
            raise ValueError("use record_self to advance the shared apex")
        event = SpacetimeEvent(
            event_id=event_id or self._next_id(domain.value),
            time_s=time_s,
            position=position,
            domain=domain,
            confidence=confidence,
            causes=causes,
        )
        self.geometry.add_event(event)
        return event

    def snapshot(self) -> WorkspaceSnapshot:
        """Compute all three domain views against the same current apex."""
        apex = self.geometry.apex
        if apex is None:
            raise ValueError("record_self must establish the shared apex first")
        return WorkspaceSnapshot(
            apex=apex,
            self_view=self.geometry.cone_for(WorldDomain.SELF),
            inner_view=self.geometry.cone_for(WorldDomain.INNER),
            outer_view=self.geometry.cone_for(WorldDomain.OUTER),
        )

    def register_events(self, events: Iterable[SpacetimeEvent]) -> None:
        """Bulk-register non-self events; self apex must be advanced explicitly."""
        for event in events:
            if event.domain is WorldDomain.SELF:
                raise ValueError("bulk registration cannot advance the self apex")
            self.geometry.add_event(event)

    def state_summary(self) -> dict[str, object]:
        """Expose concise diagnostics suitable for telemetry."""
        return self.geometry.state_summary()

"""Bridge visual two-view geometry into Genesis's shared causal workspace.

A monocular epipolar report provides relative motion and baseline-scaled depth,
not metric world position. Therefore this bridge requires an explicit calibrated
projection supplied by the caller. It refuses to invent map coordinates from
heading/depth values or to conflate camera pixels with allocentric coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from occipital_lobe.epipolar import TwoViewReport
from temporal_lobe.light_cones import SpacetimeEvent, WorldDomain
from temporal_lobe.spacetime_workspace import SpacetimeWorkspace


@dataclass(frozen=True, slots=True)
class VisualEventResult:
    """Outcome of converting a visual report to a causal workspace event."""

    event: SpacetimeEvent | None
    accepted: bool
    reason: str


CameraToWorkspace = Callable[[TwoViewReport], tuple[float, float, float]]


class VisualSpacetimeBridge:
    """Publish visual geometry only when a coordinate transform is calibrated."""

    def __init__(
        self,
        workspace: SpacetimeWorkspace,
        camera_to_workspace: CameraToWorkspace | None = None,
    ) -> None:
        self.workspace = workspace
        self.camera_to_workspace = camera_to_workspace
        self._sequence = 0

    def publish(
        self,
        report: TwoViewReport,
        *,
        time_s: float,
        event_id: str | None = None,
    ) -> VisualEventResult:
        """Publish one camera report as an outer-world event.

        time_s must be monotonic seconds from the same clock used by the
        workspace. The transform must return coordinates in the workspace's
        existing spatial frame and units. Missing calibration is a hard
        refusal, not a guessed position.
        """
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self.camera_to_workspace is None:
            return VisualEventResult(None, False, "calibrated camera-to-workspace transform required")
        if report.kind not in {"still", "turn", "travel", "unresolved"}:
            return VisualEventResult(None, False, "report has no classified two-view result")
        if report.kind == "unresolved":
            return VisualEventResult(None, False, "visual geometry is unresolved")
        position = self.camera_to_workspace(report)
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError("transform must return three finite workspace coordinates")
        self._sequence += 1
        confidence = max(0.0, min(1.0, report.inlier_ratio))
        event = self.workspace.record(
            WorldDomain.OUTER,
            tuple(float(v) for v in position),
            time_s=time_s,
            event_id=event_id or f"visual:{self._sequence}",
            confidence=confidence,
        )
        return VisualEventResult(event, True, "published")

"""Bridge visual two-view geometry into Genesis's shared causal workspace.

A monocular epipolar report provides relative motion and baseline-scaled depth,
not metric world position. Therefore this bridge requires an explicit calibrated
projection supplied by the caller. It refuses to invent map coordinates from
heading/depth values or to conflate camera pixels with allocentric coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from temporal_lobe.light_cones import SpacetimeEvent, WorldDomain
from temporal_lobe.spacetime_workspace import SpacetimeWorkspace

from occipital_lobe.epipolar import TwoViewReport


@dataclass(frozen=True, slots=True)
class VisualEventResult:
    """Outcome of converting a visual report to a causal workspace event."""

    event: SpacetimeEvent | None
    accepted: bool
    reason: str


CameraToWorkspace = Callable[[TwoViewReport], tuple[float, float, float]]


# Facing orientations as basis-change maps — the eye's image basis
# (right r, forward f, up u) expressed in the canonical egocentric
# frame (+X right, +Y forward, +Z up). Each facing is one point of
# the eye frame's orientation group, and every facing has its
# opposite:
#
#   without ↔ within   the mirror reflection across the vertical
#                      plane — the one improper map (det −1); what
#                      someone facing her sees
#   up      ↔ down     ±90° rotations about the right axis —
#                      Rx(+90°) tips the optical axis to +Z,
#                      Rx(−90°) to −Z
#
# canonical_position = R_facing · d_img
_FACING: dict[str, tuple[tuple[float, float, float], ...]] = {
    "without": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "within": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "up": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "down": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
}


def egocentric_projection(report: TwoViewReport) -> tuple[float, float, float]:
    """Project a two-view report into the canonical egocentric frame.

    The eye is the origin: +Y is gaze-forward, +X is right, +Z is up.
    ``bearing_deg`` (0° forward, 90° right) and ``elevation_deg`` give
    the direction in the look's own image frame; the report's
    ``facing`` then rotates that direction into the canonical frame
    through the basis-change map — a mirrored look un-flips, an
    upward look's forward lands overhead. Range is the report's
    baseline-scaled ``median_depth``, real only for ``travel`` —
    still and turn reports land on the unit sphere, where direction
    is honest and range carries no information.
    """
    bearing = math.radians(report.bearing_deg)
    elevation = math.radians(report.elevation_deg)
    depth = report.median_depth if report.kind == "travel" and report.median_depth > 0 else 1.0
    horizontal = depth * math.cos(elevation)
    # d_img in the look's image basis: (right, forward, up)
    d_img = (
        horizontal * math.sin(bearing),
        horizontal * math.cos(bearing),
        depth * math.sin(elevation),
    )
    r = _FACING.get(report.facing, _FACING["without"])
    return (
        r[0][0] * d_img[0] + r[0][1] * d_img[1] + r[0][2] * d_img[2],
        r[1][0] * d_img[0] + r[1][1] * d_img[1] + r[1][2] * d_img[2],
        r[2][0] * d_img[0] + r[2][1] * d_img[1] + r[2][2] * d_img[2],
    )


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
            return VisualEventResult(
                None, False, "calibrated camera-to-workspace transform required"
            )
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
            (float(position[0]), float(position[1]), float(position[2])),
            time_s=time_s,
            event_id=event_id or f"visual:{self._sequence}",
            confidence=confidence,
        )
        return VisualEventResult(event, True, "published")

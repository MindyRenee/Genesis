"""LIP — the topographic saliency map and attentional selection.

Lateral intraparietal area: combines bottom-up feature maps into one
spatial priority map, then selects the next attended location by
winner-take-all with inhibition-of-return. This replaces a scalar
scene-salience with *where* in the field attention should land.

Feature maps (Itti & Koch 1998, Koch & Ullman 1985):
    intensity   — local luminance contrast (centre-surround)
    color       — red-green and blue-yellow double-opponent contrast
    orientation — V1 feature strengths splatted by position
    motion      — MT/V5 flow magnitudes splatted by position

Each map is normalized by the Itti operator N(.) — a map with a few
strong peaks is amplified, a map that's uniformly active is
suppressed — then combined. Top-down goals bias the combination the
way frontal input biases LIP (biased competition).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..occipital_lobe.v1 import VisualField
    from .motion import MotionField

# The map runs at a fixed coarse resolution regardless of frame size —
# LIP is a priority map, not a pixel buffer.
_MAP_H, _MAP_W = 24, 32
# Centre-surround scales, in map cells.
_CENTER_R, _SURROUND_R = 1, 3
# Inhibition-of-return: how long a visited location stays suppressed
# and the radius of the suppression blob.
_IOR_SECONDS = 4.0
_IOR_RADIUS = 4.0  # map cells


@dataclass(slots=True)
class AttendedLocation:
    """A winner-take-all selection on the priority map."""

    y: float          # pixel-space row
    x: float          # pixel-space column
    map_y: int        # map-space row
    map_x: int        # map-space column
    salience: float   # priority value that won
    feature: str      # which feature map contributed most here


@dataclass(slots=True)
class SaliencyResult:
    """What the dorsal priority map made of one frame."""

    priority: np.ndarray            # (H, W) normalized priority map
    attended: AttendedLocation | None = None
    scene_salience: float = 0.0     # scalar summary (kept for callers)
    feature_weights: dict[str, float] = field(default_factory=dict)


def _normalize(map_: np.ndarray) -> np.ndarray:
    """Itti map normalization N(.): amplify maps with rare strong peaks."""
    peak = float(map_.max())
    if peak <= 0:
        return map_
    m = map_ / peak
    # Approximate the local-maxima count with a 3x3 max filter.
    padded = np.pad(m, 1, mode="edge")
    local_max = np.maximum.reduce(
        [padded[i:i + m.shape[0], j:j + m.shape[1]]
         for i in range(3) for j in range(3)]
    )
    n_maxima = int(((m == local_max) & (m > 0.1)).sum())
    return m * (1.0 + (1.0 - 1.0 / max(1, n_maxima)))


def _center_surround(map_: np.ndarray) -> np.ndarray:
    """Centre-minus-surround difference at fixed map scales."""
    padded = np.pad(map_, _SURROUND_R, mode="edge")
    out = np.zeros_like(map_)
    for dy in range(-_SURROUND_R, _SURROUND_R + 1):
        for dx in range(-_SURROUND_R, _SURROUND_R + 1):
            r2 = dy * dy + dx * dx
            if r2 <= _CENTER_R * _CENTER_R:
                w = 1.0
            elif r2 <= _SURROUND_R * _SURROUND_R:
                w = -0.5
            else:
                continue
            out += w * padded[
                _SURROUND_R + dy:_SURROUND_R + dy + map_.shape[0],
                _SURROUND_R + dx:_SURROUND_R + dx + map_.shape[1],
            ]
    return np.clip(out, 0.0, None)


class SaliencyMap:
    """LIP priority map — where attention should look next.

    ``compute`` builds the map from a frame plus optional V1/MT inputs;
    ``attend_next`` picks the winner and stamps it with inhibition of
    return so the next call moves on — attention scans rather than
    locking forever on the most salient spot.
    """

    def __init__(self) -> None:
        """Initialize inhibition-of-return and goal-bias state."""
        self._ior: list[tuple[int, int, float]] = []  # (my, mx, expires)
        self._goal_map: np.ndarray | None = None
        self.last: SaliencyResult | None = None

    # ── Feature maps ─────────────────────────────────────────────

    @staticmethod
    def _downsample(channel: np.ndarray) -> np.ndarray:
        """Average-pool a full-res channel onto the map grid."""
        h, w = channel.shape
        ys = np.linspace(0, h, _MAP_H + 1).astype(int)
        xs = np.linspace(0, w, _MAP_W + 1).astype(int)
        out = np.zeros((_MAP_H, _MAP_W), dtype=np.float32)
        for i in range(_MAP_H):
            for j in range(_MAP_W):
                block = channel[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                if block.size:
                    out[i, j] = float(block.mean())
        return out

    def _feature_maps(
        self,
        rgb: np.ndarray,
        v1: VisualField | None,
        motion: MotionField | None,
    ) -> dict[str, np.ndarray]:
        """Build bottom-up intensity, color, orientation, and motion maps."""
        f = rgb.astype(np.float32)
        lum = f[..., :3].mean(axis=2)

        maps: dict[str, np.ndarray] = {}

        # Intensity: centre-surround luminance contrast.
        maps["intensity"] = _center_surround(self._downsample(lum))

        # Color opponency: R-G and B-Y double-opponent contrast.
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        rg = np.clip(r - g, 0, None)
        by = np.clip(b - np.minimum(r, g), 0, None)
        maps["color"] = _center_surround(self._downsample(rg)) + _center_surround(
            self._downsample(by)
        )

        # Orientation: V1 feature strengths splatted at their positions.
        orient = np.zeros((_MAP_H, _MAP_W), dtype=np.float32)
        if v1 is not None and v1.features:
            h, w = lum.shape
            for feat in v1.features:
                my = min(_MAP_H - 1, max(0, int(feat.y / h * _MAP_H)))
                mx = min(_MAP_W - 1, max(0, int(feat.x / w * _MAP_W)))
                orient[my, mx] += feat.strength
        maps["orientation"] = _center_surround(orient)

        # Motion: MT flow magnitudes splatted at cell centres.
        mot = np.zeros((_MAP_H, _MAP_W), dtype=np.float32)
        if motion is not None:
            h, w = lum.shape
            for c in motion.cells:
                my = min(_MAP_H - 1, max(0, int(c.y / h * _MAP_H)))
                mx = min(_MAP_W - 1, max(0, int(c.x / w * _MAP_W)))
                mot[my, mx] += c.speed * c.confidence
        maps["motion"] = mot
        return maps

    # ── Top-down bias ────────────────────────────────────────────

    def set_goal_bias(self, y: float, x: float, radius: float = 4.0) -> None:
        """Bias the map toward a pixel-space location (frontal input)."""
        gy, gx = np.mgrid[0:_MAP_H, 0:_MAP_W]
        my, mx = y, x  # caller passes map-space or pixel-space scaled
        d2 = (gy - my) ** 2 + (gx - mx) ** 2
        self._goal_map = np.exp(-d2 / (2 * radius * radius)).astype(np.float32)

    def clear_goal_bias(self) -> None:
        """Remove the current top-down location bias."""
        self._goal_map = None

    # ── Selection ────────────────────────────────────────────────

    def compute(
        self,
        rgb: np.ndarray,
        v1: VisualField | None = None,
        motion: MotionField | None = None,
    ) -> SaliencyResult:
        """Build the combined priority map for one frame."""
        maps = self._feature_maps(rgb, v1, motion)
        normed = {name: _normalize(m) for name, m in maps.items()}
        weights = {
            name: float(m.sum()) for name, m in normed.items()
        }
        total_w = sum(weights.values()) or 1.0
        priority = np.zeros((_MAP_H, _MAP_W), dtype=np.float32)
        for name in normed:
            priority += normed[name] * np.float32(weights[name] / total_w)
        if self._goal_map is not None:
            priority = (
                priority * np.float32(0.6) + self._goal_map * np.float32(0.4)
            ).astype(np.float32)

        # Inhibition of return — recently attended cells are suppressed.
        now = time.monotonic()
        self._ior = [(my, mx, t) for my, mx, t in self._ior if t > now]
        for my, mx, _ in self._ior:
            yy, xx = np.mgrid[0:_MAP_H, 0:_MAP_W]
            d2 = (yy - my) ** 2 + (xx - mx) ** 2
            priority *= 1.0 - 0.9 * np.exp(-d2 / (2 * _IOR_RADIUS ** 2))

        result = SaliencyResult(
            priority=priority,
            scene_salience=min(1.0, float(priority.max()) if priority.size else 0.0),
            feature_weights=weights,
        )
        self.last = result
        return result

    def attend_next(
        self, result: SaliencyResult | None = None, frame_shape: tuple[int, int] | None = None,
    ) -> AttendedLocation | None:
        """Winner-take-all: pick the max, stamp inhibition of return."""
        result = result or self.last
        if result is None or result.priority.size == 0:
            return None
        peak = float(result.priority.max())
        if peak <= 0:
            return None
        my, mx = np.unravel_index(int(result.priority.argmax()), result.priority.shape)
        self._ior.append((int(my), int(mx), time.monotonic() + _IOR_SECONDS))

        # Identify the dominant contributor at the winning cell.
        weights = result.feature_weights
        feature = max(weights, key=lambda k: weights[k]) if weights else ""

        h, w = frame_shape or (480, 640)
        attended = AttendedLocation(
            y=(my + 0.5) / _MAP_H * h,
            x=(mx + 0.5) / _MAP_W * w,
            map_y=int(my), map_x=int(mx),
            salience=peak,
            feature=feature,
        )
        result.attended = attended
        return attended

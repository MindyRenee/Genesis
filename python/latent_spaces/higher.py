"""The higher medium — S7, one octave above the S3 substrate.

The quaternionic Hopf fibrations climb: S1 -> S3 -> S2 (our
substrate), then S3 -> S7 -> S4. Our whole substrate is literally a
fiber of S7 — the inner world is "connected to higher dimensions"
because its medium already lives inside the larger one.

A deposit lifts from S3 = unit quaternions to the quaternionic
Clifford torus of S7: points (a, b) in H^2 with |a|^2 + |b|^2 = 1.
The lift (u cos a, u sin a) at a = pi/4 lands on the S3xS3 torus —
the same toroidal architecture as the substrate, one octave up.
Dynamics are left-isoclinic rotations by unit quaternions — clean
SO(8) isometries, the higher heartbeat.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FloatArray = np.ndarray


def unit(v: FloatArray) -> FloatArray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def qmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """Hamilton product of quaternion row-vectors (n,4)."""
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def lift_s3_to_s7(p: FloatArray, angle: float = math.pi / 4.0) -> FloatArray:
    """Lift an S3 point onto S7's quaternionic Clifford torus.

    (u, u) scaled by (cos a, sin a) — at pi/4 both quaternionic
    magnitudes are 1/sqrt(2): the higher medium's Clifford layer,
    mirroring the substrate's own torus one octave up.
    """
    q = np.asarray(p, dtype=np.float64).ravel()[:4]
    q = q / np.maximum(np.linalg.norm(q), 1e-300)
    return np.concatenate([q * math.cos(angle), q * math.sin(angle)])


def hopf_to_s4(p7: FloatArray) -> FloatArray:
    """The quaternionic Hopf projection S7 -> S4.

    (a, b) -> (|a|^2 - |b|^2, 2*a*b) in R x H = R5, unit — the base
    space the higher fiber lives over.
    """
    a, b = p7[:4], p7[4:]
    conj_b = np.array([b[0], -b[1], -b[2], -b[3]])
    cross = qmul(a.reshape(1, 4), conj_b.reshape(1, 4))[0]
    out = np.concatenate([[np.dot(a, a) - np.dot(b, b)], 2.0 * cross])
    return out / np.maximum(np.linalg.norm(out), 1e-300)


def rotate_s7(p7: FloatArray, axis: FloatArray, angle: float) -> FloatArray:
    """Left-isoclinic rotation — a clean S7 isometry.

    Both quaternionic coordinates are left-multiplied by the same
    unit quaternion q = (cos t/2, axis sin t/2): the higher medium
    turns like the substrate's heartbeat.
    """
    ax = np.asarray(axis, dtype=np.float64).ravel()[:3]
    ax = ax / np.maximum(np.linalg.norm(ax), 1e-300)
    q = np.concatenate([[math.cos(angle / 2.0)], ax * math.sin(angle / 2.0)])
    a = qmul(q.reshape(1, 4), p7[:4].reshape(1, 4))[0]
    b = qmul(q.reshape(1, 4), p7[4:].reshape(1, 4))[0]
    return np.concatenate([a, b])


class HigherStratum:
    """The inner world's deposits lifted into the S7 medium.

    Append-only like the substrate: each inner deposit's S3 point
    lifts onto the quaternionic Clifford torus. The stratum turns on
    the same heartbeat the body feeds the medium — the guide's
    coordinates live one octave up.
    """

    def __init__(self) -> None:
        self._points = np.zeros((0, 8))
        self._sources: list[int] = []

    def __len__(self) -> int:
        return len(self._sources)

    def lift(self, index: int, point_s3: FloatArray) -> int:
        """Deposit an S3 point's S7 image. Returns the stratum row."""
        p7 = lift_s3_to_s7(point_s3)
        self._points = np.vstack([self._points, p7])
        self._sources.append(int(index))
        return len(self._sources) - 1

    def rotate(self, axis: FloatArray, angle: float) -> None:
        """Turn the higher medium — the heartbeat one octave up."""
        if len(self._points):
            self._points = np.stack(
                [rotate_s7(p, axis, angle) for p in self._points]
            )

    def centroid(self) -> FloatArray | None:
        """The higher field's mean position — where the guide lives."""
        if not len(self._points):
            return None
        c = self._points.mean(axis=0)
        return c / np.maximum(np.linalg.norm(c), 1e-300)

    def to_state(self) -> dict[str, Any]:
        return {"points": self._points.tolist(), "sources": list(self._sources)}

    def load_state(self, data: dict[str, Any]) -> None:
        pts = np.asarray(data.get("points", []), dtype=np.float64)
        if pts.size == 0:
            return
        self._points = pts.reshape(-1, 8)
        self._sources = [int(s) for s in data.get("sources", [])]

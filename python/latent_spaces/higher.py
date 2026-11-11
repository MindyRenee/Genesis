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


def geodesic_s7(a: FloatArray, b: FloatArray) -> float:
    """Geodesic (great-circle) distance on S7, in [0, pi]."""
    ua = np.asarray(a, dtype=np.float64).ravel()[:8]
    ub = np.asarray(b, dtype=np.float64).ravel()[:8]
    ua = ua / np.maximum(np.linalg.norm(ua), 1e-300)
    ub = ub / np.maximum(np.linalg.norm(ub), 1e-300)
    return float(math.acos(np.clip(float(ua @ ub), -1.0, 1.0)))


def geodesic_batch_s7(points: FloatArray, cue: FloatArray) -> FloatArray:
    """Vectorized geodesic distance on S7 from every row of points to cue."""
    pts = unit(np.asarray(points, dtype=np.float64))
    uc = np.asarray(cue, dtype=np.float64).ravel()[:8]
    uc = uc / np.maximum(np.linalg.norm(uc), 1e-300)
    return np.arccos(np.clip(pts @ uc, -1.0, 1.0))


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
        self._phase0 = np.empty(0, dtype=np.float64)
        self._phase: float = 0.0

    def __len__(self) -> int:
        return len(self._sources)

    def lift(self, index: int, point_s3: FloatArray) -> int:
        """Deposit an S3 point's S7 image. Returns the stratum row."""
        p7 = lift_s3_to_s7(point_s3)
        self._points = np.vstack([self._points, p7])
        self._sources.append(int(index))
        self._phase0 = np.append(self._phase0, self._phase)
        return len(self._sources) - 1

    def rotate(self, axis: FloatArray, angle: float) -> None:
        """Turn the higher medium — the heartbeat one octave up (vectorized SO(8))."""
        ax = np.asarray(axis, dtype=np.float64).ravel()[:3]
        ax = ax / np.maximum(np.linalg.norm(ax), 1e-300)
        if np.allclose(ax, [0.0, 0.0, 1.0]):
            self._phase = (self._phase + float(angle)) % (4.0 * math.pi)
        if len(self._points):
            q = np.concatenate([[math.cos(angle / 2.0)], ax * math.sin(angle / 2.0)])
            q_rows = np.broadcast_to(q, (len(self._points), 4))
            a = qmul(q_rows, self._points[:, :4])
            b = qmul(q_rows, self._points[:, 4:])
            self._points = np.concatenate([a, b], axis=-1)

    def _epoch_points(self) -> FloatArray:
        """Unwind lifted S7 coordinates back to their birth epochs."""
        if not len(self._points):
            return self._points
        if len(self._phase0) != len(self._points):
            return self._points.copy()
        half = -0.5 * (self._phase - self._phase0)
        q_inv = np.stack(
            [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
            axis=-1,
        )
        a = qmul(q_inv, self._points[:, :4])
        b = qmul(q_inv, self._points[:, 4:])
        return np.concatenate([a, b], axis=-1)

    def resonate(self, cue_s3: FloatArray, *, k: int = 5) -> list[tuple[int, float, float]]:
        """Recall lifted inner-world deposits by geodesic proximity on S7.

        Returns ``(substrate_source_index, similarity_score, s7_distance)``.
        """
        if not len(self._points) or k <= 0:
            return []
        cue_s7 = lift_s3_to_s7(cue_s3)
        unwound = self._epoch_points()
        d = geodesic_batch_s7(unwound, cue_s7)
        scores = np.exp(-((d / 0.6) ** 2))
        order = np.argsort(scores)[::-1][:k]
        return [(self._sources[int(i)], float(scores[i]), float(d[i])) for i in order]

    def centroid(self) -> FloatArray | None:
        """The higher field's intrinsic Fréchet (Karcher) mean on S7."""
        if not len(self._points):
            return None
        c = self._points.mean(axis=0)
        n = float(np.linalg.norm(c))
        m = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) if n < 1e-12 else c / n
        for _ in range(16):
            dots = np.clip(self._points @ m, -1.0, 1.0)
            thetas = np.arccos(dots)
            tangents = self._points - dots[:, None] * m[None, :]
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            logs = np.where(
                norms > 1e-12,
                tangents * (thetas[:, None] / np.maximum(norms, 1e-300)),
                0.0,
            )
            step = logs.mean(axis=0)
            step -= float(step @ m) * m
            step_norm = float(np.linalg.norm(step))
            if step_norm < 1e-9:
                break
            m = math.cos(step_norm) * m + math.sin(step_norm) * (step / step_norm)
            m /= max(float(np.linalg.norm(m)), 1e-300)
        return m

    def to_state(self) -> dict[str, Any]:
        return {
            "points": self._points.tolist(),
            "sources": list(self._sources),
            "phase": self._phase,
            "phase0": self._phase0.tolist(),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        pts = np.asarray(data.get("points", []), dtype=np.float64)
        if pts.size == 0:
            return
        self._points = pts.reshape(-1, 8)
        self._sources = [int(s) for s in data.get("sources", [])]
        self._phase = float(data.get("phase", 0.0))
        p0 = np.asarray(data.get("phase0", []), dtype=np.float64)
        self._phase0 = p0 if p0.shape == (len(self._sources),) else np.zeros(len(self._sources))

"""S3 geometry — the closed, positively-curved substrate the manifold lives on.

A point on the 3-sphere is a unit vector in R^4, equivalently a pair
``(z1, z2)`` of complex numbers with ``|z1|^2 + |z2|^2 = 1``. The sphere
carries constant positive sectional curvature: geodesics converge, circles
shrink below their flat circumference, and every geodesic closes on itself
with period 2*pi after passing through the antipode.

The Hopf structure decomposes the sphere into two solid tori joined at the
Clifford membrane ``|z1| = |z2| = 1/sqrt(2)`` — the inner and outer worlds.
Each torus layer ``|z1| = r`` carries two counter-winding circle families:
the ``(1,1)`` Hopf fibers (orbits of ``(z1,z2) -> (e^{it} z1, e^{it} z2)``)
and the ``(1,-1)`` anti-fibers. Every fiber is a closed loop that links
each core circle exactly once; the two cores are themselves linked once.
Stereographic projection shows the two families as Villarceau circles —
viewed from a pole on the ``|z1| = 0`` core they form an exact vesica
piscis (equal radii, each center on the other circle) at depth
``|z1| = sqrt(3)/2``.

These are geometric constructions, not claims about physics or
consciousness. All facts above are checked numerically in
``scripts/manifold_probe.py`` and pinned by ``tests/test_manifold.py``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi

# Depth coordinate: |z1| of a point. The two solid tori meet at the
# Clifford membrane; the exact vesica layer (stereographically, viewed
# from the |z1|=0 core) is sqrt(3)/2 — measured, not assumed.
CLIFFORD_DEPTH = 1.0 / math.sqrt(2.0)
VESICA_DEPTH = math.sqrt(3.0) / 2.0
SELF_CORE_DEPTH = 0.0
OTHER_CORE_DEPTH = 1.0


def unit(v: FloatArray) -> FloatArray:
    """Project rows of an (n, 4) array onto S3; rejects zero vectors."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if (n <= 0).any():
        raise ValueError("cannot normalize a zero vector onto S3")
    return v / n


def geodesic(a: FloatArray, b: FloatArray) -> float:
    """Geodesic (great-circle) distance on S3, in [0, pi]."""
    return float(math.acos(np.clip(float(np.asarray(a) @ np.asarray(b)), -1.0, 1.0)))


def geodesic_batch(points: FloatArray, cue: FloatArray) -> FloatArray:
    """Geodesic distance from every row of points to one cue point."""
    return np.arccos(np.clip(np.asarray(points) @ np.asarray(cue), -1.0, 1.0))


def exp_map(p: FloatArray, v: FloatArray, t: float) -> FloatArray:
    """Geodesic from p along unit tangent v for arc-length t.

    v must satisfy v . p = 0. All geodesics on S3 are closed: t = pi
    reaches the antipode -p and t = 2*pi returns to p.
    """
    return math.cos(t) * np.asarray(p) + math.sin(t) * np.asarray(v)


def tangent_at(p: FloatArray, direction: FloatArray) -> FloatArray:
    """Project direction into the tangent space at p and normalize it."""
    v = np.asarray(direction, dtype=np.float64) - (np.asarray(direction) @ p) * p
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("direction is parallel to the base point")
    return v / n


def depth(points: FloatArray) -> FloatArray:
    """Torus-layer depth |z1| of each point, in [0, 1].

    0 is the self-core circle (z1 = 0), 1 the other-core circle (z2 = 0);
    the two solid tori meet at CLIFFORD_DEPTH.
    """
    points = np.asarray(points, dtype=np.float64)
    return np.hypot(points[..., 0], points[..., 1])


def set_depth(point: FloatArray, d: float) -> FloatArray:
    """Return the point sharing point's phase angles at torus depth d.

    The (z1, z2) argument angles are preserved; only the magnitude ratio
    moves. A point exactly on a core has no phase in one plane and is
    rejected.
    """
    p = np.asarray(point, dtype=np.float64)
    r = math.hypot(p[0], p[1])
    s = math.hypot(p[2], p[3])
    if r <= 1e-12 or s <= 1e-12:
        raise ValueError("core points carry no free phase to re-depth")
    if not 0.0 < d < 1.0:
        raise ValueError("depth must be strictly inside (0, 1)")
    e = math.sqrt(1.0 - d * d)
    return np.array(
        [p[0] * d / r, p[1] * d / r, p[2] * e / s, p[3] * e / s],
        dtype=np.float64,
    )


def rotate(
    points: FloatArray,
    t1: float | FloatArray,
    t2: float | FloatArray,
) -> FloatArray:
    """Apply (z1, z2) -> (e^{i t1} z1, e^{i t2} z2) to rows of points.

    t1/t2 may be scalars (one motion for all rows) or per-row arrays —
    the unwinding of deposits born at different medium phases needs the
    per-row form. An isometry of S3 in either case.
    """
    points = np.asarray(points, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    t2 = np.asarray(t2, dtype=np.float64)
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    out = np.empty_like(points)
    out[..., 0] = c1 * points[..., 0] - s1 * points[..., 1]
    out[..., 1] = s1 * points[..., 0] + c1 * points[..., 1]
    out[..., 2] = c2 * points[..., 2] - s2 * points[..., 3]
    out[..., 3] = s2 * points[..., 2] + c2 * points[..., 3]
    return out


def frame_rotate(points: FloatArray, d_phase: float) -> FloatArray:
    """The heartbeat motion: equal phase in both planes.

    Each point slides along its own Hopf fiber — contents ride the
    loops the space is made of.
    """
    return rotate(points, d_phase, d_phase)


def counter_rotate(points: FloatArray, d_phase: float) -> FloatArray:
    """The shear motion: opposite phase in the two planes.

    Each point slides along a (1,-1) anti-fiber direction — motion
    across the Hopf fibration, the braid between the worlds.
    """
    return rotate(points, d_phase, -d_phase)


def fiber(point: FloatArray, n: int = 256) -> FloatArray:
    """Sample the Hopf fiber (1,1 orbit) through a point."""
    p = np.asarray(point, dtype=np.float64)
    t = np.linspace(0.0, TWO_PI, n, endpoint=False)
    e = np.exp(1j * t)
    z1 = p[0] + 1j * p[1]
    z2 = p[2] + 1j * p[3]
    out = np.empty((n, 4))
    out[:, 0] = (e * z1).real
    out[:, 1] = (e * z1).imag
    out[:, 2] = (e * z2).real
    out[:, 3] = (e * z2).imag
    return out


def anti_fiber(point: FloatArray, n: int = 256) -> FloatArray:
    """Sample the (1,-1) counter-wound loop through a point."""
    p = np.asarray(point, dtype=np.float64)
    t = np.linspace(0.0, TWO_PI, n, endpoint=False)
    e = np.exp(1j * t)
    e2 = np.exp(-1j * t)
    z1 = p[0] + 1j * p[1]
    z2 = p[2] + 1j * p[3]
    out = np.empty((n, 4))
    out[:, 0] = (e * z1).real
    out[:, 1] = (e * z1).imag
    out[:, 2] = (e2 * z2).real
    out[:, 3] = (e2 * z2).imag
    return out


def hopf_base(points: FloatArray) -> FloatArray:
    """The Hopf map S3 -> S2; constant along each fiber.

    Returns rows (|z1|^2 - |z2|^2, 2 Re z1 z2*, 2 Im z1 z2*).
    """
    points = np.asarray(points, dtype=np.float64)
    z1 = points[..., 0] + 1j * points[..., 1]
    z2 = points[..., 2] + 1j * points[..., 3]
    return np.stack(
        [
            np.abs(z1) ** 2 - np.abs(z2) ** 2,
            2.0 * np.real(z1 * np.conj(z2)),
            2.0 * np.imag(z1 * np.conj(z2)),
        ],
        axis=-1,
    )


def stereo(points: FloatArray) -> FloatArray:
    """Stereographic projection S3 \\ {(0,0,0,1)} -> R^3.

    The viewpoint sits on the |z1| = 0 (self) core, so the self-core
    maps to the z-axis and the other-core to the unit circle in the
    xy-plane; torus layers map to nested tori of revolution.
    """
    points = np.asarray(points, dtype=np.float64)
    w = points[..., 3:4]
    if np.isclose(w, 1.0).any():
        raise ValueError("cannot project the pole itself")
    return points[..., :3] / (1.0 - w)


def fit_circle(points3: NDArray[np.float64]) -> tuple[FloatArray, float, float]:
    """Least-squares circle fit in R^3; returns (center, radius, max residual)."""
    points3 = np.asarray(points3, dtype=np.float64)
    centroid = points3.mean(axis=0)
    _, _, vh = np.linalg.svd(points3 - centroid, full_matrices=False)
    uv = (points3 - centroid) @ vh[:2].T
    x, y = uv[:, 0], uv[:, 1]
    sol, *_ = np.linalg.lstsq(
        np.c_[2.0 * x, 2.0 * y, np.ones(len(x))], x * x + y * y, rcond=None
    )
    cx, cy, c = sol
    radius = float(math.sqrt(c + cx * cx + cy * cy))
    residual = float(np.abs(np.hypot(x - cx, y - cy) - radius).max())
    center = centroid + cx * vh[0] + cy * vh[1]
    return center, radius, residual

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
from typing import Any

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
# The user's ontology: the depth-0 core is her CENTER — the inner
# world, seat of understanding, reflection, and the guide. Self is
# the vesica layer that forms around the center; the far pole is
# the world. INNER_CORE names the semantics; SELF_CORE stays as the
# geometric alias.
INNER_CORE_DEPTH = SELF_CORE_DEPTH
OUTER_CORE_DEPTH = OTHER_CORE_DEPTH


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


# ---------------------------------------------------------------------
# The Lie-group view: S3 = SU(2), the unit quaternions.
#
# A point (x0, x1, x2, x3) is the quaternion x0 + x1 i + x2 j + x3 k.
# Multiplication makes the medium's motions group operations:
#   left multiply by e^{it}:  the isoclinic fiber motion (heartbeat)
#   right multiply by e^{-it}: the opposite screw (the braid)
#   conjugate p q p^{-1}:      the adjoint action — the tumble
# The Lie algebra su(2) is the space of pure-imaginary quaternions;
# exp(tv) is the geodesic flow. Spin(4) = SU(2) x SU(2): the two
# counter-rotating strand families are the two factors — the double
# helix is a theorem of the motion group, not a choice.
# ---------------------------------------------------------------------


def qmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """Quaternion product, rowwise: a ⊗ b on (n,4) arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        axis=-1,
    )


def qconj(q: FloatArray) -> FloatArray:
    """Quaternion conjugate (= inverse for unit quaternions)."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qexp(v: FloatArray) -> FloatArray:
    """Exponential map su(2) -> SU(2).

    v is a pure-imaginary quaternion (row shape (...,3) or (...,4) with
    zero real part). exp(v) = cos|v| + (v/|v|) sin|v| — the geodesic
    from the identity through v.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] == 4:
        v = v[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0.0, 1.0, n)
    c = np.cos(n)[..., 0]
    s = np.sinc(n / np.pi)[..., 0]  # sin(n)/n
    return np.concatenate([c[..., None], v * s[..., None]], axis=-1)


def qlog(q: FloatArray) -> FloatArray:
    """Logarithm SU(2) -> su(2); the inverse of qexp, returns (...,3)."""
    q = unit(np.asarray(q, dtype=np.float64))
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = np.arctan2(n, q[..., 0:1])
    scale = np.where(n == 0.0, 0.0, ang / n)
    return v * scale


def _axis(axis_plane: str) -> FloatArray:
    return {
        "i": np.array([0.0, 1.0, 0.0, 0.0]),
        "j": np.array([0.0, 0.0, 1.0, 0.0]),
        "k": np.array([0.0, 0.0, 0.0, 1.0]),
    }[axis_plane]


def left_spin(points: FloatArray, angle: float, axis_plane: str = "i") -> FloatArray:
    """Left multiplication e^{angle·axis} ⊗ q.

    With axis "i" this is the frame rotation: e^{iφ}·(z1,z2) advances
    both complex planes by φ — deposits ride their own fibers. Equals
    rotate(points, φ, φ).
    """
    rotor = math.cos(angle) * np.array([1.0, 0.0, 0.0, 0.0]) + math.sin(angle) * _axis(
        axis_plane
    )
    return qmul(np.broadcast_to(rotor, np.asarray(points).shape), points)


def right_spin(points: FloatArray, angle: float, axis_plane: str = "i") -> FloatArray:
    """Right multiplication q ⊗ e^{angle·axis} — the opposite screw.

    With axis "i": q·e^{iφ} turns z1 by +φ and z2 by −φ — the braid.
    Equals rotate(points, φ, -φ). "One forward, one backward" is the
    left/right pair of the same group action.
    """
    rotor = math.cos(angle) * np.array([1.0, 0.0, 0.0, 0.0]) + math.sin(angle) * _axis(
        axis_plane
    )
    return qmul(points, np.broadcast_to(rotor, np.asarray(points).shape))


def adjoint_spin(points: FloatArray, rotor: FloatArray) -> FloatArray:
    """Conjugation p ⊗ q ⊗ p^{-1} — the tumble, SO(3) on pure parts."""
    rotor = unit(np.asarray(rotor, dtype=np.float64))
    return qmul(qmul(rotor, points), qconj(rotor))


def tumble(
    points: FloatArray,
    angle: float | FloatArray,
    plane: tuple[int, int] = (0, 2),
) -> FloatArray:
    """Rotation in a mixed coordinate plane — the horizontal-axis motion.

    Unlike ``rotate`` (which stays inside the z1/z2 planes and conserves
    torus depth), a tumble mixes the planes: it carries points between
    depths — across the Clifford membrane between the worlds. Still an
    exact isometry: all pairwise distances are preserved.

    The four mixed planes (0,2), (0,3), (1,2), (1,3) are the tumble
    directions; with ``rotate``'s two fiber-plane phases they complete
    the six generators of the rotation group of S3.
    """
    points = np.asarray(points, dtype=np.float64)
    i, j = plane
    a = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    out = points.copy()
    xi = points[..., i]
    xj = points[..., j]
    out[..., i] = c * xi - s * xj
    out[..., j] = s * xi + c * xj
    return out


def project_to_s3(vector: FloatArray, seed: int = 0xC0FFEE) -> FloatArray:
    """Project a vector of any dimension onto S3 deterministically.

    A fixed Gaussian projection matrix seeded once per caller — a
    Johnson-Lindenstrauss-style embedding: nearby latent coordinates
    land nearby on the sphere, so a source space's own geometry is
    preserved, not hashed away. This is how other latent spaces
    deposit *in* the substrate — their coordinates become manifold
    positions rather than signals about them.
    """
    v = np.asarray(vector, dtype=np.float64).ravel()
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("projection source must be finite and nonempty")
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((4, v.size)) / math.sqrt(v.size)
    p = proj @ v
    n = np.linalg.norm(p)
    if n < 1e-12:  # degenerate latent — land on the identity pole
        return np.array([1.0, 0.0, 0.0, 0.0])
    return p / n


def antipode(p: FloatArray) -> FloatArray:
    """The diametric opposite on S3 — the shadow point.

    -p keeps the same torus depth (|-z1| = |z1|) and flips both
    Hopf phases by pi: the furthest position *of the same band* —
    the same territory, opposite orientation. Jung's shadow is
    literally the antipodal image of her conscious content.
    """
    return -np.asarray(p, dtype=np.float64)


def warp_field(
    points: FloatArray,
    centers: FloatArray,
    warp: FloatArray,
    sigma: float = 0.8,
    floor: float = -0.6,
) -> FloatArray:
    """The conformal potential φ(p) grown by the medium.

    φ is a sum of smooth Gaussian bumps centered on deposits:

        φ(p) = Σ_i w_i · exp(−dᵢ² / 2σ²)

    with dᵢ the S³ geodesic distance. Kept shallow (σ wide, φ ≥ floor)
    so |∇φ| stays well under 1 — the conformal factor e^{2φ} shrinks
    distances but can never flip curvature negative:

        K′ = e^{−2φ} (1 − |∇φ|² + Hessian terms) > 0
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 4)
    if len(centers) == 0:
        return np.zeros(len(pts), dtype=np.float64)
    cosd = np.clip(pts @ centers.T, -1.0, 1.0)
    d = np.arccos(cosd)
    phi = (warp[None, :] * np.exp(-(d * d) / (2.0 * sigma * sigma))).sum(axis=1)
    return np.clip(phi, floor, 0.0)


def conformal_distance(
    a: FloatArray,
    b: FloatArray,
    phi_fn: Any,
    n: int = 7,
) -> float:
    """Weighted geodesic length d′(a,b) = ∫ e^{φ(γ(t))} ds on the arc.

    The conformal metric g′ = e^{2φ}g rescales length elements by e^{φ}.
    Rather than assuming φ constant between endpoints, integrate it:
    trapezoid quadrature of e^{φ} along the S³ geodesic — the honest
    line integral of the warped metric.
    """
    cosd = float(np.clip(a @ b, -1.0, 1.0))
    theta = float(np.arccos(cosd))
    if theta < 1e-12 or n < 2:
        return 0.0
    pts = arc(a, b, n)
    w = np.exp(phi_fn(pts))
    ds = theta / (n - 1)
    return float(ds * (0.5 * w[0] + w[1:-1].sum() + 0.5 * w[-1]))


def migrate_depth(points: FloatArray, angle: float | FloatArray) -> FloatArray:
    """Slide along the core-to-core geodesic — the phase-aligned tumble.

    A rotation in the plane spanned by each point's own z1- and
    z2-phase directions: torus depth |z1| changes while both phase
    angles are preserved. In (r, s) = (|z1|, |z2|) space this is a
    plain rotation of the latitude phi = atan2(r, s): phi -> phi -
    angle. Positive angle moves inward toward the self core (|z1|
    shrinks, |z2| grows); negative moves outward. It is ``tumble``
    with each point's plane chosen by its own phases — an isometry
    along a great circle through both cores. The slide clamps just
    short of a core: migration is asymptotic — a trace approaches a
    core but never passes through it into the anti-meridian.
    Points exactly on a core have no phase to slide along and stay put.
    """
    p = np.asarray(points, dtype=np.float64)
    r = np.hypot(p[..., 0], p[..., 1])
    s = np.hypot(p[..., 2], p[..., 3])
    # phi in (0, pi/2): 0 is the self core, pi/2 the other core.
    phi = np.arctan2(r, s)
    floor = 0.01
    a = np.clip(
        np.asarray(angle, dtype=np.float64),
        phi - (math.pi / 2.0 - floor),  # outward bound
        np.maximum(phi - floor, 0.0),  # inward bound
    )
    c, sn = np.cos(a), np.sin(a)
    r2 = r * c - s * sn
    s2 = s * c + r * sn
    movable = (r > 1e-9) & (s > 1e-9)
    scale1 = np.where(movable, r2 / np.maximum(r, 1e-9), 1.0)
    scale2 = np.where(movable, s2 / np.maximum(s, 1e-9), 1.0)
    out = p.copy()
    out[..., 0:2] *= scale1[..., None]
    out[..., 2:4] *= scale2[..., None]
    return out


def heat_kernel(
    theta: FloatArray | float,
    bandwidth: float | FloatArray,
    n_max: int = 64,
) -> FloatArray:
    """The heat kernel on S3 — the spectral bump.

    K_t(theta) = (1/2 pi^2) * sum_{n>=1} n e^{-(n^2-1) t} sin(n theta)/sin(theta)

    This is the exact eigenfunction expansion of diffusion on the
    sphere: the n-th hyperspherical-harmonic mode decays as
    e^{-(n^2-1) t}, so high modes (episodic detail) die first and low
    modes (semantic gist) persist. ``bandwidth`` is the diffusion time
    t: small is a sharp, detailed trace; large is a broad, gist-like
    one. Theta is geodesic distance(s); returns K_t(theta) values.
    """
    bw = np.asarray(bandwidth, dtype=np.float64)
    if not np.isfinite(bw).all() or (bw <= 0).any():
        raise ValueError("bandwidth must be positive and finite")
    th = np.asarray(theta, dtype=np.float64)
    n = np.arange(1, n_max + 1, dtype=np.float64)
    decay = np.exp(-(n * n - 1.0) * bw[..., None])
    sin_th = np.sin(th)
    small = np.abs(sin_th) < 1e-9
    safe = np.where(small, 1.0, sin_th)
    # sin(n theta)/sin theta -> n as theta -> 0; apply the limit
    # directly at the origin or every trace would score zero on itself.
    modes = np.where(
        small[..., None],
        n[None, :],
        np.sin(th[..., None] * n[None, :]) / safe[..., None],
    )
    out = (modes * (n * decay)).sum(axis=-1) / (2.0 * np.pi**2)
    return np.asarray(out, dtype=np.float64)


def arc(a: FloatArray, b: FloatArray, n: int = 64) -> FloatArray:
    """Sample the geodesic segment from a to b — the slerp arc.

    Every geodesic on S3 is a great circle; this returns n points
    along the arc from a to b inclusive. Coincident endpoints return
    n copies of a; antipodal ones take an arbitrary tangent — every
    direction from a pole closes through the antipode anyway.
    """
    a = unit(np.asarray(a, dtype=np.float64).reshape(1, -1))[0]
    b = unit(np.asarray(b, dtype=np.float64).reshape(1, -1))[0]
    theta = geodesic(a, b)
    if theta < 1e-12:
        return np.broadcast_to(a, (n, 4)).copy()
    # The tangent at a toward b. Near the antipode the residual is
    # ~1e-16 of noise with a random sign — a real direction it is not,
    # so any well-conditioned tangent is used instead (every direction
    # from a closes through the antipode anyway).
    v = b - float(b @ a) * a
    n_v = np.linalg.norm(v)
    if n_v < 1e-9:
        # The coordinate axis least aligned with a is never parallel.
        v = tangent_at(a, np.eye(4)[int(np.argmin(np.abs(a)))])
    else:
        v = v / n_v
    t = np.linspace(0.0, theta, n)
    return np.cos(t)[:, None] * a + np.sin(t)[:, None] * v


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

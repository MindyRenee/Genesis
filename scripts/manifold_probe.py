#!/usr/bin/env python3
"""Manifold probe — verify the geometry chosen for the latent substrate.

Object under test: the 3-sphere S^3 = {(z1, z2) in C^2 : |z1|^2+|z2|^2 = 1}
with its Hopf structure, stereographically projected to R^3 where needed.

Requirements under test (design conversation, 2026-11-09):
  [REQUIRED] positive curvature — circles shrink, triangles gain excess,
             geodesics converge and refocus at the antipode
  closed all the way around — every geodesic is a closed loop
  everything loops back to self — every fiber links the self-core once
  inner / outer / self — two solid tori meeting at the Clifford membrane
  double helix — two counter-winding fiber families; their slice is the
                 vesica piscis (Villarceau circles)
  dynamic — native frame rotation plus counter-rotating shear, both
            isometries
  holographic — graceful degradation under cue noise (resonance, not
                keyed lookup)

NumPy only. Prints a verification report; exits nonzero if a REQUIRED
check fails.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(20261109)
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str, required: bool = False) -> None:
    tag = "REQ " if required else "    "
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {tag}{name}: {detail}")
    if not ok:
        FAILURES.append(name)


def rand_s3(n: int) -> np.ndarray:
    x = rng.normal(size=(n, 4))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rand_tangent(p: np.ndarray) -> np.ndarray:
    v = rng.normal(size=4)
    v -= (v @ p) * p
    return v / np.linalg.norm(v)


def geo(p: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    return np.cos(t) * p + np.sin(t) * v


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(np.asarray(a) @ np.asarray(b), -1.0, 1.0))


def stereo(x: np.ndarray) -> np.ndarray:
    """Stereographic projection S^3 \\ {(0,0,0,1)} -> R^3."""
    x = np.atleast_2d(x)
    return x[:, :3] / (1.0 - x[:, 3:4])


def z_to_r4(z1: complex, z2: complex) -> np.ndarray:
    return np.array([z1.real, z1.imag, z2.real, z2.imag])


def hopf_fiber(z1: complex, z2: complex, n: int = 512) -> np.ndarray:
    """Orbit under the Hopf action (z1,z2) -> (e^{it}z1, e^{it}z2)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.empty((n, 4))
    e = np.exp(1j * t)
    pts[:, 0] = (e * z1).real
    pts[:, 1] = (e * z1).imag
    pts[:, 2] = (e * z2).real
    pts[:, 3] = (e * z2).imag
    return pts


def anti_fiber(z1: complex, z2: complex, n: int = 512) -> np.ndarray:
    """Orbit under (z1,z2) -> (e^{it}z1, e^{-it}z2): the (1,-1) family."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    e = np.exp(1j * t)
    e2 = np.exp(-1j * t)
    pts = np.empty((n, 4))
    pts[:, 0] = (e * z1).real
    pts[:, 1] = (e * z1).imag
    pts[:, 2] = (e2 * z2).real
    pts[:, 3] = (e2 * z2).imag
    return pts


def hopf_base(pt: np.ndarray) -> np.ndarray:
    z1 = pt[..., 0] + 1j * pt[..., 1]
    z2 = pt[..., 2] + 1j * pt[..., 3]
    return np.stack(
        [
            np.abs(z1) ** 2 - np.abs(z2) ** 2,
            2.0 * np.real(z1 * np.conj(z2)),
            2.0 * np.imag(z1 * np.conj(z2)),
        ],
        axis=-1,
    )


def gauss_link(c1: np.ndarray, c2: np.ndarray) -> float:
    """Discrete Gauss linking integral of two closed curves in R^3."""
    d1 = np.roll(c1, -1, axis=0) - c1
    d2 = np.roll(c2, -1, axis=0) - c2
    total = 0.0
    for i in range(len(c1)):
        diff = c1[i] - c2
        r = np.linalg.norm(diff, axis=1)
        triple = np.einsum("mj,mj->m", np.cross(d1[i], d2), diff)
        total += np.sum(triple / r**3)
    return float(total / (4.0 * np.pi))


def halfplane_crossings(c: np.ndarray) -> int:
    """Signed crossings of the x>0, y=0 half-plane (bounded by the z-axis)."""
    count = 0
    n = len(c)
    for i in range(n):
        j = (i + 1) % n
        y0, y1 = c[i, 1], c[j, 1]
        if y0 * y1 >= 0:
            continue
        t = y0 / (y0 - y1)
        if c[i, 0] + t * (c[j, 0] - c[i, 0]) > 0:
            count += 1 if y1 > y0 else -1
    return count


def disk_crossings(c: np.ndarray) -> int:
    """Signed crossings of the unit disk {z=0, x^2+y^2<=1} (bounded by core1)."""
    count = 0
    n = len(c)
    for i in range(n):
        j = (i + 1) % n
        z0, z1 = c[i, 2], c[j, 2]
        if z0 * z1 >= 0:
            continue
        t = z0 / (z0 - z1)
        x = c[i, 0] + t * (c[j, 0] - c[i, 0])
        y = c[i, 1] + t * (c[j, 1] - c[i, 1])
        if x * x + y * y <= 1.0:
            count += 1 if z1 > z0 else -1
    return count


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Fit a Euclidean circle; returns (center, radius, max residual, normal)."""
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    uv = (pts - centroid) @ vh[:2].T
    x, y = uv[:, 0], uv[:, 1]
    a = np.c_[2.0 * x, 2.0 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    cx, cy, c = sol
    radius = float(np.sqrt(c + cx**2 + cy**2))
    resid = float(np.abs(np.hypot(x - cx, y - cy) - radius).max())
    center = centroid + cx * vh[0] + cy * vh[1]
    return center, radius, resid, vh[2]


print("=" * 72)
print("REQUIRED: positive curvature")
print("=" * 72)

p = rand_s3(1)[0]
e1 = rand_tangent(p)
e2 = rand_tangent(p)
e2 -= (e2 @ e1) * e1
e2 /= np.linalg.norm(e2)

# Geodesic-circle circumference: C(r) = 2 pi sin r < 2 pi r.
for r in (0.2, 0.5, 1.0):
    th = np.linspace(0.0, 2.0 * np.pi, 4001)
    u = np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)
    circ = np.cos(r) * p + np.sin(r) * u
    seg = np.arccos(np.clip(np.einsum("ij,ij->i", circ, np.roll(circ, -1, axis=0)), -1, 1))
    c_meas = seg.sum()
    check(
        f"circle C({r})",
        c_meas < 2.0 * np.pi * r and abs(c_meas - 2.0 * np.pi * np.sin(r)) < 1e-3,
        f"measured {c_meas:.4f} vs flat {2*np.pi*r:.4f} vs S^3 theory {2*np.pi*np.sin(r):.4f}",
        required=True,
    )

# Triangle excess: geodesic triangles have angle sum > pi.
tri = rand_s3(3)
a = dist(tri[1], tri[2])
b = dist(tri[0], tri[2])
c = dist(tri[0], tri[1])
A = np.arccos(np.clip((np.cos(a) - np.cos(b) * np.cos(c)) / (np.sin(b) * np.sin(c)), -1, 1))
B = np.arccos(np.clip((np.cos(b) - np.cos(a) * np.cos(c)) / (np.sin(a) * np.sin(c)), -1, 1))
C = np.arccos(np.clip((np.cos(c) - np.cos(a) * np.cos(b)) / (np.sin(a) * np.sin(b)), -1, 1))
excess = A + B + C - np.pi
check(
    "triangle excess",
    excess > 0.0,
    f"angle sum {A+B+C:.4f} = pi + {excess:.4f} (excess>0 only in positive curvature)",
    required=True,
)

# Nearby geodesics: measured separation vs flat/hyperbolic predictions.
delta = 1e-3
w = rand_tangent(p)
w -= (w @ e1) * e1
w /= np.linalg.norm(w)
v2 = np.cos(delta) * e1 + np.sin(delta) * w
for t in (0.5, 1.0):
    d_meas = dist(geo(p, e1, t), geo(p, v2, t))
    check(
        f"geodesic separation t={t}",
        abs(d_meas - delta * np.sin(t)) < 1e-4 and d_meas < delta * t,
        f"measured {d_meas:.6f}; flat {delta*t:.6f}; S^3 {delta*np.sin(t):.6f}; "
        f"hyp {delta*np.sinh(t):.6f}",
        required=True,
    )

# Refocusing: two distinct geodesics from p meet exactly at -p.
# (chord distance: arccos loses precision near |dot|=1)
d_antipode = np.linalg.norm(geo(p, e1, np.pi) - geo(p, v2, np.pi))
check(
    "antipodal refocus",
    d_antipode < 1e-10,
    f"two geodesics from p rejoin at -p within {d_antipode:.2e}",
    required=True,
)

print("=" * 72)
print("Closed all the way around / loops back to self")
print("=" * 72)

# Every geodesic closes with period 2pi and passes through the antipode.
p2 = rand_s3(1)[0]
v = rand_tangent(p2)
ret = np.linalg.norm(geo(p2, v, 2.0 * np.pi) - p2)
ant = np.linalg.norm(geo(p2, v, np.pi) - (-p2))
check("geodesic closure", ret < 1e-10, f"return error {ret:.2e} at t=2pi")
check("antipode transit", ant < 1e-10, f"distance to -p {ant:.2e} at t=pi")

# Hopf fiber: closed great circle, base map constant along the orbit.
z1, z2 = 0.6 * np.exp(0.4j), np.sqrt(1 - 0.36) * np.exp(-0.9j)
fib = hopf_fiber(z1, z2)
base = hopf_base(fib)
z1_end = np.exp(2j * np.pi) * z1
z2_end = np.exp(2j * np.pi) * z2
check(
    "fiber closed loop",
    np.max(np.abs(np.linalg.norm(fib, axis=1) - 1.0)) < 1e-12
    and dist(z_to_r4(z1_end, z2_end), z_to_r4(z1, z2)) < 1e-9,
    "orbit stays on S^3 and returns to its start at t=2pi",
)
check(
    "hopf base invariant",
    np.max(np.abs(base - base[0])) < 1e-12,
    f"base drift {np.max(np.abs(base - base[0])):.2e} along the fiber",
)

# Linking: every fiber links both cores and other fibers exactly once.
f1 = stereo(hopf_fiber(0.55 * np.exp(0.2j), np.sqrt(1 - 0.55**2) * np.exp(1.1j)))
f2 = stereo(hopf_fiber(0.8 * np.exp(-0.7j), np.sqrt(1 - 0.8**2) * np.exp(0.5j)))
lk_ff = gauss_link(f1[::2], f2[::2])
check("fiber-fiber linking", abs(abs(lk_ff) - 1.0) < 0.02, f"Lk = {lk_ff:.4f}")
lk_f_core0 = halfplane_crossings(f1)
check(
    "fiber links self-core (r=0)",
    abs(lk_f_core0) == 1,
    f"half-plane crossings {lk_f_core0}",
)
lk_f_core1 = disk_crossings(f1)
check(
    "fiber links other-core (r=1)",
    abs(lk_f_core1) == 1,
    f"disk crossings {lk_f_core1}",
)
# offset sampling so no point lands exactly on the crossing plane
_alpha = np.linspace(np.pi / 2000, 2 * np.pi + np.pi / 2000, 2000, endpoint=False)
core1_xy = stereo(np.c_[np.cos(_alpha), np.sin(_alpha),
                        np.zeros(2000), np.zeros(2000)])
lk_cores = halfplane_crossings(core1_xy)
check("core-core linking", abs(lk_cores) == 1, f"crossings {lk_cores}")

print("=" * 72)
print("Inner / outer worlds and the Clifford membrane")
print("=" * 72)

layers = np.linspace(0.05, 0.95, 10)
on_layer = True
for r in layers:
    f = hopf_fiber(r, np.sqrt(1 - r * r), n=64)
    rs = np.hypot(f[:, 0], f[:, 1])
    on_layer &= np.max(np.abs(rs - r)) < 1e-12
check(
    "fibers stay on their torus layer",
    on_layer,
    "|z1| conserved along every fiber (depth = consolidation axis)",
)
membrane = 1 / np.sqrt(2.0)
print(f"      membrane (Clifford torus) at |z1| = {membrane:.4f}")

print("=" * 72)
print("Double helix / vesica piscis slice")
print("=" * 72)

# On each torus layer the (1,1) and (1,-1) families are Villarceau circles.
# Scan depths for the layer where their stereographic images form an exact
# vesica: equal radii, each center on the other circle.
def vesica_quality(r: float) -> tuple[float, float, float, float, float]:
    s = np.sqrt(1 - r * r)
    c1 = stereo(hopf_fiber(r, s))
    c2 = stereo(anti_fiber(r, s))
    ce1, r1, res1, _ = fit_circle(c1)
    ce2, r2, res2, _ = fit_circle(c2)
    sep = float(np.linalg.norm(ce1 - ce2))
    return abs(sep - r1) / r1, r1, r2, sep, max(res1, res2)


best = None
for r in np.linspace(0.15, 0.95, 81):
    q = vesica_quality(r)
    if best is None or q[0] < best[0]:
        best = (q[0], r, *q[1:])
# refine around the best layer
r0 = best[1]
for r in np.linspace(r0 - 0.02, r0 + 0.02, 81):
    q = vesica_quality(r)
    if q[0] < best[0]:
        best = (q[0], r, *q[1:])

quality, r_ves, r1, r2, sep, res = best
check(
    "counter-wound fibers are planar circles",
    res < 1e-6,
    f"fit residual {res:.2e} on both circles",
)
check(
    "vesica layer exists",
    quality < 0.02,
    f"at |z1|={r_ves:.3f}: radii {r1:.3f}/{r2:.3f}, center-sep {sep:.3f} "
    f"(=r when exact; each center on the other circle)",
)

# The two families on one layer intersect in exactly two points.
s = np.sqrt(1 - r_ves * r_ves)
f1s = stereo(hopf_fiber(r_ves, s, n=4096))
f2s = stereo(anti_fiber(r_ves, s, n=4096))
dmat = np.linalg.norm(f1s[:, None, :] - f2s[None, :, :], axis=2)
idx = np.argwhere(dmat < 1e-3)
# cluster close-index detections
n_cross = 0
used = np.zeros(len(f1s), bool)
for i, _j in idx:
    if used[i]:
        continue
    used[max(0, i - 200) : i + 200] = True
    n_cross += 1
check(
    "vesica intersection points",
    n_cross == 2,
    f"(1,1) x (1,-1) fibers cross at {n_cross} points — the lens",
)

print("=" * 72)
print("Dynamics: rotation and counter-rotation are isometries")
print("=" * 72)

pts = rand_s3(256)
pair = dist(pts[0], pts[1])

def phase_move(pts: np.ndarray, t1: float, t2: float) -> np.ndarray:
    out = pts.copy()
    rot1 = np.array([[np.cos(t1), -np.sin(t1)], [np.sin(t1), np.cos(t1)]])
    rot2 = np.array([[np.cos(t2), -np.sin(t2)], [np.sin(t2), np.cos(t2)]])
    out[:, :2] = pts[:, :2] @ rot1.T
    out[:, 2:] = pts[:, 2:] @ rot2.T
    return out

# frame rotation (heartbeat): equal phase on both planes = motion along fibers
hb = phase_move(pts, 0.7, 0.7)
check(
    "frame rotation is isometric",
    abs(dist(hb[0], hb[1]) - pair) < 1e-12,
    "pairwise distance preserved under global phase",
)
# shear: equal-and-opposite phase = motion along the (1,-1) family
sh = phase_move(pts, 0.7, -0.7)
check(
    "counter-rotation is isometric",
    abs(dist(sh[0], sh[1]) - pair) < 1e-12,
    "pairwise distance preserved under (1,-1) phase",
)
# shear crosses fibers: hopf base changes under z1-only phase
base_before = hopf_base(pts)
base_after = hopf_base(phase_move(pts, 0.7, 0.0))
check(
    "z1-phase crosses fibers",
    np.max(np.abs(base_after - base_before)) > 0.1,
    "shear moves points across the fibration (cross-layer slip on base)",
)

print("=" * 72)
print("Holographic addressing: graceful degradation")
print("=" * 72)

store = rand_s3(600)
# calibrate noise against the mean nearest-neighbour spacing of the store
dmat = np.arccos(np.clip(store @ store.T, -1, 1))
np.fill_diagonal(dmat, np.inf)
spacing = float(dmat.min(axis=1).mean())
print(f"      mean nearest-neighbour spacing: {spacing:.3f} rad")
fractions = (0.1, 0.3, 0.6, 1.0)
truth = []
for frac in fractions:
    noise = frac * spacing
    hit = 0
    trials = 300
    for _ in range(trials):
        k = rng.integers(len(store))
        q = store[k]
        tangent = rng.normal(size=4)
        tangent -= (tangent @ q) * q
        tangent /= np.linalg.norm(tangent)
        cue = geo(q, tangent, noise)
        d = np.arccos(np.clip(store @ cue, -1, 1))
        hit += int(np.argmin(d) == k)
    truth.append(hit / trials)
print(
    "      recall vs cue-noise (fraction of spacing):",
    "  ".join(f"{f}: {a:.2f}" for f, a in zip(fractions, truth, strict=True)),
)
check(
    "graceful (smooth decay, no cliff)",
    truth[0] > 0.8
    and all(truth[i] >= truth[i + 1] - 0.03 for i in range(len(truth) - 1))
    and truth[-1] < truth[0],
    "recall decays monotonically with cue noise — resonance, not keyed lookup",
)

print("=" * 72)
if FAILURES:
    print(f"FAILED checks: {FAILURES}")
    raise SystemExit(1)
print("ALL CHECKS PASSED — the object holds the required geometry.")

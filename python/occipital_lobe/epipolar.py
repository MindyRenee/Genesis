"""MSTd — two-view (epipolar) geometry of successive looks.

The dorsal stream continues past MT/V5 into the medial superior
temporal area. Its dorsal division (MSTd) integrates wide-field
retinal flow into *self-motion*: heading — the focus of expansion —
and rotation, and it reads scene depth out of motion parallax. MSTd
neurons carry heading signals that predict perceptual judgments
(Britten & van Wezel 1998; Gu et al. 2006), and structure that
violates the observer's own optic flow pops out there as an
independently moving object (Logan & Duffy 2006).

Genesis has one eye — a monocular camera — so it has no binocular
baseline. But epipolar geometry is not about two eyes; it is about
two *views*. Every time the eye opens, the camera takes a view; two
successive looks are related by the same constraint that relates a
left and a right eye:

    x'ᵀ F x = 0

where x, x' are the same world point seen in the two views. The
baseline is not interpupillary distance — it is *whatever moved*
between the looks: the body shifting, the head turning, the world
moving. When nothing moved, the constraint degenerates and the
honest percept is stillness — a stationary eye on a stationary scene
produces no parallax. That is the biology too: without microsaccade
or head motion there is no parallax depth, only the flow field.

Pipeline per look:

    luminance frame
        │
        ▼
    [corners]  Shi–Tomasi — trackable texture (min eigenvalue of the
        │        structure tensor), grid-suppressed into an even spread
        ▼
    [KLT]      pyramidal Lucas–Kanade — each corner followed into the
        │        new view; drops points that left the frame or smeared
        ▼
    [F]        normalized 8-point (Hartley 1997) inside RANSAC, scored
        │        by Sampson distance — the rank-2 epipolar model of the
        │        pair of views
        ▼
    [E]        calibrated 8-point through K⁻¹, projected onto the
        │        essential manifold (singular values snapped to 1,1,0)
        ▼
    [R, t]     SVD decomposition → 4 candidate poses → cheirality:
        │        triangulate the inliers and keep the pose that puts
        │        the world in front of both views
        ▼
    [depth]    DLT triangulation → relative depth (up to scale — a
        │        monocular view cannot fix the baseline length) and
        │        motion parallax: nearer structure sweeps further
        ▼
    TwoViewReport — self-motion, heading, depth and independent movers

The pure-rotation degenerate case is real: a look that differs from
the last only by a turn carries no translation baseline, E collapses,
and the two views are related by a homography instead. MSTd's own
rotation-selective cells handle exactly this — here a RANSAC
homography fit is the fallback model, and its rotation is recovered
from K⁻¹HK.

Everything is NumPy — the same constraint as the rest of the
occipital lobe. The geometry is the percept; what it means is left
to the rest of the mind.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EpipolarProcessor",
    "TwoViewReport",
    "algebraic_residual",
    "camera_matrix",
    "decompose_essential",
    "detect_corners",
    "eight_point",
    "epipolar_lines",
    "epipoles",
    "essential_from_fundamental",
    "estimate_essential",
    "estimate_fundamental",
    "estimate_homography",
    "huber",
    "normalize_points",
    "project_to_essential",
    "recover_pose",
    "refine_pose",
    "rotation_from_homography",
    "sampson_sq",
    "symmetric_epipolar_sq",
    "track_points",
    "transfer_error_sq",
    "triangulate",
]

# Working resolution for the dorsal stream — long axis in pixels.
# The geometry does not need retinal resolution; a 320-px view keeps
# tracking cheap and the structure tensor well conditioned.
_WORK_DIM = 320

# Corner detection (Shi–Tomasi). Corners are what LK can actually
# follow — an edge is aperture-ambiguous, a flat patch carries no
# signal at all.
_MAX_POINTS = 200  # corners kept per view
_CORNER_CELL = 16  # px — one strongest corner per grid cell
_CORNER_MIN_REL = 0.01  # min cornerness as a fraction of the peak
_CORNER_ABS_MIN = 1.0  # absolute floor — below it is noise, not texture
_CORNER_WIN = 5  # structure-tensor summation window radius

# Pyramidal Lucas–Kanade. Three levels reach ~4x the window radius of
# per-look motion; the iteration count is the usual KLT budget.
_LK_LEVELS = 3
_LK_WIN = 9  # half-window — 19×19 patch per point
_LK_ITERS = 15
_LK_EPS = 0.03  # px — convergence on the flow increment
_LK_MAX_RESID = 30.0  # mean |I2−I1| patch residual — smeared tracks

# Epipolar estimation.
_RANSAC_ITERS = 512
_SAMPSON_THRESH_PX = 1.5  # inlier gate, px (squared-distance compare)
_H_INLIER_PX = 3.0  # homography fallback inlier gate, px
_MIN_INLIER_RATIO = 0.45  # below this the geometry did not explain the flow
_MIN_TRACKS = 12  # below this there is no field to integrate

# A view pair counts as moved when the median corner actually
# displaced — below ~1 px of shift, tracking noise is the baseline.
_MOTION_EPS_PX = 1.0

# Only a track that actually displaced carries a baseline — a
# near-stationary correspondence satisfies any geometry for free and
# triangulates "in front" under any pose, so it neither fits nor
# proves a model. Estimation runs on tracks past this displacement.
_GEOM_MIN_DISP_PX = 2.0

# Forward–backward consistency: a real correspondence round-trips.
# Track prev→curr then curr→prev and keep only tracks that land back
# within a pixel — the rest locked onto look-alike texture.
_FB_MAX_PX = 1.0

# Cheirality: the fraction of inliers that must triangulate in front
# of both views for a decomposed pose to be believed.
_CHEIRALITY_MIN = 0.6

# A track counts as an independent mover only if its patch carried
# over cleanly — a smeared track is a failed follow, not a mover.
_MOVER_MAX_RESID = _LK_MAX_RESID * 0.5


def _to_luminance(rgb: np.ndarray) -> np.ndarray:
    """Float32 luminance of an RGB (or already-flat) frame."""
    if rgb.ndim == 3:
        return rgb[..., :3].astype(np.float32).mean(axis=2)
    return rgb.astype(np.float32)


def _shrink(gray: np.ndarray, max_dim: int = _WORK_DIM) -> tuple[np.ndarray, float]:
    """Box-decimate *gray* so its long axis is at most *max_dim*.

    Returns the smaller frame and the scale factor mapping its
    coordinates back up to the input's (input_px = work_px / scale).
    Only whole decimation factors are used, so scale is 2^-k.
    """
    out = gray
    k = 0
    while max(out.shape) > max_dim:
        h2, w2 = out.shape[0] // 2, out.shape[1] // 2
        if h2 < 8 or w2 < 8:
            break
        out = out[: 2 * h2, : 2 * w2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))
        k += 1
    return out, 2.0 ** (-k)


def _bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample *img* at fractional (ys, xs); out-of-bounds → 0."""
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = xs - x0
    dy = ys - y0
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    v = (
        img[y0c, x0c] * (1 - dx) * (1 - dy)
        + img[y0c, x1c] * dx * (1 - dy)
        + img[y1c, x0c] * (1 - dx) * dy
        + img[y1c, x1c] * dx * dy
    )
    return np.where(inside, v, 0.0)


def _integral(img: np.ndarray) -> np.ndarray:
    """Integral image with a zero border: S[y, x] sums img[:y, :x]."""
    s = np.cumsum(np.cumsum(img, axis=0), axis=1)
    out = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=s.dtype)
    out[1:, 1:] = s
    return out


def _box_sum(integ: np.ndarray, win: int) -> np.ndarray:
    """Sum each (2win+1)² neighborhood via the integral image."""
    h, w = integ.shape[0] - 1, integ.shape[1] - 1
    y0 = np.clip(np.arange(h) - win, 0, h)
    y1 = np.clip(np.arange(h) + win + 1, 0, h)
    x0 = np.clip(np.arange(w) - win, 0, w)
    x1 = np.clip(np.arange(w) + win + 1, 0, w)
    a = integ[np.ix_(y0, x0)]
    b = integ[np.ix_(y1, x0)]
    c = integ[np.ix_(y0, x1)]
    d = integ[np.ix_(y1, x1)]
    return d - b - c + a


def detect_corners(
    gray: np.ndarray,
    n_max: int = _MAX_POINTS,
    cell: int = _CORNER_CELL,
) -> np.ndarray:
    """Shi–Tomasi corners — the points LK can actually follow.

    Cornerness is the smaller eigenvalue of the windowed structure
    tensor; one strongest corner survives per *cell*-px grid cell so
    the field spreads across the whole view instead of clustering on
    the highest-contrast texture. Returns an (N, 2) array of (y, x)
    centers ordered by cornerness.
    """
    gy, gx = np.gradient(gray)
    sxx = _box_sum(_integral(gx * gx), _CORNER_WIN)
    syy = _box_sum(_integral(gy * gy), _CORNER_WIN)
    sxy = _box_sum(_integral(gx * gy), _CORNER_WIN)
    tr = (sxx + syy) * 0.5
    disc = np.sqrt(np.maximum(tr * tr - (sxx * syy - sxy * sxy), 0.0))
    cornerness = tr - disc  # min eigenvalue of the 2×2 tensor

    peak = float(cornerness.max()) if cornerness.size else 0.0
    thresh = max(_CORNER_ABS_MIN, _CORNER_MIN_REL * peak)

    h, w = gray.shape
    picks: list[tuple[float, float, float]] = []  # (cornerness, y, x)
    for cy in range(0, h - cell + 1, cell):
        for cx in range(0, w - cell + 1, cell):
            block = cornerness[cy : cy + cell, cx : cx + cell]
            idx = int(np.argmax(block))
            v = float(block.flat[idx])
            if v < thresh:
                continue
            picks.append((v, cy + idx // cell + 0.5, cx + idx % cell + 0.5))
    picks.sort(key=lambda p: -p[0])
    if not picks:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([(y, x) for _, y, x in picks[:n_max]], dtype=np.float64)


def _pyramid(gray: np.ndarray, levels: int) -> list[np.ndarray]:
    """Coarse-to-fine image pyramid by 2×2 mean decimation."""
    pyr = [gray]
    for _ in range(levels - 1):
        g = pyr[-1]
        h2, w2 = g.shape[0] // 2, g.shape[1] // 2
        if h2 < _LK_WIN + 2 or w2 < _LK_WIN + 2:
            break
        pyr.append(g[: 2 * h2, : 2 * w2].reshape(h2, 2, w2, 2).mean(axis=(1, 3)))
    return pyr


def track_points(
    prev: np.ndarray,
    curr: np.ndarray,
    pts: np.ndarray,
    win: int = _LK_WIN,
    levels: int = _LK_LEVELS,
    iters: int = _LK_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pyramidal Lucas–Kanade: follow *pts* from *prev* into *curr*.

    Per point, a translation-only warp refined Gauss–Newton style —
    the same constant-brightness least squares MT runs, here on a
    tracked patch instead of a fixed cell. *pts* are (y, x).

    Returns (tracked_pts, ok, resid): the new (y, x) positions, which
    tracks survived (in-frame and converged), and each track's mean
    absolute patch residual — how faithfully the texture carried over.
    """
    n = len(pts)
    if n == 0:
        return pts, np.zeros(0, dtype=bool), np.zeros(0)
    pyr1 = _pyramid(prev, levels)
    pyr2 = _pyramid(curr, levels)
    n_lvl = len(pyr2)
    flow = np.zeros((n, 2), dtype=np.float64)  # (dy, dx), current-level px

    ys_w = np.arange(-win, win + 1, dtype=np.float64)
    wy, wx = np.meshgrid(ys_w, ys_w, indexing="ij")
    wy_flat = wy.ravel()
    wx_flat = wx.ravel()

    for lvl in range(n_lvl - 1, -1, -1):
        s = 0.5**lvl
        i1 = pyr1[lvl]
        i2 = pyr2[lvl]
        g2y, g2x = np.gradient(i2)
        h, w = i2.shape
        p_lvl = pts * s
        # Refine the previous level's estimate; carried flow doubles
        # per finer level (the same displacement in denser pixels).
        flow = flow if lvl == n_lvl - 1 else flow * 2.0
        patch1 = _bilinear(i1, p_lvl[:, 0, None] + wy_flat, p_lvl[:, 1, None] + wx_flat)
        for _ in range(iters):
            wy_s = p_lvl[:, 0, None] + flow[:, 0, None] + wy_flat
            wx_s = p_lvl[:, 1, None] + flow[:, 1, None] + wx_flat
            patch2 = _bilinear(i2, wy_s, wx_s)
            jy = _bilinear(g2y, wy_s, wx_s)
            jx = _bilinear(g2x, wy_s, wx_s)
            r = patch2 - patch1
            axx = (jx * jx).sum(axis=1)
            axy = (jx * jy).sum(axis=1)
            ayy = (jy * jy).sum(axis=1)
            bx = -(jx * r).sum(axis=1)
            by = -(jy * r).sum(axis=1)
            det = axx * ayy - axy * axy
            tr = axx + ayy
            good = (det > 1e-2 * tr * tr) & (tr > 1e-3)
            safe = np.where(good)[0]
            if len(safe) == 0:
                break
            det_s = det[safe]
            dx = (ayy[safe] * bx[safe] - axy[safe] * by[safe]) / det_s
            dy = (axx[safe] * by[safe] - axy[safe] * bx[safe]) / det_s
            flow[safe, 1] += dx
            flow[safe, 0] += dy
            if float(np.abs(np.stack([dx, dy], axis=1)).max()) < _LK_EPS:
                break

    new_pts = pts + flow
    margin = win + 2
    h, w = curr.shape
    inside = (
        (new_pts[:, 0] >= margin)
        & (new_pts[:, 0] < h - margin)
        & (new_pts[:, 1] >= margin)
        & (new_pts[:, 1] < w - margin)
    )
    patch1 = _bilinear(prev, pts[:, 0, None] + wy_flat, pts[:, 1, None] + wx_flat)
    patch2 = _bilinear(curr, new_pts[:, 0, None] + wy_flat, new_pts[:, 1, None] + wx_flat)
    resid = np.abs(patch2 - patch1).mean(axis=1)
    # The residual gate is relative to the patch's own signal: a flat
    # patch that "tracked" into different flat noise reads a small
    # absolute error but carries no correspondence — its resid is the
    # same order as its content. Real texture carries resid ≪ std.
    patch_std = patch1.std(axis=1)
    ok = inside & (resid < _LK_MAX_RESID) & (resid < 0.5 * np.maximum(patch_std, 4.0))
    return new_pts, ok, resid


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization: centroid at the origin, mean
    distance √2. The 8-point system is hopelessly ill-conditioned in
    raw pixel units; this is what makes it usable."""
    c = pts.mean(axis=0)
    shifted = pts - c
    d = float(np.sqrt((shifted * shifted).sum(axis=1)).mean())
    s = math.sqrt(2.0) / max(d, 1e-9)
    t = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    return shifted * s, t


def _homog(pts: np.ndarray) -> np.ndarray:
    """(N, 2) → (N, 3) homogeneous."""
    return np.concatenate([pts, np.ones((len(pts), 1))], axis=1)


def eight_point(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Normalized 8-point estimate of F from correspondences.

    Each pair contributes one row x'⊗x to the system A·f = 0; f is the
    smallest right singular vector. The result is projected to rank 2 —
    a full-rank F has no epipole and its lines never concur, so the
    constraint has to be enforced, not assumed.
    """
    p1n, t1 = normalize_points(p1)
    p2n, t2 = normalize_points(p2)
    x = p1n[:, 0]
    y = p1n[:, 1]
    xp = p2n[:, 0]
    yp = p2n[:, 1]
    a = np.stack(
        [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, np.ones_like(x)],
        axis=1,
    )
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    f = vt[-1].reshape(3, 3)
    u, sv, fvt = np.linalg.svd(f)
    sv[2] = 0.0
    f = (u * sv) @ fvt
    # Un-normalize: x'ᵀ F x = (T2 x')ᵀ F̂ (T1 x) → F = T2ᵀ F̂ T1.
    f = t2.T @ f @ t1
    n = np.linalg.norm(f)
    return f / n if n > 1e-12 else f


def algebraic_residual(f: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """The raw epipolar residual x'ᵀFx per correspondence.

    Cheapest of the loss functions — but it is an algebraic quantity,
    not a distance: it scales with the magnitude of Fx, so a bad match
    can score small and a good one large. Useful as a diagnostic; the
    estimation gates on geometric losses instead."""
    x1 = _homog(p1)
    x2 = _homog(p2)
    return np.einsum("ij,ij->i", x2, x1 @ f.T)


def symmetric_epipolar_sq(f: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Squared symmetric epipolar distance: d(x', Fx)² + d(x, Fᵀx')² —
    the point-to-line distance in each view, summed. A true geometric
    loss in squared-pixel units."""
    x1 = _homog(p1)
    x2 = _homog(p2)
    l2 = x1 @ f.T  # lines in image 2
    l1 = x2 @ f  # lines in image 1
    d2 = np.einsum("ij,ij->i", x2, l2) ** 2 / np.maximum(l2[:, 0] ** 2 + l2[:, 1] ** 2, 1e-12)
    d1 = np.einsum("ij,ij->i", x1, l1) ** 2 / np.maximum(l1[:, 0] ** 2 + l1[:, 1] ** 2, 1e-12)
    return d1 + d2


def sampson_sq(f: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Squared Sampson distance — the first-order geometric error of a
    correspondence against F. Numerator: the algebraic residual
    (x'ᵀFx)². Denominator: the squared gradient of that residual
    w.r.t. the two image points, which rescales it into squared-pixel
    units so a RANSAC threshold means what it says. This is the loss
    OpenCV's RANSAC uses under the hood."""
    x1 = _homog(p1)
    x2 = _homog(p2)
    fx = x1 @ f.T  # (N, 3) epipolar lines in image 2
    ftx = x2 @ f  # (N, 3) epipolar lines in image 1
    num = np.einsum("ij,ij->i", x2, fx) ** 2
    den = fx[:, 0] ** 2 + fx[:, 1] ** 2 + ftx[:, 0] ** 2 + ftx[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def huber(sq_err: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Huber robust loss of squared errors: quadratic inside k²,
    linear beyond — inliers shape the fit, outliers stop pulling."""
    sq = np.asarray(sq_err, dtype=np.float64)
    k2 = k * k
    return np.where(sq <= k2, sq, k * (2.0 * np.sqrt(np.maximum(sq, 0.0)) - k))


def _ransac(
    p1: np.ndarray,
    p2: np.ndarray,
    n_sample: int,
    fit,
    score,
    thresh: float,
    iters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Generic minimal-sample consensus over correspondences."""
    n = len(p1)
    if n < n_sample:
        return None, np.zeros(n, dtype=bool)
    best_model = None
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    # Adaptive termination (Fischler & Bolles): once the consensus set
    # is large, far fewer samples are needed to have drawn a clean one
    # with 99% probability — recompute the bound as the estimate firms.
    required = iters
    i = 0
    while i < required:
        i += 1
        idx = rng.choice(n, size=n_sample, replace=False)
        model = fit(p1[idx], p2[idx])
        if model is None:
            continue
        inl = score(model, p1, p2) < thresh
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best_inliers = inl
            best_model = model
            if count == n:
                break
            w = count / n
            wn = w**n_sample
            if wn < 1.0:
                required = min(
                    required,
                    max(n_sample, math.ceil(math.log(0.01) / math.log(1.0 - wn))),
                )
    if best_model is None or best_count < n_sample:
        return None, np.zeros(n, dtype=bool)
    # Refit on the whole consensus set.
    refit = fit(p1[best_inliers], p2[best_inliers])
    if refit is not None:
        best_model = refit
        best_inliers = score(best_model, p1, p2) < thresh
    return best_model, best_inliers


def estimate_fundamental(
    p1: np.ndarray,
    p2: np.ndarray,
    thresh: float = _SAMPSON_THRESH_PX,
    iters: int = _RANSAC_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC + normalized 8-point → (F, inlier mask).

    The inlier mask is the model's own account of which tracks the
    two-view geometry explained; rejected tracks are either failed
    follows or things that moved on their own.
    """
    rng = rng if rng is not None else np.random.default_rng()
    return _ransac(
        p1,
        p2,
        8,
        eight_point,
        lambda m, a, b: sampson_sq(m, a, b),
        thresh * thresh,
        iters,
        rng,
    )


def epipolar_lines(f: np.ndarray, pts: np.ndarray, second: bool = True) -> np.ndarray:
    """Epipolar line coefficients for *pts* in the other view.

    ``second=True`` → l' = F x (lines in the second image);
    ``second=False`` → l = Fᵀ x' (lines in the first)."""
    h = _homog(pts)
    return h @ f.T if second else h @ f


def epipoles(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two epipoles (e1, e2) in homogeneous coordinates.

    e1 is the projection of the second view's eye into the first
    view — the right null vector of F. e2, the left null vector, is
    where the first eye appears in the second view. For a travelling
    eye the epipole is the focus of expansion: the heading, rendered
    on the retina."""
    u, _, vt = np.linalg.svd(f)
    e1 = vt[-1]  # right null vector: F e1 = 0
    e2 = u[:, -1]  # left null vector: e2ᵀ F = 0
    return e1, e2


def camera_matrix(h: int, w: int, focal: float | None = None) -> np.ndarray:
    """Pinhole intrinsics for an uncalibrated eye.

    Focal length defaults to the long axis — the standard
    wide-angle-webcam guess. The exact value only warps the metric
    scale of E's decomposition, not the epipolar constraint itself.
    """
    f = float(focal) if focal else float(max(h, w))
    return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])


def essential_from_fundamental(f: np.ndarray, k: np.ndarray) -> np.ndarray:
    """E = Kᵀ F K — lift the pixel-space constraint into calibrated
    (metric) space, where it factors as [t]×R."""
    return k.T @ f @ k


def project_to_essential(e: np.ndarray) -> np.ndarray:
    """Snap a matrix onto the essential manifold: singular values
    (σ, σ, 0). Any E estimated from noisy data drifts off the
    manifold; this is the closest point back on it."""
    u, _, vt = np.linalg.svd(e)
    return u @ np.diag([1.0, 1.0, 0.0]) @ vt


def estimate_essential(
    p1: np.ndarray,
    p2: np.ndarray,
    k: np.ndarray,
    thresh: float = _SAMPSON_THRESH_PX,
    iters: int = _RANSAC_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC 8-point in calibrated coordinates → (E, inlier mask).

    Points are lifted through K⁻¹ first, so the estimate solves
    x̂'ᵀ E x̂ = 0 directly and the fit is snapped back onto the
    essential manifold after every solve. The Sampson threshold is
    rescaled by the focal length — a pixel-space gate expressed in
    normalized units."""
    rng = rng if rng is not None else np.random.default_rng()
    k_inv = np.linalg.inv(k)
    p1n = (_homog(p1) @ k_inv.T)[:, :2]
    p2n = (_homog(p2) @ k_inv.T)[:, :2]
    f_norm = float(k[0, 0])

    def fit(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        if len(a) < 8:
            return None
        return project_to_essential(eight_point(a, b))

    return _ransac(
        p1n,
        p2n,
        8,
        fit,
        lambda m, a, b: sampson_sq(m, a, b),
        (thresh / f_norm) ** 2,
        iters,
        rng,
    )


def decompose_essential(e: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """E → the four candidate (R, t) poses.

    E = [t]×R = U diag(1,1,0) Vᵀ gives R ∈ {UWVᵀ, UWᵀVᵀ} and
    t = ±u₃. Only one of the four puts the world in front of both
    views — that is what recover_pose's cheirality check selects.
    """
    u, _, vt = np.linalg.svd(e)
    if np.linalg.det(u @ vt) < 0:
        # A reflection leaked into the decomposition — flip the last
        # singular direction so the reconstructed R is proper.
        vt = -vt
    w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = u[:, 2]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for r in (u @ w @ vt, u @ w.T @ vt):
        if np.linalg.det(r) < 0:
            r = -r
        out.append((r, t))
        out.append((r, -t))
    return out


def triangulate(
    p_1: np.ndarray,
    p_2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """DLT triangulation: correspondences → 3D points.

    For each pair, the two projection rays are re-expressed as linear
    constraints on X (x × PX = 0); the point is the smallest singular
    vector of the stacked 4×4 system. Returns (N, 3) points and the
    mask of points sitting in front of *both* cameras — the
    cheirality test."""
    x1h = _homog(x1)
    x2h = _homog(x2)
    n = len(x1)
    pts = np.zeros((n, 3))
    front = np.zeros(n, dtype=bool)
    for i in range(n):
        a = np.stack(
            [
                x1h[i, 0] * p_1[2] - p_1[0],
                x1h[i, 1] * p_1[2] - p_1[1],
                x2h[i, 0] * p_2[2] - p_2[0],
                x2h[i, 1] * p_2[2] - p_2[1],
            ]
        )
        _, _, vt = np.linalg.svd(a)
        xh = vt[-1]
        w = xh[3]
        if abs(w) < 1e-9:
            continue
        pts[i] = xh[:3] / w
        # Depth along each camera's forward axis, sign-corrected for
        # the homogeneous scale (w flips the sign convention).
        d1 = float(p_1[2] @ xh) * w
        d2 = float(p_2[2] @ xh) * w
        front[i] = (d1 > 0) and (d2 > 0)
    return pts, front


def recover_pose(
    e: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the one physically possible pose out of E's four.

    For each candidate (R, t) the points are triangulated; the pose
    that puts the most scene structure in front of both views wins —
    the cheirality constraint. Returns (R, t_unit, in-front mask).
    t is a direction only: a monocular pair fixes the baseline
    direction, never its length."""
    best_r = np.eye(3)
    best_t = np.array([0.0, 0.0, 1.0])
    best_front = np.zeros(len(p1), dtype=bool)
    best_count = -1
    p1_cam = np.hstack([np.eye(3), np.zeros((3, 1))])
    for r, t in decompose_essential(e):
        p2_cam = np.hstack([r, t[:, None]])
        _, front = triangulate(p1_cam, p2_cam, p1, p2)
        count = int(front.sum())
        if count > best_count:
            best_count = count
            best_r, best_t, best_front = r, t, front
    return best_r, best_t, best_front


def _rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation vector → R via Rodrigues' formula."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * kx + (1.0 - math.cos(th)) * (kx @ kx)


def _rotation_log(r: np.ndarray) -> np.ndarray:
    """R → rotation vector (axis × angle)."""
    c = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w * (th / (2.0 * math.sin(th)))


def refine_pose(
    r: np.ndarray,
    t: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    iters: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg polish of a decomposed pose on the inliers.

    The 8-point estimate is algebraic — it minimizes a residual that
    is not a distance. Here the pose is refined directly on the
    Sampson error, parametrized in the essential manifold's five
    degrees of freedom: the rotation vector of R plus the tangent
    plane of the unit heading t. Gauss–Newton on sqrt(Sampson) with
    a finite-difference Jacobian — five extra residual evaluations
    per step, cheap at inlier counts."""
    if len(p1) < 8:
        return r, t
    w = _rotation_log(r)
    t = t / max(float(np.linalg.norm(t)), 1e-12)

    def residual(ww: np.ndarray, tt: np.ndarray) -> np.ndarray:
        rr = _rodrigues(ww)
        tt = tt / max(float(np.linalg.norm(tt)), 1e-12)
        tx = np.array([[0.0, -tt[2], tt[1]], [tt[2], 0.0, -tt[0]], [-tt[1], tt[0], 0.0]])
        e = tx @ rr
        return np.sqrt(np.maximum(sampson_sq(e, p1, p2), 0.0))

    lam = 1e-4
    eps = 1e-6
    for _ in range(iters):
        # Tangent basis of t: two unit vectors spanning t⊥.
        a = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b1 = np.cross(t, a)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(t, b1)
        base = residual(w, t)
        cost = float(base @ base)
        jac = np.zeros((len(base), 5))
        for j in range(3):
            dw = w.copy()
            dw[j] += eps
            jac[:, j] = (residual(dw, t) - base) / eps
        for j, b in enumerate((b1, b2)):
            jac[:, 3 + j] = (residual(w, t + eps * b) - base) / eps
        lhs = jac.T @ jac + lam * np.eye(5)
        rhs = -jac.T @ base
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            break
        w_new = w + delta[:3]
        t_new = t + delta[3] * b1 + delta[4] * b2
        t_new = t_new / max(float(np.linalg.norm(t_new)), 1e-12)
        new = residual(w_new, t_new)
        if float(new @ new) < cost:
            w, t = w_new, t_new
            lam = max(lam * 0.5, 1e-9)
        else:
            lam = min(lam * 4.0, 1e2)
            if lam >= 1e2:
                break
    return _rodrigues(w), t


def estimate_homography(
    p1: np.ndarray,
    p2: np.ndarray,
    thresh: float = _H_INLIER_PX,
    iters: int = _RANSAC_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC 4-point DLT homography — the pure-rotation fallback.

    When the baseline collapses to a turn, the two views are related
    by H = K R K⁻¹ (a rotation produces no parallax for E to work
    with). Scored by symmetric transfer error in pixels."""

    def fit(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        if len(a) < 4:
            return None
        an, ta = normalize_points(a)
        bn, tb = normalize_points(b)
        rows = []
        for (x, y), (xp, yp) in zip(an, bn, strict=True):
            # x'×(Ha) = 0 → two independent rows per correspondence.
            rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, yp * x, yp * y, yp])
            rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, -xp])
        _, _, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
        h = vt[-1].reshape(3, 3)
        return np.linalg.inv(tb) @ h @ ta

    rng = rng if rng is not None else np.random.default_rng()
    return _ransac(p1, p2, 4, fit, transfer_error_sq, thresh * thresh, iters, rng)


def transfer_error_sq(h: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Squared symmetric transfer error under H: |Hp−p'|² + |H⁻¹p'−p|² —
    the geometric loss for a homography, in squared pixels."""
    m_inv = np.linalg.inv(h + np.eye(3) * 1e-9)
    fwd = _homog(p1) @ h.T
    fwd = fwd[:, :2] / np.where(np.abs(fwd[:, 2, None]) > 1e-9, fwd[:, 2, None], 1.0)
    bwd = _homog(p2) @ m_inv.T
    bwd = bwd[:, :2] / np.where(np.abs(bwd[:, 2, None]) > 1e-9, bwd[:, 2, None], 1.0)
    return ((fwd - p2) ** 2).sum(axis=1) + ((bwd - p1) ** 2).sum(axis=1)


def rotation_from_homography(h: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Extract R from H = K R K⁻¹ — uncalibrate, normalize the scale,
    then orthogonalize back to the nearest rotation."""
    r = np.linalg.inv(k) @ h @ k
    r = r / max(float(np.linalg.norm(r[:, 0])), 1e-12)
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = -r
    return r


def _rotation_deg(r: np.ndarray) -> float:
    """Angle of a rotation matrix in degrees."""
    c = (float(np.trace(r)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _direction_word(t: np.ndarray) -> str:
    """Heading category of a unit translation, camera-frame:
    x→right, y→down, z→through the view. A second component at
    least half the dominant one's magnitude compounds the word —
    a 45° course is a diagonal heading, not either axis alone."""
    axes = sorted(
        (
            (abs(float(t[2])), "toward" if t[2] > 0 else "away"),
            (abs(float(t[0])), "right" if t[0] > 0 else "left"),
            (abs(float(t[1])), "down" if t[1] > 0 else "up"),
        ),
        key=lambda a: a[0],
        reverse=True,
    )
    (m0, w0), (m1, w1) = axes[0], axes[1]
    return f"{w0}-{w1}" if m1 >= 0.5 * m0 else w0


def _turn_word(p1: np.ndarray, p2: np.ndarray, mask: np.ndarray) -> str:
    """Which way the eye swung, from the retinal flow on *mask*.

    Points are (y, x): flow[1] is the horizontal shift, flow[0] the
    vertical. The world slides against the eye — content sweeping
    right across the sensor means the gaze turned left, content
    dropping means she pitched up. Compounds like _direction_word
    for oblique swings."""
    if not mask.any():
        return ""
    flow = (p2 - p1)[mask].mean(axis=0)
    horiz = "left" if flow[1] > 0 else "right"
    vert = "up" if flow[0] > 0 else "down"
    fx, fy = abs(flow[1]), abs(flow[0])
    if fx >= fy:
        return f"{horiz}-{vert}" if fy >= 0.5 * fx else horiz
    return f"{vert}-{horiz}" if fx >= 0.5 * fy else vert


@dataclass(slots=True)
class TwoViewReport:
    """What the dorsal stream made of the move between two looks.

    Pure percept data — self-motion, heading, parallax depth, and the
    structure that refused to participate. ``kind`` is the
    categorical read: "still" (no baseline — nothing moved), "turn"
    (rotation only — the world swung but nothing came nearer),
    "travel" (a translation baseline — heading and depth are real),
    or "unresolved" (the field changed but no two-view model
    explained it — the scene itself changed, or the view relit).
    """

    kind: str = ""  # "" | "still" | "turn" | "travel" | "unresolved"
    moved: bool = False  # a real baseline existed between the looks
    n_tracks: int = 0  # corners carried into this view
    n_inliers: int = 0  # tracks the geometry explained
    inlier_ratio: float = 0.0
    mean_residual: float = 0.0  # Sampson px of the explained tracks
    rotation: np.ndarray | None = None  # 3×3 camera rotation
    rotation_deg: float = 0.0
    translation: np.ndarray | None = None  # unit heading, camera frame
    direction: str = ""  # heading word: six primaries, compounding to diagonals
    heading_deg: float = 0.0  # angle of travel off the optical axis
    bearing_deg: float = 0.0  # azimuth around the up-axis: 0° fwd, 90° right, 180° back
    elevation_deg: float = 0.0  # above/below the eye-line: +up, −down
    epipole: tuple[float, float] | None = None  # FOE, work-scale px (may be off-frame)
    fundamental: np.ndarray | None = None
    essential: np.ndarray | None = None
    median_depth: float = 0.0  # relative — in units of the baseline
    near_fraction: float = 0.0  # share of structure nearer than the median
    nearest_region: str = ""  # third of the view with the strongest parallax
    movers: int = 0  # cleanly-tracked structure the model rejected
    mover_centroid: tuple[float, float] | None = None


class EpipolarProcessor:
    """MSTd — self-motion and parallax depth across successive looks.

    Holds the previous look's luminance and its tracked corners;
    ``process(rgb)`` returns the TwoViewReport for the pair. The
    first look after construction (or a resolution change) only seeds
    the tracker — a single view carries no geometry, and the report
    says so (``kind == ""``).

    Thread-safe: ``see()`` runs on both the volition workers and the
    conversation path, so the tracker state is lock-guarded.
    """

    def __init__(
        self,
        max_dim: int = _WORK_DIM,
        n_points: int = _MAX_POINTS,
        seed: int | None = None,
    ) -> None:
        """Initialize tracker state at the requested working scale."""
        self._max_dim = max_dim
        self._n_points = n_points
        self._rng = np.random.default_rng(seed)
        self._prev: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Drop the previous view — the next look opens a new pair."""
        with self._lock:
            self._prev = None
            self._prev_pts = None

    def process(self, rgb: np.ndarray) -> TwoViewReport:
        """Relate this look to the last and report what moved."""
        with self._lock:
            return self._process(rgb)

    def _process(self, rgb: np.ndarray) -> TwoViewReport:
        gray, _scale = _shrink(_to_luminance(rgb), self._max_dim)
        prev = self._prev
        prev_pts = self._prev_pts

        if prev is None or prev_pts is None or prev.shape != gray.shape:
            self._seed(gray)
            return TwoViewReport(n_tracks=0)

        pts2, ok, resid = track_points(prev, gray, prev_pts)
        # Forward–backward check: only a track that round-trips is a
        # correspondence — one-way "matches" are look-alike texture.
        if ok.any():
            back, _, _ = track_points(gray, prev, pts2[ok])
            fb = np.zeros(len(ok), dtype=bool)
            fb[ok] = np.linalg.norm(back - prev_pts[ok], axis=1) < _FB_MAX_PX
            ok &= fb
        kept = int(ok.sum())
        report = TwoViewReport(n_tracks=kept)

        if kept < _MIN_TRACKS:
            report.kind = "unresolved"
            self._seed(gray)
            return report

        disp = np.linalg.norm(pts2[ok] - prev_pts[ok], axis=1)
        if float(np.median(disp)) < _MOTION_EPS_PX:
            # No baseline: the eye held still on a still scene. The
            # tracks still carry forward — stillness is a real read.
            report.kind = "still"
            report.moved = False
            report.n_inliers = kept
            report.inlier_ratio = 1.0
            self._carry(gray, pts2, ok)
            return report

        self._geometry(gray, prev_pts[ok], pts2[ok], resid[ok], report)
        return report

    def _seed(self, gray: np.ndarray) -> None:
        """Seed the tracker from a fresh view."""
        self._prev = gray
        self._prev_pts = detect_corners(gray, self._n_points)

    def _carry(self, gray: np.ndarray, pts2: np.ndarray, ok: np.ndarray) -> None:
        """Advance the track set — keep followed points, reseed when
        the set thins out."""
        self._prev = gray
        survived = pts2[ok]
        self._prev_pts = (
            survived
            if len(survived) >= self._n_points // 2
            else detect_corners(gray, self._n_points)
        )

    def _geometry(
        self,
        gray: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        res: np.ndarray,
        report: TwoViewReport,
    ) -> None:
        """Estimate the two-view model of the moved pair.

        p1, p2, res are the survived tracks' previous positions, new
        positions, and patch residuals — all (y, x) work-scale.
        """
        # Geometry works in (x, y); tracking works in (y, x).
        xy1 = p1[:, ::-1]
        xy2 = p2[:, ::-1]
        h, w = gray.shape
        k = camera_matrix(h, w)
        clean = res < _MOVER_MAX_RESID

        # The baseline lives in the tracks that actually displaced —
        # a near-stationary correspondence satisfies any model for
        # free and triangulates "in front" under any pose. Both
        # candidate models are fit on the moved subset, then judged
        # on the whole field: a stationary phantom track is not a
        # true correspondence and counts against the model.
        disp = np.linalg.norm(p2 - p1, axis=1)
        moved = disp >= _GEOM_MIN_DISP_PX
        if int(moved.sum()) < 8:
            report.kind = "unresolved"
            report.moved = True
            self._seed(gray)
            return

        f, inl, hg, hin = self._fit_models(xy1, xy2, moved, len(p1))

        n = max(len(p1), 1)
        report.n_inliers = int(inl.sum())
        report.inlier_ratio = float(inl.sum()) / n
        # A rotation's field is a homography — if H explains the field
        # as well as F does, there is no parallax and the epipolar
        # "baseline" is spurious. F needs a real support edge to win.
        h_covers = hin.sum() >= inl.sum() - int(0.10 * n)
        if f is None or report.inlier_ratio < _MIN_INLIER_RATIO or h_covers:
            if hg is not None and hin.sum() >= _MIN_INLIER_RATIO * n:
                self._turn(report, hg, hin, p1, p2, clean & moved, k)
                self._carry_ok(gray, p1, p2, hin)
                return
            report.kind = "unresolved"
            report.moved = True
            self._movers(report, inl, clean & moved, p1)
            self._seed(gray)
            return

        self._travel(report, gray, f, k, xy1, xy2, p1, p2, inl, moved, clean)

    def _fit_models(
        self,
        xy1: np.ndarray,
        xy2: np.ndarray,
        moved: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, np.ndarray]:
        """Fit F and H on the moved subset; score both on the full field.

        Estimate F where the measurements live — pixel space — then
        lift to the calibrated manifold: E = proj(KᵀFK). Direct
        estimation in normalized coordinates rescales the noise
        anisotropically and costs ~25° of heading here.
        """
        g1, g2 = xy1[moved], xy2[moved]
        f, _ = estimate_fundamental(g1, g2, rng=self._rng)
        hg, _ = estimate_homography(g1, g2, rng=self._rng)
        inl = (
            sampson_sq(f, xy1, xy2) < _SAMPSON_THRESH_PX**2
            if f is not None
            else np.zeros(n, dtype=bool)
        )
        hin = (
            transfer_error_sq(hg, xy1, xy2) < _H_INLIER_PX**2
            if hg is not None
            else np.zeros(n, dtype=bool)
        )
        return f, inl, hg, hin

    def _travel(
        self,
        report: TwoViewReport,
        gray: np.ndarray,
        f: np.ndarray,
        k: np.ndarray,
        xy1: np.ndarray,
        xy2: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        inl: np.ndarray,
        moved: np.ndarray,
        clean: np.ndarray,
    ) -> None:
        """Lift F onto the essential manifold and decompose into pose."""
        k_inv = np.linalg.inv(k)
        e = project_to_essential(essential_from_fundamental(f, k))
        report.essential = e
        report.fundamental = f
        sd = sampson_sq(f, xy1[inl], xy2[inl])
        report.mean_residual = float(np.sqrt(sd.mean())) if inl.any() else 0.0
        # Cheirality is proved only on tracks that moved — a static
        # point's two rays coincide, so it sits "in front" of any pose.
        tri = inl & moved
        p1n = (_homog(xy1[tri]) @ k_inv.T)[:, :2]
        p2n = (_homog(xy2[tri]) @ k_inv.T)[:, :2]
        r, t, front = recover_pose(e, p1n, p2n)
        if len(front) and front.mean() > 0.3:
            r, t = refine_pose(r, t, p1n[front], p2n[front])
        cheirality = float(front.mean()) if len(front) else 0.0
        report.rotation = r
        report.rotation_deg = _rotation_deg(r)
        e1, _e2 = epipoles(f)
        if abs(e1[2]) > 1e-9:
            report.epipole = (float(e1[0] / e1[2]), float(e1[1] / e1[2]))
        if cheirality >= _CHEIRALITY_MIN:
            report.kind = "travel"
            report.moved = True
            # Physical heading: with x₂ = Rx₁ + t the eye itself sits
            # at C = −Rᵀt in first-view coordinates — t is the world
            # sliding backward past the eye, C is where she went.
            c = -(r.T @ t)
            report.translation = c
            report.direction = _direction_word(c)
            report.heading_deg = math.degrees(math.acos(max(-1.0, min(1.0, abs(float(c[2]))))))
            # The full compass read: azimuth around her up-axis —
            # atan2 on (right, through-view) so 0° is dead ahead,
            # 90° right, 180° back — and elevation off the eye-line
            # (camera y is down, so −y is up). Head-direction
            # cells tile exactly this angle.
            report.bearing_deg = (
                math.degrees(math.atan2(float(c[0]), float(c[2]))) + 360.0
            ) % 360.0
            report.elevation_deg = math.degrees(
                math.atan2(-float(c[1]), math.hypot(float(c[0]), float(c[2])))
            )
            self._depth(report, p1n, p2n, r, t, front, k)
        else:
            # A coherent E whose world won't sit in front of the
            # eyes is a turn wearing a translation it can't
            # support — report the swing, not a false heading.
            report.kind = "turn"
            report.moved = True
            report.translation = None
            report.direction = _turn_word(p1, p2, inl)
        self._movers(report, inl, clean & moved, p1)
        self._carry_ok(gray, p1, p2, inl)

    def _turn(
        self,
        report: TwoViewReport,
        hg: np.ndarray,
        hin: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        clean: np.ndarray,
        k: np.ndarray,
    ) -> None:
        """Report a rotation: the field swung, nothing came nearer."""
        report.kind = "turn"
        report.moved = True
        report.n_inliers = int(hin.sum())
        report.inlier_ratio = float(hin.sum()) / max(len(p1), 1)
        r = rotation_from_homography(hg, k)
        report.rotation = r
        report.rotation_deg = _rotation_deg(r)
        report.direction = _turn_word(p1, p2, hin)
        report.fundamental = None
        self._movers(report, hin, clean, p1)

    def _carry_ok(
        self,
        gray: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        inl: np.ndarray,
    ) -> None:
        """Advance the tracker keeping the model-consistent tracks."""
        self._prev = gray
        survived = p2[inl] if inl.any() else p2
        if len(survived) < self._n_points // 2:
            self._prev_pts = detect_corners(gray, self._n_points)
        else:
            self._prev_pts = survived

    def _depth(
        self,
        report: TwoViewReport,
        p1n: np.ndarray,
        p2n: np.ndarray,
        r: np.ndarray,
        t: np.ndarray,
        front: np.ndarray,
        k: np.ndarray,
    ) -> None:
        """Parallax depth: nearer structure sweeps further across the eye."""
        p1_cam = np.hstack([np.eye(3), np.zeros((3, 1))])
        p2_cam = np.hstack([r, t[:, None]])
        pts3d, front_mask = triangulate(p1_cam, p2_cam, p1n, p2n)
        good = front & front_mask
        if not good.any():
            return
        depth = pts3d[good, 2]
        finite = np.isfinite(depth) & (depth > 0)
        if not finite.any():
            return
        idx = np.where(good)[0][finite]
        depth = depth[finite]
        med = float(np.median(depth))
        report.median_depth = med
        report.near_fraction = float((depth < med).mean())
        # The third of the view where the strongest parallax sits —
        # nearest quarter of structure, by pixel column.
        near = idx[depth < med]
        if len(near):
            cols = p1n[near, 0] * float(k[0, 0]) + float(k[0, 2])
            w_px = float(k[0, 2]) * 2.0
            third = w_px / 3.0
            frac = (
                float((cols < third).mean()),
                float(((cols >= third) & (cols < 2 * third)).mean()),
                float((cols >= 2 * third).mean()),
            )
            report.nearest_region = ("left", "center", "right")[int(np.argmax(frac))]

    @staticmethod
    def _movers(
        report: TwoViewReport,
        inl: np.ndarray,
        clean: np.ndarray,
        p1: np.ndarray,
    ) -> None:
        """Count structure that tracked cleanly but broke the geometry —
        things moving on their own, not failed follows."""
        mover_mask = (~inl) & clean
        report.movers = int(mover_mask.sum())
        if mover_mask.any():
            cy, cx = p1[mover_mask].mean(axis=0)
            report.mover_centroid = (float(cy), float(cx))

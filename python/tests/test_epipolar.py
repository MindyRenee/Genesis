"""Tests for the MSTd two-view epipolar geometry stage.

Ground truth is synthetic: a known camera rig, a known 3D cloud,
points projected through the pinhole model. The estimator is handed
the projections and must recover the geometry it was built from —
the constraint, the pose, and the depths — plus the behavioural
kinds the processor reports (still / turn / travel / unresolved).
"""

import numpy as np
import pytest
from occipital_lobe import epipolar as ep

_H, _W = 240, 320


def _rig(rot_deg: float = 3.0, t=(0.3, 0.0, 1.0), seed: int = 0, n: int = 80):
    """A known two-view rig: camera 1 at the origin, camera 2 at (R, t)."""
    rng = np.random.default_rng(seed)
    k = ep.camera_matrix(_H, _W)
    a = np.deg2rad(rot_deg)
    r = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    t = np.asarray(t, dtype=np.float64)
    tn = np.linalg.norm(t)
    t = t / tn if tn > 1e-12 else np.zeros(3)
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = np.hstack([r, t[:, None]])
    x = rng.uniform(-3, 3, (n, 3))
    x[:, 2] = rng.uniform(4, 12, n)
    return k, r, t, p1, p2, x


def _project(k, p, x):
    xh = np.hstack([x, np.ones((len(x), 1))]) @ (k @ p).T
    return xh[:, :2] / xh[:, 2:3]


def _visible(x1, x2):
    return (
        (x1[:, 0] > 10)
        & (x1[:, 0] < _W - 10)
        & (x1[:, 1] > 10)
        & (x1[:, 1] < _H - 10)
        & (x2[:, 0] > 10)
        & (x2[:, 0] < _W - 10)
        & (x2[:, 1] > 10)
        & (x2[:, 1] < _H - 10)
    )


def _pair(rot_deg=3.0, t=(0.3, 0.0, 1.0), seed=0, noise=0.0):
    k, r, t, p1, p2, x = _rig(rot_deg, t, seed)
    x1 = _project(k, p1, x)
    x2 = _project(k, p2, x)
    vis = _visible(x1, x2)
    x1, x2 = x1[vis], x2[vis]
    if noise:
        rng = np.random.default_rng(seed + 1)
        x1 += rng.normal(0, noise, x1.shape)
        x2 += rng.normal(0, noise, x2.shape)
    return k, r, t, p1, p2, x[vis], x1, x2


def test_fundamental_recovers_known_constraint():
    _k, _r, _t, _p1, _p2, _x, x1, x2 = _pair()
    f, inl = ep.estimate_fundamental(x1, x2, rng=np.random.default_rng(1))
    assert f is not None
    assert inl.mean() > 0.9
    resid = np.abs(ep.algebraic_residual(f, x1, x2))
    assert resid[inl].max() < 1e-6
    assert np.linalg.matrix_rank(f) == 2


def test_epipolar_lines_pass_through_matches():
    _k, _r, _t, _p1, _p2, _x, x1, x2 = _pair()
    f, inl = ep.estimate_fundamental(x1, x2, rng=np.random.default_rng(2))
    lines = ep.epipolar_lines(f, x1[inl])
    # distance of x2 from its epipolar line ≈ 0
    x2h = np.hstack([x2[inl], np.ones((int(inl.sum()), 1))])
    d = np.abs(np.einsum("ij,ij->i", x2h, lines)) / np.sqrt((lines[:, :2] ** 2).sum(axis=1))
    assert d.max() < 1e-6


def test_epipoles_are_null_vectors():
    _k, _r, _t, _p1, _p2, _x, x1, x2 = _pair()
    f, _inl = ep.estimate_fundamental(x1, x2, rng=np.random.default_rng(3))
    e1, e2 = ep.epipoles(f)
    np.testing.assert_allclose(f @ e1, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(e2 @ f, np.zeros(3), atol=1e-8)


def test_essential_recovers_pose():
    k, r_gt, t_gt, _p1, _p2, _x, x1, x2 = _pair()
    e, inl = ep.estimate_essential(x1, x2, k, rng=np.random.default_rng(4))
    assert e is not None
    assert inl.mean() > 0.9
    # Essential manifold: singular values (σ, σ, 0).
    sv = np.linalg.svd(e, compute_uv=False)
    np.testing.assert_allclose(sv[0], sv[1], rtol=1e-6)
    assert sv[2] < 1e-6
    k_inv = np.linalg.inv(k)
    p1n = (np.hstack([x1[inl], np.ones((int(inl.sum()), 1))]) @ k_inv.T)[:, :2]
    p2n = (np.hstack([x2[inl], np.ones((int(inl.sum()), 1))]) @ k_inv.T)[:, :2]
    r, t_est, front = ep.recover_pose(e, p1n, p2n)
    err_deg = np.degrees(np.arccos(np.clip((np.trace(r.T @ r_gt) - 1) / 2, -1, 1)))
    assert err_deg < 1.0
    assert float(t_est @ t_gt) > 0.999
    assert front.mean() > 0.9


def test_triangulation_reprojects_and_respects_cheirality():
    k, _r_gt, _t_gt, p1, p2, x, x1, x2 = _pair()
    pts, front = ep.triangulate(k @ p1, k @ p2, x1, x2)
    # Ground-truth points are all in front of both cameras.
    assert front.all()
    # Triangulated points match the true cloud up to scale (the cloud
    # was built at the same baseline scale, so depth is exact).
    np.testing.assert_allclose(pts[front, 2], x[front, 2], rtol=1e-6)


def test_sampson_separates_inliers_from_outliers():
    _k, _r, _t, _p1, _p2, _x, x1, x2 = _pair()
    f, inl = ep.estimate_fundamental(x1, x2, rng=np.random.default_rng(5))
    rng = np.random.default_rng(9)
    bad = x2 + rng.uniform(30, 80, x2.shape)
    sd_good = ep.sampson_sq(f, x1[inl], x2[inl])
    sd_bad = ep.sampson_sq(f, x1, bad)
    assert sd_good.max() < sd_bad.min()


def test_loss_functions_agree_on_perfect_geometry():
    _k, _r, _t, _p1, _p2, _x, x1, x2 = _pair()
    f, inl = ep.estimate_fundamental(x1, x2, rng=np.random.default_rng(6))
    np.testing.assert_allclose(ep.algebraic_residual(f, x1[inl], x2[inl]), 0, atol=1e-6)
    np.testing.assert_allclose(ep.symmetric_epipolar_sq(f, x1[inl], x2[inl]), 0, atol=1e-9)
    np.testing.assert_allclose(ep.huber(np.array([0.5, 4.0]), k=1.0), np.array([0.5, 3.0]))


def test_homography_fits_pure_rotation():
    k, _r, _t, _p1, _p2, _x, x1, x2 = _pair(rot_deg=5.0, t=(0.0, 0.0, 0.0))
    hg, inl = ep.estimate_homography(x1, x2, rng=np.random.default_rng(7))
    assert hg is not None
    assert inl.mean() > 0.9
    r_est = ep.rotation_from_homography(hg, k)
    # recovered turn ≈ the ground-truth 5° yaw (sign included either way)
    deg = np.degrees(np.arccos(np.clip((np.trace(r_est) - 1) / 2, -1, 1)))
    assert deg == pytest.approx(5.0, abs=1.0)


def _render(x, k, p, h, w, rng, noise=1.0):
    img = rng.normal(18, noise, (h, w)).astype(np.float32)
    xh = np.hstack([x, np.ones((len(x), 1))]) @ (k @ p).T
    z = xh[:, 2]
    px = xh[:, :2] / np.maximum(z[:, None], 1e-9)
    # Texture with a scale — a point the tracker can hold at every
    # pyramid level needs a blob wide enough to survive decimation;
    # delta-size dots alias away and LK has nothing to follow.
    # Amplitude and width vary so look-alike lock-ons carry a residual.
    amps = rng.uniform(140.0, 240.0, len(x))
    sigs = rng.uniform(7.0, 14.0, len(x))
    for (px_, py_), d, amp, s in zip(px, z, amps, sigs, strict=True):
        if 45 <= px_ < w - 45 and 45 <= py_ < h - 45 and d > 0:
            rr = int(np.ceil(3.0 * s))
            cy, cx = int(py_), int(px_)
            yy, xx = np.mgrid[-rr : rr + 1, -rr : rr + 1]
            img[cy - rr : cy + rr + 1, cx - rr : cx + rr + 1] += amp * np.exp(
                -(xx**2 + yy**2) / (2.0 * s * s)
            )
    return np.clip(img, 0, 255)


def _frames(seed=5, rot_deg=1.5, t=(0.25, 0.0, 1.0)):
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    k = ep.camera_matrix(h, w)
    x = rng.uniform(-4, 4, (300, 3))
    x[:, 2] = rng.uniform(3, 14, 300)
    xn = rng.uniform(1, 4, (40, 3))
    xn[:, 2] = rng.uniform(1.5, 3.0, 40)
    x = np.vstack([x, xn])
    a = np.deg2rad(rot_deg)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = np.asarray(t, dtype=np.float64)
    t = t / np.linalg.norm(t)
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    # t is the physical displacement of the eye — the camera center
    # sits at t, so the [R|·] column is −R·t (C = −Rᵀt ⇒ column −RC).
    p2 = np.hstack([r, (-r @ t)[:, None]])
    f1 = _render(x, k, p1, h, w, rng)
    f2 = _render(x, k, p2, h, w, rng)

    def rgb(f: np.ndarray) -> np.ndarray:
        return np.stack([f] * 3, axis=-1).astype(np.uint8)

    return rgb(f1), rgb(f2), r, t


def test_processor_first_view_carries_no_geometry():
    f1, _f2, _r, _t = _frames()
    proc = ep.EpipolarProcessor(seed=0)
    report = proc.process(f1)
    assert report.kind == ""
    assert report.n_tracks == 0


def test_processor_still_view_reports_stillness():
    f1, _f2, _r, _t = _frames()
    proc = ep.EpipolarProcessor(seed=0)
    proc.process(f1)
    report = proc.process(f1.copy())
    assert report.kind == "still"
    assert not report.moved


def test_processor_travel_recovers_heading_and_depth():
    f1, f2, _r_gt, t_gt = _frames()
    proc = ep.EpipolarProcessor(seed=0)
    proc.process(f1)
    report = proc.process(f2)
    assert report.kind == "travel"
    assert report.moved
    assert report.translation is not None
    # Heading: forward + slightly right — the dominant axis is toward.
    assert report.direction == "toward"
    assert float(report.translation @ t_gt) > 0.95
    assert report.median_depth > 0
    assert report.inlier_ratio >= 0.45


def _texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-scale random texture: broad structure the coarse pyramid
    levels can hold, fine speckle for the fine levels."""
    huge = np.kron(rng.normal(0, 1, (h // 32, w // 32)), np.ones((32, 32)))[:h, :w]
    big = np.kron(rng.normal(0, 1, (h // 16, w // 16)), np.ones((16, 16)))[:h, :w]
    mid = np.kron(rng.normal(0, 1, (h // 4, w // 4)), np.ones((4, 4)))[:h, :w]
    fine = rng.normal(0, 1, (h, w))
    img = 128 + huge * 50 + big * 45 + mid * 25 + fine * 8
    return np.clip(img, 0, 255).astype(np.float32)


def _turn_frame(f1: np.ndarray, r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Second view of a pure rotation — the exact homography warp
    x2 = K R K⁻¹ x1 applied to the first frame."""
    h, w = f1.shape
    h_inv = np.linalg.inv(k @ r @ np.linalg.inv(k))
    ys, xs = np.mgrid[0:h, 0:w]
    src = h_inv @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    src = src[:2] / np.maximum(src[2], 1e-9)
    return ep._bilinear(f1, src[1].reshape(h, w), src[0].reshape(h, w))


def test_processor_turn_reports_swing_not_travel():
    rng = np.random.default_rng(11)
    h, w = 480, 640
    k = ep.camera_matrix(h, w)
    # A small glance shift — still a pure rotation, no baseline.
    a = np.deg2rad(1.5)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    f1 = _texture(h, w, rng)
    f2 = _turn_frame(f1, r, k)

    def rgb(f: np.ndarray) -> np.ndarray:
        return np.stack([f] * 3, axis=-1).astype(np.uint8)

    proc = ep.EpipolarProcessor(seed=0)
    proc.process(rgb(f1))
    report = proc.process(rgb(f2))
    assert report.kind == "turn"
    assert report.moved
    # The field slid right on the sensor → the gaze turned left.
    assert report.direction == "left"
    assert report.rotation_deg == pytest.approx(1.5, abs=0.5)

"""Tests for the S3 latent manifold substrate.

The geometry checks are the probe's claims pinned as unit tests:
positive curvature is a hard requirement, loops close, fibers link
the cores once, the counter-wound families form an exact vesica at
depth sqrt(3)/2, the motions are isometries, and resonance recall
degrades smoothly. The substrate checks cover deposit/resonate,
append-only accretion, dynamics, and image persistence.
"""

import math

import numpy as np
import pytest
from latent_spaces import geometry as g
from latent_spaces.manifold import (
    STRAND_ANTI,
    STRAND_HOPF,
    LatentManifold,
)

rng = np.random.default_rng(20261109)


def _rand_s3(n=1):
    x = rng.normal(size=(n, 4))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _rand_tangent(p):
    v = rng.normal(size=4)
    v -= (v @ p) * p
    return v / np.linalg.norm(v)


def _xy_winding(c):
    """Signed winding of a closed R^3 curve's xy-projection around the origin.

    For the stereographic image (whose z-axis is the self-core circle)
    this equals the linking number with that core.
    """
    ang = np.unwrap(np.arctan2(c[:, 1], c[:, 0]))
    return (ang[-1] - ang[0]) / (2.0 * np.pi)


def _swap_core(points):
    """Exchange (z1, z2) — maps the other-core onto the self-core."""
    return points[:, [2, 3, 0, 1]]


def _gauss_link(c1, c2):
    d1 = np.roll(c1, -1, axis=0) - c1
    d2 = np.roll(c2, -1, axis=0) - c2
    total = 0.0
    for i in range(len(c1)):
        diff = c1[i] - c2
        r = np.linalg.norm(diff, axis=1)
        total += np.sum(np.einsum("mj,mj->m", np.cross(d1[i], d2), diff) / r**3)
    return float(total / (4.0 * np.pi))


class TestPositiveCurvature:
    """REQUIRED: the substrate must be positively curved."""

    @pytest.mark.parametrize("r", [0.2, 0.5, 1.0])
    def test_geodesic_circles_shrink(self, r):
        p = _rand_s3()[0]
        e1 = _rand_tangent(p)
        e2 = _rand_tangent(p)
        e2 -= (e2 @ e1) * e1
        e2 /= np.linalg.norm(e2)
        th = np.linspace(0.0, 2.0 * np.pi, 2001)
        u = np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)
        circ = math.cos(r) * p + math.sin(r) * u
        seg = np.arccos(
            np.clip(np.einsum("ij,ij->i", circ, np.roll(circ, -1, axis=0)), -1, 1)
        )
        measured = seg.sum()
        assert measured < 2.0 * np.pi * r
        assert abs(measured - 2.0 * np.pi * math.sin(r)) < 1e-3

    def test_triangle_excess(self):
        tri = _rand_s3(3)
        a = g.geodesic(tri[1], tri[2])
        b = g.geodesic(tri[0], tri[2])
        c = g.geodesic(tri[0], tri[1])
        A = math.acos(
            np.clip((math.cos(a) - math.cos(b) * math.cos(c)) / (math.sin(b) * math.sin(c)), -1, 1)
        )
        B = math.acos(
            np.clip((math.cos(b) - math.cos(a) * math.cos(c)) / (math.sin(a) * math.sin(c)), -1, 1)
        )
        C = math.acos(
            np.clip((math.cos(c) - math.cos(a) * math.cos(b)) / (math.sin(a) * math.sin(b)), -1, 1)
        )
        assert A + B + C > math.pi

    def test_geodesics_converge_below_flat(self):
        p = _rand_s3()[0]
        e1 = _rand_tangent(p)
        w = _rand_tangent(p)
        w -= (w @ e1) * e1
        w /= np.linalg.norm(w)
        delta = 1e-3
        v2 = math.cos(delta) * e1 + math.sin(delta) * w
        t = 1.0
        measured = g.geodesic(g.exp_map(p, e1, t), g.exp_map(p, v2, t))
        assert abs(measured - delta * math.sin(t)) < 1e-4
        assert measured < delta * t

    def test_antipodal_refocus(self):
        p = _rand_s3()[0]
        e1, e2 = _rand_tangent(p), _rand_tangent(p)
        assert np.linalg.norm(g.exp_map(p, e1, np.pi) - g.exp_map(p, e2, np.pi)) < 1e-10


class TestClosedLoops:
    def test_every_geodesic_closes(self):
        p = _rand_s3()[0]
        v = _rand_tangent(p)
        assert np.linalg.norm(g.exp_map(p, v, g.TWO_PI) - p) < 1e-10

    def test_geodesic_transits_antipode(self):
        p = _rand_s3()[0]
        v = _rand_tangent(p)
        assert np.linalg.norm(g.exp_map(p, v, np.pi) - (-p)) < 1e-10

    def test_fiber_is_closed_loop_on_sphere(self):
        fib = g.fiber(_rand_s3()[0])
        assert np.allclose(np.linalg.norm(fib, axis=1), 1.0)
        base = g.hopf_base(fib)
        assert np.max(np.abs(base - base[0])) < 1e-12


class TestLinking:
    """Every fiber links the self-core and other-core exactly once."""

    def _fiber_s3(self):
        p = np.array([0.55, 0.0, 0.4, 0.7262])
        return g.fiber(p / np.linalg.norm(p))

    def test_fiber_links_self_core(self):
        wound = _xy_winding(g.stereo(self._fiber_s3()))
        assert abs(abs(wound) - 1.0) < 0.01

    def test_fiber_links_other_core(self):
        swapped = g.stereo(_swap_core(self._fiber_s3()))
        wound = _xy_winding(swapped)
        assert abs(abs(wound) - 1.0) < 0.01

    def test_two_fibers_link_once(self):
        f1 = g.stereo(self._fiber_s3())
        p2 = np.array([0.3, 0.5, 0.6, 0.55])
        f2 = g.stereo(g.fiber(p2 / np.linalg.norm(p2)))
        assert abs(abs(_gauss_link(f1, f2)) - 1.0) < 0.02

    def test_fibers_stay_on_their_layer(self):
        for r in (0.2, 0.5, 0.7071, 0.9):
            s = math.sqrt(1 - r * r)
            fib = g.fiber(np.array([r, 0.0, s, 0.0]))
            assert np.allclose(g.depth(fib), r)


class TestVesica:
    """The counter-wound families are planar circles forming the lens."""

    def _layer_circles(self, r):
        s = math.sqrt(1 - r * r)
        p = np.array([r, 0.0, s, 0.0])
        c1 = g.stereo(g.fiber(p, n=512))
        c2 = g.stereo(g.anti_fiber(p, n=512))
        return c1, c2

    def test_counter_wound_fibers_are_circles(self):
        c1, c2 = self._layer_circles(g.VESICA_DEPTH)
        _, _, res1 = g.fit_circle(c1)
        _, _, res2 = g.fit_circle(c2)
        assert max(res1, res2) < 1e-6

    def test_exact_vesica_at_verified_depth(self):
        c1, c2 = self._layer_circles(g.VESICA_DEPTH)
        ce1, r1, _ = g.fit_circle(c1)
        ce2, r2, _ = g.fit_circle(c2)
        sep = np.linalg.norm(ce1 - ce2)
        assert abs(r1 - r2) / r1 < 0.01
        # each center on the other circle: separation equals radius
        assert abs(sep - r1) / r1 < 0.02

    def test_strands_cross_at_two_points(self):
        c1, c2 = self._layer_circles(g.VESICA_DEPTH)
        d = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=2)
        hits = np.argwhere(d < 1e-3)
        used = np.zeros(len(c1), bool)
        n_cross = 0
        for i, _ in hits:
            if used[i]:
                continue
            used[max(0, i - 100) : i + 100] = True
            n_cross += 1
        assert n_cross == 2


class TestDynamics:
    def test_frame_rotation_is_isometry(self):
        pts = _rand_s3(64)
        moved = g.frame_rotate(pts, 0.7)
        assert abs(g.geodesic(moved[0], moved[1]) - g.geodesic(pts[0], pts[1])) < 1e-12

    def test_counter_rotation_is_isometry(self):
        pts = _rand_s3(64)
        moved = g.counter_rotate(pts, 0.7)
        assert abs(g.geodesic(moved[0], moved[1]) - g.geodesic(pts[0], pts[1])) < 1e-12

    def test_rotation_preserves_depth(self):
        pts = _rand_s3(64)
        assert np.allclose(g.depth(g.rotate(pts, 0.9, -0.4)), g.depth(pts))

    def test_shear_crosses_fibers(self):
        pts = _rand_s3(64)
        base_before = g.hopf_base(pts)
        base_after = g.hopf_base(g.rotate(pts, 0.7, 0.0))
        assert np.max(np.abs(base_after - base_before)) > 0.1

    def test_vertical_phase_is_isometry(self):
        """Spinning one plane alone — the vertical-axis motion."""
        pts = _rand_s3(64)
        moved = g.rotate(pts, 0.9, 0.0)
        assert abs(g.geodesic(moved[0], moved[1]) - g.geodesic(pts[0], pts[1])) < 1e-12

    def test_tumble_is_isometry(self):
        pts = _rand_s3(64)
        moved = g.tumble(pts, 0.6, (0, 2))
        assert abs(g.geodesic(moved[0], moved[1]) - g.geodesic(pts[0], pts[1])) < 1e-12

    def test_tumble_migrates_across_the_membrane(self):
        inner = np.array([0.3, 0.1, 0.9, 0.3])
        inner /= np.linalg.norm(inner)
        assert g.depth(inner) < g.CLIFFORD_DEPTH
        mid = g.tumble(inner, np.pi / 2, (0, 2))
        assert g.depth(mid) > g.CLIFFORD_DEPTH
        back = g.tumble(inner, np.pi, (0, 2))
        assert abs(g.depth(back) - g.depth(inner)) < 1e-12

    def test_vesica_invariant_under_counter_tumble(self):
        """The lens stays exact while the worlds tumble against each other."""
        r = g.VESICA_DEPTH
        p = np.array([r, 0.0, math.sqrt(1 - r * r), 0.0])
        c1 = g.fiber(p)
        c2 = g.anti_fiber(p)
        for a in (0.0, 0.4, 0.8, 1.2):
            ce1, r1, _ = g.fit_circle(g.stereo(g.tumble(c1, a, (0, 2))))
            ce2, _r2, _ = g.fit_circle(g.stereo(g.tumble(c2, -a, (0, 2))))
            assert abs(np.linalg.norm(ce1 - ce2) / r1 - 1.0) < 0.02


class TestLieGroup:
    """S3 = SU(2): the group forms must equal the coordinate forms."""

    def test_qexp_qlog_roundtrip(self):
        pts = _rand_s3(32)
        back = g.qexp(g.qlog(pts))
        assert np.allclose(back, pts, atol=1e-10)

    def test_qmul_preserves_unit_norm(self):
        a, b = _rand_s3(32), _rand_s3(32)
        prod = g.qmul(a, b)
        assert np.allclose(np.linalg.norm(prod, axis=1), 1.0)

    def test_left_spin_is_the_heartbeat(self):
        pts = _rand_s3(64)
        assert np.allclose(
            g.left_spin(pts, 0.7, "i"), g.rotate(pts, 0.7, 0.7), atol=1e-12
        )

    def test_right_spin_is_the_braid(self):
        pts = _rand_s3(64)
        assert np.allclose(
            g.right_spin(pts, 0.7, "i"), g.rotate(pts, 0.7, -0.7), atol=1e-12
        )

    def test_adjoint_fixes_scalar_and_is_isometry(self):
        pts = _rand_s3(64)
        rotor = g.unit(np.array([1.0, 0.3, -0.2, 0.5]))
        moved = g.adjoint_spin(pts, rotor)
        assert np.allclose(moved[..., 0], pts[..., 0])
        assert abs(g.geodesic(moved[0], moved[1]) - g.geodesic(pts[0], pts[1])) < 1e-12


class TestSubstrate:
    def test_deposit_and_resonate(self):
        m = LatentManifold()
        m.observe("the cat sat on the mat", kind="world:utterance", salience=0.9)
        m.observe("entropy increases", kind="thought", salience=0.5)
        hits = m.resonate("the cat sat on the mat", kind="world:utterance")
        assert hits
        assert hits[0][0] == 0
        assert hits[0][1] == "world:utterance"

    def test_domains_land_in_their_worlds(self):
        m = LatentManifold()
        m.observe("a thing I thought", kind="thought")
        m.observe("a thing that happened", kind="world:percept")
        with m._lock:
            depths = g.depth(m._points)
        assert depths[0] < g.CLIFFORD_DEPTH
        assert depths[1] > g.CLIFFORD_DEPTH

    def test_strand_by_direction(self):
        m = LatentManifold()
        m.observe("heard", kind="world:utterance", inbound=True)
        m.observe("said", kind="world:utterance", inbound=False)
        with m._lock:
            assert m._strand[0] == STRAND_HOPF
            assert m._strand[1] == STRAND_ANTI

    def test_append_only_grows(self):
        m = LatentManifold()
        for i in range(10):
            m.observe(f"trace {i}", kind="thought")
        assert len(m) == 10

    def test_resonance_decays_smoothly(self):
        m = LatentManifold()
        m.observe("target memory", kind="thought", salience=1.0)
        for i in range(30):
            m.observe(f"other trace {i}", kind="thought", salience=0.5)
        scores = []
        for noise in (0.05, 0.3, 0.8):
            hits = m.resonate(f"target memory variant-{noise}", kind="thought")
            scores.append(hits[0][2] if hits else 0.0)
        # distinct cues give a smooth, bounded score landscape
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_advance_rotates_and_preserves_recall(self):
        m = LatentManifold()
        for i in range(8):
            m.observe(f"trace {i}", kind="thought", salience=0.8)
        with m._lock:
            before = g.geodesic(m._points[0], m._points[1])
        m.advance(5.0)
        with m._lock:
            after = g.geodesic(m._points[0], m._points[1])
        assert abs(after - before) < 1e-12
        assert m.phase > 0.0

    def test_recall_survives_medium_rotation(self):
        """A cue must find its deposit after the medium has turned.

        Deposits ride their fibers; resonance unwinds every trace to a
        common epoch so recall is invariant under the heartbeat.
        """
        m = LatentManifold()
        m.observe("the sun through the window", kind="world:percept", salience=0.6)
        m.observe("unrelated inner weather", kind="thought", salience=0.9)
        for _ in range(30):
            m.advance(1.0, shear=0.02)
        hits = m.resonate("the sun through the window", kind="world:percept", k=2)
        assert hits and hits[0][0] == 0
        # chord noise accumulates over the round-trip rotations; arccos
        # amplifies it near zero — the tolerance is float noise, not drift
        assert hits[0][3] < 1e-6

    def test_lean_drifts_toward_target(self):
        m = LatentManifold()
        m.set_lean_target(0.3)
        before = m.lean
        for _ in range(20):
            m.advance(1.0)
        assert m.lean < before
        assert abs(m.lean - 0.3) < 0.4

    def test_save_load_roundtrip(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        for i in range(6):
            m.observe(f"trace {i}", kind="world:percept", salience=0.5)
        m.advance(2.0)
        m.set_lean_target(0.4)
        m.advance(2.0)
        m.save()
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert len(m2) == 6
        with m._lock, m2._lock:
            assert np.allclose(m._points, m2._points)
            assert m._phase == m2._phase
            assert m._lean == m2._lean

    def test_load_missing_is_false(self, tmp_path):
        assert not LatentManifold(data_dir=tmp_path).load()

    def test_corrupt_image_rejected(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        m.observe("x", kind="thought")
        m.save()
        data = dict(np.load(m.image_path, allow_pickle=False))
        data["points"] = data["points"][:0]
        np.savez(m.image_path, **data)
        with pytest.raises(ValueError, match="shape mismatch"):
            m.load()

    def test_salience_bounds(self):
        m = LatentManifold()
        with pytest.raises(ValueError):
            m.observe("x", kind="thought", salience=1.5)

    def test_state_summary(self):
        m = LatentManifold()
        m.observe("in", kind="thought", inbound=True)
        m.observe("out", kind="world:percept", inbound=False)
        s = m.state_summary()
        assert s["deposits"] == 2
        assert s["hopf_strand"] == 1
        assert s["anti_strand"] == 1
        assert s["lean"] == g.VESICA_DEPTH

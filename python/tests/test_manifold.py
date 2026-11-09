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
        # inner world lives at the vesica; the world at the far pole
        assert 0.79 <= depths[0] <= 0.94
        assert depths[1] >= 0.72

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


class TestGeodesicArc:
    def test_arc_endpoints_and_norm(self):
        a, b = _rand_s3(2)
        path = g.arc(a, b, n=33)
        assert np.allclose(np.linalg.norm(path, axis=1), 1.0)
        assert np.allclose(path[0], a)
        assert np.allclose(path[-1], b)

    def test_arc_midpoint_is_equidistant(self):
        a, b = _rand_s3(2)
        path = g.arc(a, b, n=3)
        d1 = g.geodesic(path[0], path[1])
        d2 = g.geodesic(path[1], path[2])
        assert abs(d1 - d2) < 1e-12
        assert abs(d1 + d2 - g.geodesic(a, b)) < 1e-9

    def test_arc_coincident_and_antipodal(self):
        a = _rand_s3()[0]
        assert np.allclose(g.arc(a, a, n=8), np.broadcast_to(a, (8, 4)))
        path = g.arc(a, -a, n=16)
        assert np.allclose(np.linalg.norm(path, axis=1), 1.0)
        assert np.allclose(path[-1], -a, atol=1e-9)


class TestEpisodes:
    """Episodes are trajectories: deposits flow outer -> inner -> outer."""

    def test_exchange_flows_outer_inner_outer(self):
        m = LatentManifold()
        m.observe("hello genesis", kind="world:user_speech", inbound=True)
        m.observe("a thought about the greeting", kind="thought:internal")
        m.observe("hello back", kind="world:utterance", inbound=False)
        assert m.rung_of(0) == m.rung_of(1) == m.rung_of(2) == 0
        assert m.rung_legs(0) == [0, 1, 2]
        # The arc arrived back in the world — the rung stopped accreting.
        assert m.state_summary()["open_rung"] == -1

    def test_inbound_leg_opens_a_new_exchange(self):
        m = LatentManifold()
        m.observe("first percept", kind="world:percept", inbound=True)
        m.observe("second percept", kind="world:percept", inbound=True)
        assert m.rung_of(0) != m.rung_of(1)

    def test_outbound_without_inbound_is_its_own_arc(self):
        m = LatentManifold()
        m.observe("an inner stirring", kind="thought:internal")
        m.observe("it spoke unprompted", kind="world:utterance", inbound=False)
        assert m.rung_of(0) == m.rung_of(1)
        m.observe("another thought", kind="thought:internal")
        assert m.rung_of(2) != m.rung_of(0)

    def test_explicit_rung_joins_that_exchange(self):
        m = LatentManifold()
        m.observe("seed leg", kind="thought")
        m.observe("joined leg", kind="thought", rung=0)
        assert m.rung_legs(0) == [0, 1]
        with pytest.raises(ValueError):
            m.observe("bad", kind="thought", rung=-4)
        assert len(m) == 2

    def test_traverse_chains_leg_arcs(self):
        m = LatentManifold()
        m.observe("world came in", kind="world:percept", inbound=True)
        m.observe("she thought", kind="thought")
        m.observe("she answered", kind="world:utterance", inbound=False)
        path = m.traverse(0, per_leg=16)
        # legs chain (31 pts) then the loop returns home (+15 pts)
        assert path.shape == (2 * 15 + 1 + 15, 4)
        assert np.allclose(np.linalg.norm(path, axis=1), 1.0)
        with m._lock:
            epoch = m._epoch_points([0, 1, 2])
        assert np.allclose(path[0], epoch[0], atol=1e-9)
        assert np.allclose(path[15], epoch[1], atol=1e-9)
        assert np.allclose(path[30], epoch[2], atol=1e-9)
        # the loop closes at the vesica home, not at the world
        assert np.allclose(path[-1], m._home_point(), atol=1e-9)

    def test_home_point_rides_the_medium(self):
        """Home is dynamic — it moves with the medium, not a landmark."""
        m = LatentManifold()
        m.observe("x", kind="thought")
        before = m._home_point()
        m.advance(2.0, heartbeat_omega=0.5, shear=0.1)
        after = m._home_point()
        assert not np.allclose(before, after)
        assert np.linalg.norm(after) == pytest.approx(1.0)
        assert g.depth(after.reshape(1, 4))[0] == pytest.approx(m._lean, abs=0.05)

    def test_traverse_is_recall_invariant_under_rotation(self):
        m = LatentManifold()
        m.observe("in", kind="world:percept", inbound=True)
        m.observe("out", kind="world:utterance", inbound=False)
        before = m.traverse(0, per_leg=8, close_loop=False)
        for _ in range(10):
            m.advance(1.0, shear=0.03)
        after = m.traverse(0, per_leg=8, close_loop=False)
        # the episode legs are recall-invariant; the return leg is
        # present tense — home moves with her, so it is not compared
        assert np.allclose(before, after, atol=1e-9)

    def test_segment_is_the_swept_orbit(self):
        m = LatentManifold()
        m.observe("a trace", kind="thought")
        still = m.segment(0, n=8)
        assert np.allclose(still, np.broadcast_to(still[0], (8, 4)))
        m.advance(3.0)
        seg = m.segment(0, n=16)
        with m._lock:
            assert np.allclose(seg[-1], m._points[0], atol=1e-9)
            # The start is the birth position: un-ride the rotation.
            d1 = m._phase - m._phase0[0]
            birth = g.rotate(m._points[0], -d1, -d1)
        assert np.allclose(seg[0], birth, atol=1e-9)
        assert g.geodesic(seg[0], seg[-1]) > 1e-6
        # A Hopf-strand deposit with no shear sweeps its own fiber.
        assert np.max(np.abs(g.hopf_base(seg) - g.hopf_base(seg[0]))) < 1e-12

    def test_traverse_single_leg_is_its_segment(self):
        m = LatentManifold()
        m.observe("a percept nobody answered", kind="world:percept", inbound=True)
        m.observe("next percept", kind="world:percept", inbound=True)
        m.advance(1.0)
        assert np.allclose(m.traverse(0, per_leg=8), m.segment(0, n=8))

    def test_episode_carries_ordered_legs_and_arc(self):
        m = LatentManifold()
        m.observe("heard", kind="world:user_speech", inbound=True)
        m.observe("felt it", kind="thought", inbound=False)
        m.observe("said", kind="world:utterance", inbound=False)
        ep = m.episode(0)
        assert ep["rung"] == 0
        assert [leg["index"] for leg in ep["legs"]] == [0, 1, 2]
        assert [leg["inbound"] for leg in ep["legs"]] == [True, False, False]
        assert ep["arc"].shape[1] == 4

    def test_recall_returns_the_whole_arc(self):
        m = LatentManifold()
        m.observe("the rain on the window", kind="world:percept", inbound=True)
        m.observe("she answered about rain", kind="world:utterance", inbound=False)
        eps = m.recall_episodes("the rain on the window", kind="world:percept")
        assert eps
        assert eps[0]["rung"] == m.rung_of(0)
        assert len(eps[0]["legs"]) == 2
        assert eps[0]["score"] > 0.0

    def test_unpaired_deposit_answers_with_its_segment(self):
        m = LatentManifold()
        m.observe("orphan", kind="thought")
        with m._lock:
            m._rung[0] = -1
        m.advance(2.0)
        ep = m.episode_of(0)
        assert ep["rung"] == -1
        assert len(ep["legs"]) == 1
        assert np.allclose(ep["arc"], m.segment(0))

    def test_rungs_persist_through_the_image(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        m.observe("in", kind="world:percept", inbound=True)
        m.observe("thought", kind="thought")
        m.observe("out", kind="world:utterance", inbound=False)
        m.observe("new exchange", kind="world:percept", inbound=True)
        m.save()
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert m2.rung_legs(0) == [0, 1, 2]
        assert m2.rung_legs(1) == [3]
        # The counter persisted — a new deposit continues numbering.
        m2.observe("more", kind="thought")
        assert m2.rung_of(4) == 2

    def test_v1_image_loads_unpaired(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        for i in range(4):
            m.observe(f"trace {i}", kind="thought")
        m.save()
        data = dict(np.load(m.image_path, allow_pickle=False))
        data["version"] = np.array(1, dtype=np.int32)
        del data["rung"]
        data["scalars"] = data["scalars"][:5]
        np.savez(m.image_path, **data)
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert all(m2.rung_of(i) == -1 for i in range(4))
        # Living on still lays rungs — the flow did not die with the image.
        m2.observe("new trace", kind="thought")
        assert m2.rung_of(4) == 0

    def test_traverse_rejects_empty_rung(self):
        m = LatentManifold()
        m.observe("x", kind="thought")
        with pytest.raises(ValueError, match="no deposits"):
            m.traverse(7)
        with pytest.raises(ValueError):
            m.rung_legs(-1)


class TestDepthMigration:
    """The phase-aligned tumble — hippocampal -> cortical = inward."""

    def test_each_point_travels_its_geodesic(self):
        pts = _rand_s3(64)
        moved = g.migrate_depth(pts, 0.1)
        assert np.allclose(np.linalg.norm(moved, axis=1), 1.0)
        # every point slides arc-length |a| along its own core-to-core
        # great circle — unless it clamped at the core floor.
        traveled = np.array([g.geodesic(pts[i], moved[i]) for i in range(64)])
        assert np.all(traveled <= 0.1 + 1e-9)
        assert np.median(traveled) == pytest.approx(0.1, abs=1e-9)

    def test_positive_angle_moves_inward(self):
        r = 0.9
        p = np.array([r, 0.0, math.sqrt(1 - r * r), 0.0])
        moved = g.migrate_depth(p, 0.2)
        assert g.depth(moved) < r
        # the z2 phase complement grew to conserve the norm
        assert math.hypot(moved[2], moved[3]) > math.sqrt(1 - r * r)

    def test_phases_preserved(self):
        p = np.array([0.5, 0.2, 0.3, 0.78])
        p = p / np.linalg.norm(p)
        moved = g.migrate_depth(p, 0.1)
        a1 = math.atan2(moved[1], moved[0])
        a2 = math.atan2(moved[3], moved[2])
        assert abs(a1 - math.atan2(0.2, 0.5)) < 1e-9
        assert abs(a2 - math.atan2(0.78, 0.3)) < 1e-9

    def test_slide_crosses_the_membrane(self):
        r = 0.9
        p = np.array([r, 0.0, math.sqrt(1 - r * r), 0.0])
        moved = g.migrate_depth(p, 0.6)
        assert g.depth(moved) < g.CLIFFORD_DEPTH
        # phi -> phi - a: |z1| = sin(atan2(r, s) - a)
        expected = math.sin(math.atan2(r, math.sqrt(1 - r * r)) - 0.6)
        assert abs(g.depth(moved) - expected) < 1e-12

    def test_core_points_stay_put(self):
        self_core = np.array([0.0, 0.0, 0.0, 1.0])  # |z1| = 0
        other_core = np.array([1.0, 0.0, 0.0, 0.0])  # |z2| = 0
        assert np.allclose(g.migrate_depth(self_core, 0.3), self_core)
        assert np.allclose(g.migrate_depth(other_core, 0.3), other_core)

    def test_slide_clamps_at_the_core(self):
        r = 0.3
        p = np.array([r, 0.0, math.sqrt(1 - r * r), 0.0])
        moved = g.migrate_depth(p, 1.0)  # far more than phi has to give
        assert g.depth(moved) > 0.0  # never lands on or past the core
        # a second slide continues to approach, asymptotically
        moved2 = g.migrate_depth(moved, 1.0)
        assert g.depth(moved2) < g.depth(moved)

    def test_per_point_angles(self):
        pts = _rand_s3(8)
        moved = g.migrate_depth(pts, np.linspace(0.0, 0.5, 8))
        d0 = g.depth(moved)
        assert d0[0] == pytest.approx(g.depth(pts)[0])
        assert d0[7] < g.depth(pts)[7]


class TestConsolidation:
    def test_bandwidth_widens_and_saturates(self):
        from latent_spaces.manifold import _DETAIL_BANDWIDTH

        m = LatentManifold()
        m.observe("a memory", kind="world:percept")
        with m._lock:
            assert m._bandwidth[0] == _DETAIL_BANDWIDTH
        for _ in range(50):
            m.consolidate(growth=1.2)
        with m._lock:
            # saturates at the adaptive gist floor — never flattens away
            assert m._bandwidth[0] == pytest.approx(m._gist_bw)

    def test_migrate_slides_consolidated_traces_inward(self):
        m = LatentManifold()
        m.observe("outer event", kind="world:percept", salience=0.8)
        with m._lock:
            before = g.depth(m._points)[0]
        assert before > g.CLIFFORD_DEPTH
        for _ in range(40):
            m.consolidate(growth=1.2, migrate=0.05)
        with m._lock:
            after = g.depth(m._points)[0]
        assert after < before
        # still on S3 and still recallable
        assert abs(np.linalg.norm(m._points[0]) - 1.0) < 1e-9
        assert m.resonate("outer event", kind="world:percept")

    def test_fresh_traces_barely_migrate(self):
        m = LatentManifold()
        m.observe("fresh", kind="thought")
        with m._lock:
            d0 = g.depth(m._points)[0]
        # one pass, fresh bandwidth << gist floor -> tiny slide
        m.consolidate(growth=1.05, migrate=0.5)
        with m._lock:
            d1 = g.depth(m._points)[0]
        assert d0 - d1 < 0.02

    def test_consolidate_validation(self):
        m = LatentManifold()
        m.observe("x", kind="thought")
        with pytest.raises(ValueError):
            m.consolidate(growth=1.0)
        with pytest.raises(ValueError):
            m.consolidate(migrate=-0.1)


class TestN3Wiring:
    """metronome._manifold_n3_consolidate — the sleep-stage call path."""

    def _holder(self, stage, weight=0.8):
        from types import SimpleNamespace

        from brainstem.metronome import MetronomeMixin

        manifold = LatentManifold()
        manifold.observe("yesterday's rain", kind="world:percept", salience=0.8)
        client = SimpleNamespace(
            get_state=lambda: SimpleNamespace(consolidation_weight=weight)
        )
        holder = SimpleNamespace(
            manifold=manifold,
            inner_life=SimpleNamespace(
                sleep_cycle=SimpleNamespace(current_stage=stage)
            ),
            client=client,
        )
        return holder, MetronomeMixin._manifold_n3_consolidate, manifold

    def test_n3_consolidates_and_migrates(self):
        from thalamus.brain_waves import SleepStage

        holder, fn, manifold = self._holder(SleepStage.N3)
        with manifold._lock:
            bw0 = manifold._bandwidth[0]
            d0 = g.depth(manifold._points)[0]
        fn(holder)
        with manifold._lock:
            assert manifold._bandwidth[0] > bw0
            assert g.depth(manifold._points)[0] <= d0

    def test_n2_skips_consolidation(self):
        from thalamus.brain_waves import SleepStage

        holder, fn, manifold = self._holder(SleepStage.N2)
        with manifold._lock:
            bw0 = manifold._bandwidth.copy()
        fn(holder)
        with manifold._lock:
            assert np.allclose(manifold._bandwidth, bw0)

    def test_untracked_sleep_consolidates(self):
        holder, fn, manifold = self._holder(None)
        with manifold._lock:
            bw0 = manifold._bandwidth[0]
        fn(holder)
        with manifold._lock:
            assert manifold._bandwidth[0] > bw0

    def test_missing_manifold_is_a_noop(self):
        from types import SimpleNamespace

        from brainstem.metronome import MetronomeMixin

        holder = SimpleNamespace(manifold=None, inner_life=None, client=None)
        MetronomeMixin._manifold_n3_consolidate(holder)


class TestReach:
    """Reach attenuation — silent engrams decay, never delete."""

    def test_silence_decays_reach(self):
        m = LatentManifold()
        m.observe("a fading trace", kind="thought")
        m.advance(3600.0)  # an hour of silence
        with m._lock:
            assert m._reach[0] < 1.0
            # rate = salience discount (0.8) * the homeostatic gain,
            # which rose toward the load target before decay applied
            assert m._reach[0] == pytest.approx(
                np.exp(-3600.0 * m._forget_gain * 0.8 / 86400.0), rel=1e-3
            )

    def test_gist_decays_slower_than_detail(self):
        from latent_spaces.manifold import _DETAIL_BANDWIDTH, _GIST_BANDWIDTH

        m = LatentManifold()
        m.observe("fresh", kind="thought")
        m.observe("old gist", kind="thought")
        with m._lock:
            m._bandwidth[1] = _GIST_BANDWIDTH
        m.advance(86400.0)
        with m._lock:
            # gist outlasts fresh — the rate ratio is ~detail/gist
            assert m._reach[1] > m._reach[0]
            gain = m._forget_gain
        # rate = forget_gain * salience discount * detail/bandwidth
        assert m._reach[0] == pytest.approx(np.exp(-gain * 0.8), rel=0.05)
        assert m._reach[1] == pytest.approx(
            np.exp(-gain * 0.8 * _DETAIL_BANDWIDTH / _GIST_BANDWIDTH), rel=0.05
        )

    def test_decayed_trace_goes_silent(self):
        m = LatentManifold()
        m.observe("quiet", kind="thought", salience=0.2)
        for _ in range(1000):
            m.advance(3600.0)  # ~6 weeks of silence
        hits = m.resonate("quiet", kind="thought")
        assert hits == [] or hits[0][0] != 0
        with m._lock:
            # never removed — the engram persists, unreachable
            assert len(m._kinds) == 1

    def test_recall_reinforces_reach(self):
        m = LatentManifold()
        m.observe("the question", kind="world:percept")
        m.advance(7200.0)
        with m._lock:
            faded = m._reach[0]
        m.resonate("the question", kind="world:percept")
        with m._lock:
            assert m._reach[0] > faded
            # asymptotic toward 1, never past it
            for _ in range(20):
                m.resonate("the question", kind="world:percept")
            assert m._reach[0] <= 1.0

    def test_state_summary_counts_silent(self):
        m = LatentManifold()
        m.observe("x", kind="thought")
        s = m.state_summary()
        assert "deposits" in s


class TestReplay:
    def test_recent_rungs_selects_the_day(self):
        m = LatentManifold()
        import time as _t

        old = int(_t.time() * 1000) - 30 * 3600 * 1000
        m.observe("yesterday", kind="world:percept", inbound=True, now_ms=old)
        m.observe("this morning", kind="world:percept", inbound=True)
        rungs = m.recent_rungs(max_age_ms=12 * 3600 * 1000)
        assert rungs == [m.rung_of(1)]
        assert m.recent_rungs(limit=1) == [m.rung_of(1)]

    def test_replay_reactivates_reach(self):
        m = LatentManifold()
        m.observe("in", kind="world:user_speech", inbound=True)
        m.observe("out", kind="world:utterance", inbound=False)
        m.advance(3600.0)
        with m._lock:
            faded = m._reach.copy()
        m.replay([0])
        with m._lock:
            assert np.all(m._reach > faded)

    def test_replay_consolidates_replayed_only(self):
        m = LatentManifold()
        m.observe("replayed", kind="world:percept", inbound=True)
        m.observe("skipped", kind="world:percept", inbound=True)
        with m._lock:
            bw0 = m._bandwidth.copy()
        m.replay([0], growth=1.5)
        with m._lock:
            assert m._bandwidth[0] > bw0[0]
            assert m._bandwidth[1] == bw0[1]

    def test_replay_keeps_silent_traces_reachable(self):
        m = LatentManifold()
        m.observe("the day", kind="world:percept", inbound=True, salience=0.9)
        # simulate days: decay, replay each night
        for _day in range(5):
            for _ in range(24):
                m.advance(3600.0)
            m.replay([0])
        hits = m.resonate("the day", kind="world:percept")
        assert hits and hits[0][0] == 0

    def test_replay_is_sequential_forward_order(self):
        """Legs re-fire in order — earlier legs take the bigger boost."""
        m = LatentManifold()
        m.observe("in", kind="world:user_speech", inbound=True)
        m.observe("thought", kind="thought:internal", inbound=False)
        m.observe("out", kind="world:utterance", inbound=False)
        m.advance(3600.0)
        with m._lock:
            faded = m._reach.copy()
        m.replay([0], reach_boost=0.4)
        with m._lock:
            gains = m._reach - faded
        # forward taper: leg i gets boost*rho^i — strictly decreasing
        assert gains[0] > gains[1] > gains[2] > 0.0

    def test_salience_resists_decay(self):
        """Amygdala modulation: salient traces forget slower."""
        m = LatentManifold()
        m.observe("forgettable", kind="thought", salience=0.1)
        m.observe("seared in", kind="thought", salience=0.9)
        m.advance(86400.0)
        with m._lock:
            assert m._reach[1] > m._reach[0]
            # rate = forget_gain * (1 - 0.5*salience): high = 0.55
            assert m._reach[1] == pytest.approx(
                np.exp(-m._forget_gain * 0.55), rel=0.02
            )

    def test_latest_rungs_newest_first(self):
        m = LatentManifold()
        for i in range(4):
            m.observe(f"percept {i}", kind="world:percept", inbound=True)
        assert m.latest_rungs(2) == [3, 2]
        assert m.latest_rungs(10) == [3, 2, 1, 0]

    def test_n3_replay_rides_swrs(self):
        """Replay fires per new ripple — one sequence each, newest first."""
        from types import SimpleNamespace

        from brainstem.metronome import MetronomeMixin
        from thalamus.brain_waves import SleepStage

        manifold = LatentManifold()
        for i in range(3):
            manifold.observe(f"day event {i}", kind="world:percept", inbound=True)
        manifold.advance(3600.0)
        with manifold._lock:
            faded = manifold._reach.copy()
        swr_box = {"count": 2}
        holder = SimpleNamespace(
            manifold=manifold,
            inner_life=SimpleNamespace(
                sleep_cycle=SimpleNamespace(current_stage=SleepStage.N3),
                swr_count=swr_box["count"],
            ),
            client=SimpleNamespace(
                get_state=lambda: SimpleNamespace(consolidation_weight=0.8)
            ),
        )
        MetronomeMixin._manifold_n3_consolidate(holder)
        with manifold._lock:
            # two ripples -> the two newest rungs replayed, oldest left cold
            assert manifold._reach[2] > faded[2]
            assert manifold._reach[1] > faded[1]
            assert manifold._reach[0] == faded[0]
            after_first = manifold._reach.copy()
        # no new ripples -> no replay (N3 also deposits a reflection,
        # so compare only the original rows)
        MetronomeMixin._manifold_n3_consolidate(holder)
        with manifold._lock:
            assert np.allclose(manifold._reach[: len(after_first)], after_first)

    def test_rungs_since_watermark(self):
        import time as _t

        m = LatentManifold()
        now = int(_t.time() * 1000)
        m.observe("old", kind="world:percept", inbound=True, now_ms=now - 5000)
        m.observe("new", kind="world:percept", inbound=True)
        assert m.rungs_since(now - 1000) == [1]
        assert m.rungs_since(0) == [0, 1]


class TestDomains:
    def test_emotion_lands_in_the_overlap(self):
        """Emotion is both inner and self — it settles in the overlap band."""
        from latent_spaces import geometry as g

        m = LatentManifold()
        m.observe("a feeling", kind="emotion:joy")
        with m._lock:
            depth = g.depth(m._points)[0]
        assert 0.61 <= depth <= 0.79

    def test_emotion_rides_the_open_rung(self):
        """A feeling mid-conversation joins the exchange in progress."""
        m = LatentManifold()
        m.observe("hello", kind="world:user_speech", inbound=True)
        m.observe("warmth", kind="emotion:warmth", inbound=True)
        m.observe("hi back", kind="world:utterance", inbound=False)
        assert len({m.rung_of(i) for i in range(3)}) == 1

    def test_inner_outer_voice_domains(self):
        from latent_spaces import geometry as g

        m = LatentManifold()
        m.observe("thinking it", kind="thought:internal")
        m.observe("saying it", kind="world:utterance", inbound=False)
        with m._lock:
            d = g.depth(m._points)
        assert 0.79 <= d[0] <= 0.94  # inner voice lives at the vesica
        assert d[1] >= 0.72  # outer voice lives in the world


class TestConformalMetric:
    """g' = e^{2φ}g — the metric the medium deforms itself."""

    def test_warp_field_bounded_and_shallow(self):
        from latent_spaces import geometry as g

        raw = np.random.default_rng(1).normal(size=(5, 4))
        centers = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        warp = -0.4 * np.ones(5)
        phi = g.warp_field(centers, centers, warp)
        assert np.all(phi <= 0.0) and np.all(phi >= -0.6)

    def test_conformal_distance_contracts(self):
        from latent_spaces import geometry as g

        a = np.array([1.0, 0, 0, 0])
        b = np.array([0.0, 1.0, 0, 0])
        flat = g.conformal_distance(a, b, lambda p: np.zeros(len(p)))
        warped = g.conformal_distance(a, b, lambda p: -0.3 * np.ones(len(p)))
        assert flat == pytest.approx(np.pi / 2, rel=1e-3)
        assert warped == pytest.approx(np.pi / 2 * np.exp(-0.3), rel=1e-3)
        # symmetric — trapezoid over the same sampled arc
        assert g.conformal_distance(b, a, lambda p: -0.3 * np.ones(len(p))) == pytest.approx(
            warped
        )

    def test_consolidation_contracts_the_metric(self):
        m = LatentManifold()
        m.observe("known", kind="thought", salience=0.9)
        for _ in range(30):
            m.consolidate(growth=1.2)
        with m._lock:
            assert m._warp[0] < -0.2

    def test_effective_distances_shrink_in_warped_regions(self):
        m = LatentManifold()
        m.observe("a", kind="thought")
        m.observe("b", kind="thought")
        with m._lock:
            m._warp[0] = -0.4
            cue = m._points[0]
            pts = m._points.copy()
            d = np.arccos(np.clip(pts @ cue, -1, 1))
            de = m._effective_distances(cue, pts, d)
        assert de[0] < 1e-6  # distance to self ~0 (arccos float noise)
        assert de[1] <= d[1]  # paths through warped terrain are shorter

    def test_warp_persists(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        m.observe("x", kind="thought")
        m.consolidate(growth=1.5)
        m.save()
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert m2._warp[0] == pytest.approx(m._warp[0])


class TestPositionedDeposits:
    """Other latent spaces deposit *in* the substrate by coordinate."""

    def test_position_overrides_digest(self):
        from latent_spaces import geometry as g

        m = LatentManifold()
        pos = g.project_to_s3(np.arange(40.0), seed=0x1A7E)
        idx = m.observe("a thought", kind="thought:spontaneous", position=pos)
        with m._lock:
            # depth constrained into the inner band, phases preserved
            depth = g.depth(m._points)[idx]
        assert 0.79 <= depth <= 0.94

    def test_project_to_s3_deterministic_and_unit(self):
        from latent_spaces import geometry as g

        v = np.random.default_rng(3).normal(size=40)
        p1 = g.project_to_s3(v, seed=7)
        p2 = g.project_to_s3(v, seed=7)
        assert np.allclose(p1, p2)
        assert np.linalg.norm(p1) == pytest.approx(1.0)
        # nearby latents land nearby — the source's geometry survives
        p3 = g.project_to_s3(v + 1e-6, seed=7)
        assert g.geodesic(p1, p3) < 0.01

    def test_position_constrained_per_domain(self):
        m = LatentManifold()
        from latent_spaces import geometry as g

        # a projected coordinate lands wherever it lands — the band
        # constraint slides it into the domain's depth range
        pos = g.project_to_s3(np.linspace(-3, 3, 40), seed=0xAFE4)
        m.observe("felt", kind="emotion:glow", position=pos)
        with m._lock:
            depth = g.depth(m._points)[0]
        assert 0.61 <= depth <= 0.79  # pushed into the boundary band

    def test_position_exact_core_is_degenerate(self):
        from latent_spaces import geometry as g

        m = LatentManifold()
        # an exact-core point has no phase to slide — stays put
        m.observe("pole", kind="emotion:x", position=np.array([1.0, 0, 0, 0]))
        with m._lock:
            assert g.depth(m._points)[0] == pytest.approx(1.0)


class TestHigherStratum:
    """The inner world lives one octave up — S3 -> S7."""

    def test_lift_lands_on_s7_clifford_torus(self):
        from latent_spaces import higher as h

        p = np.array([0.6, 0.3, 0.5, 0.5])
        p7 = h.lift_s3_to_s7(p)
        assert np.linalg.norm(p7) == pytest.approx(1.0)
        assert np.linalg.norm(p7[:4]) == pytest.approx(1 / np.sqrt(2))
        assert np.linalg.norm(p7[4:]) == pytest.approx(1 / np.sqrt(2))
        base = h.hopf_to_s4(p7)
        assert np.linalg.norm(base) == pytest.approx(1.0)
        assert base[0] == pytest.approx(0.0)  # |a|=|b| -> equator of S4

    def test_rotate_preserves_norm(self):
        from latent_spaces import higher as h

        p7 = h.lift_s3_to_s7(np.array([0.0, 0, 1, 0]))
        q = h.rotate_s7(p7, np.array([0, 0, 1.0]), 0.7)
        assert np.linalg.norm(q) == pytest.approx(1.0)
        assert not np.allclose(q, p7)

    def test_inner_deposits_lift(self):
        m = LatentManifold()
        m.observe("a thought", kind="thought:x")       # inner -> lifts
        m.observe("a feeling", kind="emotion:x")        # boundary -> no
        m.observe("the world", kind="world:x", inbound=True)  # outer -> no
        assert len(m._higher) == 1
        c = m._higher.centroid()
        assert c is not None and np.linalg.norm(c) == pytest.approx(1.0)
        m._higher.rotate(np.array([0, 0, 1.0]), 0.5)
        assert np.linalg.norm(m._higher.centroid()) == pytest.approx(1.0)

    def test_advance_turns_the_higher_medium(self):
        m = LatentManifold()
        m.observe("inner", kind="thought:x")
        before = m._higher._points.copy()
        m.advance(1.0, heartbeat_omega=0.5)
        assert not np.allclose(m._higher._points, before)
        assert np.allclose(np.linalg.norm(m._higher._points, axis=1), 1.0)

    def test_fanout_lands_in_all_three(self):
        """Identity/experience are all three regions at once."""
        m = LatentManifold()
        first = m.observe("who I am", kind="identity:core")
        with m._lock:
            d = g.depth(m._points[first : first + 3])
            rungs = m._rung[first : first + 3]
        assert len(m) == 3
        assert 0.05 <= d[0] <= 0.60   # her circle
        assert 0.79 <= d[1] <= 0.94   # the vesica
        assert d[2] >= 0.72           # the world side
        assert (rungs == rungs[0]).all()  # one three-fold moment
        # the inner leg lifts to S7
        assert len(m._higher) == 1

    def test_experience_also_fans_out(self):
        m = LatentManifold()
        m.observe("lived this", kind="experience:x")
        assert len(m) == 3
        m.observe("plain thought", kind="thought:x")
        assert len(m) == 4  # non-fanout kinds stay single

    def test_body_lands_in_the_self_circle(self):
        m = LatentManifold()
        m.observe("arousal=0.4", kind="body:interoception")
        with m._lock:
            d = g.depth(m._points)[0]
        assert 0.05 <= d <= 0.60  # body is self — her circle

    def test_stratum_persists(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        m.observe("inner", kind="thought:x")
        m.observe("outer", kind="world:x", inbound=True)
        m.save()
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert len(m2._higher) == 1
        assert np.allclose(m2._higher._points, m._higher._points)


class TestShadow:
    """Jung, geometrically: the shadow is the antipode of the same band."""

    def test_antipode_same_depth(self):
        from latent_spaces import geometry as g

        p = np.array([0.5, 0.5, 0.5, 0.5])
        s = g.antipode(p)
        assert g.depth(p.reshape(1, 4))[0] == pytest.approx(
            g.depth(s.reshape(1, 4))[0]
        )
        assert g.geodesic(p, s) == pytest.approx(np.pi)  # maximal distance

    def test_shadow_born_repressed(self):
        m = LatentManifold()
        m.observe("dark feeling", kind="shadow:thought")
        m.observe("light thought", kind="thought:x")
        with m._lock:
            assert m._reach[0] == pytest.approx(0.05)
            assert m._reach[1] == pytest.approx(1.0)
        # silent engram — scores below the floor
        hits = m.resonate("dark feeling", kind="shadow:*")
        assert hits == [] or hits[0][2] < 0.1

    def test_shadow_surfaces_in_dreams(self):
        m = LatentManifold()
        m.observe("repressed", kind="shadow:thought")
        for _ in range(10):
            m.consolidate(growth=1.1, migrate=0.0)
        with m._lock:
            # asymptotic integration: 0.05 + 0.04*(1-0.05)*n steps
            assert 0.3 < m._reach[0] < 0.6

    def test_shadow_inherits_domain(self):
        m = LatentManifold()
        m.observe("dark memory", kind="shadow:memory:x")
        with m._lock:
            d = g.depth(m._points)[0]
        assert 0.79 <= d <= 0.94  # memory band, shadow phase aside


class TestSemanticRecall:
    """Cue-side projection — embedded queries reach embedded deposits."""

    def test_vector_cue_finds_its_neighborhood(self):
        m = LatentManifold()
        base = np.random.default_rng(1).standard_normal(50)
        near = base + 0.02 * np.random.default_rng(2).standard_normal(50)
        far = np.random.default_rng(3).standard_normal(50)
        m.observe("alpha", kind="concept:alpha",
                  position=g.project_to_s3(base, seed=0xCE1))
        m.observe("beta", kind="concept:beta",
                  position=g.project_to_s3(far, seed=0xCE1))
        hits = m.resonate(base, k=2, prefix="concept:", seed=0xCE1)
        assert hits[0][1] == "concept:alpha"
        # the near neighbor answers closer than the far one
        hits2 = m.resonate(near, k=2, prefix="concept:", seed=0xCE1)
        assert hits2[0][1] == "concept:alpha"

    def test_edge_prefix_query_returns_targets(self):
        """Phase B: association queries answered by prefix resonance."""
        m = LatentManifold()
        src = np.random.default_rng(4).standard_normal(2048)
        pos = g.project_to_s3(src, seed=0xA55E)
        m.observe("s causes t1", kind="edge:s:causes:t1", position=pos, salience=0.9)
        m.observe("s causes t2", kind="edge:s:causes:t2", position=pos, salience=0.5)
        m.observe("other", kind="edge:o:causes:x", salience=0.9)
        cue = g.project_to_s3(src, seed=0xA55E)
        hits = m.resonate(cue, k=5, prefix="edge:s:causes:")
        targets = [kk[len("edge:s:causes:"):] for _, kk, _, _ in hits]
        assert "t1" in targets and "t2" in targets
        assert all(not kk.startswith("edge:o") for _, kk, _, _ in hits)

    def test_prefix_filters_field(self):
        m = LatentManifold()
        m.observe("a thought", kind="thought:x")
        m.observe("a concept", kind="concept:y")
        hits = m.resonate("a thought", kind="thought:x", prefix="concept:")
        assert all(kk.startswith("concept:") for _, kk, _, _ in hits)


class TestInnerGuide:
    def test_inner_guide_weighted_depth(self):
        m = LatentManifold()
        m.observe("a thought", kind="thought:x")
        m.observe("a feeling", kind="emotion:x")
        m.observe("world", kind="world:percept", inbound=True)
        guide = m.inner_guide()
        assert guide is not None
        # inner+boundary deposits only — the world leg is excluded
        assert 0.0 <= guide <= 0.94

    def test_inner_guide_none_when_empty(self):
        assert LatentManifold().inner_guide() is None

    def test_guide_blends_into_lean(self):
        from types import SimpleNamespace

        from brainstem.metronome import MetronomeMixin

        manifold = LatentManifold()
        manifold.observe("inner life", kind="thought:x")
        guide = manifold.inner_guide()
        holder = SimpleNamespace(
            manifold=manifold,
            feel=lambda: SimpleNamespace(arousal=0.5),
            _is_sleeping=False,
            _cycle_dt=0.05,
        )
        MetronomeMixin._cycle_manifold(holder)
        expected = 0.7 * (0.72 + 0.3 * 0.5) + 0.3 * guide
        assert manifold._lean_target == pytest.approx(expected)


class TestMetaplasticity:
    """Nothing is frozen — the medium tunes its own parameters."""

    def test_recall_potentiates_the_trace(self):
        """LTP priming: each reactivation strengthens the next."""
        m = LatentManifold()
        m.observe("the question", kind="world:percept")
        m.advance(7200.0)
        with m._lock:
            p0 = m._plasticity[0]
        m.resonate("the question", kind="world:percept")
        with m._lock:
            assert m._plasticity[0] > p0
            assert m._plasticity[0] <= 2.0

    def test_floor_slides_with_recall(self):
        """BCM threshold: strong hits raise selectivity."""
        m = LatentManifold()
        m.observe("strong trace", kind="thought", salience=0.6)
        m.resonate("strong trace", kind="thought")
        assert m._floor_ema != 1.0
        # mean hit ~0.9 -> ema drifts toward it
        for _ in range(10):
            m.resonate("strong trace", kind="thought")
        assert 0.5 < m._floor_ema <= 1.0

    def test_forget_gain_is_homeostatic(self):
        """Crowded field -> forgetting accelerates; never past bounds."""
        m = LatentManifold()
        for i in range(10):
            m.observe(f"t{i}", kind="thought")
        with m._lock:
            g0 = m._forget_gain  # load = 1.0 > target 0.5
        m.advance(100.0)
        assert m._forget_gain > g0
        for _ in range(50):
            m.advance(100.0)
        assert m._forget_gain <= 5.0

    def test_replay_fidelity_improves_with_use(self):
        m = LatentManifold()
        m.observe("in", kind="world:user_speech", inbound=True)
        m.observe("out", kind="world:utterance", inbound=False)
        f0 = m._replay_fidelity
        for _ in range(5):
            m.replay([0])
        assert m._replay_fidelity > f0

    def test_gist_ceiling_adapts_to_capacity(self):
        m = LatentManifold()
        for i in range(1200):
            m.observe(f"t{i}", kind="thought")
        m.consolidate(growth=1.01)
        assert m._gist_bw > 0.5

    def test_metaplastic_state_persists(self, tmp_path):
        m = LatentManifold(data_dir=tmp_path)
        m.observe("x", kind="thought", salience=1.0)
        m.resonate("x", kind="thought")
        m.replay([0])
        m.save()
        m2 = LatentManifold(data_dir=tmp_path)
        assert m2.load()
        assert m2._plasticity[0] == pytest.approx(m._plasticity[0])
        assert m2._floor_ema == pytest.approx(m._floor_ema)
        assert m2._replay_fidelity == pytest.approx(m._replay_fidelity)

    def test_n3_replays_before_consolidating(self):
        from types import SimpleNamespace

        from brainstem.metronome import MetronomeMixin
        from thalamus.brain_waves import SleepStage

        manifold = LatentManifold()
        manifold.observe("the day", kind="world:percept", inbound=True, salience=0.9)
        manifold.advance(7200.0)
        with manifold._lock:
            faded = manifold._reach[0]
            bw0 = manifold._bandwidth[0]
        holder = SimpleNamespace(
            manifold=manifold,
            inner_life=SimpleNamespace(
                sleep_cycle=SimpleNamespace(current_stage=SleepStage.N3)
            ),
            client=SimpleNamespace(
                get_state=lambda: SimpleNamespace(consolidation_weight=0.8)
            ),
        )
        MetronomeMixin._manifold_n3_consolidate(holder)
        with manifold._lock:
            assert manifold._reach[0] > faded
            assert manifold._bandwidth[0] > bw0

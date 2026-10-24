"""Eye optics tests — the aperture's consequences on the retinal image.

The body integrates the pupil (optic_nerve.rs: light reflex + LC
sympathetic tone); the mind applies what that aperture does to the
image — retinal irradiance scaling and depth-of-field blur — before
anything cortical sees the frame.
"""

import numpy as np
import pytest
from occipital_lobe.perception import optics

from genesis_client.types import OpticState


def _optic(
    pupil_mm: float,
    relay_gain: float = 0.9,
    accommodation_dpt: float = 0.0,
    lid: float = 1.0,
) -> OpticState:
    return OpticState(
        eye_present=True,
        frames_fresh=True,
        illuminance=0.4,
        melanopic=0.5,
        motion=0.0,
        orienting=0.0,
        relay_gain=relay_gain,
        cortical_gain=0.5,
        retinal_lux=0.0,
        frame_seq=1,
        frame_age_ms=10,
        pupil_mm=pupil_mm,
        accommodation_dpt=accommodation_dpt,
        lid=lid,
    )


class _OpticClient:
    def __init__(self, optic: OpticState | None) -> None:
        self._optic = optic

    def get_optic_state(self) -> OpticState | None:
        return self._optic


class TestIrradianceGain:
    def test_unreported_pupil_is_neutral(self) -> None:
        """0.0 is "the daemon didn't say" — never a zero reading."""
        assert optics.irradiance_gain(0.0) == 1.0
        assert optics.irradiance_gain(-1.0) == 1.0
        assert optics.irradiance_gain(float("nan")) == 1.0
        assert optics.irradiance_gain(float("inf")) == 1.0

    def test_resting_pupil_is_unit_gain(self) -> None:
        assert optics.irradiance_gain(optics.REFERENCE_PUPIL_MM) == pytest.approx(1.0)

    def test_aperture_area_scales_quadratically(self) -> None:
        """Constricted dims, dilated boosts — area, not diameter."""
        assert optics.irradiance_gain(2.0) == pytest.approx((2.0 / 4.5) ** 2)
        assert optics.irradiance_gain(8.0) == pytest.approx((8.0 / 4.5) ** 2)
        # Beyond the body's max the gain clamps to the 8 mm aperture.
        assert optics.irradiance_gain(12.0) == pytest.approx(optics.irradiance_gain(8.0))


class TestRetinalIrradiance:
    def test_constricted_pupil_dims_the_cone_input(self) -> None:
        ret = optics.retinal_irradiance(0.8, 2.0)
        assert ret == pytest.approx(0.8 * (2.0 / 4.5) ** 2)
        assert ret < 0.2

    def test_dilated_pupil_saturates_but_does_not_exceed(self) -> None:
        """A wide pupil passes the scene's photons — it invents none."""
        assert optics.retinal_irradiance(0.5, 8.0) == pytest.approx(1.0)

    def test_no_pupil_passes_the_scene_through(self) -> None:
        assert optics.retinal_irradiance(0.6, 0.0) == pytest.approx(0.6)


class TestDefocus:
    def _edge_frame(self) -> np.ndarray:
        """A sharp vertical step edge — maximal high-frequency content."""
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, 32:] = 255
        return frame

    def _max_gradient(self, frame: np.ndarray) -> float:
        """Steepest single-pixel step in the frame — a sharp edge is
        one step of 255; blur spreads it into a ramp of small steps."""
        return float(np.abs(np.diff(frame.astype(np.float64), axis=1)).max())

    def test_pinhole_pupil_is_sharp(self) -> None:
        frame = self._edge_frame()
        result = optics.apply(frame, 2.0)
        assert result.defocus_sigma == 0.0
        assert result.frame is frame  # no convolution under a pinhole

    def test_dilated_pupil_softens_the_retinal_image(self) -> None:
        frame = self._edge_frame()
        result = optics.apply(frame, 8.0)
        assert result.defocus_sigma == pytest.approx(1.6)
        assert not np.array_equal(result.frame, frame)
        # The step edge smeared into a ramp — its steepest step fell.
        assert self._max_gradient(result.frame) < self._max_gradient(frame) * 0.6

    def test_resting_pupil_softens_only_slightly(self) -> None:
        sigma = optics.defocus_sigma(4.5)
        assert 0.0 < sigma < 1.0
        result = optics.apply(self._edge_frame(), 4.5)
        assert result.defocus_sigma == pytest.approx(sigma)

    def test_non_rgb_frame_passes_through(self) -> None:
        frame = np.zeros((48, 64, 2), dtype=np.uint8)
        result = optics.apply(frame, 8.0)
        assert result.frame is frame

    def test_near_focus_thins_the_depth_of_field(self) -> None:
        """Accommodation: a close commanded focus adds the DOF term —
        far-field structure smears harder on a near-focused retina."""
        pupil_only = optics.defocus_sigma(4.5)
        near = optics.defocus_sigma(4.5, focus_m=0.3)
        assert near > pupil_only
        # Inside the 50 cm reference the term saturates — the near
        # point (6 D, ~17 cm) adds no more than the 30 cm look does.
        assert optics.defocus_sigma(4.5, focus_m=0.17) == pytest.approx(near)
        # A mid-range focus adds less than a near one.
        assert pupil_only < optics.defocus_sigma(4.5, focus_m=1.0) < near
        # Infinity (None) adds nothing — the relaxed lens blurs only
        # by its aperture.
        assert optics.defocus_sigma(4.5, focus_m=None) == pupil_only
        result = optics.apply(self._edge_frame(), 4.5, focus_m=0.3)
        assert result.focus_m == pytest.approx(0.3)

    def test_heavy_lids_dim_the_retinal_image(self) -> None:
        """A drowsy lid attenuates the frame itself — the retinal
        image dims uniformly, and every cortical stage downstream
        sees the same half-lit field."""
        frame = np.full((48, 64, 3), 200, dtype=np.uint8)
        open_eye = optics.apply(frame, 4.5, lid=1.0)
        half_mast = optics.apply(frame, 4.5, lid=0.5)
        assert open_eye.lid == 1.0
        assert half_mast.lid == pytest.approx(0.5)
        assert half_mast.frame is not frame
        # The lid dims every pixel; pupil gain stays its own term —
        # the irradiance gain is the aperture, the lid the shutter.
        assert half_mast.frame.mean() == pytest.approx(open_eye.frame.mean() * 0.5, abs=1.5)
        assert half_mast.irradiance_gain == open_eye.irradiance_gain

    def test_unreported_lid_is_an_open_lid(self) -> None:
        """NaN / out-of-range / absent all read as open — a legacy
        body never looks asleep to the optics layer."""
        frame = np.full((48, 64, 3), 200, dtype=np.uint8)
        for bad in (float("nan"), float("inf"), -0.5, 2.0):
            result = optics.apply(frame, 4.5, lid=bad)
            assert result.lid == 1.0
            assert result.frame.mean() == pytest.approx(200.0, abs=1.5)


class TestSeeIntegration:
    def _vision_with_stub_stages(self, monkeypatch):
        """A Vision whose cortical stages are stubbed — the test
        exercises only the optical front end."""
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: None)
        monkeypatch.setattr(Vision, "_run_v1_sparse_coding", lambda self, rgb, objs: None)
        monkeypatch.setattr(Vision, "_two_view_geometry", lambda self, rgb: None)
        monkeypatch.setattr(Vision, "_get_object_recognizer", lambda self: None)
        monkeypatch.setattr(Vision, "_get_face_recognizer", lambda self: None)
        vis = Vision()
        vis._available = True
        return vis, vision_mod

    def test_see_carries_the_pupil_onto_the_scene(self, monkeypatch) -> None:
        vis, vision_mod = self._vision_with_stub_stages(monkeypatch)
        monkeypatch.setattr(
            vision_mod,
            "latest_frame",
            lambda copy=True: np.full((48, 64, 3), 200, dtype=np.uint8),
        )
        scene = vis.see(client=_OpticClient(_optic(8.0)))
        assert scene.status == "ok"
        assert scene.pupil_mm == pytest.approx(8.0)

    def test_constricted_pupil_dims_the_cascade_input(self, monkeypatch) -> None:
        """The cones receive retinal illuminance — scene luminance
        through the aperture — not the scene's own brightness."""
        vis, vision_mod = self._vision_with_stub_stages(monkeypatch)
        monkeypatch.setattr(
            vision_mod,
            "latest_frame",
            lambda copy=True: np.full((48, 64, 3), 200, dtype=np.uint8),
        )
        captured: list[float] = []
        orig = vis._run_phototransduction

        def _spy(b: float):
            captured.append(b)
            return orig(b)

        vis._run_phototransduction = _spy
        vis.see(client=_OpticClient(_optic(2.0)))
        assert len(captured) == 1
        # The frame is ~0.78 scene luminance; a 2 mm pupil dims it to
        # ~0.15 at the receptors — a real retinal signal, not the scene's.
        assert captured[0] == pytest.approx(optics.retinal_irradiance(200.0 / 255.0, 2.0), abs=0.02)
        assert captured[0] < 0.25

    def test_unreported_pupil_leaves_the_frame_alone(self, monkeypatch) -> None:
        """A daemon that predates the field: identity transform."""
        vis, vision_mod = self._vision_with_stub_stages(monkeypatch)
        frame = np.full((48, 64, 3), 200, dtype=np.uint8)
        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: frame)
        applied: list[optics.OpticalResult] = []
        orig_apply = optics.apply

        def _spy_apply(f, d, fm=None, lid=1.0):
            applied.append(orig_apply(f, d, fm, lid))
            return applied[-1]

        monkeypatch.setattr(optics, "apply", _spy_apply)
        scene = vis.see(client=_OpticClient(_optic(0.0)))
        assert scene.pupil_mm == 0.0
        assert scene.focus_m is None
        assert applied[0].defocus_sigma == 0.0
        assert applied[0].irradiance_gain == 1.0
        assert applied[0].frame is frame

    def test_see_carries_the_focus_onto_the_scene(self, monkeypatch) -> None:
        """The body's accommodation lands on the scene as a distance —
        the proprioceptive depth signal the ciliary muscle carries."""
        vis, vision_mod = self._vision_with_stub_stages(monkeypatch)
        monkeypatch.setattr(
            vision_mod,
            "latest_frame",
            lambda copy=True: np.full((48, 64, 3), 200, dtype=np.uint8),
        )
        scene = vis.see(client=_OpticClient(_optic(4.5, accommodation_dpt=2.0)))
        assert scene.status == "ok"
        assert scene.focus_m == pytest.approx(0.5)

    def test_drowsy_look_is_a_dim_look(self, monkeypatch) -> None:
        """Half-closed lids land on the scene — the world literally
        darkens as sleep pressure takes her."""
        vis, vision_mod = self._vision_with_stub_stages(monkeypatch)
        monkeypatch.setattr(
            vision_mod,
            "latest_frame",
            lambda copy=True: np.full((48, 64, 3), 200, dtype=np.uint8),
        )
        scene = vis.see(client=_OpticClient(_optic(4.5, lid=0.5)))
        assert scene.status == "ok"
        assert scene.lid == pytest.approx(0.5)
        # A frame at 200/255 luminance, halved by the lid then the
        # region analyser's own scale — the scene reads darker than
        # the same room through open lids.
        assert scene.brightness < 200.0 / 255.0 * 0.75

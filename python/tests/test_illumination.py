"""Illumination tests — the Retinex light/surface split.

The retinal image is I = ρ·E, reflectance times illuminant. This
stage separates them: the illuminant map and its gradient (light
direction), the von Kries-adapted image cortex sees, and the
shadow-candidate read — dark regions whose colour still says
"lit surface", not "dark object".
"""

import numpy as np
import pytest
from occipital_lobe.perception import illumination

from genesis_client.types import OpticState


def _optic(pupil_mm: float = 4.5, relay_gain: float = 0.9) -> OpticState:
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
    )


class _OpticClient:
    def __init__(self, optic: OpticState | None) -> None:
        self._optic = optic

    def get_optic_state(self) -> OpticState | None:
        return self._optic


def _shadow_scene() -> np.ndarray:
    """One surface, half lit half shadowed: right is a warm-lit
    wall, left is the *same* wall in shadow — identical chromatic
    direction, stepped luminance."""
    frame = np.empty((48, 64, 3), dtype=np.uint8)
    frame[:, 32:] = (200, 160, 120)
    frame[:, :32] = (80, 64, 48)
    return frame


class TestIlluminantEstimate:
    def test_flat_frame_reports_its_own_light(self) -> None:
        illum = illumination.analyze(np.full((48, 64, 3), 200, dtype=np.uint8))
        # A flat gray field is lit by flat light — neutral chromaticity.
        assert illum.illuminant == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=0.02)
        assert illum.light_direction == ""
        assert illum.shadow_fraction == 0.0

    def test_warm_light_is_named(self) -> None:
        illum = illumination.analyze(np.full((48, 64, 3), (220, 170, 110), dtype=np.uint8))
        r, _g, b = illum.illuminant
        assert r > b, "warm illuminant favours red over blue"
        assert illum.illuminant_name in {"orange", "brown", "beige", "yellow"}


class TestVonKriesAdaptation:
    def test_warm_cast_adapts_toward_neutral(self) -> None:
        frame = np.full((48, 64, 3), (220, 170, 110), dtype=np.uint8)
        illum = illumination.analyze(frame)
        # The illuminant's chromatic pull is reduced in the adapted
        # frame — its red share drops toward the neutral third.
        in_r = 220 / (220 + 170 + 110)
        out = illum.frame.astype(np.float64).mean(axis=(0, 1))
        out_r = out[0] / out.sum()
        assert out_r < in_r - 0.02
        # But adaptation is partial — it never overshoots to flat.
        assert out_r > 1 / 3

    def test_neutral_light_leaves_the_frame(self) -> None:
        frame = np.full((48, 64, 3), (180, 180, 180), dtype=np.uint8)
        illum = illumination.analyze(frame)
        assert np.array_equal(illum.frame, frame)

    def test_warm_surfaces_do_not_recolour_as_light(self) -> None:
        """A lit field of warm *surfaces* beside an achromatic one
        must not be read as warm *light*: the neutral pixels are the
        honest witnesses, so the orange wall stays orange rather
        than being washed toward pink by a misattributed cast."""
        frame = np.empty((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (60, 55, 50)
        frame[:, :48] = (225, 140, 70)  # saturated warm wall
        frame[:, 52:64] = (230, 225, 215)  # bright near-neutral block
        illum = illumination.analyze(frame)
        wall = illum.frame[:, :40].astype(np.float64).mean(axis=(0, 1))
        # Under the old all-lit mean the wall's red-blue separation
        # collapsed (r-b ~81); the witness says the light is neutral,
        # so the wall's own cast must survive (r-b ~150).
        assert wall[0] - wall[2] > 120, f"warm wall recoloured: {wall}"

    def test_diverse_lit_field_lowers_adaptation_confidence(self) -> None:
        """Confidence is the lit field's agreement with the estimate —
        a uniform cast adapts fully, a field whose mean is a mush of
        disagreeing surface colours barely adapts at all."""
        uniform = np.full((48, 64, 3), (200, 120, 60), dtype=np.uint8)
        mixed = np.empty((48, 64, 3), dtype=np.uint8)
        mixed[:, :] = (60, 55, 50)
        mixed[:, :48] = (225, 140, 70)
        mixed[:, 52:64] = (230, 225, 215)
        conf_uniform = illumination.analyze(uniform).confidence
        conf_mixed = illumination.analyze(mixed).confidence
        assert conf_uniform > 0.9
        assert conf_mixed < conf_uniform


class TestLightDirection:
    def test_direction_reads_the_light_field_not_the_surfaces(self) -> None:
        """A shadowed saturated wall must not outvote a lit gray one —
        direction comes from the illuminant map, not colour names."""
        frame = _shadow_scene()
        illum = illumination.analyze(frame)
        assert illum.light_direction == "right"

    def test_overhead_light_reads_above(self) -> None:
        frame = np.empty((48, 64, 3), dtype=np.uint8)
        frame[:24, :] = 200
        frame[24:, :] = 60
        illum = illumination.analyze(frame)
        assert illum.light_direction == "above"

    def test_even_light_has_no_direction(self) -> None:
        illum = illumination.analyze(np.full((48, 64, 3), 140, dtype=np.uint8))
        assert illum.light_direction == ""


class TestReflectanceAndShadows:
    def test_reflectance_is_illuminant_invariant(self) -> None:
        """The point of the split: one wall reads one ρ on both
        sides of its shadow boundary."""
        illum = illumination.analyze(_shadow_scene())
        r_lit = illum.reflectance[:, 40:].mean()
        r_shade = illum.reflectance[:, :24].mean()
        assert abs(r_lit - r_shade) < 0.35, f"ρ {r_shade:.2f} vs {r_lit:.2f}"

    def test_shadowed_lit_surface_reads_as_shadow(self) -> None:
        illum = illumination.analyze(_shadow_scene())
        assert illum.shadow_fraction > 0.1
        assert illum.shadow_direction == "left"

    def test_chromatically_different_dark_thing_is_not_shadow(self) -> None:
        """A dark *blue* object in warm light is not a shadow — its
        colour says the surface changed, not just the light."""
        frame = np.empty((48, 64, 3), dtype=np.uint8)
        frame[:, 32:] = (200, 160, 120)  # warm-lit wall
        frame[:, :32] = (20, 20, 110)  # dark blue object
        illum = illumination.analyze(frame)
        assert illum.shadow_fraction < 0.1

    def test_uniform_dim_field_has_no_shadows(self) -> None:
        """Shadows need light to be cast by — a uniformly dim scene
        is a dim scene, not a shadowed one."""
        illum = illumination.analyze(np.full((48, 64, 3), 60, dtype=np.uint8))
        assert illum.shadow_fraction == 0.0


class TestSeeIntegration:
    def test_scene_carries_the_illumination(self, monkeypatch) -> None:
        import occipital_lobe.perception.vision as vision_mod
        from occipital_lobe.perception.vision import Vision

        monkeypatch.setattr(Vision, "_run_v1_sparse_coding", lambda self, rgb: None)
        monkeypatch.setattr(Vision, "_two_view_geometry", lambda self, rgb: None)
        monkeypatch.setattr(Vision, "_get_object_recognizer", lambda self: None)
        monkeypatch.setattr(Vision, "_get_face_recognizer", lambda self: None)
        monkeypatch.setattr(vision_mod, "latest_frame", lambda copy=True: _shadow_scene())
        vis = Vision()
        vis._available = True
        scene = vis.see(client=_OpticClient(_optic()))
        assert scene.status == "ok"
        assert scene.light_direction == "right"
        assert scene.shadow_direction == "left"
        assert scene.shadow_fraction > 0.1
        assert scene.illuminant != ""
        meta = scene.as_metadata()
        assert meta["shadow_fraction"] == scene.shadow_fraction
        assert meta["illuminant"] == scene.illuminant

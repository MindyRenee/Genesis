"""Illumination — decomposing retinal radiance into light and surface.

The receptors only receive radiance ``I = ρ·E`` — surface reflectance
times illumination, multiplied together. Everything downstream wants
them separated: colour names belong to surfaces (ρ), light reports
belong to the illuminant (E), and a shadow is exactly the place where
luminance drops while reflectance and chromaticity say the surface
didn't change.

The split is the retinal/LGN computation scaled to the field —
center-surround contrast: a low-pass luminance map is the illuminant
estimate (Land's Retinex), and dividing it out leaves reflectance:

    log I = log ρ + log E      E = low-pass(L),   ρ = L / E

From the split:

- **Illuminant chromaticity** — the mean colour of the well-lit
  field, voted first by its near-achromatic surfaces: the light
  actually falling on the scene. A von Kries diagonal gain
  (per-cone-class gain adaptation, applied at partial strength like
  the real chromatic-adaptation mechanism, which is never complete,
  and scaled by how much of the lit field corroborates the
  estimate) maps it toward neutral and yields the adapted frame
  every cortical stage sees — colour constancy, computed the way
  the retina computes it.

- **Light direction** — the illuminant map's gradient: the bright
  side of the *light field*, not of the surface colours, so a
  shadowed saturated wall can't outvote a lit gray one. A 2D
  azimuth — "above" and "below" exist, not only "left"/"right".

- **Shadows** — a shadowed lit surface is dark in ``I`` while its
  reflectance and its chromatic direction stay consistent with the
  illuminant-lit field: luminance stepped, colour didn't. That
  signature separates "object in shadow" from "dark object" — up to
  the monocular limit: a matte black surface under white light is
  chromatically identical to a white wall in deep shadow, and no
  single-view computation resolves that. What is reported here is
  the shadow-*candidate* share of the field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from occipital_lobe.v1 import name_color

from .optics import gaussian_blur_np

# The illuminant varies slowly across the field — the estimate is a
# low-pass luminance map at a fraction of the frame diagonal, the
# same assumption retinal center-surround makes at smaller scale.
_ILLUMINANT_SIGMA_DIV = 12.0

# Chromatic adaptation is partial in the living eye (~60–90% under
# natural illuminants); the gain blends toward full von Kries, never
# past it, and is clamped so a pathological illuminant estimate
# cannot invert the frame.
_ADAPT_STRENGTH = 0.8
_GAIN_LO, _GAIN_HI = 0.5, 2.0

# Only a near-achromatic surface shows the light's colour plainly —
# its chromaticity IS the illuminant's, where a saturated surface
# conflates ρ with E. The estimate therefore prefers the lit
# field's neutral pixels and falls back to the lit mean when too
# few neutral surfaces are present to vote.
_NEUTRAL_CHROMA_TOL = 0.12
_MIN_NEUTRAL_LIT_PX = 32

# Confidence in the estimate is the share of the lit field whose
# chromaticity actually agrees with it. A uniform cast is a
# confident read and adapts at full strength; an "illuminant" that
# is really the average of disagreeing surface colours is untrusted
# and adapts weakly — dividing it out would recolour the scene.
_AGREE_CHROMA_TOL = 0.10

# Shadows exist only where there is light to cast them: below this
# share of lit field there is nothing to be a shadow of.
_LIT_LUMA = 0.5
_MIN_LIT_FRACTION = 0.10
# A shadow-candidate pixel: darker than the lit field, but reflective
# (ρ near the lit surfaces', not a dark object's) and chromatically
# consistent with the illuminant — the edge dims, it doesn't recolor.
_SHADOW_LUMA = 0.45
_SHADOW_MIN_REFLECTANCE = 0.55
_SHADOW_CHROMA_TOL = 0.30
# Below this share of the field the shadow mass is noise.
_MIN_SHADOW_FRACTION = 0.02

_EPS = 1e-3


@dataclass(frozen=True, slots=True)
class Illumination:
    """One frame's illuminant/reflectance decomposition.

    ``frame`` is the chromatic-adapted retinal image — what cortex
    should analyse. ``reflectance`` is the ρ map (illuminant-invariant
    surface signal, ~1 where surface and illumination agree).
    ``illuminant`` is the estimated light's normalized chromaticity;
    ``illuminant_name`` its colour name. ``light_direction`` is the
    illuminant gradient's azimuth ("left" | "right" | "above" |
    "below" | ""), ``light_vector`` its raw (gx, gy). ``shadow_*``
    carry the shadow-candidate share of the field and where its mass
    sits ("left" | "center" | "right" | "").
    """

    frame: np.ndarray
    illuminant: tuple[float, float, float]
    illuminant_name: str
    light_direction: str
    light_vector: tuple[float, float]
    shadow_fraction: float
    shadow_direction: str
    reflectance: np.ndarray
    confidence: float = 1.0


def _luma(frame: np.ndarray) -> np.ndarray:
    """Rec.601 luminance in [0, 1]."""
    f = frame.astype(np.float32) / 255.0
    return f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114


def _chroma(frame: np.ndarray) -> np.ndarray:
    """Per-pixel normalized colour direction — rgb / (r+g+b), the
    component illumination scales uniformly and surface hue keeps."""
    f = frame.astype(np.float32) / 255.0
    return f / (f.sum(axis=-1, keepdims=True) + _EPS)


def _estimate_illuminant(frame: np.ndarray, e_map: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean chromaticity of the well-lit field, normalized to sum 1,
    and the lit-field agreement share that trusts it.

    The light is estimated where it is strongest — the top quartile
    of the illuminant map — so dark surfaces don't drag the estimate.
    Within that lit set, near-achromatic pixels vote first: only a
    surface whose own colour is neutral reveals the illuminant's
    colour rather than its own. The returned second value is the
    fraction of the lit field whose chromaticity agrees with the
    estimate — the confidence the von Kries stage scales itself by.
    Falls back to the whole field when the map is degenerate.
    """
    thresh = float(np.quantile(e_map, 0.75)) if e_map.size else 0.0
    lit = e_map >= max(thresh, _EPS)
    if not lit.any():
        lit = np.ones(e_map.shape, dtype=bool)
    lit_px = frame[lit].astype(np.float32) / 255.0
    chroma = lit_px / (lit_px.sum(axis=1, keepdims=True) + _EPS)
    neutral = np.abs(chroma - 1.0 / 3.0).sum(axis=1) < _NEUTRAL_CHROMA_TOL
    voters = lit_px[neutral] if int(neutral.sum()) >= _MIN_NEUTRAL_LIT_PX else lit_px
    mean_rgb = voters.mean(axis=0)
    total = float(mean_rgb.sum())
    if total < _EPS:
        return np.full(3, 1.0 / 3.0, dtype=np.float64), 1.0
    # float64 through the gain computation — a neutral estimate must
    # yield gains of exactly 1.0 so the frame passes through unscaled.
    illuminant = mean_rgb.astype(np.float64) / total
    agree = float((np.abs(chroma - illuminant).sum(axis=1) < _AGREE_CHROMA_TOL).mean())
    return illuminant, agree


def _von_kries(frame: np.ndarray, illuminant: np.ndarray, confidence: float = 1.0) -> np.ndarray:
    """Partial-strength von Kries adaptation — per-channel gain that
    maps the illuminant toward neutral while preserving its luma.

    ``confidence`` is the lit field's agreement with the estimate;
    it scales the already-partial strength so an estimate the field
    does not corroborate moves the frame hardly at all.
    """
    luma_ill = float(illuminant @ np.array([0.299, 0.587, 0.114], dtype=np.float64))
    gains = luma_ill / np.maximum(illuminant, _EPS)
    gains = 1.0 + _ADAPT_STRENGTH * confidence * (gains - 1.0)
    gains = np.clip(gains, _GAIN_LO, _GAIN_HI).astype(np.float32)
    return np.clip(frame.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)


def _light_direction(e_map: np.ndarray) -> tuple[str, tuple[float, float]]:
    """Azimuth of the illuminant gradient — the bright side of the
    light field is the side the light arrives from."""
    h, w = e_map.shape
    gx = float(e_map[:, 2 * w // 3 :].mean() - e_map[:, : w // 3].mean())
    gy = float(e_map[h // 2 :, :].mean() - e_map[: h // 2, :].mean())
    if max(abs(gx), abs(gy)) < 0.05:
        return "", (gx, gy)
    if abs(gx) >= abs(gy):
        return ("right" if gx > 0 else "left"), (gx, gy)
    return ("below" if gy > 0 else "above"), (gx, gy)


def _shadows(
    frame: np.ndarray, luma: np.ndarray, e_map: np.ndarray, reflectance: np.ndarray
) -> tuple[float, str]:
    """Field share reading as shadow, and where its mass sits.

    A shadowed lit surface keeps the lit field's chromatic direction
    (the light dims, it doesn't change colour) and keeps surface
    reflectance (ρ stays near the lit surfaces', not at a dark
    object's). Both signatures are required — luminance alone calls
    every dark thing a shadow.
    """
    lit = luma > _LIT_LUMA
    if float(lit.mean()) < _MIN_LIT_FRACTION:
        return 0.0, ""
    chroma = _chroma(frame)
    lit_chroma = chroma[lit].mean(axis=0)
    chroma_dist = np.abs(chroma - lit_chroma).sum(axis=-1)
    mask = (
        (luma < _SHADOW_LUMA)
        & (reflectance > _SHADOW_MIN_REFLECTANCE)
        & (chroma_dist < _SHADOW_CHROMA_TOL)
    )
    fraction = float(mask.mean())
    if fraction < _MIN_SHADOW_FRACTION:
        return 0.0, ""
    w = mask.shape[1]
    centroid_x = float(mask.nonzero()[1].mean()) if mask.any() else w / 2.0
    if centroid_x < w / 3:
        direction = "left"
    elif centroid_x < 2 * w / 3:
        direction = "center"
    else:
        direction = "right"
    return fraction, direction


def _illuminant_name(illuminant: np.ndarray) -> str:
    """Name the light's colour — "warm light" is illuminant data."""
    r, g, b = (float(v) for v in illuminant)
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < _EPS:
        hue = 0.0
    elif mx == r:
        hue = (60.0 * ((g - b) / d)) % 360.0
    elif mx == g:
        hue = 60.0 * ((b - r) / d) + 120.0
    else:
        hue = 60.0 * ((r - g) / d) + 240.0
    sat = d / mx if mx > _EPS else 0.0
    return name_color(hue, sat, float(mx))


def analyze(frame: np.ndarray) -> Illumination:
    """Decompose one retinal frame into light and surface.

    Runs after the eye's optics (the aperture's irradiance and
    point-spread are already in the frame) and before every cortical
    stage: color analysis, recognition, V1, and the dorsal stream all
    see the chromatic-adapted image — the one colour constancy
    produces.
    """
    luma = _luma(frame)
    h, w = luma.shape
    sigma = max(h, w) / _ILLUMINANT_SIGMA_DIV
    try:
        import cv2  # type: ignore[import-not-found]

        e_map = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma)
    except ImportError:
        e_map = gaussian_blur_np(luma, sigma)
    reflectance = luma / (e_map + _EPS)

    illuminant, confidence = _estimate_illuminant(frame, e_map)
    adapted = _von_kries(frame, illuminant, confidence)
    direction, vector = _light_direction(e_map)
    shadow_fraction, shadow_direction = _shadows(frame, luma, e_map, reflectance)

    return Illumination(
        frame=adapted,
        illuminant=(float(illuminant[0]), float(illuminant[1]), float(illuminant[2])),
        illuminant_name=_illuminant_name(illuminant),
        light_direction=direction,
        light_vector=vector,
        shadow_fraction=shadow_fraction,
        shadow_direction=shadow_direction,
        reflectance=reflectance,
        confidence=confidence,
    )

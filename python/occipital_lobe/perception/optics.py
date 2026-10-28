"""Eye optics — the cornea/lens stage in front of the receptor sheet.

One eye, one pupil, one retinal image. Genesis is monocular — a
single camera, a single shared-memory segment, a single aperture —
so there is no stereo disparity and no vergence anywhere in the
visual system. Her depth cues are the monocular set: motion parallax
(the epipolar dorsal stage) and the defocus this layer applies.

The model is the paraxial thin-lens account of what the aperture the
body sets does to the image that lands on the receptors:

- **Pupil → retinal irradiance.** Retinal illuminance scales with
  pupil area — the troland relation ``E_ret = E_scene · (d/d_ref)²``.
  The body integrates the aperture itself (`optic_nerve` —
  the light reflex plus LC sympathetic tone, read here as
  ``OpticState.pupil_mm``); this layer applies its optical
  consequence before the photoreceptors. The reported light level
  stays scene luminance — she still says "bright" while her
  receptors work inside their adapted range, which is exactly what
  the reflex is for.

- **Pupil → depth of field.** A wider aperture accepts a wider cone
  of rays, so off-focus structure projects to larger blur circles —
  depth of field shrinks as the pupil opens (the circle-of-confusion
  relation). With no depth map the consequence is applied as the
  eye's point-spread: blur σ grows with diameter, zero at the
  constricted end where the pupil approaches a pinhole and the whole
  field is sharp.

- **The lens inverts the image on the retina.** Cortex never observes
  the inversion — the retinotopic map is re-referenced upstream — so
  this layer transforms intensity and acuity, never geometry.

- **The view is mirror-oriented.** ``mirror_view`` applies the
  single-axis reflection at presentation: the world lands on the
  receptors the way it faces her — upright, left-right swapped from
  the raw capture. Chirality flips here, which is the point: the
  person facing her reads the same handedness she reports. This is
  deliberately *not* the lens's 180° inversion (chirality-preserving
  rotation, anatomically real but re-referenced before cortex) — it
  is the facing orientation, and it propagates: every cortical stage
  downstream sees the mirrored field, so her reported left/right,
  region colours, and motion directions describe the view as a
  mirror holds it.

What is deliberately NOT here: illuminant estimation, reflectance
decomposition, shadow edges — the inverse-optics work belongs to the
illumination stage downstream (Retinex / von Kries), not to the lens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Reference aperture — the mid-range diameter the daemon rests at
# (PUPIL_REST_MM in optic_nerve.rs). Unit gain here: the camera
# frame at a resting pupil is taken as the reference irradiance.
REFERENCE_PUPIL_MM = 4.5

# Physiological bounds, mirroring the body's clamp in optic_nerve.rs.
PUPIL_MIN_MM = 2.0
PUPIL_MAX_MM = 8.0

# Below this aperture the eye is effectively a pinhole — the modelled
# point-spread is diffraction-limited and no defocus blur is applied.
_DEFOCUS_REF_MM = 3.0
# Blur radius (pixels, on the 640×480 retinal image) at full
# dilation — shallow depth of field, mild aberration softening.
_DEFOCUS_MAX_SIGMA = 1.6
# Sub-pixel blur is not worth a convolution pass.
_DEFOCUS_MIN_SIGMA = 0.4

# Depth of field thins as accommodation pulls the focus near —
# physically ∝ focus² for a fixed aperture, so far structure
# smears harder across a near-focused retina. The term saturates at
# the reference distance below: past ~50 cm of commanded focus the
# extra softening stops growing.
_FOCUS_REF_M = 0.5
_FOCUS_MAX_SIGMA = 0.7


@dataclass(frozen=True, slots=True)
class OpticalResult:
    """What the eye's optics did to one frame.

    ``frame`` is the retinal image — the input with the eye's
    point-spread applied (the same array object when the pupil left
    nothing to apply). ``irradiance_gain`` is the pupil-area factor
    that scales scene luminance into retinal illuminance;
    ``defocus_sigma`` is the applied blur radius; ``pupil_mm`` is the
    aperture that produced it (``0.0`` when the body did not report
    one — the transform is then the identity).
    """

    frame: np.ndarray
    pupil_mm: float
    irradiance_gain: float
    defocus_sigma: float
    # The distance the lens was powered for, metres — ``None`` when
    # the eye rested at infinity or the body did not report one.
    focus_m: float | None = None
    # Eyelid transmission the frame was attenuated by — 1.0 open,
    # ~0.5 drowsy. Already applied to ``frame``; callers scale the
    # retinal image, not the gain twice.
    lid: float = 1.0


def irradiance_gain(pupil_mm: float) -> float:
    """Pupil-area gain on scene luminance: ``(d / d_ref)²``.

    ``pupil_mm <= 0`` is the unreported-aperture condition (a daemon
    that predates the field) — neutral gain, never a zero reading.
    """
    if pupil_mm <= 0.0 or not math.isfinite(pupil_mm):
        return 1.0
    d = min(pupil_mm, PUPIL_MAX_MM)
    return (d / REFERENCE_PUPIL_MM) ** 2


def retinal_irradiance(scene_brightness: float, pupil_mm: float) -> float:
    """Scene luminance through the aperture, normalized for the
    photoreceptor cascade's [0, 1] input.

    A constricted pupil dims what reaches the cones (the reflex
    compressing bright scenes into the receptors' adapted range); a
    dilated pupil passes the dark scene's spare photons — it cannot
    create photons the sensor never captured, so the boost saturates
    at full scale.
    """
    gain = irradiance_gain(pupil_mm)
    return min(1.0, max(0.0, scene_brightness * gain))


def defocus_sigma(pupil_mm: float, focus_m: float | None = None) -> float:
    """Blur radius the aperture's depth of field implies, in pixels.

    At or below ``_DEFOCUS_REF_MM`` the pupil is near-pinhole — the
    whole field is sharp. Toward ``PUPIL_MAX_MM`` the cone of
    accepted rays widens and off-focus structure smears.

    ``focus_m`` adds the accommodation term: depth of field thins as
    the commanded focus comes near (physically ∝ distance²), so the
    eye's point-spread grows when she is focused close. It is a
    *global* blur — without a depth map or fixation point the layer
    cannot know which structure is off the focal plane; what it
    models is the DOF penalty itself, not a localized bokeh.
    """
    sigma = 0.0
    if math.isfinite(pupil_mm) and pupil_mm > _DEFOCUS_REF_MM:
        frac = (pupil_mm - _DEFOCUS_REF_MM) / (PUPIL_MAX_MM - _DEFOCUS_REF_MM)
        sigma += _DEFOCUS_MAX_SIGMA * min(frac, 1.0)
    if focus_m is not None and math.isfinite(focus_m) and focus_m > 0.0:
        sigma += _FOCUS_MAX_SIGMA * min(1.0, _FOCUS_REF_M / focus_m)
    return sigma


def mirror_view(frame: np.ndarray) -> np.ndarray:
    """Flip the frame across the vertical axis — the mirror view.

    Applied at presentation, before the lens: the raw capture is a
    camera's view *of* the world; the mirror is her orientation *in*
    it — the view someone looking out at a face holds. Everything
    downstream (V1 orientations, region colours, object positions,
    two-view motion) then lives in the mirrored frame, so what she
    reports describes the world the way it faces her. Returns a
    contiguous copy — downstream OpenCV paths do not all accept the
    negative stride of a bare ``[:, ::-1]`` view.
    """
    if frame.ndim < 2 or frame.shape[1] == 0:
        return frame
    return np.ascontiguousarray(frame[:, ::-1])


def apply(
    frame: np.ndarray,
    pupil_mm: float,
    focus_m: float | None = None,
    lid: float = 1.0,
) -> OpticalResult:
    """Pass a captured frame through the eye's optics.

    Applies the aperture's point-spread (defocus blur) to the frame
    and reports the irradiance gain the caller folds into the
    photoreceptor input. ``lid`` attenuates the retinal image itself
    — the scalar model of eyelid transmission (the real lid's
    vignette is spatial; the uniform term is the honest first term).
    The frame is not resampled, inverted, or cropped — geometry
    belongs to the retinotopic map downstream. Facing orientation
    (the mirror view) is the caller's job, applied upstream via
    ``mirror_view``.
    """
    sigma = defocus_sigma(pupil_mm, focus_m)
    gain = irradiance_gain(pupil_mm)
    # Out-of-range or non-finite is not a lid reading — the eye is
    # open. Only a reported [0, 1] transmission dims the image.
    lid = lid if math.isfinite(lid) and 0.0 <= lid <= 1.0 else 1.0
    out = frame if lid >= 1.0 else (frame * lid).astype(frame.dtype)
    if sigma >= _DEFOCUS_MIN_SIGMA and out.ndim == 3 and out.shape[2] == 3:
        try:
            import cv2  # type: ignore[import-not-found]

            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        except ImportError:
            out = gaussian_blur_np(out, sigma)
    return OpticalResult(
        frame=out,
        pupil_mm=pupil_mm if math.isfinite(pupil_mm) else 0.0,
        irradiance_gain=gain,
        defocus_sigma=sigma,
        focus_m=focus_m if focus_m is not None and math.isfinite(focus_m) else None,
        lid=lid,
    )


def gaussian_blur_np(frame: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian in pure NumPy — the cv2-absent fallback.

    Same kernel shape as ``cv2.GaussianBlur`` (radius = round(3σ));
    ``mode="reflect"`` padding matches cv2's BORDER_REFLECT_101
    default edge handling. Works on (H, W) and (H, W, C) — only the
    two spatial axes are convolved.
    """
    radius = max(1, round(3.0 * sigma))
    # 'reflect' padding needs the pad inside the axis length; a blur
    # wider than the frame is DC — clamp rather than crash on a
    # small crop.
    radius = min(radius, max(1, min(frame.shape[0], frame.shape[1]) - 1))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    f = frame.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * f.ndim
        pad[axis] = (radius, radius)
        f = np.pad(f, pad, mode="reflect")
        windows = sliding_window_view(f, kernel.size, axis=axis)
        f = np.einsum("...k,k->...", windows, kernel)
    return f.astype(frame.dtype)

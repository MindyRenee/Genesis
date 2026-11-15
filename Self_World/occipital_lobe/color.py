"""Color as the nervous system computes it — cone excitations to opponent space.

The pipeline mirrors the physiology: display light is decomposed into
cone excitations first, then re-coded onto the opponent axes the LGN
actually carries, then compressed into the perceptual uniformities V4
represents:

    sRGB  →(inverse EOTF)→  linear RGB  →(M_sRGB)→  XYZ(D65)
    XYZ   →(M_HPE)→  LMS cone excitations
    LMS   →(von Kries, vs adapting white)→  cone-contrast space
          →  luminance L+M · red-green L−M · blue-yellow S−(L+M)

The opponent stage is a change of basis in *normalized* cone space —
the DKL construction (Derrington, Krauskopf & Lennie 1984, J.
Physiol. 357, 241): two isoluminant chromatic axes and one achromatic
axis meeting at the adapting white point. The von Kries step is the
physiological gain control — each cone class divided by its own
excitation under the adapting field — so the white point sits at the
origin of the opponent plane by construction.

Constants are drawn from the CIE-sanctioned standards: the sRGB
encoding (IEC 61966-2-1), the D65 illuminant, the Hunt–Pointer–
Estevez cone-response matrix (CIE 159:2004 Table 1), and CIELAB
(CIE 1976) for perceptual distance. Hue lives on a circle; its
summary statistics are circular statistics (mean resultant vector,
Mardia & Jupp) — never the linear mean of an angle.
"""

from __future__ import annotations

import numpy as np

# ── Forward transforms ───────────────────────────────────────────

# Linear-light sRGB → CIE XYZ, D65 (IEC 61966-2-1, Y normalized to 1;
# the standard white it yields is (0.95047, 1.0, 1.08883)).
_SRGB_TO_XYZ = np.array(
    [
        [0.4123908, 0.3575843, 0.1804808],
        [0.2126390, 0.7151687, 0.0721923],
        [0.0193308, 0.1191948, 0.9505322],
    ]
)

# XYZ → LMS cone excitations, Hunt–Pointer–Estevez (CIE 159:2004
# Table 1). HPE rather than the sharpened CAT02: this row space is the
# cone fundamentals' own, the physiological quantity the model needs,
# not the engineering fit CAT02 was optimized for.
_XYZ_TO_LMS = np.array(
    [
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0.00000, 0.00000, 1.00000],
    ]
)

# CIE D65 reference white on the Y=100 CIELAB scale.
D65_XYZ = np.array([0.95047, 1.0, 1.08883])

# CIELAB constants (CIE 1976): δ = 6/29, its cube, and the linear-segment
# slope 1/(3δ²) and offset 4/29 below it.
_LAB_DELTA3 = (6.0 / 29.0) ** 3
_LAB_LO_SLOPE = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
_LAB_LO_OFFSET = 4.0 / 29.0


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded intensities [0,1] to linear-light values.

    The inverse of the IEC 61966-2-1 electro-optical transfer
    function: a linear toe segment below 0.04045, a γ2.4 power above.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] (any leading shape) → CIE XYZ, Y≈1 scale."""
    linear = srgb_to_linear(rgb)
    return linear @ _SRGB_TO_XYZ.T


def xyz_to_lms(xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ → L, M, S cone excitations (Hunt–Pointer–Estevez)."""
    return np.asarray(xyz, dtype=np.float64) @ _XYZ_TO_LMS.T


def srgb_to_lms(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → L, M, S cone excitations."""
    return xyz_to_lms(srgb_to_xyz(rgb))


# ── Opponent coding — the DKL basis over cone contrast ───────────


def cone_opponent(
    rgb: np.ndarray,
    white_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cone-opponent channels for sRGB input, von Kries adapted.

    Args:
        rgb: sRGB in [0,1], shape (..., 3).
        white_rgb: the adapting field's white (sRGB [0,1]) — the frame
            mean serves as the illuminant estimate (grey-world); None
            adapts to the D65 reference white the display is calibrated
            for.

    Returns:
        (luminance, red_green, blue_yellow) arrays shaped rgb.shape[:-1]:
        ``luminance`` = cL + cM (achromatic, ≈1.0 at the white point),
        ``red_green`` = cL − cM (DKL constant-B axis, azimuth 0/180),
        ``blue_yellow`` = cS − (cL + cM)/2 (DKL constant-R&G axis,
        azimuth 90/270), where cK = K/K_white are the von Kries cone
        excitations — each cone class gain-normalized by the adapting
        field, which is what places the origin at white.
    """
    lms = srgb_to_lms(rgb)
    white_lms = (
        xyz_to_lms(srgb_to_xyz(np.asarray(white_rgb, dtype=np.float64)))
        if white_rgb is not None
        else xyz_to_lms(D65_XYZ)
    )
    white_lms = np.maximum(white_lms, 1e-9)
    cone = lms / white_lms
    lum = cone[..., 0] + cone[..., 1]
    rg = cone[..., 0] - cone[..., 1]
    by = cone[..., 2] - 0.5 * lum
    return lum, rg, by


def hue_angle(rg: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Azimuth on the isoluminant opponent plane, ∈ [0, 2π)."""
    return np.mod(np.arctan2(by, rg), 2.0 * np.pi)


def chroma(rg: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Eccentricity on the opponent plane — distance from white."""
    return np.hypot(rg, by)


# ── CIELAB — perceptual uniformity ───────────────────────────────


def xyz_to_lab(xyz: np.ndarray, white_xyz: np.ndarray = D65_XYZ) -> np.ndarray:
    """CIE XYZ (Y≈1) → CIELAB L*a*b* (CIE 1976, D65 by default).

    The cube-root compression models the cone signal's nonlinear
    response range; normalization by the adapted white is again von
    Kries. ``L*`` is lightness 0–100, ``a*``/``b*`` the opponent
    coordinates.
    """
    ratio = np.asarray(xyz, dtype=np.float64) / np.asarray(white_xyz, dtype=np.float64)
    f = np.where(
        ratio > _LAB_DELTA3,
        np.cbrt(np.maximum(ratio, 0.0)),
        _LAB_LO_SLOPE * ratio + _LAB_LO_OFFSET,
    )
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(rgb: np.ndarray, white_xyz: np.ndarray = D65_XYZ) -> np.ndarray:
    """sRGB [0,1] → CIELAB L*a*b*."""
    return xyz_to_lab(srgb_to_xyz(rgb), white_xyz)


def delta_e(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE — Euclidean distance in L*a*b*.

    The canonical perceptual-difference metric: one unit ≈ one
    just-noticeable difference. Used wherever "how different do these
    two colors *look*" is the question, not the pixel-distance proxy.
    """
    diff = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ── Circular statistics for hue ──────────────────────────────────


def circular_mean(angles: np.ndarray) -> float:
    """Mean direction of angles (radians) — the resultant vector's angle.

    A linear mean of hue is undefined: 350° and 10° average to 0°,
    not 180°. The circular mean is atan2 of the component sums.
    """
    angles = np.asarray(angles, dtype=np.float64)
    return float(np.mod(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()), 2.0 * np.pi))


def circular_resultant(angles: np.ndarray) -> float:
    """Mean resultant length R ∈ [0,1] — hue concentration.

    1.0 when every hue agrees (a saturated field), →0 as hues spread
    around the circle (an achromatic or mixed field). Doubles as the
    confidence that the circular mean means anything.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if angles.size == 0:
        return 0.0
    r = np.hypot(np.cos(angles).sum(), np.sin(angles).sum()) / angles.size
    return float(np.clip(r, 0.0, 1.0))


# ── Palette classification — rendered pixels to wire colors ──────


def palette_lab_table(palette_rgb: np.ndarray) -> np.ndarray:
    """Precompute CIELAB coordinates for a palette (N×3 sRGB [0,1])."""
    return srgb_to_lab(np.asarray(palette_rgb, dtype=np.float64))


def nearest_palette(rgb: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Nearest palette index per color, by ΔE.

    Args:
        rgb: (..., 3) sRGB [0,1] samples.
        palette_lab: (N, 3) CIELAB coordinates, from
            ``palette_lab_table`` — indexing palette index N.

    Returns:
        (...) int array of the argmin-ΔE palette index per sample.
        Classification in a perceptually uniform space — antialiasing,
        compositing, and gamma drift bend RGB distances where they
        don't bend perceived color.
    """
    lab = srgb_to_lab(np.asarray(rgb, dtype=np.float64))
    flat = lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - np.asarray(palette_lab, dtype=np.float64)[None, :, :]) ** 2).sum(
        axis=-1
    )
    return d2.argmin(axis=1).reshape(lab.shape[:-1])

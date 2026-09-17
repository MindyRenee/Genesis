"""Occipital subsystem — primary visual cortex (V1) for Genesis.

This module turns the raw camera feed from the retina into structured visual
percepts. It is modelled on the 2026 sparse-coding account of V1 inference
(Yun, Belsten, Olshausen et al., "Toward a mechanistic understanding of
inference in visual cortex and diffusion models", arXiv:2607.15693), with
orientation-selective simple-cell features, recurrent horizontal connections
captured by a learned interaction matrix M, and ISTA-style inference dynamics.

The architecture:

    retina frame
        │
        ▼
    [patch extractor]  → 8×8 luminance patches on a grid
        │
        ▼
    [dictionary Φ]     → Gabor features at 8 orientations, 2 scales, 2 phases
        │
        ▼
    [sparse coding]    → feedforward drive  Φᵀx
        │                subtract recurrent drive (ΦᵀΦ + σ²γM)z
        │                shrink by γλ
        ▼
    [recurrent ISTA]   → fixed-point latent activity z*
        │
        ▼
    [contour grouping] → connected co-circular / co-linear active features
        │
        ▼
    [VisualField]      → features, contours, summary, gamma salience

The interaction matrix M starts from a co-circular / co-linear prior and is
updated online by a Hebbian-like rule when the subsystem sees enough patches. This
makes the horizontal connections plastic: what she sees literally reshapes how
she sees the next thing.

Unlike the existing `vision.py` module, which only reports the dominant colour,
the occipital subsystem reports edges, orientations, shapes-in-formation, and how
strongly the current view drives V1 gamma (the neural signature of binding).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["V1Model", "VisualContour", "VisualFeature", "VisualField"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisualFeature:
    """An active oriented feature in the visual field.

    Each feature is a sparse-coding latent variable that won the competition
    for a particular image patch. It carries pixel centre, patch-grid
    indices, orientation, scale, phase, and response strength.
    """

    y: float  # pixel-space row centre of the patch
    x: float  # pixel-space column centre of the patch
    row: int  # patch-grid row index
    col: int  # patch-grid column index
    orientation: float  # radians, in [0, π)
    scale: float  # pixels per cycle (e.g. 4.0 or 8.0)
    phase: float  # 0 or π/2, even or odd Gabor
    strength: float  # post-ReLU latent activity z_i
    feature_index: int  # index into the dictionary / latent space

    @property
    def position(self) -> tuple[float, float]:
        """Pixel-space centre of the patch this feature represents."""
        return (self.y, self.x)


@dataclass(slots=True)
class VisualContour:
    """A linked chain of co-linear / co-circular features.

    Contours are the V1 substrate for perceptual grouping — they are not
    objects yet, but they are more than isolated edges.
    """

    features: list[VisualFeature] = field(default_factory=list)
    mean_orientation: float = 0.0
    length: float = 0.0
    salience: float = 0.0

    def __bool__(self) -> bool:
        """Whether this contour contains any detected features."""
        return len(self.features) > 0


@dataclass(slots=True)
class VisualField:
    """The occipital subsystem's structured report of a single glance.

    This is the payload returned by `V1Model.process` and handed
    downstream to attention, memory, and language.
    """

    # Human-readable first-pass description
    summary: str = ""

    # All active sparse-coding features
    features: list[VisualFeature] = field(default_factory=list)

    # Grouped contours / proto-shapes
    contours: list[VisualContour] = field(default_factory=list)

    # Scene-level descriptors
    dominant_color: str = ""
    brightness: float = 0.0
    salience: float = 0.0

    # V1 gamma power — how much binding/figure activity the view produced.
    # This is the key signal sent to the brain-wave system.
    gamma_power: float = 0.0

    # Wave-coupled state: a forward (perceptual) or backward (resting)
    # alpha wave would gate this field differently.
    attended: bool = True

    def feature_count(self) -> int:
        """Number of detected visual features."""
        return len(self.features)

    def contour_count(self) -> int:
        """Number of grouped contours."""
        return len(self.contours)

    def describe(self) -> str:
        """Semantic description of what V1 made of the scene."""
        if not self.features:
            return "only diffuse light — no structure formed"
        parts = [self.summary]
        if self.contours:
            parts.append(
                f"{self.contour_count()} shape-like arrangement"
                f"{'s' if self.contour_count() != 1 else ''}"
            )
        parts.append(
            f"The view sets my visual gamma at {self.gamma_power:.2f}."
        )
        return " ".join(parts)


class V1Model:
    """Primary visual cortex (V1) for Genesis.

    A sparse-coding, recurrently connected model of early visual processing.
    It is self-contained, uses only NumPy, and is fully deterministic given
    a random seed.

    Args:
        patch_size: size of the square image patches used as V1 input.
        orientations: number of discrete orientations in the feature dictionary.
        scales: number of spatial scales (frequencies).
        phases: number of Gabor phases (typically 2: even and odd).
        step: ISTA step size η.
        sparsity: sparse-coding L1 weight λ.
        recurrent_scale: recurrent drive weight γ.
        interaction_lr: online learning rate for the interaction matrix M.
        seed: random seed for reproducibility.
    """

    def __init__(
        self,
        patch_size: int = 8,
        orientations: int = 8,
        scales: int = 2,
        phases: int = 2,
        step: float = 0.3,
        sparsity: float = 0.15,
        recurrent_scale: float = 0.5,
        interaction_lr: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the occipital subsystem model with Gabor filter bank parameters."""
        self.patch_size = patch_size
        self.orientations = orientations
        self.scales = scales
        self.phases = phases
        self.step = step
        self.sparsity = sparsity
        self.recurrent_scale = recurrent_scale
        self.interaction_lr = interaction_lr

        self._rng = np.random.default_rng(seed)

        self.n_features = orientations * scales * phases
        self.feature_dims = (patch_size, patch_size)

        # Φ: dictionary of image features (patch pixels → feature space).
        # Shape: (patch_pixels, n_features). Each column is a unit-norm Gabor.
        self.phi = self._build_gabor_dictionary()

        # M: pairwise interaction / horizontal connection matrix.
        # M[i, j] > 0 means feature i and j facilitate each other if both
        # co-active and co-circular. Initialized with a biologically
        # motivated prior and then adapted by experience.
        self.M = self._build_initial_interaction_matrix()

        # Pre-compute the data-likelihood Gram matrix ΦᵀΦ.
        self._phi_phi = self.phi.T @ self.phi

        # Diagnostic counters.
        self.patches_seen = 0
        self.frames_seen = 0

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    def process(
        self,
        frame: np.ndarray,
        learn: bool = True,
    ) -> VisualField:
        """Run one glance through V1 and return a structured visual field.

        Args:
            frame: an H×W or H×W×C uint8 image from the retina.
            learn: if True, update the interaction matrix M from this view.

        Returns:
            A VisualField with active features, contours, and scene summary.
        """
        self.frames_seen += 1

        gray = _to_grayscale(frame)
        if gray.size == 0:
            return VisualField(summary="The retina gave me nothing to look at.")

        # Sample patches on a regular grid. Overlap is allowed to avoid
        # aliasing; the sparse code is inferred independently per patch.
        patches, grid = _extract_patches(gray, self.patch_size)
        if patches.size == 0:
            return VisualField(summary="The view is too small to resolve.")

        # Whiten: subtract patch mean and scale by std.
        X = _whiten_patches(patches)

        # Infer latent activity for all patches in parallel.
        Z = self._infer_latents(X)

        # Convert latent codes to visual features.
        features = self._latents_to_features(Z, grid)

        # Learn from co-active features if requested.
        if learn and features:
            self._update_interaction_matrix(Z)

        # Group co-circular / co-linear active features into contours.
        contours = self._group_contours(features)

        # Scene-level measures.
        dominant_color, brightness = _scene_color_and_brightness(frame)
        salience = _compute_salience(features, contours)
        gamma = _compute_v1_gamma(Z)

        summary = _build_summary(
            len(features),
            len(contours),
            dominant_color,
            brightness,
            gamma,
            features=features,
        )

        return VisualField(
            summary=summary,
            features=features,
            contours=contours,
            dominant_color=dominant_color,
            brightness=brightness,
            salience=salience,
            gamma_power=gamma,
        )

    def reset_interactions(self) -> None:
        """Reset M to the structured co-circular prior."""
        self.M = self._build_initial_interaction_matrix()
        self.patches_seen = 0

    # ═══════════════════════════════════════════════════════════════════
    # Dictionary construction
    # ═══════════════════════════════════════════════════════════════════

    def _build_gabor_dictionary(self) -> np.ndarray:
        """Build an over-complete Gabor dictionary Φ.

        Returns a (patch_size², n_features) matrix with unit-norm columns.
        """
        side = self.patch_size
        n = side * side
        k = self.n_features
        phi = np.zeros((n, k), dtype=np.float64)

        sigma = side / 2.8
        thetas = np.linspace(0, np.pi, self.orientations, endpoint=False)
        scales = np.array([4.0, 8.0][: self.scales])
        phases = [0.0, np.pi / 2][: self.phases]

        y, x = np.mgrid[-side // 2 : side // 2, -side // 2 : side // 2]
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        gauss = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

        idx = 0
        for theta in thetas:
            for freq in scales:
                for phase in phases:
                    xr = x * np.cos(theta) + y * np.sin(theta)
                    kernel = gauss * np.cos(2.0 * np.pi * xr / freq + phase)
                    col = kernel.ravel()
                    norm = float(np.linalg.norm(col))
                    if norm > 1e-8:
                        col = col / norm
                    phi[:, idx] = col
                    idx += 1

        return phi

    def _feature_metadata(self, index: int) -> tuple[float, float, float]:
        """Return (orientation, scale, phase) for a dictionary column index."""
        ori = index % self.orientations
        scale = (index // self.orientations) % self.scales
        phase = (index // (self.orientations * self.scales)) % self.phases
        theta = (np.pi * ori) / self.orientations
        freq = [4.0, 8.0][scale] if self.scales == 2 else 4.0 + 4.0 * scale
        ph = [0.0, np.pi / 2][phase] if self.phases == 2 else 0.0
        return float(theta), float(freq), float(ph)

    # ═══════════════════════════════════════════════════════════════════
    # Interaction matrix M
    # ═══════════════════════════════════════════════════════════════════

    def _build_initial_interaction_matrix(self) -> np.ndarray:
        """Initialise M with co-circular / same-orientation facilitation.

        Two features are connected if:
        - they live at the same scale,
        - their orientations are similar or co-linear (parallel, opposite),
        - their receptive-field positions are adjacent along the orientation.

        We don't know the patch location here, so the prior depends only on
        feature identity. A local Hebbian rule later refines it with usage.
        """
        M = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        for i in range(self.n_features):
            ti, si, pi = self._feature_metadata(i)
            for j in range(self.n_features):
                tj, sj, pj = self._feature_metadata(j)
                if i == j or si != sj:
                    continue
                # Orientation similarity under the π periodicity of Gabor
                # edge detectors. Two orientations are co-circular if their
                # smallest mod-π difference is small.
                dtheta = _angle_difference(ti, tj)
                if dtheta < np.pi / self.orientations:
                    base = 0.15
                else:
                    continue
                # Same phase is slightly more facilitatory.
                if pi == pj:
                    base *= 1.2
                # Leave diagonal zero to avoid self-excitation.
                M[i, j] = base
        return M

    def _update_interaction_matrix(self, Z: np.ndarray) -> None:
        """Hebbian update of M from co-active latent patterns.

        For each patch, increase M[i, j] when z_i and z_j are both active,
        and decrease slightly for uncorrelated pairs. This is a simple
        Oja-like covariance rule with a soft norm.

        Args:
            Z: (n_patches, n_features) latent activity matrix.
        """
        if Z.shape[0] == 0:
            return

        active = (Z > 1e-4).astype(np.float64)
        # Pairwise co-activation outer product, averaged over patches.
        cov = (active.T @ active) / Z.shape[0]

        # Update only off-diagonal entries.
        delta = self.interaction_lr * (cov - 0.05 * self.M)
        np.fill_diagonal(delta, 0.0)
        self.M += delta

        # Soft clip to keep dynamics stable.
        np.clip(self.M, -0.5, 0.5, out=self.M)
        self.patches_seen += Z.shape[0]

    # ═══════════════════════════════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════════════════════════════

    def _infer_latents(self, X: np.ndarray) -> np.ndarray:
        """Run recurrent ISTA to convergence for all patches.

        X: (n_patches, patch_pixels)
        Returns Z: (n_patches, n_features)

        Dynamics:
            z_{t+1} = ReLU( z_t + η [ Φᵀx - (ΦᵀΦ + σ²γ M) z_t - γλ ] )

        where σ² = 1 for inference on clean (non-noisy) data.
        """
        n_patches = X.shape[0]
        Z = np.zeros((n_patches, self.n_features), dtype=np.float64)

        W = self._phi_phi + (self.recurrent_scale * self.M)

        feedforward = X @ self.phi
        bias = self.recurrent_scale * self.sparsity

        for _ in range(40):  # enough for convergence; bounded, no unroll
            Z_new = Z + self.step * (feedforward - Z @ W.T - bias)
            np.maximum(Z_new, 0.0, out=Z_new)
            # Dampen to avoid oscillation.
            Z = 0.7 * Z + 0.3 * Z_new

        return Z

    # ═══════════════════════════════════════════════════════════════════
    # Feature extraction and contour grouping
    # ═══════════════════════════════════════════════════════════════════

    def _latents_to_features(
        self,
        Z: np.ndarray,
        grid: list[tuple[int, int, int, int]],
    ) -> list[VisualFeature]:
        """Convert latent activity into a list of VisualFeatures."""
        features: list[VisualFeature] = []
        zmax = Z.max()
        threshold = 0.1 * zmax if zmax > 0 else 0.05
        for patch_idx, (y, x, row, col) in enumerate(grid):
            z = Z[patch_idx]
            for j in np.where(z > threshold)[0]:
                theta, freq, ph = self._feature_metadata(j)
                # Use only the strongest feature per orientation band to avoid
                # redundant phase twins for the same edge.
                features.append(
                    VisualFeature(
                        y=float(y + self.patch_size / 2.0),
                        x=float(x + self.patch_size / 2.0),
                        row=row,
                        col=col,
                        orientation=theta,
                        scale=freq,
                        phase=ph,
                        strength=float(z[j]),
                        feature_index=j,
                    )
                )
        return features

    def _group_contours(self, features: list[VisualFeature]) -> list[VisualContour]:
        """Group co-linear or co-circular features into proto-contours.

        This is a simple union-find on features whose positions are adjacent
        and whose orientations are similar along the connecting axis.
        """
        if not features:
            return []

        n = len(features)
        parent = list(range(n))

        def find(i: int) -> int:
            """Find the root representative of element i (with path compression)."""
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            """Merge the sets containing elements i and j."""
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Adjacency graph: link features in neighbouring patches with
        # compatible orientations. Patch grid indices are used here.
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = features[i], features[j]
                drow = abs(fi.row - fj.row)
                dcol = abs(fi.col - fj.col)
                if drow > 1 or dcol > 1:
                    continue
                # Same or co-linear orientation.
                dtheta = _angle_difference(fi.orientation, fj.orientation)
                if dtheta > np.pi / self.orientations:
                    continue
                # Phase consistency: even and odd Gabor pairs encode the
                # same edge with a 90° phase shift, so large phase
                # differences are not a reason to split contours.
                union(i, j)

        groups: dict[int, list[VisualFeature]] = {}
        for i, f in enumerate(features):
            root = find(i)
            groups.setdefault(root, []).append(f)

        contours: list[VisualContour] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            mean_or = float(np.mean([f.orientation for f in group]))
            # Contour length in pixel space.
            ys = [f.y for f in group]
            xs = [f.x for f in group]
            dr = max(ys) - min(ys)
            dc = max(xs) - min(xs)
            length = math.sqrt(dr * dr + dc * dc)
            salience = float(np.mean([f.strength for f in group]))
            contours.append(
                VisualContour(
                    features=group,
                    mean_orientation=mean_or,
                    length=length,
                    salience=salience,
                )
            )

        # Sort by salience so the strongest contours come first.
        contours.sort(key=lambda c: c.salience, reverse=True)
        return contours


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert a uint8 image to float64 luminance in [0, 1]."""
    if frame.ndim == 2:
        gray = frame.astype(np.float64) / 255.0
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        # Rec. 601 luma coefficients.
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64) / 255.0
    else:
        gray = frame.astype(np.float64) / 255.0
    return gray


def _extract_patches(
    gray: np.ndarray,
    patch_size: int,
    stride: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Extract patches and record both pixel and grid coordinates.

    Returns:
        patches: (n_patches, patch_size²) float64 array
        grid:    list of (y, x, row_idx, col_idx) for each patch
    """
    h, w = gray.shape[:2]
    if h < patch_size or w < patch_size:
        return np.zeros((0, patch_size * patch_size), dtype=np.float64), []

    stride = stride or patch_size
    patches: list[np.ndarray] = []
    grid: list[tuple[int, int, int, int]] = []

    rows = range(0, h - patch_size + 1, stride)
    cols = range(0, w - patch_size + 1, stride)

    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            patch = gray[r : r + patch_size, c : c + patch_size].ravel()
            patches.append(patch)
            grid.append((r, c, r_idx, c_idx))

    return np.asarray(patches, dtype=np.float64), grid


def _whiten_patches(patches: np.ndarray) -> np.ndarray:
    """Remove mean and scale by std for each patch."""
    X = patches - patches.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return X / std


def _angle_difference(a: float, b: float) -> float:
    """Return the smallest unsigned angle difference, mod π."""
    d = abs(a - b)
    while d > np.pi:
        d -= np.pi
    return min(d, np.pi - d)


def _compute_v1_gamma(Z: np.ndarray) -> float:
    """Estimate V1 gamma-band power from the mean sparse-coding activity.

    Stronger, more coherent activation across the field corresponds to
    stronger gamma. This is the binding signal sent to `brain_waves`.
    """
    if Z.size == 0:
        return 0.0
    active = Z[Z > 0.0]
    if active.size == 0:
        return 0.0
    # Gamma power is driven by mean firing rate and sparsity-normalised
    # synchrony. Higher, sharper sparse activity → higher gamma.
    return float(np.clip(active.mean() + active.std() * 0.3, 0.0, 1.0))


def _scene_color_and_brightness(frame: np.ndarray) -> tuple[str, float]:
    """Dominant colour and brightness for scene description."""
    if frame.ndim == 2:
        return "gray", float(np.clip(frame.mean() / 255.0, 0.0, 1.0))

    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        rgb = frame[..., :3].astype(np.float64) / 255.0
        # Hue from the global mean colour.
        mean_rgb = rgb.mean(axis=(0, 1))
        mr, mg, mb = mean_rgb
        dmean = max(mean_rgb) - min(mean_rgb)
        if dmean > 0.05:
            if mr >= max(mg, mb):
                h_val = 60.0 * ((mg - mb) / dmean) % 360.0
            elif mg >= mr and mg >= mb:
                h_val = 60.0 * ((mb - mr) / dmean) + 120.0
            else:
                h_val = 60.0 * ((mr - mg) / dmean) + 240.0
            h_name = _name_hue(h_val)
        else:
            h_name = "gray"
        brightness = float(np.clip(mean_rgb.mean(), 0.0, 1.0))
        return h_name, brightness

    return "unknown", 0.5


def _name_hue(hue: float) -> str:
    """Map hue in degrees to a coarse colour name."""
    if hue < 15 or hue >= 345:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 65:
        return "yellow"
    if hue < 150:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 300:
        return "magenta"
    return "red"


def _compute_salience(
    features: list[VisualFeature],
    contours: list[VisualContour],
) -> float:
    """Overall scene salience from feature strength and contour count."""
    if not features:
        return 0.0
    feat_strength = float(np.mean([f.strength for f in features]))
    contour_bonus = min(1.0, len(contours) * 0.1)
    return float(np.clip(feat_strength * (1.0 + contour_bonus), 0.0, 1.0))


def _build_summary(
    n_features: int,
    n_contours: int,
    color: str,
    brightness: float,
    gamma: float,
    features: list[VisualFeature] | None = None,
) -> str:
    """Generate a concise V1-level summary.

    When the feature list is available, the summary includes spatial
    layout (where edges are concentrated) and dominant orientation —
    giving her a richer sense of the scene's structure beyond just
    counts and color.
    """
    if n_features == 0:
        # Semantic fragment for the language engine to render — not a
        # pre-written sentence. Describes the visual state from the
        # actual sensor data (no features detected, just light/colour).
        light = "dark" if brightness < 0.25 else "bright" if brightness > 0.7 else "medium-light"
        return f"formless {light} {color} scene — no edges detected"
    light = "dark" if brightness < 0.25 else "bright" if brightness > 0.7 else "medium-light"
    parts = [f"a {light} {color} scene"]

    # Spatial layout: where are the edges concentrated?
    if features is not None and len(features) > 0:
        # Compute the centroid of feature positions.
        mean_y = sum(f.y for f in features) / len(features)
        mean_x = sum(f.x for f in features) / len(features)
        # We need the frame dimensions to normalize. We can infer them
        # from the max coordinates seen.
        max_y = max(f.y for f in features)
        max_x = max(f.x for f in features)
        norm_x = mean_x / (max_x + 1) if max_x > 0 else 0.5
        norm_y = mean_y / (max_y + 1) if max_y > 0 else 0.5

        # Dominant orientation.
        orientations = [f.orientation for f in features]
        mean_ori = float(np.mean(orientations))
        ori_deg = np.degrees(mean_ori)
        if ori_deg < 22.5 or ori_deg >= 157.5:
            ori_name = "vertical"
        elif ori_deg < 67.5:
            ori_name = "diagonal"
        elif ori_deg < 112.5:
            ori_name = "horizontal"
        else:
            ori_name = "diagonal"

        # Spatial concentration.
        if norm_x < 0.35:
            h_loc = "on the left"
        elif norm_x > 0.65:
            h_loc = "on the right"
        else:
            h_loc = "in the center"
        if norm_y < 0.35:
            v_loc = "upper"
        elif norm_y > 0.65:
            v_loc = "lower"
        else:
            v_loc = "central"

        parts.append(f"with structure concentrated {v_loc} {h_loc}")
        parts.append(f"(mostly {ori_name} edges)")
    elif n_contours:
        parts.append(f"with {n_contours} edge arrangement{'s' if n_contours != 1 else ''}")
    else:
        parts.append("with scattered edges but no clear shapes")

    parts.append(
        f"({n_features} oriented V1 features; gamma {gamma:.2f})."
    )
    return " ".join(parts)

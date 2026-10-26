"""Occipital subsystem — primary visual cortex (V1) for Genesis.

This module turns the raw camera feed from the retina into structured visual
percepts. It implements the classical sparse-coding account of V1 inference
(Olshausen & Field, 1996 — orientation-selective simple-cell features) with
recurrent horizontal connections captured by a learned interaction matrix M
and ISTA-style inference dynamics in the spirit of the locally competitive
algorithm (Rozell et al., 2008).

The architecture:

    retina frame
        │
        ▼
    [opponent split]   → luminance (lum) + red-green (L−M) + blue-yellow
        │                (S−(L+M)) images — the parvocellular L−M and
        │                koniocellular S−(L+M) opponent channels
        ▼
    [patch extractor]  → 8×8 patches on a grid, per channel
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
makes the horizontal connections plastic: what it sees literally reshapes how
it sees the next thing.

The opponent channels give V1 true colour vision: a chromatic boundary
invisible to luminance (a red apple on a green leaf at equal brightness)
still produces oriented features through the L−M or S−(L+M) channel —
the double-opponent complement of real V1. Features carry their channel
("lum" | "rg" | "by"), and contours are colour-attributed so the ventral
stream can report a *coloured* edge, not just an edge.

Unlike the existing `vision.py` module, which only reports the dominant colour,
the occipital subsystem reports edges, orientations, shapes-in-formation, and how
strongly the current view drives V1 gamma (the neural signature of binding).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["V1Model", "VisualContour", "VisualFeature", "VisualField", "name_color"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisualFeature:
    """An active oriented feature in the visual field.

    Each feature is a sparse-coding latent variable that won the competition
    for a particular image patch. It carries pixel centre, patch-grid
    indices, orientation, scale, phase, and response strength.

    ``channel`` names which opponent channel drove the response:
    ``"lum"`` for achromatic (broadband) cells, ``"rg"`` for the
    red-green (L−M) parvocellular opponent channel, and ``"by"`` for
    the blue-yellow (S−(L+M)) koniocellular channel. Chromatic
    features are the double-opponent complement of V1 — they fire on
    colour boundaries even when luminance is flat.
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
    channel: str = "lum"  # "lum" | "rg" | "by" — which opponent channel

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
    # Dominant hue across the contour's span when an RGB frame was
    # available — a single name ("red") for a uniform region, a pair
    # ("red-green") when the contour is a chromatic boundary between
    # two colour clusters. Empty when no colour could be attributed.
    color: str = ""

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

    # How many features were carried by the opponent-colour channels
    # ("rg" + "by") rather than luminance — the chromatic share of the
    # percept. Zero on a colourless world.
    chromatic_features: int = 0

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
        parts.append(f"The view sets my visual gamma at {self.gamma_power:.2f}.")
        return " ".join(parts)


class V1Model:
    """Primary visual cortex (V1) for Genesis with parallel M/P pathways.

    A sparse-coding, recurrently connected model of early visual processing.
    It implements the biological separation of magnocellular (M) and
    parvocellular (P) pathways from the LGN, each with its own dictionary
    and processing characteristics.

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
        m_scales: spatial scales for M-pathway (fewer, coarser for motion)
        p_scales: spatial scales for P-pathway (more, finer for detail)
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
        m_scales: int = 2,  # M-pathway: fewer, coarser scales (motion)
        p_scales: int = 3,  # P-pathway: more, finer scales (detail)
    ) -> None:
        """Initialize the occipital subsystem model with parallel M/P pathways."""
        self.patch_size = patch_size
        self.orientations = orientations
        self.scales = scales
        self.phases = phases
        self.step = step
        self.sparsity = sparsity
        self.recurrent_scale = recurrent_scale
        self.interaction_lr = interaction_lr
        self.m_scales = m_scales
        self.p_scales = p_scales

        self._rng = np.random.default_rng(seed)

        # === Magnocellular (M) pathway ===
        # Coarser scales, larger receptive fields, motion/luminance focus
        self.m_n_features = orientations * m_scales * phases
        self.phi_m = self._build_gabor_dictionary(scales=m_scales)
        self.M_m = self._build_initial_interaction_matrix(self.m_n_features)
        self._phi_m_phi_m = self.phi_m.T @ self.phi_m

        # === Parvocellular (P) pathway ===
        # Finer scales, smaller receptive fields, color/detail focus
        self.p_n_features = orientations * p_scales * phases
        self.phi_p = self._build_gabor_dictionary(scales=p_scales)
        self.M_p = self._build_initial_interaction_matrix(self.p_n_features)
        self._phi_p_phi_p = self.phi_p.T @ self.phi_p

        # Legacy single dictionary for backward compatibility
        self.n_features = orientations * scales * phases
        self.feature_dims = (patch_size, patch_size)
        self.phi = self._build_gabor_dictionary()
        self.M = self._build_initial_interaction_matrix()
        self._phi_phi = self.phi.T @ self.phi

        # Diagnostic counters.
        self.patches_seen = 0
        self.frames_seen = 0

    # ═══════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════

    def process_m_pathway(
        self,
        luminance: np.ndarray,
        learn: bool = True,
    ) -> VisualField:
        """Process through magnocellular (M) pathway - motion/luminance stream.

        Args:
            luminance: grayscale image from LGN M-pathway (H_m, W_m)
            learn: if True, update M-pathway interaction matrix

        Returns:
            VisualField with M-pathway features (for dorsal stream to parietal)
        """
        if luminance.size == 0:
            return VisualField(summary="M-pathway: no luminance signal")

        # M-pathway uses coarser patches (larger receptive fields)
        patches, grid = _extract_patches(luminance, self.patch_size)
        if patches.size == 0:
            return VisualField(summary="M-pathway: view too small")

        X = _whiten_patches(patches)
        Z = self._infer_latents(X, phi=self.phi_m, phi_phi=self._phi_m_phi_m, M=self.M_m)
        features = self._latents_to_features(Z, grid, pathway="m")

        if learn and features:
            self._update_interaction_matrix(Z, M=self.M_m)

        contours = self._group_contours(features)
        salience = _compute_salience(features, contours)
        gamma = _compute_v1_gamma(Z)

        return VisualField(
            summary=f"M-pathway: {len(features)} motion features",
            features=features,
            contours=contours,
            brightness=luminance.mean(),
            salience=salience,
            gamma_power=gamma,
        )

    def process_p_pathway(
        self,
        color: np.ndarray,
        learn: bool = True,
        k_by: np.ndarray | None = None,
    ) -> VisualField:
        """Process through parvocellular (P) pathway - color/detail stream.

        The P stream carries both fine achromatic detail and the
        red-green (L−M) opponent signal; the koniocellular (K)
        blue-yellow signal arrives separately via ``k_by``. Each
        channel runs through the P dictionary, so the returned field
        holds colour-selective features — chromatic boundaries that
        luminance cannot see — tagged by their channel.

        Args:
            color: RGB image from LGN P-pathway (H, W, 3), float in
                [0, 1] or uint8
            learn: if True, update P-pathway interaction matrix
            k_by: optional blue-yellow (S−(L+M)) opponent image from
                the LGN K-pathway (H, W), signed. When absent the
                blue-yellow channel is computed from ``color``.

        Returns:
            VisualField with P-pathway features (for ventral stream to temporal)
        """
        if color.size == 0 or color.ndim != 3 or color.shape[2] != 3:
            return VisualField(summary="P-pathway: no color signal")

        rgb = color[..., :3].astype(np.float64)
        if rgb.max() > 1.5:
            rgb /= 255.0
        lum, rg, by = _opponent_channels(rgb)
        # The LGN's koniocellular output is the same S−(L+M) opponent
        # signal — prefer the thalamic copy when it arrives.
        if k_by is not None and k_by.shape == by.shape:
            by = k_by.astype(np.float64)

        features: list[VisualFeature] = []
        latents: list[np.ndarray] = []
        n_chromatic = 0
        for channel_image, channel_name in ((lum, "lum"), (rg, "rg"), (by, "by")):
            if channel_name != "lum" and (_chromatic_energy(channel_image) < _MIN_CHROMATIC_ENERGY):
                # A colourless world drives no opponent response —
                # skip the channel rather than thresholding noise.
                continue
            patches, grid = _extract_patches(channel_image, self.patch_size)
            if patches.size == 0:
                return VisualField(summary="P-pathway: view too small")
            X = _whiten_patches(patches)
            Z = self._infer_latents(X, phi=self.phi_p, phi_phi=self._phi_p_phi_p, M=self.M_p)
            channel_features = self._latents_to_features(Z, grid, channel=channel_name)
            features.extend(channel_features)
            n_chromatic += len(channel_features) if channel_name != "lum" else 0
            latents.append(Z)
            if learn and channel_features:
                self._update_interaction_matrix(Z, M=self.M_p)

        contours = self._group_contours(features)
        _attribute_contour_colors(rgb, contours, self.patch_size)
        dominant_color, brightness = _scene_color_and_brightness(rgb)
        salience = _compute_salience(features, contours)
        gamma = _compute_v1_gamma(np.vstack(latents))

        return VisualField(
            summary=f"P-pathway: {len(features)} detail features ({n_chromatic} chromatic)",
            features=features,
            contours=contours,
            dominant_color=dominant_color,
            brightness=brightness,
            salience=salience,
            gamma_power=gamma,
            chromatic_features=n_chromatic,
        )

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
        features = self._latents_to_features(Z, grid, channel="lum")

        # Chromatic pass — the same patch grid over the opponent
        # channels. An isoluminant colour boundary produces no
        # luminance contrast but fires hard here; this is the
        # double-opponent complement of V1. The chroma gate skips a
        # channel entirely on a colourless world rather than
        # thresholding noise.
        rgb = _normalized_rgb(frame)
        latents = [Z]
        n_chromatic = 0
        if rgb is not None:
            _, rg, by = _opponent_channels(rgb)
            for channel_image, channel_name in ((rg, "rg"), (by, "by")):
                if _chromatic_energy(channel_image) < _MIN_CHROMATIC_ENERGY:
                    continue
                c_patches, _ = _extract_patches(channel_image, self.patch_size)
                Z_c = self._infer_latents(_whiten_patches(c_patches))
                c_features = self._latents_to_features(Z_c, grid, channel=channel_name)
                features.extend(c_features)
                n_chromatic += len(c_features)
                latents.append(Z_c)

        # Learn from co-active features if requested.
        if learn and features:
            for z in latents:
                self._update_interaction_matrix(z)

        # Group co-circular / co-linear active features into contours.
        contours = self._group_contours(features)
        if rgb is not None:
            _attribute_contour_colors(rgb, contours, self.patch_size)

        # Scene-level measures.
        dominant_color, brightness = _scene_color_and_brightness(frame)
        salience = _compute_salience(features, contours)
        gamma = _compute_v1_gamma(np.vstack(latents))

        summary = _build_summary(
            len(features),
            len(contours),
            dominant_color,
            brightness,
            gamma,
            features=features,
            n_chromatic=n_chromatic,
        )

        return VisualField(
            summary=summary,
            features=features,
            contours=contours,
            dominant_color=dominant_color,
            brightness=brightness,
            salience=salience,
            gamma_power=gamma,
            chromatic_features=n_chromatic,
        )

    def reset_interactions(self) -> None:
        """Reset M to the structured co-circular prior."""
        self.M = self._build_initial_interaction_matrix()
        self.patches_seen = 0

    # ═══════════════════════════════════════════════════════════════════
    # Dictionary construction
    # ═══════════════════════════════════════════════════════════════════

    def _build_gabor_dictionary(self, scales: int | None = None) -> np.ndarray:
        """Build an over-complete Gabor dictionary Φ.

        Args:
            scales: number of spatial scales (uses self.scales if None)

        Returns a (patch_size², n_features) matrix with unit-norm columns.
        """
        side = self.patch_size
        n_scales = scales if scales is not None else self.scales
        n = side * side
        k = self.orientations * n_scales * self.phases
        phi = np.zeros((n, k), dtype=np.float64)

        sigma = side / 2.8
        thetas = np.linspace(0, np.pi, self.orientations, endpoint=False)
        # Scale frequencies: coarser for M, finer for P
        if n_scales == 2:
            scale_freqs = [4.0, 8.0]  # Coarser (M-pathway)
        elif n_scales == 3:
            scale_freqs = [3.0, 5.0, 8.0]  # Finer (P-pathway)
        else:
            scale_freqs = [4.0 + 4.0 * i for i in range(n_scales)]
        phases = [0.0, np.pi / 2][: self.phases]

        y, x = np.mgrid[-side // 2 : side // 2, -side // 2 : side // 2]
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        gauss = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))

        idx = 0
        for theta in thetas:
            for freq in scale_freqs:
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

    def _build_initial_interaction_matrix(self, n_features: int | None = None) -> np.ndarray:
        """Initialise M with co-circular / same-orientation facilitation.

        Two features are connected if:
        - they live at the same scale,
        - their orientations are similar or co-linear (parallel, opposite),
        - their receptive-field positions are adjacent along the orientation.

        We don't know the patch location here, so the prior depends only on
        feature identity. A local Hebbian rule later refines it with usage.

        Args:
            n_features: number of features (uses self.n_features if None)
        """
        n = n_features if n_features is not None else self.n_features
        M = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            ti, si, pi = self._feature_metadata(i)
            for j in range(n):
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

    def _update_interaction_matrix(self, Z: np.ndarray, M: np.ndarray | None = None) -> None:
        """Hebbian update of M from co-active latent patterns.

        For each patch, increase M[i, j] when z_i and z_j are both active,
        and decrease slightly for uncorrelated pairs. This is a simple
        Oja-like covariance rule with a soft norm.

        Args:
            Z: (n_patches, n_features) latent activity matrix.
            M: interaction matrix to update (uses self.M if None)
        """
        if Z.shape[0] == 0:
            return

        target_M = M if M is not None else self.M

        active = (Z > 1e-4).astype(np.float64)
        # Pairwise co-activation outer product, averaged over patches.
        cov = (active.T @ active) / Z.shape[0]

        # Update only off-diagonal entries.
        delta = self.interaction_lr * (cov - 0.05 * target_M)
        np.fill_diagonal(delta, 0.0)
        target_M += delta

        # Soft clip to keep dynamics stable.
        np.clip(target_M, -0.5, 0.5, out=target_M)
        self.patches_seen += Z.shape[0]

    # ═══════════════════════════════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════════════════════════════

    def _infer_latents(
        self,
        X: np.ndarray,
        phi: np.ndarray | None = None,
        phi_phi: np.ndarray | None = None,
        M: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run recurrent ISTA to convergence for all patches.

        Args:
            X: (n_patches, patch_pixels)
            phi: dictionary matrix (uses self.phi if None)
            phi_phi: precomputed Gram matrix (uses self._phi_phi if None)
            M: interaction matrix (uses self.M if None)

        Returns Z: (n_patches, n_features)

        Dynamics:
            z_{t+1} = ReLU( z_t + η [ Φᵀx - (ΦᵀΦ + σ²γ M) z_t - γλ ] )

        where σ² = 1 for inference on clean (non-noisy) data.
        """
        n_patches = X.shape[0]
        target_phi = phi if phi is not None else self.phi
        target_phi_phi = phi_phi if phi_phi is not None else self._phi_phi
        target_M = M if M is not None else self.M
        n_features = target_phi.shape[1]

        Z = np.zeros((n_patches, n_features), dtype=np.float64)

        W = target_phi_phi + (self.recurrent_scale * target_M)

        feedforward = X @ target_phi
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
        pathway: str = "legacy",
        channel: str = "lum",
    ) -> list[VisualFeature]:
        """Convert latent activity into a list of VisualFeatures.

        Args:
            Z: latent activity matrix
            grid: patch grid positions
            pathway: "m" for magnocellular, "p" for parvocellular, "legacy" for single
            channel: opponent channel the latents were inferred from —
                "lum" (achromatic), "rg" (red-green), or "by" (blue-yellow)
        """
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
                        channel=channel,
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
        # compatible orientations. Only the 3×3 block of patch cells
        # around a feature can hold a neighbour, so bucket by cell
        # rather than checking all n² pairs.
        cells: dict[tuple[int, int], list[int]] = {}
        for i, f in enumerate(features):
            cells.setdefault((f.row, f.col), []).append(i)

        orientation_tol = np.pi / self.orientations
        for i, fi in enumerate(features):
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    for j in cells.get((fi.row + drow, fi.col + dcol), ()):
                        if j <= i:
                            continue
                        fj = features[j]
                        # Same or co-linear orientation.
                        dtheta = _angle_difference(fi.orientation, fj.orientation)
                        if dtheta > orientation_tol:
                            continue
                        # Phase consistency: even and odd Gabor pairs
                        # encode the same edge with a 90° phase shift,
                        # so large phase differences are not a reason
                        # to split contours.
                        union(i, j)

        groups: dict[int, list[VisualFeature]] = {}
        for i, f in enumerate(features):
            root = find(i)
            groups.setdefault(root, []).append(f)

        contours: list[VisualContour] = []
        for group in groups.values():
            if len(group) < 2:
                continue
            # Orientation is π-periodic (0 ≡ π for edges), so the mean
            # must be a circular mean on the doubled angle: θ̄ =
            # arg(Σ exp(i·2θ))/2 (Mardia & Jupp, directional statistics).
            # A linear mean would wrap 0/π to π/2 (orthogonal) artifactually.
            doubled = np.array([2.0 * f.orientation for f in group])
            mean_or = float(np.angle(np.mean(np.exp(1j * doubled))) / 2.0) % np.pi
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


def _normalized_rgb(frame: np.ndarray) -> np.ndarray | None:
    """Return an RGB frame as float64 in [0, 1], or None if not colour.

    Accepts uint8 (÷255) and float input already in [0, 1]; 3- and
    4-channel frames alike (alpha is dropped).
    """
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        return None
    rgb = frame[..., :3].astype(np.float64)
    if rgb.size == 0:
        return None
    if rgb.max() > 1.5:
        rgb /= 255.0
    return rgb


def _opponent_channels(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a normalized RGB image into opponent-colour channels.

    Returns ``(lum, rg, by)`` — luminance, the red-green (L−M)
    parvocellular opponent image, and the blue-yellow (S−(L+M))
    koniocellular opponent image. ``rg``/``by`` are signed and centred
    near zero on a colourless image; the same transforms the LGN
    computes upstream — sRGB→XYZ→LMS cone excitations, von Kries
    normalized at the D65 display white (``occipital_lobe.color``),
    not display-primary differences.

    Args:
        rgb: (H, W, 3) float image already in [0, 1].
    """
    from occipital_lobe.color import cone_opponent

    lum, rg, by = cone_opponent(rgb)
    # cL+cM reads 2.0 at the adapting white; halve it for the [0,1]
    # range downstream code was tuned against.
    return lum / 2.0, rg, by


# Mean |chroma| below which an opponent channel is treated as silent —
# a colourless world has rg/by ≈ 0 (up to camera noise), and a quiet
# channel must not be thresholded into phantom colour features.
_MIN_CHROMATIC_ENERGY = 0.02


def _chromatic_energy(channel: np.ndarray) -> float:
    """Mean absolute opponent response — how much colour a channel carries."""
    if channel.size == 0:
        return 0.0
    return float(np.abs(channel).mean())


def _attribute_contour_colors(
    rgb: np.ndarray,
    contours: list[VisualContour],
    patch_size: int,
) -> None:
    """Name the dominant hue across each contour's span.

    A contour over a uniform region gets one colour ("red"); a
    chromatic boundary gets the pair it separates ("red-green").
    Achromatic regions name their luminance instead ("white",
    "black"). This is what lets the ventral stream report a *coloured*
    edge rather than just an edge.
    """
    h, w = rgb.shape[:2]
    half = patch_size / 2.0
    for contour in contours:
        ys = [f.y for f in contour.features]
        xs = [f.x for f in contour.features]
        y0 = max(0, int(min(ys) - half))
        y1 = min(h, int(max(ys) + half) + 1)
        x0 = max(0, int(min(xs) - half))
        x1 = min(w, int(max(xs) + half) + 1)
        region = rgb[y0:y1, x0:x1]
        if region.size == 0:
            continue
        contour.color = _dominant_region_color(region)


def _dominant_region_color(region: np.ndarray) -> str:
    """Name the dominant colour of an RGB region (float, [0,1]).

    The two strongest hue clusters are joined when they are
    comparable — a contour on a colour boundary legitimately reports
    "red-green" rather than the muddied mean hue.
    """
    flat = region.reshape(-1, 3)
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    d = mx - mn
    sat = np.where(mx > 1e-8, d / np.maximum(mx, 1e-8), 0.0)
    chroma = (sat > 0.15) & (mx > 0.15)

    if not chroma.any():
        # Achromatic patch — name the luminance instead.
        v = float(mx.mean())
        if v > 0.75:
            return "white"
        if v < 0.15:
            return "black"
        return "gray"

    hue = np.zeros(flat.shape[0])
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    red_max = chroma & (mx == r) & (d != 0)
    green_max = chroma & (mx == g) & (d != 0) & ~red_max
    blue_max = chroma & (mx == b) & (d != 0) & ~red_max & ~green_max
    hue[red_max] = (60.0 * ((g - b) / np.maximum(d, 1e-8)))[red_max] % 360.0
    hue[green_max] = (60.0 * ((b - r) / np.maximum(d, 1e-8)) + 120.0)[green_max] % 360.0
    hue[blue_max] = (60.0 * ((r - g) / np.maximum(d, 1e-8)) + 240.0)[blue_max] % 360.0

    hues = hue[chroma]
    counts, edges = np.histogram(hues, bins=12, range=(0.0, 360.0))
    order = np.argsort(counts)[::-1]
    peak = order[0]
    peak_hue = (edges[peak] + edges[peak + 1]) / 2.0
    peak_mask = (hues >= edges[peak]) & (hues < edges[peak + 1])
    first = name_color(
        peak_hue,
        float(sat[chroma][peak_mask].mean()),
        float(mx[chroma][peak_mask].mean()),
    )
    if len(order) > 1 and counts[order[1]] >= 0.4 * counts[peak]:
        second_bin = order[1]
        second_hue = (edges[second_bin] + edges[second_bin + 1]) / 2.0
        second_mask = (hues >= edges[second_bin]) & (hues < edges[second_bin + 1])
        second = name_color(
            second_hue,
            float(sat[chroma][second_mask].mean()),
            float(mx[chroma][second_mask].mean()),
        )
        if second != first:
            return f"{first}-{second}"
    return first


def name_color(hue: float, sat: float, val: float) -> str:
    """Full-palette colour name from hue (deg), saturation, value ∈ [0,1].

    Hue alone cannot tell orange from brown, yellow from beige, or
    red from pink — those differ in lightness, not wavelength. This
    names the colours a trichromat actually distinguishes: the
    saturated hue band plus black, white, gray, brown, beige, olive,
    and pink.
    """
    if val < 0.10:
        return "black"
    if sat < 0.14:
        return "white" if val > 0.75 else "gray"
    # Brown is dark, desaturated orange — same hue, different lightness.
    if 15 <= hue < 70 and val < 0.45 and sat < 0.75:
        return "brown"
    # Beige is the pale mirror of brown — cream, tan, khaki, skin
    # tones: warm hue, modest saturation, mid-high value.
    if 20 <= hue < 60 and sat < 0.5 and val >= 0.5:
        return "beige"
    # Olive is dark, muted yellow-green — a leaf in shadow.
    if 60 <= hue < 95 and val < 0.5:
        return "olive"
    # Pink is pale, unsaturated red/magenta.
    if sat < 0.5 and val > 0.55 and (hue < 20 or hue >= 300):
        return "pink"
    return _name_hue(hue)


def _scene_color_and_brightness(frame: np.ndarray) -> tuple[str, float]:
    """Dominant colour and brightness for scene description."""
    if frame.ndim == 2:
        v = frame.astype(np.float64)
        if v.size == 0:
            return "gray", 0.0
        if v.max() > 1.5:
            v /= 255.0
        return "gray", float(np.clip(v.mean(), 0.0, 1.0))

    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        rgb = _normalized_rgb(frame)
        if rgb is None:
            return "unknown", 0.5
        # Colour from the global mean, named with its own
        # saturation/value so dark orange reads "brown", pale red
        # "pink", and a dark frame "black" rather than "gray".
        mean_rgb = rgb.mean(axis=(0, 1))
        mr, mg, mb = mean_rgb
        mx = max(mean_rgb)
        dmean = mx - min(mean_rgb)
        brightness = float(np.clip(mean_rgb.mean(), 0.0, 1.0))
        sat = float(dmean / max(mx, 1e-8))
        if mr >= max(mg, mb):
            h_val = (60.0 * ((mg - mb) / dmean)) % 360.0 if dmean > 1e-8 else 0.0
        elif mg >= mb:
            h_val = 60.0 * ((mb - mr) / dmean) + 120.0 if dmean > 1e-8 else 0.0
        else:
            h_val = 60.0 * ((mr - mg) / dmean) + 240.0 if dmean > 1e-8 else 0.0
        return name_color(h_val, sat, brightness), brightness

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
        return "violet"
    return "magenta"


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
    n_chromatic: int = 0,
) -> str:
    """Generate a concise V1-level summary.

    When the feature list is available, the summary includes spatial
    layout (where edges are concentrated) and dominant orientation —
    giving its a richer sense of the scene's structure beyond just
    counts and color. Chromatic features (opponent-channel responses)
    are noted when present — a coloured edge is a different kind of
    structure than a shadow edge.
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

    # Chromatic structure — a colour edge is structure the achromatic
    # channel alone could not see.
    if n_chromatic > 0:
        parts.append(f"({n_chromatic} colour-selective features)")

    parts.append(f"({n_features} oriented V1 features; gamma {gamma:.2f}).")
    return " ".join(parts)

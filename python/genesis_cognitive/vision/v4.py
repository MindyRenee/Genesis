"""V4 — Mid-level visual cortex.

Pools V1 latents into larger receptive fields and learns a dictionary
of mid-level features (curvature, texture, shape complexes, color
blobs) via unsupervised sparse coding. See vision/__init__.py
for the full mathematical foundation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import default_data_dir

logger = logging.getLogger(__name__)

_DATA_DIR = default_data_dir() / "visual_cortex"
_V4_FILE = _DATA_DIR / "v4_dictionary.npz"

class V4Model:
    """Mid-level visual cortex — learns features from V1 latents.

    V4 pools V1 patch activations into larger receptive fields and
    learns a dictionary of mid-level features (curvature, texture
    patterns, shape complexes, color blobs) via unsupervised sparse
    coding. This mirrors how V4 in the primate brain integrates V1
    outputs into more complex shape and texture selectivity.

    The dictionary is learned from natural image statistics — the
    same principle as V1's Gabor dictionary, but operating on V1
    latents instead of pixels. Features that co-occur in natural
    scenes become V4 dictionary elements.

    Color channels are added at V4 (not V1) because V4 contains the
    earliest substantial population of color-selective neurons with
    large enough receptive fields to integrate color over regions.

    Architecture:
        V1 latents (n_patches × n_v1_features)
            │  pool 2×2 groups of adjacent patches
            ▼
        Pooled V1 (n_v4_units × n_v1_features × pool_size)
            │  concatenate with color statistics per pool
            ▼
        V4 input (n_v4_units × v4_input_dim)
            │  sparse code with learned dictionary Φ4
            ▼
        V4 latents (n_v4_units × n_v4_features)
    """

    def __init__(
        self,
        n_v1_features: int = 32,
        pool_size: int = 2,
        n_features: int = 64,
        sparsity: float = 0.1,
        step: float = 0.2,
        n_ista_iters: int = 30,
        color_dims: int = 12,
        seed: int | None = None,
    ) -> None:
        """Initialize V4.

        Args:
            n_v1_features: number of V1 latent features (from occipital subsystem).
            pool_size: how many V1 patches to pool into one V4 unit (per axis).
            n_features: number of V4 dictionary elements to learn.
            sparsity: L1 sparsity weight for V4 sparse coding.
            step: ISTA step size.
            n_ista_iters: number of ISTA iterations for inference.
            color_dims: dimensionality of the color statistics vector
                appended to each pooled unit.
            seed: random seed.
        """
        self.n_v1_features = n_v1_features
        self.pool_size = pool_size
        self.n_features = n_features
        self.sparsity = sparsity
        self.step = step
        self.n_ista_iters = n_ista_iters
        self.color_dims = color_dims

        # Input dimension: pooled V1 features + color stats
        self.input_dim = n_v1_features * (pool_size * pool_size) + color_dims

        self._rng = np.random.default_rng(seed)

        # Dictionary Φ4: (input_dim, n_features)
        # Initialized random, learned via online sparse coding
        self.phi4 = self._init_dictionary()

        # Gram matrix Φ4ᵀΦ4 (updated when dictionary changes)
        self._phi4_gram = self.phi4.T @ self.phi4

        # Learning state
        self._learning_enabled = True
        self._samples_seen = 0
        self._dict_update_interval = 50  # recompute Gram every N samples

    def _init_dictionary(self) -> np.ndarray:
        """Initialize dictionary with random unit-norm columns."""
        phi = self._rng.standard_normal((self.input_dim, self.n_features))
        norms = np.linalg.norm(phi, axis=0, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return phi / norms

    def process(
        self,
        v1_latents: np.ndarray,
        v1_grid_shape: tuple[int, int],
        frame: np.ndarray | None = None,
        learn: bool = True,
    ) -> np.ndarray:
        """Process V1 latents through V4.

        Args:
            v1_latents: (n_patches, n_v1_features) V1 latent activations.
            v1_grid_shape: (n_rows, n_cols) shape of the V1 patch grid.
            frame: optional original RGB frame for color statistics.
            learn: if True, update the dictionary.

        Returns:
            v4_latents: (n_v4_units, n_features) V4 latent activations.
        """
        n_rows, n_cols = v1_grid_shape
        if n_rows < self.pool_size or n_cols < self.pool_size:
            # Not enough V1 patches to pool — return zeros
            return np.zeros((0, self.n_features), dtype=np.float64)

        # Pool V1 latents: group 2×2 patches into one V4 unit
        pooled, pool_positions = self._pool_v1(v1_latents, n_rows, n_cols)

        # Add color statistics if frame is available
        if frame is not None and frame.ndim == 3 and frame.shape[2] >= 3:
            color_stats = self._compute_color_stats(frame, pool_positions, n_rows, n_cols)
            if color_stats.shape[0] == pooled.shape[0]:
                v4_input = np.hstack([pooled, color_stats])
            else:
                v4_input = pooled
        else:
            # Pad with zeros for color dims
            v4_input = np.hstack([
                pooled,
                np.zeros((pooled.shape[0], self.color_dims), dtype=np.float64),
            ])

        # Sparse code
        v4_latents = self._infer_latents(v4_input)

        # Learn dictionary
        if learn and self._learning_enabled and v4_input.shape[0] > 0:
            self._update_dictionary(v4_input, v4_latents)

        return v4_latents

    def _pool_v1(
        self,
        v1_latents: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Pool V1 patches into V4 units.

        Groups pool_size×pool_size adjacent V1 patches and concatenates
        their latent vectors. This gives V4 larger receptive fields.

        Returns:
            pooled: (n_v4_units, n_v1_features * pool_size²)
            positions: list of (row, col) top-left V1 grid position for each unit
        """
        ps = self.pool_size
        n_v4_rows = n_rows // ps
        n_v4_cols = n_cols // ps

        # Reshape V1 latents into grid form
        grid = v1_latents[:n_rows * n_cols].reshape(n_rows, n_cols, -1)

        # Trim to exact multiples
        grid = grid[:n_v4_rows * ps, :n_v4_cols * ps]
        grid = grid.reshape(n_v4_rows, ps, n_v4_cols, ps, -1)
        # Transpose to (n_v4_rows, n_v4_cols, ps, ps, n_features)
        grid = grid.transpose(0, 2, 1, 3, 4)
        # Flatten pool dimensions
        pooled = grid.reshape(n_v4_rows * n_v4_cols, -1)

        positions = [
            (r * ps, c * ps)
            for r in range(n_v4_rows)
            for c in range(n_v4_cols)
        ]

        return pooled.astype(np.float64), positions

    def _compute_color_stats(
        self,
        frame: np.ndarray,
        pool_positions: list[tuple[int, int]],
        n_v1_rows: int,
        n_v1_cols: int,
    ) -> np.ndarray:
        """Compute color statistics for each V4 pooling region.

        Extracts mean RGB, saturation, and hue histogram per region.
        This gives V4 the chromatic information that V1 lacks.
        """
        h, w = frame.shape[:2]
        rgb = frame[..., :3].astype(np.float64) / 255.0

        # V1 patch size — infer from grid
        patch_h = h / max(n_v1_rows, 1)
        patch_w = w / max(n_v1_cols, 1)

        stats = np.zeros((len(pool_positions), self.color_dims), dtype=np.float64)

        for i, (r, c) in enumerate(pool_positions):
            # Pixel region for this V4 unit
            y0 = int(r * patch_h)
            y1 = int((r + self.pool_size) * patch_h)
            x0 = int(c * patch_w)
            x1 = int(c + self.pool_size) * int(patch_w)
            y1 = min(y1, h)
            x1 = min(x1, w)

            region = rgb[y0:y1, x0:x1]
            if region.size == 0:
                continue

            # Mean RGB (3 dims)
            mean_rgb = region.mean(axis=(0, 1))
            # Std RGB (3 dims)
            std_rgb = region.std(axis=(0, 1))
            # HSV stats (6 dims): mean hue, mean sat, mean val, std hue, std sat, std val
            mean_hsv = self._rgb_to_hsv_stats(region)

            vec = np.concatenate([mean_rgb, std_rgb, mean_hsv])
            # Pad or truncate to color_dims
            if len(vec) < self.color_dims:
                vec = np.pad(vec, (0, self.color_dims - len(vec)))
            else:
                vec = vec[:self.color_dims]

            stats[i] = vec

        return stats

    @staticmethod
    def _rgb_to_hsv_stats(region: np.ndarray) -> np.ndarray:
        """Compute HSV statistics for an image region."""
        r, g, b = region[..., 0], region[..., 1], region[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        delta = mx - mn

        # Value
        v = mx
        # Saturation
        s = np.where(mx > 1e-8, delta / np.maximum(mx, 1e-8), 0.0)

        # Hue
        hue = np.zeros_like(mx)
        mask_r = (delta > 1e-8) & (mx == r)
        mask_g = (delta > 1e-8) & (mx == g) & ~mask_r
        mask_b = (delta > 1e-8) & (mx == b) & ~mask_r & ~mask_g

        hue[mask_r] = 60.0 * (((g - b) / np.maximum(delta, 1e-8))[mask_r] % 6.0)
        hue[mask_g] = 60.0 * (((b - r) / np.maximum(delta, 1e-8))[mask_g] + 2.0)
        hue[mask_b] = 60.0 * (((r - g) / np.maximum(delta, 1e-8))[mask_b] + 4.0)

        return np.array([
            np.mean(hue) / 360.0,
            np.mean(s),
            np.mean(v),
            np.std(hue) / 360.0,
            np.std(s),
            np.std(v),
        ])

    def _infer_latents(self, X: np.ndarray) -> np.ndarray:
        """Run ISTA sparse coding on V4 input.

        z_{t+1} = ReLU(z_t + η [Φᵀx - (ΦᵀΦ)z_t - λ])
        """
        if X.shape[0] == 0:
            return np.zeros((0, self.n_features), dtype=np.float64)

        Z = np.zeros((X.shape[0], self.n_features), dtype=np.float64)
        feedforward = X @ self.phi4
        bias = self.sparsity

        for _ in range(self.n_ista_iters):
            Z_new = Z + self.step * (feedforward - Z @ self._phi4_gram.T - bias)
            np.maximum(Z_new, 0.0, out=Z_new)
            Z = 0.8 * Z + 0.2 * Z_new

        return Z

    def _update_dictionary(self, X: np.ndarray, Z: np.ndarray) -> None:
        """Update V4 dictionary via gradient descent on reconstruction error.

        Φ4 += lr * (X - ZΦ4ᵀ)ᵀ Z  (gradient of ½‖X - Φ4 Zᵀ‖²)
        Then renormalize columns.
        """
        if Z.shape[0] == 0:
            return

        lr = 0.01
        recon = Z @ self.phi4.T  # (n_samples, input_dim)
        error = X - recon  # reconstruction error
        # Gradient: E^T Z → (input_dim, n_features)
        grad = error.T @ Z
        self.phi4 += lr * grad

        # Renormalize columns
        norms = np.linalg.norm(self.phi4, axis=0, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.phi4 = self.phi4 / norms

        self._samples_seen += Z.shape[0]
        if self._samples_seen % self._dict_update_interval == 0:
            self._phi4_gram = self.phi4.T @ self.phi4

    def save(self, path: Path = _V4_FILE) -> None:
        """Save V4 dictionary to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            phi4=self.phi4,
            phi4_gram=self._phi4_gram,
            samples_seen=self._samples_seen,
            n_v1_features=self.n_v1_features,
            pool_size=self.pool_size,
            n_features=self.n_features,
            color_dims=self.color_dims,
        )

    def load(self, path: Path = _V4_FILE) -> bool:
        """Load V4 dictionary from disk. Returns True if loaded."""
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            loaded_phi4 = data["phi4"]
            # Dimension check: the saved dictionary must match the
            # current V4 subsystem's input_dim. A mismatch happens when the
            # occipital subsystem configuration changes (e.g. different
            # number of orientations/scales/phases). Loading a
            # mismatched dictionary would cause matmul errors at
            # inference time.
            if loaded_phi4.shape[0] != self.input_dim:
                logger.warning(
                    f"V4 load skipped: saved input_dim "
                    f"{loaded_phi4.shape[0]} != current {self.input_dim}"
                )
                return False
            self.phi4 = loaded_phi4
            self._phi4_gram = data["phi4_gram"]
            self._samples_seen = int(data["samples_seen"])
            self.n_features = self.phi4.shape[1]
            self.input_dim = self.phi4.shape[0]
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"V4 load failed: {e}")
            return False

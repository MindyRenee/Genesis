"""VTC — Ventral temporal cortex.

Incremental PCA of V4 activations creates a low-dimensional feature
space whose axes capture the principal variations in mid-level visual
features. See vision/__init__.py for the full mathematical
foundation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DATA_DIR = Path.home() / ".local/share/genesis/visual_cortex"
_VTC_FILE = _DATA_DIR / "vtc_space.npz"

class VTCFeatureSpace:
    """Ventral temporal cortex — axis-based feature space.

    PCA of V4 activations creates a low-dimensional feature space
    whose axes capture the principal variations in mid-level visual
    features. This is the "neural feature space" described in the
    2026 Nature Communications paper: VTC neurons are organized along
    feature dimensions (shape, texture, color, curvature), and this
    space is where objects cluster by category.

    The space is learned incrementally — each new image updates the
    PCA basis via incremental PCA. This mirrors how the visual system
    refines its feature space through experience.

    The output is a fixed-dimensional vector (vtc_vector) that serves
    as the input to the MTL bridge.
    """

    def __init__(
        self,
        input_dim: int = 64,
        n_components: int = 32,
        forgetting_factor: float = 0.99,
    ) -> None:
        """Initialize VTC feature space.

        Args:
            input_dim: dimensionality of V4 latent vectors.
            n_components: number of PCA components (VTC axes).
            forgetting_factor: EMA forgetting factor for incremental
                updates (0.99 = slow adaptation, preserves long-term
                structure while allowing gradual refinement).
        """
        self.input_dim = input_dim
        self.n_components = n_components
        self.forgetting_factor = forgetting_factor

        # PCA state: mean, components, explained variance
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None  # (n_components, input_dim)
        self._n_samples: int = 0

        # Running covariance for incremental PCA
        self._scatter: np.ndarray | None = None  # accumulated scatter matrix

    def process(self, v4_latents: np.ndarray, learn: bool = True) -> np.ndarray:
        """Project V4 activations into the VTC feature space.

        Args:
            v4_latents: (n_v4_units, n_v4_features) V4 latent activations.
            learn: if True, update the PCA basis.

        Returns:
            vtc_vector: (n_components,) — the visual feature vector
                in VTC space. This is a global descriptor of the image,
                computed as the mean V4 activation projected through PCA.
        """
        if v4_latents.shape[0] == 0:
            return np.zeros(self.n_components, dtype=np.float64)

        # Global descriptor: mean V4 activation across all units
        # This gives a single vector summarizing the whole image
        descriptor = v4_latents.mean(axis=0)  # (n_v4_features,)

        if learn:
            self._update_pca(descriptor)

        if self._components is not None and self._mean is not None:
            # Project: center then apply PCA components
            centered = descriptor - self._mean
            vtc = self._components @ centered
            # L2 normalize
            norm = np.linalg.norm(vtc)
            if norm > 1e-8:
                vtc = vtc / norm
            return vtc
        else:
            # Not enough data for PCA yet — return raw descriptor
            # padded/truncated to n_components
            if len(descriptor) < self.n_components:
                return np.pad(descriptor, (0, self.n_components - len(descriptor)))
            return descriptor[:self.n_components]

    def _update_pca(self, descriptor: np.ndarray) -> None:
        """Incrementally update PCA with a new observation.

        Uses a simple incremental approach: accumulate the scatter
        matrix and recompute PCA periodically. This is O(d²) per
        update but d is small (64), so it's fast.
        """
        d = len(descriptor)
        if d != self.input_dim:
            # Dimension mismatch — skip
            return

        self._n_samples += 1

        if self._mean is None:
            self._mean = descriptor.copy()
            self._scatter = np.zeros((d, d), dtype=np.float64)
            return

        # Update mean with forgetting factor
        delta = descriptor - self._mean
        self._mean = self.forgetting_factor * self._mean + (1 - self.forgetting_factor) * descriptor

        # Update scatter matrix (accumulated centered outer products).
        # _scatter is always set alongside _mean (above), so it's not
        # None here — but mypy can't infer that from the early return.
        assert self._scatter is not None
        self._scatter = (
            self.forgetting_factor * self._scatter
            + np.outer(delta, delta)
        )

        # Recompute PCA every 10 samples (expensive but d is small)
        if self._n_samples % 10 == 0 and self._scatter is not None:
            try:
                # Eigendecomposition of scatter matrix
                eigenvalues, eigenvectors = np.linalg.eigh(self._scatter)
                # Sort descending
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]

                # Keep top n_components
                k = min(self.n_components, len(eigenvalues))
                self._components = eigenvectors[:, :k].T  # (k, d)
            except np.linalg.LinAlgError as e:
                logger.debug(f"VTC PCA fit failed, keeping old components: {e}")

    def save(self, path: Path = _VTC_FILE) -> None:
        """Save VTC PCA state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self._mean if self._mean is not None else np.array([]),
            components=self._components if self._components is not None else np.array([]),
            scatter=self._scatter if self._scatter is not None else np.array([]),
            n_samples=self._n_samples,
            n_components=self.n_components,
            input_dim=self.input_dim,
        )

    def load(self, path: Path = _VTC_FILE) -> bool:
        """Load VTC PCA state from disk. Returns True if loaded."""
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            mean = data["mean"]
            # Dimension check: saved mean must match current input_dim.
            if mean.size > 0 and mean.shape[0] != self.input_dim:
                logger.warning(
                    f"VTC load skipped: saved input_dim "
                    f"{mean.shape[0]} != current {self.input_dim}"
                )
                return False
            if mean.size > 0:
                self._mean = mean
            components = data["components"]
            if components.size > 0:
                self._components = components
            scatter = data["scatter"]
            if scatter.size > 0:
                self._scatter = scatter
            self._n_samples = int(data["n_samples"])
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"VTC load failed: {e}")
            return False

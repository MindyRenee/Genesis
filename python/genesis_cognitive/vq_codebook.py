"""Vector Quantization codebook — compresses Genesis's concept embedding space.

The concept network has thousands of concepts, each with a 210-dimensional
embedding (spectral + TF-IDF + GloVe). Most of these embeddings are
near-duplicates in semantic space — ``dog``, ``puppy``, ``canine`` cluster
tightly. Vector quantization replaces the full set of vectors with a
smaller **codebook** of prototype vectors. Each original vector is encoded
as ``(prototype_id, residual)`` — the prototype plus a small correction.

The residual is quantized to int8 to save space. The reconstruction is::

    vector ≈ codebook[prototype_id] + residual * residual_scale

This reduces storage from ``N × D × 4`` bytes (float32) to
``K × D × 4 + N × (4 + D)`` bytes (codebook + int32 id + int8 residual).
For 12K concepts, 210 dims, 2K prototypes::

    Before: 12,000 × 210 × 4 = 10.1 MB
    After:  2,000 × 210 × 4 + 12,000 × (4 + 210) = 1.7 MB + 2.6 MB = 4.3 MB

The codebook is retrained during algorithmic sleep (N3) via mini-batch
k-means. Concepts that share a prototype with tiny residuals are merge
candidates — the sleep compression pass can fold them together.

# Neuroscience grounding

This models the compression of episodic memories into semantic memory
during sleep consolidation. The hippocampus stores detailed, high-capacity
episodic traces; the neocortex stores compressed, schema-like semantic
knowledge. The codebook prototypes are the "schemas" — the typical
patterns — and the residuals are the episode-specific detail that
distinguishes one instance from another.

(Tononi & Cirelli, 2006; McClelland et al., 1995)
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class VQCodebook:
    """Vector quantization codebook for concept embeddings.

    Stores K prototype vectors and per-concept encoding (prototype_id +
    int8 residual). The codebook is trained via mini-batch k-means in
    pure numpy (no scipy/sklearn dependency).

    Thread-safe after training. All arrays are read-only post-fit.
    """

    # Default number of prototypes. 2K is sufficient for 10-20K concepts
    # with good reconstruction quality (>0.95 cosine similarity).
    DEFAULT_K = 2048

    # Number of k-means iterations during training.
    DEFAULT_ITERS = 25

    # Mini-batch size for k-means (controls memory during training).
    DEFAULT_BATCH_SIZE = 1024

    # Residual quantization scale. Residuals are stored as int8 in
    # [-residual_scale, +residual_scale]. A larger scale captures more
    # detail but risks int8 overflow. 0.1 is a good default for
    # L2-normalized embeddings where values are in [-1, 1].
    DEFAULT_RESIDUAL_SCALE = 0.1

    def __init__(
        self,
        dim: int = 210,
        k: int = DEFAULT_K,
        residual_scale: float = DEFAULT_RESIDUAL_SCALE,
    ) -> None:
        """Initialize the vector quantization codebook."""
        self.dim = dim
        self.k = k
        self.residual_scale = residual_scale

        # Trained state
        self._prototypes: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._prototype_ids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._residuals: np.ndarray = np.zeros((0, dim), dtype=np.int8)
        self._concept_names: list[str] = []

        # Statistics from last training
        self._reconstruction_error: float = 0.0
        self._training_iters: int = 0

    @property
    def is_trained(self) -> bool:
        """Whether the codebook has been trained."""
        return self._prototypes.shape[0] > 0

    @property
    def n_concepts(self) -> int:
        """Number of concepts currently encoded."""
        return len(self._concept_names)

    @property
    def reconstruction_error(self) -> float:
        """Mean squared reconstruction error from last training."""
        return self._reconstruction_error

    @property
    def prototypes(self) -> np.ndarray:
        """The K prototype vectors, shape (K, D)."""
        return self._prototypes

    @property
    def concept_names(self) -> list[str]:
        """Concept names in encoding order."""
        return self._concept_names

    # ─── Training ───────────────────────────────────────────────

    def fit(
        self,
        vectors: np.ndarray,
        concept_names: list[str],
        iters: int = DEFAULT_ITERS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 42,
    ) -> dict[str, float]:
        """Train the codebook via mini-batch k-means.

        Args:
            vectors: (N, D) float32 matrix of concept embeddings.
            concept_names: List of N concept names (parallel to vectors).
            iters: Number of k-means iterations.
            batch_size: Mini-batch size for k-means updates.
            seed: Random seed for reproducibility.

        Returns:
            Dict with training statistics: ``final_error``, ``iters``.
        """
        n, d = vectors.shape
        if d != self.dim:
            self.dim = d
        if n == 0:
            # No data to train on — set empty state
            self._prototypes = np.zeros((0, d), dtype=np.float32)
            self._prototype_ids = np.zeros((0,), dtype=np.int32)
            self._residuals = np.zeros((0, d), dtype=np.int8)
            self._concept_names = []
            self._reconstruction_error = 0.0
            self._training_iters = 0
            return {"final_error": 0.0, "iters": 0.0, "k": 0.0, "n": 0.0}
        k = min(self.k, n)  # can't have more prototypes than points

        rng = np.random.default_rng(seed)

        # Initialize prototypes via random sampling (k-means++ is
        # overkill for our scale; random init + enough iters works).
        indices = rng.choice(n, size=k, replace=False)
        prototypes = vectors[indices].copy().astype(np.float32)

        # Mini-batch k-means
        prev_error = float("inf")
        for iteration in range(iters):
            # Sample a mini-batch
            batch_n = min(batch_size, n)
            batch_idx = rng.choice(n, size=batch_n, replace=False)
            batch = vectors[batch_idx]

            # Assign each point to nearest prototype (Euclidean)
            # Using matrix operations: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            batch_sq = (batch ** 2).sum(axis=1, keepdims=True)  # (B, 1)
            proto_sq = (prototypes ** 2).sum(axis=1)  # (K,)
            cross = batch @ prototypes.T  # (B, K)
            dist_sq = batch_sq + proto_sq - 2 * cross  # (B, K)
            assignments = np.argmin(dist_sq, axis=1)  # (B,)

            # Update prototypes: move toward assigned points
            for ki in range(k):
                mask = assignments == ki
                if mask.any():
                    # Learning rate decays with iteration
                    lr = 1.0 / (1.0 + iteration * 0.1)
                    assigned = batch[mask].mean(axis=0)
                    prototypes[ki] = (1 - lr) * prototypes[ki] + lr * assigned

            # Compute reconstruction error on full dataset
            error = self._compute_error(vectors, prototypes)
            if abs(prev_error - error) < 1e-6:
                # Converged
                iters = iteration + 1
                break
            prev_error = error

        self._prototypes = prototypes
        self._training_iters = iters

        # Encode all vectors
        self._encode_all(vectors, concept_names)

        stats = {
            "final_error": self._reconstruction_error,
            "iters": float(self._training_iters),
            "k": float(k),
            "n": float(n),
        }
        logger.info(
            "VQ codebook trained: k=%d, n=%d, error=%.6f, iters=%d",
            k, n, self._reconstruction_error, self._training_iters,
        )
        return stats

    def _compute_error(self, vectors: np.ndarray, prototypes: np.ndarray) -> float:
        """Compute mean squared reconstruction error."""
        # Batch the distance computation to avoid memory issues
        n = vectors.shape[0]
        total_error = 0.0
        batch = 2048
        for i in range(0, n, batch):
            chunk = vectors[i:i + batch]
            chunk_sq = (chunk ** 2).sum(axis=1, keepdims=True)
            proto_sq = (prototypes ** 2).sum(axis=1)
            cross = chunk @ prototypes.T
            dist_sq = chunk_sq + proto_sq - 2 * cross
            min_dist = dist_sq.min(axis=1)
            total_error += min_dist.sum()
        return float(total_error / n)

    def _encode_all(self, vectors: np.ndarray, concept_names: list[str]) -> None:
        """Encode all vectors as (prototype_id, int8 residual)."""
        n = vectors.shape[0]

        # Assign each vector to nearest prototype
        prototype_ids = np.empty(n, dtype=np.int32)
        # Batch the assignment
        batch = 2048
        for i in range(0, n, batch):
            chunk = vectors[i:i + batch]
            chunk_sq = (chunk ** 2).sum(axis=1, keepdims=True)
            proto_sq = (self._prototypes ** 2).sum(axis=1)
            cross = chunk @ self._prototypes.T
            dist_sq = chunk_sq + proto_sq - 2 * cross
            prototype_ids[i:i + batch] = np.argmin(dist_sq, axis=1)

        # Compute residuals and quantize to int8
        reconstructed = self._prototypes[prototype_ids]
        residuals_f32 = vectors - reconstructed

        # Quantize: int8 range is [-128, 127]
        # residual_int8 = clip(round(residual / scale), -128, 127)
        scale = self.residual_scale
        residuals_int8 = np.clip(
            np.round(residuals_f32 / scale),
            -128, 127,
        ).astype(np.int8)

        self._prototype_ids = prototype_ids
        self._residuals = residuals_int8
        self._concept_names = list(concept_names)

        # Compute actual reconstruction error with quantized residuals
        recon = self._reconstruct_all()
        self._reconstruction_error = float(
            np.mean((vectors - recon) ** 2)
        )

    # ─── Encoding / Decoding ────────────────────────────────────

    def encode(self, vector: np.ndarray) -> tuple[int, np.ndarray]:
        """Encode a single vector as (prototype_id, int8 residual).

        Args:
            vector: (D,) float32 vector.

        Returns:
            Tuple of (prototype_id, residual_int8) where residual is
            (D,) int8.
        """
        if not self.is_trained:
            raise RuntimeError("codebook not trained")
        v = vector.astype(np.float32)
        # Find nearest prototype
        dist_sq = ((self._prototypes - v) ** 2).sum(axis=1)
        pid = int(np.argmin(dist_sq))
        residual = v - self._prototypes[pid]
        residual_int8 = np.clip(
            np.round(residual / self.residual_scale),
            -128, 127,
        ).astype(np.int8)
        return pid, residual_int8

    def decode(self, prototype_id: int, residual: np.ndarray) -> np.ndarray:
        """Reconstruct a vector from (prototype_id, residual).

        Args:
            prototype_id: Index into the codebook.
            residual: (D,) int8 residual.

        Returns:
            (D,) float32 reconstructed vector.
        """
        if not self.is_trained:
            raise RuntimeError("codebook not trained")
        return self._prototypes[prototype_id] + residual.astype(np.float32) * self.residual_scale

    def reconstruct(self, concept_name: str) -> np.ndarray | None:
        """Reconstruct the vector for a concept.

        Returns None if the concept is not in the codebook.
        """
        if not self.is_trained:
            return None
        try:
            idx = self._concept_names.index(concept_name)
        except ValueError:
            return None
        return self._reconstruct_at(idx)

    def _reconstruct_at(self, idx: int) -> np.ndarray:
        """Reconstruct vector at encoding index."""
        pid = self._prototype_ids[idx]
        residual = self._residuals[idx]
        return self._prototypes[pid] + residual.astype(np.float32) * self.residual_scale

    def _reconstruct_all(self) -> np.ndarray:
        """Reconstruct all vectors."""
        reconstructed = self._prototypes[self._prototype_ids]
        residuals = self._residuals.astype(np.float32) * self.residual_scale
        return reconstructed + residuals

    def reconstruct_all(self) -> np.ndarray:
        """Reconstruct all concept vectors, shape (N, D)."""
        if not self.is_trained:
            raise RuntimeError("codebook not trained")
        return self._reconstruct_all()

    # ─── Merge candidates ───────────────────────────────────────

    def find_merge_candidates(
        self,
        threshold: float = 0.02,
    ) -> list[tuple[str, str, float]]:
        """Find pairs of concepts that share a prototype with tiny residuals.

        These are near-duplicate concepts that could be merged during
        sleep consolidation. Two concepts are merge candidates if:
        - They share the same prototype
        - The L2 distance between their reconstructed vectors is below
          ``threshold``

        Args:
            threshold: Maximum L2 distance between reconstructed vectors
                to be considered merge candidates.

        Returns:
            List of (concept_a, concept_b, distance) tuples, sorted by
            distance ascending.
        """
        if not self.is_trained:
            return []

        # Group concepts by prototype
        proto_groups: dict[int, list[int]] = {}
        for i, pid in enumerate(self._prototype_ids):
            proto_groups.setdefault(int(pid), []).append(i)

        candidates: list[tuple[str, str, float]] = []
        recon = self._reconstruct_all()

        for _pid, indices in proto_groups.items():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = indices[i], indices[j]
                    dist = float(np.linalg.norm(recon[a] - recon[b]))
                    if dist < threshold:
                        candidates.append((
                            self._concept_names[a],
                            self._concept_names[b],
                            dist,
                        ))

        candidates.sort(key=lambda x: x[2])
        return candidates

    # ─── Persistence ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the codebook to an .npz file."""
        if not self.is_trained:
            raise RuntimeError("cannot save untrained codebook")
        np.savez(
            path,
            prototypes=self._prototypes,
            prototype_ids=self._prototype_ids,
            residuals=self._residuals,
            concept_names=np.array(self._concept_names, dtype=object),
            dim=np.array(self.dim, dtype=np.int32),
            k=np.array(self.k, dtype=np.int32),
            residual_scale=np.array(self.residual_scale, dtype=np.float32),
            reconstruction_error=np.array(self._reconstruction_error, dtype=np.float32),
        )

    def load(self, path: str) -> bool:
        """Load the codebook from an .npz file.

        Returns True if loaded, False if file doesn't exist or is
        incompatible.
        """
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            self._prototypes = data["prototypes"].astype(np.float32)
            self._prototype_ids = data["prototype_ids"].astype(np.int32)
            self._residuals = data["residuals"].astype(np.int8)
            self._concept_names = list(data["concept_names"])
            self.dim = int(data["dim"])
            self.k = int(data["k"])
            self.residual_scale = float(data["residual_scale"])
            self._reconstruction_error = float(data["reconstruction_error"])
            return True
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Failed to load VQ codebook: %s", e)
            return False

    # ─── Statistics ─────────────────────────────────────────────

    def compression_ratio(self) -> float:
        """Compute the compression ratio vs. storing full float32 vectors.

        Returns the ratio (uncompressed_size / compressed_size).
        A ratio of 2.0 means the codebook uses half the storage.
        """
        if not self.is_trained:
            return 1.0
        n = len(self._concept_names)
        uncompressed = n * self.dim * 4  # float32
        compressed = (
            self.k * self.dim * 4  # prototypes (float32)
            + n * 4  # prototype_ids (int32)
            + n * self.dim  # residuals (int8)
        )
        return uncompressed / compressed if compressed > 0 else 1.0

    def reconstruction_quality(self) -> float:
        """Return a proxy for reconstruction quality.

        Since the original vectors are not retained after training, a
        direct cosine similarity cannot be computed here. Instead, this
        returns ``1 - mean_squared_reconstruction_error``, clamped to
        ``[0, 1]``. Values close to 1.0 mean low reconstruction error
        (near-lossless compression); values near 0 mean high error.
        """
        if not self.is_trained:
            return 0.0
        return max(0.0, 1.0 - self._reconstruction_error)

    def stats(self) -> dict[str, float]:
        """Return a summary of codebook statistics."""
        return {
            "k": float(self.k) if self.is_trained else 0.0,
            "n_concepts": float(self.n_concepts),
            "dim": float(self.dim),
            "compression_ratio": self.compression_ratio(),
            "reconstruction_error": self._reconstruction_error,
            "residual_scale": self.residual_scale,
        }

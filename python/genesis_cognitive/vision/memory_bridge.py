"""MTL bridge — VTC to concept embedding projection.

Ridge-regression projection from VTC feature space to concept
embeddings, with one-shot learning and a generative path for
imagination. See vision/__init__.py for the full
mathematical foundation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import default_data_dir

logger = logging.getLogger(__name__)

_DATA_DIR = default_data_dir() / "visual_cortex"
_MTL_FILE = _DATA_DIR / "memory_bridge.npz"

#: Bound on retained training examples. Ridge refits run over all
#: examples on every association, and save() serializes all of them —
#: without a cap, a long-running mind's memory-bridge file grows
#: without bound. The most recent examples are kept.
_MAX_EXAMPLES = 500

@dataclass
class TrainingExample:
    """A single visual teaching example."""
    vtc_vector: list[float]  # VTC feature vector
    concept_name: str  # concept ID in the network
    timestamp: float
    correct: bool = True  # False if this was a correction (negative example)


class MemoryBridge:
    """Medial temporal subsystem bridge — VTC → concept embedding projection.

    This is the key transformation: dense visual feature representations
    in VTC space become sparse concept representations in the MTL. The
    bridge is a linear projection matrix W that maps VTC vectors to
    concept embedding space.

    Learning is one-shot: collect (vtc_vector, concept_embedding) pairs
    and solve least-squares: W = argmin ‖E - V Wᵀ‖² where E is the
    matrix of concept embeddings and V is the matrix of VTC vectors.

    This mirrors how the MTL rapidly encodes new associations —
    human medial-temporal neurons can form selective responses from
    a single exposure (single-trial encoding; Rutishauser et al.,
    2006).

    Associations form naturally: when Genesis reads a Wikipedia article
    about "tree" and sees the article's lead image, the VTC vector from
    that image is paired with the "tree" concept embedding. No manual
    teaching — she learns to see the same way she learns to read: by
    encountering images in context.
    """

    def __init__(
        self,
        vtc_dim: int = 32,
        embedding_dim: int = 193,
        ridge_lambda: float = 0.1,
    ) -> None:
        """Initialize MTL bridge.

        Args:
            vtc_dim: dimensionality of VTC feature vectors.
            embedding_dim: dimensionality of concept embeddings.
            ridge_lambda: L2 regularization for ridge regression
                (prevents overfitting with few examples).
        """
        self.vtc_dim = vtc_dim
        self.embedding_dim = embedding_dim
        self.ridge_lambda = ridge_lambda

        # Projection matrix W: (embedding_dim, vtc_dim)
        # Maps VTC vectors to concept embedding space
        self.W: np.ndarray | None = None

        # Training examples — bounded: every image-in-context appends
        # one, and save() serializes them all, so an unbounded list
        # would grow the state file and the O(n) refit forever.
        self._examples: list[TrainingExample] = []

        # Concept embedding cache (updated when W is refit)
        self._concept_embeddings: dict[str, np.ndarray] = {}

    def learn_association(
        self,
        vtc_vector: np.ndarray,
        concept_name: str,
        concept_embedding: np.ndarray,
    ) -> None:
        """Learn a visual-concept association from natural experience.

        This is called automatically when Genesis encounters an image
        in context (e.g., the lead image of a Wikipedia article about
        "tree"). The VTC vector from the image is paired with the
        concept embedding, and the projection matrix W is refit.

        Args:
            vtc_vector: VTC feature vector for the image.
            concept_name: concept ID (e.g., "tree").
            concept_embedding: embedding vector for the concept.
        """
        self._examples.append(TrainingExample(
            vtc_vector=vtc_vector.tolist(),
            concept_name=concept_name,
            timestamp=time.time(),
            correct=True,
        ))
        self._concept_embeddings[concept_name] = concept_embedding.copy()
        if len(self._examples) > _MAX_EXAMPLES:
            # Evict the oldest examples and drop embedding entries for
            # concepts that no longer have any example — the embedding
            # map must stay consistent with the example set because
            # _refit() indexes it per example.
            del self._examples[: len(self._examples) - _MAX_EXAMPLES]
            live = {ex.concept_name for ex in self._examples}
            for name in list(self._concept_embeddings):
                if name not in live:
                    del self._concept_embeddings[name]
        self._refit()

    def recognize(self, vtc_vector: np.ndarray) -> tuple[str | None, float]:
        """Recognize what concept an image corresponds to.

        Args:
            vtc_vector: VTC feature vector for the image.

        Returns:
            (concept_name, confidence) or (None, 0.0) if unrecognized.
            Confidence is cosine similarity to the nearest concept.
        """
        if self.W is None:
            return None, 0.0

        # Project VTC vector to concept embedding space
        predicted = self.W @ vtc_vector  # (embedding_dim,)

        # Find nearest concept by cosine similarity
        best_name = None
        best_sim = -1.0

        for name, embedding in self._concept_embeddings.items():
            sim = self._cosine_sim(predicted, embedding)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        return best_name, float(max(best_sim, 0.0))

    def recognize_all(self, vtc_vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k concept matches for a VTC vector."""
        if self.W is None:
            return []

        predicted = self.W @ vtc_vector
        results = []
        for name, embedding in self._concept_embeddings.items():
            sim = self._cosine_sim(predicted, embedding)
            results.append((name, float(max(sim, 0.0))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def generate(self, concept_embedding: np.ndarray) -> np.ndarray | None:
        """Generate a VTC vector from a concept embedding (top-down path).

        This is the generative direction: given a concept, predict what
        VTC features it should produce. Used for imagination and
        prediction in the predictive coding loop.

        Args:
            concept_embedding: embedding vector for the concept.

        Returns:
            Predicted VTC vector, or None if bridge isn't trained.
        """
        if self.W is None:
            return None
        # Pseudo-inverse: vtc ≈ W⁺ @ concept_embedding
        # W is (embedding_dim, vtc_dim), so W⁺ is (vtc_dim, embedding_dim)
        return self._W_pinv @ concept_embedding

    def _refit(self) -> None:
        """Refit the projection matrix W from all training examples.

        Uses ridge regression: W = Eᵀ V (Vᵀ V + λI)⁻¹
        where E = concept embeddings, V = VTC vectors.
        """
        # Skip examples whose concept embedding isn't available this
        # session — embeddings aren't persisted (W is retrained from
        # examples on load), so a restored example can name a concept
        # whose embedding hasn't been re-learned yet. Indexing it
        # directly would KeyError.
        positive = [
            ex for ex in self._examples
            if ex.correct and ex.concept_name in self._concept_embeddings
        ]
        if not positive:
            return

        # Build matrices
        V = np.array([ex.vtc_vector for ex in positive], dtype=np.float64)  # (n, vtc_dim)
        E = np.array([
            self._concept_embeddings[ex.concept_name]
            for ex in positive
        ], dtype=np.float64)  # (n, embedding_dim)

        if V.shape[0] == 0:
            return

        # Ridge regression: W = Eᵀ V (Vᵀ V + λI)⁻¹
        # W shape: (embedding_dim, vtc_dim)
        VtV = V.T @ V  # (vtc_dim, vtc_dim)
        reg = self.ridge_lambda * np.eye(self.vtc_dim)
        try:
            W = E.T @ V @ np.linalg.inv(VtV + reg)
            self.W = W
            # Compute pseudo-inverse for generative path
            self._W_pinv = np.linalg.pinv(W)  # (vtc_dim, embedding_dim)
        except np.linalg.LinAlgError:
            logger.debug("MTL bridge refit failed — singular matrix")

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def save(self, path: Path = _MTL_FILE) -> None:
        """Save MTL bridge to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "W": self.W.tolist() if self.W is not None else None,
            "vtc_dim": self.vtc_dim,
            "embedding_dim": self.embedding_dim,
            "ridge_lambda": self.ridge_lambda,
            "examples": [
                {
                    "vtc_vector": ex.vtc_vector,
                    "concept_name": ex.concept_name,
                    "timestamp": ex.timestamp,
                    "correct": ex.correct,
                }
                for ex in self._examples
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: Path = _MTL_FILE) -> bool:
        """Load MTL bridge from disk. Returns True if loaded."""
        if not path.exists():
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("W") is not None:
                loaded_W = np.array(data["W"], dtype=np.float64)
                # Dimension check: W must be (embedding_dim, vtc_dim).
                if loaded_W.shape != (self.embedding_dim, self.vtc_dim):
                    logger.warning(
                        f"MTL load skipped W: saved shape "
                        f"{loaded_W.shape} != current "
                        f"({self.embedding_dim}, {self.vtc_dim})"
                    )
                    # Clear the stale W in the saved file so this
                    # one-time migration warning doesn't recur on
                    # every startup. The W will be retrained from the
                    # loaded examples via _refit_bridge.
                    data["W"] = None
                    data["embedding_dim"] = self.embedding_dim
                    data["vtc_dim"] = self.vtc_dim
                    try:
                        with open(path, "w") as wf:
                            json.dump(data, wf)
                    except OSError as e:
                        logger.debug(f"MTL stale-W clear failed: {e}")
                else:
                    self.W = loaded_W
                    self._W_pinv = np.linalg.pinv(self.W)
            self._examples = [
                TrainingExample(
                    vtc_vector=ex["vtc_vector"],
                    concept_name=ex["concept_name"],
                    timestamp=ex["timestamp"],
                    correct=ex.get("correct", True),
                )
                for ex in data.get("examples", [])[-_MAX_EXAMPLES:]
            ]
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"MTL bridge load failed: {e}")
            return False

    @property
    def n_examples(self) -> int:
        """Number of training examples."""
        return len(self._examples)

    @property
    def n_concepts(self) -> int:
        """Number of distinct concepts learned."""
        return len(self._concept_embeddings)

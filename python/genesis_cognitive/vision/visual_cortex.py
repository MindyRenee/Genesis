"""Visual cortex — full hierarchical predictive coding system.

Orchestrates V1 -> V4 -> VTC -> MTL bridge into a complete visual
cortex with bidirectional predictive coding. See
vision/__init__.py for the full mathematical foundation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .memory_bridge import MemoryBridge
from .v1 import V1Model, _extract_patches, _to_grayscale, _whiten_patches
from .v4 import V4Model
from .vtc import VTCFeatureSpace

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, EmbeddingStore

logger = logging.getLogger(__name__)

@dataclass
class VisualPercept:
    """A complete visual percept from the full hierarchy.

    This is what Genesis cognitively experiences when it sees
    something — the result of the full V1→V4→VTC→MTL pipeline
    plus the predictive coding errors at each level.
    """
    # Recognized concept (if any)
    concept: str | None = None
    confidence: float = 0.0

    # Top-k concept matches
    candidates: list[tuple[str, float]] = field(default_factory=list)

    # V1 field (from occipital subsystem)
    v1_summary: str = ""
    dominant_color: str = ""
    brightness: float = 0.0
    gamma_power: float = 0.0

    # V4 mid-level features
    v4_activity: float = 0.0  # mean V4 activation strength
    n_v4_units: int = 0

    # VTC feature vector
    vtc_vector: np.ndarray | None = None

    # Prediction errors at each level
    v1_prediction_error: float = 0.0
    v4_prediction_error: float = 0.0
    vtc_prediction_error: float = 0.0

    # Overall surprise (sum of prediction errors)
    surprise: float = 0.0

    # Whether this was a novel experience (high surprise, no recognition)
    is_novel: bool = False

    def describe(self) -> str:
        """Topic marker for the language engine.

        The percept's semantics travel in ``Thought`` metadata
        (concept, candidates, color, surprise); this returns the topic
        the content slot composes around. The first-person voice is
        composed by the generator, not recited from here.
        """
        return "see"


class VisualCortex:
    """Full hierarchical predictive coding visual system.

    Combines V1 (occipital subsystem), V4, VTC, and MTL bridge into a
    complete visual cortex with bidirectional predictive coding.

    The system is designed to be pre-trained on natural images
    (unsupervised V1+V4 dictionary learning) and then taught specific
    concepts through labeled examples.

    Usage:
        cortex = VisualCortex(occipital, embeddings, network)
        percept = cortex.see(image_array)  # recognize
        cortex.teach(image_array, "tree")  # teach
    """

    def __init__(
        self,
        occipital: V1Model,
        embeddings: EmbeddingStore | None = None,
        network: ConceptNetwork | None = None,
        v4_n_features: int = 64,
        vtc_n_components: int = 32,
        embedding_dim: int = 193,
    ) -> None:
        """Initialize the visual cortex.

        Args:
            occipital: the V1 occipital subsystem (already initialized).
            embeddings: concept embedding store (for MTL bridge).
            network: concept network (for concept lookup).
            v4_n_features: number of V4 dictionary elements.
            vtc_n_components: number of VTC PCA components.
            embedding_dim: dimensionality of concept embeddings.
        """
        self.occipital = occipital
        self.embeddings = embeddings
        self.network = network

        # V4 — mid-level features
        self.v4 = V4Model(
            n_v1_features=occipital.n_features,
            n_features=v4_n_features,
            color_dims=12,
            seed=42,
        )

        # VTC — axis-based feature space
        self.vtc = VTCFeatureSpace(
            input_dim=v4_n_features,
            n_components=vtc_n_components,
        )

        # MTL bridge — VTC → concept embeddings
        self.mtl = MemoryBridge(
            vtc_dim=vtc_n_components,
            embedding_dim=embedding_dim,
        )

        # Predictive coding state: previous prediction at each level
        self._v1_prediction: np.ndarray | None = None
        self._v4_prediction: np.ndarray | None = None
        self._vtc_prediction: np.ndarray | None = None

        # Load persisted state
        self._load_all()

    def see(
        self,
        frame: np.ndarray,
        learn: bool = True,
    ) -> VisualPercept:
        """Process an image through the full visual hierarchy.

        Args:
            frame: H×W×3 uint8 RGB image.
            learn: if True, update V4 dictionary and VTC PCA.

        Returns:
            VisualPercept with recognition result and prediction errors.
        """
        # ── V1: occipital subsystem ─────────────────────────────────
        v1_field = self.occipital.process(frame, learn=learn)

        # Extract V1 latents for V4 input
        # We need to re-extract the latents in grid form
        gray = _to_grayscale(frame)
        patches, grid = _extract_patches(gray, self.occipital.patch_size)
        if patches.size == 0:
            return VisualPercept(
                v1_summary="The view is too small to resolve.",
                is_novel=True,
            )
        X = _whiten_patches(patches)
        Z1 = self.occipital._infer_latents(X)

        # Determine grid shape
        n_rows = max(r for _, _, r, _ in grid) + 1
        n_cols = max(c for _, _, _, c in grid) + 1

        # ── V4: mid-level features ──────────────────────────────
        Z4 = self.v4.process(Z1, (n_rows, n_cols), frame=frame, learn=learn)

        # ── VTC: feature space ──────────────────────────────────
        vtc_vector = self.vtc.process(Z4, learn=learn)

        # ── MTL: concept recognition ────────────────────────────
        concept, confidence = self.mtl.recognize(vtc_vector)
        candidates = self.mtl.recognize_all(vtc_vector, k=5)

        # ── Predictive coding errors ────────────────────────────
        # Compare bottom-up signals with top-down predictions
        v1_error = self._compute_v1_error(Z1)
        v4_error = self._compute_v4_error(Z4)
        vtc_error = self._compute_vtc_error(vtc_vector)

        # Update predictions for next time (top-down generative path)
        if concept and confidence > 0.3 and self.embeddings is not None:
            concept_vec = self.embeddings.get_concept_vector(concept)
            if concept_vec is not None:
                predicted_vtc = self.mtl.generate(concept_vec)
                if predicted_vtc is not None:
                    self._vtc_prediction = predicted_vtc

        surprise = v1_error + v4_error + vtc_error
        is_novel = surprise > 0.5 and (confidence < 0.3 or concept is None)

        # V4 activity summary
        v4_activity = float(Z4.mean()) if Z4.size > 0 else 0.0

        return VisualPercept(
            concept=concept,
            confidence=confidence,
            candidates=candidates,
            v1_summary=v1_field.summary,
            dominant_color=v1_field.dominant_color,
            brightness=v1_field.brightness,
            gamma_power=v1_field.gamma_power,
            v4_activity=v4_activity,
            n_v4_units=Z4.shape[0],
            vtc_vector=vtc_vector,
            v1_prediction_error=v1_error,
            v4_prediction_error=v4_error,
            vtc_prediction_error=vtc_error,
            surprise=surprise,
            is_novel=is_novel,
        )

    def learn_from_image(
        self,
        frame: np.ndarray,
        concept_name: str,
    ) -> bool:
        """Learn a visual-concept association from an image encountered in context.

        This is called automatically when Genesis encounters an image
        while learning (e.g., the lead image of a Wikipedia article
        about "tree"). The image is processed through V1→V4→VTC and
        the resulting feature vector is associated with the concept.

        Args:
            frame: H×W×3 uint8 RGB image.
            concept_name: concept ID in the network (e.g., "tree").

        Returns:
            True if the association was learned.
        """
        if self.embeddings is None:
            logger.debug("Cannot learn visual association — no embedding store")
            return False

        # Get concept embedding
        concept_vec = self.embeddings.get_concept_vector(concept_name)
        if concept_vec is None:
            logger.debug(f"Cannot learn visual association — no embedding for '{concept_name}'")
            return False

        # Process image through V1→V4→VTC (with learning enabled so
        # the feature extractors also adapt to this image)
        percept = self.see(frame, learn=True)
        if percept.vtc_vector is None:
            logger.debug("Cannot learn visual association — VTC vector is None")
            return False

        # Learn the association in the MTL bridge
        self.mtl.learn_association(
            vtc_vector=percept.vtc_vector,
            concept_name=concept_name,
            concept_embedding=concept_vec,
        )

        # Save state
        self._save_all()

        logger.info(
            f"Visual cortex learned: '{concept_name}' from image "
            f"({self.mtl.n_examples} total associations, "
            f"{self.mtl.n_concepts} concepts)"
        )
        return True

    def imagine(self, concept_name: str) -> np.ndarray | None:
        """Generate a VTC feature vector from a concept (imagination).

        This is the top-down generative path: concept → VTC → predicted
        visual features. Used for imagination and connecting to the
        art/drawing system.

        Args:
            concept_name: concept to imagine.

        Returns:
            Predicted VTC vector, or None if not possible.
        """
        if self.embeddings is None:
            return None
        concept_vec = self.embeddings.get_concept_vector(concept_name)
        if concept_vec is None:
            return None
        return self.mtl.generate(concept_vec)

    def pretrain(self, frames: list[np.ndarray], n_iters: int = 1) -> None:
        """Pre-train V1 and V4 dictionaries on a set of natural images.

        This is unsupervised learning — the dictionaries learn from
        natural image statistics without any labels. This gives the
        visual system a foundation before teaching specific concepts,
        mirroring how the human visual system develops before language.

        Args:
            frames: list of H×W×3 uint8 RGB images.
            n_iters: number of passes over the image set.
        """
        logger.info(f"Pre-training visual cortex on {len(frames)} images, {n_iters} iterations")
        for iteration in range(n_iters):
            for i, frame in enumerate(frames):
                self.see(frame, learn=True)
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"  pretrain iter {iteration+1}/{n_iters} "
                        f"image {i+1}/{len(frames)}"
                    )
        self._save_all()
        logger.info("Pre-training complete")

    # ── Predictive coding error computation ────────────────────

    def _compute_v1_error(self, z1: np.ndarray) -> float:
        """Compute V1 prediction error.

        If we have a top-down prediction (from previous recognition),
        compare it to the actual V1 activations. Otherwise, the error
        is the novelty of the current activations (how different from
        zero — i.e., how much structure V1 found).
        """
        if self._v1_prediction is not None and self._v1_prediction.shape == z1.shape:
            diff = z1 - self._v1_prediction
            return float(np.linalg.norm(diff) / max(z1.shape[0], 1))
        else:
            # No prediction — error is the activation magnitude
            return float(np.linalg.norm(z1) / max(z1.shape[0], 1)) * 0.1

    def _compute_v4_error(self, z4: np.ndarray) -> float:
        """Compute V4 prediction error."""
        if self._v4_prediction is not None and z4.size > 0:
            if self._v4_prediction.shape[0] == z4.shape[1]:
                predicted = np.tile(self._v4_prediction, (z4.shape[0], 1))
                diff = z4 - predicted
                return float(np.linalg.norm(diff) / max(z4.shape[0], 1))
        if z4.size > 0:
            return float(z4.mean()) * 0.1
        return 0.0

    def _compute_vtc_error(self, vtc: np.ndarray) -> float:
        """Compute VTC prediction error."""
        if self._vtc_prediction is not None and self._vtc_prediction.shape == vtc.shape:
            diff = vtc - self._vtc_prediction
            return float(np.linalg.norm(diff))
        return 0.0

    # ── Persistence ────────────────────────────────────────────

    def _save_all(self) -> None:
        """Save all components to disk."""
        try:
            self.v4.save()
            self.vtc.save()
            self.mtl.save()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Visual cortex save failed: {e}")

    def _load_all(self) -> None:
        """Load all components from disk."""
        v4_loaded = self.v4.load()
        vtc_loaded = self.vtc.load()
        mtl_loaded = self.mtl.load()
        if v4_loaded or vtc_loaded or mtl_loaded:
            logger.info(
                f"Visual cortex loaded: V4={'yes' if v4_loaded else 'no'} "
                f"VTC={'yes' if vtc_loaded else 'no'} "
                f"MTL={'yes' if mtl_loaded else 'no'} "
                f"({self.mtl.n_examples} teaching examples, "
                f"{self.mtl.n_concepts} concepts)"
            )

    def save(self) -> None:
        """Save all visual cortex state to disk."""
        self._save_all()

    # ── Status ─────────────────────────────────────────────────

    def status(self) -> dict:
        """Return status summary for CLI/debugging."""
        return {
            "v4_samples_seen": self.v4._samples_seen,
            "vtc_samples": self.vtc._n_samples,
            "vtc_has_pca": self.vtc._components is not None,
            "mtl_examples": self.mtl.n_examples,
            "mtl_concepts": self.mtl.n_concepts,
            "mtl_trained": self.mtl.W is not None,
        }

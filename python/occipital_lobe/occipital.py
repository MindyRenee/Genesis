"""OccipitalLobe — the ventral visual stream: V1 through V4/VTC.

The retinotopic pipeline (``Vision`` — V1 features, gamma band) is
grown at construction; the visual cortex association layers are grown
once the embedding space exists, since V4/VTC bind features to the
concept latent space.
"""

from __future__ import annotations

from occipital_lobe.perception.vision import Vision
from occipital_lobe.visual_cortex import VisualCortex


class OccipitalLobe:
    """The occipital region: sight becomes structure here.

    Attributes:
        vision: the V1 retinotopic pipeline (features, gamma, faces).
        visual_cortex: V4/VTC association (grown with the embeddings).
    """

    def __init__(self) -> None:
        # V1 — retinotopic feature extraction
        self.vision = Vision()
        # V4/VTC — grown when the latent space is available
        self.visual_cortex: VisualCortex | None = None

    def grow_cortex(self, embeddings, network) -> VisualCortex:
        """Grow V4/VTC association once the embedding space exists."""
        self.visual_cortex = VisualCortex(
            occipital=self.vision._get_occipital(),
            embeddings=embeddings,
            network=network,
        )
        return self.visual_cortex

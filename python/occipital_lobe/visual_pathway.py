"""Visual pathway orchestrator — retina → LGN → V1 → dorsal/ventral streams.

This module orchestrates the full biological visual pathway:
    retina → LGN (M/P/K separation) → V1 (parallel M/P processing) →
        dorsal stream (M → parietal lobe for motion/spatial)
        ventral stream (P → temporal lobe for object/face recognition)

This replaces the direct retina → V1 connection with the biologically
accurate thalamic relay and parallel pathway architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .lgn import LGNOutput, LGNRelay
from .v1 import V1Model, VisualField

if TYPE_CHECKING:
    from genesis_client.retina import Retina

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisualPathwayOutput:
    """Output from the full visual pathway with dorsal/ventral streams."""

    # LGN output (thalamus)
    lgn: LGNOutput | None = None

    # Dorsal stream output (M-pathway → parietal)
    # Motion, spatial location, visually-guided action
    dorsal_field: VisualField | None = None

    # Ventral stream output (P-pathway → temporal)
    # Object recognition, faces, color, form
    ventral_field: VisualField | None = None

    # Combined percept (for backward compatibility)
    combined_field: VisualField | None = None

    # Metadata
    processing_time_ms: float = 0.0
    frame_counter: int = 0


class VisualPathway:
    """Full biological visual pathway orchestrator.

    Coordinates retina → LGN → V1(M/P) → dorsal/ventral streams.

    Usage:
        retina = Retina()
        pathway = VisualPathway()
        output = pathway.process_from_retina(retina)
        # output.dorsal_field → parietal lobe (motion/spatial)
        # output.ventral_field → temporal lobe (object/face)
    """

    def __init__(
        self,
        lgn: LGNRelay | None = None,
        v1: V1Model | None = None,
    ):
        """Initialize the visual pathway.

        Args:
            lgn: LGN relay instance (created if None)
            v1: V1 model instance (created if None)
        """
        self.lgn = lgn if lgn is not None else LGNRelay()
        self.v1 = v1 if v1 is not None else V1Model()
        self._frame_counter = 0

    def process_from_retina(self, retina: Retina) -> VisualPathwayOutput:
        """Process a frame from retina through the full pathway.

        Args:
            retina: Retina instance to read from

        Returns:
            VisualPathwayOutput with dorsal and ventral stream outputs
        """
        import time

        t0 = time.time()

        # Step 1: Read from retina
        frame = retina.latest(copy=True)
        if frame is None:
            return VisualPathwayOutput()

        # Step 2: LGN relay (M/P/K separation)
        lgn_output = self.lgn.process(frame)

        # Step 3: V1 M-pathway (dorsal stream → parietal)
        dorsal_field = self.v1.process_m_pathway(lgn_output.m_luminance, learn=True)

        # Step 4: V1 P-pathway (ventral stream → temporal), fed the
        # koniocellular blue-yellow signal alongside the P colour
        ventral_field = self.v1.process_p_pathway(
            lgn_output.p_color, learn=True, k_by=lgn_output.k_blue_yellow
        )

        # Step 5: Combined field (backward compatibility)
        combined_field = self._combine_fields(dorsal_field, ventral_field)

        self._frame_counter += 1
        processing_time = (time.time() - t0) * 1000.0

        return VisualPathwayOutput(
            lgn=lgn_output,
            dorsal_field=dorsal_field,
            ventral_field=ventral_field,
            combined_field=combined_field,
            processing_time_ms=processing_time,
            frame_counter=self._frame_counter,
        )

    def process_frame(self, frame: np.ndarray) -> VisualPathwayOutput:
        """Process a raw frame through the full pathway.

        Args:
            frame: RGB image array (H, W, 3), dtype uint8

        Returns:
            VisualPathwayOutput with dorsal and ventral stream outputs
        """
        import time

        t0 = time.time()

        # Step 1: LGN relay
        lgn_output = self.lgn.process(frame)

        # Step 2: V1 M-pathway
        dorsal_field = self.v1.process_m_pathway(lgn_output.m_luminance, learn=True)

        # Step 3: V1 P-pathway, fed the koniocellular blue-yellow signal
        ventral_field = self.v1.process_p_pathway(
            lgn_output.p_color, learn=True, k_by=lgn_output.k_blue_yellow
        )

        # Step 4: Combined field
        combined_field = self._combine_fields(dorsal_field, ventral_field)

        self._frame_counter += 1
        processing_time = (time.time() - t0) * 1000.0

        return VisualPathwayOutput(
            lgn=lgn_output,
            dorsal_field=dorsal_field,
            ventral_field=ventral_field,
            combined_field=combined_field,
            processing_time_ms=processing_time,
            frame_counter=self._frame_counter,
        )

    def _combine_fields(self, dorsal: VisualField, ventral: VisualField) -> VisualField:
        """Combine dorsal and ventral fields into a unified percept.

        This provides backward compatibility with code that expects a single
        VisualField output from V1.
        """
        # Merge features from both pathways
        all_features = dorsal.features + ventral.features

        # Merge contours
        all_contours = dorsal.contours + ventral.contours

        # Use ventral color/brightness (P-pathway has full color)
        dominant_color = ventral.dominant_color
        brightness = ventral.brightness

        # Average salience and gamma
        salience = (dorsal.salience + ventral.salience) / 2.0
        gamma = (dorsal.gamma_power + ventral.gamma_power) / 2.0

        # Build combined summary
        summary_parts = []
        if dorsal.features:
            summary_parts.append(f"{len(dorsal.features)} motion features")
        if ventral.features:
            summary_parts.append(f"{len(ventral.features)} detail features")
        if all_contours:
            summary_parts.append(f"{len(all_contours)} contours")
        summary = " and ".join(summary_parts) if summary_parts else "no features detected"

        return VisualField(
            summary=summary,
            features=all_features,
            contours=all_contours,
            dominant_color=dominant_color,
            brightness=brightness,
            salience=salience,
            gamma_power=gamma,
        )


def create_visual_pathway() -> VisualPathway:
    """Factory function to create a visual pathway with default settings."""
    return VisualPathway()

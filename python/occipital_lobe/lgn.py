"""LGN (Lateral Geniculate Nucleus) — thalamic visual relay with parallel pathways.

The LGN is the thalamic relay station between the retina and primary visual cortex.
It organizes retinal signals into three parallel processing streams:

- **Magnocellular (M) pathway**: Luminance, motion, high temporal resolution,
  low spatial resolution. Receives input from parasol retinal ganglion cells.
  Projects to V1 layer 4Cα. Fast, transient responses.

- **Parvocellular (P) pathway**: Color, fine detail, high spatial resolution,
  high chromatic selectivity. Receives input from midget retinal ganglion cells.
  Projects to V1 layer 4Cβ. Slower, sustained responses.

- **Koniocellular (K) pathway**: Blue-yellow color, other non-M/P signals.
  Projects to V1 layers 2/3.

This module implements the LGN as a preprocessing layer that separates the
retina frame into these three parallel streams, each with appropriate spatial
and temporal filtering to match the biological properties of each pathway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from occipital_lobe.color import cone_opponent

if TYPE_CHECKING:
    from genesis_client.retina import Retina

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LGNOutput:
    """Output from the LGN relay — three parallel pathway streams.

    Each pathway carries a filtered version of the retinal input with
    properties matched to its biological role.
    """

    # Magnocellular pathway (motion/luminance)
    # - Luminance (grayscale) only, no color
    # - Spatially downsampled (lower resolution)
    # - Temporally enhanced (motion-sensitive)
    m_luminance: np.ndarray  # shape: (h_m, w_m), dtype: float32
    p_color: np.ndarray  # shape: (h, w, 3), dtype: float32
    # Blue-yellow opponent image S−(L+M), SIGNED: positive is blue,
    # negative is yellow. The sign is the information — clipping it
    # would delete the yellow half of the opponent axis.
    k_blue_yellow: np.ndarray  # shape: (h, w), dtype: float32

    # Timestamps
    m_timestamp: float = 0.0
    p_timestamp: float = 0.0
    k_timestamp: float = 0.0

    # Metadata
    frame_counter: int = 0
    processing_time_ms: float = 0.0


class LGNRelay:
    """Thalamic visual relay with M/P/K pathway separation.

    The LGN receives retinal input and separates it into three parallel
    streams that project to different layers of V1. This implements the
    biological organization of the primate LGN.

    Usage:
        retina = Retina()
        lgn = LGNRelay()
        while True:
            frame = retina.latest(copy=True)
            if frame is not None:
                output = lgn.process(frame)
                # output.m_luminance -> V1 M-pathway (dorsal stream)
                # output.p_color -> V1 P-pathway (ventral stream)
                # output.k_blue_yellow -> V1 K-pathway (layers 2/3)
    """

    def __init__(
        self,
        # M-pathway: downsample by factor (lower spatial resolution)
        m_downsample: int = 2,
        # Temporal enhancement for M-pathway (motion sensitivity)
        m_temporal_alpha: float = 0.3,
        # Chromatic enhancement for P-pathway
        p_chromatic_gain: float = 1.5,
    ):
        """Initialize the LGN relay.

        Args:
            m_downsample: Spatial downsampling factor for M-pathway (default 2)
            m_temporal_alpha: Temporal smoothing factor for M-pathway (0-1)
            p_chromatic_gain: Gain for chromatic channels in P-pathway
        """
        self.m_downsample = m_downsample
        self.m_temporal_alpha = m_temporal_alpha
        self.p_chromatic_gain = p_chromatic_gain

        # Temporal filter state for M-pathway (motion enhancement)
        self._m_prev_frame: np.ndarray | None = None
        self._frame_counter = 0

    def process(self, frame: np.ndarray) -> LGNOutput:
        """Process a retinal frame through the LGN relay.

        Args:
            frame: RGB image from retina, shape (H, W, 3), dtype uint8

        Returns:
            LGNOutput with three parallel pathway streams
        """
        import time

        t0 = time.time()

        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning(f"LGN received invalid frame shape: {frame.shape}")
            return self._neutral_output()

        h, w = frame.shape[:2]

        # Convert to float32 for processing
        frame_f32 = frame.astype(np.float32) / 255.0

        # === Magnocellular pathway (M) ===
        # Cone-space luminance L+M — the magnocellular input is the
        # achromatic channel, so it comes from the opponent pipeline
        # (sRGB→XYZ→LMS, von Kries at the D65 display white), not the
        # Rec.601 luma that weighs encoded primaries.
        luminance, rg, by = cone_opponent(frame_f32)
        # Normalize to the [0,1] range the rest of this pathway
        # expects: cL+cM reads 2.0 at the adapting white.
        luminance = luminance / 2.0

        # Spatial downsampling (lower resolution, larger receptive fields)
        if self.m_downsample > 1:
            h_m = h // self.m_downsample
            w_m = w // self.m_downsample
            # Block-average downsampling, vectorized: reshape to
            # (h_m, ds, w_m, ds) and take the patch mean.
            luminance_ds = (
                luminance[: h_m * self.m_downsample, : w_m * self.m_downsample]
                .reshape(h_m, self.m_downsample, w_m, self.m_downsample)
                .mean(axis=(1, 3))
            )
            m_raw = luminance_ds
        else:
            m_raw = luminance

        # Temporal enhancement (motion-sensitive): the difference is
        # taken against the previous *input* frame — storing the
        # enhanced blend would feed the motion term back into itself.
        if self._m_prev_frame is not None and self._m_prev_frame.shape == m_raw.shape:
            temporal_diff = np.abs(m_raw - self._m_prev_frame)
            m_luminance = (
                1 - self.m_temporal_alpha
            ) * m_raw + self.m_temporal_alpha * temporal_diff
        else:
            m_luminance = m_raw
        self._m_prev_frame = m_raw.copy()

        # === Parvocellular pathway (P) ===
        # Full RGB color with chromatic enhancement.
        # The opponent axes are the DKL cone-contrast channels computed
        # above — L−M and S−(L+M) in von Kries-normalized cone space.
        # Display-primary differences (R−G, B−(R+G)/2) only
        # approximate these: pure blue is invisible in R−G but drives
        # both L and M cones differently.
        p_color = frame_f32.copy()

        # Apply chromatic gain to color channels
        p_color[:, :, 0] += self.p_chromatic_gain * np.clip(rg, -0.5, 0.5)
        p_color[:, :, 1] -= self.p_chromatic_gain * np.clip(rg, -0.5, 0.5)
        p_color[:, :, 2] += self.p_chromatic_gain * np.clip(by, -0.5, 0.5)

        # Clip to valid range
        p_color = np.clip(p_color, 0.0, 1.0)

        # === Koniocellular pathway (K) ===
        # Blue-yellow opponent channel — kept signed (blue +, yellow
        # −). Downstream (V1) whitens patches, so the zero-centred
        # opponent signal is exactly what it wants; clipping here
        # would silently delete the yellow half of the axis.
        k_blue_yellow = by.copy()

        # === Build output ===
        self._frame_counter += 1
        processing_time = (time.time() - t0) * 1000.0

        return LGNOutput(
            m_luminance=m_luminance.astype(np.float32),
            m_timestamp=time.time(),
            p_color=p_color.astype(np.float32),
            p_timestamp=time.time(),
            k_blue_yellow=k_blue_yellow.astype(np.float32),
            k_timestamp=time.time(),
            frame_counter=self._frame_counter,
            processing_time_ms=processing_time,
        )

    def _neutral_output(self) -> LGNOutput:
        """Return a neutral output when no valid frame is available."""
        return LGNOutput(
            m_luminance=np.zeros((32, 32), dtype=np.float32),
            p_color=np.zeros((32, 32, 3), dtype=np.float32),
            k_blue_yellow=np.zeros((32, 32), dtype=np.float32),
        )


# The relay is stateful — the M pathway's motion enhancement diffs
# against the previous frame — so the convenience path shares one
# instance: a per-call relay would be permanently blind to motion.
_SHARED_RELAY: LGNRelay | None = None


def lgn_from_retina(retina: Retina, lgn: LGNRelay | None = None) -> LGNOutput | None:
    """Convenience function: read from retina and process through LGN.

    Args:
        retina: Retina instance to read from
        lgn: Relay to use; when omitted a process-shared instance keeps
            the M pathway's temporal state between calls.

    Returns:
        LGNOutput if a frame was available, None otherwise
    """
    frame = retina.latest(copy=True)
    if frame is None:
        return None

    global _SHARED_RELAY
    if lgn is None:
        if _SHARED_RELAY is None:
            _SHARED_RELAY = LGNRelay()
        lgn = _SHARED_RELAY
    return lgn.process(frame)

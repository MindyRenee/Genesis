"""Projection of MSTd two-view geometry into the dorsal stream.

The estimator is defined in ``occipital_lobe.epipolar`` — the MSTd
stage sits anatomically at the occipitotemporal border and feeds
parietal. Spatial attention, the world model, and the sensorimotor
systems consume its TwoViewReport; they do not contain MSTd itself.
"""

from occipital_lobe.epipolar import EpipolarProcessor, TwoViewReport

__all__ = ["EpipolarProcessor", "TwoViewReport"]

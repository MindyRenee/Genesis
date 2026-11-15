"""Compatibility projection of occipitotemporal MT/V5 into the dorsal stream.

The motion estimator is defined in ``occipital_lobe.motion``. Parietal
attention and sensorimotor systems consume its output; they do not
contain MT/V5 itself.
"""

from occipital_lobe.motion import MotionCell, MotionField, MotionProcessor, MotionRegion

__all__ = ["MotionCell", "MotionField", "MotionProcessor", "MotionRegion"]

"""Latent space — the one substrate.

There is a single latent space: the closed, positively-curved S^3
manifold in ``manifold.py`` — the holographic, rotating substrate the
daemon's fields are projections of and the mind's deposits live on.
Everything in this package is a facet of that one space: its
stereographic ``geometry``, the ``higher`` S^4/S^7 lifts, the
``spectrum`` band constants, the ``seed`` used to populate an empty
image, and ``self_geometry`` — the self-region's own geometric state
model (agency fibers, consistency, autopoietic flow) that encodes the
signals deposited at the manifold's self domain.
"""

from .higher import HigherStratum, geodesic_s7, hopf_to_s4, lift_s3_to_s7, rotate_s7
from .manifold import STRAND_ANTI, STRAND_HOPF, LatentManifold
from .spectrum import FIELD_BAND_HZ, SPECTRUM_STEP_HZ

__all__ = [
    "FIELD_BAND_HZ",
    "SPECTRUM_STEP_HZ",
    "STRAND_ANTI",
    "STRAND_HOPF",
    "HigherStratum",
    "LatentManifold",
    "geodesic_s7",
    "hopf_to_s4",
    "lift_s3_to_s7",
    "rotate_s7",
]

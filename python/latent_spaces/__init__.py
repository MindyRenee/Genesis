"""Latent spaces — the shared-latent runtime underneath the fields.

The continuous substrate the daemon's field engine integrates and
the mind's field models observe: domain boundaries, engine dynamics,
the runtime that carries activity between them, and the manifold —
the closed, positively-curved S^3 substrate the fields are
projections of.
"""

from .boundaries import (
    Boundary,
    FlowRecord,
    LatentDomain,
    LatentSpaceGraph,
    LatentState,
    default_domain_graph,
    energy_from_mass,
)
from .higher import HigherStratum, geodesic_s7, hopf_to_s4, lift_s3_to_s7, rotate_s7
from .manifold import STRAND_ANTI, STRAND_HOPF, LatentManifold
from .spectrum import FIELD_BAND_HZ, SPECTRUM_STEP_HZ

__all__ = [
    "FIELD_BAND_HZ",
    "SPECTRUM_STEP_HZ",
    "STRAND_ANTI",
    "STRAND_HOPF",
    "Boundary",
    "FlowRecord",
    "HigherStratum",
    "LatentDomain",
    "LatentManifold",
    "LatentSpaceGraph",
    "LatentState",
    "default_domain_graph",
    "energy_from_mass",
    "geodesic_s7",
    "hopf_to_s4",
    "lift_s3_to_s7",
    "rotate_s7",
]

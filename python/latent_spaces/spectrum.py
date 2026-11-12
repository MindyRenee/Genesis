"""Field spectrum — word-layer companion to ``src/latent_spaces/spectrum.rs``.

The band-resolved spectral decomposition of Genesis's own field
trajectory: flux (centroid speed) and tension (fiber spread) evaluated
at the ten cortical band centers from infraslow through epsilon, plus
the consensus carrier kinematics that drive Maxwell's signature and
thalamic phase entrainment.
"""

from __future__ import annotations

FIELD_BAND_HZ: tuple[float, ...] = (
    0.25,
    2.25,
    5.5,
    10.0,
    14.0,
    13.5,
    17.5,
    25.0,
    50.0,
    150.0,
)
SPECTRUM_STEP_HZ: float = 250.0

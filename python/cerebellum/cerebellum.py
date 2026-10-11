"""Cerebellum — the forward model: prediction-error machinery.

Cerebellar circuits contribute to predictive motor and cognitive
processing through cerebellar output pathways. This reduced forward
model forecasts perceptual context and learns from prediction errors.
Its callback requests modulation from the daemon's shared chemical
system; it does not represent cerebellar synthesis of dopamine or
norepinephrine, nor the detailed deep-nuclear/thalamocortical circuit.
"""

from __future__ import annotations

from .predictive import PredictiveCodingLayer


class Cerebellum:
    """The cerebellar region: supervised error-driven learning.

    Attributes:
        predictive_coding: the forward model — predict, compare,
            correct.
    """

    def __init__(self, on_neuro_impulse) -> None:
        self.predictive_coding = PredictiveCodingLayer(
            on_neuro_impulse=on_neuro_impulse,
        )

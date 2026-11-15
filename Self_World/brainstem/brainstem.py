"""Brainstem — arousal, autonomic monitoring, the oldest part.

The reticular core: the module EEG that watches which subsystem is
firing (``ModuleSampler``) lives here, alongside the heartbeat,
lifecycle, and sleep-wake mixins that implement the autonomic loop.
"""

from __future__ import annotations

from brainstem.module_sampler import ModuleSampler


class Brainstem:
    """The brainstem arousal component of the distributed sleep-wake circuit.

    Attributes:
        sampler: the module EEG — reticular monitoring of which
            subsystem is firing.
    """

    def __init__(self, credit) -> None:
        self.sampler = ModuleSampler(credit)

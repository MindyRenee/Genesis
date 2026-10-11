"""ParietalLobe — attention, the situation model, and spatial imagery.

The dorsal attention network (IPS substrate), the where/what-if world
model, and the spatial substrate (grid, scene, transforms, imagery)
live here.
"""

from __future__ import annotations

from parietal_lobe.attention import AttentionSystem
from parietal_lobe.situation_model import SituationModel


class ParietalLobe:
    """The parietal region: the where-pathway and attentional control.

    Attributes:
        attention: dorsal attention network — selects what reaches cortex.
        situation_model: the running model of the current scene.
    """

    def __init__(self, network) -> None:
        self.network = network
        # Dorsal attention network — IPS substrate
        self.attention = AttentionSystem(network=network)
        # The running world model — entities, relations, affordances
        self.situation_model = SituationModel()

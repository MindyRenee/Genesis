"""Spatial — the perceptual-symbolic layer of Genesis's reasoning.

Semantic reasoning traverses relations between *concepts*. This
package gives Genesis the complementary capacity: perceiving structure
in spatial input and reasoning about it symbolically — the parietal
"where/how" stream made explicit.

Pipeline:

    Grid (raw cells)
        |
        v
    perceive() -> Scene           objects + spatial relations
        |                           (the perceptual parse)
        +--> ground_scene()        Scene -> concept network edges
        |                           (semantic grounding)
        v
    SpatialReasoner.solve()        Transform hypotheses, verified
        |                           against every example
        v
    SpatialSolution                rule + predictions

Consumers:
    - The cognition engine holds a ``SpatialReasoner`` instance so
      spatial problems raised in conversation can be perceived and
      reasoned about, not just talked past.
"""

from __future__ import annotations

from parietal_lobe.grid import Grid
from parietal_lobe.grounding import ground_scene, scene_facts, scene_relations_for
from parietal_lobe.scene import (
    ObjectRelation,
    PerceivedObject,
    Scene,
    SpatialRelationKind,
    perceive,
)
from parietal_lobe.transforms import Example, Transform, propose

from .solver import SpatialHypothesis, SpatialReasoner, SpatialSolution

__all__ = [
    "Example",
    "Grid",
    "ObjectRelation",
    "PerceivedObject",
    "Scene",
    "SpatialHypothesis",
    "SpatialReasoner",
    "SpatialRelationKind",
    "SpatialSolution",
    "Transform",
    "ground_scene",
    "perceive",
    "propose",
    "scene_facts",
    "scene_relations_for",
]

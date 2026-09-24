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
    - ``SpatialPractice`` — its gated puzzle curriculum — drives the
      reasoner directly.
    - The cognition engine holds a ``SpatialReasoner`` instance so
      spatial problems raised in conversation can be perceived and
      reasoned about, not just talked past.
"""

from __future__ import annotations

from .agent import ActionStats, EpisodeResult, SpatialAgent, frame_diff, frame_to_grid
from .grid import Grid
from .grounding import ground_scene, scene_facts, scene_relations_for
from .practice import PracticeAttempt, SpatialPractice
from .scene import (
    ObjectRelation,
    PerceivedObject,
    Scene,
    SpatialRelationKind,
    perceive,
)
from .solver import SpatialHypothesis, SpatialReasoner, SpatialSolution
from .transforms import Example, Transform, propose

__all__ = [
    "ActionStats",
    "EpisodeResult",
    "Example",
    "Grid",
    "ObjectRelation",
    "PerceivedObject",
    "PracticeAttempt",
    "Scene",
    "SpatialAgent",
    "SpatialHypothesis",
    "SpatialPractice",
    "SpatialReasoner",
    "SpatialRelationKind",
    "SpatialSolution",
    "Transform",
    "frame_diff",
    "frame_to_grid",
    "ground_scene",
    "perceive",
    "propose",
    "scene_facts",
    "scene_relations_for",
]

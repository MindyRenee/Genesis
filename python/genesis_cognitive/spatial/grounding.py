"""Grounding — injecting perceived scenes into the concept network.

Perception produces a ``Scene``: objects with properties and typed
spatial relations. Grounding writes that structure into the concept
network so the *semantic* machinery — spreading activation, the
reasoning engine, analogy, the language engine — can operate on it.
This is the bridge between the perceptual-symbolic layer and the
semantic layer: a red square above a blue circle becomes real edges
(``scene:obj:0 --spatial_relation(above)--> scene:obj:1``), not an
opaque bitmap.

Concept names are namespaced per scene (``<ns>:obj:<i>``) so objects
from different scenes don't collide, and so a scene's contribution
can be recognized later (all names share the ``<ns>:`` prefix).

The grounding layer emits *structure*, not sentences. What Genesis
says about a scene is composed by her language engine from these
edges — nothing here writes words for her.
"""

from __future__ import annotations

from typing import Any

from ..concepts.types import RelationType
from .scene import ObjectRelation, Scene

# Edge has no metadata field, so the specific relation kind is encoded
# in the target concept name: the edge is
#   obj:0 --SPATIAL_RELATION--> scene:rel:above:obj:1
# which keeps each edge a proper directed triple while preserving
# which spatial relation holds.


def _object_concept_name(ns: str, index: int) -> str:
    return f"{ns}:obj:{index}"


def ground_scene(
    network: Any,
    scene: Scene,
    ns: str = "scene",
) -> list[str]:
    """Write a scene's objects and relations into the concept network.

    Args:
        network: a ``ConceptNetwork``.
        scene: the perceived scene to ground.
        ns: namespace prefix for this scene's concepts.

    Returns:
        The list of concept names created, for callers that want to
        activate or reference them afterward.
    """
    created: list[str] = []
    for obj in scene.objects:
        name = _object_concept_name(ns, obj.index)
        top, left, bottom, right = obj.bbox
        network.add_concept(
            name,
            origin="perception",
            properties={
                "kind": "spatial_object",
                "color": obj.color,
                "size": obj.size,
                "shape_signature": obj.shape_signature,
                "bbox": (top, left, bottom, right),
                "scene": ns,
            },
        )
        created.append(name)

    for rel in scene.relations:
        subject = _object_concept_name(ns, rel.subject)
        # Target concept encodes both relation kind and object index so
        # the edge remains a directed triple: obj:0 → "above(obj:1)".
        target = f"{ns}:rel:{rel.kind.value}:{rel.object}"
        network.add_concept(
            target,
            origin="perception",
            properties={
                "kind": "spatial_relation",
                "relation": rel.kind.value,
                "object": _object_concept_name(ns, rel.object),
                "scene": ns,
            },
        )
        network.add_edge(
            subject,
            target,
            RelationType.SPATIAL_RELATION,
            weight=0.8,
            origin="perceived",
        )
        created.append(target)

    return created


def scene_facts(scene: Scene, ns: str = "scene") -> list[tuple[str, str, str]]:
    """The scene as symbolic triples: (subject, relation, object).

    These are the raw material for the language engine or the reasoning
    engine — structured facts, not composed utterances.
    """
    facts: list[tuple[str, str, str]] = []
    for obj in scene.objects:
        name = _object_concept_name(ns, obj.index)
        facts.append((name, "has_color", str(obj.color)))
        facts.append((name, "has_size", str(obj.size)))
    for rel in scene.relations:
        facts.append(
            (
                _object_concept_name(ns, rel.subject),
                rel.kind.value,
                _object_concept_name(ns, rel.object),
            )
        )
    return facts


def scene_relations_for(
    scene: Scene, obj_index: int, ns: str = "scene"
) -> list[ObjectRelation]:
    """All relations where the given object is the subject."""
    return [r for r in scene.relations if r.subject == obj_index]

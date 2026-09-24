"""Concept types — enums and record types for the concept network."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CorticalLayer(Enum):
    """Laminar layer of a cortical column for a concept.

    - L4: input / sensory-associative (many incoming edges)
    - L2_3: association (well-connected both ways)
    - L5: output / projection (many outgoing edges)
    - L6: context / modulation (fewer connections, contextual role)
    """

    L4 = "L4"
    L2_3 = "L2_3"
    L5 = "L5"
    L6 = "L6"
class RelationType(Enum):
    """Types of relationships between concepts.

    These are the fundamental semantic relations — the same ones
    identified by cognitive science (WordNet, ConceptNet, etc.).
    """

    IS_A = "is_a"  # dog is_a animal
    PART_OF = "part_of"  # wheel is_part_of car
    CAUSES = "causes"  # fire causes smoke
    EMERGES_FROM = "emerges_from"  # cognition emerges_from neural_activity
    SIMILAR_TO = "similar_to"  # joy similar_to happiness
    OPPOSITE_OF = "opposite_of"  # hot opposite_of cold
    DEPENDS_ON = "depends_on"  # thinking depends_on memory
    ENABLES = "enables"  # memory enables learning
    INSTANCE_OF = "instance_of"  # genesis instance_of ai
    RELATED_TO = "related_to"  # generic association
    HAS_PROPERTY = "has_property"  # dog has_property furry
    CONTRADICTS = "contradicts"  # free will contradicts determinism
    LEADS_TO = "leads_to"  # practice leads_to mastery
    CREATES = "creates"  # alice creates genesis
    HARMS = "harms"  # stress harms memory
    BRIDGES = "bridges"  # inter-column bridge (same concept, different origin)
    # ── Tier 3: additional semantic relations ──────────────────
    TEMPORAL_ORDER = "temporal_order"  # X comes before Y (temporal sequence)
    SPATIAL_RELATION = "spatial_relation"  # X is inside/above/below/beside Y
    AGENT_PATIENT = "agent_patient"  # X acts on Y (thematic role)
    GOAL_DIRECTED = "goal_directed"  # X aims to achieve Y (teleological)
    # ── Tier 4: inhibitory relations ─────────────────────────────
    PREVENTS = "prevents"  # X prevents Y (inhibitory, the inverse of ENABLES)
    # ── Word-to-category relations ───────────────────────────────
    # A word EXPRESSES a structural category (emotion, cause, mode,
    # plasticity). This connects learned vocabulary to category hub
    # concepts via typed edges, so word retrieval uses graph traversal
    # (get_neighbors from the hub) instead of linear property scans.
    EXPRESSES = "expresses"  # "overwhelmed" expresses _cat:emotion:stressed
    # ── Code-structure relations (separate from semantic) ────────
    # These decouple code graph edges from conceptual relations so
    # that spreading activation and reasoning don't confuse "function
    # A calls function B" with "curiosity leads_to learning".
    CALLS = "calls"  # function A calls function B (code call graph)
    DEFINES = "defines"  # module/class A defines member B (code structure)
def _is_typed_relation(rel: RelationType) -> bool:
    """True if this relation type carries semantic meaning (not just co-occurrence)."""
    return rel not in _UNTYPED_RELATIONS
class ConceptCategory(Enum):
    """Category of a concept, modeling the biological distinction between
    living and non-living things.

    In the brain, different categories of objects are processed in
    different cortical regions. Living things (animals, plants, people)
    are processed primarily in the temporal subsystem, which is rich in
    sensory (visual, auditory) associations. Tools and manipulable
    objects are processed in frontoparietal regions, which encode
    motion and action affordances. Abstract concepts recruit a more
    distributed network including prefrontal cortex.

    This category field lets the concept network model these
    category-specific organization and activation dynamics.

    References:
        - Warrington & McCarthy (1983, 1987): category-specific deficits
        - Martin & Chao (2001): category-specific cortical organization
        - Damasio et al. (1996): neural systems behind concept retrieval
    """

    LIVING = "living"  # animate things (animals, plants, people)
    NON_LIVING = "non_living"  # inanimate objects (tools, machines, materials)
    ABSTRACT = "abstract"  # concepts without physical form (ideas, emotions)
    EVENT = "event"  # actions, processes, occurrences
    UNKNOWN = "unknown"  # not yet categorized
class ConceptModality(Enum):
    """Sensorimotor modality of a concept.

    In the brain, concepts are grounded in sensorimotor systems.
    Visual concepts (e.g., "red", "tree") activate visual cortex;
    motor concepts (e.g., "run", "grasp") activate motor cortex;
    auditory concepts (e.g., "music", "thunder") activate auditory
    cortex; verbal concepts (e.g., "truth", "justice") rely on
    language areas. Abstract concepts recruit a distributed network
    with no single sensory-motor anchor.

    This modality field lets the concept network model modality-specific
    activation dynamics: visual concepts activate faster, motor concepts
    have shorter decay, and within-modality priming boosts related
    concepts in the same modality.

    References:
        - Barsalou (1999): Perceptual symbol systems
        - Gallese & Lakoff (2005): The brain's concepts
        - Binder & Desai (2011): The neurobiology of semantic memory
    """

    VISUAL = "visual"  # perceived through sight (colors, shapes, objects)
    MOTOR = "motor"  # involves bodily action (verbs of motion, tools)
    AUDITORY = "auditory"  # perceived through hearing (sounds, music)
    VERBAL = "verbal"  # language-based (words, propositions, rules)
    ABSTRACT = "abstract"  # no single sensory-motor anchor (ideas, emotions)
    MIXED = "mixed"  # spans multiple modalities
    UNKNOWN = "unknown"  # not yet classified
@dataclass
class Concept:
    """A concept in Genesis's mind.

    A concept is not a word — it's an idea. Multiple words can map
    to the same concept (aliases). A concept has relationships to
    other concepts, an activation level, and a confidence.

    A concept can belong to multiple cortical columns (e.g., "cognition"
    was first added from the dictionary but also referenced from identity).
    The `columns` set tracks all columns that have contributed to this
    concept, enabling column-aware activation and inhibition.
    """

    id: str  # canonical name, e.g. "cognition"
    aliases: set[str] = field(default_factory=set)  # other words for this
    activation: float = 0.0  # 0..1, decays over time, boosted on use
    confidence: float = 0.5  # 0..1, how well-formed this concept is
    origin: str = "conversation"  # where it came from (primary column)
    columns: set[str] = field(default_factory=set)  # all columns that contributed
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    # Spaced repetition fields
    review_count: int = 0  # how many times this concept has been reviewed
    last_reviewed: int = 0  # timestamp of last review (ms), 0 = never
    properties: dict[str, Any] = field(default_factory=dict)  # arbitrary metadata
    # Category-specific organization (living/non-living/abstract/event).
    # Models the biological distinction where different categories are
    # processed in different cortical regions (temporal subsystem for living
    # things, frontoparietal for tools, distributed for abstract).
    category: ConceptCategory = ConceptCategory.UNKNOWN
    is_animacy_detected: bool = False  # True if category was set by detection
    # Sensorimotor modality (visual/motor/auditory/verbal/abstract/mixed).
    # Models embodied cognition — concepts are grounded in the same
    # sensorimotor systems that perceive and act on the world. Visual
    # concepts activate faster, motor concepts have shorter decay.
    modality: ConceptModality = ConceptModality.UNKNOWN
    # Semantic hub flag — set by detect_semantic_hubs(). Hubs are
    # high-betweenness convergence zones (Damasio) that integrate
    # information across the network. They get stronger activation
    # spread (1.3x) and slower decay (0.7x).
    is_semantic_hub: bool = False

    def describe(self) -> str:
        """Brief description for introspection."""
        parts = [self.id]
        if self.aliases:
            parts.append(f"(aka: {', '.join(sorted(self.aliases))})")
        parts.append(f"confidence={self.confidence:.2f}")
        parts.append(f"activation={self.activation:.2f}")
        return " ".join(parts)
@dataclass
class Edge:
    """A typed relationship between two concepts."""

    source: str  # concept id
    target: str  # concept id
    relation: RelationType
    weight: float = 0.5  # 0..1, strength of the relationship
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    origin: str = "inferred"  # "stated", "inferred", "observed"
@dataclass
class Provenance:
    """A retrieved item with its origin metadata.

    Wraps a text string (utterance, template, verb phrase, etc.) with
    the concept origin that produced it. ``is_seed`` is True when the
    origin is in ``_SEED_ORIGINS`` — meaning the content is a
    developer-authored building block, not something Genesis learned
    or composed itself.

    Generators use this to prefer learned/composed content over seeds
    and to disclose when output is seed-only.
    """

    text: str
    origin: str
    is_seed: bool

    @classmethod
    def from_concept(cls, text: str, concept: Concept) -> Provenance:
        """Build a Provenance record from a concept's origin."""
        return cls(text=text, origin=concept.origin, is_seed=is_seed_origin(concept.origin))
def is_seed_origin(origin: str) -> bool:
    """Return True if this origin marks developer-authored seed content.

    This is the single source of truth for distinguishing seeds from
    content Genesis composed or learned itself. Generators use it to
    prefer learned content and to disclose when output is seed-only.
    """
    return origin in _SEED_ORIGINS


_UNTYPED_RELATIONS = frozenset({
    RelationType.RELATED_TO,
    RelationType.BRIDGES,
})

_SEED_ORIGINS: frozenset[str] = frozenset({
    # Concept origins
    "seeded", "structural",
    # Edge origins
    "vocabulary_seed",
    "relation_verb_seed", "relation_phrase_seed",
    "narrative_template_seed",
})

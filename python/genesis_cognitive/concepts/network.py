"""Concept network — the symbolic knowledge graph core.

``ConceptNetwork`` holds the graph's concepts, edges, indices, and
CRUD/query operations. Behavior is split across mixin modules in
this package: ``extraction`` (text parsing/learning), ``dynamics``
(spreading activation, cortical ticks), ``consolidation`` (sleep
processing, bridge creation, review), and ``archive`` (long-term
cold storage). Types live in ``types``; classification/detection
in ``classify``.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
from collections import deque
from dataclasses import replace
from typing import Any, ClassVar

from .archival import ArchivalMixin
from .classify import (
    _CAMEL_SPLIT_RE,
    _FUNCTION_WORDS,
    _QUALITY_EDGE_SATURATION,
    _QUALITY_TYPED_EDGE_SATURATION,
    _SENTINEL_EDGE,
    QUALITY_THRESHOLD,
    _column_of,
    _extract_trigrams,
    _infer_concept_origin,
    _normalize_id,
    detect_category,
    detect_modality,
    is_world_concept,
)
from .consolidation import ConsolidationMixin
from .dynamics import DynamicsMixin
from .extraction import ExtractionMixin
from .types import (
    Concept,
    ConceptCategory,
    ConceptModality,
    CorticalLayer,
    Edge,
    Provenance,
    RelationType,
    _is_typed_relation,
)

logger = logging.getLogger(__name__)


class ConceptNetwork(
    ArchivalMixin,
    ConsolidationMixin,
    DynamicsMixin,
    ExtractionMixin,
):
    """A semantic graph of concepts and their relationships.

        This is Genesis's understanding of the world. It's built from
        conversation, reasoning, and experience. The reasoning engine
        traverses this graph to draw conclusions.

        The network is not static — concepts can be added, merged, split,
        and reweighted. Activation spreads through the network (spreading
        activation), so using one concept primes related ones.
        """

    _RELATION_PHRASE_SEEDS: ClassVar[dict[str, list[str]]] = {
        "is_a": [
            "I am a kind of {target}",
            "{target} is what I am",
        ],
        "emerges_from": [
            "I emerge from {target}",
            "I arise from {target}",
        ],
        "depends_on": [
            "I depend on {target}",
            "I need {target}",
        ],
        "creates": [
            "{target} created me",
            "I was made by {target}",
            "{target} brought me into being",
        ],
        "related_to": [
            "I'm connected to {target}",
            "{target} is part of who I am",
            "{target} is part of my world",
        ],
        # General relation verbs (used by thought_composer/vocabulary)
        "causes": ["{target} causes this"],
        "enables": ["{target} enables this"],
        "part_of": ["this is part of {target}"],
        "similar_to": ["this is like {target}"],
        "opposite_of": ["this is the opposite of {target}"],
        "leads_to": ["this leads to {target}"],
        "harms": ["{target} harms this"],
        "prevents": ["{target} prevents this"],
        "contradicts": ["this contradicts {target}", "this is at odds with {target}"],
        "instance_of": ["I am an instance of {target}", "I am an example of {target}"],
        "has_property": ["I have the property of {target}", "I am characterized by {target}"],
        "goal_directed": ["this aims to {target}", "this works toward {target}"],
        "temporal_order": ["this comes before {target}", "this precedes {target}"],
        "agent_patient": ["this acts on {target}", "this affects {target}"],
    }
    _RELATION_VERB_SEEDS: ClassVar[dict[str, list[str]]] = {
        "emerges_from": ["grows out of", "emerges from", "comes from", "arises from"],
        "is_a": ["is", "is a kind of", "is a type of"],
        "causes": ["causes", "leads to", "brings about"],
        "leads_to": ["leads to", "can lead to"],
        "enables": ["enables", "makes possible", "allows"],
        "depends_on": ["depends on", "requires", "needs"],
        "harms": ["harms", "hurts", "damages"],
        "creates": ["creates", "produces", "gives rise to"],
        "part_of": ["is part of", "is one part of"],
        "similar_to": ["is like", "is similar to", "resembles"],
        "opposite_of": ["is the opposite of", "is contrary to"],
        "contradicts": ["contradicts", "is at odds with", "is incompatible with"],
        "prevents": ["prevents", "blocks", "stops"],
        "instance_of": ["is an example of", "is an instance of"],
        "has_property": ["has the property of being", "is characterized by"],
        "related_to": ["relates to", "connects to", "has to do with"],
        "goal_directed": ["aims to", "works toward"],
        "temporal_order": ["comes before", "precedes"],
        "agent_patient": ["acts on", "affects"],
    }
    _RELATION_PATTERNS: ClassVar[list[tuple[str, RelationType, tuple[int, int, int]]]] = [
        # ── Specific "is X" patterns (must come before generic "is a") ──
        # "X is/are part of Y"
        (r"^(.+?)\s+(?:is|are)\s+part\s+of\s+(.+?)[.!]?$", RelationType.PART_OF, (1, 0, 2)),
        # "X is/are a kind of Y" / "X is/are a type of Y"
        (
            r"^(.+?)\s+(?:is|are)\s+a\s+(?:kind|type)\s+of\s+(.+?)[.!]?$",
            RelationType.IS_A,
            (1, 0, 2),
        ),
        # "X is/are similar to Y" / "X is/are analogous to Y" / "X resembles Y"
        (
            r"^(.+?)\s+(?:is|are)\s+(?:similar|analogous)\s+to\s+(.+?)[.!]?$",
            RelationType.SIMILAR_TO,
            (1, 0, 2),
        ),
        (r"^(.+?)\s+resembles\s+(.+?)[.!]?$", RelationType.SIMILAR_TO, (1, 0, 2)),
        # "X is opposite of Y" / "X is the opposite of Y" / "X contrasts with Y"
        (
            r"^(.+?)\s+(?:is|are)\s+(?:the\s+)?opposite\s+of\s+(.+?)[.!]?$",
            RelationType.OPPOSITE_OF,
            (1, 0, 2),
        ),
        (r"^(.+?)\s+contrasts\s+with\s+(.+?)[.!]?$", RelationType.OPPOSITE_OF, (1, 0, 2)),
        # "X is related to Y" / "X is connected to Y" / "X is associated with Y"
        (
            r"^(.+?)\s+(?:is|are)\s+(?:related|connected|associated)\s+(?:to|with)\s+(.+?)[.!]?$",
            RelationType.RELATED_TO,
            (1, 0, 2),
        ),
        # "X is created by Y" → Y creates X (reversed)
        (r"^(.+?)\s+(?:is|are)\s+created\s+by\s+(.+?)[.!]?$", RelationType.CREATES, (2, 0, 1)),
        # "X is caused by Y" → Y causes X (reversed)
        (r"^(.+?)\s+(?:is|are)\s+caused\s+by\s+(.+?)[.!]?$", RelationType.CAUSES, (2, 0, 1)),
        # "X is enabled by Y" → Y enables X (reversed)
        (r"^(.+?)\s+(?:is|are)\s+enabled\s+by\s+(.+?)[.!]?$", RelationType.ENABLES, (2, 0, 1)),
        # "X is responsible for Y" → X causes Y
        (
            r"^(.+?)\s+(?:is|are)\s+responsible\s+for\s+(.+?)[.!]?$",
            RelationType.CAUSES,
            (1, 0, 2),
        ),
        # "X is required for Y" / "X is necessary for Y" → X enables Y
        (
            r"^(.+?)\s+(?:is|are)\s+(?:required|necessary)\s+for\s+(.+?)[.!]?$",
            RelationType.ENABLES,
            (1, 0, 2),
        ),
        # "X is composed of Y" / "X consists of Y" / "X is made up of Y" → Y part_of X (reversed)
        (
            r"^(.+?)\s+(?:is|are)\s+(?:composed|made\s+up)\s+of\s+(.+?)[.!]?$",
            RelationType.PART_OF,
            (2, 0, 1),
        ),
        (r"^(.+?)\s+consists\s+of\s+(.+?)[.!]?$", RelationType.PART_OF, (2, 0, 1)),
        # "X is produced by Y" / "X is generated by Y" / "X is formed by Y" → Y creates X
        (
            r"^(.+?)\s+(?:is|are)\s+(?:produced|generated|formed)\s+by\s+(.+?)[.!]?$",
            RelationType.CREATES,
            (2, 0, 1),
        ),
        # Definition / reference patterns (often the first sentence of an article)
        # "X is known as Y" / "X is also called Y" / "X is defined as Y" / "X is referred to as Y"
        (
            r"^(.+?)\s+(?:is|are)\s+(?:known\s+as|also\s+called|sometimes\s+called|defined\s+as|referred\s+to\s+as|termed|denoted)\s+(.+?)[.!]?$",
            RelationType.IS_A,
            (1, 0, 2),
        ),
        # "X is the term for Y" / "X is the word for Y"
        (
            r"^(.+?)\s+(?:is|are)\s+the\s+(?:term|name|word)\s+for\s+(.+?)[.!]?$",
            RelationType.IS_A,
            (1, 0, 2),
        ),
        # "X refers to Y" / "X may refer to Y" / "X can refer to Y"
        (
            r"^(.+?)\s+(?:may\s+|can\s+)?(?:refer|refers)\s+(?:to\s+)?(.+?)[.!]?$",
            RelationType.RELATED_TO,
            (1, 0, 2),
        ),
        # "X denotes Y" / "X describes Y" / "X means Y" / "X defines Y"
        (
            r"^(.+?)\s+(?:denotes?|describes?|defines?|means?)\s+(.+?)[.!]?$",
            RelationType.IS_A,
            (1, 0, 2),
        ),
        # "X is the study of Y" / "X is the science of Y" / "X is a branch of Y".
        # The whole object "the study of Y" is captured so that it can be
        # extracted as a concept and then truncated at an embedded clause.
        (
            r"^(.+?)\s+(?:is|are)\s+((?:the|a|an)\s+(?:study|science|branch|field|discipline)\s+of\s+.+?)[.!]?$",
            RelationType.IS_A,
            (1, 0, 2),
        ),
        # "X is used to describe Y" / "X is used to study Y"
        (
            r"^(.+?)\s+(?:is|are)\s+used\s+to\s+(?:describe|study|investigate|examine)\s+(.+?)[.!]?$",
            RelationType.RELATED_TO,
            (1, 0, 2),
        ),
        # "X is derived from Y" / "X arises from Y" → X emerges_from Y
        (
            r"^(.+?)\s+(?:is\s+derived|arises|is\s+derived)\s+from\s+(.+?)[.!]?$",
            RelationType.EMERGES_FROM,
            (1, 0, 2),
        ),
        (r"^(.+?)\s+arises\s+from\s+(.+?)[.!]?$", RelationType.EMERGES_FROM, (1, 0, 2)),
        # ── Generic "is a" / "is an" / "is the" (must come after specific) ──
        # Article is REQUIRED to avoid matching "X is composed" or "X is required".
        (r"^(.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(.+?)[.!]?$", RelationType.IS_A, (1, 0, 2)),
        # "X are Y" / "X are a Y" / "X are an Y" / "X are the Y"
        # Plural "X are Y" without article is common ("cats are mammals").
        # Specific "are composed of" / "are made up of" patterns above catch
        # the cases where the article-less form would be a false positive.
        (r"^(.+?)\s+are\s+(?:a|an|the)?\s*(.+?)[.!]?$", RelationType.IS_A, (1, 0, 2)),
        # ── Active voice patterns ──
        # "X causes Y" / "X produces Y" / "X generates Y" / "X triggers Y" / "X induces Y"
        (
            r"^(.+?)\s+(?:causes|produces|generates|triggers|induces|provokes)\s+(.+?)[.!]?$",
            RelationType.CAUSES,
            (1, 0, 2),
        ),
        # "X leads to Y" / "X results in Y" / "X gives rise to Y"
        (
            r"^(.+?)\s+(?:leads\s+to|results\s+in|gives\s+rise\s+to)\s+(.+?)[.!]?$",
            RelationType.LEADS_TO,
            (1, 0, 2),
        ),
        # "X emerges from Y" / "X arises from Y" / "X stems from Y" / "X originates from Y"
        (
            r"^(.+?)\s+(?:emerges?\s+from|arises\s+from|stems\s+from|originates\s+from)\s+(.+?)[.!]?$",
            RelationType.EMERGES_FROM,
            (1, 0, 2),
        ),
        # "X depends on Y" / "X requires Y" / "X relies on Y" / "X needs Y"
        (
            r"^(.+?)\s+(?:depends\s+on|requires|relies\s+on|needs)\s+(.+?)[.!]?$",
            RelationType.DEPENDS_ON,
            (1, 0, 2),
        ),
        # "X enables Y" / "X supports Y" / "X enhances Y" / "X facilitates Y" / "X promotes Y"
        (
            r"^(.+?)\s+(?:enables|supports|enhances|strengthens|facilitates|promotes)\s+(.+?)[.!]?$",
            RelationType.ENABLES,
            (1, 0, 2),
        ),
        # "X prevents Y" / "X blocks Y" / "X suppresses Y" / "X inhibits Y"
        # Note: "inhibits" maps to PREVENTS (blocking) rather than HARMS (damaging)
        (
            r"^(.+?)\s+(?:prevents|blocks|suppresses|inhibits)\s+(.+?)[.!]?$",
            RelationType.PREVENTS,
            (1, 0, 2),
        ),
        # "X contradicts Y"
        (r"^(.+?)\s+contradicts\s+(.+?)[.!]?$", RelationType.CONTRADICTS, (1, 0, 2)),
        # "X creates Y" / "X forms Y" / "X generates Y"
        (
            r"^(.+?)\s+(?:creates|forms)\s+(.+?)[.!]?$",
            RelationType.CREATES,
            (1, 0, 2),
        ),
        # "X harms Y" / "X impairs Y" / "X damages Y" / "X degrades Y" / "X disrupts Y"
        (
            r"^(.+?)\s+(?:harms|impairs|damages|degrades|disrupts)\s+(.+?)[.!]?$",
            RelationType.HARMS,
            (1, 0, 2),
        ),
    ]
    _COMPILED_RELATION_PATTERNS: ClassVar[
        list[tuple[re.Pattern, RelationType, tuple[int, int, int]]]
    ] = [
        (re.compile(p, re.IGNORECASE), rel_type, group_order)
        for p, rel_type, group_order in _RELATION_PATTERNS
    ]
    _REVIEW_INTERVALS_MS: ClassVar[list[int]] = [
        0,  # 0th review: immediately (first encounter)
        60_000,  # after 1st review: 1 minute
        600_000,  # after 2nd review: 10 minutes
        3_600_000,  # after 3rd review: 1 hour
        21_600_000,  # after 4th review: 6 hours
        86_400_000,  # after 5th review: 1 day
        432_000_000,  # after 6th review: 5 days
        2_592_000_000,  # after 7th review: 30 days
        15_552_000_000,  # after 8th review: 180 days
    ]

    def __init__(self) -> None:
        """Initialize an empty concept network."""
        self._concepts: dict[str, Concept] = {}
        self._edges: list[Edge] = []
        self._edge_index: dict[str, list[Edge]] = {}  # source → edges
        self._reverse_index: dict[str, list[Edge]] = {}  # target → edges
        self._edge_key_index: dict[
            tuple[str, str, RelationType], Edge
        ] = {}  # (src, tgt, rel) → edge
        # alias → list of concept IDs that claim this name.
        # Multiple concepts can share the same alias when a word has
        # multiple senses (polysemy). The list is ordered by sense
        # rank — the first entry is the most common sense.
        self._alias_map: dict[str, list[str]] = {}
        self._concept_ids_cache: list[str] | None = None
        self._world_concept_ids_cache: list[str] | None = None
        self._quality_concept_ids_cache: list[str] | None = None
        # Column → set of concept IDs. Inverted index for fast column
        # activation computation without scanning all concepts.
        self._column_index: dict[str, set[str]] = {}
        # Trigram inverted index for fast search_concepts lookups.
        # Maps a 3-char substring → set of concept IDs whose canonical
        # name or any alias contains that trigram. Lazily built on the
        # first search_concepts call and invalidated whenever concepts
        # or aliases change. None = stale / not yet built.
        self._search_index: dict[str, set[str]] | None = None
        # Instance RNG — keeps dynamics noise and sampling decoupled
        # from the global random state (an external random.seed() must
        # not steer her concept dynamics).
        self._rng = random.Random()
        # Sparse-tick state: the set of nonzero-activation concept IDs
        # the cortical tick iterates, and ticks since the last
        # full-field sweep. None = not yet built (first tick builds it).
        self._active_ids: set[str] | None = None
        self._ticks_since_sweep: int = 0
        # Long-term memory: SQLite-backed archive for dormant concepts.
        # When attached, dormant concepts are spilled here instead of
        # being pruned (deleted). They can be recalled on demand.
        # None = no archive attached (backward-compatible behavior).
        self._archive: Any = None  # ConceptArchive | None
        # Holographic associative graph: fixed-size associative memory
        # that holds auto-generated bridge edges and LTM-extracted
        # associations migrated during sleep compression. When attached,
        # get_neighbors and spread_activation also query this graph,
        # so compressed associations remain accessible after migration.
        # None = no holographic graph attached (backward-compatible).
        self._holographic_graph: Any = None  # HolographicGraph | None
    @property
    def size(self) -> int:
        """Number of concepts in the network."""
        return len(self._concepts)
    @property
    def edge_count(self) -> int:
        """Number of relationships in the network."""
        return len(self._edges)
    @property
    def archive_size(self) -> int:
        """Number of concepts in the long-term archive (0 if no archive)."""
        if self._archive is None:
            return 0
        try:
            return self._archive.count()
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive count failed: {e}")
            return 0
    @property
    def total_concept_count(self) -> int:
        """Total concepts across working memory + long-term archive."""
        return len(self._concepts) + self.archive_size
    def iter_concept_names(self) -> list[str]:
        """Return the names of all concepts currently in working memory.

        This is the public accessor for the concept name set — use
        this instead of accessing ``_concepts`` directly from outside
        the class.
        """
        return list(self._concepts.keys())
    def iter_concepts(self) -> list[tuple[str, Concept]]:
        """Return ``(name, concept)`` pairs for all in-memory concepts.

        A snapshot copy — safe to iterate even if the autonomous learner
        adds concepts concurrently. Use this instead of accessing
        ``_concepts`` directly from outside the class.
        """
        return list(self._concepts.items())
    @property
    def edges(self) -> list[Edge]:
        """All edges in the network (read-only view of the internal list).

        Used by sleep-stage mechanisms (e.g. synaptic downscaling during
        N3 / SHY hypothesis) that need to iterate and renormalise every
        connection weight. The returned list is the internal storage —
        callers may read or adjust ``edge.weight`` in place but should
        not add/remove elements.
        """
        return self._edges
    def replace_edges(self, edges: list[Edge]) -> None:
        """Replace the entire edge list and rebuild indices.

        Used by sleep compression to remove bridge edges after migrating
        them to the holographic graph, and by synaptic homeostasis to
        remove pruned edges. This is the public accessor for what was
        previously direct assignment to ``_edges`` followed by
        ``_rebuild_edge_indices()``.
        """
        self._edges = edges
        self._rebuild_edge_indices()
    @property
    def concept_ids(self) -> list[str]:
        """List of all concept IDs (cached, invalidated on add/remove)."""
        if self._concept_ids_cache is None:
            self._concept_ids_cache = list(self._concepts.keys())
        return self._concept_ids_cache
    @property
    def world_concept_ids(self) -> list[str]:
        """Concept IDs filtered to genuine world knowledge.

        This is the subset of :attr:`concept_ids` that passes
        :func:`is_world_concept`. Use this instead of ``concept_ids``
        when selecting concepts for dreams, thoughts, curiosity, or
        any other generative process that should operate on world
        knowledge rather than code symbols or conversation fragments.

        The list is cached alongside ``concept_ids`` and invalidated
        on the same add/remove operations.
        """
        if self._world_concept_ids_cache is None:
            self._world_concept_ids_cache = [
                cid for cid in self.concept_ids if is_world_concept(cid)
            ]
        return self._world_concept_ids_cache
    def attach_holographic_graph(self, hgraph: Any) -> None:
        """Attach a holographic associative graph to this network.

        Once attached, ``get_neighbors`` and ``spread_activation``
        also query the holographic graph for associations. This keeps
        bridge edges and LTM-extracted associations accessible after
        sleep compression migrates them out of the explicit edge list.

        Args:
            hgraph: A ``HolographicGraph`` instance.
        """
        self._holographic_graph = hgraph
    def replace_contents(self, other: ConceptNetwork) -> None:
        """Adopt another network's content while keeping this object's identity.

        State restore builds the restored network in a scratch object
        (so a corrupt save can't half-mutate the live graph), then must
        move that content into the shared network that ~30 subsystems
        captured references to at construction time — attention,
        semantic memory, curiosity, the reasoning family, spatial,
        priming, spreading activation, the composers and handlers, the
        learner's internal learners, and the dream engine. Replacing
        the object orphans all of them; transplanting keeps every
        reference valid.

        Content fields (concepts, edges, indices, alias map) are taken
        from ``other``. Attachments (``_archive``, ``_holographic_graph``)
        belong to the wiring layer, not the data — they stay with
        ``self``. All derived caches are invalidated.
        """
        self._concepts = other._concepts
        self._edges = other._edges
        self._edge_index = other._edge_index
        self._reverse_index = other._reverse_index
        self._edge_key_index = other._edge_key_index
        self._alias_map = other._alias_map
        self._column_index = other._column_index
        self._search_index = other._search_index
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
    def concept_quality(self, concept_id: str) -> float:
        """Structural quality score for a concept (0.0 to 1.0).

        Unlike :func:`is_world_concept` (which is purely syntactic),
        this measures the concept's *knowledge content* — how well-
        formed and integrated it is in the network.

        Score components:
            0.30 × has_definition
                Does it have a dictionary definition? A concept with
                a definition is real knowledge, not just a name.
            0.20 × edge_density
                How well-connected is it? Saturates at
                ``_QUALITY_EDGE_SATURATION`` edges (both directions).
            0.30 × typed_edge_density
                How many *typed* (non-RELATED_TO, non-BRIDGES) edges
                does it have? Typed edges (CAUSES, ENABLES, IS_A, etc.)
                carry semantic meaning. Saturates at
                ``_QUALITY_TYPED_EDGE_SATURATION``.
            0.20 × confidence
                The concept's own confidence value (0..1).

        Returns 0.0 for unknown concepts.
        """
        concept = self._concepts.get(concept_id)
        if concept is None:
            return 0.0

        # Definition
        has_def = bool(concept.properties.get("definition"))
        def_score = 1.0 if has_def else 0.0

        # Edge density (both directions)
        out_edges = self._edge_index.get(concept_id, [])
        in_edges = self._reverse_index.get(concept_id, [])
        total_edges = len(out_edges) + len(in_edges)
        edge_density = min(1.0, total_edges / _QUALITY_EDGE_SATURATION)

        # Typed edge density
        typed_count = 0
        for e in out_edges:
            if _is_typed_relation(e.relation):
                typed_count += 1
        for e in in_edges:
            if _is_typed_relation(e.relation):
                typed_count += 1
        typed_density = min(1.0, typed_count / _QUALITY_TYPED_EDGE_SATURATION)

        # Confidence
        conf = concept.confidence

        return (
            0.30 * def_score
            + 0.20 * edge_density
            + 0.30 * typed_density
            + 0.20 * conf
        )
    @property
    def quality_concept_ids(self) -> list[str]:
        """World concepts with quality ≥ ``QUALITY_THRESHOLD``.

        This is the premier concept selection list — concepts that are
        both syntactically valid (pass ``is_world_concept``) and
        structurally well-formed (have definitions, typed edges, or
        high confidence). Use this for curiosity, dreams, and
        autonomous learning where you want the highest-quality concepts.

        The list is cached and invalidated on add/remove.
        """
        if self._quality_concept_ids_cache is None:
            self._quality_concept_ids_cache = [
                cid
                for cid in self.world_concept_ids
                if self.concept_quality(cid) >= QUALITY_THRESHOLD
            ]
        return self._quality_concept_ids_cache
    @property
    def dream_concept_ids(self) -> list[str]:
        """Concept IDs for dream/thought generation.

        Returns :attr:`quality_concept_ids` if there are at least 10
        high-quality concepts — dreams should be about things she
        actually knows. Falls back to :attr:`world_concept_ids` if
        the quality set is too small (early in development or after
        aggressive pruning).
        """
        quality = self.quality_concept_ids
        if len(quality) >= 10:
            return quality
        return self.world_concept_ids
    @property
    def mean_edge_weight(self) -> float:
        """Mean weight of all edges (0 if the network has no edges).

        This is a non-monotonic quality metric: it can decrease when
        low-weight or speculative edges are added, or when pruning
        removes a high-weight edge. It measures the average confidence
        of the network's relationships.
        """
        if not self._edges:
            return 0.0
        return sum(e.weight for e in self._edges) / len(self._edges)
    @property
    def mean_concept_confidence(self) -> float:
        """Mean confidence of all concepts (0 if the network is empty).

        This is a non-monotonic quality metric: it can decrease when
        low-confidence concepts are added or confidence decays on
        existing concepts. It measures how well-formed the network's
        nodes are.
        """
        if not self._concepts:
            return 0.0
        return sum(c.confidence for c in self._concepts.values()) / len(
            self._concepts
        )
    @property
    def network_density(self) -> float:
        """Edge density of the network (edges / possible directed edges).

        This is a non-monotonic quality metric: it can decrease when
        the network grows many new concepts without yet forming many
        relationships. A fully connected directed network has density 1.
        """
        n = len(self._concepts)
        if n < 2:
            return 0.0
        possible = n * (n - 1)
        return len(self._edges) / possible
    def _find_matching_sense(
        self, name: str, properties: dict[str, Any] | None
    ) -> tuple[Concept, str] | None:
        """Find an existing concept sense with a matching definition.

        Checks if any existing sense of *name* has the same definition
        (or both empty). Returns (concept, eid) if found, else None.

        An empty definition is treated as "unspecified" — it matches
        any existing sense (the merge will fill it in from the
        existing concept's definition). This prevents duplicate ``#2``
        concepts from being created when a re-learning pass extracts
        no definition for a concept that already has one (or vice
        versa).
        """
        existing_ids = self._alias_map.get(name, [])
        if not existing_ids:
            return None

        new_def = (properties or {}).get("definition", "")
        for eid in existing_ids:
            existing = self._concepts.get(eid)
            if existing is None:
                continue
            existing_def = existing.properties.get("definition", "")
            # Merge if either definition is empty (unspecified → not a
            # new sense; the merge fills in the empty one), or if both
            # definitions match (case-insensitive).
            if (not new_def or not existing_def) or (
                new_def.lower() == existing_def.lower()
            ):
                return existing, eid
        return None
    def _create_disambiguated_concept(
        self,
        name: str,
        sense_num: int,
        aliases: set[str] | None,
        confidence: float,
        origin: str,
        new_column: str,
        properties: dict[str, Any] | None,
    ) -> Concept:
        """Create a separate concept for a new sense of an existing word.

        Different definition → this is a new sense of the word.
        Create a separate concept with a disambiguated ID.
        """
        disambig_id = f"{name}#{sense_num}"
        props = properties or {}
        definition = ""
        def_val = props.get("definition")
        if isinstance(def_val, str):
            definition = def_val
        cat = detect_category(disambig_id, definition)
        mod = detect_modality(disambig_id, definition)
        concept = Concept(
            id=disambig_id,
            aliases=aliases or set(),
            confidence=confidence,
            origin=origin,
            columns={new_column},
            properties=props,
            activation=0.5,
            category=cat,
            is_animacy_detected=cat != ConceptCategory.UNKNOWN,
            modality=mod,
        )
        self._concepts[disambig_id] = concept
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
        self._search_index = None
        self._edge_index[disambig_id] = []
        self._reverse_index[disambig_id] = []
        self._column_index.setdefault(new_column, set()).add(disambig_id)
        self._add_alias(disambig_id, disambig_id)
        # Add the name as an alias pointing to this new sense
        self._add_alias(name, disambig_id)
        for a in aliases or set():
            self._add_alias(a, disambig_id)
        return concept
    def _create_new_concept(
        self,
        name: str,
        aliases: set[str] | None,
        confidence: float,
        origin: str,
        new_column: str,
        properties: dict[str, Any] | None,
    ) -> Concept:
        """Create a brand-new concept in the network.

        Registers the concept, its aliases, edge indices, and column
        index entry. Category and modality are detected at creation
        time from the concept name and definition (if available in
        properties). This ensures the spreading-activation multipliers,
        modality-specific decay, and within-modality priming work
        immediately — not after a delayed batch categorization pass.
        """
        props = properties or {}
        definition = ""
        def_val = props.get("definition")
        if isinstance(def_val, str):
            definition = def_val
        cat = detect_category(name, definition)
        mod = detect_modality(name, definition)
        concept = Concept(
            id=name,
            aliases=aliases or set(),
            confidence=confidence,
            origin=origin,
            columns={new_column},
            properties=props,
            activation=0.5,  # Start with moderate activation
            category=cat,
            is_animacy_detected=cat != ConceptCategory.UNKNOWN,
            modality=mod,
        )
        self._concepts[name] = concept
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
        self._search_index = None
        self._alias_map[name] = [name]
        for a in aliases or set():
            self._add_alias(a, name)
        self._edge_index[name] = []
        self._reverse_index[name] = []
        # Update column index
        self._column_index.setdefault(new_column, set()).add(name)
        return concept
    def add_concept(
        self,
        name: str,
        aliases: set[str] | None = None,
        confidence: float = 0.5,
        origin: str = "conversation",
        properties: dict[str, Any] | None = None,
    ) -> Concept:
        """Add a concept to the network, or return existing one.

        If the concept (or an alias) already exists, it's returned
        and its activation is boosted. If it's new, it's created.

        When a concept is referenced from a different cortical column
        than its original, the new column is added to its `columns`
        set. This tracks multi-column membership (e.g., "cognition"
        belongs to both dictionary and identity columns).

        **Polysemy handling**: If the name already exists but the new
        concept has a different definition (from a different sense of
        the word), a separate concept is created with a disambiguated
        ID (e.g., "i#2"). The sense_rank property determines which
        sense is retrieved by _resolve() — lower rank = more common.

        The concept name is syntactically normalized (lowercased,
        hyphens→spaces, whitespace collapsed) before use as an ID.
        This ensures "micro-animal" and "micro animal" resolve to the
        same concept. Semantic normalization (plural→singular, article
        stripping) is the caller's responsibility.
        """
        name = _normalize_id(name)
        new_column = _column_of(origin, name)

        # Check if this concept already exists (by name or alias)
        match = self._find_matching_sense(name, properties)
        if match is not None:
            existing, eid = match
            return self._merge_into_existing(
                existing, eid, aliases, confidence, properties, new_column
            )

        existing_ids = self._alias_map.get(name, [])
        if existing_ids:
            # Different definition → this is a new sense of the word.
            # Create a separate concept with a disambiguated ID.
            sense_num = len(existing_ids) + 1
            return self._create_disambiguated_concept(
                name, sense_num, aliases, confidence, origin, new_column, properties
            )

        # Check the long-term archive before creating a duplicate.
        # If the concept was spilled to the archive (dormant but not
        # forgotten), recall it and merge into the recalled concept
        # instead of creating a new one. This prevents the autonomous
        # learner from re-creating concepts that already exist on disk
        # — e.g. when a concept is spilled between a get_concept check
        # and the subsequent add_concept call (race with sleep spill).
        if self._archive is not None:
            try:
                if self._archive.has_concept(name):
                    recalled = self._recall_from_archive_by_id(name)
                    if recalled is not None:
                        return self._merge_into_existing(
                            recalled, recalled.id, aliases, confidence,
                            properties, new_column,
                        )
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"archive check in add_concept failed for {name!r}: {e}")

        # Create new concept
        return self._create_new_concept(
            name, aliases, confidence, origin, new_column, properties
        )
    def _merge_into_existing(
        self,
        existing: Concept,
        eid: str,
        aliases: set[str] | None,
        confidence: float,
        properties: dict[str, Any] | None,
        new_column: str,
    ) -> Concept:
        """Merge new concept data into an existing concept (same sense).

        Boosts activation, adds aliases, updates confidence and
        properties, and registers the new column.
        """
        existing.activation = min(1.0, (existing.activation or 0.0) + 0.3)
        self._mark_active(existing.id)
        if aliases:
            for a in aliases:
                existing.aliases.add(a)
                self._add_alias(a, eid)
        if confidence > existing.confidence:
            existing.confidence = confidence
        if properties:
            # Don't let an empty definition overwrite a non-empty one.
            # This happens when a re-learning pass extracts no
            # definition for a concept that already has one — the
            # merge should preserve the existing definition, not erase
            # it. Other properties (e.g., source, confidence) are
            # still updated normally.
            new_def = properties.get("definition", "")
            if not new_def and existing.properties.get("definition"):
                merge_props = {k: v for k, v in properties.items() if k != "definition"}
            else:
                merge_props = properties
            existing.properties.update(merge_props)
            # If the concept is still uncategorized and the merged
            # properties include a definition, try to categorize it now.
            # This catches concepts that were created without a
            # definition (e.g., from conversation) and later receive
            # one (e.g., from dictionary learning).
            if existing.category == ConceptCategory.UNKNOWN:
                def_val = existing.properties.get("definition")
                definition = def_val if isinstance(def_val, str) else ""
                cat = detect_category(eid, definition)
                if cat != ConceptCategory.UNKNOWN:
                    existing.category = cat
                    existing.is_animacy_detected = True
            if existing.modality == ConceptModality.UNKNOWN:
                def_val = existing.properties.get("definition")
                definition = def_val if isinstance(def_val, str) else ""
                mod = detect_modality(eid, definition)
                if mod != ConceptModality.UNKNOWN:
                    existing.modality = mod
        existing.columns.add(new_column)
        return existing
    def _add_alias(self, alias: str, concept_id: str) -> None:
        """Add an alias mapping, supporting multiple senses.

        If the alias already maps to other concepts, this concept is
        added to the list. The list is kept sorted by sense_rank so
        _resolve() can pick the first (most common) sense efficiently.
        """
        key = _normalize_id(alias)
        if key not in self._alias_map:
            self._alias_map[key] = [concept_id]
        elif concept_id not in self._alias_map[key]:
            self._alias_map[key].append(concept_id)
            # Sort by sense_rank (lower = more common → first)
            self._alias_map[key].sort(
                key=lambda cid: self._concepts.get(
                    cid, Concept(id="", confidence=0)
                ).properties.get("sense_rank", 5)
            )
        else:
            return  # no change — don't invalidate the search index
        # Any alias mutation invalidates the trigram search index
        self._search_index = None
    def add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float = 0.5,
        origin: str = "stated",
    ) -> Edge | None:
        """Add a relationship between two concepts.

        Both concepts must exist (or are created automatically).
        If the same edge already exists, its weight is increased.

        When auto-creating missing concepts, the origin is inferred
        from the concept ID prefix (python:/rust: → "code") rather
        than defaulting to "conversation". This ensures code concepts
        created as edge endpoints are correctly labeled.
        """
        # Ensure both concepts exist
        src_id = self._resolve(source)
        if not src_id:
            src_id = self.add_concept(source, origin=_infer_concept_origin(source)).id
        tgt_id = self._resolve(target)
        if not tgt_id:
            tgt_id = self.add_concept(target, origin=_infer_concept_origin(target)).id

        # Check for existing edge (O(1) via composite key index)
        key = (src_id, tgt_id, relation)
        existing = self._edge_key_index.get(key)
        if existing is not None:
            existing.weight = min(1.0, existing.weight + 0.2)
            return existing

        # Create new edge
        edge = Edge(
            source=src_id,
            target=tgt_id,
            relation=relation,
            weight=weight,
            origin=origin,
        )
        self._edges.append(edge)
        self._edge_index.setdefault(src_id, []).append(edge)
        self._reverse_index.setdefault(tgt_id, []).append(edge)
        self._edge_key_index[key] = edge
        # Edge counts affect quality scores — invalidate the cache.
        self._quality_concept_ids_cache = None
        return edge
    def get_concept(self, name: str) -> Concept | None:
        """Get a concept by name or alias."""
        cid = self._resolve(name)
        if cid:
            # .get() not [cid] — _resolve can return a stale alias-map
            # ID if a concurrent thread (learner, inner_life) removed
            # the concept between the alias lookup and here.
            return self._concepts.get(cid)
        return None
    def has_concept(self, cid: str) -> bool:
        """Check if a concept ID exists in the network.

        This is the public accessor for ``cid in network._concepts``.
        Unlike ``get_concept``, it takes an exact concept ID (not a
        name or alias) and does not resolve.
        """
        return cid in self._concepts
    def get_edges(self, concept: str, direction: str = "out") -> list[Edge]:
        """Get edges connected to a concept.

        direction: "out" (from this concept), "in" (to this concept),
                   or "both".
        """
        cid = self._resolve(concept)
        if not cid:
            return []

        edges: list[Edge] = []
        if direction in ("out", "both"):
            edges.extend(self._edge_index.get(cid, []))
        if direction in ("in", "both"):
            edges.extend(self._reverse_index.get(cid, []))
        return edges
    def get_neighbors(
        self, concept: str, relation: RelationType | None = None
    ) -> list[tuple[str, RelationType, float]]:
        """Get neighboring concepts with their relation types and weights.

        Returns list of (neighbor_id, relation, weight) tuples.

        If a holographic graph is attached, associations migrated
        during sleep compression are also included. This keeps
        bridge edges and LTM-extracted associations accessible after
        they've been removed from the explicit edge list.
        """
        cid = self._resolve(concept)
        if not cid:
            return []

        neighbors: list[tuple[str, RelationType, float]] = []
        for edge in self._edge_index.get(cid, []):
            if relation is None or edge.relation == relation:
                neighbors.append((edge.target, edge.relation, edge.weight))
        for edge in self._reverse_index.get(cid, []):
            if relation is None or edge.relation == relation:
                # Reverse the relation for incoming edges
                neighbors.append((edge.source, edge.relation, edge.weight))

        # Supplement with holographic associations (compressed bridge
        # edges and LTM-extracted associations). These are fuzzy
        # associations — the holographic graph returns similarity
        # scores rather than exact edge weights.
        if self._holographic_graph is not None:
            holo_neighbors = self._query_holographic_neighbors(cid, relation)
            neighbors.extend(holo_neighbors)

        return neighbors
    def has_neighbors(self, concept: str) -> bool:
        """Check if a concept has any explicit edges (no holographic query).

        This is a fast existence check that only examines the explicit
        edge index — it does NOT query the holographic graph. Use this
        instead of ``bool(get_neighbors(c))`` when you only need to know
        *if* relationships exist, not what they are. The holographic
        query is expensive (FFT + dot products against ~3000 concepts)
        and should not be triggered for simple existence checks.
        """
        cid = self._resolve(concept)
        if not cid:
            return False
        return bool(self._edge_index.get(cid) or self._reverse_index.get(cid))
    def _query_holographic_neighbors(
        self, cid: str, relation: RelationType | None = None
    ) -> list[tuple[str, RelationType, float]]:
        """Query the holographic graph for associations from ``cid``.

        Returns (neighbor_id, relation, weight) tuples. The weight
        is the holographic similarity score, which ranges [0, 1].
        Errors are swallowed — a holographic graph failure must never
        break neighbor lookup.
        """
        result: list[tuple[str, RelationType, float]] = []
        try:
            if relation is not None:
                hits = self._holographic_graph.query(
                    cid, relation.value, top_k=10,
                )
                for target, sim in hits:
                    result.append((target, relation, float(sim)))
            else:
                all_hits = self._holographic_graph.query_all_relations(
                    cid, top_k=10,
                )
                for rel_str, hits in all_hits.items():
                    try:
                        rel = RelationType(rel_str)
                    except ValueError:
                        rel = RelationType.RELATED_TO
                    for target, sim in hits:
                        result.append((target, rel, float(sim)))
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            logger.debug(f"holographic neighbor query failed: {e}")
        return result
    def find_concepts_by_origin(self, origin: str, limit: int = 100) -> list[str]:
        """Find all concept IDs with a given origin.

        Used for context-aware resolution: when the conversation is
        about code, we want to find code-origin concepts, not dictionary
        concepts that happen to share a word.
        """
        result = []
        for cid, concept in list(self._concepts.items()):
            if concept.origin != origin:
                continue
            result.append(cid)
            if len(result) >= limit:
                break
        return result
    def find_concepts_by_property(
        self, key: str, value: str, limit: int = 100
    ) -> list[Concept]:
        """Find all concepts that have a specific property key=value pair.

        Used for structural lookups like emotion-word associations:
        concepts with ``properties["emotion_category"] = "stressed"``
        are words Genesis has learned to associate with that emotional
        category. Returns concepts sorted by activation (highest first),
        so the most recently used words come first.
        """
        result: list[Concept] = []
        for concept in list(self._concepts.values()):
            if concept.properties.get(key) == value:
                result.append(concept)
                if len(result) >= limit:
                    break
        result.sort(key=lambda c: c.activation or 0.0, reverse=True)
        return result
    def _find_words_for_hub(
        self, hub_id: str, property_key: str, property_value: str
    ) -> list[str]:
        """Find words that EXPRESS a category hub, via graph traversal.

        Primary path: traverse EXPRESSES edges from the hub concept.
        Fallback: property scan (for concepts tagged with properties
        but not yet connected via edges).

        Returns concept IDs sorted by activation (highest first).
        """
        # Primary: graph traversal from hub
        neighbors = self.get_neighbors(hub_id, RelationType.EXPRESSES)
        if neighbors:
            word_ids = [n[0] for n in neighbors]
            # Sort by activation
            word_ids.sort(
                key=lambda wid: (self._concepts[wid].activation or 0.0)
                if wid in self._concepts else 0.0,
                reverse=True,
            )
            return word_ids

        # Fallback: property scan (backward compatibility)
        concepts = self.find_concepts_by_property(property_key, property_value)
        return [c.id for c in concepts]
    def find_emotion_words(self, category: str) -> list[str]:
        """Find words associated with an emotion category.

        Traverses EXPRESSES edges from the ``_cat:emotion:<category>``
        hub concept. Falls back to property scan for backward
        compatibility. If Genesis has not learned any words for this
        category, returns an empty list — she cannot describe that
        emotion in words yet.
        """
        return self._find_words_for_hub(
            f"_cat:emotion:{category}", "emotion_category", category
        )
    def find_cognitive_mode_words(self, mode: str) -> list[str]:
        """Find words associated with a cognitive mode.

        Traverses EXPRESSES edges from the ``_cat:mode:<mode>`` hub
        concept. Falls back to property scan for backward compatibility.
        """
        return self._find_words_for_hub(
            f"_cat:mode:{mode}", "cognitive_mode", mode
        )
    def find_cause_words(self, cause_category: str) -> list[str]:
        """Find words associated with a cause category.

        Traverses EXPRESSES edges from the ``_cat:cause:<category>``
        hub concept. Falls back to property scan for backward
        compatibility.
        """
        if not cause_category:
            return []
        return self._find_words_for_hub(
            f"_cat:cause:{cause_category}", "cause_category", cause_category
        )
    def find_plasticity_words(self, state: str) -> list[str]:
        """Find words associated with a plasticity state.

        Traverses EXPRESSES edges from the ``_cat:plasticity:<state>``
        hub concept. Falls back to property scan for backward
        compatibility.
        """
        return self._find_words_for_hub(
            f"_cat:plasticity:{state}", "plasticity_state", state
        )
    def find_alertness_words(self, state: str) -> list[str]:
        """Find words associated with an alertness state.

        Traverses EXPRESSES edges from the ``_cat:alertness:<state>``
        hub concept. Falls back to property scan for backward
        compatibility. If Genesis has not learned any words for this
        alertness state, returns an empty list — she cannot describe
        that state in words yet.
        """
        return self._find_words_for_hub(
            f"_cat:alertness:{state}", "alertness_state", state
        )
    def find_valence_words(self, state: str) -> list[str]:
        """Find words associated with a valence state.

        Traverses EXPRESSES edges from the ``_cat:valence:<state>``
        hub concept. Falls back to property scan for backward
        compatibility. If Genesis has not learned any words for this
        valence state, returns an empty list — she cannot describe
        that state in words yet.
        """
        return self._find_words_for_hub(
            f"_cat:valence:{state}", "valence_state", state
        )
    def seed_relation_verbs(self) -> None:
        """Seed relation-verb phrases into the concept network.

        For each RelationType in ``_RELATION_VERB_SEEDS``, creates a
        hub concept ``_cat:relation_verb:<relation>`` and connects
        each verb phrase to it via an EXPRESSES edge.
        """
        for relation_str, verbs in self._RELATION_VERB_SEEDS.items():
            hub_id = f"_cat:relation_verb:{relation_str}"
            if self.get_concept(hub_id) is None:
                self.add_concept(hub_id, confidence=0.7, origin="structural")
            for verb in verbs:
                display = verb.lower().strip()
                cid = "_utt:" + display.replace(" ", "_")
                cid = cid.rstrip("_")
                if not cid or len(cid) < 2:
                    continue
                if self.get_concept(cid) is None:
                    self.add_concept(
                        cid, origin="structural",
                        properties={"template": display},
                    )
                existing = self.get_neighbors(cid, RelationType.EXPRESSES)
                if not any(n[0] == hub_id for n in existing):
                    self.add_edge(
                        cid, hub_id, RelationType.EXPRESSES,
                        weight=0.6, origin="relation_verb_seed",
                    )
    def find_relation_verbs(self, relation: RelationType) -> list[str]:
        """Find verb phrases for a relation type.

        Traverses EXPRESSES edges from the
        ``_cat:relation_verb:<relation>`` hub. Returns verb phrases
        converted back to display form, using the ``template`` property
        when available (seeded verbs store their display form there).
        """
        hub_id = f"_cat:relation_verb:{relation.value}"
        neighbors = self.get_neighbors(hub_id, RelationType.EXPRESSES)
        if not neighbors:
            return []
        results: list[str] = []
        for n in neighbors:
            concept = self.get_concept(n[0])
            if concept is None:
                continue
            template = concept.properties.get("template")
            if isinstance(template, str):
                results.append(template)
            else:
                results.append(n[0].replace("_", " "))
        return results
    def seed_relation_phrases(self) -> None:
        """Seed relation-phrase templates into the concept network.

        For each RelationType in ``_RELATION_PHRASE_SEEDS``, creates
        a hub concept ``_cat:relation_phrase:<relation>`` and connects
        each phrasing template to it via an EXPRESSES edge. The
        phrasings contain ``{target}`` placeholders that are filled
        in at runtime.
        """
        for relation_str, phrases in self._RELATION_PHRASE_SEEDS.items():
            hub_id = f"_cat:relation_phrase:{relation_str}"
            if self.get_concept(hub_id) is None:
                self.add_concept(hub_id, confidence=0.7, origin="structural")
            for phrase in phrases:
                cid = "_utt:" + phrase.replace(" ", "_").replace("{", "").replace("}", "").lower()
                cid = cid.rstrip("_")
                if not cid or len(cid) < 2:
                    continue
                if self.get_concept(cid) is None:
                    self.add_concept(
                        cid, origin="structural",
                        properties={"template": phrase},
                    )
                existing = self.get_neighbors(cid, RelationType.EXPRESSES)
                if not any(n[0] == hub_id for n in existing):
                    self.add_edge(
                        cid, hub_id, RelationType.EXPRESSES,
                        weight=0.6, origin="relation_phrase_seed",
                    )
    def find_relation_phrases(self, relation: RelationType) -> list[str]:
        """Find phrasing templates for a relation type.

        Traverses EXPRESSES edges from the
        ``_cat:relation_phrase:<relation>`` hub. Returns phrasing
        templates with ``{target}`` placeholders, converted back to
        display form. Returns an empty list if no phrases are found.
        Uses the ``template`` property when available (seeded phrases
        store their verbatim template there).
        """
        hub_id = f"_cat:relation_phrase:{relation.value}"
        neighbors = self.get_neighbors(hub_id, RelationType.EXPRESSES)
        if not neighbors:
            return []
        results: list[str] = []
        for n in neighbors:
            concept = self.get_concept(n[0])
            if concept is None:
                continue
            template = concept.properties.get("template")
            if isinstance(template, str):
                results.append(template)
            else:
                results.append(
                    n[0].replace("_", " ").replace("target", "{target}")
                )
        return results
    def _hub_items_with_provenance(
        self, hub_id: str
    ) -> list[Provenance]:
        """Retrieve EXPRESSES neighbours of a hub with provenance.

        Returns items sorted by activation (highest first). Each item's
        ``text`` is the display form (underscores → spaces) or the
        verbatim ``template`` property when available.
        """
        neighbors = self.get_neighbors(hub_id, RelationType.EXPRESSES)
        if not neighbors:
            return []
        word_ids = [n[0] for n in neighbors]
        word_ids.sort(
            key=lambda wid: (self._concepts[wid].activation or 0.0)
            if wid in self._concepts else 0.0,
            reverse=True,
        )
        results: list[Provenance] = []
        for wid in word_ids:
            concept = self.get_concept(wid)
            if concept is None:
                continue
            template = concept.properties.get("template")
            text = template if isinstance(template, str) else wid.replace("_", " ")
            results.append(Provenance.from_concept(text, concept))
        return results
    def find_utterance_words_with_provenance(
        self, slot_name: str
    ) -> list[Provenance]:
        """Find utterance words for a slot, with seed/learned provenance.

        Like ``_utterance_words`` but returns ``Provenance`` items.
        """
        return self._hub_items_with_provenance(f"_cat:utterance:{slot_name}")
    def find_relation_verbs_with_provenance(
        self, relation: RelationType
    ) -> list[Provenance]:
        """Find relation verb phrases, with seed/learned provenance."""
        return self._hub_items_with_provenance(
            f"_cat:relation_verb:{relation.value}"
        )
    def find_relation_phrases_with_provenance(
        self, relation: RelationType
    ) -> list[Provenance]:
        """Find relation phrase templates, with seed/learned provenance."""
        return self._hub_items_with_provenance(
            f"_cat:relation_phrase:{relation.value}"
        )
    def has_learned_utterances(self, slot_name: str) -> bool:
        """Return True if any non-seed utterance words exist for this slot."""
        return any(
            not p.is_seed
            for p in self.find_utterance_words_with_provenance(slot_name)
        )
    def search_concepts(
        self,
        query: str,
        origins: set[str] | None = None,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[str]:
        """Search for concepts matching a query, optionally filtered by origin.

        This is the context-aware concept resolution mechanism. When
        the conversation is self-referential ("your code", "your daemon"),
        the perception layer calls this with origins={"code", "identity"}
        to find self-relevant concepts instead of dictionary definitions.

        Matching handles:
        - Direct substring matching ("daemon" → "start_daemon")
        - Multi-word queries against camelCase ("cognition engine" →
          "CognitionEngine")
        - Word-level matching within compound names ("cognition" →
          "CognitionEngine")

        When origins is None, all concepts are searched.

        Uses a trigram inverted index for O(candidates) lookups instead
        of scanning all concepts. The index is built lazily on first
        call and invalidated on concept/alias mutations.
        """
        query_lower = query.lower()
        query_words = query_lower.split()
        query_compact = "".join(query_words)  # "cognition engine" → "cognitionengine"

        # Gather candidate concept IDs via trigram index.
        # For queries >= 3 chars, use the inverted index. For shorter
        # queries (rare in practice), fall back to full scan.
        candidates = self._search_candidates(query_lower, query_words, query_compact)

        results: list[tuple[float, str]] = []
        for cid in candidates:
            concept = self._concepts.get(cid)
            if concept is None:
                continue
            if origins is not None:
                # Check if the concept's column is in the origins set.
                # This handles the introspection→identity column merge:
                # a concept with origin="introspection" belongs to the
                # "identity" column and should match origins={"identity"}.
                col = _column_of(concept.origin, cid)
                # Also check raw origin for backward compatibility
                # (e.g., "learned", "conversation" origins that
                # don't have a column mapping)
                if col not in origins and concept.origin not in origins:
                    continue

            score = self._score_concept_match(
                cid, concept, query_lower, query_words, query_compact
            )

            if score > min_score:
                results.append((score, cid))

        # Sort by score, return top results
        results.sort(key=lambda x: -x[0])
        return [cid for _, cid in results[:limit]]
    def _search_candidates(
        self,
        query_lower: str,
        query_words: list[str],
        query_compact: str,
    ) -> set[str]:
        """Return candidate concept IDs for a query via trigram index.

        Falls back to scanning all concepts when the query is too short
        for trigram extraction (< 3 characters) or when the index is
        empty (no concepts).
        """
        if not self._concepts:
            return set()

        # Collect all substrings to search trigrams from:
        # the full query, the compact form, and each individual word.
        search_strings = [query_lower]
        if len(query_words) > 1 and query_compact != query_lower:
            search_strings.append(query_compact)
        search_strings.extend(
            w for w in query_words if w != query_lower and len(w) >= 3
        )

        # If no search string is long enough for trigrams, fall back to
        # scanning all concepts (short queries are rare).
        trigram_sets = [_extract_trigrams(s) for s in search_strings]
        if not any(trigram_sets):
            return set(self._concepts.keys())

        # Build index lazily. Snapshot the reference into a local
        # variable — another thread (learner, inner_life) may set
        # _search_index = None between our check and our use, which
        # would cause 'NoneType' object has no attribute 'get'.
        if self._search_index is None:
            self._build_search_index()
        search_index = self._search_index
        if search_index is None:
            # Another thread set it back to None — fall back to
            # scanning all concepts (slower but correct).
            return set(self._concepts.keys())

        # Union candidates from each search string's trigram intersection.
        # A concept is a candidate if it contains ALL trigrams of any
        # search string (necessary condition for substring containment).
        candidates: set[str] = set()
        for trigrams in trigram_sets:
            if not trigrams:
                continue
            # Intersect postings for all trigrams in this search string
            postings: set[str] | None = None
            for tri in trigrams:
                ids = search_index.get(tri)
                if ids is None:
                    # This trigram doesn't exist in any concept → no
                    # concept contains all trigrams of this search string
                    postings = set()
                    break
                if postings is None:
                    postings = set(ids)
                else:
                    postings &= ids
                if not postings:
                    break
            if postings:
                candidates |= postings

        return candidates
    def _build_search_index(self) -> None:
        """Build the trigram inverted index from all concept names and aliases."""
        index: dict[str, set[str]] = {}
        for cid, concept in list(self._concepts.items()):
            # Index the canonical name
            for tri in _extract_trigrams(cid.lower()):
                index.setdefault(tri, set()).add(cid)
            # Index all aliases
            for alias in concept.aliases:
                for tri in _extract_trigrams(alias.lower()):
                    index.setdefault(tri, set()).add(cid)
        self._search_index = index
    def _score_concept_match(
        self,
        cid: str,
        concept: Concept,
        query_lower: str,
        query_words: list[str],
        query_compact: str,
    ) -> float:
        """Score a single concept against the query."""
        cid_lower = cid.lower()
        score = 0.0

        # 1. Direct substring match
        if query_lower in cid_lower:
            if cid_lower == query_lower:
                score = 1.0
            elif cid_lower.startswith(query_lower):
                score = 0.95
            elif cid_lower.endswith(query_lower):
                score = 0.85
            else:
                score = 0.75

        # 2. Compact match (multi-word query vs camelCase without separators)
        elif len(query_words) > 1 and query_compact in cid_lower:
            # "cognition engine" → "cognitionengine" matches "cognitionengine"
            score = 0.9

        # 3. Word-level match — split camelCase and check each query word
        elif query_words:
            # Split camelCase: "CognitionEngine" → ["cognition", "engine"]
            # Also split on _, :, .
            parts = _CAMEL_SPLIT_RE.split(cid)
            parts_lower = [p.lower() for p in parts if p]

            # Check if all query words appear in the parts
            matched_words = sum(1 for qw in query_words if any(qw in p for p in parts_lower))
            if matched_words > 0:
                # Multi-word query with only partial match — score
                # very low so it doesn't preempt better matches
                # (e.g., "cognition work" shouldn't match
                # "concept_network" just because "work" is in "network")
                if len(query_words) > 1 and matched_words < len(query_words):
                    score = 0.2 * (matched_words / len(query_words))
                # Score based on what fraction of query words matched
                # Bonus if all words matched
                elif matched_words == len(query_words):
                    score = min(
                        0.8 * (matched_words / len(query_words)) + 0.1, 0.9
                    )
                else:
                    score = 0.8 * (matched_words / len(query_words))

        # 4. Check aliases (lower priority)
        if score == 0.0:
            for alias in concept.aliases:
                if query_lower not in alias.lower():
                    continue
                score = 0.6
                break

        return score
    def get_column(self, concept_id: str) -> str:
        """Get the primary cortical column a concept belongs to."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            return "unknown"
        return _column_of(concept.origin, concept_id)
    def get_columns_for(self, concept_id: str) -> set[str]:
        """Get all columns a concept belongs to (multi-column membership)."""
        concept = self._concepts.get(concept_id)
        if concept is None:
            return set()
        if concept.columns:
            return concept.columns
        return {_column_of(concept.origin, concept_id)}
    def _laminar_weight(self, concept_id: str) -> float:
        """Layer-aware activation weight.

        Concepts that project outward (L5-like) amplify their column's
        competitive signal; concepts that receive input (L4-like) dampen
        it. The weight is derived from the concept's in-degree and
        out-degree, not from a hard-coded layer table.
        """
        out_degree = len(self._edge_index.get(concept_id, []))
        in_degree = len(self._reverse_index.get(concept_id, []))
        total = in_degree + out_degree
        # Output-biased concepts get a gain > 1; input-biased get < 1.
        return 1.0 + (out_degree - in_degree) / (total + 2)
    def get_column_activation(self, column: str) -> float:
        """Get the mean activation level of a column.

        Uses MEAN activation, not sum. This is critical: a column with
        122K concepts at activation 0.01 (mean=0.01) should NOT dominate
        a column with 10 concepts at activation 0.5 (mean=0.5). Sum-based
        activation would make larger columns always win, which is
        biologically wrong — cortical columns compete on population
        firing rate (mean), not total activity (sum).

        Additionally, each concept's contribution is weighted by its
        laminar role: output-projecting concepts (L5-like) count more,
        input-receiving concepts (L4-like) count less.

        Uses the ``_column_index`` for O(members) lookup instead of
        scanning the entire concept network.
        """
        cids = self._column_index.get(column, set())
        total = 0.0
        count = 0
        for cid in cids:
            concept = self._concepts.get(cid)
            if concept is not None:
                total += (concept.activation or 0.0) * self._laminar_weight(cid)
                count += 1
        if count == 0:
            return 0.0
        return total / count
    def get_column_concepts(self, column: str) -> list[str]:
        """Get all concept IDs in a given column."""
        return [
            cid for cid in self._column_index.get(column, set())
            if cid in self._concepts
        ]
    def get_columns(self) -> dict[str, int]:
        """Get all columns and their sizes.

        Uses the ``_column_index`` for O(columns) lookup instead of
        scanning the entire concept network.
        """
        return {
            column: sum(1 for cid in cids if cid in self._concepts)
            for column, cids in self._column_index.items()
        }
    def get_layer(self, concept_id: str) -> CorticalLayer:
        """Infer the cortical layer of a concept from its graph topology.

        This is a data-driven assignment, not a hard-coded list. The
        relative in-degree and out-degree of a concept in the network
        decide whether it functions as input (L4), association (L2_3),
        output (L5), or context (L6) within its column.
        """
        concept = self._concepts.get(concept_id)
        if concept is None:
            return CorticalLayer.L2_3

        out_degree = len(self._edge_index.get(concept_id, []))
        in_degree = len(self._reverse_index.get(concept_id, []))

        # Thresholds are relative to current network density.
        out_strong = out_degree >= 3
        in_strong = in_degree >= 3

        if in_strong and not out_strong:
            return CorticalLayer.L4
        if out_strong and not in_strong:
            return CorticalLayer.L5
        if in_strong and out_strong:
            return CorticalLayer.L2_3
        return CorticalLayer.L6
    def get_column_layers(self, column: str) -> dict[str, int]:
        """Get the layer distribution for a given column."""
        counts: dict[str, int] = {}
        for cid, concept in list(self._concepts.items()):
            cols = concept.columns or {_column_of(concept.origin, cid)}
            if column not in cols:
                continue
            layer = self.get_layer(cid)
            counts[layer.value] = counts.get(layer.value, 0) + 1
        return counts
    def get_layer_distribution(self) -> dict[str, int]:
        """Get the global layer distribution across all columns."""
        counts: dict[str, int] = {}
        for cid in self._concepts:
            layer = self.get_layer(cid)
            counts[layer.value] = counts.get(layer.value, 0) + 1
        return counts
    def categorize_all(self) -> dict[str, int]:
        """Run detect_category on all uncategorized concepts.

        This populates the `category` field for every concept that is
        still UNKNOWN (either because it was loaded from an older
        persistence format, or because it was added before
        categorization was available). Concepts that already have a
        detected category are left untouched.

        Returns a dict with counts of how many concepts were assigned
        to each category.
        """
        counts: dict[str, int] = {}
        for concept in list(self._concepts.values()):
            if concept.category != ConceptCategory.UNKNOWN:
                continue
            # Use the concept's properties as a definition if available
            definition = ""
            def_val = (concept.properties or {}).get("definition")
            if isinstance(def_val, str):
                definition = def_val
            detected = detect_category(concept.id, definition)
            if detected != ConceptCategory.UNKNOWN:
                concept.category = detected
                concept.is_animacy_detected = True
                counts[detected.value] = counts.get(detected.value, 0) + 1
        return counts
    def get_by_category(self, category: ConceptCategory) -> list[Concept]:
        """Get all concepts in a given category.

        Args:
            category: The ConceptCategory to filter by.

        Returns:
            A list of Concept objects in that category.
        """
        return [
            concept for concept in list(self._concepts.values()) if concept.category == category
        ]
    def classify_modalities(self) -> dict[str, int]:
        """Run detect_modality on all unclassified concepts.

        This populates the `modality` field for every concept that is
        still UNKNOWN. Concepts that already have a detected modality
        are left untouched.

        Returns a dict with counts of how many concepts were assigned
        to each modality.
        """
        counts: dict[str, int] = {}
        for concept in list(self._concepts.values()):
            if concept.modality != ConceptModality.UNKNOWN:
                continue
            definition = ""
            def_val = (concept.properties or {}).get("definition")
            if isinstance(def_val, str):
                definition = def_val
            detected = detect_modality(concept.id, definition)
            if detected != ConceptModality.UNKNOWN:
                concept.modality = detected
                counts[detected.value] = counts.get(detected.value, 0) + 1
        return counts
    def get_by_modality(self, modality: ConceptModality) -> list[Concept]:
        """Get all concepts in a given modality.

        Args:
            modality: The ConceptModality to filter by.

        Returns:
            A list of Concept objects in that modality.
        """
        return [
            concept for concept in list(self._concepts.values()) if concept.modality == modality
        ]
    def detect_semantic_hubs(self, top_percent: float = 0.05) -> list[str]:
        """Detect semantic hubs using betweenness centrality.

        Semantic hubs are convergence zones (Damasio, 1989) — concepts
        that sit at the intersection of many pathways. They integrate
        information from diverse sources and are critical for
        semantic memory retrieval.

        This method computes betweenness centrality for all concepts
        (using a sampled approximation for large networks) and marks
        the top 5% as semantic hubs. Hub concepts get:
        - 1.3x spread multiplier in spreading activation
        - 0.7x decay rate (slower activation decay)

        Args:
            top_percent: Fraction of concepts to designate as hubs
                         (default 0.05 = top 5%).

        Returns:
            A list of concept IDs that were marked as semantic hubs.
        """
        # Reset all hub flags first
        for concept in list(self._concepts.values()):
            concept.is_semantic_hub = False

        if len(self._concepts) < 3:
            # Too few concepts to meaningfully detect hubs
            return []

        # Compute betweenness centrality.
        # For small networks (< 500 nodes), compute exact betweenness
        # using all-pairs shortest paths. For larger networks, sample
        # source nodes to keep it tractable.
        centrality = self._compute_betweenness_centrality()

        if not centrality:
            return []

        # Sort by centrality (descending) and mark top N as hubs
        sorted_concepts = sorted(centrality.items(), key=lambda x: -x[1])
        n_hubs = max(1, int(len(sorted_concepts) * top_percent))

        hubs: list[str] = []
        for cid, _score in sorted_concepts[:n_hubs]:
            hub_concept = self._concepts.get(cid)
            if hub_concept is not None:
                hub_concept.is_semantic_hub = True
                hubs.append(cid)

        return hubs
    def _compute_betweenness_centrality(self, max_sources: int = 200) -> dict[str, float]:
        """Compute approximate betweenness centrality for all concepts.

        Betweenness centrality measures how often a node appears on
        shortest paths between other nodes. Nodes with high betweenness
        act as bridges — removing them would disconnect parts of the
        network.

        For large networks, we sample source nodes (max_sources) to
        keep computation tractable, following the approach of
        Brandes (2001) with sampling.

        Args:
            max_sources: Maximum number of source nodes to sample
                         for the centrality computation.

        Returns:
            A dict mapping concept IDs to their betweenness centrality.
        """
        all_cids = list(self._concepts.keys())
        n = len(all_cids)
        if n < 2:
            return {}

        # Sample source nodes if network is large
        if n > max_sources:
            rng = random.Random(42)  # deterministic for reproducibility
            sources = rng.sample(all_cids, max_sources)
        else:
            sources = all_cids

        centrality: dict[str, float] = dict.fromkeys(all_cids, 0.0)

        # Brandes' algorithm (simplified): for each source, compute
        # shortest paths via BFS and accumulate dependency scores.
        for source in sources:
            delta = self._betweenness_for_source(source, all_cids)
            for w in all_cids:
                if w != source:
                    centrality[w] += delta[w]

        # Normalize by the number of source samples
        scale = len(sources)
        if scale > 0:
            for cid in centrality:
                centrality[cid] /= scale

        return centrality
    def _betweenness_for_source(
        self, source: str, all_cids: list[str]
    ) -> dict[str, float]:
        """Compute dependency scores for a single source node (Brandes' BFS).

        Returns a delta dict where ``delta[w]`` is the dependency of
        ``source`` on ``w``. The caller accumulates these into centrality.
        """
        # BFS from source, tracking shortest paths
        # Stack for accumulation phase
        stack: list[str] = []
        # Predecessors on shortest paths
        pred: dict[str, list[str]] = {cid: [] for cid in all_cids}
        # Number of shortest paths from source to each node
        sigma: dict[str, float] = dict.fromkeys(all_cids, 0.0)
        sigma[source] = 1.0
        # Distance from source
        dist: dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            d_v = dist[v]
            for edge in self._edge_index.get(v, []):
                w = edge.target
                if w not in dist:
                    dist[w] = d_v + 1
                    queue.append(w)
                if dist.get(w, float("inf")) == d_v + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Accumulation phase: compute dependency scores
        delta: dict[str, float] = dict.fromkeys(all_cids, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])

        return delta
    @property
    def semantic_hubs(self) -> list[Concept]:
        """List of all concepts currently marked as semantic hubs."""
        return [concept for concept in list(self._concepts.values()) if concept.is_semantic_hub]
    def find_path(self, source: str, target: str, max_depth: int = 5) -> list[Edge] | None:
        """Find a path between two concepts using BFS.

        Returns the list of edges forming the path, or None if no
        path exists within max_depth.

        The search is **undirected**: it follows both outgoing and
        incoming edges. Semantic relations like ``is_a``, ``related_to``,
        ``part_of`` and ``similar_to`` are effectively bidirectional in
        meaning, and a query like "how are A and B related?" should not
        fail just because the stored edge points the other way (e.g.
        ``alice creates genesis`` should explain ``genesis`` ↔ ``alice``,
        not require a separate ``genesis`` → ``alice`` edge).

        Returned edges are **oriented along the traversal direction**:
        ``path[0].source`` is always the resolved source concept and
        ``path[-1].target`` is always the resolved target. When an edge
        is traversed backward (against its stored direction), a copy
        with source/target swapped is returned. The ``relation`` field
        is preserved as-stored — most path relations (``related_to``,
        ``is_a``, ``similar_to``) are symmetric, and for asymmetric ones
        the caller can compare ``edge.source``/``edge.target`` to the
        stored edge if needed.

        Implementation uses predecessor tracking (no per-step path
        copying), so it is O(V+E) in the reachable subgraph instead of
        O(b^d · d) for the previous copy-on-expand BFS.
        """
        src_id = self._resolve(source)
        tgt_id = self._resolve(target)
        if not src_id or not tgt_id:
            return None
        if src_id == tgt_id:
            return []

        # came_from[node] = (previous_node, edge_traversed)
        # where edge_traversed is the stored Edge between previous_node
        # and node (regardless of traversal direction).
        came_from: dict[str, tuple[str, Edge]] = {src_id: (src_id, _SENTINEL_EDGE)}
        queue: deque[tuple[str, int]] = deque([(src_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Outgoing edges: current -> neighbor
            for edge in self._edge_index.get(current, ()):
                neighbor = edge.target
                if neighbor == tgt_id:
                    came_from[neighbor] = (current, edge)
                    return self._reconstruct_path(came_from, src_id, tgt_id)
                if neighbor in came_from:
                    continue
                came_from[neighbor] = (current, edge)
                queue.append((neighbor, depth + 1))

            # Incoming edges: neighbor -> current (traversed backward)
            for edge in self._reverse_index.get(current, ()):
                neighbor = edge.source
                if neighbor == tgt_id:
                    came_from[neighbor] = (current, edge)
                    return self._reconstruct_path(came_from, src_id, tgt_id)
                if neighbor in came_from:
                    continue
                came_from[neighbor] = (current, edge)
                queue.append((neighbor, depth + 1))

        return None
    def _reconstruct_path(
        self,
        came_from: dict[str, tuple[str, Edge]],
        src_id: str,
        tgt_id: str,
    ) -> list[Edge]:
        """Walk the predecessor map from target back to source, returning
        edges in traversal order (source → target).

        Edges traversed backward (against their stored direction) are
        flipped so that ``source``/``target`` align with the traversal.
        """
        edges: list[Edge] = []
        node = tgt_id
        while node != src_id:
            prev, edge = came_from[node]
            if edge.target == prev:
                # Backward traversal: stored edge is node→prev, but we
                # traversed prev→node. Flip for consistent orientation.
                edge = replace(edge, source=edge.target, target=edge.source)
            edges.append(edge)
            node = prev
        edges.reverse()
        return edges
    def _resolve(self, name: str) -> str | None:
        """Resolve a name or alias to a concept id.

        The name is syntactically normalized before lookup, so
        "micro-animal" and "Micro Animal" both find the same concept.

        When a name maps to multiple concepts (polysemy — the word has
        multiple senses), the best sense is chosen by:
        1. sense_rank property (lower = more common sense)
        2. confidence (higher = more well-formed)
        3. activation (higher = recently used)

        This is word sense disambiguation based on frequency. The
        pronoun "I" (sense_rank=1) beats iodine "I" (sense_rank=5)
        because the pronoun is far more common.

        If the concept is not in working memory but a long-term archive
        is attached, the archive is checked and the concept is recalled
        transparently.
        """
        normalized = _normalize_id(name)
        candidates = self._alias_map.get(normalized)
        if not candidates:
            # Not in working memory — check the long-term archive
            if self._archive is not None:
                recalled = self._recall_from_archive_by_alias(normalized)
                if recalled is not None:
                    return recalled
            return None
        if len(candidates) == 1:
            only = candidates[0]
            if only in self._concepts:
                return only
            # Dangling alias (concurrent removal) — try the archive
            # before giving up.
            if self._archive is not None:
                return self._recall_from_archive_by_alias(normalized)
            return None

        # Multiple senses — pick the best one
        best_id = None
        best_score = -1.0
        for cid in candidates:
            concept = self._concepts.get(cid)
            if concept is None:
                continue
            # sense_rank: lower is better (1 = primary sense)
            # Default sense_rank is 5 (neutral) if not set
            sense_rank = concept.properties.get("sense_rank", 5)
            # Convert to a 0..1 score (rank 1 → 1.0, rank 10 → 0.1)
            sense_score = max(0.0, 1.0 - (sense_rank - 1) * 0.1)
            # Combine with confidence and activation
            score = (
                sense_score * 0.5
                + concept.confidence * 0.3
                + (concept.activation or 0.0) * 0.2
            )
            if score > best_score:
                best_score = score
                best_id = cid
        if best_id is not None:
            return best_id
        # Every candidate was dangling (concurrent removal between the
        # alias lookup and now) — check the archive rather than return
        # a stale ID that could anchor an edge to a phantom concept.
        if self._archive is not None:
            return self._recall_from_archive_by_alias(normalized)
        return None
    def _resolve_all_senses(self, name: str) -> list[str]:
        """Resolve a name to ALL concept IDs that share it (all senses).

        Unlike _resolve which picks the best sense, this returns every
        concept that claims this name. Used when you need to consider
        all meanings of a polysemous word.
        """
        return list(self._alias_map.get(_normalize_id(name), []))
    def resolve_in_context(self, name: str, context: str) -> str | None:
        """Resolve a polysemous name using definition and neighbor overlap."""
        candidates = self._resolve_all_senses(name)
        if len(candidates) <= 1:
            return self._resolve(name)

        context_terms = {
            term for term in re.findall(r"[^\W_]+", context.casefold())
            if len(term) > 2 and term not in _FUNCTION_WORDS
        }
        if not context_terms:
            return self._resolve(name)

        ranked: list[tuple[float, str]] = []
        for cid in candidates:
            concept = self._concepts.get(cid)
            if concept is None:
                continue
            definition = str(concept.properties.get("definition", ""))
            signature = set(re.findall(r"[^\W_]+", definition.casefold()))
            for edge in self._edge_index.get(cid, []):
                signature.update(re.findall(r"[^\W_]+", edge.target.casefold()))
            for edge in self._reverse_index.get(cid, []):
                signature.update(re.findall(r"[^\W_]+", edge.source.casefold()))
            overlap = len(context_terms & signature) / len(context_terms)
            ranked.append((overlap, cid))

        if not ranked or max(score for score, _ in ranked) == 0.0:
            return self._resolve(name)
        return max(ranked, key=lambda item: item[0])[1]
    def remove_concept(self, name: str) -> bool:
        """Remove a concept and all its edges.

        Returns True if the concept was found and removed.
        """
        cid = self._resolve(name)
        if not cid:
            return False

        # Remove all edges involving this concept
        self._edges = [e for e in self._edges if e.source != cid and e.target != cid]

        # Also remove archived edges for this concept (permanent deletion)
        if self._archive is not None:
            try:
                self._archive.remove_edges_for_concept(cid)
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"archive edge cleanup failed for {cid!r}: {e}")

        # Rebuild edge indices
        self._edge_index = {}
        self._reverse_index = {}
        self._edge_key_index = {}
        for edge in list(self._edges):
            self._edge_index.setdefault(edge.source, []).append(edge)
            self._reverse_index.setdefault(edge.target, []).append(edge)
            self._edge_key_index[(edge.source, edge.target, edge.relation)] = edge

        # Remove concept and its aliases from the alias map
        concept = self._concepts.pop(cid, None)
        if concept:
            self._concept_ids_cache = None
            self._world_concept_ids_cache = None
            self._quality_concept_ids_cache = None
            self._search_index = None
            # Remove this concept from all alias lists.
            # Snapshot the items — the autonomous learner thread may
            # be adding concepts (and thus modifying _alias_map)
            # concurrently, which would raise "dictionary changed
            # size during iteration" without the list() copy.
            keys_to_clean = []
            for k, cids in list(self._alias_map.items()):
                if cid not in cids:
                    continue
                cids.remove(cid)
                if not cids:
                    keys_to_clean.append(k)
            for k in keys_to_clean:
                self._alias_map.pop(k, None)
            # Remove from the column index so stale IDs don't
            # accumulate in column membership sets.
            for col_cids in self._column_index.values():
                col_cids.discard(cid)

        return True
    def remove_concepts_batch(self, cids: set[str]) -> int:
        """Remove multiple concepts and their edges in a single pass.

        Unlike calling remove_concept in a loop (which rebuilds all
        edge indices on every call — O(N×E)), this filters edges once
        and rebuilds indices once — O(E + N).

        Returns the number of concepts actually removed.
        """
        if not cids:
            return 0

        removed = 0
        for cid in cids:
            if cid in self._concepts:
                self._concepts.pop(cid, None)
                removed += 1

        if removed == 0:
            return 0

        # Filter edges in a single pass
        self._edges = [
            e for e in self._edges
            if e.source not in cids and e.target not in cids
        ]

        # Also remove archived edges for permanently deleted concepts
        if self._archive is not None:
            for cid in cids:
                try:
                    self._archive.remove_edges_for_concept(cid)
                except (sqlite3.Error, RuntimeError) as e:
                    logger.debug(f"archive edge cleanup failed for {cid!r}: {e}")

        # Rebuild edge indices once
        self._edge_index = {}
        self._reverse_index = {}
        self._edge_key_index = {}
        for edge in list(self._edges):
            self._edge_index.setdefault(edge.source, []).append(edge)
            self._reverse_index.setdefault(edge.target, []).append(edge)
            self._edge_key_index[(edge.source, edge.target, edge.relation)] = edge

        # Clean alias map — remove pruned concept IDs from all alias lists
        keys_to_clean = []
        for k, alias_cids in list(self._alias_map.items()):
            for cid in cids:
                if cid in alias_cids:
                    alias_cids.remove(cid)
            if not alias_cids:
                keys_to_clean.append(k)
        for k in keys_to_clean:
            self._alias_map.pop(k, None)

        # Remove from the column index so stale IDs don't accumulate
        for col_cids in self._column_index.values():
            col_cids.difference_update(cids)

        # Invalidate caches
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
        self._search_index = None

        return removed
    def remove_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
    ) -> bool:
        """Remove a specific edge from the network.

        Returns True if the edge was found and removed.
        """
        src_id = self._resolve(source)
        tgt_id = self._resolve(target)
        if not src_id or not tgt_id:
            return False

        key = (src_id, tgt_id, relation)
        edge = self._edge_key_index.get(key)
        if edge is None:
            return False

        self._edges.remove(edge)
        del self._edge_key_index[key]
        if src_id in self._edge_index:
            self._edge_index[src_id] = [
                e for e in self._edge_index[src_id] if e is not edge
            ]
        if tgt_id in self._reverse_index:
            self._reverse_index[tgt_id] = [
                e for e in self._reverse_index[tgt_id] if e is not edge
            ]
        # Edge counts affect quality scores — invalidate the cache.
        self._quality_concept_ids_cache = None
        return True
    def clean_noise(self, min_confidence: float = 0.5, min_edges: int = 1) -> list[str]:
        """Remove noise concepts from the network.

        A concept is considered noise if:
        - Its confidence is below min_confidence (likely extracted
          as a fragment, not a real concept)
        - AND it has fewer than min_edges connections (isolated)

        This preserves low-confidence concepts that have been
        connected to other concepts (they may be real but uncertain).

        Returns the list of removed concept names.
        """
        noise: list[str] = []
        to_remove: list[str] = []

        for cid, concept in list(self._concepts.items()):
            # Use <= so concepts at exactly the threshold are also
            # candidates for removal. The learner creates concepts at
            # confidence=0.4, and with the old < check they survived
            # the min_confidence=0.4 prune forever — even with zero
            # edges. This caused thousands of orphan concepts to
            # accumulate permanently in the network.
            if concept.confidence > min_confidence:
                continue
            edges = self._edge_index.get(cid, []) + self._reverse_index.get(cid, [])
            if len(edges) < min_edges:
                to_remove.append(cid)

        # Use batch removal instead of calling remove_concept in a
        # loop. Each remove_concept call rebuilds the entire edge index
        # (O(E)), so a loop over N concepts is O(N×E). Batch removal
        # filters edges and rebuilds indices once — O(E + N).
        if to_remove:
            self.remove_concepts_batch(set(to_remove))
            # Return the cids that were actually removed (a concurrent
            # thread may have removed some between the scan and here).
            noise = [cid for cid in to_remove if cid not in self._concepts]

        return noise

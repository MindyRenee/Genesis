"""Concept network tests.

Core network, holographic graph, pruning, archive,
quality filter, relation extraction, orphan fixes, name extraction.
"""

import gzip
import json
import logging
import math
import os
import random
import tempfile

import numpy as np
import pytest

# scripts/ is not a package — add it to sys.path
from prune_dead_concepts import (
    _build_edge_index,
    _classify_concepts,
    _has_only_weak_bridges,
    _is_dead,
    prune_state,
)

from genesis_cognitive.concepts import (
    QUALITY_THRESHOLD,
    Concept,
    ConceptCategory,
    ConceptModality,
    ConceptNetwork,
    HolographicGraph,
    RelationType,
    circular_convolve,
    circular_correlate,
    is_world_concept,
    open_archive,
)
from genesis_cognitive.memory import SemanticMemory
from genesis_cognitive.memory.semantic import _relation_to_edge

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_concept_network.py
# ======================================================================


def _c(net: ConceptNetwork, name: str) -> Concept:
    """Get a concept, asserting it exists (for test convenience)."""
    c = net.get_concept(name)
    assert c is not None, f"concept '{name}' not found"
    return c


# ═══════════════════════════════════════════════════════════════════
# ConceptNetwork — initialization
# ═══════════════════════════════════════════════════════════════════


def test_network_quality_properties() -> None:
    """Mean edge weight, concept confidence, and density are exposed."""
    from genesis_cognitive.concepts import (
        ConceptNetwork,
        RelationType,
    )

    network = ConceptNetwork()
    assert network.mean_edge_weight == 0.0
    assert network.mean_concept_confidence == 0.0
    assert network.network_density == 0.0

    network.add_concept("a", confidence=0.9)
    network.add_concept("b", confidence=0.5)
    network.add_concept("c", confidence=0.4)
    network.add_edge("a", "b", RelationType.RELATED_TO, weight=0.8)

    assert network.mean_edge_weight == 0.8
    assert math.isclose(network.mean_concept_confidence, 0.6, rel_tol=1e-3)
    assert math.isclose(network.network_density, 1 / 6, rel_tol=1e-3)


def test_network_quality_decreases_with_low_confidence_edges() -> None:
    """Mean edge weight is non-monotonic: low-weight edges reduce it."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.RELATED_TO, weight=0.9)
    assert network.mean_edge_weight == 0.9

    network.add_edge("b", "c", RelationType.RELATED_TO, weight=0.1)
    assert network.mean_edge_weight == 0.5


# ═══════════════════════════════════════════════════════════════════
# add_concept
# ═══════════════════════════════════════════════════════════════════


def test_add_concept_normalizes_id() -> None:
    """add_concept normalizes the concept name."""
    network = ConceptNetwork()
    concept = network.add_concept("  Dogs  ")
    assert concept.id == "dogs"


def test_add_concept_existing_returns_same() -> None:
    """add_concept returns existing concept for duplicates."""
    network = ConceptNetwork()
    c1 = network.add_concept("dogs")
    c2 = network.add_concept("dogs")
    assert c1 is c2
    assert network.size == 1


def test_add_concept_boosts_activation_on_readd() -> None:
    """Re-adding a concept boosts its activation."""
    network = ConceptNetwork()
    c1 = network.add_concept("dogs")
    initial_activation = c1.activation
    c2 = network.add_concept("dogs")
    assert c2.activation > initial_activation


def test_add_concept_empty_def_merges_with_existing_def() -> None:
    """Re-adding a concept with no definition merges, doesn't create #2.

    This is the re-learning bug: a code learner re-analyzes a file and
    this pass extracts no definition for a concept that already has
    one. The empty definition should merge into the existing concept,
    not create a disambiguated ``#2`` duplicate.
    """
    network = ConceptNetwork()
    c1 = network.add_concept(
        "python:mind._run_persistence_test",
        properties={"definition": "related to python:self._category_stats.get"},
    )
    # Re-add with no definition — should merge, not create #2
    c2 = network.add_concept("python:mind._run_persistence_test")
    assert c1 is c2
    assert network.size == 1
    # Existing definition preserved
    assert c1.properties.get("definition") == "related to python:self._category_stats.get"


def test_add_concept_with_def_merges_into_empty_existing() -> None:
    """Adding a definition to a concept that had none merges, no #2."""
    network = ConceptNetwork()
    c1 = network.add_concept("python")
    # Later learning provides a definition
    c2 = network.add_concept(
        "python",
        properties={"definition": "a high-level programming language"},
    )
    assert c1 is c2
    assert network.size == 1
    assert c1.properties.get("definition") == "a high-level programming language"


def test_add_concept_different_def_creates_new_sense() -> None:
    """Genuinely different definitions still create a disambiguated sense."""
    network = ConceptNetwork()
    c1 = network.add_concept(
        "orange",
        properties={"definition": "a citrus fruit"},
    )
    c2 = network.add_concept(
        "orange",
        properties={"definition": "a color between red and yellow"},
    )
    assert c1 is not c2
    assert c2.id == "orange#2"
    assert network.size == 2


def test_add_edge_preserves_disambiguated_concept() -> None:
    """Test add edge preserves disambiguated concept."""
    network = ConceptNetwork()
    network.add_concept("orange", properties={"definition": "a citrus fruit"})
    sense = network.add_concept(
        "orange", properties={"definition": "a color between red and yellow"},
    )
    network.add_edge(sense.id, "color", RelationType.IS_A, 0.8)
    assert network.get_concept(sense.id) is sense
    assert sense.properties["definition"] == "a color between red and yellow"
    assert sense.id in network._column_index["conversation"]
    assert network.size == 3


# ═══════════════════════════════════════════════════════════════════
# get_concept
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "query,expected_id",
    [
        ("dogs", "dogs"),       # exact match
        ("unknown", None),      # unknown returns None
        ("DOGS", "dogs"),       # case-insensitive
    ],
)
def test_get_concept(query, expected_id) -> None:
    """get_concept returns the concept by name, None for unknown, case-insensitive."""
    network = ConceptNetwork()
    network.add_concept("dogs")
    concept = network.get_concept(query)
    if expected_id is None:
        assert concept is None
    else:
        assert concept is not None
        assert concept.id == expected_id


def test_get_concept_by_alias() -> None:
    """get_concept resolves aliases."""
    network = ConceptNetwork()
    network.add_concept("dogs", aliases={"canine"})
    concept = network.get_concept("canine")
    assert concept is not None
    assert concept.id == "dogs"


# ═══════════════════════════════════════════════════════════════════
# add_edge
# ═══════════════════════════════════════════════════════════════════


def test_add_edge() -> None:
    """add_edge creates a relationship between concepts."""
    network = ConceptNetwork()
    network.add_concept("dogs")
    network.add_concept("mammals")
    edge = network.add_edge("dogs", "mammals", RelationType.IS_A)
    assert edge is not None
    assert edge.source == "dogs"
    assert edge.target == "mammals"
    assert network.edge_count == 1


def test_add_edge_auto_creates_concepts() -> None:
    """add_edge creates missing concepts automatically."""
    network = ConceptNetwork()
    network.add_edge("dogs", "mammals", RelationType.IS_A)
    assert network.size == 2
    assert network.get_concept("dogs") is not None
    assert network.get_concept("mammals") is not None


def test_add_edge_duplicate_increases_weight() -> None:
    """Adding the same edge again increases its weight."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    e1 = network.add_edge("a", "b", RelationType.RELATED_TO, weight=0.5)
    e2 = network.add_edge("a", "b", RelationType.RELATED_TO)
    assert e2 is not None
    assert e1 is e2
    assert e2.weight > 0.5


# ═══════════════════════════════════════════════════════════════════
# get_edges
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "direction,concept,expected_count,check_field,check_value",
    [
        ("out", "a", 1, "source", "a"),        # outgoing edges
        ("in", "b", 1, "target", "b"),         # incoming edges
        ("both", "a", 2, None, None),          # both directions
        ("out", "unknown", 0, None, None),     # unknown concept
    ],
)
def test_get_edges(direction, concept, expected_count, check_field, check_value) -> None:
    """get_edges returns edges in the requested direction; empty for unknown."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.RELATED_TO)
    network.add_edge("c", "a", RelationType.RELATED_TO)
    edges = network.get_edges(concept, direction)
    assert len(edges) == expected_count
    if check_field is not None:
        assert getattr(edges[0], check_field) == check_value


# ═══════════════════════════════════════════════════════════════════
# get_neighbors
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "concept,relation,expected_count,expected_neighbor",
    [
        ("a", None, 2, "b"),                       # all neighbors
        ("a", RelationType.IS_A, 1, "b"),          # filtered by relation
        ("unknown", None, 0, None),                # unknown concept
    ],
)
def test_get_neighbors(concept, relation, expected_count, expected_neighbor) -> None:
    """get_neighbors returns neighbors, optionally filtered by relation; empty for unknown."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.IS_A, weight=0.7)
    network.add_edge("a", "c", RelationType.RELATED_TO, weight=0.5)
    neighbors = network.get_neighbors(concept, relation=relation)
    assert len(neighbors) == expected_count
    if expected_neighbor is not None:
        assert expected_neighbor in [n[0] for n in neighbors]


@pytest.mark.parametrize(
    "concept,expected",
    [
        ("a", True),          # outgoing edge
        ("b", True),          # incoming edge only
        ("c", False),         # no edges
        ("unknown", False),   # unknown concept
    ],
)
def test_has_neighbors(concept, expected) -> None:
    """has_neighbors returns True for concepts with any edges, False otherwise."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.RELATED_TO, weight=0.7)
    assert network.has_neighbors(concept) is expected


# ═══════════════════════════════════════════════════════════════════
# find_concepts_by_origin
# ═══════════════════════════════════════════════════════════════════


def test_find_concepts_by_origin() -> None:
    """find_concepts_by_origin filters by origin."""
    network = ConceptNetwork()
    network.add_concept("a", origin="conversation")
    network.add_concept("b", origin="learned")
    result = network.find_concepts_by_origin("conversation")
    assert "a" in result
    assert "b" not in result


def test_find_concepts_by_origin_limit() -> None:
    """find_concepts_by_origin respects the limit."""
    network = ConceptNetwork()
    for i in range(10):
        network.add_concept(f"concept_{i}", origin="conversation")
    result = network.find_concepts_by_origin("conversation", limit=5)
    assert len(result) == 5


# ═══════════════════════════════════════════════════════════════════
# search_concepts
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "query,limit,expected_contains,expected_len",
    [
        ("dogs", None, ["dogs"], None),          # exact match
        ("dog", None, ["dog_training"], None),   # substring match
        ("xyz", None, [], 0),                    # no match
        ("dog", 2, None, 2),                     # limit
    ],
)
def test_search_concepts(query, limit, expected_contains, expected_len) -> None:
    """search_concepts finds exact/substring matches, respects limit, empty for no match."""
    network = ConceptNetwork()
    network.add_concept("dogs")
    network.add_concept("cats")
    network.add_concept("dog_training")
    network.add_concept("dog_a")
    network.add_concept("dog_b")
    network.add_concept("dog_c")
    if limit:
        results = network.search_concepts(query, limit=limit)
    else:
        results = network.search_concepts(query)
    if expected_contains is not None:
        for name in expected_contains:
            assert name in results
    if expected_len is not None:
        assert len(results) == expected_len


# ═══════════════════════════════════════════════════════════════════
# find_path
# ═══════════════════════════════════════════════════════════════════


def test_find_path_direct() -> None:
    """find_path finds a direct path."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_edge("a", "b", RelationType.RELATED_TO)
    path = network.find_path("a", "b")
    assert path is not None
    assert len(path) == 1


def test_find_path_multi_hop() -> None:
    """find_path finds a multi-hop path."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.RELATED_TO)
    network.add_edge("b", "c", RelationType.RELATED_TO)
    path = network.find_path("a", "c")
    assert path is not None
    assert len(path) == 2


def test_find_path_no_path() -> None:
    """find_path returns None when no path exists."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    # No edge between a and b
    path = network.find_path("a", "b")
    assert path is None


def test_find_path_same_concept() -> None:
    """find_path returns empty list for same source and target."""
    network = ConceptNetwork()
    network.add_concept("a")
    path = network.find_path("a", "a")
    assert path == []


def test_find_path_unknown_concept() -> None:
    """find_path returns None for unknown concepts."""
    network = ConceptNetwork()
    path = network.find_path("unknown1", "unknown2")
    assert path is None


# ═══════════════════════════════════════════════════════════════════
# find_path — undirected (bidirectional) traversal
# ═══════════════════════════════════════════════════════════════════


def test_find_path_reverse_direction() -> None:
    """find_path follows edges in both directions (undirected BFS).

    If A → B exists, find_path(B, A) should find the path by
    traversing the edge backward, returning it oriented B → A.
    """
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_edge("a", "b", RelationType.RELATED_TO)
    path = network.find_path("b", "a")
    assert path is not None
    assert len(path) == 1
    # Edge should be oriented along traversal: source=b, target=a
    assert path[0].source == "b"
    assert path[0].target == "a"


def test_find_path_reverse_multi_hop() -> None:
    """find_path traverses backward through multi-hop paths."""
    network = ConceptNetwork()
    for c in ("a", "b", "c"):
        network.add_concept(c)
    network.add_edge("a", "b", RelationType.RELATED_TO)
    network.add_edge("b", "c", RelationType.RELATED_TO)
    # Path from c to a: traverse c→b (backward on b→c), b→a (backward on a→b)
    path = network.find_path("c", "a")
    assert path is not None
    assert len(path) == 2
    assert path[0].source == "c"
    assert path[0].target == "b"
    assert path[1].source == "b"
    assert path[1].target == "a"


def test_find_path_asymmetric_relation_preserved() -> None:
    """find_path preserves the relation type even when traversing backward.

    For 'alice creates genesis', find_path('genesis', 'alice') should
    return an edge with relation=CREATES, oriented genesis→alice.
    """
    network = ConceptNetwork()
    network.add_concept("alice")
    network.add_concept("genesis")
    network.add_edge("alice", "genesis", RelationType.CREATES)
    path = network.find_path("genesis", "alice")
    assert path is not None
    assert len(path) == 1
    assert path[0].relation == RelationType.CREATES
    assert path[0].source == "genesis"
    assert path[0].target == "alice"


def test_find_path_mixed_direction() -> None:
    """find_path handles paths that mix forward and backward edge traversal."""
    network = ConceptNetwork()
    for c in ("a", "b", "c", "d"):
        network.add_concept(c)
    # a→b (forward), c→b (backward from b), c→d (forward from c)
    network.add_edge("a", "b", RelationType.RELATED_TO)
    network.add_edge("c", "b", RelationType.RELATED_TO)
    network.add_edge("c", "d", RelationType.RELATED_TO)
    # Path a→d: a→b (forward), b→c (backward on c→b), c→d (forward)
    path = network.find_path("a", "d")
    assert path is not None
    assert len(path) == 3
    assert path[0].source == "a"
    assert path[-1].target == "d"


# ═══════════════════════════════════════════════════════════════════
# search_concepts — trigram index
# ═══════════════════════════════════════════════════════════════════


def test_search_concepts_index_correctness() -> None:
    """search_concepts returns same results with trigram index as without."""
    network = ConceptNetwork()
    network.add_concept("dog_training")
    network.add_concept("cat")
    network.add_concept("dogs")
    results = network.search_concepts("dog")
    assert "dogs" in results
    assert "dog_training" in results


def test_search_concepts_index_alias_match() -> None:
    """search_concepts finds concepts via alias through the trigram index."""
    network = ConceptNetwork()
    network.add_concept("feline", aliases={"cat"})
    results = network.search_concepts("cat")
    assert "feline" in results


def test_search_concepts_index_invalidated_on_add() -> None:
    """search_concepts index is invalidated when new concepts are added."""
    network = ConceptNetwork()
    network.add_concept("apple")
    # Build the index
    results = network.search_concepts("apple")
    assert "apple" in results
    # Add a new concept and search again — index should be rebuilt
    network.add_concept("applesauce")
    results = network.search_concepts("apple")
    assert "applesauce" in results


def test_search_concepts_index_invalidated_on_remove() -> None:
    """search_concepts index is invalidated when concepts are removed."""
    network = ConceptNetwork()
    network.add_concept("apple")
    network.add_concept("banana")
    # Build the index
    network.search_concepts("apple")
    # Remove a concept and search again — index should be rebuilt
    network.remove_concept("apple")
    results = network.search_concepts("apple")
    assert "apple" not in results


def test_search_concepts_short_query_fallback() -> None:
    """search_concepts falls back to full scan for queries < 3 chars."""
    network = ConceptNetwork()
    network.add_concept("ab")
    network.add_concept("abc")
    network.add_concept("xyz")
    # "ab" is 2 chars — no trigrams, should fall back to scanning
    results = network.search_concepts("ab")
    assert "ab" in results
    assert "abc" in results


# ═══════════════════════════════════════════════════════════════════
# spread_activation
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "seeds,depth,expected_in,expected_not_in",
    [
        (["a"], None, ["a", "b"], []),           # basic spread
        (["unknown"], None, [], []),             # unknown concept
        (["a"], 1, ["a", "b"], ["c"]),           # depth limit
    ],
)
def test_spread_activation(seeds, depth, expected_in, expected_not_in) -> None:
    """spread_activation activates neighbors, respects depth, ignores unknown."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    network.add_concept("c")
    network.add_edge("a", "b", RelationType.RELATED_TO)
    network.add_edge("b", "c", RelationType.RELATED_TO)
    if depth is not None:
        activated = network.spread_activation(seeds, amount=0.5, depth=depth)
    else:
        activated = network.spread_activation(seeds, amount=0.5)
    for name in expected_in:
        assert name in activated
    for name in expected_not_in:
        assert name not in activated


# ═══════════════════════════════════════════════════════════════════
# Column methods
# ═══════════════════════════════════════════════════════════════════


def test_get_column_concepts() -> None:
    """get_column_concepts returns concepts in a column."""
    network = ConceptNetwork()
    network.add_concept("a", origin="conversation")
    network.add_concept("b", origin="conversation")
    col = network.get_column("a")
    concepts = network.get_column_concepts(col)
    assert "a" in concepts


def test_get_column_activation() -> None:
    """get_column_activation returns mean activation."""
    network = ConceptNetwork()
    network.add_concept("a", origin="conversation")
    concept = network.get_concept("a")
    assert concept is not None
    concept.activation = 0.5
    col = network.get_column("a")
    activation = network.get_column_activation(col)
    assert activation > 0.0


# ═══════════════════════════════════════════════════════════════════
# Category methods
# ═══════════════════════════════════════════════════════════════════


def test_concept_categorized_at_creation_time() -> None:
    """Concepts are categorized at creation time, not just via categorize_all.

    This is a regression test for the issue where detect_category and
    detect_modality were never called in production — only in tests via
    categorize_all/classify_modalities. Now _create_new_concept calls
    them at creation time so spreading-activation multipliers, modality-
    specific decay, and within-modality priming work immediately.
    """
    network = ConceptNetwork()

    # "dog" is in _LIVING_KEYWORDS → should be LIVING at creation
    network.add_concept("dog")
    dog = network.get_concept("dog")
    assert dog is not None
    assert dog.category == ConceptCategory.LIVING
    assert dog.is_animacy_detected is True

    # "cognition" is in _ABSTRACT_KEYWORDS → should be ABSTRACT
    network.add_concept("cognition")
    cons = network.get_concept("cognition")
    assert cons is not None
    assert cons.category == ConceptCategory.ABSTRACT

    # "red" is in _VISUAL_KEYWORDS → should be VISUAL modality
    network.add_concept("red")
    red = network.get_concept("red")
    assert red is not None
    assert red.modality == ConceptModality.VISUAL

    # "run" is in _MOTOR_KEYWORDS → should be MOTOR modality
    network.add_concept("run")
    run = network.get_concept("run")
    assert run is not None
    assert run.modality == ConceptModality.MOTOR

    # Neutral name → stays UNKNOWN
    network.add_concept("alpha")
    alpha = network.get_concept("alpha")
    assert alpha is not None
    assert alpha.category == ConceptCategory.UNKNOWN
    assert alpha.modality == ConceptModality.UNKNOWN


def test_categorize_all_catches_concepts_with_later_definitions() -> None:
    """categorize_all catches concepts that received definitions after creation.

    In production, many code paths (self_directed_learning, autonomous_learner,
    curriculum, introspection) set concept.properties["definition"] directly,
    bypassing add_concept. These concepts start UNKNOWN and are caught by
    categorize_all during N3 sleep or at startup.
    """
    network = ConceptNetwork()

    # Create a concept with no definition — stays UNKNOWN
    network.add_concept("alpha")
    alpha = network.get_concept("alpha")
    assert alpha is not None
    assert alpha.category == ConceptCategory.UNKNOWN

    # Simulate a production code path that directly sets a definition
    alpha.properties["definition"] = "a wild animal"

    # categorize_all should now catch it
    result = network.categorize_all()
    assert "living" in result
    alpha = network.get_concept("alpha")
    assert alpha is not None
    assert alpha.category == ConceptCategory.LIVING


def test_get_by_category() -> None:
    """get_by_category returns concepts in a category."""
    network = ConceptNetwork()
    network.add_concept("dog")
    concept = network.get_concept("dog")
    assert concept is not None
    concept.category = ConceptCategory.LIVING
    result = network.get_by_category(ConceptCategory.LIVING)
    # get_by_category returns Concept objects
    assert any(c.id == "dog" for c in result)


# ═══════════════════════════════════════════════════════════════════
# Modality methods
# ═══════════════════════════════════════════════════════════════════


def test_get_by_modality() -> None:
    """get_by_modality returns concepts in a modality."""
    network = ConceptNetwork()
    network.add_concept("dog")
    concept = network.get_concept("dog")
    assert concept is not None
    concept.modality = ConceptModality.VISUAL
    result = network.get_by_modality(ConceptModality.VISUAL)
    # get_by_modality returns Concept objects
    assert any(c.id == "dog" for c in result)


# ═══════════════════════════════════════════════════════════════════
# ground_activation
# ═══════════════════════════════════════════════════════════════════


def test_ground_activation() -> None:
    """ground_activation boosts activation of all concepts."""
    network = ConceptNetwork()
    network.add_concept("a")
    network.add_concept("b")
    concept_a = network.get_concept("a")
    assert concept_a is not None
    initial_a = concept_a.activation
    network.ground_activation(boost=0.1)
    concept_a2 = network.get_concept("a")
    assert concept_a2 is not None
    assert concept_a2.activation >= initial_a


# ═══════════════════════════════════════════════════════════════════
# create_bridge
# ═══════════════════════════════════════════════════════════════════


def test_create_bridge() -> None:
    """create_bridge creates an edge between concepts in different columns."""
    network = ConceptNetwork()
    # Use different origins so they land in different columns
    network.add_concept("a", origin="conversation")
    network.add_concept("b", origin="learned")
    edge = network.create_bridge("a", "b", weight=0.3)
    # Bridge is only created if columns differ
    if edge is not None:
        assert edge.source == "a"
        assert edge.target == "b"


def test_create_bridge_same_column_returns_none() -> None:
    """create_bridge returns None for concepts in the same column."""
    network = ConceptNetwork()
    network.add_concept("a", origin="conversation")
    network.add_concept("b", origin="conversation")
    edge = network.create_bridge("a", "b", weight=0.3)
    assert edge is None


def test_create_bridge_unknown_concept() -> None:
    """create_bridge returns None for unknown concepts."""
    network = ConceptNetwork()
    edge = network.create_bridge("unknown1", "unknown2")
    # create_bridge auto-creates concepts via add_edge
    # so it should succeed
    if edge is not None:
        assert edge.source == "unknown1"


# ═══════════════════════════════════════════════════════════════════
# Function-word filtering in parse_relationships
# ═══════════════════════════════════════════════════════════════════


def test_parse_relationships_skips_function_word_subjects() -> None:
    """Function words (pronouns, conjunctions, question words) should
    never be extracted as subjects of semantic relations.

    Sentences like "What is cognition?" or "It is recommended that..."
    match the generic 'X is a Y' pattern, but the subject is a grammatical
    function word, not a concept. Without filtering, this creates garbage
    edges like 'what is_a cognition'.
    """
    network = ConceptNetwork()

    # "What is cognition?" — should NOT produce what is_a cognition
    results = network.parse_relationships("What is cognition?")
    for subj, rel, tgt in results:
        assert subj != "what", f"'what' should not be a relation subject, got {subj} {rel} {tgt}"

    # "It is recommended that you write tests" — should NOT produce it is_a recommended...
    results = network.parse_relationships("It is recommended that you write tests.")
    for subj, rel, tgt in results:
        assert subj != "it", f"'it' should not be a relation subject, got {subj} {rel} {tgt}"

    # "If is a possibility" — should NOT produce if is_a possibility
    results = network.parse_relationships("If is a possibility.")
    for subj, rel, tgt in results:
        assert subj != "if", f"'if' should not be a relation subject, got {subj} {rel} {tgt}"


def test_parse_relationships_still_extracts_real_concepts() -> None:
    """Real concept subjects should still be extracted after the
    function-word filter is in place."""
    network = ConceptNetwork()

    # "Water is a liquid" — SHOULD produce water is_a liquid
    results = network.parse_relationships("Water is a liquid.")
    assert any(
        subj == "water" and rel == RelationType.IS_A and tgt == "liquid"
        for subj, rel, tgt in results
    ), f"Should extract 'water is_a liquid', got {results}"

    # "Fire causes smoke" — SHOULD produce fire causes smoke
    results = network.parse_relationships("Fire causes smoke.")
    assert any(
        subj == "fire" and rel == RelationType.CAUSES and tgt == "smoke"
        for subj, rel, tgt in results
    ), f"Should extract 'fire causes smoke', got {results}"


# ═══════════════════════════════════════════════════════════════════
# Cortical tick — continuous dynamical evolution
# ═══════════════════════════════════════════════════════════════════


def test_cortical_tick_empty_network() -> None:
    """An empty network should return empty dict and not crash."""
    net = ConceptNetwork()
    result = net.cortical_tick()
    assert result == {}


def test_cortical_tick_decay() -> None:
    """Activations should decay over ticks without spreading input."""
    net = ConceptNetwork()
    net.add_concept("alpha", confidence=0.8)
    net.add_concept("beta", confidence=0.8)
    # No edges — no spreading, only decay
    _c(net, "alpha").activation = 0.9
    _c(net, "beta").activation = 0.5

    initial_alpha = _c(net, "alpha").activation
    net.cortical_tick(arousal=0.3, gaba=0.5, ach=0.1, serotonin=0.5, noise=0.0)
    after_alpha = _c(net, "alpha").activation
    assert after_alpha < initial_alpha, "activation should decay"


def test_cortical_tick_spreading() -> None:
    """Activation should spread from active concepts to neighbors."""
    net = ConceptNetwork()
    net.add_concept("fire", confidence=0.8)
    net.add_concept("heat", confidence=0.8)
    net.add_concept("smoke", confidence=0.8)
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.9)
    net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.7)

    # Fire is active, heat and smoke are dormant
    _c(net, "fire").activation = 0.9
    _c(net, "heat").activation = 0.0
    _c(net, "smoke").activation = 0.0

    # Run several ticks to let activation spread
    for _ in range(5):
        net.cortical_tick(arousal=0.7, gaba=0.1, ach=0.2, serotonin=0.5, noise=0.0)

    heat = _c(net, "heat").activation
    smoke = _c(net, "smoke").activation
    assert heat > 0.01, f"heat should receive activation from fire: {heat}"
    assert smoke > 0.01, f"smoke should receive activation from fire: {smoke}"


def test_cortical_tick_gaba_increases_decay() -> None:
    """High GABA should cause faster decay than low GABA."""
    net1 = ConceptNetwork()
    net2 = ConceptNetwork()
    for net in [net1, net2]:
        net.add_concept("thought", confidence=0.8)
        _c(net, "thought").activation = 0.8

    # Run 10 ticks with low GABA (alert)
    for _ in range(10):
        net1.cortical_tick(arousal=0.8, gaba=0.1, ach=0.2, serotonin=0.5, noise=0.0)
    # Run 10 ticks with high GABA (calm)
    for _ in range(10):
        net2.cortical_tick(arousal=0.2, gaba=0.7, ach=0.2, serotonin=0.5, noise=0.0)

    alert_activation = _c(net1, "thought").activation
    calm_activation = _c(net2, "thought").activation
    assert alert_activation > calm_activation, (
        f"alert ({alert_activation}) should retain more activation than calm ({calm_activation})"
    )


def test_cortical_tick_noise_perturbs() -> None:
    """Noise should cause small random perturbations."""
    random.seed(42)
    net = ConceptNetwork()
    net.add_concept("idea", confidence=0.8)
    _c(net, "idea").activation = 0.5

    # Run with noise — activation should vary
    activations = []
    for _ in range(20):
        net.cortical_tick(arousal=0.5, gaba=0.3, ach=0.2, serotonin=0.1, noise=0.05)
        activations.append(_c(net, "idea").activation)

    # With noise, not all values should be identical
    assert len({round(a, 6) for a in activations}) > 1, (
        "noise should cause variation in activation"
    )


def test_cortical_tick_clamp_range() -> None:
    """Activations should stay in [0, 1] even with extreme input."""
    net = ConceptNetwork()
    net.add_concept("excited", confidence=0.8)
    net.add_concept("target", confidence=0.8)
    net.add_edge("excited", "target", RelationType.RELATED_TO, weight=1.0)

    _c(net, "excited").activation = 1.0
    _c(net, "target").activation = 1.0

    for _ in range(50):
        net.cortical_tick(arousal=1.0, gaba=0.0, ach=0.0, serotonin=0.5, noise=0.1)

    for cid in ["excited", "target"]:
        a = _c(net, cid).activation
        assert 0.0 <= a <= 1.0, f"{cid} activation out of range: {a}"


def test_cortical_tick_ach_focus() -> None:
    """High ACh should sharpen winner-take-all dynamics."""
    net = ConceptNetwork()
    net.add_concept("focus", confidence=0.8)
    net.add_concept("distraction1", confidence=0.8)
    net.add_concept("distraction2", confidence=0.8)

    # All start at similar activation
    for cid in ["focus", "distraction1", "distraction2"]:
        _c(net, cid).activation = 0.5
    # Make "focus" slightly more active
    _c(net, "focus").activation = 0.6

    # Run with high ACh (focused attention)
    for _ in range(10):
        net.cortical_tick(arousal=0.6, gaba=0.2, ach=0.9, serotonin=0.5, noise=0.0)

    focus_a = _c(net, "focus").activation
    d1_a = _c(net, "distraction1").activation
    d2_a = _c(net, "distraction2").activation
    # With high ACh, the top concept should be boosted relative to others
    assert focus_a > d1_a or focus_a > d2_a, (
        f"focus ({focus_a}) should be boosted above distractions ({d1_a}, {d2_a})"
    )


def test_cortical_tick_returns_active_concepts() -> None:
    """The return dict should contain concepts above threshold."""
    net = ConceptNetwork()
    net.add_concept("bright", confidence=0.8)
    net.add_concept("dim", confidence=0.8)
    _c(net, "bright").activation = 0.8
    _c(net, "dim").activation = 0.02

    result = net.cortical_tick(arousal=0.5, gaba=0.3, ach=0.2, serotonin=0.5, noise=0.0)
    # "bright" should be in the result, "dim" should not
    assert "bright" in result
    assert "dim" not in result


def test_cortical_tick_stability() -> None:
    """The network should be stable over many ticks (no NaN/inf)."""
    net = ConceptNetwork()
    for name in ["alpha", "beta", "gamma", "delta"]:
        net.add_concept(name, confidence=0.7)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, weight=0.5)
    net.add_edge("beta", "gamma", RelationType.RELATED_TO, weight=0.5)
    net.add_edge("gamma", "delta", RelationType.RELATED_TO, weight=0.5)
    net.add_edge("delta", "alpha", RelationType.RELATED_TO, weight=0.5)

    # Seed with some activation
    _c(net, "alpha").activation = 0.9

    for _ in range(200):
        net.cortical_tick(arousal=0.6, gaba=0.3, ach=0.3, serotonin=0.5, noise=0.02)

    for cid in ["alpha", "beta", "gamma", "delta"]:
        a = _c(net, cid).activation
        assert math.isfinite(a), f"{cid} activation is not finite: {a}"
        assert 0.0 <= a <= 1.0, f"{cid} activation out of range: {a}"


# ═══════════════════════════════════════════════════════════════════
# Utterance / relation template seed & retrieval
# ═══════════════════════════════════════════════════════════════════


def test_utterance_words_provenance() -> None:
    """Utterance word retrieval exposes seed vs learned provenance."""
    net = ConceptNetwork()
    from genesis_cognitive.language import Vocabulary
    Vocabulary(seed=42, network=net)  # seeds utterance words into net

    items = net.find_utterance_words_with_provenance("greeting_word")
    assert items, "expected seeded greeting words"
    for item in items:
        assert item.is_seed, f"seeded utterance should be marked as seed: {item}"


# ======================================================================
# From tests/test_holographic_graph.py
# ======================================================================

class TestVectorOperations:
    """Tests for vector operations."""
    def test_circular_convolve_unit_vectors(self):
        """Convolution of unit vectors should produce a unit vector."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(512).astype(np.float32)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(512).astype(np.float32)
        b /= np.linalg.norm(b)
        c = circular_convolve(a, b)
        assert c.shape == (512,)
        assert np.linalg.norm(c) == pytest.approx(1.0, abs=0.01)

    def test_bind_unbind_roundtrip(self):
        """Unbinding should approximately recover the bound vector."""
        dim = 2048
        rng = np.random.default_rng(42)
        a = rng.standard_normal(dim).astype(np.float32)
        a /= np.linalg.norm(a)
        b = rng.standard_normal(dim).astype(np.float32)
        b /= np.linalg.norm(b)

        bound = circular_convolve(a, b)
        recovered = circular_correlate(bound, b)

        # The recovered vector should be correlated with a.
        # HRR unbinding is approximate — the similarity depends on
        # dimensionality and normalization. With dim=2048 and unit
        # vectors, we expect similarity > 0.5.
        sim = float(np.dot(recovered, a) / (np.linalg.norm(recovered) * np.linalg.norm(a)))
        assert sim > 0.5, f"Recovery similarity {sim} should be > 0.5"


class TestHolographicGraph:
    """Tests for holographic graph."""
    def test_add_and_query(self):
        """Adding an association and querying should return the target."""
        hg = HolographicGraph(dim=2048, n_buckets=4)
        hg.add("dog", "related_to", "cat", weight=1.0)
        hg.add("dog", "related_to", "bird", weight=0.5)

        results = hg.query("dog", "related_to", top_k=5)
        assert len(results) > 0
        # "cat" should be in the results (it had higher weight)
        targets = [r[0] for r in results]
        assert "cat" in targets

    def test_query_nonexistent_source(self):
        """Querying a nonexistent source should return empty."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("dog", "related_to", "cat")
        results = hg.query("nonexistent", "related_to")
        assert results == []

    def test_query_nonexistent_relation(self):
        """Querying a nonexistent relation should return empty."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("dog", "related_to", "cat")
        results = hg.query("dog", "is_a")
        assert results == []

    def test_query_excludes_source(self):
        """Query should not return the source itself."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("dog", "related_to", "cat")
        results = hg.query("dog", "related_to", top_k=10)
        targets = [r[0] for r in results]
        assert "dog" not in targets

    def test_multiple_relations(self):
        """Multiple relation types should be stored independently."""
        hg = HolographicGraph(dim=2048, n_buckets=4)
        hg.add("dog", "is_a", "animal")
        hg.add("dog", "related_to", "cat")

        is_a_results = hg.query("dog", "is_a", top_k=5)
        related_results = hg.query("dog", "related_to", top_k=5)

        # Both should have results
        assert len(is_a_results) > 0
        assert len(related_results) > 0

        # "animal" should be in is_a results, "cat" in related_to
        is_a_targets = [r[0] for r in is_a_results]
        related_targets = [r[0] for r in related_results]
        assert "animal" in is_a_targets
        assert "cat" in related_targets

    def test_query_all_relations(self):
        """query_all_relations should return all relation types."""
        hg = HolographicGraph(dim=2048, n_buckets=4)
        hg.add("dog", "is_a", "animal")
        hg.add("dog", "related_to", "cat")

        results = hg.query_all_relations("dog", top_k=5)
        assert "is_a" in results
        assert "related_to" in results

    def test_edge_count(self):
        """Edge count should track added associations."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        assert hg.edge_count == 0
        hg.add("a", "related_to", "b")
        assert hg.edge_count == 1
        hg.add("a", "related_to", "c")
        assert hg.edge_count == 2

    def test_clear(self):
        """Clear should reset all associations."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("a", "related_to", "b")
        hg.add("a", "related_to", "c")
        assert hg.edge_count == 2

        hg.clear()
        assert hg.edge_count == 0
        # After clearing, query scores should be ~0 (memory is zeroed)
        results = hg.query("a", "related_to", top_k=5)
        for _target, sim in results:
            assert abs(sim) < 0.01, f"Score {sim} should be ~0 after clear"

    def test_add_edges_bulk(self):
        """Bulk add should work correctly."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        edges = [
            ("a", "related_to", "b", 1.0),
            ("a", "is_a", "c", 0.8),
            ("b", "related_to", "d", 0.5),
        ]
        count = hg.add_edges(edges)
        assert count == 3
        assert hg.edge_count == 3

    def test_rebuild_from_edges(self):
        """Rebuild should clear and re-add all edges."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("old", "related_to", "edge")
        assert hg.edge_count == 1

        new_edges = [
            ("a", "related_to", "b", 1.0),
            ("c", "is_a", "d", 0.8),
        ]
        count = hg.rebuild_from_edges(new_edges)
        assert count == 2
        assert hg.edge_count == 2

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve the graph."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("dog", "is_a", "animal")
        hg.add("dog", "related_to", "cat")
        hg.add("cat", "is_a", "animal")

        path = str(tmp_path / "hgraph_test.npz")
        hg.save(path)
        assert os.path.exists(path)

        hg2 = HolographicGraph(dim=512, n_buckets=4)
        loaded = hg2.load(path)
        assert loaded
        assert hg2.edge_count == 3
        assert hg2.n_concepts == 3  # dog, cat, animal

        # Query should still work
        results = hg2.query("dog", "is_a", top_k=5)
        assert len(results) > 0
        targets = [r[0] for r in results]
        assert "animal" in targets

    def test_load_nonexistent(self):
        """Loading a nonexistent file should return False."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        assert not hg.load("/nonexistent/path/file.npz")

    def test_storage_bytes(self):
        """Storage size should be reasonable for the dimensionality."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("a", "related_to", "b")
        size = hg.storage_bytes
        # Should be > 0 and reasonable
        assert size > 0
        # For dim=512, 1 relation, 2 concepts: ~512*4*4 + 2*512*4 + 512*4
        # = 8192 + 4096 + 2048 = ~14KB
        assert size < 100_000  # should be small

    def test_stats(self):
        """Stats should return expected fields."""
        hg = HolographicGraph(dim=512, n_buckets=4)
        hg.add("a", "related_to", "b")
        stats = hg.stats()
        assert "dim" in stats
        assert "n_buckets" in stats
        assert "edge_count" in stats
        assert "n_concepts" in stats
        assert "n_relations" in stats
        assert "storage_mb" in stats

    def test_deterministic_addresses(self):
        """Same concept ID should always get the same address vector."""
        hg1 = HolographicGraph(dim=512, n_buckets=4)
        hg2 = HolographicGraph(dim=512, n_buckets=4)

        hg1.add("dog", "related_to", "cat")
        hg2.add("dog", "related_to", "cat")

        addr1 = hg1._addresses["dog"]
        addr2 = hg2._addresses["dog"]
        np.testing.assert_array_equal(addr1, addr2)

    def test_many_associations(self):
        """Graph should handle many associations without crashing."""
        hg = HolographicGraph(dim=2048, n_buckets=16)
        # Add 100 associations
        for i in range(100):
            hg.add(f"concept_{i}", "related_to", f"concept_{(i+1) % 100}")

        assert hg.edge_count == 100
        # Query should still work
        results = hg.query("concept_0", "related_to", top_k=5)
        assert len(results) > 0


# ======================================================================
# From tests/test_concept_archive.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# Archive store tests
# ═══════════════════════════════════════════════════════════════════


def test_archive_create_and_count() -> None:
    """Archive starts empty and can count its contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        assert archive.count() == 0
        archive.close()


def test_archive_concept_roundtrip() -> None:
    """A concept can be archived and recalled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        concept_data = {
            "id": "neuroscience",
            "aliases": ["neurobiology"],
            "activation": 0.0,
            "confidence": 0.8,
            "origin": "learned",
            "columns": ["dictionary"],
            "created_at": 1700000000000,
            "review_count": 0,
            "last_reviewed": 0,
            "properties": {"definition": "The study of the nervous system."},
            "category": "abstract",
            "is_animacy_detected": False,
            "modality": "abstract",
            "is_semantic_hub": False,
        }
        archive.archive_concept("neuroscience", concept_data, {"neurobiology"})
        assert archive.count() == 1

        # Recall by ID
        recalled = archive.recall_concept("neuroscience")
        assert recalled is not None
        assert recalled["id"] == "neuroscience"
        assert recalled["confidence"] == 0.8
        assert archive.count() == 0  # removed after recall
        archive.close()


def test_archive_find_by_alias() -> None:
    """Concepts can be found by alias in the archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        concept_data = {
            "id": "ai",
            "aliases": ["artificial intelligence", "machine intelligence"],
            "activation": 0.0,
            "confidence": 0.9,
            "origin": "learned",
            "columns": [],
            "created_at": 0,
            "review_count": 0,
            "last_reviewed": 0,
            "properties": {},
            "category": "unknown",
            "is_animacy_detected": False,
            "modality": "unknown",
            "is_semantic_hub": False,
        }
        archive.archive_concept(
            "ai", concept_data,
            {"artificial intelligence", "machine intelligence"},
        )

        # Find by alias
        ids = archive.find_by_alias("artificial intelligence")
        assert ids == ["ai"]

        # Find by canonical name (auto-indexed)
        ids = archive.find_by_alias("ai")
        assert ids == ["ai"]

        # Case insensitive
        ids = archive.find_by_alias("Artificial Intelligence")
        assert ids == ["ai"]

        # Non-existent alias
        ids = archive.find_by_alias("nonexistent")
        assert ids == []
        archive.close()


def test_archive_batch_archive() -> None:
    """Multiple concepts can be archived in a batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        items: list[tuple[str, dict, set]] = []
        for i in range(10):
            items.append((
                f"concept_{i}",
                {
                    "id": f"concept_{i}",
                    "aliases": [],
                    "activation": 0.0,
                    "confidence": 0.5,
                    "origin": "learned",
                    "columns": [],
                    "created_at": 0,
                    "review_count": 0,
                    "last_reviewed": 0,
                    "properties": {},
                    "category": "unknown",
                    "is_animacy_detected": False,
                    "modality": "unknown",
                    "is_semantic_hub": False,
                },
                set(),
            ))
        count = archive.archive_concepts_batch(items)
        assert count == 10
        assert archive.count() == 10
        archive.close()


def test_archive_persistence() -> None:
    """Archive persists across close/reopen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        concept_data = {
            "id": "persistence",
            "aliases": [],
            "activation": 0.0,
            "confidence": 0.5,
            "origin": "learned",
            "columns": [],
            "created_at": 0,
            "review_count": 0,
            "last_reviewed": 0,
            "properties": {},
            "category": "unknown",
            "is_animacy_detected": False,
            "modality": "unknown",
            "is_semantic_hub": False,
        }
        archive.archive_concept("persistence", concept_data, set())
        archive.close()

        # Reopen
        archive2 = open_archive(tmpdir)
        assert archive2.count() == 1
        recalled = archive2.recall_concept("persistence")
        assert recalled is not None
        assert recalled["id"] == "persistence"
        archive2.close()


def test_archive_has_concept() -> None:
    """has_concept checks if a concept is in the archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = open_archive(tmpdir)
        concept_data = {
            "id": "test_concept",
            "aliases": [],
            "activation": 0.0,
            "confidence": 0.5,
            "origin": "learned",
            "columns": [],
            "created_at": 0,
            "review_count": 0,
            "last_reviewed": 0,
            "properties": {},
            "category": "unknown",
            "is_animacy_detected": False,
            "modality": "unknown",
            "is_semantic_hub": False,
        }
        archive.archive_concept("test_concept", concept_data, set())
        assert archive.has_concept("test_concept")
        assert not archive.has_concept("nonexistent")
        archive.close()


# ═══════════════════════════════════════════════════════════════════
# ConceptNetwork + Archive integration tests
# ═══════════════════════════════════════════════════════════════════


def _make_network_with_archive(tmpdir: str) -> ConceptNetwork:
    """Create a ConceptNetwork with an attached archive."""
    net = ConceptNetwork()
    net.attach_archive(open_archive(tmpdir))
    return net


def test_spill_dormant_concepts() -> None:
    """Dormant concepts are spilled to the archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Add active concept (high activation)
        net.add_concept("active_concept", confidence=0.8, origin="conversation")
        c = net.get_concept("active_concept")
        assert c is not None
        c.activation = 0.9

        # Add dormant concepts (low activation, learned origin)
        for i in range(20):
            net.add_concept(f"dormant_{i}", confidence=0.4, origin="learned")
            c = net.get_concept(f"dormant_{i}")
            assert c is not None
            c.activation = 0.0  # fully dormant

        assert net.size == 21

        # Spill with max_in_memory=5 — should spill 16 dormant concepts
        spilled = net.spill_dormant(activation_threshold=0.01, max_in_memory=5)
        assert spilled == 16
        assert net.size == 5  # working memory reduced
        assert net.archive_size == 16  # archive has the rest

        # Active concept should still be in working memory
        assert net.get_concept("active_concept") is not None
        net._archive.close()


def test_spill_protected_origins() -> None:
    """Protected origin concepts are never spilled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Add a protected concept with zero activation
        net.add_concept("identity_concept", confidence=0.5, origin="identity")
        c = net.get_concept("identity_concept")
        assert c is not None
        c.activation = 0.0

        # Add some non-protected dormant concepts to trigger spilling
        for i in range(20):
            net.add_concept(f"learned_{i}", confidence=0.4, origin="learned")
            c = net.get_concept(f"learned_{i}")
            assert c is not None
            c.activation = 0.0

        spilled = net.spill_dormant(activation_threshold=0.01, max_in_memory=5)
        assert spilled > 0

        # Protected concept should still be in working memory
        assert net.get_concept("identity_concept") is not None
        net._archive.close()


def test_spill_reviewed_concepts_kept() -> None:
    """Concepts with review_count > 0 are not spilled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Add a reviewed concept with zero activation
        net.add_concept("reviewed_concept", confidence=0.4, origin="learned")
        c = net.get_concept("reviewed_concept")
        assert c is not None
        c.activation = 0.0
        c.review_count = 5

        # Add non-reviewed dormant concepts
        for i in range(20):
            net.add_concept(f"dormant_{i}", confidence=0.4, origin="learned")
            c = net.get_concept(f"dormant_{i}")
            assert c is not None
            c.activation = 0.0

        spilled = net.spill_dormant(activation_threshold=0.01, max_in_memory=5)
        assert spilled > 0

        # Reviewed concept should still be in working memory
        assert net.get_concept("reviewed_concept") is not None
        net._archive.close()


def test_transparent_recall_by_name() -> None:
    """A spilled concept is recalled transparently when accessed by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Add and then spill a concept
        net.add_concept(
            "neuroscience",
            aliases={"neurobiology"},
            confidence=0.8,
            origin="learned",
            properties={"definition": "Study of the nervous system."},
        )
        c = net.get_concept("neuroscience")
        assert c is not None
        c.activation = 0.0

        # Force spill by setting a very low max_in_memory
        net.spill_dormant(activation_threshold=0.01, max_in_memory=0)
        assert net.size == 0
        assert net.archive_size == 1

        # Access by name — should trigger transparent recall
        recalled = net.get_concept("neuroscience")
        assert recalled is not None
        assert recalled.id == "neuroscience"
        assert recalled.confidence == 0.8
        assert recalled.properties["definition"] == "Study of the nervous system."
        assert net.size == 1  # back in working memory
        assert net.archive_size == 0  # removed from archive
        net._archive.close()


def test_transparent_recall_by_alias() -> None:
    """A spilled concept is recalled when accessed by an alias."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        net.add_concept(
            "ai",
            aliases={"artificial intelligence"},
            confidence=0.9,
            origin="learned",
        )
        c = net.get_concept("ai")
        assert c is not None
        c.activation = 0.0

        net.spill_dormant(activation_threshold=0.01, max_in_memory=0)
        assert net.archive_size == 1

        # Access by alias — should trigger transparent recall
        recalled = net.get_concept("artificial intelligence")
        assert recalled is not None
        assert recalled.id == "ai"
        assert net.archive_size == 0
        net._archive.close()


def test_edges_preserved_after_spill() -> None:
    """Edges stay in memory when a concept is spilled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Create two connected concepts
        net.add_concept("brain", confidence=0.8, origin="conversation")
        net.add_concept("neuron", confidence=0.6, origin="learned")
        net.add_edge("brain", "neuron", RelationType.PART_OF)

        # Make neuron dormant and spill it
        c = net.get_concept("neuron")
        assert c is not None
        c.activation = 0.0

        net.spill_dormant(activation_threshold=0.01, max_in_memory=1)
        assert net.size == 1  # only brain in working memory
        assert net.archive_size == 1  # neuron in archive

        # Edge should still be in memory
        assert net.edge_count == 1
        edges = net.get_edges("brain", "out")
        assert len(edges) == 1
        assert edges[0].target == "neuron"

        # Recall neuron — edge should still be accessible
        recalled = net.get_concept("neuron")
        assert recalled is not None
        edges = net.get_edges("brain", "out")
        assert len(edges) == 1
        net._archive.close()


def test_dedupe_archive_removes_stale_shadow_rows() -> None:
    """Archive rows for concepts present in working memory are dropped.

    Regression test: a save file written before a spill ran (autosave
    cadence, or a kill between spill and save) restores the concept to
    working memory while its archive row stays on disk. Once hot,
    nothing ever touches the stale row — dedupe_archive cleans it so
    the archive only holds genuinely dormant concepts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Spill a concept, then simulate a stale save resurrecting it
        # by writing it back into working memory directly (the path
        # _restore_concepts uses — it bypasses add_concept's archive
        # check).
        net.add_concept("shadow", confidence=0.6, origin="learned")
        c = net.get_concept("shadow")
        assert c is not None
        c.activation = 0.0
        net.spill_dormant(activation_threshold=0.01, max_in_memory=0)
        assert net.archive_size == 1
        assert net.size == 0

        # Resurrect without going through recall (stale-save restore)
        net._concepts["shadow"] = c
        net._alias_map.setdefault("shadow", []).append("shadow")

        removed = net.dedupe_archive()
        assert removed == 1
        assert net.archive_size == 0
        assert net.size == 1  # working memory copy untouched
        net._archive.close()


def test_dedupe_archive_keeps_genuinely_dormant_rows() -> None:
    """Dedupe only removes rows whose concepts are in working memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)
        net.add_concept("dormant_thing", confidence=0.4, origin="learned")
        c = net.get_concept("dormant_thing")
        assert c is not None
        c.activation = 0.0
        net.spill_dormant(activation_threshold=0.01, max_in_memory=0)
        assert net.archive_size == 1

        # Nothing duplicated — dedupe is a no-op
        assert net.dedupe_archive() == 0
        assert net.archive_size == 1
        net._archive.close()


def test_no_archive_no_spill() -> None:
    """Without an archive, spill_dormant returns 0."""
    net = ConceptNetwork()  # no archive attached
    for i in range(20):
        net.add_concept(f"dormant_{i}", confidence=0.4, origin="learned")
        c = net.get_concept(f"dormant_{i}")
        assert c is not None
        c.activation = 0.0

    spilled = net.spill_dormant(activation_threshold=0.01, max_in_memory=5)
    assert spilled == 0
    assert net.size == 20  # nothing spilled


def test_total_concept_count() -> None:
    """total_concept_count includes both working memory and archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        for i in range(20):
            net.add_concept(f"concept_{i}", confidence=0.4, origin="learned")
            c = net.get_concept(f"concept_{i}")
            assert c is not None
            c.activation = 0.0

        net.spill_dormant(activation_threshold=0.01, max_in_memory=5)
        assert net.size == 5
        assert net.archive_size == 15
        assert net.total_concept_count == 20
        net._archive.close()


def test_consolidate_during_sleep_spills() -> None:
    """consolidate_during_sleep includes the spill step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        net = _make_network_with_archive(tmpdir)

        # Add many dormant concepts with high enough confidence to
        # survive clean_noise (which removes confidence <= 0.4 with
        # < 1 edge), but low enough activation to be spillable.
        for i in range(20):
            net.add_concept(f"sleep_dormant_{i}", confidence=0.6, origin="learned")
            c = net.get_concept(f"sleep_dormant_{i}")
            assert c is not None
            c.activation = 0.0

        result = net.consolidate_during_sleep(max_in_memory=5)
        assert "spilled" in result
        assert result["spilled"] > 0
        assert net.archive_size > 0
        net._archive.close()


def test_archive_persistence_across_network_restart() -> None:
    """Archive persists when a new ConceptNetwork is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First network: add and spill concepts
        net1 = _make_network_with_archive(tmpdir)
        net1.add_concept("persistent_concept", confidence=0.7, origin="learned")
        c = net1.get_concept("persistent_concept")
        assert c is not None
        c.activation = 0.0
        net1.spill_dormant(activation_threshold=0.01, max_in_memory=0)
        assert net1.archive_size == 1
        net1._archive.close()

        # Second network: attach to the same archive
        net2 = ConceptNetwork()
        net2.attach_archive(open_archive(tmpdir))
        assert net2.archive_size == 1

        # Should be able to recall the spilled concept
        recalled = net2.get_concept("persistent_concept")
        assert recalled is not None
        assert recalled.id == "persistent_concept"
        assert recalled.confidence == 0.7
        net2._archive.close()


# ======================================================================
# From tests/test_concept_quality_filter.py
# ======================================================================

@pytest.mark.parametrize(
    "concept_id",
    [
        # Single words
        "dopamine",
        "cognition",
        "serotonin",
        "memory",
        "neurochemistry",
        # Multi-word concepts
        "neural activity",
        "supreme court",
        "bill of rights",
        "emotional state",
        "cortisol receptor",
        # Capitalized proper nouns
        "United States",
        "Supreme Court",
        "Friston",
        # With sense suffix (should still pass — suffix is stripped)
        "orange#2",
        "light#3",
    ],
)
def test_is_world_concept_accepts_genuine_concepts(concept_id: str) -> None:
    """Test is world concept accepts genuine concepts."""
    assert is_world_concept(concept_id), f"Should accept '{concept_id}'"


# ─── is_world_concept: rejected concepts ──────────────────────────


@pytest.mark.parametrize(
    "concept_id",
    [
        # Code symbols
        "python:mind.deny_site",
        "rust:state.neurochemical",
        "python:genesis_cognitive.concepts.conceptnetwork",
        "python:brain_waves.compute_gamma_synchrony",
        # Structural hubs
        "_cat:emotion:joy",
        "_cat:mode:contemplation",
        "identity:genesis",
        # Function words
        "and",
        "it",
        "is",
        "the",
        "a",
        "of",
        "to",
        "what",
        "you",
        "i",
        # Conversation fragments (start with function word)
        "i know the computer",
        "are related because genesis",
        "to rush",
        "we dont",
        "does not know what",
        "in what sense",
        # With digits (not sense suffixes)
        "2024",
        "166-169 ad",
        # Too short
        "ab",
        "x",
        # Code-path-like (dot + underscore)
        "mind.deny_site",
        "concept_network.conceptnetwork",
        # Empty
        "",
        "   ",
    ],
)
def test_is_world_concept_rejects_non_concepts(concept_id: str) -> None:
    """Test is world concept rejects non concepts."""
    assert not is_world_concept(concept_id), f"Should reject '{concept_id}'"


def test_is_world_concept_genesis_is_accepted() -> None:
    """'genesis' is a real concept (her name), not a function word."""
    assert is_world_concept("genesis")


def test_is_world_concept_strips_sense_suffix_before_checking() -> None:
    """Sense suffixes (#N) should be stripped before checking."""
    assert is_world_concept("orange#2")
    assert is_world_concept("cognition#5")
    # But a fragment with a sense suffix should still be rejected
    assert not is_world_concept("and#2")


# ─── world_concept_ids property ───────────────────────────────────


def test_world_concept_ids_filters_correctly() -> None:
    """The world_concept_ids property should only return world concepts."""
    net = ConceptNetwork()
    # Add a mix of concepts
    net.add_concept("dopamine", origin="learned")
    net.add_concept("neural activity", origin="learned")
    net.add_concept("python:mind.deny_site", origin="code")
    net.add_concept("and", origin="learned")
    net.add_concept("_cat:emotion:joy", origin="structural")
    net.add_concept("i know the computer", origin="learned")
    net.add_concept("cognition", origin="learned")

    world_ids = net.world_concept_ids
    # Should only include genuine world concepts
    assert "dopamine" in world_ids
    assert "neural activity" in world_ids
    assert "cognition" in world_ids
    # Should NOT include non-world concepts
    assert "python:mind.deny_site" not in world_ids
    assert "and" not in world_ids
    assert "_cat:emotion:joy" not in world_ids
    assert "i know the computer" not in world_ids


def test_world_concept_ids_cache_invalidated_on_add() -> None:
    """Adding a concept should invalidate the world_concept_ids cache."""
    net = ConceptNetwork()
    net.add_concept("dopamine", origin="learned")
    first = net.world_concept_ids
    assert "dopamine" in first

    net.add_concept("serotonin", origin="learned")
    second = net.world_concept_ids
    assert "serotonin" in second
    assert "dopamine" in second


def test_world_concept_ids_cache_invalidated_on_remove() -> None:
    """Removing a concept should invalidate the world_concept_ids cache."""
    net = ConceptNetwork()
    net.add_concept("dopamine", origin="learned")
    net.add_concept("serotonin", origin="learned")
    first = net.world_concept_ids
    assert len(first) == 2

    net.remove_concept("dopamine")
    second = net.world_concept_ids
    assert "dopamine" not in second
    assert "serotonin" in second


def test_world_concept_ids_empty_network() -> None:
    """An empty network should return an empty list."""
    net = ConceptNetwork()
    assert net.world_concept_ids == []


# ─── Curiosity engine filtering ───────────────────────────────────


def test_curiosity_engine_skips_non_world_concepts() -> None:
    """The curiosity engine should not generate questions about
    code symbols or function words."""
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.learning import CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    # Add a code symbol with high activation
    net.add_concept("python:mind.deny_site", origin="code")
    c = net.get_concept("python:mind.deny_site")
    if c:
        c.activation = 1.0
    # Add a function word with high activation
    net.add_concept("and", origin="learned")
    c = net.get_concept("and")
    if c:
        c.activation = 1.0
    # Add a real concept with high activation
    net.add_concept("dopamine", origin="learned")
    c = net.get_concept("dopamine")
    if c:
        c.activation = 1.0

    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)

    emotion = EmotionalState(
        label="neutral",
        cognitive_style="analytical",
        valence=0.5,
        alertness=0.5,
        plasticity=0.8,
        creativity=0.7,
        openness_to_engage=0.3,
    )

    questions = curiosity.generate_questions(emotion, max_questions=5)
    # No question should target a code symbol or function word
    for q in questions:
        assert is_world_concept(q.target_concept), (
            f"Question targets non-world concept: {q.target_concept}"
        )


# ─── Autonomous learner topic filtering ───────────────────────────


def test_autonomous_learner_add_topic_rejects_garbage() -> None:
    """add_topic should reject code symbols and function words."""
    from genesis_cognitive.learning import AutonomousLearner, CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    learner = AutonomousLearner(
        network=net,
        curiosity=curiosity,
        data_dir=None,
    )

    # These should be rejected
    learner.add_topic("python:mind.deny_site")
    learner.add_topic("and")
    learner.add_topic("it")
    learner.add_topic("i know the computer")

    # These should be accepted
    learner.add_topic("dopamine")
    learner.add_topic("neural activity")

    # Check the topic queue (it's a list)
    with learner._queue_lock:
        queue = learner._topic_queue
    assert "python:mind.deny_site" not in queue
    assert "and" not in queue
    assert "it" not in queue
    assert "i know the computer" not in queue
    assert "dopamine" in queue
    assert "neural activity" in queue


# ─── concept_quality: structural scoring ──────────────────────────


def test_concept_quality_orphan_is_low() -> None:
    """An orphan concept with no definition, no edges, low confidence
    should have a quality score below the threshold."""
    net = ConceptNetwork()
    net.add_concept("orphan", origin="learned")
    c = net.get_concept("orphan")
    assert c is not None
    c.confidence = 0.1
    q = net.concept_quality("orphan")
    assert q < QUALITY_THRESHOLD, f"Orphan quality {q} should be < {QUALITY_THRESHOLD}"


def test_concept_quality_with_definition_passes() -> None:
    """A concept with a definition should pass the quality threshold."""
    net = ConceptNetwork()
    net.add_concept("defined", origin="learned")
    c = net.get_concept("defined")
    assert c is not None
    c.confidence = 0.5
    c.properties["definition"] = "A concept with meaning."
    q = net.concept_quality("defined")
    assert q >= QUALITY_THRESHOLD, f"Defined quality {q} should be >= {QUALITY_THRESHOLD}"


def test_concept_quality_with_typed_edges_passes() -> None:
    """A concept with typed edges (CAUSES, ENABLES) should pass."""
    net = ConceptNetwork()
    net.add_concept("connector", origin="learned")
    net.add_concept("target_a", origin="learned")
    net.add_concept("target_b", origin="learned")
    net.add_edge("connector", "target_a", RelationType.CAUSES)
    net.add_edge("connector", "target_b", RelationType.ENABLES)
    c = net.get_concept("connector")
    assert c is not None
    c.confidence = 0.5
    q = net.concept_quality("connector")
    assert q >= QUALITY_THRESHOLD


def test_concept_quality_full_concept_is_high() -> None:
    """A concept with definition + typed edges + high confidence
    should have a quality score near 1.0."""
    net = ConceptNetwork()
    net.add_concept("full", origin="learned")
    net.add_concept("a", origin="learned")
    net.add_concept("b", origin="learned")
    net.add_concept("c", origin="learned")
    net.add_edge("full", "a", RelationType.CAUSES, weight=0.9)
    net.add_edge("full", "b", RelationType.ENABLES, weight=0.8)
    net.add_edge("full", "c", RelationType.IS_A, weight=0.7)
    c = net.get_concept("full")
    assert c is not None
    c.confidence = 0.9
    c.properties["definition"] = "A fully-formed concept."
    q = net.concept_quality("full")
    assert q >= 0.8, f"Full concept quality {q} should be >= 0.8"


def test_concept_quality_unknown_concept_returns_zero() -> None:
    """Unknown concepts should return 0.0."""
    net = ConceptNetwork()
    assert net.concept_quality("nonexistent") == 0.0


def test_concept_quality_untyped_edges_dont_help_much() -> None:
    """RELATED_TO edges alone (co-occurrence) should not boost quality
    as much as typed edges."""
    net = ConceptNetwork()
    # Concept with only RELATED_TO edges
    net.add_concept("untyped", origin="learned")
    net.add_concept("a", origin="learned")
    net.add_concept("b", origin="learned")
    net.add_edge("untyped", "a", RelationType.RELATED_TO)
    net.add_edge("untyped", "b", RelationType.RELATED_TO)
    c = net.get_concept("untyped")
    assert c is not None
    c.confidence = 0.3
    q_untyped = net.concept_quality("untyped")

    # Concept with typed edges
    net.add_concept("typed", origin="learned")
    net.add_edge("typed", "a", RelationType.CAUSES)
    net.add_edge("typed", "b", RelationType.ENABLES)
    c2 = net.get_concept("typed")
    assert c2 is not None
    c2.confidence = 0.3
    q_typed = net.concept_quality("typed")

    assert q_typed > q_untyped, (
        f"Typed ({q_typed}) should be > untyped ({q_untyped})"
    )


# ─── quality_concept_ids property ─────────────────────────────────


def test_quality_concept_ids_filters_orphans() -> None:
    """quality_concept_ids should exclude low-quality orphans."""
    net = ConceptNetwork()
    # Orphan
    net.add_concept("orphan", origin="learned")
    c = net.get_concept("orphan")
    assert c is not None
    c.confidence = 0.1
    # Quality concept
    net.add_concept("real", origin="learned")
    c = net.get_concept("real")
    assert c is not None
    c.confidence = 0.8
    c.properties["definition"] = "A real concept."

    qids = net.quality_concept_ids
    assert "real" in qids
    assert "orphan" not in qids


def test_quality_concept_ids_cache_invalidated_on_add() -> None:
    """Adding a concept should invalidate the quality cache."""
    net = ConceptNetwork()
    net.add_concept("first", origin="learned")
    c = net.get_concept("first")
    assert c is not None
    c.confidence = 0.8
    c.properties["definition"] = "First concept."
    first = net.quality_concept_ids
    assert "first" in first

    net.add_concept("second", origin="learned")
    c = net.get_concept("second")
    assert c is not None
    c.confidence = 0.8
    c.properties["definition"] = "Second concept."
    second = net.quality_concept_ids
    assert "second" in second
    assert "first" in second


def test_quality_concept_ids_cache_invalidated_on_edge() -> None:
    """Adding an edge should invalidate the quality cache
    (edge counts affect quality scores)."""
    net = ConceptNetwork()
    net.add_concept("growing", origin="learned")
    net.add_concept("target", origin="learned")
    c = net.get_concept("growing")
    assert c is not None
    c.confidence = 0.3  # low confidence, no definition, no edges

    # Initially below threshold
    assert "growing" not in net.quality_concept_ids

    # Add typed edges to boost quality
    net.add_edge("growing", "target", RelationType.CAUSES)
    net.add_edge("growing", "target", RelationType.ENABLES)

    # Cache should be invalidated, concept should now appear
    assert "growing" in net.quality_concept_ids


# ─── dream_concept_ids property ───────────────────────────────────


def test_dream_concept_ids_uses_quality_when_enough() -> None:
    """When there are ≥10 quality concepts, dream_concept_ids
    should return only quality concepts."""
    net = ConceptNetwork()
    # Add 10 quality concepts (names without digits — is_world_concept
    # rejects names containing digits)
    quality_names = [
        "alpha", "bravo", "charlie", "delta", "echo",
        "foxtrot", "golf", "hotel", "india", "juliet",
    ]
    for name in quality_names:
        net.add_concept(name, origin="learned")
        c = net.get_concept(name)
        assert c is not None
        c.confidence = 0.8
        c.properties["definition"] = f"Quality concept {name}."
    # Add some orphans
    orphan_names = ["orphan_alpha", "orphan_bravo", "orphan_charlie"]
    for name in orphan_names:
        net.add_concept(name, origin="learned")
        c = net.get_concept(name)
        assert c is not None
        c.confidence = 0.1

    dream_ids = net.dream_concept_ids
    # Should only contain quality concepts
    for cid in dream_ids:
        assert cid in quality_names, f"Dream concept {cid} should be a quality concept"
    assert len(dream_ids) == 10


def test_dream_concept_ids_falls_back_to_world() -> None:
    """When there are <10 quality concepts, dream_concept_ids
    should fall back to world_concept_ids."""
    net = ConceptNetwork()
    # Add only 2 quality concepts
    net.add_concept("quality_a", origin="learned")
    c = net.get_concept("quality_a")
    assert c is not None
    c.confidence = 0.8
    c.properties["definition"] = "A."
    net.add_concept("quality_b", origin="learned")
    c = net.get_concept("quality_b")
    assert c is not None
    c.confidence = 0.8
    c.properties["definition"] = "B."
    # Add some orphans
    net.add_concept("orphan_a", origin="learned")
    c = net.get_concept("orphan_a")
    assert c is not None
    c.confidence = 0.1

    dream_ids = net.dream_concept_ids
    # Should fall back to all world concepts
    assert "orphan_a" in dream_ids
    assert "quality_a" in dream_ids


# ─── Curiosity engine quality preference ──────────────────────────


def test_curiosity_engine_prefers_quality_concepts() -> None:
    """When selecting from activation (idle curiosity), the curiosity
    engine should prefer high-quality concepts over low-quality ones."""
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.learning import CuriosityEngine
    from genesis_cognitive.reasoning import ReasoningEngine

    net = ConceptNetwork()
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)

    # Add low-quality concepts with high activation
    for name in ["zzz_low_1", "zzz_low_2", "zzz_low_3", "zzz_low_4", "zzz_low_5"]:
        net.add_concept(name, origin="learned")
        c = net.get_concept(name)
        assert c is not None
        c.activation = 1.0
        c.confidence = 0.05  # very low quality

    # Add high-quality concepts with slightly lower activation
    for name in ["aaa_high_1", "aaa_high_2", "aaa_high_3"]:
        net.add_concept(name, origin="learned")
        c = net.get_concept(name)
        assert c is not None
        c.activation = 0.8
        c.confidence = 0.9
        c.properties["definition"] = f"High quality concept {name}."

    emotion = EmotionalState(
        label="neutral",
        cognitive_style="analytical",
        valence=0.5,
        alertness=0.5,
        plasticity=0.8,
        creativity=0.7,
        openness_to_engage=0.3,
    )

    questions = curiosity.generate_questions(emotion, max_questions=5)
    # All questions should target high-quality concepts
    for q in questions:
        q_concept = net.get_concept(q.target_concept)
        if q_concept:
            assert q_concept.confidence >= 0.5, (
                f"Question targets low-quality concept: {q.target_concept} "
                f"(confidence={q_concept.confidence})"
            )


# ======================================================================
# From tests/test_typed_relation_extraction.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# _relation_to_edge mapping completeness
# ═══════════════════════════════════════════════════════════════════


def test_relation_to_edge_maps_all_typed_relations() -> None:
    """Every typed relation string should map to a RelationType.

    Before this fix, only 6 of 14 relation types were mapped —
    emerges_from, depends_on, similar_to, opposite_of, prevents,
    leads_to, creates, harms, and contradicts were all dropped
    (returned None), meaning extracted facts for those relations
    were silently discarded instead of becoming edges.
    """
    expected = {
        "is_a": RelationType.IS_A,
        "part_of": RelationType.PART_OF,
        "causes": RelationType.CAUSES,
        "enables": RelationType.ENABLES,
        "relates_to": RelationType.RELATED_TO,
        "has_property": RelationType.HAS_PROPERTY,
        "emerges_from": RelationType.EMERGES_FROM,
        "depends_on": RelationType.DEPENDS_ON,
        "similar_to": RelationType.SIMILAR_TO,
        "opposite_of": RelationType.OPPOSITE_OF,
        "prevents": RelationType.PREVENTS,
        "leads_to": RelationType.LEADS_TO,
        "creates": RelationType.CREATES,
        "harms": RelationType.HARMS,
        "contradicts": RelationType.CONTRADICTS,
    }
    for rel_str, expected_type in expected.items():
        result = _relation_to_edge(rel_str)
        assert result == expected_type, (
            f"_relation_to_edge('{rel_str}') returned {result}, "
            f"expected {expected_type}"
        )


def test_relation_to_edge_case_insensitive() -> None:
    """Relation strings should be case-insensitive."""
    assert _relation_to_edge("CAUSES") == RelationType.CAUSES
    assert _relation_to_edge("Emerges_From") == RelationType.EMERGES_FROM


def test_relation_to_edge_unknown_returns_none() -> None:
    """Unknown relation strings should return None, not raise."""
    assert _relation_to_edge("unknown_relation") is None
    assert _relation_to_edge("") is None


# ═══════════════════════════════════════════════════════════════════
# ConceptNetwork.parse_relationships — expanded patterns
# ═══════════════════════════════════════════════════════════════════


def _parse(network: ConceptNetwork, text: str) -> list[tuple[str, RelationType, str]]:
    """Helper: parse text and return results."""
    return network.parse_relationships(text)


def test_parse_causes_synonyms() -> None:
    """Synonyms for 'causes' should produce CAUSES edges."""
    network = ConceptNetwork()
    # "X produces Y" → CAUSES
    results = _parse(network, "Cortisol produces stress.")
    assert any(
        r[0] == "cortisol" and r[1] == RelationType.CAUSES and r[2] == "stress"
        for r in results
    ), f"'produces' should map to CAUSES, got {results}"

    # "X triggers Y" → CAUSES
    results = _parse(network, "Glutamate triggers excitation.")
    assert any(
        r[1] == RelationType.CAUSES for r in results
    ), f"'triggers' should map to CAUSES, got {results}"

    # "X is responsible for Y" → CAUSES
    results = _parse(network, "Cortisol is responsible for stress.")
    assert any(
        r[0] == "cortisol" and r[1] == RelationType.CAUSES and r[2] == "stress"
        for r in results
    ), f"'is responsible for' should map to CAUSES, got {results}"


def test_parse_emerges_from_synonyms() -> None:
    """Synonyms for 'emerges from' should produce EMERGES_FROM edges."""
    network = ConceptNetwork()
    # "X arises from Y" → EMERGES_FROM
    results = _parse(network, "Cognition arises from neural activity.")
    assert any(
        r[0] == "cognition" and r[1] == RelationType.EMERGES_FROM
        for r in results
    ), f"'arises from' should map to EMERGES_FROM, got {results}"

    # "X stems from Y" → EMERGES_FROM
    results = _parse(network, "Memory stems from synaptic plasticity.")
    assert any(
        r[1] == RelationType.EMERGES_FROM for r in results
    ), f"'stems from' should map to EMERGES_FROM, got {results}"


def test_parse_depends_on_synonyms() -> None:
    """Synonyms for 'depends on' should produce DEPENDS_ON edges."""
    network = ConceptNetwork()
    # "X relies on Y" → DEPENDS_ON
    results = _parse(network, "Memory relies on hippocampus.")
    assert any(
        r[0] == "memory" and r[1] == RelationType.DEPENDS_ON
        for r in results
    ), f"'relies on' should map to DEPENDS_ON, got {results}"

    # "X needs Y" → DEPENDS_ON
    results = _parse(network, "Learning needs dopamine.")
    assert any(
        r[1] == RelationType.DEPENDS_ON for r in results
    ), f"'needs' should map to DEPENDS_ON, got {results}"


def test_parse_enables_synonyms() -> None:
    """Synonyms for 'enables' should produce ENABLES edges."""
    network = ConceptNetwork()
    # "X facilitates Y" → ENABLES
    results = _parse(network, "Sleep facilitates memory consolidation.")
    assert any(
        r[1] == RelationType.ENABLES for r in results
    ), f"'facilitates' should map to ENABLES, got {results}"

    # "X promotes Y" → ENABLES
    results = _parse(network, "Exercise promotes neurogenesis.")
    assert any(
        r[1] == RelationType.ENABLES for r in results
    ), f"'promotes' should map to ENABLES, got {results}"

    # "X is required for Y" → ENABLES (X enables Y)
    results = _parse(network, "Hippocampus is required for memory formation.")
    assert any(
        r[1] == RelationType.ENABLES for r in results
    ), f"'is required for' should map to ENABLES, got {results}"


def test_parse_prevents_synonyms() -> None:
    """Synonyms for 'prevents' should produce PREVENTS edges."""
    network = ConceptNetwork()
    # "X prevents Y" → PREVENTS
    results = _parse(network, "GABA prevents excitation.")
    assert any(
        r[0] == "gaba" and r[1] == RelationType.PREVENTS
        for r in results
    ), f"'prevents' should map to PREVENTS, got {results}"

    # "X blocks Y" → PREVENTS
    results = _parse(network, "Endorphin blocks pain.")
    assert any(
        r[1] == RelationType.PREVENTS for r in results
    ), f"'blocks' should map to PREVENTS, got {results}"

    # "X inhibits Y" → PREVENTS (not HARMS — inhibiting is blocking)
    results = _parse(network, "Serotonin inhibits aggression.")
    assert any(
        r[1] == RelationType.PREVENTS for r in results
    ), f"'inhibits' should map to PREVENTS, got {results}"


def test_parse_leads_to_synonyms() -> None:
    """Synonyms for 'leads to' should produce LEADS_TO edges."""
    network = ConceptNetwork()
    # "X results in Y" → LEADS_TO
    results = _parse(network, "Practice results in mastery.")
    assert any(
        r[0] == "practice" and r[1] == RelationType.LEADS_TO
        for r in results
    ), f"'results in' should map to LEADS_TO, got {results}"

    # "X gives rise to Y" → LEADS_TO
    results = _parse(network, "Curiosity gives rise to learning.")
    assert any(
        r[1] == RelationType.LEADS_TO for r in results
    ), f"'gives rise to' should map to LEADS_TO, got {results}"


def test_parse_similar_to_synonyms() -> None:
    """Synonyms for 'similar to' should produce SIMILAR_TO edges."""
    network = ConceptNetwork()
    # "X is analogous to Y" → SIMILAR_TO
    results = _parse(network, "Dopamine is analogous to serotonin.")
    assert any(
        r[1] == RelationType.SIMILAR_TO for r in results
    ), f"'is analogous to' should map to SIMILAR_TO, got {results}"

    # "X resembles Y" → SIMILAR_TO
    results = _parse(network, "GABA resembles glycine.")
    assert any(
        r[1] == RelationType.SIMILAR_TO for r in results
    ), f"'resembles' should map to SIMILAR_TO, got {results}"


def test_parse_opposite_of_synonyms() -> None:
    """Synonyms for 'opposite of' should produce OPPOSITE_OF edges."""
    network = ConceptNetwork()
    # "X contrasts with Y" → OPPOSITE_OF
    results = _parse(network, "Wakefulness contrasts with sleep.")
    assert any(
        r[1] == RelationType.OPPOSITE_OF for r in results
    ), f"'contrasts with' should map to OPPOSITE_OF, got {results}"


def test_parse_harms_synonyms() -> None:
    """Synonyms for 'harms' should produce HARMS edges."""
    network = ConceptNetwork()
    # "X impairs Y" → HARMS
    results = _parse(network, "Stress impairs memory.")
    assert any(
        r[0] == "stress" and r[1] == RelationType.HARMS
        for r in results
    ), f"'impairs' should map to HARMS, got {results}"

    # "X disrupts Y" → HARMS
    results = _parse(network, "Cortisol disrupts sleep.")
    assert any(
        r[1] == RelationType.HARMS for r in results
    ), f"'disrupts' should map to HARMS, got {results}"

    # "X damages Y" → HARMS
    results = _parse(network, "Chronic stress damages hippocampus.")
    assert any(
        r[1] == RelationType.HARMS for r in results
    ), f"'damages' should map to HARMS, got {results}"


def test_parse_part_of_synonyms() -> None:
    """Synonyms for 'part of' should produce PART_OF edges."""
    network = ConceptNetwork()
    # "X consists of Y" → Y part_of X (reversed)
    results = _parse(network, "Brain consists of neurons.")
    assert any(
        r[1] == RelationType.PART_OF for r in results
    ), f"'consists of' should map to PART_OF, got {results}"

    # "X is composed of Y" → Y part_of X (reversed)
    results = _parse(network, "Nucleus is composed of protons.")
    assert any(
        r[1] == RelationType.PART_OF for r in results
    ), f"'is composed of' should map to PART_OF, got {results}"


def test_parse_existing_patterns_still_work() -> None:
    """The original patterns should still work after the expansion."""
    network = ConceptNetwork()
    # "X is a Y" → IS_A
    results = _parse(network, "Neuron is a cell.")
    assert any(
        r[0] == "neuron" and r[1] == RelationType.IS_A and r[2] == "cell"
        for r in results
    ), f"'is a' should still map to IS_A, got {results}"

    # "X causes Y" → CAUSES
    results = _parse(network, "Fire causes smoke.")
    assert any(
        r[0] == "fire" and r[1] == RelationType.CAUSES and r[2] == "smoke"
        for r in results
    ), f"'causes' should still map to CAUSES, got {results}"

    # "X is part of Y" → PART_OF
    results = _parse(network, "Hippocampus is part of limbic system.")
    assert any(
        r[1] == RelationType.PART_OF for r in results
    ), f"'is part of' should still map to PART_OF, got {results}"


# ═══════════════════════════════════════════════════════════════════
# SemanticMemory.extract_facts — typed fact extraction
# ═══════════════════════════════════════════════════════════════════


def test_extract_facts_emerges_from() -> None:
    """extract_facts should extract EMERGES_FROM facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Cognition emerges from neural activity.")
    assert any(
        f.relation == "emerges_from" for f in facts
    ), f"Should extract emerges_from fact, got {[f.relation for f in facts]}"


def test_extract_facts_depends_on() -> None:
    """extract_facts should extract DEPENDS_ON facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Memory depends on hippocampus.")
    assert any(
        f.relation == "depends_on" for f in facts
    ), f"Should extract depends_on fact, got {[f.relation for f in facts]}"


def test_extract_facts_prevents() -> None:
    """extract_facts should extract PREVENTS facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("GABA prevents excitation.")
    assert any(
        f.relation == "prevents" for f in facts
    ), f"Should extract prevents fact, got {[f.relation for f in facts]}"


def test_extract_facts_leads_to() -> None:
    """extract_facts should extract LEADS_TO facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Practice leads to mastery.")
    assert any(
        f.relation == "leads_to" for f in facts
    ), f"Should extract leads_to fact, got {[f.relation for f in facts]}"


def test_extract_facts_harms() -> None:
    """extract_facts should extract HARMS facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Stress impairs memory.")
    assert any(
        f.relation == "harms" for f in facts
    ), f"Should extract harms fact, got {[f.relation for f in facts]}"


def test_extract_facts_similar_to() -> None:
    """extract_facts should extract SIMILAR_TO facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Dopamine is similar to serotonin.")
    assert any(
        f.relation == "similar_to" for f in facts
    ), f"Should extract similar_to fact, got {[f.relation for f in facts]}"


def test_extract_facts_opposite_of() -> None:
    """extract_facts should extract OPPOSITE_OF facts."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Wakefulness is the opposite of sleep.")
    assert any(
        f.relation == "opposite_of" for f in facts
    ), f"Should extract opposite_of fact, got {[f.relation for f in facts]}"


def test_extract_facts_creates_typed_edge_in_network() -> None:
    """Extracted facts should consolidate into typed edges in the network."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    # Extract a causes fact
    sm.extract_facts("Cortisol causes stress.")
    # The network should now have a CAUSES edge
    edges = network.get_edges("cortisol", direction="out")
    causes_edges = [e for e in edges if e.relation == RelationType.CAUSES]
    edge_summary = [(e.source, e.relation, e.target) for e in edges]
    assert any(
        e.target == "stress" for e in causes_edges
    ), f"Network should have cortisol CAUSES stress edge, got {edge_summary}"


def test_extract_facts_emerges_from_creates_typed_edge() -> None:
    """EMERGES_FROM facts should create EMERGES_FROM edges (not be dropped)."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    sm.extract_facts("Cognition emerges from neural activity.")
    edges = network.get_edges("cognition", direction="out")
    emerges_edges = [e for e in edges if e.relation == RelationType.EMERGES_FROM]
    edge_summary = [(e.source, e.relation, e.target) for e in edges]
    assert any(
        e.target == "neural activity" for e in emerges_edges
    ), f"Network should have cognition EMERGES_FROM neural activity edge, got {edge_summary}"


# ═══════════════════════════════════════════════════════════════════
# Regression: existing extraction still works
# ═══════════════════════════════════════════════════════════════════


def test_extract_facts_is_a_still_works() -> None:
    """The original is_a extraction should still work."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Neuron is a cell.")
    fact_summary = [(f.subject, f.relation, f.object) for f in facts]
    assert any(
        f.relation == "is_a" and f.subject == "neuron" and f.object == "cell"
        for f in facts
    ), f"is_a extraction should still work, got {fact_summary}"


def test_extract_facts_causes_still_works() -> None:
    """The original causes extraction should still work."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Fire causes smoke.")
    assert any(
        f.relation == "causes" for f in facts
    ), f"causes extraction should still work, got {[f.relation for f in facts]}"


def test_extract_facts_part_of_still_works() -> None:
    """The original part_of extraction should still work."""
    network = ConceptNetwork()
    sm = SemanticMemory(network=network)
    facts = sm.extract_facts("Hippocampus is part of brain.")
    assert any(
        f.relation == "part_of" for f in facts
    ), f"part_of extraction should still work, got {[f.relation for f in facts]}"


# ======================================================================
# From tests/test_orphan_fixes.py
# ======================================================================

def _make_minimal_network(extra_concepts: int = 0) -> ConceptNetwork:
    """Build a network with >= 50 concepts so _attach_orphans_to_hubs runs."""
    net = ConceptNetwork()
    net.add_concept("hub", confidence=0.9)
    for i in range(6):
        net.add_concept(f"leaf_{i}", confidence=0.7)
        net.add_edge("hub", f"leaf_{i}", RelationType.RELATED_TO, weight=0.5)
    for i in range(44 + extra_concepts):
        net.add_concept(f"filler_{i}", confidence=0.4)
    return net


# ── Bug 1: zero-edge concepts are now visible to _attach_orphans_to_hubs ──


def test_attach_orphans_sees_zero_edge_concepts() -> None:
    """Zero-edge concepts should be connected by _attach_orphans_to_hubs."""
    net = _make_minimal_network()

    # Add a zero-edge concept that shares a word with the hub
    net.add_concept("hub_related_thing", confidence=0.4)

    # Before the fix, this concept would not be in the degree dict
    # and thus invisible to _attach_orphans_to_hubs.
    created = net._attach_orphans_to_hubs(max_new=100)
    assert created > 0

    # The zero-edge concept should now have at least one edge
    edges = net._edge_index.get("hub_related_thing", [])
    assert len(edges) > 0, "zero-edge concept was not connected"


def test_attach_orphans_connects_concepts_with_no_word_overlap() -> None:
    """Zero-edge concepts with no word overlap get connected to super hubs."""
    net = ConceptNetwork()
    # Build several hubs and a large enough network
    net.add_concept("science", confidence=0.9)
    net.add_concept("biology", confidence=0.9)
    net.add_concept("chemistry", confidence=0.9)
    for i in range(6):
        net.add_concept(f"sci_leaf_{i}", confidence=0.7)
        net.add_edge("science", f"sci_leaf_{i}", RelationType.RELATED_TO, weight=0.5)
        net.add_edge("biology", f"sci_leaf_{i}", RelationType.RELATED_TO, weight=0.5)
        net.add_edge("chemistry", f"sci_leaf_{i}", RelationType.RELATED_TO, weight=0.5)
    for i in range(44):
        net.add_concept(f"filler_{i}", confidence=0.4)

    # Add a zero-edge concept with no word overlap to any hub
    net.add_concept("xyzqqq", confidence=0.4)

    created = net._attach_orphans_to_hubs(max_new=100)
    assert created > 0

    # Should be connected to a super hub even without word overlap
    edges = net._edge_index.get("xyzqqq", [])
    assert len(edges) > 0, "zero-edge concept with no word overlap was not connected"


# ── Bug 2: clean_noise now prunes concepts at exactly the threshold ──


def test_clean_noise_prunes_concept_at_exact_threshold() -> None:
    """Concepts at exactly min_confidence with 0 edges should be pruned."""
    net = ConceptNetwork()

    # Concept at exactly 0.4 with no edges — the learner's default
    net.add_concept("orphan_at_threshold", confidence=0.4)

    # Concept at 0.4 with an edge — should survive
    net.add_concept("connected_at_threshold", confidence=0.4)
    net.add_concept("hub", confidence=0.8)
    net.add_edge("connected_at_threshold", "hub", RelationType.RELATED_TO, weight=0.5)

    removed = net.clean_noise(min_confidence=0.4, min_edges=1)

    assert "orphan_at_threshold" in removed
    assert "connected_at_threshold" not in removed
    assert net.get_concept("orphan_at_threshold") is None
    assert net.get_concept("connected_at_threshold") is not None


def test_clean_noise_preserves_high_confidence_isolated() -> None:
    """High-confidence isolated concepts should still survive."""
    net = ConceptNetwork()
    net.add_concept("important_orphan", confidence=0.9)

    removed = net.clean_noise(min_confidence=0.4, min_edges=1)
    assert "important_orphan" not in removed
    assert net.get_concept("important_orphan") is not None


def test_clean_noise_preserves_connected_at_threshold() -> None:
    """Connected concepts at the threshold should survive."""
    net = ConceptNetwork()
    net.add_concept("connected", confidence=0.4)
    net.add_concept("hub", confidence=0.8)
    net.add_edge("connected", "hub", RelationType.RELATED_TO, weight=0.5)

    removed = net.clean_noise(min_confidence=0.4, min_edges=1)
    assert "connected" not in removed


# ── Bug 3: consolidate_during_sleep connects before pruning ──


def test_consolidate_connects_orphans_before_pruning() -> None:
    """Sleep should connect orphans to hubs before pruning them.

    The old order (prune → connect) deleted concepts that could have
    been connected. The new order (connect → prune) gives orphans a
    chance to be attached first.
    """
    net = _make_minimal_network()

    # Add a connectable orphan (shares word "hub")
    net.add_concept("hub_related_thing", confidence=0.4)

    net.consolidate_during_sleep()

    # The connectable orphan should have been connected
    edges = net._edge_index.get("hub_related_thing", [])
    assert len(edges) > 0, "connectable orphan was not connected during sleep"
    # clean_noise should NOT remove it once connected
    assert net.get_concept("hub_related_thing") is not None


def test_consolidate_removes_truly_isolated_noise() -> None:
    """After hub attachment, truly unconnectable noise is pruned."""
    net = _make_minimal_network()

    # Add noise at confidence 0.2 (below threshold, no edges)
    net.add_concept("noise_low_conf", confidence=0.2)

    result = net.consolidate_during_sleep()
    assert result["noise_removed"] >= 1
    assert net.get_concept("noise_low_conf") is None


# ── Regression: existing clean_noise behavior preserved ──


def test_clean_noise_still_removes_low_confidence() -> None:
    """Concepts below the threshold with no edges are still removed."""
    net = ConceptNetwork()
    net.add_concept("low_conf_orphan", confidence=0.3)

    removed = net.clean_noise(min_confidence=0.5, min_edges=1)
    assert "low_conf_orphan" in removed
    assert net.get_concept("low_conf_orphan") is None


def test_clean_noise_still_preserves_connected_low_conf() -> None:
    """Connected concepts below the threshold still survive."""
    net = ConceptNetwork()
    net.add_concept("connected_low", confidence=0.3)
    net.add_concept("hub", confidence=0.8)
    net.add_edge("connected_low", "hub", RelationType.RELATED_TO, weight=0.5)

    removed = net.clean_noise(min_confidence=0.5, min_edges=1)
    assert "connected_low" not in removed


# ======================================================================
# From tests/test_name_identity_extraction.py
# ======================================================================

def _find(facts, subj, rel, obj):
    """Helper: find."""
    return any(f.subject == subj and f.relation == rel and f.object == obj for f in facts)


def test_indefinite_is_a() -> None:
    """"X is a/an Y" extracts an is_a fact."""
    sm = SemanticMemory()
    facts = sm.extract_facts("A dog is an animal.")
    assert _find(facts, "dog", "is_a", "animal")
    assert all(f.confidence >= 0.5 for f in facts)


def test_plural_is_a() -> None:
    """"X are Y" extracts an is_a fact with lower confidence."""
    sm = SemanticMemory()
    facts = sm.extract_facts("Cats are mammals.")
    assert _find(facts, "cats", "is_a", "mammals")
    assert all(f.confidence < 0.5 for f in facts if f.relation == "is_a")


def test_relational_noun() -> None:
    """"X is the Y of Z" extracts is_a(X, Y) and relates_to(X, Z)."""
    sm = SemanticMemory()
    facts = sm.extract_facts("Alice is the creator of Genesis.")
    assert _find(facts, "alice", "is_a", "creator")
    assert _find(facts, "alice", "relates_to", "genesis")


def test_definite_without_of_is_skipped() -> None:
    """Bare "X is the Y" without a relation is too specific; not extracted."""
    sm = SemanticMemory()
    facts = sm.extract_facts("Alice is the creator.")
    assert not facts


def test_composition_split() -> None:
    """"X is composed/made of Y and Z" splits components into part_of."""
    sm = SemanticMemory()
    facts = sm.extract_facts("Water is composed of hydrogen and oxygen.")
    assert _find(facts, "hydrogen", "part_of", "water")
    assert _find(facts, "oxygen", "part_of", "water")
    assert not _find(facts, "water", "relates_to", "composed of hydrogen and oxygen")


def test_my_name_is_ignored() -> None:
    """Identity statements are not turned into concept-network facts."""
    sm = SemanticMemory()
    for text in (
        "My name is Alice.",
        "His name is John.",
        "Her name is Alice.",
    ):
        facts = sm.extract_facts(text)
        for f in facts:
            assert f.subject != "name"
            assert f.object != "alice" or f.relation != "is_a"


def test_possessive_property_still_works() -> None:
    """"My X is Y" still extracts has_property for non-identity properties."""
    sm = SemanticMemory()
    facts = sm.extract_facts("My favorite color is purple.")
    assert _find(facts, "favorite color", "has_property", "purple")


def test_bare_is_y_skips_adjective() -> None:
    """Bare "X is Y" with single-word object is skipped."""
    sm = SemanticMemory()
    facts = sm.extract_facts("The sky is blue.")
    assert not facts


def test_bare_is_y_multi_word_noun_phrase() -> None:
    """Bare "X is Y" with a multi-word noun phrase becomes a weak relates_to."""
    sm = SemanticMemory()
    facts = sm.extract_facts("The system is human centered design.")
    # "human centered design" is multi-word, no determiner
    assert _find(facts, "system", "relates_to", "human centered design")
    assert all(f.confidence < 0.5 for f in facts if f.relation == "relates_to")


def test_typed_relations_still_work() -> None:
    """Typed relation patterns continue to work."""
    sm = SemanticMemory()
    facts = sm.extract_facts("Fire causes heat.")
    assert _find(facts, "fire", "causes", "heat")

# ======================================================================
# From tests/test_prune_dead_concepts.py
# ======================================================================


def _concept(
    cid: str,
    origin: str = "learned",
    activation: float = 0.0,
    review_count: int = 0,
) -> dict:
    """Helper: concept."""
    return {
        "id": cid,
        "origin": origin,
        "activation": activation,
        "review_count": review_count,
        "confidence": 0.5,
    }


def _edge(
    source: str,
    target: str,
    weight: float = 0.5,
    origin: str = "learned",
) -> dict:
    """Helper: edge."""
    return {"source": source, "target": target, "weight": weight, "origin": origin}


# ─── _is_dead ──────────────────────────────────────────────────


def test_is_dead_zero_activation_zero_review() -> None:
    """A concept with 0 activation and 0 review_count is dead."""
    assert _is_dead(_concept("a", activation=0.0, review_count=0))


def test_is_dead_nonzero_activation() -> None:
    """A concept with non-zero activation is not dead."""
    assert not _is_dead(_concept("a", activation=0.1, review_count=0))


def test_is_dead_nonzero_review() -> None:
    """A concept with non-zero review_count is not dead."""
    assert not _is_dead(_concept("a", activation=0.0, review_count=1))


# ─── _has_only_weak_bridges ────────────────────────────────────


def test_has_only_weak_bridges_all_weak() -> None:
    """Concept with only weak bridge edges returns True."""
    edges = [
        _edge("a", "b", weight=0.15, origin="hub_attachment"),
        _edge("a", "c", weight=0.2, origin="semantic_bridge"),
    ]
    index = _build_edge_index(edges)
    assert _has_only_weak_bridges("a", index)


def test_has_only_weak_bridges_has_deliberate_edge() -> None:
    """Concept with a deliberate (non-bridge) edge returns False."""
    edges = [
        _edge("a", "b", weight=0.15, origin="hub_attachment"),
        _edge("a", "c", weight=0.5, origin="learned"),
    ]
    index = _build_edge_index(edges)
    assert not _has_only_weak_bridges("a", index)


def test_has_only_weak_bridges_has_strong_bridge() -> None:
    """Concept with a strong bridge edge (weight >= 0.3) returns False."""
    edges = [
        _edge("a", "b", weight=0.15, origin="hub_attachment"),
        _edge("a", "c", weight=0.4, origin="semantic_bridge"),
    ]
    index = _build_edge_index(edges)
    assert not _has_only_weak_bridges("a", index)


def test_has_only_weak_bridges_no_edges() -> None:
    """Concept with no edges returns False (handled by 0-edge check)."""
    index = _build_edge_index([])
    assert not _has_only_weak_bridges("a", index)


# ─── _classify_concepts ────────────────────────────────────────


def test_classify_prunes_dead_orphaned_concept() -> None:
    """Dead concept with 0 edges is pruned regardless of origin."""
    concepts = [_concept("dead_orphan", origin="learned")]
    keep_ids, prune_counts = _classify_concepts(concepts, [])
    assert "dead_orphan" not in keep_ids
    assert prune_counts["learned"] == 1


def test_classify_keeps_active_concept() -> None:
    """Active concept is kept regardless of origin."""
    concepts = [_concept("active", origin="learned", activation=0.5)]
    keep_ids, prune_counts = _classify_concepts(concepts, [])
    assert "active" in keep_ids
    assert len(prune_counts) == 0


def test_classify_keeps_reviewed_concept() -> None:
    """Reviewed concept is kept even if activation is 0."""
    concepts = [_concept("reviewed", origin="learned", review_count=3)]
    keep_ids, _ = _classify_concepts(concepts, [])
    assert "reviewed" in keep_ids


def test_classify_protects_identity_origin() -> None:
    """Protected origins are never pruned even if dead and orphaned."""
    for origin in ("identity", "introspection", "foundational", "seeded", "structural"):
        concepts = [_concept(f"dead_{origin}", origin=origin)]
        keep_ids, prune_counts = _classify_concepts(concepts, [])
        assert f"dead_{origin}" in keep_ids
        assert len(prune_counts) == 0


def test_classify_keeps_dead_concept_with_real_edges() -> None:
    """Dead concept with deliberate edges is kept in default mode."""
    concepts = [_concept("dead_connected", origin="learned")]
    edges = [_edge("dead_connected", "other", weight=0.5, origin="learned")]
    keep_ids, prune_counts = _classify_concepts(concepts, edges)
    assert "dead_connected" in keep_ids
    assert len(prune_counts) == 0


def test_classify_aggressive_prunes_dead_with_only_weak_bridges() -> None:
    """Aggressive mode prunes dead concepts whose only edges are weak bridges."""
    concepts = [
        _concept("dead_bridged", origin="learned"),
        _concept("hub", origin="learned", activation=0.5),
    ]
    edges = [
        _edge("dead_bridged", "hub", weight=0.15, origin="hub_attachment"),
    ]
    keep_ids, prune_counts = _classify_concepts(concepts, edges, aggressive=True)
    assert "dead_bridged" not in keep_ids
    assert "hub" in keep_ids
    assert prune_counts["learned"] == 1


def test_classify_aggressive_keeps_dead_with_deliberate_edge() -> None:
    """Aggressive mode keeps dead concepts with deliberate edges."""
    concepts = [
        _concept("dead_deliberate", origin="learned"),
        _concept("hub", origin="learned", activation=0.5),
    ]
    edges = [
        _edge("dead_deliberate", "hub", weight=0.5, origin="learned"),
    ]
    keep_ids, _ = _classify_concepts(concepts, edges, aggressive=True)
    assert "dead_deliberate" in keep_ids


# ─── prune_state (integration) ─────────────────────────────────


def test_prune_state_dry_run_does_not_modify_file(tmp_path: str) -> None:
    """Dry run reports but does not modify the state file."""
    state = {
        "version": 2,
        "concept_network": {
            "concepts": [
                _concept("dead_orphan", origin="learned"),
                _concept("active", origin="learned", activation=0.5),
            ],
            "edges": [],
        },
    }
    state_path = os.path.join(tmp_path, "cognitive_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    original_content = open(state_path).read()
    result = prune_state(tmp_path, dry_run=True)

    assert result["pruned_concepts"] == 1
    assert result["kept_concepts"] == 1
    assert open(state_path).read() == original_content  # unchanged


def test_prune_state_writes_pruned_file(tmp_path: str) -> None:
    """Non-dry-run writes the pruned state and creates a backup."""
    state = {
        "version": 2,
        "concept_network": {
            "concepts": [
                _concept("dead_orphan", origin="learned"),
                _concept("active", origin="learned", activation=0.5),
            ],
            "edges": [],
        },
    }
    state_path = os.path.join(tmp_path, "cognitive_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    result = prune_state(tmp_path, dry_run=False)

    assert result["pruned_concepts"] == 1
    assert result["kept_concepts"] == 1
    import glob
    assert glob.glob(state_path + ".pre-prune-backup.*.xz")

    with gzip.open(state_path, "rt") as f:
        new_state = json.load(f)
    ids = [c["id"] for c in new_state["concept_network"]["concepts"]]
    assert "dead_orphan" not in ids
    assert "active" in ids


def test_prune_state_aggressive_mode(tmp_path: str) -> None:
    """Aggressive mode prunes dead concepts with only weak bridges."""
    state = {
        "version": 2,
        "concept_network": {
            "concepts": [
                _concept("dead_bridged", origin="learned"),
                _concept("hub", origin="learned", activation=0.5),
                _concept("dead_orphan", origin="code"),
            ],
            "edges": [
                _edge("dead_bridged", "hub", weight=0.15, origin="hub_attachment"),
            ],
        },
    }
    state_path = os.path.join(tmp_path, "cognitive_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    # Default mode: only the orphan is pruned (1)
    result_default = prune_state(tmp_path, dry_run=True)
    assert result_default["pruned_concepts"] == 1

    # Aggressive mode: orphan + weak-bridged concept (2)
    result_aggressive = prune_state(tmp_path, dry_run=True, aggressive=True)
    assert result_aggressive["pruned_concepts"] == 2
    assert result_aggressive["mode"] == "aggressive"


def test_prune_state_protects_identity_origin(tmp_path: str) -> None:
    """Protected origins are never pruned."""
    state = {
        "version": 2,
        "concept_network": {
            "concepts": [
                _concept("dead_identity", origin="identity"),
                _concept("dead_introspection", origin="introspection"),
                _concept("dead_learned", origin="learned"),
            ],
            "edges": [],
        },
    }
    state_path = os.path.join(tmp_path, "cognitive_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    result = prune_state(tmp_path, dry_run=True)
    assert result["pruned_concepts"] == 1  # only "dead_learned"
    assert result["by_origin"] == {"learned": 1}


def test_prune_state_cleans_orphaned_edges(tmp_path: str) -> None:
    """Edges referencing pruned concepts are also removed."""
    state = {
        "version": 2,
        "concept_network": {
            "concepts": [
                _concept("dead_orphan", origin="learned"),
                _concept("active", origin="learned", activation=0.5),
            ],
            "edges": [
                # Weak bridge — dead_orphan will be pruned in aggressive mode
                _edge("dead_orphan", "active", weight=0.15, origin="hub_attachment"),
            ],
        },
    }
    state_path = os.path.join(tmp_path, "cognitive_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    result = prune_state(tmp_path, dry_run=True, aggressive=True)
    # "dead_orphan" is pruned → its edge to "active" is orphaned
    assert result["pruned_concepts"] == 1
    assert result["pruned_edges"] == 1
    assert result["kept_edges"] == 0


# ═══════════════════════════════════════════════════════════════════
# Relation verbs
# ═══════════════════════════════════════════════════════════════════


def test_seed_relation_verbs_enables_find_relation_verbs() -> None:
    """After seeding, find_relation_verbs returns natural verb phrases.

    The seeder was defined but never called, so find_relation_verbs()
    always returned [] and thought composition fell back to the raw
    semantic triple ("X related to Y" instead of "X relates to Y").
    """
    net = ConceptNetwork()
    assert net.find_relation_verbs(RelationType.RELATED_TO) == []

    net.seed_relation_verbs()

    verbs = net.find_relation_verbs(RelationType.RELATED_TO)
    assert "relates to" in verbs
    assert "connects to" in verbs


def test_seed_relation_verbs_is_idempotent() -> None:
    """Seeding relation verbs twice does not duplicate edges."""
    net = ConceptNetwork()
    net.seed_relation_verbs()
    edge_count = len(net.edges)
    net.seed_relation_verbs()
    assert len(net.edges) == edge_count

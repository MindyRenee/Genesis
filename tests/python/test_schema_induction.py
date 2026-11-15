"""Tests for schema induction — reification of shared relational structure."""

from __future__ import annotations

import pytest
from temporal_lobe.concepts import (
    ConceptNetwork,
    RelationType,
    SchemaInducer,
)


@pytest.fixture
def stress_network() -> ConceptNetwork:
    """A network where several concepts share a relation to one endpoint."""
    net = ConceptNetwork()
    for name in (
        "caffeine",
        "exercise",
        "morning",
        "alertness",
        "cortisol",
        "deadlines",
        "noise",
        "stress",
        "memory",
        "sleep",
    ):
        net.add_concept(name, confidence=0.8, origin="test")
    # Three things cause alertness -> "causes of alertness" schema.
    net.add_edge("caffeine", "alertness", RelationType.CAUSES, 0.7, origin="observed")
    net.add_edge("exercise", "alertness", RelationType.CAUSES, 0.6, origin="observed")
    net.add_edge("morning", "alertness", RelationType.CAUSES, 0.5, origin="observed")
    # Three things cause stress -> "causes of stress" schema.
    net.add_edge("cortisol", "stress", RelationType.CAUSES, 0.8, origin="observed")
    net.add_edge("deadlines", "stress", RelationType.CAUSES, 0.7, origin="observed")
    net.add_edge("noise", "stress", RelationType.CAUSES, 0.6, origin="observed")
    # Two only — below the min_members floor, must not reify.
    net.add_edge("stress", "memory", RelationType.HARMS, 0.7, origin="observed")
    net.add_edge("cortisol", "memory", RelationType.HARMS, 0.5, origin="observed")
    # Shared source: stress harms three things -> "things stress harms".
    net.add_edge("stress", "memory", RelationType.HARMS, 0.7, origin="observed")
    net.add_edge("stress", "sleep", RelationType.HARMS, 0.6, origin="observed")
    net.add_edge("stress", "alertness", RelationType.HARMS, 0.4, origin="observed")
    return net


class TestDiscovery:
    """Discovery finds shared relational structure without writing."""

    def test_finds_shared_target_schema(self, stress_network: ConceptNetwork):
        inducer = SchemaInducer(stress_network)
        schemas = inducer.discover_schemas()
        hubs = {s.hub_id for s in schemas}
        assert "_gen:causes:to:stress" in hubs
        assert "_gen:causes:to:alertness" in hubs

    def test_finds_shared_source_schema(self, stress_network: ConceptNetwork):
        inducer = SchemaInducer(stress_network)
        schemas = inducer.discover_schemas()
        hubs = {s.hub_id for s in schemas}
        assert "_gen:harms:from:stress" in hubs

    def test_below_floor_not_reified(self, stress_network: ConceptNetwork):
        """Two members is not a class — no schema for harms->memory."""
        inducer = SchemaInducer(stress_network)
        schemas = inducer.discover_schemas()
        assert "_gen:harms:to:memory" not in {s.hub_id for s in schemas}

    def test_association_does_not_induce(self, stress_network: ConceptNetwork):
        """RELATED_TO carries no class structure — it never induces."""
        stress_network.add_edge("memory", "sleep", RelationType.RELATED_TO, 0.9, origin="test")
        stress_network.add_edge("memory", "alertness", RelationType.RELATED_TO, 0.9, origin="test")
        stress_network.add_edge("memory", "stress", RelationType.RELATED_TO, 0.9, origin="test")
        inducer = SchemaInducer(stress_network)
        hubs = {s.hub_id for s in inducer.discover_schemas()}
        assert "_gen:related_to:from:memory" not in hubs

    def test_is_a_does_not_induce(self, stress_network: ConceptNetwork):
        """IS_A's target is already the class — reifying it is redundant."""
        for n in ("dog", "cat", "horse"):
            stress_network.add_concept(n, confidence=0.8, origin="test")
            stress_network.add_edge(n, "stress", RelationType.IS_A, 0.8, origin="test")
        inducer = SchemaInducer(stress_network)
        hubs = {s.hub_id for s in inducer.discover_schemas()}
        assert "_gen:is_a:to:stress" not in hubs

    def test_internal_namespaces_excluded(self, stress_network: ConceptNetwork):
        """Code symbols and utterance seeds are not schema material."""
        for i in range(4):
            stress_network.add_edge(
                f"python:mod.fn{i}",
                "python:np.savez",
                RelationType.DEPENDS_ON,
                0.9,
                origin="observed",
            )
        inducer = SchemaInducer(stress_network)
        hubs = {s.hub_id for s in inducer.discover_schemas()}
        assert not any("python:" in h for h in hubs)

    def test_members_ranked_by_weight(self, stress_network: ConceptNetwork):
        inducer = SchemaInducer(stress_network)
        schema = next(s for s in inducer.discover_schemas() if s.hub_id == "_gen:causes:to:stress")
        assert schema.members[0] == "cortisol"  # highest member weight
        assert set(schema.members) == {"cortisol", "deadlines", "noise"}


class TestAcceptance:
    """Acceptance writes hubs, membership edges, and schema edges."""

    def test_writes_hub_and_members(self, stress_network: ConceptNetwork):
        inducer = SchemaInducer(stress_network)
        added = inducer.discover_and_accept()
        assert added > 0
        net = stress_network
        assert net.get_concept("_gen:causes:to:stress") is not None
        # Members folded under the hub.
        for m in ("cortisol", "deadlines", "noise"):
            out = {(e.target, e.relation) for e in net.get_edges(m, "out")}
            assert ("_gen:causes:to:stress", RelationType.IS_A) in out
        # The compressed claim: the class itself bears the relation.
        hub_edges = net.get_edges("_gen:causes:to:stress", "out")
        assert any(e.target == "stress" and e.relation == RelationType.CAUSES for e in hub_edges)

    def test_shared_source_schema_edge(self, stress_network: ConceptNetwork):
        inducer = SchemaInducer(stress_network)
        inducer.discover_and_accept()
        edges = stress_network.get_edges("stress", "out")
        assert any(
            e.target == "_gen:harms:from:stress" and e.relation == RelationType.HARMS for e in edges
        )

    def test_reified_origin(self, stress_network: ConceptNetwork):
        """Everything induction writes carries origin='reified'."""
        SchemaInducer(stress_network).discover_and_accept()
        for e in stress_network.get_edges("cortisol", "out"):
            if e.target == "_gen:causes:to:stress":
                assert e.origin == "reified"
                return
        pytest.fail("member edge not found")

    def test_idempotent_second_pass(self, stress_network: ConceptNetwork):
        """A second pass refreshes rather than duplicating edges."""
        inducer = SchemaInducer(stress_network)
        inducer.discover_and_accept()
        n_edges_1 = len(stress_network._edges)
        inducer.discover_and_accept()
        n_edges_2 = len(stress_network._edges)
        assert n_edges_1 == n_edges_2

    def test_no_is_a_cycle(self, stress_network: ConceptNetwork):
        """A member that is already an ancestor of the hub is skipped."""
        # Make 'stress' a kind of the (future) hub before induction.
        stress_network.add_concept("_gen:causes:to:stress", origin="test")
        stress_network.add_edge(
            "_gen:causes:to:stress",
            "cortisol",
            RelationType.IS_A,
            0.9,
            origin="test",
        )
        inducer = SchemaInducer(stress_network)
        inducer.discover_and_accept()
        # cortisol -IS_A-> hub would close hub -IS_A-> cortisol -IS_A-> hub.
        cortisol_edges = stress_network.get_edges("cortisol", "out")
        assert not any(
            e.target == "_gen:causes:to:stress" and e.relation == RelationType.IS_A
            for e in cortisol_edges
        )

    def test_second_order_schemas(self, stress_network: ConceptNetwork):
        """Reified hubs can themselves become members of schemas."""
        # Three schemas that all CAUSE the same endpoint -> a schema
        # over schemas on a second pass.
        inducer = SchemaInducer(stress_network)
        inducer.discover_and_accept()
        for h in ("_gen:causes:to:stress", "_gen:causes:to:alertness", "_gen:harms:from:stress"):
            stress_network.add_edge(h, "memory", RelationType.HARMS, 0.8, origin="test")
        schemas = inducer.discover_schemas()
        hub = next((s for s in schemas if s.hub_id == "_gen:harms:to:memory"), None)
        assert hub is not None
        gen_members = {
            "_gen:causes:to:stress",
            "_gen:causes:to:alertness",
            "_gen:harms:from:stress",
        }
        assert gen_members <= set(hub.members)

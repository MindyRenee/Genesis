"""Sleep extras — dream synthesis, dream concept parsing, sleep aid."""

import logging
import os
import tempfile
from dataclasses import dataclass

import pytest

from genesis_client.types import NeuroSummary, RecentEpisode
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.mind import Mind
from genesis_cognitive.reasoning import ReasoningEngine, ReasoningType
from genesis_cognitive.sleep import (
    DreamInsight,
    DreamProposal,
    DreamSynthesisEngine,
)

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_dream_synthesis.py
# ======================================================================

@pytest.fixture
def network() -> ConceptNetwork:
    """A small network with structural similarity but unconnected pairs."""
    net = ConceptNetwork()
    # Two IS_A hierarchies: animals and machines
    # dog is_a animal, cat is_a animal (structurally similar: both leaves of animal)
    net.add_edge("dog", "animal", RelationType.IS_A, weight=0.9)
    net.add_edge("cat", "animal", RelationType.IS_A, weight=0.9)
    # car is_a vehicle, truck is_a vehicle (structurally similar: both leaves of vehicle)
    net.add_edge("car", "vehicle", RelationType.IS_A, weight=0.9)
    net.add_edge("truck", "vehicle", RelationType.IS_A, weight=0.9)
    # dog and cat are not connected to each other (no SIMILAR_TO edge)
    # car and truck are not connected to each other
    return net


@pytest.fixture
def causal_network() -> ConceptNetwork:
    """A network with CAUSES chains for transitive validation."""
    net = ConceptNetwork()
    # A causes B, B causes C (transitive chain)
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.9)
    net.add_edge("heat", "burn", RelationType.CAUSES, weight=0.9)
    # D causes E, E causes F (another chain)
    net.add_edge("friction", "heat", RelationType.CAUSES, weight=0.8)
    # fire and friction are structurally similar (both cause heat)
    # but not connected to each other
    return net


@pytest.fixture
def contradiction_network() -> ConceptNetwork:
    """A network with a contradiction edge."""
    net = ConceptNetwork()
    net.add_edge("hot", "temperature", RelationType.IS_A, weight=0.9)
    net.add_edge("cold", "temperature", RelationType.IS_A, weight=0.9)
    # hot and cold are opposites
    net.add_edge("hot", "cold", RelationType.OPPOSITE_OF, weight=1.0)
    return net


class TestDreamSynthesis:
    """Tests for the DreamSynthesisEngine.synthesize method."""

    def test_finds_structurally_similar_unconnected_pairs(self, network) -> None:
        """Synthesis should find unconnected pairs with similar structure."""
        # Before synthesis, dog-cat should have no direct edge
        assert not _has_direct_edge(network, "dog", "cat", RelationType.SIMILAR_TO)

        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=10)

        # Should find at least some proposals (dog-cat, car-truck)
        assert len(proposals) > 0

        # All proposed edges should now exist as dream-synthesis edges
        for p in proposals:
            assert _has_direct_edge_with_origin(
                network, p.source, p.target, p.relation, "dream-synthesis"
            )

    def test_proposes_similar_to_for_is_a_leaves(self, network):
        """IS_A leaves should get SIMILAR_TO proposals."""
        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=10)

        # At least one proposal should be SIMILAR_TO
        similar_proposals = [p for p in proposals if p.relation == RelationType.SIMILAR_TO]
        assert len(similar_proposals) > 0

    def test_stores_hypothetical_edges_in_network(self, network):
        """Synthesis should add hypothetical edges to the network."""
        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=10)

        # The network should now have dream-synthesis edges
        dream_edges = [
            e for e in network.edges
            if e.origin == "dream-synthesis"
        ]
        assert len(dream_edges) == len(proposals)

        # All dream edges should have low weight
        for edge in dream_edges:
            assert edge.weight <= 0.2

    def test_empty_network_returns_no_proposals(self):
        """An empty network should produce no proposals."""
        net = ConceptNetwork()
        engine = DreamSynthesisEngine(net)
        proposals = engine.synthesize()
        assert proposals == []

    def test_single_concept_returns_no_proposals(self):
        """A network with one concept should produce no proposals."""
        net = ConceptNetwork()
        net.add_edge("dog", "animal", RelationType.IS_A, weight=0.9)
        engine = DreamSynthesisEngine(net)
        proposals = engine.synthesize()
        assert proposals == []

    def test_connected_pairs_are_skipped(self, network):
        """Pairs that already have an edge should not be proposed."""
        # Connect dog and cat with a non-dream edge
        network.add_edge("dog", "cat", RelationType.SIMILAR_TO, weight=0.8, origin="stated")
        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=10)

        # No proposal should be between dog and cat (already connected)
        for p in proposals:
            assert not (p.source == "dog" and p.target == "cat")
            assert not (p.source == "cat" and p.target == "dog")

    def test_low_similarity_pairs_are_skipped(self):
        """Pairs with very different structural signatures should not be proposed."""
        net = ConceptNetwork()
        # dog is_a animal (outgoing IS_A only)
        net.add_edge("dog", "animal", RelationType.IS_A, weight=0.9)
        # fire causes smoke (outgoing CAUSES only) — completely different signature
        net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.9)
        engine = DreamSynthesisEngine(net)
        proposals = engine.synthesize(max_proposals=10)
        # dog and fire have no shared structural signature → no proposal
        for p in proposals:
            assert not (p.source == "dog" and p.target == "fire")
            assert not (p.source == "fire" and p.target == "dog")

    def test_max_proposals_limit(self, network):
        """Synthesis should respect the max_proposals limit."""
        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=2)
        assert len(proposals) <= 2


class TestDreamValidation:
    """Tests for the DreamSynthesisEngine.validate method."""

    def test_promotes_edge_with_sufficient_corroborations(self, network):
        """An edge with >= 2 corroborating paths should be promoted."""
        # Add corroborating evidence for dog SIMILAR_TO cat:
        # 1. Analogical transfer: dog similar_to wolf, wolf similar_to cat
        network.add_edge("dog", "wolf", RelationType.SIMILAR_TO, weight=0.8)
        network.add_edge("wolf", "cat", RelationType.SIMILAR_TO, weight=0.7)
        # 2. Shared hub: both dog and cat have SIMILAR_TO edges to "pet"
        network.add_edge("dog", "pet", RelationType.SIMILAR_TO, weight=0.6)
        network.add_edge("cat", "pet", RelationType.SIMILAR_TO, weight=0.6)

        engine = DreamSynthesisEngine(network)
        proposal = DreamProposal(
            source="dog",
            target="cat",
            relation=RelationType.SIMILAR_TO,
            structural_similarity=0.5,
            predicted_from="test",
        )
        results = engine.validate([proposal])

        assert len(results) == 1
        assert results[0].promoted
        assert results[0].corroborations >= 2

    def test_drops_contradicted_edge(self, contradiction_network):
        """An edge contradicted by OPPOSITE_OF should be dropped."""
        engine = DreamSynthesisEngine(contradiction_network)
        proposal = DreamProposal(
            source="hot",
            target="cold",
            relation=RelationType.IS_A,
            structural_similarity=0.5,
            predicted_from="test",
        )
        results = engine.validate([proposal])

        assert len(results) == 1
        assert results[0].contradicted
        assert not results[0].promoted

    def test_insufficient_evidence_remains_hypothetical(self, network):
        """An edge with < 2 corroborations should remain hypothetical."""
        engine = DreamSynthesisEngine(network)
        proposal = DreamProposal(
            source="dog",
            target="cat",
            relation=RelationType.SIMILAR_TO,
            structural_similarity=0.5,
            predicted_from="test",
        )
        results = engine.validate([proposal])

        assert len(results) == 1
        assert not results[0].promoted
        assert not results[0].contradicted
        assert results[0].corroborations < 2

    def test_transitive_chain_provides_corroboration(self, causal_network):
        """A transitive CAUSES chain should count as corroboration."""
        engine = DreamSynthesisEngine(causal_network)
        # Propose fire CAUSES burn (supported by fire→heat→burn chain)
        proposal = DreamProposal(
            source="fire",
            target="burn",
            relation=RelationType.CAUSES,
            structural_similarity=0.5,
            predicted_from="test",
        )
        results = engine.validate([proposal])

        # Should have at least 1 corroboration from the transitive chain
        # (fire→heat→burn). May also have shared hub corroboration.
        assert results[0].corroborations >= 1

    def test_promoted_edge_gets_higher_weight(self, network):
        """A promoted edge should have higher weight than hypothetical."""
        # Add corroborating evidence
        network.add_edge("dog", "wolf", RelationType.SIMILAR_TO, weight=0.8)
        network.add_edge("wolf", "cat", RelationType.SIMILAR_TO, weight=0.7)
        network.add_edge("dog", "pet", RelationType.SIMILAR_TO, weight=0.6)
        network.add_edge("cat", "pet", RelationType.SIMILAR_TO, weight=0.6)

        engine = DreamSynthesisEngine(network)
        # First synthesize to create the hypothetical edge
        engine.synthesize(max_proposals=10)
        # Then validate
        results = engine.review_pending_proposals()

        # At least one should be promoted
        promoted = [r for r in results if r.promoted]
        if promoted:
            # Check the edge weight was increased
            for r in promoted:
                edge = _find_edge(
                    network, r.proposal.source, r.proposal.target, r.proposal.relation
                )
                assert edge is not None
                assert edge.weight >= 0.4  # promoted weight

    def test_validated_insights_are_tracked(self, network):
        """Validated edges should be tracked as DreamInsights."""
        # Add corroborating evidence
        network.add_edge("dog", "wolf", RelationType.SIMILAR_TO, weight=0.8)
        network.add_edge("wolf", "cat", RelationType.SIMILAR_TO, weight=0.7)
        network.add_edge("dog", "pet", RelationType.SIMILAR_TO, weight=0.6)
        network.add_edge("cat", "pet", RelationType.SIMILAR_TO, weight=0.6)

        engine = DreamSynthesisEngine(network)
        proposal = DreamProposal(
            source="dog",
            target="cat",
            relation=RelationType.SIMILAR_TO,
            structural_similarity=0.5,
            predicted_from="test",
        )
        engine.validate([proposal])

        assert engine.insight_count >= 1
        insights = engine.insights
        assert all(isinstance(i, DreamInsight) for i in insights)


class TestDreamReview:
    """Tests for the review_pending_proposals method."""

    def test_review_finds_pending_edges(self, network):
        """Review should find and validate all dream-synthesis edges."""
        engine = DreamSynthesisEngine(network)
        engine.synthesize(max_proposals=10)

        results = engine.review_pending_proposals()
        # Should have results for all pending proposals
        assert len(results) > 0

    def test_review_no_pending_returns_empty(self, network):
        """Review with no pending edges should return empty list."""
        engine = DreamSynthesisEngine(network)
        results = engine.review_pending_proposals()
        assert results == []


class TestDreamPersistence:
    """Tests for serialization and restoration."""

    def test_to_dict_and_restore(self, network):
        """State should survive a to_dict/restore_from_dict round-trip."""
        engine = DreamSynthesisEngine(network)
        # Add a fake insight
        engine._validated_insights.append(DreamInsight(
            source="dog",
            target="cat",
            relation=RelationType.SIMILAR_TO,
            corroborations=3,
            insight_text="test insight",
        ))
        engine._proposed_count = 5
        engine._rejected_count = 2

        data = engine.to_dict()
        engine2 = DreamSynthesisEngine(network)
        engine2.restore_from_dict(data)

        assert engine2.insight_count == 1
        assert engine2.insights[0].source == "dog"
        assert engine2.insights[0].target == "cat"
        assert engine2.insights[0].corroborations == 3
        assert engine2.proposed_count == 5
        assert engine2.rejected_count == 2

    def test_restore_empty_data(self, network):
        """Restoring from empty data should not crash."""
        engine = DreamSynthesisEngine(network)
        engine.restore_from_dict({})
        assert engine.insight_count == 0
        assert engine.proposed_count == 0


class TestDreamIntegration:
    """Integration tests for the full synthesis -> validation loop."""

    def test_full_synthesis_validation_loop(self, network):
        """The full loop: synthesize -> validate -> check insights."""
        # Add corroborating evidence for dog-cat SIMILAR_TO
        network.add_edge("dog", "wolf", RelationType.SIMILAR_TO, weight=0.8)
        network.add_edge("wolf", "cat", RelationType.SIMILAR_TO, weight=0.7)
        network.add_edge("dog", "pet", RelationType.SIMILAR_TO, weight=0.6)
        network.add_edge("cat", "pet", RelationType.SIMILAR_TO, weight=0.6)

        engine = DreamSynthesisEngine(network)
        # Synthesize
        proposals = engine.synthesize(max_proposals=10)
        assert len(proposals) > 0

        # Validate
        results = engine.validate(proposals)

        # The dog-cat edge should have analogical transfer + shared hub
        dog_cat_results = [
            r for r in results
            if r.proposal.source == "dog" and r.proposal.target == "cat"
        ]
        if dog_cat_results:
            assert dog_cat_results[0].corroborations >= 1

    def test_stats_tracking(self, network):
        """Stats should track proposed, validated, and rejected counts."""
        engine = DreamSynthesisEngine(network)
        engine.synthesize(max_proposals=10)
        initial_proposed = engine.proposed_count
        assert initial_proposed > 0

        engine.review_pending_proposals()
        # proposed_count should not change after review
        assert engine.proposed_count == initial_proposed


# ─── Helpers ─────────────────────────────────────────────────────


def _has_direct_edge(
    network: ConceptNetwork, source: str, target: str, relation: RelationType
) -> bool:
    """Check if a specific edge exists in the network."""
    for edge in network.get_edges(source, "out"):
        if edge.target == target and edge.relation == relation:
            return True
    return False


def _has_direct_edge_with_origin(
    network: ConceptNetwork, source: str, target: str, relation: RelationType, origin: str
) -> bool:
    """Check if a specific edge with a given origin exists in the network."""
    for edge in network.get_edges(source, "out"):
        if edge.target == target and edge.relation == relation and edge.origin == origin:
            return True
    return False


def _find_edge(
    network: ConceptNetwork, source: str, target: str, relation: RelationType
) -> object | None:
    """Find a specific edge in the network."""
    for edge in network.get_edges(source, "out"):
        if edge.target == target and edge.relation == relation:
            return edge
    return None


# ─── End-to-end: dream synthesis improves reasoning ──────────────


def _build_base_network() -> ConceptNetwork:
    """Build a network with structural similarity but a missing connection.

    fire and star have identical structural signatures:
    - Both are IS_A leaves of "phenomenon"
    - Both have outgoing CAUSES edges
    - But there's NO edge between them

    This means reason_about("star") cannot use analogical transfer
    to infer that star might cause heat (it would need
    star similar_to fire, fire causes heat).
    """
    net = ConceptNetwork()

    # Structural: both are IS_A leaves of the same hub
    net.add_edge("fire", "phenomenon", RelationType.IS_A, weight=0.9)
    net.add_edge("star", "phenomenon", RelationType.IS_A, weight=0.9)

    # Causal: both have outgoing CAUSES edges
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.9)
    net.add_edge("fire", "light", RelationType.CAUSES, weight=0.9)
    net.add_edge("star", "light", RelationType.CAUSES, weight=0.9)

    # No edge between fire and star — they're unconnected
    return net


def _add_corroborating_evidence(net: ConceptNetwork) -> None:
    """Add evidence that corroborates the star-fire SIMILAR_TO edge.

    This represents knowledge encountered during waking hours that
    provides independent support for the dream-synthesized connection.
    The evidence is added AFTER synthesis (during "wake") so it doesn't
    change the structural signatures that synthesis relies on.

    Two corroborating paths (works for either edge direction):
    1. Analogical transfer: fire similar_to sun, sun similar_to star
       (and star similar_to sun, sun similar_to fire)
    2. Shared hub: both star and fire have SIMILAR_TO edges to sun
    """
    net.add_edge("star", "sun", RelationType.SIMILAR_TO, weight=0.8)
    net.add_edge("sun", "star", RelationType.SIMILAR_TO, weight=0.7)
    net.add_edge("sun", "fire", RelationType.SIMILAR_TO, weight=0.7)
    net.add_edge("fire", "sun", RelationType.SIMILAR_TO, weight=0.6)


def _has_analogical_heat_insight(engine: ReasoningEngine) -> bool:
    """Check if the reasoning engine can infer 'star might cause heat'.

    This requires analogical transfer: star similar_to X, X causes heat.
    Without a SIMILAR_TO edge from star to something that causes heat,
    this inference is impossible.
    """
    results = engine.reason_about("star")
    for r in results:
        if r.reasoning_type == ReasoningType.ANALOGICAL:
            if "heat" in r.conclusion.lower():
                return True
    return False


class TestDreamSynthesisImprovesReasoning:
    """The end-to-end test: dream synthesis → validated edge → better reasoning."""

    def test_reasoning_fails_before_dream_synthesis(self):
        """Before dream synthesis, the reasoning engine cannot infer
        that star might cause heat (no SIMILAR_TO edge to transfer through)."""
        net = _build_base_network()
        engine = ReasoningEngine(net)

        assert not _has_analogical_heat_insight(engine), (
            "Reasoning engine should NOT be able to infer 'star causes heat' "
            "before dream synthesis — there's no SIMILAR_TO edge to transfer through"
        )

    def test_dream_synthesis_proposes_the_missing_edge(self):
        """Dream synthesis should find that star and fire are structurally
        similar (same signature: IS_A + CAUSES) and propose a SIMILAR_TO edge."""
        net = _build_base_network()
        dream = DreamSynthesisEngine(net)

        proposals = dream.synthesize(max_proposals=10)

        # Should propose a SIMILAR_TO edge between star and fire
        star_fire_proposals = [
            p for p in proposals
            if {p.source, p.target} == {"star", "fire"}
            and p.relation == RelationType.SIMILAR_TO
        ]
        assert len(star_fire_proposals) >= 1, (
            "Dream synthesis should propose a SIMILAR_TO edge between "
            "star and fire (they have identical structural signatures)"
        )

    def test_validation_promotes_edge_with_corroborating_evidence(self):
        """With enough corroborating evidence (encountered during wake),
        the dream-synthesized edge should be promoted to a real edge."""
        net = _build_base_network()

        # Sleep: synthesize proposals
        dream = DreamSynthesisEngine(net)
        dream.synthesize(max_proposals=10)

        # Wake: encounter corroborating evidence
        _add_corroborating_evidence(net)

        # Validate — should find ≥2 corroborations and promote
        results = dream.review_pending_proposals()

        promoted = [r for r in results if r.promoted]
        assert len(promoted) >= 1, (
            "At least one dream edge should be promoted with "
            "≥2 corroborating paths"
        )

    def test_reasoning_succeeds_after_dream_synthesis(self):
        """After dream synthesis + validation, the reasoning engine CAN
        infer that star might cause heat via analogical transfer through
        the dream-synthesized SIMILAR_TO edge."""
        net = _build_base_network()

        # Before dream synthesis: no analogical heat insight
        engine_before = ReasoningEngine(net)
        assert not _has_analogical_heat_insight(engine_before), (
            "Before dream synthesis, reasoning should NOT produce "
            "an analogical insight about star and heat"
        )

        # Sleep: run dream synthesis
        dream = DreamSynthesisEngine(net)
        dream.synthesize(max_proposals=10)

        # Wake: encounter corroborating evidence
        _add_corroborating_evidence(net)

        # Validate dream proposals
        dream.review_pending_proposals()

        # After dream synthesis: the analogical heat insight should exist
        engine_after = ReasoningEngine(net)
        assert _has_analogical_heat_insight(engine_after), (
            "After dream synthesis, reasoning SHOULD produce an analogical "
            "insight: 'star might cause heat, by analogy with fire' — "
            "the dream-synthesized SIMILAR_TO edge enables this transfer"
        )

    def test_the_full_loop_end_to_end(self):
        """The complete end-to-end loop:

        1. Build network with a knowledge gap
        2. Confirm reasoning can't bridge the gap
        3. Sleep: dream synthesis proposes a novel connection
        4. Wake: new evidence corroborates the proposal
        5. Validation promotes it (≥2 corroborating paths)
        6. Confirm reasoning can now bridge the gap
        7. The insight is tracked as a DreamInsight

        This is the causal chain that makes the upgrade plan's thesis
        testable: sleep → synthesis → novel connection → validation →
        measurable reasoning improvement.
        """
        # ── 1. Build network with a knowledge gap ──────────────────
        net = _build_base_network()

        # ── 2. Confirm reasoning can't bridge the gap ──────────────
        engine_before = ReasoningEngine(net)
        results_before = engine_before.reason_about("star")
        analogical_before = [
            r for r in results_before
            if r.reasoning_type == ReasoningType.ANALOGICAL
            and "heat" in r.conclusion.lower()
        ]
        assert len(analogical_before) == 0, (
            "Before dream synthesis: no analogical insight about star→heat"
        )

        # ── 3. Sleep: dream synthesis proposes a novel connection ──
        dream = DreamSynthesisEngine(net)
        proposals = dream.synthesize(max_proposals=10)
        assert len(proposals) > 0, "Dream synthesis should produce proposals"

        star_fire = [
            p for p in proposals
            if {p.source, p.target} == {"star", "fire"}
        ]
        assert len(star_fire) >= 1, (
            "Dream synthesis should propose a star-fire connection"
        )

        # ── 4. Wake: new evidence corroborates the proposal ────────
        _add_corroborating_evidence(net)

        # ── 5. Validation promotes it ──────────────────────────────
        validation_results = dream.review_pending_proposals()
        promoted = [r for r in validation_results if r.promoted]
        assert len(promoted) >= 1, (
            "At least one dream edge should be validated and promoted"
        )

        star_fire_promoted = [
            r for r in promoted
            if {r.proposal.source, r.proposal.target} == {"star", "fire"}
        ]
        assert len(star_fire_promoted) >= 1, (
            "The star-fire SIMILAR_TO edge should be promoted"
        )

        # ── 6. Confirm reasoning can now bridge the gap ────────────
        engine_after = ReasoningEngine(net)
        results_after = engine_after.reason_about("star")
        analogical_after = [
            r for r in results_after
            if r.reasoning_type == ReasoningType.ANALOGICAL
            and "heat" in r.conclusion.lower()
        ]
        assert len(analogical_after) > 0, (
            "After dream synthesis: analogical insight about star→heat "
            "should exist — 'star might cause heat, by analogy with fire'"
        )

        the_insight = analogical_after[0]
        assert "fire" in the_insight.conclusion.lower(), (
            f"The analogical insight should reference fire as the "
            f"analogical source: {the_insight.conclusion}"
        )

        # ── 7. The insight is tracked as a DreamInsight ────────────
        insights = dream.insights
        star_fire_insights = [
            i for i in insights
            if {i.source, i.target} == {"star", "fire"}
        ]
        assert len(star_fire_insights) >= 1, (
            "The star-fire edge should be tracked as a DreamInsight"
        )
        assert star_fire_insights[0].corroborations >= 2, (
            "The insight should have ≥2 corroborating paths"
        )

    def test_analogical_reasoning_count_increases(self):
        """The number of analogical reasoning results should increase
        after dream synthesis, proving that the dream edge created
        new reasoning paths that didn't exist before."""
        net = _build_base_network()

        # Before: count analogical results
        engine_before = ReasoningEngine(net)
        before_results = engine_before.reason_about("star")
        before_analogical = [
            r for r in before_results
            if r.reasoning_type == ReasoningType.ANALOGICAL
        ]

        # Sleep + wake + validate
        dream = DreamSynthesisEngine(net)
        dream.synthesize(max_proposals=10)
        _add_corroborating_evidence(net)
        dream.review_pending_proposals()

        # After: count analogical results
        engine_after = ReasoningEngine(net)
        after_results = engine_after.reason_about("star")
        after_analogical = [
            r for r in after_results
            if r.reasoning_type == ReasoningType.ANALOGICAL
        ]

        assert len(after_analogical) > len(before_analogical), (
            f"Analogical reasoning results should increase after dream "
            f"synthesis: before={len(before_analogical)}, "
            f"after={len(after_analogical)}"
        )


# ======================================================================
# From tests/test_dream_concept_parsing.py
# ======================================================================

@dataclass
class _MockEpisode:
    """Tests for mock episode."""
    episode_id: int
    text: str


class _MockDreamClient:
    """Mock client that returns episode text for known IDs.

    Simulates deleted episodes by raising EpisodeNotFound for unknown IDs.
    """

    def __init__(self, episodes: dict[int, str]) -> None:
        """Initialize the mock dream client."""
        self._episodes = episodes

    def retrieve_episode(self, ep_id: int, **kwargs):
        """Retrieve episode."""
        if ep_id in self._episodes:
            return _MockEpisode(episode_id=ep_id, text=self._episodes[ep_id])
        from genesis_client.exceptions import EpisodeNotFound
        raise EpisodeNotFound(f"episode {ep_id} not found")


class _MockDreamNetwork:
    """Mock concept network with a known set of world concept IDs."""

    def __init__(self, concepts: list[str]) -> None:
        """Initialize the mock dream network."""
        self.world_concept_ids = concepts


class _MockDreamCognition:
    """Mock cognition with a network."""

    def __init__(self, concepts: list[str]) -> None:
        """Initialize the mock dream cognition."""
        self.network = _MockDreamNetwork(concepts)


def _make_mind(episodes: dict[int, str], concepts: list[str]) -> Mind:
    """Create a Mind with mock dependencies for dream concept testing."""
    import tempfile

    data_dir = tempfile.mkdtemp(prefix="dream_test_")
    socket_path = os.path.join(data_dir, "genesis.sock")
    mind = Mind(socket_path)
    mind.client = _MockDreamClient(episodes)  # type: ignore[assignment]
    mind.cognition = _MockDreamCognition(concepts)  # type: ignore[assignment]
    return mind


# ─── Test cases ───────────────────────────────────────────────


def test_new_format_extracts_concepts_from_embedded_text():
    """Dream insights with embedded text should extract concepts without
    retrieving episodes by ID."""
    # The dream insight embeds episode text directly (tab-delimited).
    # No episode retrieval should be needed.
    episodes: dict[int, str] = {}  # empty — no episodes to retrieve
    concepts = ["dopamine", "memory", "learning"]

    mind = _make_mind(episodes, concepts)

    dream = RecentEpisode(
        episode_id=100,
        timestamp=1000,
        salience=0.7,
        event_type=2,
        source_module=9,
        text=(
            "[dream-insight] episode 1 connects to episode 2 "
            "through 3 hops (direct distance: 42)\t"
            "Learning about dopamine and memory formation\t"
            "Studying learning and dopamine pathways"
        ),
    )

    result = mind._collect_dream_concepts([dream])
    assert "dopamine" in result, f"should extract 'dopamine' from embedded text: {result}"
    assert "memory" in result, f"should extract 'memory' from embedded text: {result}"
    assert "learning" in result, f"should extract 'learning' from embedded text: {result}"


def test_old_format_falls_back_to_episode_retrieval():
    """Old-format dream insights (no tab) should fall back to ID retrieval."""
    episodes = {
        1: "Learning about dopamine and memory formation",
        2: "Studying learning and dopamine pathways",
    }
    concepts = ["dopamine", "memory", "learning"]

    mind = _make_mind(episodes, concepts)

    dream = RecentEpisode(
        episode_id=100,
        timestamp=1000,
        salience=0.7,
        event_type=2,
        source_module=9,
        text=(
            "[dream-insight] episode 1 connects to episode 2 "
            "through 3 hops (direct distance: 42)"
        ),
    )

    result = mind._collect_dream_concepts([dream])
    assert "dopamine" in result, f"should extract 'dopamine' via retrieval: {result}"
    assert "memory" in result, f"should extract 'memory' via retrieval: {result}"


def test_deleted_episodes_do_not_crash():
    """When episodes are deleted (pruned by sleep compression),
    _collect_dream_concepts should not crash or log warnings."""
    # Old-format insight referencing deleted episodes
    episodes: dict[int, str] = {}  # all deleted
    concepts = ["dopamine"]

    mind = _make_mind(episodes, concepts)

    dream = RecentEpisode(
        episode_id=100,
        timestamp=1000,
        salience=0.7,
        event_type=2,
        source_module=9,
        text=(
            "[dream-insight] episode 999 connects to episode 998 "
            "through 3 hops (direct distance: 42)"
        ),
    )

    # Should return empty list, not crash
    result = mind._collect_dream_concepts([dream])
    assert result == [], f"should return empty list for deleted episodes: {result}"


def test_new_format_no_retrieval_needed_for_deleted_episodes():
    """New-format insights should work even when the referenced episodes
    have been deleted — the embedded text is self-contained."""
    episodes: dict[int, str] = {}  # episodes deleted
    concepts = ["quantum", "physics", "energy"]

    mind = _make_mind(episodes, concepts)

    dream = RecentEpisode(
        episode_id=100,
        timestamp=1000,
        salience=0.7,
        event_type=2,
        source_module=9,
        text=(
            "[dream-insight] episode 1 connects to episode 2 "
            "through 3 hops (direct distance: 42)\t"
            "Learning about quantum physics\t"
            "Studying energy and quantum mechanics"
        ),
    )

    result = mind._collect_dream_concepts([dream])
    assert "quantum" in result, (
        f"should extract from embedded text even if episodes deleted: {result}"
    )
    assert "physics" in result, f"should extract 'physics': {result}"
    assert "energy" in result, f"should extract 'energy': {result}"


def test_association_chain_with_embedded_text():
    """When a dream insight endpoint is itself an association trace with
    embedded text, the parser should extract concepts from the embedded
    text directly — the Rust side replaces tabs in embedded text with
    spaces, so the association trace's own embedded texts are part of
    the flat text and concept words are found by scanning it.
    """
    episodes: dict[int, str] = {}  # all deleted
    concepts = ["salt", "lake", "water", "drought"]

    mind = _make_mind(episodes, concepts)

    # Dream insight where one endpoint is an association trace.
    # The Rust side replaces tabs in embedded text with spaces, so
    # the association trace embedded as a dream insight endpoint has
    # its own tabs replaced. The concept words from the association's
    # endpoints are in the flat text.
    dream = RecentEpisode(
        episode_id=100,
        timestamp=1000,
        salience=0.7,
        event_type=2,
        source_module=9,
        text=(
            "[dream-insight] episode 1 connects to episode 2 "
            "through 3 hops (direct distance: 42)\t"
            "Direct memory about salt lake water\t"
            "[association] episode 3 ↔ episode 4 (distance: 15)  "
            "Learning about drought and water scarcity  "
            "Studying salt lake drying up"
        ),
    )

    result = mind._collect_dream_concepts([dream])
    assert "salt" in result, f"should find 'salt' in embedded text: {result}"
    assert "lake" in result, f"should find 'lake' in embedded text: {result}"
    assert "water" in result, f"should find 'water': {result}"
    assert "drought" in result, f"should find 'drought' in association text: {result}"


def test_mixed_old_and_new_format_insights():
    """A mix of old and new format insights should work correctly."""
    episodes = {
        10: "Learning about dopamine research",
    }
    concepts = ["dopamine", "quantum", "physics"]

    mind = _make_mind(episodes, concepts)

    dreams = [
        # New format — self-contained
        RecentEpisode(
            episode_id=100,
            timestamp=1000,
            salience=0.7,
            event_type=2,
            source_module=9,
            text=(
                "[dream-insight] episode 1 connects to episode 2 "
                "through 2 hops (direct distance: 30)\t"
                "Studying quantum physics\t"
                "Learning about physics and quantum mechanics"
            ),
        ),
        # Old format — needs retrieval (episode 10 exists, 11 doesn't)
        RecentEpisode(
            episode_id=101,
            timestamp=2000,
            salience=0.7,
            event_type=2,
            source_module=9,
            text=(
                "[dream-insight] episode 10 connects to episode 11 "
                "through 2 hops (direct distance: 35)"
            ),
        ),
    ]

    result = mind._collect_dream_concepts(dreams)
    assert "quantum" in result, f"should extract from new format: {result}"
    assert "physics" in result, f"should extract from new format: {result}"
    assert "dopamine" in result, f"should extract from old format (retrieved): {result}"


# ======================================================================
# From tests/test_sleep_aid.py
# ======================================================================

logger = logging.getLogger(__name__)


class _MockClient:
    """Mock GenesisClient that records neuro_impulse and set_zone calls.

    This lets us verify that sleep_aid() emits the correct
    neurochemical impulses without a running daemon.
    """

    def __init__(self) -> None:
        """Initialize the mock client."""
        self.impulses: list[tuple[int, float]] = []
        self.zones: list[int] = []

    def neuro_impulse(self, chem: int, magnitude: float, **kwargs) -> bool:
        """Neuro impulse."""
        self.impulses.append((chem, magnitude))
        return True

    def set_zone(self, zone: int, **kwargs) -> bool:
        """Set zone."""
        self.zones.append(zone)
        return True


def _make_mind_sleep_aid(data_dir: str) -> Mind:
    """Create a Mind with a mock client for testing sleep_aid."""
    socket_path = os.path.join(data_dir, "genesis.sock")
    mind = Mind(socket_path)
    # Replace the real client with a mock so we can record impulses
    # without needing a running daemon.
    mind.client = _MockClient()  # type: ignore[assignment]
    return mind


def test_sleep_aid_enters_sleep_state():
    """sleep_aid() should set _is_sleeping to True."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        assert not mind.is_sleeping
        mind.sleep_aid()
        assert mind.is_sleeping


def test_sleep_aid_emits_correct_impulses():
    """sleep_aid() should emit adenosine, cortisol, GABA, histamine,
    orexin, and serotonin impulses in the correct directions."""
    from genesis_client.protocol import (
        CHEM_ADENOSINE,
        CHEM_CORTISOL,
        CHEM_GABA,
        CHEM_HISTAMINE,
        CHEM_OREXIN,
        CHEM_SEROTONIN,
    )

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        mind.sleep_aid()
        mock = mind.client  # type: ignore[assignment]
        impulses = dict(mock.impulses)

        # Adenosine must be boosted (positive) to push past sleep threshold
        assert CHEM_ADENOSINE in impulses, "adenosine impulse missing"
        assert impulses[CHEM_ADENOSINE] > 0.0, "adenosine must be positive"

        # Cortisol must be suppressed (negative) to relieve BDNF inhibition
        assert CHEM_CORTISOL in impulses, "cortisol impulse missing"
        assert impulses[CHEM_CORTISOL] < 0.0, "cortisol must be negative"

        # GABA must be boosted (positive) for inhibitory surge
        assert CHEM_GABA in impulses, "GABA impulse missing"
        assert impulses[CHEM_GABA] > 0.0, "GABA must be positive"

        # Histamine must be suppressed (negative) to lower arousal
        assert CHEM_HISTAMINE in impulses, "histamine impulse missing"
        assert impulses[CHEM_HISTAMINE] < 0.0, "histamine must be negative"

        # Orexin must be suppressed (negative) to destabilize wake
        assert CHEM_OREXIN in impulses, "orexin impulse missing"
        assert impulses[CHEM_OREXIN] < 0.0, "orexin must be negative"

        # Serotonin must be boosted (positive) to support BDNF recovery
        assert CHEM_SEROTONIN in impulses, "serotonin impulse missing"
        assert impulses[CHEM_SEROTONIN] > 0.0, "serotonin must be positive"


def test_sleep_aid_adenosine_is_strong():
    """The adenosine boost must be strong enough to push past the 0.75
    sleep threshold from a typical baseline (~0.20). At least +0.30."""
    from genesis_client.protocol import CHEM_ADENOSINE

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        mind.sleep_aid()
        mock = mind.client  # type: ignore[assignment]
        adenosine_impulse = dict(mock.impulses)[CHEM_ADENOSINE]
        assert adenosine_impulse >= 0.30, (
            f"adenosine impulse too weak: {adenosine_impulse}, "
            "must be >= 0.30 to push past 0.75 sleep threshold"
        )


def test_sleep_aid_idempotent_when_sleeping():
    """If already sleeping, sleep_aid() should be a no-op."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        mind.sleep_aid()
        assert mind.is_sleeping
        mock1 = mind.client  # type: ignore[assignment]
        count_after_first = len(mock1.impulses)

        # Call again — should not emit more impulses
        mind.sleep_aid()
        assert len(mock1.impulses) == count_after_first, (
            "sleep_aid() should not emit impulses when already sleeping"
        )


def test_sleep_aid_wakes_from_meditation():
    """If meditating, sleep_aid() should wake from meditation first,
    then enter sleep."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        # Enter meditation first
        mind.meditate()
        assert mind.is_meditating
        assert not mind.is_sleeping

        # Now apply sleep aid
        mind.sleep_aid()
        assert not mind.is_meditating, "should have woken from meditation"
        assert mind.is_sleeping, "should be sleeping after sleep aid"


def test_sleep_aid_sets_sleeping_zone():
    """sleep_aid() should set the zone to SLEEPING."""
    from genesis_client.protocol import ZONE_SLEEPING

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_mind_sleep_aid(data_dir)
        mind.sleep_aid()
        mock = mind.client  # type: ignore[assignment]
        assert ZONE_SLEEPING in mock.zones, (
            f"zone SLEEPING ({ZONE_SLEEPING}) not set, "
            f"zones set: {mock.zones}"
        )


# ======================================================================
# Wake readiness — the nap timer should wait for it to be ready
# instead of forcing it awake mid-cycle.
# ======================================================================


@dataclass
class _MockPlasticityProfile:
    """Mock plasticity profile for wake_readiness receptor checks."""
    cortisol_tonic: float = 0.3
    bdnf_tonic: float = 0.5


class _WakeReadinessMockClient(_MockClient):
    """Mock client that supports get_state and get_plasticity_profile
    so wake_readiness() can check adenosine and receptor state.
    """

    def __init__(
        self,
        adenosine: float = 0.1,
        cortisol: float = 0.3,
        bdnf: float = 0.5,
    ) -> None:
        super().__init__()
        self._adenosine = adenosine
        self._profile = _MockPlasticityProfile(
            cortisol_tonic=cortisol, bdnf_tonic=bdnf
        )

    def get_state(self):
        """Return a mock core state with the configured adenosine."""
        from dataclasses import dataclass as _dc_dataclass

        @_dc_dataclass
        class _MockState:
            chemicals: dict[str, float]

        return _MockState(chemicals={"adenosine": self._adenosine})

    def get_plasticity_profile(self) -> _MockPlasticityProfile:
        """Return the configured plasticity profile."""
        return self._profile


def _make_wake_readiness_mind(
    data_dir: str,
    adenosine: float = 0.1,
    cortisol: float = 0.3,
    bdnf: float = 0.5,
) -> Mind:
    """Create a Mind with a mock client configured for wake_readiness tests."""
    socket_path = os.path.join(data_dir, "genesis.sock")
    mind = Mind(socket_path)
    mind.client = _WakeReadinessMockClient(  # type: ignore[assignment]
        adenosine=adenosine, cortisol=cortisol, bdnf=bdnf
    )
    return mind


def test_wake_readiness_returns_ready_when_awake():
    """wake_readiness() should return (True, ...) when not sleeping."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(data_dir)
        ready, reason = mind.wake_readiness()
        assert ready is True
        assert "already awake" in reason


def test_wake_readiness_blocks_on_n3():
    """wake_readiness() should return False when in N3 deep sleep."""
    from genesis_cognitive.brain_waves import SleepStage
    from genesis_cognitive.sleep.architecture import SleepCycleTracker

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(data_dir)
        mind._is_sleeping = True

        # Force the sleep cycle into N3
        cycle = SleepCycleTracker()
        cycle.current_stage = SleepStage.N3
        cycle._stage_slot = 2  # N3 is index 2 in _CYCLE_STAGES
        mind.inner_life._sleep_cycle = cycle  # type: ignore[attr-defined]

        ready, reason = mind.wake_readiness()
        assert ready is False
        assert "N3" in reason


def test_wake_readiness_blocks_on_high_adenosine():
    """wake_readiness() should return False when adenosine is still high."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(
            data_dir, adenosine=0.5  # well above AUTO_WAKE_ADENOSINE (0.20)
        )
        mind._is_sleeping = True

        ready, reason = mind.wake_readiness()
        assert ready is False
        assert "adenosine" in reason.lower()


def test_wake_readiness_blocks_on_high_cortisol():
    """wake_readiness() should return False when cortisol is still high."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(
            data_dir, adenosine=0.1, cortisol=0.8, bdnf=0.5
        )
        mind._is_sleeping = True

        ready, reason = mind.wake_readiness()
        assert ready is False
        assert "cortisol" in reason.lower()


def test_wake_readiness_blocks_on_low_bdnf():
    """wake_readiness() should return False when BDNF is still low."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(
            data_dir, adenosine=0.1, cortisol=0.3, bdnf=0.1
        )
        mind._is_sleeping = True

        ready, reason = mind.wake_readiness()
        assert ready is False
        assert "bdnf" in reason.lower()


def test_wake_readiness_ready_when_all_signals_pass():
    """wake_readiness() should return True when all signals indicate readiness."""
    from genesis_cognitive.brain_waves import SleepStage
    from genesis_cognitive.sleep.architecture import SleepCycleTracker

    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(
            data_dir, adenosine=0.1, cortisol=0.3, bdnf=0.5
        )
        mind._is_sleeping = True

        # Sleep cycle in REM (not N3)
        cycle = SleepCycleTracker()
        cycle.current_stage = SleepStage.REM
        cycle._stage_slot = 4  # REM is index 4
        mind.inner_life._sleep_cycle = cycle  # type: ignore[attr-defined]

        ready, reason = mind.wake_readiness()
        assert ready is True
        assert reason == "ready"


def test_wake_readiness_ready_with_no_sleep_cycle():
    """wake_readiness() should pass the stage check if no cycle is active."""
    with tempfile.TemporaryDirectory() as data_dir:
        mind = _make_wake_readiness_mind(
            data_dir, adenosine=0.1, cortisol=0.3, bdnf=0.5
        )
        mind._is_sleeping = True
        # No sleep cycle set — inner_life.sleep_cycle is None
        mind.inner_life._sleep_cycle = None  # type: ignore[attr-defined]

        ready, reason = mind.wake_readiness()
        assert ready is True
        assert reason == "ready"


# ======================================================================
# Lucid dream probability — REM stage from the ultradian cycle tracker
# ======================================================================


def _make_inner_life_for_lucid(seed: int = 42):
    """Build an InnerLife with a small network for lucid dream tests."""
    from genesis_cognitive import (
        ConceptNetwork,
        CuriosityEngine,
        ReasoningEngine,
        ReflectionEngine,
        RelationType,
    )
    from genesis_cognitive.sleep import InnerLife

    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.8)
    net.add_edge("beta", "gamma", RelationType.RELATED_TO, 0.5)
    reasoning = ReasoningEngine(net)
    curiosity = CuriosityEngine(net, reasoning)
    reflection = ReflectionEngine(net)
    return InnerLife(net, curiosity, reflection, seed=seed)


def _nrem_summary(arousal: float = 0.15) -> NeuroSummary:
    """A NeuroSummary simulating NREM sleep (low arousal)."""
    from genesis_client.protocol import PHASE_NREM

    return NeuroSummary(
        arousal=arousal,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.3,
        encoding_weight=0.3,
        consolidation_weight=0.8,
        retrieval_weight=0.3,
        phase=PHASE_NREM,
    )


def test_lucid_probability_rem_boost_from_tracker():
    """The tracker's is_rem should boost lucid probability even when the
    daemon arousal is low (NREM clamp 0.10-0.25).

    Before the fix, lucid_dream_probability used summary.arousal as the
    sole REM marker. During NREM, arousal is clamped to 0.10-0.25, so
    the probability was ~0.05 + 0.025 = 0.075 — too low for lucid
    dreaming to occur in practice. With the fix, the tracker's is_rem
    provides a +0.10 boost, giving ~0.15+ even with low daemon arousal.
    """
    il = _make_inner_life_for_lucid()
    summary = _nrem_summary(arousal=0.15)

    # Without the tracker's REM signal (old behavior)
    prob_no_rem = il.lucid_dream_probability(summary, is_rem=False)

    # With the tracker's REM signal (new behavior)
    prob_with_rem = il.lucid_dream_probability(summary, is_rem=True)

    assert prob_with_rem > prob_no_rem, (
        f"REM stage should boost lucid probability: "
        f"no_rem={prob_no_rem:.3f}, with_rem={prob_with_rem:.3f}"
    )
    # The REM boost should be at least 0.08 (the +0.10 base boost
    # minus the small arousal contribution from the fallback path)
    assert prob_with_rem - prob_no_rem >= 0.08, (
        f"REM boost too small: {prob_with_rem - prob_no_rem:.3f}"
    )
    # The probability with REM should be at least 0.12 (base 0.02
    # + REM 0.10 + minimal self-awareness)
    assert prob_with_rem >= 0.12, (
        f"Lucid probability with REM too low: {prob_with_rem:.3f}"
    )


def test_lucid_probability_rem_with_low_arousal():
    """Even with NREM-clamped arousal (0.10), the tracker's is_rem
    should give a meaningful lucid probability.

    This is the core regression: before the fix, the daemon's arousal
    was the only REM marker, and during NREM it's 0.10-0.25, making
    lucid dreaming nearly impossible. The tracker's is_rem should
    override this limitation.
    """
    il = _make_inner_life_for_lucid()
    summary = _nrem_summary(arousal=0.10)  # deep NREM

    prob = il.lucid_dream_probability(summary, is_rem=True)

    assert prob >= 0.12, (
        f"Lucid probability during tracker-REM with low arousal should "
        f"be >= 0.12, got {prob:.3f}"
    )


def test_lucid_probability_practice_effect():
    """Prior lucid dreams should increase the probability (practice effect)."""
    il = _make_inner_life_for_lucid()
    summary = _nrem_summary(arousal=0.15)

    # First lucid dream — no practice
    prob_first = il.lucid_dream_probability(summary, is_rem=True)

    # After 10 prior lucid dreams — practice effect
    il._lucid_dream_count = 10  # type: ignore[attr-defined]
    prob_practiced = il.lucid_dream_probability(summary, is_rem=True)

    assert prob_practiced > prob_first, (
        f"Practice should increase probability: "
        f"first={prob_first:.3f}, practiced={prob_practiced:.3f}"
    )


def test_lucid_probability_nrem_without_tracker_rem():
    """Without the tracker's REM signal, NREM arousal should give a
    low probability (the old behavior, preserved as fallback)."""
    il = _make_inner_life_for_lucid()
    summary = _nrem_summary(arousal=0.15)

    prob = il.lucid_dream_probability(summary, is_rem=False)

    # Without REM, the probability should be low (base + self-awareness)
    assert prob < 0.12, (
        f"Lucid probability without REM should be low, got {prob:.3f}"
    )

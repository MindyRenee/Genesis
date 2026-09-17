"""Tests for the sleep compression module."""


from dataclasses import dataclass

import pytest

from genesis_cognitive.concepts import (
    ConceptNetwork,
)
from genesis_cognitive.sleep import SleepCompressor


@dataclass
class _MockEpisode:
    """Minimal episode mock matching RecentEpisode's interface."""
    episode_id: int
    timestamp: int
    salience: float
    event_type: int
    source_module: int
    text: str


class _MockLTMClient:
    """Mock LTM client that simulates the daemon's 1000-episode page cap.

    The real daemon (ipc.rs) caps search_episodes at 1000 per call
    regardless of the requested limit. This mock reproduces that
    behavior so we can test that sleep compression paginates correctly.
    """

    def __init__(self, episodes: list[_MockEpisode]) -> None:
        """Initialize the mock l t m client."""
        self._episodes = episodes
        self.archived_ids: set[int] = set()
        self._page_cap = 1000

    def search_episodes(self, limit: int = 1000, offset: int = 0) -> list[_MockEpisode]:
        """Page through episodes with the daemon's 1000-cap."""
        capped = min(limit, self._page_cap)
        return self._episodes[offset : offset + capped]

    def archive_episode(self, episode_id: int) -> bool:
        """Archive episode."""
        self.archived_ids.add(episode_id)
        return True


@pytest.fixture
def small_network():
    """Create a small concept network with bridge and non-bridge edges."""
    net = ConceptNetwork()
    # Add some concepts
    for name in ["dog", "cat", "animal", "mammal", "bird", "fish", "reptile"]:
        net.add_concept(name, origin="stated")

    # Add "real" edges (stated origin)
    net.add_edge("dog", "is_a", "mammal", weight=0.9, origin="stated")
    net.add_edge("cat", "is_a", "mammal", weight=0.9, origin="stated")
    net.add_edge("mammal", "is_a", "animal", weight=0.9, origin="stated")
    net.add_edge("bird", "is_a", "animal", weight=0.9, origin="stated")
    net.add_edge("fish", "is_a", "animal", weight=0.9, origin="stated")

    # Add bridge edges (auto-generated)
    net.add_edge("dog", "related_to", "cat", weight=0.3, origin="semantic_bridge")
    net.add_edge("dog", "related_to", "bird", weight=0.2, origin="hub_attachment")
    net.add_edge("cat", "related_to", "fish", weight=0.15, origin="associative_bridge")
    net.add_edge("bird", "related_to", "reptile", weight=0.25, origin="semantic_bridge")

    return net


class TestSleepCompressor:
    """Tests for sleep compressor."""
    def test_init(self, tmp_path):
        """SleepCompressor should initialize correctly."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        assert sc.vq_codebook is not None
        assert sc.holographic_graph is not None
        assert not sc.vq_codebook.is_trained
        assert sc.holographic_graph.edge_count == 0

    def test_holographize_migrates_bridges(self, tmp_path, small_network):
        """Holographize should migrate bridge edges and keep real edges."""
        sc = SleepCompressor(data_dir=str(tmp_path))

        initial_edges = len(small_network.edges)
        initial_bridges = sum(
            1 for e in small_network.edges if e.origin in
            {"semantic_bridge", "hub_attachment", "associative_bridge", "bridge"}
        )

        stats = sc.compress(network=small_network, embeddings=None)

        # Bridge edges should be migrated
        assert stats["bridges_migrated"] == initial_bridges
        # Remaining edges should be the "real" ones
        assert stats["explicit_edges_remaining"] == initial_edges - initial_bridges
        # Holographic graph should have the migrated edges
        assert sc.holographic_graph.edge_count == initial_bridges

        # Verify no bridge origins remain in the explicit graph
        for edge in small_network.edges:
            assert edge.origin not in {
                "semantic_bridge", "hub_attachment", "associative_bridge", "bridge"
            }

    def test_homeostasis_scales_weights(self, tmp_path, small_network):
        """Synaptic homeostasis should scale down all edge weights."""
        sc = SleepCompressor(data_dir=str(tmp_path))

        # Record original weights of non-bridge edges
        original_weights = {
            (e.source, e.target, e.relation): e.weight
            for e in small_network.edges
            if e.origin not in {"semantic_bridge", "hub_attachment",
                                "associative_bridge", "bridge"}
        }

        sc.compress(network=small_network, embeddings=None)

        # Check that remaining edges have been scaled
        for edge in small_network.edges:
            key = (edge.source, edge.target, edge.relation)
            if key in original_weights:
                assert edge.weight < original_weights[key]

    def test_compress_without_embeddings(self, tmp_path, small_network):
        """Compression should work without embeddings (skip VQ)."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=small_network, embeddings=None)

        # VQ should be skipped
        assert stats.get("vq_trained", 0) == 0
        # But holographize and homeostasis should run
        assert "bridges_migrated" in stats
        assert "homeostasis_scaled" in stats

    def test_save_load_persistence(self, tmp_path, small_network):
        """Compression state should persist across instances."""
        sc1 = SleepCompressor(data_dir=str(tmp_path))
        sc1.compress(network=small_network, embeddings=None)

        # Create a new instance — should load the saved state
        sc2 = SleepCompressor(data_dir=str(tmp_path))
        assert sc2.holographic_graph.edge_count == sc1.holographic_graph.edge_count

    def test_holographic_graph_attached_to_network(self, tmp_path, small_network):
        """After compression, get_neighbors should query the holographic graph.

        The holographic graph is attached to the network so compressed
        associations remain accessible. Without this, migrated bridge
        edges would be invisible to get_neighbors and spread_activation.
        """
        sc = SleepCompressor(data_dir=str(tmp_path))
        sc.compress(network=small_network, embeddings=None)

        # Attach the holographic graph to the network (as mind.py does)
        small_network.attach_holographic_graph(sc.holographic_graph)

        # Query neighbors for "dog" — should include holographic
        # associations (the migrated bridge edges to "cat" and "bird")
        neighbors = small_network.get_neighbors("dog")
        neighbor_ids = {n[0] for n in neighbors}
        # "cat" and "bird" were bridge edges migrated to the holographic graph
        assert "cat" in neighbor_ids or "bird" in neighbor_ids

    def test_no_bridges_to_migrate(self, tmp_path):
        """Compression should handle a network with no bridge edges."""
        net = ConceptNetwork()
        net.add_concept("a", origin="stated")
        net.add_concept("b", origin="stated")
        net.add_edge("a", "is_a", "b", weight=0.8, origin="stated")

        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=net, embeddings=None)

        assert stats["bridges_migrated"] == 0
        assert stats["explicit_edges_remaining"] == 1

    def test_empty_network(self, tmp_path):
        """Compression should handle an empty network."""
        net = ConceptNetwork()
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=net, embeddings=None)

        assert stats["bridges_migrated"] == 0
        assert stats["explicit_edges_remaining"] == 0

    def test_stats_has_expected_fields(self, tmp_path, small_network):
        """Compression stats should have all expected fields."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=small_network, embeddings=None)

        expected_fields = [
            "bridges_migrated",
            "explicit_edges_remaining",
            "holographic_edges",
            "homeostasis_scaled",
            "homeostasis_pruned",
            "elapsed_s",
        ]
        for field in expected_fields:
            assert field in stats, f"Missing field: {field}"

    def test_ltm_pagination_compacts_all_episodes(self, tmp_path):
        """LTM compaction must paginate past the daemon's 1000-episode cap.

        The daemon caps search_episodes at 1000 per call. Without
        pagination, only the first 1000 episodes would be compacted
        and the LTM would grow unboundedly. This test creates 2500
        episodes and verifies that all are seen and the low-salience
        ones beyond max_ltm_episodes are deleted.
        """
        # Build a network with concepts that appear in episode text
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")
        net.add_concept("cat", origin="stated")

        # Create 2500 episodes — well past the 1000-episode cap.
        # Salience = i/2500, so episode 0 has salience 0.0 and
        # episode 2499 has salience ~0.9996.
        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=i,
                salience=i / 2500.0,
                event_type=0,
                source_module=0,
                text=f"dog and cat episode {i}",
            )
            for i in range(2500)
        ]
        ltm = _MockLTMClient(episodes)

        sc = SleepCompressor(
            data_dir=str(tmp_path),
            max_ltm_episodes=100,  # retain only top 100 by salience
            salience_threshold=0.6,
        )
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        # All 2500 episodes must be seen — not just the first 1000.
        # This is the core assertion: without pagination, this would
        # be 1000 and the remaining 1500 episodes would never compact.
        assert stats["ltm_total_seen"] == 2500

        # Top 100 by salience (episodes 2400..2499) are retained
        assert stats["ltm_retained"] == 100

        # The to_compact set is episodes 0..2399 (2400 episodes).
        # Every non-retained episode is compacted — high-salience
        # overflow (i in 1500..2399) included: their associations
        # carry proportionally more weight via salience scaling.
        assert stats["ltm_replayed"] == 2400
        assert stats["ltm_archived"] == 2400

        # Verify exactly the right episodes were deleted: everything
        # outside the retained top-100.
        retained_ids = {e.episode_id for e in episodes[2400:2500]}
        expected_deleted = {
            e.episode_id for e in episodes
            if e.episode_id not in retained_ids
        }
        assert ltm.archived_ids == expected_deleted

    def test_ltm_pagination_partial_failure(self, tmp_path):
        """LTM compaction should process fetched episodes if daemon disconnects mid-pagination."""
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")

        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=i,
                salience=0.1,  # below threshold → will be deleted
                event_type=0,
                source_module=0,
                text=f"dog episode {i}",
            )
            for i in range(1500)
        ]

        class _DisconnectingLTMClient(_MockLTMClient):
            """Simulates daemon disconnect after the first page."""
            _call_count = 0

            def search_episodes(self, limit=1000, offset=0):
                """Search episodes."""
                self._call_count += 1
                if self._call_count > 1:
                    raise OSError("daemon disconnected")
                return super().search_episodes(limit, offset)

        ltm = _DisconnectingLTMClient(episodes)
        sc = SleepCompressor(
            data_dir=str(tmp_path),
            max_ltm_episodes=50,
            salience_threshold=0.6,
        )
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        # Should have processed the first 1000 episodes despite the
        # daemon disconnecting on the second page
        assert stats["ltm_total_seen"] == 1000
        assert stats["ltm_archived"] > 0

    def test_atomicity_restores_on_failure(self, tmp_path, small_network):
        """If a phase fails, the network edge list must be restored.

        Without atomicity, a partial failure would leave bridge edges
        removed from the network but not saved to the holographic
        graph on disk — causing data loss on restart. The snapshot
        must be restored so the next pass can try again.
        """
        sc = SleepCompressor(data_dir=str(tmp_path))

        # Record original state
        original_edge_count = len(small_network.edges)
        original_weights = {
            (e.source, e.target, e.relation): e.weight
            for e in small_network.edges
        }

        # Patch _apply_homeostasis to raise, simulating a mid-pass failure
        # after phase 2 (holographize) has already mutated the network.
        def _boom(network):
            """Raise RuntimeError to simulate a mid-pass homeostasis failure."""
            raise RuntimeError("simulated failure")
        sc._apply_homeostasis = _boom  # type: ignore[method-assign]

        stats = sc.compress(network=small_network, embeddings=None)

        # Should report the error
        assert "error" in stats

        # The network must be restored to its original state —
        # same edge count and same weights.
        assert len(small_network.edges) == original_edge_count
        for edge in small_network.edges:
            key = (edge.source, edge.target, edge.relation)
            assert edge.weight == original_weights[key]

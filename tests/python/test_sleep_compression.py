"""Tests for the sleep compression module."""

import time
from dataclasses import dataclass

import pytest
from sleep.compression import SleepCompressor
from temporal_lobe.concepts import ConceptNetwork


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

    def test_empty_rebuild_does_not_wipe_holographic_graph(self, tmp_path):
        """A rebuild with no bridge edges must leave the graph intact.

        Regression test for silent destruction of associative memory.
        `_maybe_rebuild_holographic_graph` runs every 10th compression
        cycle behind a guard on the holographic graph's *pre-rebuild*
        edge count, then called `rebuild_from_edges(all_bridge_edges)`.
        When that list was empty the rebuild still began with
        `clear()`, so the guard passed and the store was wiped.

        Empty is the common case: most of `_BRIDGE_ORIGINS` are also in
        `DERIVABLE_ORIGINS`, and derivable edges are never
        materialised — the edge log rejects them at add_edge and they
        are filtered on spill and recall. So `network.edges` rarely
        holds any, meaning every ~10th cycle discarded the graph.
        """
        sc = SleepCompressor(data_dir=str(tmp_path))
        sc.holographic_graph.add("dog", "related_to", "cat", weight=1.0)
        assert sc.holographic_graph.edge_count == 1

        net = ConceptNetwork()
        net.add_concept("unrelated", confidence=0.9)
        # A network with no bridge-origin edges at all.
        for _ in range(10):
            sc._maybe_rebuild_holographic_graph(net)

        assert sc.holographic_graph.edge_count == 1, (
            "holographic graph was wiped by a rebuild with no bridge edges"
        )
        assert [r[0] for r in sc.holographic_graph.query("dog", "related_to", top_k=5)].count(
            "cat"
        ) == 1

    def test_periodic_rebuild_uses_journal_not_network_edges(self, tmp_path):
        """The ~10th-cycle rebuild re-encodes the journaled set.

        The live graph grew to ~1.9M raw bindings because nothing ever
        rebuilt it: the old scan rebuilt only from materialized bridge
        edges, which derivable origins never produce. Now the journal
        is the rebuild source — after the cycle the edge count is the
        deduplicated journal size, not the raw add count.
        """
        sc = SleepCompressor(data_dir=str(tmp_path))
        net = ConceptNetwork()

        # The same association bound 30 times (e.g. an LTM co-occurrence
        # pair re-extracted every pass): raw adds flood, journal dedups.
        for _ in range(30):
            sc.holographic_graph.add("dog", "related_to", "cat", weight=0.3)
        assert sc.holographic_graph.edge_count == 30

        for _ in range(10):
            sc._maybe_rebuild_holographic_graph(net)

        assert sc.holographic_graph.edge_count == 1, (
            "rebuild should re-encode the deduplicated journal"
        )
        assert sc.holographic_graph.distinct_edges == 1
        assert [r[0] for r in sc.holographic_graph.query("dog", "related_to", top_k=5)].count(
            "cat"
        ) == 1

    def test_rebuild_skipped_when_journal_empty(self, tmp_path):
        """A journalless (legacy) graph is never wiped by the rebuild."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        # Simulate a pre-journal load: edges in memory, journal empty.
        sc.holographic_graph._bind("dog", "related_to", "cat", 0.9)
        sc.holographic_graph._edge_count = 1
        sc.holographic_graph._journal = {}

        net = ConceptNetwork()
        for _ in range(10):
            sc._maybe_rebuild_holographic_graph(net)

        assert sc.holographic_graph.edge_count == 1
        assert sc.holographic_graph.distinct_edges == 0

    def test_holographize_migrates_bridges(self, tmp_path, small_network):
        """Holographize should migrate bridge edges and keep real edges."""
        sc = SleepCompressor(data_dir=str(tmp_path))

        initial_edges = len(small_network.edges)
        initial_bridges = sum(
            1
            for e in small_network.edges
            if e.origin in {"semantic_bridge", "hub_attachment", "associative_bridge", "bridge"}
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
                "semantic_bridge",
                "hub_attachment",
                "associative_bridge",
                "bridge",
            }

    def test_compress_deletes_no_edges(self, tmp_path, small_network):
        """Compression migrates but never mass-deletes: every edge that
        leaves the in-memory graph went to a queryable store (hologram
        or LTM index) — there is no homeostasis kill floor."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        before = len(small_network.edges)
        stats = sc.compress(network=small_network, embeddings=None)
        migrated = stats.get("bridges_migrated", 0)
        assert before - len(small_network.edges) <= migrated

    def test_compress_without_embeddings(self, tmp_path, small_network):
        """Compression should work without embeddings (skip VQ)."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=small_network, embeddings=None)

        # VQ should be skipped
        assert stats.get("vq_trained", 0) == 0
        # But holographize should still run
        assert "bridges_migrated" in stats

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
            # The per-sweep archive budget is a default-behavior guard
            # tested separately — this test checks pagination itself.
            max_archive_per_sweep=2500,
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
        expected_deleted = {e.episode_id for e in episodes if e.episode_id not in retained_ids}
        assert ltm.archived_ids == expected_deleted

    def test_ltm_compaction_preserves_dream_episodes(self, tmp_path):
        """Dream insights (source_module=9) must never be archived.

        Regression test for sleep compression erasing the dream record.
        ``_phase_compact_ltm`` used to page every active episode into
        the salience economy and archive all but the top
        ``max_ltm_episodes``. Dream insights are stored at a fixed 0.70
        salience, which can never compete in a global top-N once the
        store holds more than N higher-salience episodes — so every N3
        pass archived every dream it saw. ``get_recent_episodes``
        reads only the active list, so wake residues, ``/sleep``
        status, and dream notifications all read empty even though
        the dreams existed (archived, not deleted — but invisible to
        every surface that consumes them).
        """
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")

        # 10 real-content episodes at distinct retention-eligible
        # saliences 0.60–0.87 (cap 5 → ids 5–9 retained, 0–4 compacted)
        # plus 5 dream insights at the fixed 0.70 dream salience.
        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=i,
                salience=0.6 + i * 0.03,
                event_type=0,
                source_module=4,
                text=f"dog episode {i}",
            )
            for i in range(10)
        ] + [
            _MockEpisode(
                episode_id=100 + i,
                timestamp=100 + i,
                salience=0.7,
                event_type=2,
                source_module=9,
                text=f"[dream-insight] episode {i} connects to episode {i + 1}",
            )
            for i in range(5)
        ]
        ltm = _MockLTMClient(episodes)

        sc = SleepCompressor(
            data_dir=str(tmp_path),
            max_ltm_episodes=5,
            salience_threshold=0.6,
        )
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        dream_ids = {100 + i for i in range(5)}
        # Not one dream episode archived, however the ranking fell out.
        assert ltm.archived_ids.isdisjoint(dream_ids)
        # The retention cap applies to real content only: top 5 of the
        # ten 0.9-salience episodes retained, the rest archived.
        assert ltm.archived_ids == {0, 1, 2, 3, 4}
        assert stats["ltm_retained"] == 5
        assert stats["ltm_archived"] == 5
        assert stats["ltm_dreams_preserved"] == 5
        assert stats["ltm_total_seen"] == 15

    def test_ltm_compaction_defers_beyond_archive_budget(self, tmp_path):
        """A sweep retires at most max_archive_per_sweep, weakest first.

        Regression guard against the one-pass collapse: forgetting is
        gradual — the retention bound is reached over consecutive
        nights, retirements coming off the low-salience tail.
        """
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")

        # 150 old episodes (timestamp 0 — far past the recency grace),
        # all below the salience threshold. ids 0..149, salience
        # ascending — the weakest are the lowest ids.
        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=0,
                salience=i / 300.0,
                event_type=0,
                source_module=4,
                text=f"dog episode {i}",
            )
            for i in range(150)
        ]
        ltm = _MockLTMClient(episodes)

        sc = SleepCompressor(
            data_dir=str(tmp_path),
            salience_threshold=0.6,
            max_archive_per_sweep=40,
        )
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        assert stats["ltm_archived"] == 40
        assert stats["ltm_deferred"] == 110
        # Weakest-first: the sorted-descending list's tail is the
        # lowest-salience ids 0..39.
        assert ltm.archived_ids == set(range(40))

    def test_ltm_compaction_protects_recent_episodes(self, tmp_path):
        """Episodes inside the recency grace are never compaction fodder.

        A memory too young to have consolidated is exempt regardless
        of salience — systems consolidation needs nights, not hours.
        """
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")
        now_ms = int(time.time() * 1000)
        one_hour = 3_600_000
        ten_days = 10 * 86_400_000

        episodes = [
            # Recent, low-salience — would compact without the grace.
            _MockEpisode(
                episode_id=i,
                timestamp=now_ms - one_hour,
                salience=0.1,
                event_type=0,
                source_module=4,
                text=f"fresh dog episode {i}",
            )
            for i in range(5)
        ] + [
            # Old, low-salience — compaction-eligible.
            _MockEpisode(
                episode_id=100 + i,
                timestamp=now_ms - ten_days,
                salience=0.1,
                event_type=0,
                source_module=4,
                text=f"old dog episode {i}",
            )
            for i in range(5)
        ]
        ltm = _MockLTMClient(episodes)

        sc = SleepCompressor(data_dir=str(tmp_path), salience_threshold=0.6)
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        assert stats["ltm_recent_protected"] == 5
        assert ltm.archived_ids == {100 + i for i in range(5)}
        # A timestamp-less episode is treated as old — grace protects
        # only what is provably recent.
        assert stats["ltm_archived"] == 5

    def test_ltm_compaction_skips_when_client_disconnected(self, tmp_path):
        """A dead control channel must skip phase 3, not half-run it.

        search_episodes rides the daemon's read channel and can succeed
        while archive_episode (control channel) raises ConnectionError
        on every call — the Oct-2026 pass that committed extracted
        associations while archiving nothing. The is_connected preflight
        skips the phase cleanly; clients without the attribute (test
        mocks) are treated as connected.
        """
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")

        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=i,
                salience=0.1,
                event_type=0,
                source_module=4,
                text=f"dog episode {i}",
            )
            for i in range(10)
        ]

        class _DeadLTMClient(_MockLTMClient):
            """Client whose control channel is down (search would work,
            archive cannot)."""

            is_connected = False

        ltm = _DeadLTMClient(episodes)
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        assert stats["ltm_skipped"] == 1
        assert stats["ltm_total_seen"] == 0
        assert stats["ltm_archived"] == 0
        assert ltm.archived_ids == set()

    def test_ltm_compaction_counts_archive_failures(self, tmp_path):
        """Archive failures must be counted, not silently swallowed."""
        net = ConceptNetwork()
        net.add_concept("dog", origin="stated")

        episodes = [
            _MockEpisode(
                episode_id=i,
                timestamp=i,
                salience=0.1,
                event_type=0,
                source_module=4,
                text=f"dog episode {i}",
            )
            for i in range(5)
        ]

        class _FailingArchiveClient(_MockLTMClient):
            """archive_episode raises on every call (mid-pass drop)."""

            def archive_episode(self, episode_id: int) -> bool:
                raise OSError("connection lost mid-pass")

        ltm = _FailingArchiveClient(episodes)
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc.compress(network=net, embeddings=None, ltm_client=ltm)

        assert stats["ltm_archived"] == 0
        assert stats["ltm_archive_failed"] == 5
        assert stats["ltm_replayed"] == 5

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
        original_weights = {(e.source, e.target, e.relation): e.weight for e in small_network.edges}

        # Patch the final save-step hook to raise, simulating a mid-pass
        # failure after holographize has already mutated the network.
        def _boom(*_args):
            """Raise RuntimeError to simulate a mid-pass save failure."""
            raise RuntimeError("simulated failure")

        sc._maybe_rebuild_holographic_graph = _boom  # type: ignore[method-assign]

        stats = sc.compress(network=small_network, embeddings=None)

        # Should report the error
        assert "error" in stats

        # The network must be restored to its original state —
        # same edge count and same weights.
        assert len(small_network.edges) == original_edge_count
        for edge in small_network.edges:
            key = (edge.source, edge.target, edge.relation)
            assert edge.weight == original_weights[key]


# ═══════════════════════════════════════════════════════════════════
# Engram release — the prune phase
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _MockArchivedEntry:
    """Minimal archived-entry mock matching ArchivedEpisode's interface."""

    episode_id: int
    timestamp: int
    salience: float
    flags: int
    source_module: int
    emotional_tag: tuple


class _PrunableLTMClient:
    """Mock LTM client with the archived-tier operations."""

    def __init__(self, entries: list[_MockArchivedEntry]) -> None:
        self._entries = entries
        self.deleted_ids: list[int] = []
        self.is_connected = True

    def search_archived(self, limit: int = 1000, offset: int = 0) -> list:
        return self._entries[offset : offset + min(limit, 1000)]

    def delete_episode(self, episode_id: int) -> bool:
        self.deleted_ids.append(episode_id)
        return True


def _entry(eid: int, age_ms: int, salience: float = 0.5, flags: int = 4, source: int = 1):
    now_ms = int(time.time() * 1000)
    return _MockArchivedEntry(
        episode_id=eid,
        timestamp=now_ms - age_ms,
        salience=salience,
        flags=flags,
        source_module=source,
        emotional_tag=(0.3, 0.5, 0.3, 0.4),
    )


class TestPruneLtm:
    """The prune phase: she decides, the compressor carries it out."""

    def test_release_fn_true_deletes(self, tmp_path):
        """Entries the model reports dead are deleted."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        client = _PrunableLTMClient([_entry(1, 100 * 86_400_000), _entry(2, 200 * 86_400_000)])
        stats = sc._phase_prune_ltm(client, lambda e: True)
        assert client.deleted_ids == [1, 2]
        assert stats["ltm_pruned"] == 2
        assert stats["ltm_archived_seen"] == 2

    def test_release_fn_false_keeps(self, tmp_path):
        """Entries the model keeps are not touched."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        client = _PrunableLTMClient([_entry(1, 100 * 86_400_000)])
        stats = sc._phase_prune_ltm(client, lambda e: False)
        assert client.deleted_ids == []
        assert stats["ltm_pruned"] == 0

    def test_meta_memories_never_released(self, tmp_path):
        """Meta entries are skipped before the model is even asked —
        they are the cortical residue that justifies the forgetting."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        # flags=6 → ARCHIVED|META
        client = _PrunableLTMClient([_entry(1, 400 * 86_400_000, flags=6)])
        called = []
        stats = sc._phase_prune_ltm(client, lambda e: called.append(e) or True)
        assert called == []
        assert client.deleted_ids == []
        assert stats["ltm_meta_skipped"] == 1

    def test_dreams_never_released(self, tmp_path):
        """Dream-source episodes are never release candidates."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        client = _PrunableLTMClient([_entry(1, 400 * 86_400_000, source=9)])
        stats = sc._phase_prune_ltm(client, lambda e: True)
        assert client.deleted_ids == []
        assert stats["ltm_dreams_skipped"] == 1

    def test_budget_bounds_deletion(self, tmp_path):
        """The per-sweep budget caps deletion; the rest defer."""
        sc = SleepCompressor(data_dir=str(tmp_path), max_delete_per_sweep=2)
        client = _PrunableLTMClient([_entry(i, 400 * 86_400_000) for i in range(1, 6)])
        stats = sc._phase_prune_ltm(client, lambda e: True)
        assert client.deleted_ids == [1, 2]  # oldest first
        assert stats["ltm_pruned"] == 2
        assert stats["ltm_release_deferred"] == 3

    def test_missing_ops_skips_gracefully(self, tmp_path):
        """A client without the archived-tier ops is skipped."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        stats = sc._phase_prune_ltm(object(), lambda e: True)
        assert stats["ltm_prune_skipped"] == 1

    def test_disconnected_skips(self, tmp_path):
        """A dead client can't delete — skip rather than half-commit."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        client = _PrunableLTMClient([_entry(1, 400 * 86_400_000)])
        client.is_connected = False
        stats = sc._phase_prune_ltm(client, lambda e: True)
        assert client.deleted_ids == []
        assert stats["ltm_prune_skipped"] == 1

    def test_failed_delete_counted(self, tmp_path):
        """A refused delete is counted, not silently dropped."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        client = _PrunableLTMClient([_entry(1, 400 * 86_400_000)])
        client.delete_episode = lambda eid: False
        stats = sc._phase_prune_ltm(client, lambda e: True)
        assert stats["ltm_prune_failed"] == 1
        assert stats["ltm_pruned"] == 0

    def test_step_only_with_release_fn(self, tmp_path):
        """The prune step exists only when a verdict is supplied —
        the compressor never forgets on its own authority."""
        sc = SleepCompressor(data_dir=str(tmp_path))
        net = ConceptNetwork()
        client = _PrunableLTMClient([])

        without = dict(sc.compress_steps(net, ltm_client=client)).keys()
        assert "prune_ltm" not in without

        with_fn = dict(sc.compress_steps(net, ltm_client=client, release_fn=lambda e: False))
        assert "prune_ltm" in with_fn

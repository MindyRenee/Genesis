"""Canonical edge log tests.

The edge log is the single source of truth for relationships:
assert/retract/snapshot events, last-write-wins fold, compaction,
derivable-edge rejection, and the ConceptNetwork wiring that makes
``_edges`` a materialized view of the log.
"""

import json

from genesis_cognitive.concepts.edge_log import (
    is_derivable_edge,
    is_web_origin,
    open_edge_log,
    web_domain_of,
    web_origin,
)
from genesis_cognitive.concepts.network import ConceptNetwork
from genesis_cognitive.concepts.types import Edge, RelationType


def _edge(src="a", tgt="b", rel=RelationType.IS_A, w=0.7, origin="stated"):
    return Edge(source=src, target=tgt, relation=rel, weight=w,
                created_at=1000, origin=origin)


class TestClassification:
    def test_typed_relations_always_canonical(self):
        for rel in (RelationType.IS_A, RelationType.CAUSES,
                    RelationType.CALLS, RelationType.PART_OF):
            assert not is_derivable_edge(rel, "hub_attachment")
            assert not is_derivable_edge(rel, "inferred")

    def test_geometric_with_pipeline_origin_derivable(self):
        assert is_derivable_edge(RelationType.RELATED_TO, "hub_attachment")
        assert is_derivable_edge(RelationType.BRIDGES, "semantic_bridge")
        assert is_derivable_edge(RelationType.SIMILAR_TO, "co_occurrence")
        assert is_derivable_edge(RelationType.RELATED_TO, "inferred")

    def test_geometric_with_earned_origin_canonical(self):
        assert not is_derivable_edge(RelationType.RELATED_TO, "stated")
        assert not is_derivable_edge(RelationType.RELATED_TO, "observed")
        assert not is_derivable_edge(RelationType.SIMILAR_TO, "vocabulary_seed")

    def test_web_origin_normalizes_domain(self):
        assert web_origin("en.wikipedia.org") == "web:en.wikipedia.org"
        assert web_origin("https://EN.wikipedia.org/wiki/Cat") == "web:en.wikipedia.org"
        assert web_origin("http://example.com:8080/x?y=1") == "web:example.com"
        assert web_origin("") == "web:unknown"
        assert web_origin("not a domain!!") == "web:unknown"

    def test_is_web_origin_matches_prefix_only(self):
        assert is_web_origin("web:en.wikipedia.org")
        assert is_web_origin("web")
        assert not is_web_origin("stated")
        assert not is_web_origin("learned")
        assert not is_web_origin("observed")
        assert not is_web_origin("cognition_lesson")

    def test_web_domain_of_round_trips(self):
        assert web_domain_of("web:en.wikipedia.org") == "en.wikipedia.org"
        assert web_domain_of("web") == "unknown"
        assert web_domain_of("stated") is None
        assert web_domain_of("learned") is None

    def test_web_typed_edge_is_canonical_but_quarantinable(self):
        # A typed web claim is a factual claim (canonical), yet still
        # identifiable as untrusted-source by prefix.
        assert not is_derivable_edge(RelationType.IS_A, "web:example.com")
        assert is_web_origin("web:example.com")
        assert not is_derivable_edge(RelationType.BRIDGES, "labeling")

    def test_unknown_origin_defaults_canonical(self):
        # Unlisted provenance is not provably geometry — keep it.
        # Experiential origins (sleep replay, introspection, sensory
        # co-activation) carry bindings similarity cannot rebuild.
        for origin in ("swr-replay", "introspection", "sensory",
                       "spindle", "lucid-dream", "learned", "unknown-x"):
            assert not is_derivable_edge(RelationType.RELATED_TO, origin)

    def test_accepts_relation_value_strings(self):
        assert is_derivable_edge("related_to", "hub_attachment")
        assert not is_derivable_edge("is_a", "hub_attachment")


class TestEdgeLog:
    def test_assert_fold_roundtrip(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1000)
        live = log.fold()
        log.close()
        assert len(live) == 1
        e = next(iter(live.values()))
        assert (e.source, e.target, e.relation) == ("a", "b", RelationType.IS_A)
        assert e.weight == 0.7 and e.origin == "stated"

    def test_last_write_wins(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.5, "stated", 1)
        log.assert_edge("a", "b", RelationType.IS_A, 0.9, "stated", 2)
        live = log.fold()
        log.close()
        assert next(iter(live.values())).weight == 0.9

    def test_retract_tombstones(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        log.retract_edge("a", "b", RelationType.IS_A)
        assert log.fold() == {}
        log.close()

    def test_snapshot_resets_fold_base(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("old", "edge", RelationType.IS_A, 0.5, "stated", 1)
        log.snapshot([_edge("x", "y"), _edge("y", "z", RelationType.CAUSES)])
        live = log.fold()
        log.close()
        assert len(live) == 2
        assert all(e.source in ("x", "y") for e in live.values())

    def test_derivable_rejected_at_boundary(self, tmp_path):
        log = open_edge_log(tmp_path)
        assert not log.assert_edge(
            "a", "b", RelationType.RELATED_TO, 0.5, "hub_attachment", 1
        )
        assert log.fold() == {}
        log.close()

    def test_snapshot_filters_derivables(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.snapshot([
            _edge("a", "b", RelationType.IS_A),
            _edge("a", "c", RelationType.RELATED_TO, origin="hub_attachment"),
        ])
        live = log.fold()
        log.close()
        assert len(live) == 1

    def test_compaction_is_deterministic_for_same_edges(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("b", "c", RelationType.IS_A, 0.4, "stated", 2)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        log.compact()
        first = log.path.read_text(encoding="utf-8")
        log.compact()
        second = log.path.read_text(encoding="utf-8")
        log.close()
        assert first == second

    def test_malformed_json_events_do_not_break_fold(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        log.close()
        with open(log.path, "a", encoding="utf-8") as f:
            f.write('{"op":"assert","source":"bad","target":"x","relation":"is_a","weight":"nan"}\n')
            f.write('{"op":"assert","source":"bad2","target":"x","relation":"is_a","created_at":{}}\n')
            f.write('{"op":"snapshot","edges":"not-a-list"}\n')
        reopened = open_edge_log(tmp_path)
        live = reopened.fold()
        reopened.close()
        assert len(live) == 1
        assert next(iter(live.values())).source == "a"

    def test_torn_tail_skipped(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        log.close()
        with open(log.path, "a") as f:
            f.write('{"op": "assert", "source": "tor')
        live = log.fold()
        assert len(live) == 1

    def test_compact_single_line_same_fold(self, tmp_path):
        log = open_edge_log(tmp_path)
        for i in range(50):
            log.assert_edge(f"c{i}", "hub", RelationType.RELATED_TO,
                            0.5, "stated", i)
        log.retract_edge("c0", "hub", RelationType.RELATED_TO)
        before = log.fold()
        n = log.compact()
        after = log.fold()
        lines = log.path.read_text().strip().split("\n")
        log.close()
        assert n == 49 == len(before) == len(after)
        assert len(lines) == 1
        assert json.loads(lines[0])["op"] == "snapshot"

    def test_is_empty_semantics(self, tmp_path):
        log = open_edge_log(tmp_path)
        assert log.is_empty  # file created but holds no events
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        assert not log.is_empty
        log.close()

    def test_reopen_preserves_events(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        log.close()
        log2 = open_edge_log(tmp_path)
        assert len(log2.fold()) == 1
        log2.close()


class TestNetworkWiring:
    def test_attach_loads_existing_log(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        net = ConceptNetwork()
        net.add_concept("a")
        net.add_concept("b")
        net.attach_edge_log(log)
        assert net.edge_count == 1
        assert net.get_neighbors("a") == [("b", RelationType.IS_A, 0.7)]
        log.close()

    def test_empty_log_does_not_wipe(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")
        net.attach_edge_log(log)
        # Empty log isn't authoritative — current edges were snapshotted
        # in as the seed and stay live.
        assert net.edge_count == 1
        assert not log.is_empty
        log.close()

    def test_add_edge_writes_through(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")  # reinforce
        live = log.fold()
        e = next(iter(live.values()))
        log.close()
        assert abs(e.weight - 0.9) < 1e-6  # 0.7 + 0.2 bump

    def test_derivable_never_materializes(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        result = net.add_edge("a", "b", RelationType.RELATED_TO, 0.5,
                              "hub_attachment")
        assert result is None
        assert net.edge_count == 0
        assert log.fold() == {}
        log.close()

    def test_remove_edge_retracts(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")
        assert net.remove_edge("a", "b", RelationType.IS_A)
        assert log.fold() == {}
        log.close()

    def test_replace_edges_snapshots(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("old", "gone", RelationType.IS_A, 0.7, "stated")
        net.replace_edges([_edge("x", "y")])
        live = log.fold()
        log.close()
        assert len(live) == 1
        assert next(iter(live.values())).source == "x"

    def test_replace_contents_seeds_empty_log(self, tmp_path):
        log = open_edge_log(tmp_path)
        shared = ConceptNetwork()
        shared.attach_edge_log(log)
        staged = ConceptNetwork()
        staged.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")

        shared.replace_contents(staged)

        assert shared.edge_count == 1
        live = log.fold()
        assert len(live) == 1
        assert next(iter(live.values())).source == "a"
        log.close()

    def test_replace_contents_log_overrides_transplant(self, tmp_path):
        log = open_edge_log(tmp_path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "stated", 1)
        shared = ConceptNetwork()
        shared.attach_edge_log(log)
        staged = ConceptNetwork()
        staged.add_edge("junk", "edge", RelationType.RELATED_TO, 0.5, "stated")
        staged.add_concept("a")
        staged.add_concept("b")
        shared.replace_contents(staged)
        assert shared.edge_count == 1
        assert shared.get_neighbors("a") == [("b", RelationType.IS_A, 0.7)]
        log.close()

    def test_get_associations_tags_provenance(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")
        net.attach_similarity_provider(
            lambda cid: [("z", 0.9), ("y", 0.6)]
        )
        assoc = net.get_associations("a")
        log.close()
        exact = [n for n in assoc if n.provenance == "exact"]
        virtual = [n for n in assoc if n.provenance == "embedding"]
        assert [(n.concept, n.relation) for n in exact] == [
            ("b", RelationType.IS_A)
        ]
        assert [n.concept for n in virtual] == ["z", "y"]
        assert all(n.relation is None for n in virtual)

    def test_get_associations_respects_relation_filter(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.7, "stated")
        net.add_edge("a", "c", RelationType.CAUSES, 0.7, "stated")
        net.attach_similarity_provider(lambda cid: [("z", 0.9)])
        assoc = net.get_associations("a", relation=RelationType.IS_A)
        log.close()
        assert all(n.relation == RelationType.IS_A for n in assoc)
        assert all(n.provenance == "exact" for n in assoc)


class TestLegacyJsonMigration:
    """Pre-log state migration must not reintroduce derivable geometry.

    `_restore_edges` appends directly to `network._edges`, bypassing both
    `add_edge` and `replace_edges`, so the derivable-edge rule has to be
    enforced on the restore path too. It was not, and the missing-origin
    default made it worse: it used "inferred", a listed derivable origin,
    which inverted the documented "unknown provenance defaults to
    canonical" rule and silently deleted real legacy edges.
    """

    def test_unknown_origin_is_canonical(self):
        # Unlisted provenance defaults to canonical: losing an
        # experiential binding is irrecoverable, a false positive only
        # costs decay bandwidth.
        assert not is_derivable_edge(RelationType.RELATED_TO, "")
        assert not is_derivable_edge(RelationType.RELATED_TO, "dream_replay")
        # An origin that IS listed stays derivable when explicitly
        # present in the data — that is the rule working, not a bug.
        assert is_derivable_edge(RelationType.RELATED_TO, "inferred")

    def test_restore_keeps_legacy_edge_without_origin(self, tmp_path):
        from genesis_cognitive.persistence import _restore_edges

        net = ConceptNetwork()
        for n in ("a", "b"):
            net.add_concept(n, confidence=0.8)
        # No "origin" key at all — the common legacy shape.
        _restore_edges(
            net,
            [{"source": "a", "target": "b",
              "relation": RelationType.RELATED_TO.value, "weight": 0.6}],
            {"a": "a", "b": "b"},
        )
        assert net.edge_count == 1, (
            "legacy edge with no origin was dropped as if it were derivable"
        )

    def test_restore_drops_derivable_legacy_edge(self, tmp_path):
        from genesis_cognitive.persistence import _restore_edges

        net = ConceptNetwork()
        for n in ("a", "b"):
            net.add_concept(n, confidence=0.8)
        _restore_edges(
            net,
            [{"source": "a", "target": "b",
              "relation": RelationType.RELATED_TO.value, "weight": 0.6,
              "origin": "semantic_bridge"}],
            {"a": "a", "b": "b"},
        )
        assert net.edge_count == 0

    def test_restore_keeps_typed_edge_with_pipeline_origin(self, tmp_path):
        from genesis_cognitive.persistence import _restore_edges

        net = ConceptNetwork()
        for n in ("a", "b"):
            net.add_concept(n, confidence=0.8)
        # Typed relations are always canonical facts, whatever the origin.
        _restore_edges(
            net,
            [{"source": "a", "target": "b",
              "relation": RelationType.IS_A.value, "weight": 0.6,
              "origin": "inferred"}],
            {"a": "a", "b": "b"},
        )
        assert net.edge_count == 1

    def test_migration_does_not_seed_log_with_derivables(self, tmp_path):
        """The canonical log must be seeded with canonical edges only.

        The old order snapshotted the incoming list first and stripped
        memory second, leaving the log holding edges the network had
        already discarded.
        """
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        for n in ("a", "b", "c", "d"):
            net.add_concept(n, confidence=0.8)
        # Pre-log state: one canonical edge, one derivable leftover.
        net._edges = [
            Edge(source="a", target="b", relation=RelationType.IS_A,
                 weight=0.7, origin="stated"),
            Edge(source="c", target="d", relation=RelationType.RELATED_TO,
                 weight=0.6, origin="co_occurrence"),
        ]
        net._rebuild_edge_indices()

        net.attach_edge_log(log)
        live = log.fold()
        log.close()

        assert len(live) == 1, "derivable edge was written to the canonical log"
        assert net.edge_count == 1


class TestWeightWriteThrough:
    """Weight changes must reach the log, not just the in-memory fold.

    The edge log is the canonical source of truth for relationships. A
    weight assigned straight onto an `Edge` object updates the fold's
    materialization only — `fold()` keeps reporting the old weight, and
    the change is lost if the process exits before the next save.
    """

    def test_taught_confidence_reaches_the_log(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        edge = net.teach("dog", RelationType.IS_A, "animal", confidence=0.9)
        assert edge is not None
        log.close()
        assert edge.weight == 0.9
        # The log must agree, not just the returned object.
        reloaded = open_edge_log(tmp_path).fold()
        assert len(reloaded) == 1
        assert next(iter(reloaded.values())).weight == 0.9

    def test_taught_confidence_survives_no_resave(self, tmp_path):
        """The boost must be durable immediately, not at the next save.

        A direct attribute assignment is only folded in by the next
        `sync_edge_log` snapshot. Reopening the log without one shows
        the difference.
        """
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.teach("dog", RelationType.IS_A, "animal", confidence=0.9)
        log.close()

        net2 = ConceptNetwork()
        net2.attach_edge_log(open_edge_log(tmp_path))
        assert net2.edge_count == 1
        assert next(iter(net2._edges)).weight == 0.9

    def test_taught_edge_with_existing_weak_edge(self, tmp_path):
        """Teaching over a weak pre-existing edge must still write through."""
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("dog", "animal", RelationType.IS_A, weight=0.2,
                     origin="stated")
        net.teach("dog", RelationType.IS_A, "animal", confidence=0.9)
        log.close()
        reloaded = open_edge_log(tmp_path).fold()
        assert next(iter(reloaded.values())).weight >= 0.9

    def test_set_edge_weight_on_missing_edge_is_none(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_concept("dog", confidence=0.8)
        net.add_concept("cat", confidence=0.8)
        assert net.set_edge_weight("dog", "cat", RelationType.IS_A, 0.9) is None
        log.close()

    def test_add_edge_reinforcement_reaches_the_log(self, tmp_path):
        """Re-adding an edge raises it by 0.2 — and the log must see that."""
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("dog", "animal", RelationType.IS_A, weight=0.5,
                     origin="stated")
        net.add_edge("dog", "animal", RelationType.IS_A, origin="stated")
        log.close()
        reloaded = open_edge_log(tmp_path).fold()
        assert next(iter(reloaded.values())).weight == 0.7


class TestCompactionFailureIsReported:
    """A log that cannot be compacted must not fail silently.

    `sync_edge_log` previously swallowed OSError from both the size
    check and the compaction, so an uncompactable log grew without
    bound with no signal — the exact failure the warning exists to
    surface.
    """

    def test_compaction_failure_is_logged_not_swallowed(self, tmp_path, caplog):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.5, "stated")

        real = log.compact

        def boom() -> int:
            raise OSError(28, "No space left on device")

        log.compact = boom  # type: ignore[method-assign]
        # Force the threshold so compaction is attempted.
        import genesis_cognitive.concepts.network as nmod
        original = nmod._COMPACT_THRESHOLD_BYTES
        nmod._COMPACT_THRESHOLD_BYTES = 0
        try:
            with caplog.at_level("WARNING"):
                net.sync_edge_log()
        finally:
            nmod._COMPACT_THRESHOLD_BYTES = original
            log.compact = real  # type: ignore[method-assign]
            log.close()

        assert any("compaction failed" in r.message for r in caplog.records), (
            "compaction OSError was swallowed silently"
        )

    def test_failure_counter_resets_on_success(self, tmp_path):
        log = open_edge_log(tmp_path)
        net = ConceptNetwork()
        net.attach_edge_log(log)
        net.add_edge("a", "b", RelationType.IS_A, 0.5, "stated")
        assert net._compact_failures == 0
        net.sync_edge_log()
        assert net._compact_failures == 0
        log.close()

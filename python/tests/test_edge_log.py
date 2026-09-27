"""Canonical edge log tests.

The edge log is the single source of truth for relationships:
assert/retract/snapshot events, last-write-wins fold, compaction,
derivable-edge rejection, and the ConceptNetwork wiring that makes
``_edges`` a materialized view of the log.
"""

import json

from genesis_cognitive.concepts.edge_log import (
    is_derivable_edge,
    open_edge_log,
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

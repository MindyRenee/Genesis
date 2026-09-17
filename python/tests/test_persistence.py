"""Tests for persistence — saving and loading Genesis's cognitive state."""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from collections import deque

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.narrative import NarrativeEngine
from genesis_cognitive.persistence import (
    load_state,
    restore_narrative,
    restore_network,
    restore_reflection,
    restore_self_directed_learner,
    restore_self_model,
    save_state,
)
from genesis_cognitive.self import (
    Insight,
    PersonalityTraits,
    ReflectionEngine,
    SelfDirectedLearner,
    SelfModel,
)

logger = logging.getLogger(__name__)


def _make_network() -> ConceptNetwork:
    """Construct a network for tests."""
    net = ConceptNetwork()
    net.add_concept("dog", aliases={"puppy", "hound"}, confidence=0.8)
    net.add_concept("mammal", confidence=0.8)
    net.add_concept("animal", confidence=0.8)
    net.add_edge("dog", "mammal", RelationType.IS_A, 0.9)
    net.add_edge("mammal", "animal", RelationType.IS_A, 0.9)
    return net


def _make_reflection() -> ReflectionEngine:
    """Construct a reflection for tests."""
    refl = ReflectionEngine(_make_network())
    refl._interaction_count = 5
    refl.insights.append(
        Insight(
            type="growth",
            content="I learned something new",
            confidence=0.8,
            actionable=True,
            action="remember this",
        )
    )
    refl._mood_history = deque([(1000, 0.5, 0.3), (2000, 0.6, 0.4)], maxlen=50)
    return refl


def _make_narrative() -> NarrativeEngine:
    """Construct a narrative for tests."""
    self_model = SelfModel()
    narrative = NarrativeEngine(self_model, _make_network())
    return narrative


def _make_self_model() -> SelfModel:
    """Construct a self model for tests."""
    sm = SelfModel()
    sm.personality = PersonalityTraits(
        openness=0.8,
        conscientiousness=0.7,
        extraversion=0.6,
        agreeableness=0.8,
        neuroticism=0.3,
    )
    sm.self_knowledge["test_key"] = "test_value"
    return sm


# ─── Tests ──────────────────────────────────────────────────


def test_save_creates_file():
    """save_state writes a JSON file to the data dir."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)

        path = os.path.join(d, "cognitive_state.json")
        assert os.path.exists(path)


def test_load_returns_none_if_no_file():
    """load_state returns None if no saved state exists."""
    with tempfile.TemporaryDirectory() as d:
        result = load_state(d)
        assert result is None


def test_round_trip_concept_network():
    """Concept network survives save/load round-trip."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        # Restore into a fresh network
        fresh = ConceptNetwork()
        restore_network(fresh, data["concept_network"])

        assert fresh.size == 3
        assert fresh.edge_count == 2
        # Check aliases survived
        dog = fresh.get_concept("dog")
        assert dog is not None
        assert "puppy" in dog.aliases
        assert dog.confidence == 0.8


def test_round_trip_edges():
    """Edges (relationships) survive save/load round-trip."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh = ConceptNetwork()
        restore_network(fresh, data["concept_network"])

        neighbors = fresh.get_neighbors("dog")
        assert len(neighbors) == 1
        target, relation, weight = neighbors[0]
        assert target == "mammal"
        assert relation == RelationType.IS_A
        assert weight == 0.9


def test_round_trip_reflection():
    """Reflection state survives save/load round-trip."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh_net = ConceptNetwork()
        fresh_refl = ReflectionEngine(fresh_net)
        restore_reflection(fresh_refl, data["reflection"])

        assert fresh_refl._interaction_count == 5
        assert len(fresh_refl.insights) == 1
        assert fresh_refl.insights[0].content == "I learned something new"
        assert fresh_refl.insights[0].actionable is True
        assert len(fresh_refl._mood_history) == 2


def test_round_trip_narrative():
    """Narrative state survives save/load round-trip."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh_net = ConceptNetwork()
        fresh_sm = SelfModel()
        fresh_narr = NarrativeEngine(fresh_sm, fresh_net)
        restore_narrative(fresh_narr, data["narrative"])

        # Should have at least one chapter (Awakening)
        assert len(fresh_narr.chapters) >= 1


def test_round_trip_self_model():
    """Self-model survives save/load round-trip."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh_sm = SelfModel()
        restore_self_model(fresh_sm, data["self_model"])

        assert fresh_sm.personality.openness == 0.8
        assert fresh_sm.personality.agreeableness == 0.8
        assert fresh_sm.self_knowledge.get("test_key") == "test_value"


def test_atomic_write():
    """Save is atomic — no temp file left behind on success."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)

        # No temp files should remain
        files = os.listdir(d)
        assert len(files) == 1
        assert files[0] == "cognitive_state.json"


@pytest.mark.parametrize("raw", [b"", b"{", b"\x1f\x8b", b"[]", b"null", b'{"version":999}'])
def test_unreadable_state_is_not_treated_as_first_boot(tmp_path, raw):
    path = tmp_path / "cognitive_state.json"
    path.write_bytes(raw)
    with pytest.raises(OSError, match="cognitive_state"):
        load_state(str(tmp_path))
    assert path.read_bytes() == raw


def test_save_flushes_file_before_replace_and_directory_after(tmp_path):
    from genesis_cognitive.persistence import _atomic_write_state

    events = []
    replace = os.replace

    def record_replace(source, destination):
        events.append("replace")
        replace(source, destination)

    with (
        patch(
            "genesis_cognitive.persistence.os.fsync", side_effect=lambda fd: events.append("sync")
        ),
        patch("genesis_cognitive.persistence.os.replace", side_effect=record_replace),
    ):
        _atomic_write_state(str(tmp_path), {"version": 2})
    assert events == ["sync", "replace", "sync"]


def test_failed_flush_preserves_previous_save(tmp_path):
    from genesis_cognitive.persistence import _atomic_write_state

    path = tmp_path / "cognitive_state.json"
    path.write_bytes(b'{"version":2}')
    with patch("genesis_cognitive.persistence.os.fsync", side_effect=OSError("disk error")):
        with pytest.raises(OSError, match="disk error"):
            _atomic_write_state(str(tmp_path), {"version": 2, "new": True})
    assert path.read_bytes() == b'{"version":2}'
    assert list(tmp_path.iterdir()) == [path]


def test_large_network_round_trip():
    """A larger network with many concepts and edges round-trips correctly."""
    net = ConceptNetwork()
    for i in range(20):
        net.add_concept(f"concept_{i}", confidence=0.5 + i * 0.02)
    for i in range(19):
        net.add_edge(f"concept_{i}", f"concept_{i + 1}", RelationType.RELATED_TO, 0.5 + i * 0.02)

    with tempfile.TemporaryDirectory() as d:
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh = ConceptNetwork()
        restore_network(fresh, data["concept_network"])

        assert fresh.size == 20
        assert fresh.edge_count == 19
        # Check a specific concept
        c19 = fresh.get_concept("concept_19")
        assert c19 is not None
        assert abs(c19.confidence - (0.5 + 19 * 0.02)) < 0.01


def test_load_preserves_activation():
    """Activation values survive the round-trip."""
    net = _make_network()
    net.get_concept("dog").activation = 0.75

    with tempfile.TemporaryDirectory() as d:
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        fresh = ConceptNetwork()
        restore_network(fresh, data["concept_network"])

        dog = fresh.get_concept("dog")
        assert abs(dog.activation - 0.75) < 0.01


def test_self_directed_learner_persistence():
    """Self-directed learner state survives save/restore."""
    net = _make_network()
    learner = SelfDirectedLearner(net)

    # Generate some learning activity
    learner.learn_from_input("A dog is a mammal.")
    learner.learn_from_input("A cat is a mammal.")
    learner.run_inference_cycle(force=True)
    learner.learn_from_correction("No, actually a dog is a canine.", "dog")

    stats_before = learner.get_stats()
    assert stats_before["conversation_facts"] > 0
    assert stats_before["corrections"] > 0
    log_before = list(learner._learning_log)

    with tempfile.TemporaryDirectory() as d:
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        save_state(d, net, refl, narr, sm, self_directed_learner=learner)

        # Verify the key exists in saved data
        data = load_state(d)
        assert "self_directed_learner" in data

        # Restore into a fresh learner
        fresh_net = ConceptNetwork()
        fresh_learner = SelfDirectedLearner(fresh_net)
        restore_self_directed_learner(fresh_learner, data["self_directed_learner"])

        # Check stats match
        stats_after = fresh_learner.get_stats()
        assert stats_after == stats_before, f"Stats mismatch: {stats_after} vs {stats_before}"

        # Check learning log matches
        log_after = list(fresh_learner._learning_log)
        assert len(log_after) == len(log_before)
        for before, after in zip(log_before, log_after, strict=True):
            assert before.event_type == after.event_type
            assert before.description == after.description
            assert before.confidence == after.confidence

        # Check studied concepts match
        assert fresh_learner._studied_concepts == learner._studied_concepts


def test_self_directed_learner_backward_compat():
    """Loading old state without self_directed_learner doesn't crash."""
    with tempfile.TemporaryDirectory() as d:
        net = _make_network()
        refl = _make_reflection()
        narr = _make_narrative()
        sm = _make_self_model()

        # Save without the learner
        save_state(d, net, refl, narr, sm)
        data = load_state(d)

        # The key should be absent
        assert "self_directed_learner" not in data

        # This should not crash
        fresh_net = ConceptNetwork()
        fresh_learner = SelfDirectedLearner(fresh_net)
        # No restore call — just verify the learner starts with defaults
        assert fresh_learner.get_stats()["total_inferences"] == 0


def test_polysemous_concepts_survive_save_restore():
    """Test polysemous concepts survive save restore."""
    net = ConceptNetwork()
    first = net.add_concept("bank", properties={"definition": "a financial institution"})
    second = net.add_concept("bank", properties={"definition": "the edge of a river"})
    net.add_concept("money")
    net.add_concept("river")
    net.add_edge(first.id, "money", RelationType.RELATED_TO, 0.7)
    net.add_edge(second.id, "river", RelationType.RELATED_TO, 0.8)

    with tempfile.TemporaryDirectory() as d:
        save_state(d, net, _make_reflection(), _make_narrative(), _make_self_model())
        data = load_state(d)
        fresh = ConceptNetwork()
        restore_network(fresh, data["concept_network"])

    assert fresh.size == 4
    restored = fresh.get_concept(second.id)
    assert restored is not None
    assert restored.properties["definition"] == "the edge of a river"
    assert {target for target, _, _ in fresh.get_neighbors(second.id)} == {"river"}
    assert {target for target, _, _ in fresh.get_neighbors(first.id)} == {"money"}
    relearned = fresh.add_concept("bank", properties={"definition": "the edge of a river"})
    assert relearned.id == second.id
    assert fresh.size == 4


def test_restore_preserves_zero_confidence():
    """Test restore preserves zero confidence."""
    fresh = ConceptNetwork()
    restore_network(fresh, {"concepts": [{"id": "uncertain", "confidence": 0.0}], "edges": []})
    assert fresh.get_concept("uncertain").confidence == 0.0


def test_restore_merges_only_unambiguous_legacy_duplicates():
    """Test restore merges only unambiguous legacy duplicates."""
    data = {
        "concepts": [
            {"id": "bank", "properties": {"definition": "a financial institution"}},
            {"id": "bank#2", "aliases": ["banking"], "properties": {}},
            {"id": "money"},
        ],
        "edges": [{
            "source": "bank#2", "target": "money", "relation": "related_to", "weight": 0.7,
        }],
    }
    fresh = ConceptNetwork()
    restore_network(fresh, data)
    assert fresh.size == 2
    assert fresh.get_concept("banking").id == "bank"
    assert {target for target, _, _ in fresh.get_neighbors("bank")} == {"money"}


def test_restore_preserves_conflicting_senses_with_undefined_base():
    """Test restore preserves conflicting senses with undefined base."""
    data = {
        "concepts": [
            {"id": "bank", "properties": {}},
            {"id": "bank#2", "properties": {"definition": "a financial institution"}},
            {"id": "bank#3", "properties": {"definition": "the edge of a river"}},
        ],
        "edges": [],
    }
    fresh = ConceptNetwork()
    restore_network(fresh, data)
    assert fresh.size == 3
    assert fresh.get_concept("bank#2").properties["definition"] == "a financial institution"
    assert fresh.get_concept("bank#3").properties["definition"] == "the edge of a river"
    assert set(fresh._alias_map["bank"]) == {"bank", "bank#2", "bank#3"}


# ─── Test runner ────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Persistence tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

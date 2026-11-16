"""Tests for persistence — saving and loading Genesis's cognitive state."""

import logging
import os
import sys
import tempfile
from collections import deque
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.narrative import Narrator
from infrastructure.persistence import (
    load_state,
    restore_narrative,
    restore_network,
    restore_reflection,
    restore_self_directed_learner,
    restore_self_model,
    restore_semantic_memory,
    restore_synapses,
    save_state,
)
from self.learning import SelfDirectedLearner
from self.reflection import Insight, Reflection
from self.self_model import PersonalityTraits, SelfModel
from temporal_lobe.concepts import ConceptNetwork, RelationType
from temporal_lobe.synapses import SynapticStore

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


def _make_reflection() -> Reflection:
    """Construct a reflection for tests."""
    refl = Reflection(_make_network())
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


def _make_narrative() -> Narrator:
    """Construct a narrative for tests."""
    self_model = SelfModel()
    narrative = Narrator(self_model, _make_network())
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


def test_synaptic_efficacy_persistence():
    """Learned synaptic efficacy survives save/load round-trip."""
    net = _make_network()
    synapses = SynapticStore()
    synapses.set_weight("dog", "mammal", 0.73)
    synapses.set_weight("mammal", "animal", 0.41)

    with tempfile.TemporaryDirectory() as d:
        save_state(
            d,
            net,
            _make_reflection(),
            _make_narrative(),
            _make_self_model(),
            synapses=synapses,
        )
        data = load_state(d)

        assert data["synapses"] == {
            "dog\tmammal": 0.73,
            "mammal\tanimal": 0.41,
        }

        fresh = SynapticStore()
        restore_synapses(fresh, data["synapses"])
        assert fresh.get_weight("dog", "mammal") == pytest.approx(0.73)
        assert fresh.get_weight("mammal", "animal") == pytest.approx(0.41)
        assert fresh.get_weight("missing", "connection") == 0.0


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
        fresh_refl = Reflection(fresh_net)
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
        fresh_narr = Narrator(fresh_sm, fresh_net)
        restore_narrative(fresh_narr, data["narrative"])

        # Should have at least one chapter (Awakening)
        assert len(fresh_narr.chapters) >= 1


def test_restore_temporal_link_preserves_finite_weight():
    """Temporal-link weights are not range-limited during restore."""
    narr = _make_narrative()
    restore_narrative(
        narr,
        {
            "chapters": [
                {
                    "title": "Chapter",
                    "theme": "test",
                    "start_time": 1000,
                    "events": [
                        {
                            "id": "event-1",
                            "timestamp": 1000,
                            "summary": "first",
                            "significance": 0.5,
                            "emotion_at_time": {},
                        },
                        {
                            "id": "event-2",
                            "timestamp": 1001,
                            "summary": "second",
                            "significance": 0.5,
                            "emotion_at_time": {},
                        },
                    ],
                }
            ],
            "temporal_links": [
                {
                    "source_id": "event-1",
                    "target_id": "event-2",
                    "link_type": "led_to",
                    "weight": 1.7,
                }
            ],
        },
    )

    assert len(narr._temporal_links) == 1
    assert narr._temporal_links[0].weight == 1.7


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
    from infrastructure.persistence import _atomic_write_state

    events = []
    replace = os.replace

    def record_replace(source, destination):
        events.append("replace")
        replace(source, destination)

    with (
        patch(
            "infrastructure.persistence.os.fsync",
            side_effect=lambda fd: events.append("sync"),
        ),
        patch(
            "infrastructure.persistence.os.replace",
            side_effect=record_replace,
        ),
    ):
        _atomic_write_state(str(tmp_path), {"version": 2})
    assert events == ["sync", "replace", "sync"]


def test_failed_flush_preserves_previous_save(tmp_path):
    from infrastructure.persistence import _atomic_write_state

    path = tmp_path / "cognitive_state.json"
    path.write_bytes(b'{"version":2}')
    with patch(
        "infrastructure.persistence.os.fsync",
        side_effect=OSError("disk error"),
    ):
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
        "edges": [
            {
                "source": "bank#2",
                "target": "money",
                "relation": "related_to",
                "weight": 0.7,
            }
        ],
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


# ─── Semantic memory durability ──────────────────────────────────────
#
# SemanticMemory previously had no serialisation at all, and its
# consolidated facts are labelled origin="semantic", which is in
# DERIVABLE_ORIGINS — so the canonical edge log rejects them. The two
# together meant every fact learned into semantic memory was lost on
# restart: no graph edge, no saved state. These pin that facts and
# schemas now survive a round trip.


def test_semantic_memory_facts_survive_a_round_trip(tmp_path):
    """Facts consolidated into semantic memory must persist.

    Regression test for silent data loss: nothing was written, and the
    edge log cannot hold these edges, so a fact learned from a
    conversation previously existed only in RAM.
    """
    from memory.semantic import Fact, SemanticMemory

    network = ConceptNetwork()
    network.add_concept("dog", confidence=0.9)
    network.add_concept("animal", confidence=0.9)

    dog_fact = Fact("dog", "is_a", "animal", 0.8, 3, 111, 222)
    water_fact = Fact("water", "is_a", "liquid", 0.6)
    sem = SemanticMemory(network=network)
    sem._facts[dog_fact.key] = dog_fact
    sem._facts[water_fact.key] = water_fact
    sem.facts_extracted = 2

    sm = SelfModel()
    save_state(
        data_dir=str(tmp_path),
        network=network,
        reflection=Reflection(network),
        narrative=Narrator(sm, network),
        self_model=sm,
        semantic_memory=sem,
    )

    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert "semantic_memory" in loaded, "semantic_memory was not persisted"

    restored = SemanticMemory(network=network)
    restore_semantic_memory(restored, loaded["semantic_memory"])

    assert restored.fact_count == 2
    assert restored.facts_extracted == 2
    facts = {(f.subject, f.relation, f.object): f for f in restored._facts.values()}
    assert ("dog", "is_a", "animal") in facts
    got = facts[("dog", "is_a", "animal")]
    assert got.confidence == pytest.approx(0.8)
    assert got.source_count == 3
    assert got.extracted_at == 111
    assert got.last_reinforced == 222


def test_semantic_memory_restore_tolerates_corrupt_payload():
    """A corrupt payload costs the facts, not the ability to start."""
    from memory.semantic import SemanticMemory

    network = ConceptNetwork()
    sem = SemanticMemory(network=network)
    sem.restore(
        {
            "facts": [
                {"subject": "ok", "relation": "is_a", "object": "thing"},
                {"relation": "is_a"},
                "not-a-dict",
                {"subject": "b", "relation": "is_a", "object": "c", "confidence": "x"},
            ],
            "schemas": [{"no_concept_key": True}],
        }
    )
    assert sem.fact_count == 1
    assert any(f.subject == "ok" for f in sem._facts.values())


def test_semantic_memory_does_not_restore_stale_priming():
    """Priming is transient and must not survive a restart.

    Restoring a previous session's priming set would bias retrieval
    toward concepts that were salient then, which is not what semantic
    priming models.
    """
    from memory.semantic import SemanticMemory

    network = ConceptNetwork()
    sem = SemanticMemory(network=network)
    sem._primed_concepts["dog"] = 0.9
    sem.restore({"facts": []})
    assert sem._primed_concepts == {}


def test_allostatic_load_round_trip():
    """A real allostatic load survives save/restore."""
    from infrastructure.persistence import (
        _serialize_allostatic_load,
        restore_allostatic_load,
    )
    from limbic_system.emotional_regulator import AllostaticLoadTracker

    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(0.6)
    for _ in range(50):
        tracker.record_stress(0.6, dt=1.0)
    expected = tracker.get_allostatic_load()
    assert expected > 0.0

    data = _serialize_allostatic_load(tracker)
    fresh = AllostaticLoadTracker()
    restore_allostatic_load(fresh, data)

    assert fresh.get_allostatic_load() == expected
    # The elevation counter lives at the top level of the payload, not
    # inside "state" — a mismatch here silently resets chronic-stress
    # tracking on every restart.
    assert fresh._elevated_stress_duration == tracker._elevated_stress_duration
    assert fresh._elevated_stress_duration > 0.0
    # Substrate connectivity is deliberately not restored: until the
    # daemon next reports in, the tracker runs on the fallback.
    assert fresh.get_state().substrate_connected is False


def test_allostatic_restore_preserves_finite_load():
    """Finite saved loads restore verbatim; malformed values do not."""
    from infrastructure.persistence import restore_allostatic_load
    from limbic_system.emotional_regulator import AllostaticLoadTracker

    for saved in (-3.0, 0.5, 50.0):
        tracker = AllostaticLoadTracker()
        restore_allostatic_load(
            tracker,
            {"state": {"allostatic_load": saved, "is_chronic": False}},
        )
        assert tracker.get_allostatic_load() == saved

    for malformed in (float("nan"), float("inf"), "not-a-number"):
        tracker = AllostaticLoadTracker()
        restore_allostatic_load(
            tracker,
            {"state": {"allostatic_load": malformed, "is_chronic": False}},
        )
        assert tracker.get_allostatic_load() == 0.0


def test_allostatic_restore_tolerates_corrupt_duration():
    """A non-finite elevation counter must not make stress permanently chronic."""
    from infrastructure.persistence import restore_allostatic_load
    from limbic_system.emotional_regulator import AllostaticLoadTracker

    tracker = AllostaticLoadTracker()
    restore_allostatic_load(
        tracker,
        {
            "state": {"allostatic_load": 0.5, "is_chronic": True},
            "elevated_stress_duration": float("nan"),
        },
    )
    assert tracker._elevated_stress_duration == 0.0
    # The counter is recomputed on the next sample, so a corrupt value
    # cannot pin the chronic flag.
    tracker.set_inference_load(0.0)
    tracker.record_stress(0.0, dt=1.0)
    assert tracker.is_chronic_stress() is False


def test_allostatic_restore_ignores_removed_fields():
    """Old save files carrying retired fields still restore cleanly."""
    from infrastructure.persistence import restore_allostatic_load
    from limbic_system.emotional_regulator import AllostaticLoadTracker

    tracker = AllostaticLoadTracker()
    restore_allostatic_load(
        tracker,
        {
            "state": {
                "allostatic_load": 0.3,
                "is_chronic": True,
                # Removed fields, present in older saves.
                "acute_stress": 0.9,
                "stress_history": [0.1] * 600,
                "adjusted_baselines": {"cortisol": 0.4},
            },
            "tick_count": 1234,
        },
    )
    assert tracker.get_allostatic_load() == 0.3
    assert tracker.is_chronic_stress() is True


def test_allostatic_serialization_is_compact():
    """The serialized tracker must not carry per-sample history.

    The retired `stress_history` was a 600-sample cortisol deque that
    was appended to every regulation cycle, serialized on every save
    (~3.6 KB), restored, and never read for any decision.
    """
    from infrastructure.persistence import _serialize_allostatic_load
    from limbic_system.emotional_regulator import AllostaticLoadTracker

    tracker = AllostaticLoadTracker()
    tracker.set_inference_load(0.5)
    for _ in range(100):
        tracker.record_stress(0.5, dt=1.0)

    payload = _serialize_allostatic_load(tracker)
    assert "stress_history" not in payload["state"]
    assert "acute_stress" not in payload["state"]
    assert "tick_count" not in payload
    # A flat scalar dict, not a list of samples.
    assert set(payload["state"]) == {"allostatic_load", "is_chronic"}


def test_cognitive_map_save_state_is_json_serializable():
    """CognitiveMap.save_state() must emit only str dict keys.

    Regression: transition destinations and content Counters are keyed
    by int place ids and were saved raw, so every autosave died on
    'Dict key must be str' and the map was never persisted.
    """
    import json as _json
    from collections import Counter

    from temporal_lobe.cognitive_map import CognitiveMap, ContextMap

    ctx = ContextMap(signature=frozenset({(0, 0, 0, 0)}), bounds=(10, 10))
    ctx.transitions[(1, "move")] = Counter({2: 3, 5: 1})
    ctx.contents[1] = Counter({7: 2})
    m = CognitiveMap()
    m.contexts.append(ctx)

    state = m.save_state()

    _json.dumps(state)  # raises TypeError if any dict key is not a str


def test_cognitive_map_save_load_round_trip():
    """The str-keyed save payload restores to the int-keyed map."""
    from collections import Counter

    from temporal_lobe.cognitive_map import CognitiveMap, ContextMap

    ctx = ContextMap(signature=frozenset({(0, 0, 0, 0)}), bounds=(10, 10))
    ctx.transitions[(1, "move")] = Counter({2: 3, 5: 1})
    ctx.contents[1] = Counter({7: 2})
    m = CognitiveMap()
    m.contexts.append(ctx)

    restored = CognitiveMap()
    restored.load_state(m.save_state())

    rctx = restored.contexts[0]
    assert rctx.transitions[(1, "move")] == Counter({2: 3, 5: 1})
    assert rctx.contents[1] == Counter({7: 2})


def test_predictive_coding_intent_marginal_round_trip():
    """The global intent prior must persist with the learned models."""
    from cerebellum.predictive import PredictiveCodingLayer
    from infrastructure.persistence import (
        _serialize_predictive_coding,
        restore_predictive_coding,
    )

    predictive = PredictiveCodingLayer()
    predictive._intent_marginal["question"] = 1.25
    payload = _serialize_predictive_coding(predictive)

    restored = PredictiveCodingLayer()
    restore_predictive_coding(restored, payload)

    assert restored._intent_marginal["question"] == 1.25


def test_restore_reflection_replaces_stale_and_keeps_latest_pattern():
    """Restore is a replacement, and pattern dedupe keeps the newest item."""
    reflection = _make_reflection()
    restore_reflection(
        reflection,
        {
            "insights": [
                {
                    "type": "pattern",
                    "content": "Overusing 'inform' (7/10)",
                    "confidence": 0.6,
                },
                {
                    "type": "pattern",
                    "content": "Overusing 'inform' (8/10)",
                    "confidence": 0.7,
                },
            ]
        },
    )

    assert [i.content for i in reflection.insights] == ["Overusing 'inform' (8/10)"]


def test_restore_narrative_replaces_constructor_seed_state():
    """A staged/fresh Narrator's seed snapshot and hierarchy cannot leak in."""
    narrative = _make_narrative()
    restore_narrative(
        narrative,
        {
            "chapters": [],
            "personality_snapshots": [],
            "event_counter": 0,
            "hierarchy": {},
            "life_script": [],
            "temporal_links": [],
        },
    )

    assert narrative._personality_snapshots == []
    assert all(not nodes for nodes in narrative._hierarchy.values())
    assert narrative._temporal_links == []
    assert narrative._links_out == {}
    assert narrative._links_in == {}
    assert narrative.life_script.milestones == []


def test_restore_procedural_memory_replaces_stale_skills():
    """Skills absent from the save must not survive restore."""
    from infrastructure.persistence import restore_procedural_memory
    from memory import ProceduralMemory

    memory = ProceduralMemory()
    memory.learn_skill("stale", "old trigger")
    restore_procedural_memory(
        memory,
        {
            "skills": [
                {
                    "name": "saved",
                    "trigger_condition": "new trigger",
                    "strategy": {"cognitive_route": "general"},
                }
            ]
        },
    )

    assert set(memory._skills) == {"saved"}
    assert memory._trigger_index == {"new trigger": ["saved"]}


def test_restore_spaced_repetition_replaces_and_preserves_zero():
    """Zero is a meaningful saved value, not a missing-value sentinel."""
    from infrastructure.persistence import restore_spaced_repetition
    from memory import ReviewRecord, SpacedRepetitionScheduler

    scheduler = SpacedRepetitionScheduler(_make_network())
    scheduler._records["stale"] = ReviewRecord(concept="stale", stability=1.0)
    restore_spaced_repetition(
        scheduler,
        {
            "records": [
                {
                    "concept": "saved",
                    "stability": 0.0,
                    "review_count": 2,
                    "success_count": 1,
                    "last_review": 0.0,
                    "ease_factor": 0.0,
                }
            ]
        },
    )

    assert set(scheduler._records) == {"saved"}
    record = scheduler._records["saved"]
    assert record.stability == 0.0
    assert record.last_review == 0.0
    assert record.ease_factor == 0.0


def test_restore_td_learner_replaces_history_and_enforces_cap():
    """TD restore must not append saved transitions to staged history."""
    from basal_ganglia.td import TDLearner, TDTransition
    from infrastructure.persistence import restore_td_learner

    learner = TDLearner(_make_network())
    learner._history.append(
        TDTransition(state=["stale"], reward=0.0, next_state=[], rpe=0.0, timestamp=0)
    )
    history = [
        {
            "state": [f"state-{i}"],
            "reward": 0.0,
            "next_state": [],
            "rpe": 0.0,
            "timestamp": i,
        }
        for i in range(learner._max_history + 1)
    ]
    restore_td_learner(learner, {"history": history})

    assert len(learner._history) == learner._max_history
    assert learner._history[0].state == ["state-1"]
    assert "stale" not in learner._history[0].state


def test_restore_emotional_memory_replaces_stale_memories():
    """Emotional memories absent from the save must not survive restore."""
    from infrastructure.persistence import restore_emotional_memory
    from memory import EmotionalMemory, EmotionalMemorySystem

    memory = EmotionalMemorySystem()
    memory._memories["stale"] = EmotionalMemory(stimulus="stale")
    restore_emotional_memory(
        memory,
        {
            "memories": [
                {
                    "stimulus": "saved",
                    "emotional_tag": [0.5] * 12,
                    "association_strength": 0.7,
                }
            ]
        },
    )

    assert set(memory._memories) == {"saved"}
    assert memory._memories["saved"].association_strength == 0.7


def test_restore_attractor_replaces_patterns_and_missing_weights():
    """An old save without weights resets the matrix instead of keeping stale weights."""
    from infrastructure.persistence import restore_attractor
    from memory import AttractorNetwork, AttractorPattern

    attractor = AttractorNetwork(size=4)
    attractor._patterns["stale"] = AttractorPattern(id="stale", vector=[1] * 4)
    attractor._weights[0][1] = 0.9
    restore_attractor(
        attractor,
        {
            "size": 4,
            "patterns": [{"id": "saved", "vector": [1, -1, 1, -1]}],
        },
    )

    assert set(attractor._patterns) == {"saved"}
    assert attractor._weights == [[0.0] * 4 for _ in range(4)]


def test_restore_error_monitor_replaces_stale_monitoring_state():
    """Error history, last error, and channel demand are replace-on-restore."""
    from infrastructure.persistence import restore_error_monitor
    from self.reflection import ErrorMonitor

    monitor = ErrorMonitor()
    monitor.record_prediction("yes", "no")
    assert monitor._errors
    restore_error_monitor(
        monitor,
        {
            "error_count": 0,
            "cumulative_error_signal": 0.0,
            "errors": [],
            "channel_demand": {"decision": 1.25},
        },
    )

    assert monitor._errors == deque()
    assert monitor._last_error is None
    assert monitor._channel_demand == {"decision": 1.25}


def test_restore_hpa_axis_clears_transient_delay_state():
    """HPA delay buffers and onset bookkeeping cannot leak across restore."""
    from infrastructure.persistence import restore_hpa_axis
    from limbic_system.emotional_regulator import HPAAxis

    axis = HPAAxis()
    axis.trigger_stress(0.8)
    axis.tick(1.0)
    restore_hpa_axis(axis, {"state": {"crh_level": 0.4, "stress_active": True}})

    assert axis._state.crh_level == 0.4
    assert not axis._crh_history
    assert not axis._acth_history
    assert axis._stress_onset_time is None
    assert axis._peak_reached is False
    assert axis._avg_dt == 1.0


def test_restore_self_model_replaces_stale_self_knowledge():
    """Self-knowledge keys absent from the save must not survive restore."""
    model = _make_self_model()
    restore_self_model(
        model,
        {
            "self_knowledge": [{"key": "saved_key", "value": "saved_value"}],
        },
    )

    assert model.self_knowledge == {"saved_key": "saved_value"}


def test_restore_self_directed_learner_replaces_learning_log():
    """The learning log is a serialized collection, not a merge target."""
    from self.learning import LearningEvent

    learner = SelfDirectedLearner(_make_network())
    learner._learning_log.append(LearningEvent("stale", "stale event"))
    restore_self_directed_learner(learner, {"learning_log": []})

    assert list(learner._learning_log) == []


def test_restore_network_sanitizes_malformed_numeric_fields():
    """Malformed learned numbers cannot poison a restored network."""
    network = ConceptNetwork()
    restore_network(
        network,
        {
            "concepts": [
                {
                    "id": "saved",
                    "activation": float("nan"),
                    "confidence": "bad",
                }
            ],
            "edges": [
                {
                    "source": "saved",
                    "target": "saved",
                    "relation": "related_to",
                    "weight": float("inf"),
                }
            ],
        },
    )

    concept = network.get_concept("saved")
    assert concept.activation == 0.0
    assert concept.confidence == 0.5
    assert network._edges[0].weight == 0.5


def test_restore_td_learner_preserves_finite_lam():
    """λ is a persisted value, not a range-limited one."""
    from basal_ganglia.td import TDLearner
    from infrastructure.persistence import restore_td_learner

    learner = TDLearner(_make_network())
    restore_td_learner(learner, {"lam": 1.5})
    assert learner.lam == 1.5

    restore_td_learner(learner, {"lam": float("nan")})
    assert learner.lam == 1.5


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

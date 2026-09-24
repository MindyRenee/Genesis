"""Tests for the integrated intelligence: thought composer, working memory,
and the connections between systems.

These tests verify that Genesis can compose novel thoughts from its
knowledge, track attention across turns, and actually use its
intelligence modules to drive its responses.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from genesis_cognitive.cognition.thought_composer import ThoughtComposer
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.memory import Turn, WorkingMemory
from genesis_cognitive.reasoning import ReasoningEngine

logger = logging.getLogger(__name__)

# ─── Helpers ───────────────────────────────────────────────────


def _make_emotion(
    label: str = "neutral",
    valence: float = 0.0,
    alertness: float = 0.5,
    creativity: float = 0.5,
    caution=0.3,
    openness_to_engage=0.7,
) -> EmotionalState:
    """Construct a emotion for tests."""
    return EmotionalState(
        label=label,
        nuance="baseline",
        cognitive_style="steady",
        valence=valence,
        alertness=alertness,
        plasticity=0.5,
        creativity=creativity,
        caution=caution,
        openness_to_engage=openness_to_engage,
    )


def _make_network_with_knowledge() -> ConceptNetwork:
    """Create a concept network with seeded knowledge."""
    net = ConceptNetwork()
    net.add_concept("dog", confidence=0.8)
    net.add_concept("mammal", confidence=0.8)
    net.add_concept("animal", confidence=0.8)
    net.add_concept("spark", confidence=0.7)
    net.add_concept("fire", confidence=0.7)
    net.add_concept("smoke", confidence=0.7)
    net.add_edge("dog", "mammal", RelationType.IS_A, 0.9)
    net.add_edge("mammal", "animal", RelationType.IS_A, 0.9)
    net.add_edge("spark", "fire", RelationType.CAUSES, 0.8)
    net.add_edge("fire", "smoke", RelationType.CAUSES, 0.8)
    return net


# ─── Thought composer tests ────────────────────────────────────


def test_compose_about_known_concept() -> None:
    """Composer can compose a thought about a concept it knows."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_about("dog", emotion)
    assert thought is not None
    assert "dog" in thought.content.lower()
    assert thought.confidence > 0.3


def test_compose_about_unknown_concept() -> None:
    """Composer returns None for unknown concepts."""
    net = ConceptNetwork()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_about("nonexistent", emotion)
    assert thought is None


def test_compose_about_isolated_concept() -> None:
    """Composer returns None for concepts with no relationships.

    Per AGENTS.md, Genesis never recites pre-written "I can't
    articulate" templates. If it has no knowledge to compose from,
    it stays silent (returns None).
    """
    net = ConceptNetwork()
    net.add_concept("mystery", confidence=0.2)
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_about("mystery", emotion)
    assert thought is None  # no knowledge → silence, not a template


def test_compose_answer_from_reasoning() -> None:
    """Composer answers using reasoning, not hardcoded strings."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    # "What is a dog?" → should use the IS_A chain
    thought = composer.compose_answer("What is a dog?", ["dog"], emotion, "what_is")
    assert thought is not None
    # Should mention dog or mammal or animal
    content_lower = thought.content.lower()
    assert any(w in content_lower for w in ["dog", "mammal", "animal"])


def test_compose_answer_causal() -> None:
    """Composer can answer causal questions."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_answer("What causes smoke?", ["smoke"], emotion, "what_causes")
    assert thought is not None
    # Should mention fire or spark (the causal chain)
    content_lower = thought.content.lower()
    assert any(w in content_lower for w in ["fire", "spark", "smoke"])


def test_compose_reflection() -> None:
    """Composer can reflect on a topic."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion(creativity=0.7)

    thought = composer.compose_reflection("dog", emotion)
    assert thought is not None
    assert thought.self_reflection
    assert "dog" in thought.content.lower()


def test_compose_reflection_unknown_topic() -> None:
    """Composer returns None for topics with no knowledge.

    Per AGENTS.md, Genesis doesn't recite "I can't articulate"
    templates for unknown topics. It stays silent (returns None).
    """
    net = ConceptNetwork()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_reflection("quantum_mechanics", emotion)
    assert thought is None  # no knowledge → silence, not a template


def test_compose_novel_connection() -> None:
    """Composer can express novel connections."""
    net = _make_network_with_knowledge()
    # Activate concepts to create potential for novel connections
    dog = net.get_concept("dog")
    assert dog is not None
    dog.activation = 0.8
    animal = net.get_concept("animal")
    assert animal is not None
    animal.activation = 0.7
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion(creativity=0.7)

    thought = composer.compose_novel_connection(emotion)
    # May or may not find a connection, but shouldn't crash
    if thought is not None:
        assert len(thought.content) > 0
        assert thought.self_reflection


def test_compose_answer_no_topics() -> None:
    """Composer returns None when no topics are provided."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    thought = composer.compose_answer("Hello", [], emotion)
    assert thought is None


def test_compose_about_tracks_said() -> None:
    """Composer tracks what it's said about concepts."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    composer.compose_about("dog", emotion)
    assert composer.has_said_similar("dog", "dog is a mammal", threshold=0.3)


def test_compose_about_varies() -> None:
    """Composer produces varied output for the same concept."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=None)
    emotion = _make_emotion(creativity=0.8, openness_to_engage=0.8)

    outputs = set()
    for _ in range(10):
        thought = composer.compose_about("dog", emotion)
        if thought:
            outputs.add(thought.content[:50])

    # Should have some variation (not all identical)
    assert len(outputs) > 1


def test_compose_confidence_reflects_knowledge() -> None:
    """Confidence is higher for well-known concepts.

    A concept with relationships produces a thought; a concept with
    no knowledge returns None (silence, not a template).
    """
    net = ConceptNetwork()
    net.add_concept("certain", confidence=0.9)
    net.add_concept("other", confidence=0.9)
    net.add_edge("certain", "other", RelationType.RELATED_TO, 0.9)

    net.add_concept("uncertain", confidence=0.2)

    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    certain_thought = composer.compose_about("certain", emotion)
    uncertain_thought = composer.compose_about("uncertain", emotion)

    assert certain_thought is not None  # has knowledge → composes
    assert uncertain_thought is None    # no knowledge → silence


# ─── Working memory tests ──────────────────────────────────────


def test_working_memory_basic() -> None:
    """Working memory starts empty."""
    wm = WorkingMemory()
    assert wm.total_turns == 0
    assert wm.thread_count == 0
    assert wm.get_top_attention() == []


def test_working_memory_update() -> None:
    """Working memory tracks turns."""
    wm = WorkingMemory()
    turn = Turn(
        user_input="Tell me about cognition",
        genesis_response="I know that cognition is related to mind.",
        topics=["cognition", "mind"],
        intent="question",
    )
    wm.update(turn)
    assert wm.total_turns == 1
    assert wm.thread_count == 1
    assert "cognition" in wm.get_top_attention()


def test_working_memory_attention_decay() -> None:
    """Attention decays over turns."""
    wm = WorkingMemory(attention_decay=0.5)
    turn1 = Turn("about dogs", "dogs are mammals", ["dog"], "statement")
    wm.update(turn1)
    assert "dog" in wm.get_attention()

    # Add a turn about something else
    turn2 = Turn("about cats", "cats are mammals", ["cat"], "statement")
    wm.update(turn2)

    # Dog attention should have decayed
    attention = wm.get_attention()
    if "dog" in attention:
        assert attention["dog"] < 0.8  # decayed from initial activation


def test_working_memory_thread_continuity() -> None:
    """Related turns stay in the same thread."""
    wm = WorkingMemory()
    turn1 = Turn("about dogs", "dogs are mammals", ["dog", "mammal"], "statement")
    turn2 = Turn("about mammals", "mammals are animals", ["mammal", "animal"], "statement")

    wm.update(turn1)
    wm.update(turn2)

    # Should be in the same thread (shared concept: mammal)
    assert wm.thread_count == 1
    assert wm.turn_count == 2


def test_working_memory_thread_shift() -> None:
    """Unrelated turns start a new thread."""
    wm = WorkingMemory()
    turn1 = Turn("about dogs", "dogs are mammals", ["dog", "mammal"], "statement")
    turn2 = Turn("about cooking", "I like cooking", ["cooking", "food"], "statement")

    wm.update(turn1)
    wm.update(turn2)

    # Should be in different threads (no overlap)
    assert wm.thread_count == 2


def test_working_memory_was_topic_discussed() -> None:
    """Can check if a topic was discussed recently."""
    wm = WorkingMemory()
    turn = Turn("about cognition", "I know about cognition", ["cognition"], "question")
    wm.update(turn)

    assert wm.was_topic_discussed("cognition")
    assert not wm.was_topic_discussed("cooking")


def test_working_memory_open_questions() -> None:
    """Can track open questions."""
    wm = WorkingMemory()
    wm.add_open_question("What is cognition?")
    wm.add_open_question("Do I have free will?")

    questions = wm.get_open_questions()
    assert len(questions) == 2
    assert "What is cognition?" in questions

    wm.clear_open_question("What is cognition?")
    questions = wm.get_open_questions()
    assert len(questions) == 1


def test_working_memory_describe() -> None:
    """Can describe its state."""
    wm = WorkingMemory()
    turn = Turn("about dogs", "dogs are mammals", ["dog"], "statement")
    wm.update(turn)

    desc = wm.describe()
    assert "Working memory" in desc
    assert "dog" in desc


def test_working_memory_recent_turns() -> None:
    """Can retrieve recent turns."""
    wm = WorkingMemory(max_turns=5)
    for i in range(7):
        turn = Turn(f"input {i}", f"response {i}", [f"topic{i}"], "statement")
        wm.update(turn)

    recent = wm.get_recent_turns(3)
    assert len(recent) == 3
    # Should be the last 3
    assert recent[-1].user_input == "input 6"


def test_working_memory_is_on_topic() -> None:
    """Can check if topics match the current thread."""
    wm = WorkingMemory()
    turn = Turn("about dogs", "dogs are mammals", ["dog", "mammal"], "statement")
    wm.update(turn)

    assert wm.is_on_topic(["dog"])
    assert wm.is_on_topic(["mammal"])
    assert not wm.is_on_topic(["cooking"])


# ─── Integration: composer + reasoning + network ───────────────


def test_integration_learning_changes_responses() -> None:
    """After learning new relationships, responses change.

    Before learning, a concept with no edges returns None (silence).
    After learning, it produces a thought with the new knowledge.
    """
    net = ConceptNetwork()
    net.add_concept("cognition", confidence=0.6)
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    # Before learning: no knowledge → silence (None)
    thought_before = composer.compose_about("cognition", emotion)
    assert thought_before is None

    # Learn a new relationship
    net.add_concept("neural_activity", confidence=0.7)
    net.add_edge("cognition", "neural_activity", RelationType.EMERGES_FROM, 0.8)

    # After learning: its response should include the new knowledge
    thought_after = composer.compose_about("cognition", emotion)
    assert thought_after is not None
    assert "neural" in thought_after.content.lower() or "emerges" in thought_after.content.lower()


def test_integration_reasoning_drives_answer() -> None:
    """Reasoning results drive composed answers."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    emotion = _make_emotion()

    # "What is a dog?" — should use the IS_A chain: dog → mammal → animal
    thought = composer.compose_answer("What is a dog?", ["dog"], emotion, "what_is")
    assert thought is not None
    # The answer should reference the reasoning chain
    content = thought.content.lower()
    assert any(w in content for w in ["dog", "mammal", "animal"])


def test_integration_working_memory_with_composer() -> None:
    """Working memory and composer work together."""
    net = _make_network_with_knowledge()
    reasoner = ReasoningEngine(net)
    composer = ThoughtComposer(net, reasoner, seed=42)
    wm = WorkingMemory()
    emotion = _make_emotion()

    # First turn: ask about dogs
    thought1 = composer.compose_about("dog", emotion)
    assert thought1 is not None
    turn1 = Turn("What is a dog?", thought1.content, ["dog", "mammal"], "question")
    wm.update(turn1)

    # Second turn: ask about mammals (related to dog)
    thought2 = composer.compose_about("mammal", emotion)
    assert thought2 is not None
    turn2 = Turn("What is a mammal?", thought2.content, ["mammal", "animal"], "question")
    wm.update(turn2)

    # Working memory should track both
    assert wm.total_turns == 2
    assert wm.was_topic_discussed("dog")
    assert wm.was_topic_discussed("mammal")
    # Should be in the same thread (dog and mammal are related)
    assert wm.thread_count == 1


# ─── Unreliable-edge filtering (spoken knowledge quality) ─────


def test_gather_knowledge_excludes_hub_attachment_edges() -> None:
    """Reachability scaffolding is not spoken as knowledge.

    ``connect_orphans`` attaches bare concepts to super hubs with
    ``hub_attachment`` edges (weight 0.10) purely so the graph stays
    connected. "makes --related_to--> emotion" is such an edge — a
    verb attached to a hub, with no semantic content. It must not be
    composed into speech ("Makes related to emotion.").
    """
    net = ConceptNetwork()
    net.add_concept("makes", confidence=0.9)
    net.add_concept("emotion", confidence=1.0)
    net.add_edge(
        "makes", "emotion", RelationType.RELATED_TO, 0.10, origin="hub_attachment"
    )
    composer = ThoughtComposer(net, ReasoningEngine(net), seed=42)

    assert composer._gather_knowledge("makes", depth=2) == []


def test_gather_knowledge_excludes_low_weight_edges() -> None:
    """Edges below the reliability floor are not spoken."""
    net = ConceptNetwork()
    net.add_concept("a", confidence=0.8)
    net.add_concept("b", confidence=0.8)
    net.add_edge("a", "b", RelationType.RELATED_TO, 0.1, origin="co_occurrence")
    composer = ThoughtComposer(net, ReasoningEngine(net), seed=42)

    assert composer._gather_knowledge("a", depth=2) == []


def test_gather_knowledge_keeps_reliable_edges() -> None:
    """Reliable, sufficiently-weighted edges are still spoken."""
    net = _make_network_with_knowledge()
    composer = ThoughtComposer(net, ReasoningEngine(net), seed=42)

    knowledge = composer._gather_knowledge("dog", depth=2)
    assert any(rel == "is a" and tgt == "mammal" for rel, tgt, _ in knowledge)


# ─── Definition-only concepts ─────────────────────────────────


def test_compose_about_definition_only_concept() -> None:
    """A concept with a definition but no edges still gets an answer.

    Previously such a concept produced nothing (the definition was
    only emitted in "definition" mode), so a junk neighbour could win
    the answer instead. The stored definition is real knowledge.
    """
    net = ConceptNetwork()
    net.add_concept(
        "happy",
        confidence=1.0,
        properties={"definition": "enjoying or showing or marked by joy or pleasure"},
    )
    composer = ThoughtComposer(net, ReasoningEngine(net), seed=42)

    thought = composer.compose_about("happy", _make_emotion(), focused=True)
    assert thought is not None
    assert "joy" in thought.content.lower()


# ─── Natural relation phrasing ────────────────────────────────


def test_compose_natural_fact_uses_seeded_verb() -> None:
    """With relation verbs seeded, facts are phrased with a natural verb.

    The seeder was never called, so ``find_relation_verbs`` returned []
    and the composer emitted the raw semantic triple
    ("makes related to emotion"). With the verbs seeded it says
    "makes relates to emotion".
    """
    net = ConceptNetwork()
    net.add_concept("makes", confidence=0.9)
    net.add_concept("emotion", confidence=1.0)
    net.seed_relation_verbs()
    composer = ThoughtComposer(net, ReasoningEngine(net), seed=42)

    fact = composer._compose_natural_fact("makes", "related_to", "emotion")
    # Any of the seeded verbs is fine; the raw triple is not.
    assert any(v in fact for v in ("relates to", "connects to", "has to do with"))
    assert "related to" not in fact


# ─── Test runner ──────────────────────────────────────────────


def run_all() -> bool:
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
    logger.info(f"\n  Integrated intelligence tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

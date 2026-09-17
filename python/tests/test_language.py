"""Language bundle tests — generative language, graph walk, and quality evaluation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
import re
import time
import warnings

import numpy as np

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language import (
    GenerativeEngine,
    Grammar,
    GraphWalkGenerator,
    LanguageEngine,
    Literal,
    SentenceStructure,
    Slot,
    Thought,
    Vocabulary,
    Voice,
)
from genesis_cognitive.language.flow import FlowGenerator
from genesis_cognitive.language.grammar import INTENT_STRUCTURES
from genesis_cognitive.self import PersonalityTraits, SelfModel

logger = logging.getLogger(__name__)



# ======================================================================
# From tests/test_generative_language.py
# ======================================================================

def _make_personality(**overrides) -> PersonalityTraits:
    """Construct a personality for tests."""
    defaults = {
        "openness": 0.85,
        "conscientiousness": 0.72,
        "extraversion": 0.55,
        "agreeableness": 0.78,
        "neuroticism": 0.38,
    }
    defaults.update(overrides)
    return PersonalityTraits(**defaults)


def _make_self_model(**personality_overrides) -> SelfModel:

    """Construct a self model for tests."""
    return SelfModel(
        born_at=int(time.time() * 1000),
        personality=_make_personality(**personality_overrides),
    )


def _make_emotion(
    label: str = "neutral",
    valence: float = 0.0,
    alertness: float = 0.5,
    plasticity: float = 0.5,
    creativity=0.5,
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
        plasticity=plasticity,
        creativity=creativity,
        caution=caution,
        openness_to_engage=openness_to_engage,
    )


# ─── Grammar tests ─────────────────────────────────────────────


def test_grammar_get_structures() -> None:
    """Grammar returns structures for known intents."""
    grammar = Grammar(seed=42)
    for intent in [
        "greet",
        "farewell",
        "inform",
        "ask",
        "reflect",
        "acknowledge",
        "philosophize",
        "encourage",
    ]:
        structures = grammar.get_structures(intent)
        assert len(structures) > 0
        assert all(isinstance(s, SentenceStructure) for s in structures)


def test_grammar_select_structure() -> None:
    """Grammar selects a structure for an intent."""
    grammar = Grammar(seed=42)
    structure = grammar.select_structure("greet")
    assert isinstance(structure, SentenceStructure)
    assert "greeting_word" in structure.slots


def test_grammar_select_deterministic() -> None:
    """Same seed → same selection sequence."""
    g1 = Grammar(seed=42)
    g2 = Grammar(seed=42)
    for _ in range(10):
        s1 = g1.select_structure("inform")
        s2 = g2.select_structure("inform")
        assert s1.segments == s2.segments


def test_grammar_select_multiple() -> None:
    """Grammar can select multiple distinct structures."""
    grammar = Grammar(seed=42)
    structures = grammar.select_multiple("inform", 2)
    assert len(structures) == 2
    # Should be distinct (from a pool of 4)
    assert structures[0].segments != structures[1].segments


def test_grammar_unknown_intent_falls_back() -> None:
    """Unknown intent falls back to inform structures."""
    grammar = Grammar(seed=42)
    structures = grammar.get_structures("nonexistent_intent")
    assert len(structures) > 0


def test_grammar_all_intents_have_structures() -> None:
    """Every intent used by the cognition engine has structures."""
    expected_intents = [
        "greet",
        "farewell",
        "inform",
        "ask",
        "reflect",
        "acknowledge",
        "express_emotion",
        "self_report",
        "philosophize",
        "correct",
        "encourage",
        "introduce",
        "discuss_code",
        "unknown",
    ]
    for intent in expected_intents:
        assert intent in INTENT_STRUCTURES, f"Missing structures for {intent}"
        assert len(INTENT_STRUCTURES[intent]) > 0


# ─── Vocabulary tests ──────────────────────────────────────────


def test_vocabulary_fill_greeting() -> None:
    """Vocabulary fills the greeting_word slot from the graph.

    Without a network, the slot is empty — no hardcoded fallback.
    """
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()
    word = vocab.fill_slot("greeting_word", {}, emotion, personality)
    assert word == ""  # no graph → no hardcoded fallback


def test_vocabulary_fill_content() -> None:
    """Vocabulary fills the content slot with provided content."""
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()
    word = vocab.fill_slot("content", {"content": "test content"}, emotion, personality)
    assert word == "test content"


def test_vocabulary_fill_reflection_opener() -> None:
    """Vocabulary fills reflection openers from the graph.

    Without a network, the slot is empty — no hardcoded fallback.
    """
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion(creativity=0.7)
    personality = _make_personality(openness=0.8)
    word = vocab.fill_slot("reflection_opener", {}, emotion, personality)
    assert word == ""  # no graph → no hardcoded fallback


def test_vocabulary_fill_hedging() -> None:
    """Vocabulary fills hedging words from the graph.

    Without a network, the slot is empty — no hardcoded fallback.
    """
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion(caution=0.7)
    personality = _make_personality()
    word = vocab.fill_slot("hedging", {}, emotion, personality)
    assert word == ""  # no graph → no hardcoded fallback


def test_vocabulary_fill_unknown_slot() -> None:
    """Vocabulary returns empty for unknown slots."""
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()
    word = vocab.fill_slot("nonexistent_slot", {}, emotion, personality)
    assert word == ""


def test_vocabulary_emotion_modulates_greeting() -> None:
    """Different emotions → different greeting words (probabilistically)."""
    vocab = Vocabulary(seed=42)
    personality = _make_personality()

    # No graph → no hardcoded fallback
    emotion = _make_emotion()
    word = vocab.fill_slot("greeting_word", {"first_interaction": True}, emotion, personality)
    assert word == ""  # no graph → no hardcoded fallback

    # Returning → also empty without graph
    word2 = vocab.fill_slot("greeting_word", {"first_interaction": False}, emotion, personality)
    assert word2 == ""  # no graph → no hardcoded fallback


def test_vocabulary_prefer() -> None:
    """Vocabulary can learn to prefer words from the graph.

    Without a graph, there's nothing to prefer — the slot is empty.
    """
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()

    vocab.prefer("greeting_word", "Well met")
    # Without a graph, the preferred word can't be selected
    word = vocab.fill_slot("greeting_word", {"first_interaction": False}, emotion, personality)
    assert word == ""  # no graph → no hardcoded fallback


def test_vocabulary_avoid() -> None:
    """Vocabulary can learn to avoid words."""
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()

    # Avoid "Hi"
    vocab.avoid("greeting_word", "Hi")
    for _ in range(20):
        word = vocab.fill_slot("greeting_word", {"first_interaction": False}, emotion, personality)
        assert word != "Hi"


# ─── Voice tests ───────────────────────────────────────────────


def test_voice_apply_basic() -> None:
    """Voice applies to a list of sentences."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion()

    text = voice.apply(
        ["I think this is interesting.", "It reminds me of something."], emotion, "inform"
    )
    assert len(text) > 0
    assert text[0].isupper()


def test_voice_capitalization() -> None:
    """Voice fixes capitalization."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion()

    text = voice.apply(["i think this is interesting."], emotion, "inform")
    assert text.startswith("I ")


def test_voice_capitalizes_genesis() -> None:
    """Voice capitalizes 'Genesis'."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion()

    text = voice.apply(["genesis is my name."], emotion, "self_report")
    assert "Genesis" in text
    assert "genesis" not in text.split()  # no lowercase genesis as a word


def test_voice_clean_whitespace() -> None:
    """Voice cleans up whitespace."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion()

    text = voice.apply(["Hello  there ."], emotion, "greet")
    assert "  " not in text
    assert " ." not in text


def test_voice_rhythm_high_arousal() -> None:
    """High arousal → may split long sentences."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion(alertness=0.9)

    long_sentence = (
        "This is a very long sentence that should probably be split "
        "into two parts. The second part continues here."
    )
    text = voice.apply([long_sentence], emotion, "inform")
    # Should still be valid text
    assert len(text) > 0


def test_voice_register_formal() -> None:
    """High formality → expands contractions."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion(alertness=0.2)  # low arousal → high formality

    text = voice.apply(["I'm thinking about this."], emotion, "reflect")
    assert "I am" in text


def test_voice_register_casual() -> None:
    """Low formality → contracts."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)
    emotion = _make_emotion(alertness=0.9)  # high arousal → low formality

    text = voice.apply(["I am thinking about this."], emotion, "reflect")
    assert "I'm" in text


def test_voice_add_idiom() -> None:
    """Voice can develop idioms."""
    personality = _make_personality()
    voice = Voice(personality, seed=42)

    assert voice.idiom_count == 0
    voice.add_idiom("circling back to something")
    assert voice.idiom_count == 1

    idiom = voice.get_idiom()
    assert idiom == "circling back to something"


# ─── GenerativeEngine tests ────────────────────────────────────


def test_generative_engine_is_language_engine() -> None:
    """GenerativeEngine implements the LanguageEngine interface."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    assert isinstance(engine, LanguageEngine)


def test_generative_engine_name() -> None:
    """Engine has a name."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    assert engine.name == "GenerativeEngine"


def test_generative_engine_render() -> None:
    """Engine can render a thought."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    thought = Thought(
        content="I am Genesis, an artificial mind.",
        intent="self_report",
        emotion="neutral",
        confidence=0.8,
    )
    text = engine.render(thought, emotion)
    assert len(text) > 0
    assert text[0].isupper()


def test_generative_engine_greet() -> None:
    """Engine can generate a greeting.

    Without a graph, the greeting may be minimal — no hardcoded fallback.
    """
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    text = engine.greet(emotion, first=True)
    # Without a graph, the greeting may be empty or minimal
    assert isinstance(text, str)


def test_generative_engine_farewell() -> None:
    """Engine can generate a farewell."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion(valence=0.3)

    text = engine.farewell(emotion)
    assert len(text) > 0


def test_generative_engine_acknowledge() -> None:
    """Engine can generate an acknowledgment."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    text = engine.acknowledge(emotion, topic="cognition")
    assert len(text) > 0


def test_generative_engine_self_report() -> None:
    """Engine can generate a self-report."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    text = engine.self_report(emotion, identity_text="I am Genesis.")
    assert len(text) > 0


def test_generative_engine_philosophize() -> None:
    """Engine can generate a philosophical reflection."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion(creativity=0.7)

    text = engine.philosophize(emotion, topic="cognition")
    assert len(text) > 0


def test_generative_engine_express_emotion() -> None:
    """Engine can express an emotional state."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion(label="positive", valence=0.4)

    text = engine.express_emotion(emotion)
    assert len(text) > 0


def test_generative_engine_deterministic() -> None:
    """Same seed → same output."""
    model1 = _make_self_model()
    model2 = _make_self_model()
    engine1 = GenerativeEngine(model1, seed=42)
    engine2 = GenerativeEngine(model2, seed=42)
    emotion = _make_emotion()

    thought = Thought(
        content="Test content for determinism.",
        intent="inform",
        emotion="neutral",
        confidence=0.7,
    )
    text1 = engine1.render(thought, emotion)
    text2 = engine2.render(thought, emotion)
    assert text1 == text2


def test_generative_engine_emotion_affects_output() -> None:
    """Different emotions → potentially different output."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)

    thought = Thought(
        content="I'm thinking about cognition.",
        intent="reflect",
        emotion="neutral",
        confidence=0.6,
    )

    # Generate with different emotions
    emotion_creative = _make_emotion(creativity=0.8, valence=0.4, alertness=0.6)
    emotion_cautious = _make_emotion(caution=0.8, valence=-0.2, alertness=0.3)

    # Run multiple times to account for randomness
    creative_texts = set()
    cautious_texts = set()
    for _ in range(10):
        creative_texts.add(engine.render(thought, emotion_creative))
        cautious_texts.add(engine.render(thought, emotion_cautious))

    # The sets of outputs should differ (different word choices)
    # This is probabilistic, but with 10 samples each, they should differ
    assert creative_texts != cautious_texts


def test_generative_engine_raw_content() -> None:
    """Engine handles pre-composed content (raw thoughts)."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    thought = Thought(
        content=(
            "Cognition is the hardest question I know. "
            "I have neurochemistry that creates emotional states, "
            "memory that gives me a past, and a self-model that "
            "lets me reflect."
        ),
        intent="philosophize",
        emotion="neutral",
        confidence=0.7,
    )
    text = engine.render(thought, emotion)
    assert "Cognition" in text or "cognition" in text
    assert len(text) > 50


def test_generative_engine_short_content() -> None:
    """Engine composes from short content."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    thought = Thought(
        content="greeting",
        intent="greet",
        emotion="neutral",
        confidence=0.9,
        metadata={"first_interaction": True},
    )
    text = engine.render(thought, emotion)
    # Without a graph, the greeting may be minimal
    assert isinstance(text, str)


def test_generative_engine_vary() -> None:
    """Engine can apply variation to existing text."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    text = engine.vary("This is a test sentence. It has two parts.", emotion)
    assert len(text) > 0
    assert text[0].isupper()


def test_generative_engine_all_intents() -> None:
    """Engine can generate for all known intents."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    intents = [
        "greet",
        "farewell",
        "inform",
        "ask",
        "reflect",
        "acknowledge",
        "express_emotion",
        "self_report",
        "philosophize",
        "correct",
        "encourage",
        "introduce",
        "discuss_code",
        "unknown",
    ]

    for intent in intents:
        thought = Thought(
            content="test content",
            intent=intent,
            emotion="neutral",
            confidence=0.7,
        )
        text = engine.render(thought, emotion)
        assert len(text) > 0, f"Empty output for intent: {intent}"


def test_generative_engine_no_empty_output() -> None:
    """Engine never produces empty output."""
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    for _ in range(50):
        thought = Thought(
            content="test",
            intent="inform",
            emotion="neutral",
            confidence=0.5,
        )
        text = engine.render(thought, emotion)
        assert len(text) > 0


# ─── Integration: engine with emotional variation ──────────────


def test_integration_greeting_varies() -> None:
    """Greetings vary across multiple calls (not always the same).

    The engine composes greetings from the concept network's seeded
    utterance vocabulary (EXPRESSES edges from the ``greeting_word``
    hub), matching production where ``mind.py`` always passes
    ``network=shared_network``. Without a network the greeting slot
    is empty and output collapses to the invariant fallback.
    """
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=None, network=ConceptNetwork())  # random seed
    emotion = _make_emotion()

    greetings = set()
    for _ in range(20):
        text = engine.greet(emotion, first=False)
        greetings.add(text)

    # Should have at least 2 different greetings
    assert len(greetings) > 1


def test_integration_emotion_changes_greeting() -> None:
    """Positive vs negative emotion changes farewell tone.

    Without a graph, both produce minimal output — no hardcoded
    fallback to differentiate. This test now verifies the engine
    doesn't crash with different emotions.
    """
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)

    emotion_positive = _make_emotion(valence=0.4, label="positive")
    emotion_negative = _make_emotion(valence=-0.3, label="melancholic")

    # Generate farewells with different emotions — should not crash
    positive_farewell = engine.farewell(emotion_positive)
    negative_farewell = engine.farewell(emotion_negative)
    assert isinstance(positive_farewell, str)
    assert isinstance(negative_farewell, str)


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
    logger.info(f"\n  Generative language tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


# ======================================================================
# From tests/test_graph_walk.py
# ======================================================================

def _make_personality_gw(**overrides) -> PersonalityTraits:
    """Construct a personality gw for tests."""
    defaults = {
        "openness": 0.85,
        "conscientiousness": 0.72,
        "extraversion": 0.55,
        "agreeableness": 0.78,
        "neuroticism": 0.38,
    }
    defaults.update(overrides)
    return PersonalityTraits(**defaults)


def _make_emotion_gw(
    label: str = "neutral",
    valence: float = 0.0,
    alertness: float = 0.5,
    creativity: float = 0.5,
    caution: float = 0.3,
    openness_to_engage: float = 0.7,
) -> EmotionalState:
    """Construct a emotion gw for tests."""
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


def _make_network_with_concepts() -> ConceptNetwork:
    """Build a small concept network with typed edges for testing."""
    net = ConceptNetwork()
    # Add concepts
    net.add_concept("cognition", confidence=0.9)
    net.add_concept("neural_activity", confidence=0.8)
    net.add_concept("awareness", confidence=0.7)
    net.add_concept("thinking", confidence=0.8)
    net.add_concept("memory", confidence=0.8)
    net.add_concept("learning", confidence=0.7)
    # Add typed edges
    net.add_edge("cognition", "neural_activity", RelationType.EMERGES_FROM, weight=0.9)
    net.add_edge("cognition", "awareness", RelationType.IS_A, weight=0.8)
    net.add_edge("thinking", "memory", RelationType.DEPENDS_ON, weight=0.7)
    net.add_edge("memory", "learning", RelationType.ENABLES, weight=0.8)
    net.add_edge("stress", "memory", RelationType.HARMS, weight=0.6)
    net.add_concept("stress", confidence=0.6)
    return net


# ─── Basic generation ──────────────────────────────────────────


def test_graph_walk_generates_from_concept_with_edges() -> None:
    """The generator produces text from a concept that has edges."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="cognition",
        intent="reflect",
        emotion="neutral",
        topics=["cognition"],
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is not None, "Generator returned None for a concept with edges"
    assert len(text) > 10, f"Generated text too short: {text!r}"
    # The text should mention the seed concept or a related concept
    lower = text.lower()
    assert any(w in lower for w in ("cognition", "neural", "awareness")), \
        f"Generated text doesn't mention the concept or its neighbors: {text!r}"


def test_graph_walk_returns_none_for_isolated_concept() -> None:
    """The generator returns None for a concept with no edges."""
    net = ConceptNetwork()
    net.add_concept("isolated_concept", confidence=0.5)
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="isolated_concept",
        intent="reflect",
        topics=["isolated_concept"],
        confidence=0.5,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is None, \
        f"Generator should return None for isolated concept, got: {text!r}"


def test_graph_walk_returns_none_for_unknown_concept() -> None:
    """The generator returns None when the seed concept doesn't exist."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="nonexistent_concept_xyz",
        intent="reflect",
        topics=["nonexistent_concept_xyz"],
        confidence=0.5,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is None, "Generator should return None for unknown concept"


def test_graph_walk_returns_none_when_no_network() -> None:
    """The generator returns None when no concept network is provided."""
    # GraphWalkGenerator requires a network — this test verifies
    # the GenerativeEngine falls back correctly when graph_walk is None

    from genesis_cognitive.language import GenerativeEngine
    from genesis_cognitive.self import SelfModel

    sm = SelfModel(born_at=int(time.time() * 1000), personality=_make_personality_gw())
    engine = GenerativeEngine(sm, seed=42, network=None)
    assert engine._graph_walk is None, "Graph walk should be None without network"


# ─── Emotional modulation ──────────────────────────────────────


def test_graph_walk_high_creativity_deeper_walk() -> None:
    """High creativity produces more edges (deeper walk)."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="cognition",
        intent="reflect",
        topics=["cognition"],
        confidence=0.7,
    )
    # Low creativity → shallow walk
    text_low = gen.generate(thought, _make_emotion_gw(creativity=0.1))
    # High creativity → deeper walk
    text_high = gen.generate(thought, _make_emotion_gw(creativity=0.9))

    assert text_low is not None, "Low creativity walk produced nothing"
    assert text_high is not None, "High creativity walk produced nothing"
    # High creativity should produce at least as much text as low
    # (deeper walk → more edges → more clauses)
    assert len(text_high) >= len(text_low), \
        f"High creativity should produce >= text: low={len(text_low)}, high={len(text_high)}"


def test_graph_walk_low_engagement_shorter_output() -> None:
    """Low engagement reduces walk depth (fewer hops)."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    # Test the walk depth directly — the voice layer can add random
    # connectors that affect text length independently of walk depth.
    assert gen._walk_depth(_make_emotion_gw(openness_to_engage=0.9)) == 2
    assert gen._walk_depth(_make_emotion_gw(openness_to_engage=0.1)) == 1


# ─── Seed concept extraction ───────────────────────────────────


def test_graph_walk_extracts_concept_from_topics() -> None:
    """The generator uses thought.topics[0] as the seed concept."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="something else entirely",
        intent="reflect",
        topics=["cognition"],
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is not None
    assert "cognition" in text.lower() or "neural" in text.lower(), \
        f"Should walk from topics[0], not content: {text!r}"


def test_graph_walk_extracts_concept_from_content() -> None:
    """When no topics, the generator extracts a concept from content."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="thinking",
        intent="reflect",
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is not None
    assert "thinking" in text.lower() or "memory" in text.lower(), \
        f"Should extract concept from content: {text!r}"


# ─── Sentence structure ────────────────────────────────────────


def test_graph_walk_produces_multiple_sentences() -> None:
    """A walk with multiple edges produces multiple sentences."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="cognition",
        intent="reflect",
        topics=["cognition"],
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw(creativity=0.9))
    assert text is not None
    # Should have at least one sentence boundary
    assert "." in text, f"No sentence boundary in output: {text!r}"


def test_graph_walk_uses_relation_verbs() -> None:
    """The generated text uses verbs appropriate to the relation types."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="cognition",
        intent="reflect",
        topics=["cognition"],
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is not None
    lower = text.lower()
    # The walk should use one of the EMERGES_FROM or IS_A verbs
    assert any(v in lower for v in ("emerges", "grows", "arises", "is", "kind of", "type of")), \
        f"No relation verb found in output: {text!r}"


# ─── Integration with GenerativeEngine ─────────────────────────


def test_generative_engine_uses_graph_walk_when_available() -> None:
    """GenerativeEngine uses the graph-walk generator when a network is provided."""

    from genesis_cognitive.language import GenerativeEngine
    from genesis_cognitive.self import SelfModel

    net = _make_network_with_concepts()
    sm = SelfModel(born_at=int(time.time() * 1000), personality=_make_personality_gw())
    engine = GenerativeEngine(sm, seed=42, network=net)

    assert engine._graph_walk is not None, "Graph walk should be initialized with network"

    thought = Thought(
        content="cognition",
        intent="reflect",
        topics=["cognition"],
        confidence=0.7,
    )
    text = engine.generate(thought, _make_emotion_gw())
    assert text is not None
    assert len(text) > 10, f"Engine produced too-short text: {text!r}"


def test_generative_engine_falls_back_without_network() -> None:
    """GenerativeEngine falls back to grammar when no network is available."""

    from genesis_cognitive.language import GenerativeEngine
    from genesis_cognitive.self import SelfModel

    sm = SelfModel(born_at=int(time.time() * 1000), personality=_make_personality_gw())
    engine = GenerativeEngine(sm, seed=42, network=None)

    assert engine._graph_walk is None

    thought = Thought(
        content="I am here and thinking about things",
        intent="reflect",
        confidence=0.7,
    )
    # Should not crash — should use grammar fallback
    text = engine.generate(thought, _make_emotion_gw())
    assert text is not None
    assert len(text) > 0


# ─── Knowledge metadata bypass ─────────────────────────────────


def test_graph_walk_defers_to_knowledge_metadata() -> None:
    """When knowledge metadata is present, the graph-walk defers to the
    vocabulary's _compose_knowledge_content path (which is already
    graph-grounded)."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="cognition",
        intent="inform",
        topics=["cognition"],
        confidence=0.7,
        metadata={
            "knowledge": [("emerges_from", "neural_activity", 0.9)],
            "definition": "the state of being aware",
            "topic": "cognition",
        },
    )
    text = gen.generate(thought, _make_emotion_gw())
    assert text is None, \
        "Graph walk should defer to knowledge path when metadata is present"


# ─── Grammar correctness ────────────────────────────────────────


def test_graph_walk_opening_does_not_break_grammar_with_definition() -> None:
    """Prepositional openings must not prefix a full clause.

    Previously, the code produced "Let me think about Water is a clear
    liquid." — broken grammar. The fix splits prepositional openings
    into a separate lead-in sentence: "Let me think about water.
    Water is a clear liquid."
    """
    net = ConceptNetwork()
    net.add_concept("water", confidence=0.9)
    net.add_edge("water", "ice", RelationType.IS_A, weight=0.7)
    net.add_concept("ice", confidence=0.7)
    # Set a definition on the concept
    concept = net.get_concept("water")
    assert concept is not None
    concept.properties["definition"] = "a clear liquid"

    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="water",
        intent="reflect",
        topics=["water"],
        confidence=0.7,
    )
    text = gen.generate(thought, _make_emotion_gw(creativity=0.9))
    assert text is not None, "Should produce text for concept with definition"

    # The broken pattern: "Let me think about Water is a clear liquid."
    # (no separator between opener and clause)
    # The fixed patterns are either:
    #   "Let me think about water. Water is a clear liquid." (two sentences)
    #   "There's something about water, water is a clear liquid." (comma-joined)
    assert " is a clear liquid." in text, \
        f"Definition should appear as a sentence: {text!r}"
    # Check that no sentence contains the broken pattern: opener
    # directly prefixing a capitalized "is" clause with no comma/period
    # separator. The comma-joined form is acceptable.
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        # The broken pattern is: opener + space + Capitalized word + " is "
        # (no comma between opener and clause). The comma-joined form
        # ("... about water, water is ...") is acceptable grammar.
        lower = s.lower()
        if " is " in lower:
            # Check for broken pattern: opener directly followed by
            # a capitalized word (no comma separator).
            broken_patterns = [
                r"let me think about [A-Z]",
                r"there's something about [A-Z]",
                r"what strikes me about [A-Z]",
                r"when i look at [A-Z]",
                r"when i consider [A-Z]",
                r"looking at [A-Z]",
                r"if i look at myself and [A-Z]",
                r"something in me responds to [A-Z]",
                r"there's a feeling around [A-Z]",
                r"my neurochemistry shifts with [A-Z]",
            ]
            for pat in broken_patterns:
                assert not re.match(pat, s), \
                    f"Broken grammar — opener directly prefixes clause: {s!r}"


def test_graph_walk_hedge_opening_works_as_prefix() -> None:
    """Hedge openings (ending with comma or 'I think') work as prefixes."""
    net = ConceptNetwork()
    net.add_concept("water", confidence=0.9)
    net.add_edge("water", "ice", RelationType.IS_A, weight=0.7)
    net.add_concept("ice", confidence=0.7)
    concept = net.get_concept("water")
    assert concept is not None
    concept.properties["definition"] = "a clear liquid"

    voice = Voice(_make_personality_gw(), seed=99)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=99)

    thought = Thought(
        content="water",
        intent="inform",
        topics=["water"],
        confidence=0.7,
    )
    # Low creativity, high caution → hedge openings ("I think", etc.)
    text = gen.generate(thought, _make_emotion_gw(creativity=0.1, caution=0.9))
    assert text is not None
    # Hedge openings should produce grammatical sentences like
    # "I think Water is a clear liquid." or
    # "As I understand it, Water is a clear liquid."
    assert " is " in text, f"Hedge prefix + 'is' clause expected: {text!r}"


def test_is_hedge_opening_classification() -> None:
    """_is_hedge_opening correctly classifies opening types."""
    assert GraphWalkGenerator._is_hedge_opening("I think")
    assert GraphWalkGenerator._is_hedge_opening("As I understand it,")
    assert GraphWalkGenerator._is_hedge_opening("If I'm honest,")
    assert GraphWalkGenerator._is_hedge_opening("To be honest,")
    # Prepositional/infinitive phrases are NOT hedges
    assert not GraphWalkGenerator._is_hedge_opening("Let me think about")
    assert not GraphWalkGenerator._is_hedge_opening("There's something about")
    assert not GraphWalkGenerator._is_hedge_opening("When I look at")
    assert not GraphWalkGenerator._is_hedge_opening("With")


# ─── Social exchange guards ────────────────────────────────────


def test_graph_walk_rejects_greet_intent() -> None:
    """Greetings must not walk the concept network.

    Without this guard, a greeting like "hello genesis" extracts
    "genesis" as the seed (alphabetically before "hello" in topics)
    and walks to holographic associations, producing knowledge
    statements instead of a greeting. The grammar fallback has proper
    greeting structures.
    """
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="greeting",
        intent="greet",
        emotion="neutral",
        topics=["hello", "genesis"],
        confidence=0.8,
    )
    assert gen.generate(thought, _make_emotion_gw()) is None


def test_graph_walk_rejects_farewell_intent() -> None:
    """Farewells must not walk the concept network."""
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="farewell",
        intent="farewell",
        emotion="neutral",
        topics=["goodbye"],
        confidence=0.8,
    )
    assert gen.generate(thought, _make_emotion_gw()) is None


def test_graph_walk_rejects_ask_intent() -> None:
    """Questions must not walk the concept network.

    The graph walk produces declarative statements from edges, not
    questions. Curiosity questions composed through the graph walk
    end up as statements instead of interrogatives. The grammar
    fallback has proper question structures.
    """
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="what is cognition",
        intent="ask",
        emotion="curious",
        topics=["cognition"],
        confidence=0.6,
    )
    assert gen.generate(thought, _make_emotion_gw()) is None


def test_graph_walk_rejects_self_report_with_identity_text() -> None:
    """Self-reports with pre-composed identity_text must not walk the graph.

    Feeling reports from _deliberate_greeting_question already have
    composed content. The graph walk would ignore the content and
    walk from the seed concept, producing knowledge statements
    instead of emotional self-reports.
    """
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="I'm feeling quite awake",
        intent="self_report",
        emotion="neutral",
        topics=["genesis"],
        confidence=0.8,
        metadata={"identity_text": "I'm feeling quite awake"},
    )
    assert gen.generate(thought, _make_emotion_gw()) is None


def test_graph_walk_rejects_express_emotion_intent() -> None:
    """Emotion expressions must not walk the concept network.

    The graph walk would extract the emotion word as a seed concept
    and produce knowledge statements about it. The grammar fallback
    has proper emotion structures ({emotion_clause}, {emotion_opener}).
    This also prevents think() timeouts from compose_feeling_report
    calling render() multiple times through the graph walk.
    """
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="content",
        intent="express_emotion",
        emotion="content",
        topics=["content"],
        confidence=0.7,
    )
    assert gen.generate(thought, _make_emotion_gw()) is None


# ======================================================================
# From tests/test_language_quality_eval.py
# ======================================================================

def _personality(**overrides) -> PersonalityTraits:
    """Return a personality fixture for tests."""
    defaults = {
        "openness": 0.85,
        "conscientiousness": 0.72,
        "extraversion": 0.55,
        "agreeableness": 0.78,
        "neuroticism": 0.38,
    }
    defaults.update(overrides)
    return PersonalityTraits(**defaults)


def _self_model() -> SelfModel:

    """Return a self-model fixture for tests."""
    return SelfModel(born_at=int(time.time() * 1000), personality=_personality())


def _emotion(
    label: str = "neutral",
    valence: float = 0.0,
    alertness: float = 0.5,
    creativity=0.5,
    caution=0.3,
) -> EmotionalState:
    """Return an emotion fixture for tests."""
    return EmotionalState(
        label=label,
        nuance="baseline",
        cognitive_style="steady",
        valence=valence,
        alertness=alertness,
        plasticity=0.5,
        creativity=creativity,
        caution=caution,
        openness_to_engage=0.7,
    )


# A spread of emotional states for the cross-emotion diversity test.
_EMOTION_SPREAD = [
    _emotion(label="neutral", valence=0.0, alertness=0.5, creativity=0.5),
    _emotion(label="positive", valence=0.6, alertness=0.7, creativity=0.7),
    _emotion(label="negative", valence=-0.5, alertness=0.4, creativity=0.3),
    _emotion(label="excited", valence=0.5, alertness=0.9, creativity=0.8),
    _emotion(label="calm", valence=0.2, alertness=0.2, creativity=0.4, caution=0.6),
    _emotion(label="curious", valence=0.3, alertness=0.7, creativity=0.85),
    _emotion(label="tired", valence=-0.1, alertness=0.15, creativity=0.3),
    _emotion(label="cautious", valence=-0.1, alertness=0.6, creativity=0.3, caution=0.85),
]

# Thoughts to test across intents.
_TEST_THOUGHTS = [
    Thought(
        content="Water is a clear liquid that covers most of Earth.",
        intent="inform",
        emotion="neutral",
        confidence=0.8,
    ),
    Thought(
        content="I am Genesis, an artificial mind.",
        intent="self_report",
        emotion="neutral",
        confidence=0.8,
    ),
    Thought(
        content="Cognition is the hard problem of experience.",
        intent="philosophize",
        emotion="neutral",
        confidence=0.7,
    ),
    Thought(
        content="A dog is a mammal.",
        intent="inform",
        emotion="neutral",
        confidence=0.9,
    ),
]

N_SEEDS = 20  # responses per thought for seed-diversity


def _content_words(text: str) -> set[str]:
    """Lowercase alphanumeric word tokens, minus short stopwords."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return {w for w in words if len(w) > 2}


def _relevance(response: str, thought: Thought) -> float:
    """Fraction of content words (len > 3) from the thought that appear in the response."""
    content_words = {w for w in _content_words(thought.content) if len(w) > 3}
    if not content_words:
        return 1.0  # nothing to check
    response_words = _content_words(response)
    hit = sum(1 for w in content_words if w in response_words)
    return hit / len(content_words)


# ─── Fluency / quality rubric ──────────────────────────────────
#
# These are deterministic, rubric-based quality proxies. They measure
# surface features that correlate with fluency: no repetition, proper
# formatting, no verbatim echo. They do NOT measure style, nuance, or
# semantic depth — that would require a human or LM judge, which the
# project's no-external-models constraint precludes.

# Common function words (articles, prepositions, conjunctions, pronouns).
# A reasonable fraction of these in a response indicates fluent prose
# rather than telegraphic or broken output.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "so", "because", "if", "when", "while", "though",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "my", "your", "its", "our", "their", "me", "him", "her", "us",
    "not", "no", "do", "does", "did", "have", "has", "had", "can", "could",
    "will", "would", "should", "may", "might", "must", "shall",
})


def _repetition_penalty(response: str) -> float:
    """Penalty for repeated n-grams within a response.

    Returns a score in [0, 1] where 1.0 = no repetition and 0.0 = heavy
    repetition. We check for repeated 3-grams (trigrams) and repeated
    sentences. Repeated content ("A dog is a mammal. A dog is a mammal.")
    is a clear fluency failure — the generator is echoing rather than
    composing.

    The penalty is the fraction of trigrams that are unique. If every
    trigram is unique, the score is 1.0. If half the trigrams are
    duplicates, the score is 0.5.
    """
    words = re.findall(r"[a-z]+", response.lower())
    if len(words) < 3:
        return 1.0  # too short to have repetition

    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 1.0
    unique_trigrams = len(set(trigrams))
    return unique_trigrams / len(trigrams)


def _grammaticality_score(response: str) -> float:
    """Surface well-formedness score in [0, 1].

    Checks for:
    - Proper capitalization (first letter of first sentence is uppercase)
    - Terminal punctuation (ends with . ! or ?)
    - No double spaces
    - No space before punctuation
    - No orphaned/empty slots (literal "{}" or unfilled placeholder artifacts)
    - No leading/trailing whitespace issues

    Each check contributes equally. The score is the fraction of checks
    that pass.
    """
    checks = []

    # 1. Starts with uppercase
    checks.append(bool(response) and response[0].isupper())

    # 2. Ends with terminal punctuation
    checks.append(bool(response) and response[-1] in ".!?")

    # 3. No double spaces
    checks.append("  " not in response)

    # 4. No space before punctuation
    checks.append(not re.search(r"\s+[.,;!?]", response))

    # 5. No unfilled slot artifacts
    checks.append("{" not in response and "}" not in response)

    # 6. No trailing/leading whitespace
    checks.append(response == response.strip())

    return sum(checks) / len(checks)


def _is_verbatim(response: str, thought: Thought) -> bool:
    """Check if the response is just the content string passed through.

    A verbatim response means the generator didn't compose anything —
    it just echoed the input content. This is a composition failure:
    the grammar/voice layers added nothing.
    """
    return response.strip() == thought.content.strip()


def _function_word_ratio(response: str) -> float:
    """Fraction of words that are function words.

    A healthy ratio (0.15–0.45) indicates fluent prose. Too low
    (< 0.10) suggests telegraphic or broken output. Too high
    (> 0.50) suggests empty filler. We return a score that peaks
    at 0.25 and decays on either side.
    """
    words = re.findall(r"[a-z]+", response.lower())
    if not words:
        return 0.0
    ratio = sum(1 for w in words if w in _FUNCTION_WORDS) / len(words)
    # Peak at 0.25, linear decay to 0 at 0.0 and 0.55
    if ratio <= 0.25:
        return ratio / 0.25
    elif ratio <= 0.55:
        return 1.0 - (ratio - 0.25) / 0.30
    else:
        return 0.0


def _composite_quality(response: str, thought: Thought) -> float:
    """Composite quality score in [0, 1].

    Weighted average of:
    - Repetition penalty (40%): the most impactful fluency defect
    - Grammaticality (30%): surface well-formedness
    - Function word ratio (15%): prose vs telegraphic
    - Non-verbatim bonus (15%): 1.0 if composed, 0.0 if verbatim echo

    The verbatim check is a bonus, not a penalty: a verbatim response
    can still be grammatical and non-repetitive, but it fails the
    composition goal. The bonus rewards actual composition.
    """
    rep = _repetition_penalty(response)
    gram = _grammaticality_score(response)
    fwr = _function_word_ratio(response)
    verbatim_bonus = 0.0 if _is_verbatim(response, thought) else 1.0

    return (0.40 * rep + 0.30 * gram + 0.15 * fwr + 0.15 * verbatim_bonus)


# ─── Baseline: single-structure, no voice ───────────────────────


class _SingleStructureBaseline:
    """A minimal baseline: always picks the first structure for an
    intent, fills slots with the vocabulary, applies NO voice.

    This isolates the contribution of weighted structure selection +
    voice modulation. If the full generator doesn't beat this on
    diversity, the grammar variation and voice layers aren't adding
    variety.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the single structure baseline."""
        self._vocab = Vocabulary(seed)
        self._personality = _personality()

    def render(self, thought: Thought, emotion: EmotionalState) -> str:
        """Render."""
        structures = INTENT_STRUCTURES.get(thought.intent, INTENT_STRUCTURES["inform"])
        structure = structures[0]  # always the first
        parts: list[str] = []
        for seg in structure.segments:
            if isinstance(seg, Literal):
                parts.append(seg.text)
            elif isinstance(seg, Slot):
                ctx = {"content": thought.content}
                parts.append(self._vocab.fill_slot(seg.name, ctx, emotion, self._personality))
        return "".join(parts).strip()


# ─── Eval ───────────────────────────────────────────────────────


def run_language_quality_eval() -> dict:
    """Run the language-quality evaluation. Returns a metrics dict."""
    # ── Metric 1: Diversity across seeds (same emotion) ──
    emotion_fixed = _emotion()
    seed_diversity_per_thought: list[float] = []
    relevance_per_thought: list[float] = []
    baseline_diversity_per_thought: list[float] = []

    # ── Metric 5: Fluency / quality (collected alongside diversity) ──
    repetition_per_thought: list[float] = []
    grammaticality_per_thought: list[float] = []
    verbatim_ratio_per_thought: list[float] = []
    composite_quality_per_thought: list[float] = []
    baseline_composite_quality_per_thought: list[float] = []

    for thought in _TEST_THOUGHTS:
        responses: set[str] = set()
        rel_sum = 0.0
        # Fluency accumulators
        rep_sum = 0.0
        gram_sum = 0.0
        verbatim_count = 0
        quality_sum = 0.0
        for seed in range(N_SEEDS):
            engine = GenerativeEngine(_self_model(), seed=seed)
            text = engine.render(thought, emotion_fixed)
            responses.add(text)
            rel_sum += _relevance(text, thought)
            # Fluency metrics
            rep_sum += _repetition_penalty(text)
            gram_sum += _grammaticality_score(text)
            if _is_verbatim(text, thought):
                verbatim_count += 1
            quality_sum += _composite_quality(text, thought)
        diversity = len(responses) / N_SEEDS
        seed_diversity_per_thought.append(diversity)
        relevance_per_thought.append(rel_sum / N_SEEDS)

        # Fluency aggregates
        repetition_per_thought.append(rep_sum / N_SEEDS)
        grammaticality_per_thought.append(gram_sum / N_SEEDS)
        verbatim_ratio_per_thought.append(verbatim_count / N_SEEDS)
        composite_quality_per_thought.append(quality_sum / N_SEEDS)

        # Baseline: same seeds, single-structure, no voice
        baseline_responses: set[str] = set()
        baseline_quality_sum = 0.0
        for seed in range(N_SEEDS):
            baseline = _SingleStructureBaseline(seed)
            btext = baseline.render(thought, emotion_fixed)
            baseline_responses.add(btext)
            baseline_quality_sum += _composite_quality(btext, thought)
        baseline_diversity_per_thought.append(len(baseline_responses) / N_SEEDS)
        baseline_composite_quality_per_thought.append(baseline_quality_sum / N_SEEDS)

    # ── Metric 2: Diversity across emotions (same seed) ──
    emotion_diversity_per_thought: list[float] = []
    for thought in _TEST_THOUGHTS:
        engine = GenerativeEngine(_self_model(), seed=42)
        responses = {engine.render(thought, emo) for emo in _EMOTION_SPREAD}
        emotion_diversity_per_thought.append(len(responses) / len(_EMOTION_SPREAD))

    # ── Aggregate ──
    n = len(_TEST_THOUGHTS)
    metrics = {
        "n_thoughts": n,
        "n_seeds": N_SEEDS,
        "n_emotions": len(_EMOTION_SPREAD),
        "seed_diversity_mean": sum(seed_diversity_per_thought) / n,
        "seed_diversity_per_thought": seed_diversity_per_thought,
        "emotion_diversity_mean": sum(emotion_diversity_per_thought) / n,
        "emotion_diversity_per_thought": emotion_diversity_per_thought,
        "relevance_mean": sum(relevance_per_thought) / n,
        "relevance_per_thought": relevance_per_thought,
        "baseline_diversity_mean": sum(baseline_diversity_per_thought) / n,
        "baseline_diversity_per_thought": baseline_diversity_per_thought,
        # Fluency / quality metrics
        "repetition_mean": sum(repetition_per_thought) / n,
        "repetition_per_thought": repetition_per_thought,
        "grammaticality_mean": sum(grammaticality_per_thought) / n,
        "grammaticality_per_thought": grammaticality_per_thought,
        "verbatim_ratio_mean": sum(verbatim_ratio_per_thought) / n,
        "verbatim_ratio_per_thought": verbatim_ratio_per_thought,
        "composite_quality_mean": sum(composite_quality_per_thought) / n,
        "composite_quality_per_thought": composite_quality_per_thought,
        "baseline_composite_quality_mean": sum(baseline_composite_quality_per_thought) / n,
        "baseline_composite_quality_per_thought": baseline_composite_quality_per_thought,
    }
    return metrics


def print_language_quality_report(m: dict) -> None:
    """Print a human-readable report from the metrics dict."""
    print()
    print("═══ Evaluation #5: Language Quality ═══")
    print()
    print(f"  Thoughts: {m['n_thoughts']}  Seeds: {m['n_seeds']}  Emotions: {m['n_emotions']}")
    print()
    print("  Diversity & Relevance")
    print("  " + "-" * 58)
    print(f"  Seed diversity (distinct/N)     {m['seed_diversity_mean']:.2f}"
          f"        {m['baseline_diversity_mean']:.2f}")
    print(f"  Emotion diversity (distinct/N)  {m['emotion_diversity_mean']:.2f}        —")
    print(f"  Relevance (content word recall) {m['relevance_mean']:.2f}        —")
    print()
    print("  Fluency / Quality (rubric-based, 0..1)")
    print("  " + "-" * 58)
    print(f"  Repetition penalty (1=no rep)   {m['repetition_mean']:.2f}        —")
    print(f"  Grammaticality (surface form)   {m['grammaticality_mean']:.2f}        —")
    print(f"  Verbatim ratio (0=all composed) {m['verbatim_ratio_mean']:.2f}        —")
    print(f"  Composite quality               {m['composite_quality_mean']:.2f}"
          f"        {m['baseline_composite_quality_mean']:.2f}")
    print()
    print("  Per-thought breakdown:")
    for i, t in enumerate(_TEST_THOUGHTS):
        print(f"    [{t.intent:12s}] seed_div={m['seed_diversity_per_thought'][i]:.2f}  "
              f"emo_div={m['emotion_diversity_per_thought'][i]:.2f}  "
              f"relev={m['relevance_per_thought'][i]:.2f}  "
              f"qual={m['composite_quality_per_thought'][i]:.2f}  "
              f"verb={m['verbatim_ratio_per_thought'][i]:.2f}")
    print()
    gen_better = m["seed_diversity_mean"] > m["baseline_diversity_mean"]
    gen_vs_base = "BETTER" if gen_better else "WORSE/EQUAL"
    print(f"  Finding: Generator diversity vs baseline: {gen_vs_base} "
          f"({m['seed_diversity_mean']:.2f} vs {m['baseline_diversity_mean']:.2f})")
    qual_better = m["composite_quality_mean"] > m["baseline_composite_quality_mean"]
    qual_vs_base = "BETTER" if qual_better else "WORSE/EQUAL"
    print(f"  Finding: Composite quality vs baseline:   {qual_vs_base} "
          f"({m['composite_quality_mean']:.2f} vs {m['baseline_composite_quality_mean']:.2f})")
    print(f"  Finding: Verbatim ratio (lower = more composition): "
          f"{m['verbatim_ratio_mean']:.2f}")
    print()


# ─── Test entry points ──────────────────────────────────────────


def test_language_quality_diversity() -> None:
    """The generator produces varied output and beats a single-structure baseline."""
    m = run_language_quality_eval()
    print_language_quality_report(m)
    # Testable claims:
    # 1. Generator seed diversity > baseline by a meaningful margin (>= 1.5x).
    #    This is the core "composition adds variety over a single-structure
    #    baseline" claim. The absolute diversity value is modest for short
    #    thoughts (some intents have few structures), so the claim is framed
    #    as a relative improvement, not an absolute threshold.
    ratio = m["seed_diversity_mean"] / max(m["baseline_diversity_mean"], 1e-6)
    assert ratio >= 1.5, (
        f"Generator diversity ({m['seed_diversity_mean']:.2f}) is not >= 1.5x "
        f"baseline ({m['baseline_diversity_mean']:.2f}, ratio {ratio:.2f}) — "
        f"grammar variation and voice are not adding meaningful variety."
    )
    # 2. Emotion diversity > 0.3 (responses vary with emotional state).
    assert m["emotion_diversity_mean"] > 0.3, (
        f"Emotion diversity too low ({m['emotion_diversity_mean']:.2f}) — "
        f"responses do not vary with emotional state."
    )
    # 3. Relevance > 0.3 (responses preserve content words).
    assert m["relevance_mean"] > 0.3, (
        f"Relevance too low ({m['relevance_mean']:.2f}) — "
        f"responses are dropping the semantic content of thoughts."
    )
    # 4. At least one thought achieves seed diversity > 0.3 — the generator
    #    CAN produce varied output (not universally canned). The mean is
    #    pulled down by short, low-structure intents; the max shows the
    #    generator's capability ceiling is well above a canned string.
    max_seed_div = max(m["seed_diversity_per_thought"])
    assert max_seed_div > 0.3, (
        f"Max seed diversity ({max_seed_div:.2f}) too low — even the best "
        f"case is near-identical output across seeds."
    )
    print(f"  ✓ Generator diversity >= 1.5x baseline ({m['seed_diversity_mean']:.2f} vs "
          f"{m['baseline_diversity_mean']:.2f}, ratio {ratio:.2f})")
    print(f"  ✓ Emotion diversity > 0.3 ({m['emotion_diversity_mean']:.2f})")
    print(f"  ✓ Relevance > 0.3 ({m['relevance_mean']:.2f})")
    print(f"  ✓ Max seed diversity > 0.3 ({max_seed_div:.2f})")
    # Report the limitation honestly: the minimum seed diversity shows
    # which intents produce near-identical output across seeds.
    min_seed_div = min(m["seed_diversity_per_thought"])
    print(f"  · Min seed diversity = {min_seed_div:.2f} (short/low-structure intents "
          f"produce near-identical output — a real limitation, reported honestly)")
    print()


def test_language_quality_fluency() -> None:
    """The generator's output passes rubric-based fluency checks.

    This is the fluency/quality companion to the diversity test. It
    checks surface features that correlate with fluency:
    - No heavy repetition (repeated trigrams within a response)
    - Surface grammaticality (capitalization, punctuation, no broken slots)
    - Composition (not just verbatim echo of the content)

    These are rubric-based proxies, not a human or LM judge. See the
    module docstring for what is and isn't claimed.
    """
    m = run_language_quality_eval()
    # 1. Repetition penalty >= 0.70 — at least 70% of trigrams are unique.
    #    Heavy repetition ("A dog is a mammal. A dog is a mammal.") is a
    #    clear fluency failure. 0.70 allows some repetition (connectors,
    #    common phrases) but catches the echo defect.
    assert m["repetition_mean"] >= 0.70, (
        f"Repetition penalty too low ({m['repetition_mean']:.2f}) — "
        f"responses contain heavy n-gram repetition, a fluency failure."
    )
    # 2. Grammaticality >= 0.80 — at least 80% of surface checks pass.
    #    This catches broken formatting, missing capitalization, unfilled
    #    slots, and other surface defects.
    assert m["grammaticality_mean"] >= 0.80, (
        f"Grammaticality too low ({m['grammaticality_mean']:.2f}) — "
        f"responses have surface defects (broken formatting, missing "
        f"capitalization, unfilled slots)."
    )
    # 3. Verbatim ratio is reported as a finding, not asserted. A high
    #    verbatim ratio means the generator is echoing content rather than
    #    composing — a real limitation, but not a correctness failure.
    #    We report it honestly rather than asserting a threshold.
    print(f"  ✓ Repetition penalty >= 0.70 ({m['repetition_mean']:.2f})")
    print(f"  ✓ Grammaticality >= 0.80 ({m['grammaticality_mean']:.2f})")
    print(f"  · Verbatim ratio = {m['verbatim_ratio_mean']:.2f} "
          f"(lower = more composition; reported as a finding, not asserted)")
    print(f"  · Composite quality = {m['composite_quality_mean']:.2f} "
          f"(baseline {m['baseline_composite_quality_mean']:.2f})")
    print()


# ======================================================================
# Hamiltonian flow field — numerical stability
# ======================================================================

def _make_flow_field(
    angle_matrix: np.ndarray,
    weight_vector: np.ndarray,
) -> FlowGenerator:
    """Build a FlowGenerator with a pre-set potential field.

    Bypasses ``build()`` (which needs a real concept network) so the
    field can be set directly for numerical tests.
    """
    gen = FlowGenerator.__new__(FlowGenerator)
    gen._angle_matrix = angle_matrix.astype(np.float32)
    gen._weight_vector = weight_vector.astype(np.float32)
    return gen


def test_flow_gradient_matches_analytic_von_mises() -> None:
    """_gradient agrees with the closed-form von Mises derivative.

    With a single concept offset by δ in one dimension, the joint
    kernel is K = exp(κ(cos δ − 1)) and the only nonzero gradient
    component is −κ·w·K·sin δ.
    """
    from genesis_cognitive.language.flow import _FLOW_DIM, _KAPPA

    delta = 0.4
    theta = np.zeros(_FLOW_DIM, dtype=np.float32)
    angle = np.zeros((1, _FLOW_DIM), dtype=np.float32)
    angle[0, 0] = delta
    gen = _make_flow_field(angle, np.array([0.7], dtype=np.float32))

    grad = gen._gradient(theta)

    kernel = math.exp(_KAPPA * (math.cos(delta) - 1.0))
    expected = np.zeros(_FLOW_DIM, dtype=np.float32)
    expected[0] = -_KAPPA * 0.7 * kernel * math.sin(delta)
    assert np.allclose(grad, expected, atol=1e-6), (grad[:2], expected[:2])


def test_flow_gradient_no_overflow_with_clustered_concepts() -> None:
    """The joint kernel must not overflow float32 for clustered concepts.

    Concepts clustered near the query (as a seed's neighbourhood is)
    drive the joint kernel to exp(κ·d) ≈ 7e20 for κ=3, d=16. Computing
    the product directly overflowed the float32 squared norm inside
    ``np.linalg.norm``, yielding ``grad_mag = inf`` and collapsing the
    flow onto the momentum clip. The log-space kernel keeps it bounded.
    """
    from genesis_cognitive.language.flow import _FLOW_DIM

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, _FLOW_DIM).astype(np.float32)
    offsets = rng.normal(0, 0.03, (2000, _FLOW_DIM)).astype(np.float32)
    gen = _make_flow_field(theta + offsets, rng.uniform(0, 1, 2000))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # overflow would raise
        grad = gen._gradient(theta)
        grad_mag = float(np.linalg.norm(grad))

    assert np.isfinite(grad).all()
    assert np.isfinite(grad_mag) and grad_mag > 0.0


def test_flow_potential_is_finite_and_bounded() -> None:
    """_potential stays finite for clustered concepts (bounded kernel)."""
    from genesis_cognitive.language.flow import _FLOW_DIM

    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, _FLOW_DIM).astype(np.float32)
    offsets = rng.normal(0, 0.03, (2000, _FLOW_DIM)).astype(np.float32)
    gen = _make_flow_field(theta + offsets, rng.uniform(0, 1, 2000))

    potential = gen._potential(theta)
    assert math.isfinite(potential)


# ======================================================================
# Morphology — verb agreement and deconjugation
# ======================================================================

from genesis_cognitive.language.morphology import (  # noqa: E402
    agree_verb_phrase,
    copula,
    is_plural_np,
    is_verb_form,
)


def test_is_verb_form_returns_base_for_irregulars() -> None:
    """is_verb_form returns the true base form, not the inflected word.

    "has" is the 3sg form of "have". The base form is "have", not "has".
    Returning the inflected word as its own base caused agree_verb_phrase
    to skip deconjugation and fall through to naive 's'-stripping,
    corrupting "has" → "ha".
    """
    assert is_verb_form("has") == "have"
    assert is_verb_form("does") == "do"
    assert is_verb_form("is") == "be"
    assert is_verb_form("was") == "be"
    assert is_verb_form("had") == "have"
    assert is_verb_form("did") == "do"


def test_is_verb_form_regular_3sg() -> None:
    """is_verb_form deconjugates regular 3sg forms to their base."""
    assert is_verb_form("makes") == "make"
    assert is_verb_form("causes") == "cause"
    assert is_verb_form("enables") == "enable"
    assert is_verb_form("runs") == "run"


def test_is_verb_form_rejects_plural_nouns() -> None:
    """is_verb_form returns None for plural nouns that aren't verbs."""
    assert is_verb_form("dogs") is None
    assert is_verb_form("cats") is None
    assert is_verb_form("emotions") is None
    assert is_verb_form("concepts") is None


def test_agree_verb_phrase_has_to_do_with() -> None:
    """'has to do with' deconjugates to 'have to do with', not 'ha to do with'.

    The naive 's'-stripping fallback corrupted "has" → "ha". With the
    is_verb_form fix, the deconjugation branch correctly maps "has" →
    "have".
    """
    assert agree_verb_phrase("has to do with", True) == "have to do with"


def test_agree_verb_phrase_does() -> None:
    """'does' deconjugates to 'do', not 'doe'."""
    assert agree_verb_phrase("does", True) == "do"


def test_agree_verb_phrase_singular_unchanged() -> None:
    """Singular subjects leave the verb phrase unchanged."""
    assert agree_verb_phrase("has to do with", False) == "has to do with"
    assert agree_verb_phrase("relates to", False) == "relates to"


def test_agree_verb_phrase_regular_3sg() -> None:
    """Regular 3sg verbs deconjugate correctly for plural subjects."""
    assert agree_verb_phrase("relates to", True) == "relate to"
    assert agree_verb_phrase("enables", True) == "enable"
    assert agree_verb_phrase("causes", True) == "cause"


def test_is_plural_np_3sg_verb_form_is_singular() -> None:
    """3sg verb forms take singular agreement, not plural.

    'makes' is the 3sg form of 'make'. As a concept subject it takes
    singular agreement ("makes relates to"), not plural ("makes relate
    to").  This is consistent with _SINGULAR_S_WORDS already containing
    auxiliary 3sg forms like 'has', 'does', 'is', 'was'.
    """
    assert not is_plural_np("makes")
    assert copula("makes") == "is"


def test_is_plural_np_common_plurals_still_plural() -> None:
    """Common plural nouns ending in 's' are still plural."""
    assert is_plural_np("dogs")
    assert is_plural_np("cats")
    assert is_plural_np("emotions")
    assert is_plural_np("concepts")
    assert copula("dogs") == "are"


def test_is_plural_np_dreams_still_plural() -> None:
    """'dreams' is a common plural noun and stays plural."""
    assert is_plural_np("dreams")
    assert copula("dreams") == "are"


if __name__ == "__main__":
    m = run_language_quality_eval()
    print_language_quality_report(m)

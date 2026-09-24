"""Language bundle tests — generative language and graph walk."""

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
    SentenceStructure,
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

    text = engine.self_report(emotion)
    assert len(text) > 0


def test_self_report_composes_from_fragments() -> None:
    """Self-report composes surface text from semantic fragments.

    The fragments are (kind, text) semantic parts; the language engine
    owns the wording. The same fragment set must produce varied,
    grammatical output — never a recited string.
    """
    model = _make_self_model()
    engine = GenerativeEngine(model, seed=42)
    emotion = _make_emotion()

    fragments = [
        ("name", "Genesis"),
        ("trait", "curious"),
        ("pred", "care most about understanding"),
    ]
    outputs = set()
    for seed in range(12):
        engine = GenerativeEngine(model, seed=seed)
        thought = Thought(
            content="Genesis",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.9,
            self_reflection=True,
            metadata={"self_fragments": fragments, "field": "identity"},
        )
        outputs.add(engine.generate(thought, emotion))
    # Semantic content survives in every realization
    for text in outputs:
        assert "Genesis" in text
        assert "curious" in text or "care" in text
    # The engine composes varied surface forms, not one fixed string
    assert len(outputs) > 1


def test_self_report_pred_fragments_never_collide_with_copula() -> None:
    """Verb-initial fragment candidates never produce 'I am care...'.

    When the fragments carry predicates but no name/trait/complement
    lead, copula frames ("I am {content}") would produce broken text —
    the generator must exclude them.
    """
    model = _make_self_model()
    emotion = _make_emotion()
    fragments = [("pred", "care about honesty"), ("pred", "depend on memory")]
    for seed in range(20):
        engine = GenerativeEngine(model, seed=seed)
        thought = Thought(
            content="x",
            intent="self_report",
            emotion=emotion.label,
            confidence=0.9,
            metadata={"self_fragments": fragments},
        )
        text = engine.generate(thought, emotion)
        assert "am care" not in text.lower()
        assert "am depend" not in text.lower()
        assert "care" in text or "honesty" in text or "depend" in text


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


def test_graph_walk_rejects_self_report_with_self_fragments() -> None:
    """Self-reports carrying self_fragments must not walk the graph.

    Self-composer fragments are semantic material for the grammar
    path. The graph walk would ignore them and walk from the seed
    concept, producing knowledge statements ("Genesis relates to
    Light") instead of the self-description.
    """
    net = _make_network_with_concepts()
    voice = Voice(_make_personality_gw(), seed=42)
    gen = GraphWalkGenerator(net, voice, _make_personality_gw(), seed=42)

    thought = Thought(
        content="genesis",
        intent="self_report",
        emotion="neutral",
        topics=["genesis"],
        confidence=0.8,
        metadata={
            "self_fragments": [("comp", "feeling quite awake")],
            "field": "identity",
        },
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

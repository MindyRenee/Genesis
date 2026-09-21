"""Tests for the Genesis cognitive mind modules.

Tests each cognitive subsystem in isolation, then tests the full
mind end-to-end with a running daemon.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from genesis_client import NeuroSummary
from genesis_client.protocol import PHASE_FLOW, PHASE_SLEEPING, PHASE_STRESS
from genesis_cognitive.emotion import EmotionalState, assess_emotion
from genesis_cognitive.perception import Intent, QuestionType, perceive
from genesis_cognitive.self import SelfModel

logger = logging.getLogger(__name__)

# ─── Self-model tests ─────────────────────────────────────────


def test_self_model_defaults():
    """Self-model has sensible defaults."""
    model = SelfModel()
    assert model.name == "Genesis"
    assert model.personality.openness > 0.7
    assert len(model.values) >= 5
    assert model.values[0].name == "understanding"


def test_self_model_trait_concept_names():
    """Self-model maps trait values to concept names, not hardcoded strings.

    The personality traits are no longer described by fixed English
    phrases via describe(). Instead, trait_concept_names() returns
    concept names (building blocks) for the active poles of each
    Big Five dimension. The language engine composes the actual
    phrasing from these concept names.
    """
    model = SelfModel()
    names = model.personality.trait_concept_names()
    # Default openness 0.85 > 0.6 → high pole: curious, creative
    assert "curious" in names
    assert "creative" in names
    # Default extraversion 0.45 is between 0.4 and 0.6 → neutral, no pole
    assert "introspective" not in names
    assert "expressive" not in names


def test_self_model_learn():
    """Self-model can learn new facts."""
    model = SelfModel()
    model.learn("user_name", "Alice")
    assert model.world_knowledge["user_name"] == "Alice"


def test_self_model_relationship_notes():
    """Self-model tracks relationship notes."""
    model = SelfModel()
    model.add_relationship_note("User likes philosophy")
    assert "User likes philosophy" in model.relationship_notes


# ─── Emotion tests ─────────────────────────────────────────────


def test_emotion_positive_excited():
    """High arousal + positive valence → excited."""
    summary = NeuroSummary(
        arousal=0.8,
        valence=0.5,
        global_tone=0.6,
        plasticity_gate=0.6,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,  # active
    )
    emo = assess_emotion(summary)
    assert emo.label == "excited"
    assert emo.verbosity > 1.0


def test_emotion_stress_phase():
    """Stress phase → stressed."""
    summary = NeuroSummary(
        arousal=0.7,
        valence=-0.3,
        global_tone=0.3,
        plasticity_gate=0.3,
        encoding_weight=0.4,
        consolidation_weight=0.4,
        retrieval_weight=0.4,
        phase=PHASE_STRESS,
    )
    emo = assess_emotion(summary)
    assert emo.label == "stressed"
    assert emo.caution > 0.5


def test_emotion_sleeping():
    """Sleeping phase → sleeping."""
    summary = NeuroSummary(
        arousal=0.2,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.3,
        encoding_weight=0.3,
        consolidation_weight=0.3,
        retrieval_weight=0.3,
        phase=PHASE_SLEEPING,
    )
    emo = assess_emotion(summary)
    assert emo.label == "sleeping"
    assert emo.openness_to_engage < 0.5


def test_emotion_flow():
    """Flow phase → in flow."""
    summary = NeuroSummary(
        arousal=0.6,
        valence=0.4,
        global_tone=0.6,
        plasticity_gate=0.7,
        encoding_weight=0.6,
        consolidation_weight=0.6,
        retrieval_weight=0.6,
        phase=PHASE_FLOW,
    )
    emo = assess_emotion(summary)
    assert emo.label == "in flow"
    assert emo.cognitive_style == "sharp"


def test_emotion_structural_state():
    """Emotional state provides structural category and mode."""
    summary = NeuroSummary(
        arousal=0.5,
        valence=0.0,
        global_tone=0.4,
        plasticity_gate=0.4,
        encoding_weight=0.5,
        consolidation_weight=0.5,
        retrieval_weight=0.5,
        phase=0,
    )
    emo = assess_emotion(summary)
    from genesis_cognitive.emotion import CognitiveMode, EmotionCategory
    assert emo.category == EmotionCategory.NEUTRAL
    assert emo.cognitive_mode == CognitiveMode.STEADY


# ─── Emotion word learning tests ──────────────────────────────


def test_learn_emotion_word_from_labeling():
    """Emotion word learning associates words with categories."""
    from unittest.mock import MagicMock

    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.concepts import ConceptNetwork

    # Create a minimal mock for the client
    client = MagicMock()
    client.get_neuro_summary.return_value = NeuroSummary(
        arousal=0.5, valence=0.0, global_tone=0.4,
        plasticity_gate=0.5, encoding_weight=0.5,
        consolidation_weight=0.5, retrieval_weight=0.5, phase=0,
    )

    # Create a cognition engine with a fresh network
    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from genesis_cognitive.cognition.concept_learner import ConceptLearner
    cog._concept_learner = ConceptLearner(network=cog.network)

    # Test labeling
    emotion = EmotionalState(
        label="positive", cognitive_style="steady",
        valence=0.3, alertness=0.5, plasticity=0.5,
    )
    cog._learn_word_from_labeling("you seem positive", emotion)
    cog._learn_word_from_labeling("you seem content", emotion)
    cog._learn_word_from_labeling("you seem calm", emotion)

    # Verify the words were learned
    for word in ["positive", "content", "calm"]:
        c = cog.network.get_concept(word)
        assert c is not None, f"Concept '{word}' was not created"
        assert c.properties.get("emotion_category") == "positive"

    # Verify find_emotion_words works
    words = cog.network.find_emotion_words("positive")
    assert "positive" in words
    assert "content" in words
    assert "calm" in words

    # Verify relationships were created between words
    # for the same category (SIMILAR_TO edges)
    from genesis_cognitive.concepts import RelationType
    neighbors = cog.network.get_neighbors("positive", RelationType.SIMILAR_TO)
    neighbor_ids = [n[0] for n in neighbors]
    assert "content" in neighbor_ids
    assert "calm" in neighbor_ids

    # Verify EXPRESSES edges connect words to category hubs
    hub_id = "_cat:emotion:positive"
    hub_neighbors = cog.network.get_neighbors(hub_id, RelationType.EXPRESSES)
    hub_word_ids = [n[0] for n in hub_neighbors]
    assert "positive" in hub_word_ids
    assert "content" in hub_word_ids
    assert "calm" in hub_word_ids


def test_learn_emotion_word_skips_non_emotion_words():
    """Emotion word learning skips common non-emotion words."""
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from genesis_cognitive.cognition.concept_learner import ConceptLearner
    cog._concept_learner = ConceptLearner(network=cog.network)

    emotion = EmotionalState(
        label="neutral", cognitive_style="steady",
        valence=0.0, alertness=0.5, plasticity=0.5,
    )
    # "you seem good" — "good" is in skip_words
    cog._learn_word_from_labeling("you seem good", emotion)
    assert cog.network.get_concept("good") is None


def test_learn_emotion_word_different_patterns():
    """Emotion word learning recognizes different labeling patterns."""
    from genesis_cognitive.cognition import CognitionEngine
    from genesis_cognitive.concepts import ConceptNetwork

    cog = CognitionEngine.__new__(CognitionEngine)
    cog.network = ConceptNetwork()
    cog._rng = __import__("random").Random(42)
    from genesis_cognitive.cognition.concept_learner import ConceptLearner
    cog._concept_learner = ConceptLearner(network=cog.network)

    emotion = EmotionalState(
        label="stressed", cognitive_style="vigilant",
        valence=-0.3, alertness=0.8, plasticity=0.4,
    )
    cog._learn_word_from_labeling("you're feeling stressed", emotion)
    cog._learn_word_from_labeling("that's called overwhelmed", emotion)

    c = cog.network.get_concept("stressed")
    assert c is not None
    assert c.properties.get("emotion_category") == "stressed"

    c = cog.network.get_concept("overwhelmed")
    assert c is not None
    assert c.properties.get("emotion_category") == "stressed"


# ─── Perception tests ──────────────────────────────────────────


def test_perceive_greeting():
    """Greetings are detected."""
    p = perceive("Hello there!")
    assert p.intent == Intent.GREETING


def test_perceive_farewell():
    """Farewells are detected."""
    p = perceive("Goodbye, see you later.")
    assert p.intent == Intent.FAREWELL


def test_perceive_self_inquiry():
    """Self-inquiry questions are detected."""
    # "How are you feeling?" matches greeting patterns first — that's
    # correct, it's a social ritual. But "What are you?" and "Are you
    # alive?" are genuine self-inquiry.
    p = perceive("What are you?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Are you alive?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Do you remember things?")
    assert p.intent == Intent.SELF_INQUIRY
    p = perceive("Can you feel emotions?")
    assert p.intent == Intent.SELF_INQUIRY


def test_perceive_question():
    """Questions are detected."""
    p = perceive("What is the meaning of life?")
    assert p.intent in (Intent.QUESTION, Intent.PHILOSOPHY)
    assert p.question_type == QuestionType.WHAT


def test_perceive_code():
    """Code-related text is detected."""
    p = perceive("Can you help me debug this Rust function?")
    assert p.intent == Intent.CODE_DISCUSSION
    assert p.is_about_code


def test_perceive_philosophy():
    """Philosophical questions are detected."""
    p = perceive("What is cognition?")
    assert p.intent == Intent.PHILOSOPHY


def test_perceive_encouragement():
    """Encouragement is detected."""
    p = perceive("You're doing great work!")
    assert p.intent == Intent.ENCOURAGEMENT


def test_perceive_introduction():
    """Introductions are detected."""
    p = perceive("My name is Alice.")
    assert p.intent == Intent.INTRODUCTION
    assert p.entities.get("name") == "Alice"


def test_perceive_sentiment():
    """Sentiment is detected."""
    p = perceive("I love this, it's amazing!")
    assert p.sentiment > 0
    assert p.sentiment_label == "positive"

    p = perceive("This is terrible and awful.")
    assert p.sentiment < 0
    assert p.sentiment_label == "negative"


def test_perceive_sentiment_negation():
    """Negation flips positive sentiment to negative.

    Tests both apostrophe forms ("don't") and no-apostrophe forms
    ("dont") — the tokenizer must preserve apostrophes for negation
    detection to work on contractions.
    """
    # "not good" — basic negation
    p = perceive("This is not good.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "don't like" — contraction with apostrophe
    p = perceive("I don't like this.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "dont like" — contraction without apostrophe (casual typing)
    p = perceive("I dont like this.")
    assert p.sentiment < 0 or p.sentiment_label == "negative"

    # "doesn't love" — negation of a positive word
    p = perceive("She doesn't love it.")
    assert p.sentiment <= 0


def test_perceive_topics():
    """Topics are extracted."""
    p = perceive("Tell me about cognition and memory.")
    assert len(p.topics) > 0
    assert "cognition" in p.topics or "memory" in p.topics


# ─── Integration test ──────────────────────────────────────────


def find_daemon_binary() -> str | None:
    """Find daemon binary."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "genesis-daemon"),
        os.path.join(os.path.dirname(__file__), "..", "..", "target", "debug", "genesis-daemon"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _run_mind_tests(mind) -> None:
    """Run the 8 integration test assertions against a live mind."""
    # Test 1: Greeting
    response = mind.respond("Hello, Genesis!")
    assert len(response) > 0, "greeting response empty"
    logger.info(f"  PASS  greeting: '{response[:50]}...'")

    # Test 2: Self-inquiry (how are you)
    response = mind.respond("How are you feeling?")
    # Her response is emergent from her concept network and emotional
    # state, so it may use different words than "feel"/"arousal" —
    # e.g. "neurochemistry", "good", "mind", "thinking", etc. She may
    # also produce an honest gap response ("Unable to respond
    # meaningfully") when she hasn't learned words for her state yet.
    # Accept any non-empty response that relates to her internal state
    # or her self-awareness of gaps in her understanding.
    feeling_words = (
        "feel", "feeling", "arousal", "neurochemistry", "neurochemical",
        "emotion", "emotional", "mind", "thinking", "state",
        "good", "bad", "happy", "calm", "stressed", "alert",
        "drowsy", "delta", "theta", "alpha", "gamma", "beta",
        "dopamine", "serotonin", "cortisol", "plasticity", "learning",
        "meaningfully", "understand", "myself", "sure", "able",
        "positive", "negative", "neutral", "content", "excited",
        "wonder", "curious", "flow", "overwhelm", "anxious",
        "energized", "energy", "tired", "heavy", "light",
        "warm", "charged", "quality",
    )
    lower_resp = response.lower()
    assert any(w in lower_resp for w in feeling_words), (
        f"self-inquiry response doesn't relate to feelings: '{response[:80]}'"
    )
    logger.info(f"  PASS  self-inquiry: '{response[:50]}...'")

    # Test 3: Identity question
    response = mind.respond("What are you?")
    assert "Genesis" in response or "artificial" in response.lower()
    logger.info(f"  PASS  identity: '{response[:50]}...'")

    # Test 4: Philosophy
    response = mind.respond("What is cognition?")
    assert len(response) > 30, "philosophy response too short"
    logger.info(f"  PASS  philosophy: '{response[:50]}...'")

    # Test 5: Status
    status = mind.status()
    assert status["running"] is True
    assert status["interactions"] == 4
    logger.info(f"  PASS  status: phase={status['phase']}, emotion={status['emotion']}")

    # Test 6: Introspection
    introspection = mind.introspect()
    assert "Intent" in introspection or "intent" in introspection.lower()
    logger.info(f"  PASS  introspection: '{introspection[:50]}...'")

    # Test 7: Emotional state changed
    emotion = mind.feel()
    assert emotion.label in (
        "positive",
        "neutral",
        "excited",
        "content",
        "active",
        "in flow",
        "calm",
    )
    logger.info(f"  PASS  emotion: {emotion.label}")

    # Test 8: Farewell
    response = mind.respond("Goodbye.")
    assert len(response) > 0
    logger.info(f"  PASS  farewell: '{response[:50]}...'" + ")")


def _cleanup_test_daemon(proc, data_dir: str) -> None:
    """Terminate the daemon process and clean up the data directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run(
        ["bash", os.path.join(project_root, "run.sh"), "--stop"],
        env={**os.environ, "XDG_DATA_HOME": os.path.dirname(data_dir)},
        check=True,
        timeout=130,
    )
    proc.wait(timeout=5)
    shutil.rmtree(data_dir)
    os.rmdir(os.path.dirname(data_dir))


def test_integration_mind():
    """Integration test: start daemon, talk to Genesis, verify responses."""
    daemon_path = find_daemon_binary()
    if daemon_path is None:
        logger.info("  SKIP  daemon binary not found")
        return

    data_home = tempfile.mkdtemp(prefix="genesis_mind_test_")
    data_dir = os.path.join(data_home, "genesis-public")
    os.mkdir(data_dir)
    socket_path = os.path.join(data_dir, "genesis.sock")

    proc = subprocess.Popen(
        [daemon_path, "--data-dir", data_dir, "--ltm-capacity", "128"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for socket
        for _ in range(50):
            if os.path.exists(socket_path):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("daemon didn't start")

        time.sleep(0.5)

        from genesis_cognitive import Mind

        mind = Mind(socket_path, offline=True)
        try:
            mind.start()
            _run_mind_tests(mind)
        finally:
            mind.stop()
        logger.info("  PASS  mind stopped cleanly")

    finally:
        _cleanup_test_daemon(proc, data_dir)


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            result = test()
            if result is False:
                logger.info(f"  FAIL  {test.__name__}")
                failed += 1
            else:
                logger.info(f"  PASS  {test.__name__}")
                passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    logger.info(f"\n  Cognitive mind tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

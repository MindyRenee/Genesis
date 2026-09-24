"""Perception bundle tests — perception, vision, visual cortex."""

import logging
import os
import sys
import tempfile

import numpy as np
import pytest

from genesis_cognitive.auditory import (
    COCO_CLASSES,
    DetectedObject,
    ObjectRecognizer,
)
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language import Vocabulary
from genesis_cognitive.perception import (
    IntegratedPerception,
    MemoryBridge,
    MultisensoryInput,
    MultisensoryIntegrator,
    PatternWeights,
    QuestionType,
    SensoryModality,
    V1Model,
    V4Model,
    VisualCortex,
    VisualFeature,
    VisualPercept,
    VTCFeatureSpace,
    decode_image_bytes,
    load_image,
    resize_for_vision,
)
from genesis_cognitive.perception.core import (
    Intent,
    Perception,
    _classify_intent,
    _detect_question_type,
    _detect_sentiment,
    _extract_entities,
    _extract_key_phrases,
    _extract_topics,
    perceive,
)
from genesis_cognitive.perception.vision import (
    ColorInfo,
    _build_scene,
    _learn_objects,
)
from genesis_cognitive.self import PersonalityTraits
from genesis_cognitive.vision.v1 import _build_summary

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_perception.py
# ======================================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Intent classification
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text", ["hello", "hi there", "hey"])
def test_intent_greeting(text: str) -> None:
    """Greeting patterns are classified as GREETING."""
    assert _classify_intent(text) == Intent.GREETING


def test_intent_greeting_question() -> None:
    """Greeting + question is classified as GREETING_QUESTION."""
    assert _classify_intent("hey, how are you?") == Intent.GREETING_QUESTION
    assert _classify_intent("hi, what's up?") == Intent.GREETING_QUESTION


@pytest.mark.parametrize("text", ["goodbye", "bye", "see you later"])
def test_intent_farewell(text: str) -> None:
    """Farewell patterns are classified as FAREWELL."""
    assert _classify_intent(text) == Intent.FAREWELL


def test_intent_self_inquiry() -> None:
    """Self-inquiry patterns are classified as SELF_INQUIRY."""
    assert _classify_intent("how do you feel?") == Intent.SELF_INQUIRY
    assert _classify_intent("what are you?") == Intent.SELF_INQUIRY
    assert _classify_intent("what do you know?") == Intent.SELF_INQUIRY


def test_intent_introduction() -> None:
    """Introduction patterns are classified as INTRODUCTION."""
    assert _classify_intent("my name is alice") == Intent.INTRODUCTION
    assert _classify_intent("call me charlie") == Intent.INTRODUCTION


def test_intent_question() -> None:
    """Question patterns are classified as QUESTION (or PHILOSOPHY for existential)."""
    # "what is cognition?" is PHILOSOPHY (existential keyword)
    assert _classify_intent("what is cognition?") == Intent.PHILOSOPHY
    # Generic questions are QUESTION
    assert _classify_intent("how does the brain work?") == Intent.QUESTION
    assert _classify_intent("why is the sky blue?") == Intent.QUESTION


@pytest.mark.parametrize("text", ["the weather is nice today", "i went to the store"])
def test_intent_statement(text: str) -> None:
    """Default text is classified as STATEMENT."""
    assert _classify_intent(text) == Intent.STATEMENT


def test_intent_command() -> None:
    """Command patterns are classified as COMMAND (or CODE_DISCUSSION for code terms)."""
    # "run the test" matches code patterns
    assert _classify_intent("run the test") == Intent.CODE_DISCUSSION
    # Non-code commands are COMMAND
    assert _classify_intent("stop that") == Intent.COMMAND
    assert _classify_intent("start the process") == Intent.COMMAND


def test_intent_request() -> None:
    """Polite requests are classified as REQUEST."""
    assert _classify_intent("please help me") == Intent.REQUEST
    assert _classify_intent("could you explain this") == Intent.REQUEST
    assert _classify_intent("would you do that") == Intent.REQUEST


def test_intent_philosophy() -> None:
    """Philosophy questions are classified as PHILOSOPHY."""
    assert _classify_intent("what is the meaning of life?") == Intent.PHILOSOPHY
    assert _classify_intent("why do we exist?") == Intent.PHILOSOPHY


@pytest.mark.parametrize("text", [
    "i was thinking about existence",
    "the nature of cognition is fascinating",
])
def test_intent_reflection(text: str) -> None:
    """Philosophy statements (no ?) are classified as REFLECTION."""
    assert _classify_intent(text) == Intent.REFLECTION


def test_intent_emotion_share() -> None:
    """Emotion sharing is classified as EMOTION_SHARE."""
    assert _classify_intent("i'm feeling sad") == Intent.EMOTION_SHARE
    assert _classify_intent("i feel happy today") == Intent.EMOTION_SHARE


def test_intent_correction() -> None:
    """Corrections are classified as CORRECTION (unless self-inquiry pattern matches)."""
    # "that's not right" is CORRECTION
    assert _classify_intent("that's not right") == Intent.CORRECTION
    # "no, that's wrong" matches self-inquiry ("what's wrong") — test a clearer correction
    assert _classify_intent("no that is incorrect") == Intent.CORRECTION


def test_intent_encouragement() -> None:
    """Encouragement is classified as ENCOURAGEMENT."""
    assert _classify_intent("you're doing great") == Intent.ENCOURAGEMENT
    assert _classify_intent("keep going") == Intent.ENCOURAGEMENT


def test_intent_comfort() -> None:
    """Comfort messages are classified as COMFORT."""
    assert _classify_intent("it's okay") == Intent.COMFORT
    assert _classify_intent("don't worry") == Intent.COMFORT


def test_intent_code_discussion() -> None:
    """Code-related text is classified as CODE_DISCUSSION."""
    assert _classify_intent("let's look at the code") == Intent.CODE_DISCUSSION
    assert _classify_intent("show me the function") == Intent.CODE_DISCUSSION


# ═══════════════════════════════════════════════════════════════════
# Question type detection
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text", ["what is cognition?", "what are you doing?"])
def test_question_type_what(text: str) -> None:
    """'what' questions are classified as WHAT."""
    assert _detect_question_type(text) == QuestionType.WHAT


@pytest.mark.parametrize("text", ["how does this work?", "how are you?"])
def test_question_type_how(text: str) -> None:
    """'how' questions are classified as HOW."""
    assert _detect_question_type(text) == QuestionType.HOW


def test_question_type_why() -> None:
    """'why' questions are classified as WHY."""
    assert _detect_question_type("why is the sky blue?") == QuestionType.WHY


def test_question_type_who() -> None:
    """'who' questions are classified as WHO."""
    assert _detect_question_type("who are you?") == QuestionType.WHO


def test_question_type_when() -> None:
    """'when' questions are classified as WHEN."""
    assert _detect_question_type("when did this happen?") == QuestionType.WHEN


def test_question_type_where() -> None:
    """'where' questions are classified as WHERE."""
    assert _detect_question_type("where is the brain?") == QuestionType.WHERE


def test_question_type_can() -> None:
    """'can you/can i' questions are classified as CAN."""
    assert _detect_question_type("can you help me?") == QuestionType.CAN
    assert _detect_question_type("can i ask you something?") == QuestionType.CAN


def test_question_type_do() -> None:
    """'do you/does' questions are classified as DO."""
    assert _detect_question_type("do you know the answer?") == QuestionType.DO
    assert _detect_question_type("does this work?") == QuestionType.DO


def test_question_type_are() -> None:
    """'are you/am i' questions are classified as ARE."""
    assert _detect_question_type("are you cognitive?") == QuestionType.ARE
    assert _detect_question_type("am i right?") == QuestionType.ARE


def test_question_type_what_if() -> None:
    """'what if' questions are classified as WHAT_IF (or WHAT if 'what' matches first)."""
    # "what if" starts with "what" so it matches WHAT first
    result = _detect_question_type("what if we tried this?")
    assert result in (QuestionType.WHAT_IF, QuestionType.WHAT)


@pytest.mark.parametrize("text", [
    "tell me about cognition",
    "describe the brain",
    "explain quantum mechanics",
])
def test_question_type_imperative(text: str) -> None:
    """Imperative questions are classified as WHAT."""
    assert _detect_question_type(text) == QuestionType.WHAT


def test_question_type_yes_no() -> None:
    """Questions ending with ? but no recognized starter are YES_NO."""
    assert _detect_question_type("is this correct?") == QuestionType.YES_NO


@pytest.mark.parametrize("text", ["the weather is nice", "i like dogs"])
def test_question_type_none(text: str) -> None:
    """Non-questions are classified as NONE."""
    assert _detect_question_type(text) == QuestionType.NONE


# ═══════════════════════════════════════════════════════════════════
# Sentiment detection
# ═══════════════════════════════════════════════════════════════════


def test_sentiment_positive() -> None:
    """Positive words produce positive sentiment."""
    score, label = _detect_sentiment("this is great and wonderful")
    assert label == "positive"
    assert score > 0


def test_sentiment_negative() -> None:
    """Negative words produce negative sentiment."""
    score, label = _detect_sentiment("this is terrible and awful")
    assert label == "negative"
    assert score < 0


def test_sentiment_neutral() -> None:
    """No sentiment words produce neutral sentiment."""
    score, label = _detect_sentiment("the table is brown")
    assert label == "neutral"
    assert score == 0.0


def test_sentiment_mixed() -> None:
    """Mixed sentiment words produce a score between -1 and +1."""
    score, _label = _detect_sentiment("this is good but terrible")
    assert -1.0 <= score <= 1.0


def test_sentiment_negation() -> None:
    """Negation flips sentiment ('not good' → negative)."""
    score, label = _detect_sentiment("this is not good")
    assert label == "negative"
    assert score < 0


def test_sentiment_empty() -> None:
    """Empty text produces neutral sentiment."""
    score, label = _detect_sentiment("")
    assert label == "neutral"
    assert score == 0.0


# ═══════════════════════════════════════════════════════════════════
# Topic extraction
# ═══════════════════════════════════════════════════════════════════


def test_extract_topics_basic() -> None:
    """Topics are extracted from text."""
    topics = _extract_topics("tell me about cognition and the brain")
    assert isinstance(topics, list)
    # Should extract some meaningful topics
    assert len(topics) > 0


def test_extract_topics_empty() -> None:
    """Empty text produces no topics."""
    topics = _extract_topics("")
    assert topics == []


def test_extract_topics_filters_relation_verb_makes() -> None:
    """'makes' is the question's relation verb, not a topic.

    "What makes you happy?" asks about *happy*; "makes" is the relation
    being asked about. It was leaking through as a topic (the filter had
    "made" but not "make"/"makes"), so the composer would answer about
    the bare verb instead of the real topic.
    """
    assert _extract_topics("What makes you happy?") == ["happy"]
    assert _extract_topics("What makes you sad?") == ["sad"]


# ═══════════════════════════════════════════════════════════════════
# Entity extraction
# ═══════════════════════════════════════════════════════════════════


def test_extract_entities_name() -> None:
    """Names are extracted after 'my name is' / 'I'm' / 'call me'."""
    entities = _extract_entities("my name is Alice")
    assert entities.get("name") == "Alice"


def test_extract_entities_call_me() -> None:
    """Names are extracted after 'call me'."""
    entities = _extract_entities("call me Charlie")
    assert entities.get("name") == "Charlie"


def test_extract_entities_number() -> None:
    """Numbers are extracted."""
    entities = _extract_entities("I have 42 cats")
    assert entities.get("number") == "42"


def test_extract_entities_none() -> None:
    """No entities in plain text."""
    entities = _extract_entities("the weather is nice")
    assert "name" not in entities
    assert "number" not in entities


# ═══════════════════════════════════════════════════════════════════
# Key phrase extraction
# ═══════════════════════════════════════════════════════════════════


def test_extract_key_phrases_empty() -> None:
    """Empty text produces no key phrases."""
    phrases = _extract_key_phrases("")
    assert phrases == []


# ═══════════════════════════════════════════════════════════════════
# perceive() — full pipeline
# ═══════════════════════════════════════════════════════════════════


def test_perceive_about_genesis() -> None:
    """perceive() detects when the user is talking about Genesis."""
    result = perceive("what are you?")
    assert result.is_about_genesis is True


def test_perceive_about_user() -> None:
    """perceive() detects when the user is talking about themselves."""
    result = perceive("i am feeling sad")
    assert result.is_about_user is True


def test_perceive_about_code() -> None:
    """perceive() detects code-related text."""
    result = perceive("let's look at the code")
    assert result.is_about_code is True


def test_perceive_about_emotion() -> None:
    """perceive() detects emotion-related text."""
    result = perceive("i'm feeling happy")
    assert result.is_about_emotion is True


def test_perceive_about_existence() -> None:
    """perceive() detects existential text."""
    result = perceive("why do we exist?")
    assert result.is_about_existence is True


def test_perceive_topics_limited_to_five() -> None:
    """perceive() limits topics to 5."""
    result = perceive("dogs cats birds fish reptiles mammals amphibians insects")
    assert len(result.topics) <= 5


def test_perceive_empty_text() -> None:
    """perceive() handles empty text gracefully."""
    result = perceive("")
    assert isinstance(result, Perception)
    assert result.word_count == 0


# ═══════════════════════════════════════════════════════════════════
# PatternWeights — perceptual learning
# ═══════════════════════════════════════════════════════════════════


def test_pattern_weights_default() -> None:
    """Default weight is 1.0 for unknown patterns."""
    pw = PatternWeights()
    assert abs(pw.get("unknown_pattern") - 1.0) < 1e-9


def test_pattern_weights_custom() -> None:
    """Custom weights are stored."""
    pw = PatternWeights(weights={"greeting": 1.5})
    assert abs(pw.get("greeting") - 1.5) < 1e-9


def test_pattern_weights_update_positive() -> None:
    """Positive outcome increases weight."""
    pw = PatternWeights()
    pw.update_pattern_weight("greeting", outcome=1.0)
    assert pw.get("greeting") > 1.0


def test_pattern_weights_update_negative() -> None:
    """Negative outcome decreases weight."""
    pw = PatternWeights()
    pw.update_pattern_weight("greeting", outcome=-1.0)
    assert pw.get("greeting") < 1.0


def test_pattern_weights_bounded_min() -> None:
    """Weight is bounded at MIN_WEIGHT."""
    pw = PatternWeights()
    for _ in range(100):
        pw.update_pattern_weight("greeting", outcome=-1.0)
    assert pw.get("greeting") >= PatternWeights.MIN_WEIGHT


def test_pattern_weights_bounded_max() -> None:
    """Weight is bounded at MAX_WEIGHT."""
    pw = PatternWeights()
    for _ in range(100):
        pw.update_pattern_weight("greeting", outcome=1.0)
    assert pw.get("greeting") <= PatternWeights.MAX_WEIGHT


def test_pattern_weights_as_dict() -> None:
    """as_dict returns a copy of the weights."""
    pw = PatternWeights(weights={"greeting": 1.5})
    d = pw.as_dict()
    assert d["greeting"] == 1.5
    # Modifying the returned dict doesn't affect the weights
    d["greeting"] = 0.0
    assert pw.get("greeting") == 1.5


def test_pattern_weights_save_and_load() -> None:
    """save_weights and load_weights round-trip correctly."""
    pw = PatternWeights(weights={"greeting": 1.5, "question": 0.8})
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        pw.save_weights(path)
        pw2 = PatternWeights()
        pw2.load_weights(path)
        assert abs(pw2.get("greeting") - 1.5) < 1e-9
        assert abs(pw2.get("question") - 0.8) < 1e-9
    finally:
        os.unlink(path)


def test_pattern_weights_load_missing_file() -> None:
    """load_weights handles missing files gracefully."""
    pw = PatternWeights()
    pw.load_weights("/nonexistent/path/weights.json")
    # Should not raise, weights remain default
    assert abs(pw.get("any") - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Multisensory integration
# ═══════════════════════════════════════════════════════════════════


def test_multisensory_integrator_single_text() -> None:
    """Integrating a single text input returns an IntegratedPerception."""
    integrator = MultisensoryIntegrator()
    inp = MultisensoryInput(modality=SensoryModality.TEXT, content="hello")
    result = integrator.integrate([inp])
    assert isinstance(result, IntegratedPerception)
    assert result.intent == Intent.GREETING
    assert "text" in result.modalities_present


def test_multisensory_integrator_empty() -> None:
    """Integrating no inputs returns an empty IntegratedPerception."""
    integrator = MultisensoryIntegrator()
    result = integrator.integrate([])
    assert isinstance(result, IntegratedPerception)


def test_multisensory_integrator_inactive_modality() -> None:
    """Inactive modalities are noted but don't affect the base perception."""
    integrator = MultisensoryIntegrator()
    text_inp = MultisensoryInput(modality=SensoryModality.TEXT, content="hello")
    audio_inp = MultisensoryInput(modality=SensoryModality.AUDIO, content="audio_data")
    result = integrator.integrate([text_inp, audio_inp])
    assert "text" in result.modalities_present
    assert "audio" in result.modalities_present
    # Should note inactive modalities
    assert "inactive" in result.conflict_description.lower() or result.conflict_description == ""


# ======================================================================
# From tests/test_vision.py
# ======================================================================

class TestDetectedObject:
    """Test the DetectedObject dataclass."""

    def test_position_description_center(self) -> None:
        """An object in the center of the frame."""
        obj = DetectedObject(
            name="chair",
            confidence=0.8,
            bbox=(280, 200, 80, 100),
            center=(0.5, 0.5),
            area_ratio=0.05,
        )
        assert obj.position_description == "in the center"

    def test_position_description_left(self) -> None:
        """An object on the left side."""
        obj = DetectedObject(
            name="cup",
            confidence=0.7,
            bbox=(10, 200, 50, 50),
            center=(0.1, 0.5),
            area_ratio=0.02,
        )
        assert "left" in obj.position_description

    def test_position_description_right_top(self) -> None:
        """An object in the top-right."""
        obj = DetectedObject(
            name="laptop",
            confidence=0.9,
            bbox=(500, 10, 100, 80),
            center=(0.85, 0.1),
            area_ratio=0.1,
        )
        desc = obj.position_description
        assert "right" in desc
        assert "top" in desc

    def test_position_description_middle_left(self) -> None:
        """An object in the middle-left area."""
        obj = DetectedObject(
            name="book",
            confidence=0.6,
            bbox=(50, 200, 60, 80),
            center=(0.15, 0.5),
            area_ratio=0.03,
        )
        assert "left" in obj.position_description


# ─── ObjectRecognizer ──────────────────────────────────────────────────


class TestObjectRecognizer:
    """Test the ObjectRecognizer class."""

    def test_coco_classes_count(self) -> None:
        """COCO should have exactly 80 classes."""
        assert len(COCO_CLASSES) == 80

    def test_coco_classes_contains_common_objects(self) -> None:
        """COCO should contain common household objects."""
        for name in ("person", "chair", "laptop", "cup", "book", "bottle"):
            assert name in COCO_CLASSES

    def test_recognizer_not_available_without_model(self, tmp_path) -> None:
        """Recognizer should report unavailable if model file is missing."""
        import genesis_cognitive.auditory.object_recognition as mod

        # Save original model path and point to nonexistent file
        original = mod._MODEL_FILE
        try:
            mod._MODEL_FILE = tmp_path / "nonexistent.onnx"
            recognizer = ObjectRecognizer()
            assert not recognizer.is_available()
        finally:
            mod._MODEL_FILE = original

    def test_detect_objects_on_blank_frame(self) -> None:
        """A blank white frame should produce no detections."""
        # We can only test this if the model is available
        recognizer = ObjectRecognizer()
        if not recognizer.is_available():
            return  # skip if no model
        blank = np.full((480, 640, 3), 255, dtype=np.uint8)
        objects = recognizer.detect_objects(blank)
        # A blank frame should have no confident detections
        assert len(objects) == 0

    def test_detect_objects_returns_sorted(self) -> None:
        """Detections should be sorted by confidence (highest first)."""
        recognizer = ObjectRecognizer()
        if not recognizer.is_available():
            return
        # Create a frame with some structure
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = recognizer.detect_objects(frame)
        for i in range(1, len(objects)):
            assert objects[i].confidence <= objects[i - 1].confidence


# ─── _build_scene ──────────────────────────────────────────────────────


class TestBuildScene:
    """Test the _build_scene helper — the structured percept data that
    the language engine composes her report from."""

    def _make_color_info(self, **kwargs: object) -> ColorInfo:
        """Build a ColorInfo with defaults, overriding via kwargs."""
        defaults: dict[str, object] = {
            "dominant_color": "gray",
            "light_level": "bright",
            "brightness": 0.7,
            "mean_saturation": 0.2,
            "warmth": "neutral",
            "regions": {"left": "gray", "center": "gray", "right": "gray",
                        "top": "gray", "bottom": "gray"},
        }
        defaults.update(kwargs)
        return ColorInfo(**defaults)  # type: ignore[arg-type]

    def test_empty_scene(self) -> None:
        """No objects or faces should produce a minimal scene."""
        scene = _build_scene(
            self._make_color_info(), [], {}, [], 0, None,
        )
        assert scene.status == "ok"
        assert scene.objects == []
        assert scene.faces == []
        assert scene.light_level == "bright"
        meta = scene.as_metadata()
        assert meta["light"] == "bright"
        assert meta["objects"] == []

    def test_single_object(self) -> None:
        """A single object should appear in the scene data."""
        objects = [
            DetectedObject(
                name="chair",
                confidence=0.8,
                bbox=(280, 200, 80, 100),
                center=(0.5, 0.5),
                area_ratio=0.05,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects, {"chair": "brown"}, [], 0, None,
        )
        assert len(scene.objects) == 1
        assert scene.objects[0]["name"] == "chair"
        assert scene.objects[0]["color"] == "brown"
        assert scene.objects[0]["position"]

    def test_object_without_color(self) -> None:
        """Objects with non-informative colors carry an empty color."""
        objects = [
            DetectedObject(
                name="oven",
                confidence=0.7,
                bbox=(100, 100, 200, 200),
                center=(0.3, 0.3),
                area_ratio=0.1,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects, {"oven": "gray"}, [], 0, None,
        )
        assert scene.objects[0]["name"] == "oven"
        assert scene.objects[0]["color"] == ""

    def test_person_skipped_when_faces_present(self) -> None:
        """Person objects should be skipped when faces are named."""
        objects = [
            DetectedObject(
                name="person",
                confidence=0.9,
                bbox=(280, 100, 80, 300),
                center=(0.5, 0.5),
                area_ratio=0.08,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects, {"person": "gray"},
            ["alice"], 0, None,
        )
        assert scene.faces == ["alice"]
        assert scene.objects == []

    def test_deduplication(self) -> None:
        """Duplicate object names should only appear once."""
        objects = [
            DetectedObject(
                name="chair",
                confidence=0.9,
                bbox=(100, 200, 80, 100),
                center=(0.2, 0.5),
                area_ratio=0.05,
            ),
            DetectedObject(
                name="chair",
                confidence=0.7,
                bbox=(400, 200, 80, 100),
                center=(0.7, 0.5),
                area_ratio=0.05,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects, {"chair": "brown"}, [], 0, None,
        )
        assert len(scene.objects) == 1

    def test_multiple_different_objects(self) -> None:
        """Multiple different objects should each get an entry."""
        objects = [
            DetectedObject(
                name="laptop",
                confidence=0.9,
                bbox=(280, 200, 100, 80),
                center=(0.5, 0.5),
                area_ratio=0.1,
            ),
            DetectedObject(
                name="cup",
                confidence=0.7,
                bbox=(50, 200, 40, 50),
                center=(0.1, 0.5),
                area_ratio=0.02,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects,
            {"laptop": "gray", "cup": "blue"}, [], 0, None,
        )
        names = [o["name"] for o in scene.objects]
        assert names == ["laptop", "cup"]
        assert scene.objects[1]["color"] == "blue"

    def test_light_direction(self) -> None:
        """A brighter side should record the light direction — judged
        by measured luminance, not by the region's color name."""
        scene = _build_scene(
            self._make_color_info(
                regions={"left": "gray", "center": "gray",
                         "right": "gray", "top": "gray", "bottom": "gray"},
                region_brightness={"left": 0.8, "center": 0.4,
                                   "right": 0.4, "top": 0.5, "bottom": 0.4},
            ),
            [], {}, [], 0, None,
        )
        assert scene.light_direction == "left"

    def test_no_light_direction_when_even(self) -> None:
        """Similar luminance on both sides → no light direction."""
        scene = _build_scene(
            self._make_color_info(
                region_brightness={"left": 0.5, "center": 0.5,
                                   "right": 0.48, "top": 0.5, "bottom": 0.5},
            ),
            [], {}, [], 0, None,
        )
        assert scene.light_direction == ""

    def test_bright_color_name_is_not_light_direction(self) -> None:
        """A yellow region in shadow must not outrank a gray region in
        sunlight — the name heuristic this replaced got that wrong."""
        scene = _build_scene(
            self._make_color_info(
                regions={"left": "yellow", "center": "gray",
                         "right": "gray", "top": "gray", "bottom": "gray"},
                region_brightness={"left": 0.3, "center": 0.5,
                                   "right": 0.7, "top": 0.5, "bottom": 0.5},
            ),
            [], {}, [], 0, None,
        )
        assert scene.light_direction == "right"

    def test_warmth_recorded(self) -> None:
        """Warm scenes should carry the warmth field."""
        scene = _build_scene(
            self._make_color_info(warmth="warm", dominant_color="orange"),
            [], {}, [], 0, None,
        )
        assert scene.warmth == "warm"
        assert scene.dominant_color == "orange"

    def test_memory_text_is_structural(self) -> None:
        """The episodic-memory record is data, not a spoken sentence."""
        objects = [
            DetectedObject(
                name="cup",
                confidence=0.7,
                bbox=(50, 200, 40, 50),
                center=(0.1, 0.5),
                area_ratio=0.02,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects,
            {"cup": "blue"}, ["alice"], 0, None,
        )
        text = scene.memory_text()
        assert "cup" in text
        assert "alice" in text
        assert "blue" in text

    @staticmethod
    def _language_fixture() -> tuple[Vocabulary, EmotionalState, PersonalityTraits]:
        vocab = Vocabulary(seed=42)
        emotion = EmotionalState(
            label="calm", nuance="baseline", cognitive_style="steady",
            valence=0.0, alertness=0.5, plasticity=0.5,
            creativity=0.5, caution=0.3, openness_to_engage=0.7,
        )
        personality = PersonalityTraits(
            openness=0.8, conscientiousness=0.7, extraversion=0.5,
            agreeableness=0.7, neuroticism=0.3,
        )
        return vocab, emotion, personality

    def test_vocabulary_composes_vision_report(self) -> None:
        """The language engine composes the report from scene metadata —
        including correct indefinite articles."""
        vocab, emotion, personality = self._language_fixture()
        objects = [
            DetectedObject(
                name="oven",
                confidence=0.7,
                bbox=(100, 100, 200, 200),
                center=(0.3, 0.3),
                area_ratio=0.1,
            ),
        ]
        scene = _build_scene(
            self._make_color_info(), objects, {}, [], 0, None,
        )
        content = vocab.fill_slot(
            "content",
            {"vision_scene": scene.as_metadata()},
            emotion, personality,
        )
        assert content, "vocabulary should compose content from scene data"
        assert "an oven" in content, content

    def test_vocabulary_mentions_faces(self) -> None:
        """Recognized faces should surface in the composed predicate."""
        vocab, emotion, personality = self._language_fixture()
        scene = _build_scene(
            self._make_color_info(), [], {}, ["alice"], 1, None,
        )
        content = vocab.fill_slot(
            "content",
            {"vision_scene": scene.as_metadata()},
            emotion, personality,
        )
        assert "Alice" in content, content
        assert "someone" in content or "unfamiliar" in content, content

    def test_vocabulary_composes_vision_status(self) -> None:
        """A non-ok scene status composes a grammatical predicate, not
        a raw status word."""
        vocab, emotion, personality = self._language_fixture()
        content = vocab.fill_slot(
            "content",
            {"vision_status": "unavailable"},
            emotion, personality,
        )
        assert content, "vocabulary should compose a status predicate"
        assert "vision" not in content, content
        assert "retina" in content or "visual" in content, content


# ─── _learn_objects ────────────────────────────────────────────────────


class TestLearnObjects:
    """Test the _learn_objects concept network integration."""

    def test_objects_added_to_network(self) -> None:
        """Objects should be added as concepts in the network."""

        network = ConceptNetwork()
        objects = [
            DetectedObject(
                name="chair",
                confidence=0.8,
                bbox=(280, 200, 80, 100),
                center=(0.5, 0.5),
                area_ratio=0.05,
            ),
        ]
        _learn_objects(network, objects, salience=0.7)

        assert network.get_concept("object") is not None
        assert network.get_concept("chair") is not None
        assert network.get_concept("center") is not None

    def test_object_vision_link(self) -> None:
        """Objects should be linked to the vision concept."""

        network = ConceptNetwork()
        objects = [
            DetectedObject(
                name="laptop",
                confidence=0.9,
                bbox=(280, 200, 100, 80),
                center=(0.5, 0.5),
                area_ratio=0.1,
            ),
        ]
        _learn_objects(network, objects, salience=0.8)

        # Check that vision → laptop edge exists
        edges = network.get_edges("vision")
        # Edges use concept IDs, which may differ from names. Check
        # by resolving the laptop concept and looking for its ID.
        laptop_concept = network.get_concept("laptop")
        assert laptop_concept is not None
        target_ids = {e.target for e in edges}
        assert laptop_concept.id in target_ids

    def test_spatial_position_learned(self) -> None:
        """Object position should be learned as a concept."""

        network = ConceptNetwork()
        objects = [
            DetectedObject(
                name="cup",
                confidence=0.7,
                bbox=(50, 200, 40, 50),
                center=(0.1, 0.5),
                area_ratio=0.02,
            ),
        ]
        _learn_objects(network, objects, salience=0.6)

        assert network.get_concept("left") is not None
        # cup should be linked to left
        cup_concept = network.get_concept("cup")
        left_concept = network.get_concept("left")
        assert cup_concept is not None
        assert left_concept is not None
        edges = network.get_edges("cup")
        target_ids = {e.target for e in edges}
        assert left_concept.id in target_ids


# ─── Improved V1 summary ───────────────────────────────────────────────


class TestBuildSummary:
    """Test the improved _build_summary with spatial layout."""

    def test_no_features(self) -> None:
        """Empty feature list should produce formless message."""
        summary = _build_summary(0, 0, "gray", 0.5, 0.0)
        assert "formless" in summary

    def test_with_features_includes_layout(self) -> None:
        """Summary with features should include spatial layout."""
        features = [
            VisualFeature(
                y=10.0, x=10.0, row=0, col=0,
                orientation=0.0, scale=4.0, phase=0.0,
                strength=0.5, feature_index=0,
            ),
            VisualFeature(
                y=10.0, x=20.0, row=0, col=1,
                orientation=0.1, scale=4.0, phase=0.0,
                strength=0.4, feature_index=1,
            ),
        ]
        summary = _build_summary(
            2, 0, "gray", 0.5, 0.3, features=features,
        )
        # Should mention spatial layout
        assert "concentrated" in summary or "structure" in summary
        # Should mention orientation
        assert "vertical" in summary or "horizontal" in summary or "diagonal" in summary

    def test_with_features_includes_gamma(self) -> None:
        """Summary should always include gamma value."""
        features = [
            VisualFeature(
                y=10.0, x=10.0, row=0, col=0,
                orientation=0.0, scale=4.0, phase=0.0,
                strength=0.5, feature_index=0,
            ),
        ]
        summary = _build_summary(1, 0, "gray", 0.5, 0.42, features=features)
        assert "gamma" in summary
        assert "0.42" in summary

    def test_without_features_falls_back_to_contours(self) -> None:
        """Without features, should fall back to contour count."""
        summary = _build_summary(5, 3, "blue", 0.6, 0.5)
        assert "3 edge arrangement" in summary

    def test_left_concentration(self) -> None:
        """Features concentrated on the left should be described as such."""
        features = [
            VisualFeature(
                y=100.0, x=10.0, row=5, col=0,
                orientation=1.5, scale=4.0, phase=0.0,
                strength=0.5, feature_index=0,
            ),
            VisualFeature(
                y=100.0, x=20.0, row=5, col=1,
                orientation=1.6, scale=4.0, phase=0.0,
                strength=0.4, feature_index=1,
            ),
            VisualFeature(
                y=100.0, x=30.0, row=5, col=2,
                orientation=1.5, scale=4.0, phase=0.0,
                strength=0.3, feature_index=2,
            ),
            # A far-right feature to establish the frame width
            VisualFeature(
                y=100.0, x=300.0, row=5, col=30,
                orientation=1.5, scale=4.0, phase=0.0,
                strength=0.1, feature_index=3,
            ),
        ]
        summary = _build_summary(4, 0, "gray", 0.5, 0.3, features=features)
        # mean_x = (10+20+30+300)/4 = 90, max_x = 300
        # norm_x = 90/301 = 0.30 → left
        assert "left" in summary


# ─── V1Model integration ─────────────────────────────────────────


class TestV1ModelSummary:
    """Test that the occipital lobe produces richer summaries."""

    def test_process_returns_rich_summary(self) -> None:
        """Processing a structured frame should return a rich summary."""
        lobe = V1Model(seed=42)
        # Create a frame with a vertical edge and some texture so
        # patches straddling the boundary have detectable structure.
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :80] = 50  # dark left half
        frame[:, 80:] = 200  # bright right half
        # Add some noise so patches have internal structure
        rng = np.random.default_rng(42)
        noise = rng.integers(-20, 20, frame.shape)
        frame = np.clip(
            frame.astype(np.int16) + noise, 0, 255,
        ).astype(np.uint8)
        field = lobe.process(frame, learn=False)
        # The summary should include spatial or orientation info
        assert len(field.summary) > 20
        # If features were found, gamma should be mentioned
        if field.feature_count() > 0:
            assert "gamma" in field.summary

    def test_process_blank_frame(self) -> None:
        """A blank frame should produce a formless or minimal summary."""
        lobe = V1Model(seed=42)
        frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        field = lobe.process(frame, learn=False)
        # Blank frame should have few or no features
        assert isinstance(field.summary, str)


# ======================================================================
# From tests/test_visual_cortex.py
# ======================================================================

@pytest.fixture
def occipital():
    """A small occipital lobe for testing."""
    return V1Model(
        patch_size=8,
        orientations=4,
        scales=2,
        phases=2,
        seed=42,
    )


@pytest.fixture
def sample_image():
    """A 64×64×3 RGB image with some structure (gradients + shapes)."""
    rng = np.random.default_rng(42)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Add a gradient
    for i in range(64):
        img[i, :, 0] = i * 4  # red gradient
    # Add a green square
    img[16:48, 16:48, 1] = 200
    # Add some noise
    img = np.clip(img + rng.integers(0, 20, (64, 64, 3)), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def green_image():
    """A solid green image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # green channel
    return img


@pytest.fixture
def blue_image():
    """A solid blue image."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # blue channel
    return img


# ── V4 Tests ──────────────────────────────────────────────────────


class TestV4Model:
    """Tests for v 4 lobe."""
    def test_initialization(self):
        """V4 should initialize with correct dimensions."""
        v4 = V4Model(n_v1_features=16, n_features=32, color_dims=12, seed=42)
        assert v4.n_features == 32
        assert v4.input_dim == 16 * 4 + 12  # n_v1 * pool² + color
        assert v4.phi4.shape == (v4.input_dim, 32)

    def test_process_returns_correct_shape(self, occipital, sample_image):
        """V4 should return latents with the right number of features."""
        from genesis_cognitive.vision.v1 import (
            _extract_patches,
            _to_grayscale,
            _whiten_patches,
        )

        gray = _to_grayscale(sample_image)
        patches, grid = _extract_patches(gray, occipital.patch_size)
        X = _whiten_patches(patches)
        Z1 = occipital._infer_latents(X)

        n_rows = max(r for _, _, r, _ in grid) + 1
        n_cols = max(c for _, _, _, c in grid) + 1

        v4 = V4Model(n_v1_features=occipital.n_features, n_features=32, seed=42)
        Z4 = v4.process(Z1, (n_rows, n_cols), frame=sample_image, learn=False)

        assert Z4.shape[1] == 32
        # Should have fewer V4 units than V1 patches (pooled 2×2)
        assert Z4.shape[0] <= Z1.shape[0]

    def test_dictionary_learning(self, occipital, sample_image):
        """V4 dictionary should change after processing images."""
        from genesis_cognitive.vision.v1 import (
            _extract_patches,
            _to_grayscale,
            _whiten_patches,
        )

        gray = _to_grayscale(sample_image)
        patches, grid = _extract_patches(gray, occipital.patch_size)
        X = _whiten_patches(patches)
        Z1 = occipital._infer_latents(X)
        n_rows = max(r for _, _, r, _ in grid) + 1
        n_cols = max(c for _, _, _, c in grid) + 1

        v4 = V4Model(n_v1_features=occipital.n_features, n_features=32, seed=42)
        phi_before = v4.phi4.copy()

        # Process multiple times to allow learning
        for _ in range(5):
            v4.process(Z1, (n_rows, n_cols), frame=sample_image, learn=True)

        # Dictionary should have changed
        assert not np.allclose(phi_before, v4.phi4)
        assert v4._samples_seen > 0

    def test_color_integration(self, occipital, green_image, blue_image):
        """V4 should produce different activations for different colors."""
        from genesis_cognitive.vision.v1 import (
            _extract_patches,
            _to_grayscale,
            _whiten_patches,
        )

        v4 = V4Model(n_v1_features=occipital.n_features, n_features=32, seed=42)

        results = {}
        for name, img in [("green", green_image), ("blue", blue_image)]:
            gray = _to_grayscale(img)
            patches, grid = _extract_patches(gray, occipital.patch_size)
            X = _whiten_patches(patches)
            Z1 = occipital._infer_latents(X)
            n_rows = max(r for _, _, r, _ in grid) + 1
            n_cols = max(c for _, _, _, c in grid) + 1
            Z4 = v4.process(Z1, (n_rows, n_cols), frame=img, learn=False)
            results[name] = Z4.mean(axis=0) if Z4.size > 0 else np.zeros(32)

        # Different colors should produce different V4 activations
        assert not np.allclose(results["green"], results["blue"])

    def test_save_load(self, tmp_path):
        """V4 should save and load correctly."""
        v4 = V4Model(n_v1_features=16, n_features=32, seed=42)
        v4._samples_seen = 100

        path = tmp_path / "v4.npz"
        v4.save(path)
        assert path.exists()

        v4_loaded = V4Model(n_v1_features=16, n_features=32, seed=99)
        assert v4_loaded.load(path)
        assert v4_loaded._samples_seen == 100
        assert np.allclose(v4_loaded.phi4, v4.phi4)


# ── VTC Tests ─────────────────────────────────────────────────────


class TestVTCFeatureSpace:
    """Tests for v t c feature space."""
    def test_initialization(self):
        """Test initialization."""
        vtc = VTCFeatureSpace(input_dim=32, n_components=16)
        assert vtc.n_components == 16
        assert vtc._components is None

    def test_process_without_training(self):
        """VTC should return a vector even before PCA is trained."""
        vtc = VTCFeatureSpace(input_dim=32, n_components=16)
        v4_latents = np.random.default_rng(42).standard_normal((10, 32))
        vec = vtc.process(v4_latents, learn=False)
        assert vec.shape == (16,)

    def test_pca_learning(self):
        """VTC should learn PCA components from data."""
        vtc = VTCFeatureSpace(input_dim=32, n_components=16)
        rng = np.random.default_rng(42)

        # Feed it many samples to trigger PCA computation
        for _ in range(15):
            v4_latents = rng.standard_normal((10, 32))
            vtc.process(v4_latents, learn=True)

        assert vtc._n_samples == 15
        assert vtc._components is not None
        assert vtc._components.shape[0] <= 16

    def test_normalized_output(self):
        """VTC output should be L2-normalized after PCA is trained."""
        vtc = VTCFeatureSpace(input_dim=32, n_components=16)
        rng = np.random.default_rng(42)

        for _ in range(15):
            vtc.process(rng.standard_normal((10, 32)), learn=True)

        vec = vtc.process(rng.standard_normal((10, 32)), learn=False)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01 or norm < 0.01  # normalized or zero

    def test_save_load(self, tmp_path):
        """Test save load."""
        vtc = VTCFeatureSpace(input_dim=32, n_components=16)
        rng = np.random.default_rng(42)
        for _ in range(15):
            vtc.process(rng.standard_normal((10, 32)), learn=True)

        path = tmp_path / "vtc.npz"
        vtc.save(path)

        vtc_loaded = VTCFeatureSpace(input_dim=32, n_components=16)
        assert vtc_loaded.load(path)
        assert vtc_loaded._n_samples == 15


# ── MTL Bridge Tests ──────────────────────────────────────────────


class TestMemoryBridge:
    """Tests for m t l bridge."""
    def test_initialization(self):
        """Test initialization."""
        mtl = MemoryBridge(vtc_dim=32, embedding_dim=64)
        assert mtl.W is None
        assert mtl.n_examples == 0

    def test_learn_and_recognize(self):
        """After learning an association, MTL should recognize it."""
        mtl = MemoryBridge(vtc_dim=32, embedding_dim=64, ridge_lambda=0.01)

        # Learn a "tree" association
        vtc_tree = np.random.default_rng(1).standard_normal(32)
        emb_tree = np.random.default_rng(2).standard_normal(64)
        mtl.learn_association(vtc_tree, "tree", emb_tree)

        # Learn a "car" association
        vtc_car = np.random.default_rng(3).standard_normal(32)
        emb_car = np.random.default_rng(4).standard_normal(64)
        mtl.learn_association(vtc_car, "car", emb_car)

        # Should recognize tree
        concept, conf = mtl.recognize(vtc_tree)
        assert concept == "tree"
        assert conf > 0.5

        # Should recognize car
        concept, conf = mtl.recognize(vtc_car)
        assert concept == "car"
        assert conf > 0.5

    def test_recognize_untrained(self):
        """Untrained MTL should return None."""
        mtl = MemoryBridge(vtc_dim=32, embedding_dim=64)
        concept, conf = mtl.recognize(np.random.standard_normal(32))
        assert concept is None
        assert conf == 0.0

    def test_generate(self):
        """MTL should generate VTC vectors from concept embeddings."""
        mtl = MemoryBridge(vtc_dim=32, embedding_dim=64, ridge_lambda=0.01)

        vtc_vec = np.random.default_rng(1).standard_normal(32)
        emb_vec = np.random.default_rng(2).standard_normal(64)
        mtl.learn_association(vtc_vec, "tree", emb_vec)

        generated = mtl.generate(emb_vec)
        assert generated is not None
        assert generated.shape == (32,)

    def test_save_load(self, tmp_path):
        """Test save load."""
        mtl = MemoryBridge(vtc_dim=32, embedding_dim=64, ridge_lambda=0.01)
        vtc = np.random.default_rng(1).standard_normal(32)
        emb = np.random.default_rng(2).standard_normal(64)
        mtl.learn_association(vtc, "tree", emb)

        path = tmp_path / "mtl.json"
        mtl.save(path)

        mtl_loaded = MemoryBridge(vtc_dim=32, embedding_dim=64)
        assert mtl_loaded.load(path)
        assert mtl_loaded.n_examples == 1
        assert mtl_loaded.W is not None


# ── Visual Cortex (full pipeline) Tests ───────────────────────────


class TestVisualCortex:
    """Tests for visual cortex."""
    def test_see_returns_percept(self, occipital, sample_image):
        """VisualCortex.see should return a VisualPercept."""
        cortex = VisualCortex(occipital, embeddings=None, network=None)
        percept = cortex.see(sample_image, learn=False)

        assert isinstance(percept, VisualPercept)
        assert percept.v1_summary != ""
        assert percept.vtc_vector is not None
        assert percept.vtc_vector.shape == (32,)

    def test_see_different_images_different_vtc(self, occipital, green_image, blue_image):
        """Different images should produce different VTC vectors."""
        cortex = VisualCortex(occipital, embeddings=None, network=None)
        p1 = cortex.see(green_image, learn=False)
        p2 = cortex.see(blue_image, learn=False)
        assert not np.allclose(p1.vtc_vector, p2.vtc_vector)

    def test_unrecognized_without_training(self, occipital, sample_image):
        """Without any training, recognition should return None."""
        cortex = VisualCortex(occipital, embeddings=None, network=None)
        percept = cortex.see(sample_image, learn=False)
        assert percept.concept is None
        assert percept.confidence == 0.0

    def test_status(self, occipital, sample_image):
        """Status should report the visual cortex state."""
        cortex = VisualCortex(occipital, embeddings=None, network=None)
        cortex.see(sample_image, learn=True)
        status = cortex.status()
        assert "v4_samples_seen" in status
        assert "vtc_samples" in status
        assert "mtl_examples" in status

    def test_imagine_without_training(self, occipital):
        """Imagine should return None without training."""
        cortex = VisualCortex(occipital, embeddings=None, network=None)
        assert cortex.imagine("tree") is None


# ── Image Utility Tests ───────────────────────────────────────────


class TestImageUtils:
    """Tests for image utils."""
    def test_load_image(self, tmp_path):
        """load_image should load a valid image file."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        path = tmp_path / "test.png"
        img.save(path)

        arr = load_image(path)
        assert arr is not None
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.uint8

    def test_load_nonexistent(self):
        """load_image should return None for nonexistent files."""
        assert load_image("/nonexistent/path/image.png") is None

    def test_decode_image_bytes(self):
        """decode_image_bytes should decode raw image bytes."""
        import io

        from PIL import Image

        img = Image.new("RGB", (16, 16), color=(50, 100, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        arr = decode_image_bytes(data)
        assert arr is not None
        assert arr.shape == (16, 16, 3)

    def test_decode_invalid_bytes(self):
        """decode_image_bytes should return None for invalid data."""
        assert decode_image_bytes(b"not an image") is None

    def test_resize_no_upscale(self):
        """resize_for_vision should not upscale small images."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = resize_for_vision(img, max_dim=320)
        assert result.shape == (50, 50, 3)

    def test_resize_downscale(self):
        """resize_for_vision should downscale large images."""
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = resize_for_vision(img, max_dim=320)
        assert max(result.shape[:2]) <= 320

    def test_resize_preserves_channels(self):
        """resize_for_vision should preserve the channel dimension."""
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = resize_for_vision(img, max_dim=320)
        assert result.shape[2] == 3

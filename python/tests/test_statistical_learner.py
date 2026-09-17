"""Tests for the statistical language learner.

Covers ``NGramModel`` counts/probability/prefix-index behaviour,
``StatisticalLanguageLearner`` style profiling, next-word prediction,
sequence fluency, and the integration of these signals into vocabulary
slot selection and generator realization scoring.
"""

import pytest

from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language.statistical_learner import (
    NGramModel,
    StatisticalLanguageLearner,
)
from genesis_cognitive.language.vocabulary import Vocabulary
from genesis_cognitive.self import PersonalityTraits, SelfModel

# ─── NGramModel ───────────────────────────────────────────────────


def test_ngram_observe_records_counts_and_context():
    model = NGramModel(n=2)
    model.observe(["the", "cat", "sat"])
    assert model.counts[("the", "cat")] == 1
    assert model.counts[("cat", "sat")] == 1
    assert model.context_counts[("the",)] == 1
    assert model.context_counts[("cat",)] == 1


def test_ngram_observe_too_short_is_noop():
    model = NGramModel(n=3)
    model.observe(["only", "two"])
    assert model.counts == {}
    assert model.context_counts == {}


def test_ngram_observe_accumulates_repeats():
    model = NGramModel(n=2)
    model.observe(["a", "b"])
    model.observe(["a", "b"])
    model.observe(["a", "c"])
    assert model.counts[("a", "b")] == 2
    assert model.counts[("a", "c")] == 1
    assert model.context_counts[("a",)] == 3


def test_ngram_probability_is_smoothed_and_bounded():
    model = NGramModel(n=2)
    model.observe(["a", "b"])
    model.observe(["a", "b"])
    # P(b|a) = (2 + 1) / (2 + V), V = len(counts) = 1
    assert model.probability(("a", "b")) == pytest.approx(3.0 / 3.0)
    # Unseen completion still gets smoothing mass.
    assert 0.0 < model.probability(("a", "z")) < 1.0
    # Probabilities are in (0, 1].
    assert model.probability(("a", "b")) <= 1.0


def test_ngram_prefix_index_matches_counts():
    model = NGramModel(n=3)
    model.observe(["a", "b", "c"])
    model.observe(["a", "b", "c"])
    model.observe(["a", "b", "d"])
    bucket = model.completions_for(("a", "b"))
    assert bucket == {"c": 2, "d": 1}
    # Unknown prefix yields an empty mapping (no KeyError).
    assert model.completions_for(("x", "y")) == {}


def test_ngram_most_common_returns_sorted_pairs():
    model = NGramModel(n=2)
    model.observe(["a", "b"])
    model.observe(["a", "b"])
    model.observe(["c", "d"])
    top = model.most_common(2)
    assert top[0] == (("a", "b"), 2)
    assert top[1] == (("c", "d"), 1)


# ─── StatisticalLanguageLearner ───────────────────────────────────


def test_learn_from_input_updates_token_and_sentence_counts():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("Hello world. Foo bar baz!")
    assert learner.total_inputs == 2
    assert learner.vocabulary_size == 5


def test_learn_from_input_ignores_empty():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("   ")
    learner.learn_from_input("")
    assert learner.total_inputs == 0
    assert learner.vocabulary_size == 0


def test_word_frequency_is_case_insensitive():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("Hello hello HELLO")
    assert learner.word_frequency("hello") == 3
    assert learner.word_frequency("Hello") == 3
    assert learner.word_frequency("missing") == 0


def test_bigram_probability_uses_lowercased_model():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat the cat ran")
    # "the" is always followed by "cat" (twice); "ran" follows "cat", not "the".
    p_cat = learner.bigram_probability("The", "Cat")
    p_cat_lower = learner.bigram_probability("the", "cat")
    assert p_cat == pytest.approx(p_cat_lower)
    assert p_cat > 0.0
    # An unseen completion after "the" still gets smoothing mass.
    p_unseen = learner.bigram_probability("the", "ran")
    assert 0.0 < p_unseen < p_cat


def test_suggest_next_word_picks_most_frequent_completion():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat")
    learner.learn_from_input("the cat ran")
    learner.learn_from_input("the cat sat")
    # "the cat" -> "sat" (2) beats "ran" (1).
    assert learner.suggest_next_word(["the", "cat"]) == "sat"


def test_suggest_next_word_falls_back_to_bigram():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat")
    learner.learn_from_input("the dog")
    learner.learn_from_input("the dog")
    # No trigram context, but bigram "the" -> "dog" (2) beats "cat" (1).
    assert learner.suggest_next_word(["the"]) == "dog"


def test_suggest_next_word_returns_none_without_data():
    learner = StatisticalLanguageLearner()
    assert learner.suggest_next_word(["anything"]) is None


def test_suggest_next_word_returns_none_for_unseen_prefix():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat")
    assert learner.suggest_next_word(["never", "seen"]) is None


def test_sequence_fluency_neutral_without_evidence():
    learner = StatisticalLanguageLearner()
    assert learner.sequence_fluency("anything at all here") == 0.5


def test_sequence_fluency_neutral_for_unseen_transitions():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("alpha beta gamma")
    # Tokens the model has never observed in this arrangement get neutral.
    assert learner.sequence_fluency("zzz qqq www") == 0.5


def test_sequence_fluency_higher_for_learned_text():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat the cat sat the cat sat")
    learned = learner.sequence_fluency("the cat sat")
    novel = learner.sequence_fluency("zzz qqq www")
    assert learned > 0.5
    assert novel == 0.5
    assert learned > novel


def test_user_style_profile_neutral_when_no_markers():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("hello world foo bar")
    profile = learner.user_style_profile()
    assert profile["style"] == "neutral"
    assert profile["formal_score"] == 0.0
    assert profile["casual_score"] == 0.0
    assert profile["technical_score"] == 0.0
    assert profile["total_sentences"] == 1
    assert profile["vocabulary_size"] == 4


def test_user_style_profile_detects_formal():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("Therefore, I would request you to assist.")
    learner.learn_from_input("Furthermore, please provide the details.")
    profile = learner.user_style_profile()
    assert profile["style"] == "formal"
    assert profile["formal_score"] > 0.4
    assert profile["formal_score"] > profile["casual_score"]


def test_user_style_profile_detects_casual():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("yeah that's cool, gonna do it")
    learner.learn_from_input("hey guys, ok lol")
    profile = learner.user_style_profile()
    assert profile["style"] == "casual"
    assert profile["casual_score"] > 0.4


def test_user_style_profile_detects_technical():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the function takes a parameter and returns")
    learner.learn_from_input("refactor the module interface at runtime")
    profile = learner.user_style_profile()
    assert profile["style"] == "technical"
    assert profile["technical_score"] > 0.4


def test_user_style_profile_top_ngrams_present():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat the cat ran")
    profile = learner.user_style_profile()
    assert profile["top_bigrams"]
    assert isinstance(profile["top_bigrams"][0][0], list)
    assert profile["top_trigrams"] is not None


def test_total_tokens_property():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("hello world foo bar baz")
    assert learner.total_tokens == 5
    assert learner.total_inputs == 1


def test_total_tokens_zero_when_empty():
    learner = StatisticalLanguageLearner()
    assert learner.total_tokens == 0


def test_trigram_probability_smoothed():
    learner = StatisticalLanguageLearner()
    learner.learn_from_input("the cat sat on the mat")
    # P(sat | the, cat) — observed once
    p = learner.trigram_probability("the", "cat", "sat")
    assert 0.0 < p <= 1.0
    # Unseen trigram still gets smoothing mass
    p_unseen = learner.trigram_probability("the", "cat", "zzz")
    assert 0.0 < p_unseen < p


def test_suggest_next_word_uses_trigram_over_bigram():
    """When trigram data is available, it should take priority."""
    learner = StatisticalLanguageLearner()
    # "the cat" is followed by "sat" 3 times and "ran" 1 time
    learner.learn_from_input("the cat sat the cat sat the cat sat the cat ran")
    assert learner.suggest_next_word(["the", "cat"]) == "sat"


def test_word_frequency_returns_zero_for_unseen():
    learner = StatisticalLanguageLearner()
    assert learner.word_frequency("nonexistent") == 0


# ─── Vocabulary integration (fluency shaping) ────────────────────


def _make_emotion(
    creativity: float = 0.5,
    caution: float = 0.3,
    openness_to_engage: float = 0.6,
) -> EmotionalState:
    """Neutral emotion for vocabulary tests."""
    return EmotionalState(
        label="neutral",
        cognitive_style="analytical",
        creativity=creativity,
        caution=caution,
        openness_to_engage=openness_to_engage,
    )


def _make_personality() -> PersonalityTraits:
    return PersonalityTraits(
        openness=0.5,
        conscientiousness=0.5,
        extraversion=0.5,
        agreeableness=0.5,
        neuroticism=0.5,
    )


def test_vocabulary_word_frequency_biases_selection():
    """Frequently-observed user words are preferred in slot selection."""
    learner = StatisticalLanguageLearner()
    # "cat" appears 20 times, "dog" appears 1 time
    for _ in range(20):
        learner.learn_from_input("cat")
    learner.learn_from_input("dog")

    vocab = Vocabulary(seed=42)
    vocab.set_fluency_model(learner)
    emotion = _make_emotion()
    personality = _make_personality()

    picks: dict[str, int] = {}
    for _ in range(200):
        word = vocab._select_word(
            ["cat", "dog"], emotion, personality
            # no prev_word → only unigram bias applies
        )
        picks[word] = picks.get(word, 0) + 1

    assert picks.get("cat", 0) > picks.get("dog", 0)


def test_vocabulary_trigram_biases_selection():
    """Trigram context shapes slot selection more than bigram alone."""
    learner = StatisticalLanguageLearner()
    # "the cat sat" observed many times; "the cat ran" once
    for _ in range(10):
        learner.learn_from_input("the cat sat")
    learner.learn_from_input("the cat ran")

    vocab = Vocabulary(seed=42)
    vocab.set_fluency_model(learner)
    emotion = _make_emotion()
    personality = _make_personality()

    picks: dict[str, int] = {}
    for _ in range(200):
        word = vocab._select_word(
            ["ran", "sat"],
            emotion,
            personality,
            prev_word="cat",
            prev_prev_word="the",
        )
        picks[word] = picks.get(word, 0) + 1

    assert picks.get("sat", 0) > picks.get("ran", 0)


def test_vocabulary_bigram_fallback_without_prev_prev():
    """With only one previous word, bigram (not trigram) shapes selection."""
    learner = StatisticalLanguageLearner()
    # "cat sat" observed many times
    for _ in range(10):
        learner.learn_from_input("cat sat")
    learner.learn_from_input("cat ran")

    vocab = Vocabulary(seed=42)
    vocab.set_fluency_model(learner)
    emotion = _make_emotion()
    personality = _make_personality()

    picks: dict[str, int] = {}
    for _ in range(200):
        word = vocab._select_word(
            ["ran", "sat"],
            emotion,
            personality,
            prev_word="cat",
            # no prev_prev_word → bigram fallback
        )
        picks[word] = picks.get(word, 0) + 1

    assert picks.get("sat", 0) > picks.get("ran", 0)


def test_vocabulary_no_fluency_model_works():
    """Without a fluency model, selection still works (uniform-ish)."""
    vocab = Vocabulary(seed=42)
    emotion = _make_emotion()
    personality = _make_personality()
    word = vocab._select_word(["hello", "world"], emotion, personality)
    assert word in ("hello", "world")


# ─── Generator integration (prediction alignment) ────────────────


def test_prediction_alignment_neutral_without_data():
    """Without learned data, prediction alignment is neutral (0.5)."""
    from genesis_cognitive.language.generator import GenerativeEngine

    engine = GenerativeEngine(SelfModel(born_at=0), seed=42)
    assert engine._prediction_alignment("anything goes here") == 0.5


def test_prediction_alignment_higher_for_predicted_text():
    """Text following the learner's predictions scores higher."""
    from genesis_cognitive.language.generator import GenerativeEngine

    engine = GenerativeEngine(SelfModel(born_at=0), seed=42)
    # Teach a strong pattern: "the cat sat" repeated many times
    for _ in range(20):
        engine.acquire_from_input("the cat sat")
    # "the cat sat" follows the predicted path
    aligned = engine._prediction_alignment("the cat sat")
    # Random text doesn't follow predictions
    random = engine._prediction_alignment("zzz qqq www")
    assert aligned > random


def test_prediction_alignment_neutral_for_short_text():
    """Very short text gets neutral alignment."""
    from genesis_cognitive.language.generator import GenerativeEngine

    engine = GenerativeEngine(SelfModel(born_at=0), seed=42)
    engine.acquire_from_input("the cat sat on the mat")
    assert engine._prediction_alignment("x") == 0.5

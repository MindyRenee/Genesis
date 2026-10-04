"""Pytest configuration for Genesis tests.

Adds the parent directory (``python/``) to ``sys.path`` so that
``genesis_conscious`` and ``genesis_client`` are importable without
each test file having to manipulate ``sys.path`` itself. This keeps
test imports at the top of the file (no E402 violations) while
preserving the same import behavior.
"""

import os
import sys

import pytest

# Insert the python/ directory (parent of tests/) so that both
# genesis_conscious and genesis_client are importable. This replaces
# the per-file ``sys.path.insert(0, ...)`` calls that were scattered
# across the test suite and triggered E402 (module-level import not
# at top of file) lint violations.
_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

# Some tests also need the scripts/ directory on the path (e.g.
# test_concept_network imports from prune_dead_concepts.py). The
# scripts/ directory lives at the project root (one level above
# python/), not under python/.
_PROJECT_ROOT = os.path.dirname(_PYTHON_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if os.path.isdir(_SCRIPTS_DIR) and _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Genesis has no seed sentiment lexicon: sentiment is learned through
# interaction (``sentiment.learn_sentiment``), not shipped as a table.
# Tests of the sentiment *logic* — negation, intensifiers, dimishers,
# contrastive conjunction, accumulation — still need words of known
# valence, so they teach the words first through the same public API a
# real interaction would use. Neutral-probe words ("brown", "blue",
# "table") are deliberately absent so "unknown stays unknown" remains
# testable.
_POSITIVE_TESTS = {
    "good": 2.0, "great": 3.0, "wonderful": 3.0, "excellent": 3.0,
    "amazing": 3.0, "awesome": 3.0, "love": 3.0, "like": 1.5,
    "happy": 2.7, "glad": 2.3, "kind": 1.8, "better": 2.0,
    "best": 3.0, "content": 1.6, "thankful": 2.3, "cheerful": 2.2,
    "satisfied": 2.1, "confident": 2.0, "optimistic": 2.2,
    "serene": 2.2, "joy": 2.8, "joyful": 2.8, "helpful": 2.0,
    "nice": 1.8, "fine": 1.0, "calm": 1.3, "hope": 1.8,
    "proud": 2.5, "excited": 2.8, "relieved": 2.0, "grateful": 2.5,
}

_NEGATIVE_TESTS = {
    "bad": -2.0, "terrible": -3.0, "awful": -2.8, "horrible": -3.0,
    "hate": -3.2, "wrong": -2.0, "broken": -2.3, "stupid": -2.5,
    "stress": -1.6, "stressed": -2.2, "stressful": -2.2,
    "anxious": -2.3, "worry": -1.8, "upset": -2.1, "lonely": -2.2,
    "afraid": -2.0, "nervous": -2.0, "miserable": -3.0,
    "hopeless": -2.8, "overwhelmed": -2.3, "exhausted": -2.3,
    "frustrated": -2.5, "sad": -2.3, "unhappy": -2.3, "angry": -2.5,
    "slow": -1.3, "confusing": -1.8,
    "confused": -1.8, "pain": -2.0, "hurt": -2.0, "ugly": -2.0,
    "disappointing": -2.5, "tired": -1.5, "boring": -2.0,
    "worse": -2.0, "worst": -3.0, "problem": -1.8, "distress": -2.5,
}


@pytest.fixture
def learned_sentiment():
    """Teach a test vocabulary of sentiment words, then clean up.

    Yields the ``learn_sentiment`` callable so a test can teach
    additional words of its own. The learned lexicon is module-global,
    so it is cleared on teardown to keep tests independent.
    """
    from genesis_conscious.language.sentiment import (
        LEARNED_LEXICON,
        learn_sentiment,
    )

    for word, score in {**_POSITIVE_TESTS, **_NEGATIVE_TESTS}.items():
        learn_sentiment(word, score)
    try:
        yield learn_sentiment
    finally:
        LEARNED_LEXICON.clear()


# Likewise the verb inventory is learned (``morphology.register_learned_verb``)
# rather than seeded. Comprehension uses verb knowledge to decide whether
# an inflected word is a predicate, so tests that assert on parsed
# predicates must first teach the verbs they exercise.
_TEST_VERBS = (
    # Auxiliaries only. Copulas ("be", "is", ...) are deliberately NOT
    # taught here: the parser already carries them in _COPULAR_VERBS, and
    # registering "be" as a learned verb makes "reach 12 ..." parse as
    # subject="reach 12" instead of an imperative, which breaks the
    # quantity-intake tests.
    "do", "does", "did", "done", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    # relation verbs
    "cause", "enable", "lead", "create", "produce", "harm", "prevent",
    "stop", "contradict", "depend", "require",
    "emerge", "relate", "connect", "resemble", "contain", "include",
    "accept", "fit", "support", "affect", "form", "define", "belong",
    # common action verbs
    "say", "tell", "know", "see", "look", "think", "feel", "go", "come",
    "make", "take", "give", "find", "want", "use", "try", "help", "turn",
    "move", "hold", "bring", "keep", "let", "get",
    "become", "leave", "return", "start", "work", "play", "live",
    "believe", "learn", "remember", "understand", "choose", "open",
    "close", "read", "write", "speak", "talk", "hear", "listen", "ask",
    "answer", "call", "walk", "run", "eat", "drink", "sleep", "wake", "dream",
    "put", "bite", "reach", "sort", "count", "like", "love", "hate",
    "change", "grow", "build", "make",
)


@pytest.fixture
def learned_verbs():
    """Teach a test vocabulary of verb base forms, then clean up."""
    from genesis_conscious.language.morphology import (
        _LEARNED_VERBS,
        register_learned_verb,
    )

    for base in _TEST_VERBS:
        register_learned_verb(base)
    try:
        yield register_learned_verb
    finally:
        _LEARNED_VERBS.clear()

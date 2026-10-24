"""Regression tests for infrastructure.text — the canonical
whitespace/invisible/word-boundary definitions.

Each test pins a defect class that was verified empirically:

- ``str.strip()``-based blank tests passed zero-width-space input
  through as "speech" (presence without content).
- ASCII-only word regexes fragmented curly-apostrophe contractions
  ("don’t" → "don","t"), which silently dropped the negation token
  and flipped sentiment valence.
- Regex ``\\s`` misses FS/GS/RS/US and every Cf character, so
  whitespace-evasion strings like "sudo\\x1frm" and bidi-marked
  commands bypassed normalizers that split on ``\\s+``.
"""

import pytest

from infrastructure.text import (
    canon_text,
    collapse_ws,
    is_blank,
    sentences,
    split_tokens,
    words,
)


class TestIsBlank:
    """The silence test: "was anything perceivable said?" """

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   \t\n  ",
            " ",  # NBSP — whitespace, not a letter
            "     　",  # the rest of White_Space
            "\x1c\x1d\x1e\x1f",  # FS/GS/RS/US — isspace() but not \s
        ],
    )
    def test_whitespace_only_is_blank(self, text: str) -> None:
        assert is_blank(text)

    @pytest.mark.parametrize(
        "text",
        [
            "​",  # ZWSP — the original defect: looked like content
            "​" * 40,
            "‍‌",  # ZWJ / ZWNJ
            "‏‎‮",  # bidi marks + override
            "⁠",  # word joiner
            "﻿",  # BOM / ZWNBSP
            "­",  # soft hyphen
            " ​ ​ ​ ",  # whitespace padded with invisibles
        ],
    )
    def test_invisible_only_is_blank(self, text: str) -> None:
        # str.strip() says these are non-empty; the canonical test says
        # nothing perceivable was sent.
        assert text.strip() != "" or text == ""
        assert is_blank(text)

    def test_none_is_blank(self) -> None:
        assert is_blank(None)

    @pytest.mark.parametrize("text", ["hi", " a ", "​x​", "."])
    def test_real_content_is_not_blank(self, text: str) -> None:
        assert not is_blank(text)


class TestCanonText:
    def test_fullwidth_folds_to_ascii(self) -> None:
        assert canon_text("ｈｅｌｌｏ　ｗｏｒｌｄ") == "hello world"

    def test_ligature_folds(self) -> None:
        assert canon_text("ﬁle") == "file"

    def test_bidi_marks_removed(self) -> None:
        assert canon_text("rm‮ ⁠-rf") == "rm -rf"

    def test_zwsp_does_not_glue_words(self) -> None:
        # "su​do" is one glyph-cluster to the eye but two tokens after
        # the invisible layer is stripped.
        assert canon_text("su​do") == "sudo"

    def test_apostrophe_variants_fold(self) -> None:
        assert canon_text("don’t") == "don't"
        assert canon_text("don‘t") == "don't"
        assert canon_text("donʼt") == "don't"
        assert canon_text("ｄｏｎ’ｔ") == "don't"

    def test_empty(self) -> None:
        assert canon_text("") == ""


class TestCollapseWs:
    def test_runs_fold_to_single_space(self) -> None:
        assert collapse_ws("a  \t\n b") == "a b"

    def test_separator_controls_collapse(self) -> None:
        # FS/GS/RS/US are isspace() but not regex \s — a \s+ splitter
        # would leave them embedded and let "sudo\x1frm" evade.
        assert collapse_ws("sudo\x1frm\x1d-rf") == "sudo rm -rf"

    def test_invisible_glue_removed(self) -> None:
        assert collapse_ws("su​do ​rm -rf") == "sudo rm -rf"

    def test_ends_trimmed(self) -> None:
        assert collapse_ws("  hello  ") == "hello"


class TestSplitTokens:
    def test_separator_controls_split(self) -> None:
        assert split_tokens("a\x1cb\x1ec") == ["a", "b", "c"]

    def test_invisible_only_yields_nothing(self) -> None:
        assert split_tokens("​ ​ ​") == []

    def test_matches_split_semantics(self) -> None:
        text = "the quick  brown\tfox\njumps"
        assert split_tokens(text) == text.split()


class TestWords:
    def test_curly_contraction_stays_whole(self) -> None:
        # The verified valence-flip: "don’t" used to become "don","t".
        assert words("i don’t hate this") == ["i", "don't", "hate", "this"]

    def test_ascii_contraction_stays_whole(self) -> None:
        assert words("i don't hate this") == ["i", "don't", "hate", "this"]

    def test_unicode_letters_are_one_token(self) -> None:
        assert words("gérard") == ["gérard"]

    def test_underscore_is_a_separator_not_a_word(self) -> None:
        assert words("foo_bar") == ["foo", "bar"]

    def test_fullwidth_input_tokenizes(self) -> None:
        assert words("ｈｅｌｌｏ ｗｏｒｌｄ") == ["hello", "world"]

    def test_invisible_only_yields_nothing(self) -> None:
        assert words("​") == []


class TestSentences:
    def test_abbreviation_is_not_a_boundary(self) -> None:
        result = sentences("Dr. Smith went home. Really?")
        # Punkt resolves the abbreviation; a whitespace lookbehind
        # would produce three fragments.
        assert result == ["Dr. Smith went home.", "Really?"]

    def test_basic_split(self) -> None:
        result = sentences("One. Two! Three?")
        assert len(result) == 3

    def test_blank_yields_empty(self) -> None:
        assert sentences("") == []
        assert sentences("​") == []

    def test_fallback_without_punkt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate NLTK/punkt being absent — the regex fallback must
        # still segment on punctuation+whitespace.
        import sys

        import infrastructure.text as text_mod

        monkeypatch.setitem(sys.modules, "nltk", None)
        monkeypatch.setitem(sys.modules, "nltk.tokenize", None)
        result = text_mod.sentences("One. Two.")
        assert result == ["One.", "Two."]


class TestSentimentIntegration:
    """The end-to-end consequence: curly-quoted negation must reach
    the sentiment scorer as the same token as ASCII-quoted."""

    def test_curly_negation_scores_like_ascii(self) -> None:
        from limbic_system.sentiment import _tokenize

        ascii_tokens = _tokenize("i don't hate this")
        curly_tokens = _tokenize("i don’t hate this")
        assert ascii_tokens == curly_tokens
        assert "don't" in curly_tokens

    def test_negation_token_survives_in_pipeline(self) -> None:
        from limbic_system.sentiment import _normalize, _tokenize

        normalized = [_normalize(t) for t in _tokenize("i don’t hate this")]
        # "dont" is the NEGATIONS-set form; it must be present for the
        # scorer to flip "hate" instead of scoring it raw.
        assert "dont" in normalized

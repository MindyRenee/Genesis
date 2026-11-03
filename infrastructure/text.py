"""Canonical text normalization — one definition of "whitespace" and
one answer to "was anything said".

Two formal layers, kept separate because they answer different
questions:

- **Whitespace** is any character ``str.isspace()`` accepts: the 25
  Unicode ``White_Space`` codepoints plus the four ASCII separator
  controls FS/GS/RS/US (U+001C–U+001F), which ``str.split()`` honours
  even though regex ``\\s`` does not. ``text.split()`` already
  implements this widest notion; the helpers below apply it after the
  invisible layer is removed, so nothing can glue words together
  invisibly.
- **Invisible characters** are Unicode category ``Cf`` (format:
  zero-width space and joiners, bidi marks and overrides, word
  joiner, BOM, soft hyphen, tag characters) — plus ``Cc`` control
  characters that produce no glyph. They are not whitespace, so
  ``strip()`` and ``split()`` pass them straight through, which
  defeated every downstream test that assumed "non-blank = content":
  a message of pure zero-width spaces looked like speech.

``canon_text`` applies NFKC (compatibility equivalence — fullwidth
forms, ligatures) and drops the invisible layer. It is for analysis
only — a one-way normalization — never for text that will be
re-emitted to the user.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "canon_text",
    "collapse_ws",
    "is_blank",
    "sentences",
    "split_tokens",
    "words",
]

# Categories with no perceivable glyph: format controls (Cf — ZWSP,
# joiners, bidi marks/overrides, word joiner, BOM, soft hyphen) and
# unprintable controls (Cc). Whitespace is tested separately.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Cc"})

# Typographic apostrophe forms → ASCII. NFKC folds the fullwidth
# variant already; these are the ones it leaves alone. U+2032 (prime)
# folds to ' as well — in language text it is indistinguishable from
# an apostrophe and never carries separate lexical meaning here.
_APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'", "\u2032": "'"})

# Lexical word: Unicode letters, digits and combining marks (not
# underscore — it is a programming separator, not a word character),
# joined by apostrophes so contractions survive as single tokens.
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")

# Fallback sentence boundary when NLTK's Punkt model is unavailable.
# Protect common abbreviations before splitting so "Dr. Smith" remains
# one sentence fragment. This is deliberately a small deterministic
# fallback, not a replacement for a trained sentence segmenter.
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_ABBREVIATIONS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|"
    r"Fig|No|Inc|Ltd|Co|approx|Dept|Univ)\.",
    re.IGNORECASE,
)
_SENTENCE_INITIAL_RE = re.compile(r"\b([A-Z])\.")


def canon_text(text: str) -> str:
    """One-way canonical form for analysis.

    NFKC folding (fullwidth forms, ligatures become their letters),
    then invisible format/control characters removed, then
    typographic apostrophes folded to ``'`` so "don’t" and "don't"
    are the same five characters downstream.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    kept = [
        c for c in normalized if c.isspace() or unicodedata.category(c) not in _INVISIBLE_CATEGORIES
    ]
    return "".join(kept).translate(_APOSTROPHES)


def is_blank(text: str | None) -> bool:
    """True iff the text carries nothing perceivable — the silence test.

    ``""``, ``None``, whitespace-only input, and input made entirely of
    invisible format/control characters (a zero-width-space message)
    are all blank.
    """
    if not text:
        return True
    return all(c.isspace() or unicodedata.category(c) in _INVISIBLE_CATEGORIES for c in text)


def collapse_ws(text: str) -> str:
    """Canonical single-spaced form: invisibles removed, every
    whitespace run folded to one space, ends trimmed."""
    return " ".join(canon_text(text).split())


def split_tokens(text: str) -> list[str]:
    """Whitespace tokens over the canonical text — what
    ``text.split()`` would produce once invisible characters can no
    longer glue or fake tokens."""
    return canon_text(text).split()


def words(text: str) -> list[str]:
    """Lexical word tokens: Unicode letter/digit/mark runs joined by
    apostrophes, with quote variants already folded — "don’t" arrives
    as the single token "don't" instead of the pair "don","t"."""
    return _WORD_RE.findall(canon_text(text))


def sentences(text: str) -> list[str]:
    """Sentence boundaries.

    Uses NLTK's punkt segmenter (unsupervised sentence-boundary
    detection, Kiss & Strunk 2006) when it is installed: it resolves
    abbreviations ("Dr.", "e.g."), ellipsis and quoted speech that a
    whitespace lookbehind mangles. When the model is absent the
    punctuation-followed-by-whitespace rule is the documented
    fallback — less accurate, never absent.
    """
    text = canon_text(text).strip()
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except (ImportError, LookupError):
        # Replace abbreviation periods with a private-use marker while
        # splitting, then restore them. A private-use code point avoids
        # changing normal visible text or confusing punctuation handling.
        marker = "\\ue000"
        protected = _SENTENCE_ABBREVIATIONS.sub(
            lambda match: match.group(0)[:-1] + marker,
            text,
        )
        protected = _SENTENCE_INITIAL_RE.sub(
            lambda match: match.group(1) + marker,
            protected,
        )
        parts = _SENTENCE_FALLBACK_RE.split(protected)
        return [part.replace(marker, ".") for part in parts if part]

"""Lexicon-based sentiment analysis with negation, intensifier, and
diminisher handling.

This is a VADER-style (Gilbert & Hutto, 2014) sentiment analyzer — no
external dependencies, no ML model, no network calls. It runs in
microseconds and gives the dyadic affective model a much richer signal
than the previous word-list heuristic.

## What it handles that the old heuristic didn't

- **Negation**: "not bad" → positive, "don't like" → negative,
  "never good" → negative. The old system scored "not bad" as negative
  because it matched "bad".
- **Intensifiers**: "very good" > "good", "extremely terrible" < "terrible".
- **Diminishers**: "slightly good" < "good", "kind of bad" < "bad".
- **Contrastive conjunctions**: "good but slow" shifts weight to "slow".
- **Punctuation boost**: "great!!!" > "great".
- **Capitalization boost**: "GREAT" > "great".
- **Multi-word modifier phrases**: "kind of bad" < "bad", "sort of good" <
  "good", "a lot better" > "better". Phrase-level modifiers are matched
  before per-token scoring so their constituent words are not scored
  individually (see ``MODIFIER_PHRASES``).
- **Multi-word negation**: "doesn't work", "not very good".

## Limitations (documented honestly)

- No sarcasm detection ("Oh great, another bug" scores positive).
- No context beyond the message itself.
- No aspect-based sentiment ("the UI is great but the backend is slow"
  is scored as a single valence, not per-aspect).
- The lexicon is curated, not learned — it covers common English
  sentiment words but will miss domain-specific jargon.

## Reference

Gilbert, C. H. E., & Hutto, E. E. (2014). VADER: A Parsimonious Rule-based
Model for Sentiment Analysis of Social Media Text. Proceedings of the
8th International Conference on Weblogs and Social Media (ICWSM).
"""

from __future__ import annotations

import re

# ─── Lexicon ──────────────────────────────────────────────────────
# Each word maps to a sentiment score in [-4, +4].
# -4 = extremely negative, 0 = neutral, +4 = extremely positive.
# Curated from VADER's lexicon (Gilbert & Hutto, 2014), trimmed to
# the most common sentiment words (~250 entries).

LEXICON: dict[str, float] = {
    # ── Positive ──
    "good": 1.9, "great": 3.1, "awesome": 3.3, "amazing": 3.2,
    "excellent": 3.4, "perfect": 3.3, "wonderful": 3.0, "fantastic": 3.3,
    "brilliant": 3.1, "love": 3.2, "loved": 3.2, "loving": 2.8,
    "like": 1.5, "liked": 1.5, "likes": 1.2,
    "happy": 2.7, "glad": 2.3, "pleased": 2.1, "delighted": 3.0,
    "thanks": 2.2, "thank": 2.0, "grateful": 2.5, "appreciate": 2.3,
    "appreciated": 2.3, "helpful": 2.0, "cool": 1.5, "nice": 1.8,
    "beautiful": 2.7, "pretty": 1.5, "superb": 3.4, "outstanding": 3.3,
    "remarkable": 2.8, "impressive": 2.6, "enjoy": 2.4, "enjoyed": 2.4,
    "enjoying": 2.2, "fun": 2.0, "funny": 2.0, "hilarious": 2.8,
    "yes": 1.0, "yep": 1.0, "yeah": 1.0, "sure": 1.2, "okay": 0.8,
    "ok": 0.8, "fine": 1.0, "well": 1.2, "right": 1.0,
    "best": 3.0, "better": 2.0, "win": 2.5, "winning": 2.5, "won": 2.5,
    "success": 2.5, "successful": 2.7, "worked": 1.8, "works": 1.5,
    "work": 1.5, "working": 1.5,
    "help": 1.8, "helped": 1.8, "helping": 1.5,
    "improve": 1.8, "improved": 2.0, "improving": 1.8,
    "amaze": 2.8, "amazed": 3.0,
    "easy": 1.5, "clear": 1.3, "smooth": 1.8, "fast": 1.3,
    "warm": 1.5, "kind": 1.8, "caring": 2.0, "sweet": 2.0,
    "calm": 1.3, "peaceful": 2.0, "relaxed": 1.5, "comfortable": 1.8,
    "hope": 1.8, "hopeful": 1.8, "inspired": 2.5, "motivated": 2.3,
    "proud": 2.5, "excited": 2.8, "thrilled": 3.2, "wow": 2.5,
    "yay": 2.8, "hooray": 3.0,
    "content": 1.6, "cheerful": 2.2, "joy": 2.8, "joyful": 2.8,
    "thankful": 2.3, "satisfied": 2.1, "satisfying": 2.1,
    "confident": 2.0, "determined": 1.8, "encouraged": 2.0,
    "optimistic": 2.2, "serene": 2.2, "cosy": 1.6, "cozy": 1.6,
    "terrific": 2.8, "marvelous": 3.0,

    # ── Negative ──
    "bad": -1.9, "wrong": -2.0, "error": -2.0, "errors": -2.0,
    "broken": -2.3, "break": -1.5, "broke": -2.0,
    "hate": -3.2, "hated": -3.2, "hating": -2.8,
    "stupid": -2.5, "dumb": -2.0, "idiotic": -2.8,
    "no": -1.0, "nope": -1.2, "never": -1.5,
    "stop": -1.5, "fail": -2.3, "failed": -2.5, "failure": -2.7,
    "failing": -2.3, "bug": -2.0, "bugs": -2.0, "buggy": -2.3,
    "crash": -2.5, "crashed": -2.7, "crashes": -2.5,
    "annoying": -2.2, "annoyed": -2.2, "frustrating": -2.5,
    "frustrated": -2.5, "terrible": -3.0, "awful": -2.8,
    "horrible": -3.0, "horrendous": -3.2, "dreadful": -2.8,
    "confusing": -1.8, "confused": -1.8, "confusion": -1.5,
    "difficult": -1.5, "hard": -1.0, "tough": -1.0,
    "slow": -1.3, "sluggish": -1.8, "laggy": -1.8,
    "ugly": -2.0, "messy": -1.5, "chaotic": -1.8,
    "pain": -2.0, "painful": -2.3, "hurt": -2.0, "hurts": -2.0,
    "sad": -2.3, "unhappy": -2.3, "depressed": -2.8,
    "angry": -2.5, "mad": -2.0, "furious": -3.0,
    "worried": -2.0, "anxious": -2.3, "scared": -2.3,
    "tired": -1.5, "exhausted": -2.3, "overwhelmed": -2.3,
    "boring": -2.0, "bored": -1.8, "dull": -1.5,
    "disappointing": -2.5, "disappointed": -2.5,
    "useless": -2.7, "worthless": -2.7, "pointless": -2.3,
    "worse": -2.0, "worst": -3.0, "lose": -2.0, "losing": -2.0,
    "lost": -1.8, "stuck": -1.8, "blocked": -1.8,
    "destroy": -2.5, "destroyed": -2.7,
    "dislike": -2.0, "disliked": -2.0,
    "problem": -1.8, "problems": -1.8, "issue": -1.5, "issues": -1.5,
    "sorry": -1.0, "regret": -1.8, "unlucky": -1.8,
    "sucks": -2.5, "sucked": -2.5, "suck": -2.3,
    "damn": -1.5, "crap": -2.0, "garbage": -2.3,
    "trash": -2.3, "rubbish": -2.0,
    "stress": -1.6, "stressed": -2.2, "stressful": -2.0, "stressing": -1.8,
    "anxiety": -2.3, "worry": -1.8, "worries": -1.8,
    "upset": -2.1, "lonely": -2.2, "afraid": -2.0, "nervous": -1.8,
    "miserable": -3.0, "hopeless": -2.8, "desperate": -2.6,
    "grief": -2.6, "grieving": -2.6, "crying": -2.2, "cry": -1.8,
    "suffering": -2.6, "struggling": -2.0, "struggle": -1.8,
    "tiring": -1.6, "drained": -1.8, "burnout": -2.4,
    "irritated": -1.9, "rage": -2.8,
    "guilty": -2.1, "ashamed": -2.4, "embarrassed": -1.9,
}

# ─── Negation words ───────────────────────────────────────────────
# These flip the sentiment of the following sentiment-bearing word.
# "not good" → -good, "don't like" → -like, "never great" → -great.

NEGATIONS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nobody", "nothing",
    "neither", "nor", "nowhere", "dont", "cant",
    "cannot", "wont", "wouldnt", "shouldnt", "couldnt",
    "isnt", "wasnt", "arent", "werent", "doesnt", "didnt",
    "hasnt", "havent", "hadnt", "without",
})

# ─── Intensifiers ─────────────────────────────────────────────────
# These amplify the sentiment of the following word.
# "very good" → good * 1.5, "extremely terrible" → terrible * 1.8.
# Values are multipliers.

INTENSIFIERS: dict[str, float] = {
    "very": 1.5, "extremely": 1.8, "really": 1.4, "so": 1.3,
    "too": 1.3, "completely": 1.7, "absolutely": 1.8,
    "totally": 1.7, "utterly": 1.8, "highly": 1.6,
    "deeply": 1.6, "incredibly": 1.7, "remarkably": 1.5,
    "particularly": 1.4, "especially": 1.4, "exceptionally": 1.7,
    "super": 1.5, "truly": 1.4, "terribly": 1.5,
    "awfully": 1.5, "hugely": 1.5,
}

# ─── Diminishers ──────────────────────────────────────────────────
# These reduce the sentiment of the following word.
# "slightly good" → good * 0.5, "kinda bad" → bad * 0.7.
# Note: "kind" is NOT here — it's a positive sentiment word in LEXICON.
# The multi-word phrase "kind of" is handled in MODIFIER_PHRASES below,
# not as a single token.

DIMINISHERS: dict[str, float] = {
    "slightly": 0.5, "somewhat": 0.6, "kinda": 0.7,
    "sort": 0.7, "barely": 0.4, "hardly": 0.4,
    "scarcely": 0.4, "marginally": 0.5, "mildly": 0.6,
    "moderately": 0.7, "fairly": 0.8, "rather": 0.8,
    "bit": 0.7, "little": 0.7, "less": 0.6,
}

# ─── Multi-word modifier phrases ──────────────────────────────────
# Some modifiers are phrases, not single words. They must be recognized
# before per-token scoring, because their constituent words would
# otherwise be scored or discarded incorrectly. The clearest case is
# "kind of": "kind" is a *positive* sentiment word in LEXICON, so
# without phrase recognition "kind of bad" gains positive weight from
# "kind" and loses the diminisher entirely (net ~neutral), and
# "kind of good" scores *more* positive than "good" — the hedge
# inverts the sentiment. Similarly "sort of" loses its diminisher
# because "of" is a non-sentiment word that resets it, and "a lot"
# contributes no intensification at all.
#
# Maps a tuple of normalized words to (kind, multiplier), where kind is
# "intensifier" or "diminisher". Longer phrases are matched first.
MODIFIER_PHRASES: dict[tuple[str, ...], tuple[str, float]] = {
    ("a", "great", "deal"): ("intensifier", 1.5),
    ("kind", "of"): ("diminisher", 0.7),
    ("sort", "of"): ("diminisher", 0.7),
    ("a", "lot"): ("intensifier", 1.5),
    ("a", "bit"): ("diminisher", 0.7),
    ("a", "little"): ("diminisher", 0.7),
    ("a", "tad"): ("diminisher", 0.5),
    ("a", "touch"): ("diminisher", 0.6),
    ("very", "much"): ("intensifier", 1.4),
}

# ─── Contrastive conjunctions ─────────────────────────────────────
# "but" shifts focus to the following clause. We handle this by
# boosting the sentiment of words that come after "but" and reducing
# the weight of words that came before.

CONTRASTIVES: frozenset[str] = frozenset({
    "but", "however", "although", "though", "yet",
    "still", "nevertheless", "nonetheless", "except",
})

# ─── Negation window ──────────────────────────────────────────────
# How many non-sentiment words can appear between a negation and the
# sentiment word it flips. VADER uses 3; we match that. Without this,
# "not the good" loses the negation at "the" and scores positive.

_NEGATION_WINDOW = 3

# ─── Punctuation and capitalization ───────────────────────────────

_EXCLAMATION_BOOST_THRESHOLD = 3  # need this many ! to get a boost


def _punctuation_boost(text: str) -> float:
    """Boost sentiment magnitude based on exclamation marks."""
    excl_count = text.count("!")
    if excl_count >= _EXCLAMATION_BOOST_THRESHOLD:
        return 1.0 + min(0.5, 0.1 * excl_count)
    return 1.0


def _capitalization_boost(word: str) -> float:
    """Boost sentiment if the word is ALL CAPS (and has sentiment)."""
    normalized = word.lower().replace("'", "")
    if len(word) >= 3 and word.isupper() and normalized in LEXICON:
        return 1.5
    return 1.0


# ─── Tokenization ─────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split text into word tokens, preserving original case for the
    capitalization boost. Lowercasing happens later in ``_normalize``."""
    # Match words including contractions (don't, can't, won't)
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    return tokens


def _normalize(token: str) -> str:
    """Normalize a token for lexicon/negation lookup.

    Strips apostrophes from contractions: "don't" → "dont",
    "can't" → "cant". This matches the NEGATIONS set which stores
    contractions without apostrophes.
    """
    return token.lower().replace("'", "")


# ─── Sentiment analysis ───────────────────────────────────────────

def _score_sentiment_word(
    raw_token: str,
    token: str,
    intensifier_mult: float,
    diminisher_mult: float,
    negation_active: bool,
    negation_distance: int,
) -> tuple[float, float, float, bool]:
    """Score a single sentiment-bearing token with context modifiers.

    Returns (score, new_intensifier_mult, new_diminisher_mult,
    new_negation_active).
    """
    base_score = LEXICON[token]

    # Apply capitalization boost
    cap_boost = _capitalization_boost(raw_token)
    base_score *= cap_boost

    # Apply intensifier
    if intensifier_mult > 1.0:
        base_score *= intensifier_mult
        intensifier_mult = 1.0  # reset

    # Apply diminisher
    if diminisher_mult < 1.0:
        base_score *= diminisher_mult
        diminisher_mult = 1.0  # reset

    # Apply negation (flip sign) if within the negation window
    if negation_active and negation_distance <= _NEGATION_WINDOW:
        base_score = -base_score * 0.9  # slightly reduced
        negation_active = False

    # Clamp to [-4, 4]
    base_score = max(-4.0, min(4.0, base_score))

    return base_score, intensifier_mult, diminisher_mult, negation_active


def _match_modifier_phrase(
    tokens: list[str], index: int,
) -> tuple[int, str, float] | None:
    """Match a multi-word modifier phrase starting at ``index``.

    Returns ``(length, kind, multiplier)`` where ``kind`` is
    ``"intensifier"`` or ``"diminisher"``, or ``None`` if no phrase
    starts here. Longer phrases are tried first so "a great deal" is
    preferred over a hypothetical two-word prefix.
    """
    for length in (3, 2):
        end = index + length
        if end > len(tokens):
            continue
        key = tuple(_normalize(t) for t in tokens[index:end])
        entry = MODIFIER_PHRASES.get(key)
        if entry is not None:
            return length, entry[0], entry[1]
    return None


def _score_sentiment_token(
    raw_token: str,
    token: str,
    intensifier_mult: float,
    diminisher_mult: float,
    negation_active: bool,
    negation_distance: int,
    pending_negation: bool,
) -> tuple[float, float, float, float, float, bool]:
    """Score one sentiment-bearing token and return its contributions.

    Returns (base_score, pos_delta, neg_delta, new_intensifier_mult,
    new_diminisher_mult, new_negation_active). The caller appends
    base_score to the scores and contrastive lists.
    """
    base_score, intensifier_mult, diminisher_mult, negation_active = (
        _score_sentiment_word(
            raw_token, token, intensifier_mult, diminisher_mult,
            negation_active, negation_distance,
        )
    )
    pos_delta = base_score if base_score > 0 else 0.0
    neg_delta = abs(base_score) if base_score < 0 else 0.0
    # If this word was a negation+sentiment word (e.g., "no",
    # "never"), activate negation for the next sentiment word.
    if pending_negation:
        negation_active = True
    return (
        base_score, pos_delta, neg_delta,
        intensifier_mult, diminisher_mult, negation_active,
    )


def _advance_non_sentiment(
    intensifier_mult: float,
    diminisher_mult: float,
    negation_active: bool,
    negation_distance: int,
) -> tuple[float, float, bool, int]:
    """Reset modifiers after a non-sentiment word.

    Modifiers only affect the immediately following sentiment word,
    per VADER. Negation persists up to ``_NEGATION_WINDOW`` words, so
    "not the good" is correctly negated.
    """
    intensifier_mult = 1.0
    diminisher_mult = 1.0
    negation_distance += 1
    if negation_distance > _NEGATION_WINDOW:
        negation_active = False
    return intensifier_mult, diminisher_mult, negation_active, negation_distance


def _handle_negation(
    token: str,
) -> tuple[bool, bool, int]:
    """Process a negation token.

    Returns (is_pure_negation, pending_negation, new_negation_distance).
    A pure negation word (not also a sentiment word) skips scoring and
    activates negation; the caller continues the loop. A negation+
    sentiment word (e.g., "no", "never") falls through to scoring with
    ``pending_negation`` set, clearing any prior negation so it doesn't
    flip this word.
    """
    if token not in LEXICON:
        # Pure negation word — skip scoring, activate negation.
        return True, False, 0
    # Negation word that is also a sentiment word: score it as
    # sentiment (without self-negation), then activate negation for
    # the following word.
    return False, True, 0


def _match_modifier_word(
    token: str,
) -> tuple[float, float] | None:
    """If ``token`` is an intensifier or diminisher, return the new
    ``(intensifier_mult, diminisher_mult)``; otherwise return None.

    Intensifiers take precedence over diminishers if a word were both
    (matching the original check order).
    """
    if token in INTENSIFIERS:
        return INTENSIFIERS[token], 1.0
    if token in DIMINISHERS:
        return 1.0, DIMINISHERS[token]
    return None


def _score_tokens(
    tokens: list[str],
) -> tuple[list[float], float, float, int, list[float], list[float], bool]:
    """Score each sentiment-bearing token with context modifiers.

    Returns (scores, pos_sum, neg_sum, sentiment_count,
             pre_contrastive_scores, post_contrastive_scores,
             seen_contrastive).
    """
    scores: list[float] = []
    pos_sum = 0.0
    neg_sum = 0.0
    sentiment_count = 0

    # Track context: negation, intensifier, diminisher, contrastive
    negation_active = False
    negation_distance = 0
    intensifier_mult = 1.0
    diminisher_mult = 1.0

    # We need to track pre-contrastive scores separately
    pre_contrastive_scores: list[float] = []
    post_contrastive_scores: list[float] = []
    seen_contrastive = False

    i = 0
    token_count = len(tokens)
    while i < token_count:
        raw_token = tokens[i]
        token = _normalize(raw_token)

        # Check for a multi-word modifier phrase ("kind of", "a lot", …).
        # This consumes all the phrase's tokens so its constituent words
        # are not scored individually — critical for "kind of", where
        # "kind" is otherwise a positive sentiment word.
        phrase = _match_modifier_phrase(tokens, i)
        if phrase is not None:
            length, kind, multiplier = phrase
            if kind == "intensifier":
                intensifier_mult = multiplier
            else:
                diminisher_mult = multiplier
            i += length
            continue

        # Check for contrastive conjunction
        if token in CONTRASTIVES:
            seen_contrastive = True
            # Reset modifiers after contrastive
            intensifier_mult, diminisher_mult = 1.0, 1.0
            negation_active, negation_distance = False, 0
            i += 1
            continue

        # Check for negation (affects next sentiment word)
        pending_negation = False
        if token in NEGATIONS:
            is_pure, pending_negation, negation_distance = _handle_negation(token)
            if is_pure:
                # Pure negation word — skip scoring, activate negation.
                negation_active = True
                i += 1
                continue
            # Clear any prior negation so it doesn't flip this word.
            negation_active = False

        # Check for intensifier or diminisher (affects next sentiment word)
        mod = _match_modifier_word(token)
        if mod is not None:
            intensifier_mult, diminisher_mult = mod
            i += 1
            continue

        # Check if this word has sentiment
        if token in LEXICON:
            (
                base_score, pos_delta, neg_delta,
                intensifier_mult, diminisher_mult, negation_active,
            ) = _score_sentiment_token(
                raw_token, token, intensifier_mult, diminisher_mult,
                negation_active, negation_distance, pending_negation,
            )

            scores.append(base_score)
            sentiment_count += 1
            pos_sum += pos_delta
            neg_sum += neg_delta

            # Track pre/post contrastive
            if seen_contrastive:
                post_contrastive_scores.append(base_score)
            else:
                pre_contrastive_scores.append(base_score)
        else:
            # Non-sentiment word: reset modifiers (they only affect
            # the immediately following sentiment word, per VADER).
            intensifier_mult, diminisher_mult, negation_active, negation_distance = (
                _advance_non_sentiment(
                    intensifier_mult, diminisher_mult,
                    negation_active, negation_distance,
                )
            )

        i += 1

    return (
        scores, pos_sum, neg_sum, sentiment_count,
        pre_contrastive_scores, post_contrastive_scores, seen_contrastive,
    )


def _compute_proportions(
    pos_sum: float, neg_sum: float,
    sentiment_count: int, total_tokens: int,
) -> tuple[float, float, float]:
    """Compute VADER-style positive/negative/neutral proportions."""
    neutral_count = total_tokens - sentiment_count
    total = pos_sum + neg_sum + neutral_count
    if total > 0:
        positive = pos_sum / total
        negative = neg_sum / total
        neutral = neutral_count / total
    else:
        positive = 0.0
        negative = 0.0
        neutral = 1.0
    neutral = max(0.0, min(1.0, neutral))
    return positive, negative, neutral


def analyze_sentiment(text: str) -> dict:
    """Analyze the sentiment of a text.

    Returns a dict with:
        - compound: float in [-1, 1] — overall sentiment
        - positive: float in [0, 1] — proportion of positive sentiment
        - negative: float in [0, 1] — proportion of negative sentiment
        - neutral: float in [0, 1] — proportion of neutral sentiment
        - sentiment_words: int — number of sentiment-bearing words found
    """
    tokens = _tokenize(text)
    if not tokens:
        return {
            "compound": 0.0, "positive": 0.0, "negative": 0.0,
            "neutral": 1.0, "sentiment_words": 0,
        }

    (
        scores, pos_sum, neg_sum, sentiment_count,
        pre_contrastive, post_contrastive, seen_contrastive,
    ) = _score_tokens(tokens)

    if not scores:
        return {
            "compound": 0.0, "positive": 0.0, "negative": 0.0,
            "neutral": 1.0, "sentiment_words": 0,
        }

    # Apply contrastive weighting: post-contrastive scores get 1.5x weight
    if seen_contrastive and post_contrastive:
        weighted_scores: list[float] = []
        for s in pre_contrastive:
            weighted_scores.append(s * 0.5)
        for s in post_contrastive:
            weighted_scores.append(s * 1.5)
        scores = weighted_scores

        # Recompute sums
        pos_sum = sum(s for s in scores if s > 0)
        neg_sum = sum(abs(s) for s in scores if s < 0)

    # Apply punctuation boost to the final compound
    punct_boost = _punctuation_boost(text)

    # Compound score: sum of scores, normalized to [-1, 1]
    # Using the VADER normalization: sum / sqrt(sum^2 + alpha)
    raw_sum = sum(scores) * punct_boost
    alpha = 15.0  # VADER's normalization constant
    compound = raw_sum / (raw_sum * raw_sum + alpha) ** 0.5
    compound = max(-1.0, min(1.0, compound))

    positive, negative, neutral = _compute_proportions(
        pos_sum, neg_sum, sentiment_count, len(tokens)
    )

    return {
        "compound": compound,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "sentiment_words": sentiment_count,
    }

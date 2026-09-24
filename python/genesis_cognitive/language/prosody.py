r"""Prosody modeling — rhythm, pacing, emphasis, and intonation.

Prosody is the suprasegmental layer of speech: the patterns of stress,
rhythm, intonation, and pacing that carry meaning beyond the words
themselves. In spoken language, prosody disambiguates syntax, conveys
emotion, and signals discourse structure (Cutler, Dahan & van Donselaar,
1997). In written language, prosodic markers — emphasis, pauses,
punctuation rhythm — serve an analogous role, helping the reader
reconstruct the intended delivery.

This module provides:

1. **ProsodyPattern** — a dataclass capturing the prosodic properties
   of an utterance (rhythm, pacing, emphasis, pauses, intonation,
   stress).
2. **ProsodyGenerator** — maps an emotional state (valence + arousal)
   to an appropriate ProsodyPattern, following the dimensional theory
   of affect (Russell, 1980):

   - High arousal → faster pacing, more emphasis, staccato rhythm
   - Low arousal → slower pacing, flowing rhythm, longer pauses
   - Positive valence → rising intonation, lighter stress
   - Negative valence → falling intonation, heavier stress

The generator also applies prosodic markers to text via ``apply()``,
inserting emphasis markers (\*word\*), pause indicators (...), and
adjusting punctuation to reflect the pattern.

# References

- Cutler, A., Dahan, D., & van Donselaar, W. (1997). Prosody in the
  comprehension of spoken language: A literature review. *Language
  and Speech*, 40(2), 141–201.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of
  Personality and Social Psychology*, 39(6), 1161–1178.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["ProsodyGenerator", "ProsodyPattern"]

# ─── Pre-compiled regex patterns (module-level for performance) ────────
_TRAILING_PUNCT_RE = re.compile(r"([.!?]+)\s*$")


@dataclass(slots=True)
class ProsodyPattern:
    """The prosodic properties of an utteration.

    Prosody operates above the level of individual phonemes — it's the
    "music" of speech: rhythm, tempo, loudness, pitch contour, and
    pausing. This dataclass captures those properties in a form that
    language engines can use to shape generated text.

    Fields:
        rhythm: The overall rhythmic quality — "natural" (default),
            "staccato" (short, detached units), "flowing" (smooth,
            connected), "measured" (deliberate, even), "rapid" (fast,
            urgent).
        pacing: Words per minute relative to a baseline of 1.0.
            0.5 = slow, 1.0 = normal, 1.5 = fast. This is a multiplier,
            not an absolute rate.
        emphasis_words: Words to emphasize (stressed in delivery).
            In written output these are typically marked with
            asterisks.
        pause_positions: Word indices (0-based) where pauses occur.
            A pause after word *i* means the reader should break
            before word *i+1*.
        intonation: The pitch contour of the utterance — "neutral"
            (default), "rising" (questions, uncertainty, enthusiasm),
            "falling" (statements, finality, sadness), "level" (flat,
            monotone, detachment).
        stress_pattern: The metrical stress pattern — "default"
            (natural language stress), "trochaic" (stressed-unstressed,
            as in "table, "happy"), "iambic" (unstressed-stressed, as
            in "agree", "behind").
    """

    rhythm: str = "natural"
    pacing: float = 1.0
    emphasis_words: list[str] = field(default_factory=list)
    pause_positions: list[int] = field(default_factory=list)
    intonation: str = "neutral"
    stress_pattern: str = "default"


class ProsodyGenerator:
    """Generate prosody patterns from emotional state.

    Maps the two-dimensional affective space (valence × arousal) to
    prosodic properties. The mapping follows established findings in
    vocal emotion expression:

    - **Arousal** primarily controls tempo and rhythm. High arousal
      (excitement, anxiety) speeds up delivery and produces staccato
      or rapid rhythms. Low arousal (calm, sadness) slows delivery
      and produces flowing or measured rhythms with longer pauses.
    - **Valence** primarily controls intonation and stress. Positive
      valence produces rising intonation (enthusiasm, engagement) and
      lighter stress. Negative valence produces falling intonation
      (finality, resignation) and heavier stress patterns.

    The generator is deterministic given the same inputs — no
    randomness — so prosody is a faithful reflection of emotional
    state rather than noise.
    """

    def generate(
        self,
        emotion_label: str,
        valence: float,
        arousal: float,
    ) -> ProsodyPattern:
        """Generate a prosody pattern from an emotional state.

        Args:
            emotion_label: The named emotional state (e.g. "excited",
                "calm", "anxious"). Used for rhythm selection and
                emphasis heuristics.
            valence: The valence axis, typically -1.0 to +1.0.
                Positive = pleasant, negative = unpleasant.
            arousal: The arousal axis, typically 0.0 to 1.0.
                High = activated, low = deactivated.

        Returns:
            A ProsodyPattern shaped by the emotional state.
        """
        # ─── Rhythm: driven by arousal ───────────────────────
        rhythm = self._determine_rhythm(arousal)

        # ─── Pacing: linear mapping from arousal ─────────────
        # arousal 0.0 → 0.5x (slow), arousal 1.0 → 1.5x (fast)
        pacing = 0.5 + arousal * 1.0

        # ─── Intonation: driven by valence ───────────────────
        intonation = self._determine_intonation(valence)

        # ─── Stress pattern: valence-driven ──────────────────
        stress_pattern = self._determine_stress_pattern(valence)

        # ─── Emphasis words: based on emotion label ──────────
        emphasis_words = self._select_emphasis_words(emotion_label, valence, arousal)

        # ─── Pause positions: more pauses at low arousal ─────
        pause_positions = self._compute_pause_positions(arousal)

        return ProsodyPattern(
            rhythm=rhythm,
            pacing=pacing,
            emphasis_words=emphasis_words,
            pause_positions=pause_positions,
            intonation=intonation,
            stress_pattern=stress_pattern,
        )

    @staticmethod
    def _determine_rhythm(arousal: float) -> str:
        """Select a rhythm pattern based on the arousal level."""
        if arousal > 0.7:
            return "staccato"
        elif arousal > 0.5:
            return "rapid"
        elif arousal < 0.25:
            return "flowing"
        elif arousal < 0.4:
            return "measured"
        else:
            return "natural"

    @staticmethod
    def _determine_intonation(valence: float) -> str:
        """Select an intonation contour based on the valence level."""
        if valence > 0.2:
            return "rising"
        elif valence < -0.2:
            return "falling"
        else:
            return "neutral"

    @staticmethod
    def _determine_stress_pattern(valence: float) -> str:
        """Select a stress pattern based on the valence level."""
        if valence < -0.3:
            return "iambic"  # heavier, weighted
        elif valence > 0.3:
            return "trochaic"  # lighter, bouncy
        else:
            return "default"

    @staticmethod
    def _compute_pause_positions(arousal: float) -> list[int]:
        """Compute explicit pause positions from the arousal level.

        Low arousal → pauses roughly every 4-5 words.
        High arousal → fewer pauses (every 7-8 words).
        Actual positions depend on text length, so we store the
        gap as a heuristic; apply() will compute concrete positions.
        We encode the gap as negative positions meaning "every N
        words" — but to keep the field semantics clean (concrete
        indices), we leave it empty here and let apply() compute
        pauses from rhythm/pacing when no explicit positions are set.
        However, for very low arousal we add an explicit early pause.
        """
        if arousal < 0.3:
            return [2]  # pause early — hesitation
        return []

    def apply(
        self,
        text: str,
        prosody: ProsodyPattern,
        *,
        apply_intonation: bool = True,
    ) -> str:
        r"""Apply prosodic markers to text.

        Inserts emphasis markers (\*word\*), pause indicators (...),
        and adjusts punctuation to reflect the prosody pattern.

        - Emphasis words are wrapped in asterisks: *word*
        - Pause positions insert an ellipsis (...) at the word boundary
        - Rising intonation adds a trailing question mark if the text
          ends with a period (uncertainty / engagement)
        - Falling intonation is reinforced with a period (already
          the default for statements)
        - Staccato rhythm inserts brief pauses between short phrases

        Args:
            text: The input text to mark up.
            prosody: The prosody pattern to apply.
            apply_intonation: If False, skip the intonation contour
            adjustment (e.g. ``.`` → ``?``). The intonation is still
            stored in the ``ProsodyPattern`` for introspection, but the
            text's sentence type is not changed. This is useful when
            prosody is applied as an additive layer on top of already-
            composed text whose sentence type should be preserved.

        Returns:
            Text with prosodic markers inserted.
        """
        if not text:
            return text

        words = text.split()
        if not words:
            return text

        # ─── Emphasis ────────────────────────────────────────
        words = self._apply_emphasis(words, prosody)

        # ─── Pauses ──────────────────────────────────────────
        words = self._apply_pauses(words, prosody)

        text = " ".join(words)

        # ─── Intonation ──────────────────────────────────────
        if apply_intonation:
            text = self._apply_intonation(text, prosody)

        return text

    @staticmethod
    def _apply_emphasis(words: list[str], prosody: ProsodyPattern) -> list[str]:
        """Wrap emphasis words in asterisks, preserving trailing punctuation."""
        if not prosody.emphasis_words:
            return words
        emphasis_set = {w.lower() for w in prosody.emphasis_words}
        marked_words: list[str] = []
        for word in words:
            # Strip trailing punctuation for matching, then restore
            stripped = word.rstrip(".,;:!?")
            if stripped.lower() in emphasis_set:
                punct = word[len(stripped) :]
                marked_words.append(f"*{stripped}*{punct}")
            else:
                marked_words.append(word)
        return marked_words

    @staticmethod
    def _apply_pauses(words: list[str], prosody: ProsodyPattern) -> list[str]:
        """Compute and insert pause markers at clause boundaries.

        Pauses go at natural clause boundaries — after semicolons,
        colons, and before coordinating conjunctions (but, and, so,
        yet) — NOT after every comma. A comma in a list ("curious,
        thorough, and creative") is a list separator, not a clause
        boundary; inserting "..." there produces broken-looking output.

        To distinguish list commas from clause commas, we only insert
        a pause after a comma if the clause before it is long enough
        (4+ words since the last pause or sentence start). List items
        are typically 1-3 words, so this naturally skips them.

        Low arousal → longer pauses (more ellipsis dots), and an
        early hesitation pause before the first content word.
        High arousal → no pauses (fast, connected speech).
        """
        # High arousal → no pauses (fast, urgent speech)
        if prosody.rhythm in ("staccato", "rapid") and prosody.pacing > 1.2:
            return words

        # Honor explicit pause positions when the pattern carries them
        # (e.g. the early-hesitation pause generate() sets at low
        # arousal). Otherwise compute positions from clause boundaries.
        if prosody.pause_positions:
            pause_positions = [
                p for p in prosody.pause_positions if 0 <= p < len(words) - 1
            ]
            if pause_positions:
                return ProsodyGenerator._insert_pauses(words, pause_positions)

        # Compute pause positions at clause boundaries
        pause_positions = []
        clause_start = 0  # word index where the current clause began
        for i, word in enumerate(words):
            # Reset clause start after sentence-ending punctuation.
            # Without this, clause length is computed across sentence
            # boundaries — a short clause after a long sentence gets
            # an unwanted pause because the clause_start still points
            # at the previous sentence's beginning.
            if word.endswith((".", "!", "?")):
                clause_start = i + 1
                continue
            # Pause after punctuation that marks clause boundaries
            if word.endswith((",", ";", ":")):
                if i < len(words) - 1:
                    clause_len = i - clause_start
                    # Only pause after commas if the preceding clause
                    # is long enough to be a real clause (not a list
                    # item). Semicolons and colons always get a pause
                    # — they always mark clause boundaries.
                    if word.endswith((";", ":")) or clause_len >= 6:
                        pause_positions.append(i)
                    # Reset clause start to after this punctuation
                    clause_start = i + 1
            # Pause before coordinating conjunctions that start a new
            # clause (but only if both sides are full clauses).
            # Skip conjunctions in lists: "curious, thorough, and creative"
            # — the short items between commas are list elements, not
            # clauses. Also skip noun-phrase joins: "X and Y" where
            # both sides are short phrases, not full clauses. Only
            # pause before "and"/"but" if BOTH the preceding clause
            # (6+ words) AND the following clause (6+ words) are long
            # enough to be real clauses.
            elif (
                i > 2
                and word.lower().rstrip(",.!?;:") in (
                    "but", "and", "so", "yet", "because", "though",
                )
                and i < len(words) - 1
                and (i - clause_start) >= 6
            ):
                # Check the clause AFTER the conjunction — count
                # words until the next punctuation or sentence end.
                next_clause_len = 0
                for j in range(i + 1, len(words)):
                    if words[j].endswith((",", ".", "!", "?", ";", ":")):
                        break
                    next_clause_len += 1
                # Only pause if both sides are full clauses (6+ words)
                if next_clause_len >= 6:
                    pause_positions.append(i - 1)  # pause BEFORE the conjunction
                    clause_start = i

        # For very low arousal, add an early hesitation pause
        if prosody.rhythm in ("flowing", "measured") and len(words) > 3:
            # Hesitation after the first 1-2 words
            if 1 not in pause_positions and 0 not in pause_positions:
                pause_positions.insert(0, 1)

        # Insert pause markers at the specified positions.
        # Limit to at most 2 pauses per sentence to avoid over-pausing.
        if not pause_positions:
            return words
        return ProsodyGenerator._insert_pauses(words, pause_positions[:2])

    @staticmethod
    def _insert_pauses(words: list[str], pause_positions: list[int]) -> list[str]:
        """Insert "..." pause markers after the given word indices."""
        result: list[str] = []
        pause_set = set(pause_positions)
        for i, word in enumerate(words):
            result.append(word)
            if i in pause_set and i < len(words) - 1:
                # Don't add pause right before sentence-ending punct
                if not words[i + 1].rstrip().endswith((".", "!", "?")):
                    result.append("...")
        return result

    @staticmethod
    def _apply_intonation(text: str, prosody: ProsodyPattern) -> str:
        """Adjust trailing punctuation to reflect the intonation contour.

        Rising intonation converts a trailing period to a question mark
        (uncertainty / engagement). Falling intonation ensures the text
        ends with a period (finality).
        """
        if prosody.intonation == "rising":
            # Convert trailing period to question mark for rising tone
            # (but only for declarative-looking sentences)
            m = _TRAILING_PUNCT_RE.search(text)
            if m and m.group(1) == ".":
                text = text[: m.start()] + "?" + text[m.end() :]
        elif prosody.intonation == "falling":
            # Ensure it ends with a period (falling = finality)
            if not text.endswith((".", "!", "?")):
                text += "."
        return text

    # ─── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _select_emphasis_words(
        emotion_label: str,
        valence: float,
        arousal: float,
    ) -> list[str]:
        """Select words to emphasize based on emotional state.

        Returns empty — emphasis words should come from the concept
        network (learned associations between emotions and words),
        not from hardcoded lists. The prosody mechanism stays as a
        scaffold: if it learns which words to emphasize, the
        ProsodyPattern.emphasis_words field will carry them.
        """
        return []

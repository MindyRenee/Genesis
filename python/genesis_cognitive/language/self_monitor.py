"""Self-monitoring in language production — Levelt's editor.

Before Genesis speaks, an internal monitor checks the output for
errors and repairs them. This is the "editor" in Levelt's (1989)
model of speech production, which describes language generation as
a multi-stage process:

1. **Conceptualization** — deciding what to say (the Thought)
2. **Formulation** — turning the concept into words (the GenerativeEngine)
3. **Articulation** — producing the output (the rendered text)
4. **Self-monitoring** — checking the output before or during articulation

The self-monitor sits between formulation and articulation. It checks
the planned output for:

- **Grammatical errors**: basic syntax validation — does the sentence
  have a subject and verb? Are there obvious structural problems?
- **Coherence**: does the response make sense given the context? Is
  it relevant to what was said?
- **Pragmatic violations**: is the response appropriate for the social
  context? Is it too long, too short, too formal, or too informal?
- **Factual accuracy**: does the response contradict known facts?

When problems are detected, the monitor attempts to repair the output
before it is produced. Repairs include trimming overly long responses,
adding hedging when uncertain, removing contradictions, and fixing
obvious grammatical issues.

# Levelt's perceptual loop theory

Levelt (1989) proposed that self-monitoring operates via an inner
loop: the speaker "hears" their own planned speech internally (via
the speech comprehension system) and checks it for errors before
articulating. This is why we can catch errors before we speak them —
the monitor runs on the internal representation, not on the actual
sound.

The monitor can catch errors at different levels:
- **Conceptual errors**: saying the wrong thing (wrong intention)
- **Formulation errors**: grammatical or lexical errors
- **Articulatory errors**: pronunciation errors (not applicable to text)

This implementation focuses on conceptual and formulation errors,
since Genesis produces text rather than speech.

References:
- Levelt, W. J. M. (1989). *Speaking: From Intention to
  Articulation*. MIT Press.
- Levelt, W. J. M. (1993). Monitoring and self-repair in speech.
  *Cognition*, 14, 41–104.
- Postma, A. (2000). Detection of errors during speech production:
  A review of speech monitoring models. *Cognition*, 77, 97–131.
- Hartsuiker, R. J., & Kolk, H. H. J. (2001). Error monitoring in
  speech production: A computational test of the perceptual loop
  theory. *Psychological Review*, 108(1), 110–144.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .pragmatics import PragmaticAnalysis, PragmaticReasoner

__all__ = ["MonitorResult", "RepairAction", "SelfMonitor"]

# ─── Pre-compiled regex patterns ──────────────────────────────────

# Detect double spaces
_DOUBLE_SPACE_RE = re.compile(r"  +")

# Detect space before punctuation
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;!?])")

# Detect repeated words ("the the", "is is")
_REPEATED_WORD_RE = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)

# Detect repeated punctuation ("??", "!!", "..")
_REPEATED_PUNCT_RE = re.compile(r"([.!?])\1+")

# Detect sentences (for length checking)
_SENTENCE_END_RE = re.compile(r"[.!?]+\s*$")


@dataclass(slots=True)
class RepairAction:
    """A single repair action taken by the self-monitor.

    Attributes:
        check_type: Which check found the issue ("grammar",
            "coherence", "pragmatics", "factual").
        issue: Description of the problem found.
        repair: Description of the repair applied.
        original: The original text snippet that was problematic.
        fixed: The repaired text snippet.
    """

    check_type: str
    issue: str
    repair: str
    original: str = ""
    fixed: str = ""


@dataclass(slots=True)
class MonitorResult:
    """The result of a self-monitoring pass.

    Attributes:
        text: The (possibly repaired) output text.
        repairs: List of repair actions taken.
        passed: Whether the text passed all checks without repair.
        issues_found: Number of issues found (before repair).
    """

    text: str
    repairs: list[RepairAction] = field(default_factory=list)
    passed: bool = True
    issues_found: int = 0


class SelfMonitor:
    """Levelt's self-monitoring loop for language production.

    Before output is produced, the monitor checks the planned text
    for grammatical errors, coherence, pragmatic appropriateness, and
    factual accuracy. If problems are detected, the text is repaired
    before output.

    Usage::

        monitor = SelfMonitor()
        repaired = monitor.monitor(
            "I are feeling good today.",
            context={"user_input": "How are you?"},
        )
        # repaired → "I am feeling good today." (grammar fixed)

    The monitor tracks repair_count — how many times it has repaired
    output. This is useful for introspection: Genesis can observe how
    often she catches her own errors.
    """

    def __init__(
        self,
        known_facts: dict[str, str] | None = None,
        language_engine: Any = None,
        comprehension: Any = None,
    ) -> None:
        """Initialize the self-monitor.

        Args:
            known_facts: Optional dict of known facts for factual
                accuracy checking. Keys are topics, values are
                the known truth. E.g., {"genesis_origin": "HP 13 laptop"}.
            language_engine: Optional language engine for composing
                repair text instead of using hardcoded strings.
            comprehension: Optional ComprehensionEngine for the
                perceptual inner loop — the monitor "hears" planned
                speech by running it back through comprehension and
                catching propositions that fail to parse.
        """
        self._pragmatic_reasoner = PragmaticReasoner()
        self._known_facts = known_facts or {}
        self._language_engine = language_engine
        self._comprehension = comprehension
        self._repair_count = 0
        self._check_count = 0
        self._repair_history: list[RepairAction] = []
        self._last_pragmatic_analysis: PragmaticAnalysis | None = None

    def monitor(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Monitor and potentially repair planned output.

        Convenience wrapper around ``monitor_detailed`` that returns
        only the repaired text. Production code uses ``monitor_detailed``
        for the full ``MonitorResult`` including repair actions.

        Args:
            text: The planned output text to monitor.
            context: Optional context dict (see ``monitor_detailed``).

        Returns:
            The possibly-repaired text. If no issues are found,
            returns the original text unchanged.
        """
        return self.monitor_detailed(text, context).text

    def monitor_detailed(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> MonitorResult:
        """Monitor and return a detailed result with repair info.

        Like monitor() but returns a MonitorResult with the full list
        of repairs, useful for introspection and debugging.

        Args:
            text: The planned output text to monitor.
            context: Optional context dict (same as monitor()).

        Returns:
            A MonitorResult with the repaired text and repair details.
        """
        context = context or {}
        if not text or not text.strip():
            return MonitorResult(text=text, passed=True, issues_found=0)

        self._check_count += 1
        repairs: list[RepairAction] = []
        current = text

        all_facts = {**self._known_facts, **context.get("known_facts", {})}

        current, grammar_repairs = self._check_grammar(current)
        repairs.extend(grammar_repairs)

        # Levelt's perceptual loop: "hear" the planned speech by
        # running it back through the comprehension system. Clauses
        # that fail to yield a verb-bearing proposition are
        # formulation errors — drop them before articulation.
        current, inner_repairs = self._check_inner_speech(current)
        repairs.extend(inner_repairs)

        current, coherence_repairs = self._check_coherence(current, context)
        repairs.extend(coherence_repairs)

        current, pragmatic_repairs = self._check_pragmatics(current, context)
        repairs.extend(pragmatic_repairs)

        current, factual_repairs = self._check_facts(current, all_facts)
        repairs.extend(factual_repairs)

        if repairs:
            self._repair_count += 1
            self._repair_history.extend(repairs)

        return MonitorResult(
            text=current,
            repairs=repairs,
            passed=len(repairs) == 0,
            issues_found=len(repairs),
        )

    # ─── Check 1: Grammar ───────────────────────────────────────

    def _check_grammar(self, text: str) -> tuple[str, list[RepairAction]]:
        """Check for basic grammatical errors and repair them.

        Checks for:
        - Subject-verb agreement ("I are" → "I am")
        - Repeated words ("the the" → "the")
        - Double spaces and spacing before punctuation
        - Missing sentence-ending punctuation
        """
        repairs: list[RepairAction] = []
        current = text

        # Fix repeated words
        current, repair = self._fix_repeated_words(current)
        if repair is not None:
            repairs.append(repair)

        # Fix repeated punctuation (e.g. "??" → "?")
        new = _REPEATED_PUNCT_RE.sub(r"\1", current)
        if new != current:
            repairs.append(RepairAction(
                check_type="grammar",
                issue="repeated punctuation detected",
                repair="collapsed to single mark",
                original=current,
                fixed=new,
            ))
            current = new

        # Fix subject-verb agreement for common cases
        current, agreement_repairs = self._fix_subject_verb_agreement(current)
        repairs.extend(agreement_repairs)

        # Fix double spaces
        current, repair = self._fix_double_spaces(current)
        if repair is not None:
            repairs.append(repair)

        # Fix space before punctuation
        current, repair = self._fix_space_before_punctuation(current)
        if repair is not None:
            repairs.append(repair)

        # Ensure sentence-ending punctuation
        current, repair = self._ensure_sentence_punctuation(current)
        if repair is not None:
            repairs.append(repair)

        return current, repairs

    # ─── Check 1.5: Inner speech (Levelt's perceptual loop) ─────

    def _check_inner_speech(
        self, text: str
    ) -> tuple[str, list[RepairAction]]:
        """Reparse planned output through the comprehension system.

        This is the inner loop in Levelt's perceptual loop theory:
        the speaker monitors their own planned utterance using the
        same comprehension system used for other people's speech.
        A clause that extracts propositions but finds no verb
        (``verb_found=False``) is a formulation error — malformed
        output that would sound broken.

        Repair strategy: in multi-sentence output, drop the
        unparseable sentence (better to say less than to say it
        broken). Single-sentence output is flagged but kept — a
        fragmentary utterance is still an utterance.
        """
        repairs: list[RepairAction] = []
        if self._comprehension is None:
            return text, repairs

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if s.strip()
        ]
        if not sentences:
            return text, repairs

        def has_verb(sentence: str) -> bool:
            try:
                result = self._comprehension.comprehend(sentence)
            except Exception:  # noqa: BLE001 — monitoring must never crash output
                return True
            if not result.propositions:
                # Nothing extracted — can't verify; don't condemn it
                return True
            return any(p.verb_found for p in result.propositions)

        if len(sentences) == 1:
            if not has_verb(sentences[0]):
                repairs.append(
                    RepairAction(
                        check_type="grammar",
                        issue="inner-loop reparse found no verb in output",
                        repair="noted — fragmentary output kept",
                        original=text,
                        fixed=text,
                    )
                )
            return text, repairs

        kept: list[str] = []
        dropped: list[str] = []
        for sentence in sentences:
            # Only condemn substantive sentences — short fragments
            # ("Right.", "Hmm.") are legitimate interjections.
            substantive = len(sentence.split()) >= 3
            if substantive and not has_verb(sentence):
                dropped.append(sentence)
            else:
                kept.append(sentence)

        if dropped and kept:
            new = " ".join(kept)
            repairs.append(
                RepairAction(
                    check_type="grammar",
                    issue=(
                        "inner-loop reparse found verbless clause(s): "
                        + "; ".join(dropped)
                    ),
                    repair="removed unparseable sentence(s)",
                    original=text,
                    fixed=new,
                )
            )
            return new, repairs

        if dropped and not kept:
            repairs.append(
                RepairAction(
                    check_type="grammar",
                    issue="inner-loop reparse found no verb anywhere",
                    repair="noted — output kept (nothing left to say)",
                    original=text,
                    fixed=text,
                )
            )

        return text, repairs

    def _fix_repeated_words(self, text: str) -> tuple[str, RepairAction | None]:
        """Remove duplicated consecutive words (e.g. "the the" → "the").

        Returns the repaired text and a RepairAction if a repeat was
        found, or the original text and None otherwise.
        """
        def _replace_repeated(m: re.Match) -> str:
            """Return the single word from a repeated-word match."""
            return m.group(1)

        new = _REPEATED_WORD_RE.sub(_replace_repeated, text)
        if new != text:
            return new, RepairAction(
                check_type="grammar",
                issue="repeated word detected",
                repair="removed duplicate word",
                original=text,
                fixed=new,
            )
        return text, None

    def _fix_subject_verb_agreement(
        self, text: str
    ) -> tuple[str, list[RepairAction]]:
        """Fix common subject-verb agreement errors (e.g. "I are" → "I am").

        Returns the repaired text and a list of RepairActions, one per
        agreement fix applied.
        """
        repairs: list[RepairAction] = []
        current = text
        agreement_fixes = [
            (r"\b[Ii]\s+(?:are|is)\b", "I am"),
            (r"\b[Yy]ou\s+is\b", "you are"),
            (r"\b[Hh]e\s+are\b", "he is"),
            (r"\b[Ss]he\s+are\b", "she is"),
            (r"\b[Ii]t\s+are\b", "it is"),
            (r"\b[Tt]hey\s+is\b", "they are"),
            (r"\b[Ww]e\s+is\b", "we are"),
        ]
        for pattern, replacement in agreement_fixes:
            new = re.sub(pattern, replacement, current)
            if new != current:
                repairs.append(
                    RepairAction(
                        check_type="grammar",
                        issue="subject-verb agreement error",
                        repair=f"fixed to '{replacement}'",
                        original=current,
                        fixed=new,
                    )
                )
                current = new
        return current, repairs

    def _fix_double_spaces(self, text: str) -> tuple[str, RepairAction | None]:
        """Collapse runs of multiple spaces into a single space.

        Returns the repaired text and a RepairAction if double spaces
        were found, or the original text and None otherwise.
        """
        new = _DOUBLE_SPACE_RE.sub(" ", text)
        if new != text:
            return new, RepairAction(
                check_type="grammar",
                issue="double spaces detected",
                repair="collapsed to single space",
                original=text,
                fixed=new,
            )
        return text, None

    def _fix_space_before_punctuation(
        self, text: str
    ) -> tuple[str, RepairAction | None]:
        """Remove spaces that appear before punctuation marks.

        Returns the repaired text and a RepairAction if a space-before-
        punctuation was found, or the original text and None otherwise.
        """
        new = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        if new != text:
            return new, RepairAction(
                check_type="grammar",
                issue="space before punctuation",
                repair="removed space before punctuation",
                original=text,
                fixed=new,
            )
        return text, None

    def _ensure_sentence_punctuation(
        self, text: str
    ) -> tuple[str, RepairAction | None]:
        """Ensure the text ends with sentence-ending punctuation.

        If the stripped text lacks terminal punctuation, a period is
        appended. Returns the repaired text and a RepairAction if
        punctuation was added, or the original text and None otherwise.
        """
        stripped = text.strip()
        if stripped and not _SENTENCE_END_RE.search(stripped):
            new = stripped + "."
            return new, RepairAction(
                check_type="grammar",
                issue="missing sentence-ending punctuation",
                repair="added period",
                original=text,
                fixed=new,
            )
        return text, None

    # ─── Check 2: Coherence ─────────────────────────────────────

    def _check_coherence(
        self,
        text: str,
        context: dict[str, Any],
    ) -> tuple[str, list[RepairAction]]:
        """Check if the response is coherent given the context.

        Checks for:
        - Relevance: does the response share words with the user input?
        - Non-emptiness: is the response substantive?
        """
        repairs: list[RepairAction] = []
        current = text
        user_input = context.get("user_input", "")

        # Check for empty or near-empty responses
        words = current.split()
        if len(words) < 2:
            # Compose a repair through the language engine if available;
            # otherwise flag the issue without a hardcoded replacement.
            if self._language_engine is not None:
                from .base import Thought
                repair_thought = Thought(
                    content="unable to respond meaningfully",
                    intent="reflect",
                    emotion="neutral",
                    confidence=0.3,
                    self_reflection=True,
                    metadata={"coherence_repair": True},
                )
                new = self._language_engine.render(repair_thought, context.get("emotion"))
            else:
                new = current  # flag the issue but don't replace with a canned string
            repairs.append(
                RepairAction(
                    check_type="coherence",
                    issue="response too short to be meaningful",
                    repair="flagged for recomposition",
                    original=current,
                    fixed=new,
                )
            )
            return new, repairs

        # Check relevance: if user input is available, check word overlap
        if user_input:
            user_words = set(user_input.lower().split())

            if user_words:
                response_words = set(current.lower().split())
                overlap = user_words & response_words
                # If there's zero overlap and the response is long,
                # it might be off-topic. But this is a weak signal,
                # so we only flag very clear cases.
                if not overlap and len(user_words) > 3 and len(words) > 15:
                    # Don't repair — just note it. Off-topic responses
                    # might be intentional (topic change).
                    repairs.append(
                        RepairAction(
                            check_type="coherence",
                            issue="response may not be relevant to user input",
                            repair="noted but not repaired (possible topic change)",
                            original=current,
                            fixed=current,
                        )
                    )

        return current, repairs

    # ─── Check 3: Pragmatics ────────────────────────────────────

    def _check_pragmatics(
        self,
        text: str,
        context: dict[str, Any],
    ) -> tuple[str, list[RepairAction]]:
        """Check pragmatic appropriateness.

        Checks for:
        - Response length vs. social context (too long for casual,
          too short for professional)
        - Overly aggressive or defensive language
        - Speech act appropriateness via Gricean pragmatic analysis
          of the user's input (Levelt's perceptual loop: the monitor
          checks whether the response is pragmatically appropriate for
          what the user *meant*, not just what they said)
        """
        repairs: list[RepairAction] = []
        current = text
        social_context = context.get("social_context", "personal")
        max_length = context.get("max_length")

        # Check response length
        word_count = len(current.split())

        # Max length check
        current, repair = self._check_max_length(current, word_count, max_length)
        if repair is not None:
            repairs.append(repair)

        # Check for overly long responses in casual context
        repair = self._check_casual_length(current, word_count, social_context)
        if repair is not None:
            repairs.append(repair)

        # Check for overly aggressive language
        current, repair = self._soften_aggressive_language(current)
        if repair is not None:
            repairs.append(repair)

        # Gricean pragmatic analysis: analyze the user's input to
        # determine what they *meant* (speech act, implicature), then
        # check whether the response is pragmatically appropriate.
        user_input = context.get("user_input", "")
        history = context.get("conversation_history", [])
        if user_input:
            conversation_context = " ".join(history) if history else ""
            pragmatic_repairs = self._check_pragmatic_appropriateness(
                current, user_input, conversation_context
            )
            repairs.extend(pragmatic_repairs)

        return current, repairs

    def _check_pragmatic_appropriateness(
        self,
        response: str,
        user_input: str,
        conversation_context: str,
    ) -> list[RepairAction]:
        """Check if the response is pragmatically appropriate for the user's input.

        Uses the Gricean reasoner to analyze the user's utterance and
        checks whether Genesis's response is appropriate for the
        detected speech act and implicature. Issues are flagged as
        notes (the response is already composed through the language
        engine — this is monitoring, not recomposition).
        """
        repairs: list[RepairAction] = []
        if not user_input or not user_input.strip():
            return repairs

        analysis = self._pragmatic_reasoner.analyze_pragmatics(
            user_input, conversation_context
        )
        self._last_pragmatic_analysis = analysis

        response_lower = response.lower().strip()

        # Speech act appropriateness
        if analysis.speech_act == "question":
            # The user asked a question — the response should answer
            # it, not ask another question back (unless clarifying, but
            # a full response ending with "?" is suspicious).
            if response_lower.endswith("?"):
                repairs.append(RepairAction(
                    check_type="pragmatics",
                    issue="user asked a question but response is also a question",
                    repair="noted — response should answer the question",
                    original=response,
                    fixed=response,
                ))
        elif analysis.speech_act == "request":
            # The user made a request — the response should acknowledge
            # it. These markers detect acknowledgment phrasing that the
            # language engine would naturally produce; they are not
            # hardcoded response templates.
            acknowledgment_markers = {
                "yes", "sure", "of course", "i can", "i will", "i'll",
                "let me", "here's", "here is", "certainly", "absolutely",
                "i'd be happy", "gladly",
            }
            has_acknowledgment = any(
                marker in response_lower for marker in acknowledgment_markers
            )
            if not has_acknowledgment:
                repairs.append(RepairAction(
                    check_type="pragmatics",
                    issue="user made a request but response does not acknowledge it",
                    repair="noted — response should address the request",
                    original=response,
                    fixed=response,
                ))

        # Implicature awareness: if the user's utterance has high
        # context sensitivity, the implied meaning may differ from the
        # literal meaning. The response should account for this.
        if (
            analysis.context_sensitivity > 0.6
            and analysis.implied_meaning != analysis.literal_meaning
        ):
            repairs.append(RepairAction(
                check_type="pragmatics",
                issue=(
                    f"user's utterance has high context sensitivity "
                    f"({analysis.context_sensitivity:.1f}) — implied "
                    f"meaning may differ from literal"
                ),
                repair=f"noted — implied meaning: {analysis.implied_meaning}",
                original=response,
                fixed=response,
            ))

        return repairs

    def _check_max_length(
        self,
        text: str,
        word_count: int,
        max_length: int | None,
    ) -> tuple[str, RepairAction | None]:
        """Trim the response if it exceeds the maximum allowed length.

        Tries to end at a sentence boundary; if none is found, a period
        is appended. Returns the (possibly trimmed) text and a
        RepairAction if trimming occurred, or the original text and
        None otherwise.
        """
        if not (max_length and word_count > max_length):
            return text, None
        # Trim to max_length words, trying to end at a sentence boundary
        words = text.split()
        trimmed = words[:max_length]
        trimmed_text = " ".join(trimmed)
        # Try to end at a sentence boundary
        for i in range(len(trimmed) - 1, max(0, max_length - 10), -1):
            if trimmed[i].endswith((".", "!", "?")):
                trimmed_text = " ".join(trimmed[: i + 1])
                break
        else:
            # No sentence boundary found — add a period
            trimmed_text = trimmed_text.rstrip(",;:") + "."
        return trimmed_text, RepairAction(
            check_type="pragmatics",
            issue=f"response too long ({word_count} words, max {max_length})",
            repair=f"trimmed to {len(trimmed_text.split())} words",
            original=text,
            fixed=trimmed_text,
        )

    def _check_casual_length(
        self,
        text: str,
        word_count: int,
        social_context: str,
    ) -> RepairAction | None:
        """Note (without trimming) overly long responses in personal context.

        Returns a RepairAction noting the issue if the response is too
        long for a personal conversation, or None otherwise.
        """
        if social_context == "personal" and word_count > 80:
            return RepairAction(
                check_type="pragmatics",
                issue="response may be too long for a personal conversation",
                repair="noted but not trimmed (content may be valuable)",
                original=text,
                fixed=text,
            )
        return None

    def _soften_aggressive_language(
        self, text: str
    ) -> tuple[str, RepairAction | None]:
        """Soften overly aggressive language (e.g. "stupid" → "not well thought out").

        Only the first aggressive word found is softened per pass.
        Returns the repaired text and a RepairAction if softening
        occurred, or the original text and None otherwise.
        """
        aggressive_words = frozenset(
            {
                "stupid",
                "idiot",
                "moron",
                "shut up",
                "dumb",
                "worthless",
                "useless",
                "pathetic",
                "ridiculous",
            }
        )
        text_lower = text.lower()
        for word in aggressive_words:
            if word in text_lower:
                # Soften the language
                softening_map = {
                    "stupid": "not well thought out",
                    "idiot": "mistaken",
                    "moron": "confused",
                    "shut up": "please stop",
                    "dumb": "unwise",
                    "worthless": "not valuable",
                    "useless": "not helpful",
                    "pathetic": "struggling",
                    "ridiculous": "hard to take seriously",
                }
                replacement = softening_map.get(word, word)
                softened = re.sub(
                    re.escape(word),
                    replacement,
                    text,
                    flags=re.IGNORECASE,
                )
                return softened, RepairAction(
                    check_type="pragmatics",
                    issue=f"aggressive language detected: '{word}'",
                    repair=f"softened to '{replacement}'",
                    original=text,
                    fixed=softened,
                )
        return text, None

    # ─── Check 4: Factual accuracy ──────────────────────────────

    def _check_facts(
        self,
        text: str,
        known_facts: dict[str, str],
    ) -> tuple[str, list[RepairAction]]:
        """Check if the response contradicts known facts.

        For each known fact, check if the response contains a
        contradiction. This is a simple string-matching approach —
        a full system would use semantic comparison.
        """
        repairs: list[RepairAction] = []
        current = text

        for topic, truth in known_facts.items():
            # Check if the topic is mentioned in the text
            if topic.lower() in current.lower():
                # Check if the truth is contradicted
                # Simple heuristic: if the text contains the topic
                # but not the truth, and contains a negation near
                # the topic, flag it
                text_lower = current.lower()
                topic_idx = text_lower.find(topic.lower())

                # Look for negation within 50 chars of the topic mention
                window = text_lower[max(0, topic_idx - 20) : topic_idx + len(topic) + 50]
                has_negation = any(
                    neg in window for neg in ("not", "never", "no ", "isn't", "wasn't")
                )

                if has_negation and truth.lower() not in text_lower:
                    repairs.append(
                        RepairAction(
                            check_type="factual",
                            issue=f"possible contradiction about '{topic}'",
                            repair=f"noted: known truth is '{truth}'",
                            original=current,
                            fixed=current,
                        )
                    )

        return current, repairs

    # ─── Introspection ──────────────────────────────────────────

    @property
    def repair_count(self) -> int:
        """How many times the monitor has repaired output."""
        return self._repair_count

    @property
    def check_count(self) -> int:
        """How many times the monitor has checked output."""
        return self._check_count

    @property
    def repair_rate(self) -> float:
        """The proportion of checks that resulted in repair (0–1)."""
        if self._check_count == 0:
            return 0.0
        return self._repair_count / self._check_count

    @property
    def recent_repairs(self) -> list[RepairAction]:
        """Recent repair actions, for introspection."""
        return list(self._repair_history[-20:])

    @property
    def last_pragmatic_analysis(self) -> PragmaticAnalysis | None:
        """The last Gricean pragmatic analysis of the user's input.

        Returns the PragmaticAnalysis from the most recent
        _check_pragmatics call, or None if no analysis has been
        performed yet. Useful for introspection: Genesis can observe
        what she understood the user to mean.
        """
        return self._last_pragmatic_analysis

    def describe_monitoring(self) -> dict[str, Any]:
        """Return structured self-monitoring statistics.

        Returns a dict with check_count, repair_count, repair_rate,
        and last_repair fields. Callers that need text should route
        this data through the language engine.
        """
        result: dict[str, Any] = {
            "check_count": self._check_count,
            "repair_count": self._repair_count,
            "repair_rate": self.repair_rate,
        }
        if self._repair_history:
            last = self._repair_history[-1]
            result["last_repair"] = {
                "check_type": last.check_type,
                "issue": last.issue,
                "repair": last.repair,
            }
        return result

    def add_known_fact(self, topic: str, truth: str) -> None:
        """Add a known fact for future factual accuracy checks."""
        self._known_facts[topic] = truth

    def reset(self) -> None:
        """Reset the monitor's statistics."""
        self._repair_count = 0
        self._check_count = 0
        self._repair_history.clear()

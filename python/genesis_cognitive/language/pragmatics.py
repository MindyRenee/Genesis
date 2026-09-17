"""Pragmatic reasoning — meaning in context.

Pragmatics is the branch of linguistics that studies how context
contributes to meaning. The same words can mean different things in
different contexts — "it's cold in here" can be a factual observation,
a request to close a window, or a complaint about the thermostat.

This module implements pragmatic reasoning based on Paul Grice's
theory of conversational implicature (Grice, 1975). Grice proposed
that conversation is governed by a Cooperative Principle and four
maxims:

1. **Quantity**: Make your contribution as informative as required —
   not more, not less.
2. **Quality**: Try to make your contribution one that is true — do
   not say what you believe to be false, do not say that for which
   you lack adequate evidence.
3. **Relation**: Be relevant.
4. **Manner**: Be perspicuous — avoid obscurity, ambiguity, and
   be brief and orderly.

When a speaker appears to violate a maxim, the listener assumes the
violation is intentional and infers an *implicature* — what the speaker
meant but did not say. "Can you pass the salt?" is literally a question
about ability, but the implicature is a request to pass the salt.

# Context-dependent meaning

The same utterance means different things in different contexts. "I'm
tired" said at 10 PM means something different from "I'm tired" said
at 10 AM. The PragmaticReasoner takes context into account when
determining meaning.

References:
- Grice, H. P. (1975). Logic and conversation. In P. Cole & J. L.
  Morgan (Eds.), *Syntax and Semantics 3: Speech Acts* (pp. 41–58).
  Academic Press.
- Levinson, S. C. (2000). *Presumptive Meanings: The Theory of
  Generalized Conversational Implicature*. MIT Press.
- Searle, J. R. (1969). *Speech Acts: An Essay in the Philosophy of
  Language*. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PragmaticAnalysis", "PragmaticReasoner"]


@dataclass(slots=True)
class PragmaticAnalysis:
    """The result of pragmatic analysis of an utterance.

    Attributes:
        literal_meaning: The surface-level, literal meaning of the text.
        implied_meaning: What the speaker likely meant beyond the
            literal words (the implicature).
        maxims_assessed: Assessment of each Gricean maxim — whether
            it's satisfied, violated, or flouted, with an explanation.
        speech_act: The likely speech act (assertion, request, question,
            promise, apology, etc.) following Searle (1969).
        context_sensitivity: How much the meaning depends on context
            (0–1). High values mean the literal meaning alone is
            insufficient to understand the utterance.
        confidence: Overall confidence in the analysis (0–1).
    """

    literal_meaning: str
    implied_meaning: str
    maxims_assessed: dict[str, dict] = field(default_factory=dict)
    speech_act: str = "assertion"
    context_sensitivity: float = 0.3
    confidence: float = 0.5


class PragmaticReasoner:
    """Analyze the pragmatic meaning of utterances.

    Applies Gricean maxims, detects conversational implicature, and
    determines context-dependent meaning. This is the layer that
    understands *what is meant* rather than just *what is said*.

    Usage::

        reasoner = PragmaticReasoner()
        analysis = reasoner.analyze_pragmatics(
            "It's cold in here",
            context="The window is open and the user is shivering",
        )
        # analysis.implied_meaning might be "Please close the window"
    """

    # Speech act indicators — patterns that suggest the illocutionary
    # force of an utterance (Searle, 1969).
    _REQUEST_PATTERNS = frozenset(
        {
            "can you",
            "could you",
            "would you",
            "will you",
            "please",
            "mind if",
            "how about",
            "why don't",
        }
    )
    _QUESTION_PATTERNS = frozenset(
        {
            "what",
            "why",
            "how",
            "when",
            "where",
            "who",
            "is it",
            "are you",
            "do you",
            "does it",
        }
    )
    _PROMISE_PATTERNS = frozenset(
        {
            "i will",
            "i promise",
            "i'll",
            "i shall",
            "i guarantee",
            "i commit",
        }
    )
    _APOLOGY_PATTERNS = frozenset(
        {
            "sorry",
            "apologize",
            "my fault",
            "i regret",
            "pardon",
            "excuse me",
        }
    )

    # Indirect request indicators — utterances that are literally
    # questions or statements but pragmatically function as requests.
    _INDIRECT_REQUEST_INDICATORS = frozenset(
        {
            "cold",
            "hot",
            "warm",
            "thirsty",
            "hungry",
            "tired",
            "dark",
            "bright",
            "loud",
            "quiet",
            "open",
            "closed",
            "stuffy",
        }
    )

    def analyze_pragmatics(self, text: str, context: str = "") -> PragmaticAnalysis:
        """Analyze the pragmatic meaning of an utterance.

        Args:
            text: The utterance to analyze.
            context: Optional context that affects interpretation.

        Returns:
            A PragmaticAnalysis with literal meaning, implied meaning,
            maxim assessments, and speech act classification.
        """
        if not text or not text.strip():
            return PragmaticAnalysis(
                literal_meaning="",
                implied_meaning="",
                confidence=0.0,
            )

        text_lower = text.lower().strip()

        # Determine speech act
        speech_act = self._classify_speech_act(text_lower)

        # Literal meaning (surface interpretation)
        literal = self._extract_literal_meaning(text)

        # Implied meaning (implicature)
        implied = self._infer_implicature(text_lower, context, speech_act)

        # Assess Gricean maxims
        maxims = self._assess_maxims(text_lower, context)

        # Context sensitivity
        ctx_sensitivity = self._compute_context_sensitivity(text_lower, context, speech_act)

        # Confidence
        confidence = self._compute_confidence(maxims, ctx_sensitivity, speech_act)

        return PragmaticAnalysis(
            literal_meaning=literal,
            implied_meaning=implied,
            maxims_assessed=maxims,
            speech_act=speech_act,
            context_sensitivity=ctx_sensitivity,
            confidence=confidence,
        )

    # ─── Internal methods ─────────────────────────────────────────

    def _classify_speech_act(self, text_lower: str) -> str:
        """Classify the speech act of an utterance.

        Following Searle (1969), speech acts are the actions performed
        by speaking: asserting, requesting, promising, apologizing, etc.
        """
        # Check for question (ends with ? or starts with wh-word)
        if text_lower.endswith("?"):
            # Could be an indirect request
            for pattern in self._REQUEST_PATTERNS:
                if pattern in text_lower:
                    return "request"
            return "question"

        # Check for explicit request patterns
        for pattern in self._REQUEST_PATTERNS:
            if pattern in text_lower:
                return "request"

        # Check for apology
        for pattern in self._APOLOGY_PATTERNS:
            if pattern in text_lower:
                return "apology"

        # Check for promise
        for pattern in self._PROMISE_PATTERNS:
            if pattern in text_lower:
                return "promise"

        # Check for question-like patterns without ?
        for pattern in self._QUESTION_PATTERNS:
            if text_lower.startswith(pattern):
                return "question"

        # Default: assertion
        return "assertion"

    def _extract_literal_meaning(self, text: str) -> str:
        """Extract the surface-level literal meaning.

        For now, this is a simplified version that returns the text
        with a descriptive prefix. A full system would use semantic
        parsing.
        """
        text_clean = text.strip()
        if text_clean.endswith("?"):
            return f"The literal meaning is a question: {text_clean}"
        elif text_clean.endswith("!"):
            return f"The literal meaning is an exclamation: {text_clean}"
        else:
            return f"The literal meaning is a statement: {text_clean}"

    def _infer_implicature(
        self,
        text_lower: str,
        context: str,
        speech_act: str,
    ) -> str:
        """Infer what the speaker likely meant beyond the literal words.

        This is the core of pragmatic reasoning — detecting implicature.
        """
        context_lower = context.lower() if context else ""

        # Indirect requests: "It's cold in here" → "Close the window"
        indirect = self._infer_indirect_request_implicature(
            text_lower, context_lower, speech_act
        )
        if indirect is not None:
            return indirect

        # Questions that are really requests: "Can you pass the salt?"
        request_implicature = self._infer_request_question_implicature(
            text_lower, speech_act
        )
        if request_implicature is not None:
            return request_implicature

        # Understatement / Quantity violation
        understatement = self._infer_understatement_implicature(
            text_lower, speech_act
        )
        if understatement is not None:
            return understatement

        # If context mentions strong emotions but text is neutral
        emotion = self._infer_emotion_suppression_implicature(
            text_lower, context_lower
        )
        if emotion is not None:
            return emotion

        # No clear implicature detected
        return self._default_implicature(speech_act)

    def _infer_indirect_request_implicature(
        self,
        text_lower: str,
        context_lower: str,
        speech_act: str,
    ) -> str | None:
        """Detect indirect requests phrased as assertions.

        When an assertion contains an indirect-request indicator (e.g.
        "cold", "hungry") and the context supports it, the speaker is
        likely making an indirect request. Returns the implicature
        string, or None if no indirect request is detected.
        """
        if speech_act != "assertion":
            return None
        for indicator in self._INDIRECT_REQUEST_INDICATORS:
            if indicator in text_lower:
                # Check if context supports an indirect request
                if any(
                    ctx_word in context_lower
                    for ctx_word in (
                        "window",
                        "door",
                        "open",
                        "close",
                        "thermostat",
                        "temperature",
                        "hot",
                        "cold",
                        "stuffy",
                        "dark",
                        "bright",
                        "loud",
                        "quiet",
                        "drink",
                        "food",
                        "rest",
                        "sleep",
                    )
                ):
                    return (
                        "The speaker may be making an indirect request "
                        f"related to '{indicator}' — the literal statement "
                        "implies a desired action."
                    )
        return None

    def _infer_request_question_implicature(
        self,
        text_lower: str,
        speech_act: str,
    ) -> str | None:
        """Detect requests phrased as questions about ability.

        "Can you pass the salt?" is literally a question about ability,
        but pragmatically a request for action. Returns the implicature
        string, or None if not applicable.
        """
        if speech_act == "request":
            if text_lower.endswith("?"):
                return (
                    "Though phrased as a question about ability, this is "
                    "pragmatically a request for action."
                )
        return None

    def _infer_understatement_implicature(
        self,
        text_lower: str,
        speech_act: str,
    ) -> str | None:
        """Detect understatement from very brief assertions.

        A very short assertion may imply more than it literally states —
        the speaker may be understating. Returns the implicature string,
        or None if not applicable.
        """
        words = text_lower.split()
        if len(words) <= 3 and speech_act == "assertion":
            return (
                "The brevity of this utterance may imply more than it "
                "literally states — the speaker may be understating."
            )
        return None

    def _infer_emotion_suppression_implicature(
        self,
        text_lower: str,
        context_lower: str,
    ) -> str | None:
        """Detect emotional suppression when context is emotional but text is not.

        If the context mentions strong emotions but the utterance is
        emotionally neutral, the speaker may be suppressing or deflecting
        emotion. Returns the implicature string, or None if not applicable.
        """
        if not context_lower:
            return None
        emotion_words = {
            "angry",
            "sad",
            "happy",
            "scared",
            "worried",
            "frustrated",
            "excited",
            "disappointed",
        }
        ctx_has_emotion = any(w in context_lower for w in emotion_words)
        text_has_emotion = any(w in text_lower for w in emotion_words)
        if ctx_has_emotion and not text_has_emotion:
            return (
                "The context suggests emotional content, but the "
                "utterance is emotionally neutral — the speaker may "
                "be suppressing or deflecting emotion."
            )
        return None

    def _default_implicature(self, speech_act: str) -> str:
        """Return the default implicature when none is specifically detected."""
        if speech_act == "question":
            return "The speaker is seeking information."
        elif speech_act == "request":
            return "The speaker is requesting an action."
        elif speech_act == "apology":
            return "The speaker is expressing regret."
        elif speech_act == "promise":
            return "The speaker is committing to a future action."
        else:
            return "No strong implicature detected — the literal meaning appears sufficient."

    def _assess_maxims(self, text_lower: str, context: str) -> dict[str, dict]:
        """Assess each Gricean maxim.

        Returns a dict mapping maxim name to an assessment dict with
        'status' (satisfied/violated/flouted) and 'explanation'.
        """
        context_lower = context.lower() if context else ""
        words = text_lower.split()
        word_count = len(words)

        maxims: dict[str, dict] = {}

        # ── Quantity: informativeness ──────────────────────────
        maxims["quantity"] = self._assess_quantity_maxim(word_count)

        # ── Quality: truthfulness ──────────────────────────────
        maxims["quality"] = self._assess_quality_maxim(text_lower)

        # ── Relation: relevance ────────────────────────────────
        maxims["relation"] = self._assess_relation_maxim(
            text_lower, context_lower, word_count
        )

        # ── Manner: clarity ────────────────────────────────────
        maxims["manner"] = self._assess_manner_maxim(text_lower)

        return maxims

    def _assess_quantity_maxim(self, word_count: int) -> dict:
        """Assess the Quantity maxim (informativeness).

        Very brief utterances flout the maxim (implying disinterest or
        that the meaning is obvious); very long utterances violate it
        (over-informative). Otherwise the maxim is satisfied.
        """
        if word_count < 3:
            return {
                "status": "flouted",
                "explanation": (
                    "Very brief utterance — the speaker may be providing "
                    "less information than expected, which can imply "
                    "disinterest, hostility, or that the meaning is obvious."
                ),
            }
        elif word_count > 50:
            return {
                "status": "violated",
                "explanation": (
                    "Very long utterance — may be over-informative, "
                    "providing more detail than the listener needs."
                ),
            }
        else:
            return {
                "status": "satisfied",
                "explanation": "The utterance appears appropriately informative.",
            }

    def _assess_quality_maxim(self, text_lower: str) -> dict:
        """Assess the Quality maxim (truthfulness).

        Look for hedging or uncertainty markers. Hedging suggests the
        speaker is being careful about truthfulness. Without evidence of
        falsehood, the maxim is satisfied.
        """
        # Look for hedging or uncertainty markers
        hedge_words = {
            "maybe",
            "perhaps",
            "possibly",
            "might",
            "could",
            "i think",
            "i believe",
            "i guess",
            "probably",
        }
        has_hedge = any(h in text_lower for h in hedge_words)
        if has_hedge:
            return {
                "status": "satisfied",
                "explanation": (
                    "Hedging language suggests the speaker is being "
                    "careful about truthfulness — acknowledging uncertainty."
                ),
            }
        else:
            return {
                "status": "satisfied",
                "explanation": (
                    "No evidence of falsehood — the utterance appears to be offered as truthful."
                ),
            }

    def _assess_relation_maxim(
        self,
        text_lower: str,
        context_lower: str,
        word_count: int,
    ) -> dict:
        """Assess the Relation maxim (relevance).

        If context is available, check for word overlap between the text
        and context. No overlap with a sufficiently long utterance may
        indicate a topic change (flouting the maxim). Without context,
        relevance cannot be assessed.
        """
        if context_lower:
            # Check for word overlap between text and context
            text_words = set(text_lower.split())
            ctx_words = set(context_lower.split())
            overlap = text_words & ctx_words
            if not overlap and word_count > 2:
                return {
                    "status": "flouted",
                    "explanation": (
                        "The utterance shares no words with the context — "
                        "it may be a topic change, which can imply "
                        "discomfort with the current topic or a desire "
                        "to redirect."
                    ),
                }
            else:
                return {
                    "status": "satisfied",
                    "explanation": "The utterance appears relevant to the context.",
                }
        else:
            return {
                "status": "satisfied",
                "explanation": "No context provided — relevance cannot be assessed.",
            }

    def _assess_manner_maxim(self, text_lower: str) -> dict:
        """Assess the Manner maxim (clarity).

        Vague or ambiguous language (e.g. "thing", "stuff", "whatever")
        flouts the maxim, implying evasiveness or uncertainty. Otherwise
        the maxim is satisfied.
        """
        ambiguous_words = {
            "thing",
            "stuff",
            "whatever",
            "somehow",
            "somewhere",
            "someone",
            "something",
        }
        has_ambiguity = any(w in text_lower for w in ambiguous_words)
        if has_ambiguity:
            return {
                "status": "flouted",
                "explanation": (
                    "Vague or ambiguous language used — the speaker may "
                    "be intentionally unclear, which can imply "
                    "evasiveness or uncertainty."
                ),
            }
        else:
            return {
                "status": "satisfied",
                "explanation": "The utterance appears clear and unambiguous.",
            }

    def _compute_context_sensitivity(
        self,
        text_lower: str,
        context: str,
        speech_act: str,
    ) -> float:
        """Compute how much the meaning depends on context (0–1).

        Indirect speech acts and ambiguous language increase context
        sensitivity.
        """
        sensitivity = 0.3  # baseline

        # Indirect requests are highly context-dependent
        if speech_act == "request" and text_lower.endswith("?"):
            sensitivity += 0.3

        # Short utterances are more context-dependent
        word_count = len(text_lower.split())
        if word_count <= 5:
            sensitivity += 0.2

        # Ambiguous language increases context dependence
        ambiguous = {"thing", "stuff", "it", "that", "this", "whatever"}
        if any(w in text_lower.split() for w in ambiguous):
            sensitivity += 0.1

        # Indirect request indicators increase context dependence
        for indicator in self._INDIRECT_REQUEST_INDICATORS:
            if indicator in text_lower:
                sensitivity += 0.1
                break

        return min(1.0, sensitivity)

    def _compute_confidence(
        self,
        maxims: dict[str, dict],
        ctx_sensitivity: float,
        speech_act: str,
    ) -> float:
        """Compute overall confidence in the analysis.

        Confidence is higher when maxims are clearly satisfied or
        clearly flouted, and lower when context sensitivity is high
        (because the meaning is harder to pin down).
        """
        # Base confidence
        conf = 0.6

        # Clear maxim violations increase confidence in implicature
        flouted = sum(1 for m in maxims.values() if m["status"] == "flouted")
        conf += flouted * 0.1

        # High context sensitivity decreases confidence
        conf -= ctx_sensitivity * 0.2

        # Clear speech acts increase confidence
        if speech_act in ("request", "apology", "promise"):
            conf += 0.1

        return max(0.1, min(1.0, conf))

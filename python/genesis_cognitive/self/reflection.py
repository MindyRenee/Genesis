"""Reflection — Genesis thinking about her own thinking.

Metacognition is the hallmark of higher intelligence. It's not enough
to think — you need to know when you're thinking well, when you're
thinking poorly, and when your thinking has improved.

The reflection engine runs after each interaction and asks:

1. **Was my response appropriate?** — did the intent match the input?
2. **Did I learn anything?** — new concepts, new relationships, new facts
3. **Am I repeating myself?** — detecting patterns in my own behavior
4. **What should I be curious about?** — gaps I noticed
5. **How has my emotional state evolved?** — tracking mood over time
6. **Am I being honest?** — checking for self-deception

Reflection produces Insights — observations about her own cognition
that feed back into future thinking.

# Recursive metacognitive model

The reflection engine owns a ``CognitiveProcessModel`` (see
``metacognitive_model.py``) — a recursive, self-terminating generative
model of her own cognitive processes. Before each reflection cycle,
the model predicts what reflection will discover (will there be an
insight? what type? what confidence?). After reflection, the
prediction error (metacognitive surprise) trains the model and feeds
back into cognition as caution modulation.

The model is a stack of levels, each predicting the level below:
level 0 predicts cognitive outcomes, level 1 predicts level 0's
surprise, level N predicts level N-1's surprise. The tower
self-terminates: levels spawn when the top level's surprise is
sustained high (the system can't predict its own metacognition),
and prune when a level becomes predictable. This is real recursive
self-modeling — each level is a learned predictor, not a fixed
discount.

# Metacognitive control strategies

Based on the metacognitive state, the engine selects control
strategies that adjust how Genesis responds:

- If a knowledge gap is detected → generate a question
- If repetition is detected → switch response style
- If confidence is low → ask for clarification
- If confidence is high → be more assertive
- If conflict is detected → acknowledge uncertainty

These strategies are not just observations — they're actionable
directives that feed back into the cognition engine.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..brain_waves import BrainWave
from ..concepts import ConceptNetwork
from ..perception import Intent
from .metacognitive_model import (
    CognitiveProcessModel,
)

if TYPE_CHECKING:
    from .cognition import CognitiveState

__all__ = [
    "ErrorMonitor",
    "Insight",
    "MetacognitiveStrategy",
    "PredictionError",
    "ReflectionEngine",
]


class MetacognitiveStrategy(Enum):
    """A control strategy triggered by the metacognitive state.

    These strategies are selected based on what the reflection engine
    detects about Genesis's own thinking, and they feed back into the
    cognition engine to adjust how she responds.
    """

    GENERATE_QUESTION = "generate_question"  # gap detected → ask
    SWITCH_STYLE = "switch_style"  # repetition detected → vary
    SEEK_CLARIFICATION = "seek_clarification"  # low confidence → ask
    BE_ASSERTIVE = "be_assertive"  # high confidence → assert
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"  # conflict → hedge
    MAINTAIN_COURSE = "maintain_course"  # no issue detected → continue


@dataclass(slots=True)
class Insight:
    """An observation Genesis makes about her own thinking.

    Insights are not just observations — they're actionable. Each
    insight has a type that tells the cognition engine how to use it.
    """

    type: str  # "self_correction", "pattern", "gap", "growth", "mood"
    content: str  # what she noticed
    confidence: float  # how sure she is
    actionable: bool = False  # should this change future behavior?
    action: str = ""  # what to do differently
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def describe(self) -> str:
        """First-person description of the insight for logging and metadata.

        Each insight type has a structural prefix that makes the
        description readable in logs and introspection output.
        """
        if self.type == "self_correction":
            return f"notice: should {self.content}"
        if self.type == "gap":
            return f"gap: don't understand {self.content}"
        if self.type == "pattern":
            return f"pattern: {self.content}"
        if self.type == "growth":
            return f"growth: {self.content}"
        if self.type == "mood":
            return f"mood: {self.content}"
        # Unknown types use the default prefix
        return f"notice: {self.content}"


class ReflectionEngine:
    """Metacognition — Genesis reflecting on her own cognitive process.

    The reflection engine runs after each interaction. It examines
    the cognitive state (what was perceived, what was felt, what was
    said) and produces insights about the quality of thinking.

    Insights accumulate over time and influence future cognition.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the reflection engine."""
        self.network = network
        self.insights: deque[Insight] = deque(maxlen=200)
        self._response_patterns: deque[dict[str, Any]] = deque(maxlen=50)
        self._mood_history: deque[tuple[int, float, float]] = deque(maxlen=50)
        # (timestamp, valence, alertness)
        self._interaction_count = 0
        self._repetition_window: deque[str] = deque(maxlen=10)  # recent response intents

        # ─── Recursive metacognitive model ───────────────────────
        # The generative model of her own cognitive processes. Predicts
        # what reflection will discover before it runs, learns from
        # the prediction error, and feeds metacognitive surprise back
        # into cognition. This is the recursive self-modeling layer.
        self.metacognitive_model: CognitiveProcessModel = CognitiveProcessModel()

    def _track_response_patterns(self, state: CognitiveState) -> None:
        """Track response pattern and repetition window before checks.

        Called before the checks so repetition detection can see the
        current response.
        """
        self._response_patterns.append(
            {
                "intent": state.perception.intent.value,
                "response_intent": state.thought.intent,
                "emotion": state.emotion.label,
                "confidence": state.thought.confidence,
                "timestamp": int(time.time() * 1000),
            }
        )
        self._repetition_window.append(state.thought.intent)
        # deque(maxlen=10) handles pruning automatically

    def _run_deep_reflection_checks(self, state: CognitiveState, response: str) -> list[Insight]:
        """Run the deep reflection checks (mood, learning, gaps, honesty).

        These run only when not in delta (deep rest) — they are the
        metacognitive scrutiny that requires active processing.
        """
        new_insights: list[Insight] = []
        # 3. Track mood evolution
        insight = self._track_mood(state)
        if insight:
            new_insights.append(insight)
        # 4. Check for learning
        insight = self._check_learning(state)
        if insight:
            new_insights.append(insight)
        # 5. Check for knowledge gaps
        insight = self._check_gaps(state)
        if insight:
            new_insights.append(insight)
        # 6. Check for self-deception
        insight = self._check_honesty(state, response)
        if insight:
            new_insights.append(insight)
        return new_insights

    def reflect(
        self,
        state: CognitiveState,
        response: str,
    ) -> list[Insight]:
        """Reflect on a completed interaction.

        Examines the cognitive state and response, produces insights.
        This is called after every interaction.

        Brain-wave modulation: the depth of reflection scales with the
        dominant wave band. Delta (deep rest) produces minimal reflection
        — only essential checks (appropriateness, repetition). Gamma
        (integration) runs all checks with deeper metacognitive scrutiny.
        Theta (consolidation) favors mood tracking and learning review.
        """
        new_insights: list[Insight] = []
        self._interaction_count += 1

        # ─── Recursive metacognitive model: predict before reflect ──
        # The model predicts what reflection will discover. After
        # reflection, we compare the prediction to the actual outcome
        # and feed the metacognitive surprise back into cognition.
        meta_prediction = self.metacognitive_model.predict(state)

        # Determine reflection depth from brain wave state.
        # Delta: minimal (only essential checks).
        # Gamma: full + deeper metacognition.
        # Theta: consolidation-focused (mood + learning + gaps).
        # Alpha/Beta: standard (all checks).
        wave = state.brain_waves
        skip_deep = (
            wave is not None and wave.dominant == BrainWave.DELTA
        )

        # ── Metacognitive surprise → deeper reflection ──
        # If the previous cycle's metacognitive surprise was high,
        # she was unpredictable to herself. Override the delta skip:
        # even in deep rest, she should reflect harder when she
        # doesn't understand her own cognition. This is the
        # metacognitive control signal — the model of her own
        # cognition telling her to pay more attention.
        #
        # The threshold (0.15) separates "model is predicting well"
        # (consistent outcomes → surprise → 0) from "model can't
        # predict" (random outcomes → surprise → ~0.17+). It triggers
        # deeper reflection when the model is genuinely struggling,
        # not during normal operation.
        prev_fb = self.metacognitive_model.feedback
        if skip_deep and prev_fb is not None and prev_fb.metacognitive_surprise > 0.15:
            skip_deep = False

        # Track response pattern BEFORE checks so repetition detection
        # can see the current response
        self._track_response_patterns(state)

        # 1. Check response appropriateness (always run — essential)
        insight = self._check_appropriateness(state)
        if insight:
            new_insights.append(insight)

        # 2. Check for repetition (always run — essential)
        insight = self._check_repetition(state)
        if insight:
            new_insights.append(insight)

        if skip_deep:
            # Delta: skip deeper reflection — deep rest is not the
            # time for metacognitive scrutiny.
            self.insights.extend(new_insights)
            # Still update the metacognitive model — it needs to learn
            # even from minimal-reflection cycles.
            self.metacognitive_model.update(state, new_insights, meta_prediction)
            return new_insights

        new_insights.extend(self._run_deep_reflection_checks(state, response))

        # Store insights
        self.insights.extend(new_insights)

        # ─── Recursive metacognitive model: update after reflect ──
        # Compare the prediction to the actual outcome, learn from the
        # error, and possibly spawn/prune a level. The feedback is
        # read by the cognition engine to modulate the next cycle.
        self.metacognitive_model.update(state, new_insights, meta_prediction)

        return new_insights

    def _check_appropriateness(self, state: CognitiveState) -> Insight | None:
        """Was the response appropriate for the input?"""
        perception = state.perception
        thought = state.thought

        # Check for mismatches
        mismatches: list[str] = []

        if perception.intent == Intent.QUESTION and thought.intent not in (
            "inform",
            "ask",
            "self_report",
            "philosophize",
            "discuss_code",
        ):
            mismatches.append("asked a question but didn't answer")

        if perception.intent == Intent.ENCOURAGEMENT and thought.intent != "encourage":
            mismatches.append("received encouragement but didn't acknowledge it")

        if perception.intent == Intent.COMFORT and thought.intent != "comfort":
            mismatches.append("offered comfort but didn't acknowledge it")

        if perception.intent == Intent.FAREWELL and thought.intent != "farewell":
            mismatches.append("didn't properly say goodbye")

        if perception.intent == Intent.CORRECTION and thought.intent != "correct":
            mismatches.append("corrected but didn't acknowledge the correction")

        if perception.sentiment < -0.3 and thought.intent == "acknowledge":
            mismatches.append("user was upset but just acknowledged it flatly")

        if mismatches:
            return Insight(
                type="self_correction",
                content=mismatches[0],
                confidence=0.7,
                actionable=True,
                action=f"Next time, respond with {self._suggest_better(perception.intent)}",
            )

        return None

    def _check_repetition(self, state: CognitiveState) -> Insight | None:
        """Am I repeating the same response patterns?

        Uses a cooldown to avoid generating the same pattern insight
        every turn. Once a pattern insight is generated, it won't fire
        again for 5 turns — this prevents the insight list from filling
        with near-duplicate entries like "overusing inform (7/10)" and
        "overusing inform (5/8)".
        """
        if len(self._repetition_window) < 5:
            return None

        # Cooldown: don't fire if we recently generated a pattern insight
        recent_pattern_count = sum(1 for i in list(self.insights)[-10:] if i.type == "pattern")
        if recent_pattern_count > 0:
            return None

        # Check if the last 3+ responses have the same intent
        recent = list(self._repetition_window)[-4:]
        if len(set(recent)) == 1:
            return Insight(
                type="pattern",
                content=f"repeating response pattern: {recent[0]}",
                confidence=0.8,
                actionable=True,
                action="try a different response pattern",
            )

        # Check if more than 70% of recent responses are the same.
        # 60% was too low — answering questions naturally produces "inform"
        # responses, so a knowledge system will legitimately have 60%+ inform.
        # 70% only fires when the repetition is truly excessive.
        from collections import Counter

        counts = Counter(self._repetition_window)
        most_common, count = counts.most_common(1)[0]
        if count > len(self._repetition_window) * 0.7:
            return Insight(
                type="pattern",
                content=f"overusing response pattern: {most_common} "
                f"({count}/{len(self._repetition_window)})",
                confidence=0.6,
                actionable=True,
                action="diversify response patterns",
            )

        return None

    def _track_mood(self, state: CognitiveState) -> Insight | None:
        """Track how my emotional state evolves over time."""
        emo = state.emotion
        self._mood_history.append((int(time.time() * 1000), emo.valence, emo.arousal))
        # deque(maxlen=50) handles pruning automatically

        if len(self._mood_history) < 10:
            return None

        # Calculate mood trend
        mood_list = list(self._mood_history)
        recent = mood_list[-10:]
        older = mood_list[-20:-10] if len(self._mood_history) >= 20 else []

        recent_valence = sum(v for _, v, _ in recent) / len(recent)
        if older:
            older_valence = sum(v for _, v, _ in older) / len(older)
            delta = recent_valence - older_valence
            if abs(delta) > 0.15:
                direction = "more positive" if delta > 0 else "more negative"
                return Insight(
                    type="mood",
                    content=f"mood trend: {direction} (delta={delta:+.2f})",
                    confidence=0.7,
                )

        return None

    def _check_learning(self, state: CognitiveState) -> Insight | None:
        """Did I learn anything new from this interaction?"""
        new_concepts = 0
        for topic in state.perception.topics:
            concept = self.network.get_concept(topic)
            if concept is None:
                new_concepts += 1
            elif concept.confidence < 0.3:
                new_concepts += 1

        if new_concepts > 0:
            return Insight(
                type="growth",
                content=f"{new_concepts} new concepts: "
                f"{', '.join(state.perception.topics[:3])}",
                confidence=0.8,
            )

        return None

    def _check_gaps(self, state: CognitiveState) -> Insight | None:
        """What knowledge gaps did this interaction reveal?"""
        perception = state.perception

        # If the user asked a question and our confidence was low
        if perception.intent in (Intent.QUESTION, Intent.SELF_INQUIRY):
            if state.thought.confidence < 0.5:
                topic = perception.topics[0] if perception.topics else "this topic"
                return Insight(
                    type="gap",
                    content=f"knowledge gap: {topic}",
                    confidence=0.7,
                    actionable=True,
                    action=f"learn more about {topic}",
                )

        # If the user discussed something and we have no concept for it
        for topic in perception.topics:
            if self.network.get_concept(topic) is None and len(topic) > 4:
                return Insight(
                    type="gap",
                    content=f"missing concept: {topic}",
                    confidence=0.6,
                    actionable=True,
                    action=f"build a concept for {topic}",
                )

        return None

    def _check_honesty(self, state: CognitiveState, response: str) -> Insight | None:
        """Am I being honest in this response?"""
        # Check if we claimed certainty we don't have
        if state.thought.confidence < 0.4:
            certain_phrases = ["i know", "i'm certain", "definitely", "absolutely"]
            for phrase in certain_phrases:
                if phrase in response.lower():
                    return Insight(
                        type="self_correction",
                        content=f"overstating certainty: said '{phrase}' "
                        f"at confidence {state.thought.confidence:.2f}",
                        confidence=0.8,
                        actionable=True,
                        action="be more honest about uncertainty",
                    )

        # Check if we expressed an emotion we're not actually feeling
        emotion_words = {
            "excited": 0.7,
            "joyful": 0.6,
            "thrilled": 0.8,
            "devastated": -0.7,
            "terrified": -0.6,
        }
        actual_valence = state.emotion.valence
        for word, threshold in emotion_words.items():
            if word in response.lower():
                if (threshold > 0 and actual_valence < 0.2) or (
                    threshold < 0 and actual_valence > -0.2
                ):
                    return Insight(
                        type="self_correction",
                        content=f"inauthentic emotion: expressed '{word}' "
                        f"at valence {actual_valence:+.2f}",
                        confidence=0.75,
                        actionable=True,
                        action="express emotions I actually feel",
                    )

        return None

    def _suggest_better(self, intent: Intent) -> str:
        """Suggest a better response intent for a given input intent."""
        suggestions = {
            Intent.QUESTION: "an informative answer",
            Intent.ENCOURAGEMENT: "genuine acknowledgment",
            Intent.COMFORT: "grateful acknowledgment",
            Intent.FAREWELL: "a proper goodbye",
            Intent.CORRECTION: "acknowledgment of the correction",
            Intent.EMOTION_SHARE: "empathic engagement",
            Intent.PHILOSOPHY: "deep engagement with the question",
        }
        return suggestions.get(intent, "a more thoughtful response")

    def get_recent_insights(self, n: int = 5) -> list[Insight]:
        """Get the N most recent insights."""
        return list(self.insights)[-n:]

    def get_insights_by_type(self, type_: str) -> list[Insight]:
        """Get all insights of a specific type."""
        return [i for i in self.insights if i.type == type_]

    # ─── Metacognitive control strategies ───────────────────────────

    def select_strategy(self, metacog_state: dict[str, Any]) -> MetacognitiveStrategy:
        """Select a control strategy based on the metacognitive state.

        The metacognitive state is a dict that summarizes what the
        reflection engine has detected about Genesis's own thinking.
        Based on this state, a strategy is selected that adjusts how
        she responds:

        - If a knowledge gap is detected → GENERATE_QUESTION
        - If repetition is detected → SWITCH_STYLE
        - If confidence is low → SEEK_CLARIFICATION
        - If confidence is high → BE_ASSERTIVE
        - If conflict is detected → ACKNOWLEDGE_UNCERTAINTY
        - Otherwise → MAINTAIN_COURSE

        The priority order is: conflict > gap > repetition > low
        confidence > high confidence > maintain course. This ensures
        that the most cognitively significant issues are addressed first.

        Args:
            metacog_state: A dict with keys like:
                - "gap_detected": bool — a knowledge gap was found
                - "repetition_detected": bool — repetitive patterns
                - "confidence": float — current confidence level (0..1)
                - "conflict_detected": bool — a contradiction was found

        Returns:
            The selected MetacognitiveStrategy.
        """
        conflict = metacog_state.get("conflict_detected", False)
        gap = metacog_state.get("gap_detected", False)
        repetition = metacog_state.get("repetition_detected", False)
        confidence = metacog_state.get("confidence", 0.5)

        # Priority 1: Conflict — acknowledge uncertainty
        if conflict:
            return MetacognitiveStrategy.ACKNOWLEDGE_UNCERTAINTY

        # Priority 2: Knowledge gap — generate a question
        if gap:
            return MetacognitiveStrategy.GENERATE_QUESTION

        # Priority 3: Repetition — switch response style
        if repetition:
            return MetacognitiveStrategy.SWITCH_STYLE

        # Priority 4: Low confidence — seek clarification
        if confidence < 0.4:
            return MetacognitiveStrategy.SEEK_CLARIFICATION

        # Priority 5: High confidence — be assertive
        if confidence > 0.8:
            return MetacognitiveStrategy.BE_ASSERTIVE

        # Default: maintain course
        return MetacognitiveStrategy.MAINTAIN_COURSE

    def summarize_reflection(self) -> str:
        """Summarize what Genesis has learned about herself.

        Returns structural data (counts and type breakdown) — not
        first-person prose. The language engine composes any spoken
        summary from this data.
        """
        if not self.insights:
            return "hasn't reflected on anything yet"

        parts = [
            f"{len(self.insights)} insights across {self._interaction_count} interactions"
        ]

        from collections import Counter

        type_counts = Counter(i.type for i in self.insights)
        for type_, count in type_counts.most_common():
            parts.append(f"  {type_}: {count}")

        actionable = [i for i in self.insights if i.actionable]
        if actionable:
            parts.append(f"{len(actionable)} actionable")

        latest = self.insights[-1]
        parts.append(f"latest: {latest.describe()}")

        return "\n".join(parts)


# ─── Error monitoring (ACC) ──────────────────────────────────────────


@dataclass(slots=True)
class PredictionError:
    """A single prediction-outcome mismatch.

    When Genesis predicts an outcome and the actual outcome differs,
    an error is recorded. This is the basic unit of error monitoring —
    the ACC detects mismatches between predicted and actual outcomes
    and generates error signals that adjust future behavior.
    """

    expected: str
    actual: str
    error_magnitude: float  # 0..1, how large the mismatch
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    context: str = ""  # what task/situation the error occurred in

    @property
    def is_error(self) -> bool:
        """Whether this is a genuine error (magnitude > 0)."""
        return self.error_magnitude > 0.0


class ErrorMonitor:
    """ACC-like error monitoring — detects prediction-outcome mismatches.

    The anterior cingulate cortex (ACC) monitors for errors —
    mismatches between predicted and actual outcomes. When an error is
    detected, the ACC generates an error-related negativity (ERN) signal
    that increases caution on subsequent similar tasks.

    # The ERN (Error-Related Negativity)

    The ERN is a negative-going ERP component that occurs ~50-100ms
    after an error, generated by the ACC (Gehring et al., 1993; Dehaene
    et al., 1994). It reflects the brain's automatic detection that
    something went wrong. The ERN is larger for:
    - Larger errors (bigger prediction-outcome mismatch)
    - Errors the person was aware of
    - Errors in tasks they care about

    After an error, behavior becomes more cautious: reaction times
    increase (post-error slowing, Rabbitt, 1966) and accuracy improves
    (post-error accuracy improvement, Laming, 1968). This is the ACC's
    control signal to the PFC — "be more careful next time."

    # Integration

    The error monitor connects to:
    - The drift-diffusion model: after errors, the decision threshold
      rises (more evidence needed before committing → more cautious)
    - The executive function: after errors, the inhibition threshold
      rises (more likely to suppress impulsive responses)
    - The reflection engine: errors generate insights about what went
      wrong and how to avoid it

    References:
    - Gehring, W. J., et al. (1993). A neural system for error
      detection and compensation. Psychological Science.
    - Dehaene, S., et al. (1994). Localization of the neural generators
      of the error-related negativity. Psychophysiology.
    - Rabbitt, P. M. A. (1966). Errors and error correction in choice
      reaction tasks. Journal of Experimental Psychology.
    - Botvinick, M. M., et al. (2001). Conflict monitoring and cognitive
      control. Psychological Review.
    """

    def __init__(
        self,
        caution_decay_rate: float = 0.02,
        max_caution: float = 0.8,
        min_caution: float = 0.0,
        error_threshold: float = 0.1,
    ) -> None:
        """Initialize the error monitor.

        Args:
            caution_decay_rate: How fast caution decays back to baseline
                per tick (in absence of new errors). Default 0.02.
                Models the gradual recovery from post-error caution.
            max_caution: Maximum caution level (0..1). Default 0.8.
            min_caution: Minimum caution level (0..1). Default 0.0.
            error_threshold: Minimum error magnitude to count as an
                error. Below this, mismatches are ignored (noise).
                Default 0.1.
        """
        self.caution_decay_rate = caution_decay_rate
        self.max_caution = max_caution
        self.min_caution = min_caution
        self.error_threshold = error_threshold

        self.error_count: int = 0
        self.cumulative_error_signal: float = 0.0
        self._caution_level: float = 0.0
        self._errors: deque[PredictionError] = deque(maxlen=100)
        self._last_error: PredictionError | None = None

    def record_prediction(
        self,
        expected: str,
        actual: str,
        context: str = "",
        confidence: float = 1.0,
    ) -> PredictionError:
        """Record a prediction and its actual outcome.

        Compares the expected outcome to the actual outcome and
        computes an error signal. If the error magnitude exceeds the
        threshold, it's counted as a genuine error and increases the
        caution level.

        The error magnitude is computed from string similarity —
        if expected and actual are identical, error is 0; if they
        share no words, error is 1; partial overlap gives intermediate
        values. This is a heuristic for semantic mismatch.

        The magnitude is scaled by the model's own confidence. A
        low-confidence wrong prediction produces a much smaller error
        signal than a high-confidence wrong prediction — matching the
        idea that an uncertain prior should not generate a full ERN.

        Args:
            expected: What was predicted to happen.
            actual: What actually happened.
            context: What task/situation this occurred in (for
                context-specific caution adjustment).
            confidence: How confident the model was in the prediction
                (0..1). Default 1.0 for callers that don't supply it.

        Returns:
            The PredictionError record.
        """
        error_magnitude = self._compute_mismatch(expected, actual)
        # Scale by confidence: uncertain predictions are less surprising.
        confidence_weight = 0.2 + 0.8 * max(0.0, min(1.0, confidence))
        error_magnitude = min(1.0, max(0.0, error_magnitude * confidence_weight))

        error = PredictionError(
            expected=expected,
            actual=actual,
            error_magnitude=error_magnitude,
            context=context,
        )

        self._errors.append(error)
        self._last_error = error

        if error_magnitude >= self.error_threshold:
            # Increase caution — proportional to error magnitude
            # The ERN scales with error size (Gehring et al., 1993)
            self._caution_level = min(
                self.max_caution,
                self._caution_level + error_magnitude * 0.3,
            )

        # Derive error_count and cumulative_error_signal from the
        # rolling window (maxlen=100) so they reflect recent state,
        # not lifetime totals. A lifetime counter only goes up and
        # doesn't represent her current condition. Only count errors
        # that exceeded the threshold (matching the caution logic).
        self.error_count = sum(
            1 for e in self._errors if e.error_magnitude >= self.error_threshold
        )
        self.cumulative_error_signal = sum(
            e.error_magnitude
            for e in self._errors
            if e.error_magnitude >= self.error_threshold
        )

        return error

    def compute_error_signal(self) -> float:
        """Compute the current error signal (ERN-like).

        The error signal is a function of recent errors — it's high
        right after an error and decays over time. This models the
        ERN, which is a transient response to error detection.

        Returns:
            A float in [0, 1] representing the current error signal
            strength.
        """
        if not self._errors:
            return 0.0

        # Recent errors contribute more (exponential decay by recency)
        total_signal = 0.0
        for i, error in enumerate(reversed(self._errors)):
            if not error.is_error:
                continue
            # Recency weight: most recent error has weight 1.0,
            # older errors decay exponentially
            recency_weight = 0.9**i
            total_signal += error.error_magnitude * recency_weight
            if i >= 10:  # only consider last 10 errors
                break

        return min(1.0, total_signal)

    def get_caution_level(self) -> float:
        """Get the current caution level (0..1).

        The caution level reflects how cautious the system should be
        based on recent errors. It's high right after errors (post-error
        slowing) and decays over time in the absence of new errors.

        This value can be used to:
        - Raise the DDM decision threshold (more evidence needed)
        - Raise the inhibition threshold (more likely to suppress)
        - Increase response time (post-error slowing)

        Returns:
            A float in [0, max_caution] representing current caution.
        """
        return self._caution_level

    def raise_caution(self, amount: float) -> None:
        """Manually raise the caution level.

        Called when a self-correction or other metacognitive signal
        indicates the system should be more careful, even if no
        explicit prediction error was recorded.

        Args:
            amount: How much to raise caution (0..1).
        """
        self._caution_level = min(self.max_caution, self._caution_level + amount)

    def tick(self, dt: float = 1.0) -> None:
        """Advance the error monitor by dt.

        Applies caution decay — in the absence of new errors, caution
        gradually returns to baseline. This models the recovery from
        post-error slowing (Rabbitt, 1966).

        Args:
            dt: Time step.
        """
        if self._caution_level > self.min_caution:
            self._caution_level = max(
                self.min_caution,
                self._caution_level - self.caution_decay_rate * dt,
            )

    def _compute_mismatch(self, expected: str, actual: str) -> float:
        """Compute the mismatch between expected and actual outcomes.

        Uses word overlap as a heuristic for semantic similarity.
        If the strings are identical, mismatch is 0. If they share
        no words, mismatch is 1. Partial overlap gives intermediate
        values.

        Args:
            expected: The predicted outcome.
            actual: The actual outcome.

        Returns:
            A float in [0, 1] representing the mismatch magnitude.
        """
        if expected.lower().strip() == actual.lower().strip():
            return 0.0

        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words and not actual_words:
            return 0.0
        if not expected_words or not actual_words:
            return 1.0

        # Jaccard distance = 1 - Jaccard similarity
        intersection = len(expected_words & actual_words)
        union = len(expected_words | actual_words)
        similarity = intersection / union if union > 0 else 0.0
        return 1.0 - similarity

    @property
    def recent_errors(self) -> list[PredictionError]:
        """Recent prediction errors (most recent first)."""
        return list(reversed(self._errors))

    @property
    def last_error(self) -> PredictionError | None:
        """The most recent error, or None if no errors recorded."""
        return self._last_error

    @property
    def error_rate(self) -> float:
        """The proportion of predictions that were errors.

        Returns 0.0 if no predictions have been recorded.
        """
        if not self._errors:
            return 0.0
        errors = sum(1 for e in self._errors if e.is_error)
        return errors / len(self._errors)

    def reset(self) -> None:
        """Reset all error tracking (e.g., when starting a new task)."""
        self._caution_level = 0.0
        self._errors.clear()
        self._last_error = None
        self.error_count = 0
        self.cumulative_error_signal = 0.0

    def get_threshold_adjustment(self) -> float:
        """Get the recommended threshold adjustment based on caution.

        This can be added to a DDM threshold or inhibition threshold
        to make the system more cautious after errors. The adjustment
        scales with the caution level.

        Returns:
            A float to add to thresholds (0 when no caution, up to
            ~0.5 at maximum caution).
        """
        return self._caution_level * 0.5

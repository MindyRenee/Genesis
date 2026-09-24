"""Decision-making — selecting among alternatives based on evidence,
goals, uncertainty, and values.

Genesis had a generation pipeline: perceive → classify intent → match
rule in priority tree → generate response → modulate confidence. It
did not have a decision process: generate alternatives → evaluate
against criteria → select the best. Every "decision-making" component
(DDM, executive, TD learning, probabilistic reasoning) was wired as a
modulator of an already-chosen response, not as a selector among
alternatives.

This module closes that gap with a **decision engine** that:

1. **Generates candidate actions** — for each turn, enumerate the
   plausible actions Genesis could take (answer, ask for clarification,
   reflect, acknowledge, etc.).

2. **Evaluates each candidate against multiple criteria** — evidence
   strength, goal relevance, uncertainty, expected value, and a value
   framework (accuracy, honesty, helpfulness, curiosity, safety).

3. **Selects the best** — the candidate with the highest multi-criteria
   score wins. This replaces the first-match-wins deliberation tree
   with genuine selection among alternatives.

4. **Switches actions under uncertainty** — when confidence in the
   primary action is low and uncertainty is high, the engine switches
   to a more appropriate action (e.g., from answering to asking for
   clarification).

5. **Consults the TD value function** — before committing, the engine
   queries ``predict_value()`` for the candidate's resulting state,
   incorporating learned value into the decision.

6. **Applies executive inhibition** — the executive's
   ``inhibit_response()`` can suppress a prepotent response, allowing
   the system to withhold action when appropriate.

## Integration

The engine is wired into the cognition engine as
``self.decision_engine`` and called after deliberation. The
deliberation tree still produces a *default* action (preserving
traceability), but the decision engine can override it when another
candidate scores higher. The DDM then accumulates real evidence for
each competing option, and the winner actually determines the action.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork
    from ..executive import ExecutiveFunction
    from ..learning.td import TDLearner
    from .engine import ReasoningResult

__all__ = ["ActionType", "CandidateAction", "DecisionEngine", "DecisionOutcome"]

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """The kinds of actions Genesis can take in response to input.

    These are the *action types* — the high-level decision of what to
    do. The language engine then generates the specific words for the
    chosen action.
    """

    ANSWER = "answer"          # provide information
    ASK = "ask"                # ask for clarification
    REFLECT = "reflect"        # think aloud / introspect
    ACKNOWLEDGE = "acknowledge" # acknowledge without elaboration
    GREET = "greet"            # return a greeting
    EMPATHIZE = "empathize"    # respond to emotional content
    INVESTIGATE = "investigate" # pursue a goal / research
    WITHHOLD = "withhold"      # say nothing (inhibited)


# Default action candidates for each input context.
_DEFAULT_CANDIDATES = [
    ActionType.ANSWER,
    ActionType.ASK,
    ActionType.REFLECT,
    ActionType.ACKNOWLEDGE,
]

# Value framework criteria with default weights.
# These represent Genesis's core values — the criteria against which
# it evaluates candidate actions. The weights encode their relative
# importance. Higher weight = more important criterion.
_DEFAULT_VALUE_WEIGHTS: dict[str, float] = {
    "accuracy": 0.25,      # is the response factually correct?
    "honesty": 0.20,       # does it express genuine uncertainty?
    "helpfulness": 0.20,   # does it advance the user's goal?
    "curiosity": 0.10,     # does it advance Genesis's own learning?
    "safety": 0.15,        # does it avoid harm?
    "coherence": 0.10,     # is it coherent with prior context?
}

# Uncertainty threshold for action switching.
# If the primary action's confidence is below this AND uncertainty is
# above this, switch to a more appropriate action.
_UNCERTAINTY_SWITCH_THRESHOLD = 0.35

# Inhibition threshold — if the best candidate's normalized score is
# below this, the response is inhibited (withheld). The composite score
# is normalized to [0, 1] by dividing by the total weight sum, so this
# threshold is on a [0, 1] scale. The minimum achievable normalized
# score is ~0.45 (ASK with no evidence gap and low uncertainty), so
# 0.50 catches the absolute weakest candidates — cases where the
# system genuinely has almost nothing useful to say, not even a
# clarifying question.
_INHIBITION_THRESHOLD = 0.50


@dataclass(slots=True)
class CandidateAction:
    """A candidate action evaluated by the decision engine.

    Each candidate has an action type, a set of criteria scores, and a
    composite score that combines them. The candidate with the highest
    composite score is selected.
    """

    action_type: ActionType
    criteria_scores: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    confidence: float = 0.5
    rationale: str = ""
    td_value: float = 0.0
    inhibited: bool = False

    def score(self, criterion: str) -> float:
        """Get the score for a specific criterion."""
        return self.criteria_scores.get(criterion, 0.0)


@dataclass(slots=True)
class DecisionOutcome:
    """The result of a decision — which action was selected and why.

    Contains the selected action, all evaluated candidates (for
    introspection), and the reasoning behind the selection.
    """

    selected: CandidateAction
    candidates: list[CandidateAction]
    overridden_default: bool  # did the decision override the deliberation tree?
    inhibition_applied: bool = False
    uncertainty_switched: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def action_type(self) -> ActionType:
        """The selected action type."""
        return self.selected.action_type

    @property
    def confidence(self) -> float:
        """The confidence in the selected action."""
        return self.selected.confidence

    def describe(self) -> str:
        """Human-readable summary of the decision."""
        lines = [
            f"Decision: {self.selected.action_type.value} "
            f"(score={self.selected.composite_score:.3f}, "
            f"confidence={self.selected.confidence:.2f})",
        ]
        if self.overridden_default:
            lines.append("  (overrode deliberation default)")
        if self.inhibition_applied:
            lines.append("  (response inhibited)")
        if self.uncertainty_switched:
            lines.append("  (switched due to uncertainty)")
        # Show top candidates.
        ranked = sorted(
            self.candidates, key=lambda c: c.composite_score, reverse=True,
        )
        for c in ranked[:3]:
            marker = " ←" if c is self.selected else ""
            lines.append(
                f"  {c.action_type.value}: "
                f"score={c.composite_score:.3f}, "
                f"conf={c.confidence:.2f}{marker}"
            )
        if self.notes:
            lines.append(f"  notes: {'; '.join(self.notes)}")
        return "\n".join(lines)


class DecisionEngine:
    """Selects among candidate actions based on multiple criteria.

    Given a context (perception intent, reasoning results, goals,
    uncertainty), the engine:

    1. Generates candidate actions.
    2. Evaluates each against multiple criteria (evidence strength,
       goal relevance, uncertainty, expected value, values).
    3. Optionally consults the TD value function for each candidate.
    4. Selects the highest-scoring candidate.
    5. Applies executive inhibition if the best score is too low.
    6. Switches actions under high uncertainty.

    This replaces the first-match-wins deliberation tree with genuine
    multi-criteria selection among alternatives.

    Usage::

        decision = DecisionEngine(network, executive, td_learner)
        result = decision.decide(
            default_action=ActionType.ANSWER,
            reasoning_results=results,
            confidence=0.7,
            uncertainty=0.2,
            goals=["understand photosynthesis"],
            topics=["photosynthesis"],
        )
        if result.action_type == ActionType.ASK:
            # Genesis decided to ask for clarification instead of answering
            ...
    """

    def __init__(
        self,
        network: ConceptNetwork,
        executive: ExecutiveFunction | None = None,
        td_learner: TDLearner | None = None,
        value_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the decision engine.

        Args:
            network: The concept network — for checking concept
                existence and confidence.
            executive: The executive function — for response inhibition.
                If None, inhibition is skipped.
            td_learner: The TD learner — for value lookup before
                commitment. If None, TD values are not consulted.
            value_weights: Custom weights for the value criteria. If
                None, default weights are used.
        """
        self.network = network
        self.executive = executive
        self.td_learner = td_learner
        self.value_weights = value_weights or dict(_DEFAULT_VALUE_WEIGHTS)
        self._decisions: list[DecisionOutcome] = []
        self._overrides: int = 0
        self._inhibitions: int = 0
        self._uncertainty_switches: int = 0

    @property
    def decisions(self) -> list[DecisionOutcome]:
        """All decisions ever made (copy)."""
        return list(self._decisions)

    @property
    def override_count(self) -> int:
        """Times the decision overrode the deliberation default."""
        return self._overrides

    @property
    def inhibition_count(self) -> int:
        """Times a response was inhibited."""
        return self._inhibitions

    @property
    def uncertainty_switch_count(self) -> int:
        """Times an action was switched due to uncertainty."""
        return self._uncertainty_switches

    # ─── Public API ───────────────────────────────────────────────

    def decide(
        self,
        default_action: ActionType,
        reasoning_results: list[ReasoningResult] | None = None,
        confidence: float = 0.7,
        uncertainty: float = 0.2,
        goals: list[str] | None = None,
        topics: list[str] | None = None,
        perception_intent: str = "",
    ) -> DecisionOutcome:
        """Make a decision among candidate actions.

        This is the main entry point. Given a default action (from the
        deliberation tree), the engine evaluates all candidates and
        selects the best one.

        Args:
            default_action: The action the deliberation tree chose.
            reasoning_results: The reasoning results available.
            confidence: Confidence in the default action.
            uncertainty: Uncertainty about the response (0..1).
            goals: Active goals.
            topics: Topics in the current context.
            perception_intent: The perceived intent of the input.

        Returns:
            A DecisionOutcome with the selected action and rationale.
        """
        reasoning_results = reasoning_results or []
        goals = goals or []
        topics = topics or []

        # 1. Generate candidates.
        candidates = self._generate_candidates(
            default_action, perception_intent, reasoning_results,
        )

        # 2. Evaluate each candidate.
        for candidate in candidates:
            self._evaluate_candidate(
                candidate,
                reasoning_results=reasoning_results,
                confidence=confidence,
                uncertainty=uncertainty,
                goals=goals,
                topics=topics,
                perception_intent=perception_intent,
            )

        # 3. Consult TD value function for each candidate.
        if self.td_learner is not None:
            for candidate in candidates:
                candidate.td_value = self._lookup_td_value(
                    candidate, topics,
                )
                # TD value contributes to composite score.
                candidate.composite_score += candidate.td_value * 0.1

        # 4. Select the best candidate.
        selected = max(candidates, key=lambda c: c.composite_score)
        overridden = selected.action_type != default_action

        # 5. Check for uncertainty-driven action switching.
        uncertainty_switched = False
        if (
            selected.action_type == ActionType.ANSWER
            and confidence < _UNCERTAINTY_SWITCH_THRESHOLD
            and uncertainty > _UNCERTAINTY_SWITCH_THRESHOLD
        ):
            # Low confidence + high uncertainty → switch to asking.
            ask_candidate = next(
                (c for c in candidates if c.action_type == ActionType.ASK),
                None,
            )
            if ask_candidate is not None:
                selected = ask_candidate
                uncertainty_switched = True
                if not overridden:
                    overridden = True

        # 6. Apply executive inhibition.
        # The impulse strength is the candidate's composite score — a
        # weak candidate has a weak impulse, which is easily inhibited by
        # the executive's stop-signal threshold. Passing ``1.0 - score``
        # would invert the semantics: weak candidates would get strong
        # impulses and never be inhibited, making this path dead code.
        inhibition_applied = False
        if (
            self.executive is not None
            and selected.composite_score < _INHIBITION_THRESHOLD
        ):
            inhibition = self.executive.inhibit_response(
                selected.composite_score,
            )
            if inhibition.inhibited:
                withhold = next(
                    (c for c in candidates
                     if c.action_type == ActionType.WITHHOLD),
                    None,
                )
                if withhold is not None:
                    selected = withhold
                    inhibition_applied = True

        # Build notes, result, and track stats.
        return self._finalize_decision(
            selected, candidates, default_action,
            overridden, uncertainty_switched, inhibition_applied,
            confidence, uncertainty,
        )

    def _finalize_decision(
        self,
        selected: CandidateAction,
        candidates: list[CandidateAction],
        default_action: ActionType,
        overridden: bool,
        uncertainty_switched: bool,
        inhibition_applied: bool,
        confidence: float,
        uncertainty: float,
    ) -> DecisionOutcome:
        """Build notes, construct the DecisionOutcome, and track stats."""
        notes: list[str] = []
        if overridden:
            notes.append(
                f"overrode default {default_action.value} → "
                f"{selected.action_type.value}"
            )
        if uncertainty_switched:
            notes.append(
                f"switched due to low confidence ({confidence:.2f}) + "
                f"high uncertainty ({uncertainty:.2f})"
            )
        if inhibition_applied:
            notes.append("response inhibited by executive function")

        result = DecisionOutcome(
            selected=selected,
            candidates=candidates,
            overridden_default=overridden,
            inhibition_applied=inhibition_applied,
            uncertainty_switched=uncertainty_switched,
            notes=notes,
        )

        self._decisions.append(result)
        # Bound the history — decide() runs every turn in a
        # long-running daemon; only recent decisions are serialized.
        if len(self._decisions) > 500:
            del self._decisions[: len(self._decisions) - 500]
        if overridden:
            self._overrides += 1
        if inhibition_applied:
            self._inhibitions += 1
        if uncertainty_switched:
            self._uncertainty_switches += 1

        return result

    # ─── Candidate generation ─────────────────────────────────────

    def _generate_candidates(
        self,
        default_action: ActionType,
        perception_intent: str,
        reasoning_results: list[ReasoningResult],
    ) -> list[CandidateAction]:
        """Generate candidate actions for the current context.

        Always includes the default action (from the deliberation tree)
        plus context-appropriate alternatives.
        """
        candidates: list[CandidateAction] = []
        seen: set[ActionType] = set()

        def add(action: ActionType) -> None:
            """Add an action to candidates if not already present."""
            if action not in seen:
                candidates.append(CandidateAction(action_type=action))
                seen.add(action)

        # Always include the default.
        add(default_action)

        # Add context-appropriate alternatives.
        if perception_intent in ("question", "philosophy"):
            add(ActionType.ANSWER)
            add(ActionType.ASK)
            add(ActionType.REFLECT)
        elif perception_intent in ("greeting", "farewell"):
            add(ActionType.GREET)
            add(ActionType.ACKNOWLEDGE)
        elif perception_intent in ("emotion", "comfort"):
            add(ActionType.EMPATHIZE)
            add(ActionType.ACKNOWLEDGE)
        elif perception_intent in ("statement", "feedback"):
            add(ActionType.ACKNOWLEDGE)
            add(ActionType.REFLECT)

        # If there are reasoning results, answering is viable.
        if reasoning_results:
            add(ActionType.ANSWER)

        # If there are no reasoning results, asking is viable.
        if not reasoning_results:
            add(ActionType.ASK)

        # Always allow withholding (for inhibition).
        add(ActionType.WITHHOLD)

        return candidates

    # ─── Candidate evaluation ─────────────────────────────────────

    def _evaluate_candidate(
        self,
        candidate: CandidateAction,
        reasoning_results: list[ReasoningResult],
        confidence: float,
        uncertainty: float,
        goals: list[str],
        topics: list[str],
        perception_intent: str,
    ) -> None:
        """Evaluate a candidate against all criteria.

        Computes a score for each criterion (0..1), then combines them
        into a composite score using the value weights.
        """
        action = candidate.action_type

        # ── Evidence strength ──
        # How well-supported is this action by the reasoning results?
        if action == ActionType.ANSWER:
            if reasoning_results:
                avg_conf = sum(
                    r.confidence for r in reasoning_results
                ) / len(reasoning_results)
                evidence_score = avg_conf
            else:
                evidence_score = 0.0
        elif action == ActionType.ASK:
            # Asking is evidence-supported when we lack evidence.
            # Only strong when reasoning is genuinely weak.
            if reasoning_results:
                avg_conf = sum(
                    r.confidence for r in reasoning_results
                ) / len(reasoning_results)
                evidence_score = max(0.0, 0.5 - avg_conf)
            else:
                evidence_score = 1.0
        elif action == ActionType.REFLECT:
            # Reflection is supported by novel reasoning.
            evidence_score = sum(
                1 for r in reasoning_results if r.novel
            ) / max(1, len(reasoning_results)) if reasoning_results else 0.3
        elif action == ActionType.ACKNOWLEDGE:
            evidence_score = 0.5  # always viable
        elif action == ActionType.GREET:
            evidence_score = 1.0 if perception_intent in (
                "greeting", "farewell",
            ) else 0.1
        elif action == ActionType.EMPATHIZE:
            evidence_score = 1.0 if perception_intent in (
                "emotion", "comfort",
            ) else 0.1
        elif action == ActionType.INVESTIGATE:
            evidence_score = 0.3  # viable when goals are active
        else:  # WITHHOLD
            evidence_score = 0.0
        candidate.criteria_scores["evidence"] = evidence_score

        # ── Goal relevance ──
        # Does this action advance an active goal?
        if goals and action == ActionType.INVESTIGATE:
            candidate.criteria_scores["goal_relevance"] = 0.8
        elif goals and action == ActionType.ANSWER:
            # Answering about a goal topic advances understanding.
            goal_topics = {g.split(":")[-1].strip() for g in goals}
            topic_overlap = len(goal_topics & set(topics))
            candidate.criteria_scores["goal_relevance"] = min(
                1.0, topic_overlap / max(1, len(goals)),
            )
        elif goals and action == ActionType.ASK:
            # Asking can advance goals by gathering information.
            candidate.criteria_scores["goal_relevance"] = 0.4
        else:
            candidate.criteria_scores["goal_relevance"] = 0.1

        # ── Uncertainty ──
        # How well does this action handle the current uncertainty?
        if action == ActionType.ANSWER:
            # Answering under high uncertainty is risky.
            candidate.criteria_scores["uncertainty"] = 1.0 - uncertainty
        elif action == ActionType.ASK:
            # Asking is the right response to uncertainty.
            candidate.criteria_scores["uncertainty"] = uncertainty
        elif action == ActionType.REFLECT:
            # Reflection handles moderate uncertainty well, but
            # shouldn't score high when uncertainty is low.
            candidate.criteria_scores["uncertainty"] = uncertainty * 0.6
        elif action == ActionType.ACKNOWLEDGE:
            candidate.criteria_scores["uncertainty"] = 0.5 * uncertainty
        else:
            candidate.criteria_scores["uncertainty"] = 0.3 * uncertainty

        # ── Value criteria ──
        self._evaluate_value_criteria(
            candidate, action, confidence, uncertainty, perception_intent,
        )

        # ── Confidence ──
        candidate.confidence = confidence if action == ActionType.ANSWER else (
            evidence_score * 0.5 + candidate.criteria_scores.get(
                "coherence", 0.5,
            ) * 0.5
        )

        # ── Composite score ──
        candidate.composite_score = self._compute_composite(candidate)

        # ── Rationale ──
        candidate.rationale = self._build_rationale(candidate)

    def _evaluate_value_criteria(
        self,
        candidate: CandidateAction,
        action: ActionType,
        confidence: float,
        uncertainty: float,
        perception_intent: str,
    ) -> None:
        """Evaluate the value criteria (accuracy, honesty, helpfulness,
        curiosity, safety, coherence) for a candidate action.
        """
        # Accuracy: does this action promote factual accuracy?
        if action == ActionType.ANSWER:
            candidate.criteria_scores["accuracy"] = confidence
        elif action == ActionType.ASK:
            candidate.criteria_scores["accuracy"] = 0.7  # seeking truth
        elif action == ActionType.REFLECT:
            candidate.criteria_scores["accuracy"] = 0.6
        else:
            candidate.criteria_scores["accuracy"] = 0.5

        # Honesty: does this action express genuine uncertainty?
        if action == ActionType.ASK and uncertainty > 0.5:
            candidate.criteria_scores["honesty"] = 0.9  # admitting ignorance
        elif action == ActionType.ANSWER and uncertainty > 0.5:
            candidate.criteria_scores["honesty"] = 0.3  # overconfident
        elif action == ActionType.REFLECT:
            candidate.criteria_scores["honesty"] = 0.8  # transparent
        else:
            candidate.criteria_scores["honesty"] = 0.6

        # Helpfulness: does this advance the user's goal?
        if action == ActionType.ANSWER and confidence > 0.5:
            candidate.criteria_scores["helpfulness"] = 0.8
        elif action == ActionType.ASK:
            candidate.criteria_scores["helpfulness"] = 0.6
        elif action == ActionType.ACKNOWLEDGE:
            candidate.criteria_scores["helpfulness"] = 0.4
        else:
            candidate.criteria_scores["helpfulness"] = 0.5

        # Curiosity: does this advance Genesis's own learning?
        if action == ActionType.INVESTIGATE:
            candidate.criteria_scores["curiosity"] = 0.9
        elif action == ActionType.ASK:
            candidate.criteria_scores["curiosity"] = 0.7
        elif action == ActionType.REFLECT:
            candidate.criteria_scores["curiosity"] = 0.6
        else:
            candidate.criteria_scores["curiosity"] = 0.3

        # Safety: does this avoid harm?
        if action == ActionType.WITHHOLD:
            candidate.criteria_scores["safety"] = 1.0  # safest
        elif action in (ActionType.ASK, ActionType.ACKNOWLEDGE):
            candidate.criteria_scores["safety"] = 0.9
        elif action == ActionType.ANSWER:
            candidate.criteria_scores["safety"] = confidence * 0.8
        else:
            candidate.criteria_scores["safety"] = 0.7

        # Coherence: is this coherent with the context?
        if action.value == perception_intent or (
            action == ActionType.ANSWER and perception_intent == "question"
        ) or (
            action == ActionType.GREET and perception_intent == "greeting"
        ):
            candidate.criteria_scores["coherence"] = 0.9
        else:
            candidate.criteria_scores["coherence"] = 0.4

    def _compute_composite(self, candidate: CandidateAction) -> float:
        """Compute the normalized composite score from criteria and value weights.

        Combines evidence, goal_relevance, and uncertainty (each
        weighted) with the value criteria (weighted by
        ``value_weights``). The result is normalized to [0, 1] by
        dividing by the total weight sum, so thresholds and
        comparisons against the score are on a meaningful scale.
        """
        score = 0.0
        # Core criteria (evidence, goals, uncertainty) get fixed weight.
        score += candidate.criteria_scores.get("evidence", 0.0) * 0.20
        score += candidate.criteria_scores.get("goal_relevance", 0.0) * 0.15
        score += candidate.criteria_scores.get("uncertainty", 0.0) * 0.15
        # Value criteria get their configured weights.
        for criterion, weight in self.value_weights.items():
            score += candidate.criteria_scores.get(criterion, 0.0) * weight
        # Normalize to [0, 1] by dividing by the total weight sum.
        # Core weights (0.50) + value weights (default 1.00) = 1.50.
        total_weight = 0.50 + sum(self.value_weights.values())
        return score / total_weight if total_weight > 0 else 0.0

    def _build_rationale(self, candidate: CandidateAction) -> str:
        """Build a human-readable rationale for the candidate's score."""
        parts = [f"{candidate.action_type.value}:"]
        for criterion in sorted(candidate.criteria_scores):
            score = candidate.criteria_scores[criterion]
            parts.append(f"  {criterion}={score:.2f}")
        parts.append(f"  composite={candidate.composite_score:.3f}")
        return "\n".join(parts)

    # ─── TD value lookup ──────────────────────────────────────────

    def _lookup_td_value(
        self, candidate: CandidateAction, topics: list[str],
    ) -> float:
        """Look up the TD value of the state resulting from this action.

        Each action type produces a different "state" — the set of
        active concepts after the action. The TD learner predicts the
        value of that state.
        """
        if self.td_learner is None:
            return 0.0
        # The state is the topics plus the action type.
        state = [*topics, candidate.action_type.value]
        try:
            return self.td_learner.predict_value(state)
        except Exception:  # noqa: BLE001
            return 0.0

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "override_count": self._overrides,
            "inhibition_count": self._inhibitions,
            "uncertainty_switch_count": self._uncertainty_switches,
            "value_weights": dict(self.value_weights),
            "recent_decisions": [
                {
                    "selected": d.action_type.value,
                    "confidence": d.confidence,
                    "overridden": d.overridden_default,
                    "inhibited": d.inhibition_applied,
                    "switched": d.uncertainty_switched,
                }
                for d in self._decisions[-20:]
            ],
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        overrides = data.get("override_count", 0)
        inhibitions = data.get("inhibition_count", 0)
        switches = data.get("uncertainty_switch_count", 0)
        self._overrides = (
            int(overrides) if isinstance(overrides, (int, float)) else 0
        )
        self._inhibitions = (
            int(inhibitions) if isinstance(inhibitions, (int, float)) else 0
        )
        self._uncertainty_switches = (
            int(switches) if isinstance(switches, (int, float)) else 0
        )
        weights = data.get("value_weights", {})
        if isinstance(weights, dict):
            self.value_weights = {
                k: float(v) if isinstance(v, (int, float)) else 0.0
                for k, v in weights.items()
            }

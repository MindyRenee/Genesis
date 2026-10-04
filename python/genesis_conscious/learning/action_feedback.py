"""Action-feedback learning — action → prediction → observation → revision.

This module implements the reusable loop the rest of the cognition stack
adapts to its own action domains:

1. ``begin`` records an action and the state Genesis predicts it will
   produce. The prediction may come from a domain policy, an explicit
   expectation, and/or the learned action model.
2. ``observe`` records the independently observed post-action state —
   never by copying the intended outcome — computes a structured
   discrepancy, updates the learned action model, and diagnoses likely
   causes.
3. ``recommend`` turns the diagnosis into a bounded revision: retry,
   choose a different candidate action, diagnose further, wait, or stop.
4. ``run_action`` is a convenience wrapper for domains that can execute
   a candidate action immediately.

The learner intentionally separates three kinds of state:

- **Attempt history**: what was tried, expected, observed, and inferred.
- **Action models**: durable learned statistics for a domain/action
  signature (success probability, modal outcomes, numeric effects,
  repeated failure causes).
- **Concept-network evidence**: observed action→outcome and
  cause→outcome edges. These give the existing causal/abductive
  machinery concrete evidence to reason over instead of keeping action
  learning trapped in a private table.

The engine does not invent explanations when evidence is absent. A
mismatch can produce an explicit ``unknown_discrepancy`` hypothesis;
known causes are ranked by observed frequency, network evidence, and any
domain-specific abducer supplied by the caller.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeGuard

from ..concepts import ConceptNetwork, RelationType

if TYPE_CHECKING:
    from ..reasoning.engine import ProbabilisticReasoning

__all__ = [
    "ActionAttempt",
    "ActionFeedbackEngine",
    "ActionPrediction",
    "ActionRevision",
    "ActionSpec",
    "CausalHypothesis",
    "Discrepancy",
    "OutcomeState",
    "RevisionKind",
]

logger = logging.getLogger(__name__)

# How many completed attempts and pending actions to retain. The durable
# model is the important state; the history is bounded introspection.
_MAX_HISTORY = 200
_MAX_PENDING = 500

# Fields whose wrongness is more informative than ordinary metadata.
_FIELD_WEIGHTS: dict[str, float] = {
    "ok": 2.0,
    "status": 1.5,
    "verified": 1.5,
    "exists": 1.5,
    "readable": 1.2,
    "writable": 1.2,
    "tests_passed": 1.2,
    "syntax_ok": 1.2,
    "level": 1.2,
    "delta": 1.2,
}

_TRANSIENT_CAUSES = {
    "timeout",
    "timed_out",
    "temporary",
    "busy",
    "interrupted",
    "transient",
}

_TERMINAL_CAUSES = {
    "unsafe_action",
    "outside_root",
    "permission_denied",
    "invalid_action",
    "unsupported_action",
    "unknown_action",
    "cancelled",
}

_WAIT_CAUSES = {
    "daily_limit",
    "quota_exceeded",
    "rate_limited",
    "resource_exhausted",
    "offline",
    "awaiting_approval",
}

# Reward mapping bounds. The reward reflects the *utility* of the
# observed outcome; the reward prediction error (computed by the TD
# learner) already encodes how surprising it was, so the mapping only
# modulates by discrepancy magnitude and failure-cause class.
_REWARD_MATCHED_SUCCESS = 1.0
_REWARD_SUCCESS_DISCREPANCY_WEIGHT = 0.4
_REWARD_FAILURE = -0.5
_REWARD_FAILURE_DISCREPANCY_WEIGHT = 0.3
_REWARD_TERMINAL = -1.0
_REWARD_WAIT = -0.4
_REWARD_UNSCORED = 0.5


class RevisionKind(StrEnum):
    """The kind of revised action recommended after feedback."""

    NONE = "none"
    RETRY = "retry"
    CHOOSE = "choose"
    DIAGNOSE = "diagnose"
    WAIT = "wait"
    AVOID = "avoid"


@dataclass(slots=True)
class ActionSpec:
    """A concrete action Genesis can perform.

    ``domain`` separates tool use, plan steps, body regulation, and other
    environments. ``signature`` is the learned context bucket: adapters
    can set it to a target type (for example ``python_file`` or a
    chemical name) without making every individual target a separate
    model.
    """

    domain: str
    name: str
    target: str = ""
    signature: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        """Stable key for the learned action model."""
        signature = self.signature or self.parameters.get("signature") or "*"
        return f"{self.domain}:{self.name}:{signature}"

    @property
    def concept_id(self) -> str:
        """Concept-network identifier for this kind of action."""
        return f"action:{self.domain}:{self.name}"

    def with_parameters(self, **updates: Any) -> ActionSpec:
        """Return a revised action with selected parameters replaced."""
        parameters = dict(self.parameters)
        parameters.update(updates)
        return ActionSpec(
            domain=self.domain,
            name=self.name,
            target=self.target,
            signature=self.signature,
            parameters=parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action."""
        return {
            "domain": self.domain,
            "name": self.name,
            "target": self.target,
            "signature": self.signature,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        """Restore an action from serialized data."""
        parameters = data.get("parameters", {})
        return cls(
            domain=str(data.get("domain", "")),
            name=str(data.get("name", "")),
            target=str(data.get("target", "")),
            signature=str(data.get("signature", "")),
            parameters=dict(parameters) if isinstance(parameters, dict) else {},
        )


@dataclass(slots=True)
class OutcomeState:
    """A structured state estimate or observation.

    ``observed=False`` marks a prediction. ``observed=True`` marks a
    real post-action observation supplied by an execution adapter.
    """

    fields: dict[str, Any] = field(default_factory=dict)
    observed: bool = False
    summary: str = ""

    def get(self, field_name: str, default: Any = None) -> Any:
        """Return a state field."""
        return self.fields.get(field_name, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state."""
        return {
            "fields": self.fields,
            "observed": self.observed,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeState:
        """Restore a state from serialized data."""
        fields = data.get("fields", {})
        return cls(
            fields=dict(fields) if isinstance(fields, dict) else {},
            observed=bool(data.get("observed", False)),
            summary=str(data.get("summary", "")),
        )


@dataclass(slots=True)
class ActionPrediction:
    """The pre-action prediction made by a domain policy plus the model."""

    expected: OutcomeState
    confidence: float
    basis: str
    expected_success: float | None = None
    model_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction."""
        return {
            "expected": self.expected.to_dict(),
            "confidence": self.confidence,
            "basis": self.basis,
            "expected_success": self.expected_success,
            "model_key": self.model_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionPrediction:
        """Restore a prediction."""
        expected_data = data.get("expected", {})
        return cls(
            expected=(
                OutcomeState.from_dict(expected_data)
                if isinstance(expected_data, dict)
                else OutcomeState()
            ),
            confidence=float(data.get("confidence", 0.5)),
            basis=str(data.get("basis", "declared")),
            expected_success=(
                float(data["expected_success"])
                if isinstance(data.get("expected_success"), (int, float))
                else None
            ),
            model_key=str(data.get("model_key", "")),
        )


@dataclass(slots=True)
class Discrepancy:
    """The computed difference between predicted and observed state."""

    mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    magnitude: float = 0.0
    compared_fields: int = 0

    @property
    def matched(self) -> bool:
        """Whether every compared field matched."""
        return not self.mismatches and not self.missing

    def to_dict(self) -> dict[str, Any]:
        """Serialize the discrepancy."""
        return {
            "mismatches": self.mismatches,
            "missing": self.missing,
            "magnitude": self.magnitude,
            "compared_fields": self.compared_fields,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Discrepancy:
        """Restore a discrepancy."""
        mismatches = data.get("mismatches", {})
        missing = data.get("missing", [])
        return cls(
            mismatches=(
                dict(mismatches) if isinstance(mismatches, dict) else {}
            ),
            missing=[str(m) for m in missing] if isinstance(missing, list) else [],
            magnitude=float(data.get("magnitude", 0.0)),
            compared_fields=int(data.get("compared_fields", 0)),
        )


@dataclass(slots=True)
class CausalHypothesis:
    """A ranked possible cause for an observed discrepancy."""

    cause: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    source: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hypothesis."""
        return {
            "cause": self.cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalHypothesis:
        """Restore a hypothesis."""
        evidence = data.get("evidence", [])
        return cls(
            cause=str(data.get("cause", "unknown_discrepancy")),
            confidence=float(data.get("confidence", 0.0)),
            evidence=[str(e) for e in evidence] if isinstance(evidence, list) else [],
            source=str(data.get("source", "observed")),
        )


@dataclass(slots=True)
class ActionRevision:
    """A bounded recommendation for what to do after feedback."""

    kind: RevisionKind = RevisionKind.NONE
    action: ActionSpec | None = None
    reason: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the revision."""
        return {
            "kind": self.kind.value,
            "action": self.action.to_dict() if self.action is not None else None,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRevision:
        """Restore a revision."""
        action_data = data.get("action")
        try:
            kind = RevisionKind(str(data.get("kind", "none")))
        except ValueError:
            kind = RevisionKind.NONE
        return cls(
            kind=kind,
            action=(
                ActionSpec.from_dict(action_data)
                if isinstance(action_data, dict)
                else None
            ),
            reason=str(data.get("reason", "")),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(slots=True)
class ActionAttempt:
    """One complete or pending action-feedback episode."""

    attempt_id: str
    action: ActionSpec
    prediction: ActionPrediction
    created_at: float
    actual: OutcomeState | None = None
    discrepancy: Discrepancy | None = None
    hypotheses: list[CausalHypothesis] = field(default_factory=list)
    revision: ActionRevision | None = None
    completed_at: float | None = None
    retry_of: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        """Whether a post-action state was supplied."""
        return self.actual is not None and self.actual.observed

    @property
    def succeeded(self) -> bool:
        """Whether the observed outcome marked the action successful."""
        return bool(self.actual and self.actual.get("ok") is True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attempt."""
        return {
            "attempt_id": self.attempt_id,
            "action": self.action.to_dict(),
            "prediction": self.prediction.to_dict(),
            "created_at": self.created_at,
            "actual": self.actual.to_dict() if self.actual else None,
            "discrepancy": (
                self.discrepancy.to_dict() if self.discrepancy else None
            ),
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "revision": self.revision.to_dict() if self.revision else None,
            "completed_at": self.completed_at,
            "retry_of": self.retry_of,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionAttempt:
        """Restore an attempt."""
        actual_data = data.get("actual")
        discrepancy_data = data.get("discrepancy")
        revision_data = data.get("revision")
        hypotheses = data.get("hypotheses", [])
        metadata = data.get("metadata", {})
        return cls(
            attempt_id=str(data.get("attempt_id", "")),
            action=ActionSpec.from_dict(
                data.get("action", {}) if isinstance(data.get("action"), dict) else {}
            ),
            prediction=ActionPrediction.from_dict(
                data.get("prediction", {})
                if isinstance(data.get("prediction"), dict)
                else {}
            ),
            created_at=float(data.get("created_at", 0.0)),
            actual=(
                OutcomeState.from_dict(actual_data)
                if isinstance(actual_data, dict)
                else None
            ),
            discrepancy=(
                Discrepancy.from_dict(discrepancy_data)
                if isinstance(discrepancy_data, dict)
                else None
            ),
            hypotheses=[
                CausalHypothesis.from_dict(h)
                for h in hypotheses
                if isinstance(h, dict)
            ] if isinstance(hypotheses, list) else [],
            revision=(
                ActionRevision.from_dict(revision_data)
                if isinstance(revision_data, dict)
                else None
            ),
            completed_at=(
                float(data["completed_at"])
                if isinstance(data.get("completed_at"), (int, float))
                else None
            ),
            retry_of=str(data.get("retry_of", "")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass(slots=True)
class _NumericStat:
    """Online mean/variance for numeric outcome fields."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, value: float) -> None:
        """Add a numeric observation."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance estimate."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the statistic."""
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _NumericStat:
        """Restore the statistic."""
        return cls(
            count=int(data.get("count", 0)),
            mean=float(data.get("mean", 0.0)),
            m2=float(data.get("m2", 0.0)),
        )


@dataclass(slots=True)
class _ActionModel:
    """Learned statistics for one domain/action/context signature."""

    key: str
    domain: str
    name: str
    signature: str = ""
    attempts: int = 0
    successes: int = 0
    alpha: float = 1.0
    beta: float = 1.0
    outcome_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    numeric_stats: dict[str, _NumericStat] = field(default_factory=dict)
    failure_causes: dict[str, int] = field(default_factory=dict)
    last_error_signature: str = ""
    repeated_error_count: int = 0
    discrepancy_ema: float = 0.0
    updated_at: float = 0.0

    @property
    def expected_success(self) -> float:
        """Bayesian expected success probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def success_rate(self) -> float:
        """Observed success rate."""
        return self.successes / self.attempts if self.attempts else 0.0

    def modal_value(self, field_name: str) -> tuple[Any, float] | None:
        """Return the modal observed value and its empirical probability."""
        counts = self.outcome_counts.get(field_name, {})
        total = sum(counts.values())
        if not total:
            return None
        value_key, count = max(counts.items(), key=lambda item: item[1])
        return _decode_value(value_key), count / total

    def top_failure_cause(self) -> tuple[str, int] | None:
        """Return the most frequent learned failure cause."""
        if not self.failure_causes:
            return None
        return max(self.failure_causes.items(), key=lambda item: item[1])

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model."""
        return {
            "key": self.key,
            "domain": self.domain,
            "name": self.name,
            "signature": self.signature,
            "attempts": self.attempts,
            "successes": self.successes,
            "alpha": self.alpha,
            "beta": self.beta,
            "outcome_counts": self.outcome_counts,
            "numeric_stats": {
                field_name: stat.to_dict()
                for field_name, stat in self.numeric_stats.items()
            },
            "failure_causes": self.failure_causes,
            "last_error_signature": self.last_error_signature,
            "repeated_error_count": self.repeated_error_count,
            "discrepancy_ema": self.discrepancy_ema,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ActionModel:
        """Restore a model."""
        outcome_counts_raw = data.get("outcome_counts", {})
        outcome_counts: dict[str, dict[str, int]] = {}
        if isinstance(outcome_counts_raw, dict):
            for field_name, counts in outcome_counts_raw.items():
                if isinstance(counts, dict):
                    outcome_counts[str(field_name)] = {
                        str(value_key): int(count)
                        for value_key, count in counts.items()
                        if isinstance(count, (int, float))
                    }
        numeric_raw = data.get("numeric_stats", {})
        numeric_stats: dict[str, _NumericStat] = {}
        if isinstance(numeric_raw, dict):
            for field_name, stat_data in numeric_raw.items():
                if isinstance(stat_data, dict):
                    numeric_stats[str(field_name)] = _NumericStat.from_dict(stat_data)
        failure_raw = data.get("failure_causes", {})
        failure_causes = {
            str(cause): int(count)
            for cause, count in failure_raw.items()
            if isinstance(count, (int, float))
        } if isinstance(failure_raw, dict) else {}
        return cls(
            key=str(data.get("key", "")),
            domain=str(data.get("domain", "")),
            name=str(data.get("name", "")),
            signature=str(data.get("signature", "")),
            attempts=int(data.get("attempts", 0)),
            successes=int(data.get("successes", 0)),
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 1.0)),
            outcome_counts=outcome_counts,
            numeric_stats=numeric_stats,
            failure_causes=failure_causes,
            last_error_signature=str(data.get("last_error_signature", "")),
            repeated_error_count=int(data.get("repeated_error_count", 0)),
            discrepancy_ema=float(data.get("discrepancy_ema", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
        )


class ActionFeedbackEngine:
    """Learn action consequences from prediction errors.

    The engine is deliberately domain-neutral. Execution adapters supply
    the independently observed state; the learner owns comparison,
    diagnosis, model update, revision selection, persistence, and
    concept-network grounding.
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        *,
        probabilistic: ProbabilisticReasoning | None = None,
        causal_infer: Callable[
            [ActionSpec, Discrepancy, OutcomeState],
            Iterable[CausalHypothesis],
        ] | None = None,
        repair_suggester: Callable[
            [ActionSpec, list[CausalHypothesis], OutcomeState],
            ActionSpec | None,
        ] | None = None,
        attempt_listener: Callable[[ActionAttempt, str], None] | None = None,
        reward_learner: Callable[[ActionAttempt, float], float | None] | None = None,
        value_estimator: Callable[[ActionSpec], float | None] | None = None,
    ) -> None:
        """Initialize the reusable feedback learner.

        Args:
            network: Optional concept network used to ground observed
                action→outcome and cause→outcome evidence.
            probabilistic: Optional Bayesian belief engine. Observed
                outcomes become positive/negative evidence for the
                corresponding action→outcome relation.
            causal_infer: Optional domain-specific abducer returning
                extra hypotheses.
            repair_suggester: Optional domain-specific adapter that maps
                a diagnosis to a diagnostic/repair action.
            attempt_listener: Optional observer called as attempts move
                through the loop. Phases are ``"begin"`` after a pending
                action is registered and ``"observed"`` after its real
                outcome has been evaluated and learned from.
            reward_learner: Optional scalar-reward sink (typically a TD
                update). Called with the completed attempt and the
                utility reward computed from its independently observed
                outcome; may return the resulting reward prediction
                error, which is recorded on the attempt.
            value_estimator: Optional learned-value lookup for an action
                candidate (typically TD value over the action's concept
                state). Blended into ``_action_score`` so repeated
                outcomes bias alternative selection.
        """
        self.network = network
        self.probabilistic = probabilistic
        self.causal_infer = causal_infer
        self.repair_suggester = repair_suggester
        self.attempt_listener = attempt_listener
        self.reward_learner = reward_learner
        self.value_estimator = value_estimator
        self._models: dict[str, _ActionModel] = {}
        self._pending: dict[str, ActionAttempt] = {}
        self._history: list[ActionAttempt] = []
        self._counter = 0
        # Shared-state lock. Action selection is volitional — Genesis
        # chooses when to act — but the engine is shared across the
        # conversation thread, the autonomic regulator thread, and the
        # volition-gated learner thread, so begin/observe windows can
        # still overlap at a pause/grant boundary. The lock serializes
        # only state mutations and snapshots; external callbacks
        # (listener, reward learner, value estimator, suggesters) run
        # outside it so they cannot wedge the engine.
        self._lock = threading.RLock()

    @property
    def history(self) -> list[ActionAttempt]:
        """Recent completed attempts (copy)."""
        with self._lock:
            return list(self._history)

    @property
    def pending_count(self) -> int:
        """How many begun actions are still awaiting observation."""
        with self._lock:
            return len(self._pending)

    @property
    def model_count(self) -> int:
        """How many action/context models have been learned."""
        with self._lock:
            return len(self._models)

    # ─── Prediction ───────────────────────────────────────────────

    def begin(
        self,
        action: ActionSpec,
        *,
        expected: OutcomeState | dict[str, Any] | None = None,
        confidence: float | None = None,
        retry_of: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ActionAttempt:
        """Record an action and its predicted post-action state.

        The returned attempt must later be closed with ``observe`` using
        independently gathered outcome data. Until then it remains in the
        pending set, so unobserved actions are explicit rather than being
        silently treated as successful.
        """
        with self._lock:
            attempt = self._begin(
                action,
                expected=expected,
                confidence=confidence,
                retry_of=retry_of,
                metadata=metadata,
            )
        self._notify_attempt(attempt, "begin")
        return attempt

    def _begin(
        self,
        action: ActionSpec,
        *,
        expected: OutcomeState | dict[str, Any] | None = None,
        confidence: float | None = None,
        retry_of: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ActionAttempt:
        """Build the pending attempt; caller holds ``self._lock``."""
        model = self._model_for(action)
        expected_state = self._coerce_expected(expected)
        expected_fields = dict(expected_state.fields)

        learned_used = False
        if model.attempts:
            learned_used = self._apply_learned_expectation(
                expected_fields, model,
            )

        expected_success = self._expected_success(expected_fields, model)
        if "ok" not in expected_fields:
            expected_fields["ok"] = expected_success >= 0.5
            learned_used = learned_used or model.attempts > 0
        if "status" not in expected_fields:
            expected_fields["status"] = (
                "completed" if expected_success >= 0.5 else "failed"
            )
            learned_used = learned_used or model.attempts > 0

        expected_state = OutcomeState(
            fields=expected_fields,
            observed=False,
            summary=expected_state.summary,
        )
        basis = "learned+declared" if learned_used else "declared"
        if model.attempts and not expected:
            basis = "learned"

        if confidence is None:
            # Confidence grows with evidence but is bounded by model
            # uncertainty. A fresh action is deliberately only mildly
            # confident.
            evidence = min(model.attempts, 20)
            confidence = min(0.9, 0.55 + evidence * 0.015)
            if model.attempts:
                confidence = max(
                    confidence,
                    min(0.95, abs(expected_success - 0.5) * 2.0 * 0.75 + 0.45),
                )
        confidence = _clamp01(confidence)

        self._counter += 1
        attempt = ActionAttempt(
            attempt_id=(
                f"{action.domain}-{self._counter}-"
                f"{int(time.time() * 1000)}"
            ),
            action=action,
            prediction=ActionPrediction(
                expected=expected_state,
                confidence=confidence,
                basis=basis,
                expected_success=expected_success,
                model_key=model.key,
            ),
            created_at=time.time(),
            retry_of=retry_of,
            metadata=dict(metadata or {}),
        )
        if len(self._pending) >= _MAX_PENDING:
            oldest = next(iter(self._pending))
            self._pending.pop(oldest, None)
        self._pending[attempt.attempt_id] = attempt
        return attempt

    def _notify_attempt(self, attempt: ActionAttempt, phase: str) -> None:
        """Notify the optional lifecycle listener without breaking action execution."""
        if self.attempt_listener is None:
            return
        try:
            self.attempt_listener(attempt, phase)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"action attempt listener failed: {e}")

    def _coerce_expected(
        self, expected: OutcomeState | dict[str, Any] | None,
    ) -> OutcomeState:
        """Normalize caller-supplied expectations."""
        if expected is None:
            return OutcomeState(observed=False)
        if isinstance(expected, OutcomeState):
            return OutcomeState(
                fields=dict(expected.fields),
                observed=False,
                summary=expected.summary,
            )
        if isinstance(expected, dict):
            return OutcomeState(fields=dict(expected), observed=False)
        return OutcomeState(observed=False)

    def _expected_success(
        self,
        expected_fields: dict[str, Any],
        model: _ActionModel,
    ) -> float:
        """Estimate expected success from declaration and learned model."""
        declared = expected_fields.get("ok")
        declared_p = (
            float(declared)
            if isinstance(declared, (int, float)) and not isinstance(declared, bool)
            else (1.0 if declared is True else 0.0 if declared is False else 0.6)
        )
        if model.attempts:
            # Blend the domain-declared prior with the learned posterior;
            # more attempts move the estimate toward experience.
            weight = min(0.85, model.attempts / (model.attempts + 3.0))
            return _clamp01(
                declared_p * (1.0 - weight) + model.expected_success * weight
            )
        return _clamp01(declared_p)

    def _apply_learned_expectation(
        self,
        expected_fields: dict[str, Any],
        model: _ActionModel,
    ) -> bool:
        """Apply durable learned expectations to declared fields."""
        learned = False
        for field_name, stat in model.numeric_stats.items():
            if stat.count >= 2 and (
                field_name not in expected_fields
                or _is_number(expected_fields.get(field_name))
            ):
                expected_fields[field_name] = stat.mean
                learned = True
        for field_name in model.outcome_counts:
            if field_name in model.numeric_stats:
                continue
            modal = model.modal_value(field_name)
            if modal is None:
                continue
            value, probability = modal
            if probability >= 0.65 and model.attempts >= 3:
                expected_fields[field_name] = value
                learned = True
        return learned

    # ─── Observation and discrepancy ──────────────────────────────

    def observe(
        self,
        attempt_or_id: ActionAttempt | str,
        actual: OutcomeState | dict[str, Any],
        *,
        alternatives: Iterable[ActionSpec] | None = None,
        repair: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ActionAttempt:
        """Close an attempt with the independently observed state.

        ``actual`` is mandatory and is never copied from the prediction.
        A dict is treated as an observed ``OutcomeState``; callers that
        could not observe the world should pass ``observed=False`` in an
        ``OutcomeState`` rather than fabricating a result.
        """
        with self._lock:
            attempt = self._observe(
                attempt_or_id,
                actual,
                alternatives=alternatives,
                repair=repair,
                metadata=metadata,
            )
        self._apply_reward_learning(attempt)
        self._notify_attempt(attempt, "observed")
        return attempt

    def _observe(
        self,
        attempt_or_id: ActionAttempt | str,
        actual: OutcomeState | dict[str, Any],
        *,
        alternatives: Iterable[ActionSpec] | None = None,
        repair: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> ActionAttempt:
        """Evaluate the observed outcome; caller holds ``self._lock``."""
        attempt = self._resolve_attempt(attempt_or_id)
        actual_state = self._coerce_actual(actual)
        if metadata:
            attempt.metadata.update(metadata)

        attempt.actual = actual_state
        attempt.completed_at = time.time()
        attempt.discrepancy = self.compare(
            attempt.prediction.expected,
            actual_state,
        )
        model = self._model_for(attempt.action)
        self._update_model(model, attempt)
        attempt.hypotheses = self._diagnose(attempt, model)
        attempt.revision = self.recommend(
            attempt,
            alternatives=alternatives,
            repair=repair,
        )

        self._pending.pop(attempt.attempt_id, None)
        self._history.append(attempt)
        if len(self._history) > _MAX_HISTORY:
            del self._history[: len(self._history) - _MAX_HISTORY]
        return attempt

    # ─── Reward / value learning ──────────────────────────────────

    def _apply_reward_learning(self, attempt: ActionAttempt) -> None:
        """Feed the observed outcome's utility into value learning.

        The reward reflects how the world actually went — success,
        failure, or an unscored observation — modulated by prediction
        discrepancy. The ``reward_learner`` callback (a TD update in
        cognition) turns that utility into a reward prediction error,
        so expected failures stop teaching and surprising ones teach
        most. The returned RPE is kept on the attempt for
        introspection. Reward learning never breaks the action path.
        """
        reward = self._attempt_reward(attempt)
        if reward is None:
            return
        attempt.metadata["reward"] = reward
        if self.reward_learner is None:
            return
        try:
            rpe = self.reward_learner(attempt, reward)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"action reward learner failed: {e}")
            return
        if rpe is not None:
            attempt.metadata["rpe"] = rpe

    def _attempt_reward(self, attempt: ActionAttempt) -> float | None:
        """Compute the scalar utility of an independently observed outcome.

        ``ok``-scored outcomes map to bounded reward: terminal causes
        are the worst, waitable causes are mildly negative (right action,
        wrong time), and ordinary failures scale with how wrong the
        prediction was. Unscored outcomes (no ``ok`` field) earn a
        modest reward when the prediction matched and less when it did
        not — predicting the world correctly is itself valuable.
        """
        actual = attempt.actual
        if actual is None or not actual.observed:
            return None
        discrepancy = attempt.discrepancy
        magnitude = discrepancy.magnitude if discrepancy is not None else 0.0
        ok = actual.get("ok")
        if ok is True:
            reward = (
                _REWARD_MATCHED_SUCCESS
                - _REWARD_SUCCESS_DISCREPANCY_WEIGHT * magnitude
            )
        elif ok is False:
            cause = self._actual_cause(actual)
            if cause in _TERMINAL_CAUSES:
                reward = _REWARD_TERMINAL
            elif cause in _WAIT_CAUSES:
                reward = _REWARD_WAIT
            else:
                reward = (
                    _REWARD_FAILURE
                    - _REWARD_FAILURE_DISCREPANCY_WEIGHT * magnitude
                )
        else:
            reward = _REWARD_UNSCORED - magnitude
        return max(-1.0, min(1.0, reward))

    def _resolve_attempt(self, attempt_or_id: ActionAttempt | str) -> ActionAttempt:
        """Resolve an attempt object or ID from the pending set."""
        attempt: ActionAttempt | None
        if isinstance(attempt_or_id, ActionAttempt):
            attempt_id = attempt_or_id.attempt_id
            attempt = self._pending.get(attempt_id, attempt_or_id)
        else:
            attempt = self._pending.get(attempt_or_id)
            attempt_id = attempt_or_id
        if attempt is None:
            raise KeyError(f"unknown or already-observed action attempt: {attempt_id}")
        return attempt

    @staticmethod
    def _coerce_actual(actual: OutcomeState | dict[str, Any]) -> OutcomeState:
        """Normalize observed state without inferring it from intent."""
        if isinstance(actual, OutcomeState):
            return OutcomeState(
                fields=dict(actual.fields),
                observed=actual.observed,
                summary=actual.summary,
            )
        return OutcomeState(fields=dict(actual), observed=True)

    @staticmethod
    def compare(
        expected: OutcomeState,
        actual: OutcomeState,
    ) -> Discrepancy:
        """Compare predicted and observed state field-by-field."""
        mismatch_weight = 0.0
        total_weight = 0.0
        mismatches: dict[str, dict[str, Any]] = {}
        missing: list[str] = []

        for field_name, expected_value in expected.fields.items():
            if expected_value is None:
                continue
            weight = _FIELD_WEIGHTS.get(field_name, 1.0)
            total_weight += weight
            if field_name not in actual.fields:
                missing.append(field_name)
                mismatch_weight += weight * 0.5
                continue
            actual_value = actual.fields[field_name]
            if not _values_match(expected_value, actual_value):
                mismatches[field_name] = {
                    "expected": expected_value,
                    "actual": actual_value,
                    "delta": (
                        actual_value - expected_value
                        if _is_number(actual_value) and _is_number(expected_value)
                        else None
                    ),
                }
                mismatch_weight += weight

        magnitude = (
            min(1.0, mismatch_weight / total_weight)
            if total_weight > 0
            else 0.0
        )
        return Discrepancy(
            mismatches=mismatches,
            missing=missing,
            magnitude=magnitude,
            compared_fields=sum(
                1 for value in expected.fields.values() if value is not None
            ),
        )

    # ─── Model update and grounding ───────────────────────────────

    def _model_for(self, action: ActionSpec) -> _ActionModel:
        """Return or create the learned model for this action signature."""
        key = action.model_key
        model = self._models.get(key)
        if model is None:
            model = _ActionModel(
                key=key,
                domain=action.domain,
                name=action.name,
                signature=action.signature,
            )
            self._models[key] = model
        return model

    def model_for(self, action: ActionSpec) -> dict[str, Any]:
        """Return a read-only summary of the learned model, if any."""
        with self._lock:
            model = self._models.get(action.model_key)
            if model is None:
                return {}
            return {
                "attempts": model.attempts,
                "successes": model.successes,
                "success_rate": model.success_rate,
                "expected_success": model.expected_success,
                "failure_causes": dict(model.failure_causes),
                "discrepancy_ema": model.discrepancy_ema,
                "repeated_error_count": model.repeated_error_count,
            }

    def expected_numeric(self, action: ActionSpec, field_name: str) -> float | None:
        """Return the learned numeric expectation for a field, if known."""
        with self._lock:
            model = self._models.get(action.model_key)
            if model is None:
                return None
            stat = model.numeric_stats.get(field_name)
            if stat is None or stat.count == 0:
                return None
            return stat.mean

    def _update_model(self, model: _ActionModel, attempt: ActionAttempt) -> None:
        """Update the durable action model from an observed outcome."""
        actual = attempt.actual
        if actual is None:
            return
        model.attempts += 1
        model.updated_at = time.time()

        ok = actual.get("ok")
        if ok is True:
            model.successes += 1
            model.alpha += 1.0
        elif ok is False:
            model.beta += 1.0

        for field_name, value in actual.fields.items():
            if _is_number(value):
                stat = model.numeric_stats.setdefault(field_name, _NumericStat())
                stat.observe(float(value))
            key = _value_key(value)
            counts = model.outcome_counts.setdefault(field_name, {})
            counts[key] = counts.get(key, 0) + 1

        error_signature = self._error_signature(actual)
        if ok is False:
            cause = self._actual_cause(actual) or "unknown_discrepancy"
            model.failure_causes[cause] = model.failure_causes.get(cause, 0) + 1
            if error_signature and error_signature == model.last_error_signature:
                model.repeated_error_count += 1
            else:
                model.repeated_error_count = 1
            model.last_error_signature = error_signature
        else:
            model.repeated_error_count = 0
            model.last_error_signature = ""

        discrepancy = attempt.discrepancy.magnitude if attempt.discrepancy else 0.0
        alpha = 0.25
        model.discrepancy_ema = (
            (1.0 - alpha) * model.discrepancy_ema + alpha * discrepancy
        )

        self._ground_observation(attempt, model)

    @staticmethod
    def _error_signature(actual: OutcomeState) -> str:
        """Canonical signature for repeated identical failures."""
        parts = [
            str(actual.get("status", "")),
            str(actual.get("cause", "")),
            str(actual.get("error_kind", "")),
        ]
        return "|".join(part for part in parts if part)

    @staticmethod
    def _actual_cause(actual: OutcomeState) -> str:
        """Extract the adapter-provided cause, if present."""
        for field_name in ("cause", "error_kind", "blocked_by"):
            value = actual.get(field_name)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _outcome_key(domain: str, actual: OutcomeState) -> str:
        """Stable outcome-concept key shared by grounding and reward paths."""
        if actual.get("ok") is True:
            return "success"
        cause = ActionFeedbackEngine._actual_cause(actual)
        return cause or str(actual.get("status", "observed"))

    def _ground_observation(
        self, attempt: ActionAttempt, model: _ActionModel,
    ) -> None:
        """Ground observed action/outcome evidence in the network."""
        actual = attempt.actual
        if actual is None:
            return
        action_cid = attempt.action.concept_id
        cause = self._actual_cause(actual)
        outcome_key = self._outcome_key(attempt.action.domain, actual)
        outcome_cid = f"outcome:{attempt.action.domain}:{outcome_key}"

        if self.network is not None:
            self.network.add_concept(
                action_cid,
                confidence=0.55,
                origin="action-feedback",
                properties={
                    "kind": "action",
                    "domain": attempt.action.domain,
                    "name": attempt.action.name,
                },
            )
            if model.signature:
                # The signature-scoped concept lets value learning assign
                # credit at the action/context granularity (e.g. reading
                # a python_file vs reading a directory), while the parent
                # kind-level concept accumulates general evidence.
                self.network.add_concept(
                    f"action:{model.key}",
                    confidence=0.55,
                    origin="action-feedback",
                    properties={
                        "kind": "action_context",
                        "domain": model.domain,
                        "name": model.name,
                        "signature": model.signature,
                    },
                )
                self.network.add_edge(
                    f"action:{model.key}",
                    action_cid,
                    RelationType.IS_A,
                    weight=0.6,
                    origin="observed",
                )
            self.network.add_concept(
                outcome_cid,
                confidence=0.55,
                origin="action-feedback",
                properties={
                    "kind": "outcome",
                    "domain": attempt.action.domain,
                    "outcome": outcome_key,
                },
            )
            self.network.add_edge(
                action_cid,
                outcome_cid,
                RelationType.LEADS_TO,
                weight=0.35,
                origin="observed",
            )
            if cause:
                cause_cid = f"cause:{attempt.action.domain}:{cause}"
                self.network.add_concept(
                    cause_cid,
                    confidence=0.55,
                    origin="action-feedback",
                    properties={
                        "kind": "cause",
                        "domain": attempt.action.domain,
                        "cause": cause,
                    },
                )
                self.network.add_edge(
                    cause_cid,
                    outcome_cid,
                    RelationType.CAUSES,
                    weight=0.4,
                    origin="observed",
                )

        if self.probabilistic is not None:
            try:
                self.probabilistic.observe_evidence(
                    action_cid,
                    RelationType.LEADS_TO.value,
                    outcome_cid,
                    positive=True,
                    weight=0.75,
                )
                if (
                    attempt.prediction.expected.get("ok") is True
                    and actual.get("ok") is False
                ):
                    # The predicted success was directly disconfirmed.
                    self.probabilistic.observe_evidence(
                        action_cid,
                        RelationType.LEADS_TO.value,
                        f"outcome:{attempt.action.domain}:success",
                        positive=False,
                        weight=0.75,
                    )
            except (ArithmeticError, ValueError) as e:
                logger.debug(f"action feedback probabilistic update failed: {e}")

    # ─── Causal diagnosis ─────────────────────────────────────────

    def _diagnose(
        self,
        attempt: ActionAttempt,
        model: _ActionModel,
    ) -> list[CausalHypothesis]:
        """Infer likely causes for the observed discrepancy."""
        actual = attempt.actual
        discrepancy = attempt.discrepancy
        if actual is None or discrepancy is None:
            return []

        hypotheses: dict[str, CausalHypothesis] = {}

        def add(
            cause: str,
            confidence: float,
            evidence: Iterable[str],
            source: str,
        ) -> None:
            """Add or strengthen one causal hypothesis."""
            cause = cause.strip() or "unknown_discrepancy"
            existing = hypotheses.get(cause)
            evidence_list = [e for e in evidence if e]
            if existing is None:
                hypotheses[cause] = CausalHypothesis(
                    cause=cause,
                    confidence=_clamp01(confidence),
                    evidence=evidence_list[:6],
                    source=source,
                )
            else:
                existing.confidence = max(existing.confidence, _clamp01(confidence))
                existing.evidence = list(dict.fromkeys(
                    [*existing.evidence, *evidence_list]
                ))[:8]

        actual_cause = self._actual_cause(actual)
        if actual_cause:
            add(
                actual_cause,
                0.9,
                [actual.summary, str(actual.get("error", ""))],
                "observed",
            )

        for field_name, mismatch in discrepancy.mismatches.items():
            cause = self._cause_for_mismatch(field_name, mismatch, actual)
            add(
                cause,
                0.55 + discrepancy.magnitude * 0.25,
                [
                    f"{field_name}: expected {mismatch.get('expected')!r}, "
                    f"observed {mismatch.get('actual')!r}"
                ],
                "discrepancy",
            )
        for field_name in discrepancy.missing:
            add(
                "missing_observation",
                0.35,
                [f"no observation for expected field {field_name!r}"],
                "discrepancy",
            )

        top = model.top_failure_cause()
        if top is not None and actual.get("ok") is False:
            cause, count = top
            add(
                cause,
                min(0.9, 0.45 + count * 0.1),
                [f"learned failure cause observed {count} times"],
                "learned",
            )

        if self.network is not None and actual_cause:
            outcome_cid = f"outcome:{attempt.action.domain}:{actual_cause}"
            for edge in self.network.get_edges(outcome_cid, "in"):
                if edge.relation == RelationType.CAUSES:
                    add(
                        edge.source,
                        min(0.9, edge.weight),
                        [f"{edge.source} causes {outcome_cid}"],
                        "graph",
                    )

        if self.causal_infer is not None:
            try:
                for hypothesis in self.causal_infer(
                    attempt.action, discrepancy, actual,
                ):
                    add(
                        hypothesis.cause,
                        hypothesis.confidence,
                        hypothesis.evidence,
                        hypothesis.source or "abductive",
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"causal inference adapter failed: {e}")

        if not hypotheses:
            add(
                "unknown_discrepancy",
                0.25,
                ["prediction differed but no direct cause was observed"],
                "diagnostic",
            )

        ranked = sorted(
            hypotheses.values(),
            key=lambda h: (h.source != "observed", -h.confidence, h.cause),
        )
        return ranked[:8]

    @staticmethod
    def _cause_for_mismatch(
        field_name: str,
        mismatch: dict[str, Any],
        actual: OutcomeState,
    ) -> str:
        """Map a structured mismatch to a conservative cause category."""
        expected = mismatch.get("expected")
        observed = mismatch.get("actual")
        if field_name in {"exists", "target_exists", "path_exists"}:
            if expected is True and observed is False:
                return "missing_target"
            if expected is False and observed is True:
                return "unexpected_target"
        if field_name in {"ok", "verified", "tests_passed", "syntax_ok", "readable", "writable"}:
            if expected is True and observed is False:
                return ActionFeedbackEngine._actual_cause(actual) or "execution_failed"
            if expected is False and observed is True:
                return "unexpected_success"
        if field_name == "status":
            if observed in {"blocked", "cancelled", "awaiting_confirmation"}:
                return str(observed)
            return "unexpected_status"
        if field_name == "knowledge_nonempty":
            return "insufficient_evidence"
        if field_name in {"level", "delta"}:
            if _is_number(expected) and _is_number(observed):
                if observed < expected:
                    return "regulation_undershoot"
                if observed > expected:
                    return "regulation_overshoot"
            return "competing_dynamics"
        return "unexpected_outcome"

    # ─── Revision / retry ─────────────────────────────────────────

    def recommend(
        self,
        attempt: ActionAttempt,
        *,
        alternatives: Iterable[ActionSpec] | None = None,
        repair: bool = True,
    ) -> ActionRevision:
        """Choose a bounded next action from feedback.

        The recommendation is intentionally conservative: unobserved
        actions wait, terminal/unsafe causes stop, transient causes can
        retry, alternatives are selected only when they have a better
        learned or domain-supplied expected value, and repeated identical
        failures are not blindly repeated.
        """
        with self._lock:
            return self._recommend(
                attempt,
                alternatives=alternatives,
                repair=repair,
            )

    def _recommend(
        self,
        attempt: ActionAttempt,
        *,
        alternatives: Iterable[ActionSpec] | None = None,
        repair: bool = True,
    ) -> ActionRevision:
        """Choose the bounded next action; caller holds ``self._lock``."""
        actual = attempt.actual
        discrepancy = attempt.discrepancy
        if actual is None or not actual.observed:
            return ActionRevision(
                kind=RevisionKind.WAIT,
                reason="post-action state was not independently observed",
                confidence=0.3,
            )
        if discrepancy is None:
            return ActionRevision(kind=RevisionKind.NONE)

        ok = actual.get("ok")
        if ok is True:
            if discrepancy.magnitude >= 0.5:
                return ActionRevision(
                    kind=RevisionKind.DIAGNOSE,
                    reason="action succeeded but produced an unexpected state",
                    confidence=discrepancy.magnitude,
                )
            return ActionRevision(kind=RevisionKind.NONE)

        cause = self._actual_cause(actual)
        if cause in _TERMINAL_CAUSES:
            return ActionRevision(
                kind=RevisionKind.AVOID,
                reason=f"failure cause {cause} should not be retried",
                confidence=0.75,
            )
        if cause in _WAIT_CAUSES:
            return ActionRevision(
                kind=RevisionKind.WAIT,
                reason=f"failure cause {cause} should wait before retrying",
                confidence=0.65,
            )

        # ``retryable: False`` forbids repeating *this* action, but a
        # different action (diagnostic probe, alternative approach) may
        # still be the right next step — fall through to alternatives.
        if actual.get("retryable") is not False and (
            cause in _TRANSIENT_CAUSES or actual.get("retryable") is True
        ):
            revised = self._retry_action(attempt)
            return ActionRevision(
                kind=RevisionKind.RETRY,
                action=revised,
                reason=f"transient cause {cause or 'retryable'}",
                confidence=0.65,
            )

        candidate = self._choose_alternative(attempt, alternatives)
        if candidate is not None:
            return ActionRevision(
                kind=RevisionKind.CHOOSE,
                action=candidate,
                reason="learned action model prefers an alternative",
                confidence=0.65,
            )

        if repair and self.repair_suggester is not None:
            try:
                repair_action = self.repair_suggester(
                    attempt.action,
                    attempt.hypotheses,
                    actual,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"repair suggester failed: {e}")
                repair_action = None
            if repair_action is not None:
                return ActionRevision(
                    kind=RevisionKind.DIAGNOSE,
                    action=repair_action,
                    reason="diagnostic action may identify the cause",
                    confidence=0.55,
                )

        if model_repeated_failure(self._model_for(attempt.action)):
            return ActionRevision(
                kind=RevisionKind.AVOID,
                reason="the same failure signature repeated",
                confidence=0.7,
            )

        return ActionRevision(
            kind=RevisionKind.DIAGNOSE,
            reason=(
                attempt.hypotheses[0].cause
                if attempt.hypotheses
                else "unknown discrepancy"
            ),
            confidence=max(0.35, discrepancy.magnitude),
        )

    def _retry_action(self, attempt: ActionAttempt) -> ActionSpec:
        """Build a revised retry action with bounded parameter changes."""
        parameters = dict(attempt.action.parameters)
        timeout = parameters.get("timeout")
        if _is_number(timeout):
            parameters["timeout"] = min(120.0, float(timeout) * 2.0)
        parameters["retry_count"] = int(parameters.get("retry_count", 0)) + 1
        return attempt.action.with_parameters(**parameters)

    def _choose_alternative(
        self,
        attempt: ActionAttempt,
        alternatives: Iterable[ActionSpec] | None,
    ) -> ActionSpec | None:
        """Select the best-scoring different candidate action."""
        if alternatives is None:
            return None
        candidates = [
            candidate for candidate in alternatives
            if candidate.model_key != attempt.action.model_key
            or candidate.parameters != attempt.action.parameters
        ]
        if not candidates:
            return None
        current_score = self._action_score(attempt.action)
        ranked = sorted(
            candidates,
            key=lambda candidate: self._action_score(candidate),
            reverse=True,
        )
        best = ranked[0]
        best_score = self._action_score(best)
        # An alternative must be meaningfully better than the failed
        # action, or the failed action must have a very poor learned
        # record. This avoids random strategy churn on one surprise.
        if best_score > current_score + 0.08 or current_score < 0.35:
            return best
        return None

    def _action_score(self, action: ActionSpec) -> float:
        """Score an action candidate from learned evidence.

        The per-signature model supplies the primary score; the optional
        ``value_estimator`` (TD value over the action's concept state)
        contributes a blended term so reward learning biases alternative
        selection toward actions whose outcomes were actually better.
        """
        model = self._models.get(action.model_key)
        if model is None or model.attempts == 0:
            score = 0.55
        else:
            score = model.expected_success
            score -= min(0.2, model.discrepancy_ema * 0.15)
            top = model.top_failure_cause()
            if top is not None:
                _cause, count = top
                score -= min(0.2, count * 0.03)
            score = _clamp01(score)
        if self.value_estimator is not None:
            try:
                value = self.value_estimator(action)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"action value estimator failed: {e}")
                value = None
            if value is not None:
                score = _clamp01(0.65 * score + 0.35 * _clamp01(value))
        return score

    # ─── Convenience execution loop ───────────────────────────────

    def run_action(
        self,
        action: ActionSpec,
        *,
        execute: Callable[[ActionSpec, ActionAttempt], Any],
        observe: Callable[[ActionSpec, Any, ActionAttempt], OutcomeState | dict[str, Any]],
        expected: OutcomeState | dict[str, Any] | None = None,
        alternatives: Iterable[ActionSpec] | None = None,
        max_attempts: int = 1,
    ) -> list[ActionAttempt]:
        """Execute an action-feedback loop with bounded revision.

        ``execute`` performs the action. ``observe`` must read the actual
        post-action world/tool result and convert it to an OutcomeState.
        The learner does not assume success from the action request.
        """
        attempts: list[ActionAttempt] = []
        current = action
        for _ in range(max(1, max_attempts)):
            attempt = self.begin(
                current,
                expected=expected,
                retry_of=attempts[-1].attempt_id if attempts else "",
            )
            raw_result = execute(current, attempt)
            actual = observe(current, raw_result, attempt)
            attempt = self.observe(
                attempt,
                actual,
                alternatives=alternatives,
            )
            attempts.append(attempt)
            revision = attempt.revision
            if (
                revision is None
                or revision.kind not in {RevisionKind.RETRY, RevisionKind.CHOOSE}
                or revision.action is None
            ):
                break
            current = revision.action
        return attempts

    # ─── Persistence / introspection ──────────────────────────────

    def statistics(self) -> dict[str, Any]:
        """Return aggregate learning statistics."""
        with self._lock:
            total_attempts = sum(
                model.attempts for model in self._models.values()
            )
            total_successes = sum(
                model.successes for model in self._models.values()
            )
            return {
                "models": len(self._models),
                "pending": len(self._pending),
                "history": len(self._history),
                "attempts": total_attempts,
                "successes": total_successes,
                "success_rate": (
                    total_successes / total_attempts if total_attempts else 0.0
                ),
            }

    def to_dict(self) -> dict[str, Any]:
        """Serialize learned action models and bounded recent history."""
        with self._lock:
            return {
                "version": 1,
                "counter": self._counter,
                "models": {
                    key: model.to_dict() for key, model in self._models.items()
                },
                "history": [
                    attempt.to_dict() for attempt in self._history[-100:]
                ],
            }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore learned action models from persistence."""
        if not isinstance(data, dict):
            return
        with self._lock:
            self._counter = int(data.get("counter", 0))
            models_raw = data.get("models", {})
            self._models = {}
            if isinstance(models_raw, dict):
                for key, model_data in models_raw.items():
                    if not isinstance(model_data, dict):
                        continue
                    model = _ActionModel.from_dict(model_data)
                    model.key = model.key or str(key)
                    self._models[model.key] = model
            history_raw = data.get("history", [])
            self._history = []
            self._pending = {}
            if isinstance(history_raw, list):
                for attempt_data in history_raw[-_MAX_HISTORY:]:
                    if isinstance(attempt_data, dict):
                        self._history.append(ActionAttempt.from_dict(attempt_data))
        # Pending attempts are intentionally not restored: after a
        # restart there is no reliable post-action observation window.


def model_repeated_failure(model: _ActionModel) -> bool:
    """Whether the same failure signature has repeated."""
    return model.repeated_error_count >= 2


def _clamp01(value: float) -> float:
    """Clamp a score to [0, 1]."""
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _is_number(value: Any) -> TypeGuard[int | float]:
    """Whether a value is numeric but not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _values_match(expected: Any, actual: Any) -> bool:
    """Domain-neutral comparison with numeric tolerance."""
    if _is_number(expected) and _is_number(actual):
        expected_f = float(expected)
        actual_f = float(actual)
        tolerance = max(0.025, abs(expected_f) * 0.10)
        return abs(actual_f - expected_f) <= tolerance
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    if isinstance(expected, (list, tuple, set)) and isinstance(actual, (list, tuple, set)):
        try:
            return set(expected) == set(actual)
        except TypeError:
            return list(expected) == list(actual)
    return expected == actual


def _value_key(value: Any) -> str:
    """Canonical JSON-ish key for an observed outcome value."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _decode_value(value_key: str) -> Any:
    """Decode a value key back to a Python value."""
    try:
        return json.loads(value_key)
    except (TypeError, ValueError):
        return value_key

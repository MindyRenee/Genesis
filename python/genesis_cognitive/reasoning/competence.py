"""Task competence — domain-general situation → action → skill memory.

This module is the shared substrate underneath domain-specific solvers.
It does not know about grids, tools, code, or conversation. A domain
adapter supplies structured observations and operator names; the
substrate supplies the parts every task needs:

- **Schema recognition:** group states by their structural features,
  available actions, entities, and success conditions rather than by
  surface identity.
- **Affordance learning:** learn which state variables an action
  changes, and with what confidence.
- **Prediction checking:** compare expected effects with observed
  transitions so "solved" means the world actually satisfied a goal.
- **Skill consolidation:** retain verified action sequences and
  retrieve them on structurally similar tasks.

The design separates *schema* from *binding*: the signature says what
kind of situation this is; the concrete state values bind that schema
to the current instance. That mirrors current evidence that task-type
recognition and instance bindings are dissociable, while remaining
small enough to persist as ordinary JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

Scalar = str | int | float | bool | None
State = Mapping[str, Any]

logger = logging.getLogger(__name__)

_MATCH_THRESHOLD = 0.55
_RELIABILITY_OBSERVATIONS = 3.0
# A foreign-domain skill is offered when the task signatures share
# enough schema-level role vocabulary — "constrain", "select" — for an
# analogy to be claimed at all. Variable names can never overlap across
# domains, so roles are the only bridge. 0.2 lets the
# constraint-satisfaction families (sorter/relations/assembly/
# classification) see each other while keeping unrelated families
# (sequence continuation, quantity composition, grid transforms)
# out of reach.
_CROSS_DOMAIN_ROLES = 0.2


def _publishes_evidence(skill: LearnedSkill) -> bool:
    """Whether a foreign skill is readable at all.

    Vocabulary overlap alone is not an analogy — a skill crosses
    domains only when its procedure exposes the normalized
    "support" axis (fraction of an option's constraints satisfied)
    that foreign adapters can rebind onto their own affordance.
    """
    return any(
        isinstance(step.parameters.get("support"), int | float)
        for step in skill.steps
    )

# How an episode's success was established, weakest to strongest:
# - "reported": a caller asserted success; nothing checked it.
# - "solver": an epistemic verifier (e.g. ProblemSolver) confirmed —
#   knowledge-level truth, not observed world change.
# - "external": a world-state check observed the outcome — exact
#   replay, an environment's terminal condition, or a real actuator.
VERIFICATION_KINDS: tuple[str, ...] = ("reported", "solver", "external")
_VERIFICATION_RANK = {kind: rank for rank, kind in enumerate(VERIFICATION_KINDS)}
# Transfer weight by evidence strength — a procedure verified only
# epistemically carries less authority than one checked against the
# world, and an unverified report carries least.
_VERIFICATION_WEIGHT = {"reported": 0.5, "solver": 0.8, "external": 1.0}

# Neuromodulatory consolidation floor — episodes encoded under
# dopamine/arousal below this stay episodic; they don't
# proceduralize into skills.
_SALIENCE_FLOOR = 0.2


def verification_rank(kind: str) -> int:
    """Rank of a verification kind; unknown kinds count as reported."""
    return _VERIFICATION_RANK.get(kind, 0)


def verification_weight(kind: str) -> float:
    """Transfer-confidence weight of a verification kind."""
    return _VERIFICATION_WEIGHT.get(kind, 0.5)


def _canonical(value: Any) -> str:
    """Return a stable, JSON-safe representation of structured data."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(parts: Any) -> str:
    return hashlib.sha256(_canonical(parts).encode("utf-8")).hexdigest()[:16]


def _scalar(value: Any) -> Scalar:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return _canonical(value)


def _flatten_state(state: State, prefix: str = "") -> dict[str, Scalar]:
    """Flatten a structured observation into named scalar variables."""
    flat: dict[str, Scalar] = {}
    for key, value in state.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_state(value, name))
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            if all(
                item is None or isinstance(item, str | int | float | bool)
                for item in value
            ):
                flat[name] = _canonical(list(value))
            else:
                for i, item in enumerate(value):
                    if isinstance(item, Mapping):
                        flat.update(_flatten_state(item, f"{name}.{i}"))
                    else:
                        flat[f"{name}.{i}"] = _scalar(item)
        elif isinstance(value, set | frozenset):
            flat[name] = _canonical(sorted(value, key=repr))
        else:
            flat[name] = _scalar(value)
    return flat


def _feature_set(state: State) -> frozenset[str]:
    """Extract schema-level features from a state observation.

    Concrete values are bindings, not schema identity. The signature
    therefore keeps variable names and qualitative buckets rather than
    every literal value.
    """
    features: set[str] = set()
    for key, value in _flatten_state(state).items():
        features.add(key)
        if isinstance(value, bool):
            if value:
                features.add(f"{key}:true")
        elif isinstance(value, int | float):
            if value == 0:
                bucket = "zero"
            elif value > 0:
                bucket = "positive"
            else:
                bucket = "negative"
            features.add(f"{key}:{bucket}")
        elif isinstance(value, str) and value:
            features.add(f"{key}:{value}")
    return frozenset(features)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


@dataclass(frozen=True, slots=True)
class GoalCondition:
    """A serializable success predicate over a state variable."""

    key: str
    op: str = "eq"
    value: Any = True
    weight: float = 1.0
    tolerance: float = 1e-9

    def descriptor(self) -> str:
        return f"{self.key}|{self.op}|{_canonical(self.value)}"

    def score(self, state: Mapping[str, Scalar]) -> float:
        """Return how satisfied this condition is, in ``[0, 1]``."""
        present = self.key in state and state[self.key] is not None
        actual = state.get(self.key)
        if self.op == "exists":
            return 1.0 if present else 0.0
        if self.op == "missing":
            return 0.0 if present else 1.0
        if not present:
            return 0.0
        if self.op == "truthy":
            return 1.0 if bool(actual) else 0.0
        if self.op == "contains":
            try:
                return 1.0 if self.value in actual else 0.0  # type: ignore[operator]
            except TypeError:
                return 0.0

        a_num = _as_float(actual)
        e_num = _as_float(self.value)
        if self.op == "eq":
            if a_num is not None and e_num is not None:
                scale = max(abs(e_num), self.tolerance, 1.0)
                return max(0.0, 1.0 - abs(a_num - e_num) / scale)
            return 1.0 if actual == self.value else 0.0
        if self.op == "neq":
            return 0.0 if actual == self.value else 1.0
        if a_num is None or e_num is None:
            return 0.0
        if self.op == "lt":
            return 1.0 if a_num < e_num else 0.0
        if self.op == "le":
            return 1.0 if a_num <= e_num else 0.0
        if self.op == "gt":
            return 1.0 if a_num > e_num else 0.0
        if self.op == "ge":
            return 1.0 if a_num >= e_num else 0.0
        return 0.0

    def satisfied(self, state: Mapping[str, Scalar]) -> bool:
        return self.score(state) >= 1.0 - 1e-9

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "op": self.op,
            "value": self.value,
            "weight": self.weight,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoalCondition:
        return cls(
            key=str(data["key"]),
            op=str(data.get("op", "eq")),
            value=data.get("value", True),
            weight=float(data.get("weight", 1.0)),
            tolerance=float(data.get("tolerance", 1e-9)),
        )


@dataclass(frozen=True, slots=True)
class TaskSignature:
    """The abstract identity of a task, independent of bound values.

    ``roles`` is the schema-level vocabulary: domain-independent
    structural tags an adapter emits (``"self"``, ``"navigate"``,
    ``"procedure"``…). Where ``state``/``actions``/``entities`` carry
    domain terms that can never match across domains, roles are what
    lets "approach a target while avoiding movers" in one engine look
    like the same family in another.
    """

    domain: str
    state: frozenset[str]
    actions: frozenset[str]
    goals: frozenset[str]
    entities: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()

    def similarity(self, other: TaskSignature) -> float:
        """Structural similarity across state, affordances, and goals."""
        # Roles only count when at least one side claims them — two
        # role-less signatures are not thereby identical.
        role_overlap = (
            _jaccard(self.roles, other.roles)
            if self.roles or other.roles
            else 0.0
        )
        score = (
            0.35 * _jaccard(self.state, other.state)
            + 0.16 * _jaccard(self.actions, other.actions)
            + 0.24 * _jaccard(self.goals, other.goals)
            + 0.08 * _jaccard(self.entities, other.entities)
            + 0.17 * role_overlap
        )
        # A different domain can still be analogous, but it should not
        # look identical merely because the feature names overlap.
        return score if self.domain == other.domain else score * 0.72

    def key(self) -> str:
        return _hash(
            {
                "domain": self.domain,
                "state": sorted(self.state),
                "actions": sorted(self.actions),
                "goals": sorted(self.goals),
                "entities": sorted(self.entities),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "state": sorted(self.state),
            "actions": sorted(self.actions),
            "goals": sorted(self.goals),
            "entities": sorted(self.entities),
            "roles": sorted(self.roles),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskSignature:
        return cls(
            domain=str(data.get("domain", "general")),
            state=frozenset(str(v) for v in data.get("state", [])),
            actions=frozenset(str(v) for v in data.get("actions", [])),
            goals=frozenset(str(v) for v in data.get("goals", [])),
            entities=frozenset(str(v) for v in data.get("entities", [])),
            roles=frozenset(str(v) for v in data.get("roles", [])),
        )


@dataclass(slots=True)
class EffectPrediction:
    """A predicted state-variable change for one action."""

    key: str
    kind: str
    expected: Scalar
    confidence: float
    delta: float | None = None

    def matches(self, actual: Scalar) -> bool:
        if self.kind == "delta":
            a_num = _as_float(actual)
            e_num = _as_float(self.expected)
            return a_num is not None and e_num is not None and abs(a_num - e_num) <= 1e-6
        return actual == self.expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "expected": self.expected,
            "confidence": self.confidence,
            "delta": self.delta,
        }


@dataclass(slots=True)
class TransitionCheck:
    """Expected-vs-observed result for one action."""

    action: str
    predicted: tuple[EffectPrediction, ...]
    observed_changes: tuple[str, ...]
    score: float
    mismatches: tuple[str, ...]


@dataclass(slots=True)
class EffectModel:
    """Learned effect distribution for one state variable."""

    key: str
    observations: int = 0
    unchanged: int = 0
    changed: int = 0
    numeric_changes: int = 0
    delta_sum: float = 0.0
    to_counts: dict[str, int] = field(default_factory=dict)
    to_values: dict[str, Scalar] = field(default_factory=dict)

    def observe(self, before: Scalar, after: Scalar) -> None:
        self.observations += 1
        if before == after:
            self.unchanged += 1
            return
        self.changed += 1
        b_num = _as_float(before)
        a_num = _as_float(after)
        if b_num is not None and a_num is not None:
            self.numeric_changes += 1
            self.delta_sum += a_num - b_num
            outcome = "delta"
            value: Scalar = None
        elif before is None:
            outcome = "added"
            value = after
        elif after is None:
            outcome = "removed"
            value = None
        else:
            outcome = "set"
            value = after
        canonical = _canonical(value)
        key = f"{outcome}:{canonical}"
        self.to_counts[key] = self.to_counts.get(key, 0) + 1
        self.to_values[key] = value

    def predict(self, before: Scalar) -> EffectPrediction | None:
        if self.observations == 0:
            return None
        reliability = min(1.0, self.observations / _RELIABILITY_OBSERVATIONS)
        if self.unchanged >= self.changed:
            confidence = (self.unchanged / self.observations) * reliability
            return EffectPrediction(
                self.key, "unchanged", before, confidence, delta=0.0
            )
        outcome, count = max(self.to_counts.items(), key=lambda kv: kv[1])
        confidence = (count / self.observations) * reliability
        if outcome.startswith("delta:"):
            delta = self.delta_sum / max(1, self.numeric_changes)
            b_num = _as_float(before)
            expected: Scalar = None if b_num is None else b_num + delta
            return EffectPrediction(
                self.key, "delta", expected, confidence, delta=delta
            )
        return EffectPrediction(
            self.key,
            outcome.split(":", 1)[0],
            self.to_values.get(outcome),
            confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "observations": self.observations,
            "unchanged": self.unchanged,
            "changed": self.changed,
            "numeric_changes": self.numeric_changes,
            "delta_sum": self.delta_sum,
            "to_counts": dict(self.to_counts),
            "to_values": dict(self.to_values),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EffectModel:
        return cls(
            key=str(data["key"]),
            observations=int(data.get("observations", 0)),
            unchanged=int(data.get("unchanged", 0)),
            changed=int(data.get("changed", 0)),
            numeric_changes=int(data.get("numeric_changes", 0)),
            delta_sum=float(data.get("delta_sum", 0.0)),
            to_counts={
                str(k): int(v) for k, v in data.get("to_counts", {}).items()
            },
            to_values={
                str(k): _scalar(v) for k, v in data.get("to_values", {}).items()
            },
        )


@dataclass(slots=True)
class OperatorModel:
    """Learned affordance model for one named action/operator."""

    action: str
    uses: int = 0
    successes: int = 0
    failures: int = 0
    effects: dict[str, EffectModel] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return (self.successes + 0.5) / (self.uses + 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.failures,
            "effects": [e.to_dict() for e in self.effects.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperatorModel:
        op = cls(
            action=str(data["action"]),
            uses=int(data.get("uses", 0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
        )
        for effect in data.get("effects", []):
            model = EffectModel.from_dict(effect)
            op.effects[model.key] = model
        return op


@dataclass(slots=True)
class TaskSchema:
    """A learned task frame: signature plus operators and skills."""

    schema_id: str
    signature: TaskSignature
    episodes: int = 0
    successes: int = 0
    failures: int = 0
    operators: dict[str, OperatorModel] = field(default_factory=dict)
    skill_ids: list[str] = field(default_factory=list)
    last_score: float = 0.0
    # Which proposal families solved tasks of this schema — the
    # learned search prior. "Learning to search": the solver's
    # proposal ordering adapts per task family, not just globally.
    family_priors: dict[str, int] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return (self.successes + 0.5) / (self.episodes + 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "signature": self.signature.to_dict(),
            "episodes": self.episodes,
            "successes": self.successes,
            "failures": self.failures,
            "operators": [op.to_dict() for op in self.operators.values()],
            "skill_ids": list(self.skill_ids),
            "last_score": self.last_score,
            "family_priors": dict(self.family_priors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskSchema:
        schema = cls(
            schema_id=str(data["schema_id"]),
            signature=TaskSignature.from_dict(data.get("signature", {})),
            episodes=int(data.get("episodes", 0)),
            successes=int(data.get("successes", 0)),
            failures=int(data.get("failures", 0)),
            skill_ids=[str(s) for s in data.get("skill_ids", [])],
            last_score=float(data.get("last_score", 0.0)),
            family_priors={
                str(k): int(v)
                for k, v in dict(data.get("family_priors", {})).items()
            },
        )
        for op_data in data.get("operators", []):
            op = OperatorModel.from_dict(op_data)
            schema.operators[op.action] = op
        return schema


@dataclass(frozen=True, slots=True)
class ProcedureStep:
    """One serializable step in a reusable procedure."""

    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    family: str = ""
    description: str = ""

    @property
    def label(self) -> str:
        if self.description:
            return self.description
        if not self.parameters:
            return self.action
        args = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.action}({args})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "parameters": self.parameters,
            "family": self.family,
            "description": self.description or self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProcedureStep:
        return cls(
            action=str(data["action"]),
            parameters=dict(data.get("parameters", {})),
            family=str(data.get("family", "")),
            description=str(data.get("description", "")),
        )


@dataclass(slots=True)
class LearnedSkill:
    """A verified procedure reusable under a task schema.

    ``verification`` records the strongest evidence behind the skill:
    "reported" (asserted), "solver" (epistemically verified), or
    "external" (checked against observed world state).
    """

    skill_id: str
    signature: TaskSignature
    steps: tuple[ProcedureStep, ...]
    success_conditions: tuple[GoalCondition, ...]
    uses: int = 1
    successes: int = 1
    failures: int = 0
    verification_score: float = 1.0
    verification: str = "reported"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def confidence(self) -> float:
        return (self.successes + 0.5) / (self.uses + self.failures + 1.0)

    def step_labels(self) -> tuple[str, ...]:
        return tuple(step.label for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "signature": self.signature.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "success_conditions": [g.to_dict() for g in self.success_conditions],
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.failures,
            "verification_score": self.verification_score,
            "verification": self.verification,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LearnedSkill:
        return cls(
            skill_id=str(data["skill_id"]),
            signature=TaskSignature.from_dict(data.get("signature", {})),
            steps=tuple(
                ProcedureStep.from_dict(s) for s in data.get("steps", [])
            ),
            success_conditions=tuple(
                GoalCondition.from_dict(g)
                for g in data.get("success_conditions", [])
            ),
            uses=int(data.get("uses", 1)),
            successes=int(data.get("successes", 1)),
            failures=int(data.get("failures", 0)),
            verification_score=float(data.get("verification_score", 1.0)),
            # Older saves carry no provenance — "reported" is the
            # honest default, not an upgrade to "external".
            verification=str(data.get("verification", "reported")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )


@dataclass(slots=True)
class SkillMatch:
    """A retrieved skill plus its structural match evidence."""

    skill: LearnedSkill
    similarity: float
    score: float


@dataclass(slots=True)
class TaskContext:
    """The result of recognizing a task instance."""

    signature: TaskSignature
    schema: TaskSchema
    similarity: float
    novel: bool
    skills: tuple[SkillMatch, ...] = ()


class TaskCompetence:
    """Domain-general task-schema, transition, and skill memory.

    Adapters remain responsible for acting in the world. This class
    owns the shared structures that let an adapter recognize a task,
    learn what actions do, verify outcomes, and reuse a verified
    procedure on a structurally similar task.
    """

    def __init__(
        self,
        *,
        match_threshold: float = _MATCH_THRESHOLD,
        network: ConceptNetwork | None = None,
        salience_getter: Callable[[], float] | None = None,
    ) -> None:
        self.match_threshold = match_threshold
        self.network = network
        # Neuromodulation is broadcast, not addressed — every adapter
        # that consolidates reads the same ambient dopamine/arousal
        # state instead of each caller threading it through.
        self.salience_getter = salience_getter
        self.schemas: dict[str, TaskSchema] = {}
        self.skills: dict[str, LearnedSkill] = {}
        self.transition_count = 0

    def _ambient_salience(self) -> float:
        """Current neuromodulatory gain; 1.0 when unmodulated."""
        if self.salience_getter is None:
            return 1.0
        try:
            return max(0.0, min(1.0, float(self.salience_getter())))
        except Exception:  # noqa: BLE001 — a broken gauge must not
            return 1.0     # block consolidation; treat as unmodulated

    def make_signature(
        self,
        *,
        domain: str,
        state: State,
        actions: Iterable[str] = (),
        goal_conditions: Iterable[GoalCondition | str] = (),
        entities: Iterable[str] = (),
        roles: Iterable[str] = (),
    ) -> TaskSignature:
        goals = frozenset(
            g.descriptor() if isinstance(g, GoalCondition) else str(g)
            for g in goal_conditions
        )
        # Goal *shape* is schema-level vocabulary: "drive a variable
        # to equality" matches across domains even when the variable
        # names don't. Bound descriptors stay in ``goals``.
        role_set = {str(r) for r in roles}
        for g in goal_conditions:
            if isinstance(g, GoalCondition):
                role_set.add(f"goal:{g.op}")
        return TaskSignature(
            domain=domain,
            state=_feature_set(state),
            actions=frozenset(str(a) for a in actions),
            goals=goals,
            entities=frozenset(str(e) for e in entities),
            roles=frozenset(role_set),
        )

    def recognize(
        self,
        *,
        domain: str,
        state: State,
        actions: Iterable[str] = (),
        goal_conditions: Iterable[GoalCondition | str] = (),
        entities: Iterable[str] = (),
        roles: Iterable[str] = (),
    ) -> TaskContext:
        """Recognize a situation as an existing schema or a novel one."""
        signature = self.make_signature(
            domain=domain,
            state=state,
            actions=actions,
            goal_conditions=goal_conditions,
            entities=entities,
            roles=roles,
        )
        schema_id = signature.key()
        exact = self.schemas.get(schema_id)
        best: tuple[TaskSchema, float] | None = None
        if exact is None:
            for schema in self.schemas.values():
                similarity = signature.similarity(schema.signature)
                if best is None or similarity > best[1]:
                    best = (schema, similarity)
        if exact is not None:
            schema, similarity, novel = exact, 1.0, False
        elif best is not None and best[1] >= self.match_threshold:
            schema, similarity, novel = best[0], best[1], False
        else:
            schema = TaskSchema(schema_id=schema_id, signature=signature)
            self.schemas[schema_id] = schema
            similarity, novel = 0.0, True
        return TaskContext(
            signature=signature,
            schema=schema,
            similarity=similarity,
            novel=novel,
            skills=tuple(self.skill_matches(signature)),
        )

    def skill_matches(
        self, signature: TaskSignature, *, limit: int = 4
    ) -> list[SkillMatch]:
        """Retrieve reusable skills, vocabulary-aligned first.

        Skills that clear full-signature similarity are offered
        first — including foreign skills whose concrete vocabulary
        genuinely aligns (the domain discount lives in
        ``TaskSignature.similarity``).

        Skills from other domains that cannot clear the full bar can
        still transfer on abstract structure alone: shared schema-level
        roles plus a readable procedure interface — a "support" axis
        the foreign adapter can rebind. For a cross match,
        ``SkillMatch.similarity`` carries the role overlap, not the
        full-signature score. Foreign skills are always ranked below
        similarity matches and capped separately, so they can inform
        but never crowd out vocabulary-aligned evidence.
        """
        same: list[SkillMatch] = []
        cross: list[SkillMatch] = []
        for skill in self.skills.values():
            similarity = signature.similarity(skill.signature)
            if similarity >= self.match_threshold:
                score = 0.68 * similarity + 0.32 * skill.confidence
                # Procedures backed by weaker verification transfer with
                # less authority: a reported or solver-only skill must
                # not outrank one checked against the world.
                score *= verification_weight(skill.verification)
                same.append(SkillMatch(skill, similarity, score))
                continue
            if skill.signature.domain == signature.domain:
                continue
            overlap = _jaccard(signature.roles, skill.signature.roles)
            if overlap < _CROSS_DOMAIN_ROLES or not _publishes_evidence(
                skill
            ):
                continue
            score = 0.68 * overlap + 0.32 * skill.confidence
            score *= verification_weight(skill.verification)
            cross.append(SkillMatch(skill, overlap, score))
        same.sort(key=lambda m: (-m.score, m.skill.skill_id))
        cross.sort(key=lambda m: (-m.score, m.skill.skill_id))
        return (same[: limit - 1] + cross[:2])[:limit]

    def predict(
        self,
        context: TaskContext,
        action: str,
        state: State,
    ) -> tuple[EffectPrediction, ...]:
        """Predict the state variables an action is likely to change."""
        op = context.schema.operators.get(action)
        if op is None:
            return ()
        before = _flatten_state(state)
        predictions: list[EffectPrediction] = []
        for key, effect in op.effects.items():
            prediction = effect.predict(before.get(key))
            if prediction is not None and prediction.kind != "unchanged":
                predictions.append(prediction)
        predictions.sort(key=lambda p: (-p.confidence, p.key))
        return tuple(predictions)

    def record_transition(
        self,
        context: TaskContext,
        action: str,
        before: State,
        after: State,
        *,
        success: bool | None = None,
    ) -> TransitionCheck:
        """Learn an action's effects from one observed transition.

        The returned check scores the prediction made *before* the new
        evidence was incorporated, so a surprising transition is visible
        as a low score rather than silently rewritten.
        """
        predicted = self.predict(context, action, before)
        before_flat = _flatten_state(before)
        after_flat = _flatten_state(after)
        observed_changes = tuple(
            key
            for key in before_flat.keys() | after_flat.keys()
            if before_flat.get(key) != after_flat.get(key)
        )
        correct = 0
        mismatches: list[str] = []
        predicted_keys = {p.key for p in predicted}
        for prediction in predicted:
            actual = after_flat.get(prediction.key)
            if prediction.matches(actual):
                correct += 1
            else:
                mismatches.append(
                    f"{prediction.key}: expected {prediction.expected!r}, "
                    f"observed {actual!r}"
                )
        missed = [k for k in observed_changes if k not in predicted_keys]
        mismatches.extend(f"{k}: unpredicted change" for k in missed)
        denominator = correct + len(mismatches)
        score = correct / denominator if denominator else 1.0

        op = context.schema.operators.setdefault(action, OperatorModel(action))
        op.uses += 1
        if success is True:
            op.successes += 1
        elif success is False:
            op.failures += 1
        for key in before_flat.keys() | after_flat.keys():
            effect = op.effects.setdefault(key, EffectModel(key))
            effect.observe(before_flat.get(key), after_flat.get(key))
        self.transition_count += 1
        return TransitionCheck(
            action=action,
            predicted=predicted,
            observed_changes=observed_changes,
            score=score,
            mismatches=tuple(mismatches),
        )

    def verify(
        self,
        goal_conditions: Iterable[GoalCondition],
        state: State,
    ) -> float:
        """Weighted satisfaction of explicit success predicates."""
        flat = _flatten_state(state)
        conditions = list(goal_conditions)
        if not conditions:
            return 0.0
        total = sum(max(0.0, g.weight) for g in conditions) or 1.0
        return sum(g.score(flat) * g.weight for g in conditions) / total

    def record_episode(
        self,
        context: TaskContext,
        *,
        steps: Iterable[ProcedureStep],
        success: bool,
        verification_score: float,
        goal_conditions: Iterable[GoalCondition] = (),
        verification: str = "reported",
        salience: float | None = None,
    ) -> LearnedSkill | None:
        """Consolidate a verified episode into a reusable skill.

        Failed episodes update schema statistics but do not become
        skills; a procedure is reusable only after a verifier says the
        outcome satisfied the task. ``verification`` records *how*
        success was established ("reported" < "solver" < "external") —
        a skill never claims stronger evidence than it received, and
        later stronger evidence upgrades the record.

        ``salience`` is the neuromodulatory gate: dopamine/arousal
        state at encoding decides whether a verified episode
        *proceduralizes*. Below the floor the episode still enters the
        schema ledger — episodic memory forms — but no skill is
        written. Low-salience experience is remembered, not learned.
        When omitted, the ambient broadcast level is read.
        """
        if salience is None:
            salience = self._ambient_salience()
        schema = context.schema
        schema.episodes += 1
        schema.last_score = max(0.0, min(1.0, verification_score))
        if success:
            schema.successes += 1
        else:
            schema.failures += 1
        if (
            not success
            or verification_score < 0.999
            or salience < _SALIENCE_FLOOR
        ):
            return None

        if verification not in _VERIFICATION_RANK:
            verification = "reported"
        step_tuple = tuple(steps)
        goals = tuple(goal_conditions)
        skill_id = _hash(
            {
                "domain": context.signature.domain,
                "goals": sorted(context.signature.goals),
                "steps": [s.label for s in step_tuple],
            }
        )
        existing = self.skills.get(skill_id)
        if existing is not None:
            existing.uses += 1
            existing.successes += 1
            existing.verification_score = max(
                existing.verification_score, schema.last_score
            )
            if verification_rank(verification) > verification_rank(
                existing.verification
            ):
                existing.verification = verification
            existing.updated_at = time.time()
            skill = existing
        else:
            skill = LearnedSkill(
                skill_id=skill_id,
                signature=context.signature,
                steps=step_tuple,
                success_conditions=goals,
                verification_score=schema.last_score,
                verification=verification,
            )
            self.skills[skill_id] = skill
        if skill_id not in schema.skill_ids:
            schema.skill_ids.append(skill_id)
        self._ground_skill(context, skill)
        return skill

    def _ground_skill(
        self, context: TaskContext, skill: LearnedSkill
    ) -> None:
        """Write a consolidated skill into the concept network.

        A skill is something the system can *do* — grounding it lets
        the semantic machinery (language, analogy, self-model) reason
        about competence rather than skills living only in this
        store. Runs on every consolidation so re-verification upgrades
        the concept's provenance property.
        """
        if self.network is None:
            return
        try:
            from ..concepts import RelationType

            name = f"skill:{skill.skill_id}"
            self.network.add_concept(
                name,
                origin="perception",
                properties={
                    "kind": "skill",
                    "domain": context.signature.domain,
                    "verification": skill.verification,
                    "steps": str(len(skill.steps)),
                    "successes": str(skill.successes),
                    "failures": str(skill.failures),
                },
            )
            domain = f"domain:{context.signature.domain}"
            self.network.add_edge(
                name, domain, RelationType.PART_OF, origin="perception"
            )
            for goal in sorted(context.signature.goals):
                # "The procedure aims to achieve this condition" —
                # teleological, not causal.
                self.network.add_edge(
                    name,
                    f"goal:{goal}",
                    RelationType.GOAL_DIRECTED,
                    origin="perception",
                )
        except Exception as e:  # noqa: BLE001
            # Grounding is best-effort; never break consolidation.
            logger.debug("skill grounding failed: %s", e)

    def mark_skill_failure(self, skill_id: str) -> None:
        skill = self.skills.get(skill_id)
        if skill is None:
            return
        skill.failures += 1
        skill.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemas": [s.to_dict() for s in self.schemas.values()],
            "skills": [s.to_dict() for s in self.skills.values()],
            "transition_count": self.transition_count,
            "match_threshold": self.match_threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskCompetence:
        competence = cls(
            match_threshold=float(data.get("match_threshold", _MATCH_THRESHOLD))
        )
        for schema_data in data.get("schemas", []):
            schema = TaskSchema.from_dict(schema_data)
            competence.schemas[schema.schema_id] = schema
        for skill_data in data.get("skills", []):
            skill = LearnedSkill.from_dict(skill_data)
            competence.skills[skill.skill_id] = skill
        competence.transition_count = int(data.get("transition_count", 0))
        return competence

    @property
    def skill_count(self) -> int:
        return len(self.skills)

    @property
    def schema_count(self) -> int:
        return len(self.schemas)

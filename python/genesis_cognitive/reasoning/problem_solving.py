"""Problem-solving engine — means-ends analysis over the concept network.

Genesis could reason about concepts (deductive, abductive, analogical,
causal) but it couldn't *solve problems*. The reasoning engine is
stimulus-driven — give it a concept and it reasons about that concept.
There was no goal-directed reasoning: given a desired state, work
backward to find what's needed to achieve it.

This module closes that gap with a **means-ends analysis** engine.
Means-ends analysis is the classic problem-solving strategy (Newell &
Simon, 1972): compare the current state to the goal state, identify the
gap, select an operator that reduces the gap, apply it, and recurse on
any new subproblems. It's the cognitive process behind "to get to the
store, I need the car; to get the car, I need the keys; to get the
keys, I need to find my bag."

## How it works

The concept network IS the state space. Each concept is a state; each
typed edge is a transition. The "current state" is what Genesis already
understands (quality concepts with definitions and typed edges). The
"goal state" is the problem's target — a concept it wants to
understand, a state it wants to achieve, a contradiction it wants to
resolve.

1. **Problem representation**: a ``Problem`` has a goal (concept name),
   a goal type (understand / achieve / explain / resolve / compare),
   and a reason. Subproblems are recursive — a Problem can decompose
   into child Problems.

2. **Decomposition**: the solver finds prerequisites — concepts the
   goal DEPENDS_ON or is ENABLED_BY. Each unsatisfied prerequisite
   becomes a subproblem. For "achieve" goals, it also finds CAUSES and
   LEADS_TO edges (the means to the goal). For "resolve" goals, it
   finds the two contradicting concepts and weighs their support.

3. **Operator application**: for each problem, the solver applies
   operators — reasoning strategies from the ReasoningEngine (deductive
   chains, abductive inference, analogical transfer, causal chains,
   hypothesis formation) and concept-network queries (find
   prerequisites, find properties, find analogies). Each operator
   produces a ``SolutionStep`` with structured knowledge triples the
   language engine can compose from.

4. **Verification**: after applying operators, the solver checks whether
   the goal is now satisfied — does the concept have a definition and
   typed edges? Was a causal chain found? Was the contradiction
   resolved? Verification is goal-type-specific.

5. **Solution composition**: the final ``Solution`` aggregates all
   steps and subproblem solutions into a tree, with an overall
   confidence and verification status. The ``all_knowledge`` property
   flattens the tree into knowledge triples for the language engine.

## Cycle safety

The concept network can have cycles (A depends_on B, B depends_on A).
The solver tracks visited concepts per-solve and refuses to recurse
into an already-visited concept, returning a "blocked" solution instead.
A max-depth limit (default 4) prevents runaway recursion on large
networks.

## Integration

The solver is wired into the cognition engine as
``self.cognition.problem_solver`` and invoked during the reasoning
phase when standard topic reasoning produces no confident results —
i.e., when Genesis encounters something it can't immediately answer,
it tries to *solve* it rather than falling back to "unsure."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..concepts import ConceptNetwork, RelationType
from .engine import ReasoningStrategy

if TYPE_CHECKING:
    from .engine import (
        CounterfactualReasoning,
        MetaReasoning,
        ProbabilisticReasoning,
        ReasoningEngine,
        TemporalReasoning,
    )

__all__ = [
    "GoalType",
    "Problem",
    "ProblemSolver",
    "ProblemStatus",
    "Solution",
    "SolutionStep",
]

logger = logging.getLogger(__name__)

# Maximum recursion depth for subproblem decomposition.
_MAX_DEPTH = 4

# Maximum subproblems per decomposition level (prevents explosion).
_MAX_SUBPROBLEMS = 5

# Minimum confidence for a reasoning result to count as evidence.
_EVIDENCE_THRESHOLD = 0.3

# Confidence for a solution that was already satisfied (no work needed).
_ALREADY_KNOWN_CONFIDENCE = 0.9

# Confidence for a blocked solution (circular dependency or max depth).
_BLOCKED_CONFIDENCE = 0.1


class GoalType(Enum):
    """What kind of problem is being solved.

    The goal type determines how the problem is decomposed, which
    operators are applied, and how the solution is verified.
    """

    UNDERSTAND = "understand"  # learn about a concept (definition, relations)
    ACHIEVE = "achieve"  # find a causal path from a controllable action to a goal
    EXPLAIN = "explain"  # find causes / explanations for an observed state
    RESOLVE = "resolve"  # resolve a contradiction between two concepts
    COMPARE = "compare"  # find the relationship between two concepts


class ProblemStatus(Enum):
    """Lifecycle state of a problem."""

    OPEN = "open"  # not yet attempted
    IN_PROGRESS = "in_progress"  # currently being solved
    SOLVED = "solved"  # goal achieved and verified
    BLOCKED = "blocked"  # cannot be solved (circular, max depth, no operators)


@dataclass(slots=True)
class Problem:
    """A problem — a gap between what Genesis knows and what it wants to know.

    The goal is a concept name (or a pair for compare/resolve). The
    goal_type determines the decomposition and verification strategy.
    Subproblems are generated recursively during decomposition.
    """

    goal: str  # the concept to understand/achieve/explain
    goal_type: GoalType = GoalType.UNDERSTAND
    reason: str = ""  # why this problem exists (for introspection)
    secondary_goal: str = ""  # second concept (for compare/resolve)
    parent_goal: str = ""  # parent problem's goal (for tree tracking)
    depth: int = 0  # recursion depth
    status: ProblemStatus = ProblemStatus.OPEN
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def key(self) -> str:
        """Unique key for deduplication (goal + type + secondary)."""
        return f"{self.goal}:{self.goal_type.value}:{self.secondary_goal}"


@dataclass(slots=True)
class SolutionStep:
    """A single step in a solution — one operator application.

    The ``knowledge`` field carries structured (relation, target,
    weight) triples that the language engine can compose from, so
    Genesis expresses the solution in its own words rather than
    reciting the ``description`` string.
    """

    description: str  # what was done (semantic seed, not the final words)
    operator: str  # which operator was used (reason, lookup, etc.)
    result: str  # what was learned/achieved (semantic seed)
    evidence: list[str]  # supporting evidence chain
    knowledge: list[tuple[str, str, float]] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(slots=True)
class Solution:
    """The result of solving a problem — a tree of steps and sub-solutions.

    The solution is verified if the goal was achieved after applying
    all steps and solving all subproblems. The ``all_knowledge``
    property flattens the entire solution tree into knowledge triples
    for the language engine.
    """

    problem: Problem
    steps: list[SolutionStep] = field(default_factory=list)
    sub_solutions: list[Solution] = field(default_factory=list)
    verified: bool = False
    confidence: float = 0.0
    blocked_reason: str = ""

    @property
    def all_steps(self) -> list[SolutionStep]:
        """All steps in this solution and all sub-solutions (depth-first)."""
        result = list(self.steps)
        for sub in self.sub_solutions:
            result.extend(sub.all_steps)
        return result

    @property
    def all_knowledge(self) -> list[tuple[str, str, float]]:
        """All knowledge triples from this solution and all sub-solutions."""
        result: list[tuple[str, str, float]] = []
        for step in self.steps:
            result.extend(step.knowledge)
        for sub in self.sub_solutions:
            result.extend(sub.all_knowledge)
        return result

    @property
    def is_blocked(self) -> bool:
        """Whether this solution was blocked (cannot be solved)."""
        return bool(self.blocked_reason)

    def describe(self) -> str:
        """Human-readable summary of the solution (for introspection)."""
        status = "verified" if self.verified else "blocked" if self.is_blocked else "unverified"
        lines = [
            f"Problem: {self.problem.goal} ({self.problem.goal_type.value}) — {status}",
            f"Confidence: {self.confidence:.2f}",
        ]
        if self.blocked_reason:
            lines.append(f"Blocked: {self.blocked_reason}")
        for step in self.steps:
            lines.append(f"  → {step.description}")
        for sub in self.sub_solutions:
            for line in sub.describe().split("\n"):
                lines.append(f"  {line}")
        return "\n".join(lines)


class ProblemSolver:
    """Means-ends analysis engine over the concept network.

    Given a goal (concept + goal type), the solver decomposes it into
    subproblems by finding prerequisites (DEPENDS_ON, ENABLES edges),
    applies reasoning operators to fill knowledge gaps, and verifies
    the solution against the goal. Subproblems are solved recursively,
    producing a solution tree.

    Usage::

        solver = ProblemSolver(network, reasoning)
        solution = solver.solve("photosynthesis", GoalType.UNDERSTAND)
        if solution.verified:
            for step in solution.all_steps:
                print(step.description)

    The solver is stateless across calls (it reads the network and
    reasoning engine; it doesn't modify them). Solution history is
    retained for introspection.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        reasoning: ReasoningEngine | None = None,
        meta_reasoning: MetaReasoning | None = None,
        probabilistic: ProbabilisticReasoning | None = None,
        temporal: TemporalReasoning | None = None,
        counterfactual: CounterfactualReasoning | None = None,
    ) -> None:
        """Initialize with a concept network and optional reasoning engines.

        Args:
            network: The concept network — the state space.
            reasoning: The reasoning engine — provides operators
                (deductive, abductive, analogical, causal, hypothesis).
                If None, the solver uses concept-network queries only.
            meta_reasoning: Meta-reasoning engine — selects which
                reasoning strategy to emphasize per step based on
                problem type and learned effectiveness. If None,
                all applicable operators are applied uniformly.
            probabilistic: Probabilistic reasoning engine — provides
                the ``probabilistic_infer`` operator for uncertain
                evidence. If None, probabilistic operator is skipped.
            temporal: Temporal reasoning engine — provides the
                ``temporal_order`` operator for before/after queries.
                If None, temporal operator is skipped.
            counterfactual: Counterfactual reasoning engine — provides
                the ``counterfactual`` operator for what-if analysis.
                If None, counterfactual operator is skipped.
        """
        self.network = network
        self.reasoning = reasoning
        self.meta_reasoning = meta_reasoning
        self.probabilistic = probabilistic
        self.temporal = temporal
        self.counterfactual = counterfactual
        self._solutions: list[Solution] = []
        self._solved_count = 0
        self._blocked_count = 0

    @property
    def solutions(self) -> list[Solution]:
        """All solutions ever produced (copy)."""
        return list(self._solutions)

    @property
    def solved_count(self) -> int:
        """Total problems solved (verified)."""
        return self._solved_count

    @property
    def blocked_count(self) -> int:
        """Total problems blocked (unsolvable)."""
        return self._blocked_count

    # ─── Public API ───────────────────────────────────────────────

    def solve(
        self,
        goal: str,
        goal_type: GoalType = GoalType.UNDERSTAND,
        reason: str = "",
        secondary_goal: str = "",
    ) -> Solution:
        """Solve a problem and return the solution.

        This is the main entry point. Given a goal concept and goal
        type, the solver decomposes, applies operators, and verifies.

        Args:
            goal: The concept to understand/achieve/explain.
            goal_type: What kind of problem this is.
            reason: Why this problem exists (for introspection).
            secondary_goal: Second concept (for compare/resolve).

        Returns:
            The Solution — a tree of steps and sub-solutions.
        """
        problem = Problem(
            goal=goal,
            goal_type=goal_type,
            reason=reason,
            secondary_goal=secondary_goal,
        )
        solution = self._solve(problem, visited=set())
        self._solutions.append(solution)
        # Bound the history — solutions accumulate on every solve in a
        # long-running daemon.
        if len(self._solutions) > 500:
            del self._solutions[: len(self._solutions) - 500]
        if solution.verified:
            self._solved_count += 1
        elif solution.is_blocked:
            self._blocked_count += 1
        return solution

    # ─── Core means-ends analysis ─────────────────────────────────

    def _solve(
        self,
        problem: Problem,
        visited: set[str],
    ) -> Solution:
        """Recursively solve a problem via means-ends analysis.

        1. Cycle/depth check → blocked if circular or too deep.
        2. Goal check → already satisfied? Return trivial solution.
        3. Decompose → find unsatisfied prerequisites → subproblems.
        4. Solve subproblems recursively.
        5. Apply operators → reasoning + network queries → steps.
        6. Verify → is the goal now satisfied?
        """
        # 1. Cycle and depth guards.
        goal_key = problem.key
        if problem.depth >= _MAX_DEPTH:
            return Solution(
                problem=problem,
                verified=False,
                confidence=_BLOCKED_CONFIDENCE,
                blocked_reason=f"max depth ({_MAX_DEPTH}) reached",
            )
        if goal_key in visited:
            return Solution(
                problem=problem,
                verified=False,
                confidence=_BLOCKED_CONFIDENCE,
                blocked_reason="circular dependency detected",
            )
        visited = visited | {goal_key}

        # 2. Goal check — is the goal already satisfied?
        if self._is_satisfied(problem):
            step = SolutionStep(
                description=f"already understand {problem.goal}",
                operator="goal_check",
                result=f"{problem.goal} is already well-understood",
                evidence=[],
                confidence=_ALREADY_KNOWN_CONFIDENCE,
            )
            return Solution(
                problem=problem,
                steps=[step],
                verified=True,
                confidence=_ALREADY_KNOWN_CONFIDENCE,
            )

        # 3. Decompose into subproblems.
        subproblems = self._decompose(problem)

        # 4. Solve subproblems recursively.
        sub_solutions: list[Solution] = []
        for sub in subproblems[:_MAX_SUBPROBLEMS]:
            sub.parent_goal = problem.goal
            sub.depth = problem.depth + 1
            sub_sol = self._solve(sub, visited)
            sub_solutions.append(sub_sol)

        # 5. Apply operators to the problem itself.
        steps = self._apply_operators(problem, sub_solutions)

        # 6. Verify — is the goal now satisfied?
        verified = self._verify(problem, steps, sub_solutions)

        # Compute overall confidence from steps and sub-solutions.
        confidence = self._compute_confidence(steps, sub_solutions, verified)

        return Solution(
            problem=problem,
            steps=steps,
            sub_solutions=sub_solutions,
            verified=verified,
            confidence=confidence,
        )

    # ─── Goal satisfaction checking ──────────────────────────────

    def _is_satisfied(self, problem: Problem) -> bool:
        """Check if the goal is already achieved (before any work).

        Goal-type-specific:
        - UNDERSTAND: concept exists, has definition, has ≥2 typed edges.
        - ACHIEVE: a causal chain from some concept to the goal exists.
        - EXPLAIN: at least one cause of the goal is known.
        - RESOLVE: the two concepts don't both have strong support.
        - COMPARE: a path between the two concepts exists.
        """
        concept = self.network.get_concept(problem.goal)
        if concept is None:
            return False

        if problem.goal_type == GoalType.UNDERSTAND:
            return self._concept_is_well_understood(problem.goal)

        if problem.goal_type == GoalType.ACHIEVE:
            # A causal chain to the goal exists if something CAUSES or
            # LEADS_TO it.
            for edge in self.network.get_edges(problem.goal, "in"):
                if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                    return True
            return False

        if problem.goal_type == GoalType.EXPLAIN:
            # An explanation exists if the goal has incoming CAUSES edges.
            for edge in self.network.get_edges(problem.goal, "in"):
                if edge.relation == RelationType.CAUSES:
                    return True
            return False

        if problem.goal_type == GoalType.RESOLVE:
            # Resolved if one side has clearly more support than the other.
            support_a = self._support_strength(problem.goal)
            support_b = self._support_strength(problem.secondary_goal)
            if support_a == 0 and support_b == 0:
                return False
            return abs(support_a - support_b) > 0.3

        if problem.goal_type == GoalType.COMPARE:
            # Satisfied if a path exists between the two concepts.
            if not problem.secondary_goal:
                return False
            path = self.network.find_path(problem.goal, problem.secondary_goal, max_depth=3)
            return path is not None and len(path) > 0

        return False

    def _concept_is_well_understood(self, concept_id: str) -> bool:
        """Check if a concept is well-understood (has definition + typed edges)."""
        concept = self.network.get_concept(concept_id)
        if concept is None:
            return False
        has_def = bool(concept.properties.get("definition"))
        typed_edges = 0
        for edge in self.network.get_edges(concept_id, "both"):
            if edge.relation != RelationType.RELATED_TO:
                typed_edges += 1
        return has_def and typed_edges >= 2

    def _support_strength(self, concept_id: str) -> float:
        """Measure how well-supported a concept is (0..1).

        Combines confidence, edge count, and definition presence.
        Used by RESOLVE to weigh contradicting concepts.
        """
        concept = self.network.get_concept(concept_id)
        if concept is None:
            return 0.0
        score = concept.confidence * 0.4
        edges = len(self.network.get_edges(concept_id, "both"))
        score += min(0.3, edges * 0.05)
        if concept.properties.get("definition"):
            score += 0.3
        return min(1.0, score)

    # ─── Decomposition ───────────────────────────────────────────

    def _decompose(self, problem: Problem) -> list[Problem]:
        """Decompose a problem into subproblems.

        Finds unsatisfied prerequisites via concept-network edges:
        - DEPENDS_ON: the goal depends on understanding this concept.
        - ENABLES: this concept enables the goal (prerequisite).
        - CAUSES / LEADS_TO: for ACHIEVE goals, the means to the goal.
        - For RESOLVE: the two contradicting concepts become subproblems.
        - For COMPARE: both concepts become UNDERSTAND subproblems.
        """
        subproblems: list[Problem] = []

        if problem.goal_type == GoalType.RESOLVE:
            # To resolve a contradiction, understand both sides first.
            if not self._concept_is_well_understood(problem.goal):
                subproblems.append(Problem(
                    goal=problem.goal,
                    goal_type=GoalType.UNDERSTAND,
                    reason=f"understand both sides of contradiction with {problem.secondary_goal}",
                    secondary_goal="",
                ))
            if problem.secondary_goal and not self._concept_is_well_understood(
                problem.secondary_goal
            ):
                subproblems.append(Problem(
                    goal=problem.secondary_goal,
                    goal_type=GoalType.UNDERSTAND,
                    reason=f"understand both sides of contradiction with {problem.goal}",
                    secondary_goal="",
                ))
            return subproblems

        if problem.goal_type == GoalType.COMPARE:
            # To compare, understand both concepts first.
            if not self._concept_is_well_understood(problem.goal):
                subproblems.append(Problem(
                    goal=problem.goal,
                    goal_type=GoalType.UNDERSTAND,
                    reason=(
                        f"understand {problem.goal} for comparison "
                        f"with {problem.secondary_goal}"
                    ),
                ))
            if problem.secondary_goal and not self._concept_is_well_understood(
                problem.secondary_goal
            ):
                subproblems.append(Problem(
                    goal=problem.secondary_goal,
                    goal_type=GoalType.UNDERSTAND,
                    reason=(
                        f"understand {problem.secondary_goal} for "
                        f"comparison with {problem.goal}"
                    ),
                ))
            return subproblems

        # UNDERSTAND, ACHIEVE, EXPLAIN: find prerequisites.
        # DEPENDS_ON edges (incoming): the goal depends on these.
        for edge in self.network.get_edges(problem.goal, "in"):
            if edge.relation == RelationType.DEPENDS_ON:
                if not self._concept_is_well_understood(edge.source):
                    subproblems.append(Problem(
                        goal=edge.source,
                        goal_type=GoalType.UNDERSTAND,
                        reason=f"{problem.goal} depends on understanding {edge.source}",
                    ))

        # ENABLES edges (incoming): these concepts enable the goal.
        for edge in self.network.get_edges(problem.goal, "in"):
            if edge.relation == RelationType.ENABLES:
                if not self._concept_is_well_understood(edge.source):
                    subproblems.append(Problem(
                        goal=edge.source,
                        goal_type=GoalType.UNDERSTAND,
                        reason=f"{edge.source} enables {problem.goal}",
                    ))

        # For ACHIEVE goals: find causes (the means to the goal).
        if problem.goal_type == GoalType.ACHIEVE:
            for edge in self.network.get_edges(problem.goal, "in"):
                if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                    if not self._concept_is_well_understood(edge.source):
                        subproblems.append(Problem(
                            goal=edge.source,
                            goal_type=GoalType.UNDERSTAND,
                            reason=(
                                f"{edge.source} causes {problem.goal} — "
                                f"understanding it is the means"
                            ),
                        ))

        # For EXPLAIN goals: find causes (the explanations).
        if problem.goal_type == GoalType.EXPLAIN:
            for edge in self.network.get_edges(problem.goal, "in"):
                if edge.relation == RelationType.CAUSES:
                    subproblems.append(Problem(
                        goal=edge.source,
                        goal_type=GoalType.UNDERSTAND,
                        reason=f"{edge.source} is a possible cause of {problem.goal}",
                    ))

        # Deduplicate by goal.
        seen: set[str] = set()
        unique: list[Problem] = []
        for sub in subproblems:
            if sub.goal not in seen and sub.goal != problem.goal:
                seen.add(sub.goal)
                unique.append(sub)
        return unique

    # ─── Operator application ─────────────────────────────────────

    def _apply_operators(
        self,
        problem: Problem,
        sub_solutions: list[Solution],
    ) -> list[SolutionStep]:
        """Apply reasoning operators to the problem.

        Operators are the "actions" the solver can take to reduce the
        gap between current and goal state:
        - ``reason``: invoke the ReasoningEngine on the goal concept.
        - ``find_prerequisites``: find DEPENDS_ON / ENABLES edges.
        - ``find_properties``: find HAS_PROPERTY edges.
        - ``find_causes``: find CAUSES / LEADS_TO edges.
        - ``find_analogies``: find SIMILAR_TO edges (analogical transfer).
        - ``resolve_contradiction``: weigh support for both sides (RESOLVE).
        - ``find_relationship``: find a path between two concepts (COMPARE).
        - ``probabilistic_infer``: Bayesian belief updating (PROBABILISTIC).
        - ``temporal_order``: before/after queries (TEMPORAL).
        - ``counterfactual``: what-if analysis (COUNTERFACTUAL).

        When MetaReasoning is available, it selects which reasoning
        strategy to emphasize, and results are recorded for learning.
        Each operator produces a SolutionStep with structured knowledge.
        """
        steps: list[SolutionStep] = []

        # Use MetaReasoning to decide which strategies to emphasize.
        # The meta-reasoner classifies the problem and picks the best
        # strategy from the available ones. We still apply all
        # applicable operators, but the selected strategy's step gets
        # a confidence boost.
        selected_strategy: ReasoningStrategy | None = None
        if self.meta_reasoning is not None:
            try:
                available = list(ReasoningStrategy)
                selected_strategy = self.meta_reasoning.meta_reason(
                    problem.goal, available, confidence=0.5,
                )
            except Exception:  # noqa: BLE001
                selected_strategy = None

        # Operator 1: Reason about the concept (if reasoning engine available).
        if self.reasoning is not None:
            reason_step = self._op_reason(problem)
            if reason_step is not None:
                # Boost confidence if MetaReasoning selected deductive/analogical.
                if selected_strategy is not None and selected_strategy.value in (
                    "deductive", "analogical",
                ):
                    reason_step.confidence = min(1.0, reason_step.confidence + 0.1)
                steps.append(reason_step)
                self._record_meta(problem, "reason", reason_step.confidence, True)

        # Operator 2: Find prerequisites.
        prereq_step = self._op_find_prerequisites(problem)
        if prereq_step is not None:
            steps.append(prereq_step)

        # Operator 3: Find properties.
        prop_step = self._op_find_properties(problem)
        if prop_step is not None:
            steps.append(prop_step)

        # Operator 3.5: Find causal role (what it causes / what causes it).
        # For UNDERSTAND, knowing what a concept causes and is caused by
        # is core to understanding it — not just for ACHIEVE/EXPLAIN.
        cause_step = self._op_find_causes(problem)
        if cause_step is not None:
            if selected_strategy is not None and selected_strategy.value == "causal":
                cause_step.confidence = min(1.0, cause_step.confidence + 0.1)
            steps.append(cause_step)
            self._record_meta(problem, "causal", cause_step.confidence, True)

        # Operator 4: Find causes (for ACHIEVE and EXPLAIN).
        if problem.goal_type in (GoalType.ACHIEVE, GoalType.EXPLAIN):
            # Already added above for all types; skip duplicate.
            pass

        # Operator 5: Find analogies (for UNDERSTAND).
        if problem.goal_type == GoalType.UNDERSTAND:
            analogy_step = self._op_find_analogies(problem)
            if analogy_step is not None:
                if selected_strategy is not None and selected_strategy.value == "analogical":
                    analogy_step.confidence = min(1.0, analogy_step.confidence + 0.1)
                steps.append(analogy_step)
                self._record_meta(problem, "analogical", analogy_step.confidence, True)

        # Operator 6+: Conditional operators (goal-type-dependent).
        self._apply_conditional_operators(
            problem, steps, selected_strategy,
        )

        return steps

    def _apply_conditional_operators(
        self,
        problem: Problem,
        steps: list[SolutionStep],
        selected_strategy: ReasoningStrategy | None,
    ) -> None:
        """Apply goal-type-dependent operators 6-10."""
        # Operator 6: Resolve contradiction (for RESOLVE).
        if problem.goal_type == GoalType.RESOLVE:
            resolve_step = self._op_resolve_contradiction(problem)
            if resolve_step is not None:
                steps.append(resolve_step)

        # Operator 7: Find relationship (for COMPARE).
        if problem.goal_type == GoalType.COMPARE:
            compare_step = self._op_find_relationship(problem)
            if compare_step is not None:
                steps.append(compare_step)

        # Operator 8: Probabilistic inference (if available).
        if self.probabilistic is not None:
            prob_step = self._op_probabilistic(problem)
            if prob_step is not None:
                if selected_strategy is not None and selected_strategy.value == "probabilistic":
                    prob_step.confidence = min(1.0, prob_step.confidence + 0.1)
                steps.append(prob_step)
                self._record_meta(problem, "probabilistic", prob_step.confidence, True)

        # Operator 9: Temporal ordering (if available).
        if self.temporal is not None:
            temporal_step = self._op_temporal(problem)
            if temporal_step is not None:
                if selected_strategy is not None and selected_strategy.value == "temporal":
                    temporal_step.confidence = min(1.0, temporal_step.confidence + 0.1)
                steps.append(temporal_step)
                self._record_meta(problem, "temporal", temporal_step.confidence, True)

        # Operator 10: Counterfactual analysis (if available).
        if self.counterfactual is not None and problem.goal_type in (
            GoalType.UNDERSTAND, GoalType.EXPLAIN, GoalType.ACHIEVE,
        ):
            cf_step = self._op_counterfactual(problem)
            if cf_step is not None:
                if selected_strategy is not None and selected_strategy.value == "counterfactual":
                    cf_step.confidence = min(1.0, cf_step.confidence + 0.1)
                steps.append(cf_step)
                self._record_meta(problem, "counterfactual", cf_step.confidence, True)

    def _record_meta(
        self, problem: Problem, strategy: str, confidence: float, success: bool,
    ) -> None:
        """Record a reasoning attempt in MetaReasoning for learning."""
        if self.meta_reasoning is None:
            return
        try:
            strategy_enum = ReasoningStrategy(strategy)
            self.meta_reasoning.record_result(
                strategy_enum, problem.goal, confidence, success,
            )
        except (ValueError, Exception) as e:  # noqa: BLE001
            logger.debug(f"_record_meta: failed to record strategy {strategy!r}: {e}")

    def _op_reason(self, problem: Problem) -> SolutionStep | None:
        """Operator: invoke the ReasoningEngine on the goal concept."""
        if self.reasoning is None:
            return None
        results = self.reasoning.reason_about(problem.goal, depth=3)
        if not results:
            return None

        # Collect knowledge triples and evidence from reasoning results.
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for rr in results:
            if rr.confidence < _EVIDENCE_THRESHOLD:
                continue
            evidence.append(rr.conclusion)
            for rel, target, weight in rr.knowledge:
                knowledge.append((rel, target, weight))

        # Also extract relations from the conclusion text (simplified).
        for rr in results[:5]:
            if rr.confidence >= _EVIDENCE_THRESHOLD:
                evidence.append(f"[{rr.reasoning_type.value}] {rr.conclusion}")

        confidence = max((r.confidence for r in results), default=0.3)
        return SolutionStep(
            description=f"reason about {problem.goal}",
            operator="reason",
            result=f"reasoning produced {len(results)} conclusions",
            evidence=evidence[:10],
            knowledge=knowledge,
            confidence=confidence,
        )

    def _op_find_prerequisites(self, problem: Problem) -> SolutionStep | None:
        """Operator: find DEPENDS_ON and ENABLES edges (prerequisites)."""
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for edge in self.network.get_edges(problem.goal, "in"):
            if edge.relation == RelationType.DEPENDS_ON:
                knowledge.append(("depends_on", edge.source, edge.weight))
                evidence.append(f"{problem.goal} depends on {edge.source}")
            elif edge.relation == RelationType.ENABLES:
                knowledge.append(("enabled_by", edge.source, edge.weight))
                evidence.append(f"{problem.goal} is enabled by {edge.source}")
        if not knowledge:
            return None
        return SolutionStep(
            description=f"find prerequisites for {problem.goal}",
            operator="find_prerequisites",
            result=f"found {len(knowledge)} prerequisites",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.7,
        )

    def _op_find_properties(self, problem: Problem) -> SolutionStep | None:
        """Operator: find HAS_PROPERTY edges."""
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for edge in self.network.get_edges(problem.goal, "out"):
            if edge.relation == RelationType.HAS_PROPERTY:
                knowledge.append(("has_property", edge.target, edge.weight))
                evidence.append(f"{problem.goal} has property {edge.target}")
        if not knowledge:
            return None
        return SolutionStep(
            description=f"find properties of {problem.goal}",
            operator="find_properties",
            result=f"found {len(knowledge)} properties",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.6,
        )

    def _op_find_causes(self, problem: Problem) -> SolutionStep | None:
        """Operator: find CAUSES and LEADS_TO edges (for ACHIEVE/EXPLAIN)."""
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        # Incoming causes (what causes the goal).
        for edge in self.network.get_edges(problem.goal, "in"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                knowledge.append(("caused_by", edge.source, edge.weight))
                evidence.append(f"{edge.source} {edge.relation.value} {problem.goal}")
        # Outgoing causes (what the goal causes).
        for edge in self.network.get_edges(problem.goal, "out"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                knowledge.append(("causes", edge.target, edge.weight))
                evidence.append(f"{problem.goal} {edge.relation.value} {edge.target}")
        if not knowledge:
            return None
        return SolutionStep(
            description=f"find causes for {problem.goal}",
            operator="find_causes",
            result=f"found {len(knowledge)} causal links",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.7,
        )

    def _op_find_analogies(self, problem: Problem) -> SolutionStep | None:
        """Operator: find SIMILAR_TO edges (analogical transfer)."""
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for edge in self.network.get_edges(problem.goal, "both"):
            if edge.relation == RelationType.SIMILAR_TO:
                other = edge.target if edge.source == problem.goal else edge.source
                knowledge.append(("similar_to", other, edge.weight))
                evidence.append(f"{problem.goal} is similar to {other}")
        if not knowledge:
            return None
        return SolutionStep(
            description=f"find analogies for {problem.goal}",
            operator="find_analogies",
            result=f"found {len(knowledge)} similar concepts",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.5,
        )

    def _op_resolve_contradiction(self, problem: Problem) -> SolutionStep | None:
        """Operator: weigh support for both sides of a contradiction (RESOLVE)."""
        support_a = self._support_strength(problem.goal)
        support_b = self._support_strength(problem.secondary_goal)

        # Find contradiction edges.
        evidence: list[str] = []
        for edge in self.network.get_edges(problem.goal, "both"):
            other = edge.target if edge.source == problem.goal else edge.source
            if other == problem.secondary_goal and edge.relation in (
                RelationType.CONTRADICTS,
                RelationType.OPPOSITE_OF,
            ):
                evidence.append(
                    f"{edge.source} {edge.relation.value} {edge.target}"
                )

        knowledge: list[tuple[str, str, float]] = []
        if support_a > support_b:
            knowledge.append(
                ("stronger_than", problem.secondary_goal, support_a - support_b)
            )
            result = (
                f"{problem.goal} is better supported "
                f"({support_a:.2f} vs {support_b:.2f})"
            )
        elif support_b > support_a:
            knowledge.append(
                ("weaker_than", problem.secondary_goal, support_b - support_a)
            )
            result = (
                f"{problem.secondary_goal} is better supported "
                f"({support_b:.2f} vs {support_a:.2f})"
            )
        else:
            result = f"both sides equally supported ({support_a:.2f})"

        return SolutionStep(
            description=(
                f"resolve contradiction between {problem.goal} "
                f"and {problem.secondary_goal}"
            ),
            operator="resolve_contradiction",
            result=result,
            evidence=evidence,
            knowledge=knowledge,
            confidence=max(support_a, support_b),
        )

    def _op_find_relationship(self, problem: Problem) -> SolutionStep | None:
        """Operator: find a path between two concepts (COMPARE)."""
        if not problem.secondary_goal:
            return None
        path = self.network.find_path(problem.goal, problem.secondary_goal, max_depth=4)
        if path is None or len(path) == 0:
            return None

        evidence: list[str] = []
        knowledge: list[tuple[str, str, float]] = []
        for edge in path:
            evidence.append(f"{edge.source} {edge.relation.value} {edge.target}")
            knowledge.append((edge.relation.value, edge.target, edge.weight))

        return SolutionStep(
            description=f"find relationship between {problem.goal} and {problem.secondary_goal}",
            operator="find_relationship",
            result=f"path found with {len(path)} hops",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.6,
        )

    def _op_probabilistic(self, problem: Problem) -> SolutionStep | None:
        """Operator: probabilistic inference over uncertain edges.

        For each typed edge involving the goal concept, query the
        probabilistic reasoning engine for the belief distribution.
        This produces confidence-weighted knowledge that reflects
        uncertainty rather than treating all edges as equally certain.
        """
        if self.probabilistic is None:
            return None
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for edge in self.network.get_edges(problem.goal, "out"):
            if edge.relation == RelationType.RELATED_TO:
                continue
            try:
                belief = self.probabilistic.get_belief(
                    problem.goal, edge.relation.value, edge.target,
                )
                knowledge.append(
                    (edge.relation.value, edge.target, belief.mean)
                )
                evidence.append(
                    f"{problem.goal} {edge.relation.value} {edge.target} "
                    f"(p={belief.mean:.2f}, conf={belief.confidence:.2f})"
                )
            except Exception:  # noqa: BLE001
                continue
        if not knowledge:
            return None
        return SolutionStep(
            description=f"probabilistic inference for {problem.goal}",
            operator="probabilistic_infer",
            result=f"assessed {len(knowledge)} uncertain relationships",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.55,
        )

    def _op_temporal(self, problem: Problem) -> SolutionStep | None:
        """Operator: temporal ordering of the goal's causal chain.

        Uses the temporal reasoning engine to order the goal's causes
        and effects into a before/after sequence. This adds temporal
        structure to the solution — not just *what* causes what, but
        *when* things happen relative to each other.
        """
        if self.temporal is None:
            return None
        # Gather the causal chain: causes of the goal (before) and
        # effects of the goal (after).
        before: list[str] = []
        for edge in self.network.get_edges(problem.goal, "in"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                before.append(edge.source)
        after: list[str] = []
        for edge in self.network.get_edges(problem.goal, "out"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                after.append(edge.target)
        if not before and not after:
            return None
        knowledge: list[tuple[str, str, float]] = []
        evidence: list[str] = []
        for src in before:
            knowledge.append(("before", src, 0.6))
            evidence.append(f"{src} happens before {problem.goal}")
        for tgt in after:
            knowledge.append(("after", tgt, 0.6))
            evidence.append(f"{tgt} happens after {problem.goal}")
        return SolutionStep(
            description=f"temporal ordering for {problem.goal}",
            operator="temporal_order",
            result=f"ordered {len(before)} causes and {len(after)} effects",
            evidence=evidence,
            knowledge=knowledge,
            confidence=0.5,
        )

    def _op_counterfactual(self, problem: Problem) -> SolutionStep | None:
        """Operator: counterfactual analysis — what if the goal were absent?

        Uses the counterfactual reasoning engine to trace what would
        happen if the goal concept were removed or replaced. This
        reveals the goal's causal role — what depends on it — which is
        core to understanding its significance.
        """
        if self.counterfactual is None:
            return None
        # Find the most significant effect of the goal to use as the
        # "fact" for the counterfactual. The intervention is "what if
        # this effect didn't happen?" — i.e., what else would change?
        effects: list[tuple[str, float]] = []
        for edge in self.network.get_edges(problem.goal, "out"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO, RelationType.ENABLES):
                effects.append((edge.target, edge.weight))
        if not effects:
            return None
        # Pick the highest-weighted effect.
        effects.sort(key=lambda x: -x[1])
        fact = effects[0][0]
        try:
            result = self.counterfactual.counterfactual(fact, problem.goal, depth=2)
        except Exception:  # noqa: BLE001
            return None
        knowledge: list[tuple[str, str, float]] = []
        for rel, target, weight in result.knowledge:
            knowledge.append((rel, target, weight))
        return SolutionStep(
            description=f"counterfactual: what if {problem.goal} were absent?",
            operator="counterfactual",
            result=f"traced effects of removing {problem.goal}",
            evidence=result.evidence[:5],
            knowledge=knowledge,
            confidence=result.confidence,
        )

    # ─── Verification ─────────────────────────────────────────────

    def _verify(
        self,
        problem: Problem,
        steps: list[SolutionStep],
        sub_solutions: list[Solution],
    ) -> bool:
        """Verify whether the goal is achieved after applying all steps.

        This re-checks goal satisfaction. If the operators produced
        enough evidence (steps with knowledge), the goal is considered
        achieved even if the concept wasn't modified (the solver is
        read-only — it doesn't add concepts; it discovers knowledge
        about them).
        """
        # For UNDERSTAND: check if we gathered enough knowledge.
        if problem.goal_type == GoalType.UNDERSTAND:
            total_knowledge = sum(len(s.knowledge) for s in steps)
            for sub in sub_solutions:
                total_knowledge += len(sub.all_knowledge)
            # If we found ≥3 knowledge items, we understand it well enough.
            return total_knowledge >= 3 or self._concept_is_well_understood(problem.goal)

        # For ACHIEVE: check if we found a causal chain.
        if problem.goal_type == GoalType.ACHIEVE:
            for step in steps:
                for rel, _target, _w in step.knowledge:
                    if rel in ("caused_by", "causes"):
                        return True
            return self._is_satisfied(problem)

        # For EXPLAIN: check if we found at least one cause.
        if problem.goal_type == GoalType.EXPLAIN:
            for step in steps:
                for rel, _target, _w in step.knowledge:
                    if rel == "caused_by":
                        return True
            return self._is_satisfied(problem)

        # For RESOLVE: check if we determined which side is stronger.
        if problem.goal_type == GoalType.RESOLVE:
            for step in steps:
                for rel, _target, _w in step.knowledge:
                    if rel in ("stronger_than", "weaker_than"):
                        return True
            return self._is_satisfied(problem)

        # For COMPARE: check if we found a path.
        if problem.goal_type == GoalType.COMPARE:
            for step in steps:
                if step.operator == "find_relationship" and step.knowledge:
                    return True
            return self._is_satisfied(problem)

        return False

    def _compute_confidence(
        self,
        steps: list[SolutionStep],
        sub_solutions: list[Solution],
        verified: bool,
    ) -> float:
        """Compute overall solution confidence from steps and sub-solutions."""
        if not steps and not sub_solutions:
            return _BLOCKED_CONFIDENCE

        # Average step confidence.
        step_conf = 0.0
        if steps:
            step_conf = sum(s.confidence for s in steps) / len(steps)

        # Sub-solution confidence (average, weighted by verification).
        sub_conf = 0.0
        if sub_solutions:
            verified_subs = [s for s in sub_solutions if s.verified]
            if verified_subs:
                sub_conf = sum(s.confidence for s in verified_subs) / len(verified_subs)

        # Combine: steps + sub-solutions, boosted by verification.
        base = (step_conf * 0.6 + sub_conf * 0.4) if sub_solutions else step_conf
        if verified:
            base = min(1.0, base + 0.15)
        return base

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "solved_count": self._solved_count,
            "blocked_count": self._blocked_count,
            "recent_solutions": [
                {
                    "goal": s.problem.goal,
                    "goal_type": s.problem.goal_type.value,
                    "verified": s.verified,
                    "confidence": s.confidence,
                    "step_count": len(s.all_steps),
                    "blocked_reason": s.blocked_reason,
                }
                for s in self._solutions[-20:]
            ],
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        solved = data.get("solved_count", 0)
        blocked = data.get("blocked_count", 0)
        self._solved_count = int(solved) if isinstance(solved, (int, float)) else 0
        self._blocked_count = int(blocked) if isinstance(blocked, (int, float)) else 0

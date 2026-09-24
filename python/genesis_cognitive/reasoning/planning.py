"""Planning — generating, executing, and revising multi-step plans
to achieve goals over multiple turns.

Genesis had scaffolding for planning but no genuine planning
process. The ``ExecutiveFunction.plan()`` was a greedy keyword-
overlap heuristic whose output only nudged confidence by ±0.1. The
``ProblemSolver`` planned *reasoning steps* (which knowledge-gathering
operators to apply), not *actions* (what to do). Goals persisted
across turns but had no plan structure — they were tried reactively
each turn, not pursued via a multi-step plan. No plan persistence,
no plan execution, no plan evaluation, no plan revision.

This module closes that gap with a **planning engine** that:

1. **Decomposes a goal into ordered steps** — uses the concept
   network's typed edges (DEPENDS_ON, ENABLES, CAUSES, LEADS_TO) to
   identify prerequisites and causal chains, producing an ordered
   sequence of steps that, if executed, achieve the goal.

2. **Evaluates plans for feasibility** — each step is checked against
   the concept network: does the prerequisite concept exist? Is
   there a known path to it? Steps that can't be accomplished are
   flagged, and the plan's overall feasibility score reflects this.

3. **Persists plans across turns** — plans are stored in the
   planning engine and survive across turns. Each turn, the engine
   advances the plan by one step.

4. **Executes one step per turn** — the current step is returned to
   the cognition engine, which uses the ProblemSolver and
   DecisionEngine to accomplish it.

5. **Tracks progress** — each step has a status (pending, in_progress,
   completed, failed). The engine tracks which step is current and
   how many remain.

6. **Revises on failure** — when a step fails, the engine can
   replan: find an alternative path, insert new prerequisites, or
   mark the goal as blocked if no path exists.

## Integration

The engine is wired into the cognition engine as
``self.planning_engine`` and called from ``_pursue_unresolved_goals``.
Instead of reactively trying to solve each goal in one shot, the
engine creates a plan for each goal and executes it step by step
across turns. This replaces the reactive goal pursuit with planned
goal achievement.

The engine also uses ``CounterfactualReasoning`` for plan
evaluation — simulating "what if this step were skipped?" to assess
each step's necessity — and the ``ProblemSolver`` for step
execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

__all__ = ["Plan", "PlanStatus", "PlanStep", "PlanStepStatus", "PlanningEngine"]

logger = logging.getLogger(__name__)


class PlanStepStatus(Enum):
    """Lifecycle state of a plan step."""

    PENDING = "pending"          # not yet started
    IN_PROGRESS = "in_progress"  # currently being executed
    COMPLETED = "completed"      # successfully done
    FAILED = "failed"            # attempted but failed
    SKIPPED = "skipped"          # bypassed during replanning


class PlanStatus(Enum):
    """Lifecycle state of an overall plan."""

    DRAFT = "draft"          # created but not yet executing
    ACTIVE = "active"        # currently executing
    COMPLETED = "completed"  # all steps done, goal achieved
    BLOCKED = "blocked"      # no path forward
    ABANDONED = "abandoned"  # given up


@dataclass(slots=True)
class PlanStep:
    """A single step in a plan.

    Each step represents one subgoal that must be accomplished to
    advance the plan. Steps are ordered by dependency: prerequisites
    come before the things that depend on them.
    """

    description: str           # what to do (e.g., "understand oxygen")
    target_concept: str        # the concept this step works on
    step_type: str             # understand, achieve, explain, resolve, verify
    prerequisites: list[str] = field(default_factory=list)  # concepts needed first
    status: PlanStepStatus = PlanStepStatus.PENDING
    attempt_count: int = 0     # how many times this step has been tried
    max_attempts: int = 3      # before marking as failed
    result_summary: str = ""   # what happened when this step was executed
    confidence: float = 0.0    # confidence in the step's outcome


@dataclass(slots=True)
class Plan:
    """A multi-step plan to achieve a goal.

    A plan is an ordered sequence of steps. Each turn, the planning
    engine advances the plan by one step. The plan persists across
    turns until completed, blocked, or abandoned.
    """

    goal: str                    # the goal this plan achieves
    goal_type: str               # understand, achieve, explain, etc.
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    current_step_index: int = 0  # which step is being executed
    created_at: int = field(default_factory=lambda: 0)
    updated_at: int = field(default_factory=lambda: 0)
    feasibility_score: float = 0.5  # 0..1, how feasible the plan is
    revision_count: int = 0       # how many times the plan has been revised
    failure_reason: str = ""      # why the plan failed (if it did)

    @property
    def current_step(self) -> PlanStep | None:
        """The step currently being executed, if any."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Whether all steps are completed."""
        return all(
            s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
            for s in self.steps
        )

    @property
    def progress(self) -> float:
        """Fraction of steps completed (0..1)."""
        if not self.steps:
            return 0.0
        done = sum(
            1 for s in self.steps
            if s.status in (PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED)
        )
        return done / len(self.steps)

    @property
    def remaining_steps(self) -> int:
        """How many steps remain to be executed."""
        return sum(
            1 for s in self.steps
            if s.status not in (
                PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED,
            )
        )

    def describe(self) -> str:
        """Human-readable plan summary."""
        lines = [
            f"Plan for: {self.goal} ({self.status.value})",
            f"  feasibility: {self.feasibility_score:.2f}, "
            f"progress: {self.progress:.0%}, "
            f"step {self.current_step_index + 1}/{len(self.steps)}",
        ]
        for i, step in enumerate(self.steps):
            marker = "→" if i == self.current_step_index else " "
            status = step.status.value
            lines.append(
                f"  {marker} {i + 1}. [{status}] {step.description}"
            )
        if self.failure_reason:
            lines.append(f"  failure: {self.failure_reason}")
        return "\n".join(lines)


class PlanningEngine:
    """Generates, executes, and revises multi-step plans for goals.

    Given a goal, the engine:

    1. Decomposes it into ordered steps using the concept network's
       typed edges (prerequisites via DEPENDS_ON/ENABLES, causal
       chains via CAUSES/LEADS_TO).
    2. Evaluates the plan's feasibility by checking whether each
       step's target concept exists or has a discoverable path.
    3. Persists the plan across turns.
    4. Advances one step per turn, returning the current step to
       the cognition engine for execution.
    5. Tracks progress and marks steps as completed or failed.
    6. Revises the plan on failure — finding alternative paths or
       marking the goal as blocked.

    Usage::

        planner = PlanningEngine(network)
        plan = planner.create_plan("understand fire", "understand")
        if plan.feasibility_score > 0.3:
            step = planner.advance(plan)
            if step:
                # Execute the step using ProblemSolver, DecisionEngine, etc.
                ...
                planner.mark_step(plan, step, success=True)
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the planning engine.

        Args:
            network: The concept network — used for goal decomposition
                and feasibility evaluation.
        """
        self.network = network
        self._plans: list[Plan] = []
        self._plans_by_goal: dict[str, Plan] = {}
        self._completed_count: int = 0
        self._blocked_count: int = 0
        self._revised_count: int = 0
        self._steps_executed: int = 0

    @property
    def active_plans(self) -> list[Plan]:
        """All currently active plans."""
        return [p for p in self._plans if p.status == PlanStatus.ACTIVE]

    @property
    def completed_count(self) -> int:
        """Number of plans completed."""
        return self._completed_count

    @property
    def blocked_count(self) -> int:
        """Number of plans blocked."""
        return self._blocked_count

    @property
    def revised_count(self) -> int:
        """Number of plan revisions."""
        return self._revised_count

    @property
    def steps_executed(self) -> int:
        """Total steps executed across all plans."""
        return self._steps_executed

    # ─── Plan creation ────────────────────────────────────────────

    def create_plan(
        self,
        goal: str,
        goal_type: str = "understand",
        max_depth: int = 4,
    ) -> Plan:
        """Create a multi-step plan to achieve a goal.

        Decomposes the goal into ordered steps using the concept
        network's typed edges. Prerequisites (DEPENDS_ON, ENABLES)
        come before the goal concept. Causal chains (CAUSES,
        LEADS_TO) provide steps for ACHIEVE goals.

        Args:
            goal: The goal to plan for (typically a concept name).
            goal_type: The type of goal (understand, achieve, explain,
                resolve, compare).
            max_depth: Maximum decomposition depth for prerequisites.

        Returns:
            A Plan with ordered steps.
        """
        import time

        plan = Plan(
            goal=goal,
            goal_type=goal_type,
            created_at=int(time.time() * 1000),
            updated_at=int(time.time() * 1000),
        )

        # Decompose the goal into steps.
        steps = self._decompose(goal, goal_type, max_depth)
        plan.steps = steps

        # Evaluate feasibility.
        plan.feasibility_score = self._evaluate_feasibility(plan)

        # Set initial status.
        if plan.feasibility_score > 0.3 and steps:
            plan.status = PlanStatus.ACTIVE
        elif not steps:
            plan.status = PlanStatus.COMPLETED
        else:
            plan.status = PlanStatus.BLOCKED
            plan.failure_reason = "low feasibility score"

        # Store the plan. Bound the history — plans accumulate over a
        # long-running daemon's lifetime; completed/blocked plans past
        # the cap are still indexed by goal in _plans_by_goal only if
        # they're the latest for that goal.
        self._plans.append(plan)
        self._plans_by_goal[goal] = plan
        if len(self._plans) > 200:
            removed = self._plans[: len(self._plans) - 200]
            del self._plans[: len(self._plans) - 200]
            for old in removed:
                if self._plans_by_goal.get(old.goal) is old:
                    del self._plans_by_goal[old.goal]

        return plan

    def _decompose(
        self,
        goal: str,
        goal_type: str,
        max_depth: int,
    ) -> list[PlanStep]:
        """Decompose a goal into ordered plan steps.

        For UNDERSTAND goals: gather prerequisites (DEPENDS_ON,
        ENABLES) as steps before the goal, then the goal itself.

        For ACHIEVE/EXPLAIN goals: find causal chains (CAUSES,
        LEADS_TO) and order them as steps leading to the goal.

        For RESOLVE goals: both sides of the contradiction become
        UNDERSTAND substeps, then a resolve step.

        For COMPARE goals: both concepts become UNDERSTAND substeps,
        then a compare step.
        """
        steps: list[PlanStep] = []
        visited: set[str] = set()

        if goal_type == "understand":
            self._decompose_understand(
                goal, steps, visited, max_depth, depth=0,
            )
        elif goal_type in ("achieve", "explain"):
            self._decompose_causal(
                goal, steps, visited, max_depth, depth=0,
            )
        elif goal_type == "resolve":
            # For resolve, we need two concepts — encoded as "A vs B".
            parts = goal.split(" vs ")
            if len(parts) == 2:
                self._decompose_understand(
                    parts[0], steps, visited, max_depth, depth=0,
                )
                self._decompose_understand(
                    parts[1], steps, visited, max_depth, depth=0,
                )
                steps.append(PlanStep(
                    description=f"resolve contradiction between "
                                f"{parts[0]} and {parts[1]}",
                    target_concept=goal,
                    step_type="resolve",
                ))
            else:
                self._decompose_understand(
                    goal, steps, visited, max_depth, depth=0,
                )
        elif goal_type == "compare":
            parts = goal.split(" vs ")
            if len(parts) == 2:
                self._decompose_understand(
                    parts[0], steps, visited, max_depth, depth=0,
                )
                self._decompose_understand(
                    parts[1], steps, visited, max_depth, depth=0,
                )
                steps.append(PlanStep(
                    description=f"compare {parts[0]} and {parts[1]}",
                    target_concept=goal,
                    step_type="compare",
                ))
            else:
                self._decompose_understand(
                    goal, steps, visited, max_depth, depth=0,
                )
        else:
            self._decompose_understand(
                goal, steps, visited, max_depth, depth=0,
            )

        # Always add a final verification step.
        if steps:
            steps.append(PlanStep(
                description=f"verify: {goal} {goal_type} achieved",
                target_concept=goal,
                step_type="verify",
            ))

        return steps

    def _decompose_understand(
        self,
        concept: str,
        steps: list[PlanStep],
        visited: set[str],
        max_depth: int,
        depth: int,
    ) -> None:
        """Decompose an UNDERSTAND goal into prerequisite steps.

        Recursively finds DEPENDS_ON and ENABLES edges — concepts
        that must be understood first — and adds them as steps before
        the goal concept.
        """
        if concept in visited or depth >= max_depth:
            return
        visited.add(concept)

        # Find prerequisites (incoming DEPENDS_ON and ENABLES edges).
        prereqs = self._find_prerequisites(concept)
        for prereq in prereqs:
            if prereq not in visited:
                self._decompose_understand(
                    prereq, steps, visited, max_depth, depth + 1,
                )

        # Add the step for this concept (if not already added).
        existing = [s for s in steps if s.target_concept == concept]
        if not existing:
            steps.append(PlanStep(
                description=f"understand {concept}",
                target_concept=concept,
                step_type="understand",
                prerequisites=prereqs,
            ))

    def _decompose_causal(
        self,
        concept: str,
        steps: list[PlanStep],
        visited: set[str],
        max_depth: int,
        depth: int,
    ) -> None:
        """Decompose an ACHIEVE/EXPLAIN goal into causal chain steps.

        Finds CAUSES and LEADS_TO edges leading to the goal concept,
        ordering them as steps that lead up to the goal.
        """
        if concept in visited or depth >= max_depth:
            return
        visited.add(concept)

        # Find causes (concepts that CAUSE or LEAD_TO this concept).
        causes = self._find_causes(concept)
        for cause in causes:
            if cause not in visited:
                self._decompose_causal(
                    cause, steps, visited, max_depth, depth + 1,
                )

        # Also find prerequisites.
        prereqs = self._find_prerequisites(concept)
        for prereq in prereqs:
            if prereq not in visited:
                self._decompose_understand(
                    prereq, steps, visited, max_depth, depth + 1,
                )

        existing = [s for s in steps if s.target_concept == concept]
        if not existing:
            steps.append(PlanStep(
                description=f"achieve {concept}",
                target_concept=concept,
                step_type="achieve",
                prerequisites=causes + prereqs,
            ))

    def _find_prerequisites(self, concept: str) -> list[str]:
        """Find prerequisite concepts for a given concept.

        Looks for incoming DEPENDS_ON and ENABLES edges — concepts
        that this concept depends on or is enabled by.
        """
        from ..concepts import RelationType

        prereqs: list[str] = []
        cid = self.network._resolve(concept)
        if not cid:
            return prereqs
        for edge in self.network.get_edges(concept, "in"):
            if edge.relation in (
                RelationType.DEPENDS_ON, RelationType.ENABLES,
            ):
                source_concept = self.network.get_concept(edge.source)
                if source_concept:
                    prereqs.append(source_concept.id)
        return list(dict.fromkeys(prereqs))  # deduplicate, preserve order

    def _find_causes(self, concept: str) -> list[str]:
        """Find causal antecedents for a given concept.

        Looks for incoming CAUSES and LEADS_TO edges — concepts that
        cause or lead to this concept.
        """
        from ..concepts import RelationType

        causes: list[str] = []
        cid = self.network._resolve(concept)
        if not cid:
            return causes
        for edge in self.network.get_edges(concept, "in"):
            if edge.relation in (
                RelationType.CAUSES, RelationType.LEADS_TO,
            ):
                source_concept = self.network.get_concept(edge.source)
                if source_concept:
                    causes.append(source_concept.id)
        return list(dict.fromkeys(causes))

    # ─── Plan evaluation ─────────────────────────────────────────

    def _evaluate_feasibility(self, plan: Plan) -> float:
        """Evaluate a plan's feasibility.

        A plan is feasible if:
        - Each step's target concept exists in the network or has a
          discoverable path.
        - The plan has a reasonable number of steps (not too many).
        - No step has too many prerequisites (not too complex).

        The *primary* concept (the goal concept) is weighted more
        heavily than verification steps — a plan for an unknown concept
        is infeasible even if the verify step is trivially feasible.

        Returns a feasibility score in [0, 1].
        """
        if not plan.steps:
            return 0.0

        score = 0.0
        total_weight = 0.0
        for step in plan.steps:
            # Verification steps have lower weight.
            weight = 0.3 if step.step_type == "verify" else 1.0
            total_weight += weight

            # Does the target concept exist?
            concept = self.network.get_concept(step.target_concept)
            if concept is not None:
                step_score = 0.8
            else:
                # Check if it's a compound concept (e.g., "A vs B").
                if " vs " in step.target_concept:
                    parts = step.target_concept.split(" vs ")
                    exists = [
                        self.network.get_concept(p) is not None
                        for p in parts
                    ]
                    step_score = 0.6 if all(exists) else 0.2
                elif step.step_type == "verify":
                    # Verification steps are always feasible.
                    step_score = 0.9
                else:
                    # Unknown concept — low feasibility.
                    step_score = 0.2

            # Penalize too many prerequisites.
            if len(step.prerequisites) > 3:
                step_score *= 0.7

            score += step_score * weight

        # Weighted average.
        score /= total_weight if total_weight > 0 else 1.0

        # Penalize plans that are too long.
        if len(plan.steps) > 8:
            score *= 0.8

        return min(1.0, score)

    # ─── Plan execution ──────────────────────────────────────────

    def advance(self, plan: Plan) -> PlanStep | None:
        """Advance the plan by one step.

        Returns the current step to be executed, or None if the plan
        is complete or blocked. The caller is responsible for
        executing the step and calling ``mark_step`` with the result.

        Args:
            plan: The plan to advance.

        Returns:
            The step to execute, or None if the plan is done.
        """
        import time

        if plan.status in (PlanStatus.COMPLETED, PlanStatus.BLOCKED):
            return None

        # Skip completed/failed/skipped steps.
        while (
            plan.current_step_index < len(plan.steps)
            and plan.steps[plan.current_step_index].status in (
                PlanStepStatus.COMPLETED,
                PlanStepStatus.SKIPPED,
                PlanStepStatus.FAILED,
            )
        ):
            plan.current_step_index += 1

        if plan.current_step_index >= len(plan.steps):
            # All steps done.
            plan.status = PlanStatus.COMPLETED
            self._completed_count += 1
            return None

        step = plan.steps[plan.current_step_index]
        step.status = PlanStepStatus.IN_PROGRESS
        step.attempt_count += 1
        plan.status = PlanStatus.ACTIVE
        plan.updated_at = int(time.time() * 1000)
        self._steps_executed += 1
        return step

    def mark_step(
        self,
        plan: Plan,
        step: PlanStep,
        success: bool,
        result_summary: str = "",
        confidence: float = 0.0,
    ) -> None:
        """Mark a step as completed or failed.

        If the step fails and has remaining attempts, it stays
        IN_PROGRESS for retry. If it exceeds max_attempts, it's
        marked FAILED and the plan may be revised.

        Args:
            plan: The plan the step belongs to.
            step: The step to mark.
            success: Whether the step succeeded.
            result_summary: What happened during execution.
            confidence: Confidence in the step's outcome.
        """
        import time

        step.result_summary = result_summary
        step.confidence = confidence
        plan.updated_at = int(time.time() * 1000)

        if success:
            step.status = PlanStepStatus.COMPLETED
            plan.current_step_index += 1
            # Check if plan is complete.
            if plan.is_complete:
                plan.status = PlanStatus.COMPLETED
                self._completed_count += 1
        else:
            if step.attempt_count >= step.max_attempts:
                step.status = PlanStepStatus.FAILED
                # Try to revise the plan.
                revised = self._revise_plan(plan, step)
                if not revised:
                    plan.status = PlanStatus.BLOCKED
                    plan.failure_reason = (
                        f"step '{step.description}' failed after "
                        f"{step.attempt_count} attempts; "
                        f"no alternative path found"
                    )
                    self._blocked_count += 1
            # else: stays IN_PROGRESS for retry

    # ─── Plan revision ───────────────────────────────────────────

    def _revise_plan(self, plan: Plan, failed_step: PlanStep) -> bool:
        """Revise a plan when a step fails.

        Attempts to find an alternative path to the goal by:
        1. Looking for alternative prerequisites.
        2. Inserting new steps to discover the missing concept.
        3. If no alternative exists, returns False (plan is blocked).

        Args:
            plan: The plan to revise.
            failed_step: The step that failed.

        Returns:
            True if the plan was successfully revised, False if
            the plan is blocked.
        """
        plan.revision_count += 1
        self._revised_count += 1

        # Strategy 1: Look for alternative prerequisites.
        # If the failed step was about understanding concept X, and X
        # has other prerequisites not yet in the plan, add them.
        alt_prereqs = self._find_prerequisites(failed_step.target_concept)
        existing_concepts = {s.target_concept for s in plan.steps}

        new_steps: list[PlanStep] = []
        for prereq in alt_prereqs:
            if prereq not in existing_concepts:
                new_steps.append(PlanStep(
                    description=f"understand {prereq} (alternative path)",
                    target_concept=prereq,
                    step_type="understand",
                ))

        if new_steps:
            # Insert the new steps after the failed step.
            insert_idx = plan.current_step_index + 1
            for new_step in new_steps:
                plan.steps.insert(insert_idx, new_step)
                insert_idx += 1
            # Skip the failed step (we found an alternative path).
            failed_step.status = PlanStepStatus.SKIPPED
            return True

        # Strategy 2: If the concept doesn't exist, add a discovery step.
        if self.network.get_concept(failed_step.target_concept) is None:
            # Can't discover unknown concepts without external input.
            # Mark as blocked.
            return False

        # Strategy 3: Try a different approach — mark the step as
        # skipped and see if the plan can continue without it.
        # This works for non-critical steps (e.g., understanding a
        # prerequisite that turns out to be unnecessary).
        if failed_step.step_type == "understand":
            failed_step.status = PlanStepStatus.SKIPPED
            return True

        return False

    # ─── Plan retrieval ──────────────────────────────────────────

    def get_plan_for_goal(self, goal: str) -> Plan | None:
        """Get the plan for a goal, if one exists."""
        return self._plans_by_goal.get(goal)

    def get_active_plan_for_goal(self, goal: str) -> Plan | None:
        """Get the active plan for a goal, if one exists."""
        plan = self._plans_by_goal.get(goal)
        if plan and plan.status == PlanStatus.ACTIVE:
            return plan
        return None

    # ─── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "completed_count": self._completed_count,
            "blocked_count": self._blocked_count,
            "revised_count": self._revised_count,
            "steps_executed": self._steps_executed,
            "plans": [
                {
                    "goal": p.goal,
                    "goal_type": p.goal_type,
                    "status": p.status.value,
                    "current_step_index": p.current_step_index,
                    "feasibility_score": p.feasibility_score,
                    "revision_count": p.revision_count,
                    "progress": p.progress,
                    "remaining_steps": p.remaining_steps,
                    "failure_reason": p.failure_reason,
                    "steps": [
                        {
                            "description": s.description,
                            "target_concept": s.target_concept,
                            "step_type": s.step_type,
                            "status": s.status.value,
                            "attempt_count": s.attempt_count,
                            "confidence": s.confidence,
                            "result_summary": s.result_summary,
                        }
                        for s in p.steps
                    ],
                }
                for p in self._plans[-20:]  # keep last 20
            ],
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        def _int_val(key: str) -> int:
            """Extract an int from persisted data, defaulting to 0."""
            val = data.get(key)
            if isinstance(val, (int, float)):
                return int(val)
            return 0

        self._completed_count = _int_val("completed_count")
        self._blocked_count = _int_val("blocked_count")
        self._revised_count = _int_val("revised_count")
        self._steps_executed = _int_val("steps_executed")
        # Restore plans — clear first so a repeated restore doesn't
        # accumulate duplicates in _plans (memory engine's
        # restore_records does the same).
        self._plans.clear()
        self._plans_by_goal.clear()
        plans_data = data.get("plans", [])
        if isinstance(plans_data, list):
            for pd in plans_data:
                if not isinstance(pd, dict):
                    continue
                goal = str(pd.get("goal", ""))
                goal_type = str(pd.get("goal_type", "understand"))
                plan = Plan(
                    goal=goal,
                    goal_type=goal_type,
                    feasibility_score=float(
                        pd.get("feasibility_score", 0.5)
                    ),
                    current_step_index=int(
                        pd.get("current_step_index", 0)
                    ),
                    revision_count=int(pd.get("revision_count", 0)),
                    failure_reason=str(pd.get("failure_reason", "")),
                )
                # Restore status.
                status_str = str(pd.get("status", "active"))
                try:
                    plan.status = PlanStatus(status_str)
                except ValueError:
                    plan.status = PlanStatus.ACTIVE
                # Restore steps.
                steps_data = pd.get("steps", [])
                if isinstance(steps_data, list):
                    for sd in steps_data:
                        if not isinstance(sd, dict):
                            continue
                        step = PlanStep(
                            description=str(sd.get("description", "")),
                            target_concept=str(
                                sd.get("target_concept", "")
                            ),
                            step_type=str(sd.get("step_type", "")),
                            attempt_count=int(
                                sd.get("attempt_count", 0)
                            ),
                            confidence=float(
                                sd.get("confidence", 0.0)
                            ),
                            result_summary=str(
                                sd.get("result_summary", "")
                            ),
                        )
                        step_status_str = str(
                            sd.get("status", "pending")
                        )
                        try:
                            step.status = PlanStepStatus(step_status_str)
                        except ValueError:
                            step.status = PlanStepStatus.PENDING
                        plan.steps.append(step)
                self._plans.append(plan)
                self._plans_by_goal[goal] = plan

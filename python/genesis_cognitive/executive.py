"""Executive function — the prefrontal cortex's control layer.

Executive functions are the high-level cognitive processes that control
and manage other cognitive processes. They are the "CEO" of the mind —
planning, inhibiting impulses, and switching between tasks. This module
implements three core executive functions, each grounded in prefrontal
cortex (PFC) neuroscience.

# Three executive functions

1. **Planning** — Look-ahead simulation of possible actions and their
   outcomes. The PFC (particularly dorsolateral PFC, DLPFC) supports
   planning by maintaining goals in working memory and simulating
   future states (Schacter et al., 2012; Hassabis & Maguire, 2009).
   This is "episodic future thinking" — the ability to mentally
   simulate possible futures to guide current action.

2. **Response inhibition** — Suppressing impulsive or prepotent
   responses. The right inferior frontal gyrus (rIFG) and pre-supplementary
   motor area (pre-SMA) implement response inhibition (Aron et al.,
   2014). This is the "stop" signal — when a prepotent response is
   detected but should not be executed, inhibition suppresses it.
   Modeled here as a threshold: if a response's impulse strength is
   below the inhibition threshold, it's suppressed.

3. **Task-switching** — Switching between tasks with a switching cost.
   Task-switching engages the posterior parietal cortex and PFC
   (Monsell, 2003; Wager et al., 2004). Switching has a cost: increased
   reaction time and decreased accuracy on the first trial after a
   switch (the "switch cost"). This models the reconfiguration of
   task sets — the mental rules and parameters for a task must be
   loaded into PFC when switching.

References:
- Aron, A. R., et al. (2014). Inhibition and the right inferior frontal
  cortex. Trends in Cognitive Sciences.
- Hassabis, D., & Maguire, E. A. (2009). The construction system of
  the brain. Philosophical Transactions of the Royal Society B.
- Monsell, S. (2003). Task switching. Trends in Cognitive Sciences.
- Schacter, D. L., et al. (2012). The future of memory: remembering,
  imagining, and the brain. Neuron.
- Wager, T. D., et al. (2004). Common and unique components of response
  inhibition. Journal of Cognitive Neuroscience.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class TaskState(Enum):
    """State of a task in the executive function's task set.

    - ACTIVE: Currently being executed.
    - SUSPENDED: Paused, can be resumed.
    - COMPLETED: Finished.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"


@dataclass(slots=True)
class ActionPlan:
    """A plan — a sequence of simulated actions with predicted outcomes.

        Planning involves simulating possible actions and their consequences
        before executing them. Each plan contains:
        - A sequence of steps (actions to take)
        - Predicted outcomes for each step
        - An overall expected value (utility)
        - Confidence in the predictions

        The plan with the highest expected value is selected for execution.
        This is the "model-based" decision-making of the PFC (Daw et al.,
    2005), as opposed to the "model-free" habitual learning of the basal
        ganglia.
    """

    steps: list[str] = field(default_factory=list)
    predicted_outcomes: list[str] = field(default_factory=list)
    expected_value: float = 0.0
    confidence: float = 0.5
    goal: str = ""
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def describe(self) -> str:
        """Human-readable description of the plan."""
        lines = [f"Plan for: {self.goal}", f"Expected value: {self.expected_value:.2f}"]
        for i, (step, outcome) in enumerate(zip(self.steps, self.predicted_outcomes, strict=False)):
            lines.append(f"  {i + 1}. {step} → {outcome}")
        return "\n".join(lines)


@dataclass(slots=True)
class Task:
    """A task in the executive function's task set.

    A task is a goal-directed activity with associated rules and
    parameters. Task-switching involves loading a new task set
    (its rules and parameters) into PFC, which incurs a switching cost.
    """

    name: str
    goal: str = ""
    state: TaskState = TaskState.SUSPENDED
    rules: dict[str, str] = field(default_factory=dict)
    activated_at: int = 0
    # How many times this task has been switched to (for adaptation)
    switch_count: int = 0


@dataclass(slots=True)
class ExecutiveIntention:
    """A persistent objective owned by the executive layer.

    An intention survives an individual acting episode. It records why
    Genesis is doing something, what she expects to happen, and what actually
    happened so the next decision can be based on consequences rather than
    merely repeating the last action.
    """

    objective: str
    reason: str = ""
    priority: float = 0.5
    confidence: float = 0.5
    expected_outcome: str = ""
    status: str = "pending"
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    attempts: int = 0
    last_observation: str = ""
    actual_outcome: str = ""
    actionable: bool = True


@dataclass(slots=True)
class InhibitionResult:
    """The result of a response inhibition check.

    Captures whether a response was inhibited and why — for
    introspection and debugging.
    """

    inhibited: bool
    impulse_strength: float
    threshold: float
    reason: str = ""


@dataclass(slots=True)
class SwitchResult:
    """The result of a task switch.

    Captures the switching cost (RT increase, accuracy decrease)
    that was applied, for introspection.
    """

    from_task: str
    to_task: str
    rt_cost: float  # reaction time increase (ms or relative)
    accuracy_cost: float  # accuracy decrease (0..1)
    success: bool = True


class ExecutiveFunction:
    """The prefrontal cortex's control layer.

    Implements planning, response inhibition, and task-switching —
    the three core executive functions that control and manage other
    cognitive processes.

    # Planning

    ``plan(goal, possible_actions, outcome_predictor)`` simulates
    possible action sequences and their outcomes, then selects the
    plan with the highest expected value. The outcome predictor is
    a callable that, given an action, returns a predicted outcome
    and its value.

    # Response inhibition

    ``inhibit_response(impulse_strength)`` checks if a prepotent
    response should be suppressed. If the impulse is below the
    inhibition threshold, it's inhibited. The threshold can be
    raised (more cautious) or lowered (more impulsive) dynamically.

    # Task-switching

    ``switch_task(task_name)`` switches to a new task, incurring a
    switching cost (increased RT, decreased accuracy). The cost is
    highest on the first switch to a task and decreases with practice
    (the task set becomes more accessible). This models task-set
    reconfiguration (Rogers & Monsell, 1995).
    """

    def __init__(
        self,
        inhibition_threshold: float = 0.5,
        switch_rt_cost: float = 0.3,
        switch_accuracy_cost: float = 0.15,
        planning_depth: int = 3,
    ) -> None:
        """Initialize the executive function layer.

        Args:
            inhibition_threshold: Minimum impulse strength for a response
                to pass inhibition. Default 0.5. Higher = more cautious.
            switch_rt_cost: Reaction time increase on task switch
                (as a fraction, e.g., 0.3 = 30% slower). Default 0.3.
            switch_accuracy_cost: Accuracy decrease on task switch
                (0..1). Default 0.15.
            planning_depth: How many steps ahead to simulate in planning.
                Default 3. Higher = deeper planning but slower.
        """
        self.inhibition_threshold = inhibition_threshold
        self.switch_rt_cost = switch_rt_cost
        self.switch_accuracy_cost = switch_accuracy_cost
        self.planning_depth = planning_depth

        self._tasks: dict[str, Task] = {}
        self._active_task: str = ""
        self._current_plan: ActionPlan | None = None
        self._inhibition_count: int = 0
        self._switch_count: int = 0
        self._intentions: dict[str, ExecutiveIntention] = {}
        self._active_intention: str | None = None

    # ─── Planning ─────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        possible_actions: list[str],
        outcome_predictor: Callable[[str], tuple[str, float] | str] | None = None,
    ) -> ActionPlan:
        """Simulate possible actions and select the best plan.

        Given a goal and a set of possible actions, this simulates
        action sequences up to ``planning_depth`` steps, predicts
        outcomes, and selects the plan with the highest expected value.

        The outcome predictor is a callable that takes an action string
        and returns a tuple of (predicted_outcome: str, value: float).
        Without a predictor, the planner does not invent an outcome
        model from lexical overlap; it returns an explicitly
        ungrounded plan with zero confidence. Planning without a
        transition/outcome model is not model-based planning.

        Args:
            goal: The goal to plan for.
            possible_actions: Available actions to choose from.
            outcome_predictor: Optional callable for predicting outcomes.

        Returns:
            The best ActionPlan found.
        """
        if not possible_actions:
            return ActionPlan(goal=goal, expected_value=0.0, confidence=0.0)

        if outcome_predictor is None or not callable(outcome_predictor):
            # An action's wording is not evidence that the action will
            # achieve the goal. Returning a lexical guess here creates
            # systematic false confidence and prevents callers from
            # distinguishing planning from intent matching.
            return ActionPlan(
                steps=list(possible_actions[:1]),
                predicted_outcomes=[],
                expected_value=0.0,
                confidence=0.0,
                goal=goal,
            )

        best_plan = ActionPlan(goal=goal, expected_value=float("-inf"))

        # Search the supplied transition/outcome model. The executive
        # layer owns search and value aggregation; the caller owns the
        # world model.
        for action in possible_actions:
            plan = self._simulate_action_plan(
                action, possible_actions, outcome_predictor, goal
            )
            if plan.expected_value > best_plan.expected_value:
                best_plan = plan

        self._current_plan = best_plan
        return best_plan

    def _simulate_action_plan(
        self,
        action: str,
        possible_actions: list[str],
        outcome_predictor: Callable[[str], tuple[str, float] | str],
        goal: str,
    ) -> ActionPlan:
        """Simulate a single action sequence forward and return the plan."""
        steps = [action]
        outcomes: list[str] = []
        value = 0.0

        # Simulate forward
        current_actions = possible_actions
        for depth in range(self.planning_depth):
            if not current_actions:
                break

            result = outcome_predictor(steps[-1])
            if isinstance(result, tuple):
                outcome, step_value = result
            else:
                outcome, step_value = str(result), 0.5

            outcomes.append(outcome)
            value += step_value * (0.7**depth)  # discount future

            # For greedy planning, pick the best next action
            if depth < self.planning_depth - 1 and current_actions:
                # Query every candidate through the same outcome model;
                # never choose the next action from surface lexical
                # similarity to the goal.
                candidates: list[tuple[float, str]] = []
                for candidate in current_actions:
                    candidate_result = outcome_predictor(candidate)
                    if isinstance(candidate_result, tuple):
                        _, candidate_value = candidate_result
                    else:
                        candidate_value = 0.5
                    candidates.append((float(candidate_value), candidate))
                if candidates:
                    steps.append(max(candidates, key=lambda item: item[0])[1])

        return ActionPlan(
            steps=steps,
            predicted_outcomes=outcomes,
            expected_value=value / max(len(steps), 1),
            confidence=min(1.0, max(0.0, value / 2.0)),
            goal=goal,
        )

    @property
    def current_plan(self) -> ActionPlan | None:
        """The most recently generated plan."""
        return self._current_plan

    # ─── Persistent intention / consequence loop ─────────────────

    def form_intention(
        self,
        objective: str,
        *,
        reason: str = "",
        priority: float = 0.5,
        confidence: float = 0.5,
        expected_outcome: str = "",
        actionable: bool = True,
    ) -> ExecutiveIntention:
        """Create or refresh a persistent objective."""
        key = objective.strip().lower()
        if not key:
            raise ValueError("objective must not be empty")
        current = self._intentions.get(key)
        if current is not None and current.status not in {"completed", "abandoned"}:
            current.priority = max(current.priority, max(0.0, min(1.0, priority)))
            current.confidence = max(0.0, min(1.0, confidence))
            if reason:
                current.reason = reason
            if expected_outcome:
                current.expected_outcome = expected_outcome
            current.actionable = current.actionable or actionable
            return current
        intention = ExecutiveIntention(
            objective=objective.strip(), reason=reason,
            priority=max(0.0, min(1.0, priority)),
            confidence=max(0.0, min(1.0, confidence)),
            expected_outcome=expected_outcome,
            actionable=actionable,
        )
        self._intentions[key] = intention
        return intention

    def select_intention(
        self,
        candidates: list[ExecutiveIntention],
        allowed_kinds: set[str] | None = None,
    ) -> ExecutiveIntention | None:
        """Select the strongest executable intention.

        The executive may retain advisory objectives that no current
        actuator can execute. An actuator can restrict arbitration to its
        supported objective kinds so an objective is never falsely claimed
        as executed merely because it won the global competition.
        """
        actionable = [
            c for c in candidates
            if c.actionable
            and c.status == "pending"
            and (
                allowed_kinds is None
                or c.objective.split(":", 1)[0] in allowed_kinds
            )
        ]
        if not actionable:
            return None
        selected = max(
            actionable,
            key=lambda c: (
                c.priority * max(c.confidence, 0.05),
                -c.attempts,
            ),
        )
        self._active_intention = selected.objective.strip().lower()
        selected.status = "active"
        return selected

    def observe_intention(
        self,
        intention: ExecutiveIntention,
        *,
        actual_outcome: str,
        success: bool,
    ) -> float:
        """Record an outcome and return a bounded prediction-error signal."""
        intention.attempts += 1
        intention.actual_outcome = actual_outcome
        intention.last_observation = actual_outcome
        if success:
            intention.status = "completed"
            error = 0.0 if intention.expected_outcome else 0.25
        else:
            intention.status = "pending"
            error = 1.0
        if self._active_intention == intention.objective.strip().lower():
            self._active_intention = None
        return error

    def revise_intention(
        self, intention: ExecutiveIntention, *, reason: str = ""
    ) -> ExecutiveIntention:
        """Keep an unfinished objective alive after failure and lower confidence."""
        intention.status = "pending"
        intention.confidence = max(0.05, intention.confidence * 0.8)
        intention.priority = min(1.0, intention.priority + 0.05)
        if reason:
            intention.last_observation = reason
        return intention

    def to_dict(self) -> dict[str, object]:
        """Serialize executive control state for restart-safe persistence."""
        return {
            "intentions": [
                {
                    "objective": i.objective,
                    "reason": i.reason,
                    "priority": i.priority,
                    "confidence": i.confidence,
                    "expected_outcome": i.expected_outcome,
                    "status": i.status,
                    "created_at": i.created_at,
                    "attempts": i.attempts,
                    "last_observation": i.last_observation,
                    "actual_outcome": i.actual_outcome,
                    "actionable": i.actionable,
                }
                for i in self._intentions.values()
            ],
            "active_intention": self._active_intention,
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore persistent intentions after validating their shape."""
        raw = data.get("intentions", [])
        if not isinstance(raw, list):
            raise TypeError("executive.intentions must be a list")
        restored: dict[str, ExecutiveIntention] = {}
        for item in raw:
            if not isinstance(item, dict):
                raise TypeError("executive intention must be an object")
            objective = str(item.get("objective", "")).strip()
            if not objective:
                continue
            intention = ExecutiveIntention(
                objective=objective,
                reason=str(item.get("reason", "")),
                priority=max(0.0, min(1.0, float(item.get("priority", 0.5)))),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                expected_outcome=str(item.get("expected_outcome", "")),
                status=str(item.get("status", "pending")),
                created_at=int(item.get("created_at", int(time.time() * 1000))),
                attempts=max(0, int(item.get("attempts", 0))),
                last_observation=str(item.get("last_observation", "")),
                actual_outcome=str(item.get("actual_outcome", "")),
                actionable=bool(item.get("actionable", True)),
            )
            if intention.status == "active":
                intention.status = "pending"
            restored[objective.lower()] = intention
        self._intentions = restored
        # An in-flight action cannot safely survive a process boundary.
        # Preserve the objective, but require the executive to select it
        # again after restart.
        self._active_intention = None

    @property
    def intentions(self) -> tuple[ExecutiveIntention, ...]:
        """Current persistent intentions."""
        return tuple(self._intentions.values())

    @property
    def actionable_intentions(self) -> tuple[ExecutiveIntention, ...]:
        """Persistent objectives still eligible for execution."""
        return tuple(
            intention
            for intention in self._intentions.values()
            if intention.actionable and intention.status in {"pending", "active"}
        )

    def ingest_candidates(
        self, candidates: list[dict[str, object]]
    ) -> list[ExecutiveIntention]:
        """Convert salient cognitive signals into persistent objectives.

        Signal producers provide meaning and salience; the executive owns
        normalization, deduplication, priority, and persistence.
        """
        created: list[ExecutiveIntention] = []
        mode_kind = {
            # These modes produce a question/topic that the existing
            # learning actuator can investigate.
            "curiosity": "learn",
            "memory": "learn",
            "activation": "learn",
            "emotional": "learn",
            "dream_reflection": "learn",
            "prediction_error": "learn",
            "environment": "learn",
            "social": "learn",
            "connection": "learn",
            # Code-directed signals have a real inspect actuator.
            "bug_concern": "inspect",
            "improvement": "inspect",
            # Internal/body pressure remains an executive advisory
            # objective until a matching regulator actuator exists.
            "interoceptive": "observe",
            "embodiment": "observe",
            "body_deviation": "observe",
            "distress": "observe",
        }
        for candidate in candidates:
            topic = str(candidate.get("topic", "")).strip()
            if not topic:
                continue
            mode = str(candidate.get("mode", "")).strip().lower()
            source = str(candidate.get("source", mode)).strip() or "signal"
            raw_salience = candidate.get("salience", 0.0)
            if not isinstance(raw_salience, (int, float)):
                continue
            salience = max(0.0, min(1.0, float(raw_salience)))
            if salience <= 0.0:
                continue
            kind = str(candidate.get("kind", mode_kind.get(mode, "observe"))).strip()
            if kind not in {"learn", "inspect", "explore", "observe", "measure"}:
                continue
            actionable = bool(candidate.get(
                "actionable",
                kind in {"learn", "inspect", "explore", "observe", "measure"}
                and mode not in {"interoceptive", "embodiment", "body_deviation", "distress"},
            ))
            objective = f"{kind}:{topic}"
            intention = self.form_intention(
                objective,
                reason=source,
                priority=salience,
                confidence=max(0.1, min(1.0, salience)),
                expected_outcome=(
                    "knowledge or relationships updated"
                    if kind == "learn"
                    else "code structure observed"
                    if kind == "inspect"
                    else "an observable result recorded"
                ),
                actionable=actionable,
            )
            created.append(intention)
        return created

    @property
    def active_intention(self) -> ExecutiveIntention | None:
        """The intention currently being executed, if any."""
        if self._active_intention is None:
            return None
        return self._intentions.get(self._active_intention)

    # ─── Response inhibition ──────────────────────────────────────

    def inhibit_response(self, impulse_strength: float) -> InhibitionResult:
        """Check if a prepotent response should be inhibited.

        If the impulse strength is below the inhibition threshold,
        the response is suppressed. This models the stop-signal
        paradigm: a prepotent response is prepared, but if a stop
        signal arrives (here, the threshold check), the response is
        inhibited (Aron et al., 2014).

        Args:
            impulse_strength: The strength of the prepotent response
                impulse (0..1). High = strong impulse, low = weak.

        Returns:
            InhibitionResult indicating whether the response was
            inhibited and why.
        """
        if impulse_strength < self.inhibition_threshold:
            self._inhibition_count += 1
            return InhibitionResult(
                inhibited=True,
                impulse_strength=impulse_strength,
                threshold=self.inhibition_threshold,
                reason=(
                    f"Impulse ({impulse_strength:.2f}) below inhibition "
                    f"threshold ({self.inhibition_threshold:.2f})"
                ),
            )

        return InhibitionResult(
            inhibited=False,
            impulse_strength=impulse_strength,
            threshold=self.inhibition_threshold,
            reason="Impulse strong enough to proceed",
        )

    def set_inhibition_threshold(self, threshold: float) -> None:
        """Adjust the inhibition threshold.

        Called by the error monitor when caution level changes —
        after errors, the threshold rises (more cautious); after
        successes, it falls (more impulsive). This is the PFC-ACC
        control loop (Botvinick et al., 2001).

        Args:
            threshold: New inhibition threshold (0..1).
        """
        self.inhibition_threshold = max(0.0, min(1.0, threshold))

    @property
    def inhibition_count(self) -> int:
        """Total number of responses inhibited."""
        return self._inhibition_count

    # ─── Task-switching ───────────────────────────────────────────

    def register_task(self, name: str, goal: str = "", rules: dict[str, str] | None = None) -> Task:
        """Register a task in the task set.

        Tasks must be registered before they can be switched to.

        Args:
            name: Task name (unique identifier).
            goal: What the task aims to achieve.
            rules: Task rules/parameters.

        Returns:
            The registered Task.
        """
        task = Task(name=name, goal=goal, rules=rules or {})
        self._tasks[name] = task
        return task

    def switch_task(self, task_name: str) -> SwitchResult:
        """Switch to a different task, incurring a switching cost.

                Switching tasks requires reconfiguring the task set — loading
                the new task's rules and parameters into PFC. This incurs a
                cost: increased RT and decreased accuracy on the first trial
                after the switch (Monsell, 2003).

                The cost decreases with practice: tasks that have been switched
                to more frequently have lower switch costs (the task set is
                more accessible). This models task-set priming (Allport et al.,
        1994).

                Args:
                    task_name: The task to switch to.

                Returns:
                    SwitchResult with the costs that were applied.
        """
        from_task = self._active_task

        if task_name not in self._tasks:
            # Auto-register if not registered
            self.register_task(task_name)

        task = self._tasks[task_name]

        # Suspend the current task
        if from_task and from_task in self._tasks:
            self._tasks[from_task].state = TaskState.SUSPENDED

        # Activate the new task
        task.state = TaskState.ACTIVE
        task.activated_at = int(time.time() * 1000)
        task.switch_count += 1
        self._active_task = task_name
        self._switch_count += 1

        # Compute switching cost — decreases with practice
        # First switch: full cost. After 5+ switches: 50% cost.
        practice_factor = max(0.5, 1.0 - (task.switch_count - 1) * 0.1)
        rt_cost = self.switch_rt_cost * practice_factor
        accuracy_cost = self.switch_accuracy_cost * practice_factor

        return SwitchResult(
            from_task=from_task,
            to_task=task_name,
            rt_cost=rt_cost,
            accuracy_cost=accuracy_cost,
        )

    @property
    def active_task(self) -> str:
        """The currently active task name."""
        return self._active_task

    @property
    def active_task_obj(self) -> Task | None:
        """The currently active Task object."""
        return self._tasks.get(self._active_task) if self._active_task else None

    @property
    def tasks(self) -> dict[str, Task]:
        """All registered tasks."""
        return dict(self._tasks)

    @property
    def switch_count(self) -> int:
        """Total number of task switches."""
        return self._switch_count

    def get_switch_cost(self, task_name: str) -> tuple[float, float]:
        """Predict the switching cost for a task (without switching).

        Useful for deciding whether a switch is worth it.

        Args:
            task_name: The task to evaluate.

        Returns:
            Tuple of (rt_cost, accuracy_cost) that would be incurred.
        """
        task = self._tasks.get(task_name)
        switch_count = task.switch_count if task else 0
        practice_factor = max(0.5, 1.0 - switch_count * 0.1)
        return (
            self.switch_rt_cost * practice_factor,
            self.switch_accuracy_cost * practice_factor,
        )

    def clear(self) -> None:
        """Clear all tasks and plans."""
        self._tasks.clear()
        self._active_task = ""
        self._current_plan = None

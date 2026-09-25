"""Pattern agent — solves continuation tasks through task competence.

Same loop as ``SorterAgent``, different world. The agent is not told
the period: it observes candidates (each carrying a perceptible
``lookback`` — how far back the same mark last appeared), tries them,
and learns the lookback→acceptance affordance empirically. A solved
line consolidates the learned map as a procedure; the next pattern
starts with "the right token repeats the one that came period-k
before" instead of exploring blind.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..reasoning import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
    TaskContext,
)
from .puzzle import PatternLine, Placement


@dataclass
class _LookStats:
    """Observed cost of placements at one lookback distance."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class PatternResult:
    """Outcome of one pattern-line episode."""

    solved: bool
    state: str  # "WIN" | "STUCK"
    steps: int
    placed: int
    wasted: int


class PatternAgent:
    """Trial-and-error continuation solver on the competence substrate."""

    def __init__(
        self,
        seed: int = 42,
        epsilon: float = 0.35,
        task_competence: TaskCompetence | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.epsilon = epsilon
        self.task_competence = task_competence or TaskCompetence()
        self._task_context: TaskContext | None = None
        self._episode_skill_ids: set[str] = set()
        # Learned affordance: a candidate whose mark last appeared
        # `lookback` positions back costs this much on average
        # (1 = rejected, 0 = continued the pattern).
        self._look_stats: dict[int, _LookStats] = {}
        # Schema-level prior adopted from skills: lookback -> cost.
        self._look_priors: dict[int, float] = {}
        # Move memory, episode-local: a rejection is a property of the
        # (token, position) pair — the same token may belong elsewhere.
        self._penalty: dict[tuple[int, int], int] = {}

    # ── Task competence ───────────────────────────────────────

    @staticmethod
    def _bucket(n: int) -> str:
        if n <= 0:
            return "none"
        if n <= 3:
            return "few"
        if n <= 8:
            return "some"
        return "many"

    def _line_features(self, line: PatternLine) -> dict[str, Any]:
        """World-observable structure for schema matching."""
        marks = {t.mark for t in line.pool.values()} | {
            c for c in line.cells if c is not None
        }
        return {
            "sequence.continue": True,
            "cells.count": self._bucket(len(line.cells)),
            "pool.count": self._bucket(len(line.pool)),
            "cells.unfilled": self._bucket(line.mismatches()),
            "marks.variety": self._bucket(len(marks)),
        }

    def _recognize_line(self, line: PatternLine) -> TaskContext:
        """Bind this line to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="sequence.continue",
            state=self._line_features(line),
            actions=("place",),
            goal_conditions=[GoalCondition("cells.unfilled", "eq", 0)],
            entities=("token", "position"),
            # "Continue what repeats" is the family shape — a bead
            # string, a rhythm, a counting order share it.
            roles=("sequence", "continue", "order"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed the lookback→cost model from skills won on similar
        patterns. What transfers is the *shape* — "tokens repeating a
        recent position continue the pattern" — not which mark went
        where. Marks and positions are bindings."""
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                if step.family != "continuation":
                    continue
                lookback = step.parameters.get("lookback")
                cost = step.parameters.get("mean_cost")
                if isinstance(lookback, int) and isinstance(
                    cost, int | float
                ):
                    adopted = True
                    prev = self._look_priors.get(lookback)
                    self._look_priors[lookback] = (
                        float(cost)
                        if prev is None
                        else min(prev, float(cost))
                    )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _transition_state(self, line: PatternLine) -> dict[str, Any]:
        return line.state()

    # ── Action selection ──────────────────────────────────────

    def _move_score(self, lookback: int) -> float:
        """Expected quality of a placement — negative cost, so higher
        is better. Local observations beat adopted priors."""
        st = self._look_stats.get(lookback)
        if st is not None and st.attempts:
            return -st.mean_cost
        prior = self._look_priors.get(lookback)
        if prior is not None:
            return -prior
        return 0.0

    def _choose(self, candidates: list[Placement]) -> Placement:
        def score(c: Placement) -> float:
            return (
                self._move_score(c.lookback)
                - 0.75 * self._penalty.get((c.pos, c.token_id), 0)
                + self._rng.random() * 0.05
            )

        learned = self._look_priors or any(
            st.attempts for st in self._look_stats.values()
        )
        if not learned or self._rng.random() < self.epsilon:
            # Exploration prefers untried (position, token) pairs.
            least = min(
                self._penalty.get((c.pos, c.token_id), 0)
                for c in candidates
            )
            fresh = [
                c
                for c in candidates
                if self._penalty.get((c.pos, c.token_id), 0) == least
            ]
            return self._rng.choice(fresh)
        scored = [(score(c), c) for c in candidates]
        best = max(s for s, _ in scored)
        return self._rng.choice([c for s, c in scored if s == best])

    # ── Episode ───────────────────────────────────────────────

    def step(self, line: PatternLine) -> str:
        """One move: place a token into an empty position."""
        context = self._recognize_line(line)
        candidates = line.candidates()
        if not candidates:
            return self._finish(line, "STUCK")
        before = self._transition_state(line)
        chosen = self._choose(candidates)
        outcome = line.place(chosen.token_id, chosen.pos)
        if outcome is None:
            return self._finish(line, "STUCK")
        st = self._look_stats.setdefault(
            int(outcome["lookback"]), _LookStats()
        )
        st.attempts += 1
        st.total_cost += 0.0 if outcome["accepted"] else 1.0
        if not outcome["accepted"]:
            key = (chosen.pos, chosen.token_id)
            self._penalty[key] = self._penalty.get(key, 0) + 1
        self.task_competence.record_transition(
            context, "place", before, self._transition_state(line)
        )
        if line.complete():
            return self._finish(line, "WIN")
        return "PLAYING"

    def solve(
        self, line: PatternLine, max_steps: int = 500
    ) -> PatternResult:
        """Drive one pattern to completion or exhaustion."""
        self._penalty.clear()
        steps = 0
        wasted = 0
        state = "PLAYING"
        last_unfilled = line.mismatches()
        stall = 0
        while state == "PLAYING" and steps < max_steps:
            state = self.step(line)
            steps += 1
            unfilled = line.mismatches()
            if unfilled < last_unfilled:
                stall = 0
            else:
                stall += 1
                wasted += 1
            last_unfilled = unfilled
            if stall > max(20, 3 * len(line.cells)):
                state = self._finish(line, "STUCK")
        if state == "PLAYING":
            state = self._finish(line, "STUCK")
        return PatternResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            placed=sum(1 for c in line.cells if c is not None),
            wasted=wasted,
        )

    def _finish(self, line: PatternLine, state: str) -> str:
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a verified-full line consolidates
        a reusable procedure."""
        if self._task_context is not None:
            if state != "WIN":
                for skill_id in self._episode_skill_ids:
                    self.task_competence.mark_skill_failure(skill_id)
            self.task_competence.record_episode(
                self._task_context,
                steps=self._skill_steps(),
                success=state == "WIN",
                verification_score=1.0 if state == "WIN" else 0.0,
                goal_conditions=[
                    GoalCondition("cells.unfilled", "eq", 0)
                ],
                # The line's own rule check is the world reporting
                # the outcome.
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the lookback→cost affordance map."""
        steps = []
        for lookback in sorted(self._look_stats):
            st = self._look_stats[lookback]
            if not st.attempts:
                continue
            steps.append(
                ProcedureStep(
                    action="place",
                    family="continuation",
                    parameters={
                        "lookback": lookback,
                        "mean_cost": round(st.mean_cost, 3),
                    },
                    description=(
                        f"place a token seen {lookback} back "
                        f"(cost {st.mean_cost:+.2f})"
                        if lookback
                        else f"place an unseen token "
                        f"(cost {st.mean_cost:+.2f})"
                    ),
                )
            )
        return steps

    def describe_policy(self) -> str:
        """Compact description of the learned affordance — telemetry,
        not speech."""
        parts = []
        for lookback in sorted(self._look_stats):
            st = self._look_stats[lookback]
            if not st.attempts:
                continue
            label = f"back{lookback}" if lookback else "unseen"
            parts.append(f"{label}→cost{st.mean_cost:.2f}")
        return "; ".join(parts)

"""Quantities agent — composes a target through task competence.

The agent is not told the target is a capacity: it observes options
(each carries perceptible ``fits``/``exact``/``dead`` features),
tries them, and learns the fit→cost affordance empirically —
overflows get rejected, dead remainders force a lift later. The
wasted lift is charged back to the category of the add that caused
it, so "fits but dead-ends" becomes distinguishable from "fits and
leaves room."
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
from .world import Basket, Option


@dataclass
class _KindStats:
    """Observed cost of adds of one kind."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class QuantitiesResult:
    """Outcome of one worksheet attempt."""

    solved: bool
    state: str  # "WIN" or "STUCK"
    steps: int
    adds: int
    lifts: int


def _kind(option: Option) -> str:
    if not option.fits:
        return "over"
    if option.exact:
        return "exact"
    if option.dead:
        return "dead"
    return "under"


class QuantitiesAgent:
    """Trial-and-error composition solver on the competence substrate."""

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
        # Learned affordance: adds of this kind cost this much on
        # average (rejected move or forced lift; good = 0).
        self._kind_stats: dict[str, _KindStats] = {}
        # Schema-level prior adopted from skills: kind -> prior cost.
        self._kind_priors: dict[str, float] = {}
        # Episode-local: a group that had to be lifted once is less
        # worth retrying in this composition.
        self._lift_penalty: dict[str, int] = {}
        # (name, kind) of each accepted add, in order — undo-last
        # backtracking plus the blame for any dead end it created.
        self._basket_order: list[tuple[str, str]] = []
        # How often each basket composition has been entered this
        # episode — revisiting the same mix means the composition is
        # doomed and the honest move is to dump it and start over.
        self._visit_counts: dict[frozenset, int] = {}
        self._removes_used = 0
        self._dumps_used = 0

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

    def _worksheet_features(self, world: Basket) -> dict[str, Any]:
        """World-observable structure for schema matching."""
        return {
            "puzzle.quantities": True,
            "target.size": self._bucket(world.target),
            "groups.count": self._bucket(
                len(world.pool) + len(world.basket)
            ),
            "groups.unused": self._bucket(len(world.pool)),
            "basket.remaining": self._bucket(abs(world.remaining())),
        }

    def _recognize(self, world: Basket) -> TaskContext:
        """Bind this worksheet to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="quantities.compose",
            state=self._worksheet_features(world),
            actions=("add", "remove"),
            goal_conditions=[
                GoalCondition("basket.remaining", "eq", 0),
            ],
            entities=("group", "basket"),
            # "Compose groups into an exact total under a capacity"
            # is the family shape — making change, packing a budget.
            roles=("compose", "quantity", "budget"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed the kind→cost model from skills won on similar tasks.

        What transfers is the *shape* of the affordance — "overflows
        are rejected, dead remainders cost a lift" — not which group
        went in. Names and counts are bindings.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                if step.family != "option":
                    continue
                kind = step.parameters.get("kind")
                cost = step.parameters.get("mean_cost")
                if isinstance(kind, str) and isinstance(
                    cost, int | float
                ):
                    adopted = True
                    prev = self._kind_priors.get(kind)
                    self._kind_priors[kind] = (
                        float(cost)
                        if prev is None
                        else min(prev, float(cost))
                    )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _transition_state(self, world: Basket) -> dict[str, Any]:
        return {
            "basket.sum": world.basket_sum(),
            "basket.remaining": world.remaining(),
            "groups.pool": len(world.pool),
        }

    # ── Action selection ──────────────────────────────────────

    def _option_score(self, kind: str) -> float:
        """Expected quality — negative cost, so higher is better.
        Local observations beat adopted priors."""
        st = self._kind_stats.get(kind)
        if st is not None and st.attempts:
            return -st.mean_cost
        prior = self._kind_priors.get(kind)
        if prior is not None:
            return -prior
        return 0.0

    def _choose(self, options: list[Option]) -> Option:
        def score(o: Option) -> float:
            return self._option_score(_kind(o)) - 0.75 * float(
                self._lift_penalty.get(o.name, 0)
            )

        learned = self._kind_priors or any(
            st.attempts for st in self._kind_stats.values()
        )
        if not learned or self._rng.random() < self.epsilon:
            least = min(
                self._lift_penalty.get(o.name, 0) for o in options
            )
            fresh = [
                o
                for o in options
                if self._lift_penalty.get(o.name, 0) == least
            ]
            return self._rng.choice(fresh)
        pairs = [(score(o), o) for o in options]
        best = max(s for s, _ in pairs)
        return self._rng.choice([o for s, o in pairs if s == best])

    def _choose_lift(self, world: Basket) -> tuple[str | None, str]:
        """Which basket member to lift on a dead end.

        Not strictly undo-last: the blocking choice is often an
        *early* add (a small group that fit fine but leaves the board
        unfillable). Score each candidate lift by the options it
        would reopen — a group whose removal leaves room for a
        fitting, non-dead add unblocks the board; one that re-dead-
        ends is itself penalized for next time.
        """
        best: tuple[str | None, str] = (None, "under")
        best_key: tuple[float, ...] | None = None
        order = {name: i for i, (name, _) in enumerate(self._basket_order)}
        for name, group in world.basket.items():
            kind = next(
                (k for n, k in self._basket_order if n == name), "under"
            )
            room = world.remaining() + group.count
            others = [*world.pool.values(), group]
            reopened = 0
            for g in others:
                after = room - g.count
                if after < 0:
                    continue
                if after == 0 or any(
                    o.count <= after for o in others if o is not g
                ):
                    reopened += 1
            key = (
                float(reopened),
                -float(self._lift_penalty.get(name, 0)),
                -float(order.get(name, 0)),  # prefer recent on ties
                self._rng.random(),
            )
            if best_key is None or key > best_key:
                best_key, best = key, (name, kind)
        return best

    # ── Episode ───────────────────────────────────────────────

    def step(self, world: Basket) -> str:
        """One action: lift on a dead end, or add a group."""
        context = self._recognize(world)
        before = self._transition_state(world)
        sig = frozenset(world.basket)
        self._visit_counts[sig] = self._visit_counts.get(sig, 0) + 1
        if (
            self._visit_counts[sig] >= 4
            and world.basket
            and world.pool
        ):
            # Same composition again — the mix is doomed. Tip the
            # basket out and rebuild on different groups; the lift
            # penalties remember which groups kept failing.
            world.empty()
            self._basket_order.clear()
            action, self._dumps_used = "dump", self._dumps_used + 1
            self.task_competence.record_transition(
                context, action, before, self._transition_state(world)
            )
            return "PLAYING"
        if world.dead_end():
            name, kind = self._choose_lift(world)
            if name is None:
                return self._finish("STUCK")
            if world.remove(name) is None:
                return self._finish("STUCK")
            self._basket_order = [
                (n, k) for n, k in self._basket_order if n != name
            ]
            # Blame the add that created the dead end for the wasted
            # lift — that is how "fits but dead-ends" becomes
            # distinguishable from "fits and leaves room."
            self._lift_penalty[name] = self._lift_penalty.get(name, 0) + 1
            st = self._kind_stats.setdefault(kind, _KindStats())
            st.attempts += 1
            st.total_cost += 1.0
            action, self._removes_used = "remove", self._removes_used + 1
        else:
            options = world.candidates()
            if not options:
                return self._finish("STUCK")
            chosen = self._choose(options)
            kind = _kind(chosen)
            outcome = world.add(chosen.name)
            if outcome is None:
                return self._finish("STUCK")
            action = "add"
            st = self._kind_stats.setdefault(kind, _KindStats())
            st.attempts += 1
            st.total_cost += 0.0 if outcome["accepted"] else 1.0
            if outcome["accepted"]:
                # Only an accepted add can create a dead end, so it
                # carries the blame if it later forces a lift.
                self._basket_order.append((chosen.name, kind))
        self.task_competence.record_transition(
            context, action, before, self._transition_state(world)
        )
        if world.complete():
            return self._finish("WIN")
        return "PLAYING"

    def solve(
        self, world: Basket, max_steps: int = 300
    ) -> QuantitiesResult:
        """Drive one worksheet to the exact total or exhaustion."""
        # Episode-local memory resets; learned stats and priors persist.
        self._lift_penalty.clear()
        self._basket_order.clear()
        self._visit_counts.clear()
        steps = adds = lifts = 0
        state = "PLAYING"
        while state == "PLAYING" and steps < max_steps:
            before_basket = len(world.basket)
            state = self.step(world)
            steps += 1
            if len(world.basket) > before_basket:
                adds += 1
            elif len(world.basket) < before_basket:
                lifts += 1
        if state == "PLAYING":
            state = self._finish("STUCK")
        return QuantitiesResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            adds=adds,
            lifts=lifts,
        )

    def _finish(self, state: str) -> str:
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a verified-exact basket
        consolidates a reusable procedure."""
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
                    GoalCondition("basket.remaining", "eq", 0),
                ],
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the kind→cost affordance map."""
        steps = []
        for kind in sorted(self._kind_stats):
            st = self._kind_stats[kind]
            if not st.attempts:
                continue
            steps.append(
                ProcedureStep(
                    action="add",
                    family="option",
                    parameters={
                        "kind": kind,
                        "mean_cost": round(st.mean_cost, 3),
                    },
                    description=(
                        f"add a {kind} group (cost {st.mean_cost:+.2f})"
                    ),
                )
            )
        if self._removes_used:
            steps.append(
                ProcedureStep(
                    action="remove",
                    family="repair",
                    parameters={"used": True},
                    description="lift a group when the basket dead-ends",
                )
            )
        return steps

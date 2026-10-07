"""Relations agent — arranges things through task competence.

Same loop as ``AssemblyAgent``, different world. The agent is not
told that satisfying the stated goals is the task: it observes
candidate placements (each carries a perceptible ``agrees`` count —
how many currently-decidable goals it satisfies), tries them, and
learns the agree→cost affordance empirically. Violated goals are
repaired by lifting the more suspect of the two parties — a
violation is a property of the *pair* of placements, not of either
thing alone.
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
from .world import Arrangement, Placement


@dataclass
class _AgreeStats:
    """Observed cost of placements at one agree level."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class RelationsResult:
    """Outcome of one arrangement attempt."""

    solved: bool
    state: str  # "WIN" or "STUCK"
    steps: int
    placements: int
    removals: int
    swaps: int = 0


# A (thing, position) configuration — the dyad member for pair memory.
_Config = tuple[str, int]
_Pair = tuple[_Config, _Config]


class RelationsAgent:
    """Trial-and-error arrangement solver on the competence substrate."""

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
        # Learned affordance: placements satisfying `agrees` decidable
        # goals cost this much on average (violations created; good = 0).
        self._agree_stats: dict[int, _AgreeStats] = {}
        # Schema-level prior adopted from skills: agrees -> prior cost.
        self._agree_priors: dict[int, float] = {}
        # Cross-domain affordance: foreign skills that publish a
        # normalized "support" -> cost curve seed priors on the
        # quarter grid. Local stats and native priors beat these.
        self._evidence_priors: dict[float, float] = {}
        # Goals in the active episode — the scale of the support axis.
        self._n_goals = 0
        # Episode-local pair memory: a violated goal is a property of
        # the *pair* of placements, not of either thing — the correct
        # config stays usable against other partners.
        self._pair_penalty: dict[_Pair, int] = {}
        # pos -> (thing, step placed): names the configs on both sides
        # of a violated goal; the step enables recency tiebreaks.
        self._placed_as: dict[int, tuple[str, int]] = {}
        self._clock = 0
        self._removes_used = 0
        self._swaps_used = 0
        # Escalation: how many consecutive steps the same violated
        # goals have persisted — moving only the named parties forever
        # means the *position topology* is wrong, and a bystander has
        # to move to open new options.
        self._viol_sig: frozenset | None = None
        self._viol_repeat = 0

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

    def _arrangement_features(
        self, world: Arrangement
    ) -> dict[str, Any]:
        """World-observable structure for schema matching."""
        rels = {g.rel for g in world.goals}
        return {
            "puzzle.relations": True,
            "row.size": self._bucket(world.positions),
            "things.count": self._bucket(
                len(world.pool) + len(world.board)
            ),
            "goals.count": self._bucket(len(world.goals)),
            "goals.variety": self._bucket(len(rels)),
            "goals.seriation": "ordered" in rels,
            "goals.unsatisfied": self._bucket(world.mismatches()),
        }

    def _recognize(self, world: Arrangement) -> TaskContext:
        """Bind this arrangement to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="relations.arrange",
            state=self._arrangement_features(world),
            actions=("place", "remove"),
            goal_conditions=[
                GoalCondition("goals.unsatisfied", "eq", 0),
                GoalCondition("things.placed", "eq", len(world.things)),
            ],
            entities=("thing", "position"),
            # "Arrange named things so stated relations hold" is the
            # family shape — directions, ordering, seating charts.
            # "constrain" names the abstract role it shares with the
            # other constraint-satisfaction families.
            roles=("arrange", "relation", "position", "constrain"),
        )
        self._n_goals = len(world.goals)
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed the agree→cost model from skills won on similar tasks.

        What transfers is the *shape* of the affordance — "placements
        satisfying more decidable goals cost less" — not which thing
        went where. Names and positions are bindings.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                cost = step.parameters.get("mean_cost")
                if step.family == "placement":
                    agrees = step.parameters.get("agrees")
                    if isinstance(agrees, int) and isinstance(
                        cost, int | float
                    ):
                        adopted = True
                        prev = self._agree_priors.get(agrees)
                        self._agree_priors[agrees] = (
                            float(cost)
                            if prev is None
                            else min(prev, float(cost))
                        )
                # Any family that publishes a normalized "support"
                # fraction speaks the shared evidence language —
                # agrees / n_goals is the same axis.
                support = step.parameters.get("support")
                if isinstance(support, int | float) and isinstance(
                    cost, int | float
                ):
                    adopted = True
                    key = round(max(float(support), 0.0) * 4) / 4
                    prev = self._evidence_priors.get(key)
                    self._evidence_priors[key] = (
                        float(cost)
                        if prev is None
                        else min(prev, float(cost))
                    )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _transition_state(self, world: Arrangement) -> dict[str, Any]:
        return {
            "things.placed": len(world.board),
            "things.pool": len(world.pool),
            "goals.unsatisfied": world.mismatches(),
        }

    # ── Action selection ──────────────────────────────────────

    def _placement_score(self, agrees: int) -> float:
        """Expected quality — negative cost, so higher is better.
        Local observations beat adopted priors."""
        st = self._agree_stats.get(agrees)
        if st is not None and st.attempts:
            return -st.mean_cost
        prior = self._agree_priors.get(agrees)
        if prior is not None:
            return -prior
        if self._n_goals:
            evidence = self._evidence_priors.get(
                round((agrees / self._n_goals) * 4) / 4
            )
            if evidence is not None:
                return -evidence
        return 0.0

    @staticmethod
    def _pair_key(a: _Config, b: _Config) -> _Pair:
        """Canonical ordering so (A,B) and (B,A) are one memory."""
        return (a, b) if a <= b else (b, a)

    def _config_penalty(
        self, c: Placement, world: Arrangement
    ) -> float:
        """Penalty of this candidate: every bad pairing it would
        recreate with things already on the board."""
        me = (c.thing, c.pos)
        penalty = 0
        for pos, (other_name, _) in self._placed_as.items():
            if pos not in world.board:
                continue
            key = self._pair_key(me, (other_name, pos))
            penalty += self._pair_penalty.get(key, 0)
        return float(penalty)

    def _config_suspicion(self, config: _Config) -> int:
        """Lifetime failure count of a placed config — every violated
        pairing it took part in."""
        return sum(
            count
            for key, count in self._pair_penalty.items()
            if config in key
        )

    def _choose_placement(
        self, candidates: list[Placement], world: Arrangement
    ) -> Placement:
        def score(c: Placement) -> float:
            # ``breaks`` is a certain cost the world exposes before the
            # move — no learning needed to avoid it, only to weight
            # ``agrees`` against priors and pair memory.
            return (
                self._placement_score(c.agrees)
                - 1.0 * c.breaks
                - 0.75 * self._config_penalty(c, world)
            )

        learned = self._agree_priors or any(
            st.attempts for st in self._agree_stats.values()
        )
        if not learned or self._rng.random() < self.epsilon:
            # Exploration still prefers least-bad candidates — ones
            # neither penalized by memory nor observably breaking a
            # stated goal.
            def badness(c: Placement) -> float:
                return c.breaks + self._config_penalty(c, world)

            least = min(badness(c) for c in candidates)
            fresh = [c for c in candidates if badness(c) == least]
            return self._rng.choice(fresh)
        pairs = [(score(c), c) for c in candidates]
        best = max(s for s, _ in pairs)
        return self._rng.choice([c for s, c in pairs if s == best])

    # ── Episode ───────────────────────────────────────────────

    def step(self, world: Arrangement) -> str:
        """One action: repair a violated goal or place a thing."""
        context = self._recognize(world)
        before = self._transition_state(world)
        violations = world.violations()
        if violations:
            # Record every violated goal as a *pair* failure before
            # acting — the memory applies no matter which side moves.
            involvement: dict[_Config, int] = {}
            for goal in violations:
                configs: list[_Config] = []
                for pos, (name, _) in self._placed_as.items():
                    if name in (goal.a, goal.b):
                        configs.append((name, pos))
                if len(configs) == 2:
                    key = self._pair_key(configs[0], configs[1])
                    self._pair_penalty[key] = (
                        self._pair_penalty.get(key, 0) + 1
                    )
                for config in configs:
                    involvement[config] = involvement.get(config, 0) + 1
            if not involvement:
                return self._finish(world, "STUCK")
            sig = frozenset(violations)
            if sig == self._viol_sig:
                self._viol_repeat += 1
            else:
                self._viol_sig, self._viol_repeat = sig, 1
            if self._viol_repeat >= 3:
                # The same goals keep failing while undoing only their
                # parties — the deadlock is in *positions held by
                # bystanders*. Lift an uninvolved thing to reopen the
                # board (a real repair move: "maybe this one has to
                # move too"), then fall through to normal repair.
                involved_pos = {pos for _, pos in involvement}
                bystanders = [
                    p
                    for p in self._placed_as
                    if p not in involved_pos
                ]
                if bystanders:
                    pos = self._rng.choice(bystanders)
                    self._placed_as.pop(pos, None)
                    if world.remove(pos) is not None:
                        self._viol_repeat = 0
                        action = "remove"
                        self._removes_used += 1
                        self.task_competence.record_transition(
                            context,
                            action,
                            before,
                            self._transition_state(world),
                        )
                        return "PLAYING"
            # First try "switch these two": a swap is the only repair
            # that changes a *full* row, where lifting a party just
            # frees the same slot it came from. Descend on the honest
            # violation count, preferring to move entrenched mistakes
            # and jittered on ties so equal swaps don't 2-cycle.
            n_viol = len(violations)
            swap_choice: tuple[int, int] | None = None
            swap_key: tuple[float, float] | None = None
            for name, pos in involvement:
                for pos_b in world.board:
                    if pos_b == pos:
                        continue
                    hypo = dict(world.board)
                    hypo[pos], hypo[pos_b] = hypo[pos_b], hypo[pos]
                    after = sum(
                        1
                        for g in world.goals
                        if world.goal_state_of(g, hypo) == "no"
                    )
                    if after >= n_viol:
                        continue
                    skey = (
                        float(after),
                        -self._config_suspicion((name, pos))
                        + self._rng.random(),
                    )
                    if swap_key is None or skey < swap_key:
                        swap_key, swap_choice = skey, (pos, pos_b)
            if swap_choice is not None:
                pa, pb = swap_choice
                name_a = world.board[pa].name
                name_b = world.board[pb].name
                if world.swap(pa, pb) is None:
                    return self._finish(world, "STUCK")
                step_a = self._placed_as.get(pa, (name_a, -1))[1]
                step_b = self._placed_as.get(pb, (name_b, -1))[1]
                self._placed_as[pa] = (name_b, step_b)
                self._placed_as[pb] = (name_a, step_a)
                action, self._swaps_used = "swap", self._swaps_used + 1
            else:
                # No swap helps — a local minimum. Lift the config
                # breaking the most stated goals right now, preferring
                # the entrenched mistake; jitter breaks cycles.
                _, pos = max(
                    involvement,
                    key=lambda c: (
                        involvement[c] + self._config_suspicion(c)
                        + self._rng.random()
                    ),
                )
                self._placed_as.pop(pos, None)
                if world.remove(pos) is None:
                    return self._finish(world, "STUCK")
                action = "remove"
                self._removes_used += 1
        else:
            candidates = world.candidates()
            if not candidates:
                return self._finish(world, "STUCK")
            chosen = self._choose_placement(candidates, world)
            outcome = world.place(chosen.thing, chosen.pos)
            if outcome is None:
                return self._finish(world, "STUCK")
            action = "place"
            self._placed_as[chosen.pos] = (chosen.thing, self._clock)
            self._clock += 1
            st = self._agree_stats.setdefault(
                chosen.agrees, _AgreeStats()
            )
            st.attempts += 1
            # Cost = violations created (delta beyond the guaranteed
            # -1 from filling a spot).
            st.total_cost += max(
                0.0, float(outcome["delta_unsatisfied"]) + 1.0
            )
        self.task_competence.record_transition(
            context, action, before, self._transition_state(world)
        )
        if world.complete():
            return self._finish(world, "WIN")
        return "PLAYING"

    def solve(
        self, world: Arrangement, max_steps: int = 500
    ) -> RelationsResult:
        """Drive one arrangement to completion or exhaustion."""
        # Episode-local memory resets; learned stats and priors persist.
        self._pair_penalty.clear()
        self._placed_as.clear()
        self._viol_sig = None
        self._viol_repeat = 0
        swaps0 = self._swaps_used
        steps = placements = removals = 0
        state = "PLAYING"
        while state == "PLAYING" and steps < max_steps:
            before_board = len(world.board)
            state = self.step(world)
            steps += 1
            if len(world.board) > before_board:
                placements += 1
            elif len(world.board) < before_board:
                removals += 1
        if state == "PLAYING":
            state = self._finish(world, "STUCK")
        return RelationsResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            placements=placements,
            removals=removals,
            swaps=self._swaps_used - swaps0,
        )

    def _finish(self, world: Arrangement, state: str) -> str:
        del world
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a verified-complete row
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
                    GoalCondition("goals.unsatisfied", "eq", 0),
                ],
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the agree→cost affordance map."""
        steps = []
        for agrees in sorted(self._agree_stats):
            st = self._agree_stats[agrees]
            if not st.attempts:
                continue
            steps.append(
                ProcedureStep(
                    action="place",
                    family="placement",
                    parameters={
                        "agrees": agrees,
                        "support": round(
                            agrees / self._n_goals, 3
                        )
                        if self._n_goals
                        else 0.0,
                        "mean_cost": round(st.mean_cost, 3),
                    },
                    description=(
                        f"place satisfying {agrees} goals "
                        f"(cost {st.mean_cost:+.2f})"
                    ),
                )
            )
        if self._removes_used or self._swaps_used:
            steps.append(
                ProcedureStep(
                    action="swap",
                    family="repair",
                    parameters={
                        "swaps": self._swaps_used,
                        "lifts": self._removes_used,
                    },
                    description="switch or lift things in violated goals",
                )
            )
        return steps

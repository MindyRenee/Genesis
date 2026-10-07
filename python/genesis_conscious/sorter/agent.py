"""Sorter agent — solves matching tasks through task competence.

Same loop as ``AssemblyAgent``, different world. The agent is not told
that apertures demand *every* listed attribute: it observes candidates
(each carrying a perceptible ``matched`` count), tries them, and
learns the match→acceptance affordance empirically. The lid is in the
candidate set like everything else — a move that always "works" but
fills nothing, so it gets tried early and learned against, the same
mistake toddlers make. A solved board consolidates the learned
match→cost map as a procedure; the next instance starts with that
prior — including "free actions that don't fill slots are worthless."
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
from .puzzle import Insertion, ShapeSorter

# Stats buckets: inserts key on their matched *fraction* (matched of
# needed constraints) — "satisfy every constraint" is the invariant
# that transfers across sorters, not a raw count that shifts with
# difficulty. The lid keeps its own bucket so "always accepts, never
# fills" is learned like any other affordance rather than special-cased.
_LID_BUCKET = -1.0


@dataclass
class _MatchStats:
    """Observed cost of moves at one matched level."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class SorterResult:
    """Outcome of one sorter episode."""

    solved: bool
    state: str  # "WIN" | "STUCK" | "INTERRUPTED"
    steps: int
    placed: int
    wasted: int
    lid_uses: int


class SorterAgent:
    """Trial-and-error matching solver on the competence substrate."""

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
        # Learned affordance: a candidate satisfying this fraction of
        # the slot's constraints costs this much on average (1 =
        # wasted move, 0 = filled the aperture).
        self._match_stats: dict[float, _MatchStats] = {}
        # Schema-level prior adopted from skills: fraction -> cost.
        self._match_priors: dict[float, float] = {}
        # Cross-domain affordance: foreign skills that publish a
        # normalized "support" -> cost curve seed priors on the
        # quarter grid. Local stats and native priors beat these.
        self._evidence_priors: dict[float, float] = {}
        # Move memory, episode-local: a rejection is a property of the
        # *pair* (block, slot) — the same block may fit a different
        # aperture — and block ids mean nothing across instances, so
        # penalties reset per solve. What persists is the affordance
        # model (_match_stats) and the competence skills.
        self._penalty: dict[tuple[Any, ...], int] = {}
        self._lid_uses = 0

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

    def _sorter_features(self, sorter: ShapeSorter) -> dict[str, Any]:
        """World-observable structure for schema matching."""
        arity = (
            sum(s.needed for s in sorter.slots) / len(sorter.slots)
            if sorter.slots
            else 0.0
        )
        attrs = {
            attr for s in sorter.slots for attr, _ in s.accepts
        }
        return {
            "sorter.fit": True,
            "slots.count": self._bucket(len(sorter.slots)),
            "blocks.count": self._bucket(
                len(sorter.pool) + len(sorter.board)
            ),
            "slots.unfilled": self._bucket(sorter.mismatches()),
            "constraints.arity": self._bucket(round(arity)),
            "attrs.variety": self._bucket(len(attrs)),
            "lid.present": sorter.has_lid,
        }

    def _recognize_sorter(self, sorter: ShapeSorter) -> TaskContext:
        """Bind this sorter to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="sorter.fit",
            state=self._sorter_features(sorter),
            actions=("insert", "lid", "empty_top"),
            goal_conditions=[GoalCondition("slots.unfilled", "eq", 0)],
            entities=("block", "aperture"),
            # "Select what satisfies every stated constraint" is the
            # family shape — a shape sorter, a color match, a
            # letter-to-family drop all share it. "constrain" names
            # the abstract role it shares with the other
            # constraint-satisfaction families.
            roles=("match", "select", "classify", "constrain"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed the match→cost model from skills won on similar tasks.

        What transfers is the *shape* of the affordance — "candidates
        satisfying more constraints cost less; free moves that fill
        nothing are waste" — not which block went where. Identities
        are bindings.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                cost = step.parameters.get("mean_cost")
                if step.family == "selection":
                    matched = step.parameters.get("matched")
                    if isinstance(matched, int | float) and isinstance(
                        cost, int | float
                    ):
                        adopted = True
                        prev = self._match_priors.get(matched)
                        self._match_priors[matched] = (
                            float(cost)
                            if prev is None
                            else min(prev, float(cost))
                        )
                # Any family that publishes a normalized "support"
                # fraction speaks the shared evidence language — the
                # sorter's matched fraction is the same axis.
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

    def _transition_state(self, sorter: ShapeSorter) -> dict[str, Any]:
        return sorter.state()

    # ── Action selection ──────────────────────────────────────

    def _move_score(self, matched: float) -> float:
        """Expected quality of a move — negative cost, so higher is
        better. Local observations beat adopted priors."""
        st = self._match_stats.get(matched)
        if st is not None and st.attempts:
            return -st.mean_cost
        prior = self._match_priors.get(matched)
        if prior is not None:
            return -prior
        evidence = self._evidence_priors.get(
            round(max(matched, 0.0) * 4) / 4
        )
        if evidence is not None:
            return -evidence
        return 0.0

    @staticmethod
    def _penalty_key(c: Insertion) -> tuple[Any, ...]:
        if c.action == "insert":
            return ("insert", c.block_id, c.slot)
        return (c.action, c.block_id)

    def _bucket_key(self, c: Insertion) -> float:
        """The affordance bucket: what fraction of the slot's
        constraints this move satisfies."""
        if c.action == "lid":
            return _LID_BUCKET
        return c.matched / c.needed if c.needed else 0.0

    def _penalty_of(self, c: Insertion) -> float:
        return float(self._penalty.get(self._penalty_key(c), 0))

    def _choose(self, candidates: list[Insertion]) -> Insertion:
        def score(c: Insertion) -> float:
            # Dumping the box is a known small cost — worth it only
            # when real moves look hopeless.
            if c.action == "empty_top":
                return -0.5
            return (
                self._move_score(self._bucket_key(c))
                - 0.75 * self._penalty_of(c)
                + self._rng.random() * 0.05
            )

        learned = self._match_priors or any(
            st.attempts for st in self._match_stats.values()
        )
        if not learned or self._rng.random() < self.epsilon:
            # Exploration still prefers untried moves — a pair that
            # already failed is less worth retrying than one never
            # tried.
            least = min(self._penalty_of(c) for c in candidates)
            fresh = [
                c for c in candidates if self._penalty_of(c) == least
            ]
            return self._rng.choice(fresh)
        scored = [(score(c), c) for c in candidates]
        best = max(s for s, _ in scored)
        return self._rng.choice([c for s, c in scored if s == best])

    # ── Episode ───────────────────────────────────────────────

    def step(self, sorter: ShapeSorter) -> str:
        """One move: insert, use the lid, or dump the box."""
        context = self._recognize_sorter(sorter)
        candidates = sorter.candidates()
        if not candidates:
            return self._finish(sorter, "STUCK")
        before = self._transition_state(sorter)
        chosen = self._choose(candidates)
        key = self._penalty_key(chosen)
        progress = False
        if chosen.action == "insert":
            outcome = sorter.insert(chosen.block_id, chosen.slot)
            if outcome is None:
                return self._finish(sorter, "STUCK")
            progress = bool(outcome["accepted"])
            # Learn under the evidence the agent actually perceived —
            # for a symbolic sorter that's the oracle's matched count;
            # for a perceptual one it's the seen similarity bucket.
            st = self._match_stats.setdefault(
                self._bucket_key(chosen), _MatchStats()
            )
            st.attempts += 1
            st.total_cost += 0.0 if progress else 1.0
            action = "insert"
        elif chosen.action == "lid":
            outcome = sorter.insert_top(chosen.block_id)
            if outcome is None:
                return self._finish(sorter, "STUCK")
            st = self._match_stats.setdefault(_LID_BUCKET, _MatchStats())
            st.attempts += 1
            st.total_cost += 1.0
            self._lid_uses += 1
            action = "lid"
        else:
            outcome = sorter.empty_top()
            if outcome is None:
                return self._finish(sorter, "STUCK")
            action = "empty_top"
        if not progress:
            self._penalty[key] = self._penalty.get(key, 0) + 1
        self.task_competence.record_transition(
            context, action, before, self._transition_state(sorter)
        )
        if sorter.complete():
            return self._finish(sorter, "WIN")
        return "PLAYING"

    def solve(
        self, sorter: ShapeSorter, max_steps: int = 500
    ) -> SorterResult:
        """Drive one sorter to completion or exhaustion."""
        self._penalty.clear()
        self._lid_uses = 0
        steps = 0
        wasted = 0
        state = "PLAYING"
        last_unfilled = sorter.mismatches()
        stall = 0
        while state == "PLAYING" and steps < max_steps:
            state = self.step(sorter)
            steps += 1
            unfilled = sorter.mismatches()
            if unfilled < last_unfilled:
                stall = 0
            else:
                stall += 1
                wasted += 1
            last_unfilled = unfilled
            # Stuck means "no move available" (handled in step) — but
            # a long plateau of wasted moves is exhaustion too.
            if stall > max(20, 3 * len(sorter.slots)):
                state = self._finish(sorter, "STUCK")
        if state == "PLAYING":
            state = self._finish(sorter, "STUCK")
        return SorterResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            placed=len(sorter.board),
            wasted=wasted,
            lid_uses=self._lid_uses,
        )

    def _finish(self, sorter: ShapeSorter, state: str) -> str:
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a verified-full board consolidates
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
                    GoalCondition("slots.unfilled", "eq", 0)
                ],
                # The sorter's own constraint check is the world
                # reporting the outcome.
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the match→cost affordance map."""
        steps = []
        for matched in sorted(self._match_stats):
            st = self._match_stats[matched]
            if not st.attempts:
                continue
            if matched == _LID_BUCKET:
                steps.append(
                    ProcedureStep(
                        action="lid",
                        family="selection",
                        parameters={
                            "matched": matched,
                            "support": 0.0,
                            "mean_cost": round(st.mean_cost, 3),
                        },
                        description=(
                            "lid accepts anything but fills nothing "
                            f"(cost {st.mean_cost:+.2f})"
                        ),
                    )
                )
            else:
                steps.append(
                    ProcedureStep(
                        action="insert",
                        family="selection",
                        parameters={
                            "matched": matched,
                            "support": round(max(matched, 0.0), 3),
                            "mean_cost": round(st.mean_cost, 3),
                        },
                        description=(
                            f"insert matching {matched:.0%} of "
                            f"constraints (cost {st.mean_cost:+.2f})"
                        ),
                    )
                )
        return steps

    def describe_policy(self) -> str:
        """Compact description of the learned affordance — telemetry,
        not speech."""
        parts = []
        for matched in sorted(self._match_stats):
            st = self._match_stats[matched]
            if not st.attempts:
                continue
            label = (
                "lid" if matched == _LID_BUCKET else f"matched{matched:.0%}"
            )
            parts.append(f"{label}→cost{st.mean_cost:.2f}")
        return "; ".join(parts)


class PerceptualSorterAgent(SorterAgent):
    """The sorter by sight, not by label.

    A ``PerceptualSorter`` publishes no oracle ``matched`` on its
    candidates — the agent gets the block's view and the aperture's
    view through the visual cortex, and the affordance it compiles is
    "seen similarity → cost" instead of "stated attrs matched →
    cost". Same machinery, perceptual evidence: what transfers to
    other constraint tasks is still the shape of the curve, but the
    curve is now learned from what she saw.
    """

    def _recognize_sorter(self, sorter: ShapeSorter) -> TaskContext:
        # The world reference is needed inside _bucket_key for the
        # seen-similarity estimate.
        self._sorter_world = sorter
        return super()._recognize_sorter(sorter)

    def _bucket_key(self, c: Insertion) -> float:
        if c.action != "insert":
            return super()._bucket_key(c)
        world = getattr(self, "_sorter_world", None)
        if world is None or not hasattr(world, "similarity"):
            return 0.0
        # Concept-level evidence first: when both views resolve to
        # names the cortex has been taught, the match judgment is
        # "same named shape" — what a child does once "square" means
        # something. Otherwise the raw seen-similarity stands.
        b_name = world.recognize("block", c.block_id)
        s_name = world.recognize("slot", c.slot)
        if b_name and s_name:
            return 1.0 if b_name == s_name else 0.0
        sim = world.similarity(c.block_id, c.slot)
        return round(max(0.0, min(1.0, sim)) * 4) / 4

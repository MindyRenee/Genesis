"""Assembly agent — solves piece puzzles through task competence.

Same loop as ``SpatialAgent``, different world. The agent is not told
that matching edges is the goal: it observes candidate placements
(each carries a perceptible ``fits`` count — how many edge constraints
it satisfies), tries them, and learns the placement→cost affordance
empirically. A solved puzzle consolidates the learned fit→cost map as
a procedure; the next instance of the family starts with that prior
instead of exploring blind.
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
from .puzzle import PiecePuzzle, Placement


@dataclass
class _FitStats:
    """Observed cost of placements at one fits level."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class AssemblyResult:
    """Outcome of one puzzle attempt."""

    solved: bool
    state: str  # "WIN" or "STUCK" (budget/nothing-left)
    steps: int
    placements: int
    removals: int


class AssemblyAgent:
    """Trial-and-error assembly solver on the competence substrate."""

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
        # Learned affordance: placements satisfying `fits` constraints
        # cost this much on average (Δmismatches; good = 0).
        self._fit_stats: dict[int, _FitStats] = {}
        # Schema-level prior adopted from skills: fits -> prior cost.
        self._fit_priors: dict[int, float] = {}
        # Placement memory, dyadic: a mismatch is a property of the
        # *pair* of neighboring configs, not a single piece — blaming
        # one piece either protects an entrenched mistake or punishes
        # an innocent. Pair penalties mark the bad combination, so the
        # correct config stays usable against other neighbors. Unary
        # penalties cover border violations (one piece's fault alone).
        self._place_penalty: dict[tuple[int, int, int, int], int] = {}
        self._pair_penalty: dict[
            tuple[
                tuple[int, int, int, int],
                tuple[int, int, int, int],
            ],
            int,
        ] = {}
        # slot -> (piece_id, turns, step placed). Needed to name the
        # configs on both sides of a broken edge, and for recency
        # tiebreaks when choosing which side to lift.
        self._placed_as: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._clock = 0
        self._removes_used = 0

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

    def _puzzle_features(self, puzzle: PiecePuzzle) -> dict[str, Any]:
        """World-observable structure for schema matching."""
        codes = {
            code for p in self._all_pieces(puzzle) for code in p.edges
        }
        return {
            "puzzle.assembly": True,
            "board.size": self._bucket(puzzle.rows * puzzle.cols),
            "pieces.count": self._bucket(len(puzzle.pool) + len(puzzle.board)),
            "pieces.unplaced": self._bucket(len(puzzle.pool)),
            "board.placed": self._bucket(len(puzzle.board)),
            "board.mismatches": self._bucket(puzzle.mismatches()),
            "edges.variety": self._bucket(len(codes)),
        }

    @staticmethod
    def _all_pieces(puzzle: PiecePuzzle) -> list:
        return list(puzzle.pool.values()) + list(puzzle.board.values())

    def _recognize_puzzle(self, puzzle: PiecePuzzle) -> TaskContext:
        """Bind this puzzle to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="assembly.puzzle",
            state=self._puzzle_features(puzzle),
            actions=("place", "remove"),
            goal_conditions=[
                GoalCondition("board.mismatches", "eq", 0),
                GoalCondition("pieces.unplaced", "eq", 0),
            ],
            entities=("piece", "slot"),
            # "Assemble pieces into a constraint-satisfying whole" is
            # the family shape — a jigsaw, a tile mosaic, or a parts
            # kit share it.
            roles=("assemble", "object", "spatial"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed the fit→cost model from skills won on similar puzzles.

        What transfers is the *shape* of the affordance — "placements
        satisfying more edge constraints cost less" — not which piece
        went where. Piece identities and positions are bindings.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                if step.family != "placement":
                    continue
                fits = step.parameters.get("fits")
                cost = step.parameters.get("mean_cost")
                if isinstance(fits, int) and isinstance(
                    cost, int | float
                ):
                    adopted = True
                    prev = self._fit_priors.get(fits)
                    self._fit_priors[fits] = (
                        float(cost)
                        if prev is None
                        else min(prev, float(cost))
                    )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _transition_state(self, puzzle: PiecePuzzle) -> dict[str, Any]:
        return {
            "board.placed": len(puzzle.board),
            "board.mismatches": puzzle.mismatches(),
            "pieces.unplaced": len(puzzle.pool),
        }

    # ── Action selection ──────────────────────────────────────

    def _placement_score(self, fits: int) -> float:
        """Expected quality of a placement — negative cost, so higher
        is better. Local observations beat adopted priors."""
        st = self._fit_stats.get(fits)
        if st is not None and st.attempts:
            return -st.mean_cost
        prior = self._fit_priors.get(fits)
        if prior is not None:
            return -prior
        return 0.0

    @staticmethod
    def _pair_key(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int]
    ) -> tuple[
        tuple[int, int, int, int], tuple[int, int, int, int]
    ]:
        """Canonical ordering so (A,B) and (B,A) are one memory."""
        return (a, b) if a <= b else (b, a)

    def _config_penalty(
        self, c: Placement, puzzle: PiecePuzzle
    ) -> float:
        """How penalized this candidate is: its own unary record plus
        every bad pairing it would recreate with occupied neighbors."""
        me = (c.piece_id, c.row, c.col, c.turns)
        penalty = self._place_penalty.get(me, 0)
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            slot = (c.row + dr, c.col + dc)
            other = self._placed_as.get(slot)
            if other is None:
                continue
            key = self._pair_key(
                me, (other[0], slot[0], slot[1], other[1])
            )
            penalty += self._pair_penalty.get(key, 0)
        return float(penalty)

    def _slot_suspicion(self, slot: tuple[int, int]) -> int:
        """Lifetime failure count of the config sitting at `slot` —
        every penalized pair and border violation it took part in."""
        placed = self._placed_as.get(slot)
        if placed is None:
            return 0
        me = (placed[0], slot[0], slot[1], placed[1])
        suspicion = self._place_penalty.get(me, 0)
        for key, count in self._pair_penalty.items():
            if me in key:
                suspicion += count
        return suspicion

    def _choose_placement(
        self, candidates: list[Placement], puzzle: PiecePuzzle
    ) -> Placement:
        def score(c: Placement) -> float:
            return (
                self._placement_score(c.fits)
                - 0.75 * self._config_penalty(c, puzzle)
            )

        learned = self._fit_priors or any(
            st.attempts for st in self._fit_stats.values()
        )
        if not learned or self._rng.random() < self.epsilon:
            # Exploration still prefers unpenalized configs — a
            # combination that already failed is less worth retrying
            # than one never tried.
            least = min(
                self._config_penalty(c, puzzle) for c in candidates
            )
            fresh = [
                c
                for c in candidates
                if self._config_penalty(c, puzzle) == least
            ]
            return self._rng.choice(fresh)
        best = max(score(c) for c in candidates)
        return self._rng.choice(
            [c for c in candidates if score(c) == best]
        )

    # ── Episode ───────────────────────────────────────────────

    def step(self, puzzle: PiecePuzzle) -> str:
        """One action: repair a mismatch or place a piece."""
        context = self._recognize_puzzle(puzzle)
        before = self._transition_state(puzzle)
        if puzzle.mismatches() > 0:
            pairs, unary = puzzle.violations()
            # Record every broken constraint *before* lifting — the
            # memory is the pair, so it applies no matter which side
            # gets lifted.
            for sa, sb in pairs:
                ca, cb = self._placed_as.get(sa), self._placed_as.get(sb)
                if ca is None or cb is None:
                    continue
                key = self._pair_key(
                    (ca[0], sa[0], sa[1], ca[1]),
                    (cb[0], sb[0], sb[1], cb[1]),
                )
                self._pair_penalty[key] = self._pair_penalty.get(key, 0) + 1
            for slot_u in unary:
                cu = self._placed_as.get(slot_u)
                if cu is None:
                    continue
                key_u = (cu[0], slot_u[0], slot_u[1], cu[1])
                self._place_penalty[key_u] = (
                    self._place_penalty.get(key_u, 0) + 1
                )
            involvement: dict[tuple[int, int], int] = {}
            for sa, sb in pairs:
                involvement[sa] = involvement.get(sa, 0) + 1
                involvement[sb] = involvement.get(sb, 0) + 1
            for slot_u in unary:
                involvement[slot_u] = involvement.get(slot_u, 0) + 1
            # Lift by *lifetime* suspicion: a config that has failed
            # against many different neighbors is the entrenched
            # mistake — blaming only today's violation protects it
            # forever (each newcomer takes the fall). A correct piece
            # can accrue suspicion when a stream of wrong neighbors
            # fails against it, but lifting it costs progress, not
            # correctness — pair penalties still guard the retry.
            slot = max(
                involvement,
                key=lambda s: (
                    self._slot_suspicion(s),
                    self._placed_as.get(s, (-1, -1, -1))[2],
                ),
            )
            self._placed_as.pop(slot, None)
            if puzzle.remove(*slot) is None:
                return self._finish(puzzle, "STUCK")
            action, self._removes_used = "remove", self._removes_used + 1
        else:
            candidates = puzzle.candidates()
            if not candidates:
                return self._finish(puzzle, "STUCK")
            chosen = self._choose_placement(candidates, puzzle)
            outcome = puzzle.place(
                chosen.piece_id, chosen.row, chosen.col, chosen.turns
            )
            if outcome is None:
                return self._finish(puzzle, "STUCK")
            action = "place"
            self._placed_as[(chosen.row, chosen.col)] = (
                chosen.piece_id, chosen.turns, self._clock
            )
            self._clock += 1
            st = self._fit_stats.setdefault(
                int(outcome["fits"]), _FitStats()
            )
            st.attempts += 1
            st.total_cost += float(outcome["delta_mismatches"])
        self.task_competence.record_transition(
            context, action, before, self._transition_state(puzzle)
        )
        if puzzle.complete():
            return self._finish(puzzle, "WIN")
        return "PLAYING"

    def solve(
        self, puzzle: PiecePuzzle, max_steps: int = 500
    ) -> AssemblyResult:
        """Drive one puzzle to completion or exhaustion."""
        steps = placements = removals = 0
        state = "PLAYING"
        while state == "PLAYING" and steps < max_steps:
            before_board = len(puzzle.board)
            state = self.step(puzzle)
            steps += 1
            if len(puzzle.board) > before_board:
                placements += 1
            elif len(puzzle.board) < before_board:
                removals += 1
        if state == "PLAYING":
            state = self._finish(puzzle, "STUCK")
        return AssemblyResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            placements=placements,
            removals=removals,
        )

    def _finish(self, puzzle: PiecePuzzle, state: str) -> str:
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a verified-complete board
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
                    GoalCondition("board.mismatches", "eq", 0),
                    GoalCondition("pieces.unplaced", "eq", 0),
                ],
                # The board's own constraint check is the world
                # reporting the outcome.
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the fit→cost affordance map."""
        steps = []
        for fits in sorted(self._fit_stats):
            st = self._fit_stats[fits]
            if not st.attempts:
                continue
            steps.append(
                ProcedureStep(
                    action="place",
                    family="placement",
                    parameters={
                        "fits": fits,
                        "mean_cost": round(st.mean_cost, 3),
                    },
                    description=(
                        f"place satisfying {fits} constraints "
                        f"(cost {st.mean_cost:+.2f})"
                    ),
                )
            )
        if self._removes_used:
            steps.append(
                ProcedureStep(
                    action="remove",
                    family="repair",
                    parameters={"used": True},
                    description="lift mismatched pieces to retry",
                )
            )
        return steps

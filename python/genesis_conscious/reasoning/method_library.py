"""Method library — reusable problem-solving methods learned during sleep.

Modeled on DreamCoder's wake–sleep cycle (Ellis et al., 2023): during
wake the solver produces verified solutions; during sleep the library
mines them for recurring structure and *compresses* the shared pattern
into a Method — a content-free strategy record: which operators and
which relation types sufficed, for which goal type, across how many
distinct problems. During the next wake, the library acts as the
recognition model: methods matching the problem's goal type propose a
replay (for UNDERSTAND goals, gathering the recorded relations is often
enough to verify without the expensive operators), and every replay
outcome updates the method's reliability, so the prior over strategies
is calibrated by observation, not by construction.

Honesty constraints (shared with the solver):

- A method can never manufacture success — replay still goes through
  ``ProblemSolver._verify``, which recomputes the goal criterion from
  the live concept network. The method decides *where to look*, never
  *what is true*.
- ``support`` counts distinct goals the signature verified on —
  compression evidence, the same way shared program fragments earn
  abstraction in DreamCoder's refactoring pass.
- ``successes``/``failures`` count observed replay outcomes — the
  recognition model is trained on real wake results, and a method
  whose reliability decays is archived rather than trusted forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .problem_solving import Problem, Solution

__all__ = ["Method", "MethodLibrary"]

logger = logging.getLogger(__name__)

#: A signature must verify on at least this many *distinct* goals
#: before it compresses into a method — one success is a fact about
#: the problem, two is evidence about the strategy.
_MIN_SUPPORT = 2

#: Reliability floor for suggesting a replay — below this the method
#: is a hypothesis the wake solver should not follow.
_SUGGEST_FLOOR = 0.5

#: Archive threshold: a method tried this many times and still under
#: the floor is counterevidence, not a strategy. Kept small — the
#: suggest floor already stops offering a method once its posterior
#: sinks, so only a few failures are ever observable.
_ARCHIVE_MIN_USES = 2
_ARCHIVE_FLOOR = 0.35

#: Bound on the live library — oldest, least-reliable methods are
#: dropped first when it overflows.
_MAX_METHODS = 128

#: Half-life for reliability prior strength — one consolidation bout
#: of reinforcement adds this much pseudo-observation mass.
_PRIOR_PER_SUPPORT = 0.5

# Knowledge labels produced by the solver's operators, mapped back to
# the (relation, direction) edge query that regenerates them. Replay
# re-derives content from the live network through this map — stored
# targets are never trusted. Public: the solver's replay consults it.
EDGE_FOR_LABEL = {
    "depends_on": ("depends_on", "out"),
    "enabled_by": ("enables", "in"),
    "has_property": ("has_property", "out"),
    "part_of": ("part_of", "out"),
    "is_a": ("is_a", "out"),
    "caused_by": ("causes", "in"),
    "causes": ("causes", "out"),
    "leads_to": ("leads_to", "out"),
    "led_to_by": ("leads_to", "in"),
    "similar_to": ("similar_to", "both"),
    "related_to": ("related_to", "both"),
    "contradicts": ("contradicts", "both"),
    "emerges_from": ("emerges_from", "out"),
}


@dataclass
class Method:
    """A compressed problem-solving method — a strategy schema.

    The signature (goal type + operator set + relation set) is what
    all provenance solutions shared after their entity content was
    abstracted away. Reliability is Bayesian: ``support`` is the prior
    mass from compression, ``successes``/``failures`` are observed
    wake-time replay outcomes.
    """

    method_id: str
    goal_type: str
    operators: tuple[str, ...]
    relations: tuple[str, ...]
    subgoal_types: tuple[str, ...] = ()
    support: int = 0
    successes: int = 0
    failures: int = 0
    provenance: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    last_used: float = 0.0
    archived: bool = False

    @property
    def uses(self) -> int:
        """Observed replay attempts."""
        return self.successes + self.failures

    @property
    def reliability(self) -> float:
        """Posterior success estimate: compression prior updated by
        observed replays. Unused methods start near 0.6 (support of 2);
        observed outcomes quickly dominate."""
        prior = self.support * _PRIOR_PER_SUPPORT
        total = self.uses + prior
        if total <= 0:
            return 0.0
        return (self.successes + prior * 0.6) / total


class MethodLibrary:
    """Wake–sleep method learning for the problem solver.

    Wake role (recognition model): ``suggest`` returns the best method
    for a problem's goal type; the solver replays it and reports the
    observed outcome back through ``observe_use``.

    Sleep role (abstraction): ``consolidate`` mines the day's verified
    solutions for signatures shared across distinct goals and folds
    them into methods — the same compression move as DreamCoder's
    library refactoring, applied to strategy structure rather than
    program fragments.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or ""
        self.methods: dict[str, Method] = {}
        # How many entries of the solutions list were already mined —
        # solutions accumulate on the solver, consolidate consumes the
        # new tail each bout.
        self._seen = 0
        self._archive: list[Method] = []
        if self.path:
            self._load()

    # ─── Sleep: abstraction ─────────────────────────────────────

    def consolidate(self, solutions: list[Solution]) -> dict[str, int]:
        """Mine new verified solutions into methods.

        Groups by (goal type, productive operator set, subgoal types)
        and counts distinct goals; the method's relation set is the
        intersection of evidence channels that recurred across every
        supporting goal, restricted to replayable labels. Groups with
        enough distinct-goal support and a non-empty shared pattern
        become methods; existing methods gain support and narrow.
        Returns stats for the sleep log.
        """
        stats = {"mined": 0, "created": 0, "reinforced": 0, "archived": 0}
        # The solver's solution list doesn't survive restarts (only
        # counts persist) — a persisted _seen beyond the list would
        # silently stop all mining. Clamp; re-mining is idempotent
        # since provenance dedupes per method.
        if self._seen > len(solutions):
            self._seen = 0
        new = solutions[self._seen:]
        self._seen = len(solutions)

        # group key → {goal: relation-label set it gathered}
        grouped: dict[tuple, dict[str, frozenset]] = {}
        for sol in new:
            if not sol.verified or sol.verification_basis == "blocked":
                continue
            key = self._group_key(sol)
            if key is None:
                continue
            stats["mined"] += 1
            grouped.setdefault(key, {})[sol.problem.goal] = frozenset(
                k[0] for k in sol.all_knowledge
            )

        for (goal_type, operators, sub_types), rels_by_goal in grouped.items():
            mid = self._method_id((goal_type, operators, sub_types))
            # The reusable evidence pattern is what recurred across
            # *every* supporting goal — intersection, restricted to
            # channels the solver can actually re-derive at replay.
            common = (
                frozenset.intersection(*rels_by_goal.values())
                & EDGE_FOR_LABEL.keys()
            )
            method = self.methods.get(mid)
            if method is not None:
                # Reinforce: new distinct goals add compression mass,
                # and the evidence pattern narrows to what held again.
                fresh = set(rels_by_goal) - set(method.provenance)
                if fresh:
                    method.support += len(fresh)
                    method.provenance.extend(sorted(fresh))
                    narrowed = frozenset(method.relations) & common
                    if narrowed:
                        method.relations = tuple(sorted(narrowed))
                    stats["reinforced"] += 1
                continue
            if len(rels_by_goal) < _MIN_SUPPORT or not common:
                continue
            self.methods[mid] = Method(
                method_id=mid,
                goal_type=goal_type,
                operators=operators,
                relations=tuple(sorted(common)),
                subgoal_types=sub_types,
                support=len(rels_by_goal),
                provenance=sorted(rels_by_goal),
            )
            stats["created"] += 1

        stats["archived"] = self._prune()
        if stats["created"] or stats["reinforced"]:
            logger.info("method consolidation: %s", stats)
        return stats

    @staticmethod
    def _group_key(sol: Solution) -> tuple | None:
        """The content-free group key of a verified solution:
        (goal type, productive operator set, subgoal types).

        Only operators whose steps actually carried knowledge count —
        an operator that ran but found nothing is pipeline noise, not
        strategy. The relation labels are deliberately *not* in the
        key: which evidence channels recurred is decided by the
        intersection across supporting goals, so solutions differing
        by one incidental relation still compress. Trivial solutions
        (nothing gathered) carry no method signal.
        """
        operators = {
            s.operator
            for s in sol.all_steps
            if s.operator != "goal_check" and s.knowledge
        }
        sub_types = tuple(sorted(
            sub.problem.goal_type.value for sub in sol.sub_solutions
        ))
        if not operators:
            return None
        return (
            sol.problem.goal_type.value,
            tuple(sorted(operators)),
            sub_types,
        )

    @staticmethod
    def _method_id(signature: tuple) -> str:
        """Stable id from the signature — same signature, same method,
        across restarts and independent solutions."""
        digest = hashlib.sha1(repr(signature).encode()).hexdigest()[:10]
        return f"m_{signature[0]}_{digest}"

    def _prune(self) -> int:
        """Archive methods the wake solver has shown unreliable, and
        bound the library size by dropping the weakest."""
        archived = 0
        for mid, m in list(self.methods.items()):
            if m.uses >= _ARCHIVE_MIN_USES and m.reliability < _ARCHIVE_FLOOR:
                m.archived = True
                self._archive.append(m)
                del self.methods[mid]
                archived += 1
        if len(self.methods) > _MAX_METHODS:
            weakest = sorted(
                self.methods.values(),
                key=lambda m: (m.reliability, m.last_used),
            )
            for m in weakest[: len(self.methods) - _MAX_METHODS]:
                m.archived = True
                self._archive.append(m)
                del self.methods[m.method_id]
                archived += 1
        return archived

    # ─── Wake: recognition model ────────────────────────────────

    def suggest(self, problem: Problem) -> Method | None:
        """Best live method for this problem's goal type.

        Candidate match is on goal type plus relation compatibility:
        a method whose recorded relations cannot produce evidence for
        this goal type is not suggested (e.g. a pure property-gathering
        method is not offered for ACHIEVE, where only causal support
        verifies).
        """
        goal_type = problem.goal_type.value
        best: Method | None = None
        for m in self.methods.values():
            if m.archived or m.goal_type != goal_type:
                continue
            if not self._relations_can_serve(goal_type, m.relations):
                continue
            if m.reliability < _SUGGEST_FLOOR:
                continue
            if best is None or (
                m.reliability, m.support
            ) > (best.reliability, best.support):
                best = m
        return best

    @staticmethod
    def _relations_can_serve(
        goal_type: str, relations: tuple[str, ...]
    ) -> bool:
        """Whether the method's evidence channels can satisfy the goal.

        Mirrors the solver's verification criteria: ACHIEVE needs
        incoming causal support, EXPLAIN the same narrower, COMPARE
        needs path evidence, RESOLVE support evidence — a method that
        only gathered properties cannot verify those. Only replayable
        labels count — a method whose evidence can't be re-derived
        is never suggested.
        """
        rels = set(relations) & EDGE_FOR_LABEL.keys()
        if goal_type == "achieve":
            return bool(rels & {"caused_by", "led_to_by", "leads_to", "causes"})
        if goal_type == "explain":
            return bool(rels & {"caused_by"})
        if goal_type == "compare":
            return True  # verify is path-based; any gathering counts
        if goal_type == "resolve":
            return bool(rels)
        return bool(rels)  # understand: any real knowledge counts

    def observe_use(self, method_id: str, verified: bool) -> None:
        """Record a replay outcome — the recognition model's training
        signal. Verified replays raise reliability; failures lower it
        toward the archive floor."""
        m = self.methods.get(method_id)
        if m is None:
            return
        if verified:
            m.successes += 1
        else:
            m.failures += 1
        m.last_used = time.time()

    # ─── Introspection + persistence ────────────────────────────

    def stats(self) -> dict[str, int | float]:
        """Library snapshot for the sleep log and introspection."""
        live = [m for m in self.methods.values() if not m.archived]
        return {
            "methods": len(live),
            "archived": len(self._archive),
            "uses": sum(m.uses for m in live),
            "mean_reliability": (
                round(sum(m.reliability for m in live) / len(live), 3)
                if live else 0.0
            ),
            "solutions_mined": self._seen,
        }

    def save(self) -> None:
        """Persist the library atomically (tmp + rename)."""
        if not self.path:
            return
        payload = {
            "seen": self._seen,
            "methods": [
                {
                    "method_id": m.method_id,
                    "goal_type": m.goal_type,
                    "operators": list(m.operators),
                    "relations": list(m.relations),
                    "subgoal_types": list(m.subgoal_types),
                    "support": m.support,
                    "successes": m.successes,
                    "failures": m.failures,
                    "provenance": m.provenance[-32:],
                    "created": m.created,
                    "last_used": m.last_used,
                    "archived": m.archived,
                }
                for m in self.methods.values()
            ],
            "archive": [
                {
                    "method_id": m.method_id,
                    "goal_type": m.goal_type,
                    "operators": list(m.operators),
                    "relations": list(m.relations),
                    "support": m.support,
                    "successes": m.successes,
                    "failures": m.failures,
                }
                for m in self._archive[-64:]
            ],
        }
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.debug(f"method library save failed: {e}")

    def _load(self) -> None:
        """Restore a persisted library; corrupt files start empty."""
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._seen = int(data.get("seen", 0))
        for rec in data.get("methods", []):
            try:
                m = Method(
                    method_id=str(rec["method_id"]),
                    goal_type=str(rec["goal_type"]),
                    operators=tuple(rec.get("operators", ())),
                    relations=tuple(rec.get("relations", ())),
                    subgoal_types=tuple(rec.get("subgoal_types", ())),
                    support=int(rec.get("support", 0)),
                    successes=int(rec.get("successes", 0)),
                    failures=int(rec.get("failures", 0)),
                    provenance=[str(g) for g in rec.get("provenance", [])],
                    created=float(rec.get("created", time.time())),
                    last_used=float(rec.get("last_used", 0.0)),
                    archived=bool(rec.get("archived", False)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if m.archived:
                self._archive.append(m)
            else:
                self.methods[m.method_id] = m

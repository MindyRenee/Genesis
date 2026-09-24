"""Spatial reasoner — searching transformation space for the rule.

This is the perceptual counterpart to ``ReasoningEngine``: where the
semantic engine traverses typed edges between concepts, the spatial
engine composes grid transformations and verifies them against
evidence. Solving a task means finding a sequence of ``Transform``s
that reproduces every training output exactly — the same exact-match
verification discipline the concept-network problem solver uses.

Search is breadth-first with beam pruning, ranked by how many cells
the candidate output gets right across all training pairs (fractional
match), then by sequence length — Occam's razor: shorter explanations
are preferred when they tie. A hypothesis is only a *solution* when it
reproduces every training output exactly.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .grid import Grid
from .scene import Scene, perceive
from .transforms import Example, Transform, propose

logger = logging.getLogger(__name__)


@dataclass
class SpatialHypothesis:
    """A candidate rule: a sequence of transforms plus its evidence."""

    transforms: list[Transform]
    score: float  # mean cell-accuracy across training pairs, [0, 1]
    exact: int = 0  # number of training pairs reproduced exactly

    def describe(self) -> str:
        return " → ".join(t.describe() for t in self.transforms)

    def apply(self, grid: Grid) -> Grid:
        for t in self.transforms:
            grid = t(grid)
        return grid


@dataclass
class FailureInfo:
    """A counterexample report: why the best hypothesis failed.

    Produced on unsolved tasks so the caller can see *where* the best
    rule broke rather than getting an opaque miss — which training
    pair was the counterexample, whether the failure was a shape
    mismatch or cell-level disagreement, and which colors were
    confused (predicted → expected over the differing cells).
    """

    rule: str                 # describe() of the best hypothesis
    pair_scores: list[float]  # per-training-pair accuracy of that rule
    worst_pair: int           # index of the pair it fit worst
    shape_mismatch: bool      # predicted grid had the wrong shape
    mismatched: int           # differing cells on the worst pair
    confusion: dict[str, int] = field(default_factory=dict)
    # confusion keys are "predicted->expected" over differing cells

    def describe(self) -> str:
        """Compact one-line failure summary for memory and logs."""
        if self.shape_mismatch:
            return (
                f"{self.rule} broke on pair {self.worst_pair}: "
                f"wrong shape"
            )
        top = sorted(
            self.confusion.items(), key=lambda kv: -kv[1]
        )[:3]
        conf = ", ".join(f"{k}×{v}" for k, v in top)
        return (
            f"{self.rule} broke on pair {self.worst_pair}: "
            f"{self.mismatched} diffs"
            + (f" ({conf})" if conf else "")
        )


@dataclass
class SpatialSolution:
    """The result of solving a task.

    ``hypothesis`` is the best verified rule (None if nothing solved
    all training pairs). ``predictions`` are the hypothesis applied to
    each test input — empty when unsolved. ``hypotheses`` retains the
    runner-up candidates for inspection.
    """

    solved: bool
    hypothesis: SpatialHypothesis | None
    predictions: list[Grid] = field(default_factory=list)
    hypotheses: list[SpatialHypothesis] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    nodes_explored: int = 0  # candidates evaluated — the cost of finding
    # All prediction sets from the top-k verified hypotheses (multiple
    # guesses are kept when several hypotheses verify). predictions == guesses[0].
    guesses: list[list[Grid]] = field(default_factory=list)
    # describe() of every rule verified on all training pairs — when
    # the test guess misses, these are the proven counterexamples
    # (rules that fit the training data but generalized wrong).
    verified_rules: list[str] = field(default_factory=list)
    # Why the best hypothesis failed, when nothing verified.
    failure: FailureInfo | None = None


def _cell_accuracy(candidate: Grid, target: Grid) -> float:
    """Fraction of matching cells; 0 when shapes differ."""
    if candidate.shape != target.shape:
        return 0.0
    total = candidate.width * candidate.height
    same = sum(
        1
        for r, c, v in candidate.iter_cells()
        if v == target.at(r, c)
    )
    return same / total


class SpatialReasoner:
    """Few-shot spatial inference: examples → rule → prediction.

    The reasoner may optionally hold a ``ConceptNetwork`` so perceived
    scenes and verified rules can be grounded into conceptual structure
    (see ``grounding.py``). Grounding is a side channel — the search
    itself works on grids alone.
    """

    def __init__(self, network: Any | None = None) -> None:
        self.network = network
        # Rules it has learned — from its own successful solves or
        # from being taught the correct answer after a failure. These
        # are proposed *first* on future tasks: prior knowledge acts
        # as a prior over the hypothesis space, the way recalled
        # strategies bias human problem-solving.
        self._learned: list[Transform] = []
        # Per-family success counts: which kinds of transforms have
        # worked before. Biases the search toward experienced
        # families — recognition is cheaper than exploration.
        self._family_priors: Counter[str] = Counter()

    @staticmethod
    def _family(transform: Transform) -> str:
        """The transform's family: 'gravity_down_1' → 'gravity',
        'recolor_largest' → 'recolor'."""
        return transform.name.split("_")[0]

    def learn(
        self, transforms: list[Transform], *, taught: bool = False
    ) -> None:
        """Remember a rule for future tasks.

        Called automatically on successful solves, or externally to
        teach it the correct rule after it gets a task wrong. When a
        concept network is attached, each rule is also grounded as a
        ``spatial:rule:*`` concept so its semantic machinery (memory,
        language, analogy) can reason about what it knows.

        Args:
            taught: True when the rule comes from external correction
                rather than its own verified solve — grounded with
                origin ``teaching`` so provenance stays honest.
        """
        known = {t.describe() for t in self._learned}
        for t in transforms:
            self._family_priors[self._family(t)] += 1
            if t.describe() not in known:
                self._learned.append(t)
                known.add(t.describe())
                self._ground_rule(t, taught=taught)
        if len(transforms) > 1:
            # Multi-step solutions are also stored as atomic
            # macro-rules: replaying a verified sequence as one step
            # effectively doubles the depth future searches can reach.
            seq = list(transforms)

            def apply_seq(g: Grid, s: list[Transform] = seq) -> Grid:
                for t in s:
                    g = t(g)
                return g

            desc = " → ".join(t.describe() for t in seq)
            macro = Transform(f"macro:{desc}", apply_seq, {"rule": desc})
            self._family_priors["macro"] += 1
            if macro.describe() not in known:
                self._learned.append(macro)
                self._ground_rule(macro, taught=taught)

    @property
    def learned_rules(self) -> tuple[Transform, ...]:
        """Rules learned from verified solves, oldest first (read-only)."""
        return tuple(self._learned)

    @staticmethod
    def _schema_family(transform: Transform) -> str | None:
        """Map a learned transform to its re-parameterizable family."""
        name = transform.name
        if name.startswith("gravity_"):
            return "gravity"
        for fam in (
            "erase_color",
            "select_color",
            "fill_enclosed",
            "recolor_largest",
            "recolor_smallest",
        ):
            if name == fam:
                return fam
        return None

    def _schema_transforms(
        self, examples: list[Example]
    ) -> list[Transform]:
        """Induce schemas from learned rules and instantiate them.

        Grouping learned transforms by family and re-binding their
        parameters to values observed in the current task is the
        abstraction step memory alone lacks: one learned
        ``fill_enclosed(color=4)`` becomes "fill enclosed regions with
        *some* color", and the schema proposes every instantiation the
        task's colors suggest. Verification still decides — schemas
        only widen the proposal set.
        """
        from itertools import product

        from .transforms import instantiate

        groups: dict[str, list[Transform]] = {}
        for t in self._learned:
            fam = self._schema_family(t)
            if fam is not None:
                groups.setdefault(fam, []).append(t)
        if not groups:
            return []

        # Task-observed candidate values for free parameters.
        task_colors: set[int] = set()
        for inp, out in examples:
            task_colors |= set(inp.colors()) | set(out.colors())

        known = {t.describe() for t in self._learned}
        out_transforms: list[Transform] = []
        for fam, members in groups.items():
            keys = set().union(*(t.params for t in members))
            absent: object = object()
            options: list[list[tuple[str, object]]] = []
            for key in sorted(keys):
                holders = [t for t in members if key in t.params]
                seen = {t.params[key] for t in holders}
                if all(isinstance(v, int) for v in seen):
                    vals_i = sorted(
                        {v for v in seen if isinstance(v, int)}
                        | task_colors
                    )
                    values: list[object] = list(vals_i)
                elif all(isinstance(v, str) for v in seen):
                    vals_s = sorted(
                        {v for v in seen if isinstance(v, str)}
                        | {"down", "up", "left", "right"}
                    )
                    values = list(vals_s)
                else:
                    values = list(seen)
                opts: list[tuple[str, object]] = [
                    (key, v) for v in values
                ]
                # A param present in only some members is optional in
                # the schema (e.g. gravity learned both with and
                # without a color condition).
                if len(holders) < len(members):
                    opts.insert(0, (key, absent))
                options.append(opts)
            for combo in list(product(*options))[:64]:
                params = {k: v for k, v in combo if v is not absent}
                inst = instantiate(fam, params)
                if inst is not None and inst.describe() not in known:
                    out_transforms.append(inst)
        return out_transforms

    def _ground_rule(self, transform: Transform, *, taught: bool = False) -> None:
        """Write a learned rule into the concept network."""
        if self.network is None:
            return
        try:
            self.network.add_concept(
                f"spatial:rule:{transform.describe()}",
                origin="teaching" if taught else "perception",
                properties={
                    "kind": "spatial_rule",
                    "transform": transform.name,
                    # Values are stringified: params can hold nested
                    # dicts with int keys (e.g. recolor maps), which
                    # break the daemon's state serialization.
                    "params": {
                        k: str(v) for k, v in transform.params.items()
                    },
                },
            )
        except Exception as e:  # noqa: BLE001
            # Grounding is best-effort; never break a solve.
            logger.debug("rule grounding failed: %s", e)

    def perceive(self, grid: Grid, background: int | None = None) -> Scene:
        """Segment a grid into objects and spatial relations."""
        return perceive(grid, background=background)

    def solve(
        self,
        examples: list[Example],
        tests: list[Grid],
        *,
        max_depth: int = 2,
        beam_width: int = 32,
        max_nodes: int = 2000,
        time_budget: float = 30.0,
        num_guesses: int = 2,
        exclude_rules: set[str] | None = None,
    ) -> SpatialSolution:
        """Search for a transform sequence explaining the examples.

        Args:
            examples: (input, output) training pairs.
            tests: grids to predict outputs for.
            max_depth: maximum number of composed transforms.
            beam_width: hypotheses kept per depth level.
            max_nodes: total search budget (safety cap).
            time_budget: wall-clock seconds before the search stops and
                returns the best near-miss found so far.
            num_guesses: how many verified hypotheses to keep as
                alternative prediction sets (typically 2).
            exclude_rules: describe() strings of rules already proven
                wrong on this task (counterexample pruning). Excluded
                rules can't be selected as solutions or near-misses, so
                a retry explores genuinely different hypotheses instead
                of re-deriving the same failure.
        """
        excluded = exclude_rules or set()
        scenes = [perceive(inp) for inp, _ in examples]
        if not examples:
            return SpatialSolution(solved=False, hypothesis=None, scenes=scenes)

        def evaluate(transforms: list[Transform]) -> tuple[float, int]:
            return self._evaluate(transforms, examples)

        frontier: list[SpatialHypothesis] = [SpatialHypothesis([], 0.0)]
        solved_hyps: list[SpatialHypothesis] = []
        explored = 0
        deadline = time.monotonic() + time_budget

        for _depth in range(max_depth):
            candidates: list[SpatialHypothesis] = []
            stop = False
            # Two passes: transforms from families that have solved
            # tasks before are tried first (recognition is cheaper
            # than exploration); the full proposal set runs only if
            # nothing familiar explains the data.
            for preferred_only in (True, False):
                for hyp in frontier:
                    new_cands, explored, stop = self._expand_hypothesis(
                        hyp, examples, preferred_only,
                        evaluate, explored, deadline, max_nodes,
                    )
                    candidates.extend(new_cands)
                    if stop:
                        break
                if (
                    any(h.exact == len(examples) for h in candidates)
                    or stop
                ):
                    break

            # Keep the most accurate candidates; prefer shorter
            # sequences on ties (Occam prior over explanations).
            candidates.sort(key=lambda h: (-h.score, len(h.transforms)))
            exact_hits = [
                h for h in candidates
                if h.exact == len(examples)
                and h.describe() not in excluded
            ]
            if exact_hits:
                solved_hyps = self._select_exact(exact_hits, num_guesses)
                break
            frontier = candidates[:beam_width]

        hypotheses = [
            h
            for h in sorted(
                frontier, key=lambda h: (-h.score, len(h.transforms))
            )
            if h.describe() not in excluded
        ][:beam_width]

        if not solved_hyps:
            # Surface the best near-miss so callers can inspect how
            # close the search got rather than getting an opaque None.
            best = hypotheses[0] if hypotheses else None
            return SpatialSolution(
                solved=False,
                hypothesis=best,
                hypotheses=hypotheses,
                scenes=scenes,
                nodes_explored=explored,
                failure=self._analyze_failure(best, examples),
            )

        guesses = [
            self._predict_all(hyp, tests) for hyp in solved_hyps
        ]

        # Success teaches: the most general verified rule becomes part
        # of its recalled strategy set — overfit alternates are not
        # retained as knowledge.
        self.learn(solved_hyps[0].transforms)

        return SpatialSolution(
            solved=True,
            hypothesis=solved_hyps[0],
            predictions=guesses[0],
            hypotheses=hypotheses,
            scenes=scenes,
            nodes_explored=explored,
            guesses=guesses,
            verified_rules=[h.describe() for h in solved_hyps],
        )

    @staticmethod
    def _analyze_failure(
        hyp: SpatialHypothesis | None, examples: list[Example]
    ) -> FailureInfo | None:
        """Explain why the best hypothesis failed.

        Replays the hypothesis against each training pair and reports
        the counterexample: which pair it fit worst, whether the output
        shape was wrong, and which colors were confused on the
        differing cells. This is the observational half of learning
        from mistakes — the data a reflection step needs.
        """
        if hyp is None or not hyp.transforms:
            return None
        pair_scores: list[float] = []
        worst, worst_idx = 1.0, 0
        for i, (inp, out) in enumerate(examples):
            try:
                cand = hyp.apply(inp)
                acc = 1.0 if cand == out else _cell_accuracy(cand, out)
            except Exception as e:  # noqa: BLE001 — hypothesis search must not crash on bad transforms
                logger.debug(f"hypothesis {hyp.describe()} failed on pair {i}: {e}")
                acc = 0.0
            pair_scores.append(acc)
            if acc < worst:
                worst, worst_idx = acc, i
        inp, out = examples[worst_idx]
        try:
            cand = hyp.apply(inp)
        except Exception as e:  # noqa: BLE001 — failure analysis must not crash on bad transforms
            logger.debug(f"failure analysis: hypothesis {hyp.describe()} failed: {e}")
            return FailureInfo(
                rule=hyp.describe(),
                pair_scores=pair_scores,
                worst_pair=worst_idx,
                shape_mismatch=True,
                mismatched=-1,
            )
        shape_mismatch = cand.shape != out.shape
        mismatched = 0
        confusion: dict[str, int] = {}
        if not shape_mismatch:
            for r, c, v in cand.iter_cells():
                expected = out.at(r, c)
                if v != expected:
                    mismatched += 1
                    key = f"{v}->{expected}"
                    confusion[key] = confusion.get(key, 0) + 1
        return FailureInfo(
            rule=hyp.describe(),
            pair_scores=pair_scores,
            worst_pair=worst_idx,
            shape_mismatch=shape_mismatch,
            mismatched=mismatched,
            confusion=confusion,
        )

    @staticmethod
    def _select_exact(
        exact_hits: list[SpatialHypothesis], num_guesses: int
    ) -> list[SpatialHypothesis]:
        """Pick the most general rules among exact matches.

        Fewest transforms, then fewest parameters. A rule with no
        conditions (gravity_down) beats an equally accurate but more
        specific one (gravity_down_1) — specific rules overfit the
        training colors.
        """
        exact_hits.sort(
            key=lambda h: (
                len(h.transforms),
                sum(len(t.params) for t in h.transforms),
            )
        )
        return exact_hits[:num_guesses]

    @staticmethod
    def _evaluate(
        transforms: list[Transform], examples: list[Example]
    ) -> tuple[float, int]:
        """Score a transform sequence: (mean cell accuracy, exact hits)."""
        total_acc, exact = 0.0, 0
        for inp, out in examples:
            try:
                cand = inp
                for t in transforms:
                    cand = t(cand)
            except Exception as e:  # noqa: BLE001 — scoring treats failed transforms as 0
                logger.debug(f"transform sequence failed during scoring: {e}")
                return 0.0, 0
            if cand == out:
                exact += 1
                total_acc += 1.0
            else:
                total_acc += _cell_accuracy(cand, out)
        return total_acc / len(examples), exact

    @staticmethod
    def _predict_all(hyp: SpatialHypothesis, tests: list[Grid]) -> list[Grid]:
        """Apply a hypothesis to each test grid; failures echo the input."""
        preds = []
        for test in tests:
            try:
                preds.append(hyp.apply(test))
            except Exception as e:  # noqa: BLE001 — failed transform echoes input
                logger.debug(f"prediction failed, echoing input: {e}")
                preds.append(test)  # failed transform → echo input
        return preds

    def _expand_hypothesis(
        self,
        hyp: SpatialHypothesis,
        examples: list[Example],
        preferred_only: bool,
        evaluate: Any,
        explored: int,
        deadline: float,
        max_nodes: int,
    ) -> tuple[list[SpatialHypothesis], int, bool]:
        """Expand one hypothesis by one transform.

        Returns ``(candidates, explored, stop)`` — new candidates plus
        the updated node counter and a flag set when the search budget
        (nodes or time) is exhausted.
        """
        candidates: list[SpatialHypothesis] = []
        stop = False

        # Propose transforms conditioned on this hypothesis's
        # *intermediate* outputs, not the raw inputs — this is
        # what lets composition work (e.g. crop→recolor: the
        # recolor map is only inferable once the crop has
        # produced a grid with the output's shape).
        intermediates: list[Example] = []
        for inp, out in examples:
            try:
                mid = inp
                for t in hyp.transforms:
                    mid = t(mid)
            except Exception as e:  # noqa: BLE001 — unexpandable hypothesis yields no candidates
                logger.debug(f"hypothesis expansion failed for {hyp.describe()}: {e}")
                break
            intermediates.append((mid, out))
        if len(intermediates) != len(examples):
            return candidates, explored, stop

        # Learned rules lead the proposal list — recalled
        # strategies are tried before de-novo search —
        # followed by schema instantiations: learned rules
        # abstracted and re-bound to this task's values.
        seen: set[str] = set()
        proposals = []
        for t in [
            *self._learned,
            *self._schema_transforms(intermediates),
            *propose(intermediates),
        ]:
            if t.describe() not in seen:
                seen.add(t.describe())
                proposals.append(t)
        for transform in proposals:
            preferred = (
                self._family_priors[self._family(transform)]
                > 0
            )
            if preferred != preferred_only:
                continue
            explored += 1
            if (
                explored > max_nodes
                or time.monotonic() > deadline
            ):
                stop = True
                break
            seq = [*hyp.transforms, transform]
            score, exact = evaluate(seq)
            candidates.append(
                SpatialHypothesis(seq, score, exact)
            )
        return candidates, explored, stop

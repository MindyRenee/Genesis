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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..reasoning.competence import (
    GoalCondition,
    ProcedureStep,
    SkillMatch,
    TaskCompetence,
    TaskContext,
)
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
    # The domain-general task frame this solve was recognized as, and
    # the reusable skill ids considered before search.
    task_context: TaskContext | None = None
    retrieved_skills: list[str] = field(default_factory=list)


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

    def __init__(
        self,
        network: Any | None = None,
        task_competence: TaskCompetence | None = None,
    ) -> None:
        self.network = network
        self.task_competence = task_competence or TaskCompetence()
        self._active_task: TaskContext | None = None
        self._active_skills: list[SkillMatch] = []
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
        # Schema-level priors adopted from the competence substrate:
        # which families solved tasks of *this kind* before — on other
        # instances, by other reasoners. The search policy itself is
        # learned per task family, not just the solutions.
        self._schema_priors: Counter[str] = Counter()

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
    def _jsonable(value: Any) -> Any:
        """Convert transform parameters to persistence-safe JSON values."""
        if isinstance(value, dict):
            return {str(k): SpatialReasoner._jsonable(v) for k, v in value.items()}
        if isinstance(value, list | tuple | set | frozenset):
            return [SpatialReasoner._jsonable(v) for v in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        return str(value)

    @staticmethod
    def _task_state_features(
        examples: list[Example], tests: list[Grid]
    ) -> dict[str, Any]:
        """Describe the task's structure without binding its answer."""
        features: dict[str, Any] = {
            "grid.task": True,
            "examples.multi": len(examples) > 1,
            "tests.present": bool(tests),
        }
        if not examples:
            return features

        same_shape = all(inp.shape == out.shape for inp, out in examples)
        shrinks = all(
            out.width <= inp.width and out.height <= inp.height
            and out.shape != inp.shape
            for inp, out in examples
        )
        grows = all(
            out.width >= inp.width and out.height >= inp.height
            and out.shape != inp.shape
            for inp, out in examples
        )
        features["shape.same"] = same_shape
        features["shape.shrinks"] = shrinks
        features["shape.grows"] = grows

        color_map: dict[int, int] = {}
        map_consistent = True
        added = removed = False
        fractions: list[float] = []
        input_objects = output_objects = 0
        for inp, out in examples:
            in_colors = set(inp.colors())
            out_colors = set(out.colors())
            added |= bool(out_colors - in_colors)
            removed |= bool(in_colors - out_colors)
            if inp.shape == out.shape:
                changed = sum(
                    1
                    for r, c, v in inp.iter_cells()
                    if out.at(r, c) != v
                )
                fractions.append(changed / max(1, inp.width * inp.height))
                for r, c, v in inp.iter_cells():
                    target = out.at(r, c)
                    if v in color_map and color_map[v] != target:
                        map_consistent = False
                    color_map[v] = target
            else:
                fractions.append(1.0)
            input_objects += len(perceive(inp, compute_relations=False).objects)
            output_objects += len(perceive(out, compute_relations=False).objects)

        mean_change = sum(fractions) / len(fractions)
        if mean_change == 0:
            change_bucket = "none"
        elif mean_change < 0.25:
            change_bucket = "low"
        elif mean_change < 0.65:
            change_bucket = "medium"
        else:
            change_bucket = "high"
        features["cells.change"] = change_bucket
        features["color.map.consistent"] = map_consistent and bool(color_map)
        features["color.added"] = added
        features["color.removed"] = removed
        features["objects.present"] = input_objects > 0
        features["objects.count.same"] = input_objects == output_objects
        features["objects.count.delta"] = (
            "increase"
            if output_objects > input_objects
            else "decrease" if output_objects < input_objects else "same"
        )
        return features

    @staticmethod
    def _operator_affordances(
        examples: list[Example], state: Mapping[str, Any]
    ) -> set[str]:
        """Infer broad operator affordances from example structure."""
        affordances = {"compose", "verify_exact"}
        if state.get("shape.same"):
            affordances.add("shape_preserving")
        if state.get("shape.shrinks"):
            affordances.update(("select", "crop"))
        if state.get("shape.grows"):
            affordances.update(("generate", "expand"))
        if state.get("color.map.consistent"):
            affordances.add("recolor")
        if state.get("color.added"):
            affordances.add("fill")
        if state.get("color.removed"):
            affordances.add("erase")
        if state.get("objects.present"):
            affordances.add("object_manipulation")
        if examples and all(
            inp.shape == out.shape
            and inp.to_lists() != out.to_lists()
            and Counter(v for _, _, v in inp.iter_cells())
            == Counter(v for _, _, v in out.iter_cells())
            for inp, out in examples
        ):
            affordances.add("translate")
        return affordances

    def _recognize_task(
        self, examples: list[Example], tests: list[Grid]
    ) -> TaskContext:
        """Bind this grid task to a domain-general task schema."""
        state = self._task_state_features(examples, tests)
        goal = GoalCondition("train.exact_match", "eq", 1.0)
        context = self.task_competence.recognize(
            domain="spatial.transform",
            state=state,
            actions=self._operator_affordances(examples, state),
            goal_conditions=[goal],
            entities=("grid", "cell", "object", "color", "background"),
            roles=("transform", "input-output"),
        )
        self._active_task = context
        self._active_skills = list(context.skills)
        # The schema's accumulated search prior becomes this task's
        # proposal ordering — "tasks like this usually yield to X."
        self._schema_priors = Counter(context.schema.family_priors)
        return context

    def _resolve_skill_step(
        self, step: ProcedureStep, params: dict[str, Any] | None = None
    ) -> Transform | None:
        """Rebind a serializable procedure step to an executable transform."""
        params = step.parameters if params is None else params
        if params is step.parameters:
            for learned in self._learned:
                if learned.describe() == step.description or (
                    learned.name == step.action
                    and learned.params == step.parameters
                ):
                    return learned
        from .transforms import instantiate

        names = [step.family, step.action, step.action.split("_")[0]]
        for name in dict.fromkeys(n for n in names if n):
            transform = instantiate(name, dict(params))
            if transform is not None:
                return transform
        return None

    @staticmethod
    def _rebind_candidates(
        intermediates: list[Example],
    ) -> list[int]:
        """Colors this task produces that intermediates don't contain —
        the evidence a stored binding should be swapped against."""
        produced: set[int] = set()
        for mid, out in intermediates:
            produced |= set(out.colors()) - set(mid.colors())
        return sorted(produced)

    def _rebound_skill_steps(
        self,
        skill_steps: tuple[ProcedureStep, ...],
        intermediates: list[Example],
    ) -> list[list[Transform]]:
        """Re-parameterized variants of a stored procedure.

        A skill's stored params are *bindings* — the fill color that
        was right last time, not the definition of "fill." When the
        current task's evidence offers different values, the same
        procedure is re-emitted with rebound params: the hypothesis
        space itself reuses learned structure instead of replaying it
        verbatim. Every variant is still exactly evaluated — a bad
        rebind is a bad proposal, not a claimed success.
        """
        candidates = self._rebind_candidates(intermediates)
        if not candidates:
            return []
        variants: list[list[Transform]] = []
        for i, step in enumerate(skill_steps):
            for pname, value in step.parameters.items():
                if not isinstance(value, int) or value in candidates:
                    continue
                for new_value in candidates:
                    params = dict(step.parameters)
                    params[pname] = new_value
                    resolved: list[Transform] = []
                    for j, s in enumerate(skill_steps):
                        t = self._resolve_skill_step(
                            s, params if j == i else None
                        )
                        if t is None:
                            break
                        resolved.append(t)
                    else:
                        variants.append(resolved)
                    if len(variants) >= 4:
                        return variants
        return variants

    def _skill_transforms(
        self, intermediates: list[Example] | None = None
    ) -> list[Transform]:
        """Turn retrieved skills into candidate macro-transforms.

        Emits the verbatim replay plus, when the task's evidence
        offers different bindings, re-parameterized variants — the
        skill is a *procedure*, not a recording.
        """
        transforms: list[Transform] = []
        for match in self._active_skills:
            step_lists: list[tuple[str, list[Transform]]] = []
            steps: list[Transform] = []
            for step in match.skill.steps:
                resolved = self._resolve_skill_step(step)
                if resolved is None:
                    break
                steps.append(resolved)
            if len(steps) == len(match.skill.steps) and steps:
                step_lists.append(("", steps))
            if intermediates:
                for variant in self._rebound_skill_steps(
                    match.skill.steps, intermediates
                ):
                    step_lists.append((":rebound", variant))

            for suffix, seq in step_lists:

                def apply_skill(
                    grid: Grid, seq: list[Transform] = seq
                ) -> Grid:
                    for transform in seq:
                        grid = transform(grid)
                    return grid

                # Rebound variants must describe differently — the
                # dedupe filter keys on describe(), and two variants
                # differing only in bindings would otherwise collapse.
                params: dict[str, Any] = {"skill": match.skill.skill_id}
                if suffix:
                    params["rebound"] = "|".join(
                        t.describe() for t in seq
                    )
                transforms.append(
                    Transform(
                        f"skill:{match.skill.skill_id}{suffix}",
                        apply_skill,
                        params,
                    )
                )
        return transforms

    @staticmethod
    def _transition_state(grid: Grid) -> dict[str, Any]:
        """Flat state variables used by the shared transition learner."""
        scene = perceive(grid, compute_relations=False)
        nonempty = sum(1 for _, _, v in grid.iter_cells() if v != 0)
        largest = max((obj.size for obj in scene.objects), default=0)
        return {
            "grid.width": grid.width,
            "grid.height": grid.height,
            "grid.blank": nonempty == 0,
            "cells.nonempty": nonempty,
            "colors.count": len(grid.colors()),
            "objects.count": len(scene.objects),
            "objects.largest": largest,
        }

    def _expanded_transforms(
        self, transforms: list[Transform]
    ) -> list[Transform]:
        """Expand a retrieved skill macro into its primitive transforms."""
        expanded: list[Transform] = []
        for transform in transforms:
            skill_id = transform.params.get("skill")
            skill = (
                self.task_competence.skills.get(str(skill_id))
                if isinstance(skill_id, str)
                else None
            )
            if skill is None:
                expanded.append(transform)
                continue
            for step in skill.steps:
                resolved = self._resolve_skill_step(step)
                if resolved is not None:
                    expanded.append(resolved)
        return expanded

    def _record_transition_effects(
        self,
        context: TaskContext,
        transforms: list[Transform],
        examples: list[Example],
        *,
        success: bool,
    ) -> None:
        """Replay a verified/failed procedure and learn its effects."""
        expanded = self._expanded_transforms(transforms)
        if not expanded:
            return
        for inp, _expected in examples:
            current = inp
            for transform in expanded:
                try:
                    after = transform(current)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "transition replay failed for %s: %s",
                        transform.describe(), e,
                    )
                    after = current
                self.task_competence.record_transition(
                    context,
                    transform.describe(),
                    self._transition_state(current),
                    self._transition_state(after),
                    success=success,
                )
                current = after

    def _procedure_steps(
        self, transforms: list[Transform]
    ) -> list[ProcedureStep]:
        """Serialize a verified transform sequence as procedure steps."""
        steps: list[ProcedureStep] = []
        for transform in transforms:
            skill_id = transform.params.get("skill")
            skill = (
                self.task_competence.skills.get(str(skill_id))
                if isinstance(skill_id, str)
                else None
            )
            if skill is not None:
                steps.extend(skill.steps)
                continue
            steps.append(
                ProcedureStep(
                    action=transform.name,
                    family=self._schema_family(transform) or "",
                    parameters={
                        str(k): self._jsonable(v)
                        for k, v in transform.params.items()
                    },
                    description=transform.describe(),
                )
            )
        return steps

    def record_task_outcome(
        self,
        solution: SpatialSolution,
        *,
        success: bool,
        score: float,
        examples: list[Example] | None = None,
    ) -> None:
        """Feed an externally verified solve outcome back into competence."""
        context = solution.task_context or self._active_task
        if context is None:
            return
        transforms = (
            list(solution.hypothesis.transforms)
            if solution.hypothesis is not None
            else []
        )
        used_skills = {
            str(t.params["skill"])
            for t in transforms
            if "skill" in t.params
        }
        if not success:
            for skill_id in used_skills:
                self.task_competence.mark_skill_failure(skill_id)
        else:
            # Learning to search: the schema remembers which transform
            # families solved it — a prior over proposals, not an
            # answer. Skill macros contribute their primitive
            # families so the learned bias stays portable.
            for t in self._expanded_transforms(transforms):
                context.schema.family_priors[self._family(t)] = (
                    context.schema.family_priors.get(self._family(t), 0)
                    + 1
                )
        if examples:
            self._record_transition_effects(
                context, transforms, examples, success=success
            )
        self.task_competence.record_episode(
            context,
            steps=self._procedure_steps(transforms),
            success=success,
            verification_score=score,
            goal_conditions=[GoalCondition("train.exact_match", "eq", 1.0)],
            # Exact replay against the held-out target grid is a
            # world-state check, not an assertion.
            verification="external",
        )

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

    @staticmethod
    def prediction_key(predictions: list[Grid]) -> str:
        """Stable fingerprint of a predicted test-output set."""
        return repr(tuple(tuple(tuple(row) for row in grid.to_lists()) for grid in predictions))

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
        exclude_predictions: set[str] | None = None,
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
            exclude_predictions: fingerprints of held-out test outputs
                already disproven on this task. This is behavioral pruning:
                semantically equivalent hypotheses making the same wrong
                prediction are excluded without banning an entire family.
        """
        excluded = exclude_rules or set()
        excluded_predictions = exclude_predictions or set()
        scenes = [perceive(inp) for inp, _ in examples]
        if not examples:
            return SpatialSolution(solved=False, hypothesis=None, scenes=scenes)
        task_context = self._recognize_task(examples, tests)
        retrieved_skills = [
            match.skill.skill_id for match in self._active_skills
        ]

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
            exact_hits = []
            for h in candidates:
                if h.exact != len(examples) or h.describe() in excluded:
                    continue
                predictions = self._predict_all(h, tests)
                if self.prediction_key(predictions) in excluded_predictions:
                    continue
                exact_hits.append(h)
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
                task_context=task_context,
                retrieved_skills=retrieved_skills,
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
            task_context=task_context,
            retrieved_skills=retrieved_skills,
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

        # Retrieved skills lead the proposal list, followed by
        # recalled rules and schema instantiations — all before
        # de-novo proposals. Skills compose: a macro may follow
        # primitives or another skill — "drop it, then fill it" is
        # two procedures, not one new transform. Exact training
        # replay still verifies every candidate.
        seen: set[str] = set()
        proposals = []
        skill_transforms = self._skill_transforms(intermediates)
        for t in [
            *skill_transforms,
            *self._learned,
            *self._schema_transforms(intermediates),
            *propose(intermediates),
        ]:
            if t.describe() not in seen:
                seen.add(t.describe())
                proposals.append(t)
        for transform in proposals:
            fam = self._family(transform)
            preferred = (
                self._family_priors[fam] > 0
                or self._schema_priors[fam] > 0
                or "skill" in transform.params
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

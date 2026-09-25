"""Classification agent — labels items by the rule, stated or not.

Apply mode: the predicate is written on the worksheet — the agent
reads it and evaluates each item against it. Induce mode: nothing is
stated — worked examples and per-guess feedback teach which features
predict membership (attribute presence and count are the perceptible
features; "exactly two of" is learnable as a count rule). Wrong
guesses reveal the true label — evidence always beats the prior,
including a misread stated rule.
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
from .world import GuessTarget, Item, RuleGame, eval_predicate


@dataclass
class _Stat:
    """Observed cost of one label for one feature."""

    attempts: int = 0
    total_cost: float = 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.attempts if self.attempts else 0.0


@dataclass
class ClassificationResult:
    """Outcome of one worksheet attempt."""

    solved: bool
    state: str  # "WIN" or "STUCK"
    steps: int
    guesses: int
    wrong: int


class RuleAgent:
    """Trial-and-feedback classifier on the competence substrate."""

    def __init__(
        self,
        seed: int = 42,
        epsilon: float = 0.3,
        task_competence: TaskCompetence | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.epsilon = epsilon
        self.task_competence = task_competence or TaskCompetence()
        self._task_context: TaskContext | None = None
        self._episode_skill_ids: set[str] = set()
        # Learned affordance: guessing label L when a feature is
        # present costs this much on average (0 = right, 1 = wrong).
        self._feat_stats: dict[str, dict[bool, _Stat]] = {}
        # Schema-level priors adopted from skills: (feature, label)
        # -> prior cost.
        self._feat_priors: dict[tuple[str, bool], float] = {}

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

    def _features(self, item: GuessTarget | Item) -> list[str]:
        """Perceptible features of an item: which attributes it has
        and how many — "exactly two parts" is learnable as a count."""
        attrs = item.attrs
        feats = [f"has:{a}" for a in attrs]
        feats.append(f"n:{self._bucket(len(attrs))}:{len(attrs)}")
        return feats

    def _game_features(self, world: RuleGame) -> dict[str, Any]:
        vocab = {a for it in world.items for a in it.attrs}
        return {
            "puzzle.classification": True,
            "items.count": self._bucket(len(world.items)),
            "items.unlabeled": self._bucket(world.mismatches()),
            "attrs.variety": self._bucket(len(vocab)),
            "rule.stated": world.predicate is not None,
            "examples.count": self._bucket(len(world.examples)),
        }

    def _recognize(self, world: RuleGame) -> TaskContext:
        """Bind this worksheet to a task schema once per episode."""
        if self._task_context is not None:
            return self._task_context
        context = self.task_competence.recognize(
            domain="classification.rule",
            state=self._game_features(world),
            actions=("guess",),
            goal_conditions=[
                GoalCondition("items.unlabeled", "eq", 0),
            ],
            entities=("item", "label"),
            # "Does this fit the rule?" is the family shape — tessel
            # games, odd-one-out, sorting by an unseen rule. The rule
            # IS a constraint on membership, so it shares "constrain"
            # with the sorter/relations/assembly families even though
            # its per-feature affordance axis is not theirs.
            roles=("classify", "rule", "membership", "constrain"),
        )
        self._task_context = context
        self._adopt_skill_priors(context)
        return context

    def _adopt_skill_priors(self, context: TaskContext) -> None:
        """Seed feature→label costs from skills won on similar games.

        What transfers is the *shape* — which feature kinds predict
        membership — not the items. Names and labels are bindings.
        """
        for match in context.skills:
            adopted = False
            for step in match.skill.steps:
                if step.family != "feature":
                    continue
                feat = step.parameters.get("feature")
                yes_cost = step.parameters.get("yes_cost")
                no_cost = step.parameters.get("no_cost")
                if isinstance(feat, str) and isinstance(
                    yes_cost, int | float
                ) and isinstance(no_cost, int | float):
                    adopted = True
                    for label, cost in ((True, yes_cost), (False, no_cost)):
                        prev = self._feat_priors.get((feat, label))
                        self._feat_priors[(feat, label)] = (
                            float(cost)
                            if prev is None
                            else min(prev, float(cost))
                        )
            if adopted:
                self._episode_skill_ids.add(match.skill.skill_id)

    def _transition_state(self, world: RuleGame) -> dict[str, Any]:
        return {
            "items.unlabeled": world.mismatches(),
            "items.labeled": len(world.labels()),
        }

    # ── Action selection ──────────────────────────────────────

    def _feature_evidence(self, item: GuessTarget) -> float:
        """Summed evidence for 'yes' across the item's features —
        positive means the learned stats say it belongs."""
        total = 0.0
        for f in self._features(item):
            stats = self._feat_stats.get(f)
            if stats is not None:
                yes_st, no_st = stats.get(True), stats.get(False)
                if yes_st is not None and yes_st.attempts:
                    total += 0.5 - yes_st.mean_cost
                if no_st is not None and no_st.attempts:
                    total += no_st.mean_cost - 0.5
            else:
                yes_prior = self._feat_priors.get((f, True))
                no_prior = self._feat_priors.get((f, False))
                if yes_prior is not None:
                    total += 0.5 - yes_prior
                if no_prior is not None:
                    total += no_prior - 0.5
        return total

    def _predict(self, item: GuessTarget, world: RuleGame) -> bool:
        """The guess: evaluate the stated rule when one exists,
        else vote from learned feature evidence."""
        if world.predicate is not None:
            return eval_predicate(
                world.predicate, frozenset(item.attrs)
            )
        evidence = self._feature_evidence(item)
        if evidence == 0.0:
            return self._rng.random() < 0.5
        return evidence > 0.0

    def _teach(self, item: GuessTarget | Item, truth: bool) -> None:
        """Supervision: the revealed label is evidence for every
        feature the item carries."""
        for f in self._features(item):
            stats = self._feat_stats.setdefault(
                f, {True: _Stat(), False: _Stat()}
            )
            stats[truth].attempts += 1
            stats[not truth].attempts += 1
            stats[not truth].total_cost += 1.0

    # ── Episode ───────────────────────────────────────────────

    def step(self, world: RuleGame) -> str:
        """One action: classify a pending item."""
        context = self._recognize(world)
        before = self._transition_state(world)
        targets = world.candidates()
        if not targets:
            return self._finish(world, "WIN")
        chosen = self._rng.choice(targets)
        guess = self._predict(chosen, world)
        outcome = world.guess(chosen.name, guess)
        if outcome is None:
            return self._finish(world, "STUCK")
        # Every oracle answer is supervision — a confirmed guess and
        # a correction teach the same thing: these features went with
        # this label.
        self._teach(chosen, bool(outcome["truth"]))
        self.task_competence.record_transition(
            context, "guess", before, self._transition_state(world)
        )
        if world.complete():
            return self._finish(world, "WIN")
        return "PLAYING"

    def solve(
        self, world: RuleGame, max_steps: int = 200
    ) -> ClassificationResult:
        """Drive one worksheet to all-correct or exhaustion."""
        # Worked examples are supervision shown before the test —
        # teach the stats before the first guess.
        for item, label in world.known_examples():
            self._teach(item, label)
        steps = guesses = wrong = 0
        state = "PLAYING"
        while state == "PLAYING" and steps < max_steps:
            before_n = len(world.labels())
            state = self.step(world)
            steps += 1
            guesses += 1
            if len(world.labels()) == before_n:
                wrong += 1
        if state == "PLAYING":
            state = self._finish(world, "STUCK")
        return ClassificationResult(
            solved=state == "WIN",
            state=state,
            steps=steps,
            guesses=guesses,
            wrong=wrong,
        )

    def _finish(self, world: RuleGame, state: str) -> str:
        del world
        self.on_episode_end(state)
        return state

    def on_episode_end(self, state: str) -> None:
        """Report the episode; only a fully-correct worksheet
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
                    GoalCondition("items.unlabeled", "eq", 0),
                ],
                verification="external",
            )
            self._task_context = None
            self._episode_skill_ids.clear()

    def _skill_steps(self) -> list[ProcedureStep]:
        """The learned procedure: the feature→label cost map."""
        steps = []
        for feat in sorted(self._feat_stats):
            stats = self._feat_stats[feat]
            yes_st, no_st = stats.get(True), stats.get(False)
            if not yes_st or not no_st or not yes_st.attempts:
                continue
            steps.append(
                ProcedureStep(
                    action="guess",
                    family="feature",
                    parameters={
                        "feature": feat,
                        "yes_cost": round(yes_st.mean_cost, 3),
                        "no_cost": round(no_st.mean_cost, 3),
                    },
                    description=(
                        f"{feat}: yes costs {yes_st.mean_cost:.2f}, "
                        f"no costs {no_st.mean_cost:.2f}"
                    ),
                )
            )
        return steps

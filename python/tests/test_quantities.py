"""Quantities world + competence adapter — the "make N" family.

Fill the basket to exactly the target from groups with perceptible
counts: "give me five," "which groups make ten." Overflow is
rejected (the capacity is real), dead remainders force a lift, and
the agent learns the fit→cost affordance — including that "fits but
dead-ends" differs from "fits."
"""

from __future__ import annotations

import json
import random

from genesis_cognitive.quantities import (
    Basket,
    Group,
    QuantitiesAgent,
)
from genesis_cognitive.reasoning import TaskCompetence
from genesis_cognitive.spatial.practice import (
    SpatialPractice,
    normalize_offered,
)


def _basket() -> Basket:
    return Basket(
        5,
        [
            Group(0, "pair", 2),
            Group(1, "trio", 3),
            Group(2, "quad", 4),
            Group(3, "one", 1),
        ],
    )


class TestBasket:
    """The world is honest: capacity is enforced, never filtered."""

    def test_overflow_rejected(self):
        w = _basket()
        assert w.add("quad")["accepted"]
        # 4 in basket, room 1 — trio overflows.
        out = w.add("trio")
        assert out is not None and not out["accepted"]
        assert out["overflow"] == 2
        assert w.basket_sum() == 4

    def test_exact_total_completes(self):
        w = _basket()
        w.add("pair")
        w.add("trio")
        assert w.complete() and w.mismatches() == 0

    def test_dead_end_detected(self):
        # Room 1 with only a 3 left in the pool → dead.
        w = Basket(5, [Group(0, "quad", 4), Group(1, "trio", 3)])
        w.add("quad")
        assert w.dead_end()
        assert not w.complete()

    def test_generate_is_solvable(self):
        for seed in range(10):
            w = Basket.generate(5, 9, random.Random(seed))
            assert w.solvable()

    def test_remove_returns_to_pool(self):
        w = _basket()
        w.add("pair")
        w.remove("pair")
        assert "pair" in w.pool and w.basket_sum() == 0


class TestQuantitiesAgent:
    """Trial-and-error composition on the competence substrate."""

    def test_solves_make_five(self):
        agent = QuantitiesAgent(seed=1)
        assert agent.solve(_basket(), max_steps=300).solved

    def test_generated_worksheets_solve(self):
        agent = QuantitiesAgent(seed=7)
        for seed in range(8):
            w = Basket.generate(5, 9, random.Random(seed))
            assert agent.solve(w, max_steps=300).solved

    def test_overflow_is_learned_as_costly(self):
        competence = TaskCompetence()
        agent = QuantitiesAgent(seed=2, task_competence=competence)
        assert agent.solve(_basket(), 300).solved
        # If an overflow was ever tried, it carries cost.
        if "over" in agent._kind_stats:
            assert agent._kind_stats["over"].mean_cost > 0.0

    def test_consolidates_affordance_as_skill(self):
        competence = TaskCompetence()
        agent = QuantitiesAgent(seed=1, task_competence=competence)
        assert agent.solve(_basket(), 300).solved
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        assert skill.verification == "external"
        options = [s for s in skill.steps if s.family == "option"]
        assert options
        # The procedure is the kind→cost shape — no group names.
        assert all(
            set(s.parameters) == {"kind", "mean_cost"} for s in options
        )

    def test_priors_transfer_between_worksheets(self):
        competence = TaskCompetence()
        warm = QuantitiesAgent(seed=1, task_competence=competence)
        assert warm.solve(_basket(), 300).solved
        warm2 = QuantitiesAgent(seed=9, task_competence=competence)
        warm2._recognize(
            Basket.generate(5, 9, random.Random(5))
        )
        assert warm2._kind_priors


class TestQuantitiesOffered:
    """The drop-box accepts quantities specs."""

    def test_normalize_explicit(self):
        spec = normalize_offered(
            {
                "family": "quantities",
                "target": 5,
                "groups": {"two": 2, "three": 3},
            },
            "x",
        )
        assert spec is not None and spec["target"] == 5
        assert len(spec["groups"]) == 2

    def test_normalize_generate(self):
        spec = normalize_offered(
            {
                "family": "quantities",
                "generate": {"groups": 5, "target": 9, "seed": 3},
            },
            "x",
        )
        assert spec is not None and spec["generate"]["target"] == 9

    def test_unsolvable_target_rejected(self):
        # All evens can't make an odd target.
        assert (
            normalize_offered(
                {
                    "family": "quantities",
                    "target": 5,
                    "groups": {"a": 2, "b": 4},
                },
                "x",
            )
            is None
        )

    def test_malformed_rejected(self):
        assert (
            normalize_offered({"family": "quantities"}, "x") is None
        )
        assert (
            normalize_offered(
                {"family": "quantities", "target": 0, "groups": {"a": 1}},
                "x",
            )
            is None
        )

    def test_practice_runs_quantities(self, tmp_path):
        d = tmp_path / "offered_puzzles"
        d.mkdir()
        (d / "make_five.json").write_text(
            json.dumps(
                {
                    "family": "quantities",
                    "name": "make_five",
                    "target": 5,
                    "groups": {"two": 2, "three": 3, "four": 4},
                }
            )
        )
        practice = SpatialPractice(str(tmp_path))
        from genesis_cognitive.spatial.solver import SpatialReasoner

        result = practice.attempt(SpatialReasoner())
        assert result is not None
        assert result.family == "quantities"
        assert result.solved and result.mastered
        assert practice.mastery["make_five"] == 1.0

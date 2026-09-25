"""Relations world + competence adapter — the "directions" family.

Arrange named things so stated relations hold: "star left of moon,
moon next to sun" — plus seriation through the ``ordered`` relation
("the cups go fewest to most"). The world is honest (placements
always succeed; goal satisfaction is checked against the board) and
the agent learns the agree→cost affordance by trial.
"""

from __future__ import annotations

import json
import random

from genesis_cognitive.reasoning import TaskCompetence
from genesis_cognitive.relations import (
    Arrangement,
    Goal,
    RelationsAgent,
    Thing,
)
from genesis_cognitive.spatial.practice import (
    SpatialPractice,
    normalize_offered,
)


def _directions() -> Arrangement:
    return Arrangement(
        4,
        [
            Thing(0, "star"),
            Thing(1, "moon"),
            Thing(2, "sun"),
            Thing(3, "cloud"),
        ],
        [
            Goal("left_of", "star", "moon"),
            Goal("next_to", "moon", "sun"),
            Goal("apart", "star", "sun"),
        ],
    )


def _seriation() -> Arrangement:
    things = [
        Thing(0, "cup_a", (("count", 4),)),
        Thing(1, "cup_b", (("count", 1),)),
        Thing(2, "cup_c", (("count", 7),)),
        Thing(3, "cup_d", (("count", 2),)),
    ]
    goals = [
        Goal("ordered", a, b, "count")
        for i, a in enumerate(("cup_a", "cup_b", "cup_c", "cup_d"))
        for b in ("cup_a", "cup_b", "cup_c", "cup_d")[i + 1 :]
    ]
    return Arrangement(4, things, goals)


class TestArrangement:
    """The world is honest: relations are checked, never filtered."""

    def test_relations_checked_against_board(self):
        w = _directions()
        assert w.place("moon", 0) is not None
        assert w.place("star", 2) is not None
        # star@2, moon@0 violates left_of.
        assert w.mismatches() > 0
        assert [g.rel for g in w.violations()] == ["left_of"]

    def test_solvable_and_complete(self):
        w = _directions()
        assert w.solvable()
        # A known-good row: star@0 moon@1 sun@2 cloud@3.
        for name, pos in (
            ("star", 0),
            ("moon", 1),
            ("sun", 2),
            ("cloud", 3),
        ):
            assert w.place(name, pos) is not None
        assert w.complete()

    def test_swap_repairs_a_full_row(self):
        w = _directions()
        for name, pos in (
            ("moon", 0),
            ("sun", 1),
            ("star", 2),
            ("cloud", 3),
        ):
            w.place(name, pos)
        # left_of(star,moon): 2<0 fails; apart(star,sun): |2-1|=1 fails.
        assert w.mismatches() == 2
        out = w.swap(2, 0)  # star@0, moon@2
        assert out is not None
        # left_of and next_to now hold; only apart(star,sun) left.
        assert out["delta_unsatisfied"] == -1
        assert w.mismatches() == 1

    def test_ordered_is_order_agreement(self):
        w = Arrangement(
            2,
            [
                Thing(0, "few", (("count", 1),)),
                Thing(1, "many", (("count", 9),)),
            ],
            [Goal("ordered", "few", "many", "count")],
        )
        w.place("many", 0)
        w.place("few", 1)
        assert w.mismatches() > 0  # few must sit left of many

    def test_generate_is_solvable(self):
        for seed in range(10):
            w = Arrangement.generate(4, 3, random.Random(seed))
            assert w.solvable()


class TestRelationsAgent:
    """Trial-and-error arrangement on the competence substrate."""

    def test_solves_directions(self):
        agent = RelationsAgent(seed=1)
        assert agent.solve(_directions(), max_steps=500).solved

    def test_solves_seriation(self):
        agent = RelationsAgent(seed=3)
        assert agent.solve(_seriation(), max_steps=500).solved

    def test_generated_tasks_solve(self):
        agent = RelationsAgent(seed=7)
        for seed in range(6):
            w = Arrangement.generate(4, 3, random.Random(seed))
            assert agent.solve(w, max_steps=500).solved

    def test_consolidates_affordance_as_skill(self):
        competence = TaskCompetence()
        agent = RelationsAgent(seed=1, task_competence=competence)
        assert agent.solve(_directions(), max_steps=500).solved
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        assert skill.verification == "external"
        placement = [s for s in skill.steps if s.family == "placement"]
        assert placement
        # The procedure is the agree→cost shape — no thing names or
        # positions in the parameters (those are bindings).
        assert all(
            set(s.parameters) == {"agrees", "support", "mean_cost"}
            for s in placement
        )

    def test_priors_transfer_between_arrangements(self):
        competence = TaskCompetence()
        warm = RelationsAgent(seed=1, task_competence=competence)
        assert warm.solve(_directions(), 500).solved
        # A second episode on a fresh arrangement sees the affordance
        # shape as a prior — the learned invariant "satisfy decidable
        # goals" is family-level, not instance-level.
        warm2 = RelationsAgent(seed=9, task_competence=competence)
        assert warm2._adopt_skill_priors is not None
        warm2._recognize(Arrangement.generate(4, 3, random.Random(5)))
        assert warm2._agree_priors

    def test_cross_domain_evidence_priors_from_sorter(self):
        """A foreign constraint-satisfaction skill crosses domains:
        the shared "constrain" role claims the analogy and the
        normalized support axis carries it — the shape transfers,
        never the bindings."""
        from genesis_cognitive.sorter import ShapeSorter, SorterAgent

        competence = TaskCompetence()
        warm = SorterAgent(seed=1, task_competence=competence)
        assert warm.solve(
            ShapeSorter.generate(3, random.Random(10)), max_steps=500
        ).solved
        skill = next(iter(competence.skills.values()))
        assert skill.signature.domain == "sorter.fit"

        agent = RelationsAgent(seed=9, task_competence=competence)
        context = agent._recognize(_directions())
        foreign = [
            m
            for m in context.skills
            if m.skill.signature.domain != "relations.arrange"
        ]
        assert any(m.skill is skill for m in foreign)
        # The foreign affordance lands on the shared evidence axis —
        # the native agree→cost map stays for same-domain skills.
        assert agent._evidence_priors
        assert not agent._agree_priors

    def test_role_overlap_without_support_axis_not_offered(self):
        """Role overlap alone is not an analogy: a foreign skill that
        publishes no normalized support axis is never offered."""
        from genesis_cognitive.reasoning import (
            GoalCondition,
            ProcedureStep,
        )

        competence = TaskCompetence()
        ctx = competence.recognize(
            domain="fake.constrained",
            state={"x": 1},
            actions=("do",),
            goal_conditions=[GoalCondition("x", "eq", 0)],
            # Shares "constrain" with the relations family but the
            # procedure exposes no support axis to rebind.
            roles=("constrain", "select"),
        )
        competence.record_episode(
            ctx,
            steps=[ProcedureStep("do")],
            success=True,
            verification_score=1.0,
            verification="external",
        )

        agent = RelationsAgent(seed=9, task_competence=competence)
        context = agent._recognize(_directions())
        assert not context.skills
        assert not agent._evidence_priors


class TestRelationsOffered:
    """The drop-box accepts relations specs."""

    def test_normalize_directions(self):
        spec = normalize_offered(
            {
                "family": "relations",
                "positions": 4,
                "objects": ["a", "b", "c", "d"],
                "goals": [
                    {"rel": "left_of", "a": "a", "b": "b"},
                    {"rel": "next_to", "a": "b", "b": "c"},
                ],
            },
            "x",
        )
        assert spec is not None and spec["positions"] == 4
        assert len(spec["goals"]) == 2

    def test_normalize_seriation_shorthand(self):
        spec = normalize_offered(
            {
                "family": "relations",
                "objects": [
                    {"name": "a", "attrs": {"count": 3}},
                    {"name": "b", "attrs": {"count": 1}},
                    {"name": "c", "attrs": {"count": 2}},
                ],
                "goals": [{"rel": "ordered_all", "attr": "count"}],
            },
            "x",
        )
        # Three things → three unordered ordered-pairs.
        assert spec is not None and len(spec["goals"]) == 3

    def test_unsolvable_goals_rejected(self):
        # a left of b AND b left of a — impossible.
        assert (
            normalize_offered(
                {
                    "family": "relations",
                    "objects": ["a", "b"],
                    "goals": [
                        {"rel": "left_of", "a": "a", "b": "b"},
                        {"rel": "left_of", "a": "b", "b": "a"},
                    ],
                },
                "x",
            )
            is None
        )

    def test_malformed_rejected(self):
        assert (
            normalize_offered(
                {"family": "relations", "objects": ["a"]}, "x"
            )
            is None
        )
        assert (
            normalize_offered(
                {
                    "family": "relations",
                    "objects": ["a", "b"],
                    "goals": [{"rel": "bogus", "a": "a", "b": "b"}],
                },
                "x",
            )
            is None
        )

    def test_practice_runs_relations(self, tmp_path):
        d = tmp_path / "offered_puzzles"
        d.mkdir()
        (d / "line_up.json").write_text(
            json.dumps(
                {
                    "family": "relations",
                    "name": "line_up",
                    "positions": 3,
                    "objects": ["a", "b", "c"],
                    "goals": [
                        {"rel": "left_of", "a": "a", "b": "b"},
                        {"rel": "next_to", "a": "b", "b": "c"},
                    ],
                }
            )
        )
        practice = SpatialPractice(str(tmp_path))
        from genesis_cognitive.spatial.solver import SpatialReasoner

        result = practice.attempt(SpatialReasoner())
        assert result is not None
        assert result.family == "relations"
        assert result.solved and result.mastered
        assert practice.mastery["line_up"] == 1.0

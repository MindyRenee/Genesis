"""Sorter world + competence adapter, and the offered-puzzle channel.

The sorter is the matching-under-constraints family: blocks with
attributes, apertures with conjunctive constraints, and a lid that
always "works" but never fills. The agent learns the match→cost
affordance by trial; the practice channel lets a puzzle spec arrive
in its world as data.
"""

from __future__ import annotations

import json
import random

from genesis_cognitive.reasoning import TaskCompetence
from genesis_cognitive.sorter import Block, ShapeSorter, Slot, SorterAgent
from genesis_cognitive.spatial.practice import SpatialPractice, normalize_offered


def _sorter(n: int, seed: int, **kw) -> ShapeSorter:
    return ShapeSorter.generate(n, random.Random(seed), **kw)


class TestShapeSorter:
    """The world is honest: constraints are checked, never filtered."""

    def test_insert_accepts_only_full_match(self):
        sorter = ShapeSorter(
            [Slot(0, (("shape", "star"), ("size", "2")))],
            [
                Block(0, (("shape", "star"), ("size", "2"))),
                Block(1, (("shape", "star"), ("size", "1"))),
            ],
            has_lid=False,
        )
        out = sorter.insert(1, 0)
        assert out is not None and not out["accepted"]
        assert out["matched"] == 1 and out["needed"] == 2
        out = sorter.insert(0, 0)
        assert out is not None and out["accepted"]
        assert sorter.complete()

    def test_lid_accepts_but_never_solves(self):
        sorter = _sorter(3, 5)
        before = sorter.mismatches()
        block_id = next(iter(sorter.pool))
        out = sorter.insert_top(block_id)
        assert out is not None and out["accepted"]
        assert sorter.mismatches() == before
        assert block_id not in sorter.pool
        # ...but the box can be opened and the block recovered.
        sorter.empty_top()
        assert block_id in sorter.pool

    def test_generate_is_solvable_with_decoys(self):
        for seed in range(8):
            sorter = _sorter(4, seed, difficulty=2, decoys=3)
            # Every slot must be fillable by some pool block.
            for slot in sorter.slots:
                assert any(
                    sorter.matched_attrs(b, slot) == slot.needed
                    for b in sorter.pool.values()
                )

    def test_arbitrary_attributes(self):
        """The constraint vocabulary is open — not just shapes."""
        sorter = ShapeSorter(
            [Slot(0, (("sound", "buh"),))],
            [
                Block(0, (("letter", "b"), ("sound", "buh"))),
                Block(1, (("letter", "d"), ("sound", "duh"))),
            ],
            has_lid=False,
        )
        assert not sorter.insert(1, 0)["accepted"]
        assert sorter.insert(0, 0)["accepted"]


class TestSorterCompetenceAdapter:
    """A fifth domain on the same substrate: matching tasks."""

    def test_agent_solves_and_consolidates_skill(self):
        competence = TaskCompetence()
        agent = SorterAgent(seed=1, task_competence=competence)
        result = agent.solve(_sorter(3, 10), max_steps=500)

        assert result.solved
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        # The sorter's own constraint check is world-state evidence.
        assert skill.verification == "external"
        # The learned procedure is the match→cost affordance map, not
        # a recorded answer key — no block ids or slots in params.
        selection = [s for s in skill.steps if s.family == "selection"]
        assert selection
        assert all(
            set(s.parameters) == {"matched", "support", "mean_cost"}
            for s in selection
        )

    def test_learned_affordance_is_the_full_match(self):
        """The discovered rule: satisfy *every* constraint → free."""
        competence = TaskCompetence()
        agent = SorterAgent(seed=1, task_competence=competence)
        result = agent.solve(_sorter(4, 10), max_steps=600)
        assert result.solved

        skill = next(iter(competence.skills.values()))
        costs = {
            s.parameters["matched"]: s.parameters["mean_cost"]
            for s in skill.steps
            if s.family == "selection"
        }
        assert costs[1.0] == 0.0
        partial = [v for k, v in costs.items() if 0.0 <= k < 1.0]
        assert partial and all(c > 0.0 for c in partial)

    def test_lid_learned_as_worthless(self):
        """The free move that fills nothing is learned against."""
        competence = TaskCompetence()
        agent = SorterAgent(seed=2, task_competence=competence)
        # Several episodes so the lid's cost is observed even under
        # exploitation-heavy runs.
        for seed in range(4):
            result = agent.solve(_sorter(3, seed), max_steps=500)
            assert result.solved
        # Once the lid has ever been tried, its bucket records a
        # strictly-positive cost — "accepts anything, fills nothing."
        if -1.0 in agent._match_stats:
            assert agent._match_stats[-1.0].mean_cost > 0.0

    def test_skill_prior_transfers_across_difficulty(self):
        """The invariant — match *all* constraints — survives harder
        sorters where near-misses exist."""
        competence = TaskCompetence()
        agent = SorterAgent(seed=1, task_competence=competence)
        assert agent.solve(_sorter(3, 10, difficulty=1), 500).solved
        # Warm on difficulty 2 with near-miss decoys.
        warm = agent.solve(_sorter(4, 20, difficulty=2, decoys=3), 500)
        assert warm.solved
        assert warm.lid_uses <= 1

        cold_agent = SorterAgent(seed=1, task_competence=TaskCompetence())
        cold = cold_agent.solve(
            _sorter(4, 20, difficulty=2, decoys=3), 500
        )
        assert cold.solved
        assert warm.wasted <= cold.wasted


class TestOfferedPuzzles:
    """The drop-box: puzzle specs arriving as files in its world."""

    def _practice(self, tmp_path) -> SpatialPractice:
        (tmp_path / "offered_puzzles").mkdir()
        return SpatialPractice(str(tmp_path))

    def test_offered_sorter_becomes_current_and_attemptable(self, tmp_path):
        practice = self._practice(tmp_path)
        (tmp_path / "offered_puzzles" / "shapes.json").write_text(
            json.dumps(
                {
                    "name": "shapes",
                    "family": "sorter",
                    "hint": "Each opening takes its own kind.",
                    "slots": [
                        {"accepts": {"shape": "star"}},
                        {"accepts": {"shape": "circle"}},
                    ],
                    "blocks": [
                        {"shape": "star"},
                        {"shape": "circle"},
                    ],
                    "lid": True,
                }
            )
        )
        task = practice.current_task()
        assert task is not None and task["name"] == "shapes"

        from genesis_cognitive.spatial.solver import SpatialReasoner

        result = practice.attempt(SpatialReasoner())
        assert result is not None
        assert result.family == "sorter"
        assert result.solved and result.mastered
        assert practice.mastery["shapes"] == 1.0
        # Persisted like any curriculum mastery.
        reloaded = SpatialPractice(str(tmp_path))
        assert reloaded.mastery["shapes"] == 1.0

    def test_offered_grid_runs_through_reasoner(self, tmp_path):
        practice = self._practice(tmp_path)
        (tmp_path / "offered_puzzles" / "grow.json").write_text(
            json.dumps(
                {
                    "name": "grow",
                    "family": "grid",
                    "train": [
                        [
                            [[0, 0], [0, 1]],
                            [[1, 0], [0, 1]],
                        ]
                    ],
                    "test": [
                        [
                            [[0, 2], [0, 0]],
                            [[2, 2], [0, 0]],
                        ]
                    ],
                }
            )
        )
        from genesis_cognitive.spatial.solver import SpatialReasoner

        task = practice.current_task()
        assert task is not None and task["family"] == "grid"
        result = practice.attempt(SpatialReasoner())
        assert result is not None and result.family == "grid"
        assert practice.attempts["grow"] == 1

    def test_malformed_and_impossible_offers_are_skipped(self, tmp_path):
        practice = self._practice(tmp_path)
        d = tmp_path / "offered_puzzles"
        (d / "broken.json").write_text("{not json")
        (d / "impossible.json").write_text(
            json.dumps(
                {
                    "name": "impossible",
                    "family": "sorter",
                    "slots": [{"accepts": {"shape": "star"}}],
                    "blocks": [{"shape": "circle"}],
                }
            )
        )
        assert practice.offered_tasks() == []
        # Curriculum behavior is unaffected by bad drops.
        assert practice.current_task() is not None

    def test_locked_count_with_offered_current(self, tmp_path):
        practice = self._practice(tmp_path)
        (tmp_path / "offered_puzzles" / "shapes.json").write_text(
            json.dumps(
                {
                    "name": "shapes",
                    "family": "sorter",
                    "slots": [{"accepts": {"shape": "star"}}],
                    "blocks": [{"shape": "star"}],
                }
            )
        )
        # Current task is the offer; locked_count must not crash and
        # still reports the whole curriculum waiting behind it.
        n = practice.locked_count()
        assert n >= 0

    def test_normalize_rejects_unknown_family(self):
        assert normalize_offered({"family": "chess"}, "x") is None
        assert normalize_offered("not a dict", "x") is None
        assert normalize_offered({"family": "grid"}, "x") is None


class TestPerceptualSorter:
    """The seen sorter — the world renders its pieces, the oracle
    stops leaking `matched`, and the agent works from views."""

    def _cortex(self):
        from genesis_cognitive.vision.v1 import V1Model
        from genesis_cognitive.vision.visual_cortex import VisualCortex

        return VisualCortex(V1Model())

    def test_candidates_hide_the_oracle(self):
        from genesis_cognitive.sorter import PerceptualSorter

        base = _sorter(3, 5, difficulty=1)
        ps = PerceptualSorter(
            base.slots,
            [*base.pool.values(), *base.in_top],
            has_lid=base.has_lid,
            cortex=self._cortex(),
            seed=1,
        )
        for c in ps.candidates():
            assert c.matched == -1 and c.needed == 0
        # But the physics are unchanged — insert still reports truth.
        out = ps.insert(0, ps.slots[0].index)
        assert out is not None and "matched" in out

    def test_views_are_seen_and_stable(self):
        from genesis_cognitive.sorter import PerceptualSorter

        base = _sorter(2, 9, difficulty=1)
        ps = PerceptualSorter(
            base.slots,
            [*base.pool.values()],
            cortex=self._cortex(),
            seed=2,
        )
        v = ps.view("block", 0)
        assert v is not None and len(v) > 0
        # Seen once, remembered — the cached view is returned.
        assert ps.view("block", 0) is v
        # Similarity is a real number in [0, 1].
        sim = ps.similarity(0, ps.slots[0].index)
        assert 0.0 <= sim <= 1.0

    def test_perceptual_agent_solves(self):
        from genesis_cognitive.sorter import (
            PerceptualSorter,
            PerceptualSorterAgent,
        )

        base = _sorter(3, 11, difficulty=1)
        ps = PerceptualSorter(
            base.slots,
            [*base.pool.values(), *base.in_top],
            has_lid=base.has_lid,
            cortex=self._cortex(),
            seed=3,
        )
        tc = TaskCompetence()
        res = PerceptualSorterAgent(task_competence=tc, seed=1).solve(
            ps, max_steps=300
        )
        assert res.solved
        # The compiled skill publishes the same normalized axis —
        # perceived evidence, not oracle vocabulary.
        assert tc.skills

    def test_teach_names_what_was_shown(self):
        """teach() binds each object's true shape word to its view —
        graceful without an embedding store."""
        from genesis_cognitive.sorter import PerceptualSorter

        base = _sorter(2, 13, difficulty=1)
        ps = PerceptualSorter(
            base.slots,
            [*base.pool.values()],
            cortex=self._cortex(),
            seed=4,
        )
        ps.teach()  # no embeddings — must not raise
        for name in ps._names.values():
            assert isinstance(name, str)

    def test_practice_runs_perceptual_offer(self, tmp_path):
        """practice.attempt drives the perceptual path when a cortex
        is wired and the spec asks for it."""
        practice = SpatialPractice(str(tmp_path))
        practice.cortex = self._cortex()
        from genesis_cognitive.concepts import ConceptNetwork
        from genesis_cognitive.spatial.solver import SpatialReasoner

        reasoner = SpatialReasoner(ConceptNetwork())
        task = normalize_offered(
            {
                "family": "sorter",
                "perceptual": True,
                "slots": [{"accepts": {"shape": "square"}}],
                "blocks": [
                    {"attrs": {"shape": "square"}},
                    {"attrs": {"shape": "triangle"}},
                ],
            },
            "seen_sorter",
        )
        assert task is not None and task["perceptual"]
        result = practice.attempt(reasoner, task=task)
        assert result is not None and result.solved
        assert result.details.get("placements")

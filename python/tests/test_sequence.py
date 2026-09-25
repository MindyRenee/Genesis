"""Pattern-line world + competence adapter.

The sequence family is "continue what repeats": a hidden periodic
rule, scaffolding, a token pool, and per-placement acceptance. The
agent perceives only a candidate's lookback — discovering which
lookback predicts acceptance IS discovering the period.
"""

from __future__ import annotations

import random

from genesis_cognitive.reasoning import TaskCompetence
from genesis_cognitive.sequence import PatternAgent, PatternLine


def _line(period: int, length: int, seed: int, **kw) -> PatternLine:
    return PatternLine.generate(period, length, random.Random(seed), **kw)


class TestPatternLine:
    """The world is honest: the rule checks, never reveals."""

    def test_place_accepts_only_the_rule(self):
        line = PatternLine(
            expected=["r", "b", "r", "b"],
            cells=["r", "b", None, None],
            pool=[],
        )
        from genesis_cognitive.sequence import Token

        line.pool[0] = Token(0, "r")
        line.pool[1] = Token(1, "b")
        out = line.place(0, 2)
        assert out is not None and out["accepted"]
        out = line.place(0, 3)  # token 0 already placed
        assert out is None
        out = line.place(1, 3)
        assert out is not None and out["accepted"]
        assert line.complete()

    def test_wrong_mark_rejected_and_reported(self):
        from genesis_cognitive.sequence import Token

        line = PatternLine(
            expected=["r", "b", "r"],
            cells=["r", "b", None],
            pool=[Token(0, "g")],
        )
        out = line.place(0, 2)
        assert out is not None and not out["accepted"]
        assert line.mismatches() == 1

    def test_lookback_is_perceptible(self):
        line = PatternLine(
            expected=["r", "b", "r", "b"],
            cells=["r", "b", None, None],
            pool=[],
        )
        assert line.lookback_of("b", 2) == 1
        assert line.lookback_of("r", 2) == 2
        assert line.lookback_of("g", 2) == 0

    def test_generate_contains_a_solution(self):
        for seed in range(8):
            line = _line(3, 9, seed, decoys=3)
            marks = [t.mark for t in line.pool.values()]
            for pos, cell in enumerate(line.cells):
                if cell is None:
                    assert line.expected[pos] in marks


class TestPatternCompetenceAdapter:
    """A sixth domain on the same substrate: continuation."""

    def test_agent_solves_and_consolidates_skill(self):
        competence = TaskCompetence()
        agent = PatternAgent(seed=1, task_competence=competence)
        result = agent.solve(_line(2, 6, 10, decoys=1), max_steps=400)

        assert result.solved
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        assert skill.verification == "external"
        steps = [s for s in skill.steps if s.family == "continuation"]
        assert steps
        assert all(
            set(s.parameters) == {"lookback", "mean_cost"}
            for s in steps
        )

    def test_discovers_the_period(self):
        """On an ABAB line, back-2 placements are the free ones."""
        competence = TaskCompetence()
        agent = PatternAgent(seed=1, task_competence=competence)
        result = agent.solve(_line(2, 6, 10, decoys=1), max_steps=400)
        assert result.solved

        skill = next(iter(competence.skills.values()))
        costs = {
            s.parameters["lookback"]: s.parameters["mean_cost"]
            for s in skill.steps
            if s.family == "continuation"
        }
        # The even lookbacks are correct on a period-2 line; 0 and
        # odd lookbacks are wasted moves.
        assert costs[2] == 0.0
        assert costs.get(0, 1.0) > 0.0 or costs.get(1, 1.0) > 0.0

    def test_prior_transfers_to_same_period(self):
        """A second pattern starts with the learned affordance as
        priors — the transfer channel is the competence skill."""
        competence = TaskCompetence()
        agent = PatternAgent(seed=1, task_competence=competence)
        cold = agent.solve(_line(2, 6, 10, decoys=2), max_steps=400)
        assert cold.solved
        warm = agent.solve(_line(2, 8, 21, decoys=2), max_steps=400)
        assert warm.solved
        # The consolidated skill was adopted: lookback priors loaded
        # at recognition time, before the first local observation.
        assert agent._look_priors
        assert 2 in agent._look_priors

    def test_wrong_period_prior_is_corrected_locally(self):
        """A period-2 prior misleads on a period-3 line — and local
        evidence overrides it (negative transfer, honestly handled)."""
        competence = TaskCompetence()
        agent = PatternAgent(seed=1, task_competence=competence)
        assert agent.solve(_line(2, 6, 10, decoys=1), 400).solved
        result = agent.solve(_line(3, 9, 30, decoys=1), max_steps=600)
        assert result.solved

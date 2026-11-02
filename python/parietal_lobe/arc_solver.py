"""ARC-AGI-1 bridge into Genesis' existing symbolic spatial reasoner.

No external model is used. Demonstration pairs are converted into Genesis
Grid/Example objects and solved by SpatialReasoner; only exactly verified
training rules produce predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from parietal_lobe.arc_sources import arc1_problem_view, load_arc1_task
from parietal_lobe.grid import Grid
from parietal_lobe.spatial import SpatialReasoner
from parietal_lobe.transforms import Example


@dataclass
class ARC1SolveResult:
    """Machine-readable result for one ARC-AGI-1 task."""

    task_path: str
    solved: bool
    predictions: list[list[list[int]]] = field(default_factory=list)
    # Up to ``num_guesses`` attempts, each a prediction per test input.
    guesses: list[list[list[list[int]]]] = field(default_factory=list)
    rule: str | None = None
    verified_rules: list[str] = field(default_factory=list)
    nodes_explored: int = 0
    failure: str | None = None


def solve_arc1_task(
    task_path: str | Path,
    *,
    reasoner: SpatialReasoner | None = None,
    max_depth: int = 2,
    beam_width: int = 32,
    max_nodes: int = 2000,
    time_budget: float = 30.0,
    num_guesses: int = 2,
) -> ARC1SolveResult:
    """Solve one task with Genesis' existing spatial reasoner.

    The solver sees training outputs and test inputs only. Test outputs are
    intentionally stripped before constructing the problem presented to it.
    A prediction is returned only when a rule exactly reproduces every
    demonstration output.
    """
    payload = load_arc1_task(task_path)
    problem = arc1_problem_view(payload)
    examples: list[Example] = [
        (Grid.from_lists(pair["input"]), Grid.from_lists(pair["output"]))
        for pair in problem["train"]
    ]
    tests = [Grid.from_lists(pair["input"]) for pair in problem["test"]]
    active_reasoner = reasoner or SpatialReasoner()
    solution = active_reasoner.solve(
        examples,
        tests,
        max_depth=max_depth,
        beam_width=beam_width,
        max_nodes=max_nodes,
        time_budget=time_budget,
        num_guesses=num_guesses,
    )
    return ARC1SolveResult(
        task_path=str(task_path),
        solved=solution.solved,
        predictions=[grid.to_lists() for grid in solution.predictions],
        guesses=[
            [grid.to_lists() for grid in guess]
            for guess in (getattr(solution, "guesses", None) or [solution.predictions])
            if guess
        ],
        rule=solution.hypothesis.describe() if solution.hypothesis else None,
        verified_rules=solution.verified_rules,
        nodes_explored=solution.nodes_explored,
        failure=solution.failure.describe() if solution.failure else None,
    )

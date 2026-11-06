"""Tests for the ARC-AGI-1 adapter into Genesis' spatial reasoner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from parietal_lobe.arc_solver import solve_arc1_task


def test_arc1_solver_uses_spatial_reasoner_and_returns_grid_predictions(tmp_path: Path) -> None:
    """Use Genesis' reasoner and ensure hidden test outputs do not leak in."""
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
                "test": [{"input": [[0, 2]], "output": [[9, 9]]}],
            }
        ),
        encoding="utf-8",
    )

    class FakeReasoner:
        """Record the problem passed to the reasoner and return a known result."""
        def solve(self, examples, tests, **kwargs):
            """Assert task conversion and return a deterministic fake solution."""
            assert len(examples) == 1
            assert examples[0][0].to_lists() == [[0, 1]]
            assert examples[0][1].to_lists() == [[1, 0]]
            assert tests[0].to_lists() == [[0, 2]]
            return SimpleNamespace(
                solved=True,
                predictions=[tests[0].reflect_h()],
                hypothesis=SimpleNamespace(describe=lambda: "reflect_h"),
                verified_rules=["reflect_h"],
                nodes_explored=3,
                failure=None,
            )

    result = solve_arc1_task(path, reasoner=FakeReasoner())
    assert result.solved is True
    assert result.predictions == [[[2, 0]]]
    assert result.rule == "reflect_h"
    assert result.nodes_explored == 3

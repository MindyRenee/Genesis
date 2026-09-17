"""Evaluation: ARC-style spatial abstraction.

Tests whether the perceptual-symbolic layer (genesis_cognitive.spatial)
can infer transformation rules from few-shot input/output examples —
the ARC-AGI task format. Each task provides 2-4 training pairs and one
or more test inputs; solving means finding a transform sequence that
reproduces every training output, then applying it to the tests.

Conditions:
1. Synthetic primitives — single-transform tasks the DSL provably
   covers (rotate, reflect, recolor, crop, scale, gravity). A failure
   here means the machinery is broken, not that the task is hard.
2. Synthetic compositions — two-step rules (crop → recolor, rotate →
   recolor). Tests the solver's depth-2 search.
3. ARC-AGI-1 subset — real tasks loaded from evals/arc_tasks/*.json
   when present (download from the arcprize ARC-AGI repo). Skipped
   when the directory is empty, so the eval always runs.

Scoring follows ARC convention: a task is solved only if every test
prediction exactly matches the expected output.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from genesis_cognitive.spatial import Grid, SpatialReasoner
from harness import (
    EvalResult,
    FactResult,
    print_result,
    run_condition,
    score_arc_solution,
)

_TASKS_DIR = Path(__file__).resolve().parent / "arc_tasks"


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


# ─── Synthetic tasks ─────────────────────────────────────────────
#
# Each task is (name, train_pairs, test_pairs) where pairs are
# (input, expected_output). These are hand-built to exercise the
# transformation DSL — they are seeds for the machinery, not Genesis
# utterances.

_PRIMITIVE_TASKS: list[tuple[str, list, list]] = [
    (
        "rotate90",
        [([[1, 0], [0, 0]], [[0, 1], [0, 0]])],
        [([[0, 0], [1, 0]], [[1, 0], [0, 0]])],
    ),
    (
        "reflect_h",
        [([[1, 2, 0], [0, 0, 0]], [[0, 2, 1], [0, 0, 0]])],
        [([[3, 0, 0]], [[0, 0, 3]])],
    ),
    (
        "recolor",
        [([[1, 1], [0, 0]], [[5, 5], [0, 0]])],
        [([[1, 0]], [[5, 0]])],
    ),
    (
        "crop_to_content",
        [
            ([[0, 0, 0], [0, 4, 4], [0, 4, 4]], [[4, 4], [4, 4]]),
            ([[0, 0], [6, 0]], [[6]]),
        ],
        [([[0, 9, 0], [0, 0, 0]], [[9]])],
    ),
    (
        "scale2",
        [([[1, 0]], [[1, 1, 0, 0], [1, 1, 0, 0]])],
        [([[7]], [[7, 7], [7, 7]])],
    ),
    (
        "gravity_down",
        [
            (
                [[0, 0, 0], [0, 3, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 3, 0]],
            ),
        ],
        [
            ([[2, 0], [0, 0]], [[0, 0], [2, 0]]),
        ],
    ),
]

_COMPOSITION_TASKS: list[tuple[str, list, list]] = [
    (
        "crop_then_recolor",
        [
            (
                [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                [[8, 8], [8, 8]],
            ),
            ([[0, 0], [0, 2]], [[8]]),
        ],
        [([[0, 0], [0, 2]], [[8]])],
    ),
    (
        "rotate_then_recolor",
        [
            ([[1, 0], [0, 0]], [[0, 7], [0, 0]]),
            # Second example rules out reflect_h, which also explains
            # the first pair on a single-cell input.
            ([[1, 1], [0, 0]], [[0, 7], [0, 7]]),
        ],
        [([[0, 1], [0, 0]], [[0, 0], [0, 7]])],
    ),
]


def _run_tasks(
    tasks: list[tuple[str, list, list]],
    reasoner: SpatialReasoner,
    max_depth: int,
) -> list[FactResult]:
    """Solve each task; a task passes iff every prediction is exact."""
    results: list[FactResult] = []
    for name, train_raw, test_raw in tasks:
        examples = [(_g(i), _g(o)) for i, o in train_raw]
        tests = [_g(i) for i, _ in test_raw]
        expected = [_g(o) for _, o in test_raw]

        sol = reasoner.solve(
            examples, tests, max_depth=max_depth, time_budget=15.0
        )
        passed, detail = score_arc_solution(sol, expected)
        results.append(FactResult(name, passed, detail))
    return results


def _load_arc_tasks(limit: int) -> list[tuple[str, list, list]]:
    """Load ARC-AGI-1 JSON tasks from evals/arc_tasks/ if present.

    Expected format (the official corpus layout):
        {"train": [{"input": [[...]], "output": [[...]]}, ...],
         "test":  [{"input": [[...]], "output": [[...]]}, ...]}
    """
    tasks: list[tuple[str, list, list]] = []
    if not _TASKS_DIR.is_dir():
        return tasks
    # Filter for validity first, THEN apply the limit — slicing before
    # filtering yields fewer than `limit` valid tasks when early files
    # are invalid.
    for path in sorted(_TASKS_DIR.glob("*.json")):
        if len(tasks) >= limit:
            break
        try:
            data = json.loads(path.read_text())
            train = [(p["input"], p["output"]) for p in data["train"]]
            test = [
                (p["input"], p["output"])
                for p in data["test"]
                if "output" in p
            ]
            if train and test:
                tasks.append((path.stem, train, test))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return tasks


def run(seed: int = 42, arc_task_limit: int = 50) -> EvalResult:
    """Run the ARC-style spatial abstraction evaluation."""
    del seed  # spatial search is deterministic; seed kept for API parity
    start = time.time()
    result = EvalResult(
        name="arc_spatial",
        description=(
            "Few-shot spatial abstraction: infer grid transformation "
            "rules from examples (ARC-AGI task format)"
        ),
    )

    result.conditions.append(
        run_condition(
            "synthetic_primitives",
            "Single-transform tasks the DSL provably covers",
            lambda: _run_tasks(_PRIMITIVE_TASKS, SpatialReasoner(), max_depth=1),
        )
    )
    result.conditions.append(
        run_condition(
            "synthetic_compositions",
            "Two-step rules (crop→recolor, rotate→recolor)",
            lambda: _run_tasks(_COMPOSITION_TASKS, SpatialReasoner(), max_depth=2),
        )
    )

    arc_tasks = _load_arc_tasks(limit=arc_task_limit)
    if arc_tasks:
        result.conditions.append(
            run_condition(
                "arc_agi_1_subset",
                f"Real ARC-AGI-1 tasks from {_TASKS_DIR} "
                f"({len(arc_tasks)} loaded)",
                # Fresh reasoner per condition: _learned rules must not
                # leak across conditions and inflate later scores.
                lambda: _run_tasks(arc_tasks, SpatialReasoner(), max_depth=2),
            )
        )

    result.duration_seconds = round(time.time() - start, 3)
    total = sum(c.total for c in result.conditions)
    correct = sum(c.correct for c in result.conditions)
    result.overall_accuracy = correct / total if total else 0.0
    result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


if __name__ == "__main__":
    print_result(run())

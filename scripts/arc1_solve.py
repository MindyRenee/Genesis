#!/usr/bin/env python3
"""Run one ARC-AGI-1 task through Genesis' symbolic spatial reasoner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The project stores the brain-region packages under python/.
ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from parietal_lobe.arc_solver import solve_arc1_task  # noqa: E402
from parietal_lobe.arc_sources import (  # noqa: E402
    ARCSourceError,
    list_arc1_tasks,
    load_arc1_task,
)


def score_guesses(guesses: list, expected: list) -> bool | None:
    """Official ARC-AGI-1 scoring: some guess matches every test output.

    Returns None when the task file carries no test outputs.
    """
    if not expected or any(output is None for output in expected):
        return None
    return any(guess == expected for guess in guesses)


def main() -> int:
    """Parse CLI options, solve one task, and emit a JSON result.

    Returns exit code 0 for a verified solve, 1 when no verified solution is
    found, and 2 when task loading or setup fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        help="Task JSON path; if omitted, --split and --index select a dataset task.",
    )
    parser.add_argument("--split", choices=("training", "evaluation"), default="training")
    parser.add_argument("--index", type=int, default=0, help="Zero-based task index.")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--time-budget", type=float, default=30.0)
    parser.add_argument("--num-guesses", type=int, default=2, help="Attempts per task (pass@k).")
    args = parser.parse_args()

    try:
        if args.task:
            task_path = Path(args.task).expanduser()
        else:
            tasks = list_arc1_tasks(args.split)
            if args.index < 0 or args.index >= len(tasks):
                parser.error(f"--index must be in [0, {len(tasks) - 1}] for {args.split}")
            task_path = tasks[args.index]

        result = solve_arc1_task(
            task_path,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            max_nodes=args.max_nodes,
            time_budget=args.time_budget,
            num_guesses=args.num_guesses,
        )
        expected = [pair.get("output") for pair in load_arc1_task(task_path)["test"]]
    except (ARCSourceError, OSError, ValueError) as exc:
        print(f"ARC solve error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "task_path": result.task_path,
        "solved": result.solved,
        "rule": result.rule,
        "predictions": result.predictions,
        "guesses": result.guesses,
        "correct_first_guess": score_guesses(result.guesses[:1], expected),
        f"correct_within_{args.num_guesses}_guesses": score_guesses(result.guesses, expected),
        "verified_rules": result.verified_rules,
        "nodes_explored": result.nodes_explored,
        "failure": result.failure,
    }, indent=2))
    return 0 if result.solved else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score Genesis' ARC-AGI-1 solver on a whole split with official pass@k rules."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from parietal_lobe.arc_solver import solve_arc1_task  # noqa: E402
from parietal_lobe.arc_sources import list_arc1_tasks, load_arc1_task  # noqa: E402


def score_task(task_path: Path, *, time_budget: float, num_guesses: int) -> dict:
    """Solve one task and classify it as pass@1, pass@k, wrong, or unsolved."""
    expected = [pair["output"] for pair in load_arc1_task(task_path)["test"]]
    result = solve_arc1_task(task_path, time_budget=time_budget, num_guesses=num_guesses)
    first = bool(result.guesses) and result.guesses[0] == expected
    any_guess = any(guess == expected for guess in result.guesses)
    return {
        "task": task_path.stem,
        "solved": result.solved,
        "pass1": first,
        "passk": any_guess,
        "rule": result.rule,
    }


def main() -> int:
    """Run the split in parallel and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("training", "evaluation"), default="training")
    parser.add_argument("--time-budget", type=float, default=3.0)
    parser.add_argument("--num-guesses", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Only score the first N tasks.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--details", action="store_true", help="Also list per-task results.")
    args = parser.parse_args()

    tasks = list_arc1_tasks(args.split)[: args.limit]
    run = partial(score_task, time_budget=args.time_budget, num_guesses=args.num_guesses)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, tasks))
    summary = {
        "split": args.split,
        "tasks": len(rows),
        "pass@1": sum(r["pass1"] for r in rows),
        f"pass@{args.num_guesses}": sum(r["passk"] for r in rows),
        "unsolved": sum(not r["solved"] for r in rows),
        "verified_but_wrong": sum(r["solved"] and not r["passk"] for r in rows),
        "wrong_rules": [
            f'{r["task"]}: {r["rule"]}' for r in rows if r["solved"] and not r["passk"]
        ],
    }
    if args.details:
        summary["results"] = rows
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

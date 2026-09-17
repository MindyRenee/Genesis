#!/usr/bin/env python3
"""Sequential cold evaluation of the ARC-AGI-1 corpus — one task at a
time, fresh reasoner per task, single attempt, no lessons or retries.

This is the leaderboard-honest number: the same solver and transform
library Genesis uses in practice, but evaluated task-by-task with no
miss→primitive→retry conversions. Results append to
evals/results/arc_sequential.jsonl after every task, so a partial run
still leaves a record.

Usage:
    PYTHONPATH=python python3 evals/eval_arc_sequential.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from genesis_cognitive.spatial import Grid, SpatialReasoner
from harness import score_arc_solution

_TASKS_DIR = Path(__file__).resolve().parent / "arc_tasks"
_OUT = Path(__file__).resolve().parent / "results" / "arc_sequential.jsonl"


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if _OUT.exists():
        for line in _OUT.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["task"])
            except (json.JSONDecodeError, KeyError, TypeError):
                # One corrupt resume line must not abort the whole run.
                continue

    tasks = sorted(_TASKS_DIR.glob("*.json"))
    for path in tasks:
        name = path.stem
        if name in done:
            continue
        data = json.loads(path.read_text())
        train = [(_g(p["input"]), _g(p["output"])) for p in data["train"]]
        tests = [_g(p["input"]) for p in data["test"]]
        expected = [_g(p["output"]) for p in data["test"] if "output" in p]

        start = time.monotonic()
        sol = SpatialReasoner().solve(train, tests, time_budget=15.0)
        # Single shared scorer — same rule as eval_arc.py and
        # eval_arc_transfer.py (solved + hypothesis + length-exact match).
        solved, _detail = score_arc_solution(sol, expected)
        record = {
            "task": name,
            "solved": solved,
            "rule": sol.hypothesis.describe() if sol.hypothesis else None,
            "nodes": sol.nodes_explored,
            "seconds": round(time.monotonic() - start, 2),
        }
        with _OUT.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(record, flush=True)


if __name__ == "__main__":
    main()

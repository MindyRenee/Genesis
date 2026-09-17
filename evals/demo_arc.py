#!/usr/bin/env python3
"""One-task-at-a-time ARC demo — live mind, teach mode, learned rules.

Same protocol as evals/eval_arc_transfer.py: a live Mind (offline, no
daemon), her real SpatialReasoner attached to her real concept
network. Teaching happens through teach mode; corrections use
exclude_rules — rules verified on train but wrong on test are proven
counterexamples and excluded on retry (her SpatialPractice does the
same thing).

Each step waits for Enter — narrate between steps.

Run:  PYTHONPATH=python:evals python3 evals/demo_arc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from genesis_cognitive.spatial import Grid, SpatialReasoner
from genesis_cognitive.spatial.practice import CURRICULUM
from genesis_cognitive.spatial.transforms import instantiate
from harness import EvalMind

ARC_DIR = Path(__file__).parent / "arc_tasks"

_ANSI = {
    0: 16, 1: 12, 2: 9, 3: 10, 4: 11,
    5: 8, 6: 13, 7: 208, 8: 14, 9: 52,
}


def show(grid: Grid, label: str = "") -> None:
    if label:
        print(f"    {label}")
    for row in grid.to_lists():
        cells = "".join(
            f"\x1b[48;5;{_ANSI.get(v, 15)}m  \x1b[0m" for v in row
        )
        print(f"    {cells}")
    print()


def curriculum_task(name: str) -> dict:
    return next(t for t in CURRICULUM if t["name"] == name)


def arc_task(task_id: str) -> tuple[list, list]:
    data = json.loads((ARC_DIR / f"{task_id}.json").read_text())
    train = [
        (Grid.from_lists(p["input"]), Grid.from_lists(p["output"]))
        for p in data["train"]
    ]
    test = [
        (Grid.from_lists(p["input"]), Grid.from_lists(p["output"]))
        for p in data["test"]
    ]
    return train, test


def pause(msg: str) -> None:
    input(f"\n  ── {msg} ──\n")


def wrong_rules(
    reasoner: SpatialReasoner, tests: list, expected: list
) -> set[str]:
    """Learned rules whose predictions miss the test — counterexamples."""
    bad = set()
    for t in reasoner.learned_rules:
        try:
            if any(t(g) != e for g, e in zip(tests, expected, strict=True)):
                bad.add(t.describe())
        except Exception:  # noqa: BLE001
            continue
    return bad


def attempt(
    reasoner: SpatialReasoner,
    name: str,
    train: list,
    test: list,
    budget: float = 20.0,
    exclude: set[str] | None = None,
) -> object:
    print(f"\n  ══ {name} ══")
    for i, (inp, out) in enumerate(train):
        show(inp, f"train {i + 1} input")
        show(out, f"train {i + 1} output")
    tests = [t for t, _ in test]
    expected = [o for _, o in test]
    show(tests[0], "test input — what should this become?")

    sol = reasoner.solve(
        train, tests, max_depth=2, time_budget=budget,
        exclude_rules=exclude,
    )
    if not sol.hypothesis:
        print(f"  unsolved — no verified rule [{sol.nodes_explored} nodes]")
        return sol
    print(
        f"  her rule: {sol.hypothesis.describe()}  "
        f"[{sol.nodes_explored} nodes explored]"
    )
    pred = sol.predictions
    show(pred[0], "her answer")
    show(expected[0], "expected")
    ok = all(p == e for p, e in zip(pred, expected, strict=True))
    print(f"  → {'SOLVED' if ok else 'WRONG'}")
    return sol


def main() -> None:
    print("Booting a live Mind (offline — her real reasoner and "
          "concept network)...")
    mind = EvalMind()
    r: SpatialReasoner = mind.mind.cognition.spatial
    print(f"  learned rules at start: {len(r.learned_rules)}")

    # ── 1. A task she hasn't learned ─────────────────────────────
    pause("step 1 — uniform_column, never attempted")
    t = curriculum_task("uniform_column")
    sol = attempt(r, t["name"], t["train"], t["test"])

    # ── 2. Teach mode: correct the wrong rule, teach the right one
    pause("step 2 — teach mode: 'not a recolor — mark the uniform column'")
    mind.mind.enter_teaching_mode("spatial reasoning")
    tests = [g for g, _ in t["test"]]
    expected = [o for _, o in t["test"]]
    bad = wrong_rules(r, tests, expected)
    if sol.hypothesis:
        bad.add(sol.hypothesis.describe())
    print(f"  rules proven wrong by the test: {bad}")
    tr_obj = instantiate("mark_uniform", {"axis": "column", "color": 5})
    r.learn([tr_obj], taught=True)
    mind.teach(
        "mark_uniform is a spatial transformation rule: the line where "
        "every cell is the same color gets marked; everything else "
        "becomes background."
    )
    print(f"  learned rules now: {[x.describe() for x in r.learned_rules]}")
    mind.mind.exit_teaching_mode()

    # ── 3. Retry with the counterexample excluded ────────────────
    pause("step 3 — same puzzle again, wrong answers excluded")
    attempt(r, t["name"], t["train"], t["test"], exclude=bad)

    # ── 4. Real ARC task, never seen — and a cold baseline ───────
    pause("step 4 — real ARC-AGI-1 task 25d8a9c8, never seen")
    train, test = arc_task("25d8a9c8")
    cold = SpatialReasoner()
    print("  cold baseline first (no learned rules):")
    attempt(cold, "arc:25d8a9c8 — cold", train, test, budget=20.0)
    print("  now her trained reasoner:")
    attempt(r, "arc:25d8a9c8 — trained", train, test, budget=20.0)

    # ── 5. One she knows cold ────────────────────────────────────
    pause("step 5 — gravity_intro, mastered long ago")
    g = curriculum_task("gravity_intro")
    attempt(r, g["name"], g["train"], g["test"], budget=10.0)

    print(f"\n  final learned rules: {[x.describe() for x in r.learned_rules]}")
    mind.cleanup()


if __name__ == "__main__":
    main()

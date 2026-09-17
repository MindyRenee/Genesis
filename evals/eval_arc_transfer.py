"""Evaluation: ARC train→eval transfer protocol.

The honest protocol for the ARC work:

    TRAIN PHASE (teach mode): Genesis's live mind enters teaching
    mode. Her spatial reasoner — attached to her real concept
    network — attempts training tasks. Every rule she verifies is
    retained via learn() and grounded into her concept network as
    spatial:rule:* concepts. Linguistic seeds about each rule are
    also taught through the normal learning path so her language
    engine has vocabulary for what she knows.

    EVAL PHASE (no teaching): the SAME reasoner, carrying everything
    it learned, attempts tasks it has never seen. A parallel "cold"
    reasoner with no training attempts the same tasks as the baseline.

The transfer signal: held-out accuracy of the trained reasoner vs.
the cold one. This measures whether experience actually helps — the
property no single-task benchmark captures.

Protocol rules (no contamination):
- Training tasks and eval tasks are disjoint (split of evals/arc_tasks).
- Eval tasks are never taught and never shown to the trained reasoner
  before the eval phase.
- Teaching is allowed only during the train phase, in teach mode —
  developmental history, not answer leakage.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from genesis_cognitive.spatial import Grid, SpatialReasoner
from harness import (
    EvalMind,
    EvalResult,
    FactResult,
    print_result,
    run_condition,
    score_arc_solution,
)

_TASKS_DIR = Path(__file__).resolve().parent / "arc_tasks"

# Disjoint split: first N task files are training, rest are held out.
_TRAIN_COUNT = 40

# ─── Synthetic rule families ─────────────────────────────────────
#
# Variants of the same underlying rule with different surface forms —
# the condition under which learned-rule reuse can actually show up.
# Family tasks are split: some instances are taught in training, held-
# out instances test whether the *rule* transferred.

# Synthetic rule families: each task has multiple training pairs with
# varied colors, dims, and distractor cells so that degenerate
# explanations (a global recolor, a reflection) cannot fit every
# example — only the intended abstract rule can.
_FAMILY_TRAIN: list[tuple[str, list, list]] = [
    (
        # Fill enclosed region with 4. Example 2 surrounds the box
        # with outside background — a global 0→4 recolor would flood
        # the exterior, so only the region-aware rule fits.
        "fam_fill_box_1",
        [
            (
                [[2, 2, 2], [2, 0, 2], [2, 2, 2]],
                [[2, 2, 2], [2, 4, 2], [2, 2, 2]],
            ),
            (
                [[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 0, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 4, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]],
            ),
        ],
        [(
            [[5, 5, 5, 5, 5], [5, 0, 0, 0, 5], [5, 5, 5, 5, 5]],
            [[5, 5, 5, 5, 5], [5, 4, 4, 4, 5], [5, 5, 5, 5, 5]],
        )],
    ),
    (
        # Same rule, different surface: fill color 8, other box colors,
        # other dimensions.
        "fam_fill_box_2",
        [
            (
                [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
                [[1, 1, 1, 1], [1, 8, 8, 1], [1, 8, 8, 1], [1, 1, 1, 1]],
            ),
            (
                [[0, 0, 0, 0, 0], [0, 6, 6, 6, 0], [0, 6, 0, 6, 0],
                 [0, 6, 6, 6, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 6, 6, 6, 0], [0, 6, 8, 6, 0],
                 [0, 6, 6, 6, 0], [0, 0, 0, 0, 0]],
            ),
        ],
        [(
            [[4, 4, 4], [4, 0, 4], [4, 4, 4]],
            [[4, 4, 4], [4, 8, 4], [4, 4, 4]],
        )],
    ),
    (
        # Gravity: objects fall to the bottom. Example 1's object sits
        # mid-grid — a vertical reflection would move it *up*, so only
        # downward motion explains both examples.
        "fam_gravity_1",
        [
            (
                [[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0]],
            ),
            (
                [[0, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 2, 0]],
            ),
        ],
        [(
            [[0, 0, 0], [4, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [4, 0, 0]],
        )],
    ),
    (
        "fam_gravity_2",
        [
            (
                [[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 3, 0, 0]],
            ),
            (
                [[6, 6, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [6, 6, 0]],
            ),
        ],
        [(
            [[0, 0, 0], [0, 0, 5], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 5]],
        )],
    ),
]

_FAMILY_HELDOUT: list[tuple[str, list, list]] = [
    (
        "fam_fill_box_heldout",
        [
            (
                [[6, 6, 6], [6, 0, 6], [6, 6, 6]],
                [[6, 6, 6], [6, 2, 6], [6, 6, 6]],
            ),
            (
                [[7, 7, 7, 7], [7, 0, 0, 7], [7, 7, 7, 7], [0, 0, 0, 0]],
                [[7, 7, 7, 7], [7, 2, 2, 7], [7, 7, 7, 7], [0, 0, 0, 0]],
            ),
        ],
        [(
            [[8, 8, 8, 8], [8, 0, 0, 8], [8, 0, 0, 8], [8, 8, 8, 8]],
            [[8, 8, 8, 8], [8, 2, 2, 8], [8, 2, 2, 8], [8, 8, 8, 8]],
        )],
    ),
    (
        "fam_gravity_heldout",
        [
            (
                [[0, 0, 0], [0, 0, 0], [5, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [5, 0, 0]],
            ),
            (
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
            ),
        ],
        [(
            [[0, 8, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 8, 0]],
        )],
    ),
]


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


def _load_tasks() -> list[tuple[str, list, list]]:
    """Load all ARC tasks (name, train_pairs, test_pairs)."""
    import json

    tasks = []
    for path in sorted(_TASKS_DIR.glob("*.json")):
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


def _attempt(
    tasks: list[tuple[str, list, list]],
    reasoner: SpatialReasoner,
    time_budget: float,
) -> list[FactResult]:
    """Attempt each task; pass iff all test predictions are exact."""
    results = []
    for name, train_raw, test_raw in tasks:
        examples = [(_g(i), _g(o)) for i, o in train_raw]
        tests = [_g(i) for i, _ in test_raw]
        expected = [_g(o) for _, o in test_raw]
        sol = reasoner.solve(
            examples, tests, max_depth=2, time_budget=time_budget
        )
        # Single shared scorer (harness.score_arc_solution): solved +
        # hypothesis + length-exact top-k match.
        ok, detail = score_arc_solution(sol, expected)
        if ok:
            detail = (
                f"rule: {sol.hypothesis.describe()} "
                f"[nodes={sol.nodes_explored}]"
            )
        else:
            detail = (
                f"{detail} [nodes={sol.nodes_explored}]"
                if sol.hypothesis
                else "none"
            )
        results.append(FactResult(name, ok, detail))
    return results


def _count_grounded_rules(network: object) -> int:
    return sum(
        1
        for name in network.iter_concept_names()
        if name.startswith("spatial:rule:")
    )


def run(seed: int = 42) -> EvalResult:
    """Run the train→eval transfer protocol."""
    start = time.time()
    tasks = _load_tasks()
    # Seeded shuffle before the split: filenames are arbitrary IDs, so
    # an alphabetical first-N split can bias train toward one family.
    # Shuffling with the eval seed keeps the split reproducible.
    rng = random.Random(seed)
    rng.shuffle(tasks)
    train_tasks = _FAMILY_TRAIN + tasks[:_TRAIN_COUNT]
    eval_tasks = _FAMILY_HELDOUT + tasks[_TRAIN_COUNT:]

    result = EvalResult(
        name="arc_transfer",
        description=(
            "Train in teach mode on seen tasks, evaluate on held-out "
            "tasks — does learned experience transfer?"
        ),
    )

    # ── TRAIN PHASE: teach mode, her real concept network ──
    train_stats: dict[str, object] = {"rules": 0, "solved": 0}
    with EvalMind(seed=seed) as mind:
        mind.mind.enter_teaching_mode("spatial reasoning")
        trained = mind.mind.cognition.spatial  # her real reasoner

        def _train() -> list[FactResult]:
            facts = _attempt(train_tasks, trained, time_budget=15.0)
            train_stats["solved"] = sum(1 for f in facts if f.passed)
            # Linguistic seeds: name what she learned, through the
            # normal teaching path — vocabulary, not utterances.
            for t in trained.learned_rules:
                mind.teach(
                    f"{t.describe()} is a spatial transformation rule."
                )
            train_stats["rules"] = len(trained.learned_rules)
            return facts

        result.conditions.append(
            run_condition(
                "train_phase_teach_mode",
                f"{len(train_tasks)} training tasks, rules grounded "
                "into her concept network",
                _train,
            )
        )
        mind.mind.exit_teaching_mode()
        train_stats["grounded"] = _count_grounded_rules(
            mind.mind.cognition.network
        )

        # ── EVAL PHASE: held-out tasks, trained reasoner ──
        result.conditions.append(
            run_condition(
                "held_out_trained",
                f"{len(eval_tasks)} unseen tasks, learned rules available",
                lambda: _attempt(eval_tasks, trained, time_budget=15.0),
            )
        )

    # ── COLD BASELINE: fresh reasoner, no training ──
    cold = SpatialReasoner()
    result.conditions.append(
        run_condition(
            "held_out_cold_baseline",
            f"{len(eval_tasks)} unseen tasks, no prior learning",
            lambda: _attempt(eval_tasks, cold, time_budget=15.0),
        )
    )

    result.duration_seconds = round(time.time() - start, 3)
    # Headline = transfer signal: mean of the two held-out conditions
    # (trained vs cold on unseen tasks). Including the train phase in
    # the average would inflate the headline with already-seen tasks.
    held_out = [c for c in result.conditions if c.name.startswith("held_out_")]
    held_total = sum(c.total for c in held_out)
    held_correct = sum(c.correct for c in held_out)
    result.overall_accuracy = held_correct / held_total if held_total else 0.0
    result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"  [train] solved {train_stats['solved']}, learned "
        f"{train_stats['rules']} rules, grounded "
        f"{train_stats['grounded']} concepts"
    )
    return result


if __name__ == "__main__":
    print_result(run())

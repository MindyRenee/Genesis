#!/usr/bin/env python3
"""Runner for the Genesis evaluation suite.

Runs all evaluations and produces:
- Per-eval JSON results in evals/results/
- An aggregate CSV for longitudinal tracking
- A human-readable summary on stdout

Usage:
    python3 evals/run_evals.py [--seed N] [--json] [--csv]

Options:
    --seed N   Random seed for reproducibility (default: 42).
    --json     Write per-eval JSON results to evals/results/.
    --csv      Append aggregate results to evals/results/eval_history.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure evals/ is importable so eval modules can `from harness import ...`
_EVALS_DIR = Path(__file__).resolve().parent
if str(_EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALS_DIR))

# Ensure python/ is importable so eval modules can import genesis_cognitive.
_PYTHON_DIR = _EVALS_DIR.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import eval_arc  # noqa: E402
import eval_cognitive_trajectory  # noqa: E402
import eval_emotion_gated  # noqa: E402
import eval_generalization  # noqa: E402
import eval_metacognitive  # noqa: E402
import eval_teaching  # noqa: E402
from harness import EvalResult, append_csv, print_result, write_json  # noqa: E402

# Registry of all evaluations run by default.
# Each entry is (name, module.run_function).
# Excluded by design (run separately): eval_arc3 (needs the arc-agi
# toolkit venv), eval_arc_sequential / eval_arc_transfer (long-running
# ARC sweeps with their own drivers), teach_spatial_live (interactive).
_EVALS = [
    ("teaching", eval_teaching.run),
    ("generalization", eval_generalization.run),
    ("emotion_gated", eval_emotion_gated.run),
    ("cognitive_trajectory", eval_cognitive_trajectory.run),
    ("metacognitive", eval_metacognitive.run),
    ("arc_spatial", eval_arc.run),
]


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run the Genesis evaluation suite.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--json", action="store_true", help="Write per-eval JSON.")
    parser.add_argument("--csv", action="store_true", help="Append to CSV history.")
    args = parser.parse_args()

    results_dir = _EVALS_DIR / "results"
    all_results: list[EvalResult] = []

    for name, run_fn in _EVALS:
        print(f"\n{'─' * 60}")
        print(f"  Running: {name}")
        print(f"{'─' * 60}")
        result = run_fn(seed=args.seed)
        all_results.append(result)
        print_result(result)

        if args.json:
            json_path = results_dir / f"{name}.json"
            write_json(result, json_path)
            print(f"  JSON written to: {json_path}")

        if args.csv:
            csv_path = results_dir / "eval_history.csv"
            append_csv(result, csv_path)
            print(f"  CSV appended to: {csv_path}")

    # Overall summary
    print(f"\n{'═' * 60}")
    print("  Evaluation Suite Summary")
    print(f"{'═' * 60}")
    for result in all_results:
        status = "PASS" if result.overall_accuracy >= 0.6 else "FAIL"
        print(f"  [{status}] {result.name}: {result.overall_accuracy:.1%}")
    print()

    # Exit code: 0 if all evals pass, 1 if any fail
    all_pass = all(r.overall_accuracy >= 0.6 for r in all_results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

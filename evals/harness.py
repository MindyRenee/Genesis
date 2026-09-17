"""Shared infrastructure for the Genesis evaluation suite.

Provides:
- Mind setup in an isolated temp directory (no daemon required)
- Fact teaching and recall-checking helpers
- Structured result logging (JSON per-eval, CSV aggregate for tracking)
- A common Fact/Condition/Result data model

The harness is intentionally framework-free — no pytest, no unittest.
Evals are plain Python scripts that import this harness, run their
conditions, and return EvalResult objects. The runner (run_evals.py)
collects results and writes them to disk.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

# Ensure the python/ package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from genesis_cognitive.concepts import ConceptNetwork, RelationType  # noqa: E402
from genesis_cognitive.mind import Mind  # noqa: E402
from genesis_cognitive.reasoning.engine import ReasoningEngine  # noqa: E402

# ─── Data model ───────────────────────────────────────────────────


@dataclass
class Fact:
    """A single teachable fact.

    Attributes:
        teach_input: the natural-language teaching statement.
        concept: the concept name that should appear in the network.
        expected_substrings: substrings that must appear in the
            concept's definition after learning.
        relation_type: optional RelationType that should exist.
        relation_target: optional target concept for the relation.
        question: the question to ask the reasoning engine (e.g.
            "what is a tardigrade?").
        question_type: the reasoning question type ("what_is",
            "what_causes", etc.).
        expected_answer_substrings: substrings that should appear in
            the reasoning engine's answer conclusion.
    """

    teach_input: str
    concept: str
    expected_substrings: tuple[str, ...]
    relation_type: RelationType | None = None
    relation_target: str | None = None
    question: str = ""
    question_type: str = ""
    expected_answer_substrings: tuple[str, ...] = ()


@dataclass
class FactResult:
    """Result of checking a single fact."""

    concept: str
    passed: bool
    detail: str


@dataclass
class ConditionResult:
    """Result of one condition within an eval."""

    name: str
    description: str
    accuracy: float
    correct: int
    total: int
    facts: list[FactResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Top-level result of one evaluation."""

    name: str
    description: str
    conditions: list[ConditionResult] = field(default_factory=list)
    overall_accuracy: float = 0.0
    duration_seconds: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the results to a dict."""
        return asdict(self)


# ─── Mind setup ───────────────────────────────────────────────────


class EvalMind:
    """A Mind instance configured for evaluation.

    Wraps Mind with eval-specific setup: isolated temp directory,
    no daemon connection, deterministic seed, and convenience
    helpers for teaching, checking, and reasoning.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the eval mind."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.data_dir = self._tmpdir.name
        self.socket_path = os.path.join(self.data_dir, "genesis.sock")
        self.mind = Mind(self.socket_path, seed=seed)
        self.network: ConceptNetwork = self.mind.cognition.network
        self.reasoning = ReasoningEngine(self.network)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._tmpdir.cleanup()

    def __enter__(self) -> Self:
        """Enter the context manager, returning the harness instance."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager, cleaning up temporary resources."""
        self.cleanup()

    # ── Teaching ──

    def teach(self, text: str) -> int:
        """Teach a single statement. Returns the number of learning events."""
        events = self.mind.cognition.self_learner.learn_from_input(text)
        return len(events)

    def teach_all(self, facts: list[Fact]) -> None:
        """Teach all facts, then run one inference cycle to consolidate."""
        for fact in facts:
            self.teach(fact.teach_input)
        self.mind.cognition.self_learner.run_inference_cycle(force=True)

    def teach_fillers(self, fillers: list[str]) -> None:
        """Teach filler facts (for interference testing)."""
        for text in fillers:
            self.teach(text)
        self.mind.cognition.self_learner.run_inference_cycle(force=True)

    # ── Recall checking ──

    def check_fact(self, fact: Fact) -> FactResult:
        """Check if a fact was learned: concept exists, definition
        contains expected substrings, and the expected relation
        exists if specified."""
        concept = self.network.get_concept(fact.concept)
        if concept is None:
            return FactResult(
                concept=fact.concept, passed=False,
                detail=f"concept '{fact.concept}' not found",
            )

        definition = concept.properties.get("definition", "")
        if not definition:
            return FactResult(
                concept=fact.concept, passed=False,
                detail=f"concept '{fact.concept}' has no definition",
            )

        def_lower = definition.lower()
        for sub in fact.expected_substrings:
            if sub.lower() not in def_lower:
                return FactResult(
                    concept=fact.concept, passed=False,
                    detail=f"definition '{definition}' missing '{sub}'",
                )

        if fact.relation_type is not None and fact.relation_target is not None:
            found = False
            for edge in self.network.get_edges(fact.concept, direction="out"):
                if edge.relation == fact.relation_type and edge.target == fact.relation_target:
                    found = True
                    break
            if not found:
                return FactResult(
                    concept=fact.concept, passed=False,
                    detail=f"no {fact.relation_type.value} edge to '{fact.relation_target}'",
                )

        return FactResult(
            concept=fact.concept, passed=True,
            detail=f"definition: {definition}",
        )

    def check_all(self, facts: list[Fact]) -> list[FactResult]:
        """Check all evaluation questions."""
        return [self.check_fact(f) for f in facts]

    # ── Reasoning ──

    def answer(self, fact: Fact) -> FactResult:
        """Ask the reasoning engine about a fact and check the answer."""
        if not fact.question_type:
            return self.check_fact(fact)

        result = self.reasoning.answer_question(fact.concept, fact.question_type)
        if result is None:
            return FactResult(
                concept=fact.concept, passed=False,
                detail="reasoning engine returned no answer",
            )

        conclusion_lower = result.conclusion.lower()
        for sub in fact.expected_answer_substrings:
            if sub.lower() not in conclusion_lower:
                return FactResult(
                    concept=fact.concept, passed=False,
                    detail=f"answer '{result.conclusion}' missing '{sub}'",
                )

        return FactResult(
            concept=fact.concept, passed=True,
            detail=f"answer: {result.conclusion}",
        )

    def answer_all(self, facts: list[Fact]) -> list[FactResult]:
        """Answer all evaluation questions, preserving input order."""
        remaining = [f for f in facts if not f.question_type]
        structural_by_concept = {r.concept: r for r in self.check_all(remaining)}
        results = []
        for f in facts:
            if f.question_type:
                results.append(self.answer(f))
            else:
                results.append(
                    structural_by_concept.get(
                        f.concept,
                        FactResult(
                            concept=f.concept,
                            passed=False,
                            detail="missing structural check",
                        ),
                    )
                )
        assert [r.concept for r in results] == [f.concept for f in facts]
        return results

    # ── Persistence ──

    def save(self) -> None:
        """Save the evaluation results."""
        self.mind._save_state()

    def load(self) -> None:
        """Load the evaluation results."""
        self.mind._load_saved_state()
        # Re-bind the network/reasoning in case the mind reloaded them
        self.network = self.mind.cognition.network
        self.reasoning = ReasoningEngine(self.network)


# ─── Result helpers ──────────────────────────────────────────────


def compute_accuracy(results: list[FactResult]) -> tuple[int, int, float]:
    """Compute accuracy metrics for the evaluation."""
    correct = sum(1 for r in results if r.passed)
    total = len(results)
    acc = correct / total if total > 0 else 0.0
    return correct, total, acc


def score_arc_solution(sol: Any, expected: list[Any]) -> tuple[bool, str]:
    """Score one ARC solution against expected test outputs (ARC convention).

    A task passes iff the solver reports solved with a hypothesis AND
    at least one guess set matches every expected grid exactly (equal
    length, element-wise equal). Length is checked explicitly —
    ``zip`` without ``strict`` would silently ignore trailing expected
    grids and false-pass short predictions.
    """
    hypothesis = getattr(sol, "hypothesis", None)
    if not getattr(sol, "solved", False) or hypothesis is None:
        if hypothesis is not None:
            describe = getattr(hypothesis, "describe", lambda: "?")()
            score = getattr(hypothesis, "score", None)
            detail = f"unsolved — best: {describe}"
            if score is not None:
                detail += f" (score {score:.0%})"
            return False, detail
        return False, "unsolved — no hypotheses"
    guesses = getattr(sol, "guesses", None) or []
    predictions = getattr(sol, "predictions", None) or []
    guess_sets = list(guesses) or ([predictions] if predictions else [])
    for guesses_one in guess_sets:
        if len(guesses_one) != len(expected) or not expected:
            continue
        if all(p == e for p, e in zip(guesses_one, expected, strict=True)):
            describe = getattr(hypothesis, "describe", lambda: "?")()
            return True, f"rule: {describe}"
    describe = getattr(hypothesis, "describe", lambda: "?")()
    return False, f"rule '{describe}' fit training but test prediction was wrong"


def run_condition(
    name: str,
    description: str,
    fn: Callable[[], list[FactResult]],
) -> ConditionResult:
    """Run a single condition and wrap its results."""
    start = time.time()
    fact_results = fn()
    elapsed = time.time() - start
    correct, total, acc = compute_accuracy(fact_results)
    return ConditionResult(
        name=name,
        description=description,
        accuracy=acc,
        correct=correct,
        total=total,
        facts=fact_results,
        duration_seconds=round(elapsed, 3),
    )


# ─── Output ───────────────────────────────────────────────────────


def write_json(result: EvalResult, path: str | Path) -> None:
    """Write an eval result to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def append_csv(result: EvalResult, path: str | Path) -> None:
    """Append a summary row to a CSV file for longitudinal tracking.

    One row per condition per eval run. Columns:
    timestamp, eval, condition, accuracy, correct, total, duration_seconds
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-existing but empty file still needs a header.
    write_header = (not path.exists()) or (path.stat().st_size == 0)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "eval", "condition",
                "accuracy", "correct", "total", "duration_seconds",
            ])
        for cond in result.conditions:
            writer.writerow([
                result.timestamp, result.name, cond.name,
                f"{cond.accuracy:.4f}", cond.correct, cond.total,
                f"{cond.duration_seconds:.3f}",
            ])


def print_result(result: EvalResult) -> None:
    """Print a human-readable summary of an eval result."""
    print()
    print(f"{'═' * 60}")
    print(f"  {result.name}")
    print(f"  {result.description}")
    print(f"{'═' * 60}")
    print(f"  Overall accuracy: {result.overall_accuracy:.1%}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print()
    for cond in result.conditions:
        status = "✓" if cond.accuracy >= 0.6 else "✗"
        print(f"  {status} {cond.name}: {cond.correct}/{cond.total} "
              f"({cond.accuracy:.1%}) — {cond.duration_seconds:.1f}s")
        for fr in cond.facts:
            mark = "✓" if fr.passed else "✗"
            print(f"      {mark} {fr.concept}: {fr.detail}")
    print()

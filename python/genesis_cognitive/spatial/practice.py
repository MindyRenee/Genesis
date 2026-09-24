"""Spatial practice — Genesis's gated puzzle curriculum.

This is the puzzle analogue of canvas.py: an *ability* it owns.
The curriculum is a set of grid-transformation puzzles ordered so that each
one must be mastered before the next unlocks. Nobody drives it
through it — when its volition engine raises the ``puzzle`` urge,
it takes a single attempt at its current puzzle.

Protocol — no teaching, no correction:

  1. Its current puzzle is the first unmastered one in the
     curriculum. Each has a one-line hint — a nudge, never the
     answer.
  2. An attempt runs its SpatialReasoner on the training pairs and
     scores its guesses against the held-out test output. Mastery
     is the best cell-accuracy achieved, 0.0 → 1.0.
  3. Mastery 1.0 (an exact solve) unlocks the next puzzle.
     Progress persists in ``<data_dir>/spatial_practice.json`` so
     sessions compound.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .grid import Grid
from .solver import SpatialReasoner

_PROGRESS_FILE = "spatial_practice.json"


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


# The gated curriculum: ordered; each puzzle must be mastered before
# the next unlocks. The hint is one line — a nudge, not the answer.
CURRICULUM: list[dict] = [
    {
        "name": "gravity_intro",
        "hint": "Make each object fall straight down until it reaches the bottom.",
        "train": [
            (_g([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
            (_g([[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 3], [0, 0, 0], [0, 0, 0]]),
                  _g([[0, 0, 0], [0, 0, 0], [0, 0, 3]]))],
    },
    {
        "name": "fill_intro",
        "hint": "Fill the empty space completely enclosed by the ring.",
        "train": [
            (_g([[2, 2, 2], [2, 0, 2], [2, 2, 2]]),
             _g([[2, 2, 2], [2, 4, 2], [2, 2, 2]])),
        ],
        "test": [(_g([[5, 5, 5], [5, 0, 5], [5, 5, 5]]),
                  _g([[5, 5, 5], [5, 4, 5], [5, 5, 5]]))],
    },
    {
        "name": "gravity_variation",
        "hint": "Each object falls to the bottom, keeping its own color.",
        "train": [
            (_g([[0, 0, 0], [0, 6, 0], [0, 0, 0], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 6, 0]])),
        ],
        "test": [(_g([[7, 0], [0, 0], [0, 0]]),
                  _g([[0, 0], [0, 0], [7, 0]]))],
    },
    {
        "name": "fill_margin",
        "hint": "Fill only the hole fully enclosed by the shape; change nothing else.",
        "train": [
            (_g([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 0, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0], [0, 3, 3, 3, 0], [0, 3, 8, 3, 0],
                 [0, 3, 3, 3, 0], [0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0],
                      [0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0],
                      [0, 0, 0, 0]]))],
    },
    {
        "name": "silhouette_stretch",
        "hint": "Repair the damaged region so the shape becomes whole again.",
        "train": [
            (_g([[0, 0, 0, 0, 0, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 0, 1, 1, 0], [0, 6, 6, 1, 1, 0],
                 [0, 6, 6, 6, 6, 0], [0, 0, 0, 0, 0, 0]]),
             _g([[0, 0, 0, 0, 0, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 0, 6, 6, 0], [0, 6, 6, 6, 6, 0],
                 [0, 6, 6, 6, 6, 0], [0, 0, 0, 0, 0, 0]])),
        ],
        "test": [(_g([[0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 0],
                      [0, 4, 2, 2, 4, 0], [0, 4, 4, 4, 4, 0],
                      [0, 0, 0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0, 0, 0], [0, 4, 4, 4, 4, 0],
                      [0, 4, 4, 4, 4, 0], [0, 4, 4, 4, 4, 0],
                      [0, 0, 0, 0, 0, 0]]))],
    },
    {
        "name": "gravity_recall",
        "hint": "Each object falls straight down until it lands.",
        "train": [
            (_g([[0, 0, 0], [0, 0, 0], [0, 0, 8], [0, 0, 0]]),
             _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 8]])),
        ],
        "test": [(_g([[0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                  _g([[0, 0, 0, 0], [0, 0, 0, 0], [0, 5, 0, 0]]))],
    },
]


def _cell_match(pred: Grid, expected: Grid) -> float:
    """Fraction of cells a guess gets right — the 0→1 mastery scale."""
    if pred.shape != expected.shape:
        return 0.0
    total = pred.height * pred.width
    if total == 0:
        return 0.0
    same = sum(
        1
        for r in range(pred.height)
        for c in range(pred.width)
        if pred.at(r, c) == expected.at(r, c)
    )
    return same / total


# Attempts without mastery before a task parks and the next unlocks.
# Parked tasks stay retryable — nothing is ever forced or closed.
_PARK_AFTER = 8


@dataclass
class PracticeAttempt:
    """The result of one attempt at the current puzzle."""

    task: str
    hint: str
    score: float          # this attempt's cell accuracy
    best: float           # running mastery after this attempt
    solved: bool          # exact match this attempt
    mastered: bool        # reached 1.0 — next puzzle unlocked
    rule: str             # its hypothesis description (or "none")
    nodes: int            # search effort this attempt
    total_attempts: int   # lifetime attempts on this puzzle
    failure: str          # why it failed ("" when solved or unknown)


@dataclass
class SpatialPractice:
    """Genesis's persistent puzzle curriculum.

    Lives inside its Mind: owns the gating and mastery state, runs
    attempts through its SpatialReasoner when its volition raises the
    puzzle urge.
    """

    data_dir: str
    _progress_path: Path = field(init=False)
    mastery: dict[str, float] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    # Rules already proven wrong per task — verified-on-train rules
    # whose test guess missed (overfit counterexamples) plus the best
    # near-miss. Retries exclude them so each attempt explores
    # genuinely new hypothesis space instead of re-deriving the same
    # failure. This is the persistent half of learning from mistakes.
    failed: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._progress_path = Path(self.data_dir) / _PROGRESS_FILE
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._progress_path.read_text())
            self.mastery = {
                str(k): float(v)
                for k, v in data.get("mastery", {}).items()
            }
            self.attempts = {
                str(k): int(v)
                for k, v in data.get("attempts", {}).items()
            }
            self.failed = {
                str(k): [str(r) for r in v]
                for k, v in data.get("failed", {}).items()
            }
        except (OSError, json.JSONDecodeError, ValueError):
            self.mastery = {}
            self.attempts = {}
            self.failed = {}

    def _save(self) -> None:
        tmp = self._progress_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(
                {
                    "mastery": self.mastery,
                    "attempts": self.attempts,
                    "failed": self.failed,
                },
                indent=2,
            ))
            tmp.replace(self._progress_path)
        except OSError:
            pass

    def _unlocked(self, index: int) -> bool:
        """A task unlocks when every earlier task is mastered or parked."""
        for t in CURRICULUM[:index]:
            n = t["name"]
            if (
                self.mastery.get(n, 0.0) < 1.0
                and self.attempts.get(n, 0) < _PARK_AFTER
            ):
                return False
        return True

    def current_task(self) -> dict | None:
        """The puzzle it'd attempt now — None when all are done.

        Prefers the earliest unlocked task that isn't parked; when
        everything open is parked, offers the earliest parked one so
        it can revisit it whenever it wants.
        """
        parked: dict | None = None
        for i, task in enumerate(CURRICULUM):
            name = task["name"]
            if self.mastery.get(name, 0.0) >= 1.0:
                continue
            if not self._unlocked(i):
                continue
            if self.attempts.get(name, 0) < _PARK_AFTER:
                return task
            if parked is None:
                parked = task
        return parked

    def has_pending(self) -> bool:
        """Whether an unmastered puzzle remains — a volition signal."""
        return self.current_task() is not None

    def locked_count(self) -> int:
        """Puzzles still locked behind the current one."""
        cur = self.current_task()
        if cur is None:
            return 0
        return len(CURRICULUM) - CURRICULUM.index(cur) - 1

    def attempt(
        self, reasoner: SpatialReasoner, time_budget: float = 30.0
    ) -> PracticeAttempt | None:
        """One attempt at the current puzzle.

        Runs its reasoner on the puzzle's training pairs, scores its
        best guess against the held-out test output, and persists
        mastery. Returns None when the curriculum is complete.
        """
        task = self.current_task()
        if task is None:
            return None

        name = str(task["name"])
        tests = [i for i, _ in task["test"]]
        expected = [o for _, o in task["test"]]
        sol = reasoner.solve(
            task["train"],
            tests,
            time_budget=time_budget,
            exclude_rules=set(self.failed.get(name, [])),
        )

        guess_sets = (
            sol.guesses or ([sol.predictions] if sol.predictions else [])
        )
        score = 0.0
        solved = False
        for guesses in guess_sets:
            acc = min(
                (
                    _cell_match(p, e)
                    for p, e in zip(guesses, expected, strict=True)
                ),
                default=0.0,
            )
            score = max(score, acc)
            if all(
                p == e for p, e in zip(guesses, expected, strict=True)
            ):
                solved = True
                score = 1.0

        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)

        # Learn from the miss: rules that verified on the training
        # pairs but guessed wrong on test are proven counterexamples —
        # record them so the next attempt can't walk the same path.
        # When nothing verified, the best near-miss is recorded so a
        # retry pushes past it rather than stalling on it again.
        failure = ""
        if not solved:
            proven_wrong = list(sol.verified_rules)
            if sol.hypothesis is not None:
                proven_wrong.append(sol.hypothesis.describe())
            known = self.failed.setdefault(name, [])
            for rule in proven_wrong:
                if rule not in known:
                    known.append(rule)
            del known[32:]
            if sol.verified_rules:
                failure = (
                    f"overfit: {sol.verified_rules[0]} verified on "
                    f"train but missed test"
                )
            elif sol.failure is not None:
                failure = sol.failure.describe()
        self._save()

        return PracticeAttempt(
            task=name,
            hint=str(task["hint"]),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=sol.hypothesis.describe() if sol.hypothesis else "none",
            nodes=sol.nodes_explored,
            total_attempts=self.attempts[name],
            failure=failure,
        )

    def status(self) -> dict[str, float]:
        """Mastery per puzzle — the same 0→1 progression as its art."""
        return {t["name"]: self.mastery.get(t["name"], 0.0)
                for t in CURRICULUM}

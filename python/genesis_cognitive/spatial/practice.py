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
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cognitive_journal import record_error
from .grid import Grid
from .solver import SpatialReasoner

_PROGRESS_FILE = "spatial_practice.json"
logger = logging.getLogger(__name__)


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
    family: str = "grid"  # which task family the puzzle belongs to
    # Family-specific result payload — what the attempt actually
    # produced (a classification's labels, a quantities basket...).
    # Telemetry, not presentation: callers render it themselves.
    details: dict = field(default_factory=dict)


# The drop-box: puzzles placed into its world by someone else.
# ``<data_dir>/offered_puzzles/*.json`` files are task specs — data,
# not code — so anything that can write a description can hand it a
# puzzle: a person, a file sync, eventually its own comprehension.
_OFFERED_DIR = "offered_puzzles"
_FAMILIES = (
    "grid",
    "sorter",
    "sequence",
    "assembly",
    "relations",
    "quantities",
    "classification",
)


def _rectangular(rows: Any) -> bool:
    return (
        isinstance(rows, list)
        and bool(rows)
        and all(
            isinstance(r, list)
            and bool(r)
            and all(isinstance(c, int) for c in r)
            for r in rows
        )
        and len({len(r) for r in rows}) == 1
    )


def _pairs(raw: Any) -> list[tuple[Grid, Grid]] | None:
    """Parse ``[[in_rows, out_rows], ...]`` into Grid pairs, or None."""
    if not isinstance(raw, list) or not raw:
        return None
    out: list[tuple[Grid, Grid]] = []
    for pair in raw:
        if not (isinstance(pair, list) and len(pair) == 2):
            return None
        if not (_rectangular(pair[0]) and _rectangular(pair[1])):
            return None
        out.append((_g(pair[0]), _g(pair[1])))
    return out


def _sorter_solvable(slots: list[dict], blocks: list[dict]) -> bool:
    """Can every slot get a distinct block satisfying its constraints?
    (Kuhn's bipartite matching — offered puzzles must be possible.)"""
    owner: dict[int, int] = {}  # slot index -> block index

    def fits(block: dict, accepts: dict) -> bool:
        return all(block.get(k) == v for k, v in accepts.items())

    def augment(bi: int, seen: set[int]) -> bool:
        for si, slot in enumerate(slots):
            if si in seen or not fits(blocks[bi], slot):
                continue
            seen.add(si)
            if si not in owner or augment(owner[si], seen):
                owner[si] = bi
                return True
        return False

    for bi in range(len(blocks)):
        augment(bi, set())
    return len(owner) == len(slots)


def _sorter_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize a sorter task spec, or None when malformed/impossible."""
    gen = raw.get("generate")
    if isinstance(gen, dict):
        try:
            n_slots = int(gen.get("slots", 4))
            difficulty = int(gen.get("difficulty", 1))
            decoys = int(gen.get("decoys", 0))
        except (TypeError, ValueError):
            return None
        if not (1 <= n_slots <= 12 and 1 <= difficulty <= 3):
            return None
        if not 0 <= decoys <= 24:
            return None
        return {
            "generate": {
                "slots": n_slots,
                "difficulty": difficulty,
                "decoys": decoys,
                "seed": gen.get("seed"),
                "lid": bool(gen.get("lid", True)),
            }
        }
    slots_raw, blocks_raw = raw.get("slots"), raw.get("blocks")
    if (
        not isinstance(slots_raw, list)
        or not isinstance(blocks_raw, list)
        or not slots_raw
        or not blocks_raw
        or len(slots_raw) > 32
        or len(blocks_raw) > 64
    ):
        return None
    slots: list[dict[str, str]] = []
    for s in slots_raw:
        accepts = s.get("accepts") if isinstance(s, dict) else None
        if not isinstance(accepts, dict) or not accepts:
            return None
        slots.append({str(k): str(v) for k, v in accepts.items()})
    blocks: list[dict[str, str]] = []
    for b in blocks_raw:
        attrs = b.get("attrs", b) if isinstance(b, dict) else None
        if not isinstance(attrs, dict) or not attrs:
            return None
        blocks.append(
            {str(k): str(v) for k, v in attrs.items() if k != "attrs"}
        )
    if not _sorter_solvable(slots, blocks):
        return None
    return {
        "slots": slots,
        "blocks": blocks,
        "lid": bool(raw.get("lid", True)),
        "perceptual": bool(raw.get("perceptual")),
    }


def _sequence_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize a pattern-line spec, or None when malformed."""
    gen = raw.get("generate")
    if isinstance(gen, dict):
        try:
            period = int(gen.get("period", 2))
            length = int(gen.get("length", 6))
            decoys = int(gen.get("decoys", 0))
            scaffold = (
                int(gen["scaffold"])
                if gen.get("scaffold") is not None
                else None
            )
        except (TypeError, ValueError):
            return None
        if not (1 <= period <= 6 and period < length <= 24):
            return None
        if not 0 <= decoys <= 24:
            return None
        return {
            "generate": {
                "period": period,
                "length": length,
                "scaffold": scaffold,
                "decoys": decoys,
                "seed": gen.get("seed"),
            }
        }
    pattern = raw.get("pattern")
    if (
        not isinstance(pattern, list)
        or not pattern
        or len(pattern) > 6
        or not all(isinstance(m, str) and m for m in pattern)
    ):
        return None
    try:
        length = int(raw.get("length", len(pattern) * 2))
        decoys = int(raw.get("decoys", 0))
        scaffold = (
            int(raw["scaffold"]) if raw.get("scaffold") is not None else None
        )
    except (TypeError, ValueError):
        return None
    if not len(pattern) < length <= 24 or not 0 <= decoys <= 24:
        return None
    return {
        "pattern": [str(m) for m in pattern],
        "length": length,
        "scaffold": scaffold,
        "decoys": decoys,
        "seed": raw.get("seed"),
    }


_RELATION_RELS = ("left_of", "right_of", "next_to", "apart", "ordered")


def _relations_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize an arrangement spec, or None when malformed/impossible."""
    from ..relations import Arrangement, Goal, Thing

    gen = raw.get("generate")
    if isinstance(gen, dict):
        try:
            n_objects = int(gen.get("objects", 4))
            n_goals = int(gen.get("goals", 3))
        except (TypeError, ValueError):
            return None
        if not (2 <= n_objects <= 7 and 1 <= n_goals <= 21):
            return None
        attr = gen.get("attr")
        return {
            "generate": {
                "objects": n_objects,
                "goals": n_goals,
                "attr": str(attr) if attr is not None else None,
                "seed": gen.get("seed"),
            }
        }
    objects_raw = raw.get("objects")
    if (
        not isinstance(objects_raw, list)
        or not 2 <= len(objects_raw) <= 7
    ):
        return None
    things: list[Thing] = []
    names: set[str] = set()
    for i, o in enumerate(objects_raw):
        if isinstance(o, str):
            o = {"name": o}
        if not isinstance(o, dict) or not str(o.get("name", "")).strip():
            return None
        name = str(o["name"]).strip()
        if name in names:
            return None
        names.add(name)
        attrs_raw = o.get("attrs", {})
        if not isinstance(attrs_raw, dict):
            return None
        try:
            attrs = tuple(
                (str(k), int(v)) for k, v in attrs_raw.items()
            )
        except (TypeError, ValueError):
            return None
        things.append(Thing(i, name, attrs))
    try:
        positions = int(raw.get("positions", len(things)))
    except (TypeError, ValueError):
        return None
    if not len(things) <= positions <= 12:
        return None
    goals_raw = raw.get("goals")
    if not isinstance(goals_raw, list) or not goals_raw:
        return None
    goals: list[Goal] = []
    for g in goals_raw:
        if not isinstance(g, dict):
            return None
        rel = str(g.get("rel", ""))
        if rel == "ordered_all":
            # Shorthand: pairwise order-agreement over every pair —
            # "line them up smallest to largest."
            attr = g.get("attr")
            if not isinstance(attr, str) or not attr:
                return None
            ordered_names = sorted(names)
            for i, a in enumerate(ordered_names):
                for b in ordered_names[i + 1 :]:
                    goals.append(Goal("ordered", a, b, attr))
            continue
        a, b = str(g.get("a", "")), str(g.get("b", ""))
        if (
            rel not in _RELATION_RELS
            or a not in names
            or b not in names
            or a == b
        ):
            return None
        attr = g.get("attr")
        if rel == "ordered" and not isinstance(attr, str):
            return None
        goals.append(
            Goal(rel, a, b, str(attr) if attr is not None else None)
        )
    if not goals:
        return None
    if not Arrangement(positions, things, goals).solvable():
        return None
    return {
        "positions": positions,
        "objects": objects_raw,
        "goals": [
            {"rel": g.rel, "a": g.a, "b": g.b, "attr": g.attr}
            for g in goals
        ],
    }


def _quantities_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize a make-N spec, or None when malformed/impossible."""
    from ..quantities import Basket, Group

    gen = raw.get("generate")
    if isinstance(gen, dict):
        try:
            n_groups = int(gen.get("groups", 5))
            target = int(gen["target"])
        except (TypeError, ValueError, KeyError):
            return None
        if not (1 <= target <= 60 and 2 <= n_groups <= 12):
            return None
        return {
            "generate": {
                "groups": n_groups,
                "target": target,
                "seed": gen.get("seed"),
            }
        }
    groups_raw = raw.get("groups")
    if isinstance(groups_raw, dict):
        groups_raw = [
            {"name": k, "count": v} for k, v in groups_raw.items()
        ]
    if (
        not isinstance(groups_raw, list)
        or not 2 <= len(groups_raw) <= 12
    ):
        return None
    groups: list[Group] = []
    names: set[str] = set()
    try:
        target = int(raw["target"])
        for i, g in enumerate(groups_raw):
            name = str(g.get("name", f"group_{i}")) if isinstance(
                g, dict
            ) else f"group_{i}"
            count = int(g["count"]) if isinstance(g, dict) else int(g)
            if name in names or count < 1:
                return None
            names.add(name)
            groups.append(Group(i, name, count))
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if not (1 <= target <= 60):
        return None
    if not Basket(target, groups).solvable():
        return None
    return {
        "target": target,
        "groups": [{"name": g.name, "count": g.count} for g in groups],
    }


def _classification_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize a rule-membership spec, or None when malformed.

    Two honest modes:

    - ``apply`` — a ``predicate`` is stated; the oracle scores by it.
      Items may also carry ``label`` and ``examples`` must — those
      assertions are checked *against* the predicate, so a spec whose
      stated rule contradicts its own examples is rejected rather
      than self-verifying.
    - ``induce`` — no predicate; every item carries ``label`` (the
      hidden answer key) and labeled ``examples`` provide visible
      supervision. The agent learns the rule from feedback alone.
    """
    from ..classification import eval_predicate, valid_predicate

    def _items(node: Any, require_label: bool) -> list[dict] | None:
        if not isinstance(node, list) or not 1 <= len(node) <= 24:
            return None
        out: list[dict] = []
        names: set[str] = set()
        try:
            for i, it in enumerate(node):
                if not isinstance(it, dict):
                    return None
                name = str(it.get("name", f"item_{i}")).strip().lower()
                attrs = it.get("has")
                if not name or name in names:
                    return None
                if (
                    not isinstance(attrs, list)
                    or len(attrs) > 12
                    or not all(
                        isinstance(a, str) and a.strip() for a in attrs
                    )
                ):
                    return None
                label = it.get("label")
                if label is not None and not isinstance(label, bool):
                    return None
                if require_label and label is None:
                    return None
                names.add(name)
                entry: dict[str, Any] = {
                    "name": name,
                    "has": sorted({str(a).strip().lower() for a in attrs}),
                }
                if label is not None:
                    entry["label"] = label
                out.append(entry)
        except (TypeError, ValueError, AttributeError):
            return None
        return out

    pred = raw.get("predicate")
    if pred is not None and not valid_predicate(pred):
        return None
    if isinstance(pred, dict) and isinstance(pred.get("of"), list):
        pred = {**pred, "of": [str(a).strip().lower() for a in pred["of"]]}
    items = _items(raw.get("items"), require_label=pred is None)
    if items is None:
        return None
    examples_raw = raw.get("examples")
    examples = (
        []
        if examples_raw is None
        else _items(examples_raw, require_label=True)
    )
    if examples is None:
        return None
    # Consistency: asserted labels (examples and any labeled items)
    # must agree with the stated predicate. A spec that fails this
    # describes a rule its own author contradicts — refuse it instead
    # of letting the same predicate verify itself.
    if pred is not None:
        for entry in [*examples, *items]:
            if "label" not in entry:
                continue
            if eval_predicate(pred, frozenset(entry["has"])) != bool(
                entry["label"]
            ):
                return None
    # Omit empty/None fields so the normalized dict re-validates
    # identically — ``offer()`` writes this back into the drop-box
    # and ``offered_tasks()`` normalizes it again on read.
    out: dict[str, Any] = {
        "items": items,
        "kind": str(raw.get("kind", "")).strip(),
    }
    if pred is not None:
        out["predicate"] = pred
    if examples:
        out["examples"] = examples
    return out


def normalize_offered(raw: Any, fallback_name: str) -> dict | None:
    """Validate an offered-puzzle file into a canonical task dict.

    Returns None for anything malformed — a bad offer is skipped, not
    fatal. Accepted families:

    - ``grid``       — ``{"train": [[in,out],...], "test": [...]}``
    - ``sorter``     — ``{"generate": {...}}`` or ``{"slots": [{accepts}],
                       "blocks": [{attrs...}]}``
    - ``sequence``   — ``{"generate": {...}}`` or ``{"pattern": [marks],
                       "length": n, "scaffold": n?}``
    - ``assembly``   — ``{"generate": {"rows","cols","seed"?}}``
    - ``relations``  — ``{"generate": {...}}`` or ``{"positions": n,
                       "objects": [...], "goals": [{"rel","a","b"}]}``
    - ``quantities`` — ``{"generate": {"groups","target"}}`` or
                       ``{"target": n, "groups": {"name": count}}``
    - ``classification`` — ``{"items": [{"name","has": [...]}],
                       "predicate"?: {...}, "examples"?: [{..."label"}],
                       "kind"?: str}``
    """
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or fallback_name).strip()
    if not name:
        return None
    family = str(raw.get("family", "grid")).strip().lower()
    if family not in _FAMILIES:
        return None
    hint = str(raw.get("hint", "")).strip()
    task: dict[str, Any] = {"name": name, "family": family, "hint": hint}
    if family == "grid":
        train = _pairs(raw.get("train"))
        test = _pairs(raw.get("test"))
        if not train or not test:
            return None
        task["train"], task["test"] = train, test
        return task
    if family == "sorter":
        spec = _sorter_spec(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    if family == "sequence":
        spec = _sequence_spec(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    if family == "assembly":
        gen = raw.get("generate")
        if not isinstance(gen, dict):
            return None
        try:
            rows, cols = int(gen["rows"]), int(gen["cols"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (1 <= rows <= 8 and 1 <= cols <= 8):
            return None
        task["generate"] = {
            "rows": rows,
            "cols": cols,
            "seed": gen.get("seed"),
        }
        return task
    if family == "relations":
        spec = _relations_spec(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    if family == "quantities":
        spec = _quantities_spec(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    if family == "classification":
        spec = _classification_spec(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    return None


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
    # The visual cortex, when the Mind wires one in — perceptual
    # tasks (a sorter you must look at) need it; symbolic ones don't.
    cortex: Any = None
    # Offered-puzzle file cache: filename -> (mtime, task|None).
    _offered_cache: dict[str, tuple[float, dict | None]] = field(
        default_factory=dict
    )

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
        except OSError as e:
            # Progress silently vanishing is the worst outcome — record
            # it so a lost mastery file is at least visible in the
            # cognitive journal.
            record_error("spatial_practice.save", e)

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

    def offered_dir(self) -> Path:
        """The drop-box: ``<data_dir>/offered_puzzles/``."""
        return Path(self.data_dir) / _OFFERED_DIR

    def offer(self, task: dict) -> Path | None:
        """Write a normalized task spec into the drop-box.

        The same channel the CLI's ``/puzzle offer`` uses — anything
        that produces a valid spec (including a problem heard in
        conversation) lands in the inner world as a file. Returns the
        written path, or None when the task can't be persisted. The
        offer cache entry is invalidated so the new file is seen on
        the next ``offered_tasks()`` call.
        """
        name = str(task.get("name", "")).strip()
        if not name or task.get("family") not in _FAMILIES:
            return None
        safe = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in name
        )
        directory = self.offered_dir()
        try:
            text = json.dumps(task, indent=2) + "\n"
        except (TypeError, ValueError):
            # Grid-family normalized tasks hold Grid objects — not
            # serializable; offers of those stay file-driven.
            return None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{safe}.json"
            path.write_text(text)
        except OSError:
            return None
        self._offered_cache.pop(path.name, None)
        return path

    def offered_tasks(self) -> list[dict]:
        """Puzzles placed in its world by someone else.

        Every ``*.json`` file in the drop-box is a task spec parsed by
        ``normalize_offered``; malformed files are skipped (never
        fatal), files are cached by mtime, and names colliding with
        the curriculum or an earlier offer are ignored.
        """
        out: list[dict] = []
        seen: set[str] = {str(t["name"]) for t in CURRICULUM}
        try:
            files = sorted(self.offered_dir().glob("*.json"))
        except OSError:
            files = []
        live: set[str] = set()
        for f in files:
            live.add(f.name)
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            cached = self._offered_cache.get(f.name)
            if cached is None or cached[0] != mtime:
                try:
                    raw: Any = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    raw = None
                cached = (mtime, normalize_offered(raw, f.stem))
                self._offered_cache[f.name] = cached
            task = cached[1]
            if task is None or task["name"] in seen:
                continue
            seen.add(task["name"])
            out.append(task)
        for stale in set(self._offered_cache) - live:
            del self._offered_cache[stale]
        return out

    def current_task(self) -> dict | None:
        """The puzzle it'd attempt now — None when all are done.

        Offered puzzles come first — something that appeared in its
        world is more salient than the standing curriculum — then the
        earliest unlocked curriculum task that isn't parked; when
        everything open is parked, offers the earliest parked one so
        it can revisit it whenever it wants.
        """
        parked: dict | None = None
        for task in self.offered_tasks():
            name = task["name"]
            if self.mastery.get(name, 0.0) >= 1.0:
                continue
            if self.attempts.get(name, 0) < _PARK_AFTER:
                return task
            if parked is None:
                parked = task
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
        """Curriculum puzzles still locked behind the current one."""
        cur = self.current_task()
        if cur is None:
            return 0
        names = [t["name"] for t in CURRICULUM]
        if cur["name"] not in names:
            # An offered task is current — every unmastered
            # curriculum puzzle still waits behind it.
            return sum(
                1
                for t in CURRICULUM
                if self.mastery.get(t["name"], 0.0) < 1.0
            )
        return len(CURRICULUM) - names.index(cur["name"]) - 1

    def attempt(
        self,
        reasoner: SpatialReasoner,
        time_budget: float = 30.0,
        task: dict | None = None,
    ) -> PracticeAttempt | None:
        """One attempt at a puzzle.

        ``task`` defaults to ``current_task()`` — the volition path.
        A caller holding a specific normalized spec (a task just
        compiled from conversation, say) passes it directly so the
        *that* task runs now rather than whichever pending offer is
        oldest; bookkeeping (attempts, mastery, persistence) is
        identical either way. Returns None when nothing is pending.
        """
        task = task or self.current_task()
        if task is None:
            return None

        family = str(task.get("family", "grid"))
        if family == "sorter":
            return self._attempt_sorter(task, reasoner)
        if family == "sequence":
            return self._attempt_sequence(task, reasoner)
        if family == "assembly":
            return self._attempt_assembly(task, reasoner)
        if family == "relations":
            return self._attempt_relations(task, reasoner)
        if family == "quantities":
            return self._attempt_quantities(task, reasoner)
        if family == "classification":
            return self._attempt_classification(task, reasoner)

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
        try:
            # The held-out test is the external verifier: only its
            # outcome decides whether a train-verified rule becomes a
            # reusable skill.
            reasoner.record_task_outcome(
                sol, success=solved, score=score,
                examples=list(task["train"]),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("task competence update failed: %s", e)

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

    def _attempt_sorter(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One sorter episode through the competence substrate."""
        from ..sorter import (
            Block,
            PerceptualSorter,
            PerceptualSorterAgent,
            ShapeSorter,
            Slot,
            SorterAgent,
        )

        name = str(task["name"])
        gen = task.get("generate")
        # A "perceptual" sorter is seen, not read: the world renders
        # its pieces and apertures, the oracle stops leaking `matched`
        # on candidates, and the agent learns seen-similarity→cost
        # instead of stated-attrs→cost. Falls back to symbolic when
        # no cortex is wired in.
        perceptual = (
            bool(task.get("perceptual")) and self.cortex is not None
        )
        if gen:
            seed = gen.get("seed")
            base = ShapeSorter.generate(
                int(gen["slots"]),
                random.Random(
                    seed if seed is not None else random.randrange(2**30)
                ),
                difficulty=int(gen["difficulty"]),
                decoys=int(gen["decoys"]),
                has_lid=bool(gen.get("lid", True)),
            )
            sorter: ShapeSorter = base
            if perceptual:
                sorter = PerceptualSorter(
                    base.slots,
                    [*base.pool.values(), *base.in_top],
                    has_lid=base.has_lid,
                    cortex=self.cortex,
                    seed=hash(name) & 0x7FFFFFFF,
                )
        else:
            sorter = ShapeSorter(
                [
                    Slot(i, tuple(sorted(s.items())))
                    for i, s in enumerate(task["slots"])
                ],
                [
                    Block(i, tuple(sorted(b.items())))
                    for i, b in enumerate(task["blocks"])
                ],
                has_lid=bool(task.get("lid", True)),
            )
            if perceptual:
                sorter = PerceptualSorter(
                    sorter.slots,
                    [*sorter.pool.values(), *sorter.in_top],
                    has_lid=sorter.has_lid,
                    cortex=self.cortex,
                    seed=hash(name) & 0x7FFFFFFF,
                )
        # A fresh agent each attempt — cross-attempt learning lives
        # in the shared task competence, so skills earned on earlier
        # sorters (and structurally similar tasks) apply as priors.
        agent: SorterAgent = (
            PerceptualSorterAgent(
                task_competence=reasoner.task_competence,
                seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
            )
            if perceptual
            else SorterAgent(
                task_competence=reasoner.task_competence,
                seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
            )
        )
        res = agent.solve(sorter, max_steps=int(task.get("max_steps", 400)))
        if perceptual:
            # The world names what it showed — the way a parent
            # points at a piece and says "that's a square."
            assert isinstance(sorter, PerceptualSorter)
            sorter.teach()
        solved = res.solved
        score = 1.0 if solved else res.placed / max(1, len(sorter.slots))
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        # The answer payload: which block landed in which slot —
        # structured so the utterance can report the actual
        # placements, not just a score.
        placements = [
            {
                "slot": dict(sorter.slots[si].accepts),
                "block": dict(block.attrs),
            }
            for si, block in sorted(sorter.board.items())
        ]
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=agent.describe_policy() or "none",
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{res.placed}/{len(sorter.slots)} slots filled"
                )
            ),
            family="sorter",
            details={"placements": placements} if placements else {},
        )

    def _attempt_sequence(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One pattern-line episode through the competence substrate."""
        from ..sequence import PatternAgent, PatternLine

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(
            seed if seed is not None else random.randrange(2**30)
        )
        if gen:
            line = PatternLine.generate(
                int(gen["period"]),
                int(gen["length"]),
                rng,
                scaffold=gen.get("scaffold"),
                decoys=int(gen["decoys"]),
            )
        else:
            line = PatternLine.generate(
                len(task["pattern"]),
                int(task["length"]),
                rng,
                scaffold=task.get("scaffold"),
                decoys=int(task.get("decoys", 0)),
                pattern=list(task["pattern"]),
            )
        agent = PatternAgent(
            task_competence=reasoner.task_competence,
            seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
        )
        res = agent.solve(line, max_steps=int(task.get("max_steps", 400)))
        solved = res.solved
        score = 1.0 if solved else res.placed / max(1, len(line.cells))
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=agent.describe_policy() or "none",
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{res.placed}/{len(line.cells)} cells filled"
                )
            ),
            family="sequence",
            details={
                "cells": [c for c in line.cells if c is not None],
            },
        )

    def _attempt_assembly(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One assembly episode through the competence substrate."""
        from ..assembly import AssemblyAgent, PiecePuzzle

        name = str(task["name"])
        gen = task["generate"]
        seed = gen.get("seed")
        rng = random.Random(
            seed if seed is not None else random.randrange(2**30)
        )
        rows, cols = int(gen["rows"]), int(gen["cols"])
        puzzle = PiecePuzzle.generate(rows, cols, rng)
        agent = AssemblyAgent(
            task_competence=reasoner.task_competence,
            seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
        )
        res = agent.solve(puzzle, max_steps=int(task.get("max_steps", 500)))
        solved = res.solved
        score = 1.0 if solved else res.placements / max(1, rows * cols)
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(
                f"placed {res.placements}, lifted {res.removals} "
                f"in {res.steps} steps"
            ),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{res.placements}/{rows * cols} placed"
                )
            ),
            family="assembly",
        )

    def _attempt_relations(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One arrangement episode through the competence substrate."""
        from ..relations import Arrangement, Goal, RelationsAgent, Thing

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(
            seed if seed is not None else random.randrange(2**30)
        )
        if gen:
            attr = gen.get("attr")
            world = Arrangement.generate(
                int(gen["objects"]),
                int(gen["goals"]),
                rng,
                with_attr=str(attr) if attr is not None else None,
            )
        else:
            things = []
            for i, o in enumerate(task["objects"]):
                if isinstance(o, str):
                    o = {"name": o}
                attrs = tuple(
                    (str(k), int(v))
                    for k, v in (o.get("attrs") or {}).items()
                )
                things.append(Thing(i, str(o["name"]), attrs))
            goals = [
                Goal(
                    str(g["rel"]),
                    str(g["a"]),
                    str(g["b"]),
                    str(g["attr"]) if g.get("attr") is not None else None,
                )
                for g in task["goals"]
            ]
            world = Arrangement(int(task["positions"]), things, goals)
        agent = RelationsAgent(
            task_competence=reasoner.task_competence,
            seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
        )
        res = agent.solve(world, max_steps=int(task.get("max_steps", 500)))
        solved = res.solved
        n_goals = max(1, len(world.goals))
        score = 1.0 if solved else max(
            0.0, 1.0 - world.mismatches() / (n_goals + len(world.things))
        )
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(
                f"placed {res.placements}, swapped {res.swaps}, "
                f"lifted {res.removals} in {res.steps} steps"
            ),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{world.mismatches()} goals/things unresolved"
                )
            ),
            family="relations",
            details={
                "order": [
                    world.board[p].name
                    for p in sorted(world.board)
                ],
            },
        )

    def _attempt_quantities(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One make-N episode through the competence substrate."""
        from ..quantities import Basket, Group, QuantitiesAgent

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(
            seed if seed is not None else random.randrange(2**30)
        )
        if gen:
            world = Basket.generate(
                int(gen["groups"]), int(gen["target"]), rng
            )
        else:
            world = Basket(
                int(task["target"]),
                [
                    Group(i, str(g["name"]), int(g["count"]))
                    for i, g in enumerate(task["groups"])
                ],
            )
        agent = QuantitiesAgent(
            task_competence=reasoner.task_competence,
            seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
        )
        res = agent.solve(world, max_steps=int(task.get("max_steps", 300)))
        solved = res.solved
        score = (
            1.0
            if solved
            else max(
                0.0,
                1.0 - world.mismatches() / max(1, world.target),
            )
        )
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(
                f"added {res.adds}, lifted {res.lifts} "
                f"in {res.steps} steps"
            ),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{world.basket_sum()}/{world.target} in basket"
                )
            ),
            family="quantities",
            details={
                "counts": sorted(
                    g.count for g in world.basket.values()
                ),
                "target": world.target,
            },
        )

    def _attempt_classification(
        self, task: dict, reasoner: SpatialReasoner
    ) -> PracticeAttempt:
        """One rule-membership episode through the competence substrate."""
        from ..classification import Item, RuleAgent, RuleGame

        name = str(task["name"])
        items = [
            Item(
                i,
                str(it["name"]),
                tuple(str(a) for a in it["has"]),
                it.get("label"),
            )
            for i, it in enumerate(task["items"])
        ]
        examples = [
            Item(
                i,
                str(e["name"]),
                tuple(str(a) for a in e["has"]),
                bool(e["label"]),
            )
            for i, e in enumerate(task.get("examples") or [])
        ]
        world = RuleGame(task.get("predicate"), items, examples)
        agent = RuleAgent(
            task_competence=reasoner.task_competence,
            seed=hash((name, self.attempts.get(name, 0))) & 0x7FFFFFFF,
        )
        res = agent.solve(world, max_steps=int(task.get("max_steps", 300)))
        solved = res.solved
        score = (
            1.0
            if solved
            else len(world.labels()) / max(1, len(world.items))
        )
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return PracticeAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(
                f"{res.guesses} guesses, {res.wrong} wrong "
                f"in {res.steps} steps"
            ),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (
                    f"{res.state.lower()}: "
                    f"{world.mismatches()} items unlabeled"
                )
            ),
            family="classification",
            # The answer itself — which items belong — so a caller
            # that asked the question can report the result.
            details={
                "labels": world.labels(),
                "kind": str(task.get("kind", "")),
            },
        )

    def status(self) -> dict[str, float]:
        """Mastery per puzzle — the same 0→1 progression as its art."""
        out = {t["name"]: self.mastery.get(t["name"], 0.0)
               for t in CURRICULUM}
        for t in self.offered_tasks():
            out[t["name"]] = self.mastery.get(t["name"], 0.0)
        return out

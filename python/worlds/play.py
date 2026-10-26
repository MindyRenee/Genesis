"""Game play — Genesis's game collection and how well she plays it.

This is the play analogue of canvas.py: an *ability* she owns.
The curriculum is a set of games ordered so that each must be
mastered before the next unlocks, and the ``offered_games`` drop-box
takes games anyone hands her — a file, a described problem. Nobody
drives her through it — when her volition engine raises the ``play``
urge, she takes a single go at her current game. Play is the
activity, not a rehearsal for one: mastery is just how good she's
gotten, like a person getting better at solitaire.

Protocol — no teaching, no correction:

  1. Her current game is the first unmastered one in the curriculum.
     Each has a one-line hint — a nudge, never the answer.
  2. An attempt runs the family's agent on the game and scores
     against the world's own verdict — never a given answer.
  3. Mastery 1.0 (an exact solve) unlocks the next game.
     Progress persists in ``<data_dir>/gameplay.json`` so
     sessions compound.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontal_lobe.dlpfc import DorsolateralPFC
    from parietal_lobe.spatial.solver import SpatialReasoner
    from worlds.sorter import ShapeSorter

from parietal_lobe.grid import Grid

from genesis_client.swallow import note_swallowed
from infrastructure.journal import record_error

_PROGRESS_FILE = "gameplay.json"
logger = logging.getLogger(__name__)


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


# Her standing collection — games she always has, ordered by
# difficulty. Nothing is locked; she picks what she wants. The hint
# is one line — a nudge, not the answer.
COLLECTION: list[dict] = [
    {
        "name": "gravity_intro",
        "hint": "Make each object fall straight down until it reaches the bottom.",
        "train": [
            (_g([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), _g([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
            (
                _g([[2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [2, 0, 0]]),
            ),
        ],
        "test": [(_g([[0, 0, 3], [0, 0, 0], [0, 0, 0]]), _g([[0, 0, 0], [0, 0, 0], [0, 0, 3]]))],
    },
    {
        "name": "fill_intro",
        "hint": "Fill the empty space completely enclosed by the ring.",
        "train": [
            (_g([[2, 2, 2], [2, 0, 2], [2, 2, 2]]), _g([[2, 2, 2], [2, 4, 2], [2, 2, 2]])),
        ],
        "test": [(_g([[5, 5, 5], [5, 0, 5], [5, 5, 5]]), _g([[5, 5, 5], [5, 4, 5], [5, 5, 5]]))],
    },
    {
        "name": "gravity_variation",
        "hint": "Each object falls to the bottom, keeping its own color.",
        "train": [
            (
                _g([[0, 0, 0], [0, 6, 0], [0, 0, 0], [0, 0, 0]]),
                _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 6, 0]]),
            ),
        ],
        "test": [(_g([[7, 0], [0, 0], [0, 0]]), _g([[0, 0], [0, 0], [7, 0]]))],
    },
    {
        "name": "fill_margin",
        "hint": "Fill only the hole fully enclosed by the shape; change nothing else.",
        "train": [
            (
                _g(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 3, 3, 0],
                        [0, 3, 0, 3, 0],
                        [0, 3, 3, 3, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                _g(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 3, 3, 3, 0],
                        [0, 3, 8, 3, 0],
                        [0, 3, 3, 3, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ],
        "test": [
            (
                _g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
                _g([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
            )
        ],
    },
    {
        "name": "silhouette_stretch",
        "hint": "Repair the damaged region so the shape becomes whole again.",
        "train": [
            (
                _g(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 6, 6, 6, 6, 0],
                        [0, 6, 0, 1, 1, 0],
                        [0, 6, 6, 1, 1, 0],
                        [0, 6, 6, 6, 6, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                _g(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 6, 6, 6, 6, 0],
                        [0, 6, 0, 6, 6, 0],
                        [0, 6, 6, 6, 6, 0],
                        [0, 6, 6, 6, 6, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ],
        "test": [
            (
                _g(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 4, 4, 0],
                        [0, 4, 2, 2, 4, 0],
                        [0, 4, 4, 4, 4, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                _g(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 4, 4, 0],
                        [0, 4, 4, 4, 4, 0],
                        [0, 4, 4, 4, 4, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            )
        ],
    },
    {
        "name": "gravity_recall",
        "hint": "Each object falls straight down until it lands.",
        "train": [
            (
                _g([[0, 0, 0], [0, 0, 0], [0, 0, 8], [0, 0, 0]]),
                _g([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 8]]),
            ),
        ],
        "test": [
            (
                _g([[0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                _g([[0, 0, 0, 0], [0, 0, 0, 0], [0, 5, 0, 0]]),
            )
        ],
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


# Attempts without mastery before a game gets shelved — it stops
# being fun and she sets it aside. Shelved games stay playable;
# nothing is ever forced or closed.
_SHELVE_AFTER = 8

# Windowed (system) games run on the user's actual desktop — they
# raise a window, take keyboard focus, and send real key events.
# One session, then a real gap before the next launch; a user
# interrupt ends the session and restarts the gap. Internal games
# (grids, sorters…) are silent — no gap applies to them.
_SYSTEM_GAP_SECONDS = 1800.0


@dataclass
class GameAttempt:
    """The result of one go at a game."""

    task: str
    hint: str
    score: float  # this attempt's score (cell accuracy for grids)
    best: float  # running mastery after this attempt
    solved: bool  # won / exact match this attempt
    mastered: bool  # reached 1.0 — she owns this game
    rule: str  # her hypothesis description (or "none")
    nodes: int  # search effort this attempt
    total_attempts: int  # lifetime attempts on this game
    failure: str  # why it failed ("" when solved or unknown)
    family: str = "grid"  # which game family it belongs to
    # Family-specific result payload — what the attempt actually
    # produced (a classification's labels, a quantities basket...).
    # Telemetry, not presentation: callers render it themselves.
    details: dict = field(default_factory=dict)


# The drop-box: games placed into her world by someone else.
# ``<data_dir>/offered_games/*.json`` files are game specs — data,
# not code — so anything that can write a description can hand her a
# game: a person, a file sync, eventually her own comprehension.
_OFFERED_DIR = "offered_games"
_FAMILIES = (
    "grid",
    "sorter",
    "sequence",
    "assembly",
    "relations",
    "quantities",
    "classification",
    "system",
)

# System games — installed programs she plays on the desktop. These
# live in /usr/games like anyone's installed games; entries here are
# launch helpers, not games themselves.
_SYSTEM_GAME_DIRS = ("/usr/games", "/usr/local/games")

# The official ARC-AGI-3 challenge. Not a pile of game entries — one
# standing invitation on her shelf. The card behind it is the venue's:
# it opens once per window, gives every game a single recorded try,
# and closes fifteen minutes after her last move or three days after
# opening. Taking the challenge is the choice; once she's in, the
# sitting plays the card out.
_ARC_OFFICIAL_FAMILY = "arc_official"
_ARC_OFFICIAL_CARD = "official_card.json"
_ARC_OFFICIAL_ENVS = "environment_files"
_ARC_CHALLENGE = "arc-official-challenge"

# The ARC funnel — the corpora she climbs toward the official
# challenge. Two feeds land on the same shelf:
#
#   arc_*/arc2_*   grid puzzles from the ARC-AGI-1/2 task corpora,
#                  scanned from configured directories.
#   arcagi-*       the ARC-AGI-3 environments installed locally —
#                  the offline arcade, unlimited practice.
#
# Corpus roots: ``GENESIS_ARC_CORPUS`` adds os.pathsep-separated
# directories; each root's ``data/{training,evaluation}`` layout is
# understood, and a bare directory of task files works too. The
# corpora live outside the repo — wherever on the host they were
# fetched. The two defaults cover this host's layout.
_ARC_CORPUS_ENV = "GENESIS_ARC_CORPUS"
_ARC_CORPUS_DIRS = (
    ("~/testing/ARC/data", "arc", 2),
    ("~/testing/ARC-AGI-2/data", "arc2", 3),
)
_ARC_PRACTICE_FAMILY = "arcagi"
# Every env attempt's full telemetry lands here — one JSONL line per
# result, the research record the notebooks read.
_ARC_RESULTS_LOG = "results.jsonl"

# The ladder's difficulty pull: higher tiers draw harder than easy
# shelves — the funnel leans into the frontier, not the solved end.
# Internal games/offered tasks sit at level 1 by default.
_ARC_LEVEL_PULL = 0.3

# What she was *told* about specific system games — the briefing a
# person gives a new player: what the game is, the point of it, and
# where the important things live on screen. This is instruction,
# not skill — the same knowledge she'd get from the back of the box.
# Learning to actually play still happens through reward.
_SYSTEM_KNOWN: dict[str, dict] = {
    "pacman": {
        "instructions": (
            "Pac-Man — steer the eater through the maze and clear "
            "the dots. The colored ghosts cost a life when they "
            "catch you. The eater keeps heading the way you last "
            "steered. Score and Lives count down the left edge."
        ),
        "goal": "clear the maze of dots",
        "max_steps": 150,
        "observe": {
            # Yellow pixels in the maze = the dots. Watching that
            # count fall IS the objective — the honest meter.
            "meter": {
                "region": [127, 15, 538, 415],
                "color": [224, 224, 0],
                "tol": 70,
                "direction": "down",
                "scale": 0.02,
            },
            # The displays she was told to watch: score digits and
            # lives digits on the left edge. When the lives readout
            # moves, a life was spent — that stings.
            "regions": [
                {"box": [6, 102, 78, 16], "reward": 0.15, "name": "score"},
                {"box": [6, 150, 78, 16], "reward": -0.7, "name": "lives"},
            ],
            # The dangerous ones — the ghost hues, and how close is
            # "too close". She is the fat yellow blob.
            "danger": {
                "player_color": [224, 224, 0],
                "danger_colors": [
                    [224, 0, 0],
                    [224, 64, 0],
                    [224, 128, 224],
                    [192, 32, 128],
                ],
                "near_px": 90,
                "tol": 70,
            },
        },
    },
}
_SYSTEM_SKIP = frozenset(
    {
        "gamemodelist",
        "gamemoderun",
        "gamemode-simulate-game",
    }
)
_SYSTEM_ACTIONS = ["up", "down", "left", "right", "space"]


def _rectangular(rows: Any) -> bool:
    return (
        isinstance(rows, list)
        and bool(rows)
        and all(
            isinstance(r, list) and bool(r) and all(isinstance(c, int) for c in r) for r in rows
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
    (Kuhn's bipartite matching — offered games must be possible.)"""
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
        blocks.append({str(k): str(v) for k, v in attrs.items() if k != "attrs"})
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
            scaffold = int(gen["scaffold"]) if gen.get("scaffold") is not None else None
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
        scaffold = int(raw["scaffold"]) if raw.get("scaffold") is not None else None
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
    from worlds.relations import Arrangement, Goal, Thing

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
    if not isinstance(objects_raw, list) or not 2 <= len(objects_raw) <= 7:
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
            attrs = tuple((str(k), int(v)) for k, v in attrs_raw.items())
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
        if rel not in _RELATION_RELS or a not in names or b not in names or a == b:
            return None
        attr = g.get("attr")
        if rel == "ordered" and not isinstance(attr, str):
            return None
        goals.append(Goal(rel, a, b, str(attr) if attr is not None else None))
    if not goals:
        return None
    if not Arrangement(positions, things, goals).solvable():
        return None
    return {
        "positions": positions,
        "objects": objects_raw,
        "goals": [{"rel": g.rel, "a": g.a, "b": g.b, "attr": g.attr} for g in goals],
    }


def _quantities_spec(raw: dict) -> dict[str, Any] | None:
    """Normalize a make-N spec, or None when malformed/impossible."""
    from worlds.quantities import Basket, Group

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
        groups_raw = [{"name": k, "count": v} for k, v in groups_raw.items()]
    if not isinstance(groups_raw, list) or not 2 <= len(groups_raw) <= 12:
        return None
    groups: list[Group] = []
    names: set[str] = set()
    try:
        target = int(raw["target"])
        for i, g in enumerate(groups_raw):
            name = str(g.get("name", f"group_{i}")) if isinstance(g, dict) else f"group_{i}"
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
    from worlds.classification import eval_predicate, valid_predicate

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
                    or not all(isinstance(a, str) and a.strip() for a in attrs)
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
    examples = [] if examples_raw is None else _items(examples_raw, require_label=True)
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
            if eval_predicate(pred, frozenset(entry["has"])) != bool(entry["label"]):
                return None
    # Omit empty/None fields so the normalized dict re-validates
    # identically — ``offer()`` writes this back into the drop-box
    # and ``offered_games()`` normalizes it again on read.
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
    """Validate an offered-game file into a canonical game spec.

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
    - ``system``     — ``{"command": "/usr/games/..."}`` — an
                       installed game; the command must live inside
                       the system game directories
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
        spec = _assembly_spec(raw)
        if spec is None:
            return None
        task.update(spec)
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
    if family == "system":
        spec = _system_offered(raw)
        if spec is None:
            return None
        task.update(spec)
        return task
    return None


def _assembly_spec(raw: dict) -> dict | None:
    gen = raw.get("generate")
    if not isinstance(gen, dict):
        return None
    try:
        rows, cols = int(gen["rows"]), int(gen["cols"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (1 <= rows <= 8 and 1 <= cols <= 8):
        return None
    return {"generate": {"rows": rows, "cols": cols, "seed": gen.get("seed")}}


def _system_offered(raw: dict) -> dict | None:
    """An offered system game may only name an executable inside the
    system game directories — an offer can't point her at arbitrary
    programs."""
    raw_cmd = str(raw.get("command", "")).strip()
    exe = Path(raw_cmd).resolve() if raw_cmd else None
    allowed = exe is not None and any(
        str(exe).startswith(str(Path(d).resolve()) + os.sep) for d in _SYSTEM_GAME_DIRS
    )
    if exe is None or not allowed or not exe.is_file() or not os.access(exe, os.X_OK):
        return None
    spec: dict[str, Any] = {
        "command": str(exe),
        "actions": [str(a) for a in raw.get("actions") or _SYSTEM_ACTIONS],
    }
    for opt in (
        "cell_px",
        "max_steps",
        "time_budget",
        "title_pattern",
        "start_keys",
        "instructions",
        "goal",
    ):
        if raw.get(opt) is not None:
            spec[opt] = raw[opt]
    observe = _normalize_observe(raw.get("observe"))
    if observe is not None:
        spec["observe"] = observe
    return spec


def _normalize_observe(raw: Any) -> dict | None:
    """Validate an ``observe`` block — what she was told to watch.

    Regions are [x, y, w, h] pixel boxes inside the game window;
    colors are [r, g, b]. Badly formed entries are dropped, never
    trusted — a spec that can't describe its own display simply
    gets less perception.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}

    def _box(v: Any) -> list[int] | None:
        if (
            isinstance(v, (list, tuple))
            and len(v) == 4
            and all(isinstance(n, (int, float)) and n >= 0 for n in v)
        ):
            return [int(n) for n in v]
        return None

    def _color(v: Any) -> list[int] | None:
        if (
            isinstance(v, (list, tuple))
            and len(v) == 3
            and all(isinstance(n, (int, float)) and 0 <= n <= 255 for n in v)
        ):
            return [int(n) for n in v]
        return None

    meter = raw.get("meter")
    if isinstance(meter, dict):
        region, color = _box(meter.get("region")), _color(meter.get("color"))
        if region and color:
            out["meter"] = {
                "region": region,
                "color": color,
                "tol": int(meter.get("tol", 60)),
                "direction": ("up" if meter.get("direction") == "up" else "down"),
                "scale": float(meter.get("scale", 0.02)),
            }

    regions = []
    for reg in raw.get("regions") or []:
        if not isinstance(reg, dict):
            continue
        box = _box(reg.get("box"))
        if box:
            regions.append(
                {
                    "box": box,
                    "reward": max(-1.5, min(1.5, float(reg.get("reward", 0.0)))),
                    "name": str(reg.get("name", ""))[:32],
                }
            )
    if regions:
        out["regions"] = regions[:8]

    danger = raw.get("danger")
    if isinstance(danger, dict):
        player = _color(danger.get("player_color"))
        colors = [c for c in (_color(v) for v in danger.get("danger_colors") or []) if c]
        if player and colors:
            out["danger"] = {
                "player_color": player,
                "danger_colors": colors[:8],
                "near_px": max(10, int(danger.get("near_px", 90))),
                "tol": int(danger.get("tol", 70)),
            }

    return out or None


@dataclass
class GamePlay:
    """Genesis's persistent game collection and play record.

    Lives inside her Mind: owns the gating and mastery state, runs
    attempts through each family's agent when her volition raises the
    play urge.
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
    # Test-output fingerprints already disproven for this task. Unlike
    # rule strings, these prune semantically equivalent hypotheses that
    # make the same wrong prediction through different representations.
    failed_predictions: dict[str, list[str]] = field(default_factory=dict)
    # The visual cortex, when the Mind wires one in — perceptual
    # tasks (a sorter you must look at) need it; symbolic ones don't.
    cortex: Any = None
    # The general player — one player across every environment:
    # built-in games, offered games, installed system games. Her
    # value memory is the growing edge of "knowing how to play."
    _player: Any = field(default=None, repr=False)
    # Offered-game file cache: filename -> (mtime, task|None).
    _offered_cache: dict[str, tuple[float, dict | None]] = field(default_factory=dict)
    # Windowed envs persist per game so a session reattaches to the
    # window she left up (X11Game.reset's reattach path) instead of
    # stacking a fresh process on the desktop with every firing.
    _system_envs: dict[str, Any] = field(default_factory=dict)
    # Monotonic time the last windowed episode ended — the gap clock.
    # -inf = never played one, so the first session isn't gated by a
    # clock that started at boot.
    _system_last_end: float = float("-inf")
    # The card-window key the stakes brief was last announced for —
    # she hears what the challenge is once per window, not every firing.
    _arc_briefed: str | None = None
    # Corpus spec cache — task files don't move inside a session.
    _corpus_specs: list[dict] | None = None

    def __post_init__(self) -> None:
        self._progress_path = Path(self.data_dir) / _PROGRESS_FILE
        if self._player is None:
            from worlds.player import GamePlayer

            self._player = GamePlayer()
        self._load()

    def _load(self) -> None:
        # Continuity: the file was spatial_practice.json while the
        # machinery lived under spatial/. Play predates the rename —
        # read whichever exists so mastery isn't lost.
        path = self._progress_path
        if not path.exists():
            legacy = path.with_name("spatial_practice.json")
            if legacy.exists():
                path = legacy
        try:
            data = json.loads(path.read_text())
            self.mastery = {str(k): float(v) for k, v in data.get("mastery", {}).items()}
            self.attempts = {str(k): int(v) for k, v in data.get("attempts", {}).items()}
            self.failed = {str(k): [str(r) for r in v] for k, v in data.get("failed", {}).items()}
            self.failed_predictions = {
                str(k): [str(r) for r in v] for k, v in data.get("failed_predictions", {}).items()
            }
            # Learned play values — "how she plays pacman" is a
            # skill like any other; it survives restarts.
            skills = data.get("skills")
            if isinstance(skills, dict) and self._player is not None:
                self._player._values = {
                    str(k): {str(a): float(v) for a, v in row.items()}
                    for k, row in skills.items()
                    if isinstance(row, dict)
                }
        except (OSError, json.JSONDecodeError, ValueError):
            self.mastery = {}
            self.attempts = {}
            self.failed = {}

    def _save(self) -> None:
        tmp = self._progress_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {
                        "mastery": self.mastery,
                        "attempts": self.attempts,
                        "failed": self.failed,
                        "failed_predictions": self.failed_predictions,
                        "skills": (self._player._values if self._player else {}),
                    },
                    indent=2,
                )
            )
            tmp.replace(self._progress_path)
        except OSError as e:
            # Progress silently vanishing is the worst outcome — record
            # it so a lost mastery file is at least visible in the
            # cognitive journal.
            record_error("worlds.play.save", e)

    def offered_dir(self) -> Path:
        """The drop-box: ``<data_dir>/offered_games/``."""
        return Path(self.data_dir) / _OFFERED_DIR

    def offer(self, task: dict) -> Path | None:
        """Write a normalized task spec into the drop-box.

        The same channel the CLI's ``/play add`` uses — anything
        that produces a valid spec (including a problem heard in
        conversation) lands in the inner world as a file. Returns the
        written path, or None when the task can't be persisted. The
        offer cache entry is invalidated so the new file is seen on
        the next ``offered_games()`` call.
        """
        name = str(task.get("name", "")).strip()
        if not name or task.get("family") not in _FAMILIES:
            return None
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)
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

    def offered_games(self) -> list[dict]:
        """Games someone placed in her world beyond her own collection.

        Every ``*.json`` file in the drop-box is a game spec parsed by
        ``normalize_offered``; malformed files are skipped (never
        fatal), files are cached by mtime, and names colliding with
        the collection or an earlier offer are ignored.
        """
        out: list[dict] = []
        seen: set[str] = {str(t["name"]) for t in COLLECTION}
        # Continuity: games offered before the rename still live in
        # offered_puzzles/ — read both drop-boxes so nothing someone
        # handed it is silently lost.
        files: list[Path] = []
        for directory in (
            self.offered_dir(),
            Path(self.data_dir) / "offered_puzzles",
        ):
            try:
                files.extend(directory.glob("*.json"))
            except OSError:
                continue
        files.sort(key=lambda f: f.name)
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

    def _arc_corpus_games(self) -> list[dict]:
        """The ARC-AGI-1/2 task corpora — grid puzzles, lazily bound.

        Scanning yields one thin spec per ``*.json`` task file; the
        pairs themselves load only if she actually sits at the task
        (``_corpus`` is resolved in ``attempt``). Corpus roots come
        from ``GENESIS_ARC_CORPUS`` plus the host defaults — corpora
        live outside the repo, wherever they were fetched. Prefixes
        namespace the sets: ``arc_*`` ARC-1, ``arc2_*`` ARC-2, and
        env-added roots join ARC-2's tier — newer corpora are the
        harder end. The spec list is cached: task files don't move.
        """
        if self._corpus_specs is not None:
            return self._corpus_specs
        roots: list[tuple[str, str, int]] = [
            (os.path.expanduser(d), prefix, level) for d, prefix, level in _ARC_CORPUS_DIRS
        ]
        for extra in os.environ.get(_ARC_CORPUS_ENV, "").split(os.pathsep):
            extra = extra.strip()
            if extra:
                roots.append((os.path.expanduser(extra), "arc2", 3))
        out: list[dict] = []
        # Names already covered elsewhere — an offered copy of a task
        # (her first ARC games arrived that way) keeps its entry; the
        # corpus spec for the same id would only double the shelf.
        seen: set[str] = {str(t["name"]) for t in COLLECTION}
        seen |= {str(t["name"]) for t in self.offered_games()}
        for root, prefix, level in roots:
            dirs = [
                sub for sub in (Path(root) / "training", Path(root) / "evaluation") if sub.is_dir()
            ] or ([Path(root)] if Path(root).is_dir() else [])
            for directory in dirs:
                try:
                    files = sorted(directory.glob("*.json"))
                except OSError:
                    continue
                for f in files:
                    name = f"{prefix}_{f.stem}"
                    if name in seen:
                        continue
                    seen.add(name)
                    out.append(
                        {
                            "name": name,
                            "family": "grid",
                            "hint": (
                                f"An {prefix.upper()}-corpus puzzle — "
                                "infer the rule, produce the grid."
                            ),
                            "level": level,
                            "_corpus": str(f),
                        }
                    )
        self._corpus_specs = out
        return out

    def _arc_practice_games(self) -> list[dict]:
        """The local ARC-AGI-3 environments — her practice arcade.

        Every game installed under ``<data>/arc/environment_files``
        is playable through the offline arcade: same engines the card
        uses, no venue clock, unlimited reps. The name matches the
        card's game_id so card mastery and practice mastery are the
        same record.
        """
        out: list[dict] = []
        env_dir = Path(self.data_dir) / "arc" / _ARC_OFFICIAL_ENVS
        try:
            games = sorted(d for d in env_dir.iterdir() if d.is_dir())
        except OSError:
            return out
        for d in games:
            try:
                versions = sorted(v.name for v in d.iterdir() if (v / "metadata.json").is_file())
            except OSError:
                versions = []
            for ver in versions:
                game_id = f"{d.name}-{ver}"
                out.append(
                    {
                        "name": f"arcagi-{game_id}",
                        "family": _ARC_PRACTICE_FAMILY,
                        "game_id": game_id,
                        "hint": "A local ARC-AGI-3 environment — a video game to win.",
                        "level": 4,
                    }
                )
        return out

    def collection(self) -> list[dict]:
        """Every game she can play — her collection, always there.

        The built-in collection stands on its own; games someone
        hands her join it, and so do the games installed on her
        system. Nothing gates any of them — she plays what she
        wants when she wants.
        """
        return [
            *self.offered_games(),
            *self._arc_official_games(),
            *self._arc_corpus_games(),
            *self._arc_practice_games(),
            *COLLECTION,
            *self._system_games(),
        ]

    def arc_challenge(self) -> dict | None:
        """The official challenge's standing on her shelf, or None.

        One standing invitation, not a pile of game entries: the card
        is the venue's — it opens once, gives every game a single
        recorded try, and closes on its own clock — so what she picks
        is *the challenge itself*. Game-level choices happen inside
        the sitting, when each one is actually in front of her.

        A live card keeps the challenge on the shelf as the open
        commitment it is; a card closed inside its window leaves the
        shelf empty (it cannot be replaced); no card at all means the
        downloaded environment list stands in for it — taking the
        challenge then is what opens the card.
        """
        if not os.getenv("ARC_API_KEY", "").strip():
            return None
        if not self._arc_unlocked():
            return None
        arc_dir = Path(self.data_dir) / "arc"
        try:
            card = json.loads((arc_dir / _ARC_OFFICIAL_CARD).read_text())
        except (OSError, ValueError):
            card = None
        now = time.time()
        if isinstance(card, dict) and now < card.get("expires_at", 0):
            if card.get("closed"):
                return None
            pending = [g for g in card.get("games", []) if g.get("status") == "pending"]
            if not pending:
                return None
            try:
                from worlds.arc import ArcOfficial

                idle_limit = ArcOfficial.IDLE_LIMIT_S
            except ImportError:
                idle_limit = 900.0
            # How far into the idle window the card already is — the
            # deadline pressure the appeal of the open commitment
            # scales with.
            idle_used = max(
                0.0,
                min(1.0, (now - card.get("last_action_at", now)) / idle_limit),
            )
            return {
                "key": card["scorecard_id"],
                "card": card,
                "pending": pending,
                "n_games": len(card["games"]),
                "live": True,
                "idle_pressure": idle_used,
            }
        # No live card. The dedup key still distinguishes windows:
        # an expired card's window id is its expiry, an absent card's
        # is none.
        key = f"pre:{card['expires_at']}" if isinstance(card, dict) else "pre:none"
        try:
            env_dir = arc_dir / _ARC_OFFICIAL_ENVS
            ids = sorted(
                d.name for d in env_dir.iterdir() if d.is_dir() and any(d.glob("*/metadata.json"))
            )
        except OSError:
            return None
        if not ids:
            return None
        return {
            "key": key,
            "card": card if isinstance(card, dict) else None,
            "pending_ids": ids,
            "n_games": len(ids),
            "live": False,
            "idle_pressure": 0.0,
        }

    def _arc_unlocked(self) -> bool:
        """Whether she's earned the official challenge — one ARC win.

        The card is the summit, not the starting line: it stands on
        the shelf only after she's mastered at least one ARC-family
        game — any grid puzzle or environment. Mastery is the
        credential; until then the challenge stays off the shelf and
        the funnel (corpora, practice arcade) is the climb.
        """
        prefixes = ("arc_", "arc2_", "arcagi-")
        return any(m >= 1.0 for n, m in self.mastery.items() if n.startswith(prefixes))

    def _arc_official_games(self) -> list[dict]:
        """The official ARC-AGI-3 challenge — one entry, when offered."""
        ch = self.arc_challenge()
        if ch is None:
            return []
        return [
            {
                "name": _ARC_CHALLENGE,
                "family": _ARC_OFFICIAL_FAMILY,
                "hint": (
                    "The official ARC-AGI-3 challenge — one recorded try "
                    "per game, on a card with its own clock."
                ),
                # What she was told the point of it is — her settling-in
                # moment composes the words from this.
                "goal": "win the games — a real prize, for her creator",
                "live": ch["live"],
                "idle_pressure": ch["idle_pressure"],
            }
        ]

    def arc_brief_due(self) -> str | None:
        """The challenge's stakes, owed to her once per card window.

        She can't choose what she doesn't know: before a pick that
        could take the challenge she hears what it is — how many
        games, the move budget and pace, the card's clock, and what
        winning is for. Announced once per card or window — after
        that the knowledge is hers (the announce grounds it into her
        concept network), not something repeated at her.
        """
        ch = self.arc_challenge()
        if ch is None or ch["key"] == self._arc_briefed:
            return None
        self._arc_briefed = ch["key"]
        try:
            from worlds.arc import challenge_brief
        except ImportError:
            return None
        pending = len(ch["pending"]) if ch.get("pending") else None
        return challenge_brief(ch["n_games"], pending=pending if ch["live"] else None)

    def _system_games(self) -> list[dict]:
        """The games installed on her computer — always playable.

        Scans the system game directories for executables; each one
        is a windowed environment she can launch and play. Skips
        launch helpers and anything already in her collection or
        drop-box.
        """
        known = {str(t["name"]) for t in COLLECTION}
        known |= {str(t["name"]) for t in self.offered_games()}
        out: list[dict] = []
        for directory in _SYSTEM_GAME_DIRS:
            try:
                entries = sorted(Path(directory).iterdir())
            except OSError:
                continue
            for exe in entries:
                name = exe.name
                if (
                    name in known
                    or name in _SYSTEM_SKIP
                    or name.startswith(".")
                    or not exe.is_file()
                    or not os.access(exe, os.X_OK)
                ):
                    continue
                spec: dict[str, Any] = {
                    "name": name,
                    "family": "system",
                    "command": str(exe),
                    "actions": list(_SYSTEM_ACTIONS),
                    "hint": "An installed game — watch, press keys, learn.",
                }
                extra = _SYSTEM_KNOWN.get(name)
                if extra:
                    spec.update(extra)
                out.append(spec)
        return out

    def _appeal(self, game: dict) -> float:
        """How much she wants to play this game right now.

        Desire, not a syllabus: unplayed games draw curiosity, an
        unfinished challenge pulls harder the closer she is (the
        near-miss grip), a solved game stays replayable but dimmer,
        and a game that has beaten her many times goes on the shelf —
        she sets it aside rather than grinding into frustration.
        """
        name = str(game["name"])
        if game.get("family") == _ARC_OFFICIAL_FAMILY and game.get("live"):
            # An open card is a commitment on a clock — its pull rises
            # as the idle window runs down, the way a deadline sharpens
            # an unfinished task.
            return 2.5 + 2.0 * float(game.get("idle_pressure", 0.0))
        mastery = self.mastery.get(name, 0.0)
        attempts = self.attempts.get(name, 0)
        if mastery >= 1.0:
            return 0.4  # solved — replayable, not magnetic
        appeal = 1.0 + (1.0 - mastery) * 0.6
        if attempts == 0:
            appeal += 0.5  # never played — novelty
        elif attempts >= _SHELVE_AFTER:
            appeal *= 0.15  # shelved — it stopped being fun
        # The difficulty ladder: higher-tier games pull harder, so
        # the ARC funnel leans toward the frontier.
        appeal += _ARC_LEVEL_PULL * (float(game.get("level", 1)) - 1.0)
        return appeal

    def system_ready(self) -> bool:
        """Whether a windowed game may launch — the desktop gap lapsed."""
        return time.monotonic() - self._system_last_end >= _SYSTEM_GAP_SECONDS

    def pick(
        self,
        *,
        rng: random.Random | None = None,
        allow_system: bool = True,
    ) -> dict | None:
        """Choose a game she'd enjoy playing — her collection, her choice.

        Games are always there; this is the wanting. Selection is
        weighted by appeal, so she gravitates toward unfinished
        challenges and new games without being driven through them
        in order. ``allow_system=False`` takes windowed games off
        the shelf this pick — while the desktop gap is running she
        reaches for an internal game instead of nothing. Returns
        None only when nothing eligible is left.
        """
        games = self.collection()
        if not allow_system:
            games = [g for g in games if g.get("family") != "system"]
        # Mastered games are archived — they stay in the collection
        # record and can still be pulled by name, but they stop
        # drawing her picks; the shelf is for unfinished work. The
        # official challenge is exempt: its standing is the venue's,
        # not a mastery score.
        games = [
            g
            for g in games
            if g.get("family") == _ARC_OFFICIAL_FAMILY
            or self.mastery.get(str(g["name"]), 0.0) < 1.0
        ]
        if not games:
            return None
        chooser = rng or random
        weights = [self._appeal(g) for g in games]
        return chooser.choices(games, weights=weights, k=1)[0]

    def has_games(self) -> bool:
        """Whether she owns any game at all — a volition signal."""
        return bool(self.collection())

    def attempt(
        self,
        reasoner: SpatialReasoner,
        time_budget: float = 30.0,
        task: dict | None = None,
        on_step: Any = None,
        should_stop: Any = None,
        inducer: DorsolateralPFC | None = None,
    ) -> GameAttempt | None:
        """One go at a game.

        ``task`` defaults to ``pick()`` — the volition path, her
        choosing what she wants to play. A caller holding a specific
        normalized spec (a game just compiled from conversation, say)
        passes it directly so *that* game runs now; bookkeeping
        (attempts, mastery, persistence) is identical either way.
        ``should_stop`` is the user-interrupt channel, honored by
        windowed (system) episodes between moves. ``inducer`` is her
        shared RLPFC rule-induction engine — injected by the organism
        so classification play learns across games instead of starting
        cold each attempt. Returns None when she owns no games.
        """
        task = task or self.pick()
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
            return self._attempt_classification(task, reasoner, inducer)
        if family == "system":
            return self._attempt_system(task, on_step=on_step, should_stop=should_stop)
        if family == _ARC_OFFICIAL_FAMILY:
            return self._attempt_arc(task)
        if family == _ARC_PRACTICE_FAMILY:
            return self._attempt_arc_env(task)

        # Corpus tasks carry their file, not their pairs — the grids
        # load only now that she's actually sitting at this one.
        if "_corpus" in task:
            try:
                raw = json.loads(Path(task["_corpus"]).read_text())
            except (OSError, json.JSONDecodeError):
                return None
            pairs = []
            for section in ("train", "test"):
                parsed = _pairs(
                    [
                        [p.get("input"), p.get("output")]
                        for p in raw.get(section, [])
                        if isinstance(p, dict)
                    ]
                )
                if not parsed:
                    return None
                pairs.append(parsed)
            task = {**task, "train": pairs[0], "test": pairs[1]}

        name = str(task["name"])
        tests = [i for i, _ in task["test"]]
        expected = [o for _, o in task["test"]]
        sol = reasoner.solve(
            task["train"],
            tests,
            time_budget=time_budget,
            exclude_rules=set(self.failed.get(name, [])),
            exclude_predictions=set(self.failed_predictions.get(name, [])),
        )

        guess_sets = sol.guesses or ([sol.predictions] if sol.predictions else [])
        score = 0.0
        solved = False
        for guesses in guess_sets:
            acc = min(
                (_cell_match(p, e) for p, e in zip(guesses, expected, strict=True)),
                default=0.0,
            )
            score = max(score, acc)
            if all(p == e for p, e in zip(guesses, expected, strict=True)):
                solved = True
                score = 1.0

        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)

        failure = "" if solved else self._record_miss(name, sol, reasoner)
        self._save()
        try:
            # The held-out test is the external verifier: only its
            # outcome decides whether a train-verified rule becomes a
            # reusable skill.
            reasoner.record_task_outcome(
                sol,
                success=solved,
                score=score,
                examples=list(task["train"]),
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "worlds.play.attempt",
                e,
            )

        return GameAttempt(
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

    def _attempt_arc_env(self, task: dict) -> GameAttempt | None:
        """One go at an ARC-AGI-3 environment — the practice floor.

        Same embodied loop the official sitting uses, minus the card:
        local engines offline, no venue clock, no one-try rule —
        practice reps are how she earns the challenge. The result
        books under the game's own id and lands in the research log.
        """
        mind = getattr(self, "mind", None)
        if mind is None:
            return None
        try:
            from worlds.arc import ArcPractice
        except ImportError:
            return None
        arc_dir = Path(self.data_dir) / "arc"
        adapter = ArcPractice(
            mind,
            environments_dir=arc_dir / _ARC_OFFICIAL_ENVS,
            recordings_dir=arc_dir / "recordings",
            instructions="this is a video game",
            step_delay_s=1.0,
        )
        try:
            result = adapter.play(str(task["game_id"]))
        except Exception as e:  # noqa: BLE001 — a world that faults isn't a turn
            note_swallowed("worlds.play.arc_env", e)
            return None
        self._record_arc_result(result)
        name = str(task["name"])
        return GameAttempt(
            task=name,
            hint=str(task["hint"]),
            score=self.mastery.get(name, 0.0),
            best=self.mastery.get(name, 0.0),
            solved=result.won,
            mastered=result.won,
            rule="environment play",
            nodes=result.steps,
            total_attempts=self.attempts.get(name, 0),
            failure="" if result.won else "; ".join(result.reasons[:3]),
            family=str(task.get("family", "")),
            details={
                "game_id": result.game_id,
                "state": result.state,
                "levels_completed": result.levels_completed,
                "win_levels": result.win_levels,
            },
        )

    def _attempt_arc(self, task: dict) -> GameAttempt | None:
        """The sitting — she takes up the official challenge and plays the card out.

        One urge firing brings her here; everything after is the
        sitting itself. If no card lives yet, the pick opens it — the
        commitment is hers, but the venue's clocks start with the open,
        so an unwell pick doesn't open it (the challenge simply stays
        on the shelf for when she's ready). Inside the sitting she
        picks each pending game herself and ``attempt()`` burns it for
        real. Returns None when nothing was played.
        """
        mind = getattr(self, "mind", None)
        if mind is None:
            return None
        from worlds.arc import ArcOfficial

        arc_dir = Path(self.data_dir) / "arc"
        card_path = arc_dir / _ARC_OFFICIAL_CARD
        try:
            card = json.loads(card_path.read_text())
        except (OSError, ValueError):
            card = None
        now = time.time()
        live = (
            isinstance(card, dict)
            and not card.get("closed")
            and now < card.get("expires_at", 0)
            and now - card.get("last_action_at", 0) < ArcOfficial.IDLE_LIMIT_S
        )
        if not live:
            # Opening is the commitment — the idle clock starts with
            # it, so the card only opens when she's fit to feed it.
            try:
                if not mind.readiness().ok:
                    return None
            except Exception:  # noqa: BLE001 — an unreachable gate is "not ready"
                return None
        was_gaming = getattr(mind, "is_gaming", False)
        if not was_gaming:
            # Game mode for the whole sitting: engagement pins her
            # attention on the card so the venue's idle clock never
            # outruns a wandering urge — she stays at the arcade until
            # the card is done or sealed.
            mind.enter_game_mode()
        try:
            adapter = ArcOfficial(
                mind,
                card_path=card_path,
                environments_dir=arc_dir / _ARC_OFFICIAL_ENVS,
                recordings_dir=arc_dir / "recordings",
                instructions="this is a video game",
                step_delay_s=1.0,
            )
            try:
                adapter.open_or_resume()
            except RuntimeError:
                return None  # sealed inside its window — nothing to play
            interaction_mark = getattr(mind, "_last_interaction_time", 0.0)
            return self._play_card(adapter, task, interaction_mark)
        finally:
            if not was_gaming:
                mind.exit_game_mode()

    def _play_card(
        self, adapter: Any, task: dict, interaction_mark: float = 0.0
    ) -> GameAttempt | None:
        """One sitting at the card — she stays until it's done or she walks away.

        The card's clock, not her urge, bounds the sitting: between
        games she picks what draws her next; an unwell gate waits
        inside the idle budget rather than burning the game; leaving
        game mode, sleeping, or being suppressed all end the sitting
        honestly — whatever the card holds then is the result.
        """
        mind = getattr(self, "mind", None)
        name = str(task["name"])
        played: list[Any] = []
        foreground = getattr(mind, "_foreground_active", None)
        while True:
            if mind is not None and (
                getattr(mind, "_is_sleeping", False)
                or getattr(mind, "_is_meditating", False)
                or getattr(mind, "_is_teaching", False)
                or getattr(mind, "_suppress_volition", False)
                or not getattr(mind, "is_gaming", True)
                or (foreground is not None and foreground.is_set())
                or getattr(mind, "_last_interaction_time", 0.0) > interaction_mark
            ):
                break  # she left the arcade — the card decides what's left
            card = adapter.card_state()
            if card is None or card.get("closed"):
                break
            now = time.time()
            if now >= card.get("expires_at", 0):
                # Her noticing is what seals it — the same bookkeeping
                # an attempt would have found.
                adapter.attempt_next()
                break
            idle_left = adapter.IDLE_LIMIT_S - (now - card.get("last_action_at", 0))
            if idle_left <= 0:
                adapter.attempt_next()
                break
            pending = [g for g in card.get("games", []) if g.get("status") == "pending"]
            if not pending:
                adapter.attempt_next()  # nothing to enter — seals it complete
                break
            result = adapter.attempt(str(self._pick_pending(pending)["game_id"]))
            if result is None:
                break  # the card closed under the sitting
            if result.state == "UNWELL":
                # She never entered — nothing was burned. Wait inside
                # the remaining idle budget for her to settle; if it
                # runs out, the card seals as it stands.
                if idle_left <= 30.0:
                    break
                time.sleep(min(30.0, max(0.0, idle_left - 5.0)))
                continue
            played.append(result)
            self._record_arc_result(result)
        if not played:
            return None
        wins = sum(1 for r in played if r.state == "WIN")
        score = sum(
            r.levels_completed / r.win_levels if r.win_levels else 0.0 for r in played
        ) / len(played)
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        card = adapter.card_state() or {}
        return GameAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=bool(played) and wins == len(played),
            mastered=bool(played) and wins == len(played),
            rule="none",
            nodes=sum(r.steps for r in played),
            total_attempts=self.attempts[name],
            failure="" if wins == len(played) else played[-1].state.lower(),
            family=_ARC_OFFICIAL_FAMILY,
            details={
                "scorecard_id": card.get("scorecard_id"),
                "games": len(played),
                "won": wins,
                "card_closed": bool(card.get("closed")),
                "card_reason": card.get("close_reason"),
            },
        )

    def _pick_pending(self, pending: list[dict]) -> dict:
        """Which pending card game she reaches for next — appeal over
        the card's own list. Her history with a game (a prior window's
        mastery or shelfings) still weighs in."""
        weights = [self._appeal({"name": f"arcagi-{g['game_id']}"}) for g in pending]
        return random.choices(pending, weights=weights, k=1)[0]

    def _record_arc_result(self, result: Any) -> None:
        """Book an env attempt under the game's own name — one try, kept.

        The full telemetry also lands in ``<data>/arc/results.jsonl``
        — the research record the notebooks read. Local reps and card
        tries share the same log; the card's ``scorecard_id`` marks
        which lines were official.
        """
        name = f"arcagi-{result.game_id}"
        score = result.levels_completed / result.win_levels if result.win_levels else 0.0
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        try:
            line = {
                "at": time.time(),
                "game_id": result.game_id,
                "state": result.state,
                "levels_completed": result.levels_completed,
                "win_levels": result.win_levels,
                "steps": result.steps,
                "resets": result.resets,
                "score": result.score,
                "scorecard_id": result.scorecard_id,
                "level_steps": list(result.level_steps),
                "prior_learning": result.prior_learning,
                "decision_sources": dict(result.decision_sources),
                "recovery_waits": result.recovery_waits,
            }
            log_path = Path(self.data_dir) / "arc" / _ARC_RESULTS_LOG
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
        except (OSError, TypeError, ValueError) as e:
            note_swallowed("worlds.play.arc_results_log", e)

    def _record_miss(self, name: str, sol: Any, reasoner: SpatialReasoner) -> str:
        """Learn from the miss: rules that verified on the training
        pairs but guessed wrong on test are proven counterexamples —
        record them so the next attempt can't walk the same path.
        When nothing verified, the best near-miss is recorded so a
        retry pushes past it rather than stalling on it again.
        Returns the failure summary."""
        proven_wrong = list(sol.verified_rules)
        if sol.hypothesis is not None:
            proven_wrong.append(sol.hypothesis.describe())
        known = self.failed.setdefault(name, [])
        for rule in proven_wrong:
            if rule not in known:
                known.append(rule)
        del known[32:]
        # Learn from the observed wrong answer, not just its rule
        # spelling. Different hypotheses can be behaviorally equivalent
        # on the held-out test; the next attempt must move past that
        # prediction without excluding the whole transform family.
        bad_predictions = self.failed_predictions.setdefault(name, [])
        for guesses in sol.guesses:
            key = reasoner.prediction_key(guesses)
            if key not in bad_predictions:
                bad_predictions.append(key)
        del bad_predictions[32:]
        if sol.verified_rules:
            return f"overfit: {sol.verified_rules[0]} verified on train but missed test"
        if sol.failure is not None:
            return sol.failure.describe()
        return ""

    def _attempt_sorter(self, task: dict, reasoner: SpatialReasoner) -> GameAttempt:
        """One sorter episode through the competence substrate."""
        from worlds.sorter import PerceptualSorter
        from worlds.sorter_agent import (
            PerceptualSorterAgent,
            SorterAgent,
        )

        name = str(task["name"])
        # A "perceptual" sorter is seen, not read: the world renders
        # its pieces and apertures, the oracle stops leaking `matched`
        # on candidates, and the agent learns seen-similarity→cost
        # instead of stated-attrs→cost. Falls back to symbolic when
        # no cortex is wired in.
        perceptual = bool(task.get("perceptual")) and self.cortex is not None
        sorter = self._build_sorter(task, perceptual, name)
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
        return GameAttempt(
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
                else (f"{res.state.lower()}: {res.placed}/{len(sorter.slots)} slots filled")
            ),
            family="sorter",
            details={"placements": placements} if placements else {},
        )

    def _build_sorter(self, task: dict, perceptual: bool, name: str) -> ShapeSorter:
        """Instantiate the sorter — generated or literal spec — wrapped
        in PerceptualSorter when the cortex is wired."""
        from worlds.sorter import Block, PerceptualSorter, ShapeSorter, Slot

        gen = task.get("generate")
        if gen:
            seed = gen.get("seed")
            base = ShapeSorter.generate(
                int(gen["slots"]),
                random.Random(seed if seed is not None else random.randrange(2**30)),
                difficulty=int(gen["difficulty"]),
                decoys=int(gen["decoys"]),
                has_lid=bool(gen.get("lid", True)),
            )
            if not perceptual:
                return base
            return PerceptualSorter(
                base.slots,
                [*base.pool.values(), *base.in_top],
                has_lid=base.has_lid,
                cortex=self.cortex,
                seed=hash(name) & 0x7FFFFFFF,
            )
        sorter = ShapeSorter(
            [Slot(i, tuple(sorted(s.items()))) for i, s in enumerate(task["slots"])],
            [Block(i, tuple(sorted(b.items()))) for i, b in enumerate(task["blocks"])],
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
        return sorter

    def _attempt_sequence(self, task: dict, reasoner: SpatialReasoner) -> GameAttempt:
        """One pattern-line episode through the competence substrate."""
        from worlds.sequence import PatternLine
        from worlds.sequence_agent import PatternAgent

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(seed if seed is not None else random.randrange(2**30))
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
        return GameAttempt(
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
                else (f"{res.state.lower()}: {res.placed}/{len(line.cells)} cells filled")
            ),
            family="sequence",
            details={
                "cells": [c for c in line.cells if c is not None],
            },
        )

    def _attempt_assembly(self, task: dict, reasoner: SpatialReasoner) -> GameAttempt:
        """One assembly episode through the competence substrate."""
        from worlds.assembly import PiecePuzzle
        from worlds.assembly_agent import AssemblyAgent

        name = str(task["name"])
        gen = task["generate"]
        seed = gen.get("seed")
        rng = random.Random(seed if seed is not None else random.randrange(2**30))
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
        return GameAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(f"placed {res.placements}, lifted {res.removals} in {res.steps} steps"),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                "" if solved else (f"{res.state.lower()}: {res.placements}/{rows * cols} placed")
            ),
            family="assembly",
        )

    def _attempt_relations(self, task: dict, reasoner: SpatialReasoner) -> GameAttempt:
        """One arrangement episode through the competence substrate."""
        from worlds.relations import Arrangement, Goal, Thing
        from worlds.relations_agent import RelationsAgent

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(seed if seed is not None else random.randrange(2**30))
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
                attrs = tuple((str(k), int(v)) for k, v in (o.get("attrs") or {}).items())
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
        score = (
            1.0 if solved else max(0.0, 1.0 - world.mismatches() / (n_goals + len(world.things)))
        )
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return GameAttempt(
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
                else (f"{res.state.lower()}: {world.mismatches()} goals/things unresolved")
            ),
            family="relations",
            details={
                "order": [world.board[p].name for p in sorted(world.board)],
            },
        )

    def _attempt_quantities(self, task: dict, reasoner: SpatialReasoner) -> GameAttempt:
        """One make-N episode through the competence substrate."""
        from worlds.quantities import Basket, Group
        from worlds.quantities_agent import QuantitiesAgent

        name = str(task["name"])
        gen = task.get("generate")
        seed = (gen or task).get("seed")
        rng = random.Random(seed if seed is not None else random.randrange(2**30))
        if gen:
            world = Basket.generate(int(gen["groups"]), int(gen["target"]), rng)
        else:
            world = Basket(
                int(task["target"]),
                [Group(i, str(g["name"]), int(g["count"])) for i, g in enumerate(task["groups"])],
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
        return GameAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(f"added {res.adds}, lifted {res.lifts} in {res.steps} steps"),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                ""
                if solved
                else (f"{res.state.lower()}: {world.basket_sum()}/{world.target} in basket")
            ),
            family="quantities",
            details={
                "counts": sorted(g.count for g in world.basket.values()),
                "target": world.target,
            },
        )

    def _attempt_classification(
        self,
        task: dict,
        reasoner: SpatialReasoner,
        inducer: DorsolateralPFC | None,
    ) -> GameAttempt:
        """One rule-membership episode through the competence substrate."""
        from worlds.classification import Item, RuleGame
        from worlds.rule_agent import RuleAgent

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
            inducer=inducer,
        )
        res = agent.solve(world, max_steps=int(task.get("max_steps", 300)))
        solved = res.solved
        score = 1.0 if solved else len(world.labels()) / max(1, len(world.items))
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return GameAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(f"{res.guesses} guesses, {res.wrong} wrong in {res.steps} steps"),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure=(
                "" if solved else (f"{res.state.lower()}: {world.mismatches()} items unlabeled")
            ),
            family="classification",
            # The answer itself — which items belong — so a caller
            # that asked the question can report the result.
            details={
                "labels": world.labels(),
                "kind": str(task.get("kind", "")),
            },
        )

    def _attempt_system(
        self, task: dict, on_step: Any = None, should_stop: Any = None
    ) -> GameAttempt | None:
        """One episode of an installed desktop game.

        The game is real software — she launches it, watches its
        window, presses keys, and the program itself judges her. The
        general player knows nothing of the rules; it learns which
        moves keep the game alive through value learning.

        The command must resolve to an executable inside the system
        game directories — an offered spec can't name arbitrary
        programs.
        """
        name = str(task["name"])
        raw_cmd = str(task.get("command", ""))
        exe = Path(raw_cmd).resolve()
        allowed = any(
            str(exe).startswith(str(Path(d).resolve()) + os.sep) for d in _SYSTEM_GAME_DIRS
        )
        if exe is None or not allowed or not exe.is_file() or not os.access(exe, os.X_OK):
            return None
        env = self._system_env(task, name, exe)
        max_steps = int(task.get("max_steps", 300))
        res = None
        try:
            res = self._player.play(
                env,
                max_steps=max_steps,
                time_budget=float(task.get("time_budget", 120.0)),
                on_step=on_step,
                should_stop=should_stop,
            )
        except OSError as e:
            # No desktop, no window — she can't play it right now.
            self.attempts[name] = self.attempts.get(name, 0) + 1
            self._save()
            return GameAttempt(
                task=name,
                hint=str(task.get("hint", "")),
                score=0.0,
                best=self.mastery.get(name, 0.0),
                solved=False,
                mastered=False,
                rule="none",
                nodes=0,
                total_attempts=self.attempts[name],
                failure=str(e) or "couldn't launch",
                family="system",
                details={"kind": "system"},
            )
        finally:
            # A user interrupt ends the session *and* the window —
            # she puts the game down when someone's here rather than
            # leaving it running on their desktop. Otherwise the
            # window stays up as usual for the next session to
            # reattach to.
            env.close(kill=res is not None and res.cause == "interrupted")
        self._system_last_end = time.monotonic()

        # Verdict: the game's own progress measure — how much of the
        # objective meter she moved (board cleared), plus staying
        # alive through the episode.
        progress = min(1.0, max(0.0, res.score))
        survived = min(1.0, res.steps / max(1, max_steps))
        score = max(progress, survived * 0.4)
        solved = res.won or progress >= 0.95
        self.attempts[name] = self.attempts.get(name, 0) + 1
        self.mastery[name] = max(self.mastery.get(name, 0.0), score)
        self._save()
        return GameAttempt(
            task=name,
            hint=str(task.get("hint", "")),
            score=score,
            best=self.mastery[name],
            solved=solved,
            mastered=solved,
            rule=(f"{res.learned} values learned, {self._player.skills(name)} situations known"),
            nodes=res.steps,
            total_attempts=self.attempts[name],
            failure="" if solved else res.cause,
            family="system",
            details={
                "kind": "system",
                "reward": res.reward,
                "deaths": env.deaths,
                "instructions": str(task.get("instructions", "")),
                "goal": str(task.get("goal", "")),
            },
        )

    def _system_env(self, task: dict, name: str, exe: Path) -> Any:
        """Get or create the windowed environment for this system game."""
        from worlds.windowed import X11Game

        env = self._system_envs.get(name)
        if env is None or env.command != [str(exe)]:
            env = X11Game(
                [str(exe)],
                title_pattern=str(task.get("title_pattern") or name),
                actions=[str(a) for a in task.get("actions") or _SYSTEM_ACTIONS],
                cell_px=int(task.get("cell_px", 12)),
                start_keys=[str(k) for k in task.get("start_keys") or []],
                observe=task.get("observe"),
            )
            self._system_envs[name] = env
        return env

    def status(self) -> dict[str, float]:
        """Mastery per game — how good she's gotten at each."""
        return {t["name"]: self.mastery.get(t["name"], 0.0) for t in self.collection()}

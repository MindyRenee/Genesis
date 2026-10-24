#!/usr/bin/env python3
"""ARC-AGI-1 practice-set installer — stage and drip-feed grid tasks.

Converts raw ARC-AGI-1 task files (``{"train": [{"input", "output"}],
"test": [...]}``) into the ``family: "grid"`` offered-game specs her
play loop reads from ``<data_dir>/offered_games/``. The drop-box glob
is non-recursive, so the full converted pool lives in
``offered_games/pool/`` — staged, not on her shelf — and ``--topup``
copies enough of them up to keep a bounded working set live.

Why a pool instead of all 400 at once: every never-played game carries
novelty appeal in her picker (``_appeal`` in
``python/basal_ganglia/play.py`` — 2.1 vs 0.4 for a solved game).
Offering hundreds at once turns "practice" into the only thing she
does. A working set of ~25 keeps ARC a strong pull without starving
her other urges; ``--topup`` restocks as tasks get mastered or
shelved (8 failed attempts, ``_SHELVE_AFTER``).

    # one-time setup: stage the pool, put a working set live
    python3 scripts/arc_offer.py --src ~/testing/ARC/data/training --stage
    python3 scripts/arc_offer.py --topup 25

    # later, after she's played some out
    python3 scripts/arc_offer.py --topup 25

Use the *training* split for practice. The evaluation split stays out
of her reach — a task she has never seen is the only honest test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "python"))

HINT = "Find the rule the example grids share and apply it to the new grid."

# Mirrors _SHELVE_AFTER in python/basal_ganglia/play.py — an
# unsolved task with this many attempts has been shelved; top-up
# leaves it in the pool rather than re-dealing a dead game.
_SHELVE_AFTER = 8


def _default_data_dir() -> Path:
    """Resolve the data dir the way run.sh does.

    Precedence: GENESIS_DATA_DIR, then the checkout's .genesis-data-dir
    pin, then XDG_DATA_HOME, then ~/.local/share. The pin outranks
    XDG_DATA_HOME because the latter is ambient launcher environment,
    not Genesis config — a sandboxed host sets it, which would
    otherwise point --stage/--topup at a sibling instance's
    offered_games/.
    """
    # Blank counts as unset, matching run.sh and config.py.
    explicit = (os.environ.get("GENESIS_DATA_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    pin = Path(__file__).resolve().parent.parent / ".genesis-data-dir"
    if pin.is_file():
        pinned = pin.read_text().splitlines()[0].strip()
        if pinned:
            return Path(pinned)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "genesis"


def _pairs(seq: object) -> list[list[list[list[int]]]]:
    """``[{"input": g, "output": g}]`` → ``[[in, out], ...]``."""
    out = []
    for item in seq or []:
        if not isinstance(item, dict):
            return []
        inp, outp = item.get("input"), item.get("output")
        if not inp or not outp:
            return []
        out.append([inp, outp])
    return out


def _convert(src_file: Path) -> dict | None:
    """One raw ARC task → one offered-game spec, or None if malformed."""
    try:
        raw = json.loads(src_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    train, test = _pairs(raw.get("train")), _pairs(raw.get("test"))
    if not train or not test:
        return None
    return {
        "family": "grid",
        "name": f"arc_{src_file.stem}",
        "hint": HINT,
        "train": train,
        "test": test,
    }


def _write_atomic(path: Path, task: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(task, indent=2) + "\n")
    tmp.replace(path)


def _stage(src: Path, pool: Path) -> tuple[int, int]:
    """Convert every ``*.json`` under ``src`` into ``pool``. → (written, skipped)"""
    pool.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for f in sorted(src.glob("*.json")):
        task = _convert(f)
        if task is None:
            print(f"  skip malformed: {f.name}", file=sys.stderr)
            skipped += 1
            continue
        dest = pool / f"arc_{f.stem}.json"
        if dest.exists():
            skipped += 1
            continue
        _write_atomic(dest, task)
        written += 1
    return written, skipped


def _played_out(data_dir: Path) -> dict[str, tuple[int, float]]:
    """``{game_name: (attempts, mastery)}`` from her progress file."""
    path = data_dir / "gameplay.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    attempts, mastery = data.get("attempts", {}), data.get("mastery", {})
    return {
        name: (int(attempts.get(name, 0)), float(mastery.get(name, 0.0)))
        for name in set(attempts) | set(mastery)
    }


def _topup(offered: Path, want: int) -> tuple[list[str], int]:
    """Copy pool specs into the live drop-box until ``want`` arc_* games
    are live. Skips mastered and shelved tasks — a task she solved or
    gave up on is not practice, it's history. → (added, live_now)"""
    pool = offered / "pool"
    live_now = len(list(offered.glob("arc_*.json")))
    if live_now >= want:
        return [], live_now
    played = _played_out(offered.parent)
    added: list[str] = []
    for spec in sorted(pool.glob("arc_*.json")):
        if live_now >= want:
            break
        name = spec.stem
        if (offered / spec.name).exists():
            continue
        attempts, mastery = played.get(name, (0, 0.0))
        if mastery >= 1.0 or attempts >= _SHELVE_AFTER:
            continue
        # Copy, don't move — the pool is the canonical store; a file
        # she masters can be pulled back out of the live dir without
        # losing the spec.
        _write_atomic(offered / spec.name, json.loads(spec.read_text()))
        added.append(name)
        live_now += 1
    return added, live_now


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--src",
        type=Path,
        default=Path.home() / "testing" / "ARC" / "data" / "training",
        help="directory of raw ARC-AGI-1 task JSONs (use the training split)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="her data dir (default: GENESIS_DATA_DIR or XDG)",
    )
    ap.add_argument(
        "--stage",
        action="store_true",
        help="convert all of --src into offered_games/pool/ (staged, not live)",
    )
    ap.add_argument(
        "--topup",
        type=int,
        metavar="N",
        default=0,
        help="copy pool specs up until N arc_* tasks are live",
    )
    args = ap.parse_args()

    offered = args.data_dir / "offered_games"
    pool = offered / "pool"
    rc = 0
    if args.stage:
        if not args.src.is_dir():
            print(f"error: --src {args.src} is not a directory", file=sys.stderr)
            return 2
        written, skipped = _stage(args.src, pool)
        print(f"staged {written} spec(s) into {pool} ({skipped} skipped)")
    if args.topup:
        if not pool.is_dir():
            print(f"error: no pool at {pool} — run --stage first", file=sys.stderr)
            return 2
        offered.mkdir(parents=True, exist_ok=True)
        added, live_now = _topup(offered, args.topup)
        for name in added:
            print(f"  offered {name}")
        print(f"{len(added)} added · {live_now} arc_* task(s) live in {offered}")
    if not args.stage and not args.topup:
        ap.print_help()
        rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

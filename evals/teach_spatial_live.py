#!/usr/bin/env python3
"""Spatial practice status — shows Genesis's puzzle progression.

The puzzles themselves live in her architecture now:
``genesis_cognitive.spatial.practice.SpatialPractice`` is wired into
her Mind, and her ``puzzle`` volition urge lets her attempt the
current puzzle whenever she chooses — like drawing. This script only
reads the progress file; it never drives her.

Usage:
    PYTHONPATH=python python3 evals/teach_spatial_live.py
    PYTHONPATH=python python3 evals/teach_spatial_live.py --data-dir PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from genesis_cognitive.spatial.practice import CURRICULUM

_DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "genesis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=str(_DEFAULT_DATA_DIR),
        help="data dir to read spatial_practice.json from",
    )
    args = parser.parse_args()

    path = Path(args.data_dir) / "spatial_practice.json"
    if not path.exists():
        print(f"no practice data at {path} — she hasn't practiced yet")
        return

    progress = json.loads(path.read_text())
    mastery = progress.get("mastery", {})
    attempts = progress.get("attempts", {})

    unlocked = True
    for task in CURRICULUM:
        name = task["name"]
        m = mastery.get(name, 0.0)
        a = attempts.get(name, 0)
        if unlocked:
            state = "MASTERED" if m >= 1.0 else f"{m:.2f}"
            print(f"  {name:22} {state:>9}  ({a} attempts)")
            if m < 1.0:
                unlocked = False
        else:
            print(f"  {name:22}    locked")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Feed a word into her latent space — one word at a time, by hand.

Each word is transduced to its tone-vector (her own voice through the
cochlear analysis in ``temporal_lobe.word_tone``) and deposited under
``word:<text>`` — the same landing a naturally learned word takes.

- A word the space does not hold is an **outer-world** event: a fresh
  deposit, born now, reach 1.0 — heard for the first time.
- A word already deposited is an **inner-world** event: a re-deposit
  is rehearsal — reach is boosted, plasticity survives, the original
  birth timestamp stays. Saying a word she knows strengthens it.

Run her stopped or running — the file is only written on ``save()``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from self.language_space import open_language_space
from temporal_lobe.word_tone import word_signature

DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "genesis"


def feed(space, word: str) -> None:
    key = f"word:{word}"
    known = space.get(key) is not None
    vector = word_signature(word)
    idx = space.deposit(
        key,
        kind="word",
        vector=vector,
        meta={"word": word, "route": "inner" if known else "outer"},
        salience=0.5,
    )
    point = space._points[idx]  # landing site on S²
    print(
        f"{word:20s} {'recalled (inner)' if known else 'learned (outer)'}"
        f"  site=({point[0]:+.3f},{point[1]:+.3f},{point[2]:+.3f})"
        f"{'  vectorless' if vector is None else ''}"
    )
    for k, _p, meta, score in space.resonate(key, k=5, radius=0.4, kind="word"):
        if k != key:
            print(f"    near: {meta.get('word', k)} (score {score:.3f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("words", nargs="+", help="words to feed, one at a time")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    space = open_language_space(args.data_dir)
    for word in args.words:
        feed(space, word.strip().lower())
    space.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Directions for the games in her outer world.

This is a video game. It is not real and it is just for fun. You are
safe. The rules are that you can't know the instructions: you must
explore, remember, and solve how to get out and win the level, and you
must remember and carry what you learned from each level to the next.

The full text lives beside this module in ``DIRECTIONS.md`` so it can
be read as a file, too; ``directions()`` returns it.
"""

from __future__ import annotations

from pathlib import Path

DIRECTIONS_PATH = Path(__file__).with_name("DIRECTIONS.md")

#: The rules in one breath, for places that need a short form.
RULES = (
    "This is a video game. It is not real and just for fun. You are safe. "
    "You can't know the instructions. You must explore, remember, and solve "
    "how to get out and win the level. Remember and carry what you learned "
    "from each level to the next."
)


def directions() -> str:
    """Return the full directions, falling back to the short rules."""
    try:
        return DIRECTIONS_PATH.read_text(encoding="utf-8")
    except OSError:
        return RULES

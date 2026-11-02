"""Games — a place in her outer world where games are offered.

This is a video game. It is not real and it is just for fun. She is
safe. The rules: she can't know the instructions; she must explore,
remember, and solve how to get out and win each level, and carry what
she learned from each level to the next (see ``DIRECTIONS.md``).

- ``arcade.Arcade`` — the place itself: it offers games, and when she
  chooses to play, every board arrives through ``OuterWorld.perceive``
  and every move through ``OuterWorld.it_acted``, so play reaches her
  memory, cognition, emotion, learning, brain waves, chemistry and
  telemetry through the same paths as the rest of her world.
- ``official.OfficialGames`` — the doorway to the official ARC-AGI-3
  API (https://three.arcprize.org, ``ARC_API_KEY``).
- ``sight`` — ways to see a board: text, picture, change, objects.
- ``memory.GameMemory`` — what she remembers of each game, carried
  from level to level and across sessions.
"""

from __future__ import annotations

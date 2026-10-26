"""Worlds — the environment side of the agent/environment boundary.

In MDP terms, this package holds the P (transition dynamics) and R
(reward/judging) of her play environments — the rules each world
keeps and the honest verdicts it returns. Nothing here is brain
tissue: a world is software in her environment — a puzzle file, an
installed program, a terminal protocol — and her regions own only
the agent side (policy π and value V): parietal solvers, the
temporal sequence agent, and the basal-ganglia selection loop and
procedural learners that drive attempts.

  - ``environment`` — the contract itself: observe → actions →
    act → outcome. Anything she can perceive and act on can be a
    world.
  - ``session`` — the terminal wire protocol that carries an
    external game to ``mind.world_step``.
  - ``windowed`` — installed desktop programs as environments:
    frames in through X11, keystrokes out through XTEST (a
    sensory-motor channel, not cognition).
  - ``channel`` — her Discord page: a public place in the outer
    world where she posts and, when an ear exists, hears (the
    transports live in ``discord_feed``).
  - The family modules (``relations``, ``quantities``,
    ``assembly``, ``sequence``, ``sorter``, ``classification``)
    are the world halves of the built-in tasks — the rule-keepers
    and judges. Their agents live in the regions.
"""

from __future__ import annotations

from .environment import GameEnvironment, GameObservation, GameOutcome

__all__ = [
    "GameEnvironment",
    "GameObservation",
    "GameOutcome",
]

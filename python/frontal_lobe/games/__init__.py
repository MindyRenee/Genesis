"""Games — Genesis's collection of things to play.

Not a curriculum and not a benchmark: games are software in her
world — the built-in worlds below, spec files anyone hands her,
and the programs installed on her computer — and what her brain
owns is the *ability to play them*, not the games themselves.

The region split mirrors how a brain plays:

  - ``play.py`` — ``GamePlay``: her collection, her wants (appeal),
    her play record. Prefrontal — choosing what to play and tracking
    how well she plays it is executive.
  - ``environment.py`` — the GameEnvironment contract: observe →
    actions → act → outcome. Anything she can perceive and act on
    can be a game.
  - ``player.py`` — ``GamePlayer``: the ONE player. A general
    value-learning agent over any environment — no per-game rules
    hardcoded, just perceive/choose/learn.
  - ``windowed.py`` — ``X11Game``: installed desktop programs as
    environments (frames → Grid, keys via XTEST).
  - Each family package holds a built-in game pair: a *world* (the
    rules and honest judge) and an *agent* (a decision-maker using
    the shared competence substrate).
  - Spatial board substrate (Grid, Scene, transforms) stays in
    ``parietal_lobe`` — the games borrow it; they don't own it.
  - The play urge lives in ``basal_ganglia`` (volition), the felt
    outcome of a game in ``limbic_system``, and the procedural skill
    it builds in ``reasoning.TaskCompetence``.
"""

from __future__ import annotations

from .environment import (
    GameEnvironment,
    GameObservation,
    GameOutcome,
)
from .play import COLLECTION, GameAttempt, GamePlay, normalize_offered
from .player import GamePlayer, PlayResult

__all__ = [
    "COLLECTION",
    "GameAttempt",
    "GameEnvironment",
    "GameObservation",
    "GameOutcome",
    "GamePlay",
    "GamePlayer",
    "PlayResult",
    "normalize_offered",
]

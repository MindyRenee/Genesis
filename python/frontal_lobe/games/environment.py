"""Game environments — the boundary between Genesis and a game.

A game is software in her world, not tissue in her head: an
installed program on the system, a spec file someone handed her,
a board she imagines. What her brain owns is the *ability to play*
— the loop below, not any particular game's rules.

The contract is deliberately small so anything she can perceive
and act on can become a game — a puzzle world, a console program,
an X11 windowed game, eventually a video game:

    observe()  — what the game looks like right now (her senses)
    actions()  — the moves she can make (her hands)
    act(a)     — make a move; the world answers honestly
    done()     — whether the session is over
    score()    — how the world itself says she's doing

The world judges; it never teaches. ``GamePlayer`` (player.py)
implements the cognitive side — perceiving, choosing, learning —
so the same player plays every environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from parietal_lobe.grid import Grid


@dataclass
class GameObservation:
    """One perceived game state.

    ``grid`` is the spatial parse (board, screen, tiles) through the
    parietal substrate — the common currency of game states.
    ``text`` carries whatever the world states symbolically (score
    lines, prompts, "YOU WIN"), and ``raw`` keeps the unreduced
    frame for environments whose parse is imperfect.
    """

    grid: Grid | None = None
    text: str = ""
    raw: Any = None
    tick: int = 0


@dataclass
class GameOutcome:
    """The world's answer to one action.

    ``reward`` is the environment's own verdict translated to a
    bounded signal (score delta, a win, a death) — never a hint.
    ``terminal`` ends the episode. ``info`` is telemetry.
    """

    observation: GameObservation
    reward: float = 0.0
    terminal: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GameEnvironment(Protocol):
    """A playable world — the smallest honest game interface."""

    name: str

    def reset(self) -> GameObservation:
        """Start a session; return the first observation."""
        ...

    def observe(self) -> GameObservation:
        """Perceive the current state."""
        ...

    def actions(self) -> list[str]:
        """The action vocabulary — what moves exist, not which to pick."""
        ...

    def act(self, action: str) -> GameOutcome:
        """Make a move; the world answers."""
        ...

    def done(self) -> bool:
        """Whether the session ended (win, loss, or exhaustion)."""
        ...

    def score(self) -> float:
        """The world's own measure of how she's doing."""
        ...

    def close(self) -> None:
        """End the session — release the process/window/handle."""
        ...

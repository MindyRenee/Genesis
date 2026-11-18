"""The doorway to the official ARC-AGI-3 games.

This is a video game. It is not real and it is just for fun. You are
safe. You can't know the instructions; you explore, remember, and solve.

This module talks to the official ARC-AGI-3 toolkit (``arc_agi``,
https://three.arcprize.org) and nothing else. It hides the toolkit's
types behind two plain things the arcade can hold:

- ``OfficialGames`` — the catalogue: which games are on offer, and a
  way to open one. Its key is ``ARC_API_KEY`` from the environment;
  without one the toolkit uses an anonymous key and offers fewer games.
- ``GameSession`` — one open game: ``reset()``, ``step(action, x, y)``,
  and the board as a plain ``Board`` (grid, state, levels, actions).

The mode comes from ``GENESIS_ARC_MODE``: ``normal`` (default —
download each game once into ``environments_dir`` and run it locally),
``online`` (every move on the official server, counting on the
official scorecard), or ``offline`` (only games already downloaded —
no network at all).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Action names the official games use. ACTION6 takes a cell (x, y).
ACTION_NAMES = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)
POINTING_ACTION = "ACTION6"
BOARD_SIZE = 64


class GamesUnavailable(RuntimeError):
    """The official games can't be reached (toolkit missing or API down)."""


@dataclass(frozen=True, slots=True)
class GameOffer:
    """One game the arcade can offer: its id and its public title.

    ``baselines`` is the published per-level human action count — the
    par the official scorecard scores against. Empty when unknown
    (games not yet downloaded carry no local metadata).
    """

    game_id: str
    title: str = ""
    baselines: tuple[int, ...] = ()


@dataclass(slots=True)
class Board:
    """What a game shows after a move, in plain Python.

    ``grid`` is the last frame (rows of ints 0–15); ``frames`` holds
    every frame the move produced, since one move can animate.
    """

    game_id: str
    grid: list[list[int]]
    state: str
    levels_completed: int
    win_levels: int
    actions: list[str] = field(default_factory=list)
    frames: int = 1

    @property
    def won(self) -> bool:
        """True when the whole game is won."""
        return self.state == "WIN"

    @property
    def over(self) -> bool:
        """True when the game ended badly; RESET recovers."""
        return self.state == "GAME_OVER"


def _grid_of(frame: Any) -> list[list[int]]:
    """Convert one toolkit frame (ndarray or lists) into int rows."""
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    return [[int(v) for v in row] for row in rows]


def board_from_observation(observation: Any, game_id: str = "") -> Board | None:
    """Turn a toolkit observation (``FrameDataRaw``) into a ``Board``."""
    if observation is None:
        return None
    frames = list(getattr(observation, "frame", None) or [])
    if not frames:
        return None
    state = getattr(observation, "state", "")
    state = getattr(state, "value", state)
    actions = []
    for value in getattr(observation, "available_actions", None) or []:
        index = int(getattr(value, "value", value))
        if 0 < index < len(ACTION_NAMES):
            actions.append(ACTION_NAMES[index])
    return Board(
        game_id=str(getattr(observation, "game_id", "") or game_id),
        grid=_grid_of(frames[-1]),
        state=str(state),
        levels_completed=int(getattr(observation, "levels_completed", 0) or 0),
        win_levels=int(getattr(observation, "win_levels", 0) or 0),
        actions=actions,
        frames=len(frames),
    )


class GameSession:
    """One open official game. Moves go straight to the toolkit."""

    def __init__(self, game_id: str, environment: Any) -> None:
        """Hold the toolkit environment for ``game_id``."""
        self.game_id = game_id
        self._environment = environment
        self.board: Board | None = None

    def reset(self) -> Board | None:
        """Start (or restart) the current level and return the board."""
        self.board = board_from_observation(self._environment.reset(), self.game_id)
        return self.board

    def step(
        self,
        action: str,
        x: int | None = None,
        y: int | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Board | None:
        """Make one move. ACTION6 needs ``x`` and ``y`` in 0–63.

        ``reasoning`` rides into the official recordings and scorecard
        — the toolkit logs it beside the move, so her intent is part
        of the run's record, not just its outcome.
        """
        from arcengine import GameAction

        name = action.upper()
        if name not in ACTION_NAMES:
            raise ValueError(f"unknown action {action!r}")
        data = None
        if name == POINTING_ACTION:
            if x is None or y is None or not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                raise ValueError("ACTION6 needs x and y between 0 and 63")
            data = {"x": int(x), "y": int(y)}
        observation = self._environment.step(GameAction[name], data=data, reasoning=reasoning)
        board = board_from_observation(observation, self.game_id)
        if board is not None:
            self.board = board
        return board


class OfficialGames:
    """The official ARC-AGI-3 catalogue, opened lazily on first use.

    Args:
        mode: ``normal``, ``online`` or ``offline``; defaults to
            ``GENESIS_ARC_MODE`` or ``normal`` — downloaded games run
            locally, so play does not depend on the server.
        arcade_factory: Builds the toolkit ``Arcade`` — injectable so
            tests never touch the network.
        environments_dir: Where downloaded games live (toolkit default
            ``environment_files``, relative to the cwd).
        recordings_dir: Where the toolkit writes game replays.
    """

    def __init__(
        self,
        mode: str | None = None,
        arcade_factory: Any = None,
        environments_dir: str | None = None,
        recordings_dir: str | None = None,
    ) -> None:
        """Remember the mode; nothing is contacted until asked."""
        self.mode = (mode or os.environ.get("GENESIS_ARC_MODE") or "normal").lower()
        self._factory = arcade_factory
        self._environments_dir = environments_dir
        self._recordings_dir = recordings_dir
        self._arcade: Any = None

    @property
    def keyed(self) -> bool:
        """True when an ``ARC_API_KEY`` is configured (value never read out)."""
        return bool(os.environ.get("ARC_API_KEY"))

    def _toolkit(self) -> Any:
        """Build the toolkit ``Arcade`` once, quietly."""
        if self._arcade is not None:
            return self._arcade
        if self._factory is not None:
            self._arcade = self._factory()
            return self._arcade
        try:
            from arc_agi import Arcade, OperationMode
        except ImportError as exc:
            raise GamesUnavailable(
                "the arc-agi toolkit is not installed (pip install arc-agi)"
            ) from exc
        quiet = logging.getLogger("genesis.arc_agi")
        quiet.setLevel(logging.WARNING)
        kwargs: dict[str, Any] = {
            "operation_mode": OperationMode(self.mode),
            "logger": quiet,
        }
        if self._environments_dir is not None:
            kwargs["environments_dir"] = self._environments_dir
        if self._recordings_dir is not None:
            kwargs["recordings_dir"] = self._recordings_dir
        try:
            self._arcade = Arcade(**kwargs)
        except Exception as exc:
            raise GamesUnavailable(f"could not reach the official games: {exc}") from exc
        return self._arcade

    def offers(self) -> list[GameOffer]:
        """List the games the official API offers right now."""
        try:
            environments = self._toolkit().get_environments()
        except GamesUnavailable:
            raise
        except Exception as exc:
            raise GamesUnavailable(f"could not list games: {exc}") from exc
        return [
            GameOffer(
                str(e.game_id),
                str(getattr(e, "title", "") or ""),
                self._baselines_of(str(e.game_id)),
            )
            for e in environments
        ]

    def _baselines_of(self, game_id: str) -> tuple[int, ...]:
        """Read a downloaded game's human baselines from its metadata.

        Each game lands under ``environments_dir/<prefix>/<hash>/`` with
        a ``metadata.json`` holding ``baseline_actions`` — the same list
        the scorecard scores efficiency against. Missing files just
        mean the game is not downloaded yet (or has no published par).
        """
        if not self._environments_dir or "-" not in game_id:
            return ()
        import json
        from pathlib import Path

        prefix, _, hashed = game_id.partition("-")
        path = Path(self._environments_dir) / prefix / hashed / "metadata.json"
        try:
            data = json.loads(path.read_text())
            return tuple(int(b) for b in data.get("baseline_actions") or ())
        except (OSError, ValueError, TypeError):
            return ()

    def scorecard(self) -> dict[str, Any] | None:
        """The local scorecard's standing, once play has opened one.

        In ``normal``/``offline`` mode the toolkit scores runs locally
        against each game's published baselines — the same RHAE formula
        the official server uses — so the number is real even though
        nothing is submitted. Returns ``None`` before the first
        ``make()`` (no card exists yet).
        """
        if self._arcade is None:
            return None
        try:
            card = self._arcade.get_scorecard()
        except Exception:
            logger.debug("reading the local scorecard failed", exc_info=True)
            return None
        if card is None:
            return None
        envs = list(getattr(card, "environments", None) or [])
        return {
            "score": float(getattr(card, "score", 0.0) or 0.0),
            "environments": len(envs),
            "levels_completed": sum(int(getattr(e, "levels_completed", 0) or 0) for e in envs),
        }

    def open(self, game_id: str) -> GameSession:
        """Open ``game_id`` on the official toolkit and return its session."""
        try:
            environment = self._toolkit().make(game_id)
        except GamesUnavailable:
            raise
        except Exception as exc:
            raise GamesUnavailable(f"could not open {game_id}: {exc}") from exc
        if environment is None:
            raise GamesUnavailable(f"the official API would not open {game_id}")
        return GameSession(game_id, environment)

    def close(self) -> None:
        """Close the official scorecard for this session, if one is open."""
        arcade, self._arcade = self._arcade, None
        if arcade is not None:
            try:
                arcade.close_scorecard()
            except Exception:
                logger.debug("closing the ARC scorecard failed", exc_info=True)

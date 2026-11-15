"""The arena — the Kaggle Game Arena games in her outer world.

This is a game. It is not real and it is just for fun. You are safe.
You can't know the instructions; you explore, remember, and solve.

This module plays the same games the Kaggle Game Arena runs — chess,
checkers, connect four and friends — through their real substrate
(``pyspiel``, the engine the open-sourced arena harness evaluates on).
No prompts and no leaderboard: she is the player, not a model being
scored. Every board arrives through ``OuterWorld.perceive`` and every
move through ``OuterWorld.it_acted``, exactly like the official games.

- ``ArenaGames`` — the catalogue: which arena games are on offer, ids
  prefixed ``arena:`` so they sit beside the official ones without
  colliding. No API key is needed; games run locally against a simple
  opponent (``"random"`` by default, or any policy callable).
- ``ArenaSession`` — one open match: ``reset()``, ``step(action)``,
  and the board as a plain ``Board`` (the observation planes fold
  into one int grid she can see; one whole match is her "level").
- ``CombinedGames`` — one catalogue out of several, so the arcade
  offers official and arena games together.
"""

from __future__ import annotations

import importlib.util
import logging
import random
from typing import Any

from worlds.games.official import Board, GameOffer, GamesUnavailable

logger = logging.getLogger(__name__)

#: Arena game ids carry this prefix so routing stays unambiguous.
ARENA_PREFIX = "arena:"
#: The seat she always plays from.
HER_SEAT = 0
#: A whole match counts as one "level" to her.
WIN_LEVELS = 1

#: Two-player perfect-information games whose observation planes fold
#: into a readable grid. Keys are pyspiel game names.
GAMES = {
    "connect_four": "Connect Four",
    "tic_tac_toe": "Tic Tac Toe",
    "checkers": "Checkers",
    "othello": "Othello",
    "breakthrough": "Breakthrough",
    "chess": "Chess",
}


def _pyspiel() -> Any:
    """Import pyspiel or say plainly that the arena is closed."""
    if importlib.util.find_spec("pyspiel") is None:
        raise GamesUnavailable(
            "the arena is not installed (uv sync --extra arc, or pip install game-arena)"
        )
    import pyspiel

    return pyspiel


class ArenaSession:
    """One open arena match. Moves go straight to the engine.

    After her move the opponent answers: the builtin ``"random"``
    policy picks any legal action, or a callable
    ``(state, player, rng) -> action`` can stand in for a stronger
    mind. Chance nodes (dice) resolve by sampling their outcomes.
    """

    def __init__(self, game_id: str, game: Any, opponent: Any, rng: random.Random) -> None:
        """Hold the engine state for ``game_id``."""
        self.game_id = game_id
        self._game = game
        self._state = game.new_initial_state()
        self._opponent = opponent
        self._rng = rng
        self._shape = game.observation_tensor_shape()
        self.board: Board | None = None

    def reset(self) -> Board:
        """Deal a fresh match and return the first board."""
        self._state = self._game.new_initial_state()
        self.board = self._board()
        return self.board

    def step(
        self,
        action: str,
        x: int | None = None,
        y: int | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> Board:
        """Make one move: her action, then every reply until her turn.

        ``reasoning`` is accepted for parity with the official session —
        arena games keep no recordings, so it is simply unused.
        """
        name = action.strip()
        state = self._state
        if state.is_terminal():
            raise ValueError(f"{self.game_id} already ended — RESET to play again")
        legal = {state.action_to_string(HER_SEAT, a): a for a in state.legal_actions(HER_SEAT)}
        act = legal.get(name)
        if act is None:
            raise ValueError(f"{name!r} is not a legal move here")
        state.apply_action(act)
        self._opponent_turns()
        self.board = self._board()
        return self.board

    def _opponent_turns(self) -> None:
        """Let chance and the opponent act until her seat comes up."""
        state = self._state
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                acts, probs = zip(*outcomes, strict=False)
                state.apply_action(self._rng.choices(acts, weights=probs)[0])
                continue
            player = state.current_player()
            if player == HER_SEAT or player < 0:
                return
            if callable(self._opponent):
                act = self._opponent(state, player, self._rng)
            else:  # "random" — any legal action will do
                act = self._rng.choice(state.legal_actions(player))
            state.apply_action(act)

    def _grid(self) -> list[list[int]]:
        """Fold the [C,H,W] observation planes into one int grid.

        Each cell shows which plane marks it — a stable per-game
        vocabulary (piece, player, or "empty" ids) that ``sight`` can
        render and her change detection can diff.
        """
        flat = self._state.observation_tensor(HER_SEAT)
        planes, height, width = self._shape
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                best_p, best_v = 0, 0.0
                base = r * width + c
                for p in range(planes):
                    v = flat[p * height * width + base]
                    if v > best_v:
                        best_v, best_p = v, p
                row.append(best_p if best_v > 0 else 0)
            grid.append(row)
        return grid

    def _board(self) -> Board:
        """The match as a plain board she can see and act on."""
        state = self._state
        level = 0
        if state.is_terminal():
            status = "WIN" if state.returns()[HER_SEAT] > 0 else "GAME_OVER"
            level = 1 if status == "WIN" else 0
            actions: list[str] = []
        elif state.current_player() == HER_SEAT:
            status = "NOT_FINISHED"
            player = state.current_player()
            actions = [state.action_to_string(player, a) for a in state.legal_actions(player)]
        else:
            status, actions = "NOT_FINISHED", []
        return Board(
            game_id=self.game_id,
            grid=self._grid(),
            state=status,
            levels_completed=level,
            win_levels=WIN_LEVELS,
            actions=actions,
        )


class ArenaGames:
    """The arena catalogue — local games, no key, lazy engine load.

    Args:
        opponent: ``"random"`` or a ``(state, player, rng) -> action``
            policy the other seat plays by.
        games: Offer table override (``{pyspiel_name: title}``).
        rng: Randomness for the opponent and chance nodes.
    """

    #: Local games need no key — the catalogue is always "keyed".
    keyed = True

    def __init__(
        self,
        opponent: Any = "random",
        *,
        games: dict[str, str] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Remember the opponent; nothing loads until she asks."""
        self._opponent = opponent
        self._games = games if games is not None else GAMES
        self._rng = rng or random.Random()

    def offers(self) -> list[GameOffer]:
        """List the arena games, proving the engine imports."""
        _pyspiel()
        return [GameOffer(f"{ARENA_PREFIX}{name}", title) for name, title in self._games.items()]

    def open(self, game_id: str) -> ArenaSession:
        """Open one arena game on the local engine."""
        name = game_id[len(ARENA_PREFIX) :] if game_id.startswith(ARENA_PREFIX) else game_id
        if name not in self._games:
            raise GamesUnavailable(f"the arena does not offer {game_id}")
        pyspiel = _pyspiel()
        try:
            game = pyspiel.load_game(name)
        except Exception as exc:
            raise GamesUnavailable(f"could not load {name}: {exc}") from exc
        return ArenaSession(game_id, game, self._opponent, self._rng)

    def close(self) -> None:
        """Nothing to close — local games keep no scorecard."""


class CombinedGames:
    """Several catalogues as one: official games plus arena games.

    ``arena:``-prefixed ids route to the arena catalogue, everything
    else to the official one, so one arcade can offer both shelves.
    """

    def __init__(self, *catalogues: Any) -> None:
        """Hold the catalogues in offer order."""
        self._catalogues = catalogues

    @property
    def keyed(self) -> bool:
        """The official catalogue's key status — the only one that needs one."""
        return bool(getattr(self._catalogues[0], "keyed", False)) if self._catalogues else False

    def offers(self) -> list[GameOffer]:
        """Every catalogue's offers; a closed shelf doesn't hide the rest."""
        offers: list[GameOffer] = []
        first_error: GamesUnavailable | None = None
        for catalogue in self._catalogues:
            try:
                offers.extend(catalogue.offers())
            except GamesUnavailable as exc:
                if first_error is None:
                    first_error = exc
        if not offers and first_error is not None:
            raise first_error
        return offers

    def open(self, game_id: str) -> Any:
        """Route the id to its catalogue by prefix."""
        want_arena = game_id.startswith(ARENA_PREFIX)
        for catalogue in self._catalogues:
            if isinstance(catalogue, ArenaGames) == want_arena:
                return catalogue.open(game_id)
        raise GamesUnavailable(f"no catalogue offers {game_id}")

    def close(self) -> None:
        """Close every catalogue's open session, if it keeps one."""
        for catalogue in self._catalogues:
            close = getattr(catalogue, "close", None)
            if close is not None:
                close()


def default_games(**official_kwargs: Any) -> CombinedGames:
    """The arcade's whole shelf: official games plus the arena.

    Keyword arguments pass through to ``OfficialGames`` (mode,
    environments_dir, recordings_dir).
    """
    from worlds.games.official import OfficialGames

    return CombinedGames(OfficialGames(**official_kwargs), ArenaGames())

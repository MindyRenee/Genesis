"""Tests for the arena — the Kaggle Game Arena shelf in her outer world."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.framework import ToolRegistry
from worlds.games.arcade import Arcade
from worlds.games.arena import (
    ARENA_PREFIX,
    ArenaGames,
    ArenaSession,
    CombinedGames,
    default_games,
)
from worlds.games.official import GameOffer, GamesUnavailable
from worlds.world import OuterWorld

pyspiel = pytest.importorskip("pyspiel")


class _FakeOfficial:
    """A stand-in official catalogue: one fake game, records calls."""

    keyed = False

    def __init__(self) -> None:
        self.opened: list[str] = []

    def offers(self):
        return [GameOffer("fake1", "Fake Official Game")]

    def open(self, game_id):
        self.opened.append(game_id)
        raise GamesUnavailable(f"fake official cannot open {game_id}")

    def close(self):
        pass


def _arcade():
    """An arcade with a combined shelf in an impulse-recording world."""
    impulses = []
    world = OuterWorld(neuro_impulse=lambda c, a: impulses.append((c, a)))
    events = []
    world.on_external_event = events.append
    games = CombinedGames(_FakeOfficial(), ArenaGames(rng=random.Random(0)))
    arcade = Arcade(world, games=games, wave_drive=lambda b, s: None, rng=random.Random(0))
    return world, arcade, events


def test_arena_offers_prefixed_games():
    """The arena offers its shelf under the arena: prefix."""
    offers = ArenaGames().offers()
    assert offers and all(o.game_id.startswith(ARENA_PREFIX) for o in offers)
    assert GameOffer(f"{ARENA_PREFIX}chess", "Chess") in offers


def test_open_session_shows_a_board():
    """Opening a game gives a board she can see and act on."""
    session = ArenaGames().open(f"{ARENA_PREFIX}tic_tac_toe")
    board = session.reset()
    assert board.state == "NOT_FINISHED"
    assert len(board.grid) == 3 and len(board.grid[0]) == 3
    assert len(board.actions) == 9


def test_a_move_answers_with_the_opponents():
    """Her step applies her move, then the opponent replies."""
    session = ArenaGames(rng=random.Random(1)).open(f"{ARENA_PREFIX}tic_tac_toe")
    session.reset()
    board = session.step(session.board.actions[0])
    assert board is not None
    marked = sum(v != 0 for row in board.grid for v in row)
    assert marked >= 2  # her mark plus the reply
    assert board.state in ("NOT_FINISHED", "WIN", "GAME_OVER")


def test_illegal_move_is_refused():
    """A string that is not a legal move raises, never corrupts the game."""
    session = ArenaGames().open(f"{ARENA_PREFIX}connect_four")
    session.reset()
    with pytest.raises(ValueError, match="not a legal move"):
        session.step("resign")


def test_a_full_match_reaches_terminal():
    """Two random policies finish a connect four match honestly."""
    session = ArenaGames(rng=random.Random(3)).open(f"{ARENA_PREFIX}connect_four")
    board = session.reset()
    for _ in range(50):
        if board.state != "NOT_FINISHED":
            break
        board = session.step(random.choice(board.actions))
    assert board.state in ("WIN", "GAME_OVER")
    assert board.actions == []
    if board.state == "WIN":
        assert board.levels_completed == 1


def test_combined_catalogue_merges_and_routes():
    """Combined offers cover both shelves; ids route by prefix."""
    official, arena = _FakeOfficial(), ArenaGames()
    games = CombinedGames(official, arena)
    ids = [o.game_id for o in games.offers()]
    assert "fake1" in ids and f"{ARENA_PREFIX}chess" in ids
    assert isinstance(games.open(f"{ARENA_PREFIX}chess"), ArenaSession)
    with pytest.raises(GamesUnavailable):
        games.open("fake1")
    assert official.opened == ["fake1"]


def test_combined_survives_a_closed_shelf():
    """When one catalogue is unavailable the other still offers."""

    class Closed:
        def offers(self):
            raise GamesUnavailable("offline")

        def open(self, game_id):
            raise GamesUnavailable("offline")

    games = CombinedGames(Closed(), ArenaGames())
    ids = [o.game_id for o in games.offers()]
    assert ids and all(i.startswith(ARENA_PREFIX) for i in ids)


def test_arena_games_play_through_the_arcade():
    """She can enter an arena game and play it through the world."""
    _, arcade, events = _arcade()
    board = arcade.enter(f"{ARENA_PREFIX}tic_tac_toe")
    assert board is not None
    report = arcade.move(arcade.moves(board)[0])
    assert report is not None
    assert any("tic_tac_toe" in str(e.percept or e.content) for e in events)
    arcade.leave()


def test_default_games_is_a_combined_shelf():
    """The default catalogue pairs the official API with the arena."""
    games = default_games()
    assert isinstance(games, CombinedGames)
    offers = games.offers()
    assert any(o.game_id.startswith(ARENA_PREFIX) for o in offers)


def test_missing_engine_reports_unavailable(monkeypatch):
    """No pyspiel is an honest 'closed', never a crash."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(GamesUnavailable, match="not installed"):
        ArenaGames().offers()


def test_arcade_tools_reach_arena_games():
    """Her existing arcade_* tools reach the arena shelf too."""
    _, arcade, _ = _arcade()
    registry = ToolRegistry()
    arcade.register_tools(registry)
    out = registry.run("arcade_offers").output
    assert f"{ARENA_PREFIX}chess" in out
    assert registry.run("arcade_enter", game_id=f"{ARENA_PREFIX}tic_tac_toe").success
    assert registry.run("arcade_act", move=arcade.moves(arcade.session.board)[0]).success

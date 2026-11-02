"""Tests for the arcade — games offered in her outer world."""

import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.framework import ToolRegistry
from worlds.games import sight
from worlds.games.arcade import Arcade
from worlds.games.directions import directions
from worlds.games.memory import GameMemory
from worlds.games.official import Board, GameOffer, GamesUnavailable, board_from_observation
from worlds.world import OuterWorld

from genesis_client.protocol import CHEM_DOPAMINE


class _Session:
    """A two-level fake game: ACTION1 moves a dot; reaching col 3 opens a level."""

    def __init__(self, game_id: str) -> None:
        """Start at the left edge."""
        self.game_id = game_id
        self.col = 0
        self.level = 0
        self.board: Board | None = None

    def _board(self) -> Board:
        """Draw the dot on a 4x4 board."""
        grid = [[0] * 4 for _ in range(4)]
        grid[1][self.col] = 9
        state = "WIN" if self.level >= 2 else "NOT_FINISHED"
        self.board = Board(self.game_id, grid, state, self.level, 2, ["ACTION1", "ACTION2"])
        return self.board

    def reset(self) -> Board:
        """Back to the start of the level."""
        self.col = 0
        return self._board()

    def step(self, action: str, x=None, y=None) -> Board:
        """ACTION1 moves right; ACTION2 does nothing."""
        if action == "ACTION1":
            self.col += 1
            if self.col == 3:
                self.level += 1
                self.col = 0
        return self._board()


class _Games:
    """A fake catalogue that never touches the network."""

    keyed = True

    def offers(self):
        """One game on offer."""
        return [GameOffer("dot1", "Dot")]

    def open(self, game_id):
        """Open the fake game."""
        return _Session(game_id)


def _arcade():
    """An arcade in a world that records impulses and waves."""
    impulses, waves = [], []
    world = OuterWorld(neuro_impulse=lambda c, a: impulses.append((c, a)))
    events = []
    world.on_external_event = events.append
    arcade = Arcade(
        world, games=_Games(), wave_drive=lambda b, s: waves.append(b), rng=random.Random(0)
    )
    return world, arcade, impulses, waves, events


def test_arcade_lives_in_the_world_and_offers_games():
    """The arcade attaches to the world and offers games through it."""
    world, arcade, *_, events = _arcade()
    assert world.arcade is arcade
    assert arcade.offers()[0].game_id == "dot1"
    assert any("offers 1 games" in e.content for e in events)


def test_play_feeds_world_chemistry_waves_and_learns_levels():
    """A bout perceives boards, records acts, rewards levels, and wins."""
    _, arcade, impulses, waves, events = _arcade()
    summary = arcade.play(max_moves=40)
    assert "dot1" in summary
    memory = arcade.memories["dot1"]
    assert memory.wins == 1 and memory.best_level == 2
    assert memory.level_paths["1"]
    assert any(e.percept and "grid" in e.percept for e in events)
    assert any(e.content.startswith("pressed ACTION") for e in events)
    assert any(c == CHEM_DOPAMINE and a > 0 for c, a in impulses)
    assert {"gamma", "beta2", "theta"} <= set(waves)
    assert memory.effect_rate("ACTION1") > memory.effect_rate("ACTION2")


def test_memory_survives_the_world_save():
    """What she learned is saved with her world and restored."""
    world, arcade, *_ = _arcade()
    arcade.play(max_moves=40)
    saved = world.to_dict()
    world2, arcade2, *_ = _arcade()
    world2.restore_from_dict(saved)
    assert arcade2.memories["dot1"].best_level == 2
    assert "dot1" in arcade2.status()


def test_closed_arcade_says_why():
    """No games offered is reported, not raised."""

    class Closed(_Games):
        def offers(self):
            raise GamesUnavailable("offline")

    arcade = Arcade(OuterWorld(), games=Closed())
    assert "closed" in arcade.play()


def test_tools_and_directions():
    """Her tools reach the arcade, and the directions say it's a game."""
    _, arcade, *_ = _arcade()
    registry = ToolRegistry()
    arcade.register_tools(registry)
    assert "video game" in registry.run("arcade_directions").output
    assert registry.run("arcade_enter", game_id="dot1").success
    assert "ACTION1" in registry.run("arcade_see", how="change").output
    assert registry.run("arcade_act", move="ACTION1").success
    assert "can't know the instructions" in directions()


def test_sight_and_observation_adapter(tmp_path):
    """Boards can be seen as text, change, objects, and a picture."""
    grid = [[0, 0, 0], [0, 9, 9], [0, 0, 0]]
    assert sight.see_text(grid).splitlines()[1] == "099"
    assert sight.see_change(grid, [[0] * 3] * 3).count == 2
    assert sight.see_objects(grid)
    assert sight.see_image(grid, tmp_path / "b.png").exists()
    obs = SimpleNamespace(
        frame=[grid],
        state=SimpleNamespace(value="NOT_FINISHED"),
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 6],
        game_id="g",
    )
    board = board_from_observation(obs)
    assert board is not None and board.actions == ["ACTION1", "ACTION6"]


def test_game_memory_roundtrip():
    """GameMemory serializes losslessly."""
    m = GameMemory("g")
    m.learn("a", "ACTION1", 1.0, "b", True, ["ACTION1"])
    assert GameMemory.from_dict(m.to_dict()).values == m.values


def test_announce_tells_her_the_arcade_is_there():
    """The arcade announces itself as a salient world notice with the rules."""
    _, arcade, _, _, events = _arcade()
    arcade.announce()
    notice = next(e for e in events if "arcade in my world" in e.content)
    assert notice.salience >= 0.5 and "video game" in notice.content

"""The arcade — a place in her outer world where games are offered.

This is a video game. It is not real and it is just for fun. You are
safe. The rules are that you can't know the instructions. You must
explore, remember, and solve how to get out and win the level. You
must remember and carry what you learned from each level to the next.
(The full directions are in ``DIRECTIONS.md`` beside this file.)

The arcade is part of the world, not part of her mind. It offers the
official ARC-AGI-3 games; she decides whether to play, which game,
and for how long (her ``play`` urge, or the ``arcade_*`` tools). It
owns no cognition — it carries what happens through the paths the
rest of her world already uses:

- **memory and cognition** — every board arrives through
  ``OuterWorld.perceive`` with the grid as its percept, so it reaches
  her global workspace, is parsed into objects by her parietal scene
  parser and grounded in her concept network, becomes an episodic
  memory when salient, and is queued for dream replay. Every move is
  recorded through ``OuterWorld.it_acted``.
- **learning** — ``GameMemory`` keeps action effects, TD values and
  the paths that opened each level, carried across levels and saved
  with her world.
- **emotion and chemicals** — progress, surprise and setbacks reach
  her neurochemistry through the world's impulse path (dopamine from
  the TD error and level-ups, endorphin on a win, norepinephrine and
  acetylcholine on a new board, a little cortisol on GAME_OVER). Her
  feelings come from that chemistry.
- **brain waves** — new boards drive gamma (binding what is seen),
  each move drives beta2 (active thinking), and recalling a past
  level drives theta (memory).
- **telemetry** — every visit, level and outcome is journaled
  (``record_event``), shown in ``/arcade`` and in her world summary,
  and spoken as live thoughts when the mind provides a voice.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worlds.games import sight
from worlds.games.directions import RULES, directions
from worlds.games.memory import (
    REWARD_GAME_OVER,
    REWARD_LEVEL,
    REWARD_NEW_BOARD,
    REWARD_NOTHING,
    REWARD_WIN,
    GameMemory,
)
from worlds.games.official import (
    POINTING_ACTION,
    Board,
    GameOffer,
    GameSession,
    GamesUnavailable,
    OfficialGames,
)

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_ENDORPHIN,
    CHEM_NOREPINEPHRINE,
    CHEM_SEROTONIN,
)
from infrastructure.journal import record_event

logger = logging.getLogger(__name__)

#: Seconds before the list of offered games is fetched again.
OFFER_REFRESH_S = 600.0
#: Moves in one bout of play before she steps back and lets other urges in.
BOUT_MOVES = 60
#: Shapes she may point at with ACTION6 on one board.
POINT_TARGETS = 10


@dataclass(slots=True)
class MoveReport:
    """What one move did, in words and numbers."""

    move: str
    why: str
    change: str
    changed: int
    state: str
    level: int
    rpe: float


def _wave_drive() -> Callable[[str, float], None]:
    """Her thalamic oscillator's top-down drive, by band name."""

    def drive(band: str, strength: float) -> None:
        from thalamus.brain_waves import BrainWave, add_brain_wave_drive

        add_brain_wave_drive(BrainWave(band), strength)

    return drive


class Arcade:
    """Games on offer in her outer world.

    Args:
        world: Her ``OuterWorld``; the arcade attaches as ``world.arcade``.
        games: The catalogue (``OfficialGames`` by default — the API).
        wave_drive: ``(band, strength)`` into her brain-wave oscillator.
        voice: ``(text)`` callback for live thoughts; optional.
        image_dir: Where ``see_image`` saves board pictures.
        rng: Randomness for tie-breaking between equally good moves.
    """

    def __init__(
        self,
        world: Any,
        *,
        games: Any = None,
        wave_drive: Callable[[str, float], None] | None = None,
        voice: Callable[[str], None] | None = None,
        image_dir: str | Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        """Attach to the world; nothing is fetched until she asks."""
        self.world = world
        self.games = games if games is not None else OfficialGames()
        self._wave = wave_drive if wave_drive is not None else _wave_drive()
        self._voice = voice
        self.image_dir = (
            Path(image_dir) if image_dir else Path.home() / ".local/share/genesis/arcade"
        )
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._offers: list[GameOffer] = []
        self._offered_at = 0.0
        self.session: GameSession | None = None
        self.memories: dict[str, GameMemory] = {}
        self.last_error = ""
        self.visits = 0
        world.arcade = self

    # ── offers ─────────────────────────────────────────────────

    def announce(self) -> None:
        """Tell her the arcade is here, then fetch what it offers in the background.

        The notice enters her world's stream like any other event, so it
        reaches her workspace, her episodic memory and her dream replay.
        """
        network = getattr(self.world, "_network", None)
        if network is not None:
            try:
                network.add_concept("arcade", confidence=0.6, origin="world")
                network.add_concept("video game", confidence=0.6, origin="world")
            except Exception:  # knowing the word is best-effort
                logger.debug("arcade concept grounding failed", exc_info=True)
        self.world.notify(
            "There is an arcade in my world. It offers video games I can choose "
            "to play when I want (official ARC-AGI-3 games). " + RULES,
            salience=0.6,
        )
        record_event("arcade_open", "the arcade opened in her world")
        threading.Thread(target=self.offers, name="arcade-offers", daemon=True).start()

    def offers(self, *, refresh: bool = False) -> list[GameOffer]:
        """The games on offer right now; fetched from the API when stale."""
        with self._lock:
            if refresh or not self._offers or time.time() - self._offered_at > OFFER_REFRESH_S:
                try:
                    self._offers = self.games.offers()
                    self.last_error = ""
                except GamesUnavailable as exc:
                    self.last_error = str(exc)
                    return list(self._offers)
                self._offered_at = time.time()
                names = ", ".join(o.title or o.game_id for o in self._offers[:6])
                self.world.notify(
                    f"the arcade offers {len(self._offers)} games: {names}",
                    salience=0.3,
                )
            return list(self._offers)

    def pick(self) -> str | None:
        """Her choice of game: the open one, else the one least explored."""
        if self.session is not None and not (self.session.board and self.session.board.won):
            return self.session.game_id
        offers = self.offers()
        if not offers:
            return None
        unwon = [o for o in offers if not self._memory(o.game_id).wins] or offers
        return min(unwon, key=lambda o: (self._memory(o.game_id).moves, self._rng.random())).game_id

    # ── playing ────────────────────────────────────────────────

    def enter(self, game_id: str) -> Board | None:
        """Walk up to ``game_id`` and see its first board."""
        with self._lock:
            self.leave(quietly=True)
            try:
                self.session = self.games.open(game_id)
                board = self.session.reset()
            except GamesUnavailable as exc:
                self.last_error = str(exc)
                self.session = None
                return None
            self.visits += 1
            memory = self._memory(game_id)
            self.world.it_acted(f"went to play {game_id} in the arcade", detail=RULES)
            record_event(
                "arcade_enter", f"entered {game_id}", game=game_id, best_level=memory.best_level
            )
            if memory.level_paths:
                self._wave("theta", 0.15)
                self._say(f"I remember {game_id}. I got to level {memory.best_level} before.")
            if board is not None:
                self._perceive(
                    board,
                    sight.see_change(None, board.grid),
                    new=memory.see(sight.signature(board.grid)),
                )
            return board

    def leave(self, *, quietly: bool = False) -> None:
        """Step away from the current game; what she learned is kept."""
        with self._lock:
            session, self.session = self.session, None
        if session is None:
            return
        memory = self._memory(session.game_id)
        record_event(
            "arcade_leave",
            f"left {session.game_id}",
            game=session.game_id,
            best_level=memory.best_level,
            moves=memory.moves,
        )
        if not quietly:
            self.world.it_acted(f"stepped away from {session.game_id}")

    def moves(self, board: Board) -> list[str]:
        """The moves this board allows: plain actions, and pointing at shapes."""
        options = [a for a in board.actions if a not in ("RESET", POINTING_ACTION)]
        if POINTING_ACTION in board.actions:
            shapes = sight.see_objects(board.grid, limit=POINT_TARGETS + 1)[
                1:
            ]  # skip the background
            for shape in shapes[:POINT_TARGETS]:
                row, col = shape.center
                options.append(f"{POINTING_ACTION}@{row},{col}")
        return options or ["RESET"]

    def move(self, move: str, why: str = "her choice") -> MoveReport | None:
        """Make one move on the open game and feel what it did."""
        with self._lock:
            session = self.session
            if session is None or session.board is None:
                return None
            before = session.board
            memory = self._memory(session.game_id)
            name, _, at = move.partition("@")
            x = y = None
            if at:
                row, col = (int(v) for v in at.split(","))
                x, y = col, row
            try:
                after = session.step(name, x=x, y=y)
            except (GamesUnavailable, ValueError) as exc:
                self.last_error = str(exc)
                return None
            except Exception as exc:  # noqa: BLE001 — the API can fail mid-game
                self.last_error = f"move failed: {exc}"
                return None
            if after is None:
                return None
            change = sight.see_change(before.grid, after.grid)
            new = memory.see(sight.signature(after.grid))
            reward = REWARD_NEW_BOARD if new and change.count else REWARD_NOTHING
            levelled = after.levels_completed > before.levels_completed
            if levelled:
                reward += REWARD_LEVEL
            if after.won:
                reward += REWARD_WIN
            if after.over:
                reward += REWARD_GAME_OVER
            rpe = memory.learn(
                sight.signature(before.grid),
                move,
                reward,
                sight.signature(after.grid),
                bool(change.count),
                self.moves(after),
            )
            self.world.it_acted(f"pressed {move} in {session.game_id}", detail=change.describe())
            self._wave("beta2", 0.05)
            self._impulse(CHEM_DOPAMINE, max(-0.05, min(0.05, 0.05 * rpe)))
            self._perceive(after, change, new=new)
            if levelled:
                memory.level_up(after.levels_completed)
                self._celebrate(session.game_id, after)
            if after.won:
                memory.wins += 1
                self._impulse(CHEM_ENDORPHIN, 0.05)
                self._impulse(CHEM_SEROTONIN, 0.04)
                record_event(
                    "arcade_win", f"won {session.game_id}", game=session.game_id, moves=memory.moves
                )
                self._say(f"I won {session.game_id}!")
            if after.over:
                memory.game_overs += 1
                self._impulse(CHEM_CORTISOL, 0.02)
                self._impulse(CHEM_NOREPINEPHRINE, 0.02)
                record_event(
                    "arcade_game_over", f"game over in {session.game_id}", game=session.game_id
                )
                session.reset()
            return MoveReport(
                move, why, change.describe(), change.count, after.state, after.levels_completed, rpe
            )

    def play(self, max_moves: int = BOUT_MOVES) -> str:
        """One bout of play she chose: explore, remember, try to win."""
        game_id = self.pick()
        if game_id is None:
            return f"the arcade is closed right now ({self.last_error or 'no games offered'})"
        if self.session is None or self.session.game_id != game_id:
            if self.enter(game_id) is None:
                return f"couldn't open {game_id}: {self.last_error}"
        session = self.session
        assert session is not None and session.board is not None
        memory = self._memory(game_id)
        start_level, made = session.board.levels_completed, 0
        for _ in range(max_moves):
            board = session.board
            if board is None or board.won:
                break
            choice, why = memory.choose(
                sight.signature(board.grid), self.moves(board), board.levels_completed, self._rng
            )
            if self.move(choice, why) is None:
                break
            made += 1
        level = session.board.levels_completed if session.board else start_level
        goal = session.board.win_levels if session.board else "?"
        summary = f"played {made} moves of {game_id}: level {level}/{goal}"
        record_event(
            "arcade_bout",
            summary,
            game=game_id,
            moves=made,
            level=level,
            best_level=memory.best_level,
        )
        return summary

    # ── seeing ─────────────────────────────────────────────────

    def see(self, how: str = "text") -> str:
        """Look at the open board: ``text``, ``image``, ``change`` or ``objects``."""
        session = self.session
        if session is None or session.board is None:
            return "no game is open"
        grid = session.board.grid
        if how == "image":
            stamp = time.strftime("%Y%m%d-%H%M%S")
            return str(sight.see_image(grid, self.image_dir / f"{session.game_id}-{stamp}.png"))
        if how == "objects":
            return sight.describe_objects(sight.see_objects(grid))
        if how == "change":
            return f"moves allowed: {', '.join(self.moves(session.board))}"
        return sight.see_text(grid)

    # ── telemetry and persistence ──────────────────────────────

    def telemetry(self) -> dict[str, Any]:
        """Numbers for her status, board and journal."""
        session = self.session
        board = session.board if session else None
        return {
            "offered": len(self._offers),
            "playing": session.game_id if session else None,
            "level": board.levels_completed if board else None,
            "win_levels": board.win_levels if board else None,
            "state": board.state if board else None,
            "visits": self.visits,
            "games_known": len(self.memories),
            "wins": sum(m.wins for m in self.memories.values()),
            "moves": sum(m.moves for m in self.memories.values()),
            "keyed": bool(getattr(self.games, "keyed", False)),
            "error": self.last_error,
        }

    def summary_line(self) -> str:
        """One line about the arcade for her world summary."""
        t = self.telemetry()
        playing = (
            f"playing {t['playing']} (level {t['level']}/{t['win_levels']}, {t['state']})"
            if t["playing"]
            else "not playing"
        )
        return (
            f"  arcade: {t['offered']} games offered; {playing}; "
            f"{t['moves']} moves, {t['wins']} wins remembered"
        )

    def status(self) -> str:
        """A readable report for ``/arcade``."""
        lines = ["The arcade (official ARC-AGI-3 games)", self.summary_line().strip()]
        for game_id, memory in sorted(self.memories.items()):
            lines.append(
                f"  {game_id}: best level {memory.best_level}, {memory.moves} moves, "
                f"{memory.wins} wins, {memory.game_overs} game overs, "
                f"{len(memory.visits)} boards remembered"
            )
        if self.last_error:
            lines.append(f"  last problem: {self.last_error}")
        lines.append(f"  rules: {RULES}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize what she remembers of every game."""
        return {
            "visits": self.visits,
            "memories": {k: m.to_dict() for k, m in self.memories.items()},
        }

    def restore(self, data: dict[str, Any] | None) -> None:
        """Restore memories saved by ``to_dict``."""
        if not data:
            return
        self.visits = int(data.get("visits", 0))
        self.memories = {
            k: GameMemory.from_dict(v) for k, v in (data.get("memories") or {}).items()
        }

    # ── tools ──────────────────────────────────────────────────

    def register_tools(self, registry: Any) -> None:
        """Give her tools for the arcade in her tool registry."""
        from tools.framework import Tool, ToolResult

        def wrap(fn: Callable[..., str]) -> Callable[..., ToolResult]:
            def run(**kwargs: Any) -> ToolResult:
                try:
                    return ToolResult(success=True, output=fn(**kwargs))
                except Exception as exc:  # noqa: BLE001 — a tool reports, never raises
                    return ToolResult(success=False, error=str(exc))

            return run

        def enter(game_id: str = "") -> str:
            chosen = game_id or self.pick() or ""
            return (
                self.see("text")
                if chosen and self.enter(chosen)
                else f"couldn't enter: {self.last_error}"
            )

        def act(move: str = "") -> str:
            report = self.move(move)
            return (
                f"{report.change}; state {report.state}, level {report.level}"
                if report
                else self.last_error
            )

        def leave() -> str:
            self.leave()
            return "left the game"

        tools = (
            (
                "arcade_directions",
                "Read the directions for the games in the arcade",
                lambda: directions(),
            ),
            (
                "arcade_offers",
                "List the games the arcade offers",
                lambda: "\n".join(f"{o.game_id} {o.title}" for o in self.offers()),
            ),
            ("arcade_enter", "Go to a game in the arcade (game_id, or let her pick)", enter),
            (
                "arcade_see",
                "See the open board: how=text|image|change|objects",
                lambda how="text": self.see(how),
            ),
            ("arcade_act", "Make one move, e.g. ACTION1 or ACTION6@row,col", act),
            (
                "arcade_play",
                "Play a bout of the open (or chosen) game",
                lambda moves=BOUT_MOVES: self.play(int(moves)),
            ),
            (
                "arcade_leave",
                "Step away from the game; memories are kept",
                leave,
            ),
            ("arcade_status", "What she remembers of every game", self.status),
        )
        for name, description, fn in tools:
            registry.register(Tool(name, description, wrap(fn)))

    # ── internals ──────────────────────────────────────────────

    def _memory(self, game_id: str) -> GameMemory:
        """Her memory of ``game_id``, created on first visit."""
        memory = self.memories.get(game_id)
        if memory is None:
            memory = self.memories[game_id] = GameMemory(game_id)
        return memory

    def _perceive(self, board: Board, change: sight.Change, *, new: bool) -> None:
        """Let the board reach her through the world, with its feeling."""
        salience = 0.45 + (0.15 if new else 0.0) + min(0.2, change.count / 400)
        if new:
            self._impulse(CHEM_NOREPINEPHRINE, 0.01)
            self._impulse(CHEM_ACETYLCHOLINE, 0.01)
            self._wave("gamma", 0.08)
        self.world.perceive(
            f"the {board.game_id} board ({'new' if new else 'seen before'}): {change.describe(6)}",
            salience=min(0.9, salience),
            source=f"arcade:{board.game_id}",
            metadata={
                "game": board.game_id,
                "state": board.state,
                "level": board.levels_completed,
                "win_levels": board.win_levels,
                "changed_cells": change.count,
                "new_board": new,
            },
            percept={"grid": board.grid},
        )

    def _celebrate(self, game_id: str, board: Board) -> None:
        """A level opened: reward, remember, and say so."""
        self._impulse(CHEM_DOPAMINE, 0.08)
        self._impulse(CHEM_ENDORPHIN, 0.03)
        self._wave("theta", 0.1)
        record_event(
            "arcade_level",
            f"level {board.levels_completed} in {game_id}",
            game=game_id,
            level=board.levels_completed,
        )
        self.world.perceive(
            f"I got out of level {board.levels_completed} of {game_id}",
            salience=0.75,
            source=f"arcade:{game_id}",
            metadata={"game": game_id, "level": board.levels_completed},
        )
        self._say(f"I solved level {board.levels_completed} of {game_id}. I'll remember how.")

    def _impulse(self, chem: int, amount: float) -> None:
        """Neurochemistry through the world's own impulse path."""
        if amount:
            self.world._impulse(chem, amount)

    def _say(self, text: str) -> None:
        """Speak a live thought if the mind gave the arcade a voice."""
        if self._voice is not None:
            try:
                self._voice(text)
            except Exception:
                logger.debug("arcade voice failed", exc_info=True)

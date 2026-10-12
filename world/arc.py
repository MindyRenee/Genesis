"""ARC-AGI-3 practice — real ``arcengine`` environments through her play loop.

This is the adapter ``game_turn`` was written for: the mind's game
session speaks the harness protocol (``FRAME | ACTIONS | LEVELS |
GAME | EVENT`` in, ``ACTIONn``/``RESET`` out), and the ``arc-agi``
package is the world side. This module is the seam that binds them —
it owns no cognition, only translation and the duty of care:

    arcengine FrameDataRaw  ->  FRAME line  ->  mind.game_turn
    session action token    ->  env.step(GameAction, reasoning=...)

Frames travel through the ``@file`` form the session accepts: a full
64×64 board as inline hex exceeds the line the protocol was sized
for, so each frame is written as one hex row per line to a temp file
— her "screen" for the turn.

Two contracts keep a run honest:

- **Game mode.** ``play()`` only runs while the mind is in game mode
  (``enter_game_mode()``): the session's volition/learner suppression
  and the frame-turn protocol both live behind that flag. A caller
  that skips it gets refused, because a run that doesn't suppress her
  other urges isn't measuring the same entity.
- **Readiness.** Her own health gate is checked before the run,
  at every episode boundary, and periodically mid-episode. When she
  is asleep, drowsy, stressed, overloaded, or unreachable, the run
  does not start or stops early — a body that isn't well shouldn't
  be worked.

The ARC side judges honestly and tells her nothing: available action
ids, the frame, levels completed. Goal inference, mechanics, and
planning are all on her side of the boundary. What crosses back is
her decision basis — ``env.step``'s ``reasoning`` field records the
structured self-report her striatum left behind.

This lives in the world, not the frontal lobe: it is the environment
she plays, and it is never constructed by a caller that builds its own
mind. ``Mind.play_arc()`` owns the entity side (game mode around each
run); the genesis CLI is the entrypoint::

    /arc ls20-9607627b ft09-0d8bbf25 --steps 400
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mind import Mind

_HEALTH_EVERY_STEPS = 25


@dataclass
class ArcResult:
    """What the world reported at the end of one run."""

    game_id: str
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 0
    steps: int = 0
    scorecard_id: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def won(self) -> bool:
        return self.state == "WIN"


class ArcPractice:
    """One ARC game played through her faculties via the session protocol."""

    def __init__(
        self,
        mind: Mind,
        environments_dir: str | Path | None = None,
        recordings_dir: str | Path | None = None,
        instructions: str = "",
        require_live: bool = True,
        record: bool = True,
    ) -> None:
        from arc_agi import Arcade, OperationMode

        from infrastructure.config import default_data_dir

        # Downloaded engines and recordings are local artifacts, not
        # source — they live beside her other state, not in the repo.
        # OFFLINE: the games are the local environment files; no
        # scorecard is opened on the remote service.
        arc_dir = default_data_dir() / "arc"
        self._mind = mind
        self._arcade = Arcade(
            environments_dir=str(environments_dir or arc_dir / "environment_files"),
            recordings_dir=str(recordings_dir or arc_dir / "recordings"),
            operation_mode=OperationMode.OFFLINE,
        )
        self._tmp = Path(tempfile.mkdtemp(prefix="arc_frames_"))
        self._pending_event = ""
        # The player's manual: what a human is told before the first
        # move ("this is a video game — win the levels"). Delivered on
        # the first frame of each game; her cognition gets the genre
        # framing instead of having to deduce that games have goals.
        self._instructions = instructions
        self._info_sent: set[str] = set()
        # require_live=False is the explicit disembodied-debug path —
        # the readiness gate still applies its in-process checks.
        self._require_live = require_live
        self._record = record

    # ── Frame translation ─────────────────────────────────────

    def _frame_line(self, frame: Any, game_id: str) -> str:
        """FrameDataRaw -> the session's turn line (``@file`` screen)."""
        grid = frame.frame[-1]  # newest board of the sequence
        path = self._tmp / f"{game_id.split('-')[0]}.frame.txt"
        path.write_text(
            "\n".join("".join(f"{int(c):x}" for c in row) for row in grid),
            encoding="utf-8",
        )
        avail = ",".join(str(int(a)) for a in (frame.available_actions or []))
        line = (
            f"FRAME @{path}"
            f" | ACTIONS {avail}"
            f" | LEVELS {frame.levels_completed}/{frame.win_levels}"
            f" | GAME {game_id}"
        )
        if self._instructions and game_id not in self._info_sent:
            line += f" | INFO {self._instructions}"
            self._info_sent.add(game_id)
        if self._pending_event:
            line += f" | EVENT {self._pending_event}"
            self._pending_event = ""
        return line

    @staticmethod
    def _to_step(token: str) -> tuple[Any, dict[str, int] | None]:
        """Session action token -> ``env.step`` arguments."""
        from arcengine import GameAction

        if token == "RESET":
            return GameAction.RESET, None
        parts = token.split()
        action = GameAction.from_id(int(parts[0].removeprefix("ACTION")))
        if len(parts) == 3:  # ACTION6 x y
            return action, {"x": int(parts[1]), "y": int(parts[2])}
        return action, None

    # ── Health gate ───────────────────────────────────────────

    def _check_readiness(self) -> tuple[str, ...] | None:
        """Refusal reasons when she isn't fit to play, else None."""
        try:
            report = self._mind.readiness(require_body=self._require_live)
        except Exception:  # noqa: BLE001 — a probe that fails must not pretend health
            return ("readiness probe failed",)
        return None if report.ok else report.reasons

    # ── The run ───────────────────────────────────────────────

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        """Play one game; stop on WIN, budget, illness, or dead actions."""
        from arcengine import GameState

        result = ArcResult(game_id=game_id)
        # Game mode is the contract: it is what suppresses her other
        # drives so the run measures the entity at play, not the
        # collision of play with every urge that happened to fire.
        if not self._mind.is_gaming:
            raise RuntimeError(
                "ArcPractice requires game mode — call "
                "mind.enter_game_mode() before play, exit_game_mode() after"
            )
        if (reasons := self._check_readiness()) is not None:
            result.state = "UNWELL"
            result.reasons = reasons
            return result

        scorecard = self._arcade.create_scorecard()
        env = self._arcade.make(
            game_id,
            scorecard_id=scorecard,
            save_recording=self._record,
        )
        result.scorecard_id = scorecard
        if env is None:
            result.state = "UNAVAILABLE"
            return result
        frame = env.reset()
        while result.steps < max_steps and frame is not None:
            result.levels_completed = frame.levels_completed
            result.win_levels = frame.win_levels
            state = getattr(frame.state, "name", str(frame.state))
            if state == GameState.WIN.name:
                result.state = state
                self._mind.game_turn(
                    self._frame_line(frame, game_id) + " | EVENT win"
                )
                break
            if state == GameState.GAME_OVER.name:
                self._pending_event = "over"
                frame = env.reset()
                # A death is a checkpoint — if the last episode left
                # her unwell (stress, mounting sleep pressure), the
                # next one doesn't start. Recovery beats a score.
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                continue
            if not frame.available_actions:
                result.state = "NO_ACTIONS"
                break
            token = self._mind.game_turn(self._frame_line(frame, game_id))
            result.steps += 1
            if not token:
                break
            action, data = self._to_step(token)
            # Her decision basis rides the step — the recording keeps
            # what her striatum actually computed, not a retelling.
            reasoning = self._mind.game_decision
            try:
                frame = env.step(action, data=data, reasoning=reasoning)
            except TypeError:  # older wrappers without the field
                frame = env.step(action, data=data)
            if result.steps % _HEALTH_EVERY_STEPS == 0:
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
        if result.state == "NOT_FINISHED":
            if frame is not None:
                result.state = getattr(getattr(frame, "state", None), "name", result.state)
            if result.state == "NOT_FINISHED":
                result.state = "BUDGET"
        try:
            self._arcade.close_scorecard(scorecard)
        except Exception:  # noqa: BLE001
            pass
        return result



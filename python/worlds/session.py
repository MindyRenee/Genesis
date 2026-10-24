"""External game session — the text-protocol front-end for play.

This is a *wire shim*, not a brain path: it parses the terminal
frame format and hands the turn to ``mind.world_step`` — the
embodied loop where her faculties live (perception through the
embodied stack, cognition slice, basal-ganglia action over the
motor field). Everything a frame does to her — feeling, learning,
speaking inwardly — happens inside ``world_step``; this file only
translates between that loop and a text channel a human can drive.

Wire protocol — one line per turn, all fields optional except the
frame itself:

    FRAME <rows, '/'-separated hex digits | @/path/to/frame.txt>
        | ACTIONS 1,2,3 | LEVELS a/b [| GAME name] [| EVENT win|over]

The ``@path`` form reads the frame from a file (one hex row per
line) — a full 64×64 grid exceeds the terminal's canonical input
limit, so the file is her "screen" for large frames.

``EVENT`` marks the terminal outcome of the step just taken; the
adapter resets immediately on game over, so the terminal belongs
to the frame that follows the reset. ``LEVELS`` advancing is the
goal signal — ``world_step`` marks it itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mind import Mind


class GameSession:
    """One external game stream — parser/renderer for world_step."""

    def __init__(self, mind: Mind) -> None:
        self._mind = mind
        # Mirror of the mind's decision telemetry for callers that
        # held a session reference.
        self.last_decision: dict[str, Any] | None = None

    # ── Parsing ───────────────────────────────────────────────

    @staticmethod
    def _fields(line: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for part in line.split(" | "):
            key, _, val = part.partition(" ")
            out[key.strip().upper()] = val.strip()
        return out

    @staticmethod
    def _rows(frame_ref: str) -> list[list[int]]:
        """Rows from inline '/'-separated hex or an ``@file`` path."""
        if frame_ref.startswith("@"):
            from pathlib import Path

            text = Path(frame_ref[1:]).read_text(encoding="utf-8")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            return [[int(c, 16) for c in ln] for ln in lines]
        return [[int(c, 16) for c in row] for row in frame_ref.split("/")]

    # ── Turns ─────────────────────────────────────────────────

    def turn(self, line: str) -> str:
        """One game turn in, one action token out."""
        fields = self._fields(line)
        frame_txt = fields.get("FRAME", "")
        if not frame_txt:
            return ""
        rows = self._rows(frame_txt)
        avail = [int(a) for a in fields.get("ACTIONS", "").split(",") if a.strip()]
        if not avail:
            return ""
        a_done, _, a_win = fields.get("LEVELS", "0/1").partition("/")
        levels, win_levels = int(a_done or 0), int(a_win or 1)
        game = fields.get("GAME", "the game")

        # Terminal outcome of the step just taken arrives on the
        # next frame — close the episode before the fresh frame is
        # observed.
        event = fields.get("EVENT", "").lower()
        if event in ("win", "over"):
            self._mind.world_episode_end(event, game)

        desc = self._mind.world_step(
            rows,
            available_actions=avail,
            info={
                "game": game,
                "level": levels,
                "win_levels": win_levels,
                **({"instructions": fields["INFO"]} if "INFO" in fields else {}),
            },
        )
        self.last_decision = self._mind.last_world_decision
        if desc is None:
            return ""
        action = desc["action"]
        if action == 0:
            return "RESET"
        if action in (6, 7):
            return f"ACTION{action} {desc['x']} {desc['y']}"
        return f"ACTION{action}"

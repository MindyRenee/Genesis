"""The ARC cabinet — hands for the play loop.

When a run opens inside game mode and an X11 display is reachable,
a committed action does not go straight to ``env.step``: it is
emitted as real keystrokes (``xdo``) into a small cabinet window,
and the window's keymap translates what physically arrived back
into a ``GameAction``. The world sees only what landed on the
glass — the same contract a human's keyboard gives the game, so a
fumbled press is a lost move, not a retried one.

Keymap
------
- ``1``–``5``, ``7``  — simple actions (the engine's action ids)
- ``r`` / ``0``       — RESET
- ``x NN y NN ⏎``     — ACTION6's coordinates, typed
- ``Esc``             — cancel a half-typed coordinate

Synthetic X input sprinkles stray keysyms — the decoder
whitelists its own alphabet and commits only complete tokens, so
noise can drop a move but never invent a different one.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from importlib.util import find_spec
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cell pixel size in the cabinet window — the board is 64 cells.
_SCALE = 6
# Fallback palette (index -> #rrggbb) when the engine's own
# COLOR_MAP can't be imported — matches the ARC 16-color range.
_GREY16 = [f"#{i * 17:02x}{i * 17:02x}{i * 17:02x}" for i in range(16)]

_SIMPLE_KEYS = {"1", "2", "3", "4", "5", "7"}
_ALPHABET = _SIMPLE_KEYS | {"0", "6", "8", "9", "x", "y", "r", "Return", "Escape"}


class _KeyDecoder:
    """Keysym stream -> complete action tokens.

    Pure state machine — feed it keysyms, it returns
    ``(action_id, {"x":x,"y":y}|None)`` when a token completes, else
    None. The coordinate token is ``x<digits>y<digits>Return``;
    ``Escape`` abandons it. Keysyms outside the alphabet are
    ignored — synthetic input arrives with noise on the wire.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_coord = False

    def reset(self) -> None:
        self._buf = ""
        self._in_coord = False

    def feed(self, keysym: str) -> tuple[int, dict[str, int] | None] | None:
        if keysym not in _ALPHABET:
            return None
        if self._in_coord:
            if keysym == "Escape":
                self.reset()
                return None
            if keysym == "Return":
                token, self._buf = self._buf, ""
                self._in_coord = False
                return self._parse_coord(token)
            if keysym in "xy":
                self._buf += keysym
            elif keysym.isdigit() or len(keysym) == 1:
                # digits append; other whitelisted keys don't
                self._buf += keysym if keysym.isdigit() else ""
            return None
        if keysym == "x":
            self._in_coord = True
            self._buf = "x"
            return None
        if keysym in _SIMPLE_KEYS:
            return int(keysym), None
        if keysym in ("r", "0"):
            return 0, None
        # Return/Escape/6/8/9/x-stray with no open token — noise
        return None

    @staticmethod
    def _parse_coord(token: str) -> tuple[int, dict[str, int] | None] | None:
        """``xNNyNN`` -> ACTION6; anything malformed is a lost press."""
        if not token.startswith("x") or "y" not in token[1:]:
            return None
        xs, ys = token[1:].split("y", 1)
        if not (xs.isdigit() and ys.isdigit()):
            return None
        x, y = int(xs), int(ys)
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return None
        return 6, {"x": x, "y": y}


def _keystrokes_for(action_id: int, data: dict[str, int] | None) -> list[tuple[str, str]]:
    """One action -> the emitted keystroke ops.

    Returns ``[("key", keysym-seq) | ("text", literal), ...]`` —
    simple actions and RESET are single keys; ACTION6 types its
    coordinates then presses Return.
    """
    if action_id == 0:
        return [("key", "r")]
    if action_id == 6 and data is not None:
        return [("text", f"x{data['x']}y{data['y']}"), ("key", "Return")]
    if 1 <= action_id <= 7:
        return [("key", str(action_id))]
    raise ValueError(f"no keymap for action {action_id}")


def x11_reachable() -> bool:
    """Can a cabinet window open and xdo reach it?"""
    if not os.environ.get("DISPLAY"):
        if Path("/tmp/.X11-unix/X0").exists():
            os.environ.setdefault("DISPLAY", ":0.0")
        else:
            return False
    if find_spec("tkinter") is None or find_spec("xdo") is None:
        return False
    try:
        import xdo

        xdo.xdo()
    except Exception:  # noqa: BLE001
        return False
    return True


class ArcCabinet:
    """A small X11 window that receives her keystrokes.

    ``dispatch(action, data)`` emits the action's keystrokes at the
    window, then pumps events until the decoder reports what
    physically arrived — the ``(GameAction, data)`` to submit — or
    the timeout expires (a lost press; the caller spends the move
    without stepping the world). ``render(frame)`` repaints the
    board so the next press is aimed at a visible state.
    """

    def __init__(self, game_id: str, timeout_s: float = 3.0) -> None:
        import tkinter as tk

        import xdo

        self._tk_mod = tk
        self._root = tk.Tk()
        self._root.title(f"genesis · arc · {game_id}")
        size = 64 * _SCALE
        self._canvas = tk.Canvas(self._root, width=size, height=size, highlightthickness=0)
        self._canvas.pack()
        self._dec = _KeyDecoder()
        self._arrived: deque[tuple[int, dict[str, int] | None]] = deque(maxlen=4)
        self._root.bind("<KeyPress>", self._on_key)
        self._root.update()
        self._win = self._root.winfo_id()
        self._xdo = xdo.xdo()
        self._timeout = max(0.2, float(timeout_s))
        self._photo: Any = None
        self._zoomed: Any = None

    def _on_key(self, event: Any) -> None:
        token = self._dec.feed(getattr(event, "keysym", ""))
        if token is not None:
            self._arrived.append(token)

    def dispatch(
        self, action: Any, data: dict[str, int] | None
    ) -> tuple[Any, dict[str, int] | None] | None:
        """Emit the action's keystrokes; return what arrived."""
        from arcengine import GameAction

        action_id = getattr(action, "value", action)
        if isinstance(action_id, tuple):
            action_id = action_id[0]
        action_id = int(action_id)
        self._dec.reset()
        self._arrived.clear()
        for kind, payload in _keystrokes_for(action_id, data):
            try:
                if kind == "key":
                    self._xdo.send_keysequence_window(payload, window=self._win)
                else:
                    self._xdo.enter_text_window(payload, window=self._win)
            except Exception as e:  # noqa: BLE001
                logger.info("[cabinet] keystroke emission failed: %s", e)
                return None
            time.sleep(0.02)
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                self._root.update()
            except Exception:  # noqa: BLE001 — window died
                return None
            if self._arrived:
                arrived_id, arrived_data = self._arrived.popleft()
                return GameAction.from_id(arrived_id), arrived_data
            time.sleep(0.01)
        return None

    def render(self, frame: Any) -> None:
        """Repaint the board from a FrameData/FrameDataRaw frame."""
        grid = getattr(frame, "frame", None)
        if grid is None:
            return
        # Frames may nest ([t][h][w]) — descend to the 64x64 board.
        while (
            isinstance(grid, (list, tuple))
            and grid
            and isinstance(grid[0], (list, tuple))
            and grid[0]
            and isinstance(grid[0][0], (list, tuple))
        ):
            grid = grid[-1]
        if not (isinstance(grid, (list, tuple)) and grid and isinstance(grid[0], (list, tuple))):
            return
        try:
            from arc_agi.rendering import COLOR_MAP

            colors = {k: v[:7] for k, v in COLOR_MAP.items()}
        except Exception:  # noqa: BLE001
            colors = dict(enumerate(_GREY16))
        tk = self._tk_mod
        photo = tk.PhotoImage(width=64, height=64)
        for y, row in enumerate(grid):
            cells = " ".join(colors.get(int(v) & 0x0F, "#000000") for v in row[:64])
            photo.put("{" + cells + "}", to=(0, y))
        self._zoomed = photo.zoom(_SCALE)
        self._photo = photo
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._zoomed)
        self._root.update()

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"close failed: {e}")

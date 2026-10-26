"""The ARC cabinet — a physical control surface for the play loop.

When a run opens inside game mode and an X11 display is reachable,
a committed action does not go straight to ``env.step``: it is
emitted as real input events — keys via XTEST, pointer moves and
clicks via xdotool — into a small cabinet window. The window's own
controls map what physically arrived back into a ``GameAction``.
The world sees only what landed on the glass — the same contract a
human's keyboard and mouse give the game, so a fumbled press or a
click that missed the board is a lost move, never a retried one.

Controls — the same surface the venue gives every player
--------------------------------------------------------
- ``W``/``↑``   ACTION1      ``S``/``↓``   ACTION2
- ``A``/``←``   ACTION3      ``D``/``→``   ACTION4
- ``Space``/``F``  ACTION5   ``Z``         ACTION7 (undo)
- pointer click on the board  ACTION6 at the clicked cell
- ``RESET`` button / ``R``    RESET

Synthetic X input sprinkles stray keysyms and clicks land wherever
the pointer actually is — the decoder whitelists its own control
surface, so noise can drop a move but never invent a different one.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import time
from collections import deque
from importlib.util import find_spec
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cell pixel size on the board — 64 cells across.
_SCALE = 6
# Where the cabinet parks itself on the display — a fixed seat so the
# eye's capture region is stable across runs.
_ORIGIN = (60, 60)
# Fallback palette (index -> #rrggbb) when the engine's own
# COLOR_MAP can't be imported — matches the ARC 16-color range.
_GREY16 = [f"#{i * 17:02x}{i * 17:02x}{i * 17:02x}" for i in range(16)]

# The venue's documented control surface (docs.arcprize.org/actions):
# WASD or arrows carry ACTIONS 1-4, Space/F carries ACTION5, Z is
# undo, a board click is ACTION6, RESET is its own control. Which key
# posts which action is interface structure — what an action *does*
# in a game is still hers to learn.
_KEY_ACTIONS = {
    "w": 1,
    "Up": 1,
    "s": 2,
    "Down": 2,
    "a": 3,
    "Left": 3,
    "d": 4,
    "Right": 4,
    "space": 5,
    "f": 5,
    "z": 7,
    "Z": 7,
    "r": 0,
    "R": 0,
}
_EMIT_KEY = {1: "w", 2: "s", 3: "a", 4: "d", 5: "space", 7: "z"}


class _Decoder:
    """Physical input events -> actions.

    Pure surface mapping — feed it keysyms and click positions, it
    returns ``(action_id, {"x":x,"y":y}|None)`` for each control that
    fired, else None. A click off the board is a lost press: the
    pointer hit glass that wasn't a control. Keysyms outside the
    control surface are ignored — synthetic input arrives with noise
    on the wire.
    """

    def feed_key(self, keysym: str) -> tuple[int, dict[str, int] | None] | None:
        """One keysym -> an action id, or None if it isn't a control."""
        action_id = _KEY_ACTIONS.get(keysym)
        return (action_id, None) if action_id is not None else None

    @staticmethod
    def feed_click(px: int, py: int) -> tuple[int, dict[str, int]] | None:
        """A click at board pixel (px, py) -> ACTION6 at that cell.

        The cell is what the pointer physically covered. A click that
        lands outside the 64x64 board is not a move — the venue's
        canvas ignores it too.
        """
        x, y = int(px) // _SCALE, int(py) // _SCALE
        if not (0 <= x < 64 and 0 <= y < 64):
            return None
        return 6, {"x": x, "y": y}


def _controls_for(action_id: int, data: dict[str, int] | None) -> list[tuple[Any, ...]]:
    """One action -> the physical ops to emit.

    Returns ``[("key", keysym) | ("click", cell_x, cell_y) |
    ("reset",), ...]`` — simple actions are single keys; ACTION6 is a
    pointer click at the cell; RESET clicks the RESET control the way
    the venue's button is clicked.
    """
    if action_id == 0:
        return [("reset",)]
    if action_id == 6 and data is not None:
        return [("click", int(data["x"]), int(data["y"]))]
    if action_id in _EMIT_KEY:
        return [("key", _EMIT_KEY[action_id])]
    raise ValueError(f"no control for action {action_id}")


def x11_reachable() -> bool:
    """Can a cabinet window open and input reach it?"""
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


# Her screen, not the host's: XTEST events on a shared display move the
# user's real pointer and inject keys into whatever has focus. A nested
# X server gives the cabinet a physical display of its own — real
# events, real pixels for the eye — that never borrows the user's hands.
_arc_display: str | None = None
_arc_display_proc: subprocess.Popen | None = None
_ARC_SCREEN = "760x560"


def _display_usable(display: str) -> bool:
    """Can an X client actually connect to this display?"""
    if find_spec("xdo") is None:
        return False
    prev = os.environ.get("DISPLAY")
    os.environ["DISPLAY"] = display
    try:
        import xdo

        xdo.xdo()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        if prev is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = prev


def _free_display_number() -> int:
    """First X display number not already claimed on this host."""
    used = set()
    for sock in Path("/tmp/.X11-unix").glob("X*"):
        try:
            used.add(int(sock.name[1:]))
        except ValueError:
            continue
    n = 1
    while n in used:
        n += 1
    return n


def ensure_arc_display() -> str | None:
    """The display the cabinet lives on — hers, not the user's.

    ``GENESIS_ARC_DISPLAY`` pins the choice: ``shared`` uses the
    inherited session display (her acts land on the user's screen —
    the pre-fix behavior), or any ``:N``/``host:N`` spec uses that
    display directly. Unset: she gets a nested X server of her own —
    spawned once per process and killed at exit — with the inherited
    display as fallback when no nested server is available. None means
    no display can serve her at all (the run degrades to bench play).
    """
    global _arc_display, _arc_display_proc
    override = os.getenv("GENESIS_ARC_DISPLAY", "").strip()
    if override and override not in ("shared", "user", "host"):
        return override
    if override or _arc_display is not None:
        # Explicit shared mode, or the nested display is already up.
        if override:
            return os.environ.get("DISPLAY")
        return _arc_display
    if find_spec("tkinter") is None or find_spec("xdo") is None:
        return None
    xephyr = shutil.which("Xephyr")
    if xephyr is not None:
        num = _free_display_number()
        name = f":{num}"
        try:
            proc = subprocess.Popen(
                [xephyr, name, "-screen", _ARC_SCREEN, "-br", "-title", "genesis arcade"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                if _display_usable(name):
                    _arc_display, _arc_display_proc = name, proc
                    atexit.register(_drop_arc_display)
                    logger.info("[cabinet] her screen is up on %s", name)
                    return name
                time.sleep(0.15)
            proc.terminate()
        except Exception as e:  # noqa: BLE001
            logger.info("[cabinet] nested display unavailable: %s", e)
    # No nested server — she plays on the session display if one exists.
    if os.environ.get("DISPLAY") or Path("/tmp/.X11-unix/X0").exists():
        return os.environ.setdefault("DISPLAY", ":0.0")
    return None


def _drop_arc_display() -> None:
    """Tear down the spawned nested display at process exit."""
    global _arc_display, _arc_display_proc
    proc, _arc_display, _arc_display_proc = _arc_display_proc, None, None
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


class ArcCabinet:
    """A small X11 window: the board she plays, the controls she presses.

    ``dispatch(action, data)`` emits the action's input events at the
    window — keys through XTEST, clicks through xdotool's real
    pointer — then pumps events until the decoder reports what
    physically arrived: the ``(GameAction, data)`` to submit, or
    nothing before the timeout (a lost press; the caller spends the
    move without stepping the world). ``render(frame)`` repaints the
    board so the next act is aimed at a visible state.
    """

    def __init__(self, game_id: str, timeout_s: float = 3.0, display: str | None = None) -> None:
        """Open the window titled for this game and bind the controls.

        ``timeout_s`` caps how long ``dispatch`` waits for the
        emitted input to arrive back through the window — an act
        that never lands is a lost move, not a hang. ``display`` is
        the X display the cabinet lives on: her own nested screen by
        default, so emitted input never moves a host user's pointer.
        The env is set only for the construction window — Tk and xdo
        bind their display at open, and later emission goes through
        explicit per-subprocess envs.
        """
        import tkinter as tk

        import xdo

        self._display = display
        prev = os.environ.get("DISPLAY")
        if display is not None:
            os.environ["DISPLAY"] = display
        try:
            self._tk_mod = tk
            self._root = tk.Tk()
            self._root.title(f"genesis · arc · {game_id}")
            size = 64 * _SCALE
            self._canvas = tk.Canvas(self._root, width=size, height=size, highlightthickness=0)
            self._canvas.pack()
            self._reset_btn = tk.Button(self._root, text="RESET", command=self._on_reset)
            self._reset_btn.pack(fill="x")
            self._dec = _Decoder()
            self._arrived: deque[tuple[int, dict[str, int] | None]] = deque(maxlen=4)
            self._root.bind("<KeyPress>", self._on_key)
            self._canvas.bind("<Button-1>", self._on_click)
            # A fixed seat on the display: the eye captures a stable
            # region, and the window opens already raised like a game a
            # player just launched.
            self._root.geometry(f"+{_ORIGIN[0]}+{_ORIGIN[1]}")
            self._root.update()
            self._root.lift()
            self._win = self._root.winfo_id()
            self._xdo = xdo.xdo()
            try:
                subprocess.run(
                    ["xdotool", "windowactivate", str(self._win)],
                    timeout=2,
                    check=False,
                    env=self._xenv(),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("[cabinet] windowactivate failed: %s", e)
        finally:
            if display is not None:
                if prev is None:
                    os.environ.pop("DISPLAY", None)
                else:
                    os.environ["DISPLAY"] = prev
        self._timeout = max(0.2, float(timeout_s))
        self._photo: Any = None
        self._zoomed: Any = None

    def _xenv(self) -> dict[str, str] | None:
        """Subprocess env carrying the cabinet's own display, if any."""
        if self._display is None:
            return None
        return {**os.environ, "DISPLAY": self._display}

    def _on_key(self, event: Any) -> None:
        """Tk callback: feed each physical keypress into the decoder.

        Completed controls queue on ``_arrived`` for ``dispatch`` to
        collect — what the window actually received, not what was
        asked for.
        """
        token = self._dec.feed_key(getattr(event, "keysym", ""))
        if token is not None:
            self._arrived.append(token)

    def _on_click(self, event: Any) -> None:
        """Tk callback: a board click is ACTION6 at the clicked cell."""
        token = self._dec.feed_click(getattr(event, "x", -1), getattr(event, "y", -1))
        if token is not None:
            self._arrived.append(token)

    def _on_reset(self) -> None:
        """The RESET control fired — a button click is a reset."""
        self._arrived.append((0, None))

    def _click_at(self, sx: int, sy: int) -> None:
        """Move the real pointer to screen (sx, sy) and press it.

        xdotool's XTEST events are indistinguishable from a mouse to
        the window — the click lands on whatever glass is under the
        point, exactly like a hand moving a physical mouse.
        """
        subprocess.run(
            ["xdotool", "mousemove", str(sx), str(sy), "click", "1"],
            timeout=2,
            check=True,
            env=self._xenv(),
        )

    @property
    def display(self) -> str | None:
        """The X display the cabinet lives on (None = inherited)."""
        return self._display

    def board_capture(self) -> tuple[int, int, int]:
        """The eye's capture descriptor: ``(window_id, w, h)``.

        The canvas is the window's top child, so a window-id grab
        cropped to its size sees exactly the board — no screen
        geometry, no titlebar, wherever the WM puts the window.
        """
        self._root.update()
        return self._win, self._canvas.winfo_width(), self._canvas.winfo_height()

    def dispatch(
        self, action: Any, data: dict[str, int] | None
    ) -> tuple[Any, dict[str, int] | None] | None:
        """Emit the action's input events; return what arrived."""
        from arcengine import GameAction

        action_id = getattr(action, "value", action)
        if isinstance(action_id, tuple):
            action_id = action_id[0]
        action_id = int(action_id)
        self._arrived.clear()
        self._root.update()
        for kind, *rest in _controls_for(action_id, data):
            try:
                if kind == "key":
                    self._xdo.send_keysequence_window(rest[0], window=self._win)
                elif kind == "click":
                    cx, cy = rest
                    self._click_at(
                        self._canvas.winfo_rootx() + int((cx + 0.5) * _SCALE),
                        self._canvas.winfo_rooty() + int((cy + 0.5) * _SCALE),
                    )
                else:  # reset — click the RESET control
                    self._click_at(
                        self._reset_btn.winfo_rootx() + self._reset_btn.winfo_width() // 2,
                        self._reset_btn.winfo_rooty() + self._reset_btn.winfo_height() // 2,
                    )
            except Exception as e:  # noqa: BLE001
                logger.info("[cabinet] input emission failed: %s", e)
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
        """Destroy the window; a dead tk root is already gone."""
        try:
            self._root.destroy()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"close failed: {e}")

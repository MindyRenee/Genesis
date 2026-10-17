"""Windowed games — installed programs she plays on the desktop.

This is the "games on the system computer" path: a real program
(``/usr/games/pacman``, a solitaire, eventually any windowed game)
running on the X display she shares with the user. She perceives it
the way a player does — watching the window — and acts the way a
player does — pressing keys.

Pieces, all substrate, no rules baked in:

- ``X11Game`` — the environment. Spawns the process, finds its
  window, captures frames (``import -window``), quantizes pixels to
  the parietal ``Grid``, injects keys through the XTEST extension.
- Score is whatever the window itself reports (title text), else
  survival: staying alive IS the objective of an arcade game, and
  the honest verdict the world gives her.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import re
import subprocess
import time
from typing import Any

from parietal_lobe.grid import Grid

from genesis_client.swallow import note_swallowed

from .environment import GameObservation, GameOutcome

logger = logging.getLogger(__name__)

_libX11 = None
_libXtst = None


def _x11() -> Any:
    """Lazily open the X display — unavailable off a desktop."""
    global _libX11
    if _libX11 is None:
        _libX11 = ctypes.CDLL("libX11.so.6")
        _libX11.XOpenDisplay.restype = ctypes.c_void_p
        _libX11.XKeysymToKeycode.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong]
        _libX11.XKeysymToKeycode.restype = ctypes.c_ubyte
        _libX11.XFlush.argtypes = [ctypes.c_void_p]
        _libX11.XRaiseWindow.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong]
        _libX11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _libX11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    return _libX11


def _xtst() -> Any:
    global _libXtst
    if _libXtst is None:
        _libXtst = ctypes.CDLL("libXtst.so.6")
        _libXtst.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
            ctypes.c_ulong]
    return _libXtst


class _Display:
    """A shared X11 display handle for one play session."""

    def __init__(self) -> None:
        lib = _x11()
        self.lib = lib
        lib.XSetInputFocus.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_ulong]
        lib.XGetGeometry.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        lib.XGetGeometry.restype = ctypes.c_int
        self.handle = lib.XOpenDisplay(None)
        if not self.handle:
            raise OSError("cannot open X display — no desktop?")

    def keycode(self, keysym: int) -> int:
        return int(self.lib.XKeysymToKeycode(self.handle, keysym))

    def press(self, keysym: int, hold: float = 0.05) -> None:
        """Press a key for ``hold`` seconds.

        Arcade games steer while a direction key is DOWN — a tap does
        nothing. ``hold`` is the action's duration: how long the key
        stays pressed before release.
        """
        code = self.keycode(keysym)
        _xtst().XTestFakeKeyEvent(self.handle, code, 1, 0)
        self.lib.XFlush(self.handle)
        if hold > 0.0:
            time.sleep(hold)
        _xtst().XTestFakeKeyEvent(self.handle, code, 0, 0)
        self.lib.XFlush(self.handle)

    def focus(self, wid: int, x: int = 0, y: int = 0) -> None:
        """Give the game real input focus — the difference between
        pressing keys AT a window and playing ON it. RevertToParent
        hands focus back when the window dies."""
        self.lib.XRaiseWindow(self.handle, wid)
        self.lib.XSetInputFocus(
            self.handle, wid, 1, 0  # RevertToParent, CurrentTime
        )
        self.lib.XSync(self.handle, 0)

    def window_size(self, wid: int) -> tuple[int, int] | None:
        root = ctypes.c_ulong(0)
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        w = ctypes.c_uint(0)
        h = ctypes.c_uint(0)
        bw = ctypes.c_uint(0)
        d = ctypes.c_uint(0)
        ok = self.lib.XGetGeometry(
            self.handle, wid,
            ctypes.byref(root), ctypes.byref(x), ctypes.byref(y),
            ctypes.byref(w), ctypes.byref(h),
            ctypes.byref(bw), ctypes.byref(d))
        if not ok:
            return None
        return int(w.value), int(h.value)

    def close(self) -> None:
        try:
            self.lib.XCloseDisplay(self.handle)
        except Exception as e:  # noqa: BLE001
            note_swallowed("frontal_lobe.games.windowed.xclose", e)
        self.handle = None


# Keysym constants (X11/keysymdef.h) for the action vocabulary.
_KEYSYMS: dict[str, int] = {
    "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53,
    "space": 0x20, "enter": 0xFF0D, "escape": 0xFF1B,
    "w": 0x77, "a": 0x61, "s": 0x73, "d": 0x64,
    "q": 0x71, "r": 0x72, "p": 0x70,
}


def _tree_windows() -> dict[int, tuple[int, int, str]]:
    """Every toplevel window: {id: (w, h, title-or-"")}.

    Uses the raw X tree (``xwininfo -root -children``), not the EWMH
    client list — bare X games like pacman4console never set
    _NET_WM_PID or WM_NAME, so ``wmctrl -l`` can't see them at all.
    """
    try:
        out = subprocess.run(
            ["xwininfo", "-root", "-children"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    wins: dict[int, tuple[int, int, str]] = {}
    for line in out.splitlines():
        m = re.match(
            r"\s*(0x[0-9a-fA-F]+)\s+(?:\"([^\"]*)\")?\s*.*?"
            r"(\d+)x(\d+)[+-]",
            line)
        if m:
            wid = int(m.group(1), 16)
            title = m.group(2) or ""
            wins[wid] = (int(m.group(3)), int(m.group(4)), title)
    return wins


class X11Game:
    """An installed windowed program as a GameEnvironment.

    Launches ``command``, watches the desktop for the new window,
    then plays it: frames → Grid via cell quantization, keys via
    XTEST. ``cell_px`` sets the spatial resolution of her board
    parse; ``palette`` maps RGB→symbol classes when a game ships a
    known tile palette, else classes are derived per frame.
    """

    name = "x11"

    def __init__(
        self,
        command: list[str],
        *,
        title_pattern: str = "",
        actions: list[str] | None = None,
        cell_px: int = 12,
        frame_pause: float = 0.15,
        key_hold: float = 0.35,
        start_keys: list[str] | None = None,
        observe: dict | None = None,
    ) -> None:
        self.command = list(command)
        self.title_pattern = title_pattern
        self._actions = actions or ["up", "down", "left", "right"]
        self.cell_px = cell_px
        self.frame_pause = frame_pause
        # How long each action's key stays down — arcade games steer
        # while the key is held, so a move IS a duration.
        self.key_hold = key_hold
        self.start_keys = start_keys or []

        # What she was told to watch — the briefing made perceptual.
        # ``observe`` is spec data, never compiled rules:
        #   meter:   {"region",[x,y,w,h] "color",[r,g,b] "tol",n
        #             "direction","down|up" "scale",f}
        #           a pixel count whose movement IS the objective —
        #           pacman's yellow dots, solitaire's face-down pile.
        #   regions: [{"box",[x,y,w,h] "reward",f "name",str}]
        #           displays to watch — when one changes, its reward
        #           applies (a life lost, the score rolling over).
        #   danger:  {"player_color",[r,g,b] "danger_colors",[[r,g,b]…]
        #             "near_px",n "tol",n}
        #           threat imminence — how close the dangerous
        #           sprites are to the player's, in pixels. The
        #           predatory-imminence gradient, computed.
        obs_cfg = observe or {}
        self._meter = obs_cfg.get("meter")
        self._watch = list(obs_cfg.get("regions") or [])
        self._danger = obs_cfg.get("danger")
        self._meter_last: float | None = None
        self._meter0: float | None = None
        self._watch_last: list[bytes | None] = [None] * len(self._watch)
        self.deaths = 0
        self._progress = 0.0

        self._proc: subprocess.Popen | None = None
        self._wid: int | None = None
        self._display: _Display | None = None
        self._ticks = 0
        self._baseline_windows: set[int] = set()

    # ── lifecycle ────────────────────────────────────────────

    def reset(self) -> GameObservation:
        if self._display is None:
            self._display = _Display()
        # Re-attach to the window she left up, if it's still there —
        # the game stays on the desktop between sessions.
        if self._wid is not None and self._wid in _tree_windows():
            self._display.focus(self._wid)
            self._ticks = 0
            for key in self.start_keys:
                self._send(key)
                time.sleep(0.1)
            return self._observe_reset()
        self._wid = None
        self._baseline_windows = set(_tree_windows())
        self._proc = subprocess.Popen(
            self.command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # Detached session: the game is designed to stay on the
            # desktop between sessions (reset() re-attaches to the
            # window it left up). Without setsid it stays in the
            # service cgroup and systemd's shutdown kill reaps it —
            # the "persists between sessions" contract only held
            # while nothing restarted the unit.
            start_new_session=True,
        )
        self._wid = self._find_window(timeout=10.0)
        if self._wid is None:
            self.close(kill=True)
            raise OSError(f"{self.command[0]}: no window appeared")
        self._display.focus(self._wid)
        self._ticks = 0
        for key in self.start_keys:
            self._send(key)
            time.sleep(0.1)
        return self._observe_reset()

    def _observe_reset(self) -> GameObservation:
        """First observation of a session — baselines the objective
        meter and watched displays so the first step's delta counts."""
        obs = self.observe()
        if obs.raw is not None:
            if self._meter:
                n = self._meter_count(obs.raw)
                if self._meter0 is None or n > self._meter0:
                    self._meter0 = n
                self._meter_last = n
            self._watch_last = [
                self._region_sig(obs.raw, reg["box"])
                for reg in self._watch
            ]
        self.deaths = 0
        self._progress = 0.0
        return obs

    def _find_window(self, timeout: float) -> int | None:
        """Wait for a new window owned by (or titled like) the game.

        Matches by title when the game sets one; otherwise takes the
        biggest new toplevel window — bare X games leave no other
        trace to match on.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            tree = _tree_windows()
            new_wins = {
                w: geo for w, geo in tree.items()
                if w not in self._baseline_windows
            }
            if self.title_pattern:
                for wid, (_w, _h, title) in new_wins.items():
                    if re.search(self.title_pattern, title, re.I):
                        return wid
            if new_wins:
                # The game's window: largest new toplevel (a real
                # game window is always bigger than a decoration).
                wid = max(
                    new_wins, key=lambda w: new_wins[w][0] * new_wins[w][1]
                )
                if new_wins[wid][0] * new_wins[wid][1] >= 40 * 30:
                    return wid
            if self._proc is not None and self._proc.poll() is not None:
                return None  # exited without showing a window
            time.sleep(0.2)
        return None

    def close(self, kill: bool = False) -> None:
        """End her session — by default the game stays UP.

        She walks away from the game, she doesn't quit it: the window
        stays on the desktop so someone can watch what she was
        playing and so she can come back to it. ``kill=True`` is the
        explicit "shut it down" path (launch failure, cleanup).
        """
        if kill and self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    self._proc.kill()
                except OSError as e:
                    note_swallowed(
                        "frontal_lobe.games.windowed.proc_kill", e
                    )
        if self._display is not None:
            self._display.close()
            self._display = None
        if kill:
            self._proc = None
            self._wid = None

    # ── GameEnvironment ──────────────────────────────────────

    def observe(self) -> GameObservation:
        frame = self._capture()
        grid = self._to_grid(frame) if frame is not None else None
        return GameObservation(
            grid=grid, text=self._title(), raw=frame, tick=self._ticks
        )

    def actions(self) -> list[str]:
        return list(self._actions)

    def act(self, action: str) -> GameOutcome:
        self._send(action)
        time.sleep(self.frame_pause)
        self._ticks += 1
        obs = self.observe()
        terminal = self.done()
        frame = obs.raw
        reward = -0.5 if terminal else 0.01
        died = False

        # The objective meter — dots eaten, cards cleared, whatever
        # the spec told her to watch. Its delta is the honest,
        # graded part of the verdict.
        if self._meter and frame is not None:
            n = self._meter_count(frame)
            if self._meter_last is not None:
                delta = (
                    self._meter_last - n
                    if self._meter.get("direction", "down") == "down"
                    else n - self._meter_last
                )
                reward += max(
                    -0.5, min(1.0, delta * float(self._meter.get(
                        "scale", 0.02))))
                self._progress += max(0.0, delta)
            self._meter_last = n

        # Watched displays — a changed readout applies its declared
        # reward. A life marker moving is a death; the scoreboard
        # rolling over is the game acknowledging her.
        if frame is not None:
            for i, reg in enumerate(self._watch):
                sig = self._region_sig(frame, reg["box"])
                last = self._watch_last[i]
                self._watch_last[i] = sig
                if last is not None and sig != last:
                    reward += float(reg.get("reward", 0.0))
                    if str(reg.get("name", "")) == "lives":
                        died = True
                        self.deaths += 1

        threat = self._threat_from(frame) if frame is not None else 0.0
        return GameOutcome(
            observation=obs,
            reward=max(-1.5, min(1.5, reward)),
            terminal=terminal,
            info={
                "threat": threat,
                "death": died or (terminal and reward < 0),
                "won": bool(terminal and self._progress > 0
                            and not died and self.deaths == 0),
            },
        )

    def done(self) -> bool:
        if self._proc is not None and self._proc.poll() is not None:
            return True
        if self._wid is not None:
            return self._wid not in _tree_windows()
        return True

    def score(self) -> float:
        """How the world says she's doing: objective progress — the
        fraction of the meter she's moved (dots cleared of the board),
        or survival ticks when no meter was specified."""
        if self._meter and self._meter0:
            return self._progress / self._meter0
        return float(self._ticks)

    def _meter_count(self, frame: Any) -> float:
        """Pixels matching the meter color inside its region."""
        import numpy as np

        meter = self._meter or {}
        x, y, w, h = (int(v) for v in meter["region"])
        color = np.asarray(meter["color"], dtype=np.int32)
        tol = int(meter.get("tol", 60)) * 3
        arr = np.asarray(frame, dtype=np.int32)[y:y + h, x:x + w]
        return float((np.abs(arr - color).sum(axis=2) < tol).sum())

    def _region_sig(self, frame: Any, box: list) -> bytes:
        """A small signature of one display region — enough to know
        the readout moved, without reading what it says."""
        x, y, w, h = (int(v) for v in box)
        crop = frame.crop((x, y, x + w, y + h)).resize(
            (max(1, w // 4), max(1, h // 4)))
        return crop.tobytes()

    def _threat_from(self, frame: Any) -> float:
        """Danger imminence, 0..1 — how close the dangerous sprites
        are to the player's, scaled by ``near_px``.

        The player is found as the *densest* patch of player color —
        pacman is a fat yellow blob among sparse yellow dots, so
        density separates her sprite from the objective color even
        when they share it. Fanselow's imminence continuum: this is
        the perceptual gradient the defensive response rides on
        (Mobbs et al., Science 2007 — a predator chasing you through
        a maze, literally this game).
        """
        if not self._danger:
            return 0.0
        import numpy as np

        arr = np.asarray(frame, dtype=np.int32)
        tol = int(self._danger.get("tol", 70)) * 3

        pc = np.asarray(self._danger["player_color"], dtype=np.int32)
        mask = np.abs(arr - pc).sum(axis=2) < tol
        ys, xs = np.nonzero(mask)
        if len(ys) < 20:
            return 0.0
        # Densest 16px block of player color = her sprite.
        key = (ys // 16) * 4096 + (xs // 16)
        blocks, counts = np.unique(key, return_counts=True)
        top = blocks[counts >= counts.max() * 0.5]
        keep = np.isin(key, top)
        cy, cx = float(ys[keep].mean()), float(xs[keep].mean())

        best = 1e9
        for dc in self._danger["danger_colors"]:
            d = np.abs(arr - np.asarray(dc, dtype=np.int32)).sum(axis=2)
            dys, dxs = np.nonzero(d < tol)
            if len(dys) == 0:
                continue
            dist = np.sqrt(
                (dys - cy) ** 2 + (dxs - cx) ** 2).min()
            best = min(best, float(dist))
        if best == 1e9:
            return 0.0
        near = float(self._danger.get("near_px", 90))
        return max(0.0, min(1.0, 1.0 - best / near))

    def _title(self) -> str:
        if self._wid is None:
            return ""
        geo = _tree_windows().get(self._wid)
        return geo[2] if geo else ""

    # ── perception ───────────────────────────────────────────

    def _capture(self) -> Any:
        """One frame of the window as a PIL image, or None."""
        if self._wid is None:
            return None
        try:
            out = subprocess.run(
                ["import", "-window", str(self._wid), "png:-"],
                capture_output=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if not out.stdout:
            return None
        try:
            import io

            from PIL import Image

            return Image.open(io.BytesIO(out.stdout)).convert("RGB")
        except Exception:  # noqa: BLE001 — a corrupt frame is a miss, not a crash
            return None

    def _to_grid(self, image: Any) -> Grid:
        """Quantize the frame into a color-class Grid.

        Each ``cell_px`` square becomes the mean RGB of its pixels,
        snapped to a small palette learned per frame (the game's own
        color classes — walls, pellets, sprites). The grid is her
        parietal board representation: the same datatype the built-in
        games speak.
        """
        import numpy as np

        arr = np.asarray(image, dtype=np.float32)
        h, w = arr.shape[:2]
        cs = self.cell_px
        rows = h // cs
        cols = w // cs
        if rows == 0 or cols == 0:
            return Grid.from_lists([[0]])
        cells = arr[: rows * cs, : cols * cs].reshape(
            rows, cs, cols, cs, 3
        ).mean(axis=(1, 3))  # (rows, cols, 3)

        # Palette = k-means-lite over cell means: seed with the most
        # frequent quantized colors, two refinement passes. Bounded —
        # the per-frame budget stays tight.
        flat = cells.reshape(-1, 3)
        keys = (flat // 48).astype(np.int32)
        uniq, counts = np.unique(keys, axis=0, return_counts=True)
        order = np.argsort(-counts)[:8]
        seeds = [
            flat[(keys == uniq[i]).all(axis=1)].mean(axis=0)
            for i in order
        ]
        centers = np.stack(
            [c for c in seeds if not np.isnan(c).any()]
        )
        if centers.size == 0:
            centers = np.array([[0.0, 0.0, 0.0]])
        for _ in range(2):
            d = ((flat[:, None, :] - centers[None]) ** 2).sum(axis=2)
            assign = d.argmin(axis=1)
            for k in range(centers.shape[0]):
                pts = flat[assign == k]
                if len(pts):
                    centers[k] = pts.mean(axis=0)
        board = assign.reshape(rows, cols)
        return Grid.from_lists(board.tolist())

    # ── input ────────────────────────────────────────────────

    def _send(self, action: str) -> None:
        if self._display is None:
            return
        keysym = _KEYSYMS.get(action)
        if keysym is None:
            if len(action) == 1:
                keysym = ord(action)
            else:
                return
        try:
            self._display.press(keysym, hold=self.key_hold)
        except (OSError, ctypes.ArgumentError) as e:
            logger.debug(f"key {action}: {e}")

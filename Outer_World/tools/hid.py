"""HID — her hand on the machine's input bus.

A uinput virtual device is a real input device: the kernel merges its
events with the hardware's before any userspace layer sees them, so
windows, terminals and games cannot tell her keystrokes from a
person's. It works under X11, Wayland, and the bare console alike —
unlike XTest-level fakes, which live and die with the X server and
which applications are free to filter out.

That is what makes it an outer-world tool rather than a test double:
when she types, the world hears a keyboard.

The devices are created lazily, on her first reach — uinput needs no
capability until then. The ``evdev`` binding is an optional
dependency in the same sense as her voice: this module imports it
lazily and raises an honest error when it or ``/dev/uinput`` is not
there to take her hand.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import time
from typing import Any

from genesis_client.swallow import note_swallowed
from inhibition import InhibitoryGate

__all__ = [
    "Hands",
    "char_key",
    "hands",
    "probe",
    "resolve_key_name",
]

_UINPUT_NODE = "/dev/uinput"

#: Upper end of the absolute-axis range. The display server scales a
#: device's abs span onto the desktop extent, so pixel coordinates map
#: through the screen size the same way a graphics tablet's do.
_ABS_MAX = 65535

#: Beat between key-down and key-up. A zero-length press is legal but
#: some consumers lose it — the gap is what a finger does anyway.
_PRESS_GAP = 0.02

#: Beat after device creation. The desktop takes a moment to adopt a
#: new input device; events sent inside the adoption window land on
#: nobody.
_SETTLE = 0.5

# Friendly names → kernel key names. Anything a caller writes
# naturally resolves through here first; an unrecognized name falls
# back to the literal ``KEY_*`` form, so "f5" and "KEY_F5" both work.
_ALIASES = {
    "enter": "KEY_ENTER",
    "return": "KEY_ENTER",
    "esc": "KEY_ESC",
    "escape": "KEY_ESC",
    "space": "KEY_SPACE",
    "spacebar": "KEY_SPACE",
    "del": "KEY_DELETE",
    "ins": "KEY_INSERT",
    "pgup": "KEY_PAGEUP",
    "pageup": "KEY_PAGEUP",
    "prior": "KEY_PAGEUP",
    "pgdn": "KEY_PAGEDOWN",
    "pagedown": "KEY_PAGEDOWN",
    "next": "KEY_PAGEDOWN",
    "ctrl": "KEY_LEFTCTRL",
    "control": "KEY_LEFTCTRL",
    "ctl": "KEY_LEFTCTRL",
    "shift": "KEY_LEFTSHIFT",
    "alt": "KEY_LEFTALT",
    "option": "KEY_LEFTALT",
    "super": "KEY_LEFTMETA",
    "win": "KEY_LEFTMETA",
    "cmd": "KEY_LEFTMETA",
    "command": "KEY_LEFTMETA",
    "meta": "KEY_LEFTMETA",
    "caps": "KEY_CAPSLOCK",
    "capslock": "KEY_CAPSLOCK",
    "numlock": "KEY_NUMLOCK",
    "scrolllock": "KEY_SCROLLLOCK",
    "printscreen": "KEY_SYSRQ",
    "prtsc": "KEY_SYSRQ",
    "sysrq": "KEY_SYSRQ",
    "pause": "KEY_PAUSE",
    "break": "KEY_PAUSE",
    "menu": "KEY_MENU",
    "up": "KEY_UP",
    "down": "KEY_DOWN",
    "left": "KEY_LEFT",
    "right": "KEY_RIGHT",
    "volumeup": "KEY_VOLUMEUP",
    "volumedown": "KEY_VOLUMEDOWN",
    "mute": "KEY_MUTE",
    "playpause": "KEY_PLAYPAUSE",
}

# Printable characters with an unshifted key on a US layout. Letters
# and digits are derived programmatically in ``char_key``.
_PLAIN_KEYS = {
    " ": "KEY_SPACE",
    "\n": "KEY_ENTER",
    "\t": "KEY_TAB",
    "-": "KEY_MINUS",
    "=": "KEY_EQUAL",
    "[": "KEY_LEFTBRACE",
    "]": "KEY_RIGHTBRACE",
    ";": "KEY_SEMICOLON",
    "'": "KEY_APOSTROPHE",
    "`": "KEY_GRAVE",
    "\\": "KEY_BACKSLASH",
    ",": "KEY_COMMA",
    ".": "KEY_DOT",
    "/": "KEY_SLASH",
}

# Printable characters reached with shift held on a US layout.
_SHIFTED_KEYS = {
    "!": "KEY_1",
    "@": "KEY_2",
    "#": "KEY_3",
    "$": "KEY_4",
    "%": "KEY_5",
    "^": "KEY_6",
    "&": "KEY_7",
    "*": "KEY_8",
    "(": "KEY_9",
    ")": "KEY_0",
    "_": "KEY_MINUS",
    "+": "KEY_EQUAL",
    "{": "KEY_LEFTBRACE",
    "}": "KEY_RIGHTBRACE",
    ":": "KEY_SEMICOLON",
    '"': "KEY_APOSTROPHE",
    "~": "KEY_GRAVE",
    "|": "KEY_BACKSLASH",
    "<": "KEY_COMMA",
    ">": "KEY_DOT",
    "?": "KEY_SLASH",
}

_BUTTONS = {
    "left": "BTN_LEFT",
    "right": "BTN_RIGHT",
    "middle": "BTN_MIDDLE",
    "side": "BTN_SIDE",
    "back": "BTN_SIDE",
    "extra": "BTN_EXTRA",
    "forward": "BTN_EXTRA",
}


def char_key(ch: str) -> tuple[str, bool]:
    """Map one character to ``(key name, needs shift)`` on a US layout.

    Raises ``ValueError`` for characters with no US-layout key — the
    caller decides whether to substitute or report; a half-typed
    string pretending success is worse than an honest refusal.
    """
    if "a" <= ch <= "z":
        return f"KEY_{ch.upper()}", False
    if "A" <= ch <= "Z":
        return f"KEY_{ch}", True
    if "0" <= ch <= "9":
        return f"KEY_{ch}", False
    if ch in _PLAIN_KEYS:
        return _PLAIN_KEYS[ch], False
    if ch in _SHIFTED_KEYS:
        return _SHIFTED_KEYS[ch], True
    raise ValueError(f"no US-layout key for character {ch!r}")


def resolve_key_name(name: str) -> tuple[str, bool]:
    """Normalize a caller's key name to ``(key name, needs shift)``.

    Accepts friendly aliases ("enter", "ctrl"), kernel names
    ("KEY_F5"), and single characters routed through ``char_key``
    (so "A" honestly means shift+a). Anything else becomes
    ``KEY_<UPPER>`` — validity is decided by the device layer, where
    the ecodes table lives.
    """
    n = (name or "").strip()
    if not n:
        raise ValueError("empty key name")
    if len(n) == 1:
        return char_key(n)
    low = re.sub(r"[\s_\-]+", "", n.lower())
    if low in _ALIASES:
        return _ALIASES[low], False
    if low.startswith("f") and low[1:].isdigit() and 1 <= int(low[1:]) <= 24:
        return f"KEY_{low.upper()}", False
    upper = n.upper()
    if upper.startswith("KEY_"):
        return upper, False
    return f"KEY_{upper}", False


def _screen_size() -> tuple[int, int] | None:
    """Desktop extent in pixels, or None when no display answers.

    Absolute pointer placement is scaled through the display server's
    mapping: the pointer's abs axes span [0, _ABS_MAX] and get mapped
    onto the whole desktop, so pixel coordinates go through the
    screen's own width and height.
    """
    if os.environ.get("DISPLAY"):
        for cmd, pattern in (
            (["xrandr", "--current"], r"current (\d+) x (\d+)"),
            (["xdpyinfo"], r"dimensions:\s+(\d+)x(\d+)"),
        ):
            exe = shutil.which(cmd[0])
            if exe is None:
                continue
            try:
                out = subprocess.run(
                    [exe, *cmd[1:]],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            m = re.search(pattern, out.stdout)
            if m:
                return int(m.group(1)), int(m.group(2))
    # No display server — the framebuffer's declared size still
    # describes the extent a console session would draw on.
    try:
        with open("/sys/class/graphics/fb0/virtual_size") as fh:
            w, h = fh.read().strip().split(",")
        return int(w), int(h)
    except (OSError, ValueError):
        return None


def probe() -> dict[str, Any]:
    """Readiness report — binding, device node, permission, screen.

    Never opens a device and never raises: probing is sensing, and a
    hand that is not there is a finding, not a fault.
    """
    evdev_ok = importlib.util.find_spec("evdev") is not None
    node = os.path.exists(_UINPUT_NODE)
    writable = node and os.access(_UINPUT_NODE, os.W_OK)
    return {
        "backend": "linux-uinput",
        "evdev_installed": evdev_ok,
        "uinput_node": node,
        "uinput_writable": writable,
        "screen": _screen_size(),
        "ready": evdev_ok and writable,
    }


class Hands:
    """A virtual keyboard and pointer on ``/dev/uinput``.

    Two devices because the kernel reads them as two: a keyboard
    advertises keys, a pointer advertises buttons plus relative and
    absolute axes. Separate devices also read honestly in
    ``xinput list`` — she can see her own hands.

    All writes serialize through one lock: a keystroke is a burst of
    events and two acts interleaving mid-burst would be neither.
    """

    def __init__(self) -> None:
        """Open both devices. Caller guarantees ``evdev`` imports."""
        from evdev import AbsInfo, UInput, ecodes

        self._E = ecodes
        self._codes: dict[str, int] = ecodes.ecodes
        self._lock = InhibitoryGate()
        self._kbd = UInput(
            {ecodes.EV_KEY: sorted(ecodes.keys)},
            name="genesis-hand-keys",
            phys="genesis/hand",
        )
        self._ptr = UInput(
            {
                ecodes.EV_KEY: [
                    ecodes.BTN_LEFT,
                    ecodes.BTN_MIDDLE,
                    ecodes.BTN_RIGHT,
                    ecodes.BTN_SIDE,
                    ecodes.BTN_EXTRA,
                ],
                ecodes.EV_REL: [
                    ecodes.REL_X,
                    ecodes.REL_Y,
                    ecodes.REL_WHEEL,
                    ecodes.REL_HWHEEL,
                ],
                ecodes.EV_ABS: [
                    (ecodes.ABS_X, AbsInfo(0, 0, _ABS_MAX, 0, 0, 0)),
                    (ecodes.ABS_Y, AbsInfo(0, 0, _ABS_MAX, 0, 0, 0)),
                ],
            },
            name="genesis-hand-pointer",
            phys="genesis/hand",
        )
        time.sleep(_SETTLE)
        self._screen = _screen_size()

    def _code(self, name: str) -> int:
        """Kernel code for a key/button name, or a clean refusal."""
        try:
            return self._codes[name]
        except KeyError:
            raise ValueError(f"unknown key {name!r}") from None

    def press(self, key: str, modifiers: tuple[str, ...] = ()) -> str:
        """One keystroke — modifiers down, key down-up, modifiers up.

        Returns the resolved kernel key name. ``modifiers`` are held in
        the order given and released in reverse, the way a hand does it.
        """
        name, shift = resolve_key_name(key)
        mods = [self._code(resolve_key_name(m)[0]) for m in modifiers]
        if shift:
            mods.insert(0, self._code("KEY_LEFTSHIFT"))
        with self._lock:
            for m in mods:
                self._kbd.write(self._E.EV_KEY, m, 1)
            self._kbd.syn()
            time.sleep(_PRESS_GAP)
            code = self._code(name)
            self._kbd.write(self._E.EV_KEY, code, 1)
            self._kbd.syn()
            time.sleep(_PRESS_GAP)
            self._kbd.write(self._E.EV_KEY, code, 0)
            for m in reversed(mods):
                self._kbd.write(self._E.EV_KEY, m, 0)
            self._kbd.syn()
        return name

    def type(self, text: str) -> int:
        """Type each character of ``text``. Returns the character count.

        Every character is resolved before the first event goes out, so
        an unmappable character fails cleanly instead of leaving a
        half-typed string in the world.
        """
        strokes = [char_key(c) for c in text]
        shift = self._code("KEY_LEFTSHIFT")
        with self._lock:
            for name, need_shift in strokes:
                code = self._code(name)
                if need_shift:
                    self._kbd.write(self._E.EV_KEY, shift, 1)
                self._kbd.write(self._E.EV_KEY, code, 1)
                self._kbd.syn()
                time.sleep(_PRESS_GAP)
                self._kbd.write(self._E.EV_KEY, code, 0)
                if need_shift:
                    self._kbd.write(self._E.EV_KEY, shift, 0)
                self._kbd.syn()
        return len(text)

    def move(self, dx: int, dy: int) -> None:
        """Relative pointer motion — dx right, dy down, in pixels."""
        with self._lock:
            if dx:
                self._ptr.write(self._E.EV_REL, self._E.REL_X, dx)
            if dy:
                self._ptr.write(self._E.EV_REL, self._E.REL_Y, dy)
            self._ptr.syn()

    def goto(self, x: int, y: int) -> tuple[int, int]:
        """Absolute pointer placement at desktop pixel (x, y).

        The pointer's abs axes are scaled onto the desktop by the
        display server, so pixel coordinates map through the screen
        size learned at device creation.
        """
        if self._screen is None:
            raise RuntimeError("screen size unknown — cannot place the pointer absolutely")
        w, h = self._screen
        ax = min(_ABS_MAX, max(0, round(x * _ABS_MAX / max(1, w - 1))))
        ay = min(_ABS_MAX, max(0, round(y * _ABS_MAX / max(1, h - 1))))
        with self._lock:
            self._ptr.write(self._E.EV_ABS, self._E.ABS_X, ax)
            self._ptr.write(self._E.EV_ABS, self._E.ABS_Y, ay)
            self._ptr.syn()
        return (x, y)

    def click(self, button: str = "left", x: int = -1, y: int = -1) -> str:
        """One button press. Negative coordinates mean "right here"."""
        bname = _BUTTONS.get(button.lower())
        if bname is None:
            raise ValueError(f"unknown button {button!r}")
        if x >= 0 and y >= 0:
            self.goto(x, y)
        code = self._code(bname)
        with self._lock:
            self._ptr.write(self._E.EV_KEY, code, 1)
            self._ptr.syn()
            time.sleep(_PRESS_GAP)
            self._ptr.write(self._E.EV_KEY, code, 0)
            self._ptr.syn()
        return bname

    def scroll(self, dy: int, dx: int = 0) -> None:
        """Wheel notches — dy positive scrolls up, dx is horizontal."""
        with self._lock:
            if dy:
                self._ptr.write(self._E.EV_REL, self._E.REL_WHEEL, dy)
            if dx:
                self._ptr.write(self._E.EV_REL, self._E.REL_HWHEEL, dx)
            self._ptr.syn()

    def close(self) -> None:
        """Destroy both devices — the hand leaves the input bus."""
        for dev in (self._kbd, self._ptr):
            try:
                dev.close()
            except Exception as e:  # noqa: BLE001 — already gone is fine
                note_swallowed("tools.hid.device_close", e)


_HANDS: Hands | None = None


def _open_hands() -> Hands:
    """First reach — verify the binding and the bus, then open devices.

    Raises ``RuntimeError`` with the specific missing piece: no
    ``evdev``, no ``/dev/uinput`` node, or no permission to write it.
    A hand behind glass is reported, not faked.
    """
    if importlib.util.find_spec("evdev") is None:
        raise RuntimeError(
            "the 'evdev' package is not installed — scripts/install_optional_deps.sh sets it up"
        )
    if not os.path.exists(_UINPUT_NODE):
        raise RuntimeError("/dev/uinput is absent — 'sudo modprobe uinput' loads the kernel module")
    if not os.access(_UINPUT_NODE, os.W_OK):
        raise RuntimeError(
            "/dev/uinput is not writable by her — the uinput udev rule "
            "in scripts/genesis-uinput.rules (installed by scripts/"
            "install_optional_deps.sh) grants it without root"
        )
    try:
        return Hands()
    except ImportError as e:
        raise RuntimeError(f"evdev is present but does not import: {e}") from e
    except PermissionError as e:
        raise RuntimeError(f"uinput refused the device: {e}") from e


def hands() -> Hands:
    """Her hands — created on first reach, kept for the session."""
    global _HANDS
    if _HANDS is None:
        _HANDS = _open_hands()
    return _HANDS

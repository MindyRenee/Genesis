"""Tests for the virtual HID — her hand on the machine's input bus.

The contract under test: the name layer resolves what a caller would
naturally write, the character map covers printable ASCII on a US
layout, and every tool call degrades to an honest error when no
device can be opened — never a crash, never a silent drop. The live
device test is gated: it only runs where a writable /dev/uinput and
the evdev binding actually exist.
"""

import os

import pytest
from tools import framework, hid


class FakeHands:
    """Records what would have been written to the devices."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def press(self, key: str, modifiers: tuple = ()) -> str:
        self.calls.append(("press", key, tuple(modifiers)))
        return "KEY_" + key.upper()

    def type(self, text: str) -> int:
        self.calls.append(("type", text))
        return len(text)

    def move(self, dx: int, dy: int) -> None:
        self.calls.append(("move", dx, dy))

    def goto(self, x: int, y: int) -> tuple:
        self.calls.append(("goto", x, y))
        return (x, y)

    def click(self, button: str = "left", x: int = -1, y: int = -1) -> str:
        self.calls.append(("click", button, x, y))
        return "BTN_" + button.upper()

    def scroll(self, dy: int, dx: int = 0) -> None:
        self.calls.append(("scroll", dy, dx))


@pytest.fixture
def fake_hands(monkeypatch):
    """Swap her hands for a recorder — no device, no real keystrokes."""
    fake = FakeHands()
    monkeypatch.setattr(hid, "_HANDS", fake)
    return fake


# ── Name resolution ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("enter", "KEY_ENTER"),
        ("return", "KEY_ENTER"),
        ("esc", "KEY_ESC"),
        ("escape", "KEY_ESC"),
        ("space", "KEY_SPACE"),
        ("tab", "KEY_TAB"),
        ("backspace", "KEY_BACKSPACE"),
        ("delete", "KEY_DELETE"),
        ("del", "KEY_DELETE"),
        ("ctrl", "KEY_LEFTCTRL"),
        ("control", "KEY_LEFTCTRL"),
        ("shift", "KEY_LEFTSHIFT"),
        ("alt", "KEY_LEFTALT"),
        ("option", "KEY_LEFTALT"),
        ("super", "KEY_LEFTMETA"),
        ("win", "KEY_LEFTMETA"),
        ("cmd", "KEY_LEFTMETA"),
        ("up", "KEY_UP"),
        ("down", "KEY_DOWN"),
        ("left", "KEY_LEFT"),
        ("right", "KEY_RIGHT"),
        ("home", "KEY_HOME"),
        ("end", "KEY_END"),
        ("pgup", "KEY_PAGEUP"),
        ("pagedown", "KEY_PAGEDOWN"),
        ("f5", "KEY_F5"),
        ("F12", "KEY_F12"),
        ("KEY_A", "KEY_A"),
        (" printscreen ", "KEY_SYSRQ"),
    ],
)
def test_resolve_key_name(given, expected):
    name, shift = hid.resolve_key_name(given)
    assert name == expected
    assert not shift


def test_resolve_single_char_uses_char_map():
    assert hid.resolve_key_name("a") == ("KEY_A", False)
    assert hid.resolve_key_name("A") == ("KEY_A", True)
    assert hid.resolve_key_name("!") == ("KEY_1", True)


def test_resolve_empty_rejected():
    with pytest.raises(ValueError):
        hid.resolve_key_name("   ")


# ── Character map ──────────────────────────────────────────────


def test_char_map_covers_printable_ascii():
    for code in range(32, 127):
        name, shift = hid.char_key(chr(code))
        assert name.startswith("KEY_"), f"no key for {chr(code)!r}"
        assert isinstance(shift, bool)


def test_char_map_specials():
    assert hid.char_key("\n") == ("KEY_ENTER", False)
    assert hid.char_key(" ") == ("KEY_SPACE", False)
    assert hid.char_key("+") == ("KEY_EQUAL", True)
    assert hid.char_key("?") == ("KEY_SLASH", True)


def test_char_map_refuses_non_layout():
    with pytest.raises(ValueError):
        hid.char_key("é")


# ── Probe ──────────────────────────────────────────────────────


def test_probe_reports_without_raising():
    report = hid.probe()
    assert report["backend"] == "linux-uinput"
    assert set(report) >= {
        "evdev_installed",
        "uinput_node",
        "uinput_writable",
        "screen",
        "ready",
    }
    assert report["ready"] == (report["evdev_installed"] and report["uinput_writable"])


# ── Tool layer ─────────────────────────────────────────────────


def test_key_press_combo_parsing(fake_hands):
    result = framework.key_press("ctrl+c")
    assert result.success
    assert fake_hands.calls == [("press", "c", ("ctrl",))]


def test_key_press_modifier_string(fake_hands):
    assert framework.key_press("tab", modifiers="shift, ctrl").success
    assert fake_hands.calls == [("press", "tab", ("shift", "ctrl"))]


def test_key_press_modifier_list(fake_hands):
    assert framework.key_press("delete", modifiers=["ctrl", "alt"]).success
    assert fake_hands.calls == [("press", "delete", ("ctrl", "alt"))]


def test_key_press_bare_plus_is_a_key(fake_hands):
    assert framework.key_press("+").success
    assert fake_hands.calls == [("press", "+", ())]


def test_type_text(fake_hands):
    result = framework.type_text("hello world")
    assert result.success
    assert fake_hands.calls == [("type", "hello world")]
    assert result.data["chars"] == 11


def test_mouse_move_and_scroll(fake_hands):
    assert framework.mouse_move(5, -3).success
    assert framework.mouse_scroll(2).success
    assert ("move", 5, -3) in fake_hands.calls
    assert ("scroll", 2, 0) in fake_hands.calls


def test_mouse_goto_and_click(fake_hands):
    assert framework.mouse_goto(640, 480).success
    assert framework.mouse_click("left", x=100, y=50).success
    assert framework.mouse_click("right").success
    assert ("goto", 640, 480) in fake_hands.calls
    assert ("click", "left", 100, 50) in fake_hands.calls
    assert ("click", "right", -1, -1) in fake_hands.calls


def test_hid_status_returns_report():
    result = framework.hid_status()
    assert "evdev_installed" in result.data
    assert result.success == result.data["ready"]


def test_unavailable_degrades_to_error(monkeypatch):
    def _refuse():
        raise RuntimeError("/dev/uinput is not writable by her")

    monkeypatch.setattr(hid, "_HANDS", None)
    monkeypatch.setattr(hid, "_open_hands", _refuse)
    result = framework.type_text("hi")
    assert not result.success
    assert "uinput" in result.error


# ── Live device — only where a hand can actually open ─────────


@pytest.mark.skipif(
    not (os.path.exists("/dev/uinput") and os.access("/dev/uinput", os.W_OK)),
    reason="needs a writable /dev/uinput",
)
def test_live_devices_open_and_close():
    pytest.importorskip("evdev")
    hands = hid.hands()
    try:
        # fileno is the honest "opened" signal: writes go through the
        # uinput fd, while .device only resolves later (and only
        # matters for reading the created node's own events).
        assert hands._kbd.fileno() > 0
        assert hands._ptr.fileno() > 0
    finally:
        hands.close()
        hid._HANDS = None

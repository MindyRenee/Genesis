"""ARC cabinet tests — the control-surface contract.

The window is the only actuation path when hands are on: these
tests pin the surface itself (which key or click is which action),
the decoder's tolerance for the noise synthetic X input carries,
and the round-trip — every emitted control must decode back to the
same action, because the world only sees what lands on the glass.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from worlds import arc_cabinet
from worlds.arc_cabinet import (
    _SCALE,
    _controls_for,
    _Decoder,
    _drop_arc_display,
    _free_display_number,
    ensure_arc_display,
)


def _resolve_ops(ops: list[tuple[Any, ...]]) -> tuple[int, dict | None] | None:
    """Replay emitted ops through the decoder the way the window would.

    A ``("key", keysym)`` op presses that keysym; a ``("click", x, y)``
    op presses the center of cell (x, y) on the board; a ``("reset",)``
    op fires the RESET control. What decodes is what landed.
    """
    dec = _Decoder()
    result = None
    for kind, *rest in ops:
        if kind == "key":
            result = dec.feed_key(rest[0]) or result
        elif kind == "click":
            cx, cy = rest
            result = dec.feed_click(int((cx + 0.5) * _SCALE), int((cy + 0.5) * _SCALE)) or result
        else:  # reset — the control's own firing, already decoded
            result = (0, None)
    return result


class TestDecoderKeys:
    """Keysyms -> actions: the venue's documented control surface."""

    def test_direction_keys(self):
        """WASD carries ACTIONS 1-4 — the primary scheme."""
        dec = _Decoder()
        for key, aid in (("w", 1), ("s", 2), ("a", 3), ("d", 4)):
            assert dec.feed_key(key) == (aid, None)

    def test_arrow_keys(self):
        """Arrows carry the same actions — the alternate scheme."""
        dec = _Decoder()
        for key, aid in (("Up", 1), ("Down", 2), ("Left", 3), ("Right", 4)):
            assert dec.feed_key(key) == (aid, None)

    def test_interact_and_undo(self):
        """Space/F is ACTION5, Z (either case) is ACTION7."""
        dec = _Decoder()
        assert dec.feed_key("space") == (5, None)
        assert dec.feed_key("f") == (5, None)
        assert dec.feed_key("z") == (7, None)
        assert dec.feed_key("Z") == (7, None)

    def test_reset_key(self):
        """R (either case) decodes to RESET."""
        dec = _Decoder()
        assert dec.feed_key("r") == (0, None)
        assert dec.feed_key("R") == (0, None)

    def test_noise_keysyms_ignored(self):
        """Stray keysyms commit nothing — modifiers, digits, phantoms."""
        dec = _Decoder()
        for junk in ("Shift_L", "1", "7", "e", "Meta_L", "Tab", "Caps_Lock"):
            assert dec.feed_key(junk) is None
        assert dec.feed_key("d") == (4, None)


class TestDecoderClicks:
    """Board clicks -> ACTION6: the pointer says where it landed."""

    def test_click_decodes_cell(self):
        """A click at a cell's pixel decodes to ACTION6 there."""
        dec = _Decoder()
        assert dec.feed_click(12 * _SCALE + 1, 7 * _SCALE + 3) == (6, {"x": 12, "y": 7})

    def test_click_corner_cells(self):
        """First and last cells of the board decode."""
        dec = _Decoder()
        assert dec.feed_click(0, 0) == (6, {"x": 0, "y": 0})
        assert dec.feed_click(63 * _SCALE + _SCALE - 1, 63 * _SCALE + 1) == (
            6,
            {"x": 63, "y": 63},
        )

    def test_click_off_board_is_lost(self):
        """A click past the board's edge is not a move."""
        dec = _Decoder()
        assert dec.feed_click(-1, 10) is None
        assert dec.feed_click(10, 64 * _SCALE) is None
        assert dec.feed_click(64 * _SCALE, 10) is None


class TestControlsFor:
    """Action -> emitted ops: what the hands are told to do."""

    def test_simple_emits_single_key(self):
        """A simple action is one key press on the real surface."""
        assert _controls_for(3, None) == [("key", "a")]
        assert _controls_for(5, None) == [("key", "space")]
        assert _controls_for(7, None) == [("key", "z")]

    def test_reset_clicks_the_reset_control(self):
        """RESET is a click on the RESET button — like the venue's."""
        assert _controls_for(0, None) == [("reset",)]

    def test_action6_is_a_click_at_the_cell(self):
        """ACTION6 emits a pointer click at its cell, never typed text."""
        ops = _controls_for(6, {"x": 33, "y": 12})
        assert ops == [("click", 33, 12)]

    def test_unknown_action_has_no_control(self):
        """An action with no control raises — never guessed."""
        with pytest.raises(ValueError):
            _controls_for(9, None)


class TestRoundTrip:
    """Every emitted control resolves back to its action."""

    @pytest.mark.parametrize("aid", [0, 1, 2, 3, 4, 5, 7])
    def test_simple_round_trip(self, aid):
        """Emit then decode a simple action — same id back."""
        assert _resolve_ops(_controls_for(aid, None)) == (aid, None)

    def test_action6_round_trip(self):
        """Emit then decode ACTION6 — same id and cells back."""
        ops = _controls_for(6, {"x": 40, "y": 21})
        assert _resolve_ops(ops) == (6, {"x": 40, "y": 21})


class TestEnsureArcDisplay:
    """Display selection — her own screen, never the user's by default."""

    @pytest.fixture(autouse=True)
    def clean_display_cache(self, monkeypatch):
        """Each test starts with no cached screen and no pinned spec."""
        monkeypatch.setattr(arc_cabinet, "_arc_display", None)
        monkeypatch.setattr(arc_cabinet, "_arc_display_proc", None)
        monkeypatch.delenv("GENESIS_ARC_DISPLAY", raising=False)

    def test_explicit_display_spec_wins(self, monkeypatch):
        """GENESIS_ARC_DISPLAY=:N is used verbatim — no spawn, no probe."""
        monkeypatch.setenv("GENESIS_ARC_DISPLAY", ":42")
        assert ensure_arc_display() == ":42"

    def test_host_display_spec_wins(self, monkeypatch):
        """host:N counts as an explicit spec too."""
        monkeypatch.setenv("GENESIS_ARC_DISPLAY", "host:5")
        assert ensure_arc_display() == "host:5"

    def test_shared_mode_uses_session_display(self, monkeypatch):
        """GENESIS_ARC_DISPLAY=shared plays on the user's screen."""
        monkeypatch.setenv("GENESIS_ARC_DISPLAY", "shared")
        monkeypatch.setenv("DISPLAY", ":0.0")
        assert ensure_arc_display() == ":0.0"

    def test_shared_mode_without_session_display(self, monkeypatch):
        """shared with no DISPLAY means there is no screen to play on."""
        monkeypatch.setenv("GENESIS_ARC_DISPLAY", "shared")
        monkeypatch.delenv("DISPLAY", raising=False)
        assert ensure_arc_display() is None

    def test_cached_display_is_reused(self, monkeypatch):
        """Once her screen is up it is returned again — never respawned."""
        monkeypatch.setattr(arc_cabinet, "_arc_display", ":7")

        def forbidden_spawn(*args, **kwargs):
            """Tripwire — a cached display must not spawn a second server."""
            pytest.fail("cached display spawned a second Xephyr")

        monkeypatch.setattr(arc_cabinet.subprocess, "Popen", forbidden_spawn)
        assert ensure_arc_display() == ":7"

    def test_xephyr_spawn_caches_and_registers_teardown(self, monkeypatch):
        """A spawned nested server is cached and reaped at process exit."""
        registered = []

        class _Proc:
            """A live fake Xephyr — poll() says running, terminate ends it."""

            def __init__(self):
                """Alive until told otherwise."""
                self.dead = False

            def poll(self):
                """Still running — the spawn wait keeps probing."""
                return None

            def terminate(self):
                """Record the polite stop."""
                self.dead = True

            def wait(self, timeout=None):
                """Exit promptly on terminate."""
                return 0

            def kill(self):
                """Record the hard stop."""
                self.dead = True

        proc = _Proc()
        monkeypatch.setattr(arc_cabinet.shutil, "which", lambda name: "/usr/bin/Xephyr")
        monkeypatch.setattr(arc_cabinet, "_free_display_number", lambda: 9)
        monkeypatch.setattr(arc_cabinet.subprocess, "Popen", lambda *a, **k: proc)
        monkeypatch.setattr(arc_cabinet, "_display_usable", lambda name: True)
        monkeypatch.setattr(arc_cabinet.atexit, "register", lambda fn: registered.append(fn))
        assert ensure_arc_display() == ":9"
        assert arc_cabinet._arc_display == ":9"
        assert arc_cabinet._arc_display_proc is proc
        assert registered == [_drop_arc_display]

    def test_no_xephyr_falls_back_to_session_display(self, monkeypatch):
        """No nested server available — the session display serves."""
        monkeypatch.setattr(arc_cabinet.shutil, "which", lambda name: None)
        monkeypatch.setenv("DISPLAY", ":9")
        assert ensure_arc_display() == ":9"

    def test_no_server_no_display_returns_none(self, monkeypatch):
        """Nothing to play on at all — the run degrades to bench play."""
        monkeypatch.setattr(arc_cabinet.shutil, "which", lambda name: None)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(arc_cabinet, "Path", lambda p: SimpleNamespace(exists=lambda: False))
        assert ensure_arc_display() is None


class TestDisplayHousekeeping:
    """The spawned screen's lifecycle: number pick and teardown."""

    def test_free_display_number_skips_claimed_sockets(self, monkeypatch):
        """The first unclaimed X display number wins; junk names skip."""
        socks = [
            SimpleNamespace(name="X0"),
            SimpleNamespace(name="X1"),
            SimpleNamespace(name="X-lock"),
        ]
        monkeypatch.setattr(arc_cabinet, "Path", lambda p: SimpleNamespace(glob=lambda pat: socks))
        assert _free_display_number() == 2

    def test_drop_arc_display_terminates_the_server(self, monkeypatch):
        """Teardown stops Xephyr politely and clears the cache."""
        ended = {}

        class _Proc:
            """The fake server — records how it was stopped."""

            def terminate(self):
                """The polite stop landed."""
                ended["term"] = True

            def wait(self, timeout=None):
                """It exited inside the grace window."""
                return 0

            def kill(self):
                """The hard stop landed."""
                ended["kill"] = True

        monkeypatch.setattr(arc_cabinet, "_arc_display", ":9")
        monkeypatch.setattr(arc_cabinet, "_arc_display_proc", _Proc())
        _drop_arc_display()
        assert ended == {"term": True}
        assert arc_cabinet._arc_display is None
        assert arc_cabinet._arc_display_proc is None

    def test_drop_arc_display_kills_a_server_that_ignores_terminate(self, monkeypatch):
        """A wedged server gets the kill after the grace window."""
        ended = {}

        class _WedgedProc:
            """The fake server that won't die politely."""

            def terminate(self):
                """Ignored — it keeps running."""

            def wait(self, timeout=None):
                """Never exits inside the grace window."""
                raise arc_cabinet.subprocess.TimeoutExpired("Xephyr", timeout)

            def kill(self):
                """The hard stop landed."""
                ended["kill"] = True

        monkeypatch.setattr(arc_cabinet, "_arc_display", ":9")
        monkeypatch.setattr(arc_cabinet, "_arc_display_proc", _WedgedProc())
        _drop_arc_display()
        assert ended == {"kill": True}

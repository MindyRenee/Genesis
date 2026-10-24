"""ARC cabinet tests — the keystroke contract.

The window is the only actuation path when hands are on: these
tests pin the wire format (what gets emitted per action), the
decoder's tolerance for the noise synthetic X input carries, and
the round-trip — every emitted sequence must decode back to the
same action, because the world only sees what lands on the glass.
"""

from __future__ import annotations

import pytest
from parietal_lobe.arc_cabinet import _KeyDecoder, _keystrokes_for


def _decode_wire(ops: list[tuple[str, str]]) -> tuple[int, dict | None] | None:
    """Expand emitted ops into the keysym stream the window sees.

    A ``("key", seq)`` op sends its payload as one keysym (``r``,
    ``Return``); a ``("text", s)`` op types each char literally.
    """
    dec = _KeyDecoder()
    result = None
    for kind, payload in ops:
        if kind == "text":
            for ch in payload:
                assert dec.feed(ch) is None  # mid-token must not commit
        else:
            result = dec.feed(payload) or result
    return result


class TestKeyDecoder:
    def test_simple_actions(self):
        dec = _KeyDecoder()
        for key, aid in (("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5), ("7", 7)):
            assert dec.feed(key) == (aid, None)

    def test_reset(self):
        dec = _KeyDecoder()
        assert dec.feed("r") == (0, None)
        assert dec.feed("0") == (0, None)

    def test_coordinate_token(self):
        dec = _KeyDecoder()
        for ch in "x12y7":
            assert dec.feed(ch) is None
        assert dec.feed("Return") == (6, {"x": 12, "y": 7})

    def test_coordinate_escape_cancels(self):
        dec = _KeyDecoder()
        for ch in "x12y":
            dec.feed(ch)
        dec.feed("Escape")
        # Abandoned — a following Return completes nothing.
        assert dec.feed("Return") is None
        assert dec.feed("5") == (5, None)

    def test_malformed_coord_is_lost(self):
        dec = _KeyDecoder()
        for ch in "x1x2y3":  # stray x mid-token
            dec.feed(ch)
        assert dec.feed("Return") is None

    def test_out_of_range_coord_is_lost(self):
        dec = _KeyDecoder()
        for ch in "x64y0":  # 64 is off the board
            dec.feed(ch)
        assert dec.feed("Return") is None

    def test_noise_keysyms_ignored(self):
        dec = _KeyDecoder()
        # Synthetic input arrives with junk on the wire — stray
        # modifiers, phantom chars — none of it is a move.
        for junk in ("Shift_L", "e", "d", "Meta_L", "Tab", "Caps_Lock"):
            assert dec.feed(junk) is None
        assert dec.feed("3") == (3, None)

    def test_stray_return_with_no_token(self):
        dec = _KeyDecoder()
        assert dec.feed("Return") is None

    def test_digits_8_9_6_alone_are_noise(self):
        dec = _KeyDecoder()
        for k in ("6", "8", "9"):
            assert dec.feed(k) is None


class TestKeystrokes:
    def test_simple_emits_single_key(self):
        assert _keystrokes_for(3, None) == [("key", "3")]

    def test_reset_emits_r(self):
        assert _keystrokes_for(0, None) == [("key", "r")]

    def test_action6_types_coords_then_return(self):
        ops = _keystrokes_for(6, {"x": 33, "y": 12})
        assert ops == [("text", "x33y12"), ("key", "Return")]

    def test_unknown_action_has_no_keymap(self):
        with pytest.raises(ValueError):
            _keystrokes_for(9, None)


class TestRoundTrip:
    """Every emitted sequence decodes back to its action."""

    @pytest.mark.parametrize("aid", [0, 1, 2, 3, 4, 5, 7])
    def test_simple_round_trip(self, aid):
        assert _decode_wire(_keystrokes_for(aid, None)) == (aid, None)

    def test_action6_round_trip(self):
        ops = _keystrokes_for(6, {"x": 40, "y": 21})
        assert _decode_wire(ops) == (6, {"x": 40, "y": 21})

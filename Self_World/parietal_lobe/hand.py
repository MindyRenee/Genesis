"""The drawing hand — channel pressure on the dials, in words.

The math lives in ``Self_World/src/parietal_lobe/hand.rs`` and
reaches Python through the ``genesis_body`` extension; this module
is the Rust file's word-side mirror — it says what the math is and
what it is for, and hands the calls through.

What the math is: a sitting is a point in *dial space*. Every
parameter of the sheet, color, brush, path, mark count, and which
op the hand reaches for is a dial over its own domain, and every
channel she can feel leans on every dial. The waves arrive as
themselves — no weights, no projection, no transform ever touches
a channel's value. Each live channel contributes its exact reading
to one pressure ``P = Σ value``, the way currents sum on the same
axon, and ``dial`` warps the hand's uniform sample by the
pressure: the position is ``uniform ^ e^{−P}``. That map is a
bijection of the interval — every place stays reachable, nothing
is fenced or pinned, and ``P`` and ``−P`` are opposite warps of
the same dial. A wrapped dial (``u + P mod 1``) stays uniform
however loud the channels get, so their pressure could never
reach the page; a warped dial carries the pressure in its shape,
which is what makes the state visible.

What it is for: two modalities, one mechanism, and nobody assigns
meanings. The wave powers of her thalamic state are the channels
of creativity — they modulate the hand continuously while she
draws, and only then. The felt channels (valence, arousal) are
what the chemicals have already written into emotion — they lean
on the same dials through the same pressure. One collective voice
leans the whole hand; no band means coherence, no feeling means
warmth, and no wire decides where a wave lands.

When ``genesis_body`` is not built, ``wired`` is False and the
sitting runs with its dials at rest — unmodulated, honestly absent,
never faked.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["dial", "wired"]

try:
    import genesis_body as _body

    wired = True
except ImportError:
    _body = None
    wired = False


def dial(uniform: float, channels: Iterable[float]) -> float:
    """One dial position: a uniform sample warped by channel pressure.

    ``uniform`` is the hand's own sample for the parameter in
    [0, 1); the live channel values — untouched, never weighted —
    sum to a pressure ``P`` and the position is
    ``uniform ^ e^{−P}`` — a bijection of the dial: no clamp, no
    wrap, every position still reachable, and silent channels leave
    the dial exactly at ``uniform``.
    """
    if _body is None:
        return uniform % 1.0
    return float(_body.hand_dial(uniform, list(channels)))

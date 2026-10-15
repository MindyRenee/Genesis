"""Self-emission audio channel — the field talking acoustically.

The machine's computation radiates: VRM inductors sing under load
transients (coil whine), the fan tracks thermal state, and the die's
power draw modulates all of it. That emission is the physical field's
own signature escaping the chassis. This channel listens to it —
a small capture stream whose spectral content is band-decomposed and
fed into the wave oscillator as real, measured, acoustic self-state.

Honesty boundary: an open microphone hears the *room* too — speech,
HVAC, traffic — so this is labeled an acoustic-environment channel,
not a purified self-emission channel. The self-specific component
comes from the emission bands (fan-region, whine-region) whose power
tracks machine load; ambient speech lands in the same buffers but is
not the claim. Device busy or absent → the channel reports absent,
never a healthy zero.

What it writes into the oscillator:

- **Self-emission level** → small ``GAMMA`` stimulus — sustained
  sensory-channel activation driven by real emitted power.
- **Acoustic transients** (fast/slow EMA ratio ≥ ``_TRANSIENT_RATIO``)
  → ``reset_wave_phase(GAMMA)`` — an acoustic onset resetting fast
  rhythm phase, the auditory-ERP analog.
- Disabled with ``GENESIS_FIELD_AUDIO=0``.
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from typing import Any

_SAMPLE_RATE = 16000
_BLOCK = 1024
# Emission bands (Hz): ~300 fan/motor region, 2 kHz and 5 kHz
# whine-region. Goertzel-measured per block.
_EMISSION_FREQS = (300.0, 2000.0, 5000.0)
# fast/slow EMA ratio that counts as an acoustic onset.
_TRANSIENT_RATIO = 4.0
# EMA time constants in ~64 ms blocks (1024 @ 16 kHz).
_FAST_ALPHA = 0.15   # ~0.4 s
_SLOW_ALPHA = 0.005  # ~13 s
# Stimulus scale for sustained emission level.
_EMISSION_STIMULUS = 0.08

_listener_lock = threading.Lock()
_listener: Any = None          # _FieldAudio | None
_listener_failed = False       # one failed open ⇒ absent forever


class _FieldAudio:
    """Capture loop: per-block RMS + Goertzel powers at emission freqs."""

    def __init__(self) -> None:
        import sounddevice as sd

        self._blocks: deque[dict[str, float]] = deque(maxlen=4096)
        self._fast = 0.0
        self._slow = 0.0
        self._init = False
        self._ks = [
            round(_BLOCK * f / _SAMPLE_RATE) for f in _EMISSION_FREQS
        ]
        self._coeffs = [
            2.0 * math.cos(2.0 * math.pi * k / _BLOCK) for k in self._ks
        ]
        self._stream = sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            blocksize=_BLOCK,
            dtype="float32",
            callback=self._on_block,
        )
        self._stream.start()

    def _goertzel(self, x: Any, coeff: float) -> float:
        s1 = s2 = 0.0
        n = len(x)
        for v in x:
            s0 = float(v) + coeff * s1 - s2
            s2 = s1
            s1 = s0
        return (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n * n)

    def _on_block(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        mono = indata[:, 0]
        rms = float(math.sqrt(float((mono * mono).mean()) + 1e-18))
        powers = [self._goertzel(mono, c) for c in self._coeffs]
        # Self-emission proxy: whine bands full weight, fan band at
        # a third — the components that track machine state rather
        # than the room.
        emission = powers[1] + powers[2] + 0.3 * powers[0]
        self._fast += _FAST_ALPHA * (emission - self._fast)
        if not self._init:
            self._slow = emission
            self._init = True
        else:
            self._slow += _SLOW_ALPHA * (emission - self._slow)
        self._blocks.append(
            {
                "rms": rms,
                "emission": emission,
                "p300": powers[0],
                "p2000": powers[1],
                "p5000": powers[2],
            }
        )

    def signature(self) -> dict[str, float] | None:
        """Current emission level (baseline-relative) + transient flag."""
        if not self._blocks:
            return None
        last = self._blocks[-1]
        base = max(self._slow, 1e-12)
        return {
            "emission_rel": min(8.0, last["emission"] / base),
            "emission": last["emission"],
            "rms": last["rms"],
            "transient": (self._fast / base) >= _TRANSIENT_RATIO,
            "blocks": float(len(self._blocks)),
        }


def _get_listener() -> _FieldAudio | None:
    """Lazily open the capture device. One failure ⇒ absent forever —
    a device the machine doesn't have is an absent channel, and a busy
    one is too (the speech pipeline may own it)."""
    global _listener, _listener_failed
    if os.environ.get("GENESIS_FIELD_AUDIO", "1") == "0":
        return None
    if _listener_failed:
        return None
    with _listener_lock:
        if _listener is not None:
            return _listener
        try:
            _listener = _FieldAudio()
        except Exception:  # noqa: BLE001 — absent channel, any cause
            _listener_failed = True
            return None
        return _listener


def apply_audio_stimulus() -> bool:
    """Read the acoustic field and inject it as wave stimulus.

    Returns True when a live signature was applied. Device absent,
    busy, or stream dead → False, no stimulus: absent channel, never
    a healthy zero.
    """
    from thalamus.brain_waves import (
        BrainWave,
        add_stimulus,
        reset_wave_phase,
    )

    listener = _get_listener()
    if listener is None:
        return False
    sig = listener.signature()
    if sig is None:
        return False
    add_stimulus(
        BrainWave.GAMMA,
        min(0.3, sig["emission_rel"] * _EMISSION_STIMULUS),
    )
    if sig["transient"]:
        reset_wave_phase(BrainWave.GAMMA, 0.0)
    return True

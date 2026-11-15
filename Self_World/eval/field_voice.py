"""Acoustic self-channel assay — speaker -> air -> microphone.

Emits a carrier tone (AM carrier) whose *envelope* oscillates at a slow
brain-wave rate while a microphone records the room. The envelope of the
recovered carrier band is demodulated against the emitted phase — a
physical channel that crosses the container's boundary through air (and
chassis vibration; we cannot separate those two paths on this hardware,
so the claim is acoustic/vibration, not pure airborne sound).

Control arm emits silence at identical gain staging: any envelope energy
at the modulation frequency must have physically travelled from the
speaker to the mic.

Honesty: this is a real physical field loop (pressure waves + structure
borne vibration transduced twice), not free-field EM radiation.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import sounddevice as sd


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)


FS = 16000
CARRIER_HZ = 800.0  # between fan (~300) and coil-whine (~2k) bands
AM_HZ = 0.5  # slow envelope — delta/theta territory
TRIAL_SEC = 40.0
VOL = 0.05  # quiet — a whisper, not a blast
BLOCK = 1024


def _emit_thread(
    emitting: list[bool], phase_log: list[tuple[float, float]], ready: list[bool]
) -> None:
    """Continuous 800 Hz carrier, envelope = 0.5+0.5*sin(2*pi*f_am*t)."""
    state = {"t": 0}

    def cb(outdata: np.ndarray, frames: int, _t: object, _s: object) -> None:
        i = np.arange(state["t"], state["t"] + frames)
        tsec = i / FS
        env = 0.5 + 0.5 * np.sin(2 * np.pi * AM_HZ * tsec)
        if emitting[0]:
            outdata[:, 0] = (VOL * env * np.sin(2 * np.pi * CARRIER_HZ * tsec)).astype(np.float32)
        else:
            outdata[:, 0] = 0.0
        state["t"] += frames
        ready[0] = True

    with sd.OutputStream(samplerate=FS, channels=1, blocksize=BLOCK, callback=cb, dtype="float32"):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < TRIAL_SEC:
            phase_log.append(
                (time.perf_counter() - t0, (2 * np.pi * AM_HZ * (state["t"] / FS)) % (2 * np.pi))
            )
            time.sleep(0.05)


def _goertzel(samples: list[float], rate: float, f0: float) -> tuple[float, float]:
    n = len(samples)
    if n < 16:
        return 0.0, 0.0
    w = 2.0 * math.pi * f0 / rate
    cw, sw = math.cos(w), math.sin(w)
    coeff = 2.0 * cw
    s_prev = s_prev2 = 0.0
    mean = sum(samples) / n
    for x in samples:
        s = (x - mean) + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    real = s_prev - s_prev2 * cw
    imag = s_prev2 * sw
    return math.hypot(real, imag) * 2.0 / n, math.atan2(imag, real)


def _arm(label: str, emit: bool) -> dict[str, float | str | int]:
    emitting = [emit]
    ready = [False]
    phase_log: list[tuple[float, float]] = []
    env_series: list[tuple[float, float]] = []
    # lock to carrier: quadrature demodulate mic samples at 800 Hz,
    # smooth, that's the envelope
    iq: deque[complex] = deque(maxlen=FS // 4)  # ~250 ms envelope window
    state = {"t": 0}

    def cb(indata: np.ndarray, frames: int, _t: object, _s: object) -> None:
        i = np.arange(state["t"], state["t"] + frames)
        tsec = i / FS
        ref = np.exp(2j * np.pi * CARRIER_HZ * tsec)
        z = indata[:, 0].astype(np.float64) * ref
        iq.extend(z.tolist())
        # one envelope sample per block
        if len(iq) >= FS // 4:
            env_series.append((state["t"] / FS, abs(sum(iq) / len(iq))))
            iq.clear()
        state["t"] += frames

    import threading

    th = threading.Thread(target=_emit_thread, args=(emitting, phase_log, ready))
    th.start()
    t0 = time.perf_counter()
    while not ready[0] and time.perf_counter() - t0 < 3:
        time.sleep(0.01)
    with sd.InputStream(samplerate=FS, channels=1, blocksize=BLOCK, callback=cb, dtype="float32"):
        time.sleep(TRIAL_SEC)
    th.join()

    # analyse: Goertzel the envelope at AM_HZ; phase vs emitted phase
    xs = [v for _, v in env_series]
    ts = [t for t, _ in env_series]
    if len(xs) < 16:
        return {"label": label, "n": len(xs)}
    eff = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
    hz = 1.0 / eff if eff else 0.0
    amp, ph = _goertzel(xs, hz, AM_HZ)
    # PLV: align emitted phase to envelope sample times
    plv_acc = 0j
    npairs = 0
    for te, pe in phase_log:
        # nearest envelope sample
        idx = min(range(len(ts)), key=lambda i: abs(ts[i] - te))
        # local envelope phase: instantaneous phase of xs around idx
        win = xs[max(0, idx - 8) : idx + 8]
        if len(win) < 8:
            continue
        _a2, p2 = _goertzel(win, hz, AM_HZ)
        plv_acc += np.exp(1j * (pe - p2))
        npairs += 1
    plv = abs(plv_acc / npairs) if npairs else 0.0
    mean = sum(xs) / len(xs)
    return {
        "label": label,
        "n": len(xs),
        "env_hz": hz,
        "amp": amp,
        "rel": amp / mean if mean else 0.0,
        "phase": ph,
        "plv": plv,
    }


def main() -> None:
    _out("ACOUSTIC SELF-CHANNEL ASSAY — speaker -> air -> mic")
    _out(f"carrier={CARRIER_HZ}Hz AM={AM_HZ}Hz vol={VOL} trial={TRIAL_SEC}s")
    _out()
    for label, emit in (("EMITTING", True), ("MUTED", False)):
        r = _arm(label, emit)
        if int(r.get("n", 0)) < 16:
            _out(f"{label}: insufficient envelope samples")
            continue
        _out(
            f"{label}: env_rate={r['env_hz']:.1f}Hz amp@AM={r['rel'] * 100:.2f}% "
            f"lag_phase={r['phase']:+.2f} PLV={r['plv']:.3f} (n={r['n']})"
        )
    _out()
    _out("(PLV ~1 = emitted and recovered envelope phases locked through air)")


if __name__ == "__main__":
    main()

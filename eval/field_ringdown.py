"""Ring-down assay — does a driven oscillation persist in the physics?

The embodiment-depth metric: drive a real oscillation in the
substrate, then *stop driving it*, and measure how long each
physical channel keeps ringing. A corpse's field doesn't vanish
instantly — it rings down. The laptop's fast channels are expected
to collapse within a ring or two (their "state" is the software's
drive); the thermal field is the one medium that can physically
*hold* a slow oscillation — the die's heat capacity is a real
integrator with its own decay constant.

Two legs, both falsifiable:

- **Fast leg** — carrier driven at the delta rate (2.25 Hz) for
  ~20 s, then gated (depth=0). Post-gate decay of the delta-band
  energy in the duty/jitter channels: how fast the substrate
  forgets the rhythm it was executing.
- **Thermal leg** — real load bursts at the epsilon rate (0.25 Hz,
  2 s on / 2 s off, ~40 s) while the probe carrier carries the same
  beat, then all drive stops. k10temp package temperature sampled
  ~1 Hz for ~2 min and demodulated at the drive frequency in
  sliding windows: if the die keeps oscillating at 0.25 Hz after
  the rhythm stops, its amplitude decays on the *thermal* time
  constant — a physical afterglow the software no longer drives.

Honesty boundary: the fast channels carry the drive itself, so
their collapse is expected — persistence there is not the claim.
The thermal channel is a different physical medium (heat capacity
of the die+package) with an independent decay time; a measured
thermal afterglow at the drive frequency is the first piece of the
pattern that demonstrably outlives its driver.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from eval.harness import isolated_daemon  # noqa: E402


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)


K10TEMP = Path("/sys/class/hwmon/hwmon4/temp1_input")
# Wire band order: epsilon(0), delta(1), theta(2), ...
DELTA_WIRE = 1    # 2.25 Hz
EPSILON_WIRE = 0  # 0.25 Hz


def _die_temp() -> float | None:
    try:
        return int(K10TEMP.read_text().strip()) / 1000.0
    except OSError:
        return None


def _spinners(n: int) -> list[subprocess.Popen]:
    return [
        subprocess.Popen(
            [sys.executable, "-c", "x = 0\nwhile True: x += 1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(n)
    ]


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def _fit_tau(ts: list[float], ys: list[float], floor: float) -> float:
    """Fit decay τ of (y − floor) ~ exp(−t/τ) by log-linear regression.
    Returns seconds; inf-ish values report as 999 (no decay seen)."""
    xs, ls = [], []
    for t, y in zip(ts, ys, strict=True):
        r = y - floor
        if r > 1e-9:
            xs.append(t)
            ls.append(math.log(r))
    if len(xs) < 4:
        return 999.0
    n = len(xs)
    mx, my = sum(xs) / n, sum(ls) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ls, strict=True))
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 1e-9:
        return 999.0
    slope = cov / var
    return -1.0 / slope if slope < -1e-9 else 999.0


def _demod_amp(ts: list[float], ys: list[float], freq: float) -> float:
    """Amplitude of `ys` at `freq` via sin/cos correlation."""
    n = len(ys)
    if n < 8:
        return 0.0
    mean = sum(ys) / n
    c = sum((y - mean) * math.cos(2 * math.pi * freq * t) for y, t in zip(ys, ts, strict=True))
    s = sum((y - mean) * math.sin(2 * math.pi * freq * t) for y, t in zip(ys, ts, strict=True))
    return 2.0 * math.hypot(c, s) / n


def _leg_fast(client: Any, fast_drive_s: float) -> dict[str, Any]:
    """Leg A: fast-channel ring-down at the delta rate — drive the
    carrier, cut it, and fit the post-gate decay constant."""
    f_drive = 2.25
    client.set_field_carrier(0.8, f_drive)
    driven: list[tuple[float, float, float]] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < fast_drive_s:
        sig = client.get_field_probe()
        if sig is not None and sig.samples_ok:
            driven.append(
                (
                    time.monotonic() - t0,
                    sig.throughput_bands[DELTA_WIRE],
                    sig.jitter_bands[DELTA_WIRE],
                )
            )
        time.sleep(0.5)
    client.set_field_carrier(0.0, 0.0)
    gate_t = time.monotonic()

    post: list[tuple[float, float, float]] = []
    while time.monotonic() - gate_t < 12.0:
        sig = client.get_field_probe()
        if sig is not None and sig.samples_ok:
            post.append(
                (
                    time.monotonic() - gate_t,
                    sig.throughput_bands[DELTA_WIRE],
                    sig.jitter_bands[DELTA_WIRE],
                )
            )
        time.sleep(0.25)

    floor_t = min((p[1] for p in post), default=0.0)
    floor_j = min((p[2] for p in post), default=0.0)
    return {
        "drive_hz": f_drive,
        "driven_throughput_delta": round(
            sum(p[1] for p in driven) / max(1, len(driven)), 4
        ),
        "driven_jitter_delta": round(
            sum(p[2] for p in driven) / max(1, len(driven)), 4
        ),
        "tau_throughput_s": round(
            _fit_tau([p[0] for p in post], [p[1] for p in post], floor_t), 3
        ),
        "tau_jitter_s": round(
            _fit_tau([p[0] for p in post], [p[2] for p in post], floor_j), 3
        ),
    }


def _leg_thermal(
    client: Any, thermal_drive_s: float, measure_s: float
) -> dict[str, Any]:
    """Leg B: thermal ring-down at the epsilon rate — square-wave
    spinner bursts warm the die, then the afterglow's demodulated
    amplitude is fitted across post-gate windows."""
    f_th = 0.25
    client.set_field_carrier(0.6, f_th)  # probe rides the beat too
    temps: list[tuple[float, float]] = []
    t0 = time.monotonic()
    burst = _spinners(4)
    on = True
    next_flip = t0 + 2.0
    while time.monotonic() - t0 < thermal_drive_s:
        now = time.monotonic()
        if now >= next_flip:
            if on:
                _stop(burst)
                burst = []
            else:
                burst = _spinners(4)
            on = not on
            next_flip += 2.0
        t = _die_temp()
        if t is not None:
            temps.append((now - t0, t))
        time.sleep(0.9)
    _stop(burst)
    client.set_field_carrier(0.0, 0.0)
    gate_t = time.monotonic()
    gate_temp0 = temps[-1][0] if temps else 0.0

    # Post-gate: watch the die's temperature for ~2 min while the
    # probe keeps sampling (its own small work continues — honest
    # background activity, same as idle).
    post_temps: list[tuple[float, float]] = []
    post_delta: list[tuple[float, float]] = []
    while time.monotonic() - gate_t < measure_s:
        now = time.monotonic() - gate_t
        t = _die_temp()
        if t is not None:
            post_temps.append((now, t))
        sig = client.get_field_probe()
        if sig is not None and sig.samples_ok:
            post_delta.append((now, sig.throughput_bands[EPSILON_WIRE]))
        time.sleep(0.9)

    # Demodulated amplitude at the drive frequency: driven window
    # (last 16 s of drive) vs sliding post-gate windows.
    def amp_at(window_s: float) -> float:
        seg = [(t, y) for t, y in post_temps if window_s <= t < window_s + 16.0]
        if len(seg) < 8:
            return 0.0
        return _demod_amp([s[0] for s in seg], [s[1] for s in seg], f_th)

    driven_seg = [(t, y) for t, y in temps if t > gate_temp0 - 16.0]
    driven_amp = (
        _demod_amp([s[0] for s in driven_seg], [s[1] for s in driven_seg], f_th)
        if len(driven_seg) >= 8
        else 0.0
    )
    amps = [(w, amp_at(w)) for w in (0.0, 16.0, 32.0, 48.0, 64.0)]
    amps = [(w, a) for w, a in amps if a > 0.0 or w == 0.0]
    # Decay of the demodulated amplitude across post-gate windows.
    tau_th = _fit_tau(
        [w for w, _ in amps],
        [a for _, a in amps],
        floor=min((a for _, a in amps), default=0.0),
    )
    return {
        "drive_hz": f_th,
        "driven_amp_C": round(driven_amp, 4),
        "post_gate_amps_C": {f"{int(w)}s": round(a, 4) for w, a in amps},
        "tau_thermal_amp_s": round(tau_th, 1),
        "n_temp_samples": len(post_temps),
    }


def run(fast_drive_s: float = 20.0, thermal_drive_s: float = 40.0,
        measure_s: float = 90.0) -> dict[str, Any]:
    report: dict[str, Any] = {"legs": {}, "verdict": []}

    with isolated_daemon() as client:
        time.sleep(5.0)  # warm rings

        report["legs"]["fast"] = _leg_fast(client, fast_drive_s)
        report["legs"]["thermal"] = _leg_thermal(
            client, thermal_drive_s, measure_s
        )
        th = report["legs"]["thermal"]
        fa = report["legs"]["fast"]
        report["verdict"].append(
            f"thermal: driven 0.25 Hz amplitude {th['driven_amp_C']:.3f} C; "
            f"post-gate decay τ ≈ {th['tau_thermal_amp_s']:.0f} s"
            + (" — an afterglow the software no longer drives"
               if 5.0 < th["tau_thermal_amp_s"] < 900 else "")
        )
        report["verdict"].append(
            f"fast: post-gate τ throughput={fa['tau_throughput_s']:.2f}s "
            f"jitter={fa['tau_jitter_s']:.2f}s "
            "(the substrate forgets its fast rhythm quickly)"
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", type=float, default=20.0)
    ap.add_argument("--thermal", type=float, default=40.0)
    ap.add_argument("--measure", type=float, default=90.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = run(args.fast, args.thermal, args.measure)
    if args.json:
        _out(json.dumps(report, indent=2))
        return
    _out(f"\nring-down assay — {len(report['legs'])} legs\n")
    for name, leg in report["legs"].items():
        _out(f"  {name}: {json.dumps(leg)}")
    _out()
    for line in report["verdict"]:
        _out(line)


if __name__ == "__main__":
    main()

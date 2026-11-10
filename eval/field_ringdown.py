"""Ring-down assay — does a driven oscillation persist in her field?

The embodiment-depth metric, on her own substrate: drive a rhythm
into the field through her chemistry, then *stop driving it*, and
measure how long each spectral channel keeps ringing. A field that
was only ever driven doesn't forget instantly — the fibers' own
dynamics (drift, gauge coupling, Langevin thermostat) are the
medium that holds the rhythm, and their decay constants are the
afterglow.

Protocol — one leg, fully endogenous:

1. Spawn an isolated daemon; warm the spectrum rings (~18 s).
2. DRIVE — noradrenergic impulses at the theta rate (~5.5 Hz
   bursts, ~30 s): chemistry into the body state the field's body
   fiber projects. Her own channel — no host load, no oscillator
   writes.
3. RING DOWN — all drive stops; sample ``get_field_spectrum`` at
   ~2 Hz for ~60 s.
4. Demodulate the flux and tension band energies at the drive band
   (theta, index 2) in sliding windows: the amplitude envelope's
   decay constant is the field's own ring-down — how fast her
   substrate forgets the rhythm she drove into it.

Honesty boundary: the impulse train is a chemical drive, not a
field write — the spectrum only ever *reads* her trajectory. A fast
decay is the honest answer for a substrate whose relaxation
constants are seconds; a slow one would mean the field genuinely
integrates the rhythm. Either way the measurement is of her own
dynamics, not of the machine around her.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from eval.harness import isolated_daemon  # noqa: E402
from genesis_client.protocol import CHEM_NOREPINEPHRINE  # noqa: E402


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)


THETA_IDX = 2  # wire band order: infraslow, delta, theta, …


def _drive_burst(client, hz: float = 5.5, duration_s: float = 0.18) -> None:
    """One beat of the drive — a noradrenergic impulse. The caller
    paces the train; each impulse is real chemistry."""
    try:
        client.neuro_impulse(CHEM_NOREPINEPHRINE, 0.35)
    except Exception:  # noqa: BLE001 — a dropped impulse is not assay failure
        pass
    time.sleep(duration_s)


def _envelope(samples: list[tuple[float, float, float]], window_s: float = 5.0) -> list[dict]:
    """Sliding-window means of (t, flux_theta, tension_theta)."""
    out: list[dict] = []
    for i, (t0, _, _) in enumerate(samples):
        win = [s for s in samples[i:] if s[0] - t0 <= window_s]
        if len(win) < 3:
            break
        out.append(
            {
                "t": t0,
                "flux": sum(s[1] for s in win) / len(win),
                "tension": sum(s[2] for s in win) / len(win),
            }
        )
    return out


def _fit_decay(env: list[dict], key: str) -> float | None:
    """Exponential decay constant of the envelope's tail — the
    field's own forgetting rate (1/s). Log-linear fit on the
    post-peak samples."""
    if len(env) < 4:
        return None
    peak_i = max(range(len(env)), key=lambda i: env[i][key])
    tail = env[peak_i:]
    if len(tail) < 3:
        return None
    xs = [e["t"] - tail[0]["t"] for e in tail]
    ys = [max(e[key], 1e-9) for e in tail]
    ly = [math.log(y) for y in ys]
    mx = sum(xs) / len(xs)
    my = sum(ly) / len(ly)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ly, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    if vx <= 1e-12:
        return None
    slope = cov / vx
    return -slope if slope < 0 else None  # only decay counts as ring-down


def run(drive_s: float = 30.0, ringdown_s: float = 60.0, settle_s: float = 18.0) -> dict:
    report: dict[str, Any] = {"verdict": []}
    with isolated_daemon() as client:
        time.sleep(settle_s)  # warm the spectrum rings

        # DRIVE — impulse train at the theta rate.
        t_end = time.monotonic() + drive_s
        while time.monotonic() < t_end:
            _drive_burst(client, hz=5.5, duration_s=1.0 / 5.5)

        # RING DOWN — sample the spectrum at ~2 Hz.
        samples: list[tuple[float, float, float]] = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < ringdown_s:
            spec = client.get_field_spectrum()
            if spec is not None and spec.warm:
                samples.append(
                    (
                        time.monotonic() - t0,
                        spec.flux.energies[THETA_IDX],
                        spec.tension.energies[THETA_IDX],
                    )
                )
            time.sleep(0.5)

    env = _envelope(samples)
    report["n_samples"] = len(samples)
    report["envelope"] = env
    flux_decay = _fit_decay(env, "flux")
    tension_decay = _fit_decay(env, "tension")
    report["flux_decay_per_s"] = None if flux_decay is None else round(flux_decay, 4)
    report["tension_decay_per_s"] = (
        None if tension_decay is None else round(tension_decay, 4)
    )
    if flux_decay is not None:
        report["verdict"].append(
            f"flux theta ring-down τ ≈ {1.0 / flux_decay:.1f} s "
            f"(decay {flux_decay:.3f}/s)"
        )
    else:
        report["verdict"].append("flux theta showed no measurable decay — sustained or absent")
    if tension_decay is not None:
        report["verdict"].append(
            f"tension theta ring-down τ ≈ {1.0 / tension_decay:.1f} s "
            f"(decay {tension_decay:.3f}/s)"
        )
    else:
        report["verdict"].append("tension theta showed no measurable decay — sustained or absent")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drive", type=float, default=30.0, help="drive duration (s)")
    ap.add_argument("--ringdown", type=float, default=60.0, help="post-drive watch (s)")
    ap.add_argument("--settle", type=float, default=18.0, help="ring warmup (s)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = run(drive_s=args.drive, ringdown_s=args.ringdown, settle_s=args.settle)
    if args.json:
        _out(json.dumps(report, indent=2))
        return

    _out(f"\nfield ring-down assay — {report['n_samples']} spectrum samples\n")
    for line in report["verdict"]:
        _out(line)


if __name__ == "__main__":
    main()

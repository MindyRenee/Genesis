"""Field-coupling assay — does her own substrate measurably move the
wave model?

Her field substrate (``src/field/``) integrates body, prediction,
and agency fibers on the latent manifold; ``src/field/spectrum.rs``
decomposes that trajectory at the wave model's band centers and
``brainstem.field_probe`` maps it onto ``BrainWaveOscillator`` as
stimulus, evoked resets, carrier phase lock, and per-band entrainment.
This assay asks the falsifiable question: when her substrate is
perturbed through her own chemistry, does the *coupled* wave state
move where the *uncoupled* state does not?

Protocol:

1. Spawn an isolated daemon (the field integrates with it).
2. Warm the spectrum rings (~16 s for the full window).
3. Three phases — idle, driven, recovery — ~10 samples each at
   ~1 Hz. The drive is a norepinephrine surge through her own
   ``neuro_impulse`` channel: real chemistry into the body state the
   field's body fiber projects, not a load on the host. Per sample:
   a. fetch the spectral signature (``get_field_spectrum``),
   b. CONTROL — a private ``BrainWaveOscillator`` that never sees
      field stimulus, assessed on the same ``NeuroSummary``,
   c. COUPLED — the module singleton, assessed through
      ``assess_brain_waves(..., client=client)`` so the real
      signature lands as stimulus and entrainment,
   d. record per-band powers for both arms plus the substrate
      channels (flux/tension band energies, tension cv, carrier
      frequency and phase).
4. Report per-band power deltas (driven − idle) for both arms and
   the Pearson correlation between the substrate channels and the
   coupled band powers across all samples.

Honesty notes:

- The control is *field-blind*, not chemistry-blind: the daemon's
  own neurochemistry responds to the same impulses, so the control
  arm can drift under drive. What it cannot see is the spectral
  signature — that is the contribution being measured.
- The coupled arm includes the evoked-reset and phase-lock
  pathways, which the control lacks. Their aggregate effect lands
  in these band powers.
- This is an engineering assay: it measures whether the coupling is
  causally effective, not whether it matters to experience.
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


BAND_NAMES = [
    "infraslow",
    "delta",
    "theta",
    "alpha",
    "sigma",
    "beta1",
    "beta2",
    "beta3",
    "gamma",
    "epsilon",
]


def _powers(state) -> dict[str, float]:
    """Per-band powers from a BrainWaveState, keyed by name."""
    return {b.value: p for b, p in state.powers.items()}


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _drive(client, magnitude: float = 0.4) -> None:
    """One noradrenergic surge — chemistry into the body the field
    integrates, her own channel, no host instrumentation."""
    try:
        client.neuro_impulse(CHEM_NOREPINEPHRINE, magnitude)
    except Exception:  # noqa: BLE001 — a dropped impulse is not assay failure
        pass


def run(samples_per_phase: int = 10, settle_s: float = 18.0) -> dict:
    from thalamus.brain_waves import BrainWaveOscillator, assess_brain_waves

    rows: list[dict] = []
    # Control arm: a private oscillator that never sees field
    # stimulus — the uncoupled trajectory on the same inputs.
    control = BrainWaveOscillator()
    with isolated_daemon() as client:
        time.sleep(settle_s)  # warm the spectrum rings

        for phase in ("idle", "driven", "recovery"):
            for _ in range(samples_per_phase):
                if phase == "driven":
                    _drive(client)
                spec = client.get_field_spectrum()
                summary = client.get_neuro_summary()
                ctrl_state = control.assess(summary)
                # Coupled arm: the singleton, real signature applied.
                coupled_state = assess_brain_waves(summary, client=client)
                rows.append(
                    {
                        "phase": phase,
                        "spec": None
                        if spec is None
                        else {
                            "warm": spec.warm,
                            "tension_cv": spec.tension.cv,
                            "flux_mean": spec.flux.mean,
                            "tension_mean": spec.tension.mean,
                            "carrier_freq": spec.carrier_freq_hz,
                            "carrier_phase": spec.carrier_phase,
                            "flux_theta": spec.flux.energies[2],
                            "flux_alpha": spec.flux.energies[3],
                            "tension_theta": spec.tension.energies[2],
                        },
                        "control": _powers(ctrl_state),
                        "coupled": _powers(coupled_state),
                    }
                )
                time.sleep(1.0)
            time.sleep(2.0)

    return _analyze(rows)


def _analyze(rows: list[dict]) -> dict:
    """Band deltas, channel movement, and the phase-lock leg."""

    def phase_mean(arm: str, phase: str, band: str) -> float:
        vals = [r[arm][band] for r in rows if r["phase"] == phase]
        return sum(vals) / max(1, len(vals))

    report: dict[str, Any] = {
        "n_samples": len(rows),
        "bands": {},
        "channels": {},
        "verdict": [],
    }
    for band in BAND_NAMES:
        c_delta = phase_mean("coupled", "driven", band) - phase_mean("coupled", "idle", band)
        u_delta = phase_mean("control", "driven", band) - phase_mean("control", "idle", band)
        # Correlate the tension channel's turbulence with the coupled
        # band — the substrate's own argument against the model's rhythm.
        xs = [
            r["spec"]["tension_cv"]
            for r in rows
            if r["spec"] is not None and r["spec"]["warm"]
        ]
        ys = [
            r["coupled"][band]
            for r in rows
            if r["spec"] is not None and r["spec"]["warm"]
        ]
        n = min(len(xs), len(ys))
        report["bands"][band] = {
            "control_drive_delta": round(u_delta, 4),
            "coupled_drive_delta": round(c_delta, 4),
            "tension_corr": round(_pearson(xs[:n], ys[:n]), 3),
        }

    # Channel-level: did the substrate channels themselves move?
    for ch in ("tension_cv", "flux_mean", "tension_mean", "carrier_freq"):
        idle_v = [
            r["spec"][ch] for r in rows if r["phase"] == "idle" and r["spec"] is not None
        ]
        drv_v = [
            r["spec"][ch] for r in rows if r["phase"] == "driven" and r["spec"] is not None
        ]
        if idle_v and drv_v:
            report["channels"][ch] = {
                "idle_mean": round(sum(idle_v) / len(idle_v), 4),
                "drive_mean": round(sum(drv_v) / len(drv_v), 4),
            }

    moved = [
        b
        for b, d in report["bands"].items()
        if abs(d["coupled_drive_delta"]) > abs(d["control_drive_delta"]) + 0.01
    ]
    report["verdict"].append(
        f"{len(moved)} band(s) moved more when coupled than uncoupled: "
        + (", ".join(moved) if moved else "none")
    )

    # ─── Phase lock (the entrainment leg) ──────────────────────
    from brainstem.field_probe import field_phase_lock

    lock = field_phase_lock()
    report["phase_lock"] = lock
    freqs = [
        r["spec"]["carrier_freq"]
        for r in rows
        if r["spec"] is not None and r["spec"]["warm"] and r["spec"]["carrier_freq"] > 0
    ]
    if freqs:
        report["phase_lock"]["mean_carrier_hz"] = round(sum(freqs) / len(freqs), 3)
    report["verdict"].append(
        f"phase lock: PLV={lock['plv']:.3f} over {int(lock['n'])} samples"
        + (
            f", carrier={report['phase_lock']['mean_carrier_hz']:.3f} Hz"
            if "mean_carrier_hz" in report["phase_lock"]
            else " (carrier absent)"
        )
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--settle", type=float, default=18.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = run(samples_per_phase=args.samples, settle_s=args.settle)
    if args.json:
        _out(json.dumps(report, indent=2))
        return

    _out(f"\nfield-coupling assay — {report['n_samples']} samples\n")
    _out("substrate channels (her field actually moved):")
    for ch, d in report["channels"].items():
        _out(f"  {ch:16s} idle={d['idle_mean']:<12} driven={d['drive_mean']}")
    _out(f"\n{'band':9s} {'ctrl Δ':>9s} {'coupled Δ':>10s} {'tension r':>9s}")
    for band, d in report["bands"].items():
        _out(
            f"{band:9s} {d['control_drive_delta']:>+9.4f} "
            f"{d['coupled_drive_delta']:>+10.4f} {d['tension_corr']:>9.3f}"
        )
    _out()
    for line in report["verdict"]:
        _out(line)


if __name__ == "__main__":
    main()

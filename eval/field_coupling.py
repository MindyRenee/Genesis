"""Field-coupling assay — does real physics measurably move the wave model?

The field probe (`src/daemon/field_probe.rs`) injects the machine's
real substrate dynamics into ``BrainWaveOscillator`` as stimulus and
evoked phase resets. This assay asks the falsifiable question: when
the physical substrate is perturbed by real load, does the *coupled*
wave state move where the *uncoupled* state does not?

Protocol:

1. Spawn an isolated daemon (the probe thread spawns with it).
2. Warm the probe rings (~5 s).
3. Three phases — idle, induced CPU load, recovery — ~10 samples each
   at ~1 Hz. Per sample:
   a. fetch the physical signature (``get_field_probe``),
   b. CONTROL — a private ``BrainWaveOscillator`` that never sees
      field stimulus, assessed on the same ``NeuroSummary``,
   c. COUPLED — the module singleton, assessed through
      ``assess_brain_waves(..., client=client)`` so the real
      signature lands as stimulus and evoked resets,
   d. record per-band powers for both arms plus the physical
      channels (jitter_cv, throughput mean, pstate rate).
4. Report per-band power deltas (load − idle) for both arms and the
   Pearson correlation between the physical channels and the
   coupled band powers across all samples.

Honesty notes:

- The control is *probe-blind*, not physics-blind: the daemon's own
  neurochemistry responds to real sensors via interoception, so the
  control arm can drift under load. What it cannot see is the probe
  signature — that is the contribution being measured.
- The coupled arm includes the evoked phase-reset pathway, which the
  control lacks. Phase resets alter subsequent PAC interactions;
  their aggregate effect lands in these band powers.
- This is an engineering assay: it measures whether the coupling is
  causally effective, not whether it matters to experience.
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


def _busy_spinners(n: int) -> list[subprocess.Popen]:
    """Induce real substrate load — n CPU-bound spinners."""
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


def run(samples_per_phase: int = 10, settle_s: float = 5.0) -> dict:
    from thalamus.brain_waves import BrainWaveOscillator, assess_brain_waves

    rows: list[dict] = []
    # Control arm: a private oscillator that never sees field
    # stimulus — the uncoupled trajectory on the same inputs.
    control = BrainWaveOscillator()
    with isolated_daemon() as client:
        time.sleep(settle_s)  # warm the probe rings

        phases = [("idle", None), ("load", None), ("recovery", None)]
        spinners: list[subprocess.Popen] = []
        for phase, _ in phases:
            if phase == "load":
                spinners = _busy_spinners(3)
                time.sleep(1.5)
            for _ in range(samples_per_phase):
                sig = client.get_field_probe()
                summary = client.get_neuro_summary()
                ctrl_state = control.assess(summary)
                # Coupled arm: the singleton, real signature applied.
                coupled_state = assess_brain_waves(summary, client=client)
                rows.append(
                    {
                        "phase": phase,
                        "sig": None
                        if sig is None
                        else {
                            "jitter_cv": sig.jitter_cv,
                            "throughput_mean": sig.throughput_mean,
                            "pstate_rate": sig.pstate_rate,
                            "irq_rate": sig.irq_rate,
                            "samples_ok": sig.samples_ok,
                            "carrier_lag": sig.carrier_lag,
                            "carrier_phase": sig.carrier_phase,
                            "carrier_coherence": sig.carrier_coherence,
                        },
                        "control": _powers(ctrl_state),
                        "coupled": _powers(coupled_state),
                    }
                )
                time.sleep(1.0)
            if phase == "load":
                _stop(spinners)
                spinners = []
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
        c_delta = phase_mean("coupled", "load", band) - phase_mean("coupled", "idle", band)
        u_delta = phase_mean("control", "load", band) - phase_mean("control", "idle", band)
        xs = [r["sig"]["jitter_cv"] for r in rows if r["sig"] is not None]
        ys = [r["coupled"][band] for r in rows if r["sig"] is not None]
        n = min(len(xs), len(ys))
        report["bands"][band] = {
            "control_load_delta": round(u_delta, 4),
            "coupled_load_delta": round(c_delta, 4),
            "jitter_corr": round(_pearson(xs[:n], ys[:n]), 3),
        }

    # Channel-level: did the physical channels themselves move?
    for ch in ("jitter_cv", "throughput_mean", "pstate_rate", "irq_rate"):
        idle_v = [r["sig"][ch] for r in rows if r["phase"] == "idle" and r["sig"] is not None]
        load_v = [r["sig"][ch] for r in rows if r["phase"] == "load" and r["sig"] is not None]
        if idle_v and load_v:
            report["channels"][ch] = {
                "idle_mean": round(sum(idle_v) / len(idle_v), 4),
                "load_mean": round(sum(load_v) / len(load_v), 4),
            }

    moved = [
        b
        for b, d in report["bands"].items()
        if abs(d["coupled_load_delta"]) > abs(d["control_load_delta"]) + 0.01
    ]
    report["verdict"] = [
        f"{len(moved)} band(s) moved more when coupled than uncoupled: "
        + (", ".join(moved) if moved else "none"),
    ]

    # ─── Phase lock (the entrainment leg) ──────────────────────
    from brainstem.field_probe import field_phase_lock

    lock = field_phase_lock()
    report["phase_lock"] = lock
    coh = [
        r["sig"]["carrier_coherence"]
        for r in rows
        if r["sig"] is not None and r["sig"]["carrier_coherence"] is not None
    ]
    if coh:
        report["phase_lock"]["mean_coherence"] = round(sum(coh) / len(coh), 3)
    report["verdict"].append(
        f"phase lock: PLV={lock['plv']:.3f} over {int(lock['n'])} samples"
        + (
            f", response coherence={report['phase_lock']['mean_coherence']:.3f}"
            if "mean_coherence" in report["phase_lock"]
            else " (v1 daemon — phase channel absent)"
        )
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--settle", type=float, default=5.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = run(samples_per_phase=args.samples, settle_s=args.settle)
    if args.json:
        _out(json.dumps(report, indent=2))
        return

    _out(f"\nfield-coupling assay — {report['n_samples']} samples\n")
    _out("physical channels (the substrate actually moved):")
    for ch, d in report["channels"].items():
        _out(f"  {ch:16s} idle={d['idle_mean']:<12} load={d['load_mean']}")
    _out(f"\n{'band':9s} {'ctrl Δ':>9s} {'coupled Δ':>10s} {'jitter r':>9s}")
    for band, d in report["bands"].items():
        _out(
            f"{band:9s} {d['control_load_delta']:>+9.4f} "
            f"{d['coupled_load_delta']:>+10.4f} {d['jitter_corr']:>9.3f}"
        )
    _out()
    for line in report["verdict"]:
        _out(line)


if __name__ == "__main__":
    main()

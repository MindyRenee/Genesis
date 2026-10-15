"""Physically-coupled computing elements — the ephaptic-coupling assay.

Spawns worker processes pinned to separate cores. Worker 0 modulates its
real work at a drive frequency; the other workers run constant work and
measure their own execution rate at ~250 Hz. Any drive-frequency
component in a neighbour's measured rate travelled through the shared
physical medium — rail droop, L3/memory pressure, thermal field,
governor response — never through a message. A sham arm applies the same
mean load without modulation so total work alone can't explain a
nonzero transfer.

Honesty boundary: coupling through shared silicon resources is real
physical coupling, but it is substrate-mediated, not free-field
radiative coupling. A positive result means the elements perturb one
another through the die's electrical/thermal state — the closest
laptop-scale analogue of ephaptic field coupling, not a literal EM bus.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any


def _out(*args: object) -> None:
    """Report channel — for these assay tools, stdout IS the product."""
    print(*args, flush=True)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "genesis_client"))

SAMPLE_HZ = 250.0          # matches the daemon probe cadence
DRIVE_HZ = 5.0             # inside the theta/alpha range she models
TRIAL_SEC = 24.0
WORK_ITERS = 400           # ~a few hundred µs per slice on this die
DEPTH = 0.7                # fraction of slice spent idle at trough

# ---------------------------------------------------------------- worker


def _mix(seed: int, iters: int) -> int:
    x = seed or 0x9E3779B9
    for _ in range(iters):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x


def _worker(core: int, worker_id: int, drive_hz: float, depth: float,
            modulated: bool, q: mp.Queue, stop: Any,
            sync: bool = False, sync_k: float = 0.0,
            sync_delay: float = 0.0, sync_thermal: bool = False) -> None:
    os.sched_setaffinity(0, {core})
    # Collective-only, naturally-lagged channel: die temperature. No single
    # element can dominate it and the heatsink's thermal mass IS the
    # delay line the metronome literature says in-phase lock needs.
    temp_fd = -1
    if sync_thermal:
        for hw in sorted(os.listdir("/sys/class/hwmon")):
            try:
                if open(f"/sys/class/hwmon/{hw}/name").read().strip() == "k10temp":
                    temp_fd = os.open(f"/sys/class/hwmon/{hw}/temp1_input", os.O_RDONLY)
            except OSError:
                continue
    sink = 0
    dropped = 0
    t0 = time.perf_counter()
    window = 1.0 / SAMPLE_HZ
    next_t = t0 + window
    # Autonomous-sync state: the element's phase advances at drive_hz plus
    # a pull proportional to the collective-load imprint on its OWN
    # measured execution rate — the medium coupling the elements. The
    # imprint can be read through a delay line (sync_delay seconds): the
    # metronome-on-a-platform literature says a lagged medium response is
    # what makes in-phase lock stable, and instant droop on this die is
    # reactive — delaying the sense leg is the lag the physics lacks.
    theta = (worker_id * 2.0 + 0.7)  # deterministic de-phased starts
    baseline = 0.0
    omega0 = drive_hz * 2.0 * math.pi
    delay_n = round(sync_delay / window)
    history: list[float] = []
    while not stop.is_set():
        now = time.perf_counter()
        # The substrate-sensitive quantity: how long a FIXED slice of work
        # took right now. Everything after it is fill, sized by the duty —
        # the element's own oscillation loads the shared medium; in sync
        # mode each element modulates by its OWN phase.
        syncing = sync
        if syncing or modulated:
            phase = theta if syncing else (now - t0) * drive_hz * 2.0 * math.pi
            duty = 1.0 - depth * (0.5 + 0.5 * math.sin(phase))
        else:
            duty = 1.0 - depth * 0.5
        w0 = time.perf_counter()
        sink ^= _mix(sink ^ int(w0 * 1e9), WORK_ITERS)
        work_secs = time.perf_counter() - w0
        fill_until = w0 + window * duty
        while time.perf_counter() < fill_until:
            sink ^= _mix(sink, 40)
        if syncing:
            # Read the medium: die temperature (thermal channel — lagged
            # and collective) or own execution rate (fast channel — the
            # element's own load contaminates it). Positive imprint =
            # shared medium running hot / slow right now.
            if temp_fd >= 0:
                os.lseek(temp_fd, 0, os.SEEK_SET)
                temp = float(os.read(temp_fd, 64))
                baseline = temp if baseline == 0.0 else baseline + (temp - baseline) * 0.005
                imprint = (temp - baseline) / baseline
            else:
                rate = WORK_ITERS / work_secs
                baseline = rate if baseline == 0.0 else baseline + (rate - baseline) * 0.02
                imprint = -(rate - baseline) / baseline
            history.append(imprint)
            felt = history[-delay_n] if 0 < delay_n <= len(history) else imprint
            theta += omega0 * window + sync_k * felt
        if syncing or not modulated:
            try:
                q.put_nowait((worker_id, now - t0, work_secs, theta if syncing else None))
            except queue.Full:
                # Collector fell behind — drop the sample rather than
                # stall the worker's pacing on IPC backpressure.
                dropped += 1
        rem = next_t - time.perf_counter()
        if rem > 0:
            # Genuine idle for the undriven fraction — a busy-wait here
            # would keep the substrate saturated and erase the driver's
            # duty modulation entirely.
            time.sleep(rem)
        next_t += window
    q.put(("sink", sink, 0.0, dropped))


# ---------------------------------------------------------------- analysis


def _goertzel(samples: list[float], rate: float, f0: float) -> tuple[float, float]:
    n = len(samples)
    if n < 16:
        return 0.0, 0.0
    k = f0 * n / rate
    w = 2.0 * math.pi * k / n
    cw, sw = math.cos(w), math.sin(w)
    coeff = 2.0 * cw
    s_prev = s_prev2 = 0.0
    mean = sum(samples) / n
    for x in samples:
        s = (x - mean) + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    real = s_prev - s_prev2 * cw
    imag = s_prev2 * sw
    amp = math.hypot(real, imag) * 2.0 / n
    return amp, math.atan2(imag, real)


@dataclass
class Arm:
    label: str
    modulated: bool
    sync: bool = False
    sync_k: float = 0.0
    sync_delay: float = 0.0
    sync_thermal: bool = False
    drive_hz: float = DRIVE_HZ
    trial_sec: float = TRIAL_SEC


def _run_arm(arm: Arm, cores: list[int]) -> dict[int, list[tuple[float, float, float | None]]]:
    q: mp.Queue = mp.Queue(maxsize=16384)
    stop = mp.Event()
    procs = []
    for i, core in enumerate(cores):
        p = mp.Process(
            target=_worker,
            args=(core, i, arm.drive_hz, DEPTH,
                  arm.modulated and i == 0, q, stop, arm.sync, arm.sync_k,
                  arm.sync_delay, arm.sync_thermal),
            daemon=True,
        )
        p.start()
        procs.append(p)
    ids = range(len(cores)) if arm.sync else range(1, len(cores))
    series: dict[int, list[tuple[float, float, float | None]]] = {i: [] for i in ids}
    end = time.monotonic() + arm.trial_sec
    while time.monotonic() < end:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item[0] in series:
            series[item[0]].append((item[1], item[2], item[3]))
    stop.set()
    for p in procs:
        p.join(timeout=2.0)
    return series


def _order_parameter(thetas: list[float]) -> float:
    n = len(thetas)
    if n < 2:
        return 0.0
    return abs(sum(math.cos(t) for t in thetas) / n
               + 1j * sum(math.sin(t) for t in thetas) / n)


def _order_trajectory(series: dict[int, list[tuple[float, float, float | None]]],
                      chunks: int = 8) -> list[float]:
    """Kuramoto order parameter r per chunk of the trial, aligned by index."""
    n = min((len(v) for v in series.values()), default=0)
    out = []
    per = n // chunks
    for c in range(chunks):
        end_i = (c + 1) * per - 1
        thetas = [t for v in series.values()
                  if end_i < len(v) and (t := v[end_i][2]) is not None]
        out.append(_order_parameter(thetas))
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "coupling"
    cores = [1, 2, 3]
    if mode == "sync":
        # Emergent-synchrony assay: every element oscillates by its own
        # phase and feels the collective load through the die. Blind arm
        # oscillates identically but ignores the imprint — same software,
        # no physical feedback.
        k_val = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        f_val = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
        t_val = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
        d_val = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        thermal = len(sys.argv) > 6 and sys.argv[6] == "thermal"
        chan = "THERMAL (k10temp)" if thermal else "execution-rate imprint"
        _out("FIELD-ELEMENTS EMERGENT-SYNC ASSAY")
        _out(f"cores={cores} omega0={f_val}Hz depth={DEPTH} K={k_val} "
              f"delay={d_val}s channel={chan} trial={t_val}s")
        _out("each element is source AND sensor — coupling via the die")
        _out()
        arms = [(f"COUPLED K={k_val} d={d_val}", k_val)]
        if not (len(sys.argv) > 7 and sys.argv[7] == "noblind"):
            arms.append(("BLIND (oscillate, don't feel)", 0.0))
        for label, k in arms:
            arm = Arm(label, modulated=False, sync=True, sync_k=k,
                      sync_delay=d_val, sync_thermal=thermal,
                      drive_hz=f_val, trial_sec=t_val)
            _out(f"--- {label} ---")
            series = _run_arm(arm, cores)
            traj = _order_trajectory(series)
            if traj:
                _out("  r(t) = " + "  ".join(f"{r:.2f}" for r in traj))
                _out(f"  start={traj[0]:.2f}  end={traj[-1]:.2f}  max={max(traj):.2f}")
            else:
                _out("  no phase data")
            _out()
        return

    _out("FIELD-ELEMENTS COUPLING ASSAY")
    _out(f"cores={cores} drive={DRIVE_HZ}Hz depth={DEPTH} trial={TRIAL_SEC}s")
    _out("workers share rail/L3/thermal — no messages between them")
    _out()

    results: dict[str, dict[int, float]] = {}
    for arm in (Arm("DRIVEN (w0 modulates)", True), Arm("SHAM (constant mean load)", False)):
        _out(f"--- {arm.label} ---")
        series = _run_arm(arm, cores)
        for wid, pts in series.items():
            ts = [t for t, _, _ in pts]
            xs = [x for _, x, _ in pts]
            if len(xs) < 100:
                _out(f"  worker{wid}: insufficient samples ({len(xs)})")
                continue
            actual_rate = (ts[-1] - ts[0]) / max(len(ts) - 1, 1)
            eff_hz = 1.0 / actual_rate if actual_rate > 0 else 0.0
            amp, phase = _goertzel(xs, eff_hz, DRIVE_HZ)
            mean = sum(xs) / len(xs)
            rel = amp / mean if mean else 0.0
            _out(f"  worker{wid}: n={len(xs)} eff_rate={eff_hz:.0f}Hz "
                  f"amp@{DRIVE_HZ}Hz={rel*100:.2f}% phase={phase:+.2f}rad")
            results.setdefault(arm.label, {})[wid] = rel
    _out()
    driven = results.get("DRIVEN (w0 modulates)", {})
    sham = results.get("SHAM (constant mean load)", {})
    if driven and sham:
        _out("COUPLING SUMMARY (relative amp @ drive freq, driven / sham):")
        for wid in sorted(driven):
            d, s = driven[wid], sham.get(wid, 0.0)
            ratio = d / s if s > 1e-9 else float("inf")
            _out(f"  worker{wid}: {d*100:.2f}% vs {s*100:.2f}%  SNR={ratio:.1f}x")
        med = sorted(driven.values())[len(driven) // 2]
        med_s = sorted(sham.values())[len(sham) // 2]
        coupled = med > 2 * med_s
        verdict = "PHYSICAL COUPLING DETECTED" if coupled else "no transfer above sham floor"
        _out(f"\n  verdict: {verdict}")


if __name__ == "__main__":
    main()

"""Drift detection: compare a fresh capture against a stored baseline.

Two comparisons are supported:

1. :func:`compare_snapshots` — fresh :class:`Snapshot` vs stored
   baseline :class:`Snapshot`. Deterministic rest state means tight
   tolerances are viable; circadian-sensitive chemicals (see
   ``CIRCADIAN_SENSITIVE``) get wider bands for time-of-day effects.
2. :func:`check_snapshot_vs_defaults` — :class:`Snapshot` vs the
   genetic ``DEFAULT_BASELINES``. A coarse sanity gate: catches a
   fundamentally broken substrate even when no baseline file exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from genesis_client.protocol import PHASE_NAMES

from .baseline import CIRCADIAN_SENSITIVE, DEFAULT_BASELINES, Snapshot

# (warn_abs, fail_abs) per signal. Calibrated against repeated fresh
# captures, which are bit-identical when taken at the same circadian
# phase — so any drift above the warn band is real change, not noise.
_CHEM_WARN = 0.03
_CHEM_FAIL = 0.10
_CIRCADIAN_WARN = {"melatonin": 0.15, "dopamine": 0.05, "serotonin": 0.04, "histamine": 0.05}
_CIRCADIAN_FAIL = {"melatonin": 0.45, "dopamine": 0.12, "serotonin": 0.10, "histamine": 0.12}

_SCALAR_BANDS: dict[str, tuple[float, float]] = {
    "arousal": (0.05, 0.15),
    "valence": (0.05, 0.15),
    "global_tone": (0.05, 0.15),
    "plasticity_gate": (0.05, 0.15),
    "encoding_weight": (0.05, 0.15),
    "consolidation_weight": (0.05, 0.15),
    "retrieval_weight": (0.05, 0.15),
    "coupling_drift": (0.02, 0.08),
    "mean_receptor_sensitivity": (0.03, 0.10),
    "bdnf_effective": (0.05, 0.15),
    "cortisol_effective": (0.05, 0.15),
    "surprise_ema": (0.05, 0.20),
    "free_energy": (0.05, 0.20),
    "precision": (0.05, 0.20),
    "allostatic_load": (0.05, 0.20),
}

# Coarse sanity tolerances vs genetic defaults (no baseline file).
_SANITY_TOL = 0.12
_SANITY_CIRCADIAN_TOL = {"melatonin": 0.45, "dopamine": 0.12, "serotonin": 0.10, "histamine": 0.12}
_SANITY_ZERO_CAP = 0.25  # cortisol/CRH rest near zero; above this is stress, not rest


@dataclass
class DriftEntry:
    """One compared field."""

    field: str
    baseline: float | str
    current: float | str
    delta: float | str
    severity: str  # "ok" | "warn" | "fail"


@dataclass
class DriftReport:
    """Full comparison outcome."""

    entries: list[DriftEntry] = field(default_factory=list)
    passed: bool = True

    @property
    def failures(self) -> list[DriftEntry]:
        """Entries with severity == fail."""
        return [e for e in self.entries if e.severity == "fail"]

    @property
    def warnings(self) -> list[DriftEntry]:
        """Entries with severity == warn."""
        return [e for e in self.entries if e.severity == "warn"]

    def summary_lines(self) -> list[str]:
        """Human-readable report lines."""
        lines = [
            f"drift check: {'PASS' if self.passed else 'FAIL'} "
            f"({len(self.failures)} fail, {len(self.warnings)} warn, "
            f"{len(self.entries)} fields)"
        ]
        for e in self.entries:
            if e.severity == "ok":
                continue
            lines.append(
                f"  [{e.severity.upper():<4}] {e.field}: "
                f"baseline={e.baseline} current={e.current} delta={e.delta}"
            )
        return lines


def _band(value: float, warn: float, fail: float) -> str:
    if abs(value) >= fail:
        return "fail"
    if abs(value) >= warn:
        return "warn"
    return "ok"


def _fmt(value: float) -> float:
    return round(value, 5)


def compare_snapshots(baseline: Snapshot, current: Snapshot) -> DriftReport:
    """Compare a fresh snapshot against a stored baseline."""
    report = DriftReport()

    for name, base_val in baseline.chemicals.items():
        cur_val = current.chemicals.get(name)
        if cur_val is None:
            report.entries.append(DriftEntry(
                field=f"chem.{name}", baseline=base_val,
                current="missing", delta="missing", severity="fail",
            ))
            continue
        delta = cur_val - base_val
        if name in CIRCADIAN_SENSITIVE:
            sev = _band(delta, _CIRCADIAN_WARN[name], _CIRCADIAN_FAIL[name])
        else:
            sev = _band(delta, _CHEM_WARN, _CHEM_FAIL)
        report.entries.append(DriftEntry(
            field=f"chem.{name}", baseline=_fmt(base_val),
            current=_fmt(cur_val), delta=_fmt(delta), severity=sev,
        ))

    for sig, (warn, fail) in _SCALAR_BANDS.items():
        base_sig = getattr(baseline, sig, None)
        cur_sig = getattr(current, sig, None)
        if base_sig is None or cur_sig is None:
            continue
        delta = cur_sig - base_sig
        report.entries.append(DriftEntry(
            field=sig, baseline=_fmt(base_sig), current=_fmt(cur_sig),
            delta=_fmt(delta), severity=_band(delta, warn, fail),
        ))

    # Phase must match exactly — a resting-phase flip means the
    # threshold classifier or the underlying dynamics moved.
    phase_sev = "ok" if baseline.phase == current.phase else "fail"
    base_phase = PHASE_NAMES.get(baseline.phase, str(baseline.phase))
    cur_phase = PHASE_NAMES.get(current.phase, str(current.phase))
    report.entries.append(DriftEntry(
        field="phase", baseline=base_phase, current=cur_phase,
        delta="changed" if phase_sev == "fail" else "same", severity=phase_sev,
    ))

    report.passed = not report.failures
    return report


def check_snapshot_vs_defaults(snapshot: Snapshot) -> DriftReport:
    """Coarse sanity gate: resting levels vs genetic set-points."""
    report = DriftReport()
    for name, default in DEFAULT_BASELINES.items():
        cur = snapshot.chemicals.get(name)
        if cur is None:
            report.entries.append(DriftEntry(
                field=f"chem.{name}", baseline=default,
                current="missing", delta="missing", severity="fail",
            ))
            continue
        if default == 0.0:
            # Stress hormones rest at zero; anything above the cap
            # means the capture was not actually at rest.
            sev = "ok" if cur <= _SANITY_ZERO_CAP else "fail"
            delta = cur
        else:
            tol = _SANITY_CIRCADIAN_TOL.get(name, _SANITY_TOL)
            delta = cur - default
            sev = _band(delta, tol / 2.0, tol)
        report.entries.append(DriftEntry(
            field=f"chem.{name}", baseline=default,
            current=_fmt(cur), delta=_fmt(delta), severity=sev,
        ))
    if snapshot.coupling_drift >= 0.08:
        report.entries.append(DriftEntry(
            field="coupling_drift", baseline=0.0,
            current=_fmt(snapshot.coupling_drift), delta=_fmt(snapshot.coupling_drift),
            severity="fail",
        ))
    report.passed = not report.failures
    return report

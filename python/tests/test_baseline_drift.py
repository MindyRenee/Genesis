"""Offline tests for the baseline / drift harness.

No daemon required: these cover the default-baseline table, snapshot
serialization, and drift-comparison math. Live-daemon coverage lives
in test_chemical_assay_live.py.
"""

from __future__ import annotations

import pytest

from genesis_client.protocol import CHEM_NAMES, PHASE_ACTIVE, PHASE_STRESS
from genesis_cognitive.eval import (
    DEFAULT_BASELINES,
    Snapshot,
    check_snapshot_vs_defaults,
    compare_snapshots,
)
from genesis_cognitive.eval.assay import AssayResult, ChemicalEffect


def _resting_snapshot() -> Snapshot:
    """A healthy resting snapshot built from the genetic defaults."""
    return Snapshot(
        chemicals=dict(DEFAULT_BASELINES),
        arousal=0.11,
        valence=0.30,
        global_tone=0.15,
        plasticity_gate=0.45,
        encoding_weight=0.31,
        consolidation_weight=0.43,
        retrieval_weight=0.58,
        phase=PHASE_ACTIVE,
        phase_name="active",
    )


# ─── Default-baseline table ─────────────────────────────────────


def test_default_baselines_cover_all_protocol_chemicals() -> None:
    """Every protocol chemical has exactly one default baseline."""
    assert set(DEFAULT_BASELINES) == set(CHEM_NAMES.values())
    assert len(DEFAULT_BASELINES) == 18


def test_default_baselines_match_documented_values() -> None:
    """Spot-checks against NeurochemicalId::default_baseline()."""
    assert DEFAULT_BASELINES["glutamate"] == pytest.approx(0.60)
    assert DEFAULT_BASELINES["gaba"] == pytest.approx(0.50)
    assert DEFAULT_BASELINES["bdnf"] == pytest.approx(0.45)
    assert DEFAULT_BASELINES["serotonin"] == pytest.approx(0.40)
    assert DEFAULT_BASELINES["dopamine"] == pytest.approx(0.35)
    assert DEFAULT_BASELINES["cortisol"] == pytest.approx(0.0)
    assert DEFAULT_BASELINES["crh"] == pytest.approx(0.0)
    assert DEFAULT_BASELINES["melatonin"] == pytest.approx(0.10)


# ─── Snapshot serialization ─────────────────────────────────────


def test_snapshot_round_trip() -> None:
    """to_dict/from_dict preserves every field."""
    snap = _resting_snapshot()
    clone = Snapshot.from_dict(snap.to_dict())
    assert clone == snap


def test_snapshot_ignores_unknown_keys() -> None:
    """Forward compatibility: newer files still load."""
    data = _resting_snapshot().to_dict()
    data["future_field"] = "zzz"
    assert Snapshot.from_dict(data).arousal == pytest.approx(0.11)


def test_snapshot_save_load_tmp(tmp_path) -> None:
    """JSON save/load round-trips through disk."""
    path = str(tmp_path / "base.json")
    _resting_snapshot().save(path)
    loaded = Snapshot.load(path)
    assert loaded.chemicals["dopamine"] == pytest.approx(0.35)
    assert loaded == _resting_snapshot()


def test_assay_result_round_trip() -> None:
    """Assay matrix serialization preserves effects."""
    eff = ChemicalEffect(chem_id=0, name="dopamine", d_valence_slow=0.011)
    res = AssayResult(effects=[eff])
    clone = AssayResult.from_dict(res.to_dict())
    effect = clone.by_name("dopamine")
    assert effect is not None
    assert effect.d_valence_slow == pytest.approx(0.011)
    assert clone.by_name("missing") is None


# ─── Drift comparison ───────────────────────────────────────────


def test_identical_snapshots_pass() -> None:
    """No drift between identical captures."""
    report = compare_snapshots(_resting_snapshot(), _resting_snapshot())
    assert report.passed
    assert not report.failures
    assert not report.warnings


def test_small_jitter_is_ok_or_warn_not_fail() -> None:
    """Sub-threshold jitter never fails."""
    current = _resting_snapshot()
    current.chemicals["dopamine"] += 0.02  # below warn band (0.05 circadian)
    current.arousal += 0.01
    report = compare_snapshots(_resting_snapshot(), current)
    assert report.passed


def test_large_chemical_shift_fails_with_field_name() -> None:
    """A +0.5 dopamine jump fails and names the field."""
    current = _resting_snapshot()
    current.chemicals["dopamine"] += 0.5
    report = compare_snapshots(_resting_snapshot(), current)
    assert not report.passed
    assert any(e.field == "chem.dopamine" for e in report.failures)


def test_missing_chemical_fails() -> None:
    """A dropped chemical field is a failure, not a silent skip."""
    current = _resting_snapshot()
    del current.chemicals["cortisol"]
    report = compare_snapshots(_resting_snapshot(), current)
    assert not report.passed
    assert any(e.field == "chem.cortisol" for e in report.failures)


def test_phase_flip_fails() -> None:
    """A resting-phase change always fails, however small the levels moved."""
    current = _resting_snapshot()
    current.phase = PHASE_STRESS
    report = compare_snapshots(_resting_snapshot(), current)
    assert not report.passed
    assert any(e.field == "phase" for e in report.failures)


def test_sanity_vs_defaults_passes_at_rest() -> None:
    """Genetic-default sanity gate passes for a healthy resting snapshot."""
    assert check_snapshot_vs_defaults(_resting_snapshot()).passed


def test_sanity_vs_defaults_flags_stress_capture() -> None:
    """A cortisol-flooded capture is not rest."""
    snap = _resting_snapshot()
    snap.chemicals["cortisol"] = 0.8
    report = check_snapshot_vs_defaults(snap)
    assert not report.passed
    assert any(e.field == "chem.cortisol" for e in report.failures)

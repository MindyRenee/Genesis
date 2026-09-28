"""Live-daemon tests: resting baseline + per-chemical causal assay.

These spawn an isolated throwaway daemon (temporary data directory,
never the production instance), settle it, and measure real dynamics
over IPC. Skipped when the daemon binary is not built.

Run: pytest python/tests/test_chemical_assay_live.py -o addopts=''
"""

from __future__ import annotations

import math

import pytest

from genesis_client.protocol import CHEM_NAMES, PHASE_NAMES
from genesis_cognitive.eval import (
    check_snapshot_vs_defaults,
    compare_snapshots,
    isolated_daemon,
    run_chemical_assay,
)
from genesis_cognitive.eval.assay import ASSAY_DT
from genesis_cognitive.eval.baseline import Snapshot, capture_snapshot
from genesis_cognitive.eval.harness import find_daemon_binary, settle

pytestmark = pytest.mark.slow

SETTLE = 60

# Impulses to these chemicals are absorbed by design, not by bug:
# adenosine is governed by the sleep-pressure accumulation/clearance
# mechanism (not the level variable the impulse writes to), and
# melatonin is set by the circadian oscillator each tick. Both show
# near-zero fast response; every other chemical must show the dose.
ABSORBING = frozenset({"adenosine", "melatonin"})


def _require_daemon() -> None:
    if find_daemon_binary() is None:
        pytest.skip("genesis-daemon binary not built (run: cargo build --release)")


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def test_resting_state_passes_sanity_gate() -> None:
    """A settled daemon rests near its genetic defaults with valid ranges."""
    _require_daemon()
    with isolated_daemon() as client:
        snap = capture_snapshot(client, settle_ticks=SETTLE, settle_dt=ASSAY_DT)
    assert check_snapshot_vs_defaults(snap).passed
    assert snap.phase in PHASE_NAMES
    assert _finite(
        [snap.arousal, snap.valence, snap.plasticity_gate, *snap.chemicals.values()]
    )
    assert 0.0 <= snap.arousal <= 1.0
    assert -1.0 <= snap.valence <= 1.0


def test_repeated_captures_do_not_drift() -> None:
    """Two back-to-back captures of the same daemon compare clean."""
    _require_daemon()
    with isolated_daemon() as client:
        first = capture_snapshot(client, settle_ticks=SETTLE, settle_dt=ASSAY_DT)
        settle(client, ticks=20, dt=ASSAY_DT)
        second = capture_snapshot(client, settle_ticks=0, settle_dt=ASSAY_DT)
    report = compare_snapshots(first, second)
    assert report.passed, report.summary_lines()


def test_dopamine_impulse_is_rewarding() -> None:
    """Dopamine raises its own level and valence (reward direction)."""
    _require_daemon()
    with isolated_daemon() as client:
        settle(client, ticks=SETTLE, dt=ASSAY_DT)
        pre = client.get_state()
        assert client.neuro_impulse(0, 0.30)
        client.advance_neuro(dt=ASSAY_DT)
        post = client.get_state()
    assert post.chemicals["dopamine"] - pre.chemicals["dopamine"] > 0.05
    assert post.valence - pre.valence > 0.0


def test_cortisol_impulse_is_aversive() -> None:
    """Cortisol raises its own level and lowers valence (stress direction)."""
    _require_daemon()
    with isolated_daemon() as client:
        settle(client, ticks=SETTLE, dt=ASSAY_DT)
        pre = client.get_state()
        assert client.neuro_impulse(6, 0.30)
        client.advance_neuro(dt=ASSAY_DT)
        post = client.get_state()
    assert post.chemicals["cortisol"] - pre.chemicals["cortisol"] > 0.05
    assert post.valence - pre.valence < 0.0


def test_crh_cascade_is_maturation_gated_on_fresh_daemon() -> None:
    """CRH raises itself but not cortisol on a fresh daemon (SHRP dormancy).

    The HPA cascade CRH → ACTH → cortisol is maturation-gated
    (stress hyporesponsive period, ``src/state/neurochemical.rs``):
    a young instance correctly shows almost no cortisol response to
    CRH. Later in life (e.g. late in a full assay run) the same probe
    does drive cortisol — see the assay matrix, not this test.
    """
    _require_daemon()
    with isolated_daemon() as client:
        settle(client, ticks=SETTLE, dt=ASSAY_DT)
        pre = client.get_state()
        assert client.neuro_impulse(14, 0.30)
        for _ in range(10):
            client.advance_neuro(dt=ASSAY_DT)
        post = client.get_state()
    assert post.chemicals["crh"] - pre.chemicals["crh"] > 0.0
    assert abs(post.chemicals["cortisol"] - pre.chemicals["cortisol"]) < 0.02


def test_full_assay_covers_all_chemicals_and_stays_stable() -> None:
    """All 18 chemicals respond (or absorb by design); state stays finite."""
    _require_daemon()
    with isolated_daemon() as client:
        result = run_chemical_assay(client)
    assert len(result.effects) == 18
    assert {e.name for e in result.effects} == set(CHEM_NAMES.values())
    for e in result.effects:
        assert e.phase_pre in PHASE_NAMES.values()
        assert e.phase_slow in PHASE_NAMES.values()
        assert _finite([
            e.d_arousal_slow, e.d_valence_slow, e.d_plasticity_slow,
            e.pre_self, e.post_fast_self, e.post_slow_self,
        ])
        d_fast = e.post_fast_self - e.pre_self
        if e.name in ABSORBING:
            assert abs(d_fast) < 0.05, f"{e.name} should absorb, got {d_fast:+.3f}"
        else:
            assert d_fast > 0.05, f"{e.name} showed no direct response ({d_fast:+.3f})"


def test_assay_snapshot_round_trips_through_json(tmp_path) -> None:
    """Baseline JSON files are stable artifacts for drift checks."""
    _require_daemon()
    with isolated_daemon() as client:
        snap = capture_snapshot(client, settle_ticks=SETTLE, settle_dt=ASSAY_DT)
    path = str(tmp_path / "baseline.json")
    snap.save(path)
    assert Snapshot.load(path) == snap

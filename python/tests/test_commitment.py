"""Commitment boundary tests — continuous signal → discrete state.

The CommitmentBoundary is the digitizer primitive: hysteresis
(enter/release deadband), sustained crossing (confirm_s), and
refractory dwell. Also covers the activation-field noise floor —
the per-tick re-digitization of the concept network's analog state.
"""

import pytest

from genesis_cognitive.commitment import CommitmentBoundary
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.concepts.dynamics import (
    SPARK_MIN,
    SWEEP_INTERVAL_TICKS,
)

# ─── Construction / validation ───────────────────────────────────


def test_release_above_enter_rejected() -> None:
    with pytest.raises(ValueError, match="deadband"):
        CommitmentBoundary(enter=0.5, release=0.6)


def test_negative_confirm_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        CommitmentBoundary(enter=0.5, release=0.4, confirm_s=-1.0)


def test_equal_enter_release_allowed() -> None:
    b = CommitmentBoundary(enter=0.5, release=0.5)
    assert not b.committed


# ─── Commit semantics ────────────────────────────────────────────


def test_immediate_commit_when_no_confirm() -> None:
    """confirm_s=0 reproduces legacy single-crossing behavior."""
    b = CommitmentBoundary(enter=0.5, release=0.4)
    assert b.update(0.6, now=0.0) is True
    assert b.committed
    assert b.just_committed


def test_single_blip_does_not_commit() -> None:
    """A transient spike above enter cannot commit — the flytrap rule."""
    b = CommitmentBoundary(enter=0.5, release=0.4, confirm_s=30.0)
    b.update(0.9, now=0.0)  # spike
    b.update(0.2, now=1.0)  # gone
    assert not b.committed
    assert not b.just_committed


def test_sustained_crossing_commits() -> None:
    b = CommitmentBoundary(enter=0.5, release=0.4, confirm_s=10.0)
    assert b.update(0.6, now=0.0) is False  # crossing begins
    assert b.update(0.6, now=9.9) is False  # still short
    assert b.update(0.6, now=10.0) is True
    assert b.just_committed


def test_dip_below_enter_restarts_confirm() -> None:
    """Sustained means sustained — a mid-window dip resets the clock."""
    b = CommitmentBoundary(enter=0.5, release=0.4, confirm_s=10.0)
    b.update(0.6, now=0.0)
    b.update(0.3, now=5.0)  # dips below enter — window restarts
    b.update(0.6, now=12.0)  # new crossing begins
    assert b.update(0.6, now=21.9) is False  # 9.9s into new window
    assert b.update(0.6, now=22.0) is True  # 10s sustained


def test_commit_edge_fires_once() -> None:
    """just_committed is an edge, not a level."""
    b = CommitmentBoundary(enter=0.5, release=0.4)
    assert b.update(0.6, now=0.0) is True
    assert b.just_committed
    assert b.update(0.9, now=1.0) is True  # still committed
    assert not b.just_committed  # but the edge is gone


# ─── Release semantics (deadband + refractory) ───────────────────


def test_deadband_holds_committed() -> None:
    """Inside the enter/release band the state holds — no flicker."""
    b = CommitmentBoundary(enter=0.5, release=0.4)
    b.update(0.6, now=0.0)
    assert b.update(0.45, now=10.0) is True  # in deadband — holds
    assert b.update(0.49, now=20.0) is True
    assert b.committed


def test_release_below_threshold() -> None:
    b = CommitmentBoundary(enter=0.5, release=0.4)
    b.update(0.6, now=0.0)
    assert b.update(0.4, now=10.0) is False
    assert b.just_released
    assert not b.just_committed


def test_refractory_blocks_early_release() -> None:
    b = CommitmentBoundary(enter=0.5, release=0.4, refractory_s=30.0)
    b.update(0.6, now=0.0)
    # Signal collapses immediately — refractory keeps it latched
    assert b.update(0.0, now=5.0) is True
    assert b.update(0.0, now=31.0) is False


def test_rearm_after_release() -> None:
    """After release, a fresh sustained crossing commits again."""
    b = CommitmentBoundary(enter=0.5, release=0.4, confirm_s=5.0)
    b.update(0.6, now=0.0)
    b.update(0.6, now=5.0)
    assert b.committed
    b.update(0.1, now=10.0)
    assert not b.committed
    assert b.update(0.6, now=20.0) is False
    assert b.update(0.6, now=25.0) is True


def test_reset_clears_state() -> None:
    b = CommitmentBoundary(enter=0.5, release=0.4)
    b.update(0.6, now=0.0)
    b.reset()
    assert not b.committed
    assert not b.just_committed
    # Fresh crossing required after reset
    assert b.update(0.6, now=10.0) is True
    assert b.just_committed


# ─── Activation-field noise floor (mid-layer re-digitization) ────


def test_tick_snaps_subfloor_activation() -> None:
    """Sub-floor activation is residue — snapped to clean zero."""
    net = ConceptNetwork()
    net.add_concept("residue", confidence=0.8)
    net.add_concept("signal", confidence=0.8)
    residue = net.get_concept("residue")
    signal = net.get_concept("signal")
    assert residue is not None and signal is not None
    residue.activation = 0.003  # below ACTIVATION_NOISE_FLOOR (0.005)
    signal.activation = 0.5

    net.cortical_tick(arousal=0.3, gaba=0.5, ach=0.1, serotonin=0.5, noise=0.0)

    assert residue.activation == 0.0
    assert 0.0 < signal.activation < 0.5  # decayed but alive


def test_tick_leaves_no_subfloor_tail() -> None:
    """After any tick, no concept sits in the (0, floor) dead zone."""
    net = ConceptNetwork()
    net.add_concept("a", confidence=0.8)
    net.add_concept("b", confidence=0.8)
    a = net.get_concept("a")
    b = net.get_concept("b")
    assert a is not None and b is not None
    a.activation = 0.4
    b.activation = 0.002

    for _ in range(20):
        net.cortical_tick(arousal=0.5, gaba=0.3, ach=0.2, serotonin=0.5, noise=0.0)
        for concept in net._concepts.values():
            assert concept.activation == 0.0 or concept.activation >= 0.005


# ─── Sparse tick — active set, ignition, sweep ───────────────────


def test_active_set_built_on_first_tick() -> None:
    """The sparse active set materializes lazily on first tick."""
    net = ConceptNetwork()
    net.add_concept("hot", confidence=0.8)
    net.add_concept("cold", confidence=0.8)
    hot = net.get_concept("hot")
    cold = net.get_concept("cold")
    assert hot is not None and cold is not None
    hot.activation = 0.5
    cold.activation = 0.0

    assert net._active_ids is None  # not built until first tick
    net.cortical_tick(noise=0.0)
    assert net._active_ids is not None
    assert "hot" in net._active_ids
    assert "cold" not in net._active_ids


def test_spread_target_joins_active_set() -> None:
    """A dormant concept receiving spread activation enters the set."""
    net = ConceptNetwork()
    net.add_concept("src", confidence=0.8)
    net.add_concept("dst", confidence=0.8)
    net.add_edge("src", "dst", RelationType.RELATED_TO, weight=0.9)
    src = net.get_concept("src")
    dst = net.get_concept("dst")
    assert src is not None and dst is not None
    src.activation = 0.9
    dst.activation = 0.0

    net.cortical_tick(arousal=0.7, gaba=0.1, ach=0.2, serotonin=0.5, noise=0.0)
    assert net._active_ids is not None
    assert "dst" in net._active_ids
    assert dst.activation > 0.0


def test_snapped_concept_leaves_active_set() -> None:
    net = ConceptNetwork()
    net.add_concept("fading", confidence=0.8)
    fading = net.get_concept("fading")
    assert fading is not None
    fading.activation = 0.003  # sub-floor

    net.cortical_tick(noise=0.0)
    assert fading.activation == 0.0
    assert net._active_ids is not None
    assert "fading" not in net._active_ids


def test_mark_active_registers_external_write() -> None:
    """A direct activation write joins the set via _mark_active."""
    net = ConceptNetwork()
    net.add_concept("bumped", confidence=0.8)
    net.cortical_tick(noise=0.0)  # builds the (empty) set
    bumped = net.get_concept("bumped")
    assert bumped is not None
    bumped.activation = 0.4  # write outside the tick loop
    net._mark_active("bumped")
    assert net._active_ids is not None
    assert "bumped" in net._active_ids


def test_dormant_ignition_fires_at_forced_rate() -> None:
    """With spark_rate=1.0 the Poisson draw ignites dormant concepts."""
    net = ConceptNetwork()
    for i in range(50):
        net.add_concept(f"dormant{i}", confidence=0.8)

    net.cortical_tick(noise=0.01, spark_rate=1.0)

    sparked = [
        c for c in net._concepts.values()
        if c.activation is not None and c.activation >= SPARK_MIN
    ]
    assert len(sparked) >= 5, (
        f"forced ignition should light many dormant concepts, "
        f"got {len(sparked)}"
    )
    assert net._active_ids is not None
    assert len(net._active_ids) == len(sparked)


def test_no_ignition_when_noise_zero() -> None:
    """noise=0 must mean a quiet dormant field — no sparks."""
    net = ConceptNetwork()
    for i in range(20):
        net.add_concept(f"quiet{i}", confidence=0.8)
    # add_concept seeds new concepts at 0.5 activation — silence the
    # field first so ignition is the only way anything could light.
    for c in net._concepts.values():
        c.activation = 0.0
    for _ in range(5):
        net.cortical_tick(noise=0.0)
    assert all(
        (c.activation or 0.0) == 0.0 for c in net._concepts.values()
    )


def test_sweep_rebuilds_set_and_clears_residue() -> None:
    """The periodic sweep catches unmarked writes and snaps residue."""
    net = ConceptNetwork()
    net.add_concept("unmarked", confidence=0.8)
    net.add_concept("residue", confidence=0.8)
    unmarked = net.get_concept("unmarked")
    residue = net.get_concept("residue")
    assert unmarked is not None and residue is not None
    unmarked.activation = 0.0  # dormant — excluded when the set builds
    net.cortical_tick(noise=0.0)  # build set: residue in, unmarked out
    assert net._active_ids is not None
    assert "unmarked" not in net._active_ids
    unmarked.activation = 0.5   # written without _mark_active
    residue.activation = 0.002  # sub-floor residue while still in the set

    net._ticks_since_sweep = SWEEP_INTERVAL_TICKS - 1
    net.cortical_tick(noise=0.0)  # this tick sweeps

    assert net._active_ids is not None
    assert "unmarked" in net._active_ids
    assert residue.activation == 0.0
    assert "residue" not in net._active_ids

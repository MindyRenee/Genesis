"""Consolidated tests for the sleep bundle's public API.

Covers:
- Sleep stage classification (SleepStage enum)
- Hypnagogic wake→sleep transition (HypnagogicState)
- Post-wake sleep inertia (SleepInertia)
- The 90-minute ultradian cycle (SleepCycleTracker: N1→N2→N3→N2→REM)
- Slow-wave sleep synaptic downscaling (SynapticDownscaler / SHY hypothesis)
- Hippocampal replay during N3 (HippocampalReplay)
- InnerLife integration of the sleep architecture
- Sleep-state persistence/serialization across restarts

All imports come from the sleep bundle's public API
(``genesis_cognitive.sleep``) or the package ``__init__``
(``genesis_cognitive``).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from genesis_cognitive import (
    ConceptNetwork,
    CuriosityEngine,
    NarrativeEngine,
    ReasoningEngine,
    ReflectionEngine,
    RelationType,
    SelfModel,
)
from genesis_cognitive.persistence import (
    load_state,
    restore_sleep_cycle,
    save_state,
    serialize_sleep_state,
)
from genesis_cognitive.sleep import (
    HippocampalReplay,
    HypnagogicState,
    InnerLife,
    SleepCycleTracker,
    SleepInertia,
    SleepStage,
    SynapticDownscaler,
)

# ═══════════════════════════════════════════════════════════════════
# SleepStage enum
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "stage, is_rem, is_nrem",
    [
        (SleepStage.REM, True, False),
        (SleepStage.N1, False, True),
        (SleepStage.N2, False, True),
        (SleepStage.N3, False, True),
    ],
)
def test_sleep_stage_properties(stage, is_rem, is_nrem) -> None:
    """is_rem/is_nrem classify stages correctly."""
    assert stage.is_rem is is_rem
    assert stage.is_nrem is is_nrem


# ═══════════════════════════════════════════════════════════════════
# HypnagogicState
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "progress, is_active, is_complete",
    [
        (0.0, False, False),
        (0.5, True, False),
        (1.0, False, True),
    ],
)
def test_hypnagogic_state_properties(progress, is_active, is_complete) -> None:
    """is_active/is_complete classify progress correctly."""
    state = HypnagogicState(progress=progress)
    assert state.is_active is is_active
    assert state.is_complete is is_complete


# ═══════════════════════════════════════════════════════════════════
# SleepInertia
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "remaining_seconds, is_active",
    [
        (100.0, True),
        (0.0, False),
    ],
)
def test_sleep_inertia_properties(remaining_seconds, is_active) -> None:
    """is_active is True only when remaining_seconds > 0."""
    inertia = SleepInertia(remaining_seconds=remaining_seconds, severity=0.5)
    assert inertia.is_active is is_active


def test_sleep_inertia_cognitive_impairment() -> None:
    """cognitive_impairment scales with severity and time."""
    inertia = SleepInertia(remaining_seconds=300.0, severity=0.8)
    assert inertia.cognitive_impairment > 0.0
    assert inertia.cognitive_impairment <= 1.0


def test_sleep_inertia_cognitive_impairment_zero_when_inactive() -> None:
    """cognitive_impairment is 0 when not active."""
    inertia = SleepInertia()
    assert inertia.cognitive_impairment == 0.0


# ─── Helpers ──────────────────────────────────────────────────────


def _make_network() -> ConceptNetwork:
    """Build a small concept network for testing."""
    net = ConceptNetwork()
    for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
        net.add_concept(name, confidence=0.5)
    net.add_edge("alpha", "beta", RelationType.RELATED_TO, 0.8)
    net.add_edge("beta", "gamma", RelationType.RELATED_TO, 0.5)
    net.add_edge("gamma", "delta", RelationType.RELATED_TO, 0.3)
    return net


# ─── 1. SleepCycleTracker ─────────────────────────────────────────


class TestSleepCycleTracker:
    """Tests for the 90-minute ultradian cycle tracker."""

    def test_initial_state(self):
        """Tracker starts at cycle 1, stage N1, slot 0."""
        tracker = SleepCycleTracker()
        assert tracker.current_stage is SleepStage.N1
        assert tracker.cycle_number == 1
        assert tracker.cycles_completed == 0
        assert tracker.time_in_stage == 0.0
        assert tracker.time_in_cycle == 0.0

    def test_n1_to_n2_transition(self):
        """N1 (~5 min) transitions to N2."""
        tracker = SleepCycleTracker()
        stage = tracker.advance(300.0)
        assert stage is SleepStage.N2

    def test_full_cycle_progression(self):
        """A full cycle progresses N1→N2→N3→N2→REM→(new cycle N1)."""
        tracker = SleepCycleTracker()
        # N1 (300s)
        assert tracker.advance(300.0) is SleepStage.N2
        # N2 descent (900s in cycle 1)
        assert tracker.advance(900.0) is SleepStage.N3
        # N3 (2400s in cycle 1)
        assert tracker.advance(2400.0) is SleepStage.N2
        # N2 ascent (shorter: 900*0.4 = 360s)
        assert tracker.advance(360.0) is SleepStage.REM
        # REM (600s in cycle 1) → new cycle
        stage = tracker.advance(600.0)
        assert stage is SleepStage.N1
        assert tracker.cycles_completed == 1
        assert tracker.cycle_number == 2

    def test_n3_decreases_in_later_cycles(self):
        """N3 duration is longer in cycle 1 than in later cycles."""
        tracker = SleepCycleTracker()
        n3_cycle1 = tracker._stage_duration(SleepStage.N3, 2)
        tracker.cycle_number = 4
        n3_cycle4 = tracker._stage_duration(SleepStage.N3, 2)
        assert n3_cycle1 > n3_cycle4

    def test_rem_increases_in_later_cycles(self):
        """REM duration is shorter in cycle 1 than in later cycles."""
        tracker = SleepCycleTracker()
        rem_cycle1 = tracker._stage_duration(SleepStage.REM, 4)
        tracker.cycle_number = 5
        rem_cycle5 = tracker._stage_duration(SleepStage.REM, 4)
        assert rem_cycle5 > rem_cycle1

    def test_n2_ascent_shorter_than_descent(self):
        """The ascent N2 (slot 3) is shorter than the descent N2 (slot 1)."""
        tracker = SleepCycleTracker()
        n2_descent = tracker._stage_duration(SleepStage.N2, 1)
        n2_ascent = tracker._stage_duration(SleepStage.N2, 3)
        assert n2_ascent < n2_descent

    def test_cycle_progress_increases(self):
        """cycle_progress increases as time advances."""
        tracker = SleepCycleTracker()
        p0 = tracker.cycle_progress
        tracker.advance(100.0)
        p1 = tracker.cycle_progress
        assert p1 > p0

    def test_stage_progress_increases(self):
        """stage_progress increases as time in stage advances."""
        tracker = SleepCycleTracker()
        tracker.advance(100.0)
        assert 0.0 < tracker.stage_progress < 1.0

    def test_reset(self):
        """reset() returns the tracker to cycle 1, N1."""
        tracker = SleepCycleTracker()
        tracker.advance(500.0)
        tracker.reset()
        assert tracker.current_stage is SleepStage.N1
        assert tracker.cycle_number == 1
        assert tracker.time_in_stage == 0.0

    def test_advance_zero_dt(self):
        """advance(0) does not change the stage."""
        tracker = SleepCycleTracker()
        stage = tracker.advance(0.0)
        assert stage is SleepStage.N1
        assert tracker.time_in_stage == 0.0

    def test_describe(self):
        """describe() returns a human-readable string."""
        tracker = SleepCycleTracker()
        tracker.advance(100.0)
        desc = tracker.describe()
        assert "sleep-cycle" in desc
        assert "N1" in desc


# ─── 2. SynapticDownscaler (SHY) ──────────────────────────────────


class TestSynapticDownscaler:
    """Tests for the SHY synaptic downscaling mechanism."""

    def test_accumulate_synaptic_load(self):
        """accumulate_synaptic_load increases the load."""
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(0.5)
        assert ds.get_synaptic_load() == pytest.approx(0.5)
        ds.accumulate_synaptic_load(0.3)
        assert ds.get_synaptic_load() == pytest.approx(0.8)

    def test_accumulate_zero_or_negative_ignored(self):
        """Zero or negative amounts do not increase load."""
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(0.0)
        ds.accumulate_synaptic_load(-1.0)
        assert ds.get_synaptic_load() == 0.0

    def test_downscale_reduces_weights(self):
        """downscale_during_n3 reduces edge weights."""
        net = _make_network()
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(2.0)
        weights_before = [e.weight for e in net.edges]
        removed = ds.downscale_during_n3(net, 10.0)
        weights_after = [e.weight for e in net.edges]
        assert removed > 0.0
        for wb, wa in zip(weights_before, weights_after, strict=False):
            assert wa < wb

    def test_downscale_stronger_synapses_more(self):
        """Stronger connections are downscaled proportionally more."""
        net = ConceptNetwork()
        net.add_concept("a")
        net.add_concept("b")
        net.add_concept("c")
        net.add_edge("a", "b", RelationType.RELATED_TO, 0.9)
        net.add_edge("a", "c", RelationType.RELATED_TO, 0.3)
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(2.0)
        strong_before = net.edges[0].weight
        weak_before = net.edges[1].weight
        ds.downscale_during_n3(net, 10.0)
        strong_after = net.edges[0].weight
        weak_after = net.edges[1].weight
        strong_reduction = strong_before - strong_after
        weak_reduction = weak_before - weak_after
        assert strong_reduction > weak_reduction

    def test_downscale_decreases_load(self):
        """Downscaling consumes synaptic load (homeostatic loop)."""
        net = _make_network()
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(2.0)
        load_before = ds.get_synaptic_load()
        ds.downscale_during_n3(net, 10.0)
        load_after = ds.get_synaptic_load()
        assert load_after < load_before

    def test_downscale_no_load_is_noop(self):
        """With zero synaptic load, downscaling is a no-op."""
        net = _make_network()
        ds = SynapticDownscaler()
        weights_before = [e.weight for e in net.edges]
        removed = ds.downscale_during_n3(net, 10.0)
        weights_after = [e.weight for e in net.edges]
        assert removed == 0.0
        assert weights_before == weights_after

    def test_total_downscaling_tracked(self):
        """Total downscaling is accumulated across calls."""
        net = _make_network()
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(2.0)
        ds.downscale_during_n3(net, 5.0)
        total1 = ds.total_downscaling
        ds.downscale_during_n3(net, 5.0)
        total2 = ds.total_downscaling
        assert total2 > total1

    def test_load_history_recorded(self):
        """Load snapshots are recorded in history."""
        net = _make_network()
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(2.0)
        ds.downscale_during_n3(net, 10.0)
        assert len(ds.load_history) > 0

    def test_describe(self):
        """describe() returns a human-readable string."""
        ds = SynapticDownscaler()
        ds.accumulate_synaptic_load(1.0)
        desc = ds.describe()
        assert "shy" in desc
        assert "load" in desc


# ─── 3. HippocampalReplay ─────────────────────────────────────────


class TestHippocampalReplay:
    """Tests for hippocampal replay during N3."""

    def test_queue_for_replay(self):
        """queue_for_replay adds sequences to the queue."""
        replay = HippocampalReplay()
        replay.queue_for_replay(["a", "b", "c"], 0.8)
        assert replay.queue_size == 1

    def test_queue_empty_sequence_ignored(self):
        """Empty sequences are not queued."""
        replay = HippocampalReplay()
        replay.queue_for_replay([], 0.5)
        assert replay.queue_size == 0

    def test_replay_strengthens_connections(self):
        """Replay creates/strengthens edges between co-active concepts."""
        net = ConceptNetwork()
        for name in ("x", "y", "z"):
            net.add_concept(name, confidence=0.5)
        replay = HippocampalReplay()
        replay.queue_for_replay(["x", "y", "z"], 0.8)
        replay.replay_during_n3(net, 1.0)
        neighbors = net.get_neighbors("x")
        neighbor_ids = {n[0] for n in neighbors}
        assert "y" in neighbor_ids

    def test_replay_boosts_confidence(self):
        """Replay boosts concept confidence (systems consolidation)."""
        net = ConceptNetwork()
        for name in ("x", "y", "z"):
            net.add_concept(name, confidence=0.5)
        replay = HippocampalReplay()
        replay.queue_for_replay(["x", "y", "z"], 0.8)
        replay.replay_during_n3(net, 1.0)
        concept = net.get_concept("x")
        assert concept is not None
        assert concept.confidence > 0.5

    def test_replay_count_increments(self):
        """get_replay_count increments with each replay burst."""
        net = ConceptNetwork()
        for name in ("x", "y", "z"):
            net.add_concept(name, confidence=0.5)
        replay = HippocampalReplay()
        replay.queue_for_replay(["x", "y", "z"], 0.8)
        count_before = replay.get_replay_count()
        replay.replay_during_n3(net, 1.0)
        count_after = replay.get_replay_count()
        assert count_after > count_before

    def test_replay_history_tracked(self):
        """Replay history records which sequences were replayed."""
        net = ConceptNetwork()
        for name in ("x", "y", "z"):
            net.add_concept(name, confidence=0.5)
        replay = HippocampalReplay()
        replay.queue_for_replay(["x", "y", "z"], 0.8)
        replay.replay_during_n3(net, 1.0)
        history = replay.replay_history
        assert len(history) > 0
        # The signature should contain the concept names
        any_sig = next(iter(history.keys()))
        assert "x" in any_sig

    def test_high_salience_more_replays(self):
        """High-salience sequences survive more replays than low."""
        net_hi = ConceptNetwork()
        for name in ("a", "b", "c"):
            net_hi.add_concept(name, confidence=0.5)
        net_lo = ConceptNetwork()
        for name in ("a", "b", "c"):
            net_lo.add_concept(name, confidence=0.5)

        replay_hi = HippocampalReplay()
        replay_hi.queue_for_replay(["a", "b", "c"], 1.0)
        replay_lo = HippocampalReplay()
        replay_lo.queue_for_replay(["a", "b", "c"], 0.1)

        # Give enough time for many replays
        replay_hi.replay_during_n3(net_hi, 10.0)
        replay_lo.replay_during_n3(net_lo, 10.0)
        assert replay_hi.get_replay_count() >= replay_lo.get_replay_count()

    def test_empty_queue_is_noop(self):
        """Replay with an empty queue returns nothing."""
        net = _make_network()
        replay = HippocampalReplay()
        result = replay.replay_during_n3(net, 1.0)
        assert result == []
        assert replay.get_replay_count() == 0

    def test_compressed_timescale(self):
        """Replay processes multiple bursts per second (20x compression)."""
        net = ConceptNetwork()
        for i in range(20):
            net.add_concept(f"c{i}", confidence=0.5)
        replay = HippocampalReplay()
        # Queue many items
        for i in range(15):
            replay.queue_for_replay([f"c{i}", f"c{i + 1}"], 0.1)
        replayed = replay.replay_during_n3(net, 1.0)
        # With 20x compression and 0.5s per burst, 1s → 40 bursts max
        # but limited by queue size (15)
        assert len(replayed) <= 15

    def test_describe(self):
        """describe() returns a human-readable string."""
        replay = HippocampalReplay()
        replay.queue_for_replay(["a", "b"], 0.5)
        desc = replay.describe()
        assert "replay" in desc


# ─── InnerLife integration ────────────────────────────────────────


class TestInnerLifeSleepArchitecture:
    """Tests for InnerLife integration of the sleep architecture."""

    def _make_inner_life(self) -> InnerLife:
        """Build an InnerLife with a small network and no callbacks."""
        net = _make_network()
        reasoning = ReasoningEngine(net)
        curiosity = CuriosityEngine(net, reasoning)
        reflection = ReflectionEngine(net)
        return InnerLife(
            net,
            curiosity,
            reflection,
            seed=42,
        )

    def test_synaptic_load_property(self):
        """synaptic_load property reflects the downscaler."""
        il = self._make_inner_life()
        assert il.synaptic_load == 0.0
        il.accumulate_synaptic_load(1.5)
        assert il.synaptic_load == pytest.approx(1.5)

    def test_queue_for_replay_method(self):
        """queue_for_replay delegates to the replay engine."""
        il = self._make_inner_life()
        assert il.replay_queue_size == 0
        il.queue_for_replay(["alpha", "beta"], 0.7)
        assert il.replay_queue_size == 1

    def test_sleep_cycle_starts_none(self):
        """sleep_cycle is None before sleep begins."""
        il = self._make_inner_life()
        assert il.sleep_cycle is None
        assert il.cycles_completed == 0

    def test_replay_count_property(self):
        """replay_count property reflects the replay engine."""
        il = self._make_inner_life()
        assert il.replay_count == 0

    def test_total_synaptic_downscaling_property(self):
        """total_synaptic_downscaling starts at zero."""
        il = self._make_inner_life()
        assert il.total_synaptic_downscaling == 0.0


# ═══════════════════════════════════════════════════════════════════
# Sleep-state persistence / serialization
# ═══════════════════════════════════════════════════════════════════
#
# The daemon's mmap state (core_state.bin) already persists neurochemicals
# and emergent_phase (nrem/rem). These tests verify the Python-side sleep
# state — the _is_sleeping flag, _user_initiated_sleep flag, and the
# ultradian SleepCycleTracker position (N1→N2→N3→N2→REM) — also survives
# save/load.
#
# Without this, a restart during N3 would:
# - Force her awake (Mind.start set ZONE_CONVERSATION)
# - Clear adenosine (startup wake cascade)
# - Lose her place in the sleep cycle (tracker reset to N1)
# - Skip pending N3 consolidation


def test_serialize_sleep_state_awake():
    """Serializing awake state should have is_sleeping=False and no cycle."""
    data = serialize_sleep_state(
        is_sleeping=False,
        user_initiated_sleep=False,
        sleep_cycle=None,
    )
    assert data["is_sleeping"] is False
    assert data["user_initiated_sleep"] is False
    assert data["sleep_cycle"] is None


def test_serialize_sleep_state_sleeping_n3():
    """Serializing N3 sleep should capture the cycle position."""
    tracker = SleepCycleTracker()
    # Advance to N3: N1 (300s) → N2 (900s) → N3
    tracker.advance(300.0)  # N1 → N2
    tracker.advance(900.0)  # N2 → N3
    tracker.advance(100.0)  # 100s into N3

    assert tracker.current_stage is SleepStage.N3

    data = serialize_sleep_state(
        is_sleeping=True,
        user_initiated_sleep=True,
        sleep_cycle=tracker,
    )
    assert data["is_sleeping"] is True
    assert data["user_initiated_sleep"] is True
    cycle = data["sleep_cycle"]
    assert cycle is not None
    assert cycle["current_stage"] == "n3"
    assert cycle["cycle_number"] == 1
    assert cycle["time_in_stage"] == 100.0
    assert cycle["cycles_completed"] == 0


def test_restore_sleep_cycle_n3():
    """Restoring a saved N3 cycle should resume at N3 with correct position."""
    tracker = SleepCycleTracker()
    tracker.advance(300.0)  # N1 → N2
    tracker.advance(900.0)  # N2 → N3
    tracker.advance(250.0)  # 250s into N3

    data = serialize_sleep_state(
        is_sleeping=True,
        user_initiated_sleep=False,
        sleep_cycle=tracker,
    )

    # Create a fresh tracker and restore
    fresh = SleepCycleTracker()
    assert fresh.current_stage is SleepStage.N1  # starts at N1
    restore_sleep_cycle(fresh, data["sleep_cycle"])

    # Should be at N3 with the saved position
    assert fresh.current_stage is SleepStage.N3
    assert fresh.cycle_number == 1
    assert fresh.time_in_stage == 250.0
    assert fresh.cycles_completed == 0


def test_restore_sleep_cycle_rem_cycle_2():
    """Restoring a cycle-2 REM position should preserve cycle number."""
    tracker = SleepCycleTracker()
    # Advance through a full cycle: N1→N2→N3→N2→REM
    # Cycle 1 durations: N1=300, N2=900, N3=2400, N2_ascent=360, REM=600
    tracker.advance(300.0)   # N1 → N2
    tracker.advance(900.0)   # N2 → N3
    tracker.advance(2400.0)  # N3 → N2 (ascent)
    tracker.advance(360.0)   # N2 ascent → REM
    tracker.advance(600.0)   # REM → new cycle N1
    # Now in cycle 2
    assert tracker.cycle_number == 2
    tracker.advance(300.0)   # N1 → N2
    # Cycle 2 N2 descent = 900 + 60 = 960s
    tracker.advance(960.0)   # N2 → N3

    data = serialize_sleep_state(
        is_sleeping=True,
        user_initiated_sleep=True,
        sleep_cycle=tracker,
    )

    fresh = SleepCycleTracker()
    restore_sleep_cycle(fresh, data["sleep_cycle"])
    assert fresh.current_stage is SleepStage.N3
    assert fresh.cycle_number == 2
    assert fresh.cycles_completed == 1


def test_restore_sleep_cycle_invalid_stage_falls_back():
    """An invalid stage string should fall back to N1, not crash."""
    data = {
        "current_stage": "invalid_stage",
        "cycle_number": 3,
        "stage_slot": 2,
        "time_in_stage": 500.0,
        "time_in_cycle": 2000.0,
        "cycles_completed": 2,
    }
    fresh = SleepCycleTracker()
    restore_sleep_cycle(fresh, data)
    assert fresh.current_stage is SleepStage.N1
    assert fresh.cycle_number == 3
    assert fresh.cycles_completed == 2


def test_sleep_state_roundtrip_through_save_load():
    """Full save/load roundtrip: serialize → save to disk → load → restore."""
    tracker = SleepCycleTracker()
    tracker.advance(300.0)  # N1 → N2
    tracker.advance(900.0)  # N2 → N3
    tracker.advance(150.0)  # 150s into N3

    sleep_data = serialize_sleep_state(
        is_sleeping=True,
        user_initiated_sleep=True,
        sleep_cycle=tracker,
    )

    with tempfile.TemporaryDirectory() as data_dir:
        net = ConceptNetwork()
        sm = SelfModel()
        # Save with sleep state
        save_state(
            data_dir,
            net,
            ReflectionEngine(net),
            NarrativeEngine(sm, net),
            sm,
            sleep_state=sleep_data,
        )

        # Load it back
        loaded = load_state(data_dir)
        assert loaded is not None
        assert "sleep_state" in loaded

        loaded_sleep = loaded["sleep_state"]
        assert loaded_sleep["is_sleeping"] is True
        assert loaded_sleep["user_initiated_sleep"] is True
        assert loaded_sleep["sleep_cycle"]["current_stage"] == "n3"

        # Restore the cycle tracker
        fresh = SleepCycleTracker()
        restore_sleep_cycle(fresh, loaded_sleep["sleep_cycle"])
        assert fresh.current_stage is SleepStage.N3
        assert fresh.time_in_stage == 150.0


def test_sleep_state_not_present_in_old_saves():
    """Old save files without sleep_state should not cause errors."""
    with tempfile.TemporaryDirectory() as data_dir:
        net = ConceptNetwork()
        sm = SelfModel()
        # Save without sleep_state (simulating an old save)
        save_state(
            data_dir,
            net,
            ReflectionEngine(net),
            NarrativeEngine(sm, net),
            sm,
        )

        loaded = load_state(data_dir)
        assert loaded is not None
        # Old saves don't have sleep_state — that's fine
        assert "sleep_state" not in loaded

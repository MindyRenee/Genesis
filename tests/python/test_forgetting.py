"""Unit tests for the forgetting mirror — Inner_World/memory/forgetting.py.

These verify that the Python side of the pair mirrors the Rust math in
`src/store/forgetting.rs`: the same constants, the same trace-mass and
activation equations, and the same gentle horizons (months of disuse,
not days). Wire-level coverage: FORGET_SWEEP unpack.
"""

import math
import struct
import sys

sys.path.insert(0, "Inner_World")
sys.path.insert(0, "Self_World")

from genesis_client.types import unpack_forget_sweep
from memory.forgetting import (
    CUE_WEIGHT,
    DA_BASELINE,
    FORGET_BOUND,
    PRUNE_AGE_FLOOR_DAYS,
    RETRIEVAL_TAU,
    TRACE_DECAY,
    base_level,
    days_until_forgotten,
    effective_threshold,
    forgetting_state,
    trace_mass,
)


def test_trace_mass_single_trace_exact():
    """One presentation contributes exactly age^(-d)."""
    assert math.isclose(trace_mass(1, 100.0, 100.0), 100.0 ** (-TRACE_DECAY))


def test_trace_mass_empty_is_zero():
    assert trace_mass(0, 10.0, 10.0) == 0.0


def test_trace_mass_rehearsal_after_silence_dominates():
    """A fresh rehearsal dominates a long-silent trace.

    True sum = 200^(-1/2) + 1^(-1/2) ≈ 1.07; the endpoint bound must
    keep the blend from smearing the fresh trace's strength across
    the whole span.
    """
    assert trace_mass(2, 200.0, 1.0) > 0.9


def test_horizon_is_gentle():
    """A single-hearing, mid-salience memory lasts months, not days."""
    horizon = days_until_forgotten(1, 0.5, 0.0)
    assert 100.0 < horizon < 200.0  # e^5 ≈ 148 days


def test_salience_and_rehearsal_extend_the_horizon():
    horizon = days_until_forgotten(1, 0.5, 0.0)
    # Salience 0.9 vs 0.5 stretches the horizon e^0.8 ≈ 2.2x.
    strong = days_until_forgotten(1, 0.9, 0.0)
    assert 2.0 < strong / horizon < 2.5
    # Ten rehearsals stretch it ~100x at n >= 2.
    assert days_until_forgotten(10, 0.5, 0.0) / horizon > 50.0


def test_base_level_includes_salience():
    b = base_level(1, 30.0, 30.0, 0.5)
    expected = math.log(30.0**-0.5) + 0.5
    assert math.isclose(b, expected)


def test_effective_threshold_tracks_dopamine():
    assert effective_threshold(DA_BASELINE) == RETRIEVAL_TAU
    assert effective_threshold(0.9) < RETRIEVAL_TAU
    assert effective_threshold(0.1) > RETRIEVAL_TAU
    assert math.isclose(effective_threshold(1.0), -0.2)


def test_forgetting_state_boundaries():
    """Fresh is accessible; faded needs a strong cue; gone is gone."""
    fresh = forgetting_state(1, 1.0, 1.0, 0.5)
    assert not fresh.faded and not fresh.gone

    faded = forgetting_state(1, 60.0, 60.0, 0.5)
    # ~60d single trace: B ≈ -1.55 — below tau, above the forget bound.
    assert faded.faded and not faded.gone

    gone = forgetting_state(1, 400.0, 400.0, 0.5)
    assert gone.gone

    # Below the bound but too young — the age floor protects it.
    young = forgetting_state(1, 10.0, 10.0, 0.0)
    assert not young.gone


def test_forget_bound_is_a_full_cue_below_tau():
    """Pruning only fires where no cue can reach — a whole CUE_WEIGHT
    below the retrieval threshold."""
    assert math.isclose(FORGET_BOUND, RETRIEVAL_TAU - CUE_WEIGHT)
    assert PRUNE_AGE_FLOOR_DAYS >= 30.0


def test_unpack_forget_sweep():
    payload = struct.pack("<IQ", 7, 4096)
    report = unpack_forget_sweep(payload)
    assert report.episodes == 7
    assert report.bytes == 4096
    # Short payloads decode as an empty report, not a crash.
    empty = unpack_forget_sweep(b"")
    assert empty.episodes == 0 and empty.bytes == 0

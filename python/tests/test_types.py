"""Unit tests for the Genesis client type deserialisation.

These tests verify that the Python types correctly parse the binary
wire formats produced by the Rust daemon. Uses only the standard library.
"""

import logging
import struct
import sys

from genesis_client.protocol import _F32, _I32, _U32, _U64, PHASE_FLOW
from genesis_client.types import (
    BodyControlState,
    BodyState,
    CoreState,
    Episode,
    FieldProbeSignature,
    MemoryStats,
    NeuroSummary,
    PhaseInfo,
    PingResponse,
    PlasticityProfile,
    unpack_similar_results,
)

logger = logging.getLogger(__name__)


def approx_equal(a: float, b: float, tolerance: float = 1e-5) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tolerance


def test_neuro_summary_unpack() -> None:
    """A 32-byte buffer unpacks correctly."""
    data = bytearray()
    data.extend(_F32.pack(0.65))  # arousal
    data.extend(_F32.pack(0.30))  # valence
    data.extend(_F32.pack(0.45))  # global_tone
    data.extend(_F32.pack(0.70))  # plasticity_gate
    data.extend(_F32.pack(0.55))  # encoding_weight
    data.extend(_F32.pack(0.54))  # consolidation_weight
    data.extend(_F32.pack(0.60))  # retrieval_weight
    data.append(PHASE_FLOW)  # phase
    data.extend(b"\x00\x00\x00")  # pad

    summary = NeuroSummary.unpack(bytes(data))
    assert approx_equal(summary.arousal, 0.65)
    assert approx_equal(summary.valence, 0.30)
    assert approx_equal(summary.global_tone, 0.45)
    assert approx_equal(summary.plasticity_gate, 0.70)
    assert approx_equal(summary.encoding_weight, 0.55)
    assert approx_equal(summary.consolidation_weight, 0.54)
    assert approx_equal(summary.retrieval_weight, 0.60)
    assert summary.phase == PHASE_FLOW
    assert summary.phase_name == "flow"


def test_neuro_summary_too_short() -> None:
    """A buffer shorter than 32 bytes raises ValueError."""
    try:
        NeuroSummary.unpack(b"\x00" * 31)
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))


def test_neuro_summary_unknown_phase() -> None:
    """An unknown phase ID returns 'unknown'."""
    data = bytearray(32)
    data[28] = 99
    summary = NeuroSummary.unpack(bytes(data))
    assert summary.phase_name == "unknown"


def test_memory_stats_unpack() -> None:
    """A 24-byte buffer unpacks correctly."""
    data = bytearray()
    data.extend(_U64.pack(42))
    data.extend(_U64.pack(100))
    data.extend(_U32.pack(65536))
    data.extend(_U32.pack(0))

    stats = MemoryStats.unpack(bytes(data))
    assert stats.stm_count == 42
    assert stats.ltm_count == 100
    assert stats.ltm_capacity == 65536


def test_memory_stats_too_short() -> None:
    """A buffer shorter than 24 bytes raises ValueError."""
    try:
        MemoryStats.unpack(b"\x00" * 23)
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))


# ─── PlasticityProfile ────────────────────────────────────────


def test_plasticity_profile_unpack() -> None:
    """A 40-byte buffer unpacks correctly."""
    data = bytearray()
    data.extend(_F32.pack(0.72))  # plasticity_gate
    data.extend(_F32.pack(0.65))  # bdnf_effective
    data.extend(_F32.pack(0.60))  # bdnf_tonic
    data.extend(_F32.pack(0.20))  # cortisol_effective
    data.extend(_F32.pack(0.18))  # cortisol_tonic
    data.extend(_F32.pack(0.55))  # dopamine_effective
    data.extend(_F32.pack(0.50))  # serotonin_effective
    data.extend(_F32.pack(0.003))  # coupling_drift
    data.extend(_F32.pack(0.98))  # mean_receptor_sensitivity
    data.append(2)  # emergent_phase (flow)
    data.extend(b"\x00\x00\x00")  # _pad

    profile = PlasticityProfile.unpack(bytes(data))
    assert approx_equal(profile.plasticity_gate, 0.72)
    assert approx_equal(profile.bdnf_effective, 0.65)
    assert approx_equal(profile.bdnf_tonic, 0.60)
    assert approx_equal(profile.cortisol_effective, 0.20)
    assert approx_equal(profile.cortisol_tonic, 0.18)
    assert approx_equal(profile.dopamine_effective, 0.55)
    assert approx_equal(profile.serotonin_effective, 0.50)
    assert approx_equal(profile.coupling_drift, 0.003)
    assert approx_equal(profile.mean_receptor_sensitivity, 0.98)
    assert profile.emergent_phase == 2
    assert profile.phase_name == "flow"


def test_plasticity_profile_too_short() -> None:
    """A buffer shorter than 40 bytes raises ValueError."""
    try:
        PlasticityProfile.unpack(b"\x00" * 39)
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))


def test_plasticity_profile_posture_receptive() -> None:
    """High plasticity gate + low cortisol tonic → RECEPTIVE."""
    data = bytearray()
    data.extend(_F32.pack(0.70))  # plasticity_gate (high)
    data.extend(_F32.pack(0.65))  # bdnf_effective
    data.extend(_F32.pack(0.60))  # bdnf_tonic
    data.extend(_F32.pack(0.03))  # cortisol_effective
    data.extend(_F32.pack(0.03))  # cortisol_tonic (low — envelope is ~0.02 rest/0.08 peak)
    data.extend(_F32.pack(0.55))  # dopamine_effective
    data.extend(_F32.pack(0.50))  # serotonin_effective
    data.extend(_F32.pack(0.0))  # coupling_drift
    data.extend(_F32.pack(0.98))  # mean_receptor_sensitivity
    data.append(2)  # emergent_phase
    data.extend(b"\x00\x00\x00")

    profile = PlasticityProfile.unpack(bytes(data))
    assert profile.learning_posture == "receptive"


def test_plasticity_profile_posture_protective() -> None:
    """High cortisol tonic + low plasticity gate → PROTECTIVE."""
    data = bytearray()
    data.extend(_F32.pack(0.25))  # plasticity_gate (low)
    data.extend(_F32.pack(0.15))  # bdnf_effective
    data.extend(_F32.pack(0.12))  # bdnf_tonic
    data.extend(_F32.pack(0.08))  # cortisol_effective (peak of the reachable envelope)
    data.extend(_F32.pack(0.07))  # cortisol_tonic (high — chronic stress, ~83% of envelope)
    data.extend(_F32.pack(0.30))  # dopamine_effective
    data.extend(_F32.pack(0.25))  # serotonin_effective
    data.extend(_F32.pack(0.01))  # coupling_drift
    data.extend(_F32.pack(0.85))  # mean_receptor_sensitivity
    data.append(3)  # emergent_phase (stress)
    data.extend(b"\x00\x00\x00")

    profile = PlasticityProfile.unpack(bytes(data))
    assert profile.learning_posture == "protective"


def test_plasticity_profile_posture_recovering() -> None:
    """Mid plasticity gate + dropping cortisol + BDNF rising → RECOVERING."""
    data = bytearray()
    data.extend(_F32.pack(0.40))  # plasticity_gate (mid)
    data.extend(_F32.pack(0.35))  # bdnf_effective
    data.extend(_F32.pack(0.35))  # bdnf_tonic (rising)
    data.extend(_F32.pack(0.04))  # cortisol_effective
    data.extend(_F32.pack(0.04))  # cortisol_tonic (dropping below stress, ~33% of envelope)
    data.extend(_F32.pack(0.40))  # dopamine_effective
    data.extend(_F32.pack(0.45))  # serotonin_effective
    data.extend(_F32.pack(0.005))  # coupling_drift
    data.extend(_F32.pack(0.90))  # mean_receptor_sensitivity
    data.append(0)  # emergent_phase (active)
    data.extend(b"\x00\x00\x00")

    profile = PlasticityProfile.unpack(bytes(data))
    assert profile.learning_posture == "recovering"


def test_plasticity_profile_posture_neutral() -> None:
    """Mid-range values that don't match any specific posture → NEUTRAL."""
    data = bytearray()
    data.extend(_F32.pack(0.45))  # plasticity_gate (mid)
    data.extend(_F32.pack(0.40))  # bdnf_effective
    data.extend(_F32.pack(0.20))  # bdnf_tonic (low — not recovering)
    data.extend(_F32.pack(0.05))  # cortisol_effective
    data.extend(_F32.pack(0.045))  # cortisol_tonic (mid-envelope, ~42%)
    data.extend(_F32.pack(0.40))  # dopamine_effective
    data.extend(_F32.pack(0.40))  # serotonin_effective
    data.extend(_F32.pack(0.0))  # coupling_drift
    data.extend(_F32.pack(0.95))  # mean_receptor_sensitivity
    data.append(0)  # emergent_phase
    data.extend(b"\x00\x00\x00")

    profile = PlasticityProfile.unpack(bytes(data))
    assert profile.learning_posture == "neutral"


def test_phase_info_unpack() -> None:
    """A 9-byte buffer unpacks correctly."""
    data = bytearray()
    data.append(2)
    data.extend(_F32.pack(0.7))
    data.extend(_F32.pack(0.4))

    info = PhaseInfo.unpack(bytes(data))
    assert info.phase == 2
    assert info.phase_name == "flow"
    assert approx_equal(info.arousal, 0.7)
    assert approx_equal(info.valence, 0.4)


def test_phase_info_too_short() -> None:
    """A buffer shorter than 9 bytes raises ValueError."""
    try:
        PhaseInfo.unpack(b"\x00" * 8)
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))


def test_ping_response_valid() -> None:
    """A 9-byte ping response unpacks correctly."""
    data = bytearray()
    data.append(1)
    data.extend(_U64.pack(123456))

    pong = PingResponse.unpack(bytes(data))
    assert pong.ack is True
    assert pong.uptime_ms == 123456


def test_ping_response_ack_only() -> None:
    """A 1-byte response (ack only) unpacks with uptime=0."""
    pong = PingResponse.unpack(bytes([1]))
    assert pong.ack is True
    assert pong.uptime_ms == 0


def test_ping_response_nack() -> None:
    """A 0 ack means failure."""
    data = bytes([0]) + b"\x00" * 8
    pong = PingResponse.unpack(data)
    assert pong.ack is False


def test_episode_not_found() -> None:
    """A found=0 response returns None."""
    data = bytes([0])
    ep = Episode.unpack(data)
    assert ep is None


def test_episode_empty_data() -> None:
    """Empty data returns None."""
    ep = Episode.unpack(b"")
    assert ep is None


def test_episode_valid() -> None:
    """A valid episode response unpacks correctly."""
    text = "Hello, Genesis!"
    text_bytes = text.encode("utf-8")

    data = bytearray()
    data.append(1)
    data.extend(_U64.pack(42))
    data.extend(_U64.pack(1700000))
    data.extend(_F32.pack(0.85))
    data.extend(_U64.pack(0xABCDEF))

    for v in [0.6, 0.3, 0.2, 0.7]:
        data.extend(_F32.pack(v))

    for v in [0.5] * 12:
        data.extend(_F32.pack(v))

    data.append(0)
    data.append(4)
    data.extend(_U32.pack(len(text_bytes)))
    data.extend(text_bytes)

    ep = Episode.unpack(bytes(data))
    assert ep is not None
    assert ep.episode_id == 42
    assert ep.timestamp == 1700000
    assert approx_equal(ep.salience, 0.85)
    assert ep.association_hash == 0xABCDEF
    assert len(ep.compact_emotional_tag) == 4
    assert len(ep.full_emotional_tag) == 12
    assert ep.event_type == 0
    assert ep.source_module == 4
    assert ep.text == "Hello, Genesis!"


def test_episode_chem_names() -> None:
    """The chem_names property maps chemical names to levels."""
    data = bytearray()
    data.append(1)
    data.extend(_U64.pack(1))
    data.extend(_U64.pack(0))
    data.extend(_F32.pack(0.5))
    data.extend(_U64.pack(0))
    for _ in range(4):
        data.extend(_F32.pack(0))
    chem_values = [0.8, 0.6, 0.4, 0.5, 0.3, 0.7, 0.2, 0.4, 0.3, 0.5, 0.1, 0.4]
    for v in chem_values:
        data.extend(_F32.pack(v))
    data.append(0)
    data.append(0)
    data.extend(_U32.pack(0))

    ep = Episode.unpack(bytes(data))
    assert ep is not None
    names = ep.chem_names
    assert approx_equal(names["dopamine"], 0.8)
    assert approx_equal(names["cortisol"], 0.2)
    assert approx_equal(names["bdnf"], 0.4)


def test_similar_results_empty() -> None:
    """A count=0 response returns an empty list."""
    data = _U32.pack(0)
    results = unpack_similar_results(data)
    assert results == []


def test_similar_results_too_short() -> None:
    """A response shorter than 4 bytes returns empty list."""
    results = unpack_similar_results(b"\x00\x00")
    assert results == []


def test_similar_results_valid() -> None:
    """A response with 3 results unpacks correctly."""
    data = bytearray()
    data.extend(_U32.pack(3))
    for i in range(3):
        data.extend(_U64.pack(i + 1))
        data.extend(_U32.pack(i * 5))
        data.extend(_U64.pack(1000 + i))
        data.extend(_F32.pack(0.5 + i * 0.1))

    results = unpack_similar_results(bytes(data))
    assert len(results) == 3
    assert results[0].episode_id == 1
    assert results[0].hamming_distance == 0
    assert results[1].episode_id == 2
    assert results[1].hamming_distance == 5
    assert approx_equal(results[2].salience, 0.7)


def test_core_state_unpack() -> None:
    """A 3480-byte buffer unpacks the key fields correctly.

    Tests the v5 schema (18 chemicals, 3480 bytes). The offsets are:
    - Header: 0-55 (56 bytes)
    - NeurochemicalVector: 56-2471 (2416 bytes)
      - chemicals[18]: 56-1063 (1008 bytes, 18 × 56 each)
      - coupling_matrix[18][18]: 1064-2359 (1296 bytes)
      - effective_levels[18]: 2360-2431 (72 bytes)
      - global_tone: 2432
      - arousal: 2436
      - valence: 2440
      - plasticity_gate: 2444
    - ActiveZones: 2472-2567 (96 bytes)
      - cognitive_zone: 2472
      - emergent_phase: 2473
    - MemoryPointers: 2568-2687 (120 bytes)
      - stm_count: 2584
      - ltm_episode_count: 2600
      - encoding_weight: 2632
      - consolidation_weight: 2636
      - retrieval_weight: 2640
    """
    data = bytearray(3480)

    # Header fields
    struct.pack_into("<Q", data, 16, 1000)  # created_at
    struct.pack_into("<Q", data, 24, 2000)  # last_updated
    struct.pack_into("<Q", data, 32, 42)  # heartbeat
    struct.pack_into("<Q", data, 48, 99)  # instance_id

    # NeurochemicalVector derived fields
    # effective_levels[18] at offset 2360 (56 + 1008 + 1296)
    struct.pack_into("<f", data, 2360, 0.50)  # dopamine
    struct.pack_into("<f", data, 2364, 0.55)  # serotonin
    struct.pack_into("<f", data, 2432, 0.45)  # global_tone
    struct.pack_into("<f", data, 2436, 0.65)  # arousal
    struct.pack_into("<f", data, 2440, 0.30)  # valence
    struct.pack_into("<f", data, 2444, 0.70)  # plasticity_gate

    # ActiveZones
    data[2472] = 2  # cognitive_zone
    data[2473] = 2  # emergent_phase (flow)

    # MemoryPointers
    struct.pack_into("<I", data, 2584, 15)  # stm_count
    struct.pack_into("<Q", data, 2600, 200)  # ltm_episode_count
    struct.pack_into("<f", data, 2632, 0.55)  # encoding_weight
    struct.pack_into("<f", data, 2636, 0.54)  # consolidation_weight
    struct.pack_into("<f", data, 2640, 0.60)  # retrieval_weight

    state = CoreState.unpack(bytes(data))
    assert state.heartbeat == 42
    assert state.instance_id == 99
    assert state.created_at == 1000
    assert state.last_updated == 2000
    assert approx_equal(state.arousal, 0.65)
    assert approx_equal(state.valence, 0.30)
    assert approx_equal(state.global_tone, 0.45)
    assert approx_equal(state.plasticity_gate, 0.70)
    assert approx_equal(state.chemicals["dopamine"], 0.50)
    assert approx_equal(state.chemicals["serotonin"], 0.55)
    assert state.cognitive_zone == 2
    assert state.emergent_phase == 2
    assert state.phase_name == "flow"
    assert state.stm_count == 15
    assert state.ltm_episode_count == 200
    assert approx_equal(state.encoding_weight, 0.55)
    assert approx_equal(state.consolidation_weight, 0.54)
    assert approx_equal(state.retrieval_weight, 0.60)


def test_core_state_reads_inference_signals_from_the_v5_offset() -> None:
    """The inference block is read from 3416, not the v4 offset of 3352.

    Schema v5 inserted the 64-byte sleep network at 3224 and moved the
    ion block to 3288. A v4 offset of 3352 now lands inside IonState, and
    a read there does not fail: it returns ion concentrations wearing the
    field names of inference signals, which is worse than an error
    because the regulator consumes `allostasis_load` as real burden.
    """
    data = bytearray(3480)
    # Poison the region a stale v4 read would hit, so the wrong offset is
    # unambiguously wrong rather than accidentally plausible.
    for off in range(3352, 3412, 4):
        struct.pack_into("<f", data, off, 9.99)
    # Field offsets are relative to 3416: 14 f32s in declaration order,
    # then inference_tick_count (u32) at +56, then policy_authority.
    struct.pack_into("<f", data, 3416, 0.1111)  # surprise_ema
    struct.pack_into("<f", data, 3420, 0.2222)  # free_energy
    struct.pack_into("<f", data, 3428, 0.7777)  # allostasis_load
    struct.pack_into("<f", data, 3468, 0.6666)  # model_maturity
    struct.pack_into("<I", data, 3472, 4242)  # inference_tick_count

    state = CoreState.unpack(bytes(data))
    assert state.inference is not None
    assert approx_equal(state.inference.surprise_ema, 0.1111)
    assert approx_equal(state.inference.free_energy, 0.2222)
    assert approx_equal(state.inference.allostasis_load, 0.7777)
    assert approx_equal(state.inference.model_maturity, 0.6666)
    assert state.inference.inference_tick_count == 4242


def test_core_state_too_short() -> None:
    """A buffer shorter than 3480 bytes raises ValueError."""
    try:
        CoreState.unpack(b"\x00" * 3407)
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))


def run_all() -> bool:
    """Run all tests and report results."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    logger.info(f"\n  Type tests: {passed} passed, {failed} failed")
    return failed == 0


# ─── BodyState tests ──────────────────────────────────────────


def test_body_state_unpack_normal() -> None:
    """A normal body state unpacks correctly."""
    data = bytearray()
    data.extend(_F32.pack(52.0))  # cpu_temp_c
    data.extend(_F32.pack(0.52))  # temperature
    data.extend(_F32.pack(0.7))  # arousal_freq
    data.extend(_F32.pack(0.4))  # cognitive_load
    data.extend(_F32.pack(0.1))  # io_activity
    data.extend(_F32.pack(0.2))  # stress_load
    data.extend(_F32.pack(1.0))  # energy_reserve
    data.append(1)  # on_ac_power
    data.extend(_U32.pack(8))  # num_cores
    data.append(0)  # distressed
    data.extend(_F32.pack(3.5))  # autonomic_rate
    data.extend(_F32.pack(0.6))  # thermoregulatory_effort
    data.extend(_F32.pack(0.2))  # metabolic_rate
    data.extend(_F32.pack(0.92))  # core_voltage
    data.extend(_F32.pack(12.6))  # supply_voltage
    desc = "Running normally."
    data.extend(_U32.pack(len(desc)))  # desc_len
    data.extend(desc.encode("utf-8"))

    body = BodyState.unpack(bytes(data))
    assert approx_equal(body.cpu_temp_c, 52.0)
    assert approx_equal(body.temperature, 0.52)
    assert approx_equal(body.arousal_freq, 0.7)
    assert approx_equal(body.cognitive_load, 0.4)
    assert approx_equal(body.io_activity, 0.1)
    assert approx_equal(body.stress_load, 0.2)
    assert approx_equal(body.energy_reserve, 1.0)
    assert body.on_ac_power
    assert body.num_cores == 8
    assert not body.distressed
    assert approx_equal(body.autonomic_rate, 3.5)
    assert approx_equal(body.thermoregulatory_effort, 0.6)
    assert approx_equal(body.metabolic_rate, 0.2)
    assert approx_equal(body.core_voltage, 0.92)
    assert approx_equal(body.supply_voltage, 12.6)
    assert body.description == "Running normally."


def test_body_state_unpack_distressed() -> None:
    """A distressed body state unpacks correctly."""
    data = bytearray()
    data.extend(_F32.pack(92.0))  # cpu_temp_c
    data.extend(_F32.pack(0.95))  # temperature
    data.extend(_F32.pack(0.3))  # arousal_freq
    data.extend(_F32.pack(0.92))  # cognitive_load
    data.extend(_F32.pack(0.05))  # io_activity
    data.extend(_F32.pack(1.5))  # stress_load
    data.extend(_F32.pack(0.15))  # energy_reserve
    data.append(0)  # on_ac_power (battery)
    data.extend(_U32.pack(4))  # num_cores
    data.append(1)  # distressed
    data.extend(_F32.pack(8.0))  # autonomic_rate (fast — stressed)
    data.extend(_F32.pack(0.02))  # thermoregulatory_effort (low)
    data.extend(_F32.pack(0.9))  # metabolic_rate (high)
    data.extend(_F32.pack(1.30))  # core_voltage (Vcore maxed under stress)
    data.extend(_F32.pack(10.2))  # supply_voltage (critically low battery)
    desc = "I'm overheating and overwhelmed."
    data.extend(_U32.pack(len(desc)))
    data.extend(desc.encode("utf-8"))

    body = BodyState.unpack(bytes(data))
    assert approx_equal(body.cpu_temp_c, 92.0)
    assert body.distressed
    assert not body.on_ac_power
    assert body.num_cores == 4
    assert approx_equal(body.autonomic_rate, 8.0)
    assert approx_equal(body.thermoregulatory_effort, 0.02)
    assert approx_equal(body.metabolic_rate, 0.9)
    assert approx_equal(body.core_voltage, 1.30)
    assert approx_equal(body.supply_voltage, 10.2)
    assert "overheating" in body.description


def test_body_state_unpack_empty_description() -> None:
    """Body state with empty description unpacks correctly."""
    data = bytearray()
    data.extend(_F32.pack(45.0))  # cpu_temp_c
    data.extend(_F32.pack(0.45))  # temperature
    data.extend(_F32.pack(0.5))  # arousal_freq
    data.extend(_F32.pack(0.3))  # cognitive_load
    data.extend(_F32.pack(0.2))  # io_activity
    data.extend(_F32.pack(0.1))  # stress_load
    data.extend(_F32.pack(0.9))  # energy_reserve
    data.append(1)  # on_ac_power
    data.extend(_U32.pack(2))  # num_cores
    data.append(0)  # distressed
    data.extend(_F32.pack(1.5))  # autonomic_rate
    data.extend(_F32.pack(0.1))  # thermoregulatory_effort
    data.extend(_F32.pack(0.05))  # metabolic_rate
    data.extend(_F32.pack(0.88))  # core_voltage
    data.extend(_F32.pack(12.5))  # supply_voltage
    data.extend(_U32.pack(0))  # desc_len = 0

    body = BodyState.unpack(bytes(data))
    assert body.description == ""
    assert body.num_cores == 2
    assert approx_equal(body.autonomic_rate, 1.5)
    assert approx_equal(body.thermoregulatory_effort, 0.1)
    assert approx_equal(body.metabolic_rate, 0.05)
    assert approx_equal(body.core_voltage, 0.88)
    assert approx_equal(body.supply_voltage, 12.5)


def test_body_state_unpack_too_short() -> None:
    """A buffer that's too short raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        BodyState.unpack(b"\x00\x00\x00")


def test_body_state_unpack_v3_pulse_layer() -> None:
    """A v3 packet (pulse/involuntary/senescence fields) unpacks correctly.

    The v3 layout appends 9 f32 + 2 u8 between branch_miss_rate and
    desc_len (offset 112). Older tiers must still parse: this test
    also guards that a full packet's desc_len at 112 isn't mistaken
    for the v1 desc_len at 54 or the v2 desc_len at 74.
    """
    data = bytearray()
    data.extend(_F32.pack(63.0))  # cpu_temp_c
    data.extend(_F32.pack(0.63))  # temperature
    data.extend(_F32.pack(0.8))  # arousal_freq
    data.extend(_F32.pack(0.5))  # cognitive_load
    data.extend(_F32.pack(0.3))  # io_activity
    data.extend(_F32.pack(0.6))  # stress_load
    data.extend(_F32.pack(0.9))  # energy_reserve
    data.append(1)  # on_ac_power
    data.extend(_U32.pack(4))  # num_cores
    data.append(0)  # distressed
    data.extend(_F32.pack(2.0))  # autonomic_rate
    data.extend(_F32.pack(0.3))  # thermoregulatory_effort
    data.extend(_F32.pack(0.15))  # metabolic_rate
    data.extend(_F32.pack(0.92))  # core_voltage
    data.extend(_F32.pack(12.5))  # supply_voltage
    # v2 silicon layer
    data.extend(_F32.pack(0.4))  # core_activity
    data.extend(_F32.pack(0.2))  # uncore_activity
    data.extend(_F32.pack(0.5))  # dram_activity
    data.extend(_F32.pack(0.03))  # cache_miss_rate
    data.extend(_F32.pack(0.02))  # branch_miss_rate
    # v3 timing/involuntary/senescence layer
    data.extend(_F32.pack(4100.0))  # pulse_hz — cores beating
    data.extend(_F32.pack(0.9))  # pulse (normalized)
    data.extend(_F32.pack(0.25))  # throttle_state — mild throttle
    data.extend(_F32.pack(0.55))  # top_freq_share
    data.extend(_F32.pack(0.12))  # psi_cpu
    data.extend(_F32.pack(0.04))  # psi_io
    data.extend(_F32.pack(0.08))  # psi_mem
    data.extend(_F32.pack(361.0))  # battery_cycles
    data.extend(_F32.pack(1.0))  # entropy_level
    data.append(1)  # clocksource = tsc
    data.append(3)  # suspend_caps = mem + wakealarm
    desc = "Pulse steady, mild throttle."
    data.extend(_U32.pack(len(desc)))  # desc_len at offset 112
    data.extend(desc.encode("utf-8"))

    body = BodyState.unpack(bytes(data))
    assert approx_equal(body.cpu_temp_c, 63.0)
    assert approx_equal(body.core_activity, 0.4)
    assert approx_equal(body.branch_miss_rate, 0.02)
    # The v3 fields arrive intact.
    assert approx_equal(body.pulse_hz, 4100.0)
    assert approx_equal(body.pulse, 0.9)
    assert approx_equal(body.throttle_state, 0.25)
    assert approx_equal(body.top_freq_share, 0.55)
    assert approx_equal(body.psi_cpu, 0.12)
    assert approx_equal(body.psi_mem, 0.08)
    assert approx_equal(body.battery_cycles, 361.0)
    assert approx_equal(body.entropy_level, 1.0)
    assert body.clocksource == 1
    assert body.suspend_caps == 3
    assert body.description == "Pulse steady, mild throttle."


def test_body_state_unpack_v3_with_zeroed_silicon_fields() -> None:
    """A v3 packet whose f32 at offset 54 is 0.0 must still parse as v3.

    Regression: the v1/v2/v3 probes used to accept a candidate
    desc_len with `<=` trailer length. On machines without powercap
    the silicon layer is all 0.0 — whose bits are u32 0, satisfying
    `<=` for ANY length — so v3 packets on such hardware were
    misdetected as v1 and every post-offset-54 field was zeroed.
    Exact `==` matching prevents the collision.
    """
    data = bytearray()
    data.extend(_F32.pack(61.0))  # cpu_temp_c
    data.extend(_F32.pack(0.61))  # temperature
    data.extend(_F32.pack(0.5))  # arousal_freq
    data.extend(_F32.pack(0.4))  # cognitive_load
    data.extend(_F32.pack(0.1))  # io_activity
    data.extend(_F32.pack(0.3))  # stress_load
    data.extend(_F32.pack(1.0))  # energy_reserve
    data.append(1)  # on_ac_power
    data.extend(_U32.pack(4))  # num_cores
    data.append(0)  # distressed
    data.extend(_F32.pack(1.0))  # autonomic_rate
    data.extend(_F32.pack(0.2))  # thermoregulatory_effort
    data.extend(_F32.pack(0.1))  # metabolic_rate
    data.extend(_F32.pack(0.9))  # core_voltage
    data.extend(_F32.pack(12.6))  # supply_voltage
    # Silicon layer entirely 0.0 — the no-powercap machine. Its
    # first field sits at offset 54 where the v1 probe reads a
    # candidate desc_len: f32 0.0 == u32 0 used to falsely match.
    data.extend(_F32.pack(0.0))  # core_activity   (offset 54)
    data.extend(_F32.pack(0.0))  # uncore_activity
    data.extend(_F32.pack(0.0))  # dram_activity
    data.extend(_F32.pack(0.0))  # cache_miss_rate
    data.extend(_F32.pack(0.0))  # branch_miss_rate
    # v3 layer — nonzero so detection is proven by these fields.
    data.extend(_F32.pack(3910.0))  # pulse_hz
    data.extend(_F32.pack(0.98))  # pulse
    data.extend(_F32.pack(0.0))  # throttle_state
    data.extend(_F32.pack(0.5))  # top_freq_share
    data.extend(_F32.pack(0.02))  # psi_cpu
    data.extend(_F32.pack(0.01))  # psi_io
    data.extend(_F32.pack(0.03))  # psi_mem
    data.extend(_F32.pack(354.0))  # battery_cycles
    data.extend(_F32.pack(1.0))  # entropy_level
    data.append(1)  # clocksource = tsc
    data.append(3)  # suspend_caps
    data.extend(_U32.pack(0))  # desc_len = 0 at offset 112

    body = BodyState.unpack(bytes(data))
    assert approx_equal(body.pulse_hz, 3910.0)
    assert approx_equal(body.battery_cycles, 354.0)
    assert body.clocksource == 1
    assert body.suspend_caps == 3


def test_body_state_unpack_v2_with_zeroed_silicon_fields() -> None:
    """A v2 packet with all-zero silicon fields must still parse as v2.

    Same regression class as the v3 case: f32 0.0 at offset 54 must
    not be mistaken for the v1 desc_len.
    """
    data = bytearray()
    data.extend(_F32.pack(50.0))
    data.extend(_F32.pack(0.5))
    data.extend(_F32.pack(0.6))
    data.extend(_F32.pack(0.4))
    data.extend(_F32.pack(0.1))
    data.extend(_F32.pack(0.2))
    data.extend(_F32.pack(1.0))
    data.append(1)
    data.extend(_U32.pack(4))
    data.append(0)
    data.extend(_F32.pack(2.0))
    data.extend(_F32.pack(0.3))
    data.extend(_F32.pack(0.1))
    data.extend(_F32.pack(0.9))
    data.extend(_F32.pack(12.6))
    data.extend(_F32.pack(0.0))  # core_activity — 0.0, offset 54
    data.extend(_F32.pack(0.0))
    data.extend(_F32.pack(0.0))
    data.extend(_F32.pack(0.0))
    data.extend(_F32.pack(0.0))  # branch_miss_rate (offset 70)
    desc = "quiet silicon"
    data.extend(_U32.pack(len(desc)))  # desc_len at offset 74
    data.extend(desc.encode("utf-8"))

    body = BodyState.unpack(bytes(data))
    # Parses as v2: silicon fields zero, description intact, and the
    # v3 fields default to zero.
    assert body.description == "quiet silicon"
    assert body.pulse_hz == 0.0
    assert body.clocksource == 0


def test_body_state_unpack_v2_fields_default_on_v1() -> None:
    """v1 packets unpack with zeroed v2/v3 fields (graceful default)."""
    data = bytearray()
    data.extend(_F32.pack(52.0))
    data.extend(_F32.pack(0.52))
    data.extend(_F32.pack(0.7))
    data.extend(_F32.pack(0.4))
    data.extend(_F32.pack(0.1))
    data.extend(_F32.pack(0.2))
    data.extend(_F32.pack(1.0))
    data.append(1)
    data.extend(_U32.pack(4))
    data.append(0)
    data.extend(_F32.pack(3.5))
    data.extend(_F32.pack(0.6))
    data.extend(_F32.pack(0.2))
    data.extend(_F32.pack(0.92))
    data.extend(_F32.pack(12.6))
    data.extend(_U32.pack(0))  # desc_len = 0 at offset 54

    body = BodyState.unpack(bytes(data))
    assert body.pulse_hz == 0.0
    assert body.pulse == 0.0
    assert body.throttle_state == 0.0
    assert body.psi_cpu == 0.0
    assert body.battery_cycles == 0.0
    assert body.clocksource == 0
    assert body.suspend_caps == 0


def _v3_packet(*, desc: str = "") -> bytes:
    data = bytearray()
    data.extend(_F32.pack(60.0))  # cpu_temp_c
    data.extend(_F32.pack(0.6))  # temperature
    data.extend(_F32.pack(0.7))  # arousal_freq
    data.extend(_F32.pack(0.4))  # cognitive_load
    data.extend(_F32.pack(0.2))  # io_activity
    data.extend(_F32.pack(0.5))  # stress_load
    data.extend(_F32.pack(1.0))  # energy_reserve
    data.append(1)  # on_ac_power
    data.extend(_U32.pack(4))  # num_cores
    data.append(0)  # distressed
    data.extend(_F32.pack(2.0))  # autonomic_rate
    data.extend(_F32.pack(0.2))  # thermoregulatory_effort
    data.extend(_F32.pack(0.1))  # metabolic_rate
    data.extend(_F32.pack(0.9))  # core_voltage
    data.extend(_F32.pack(12.6))  # supply_voltage
    for _ in range(5):  # v2 silicon layer
        data.extend(_F32.pack(0.0))
    for _ in range(9):  # v3 f32 layer
        data.extend(_F32.pack(0.0))
    data.append(1)  # clocksource
    data.append(0)  # suspend_caps
    data.extend(_U32.pack(len(desc)))  # desc_len at offset 112
    data.extend(desc.encode("utf-8"))
    return bytes(data)


def _v4_packet(*, desc: str = "", rest: float = 0.0, temp_trend: float = 0.0) -> bytes:
    """A v3 packet plus the v4 parasympathetic tail."""
    data = bytearray(_v3_packet(desc=""))
    # Rebuild rather than splice: the v3 helper put desc_len at 112;
    # v4 needs rest/temp_trend there and desc_len at 120.
    del data[112:]
    data.extend(_F32.pack(rest))  # rest (offset 112)
    data.extend(_F32.pack(temp_trend))  # temp_trend (offset 116)
    data.extend(_U32.pack(len(desc)))  # desc_len at offset 120
    data.extend(desc.encode("utf-8"))
    return bytes(data)


def test_body_state_unpack_v4_parasympathetic_layer() -> None:
    """A v4 packet decodes the parasympathetic tail and its
    description — and does not leak the tail into v3 fields."""
    body = BodyState.unpack(_v4_packet(rest=0.85, temp_trend=-0.4, desc="body at rest"))
    assert approx_equal(body.rest, 0.85)
    assert approx_equal(body.temp_trend, -0.4)
    assert body.description == "body at rest"
    # The v3 layer is intact — v4 is a strict tail, not a layout change.
    assert approx_equal(body.cpu_temp_c, 60.0)
    assert body.clocksource == 1


def test_body_state_unpack_v4_tail_defaults_on_v3() -> None:
    """A v3 packet unpacks with rest/temp_trend defaulting to zero —
    append-only compatibility in the backward direction."""
    body = BodyState.unpack(_v3_packet(desc="v3 era body"))
    assert body.rest == 0.0
    assert body.temp_trend == 0.0
    assert body.description == "v3 era body"


def test_body_state_unpack_v4_not_misdetected_as_v1() -> None:
    """The v4 packet's desc_len sits at 120; its rest field at 112
    must never satisfy the v3 desc_len probe at 112 either — rest
    is clamped [0, 1] whose bits are 0 or ≥ 2³¹-ish, never a small
    trailer length."""
    body = BodyState.unpack(_v4_packet(rest=0.5, temp_trend=0.1))
    assert approx_equal(body.rest, 0.5)
    assert approx_equal(body.temp_trend, 0.1)
    assert body.description == ""


def test_sensor_presence_unpack_u32() -> None:
    """GET_SENSOR_PRESENCE widened to u32 when bit 16 (cpuidle) was
    added. A 4-byte payload decodes the high bit; a 2-byte payload
    from an older daemon still decodes its low bits."""
    from genesis_client.protocol import SENSOR_CPUIDLE, SENSOR_TEMPERATURE
    from genesis_client.types import SensorPresence

    presence = SensorPresence.unpack(struct.pack("<I", SENSOR_CPUIDLE | SENSOR_TEMPERATURE))
    assert presence.has(SENSOR_CPUIDLE)
    assert presence.has(SENSOR_TEMPERATURE)
    assert "cpuidle state counters" in presence.present

    # Old u16 payload: decodes, new channels read as absent.
    old = SensorPresence.unpack(struct.pack("<H", SENSOR_TEMPERATURE))
    assert old.has(SENSOR_TEMPERATURE)
    assert not old.has(SENSOR_CPUIDLE)


# ─── BodyControlState tests ───────────────────────────────────


def test_body_control_unpack_normal() -> None:
    """A well-formed body control buffer unpacks correctly."""
    gov = b"schedutil"
    io = b"best-effort-3"
    epp = b"performance"
    desc = b"I'm running myself fast"
    data = bytearray()
    data += _U32.pack(1_000_000)  # cpu_min_freq_khz
    data += _U32.pack(2_000_000)  # cpu_max_freq_khz
    data += _U32.pack(len(gov))  # gov_len
    data += gov
    data += bytes([1])  # thermally_capped
    data += _F32.pack(82.5)  # thermal_cap_temp_c
    data += _I32.pack(-3)  # daemon_nice
    data += _I32.pack(-5)  # cognitive_nice
    data += _U32.pack(len(io))  # io_class_len
    data += io
    data += _F32.pack(0.65)  # plasticity_gate
    data += bytes([1])  # controlling_cognitive
    data += _U32.pack(len(epp))  # epp_len
    data += epp
    data += bytes([1])  # cpu_boost = enabled (engaged, turbo permitted)
    data += _U32.pack(len(desc))  # desc_len
    data += desc

    ctrl = BodyControlState.unpack(bytes(data))
    assert ctrl.cpu_min_freq_khz == 1_000_000
    assert ctrl.cpu_max_freq_khz == 2_000_000
    assert ctrl.cpu_governor == "schedutil"
    assert ctrl.thermally_capped is True
    assert approx_equal(ctrl.thermal_cap_temp_c, 82.5)
    assert ctrl.daemon_nice == -3
    assert ctrl.cognitive_nice == -5
    assert ctrl.io_class == "best-effort-3"
    assert approx_equal(ctrl.plasticity_gate, 0.65)
    assert ctrl.controlling_cognitive is True
    assert ctrl.cpu_epp == "performance"
    assert ctrl.cpu_boost == "enabled"
    assert ctrl.description == "I'm running myself fast"


def test_body_control_unpack_sleep() -> None:
    """Powersave governor with deprioritized cognitive mind."""
    gov = b"powersave"
    io = b"idle"
    epp = b"power"
    data = bytearray()
    data += _U32.pack(1_000_000)
    data += _U32.pack(1_100_000)
    data += _U32.pack(len(gov))
    data += gov
    data += bytes([0])  # not thermally capped
    data += _F32.pack(0.0)
    data += _I32.pack(0)  # daemon_nice
    data += _I32.pack(10)  # cognitive_nice (deprioritized for sleep)
    data += _U32.pack(len(io))
    data += io
    data += _F32.pack(0.05)  # very low plasticity
    data += bytes([1])  # controlling cognitive
    data += _U32.pack(len(epp))  # epp_len
    data += epp
    data += bytes([2])  # cpu_boost = disabled (asleep, no turbo)
    data += _U32.pack(0)  # empty description

    ctrl = BodyControlState.unpack(bytes(data))
    assert ctrl.cpu_governor == "powersave"
    assert ctrl.cognitive_nice == 10
    assert ctrl.io_class == "idle"
    assert ctrl.cpu_epp == "power"
    assert ctrl.cpu_boost == "disabled"
    assert ctrl.description == ""


def test_body_control_unpack_too_short() -> None:
    """A buffer that's too short raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        BodyControlState.unpack(b"\x00\x00\x00")


def test_body_control_unpack_empty_strings() -> None:
    """Empty governor, io_class, and epp unpack correctly."""
    data = bytearray()
    data += _U32.pack(0)  # cpu_min_freq_khz
    data += _U32.pack(0)  # cpu_max_freq_khz
    data += _U32.pack(0)  # gov_len = 0
    data += bytes([0])  # thermally_capped
    data += _F32.pack(0.0)
    data += _I32.pack(0)
    data += _I32.pack(0)
    data += _U32.pack(0)  # io_class_len = 0
    data += _F32.pack(0.0)
    data += bytes([0])
    data += _U32.pack(0)  # epp_len = 0
    data += bytes([0])  # cpu_boost = unavailable (no gate on this platform)
    data += _U32.pack(0)  # desc_len = 0

    ctrl = BodyControlState.unpack(bytes(data))
    assert ctrl.cpu_governor == ""
    assert ctrl.io_class == ""
    assert ctrl.cpu_epp == ""
    assert ctrl.cpu_boost == ""
    assert ctrl.description == ""


def test_body_control_unpack_unknown_boost_byte() -> None:
    """An unrecognised boost byte decodes to "" (unavailable), not a
    bogus enabled/disabled claim."""
    data = bytearray()
    data += _U32.pack(0)  # cpu_min_freq_khz
    data += _U32.pack(0)  # cpu_max_freq_khz
    data += _U32.pack(0)  # gov_len = 0
    data += bytes([0])  # thermally_capped
    data += _F32.pack(0.0)
    data += _I32.pack(0)
    data += _I32.pack(0)
    data += _U32.pack(0)  # io_class_len = 0
    data += _F32.pack(0.0)
    data += bytes([0])  # controlling_cognitive
    data += _U32.pack(0)  # epp_len = 0
    data += bytes([99])  # cpu_boost — unknown value
    data += _U32.pack(0)  # desc_len = 0

    ctrl = BodyControlState.unpack(bytes(data))
    assert ctrl.cpu_boost == ""


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


def test_core_state_reads_the_network_sleep_pressure() -> None:
    """The homeostat h is sleep_network[4] at offset 3256; the mind
    reads it as a fraction of the source's hmax (323.88)."""
    data = bytearray(3480)
    struct.pack_into("<d", data, 3256, 161.94)
    state = CoreState.unpack(bytes(data))
    assert approx_equal(state.sleep_pressure, 0.5)


def _v3_field_probe_body() -> bytearray:
    """A v3 field-probe wire body (195 bytes) laid out per
    `src/daemon/field_probe.rs`."""
    body = bytearray(195)
    body[0] = 3
    struct.pack_into("<20f", body, 1, *[0.125] * 20)
    struct.pack_into("<5f", body, 81, 1.0e6, 0.5, 100.0, 7000.0, 10.0)
    body[101] = 1  # samples_ok
    struct.pack_into("<3f", body, 102, 0.3, 1.0, 0.7)
    struct.pack_into("<10f", body, 114, *[0.2] * 10)  # power bands
    struct.pack_into("<6f", body, 154, 20.9, 89.0, 12.947, 1.5, 0.02, 0.4)
    body[178] = 0b111  # power + volts + amps present
    struct.pack_into("<4f", body, 179, 0.97, 7.17e-4, 6.76e-9, 1.81e-34)
    return body


def test_field_probe_v3_unpacks_electrical_and_maxwell() -> None:
    """The v3 tail decodes real electrical channels and the Maxwell
    estimates through the presence mask."""
    sig = FieldProbeSignature.unpack(bytes(_v3_field_probe_body()))
    assert sig.power_watts is not None
    assert sig.power_sensor_hz is not None
    assert sig.rail_volts is not None
    assert sig.supply_amps is not None
    assert sig.power_lag is not None
    assert sig.power_coherence is not None
    assert sig.carrier_current_a is not None
    assert approx_equal(sig.power_watts, 20.9, 1e-2)
    assert approx_equal(sig.power_sensor_hz, 89.0, 1e-2)
    assert approx_equal(sig.rail_volts, 12.947, 1e-3)
    assert approx_equal(sig.supply_amps, 1.5, 1e-6)
    assert sig.power_bands is not None and len(sig.power_bands) == 10
    assert approx_equal(sig.power_lag, 0.02, 1e-6)
    assert approx_equal(sig.power_coherence, 0.4, 1e-6)
    assert approx_equal(sig.carrier_current_a, 0.97, 1e-6)
    assert sig.radiated_pwr_w is not None and sig.radiated_pwr_w < 1e-30


def test_field_probe_v3_absent_sensors_decode_none() -> None:
    """Mask bits clear → absent channels are None, never zero."""
    body = _v3_field_probe_body()
    body[178] = 0
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.power_watts is None
    assert sig.power_bands is None
    assert sig.rail_volts is None
    assert sig.supply_amps is None
    assert sig.power_lag is None
    assert sig.b_est_ut is None


def test_field_probe_v4_unpacks_radio_tail() -> None:
    """The v4 tail decodes the radio sense through its presence mask:
    WLAN flux + RSSI envelopes, wireless IRQ flux, HCI counters."""
    body = _v3_field_probe_body()
    body[0] = 4
    body += struct.pack("<10f", *[0.3] * 10)  # rf flux bands
    body += struct.pack("<10f", *[0.2] * 10)  # rf rssi bands
    body += struct.pack("<8f", 4200.0, -71.0, -95.0, 33.0, 4.0, 120.0, 18.0, -3.0)
    body.append(0x7F)  # all radio channels present
    assert len(body) == 308
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_flux_bands is not None and len(sig.rf_flux_bands) == 10
    assert all(approx_equal(v, 0.3, 1e-6) for v in sig.rf_flux_bands)
    assert sig.rf_rssi_bands is not None and len(sig.rf_rssi_bands) == 10
    assert all(approx_equal(v, 0.2, 1e-6) for v in sig.rf_rssi_bands)
    assert (
        sig.rf_bytes_s is not None
        and sig.rf_rssi_dbm is not None
        and sig.rf_noise_dbm is not None
        and sig.rf_irq_rate is not None
        and sig.rf_sensor_hz is not None
        and sig.bt_bytes_s is not None
        and sig.bt_events_s is not None
        and sig.bt_rssi_dbm is not None
    )
    assert approx_equal(sig.rf_bytes_s, 4200.0, 1e-2)
    assert approx_equal(sig.rf_rssi_dbm, -71.0, 1e-6)
    assert approx_equal(sig.rf_noise_dbm, -95.0, 1e-6)
    assert approx_equal(sig.rf_irq_rate, 33.0, 1e-6)
    assert approx_equal(sig.rf_sensor_hz, 4.0, 1e-6)
    assert approx_equal(sig.bt_bytes_s, 120.0, 1e-6)
    assert approx_equal(sig.bt_events_s, 18.0, 1e-6)
    assert approx_equal(sig.bt_rssi_dbm, -3.0, 1e-6)
    assert sig.rf_present == 0x7F


def test_field_probe_v4_partial_presence_gates_fields() -> None:
    """Only masked channels decode — an iface with no RSSI reports
    flux but None for field strength."""
    body = _v3_field_probe_body()
    body[0] = 4
    body += b"\x00" * 80
    body += b"\x00" * 33
    body[307] = 0x01  # wlan only
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_flux_bands is not None
    assert sig.rf_bytes_s is not None
    assert sig.rf_rssi_bands is None
    assert sig.rf_rssi_dbm is None
    assert sig.bt_bytes_s is None


def test_field_probe_v5_unpacks_station_tail() -> None:
    """The v5 tail decodes iw station telemetry through presence
    bit 7: link-strain rates and PHY link-adaptation bitrates."""
    body = _v3_field_probe_body()
    body[0] = 5
    body += struct.pack("<10f", *[0.3] * 10)  # rf flux bands
    body += struct.pack("<10f", *[0.2] * 10)  # rf rssi bands
    body += struct.pack("<8f", 4200.0, -55.0, -95.0, 33.0, 1.0, 120.0, 18.0, -3.0)
    body.append(0xFF)  # all channels present incl. iw station
    body += struct.pack("<5f", 12.0, 0.4, 0.1, 72.2, 65.0)
    assert len(body) == 328
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_rssi_dbm is not None and approx_equal(sig.rf_rssi_dbm, -55.0, 1e-6)
    assert sig.rf_retry_rate is not None
    assert sig.rf_fail_rate is not None
    assert sig.rf_beacon_loss_rate is not None
    assert sig.rf_tx_mbps is not None
    assert sig.rf_rx_mbps is not None
    assert approx_equal(sig.rf_retry_rate, 12.0, 1e-6)
    assert approx_equal(sig.rf_fail_rate, 0.4, 1e-6)
    assert approx_equal(sig.rf_beacon_loss_rate, 0.1, 1e-6)
    assert approx_equal(sig.rf_tx_mbps, 72.2, 1e-4)
    assert approx_equal(sig.rf_rx_mbps, 65.0, 1e-6)


def test_field_probe_v5_iw_absent_decodes_none() -> None:
    """v5 with the iw bit clear → station fields are None, not 0."""
    body = _v3_field_probe_body()
    body[0] = 5
    body += b"\x00" * 80  # band energies
    body += b"\x00" * 32  # scalars
    body.append(0x01)  # wlan only
    body += struct.pack("<5f", 9.0, 9.0, 9.0, 9.0, 9.0)  # ignored tail
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_retry_rate is None
    assert sig.rf_fail_rate is None
    assert sig.rf_beacon_loss_rate is None
    assert sig.rf_tx_mbps is None
    assert sig.rf_rx_mbps is None


def test_field_probe_v6_unpacks_flux_phasors() -> None:
    """The v6 tail decodes flux-envelope phase + coherence through
    the wlan presence bit — the entrainment path's raw material."""
    phases = [0.1 * i - 1.0 for i in range(10)]
    coh = [0.05 * i for i in range(10)]
    body = _v3_field_probe_body()
    body[0] = 6
    body += struct.pack("<10f", *[0.3] * 10)  # rf flux bands
    body += struct.pack("<10f", *[0.2] * 10)  # rf rssi bands
    body += struct.pack("<8f", 4200.0, -55.0, -95.0, 33.0, 1.0, 120.0, 18.0, -3.0)
    body.append(0xFF)
    body += struct.pack("<5f", 12.0, 0.4, 0.1, 72.2, 65.0)
    body += struct.pack("<10f", *phases)
    body += struct.pack("<10f", *coh)
    assert len(body) == 408
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_flux_phases is not None and len(sig.rf_flux_phases) == 10
    assert sig.rf_flux_phase_coherence is not None
    assert all(approx_equal(v, e, 1e-6) for v, e in zip(sig.rf_flux_phases, phases, strict=True))
    assert all(
        approx_equal(v, e, 1e-6) for v, e in zip(sig.rf_flux_phase_coherence, coh, strict=True)
    )


def test_field_probe_v6_no_wlan_decodes_no_phasors() -> None:
    """v6 with the wlan bit clear → phasor fields are None."""
    body = _v3_field_probe_body()
    body[0] = 6
    body += b"\x00" * 113  # v4 radio tail, all-zero
    body += struct.pack("<5f", 0.0, 0.0, 0.0, 0.0, 0.0)
    body += struct.pack("<10f", *[9.9] * 10)  # ignored phasor tail
    body += struct.pack("<10f", *[9.9] * 10)
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.rf_flux_phases is None
    assert sig.rf_flux_phase_coherence is None


def test_field_probe_v2_still_decodes() -> None:
    """A v2 (114-byte) body keeps working — electrical fields absent."""
    body = bytearray(114)
    body[0] = 2
    struct.pack_into("<20f", body, 1, *[0.125] * 20)
    struct.pack_into("<5f", body, 81, 1.0, 2.0, 3.0, 4.0, 5.0)
    body[101] = 1
    struct.pack_into("<3f", body, 102, 0.1, 0.2, 0.3)
    sig = FieldProbeSignature.unpack(bytes(body))
    assert sig.carrier_lag is not None
    assert approx_equal(sig.carrier_lag, 0.1, 1e-6)
    assert sig.power_watts is None
    assert sig.power_bands is None


def test_core_state_rejects_truncated_body() -> None:
    """A short body is a refusal, not a misread."""
    data = bytearray(3400)
    try:
        CoreState.unpack(bytes(data))
        raise AssertionError()
    except ValueError as e:
        logger.debug(repr(e))

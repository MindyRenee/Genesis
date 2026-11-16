"""InteroceptiveRelay — the NTS/PBN afferent relay.

Verifies the anatomical routing: threat channels go to the limbic
low road (PBN→CeA via ``condition_cues``), source-level monoamine
impulses are self-scaled by live levels (positive × headroom,
negative × level), and sensor-gated arms fail closed when no
presence report exists.
"""

from __future__ import annotations

from typing import Any

import pytest
from brainstem.interoceptive_relay import InteroceptiveRelay
from genesis_client.protocol import (
    CHEM_HISTAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    SENSOR_CPUIDLE,
    SENSOR_FAN,
    SENSOR_TEMPERATURE,
)
from genesis_client.types import BodyState, SensorPresence

FULL_PRESENCE = SensorPresence(mask=SENSOR_CPUIDLE | SENSOR_TEMPERATURE | SENSOR_FAN)


def make_body(**over) -> BodyState:
    base: dict[str, Any] = {
        "cpu_temp_c": 50.0,
        "temperature": 0.5,
        "arousal_freq": 1.0,
        "cognitive_load": 0.2,
        "io_activity": 0.0,
        "stress_load": 0.1,
        "energy_reserve": 1.0,
        "on_ac_power": True,
        "num_cores": 4,
        "distressed": False,
        "autonomic_rate": 2.0,
        "thermoregulatory_effort": 0.0,
        "metabolic_rate": 0.1,
        "core_voltage": 1.0,
        "supply_voltage": 12.6,
        "description": "",
    }
    base.update(over)
    return BodyState(**base)


def make_relay(levels=None, cues_return=0.0):
    """A relay wired to recording fakes. Returns (relay, calls)."""
    calls: dict = {"impulses": [], "cues": []}
    lv = dict(levels) if levels is not None else {}

    relay = InteroceptiveRelay(
        neuro_impulse=lambda chem, mag: calls["impulses"].append((chem, mag)),
        get_levels=lambda: lv,
        condition_cues=lambda cues, us, ctx: (
            calls["cues"].append((list(cues), us, ctx)) or cues_return
        ),
    )
    return relay, calls


class TestThreatArm:
    def test_heat_reaches_the_low_road(self):
        relay, calls = make_relay()
        out = relay.relay(make_body(cpu_temp_c=85.0), FULL_PRESENCE)
        assert "body:heat" in out["threat_cues"]
        assert calls["cues"], "threat channels must reach the CeA low road"
        cues, us, ctx = calls["cues"][0]
        assert "body:heat" in cues and ctx == "interoception"
        assert us < 0.0, "threat US is aversive"

    def test_overload_fires_the_sympathetic_arousal_arm(self):
        relay, calls = make_relay()
        out = relay.relay(make_body(stress_load=1.4), FULL_PRESENCE)
        sent = dict(calls["impulses"])
        assert sent[CHEM_NOREPINEPHRINE] > 0
        assert "body:overload" in out["threat_cues"]

    def test_host_contention_is_not_an_interoceptive_channel(self):
        relay, calls = make_relay()
        out = relay.relay(make_body(psi_cpu=0.6, psi_io=0.7, psi_mem=0.8), FULL_PRESENCE)
        assert out["threat_cues"] == []
        assert calls["impulses"] == []

    def test_cooling_failure_needs_a_fan_sensor(self):
        relay, _ = make_relay()
        body = make_body(cpu_temp_c=90.0, thermoregulatory_effort=0.0)
        with_fan = relay.relay(body, SensorPresence(mask=SENSOR_FAN))
        without = relay.relay(body, SensorPresence(mask=0))
        assert "body:cooling_failure" in with_fan["threat_cues"]
        assert "body:cooling_failure" not in without["threat_cues"]

    def test_no_threat_no_low_road(self):
        relay, calls = make_relay()
        out = relay.relay(make_body(), FULL_PRESENCE)
        assert out["threat_cues"] == []
        assert calls["cues"] == []

    def test_every_active_threat_reaches_one_low_road_call(self):
        relay, calls = make_relay()
        relay.relay(
            make_body(cpu_temp_c=90.0, stress_load=1.5, cognitive_load=0.9),
            FULL_PRESENCE,
        )
        cues = calls["cues"][0][0]
        assert "body:heat" in cues
        assert "body:overload" in cues
        assert "body:cooling_failure" in cues
        assert "body:memory_pressure" not in cues


class TestParasympatheticArm:
    def test_deep_rest_withdraws_arousal_tone(self):
        # Alerting tone present so the withdrawal arm has something
        # to withdraw (negative impulses scale with live level).
        relay, calls = make_relay(
            levels={
                "norepinephrine": 0.5,
                "orexin": 0.5,
                "histamine": 0.5,
            }
        )
        out = relay.relay(make_body(rest=0.8), FULL_PRESENCE)
        sent = dict(calls["impulses"])
        assert sent[CHEM_NOREPINEPHRINE] < 0
        assert sent[CHEM_OREXIN] < 0
        assert sent[CHEM_HISTAMINE] < 0
        assert not any(mag > 0 for _, mag in calls["impulses"])
        assert out["threat_cues"] == []

    def test_rest_silent_without_the_cpuidle_sensor(self):
        relay, calls = make_relay()
        relay.relay(make_body(rest=0.9), SensorPresence(mask=0))
        sent = dict(calls["impulses"])
        # Absence of the sensor is not evidence of rest.
        assert sent == {}
        assert not any(m < 0 for _, m in calls["impulses"])

    def test_cooling_trend_is_not_an_ecb_signal(self):
        relay, calls = make_relay()
        relay.relay(
            make_body(rest=0.0, temp_trend=-0.3, cpu_temp_c=50.0),
            FULL_PRESENCE,
        )
        assert calls["impulses"] == []


class TestSelfSettingMagnitudes:
    def test_positive_impulses_scale_with_headroom(self):
        """Same signal, two states: near-empty level pushes ~7x
        harder than a level already near its ceiling."""
        low, _ = make_relay(levels={"norepinephrine": 0.1})
        high, _ = make_relay(levels={"norepinephrine": 0.9})
        calls_low, calls_high = [], []
        low._neuro_impulse = lambda c, m: calls_low.append((c, m))
        high._neuro_impulse = lambda c, m: calls_high.append((c, m))
        body = make_body(cpu_temp_c=85.0)
        low.relay(body, FULL_PRESENCE)
        high.relay(body, FULL_PRESENCE)
        ne_low = dict(calls_low)[CHEM_NOREPINEPHRINE]
        ne_high = dict(calls_high)[CHEM_NOREPINEPHRINE]
        assert 0 < ne_high < ne_low
        assert ne_low / ne_high == pytest.approx(9.0)  # (1−0.1)/(1−0.9)

    def test_saturated_chemical_gets_no_offer(self):
        relay, calls = make_relay(levels={"norepinephrine": 1.0})
        relay.relay(make_body(cpu_temp_c=85.0), FULL_PRESENCE)
        sent = dict(calls["impulses"])
        assert CHEM_NOREPINEPHRINE not in sent

    def test_negative_impulses_scale_with_level(self):
        """Withdrawal cannot exceed the tone that exists."""
        depleted, _ = make_relay(levels={"norepinephrine": 0.0, "orexin": 0.5})
        toned, _ = make_relay(levels={"norepinephrine": 0.5, "orexin": 0.5})
        c_dep, c_tone = [], []
        depleted._neuro_impulse = lambda c, m: c_dep.append((c, m))
        toned._neuro_impulse = lambda c, m: c_tone.append((c, m))
        body = make_body(rest=0.9)
        depleted.relay(body, FULL_PRESENCE)
        toned.relay(body, FULL_PRESENCE)
        ne_dep = dict(c_dep).get(CHEM_NOREPINEPHRINE)
        ne_tone = dict(c_tone).get(CHEM_NOREPINEPHRINE)
        assert ne_dep is None, "no tone to withdraw — offer must not be sent"
        assert ne_tone is not None and ne_tone < 0


class TestNonBiologicalChannels:
    def test_effort_raises_lc_drive(self):
        relay, calls = make_relay()
        relay.relay(make_body(stress_load=0.8), FULL_PRESENCE)
        sent = dict(calls["impulses"])
        assert sent[CHEM_NOREPINEPHRINE] > 0

    def test_silicon_activity_is_not_neuromodulation(self):
        relay, calls = make_relay()
        relay.relay(
            make_body(io_activity=0.8, core_activity=0.9, dram_activity=0.9),
            FULL_PRESENCE,
        )
        assert calls["impulses"] == []

    def test_hardware_prediction_errors_do_not_orient(self):
        relay, calls = make_relay()
        relay.relay(make_body(branch_miss_rate=0.9, cache_miss_rate=0.9), FULL_PRESENCE)
        assert calls["impulses"] == []


class TestResilience:
    def test_no_wiring_is_silent(self):
        """Unwired relay still computes the summary, sends nothing."""
        relay = InteroceptiveRelay()
        out = relay.relay(make_body(cpu_temp_c=95.0), FULL_PRESENCE)
        assert "body:heat" in out["threat_cues"]

    def test_failed_level_read_falls_back_to_full_headroom(self):
        relay, calls = make_relay(levels=None)
        out = relay.relay(make_body(stress_load=0.8), FULL_PRESENCE)
        assert dict(calls["impulses"]).get(CHEM_NOREPINEPHRINE, 0.0) > 0
        assert out["impulses"]

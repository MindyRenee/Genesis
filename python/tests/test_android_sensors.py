"""Tests for Android sensor payload parsing and sampling thresholds."""
from __future__ import annotations

from brainstem.android_sensors import _readings, _sensor_names


def test_sensor_names_are_filtered_to_supported_physical_metrics() -> None:
    payload = (
        '[{"name":"Accelerometer","type":1},'
        '{"name":"Light","type":5},{"name":"GPS provider","type":99}]'
    )
    assert _sensor_names(payload) == ["Accelerometer", "Light"]


def test_readings_preserve_units_values_and_ignore_nonfinite_data() -> None:
    payload = (
        '{"Accelerometer":{"name":"Accelerometer","values":[0.1,0.2,9.7],'
        '"accuracy":3,"timestamp":1234}}'
    )
    parsed = _readings(payload)
    assert parsed == [
        {
            "name": "Accelerometer",
            "values": [0.1, 0.2, 9.7],
            "accuracy": 3,
            "timestamp": 1234,
        }
    ]
    bad = '{"Light":{"name":"Light","values":[NaN]}}'
    assert _readings(bad) == []


def test_readings_accept_json_lines() -> None:
    payload = (
        '{"Light":{"name":"Light","values":[12.0]}}\n'
        '{"Proximity":{"name":"Proximity","values":[3.0]}}\n'
    )
    assert [item["name"] for item in _readings(payload)] == ["Light", "Proximity"]



def test_android_extra_sensor_names_are_discovered() -> None:
    payload = '[{"name":"Gravity"},{"name":"Ambient Temperature"},{"name":"Step Counter"}]'
    assert _sensor_names(payload) == ["Gravity", "Ambient Temperature", "Step Counter"]

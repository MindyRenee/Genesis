"""Optional Termux:API adapter for real Android device sensors.

Enable with GENESIS_ANDROID_SENSORS=1. Requires the Termux:API companion app
and package. Missing sensors, permissions, or API failures are reported as
absence; this module never substitutes invented zero readings.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_SENSOR_KEYWORDS = (
    "accelerometer",
    "gyroscope",
    "magnetic field",
    "magnetometer",
    "light",
    "proximity",
    "pressure",
    "rotation vector",
)
_SENSOR_UNITS = {
    "accelerometer": "m/s^2",
    "gyroscope": "rad/s",
    "magnetic field": "uT",
    "magnetometer": "uT",
    "light": "lux",
    "proximity": "cm",
    "pressure": "hPa",
    "rotation vector": "unitless",
}
_THRESHOLDS = {
    "accelerometer": 0.15,
    "gyroscope": 0.05,
    "magnetic field": 2.0,
    "magnetometer": 2.0,
    "light": 5.0,
    "proximity": 1.0,
    "pressure": 0.5,
    "rotation vector": 0.03,
}


def is_android_termux() -> bool:
    """Return true only for an Android environment with the Termux command."""
    android = bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"))
    return android and shutil.which("termux-sensor") is not None


def _json_documents(output: str) -> list[Any]:
    """Parse JSON output whether the API emits one document or JSON lines."""
    try:
        return [json.loads(output)]
    except json.JSONDecodeError:
        documents = []
        for line in output.splitlines():
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return documents


def _sensor_names(output: str) -> list[str]:
    names: list[str] = []
    for document in _json_documents(output):
        items = document if isinstance(document, list) else [document]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("Name")
            if isinstance(name, str) and any(key in name.casefold() for key in _SENSOR_KEYWORDS):
                if name not in names:
                    names.append(name)
    return names


def _readings(output: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for document in _json_documents(output):
        if isinstance(document, list):
            entries = [
                (str(item.get("name", "sensor")), item)
                for item in document
                if isinstance(item, dict)
            ]
        elif isinstance(document, dict):
            entries = list(document.items())
        else:
            continue
        for key, item in entries:
            if not isinstance(item, dict):
                continue
            name = item.get("name", key)
            values = item.get("values")
            if not isinstance(name, str) or not isinstance(values, list) or not values:
                continue
            numeric: list[float] = []
            valid = True
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    valid = False
                    break
                number = float(value)
                if not math.isfinite(number):
                    valid = False
                    break
                numeric.append(number)
            if not valid:
                continue
            accuracy = item.get("accuracy")
            timestamp = item.get("timestamp")
            results.append(
                {
                    "name": name,
                    "values": numeric,
                    "accuracy": accuracy if isinstance(accuracy, (int, str)) else None,
                    "timestamp": timestamp if isinstance(timestamp, (int, float)) else None,
                }
            )
    return results


class AndroidSensorAdapter:
    """Low-rate, permission-aware sensor sampler for the OuterWorld event bus."""

    def __init__(
        self,
        on_sample: Callable[[dict[str, Any]], None],
        *,
        interval_seconds: float = 2.0,
        command_timeout: float = 8.0,
    ) -> None:
        if not callable(on_sample):
            raise TypeError("on_sample must be callable")
        if not math.isfinite(interval_seconds) or interval_seconds < 1.0:
            raise ValueError("interval_seconds must be at least 1 second")
        if not math.isfinite(command_timeout) or command_timeout < 1.0:
            raise ValueError("command_timeout must be at least 1 second")
        self._on_sample = on_sample
        self._interval = interval_seconds
        self._timeout = command_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._names: list[str] = []
        self._last_values: dict[str, list[float]] = {}
        self._last_error_log = 0.0

    @property
    def available_sensors(self) -> tuple[str, ...]:
        return tuple(self._names)

    def start(self) -> bool:
        """Discover available hardware first; start only if sensors are present."""
        if self._thread is not None and self._thread.is_alive():
            return True
        if not is_android_termux():
            return False
        try:
            result = subprocess.run(
                ["termux-sensor", "-l"],
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            return False
        self._names = _sensor_names(result.stdout)
        if not self._names:
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="genesis-android-sensors",
            daemon=True,
        )
        self._thread.start()
        return True

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None
        if is_android_termux():
            try:
                subprocess.run(
                    ["termux-sensor", "-c"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(self._timeout, 3.0),
                )
            except (OSError, subprocess.SubprocessError):
                pass

    def _run(self) -> None:
        sensor_arg = ",".join(self._names)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                result = subprocess.run(
                    [
                        "termux-sensor",
                        "-s",
                        sensor_arg,
                        "-d",
                        "250",
                        "-n",
                        "1",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
                for reading in _readings(result.stdout):
                    if not self._changed(reading):
                        continue
                    try:
                        self._on_sample(reading)
                    except Exception:  # noqa: BLE001 — sensor failures cannot kill cognition
                        logger.exception("Android sensor callback failed")
            except (OSError, subprocess.SubprocessError, ValueError):
                now = time.monotonic()
                if now - self._last_error_log > 60.0:
                    logger.warning("Termux sensor sampling unavailable; retrying")
                    self._last_error_log = now
            remaining = self._interval - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))

    def _changed(self, reading: dict[str, Any]) -> bool:
        name = str(reading["name"])
        values = reading["values"]
        key = name.casefold()
        previous = self._last_values.get(name)
        self._last_values[name] = list(values)
        if previous is None or len(previous) != len(values):
            return True
        sensor_type = next((item for item in _SENSOR_KEYWORDS if item in key), "")
        threshold = _THRESHOLDS.get(sensor_type, 0.1)
        return max(abs(a - b) for a, b in zip(values, previous, strict=True)) >= threshold

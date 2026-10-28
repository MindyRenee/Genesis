"""Voice prosody — the thermal channel.

Body temperature enters ``Voice.speak(temp_c=...)`` as a fever
voice: slower rate over the same 65–85 °C ramp the daemon's
heat-stress impulse uses, plus duration-noise and breath pauses in
the Piper backend. An absent reading (``temp_c=None``) must leave
behaviour identical to before.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
from temporal_lobe.speech import Voice, _thermal_strain


class _FakeStdin:
    def write(self, _b: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeProc:
    """Popen stand-in that captures argv and finishes instantly."""

    def __init__(self, cmd: list[str], captured: list[list[str]], **_kw: Any) -> None:
        captured.append(cmd)
        self.stdin = _FakeStdin()
        self.stdout = None

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def poll(self) -> int:
        return 0


def test_thermal_strain_ramp() -> None:
    """The fever ramp matches the daemon's heat-stress band."""
    assert _thermal_strain(None) == 0.0
    assert _thermal_strain(52.0) == 0.0
    assert _thermal_strain(65.0) == 0.0
    assert _thermal_strain(75.0) == pytest.approx(0.5)
    assert _thermal_strain(85.0) == pytest.approx(1.0)
    assert _thermal_strain(95.0) == pytest.approx(1.0)


def test_piper_fever_flags(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Heat adds duration-noise and breath pauses; cool adds nothing."""
    captured: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, **kw: _FakeProc(cmd, captured, **kw)
    )

    voice = Voice.__new__(Voice)
    voice._piper_available = True
    voice._espeak_available = False
    voice._model = tmp_path / "model.onnx"
    voice._sample_rate = 22050
    voice._all_procs = []
    voice._procs_lock = __import__("threading").Lock()
    voice._speaking_count = 0
    voice._speaking_lock = __import__("threading").Lock()
    voice._last_spoke_time = 0.0
    voice._speech_tail = 0.3
    voice._speech_queue = __import__("collections").deque(maxlen=3)
    voice._queue_lock = __import__("threading").Lock()
    voice._piper_proc = None
    voice._play_proc = None

    captured.clear()
    voice._speak_piper("hello", True, 1.25, 0.0)
    cool = captured[0]
    assert "--noise-w-scale" not in cool
    assert "--sentence-silence" not in cool

    captured.clear()
    voice._speak_piper("hello", True, 1.25, 1.0)
    hot = captured[0]
    w_idx = hot.index("--noise-w-scale") + 1
    s_idx = hot.index("--sentence-silence") + 1
    assert float(hot[w_idx]) == pytest.approx(1.3)
    assert float(hot[s_idx]) == pytest.approx(0.7)


def test_speak_queues_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued utterance keeps its thermal channel."""
    voice = Voice.__new__(Voice)
    voice._piper_available = False
    voice._espeak_available = False
    voice._all_procs = []
    voice._procs_lock = __import__("threading").Lock()
    voice._speaking_count = 1  # pretend speaking so speak() queues
    voice._speaking_lock = __import__("threading").Lock()
    voice._last_spoke_time = 10**12  # far future → is_speaking True
    voice._speech_tail = 0.3
    voice._speech_queue = __import__("collections").deque(maxlen=3)
    voice._queue_lock = __import__("threading").Lock()
    voice._piper_proc = None
    voice._play_proc = None

    voice.speak("queued words", temp_c=80.0)
    assert voice._speech_queue
    queued = voice._speech_queue[0]
    assert queued[0] == "queued words"
    assert queued[5] == 80.0

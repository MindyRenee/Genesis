"""Tests for the Android system speech fallback."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

from temporal_lobe.speech import VoiceInput, _can_use_termux_speech


def test_termux_speech_detection_requires_android_and_command(monkeypatch) -> None:
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setattr("temporal_lobe.speech.shutil.which", lambda name: "/bin/" + name)
    assert _can_use_termux_speech()
    monkeypatch.delenv("ANDROID_ROOT")
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    assert not _can_use_termux_speech()


def test_termux_speech_returns_transcript(monkeypatch) -> None:
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setattr("temporal_lobe.speech.shutil.which", lambda name: "/bin/" + name)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="  turn left  \n"),
    )
    assert VoiceInput._listen_termux_speech(timeout=2.0, phrase_limit=5.0) == "turn left"


def test_termux_speech_failure_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("ANDROID_ROOT", "/system")
    monkeypatch.setattr("temporal_lobe.speech.shutil.which", lambda name: "/bin/" + name)

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("termux-speech-to-text", 15)

    monkeypatch.setattr(subprocess, "run", fail)
    assert VoiceInput._listen_termux_speech(timeout=2.0, phrase_limit=5.0) is None

"""Tests for the acoustic self-emission channel.

The contract under test:

- no device → absent channel: no stimulus, False, never a zero;
- a live emission signature lands a small GAMMA stimulus;
- an acoustic transient resets GAMMA phase (the ERP analog);
- GENESIS_FIELD_AUDIO=0 disables the channel outright.
"""

import pytest
from thalamus.brain_waves import BrainWave


class _StubListener:
    """A listener-shaped stub — no hardware touched."""

    def __init__(self, sig):
        self._sig = sig

    def signature(self):
        return self._sig


@pytest.fixture(autouse=True)
def _clean_oscillator():
    from thalamus import brain_waves

    brain_waves.reset_oscillator()
    yield
    brain_waves.reset_oscillator()


def test_absent_device_injects_nothing(monkeypatch) -> None:
    from brainstem import field_audio
    from thalamus import brain_waves

    monkeypatch.setattr(field_audio, "_get_listener", lambda: None)
    seen = []
    monkeypatch.setattr(
        brain_waves,
        "hold_stimulus",
        lambda b, s: seen.append((b, s)),
    )
    assert field_audio.apply_audio_stimulus() is False
    assert seen == []


def test_unwarmed_listener_injects_nothing(monkeypatch) -> None:
    """A listener that opened but has produced no blocks yet is still
    an absent channel — not a zero reading."""
    from brainstem import field_audio
    from thalamus import brain_waves

    monkeypatch.setattr(field_audio, "_get_listener", lambda: _StubListener(None))
    seen = []
    monkeypatch.setattr(
        brain_waves,
        "hold_stimulus",
        lambda b, s: seen.append((b, s)),
    )
    assert field_audio.apply_audio_stimulus() is False
    assert seen == []


def test_live_signature_stimulates_gamma(monkeypatch) -> None:
    from brainstem import field_audio
    from thalamus import brain_waves

    sig = {
        "emission_rel": 2.0,
        "emission": 0.01,
        "rms": 0.02,
        "transient": False,
        "blocks": 100.0,
    }
    monkeypatch.setattr(field_audio, "_get_listener", lambda: _StubListener(sig))
    seen = []
    monkeypatch.setattr(
        brain_waves,
        "hold_stimulus",
        lambda b, s: seen.append((b, s)),
    )
    assert field_audio.apply_audio_stimulus() is True
    assert seen == [(BrainWave.GAMMA, pytest.approx(0.16))]


def test_transient_resets_gamma_phase(monkeypatch) -> None:
    from brainstem import field_audio
    from thalamus import brain_waves

    sig = {
        "emission_rel": 1.5,
        "emission": 0.01,
        "rms": 0.02,
        "transient": True,
        "blocks": 100.0,
    }
    monkeypatch.setattr(field_audio, "_get_listener", lambda: _StubListener(sig))
    brain_waves.perturb_wave_phase(BrainWave.GAMMA, 1.9)
    assert field_audio.apply_audio_stimulus() is True
    assert brain_waves.wave_phase_of(BrainWave.GAMMA) == 0.0


def test_env_disable_is_absent(monkeypatch) -> None:
    from brainstem import field_audio

    monkeypatch.setenv("GENESIS_FIELD_AUDIO", "0")
    monkeypatch.setattr(field_audio, "_listener", None)
    monkeypatch.setattr(field_audio, "_listener_failed", False)
    assert field_audio._get_listener() is None
    assert field_audio.apply_audio_stimulus() is False

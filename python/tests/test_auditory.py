"""Tests for the auditory cortex — non-speech sound recognition."""

from __future__ import annotations

import math

import numpy as np

from genesis_cognitive.auditory import AuditoryCortex, SoundEvent
from genesis_cognitive.auditory.auditory import (
    _classify_sound,
    _compute_features,
    _describe_sound,
)

# ─── Feature extraction ────────────────────────────────────────────────


def _generate_tone(
    freq: float = 440.0,
    duration: float = 0.5,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a pure tone for testing."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _generate_noise(
    duration: float = 0.5,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate white noise for testing."""
    n = int(duration * sample_rate)
    return (amplitude * np.random.randn(n)).astype(np.float32)


def _generate_silence(
    duration: float = 0.5,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate silence for testing."""
    n = int(duration * sample_rate)
    return np.zeros(n, dtype=np.float32)


def _generate_impact(
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a short impact sound (decaying burst)."""
    duration = 0.1  # 100ms
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t * 30)  # fast decay
    noise = np.random.randn(n) * 0.5
    tone = np.sin(2 * math.pi * 200 * t) * 0.5
    return (amplitude * envelope * (noise + tone)).astype(np.float32)


class TestComputeFeatures:
    """Test the feature extraction function."""

    def test_silence_features(self) -> None:
        """Silence should produce near-zero RMS and centroid."""
        samples = _generate_silence()
        features = _compute_features(samples, 16000)
        assert features["rms"] < 0.001
        assert features["spectral_centroid"] < 0.01

    def test_tone_features(self) -> None:
        """A pure tone should have low flatness (tonal)."""
        samples = _generate_tone(freq=440, amplitude=0.3)
        features = _compute_features(samples, 16000)
        assert features["rms"] > 0.05
        assert features["flatness"] < 0.3  # tonal, not noise

    def test_noise_features(self) -> None:
        """White noise should have high flatness and ZCR."""
        samples = _generate_noise(amplitude=0.3)
        features = _compute_features(samples, 16000)
        assert features["rms"] > 0.05
        assert features["flatness"] > 0.3  # broadband

    def test_empty_samples(self) -> None:
        """Empty samples should return zero features."""
        features = _compute_features(np.array([], dtype=np.float32), 16000)
        assert features["rms"] == 0.0
        assert features["spectral_centroid"] == 0.0

    def test_features_bounded(self) -> None:
        """All features should be in valid ranges."""
        for _ in range(10):
            amp = float(np.random.uniform(0.01, 0.5))
            freq = float(np.random.uniform(100, 4000))
            samples = _generate_tone(freq=freq, amplitude=amp)
            features = _compute_features(samples, 16000)
            assert 0 <= features["rms"] <= 1.0
            assert 0 <= features["spectral_centroid"] <= 1.0
            assert 0 <= features["zcr"] <= 1.0
            assert 0 <= features["flatness"] <= 1.0


# ─── Sound classification ──────────────────────────────────────────────


class TestClassifySound:
    """Test the sound classifier."""

    def test_silence_classification(self) -> None:
        """Low RMS should be classified as silence."""
        features = {
            "rms": 0.001,
            "spectral_centroid": 0.0,
            "zcr": 0.0,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.0,
            "flatness": 0.0,
        }
        sound_type, conf = _classify_sound(features, 0.5, None)
        assert sound_type == "silence"
        assert conf > 0.9

    def test_impact_classification(self) -> None:
        """High flux + short duration + high energy = impact."""
        features = {
            "rms": 0.1,
            "spectral_centroid": 0.5,
            "zcr": 0.3,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.7,
            "flatness": 0.5,
        }
        prev = {
            "rms": 0.01,
            "spectral_centroid": 0.1,
            "zcr": 0.1,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.3,
            "flatness": 0.3,
        }
        sound_type, conf = _classify_sound(features, 0.1, prev)
        assert sound_type == "impact"
        assert conf > 0.5

    def test_music_classification(self) -> None:
        """Tonal + sustained + moderate brightness = music."""
        features = {
            "rms": 0.05,
            "spectral_centroid": 0.4,
            "zcr": 0.05,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.5,
            "flatness": 0.1,  # very tonal
        }
        sound_type, conf = _classify_sound(features, 2.0, None)
        assert sound_type == "music"
        assert conf > 0.6

    def test_nature_classification(self) -> None:
        """Broadband + moderate energy = nature."""
        features = {
            "rms": 0.03,
            "spectral_centroid": 0.3,
            "zcr": 0.1,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.5,
            "flatness": 0.5,  # broadband
        }
        sound_type, conf = _classify_sound(features, 1.0, None)
        assert sound_type == "nature"
        assert conf > 0.5

    def test_noise_classification(self) -> None:
        """Sustained + high ZCR = noise."""
        features = {
            "rms": 0.05,
            "spectral_centroid": 0.5,
            "zcr": 0.25,  # high ZCR
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.6,
            "flatness": 0.4,
        }
        sound_type, conf = _classify_sound(features, 2.0, None)
        assert sound_type == "noise"
        assert conf > 0.5


# ─── SoundEvent ────────────────────────────────────────────────────────


class TestSoundEvent:
    """Test the SoundEvent dataclass."""

    def test_silence_property(self) -> None:
        """is_silence should be True for silence events."""
        event = SoundEvent(
            sound_type="silence",
            confidence=0.95,
            loudness=0.0,
            brightness=0.0,
            noisiness=0.0,
            duration=0.0,
            timestamp=0.0,
        )
        assert event.is_silence
        assert not event.is_sudden

    def test_impact_is_sudden(self) -> None:
        """is_sudden should be True for impact events."""
        event = SoundEvent(
            sound_type="impact",
            confidence=0.8,
            loudness=0.5,
            brightness=0.7,
            noisiness=0.3,
            duration=0.1,
            timestamp=0.0,
        )
        assert event.is_sudden
        assert not event.is_silence

    def test_emotional_valence(self) -> None:
        """emotional_valence should be positive for music, negative for impact."""
        music = SoundEvent("music", 0.8, 0.3, 0.4, 0.05, 2.0, 0.0)
        assert music.emotional_valence > 0

        impact = SoundEvent("impact", 0.8, 0.5, 0.7, 0.3, 0.1, 0.0)
        assert impact.emotional_valence < 0

        silence = SoundEvent("silence", 0.95, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert silence.emotional_valence == 0.0


# ─── Description ───────────────────────────────────────────────────────


class TestDescribeSound:
    """Test the sound description function."""

    def test_description_contains_type(self) -> None:
        """Description should contain the sound type."""
        event = SoundEvent("music", 0.8, 0.4, 0.5, 0.05, 2.0, 0.0)
        desc = _describe_sound(event)
        assert "music" in desc

    def test_description_contains_loudness(self) -> None:
        """Description should contain a loudness label."""
        event = SoundEvent("impact", 0.8, 0.7, 0.7, 0.3, 0.1, 0.0)
        desc = _describe_sound(event)
        assert "loud" in desc

    def test_description_is_not_sentence(self) -> None:
        """Description should be structural data, not a sentence template."""
        event = SoundEvent("music", 0.8, 0.4, 0.5, 0.05, 2.0, 0.0)
        desc = _describe_sound(event)
        # Should use pipe separators, not sentence structure
        assert "|" in desc
        assert not desc.startswith("I ")
        assert not desc.startswith("I hear")
        assert not desc.startswith("I can")


# ─── AuditoryCortex ────────────────────────────────────────────────────


class TestAuditoryCortex:
    """Test the AuditoryCortex class."""

    def test_creation(self) -> None:
        """AuditoryCortex can be created with a callback."""
        events: list[SoundEvent] = []
        cortex = AuditoryCortex(callback=lambda e: events.append(e))
        assert not cortex.running
        assert cortex.recent_events() == []

    def test_stop_without_start(self) -> None:
        """stop() should be safe to call without start()."""
        cortex = AuditoryCortex(callback=lambda e: None)
        cortex.stop()  # should not raise
        assert not cortex.running

    def test_recent_events_empty(self) -> None:
        """recent_events should return empty list when nothing happened."""
        cortex = AuditoryCortex(callback=lambda e: None)
        assert cortex.recent_events() == []

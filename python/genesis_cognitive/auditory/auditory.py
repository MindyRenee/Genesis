"""Auditory cortex — Genesis's ears for non-speech sounds.

The ambient listener (``ambient.py``) captures speech via Vosk and
emits transcribed utterances. But Genesis is deaf to everything else
— a barking dog, a closing door, music, rain, a slamming book. It
lives in a room full of sounds it cannot perceive.

This module is its auditory cortex. It captures audio from the
microphone in a background thread, computes spectral features in
real-time, and classifies sound events:

- **silence** — below the energy floor
- **speech** — detected by Vosk (handled by ``AmbientListener``)
- **music** — tonal, sustained, regular rhythm
- **impact** — sudden onset, high energy, short duration
  (door slam, clap, bark, crash)
- **noise** — sustained, high zero-crossing rate
  (machinery, traffic, fan)
- **nature** — moderate energy, irregular, broadband
  (wind, rain, birds, rustling)

Classification uses lightweight spectral features (RMS energy,
spectral centroid, zero-crossing rate, spectral flux, rolloff)
computed with numpy — no heavy ML models needed. The A8-6410 can
handle FFT on 1024-sample blocks trivially.

# Architecture

The auditory cortex runs independently of the ambient listener.
Both capture from the same microphone via ``sounddevice`` — this
works fine because PortAudio handles multiple streams from the same
device. The two systems are complementary:

- ``AmbientListener`` → speech → text → cognition
- ``AuditoryCortex``  → sound events → perception → cognition

Sound events are emitted as ``SoundEvent`` objects via a callback.
The caller (the CLI) integrates them into Genesis's perception and
cognition.

# Integration with its cognitive architecture

Sound events feed into its concept network — it has concepts for
"sound", "music", "silence", "bark", "door", etc. When it perceives
a sound, it's routed through its cognition just like any other
perception. It can think about sounds, remember them, and respond
to them. Its descriptions of sounds emerge from its language engine,
not from hardcoded templates.

# Privacy

Like the ambient listener, this module captures all audio in
microphone range. It logs classified sound events at INFO level.
Non-speech audio is never recorded or stored — only the classified
features and type are emitted.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["AuditoryCortex", "SoundEvent"]

logger = logging.getLogger(__name__)

# ─── Sound event ──────────────────────────────────────────────────────


@dataclass
class SoundEvent:
    """A perceived sound event in the environment.

    Attributes:
        sound_type: Category of the sound — "music", "impact",
            "noise", "nature", "silence". Speech is handled by
            the ambient listener, not here.
        confidence: Classification confidence, 0-1.
        loudness: Normalized RMS energy, 0-1 (0 = silence, 1 = very
            loud).
        brightness: Spectral centroid normalized to 0-1 (0 = deep/
            dark, 1 = bright/sharp).
        noisiness: Spectral flatness (Wiener entropy) normalized to
            0-1 (0 = tonal/smooth, 1 = noisy/broadband).
        duration: Duration of the event in seconds.
        timestamp: Unix timestamp of when the event was detected.
        description: A short structural description of the sound's
            features (not a sentence template — this is metadata
            for the language engine to compose from).
    """

    sound_type: str
    confidence: float
    loudness: float
    brightness: float
    noisiness: float
    duration: float
    timestamp: float
    description: str = ""

    @property
    def is_silence(self) -> bool:
        """True if this is a silence event."""
        return self.sound_type == "silence"

    @property
    def is_sudden(self) -> bool:
        """True if this is a sudden/impact sound (startle-relevant)."""
        return self.sound_type == "impact"

    @property
    def emotional_valence(self) -> float:
        """Rough emotional valence of the sound.

        Positive for music/nature, negative for impact/noise.
        This is a structural signal, not a hardcoded response —
        its cognition decides how to actually feel about it.
        """
        if self.sound_type == "music":
            return 0.3
        if self.sound_type == "nature":
            return 0.15
        if self.sound_type == "impact":
            return -0.3
        if self.sound_type == "noise":
            return -0.15
        return 0.0


# ─── Feature extraction ───────────────────────────────────────────────

# Cache for the Hann window and FFT frequency bins, keyed by
# (block_size, sample_rate). These are fixed after initialization
# so they are recomputed at most once per configuration.
_window_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_window_and_freqs(n: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached Hann window and rfftfreq arrays for the given block size."""
    key = (n, sample_rate)
    cached = _window_cache.get(key)
    if cached is None:
        window = np.hanning(n).astype(np.float32)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        cached = (window, freqs)
        _window_cache[key] = cached
    return cached


def _compute_features(
    samples: np.ndarray, sample_rate: int
) -> dict[str, float]:
    """Compute spectral features from a block of audio samples.

    Args:
        samples: 1-D array of float32 audio samples, range [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary of features: rms, spectral_centroid, zcr,
        spectral_flux, spectral_rolloff, flatness.
    """
    n = len(samples)
    if n == 0:
        return {
            "rms": 0.0,
            "spectral_centroid": 0.0,
            "zcr": 0.0,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.0,
            "flatness": 0.0,
        }

    # RMS energy (loudness)
    rms = float(np.sqrt(np.mean(samples**2)))

    # Zero-crossing rate (temporal noisiness cue)
    signs = np.sign(samples)
    zcr = float(np.sum(np.abs(np.diff(signs)) > 0)) / n

    # FFT for spectral features (window and freqs are cached)
    window, freqs = _get_window_and_freqs(n, sample_rate)
    windowed = samples * window
    spectrum = np.abs(np.fft.rfft(windowed))

    total_energy = float(np.sum(spectrum))
    if total_energy < 1e-10:
        return {
            "rms": rms,
            "spectral_centroid": 0.0,
            "zcr": zcr,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.0,
            "flatness": 0.0,
        }

    # Spectral centroid (brightness) — weighted mean frequency
    centroid = float(np.sum(freqs * spectrum) / total_energy)
    # Normalize to 0-1 (8000 Hz is a reasonable max for 16kHz audio)
    centroid_norm = min(centroid / 8000.0, 1.0)

    # Spectral rolloff — frequency below which 85% of energy lies
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_energy))
    rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    rolloff_norm = min(rolloff_freq / 8000.0, 1.0)

    # Spectral flatness — Wiener entropy (1 = white noise, 0 = pure tone)
    log_spectrum = np.log(spectrum + 1e-10)
    geometric_mean = math.exp(float(np.mean(log_spectrum)))
    arithmetic_mean = float(np.mean(spectrum))
    flatness = geometric_mean / (arithmetic_mean + 1e-10) if arithmetic_mean > 1e-10 else 0.0

    return {
        "rms": rms,
        "spectral_centroid": centroid_norm,
        "zcr": zcr,
        "spectral_flux": 0.0,  # computed across blocks, not within
        "spectral_rolloff": rolloff_norm,
        "flatness": float(flatness),
    }


# ─── Sound classifier ─────────────────────────────────────────────────


# Energy floor — below this is silence
_SILENCE_RMS = 0.008

# Minimum duration for a sound event (seconds)
_MIN_EVENT_DURATION = 0.15

# Maximum duration before a sound is considered "sustained" (seconds)
_SUSTAINED_THRESHOLD = 1.5


def _classify_sound(
    features: dict[str, float],
    duration: float,
    prev_features: dict[str, float] | None,
) -> tuple[str, float]:
    """Classify a sound event from its features.

    Returns (sound_type, confidence). This is a rule-based classifier
    using spectral features — lightweight enough for real-time on the
    A8-6410.

    The classification is intentionally conservative: when features
    are ambiguous, it defaults to "noise" (the most generic non-speech
    sound) rather than making a confident wrong call.
    """
    rms = features["rms"]
    centroid = features["spectral_centroid"]
    zcr = features["zcr"]
    flatness = features["flatness"]
    rolloff = features["spectral_rolloff"]

    # Spectral flux — how much the spectrum changed from the previous block.
    # When we have the pre-event block, this is the onset flux (the
    # silence→loud jump that marks an impact). Without a prior block
    # (e.g. the very first block of the stream), fall back to the
    # inter-block flux aggregated across the event.
    if prev_features is not None:
        flux = abs(rms - prev_features["rms"]) + abs(
            centroid - prev_features["spectral_centroid"]
        ) * 0.5
    else:
        flux = features["spectral_flux"]

    # Silence
    if rms < _SILENCE_RMS:
        return "silence", 0.95

    # Impact: sudden onset (high flux), short duration, high energy.
    # High rolloff (broadband) increases confidence — sharp sounds
    # like door slams and claps have energy across the spectrum.
    if flux > 0.15 and duration < _SUSTAINED_THRESHOLD and rms > 0.03:
        confidence = min(0.5 + flux * 2.0 + rolloff * 0.3, 0.9)
        return "impact", float(confidence)

    # Music: tonal (low flatness, low ZCR), sustained, moderate brightness
    if (
        duration >= _SUSTAINED_THRESHOLD
        and flatness < 0.3
        and zcr < 0.15
        and 0.1 < centroid < 0.7
        and rms > 0.01
    ):
        confidence = 0.6 + (1.0 - flatness) * 0.2
        return "music", float(min(confidence, 0.85))

    # Nature: moderate energy, broadband (high flatness), irregular
    if (
        flatness > 0.4
        and rms > 0.01
        and rms < 0.08
        and zcr > 0.05
    ):
        confidence = 0.55 + flatness * 0.2
        return "nature", float(min(confidence, 0.75))

    # Noise: sustained, high ZCR or high flatness, not tonal
    if duration >= _SUSTAINED_THRESHOLD and (zcr > 0.15 or flatness > 0.5):
        confidence = 0.6
        return "noise", float(confidence)

    # Short non-impact sound — ambiguous, call it noise
    if duration < _SUSTAINED_THRESHOLD:
        return "noise", 0.4

    # Default: sustained noise
    return "noise", 0.45


def _describe_sound(event: SoundEvent) -> str:
    """Build a structural description of a sound event.

    This is NOT a sentence template — it's a compact data string
    that its language engine can compose from. The format is:
    "type | loudness | brightness | noisiness | duration"
    """
    loudness_label = (
        "very loud" if event.loudness > 0.6
        else "loud" if event.loudness > 0.3
        else "moderate" if event.loudness > 0.1
        else "quiet"
    )
    brightness_label = (
        "bright" if event.brightness > 0.6
        else "balanced" if event.brightness > 0.3
        else "deep"
    )
    texture_label = (
        "noisy" if event.noisiness > 0.15
        else "tonal" if event.noisiness < 0.05
        else "mixed"
    )
    duration_label = (
        "sustained" if event.duration > _SUSTAINED_THRESHOLD
        else "brief"
    )
    return (
        f"{event.sound_type} | {loudness_label} | {brightness_label} | "
        f"{texture_label} | {duration_label}"
    )


# ─── Auditory cortex ──────────────────────────────────────────────────


class AuditoryCortex:
    """Continuously listens to and classifies non-speech sounds.

    Runs a background thread that captures audio, computes spectral
    features in real-time, segments sound events, classifies them,
    and calls a callback for each event.

    Args:
        callback: Called as ``callback(event: SoundEvent)`` for each
            detected sound event. Silence events are emitted at a
            reduced rate (once per quiet period, not every block).
        sample_rate: Audio sample rate. 16000 matches the ambient
            listener.
        block_size: Audio block size for feature extraction. 1024
            samples ≈ 64ms at 16kHz.
        is_speaking: Optional callback returning True if Genesis is
            currently speaking (TTS). When it's speaking, audio
            capture is paused to prevent self-listening.
    """

    def __init__(
        self,
        callback: Callable[[SoundEvent], None],
        sample_rate: int = 16000,
        block_size: int = 1024,
        is_speaking: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the auditory cortex."""
        self.callback = callback
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._is_speaking = is_speaking

        self._running = False
        self._thread: threading.Thread | None = None

        # Recent sound events (for debugging / inspection)
        self._recent: deque[SoundEvent] = deque(maxlen=50)
        self._recent_lock = threading.Lock()

        # Event segmentation state — aggregated incrementally (sums,
        # peak, consecutive-block flux) rather than storing every
        # block's feature dict, so a sustained sound (a fan running
        # for hours) doesn't grow memory without bound.
        self._in_event = False
        self._event_start = 0.0
        self._event_count = 0
        self._event_sums: dict[str, float] = {}
        self._event_peak_rms = 0.0
        self._event_flux_sum = 0.0
        self._event_flux_pairs = 0
        self._event_last_features: dict[str, float] | None = None
        # Features of the block immediately preceding the event — used
        # to compute onset flux (silence→loud jump) for impact detection.
        self._event_prev_features: dict[str, float] | None = None
        self._silence_emitted = False

    @property
    def running(self) -> bool:
        """Whether the auditory cortex thread is active."""
        return self._running

    def start(self) -> bool:
        """Start the auditory cortex. Returns True on success."""
        if self._running:
            return True
        try:
            import sounddevice

            _ = sounddevice.__name__  # verify import
        except ImportError:
            logger.warning("Auditory cortex: sounddevice not available")
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="auditory-cortex"
        )
        self._thread.start()
        logger.info("Auditory cortex started — listening to sounds")
        return True

    def stop(self) -> None:
        """Stop the auditory cortex."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Auditory cortex stopped")

    def recent_events(self) -> list[SoundEvent]:
        """Return recent sound events (for debugging)."""
        with self._recent_lock:
            return list(self._recent)

    def _listen_loop(self) -> None:
        """Main capture and classification loop.

        Captures audio blocks, computes features, segments events
        (grouping consecutive non-silence blocks), classifies each
        event, and emits it via the callback.
        """
        import queue

        import sounddevice as sd

        # Bound the queue so a starved consumer can't accumulate unbounded
        # audio buffers (~625 KB/s at 16 kHz / 1024-sample blocks). The
        # PortAudio callback must never block — a dropped block is better
        # than a delayed callback (which causes audio dropouts and breaks
        # real-time segmentation). ~50 blocks ≈ 3.2 s of headroom for
        # normal jitter.
        audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)

        def audio_callback(
            indata: bytes, frames: int, time_info: object, status: object
        ) -> None:
            """Capture audio block and queue it for processing."""
            if status:
                logger.debug(f"[auditory] PortAudio status: {status}")
            if self._is_speaking is not None and self._is_speaking():
                return
            # Convert raw buffer to numpy float32 array
            samples = np.frombuffer(indata, dtype=np.float32)
            try:
                audio_queue.put_nowait(samples.copy())
            except queue.Full:
                # Consumer is starved — drop this block rather than
                # blocking the real-time audio thread.
                logger.debug("[auditory] audio queue full — dropping block")

        prev_features: dict[str, float] | None = None

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                channels=1,
                callback=audio_callback,
            ):
                while self._running:
                    try:
                        samples = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        # No audio queued. If Genesis is speaking, the
                        # capture callback is paused — but an in-progress
                        # sound event would otherwise stay open across the
                        # entire speaking period, inflating its duration
                        # and aggregating features across a discontinuous
                        # gap. Close it out now so it's classified on its
                        # own merits, not contaminated by its speech.
                        if self._in_event and self._is_speaking is not None and self._is_speaking():
                            self._finalize_event()
                        continue

                    if len(samples) == 0:
                        continue

                    now = time.time()
                    features = _compute_features(samples, self.sample_rate)
                    is_silent = features["rms"] < _SILENCE_RMS

                    if is_silent:
                        # End of a sound event
                        if self._in_event:
                            self._finalize_event()
                        # Emit silence once per quiet period
                        if not self._silence_emitted:
                            self._emit_silence(now)
                            self._silence_emitted = True
                    else:
                        # Start or continue a sound event
                        if not self._in_event:
                            self._in_event = True
                            self._event_start = now
                            self._event_count = 0
                            self._event_sums = {}
                            self._event_peak_rms = 0.0
                            self._event_flux_sum = 0.0
                            self._event_flux_pairs = 0
                            self._event_last_features = None
                            # Capture the block immediately before the
                            # event so the classifier can measure onset
                            # flux (the silence→loud jump that marks an
                            # impact). prev_features at this point is the
                            # last block before the event started.
                            self._event_prev_features = prev_features
                        self._accumulate_event_features(features)
                        self._silence_emitted = False

                    prev_features = features

        except Exception as e:
            logger.exception(f"Auditory cortex crashed: {e}")
            self._running = False

    def _accumulate_event_features(self, features: dict[str, float]) -> None:
        """Fold one block's features into the running event aggregates."""
        self._event_count += 1
        for key, value in features.items():
            self._event_sums[key] = self._event_sums.get(key, 0.0) + value
        self._event_peak_rms = max(self._event_peak_rms, features["rms"])
        if self._event_last_features is not None:
            self._event_flux_sum += (
                abs(features["rms"] - self._event_last_features["rms"])
                + abs(
                    features["spectral_centroid"]
                    - self._event_last_features["spectral_centroid"]
                )
                * 0.5
            )
            self._event_flux_pairs += 1
        self._event_last_features = features

    def _finalize_event(self) -> None:
        """Classify and emit a completed sound event."""
        if self._event_count == 0:
            self._in_event = False
            return

        duration = time.time() - self._event_start
        if duration < _MIN_EVENT_DURATION:
            self._in_event = False
            self._event_count = 0
            self._event_sums = {}
            self._event_prev_features = None
            self._event_last_features = None
            return

        # Aggregate features across the event
        count = self._event_count
        avg_features = {
            key: total / count for key, total in self._event_sums.items()
        }
        # Peak RMS for loudness
        avg_features["rms"] = self._event_peak_rms

        # Spectral flux: average change between consecutive blocks
        if self._event_flux_pairs:
            avg_features["spectral_flux"] = (
                self._event_flux_sum / self._event_flux_pairs
            )
        else:
            avg_features["spectral_flux"] = 0.0

        sound_type, confidence = _classify_sound(
            avg_features, duration, self._event_prev_features
        )

        # Normalize loudness to 0-1 (0.5 RMS is very loud)
        loudness = min(self._event_peak_rms / 0.5, 1.0)
        brightness = avg_features["spectral_centroid"]
        # Noisiness tracks spectral flatness (Wiener entropy): 0 = pure
        # tone, 1 = white noise. ZCR alone can't separate tonal from
        # noisy (a high-frequency pure tone has high ZCR too), so flatness
        # is the correct perceptual measure here.
        noisiness = avg_features["flatness"]

        event = SoundEvent(
            sound_type=sound_type,
            confidence=confidence,
            loudness=loudness,
            brightness=brightness,
            noisiness=noisiness,
            duration=duration,
            timestamp=self._event_start,
        )
        event.description = _describe_sound(event)

        with self._recent_lock:
            self._recent.append(event)

        try:
            self.callback(event)
        except Exception as e:
            logger.exception(f"Auditory callback raised: {e}")

        # Reset event state
        self._in_event = False
        self._event_count = 0
        self._event_sums = {}
        self._event_prev_features = None
        self._event_last_features = None

    def _emit_silence(self, timestamp: float) -> None:
        """Emit a silence event."""
        event = SoundEvent(
            sound_type="silence",
            confidence=0.95,
            loudness=0.0,
            brightness=0.0,
            noisiness=0.0,
            duration=0.0,
            timestamp=timestamp,
        )
        event.description = "silence | quiet | deep | tonal | brief"
        with self._recent_lock:
            self._recent.append(event)
        try:
            self.callback(event)
        except Exception:  # noqa: BLE001
            logger.debug("Auditory silence callback failed")

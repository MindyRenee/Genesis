"""Ambient listener — Genesis's ears.

Continuously captures audio from the microphone, transcribes it with
Vosk (offline), and emits complete utterances. The caller decides
whether to act on each utterance.

This is a streaming, always-on listener — unlike ``VoiceInput.listen``
which blocks for a single utterance, ``AmbientListener`` runs in a
background thread and calls a callback for every utterance detected
in the room.

# Wake word

The listener does **not** decide whether to respond — it just
transcribes. Wake-word detection is handled by the caller via
``contains_wake_word()``, which does fuzzy matching against "genesis"
and common Vosk mis-transcriptions.

# Resource use

Vosk small model: ~50 MB RAM, low CPU. The audio capture uses
sounddevice's callback API (no busy-waiting). On the A8-6410 this
adds negligible load to the cognitive mind's 1 GB / 2-core budget.

# Privacy

This module listens to all speech in microphone range. It logs
transcribed utterances at INFO level. The caller is responsible for
deciding what to do with them.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

__all__ = ["AmbientListener", "contains_wake_word", "strip_wake_word"]

logger = logging.getLogger(__name__)

# ─── Energy gate ──────────────────────────────────────────────────────
#
# RMS threshold below which an audio block is considered silence.
# int16 samples range from -32768 to 32768; typical ambient noise
# from a laptop mic sits around 100-300 RMS. Speech is typically
# 1000+. We set the threshold at 500 to be conservative — we'd
# rather feed a few noise blocks to Vosk than miss quiet speech.
_SILENCE_RMS_THRESHOLD: float = 500.0

# Minimum average word confidence for a Vosk transcription to be
# emitted. Vosk's small model can hallucinate transcriptions from
# background noise — these typically have word confidence below 0.3.
# Real speech has confidence above 0.5. We set the threshold at 0.4
# to filter hallucinations while keeping quieter speech.
_MIN_TRANSCRIPTION_CONFIDENCE: float = 0.4

# ─── Wake word matching ────────────────────────────────────────────────

# Vosk small-en-us model commonly mis-transcribes "genesis" as:
#   "jen isis", "jenesis", "gina says", "jena says", "genesis",
#   "gen is this", "jen is this", "jenn says", "gennis is"
# We match on the raw string, case-insensitive, looking for any of
# these patterns as whole-word matches.
_WAKE_PATTERNS = [
    r"\bgenesis\b",
    r"\bjenesis\b",
    r"\bjen\s*isis\b",
    r"\bjen\s*is\s*is\b",
    r"\bgen\s*is\s*is\b",
    r"\bgen\s*is\s*this\b",
    r"\bjen\s*is\s*this\b",
    r"\bgina\s*says\b",
    r"\bjena\s*says\b",
    r"\bjenn\s*says\b",
    r"\bgennis\b",
    r"\bjen\s*says\b",
]
_WAKE_RE = re.compile("|".join(_WAKE_PATTERNS), re.IGNORECASE)


def contains_wake_word(text: str) -> bool:
    """True if the text contains a wake word for Genesis.

    Does fuzzy matching against "genesis" and common Vosk
    mis-transcriptions.
    """
    return bool(_WAKE_RE.search(text))


def strip_wake_word(text: str) -> str:
    """Remove the wake word from the text, returning the query.

    "genesis, how are you?" → "how are you?"
    "hey genesis what's up" → "hey what's up"
    """
    # Remove the wake word and any leading punctuation/whitespace
    stripped = _WAKE_RE.sub("", text, count=1)
    # Clean up leading commas, whitespace
    stripped = stripped.lstrip(" ,.!?;:\n\t")
    return stripped.strip()


# ─── Ambient listener ──────────────────────────────────────────────────


class AmbientListener:
    """Continuously listens to the microphone and emits utterances.

    Runs a background thread that captures audio, runs it through
    Vosk, and calls the callback for each complete utterance.

    Args:
        callback: Called as ``callback(text: str)`` for each detected
            utterance. The text is the final Vosk transcription,
            stripped and lowercased. Empty/garbage utterances are
            filtered out.
        sample_rate: Audio sample rate. Vosk expects 16000.
        silence_threshold: Number of silence chunks (each ~50ms) after
            speech to consider the utterance complete. 30 ≈ 1.5s.
        min_utterance_len: Minimum word count to emit (filters out
            "uh", "hmm", false triggers).
    """

    def __init__(
        self,
        callback: Callable[[str], None],
        sample_rate: int = 16000,
        silence_threshold: int = 30,
        min_utterance_len: int = 2,
        is_speaking: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the ambient listener with the given callback and parameters."""
        self.callback = callback
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_utterance_len = min_utterance_len
        self._is_speaking = is_speaking

        self._running = False
        self._thread: threading.Thread | None = None
        self._vosk_model = None
        self._initialized = False
        self._init_error: str | None = None

        # Recent utterance log (for debugging / the caller to inspect)
        self._recent: deque[str] = deque(maxlen=20)
        self._recent_lock = threading.Lock()

    def _init_vosk(self) -> bool:
        """Load the Vosk model. Returns True on success."""
        from pathlib import Path

        model_dir = Path.home() / ".local" / "share" / "vosk-models" / "vosk-model-small-en-us-0.15"
        if not model_dir.exists():
            self._init_error = f"Vosk model not found at {model_dir}"
            return False
        try:
            from vosk import Model

            self._vosk_model = Model(str(model_dir))
            self._initialized = True
            return True
        except Exception as e:  # noqa: BLE001
            self._init_error = repr(e)
            return False

    @property
    def available(self) -> bool:
        """True if the listener can start (Vosk model loaded)."""
        if not self._initialized:
            return self._init_vosk()
        return self._vosk_model is not None

    @property
    def init_error(self) -> str | None:
        """Error message from the last Vosk initialization attempt, or None if no error."""
        return self._init_error

    def start(self) -> bool:
        """Start the ambient listener thread. Returns True on success."""
        if not self.available:
            logger.error(f"Ambient listener cannot start: {self._init_error}")
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="ambient-listener"
        )
        self._thread.start()
        logger.info("Ambient listener started — listening to the room")
        return True

    def stop(self) -> None:
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("Ambient listener stopped")

    @property
    def running(self) -> bool:
        """Whether the listener thread is currently active."""
        return self._running

    def recent_utterances(self) -> list[str]:
        """Return recent transcribed utterances (for debugging)."""
        with self._recent_lock:
            return list(self._recent)

    def _listen_loop(self) -> None:
        """Main capture loop — runs until stopped.

        Uses a single KaldiRecognizer for the lifetime of the listener.
        Vosk's AcceptWaveform returns 1 when it has a final result
        (end of utterance detected by its internal VAD). We collect
        final results in a thread-safe queue and process them in the
        main loop. This avoids any race condition with recognizer
        reassignment.

        We also track partial results to detect speech activity, and
        use our own silence counter as a secondary utterance boundary
        (Vosk's VAD is conservative; our counter catches longer pauses
        that Vosk hasn't finalized yet).
        """
        import queue

        import sounddevice as sd
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        recognizer.SetWords(True)

        # Thread-safe queue for final results from the audio callback
        final_queue: queue.Queue[str] = queue.Queue()

        # Speech activity tracking (shared between callback and main loop)
        speech_state = {"active": False, "last_partial": ""}

        # Silence block counter for the energy gate (see audio_callback).
        # Nonlocal inside the callback; reset to 0 on speech.
        _silence_block_count = 0

        import numpy as _np

        def audio_callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
            """Process incoming audio: feed to Vosk and queue final results.

            Energy-gates the audio: Vosk's AcceptWaveform runs full
            neural network inference on every block, which is the
            dominant CPU cost of the ambient listener (~20% of one
            core). By computing a cheap RMS energy check first and
            only feeding blocks above the silence threshold to Vosk,
            we skip the expensive inference during silence (which is
            the vast majority of the time for an always-on listener).
            """
            nonlocal _silence_block_count
            data = indata.tobytes() if hasattr(indata, "tobytes") else bytes(indata)

            # Mute the mic while she is speaking to prevent her from
            # hearing her own TTS output and looping back.
            if self._is_speaking is not None and self._is_speaking():
                return

            # Energy gate: compute RMS on the int16 samples and skip
            # Vosk processing for silence. This avoids running the
            # neural network on ~99% of blocks (silence), cutting
            # the ambient listener's CPU usage dramatically.
            samples = _np.frombuffer(data, dtype=_np.int16).astype(_np.float32)
            rms = _np.sqrt(_np.mean(samples ** 2)) if len(samples) > 0 else 0.0
            if rms < _SILENCE_RMS_THRESHOLD:
                # Even in silence, we must call AcceptWaveform
                # periodically so Vosk's internal VAD can finalize
                # any pending utterance. Feed a zero block every ~1s
                # (every 16th block at 64ms/block) to flush.
                _silence_block_count += 1
                if _silence_block_count % 16 != 0:
                    return
                data = b"\x00" * len(data)
            else:
                _silence_block_count = 0

            # AcceptWaveform returns 1 when a final result is ready
            if recognizer.AcceptWaveform(data):
                self._handle_final_result(recognizer, final_queue)
            # Track partial results for speech activity
            self._track_partial(recognizer, speech_state)

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=1024,
                dtype="int16",
                channels=1,
                callback=audio_callback,
            ):
                while self._running:
                    # Check for final results from Vosk's VAD
                    try:
                        text = final_queue.get(timeout=0.1)
                        self._emit(text)
                        speech_state["active"] = False
                    except queue.Empty as e:
                        logger.debug(f"no speech detected this interval: {e}")

        except Exception as e:
            logger.exception(f"Ambient listener crashed: {e}")
            self._running = False

    def _handle_final_result(self, recognizer: Any, final_queue: Any) -> None:
        """Parse a final Vosk result and queue accepted text.

        Vosk's result may include a "result" array with per-word
        confidence scores. Hallucinated transcriptions from noise have
        very low confidence (typically < 0.3). Filter them out to
        prevent responding to phantom speech.
        """
        try:
            final = json.loads(recognizer.Result())
            text = final.get("text", "").strip()
            if not text:
                return
            word_results = final.get("result", [])
            accept = True
            if word_results:
                confs = [
                    w.get("conf", 1.0)
                    for w in word_results
                    if isinstance(w, dict)
                ]
                avg_conf = sum(confs) / len(confs) if confs else 0.0
                if avg_conf < _MIN_TRANSCRIPTION_CONFIDENCE:
                    logger.debug(
                        f"Ambient: low-confidence ({avg_conf:.2f}) "
                        f"transcription filtered: '{text}'"
                    )
                    accept = False
            if accept:
                final_queue.put(text)
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"vosk final result parse failed: {e}")

    def _track_partial(self, recognizer: Any, speech_state: dict) -> None:
        """Track partial results for speech activity."""
        try:
            partial = json.loads(recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                speech_state["active"] = True
                speech_state["last_partial"] = partial_text
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"vosk partial result parse failed: {e}")

    def _emit(self, text: str) -> None:
        """Emit a transcribed utterance to the callback."""
        # Filter out very short utterances (false triggers, coughs, etc.)
        words = text.split()
        if len(words) < self.min_utterance_len:
            logger.debug(f"Ambient: filtered short utterance: '{text}'")
            return

        # Log it
        logger.info(f"Ambient heard: \"{text}\"")
        with self._recent_lock:
            self._recent.append(text)

        # Deliver to callback
        try:
            self.callback(text)
        except Exception as e:
            logger.exception(f"Ambient callback raised: {e}")

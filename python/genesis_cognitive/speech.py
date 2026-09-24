"""Voice for Genesis — text-to-speech (Piper) and speech-to-text (Vosk/Google).

This module provides two classes:

  ``Voice`` — text-to-speech (TTS). Uses Piper neural TTS when
  available (offline, high quality), falling back to espeak-ng.

  ``VoiceInput`` — speech-to-text (STT). Uses Vosk (offline) when
  a model is installed, falling back to Google Web Speech API.

Usage::

    from genesis_cognitive.speech import Voice, VoiceInput

    voice = Voice()                    # TTS
    voice.speak("Hello there")         # speaks aloud

    mic = VoiceInput()                 # STT
    text = mic.listen()                # blocks until you speak
"""

from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────
_VOICES_DIR = Path(__file__).parent.parent / "voices"
_PIPER_BIN = _VOICES_DIR / "piper"
_PIPER_MODEL = _VOICES_DIR / "amy-medium.onnx"
_PIPER_MODEL_FALLBACK = _VOICES_DIR / "amy-low.onnx"

VOSK_MODEL_DIR = Path.home() / ".local" / "share" / "vosk-models"
VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"

# Maximum character length per TTS chunk. Piper synthesizes audio
# in real-time — on slow CPUs, long text can exceed the 30-second
# reap timeout. ~150 chars keeps each invocation well under timeout.
_MAX_TTS_CHUNK_CHARS = 150


def _chunk_text_for_tts(text: str) -> list[str]:
    """Split text into sentence-sized chunks for TTS.

    Piper generates audio in real-time. On a slow CPU, a long
    utterance can take longer than the 30-second reap timeout,
    causing "TTS child timed out, killing" and no audio. Splitting
    into sentence-sized chunks keeps each Piper invocation short.

    Splits on sentence boundaries (. ! ?) first, then on commas
    for very long sentences. Returns the original text as a
    single-element list if it's already short enough.
    """
    text = text.strip()
    if len(text) <= _MAX_TTS_CHUNK_CHARS:
        return [text] if text else []

    import re

    # Split on sentence boundaries, keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # If a single sentence is too long, split on commas
        if len(sentence) > _MAX_TTS_CHUNK_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            comma_parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in comma_parts:
                if len(current) + len(part) + 1 <= _MAX_TTS_CHUNK_CHARS:
                    current = (current + " " + part).strip() if current else part
                else:
                    if current:
                        chunks.append(current.strip())
                    current = part
            continue

        if len(current) + len(sentence) + 1 <= _MAX_TTS_CHUNK_CHARS:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


# ═══════════════════════════════════════════════════════════════════
#  Text-to-Speech (Piper)
# ═══════════════════════════════════════════════════════════════════


class Voice:
    """Text-to-speech using Piper neural TTS.

    Piper runs entirely offline using ONNX models. If Piper or a
    voice model isn't available, falls back to espeak-ng.
    """

    def __init__(self) -> None:
        """Detect available TTS backends and load the Piper voice model."""
        self._piper_available = self._check_piper()
        self._espeak_available = self._check_espeak()
        self._model = (
            _PIPER_MODEL
            if _PIPER_MODEL.exists()
            else _PIPER_MODEL_FALLBACK
            if _PIPER_MODEL_FALLBACK.exists()
            else None
        )
        self._sample_rate = 22050
        if self._model is not None:
            config_path = self._model.with_name(self._model.name + ".json")
            try:
                with config_path.open() as f:
                    self._sample_rate = json.load(f)["audio"]["sample_rate"]
            except (OSError, KeyError, json.JSONDecodeError) as e:
                logger.debug(f'__init__ failed: {e}')
        self._piper_proc: subprocess.Popen | None = None
        self._play_proc: subprocess.Popen | None = None
        self._all_procs: list[subprocess.Popen] = []  # track all spawned procs
        # LOCK ORDER (deadlock prevention): _procs_lock, _speaking_lock,
        # and _queue_lock are LEAF-LEVEL — never hold one while acquiring
        # another. If nesting is ever required, acquire in declaration
        # order (procs > speaking > queue).
        self._procs_lock = threading.Lock()
        self._speaking_count = 0
        self._speaking_lock = threading.Lock()
        self._last_spoke_time = 0.0
        self._speech_tail = 0.3  # keep mic muted for 300ms after TTS ends
        # Speech queue: when speech is in progress, new utterances are
        # queued instead of cutting off the current speech. This prevents
        # her words from being truncated mid-sentence when multiple
        # thoughts or responses arrive in quick succession.
        self._speech_queue: collections.deque[
            tuple[str, float, float, float, float]
        ] = collections.deque(maxlen=3)
        self._queue_lock = threading.Lock()

    def _check_piper(self) -> bool:
        """Check if Piper binary exists and is executable."""
        return _PIPER_BIN.exists() and os.access(_PIPER_BIN, os.X_OK)

    def _check_espeak(self) -> bool:
        """Check if espeak-ng is available."""
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=2,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def is_available(self) -> bool:
        """True if any TTS backend is available."""
        return self._piper_available or self._espeak_available

    def describe(self) -> str:
        """Human-readable description of the current TTS setup."""
        if self._piper_available and self._model:
            return f"Voice: Piper neural TTS ({self._model.name})"
        elif self._espeak_available:
            return "Voice: espeak-ng (fallback)"
        return "Voice: unavailable"

    def speak(
        self,
        text: str,
        blocking: bool = False,
        alertness: float = 0.5,
        valence: float = 0.0,
        caution: float = 0.0,
        creativity: float = 0.0,
    ) -> None:
        """Speak text aloud.

        Args:
            text: The text to speak.
            blocking: If True, wait for speech to finish before
                returning. If False, return immediately while
                speech plays in the background.
            alertness: 0-1, affects speech rate (higher = faster).
            valence: -1 to 1, affects pitch (positive = higher).
            caution: 0-1, slows speech slightly (careful delivery).
            creativity: 0-1, adds slight pitch variation.
        """
        if not text.strip():
            return

        # Chunk long text to prevent TTS timeouts on slow CPUs.
        # Piper generates audio in real-time — on the A8-6410, a
        # 200-character utterance can take 20+ seconds to synthesize.
        # The 30-second reap timeout in _reap_children kills the
        # process if it takes too long, producing "TTS child timed
        # out, killing" and no audio. Splitting into sentence-sized
        # chunks (max ~150 chars each) keeps each Piper invocation
        # well under the timeout.
        chunks = _chunk_text_for_tts(text)
        if len(chunks) <= 1:
            self._speak_single(
                text, blocking, alertness, valence, caution, creativity
            )
            return
        # Speak each chunk sequentially. Non-blocking mode queues
        # the remaining chunks so they play in order without cutting
        # off the current speech.
        for i, chunk in enumerate(chunks):
            if i == 0:
                self._speak_single(
                    chunk, blocking, alertness, valence, caution, creativity
                )
            else:
                with self._queue_lock:
                    max_q = self._speech_queue.maxlen or 3
                    if len(self._speech_queue) < max_q:
                        self._speech_queue.append(
                            (chunk, alertness, valence, caution, creativity)
                        )

    def _speak_single(
        self,
        text: str,
        blocking: bool,
        alertness: float,
        valence: float,
        caution: float,
        creativity: float,
    ) -> None:
        """Speak a single chunk of text (internal helper)."""

        # If already speaking, queue the new utterance instead of
        # cutting off the current speech mid-sentence. This prevents
        # her words from being truncated when multiple thoughts or
        # responses arrive in quick succession. Blocking speech
        # always interrupts — the caller is waiting for it.
        if not blocking and self.is_speaking:
            with self._queue_lock:
                max_q = self._speech_queue.maxlen or 3
                if len(self._speech_queue) < max_q:
                    self._speech_queue.append(
                        (text, alertness, valence, caution, creativity)
                    )
            return

        self.stop()
        self._start_speaking()

        # Adjust speech rate based on emotional state
        # Base 1.25 gives a conversational pace a touch slower than the
        # previous 1.45. Higher alertness → faster speech; higher caution
        # → slower speech.
        rate_factor = 1.25 + (alertness - 0.5) * 0.4 - caution * 0.15
        rate_factor = max(0.9, min(1.8, rate_factor))

        try:
            if self._piper_available and self._model:
                self._speak_piper(text, blocking, rate_factor)
            elif self._espeak_available:
                self._speak_espeak(text, blocking, rate_factor)
        finally:
            if blocking:
                self._done_speaking()

    def _reap_children(self, procs: tuple[subprocess.Popen, ...]) -> None:
        """Reap finished TTS child processes so they don't become zombies.

        Includes a timeout so a hung Piper/aplay process (e.g. audio
        device busy, pipe full) doesn't block forever and leak. If the
        timeout expires, the process is killed.
        """
        try:
            for proc in procs:
                if proc is None:
                    continue
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("TTS child timed out, killing")
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"failed to kill TTS child: {e}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"reap child failed: {e}")
        finally:
            self._done_speaking()
            self._play_next_queued()

    def _play_next_queued(self) -> None:
        """Play the next queued utterance if any.

        Called after the current speech finishes (in _reap_children).
        Dequeues one utterance and plays it non-blocking. This is the
        drain side of the speech queue — speak() enqueues when busy,
        and this method dequeues when idle.
        """
        with self._queue_lock:
            if not self._speech_queue:
                return
            text, alertness, _valence, caution, _creativity = (
                self._speech_queue.popleft()
            )
        # Play the queued utterance. We call the internal playback
        # directly (not speak()) to avoid re-checking is_speaking,
        # which would re-queue and never play.
        self._start_speaking()
        rate_factor = 1.25 + (alertness - 0.5) * 0.4 - caution * 0.15
        rate_factor = max(0.9, min(1.8, rate_factor))
        try:
            if self._piper_available and self._model:
                self._speak_piper(text, False, rate_factor)
            elif self._espeak_available:
                self._speak_espeak(text, False, rate_factor)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"queued speak failed: {e}")
            self._done_speaking()

    def _start_speaking(self) -> None:
        """Increment the speaking counter under lock to mark speech as active."""
        with self._speaking_lock:
            self._speaking_count += 1

    def _done_speaking(self) -> None:
        """Decrement the speaking counter, recording the time when it reaches zero."""
        with self._speaking_lock:
            self._speaking_count = max(0, self._speaking_count - 1)
            if self._speaking_count == 0:
                self._last_spoke_time = time.time()

    @property
    def is_speaking(self) -> bool:
        """True while any TTS child is still speaking, plus a short tail."""
        with self._speaking_lock:
            if self._speaking_count > 0:
                return True
        return time.time() - self._last_spoke_time < self._speech_tail

    def _speak_piper(self, text: str, blocking: bool, rate_factor: float = 1.0) -> None:
        """Speak using Piper neural TTS."""
        try:
            # Piper reads text from stdin and writes raw audio to stdout.
            # We pipe that to aplay for playback.
            # Piper supports --length-scale to adjust speech rate
            # (lower = faster, higher = slower; default ~1.0)
            length_scale = 1.0 / rate_factor
            cmd = [
                str(_PIPER_BIN),
                "--model",
                str(self._model),
                "--output-raw",
                "--length-scale",
                str(length_scale),
            ]
            # Set LD_LIBRARY_PATH so piper finds its shared libs
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(_VOICES_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            # Send text to Piper
            assert proc.stdin is not None
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()

            # Play the raw audio.
            #
            # Buffer configuration: Piper generates audio in real-time
            # and pipes it to aplay. With the PipeWire ALSA plugin
            # (this system's audio backend), the default buffer sizes
            # are too small — scheduling jitter causes buffer underruns
            # that produce crackling/popping in the output.
            #
            # --buffer-time=200000 (200ms): the ALSA ring buffer holds
            #   200ms of audio. This gives PipeWire enough headroom to
            #   absorb scheduling delays without underrunning.
            # --period-time=50000 (50ms): PipeWire wakes every 50ms to
            #   pull data. Smaller periods = lower latency but more CPU;
            #   50ms is a good balance for TTS (latency is irrelevant
            #   for one-way speech output).
            #
            # The crackling was specifically caused by the default
            # ~50ms buffer with the PipeWire backend — any scheduling
            # hiccup would underrun it. 200ms is 4x that, stable.
            play_cmd = [
                "aplay",
                "-r", str(self._sample_rate),
                "-f", "S16_LE",
                "-c", "1",
                "-q",
                "--buffer-time=200000",
                "--period-time=50000",
            ]
            self._piper_proc = proc
            try:
                self._play_proc = subprocess.Popen(
                    play_cmd,
                    stdin=proc.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                # aplay failed to start — terminate the already-running
                # Piper process before falling back to espeak, otherwise
                # it becomes an orphan (not in _all_procs, never reaped).
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except (OSError, ValueError, subprocess.SubprocessError):
                    proc.kill()
                self._piper_proc = None
                raise
            with self._procs_lock:
                self._all_procs.extend([proc, self._play_proc])

            if blocking:
                self._play_proc.wait()
                proc.wait()
            else:
                threading.Thread(
                    target=self._reap_children,
                    args=((proc, self._play_proc),),
                    daemon=True,
                ).start()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Piper TTS failed: {e}")
            if self._espeak_available:
                self._speak_espeak(text, blocking, rate_factor)

    def _speak_espeak(self, text: str, blocking: bool, rate_factor: float = 1.0) -> None:
        """Speak using espeak-ng (fallback)."""
        try:
            # espeak rate: words per minute (default 175)
            rate = int(175 * rate_factor)
            cmd = ["espeak-ng", text, "-s", str(rate)]
            self._play_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._procs_lock:
                self._all_procs.append(self._play_proc)
            if blocking:
                self._play_proc.wait()
            else:
                threading.Thread(
                    target=self._reap_children,
                    args=((self._play_proc,),),
                    daemon=True,
                ).start()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"espeak-ng failed: {e}")

    def stop(self) -> None:
        """Stop any currently playing speech."""
        # Clear any queued utterances — they're stale once we stop.
        with self._queue_lock:
            self._speech_queue.clear()
        # Kill the latest procs (fast path)
        for proc in (self._piper_proc, self._play_proc):
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception as e:  # noqa: BLE001
                    logger.debug(repr(e))
        # Also kill any orphaned procs from previous non-blocking calls
        # whose _reap_children thread is still blocked on wait().
        with self._procs_lock:
            orphans = [p for p in self._all_procs if p.poll() is None]
            self._all_procs.clear()
        for proc in orphans:
            try:
                proc.terminate()
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"failed to stop TTS proc: {e}")
        self._piper_proc = None
        self._play_proc = None


# ═══════════════════════════════════════════════════════════════════
#  Speech-to-Text (Vosk / Google)
# ═══════════════════════════════════════════════════════════════════


class VoiceInput:
    """Speech-to-text from the microphone.

    Automatically selects the best available backend:
    - Vosk if a model is installed (offline, no latency)
    - Google Web Speech API as fallback (online, high accuracy)

    Audio capture uses ``sounddevice`` which ships pre-built wheels.
    """

    def __init__(self, backend: str = "auto", sample_rate: int = 16000, lazy: bool = False) -> None:
        """Initialize speech-to-text with optional lazy model loading."""
        self.sample_rate = sample_rate
        self._backend: str | None = None
        self._vosk_model: Any = None
        self._sr_recognizer: Any = None
        self._lock = threading.Lock()

        if lazy:
            # Defer model loading until the user actually uses /voice.
            # This saves several seconds of startup time in normal typing mode.
            if backend in ("auto", "vosk") and self._can_use_vosk():
                self._backend = "vosk"
                return
            if backend in ("auto", "google") and self._can_use_google():
                self._backend = "google"
                return
        else:
            if backend in ("auto", "vosk"):
                if self._init_vosk():
                    self._backend = "vosk"
                    logger.info("Voice input: using Vosk (offline)")
                    return

            if backend in ("auto", "google"):
                if self._init_google():
                    self._backend = "google"
                    logger.info("Voice input: using Google Web Speech API (online)")
                    return

        logger.warning(
            "Voice input: no backend available. Install vosk model or check internet connection."
        )

    def _can_use_vosk(self) -> bool:
        """Check whether Vosk *could* be used, without loading the model."""
        model_path = VOSK_MODEL_DIR / VOSK_MODEL_NAME
        if not model_path.exists():
            return False
        import importlib.util

        return importlib.util.find_spec("vosk") is not None

    def _can_use_google(self) -> bool:
        """Check whether the Google SR module is available."""
        import importlib.util

        return importlib.util.find_spec("speech_recognition") is not None

    def _ensure_loaded(self) -> bool:
        """If lazy-loaded, actually initialize the backend now."""
        if self._backend == "vosk" and self._vosk_model is None:
            return self._init_vosk()
        if self._backend == "google" and self._sr_recognizer is None:
            return self._init_google()
        return self._backend is not None

    def _init_vosk(self) -> bool:
        """Try to load the Vosk model."""
        model_path = VOSK_MODEL_DIR / VOSK_MODEL_NAME
        if not model_path.exists():
            logger.debug(f"Vosk model not found at {model_path}")
            return False
        try:
            from vosk import Model

            self._vosk_model = Model(str(model_path))
            logger.debug("Vosk model loaded")
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Vosk init failed: {e}")
            return False

    def _init_google(self) -> bool:
        """Try to initialize Google Web Speech API."""
        try:
            import speech_recognition as sr

            self._sr_recognizer = sr.Recognizer()
            self._sr_recognizer.energy_threshold = 300
            self._sr_recognizer.dynamic_energy_threshold = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Google init failed: {e}")
            return False

    @property
    def available(self) -> bool:
        """True if a speech recognition backend is ready."""
        return self._backend is not None

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend or "none"

    def listen(self, timeout: float | None = None, phrase_limit: float = 10) -> str | None:
        """Listen for a single utterance and return the transcribed text.

        Args:
            timeout: How long to wait for speech to start (seconds).
                None = wait indefinitely.
            phrase_limit: Maximum recording time for a single phrase.

        Returns:
            Transcribed text, or None if no speech was detected.
        """
        if not self._backend:
            return None

        if not self._ensure_loaded():
            return None

        with self._lock:
            if self._backend == "vosk":
                return self._listen_vosk(timeout, phrase_limit)
            elif self._backend == "google":
                return self._listen_google(timeout, phrase_limit)
        return None

    def _listen_vosk(self, timeout: float | None, phrase_limit: float) -> str | None:
        """Listen using Vosk (offline)."""
        import sounddevice as sd
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        recognizer.SetWords(True)

        chunks: list[bytes] = []
        silence_count = 0
        speech_started = False
        import time as _time

        def callback(indata: Any, frames: int, time_info: object, status: object) -> None:
            """Append incoming audio bytes to the chunk list for later recognition."""
            chunks.append(indata.tobytes())

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=1024,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            start = _time.time()
            fed = 0
            while True:
                _time.sleep(0.05)
                elapsed = _time.time() - start

                # Feed newly captured audio to the recognizer here, on
                # this thread — KaldiRecognizer is not thread-safe, so
                # the audio callback must not call AcceptWaveform. The
                # callback only buffers; all recognizer calls stay in
                # this loop.
                while fed < len(chunks):
                    recognizer.AcceptWaveform(chunks[fed])
                    fed += 1

                # Check Vosk partial results for speech detection
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    speech_started = True
                    silence_count = 0
                elif speech_started:
                    silence_count += 1

                # Timeout: no speech detected
                if timeout and elapsed > timeout and not speech_started:
                    return None

                # Phrase limit
                if elapsed > phrase_limit:
                    break

                # Speech ended (1.5s of silence after speech)
                if speech_started and silence_count > 30:
                    break

        # Get final result — reprocess all chunks in a fresh recognizer
        final_recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        for chunk in chunks:
            final_recognizer.AcceptWaveform(chunk)
        final = json.loads(final_recognizer.FinalResult())
        text = final.get("text", "").strip()
        return text if text else None

    def _listen_google(self, timeout: float | None, phrase_limit: float) -> str | None:
        """Listen using Google Web Speech API (online).

        Captures audio with sounddevice, saves to a temp WAV,
        then transcribes with speech_recognition.
        """
        import numpy as np
        import sounddevice as sd
        import speech_recognition as sr

        audio_chunks: list[np.ndarray] = []
        silence_frames = 0
        speech_started = False
        chunk_duration = 0.1
        chunk_samples = int(self.sample_rate * chunk_duration)
        silence_threshold = 500
        max_silence = int(1.5 / chunk_duration)
        import time as _time

        def callback(indata: Any, frames: int, time_info: object, status: object) -> None:
            """Capture audio chunks and track speech/silence via RMS energy."""
            nonlocal silence_frames, speech_started
            audio_chunks.append(indata.copy())
            rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2))
            if rms > silence_threshold:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=chunk_samples,
            dtype="int16",
            channels=1,
            callback=callback,
        ):
            start = _time.time()
            while True:
                _time.sleep(0.05)
                elapsed = _time.time() - start
                if timeout and elapsed > timeout and not speech_started:
                    return None
                if elapsed > phrase_limit:
                    break
                if speech_started and silence_frames > max_silence:
                    break

        if not audio_chunks or not speech_started:
            return None

        audio = np.concatenate(audio_chunks, axis=0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sample_rate)
                w.writeframes(audio.tobytes())

        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = self._sr_recognizer.record(source)
            text = self._sr_recognizer.recognize_google(audio_data)
            return text.strip() if text else None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.warning(f"Google Speech API error: {e}")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.debug(repr(e))

    def listen_interactive(self, prompt: str = "Listening...") -> str | None:
        """Listen with a prompt printed to stdout.

        Prints the prompt, listens for speech, and returns the text.
        Returns None if no speech is detected within 5 seconds.
        """
        sys.stderr.write(f"  [mic] {prompt}")
        sys.stderr.flush()
        text = self.listen(timeout=5.0, phrase_limit=15.0)
        if text:
            logger.info(f' -> "{text}"')
        else:
            logger.info(" (no speech detected)")
        return text

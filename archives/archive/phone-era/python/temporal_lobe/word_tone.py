"""Word tone — the sound-form of a word as a fixed-length vector.

A word never enters the brain as a symbol. Heard, it is already a
tonotopic profile — the cochlea disperses frequency content along the
basilar membrane so every auditory nerve fiber is a labeled line for
one characteristic frequency (the rate-place code; Delgutte 1997).
Read, it is recoded into the same phonological form — anterior STG
neurons tuned to phoneme sets fire to the written letters of those
phonemes (phonological recoding during reading). Recalled, it is the
same form reactivated — the phonological loop's rehearsal is inner
speech driving the same representation again.

This module performs that transduction for her: her own voice (Piper)
synthesizes the word offline to raw PCM — subvocalization rendered —
and a cochlear analysis turns it into the word's signature:

- 25 ms Hann frames, 10 ms hop,
- a mel-spaced filterbank over the speech band (60–8000 Hz) — the
  linear place-map approximation of basilar tuning,
- log energy per channel (the compressive nonlinearity of the
  cochlear amplifier),
- duration-normalized to a fixed frame count, so every word's
  signature has the same dimension and word ORDER is preserved —
  "cat" and "tack" share phonemes but not trajectories.

The result is a vector in the space's currency: deposit it and the
seeded projection lands phonetically similar words near each other —
proximity means something because the input is sound, not spelling.

Routes, in order of fidelity:

1. Piper (her voice) → raw s16le on stdout — no audio device needed.
2. espeak-ng → WAV file — a coarser voice, still a real spectrum.
3. Neither → ``None``; the caller may deposit vectorless (the space
   lands the name's digest) rather than fabricate a tone.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import wave

import numpy as np
from self.speech import _find_voices_dir

logger = logging.getLogger(__name__)

# Cochlear analysis parameters — speech band on the basilar map.
_N_CHANNELS = 48  # mel-spaced place channels
_F_MIN = 60.0  # Hz — below the lowest voicing fundamental
_F_MAX = 8000.0  # Hz — above the highest useful formant energy
_FRAME_S = 0.025  # 25 ms analysis window
_HOP_S = 0.010  # 10 ms hop
_N_FRAMES = 16  # duration-normalized trajectory length
_SILENCE_DB = 40.0  # trim frames this far below peak energy

_VOICES_DIR = _find_voices_dir()
_PIPER_BIN = _VOICES_DIR / "piper"
_PIPER_MODEL = _VOICES_DIR / "amy-medium.onnx"
_PIPER_FALLBACK = _VOICES_DIR / "amy-low.onnx"


# ─── Synthesis (the inner voice) ──────────────────────────────────


def _synthesize_piper(word: str) -> tuple[np.ndarray, int] | None:
    """Speak the word with her own voice → (s16le PCM, sample rate)."""
    model = _PIPER_MODEL if _PIPER_MODEL.exists() else _PIPER_FALLBACK
    if not (_PIPER_BIN.exists() and model.exists()):
        return None
    env = dict(os.environ, LD_LIBRARY_PATH=str(_VOICES_DIR))
    try:
        proc = subprocess.run(
            [_PIPER_BIN, "--model", str(model), "--output_raw"],
            input=(word + "\n").encode("utf-8"),
            capture_output=True,
            env=env,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("piper synthesis failed: %s", e)
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    cfg = model.with_suffix(model.suffix + ".json")
    try:
        import json

        sr = int(json.loads(cfg.read_text())["audio"]["sample_rate"])
    except (OSError, ValueError, KeyError):
        sr = 22050
    return np.frombuffer(proc.stdout, dtype="<i2"), sr


def _synthesize_espeak(word: str) -> tuple[np.ndarray, int] | None:
    """espeak-ng → WAV bytes → (s16le PCM, sample rate)."""
    try:
        proc = subprocess.run(
            ["espeak-ng", "-q", "-v", "en-us", "--stdout", word],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("espeak synthesis failed: %s", e)
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        with wave.open(io.BytesIO(proc.stdout)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            return pcm, int(w.getframerate())
    except (wave.Error, EOFError):
        return None


def synthesize(word: str) -> tuple[np.ndarray, int] | None:
    """The word as sound — her voice if she has one, espeak if not."""
    return _synthesize_piper(word) or _synthesize_espeak(word)


# ─── The cochlea ──────────────────────────────────────────────────


def _mel(f_hz: float) -> float:
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def _filterbank(n_fft: int, sample_rate: int) -> np.ndarray:
    """Triangular mel-spaced filters — the basilar place map, linearized."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    edges = np.linspace(_mel(_F_MIN), _mel(_F_MAX), _N_CHANNELS + 2)
    centers = 700.0 * (10.0 ** (edges / 2595.0) - 1.0)
    bank = np.zeros((_N_CHANNELS, freqs.size))
    for c in range(_N_CHANNELS):
        lo, mid, hi = centers[c], centers[c + 1], centers[c + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        bank[c] = np.clip(np.minimum(up, down), 0.0, 1.0)
    return bank


def cochleagram(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """Rate-place profile over time: (frames, channels) log energy."""
    x = np.asarray(pcm, dtype=np.float64) / 32768.0
    n_frame = max(8, round(_FRAME_S * sample_rate))
    n_hop = max(4, round(_HOP_S * sample_rate))
    if x.size < n_frame:
        x = np.pad(x, (0, n_frame - x.size))
    n = 1 + (x.size - n_frame) // n_hop
    window = np.hanning(n_frame)
    bank = _filterbank(n_frame, sample_rate)
    out = np.empty((n, _N_CHANNELS))
    for i in range(n):
        frame = x[i * n_hop : i * n_hop + n_frame] * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        out[i] = np.log10(bank @ power + 1e-12)
    return out


def signature(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """The word's tone-vector — duration-normalized spectral trajectory.

    (frames, channels) → silence-trimmed → resampled to _N_FRAMES →
    RMS-normalized → flattened. Deterministic: same word, same site.
    """
    gram = cochleagram(pcm, sample_rate)
    peak = float(gram.max())
    voiced = gram[gram.max(axis=1) > peak - _SILENCE_DB / 10.0]
    if voiced.shape[0] == 0:
        voiced = gram
    # Resample the time axis to the fixed trajectory length.
    t = np.linspace(0.0, 1.0, voiced.shape[0])
    tq = np.linspace(0.0, 1.0, _N_FRAMES)
    sig = np.stack([np.interp(tq, t, voiced[:, c]) for c in range(_N_CHANNELS)], axis=1)
    sig = sig - sig.mean()
    rms = float(np.sqrt((sig**2).mean()))
    if rms > 1e-9:
        sig = sig / rms
    return sig.reshape(-1)


def word_signature(word: str) -> np.ndarray | None:
    """Text → sound → spectral signature. None if she cannot voice it."""
    got = synthesize(word)
    if got is None:
        return None
    pcm, sr = got
    if pcm.size == 0:
        return None
    return signature(pcm, sr)


__all__ = [
    "cochleagram",
    "signature",
    "synthesize",
    "word_signature",
]

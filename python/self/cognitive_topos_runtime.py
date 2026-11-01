"""Live adapter connecting the cognitive topos engine to Genesis's event stream.

The adapter consumes actual outer-world experiential state, the shared self
latent state, inner-life/learning signals, and emitted language. It does not
invent a separate sensor stream. The fixed feature mapping is deterministic
and documented; it is a bridge between existing unequal-dimensional state
representations, not a learned semantic encoder.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any

import numpy as np

from self.master_cognitive_topos import MasterCognitiveToposEngine

_STATE_DIM = 32


def _resize(values: Any, dim: int = _STATE_DIM) -> np.ndarray:
    """Deterministically resample a finite numeric state to the engine width."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.zeros(dim, dtype=np.float64)
    if array.size == 1:
        return np.full(dim, np.tanh(array[0]), dtype=np.float64)
    source = np.linspace(0.0, 1.0, array.size)
    target = np.linspace(0.0, 1.0, dim)
    return np.tanh(np.interp(target, source, array))


def _text_features(kind: str, content: str, dim: int = _STATE_DIM) -> np.ndarray:
    """Stable content features plus transparent scalar observables.

    Digest bytes provide reproducible bounded identity features; scalar
    features encode length and lexical diversity. They are not embeddings and
    do not claim to encode meaning.
    """
    normalized = content[:4096].casefold()
    digest = hashlib.sha256((kind + "\0" + normalized).encode("utf-8", errors="replace")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float64)
    raw = (raw / 127.5) - 1.0
    tokens = normalized.split()
    scalars = np.array([
        min(len(normalized), 4096) / 4096.0,
        min(len(tokens), 256) / 256.0,
        len(set(tokens)) / max(len(tokens), 1),
        float(bool(normalized)),
    ])
    return _resize(np.concatenate((raw, scalars)), dim)


class CognitiveToposRuntime:
    """Thread-safe bridge from live Genesis events to the unified engine."""

    def __init__(self, latent_dim: int = _STATE_DIM) -> None:
        if (
            isinstance(latent_dim, bool)
            or not isinstance(latent_dim, int)
            or latent_dim < 1
        ):
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.engine = MasterCognitiveToposEngine(latent_dim)
        self._lock = threading.RLock()
        self._states = {
            "outer": np.zeros(latent_dim, dtype=np.float64),
            "inner": np.zeros(latent_dim, dtype=np.float64),
            "self": np.zeros(latent_dim, dtype=np.float64),
            "language": np.zeros(latent_dim, dtype=np.float64),
        }
        self._phases = np.zeros((1, latent_dim), dtype=np.float64)
        self._last_output: dict[str, Any] = {}
        self._updates = 0
        # Positive symmetric adjacency; diagonal self-coupling is excluded.
        self._metric_weights = np.ones((1, latent_dim, latent_dim), dtype=np.float64)
        np.fill_diagonal(self._metric_weights[0], 0.0)
        self._metric_weights /= max(latent_dim - 1, 1)

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    def observe(
        self,
        domain: str,
        *,
        kind: str,
        content: str,
        self_snapshot: dict[str, Any] | None = None,
        outer_snapshot: dict[str, Any] | None = None,
        inner_snapshot: dict[str, Any] | None = None,
        language_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update one live domain and recompute the integrated macro-state."""
        if domain not in self._states:
            raise ValueError(f"unknown cognitive domain: {domain}")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        with self._lock:
            if domain == "outer" and outer_snapshot:
                source = outer_snapshot.get("state")
                self._states[domain] = (
                    _resize(source, self.latent_dim)
                    if source is not None
                    else _text_features(kind, content, self.latent_dim)
                )
            elif domain == "inner" and inner_snapshot:
                source = inner_snapshot.get("recursive_meta_states")
                if source is None:
                    source = inner_snapshot.get("affective_fiber_states")
                if source is None:
                    source = inner_snapshot.get("latent_mean")
                self._states[domain] = _resize(source, self.latent_dim) if source is not None else _text_features(kind, content, self.latent_dim)
            elif domain == "self" and self_snapshot:
                source = self_snapshot.get("agency_fiber_states")
                self._states[domain] = _resize(source, self.latent_dim) if source is not None else _text_features(kind, content, self.latent_dim)
            elif domain == "language" and language_snapshot:
                source = language_snapshot.get("latent_state")
                self._states[domain] = _resize(source, self.latent_dim) if source is not None else _text_features(kind, content, self.latent_dim)
            else:
                self._states[domain] = _text_features(kind, content, self.latent_dim)

            # Self state is refreshed from the same real signal whenever it
            # exists; do not fabricate a second self model if a caller omits it.
            if self_snapshot and self_snapshot.get("agency_fiber_states") is not None:
                self._states["self"] = _resize(
                    self_snapshot["agency_fiber_states"], self.latent_dim
                )

            states = [self._states[key][None, :] for key in ("outer", "inner", "self", "language")]
            # The oscillator phases are tied to current state coordinates.
            # Integrate one bounded phase step, then carry it into the next event.
            self._phases = np.arctan2(np.sin(self._phases + self._states[domain][None, :]),
                                     np.cos(self._phases + self._states[domain][None, :]))
            output = self.engine.forward(
                *states,
                self._phases,
                self._metric_weights,
                K=1.5,
                dt=0.01,
            )
            self._phases = output["synchronized_phases"]
            self._updates += 1
            self._last_output = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in output.items()
            }
            self._last_output["updates"] = self._updates
            self._last_output["last_domain"] = domain
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "updates": self._updates,
                "states": {key: value.tolist() for key, value in self._states.items()},
                "output": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in self._last_output.items()
                },
            }

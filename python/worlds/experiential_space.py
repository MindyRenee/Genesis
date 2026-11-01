"""Numerical experiential geometry for Genesis's outer world.

This is a NumPy-only, deterministic state representation over observed
world events. It is not a claim that the representation constitutes
subjective experience or physically models spacetime. It preserves the
useful geometric ideas without introducing an untrained Torch network
or a new runtime dependency.

The state is grounded in event fields already present in OuterWorld:
event kind, direction, salience, addressedness, topic count, and time.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from temporal_lobe.events import EventKind, ExternalEvent

_STATE_DIM = 20
_KIND_COUNT = len(EventKind)
_EPS = 1e-8


class ExperientialLatentSpace:
    """Online geometric summary of the outer-world event stream."""

    def __init__(self) -> None:
        self._state = np.zeros(_STATE_DIM, dtype=np.float64)
        self._seen = 0
        self._mean = np.zeros(_STATE_DIM, dtype=np.float64)
        self._m2 = np.zeros(_STATE_DIM, dtype=np.float64)
        self._kind_counts = np.zeros(_KIND_COUNT, dtype=np.float64)
        self._last_timestamp = 0.0
        self._last: dict[str, Any] = {}

    @property
    def dimension(self) -> int:
        return _STATE_DIM

    @property
    def observations(self) -> int:
        return self._seen

    def observe(self, event: ExternalEvent) -> dict[str, Any]:
        """Encode one real world event and update online geometry."""
        features = self._encode(event)
        self._seen += 1
        delta = features - self._mean
        self._mean += delta / self._seen
        self._m2 += delta * (features - self._mean)
        kind_index = list(EventKind).index(event.kind)
        self._kind_counts[kind_index] += 1.0
        self._state = features
        self._last_timestamp = float(event.timestamp)
        self._last = self._derive(features, self._last_timestamp)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return independent, JSON-friendly values for inspection/persistence."""
        if not self._last:
            return {
                "observations": 0,
                "state": self._state.tolist(),
                "base_metric": np.eye(4).tolist(),
                "conformal_scale": 1.0,
                "toroidal_state": [0.0] * 8,
                "global_geometry": {
                    "hyperbolic_state": [0.0] * 6,
                    "spherical_state": [0.0] * 6,
                    "euclidean_state": [0.0] * 8,
                },
                "fisher_information": np.eye(_STATE_DIM).tolist(),
                "fisher_volume": 1.0,
                "symplectic_state": self._state.tolist(),
            }
        result = dict(self._last)
        result["observations"] = self._seen
        result["state"] = self._state.tolist()
        return result

    def to_dict(self) -> dict[str, Any]:
        """Persist sufficient statistics; derived matrices are recomputed."""
        return {
            "version": 1,
            "observations": self._seen,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
            "kind_counts": self._kind_counts.tolist(),
            "state": self._state.tolist(),
            "last_timestamp": self._last_timestamp,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore only shape- and range-validated numerical state."""
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported experiential-space state")
        seen = int(data.get("observations", 0))
        mean = np.asarray(data.get("mean"), dtype=np.float64)
        m2 = np.asarray(data.get("m2"), dtype=np.float64)
        counts = np.asarray(data.get("kind_counts"), dtype=np.float64)
        state = np.asarray(data.get("state"), dtype=np.float64)
        last_timestamp = float(data.get("last_timestamp", 0.0))
        if (
            seen < 0
            or mean.shape != (_STATE_DIM,)
            or m2.shape != (_STATE_DIM,)
            or state.shape != (_STATE_DIM,)
            or counts.shape != (_KIND_COUNT,)
            or not math.isfinite(last_timestamp)
            or not all(np.all(np.isfinite(a)) for a in (mean, m2, counts, state))
            or np.any(m2 < -1e-9)
            or np.any(counts < 0)
        ):
            raise ValueError("malformed experiential-space state")
        self._seen, self._mean, self._m2 = seen, mean, np.maximum(m2, 0.0)
        self._kind_counts, self._state = counts, state
        self._last_timestamp = last_timestamp
        self._last = self._derive(state, last_timestamp) if seen else {}

    @staticmethod
    def _encode(event: ExternalEvent) -> np.ndarray:
        values = np.zeros(_STATE_DIM, dtype=np.float64)
        values[list(EventKind).index(event.kind)] = 1.0
        values[8] = float(np.clip(event.salience, 0.0, 1.0))
        values[9] = float(event.addressed)
        values[10] = float(event.inbound)
        values[11] = min(len(event.topics), 5) / 5.0
        # Cyclic time coordinates encode daily phase, not absolute time.
        phase = (event.timestamp % 86400.0) / 86400.0 * 2.0 * math.pi
        values[12] = math.sin(phase)
        values[13] = math.cos(phase)
        # Deterministic source features avoid a learned or random embedding.
        source = event.source.encode("utf-8", errors="replace")
        values[14] = (sum(source) % 997) / 996.0 if source else 0.0
        values[15] = min(len(event.content), 300) / 300.0
        # Structured sensory payloads (e.g. ARC grids or numeric camera
        # features) are summarized deterministically; never stringify them.
        try:
            payload = np.asarray(event.percept, dtype=np.float64).reshape(-1)[:4096]
            payload = payload[np.isfinite(payload)]
        except (TypeError, ValueError):
            payload = np.empty(0, dtype=np.float64)
        if payload.size:
            values[16] = math.tanh(float(np.mean(payload)))
            values[17] = math.tanh(float(np.std(payload)))
            values[18] = math.tanh(float(np.min(payload)))
            values[19] = math.tanh(float(np.max(payload)))
        return values

    def _derive(self, state: np.ndarray, timestamp: float) -> dict[str, Any]:
        # Attention is bounded and derived from observed salience/direction.
        attention = np.clip(np.abs(state[8:12]), 0.0, 1.0)
        # Bound exponent to prevent overflow; this is Euclidean feature geometry,
        # not a physical spacetime metric.
        conformal_scale = float(math.exp(min(2.0, float(attention @ attention))))
        base_metric = conformal_scale * np.eye(4) + 0.1 * np.outer(attention, attention)
        # A product space: negative-curvature ball, positive-curvature sphere,
        # and flat Euclidean coordinates, each with its own valid domain.
        hyper = np.tanh(state[:6])
        norm_h = float(np.linalg.norm(hyper))
        if norm_h >= 0.95:
            hyper *= 0.95 / max(norm_h, _EPS)
        sphere = state[6:12].copy()
        sphere_norm = float(np.linalg.norm(sphere))
        if sphere_norm > _EPS:
            sphere /= sphere_norm
        else:
            sphere[0] = 1.0
        euclidean = state[12:20].copy()
        hyperbolic_norm = min(float(np.linalg.norm(hyper)), 1.0 - _EPS)
        hyperbolic_distance_from_origin = 2.0 * math.atanh(hyperbolic_norm)
        # Geodesic distance from the sphere north pole on the unit sphere.
        spherical_distance_from_north_pole = math.acos(float(np.clip(sphere[0], -1.0, 1.0)))
        euclidean_distance_from_origin = float(np.linalg.norm(euclidean))
        # Four S^1 factors encode four sensory summary coordinates. Their
        # phase is driven by observed payload values, not a random projection.
        sensory_phase = np.pi * np.tanh(state[16:20])
        toroidal = np.array(
            [component for phase in sensory_phase
             for component in (math.sin(float(phase)), math.cos(float(phase)))],
            dtype=np.float64,
        )
        # Diagonal Fisher information for independent Bernoulli feature
        # parameters: I_i = 1 / (p_i * (1-p_i)). Online means provide p_i;
        # clipping avoids singular infinities at the boundary.
        p = self._mean.copy()
        # The two cyclic coordinates are signed; map them from [-1, 1]
        # into the Bernoulli parameter domain before evaluating Fisher info.
        p[12:14] = (p[12:14] + 1.0) / 2.0
        p = np.clip(p, 1e-3, 1.0 - 1e-3)
        fisher_diag = 1.0 / (p * (1.0 - p))
        fim = np.diag(np.clip(fisher_diag, 1.0, 1000.0))
        sign, logdet = np.linalg.slogdet(fim)
        volume = float(math.exp(min(300.0, 0.5 * logdet))) if sign > 0 else 0.0
        # Exact harmonic Hamiltonian flow H=(q·q+p·p)/2, a symplectic
        # rotation; unlike explicit Euler, this preserves phase-space area.
        q, p_phase = state[:10], state[10:20]
        dt = 0.01
        c, s = math.cos(dt), math.sin(dt)
        evolved = np.concatenate((c * q + s * p_phase, c * p_phase - s * q))
        return {
            "base_metric": base_metric.tolist(),
            "conformal_scale": conformal_scale,
            "toroidal_state": toroidal.tolist(),
            "global_geometry": {
                "hyperbolic_state": hyper.tolist(),
                "spherical_state": sphere.tolist(),
                "euclidean_state": euclidean.tolist(),
                "distances_from_reference": {
                    "hyperbolic": hyperbolic_distance_from_origin,
                    "spherical": spherical_distance_from_north_pole,
                    "euclidean": euclidean_distance_from_origin,
                },
            },
            "fisher_information": fim.tolist(),
            "fisher_volume": volume,
            "symplectic_state": evolved.tolist(),
        }

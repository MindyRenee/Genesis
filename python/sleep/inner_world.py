"""Machine-native inner-world geometry for Genesis.

NumPy-only implementation of homeostatic interoception, affective circular
fibres, a genuine prefix-tree ultrametric, diagonal Gaussian KL geometry, and
a norm-preserving Hamiltonian update. This is a numerical cognitive state
model, not evidence of subjective experience. No neural-network dependency or
randomly initialized encoder is introduced.
"""
from __future__ import annotations

import math
import threading
from copy import deepcopy
from typing import Any

import numpy as np

_BASE_DIM = 4
_INTERO_DIM = 8
_FIBER_DIM = 32
_AFFECTIVE_FACTORS = 4
_TORUS_DIM = 2 * _AFFECTIVE_FACTORS
_LATENT_DIM = _FIBER_DIM + _TORUS_DIM
_EPS = 1e-8


def _finite_vector(values: Any, size: int, name: str) -> np.ndarray:
    """Return a finite float vector of exact size or raise a clear error."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


class InteroceptiveBaseManifold:
    """Homeostatic state and a positive-definite deviation-sensitive metric."""

    def __init__(self, setpoint: Any | None = None) -> None:
        self.setpoint = (
            np.zeros(_BASE_DIM, dtype=np.float64)
            if setpoint is None
            else _finite_vector(setpoint, _BASE_DIM, "setpoint").copy()
        )

    def forward(self, current: Any) -> tuple[np.ndarray, float]:
        state = _finite_vector(current, _BASE_DIM, "homeostatic state")
        delta = np.clip(state - self.setpoint, -math.sqrt(10.0), math.sqrt(10.0))
        deviation = min(float(np.dot(delta, delta)), 10.0)
        scale = math.exp(deviation)
        return np.eye(_BASE_DIM, dtype=np.float64) * scale, scale

    def adapt_setpoint(self, current: Any, rate: float = 0.001) -> None:
        """Slowly adapt the viable-state baseline without chasing transients."""
        state = _finite_vector(current, _BASE_DIM, "homeostatic state")
        if not math.isfinite(rate) or not 0.0 <= rate <= 0.01:
            raise ValueError("setpoint adaptation rate must be in [0, 0.01]")
        self.setpoint += rate * (state - self.setpoint)

    def to_dict(self) -> dict[str, Any]:
        return {"setpoint": self.setpoint.tolist()}

    def restore(self, data: dict[str, Any]) -> None:
        self.setpoint = _finite_vector(data["setpoint"], _BASE_DIM, "setpoint").copy()


class AffectiveFiberQualiaSpace:
    """Deterministic affective encoding with four genuine circular factors.

    The feature map is fixed and interpretable; it does not claim that
    generated coordinates are qualia or that an untrained network has learned.
    """

    def __init__(self) -> None:
        self._frequencies = np.arange(1, _FIBER_DIM + 1, dtype=np.float64)

    def forward(self, signals: Any) -> np.ndarray:
        x = np.clip(
            _finite_vector(signals, _INTERO_DIM, "interoceptive signals"),
            -1.0,
            1.0,
        )
        # Bounded deterministic Fourier feature map, 32 coordinates.
        projected = np.outer(self._frequencies, x)
        weights = np.arange(1, _INTERO_DIM + 1, dtype=np.float64)
        phases = projected @ (weights / np.linalg.norm(weights))
        base = np.concatenate((np.sin(phases[:16]), np.cos(phases[16:])))
        # Valence/arousal and two other measured axes define four S1 factors.
        axes = np.array([x[0], x[1], x[2], x[3]], dtype=np.float64)
        angles = np.pi * np.tanh(axes)
        torus = np.array(
            [component for angle in angles
             for component in (math.sin(float(angle)), math.cos(float(angle)))],
            dtype=np.float64,
        )
        return np.concatenate((base, torus))


class UltrametricEpisodicMemorySpace:
    """Bounded prefix-tree memory with an exact strong triangle inequality.

    Each vector is quantized into a finite hierarchical code. Distance is
    determined by the first differing prefix level, not by an L-infinity
    norm (which is not generally ultrametric). Stored records are bounded.
    """

    def __init__(self, capacity: int = 512, levels: int = 8) -> None:
        if capacity < 1 or levels < 1 or levels > 16:
            raise ValueError("capacity must be positive and levels must be in [1, 16]")
        self.capacity = capacity
        self.levels = levels
        self._records: list[dict[str, Any]] = []
        self._next_id = 0

    def _code(self, vector: Any) -> tuple[int, ...]:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size != _LATENT_DIM or not np.all(np.isfinite(values)):
            raise ValueError(f"memory vector must be finite with {_LATENT_DIM} values")
        # Map each coordinate to [0, 1); binary levels form a prefix code.
        bounded = np.clip((np.tanh(values) + 1.0) * 0.5, 0.0, 1.0 - 1e-12)
        bins = np.floor(bounded * (2**self.levels)).astype(np.int64)
        return tuple(int(item) for item in bins)

    def _distance_codes(self, a: tuple[int, ...], b: tuple[int, ...]) -> float:
        """Prefix-tree distance on validated quantized codes."""
        if a == b:
            return 0.0
        a_bits = "".join(format(value, f"0{self.levels}b") for value in a)
        b_bits = "".join(format(value, f"0{self.levels}b") for value in b)
        common = 0
        for abit, bbit in zip(a_bits, b_bits, strict=True):
            if abit != bbit:
                break
            common += 1
        return math.ldexp(1.0, -(common + 1))

    def distance(self, left: Any, right: Any) -> float:
        """Tree distance: zero for identical leaves, otherwise 2^-LCP."""
        return self._distance_codes(self._code(left), self._code(right))

    def add(self, vector: Any, label: str = "") -> int:
        code = self._code(vector)
        record_id = self._next_id
        self._next_id += 1
        self._records.append({"id": record_id, "code": list(code), "label": str(label)[:256]})
        if len(self._records) > self.capacity:
            del self._records[: len(self._records) - self.capacity]
        return record_id

    @property
    def size(self) -> int:
        return len(self._records)

    def nearest(self, vector: Any, limit: int = 5) -> list[dict[str, Any]]:
        query = self._code(vector)
        ranked = []
        for record in self._records:
            distance = self._distance_codes(query, tuple(record["code"]))
            ranked.append({"id": record["id"], "label": record["label"], "distance": distance})
        return sorted(ranked, key=lambda item: (item["distance"], item["id"]))[:max(0, limit)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "capacity": self.capacity,
            "levels": self.levels,
            "next_id": self._next_id,
            "records": [dict(item) for item in self._records],
        }

    def restore(self, data: dict[str, Any]) -> None:
        if data.get("version") != 1:
            raise ValueError("unsupported inner-memory state")
        capacity = int(data["capacity"])
        levels = int(data["levels"])
        if capacity < 1 or not 1 <= levels <= 16:
            raise ValueError("invalid inner-memory configuration")
        records = data.get("records", [])
        if not isinstance(records, list) or len(records) > capacity:
            raise ValueError("invalid inner-memory records")
        checked = []
        seen_ids = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid inner-memory record")
            record_id = int(record["id"])
            code = tuple(int(value) for value in record["code"])
            if record_id < 0 or record_id in seen_ids or len(code) != _LATENT_DIM:
                raise ValueError("invalid inner-memory id or code")
            if any(value < 0 or value >= 2**levels for value in code):
                raise ValueError("inner-memory code out of range")
            seen_ids.add(record_id)
            checked.append(
                {
                    "id": record_id,
                    "code": list(code),
                    "label": str(record.get("label", ""))[:256],
                }
            )
        next_id = int(data["next_id"])
        if next_id < 0 or (seen_ids and next_id <= max(seen_ids)):
            raise ValueError("invalid next inner-memory id")
        self.capacity, self.levels = capacity, levels
        self._records, self._next_id = checked, next_id


class VariationalFreeEnergyGeometry:
    """Diagonal Gaussian approximate-posterior KL geometry."""

    def forward(self, latent: Any) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        x = _finite_vector(latent, _LATENT_DIM, "latent state")
        # Fixed, stable posterior map: no untrained affine network.
        mu = np.tanh(x)
        logvar = np.clip(0.5 * np.tanh(np.roll(x, 1)), -8.0, 8.0)
        variance = np.exp(logvar)
        precision = 1.0 / variance
        metric = np.diag(precision)
        free_energy = float(0.5 * np.sum(mu * mu + variance - logvar - 1.0))
        return metric, free_energy, mu, variance


class RecursiveMetaCognitiveSymplecticFlow:
    """Exact harmonic Hamiltonian step, preserving phase-space norm."""

    def __init__(self, dt: float = 0.01) -> None:
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            raise ValueError("dt must be finite and in (0, 1]")
        self.dt = float(dt)

    def forward(self, state: Any) -> np.ndarray:
        z = _finite_vector(state, _LATENT_DIM, "symplectic state")
        half = _LATENT_DIM // 2
        q, p = z[:half], z[half:]
        cosine, sine = math.cos(self.dt), math.sin(self.dt)
        return np.concatenate((cosine * q + sine * p, cosine * p - sine * q))


class InnerWorldExperientialLatentSpace:
    """Integrated inner-state model wired for explicit observations."""

    def __init__(self, memory_capacity: int = 512) -> None:
        self.base_manifold = InteroceptiveBaseManifold()
        self.affective_bundle = AffectiveFiberQualiaSpace()
        self.ultrametric_memory = UltrametricEpisodicMemorySpace(capacity=memory_capacity)
        self.free_energy_geometry = VariationalFreeEnergyGeometry()
        self.recursive_symplectic_flow = RecursiveMetaCognitiveSymplecticFlow()
        self._observations = 0
        self._last: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def observations(self) -> int:
        return self._observations

    def observe(
        self,
        homeostatic_state: Any,
        interoceptive_signals: Any,
        label: str = "",
    ) -> dict[str, Any]:
        with self._lock:
            metric, scale = self.base_manifold.forward(homeostatic_state)
            self.base_manifold.adapt_setpoint(homeostatic_state)
            fiber = self.affective_bundle.forward(interoceptive_signals)
            memory_id = self.ultrametric_memory.add(fiber, label)
            kl_metric, free_energy, mu, variance = self.free_energy_geometry.forward(fiber)
            evolved = self.recursive_symplectic_flow.forward(fiber)
            self._observations += 1
            self._last = {
                "observations": self._observations,
                "internal_base_metric": metric.tolist(),
                "autonomic_scaling_factor": scale,
                "affective_fiber_states": fiber.tolist(),
                "ultrametric_memory_nodes": {
                    "memory_id": memory_id,
                    "size": self.ultrametric_memory.size,
                    "code": list(self.ultrametric_memory._code(fiber)),
                },
                "kl_information_metric": kl_metric.tolist(),
                "free_energy_divergence": free_energy,
                "latent_mean": mu.tolist(),
                "latent_variance": variance.tolist(),
                "recursive_meta_states": evolved.tolist(),
            }
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if not self._last:
                return {
                    "observations": 0,
                    "internal_base_metric": np.eye(_BASE_DIM).tolist(),
                    "autonomic_scaling_factor": 1.0,
                    "affective_fiber_states": [0.0] * _LATENT_DIM,
                    "ultrametric_memory_nodes": {
                        "memory_id": None,
                        "size": 0,
                        "code": [],
                    },
                    "kl_information_metric": np.eye(_LATENT_DIM).tolist(),
                    "free_energy_divergence": 0.0,
                    "latent_mean": [0.0] * _LATENT_DIM,
                    "latent_variance": [1.0] * _LATENT_DIM,
                    "recursive_meta_states": [0.0] * _LATENT_DIM,
                }
            return deepcopy(self._last)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "observations": self._observations,
                "last": self.snapshot(),
                "base_manifold": self.base_manifold.to_dict(),
                "memory": self.ultrametric_memory.to_dict(),
            }

    def restore(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported inner-world state")
        observations = int(data.get("observations", 0))
        if observations < 0:
            raise ValueError("invalid observation count")
        base = InteroceptiveBaseManifold()
        base.restore(data["base_manifold"])
        memory = UltrametricEpisodicMemorySpace()
        memory.restore(data["memory"])
        last = data.get("last", {})
        if not isinstance(last, dict):
            raise ValueError("invalid inner-world snapshot")
        if observations:
            expected_shapes = {
                "internal_base_metric": (_BASE_DIM, _BASE_DIM),
                "affective_fiber_states": (_LATENT_DIM,),
                "kl_information_metric": (_LATENT_DIM, _LATENT_DIM),
                "latent_mean": (_LATENT_DIM,),
                "latent_variance": (_LATENT_DIM,),
                "recursive_meta_states": (_LATENT_DIM,),
            }
            for key, shape in expected_shapes.items():
                array = np.asarray(last.get(key), dtype=np.float64)
                if array.shape != shape or not np.all(np.isfinite(array)):
                    raise ValueError(f"invalid inner-world snapshot field: {key}")
            scale = float(last.get("autonomic_scaling_factor", float("nan")))
            free_energy = float(last.get("free_energy_divergence", float("nan")))
            if not math.isfinite(scale) or scale < 1.0:
                raise ValueError("invalid autonomic scaling factor")
            if not math.isfinite(free_energy) or free_energy < -1e-9:
                raise ValueError("invalid free-energy divergence")
        with self._lock:
            self.base_manifold = base
            self.ultrametric_memory = memory
            self._observations = observations
            self._last = dict(last)

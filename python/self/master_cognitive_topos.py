"""NumPy-native unified cognitive-state transport and synchronization.

This is a deterministic mathematical subsystem for Genesis3, not a claim that
category-theoretic terminology alone establishes cognition. It deliberately
uses NumPy (already required by Genesis) rather than introducing a PyTorch
runtime dependency. All inputs are validated and all transformations are
explicit; no pretrained model or language model is used.

The four state spaces are supplied by the caller: outer world (E), inner
state (I), self state (S), and language/symbol state (L). "Topos" here names
the architecture's interface between representations; this module does not
implement a Grothendieck topos or prove categorical universal properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _finite_array(value: object, name: str, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _state(value: object, name: str, dim: int | None = None) -> FloatArray:
    result = _finite_array(value, name, 2)
    if result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must have non-empty batch and feature dimensions")
    if dim is not None and result.shape[1] != dim:
        raise ValueError(f"{name} must have {dim} features; got {result.shape[1]}")
    return result


@dataclass(slots=True)
class EhresmannConnectionTransport:
    """Learnable-by-caller linear connection between finite-dimensional fibers.

    The matrix is explicit state, not automatically trained. Row-vector input
    x is transported by x @ connection.T and normalized per sample. A zero
    vector remains zero, avoiding the arbitrary offset introduced by some
    normalization implementations.
    """

    source_dim: int
    target_dim: int
    connection: FloatArray | None = None
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.source_dim < 1 or self.target_dim < 1:
            raise ValueError("source_dim and target_dim must be positive")
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite and positive")
        if self.connection is None:
            # Deterministic, dimension-aware initialization; no RNG state.
            rows = np.arange(1, self.target_dim + 1, dtype=np.float64)[:, None]
            cols = np.arange(1, self.source_dim + 1, dtype=np.float64)[None, :]
            matrix = np.sin(rows * cols * np.sqrt(2.0))
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), self.epsilon)
            self.connection = matrix
        else:
            self.connection = _finite_array(self.connection, "connection", 2)
            if self.connection.shape != (self.target_dim, self.source_dim):
                raise ValueError(
                    f"connection must have shape {(self.target_dim, self.source_dim)}"
                )

    def forward(self, state_source: object) -> FloatArray:
        source = _state(state_source, "state_source", self.source_dim)
        transported = source @ self.connection.T
        norms = np.linalg.norm(transported, axis=-1, keepdims=True)
        return np.divide(
            transported,
            np.maximum(norms, self.epsilon),
            out=np.zeros_like(transported),
            where=norms > self.epsilon,
        )


@dataclass(slots=True)
class MasterFreeEnergyOptimizer:
    """Compute coupled state energy and alignment penalties.

    Inputs are per-sample non-negative energy vectors. This is an explicit
    objective evaluator, not an optimizer: gradients/parameter updates require
    a specified learning rule and observations, neither of which is fabricated.
    """

    lambda_1: float = 1.0
    lambda_2: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_1) or self.lambda_1 < 0:
            raise ValueError("lambda_1 must be finite and non-negative")
        if not np.isfinite(self.lambda_2) or self.lambda_2 < 0:
            raise ValueError("lambda_2 must be finite and non-negative")

    def forward(
        self,
        f_outer: object,
        f_inner: object,
        f_self: object,
        f_lang: object,
    ) -> tuple[float, float]:
        energies = [
            _finite_array(value, name, 1)
            for value, name in (
                (f_outer, "f_outer"),
                (f_inner, "f_inner"),
                (f_self, "f_self"),
                (f_lang, "f_lang"),
            )
        ]
        if any(item.size == 0 for item in energies):
            raise ValueError("energy vectors must not be empty")
        if any(item.shape != energies[0].shape for item in energies[1:]):
            raise ValueError("all energy vectors must have the same batch shape")
        if any(np.any(item < 0) for item in energies):
            raise ValueError("energies must be non-negative")
        outer, inner, self_energy, language = energies
        base_energy = float(np.mean(outer + inner + self_energy + language))
        # Penalize per-sample mismatch before aggregation. Taking the
        # absolute value only after averaging would let opposite errors
        # cancel across a batch and report false alignment.
        outer_language = float(np.mean(np.abs(outer - language)))
        inner_self = float(np.mean(np.abs(inner - self_energy)))
        coupling = outer_language + inner_self
        total = base_energy + self.lambda_1 * coupling + self.lambda_2 * (
            outer_language**2 + inner_self**2
        )
        return float(total), float(coupling)


@dataclass(slots=True)
class KuramotoRiemannianSynchronizer:
    """One explicit Euler step of weighted Kuramoto phase dynamics.

    metric_weights[b, i, j] weights oscillator j's influence on oscillator i.
    The coupling is sin(theta_j - theta_i), matching the stated Kuramoto
    equation. Euler integration is not symplectic and large dt may be unstable.
    """

    num_oscillators: int
    natural_frequencies: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.num_oscillators < 1:
            raise ValueError("num_oscillators must be positive")
        if self.natural_frequencies is None:
            self.natural_frequencies = np.zeros(self.num_oscillators, dtype=np.float64)
        else:
            self.natural_frequencies = _finite_array(
                self.natural_frequencies, "natural_frequencies", 1
            )
            if self.natural_frequencies.shape != (self.num_oscillators,):
                raise ValueError("natural_frequencies shape must match num_oscillators")

    def forward(
        self,
        phases: object,
        metric_weights: object,
        K: float = 1.5,
        dt: float = 0.01,
    ) -> FloatArray:
        theta = _state(phases, "phases", self.num_oscillators)
        weights = _finite_array(metric_weights, "metric_weights", 3)
        expected = (theta.shape[0], self.num_oscillators, self.num_oscillators)
        if weights.shape != expected:
            raise ValueError(f"metric_weights must have shape {expected}; got {weights.shape}")
        if not np.isfinite(K) or K < 0:
            raise ValueError("K must be finite and non-negative")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # diff[b, i, j] = theta_j - theta_i
        phase_diffs = theta[:, None, :] - theta[:, :, None]
        coupling = np.sum(weights * np.sin(phase_diffs), axis=-1)
        d_theta = self.natural_frequencies[None, :] + (K / self.num_oscillators) * coupling
        return theta + d_theta * dt


class MasterCognitiveToposEngine:
    """Compose E -> I -> S -> L transport, energy evaluation, and phase dynamics.

    Each domain state has shape (batch, latent_dim). Transported states are
    fused additively with normalized residual weighting, preserving the caller's
    current state rather than replacing it wholesale. Energy is reported for
    the four supplied states, not the transported intermediates.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        *,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        natural_frequencies: object | None = None,
    ) -> None:
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        self.dim = latent_dim
        self.transport_e_to_i = EhresmannConnectionTransport(latent_dim, latent_dim)
        self.transport_i_to_s = EhresmannConnectionTransport(latent_dim, latent_dim)
        self.transport_s_to_l = EhresmannConnectionTransport(latent_dim, latent_dim)
        self.free_energy_engine = MasterFreeEnergyOptimizer(lambda_1, lambda_2)
        frequencies = None if natural_frequencies is None else np.asarray(
            natural_frequencies, dtype=np.float64
        )
        self.kuramoto_sync = KuramotoRiemannianSynchronizer(latent_dim, frequencies)

    def forward(
        self,
        outer_state: object,
        inner_state: object,
        self_state: object,
        lang_state: object,
        phases: object,
        metric_weights: object,
        *,
        K: float = 1.5,
        dt: float = 0.01,
    ) -> dict[str, FloatArray | float]:
        outer = _state(outer_state, "outer_state", self.dim)
        inner = _state(inner_state, "inner_state", self.dim)
        own = _state(self_state, "self_state", self.dim)
        language = _state(lang_state, "lang_state", self.dim)
        batch_shape = outer.shape[0]
        if any(value.shape[0] != batch_shape for value in (inner, own, language)):
            raise ValueError("all four domain states must have the same batch size")

        transported_ei = self.transport_e_to_i.forward(outer)
        transported_is = self.transport_i_to_s.forward(inner + transported_ei)
        transported_sl = self.transport_s_to_l.forward(own + transported_is)
        macro_state = language + transported_sl

        def energy(state: FloatArray) -> FloatArray:
            return np.sum(state * state, axis=-1)
        total_fe, coupling_penalty = self.free_energy_engine.forward(
            energy(outer), energy(inner), energy(own), energy(language)
        )
        synchronized = self.kuramoto_sync.forward(phases, metric_weights, K=K, dt=dt)
        if synchronized.shape[0] != batch_shape:
            raise ValueError("phase batch size must match domain-state batch size")
        return {
            "transported_macro_state": macro_state,
            "transported_outer_to_inner": transported_ei,
            "transported_inner_to_self": transported_is,
            "transported_self_to_language": transported_sl,
            "master_free_energy": total_fe,
            "coupling_constraint_penalty": coupling_penalty,
            "synchronized_phases": synchronized,
        }

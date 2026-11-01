"""Geometric latent-state model for Genesis's self subsystem.

This module is NumPy-native so the self model does not require PyTorch or a
pretrained model. It implements explicit, testable geometric operations:
a positive-definite narrative metric, agency/ownership coordinates,
Grassmannian projection, a residual-based consistency metric, and a
structure-preserving linear Hamiltonian flow.

The names describe mathematical constructions, not claims of consciousness,
qualia, biological equivalence, or measured subjective experience. In
particular, the consistency metric is a residual-induced metric, not Fisher
information unless a probability model and likelihood are supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "NarrativeAgencyBaseManifold",
    "AgencyOwnershipFiberBundle",
    "GrassmannianSubspaceModule",
    "SelfConsistencyInformationGeometry",
    "AutopoieticSymplecticFlow",
    "SelfLatentSpaceArchitecture",
]


def _matrix(value: Any, name: str) -> FloatArray:
    """Convert a finite rank-2 input to float64 without silently reshaping it."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array; got shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(slots=True)
class NarrativeAgencyBaseManifold:
    """Construct a positive-definite metric over narrative coordinates.

    The efficacy scalar is bounded before exponentiation to prevent overflow.
    The metric is exp(-efficacy) I + alpha x xᵀ, which is positive definite
    for finite x and strictly positive efficacy scale.
    """

    base_dim: int = 4
    coupling: float = 0.1

    def __post_init__(self) -> None:
        _positive_int(self.base_dim, "base_dim")
        if not np.isfinite(self.coupling) or self.coupling < 0:
            raise ValueError("coupling must be finite and non-negative")

    def forward(
        self, narrative_coords: Any, causal_efficacy: Any
    ) -> tuple[FloatArray, FloatArray]:
        coords = _matrix(narrative_coords, "narrative_coords")
        efficacy = np.asarray(causal_efficacy, dtype=np.float64)
        if coords.shape[1] != self.base_dim:
            raise ValueError(
                f"narrative_coords must have {self.base_dim} columns; got {coords.shape[1]}"
            )
        if efficacy.shape not in {(coords.shape[0],), (coords.shape[0], 1)}:
            raise ValueError("causal_efficacy must have shape (batch,) or (batch, 1)")
        if not np.isfinite(efficacy).all():
            raise ValueError("causal_efficacy must contain only finite values")
        weight = np.exp(-np.clip(efficacy.reshape(-1), -5.0, 5.0))
        identity = np.broadcast_to(
            np.eye(self.base_dim, dtype=np.float64),
            (coords.shape[0], self.base_dim, self.base_dim),
        )
        metric = weight[:, None, None] * identity
        metric = metric + self.coupling * np.einsum("bi,bj->bij", coords, coords)
        return metric.copy(), weight


@dataclass(slots=True)
class AgencyOwnershipFiberBundle:
    """Encode inputs into base coordinates plus data-dependent torus phases.

    Torus phases are derived from distinct input coordinates, not from adding
    fixed offsets to one scalar (which would create redundant phases).
    """

    input_dim: int = 8
    toroidal_factors: int = 4
    fiber_dim: int = 32

    def __post_init__(self) -> None:
        _positive_int(self.input_dim, "input_dim")
        _positive_int(self.toroidal_factors, "toroidal_factors")
        _positive_int(self.fiber_dim, "fiber_dim")

    @property
    def output_dim(self) -> int:
        return self.fiber_dim + 2 * self.toroidal_factors

    def forward(self, ego_inputs: Any) -> FloatArray:
        inputs = _matrix(ego_inputs, "ego_inputs")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"ego_inputs must have {self.input_dim} columns")
        # Deterministic, bounded projection: no hidden random initialization,
        # external model, or falsely implied training. The caller can replace
        # this projection with a learned encoder when a learning rule exists.
        source = np.tanh(inputs)
        indices = np.arange(1, self.input_dim + 1, dtype=np.float64)
        rows = np.arange(1, self.fiber_dim + 1, dtype=np.float64)[:, None]
        projection = np.sin(rows * indices[None, :] * np.sqrt(2.0))
        projection /= np.linalg.norm(projection, axis=1, keepdims=True)
        base = np.tanh(source @ projection.T)
        phase_indices = np.arange(self.toroidal_factors) % self.input_dim
        phases = np.pi * np.tanh(inputs[:, phase_indices])
        torus = np.concatenate((np.sin(phases), np.cos(phases)), axis=1)
        return np.concatenate((base, torus), axis=1)


@dataclass(slots=True)
class GrassmannianSubspaceModule:
    """Project states onto a fixed k-dimensional subspace of Rⁿ.

    The basis is constructed deterministically and orthonormalized by QR.
    A projector is symmetric and idempotent up to floating-point tolerance.
    """

    latent_dim: int
    subspace_dim: int

    def __post_init__(self) -> None:
        _positive_int(self.latent_dim, "latent_dim")
        _positive_int(self.subspace_dim, "subspace_dim")
        if self.subspace_dim > self.latent_dim:
            raise ValueError("subspace_dim cannot exceed latent_dim")
        rows = np.arange(1, self.latent_dim + 1, dtype=np.float64)[:, None]
        cols = np.arange(1, self.subspace_dim + 1, dtype=np.float64)[None, :]
        raw = np.sin(rows * cols * np.sqrt(2.0)) + np.cos(rows * cols * np.sqrt(3.0))
        basis, _ = np.linalg.qr(raw, mode="reduced")
        self._basis = basis
        self._projector = basis @ basis.T

    _basis: FloatArray = None  # type: ignore[assignment]
    _projector: FloatArray = None  # type: ignore[assignment]

    def forward(self, states: Any) -> tuple[FloatArray, FloatArray]:
        values = _matrix(states, "states")
        if values.shape[1] != self.latent_dim:
            raise ValueError(f"states must have {self.latent_dim} columns")
        return values @ self._projector, self._projector.copy()


@dataclass(slots=True)
class SelfConsistencyInformationGeometry:
    """Measure prediction residuals with an explicitly defined metric.

    The predictor is an identity map by default: without training data, a
    randomly initialized prediction network would add arbitrary behavior.
    An optional linear map may be supplied by the caller. The residual metric
    I + r rᵀ is positive definite, but is not called Fisher information.
    """

    latent_dim: int
    predictor: FloatArray | None = None

    def __post_init__(self) -> None:
        _positive_int(self.latent_dim, "latent_dim")
        if self.predictor is not None:
            self.predictor = _matrix(self.predictor, "predictor")
            if self.predictor.shape != (self.latent_dim, self.latent_dim):
                raise ValueError("predictor must have shape (latent_dim, latent_dim)")

    def forward(
        self, self_states: Any, observed_actions: Any
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        states = _matrix(self_states, "self_states")
        actions = _matrix(observed_actions, "observed_actions")
        if states.shape[1] != self.latent_dim:
            raise ValueError(f"self_states must have {self.latent_dim} columns")
        if actions.shape != states.shape:
            raise ValueError("observed_actions must match self_states shape")
        predicted = states if self.predictor is None else states @ self.predictor.T
        residual = predicted - actions
        metric = np.broadcast_to(
            np.eye(self.latent_dim, dtype=np.float64),
            (states.shape[0], self.latent_dim, self.latent_dim),
        ).copy()
        metric += np.einsum("bi,bj->bij", residual, residual)
        loss = np.einsum("bi,bi->b", residual, residual)
        return metric, loss, predicted


@dataclass(slots=True)
class AutopoieticSymplecticFlow:
    """Advance a canonical linear Hamiltonian system with a symplectic step.

    H(z) = 1/2 zᵀ K z, with symmetric positive-definite K. The implicit
    midpoint update is symplectic for this linear Hamiltonian system, unlike
    explicit Euler, which generally does not preserve the symplectic form.
    This is a mathematical dynamical model, not a biological self-preservation
    mechanism.
    """

    dim: int
    stiffness: float = 1.0

    def __post_init__(self) -> None:
        _positive_int(self.dim, "dim")
        if self.dim % 2:
            raise ValueError("symplectic phase-space dimension must be even")
        if not np.isfinite(self.stiffness) or self.stiffness <= 0:
            raise ValueError("stiffness must be finite and positive")
        half = self.dim // 2
        self._J = np.block([
            [np.zeros((half, half)), np.eye(half)],
            [-np.eye(half), np.zeros((half, half))],
        ])
        self._A = self._J * self.stiffness

    _J: FloatArray = None  # type: ignore[assignment]
    _A: FloatArray = None  # type: ignore[assignment]

    def forward(self, z: Any, dt: float = 0.01) -> FloatArray:
        state = _matrix(z, "z")
        if state.shape[1] != self.dim:
            raise ValueError(f"z must have {self.dim} columns")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        half_step = 0.5 * dt * self._A
        left = np.eye(self.dim) - half_step
        right = np.eye(self.dim) + half_step
        return np.linalg.solve(left, (right @ state.T)).T


class SelfLatentSpaceArchitecture:
    """Compose the geometric self-state modules into one explicit pipeline."""

    def __init__(
        self,
        base_dim: int = 4,
        input_dim: int = 8,
        latent_fiber_dim: int = 32,
        subspace_dim: int = 16,
        toroidal_factors: int = 4,
        flow_stiffness: float = 1.0,
    ) -> None:
        self.base_manifold = NarrativeAgencyBaseManifold(base_dim=base_dim)
        self.agency_bundle = AgencyOwnershipFiberBundle(
            input_dim=input_dim,
            toroidal_factors=toroidal_factors,
            fiber_dim=latent_fiber_dim,
        )
        total_dim = self.agency_bundle.output_dim
        self.grassmannian = GrassmannianSubspaceModule(total_dim, subspace_dim)
        self.consistency_geometry = SelfConsistencyInformationGeometry(total_dim)
        self.autopoietic_flow = AutopoieticSymplecticFlow(total_dim, flow_stiffness)

    def forward(
        self,
        narrative_coords: Any,
        causal_efficacy: Any,
        ego_inputs: Any,
        observed_actions: Any,
        dt: float = 0.01,
    ) -> dict[str, FloatArray]:
        metric, efficacy_weight = self.base_manifold.forward(
            narrative_coords, causal_efficacy
        )
        fiber_states = self.agency_bundle.forward(ego_inputs)
        if metric.shape[0] != fiber_states.shape[0]:
            raise ValueError("all inputs must have the same batch size")
        subspace_states, projector = self.grassmannian.forward(fiber_states)
        consistency_metric, coherence_loss, predicted_actions = (
            self.consistency_geometry.forward(fiber_states, observed_actions)
        )
        evolved = self.autopoietic_flow.forward(fiber_states, dt=dt)
        return {
            "narrative_base_metric": metric,
            "efficacy_weight": efficacy_weight,
            "agency_fiber_states": fiber_states,
            "grassmannian_subspace_states": subspace_states,
            "grassmannian_projection_operator": projector,
            "consistency_information_metric": consistency_metric,
            "coherence_loss": coherence_loss,
            "predicted_actions": predicted_actions,
            "autopoietic_evolved_states": evolved,
        }

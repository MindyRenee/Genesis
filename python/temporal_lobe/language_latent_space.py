"""Machine-native language latent geometry with six-dot mathematical symbols.

The six-dot cells are a reversible binary code, not standard literary Braille.
The latent vector combines parse structure with deterministic hashed lexical
features. A Sinkhorn coupling transports empirical cell mass to latent-feature
mass under an explicit Hamming prototype cost. This is an optimal-transport
summary, not a learned semantic map or a claim of isometry.

NumPy only; no LLM, Torch, pretrained weights, or network access.
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from copy import deepcopy
from typing import Any

import numpy as np

from parietal_lobe.binary_language import BitCell

_LATENT_DIM = 32
_CELL_COUNT = 64
_EPS = 1e-10
_ROLE_CHANNELS = (
    "AGENT", "PATIENT", "LOCATION", "SOURCE", "DESTINATION",
    "INSTRUMENT", "TIME", "CAUSE",
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def encode_text_cells(text: str) -> tuple[tuple[BitCell, ...], int]:
    """Losslessly encode UTF-8 text into six-dot cells and exact bit length."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    raw = text.encode("utf-8")
    bits = "".join(f"{byte:08b}" for byte in raw)
    bit_length = len(bits)
    bits += "0" * ((-len(bits)) % 6)
    return tuple(BitCell(int(bits[i:i + 6], 2)) for i in range(0, len(bits), 6)), bit_length


def decode_text_cells(cells: tuple[BitCell, ...], bit_length: int) -> str:
    """Decode six-dot cells back to their original UTF-8 text."""
    if isinstance(bit_length, bool) or not isinstance(bit_length, int):
        raise TypeError("bit_length must be an integer")
    if bit_length < 0 or bit_length > len(cells) * 6 or bit_length % 8:
        raise ValueError("bit_length must be byte-aligned and fit the cells")
    bits = "".join(cell.bits for cell in cells)[:bit_length]
    raw = bytes(int(bits[i:i + 8], 2) for i in range(0, bit_length, 8))
    return raw.decode("utf-8", errors="strict")


def _cell_histogram(cells: tuple[BitCell, ...]) -> np.ndarray:
    counts = np.zeros(_CELL_COUNT, dtype=np.float64)
    for cell in cells:
        counts[cell.value] += 1.0
    total = float(counts.sum())
    if total:
        counts /= total
    return counts


def _sinkhorn(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    *,
    regularization: float = 0.12,
    iterations: int = 80,
) -> tuple[np.ndarray, float]:
    """Entropy-regularized OT coupling for finite probability measures."""
    if source.ndim != 1 or target.ndim != 1 or cost.shape != (source.size, target.size):
        raise ValueError("OT marginal and cost dimensions do not agree")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError("OT costs must be finite and non-negative")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    if not np.isclose(source.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
        raise ValueError("OT marginals must each sum to one")
    kernel = np.maximum(np.exp(-cost / regularization), _EPS)
    u = np.ones_like(source)
    v = np.ones_like(target)
    for _ in range(iterations):
        u = source / np.maximum(kernel @ v, _EPS)
        v = target / np.maximum(kernel.T @ u, _EPS)
    plan = (u[:, None] * kernel) * v[None, :]
    mass = float(plan.sum())
    if not math.isfinite(mass) or mass <= 0:
        raise ArithmeticError("OT solver produced invalid transport mass")
    plan /= mass
    return plan, float(np.sum(plan * cost))


class LanguageLatentSpace:
    """Online language state coupled to parsed structure and six-dot cells.

    Geometry is empirical and interpretable: online feature covariance,
    transition residuals, categorical Fisher information over a softmax
    distribution, and entropy-regularized OT between observed cell mass and
    latent-feature mass. The Fisher matrix is for the categorical-logit model;
    it is not asserted to be invariant under the discrete transcoding.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._observations = 0
        self._mean = np.zeros(_LATENT_DIM, dtype=np.float64)
        self._m2 = np.zeros((_LATENT_DIM, _LATENT_DIM), dtype=np.float64)
        self._last_state = np.zeros(_LATENT_DIM, dtype=np.float64)
        self._last_snapshot: dict[str, Any] = {}

    @property
    def observations(self) -> int:
        with self._lock:
            return self._observations

    @property
    def dimension(self) -> int:
        return _LATENT_DIM

    def observe(self, text: str, comprehension: Any) -> dict[str, Any]:
        """Update geometry from one real utterance and its parse result."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cells, bit_length = encode_text_cells(text)
        features = self._features(text, comprehension)
        histogram = _cell_histogram(cells)
        target = np.abs(features)
        target += _EPS
        target /= target.sum()
        source_ids = np.flatnonzero(histogram)
        source_mass = histogram[source_ids]
        target_ids = np.arange(_LATENT_DIM)
        target_mass = target[target_ids]
        costs = np.empty((source_ids.size, _LATENT_DIM), dtype=np.float64)
        for row, cell_value in enumerate(source_ids):
            costs[row] = [
                (int(cell_value ^ int(proto)).bit_count() / 6.0)
                for proto in target_ids
            ]
        plan, transport_cost = _sinkhorn(source_mass, target_mass, costs)

        logits = features - float(np.max(features))
        probabilities = np.exp(logits)
        probabilities /= max(float(probabilities.sum()), _EPS)
        fisher = np.diag(probabilities) - np.outer(probabilities, probabilities)
        regularized = fisher + 1e-8 * np.eye(_LATENT_DIM, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(regularized)
        if sign <= 0 or not math.isfinite(float(logdet)):
            raise ArithmeticError("regularized categorical Fisher matrix is not positive definite")
        fisher_volume = float(math.exp(0.5 * float(logdet)))

        with self._lock:
            previous = self._last_state.copy()
            self._observations += 1
            delta = features - self._mean
            self._mean += delta / self._observations
            centered = features - self._mean
            self._m2 += np.outer(delta, centered)
            transition = (
                float(np.linalg.norm(features - previous))
                if self._observations > 1
                else 0.0
            )
            covariance = (
                self._m2 / (self._observations - 1)
                if self._observations > 1
                else np.zeros_like(self._m2)
            )
            self._last_state = features.copy()
            snapshot = {
                "version": 1,
                "observations": self._observations,
                "latent_state": features.tolist(),
                "cell_count": len(cells),
                "bit_length": bit_length,
                "cell_distribution": histogram.tolist(),
                "transport_cost": transport_cost,
                "transport_plan_shape": [int(plan.shape[0]), int(plan.shape[1])],
                "transition_residual": transition,
                "feature_covariance_diagonal": np.diag(covariance).tolist(),
                "fisher_information": fisher.tolist(),
                "regularized_fisher_volume": fisher_volume,
                "parse_confidence": self._parse_number(comprehension, "confidence", 0.0),
                "parse_coverage": self._parse_number(comprehension, "parse_coverage", 0.0),
            }
            self._last_snapshot = deepcopy(snapshot)
            return deepcopy(snapshot)

    @staticmethod
    def _parse_number(result: Any, name: str, default: float) -> float:
        value = getattr(result, name, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    @staticmethod
    def _features(text: str, result: Any) -> np.ndarray:
        tokens = _TOKEN_RE.findall(text.casefold())
        vector = np.zeros(_LATENT_DIM, dtype=np.float64)
        vector[:8] = (
            LanguageLatentSpace._parse_number(result, "confidence", 0.0),
            LanguageLatentSpace._parse_number(result, "parse_coverage", 0.0),
            float(bool(getattr(result, "is_negated", False))),
            float(bool(getattr(result, "is_question", False))),
            float(bool(getattr(result, "is_command", False))),
            min(1.0, len(tokens) / 64.0),
            min(1.0, len(getattr(result, "propositions", ()) or ()) / 8.0),
            min(1.0, len(getattr(result, "unresolved_tokens", ()) or ()) / max(1, len(tokens))),
        )
        speech_act = getattr(getattr(result, "speech_act", None), "name", "")
        if not speech_act:
            speech_act = str(getattr(result, "speech_act", ""))
        digest = hashlib.blake2b(speech_act.casefold().encode("utf-8"), digest_size=8).digest()
        vector[8 + digest[0] % 8] += 1.0

        propositions = getattr(result, "propositions", ()) or ()
        for proposition in propositions:
            roles = getattr(proposition, "roles", {}) or {}
            for role in roles:
                role_name = getattr(role, "name", str(role)).upper()
                if role_name in _ROLE_CHANNELS:
                    vector[8 + _ROLE_CHANNELS.index(role_name)] += 1.0
            for slot in ("subject", "predicate", "object"):
                term = str(getattr(proposition, slot, "") or "").casefold().strip()
                if term:
                    token_digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                    vector[16 + token_digest[0] % 16] += -1.0 if token_digest[1] & 1 else 1.0
        for token in tokens:
            token_digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[16 + token_digest[0] % 16] += -1.0 if token_digest[1] & 1 else 1.0
        vector[16:] /= max(1.0, math.sqrt(len(tokens) + 3 * len(propositions)))
        if not np.all(np.isfinite(vector)):
            raise ValueError("language latent features must be finite")
        return vector

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._last_snapshot)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "observations": self._observations,
                "mean": self._mean.tolist(),
                "m2": self._m2.tolist(),
                "last_state": self._last_state.tolist(),
                "last_snapshot": deepcopy(self._last_snapshot),
            }

    def restore(self, data: dict[str, Any]) -> None:
        """Validate the entire checkpoint before mutating the live space."""
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported language latent checkpoint")
        observations = data.get("observations")
        if isinstance(observations, bool) or not isinstance(observations, int) or observations < 0:
            raise ValueError("invalid language latent observation count")
        mean = np.asarray(data.get("mean"), dtype=np.float64)
        m2 = np.asarray(data.get("m2"), dtype=np.float64)
        last_state = np.asarray(data.get("last_state"), dtype=np.float64)
        if mean.shape != (_LATENT_DIM,) or last_state.shape != (_LATENT_DIM,):
            raise ValueError("invalid language latent vector shape")
        if m2.shape != (_LATENT_DIM, _LATENT_DIM):
            raise ValueError("invalid language latent covariance shape")
        if not all(np.all(np.isfinite(array)) for array in (mean, m2, last_state)):
            raise ValueError("language latent checkpoint contains non-finite values")
        last_snapshot = data.get("last_snapshot", {})
        if not isinstance(last_snapshot, dict):
            raise ValueError("invalid language latent snapshot")
        with self._lock:
            self._observations = observations
            self._mean = mean.copy()
            self._m2 = m2.copy()
            self._last_state = last_state.copy()
            self._last_snapshot = deepcopy(last_snapshot)

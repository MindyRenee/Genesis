"""Machine-native language latent geometry using reversible 7-bit ASCII notation.

Text is represented as ASCII character codes (0..127); non-ASCII input is
rejected rather than silently transliterated or UTF-8 encoded. The latent
vector combines parsed structure with deterministic hashed lexical features.
A Sinkhorn coupling compares the observed ASCII-code distribution with an
explicit latent-feature codebook using normalized 7-bit Hamming cost. This is
an optimal-transport summary, not a learned semantic map or isometry.

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

_LATENT_DIM = 32
_ASCII_COUNT = 128
_EPS = 1e-10
_ROLE_CHANNELS = (
    "AGENT", "PATIENT", "LOCATION", "SOURCE", "DESTINATION",
    "INSTRUMENT", "TIME", "CAUSE",
)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def encode_ascii(text: str) -> tuple[tuple[int, ...], int]:
    """Encode arbitrary Unicode text as reversible 7-bit ASCII notation.

    Printable ASCII is preserved. A literal backslash is doubled; each
    non-ASCII code point becomes the ASCII sequence \\u{HEX};.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    escaped = []
    for character in text:
        codepoint = ord(character)
        if character == "\\\\":
            escaped.append("\\\\\\\\")
        elif codepoint < 128:
            escaped.append(character)
        else:
            escaped.append("\\\\u{" + format(codepoint, "X") + "};")
    raw = "".join(escaped).encode("ascii", errors="strict")
    return tuple(raw), len(raw) * 7


def decode_ascii(codes: tuple[int, ...], bit_length: int) -> str:
    """Decode reversible ASCII notation after validating its exact bit length."""
    if isinstance(bit_length, bool) or not isinstance(bit_length, int):
        raise TypeError("bit_length must be an integer")
    if bit_length < 0 or bit_length % 7 or bit_length != len(codes) * 7:
        raise ValueError("bit_length must equal the number of ASCII codes times 7")
    if any(isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 128
           for code in codes):
        raise ValueError("ASCII codes must be integers in range 0..127")
    encoded = bytes(codes).decode("ascii", errors="strict")
    decoded = []
    index = 0
    while index < len(encoded):
        if encoded.startswith("\\\\\\\\", index):
            decoded.append("\\\\")
            index += 2
        elif encoded.startswith("\\\\u{", index):
            end = encoded.find("};", index + 3)
            if end < 0:
                raise ValueError("malformed Unicode escape in ASCII text")
            try:
                codepoint = int(encoded[index + 3:end], 16)
                if not 0 <= codepoint <= 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                    raise ValueError
                decoded.append(chr(codepoint))
            except ValueError as exc:
                raise ValueError("invalid Unicode code point in ASCII escape") from exc
            index = end + 2
        else:
            decoded.append(encoded[index])
            index += 1
    return "".join(decoded)

def _ascii_histogram(codes: tuple[int, ...]) -> np.ndarray:
    counts = np.zeros(_ASCII_COUNT, dtype=np.float64)
    for code in codes:
        counts[code] += 1.0
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
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0)
        or np.any(target < 0)
    ):
        raise ValueError("OT marginals must be finite and non-negative")
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
    row_error = float(np.max(np.abs(plan.sum(axis=1) - source)))
    column_error = float(np.max(np.abs(plan.sum(axis=0) - target)))
    if max(row_error, column_error) > 5e-3:
        raise ArithmeticError(
            f"OT marginals did not converge (row={row_error:.3g}, column={column_error:.3g})"
        )
    return plan, float(np.sum(plan * cost))


class LanguageLatentSpace:
    """Online language state coupled to parsed structure and ASCII character codes.

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

    def observe(
        self, text: str, comprehension: Any, *, direction: str = "inbound"
    ) -> dict[str, Any]:
        """Update geometry from one parsed utterance.

        Direction separates received language from Genesis's generated output
        without treating either direction as evidence of semantic correctness.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if direction not in {"inbound", "outbound", "internal"}:
            raise ValueError("direction must be inbound, outbound, or internal")
        codes, bit_length = encode_ascii(text)
        features = self._features(text, comprehension)
        histogram = _ascii_histogram(codes if codes else (0,))
        # Each signed latent coordinate maps to a pair of ASCII-indexed
        # prototypes: even = negative, odd = positive. Remaining ASCII bins
        # retain only epsilon mass. This codebook is not semantic grounding.
        target = np.full(_ASCII_COUNT, _EPS, dtype=np.float64)
        for index, value in enumerate(features):
            target[2 * index] += max(0.0, -float(value))
            target[2 * index + 1] += max(0.0, float(value))
        target /= target.sum()
        source_ids = np.flatnonzero(histogram)
        source_mass = histogram[source_ids]
        target_ids = np.arange(_ASCII_COUNT)
        costs = np.empty((source_ids.size, _ASCII_COUNT), dtype=np.float64)
        for row, cell_value in enumerate(source_ids):
            costs[row] = [
                int(cell_value ^ int(proto)).bit_count() / 7.0
                for proto in target_ids
            ]
        plan, transport_cost = _sinkhorn(source_mass, target, costs)
        row_marginal_error = float(np.max(np.abs(plan.sum(axis=1) - source_mass)))
        column_marginal_error = float(np.max(np.abs(plan.sum(axis=0) - target)))

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
                "ascii_character_count": len(codes),
                "bit_length": bit_length,
                "direction": direction,
                "ascii_distribution": histogram.tolist(),
                "transport_cost": transport_cost,
                "transport_plan_shape": [int(plan.shape[0]), int(plan.shape[1])],
                "transport_source_ascii_codes": source_ids.tolist(),
                "transport_target_distribution": target.tolist(),
                "transport_plan": plan.tolist(),
                "transport_row_marginal_error": row_marginal_error,
                "transport_column_marginal_error": column_marginal_error,
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
            min(
                1.0,
                (
                    sum(bool(getattr(item, "fully_grounded", False)) for item in result.grounded)
                    / len(result.grounded)
                )
                if getattr(result, "grounded", None)
                else len(getattr(result, "propositions", ()) or ()) / 8.0,
            ),
            min(1.0, len(getattr(result, "unresolved_tokens", ()) or ()) / max(1, len(tokens))),
        )
        speech_act = getattr(getattr(result, "speech_act", None), "name", "")
        if not speech_act:
            speech_act = str(getattr(result, "speech_act", ""))
        digest = hashlib.blake2b(speech_act.casefold().encode("ascii", errors="backslashreplace"), digest_size=8).digest()
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
                    token_digest = hashlib.blake2b(term.encode("ascii", errors="backslashreplace"), digest_size=8).digest()
                    vector[16 + token_digest[0] % 16] += -1.0 if token_digest[1] & 1 else 1.0
        grounded = getattr(result, "grounded", ()) or ()
        for proposition in grounded:
            arguments = [
                getattr(proposition, "subject", None),
                getattr(proposition, "object", None),
                *getattr(proposition, "roles", {}).values(),
            ]
            for argument in arguments:
                concept_id = getattr(argument, "concept_id", None)
                if not isinstance(concept_id, str) or not concept_id:
                    continue
                concept_digest = hashlib.blake2b(
                    ("concept:" + concept_id).encode("ascii", errors="backslashreplace"), digest_size=8
                ).digest()
                vector[16 + concept_digest[0] % 16] += (
                    -1.0 if concept_digest[1] & 1 else 1.0
                )
        for token in tokens:
            token_digest = hashlib.blake2b(token.encode("ascii", errors="backslashreplace"), digest_size=8).digest()
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

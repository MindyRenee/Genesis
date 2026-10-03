"""Somatic snapshots — the body a memory was encoded in.

State-dependent recall: memories encoded in a bodily state similar to
the current one are easier to retrieve (Eich, 1980; Bower, 1981).
Every encoded memory is stamped with the body state at encoding
(arousal, valence, tone, plasticity, homeostatic balance, mirroring
``ProtoSelfState``); retrieval re-ranks by congruence between the
current snapshot and the encoding snapshot.

This is machine-like, not lifespan-like: the snapshot is a system-state
vector (the same idea as context-reinstatement in any memory system),
not an age. There is no maturity, no regime, no decline curve here.
Plasticity stays available for life — it is gated by present surprise,
sleep, and neuromodulation, the way biology actually gates it, which is
also what open-ended (ASI/AGI) learning requires. Nothing here is
uttered; these are numeric modulators, so the no-hardcoding rule
(AGENTS.md) holds — parameters in, composed language out.

References:
- Eich, E. (1980). The cue-dependent nature of state-dependent
  retrieval. Memory & Cognition.
- Bower, G. H. (1981). Mood and memory. American Psychologist.
- Damasio, A. (1999). The Feeling of What Happens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SOMATIC_AXES",
    "SomaticSnapshot",
    "somatic_congruence",
]

# The dimensions of a body-state snapshot.
SOMATIC_AXES: tuple[str, ...] = (
    "arousal",
    "valence",
    "tone",
    "plasticity",
    "balance",
)


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.5
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class SomaticSnapshot:
    """Body state at the moment a memory was encoded (or is recalled).

    Five axes in [0, 1] except valence in [-1, 1], mirroring
    ``ProtoSelfState``.
    """

    arousal: float = 0.5
    valence: float = 0.0
    tone: float = 0.5
    plasticity: float = 0.5
    balance: float = 0.8

    def to_dict(self) -> dict[str, float]:
        """Flatten to a JSON-safe dict for ``MemoryRecord.somatic``."""
        return {
            "arousal": _clamp01(self.arousal),
            "valence": max(-1.0, min(1.0, self.valence)),
            "tone": _clamp01(self.tone),
            "plasticity": _clamp01(self.plasticity),
            "balance": _clamp01(self.balance),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SomaticSnapshot | None:
        """Rebuild from a flat dict; None in → None out.

        Unknown keys (e.g. a legacy ``age_episodes``) are ignored so
        snapshots written by earlier revisions still load.
        """
        if not data:
            return None
        try:
            return cls(
                arousal=float(data.get("arousal", 0.5)),
                valence=float(data.get("valence", 0.0)),
                tone=float(data.get("tone", 0.5)),
                plasticity=float(data.get("plasticity", 0.5)),
                balance=float(data.get("balance", 0.8)),
            )
        except (TypeError, ValueError):
            return None


def somatic_congruence(
    current: SomaticSnapshot | dict[str, Any] | None,
    encoded: SomaticSnapshot | dict[str, Any] | None,
) -> float:
    """State-dependent retrieval affinity in [0, 1].

    1.0 = same bodily state (free recall), → 0 as states diverge.
    ``1 / (1 + d)`` over the five normalised axes; valence is
    rescaled from [-1, 1] to [0, 1] so all axes share a unit cube.
    Missing data yields 0.5 (neutral): no evidence either way, never
    a penalty.
    """
    if current is None or encoded is None:
        return 0.5
    if isinstance(current, dict):
        current = SomaticSnapshot.from_dict(current)
    if isinstance(encoded, dict):
        encoded = SomaticSnapshot.from_dict(encoded)
    if current is None or encoded is None:
        return 0.5

    def norm_v(v: float) -> float:
        return (max(-1.0, min(1.0, v)) + 1.0) / 2.0

    dist_sq = (
        (current.arousal - encoded.arousal) ** 2
        + (norm_v(current.valence) - norm_v(encoded.valence)) ** 2
        + (current.tone - encoded.tone) ** 2
        + (current.plasticity - encoded.plasticity) ** 2
        + (current.balance - encoded.balance) ** 2
    )
    return 1.0 / (1.0 + math.sqrt(dist_sq))

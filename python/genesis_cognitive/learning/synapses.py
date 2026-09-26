"""Experience-dependent synaptic efficacy for associative learning.

This is the plastic substrate used by timing-dependent learning.
It is deliberately separate from Genesis's semantic concept graph and
from the semantic embedding space:

* ConceptNetwork edges encode explicit semantic relationships.
* Embeddings encode representational similarity.
* Synaptic efficacy encodes learned strength between active units.

STDP changes synaptic efficacy; it does not rewrite semantic facts or
move semantic representations.
"""

from __future__ import annotations

import threading

__all__ = ["SynapticStore"]


class SynapticStore:
    """Sparse directed store of experience-dependent connection strength.

    A connection is identified by its pre/post active-unit labels, but
    its weight has no semantic meaning. Connections are created lazily
    when plasticity first acts on a pair. Weights are bounded to the
    biologically useful computational range [0, 1].
    """

    MIN_WEIGHT = 0.0
    MAX_WEIGHT = 1.0

    def __init__(self) -> None:
        self._weights: dict[tuple[str, str], float] = {}
        self._lock = threading.RLock()

    def get_weight(self, pre: str, post: str) -> float:
        """Return synaptic efficacy, or zero when no learned connection exists."""
        with self._lock:
            return self._weights.get((pre, post), 0.0)

    def set_weight(self, pre: str, post: str, weight: float) -> float:
        """Set and return a bounded synaptic efficacy."""
        bounded = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, float(weight)))
        with self._lock:
            self._weights[(pre, post)] = bounded
        return bounded

    def apply_delta(self, pre: str, post: str, delta: float) -> float:
        """Apply a plasticity delta to a connection and return its new weight."""
        with self._lock:
            old = self._weights.get((pre, post), 0.0)
            new = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, old + float(delta)))
            self._weights[(pre, post)] = new
            return new

    def propagate(self, seeds: list[str], amount: float = 1.0) -> dict[str, float]:
        """Propagate activation through expressed learned connections.

        This is the downstream read path for synaptic efficacy. It does
        not create or alter semantic graph edges or embeddings. Each
        directly connected post-unit receives seed activation multiplied
        by its learned efficacy.
        """
        if amount <= 0.0 or not seeds:
            return {}
        seed_set = set(seeds)
        propagated: dict[str, float] = {}
        with self._lock:
            weights = dict(self._weights)
        for (pre, post), weight in weights.items():
            if pre in seed_set and weight > 0.0:
                contribution = float(amount) * weight
                propagated[post] = max(propagated.get(post, 0.0), contribution)
        return propagated

    def has_connection(self, pre: str, post: str) -> bool:
        """Return whether this directed plastic connection has been expressed."""
        with self._lock:
            return (pre, post) in self._weights

    def connections(self) -> dict[tuple[str, str], float]:
        """Return a snapshot of all expressed plastic connections."""
        with self._lock:
            return dict(self._weights)

    def save_state(self) -> dict[str, float]:
        """Serialize expressed synaptic efficacies."""
        with self._lock:
            return {f"{pre}\t{post}": weight for (pre, post), weight in self._weights.items()}

    def load_state(self, state: dict[str, float]) -> None:
        """Restore expressed synaptic efficacies from persisted state."""
        restored: dict[tuple[str, str], float] = {}
        for key, weight in state.items():
            pre, separator, post = key.partition("\t")
            if not separator or not pre or not post:
                continue
            restored[(pre, post)] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, float(weight)))
        with self._lock:
            self._weights = restored

"""Spike-Timing-Dependent Plasticity (STDP).

STDP is a timing-dependent synaptic learning rule. The plastic target
is synaptic efficacy: the strength of a directed connection from an
active pre-unit to an active post-unit. It is intentionally separate
from Genesis's semantic graph and embedding representation.

For Δt = t_post − t_pre:
    Δt > 0 → LTP (pre before post)
    Δt < 0 → LTD (post before pre)

The timing rule is followed by a three-factor neuromodulatory gate.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .synapses import SynapticStore

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

__all__ = ["STDP", "SpikeEvent"]


@dataclass(slots=True)
class SpikeEvent:
    """A logical activation event used as a pre/post timing signal."""

    concept: str
    time_ms: float


class STDP:
    """Timing-dependent plasticity over Genesis's synaptic substrate.

    Concepts are the current computational units that emit activation
    events. STDP does not alter their semantic embeddings or typed graph
    relationships. It changes only the independent directed synaptic
    efficacy stored by ``SynapticStore``.
    """

    TAU_PLUS: float = 20.0
    TAU_MINUS: float = 20.0
    A_PLUS: float = 0.01
    A_MINUS: float = 0.012
    WINDOW_MS: float = 40.0
    LEARNING_RATE: float = 0.002

    def __init__(
        self,
        synapses: SynapticStore | Any,
        network: ConceptNetwork,
        *,
        tau_plus: float = TAU_PLUS,
        tau_minus: float = TAU_MINUS,
        a_plus: float = A_PLUS,
        a_minus: float = A_MINUS,
        learning_rate: float = LEARNING_RATE,
        window_ms: float = WINDOW_MS,
    ) -> None:
        # The legacy cognition/autonomous wiring still passes an
        # EmbeddingStore. Keep that call site working while ensuring the
        # plastic target is now always a dedicated SynapticStore. New
        # callers should pass the store explicitly so it can be shared
        # and persisted by the owning cognitive system.
        self.synapses = synapses if isinstance(synapses, SynapticStore) else SynapticStore()
        self.network = network
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.learning_rate = learning_rate
        self.window_ms = window_ms

        self._spike_history: dict[str, deque[float]] = {}
        self._max_history = 50
        self._pending: list[tuple[str, str, float, float]] = []
        self._lock = threading.RLock()
        self._modulator = 1.0

        self.total_potentiated = 0
        self.total_depressed = 0
        self.total_gated = 0

    def set_modulator(self, modulator: float) -> None:
        """Set the third-factor neuromodulatory signal."""
        with self._lock:
            self._modulator = float(modulator)

    def get_modulator(self) -> float:
        """Return the current third-factor signal."""
        with self._lock:
            return self._modulator

    def record_spike(self, concept: str, time_ms: float) -> None:
        """Record an activation and form STDP pairs with recent activations.

        Implements bidirectional all-to-all STDP (Bi & Poo, 1998; Markram
        et al., 1997): for each prior spike of another unit at
        ``other_time``, the ordered pair (other → concept) potentiates
        with Δt = time_ms − other_time ≥ 0, while the reverse ordered
        pair (concept → other) depresses with Δt = other_time − time_ms
        ≤ 0. Forward time therefore strengthens the causal direction
        (pre-before-post, LTP) and weakens the acausal direction
        (post-before-pre, LTD), as required for temporally asymmetric
        Hebbian plasticity.
        """
        concept = self.network._resolve(concept) or concept
        pairs: list[tuple[str, str, float, float]] = []

        with self._lock:
            for other, other_times in self._spike_history.items():
                if other == concept:
                    continue
                for other_time in reversed(other_times):
                    delta_t = time_ms - other_time
                    if abs(delta_t) > self.window_ms:
                        continue
                    # Forward direction: other (pre) → concept (post).
                    weight_change = self._stdp_weight(delta_t)
                    if abs(weight_change) > 1e-12:
                        pairs.append((other, concept, delta_t, weight_change))
                    # Reverse direction: concept (pre) → other (post) with
                    # negated timing. At Δt_forward > 0 this yields LTD on
                    # the acausal edge; at simultaneity both are ~0.
                    rev_change = self._stdp_weight(-delta_t)
                    if abs(rev_change) > 1e-12:
                        pairs.append((concept, other, -delta_t, rev_change))

            self._spike_history.setdefault(
                concept, deque(maxlen=self._max_history)
            ).append(float(time_ms))
            self._pending.extend(pairs)

        if pairs:
            self._apply_pending()

    def record_spikes(self, concepts: list[str], time_ms: float) -> None:
        """Record simultaneous activations for multiple units."""
        for concept in concepts:
            self.record_spike(concept, time_ms)

    def apply_updates(self) -> dict[str, int]:
        """Apply all buffered timing-dependent synaptic updates."""
        return self._apply_pending()

    def compute_stdp_window(self, delta_t: float) -> float:
        """Return the signed timing-dependent plasticity magnitude."""
        return self._stdp_weight(delta_t)

    def get_synaptic_weight(self, pre: str, post: str) -> float:
        """Return learned efficacy of the directed pre→post connection."""
        pre = self.network._resolve(pre) or pre
        post = self.network._resolve(post) or post
        return self.synapses.get_weight(pre, post)

    def get_statistics(self) -> dict[str, int | float]:
        """Return STDP activity and plasticity statistics."""
        with self._lock:
            total_spikes = sum(len(times) for times in self._spike_history.values())
            return {
                "total_potentiated": self.total_potentiated,
                "total_depressed": self.total_depressed,
                "total_gated": self.total_gated,
                "modulator": self._modulator,
                "tracked_concepts": len(self._spike_history),
                "total_spikes_buffered": total_spikes,
                "pending_updates": len(self._pending),
                "expressed_synapses": len(self.synapses.connections()),
            }

    def clear_history(self) -> None:
        """Clear transient spike timing history and pending updates."""
        with self._lock:
            self._spike_history.clear()
            self._pending.clear()

    def save_state(self) -> dict[str, object]:
        """Serialize persistent synaptic efficacy and timing parameters."""
        with self._lock:
            return {
                "synapses": self.synapses.save_state(),
                "modulator": self._modulator,
            }

    def load_state(self, state: dict[str, object]) -> None:
        """Restore persistent synaptic efficacy."""
        synapses = state.get("synapses")
        if isinstance(synapses, dict):
            self.synapses.load_state(synapses)
        modulator = state.get("modulator")
        if isinstance(modulator, (int, float)):
            self.set_modulator(float(modulator))

    def renormalize(self) -> None:
        """Compatibility no-op; synaptic efficacy is already bounded to [0, 1]."""

    def _stdp_weight(self, delta_t: float) -> float:
        # Bi & Poo (1998): Δw = A+·exp(−Δt/τ+) for Δt > 0 (LTP),
        # Δw = −A−·exp(Δt/τ−) for Δt < 0 (LTD). At exact simultaneity
        # there is no causal order, so no change (avoids a full-LTP
        # artifact when simultaneous spikes are recorded sequentially).
        if delta_t > 0:
            return self.a_plus * float(np.exp(-delta_t / self.tau_plus))
        if delta_t < 0:
            return -self.a_minus * float(np.exp(delta_t / self.tau_minus))
        return 0.0

    def _apply_pending(self) -> dict[str, int]:
        with self._lock:
            pending = self._pending
            self._pending = []
            modulator = self._modulator

        potentiated = 0
        depressed = 0
        gated = 0
        for pre, post, _delta_t, rule_delta in pending:
            delta = self.learning_rate * rule_delta * modulator
            if abs(delta) < 1e-12:
                gated += 1
                continue

            old = self.synapses.get_weight(pre, post)
            new = self.synapses.apply_delta(pre, post, delta)
            if new > old + 1e-12:
                potentiated += 1
            elif new < old - 1e-12:
                depressed += 1

        self.total_potentiated += potentiated
        self.total_depressed += depressed
        with self._lock:
            self.total_gated += gated
        return {"potentiated": potentiated, "depressed": depressed}

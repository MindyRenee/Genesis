"""Spike-Timing-Dependent Plasticity (STDP).

STDP is a biologically precise form of Hebbian learning where the
*timing* of pre- and post-synaptic spikes determines the direction
and magnitude of plasticity — not just co-occurrence. This refines
the basic Hebbian rule ("cells that fire together wire together")
into a temporally asymmetric rule:

    "Cells that fire together wire together — *if the pre-synaptic
     cell fires just before the post-synaptic cell.*"

# The STDP rule (Bi & Poo, 1998)

Let Δt = t_post − t_pre be the relative timing of the spikes.

- **Δt > 0** (pre fires *before* post): **Long-Term Potentiation (LTP)**.
  The pre-synaptic neuron helped cause the post-synaptic spike, so
  the connection is strengthened. The potentiation decays exponentially
  with Δt:

      W(Δt) = A₊ · exp(−Δt / τ₊)      for Δt > 0

- **Δt < 0** (post fires *before* pre): **Long-Term Depression (LTD)**.
  The pre-synaptic spike arrived too late to contribute to the
  post-synaptic firing, so the connection is weakened:

      W(Δt) = −A₋ · exp(Δt / τ₋)      for Δt < 0

The learning window is asymmetric: LTP and LTD have separate
amplitudes (A₊, A₋) and time constants (τ₊, τ₋). In biological
synapses, A₋ is typically slightly larger than A₊, giving a net
depressive bias that prevents runaway potentiation (Bi & Poo, 1998;
Song et al., 2000). The time constants are ~20 ms, matching the
window over which NMDA receptor activation integrates pre/post
coincidence (Markram et al., 1997).

# Application to concept embeddings

Genesis does not have literal millisecond spike trains, but concept
*activation* events serve as the analog of spikes. When concept A
becomes active shortly before concept B (a causal or predictive
relationship), STDP potentiates the A→B connection — A's embedding
moves toward B's. When B activates before A, the connection is
depressed — the embeddings move apart. This captures *temporal
causality* that plain co-occurrence Hebbian learning cannot: it
matters not just *that* two concepts co-occur, but *in what order*.

This module augments (does not replace) the basic Hebbian rule in
``plasticity.py``. The basic rule handles symmetric co-occurrence;
STDP handles asymmetric temporal causality.

References:
    - Bi & Poo (1998): Synaptic modifications in cultured hippocampal
      neurons dependent on the timing of pre- and post-synaptic firing.
      Annual Review of Neuroscience, 21(1), 473-497.
    - Markram et al. (1997): Regulation of synaptic efficacy by
      coincidence of postsynaptic APs and EPSPs. Science, 275, 213-215.
    - Song et al. (2000): Competitive Hebbian learning through
      spike-timing-dependent synaptic plasticity. Nature Neuroscience,
      3, 919-926.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, EmbeddingStore

__all__ = ["STDP", "SpikeEvent"]


@dataclass(slots=True)
class SpikeEvent:
    """A single spike (activation event) for a concept.

    Attributes:
        concept: The concept that spiked (became active).
        time_ms: The spike time in milliseconds. This is a logical
            clock — it need not be wall-clock time. It just needs to
            be monotonically increasing so that relative timing
            (Δt = t_post − t_pre) is meaningful.
    """

    concept: str
    time_ms: float


class STDP:
    """Spike-Timing-Dependent Plasticity for concept embeddings.

    Records spike (activation) times for concepts and applies
    timing-dependent potentiation or depression to the embedding
    vectors. When concept A spikes before concept B, the A→B
    connection is potentiated (embeddings move closer). When B spikes
    before A, it is depressed (embeddings move apart).

    The learning window is asymmetric and exponentially decaying,
    matching the biological STDP curve (Bi & Poo, 1998). A depressive
    bias (A₋ > A₊) prevents runaway potentiation.

    Thread-safe. All updates are protected by a lock.

    Attributes:
        tau_plus: LTP time constant (ms). Default 20 ms.
        tau_minus: LTD time constant (ms). Default 20 ms.
        a_plus: LTP amplitude. Default 0.01.
        a_minus: LTD amplitude. Default 0.012 (slightly larger for
            depressive bias).
        learning_rate: Base learning rate for embedding updates.
        window_ms: Maximum |Δt| to consider (ms). Pairs outside this
            window produce negligible plasticity and are skipped.
    """

    # Default STDP parameters (Bi & Poo, 1998; Markram et al., 1997)
    TAU_PLUS: float = 20.0  # ms — LTP decay constant
    TAU_MINUS: float = 20.0  # ms — LTD decay constant
    A_PLUS: float = 0.01  # LTP amplitude
    A_MINUS: float = 0.012  # LTD amplitude (slightly larger → depressive bias)
    WINDOW_MS: float = 40.0  # only consider pairs within this |Δt|
    LEARNING_RATE: float = 0.002  # base embedding update rate

    def __init__(
        self,
        embeddings: EmbeddingStore,
        network: ConceptNetwork,
        *,
        tau_plus: float = TAU_PLUS,
        tau_minus: float = TAU_MINUS,
        a_plus: float = A_PLUS,
        a_minus: float = A_MINUS,
        learning_rate: float = LEARNING_RATE,
        window_ms: float = WINDOW_MS,
    ) -> None:
        """Initialize STDP learner with embedding store, network, and timing constants."""
        self.embeddings = embeddings
        self.network = network

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.learning_rate = learning_rate
        self.window_ms = window_ms

        # Spike history per concept — deque of spike times (ms).
        # Bounded to keep memory and computation tractable.
        self._spike_history: dict[str, deque[float]] = {}
        self._max_history: int = 50

        # Pending STDP updates: (pre, post, delta_t, weight_change)
        self._pending: list[tuple[str, str, float, float]] = []
        self._lock = threading.Lock()

        # Three-factor modulation: the current neuromodulatory signal
        # (dopamine / TD reward prediction error). When non-zero, STDP
        # updates are gated by this signal — only plasticity that is
        # "rewarded" (positive modulator) or "punished" (negative
        # modulator) is applied. This implements three-factor STDP
        # (Pawlak et al., 2010; Frémaux & Gerstner, 2016):
        #
        #   Δw = STDP(pre, post) × modulator
        #
        # The modulator is set from the TD learner's RPE via
        # ``set_modulator()``. A value of 1.0 means no gating (standard
        # two-factor STDP). Values > 1 amplify; 0–1 attenuate; < 0
        # reverses (LTP becomes LTD and vice versa).
        self._modulator: float = 1.0

        # Statistics
        self.total_potentiated = 0
        self.total_depressed = 0
        self.total_gated: int = 0  # updates suppressed by modulator ≈ 0

    # ─── Public API ─────────────────────────────────────────────

    def set_modulator(self, modulator: float) -> None:
        """Set the neuromodulatory gating signal for three-factor STDP.

        The modulator is the dopamine / reward prediction error signal
        from the TD learner (Schultz, 1997). It gates STDP updates:

        - ``modulator = 1.0``: standard two-factor STDP (no gating).
        - ``modulator > 1.0``: amplified plasticity (dopamine burst).
        - ``0 < modulator < 1.0``: attenuated plasticity.
        - ``modulator ≈ 0``: plasticity suppressed (no reward signal).
        - ``modulator < 0``: reversed plasticity (LTP↔LTD swap —
          punishment weakens the connection that STDP would have
          strengthened).

        This implements the three-factor rule
        (Pawlak et al., 2010; Frémaux & Gerstner, 2016):

            Δw = STDP(pre, post) × modulator

        Args:
            modulator: The modulatory signal. Typically the TD
                learner's ``get_dopamine_signal()`` output, rescaled
                to a reasonable range (e.g., 0–2).
        """
        with self._lock:
            self._modulator = float(modulator)

    def get_modulator(self) -> float:
        """Return the current neuromodulatory gating signal."""
        with self._lock:
            return self._modulator

    def record_spike(self, concept: str, time_ms: float) -> None:
        """Record a spike (activation event) for a concept.

        After recording, STDP pairs are computed against the recent
        spike history of *other* concepts. Pairs within the learning
        window produce pending potentiation or depression updates.

        Args:
            concept: The concept that spiked.
            time_ms: The spike time (logical milliseconds).
        """
        concept = self.network._resolve(concept) or concept
        if not self.embeddings.has_embeddings:
            return
        if self.embeddings.get_concept_vector(concept) is None:
            return

        pairs: list[tuple[str, str, float, float]] = []

        with self._lock:
            history = self._spike_history
            # For every other concept that spiked recently, compute Δt.
            for other, other_times in history.items():
                if other == concept:
                    continue
                # Find all spikes of `other` within the learning window.
                #
                # The deque is ordered by insertion order, NOT by
                # timestamp. When timestamps are non-monotonic (e.g.,
                # the autonomous learner resets its logical clock to
                # t=0 each session, or tests use non-monotonic times
                # to exercise LTD), a recently-inserted spike may have
                # a larger t_other than an older-inserted one. Iterating
                # reversed (newest-inserted first) and breaking on the
                # first out-of-window spike would skip older-inserted
                # entries with smaller t_other that ARE within the
                # window. Use `continue` to skip out-of-window spikes
                # without stopping the scan.
                for t_other in reversed(other_times):
                    delta_t = time_ms - t_other  # t_post - t_pre
                    if abs(delta_t) > self.window_ms:
                        continue
                    # `other` is pre, `concept` is post (this spike).
                    w = self._stdp_weight(delta_t)
                    if abs(w) > 1e-9:
                        pairs.append((other, concept, delta_t, w))

            # Store this spike
            dq = history.setdefault(concept, deque(maxlen=self._max_history))
            dq.append(time_ms)
            self._pending.extend(pairs)

        if pairs:
            # Apply immediately for responsiveness (small updates).
            self._apply_pending()

    def record_spikes(self, concepts: list[str], time_ms: float) -> None:
        """Record simultaneous spikes for a set of concepts.

        Concepts that spike at the same time (Δt ≈ 0) produce weak
        symmetric potentiation — this recovers the basic Hebbian
        co-occurrence rule as a special case of STDP at Δt = 0.

        Args:
            concepts: The concepts that spiked together.
            time_ms: The spike time (logical milliseconds).
        """
        for concept in concepts:
            self.record_spike(concept, time_ms)

    def apply_updates(self) -> dict[str, int]:
        """Apply all pending STDP updates to the embeddings.

        Returns:
            A dict with 'potentiated' and 'depressed' counts.
        """
        return self._apply_pending()

    def compute_stdp_window(self, delta_t: float) -> float:
        """Compute the STDP weight change for a given timing difference.

        This is the core STDP function W(Δt):

            Δt > 0 (pre before post):  W = A₊ · exp(−Δt / τ₊)   (LTP)
            Δt < 0 (post before pre):  W = −A₋ · exp(Δt / τ₋)   (LTD)
            Δt = 0:                    W = A₊                    (max LTP)

        Args:
            delta_t: t_post − t_pre in milliseconds.

        Returns:
            The signed weight change (positive = potentiation,
            negative = depression).
        """
        return self._stdp_weight(delta_t)

    def get_statistics(self) -> dict[str, int | float]:
        """Return statistics about the STDP engine."""
        with self._lock:
            total_spikes = sum(len(dq) for dq in self._spike_history.values())
            return {
                "total_potentiated": self.total_potentiated,
                "total_depressed": self.total_depressed,
                "total_gated": self.total_gated,
                "modulator": self._modulator,
                "tracked_concepts": len(self._spike_history),
                "total_spikes_buffered": total_spikes,
                "pending_updates": len(self._pending),
            }

    def clear_history(self) -> None:
        """Clear all spike history and pending updates.

        Useful for resetting the STDP state between learning phases
        (e.g., between waking and sleep) to prevent cross-phase
        interference.
        """
        with self._lock:
            self._spike_history.clear()
            self._pending.clear()

    # ─── Internal ───────────────────────────────────────────────

    def _stdp_weight(self, delta_t: float) -> float:
        """The STDP weight-change function W(Δt)."""
        if delta_t > 0:
            # Pre before post → LTP (potentiation)
            return self.a_plus * float(np.exp(-delta_t / self.tau_plus))
        if delta_t < 0:
            # Post before pre → LTD (depression)
            return -self.a_minus * float(np.exp(delta_t / self.tau_minus))
        # Δt = 0 → maximum potentiation (co-firing)
        return self.a_plus

    def _apply_pending(self) -> dict[str, int]:
        """Apply buffered STDP updates to the embedding vectors.

        Three-factor gating: each update is multiplied by the current
        modulator (dopamine / RPE). A modulator near 0 suppresses the
        update; a negative modulator reverses it (LTP↔LTD swap).
        """
        if not self.embeddings.has_embeddings:
            return {"potentiated": 0, "depressed": 0}

        with self._lock:
            pending = self._pending
            self._pending = []
            modulator = self._modulator

        potentiated = 0
        depressed = 0
        gated = 0
        for pre, post, _delta_t, w in pending:
            # Three-factor: gate the STDP weight by the modulator.
            gated_w = w * modulator
            if abs(gated_w) < 1e-12:
                gated += 1
                continue
            if gated_w > 0:
                if self._potentiate(pre, post, gated_w):
                    potentiated += 1
            elif gated_w < 0:
                if self._depress(pre, post, -gated_w):
                    depressed += 1

        self.total_potentiated += potentiated
        self.total_depressed += depressed
        with self._lock:
            self.total_gated += gated
        return {"potentiated": potentiated, "depressed": depressed}

    def _experiential_offset(self) -> int:
        """Return the column offset where experiential data starts."""
        if self.embeddings._spectral_dim > 0:
            return self.embeddings._spectral_dim
        return self.embeddings._tfidf_offset

    def _potentiate(self, pre: str, post: str, magnitude: float) -> bool:
        """Potentiate (LTP): move pre and post embeddings closer.

        Uses Oja-normalized attraction (same as the Hebbian engine) to
        prevent unbounded norm growth.
        """
        idx_a = self.embeddings._concept_to_idx.get(pre)
        idx_b = self.embeddings._concept_to_idx.get(post)
        if idx_a is None or idx_b is None:
            return False

        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return False

        start = self._experiential_offset()
        v_a = matrix[idx_a, start:]
        v_b = matrix[idx_b, start:]

        lr = self.learning_rate * magnitude
        diff = v_b - v_a
        # Cosine similarity (vectors may not be unit-normalised between
        # STDP updates and renormalize() calls, so we normalise here to
        # keep |similarity| ≤ 1.0 — matching _depress and plasticity.
        norm_a = float(np.linalg.norm(v_a))
        norm_b = float(np.linalg.norm(v_b))
        if norm_a > 1e-8 and norm_b > 1e-8:
            similarity = float(np.dot(v_a, v_b)) / (norm_a * norm_b)
        else:
            similarity = 0.0
        # Oja-normalized attraction
        v_a += lr * diff - lr * similarity * v_a
        v_b -= lr * diff + lr * similarity * v_b
        return True

    def _depress(self, pre: str, post: str, magnitude: float) -> bool:
        """Depress (LTD): move pre and post embeddings apart.

        Uses bounded repulsion (scaled by 1 − |similarity|) to prevent
        unbounded norm growth, matching the Hebbian engine's repulsion.
        """
        idx_a = self.embeddings._concept_to_idx.get(pre)
        idx_b = self.embeddings._concept_to_idx.get(post)
        if idx_a is None or idx_b is None:
            return False

        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return False

        start = self._experiential_offset()
        v_a = matrix[idx_a, start:]
        v_b = matrix[idx_b, start:]

        lr = self.learning_rate * magnitude
        diff = v_b - v_a
        # Cosine similarity (vectors may not be unit-normalised between
        # STDP updates and renormalize() calls, so we normalise here to
        # keep |similarity| ≤ 1.0 and the repulsion scale non-negative).
        norm_a = float(np.linalg.norm(v_a))
        norm_b = float(np.linalg.norm(v_b))
        if norm_a > 1e-8 and norm_b > 1e-8:
            similarity = abs(float(np.dot(v_a, v_b)) / (norm_a * norm_b))
        else:
            similarity = 0.0
        scale = lr * (1.0 - similarity)
        v_a -= scale * diff
        v_b += scale * diff
        return True

    def renormalize(self) -> None:
        """Re-normalize all embedding rows after STDP updates.

        Should be called periodically (e.g., during sleep) to keep
        cosine similarity valid.
        """
        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        matrix /= norms
        self.embeddings._concept_cache.clear()

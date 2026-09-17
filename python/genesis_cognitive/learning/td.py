"""Temporal-difference (TD(λ)) learning — dopamine as a reward
prediction error with eligibility traces.

In the brain, dopamine neurons in the ventral tegmental area (VTA)
and substantia nigra do not signal reward directly — they signal
*reward prediction error* (RPE): the difference between the reward
actually received and the reward that was expected (Schultz, 1997;
Schultz et al., 1997).

    RPE = actual_reward − expected_reward

- **Positive RPE** (reward better than expected): dopamine neurons
  fire bursts. The state/action that led to the reward increases in
  value — the organism learns to repeat it.
- **Negative RPE** (reward worse than expected): dopamine neurons
  pause (dip below baseline). The state/action decreases in value.
- **Zero RPE** (reward as expected): no dopamine change. The
  prediction is already accurate; nothing to learn.

# TD(λ): eligibility traces

This module implements **TD(λ)** — temporal-difference learning with
*eligibility traces* (Sutton, 1988; Sutton & Barto, 2018, Ch. 12).
TD(λ) generalizes TD(0): the parameter λ ∈ [0, 1] interpolates between
one-step TD(0) (λ=0) and Monte Carlo (λ=1) by controlling how far
credit for a reward prediction error propagates backward through the
sequence of recently-visited states.

The TD(λ) backward-view update (Sutton & Barto, 2018, §12.2):

    δ = r + γ · V(s') − V(s)           # TD error (RPE)
    e ← γλ · e + ∇V(s)                 # eligibility trace update
    w ← w + α · δ · e                  # weight update

where:
    - V(s) is the value of state s,
    - r is the reward received,
    - γ is the discount factor (future rewards worth less),
    - s' is the next state,
    - α is the learning rate,
    - λ is the trace-decay parameter,
    - e is the eligibility trace (one entry per feature/weight),
    - ∇V(s) is the gradient of V(s) w.r.t. the weights.

The eligibility trace e is a decaying record of how recently and how
strongly each weight has been "visited" (its concept was active).
When a TD error arrives, ALL traced weights are updated in
proportion to their trace — not just the weights of the current
state. This is the credit-assignment mechanism: a concept visited
three steps ago still receives a share of the credit (or blame),
decayed by (γλ)³.

**Replacing traces** (Sutton & Barto, 2018, §12.2) are used instead of
accumulating traces. For sparse binary features (a concept is either
active or not), replacing traces cap the trace at the feature value
rather than letting it accumulate without bound on repeated visits.
This prevents runaway traces on frequently-visited concepts and is
the standard choice for sparse feature representations.

When λ = 0, the trace decays to zero immediately after each step, and
the update reduces *exactly* to TD(0):

    e = ∇V(s)        # trace = current gradient only
    w ← w + α · δ · ∇V(s)

This makes the upgrade fully backward-compatible: existing callers
that construct ``TDLearner(network)`` without specifying λ get TD(0)
behavior identical to the previous implementation.

# Biological basis: synaptic eligibility tags

Eligibility traces are not merely a computational convenience — they
correspond to a well-established neurobiological mechanism. The
**synaptic tag hypothesis** (Morris, 2006; Pan et al., 2005; Redondo
& Morris, 2011) proposes that:

1. When a synapse is active (pre-synaptic firing coincides with
   post-synaptic depolarization), it acquires a temporary
   **eligibility tag** — a molecular marker (likely involving
   Ca²⁺ dynamics and CAMKII autophosphorylation) that marks it as
   "eligible for plasticity."
2. The tag decays over seconds to minutes.
3. When a neuromodulator (dopamine) arrives — signaling a reward
   prediction error — only tagged (eligible) synapses undergo
   plasticity. The magnitude of change is proportional to both the
   dopamine signal and the tag strength.

This is exactly what TD(λ) computes: the eligibility trace *is* the
synaptic tag, the TD error *is* the dopamine signal, and the weight
update *is* the plasticity. Genesis's dopamine RPE (from
``get_dopamine_signal()``) modifies all recently-active concepts in
proportion to their trace strength — not just the current state.

# Value function approximation

States are sets of active concepts. The value of a state is
approximated as a weighted sum of the confidences of its active
concepts — a linear function approximator over the concept network.
This lets TD learning generalize across similar states (states
sharing concepts have similar values) without storing a value for
every possible state combination.

# Integration with dopamine

The TD error (RPE) is the dopamine signal. ``get_dopamine_signal()``
returns the most recent RPE, scaled for the neurochemical system.
Positive RPE → dopamine burst; negative RPE → dopamine dip. This
integrates with the existing dopamine impulse system in the
autonomous learner.

References:
    - Schultz (1997): Dopamine neurons and their role in reward
      mechanisms. Current Opinion in Neurobiology, 7, 191-197.
    - Schultz et al. (1997): A neural substrate of prediction and
      reward. Science, 275, 1593-1599.
    - Sutton (1988): Learning to predict by the methods of temporal
      differences. Machine Learning, 3, 9-44.
    - Sutton & Barto (2018): Reinforcement Learning: An Introduction
      (2nd ed.). MIT Press, Ch. 12 (Eligibility Traces).
    - Pan et al. (2005): Reward-predicting stimuli enhance dopamine
      release in the VTA. Nature, 437, 1066-1070.
    - Morris (2006): Elements of a neurobiological theory of
      hippocampal function. Hippocampus, 16(11), 917-926.
    - Redondo & Morris (2011): Making memories last: the synaptic
      tagging and capture hypothesis. Nature Reviews Neuroscience,
      12, 17-30.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..concepts import ConceptNetwork

__all__ = ["TDLearner", "TDTransition"]


@dataclass(slots=True)
class TDTransition:
    """A state transition record for TD learning.

    Attributes:
        state: The set of active concepts (state).
        reward: The reward received.
        next_state: The resulting state.
        rpe: The reward prediction error computed for this transition.
        timestamp: When the transition occurred.
    """

    state: list[str]
    reward: float
    next_state: list[str]
    rpe: float
    timestamp: int


class TDLearner:
    """Temporal-difference learning with eligibility traces (TD(λ)).

    Learns to predict the value of states (sets of active concepts)
    using the TD(λ) rule with replacing eligibility traces. The value
    function is a linear approximation over concept confidences,
    enabling generalization across similar states.

    The reward prediction error (RPE) serves as the dopamine signal,
    integrating with Genesis's neurochemical system. Eligibility traces
    propagate credit (or blame) backward through recently-visited
    states — the longer ago a state was visited, the smaller its
    share of the credit, decayed by (γλ)ⁿ where n is the number of
    steps elapsed.

    Args:
        network: The concept network (provides concept confidences
            for value approximation).
        learning_rate: TD learning rate α (default 0.1).
        discount: Discount factor γ for future rewards (default 0.9).
        lam: Trace-decay parameter λ ∈ [0, 1] (default 0.0).
            λ=0 gives TD(0) (one-step, no credit propagation).
            λ=1 gives Monte Carlo-like (full episode credit).
            Typical production values: 0.7–0.9. The trace decays by a
            factor of γλ per step, so the effective credit horizon is
            ~1/(1−γλ) steps.
        dopamine_scale: Scaling factor for the dopamine signal
            (default 0.05, matching the impulse magnitudes used by
            the autonomous learner).
    """

    DEFAULT_LEARNING_RATE: float = 0.1
    DEFAULT_DISCOUNT: float = 0.9
    DEFAULT_LAMBDA: float = 0.0
    DEFAULT_DOPAMINE_SCALE: float = 0.05
    _TRACE_PRUNE_THRESHOLD: float = 1e-10

    def __init__(
        self,
        network: ConceptNetwork,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount: float = DEFAULT_DISCOUNT,
        lam: float = DEFAULT_LAMBDA,
        dopamine_scale: float = DEFAULT_DOPAMINE_SCALE,
    ) -> None:
        """Initialize the TD(λ) learner with network, rates, and dopamine scaling."""
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        self.network = network
        self.learning_rate = learning_rate
        self.discount = discount
        self.lam = lam
        self.dopamine_scale = dopamine_scale

        # Per-concept value weights (learned). The state value is a
        # weighted sum of active concept confidences. Weights start
        # at 1.0 (confidence alone determines value).
        self._weights: dict[str, float] = {}

        # Eligibility traces (one per concept). Decays by γλ per step;
        # set to the gradient value for active concepts (replacing
        # traces). Traces below _TRACE_PRUNE_THRESHOLD are pruned to
        # keep the dict bounded.
        self._traces: dict[str, float] = {}

        # History of transitions (for analysis; eligibility traces
        # are tracked separately in _traces).
        self._history: list[TDTransition] = []
        self._max_history: int = 500

        # The most recent RPE — the current dopamine signal.
        self._last_rpe: float = 0.0

        # Statistics
        self.total_updates = 0
        self.total_positive_rpe = 0
        self.total_negative_rpe = 0

    # ─── Public API ─────────────────────────────────────────────

    def predict_value(self, state: list[str]) -> float:
        """Predict the value of a state.

        The value is a weighted sum of the confidences of the active
        concepts in the state. Concepts not in the network contribute
        zero. Weights are learned via TD updates.

        V(s) = Σ_i w_i · confidence(concept_i)

        Args:
            state: A list of concept names (the active concepts).

        Returns:
            The predicted value of the state.
        """
        if not state:
            return 0.0
        total = 0.0
        count = 0
        for concept in state:
            cid = self.network._resolve(concept)
            if not cid:
                continue
            c = self.network.get_concept(cid)
            if c is None:
                continue
            w = self._weights.get(cid, 1.0)
            total += w * c.confidence
            count += 1
        # Normalize by count to keep value in a stable range
        return total / count if count > 0 else 0.0

    def update(
        self,
        state: list[str],
        reward: float,
        next_state: list[str],
    ) -> float:
        """Apply a TD(λ) update and return the reward prediction error.

        The TD error (RPE) is:

            δ = reward + γ · V(s') − V(s)

        Eligibility traces are then updated (backward view):

            e ← γλ · e                    # decay all traces
            e[i] = ∇V_i(s)  for active i  # replacing trace

        and all traced weights are updated:

            w[i] ← w[i] + α · δ · e[i]

        With λ = 0, the trace for inactive concepts is zero and the
        trace for active concepts equals the gradient, so the update
        reduces exactly to TD(0): only the current state's concepts
        are updated, each receiving an equal share of α·δ.

        With λ > 0, concepts from recently-visited states retain
        decaying traces and also receive a share of the credit (or
        blame), enabling multi-step temporal credit assignment.

        Args:
            state: The current state (active concepts).
            reward: The reward received.
            next_state: The resulting state.

        Returns:
            The reward prediction error δ (the dopamine signal).
        """
        v_s = self.predict_value(state)
        v_sp = self.predict_value(next_state)
        rpe = reward + self.discount * v_sp - v_s

        # ── Eligibility trace update (backward view) ──────────────
        # 1. Decay all existing traces by γλ.
        decay = self.discount * self.lam
        if decay > 0.0 and self._traces:
            prune = self._TRACE_PRUNE_THRESHOLD
            for cid in list(self._traces):
                self._traces[cid] *= decay
                if abs(self._traces[cid]) < prune:
                    del self._traces[cid]
        elif self._traces:
            # λ = 0: traces decay to zero immediately.
            self._traces.clear()

        # 2. Resolve the active concepts and set their traces
        #    (replacing trace: overwrite, don't accumulate).
        #    The gradient of V(s) w.r.t. w_i is 1/count for each
        #    active concept (matching the TD(0) equal-share credit).
        active_cids: list[str] = []
        for concept in state:
            resolved: str | None = self.network._resolve(concept)
            if resolved is not None and self.network.get_concept(resolved) is not None:
                active_cids.append(resolved)

        if active_cids:
            grad = 1.0 / len(active_cids)
            for ac in active_cids:
                self._traces[ac] = grad

        # 3. Apply the TD error to ALL traced weights.
        #    With λ = 0, only the current state's concepts have
        #    non-zero traces → identical to the old TD(0) update.
        #    With λ > 0, previously-visited concepts also get credit.
        if self._traces:
            alpha_delta = self.learning_rate * rpe
            for tc, trace in self._traces.items():
                current = self._weights.get(tc, 1.0)
                self._weights[tc] = current + alpha_delta * trace

        # Record history
        self._history.append(
            TDTransition(
                state=list(state),
                reward=reward,
                next_state=list(next_state),
                rpe=rpe,
                timestamp=int(time.time() * 1000),
            )
        )
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self._last_rpe = rpe
        self.total_updates += 1
        if rpe > 0:
            self.total_positive_rpe += 1
        elif rpe < 0:
            self.total_negative_rpe += 1

        return rpe

    def get_dopamine_signal(self) -> float:
        """Get the current dopamine signal (scaled RPE).

        The reward prediction error is the dopamine signal in the
        brain (Schultz, 1997). This returns the most recent RPE,
        scaled to the magnitude range used by the neurochemical
        impulse system. Positive RPE → positive dopamine (burst);
        negative RPE → negative dopamine (dip).

        Returns:
            The scaled dopamine signal.
        """
        return self._last_rpe * self.dopamine_scale

    def get_rpe(self) -> float:
        """Get the most recent (unscaled) reward prediction error.

        Returns:
            The most recent RPE.
        """
        return self._last_rpe

    def get_weight(self, concept: str) -> float:
        """Get the learned value weight for a concept.

        Args:
            concept: The concept name.

        Returns:
            The weight (default 1.0 if unlearned).
        """
        cid = self.network._resolve(concept) or concept
        return self._weights.get(cid, 1.0)

    def get_trace(self, concept: str) -> float:
        """Get the eligibility trace value for a concept.

        The trace reflects how recently the concept was part of an
        active state. Higher = more recently active (more eligible
        for credit assignment). Zero = not traced.

        Args:
            concept: The concept name.

        Returns:
            The eligibility trace (0.0 if not traced).
        """
        cid = self.network._resolve(concept) or concept
        return self._traces.get(cid, 0.0)

    def reset_traces(self) -> None:
        """Clear all eligibility traces.

        Called at episode/task boundaries to prevent credit from one
        task leaking into the next. In continuous operation (no
        explicit episodes), traces decay naturally and this is rarely
        needed.
        """
        self._traces.clear()

    def get_history(self, limit: int = 10) -> list[TDTransition]:
        """Get recent TD transitions.

        Args:
            limit: Maximum number of transitions to return.

        Returns:
            A list of recent TDTransition records (most recent last).
        """
        return self._history[-limit:]

    def get_statistics(self) -> dict[str, int | float]:
        """Return statistics about the TD learner."""
        return {
            "total_updates": self.total_updates,
            "total_positive_rpe": self.total_positive_rpe,
            "total_negative_rpe": self.total_negative_rpe,
            "last_rpe": self._last_rpe,
            "dopamine_signal": self.get_dopamine_signal(),
            "tracked_weights": len(self._weights),
            "active_traces": len(self._traces),
            "lambda": self.lam,
            "history_length": len(self._history),
        }

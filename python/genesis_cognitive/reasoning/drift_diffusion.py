"""Drift-diffusion model — evidence accumulation for decision-making.

Replaces the priority-tree deliberation with a biologically plausible
evidence-accumulation process. Multiple evidence sources (perception,
emotion, memory, reasoning) contribute drift toward different response
options in parallel. When evidence for one option crosses a threshold,
that option wins.

# The drift-diffusion model (DDM)

The DDM (Ratcliff, 1978; Ratcliff & McKoon, 2008) models decision-making
as a noisy accumulation of evidence over time. A decision variable
drifts toward one of two (or more) boundaries; the first boundary
reached determines the response, and the time to reach it determines
the response time.

Key properties that make this biologically plausible:

1. **Parallel accumulation**: Multiple evidence sources contribute
   simultaneously, each pushing the decision variable toward different
   options. This mirrors how the brain integrates sensory, emotional,
   and mnemonic evidence in parallel (Gold & Shadlen, 2007).

2. **Noise**: Accumulation is noisy, reflecting neural variability.
   This produces response-time distributions and occasional errors
   that match human data (Ratcliff, 1978).

3. **Speed-accuracy tradeoff**: Lower thresholds produce faster but
   less accurate decisions; higher thresholds produce slower but more
   accurate decisions. This is the speed-accuracy tradeoff (SAT),
   controlled by the ACC-prefrontal loop (Forstmann et al., 2008).

4. **Non-decision time**: A constant offset representing perceptual
   and motor processing that occurs before/after the decision itself.

For Genesis, we extend the classic two-boundary DDM to a multi-option
race model — each response option has its own accumulator, and the
first to cross threshold wins. This is the race model architecture
(Deco & Rolls, 2006; Bogacz et al., 2006), which generalizes the DDM
to multiple alternatives while preserving its core properties.

References:
- Ratcliff, R. (1978). A theory of memory retrieval. Psychological Review.
- Ratcliff, R., & McKoon, G. (2008). The diffusion decision model.
  Psychological Review.
- Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision
  making. Annual Review of Neuroscience.
- Bogacz, R., et al. (2006). The physics of optimal decision making.
  Physical Review E.
- Forstmann, B. U., et al. (2008). Striatum and pre-SMA facilitate
  decision-making under time pressure. PNAS.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..brain_waves import BrainWave, BrainWaveState

__all__ = ["DecisionResult", "DriftDiffusionModel", "EvidenceAccumulator"]


@dataclass(slots=True)
class EvidenceAccumulator:
    """A single accumulator for one response option.

    Each accumulator integrates evidence from multiple sources toward
    a single decision threshold. The accumulator has:
    - A current evidence level (starts at 0)
    - A threshold (when evidence crosses this, the option wins)
    - Per-source drift rates (how much each source contributes per tick)

    Biological basis: each accumulator corresponds to a population of
    neurons in the prefrontal cortex / basal ganglia that integrates
    evidence for a particular action. The threshold corresponds to the
    firing rate at which the basal ganglia "selects" that action
    (Lo & Wang, 2006).
    """

    option: str
    threshold: float = 1.0
    evidence: float = 0.0
    # Per-source drift contributions accumulated since last tick
    _pending_drift: float = field(default=0.0)
    # Sources that contributed (for transparency)
    sources: dict[str, float] = field(default_factory=dict)
    # Whether this accumulator has crossed threshold
    decided: bool = False
    # Time (in ticks) when decision was reached
    decision_time: float = 0.0

    @property
    def progress(self) -> float:
        """How close to threshold (0..1)."""
        return min(1.0, self.evidence / self.threshold) if self.threshold > 0 else 0.0


@dataclass(slots=True)
class DecisionResult:
    """The outcome of a drift-diffusion decision process.

    Captures not just *what* was decided but *how* — the winning option,
    the time it took, the confidence (margin over threshold), and which
    evidence sources contributed. This makes the decision transparent
    and debuggable, just like the old priority tree.
    """

    option: str
    confidence: float
    decision_time: float  # in ticks
    evidence_sources: dict[str, dict[str, float]]  # source → {option: amount}
    all_evidence: dict[str, float]  # option → final evidence level


class DriftDiffusionModel:
    """Multi-option drift-diffusion evidence accumulation.

    Multiple evidence sources (perception, emotion, memory, reasoning)
    accumulate evidence toward different response options in parallel.
    When evidence for one option crosses a threshold, that option wins.

    This replaces the priority-tree deliberation in the cognition engine
    with a more biologically plausible process. Instead of a fixed
    priority order (greeting > farewell > question > ...), evidence
    from all sources accumulates simultaneously, and the option with
    the strongest convergent evidence wins.

    # Usage

        ddm = DriftDiffusionModel(threshold=1.0, noise=0.05)
        ddm.add_option("greet")
        ddm.add_option("inform")
        ddm.add_option("empathize")

        # Evidence from perception: input looks like a greeting
        ddm.add_evidence("perception", "greet", 0.6)
        # Evidence from emotion: feeling warm → lean toward greeting
        ddm.add_evidence("emotion", "greet", 0.2)
        # Evidence from memory: past greetings in similar context
        ddm.add_evidence("memory", "greet", 0.3)

        result = ddm.tick(dt=1.0)
        if result:
            print(f"Decided: {result.option} (confidence={result.confidence:.2f})")

    # Speed-accuracy tradeoff

    The threshold controls the speed-accuracy tradeoff:
    - Low threshold (0.5): fast decisions, more errors
    - High threshold (1.5): slow decisions, fewer errors

    The error monitor (ACC) adjusts the threshold based on recent
    error rates — after errors, the threshold rises (more cautious).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        noise: float = 0.03,
        non_decision_time: float = 0.0,
        decay: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize the drift-diffusion model.

        Args:
            threshold: Decision boundary. Higher = more cautious.
                Default 1.0. Typical human DDM thresholds range 0.5–2.5.
            noise: Standard deviation of Gaussian noise added to each
                accumulator per tick. Models neural variability.
                Default 0.03.
            non_decision_time: Constant offset (in ticks) added to
                decision time. Models perceptual/motor processing.
                Default 0.0.
            decay: Evidence decay rate per tick (0 = no decay).
                Models leaky integration. Default 0.0.
            rng: Optional random number generator for reproducibility.
        """
        self.threshold = threshold
        self.noise = noise
        self.non_decision_time = non_decision_time
        self.decay = decay
        self._rng = rng or random.Random()
        self._accumulators: dict[str, EvidenceAccumulator] = {}
        self._elapsed: float = 0.0
        # Track all evidence added this round for transparency
        self._round_evidence: dict[str, dict[str, float]] = {}

    def add_option(self, option: str, threshold: float | None = None) -> None:
        """Register a response option for the race.

        Args:
            option: The response option identifier (e.g., "greet",
                "inform", "empathize").
            threshold: Optional per-option threshold. If None, uses
                the model's default threshold.
        """
        if option not in self._accumulators:
            self._accumulators[option] = EvidenceAccumulator(
                option=option,
                threshold=threshold if threshold is not None else self.threshold,
            )
            # A zero threshold would cause divide-by-zero in _make_result.
            if self._accumulators[option].threshold <= 0:
                self._accumulators[option].threshold = self.threshold

    def add_evidence(self, source: str, option: str, amount: float) -> None:
        """Add evidence from a source toward an option.

        Multiple sources can contribute to the same option, and one
        source can contribute to multiple options. Evidence accumulates
        until tick() is called, at which point it's integrated into
        the accumulators with noise.

        Args:
            source: The evidence source (e.g., "perception", "emotion",
                "memory", "reasoning").
            option: The response option the evidence favors.
            amount: How much evidence to add (can be negative to
                oppose an option).
        """
        if option not in self._accumulators:
            self.add_option(option)

        acc = self._accumulators[option]
        acc._pending_drift += amount

        # Track per-source contributions for transparency
        if source not in acc.sources:
            acc.sources[source] = 0.0
        acc.sources[source] += amount

        # Track round evidence
        if source not in self._round_evidence:
            self._round_evidence[source] = {}
        self._round_evidence[source][option] = (
            self._round_evidence[source].get(option, 0.0) + amount
        )

    def tick(
        self, dt: float = 1.0, brain_waves: BrainWaveState | None = None,
    ) -> DecisionResult | None:
        """Advance accumulation by dt and check for a decision.

        Integrates pending evidence into each accumulator with Gaussian
        noise, applies decay, and checks if any accumulator has crossed
        its threshold. If multiple cross simultaneously, the one with
        the highest evidence wins.

        Args:
            dt: Time step (in arbitrary units). Default 1.0.
            brain_waves: Optional brain wave state. Gamma speeds up
                accumulation (fast integration), theta slows it down
                (consolidation/review mode), delta nearly halts it
                (deep rest).

        Returns:
            A DecisionResult if a decision was reached, None otherwise.
        """
        self._elapsed += dt

        # Brain-wave-modulated drift multiplier.
        # Gamma → faster accumulation (integration mode).
        # Theta → slower accumulation (consolidation, not decision).
        # Delta → very slow (deep rest, minimal decision-making).
        drift_mult = 1.0
        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.GAMMA:
                drift_mult = 1.0 + 0.3 * brain_waves.integration
            elif dom == BrainWave.THETA:
                drift_mult = 0.6
            elif dom == BrainWave.DELTA:
                drift_mult = 0.2

        winner: EvidenceAccumulator | None = None

        for acc in self._accumulators.values():
            if acc.decided:
                continue

            # Integrate pending drift with noise
            noise_val = self._rng.gauss(0.0, self.noise) if self.noise > 0 else 0.0
            drift = (acc._pending_drift + noise_val) * drift_mult
            acc.evidence += drift * dt
            acc._pending_drift = 0.0

            # Apply decay (leaky integration)
            if self.decay > 0:
                acc.evidence *= 1.0 - self.decay * dt

            # Prevent evidence from going negative (bounded below)
            if acc.evidence < 0:
                acc.evidence = 0.0

            # Check threshold
            if acc.evidence >= acc.threshold:
                acc.decided = True
                acc.decision_time = self._elapsed
                if winner is None or acc.evidence > winner.evidence:
                    winner = acc

        if winner is not None:
            return self._make_result(winner)

        return None

    def _make_result(self, winner: EvidenceAccumulator) -> DecisionResult:
        """Build a DecisionResult from the winning accumulator."""
        # Confidence: how far above threshold, normalized.
        # Guard against threshold == 0 (defensive — add_option should
        # reject it, but a directly-constructed accumulator could have 0).
        confidence = min(1.0, winner.evidence / winner.threshold) if winner.threshold > 0 else 1.0
        # If evidence is well above threshold, confidence approaches 1.0
        # If barely at threshold, confidence ~ 1.0 (it won)
        # Add margin from second-best for confidence calibration
        all_evidence = {opt: acc.evidence for opt, acc in self._accumulators.items()}

        return DecisionResult(
            option=winner.option,
            confidence=confidence,
            decision_time=winner.decision_time + self.non_decision_time,
            evidence_sources=dict(self._round_evidence),
            all_evidence=all_evidence,
        )

    def reset(self) -> None:
        """Reset all accumulators for a new decision.

        Clears all evidence and decided flags. Options remain registered.
        """
        for acc in self._accumulators.values():
            acc.evidence = 0.0
            acc._pending_drift = 0.0
            acc.decided = False
            acc.decision_time = 0.0
            acc.sources.clear()
        self._elapsed = 0.0
        self._round_evidence.clear()

    def clear_options(self) -> None:
        """Remove all options (full reset for a new decision context)."""
        self._accumulators.clear()
        self._elapsed = 0.0
        self._round_evidence.clear()

    def get_evidence(self, option: str) -> float:
        """Get current evidence level for an option."""
        acc = self._accumulators.get(option)
        return acc.evidence if acc else 0.0

    def get_progress(self, option: str) -> float:
        """Get progress toward threshold for an option (0..1)."""
        acc = self._accumulators.get(option)
        return acc.progress if acc else 0.0

    def get_sources(self, option: str) -> dict[str, float]:
        """Get per-source evidence contributions for an option."""
        acc = self._accumulators.get(option)
        return dict(acc.sources) if acc else {}

    @property
    def options(self) -> list[str]:
        """List of registered options."""
        return list(self._accumulators.keys())

    @property
    def has_pending(self) -> bool:
        """Whether any accumulator has pending (un-integrated) drift."""
        return any(acc._pending_drift != 0.0 for acc in self._accumulators.values())

    def set_threshold(self, threshold: float) -> None:
        """Adjust the decision threshold (speed-accuracy tradeoff).

        Called by the error monitor when caution level changes —
        after errors, the threshold rises (more cautious); after
        successes, it falls (faster decisions).

        Args:
            threshold: New threshold value (typically 0.5–2.5).
        """
        self.threshold = max(0.1, threshold)
        for acc in self._accumulators.values():
            if not acc.decided:
                acc.threshold = self.threshold

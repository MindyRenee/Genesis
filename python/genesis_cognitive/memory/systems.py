"""Memory subsystems — biologically grounded memory processes.

This module implements several memory subsystems that sit alongside the
main :class:`~genesis_cognitive.memory_engine.MemoryEngine`. Each
subsystem models a distinct facet of biological memory that the core
engine does not cover:

- **Emotional memory** (amygdala-dependent): fear conditioning,
  extinction learning, and emotion-prioritised retrieval.
- **Priming** (implicit memory): recent exposure to a concept speeds
  subsequent processing of related concepts via spreading activation.
- **Attractor networks**: a Hopfield-style pattern-completion /
  auto-association layer providing a biologically plausible
  association mechanism on top of the Rust-side SimHash store.
- **Spreading activation**: graded, temporally-decaying activation
  spreading over the concept network (Collins & Loftus, 1975).

All classes follow the project conventions: ``@dataclass(slots=True)``,
type hints, docstrings, and ``from __future__ import annotations``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — import only for type checkers
    from ..concepts import ConceptNetwork

__all__ = [
    "AttractorNetwork",
    "AttractorPattern",
    "EmotionalMemory",
    "EmotionalMemorySystem",
    "PrimingSystem",
    "SpreadingActivation",
]


# ─── Item 6: Emotional memory system (amygdala-dependent) ────────────


@dataclass(slots=True)
class EmotionalMemory:
    """An amygdala-dependent emotional memory association.

    Emotional memories are encoded differently from declarative
    memories — they are stronger, more vivid, and partially independent
    of contextual (hippocampal) memory. The amygdala modulates
    hippocampal encoding during emotional arousal (McGaugh, 2004).

    This dataclass represents an association between a (formerly
    neutral) stimulus and an emotional response, as formed through
    fear conditioning or other emotional learning.

    Attributes:
        stimulus: The neutral stimulus that acquired emotional salience.
        emotional_tag: 12-dimensional emotional snapshot at encoding
            (mirrors the neurochemical vector used by the daemon).
        association_strength: Strength of the stimulus→emotion link
            [0..1]. Built up during conditioning, reduced during
            extinction.
        extinction_level: How far extinction learning has progressed
            [0..1]. When this reaches ~1.0 the memory is considered
            extinguished (the association is suppressed, though the
            original trace persists — spontaneous recovery can occur).
        is_extinct: Whether the association has been extinguished.
        timestamp: When the association was first formed (ms).
        last_retrieved: When the memory was last recalled (ms).
        retrieval_count: How many times it has been retrieved.
    """

    stimulus: str
    emotional_tag: list[float] = field(default_factory=lambda: [0.5] * 12)
    association_strength: float = 0.5
    extinction_level: float = 0.0
    is_extinct: bool = False
    timestamp: int = 0
    last_retrieved: int = 0
    retrieval_count: int = 0


class EmotionalMemorySystem:
    """Amygdala-dependent emotional memory system.

    Implements fear conditioning (neutral stimulus + negative emotion →
    association), extinction learning (repeated exposure without the
    negative outcome weakens the association), and emotion-prioritised
    retrieval.

    Biological grounding:
        - The amygdala stores emotional associations independently of
          the hippocampus (LaBar & Cabeza, 2006).
        - Fear conditioning is rapid (often a single trial) and robust.
        - Extinction is new inhibitory learning, not erasure — the
          original association persists and can show spontaneous
          recovery (Bouton, 2004).
        - Emotional memories have priority in retrieval during
          matching emotional states (mood-congruent memory, Bower 1981).
    """

    def __init__(self) -> None:
        """Initialize an empty emotional memory store with zeroed event counters."""
        self._memories: dict[str, EmotionalMemory] = {}
        # Track conditioning / extinction events for introspection.
        self.conditioning_count: int = 0
        self.extinction_count: int = 0

    # ── Fear conditioning ────────────────────────────────────────

    def fear_conditioning(
        self,
        neutral_stimulus: str,
        negative_emotion_strength: float,
        emotional_tag: list[float] | None = None,
    ) -> EmotionalMemory:
        """Associate a neutral stimulus with a negative emotional response.

        This is fear conditioning (Pavlovian): a neutral stimulus is
        paired with an aversive outcome, and the stimulus comes to
        evoke the emotional response on its own. Conditioning is rapid
        — a single strong pairing can establish a robust association.

        Args:
            neutral_stimulus: The stimulus to be conditioned.
            negative_emotion_strength: Strength of the negative emotion
                [0..1]. Higher values produce stronger associations.
            emotional_tag: Optional 12-chemical emotional snapshot. If
                omitted, a default negative-biased tag is used.

        Returns:
            The created/updated :class:`EmotionalMemory`.
        """
        if emotional_tag is None:
            # Default negative-biased tag: high cortisol, low serotonin.
            # Neurochemical indices match the Rust canonical ordering:
            # 0=Dopamine, 1=Serotonin, 6=Cortisol.
            emotional_tag = [0.5] * 12
            emotional_tag[6] = max(0.8, emotional_tag[6])  # cortisol
            emotional_tag[1] = min(0.2, emotional_tag[1])  # serotonin

        existing = self._memories.get(neutral_stimulus)
        if existing is not None and not existing.is_extinct:
            # Re-conditioning strengthens the existing association.
            existing.association_strength = min(
                1.0,
                existing.association_strength + negative_emotion_strength * 0.4,
            )
            # Re-conditioning partially reverses extinction.
            existing.extinction_level = max(0.0, existing.extinction_level - 0.2)
            existing.is_extinct = False
            self.conditioning_count += 1
            return existing

        memory = EmotionalMemory(
            stimulus=neutral_stimulus,
            emotional_tag=list(emotional_tag),
            association_strength=min(1.0, 0.3 + negative_emotion_strength * 0.5),
            timestamp=int(time.time() * 1000),
        )
        self._memories[neutral_stimulus] = memory
        self.conditioning_count += 1
        return memory

    # ── Extinction learning ──────────────────────────────────────

    def extinction_learning(self, stimulus: str, exposures: int = 1) -> EmotionalMemory | None:
        """Weaken a conditioned association through repeated exposure.

        Extinction learning occurs when the conditioned stimulus is
        presented repeatedly without the aversive outcome. The
        association weakens. Extinction is *new inhibitory learning*
        rather than erasure — the original trace persists and can
        show spontaneous recovery (Bouton, 2004).

        Args:
            stimulus: The conditioned stimulus to extinguish.
            exposures: Number of extinction exposures to apply.

        Returns:
            The updated :class:`EmotionalMemory`, or ``None`` if no
            such memory exists.
        """
        memory = self._memories.get(stimulus)
        if memory is None:
            return None

        # Each exposure reduces association strength and increases
        # extinction level. The decrement per exposure diminishes
        # (extinction curves are exponential).
        for _ in range(max(1, exposures)):
            decrement = 0.15 * (1.0 - memory.extinction_level)
            memory.association_strength = max(0.0, memory.association_strength - decrement)
            memory.extinction_level = min(1.0, memory.extinction_level + 0.1)

        if memory.extinction_level >= 0.9 or memory.association_strength < 0.05:
            memory.is_extinct = True
        self.extinction_count += 1
        return memory

    # ── Emotion-prioritised retrieval ────────────────────────────

    def retrieve_emotional(
        self,
        stimulus: str,
        current_emotion: list[float] | None = None,
    ) -> EmotionalMemory | None:
        """Retrieve an emotional memory, with state-dependent priority.

        Emotional memories have priority in retrieval during matching
        emotional states (mood-congruent memory, Bower 1981). When the
        current emotional state matches the memory's emotional tag,
        the effective retrieval priority is boosted.

        Args:
            stimulus: The stimulus to look up.
            current_emotion: Optional 12-chemical current emotional
                snapshot. If provided and it matches the memory's tag,
                retrieval priority is boosted.

        Returns:
            The matching :class:`EmotionalMemory`, or ``None``.
        """
        memory = self._memories.get(stimulus)
        if memory is None:
            return None
        memory.last_retrieved = int(time.time() * 1000)
        memory.retrieval_count += 1
        # If current emotion matches, boost association (retrieval
        # strengthens emotional memories — reconsolidation).
        if current_emotion is not None and not memory.is_extinct:
            similarity = _emotional_similarity(current_emotion, memory.emotional_tag)
            if similarity > 0.6:
                memory.association_strength = min(1.0, memory.association_strength + 0.05)
        return memory

    def get_all(self) -> list[EmotionalMemory]:
        """Return all emotional memories."""
        return list(self._memories.values())

    @property
    def count(self) -> int:
        """Number of stored emotional memories."""
        return len(self._memories)

    @property
    def active_count(self) -> int:
        """Number of non-extinguished emotional memories."""
        return sum(1 for m in self._memories.values() if not m.is_extinct)


def _emotional_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two 12-chemical emotional vectors."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ─── Item 7: Priming system (implicit memory) ────────────────────────


class PrimingSystem:
    """Implicit memory priming via spreading activation.

    Recent exposure to a concept speeds subsequent processing of
    related concepts. This is *semantic priming* — the classic
    finding that hearing "doctor" speeds recognition of "nurse"
    (Meyer & Schvaneveldt, 1971). Activation spreads from the primed
    concept to its neighbours in the concept network, decaying with
    distance (Collins & Loftus, 1975).

    Priming is implicit: it does not require cognitive recall and is
    independent of declarative memory. It decays over time (typically
    lasting seconds to minutes in humans).

    This system integrates with the existing
    :class:`~genesis_cognitive.concepts.ConceptNetwork` spreading
    activation, using the network's ``spread_activation`` method to
    propagate priming to neighbours.
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        decay_rate: float = 0.02,
        spread_depth: int = 2,
        spread_decay: float = 0.5,
    ) -> None:
        """Initialize the semantic priming system."""
        self._network = network
        self._decay_rate = decay_rate  # per second
        self._spread_depth = spread_depth
        self._spread_decay = spread_decay
        # concept_id → (priming_level, last_update_timestamp)
        self._primed: dict[str, tuple[float, float]] = {}
        self._last_decay: float = time.time()

    def prime(self, concept_name: str, amount: float = 1.0) -> None:
        """Prime a concept — recent exposure boosts its accessibility.

        If a concept network is attached, priming spreads to related
        concepts via the network's spreading activation.

        Args:
            concept_name: The concept to prime.
            amount: Priming strength [0..1].
        """
        self._decay_now()
        cid = concept_name
        if self._network is not None:
            resolved = self._network._resolve(concept_name)
            if resolved:
                cid = resolved

        current, _ = self._primed.get(cid, (0.0, 0.0))
        self._primed[cid] = (min(1.0, current + amount), time.time())

        # Spread priming to neighbours via the concept network.
        if self._network is not None:
            spread = self._network.spread_activation(
                [cid], amount=amount * 0.5, depth=self._spread_depth
            )
            for neighbor_id, activation in spread.items():
                if neighbor_id == cid:
                    continue
                n_current, _ = self._primed.get(neighbor_id, (0.0, 0.0))
                self._primed[neighbor_id] = (
                    min(1.0, n_current + activation),
                    time.time(),
                )

    def is_primed(self, concept_name: str) -> float:
        """Return the current priming level for a concept [0..1].

        Returns 0.0 if the concept has not been primed or priming has
        fully decayed.
        """
        self._decay_now()
        cid = concept_name
        if self._network is not None:
            resolved = self._network._resolve(concept_name)
            if resolved:
                cid = resolved
        level, _ = self._primed.get(cid, (0.0, 0.0))
        return level

    def get_all_primed(self, threshold: float = 0.01) -> dict[str, float]:
        """Return all concepts currently above the priming threshold."""
        self._decay_now()
        return {cid: level for cid, (level, _) in self._primed.items() if level >= threshold}

    def _decay_now(self) -> None:
        """Apply time-based exponential decay to all priming levels."""
        now = time.time()
        elapsed = now - self._last_decay
        if elapsed <= 0:
            return
        self._last_decay = now
        # Exponential decay: level *= e^(-rate * elapsed)
        factor = math.exp(-self._decay_rate * elapsed)
        to_remove: list[str] = []
        for cid, (level, ts) in self._primed.items():
            new_level = level * factor
            if new_level < 0.005:
                to_remove.append(cid)
            else:
                self._primed[cid] = (new_level, ts)
        for cid in to_remove:
            del self._primed[cid]

    @property
    def primed_count(self) -> int:
        """Number of concepts currently primed above threshold."""
        self._decay_now()
        return sum(1 for level, _ in self._primed.values() if level > 0.005)


# ─── Item 8: Attractor networks (Hopfield / CA3 auto-association) ────


@dataclass(slots=True)
class AttractorPattern:
    """A stored pattern in an attractor network.

    Attributes:
        id: Human-readable identifier for the pattern.
        vector: Bipolar (+1/-1) or binary (0/1) feature vector.
        timestamp: When the pattern was stored (ms).
        retrieval_count: How many times the pattern was retrieved.
    """

    id: str
    vector: list[float] = field(default_factory=list)
    timestamp: int = 0
    retrieval_count: int = 0


class AttractorNetwork:
    """A Hopfield network for pattern completion and auto-association.

    This provides a biologically plausible association mechanism on
    top of the Rust-side SimHash store. The Hopfield network (Hopfield,
    1982) stores patterns as attractor basins in a recurrent neural
    space. Given a partial or noisy cue, the network settles into the
    nearest stored pattern — *pattern completion*.

    The CA3 region of the hippocampus is modelled as an auto-associative
    network: it can retrieve a full memory from a partial cue, and
    similar memories activate each other (Marr, 1971; Rolls & Treves,
    1998).

    This implementation uses the classic Hopfield update rule with
    asynchronous dynamics. Patterns are stored via Hebbian learning
    (outer product of each pattern with itself, summed into the
    weight matrix). Retrieval iterates the update rule until the state
    stabilises (converges to an attractor).

    Capacity: ~0.138 * N patterns for N units (Amit et al., 1985).
    """

    def __init__(self, size: int = 64) -> None:
        """Initialize the attractor network with a square weight matrix."""
        self._size = size
        # Weight matrix W[i][j] — symmetric, zero diagonal.
        self._weights: list[list[float]] = [[0.0] * size for _ in range(size)]
        self._patterns: dict[str, AttractorPattern] = {}
        self._retrieval_count: int = 0
        self._storage_count: int = 0

    @property
    def size(self) -> int:
        """Number of units in the network."""
        return self._size

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return len(self._patterns)

    @property
    def capacity(self) -> int:
        """Theoretical storage capacity (~0.138 * N)."""
        return max(1, int(0.138 * self._size))

    def _to_bipolar(self, vector: list[float]) -> list[float]:
        """Convert a binary/bipolar vector to bipolar (+1/-1), padded/truncated."""
        out = [0.0] * self._size
        for i in range(min(len(vector), self._size)):
            out[i] = 1.0 if vector[i] >= 0.5 else -1.0
        return out

    def store(self, pattern_id: str, vector: list[float]) -> None:
        """Store a pattern using Hebbian learning (outer product rule).

        Args:
            pattern_id: Identifier for the pattern.
            vector: Feature vector (values ≥0.5 map to +1, else -1).
                Will be padded/truncated to the network size.
        """
        bipolar = self._to_bipolar(vector)
        # Hebbian: W += (1/N) * x * x^T
        n = self._size
        scale = 1.0 / n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                self._weights[i][j] += scale * bipolar[i] * bipolar[j]
        self._patterns[pattern_id] = AttractorPattern(
            id=pattern_id,
            vector=bipolar,
            timestamp=int(time.time() * 1000),
        )
        self._storage_count += 1

    def retrieve(
        self,
        cue: list[float],
        max_iterations: int = 20,
    ) -> tuple[list[float], str | None]:
        """Pattern completion — retrieve the nearest stored pattern.

        Given a partial or noisy cue, iterate the Hopfield update rule
        until the state converges to a stored attractor (or the
        iteration limit is reached).

        Args:
            cue: Partial/noisy feature vector.
            max_iterations: Maximum update iterations.

        Returns:
            A tuple of (converged_vector, matched_pattern_id). The
            pattern id is ``None`` if no stored pattern matches.
        """
        state = self._to_bipolar(cue)
        n = self._size
        for _ in range(max_iterations):
            new_state = list(state)
            changed = False
            # Asynchronous updates (random order not needed for
            # convergence guarantees on symmetric weights).
            for i in range(n):
                field_sum = sum(self._weights[i][j] * state[j] for j in range(n))
                new_val = 1.0 if field_sum >= 0 else -1.0
                if new_val != new_state[i]:
                    new_state[i] = new_val
                    changed = True
            state = new_state
            if not changed:
                break

        # Match against stored patterns.
        match_id = self._match_pattern(state)
        if match_id is not None:
            self._retrieval_count += 1
            self._patterns[match_id].retrieval_count += 1
        return state, match_id

    def _match_pattern(self, state: list[float], threshold: float = 0.9) -> str | None:
        """Find the stored pattern most similar to ``state``."""
        best_id: str | None = None
        best_sim = -1.0
        for pid, pattern in self._patterns.items():
            matches = sum(1 for i in range(self._size) if state[i] == pattern.vector[i])
            sim = matches / self._size
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        if best_sim >= threshold:
            return best_id
        return None

    def auto_associate(self, cue_id: str, max_results: int = 5) -> list[str]:
        """CA3 auto-association: find patterns similar to a stored one.

        Similar memories activate each other in CA3. This returns the
        IDs of stored patterns whose overlap with the given pattern
        exceeds a threshold (excluding the query itself).

        Args:
            cue_id: The ID of a stored pattern to match against.
            max_results: Maximum number of similar patterns to return.
                Without this cap, a low threshold returns hundreds of
                patterns, each triggering an IPC episode fetch that
                blocks the think() pipeline.

        Returns:
            List of similar pattern IDs (excluding ``cue_id``), sorted
            by similarity (most similar first).
        """
        pattern = self._patterns.get(cue_id)
        if pattern is None:
            return []
        scored: list[tuple[float, str]] = []
        for pid, other in self._patterns.items():
            if pid == cue_id:
                continue
            matches = sum(1 for i in range(self._size) if pattern.vector[i] == other.vector[i])
            sim = matches / self._size
            if sim > 0.75:
                scored.append((sim, pid))
        # Sort by similarity descending, take the top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [pid for _, pid in scored[:max_results]]

    @property
    def retrieval_count(self) -> int:
        """Total number of successful pattern retrievals."""
        return self._retrieval_count


# ─── Item 9: Spreading activation (Collins & Loftus, 1975) ───────────


class SpreadingActivation:
    """Graded spreading activation over a concept network.

    Implements the Collins & Loftus (1975) semantic network model:
    when a concept is activated, activation spreads to related
    concepts with decay over distance. The amount of spread depends
    on the relationship strength. Activation also decays over time.

    This class wraps the concept network's existing
    ``spread_activation`` method, adding *temporal dynamics* —
    activation levels persist and decay over real time, so recently
    activated concepts remain partially primed. This models the
    sustained activation seen in semantic processing (the N400
    ERP component reflects residual activation).

    The activation map is keyed by concept ID and decays exponentially
    between updates.
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        decay_rate: float = 0.05,
        spread_decay: float = 0.5,
        max_depth: int = 3,
    ) -> None:
        """Initialize spreading activation with decay and depth parameters."""
        self._network = network
        self._decay_rate = decay_rate  # per second
        self._spread_decay = spread_decay
        self._max_depth = max_depth
        # concept_id → (activation, last_update_timestamp)
        self._activation: dict[str, tuple[float, float]] = {}
        self._last_decay: float = time.time()

    def activate(self, concept_name: str, amount: float = 0.3) -> dict[str, float]:
        """Activate a concept and spread activation to its neighbours.

        Args:
            concept_name: The concept to activate.
            amount: Initial activation strength [0..1].

        Returns:
            Dict of all activated concepts and their levels after
            spreading.
        """
        self._decay_now()
        cid = concept_name
        if self._network is not None:
            resolved = self._network._resolve(concept_name)
            if resolved:
                cid = resolved

        current, _ = self._activation.get(cid, (0.0, 0.0))
        self._activation[cid] = (min(1.0, current + amount), time.time())

        # Spread through the network.
        if self._network is not None:
            spread = self._network.spread_activation([cid], amount=amount, depth=self._max_depth)
            for neighbor_id, act in spread.items():
                n_current, _ = self._activation.get(neighbor_id, (0.0, 0.0))
                self._activation[neighbor_id] = (
                    min(1.0, max(n_current, act)),
                    time.time(),
                )
        return self.get_all()

    def get_activation(self, concept_name: str) -> float:
        """Return the current activation level for a concept [0..1]."""
        self._decay_now()
        cid = concept_name
        if self._network is not None:
            resolved = self._network._resolve(concept_name)
            if resolved:
                cid = resolved
        level, _ = self._activation.get(cid, (0.0, 0.0))
        return level

    def get_all(self, threshold: float = 0.01) -> dict[str, float]:
        """Return all concepts currently above the activation threshold."""
        self._decay_now()
        # Snapshot the items to avoid "dictionary changed size during
        # iteration" if activation spreads during the comprehension.
        return {
            cid: level
            for cid, (level, _) in list(self._activation.items())
            if level >= threshold
        }

    def _decay_now(self) -> None:
        """Apply time-based exponential decay to all activation levels."""
        now = time.time()
        elapsed = now - self._last_decay
        if elapsed <= 0:
            return
        self._last_decay = now
        factor = math.exp(-self._decay_rate * elapsed)
        to_remove: list[str] = []
        for cid, (level, ts) in self._activation.items():
            new_level = level * factor
            if new_level < 0.005:
                to_remove.append(cid)
            else:
                self._activation[cid] = (new_level, ts)
        for cid in to_remove:
            del self._activation[cid]

    @property
    def active_count(self) -> int:
        """Number of concepts currently above the activation threshold."""
        self._decay_now()
        return sum(1 for level, _ in self._activation.values() if level > 0.005)

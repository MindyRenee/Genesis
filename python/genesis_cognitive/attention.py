"""Attention system — selective, sustained, executive, and divided attention.

Attention is not a single mechanism but a family of related processes
that select, maintain, bias, and divide cognitive resources. This
module implements four attention types, each grounded in distinct
neural systems:

# Four attention types

1. **Selective attention** (Posner & Petersen, 1990): Filters inputs
   to focus on one while suppressing others. Implemented via a
   spotlight that amplifies attended concepts and suppresses
   unattended ones. Neural basis: frontoparietal network, particularly
   the intraparietal sulcus (IPS) and frontal eye fields (FEF).

2. **Sustained attention** (vigilance): Maintains focus on a target
   over time. Activation decays if not refreshed, modeling the
   vigilance decrement — the decline in detection rate over prolonged
   monitoring (Parasuraman et al., 1998). Neural basis: right
   hemisphere frontoparietal network, locus coeruleus-noradrenergic
   system.

3. **Executive attention** (top-down biasing): Goal-directed attention
   that amplifies goal-relevant information and suppresses
   goal-irrelevant information. This is the anterior cingulate
   cortex (ACC)–dorsolateral prefrontal cortex (DLPFC) attentional
   control network (Miller & Cohen, 2001). The current goal biases
   which concepts get amplified.

4. **Divided attention**: Splits attentional resources when multiple
   tasks need simultaneous processing. Resources are finite — dividing
   them reduces the allocation to each task, increasing RT and
   decreasing accuracy (Kahneman, 1973; Pashler, 1994). Neural basis:
   prefrontal cortex resource allocation, with dual-task costs
   reflecting limited prefrontal capacity.

# Integration with the concept network

Attention amplifies the activation of attended concepts in the concept
network. When a concept is attended, its activation is boosted; when
attention shifts away, the boost decays. This is the top-down
modulation of semantic processing — attended concepts are processed
faster and deeper (Posner & Snyder, 1975).

References:
- Posner, M. I., & Petersen, S. E. (1990). The attention system of
  the human brain. Annual Review of Neuroscience.
- Miller, E. K., & Cohen, J. D. (2001). An integrative theory of
  prefrontal cortex function. Annual Review of Neuroscience.
- Parasuraman, R., et al. (1998). Vigilance: The problem of sustaining
  attention. In Parasuraman (Ed.), The Attentive Brain.
- Kahneman, D. (1973). Attention and Effort. Prentice-Hall.
- Pashler, H. (1994). Dual-task interference in simple tasks.
  Psychological Bulletin.
- Posner, M. I., & Snyder, C. R. R. (1975). Attention and cognitive
  control. In Solso (Ed.), Information Processing and Cognition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .concepts import ConceptNetwork

__all__ = ["AttentionFocus", "AttentionSystem", "AttentionType"]

logger = logging.getLogger(__name__)

# Function words that should not capture executive attention when a goal
# is parsed into concept-like words. These carry no semantic weight for
# goal-directed biasing and would otherwise waste limited focus capacity
# (Cowan's K≈4), evicting the goal's meaningful content. This is a cognitive
# filter on attention allocation, not speech content.
_GOAL_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the",
        "and", "or", "but", "nor", "so", "yet",
        "of", "to", "in", "on", "at", "by", "for", "with", "from",
        "about", "into", "onto", "over", "under", "as", "than",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those",
        "it", "its",
    }
)


class AttentionType(Enum):
    """The type of attention currently engaged.

    Different attention types engage different neural systems and
    have different effects on processing. Knowing which type is
    active helps the cognition engine understand *how* it's attending,
    not just *what* it's attending to.
    """

    SELECTIVE = "selective"  # spotlight on one input, suppress others
    SUSTAINED = "sustained"  # maintain focus over time
    EXECUTIVE = "executive"  # top-down goal-directed biasing
    DIVIDED = "divided"  # splitting resources across tasks


@dataclass(slots=True)
class AttentionFocus:
    """A single focus of attention.

    Tracks what is being attended, how strongly, and how the focus
    is maintained over time. Each focus has:
    - A target (concept name or input identifier)
    - An intensity (0..1, how strongly attended)
    - A type (which attention mechanism is engaged)
    - A refresh timestamp (for sustained attention decay)
    - An optional associated goal (for executive attention)
    """

    target: str
    intensity: float = 1.0
    attention_type: AttentionType = AttentionType.SELECTIVE
    established_at: int = field(default_factory=lambda: int(time.time() * 1000))
    last_refreshed: int = field(default_factory=lambda: int(time.time() * 1000))
    goal: str = ""

    @property
    def age_ms(self) -> int:
        """How long this focus has been held (milliseconds)."""
        return int(time.time() * 1000) - self.established_at

    @property
    def time_since_refresh_ms(self) -> int:
        """Time since last refresh (for vigilance decay)."""
        return int(time.time() * 1000) - self.last_refreshed


class AttentionSystem:
    """Manages attention across four mechanisms.

    The attention system controls what Genesis focuses on and how.
    It integrates with the concept network — attended concepts get
    amplified activation, unattended ones get suppressed.

    # Selective attention

    ``focus_on(target)`` puts a spotlight on one target, amplifying it
    and suppressing others. ``suppress(target)`` explicitly suppresses
    a target. This is the "spotlight" model of selective attention
    (Posner, 1980).

    # Sustained attention

    Foci decay over time if not refreshed. ``refresh(target)`` restores
    a focus to full intensity. ``tick(dt)`` applies decay, modeling the
    vigilance decrement. The decay rate is slower for executive
    (goal-directed) attention than for passive selective attention,
    because goal-maintenance in PFC sustains activation (Curtis &
    D'Esposito, 2003).

    # Executive attention

    ``set_goal(goal)`` establishes a top-down bias. Concepts related to
    the goal get amplified via the concept network's spreading
    activation. This is the biased competition model (Desimone &
    Duncan, 1995) — goals bias the competition among representations
    in favor of goal-relevant ones.

    # Divided attention

    ``divide_attention(targets)`` splits resources across multiple
    targets. Each target gets a fraction of total attentional capacity.
    The more targets, the less each gets — modeling the dual-task cost
    (Pashler, 1994).
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        decay_rate: float = 0.02,
        suppression_strength: float = 0.3,
        amplification_strength: float = 0.4,
        max_foci: int = 4,
    ) -> None:
        """Initialize the attention system.

        Args:
            network: The concept network to modulate. Attention
                amplifies/suppresses concept activation in the network.
                Can be None if network integration is not needed.
            decay_rate: How much focus intensity decays per tick.
                Default 0.02. Models vigilance decrement.
            suppression_strength: How much unattended concepts are
                suppressed (0..1). Default 0.3.
            amplification_strength: How much attended concepts are
                amplified (0..1). Default 0.4.
            max_foci: Maximum simultaneous attention foci. Default 4
                (matching Cowan's K≈4).
        """
        self.network = network
        self.decay_rate = decay_rate
        self.suppression_strength = suppression_strength
        self.amplification_strength = amplification_strength
        self.max_foci = max_foci
        self._foci: dict[str, AttentionFocus] = {}
        self._current_goal: str = ""
        self._suppressed: set[str] = set()
        self._attention_boosts: dict[str, float] = {}

    # ─── Selective attention ──────────────────────────────────────

    def focus_on(
        self,
        target: str,
        intensity: float = 1.0,
        attention_type: AttentionType = AttentionType.SELECTIVE,
    ) -> AttentionFocus:
        """Focus selective attention on a target.

        Amplifies the target's activation in the concept network and
        suppresses other active foci (spotlight model). If the maximum
        number of foci is reached, the weakest existing focus is
        replaced.

        Args:
            target: What to attend to (concept name or input ID).
            intensity: How strongly to attend (0..1). Default 1.0.
            attention_type: Which attention mechanism to engage.

        Returns:
            The AttentionFocus that was established.
        """
        intensity = max(0.0, min(1.0, intensity))

        # Enforce capacity — remove weakest if at max
        if target not in self._foci and len(self._foci) >= self.max_foci:
            weakest = min(self._foci, key=lambda t: self._foci[t].intensity)
            self._remove_focus(weakest)

        focus = AttentionFocus(
            target=target,
            intensity=intensity,
            attention_type=attention_type,
            goal=self._current_goal,
        )
        self._foci[target] = focus

        # Amplify the attended concept in the network
        self._amplify_concept(target, intensity)

        return focus

    def suppress(self, target: str) -> None:
        """Explicitly suppress a target (negative attention).

        Suppresses the target's activation in the concept network.
        This is the complement of focus_on — selective attention
        involves both amplification of the attended and suppression
        of the unattended.

        Args:
            target: What to suppress (concept name or input ID).
        """
        self._suppressed.add(target)
        self._suppress_concept(target)

    # ─── Sustained attention ──────────────────────────────────────

    def refresh(self, target: str) -> None:
        """Refresh a focus to full intensity (sustained attention).

        Without refresh, foci decay over time (vigilance decrement).
        Refreshing restores the focus to its original intensity,
        modeling the top-up that occurs when attention is actively
        maintained.

        Args:
            target: The focus target to refresh.
        """
        focus = self._foci.get(target)
        if focus:
            focus.intensity = 1.0
            focus.last_refreshed = int(time.time() * 1000)
            self._amplify_concept(target, focus.intensity)

    def tick(self, dt: float = 1.0) -> list[str]:
        """Apply attention decay and remove expired foci.

        Sustained attention decays if not refreshed. Executive
        (goal-directed) attention decays more slowly, modeling PFC
        goal maintenance. Foci that decay below a minimum threshold
        are removed.

        Args:
            dt: Time step.

        Returns:
            List of targets whose foci expired (fell below threshold).
        """
        expired: list[str] = []
        min_intensity = 0.1

        for target, focus in list(self._foci.items()):
            # Executive attention decays slower (PFC goal maintenance)
            rate = self.decay_rate
            if focus.attention_type == AttentionType.EXECUTIVE:
                rate *= 0.5  # half the decay rate

            focus.intensity *= 1.0 - rate * dt
            # Clamp to non-negative — a large dt can make the decay
            # factor negative, flipping intensity's sign.
            if focus.intensity < 0.0:
                focus.intensity = 0.0

            if focus.intensity < min_intensity:
                expired.append(target)
                self._remove_focus(target)
            else:
                # Update network amplification to match current intensity
                self._amplify_concept(target, focus.intensity)

        return expired

    # ─── Executive attention ──────────────────────────────────────

    def set_goal(self, goal: str) -> None:
        """Set a goal for executive (top-down) attention.

        The goal biases attention toward goal-relevant concepts.
        Concepts semantically related to the goal in the concept
        network get amplified via spreading activation. This is the
        biased competition model (Desimone & Duncan, 1995).

        The goal string is tokenized into concept-like words. Function
        words (articles, prepositions, conjunctions, etc.) are filtered
        out so they do not waste limited focus capacity. Only words that
        correspond to existing concepts in the network capture
        executive attention, and at most ``max_foci`` of them — preventing
        the goal's own words from evicting each other. Executive foci
        from a *previous* goal are cleared first, so goal-directed
        attention tracks the current goal rather than accumulating
        stale biases.

        Args:
            goal: The goal description (e.g., "answer the question
                about cognition").
        """
        # Replace the previous goal's executive bias so attention tracks
        # the current goal, not the union of all past goals.
        if self._current_goal and self._current_goal != goal:
            for target in list(self._foci.keys()):
                focus = self._foci[target]
                if focus.attention_type == AttentionType.EXECUTIVE:
                    self._remove_focus(target)

        self._current_goal = goal

        if not (self.network and goal):
            return

        # Extract concept-like words, filtering function words that carry
        # no goal-relevant semantic weight.
        seen: set[str] = set()
        goal_words: list[str] = []
        for raw in goal.split():
            word = raw.strip(".,!?;:\"'()[]")
            if not word or word.lower() in _GOAL_STOPWORDS:
                continue
            if word in seen:
                continue
            seen.add(word)
            goal_words.append(word)

        # Only existing concepts can be amplified, and capacity is
        # finite — cap at max_foci so the goal's words never evict each
        # other mid-loop.
        for word in goal_words:
            if len(self._foci) >= self.max_foci:
                break
            if self.network.get_concept(word):
                self.focus_on(
                    word,
                    intensity=0.6,
                    attention_type=AttentionType.EXECUTIVE,
                )

    def clear_goal(self) -> None:
        """Clear the current goal and remove executive attention foci."""
        self._current_goal = ""
        # Remove executive attention foci
        for target in list(self._foci.keys()):
            if self._foci[target].attention_type == AttentionType.EXECUTIVE:
                self._remove_focus(target)

    @property
    def current_goal(self) -> str:
        """The current goal guiding executive attention."""
        return self._current_goal

    # ─── Divided attention ────────────────────────────────────────

    def divide_attention(self, targets: list[str]) -> dict[str, float]:
        """Divide attention across multiple targets.

        Resources are split equally (or by weight) across targets.
        Each target gets a fraction of total capacity. The more
        targets, the less each gets — modeling the dual-task cost
        (Pashler, 1994).

        Attentional capacity is finite (``max_foci``, Cowan's K≈4). If
        more targets are requested than capacity can hold, only the
        first ``max_foci`` (after de-duplication) are attended; the rest
        receive no allocation. The returned dict reports only targets
        that actually received attention — it never claims an allocation
        for a target that was silently evicted.

        Args:
            targets: List of targets to divide attention across.
                Duplicates are collapsed to a single focus.

        Returns:
            Dict mapping each *attended* target to its allocated
            intensity. Unattended targets (beyond capacity) are absent.
        """
        if not targets:
            return {}

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique_targets: list[str] = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)

        # Capacity is a hard limit (Cowan's K). Only what can be held
        # can be divided among.
        attended_targets = unique_targets[: self.max_foci]

        # Total capacity is 1.0; divide among the actually-attended
        # targets. But division is not perfectly efficient — there's a
        # cost to splitting (dual-task cost). Model this with a
        # superlinear allocation: each target gets (1/n)^1.2 of capacity,
        # so the total across n targets is n * (1/n)^1.2 = n^{-0.2} < 1.0.
        n = len(attended_targets)
        per_target = (1.0 / n) ** 1.2

        allocations: dict[str, float] = {}
        for target in attended_targets:
            self.focus_on(
                target,
                intensity=per_target,
                attention_type=AttentionType.DIVIDED,
            )
            allocations[target] = per_target

        return allocations

    # ─── Querying attention state ─────────────────────────────────

    @property
    def foci(self) -> list[AttentionFocus]:
        """Current attention foci."""
        return list(self._foci.values())

    @property
    def focused_targets(self) -> list[str]:
        """Targets currently being attended to."""
        return list(self._foci.keys())

    def get_focus(self, target: str) -> AttentionFocus | None:
        """Get the attention focus for a specific target."""
        return self._foci.get(target)

    @property
    def dominant_focus(self) -> AttentionFocus | None:
        """The strongest current focus (the spotlight center)."""
        if not self._foci:
            return None
        return max(self._foci.values(), key=lambda f: f.intensity)

    @property
    def is_focused(self) -> bool:
        """Whether any attention is currently engaged."""
        return len(self._foci) > 0

    @property
    def suppressed_targets(self) -> set[str]:
        """Targets currently being suppressed."""
        return set(self._suppressed)

    def is_attended(self, target: str) -> bool:
        """Check if a target is currently being attended to."""
        return target in self._foci

    def is_suppressed(self, target: str) -> bool:
        """Check if a target is currently being suppressed."""
        return target in self._suppressed

    # ─── Internal helpers ─────────────────────────────────────────

    def _amplify_concept(self, target: str, intensity: float) -> None:
        """Amplify a concept's activation in the network.

        The boost is applied relative to the concept's base activation
        (without prior attention boosts), not cumulatively. This
        prevents attended concepts from saturating at 1.0 after
        repeated tick() calls.
        """
        if not self.network:
            return
        concept = self.network.get_concept(target)
        if concept:
            # Remove previous attention boost before applying new one
            prev_boost = self._attention_boosts.get(target, 0.0)
            if prev_boost > 0.0 and concept.activation is not None:
                concept.activation = max(0.0, concept.activation - prev_boost)
            # Apply new boost
            boost = self.amplification_strength * intensity
            base = concept.activation if concept.activation is not None else 0.0
            concept.activation = min(1.0, base + boost)
            self._attention_boosts[target] = boost

    def _suppress_concept(self, target: str) -> None:
        """Suppress a concept's activation in the network."""
        if not self.network:
            return
        concept = self.network.get_concept(target)
        if concept:
            if concept.activation is None:
                concept.activation = 0.0
            concept.activation *= 1.0 - self.suppression_strength

    def _remove_focus(self, target: str) -> None:
        """Remove a focus and clean up."""
        # Remove the attention boost from the concept's activation
        prev_boost = self._attention_boosts.pop(target, 0.0)
        if prev_boost > 0.0 and self.network:
            concept = self.network.get_concept(target)
            if concept and concept.activation is not None:
                concept.activation = max(0.0, concept.activation - prev_boost)
        self._foci.pop(target, None)

    def clear(self) -> None:
        """Clear all attention foci and suppressions."""
        # Remove all attention boosts from concept activations
        if self.network:
            for target, prev_boost in self._attention_boosts.items():
                concept = self.network.get_concept(target)
                if concept and concept.activation is not None:
                    concept.activation = max(0.0, concept.activation - prev_boost)
        self._attention_boosts.clear()
        self._foci.clear()
        self._suppressed.clear()
        self._current_goal = ""

    # ─── Global workspace integration ──────────────────────────────

    def receive_broadcast(self, item: Any) -> None:
        """Receive a broadcast from the global workspace.

        This is **attentional capture** — when information ignites the
        global workspace (becomes cognitive), attention is drawn to it.
        In GWT, the broadcast IS the mechanism by which cognitive content
        gains control over attentional resources (Dehaene & Naccache,
        2001). Without this, cognitive content would be broadcast but
        couldn't influence what the system focuses on next.

        The broadcast's topics (from ``item.metadata["topics"]``) are
        focused on with intensity proportional to the broadcast
        activation. This means highly-activated cognitive content
        captures attention more strongly than fringe content.

        Args:
            item: A :class:`WorkspaceItem` carrying the broadcast content.
        """
        topics = item.metadata.get("topics") if item.metadata else None
        if not topics:
            return
        intensity = max(0.0, min(1.0, item.activation * 0.8))
        for topic in topics:
            if topic and isinstance(topic, str):
                self.focus_on(topic, intensity=intensity)

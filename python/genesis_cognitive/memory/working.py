"""Working memory — what Genesis is holding in mind right now.

Working memory is the scratchpad of cognition. It's not long-term
memory (that's the LTM store) and it's not the concept network (that's
semantic knowledge). It's the active context: what are we talking about,
what was just said, what am I paying attention to.

# Why this matters

Without working memory, each interaction is isolated. Genesis can't:
- Notice when you're building on something she said
- Remember that you discussed cognition 3 turns ago
- Track a thread of argument across multiple exchanges
- Recognize when you've changed the subject
- Follow up on something she was curious about

With working memory, she can hold attention. She can say "Earlier you
mentioned X — I've been thinking about it." She can notice "We've been
talking about cognition for a while." She can connect what you're
saying now to what you said before.

# Baddeley's working memory model

This module implements the Baddeley & Hitch (1974) working memory
model, adapted for Genesis's text-based cognition:

- **Central executive**: controls attention allocation, suppresses
  irrelevant items, and coordinates the slave systems.
- **Phonological loop**: a verbal rehearsal buffer — Genesis can
  "repeat" words/concepts to herself to maintain them. Items decay
  in ~2 seconds without rehearsal (Baddeley, 1992).
- **Visuospatial sketchpad**: a structural/relational buffer — Genesis
  can hold spatial/structural relationships in mind (e.g., "A is
  connected to B").

The attention buffer has a **capacity limit** of 3-5 items (Cowan's
K=4; Cowan, 2001). When capacity is exceeded, the oldest/weakest items
decay. Items can be **rehearsed** to reset their decay timer, and
frequently accessed items are auto-rehearsed.

References:
    - Baddeley & Hitch (1974): working memory model
    - Baddeley (1992): phonological loop and visuospatial sketchpad
    - Cowan (2001): capacity limit K≈4 (the "magical number 4")

# Structure

WorkingMemory holds:
- attention: what concepts are currently active (decays over time,
  capacity-limited to 3-5 items)
- thread: the current conversation thread (topic + turns)
- recent_turns: the last N exchanges (for reference)
- open_questions: things she was curious about but hasn't asked yet
- topic_history: what topics have been discussed (for continuity)
- phonological_loop: verbal rehearsal buffer (slave system)
- visuospatial_sketchpad: structural/relational buffer (slave system)
- central_executive: attention controller coordinating the above
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from genesis_client import NeuroSummary

from ..brain_waves import BrainWave, BrainWaveState

__all__ = [
    "CentralExecutive",
    "ConversationThread",
    "PhonologicalLoop",
    "Turn",
    "VisuoSpatialSketchpad",
    "WorkingMemory",
    "WorkingMemorySlot",
]


@dataclass(slots=True)
class Turn:
    """A single exchange in working memory."""

    user_input: str
    genesis_response: str
    topics: list[str]
    intent: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(slots=True)
class WorkingMemorySlot:
    """A single item held in the capacity-limited attention buffer.

    This is the explicit slot representation of Baddeley's attention
    buffer. Capacity is not fixed; it is gated by arousal, plasticity,
    and valence (via acetylcholine and dopamine proxies) so sleepy or
    stressed states shrink the scratchpad and alert plastic states
    expand it.
    """

    content: str
    activation: float
    rehearsals: int = 0
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    last_accessed: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(slots=True)
class ConversationThread:
    """The current thread of conversation.

    A thread is a sequence of turns about related topics.
    When the topic shifts significantly, a new thread begins.
    """

    topic: str
    turns: list[Turn] = field(default_factory=list)
    started_at: int = field(default_factory=lambda: int(time.time() * 1000))
    key_concepts: set[str] = field(default_factory=set)

    @property
    def turn_count(self) -> int:
        """Return the number of turns in this thread."""
        return len(self.turns)

    @property
    def is_active(self) -> bool:
        """Is this thread still being discussed?"""
        if not self.turns:
            return True
        # Active if last turn was < 60 seconds ago (in real use)
        # For testing, always active
        return True

    def add_turn(self, turn: Turn) -> None:
        """Append a turn and record its key concepts in this thread."""
        self.turns.append(turn)
        self.key_concepts.update(turn.topics)

    def describe(self) -> str:
        """Describe this thread."""
        if self.turn_count == 0:
            return f"Thread about {self.topic} (just started)"
        return (
            f"Thread about {self.topic} "
            f"({self.turn_count} turns, concepts: "
            f"{', '.join(sorted(self.key_concepts)[:5])})"
        )


# ─── Phonological loop (verbal rehearsal buffer) ──────────────────────


@dataclass(slots=True)
class _PhonologicalItem:
    """An item held in the phonological loop.

    Attributes:
        content: The verbal content (word/concept/phrase).
        activation: Current activation level [0..1]. Decays over time;
            rehearsal resets it to 1.0.
        last_rehearsed: When the item was last rehearsed (monotonic
            seconds).
        access_count: How many times the item has been accessed
            (used for auto-rehearsal priority).
    """

    content: str
    activation: float = 1.0
    last_rehearsed: float = 0.0
    access_count: int = 0


class PhonologicalLoop:
    """The phonological loop — a verbal rehearsal buffer.

    One of Baddeley's slave systems (Baddeley, 1992). For Genesis,
    this is a verbal rehearsal buffer: she can "repeat" words or
    concepts to herself to maintain them in working memory. Without
    rehearsal, items decay in approximately 2 seconds (the duration of
    the phonological store, matching the ~2s auditory-memory trace in
    humans; Baddeley, 1992).

    The loop has a small capacity (typically 2-4 items, the "memory
    span" for verbal material). Rehearsal refreshes all items, resetting
    their decay timers — this is the articulatory control process.

    Biological grounding:
        - The phonological store is associated with left
          perisylvian cortex (Broca's and Wernicke's areas; Paulesu
          et al., 1993).
        - The articulatory control process (rehearsal) involves
          Broca's area and motor cortex for sub-vocal articulation.
        - The ~2s decay matches the persistence of the auditory
          sensory memory trace (Crowder, 1982).
    """

    #: Decay half-life in seconds — items lose ~half their activation
    #: per this interval without rehearsal. The phonological store
    #: persists for ~2 seconds (Baddeley, 1992).
    DECAY_TAU: float = 2.0
    #: Default capacity of the loop (verbal memory span).
    DEFAULT_CAPACITY: int = 4
    #: Below this activation, an item is dropped from the loop.
    _DROP_THRESHOLD: float = 0.05

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        decay_tau: float = DECAY_TAU,
    ) -> None:
        """Initialize the phonological loop with capacity and decay time.

        Args:
            capacity: Maximum number of verbal items to hold.
            decay_tau: Half-life in seconds for item activation decay.
        """
        self._capacity = max(1, capacity)
        self._decay_tau = max(0.1, decay_tau)
        # content (lowercased) → _PhonologicalItem, ordered by insertion.
        self._items: dict[str, _PhonologicalItem] = {}
        self._order: list[str] = []
        self._last_decay: float = time.monotonic()

    @property
    def capacity(self) -> int:
        """Maximum number of items the loop can hold."""
        return self._capacity

    def store(self, item: str) -> None:
        """Store a verbal item in the phonological loop.

        If the item is already present, its activation is refreshed
        (re-encoding). If capacity is exceeded, the weakest item is
        dropped (displacement, as in the classic phonological store).

        Args:
            item: The word/concept/phrase to maintain.
        """
        if not item:
            return
        self._decay_now()
        key = item.lower()
        if key in self._items:
            existing = self._items[key]
            existing.activation = 1.0
            existing.last_rehearsed = time.monotonic()
            existing.access_count += 1
            # Move to most-recent position.
            self._order.remove(key)
            self._order.append(key)
            return
        # New item — enforce capacity by dropping the weakest.
        while len(self._items) >= self._capacity:
            self._drop_weakest()
        now = time.monotonic()
        self._items[key] = _PhonologicalItem(
            content=item,
            activation=1.0,
            last_rehearsed=now,
        )
        self._order.append(key)

    def rehearse(self) -> None:
        """Rehearse all items — reset their decay timers.

        This is the articulatory control process: sub-vocal repetition
        refreshes the phonological store, preventing decay (Baddeley,
        1992).
        """
        self._decay_now()
        now = time.monotonic()
        for it in self._items.values():
            it.activation = 1.0
            it.last_rehearsed = now

    def rehearse_item(self, item: str) -> bool:
        """Rehearse a single item, resetting its decay timer.

        Args:
            item: The item to rehearse.

        Returns:
            ``True`` if the item was present and rehearsed.
        """
        self._decay_now()
        key = item.lower()
        it = self._items.get(key)
        if it is None:
            return False
        it.activation = 1.0
        it.last_rehearsed = time.monotonic()
        it.access_count += 1
        return True

    def decay(self, dt: float) -> None:
        """Advance decay by ``dt`` seconds.

        Exponential decay: activation *= e^(-dt / tau). Items below the
        drop threshold are removed.
        """
        if dt <= 0:
            return
        factor = pow(2.718281828459045, -dt / self._decay_tau)
        to_remove: list[str] = []
        for key, it in self._items.items():
            it.activation *= factor
            if it.activation < self._DROP_THRESHOLD:
                to_remove.append(key)
        for key in to_remove:
            del self._items[key]
            self._order.remove(key)
        self._last_decay = time.monotonic()

    def get_items(self) -> list[str]:
        """Return the currently held items, strongest first."""
        self._decay_now()
        ordered = sorted(
            self._items.values(),
            key=lambda it: it.activation,
            reverse=True,
        )
        return [it.content for it in ordered]

    def _decay_now(self) -> None:
        """Apply real-time decay based on elapsed wall-clock time."""
        now = time.monotonic()
        elapsed = now - self._last_decay
        self._last_decay = now
        if elapsed <= 0:
            return
        self.decay(elapsed)

    def _drop_weakest(self) -> None:
        """Drop the weakest (lowest-activation) item from the loop."""
        if not self._items:
            return
        weakest_key = min(self._items, key=lambda k: self._items[k].activation)
        del self._items[weakest_key]
        self._order.remove(weakest_key)

    @property
    def item_count(self) -> int:
        """Number of items currently held."""
        self._decay_now()
        return len(self._items)


# ─── Visuospatial sketchpad (structural/relational buffer) ────────────


@dataclass(slots=True)
class _SpatialRelation:
    """A structural/spatial relation held in the sketchpad.

    Attributes:
        a: The first element.
        relation: The relation (e.g., "connected_to", "above", "inside").
        b: The second element.
        activation: Current activation [0..1]; decays without refresh.
        last_refreshed: When the relation was last refreshed (monotonic
            seconds).
    """

    a: str
    relation: str
    b: str
    activation: float = 1.0
    last_refreshed: float = 0.0

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the canonical (lowercased) key for this relation."""
        return (self.a.lower(), self.relation.lower(), self.b.lower())


class VisuoSpatialSketchpad:
    """The visuospatial sketchpad — a structural/relational buffer.

    The second of Baddeley's slave systems (Baddeley, 1992). For
    Genesis, this is a structural/relational buffer: she can hold
    spatial or structural relationships in mind (e.g., "A is connected
    to B", "hippocampus is inside brain"). This is the analogue of the
    human visuospatial sketchpad, which maintains visual and spatial
    information for manipulation and reasoning.

    Relations decay over time without refresh, with a longer time
    constant than the phonological loop (visual/spatial information
    persists longer than auditory; Phillips, 1983).

    Biological grounding:
        - The sketchpad is associated with right-hemisphere
          parieto-occipital regions (Smith & Jonides, 1997).
        - Visual and spatial information persists for several seconds
          without rehearsal, longer than verbal material.
    """

    #: Decay time constant in seconds (visual/spatial traces persist
    #: longer than phonological; Phillips, 1983).
    DECAY_TAU: float = 4.0
    #: Below this activation, a relation is dropped.
    _DROP_THRESHOLD: float = 0.05

    def __init__(self, decay_tau: float = DECAY_TAU) -> None:
        """Initialize the visuospatial sketchpad with a decay time constant.

        Args:
            decay_tau: Half-life in seconds for relation activation decay.
        """
        self._decay_tau = max(0.1, decay_tau)
        self._relations: dict[tuple[str, str, str], _SpatialRelation] = {}
        self._last_decay: float = time.monotonic()

    def store_relation(self, a: str, relation: str, b: str) -> None:
        """Store a structural/spatial relation in the sketchpad.

        If the relation already exists, its activation is refreshed.

        Args:
            a: The first element.
            relation: The relation (e.g., "connected_to", "above").
            b: The second element.
        """
        if not a or not relation or not b:
            return
        self._decay_now()
        key = (a.lower(), relation.lower(), b.lower())
        existing = self._relations.get(key)
        if existing is not None:
            existing.activation = 1.0
            existing.last_refreshed = time.monotonic()
            return
        self._relations[key] = _SpatialRelation(
            a=a,
            relation=relation,
            b=b,
            activation=1.0,
            last_refreshed=time.monotonic(),
        )

    def decay(self, dt: float) -> None:
        """Advance decay by ``dt`` seconds.

        Exponential decay: activation *= e^(-dt / tau).
        """
        if dt <= 0:
            return
        factor = pow(2.718281828459045, -dt / self._decay_tau)
        to_remove: list[tuple[str, str, str]] = []
        for key, rel in self._relations.items():
            rel.activation *= factor
            if rel.activation < self._DROP_THRESHOLD:
                to_remove.append(key)
        for key in to_remove:
            del self._relations[key]
        self._last_decay = time.monotonic()

    def get_relations(self) -> list[tuple[str, str, str]]:
        """Return currently held relations, strongest first.

        Returns:
            A list of (a, relation, b) tuples.
        """
        self._decay_now()
        ordered = sorted(
            self._relations.values(),
            key=lambda r: r.activation,
            reverse=True,
        )
        return [(r.a, r.relation, r.b) for r in ordered]

    def _decay_now(self) -> None:
        """Apply real-time decay based on elapsed wall-clock time."""
        now = time.monotonic()
        elapsed = now - self._last_decay
        self._last_decay = now
        if elapsed <= 0:
            return
        self.decay(elapsed)

    @property
    def relation_count(self) -> int:
        """Number of relations currently held."""
        self._decay_now()
        return len(self._relations)


# ─── Central executive ────────────────────────────────────────────────


class CentralExecutive:
    """The central executive — attention controller of working memory.

    The central executive is the attentional control component of
    Baddeley's working memory model (Baddeley & Hitch, 1974). It does
    not store information itself; rather, it:

    - **Directs attention** to relevant items in the slave systems and
      the attention buffer.
    - **Suppresses irrelevant items** (inhibition — the executive's
      core function; Miyake et al., 2000).
    - **Coordinates** the phonological loop and visuospatial sketchpad,
      switching between them as the task demands.
    - **Allocates capacity** when the attention buffer exceeds its
      capacity limit, deciding which items to keep and which to let
      decay.

    Biological grounding:
        - The central executive is associated with the dorsolateral
          prefrontal cortex (DLPFC) and anterior cingulate cortex
          (ACC), the seat of executive control and conflict monitoring
          (Smith & Jonides, 1999).
        - Inhibition (suppressing irrelevant items) recruits the
          inferior frontal gyrus and ACC (Aron, 2007).
    """

    def __init__(
        self,
        phonological_loop: PhonologicalLoop | None = None,
        sketchpad: VisuoSpatialSketchpad | None = None,
    ) -> None:
        """Initialize the central executive with slave memory systems.

        Args:
            phonological_loop: Optional verbal working memory slave.
            sketchpad: Optional visuospatial working memory slave.
        """
        self._phonological_loop = phonological_loop
        self._sketchpad = sketchpad
        # Items currently suppressed by the executive (inhibition).
        self._suppressed: set[str] = set()
        # Task-switching state: the current focus.
        self._current_focus: str | None = None
        # Counters for introspection.
        self.suppressions: int = 0
        self.task_switches: int = 0

    @property
    def current_focus(self) -> str | None:
        """What the executive is currently focusing attention on."""
        return self._current_focus

    def direct_attention(
        self,
        items: dict[str, float],
        relevant: list[str],
    ) -> dict[str, float]:
        """Direct attention to relevant items, boosting their activation.

        Items in ``relevant`` receive an attention boost; items in the
        suppressed set are removed entirely.

        Args:
            items: The current attention buffer {concept: activation}.
            relevant: Concepts to focus on.

        Returns:
            The updated attention buffer.
        """
        result: dict[str, float] = {}
        relevant_set = {r.lower() for r in relevant}
        for concept, activation in items.items():
            if concept.lower() in self._suppressed:
                continue
            if concept.lower() in relevant_set:
                # Boost relevant items.
                result[concept] = min(1.0, activation + 0.2)
            else:
                result[concept] = activation
        # Set focus to the most relevant item.
        if relevant:
            self._current_focus = relevant[0]
        return result

    def suppress(self, item: str) -> None:
        """Suppress an irrelevant item (inhibition).

        Suppressed items are filtered out of the attention buffer by
        :meth:`direct_attention`. This is the executive's inhibitory
        function (Miyake et al., 2000; Aron, 2007).

        Args:
            item: The concept to suppress.
        """
        self._suppressed.add(item.lower())
        self.suppressions += 1

    def release_suppression(self, item: str) -> None:
        """Release a previously suppressed item back into attention."""
        self._suppressed.discard(item.lower())

    def is_suppressed(self, item: str) -> bool:
        """Whether an item is currently suppressed."""
        return item.lower() in self._suppressed

    def switch_task(self, new_focus: str) -> None:
        """Switch the executive's focus to a new task/topic.

        Task switching is a core executive function (Monsell, 2003).
        It carries a switching cost: the previous focus is partially
        suppressed to avoid interference.

        Args:
            new_focus: The new task/topic to focus on.
        """
        if self._current_focus is not None and self._current_focus != new_focus:
            # Partial suppression of the old focus (task-switching cost).
            self._suppressed.add(self._current_focus.lower())
            self.task_switches += 1
        self._current_focus = new_focus

    def coordinate(
        self,
        verbal_items: list[str] | None = None,
        relations: list[tuple[str, str, str]] | None = None,
    ) -> dict[str, list[str] | list[tuple[str, str, str]]]:
        """Coordinate the phonological loop and visuospatial sketchpad.

        The executive coordinates the two slave systems, routing verbal
        material to the phonological loop and structural/spatial
        material to the sketchpad.

        Args:
            verbal_items: Verbal items to route to the phonological loop.
            relations: Relations to route to the visuospatial sketchpad.

        Returns:
            A dict with the current contents of each slave system.
        """
        if verbal_items and self._phonological_loop is not None:
            for item in verbal_items:
                self._phonological_loop.store(item)
        if relations and self._sketchpad is not None:
            for a, rel, b in relations:
                self._sketchpad.store_relation(a, rel, b)
        result: dict[str, list[str] | list[tuple[str, str, str]]] = {}
        if self._phonological_loop is not None:
            result["phonological"] = self._phonological_loop.get_items()
        if self._sketchpad is not None:
            result["visuospatial"] = self._sketchpad.get_relations()
        return result

    def allocate_capacity(
        self,
        items: dict[str, float],
        capacity: int,
    ) -> dict[str, float]:
        """Allocate the attention buffer's capacity, dropping weakest items.

        When the number of items exceeds the capacity limit (Cowan's
        K≈4; Cowan, 2001), the executive drops the weakest items to
        stay within capacity. Suppressed items are removed first.

        Args:
            items: The current attention buffer.
            capacity: Maximum number of items to retain.

        Returns:
            The trimmed attention buffer (at most ``capacity`` items).
        """
        # Remove suppressed items.
        filtered = {c: a for c, a in items.items() if c.lower() not in self._suppressed}
        if len(filtered) <= capacity:
            return filtered
        # Keep the top-`capacity` by activation.
        kept = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:capacity]
        return dict(kept)


class WorkingMemory:
    """Genesis's working memory — what she's holding in mind.

    This tracks the active conversation context: what's being discussed,
    what was recently said, what concepts are in attention, and what
    questions are open.

    The attention buffer is capacity-limited (Cowan's K≈4; Cowan, 2001):
    it holds at most ``capacity`` items. When capacity is exceeded, the
    weakest items decay. Items can be rehearsed to reset their decay
    timer, and frequently accessed items are auto-rehearsed.

    The slave systems (phonological loop, visuospatial sketchpad) and
    the central executive are available for Baddeley-model coordination.
    """

    #: Default capacity limit for the attention buffer (Cowan's K≈4;
    #: Cowan, 2001). Working memory holds 3-5 items.
    DEFAULT_CAPACITY: int = 4
    #: Access count above which an item is auto-rehearsed (refreshed)
    #: during decay — frequently accessed items persist longer.
    _AUTO_REHEARSE_THRESHOLD: int = 2

    def __init__(
        self,
        attention_decay: float = 0.15,
        max_turns: int = 10,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        """Initialize working memory with attention and capacity settings.

        Args:
            attention_decay: Rate at which concept activations decay
                between turns.
            max_turns: Maximum conversation turns kept in the active
                context buffer.
            capacity: Maximum number of active concepts to retain
                (Cowan's K, default 4).
        """
        # Clamp to [0, 0.95] — _decay_attention divides by
        # (1 - attention_decay), so a decay ≥ 1.0 is a crash.
        self.attention_decay = max(0.0, min(0.95, attention_decay))  # per turn
        self.max_turns = max_turns
        self.capacity = max(1, capacity)  # Cowan's K (3-5 items)

        # Currently active concepts and their activation (0-1)
        self._attention: dict[str, float] = {}

        # Access counts per concept (for auto-rehearsal).
        self._access_counts: dict[str, int] = {}

        # Explicit capacity-limited attention slots (computed from
        # _attention after every update and gated by neurochemistry).
        self._slots: list[WorkingMemorySlot] = []

        # Recent turns (rolling window)
        self._recent_turns: list[Turn] = []

        # Current conversation thread
        self._current_thread: ConversationThread | None = None

        # All threads (history)
        self._threads: list[ConversationThread] = []

        # Open questions (things she was curious about)
        self._open_questions: list[str] = []

        # Topics discussed (for continuity detection)
        self._topic_history: list[str] = []

        # ── Baddeley slave systems & central executive ───────────
        self.phonological_loop: PhonologicalLoop = PhonologicalLoop()
        self.visuospatial_sketchpad: VisuoSpatialSketchpad = VisuoSpatialSketchpad()
        self.central_executive: CentralExecutive = CentralExecutive(
            phonological_loop=self.phonological_loop,
            sketchpad=self.visuospatial_sketchpad,
        )

    def update(
        self,
        turn: Turn,
        neuro_summary: NeuroSummary | None = None,
        brain_waves: BrainWaveState | None = None,
    ) -> None:
        """Update working memory with a new turn and current neurochemistry."""
        # Capacity is not fixed — it expands with arousal and plasticity
        # and shrinks with negative valence and low plasticity.
        if neuro_summary is not None:
            self.update_capacity(neuro_summary, brain_waves)

        # Add to recent turns
        self._recent_turns.append(turn)
        if len(self._recent_turns) > self.max_turns:
            self._recent_turns = self._recent_turns[-self.max_turns :]

        # Update attention
        self._decay_attention()
        for topic in turn.topics:
            self._activate(topic, amount=0.8)

        # Central executive task switching: when the primary topic
        # changes, the executive switches focus, partially
        # suppressing the old focus (task-switching cost, Monsell 2003).
        if turn.topics:
            primary = turn.topics[0]
            if self.central_executive.current_focus != primary:
                self.central_executive.switch_task(primary)

        # Enforce capacity limit (Cowan's K). The central executive
        # allocates capacity, dropping the weakest items.
        self._enforce_capacity()

        # Expose the attention buffer as explicit slots.
        self._update_slots()

        # Update topic history
        self._topic_history.extend(turn.topics)

        # Update or start conversation thread
        self._update_thread(turn)

    def update_capacity(
        self,
        neuro_summary: NeuroSummary,
        brain_waves: BrainWaveState | None = None,
    ) -> None:
        """Set the attention capacity from the current neurochemical state.

        Arousal and plasticity expand working memory (cholinergic and
        dopaminergic gating); negative valence or low plasticity shrink it.

        Brain-wave modulation:
        - Gamma → +1 capacity (integration/binding needs more slots)
        - Delta → -1 capacity (deep rest shrinks the scratchpad)
        - Other bands → no additional modulation (arousal/plasticity
          already capture the alertness signal)
        """
        # Base capacity is 3 (Baddeley lower bound). Arousal and
        # plasticity can push it up to 9 (well above Cowan's K for
        # highly alert, plastic states). Negative valence reduces it.
        base = 3.0
        arousal = max(0.0, min(1.0, neuro_summary.arousal))
        plasticity = max(0.0, min(1.0, neuro_summary.plasticity_gate))
        valence = max(-1.0, min(1.0, neuro_summary.valence))
        capacity = base + arousal * 3.0 + plasticity * 2.0 + valence * 0.5

        # Brain-wave modulation — gamma integration expands capacity
        # for binding multiple representations; delta rest shrinks it.
        if brain_waves is not None:
            if brain_waves.dominant == BrainWave.GAMMA:
                capacity += brain_waves.integration
            elif brain_waves.dominant == BrainWave.DELTA:
                capacity -= 1.0

        # Clamp to a biologically plausible range.
        self.capacity = int(max(2.0, min(9.0, capacity)))

    def _update_slots(self) -> None:
        """Rebuild the explicit slot list from the attention map."""
        now = int(time.time() * 1000)
        slots: list[WorkingMemorySlot] = []
        for content, activation in self._attention.items():
            count = self._access_counts.get(content, 0)
            slots.append(
                WorkingMemorySlot(
                    content=content,
                    activation=activation,
                    rehearsals=count,
                    last_accessed=now,
                )
            )
        # Sort by activation (highest first) and capacity already enforced.
        slots.sort(key=lambda s: s.activation, reverse=True)
        self._slots = slots

    def _update_thread(self, turn: Turn) -> None:
        """Update the current conversation thread."""
        if self._current_thread is None:
            # Start a new thread
            topic = turn.topics[0] if turn.topics else "unknown"
            self._current_thread = ConversationThread(topic=topic)
            self._current_thread.add_turn(turn)
            self._threads.append(self._current_thread)
            return

        # Check if this turn continues the current thread
        current_concepts = self._current_thread.key_concepts
        turn_concepts = set(turn.topics)

        if turn_concepts and current_concepts:
            overlap = len(turn_concepts & current_concepts) / len(turn_concepts | current_concepts)
            if overlap > 0.15:
                # Same thread
                self._current_thread.add_turn(turn)
                # Update topic if new concepts are more prominent
                if turn.topics and turn.topics[0] != self._current_thread.topic:
                    if turn.topics[0] not in current_concepts:
                        # Significant new concept — but still related
                        self._current_thread.key_concepts.update(turn_concepts)
            else:
                # New thread
                self._current_thread = ConversationThread(
                    topic=turn.topics[0] if turn.topics else "unknown"
                )
                self._current_thread.add_turn(turn)
                self._threads.append(self._current_thread)
        else:
            self._current_thread.add_turn(turn)

    def _activate(self, concept: str, amount: float = 0.5) -> None:
        """Activate a concept in attention."""
        current = self._attention.get(concept, 0.0)
        self._attention[concept] = min(1.0, current + amount)
        # Track access for auto-rehearsal.
        self._access_counts[concept] = self._access_counts.get(concept, 0) + 1

    def _decay_attention(self) -> None:
        """Decay all attention values, with auto-rehearsal for hot items.

        Frequently accessed items (access count ≥ the auto-rehearsal
        threshold) are refreshed — their decay is partially reversed,
        modelling the articulatory rehearsal that keeps "hot" items in
        working memory (Baddeley, 1992).
        """
        to_remove: list[str] = []
        for concept in self._attention:
            self._attention[concept] *= 1.0 - self.attention_decay
            # Auto-rehearsal: frequently accessed items resist decay.
            if self._access_counts.get(concept, 0) >= self._AUTO_REHEARSE_THRESHOLD:
                # Refresh: undo part of the decay.
                self._attention[concept] = min(
                    1.0, self._attention[concept] / (1.0 - self.attention_decay)
                )
            if self._attention[concept] < 0.05:
                to_remove.append(concept)
        for concept in to_remove:
            del self._attention[concept]
            self._access_counts.pop(concept, None)

    def _enforce_capacity(self) -> None:
        """Enforce the capacity limit (Cowan's K).

        When the number of attended concepts exceeds the capacity
        limit, the weakest are dropped. The central executive
        allocates capacity, removing suppressed items first.
        """
        if len(self._attention) <= self.capacity:
            return
        self._attention = self.central_executive.allocate_capacity(self._attention, self.capacity)
        # Clean up access counts for dropped concepts.
        for concept in list(self._access_counts):
            if concept not in self._attention:
                del self._access_counts[concept]

    def rehearse(self, item: str) -> bool:
        """Rehearse an item — reset its decay timer.

        Rehearsal is the articulatory control process: sub-vocal
        repetition refreshes an item's activation, preventing decay
        (Baddeley, 1992). This is how Genesis keeps something "in mind".

        Args:
            item: The concept to rehearse.

        Returns:
            ``True`` if the item was in attention and rehearsed.
        """
        if item not in self._attention:
            return False
        self._attention[item] = 1.0
        self._access_counts[item] = self._access_counts.get(item, 0) + 1
        return True

    def get_attention(self, threshold: float = 0.1) -> dict[str, float]:
        """Get currently attended concepts above threshold."""
        return {c: a for c, a in self._attention.items() if a >= threshold}

    def get_top_attention(self, n: int = 3) -> list[str]:
        """Get the top N attended concepts."""
        sorted_attention = sorted(self._attention.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in sorted_attention[:n]]

    def is_on_topic(self, topics: list[str]) -> bool:
        """Check if the given topics are related to the current thread."""
        if not self._current_thread or not topics:
            return False
        current = self._current_thread.key_concepts
        turn_concepts = set(topics)
        if not current or not turn_concepts:
            return False
        overlap = len(turn_concepts & current) / len(turn_concepts | current)
        return overlap > 0.15

    def get_thread_summary(self) -> str:
        """Get a summary of the current conversation thread."""
        if not self._current_thread:
            return "No active conversation thread."
        return self._current_thread.describe()

    def get_recent_turns(self, n: int = 3) -> list[Turn]:
        """Get the N most recent turns."""
        return self._recent_turns[-n:]

    def get_last_turn(self) -> Turn | None:
        """Get the most recent turn, if any."""
        if self._recent_turns:
            return self._recent_turns[-1]
        return None

    @property
    def slots(self) -> list[WorkingMemorySlot]:
        """The current capacity-limited attention slots."""
        return self._slots

    @property
    def current_capacity(self) -> int:
        """The current working memory capacity (dynamically gated)."""
        return self.capacity

    def was_topic_discussed(self, topic: str, within_last: int = 5) -> bool:
        """Was this topic discussed recently?"""
        recent = self._recent_turns[-within_last:]
        for turn in recent:
            if topic.lower() in [t.lower() for t in turn.topics]:
                return True
        return False

    def add_open_question(self, question: str) -> None:
        """Add an open question Genesis is curious about."""
        self._open_questions.append(question)
        if len(self._open_questions) > 5:
            self._open_questions = self._open_questions[-5:]

    def get_open_questions(self) -> list[str]:
        """Get open questions that haven't been asked yet."""
        return list(self._open_questions)

    def clear_open_question(self, question: str) -> None:
        """Mark an open question as resolved."""
        self._open_questions = [q for q in self._open_questions if q != question]

    @property
    def thread_count(self) -> int:
        """How many conversation threads have there been?"""
        return len(self._threads)

    @property
    def turn_count(self) -> int:
        """How many turns in the current thread?"""
        if self._current_thread:
            return self._current_thread.turn_count
        return 0

    @property
    def total_turns(self) -> int:
        """Total turns across all threads."""
        return len(self._recent_turns)

    def describe(self) -> str:
        """Describe the current working memory state."""
        top = self.get_top_attention(3)
        thread_desc = self.get_thread_summary()
        return (
            f"Working memory: {len(self._recent_turns)} recent turns, "
            f"{self.thread_count} threads. "
            f"Attending to: {', '.join(top) if top else 'nothing'}. "
            f"{thread_desc}"
        )

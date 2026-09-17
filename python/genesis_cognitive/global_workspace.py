"""Global workspace — the brain's cognitive access broadcast system.

Implements Dehaene & Changeux's Global Workspace Theory (GWT). When
information in any specialized module reaches high activation (above
a threshold), it is "broadcast" to all other modules. This broadcast
is what makes information cognitively accessible — it becomes available
to all cognitive processes simultaneously.

# Global Workspace Theory

The brain has many specialized, modular processors (vision, hearing,
memory, emotion, motor planning) that operate in parallel and
noncognitively. Most processing never reaches awareness. But when
information in a module reaches sufficient activation, it is broadcast
to a "global workspace" — a central bottleneck that distributes the
information to all other modules. This broadcast is the neural
correlate of cognitive access (Dehaene & Naccache, 2001; Dehaene,
2014; Baars, 1988).

Key properties:

1. **Ignition**: When a module's activation crosses threshold, it
   "ignites" the workspace — a sudden, all-or-none broadcast. This
   matches the P3b ERP component, which marks cognitive access
   (Sergent et al., 2005; Dehaene & Changeux, 2011).

2. **Limited capacity**: Only 3-4 items can be simultaneously in the
   workspace (Cowan's K≈4, Cowan 2001). This is the "bottleneck" —
   competing broadcasts inhibit each other, and only the strongest
   survive. This matches the limited capacity of working memory and
   cognitive awareness.

3. **Cross-module integration**: Broadcasting enables integration
   across modules that normally operate independently. A percept can
   trigger a memory, which triggers an emotion, which triggers a
   motor plan — all because the percept was broadcast globally.

4. **Competition**: Multiple modules may try to broadcast simultaneously.
   The workspace selects the most activated items through a winner-take-all
   process, modeling the global inhibition that prevents multiple
   cognitive contents at once.

For Genesis, modules are the cognitive subsystems: perception, emotion,
memory, reasoning, self-model, narrative, etc. When any of these
produces highly activated information, it enters the workspace and
becomes available to all others.

References:
- Baars, B. J. (1988). A Cognitive Theory of Cognition.
- Dehaene, S., & Naccache, L. (2001). Towards a cognitive neuroscience
  of cognition. Cognition.
- Dehaene, S., & Changeux, J.-P. (2011). Experimental and theoretical
  approaches to cognitive processing. Neuron.
- Dehaene, S. (2014). Cognition and the Brain.
- Cowan, N. (2001). The magical number 4 in short-term memory.
  Behavioral and Brain Sciences.
- Sergent, C., et al. (2005). Timing of the brain events underlying
  access to cognition. Psychological Science.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .brain_waves import BrainWave, BrainWaveState

logger = logging.getLogger(__name__)


class WorkspaceModule(Protocol):
    """Protocol for modules that can receive workspace broadcasts.

    Any module that wants to receive broadcasts must implement
    ``receive_broadcast``. This is a structural (duck-typed) protocol,
    so modules don't need to explicitly inherit from it.
    """

    def receive_broadcast(self, item: WorkspaceItem) -> None:
        """Receive a broadcast item from the global workspace.

        Args:
            item: The workspace item that was broadcast.
        """
        ...


@dataclass(slots=True)
class WorkspaceItem:
    """A single item in the global workspace.

    An item represents a piece of information that has crossed the
    ignition threshold and been broadcast. It carries:
    - The content (what was broadcast)
    - The source module (where it came from)
    - The activation level (how strongly it ignited)
    - A timestamp (when it entered the workspace)
    - Optional metadata

    Items decay over time — if not refreshed, they fall out of the
    workspace, modeling the fleeting nature of cognitive access.
    """

    content: Any
    source: str
    activation: float
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, Any] = field(default_factory=dict)
    # How many times this item has been refreshed (sustained attention)
    refresh_count: int = 0

    @property
    def age_ms(self) -> int:
        """How old this item is (milliseconds since creation)."""
        return int(time.time() * 1000) - self.timestamp


class GlobalWorkspace:
    """The global workspace — cognitive access broadcast system.

    Receives broadcasts from modules when their information crosses
    an ignition threshold, distributes to all registered modules, and
    maintains a limited-capacity buffer of currently cognitive items.

    The workspace has a limited capacity (default 4, matching Cowan's
    K≈4). When capacity is exceeded, the weakest items are evicted
    (competitive inhibition). Items also decay over time if not
    refreshed, modeling the transient nature of cognitive access.

    # Integration

    Modules register with the workspace via ``register_module()``.
    When a module has highly-activated information, it calls
    ``broadcast()``. The workspace:
    1. Checks if the item's activation exceeds the ignition threshold
    2. If so, adds it to the workspace (evicting weak items if full)
    3. Distributes it to all registered modules via ``receive_broadcast()``

    This enables cross-module integration: a percept broadcast to the
    workspace is received by memory (triggering retrieval), emotion
    (triggering affective response), and reasoning (triggering inference).
    """

    def __init__(
        self,
        ignition_threshold: float = 0.7,
        capacity: int = 4,
        decay_rate: float = 0.05,
    ) -> None:
        """Initialize the global workspace.

        Args:
            ignition_threshold: Minimum activation for a broadcast to
                "ignite" the workspace. Default 0.7. Below this,
                information stays noncognitive (local processing only).
            capacity: Maximum number of simultaneous items (Cowan's K).
                Default 4. When exceeded, weakest items are evicted.
            decay_rate: How much activation decays per tick. Default 0.05.
                Items that aren't refreshed fade from cognition.
        """
        self.ignition_threshold = ignition_threshold
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._modules: list[WorkspaceModule] = []
        self._items: list[WorkspaceItem] = []
        self._broadcast_count: int = 0
        # The workspace is accessed from both the cognition thread
        # (during think()) and the inner life thread (spontaneous
        # thoughts broadcast + tick). This lock protects _items and
        # _broadcast_count from concurrent modification.
        self._lock = threading.Lock()

    def register_module(self, module: WorkspaceModule) -> None:
        """Register a module to receive workspace broadcasts.

        Registered modules will have ``receive_broadcast()`` called
        whenever an item ignites the workspace. This is how cross-module
        integration happens.

        Args:
            module: Any object implementing ``receive_broadcast(item)``.
        """
        if module not in self._modules:
            self._modules.append(module)

    def unregister_module(self, module: WorkspaceModule) -> None:
        """Remove a module from the broadcast distribution list."""
        if module in self._modules:
            self._modules.remove(module)

    def broadcast(
        self,
        content: Any,
        source: str,
        activation: float,
        metadata: dict[str, Any] | None = None,
        brain_waves: BrainWaveState | None = None,
    ) -> bool:
        """Attempt to broadcast an item to the global workspace.

        If the item's activation exceeds the ignition threshold, it
        enters the workspace and is distributed to all registered
        modules. If activation is below threshold, the broadcast fails
        (the information remains noncognitive/local).

        Args:
            content: The information being broadcast (any type).
            source: Which module is broadcasting (e.g., "perception").
            activation: How strongly activated the information is (0..1).
            metadata: Optional additional information about the broadcast.
            brain_waves: Optional brain wave state. Gamma synchrony
                lowers the ignition threshold (more cognitive access
                via long-range integration). Delta raises it (deep
                rest suppresses cognitive access).

        Returns:
            True if the broadcast ignited (crossed threshold), False
            if it was below threshold (remained noncognitive).
        """
        # Brain-wave-modulated ignition threshold.
        # Gamma: long-range synchrony → easier ignition (lower threshold).
        # Delta: deep rest → harder ignition (higher threshold).
        effective_threshold = self.ignition_threshold
        if brain_waves is not None:
            if brain_waves.dominant == BrainWave.GAMMA:
                effective_threshold -= 0.1 * brain_waves.integration
            elif brain_waves.dominant == BrainWave.DELTA:
                effective_threshold += 0.15
            effective_threshold = max(0.3, min(0.95, effective_threshold))

        if activation < effective_threshold:
            return False

        with self._lock:
            # Check if this is a refresh of an existing item
            for item in self._items:
                if item.source == source and _content_equal(item.content, content):
                    item.activation = max(item.activation, activation)
                    item.refresh_count += 1
                    distribute_item = item
                    break
            else:
                # New item — create it
                item = WorkspaceItem(
                    content=content,
                    source=source,
                    activation=min(1.0, activation),
                    metadata=metadata or {},
                )
                # Add to workspace (evict weakest if over capacity)
                self._items.append(item)
                self._enforce_capacity()
                self._broadcast_count += 1
                distribute_item = item

        # Distribute outside the lock to avoid holding it while
        # modules process the broadcast (they may be slow).
        self._distribute(distribute_item)
        return True

    def _distribute(self, item: WorkspaceItem) -> None:
        """Distribute an item to all registered modules."""
        for module in self._modules:
            try:
                module.receive_broadcast(item)
            except Exception as e:  # noqa: BLE001
                # A module failing to receive shouldn't crash the workspace
                logger.debug(repr(e))

    def _enforce_capacity(self) -> None:
        """Evict weakest items if over capacity (competitive inhibition)."""
        while len(self._items) > self.capacity:
            # Find the weakest item (lowest activation)
            weakest_idx = 0
            weakest_activation = self._items[0].activation
            for i, item in enumerate(self._items):
                if item.activation < weakest_activation:
                    weakest_activation = item.activation
                    weakest_idx = i
            self._items.pop(weakest_idx)

    def tick(self, dt: float = 1.0) -> list[WorkspaceItem]:
        """Advance workspace dynamics by dt.

        Applies activation decay to all items and removes items that
        have decayed below a minimum threshold. This models the
        transient nature of cognitive access — items fade if not
        refreshed (sustained attention).

        Args:
            dt: Time step.

        Returns:
            List of items that were evicted due to decay.
        """
        with self._lock:
            evicted: list[WorkspaceItem] = []
            surviving: list[WorkspaceItem] = []

            for item in self._items:
                item.activation *= 1.0 - self.decay_rate * dt
                # Clamp to non-negative — a large dt can make the decay
                # factor negative, flipping activation's sign.
                if item.activation < 0.0:
                    item.activation = 0.0
                if item.activation < 0.1:
                    evicted.append(item)
                else:
                    surviving.append(item)

            self._items = surviving
            return evicted

    @property
    def items(self) -> list[WorkspaceItem]:
        """Currently cognitive items in the workspace."""
        with self._lock:
            return list(self._items)

    @property
    def is_empty(self) -> bool:
        """Whether the workspace is currently empty (no cognitive content)."""
        with self._lock:
            return len(self._items) == 0

    @property
    def broadcast_count(self) -> int:
        """Total number of successful broadcasts since creation."""
        with self._lock:
            return self._broadcast_count

    @property
    def registered_modules(self) -> list[WorkspaceModule]:
        """List of registered modules."""
        return list(self._modules)

    def get_dominant(self) -> WorkspaceItem | None:
        """Get the most activated item (the dominant cognitive content).

        In GWT, the most activated item is the "focus of cognition"
        — what the mind is currently most aware of. Less-activated items
        are in the "fringe" of cognition.
        """
        with self._lock:
            if not self._items:
                return None
            return max(self._items, key=lambda item: item.activation)

    @property
    def integration(self) -> float:
        """How globally integrated the workspace state is [0, 1].

        Integration measures whether the workspace is acting as a
        **unified cognitive field** — multiple modules contributing
        coherent content — or is fragmented (disjoint topics from a
        single source) or empty.

        This is the workspace-level counterpart to the cognitive
        trajectory model's precision and the Rust active inference
        engine's free energy. Where those measure how well the system
        predicts itself, integration measures how **unified** the
        current cognitive field is.

        The measure combines three factors:

        1. **Topic coherence**: average pairwise Jaccard similarity of
           topic sets across items. High when items share topics
           (the workspace is about one thing). Low when items have
           disjoint topics (the workspace is scattered).

        2. **Source diversity**: fraction of distinct sources among
           items. High when multiple modules contributed (perception +
           deliberation + inner_life). Low when one source dominates
           (narrow, not cross-module integration).

        3. **Activation**: mean activation of items. High when content
           is strongly cognitive. Low when items are fading.

        Integration = coherence × diversity × activation.

        This follows the Integrated Information Theory principle that
        integration requires both **differentiation** (multiple
        distinct elements) and **integration** (they form a unified
        whole). One factor being zero makes the product zero: a
        workspace with one source (no differentiation) or disjoint
        topics (no integration) or no activation (not cognitive) has
        zero integration (Tononi, 2004; Baars, 1988).

        Returns:
            Integration in [0, 1]. 0.0 if empty or one item (no
            cross-module integration possible). Higher when multiple
            items from diverse sources share topics at high
            activation.
        """
        with self._lock:
            items = list(self._items)

        if not items:
            return 0.0

        if len(items) == 1:
            # One item: present but not integrated — no cross-module
            # content to unify. Give a small value scaled by activation
            # so a strongly cognitive single item is "more present"
            # than a weak one, but neither is "integrated."
            return items[0].activation * 0.2

        # ── Topic coherence: average pairwise Jaccard similarity ──
        topic_sets: list[set[str]] = []
        for item in items:
            raw = item.metadata.get("topics", [])
            if raw:
                topics = {str(t) for t in raw if t}
                if topics:
                    topic_sets.append(topics)

        if len(topic_sets) < 2:
            # Not enough topic-bearing items to measure coherence.
            # Neutral: we can't say it's coherent or fragmented.
            coherence = 0.5
        else:
            total_overlap = 0.0
            pairs = 0
            for i in range(len(topic_sets)):
                for j in range(i + 1, len(topic_sets)):
                    union = topic_sets[i] | topic_sets[j]
                    if union:
                        total_overlap += len(topic_sets[i] & topic_sets[j]) / len(union)
                    pairs += 1
            coherence = total_overlap / pairs if pairs > 0 else 0.5

        # ── Source diversity: fraction of distinct sources ──
        sources = {item.source for item in items}
        diversity = len(sources) / len(items)

        # ── Activation: mean activation ──
        mean_activation = sum(item.activation for item in items) / len(items)

        return max(0.0, min(1.0, coherence * diversity * mean_activation))

    def clear(self) -> None:
        """Clear all items from the workspace."""
        with self._lock:
            self._items.clear()


def _content_equal(a: Any, b: Any) -> bool:
    """Check if two content values are equal (handles unhashable types)."""
    if a is b:
        return True
    try:
        return a == b
    except Exception as e:
        logger.exception(f"content comparison failed: {e}")
        return False

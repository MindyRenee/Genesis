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

5. **Recurrence**: Ignition is not a one-shot threshold crossing. In the
   brain, cognitive access emerges from recurrent dynamics — feedforward
   drive triggers the broadcast, then slow feedback (NMDA-like
   reverberation) sustains the ignited assembly while competing
   assemblies suppress each other through lateral inhibition
   (Dehaene & Changeux, 2011; Wang, 2001). The workspace models this
   with a short recurrent settling phase on each broadcast: candidate
   and incumbent items mutually excite when topic-coherent and
   inhibit when incoherent, under divisive global inhibition that
   enforces the capacity bottleneck. Incumbents receive a self-sustain
   current, producing hysteresis — an ignited item resists being
   displaced. Subliminal candidates that fail to ignite persist
   briefly in a pending buffer and can co-ignite later if
   coherent content arrives (coalition ignition).

For Genesis, modules are the cognitive subsystems: perception, emotion,
memory, reasoning, self-model, narrative, etc. When any of these
produces highly activated information, it enters the workspace and
becomes available to all others.

References:
- Baars, B. J. (1988). Cambridge University Press.
- Dehaene, S., & Naccache, L. (2001). Cognition, 79(1-2).
- Dehaene, S., & Changeux, J.-P. (2011). Neuron, 70(2).
- Dehaene, S. (2014). Viking Press.
- Cowan, N. (2001). The magical number 4 in short-term memory.
  Behavioral and Brain Sciences.
- Sergent, C., et al. (2005). Nature Neuroscience, 8(10).
- Wang, X.-J. (2001). Synaptic reverberation underlying mnemonic
  persistent activity. Trends in Neurosciences.
- Wong, K.-F., & Wang, X.-J. (2006). A recurrent network mechanism of
  time integration in perceptual decisions. J. Neuroscience.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
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
        settle_steps: int = 4,
        coherence_excitation: float = 0.30,
        global_inhibition: float = 0.30,
        sustain: float = 0.35,
        coupling_rate: float = 0.10,
        max_coupling_delta: float = 0.15,
        pending_capacity: int = 8,
        pending_ttl_s: float = 120.0,
        pending_influence: float = 0.5,
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
            settle_steps: Recurrent iterations run on each broadcast to
                let competition resolve. Default 4 — enough for the
                coupled activations to approach equilibrium on small
                competitor sets without measurable latency.
            coherence_excitation: Strength of topic-coherence coupling
                during settle. Items sharing topics excite; disjoint
                items inhibit. Default 0.30.
            global_inhibition: Divisive inhibition from the total
                activity of the competitor pool — the soft capacity
                bottleneck. Default 0.30.
            sustain: Self-excitation current for items already in the
                workspace during settle (incumbency advantage /
                hysteresis). Default 0.35.
            coupling_rate: Per-tick coupling strength applied between
                workspace items during tick(). Default 0.10.
            max_coupling_delta: Absolute cap on one tick's coupling
                adjustment to a single item. Ticks can carry large dt,
                so the recurrent step is clamped rather than scaled
                linearly. Default 0.15.
            pending_capacity: Maximum size of the pending buffer
                (below-threshold broadcasts awaiting coalition).
                Default 8.
            pending_ttl_s: Wall-clock seconds a pending item survives
                without being refreshed or ignited. Default 120.
            pending_influence: Weight at which pending items contribute
                to couplings on workspace items. Subliminal content is
                pulled up by coherence but only weakly pushes on
                cognitive content. Default 0.5.
        """
        self.ignition_threshold = ignition_threshold
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.settle_steps = settle_steps
        self.coherence_excitation = coherence_excitation
        self.global_inhibition = global_inhibition
        self.sustain = sustain
        self.coupling_rate = coupling_rate
        self.max_coupling_delta = max_coupling_delta
        self.pending_capacity = pending_capacity
        self.pending_ttl_s = pending_ttl_s
        self.pending_influence = pending_influence
        self._modules: list[WorkspaceModule] = []
        self._items: list[WorkspaceItem] = []
        self._pending: list[WorkspaceItem] = []
        self._broadcast_count: int = 0
        # Optional hook fired once per fresh ignition (candidate or
        # coalition), after distribution. Signature: (item,
        # via_recurrent) — via_recurrent is True when ignition owed to
        # recurrent dynamics (pending co-ignition, or a sub-threshold
        # input pulled over by coherent support) rather than raw
        # feedforward drive. Refreshes don't fire it — sustained
        # attention is not a new ignition.
        self.on_ignition: Callable[[WorkspaceItem, bool], None] | None = None
        # Subliminal items that expired from the pending pool without
        # igniting. Drained via drain_subliminal() — callers can route
        # them into curiosity or memory rather than letting them vanish.
        self._expired: deque[WorkspaceItem] = deque(maxlen=20)
        # The workspace is accessed from both the cognition thread
        # (during think()) and the inner life thread (spontaneous
        # thoughts broadcast + tick). This lock protects _items,
        # _pending, and _broadcast_count from concurrent modification.
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
        modules. If activation is below threshold, the item joins a
        short-lived pending buffer instead: during subsequent
        broadcasts it participates in the recurrent competition and
        can still ignite if coherent content pulls it over threshold.

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

        to_distribute: list[WorkspaceItem] = []
        ignited = False
        just_ignited: set[int] = set()
        recurrent_ignitions: set[int] = set()
        with self._lock:
            # Refresh fast path — same content+source already in the
            # workspace updates the incumbent and rebroadcasts.
            for item in self._items:
                if item.source == source and _content_equal(item.content, content):
                    item.activation = max(item.activation, activation)
                    item.refresh_count += 1
                    to_distribute.append(item)
                    ignited = True
                    break
            else:
                self._expire_pending()
                candidate = WorkspaceItem(
                    content=content,
                    source=source,
                    activation=min(1.0, activation),
                    metadata=metadata or {},
                )
                # Recurrent competition among incumbents, the candidate,
                # and the pending pool. Settled activations decide
                # co-ignitions and eviction order; the candidate's own
                # ignition is all-or-none on its input (feedforward
                # drive) OR on its settled activation (recurrent
                # support from coherent incumbents).
                self._settle(candidate, effective_threshold)
                ignited = (
                    activation >= effective_threshold
                    or candidate.activation >= effective_threshold
                )
                if ignited:
                    self._items.append(candidate)
                    just_ignited.add(id(candidate))
                    if activation < effective_threshold:
                        recurrent_ignitions.add(id(candidate))
                    self._broadcast_count += 1
                    to_distribute.append(candidate)
                else:
                    self._upsert_pending(candidate)

                # Coalition ignition — pending items the settle pulled
                # over threshold join the workspace and broadcast.
                still_pending: list[WorkspaceItem] = []
                for p in self._pending:
                    if p.activation >= effective_threshold:
                        self._items.append(p)
                        just_ignited.add(id(p))
                        recurrent_ignitions.add(id(p))
                        self._broadcast_count += 1
                        to_distribute.append(p)
                    else:
                        still_pending.append(p)
                self._pending = still_pending

                # Tag recurrent ignitions on the item itself so
                # receivers can treat emerged content differently
                # from driven content.
                for item in self._items:
                    if id(item) in recurrent_ignitions:
                        item.metadata["ignition"] = "recurrent"

                self._enforce_capacity(just_ignited)

        # Distribute outside the lock to avoid holding it while
        # modules process the broadcast (they may be slow).
        for item in to_distribute:
            self._distribute(item)
            if id(item) in just_ignited and self.on_ignition is not None:
                try:
                    self.on_ignition(item, id(item) in recurrent_ignitions)
                except Exception as e:  # noqa: BLE001
                    # The ignition hook must never crash the workspace
                    logger.debug(f"on_ignition hook failed: {e}")
        return ignited

    def _distribute(self, item: WorkspaceItem) -> None:
        """Distribute an item to all registered modules."""
        for module in self._modules:
            try:
                module.receive_broadcast(item)
            except Exception as e:  # noqa: BLE001
                # A module failing to receive shouldn't crash the workspace
                logger.debug(repr(e))

    def _enforce_capacity(self, protected: set[int] | None = None) -> None:
        """Evict weakest items if over capacity (competitive inhibition).

        Items whose ids are in ``protected`` (fresh ignitions this
        settle) are only evicted when every unprotected item is gone —
        a just-ignited broadcast isn't immediately silenced by the
        bottleneck it triggered.
        """
        while len(self._items) > self.capacity:
            pool = [
                (i, item)
                for i, item in enumerate(self._items)
                if protected is None or id(item) not in protected
            ]
            if not pool:
                pool = list(enumerate(self._items))
            weakest_idx, _ = min(pool, key=lambda pair: pair[1].activation)
            self._items.pop(weakest_idx)

    def _coherence(self, a: WorkspaceItem, b: WorkspaceItem) -> float:
        """Topic coherence between two items, in [0, 1].

        Jaccard similarity of topic sets. Items without topics are
        neutral (0.5) — neither excitatory nor inhibitory.
        """
        topics_a = {str(t) for t in a.metadata.get("topics", []) if t}
        topics_b = {str(t) for t in b.metadata.get("topics", []) if t}
        if not topics_a or not topics_b:
            return 0.5
        union = topics_a | topics_b
        return len(topics_a & topics_b) / len(union) if union else 0.5

    def _settle(
        self, candidate: WorkspaceItem, effective_threshold: float
    ) -> None:
        """Run recurrent competition over incumbents + candidate + pending.

        Each iteration applies, from a synchronous snapshot:

        - coherence coupling: ``coherence_excitation * (2*c_ij - 1) * a_j``
          — topic-coherent items excite, incoherent items inhibit.
          Pending items contribute at ``pending_influence`` weight and
          are excluded from the inhibition pool.
        - divisive global inhibition: ``global_inhibition`` times the
          mean of the other workspace competitors — the soft capacity
          bottleneck (Carandini & Heeger normalization analogue).
        - sustain: ``sustain * a_i * (1 - a_i)`` — logistic
          self-excitation for items already in the workspace and for
          candidates at or above the ignition threshold (an igniting
          assembly is a driven, self-reinforcing one). Pending items
          never sustain — subliminal content has no reverberation.
          This is the hysteresis term: ignited assemblies resist
          displacement.
        """
        competitors = [*self._items, candidate, *self._pending]
        if len(competitors) < 2:
            return
        workspace_ids = {id(item) for item in self._items}
        sustained = set(workspace_ids)
        if candidate.activation >= effective_threshold:
            sustained.add(id(candidate))

        for _ in range(self.settle_steps):
            activations = [item.activation for item in competitors]
            deltas: list[float] = []
            for i, item in enumerate(competitors):
                exc = 0.0
                others = 0.0
                for j, other in enumerate(competitors):
                    if i == j:
                        continue
                    influence = (
                        1.0
                        if id(other) in workspace_ids or other is candidate
                        else self.pending_influence
                    )
                    coupling = 2.0 * self._coherence(item, other) - 1.0
                    exc += coupling * activations[j] * influence
                    if id(other) in workspace_ids or other is candidate:
                        others += activations[j]
                delta = self.coherence_excitation * exc
                delta -= self.global_inhibition * others / max(1, self.capacity)
                if id(item) in sustained:
                    delta += self.sustain * activations[i] * (1.0 - activations[i])
                deltas.append(delta)
            for item, delta in zip(competitors, deltas, strict=True):
                item.activation = max(0.0, min(1.0, item.activation + delta))

    def _expire_pending(self) -> None:
        """Drop pending items older than the pending TTL.

        Expired items go to the drain buffer — content that was active
        enough to be broadcast but never ignited is still meaningful
        (it was almost cognitive). Callers decide what to do with it.
        """
        ttl_ms = self.pending_ttl_s * 1000.0
        expired = [p for p in self._pending if p.age_ms > ttl_ms]
        self._expired.extend(expired)
        self._pending = [p for p in self._pending if p.age_ms <= ttl_ms]

    def drain_subliminal(self) -> list[WorkspaceItem]:
        """Return and clear pending items that expired without igniting.

        These are "almost-thoughts" — content that entered the
        pending buffer, competed, lost, and timed out. The
        caller decides their fate (curiosity topics, memory traces).
        """
        with self._lock:
            drained = list(self._expired)
            self._expired.clear()
            return drained

    def _upsert_pending(self, candidate: WorkspaceItem) -> None:
        """Add or refresh a below-threshold candidate in the pending pool."""
        for p in self._pending:
            if p.source == candidate.source and _content_equal(
                p.content, candidate.content
            ):
                p.activation = max(p.activation, candidate.activation)
                p.timestamp = candidate.timestamp
                return
        self._pending.append(candidate)
        while len(self._pending) > self.pending_capacity:
            weakest = min(self._pending, key=lambda p: p.activation)
            self._pending.remove(weakest)

    def tick(self, dt: float = 1.0) -> list[WorkspaceItem]:
        """Advance workspace dynamics by dt.

        Applies activation decay to all items, then one step of the
        recurrent coupling — coherent items mutually sustain, incoherent
        items suppress each other, and the shared inhibition pool makes
        crowded workspaces decay faster. Pending items expire by TTL.
        This models the transient nature of cognitive access — items
        fade if not refreshed (sustained attention), and competition
        between items unfolds continuously between broadcasts rather
        than only at ignition time.

        Args:
            dt: Time step.

        Returns:
            List of items that were evicted due to decay.
        """
        with self._lock:
            self._expire_pending()
            evicted: list[WorkspaceItem] = []
            surviving: list[WorkspaceItem] = []

            for item in self._items:
                item.activation *= 1.0 - self.decay_rate * dt
                # Clamp to non-negative — a large dt can make the decay
                # factor negative, flipping activation's sign.
                if item.activation < 0.0:
                    item.activation = 0.0
                surviving.append(item)

            # Recurrent coupling step. Scaled by min(dt, 1): the
            # coupling is a per-tick competitive event, not continuous
            # integration, and production ticks carry large dt.
            if len(surviving) > 1:
                scale = min(dt, 1.0)
                activations = [item.activation for item in surviving]
                for i, item in enumerate(surviving):
                    coupling = 0.0
                    others = 0.0
                    for j, other in enumerate(surviving):
                        if i == j:
                            continue
                        coupling += (
                            (2.0 * self._coherence(item, other) - 1.0)
                            * activations[j]
                        )
                        others += activations[j]
                    delta = self.coupling_rate * scale * (
                        coupling - others / max(1, self.capacity)
                    )
                    delta = max(
                        -self.max_coupling_delta,
                        min(self.max_coupling_delta, delta),
                    )
                    item.activation = max(
                        0.0, min(1.0, item.activation + delta)
                    )

            self._items = [
                item for item in surviving if item.activation >= 0.1
            ]
            evicted = [
                item for item in surviving if item.activation < 0.1
            ]
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
    def pending_count(self) -> int:
        """Items waiting in the pending buffer.

        Subliminal content that failed to ignite but hasn't expired.
        Not part of the workspace — not cognitive — but available for
        coalition ignition on subsequent broadcasts.
        """
        with self._lock:
            return len(self._pending)

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
            self._pending.clear()


def _content_equal(a: Any, b: Any) -> bool:
    """Check if two content values are equal (handles unhashable types)."""
    if a is b:
        return True
    try:
        return a == b
    except Exception as e:
        logger.exception(f"content comparison failed: {e}")
        return False

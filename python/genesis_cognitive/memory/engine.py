"""Memory engine — retrieves and contextualises memories.

The subcognitive handles raw memory storage (STM ring buffer, LTM
episodic store). The memory engine is the cognitive layer's interface
to that storage: it retrieves relevant memories, builds context for
the current conversation, and tracks what Genesis remembers.

Key responsibilities:
- Retrieve memories similar to the current topic
- Track conversation history (short-term, in-process)
- Build a "memory context" that cognition uses when thinking
- Decide which memories are worth surfacing

In addition to retrieval, the engine models several biological memory
processes on the Python side (the Rust daemon stores the raw episodes;
this layer tracks the metadata needed for consolidation dynamics):

- **Sleep-specific consolidation** — SWS replays and transfers
  hippocampal memories to neocortex; REM consolidates emotional and
  procedural memories.
- **Forgetting** — Ebbinghaus decay, proactive/retroactive
  interference, retrieval failure, and adaptive forgetting.
- **Reconsolidation** — recalled memories become briefly labile and
  can be strengthened, weakened, or emotionally re-tagged.
- **Systems consolidation** — hippocampal dependency decays over time
  as memories become neocortical (the temporal gradient of retrograde
  amnesia).
- **Synaptic vs systems consolidation** — fast LTP-like strengthening
  (minutes-hours) is distinguished from slow redistribution
  (days-weeks).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from genesis_client import GenesisClient
from genesis_client.types import Episode, SimilarEpisode

from ..brain_waves import SleepStage
from ..config import MemoryConfig
from .systems import (
    AttractorNetwork,
    EmotionalMemorySystem,
    PrimingSystem,
    SpreadingActivation,
)
from .working import (
    CentralExecutive,
    PhonologicalLoop,
    VisuoSpatialSketchpad,
)

__all__ = [
    "ConversationTurn",
    "MemoryContext",
    "MemoryEngine",
    "MemoryRecord",
    "ReconsolidationModification",
    "SourceTag",
]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover — import only for type checkers
    from ..concepts import ConceptNetwork
    from ..emotion import EmotionalState


@dataclass(slots=True)
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "genesis"
    text: str
    timestamp: int
    topics: list[str] = field(default_factory=list)
    sentiment: float = 0.0


@dataclass(slots=True)
class MemoryContext:
    """What Genesis remembers when forming a response."""

    # Recent conversation (last N turns)
    recent_turns: list[ConversationTurn] = field(default_factory=list)

    # Memories retrieved from LTM that are relevant to current topic
    retrieved: list[SimilarEpisode] = field(default_factory=list)

    # Facts Genesis has learned about the user
    user_facts: dict[str, str] = field(default_factory=dict)

    # How many episodes are in LTM total
    total_memories: int = 0

    # Whether this is the first interaction
    is_first_interaction: bool = True

    def summary(self) -> str:
        """Brief text summary for introspection."""
        parts = []
        if self.total_memories > 0:
            parts.append(f"{self.total_memories} long-term memories")
        if self.retrieved:
            parts.append(f"{len(self.retrieved)} relevant memories recalled")
        if self.recent_turns:
            parts.append(f"{len(self.recent_turns)} turns in this conversation")
        if self.user_facts:
            parts.append(f"knows {len(self.user_facts)} facts about the user")
        return ", ".join(parts) if parts else "no memories yet"


# ─── Python-side memory metadata ─────────────────────────────────────


@dataclass(slots=True)
class MemoryRecord:
    """Python-side metadata tracking a single LTM episode.

    The Rust daemon stores the raw episode data (text, emotional tag,
    salience, hashes). This record tracks the *consolidation state*
    that the Python side needs for the biological memory processes:
    hippocampal dependency, reconsolidation lability, synaptic
    consolidation strength, access times, and forgetting state.

    Attributes:
        episode_id: The LTM episode ID (matches the Rust store).
        salience: Memory salience [0..1] — may be modified by
            reconsolidation.
        emotional_tag: 12-chemical emotional snapshot (mutable —
            reconsolidation can re-tag).
        encoded_at: When the memory was encoded (ms).
        last_accessed: When the memory was last recalled (ms).
        access_count: Number of times retrieved.
        hippocampal_dependency: 1.0 = fully hippocampal-dependent
            (recent), 0.0 = fully neocortical (remote). Decays over
            time via systems consolidation.
        synaptic_strength: Fast (LTP-like) consolidation strength
            [0..1]. Boosted immediately after encoding.
        systems_strength: Slow (systems) consolidation strength
            [0..1]. Grows as the memory is redistributed to neocortex.
        retention: Current retention level [0..1] after decay and
            interference. When it drops below the forgetting
            threshold, the memory is considered forgotten.
        forgotten: Whether this memory has been forgotten.
        reconsolidation_window: Remaining time (seconds) in the
            reconsolidation labile window. >0 means the memory is
            currently modifiable.
        last_reconsolidated: When the memory was last reconsolidated
            (ms), 0 if never.
        source_tag: Source monitoring tag (Johnson et al., 1993).
            Tracks where the memory came from (user, genesis,
            learning, inference, imagination). Used for reality
            monitoring and confabulation detection. ``None`` for
            memories created before source monitoring was added.
    """

    episode_id: int
    salience: float = 0.5
    emotional_tag: list[float] = field(default_factory=lambda: [0.5] * 12)
    encoded_at: int = 0
    last_accessed: int = 0
    access_count: int = 0
    hippocampal_dependency: float = 1.0
    synaptic_strength: float = 0.3
    systems_strength: float = 0.0
    retention: float = 1.0
    forgotten: bool = False
    reconsolidation_window: float = 0.0
    last_reconsolidated: int = 0
    reconsolidation_count: int = 0
    source_tag: SourceTag | None = None


# ─── Reconsolidation modification ────────────────────────────────────


@dataclass(slots=True)
class SourceTag:
    """Source monitoring tag for a memory (Johnson et al., 1993).

    Tracks where a memory came from — whether it was said by the user,
    generated by Genesis, learned from external docs, inferred, or
    imagined. This is the basis of reality monitoring: distinguishing
    internally-generated from externally-provided information.

    Without source monitoring, the system cannot detect confabulation
    (mistaking inferred or imagined content for observed fact) or
    source amnesia (remembering a fact but not where it came from).

    Attributes:
        source: Origin of the memory. One of:
            - "user": the user said this
            - "genesis": Genesis generated this (response, thought)
            - "conversation": a user+genesis interaction
            - "learning": learned from external source (docs, web)
            - "inference": Genesis inferred this (not directly observed)
            - "imagination": hypothetical / imagined
            - "self_analysis": Genesis analyzed her own code
        modality: How the memory was received — "text", "visual",
            "auditory", "internal".
        confidence: Confidence in the source attribution [0..1].
            Lower confidence means the source is uncertain, increasing
            the risk of source monitoring errors.
    """

    source: str = "unknown"
    modality: str = "text"
    confidence: float = 0.8


# Valid source categories for source monitoring.
VALID_SOURCES: frozenset[str] = frozenset({
    "user", "genesis", "conversation", "learning",
    "inference", "imagination", "self_analysis", "unknown",
})


@dataclass(slots=True)
class ReconsolidationModification:
    """A modification applied to a memory during reconsolidation.

    When a memory is recalled it becomes labile for a brief window
    (~6 hours in humans, scaled appropriately here). During this
    window the memory can be strengthened, weakened, or emotionally
    re-tagged. The current emotional state modulates the memory's
    emotional tag (reconsolidation is state-dependent).

    Attributes:
        strength_delta: Change to salience/retention (positive
            strengthens, negative weakens).
        emotional_tag: Optional new emotional tag to blend in.
        blend_factor: How much of the new emotional tag to blend in
            [0..1] (0 = keep old, 1 = replace entirely).
    """

    strength_delta: float = 0.0
    emotional_tag: list[float] | None = None
    blend_factor: float = 0.3


class MemoryEngine:
    """Manages memory retrieval and conversation tracking.

    Wraps the GenesisClient to provide a higher-level memory interface
    that the cognition system can use. In addition to retrieval, it
    models biological memory processes: sleep-specific consolidation,
    forgetting, reconsolidation, and synaptic vs systems consolidation.
    """

    # ── Consolidation / forgetting constants ─────────────────────
    # Ebbinghaus decay time constant (seconds). Higher tau → slower
    # forgetting. Salience scales tau: more salient memories decay
    # more slowly (retention = e^(-t / (tau * salience_factor))).
    _FORGETTING_TAU: float = 86400.0 * 7.0  # ~1 week baseline
    # Below this retention, a memory is considered forgotten.
    _FORGETTING_THRESHOLD: float = 0.05
    # Reconsolidation labile window (seconds). ~6 hours in humans,
    # scaled down for Genesis's faster clock.
    _RECONSOLIDATION_WINDOW: float = 3600.0 * 6.0
    # Boundary condition: reconsolidation only triggers when prediction
    # error (mismatch between expected and actual memory content) exceeds
    # this threshold. Recall without prediction error strengthens the
    # memory via retrieval practice but does NOT make it labile.
    # (Schiller et al., 2010; Sevenster et al., 2014)
    _RECONSOLIDATION_PE_THRESHOLD: float = 0.2
    # Retrieval practice boost — recalling a memory strengthens it
    # (Roediger & Karpicke, 2006). Applied on every successful recall
    # regardless of prediction error.
    _RETRIEVAL_PRACTICE_BOOST: float = 0.03
    # Retrieval-induced forgetting (Anderson et al., 1994): when a
    # memory is retrieved, competing memories (emotionally similar
    # but not retrieved) are weakened. This is the competitive
    # inhibition underlying retrieval practice costs.
    _RIF_COMPETITION_THRESHOLD: float = 0.6  # similarity above this = competition
    _RIF_WEAKENING_RATE: float = 0.03  # weakening per competing memory
    # Experience replay during NREM sleep (Kirkpatrick et al., 2017):
    # old memories are replayed during SWS to prevent catastrophic
    # forgetting. The number of old memories replayed per N3 cycle.
    _REPLAY_SAMPLE_SIZE: int = 10
    # Replay strengthening — less than recent replay (0.1) to avoid
    # over-strengthening old memories at the expense of new ones.
    _REPLAY_STRENGTHENING: float = 0.04
    # Hippocampal dependency decay rate per systems-consolidation
    # cycle. Each cycle reduces dependency by this fraction.
    _HIPPOCAMPAL_DECAY: float = 0.15
    # Synaptic consolidation boost applied immediately after encoding.
    _SYNAPTIC_BOOST: float = 0.4

    def __init__(
        self,
        client: GenesisClient,
        max_turns: int = 20,
        network: ConceptNetwork | None = None,
        get_emotion: Callable[[], EmotionalState | list[float]] | None = None,
        *,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize the memory engine with client, network, and emotion callback."""
        self.client = client
        self.config = config if config is not None else MemoryConfig()
        self.max_turns = max_turns
        self.conversation: list[ConversationTurn] = []
        self._user_facts: dict[str, str] = {}
        self._is_first = True

        # ── Python-side memory metadata (items 1-5) ──────────────
        # episode_id → MemoryRecord. Tracks consolidation state that
        # the Rust store doesn't hold.
        self._records: dict[int, MemoryRecord] = {}
        # Counters for introspection.
        self.forgotten_count: int = 0
        self.reconsolidation_count: int = 0
        self.synaptic_consolidation_count: int = 0
        self.systems_consolidation_count: int = 0
        self.sleep_consolidation_count: int = 0
        self.facts_replayed_count: int = 0

        # Optional callback: called with replayed episode text so the
        # semantic memory / concept network can extract relationships.
        # This closes the loop between episodic replay and neocortical
        # semantic storage (systems consolidation).
        self.semantic_replay_callback: Callable[[str], Any] | None = None
        # Optional callback: called when a high-emotion memory is stored
        # so the sleep system can queue it for REM emotional processing.
        self.emotional_memory_callback: Callable[[str, float], Any] | None = None

        # ── Wired-in memory subsystems ───────────────────────────
        # These were built as standalone modules but never connected
        # to the memory lifecycle. Here they are wired into storing
        # and retrieving so Genesis actually uses them.

        # Optional hooks supplied by the Mind:
        # - ``network``: the shared concept network, used by priming
        #   and spreading activation to propagate to neighbours.
        # - ``get_emotion``: callback returning the current emotional
        #   state (an EmotionalState or a 12-chemical vector), used
        #   for emotional tagging and mood-congruent retrieval.
        self._network = network
        self._get_emotion = get_emotion

        # Emotional memory (amygdala-dependent): tags memories with
        # emotional context and prioritises emotionally congruent
        # memories during retrieval (mood-congruent memory).
        self.emotional_memory = EmotionalMemorySystem()

        # Priming (implicit memory): recent concepts prime related
        # concepts, boosting their accessibility.
        self.priming = PrimingSystem(network=network)

        # Attractor networks (CA3 auto-association): pattern
        # completion — given a partial cue, complete the pattern to
        # the nearest stored memory.
        self.attractor = AttractorNetwork(size=64)

        # Spreading activation (Collins & Loftus, 1975): activating a
        # concept spreads to its neighbours, sustaining semantic
        # priming over time.
        self.spreading_activation = SpreadingActivation(network=network)

        # ── Enhanced working memory (Baddeley model) ─────────────
        # The MemoryEngine exposes the phonological loop (verbal
        # rehearsal buffer) and central executive (attention
        # controller) so other systems can use rehearsal and
        # attention direction through the memory engine.
        self.phonological_loop = PhonologicalLoop()
        self.visuospatial_sketchpad = VisuoSpatialSketchpad()
        self.central_executive = CentralExecutive(
            phonological_loop=self.phonological_loop,
            sketchpad=self.visuospatial_sketchpad,
        )

    def add_turn(
        self,
        role: str,
        text: str,
        topics: list[str] | None = None,
        sentiment: float = 0.0,
    ) -> None:
        """Record a conversation turn."""
        turn = ConversationTurn(
            role=role,
            text=text,
            timestamp=int(time.time() * 1000),
            topics=topics or [],
            sentiment=sentiment,
        )
        self.conversation.append(turn)
        if len(self.conversation) > self.max_turns:
            self.conversation = self.conversation[-self.max_turns :]
        if role == "user":
            self._is_first = False

        # Wire the turn's concepts into priming and spreading
        # activation so recently-discussed concepts become more
        # accessible (semantic priming).
        self._activate_concepts(turn.topics)

    def store_memory(
        self,
        text: str,
        salience: float = 0.5,
        emotional_tag: list[float] | None = None,
        event_type: int = 0,
        source_module: int = 4,
        source: str = "unknown",
        source_confidence: float = 0.8,
        memory_mode: str = "balanced",
    ) -> bool:
        """Store a deliberate memory directly in LTM.

        Deliberate memory stores (conversation turns, learning events)
        bypass STM and go straight to LTM via ``store_episode``, which
        returns the real LTM episode ID synchronously. This is distinct
        from raw pre-cognitive events (visual observations, spontaneous
        thoughts) which use ``store_event`` (the STM path) and are
        filtered by salience during consolidation.

        After storing, registers a :class:`MemoryRecord` for Python-side
        consolidation tracking and immediately applies *synaptic
        consolidation* — the fast LTP-like strengthening that happens
        within minutes of encoding (item 5).

        Args:
            text: The memory text content.
            salience: Memory salience [0..1].
            emotional_tag: 12-chemical emotional snapshot.
            event_type: Event type code for the daemon.
            source_module: Source module ID for the daemon.
            source: Source monitoring tag — where this memory came
                from. One of: "user", "genesis", "conversation",
                "learning", "inference", "imagination",
                "self_analysis", "unknown". Used for reality
                monitoring (Johnson et al., 1993).
            source_confidence: Confidence in the source attribution
                [0..1]. Lower values indicate uncertain source, which
                increases the risk of source monitoring errors.
            memory_mode: Theta-gamma coupling mode — "encoding",
                "retrieval", or "balanced". In encoding mode, salience
                is boosted (the hippocampus is biased toward forming
                new associations). In retrieval mode, salience is
                dampened (the hippocampus is reactivating, not
                encoding).
        """
        # Theta-gamma coupling gates encoding strength.
        # In encoding mode, the theta trough aligns gamma bursts for
        # new association formation — boost salience so the memory is
        # more likely to consolidate. In retrieval mode, dampen it.
        if memory_mode == "encoding":
            salience = min(1.0, salience * 1.2)
        elif memory_mode == "retrieval":
            salience = max(0.0, salience * 0.8)

        if emotional_tag is None:
            emotional_tag = [0.5] * 12
        # Clamp salience and emotional_tag to [0,1] with NaN/Inf
        # replaced by neutral defaults, matching the Rust IPC's
        # finite_clamp. This ensures the Python-side MemoryRecord,
        # the Rust-side LTM store, and the wired subsystems all
        # agree on the same values.
        salience = _finite_clamp(salience, 0.0, 1.0, default=0.5)
        emotional_tag = [_finite_clamp(v, 0.0, 1.0, default=0.5) for v in emotional_tag]
        eid = self.client.store_episode(
            timestamp=int(time.time() * 1000),
            event_type=event_type,
            source_module=source_module,
            salience=salience,
            emotional_tag=emotional_tag,
            text=text,
        )
        if eid is None:
            return False
        self._register_memory_record(
            eid, salience, emotional_tag, source, source_confidence
        )
        # Item 5: immediate synaptic consolidation.
        self.synaptic_consolidate(eid)

        # ── Wire the new memory into the subsystems ──────────
        self._wire_memory_subsystems(eid, text, emotional_tag)
        return True

    def _register_memory_record(
        self,
        eid: int,
        salience: float,
        emotional_tag: list[float],
        source: str,
        source_confidence: float,
    ) -> None:
        """Register a MemoryRecord for Python-side consolidation tracking.

        The ``eid`` is the real LTM episode ID returned by
        ``store_episode``. The record tracks consolidation state that
        the Rust store doesn't hold.

        Salience and emotional_tag are clamped to [0,1] with NaN/Inf
        replaced by neutral defaults, matching the Rust IPC's
        ``finite_clamp``. Without this, the Python-side record and the
        Rust-side store would disagree on salience when the caller
        passes out-of-range values, causing the forgetting formula
        (which scales tau by rec.salience) to produce incorrect decay.
        """
        clean_salience = _finite_clamp(salience, 0.0, 1.0, default=0.5)
        clean_tag = [_finite_clamp(v, 0.0, 1.0, default=0.5) for v in emotional_tag]
        record = MemoryRecord(
            episode_id=eid,
            salience=clean_salience,
            emotional_tag=clean_tag,
            encoded_at=int(time.time() * 1000),
            last_accessed=int(time.time() * 1000),
            source_tag=SourceTag(
                source=source if source in VALID_SOURCES else "unknown",
                modality="text",
                confidence=max(0.0, min(1.0, source_confidence)),
            ),
        )
        self._records[eid] = record

    def _wire_memory_subsystems(
        self, eid: int, text: str, emotional_tag: list[float]
    ) -> None:
        """Wire a newly stored memory into emotional, attractor, and priming subsystems.

        - Emotional memory: tag the memory with the current emotional state
          (mood-congruent encoding). The stimulus is the memory's primary
          concept (or a short text key) so it can be retrieved later by
          emotional congruence.
        - Attractor network: store a pattern for this memory so partial cues
          can be completed to it later (CA3 pattern completion).
        - Priming / spreading activation: activate the concepts embedded in
          the memory text so related concepts become more accessible.
        """
        self._store_emotional_tag(text, emotional_tag)
        self.attractor.store(str(eid), _text_to_vector(text, self.attractor.size))
        self._activate_concepts(_extract_concepts(text))
        # Feed high-emotion memories to the sleep system for REM
        # emotional processing. Without this, the REM emotional
        # memory queue is always empty.
        if self.emotional_memory_callback is not None:
            intensity = _emotional_intensity(emotional_tag)
            if intensity > 0.3:
                try:
                    self.emotional_memory_callback(text[:200], intensity)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"emotional memory callback failed: {e}")

    def retrieve_relevant(
        self,
        query: str,
        limit: int = 5,
        prediction_error: float = 0.0,
    ) -> list[SimilarEpisode]:
        """Retrieve memories similar to the query text.

        Retrieved memories are marked as accessed. The reconsolidation
        labile window only opens if ``prediction_error`` exceeds the
        boundary threshold — recall without prediction error
        strengthens the memory (retrieval practice) but does not make
        it labile (Schiller et al., 2010; Sevenster et al., 2014).

        The retrieval pipeline now uses the wired-in subsystems:

        - **Attractor networks**: the query is fed to the attractor
          network for pattern completion. If a stored pattern is
          recovered, the corresponding episode is ensured a place in
          the results (pattern completion from a partial cue).
        - **Spreading activation / priming**: concepts in the query
          are activated so semantically related concepts become more
          accessible for subsequent retrievals.
        - **Mood-congruent memory**: results are re-ranked so that
          memories whose emotional tag matches the current emotional
          state are prioritised (Bower, 1981).

        Args:
            query: The text to search for.
            limit: Maximum number of results.
            prediction_error: Mismatch between expected and actual
                memory content [0..1]. When this exceeds the
                reconsolidation threshold, recalled memories become
                labile. Defaults to 0.0 (retrieval practice only, no
                reconsolidation).
        """
        # Activate the query's concepts (semantic priming for future
        # retrievals) before searching.
        _t0 = time.perf_counter()
        query_concepts = _extract_concepts(query)
        self._activate_concepts(query_concepts)
        _t_activate = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        try:
            results = self.client.find_similar(query, limit=limit)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"find similar failed: {e}")
            results = []
        results = list(results)
        _t_find = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        # Pattern completion: feed the query to the attractor network.
        # If it converges to a stored pattern whose episode is not
        # already in the results, try to retrieve that episode and
        # prepend it (the cue was partial — the network completed it).
        if results is not None:
            results = self._apply_attractor_completion(results, query, limit)
        _t_attractor = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        # Emotional memory bias: compute a weighted emotional tag for
        # the query concepts, then re-rank by emotional congruence.
        # Emotional associations shape what we remember (Bower, 1981;
        # Phelps, 2004).
        query_emotion = self._compute_query_emotion(query_concepts)
        results = self._rank_by_emotional_congruence(
            results, query_emotion=query_emotion
        )
        _t_emotion = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        now_ms = int(time.time() * 1000)
        opens_window = prediction_error >= self.config.reconsolidation_pe_threshold
        retrieved_ids = [r.episode_id for r in results]
        for r in results:
            self._mark_accessed(r.episode_id, now_ms, opens_window)
        # Retrieval-induced forgetting: weaken competing (emotionally
        # similar but non-retrieved) memories (Anderson et al., 1994).
        self._apply_retrieval_induced_forgetting(retrieved_ids)
        # Source monitoring: flag high-confabulation-risk memories so
        # the cognition engine can treat them with appropriate caution.
        # Memories from "inference" or "imagination" with low source
        # confidence are more likely to be confabulated (Johnson et al.,
        # 1993). We don't remove them — they may still be useful — but
        # we log the risk for introspection.
        for r in results:
            risk = self.check_confabulation_risk(r.episode_id)
            if risk > 0.7:
                logger.debug(
                    f"high confabulation risk ({risk:.2f}) for episode {r.episode_id}"
                )
        _t_post = time.perf_counter() - _t0
        logger.debug(
            f"retrieve_relevant sub-timing: "
            f"activate={_t_activate:.3f}s "
            f"find_similar={_t_find:.3f}s "
            f"attractor={_t_attractor:.3f}s "
            f"emotion={_t_emotion:.3f}s "
            f"post={_t_post:.3f}s "
            f"query='{query[:60]}'"
        )
        return results

    def _compute_query_emotion(
        self, query_concepts: list[str]
    ) -> list[float] | None:
        """Compute a weighted emotional tag for the query concepts.

        Retrieves emotional associations for the query concepts and
        produces a weighted-average emotional tag. This biases
        retrieval toward memories whose emotional tag matches the
        emotional associations of the query concepts (not just the
        current mood) — emotional associations shape what we remember
        (Bower, 1981; Phelps, 2004).
        """
        em_weights: list[tuple[list[float], float]] = []
        current_emotion = self._current_emotion_vector()
        for concept in query_concepts:
            em = self.emotional_memory.retrieve_emotional(concept, current_emotion)
            if em is not None and not em.is_extinct:
                em_weights.append((em.emotional_tag, em.association_strength))
        if em_weights:
            total = sum(w for _, w in em_weights)
            if total > 0:
                return [
                    sum(tag[i] * w for tag, w in em_weights) / total
                    for i in range(12)
                ]
        return None

    def retrieve_episode(
        self,
        episode_id: int,
        prediction_error: float = 0.0,
    ) -> Episode | None:
        """Retrieve a specific episode by ID.

        Marks the memory as accessed. The reconsolidation labile window
        only opens if ``prediction_error`` exceeds the boundary
        threshold (Schiller et al., 2010). Without prediction error,
        the memory is strengthened by retrieval practice but does not
        become labile.

        Args:
            episode_id: The episode ID to retrieve.
            prediction_error: Mismatch between expected and actual
                memory content [0..1]. Defaults to 0.0.
        """
        try:
            ep = self.client.retrieve_episode(episode_id)
        except Exception as e:  # noqa: BLE001
            # Expected when sleep compression has pruned the episode.
            logger.debug(f"retrieve episode {episode_id} failed (likely pruned): {e}")
            return None
        if ep is not None:
            opens_window = prediction_error >= self.config.reconsolidation_pe_threshold
            self._mark_accessed(episode_id, int(time.time() * 1000), opens_window)
        return ep

    def _mark_accessed(self, episode_id: int, now_ms: int, opens_window: bool) -> None:
        """Mark a memory as accessed, applying retrieval practice and
        optionally opening the reconsolidation window.

        Retrieval practice (Roediger & Karpicke, 2006): every
        successful recall strengthens the memory slightly, regardless
        of prediction error. The reconsolidation labile window only
        opens when ``opens_window`` is True (i.e. prediction error
        exceeded the boundary threshold).
        """
        rec = self._records.get(episode_id)
        if rec is None or rec.forgotten:
            return
        rec.last_accessed = now_ms
        rec.access_count += 1
        # Retrieval practice: recalling a memory strengthens it.
        rec.retention = min(1.0, rec.retention + self._RETRIEVAL_PRACTICE_BOOST)
        rec.synaptic_strength = min(1.0, rec.synaptic_strength + self._RETRIEVAL_PRACTICE_BOOST)
        # Boundary condition: only open the reconsolidation window
        # when prediction error exceeds threshold.
        if opens_window:
            rec.reconsolidation_window = self.config.reconsolidation_window_seconds

    def _replay_into_semantic_graph(self, episode_id: int) -> int:
        """Replay a single episode's text into the semantic graph.

        Called during sleep consolidation. If a ``semantic_replay_callback``
        is wired (e.g. :meth:`SemanticMemory.extract_facts`), the episode
        text is passed to it and the returned fact count is accumulated.

        Returns:
            Number of facts extracted from the replayed episode.
        """
        if self.semantic_replay_callback is None:
            return 0

        try:
            ep = self.client.retrieve_episode(episode_id)
        except Exception as e:  # noqa: BLE001
            # Expected when sleep compression has pruned the episode.
            logger.debug(f"replay retrieve {episode_id} failed (likely pruned): {e}")
            return 0
        if ep is None or not ep.text:
            return 0

        # Do not mark the memory as accessed here — this is an offline
        # replay, not an explicit recall. The consolidation methods
        # already update synaptic strength separately.
        try:
            result = self.semantic_replay_callback(ep.text)
            if isinstance(result, int):
                self.facts_replayed_count += result
                return result
            if isinstance(result, list):
                n_facts = len(result)
                self.facts_replayed_count += n_facts
                return n_facts
            # Treat any other truthy result as one fact extracted.
            self.facts_replayed_count += 1
            return 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"semantic replay callback failed: {e}")
            return 0

    def _apply_retrieval_induced_forgetting(
        self,
        retrieved_ids: list[int],
    ) -> int:
        """Apply retrieval-induced forgetting (Anderson et al., 1994).

        When memories are retrieved, competing memories — those with
        high emotional similarity to the retrieved ones but not
        themselves retrieved — are weakened. This models the
        competitive inhibition that underlies retrieval practice
        costs: practicing some facts makes related but unpracticed
        facts harder to recall.

        The weakening is proportional to emotional similarity (the
        closer a memory's emotional tag is to a retrieved memory's,
        the more it competes and the more it is weakened). Memories
        that are high in salience or emotional intensity are more
        resistant to RIF.

        Returns the number of memories weakened.
        """
        if not retrieved_ids:
            return 0
        retrieved_set = set(retrieved_ids)
        weakened = 0

        # Collect the emotional tags of retrieved memories.
        retrieved_tags: list[list[float]] = []
        for eid in retrieved_ids:
            rec = self._records.get(eid)
            if rec is not None and not rec.forgotten:
                retrieved_tags.append(rec.emotional_tag)

        if not retrieved_tags:
            return 0

        # Gather candidate records (non-retrieved, non-forgotten).
        candidate_eids: list[int] = []
        candidate_tags: list[list[float]] = []
        for eid, rec in self._records.items():
            if eid in retrieved_set or rec.forgotten:
                continue
            candidate_eids.append(eid)
            candidate_tags.append(rec.emotional_tag)

        if not candidate_eids:
            return 0

        # Vectorized cosine similarity: (M × D) @ (D × R) → (M × R).
        # Take the max over R to find each candidate's strongest competitor.
        cand_mat = np.array(candidate_tags, dtype=np.float64)
        retr_mat = np.array(retrieved_tags, dtype=np.float64)
        # Pad to matching column count if tags differ in length.
        d = max(cand_mat.shape[1], retr_mat.shape[1])
        if cand_mat.shape[1] < d:
            cand_mat = np.pad(cand_mat, ((0, 0), (0, d - cand_mat.shape[1])))
        if retr_mat.shape[1] < d:
            retr_mat = np.pad(retr_mat, ((0, 0), (0, d - retr_mat.shape[1])))

        cand_norms = np.linalg.norm(cand_mat, axis=1)
        retr_norms = np.linalg.norm(retr_mat, axis=1)
        # Avoid division by zero.
        cand_norms[cand_norms < 1e-9] = 1e-9
        retr_norms[retr_norms < 1e-9] = 1e-9

        sim_matrix = (cand_mat @ retr_mat.T) / np.outer(cand_norms, retr_norms)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
        max_sims = sim_matrix.max(axis=1)

        threshold = self.config.rif_competition_threshold
        for idx, eid in enumerate(candidate_eids):
            max_similarity = float(max_sims[idx])
            if max_similarity < threshold:
                continue
            rec = self._records[eid]
            # High-salience and emotionally intense memories resist RIF.
            resistance = 0.5 * rec.salience + 0.3 * _emotional_intensity(rec.emotional_tag)
            weakening = self._RIF_WEAKENING_RATE * max_similarity * (1.0 - resistance)
            rec.retention = max(0.0, rec.retention - weakening)
            rec.synaptic_strength = max(0.0, rec.synaptic_strength - weakening)
            weakened += 1
            # Check if the memory has been forgotten as a result.
            if rec.retention < self.config.forgetting_threshold:
                rec.forgotten = True
                self.forgotten_count += 1
        return weakened

    def learn_user_fact(self, key: str, value: str) -> None:
        """Learn a fact about the user."""
        self._user_facts[key] = value

    def get_user_facts(self) -> dict[str, str]:
        """Return known facts about the user."""
        return dict(self._user_facts)

    def build_context(
        self, current_topics: list[str], memory_mode: str = "balanced"
    ) -> MemoryContext:
        """Build a memory context for the current conversation state.

        Retrieves relevant memories and assembles them with conversation
        history into a context the cognition system can use.

        The retrieval query is enriched by the wired-in subsystems:

        - **Priming**: currently-primed concepts are appended to the
          query so recently-activated topics bias retrieval.
        - **Spreading activation**: concepts with residual spreading
          activation are appended, modelling semantic priming.
        - **Attractor networks / mood-congruent memory**: applied
          inside :meth:`retrieve_relevant`.

        Args:
            current_topics: Topics to retrieve memories for.
            memory_mode: Theta-gamma coupling mode — "encoding",
                "retrieval", or "balanced". In retrieval mode, more
                memories are fetched (the hippocampus is biased toward
                reactivation). In encoding mode, fewer are fetched
                (the hippocampus is biased toward forming new
                associations, not retrieving old ones).
        """
        # Theta-gamma coupling gates retrieval breadth.
        # In retrieval mode, the theta peak aligns gamma bursts for
        # memory reactivation — broaden the search. In encoding mode,
        # the theta trough aligns gamma for new associations — narrow
        # the search so the focus is on encoding, not retrieval.
        retrieval_limit = 5
        if memory_mode == "retrieval":
            retrieval_limit = 7
        elif memory_mode == "encoding":
            retrieval_limit = 3

        # Try to retrieve relevant memories from LTM
        retrieved: list[SimilarEpisode] = []
        if current_topics:
            query = self._enrich_query(" ".join(current_topics))
            retrieved = self.retrieve_relevant(query, limit=retrieval_limit)

        # Get total memory count
        try:
            stats = self.client.get_memory_stats()
            total = stats.ltm_count
        except Exception as e:  # noqa: BLE001
            logger.warning(f"get memory stats failed: {e}")
            total = 0

        return MemoryContext(
            recent_turns=list(self.conversation[-10:]),
            retrieved=retrieved,
            user_facts=dict(self._user_facts),
            total_memories=total,
            is_first_interaction=self._is_first,
        )

    def get_recent_topics(self, n: int = 5) -> list[str]:
        """Get topics from the last N conversation turns."""
        topics: list[str] = []
        for turn in self.conversation[-n:]:
            topics.extend(turn.topics)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    # ═══════════════════════════════════════════════════════════════
    #  Wired-in memory subsystem helpers
    # ═══════════════════════════════════════════════════════════════

    def _current_emotion_vector(self) -> list[float] | None:
        """Return the current emotional state as a 12-chemical vector.

        Uses the ``get_emotion`` callback supplied at construction.
        The callback may return either an :class:`EmotionalState`
        (whose ``chemicals`` dict is mapped to the 12-chemical order,
        defaulting to 0.5 for missing chemicals) or a raw 12-element
        list. Returns ``None`` if no callback is wired.
        """
        if self._get_emotion is None:
            return None
        try:
            state = self._get_emotion()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"get emotion failed: {e}")
            return None
        if state is None:
            return None
        if isinstance(state, list):
            return list(state)
        # EmotionalState — build a 12-vector from the chemicals dict.
        from genesis_client.protocol import CHEM_NAMES

        vector = [0.5] * 12
        chemicals = getattr(state, "chemicals", {}) or {}
        for idx in range(12):
            name = CHEM_NAMES.get(idx)
            if name is not None and name in chemicals:
                vector[idx] = float(chemicals[name])
        return vector

    def _activate_concepts(self, concepts: list[str]) -> None:
        """Activate concepts in priming and spreading activation.

        After each interaction, the discussed concepts prime related
        concepts (PrimingSystem) and spread activation through the
        concept network (SpreadingActivation). This is how semantic
        priming works — related concepts become more accessible.
        """
        for concept in concepts:
            if not concept:
                continue
            self.priming.prime(concept)
            self.spreading_activation.activate(concept)

    def _store_emotional_tag(
        self,
        text: str,
        emotional_tag: list[float],
    ) -> None:
        """Tag a memory with emotional context (mood-congruent encoding).

        Records an emotional association in the
        :class:`EmotionalMemorySystem` keyed by the memory's primary
        concept, using the *current* emotional state (from the
        ``get_emotion`` callback) when available, falling back to the
        memory's own emotional tag. The association strength scales
        with emotional intensity — emotionally charged memories are
        encoded more strongly (amygdala modulation of hippocampal
        encoding; McGaugh, 2004).

        Fear conditioning is reserved for negatively-valenced memories
        (high cortisol / low dopamine). Positive memories are tagged
        via their ``MemoryRecord.emotional_tag`` but do not create
        aversive associations.
        """
        stimulus = _primary_concept(text)
        if not stimulus:
            return
        # Prefer the live emotional state; fall back to the memory's
        # own tag.
        current = self._current_emotion_vector()
        tag = current if current is not None else emotional_tag
        intensity = _emotional_intensity(tag)
        # Only record an association when there is meaningful
        # emotional content; otherwise the memory is still tagged via
        # its MemoryRecord.emotional_tag.
        if intensity <= 0.05:
            return
        # Fear conditioning is specifically for aversive memories.
        # Cortisol (idx 6) is the stress hormone; dopamine (idx 0)
        # tracks valence. Only condition when the tag indicates
        # negative valence — sending positive memories through fear
        # conditioning would create spurious aversive associations.
        cortisol_idx = 6
        dopamine_idx = 0
        cortisol = tag[cortisol_idx] if len(tag) > cortisol_idx else 0.0
        dopamine = tag[dopamine_idx] if len(tag) > dopamine_idx else 0.5
        if cortisol <= 0.5 and dopamine >= 0.5:
            return
        self.emotional_memory.fear_conditioning(
            neutral_stimulus=stimulus,
            negative_emotion_strength=intensity,
            emotional_tag=list(tag),
        )

    def _enrich_query(self, query: str) -> str:
        """Enrich a retrieval query with primed/activated concepts.

        Currently-primed concepts (PrimingSystem) and concepts with
        residual spreading activation are appended to the query so
        that recently-activated topics bias retrieval (semantic
        priming).
        """
        extras: list[str] = []
        try:
            primed = self.priming.get_all_primed(threshold=self.config.priming_threshold)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"get primed concepts failed: {e}")
            primed = {}
        for concept, _level in sorted(primed.items(), key=lambda kv: kv[1], reverse=True):
            if concept.lower() not in query.lower():
                extras.append(concept)
        try:
            activated = self.spreading_activation.get_all(threshold=self.config.spreading_threshold)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"get activated concepts failed: {e}")
            activated = {}
        for concept, _level in sorted(activated.items(), key=lambda kv: kv[1], reverse=True):
            if concept.lower() not in query.lower() and concept not in extras:
                extras.append(concept)
        if not extras:
            return query
        # Limit the number of extra concepts to avoid diluting the
        # query too much.
        return query + " " + " ".join(extras[:5])

    def _apply_attractor_completion(
        self,
        results: list[SimilarEpisode],
        query: str,
        limit: int,
    ) -> list[SimilarEpisode]:
        """Use the attractor network for pattern completion.

        Feeds the query (a partial cue) to the Hopfield network. If
        it converges to a stored pattern whose episode is not already
        present in the results, the corresponding episode is fetched
        and prepended — the cue was partial and the network completed
        it (CA3 auto-association).
        """
        if self.attractor.pattern_count == 0:
            return results
        import time as _attractor_time
        _t0 = _attractor_time.perf_counter()
        try:
            _completed, match_id = self.attractor.retrieve(
                _text_to_vector(query, self.attractor.size)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"attractor retrieve failed: {e}")
            match_id = None
        _t_retrieve = _attractor_time.perf_counter() - _t0
        if match_id is None:
            logger.debug(
                f"attractor: retrieve={_t_retrieve:.3f}s "
                f"patterns={self.attractor.pattern_count} "
                f"no match"
            )
            return results
        try:
            eid = int(match_id)
        except (TypeError, ValueError):
            return results
        # Already present — boost it to the front.
        for r in results:
            if r.episode_id == eid:
                results.remove(r)
                results.insert(0, r)
                return results[:limit]
        # Not in results — try to fetch the episode and synthesize a
        # result entry for it.
        _t0 = _attractor_time.perf_counter()
        try:
            ep = self.client.retrieve_episode(eid)
        except Exception as e:  # noqa: BLE001
            # Expected when sleep compression has pruned the episode.
            logger.debug(f"retrieve episode {eid} failed (likely pruned): {e}")
            ep = None
        _t_fetch = _attractor_time.perf_counter() - _t0
        if ep is None:
            logger.debug(
                f"attractor: retrieve={_t_retrieve:.3f}s "
                f"fetch={_t_fetch:.3f}s "
                f"patterns={self.attractor.pattern_count} "
                f"match={eid} but episode pruned"
            )
            return results
        completed = SimilarEpisode(
            episode_id=eid,
            hamming_distance=0,
            timestamp=ep.timestamp,
            salience=ep.salience,
        )
        results.insert(0, completed)
        # CA3 auto-association: find other stored patterns similar
        # to the completed pattern. These are related memories that
        # should also be surfaced — the hippocampal CA3 region
        # activates similar memories when one is recalled.
        _t0 = _attractor_time.perf_counter()
        similar_ids = self.attractor.auto_associate(str(eid))
        _t_auto = _attractor_time.perf_counter() - _t0
        _t0 = _attractor_time.perf_counter()
        _fetched = 0
        for sid in similar_ids:
            try:
                sid_eid = int(sid)
            except (TypeError, ValueError):
                continue
            # Skip if already in results
            if any(r.episode_id == sid_eid for r in results):
                continue
            try:
                sep = self.client.retrieve_episode(sid_eid)
            except Exception:  # noqa: BLE001
                continue
            if sep is not None:
                results.append(SimilarEpisode(
                    episode_id=sid_eid,
                    hamming_distance=0,
                    timestamp=sep.timestamp,
                    salience=sep.salience,
                ))
                _fetched += 1
        _t_assoc_fetch = _attractor_time.perf_counter() - _t0
        logger.debug(
            f"attractor: retrieve={_t_retrieve:.3f}s "
            f"fetch={_t_fetch:.3f}s "
            f"auto_assoc={_t_auto:.3f}s "
            f"assoc_fetch={_t_assoc_fetch:.3f}s "
            f"patterns={self.attractor.pattern_count} "
            f"similar_ids={len(similar_ids)} fetched={_fetched}"
        )
        return results[:limit]

    def _rank_by_emotional_congruence(
        self,
        results: list[SimilarEpisode],
        query_emotion: list[float] | None = None,
    ) -> list[SimilarEpisode]:
        """Re-rank results by mood congruence (Bower, 1981).

        Memories whose stored emotional tag matches the current
        emotional state are boosted in the ranking. The base score is
        derived from salience (higher is better) and hamming distance
        (lower is better); an emotional-congruence bonus is added on
        top.

        If ``query_emotion`` is provided (a 12-chemical emotional tag
        derived from the query's emotional memory associations), it is
        blended with the current mood to form the target emotional
        state for congruence scoring. This biases retrieval toward
        memories that match the emotional associations of the query
        concepts, not just the current mood.
        """
        if not results:
            return results
        current = self._current_emotion_vector()
        if current is None and query_emotion is None:
            return results

        # Blend the current mood with the query's emotional associations.
        # When both are available, the query emotion gets more weight
        # (0.6) because it reflects the emotional context of the
        # retrieval, not just the ambient mood.
        if current is not None and query_emotion is not None:
            target = [0.4 * current[i] + 0.6 * query_emotion[i] for i in range(12)]
        elif query_emotion is not None:
            target = query_emotion
        else:
            target = current  # type: ignore[assignment]

        def _score(r: SimilarEpisode) -> float:
            """Score a retrieved episode by salience, recency, and emotional congruence."""
            # Base score: salience minus normalised hamming distance.
            # hamming_distance is the number of differing bits between
            # 64-bit SimHash fingerprints, so it ranges [0, 64].
            # Dividing by 64 normalises it to [0, 1] to match the
            # salience range [0, 1], so the two terms contribute on the
            # same scale.
            base = r.salience - (r.hamming_distance / 64.0)
            rec = self._records.get(r.episode_id)
            if rec is None or rec.forgotten:
                return base
            congruence = _emotional_similarity(target, rec.emotional_tag)
            return base + congruence

        return sorted(results, key=_score, reverse=True)

    # ── Enhanced working-memory access ───────────────────────────

    def rehearse(self, item: str) -> bool:
        """Rehearse a verbal item in the phonological loop.

        Exposes the phonological loop's rehearsal so other systems can
        keep a concept "in mind" through the memory engine (the
        articulatory control process; Baddeley, 1992).

        Returns:
            ``True`` if the item was present and rehearsed.
        """
        return self.phonological_loop.rehearse_item(item)

    def direct_attention(
        self,
        items: dict[str, float],
        relevant: list[str],
    ) -> dict[str, float]:
        """Direct attention to relevant concepts via the central executive.

        Exposes the central executive's attention-direction function
        so other systems can boost relevant concepts and suppress
        irrelevant ones through the memory engine.
        """
        return self.central_executive.direct_attention(items, relevant)

    @property
    def turn_count(self) -> int:
        """Total number of turns in this conversation."""
        return len(self.conversation)

    @property
    def is_first_interaction(self) -> bool:
        """Whether this is the first interaction in the session."""
        return self._is_first

    # ═══════════════════════════════════════════════════════════════
    #  Item 1: Sleep-specific consolidation
    # ═══════════════════════════════════════════════════════════════

    def consolidate_during_sleep(
        self, sleep_stage: SleepStage, duration: float, consolidation_intensity: float = 0.5
    ) -> int:
        """Consolidate memories appropriately for the given sleep stage.

        Sleep consolidates memory in a stage-specific manner
        (Diekelmann & Born, 2010):

        - **N3 (slow-wave sleep)**: Recent hippocampal memories are
          replayed in compressed time and transferred to neocortical
          storage (systems consolidation). Spindle-ripple coupling
          (thalamocortical spindles nested with hippocampal
          sharp-wave ripples) drives this transfer. SWS favours
          declarative memory.
        - **N2**: Sleep spindles drive hippocampal→neocortical
          transfer — the bulk of memory transfer happens here.
        - **REM sleep**: Emotional and procedural memories are
          consolidated. The amygdala is highly active during REM,
          strengthening emotional associations. REM favours
          procedural and emotional memory.
        - **N1**: The lightest NREM stage — minor consolidation.

        Args:
            sleep_stage: The current sleep stage (from
                :class:`~genesis_cognitive.brain_waves.SleepStage`).
            duration: Duration of the sleep period in seconds.
            consolidation_intensity: Brain-wave-derived consolidation
                drive [0, 1] from ``BrainWaveState.consolidation``.
                Scales how much consolidation happens — high theta/
                delta power with high neurochemical consolidation
                weight produces more replay and transfer. Default
                0.5 (moderate).

        Returns:
            Number of memories processed during this sleep period.
        """
        processed = 0
        now_ms = int(time.time() * 1000)
        # Recent memories = encoded within ~2x the duration window.
        recent_cutoff = now_ms - int(duration * 2000)

        # Scale the consolidation strength by the brain-wave-derived
        # consolidation intensity. High theta/delta power + high
        # neurochemical consolidation weight → stronger replay and
        # transfer. Low intensity → weaker consolidation (e.g. during
        # stress, when cortisol suppresses consolidation).
        strength_scale = 0.3 + 0.7 * max(0.0, min(1.0, consolidation_intensity))

        if sleep_stage == SleepStage.N3:
            # SWS: replay recent memories + systems consolidation.
            processed = self._consolidate_n3(recent_cutoff, strength_scale)
        elif sleep_stage == SleepStage.N2:
            # N2: sleep spindles drive hippocampal→neocortical transfer.
            # The bulk of memory transfer happens in N2.
            processed = self._consolidate_n2(strength_scale)
        elif sleep_stage == SleepStage.REM:
            # REM: consolidate emotional and procedural memories.
            processed = self._consolidate_rem(recent_cutoff, strength_scale)
        elif sleep_stage == SleepStage.N1:
            # N1: the lightest stage — minor replay.
            processed = self._consolidate_n1(recent_cutoff, strength_scale)

        self.sleep_consolidation_count += processed

        # Forgetting runs during N2 sleep — the brain prunes irrelevant
        # memories during sleep spindles. N3 is for replay and
        # consolidation (strengthening), so forgetting during N3 would
        # counteract the strengthening. N2 is for transfer, making it
        # the right stage for pruning memories that aren't being
        # transferred. Ebbinghaus decay, interference, and adaptive
        # forgetting all apply here.
        if sleep_stage == SleepStage.N2:
            self.forget(now=time.time())

        return processed

    def _consolidate_n3(self, recent_cutoff: int, strength_scale: float = 1.0) -> int:
        """SWS: replay recent memories + systems consolidation + experience replay."""
        processed = 0
        replayed_ids: list[int] = []
        for rec in self._records.values():
            if rec.forgotten:
                continue
            if rec.encoded_at >= recent_cutoff:
                # Replay in compressed time strengthens the trace.
                rec.synaptic_strength = min(1.0, rec.synaptic_strength + 0.1 * strength_scale)
                rec.retention = min(1.0, rec.retention + 0.05 * strength_scale)
                replayed_ids.append(rec.episode_id)
                processed += 1
            # Spindle-ripple coupling drives hippocampal→neocortical
            # transfer for ALL non-forgotten memories (not just
            # recent ones — the transfer is gradual).
            if rec.hippocampal_dependency > 0.0:
                rec.hippocampal_dependency = max(
                    0.0,
                    rec.hippocampal_dependency - self._HIPPOCAMPAL_DECAY * strength_scale,
                )
                rec.systems_strength = min(
                    1.0, rec.systems_strength + self._HIPPOCAMPAL_DECAY * strength_scale
                )
                # As memories become neocortical, retention stabilises
                # (remote memories are more resistant to forgetting).
                rec.retention = min(1.0, rec.retention + 0.02 * strength_scale)
        # Experience replay: sample old memories by inverse recency
        # and replay them to prevent catastrophic forgetting.
        processed += self._replay_old_memories(recent_cutoff, replayed_ids)
        # Replay the selected episode texts into the semantic graph.
        for eid in replayed_ids:
            self._replay_into_semantic_graph(eid)
        return processed

    def _replay_old_memories(
        self,
        recent_cutoff: int,
        replayed_ids: list[int] | None = None,
    ) -> int:
        """Replay old memories during NREM sleep to prevent catastrophic forgetting.

        Experience replay (Kirkpatrick et al., 2017; McClelland et al.,
        1995): during slow-wave sleep, the hippocampus replays old
        memories interleaved with new ones, preventing new learning
        from overwriting old knowledge. Without this, continual
        learning causes catastrophic forgetting — early knowledge
        degrades as new domains are learned.

        Old memories are sampled with probability proportional to
        inverse recency (older = more likely to be replayed) and
        inverse access count (rarely accessed = more likely, since
        they are more vulnerable to forgetting). Recent memories
        (already replayed above) are excluded from the sample.

        Returns the number of old memories replayed.
        """
        # Collect candidate memories: non-forgotten, not recently
        # encoded (older than the recent cutoff).
        candidates: list[MemoryRecord] = []
        for rec in self._records.values():
            if rec.forgotten or rec.encoded_at >= recent_cutoff:
                continue
            candidates.append(rec)

        if not candidates:
            return 0

        # Compute sampling weights: inverse recency × inverse access.
        # Older, rarely-accessed memories get higher weights.
        now_ms = int(time.time() * 1000)
        weights: list[float] = []
        for rec in candidates:
            age_s = max(1.0, (now_ms - rec.encoded_at) / 1000.0)
            access_factor = 1.0 / max(1, rec.access_count)
            # Weight = age (in seconds) × inverse access count.
            # Normalized later by random.choices.
            weights.append(age_s * access_factor)

        # Sample without replacement (a memory is replayed at most once
        # per N3 cycle). Use random.choices for weighted sampling, then
        # deduplicate.
        sample_size = min(self._REPLAY_SAMPLE_SIZE, len(candidates))
        indices = list(range(len(candidates)))
        chosen: set[int] = set()
        # Weighted sampling without replacement: repeatedly pick the
        # highest-weighted candidate with some randomness.
        for _ in range(sample_size):
            if not indices:
                break
            pick = random.choices(indices, weights=[weights[i] for i in indices], k=1)[0]
            chosen.add(pick)
            indices.remove(pick)

        replayed = 0
        for idx in chosen:
            rec = candidates[idx]
            # Replay strengthens the trace (less than recent replay).
            rec.synaptic_strength = min(1.0, rec.synaptic_strength + self._REPLAY_STRENGTHENING)
            rec.retention = min(1.0, rec.retention + self._REPLAY_STRENGTHENING * 0.5)
            # Replay counts as a weak access — keeps the memory "alive".
            rec.last_accessed = now_ms
            if replayed_ids is not None:
                replayed_ids.append(rec.episode_id)
            replayed += 1
        return replayed

    def _consolidate_n2(self, strength_scale: float = 1.0) -> int:
        """N2: sleep spindles drive hippocampal→neocortical transfer."""
        processed = 0
        replayed = 0
        for rec in self._records.values():
            if rec.forgotten:
                continue
            if rec.hippocampal_dependency > 0.0:
                rec.hippocampal_dependency = max(
                    0.0,
                    rec.hippocampal_dependency - self._HIPPOCAMPAL_DECAY * 0.7 * strength_scale,
                )
                rec.systems_strength = min(
                    1.0,
                    rec.systems_strength + self._HIPPOCAMPAL_DECAY * 0.7 * strength_scale,
                )
                # N2 is the main transfer stage — replay a sample of
                # episodes into the semantic graph.
                if replayed < self._REPLAY_SAMPLE_SIZE:
                    self._replay_into_semantic_graph(rec.episode_id)
                    replayed += 1
                processed += 1
        return processed

    def _consolidate_rem(self, recent_cutoff: int, strength_scale: float = 1.0) -> int:
        """REM: consolidate emotional and procedural memories."""
        processed = 0
        replayed = 0
        # Emotional memories are identified by high emotional
        # variance in the 12-chemical tag.
        for rec in self._records.values():
            if rec.forgotten:
                continue
            emotional_intensity = _emotional_intensity(rec.emotional_tag)
            if emotional_intensity > 0.6 or rec.encoded_at >= recent_cutoff:
                # Emotional memories get extra consolidation.
                rec.synaptic_strength = min(
                    1.0, rec.synaptic_strength + 0.08 * emotional_intensity * strength_scale
                )
                rec.retention = min(
                    1.0,
                    rec.retention + 0.05 * emotional_intensity * strength_scale,
                )
                # Replay emotional / recent memories into the semantic graph.
                if replayed < self._REPLAY_SAMPLE_SIZE:
                    self._replay_into_semantic_graph(rec.episode_id)
                    replayed += 1
                processed += 1
        # Fear extinction: REM sleep processes conditioned fear
        # associations without the aversive stimulus, weakening
        # them (extinction learning). This is why sleep helps with
        # PTSD and anxiety — the amygdala reprocesses fear memories
        # in a safe context.
        for stimulus in list(self.emotional_memory._memories.keys()):
            memory = self.emotional_memory._memories[stimulus]
            if not memory.is_extinct and memory.association_strength > 0.1:
                self.emotional_memory.extinction_learning(stimulus, exposures=1)
        return processed

    def _consolidate_n1(self, recent_cutoff: int, strength_scale: float = 1.0) -> int:
        """N1: the lightest stage — minor replay."""
        processed = 0
        replayed = 0
        for rec in self._records.values():
            if rec.forgotten or rec.encoded_at < recent_cutoff:
                continue
            rec.synaptic_strength = min(1.0, rec.synaptic_strength + 0.03 * strength_scale)
            # Replay a few recent memories into the semantic graph.
            if replayed < self._REPLAY_SAMPLE_SIZE:
                self._replay_into_semantic_graph(rec.episode_id)
                replayed += 1
            processed += 1
        return processed

    # ═══════════════════════════════════════════════════════════════
    #  Item 2: Forgetting mechanisms
    # ═══════════════════════════════════════════════════════════════

    def forget(self, now: float | None = None) -> int:
        """Apply forgetting mechanisms to all memory records.

        Implements multiple forgetting mechanisms:

        1. **Time-based decay (Ebbinghaus)**: retention = e^(-t/tau),
           where tau is scaled by salience (more salient memories
           decay more slowly). This is the classic forgetting curve.
        2. **Proactive interference**: old memories interfere with
           new ones — memories encoded close together compete, and
           older ones slightly reduce newer ones' retention.
        3. **Retroactive interference**: new memories interfere with
           old ones — newer memories slightly reduce older ones'
           retention.
        4. **Retrieval failure**: a memory exists but can't be
           accessed — modelled as a probabilistic access gate based
           on how long since last access.
        5. **Adaptive forgetting**: low-relevance (low salience +
           low access count) memories are forgotten to make room.

        Memories whose retention drops below the threshold are marked
        forgotten. The method returns the number newly forgotten.

        Args:
            now: Optional current time (seconds). Defaults to
                ``time.time()``.

        Returns:
            Number of memories newly forgotten in this pass.
        """
        if now is None:
            now = time.time()
        now_ms = int(now * 1000)
        newly_forgotten = 0

        # Sort records by encoded_at for interference calculations.
        sorted_records = sorted(
            (r for r in self._records.values() if not r.forgotten),
            key=lambda r: r.encoded_at,
        )

        for idx, rec in enumerate(sorted_records):
            if self._apply_forgetting_to_record(rec, idx, sorted_records, now_ms):
                newly_forgotten += 1

        self.forgotten_count += newly_forgotten
        return newly_forgotten

    def _apply_forgetting_to_record(
        self,
        rec: MemoryRecord,
        idx: int,
        sorted_records: list[MemoryRecord],
        now_ms: int,
    ) -> bool:
        """Apply all forgetting mechanisms to a single record.

        Returns True if the record was newly marked forgotten.

        All penalties are computed from first principles (time and fixed
        properties) each call, then combined into a target retention.
        The stored retention is capped by this target via ``min``, making
        the function idempotent: calling ``forget()`` twice with the same
        timestamp produces the same result. This prevents forgetting from
        depending on how often ``forget()`` is called rather than on
        elapsed time.
        """
        # ── 1. Ebbinghaus decay ───────────────────────────────
        age_s = max(0.0, (now_ms - rec.encoded_at) / 1000.0)
        # Salience scales tau: salient memories decay slower.
        salience_factor = 0.3 + rec.salience  # 0.3..1.3
        tau = self._FORGETTING_TAU * salience_factor
        decayed = math.exp(-age_s / tau)

        # ── 2. Proactive + retroactive interference ────────────
        # Older memories encoded close in time interfere with this
        # one (proactive); newer memories interfere with this older
        # one (retroactive). The total interference is a subtractive
        # penalty computed from the fixed encoding times of nearby
        # memories, so it is the same on every call (idempotent).
        total_interference = 0.0
        for older in sorted_records[:idx]:
            time_gap = (rec.encoded_at - older.encoded_at) / 1000.0
            if 0 < time_gap < 3600:  # within 1 hour
                # Closer in time → more interference.
                total_interference += 0.02 * (1.0 - time_gap / 3600.0)
        for newer in sorted_records[idx + 1 :]:
            time_gap = (newer.encoded_at - rec.encoded_at) / 1000.0
            if 0 < time_gap < 3600:
                total_interference += 0.02 * (1.0 - time_gap / 3600.0)

        # ── 3. Retrieval failure ─────────────────────────────
        # The longer since last access, the more likely retrieval
        # fails (the memory exists but can't be accessed). This is a
        # time-based multiplicative factor, not a per-call penalty.
        since_access_s = max(0.0, (now_ms - rec.last_accessed) / 1000.0)
        # Retrieval probability decays with disuse.
        retrieval_prob = math.exp(-since_access_s / (tau * 2.0))
        retrieval_mult = 0.8 if retrieval_prob < 0.1 else 1.0

        # ── 4. Adaptive forgetting ───────────────────────────
        # Low-relevance memories (low salience + rarely accessed)
        # are forgotten to make room for new memories. This is a
        # property-based multiplicative factor, not a per-call penalty.
        adaptive_mult = 0.5 if (rec.salience < 0.2 and rec.access_count == 0) else 1.0

        # ── Combine: target retention from first principles ───
        # All penalties are applied to the Ebbinghaus decayed value,
        # producing a target that depends only on time and fixed
        # properties. The stored retention is capped by this target,
        # ensuring monotonic decay (retention can only decrease)
        # while making the function idempotent.
        target = max(0.0, decayed - total_interference) * retrieval_mult * adaptive_mult
        rec.retention = min(rec.retention, target)

        # ── Mark forgotten if below threshold ────────────────
        if rec.retention < self.config.forgetting_threshold:
            rec.forgotten = True
            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    #  Item 3: Reconsolidation mechanism
    # ═══════════════════════════════════════════════════════════════

    def reconsolidate(
        self,
        memory_id: int,
        modification: ReconsolidationModification,
    ) -> bool:
        """Reconsolidate a recalled memory with the given modification.

        When a memory is recalled with prediction error, it becomes
        labile (modifiable) for a brief window (~6 hours in humans).
        During this window the memory can be strengthened, weakened,
        or emotionally re-tagged. The current emotional state during
        reconsolidation affects the memory's emotional tag (emotional
        modulation).

        Boundary condition: the reconsolidation window only opens when
        prediction error exceeds the threshold (Schiller et al., 2010;
        Sevenster et al., 2014). Recall without prediction error
        strengthens the memory via retrieval practice but does not make
        it labile. If the window was never opened (or has closed), the
        modification is not applied and the method returns ``False``.

        Args:
            memory_id: The episode ID of the memory to reconsolidate.
            modification: The modification to apply (strength delta
                and/or emotional re-tagging).

        Returns:
            ``True`` if the modification was applied, ``False`` if the
            memory was not found, already forgotten, or no longer in
            the reconsolidation window.
        """
        rec = self._records.get(memory_id)
        if rec is None or rec.forgotten:
            return False
        if rec.reconsolidation_window <= 0:
            # Window has closed — memory is no longer labile.
            return False

        # Apply strength modification (strengthen or weaken).
        rec.salience = max(0.0, min(1.0, rec.salience + modification.strength_delta))
        rec.retention = max(0.0, min(1.0, rec.retention + modification.strength_delta))
        rec.synaptic_strength = max(
            0.0,
            min(1.0, rec.synaptic_strength + modification.strength_delta),
        )

        # Emotional modulation: blend the current emotional state
        # into the memory's emotional tag.
        if modification.emotional_tag is not None:
            blend = max(0.0, min(1.0, modification.blend_factor))
            for i in range(min(len(rec.emotional_tag), len(modification.emotional_tag))):
                rec.emotional_tag[i] = max(
                    0.0,
                    min(
                        1.0,
                        rec.emotional_tag[i] * (1.0 - blend)
                        + modification.emotional_tag[i] * blend,
                    ),
                )

        # Close the reconsolidation window (memory is re-stored).
        rec.reconsolidation_window = 0.0
        rec.last_reconsolidated = int(time.time() * 1000)
        rec.reconsolidation_count += 1
        self.reconsolidation_count += 1
        return True

    def tick_reconsolidation(self, dt: float) -> None:
        """Advance reconsolidation windows by ``dt`` seconds.

        Should be called periodically to close reconsolidation windows
        as time passes.

        Args:
            dt: Elapsed time in seconds.
        """
        for rec in self._records.values():
            if rec.reconsolidation_window > 0:
                rec.reconsolidation_window = max(0.0, rec.reconsolidation_window - dt)

    # ═══════════════════════════════════════════════════════════════
    #  Source monitoring (Johnson et al., 1993)
    # ═══════════════════════════════════════════════════════════════

    def monitor_source(self, memory_id: int, actual_source: str) -> bool:
        """Check whether a memory's source tag matches the actual source.

        Source monitoring errors — mistaking the origin of a memory —
        are the basis of false memories and confabulation (Johnson et
        al., 1993). This method compares the stored source attribution
        against the ground truth, weighted by the source tag's
        confidence. Low-confidence source tags are more likely to
        produce monitoring errors.

        Args:
            memory_id: The episode ID to check.
            actual_source: The true source of the memory.

        Returns:
            ``True`` if the source attribution is correct, ``False``
            if the memory was not found, has no source tag, or the
            attribution is wrong.
        """
        rec = self._records.get(memory_id)
        if rec is None or rec.source_tag is None:
            return False
        return rec.source_tag.source == actual_source

    def check_confabulation_risk(self, memory_id: int) -> float:
        """Assess the risk that a memory is confabulated.

        Confabulation risk is high when:
        - The source is "inference" or "imagination" (internally
          generated rather than observed)
        - Source confidence is low (uncertain origin)
        - The memory has been reconsolidated many times (each
          reconsolidation can drift the content)

        Returns a risk score [0..1] where 0 = no risk (the memory is
        clearly from an external source with high confidence) and
        1 = high risk (the memory is internally generated with low
        confidence and has been reconsolidated repeatedly).
        """
        rec = self._records.get(memory_id)
        if rec is None or rec.source_tag is None:
            return 0.5  # Unknown — moderate risk

        risk = 0.0
        # Internally-generated sources carry inherent confabulation risk
        if rec.source_tag.source in ("inference", "imagination"):
            risk += 0.4
        elif rec.source_tag.source == "genesis":
            risk += 0.2
        # Low source confidence increases risk
        risk += (1.0 - rec.source_tag.confidence) * 0.3
        # Repeated reconsolidation can drift content
        if rec.reconsolidation_count > 5:
            risk += 0.1 * min(1.0, rec.reconsolidation_count / 10.0)
        return min(1.0, risk)

    def get_memories_by_source(self, source: str) -> list[int]:
        """Return episode IDs for all non-forgotten memories from a
        given source.

        Useful for introspection ("what did I learn from docs vs
        conversation vs inference?") and for source-based retrieval
        filtering.
        """
        return [
            rec.episode_id
            for rec in self._records.values()
            if not rec.forgotten
            and rec.source_tag is not None
            and rec.source_tag.source == source
        ]

    def source_summary(self) -> dict[str, int]:
        """Return a count of memories per source category.

        Useful for introspection and metacognitive awareness of where
        knowledge came from.
        """
        counts: dict[str, int] = {}
        for rec in self._records.values():
            if rec.forgotten:
                continue
            src = rec.source_tag.source if rec.source_tag else "unknown"
            counts[src] = counts.get(src, 0) + 1
        return counts

    # ═══════════════════════════════════════════════════════════════
    #  Items 4 & 5: Systems consolidation & synaptic consolidation
    # ═══════════════════════════════════════════════════════════════

    def synaptic_consolidate(self, memory_id: int) -> bool:
        """Apply fast synaptic consolidation to a recently-encoded memory.

        Synaptic consolidation is the fast (minutes-hours) LTP-like
        strengthening that stabilises a memory trace immediately after
        encoding. This is distinct from systems consolidation (the
        slow hippocampal→neocortical redistribution over days-weeks).

        This boosts the memory's synaptic strength and retention,
        modelling the protein-synthesis-dependent late phase of LTP
        (Kandel, 2001).

        Args:
            memory_id: The episode ID to consolidate.

        Returns:
            ``True`` if applied, ``False`` if the memory was not found.
        """
        rec = self._records.get(memory_id)
        if rec is None or rec.forgotten:
            return False
        rec.synaptic_strength = min(1.0, rec.synaptic_strength + self._SYNAPTIC_BOOST)
        rec.retention = min(1.0, rec.retention + 0.1)
        self.synaptic_consolidation_count += 1
        return True

    def systems_consolidate(self) -> int:
        """Redistribute memories from hippocampal to neocortical storage.

        Systems consolidation is the slow (days-weeks) process by
        which hippocampal-dependent memories gradually become
        independent of the hippocampus and stored in neocortex. This
        captures the temporal gradient of retrograde amnesia: recent
        memories (high hippocampal dependency) are impaired by
        hippocampal damage, but remote memories (low dependency) are
        spared (Squire & Alvarez, 1995; Frankland & Bontempi, 2005).

        Each call reduces the hippocampal dependency of all
        non-forgotten memories by a fixed fraction and correspondingly
        increases their neocortical (systems) strength. This is
        normally called during SWS (see :meth:`consolidate_during_sleep`)
        but can also be called independently to model time-based
        consolidation.

        Returns:
            Number of memories processed.
        """
        processed = 0
        for rec in self._records.values():
            if rec.forgotten or rec.hippocampal_dependency <= 0.0:
                continue
            rec.hippocampal_dependency = max(
                0.0,
                rec.hippocampal_dependency - self._HIPPOCAMPAL_DECAY,
            )
            rec.systems_strength = min(1.0, rec.systems_strength + self._HIPPOCAMPAL_DECAY)
            # As memories become neocortical, retention stabilises
            # (remote memories are more resistant to forgetting).
            rec.retention = min(1.0, rec.retention + 0.02)
            processed += 1
        self.systems_consolidation_count += processed
        return processed

    # ── Introspection helpers ────────────────────────────────────

    def get_memory_record(self, episode_id: int) -> MemoryRecord | None:
        """Return the Python-side metadata record for an episode."""
        return self._records.get(episode_id)

    @property
    def tracked_memory_count(self) -> int:
        """Number of memories with Python-side consolidation metadata."""
        return len(self._records)

    @property
    def active_memory_count(self) -> int:
        """Number of non-forgotten tracked memories."""
        return sum(1 for r in self._records.values() if not r.forgotten)

    # ── Persistence ──────────────────────────────────────────────

    def serialize_records(self) -> dict[str, Any]:
        """Serialize Python-side memory metadata for persistence.

        The Rust daemon persists the raw LTM episodes (text, emotional
        tags, salience). This serializes the Python-side consolidation
        state that the Rust store doesn't hold: hippocampal dependency,
        synaptic/systems strength, retention, forgetting state,
        reconsolidation windows, source tags, and access counts.

        Without this, every restart resets all biological memory
        processes — forgetting, reconsolidation, sleep consolidation,
        and source monitoring start from scratch.
        """
        return {
            "records": [
                {
                    "episode_id": rec.episode_id,
                    "salience": rec.salience,
                    "emotional_tag": list(rec.emotional_tag),
                    "encoded_at": rec.encoded_at,
                    "last_accessed": rec.last_accessed,
                    "access_count": rec.access_count,
                    "hippocampal_dependency": rec.hippocampal_dependency,
                    "synaptic_strength": rec.synaptic_strength,
                    "systems_strength": rec.systems_strength,
                    "retention": rec.retention,
                    "forgotten": rec.forgotten,
                    "reconsolidation_window": rec.reconsolidation_window,
                    "last_reconsolidated": rec.last_reconsolidated,
                    "reconsolidation_count": rec.reconsolidation_count,
                    "source": rec.source_tag.source if rec.source_tag else None,
                    "source_modality": rec.source_tag.modality if rec.source_tag else "text",
                    "source_confidence": rec.source_tag.confidence if rec.source_tag else 0.8,
                }
                for rec in self._records.values()
            ],
            "forgotten_count": self.forgotten_count,
            "reconsolidation_count": self.reconsolidation_count,
            "synaptic_consolidation_count": self.synaptic_consolidation_count,
            "systems_consolidation_count": self.systems_consolidation_count,
            "sleep_consolidation_count": self.sleep_consolidation_count,
            "facts_replayed_count": self.facts_replayed_count,
        }

    def restore_records(self, data: dict[str, Any]) -> None:
        """Restore Python-side memory metadata from persisted state.

        Rebuilds the ``_records`` dict and consolidation counters from
        a previously serialized state (see :meth:`serialize_records`).
        Called on startup after the LTM store is available.
        """
        self._records.clear()
        for entry in data.get("records", []):
            try:
                eid = int(entry["episode_id"])
            except (KeyError, TypeError, ValueError):
                continue
            source = entry.get("source")
            source_tag = None
            if source is not None:
                source_tag = SourceTag(
                    source=source if source in VALID_SOURCES else "unknown",
                    modality=entry.get("source_modality", "text"),
                    confidence=max(0.0, min(1.0, float(entry.get("source_confidence", 0.8)))),
                )
            rec = MemoryRecord(
                episode_id=eid,
                salience=float(entry.get("salience", 0.5)),
                emotional_tag=list(entry.get("emotional_tag", [0.5] * 12)),
                encoded_at=int(entry.get("encoded_at", 0)),
                last_accessed=int(entry.get("last_accessed", 0)),
                access_count=int(entry.get("access_count", 0)),
                hippocampal_dependency=float(entry.get("hippocampal_dependency", 1.0)),
                synaptic_strength=float(entry.get("synaptic_strength", 0.3)),
                systems_strength=float(entry.get("systems_strength", 0.0)),
                retention=float(entry.get("retention", 1.0)),
                forgotten=bool(entry.get("forgotten", False)),
                reconsolidation_window=float(entry.get("reconsolidation_window", 0.0)),
                last_reconsolidated=int(entry.get("last_reconsolidated", 0)),
                reconsolidation_count=int(entry.get("reconsolidation_count", 0)),
                source_tag=source_tag,
            )
            self._records[eid] = rec
        self.forgotten_count = int(data.get("forgotten_count", 0))
        self.reconsolidation_count = int(data.get("reconsolidation_count", 0))
        self.synaptic_consolidation_count = int(
            data.get("synaptic_consolidation_count", 0)
        )
        self.systems_consolidation_count = int(
            data.get("systems_consolidation_count", 0)
        )
        self.sleep_consolidation_count = int(
            data.get("sleep_consolidation_count", 0)
        )
        self.facts_replayed_count = int(data.get("facts_replayed_count", 0))


def _emotional_intensity(tag: list[float]) -> float:
    """Compute emotional intensity from a 12-chemical tag.

    High variance and high arousal chemicals (cortisol, noradrenaline,
    dopamine) indicate an emotionally intense memory.
    """
    if not tag:
        return 0.0
    mean = sum(tag) / len(tag)
    variance = sum((x - mean) ** 2 for x in tag) / len(tag)
    # Arousal chemicals: DA (idx 0), NE (idx 2), cortisol (idx 6).
    arousal_chems = [tag[i] for i in (0, 2, 6) if i < len(tag)]
    arousal = sum(arousal_chems) / max(1, len(arousal_chems))
    return min(1.0, 0.5 * arousal + 0.5 * min(1.0, variance * 4.0))


def _finite_clamp(value: float, lo: float, hi: float, *, default: float = 0.5) -> float:
    """Clamp a float to [lo, hi], replacing NaN/Inf with a default.

    Mirrors the Rust daemon's ``finite_clamp``: non-finite values
    (NaN, +Inf, -Inf) are replaced by ``default`` before clamping to
    the range. This prevents NaN from propagating through the
    forgetting and consolidation formulas.
    """
    if math.isnan(value) or math.isinf(value):
        return max(lo, min(hi, default))
    return max(lo, min(hi, value))


def _emotional_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two emotional vectors (mood congruence)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _text_to_vector(text: str, size: int = 64) -> list[float]:
    """Deterministically map text to a fixed-size feature vector.

    Used to feed memories/queries into the attractor network for
    pattern completion. Each character hashes to a bucket; the bucket
    is set to 1.0 (bipolar +1 after the network's internal
    conversion). This gives a stable, content-addressable fingerprint
    so similar texts produce similar (overlapping) vectors.
    """
    vector = [0.0] * size
    if not text:
        return vector
    for ch in text.lower():
        h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
        vector[h % size] = 1.0
    return vector


def _extract_concepts(text: str) -> list[str]:
    """Extract candidate concepts from a text string.

    A lightweight, dependency-free tokeniser: splits on
    whitespace/punctuation and keeps alphabetic tokens of length ≥ 3,
    deduplicated while preserving order. These concepts are fed to
    priming and spreading activation.
    """
    if not text:
        return []
    seen: set[str] = set()
    concepts: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                token = "".join(current)
                current = []
                _maybe_add(token, seen, concepts)
    if current:
        _maybe_add("".join(current), seen, concepts)
    return concepts


def _maybe_add(token: str, seen: set[str], concepts: list[str]) -> None:
    """Add a token to the concept list if it qualifies."""
    if len(token) < 3:
        return
    if not token.isalpha():
        return
    key = token.lower()
    if key in seen:
        return
    seen.add(key)
    concepts.append(key)


def _primary_concept(text: str) -> str:
    """Return the primary concept (first qualifying token) from text.

    Used as the stimulus key for emotional-memory tagging.
    """
    concepts = _extract_concepts(text)
    return concepts[0] if concepts else ""

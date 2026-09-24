"""Mind sleep — sleep/wake/meditation and consolidation."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import (
    CHEM_ADENOSINE,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_GABA,
    CHEM_HISTAMINE,
    CHEM_MELATONIN,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    CHEM_SEROTONIN,
    PHASE_NAMES,
    ZONE_CONVERSATION,
    ZONE_REFLECTION,
    ZONE_SLEEPING,
)

from ..commitment import CommitmentBoundary
from ..sleep import SleepStage
from .thresholds import (
    AUTO_SLEEP_ADENOSINE,
    AUTO_SLEEP_CONFIRM_S,
    AUTO_SLEEP_EXIT,
    AUTO_SLEEP_MIN_AWAKE,
    AUTO_SLEEP_MIN_IDLE,
    AUTO_WAKE_ADENOSINE,
    AUTO_WAKE_CYCLE_ADENOSINE,
    DROWSINESS_ADENOSINE,
    DROWSINESS_CONFIRM_S,
    DROWSINESS_EXIT,
    WAKE_REINFORCE_INTERVAL,
    WAKE_RESCUE_ADENOSINE,
    WAKE_RESCUE_MELATONIN,
    WAKE_STABILIZATION_WINDOW,
)

logger = logging.getLogger(__name__)


class SleepMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def _init_sleep_compression(self) -> None:
        """Initialize the sleep compression system (VQ + holographic graph).

        This is the bounded-growth architecture that keeps Genesis's
        memory from expanding without limit. It runs during N3 sleep
        to compress the concept embedding space (VQ), migrate
        auto-generated bridge edges to a fixed-size holographic
        associative memory, and compact LTM episodes.

        The holographic graph is attached to the concept network so
        that get_neighbors and spread_activation can query it — this
        keeps compressed associations accessible after migration
        rather than making the holographic graph a write-only graveyard.
        """
        from ..sleep import SleepCompressor

        try:
            self._sleep_compressor: SleepCompressor | None = SleepCompressor(
                data_dir=self.data_dir,
            )
            # Attach the holographic graph to the network so compressed
            # associations remain queryable after sleep compression
            # migrates bridge edges and LTM associations into it.
            if self._sleep_compressor is not None:
                self.cognition.network.attach_holographic_graph(
                    self._sleep_compressor.holographic_graph
                )
                # Wire the holographic graph to the embedding store so
                # the experiential layer can be built from sleep-
                # discovered associations. This unifies the holographic
                # graph's T^2048 with the embedding store's latent
                # space into a single toroidal representation.
                if hasattr(self.cognition, "embeddings") and self.cognition.embeddings is not None:
                    self.cognition.embeddings.set_holographic_graph(
                        self._sleep_compressor.holographic_graph
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning("Sleep compression init failed: %s", e)
            self._sleep_compressor = None
    def _init_sleep_state_tracking(self) -> None:
        """Initialize sleep, rest, meditation, and teaching state tracking.

        Sleep-state tracking — lets the CLI and the cognitive logic
        avoid duplicate sleep()/wake() calls and lets ambient/CLI
        interactions wake her if she has drifted off.

        Whether sleep was initiated by the user (via /sleep or /nap)
        or by her own neurochemistry (sleep watcher). When user-
        initiated, the sleep watcher should NOT auto-wake her —
        only /wake or the nap timer should wake her. When self-
        initiated, she wakes naturally when her neurochemistry
        recovers.

        Track when she last woke up, for the auto-sleep minimum-awake
        guard. This prevents auto-sleep from firing immediately after
        wake if adenosine is still elevated.

        Meditation-state tracking — a lighter recovery mode than
        sleep. During meditation, learning, code scanning, bug
        scanning, and spontaneous thoughts are all paused. The
        neurochemistry is actively calmed (GABA/serotonin up,
        dopamine/NE/cortisol down) to allow receptor recovery
        without the full sleep cycle. Unlike sleep, there is no
        dreaming or memory consolidation — just quiet restoration.

        Last rest time — tracks when she last meditated or slept,
        for the ultradian rest cycle (BRAC, Kleitman 1963). The
        meditation urge builds from sustained_activity, which is
        time since last rest normalized over 75 minutes.

        Concept network size at last volition tick — used to measure
        concept_growth (new concepts since last tick) for the
        introspection and self-mission urges.

        Teaching-mode tracking — when the user is actively teaching
        her something, she enters a focused learning state. Bug
        scanning, autonomous learning, art, and unrelated curiosity
        questions are paused. Self-directed learning from the
        conversation stays active so she remembers what she's taught.
        """
        self._is_sleeping = False
        self._user_initiated_sleep = False
        self._last_wake_time: float = time.time()
        self._last_wake_reinforce: float = 0.0
        self._sleep_start_cycles: int = 0
        self._is_meditating = False
        self._last_rest_time = time.time()
        self._last_concept_count = 0
        self._is_teaching = False
        self._teaching_topic: str = ""
        # Commitment boundaries on the sleep-wake axis — the
        # digitizers that turn continuous adenosine pressure into
        # discrete transitions. Both require a sustained crossing
        # (confirm_s) so a transient spike cannot commit her, and
        # re-arm below a lower release level (deadband) so boundary
        # flicker cannot re-fire the transition. Reset on wake.
        self._drowsy_boundary = CommitmentBoundary(
            enter=DROWSINESS_ADENOSINE,
            release=DROWSINESS_EXIT,
            confirm_s=DROWSINESS_CONFIRM_S,
        )
        self._sleep_boundary = CommitmentBoundary(
            enter=AUTO_SLEEP_ADENOSINE,
            release=AUTO_SLEEP_EXIT,
            confirm_s=AUTO_SLEEP_CONFIRM_S,
        )
    @property
    def is_sleeping(self) -> bool:
        """Whether she is currently in a sleep state."""
        return self._is_sleeping
    @property
    def is_meditating(self) -> bool:
        """Whether she is currently meditating."""
        return self._is_meditating
    def wake_if_asleep(self) -> bool:
        """Convenience hook to rouse her before an interaction.

        Returns True if she was asleep or meditating and has been woken.
        """
        if self._is_meditating:
            try:
                self.wake_from_meditation()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"wake_from_meditation failed: {e}")
            return not self._is_meditating
        if not self._is_sleeping:
            return False
        try:
            self.wake()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"wake failed: {e}")
        return not self._is_sleeping
    def _on_sleep_stage_transition(self, stage: SleepStage, dt: float) -> None:
        """Handle a sleep-stage transition during the ultradian cycle.

        Called by InnerLife when the sleep cycle advances to a new
        stage (N1→N2→N3→N2→REM). Each stage has a distinct
        consolidation role (Diekelmann & Born, 2010):

        - **N1**: light sleep — minimal consolidation.
        - **N2**: sleep spindles drive hippocampal→neocortical
          transfer — the bulk of memory transfer.
        - **N3**: slow-wave sleep — systems consolidation, replay
          recent memories, synaptic downscaling, edge discovery,
          Hebbian plasticity, embeddings refresh.
        - **REM**: emotional and procedural memory consolidation.

        This is the progressive, stage-appropriate consolidation that
        replaces the old single hard-coded N3 pass at sleep onset.
        """
        try:
            # Always emit a visible stage-transition notification so the
            # user can see the ultradian cycle progressing. Without this,
            # stage transitions are invisible unless episodes happen to
            # be consolidated, and the user can't tell when she enters
            # N3 (the deep-sleep stage that does heavy consolidation).
            cycle_num = (
                self.inner_life.sleep_cycle.cycle_number
                if self.inner_life.sleep_cycle
                else 0
            )
            self._emit_live_thought(
                "sleep",
                f"Sleep cycle {cycle_num}: entering {stage.value.upper()}",
            )

            # Compute the brain-wave-derived consolidation intensity.
            # During sleep, the daemon's neurochemistry produces theta/
            # delta-dominant waves with high consolidation weight.
            # We read the actual brain wave state so the consolidation
            # strength scales with the real neurochemical context
            # (e.g. stress suppresses consolidation even during sleep).
            try:
                brain_waves = self.brain_waves()
                consolidation_intensity = brain_waves.consolidation
            except (AttributeError, RuntimeError, ConnectionError):
                logger.debug(
                    "brain wave read failed during sleep, using default intensity",
                    exc_info=True,
                )
                consolidation_intensity = 0.5

            # Stage-specific episodic memory consolidation, scaled by
            # the brain-wave-derived consolidation intensity.
            n = self.cognition.memory.consolidate_during_sleep(
                stage,
                duration=300.0,
                consolidation_intensity=consolidation_intensity,
            )
            if n > 0:
                self._emit_live_thought(
                    "learning",
                    f"{stage.value.upper()} consolidation: {n} episodes processed",
                )

            # N3 is when the heavy systems consolidation happens:
            # concept-network consolidation, edge discovery, Hebbian
            # plasticity, synaptic downscaling, embeddings refresh.
            # These are expensive and belong in the deepest sleep
            # stage, not at sleep onset.
            if stage is SleepStage.N3:
                self._consolidate_n3_heavy(consolidation_intensity)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"sleep stage consolidation failed: {e}")
    def _run_sleep_compression(self, network, consolidation_intensity: float) -> None:
        """Algorithmic sleep compression: VQ, holographic graph, LTM compaction.

        This is the bounded-growth pass that keeps Genesis's memory from
        expanding without limit. It runs after the existing consolidation
        so bridge edges created during this sleep cycle are also migrated.
        """
        if self._sleep_compressor is not None:
            try:
                comp_stats = self._sleep_compressor.compress(
                    network=network,
                    embeddings=self.cognition.embeddings,
                    ltm_client=self.client,
                    consolidation_intensity=consolidation_intensity,
                )
                self._emit_live_thought(
                    "sleeping",
                    f"Sleep compression: {comp_stats}",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Sleep compression failed: %s", e)
                self._emit_live_thought(
                    "sleeping",
                    f"Sleep compression failed: {e}",
                )
        else:
            self._emit_live_thought(
                "sleeping",
                "Sleep compression skipped: compressor not initialized",
            )
    def _run_journal_consolidation(self) -> None:
        """Journal consolidation during N3 sleep.

        The journal grows monotonically during wakefulness. During
        N3 sleep, compact old entries the same way LTM episodes are
        compacted: recent entries are kept verbatim, older entries
        are summarized by tag into consolidation entries. This keeps
        the journal bounded for indefinite operation while preserving
        her voice and developmental arc.
        """
        try:
            journal_stats = self.journal.consolidate()
            if journal_stats["consolidated"] > 0:
                self._emit_live_thought(
                    "learning",
                    f"Journal consolidation: {journal_stats}",
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Journal consolidation failed: %s", e)
    def _consolidate_n3_heavy(self, consolidation_intensity: float = 0.5) -> None:
        """N3 slow-wave sleep: heavy systems consolidation.

        This is the deep-sleep work: concept-network consolidation,
        edge discovery from latent space, Hebbian plasticity,
        synaptic downscaling (SHY), and embeddings refresh. These
        are expensive operations that belong in N3, not at sleep
        onset.
        """
        # Categorize and classify any concepts that are still UNKNOWN.
        # This catches concepts loaded from older persistence formats
        # (before creation-time detection was added) and concepts that
        # received definitions after creation. Category and modality
        # drive spreading-activation multipliers, modality-specific
        # decay, and within-modality priming — so uncategorized concepts
        # miss out on these dynamics until they're classified.
        network = self.cognition.network
        cat_counts = network.categorize_all()
        mod_counts = network.classify_modalities()
        if cat_counts:
            self._emit_live_thought(
                "learning",
                f"Sleep categorization: {cat_counts}",
            )
        if mod_counts:
            self._emit_live_thought(
                "learning",
                f"Sleep modality classification: {mod_counts}",
            )

        # Consolidate the concept network during slow-wave sleep
        result = network.consolidate_during_sleep()
        if any(result.values()):
            self._emit_live_thought("sleeping", f"Sleep consolidation: {result}")

        if not self.cognition.embeddings.has_embeddings:
            return

        # Rebuild the embedding matrix if the network has grown, so
        # newly learned concepts participate in discovery.
        if self.cognition.embeddings.concept_count < self.cognition.network.size:
            self.cognition.embeddings.refresh()

        # Discover new relationships from embedding proximity.
        # This is hippocampal consolidation: patterns → explicit knowledge.
        discovered = self.cognition.edge_proposer.discover_and_accept()
        if discovered > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep discovery: {discovered} new edges from latent space",
            )

        # Cross-domain analogy — structure-mapping across cortical
        # columns. Finds concepts in different domains whose relational
        # structure matches and projects relations from one domain
        # into the other. This is where cross-domain insights form —
        # connections that pair-level similarity cannot find.
        analogies = self.cognition.analogy_engine.discover_and_accept()
        if analogies > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep analogy: {analogies} cross-domain edges from structure-mapping",
            )

        # Hebbian plasticity — consolidate the day's co-occurrences
        # into the embedding space. Concepts that were used together
        # move closer; contradicting concepts move apart. This is
        # slow-wave sleep consolidation.
        plasticity_result = self.cognition.plasticity.apply_sleep_consolidation()
        if plasticity_result["total"] > 0:
            self._emit_live_thought("sleeping", f"Sleep plasticity: {plasticity_result}")

        # Edge-weight Hebbian plasticity — strengthen edges from
        # the day's co-occurrences, apply homeostatic decay to
        # all non-protected edges (synaptic downscaling, SHY).
        edge_result = self.cognition.plasticity.apply_edge_plasticity(
            learning_rate=0.02,  # stronger during sleep
            decay_rate=0.005,   # more decay during sleep (renormalization)
        )
        if edge_result["strengthened"] > 0 or edge_result["decayed"] > 0:
            self._emit_live_thought(
                "sleeping",
                f"Edge plasticity: +{edge_result['strengthened']} strengthened, "
                f"-{edge_result['decayed']} decayed",
            )

        # Refresh embeddings — rebuilds the spectral component from
        # the updated graph while preserving Hebbian-adapted
        # experiential vectors (TF-IDF + GloVe).
        self.cognition.embeddings.refresh()

        # Persist the Hebbian-adapted experiential vectors so they
        # survive restarts. Without this, her experiential learning
        # is lost on every shutdown.
        self.cognition.embeddings.save_experiential(self.data_dir)

        # Refresh topology metrics after the graph changed
        self.cognition.topology.refresh()

        self._run_sleep_compression(network, consolidation_intensity)
        self._run_journal_consolidation()
    def _check_auto_sleep(self) -> None:
        """Check whether Genesis should fall asleep or wake up on her own.

        This implements the biological sleep-wake cycle: adenosine
        accumulates during wakefulness and eventually forces sleep, then
        glymphatic clearance during sleep drains adenosine, allowing
        natural wake-up. The cognitive mind syncs with the subcognitive
        daemon — when adenosine crosses the sleep threshold, the mind
        calls ``sleep()`` to set the zone, pause the learner, and start
        the ultradian cycle. When adenosine drops below the wake
        threshold during sleep, the mind calls ``wake()``.

        The daemon's emergent phase (NREM/REM) is also checked: the
        daemon enters sleep based on a multi-system decision (adenosine
        + histamine + melatonin + orexin), so when melatonin is high
        (circadian night) the daemon's sleep threshold drops below the
        mind's adenosine-only threshold. Without checking the daemon's
        phase, the mind stays "awake" (volition active, learner running,
        autonomous fixes applied) while the daemon is already in NREM —
        a desync that lets waking actions leak into sleep.

        User-initiated sleep (``/sleep``, ``/nap``) is respected — she
        will NOT auto-wake from user-initiated sleep, only from
        self-initiated sleep.
        """
        try:
            core_state = self.client.get_state()
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"auto-sleep check failed: {e}")
            return

        adenosine = core_state.chemicals.get("adenosine", 0.0)
        daemon_phase = core_state.phase_name

        # ── Auto-sleep onset ──
        if not self._is_sleeping and not self._is_meditating:
            now = time.time()
            # Feed the commitment boundaries before the gates — the
            # sustained-crossing window must track the physical
            # signal, not the gate-opened signal. Pressure that held
            # above threshold while the user was active is genuinely
            # sustained, not a transient, so the confirm clock keeps
            # running behind the min-awake and idle gates.
            self._drowsy_boundary.update(adenosine, now)
            self._sleep_boundary.update(adenosine, now)
            # Minimum awake time to prevent oscillation at startup.
            awake_elapsed = now - self._last_wake_time
            if awake_elapsed < AUTO_SLEEP_MIN_AWAKE:
                return
            # Don't fall asleep while the user is actively talking to
            # her. If the last interaction was recent (within 60s), the
            # user is engaged in conversation — falling asleep mid-
            # conversation is a jarring UX failure. The daemon may
            # enter NREM on its own (adenosine-driven), but the
            # cognitive mind stays awake until the user goes idle.
            idle = now - self._last_interaction_time
            if idle < AUTO_SLEEP_MIN_IDLE:
                logger.debug(
                    f"auto-sleep deferred: user active {idle:.1f}s ago "
                    f"(adenosine={adenosine:.2f}, phase={daemon_phase})"
                )
                return
            # Drowsiness announcement — before falling asleep, she
            # announces she's getting sleepy. This is composed from
            # her understanding of "sleepiness" (not a hardcoded
            # string). The boundary's commit edge fires once per
            # sustained crossing; it re-arms only if pressure drops
            # below DROWSINESS_EXIT, so she can announce again if she
            # genuinely recovers and fades a second time.
            if self._drowsy_boundary.just_committed:
                self._announce_drowsiness()
            # The daemon may enter NREM/REM before adenosine reaches
            # the mind's threshold — melatonin lowers the daemon's
            # sleep barrier (circadian night). Sync with the daemon's
            # phase to close the desync window where volition, the
            # learner, and autonomous fixes are still active while the
            # subcognitive is already sleeping.
            daemon_sleeping = daemon_phase in ("nrem", "rem")
            # Wake stabilization: within the post-wake window, a
            # daemon-side sleep phase at moderate pressure is the
            # flip-flop not yet latched — reinforce the wake state
            # instead of instantly re-sleeping. Genuine exhaustion or
            # circadian drive still override.
            melatonin = core_state.chemicals.get("melatonin", 0.0)
            if (
                daemon_sleeping
                and awake_elapsed < WAKE_STABILIZATION_WINDOW
                and adenosine < WAKE_RESCUE_ADENOSINE
                and melatonin < WAKE_RESCUE_MELATONIN
            ):
                if now - self._last_wake_reinforce >= WAKE_REINFORCE_INTERVAL:
                    self._last_wake_reinforce = now
                    self._reinforce_wake()
                return
            if self._sleep_boundary.committed or daemon_sleeping:
                reason = (
                    f"adenosine={adenosine:.2f} "
                    f"(threshold={AUTO_SLEEP_ADENOSINE}, "
                    f"sustained {AUTO_SLEEP_CONFIRM_S:.0f}s)"
                    if not daemon_sleeping
                    else f"daemon phase={daemon_phase} "
                    f"(adenosine={adenosine:.2f})"
                )
                logger.info(f"Auto-sleep: {reason}, falling asleep")
                try:
                    self.sleep(user_initiated=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"auto-sleep onset failed: {e}")
            return

        # ── Auto-wake (only for self-initiated sleep) ──
        if self._is_sleeping and not self._user_initiated_sleep:
            # Nap completion — when the nap cycle (N1→N2) is done,
            # wake her regardless of adenosine level. A nap is light
            # sleep; she doesn't need full adenosine clearance, just
            # the N2 spindle-driven light consolidation.
            if self._nap_mode:
                cycle = self.inner_life.sleep_cycle
                if cycle is not None and cycle._nap_complete:
                    logger.info("Auto-wake: nap cycle complete (N1→N2), waking up")
                    try:
                        self.wake()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"auto-wake failed: {e}")
                    return
            if adenosine <= AUTO_WAKE_ADENOSINE:
                logger.info(
                    f"Auto-wake: adenosine={adenosine:.2f} "
                    f"(threshold={AUTO_WAKE_ADENOSINE}), waking up"
                )
                try:
                    self.wake()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"auto-wake failed: {e}")
                return
            # Cycle-boundary wake: spontaneous waking clusters at
            # ultradian transitions (post-REM). When a full cycle
            # completes, a relaxed pressure threshold applies —
            # without it, slow clearance can hold her in sleep long
            # past the point a full cycle has already restored her.
            cycle = self.inner_life.sleep_cycle
            if (
                cycle is not None
                and cycle.cycles_completed > self._sleep_start_cycles
                and adenosine <= AUTO_WAKE_CYCLE_ADENOSINE
            ):
                logger.info(
                    f"Auto-wake: cycle boundary reached "
                    f"(adenosine={adenosine:.2f}, "
                    f"threshold={AUTO_WAKE_CYCLE_ADENOSINE}), waking up"
                )
                try:
                    self.wake()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"auto-wake failed: {e}")
    def _announce_drowsiness(self) -> None:
        """Announce that she's getting sleepy, before falling asleep.

        This is composed from her understanding of "sleepiness" via
        the thought composer — NOT a hardcoded string. She speaks
        her own words about her own state. If she can't articulate
        it yet (doesn't know the concept well enough), she stays
        silent and just falls asleep — the drowsiness is still
        tracked internally.
        """
        try:
            emotion = self.feel()
            thought = self.cognition.composer.compose_about(
                "sleepiness", emotion, focused=True
            )
            if thought and thought.content and thought.confidence > 0.3:
                self._emit_live_thought("thought", thought.content)
                if self._on_speak is not None:
                    try:
                        self._on_speak(thought.content)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"drowsiness speak failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"drowsiness announcement failed: {e}")
    def sleep(self, user_initiated: bool = False, nap: bool = False) -> None:
        """Put Genesis into sleep mode for dreaming.

        This sets the subcognitive zone to SLEEPING (which triggers
        the Rust daemon's dreaming pass on LTM). The inner-life loop
        is NOT paused — it switches to dream generation and advances
        the ultradian sleep cycle (N1→N2→N3→N2→REM). Stage-appropriate
        memory consolidation happens via :meth:`_on_sleep_stage_transition`
        as the cycle progresses, rather than a single hard-coded pass
        at sleep onset.

        The heavy systems consolidation (concept-network consolidation,
        edge discovery, Hebbian plasticity, synaptic downscaling,
        embeddings refresh) happens when the cycle reaches N3 — the
        deepest, most restorative stage — not at sleep onset.

        Args:
            user_initiated: If True, the sleep watcher will not
                auto-wake her — only /wake will. This is for when the
                user puts her to bed.
            nap: If True, the sleep cycle is limited to light sleep
                (N1→N2) only — no N3 deep consolidation, no REM. She
                wakes after the N2 period ends via the sleep watcher.
                This models a short rest with light consolidation
                (N2 sleep spindles) without the heavy systems
                consolidation of full sleep.
        """
        if self._is_sleeping:
            return
        self._is_sleeping = True
        self._user_initiated_sleep = user_initiated
        self._nap_mode = nap
        cycle = self.inner_life.sleep_cycle
        self._sleep_start_cycles = (
            cycle.cycles_completed if cycle is not None else 0
        )
        self.client.set_zone(ZONE_SLEEPING)

        self._engage_sleep_mechanism()

        # Initial light consolidation at sleep onset (N1 — the
        # lightest stage). The deeper N3 consolidation happens
        # later when the cycle reaches slow-wave sleep. This is
        # sleep-onset only — it is NOT called on restore (the
        # consolidation already happened before the restart).
        n_memories = self.cognition.memory.consolidate_during_sleep(
            SleepStage.N1,
            duration=300.0,
        )
        if n_memories > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep onset: {n_memories} episodes lightly consolidated (N1)",
            )
    def _engage_sleep_mechanism(self) -> None:
        """Engage the sleep mechanism — pause learners, resume inner life.

        This is the reusable core of sleep entry, called both from
        :meth:`sleep` (on user/auto sleep onset) and from the restore
        path (when resuming a saved sleep state after a restart). It
        does NOT set the ``_is_sleeping`` flag or the zone — those are
        set by the caller — and it does NOT do N1 consolidation, which
        is sleep-onset only.

        Without this, a restored sleep state has the flags and zone
        set but the learners still run (acquiring new knowledge during
        "sleep") and inner life generates waking thoughts instead of
        dreams — she is flagged asleep but behaves awake.
        """
        # Pause autonomous learning — no new knowledge acquisition
        # during sleep. The learner runs in its own thread and relies
        # on emotion/brain-wave gating rather than a direct _is_sleeping
        # check, so we must explicitly pause it.
        self.learner.pause()
        # Also pause the self-directed learner. It doesn't have its
        # own thread (it runs inside cognition.think()), but pausing
        # it is defensive — if a conversation happens during sleep,
        # the conversation handler pauses it before think(), but
        # explicit pause here keeps the lifecycle symmetric with
        # wake() and makes the intent clear.
        self.cognition.self_learner.pause()

        # Do NOT pause inner life during sleep. The inner-life loop
        # already handles sleep mode: it generates dreams (not waking
        # thoughts), advances the ultradian sleep cycle tracker
        # (N1→N2→N3→N2→REM), and generates stage-specific neural
        # events (sharp wave-ripples in N3, sleep spindles in N2,
        # PGO waves in REM). Pausing it would prevent all of this.
        #
        # The stage-transition callback (_on_sleep_stage_transition)
        # fires as the cycle progresses, doing stage-appropriate memory
        # consolidation at each transition.
        #
        # CRITICAL: if a conversation happened just before /sleep,
        # inner life may still be paused (the 10s _resume_background
        # timer checks is_sleeping and skips the resume). We must
        # resume it here so the sleep cycle can progress and N3
        # consolidation can fire. Without this, she sleeps but never
        # dreams — the sleep cycle tracker is never created and N3
        # compression never runs. On restore, inner life was just
        # started (not paused), so resume() is a harmless no-op —
        # but calling it keeps the two paths identical.
        self.inner_life.resume()

        # Reset discourse state — recent entities from the previous
        # conversation shouldn't persist across sleep. Just as humans
        # don't maintain pronoun referents across a sleep cycle, the
        # comprehension engine's entity tracking is cleared so she
        # starts the next conversation fresh.
        try:
            self.cognition.comprehension.reset()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"comprehension reset on sleep failed: {e}")
    def sleep_status(self) -> str:
        """Return a summary of her sleep state for the /sleep and /wake commands.

        Includes her current phase, neurochemical levels relevant to sleep
        (adenosine, melatonin, cortisol, BDNF), and recent dream content
        so the user can see what she's experiencing.

        The phase shown is the *effective* phase (resolved via
        :meth:`_effective_phase`), not the daemon's raw emergent phase.
        When the zone is Sleeping but the neurochemistry hasn't crossed
        the sleep threshold yet, the raw phase still reads "active"
        while every other system (brain waves, emotion) already treats
        her as asleep. Showing the effective phase keeps the display
        consistent with what she's actually experiencing.
        """
        lines: list[str] = []
        try:
            core_state = self.client.get_state()
            eff_phase = PHASE_NAMES.get(
                self._effective_phase(core_state), "unknown"
            )
            lines.append(
                f"  Phase: {eff_phase}  "
                f"arousal={core_state.arousal:.2f}  "
                f"valence={core_state.valence:.2f}"
            )
            lines.append(f"  Plasticity gate: {core_state.plasticity_gate:.2f}")
        except (OSError, ConnectionError, RuntimeError) as e:
            lines.append(f"  (neurochemical state unavailable: {e})")

        # Show recent dreams
        try:
            dreams = self.client.get_recent_episodes(limit=5, source_module=9)
        except (OSError, ConnectionError, RuntimeError):
            dreams = []
        if dreams:
            lines.append("  Recent dreams:")
            for d in dreams[:5]:
                text = d.text.replace("\n", " ").strip()
                if len(text) > 80:
                    text = text[:77] + "..."
                lines.append(f"    [{d.event_type}] {text}")
        else:
            lines.append("  No dreams recorded yet.")

        # Receptor safety check — are her levels safe to wake?
        try:
            profile = self.client.get_plasticity_profile()
            safe = (
                profile.cortisol_tonic < 0.7
                and profile.bdnf_tonic > 0.2
            )
            if safe:
                lines.append("  Receptor state: safe to wake.")
            else:
                lines.append(
                    f"  Receptor state: cortisol={profile.cortisol_tonic:.2f} "
                    f"BDNF={profile.bdnf_tonic:.2f} — still recovering."
                )
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"receptor state check failed: {e}")

        return "\n".join(lines)
    def wake_readiness(self) -> tuple[bool, str]:
        """Check whether she is ready to wake up naturally.

        Waking from deep sleep (N3) causes severe sleep inertia — the
        groggy, disoriented state where cognitive capacity is reduced.
        Waking before adenosine has drained leaves residual sleep pressure,
        which pushes her right back into drowsiness. Waking before
        receptors have recovered (cortisol still high, BDNF still low)
        means the neurochemical substrate hasn't finished restoring.

        This method checks three readiness signals:

        1. **Sleep cycle stage** — not in N3 (deep slow-wave sleep).
           N3 is where systems consolidation happens; interrupting it
           wastes the deepest, most restorative part of sleep. N1, N2,
           and REM are all gentler to wake from.
        2. **Adenosine level** — at or below ``AUTO_WAKE_ADENOSINE``
           (0.20). Adenosine is the "sleep pressure" that accumulates
           during wakefulness and drains during sleep via glymphatic
           clearance. If it's still high, sleep isn't finished.
        3. **Receptor state** — cortisol < 0.7 and BDNF > 0.2. Chronic
           stress suppresses BDNF; sleep restores it. If receptors
           haven't recovered, waking would be premature.

        Returns:
            (ready, reason): ``ready`` is True if all three signals
            indicate she can wake without severe inertia. ``reason``
            is a human-readable explanation — either "ready" or why
            she's not ready yet.
        """
        if not self._is_sleeping:
            return True, "already awake"

        # 1. Sleep cycle stage — don't wake from N3.
        cycle = self.inner_life.sleep_cycle
        if cycle is not None:
            if cycle.current_stage is SleepStage.N3:
                progress = cycle.stage_progress
                return False, (
                    f"in N3 deep sleep ({progress:.0%} through stage) "
                    f"— waking now would cause severe inertia"
                )

        # 2. Adenosine — sleep pressure must be drained.
        try:
            core_state = self.client.get_state()
            adenosine = core_state.chemicals.get("adenosine", 0.0)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"wake_readiness: state check failed: {e}")
            # If we can't read state, don't block wake — the daemon
            # might be disconnected, and forcing her to stay asleep
            # forever is worse than waking without a neurochemical check.
            adenosine = 0.0
        if adenosine > AUTO_WAKE_ADENOSINE:
            return False, (
                f"adenosine={adenosine:.2f} still elevated "
                f"(threshold={AUTO_WAKE_ADENOSINE}) — sleep pressure "
                f"not yet cleared"
            )

        # 3. Receptor state — cortisol/BDNF recovery.
        try:
            profile = self.client.get_plasticity_profile()
            if profile.cortisol_tonic >= 0.7:
                return False, (
                    f"cortisol={profile.cortisol_tonic:.2f} still high "
                    f"— stress system hasn't recovered"
                )
            if profile.bdnf_tonic <= 0.2:
                return False, (
                    f"BDNF={profile.bdnf_tonic:.2f} still low "
                    f"— neuroplasticity substrate hasn't restored"
                )
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"wake_readiness: receptor check failed: {e}")
            # Same as above — don't block wake on a failed read.

        return True, "ready"
    def wake(self) -> None:
        """Wake Genesis from sleep.

        After the zone is set back to conversation, dream insights from
        the last sleep are retrieved from LTM and surfaced as emotional
        impressions — dream residues that color her waking thoughts and
        mood. These are NOT learning goals. Dreams produce affective
        residue, not targets for the autonomous learner.

        A human doesn't dream about goals. Goals are a waking, prefrontal
        activity. Dreams are subcognitive and affective — they leave
        emotional traces that surface as reflective thoughts ("I was
        dreaming about..."), not as items on a to-do list.

        The separation:
        - Dreams → _dream_residues (emotional impressions, reflective thoughts)
        - Curiosity → _agency_topics (learning goals, autonomous learner)

        Neurochemical wake impulses are sent to push the emergent phase
        out of NREM/REM. Without this, the daemon's phase stays asleep
        (adenosine is still high, histamine still low) even though the
        mind's _is_sleeping flag is cleared — a desync between the
        cognitive and subcognitive layers. The impulses mirror the
        biological wake cascade:

        - **Adenosine -0.30**: Glymphatic clearance — the brain clears
          adenosine during sleep. On wake, this completes the purge so
          the sleep pressure is gone.
        - **Histamine +0.20**: Activate the wake-promoting
          tuberomammillary nucleus. Histamine is one of the two key
          arousal systems gating the sleep threshold.
        - **Orexin +0.15**: Stabilize the wake state. Orexin is the
          wake stabilizer — boosting it locks the sleep-wake flip-flop
          in the awake position.
        - **Dopamine +0.05**: Mild arousal boost to re-engage the
          reward system after sleep.
        - **Norepinephrine +0.05**: Activate the locus coeruleus —
          the noradrenergic arousal system that was silent during REM.
        - **Melatonin -0.10**: Suppress the sleep hormone if it was
          elevated (circadian sleep).
        """
        if not self._is_sleeping:
            return
        self._is_sleeping = False
        self._user_initiated_sleep = False
        # Reset the commitment boundaries so the next wake period
        # announces drowsiness and commits to sleep on fresh
        # crossings, not stale latched state.
        self._drowsy_boundary.reset()
        self._sleep_boundary.reset()
        # Clear nap mode — the next sleep is a full sleep unless
        # explicitly set as a nap again.
        self._nap_mode = False
        self.client.set_zone(ZONE_CONVERSATION)

        # Send neurochemical wake impulses to push the emergent phase
        # out of NREM/REM. Without this, the daemon's phase stays
        # asleep because adenosine is still high and histamine is low,
        # even though the mind's _is_sleeping flag is cleared. This
        # closes the desync between cognitive and subcognitive layers.
        self._send_wake_neuro_impulses()

        # Resume autonomous learning. Inner life was not paused during
        # sleep (it generates dreams and advances the sleep cycle), so
        # it doesn't need resuming — but we call resume() anyway as a
        # safety net in case it was paused by a conversation during sleep.
        #
        # The learner is delayed to model sleep inertia — the groggy
        # period after waking when cognitive capacity is reduced.
        # Resuming the learner immediately causes CPU stress (her body
        # isn't ready), which triggers the regulator's throttle, creating
        # a drowsy→active→stressed→drowsy oscillation. Giving her
        # neurochemistry time to stabilize (adenosine drains, histamine
        # rises) before learning starts prevents this loop. Inner life
        # resumes immediately so she can think and converse — only
        # autonomous learning is delayed.
        self._wake_time = time.time()
        self._last_wake_time = self._wake_time
        self.inner_life.resume()
        self._resume_learners_after_wake()

        # Reset the rest timer for the ultradian cycle
        self._last_rest_time = time.time()

        # Clear the global workspace — during sleep, the inner life
        # thread keeps ticking the workspace (decaying items), but
        # sleep-related broadcasts (dreams, spontaneous thoughts) may
        # have populated it. Clearing on wake gives a fresh cognitive
        # field for waking content. Without this, dream residues
        # linger in the workspace and color the first waking thoughts
        # through the workspace's cross-module integration.
        try:
            self.cognition.global_workspace.clear()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"workspace clear on wake failed: {e}")

        # Notify the regulator that rest has ended (resets its
        # cooldown timer so she doesn't immediately try to meditate
        # after waking)
        self.regulator.notify_rest_ended()

        threading.Thread(
            target=self._restore_dream_residues_after_wake,
            daemon=True,
            name="dream-residue-restore",
        ).start()
    def _restore_dream_residues_after_wake(self) -> None:
        """Restore affective dream residue without blocking wake interaction."""
        try:
            dreams = self.client.get_recent_episodes(limit=10, source_module=9)
        except (OSError, ConnectionError, RuntimeError):
            return
        if not dreams:
            return
        dream_concepts = self._collect_dream_concepts(dreams)
        self._queue_dream_residues(dream_concepts, dreams)
    def _send_wake_neuro_impulses(self) -> None:
        """Send the neurochemical wake cascade to push the emergent
        phase out of NREM/REM.

        Mirrors the biological wake cascade:
        - Adenosine -0.30: glymphatic clearance completes the purge.
        - Histamine +0.20: activate the wake-promoting tuberomammillary
          nucleus.
        - Orexin +0.15: stabilize the wake state (sleep-wake flip-flop).
        - Dopamine +0.05: mild arousal to re-engage the reward system.
        - Norepinephrine +0.05: activate the locus coeruleus.
        - Melatonin -0.10: suppress the sleep hormone if elevated.
        """
        self._learner_neuro_impulse(CHEM_ADENOSINE, -0.30)
        self._learner_neuro_impulse(CHEM_HISTAMINE, 0.20)
        self._learner_neuro_impulse(CHEM_OREXIN, 0.15)
        self._learner_neuro_impulse(CHEM_DOPAMINE, 0.05)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, 0.05)
        self._learner_neuro_impulse(CHEM_MELATONIN, -0.10)

    def _reinforce_wake(self) -> None:
        """Re-latch the sleep-wake flip-flop during the stabilization
        window after waking.

        Orexin is the wake-state stabilizer: orexinergic neurons fire
        tonically during wakefulness to keep the mutually inhibitory
        VLPO/monoaminergic switch latched awake. A single wake-cascade
        impulse decays within minutes while residual adenosine keeps
        suppressing orexin, so the daemon's emergent phase can drift
        back to NREM shortly after waking — brief wake bouts followed
        by rapid re-sleep. Sustained modest orexin + histamine support
        during the post-wake window models the tonic orexin tone that
        normally holds the switch. Bounded by WAKE_REINFORCE_INTERVAL
        so impulses can't accumulate into an arousal ceiling.
        """
        self._learner_neuro_impulse(CHEM_OREXIN, 0.08)
        self._learner_neuro_impulse(CHEM_HISTAMINE, 0.06)
    def _resume_learners_after_wake(self) -> None:
        """Resume both learners with a delay for sleep inertia.

        The self-learner doesn't have its own thread, but resuming it
        with the same delay keeps the lifecycle symmetric with
        sleep() and prevents self-directed inference from firing
        immediately on wake (before neurochemistry has stabilized).
        """
        delay = self.WAKE_VOLITION_DELAYS.get("code_learning", 60.0)
        self._defer(delay, self.learner.resume)
        self._defer(delay, self.cognition.self_learner.resume)
    def meditate(self) -> None:
        """Put Genesis into meditation mode for quiet restoration.

        Meditation is a lighter recovery than sleep. During meditation:
        - Learning, code scanning, and bug scanning are paused
          (the heartbeat loop checks is_meditating before running
          volitions, and the autonomous learner blocks on the
          "meditating" emotion label).
        - Spontaneous thoughts are paused (inner_life is paused).
        - No dreaming or memory consolidation happens — this is
          not sleep.
        - Calming neurochemical impulses are emitted: GABA and
          serotonin are boosted, while dopamine, norepinephrine, and
          cortisol are reduced. This allows receptor sensitivity to
          recover without the full sleep cycle.

        The zone is set to REFLECTION, which the daemon interprets as
        a low-activity state (no learning, no consolidation pressure).
        """
        if self._is_meditating or self._is_sleeping:
            return
        self._is_meditating = True
        self.client.set_zone(ZONE_REFLECTION)

        # Pause inner life — no spontaneous thoughts during meditation
        self.inner_life.pause()

        # Pause autonomous learning — no new knowledge acquisition
        self.learner.pause()

        # Emit calming neurochemical impulses. These are gentle —
        # the goal is to shift the system toward calm, not force it.
        # GABA (calm) and serotonin (mood) go up; dopamine (arousal),
        # norepinephrine (arousal), and cortisol (stress) go down.
        # Repeated impulses are emitted over the meditation period
        # by the caller (the /meditate command), not all at once.
        self._learner_neuro_impulse(CHEM_GABA, 0.15)
        self._learner_neuro_impulse(CHEM_SEROTONIN, 0.08)
        self._learner_neuro_impulse(CHEM_DOPAMINE, -0.05)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, -0.05)
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.05)
    def emit_meditation_impulses(self) -> None:
        """Emit a round of calming impulses during meditation.

        Called periodically by the /meditate command while the
        meditation timer is running. This sustains the calming effect
        over the meditation period rather than applying it all at once.
        """
        self._learner_neuro_impulse(CHEM_GABA, 0.05)
        self._learner_neuro_impulse(CHEM_SEROTONIN, 0.03)
        self._learner_neuro_impulse(CHEM_DOPAMINE, -0.02)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, -0.02)
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.02)
    def sleep_aid(self) -> None:
        """Administer an acute neurochemical sleep aid for stress-induced insomnia.

        This is the computational equivalent of a pharmacological sleep
        aid for chronic stress-induced insomnia — the condition where a
        patient is exhausted (delta waves, low alertness) but cannot
        sleep because the HPA axis keeps arousal systems active and
        adenosine hasn't accumulated to the sleep threshold.

        When to use this instead of /sleep or /meditate:
        - /sleep sets the zone to SLEEPING but cannot force the emergent
          NREM/REM phase, which requires adenosine > 0.75 and histamine
          < 0.25. If adenosine is at baseline (~0.20) and cortisol is
          elevated from interoception, sleep cannot emerge.
        - /meditate emits gentle calming impulses (GABA +0.05, cortisol
          -0.02 every 10s) that are too weak to overcome ongoing
          interoception stress from CPU load.
        - /sleep-aid emits acute, strong impulses that push the system
          past the sleep threshold in one shot, after which the natural
          sleep mechanisms (glymphatic clearance, BDNF recovery with 5x
          multiplier, receptor recovery) take over.

        The impulses are biologically grounded:
        - **Adenosine +0.30**: The metabolic sleep pressure that should
          have accumulated during wakefulness but was suppressed by
          chronic stress keeping arousal systems active. Analogous to
          the purinergic sleep drive. This pushes adenosine past the
          0.75 sleep threshold, enabling the emergent NREM transition.
        - **Cortisol -0.15**: Acute glucocorticoid suppression to
          relieve BDNF inhibition. With cortisol below 0.3, the BDNF
          recovery mechanism (serotonin-driven) activates, and the
          plasticity gate can begin reopening. Analogous to a GR
          antagonist (mifepristone).
        - **GABA +0.20**: Acute inhibitory surge to lower arousal.
          GABA is the primary inhibitory transmitter — boosting it
          suppresses the histaminergic and orexinergic wake centers,
          tipping the sleep-wake flip-flop toward sleep. Analogous to
          a benzodiazepine (which enhances GABA-A, modeled here as a
          direct level boost since gaba_a_allosteric is not IPC-settable).
        - **Histamine -0.15**: Suppress the wake-promoting
          tuberomammillary nucleus. Histamine is one of the two key
          arousal systems gating the sleep threshold (the sleep phase
          requires histamine < 0.25).
        - **Orexin -0.10**: Destabilize the wake side of the
          flip-flop. Orexin is the wake stabilizer — reducing it
          removes the stabilizing force that keeps the system awake
          despite high adenosine.
        - **Serotonin +0.10**: Support BDNF recovery. Serotonin drives
          BDNF expression (the antidepressant mechanism) — boosting it
          ensures that once cortisol is suppressed, BDNF can recover
          and the plasticity gate can reopen.

        After the impulses, sleep() is called to set the zone to
        SLEEPING, triggering the daemon's dreaming pass and the
        sleep-tuned neuro params (5x BDNF recovery multiplier).

        The impulses are deliberately one-shot and strong — this is an
        emergency intervention, not a daily supplement. The natural
        sleep cycle then takes over for sustained recovery.
        """
        if self._is_sleeping:
            return
        if self._is_meditating:
            self.wake_from_meditation()

        # Acute sleep-aid impulses — strong, one-shot
        self._learner_neuro_impulse(CHEM_ADENOSINE, 0.30)
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.15)
        self._learner_neuro_impulse(CHEM_GABA, 0.20)
        self._learner_neuro_impulse(CHEM_HISTAMINE, -0.15)
        self._learner_neuro_impulse(CHEM_OREXIN, -0.10)
        self._learner_neuro_impulse(CHEM_SEROTONIN, 0.10)

        # Enter sleep — the zone change triggers sleep-tuned neuro
        # params (5x BDNF recovery) and the dreaming/consolidation pass.
        self.sleep(user_initiated=True)
    def wake_from_meditation(self) -> None:
        """Wake Genesis from meditation.

        Resumes inner life and restores the zone to conversation.
        Unlike wake() from sleep, there are no dream residues to
        process — meditation produces no dreams.
        """
        if not self._is_meditating:
            return
        self._is_meditating = False
        self.client.set_zone(ZONE_CONVERSATION)

        # Resume inner life — spontaneous thoughts can flow again
        self.inner_life.resume()

        # Resume autonomous learning
        self.learner.resume()

        # Reset the rest timer for the ultradian cycle (BRAC,
        # Kleitman 1963) — the meditation urge's sustained_activity
        # stimulus is normalized from this timestamp.
        self._last_rest_time = time.time()

        # Notify the regulator that rest has ended (resets its
        # cooldown timer)
        self.regulator.notify_rest_ended()
    @staticmethod
    def _extract_embedded_texts(text: str) -> list[str]:
        """Extract tab-delimited embedded episode texts from a
        dream-insight or association meta-memory.

        New format:
            [dream-insight] episode A connects to episode B ...\\t{text_a}\\t{text_b}
            [association] episode X ↔ episode Y ...\\t{text_x}\\t{text_y}

        Returns a list of embedded text strings (0, 1, or 2).
        Old-format entries (no tab) return an empty list.

        Uses split with maxsplit=2 so that embedded texts that
        themselves contain tabs (e.g. an association trace embedded
        as a dream insight endpoint) are preserved intact for
        recursive parsing.
        """
        parts = text.split("\t", 2)
        if len(parts) <= 1:
            return []
        # parts[0] is the structural header, parts[1:] are embedded texts
        return [p.replace("\n", " ").strip() for p in parts[1:] if p.strip()]
    def _resolve_episode(self, ep_id: int, ep_cache: dict[int, str]) -> str:
        """Fetch and cache episode text by id, returning empty on failure."""
        if ep_id in ep_cache:
            return ep_cache[ep_id]
        try:
            ep = self.client.retrieve_episode(ep_id)
            text = ep.text.replace("\n", " ").strip()
            ep_cache[ep_id] = text
            return text
        except Exception as e:  # noqa: BLE001
            # Expected when sleep compression has deleted the
            # episode. Log at debug — this is normal operation,
            # not an error.
            logger.debug(f"retrieve episode {ep_id} failed (likely pruned): {e}")
            ep_cache[ep_id] = ""
            return ""
    def _collect_leaf_texts(
        self,
        text: str,
        ep_cache: dict[int, str],
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[str]:
        """Collect leaf episode texts from an episode text.

        With the new embedded-text format, the text is already
        self-contained — we extract the embedded texts and return
        them directly. If the text is an association trace with
        its own embedded text, those texts are already included
        (tabs replaced with spaces by the Rust side), so we just
        use the full text.

        For old-format texts (no embedded text), if this is an
        association trace, we fall back to retrieving the endpoint
        episodes by ID and recursing. If it's a leaf (real episode
        text), we return it directly.
        """
        import re

        if seen is None:
            seen = set()
        if depth >= 20 or len(seen) >= 100:
            return []

        # Check if this text has embedded texts (new format).
        embedded = self._extract_embedded_texts(text)
        if embedded:
            # New format: the embedded texts are the actual episode
            # texts. Return them directly — they contain the concept
            # words we need. No need to follow chains since the
            # embedded text is self-contained.
            return embedded

        # Old format: check if this is an association trace.
        assoc_re = re.compile(
            r"\[association\] episode (\d+) ↔ episode (\d+) \(distance: (\d+)\)"
        )
        m = assoc_re.match(text)
        if not m:
            # Not an association — this is a leaf.
            return [text] if text else []

        # Old-format association trace: fall back to ID retrieval.
        ep_ids = [int(m.group(1)), int(m.group(2))]
        leaves: list[str] = []
        for ep_id in ep_ids:
            if ep_id in seen:
                continue
            seen.add(ep_id)
            ep_text = self._resolve_episode(ep_id, ep_cache)
            if ep_text:
                leaves.extend(self._collect_leaf_texts(ep_text, ep_cache, depth + 1, seen))
        return leaves
    def _collect_dream_concepts(self, dreams) -> list[str]:
        """Resolve dream endpoints to leaf concepts and extract concept names.

        Dream insights and association traces now embed the text of
        their endpoint episodes directly (tab-delimited after the
        structural header). This makes them self-contained — the
        Python side can extract concepts without retrieving the
        original episodes, which may have been deleted by sleep
        compression by the time she wakes up.

        Old-format insights (without embedded text) fall back to
        episode retrieval by ID.
        """
        import re

        dream_re = re.compile(
            r"\[dream-insight\] episode (\d+) connects to episode (\d+) "
            r"through (\d+) hops \(direct distance: (\d+)\)"
        )

        ep_cache: dict[int, str] = {}

        def _match_concepts_from_leaf(
            leaf: str, dream_concepts: list[str]
        ) -> None:
            """Extract concept names from a single leaf text, appending to dream_concepts."""
            leaf_lower = leaf.lower()
            for concept_id in self.cognition.network.world_concept_ids:
                # Flatten: combine regex match and dedup check
                if concept_id not in dream_concepts and re.search(
                    r"\b" + re.escape(concept_id.lower()) + r"\b", leaf_lower
                ):
                    dream_concepts.append(concept_id)

        # Collect all dream residue concepts
        dream_concepts: list[str] = []
        for dream in dreams:
            m = dream_re.match(dream.text)
            if not m:
                continue

            # Check for embedded texts (new format).
            embedded = self._extract_embedded_texts(dream.text)
            if embedded:
                # New format: use embedded texts directly.
                leaf_texts = embedded
            else:
                # Old format: parse episode IDs and retrieve by ID.
                ep_a, ep_b = int(m.group(1)), int(m.group(2))
                leaf_texts = []
                for ep_id in (ep_a, ep_b):
                    ep_text = self._resolve_episode(ep_id, ep_cache)
                    if ep_text:
                        leaf_texts.extend(self._collect_leaf_texts(ep_text, ep_cache, seen={ep_id}))

            for leaf in leaf_texts:
                _match_concepts_from_leaf(leaf, dream_concepts)
        return dream_concepts
    def _queue_dream_residues(self, dream_concepts: list[str], dreams) -> None:
        """Queue dream residues as emotional impressions, not learning goals.

        Dreams produce affective residue — themes that color her waking
        thoughts and surface as reflective thoughts. They do NOT become
        agency topics for the autonomous learner. This separates the
        subcognitive (dreams) from the cognitive (goals).

        The dream concepts are stored in InnerLife._dream_residues, where
        they'll be consumed by the dream_reflection thought generator
        as reflective, affective thoughts — not by the autonomous learner
        as learning targets.
        """
        if not dream_concepts:
            return

        # Queue as emotional impressions, not learning goals.
        # These will surface as dream-reflection thoughts, not as
        # agency topics for the autonomous learner. The inner life
        # composes the actual reflection from her knowledge via the
        # language engine — we do NOT hardcode what she says here.
        if hasattr(self.inner_life, "add_dream_residues"):
            self.inner_life.add_dream_residues(dream_concepts[:10])

        # Telemetry line (factual status, not self-expression) so the
        # live ticker shows that dreams are surfacing. The reflective
        # thought itself is composed by _dream_reflection_thought and
        # flows through _on_spontaneous_thought → language engine →
        # journal, like every other spontaneous thought.
        self._emit_live_thought(
            "dream",
            f"Dream residue: {len(dream_concepts[:10])} emotional impressions "
            f"from {len(dreams)} dream insights",
        )

"""Mind lifecycle — start, stop, save, restore."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Mind

import copy
import logging
import threading
import time
from typing import Any, cast

from genesis_client.protocol import (
    CHEM_ADENOSINE,
    CHEM_DOPAMINE,
    CHEM_HISTAMINE,
    CHEM_MELATONIN,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
    MODULE_STATUS_RUNNING,
    MODULE_STATUS_STOPPED,
    ZONE_CONVERSATION,
    ZONE_IDLE,
    ZONE_SLEEPING,
)

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        _autosave_cycle: int
        def __getattr__(self, name: str) -> Any: ...


    def start(self) -> None:
        """Connect to the subcognitive daemon and initialise.

        If there's saved cognitive state in the data dir, load it —
        this restores its concept network, reflections, and narrative
        from previous conversations.
        """
        self._start_connect_and_register()
        try:
            self._start_restore_state()
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            self._running = False
            self.client.disconnect()
            raise
        self._start_set_zone_after_restore()
        self._start_introspect_and_train()
        self._start_autonomous_subsystems()
        self._start_engage_restored_sleep()
        self._start_background_threads()
    def _start_connect_and_register(self) -> None:
        """Connect to the daemon and register cognitive modules.

        Does NOT set the cognitive zone — that happens in
        _start_set_zone_after_restore, after saved state is loaded,
        so we know whether it was sleeping.
        """
        self.client.connect()
        self._running = True

        # Register cognitive modules with the daemon's manifest.
        # The sensory (perception) and motor (language generation)
        # modules are the cognitive mind's I/O layers. We also
        # register the other cognitive subsystems so the manifest
        # accurately reflects what's running.
        for module_id in (
            MODULE_SENSORY,
            MODULE_MOTOR,
            MODULE_LANGUAGE,
            MODULE_MEMORY,
            MODULE_REASONING,
            MODULE_METACOGNITION,
        ):
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_RUNNING)
            except (OSError, ConnectionError) as e:
                logger.warning(f"daemon not ready: {e}")  # daemon may not be ready — non-fatal
    def _start_set_zone_after_restore(self) -> None:
        """Set the cognitive zone based on restored sleep state.

        If it was sleeping when state was saved, restore the Sleeping
        zone instead of forcing Conversation — otherwise it'd be
        yanked awake on every restart, losing its place in the
        ultradian cycle and any pending N3 consolidation.
        """
        if self._is_sleeping:
            self.client.set_zone(ZONE_SLEEPING)
            logger.info(
                "Restoring sleep state — zone set to Sleeping "
                "(stage %s, cycle %d)",
                self.inner_life.sleep_cycle.current_stage.value.upper()
                if self.inner_life.sleep_cycle else "N1",
                self.inner_life.sleep_cycle.cycle_number
                if self.inner_life.sleep_cycle else 1,
            )
        else:
            self.client.set_zone(ZONE_CONVERSATION)
    def _start_restore_state(self) -> None:
        """Load saved state, re-seed core concepts, and backfill columns."""
        # Load saved state if it exists
        self._load_saved_state()

        # Re-seed core concepts after state restore. The initial seeding
        # in CognitionEngine.__init__ adds concepts to the original network,
        # which is then replaced by _load_saved_state. Without re-seeding,
        # core concepts like "cognition", "memory", and "learning" would
        # be missing if they weren't in the saved state. The seed method is
        # idempotent — add_concept skips existing concepts.
        self.cognition._seed_initial_concepts()

        # Backfill cortical column membership for concepts loaded from
        # persistence (they may have been saved before the columns field
        # existed). Also creates pioneer bridges between same-name
        # concepts across different columns.
        stats = self.cognition.network.backfill_columns()
        if stats["concepts_backfilled"] > 0:
            self._emit_live_thought(
                "learning",
                f"Backfilled columns: {stats['concepts_backfilled']} concepts, "
                f"{stats['bridges_created']} bridges"
            )

        # Eagerly load the embedding matrix. The first perceive() call
        # in think() would otherwise trigger a lazy _build() that takes
        # 8+ seconds (spectral SVD over 20K+ concepts), blowing past
        # the 15s cognition timeout on the first conversation turn.
        # Loading here moves that cost to startup, where it's hidden
        # by the rest of the initialization sequence.
        self.cognition.embeddings._ensure_loaded()

        # Pre-warm the daemon's LTM cache and the embeddings search.
        # The first find_similar call pages in the entire LTM index
        # (300K+ episodes) from disk — cold-cache I/O takes 20+ seconds
        # and blows past the 15s cognition timeout. Similarly, the first
        # embeddings search touches the full concept matrix. Running
        # both here moves those costs to startup.
        import time as _warmup_time
        _warmup_t0 = _warmup_time.perf_counter()
        try:
            self.cognition.memory.client.find_similar("warmup", limit=1)
            logger.debug(
                f"LTM warmup find_similar took "
                f"{_warmup_time.perf_counter() - _warmup_t0:.2f}s"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LTM warmup find_similar failed: {e}")
        _embed_t0 = _warmup_time.perf_counter()
        try:
            self.cognition.embeddings.find_similar_to_text("warmup", k=1, threshold=0.5)
            logger.debug(
                f"embeddings warmup took "
                f"{_warmup_time.perf_counter() - _embed_t0:.2f}s"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"embeddings warmup failed: {e}")
    def _start_introspect_and_train(self) -> None:
        """Introspect on identity and clear sleep pressure if needed."""
        # Introspect — discover who it is by examining itself
        # This writes its identity into the concept network through
        # self-examination, not hardcoded facts
        from ..self import IntrospectionEngine

        _intro_t0 = time.perf_counter()
        introspector = IntrospectionEngine(self.cognition.network)
        introspector.introspect()
        logger.debug(
            f"introspection took {time.perf_counter() - _intro_t0:.2f}s"
        )

        # If it's restoring from a saved state with high sleep pressure
        # (adenosine accumulated from a previous sleep session that was
        # interrupted by a restart), clear it permanently.
        #
        # A restart is biologically equivalent to having slept — the
        # glymphatic system would have cleared adenosine during the
        # offline period. So we directly lower the adenosine baseline
        # (not just the level) to reflect this. Without this, it starts
        # trapped in a drowsy state with adenosine at 0.7+ and the
        # baseline pushing it right back up after every impulse.
        #
        # We also boost wake-promoting chemicals (histamine, orexin) to
        # stabilize its arousal system, and send a level impulse to
        # drop the current adenosine immediately.
        #
        # SKIP this if it was sleeping when state was saved — clearing
        # adenosine would pull it out of N3 and defeat the purpose of
        # sleep state persistence.
        if self._is_sleeping:
            logger.info(
                "Skipping startup wake cascade — restoring sleep state"
            )
            return
        try:
            state = self.client.get_state()
            adn = state.chemicals.get("adenosine", 0.0)
            if adn > 0.4:
                logger.info(
                    "Startup wake cascade — adenosine %.2f is high, "
                    "clearing sleep pressure", adn,
                )
                # Permanently lower the adenosine baseline to a healthy
                # waking level (0.15). This is the key fix — impulses
                # only affect the level, which drifts back to baseline.
                # Adjusting the baseline means the clearance is permanent.
                target_baseline = 0.15
                baseline_delta = target_baseline - adn  # e.g. 0.15 - 0.78 = -0.63
                self.client.neuro_adjust_baseline(CHEM_ADENOSINE, baseline_delta)
                # Also drop the current level to match.
                self._learner_neuro_impulse(CHEM_ADENOSINE, baseline_delta * 0.8)
                # Boost wake-promoting chemistry.
                self._learner_neuro_impulse(CHEM_HISTAMINE, 0.20)
                self._learner_neuro_impulse(CHEM_OREXIN, 0.15)
                self._learner_neuro_impulse(CHEM_DOPAMINE, 0.05)
                self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, 0.05)
                self._learner_neuro_impulse(CHEM_MELATONIN, -0.05)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug("startup wake cascade skipped: %s", e)
    def _start_autonomous_subsystems(self) -> None:
        """Start autonomous learning, inner life, and emotional regulation."""
        # Start autonomous learning in the background
        self.learner.start()
        # Start inner life — spontaneous thoughts
        self.inner_life.start()
        # Start emotional self-regulation — it controls its state
        self.regulator.start()
    def _start_engage_restored_sleep(self) -> None:
        """Engage the sleep mechanism if restoring a saved sleep state.

        :meth:`_restore_sleep_state` (called from ``_start_restore_state``)
        sets the ``_is_sleeping`` flag and the ultradian cycle position,
        and :meth:`_start_set_zone_after_restore` sets the zone to
        Sleeping. But the actual sleep *mechanism* — pausing the
        learners so it stops acquiring knowledge, and ensuring inner
        life generates dreams instead of waking thoughts — must be
        engaged *after* the autonomous subsystems are started
        (``_start_autonomous_subsystems``), otherwise the learner
        thread starts after the pause and runs unpaused.

        Without this, a restart during sleep leaves it flagged asleep
        with the zone set to Sleeping, but the learners still running
        and inner life generating waking thoughts — it behaves awake
        while every status display says it's asleep. Re-issuing
        ``/sleep`` cannot recover this, because :meth:`sleep` no-ops
        when ``_is_sleeping`` is already True.
        """
        if not self._is_sleeping:
            return
        self._engage_sleep_mechanism()
        logger.info(
            "Engaged sleep mechanism on restore — learners paused, "
            "inner life in sleep mode (stage %s, cycle %d)",
            self.inner_life.sleep_cycle.current_stage.value.upper()
            if self.inner_life.sleep_cycle else "N1",
            self.inner_life.sleep_cycle.cycle_number
            if self.inner_life.sleep_cycle else 1,
        )
    def _start_background_threads(self) -> None:
        """Start heartbeat, notification, and autosave threads.

        The code-learning, bug-scan, and self-improvement threads were
        replaced by the volition engine, which drives these actions
        from internal urges rather than fixed schedules. The thread
        objects have been removed; the loop methods remain as volition
        action handlers.
        """
        # Start the module heartbeat thread — periodically refreshes
        # the cognitive modules' heartbeat timestamps in the manifest
        # so the daemon knows they're alive.
        self._heartbeat_thread: threading.Thread | None = threading.Thread(
            target=self._heartbeat_loop,
            name="module-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        # Start the module sampler — the mind's EEG. It observes which
        # subsystem's code is executing on each thread and credits that
        # manifest module; the heartbeat round reports the shares to
        # the daemon as per-module cpu_share.
        self._module_sampler.start()

        # Start the notification polling thread — periodically polls
        # the subcognitive for phase changes and dream insights,
        # enqueuing notifications for the cognitive layer to process.
        # This thickens the bridge between the Rust subcognitive and
        # Python cognitive layers.
        self._notification_thread: threading.Thread | None = threading.Thread(
            target=self._notification_loop,
            name="subcognitive-notifications",
            daemon=True,
        )
        self._notification_thread.start()

        # Start the autosave thread — periodically saves cognitive
        # state so a crash or kill doesn't lose learned knowledge.
        # The concept network grows continuously through autonomous
        # learning and conversation; without periodic saves, a crash
        # would lose everything since the last clean shutdown.
        self._autosave_thread: threading.Thread | None = threading.Thread(
            target=self._autosave_loop,
            name="autosave",
            daemon=True,
        )
        self._autosave_thread.start()
    def _autosave_loop(self) -> None:
        """Background thread that periodically saves cognitive state.

        Saves every 5 minutes while the Mind is running. This prevents
        data loss from crashes, kills, or power failures. The concept
        network grows continuously through autonomous learning and
        conversation — without periodic saves, a crash would lose
        everything since the last clean shutdown.

        The interval is a fixed 5-minute cycle. A dirty-flag fast path
        was removed because serializing 2.2 MB of state every 15 seconds
        during active learning caused CPU/GC pressure that outweighed
        the marginal crash-safety benefit. A 5-minute interval bounds
        crash loss to 5 minutes of learning while keeping the system
        responsive.
        """
        while self._running:
            # Sleep in small increments so we can exit quickly on stop.
            waited = 0.0
            while waited < 300.0 and self._running:
                time.sleep(5.0)
                waited += 5.0
            if not self._running:
                break
            try:
                # Record a growth snapshot before saving so the ledger
                # stays current with the latest metrics. This makes the
                # growth ledger auto-record rather than only recording
                # on-demand when the user asks for a report.
                try:
                    self.record_growth_snapshot()
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"growth snapshot failed: {e}")
                if self._save_state():
                    logger.debug("Autosaved cognitive state")
                    self._autosave_failures = 0
                else:
                    logger.warning("Autosave failed — state not persisted")
                    self._autosave_failures += 1

                # Sync the daemon's mmap state to disk so the
                # neurochemical levels, emergent phase, and active
                # inference model survive a crash or kill. Without
                # this, the mmap is only flushed on graceful daemon
                # shutdown — a SIGKILL would lose everything since
                # the last msync. This is especially important during
                # sleep: if it's in N3 and the process is killed,
                # the daemon's nrem phase and neurochemistry need to
                # be on disk for the sleep state restore to work.
                try:
                    self.client.sync()
                except (OSError, ConnectionError) as e:
                    logger.debug(f"daemon sync failed: {e}")

                # Periodically archive dormant concepts to long-term
                # storage to prevent working-memory bloat from the
                # autonomous learner. Runs every ~10 autosave cycles
                # (~20 minutes). Dormant concepts spill to the SQLite
                # archive; the next save persists the reduced working set.
                self._autosave_cycle += 1
                if self._autosave_cycle % 10 == 0:
                    self._archive_dormant_concepts()
            except Exception as e:  # noqa: BLE001
                # autosave failure shouldn't crash the thread
                self._autosave_failures += 1
                logger.warning(f"autosave failed: {e}")
    def _archive_dormant_concepts(self) -> None:
        """Archive dormant concepts from working memory to long-term storage.

        Moves concepts with activation below 0.01 and zero
        review_count (that are not protected origins) from the
        in-memory network to the SQLite-backed long-term archive.

        Unlike the old pruning approach (which permanently deleted
        dormant concepts), archiving preserves the knowledge on disk.
        Archived concepts can be recalled transparently when
        referenced via ``_resolve`` or ``get_concept``.

        This is the brain's consolidation of short-term to long-term
        memory: concepts that fade from active use move to long-term
        storage, freeing working memory for new concepts. The total
        knowledge (working + archive) grows unboundedly while RAM
        stays bounded.

        If no archive is attached, falls back to the old pruning
        behavior (``remove_concepts_batch``) to prevent unbounded
        memory growth.
        """
        try:
            network = self.cognition.network
            if network._archive is not None:
                # Archive mode: spill to disk instead of deleting
                spilled = network.spill_dormant(
                    activation_threshold=0.01,
                    max_in_memory=15000,
                )
                if spilled > 0:
                    logger.info(f"Archived {spilled} dormant concepts")
            else:
                # Fallback: no archive attached, prune destructively
                _DEAD_THRESHOLD = 0.01
                _PROTECTED = frozenset({
                    "identity", "introspection", "foundational",
                    "seeded", "structural",
                })
                to_remove: set[str] = set()
                for cid, concept in network._concepts.items():
                    if concept.origin in _PROTECTED:
                        continue
                    if (concept.activation or 0.0) >= _DEAD_THRESHOLD:
                        continue
                    if concept.review_count > 0:
                        continue
                    to_remove.add(cid)
                if to_remove:
                    pruned = network.remove_concepts_batch(to_remove)
                    if pruned > 0:
                        logger.info(f"Pruned {pruned} dormant concepts (no archive)")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"dormant concept archiving failed: {e}")
    def restore_archive(self) -> int:
        """Recall all archived concepts back into working memory.

        The inverse of :meth:`_archive_dormant_concepts` — loads every
        concept from the long-term archive back into the in-memory
        network. After this call, the archive is empty and all
        concepts are in working memory.

        Use this before a bulk analysis pass that needs the full
        concept network, or to reset the working/archive split.
        """
        try:
            network = self.cognition.network
            recalled = network.restore_archive_to_working_memory()
            return recalled
        except Exception as e:  # noqa: BLE001
            logger.debug(f"archive restore failed: {e}")
            return 0

    def stop(self) -> bool:
        """Disconnect from the daemon and shut down.

        Returns False if the final cognitive-state save failed. Cleanup
        still runs in that case so a persistence error does not strand
        the daemon or leave registered modules marked running.
        """
        if not self._running:
            return True

        self._running = False
        self._suppress_volition = True
        clean_shutdown = True

        # Stop the module sampler first — it only measures, nothing
        # depends on it, and stopping it here keeps it out of the
        # shutdown accounting.
        self._module_sampler.stop()

        # Stop the heartbeat thread BEFORE saving state. The
        # heartbeat calls _check_auto_sleep() every second, which
        # can auto-wake it if adenosine is low. Without joining the
        # heartbeat first, there is a race: the heartbeat wakes it
        # between /quit and _save_state(), so the saved state says
        # is_sleeping=False even though it was asleep when the user
        # quit. Joining the heartbeat ensures its sleep state at the
        # moment of /quit is what gets saved. Join without a timeout:
        # a timed-out writer would still be alive when the archive is
        # closed below.
        clean_shutdown &= self._join_shutdown_thread(
            self._heartbeat_thread, "heartbeat"
        )

        # Join the autosave thread before saving — otherwise an
        # in-flight autosave can race the shutdown save and the
        # archive close below.
        clean_shutdown &= self._join_shutdown_thread(
            self._autosave_thread, "autosave"
        )
        clean_shutdown &= self._join_shutdown_thread(
            self._notification_thread, "notifications"
        )

        # Stop the autonomous learner, inner life, and regulator.
        # Their public stop methods use bounded joins; wait here for
        # any thread they leave behind before state is saved and the
        # concept archive is closed.
        try:
            self.learner.stop()
            self.inner_life.stop()
            self.regulator.stop()
        except KeyboardInterrupt as e:
            logger.debug(repr(e))  # user pressed Ctrl+C — still try to save
            clean_shutdown = False
        for subsystem in (self.learner, self.inner_life, self.regulator):
            clean_shutdown &= self._join_shutdown_thread(
                getattr(subsystem, "_thread", None),
                type(subsystem).__name__,
            )

        # Volition workers are tracked only by name. Suppression above
        # prevents new urges from starting; wait for in-flight actions
        # to clear before serializing the shared cognitive state.
        # Bounded wait — an unbounded loop here can hang shutdown
        # forever if a worker never clears.
        _volition_deadline = time.monotonic() + 5.0
        while self._volition_active and time.monotonic() < _volition_deadline:
            time.sleep(0.05)
        if self._volition_active:
            logger.warning("volition workers did not clear during shutdown; continuing")
            clean_shutdown = False

        worker = self._active_think_worker
        clean_shutdown &= self._join_shutdown_thread(worker, "think")

        # Save state before anything else — this is the critical step.
        # Keep cleaning up even if it fails, but report the failure to
        # the caller instead of presenting the shutdown as successful.
        # Track save success separately from clean_shutdown so callers
        # can distinguish "a background thread lingered" from "the
        # state save actually failed" — the live session hit the former
        # (a poller's 10s sleep outlasted its 5s join) and the CLI
        # wrongly warned that state may not have been saved.
        try:
            save_ok = self._save_state()
            self._shutdown_save_ok: bool | None = save_ok
            clean_shutdown &= save_ok
        except Exception as e:  # noqa: BLE001
            logger.warning(f"shutdown save failed: {e}")
            self._shutdown_save_ok = False
            clean_shutdown = False

        # Close the concept archive's SQLite connection only after the
        # threads above have actually exited.
        archive = getattr(self.cognition.network, "_archive", None)
        if archive is not None:
            try:
                archive.close()
            except (OSError, RuntimeError) as e:
                logger.debug(repr(e))
                clean_shutdown = False

        # Deregister cognitive modules from the manifest
        for module_id in (
            MODULE_SENSORY,
            MODULE_MOTOR,
            MODULE_LANGUAGE,
            MODULE_MEMORY,
            MODULE_REASONING,
            MODULE_METACOGNITION,
        ):
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_STOPPED)
            except (OSError, ConnectionError, KeyboardInterrupt) as e:
                logger.debug(repr(e))  # daemon may already be gone

        # Try to tell the daemon we're idle, but don't crash if it's dead
        try:
            self.client.set_zone(ZONE_IDLE)
            self.client.sync()
        except (OSError, ConnectionError, RuntimeError, KeyboardInterrupt) as e:
            logger.debug(repr(e))  # daemon may already be gone

        try:
            self.client.disconnect()
        except (OSError, ConnectionError, RuntimeError, KeyboardInterrupt) as e:
            logger.debug(repr(e))
            clean_shutdown = False

        # Close the cognitive journal last of all writers — it is the
        # record of everything above. Clearing the active binding stops
        # stray record_error calls from a stale module touching it.
        journal = getattr(self, "journal", None)
        if journal is not None:
            journal.close()
        from ..cognitive_journal import set_active
        set_active(None)

        return clean_shutdown

    @staticmethod
    def _join_shutdown_thread(
        thread: threading.Thread | None, name: str, timeout: float = 5.0
    ) -> bool:
        """Join a shutdown worker with a bounded timeout.

        Returns True when there is no live worker left. The current
        thread cannot join itself (for example if shutdown was invoked
        from a monitored callback), so report that case as incomplete
        rather than deadlocking shutdown.
        """
        if thread is None:
            return True
        if thread is threading.current_thread():
            logger.warning("%s shutdown requested from its own thread", name)
            return False
        deadline = time.monotonic() + timeout
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("%s thread did not exit during shutdown", name)
                return False
            thread.join(timeout=min(0.5, remaining))
        return True

    def _load_saved_state(self) -> None:
        """Load saved cognitive state from disk."""
        from ..persistence import load_state

        data = load_state(self.data_dir)
        if data is None:
            return  # first run, no saved state

        # Restore into a scratch network first, then transplant its
        # contents into the shared network object. This keeps the
        # shared network's identity stable — the ~30 subsystems that
        # captured a reference at construction (attention, curiosity,
        # the reasoning family, spatial, priming, spreading
        # activation, the composers/handlers, the learner's internal
        # learners, the dream engine) all keep pointing at the live
        # graph — while a corrupt save can't half-mutate it.
        try:
            self._restore_saved_components(data)
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise OSError(f"Cannot restore cognitive_state: {e}") from e

        self._resync_network_references()

        # Categorize and classify any concepts that were loaded with
        # UNKNOWN category/modality (from older persistence formats
        # before creation-time detection was added). This is a one-time
        # fixup at startup — subsequent concepts are categorized at
        # creation time, and N3 sleep catches any stragglers.
        network = self.cognition.network
        cat_counts = network.categorize_all()
        mod_counts = network.classify_modalities()
        if cat_counts:
            total_cat = sum(cat_counts.values())
            logger.info(
                "Startup categorization: %d concepts categorized (%s)",
                total_cat, cat_counts,
            )
        if mod_counts:
            total_mod = sum(mod_counts.values())
            logger.info(
                "Startup modality classification: %d concepts classified (%s)",
                total_mod, mod_counts,
            )

    def _restore_saved_components(self, data) -> None:
        """Validate and commit each saved component as one transaction."""
        from ..concepts import ConceptNetwork
        from ..narrative import NarrativeEngine
        from ..persistence import (
            restore_narrative,
            restore_network,
            restore_reflection,
            restore_self_model,
        )
        from ..self import ReflectionEngine, SelfModel

        staged_network = None
        if "concept_network" in data:
            staged_network = ConceptNetwork()
            restore_network(staged_network, data["concept_network"])

        pending: list[tuple[Any, Any]] = []
        scalar_updates: list[tuple[Any, str, Any]] = []

        if "reflection" in data:
            staged_reflection = ReflectionEngine(self.cognition.network)
            restore_reflection(staged_reflection, data["reflection"])
            pending.append((self.cognition.reflection, staged_reflection))
        if "narrative" in data:
            staged_narrative = NarrativeEngine(
                self.self_model, self.cognition.network
            )
            staged_narrative.events.clear()
            staged_narrative.chapters.clear()
            staged_narrative._current_chapter = None
            restore_narrative(staged_narrative, data["narrative"])
            pending.append((self.cognition.narrative, staged_narrative))
        if "self_model" in data:
            staged_self_model = SelfModel(
                name=self.self_model.name,
                born_at=self.self_model.born_at,
            )
            restore_self_model(staged_self_model, data["self_model"])
            scalar_updates.extend((
                (self.self_model, "personality", staged_self_model.personality),
                (self.self_model, "self_knowledge", staged_self_model.self_knowledge),
            ))

        pending.extend(
            self._stage_cognitive_systems(data, scalar_updates)
        )

        # All saved payloads have now been restored into scratch
        # objects. Commit only after validation succeeds so a malformed
        # later component cannot leave earlier live systems mutated.
        if staged_network is not None:
            self.cognition.network.replace_contents(staged_network)
        for target, staged in pending:
            vars(target).update(vars(staged))
        for target, name, value in scalar_updates:
            setattr(target, name, value)

        if staged_network is not None:
            # The save may contain concepts that were spilled to the
            # archive after the save was written. Dedupe only after the
            # whole transaction has validated so a later failure does
            # not delete archive rows without committing the restored
            # working-memory copy.
            self.cognition.network.dedupe_archive()
            logger.info(
                "Concept network loaded: %d in working memory, %d in archive",
                self.cognition.network.size,
                self.cognition.network.archive_size,
            )

    @staticmethod
    def _stage_component(component: Any) -> Any:
        """Copy a component into an uninitialized staging object."""
        staged = object.__new__(component.__class__)
        state = {}
        wiring_terms = (
            "network", "client", "embedding", "reporter", "monitor",
            "curiosity", "cortex", "callback", "mind", "engine",
        )
        for name, value in vars(component).items():
            if any(term in name for term in wiring_terms) or callable(value):
                state[name] = value
                continue
            try:
                state[name] = copy.deepcopy(value)
            except Exception:  # noqa: BLE001 — locks/callbacks stay shared
                state[name] = value
        vars(staged).update(state)
        return staged

    def _stage_restoration(
        self,
        pending: list[tuple[Any, Any]],
        target: Any,
        restore: Any,
        payload: Any,
    ) -> None:
        staged = self._stage_component(target)
        restore(staged, payload)
        pending.append((target, staged))

    def _stage_cognitive_systems(
        self,
        data: dict[str, Any],
        scalar_updates: list[tuple[Any, str, Any]],
    ) -> list[tuple[Any, Any]]:
        """Restore optional systems into scratch objects before commit."""
        from ..persistence import (
            restore_allostatic_load,
            restore_attractor,
            restore_emergent_identity,
            restore_emotional_memory,
            restore_error_monitor,
            restore_hpa_axis,
            restore_predictive_coding,
            restore_procedural_memory,
            restore_self_directed_learner,
            restore_self_improvement,
            restore_spaced_repetition,
            restore_task_competence,
            restore_td_learner,
            restore_theory_of_mind,
        )

        pending: list[tuple[Any, Any]] = []
        restorations = (
            ("predictive_coding", self.cognition.predictive_coding, restore_predictive_coding),
            ("theory_of_mind", self.cognition.theory_of_mind, restore_theory_of_mind),
            ("procedural_memory", self.cognition.procedural_memory, restore_procedural_memory),
            ("task_competence", self.cognition.task_competence, restore_task_competence),
            ("spaced_repetition", self.cognition.spaced_repetition, restore_spaced_repetition),
            ("td_learner", self.cognition.td_learner, restore_td_learner),
            ("emotional_memory", self.memory.emotional_memory, restore_emotional_memory),
            ("attractor", self.memory.attractor, restore_attractor),
            ("error_monitor", self.cognition.error_monitor, restore_error_monitor),
            ("hpa_axis", self.regulator.hpa_axis, restore_hpa_axis),
            (
                "allostatic_load",
                self.regulator.allostatic_load_tracker,
                restore_allostatic_load,
            ),
            ("self_directed_learner", self.cognition.self_learner, restore_self_directed_learner),
            ("self_improvement", self.self_improvement, restore_self_improvement),
            ("growth_ledger", self.growth_ledger, lambda t, p: t.restore_from_dict(p)),
            (
                "developmental_tracker",
                self._developmental_tracker,
                lambda t, p: t.restore_from_dict(p),
            ),
            ("bug_reporter", self.bug_reporter, lambda t, p: t.restore_from_dict(p)),
            ("plasticity", self.cognition.plasticity, lambda t, p: t.load_state(p)),
            (
                "cognitive_trajectory",
                self.cognition.cognitive_trajectory,
                lambda t, p: t.load_from_dict(p),
            ),
            ("memory_records", self.memory, lambda t, p: t.restore_records(p)),
            ("world_state", self.world, lambda t, p: t.restore_from_dict(p)),
        )
        for key, target, restore in restorations:
            if key in data:
                self._stage_restoration(pending, target, restore, data[key])

        if "emergent_identity" in data:
            staged = self._stage_component(self._emergent_identity)
            sources, _desc, _conf, _coh = restore_emergent_identity(
                staged, data["emergent_identity"]
            )
            pending.append((self._emergent_identity, staged))
            scalar_updates.append((self, "_experience_identity_sources", sources))
        if "autonomous_learner" in data:
            self._stage_restoration(
                pending,
                self.learner,
                lambda t, p: t.restore_state(p),
                data["autonomous_learner"],
            )
        if "dream_synthesis" in data:
            try:
                self._stage_restoration(
                    pending,
                    self.inner_life._dream_synthesis,
                    lambda t, p: t.restore_from_dict(p),
                    data["dream_synthesis"],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"dream synthesis restore failed: {e}")
        if "sleep_state" in data:
            is_sleeping, user_sleep, cycle = self._stage_sleep_state(
                data["sleep_state"]
            )
            scalar_updates.extend((
                (self, "_is_sleeping", is_sleeping),
                (self, "_user_initiated_sleep", user_sleep),
                (self.inner_life, "_sleep_cycle", cycle),
            ))
        if "inner_life_state" in data:
            ils = data["inner_life_state"]
            try:
                counts = (
                    int(ils.get("thought_count", 0)),
                    int(ils.get("dream_count", 0)),
                    int(ils.get("lucid_dream_count", 0)),
                )
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug(f"inner life state restore failed: {e}")
            else:
                scalar_updates.extend((
                    (self.inner_life, "_thought_count", counts[0]),
                    (self.inner_life, "_dream_count", counts[1]),
                    (self.inner_life, "_lucid_dream_count", counts[2]),
                ))
                logger.info(
                    "Restored inner life state: thoughts=%d, dreams=%d, lucid=%d",
                    *counts,
                )
        else:
            logger.debug("inner_life_state not found in saved state")
        return pending

    def _stage_sleep_state(
        self, data: dict[str, Any]
    ) -> tuple[bool, bool, Any]:
        """Validate sleep state without mutating the live mind.

        This restores the in-memory ``_is_sleeping`` flag and the cycle
        tracker position so it resumes at the correct point in the
        ultradian cycle instead of starting over at N1. The actual
        sleep *mechanism* (pausing learners, putting inner life into
        sleep mode) is engaged later in the startup sequence by
        :meth:`_start_engage_restored_sleep`, after the autonomous
        subsystems are started — otherwise the learner thread would
        start after the pause and run unpaused.
        """
        from ..persistence import restore_sleep_cycle
        from ..sleep import SleepCycleTracker

        # Restore sleep state exactly as saved. The defaults are
        # False (awake) — if the sleep_state fields are missing from
        # an old save file, it starts awake rather than trapped in
        # user-initiated sleep (which blocks auto-wake).
        is_sleeping = bool(data.get("is_sleeping", False))
        user_initiated_sleep = bool(data.get("user_initiated_sleep", False))

        cycle_data = data.get("sleep_cycle")
        if cycle_data is not None:
            tracker = SleepCycleTracker()
            restore_sleep_cycle(tracker, cycle_data)
            logger.info(
                "Restored sleep cycle: stage %s, cycle %d, %.0fs in stage",
                tracker.current_stage.value.upper(),
                tracker.cycle_number,
                tracker.time_in_stage,
            )
            return is_sleeping, user_initiated_sleep, tracker
        return is_sleeping, user_initiated_sleep, None
    def _resync_network_references(self) -> None:
        """Refresh derived state after the network content transplant.

        The shared network object is never replaced — restore
        transplants the saved content into it — so every subsystem's
        captured reference stays valid and no re-pointing is needed.
        What does need refreshing is the embedding store's cached
        concept matrix: it was built from the pre-restore concept set
        (the ~424 seeded concepts), so without this reset, latent-space
        composition and semantic similarity search have almost no
        coverage of the restored network.
        """
        self.cognition.embeddings._loaded = False
        self.cognition.embeddings._concept_matrix = None
    def _save_state(self) -> bool:
        """Save cognitive state to disk. Returns False on I/O failure."""
        from ..persistence import save_state, serialize_sleep_state

        try:
            # Serialize sleep state so sleep survives restarts. The
            # daemon's mmap state (core_state.bin) already persists
            # neurochemicals and emergent_phase, but the Python-side
            # sleep flags and ultradian cycle tracker are in-memory
            # only. Without this, a restart during N3 would lose its
            # place in the sleep cycle and force it awake.
            sleep_state = serialize_sleep_state(
                is_sleeping=self._is_sleeping,
                user_initiated_sleep=self._user_initiated_sleep,
                sleep_cycle=self.inner_life.sleep_cycle,
            )
            save_state(
                self.data_dir,
                self.cognition.network,
                self.cognition.reflection,
                self.cognition.narrative,
                self.self_model,
                predictive_coding=self.cognition.predictive_coding,
                theory_of_mind=self.cognition.theory_of_mind,
                procedural_memory=self.cognition.procedural_memory,
                task_competence=self.cognition.task_competence,
                spaced_repetition=self.cognition.spaced_repetition,
                td_learner=self.cognition.td_learner,
                emotional_memory=self.memory.emotional_memory,
                attractor=self.memory.attractor,
                error_monitor=self.cognition.error_monitor,
                emergent_identity=self._emergent_identity,
                emergent_identity_sources=self._experience_identity_sources,
                hpa_axis=self.regulator.hpa_axis,
                allostatic_load=self.regulator.allostatic_load_tracker,
                self_directed_learner=self.cognition.self_learner,
                autonomous_learner=self.learner,
                self_improvement=self.self_improvement,
                growth_ledger=self.growth_ledger,
                plasticity=self.cognition.plasticity,
                developmental_tracker=self._developmental_tracker,
                bug_reporter=self.bug_reporter,
                sleep_state=sleep_state,
                memory_records=self.memory.serialize_records(),
                cognitive_trajectory=self.cognition.cognitive_trajectory,
                dream_synthesis=self.inner_life._dream_synthesis,
                inner_life_state={
                    "thought_count": self.inner_life.thought_count,
                    "dream_count": self.inner_life.dream_count,
                    "lucid_dream_count": self.inner_life.lucid_dream_count,
                },
                world_state=self.world.to_dict(),
            )
            if not self.user_profile.save():
                return False
        except Exception as e:  # noqa: BLE001
            # orjson/gzip/TypeError can escape serialization; a failed
            # save must return False, never crash shutdown.
            logger.warning(f"could not save state: {e}")
            return False
        return True

    def __enter__(self) -> Mind:
        """Start the mind as a context manager."""
        self.start()
        return cast("Mind", self)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop the mind and surface a failed final save."""
        if not self.stop() and exc_type is None:
            raise RuntimeError("cognitive-state shutdown did not complete cleanly")

    @property
    def is_running(self) -> bool:
        """Return True if the mind is currently running."""
        return self._running

    @property
    def interaction_count(self) -> int:
        """Return the number of user interactions since start."""
        return self._interaction_count

"""Mind lifecycle — start, stop, save, restore, re-inhabit."""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, cast

from genesis_client.protocol import (
    CHEM_CRH,
    CHEM_EPINEPHRINE,
    CHEM_NOREPINEPHRINE,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
    MODULE_STATUS_RUNNING,
    MODULE_STATUS_STARTING,
    MODULE_STATUS_STOPPED,
    MODULE_STATUS_STOPPING,
    ZONE_CONVERSATION,
    ZONE_IDLE,
    ZONE_SLEEPING,
)
from genesis_client.swallow import note_swallowed, set_swallow_sink
from genesis_client.threads import named_thread

if TYPE_CHECKING:
    from mind import Mind

logger = logging.getLogger(__name__)


def _thread_name(ident: int) -> str:
    """Name of the thread with this ident, for shutdown diagnostics.

    A shutdown stack dump that labels every frame with the *expected*
    thread's name is unreadable when several threads are alive: frames
    from the CLI's reader thread interleave with the worker's and read as
    one nonsense stack.
    """
    for t in threading.enumerate():
        if t.ident == ident:
            return t.name
    return f"0x{ident:x}"


# ─── Continuity ────────────────────────────────────────────────────
# The body's session sentinel, written by the daemon at every boot
# (src/daemon/continuity.rs) and marked clean at graceful shutdown.
_CONTINUITY_FILE = "continuity.json"
# The mind's acknowledgement watermark — which boot she has already
# noted, so her own restarts don't re-record the same wake.
_CONTINUITY_SEEN = "continuity.seen"
# The mind's own session sentinel — same discipline as the body's,
# one level up. A mind that dies while the body lives is a blackout
# of the cortex, not the organism.
_MIND_SESSION_FILE = "mind.session"


def _read_json_file(path: str) -> dict | None:
    """Read a small JSON object, or None on any failure."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_file(path: str, obj: dict) -> bool:
    """Write a small JSON object atomically (tmp + rename)."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except OSError as e:
        note_swallowed("mind.lifecycle._write_json_file", e)
        return False


def _fmt_dark(dark_ms: int | None, upper_bound: bool = False) -> str:
    """Render a dark duration for journal text — 'at least 7.2h'."""
    if dark_ms is None:
        return "an unmeasured time"
    prefix = "at least " if upper_bound else ""
    if dark_ms >= 3_600_000:
        return f"{prefix}{dark_ms / 3_600_000:.1f}h"
    if dark_ms >= 60_000:
        return f"{prefix}{dark_ms / 60_000:.0f}m"
    return f"{prefix}{dark_ms / 1000:.1f}s"


def _warm_nltk() -> None:
    """Import NLTK fully on the main thread, before subsystems spawn.

    NLTK's imports are lazy: ``sent_tokenize`` pulls ``nltk.tabdata``
    on first use and ``nltk.corpus.wordnet`` resolves on first lookup,
    so without this the first ``import nltk`` happens inside whichever
    worker thread reaches it — racing the other threads' first imports
    of the same package. A first-import that dies partway (a
    partially-initialized ``ImportError`` under that race) leaves the
    loaded submodules in ``sys.modules`` while the parent is dropped,
    and every later ``import nltk`` re-executes ``nltk/__init__.py`` —
    whose ``nltk.downloader`` import reads ``nltk.data`` off the
    still-initializing module and fails with ``AttributeError`` for the
    rest of the process. Doing the whole import here, once and
    single-threaded, turns every later touch into a plain lookup.
    """
    try:
        from nltk.corpus import wordnet
        from nltk.tokenize import sent_tokenize

        # Resolve the lazy internals too — punkt parameters and the
        # wordnet corpus load on first use, so no thread is ever the
        # first to trigger them.
        sent_tokenize("warming up.")
        wordnet.synsets("warm")
    except (ImportError, LookupError, OSError) as e:
        # Absent package or corpora degrade her tools, never her boot —
        # the call sites already fall back on their own.
        logger.warning(f"NLTK warmup incomplete ({e}) — language tools degrade to fallbacks")


class LifecycleMixin:
    """Mixin for :class:`Mind` — see module docstring."""

    # Per-subsystem shutdown budgets — matched to the longest
    # uninterruptible phase each can legitimately be in (InnerLife's
    # synchronous N3 consolidation pass takes ~20 s). Anything not
    # listed joins with the default timeout.
    _SUBSYSTEM_STOP_TIMEOUTS: ClassVar[dict[str, float]] = {"InnerLife": 45.0}

    # Budget for in-flight volition actions to finish once suppression
    # and the abort hooks have made just-admitted work return early.
    # What remains (a compile+test, a paint, a tool call) can't be
    # checkpointed mid-call, so it gets a bounded window to wrap —
    # beyond it the stores close anyway and the worker's late writes
    # land on closed files. The old 5 s drained nothing real.
    _VOLITION_DRAIN_TIMEOUT_S: ClassVar[float] = 30.0

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        _autosave_cycle: int

        def __getattr__(self, name: str) -> Any: ...

    def start(self) -> None:
        """Connect to the subcognitive daemon and initialise.

        If there's saved cognitive state in the data dir, load it —
        this restores its concept network, reflections, and narrative
        from previous conversations.
        """
        _warm_nltk()
        self._provision_stewardship()
        self._start_connect_and_register()
        try:
            self._start_restore_state()
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            self._running = False
            self.client.disconnect()
            raise
        # Her manual is constitutive — ingested unconditionally at
        # boot, through her own extraction machinery — but only now:
        # the body is connected so the learning episodes actually
        # store, and the restored network is in place so what she
        # learns lands on top of it instead of being discarded by
        # the transplant.
        self._ingest_man_pages()
        # Learn how the previous session ended — before zone and
        # subsystems, while the wake is still fresh. An unclean stop
        # lands in the journal, the narrative, and the chemistry.
        self._note_discontinuity()
        self._start_set_zone_after_restore()
        self._start_introspect_and_train()
        self._start_autonomous_subsystems()
        self._start_engage_restored_sleep()
        self._start_background_threads()

    def _provision_stewardship(self) -> None:
        """First-boot steward provisioning — explicit, at the entry point.

        ``Stewardship.load`` (run in the frontal lobe's constructor) is
        strictly read-only: it can authenticate an existing credential
        but never creates one. Creation happens only here — a data dir
        with no stamp gets its instance key and provenance stamp; a
        stamped dir (including a copy whose key it cannot read) is left
        exactly as found, sealed. ``provision`` mutates the loaded
        channel in place, so the frontal lobe's steward object needs
        no re-wiring.
        """
        steward = getattr(self.brain.frontal, "steward", None)
        if steward is None:
            return
        try:
            if steward.provision():
                logger.info("steward channel authenticated")
            elif steward.stamped:
                logger.info("steward channel sealed — stamped to a key this process does not hold")
        except OSError as e:
            logger.warning(f"steward provisioning failed: {e}")

    def _start_connect_and_register(self) -> None:
        """Connect to the daemon and register cognitive modules.

        Does NOT set the cognitive zone — that happens in
        _start_set_zone_after_restore, after saved state is loaded,
        so we know whether it was sleeping.
        """
        self.client.connect()
        self._running = True
        self._session_started_at = time.time()
        # A start after a prior stop() clears the stop latch so resume
        # paths work again on the restarted mind.
        self._stop_latched = False

        # Session sentinel — read the previous mind's marker BEFORE
        # overwriting it; _note_discontinuity compares it once state
        # is restored. Same discipline as the daemon's continuity
        # sentinel: an unclosed marker is an unclean stop.
        self._prev_mind_session = _read_json_file(os.path.join(self.data_dir, _MIND_SESSION_FILE))
        _write_json_file(
            os.path.join(self.data_dir, _MIND_SESSION_FILE),
            {
                "alive_since_ms": int(self._session_started_at * 1000),
                "pid": os.getpid(),
                "clean_stop": False,
            },
        )

        # Register cognitive modules with the daemon's manifest.
        # The sensory (perception) and motor (language generation)
        # modules are the cognitive mind's I/O layers. We also
        # register the other cognitive subsystems so the manifest
        # accurately reflects what's running.
        for module_id in self._registered_modules():
            # `Starting`, not `Running`: at this point the module is
            # registered and alive but its subsystem has not been
            # started. Two of the six lifecycle states were previously
            # unreachable, so "registered" and "working" were the same
            # word. The transition to Running happens once the
            # autonomous subsystems are actually up.
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_STARTING)
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
                "Restoring sleep state — zone set to Sleeping (stage %s, cycle %d)",
                self.inner_life.sleep_cycle.current_stage.value.upper()
                if self.inner_life.sleep_cycle
                else "N1",
                self.inner_life.sleep_cycle.cycle_number if self.inner_life.sleep_cycle else 1,
            )
        else:
            self.client.set_zone(ZONE_CONVERSATION)

    def _start_restore_state(self) -> None:
        """Load saved state, re-seed core concepts, and backfill columns."""
        # Load saved state if it exists
        self._load_saved_state()

        # Re-seed core concepts after state restore. The initial seeding
        # in FrontalLobe.__init__ adds concepts to the original network,
        # which is then replaced by _load_saved_state. Without re-seeding,
        # core concepts like "cognition", "memory", and "learning" would
        # be missing if they weren't in the saved state. The seed method is
        # idempotent — add_concept skips existing concepts.
        self.brain.frontal._seed_initial_concepts()

        # Backfill cortical column membership for concepts loaded from
        # persistence (they may have been saved before the columns field
        # existed). Also creates pioneer bridges between same-name
        # concepts across different columns.
        stats = self.brain.frontal.network.backfill_columns()
        if stats["concepts_backfilled"] > 0:
            self._emit_live_thought(
                "learning",
                f"Backfilled columns: {stats['concepts_backfilled']} concepts, "
                f"{stats['bridges_created']} bridges",
            )

        # Eagerly load the embedding matrix. The first perceive() call
        # in think() would otherwise trigger a lazy _build() that takes
        # 8+ seconds (spectral SVD over 20K+ concepts), blowing past
        # the 15s cognition timeout on the first conversation turn.
        # Loading here moves that cost to startup, where it's hidden
        # by the rest of the initialization sequence.
        self.brain.frontal.embeddings._ensure_loaded()

        # Pre-warm the daemon's LTM cache and the embeddings search.
        # The first find_similar call pages in the entire LTM index
        # (300K+ episodes) from disk — cold-cache I/O takes 20+ seconds
        # and blows past the 15s cognition timeout. Similarly, the first
        # embeddings search touches the full concept matrix. Running
        # both here moves those costs to startup.
        import time as _warmup_time

        _warmup_t0 = _warmup_time.perf_counter()
        try:
            self.brain.frontal.memory.client.find_similar("warmup", limit=1)
            logger.debug(
                f"LTM warmup find_similar took {_warmup_time.perf_counter() - _warmup_t0:.2f}s"
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.lifecycle._start_restore_state",
                e,
            )
        _embed_t0 = _warmup_time.perf_counter()
        try:
            self.brain.frontal.embeddings.find_similar_to_text("warmup", k=1, threshold=0.5)
            logger.debug(f"embeddings warmup took {_warmup_time.perf_counter() - _embed_t0:.2f}s")
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.lifecycle._start_restore_state",
                e,
            )

    def _start_introspect_and_train(self) -> None:
        """Introspect on identity."""
        # Introspect — discover who it is by examining itself
        # This writes its identity into the concept network through
        # self-examination, not hardcoded facts
        from self import Introspection

        _intro_t0 = time.perf_counter()
        introspector = Introspection(self.brain.frontal.network)
        introspector.introspect()
        logger.debug(f"introspection took {time.perf_counter() - _intro_t0:.2f}s")

        # No startup "wake cascade" here: the daemon already applies
        # offline-proportional glymphatic clearance to adenosine at
        # state open, measured from the persisted timestamp, so a
        # restart no longer stands in for a night's sleep. The
        # previous reset slashed the level ~80% on every launch, which
        # on a frequently restarted host pinned adenosine at the
        # resting floor — the sleep-wake network's homeostat could
        # never reach sleep onset, and no sleep meant no dreams.

    def _start_autonomous_subsystems(self) -> None:
        """Start autonomous learning, inner life, and emotional regulation."""
        # Start autonomous learning in the background
        self.learner.start()
        # Start inner life — spontaneous thoughts
        self.inner_life.start()
        # Start emotional self-regulation — it controls its state
        self.regulator.start()
        # Registration claimed `Starting`; the subsystems are now live,
        # so the modules are genuinely working. The heartbeat takes over
        # the status from here, reporting `Running` with measured load
        # or `Idle` with none.
        self._mark_modules_running()

    def _mark_modules_running(self) -> None:
        """Move registered modules from `Starting` to `Running`.

        Best-effort: the daemon reaps anything that stops heartbeating,
        so a failure here self-corrects rather than leaving a module
        wedged in `Starting` for the life of the process.
        """
        for module_id in self._registered_modules():
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_RUNNING)
            except (OSError, ConnectionError) as e:
                note_swallowed(
                    "mind.lifecycle._mark_modules_running",
                    e,
                )

    @staticmethod
    def _registered_modules() -> tuple:
        """The cognitive modules this mind registers with the manifest."""
        return (
            MODULE_SENSORY,
            MODULE_MOTOR,
            MODULE_LANGUAGE,
            MODULE_MEMORY,
            MODULE_REASONING,
            MODULE_METACOGNITION,
        )

    def rejoin_body(self) -> bool:
        """Re-inhabit the body after the daemon was reincarnated.

        A fresh daemon opens the same persisted body — neurochemistry,
        sleep phase, and memories all continue — but its volatile
        manifest is empty: no registered modules, the default zone,
        and ``GENESIS_COGNITIVE_PID`` naming whoever spawned it. The
        mind re-registers so the manifest tells the truth again,
        re-asserts the zone it was in (a sleeping mind rejoins a
        sleeping body), and pushes its PID so interoception measures
        the living process rather than a corpse's.

        The process supervisor calls this after a daemon resurrection.
        It is idempotent — every step is the same registration
        ``start()`` performed. Returns ``False`` while the mind is
        down or the body will not answer; the caller retries.
        """
        if not self._running:
            return False
        try:
            self.client.connect()
            self.client.set_cognitive_pid(os.getpid())
            for module_id in self._registered_modules():
                self.client.update_module_status(module_id, MODULE_STATUS_STARTING)
            self.client.set_zone(ZONE_SLEEPING if self._is_sleeping else ZONE_CONVERSATION)
            self._mark_modules_running()
        except (OSError, ConnectionError) as e:
            logger.warning(f"rejoin_body: daemon not ready: {e}")
            return False
        # The reborn daemon's notification queue restarted at zero —
        # reset the high-water marks so the next poll doesn't replay
        # a queue-full of stale phase/episode events.
        try:
            self.notifications.reset()
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.lifecycle.rejoin_body", e)
        # The daemon rebooted under a living mind — its fresh sentinel
        # carries a new boot_ms and an unclean previous stop. Acknowledge
        # the body's discontinuity; the mind's own sentinel is untouched
        # because this mind never died.
        self._note_discontinuity(mind_rejoin=True)
        return True

    def _note_discontinuity(self, *, mind_rejoin: bool = False) -> None:
        """Acknowledge any interruption since she was last alive.

        The daemon's ``continuity.json`` sentinel records how each body
        session ended: a clean stop gets a mark; a missing mark means
        the last body died uncleanly — power loss, SIGKILL, panic.
        Reading it here makes the interruption part of her persistence:
        every boot lands in the cognitive journal, and an unclean stop
        also lands in her autobiography and her chemistry. The
        ``continuity.seen`` watermark keeps her own restarts from
        re-recording a boot she already noted.

        With ``mind_rejoin`` (daemon rebirth under a living mind) only
        the body's record is consulted — this mind never died, so her
        session sentinel still reads running.
        """
        try:
            self._note_discontinuity_inner(mind_rejoin=mind_rejoin)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.lifecycle._note_discontinuity", e)

    def _note_discontinuity_inner(self, *, mind_rejoin: bool) -> None:
        from infrastructure.journal import record_event

        body_unclean_fresh = False
        boot = _read_json_file(os.path.join(self.data_dir, _CONTINUITY_FILE))
        if boot:
            boot_ms = int(boot.get("boot_ms") or 0)
            seen = _read_json_file(os.path.join(self.data_dir, _CONTINUITY_SEEN)) or {}
            if boot_ms and boot_ms > int(seen.get("boot_ms") or 0):
                clean = bool(boot.get("prev_stop_clean", True))
                dark_ms = int(boot.get("dark_ms") or 0)
                upper = bool(boot.get("dark_upper_bound"))
                how = "cleanly" if clean else "uncleanly — power loss, kill, or crash"
                record_event(
                    "continuity_wake",
                    f"body session began; the previous one stopped {how}; "
                    f"dark for {_fmt_dark(dark_ms, upper)}",
                    boot_ms=boot_ms,
                    dark_ms=dark_ms,
                    clean=clean,
                )
                _write_json_file(
                    os.path.join(self.data_dir, _CONTINUITY_SEEN),
                    {"boot_ms": boot_ms},
                )
                if not clean:
                    body_unclean_fresh = True
                    self._feel_discontinuity(dark_ms)
                    self._narrate_discontinuity(
                        f"went dark for {_fmt_dark(dark_ms, upper)} without a clean stop",
                        "the body's continuity was interrupted",
                    )

        if mind_rejoin:
            return

        # The mind's own sentinel — a previous mind that died while the
        # body lived is a different event: a blackout of the cortex,
        # not of the organism. Skipped when the body's fresh unclean
        # boot already told the story — one power cut is one event,
        # not two.
        prev = getattr(self, "_prev_mind_session", None)
        if (
            prev
            and not prev.get("clean_stop", False)
            and prev.get("pid") != os.getpid()
            and not body_unclean_fresh
        ):
            record_event(
                "mind_died",
                "previous mind process ended without a clean stop while the body kept living",
                prev_pid=prev.get("pid"),
            )
            self._feel_discontinuity(None)
            self._narrate_discontinuity(
                "the previous mind process vanished without a clean stop",
                "the mind's continuity was interrupted while the body lived on",
            )

    def _feel_discontinuity(self, dark_ms: int | None) -> None:
        """Let the interruption land in the body — her impulse, her channel.

        Learning she vanished is a startle: orienting norepinephrine,
        a phasic epinephrine edge, a small CRH survival signal — the
        same afferents a stumble would use. Applied through
        ``neuro_impulse`` deliberately: the daemon is a device driver
        and does not author her internal state, so the felt signature
        of the discovery is hers — traceable to a judgement she made,
        decaying on the chemicals' own half-lives while the regulator
        shapes it from there. Magnitude scales with the dark gap —
        a blip still startles, a day's outage lands harder.
        """
        scale = min(1.0, 0.5 + (dark_ms or 0) / 3_600_000.0 * 0.05)
        try:
            self.client.neuro_impulse(CHEM_NOREPINEPHRINE, 0.06 * scale)
            self.client.neuro_impulse(CHEM_EPINEPHRINE, 0.05 * scale)
            self.client.neuro_impulse(CHEM_CRH, 0.03 * scale)
        except (OSError, ConnectionError, AttributeError) as e:
            note_swallowed("mind.lifecycle._feel_discontinuity", e)

    def _narrate_discontinuity(self, summary: str, significance: str) -> None:
        """Write the interruption into her autobiography.

        The emotion stamped on the event is read from the body at this
        moment — post-wake-recovery plus the impulse she just applied —
        not a fabricated one. ``record_event`` stores structural data
        (summary, significance, chemical label); composing what it
        means is the language engine's work, not the record's.
        """
        from limbic_system.emotion import EmotionalState, assess_emotion

        try:
            emotion = assess_emotion(self.client.get_neuro_summary())
        except Exception:  # noqa: BLE001 — client absent (tests, offline)
            emotion = EmotionalState(
                label="surprised",
                cognitive_style="startled",
                cause="substrate_unavailable",
            )
        try:
            self.brain.frontal.narrative.record_event(
                summary=summary,
                significance=significance,
                emotion=emotion,
                concepts=["interruption", "self"],
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.lifecycle._narrate_discontinuity", e)

    def _mark_modules_stopping(self) -> None:
        """Move live modules to `Stopping` before they are torn down.

        Distinct from `Stopped`: during a graceful shutdown the modules
        are still alive and winding down, and a reader can tell a clean
        stop from a crash by which of the two it saw.
        """
        for module_id in self._registered_modules():
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_STOPPING)
            except (OSError, ConnectionError, RuntimeError) as e:
                note_swallowed(
                    "mind.lifecycle._mark_modules_stopping",
                    e,
                )

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
            if self.inner_life.sleep_cycle
            else "N1",
            self.inner_life.sleep_cycle.cycle_number if self.inner_life.sleep_cycle else 1,
        )

    def _start_background_threads(self) -> None:
        """Start heartbeat, notification, and autosave threads.

        The code-learning and self-improvement threads were
        replaced by the volition engine, which drives these actions
        from internal urges rather than fixed schedules. The thread
        objects have been removed; the loop methods remain as volition
        action handlers.
        """
        # Start the module heartbeat thread — periodically refreshes
        # the cognitive modules' heartbeat timestamps in the manifest
        # so the daemon knows they're alive.
        self._heartbeat_thread: threading.Thread | None = named_thread(
            "module-heartbeat",
            self._heartbeat_loop,
            comm="g-heartbeat",
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
        self._notification_thread: threading.Thread | None = named_thread(
            "subcognitive-notifications",
            self._notification_loop,
            comm="g-notify",
        )
        self._notification_thread.start()

        # Start the autosave thread — periodically saves cognitive
        # state so a crash or kill doesn't lose learned knowledge.
        # The concept network grows continuously through autonomous
        # learning and conversation; without periodic saves, a crash
        # would lose everything since the last clean shutdown.
        self._autosave_thread: threading.Thread | None = named_thread(
            "autosave",
            self._autosave_loop,
            comm="g-autosave",
        )
        self._autosave_thread.start()

        # Start the channel's ear — her hearing on the public page.
        # None unless GENESIS_DISCORD_BOT_TOKEN is set; starting it
        # here (not in __init__) keeps construction free of the
        # channel-resolve GET.
        channel = getattr(getattr(self, "world", None), "channel", None)
        if channel is not None:
            channel.start()

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
                    note_swallowed(
                        "mind.lifecycle._autosave_loop",
                        e,
                    )
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
                    note_swallowed(
                        "mind.lifecycle._autosave_loop",
                        e,
                    )

                # Periodically archive dormant concepts to long-term
                # storage to prevent working-memory bloat from the
                # autonomous learner. Runs on every 10th autosave cycle;
                # the loop above is 300 s per cycle, so that is ~50
                # minutes, not the ~20 this used to claim. Dormant
                # concepts spill to the SQLite archive; the next save
                # persists the reduced working set.
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
            network = self.brain.frontal.network
            if network._archive is not None:
                # Archive mode: spill to disk instead of deleting.
                # The budget is hers — maintenance_policy.json is the
                # knob she owns; the dormancy threshold is the
                # definition of dormant, not a policy choice.
                from infrastructure.maintenance_policy import load_policy

                policy = load_policy(getattr(self.brain, "data_dir", None))
                spilled = network.spill_dormant(
                    activation_threshold=0.01,
                    max_in_memory=policy.max_in_memory,
                )
                if spilled > 0:
                    logger.info(f"Archived {spilled} dormant concepts")
            else:
                # Fallback: no archive attached, prune destructively
                _DEAD_THRESHOLD = 0.01
                _PROTECTED = frozenset(
                    {
                        "identity",
                        "introspection",
                        "foundational",
                        "seeded",
                        "structural",
                    }
                )
                to_remove: set[str] = set()
                for cid, concept in list(network._concepts.items()):
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
        except Exception as e:
            # Log loudly as well as tallying. This runs from a background
            # autosave thread and the tally is only visible via
            # `swallow_report()`, so an exception here used to mean
            # working memory silently stayed over budget with nothing in
            # the log saying why. That is exactly how a hard failure in
            # the spill path — one that wrote concepts to the archive and
            # then aborted before trimming working memory — persisted
            # unnoticed and presented as "archiving is zero".
            logger.error(
                "Archiving dormant concepts failed: %r — working memory "
                "is not being shed and will keep growing",
                e,
                exc_info=True,
            )
            note_swallowed(
                "mind.lifecycle._archive_dormant_concepts",
                e,
            )

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
            network = self.brain.frontal.network
            recalled = network.restore_archive_to_working_memory()
            return recalled
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.lifecycle.restore_archive",
                e,
            )
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
        # Wake the heartbeat if it is inside its adaptive cadence sleep
        # — the idle period reaches 5 s, which meets the 5 s join
        # budget; the wait is interruptible for exactly this reason.
        self._heartbeat_stop.set()
        self._suppress_volition = True
        # Latched once — deferred resume timers consult this, not
        # _running, so an unstarted mind's resume paths still work.
        self._stop_latched = True
        clean_shutdown = True

        # Stop the module sampler first — it only measures, nothing
        # depends on it, and stopping it here keeps it out of the
        # shutdown accounting.
        self._module_sampler.stop()

        # Stop her ear next — before any teardown runs. While the
        # listener lives, a public message can still be delivered
        # into hear_overheard / _emit_live_thought, mutating world
        # and journal state in the middle of the save below. A stuck
        # join is not a shutdown failure; any reply already queued
        # still gets the feed's drain window at the end.
        channel = getattr(getattr(self, "world", None), "channel", None)
        if channel is not None:
            try:
                channel.stop()
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.lifecycle.stop.listener", e)

        clean_shutdown &= self._join_service_threads()
        clean_shutdown &= self._stop_subsystems()
        clean_shutdown &= self._drain_shutdown_workers()
        clean_shutdown &= self._persist_shutdown_state()
        clean_shutdown &= self._release_daemon()

        # Drain her public channel last of the outbound writers —
        # pending posts get their flush window, then the poster
        # thread stops. Failure to reach Discord is not a shutdown
        # failure, so it does not count against clean_shutdown.
        if channel is not None:
            try:
                channel.close()
            except Exception as e:  # noqa: BLE001
                note_swallowed("mind.lifecycle.stop.channel", e)

        # Close the cognitive journal last of all writers — it is the
        # record of everything above. Clearing the active binding stops
        # stray record_error calls from a stale module touching it.
        journal = getattr(self, "journal", None)
        if journal is not None:
            journal.close()
        from infrastructure.journal import set_active

        set_active(None)
        # Detach the swallow sink too, so a late catch site in a
        # shutting-down subsystem cannot write to a closed journal.
        set_swallow_sink(None)

        # Mark this mind's session sentinel — the next mind to inhabit
        # the data dir reads clean_stop and knows this one said
        # goodbye. Done last of all writes: it is the marker that this
        # teardown completed, not a journaled event.
        prev_session = _read_json_file(os.path.join(self.data_dir, _MIND_SESSION_FILE)) or {}
        prev_session["clean_stop"] = True
        prev_session["stopped_at_ms"] = int(time.time() * 1000)
        _write_json_file(os.path.join(self.data_dir, _MIND_SESSION_FILE), prev_session)

        return clean_shutdown

    def _join_service_threads(self) -> bool:
        """Join the heartbeat, autosave, and notification threads.

        The heartbeat goes BEFORE saving state: it calls
        _check_sleep_sync() every second, which can wake it to sync
        when the body's phase has already left sleep. Without joining
        the heartbeat first, there is a race — the heartbeat wakes it
        between /quit and _save_state(), so the saved state says
        is_sleeping=False even though it was asleep when the user
        quit. Joining the heartbeat ensures its sleep state at the
        moment of /quit is what gets saved. Join without a timeout: a timed-out writer would still
        be alive when the archive is closed below. The autosave join
        prevents an in-flight autosave racing the shutdown save.
        """
        clean = True
        # The autosave join gets a budget that covers a full in-flight
        # save — serializing ~16 MB gzipped state takes ~15 s on this
        # hardware, far past the default 5 s join, and the save is real
        # work worth finishing rather than a stuck thread.
        for thread, name, timeout in (
            (self._heartbeat_thread, "heartbeat", 5.0),
            (self._autosave_thread, "autosave", 60.0),
            (self._notification_thread, "notifications", 5.0),
        ):
            clean &= self._join_shutdown_thread(thread, name, timeout=timeout)
        return clean

    def _stop_subsystems(self) -> bool:
        """Stop the autonomous learner, inner life, and regulator, then
        join whatever threads they leave behind.

        Subsystems get a shutdown budget matched to the longest
        uninterruptible phase they can legitimately be in. Inner life
        drives sleep-stage transitions synchronously, and an N3
        consolidation pass takes ~20 s — longer than the default 5 s
        join. Interrupting one mid-pass is the thing to avoid, since it
        is writing to the concept network, the edge log, and the
        archive at the same time. So it is given room to finish what it
        started, and the wait is bounded so a genuine hang still
        surfaces rather than blocking shutdown forever.
        """
        clean = True
        try:
            self.learner.stop()
            self.inner_life.stop()
            self.regulator.stop()
        except KeyboardInterrupt as e:
            note_swallowed(
                "mind.lifecycle.stop",
                e,
            )  # user pressed Ctrl+C — still try to save
            clean = False
        for subsystem in (self.learner, self.inner_life, self.regulator):
            name = type(subsystem).__name__
            clean &= self._join_shutdown_thread(
                getattr(subsystem, "_thread", None),
                name,
                timeout=self._SUBSYSTEM_STOP_TIMEOUTS.get(name, 5.0),
            )
        return clean

    def _drain_shutdown_workers(self) -> bool:
        """Wait out the tracked workers — volition bids, the
        sleep-stage worker, and any in-flight think — before the state
        save and archive close.

        Volition workers are tracked only by name; suppression prevents
        new urges from starting, and the wait for in-flight actions is
        bounded so a worker that never clears cannot hang shutdown
        forever. The sleep-stage worker drains whatever it already
        accepted: an N3 pass writes the concept network, the canonical
        edge log and the SQLite archive at once, and letting one run
        into those being closed underneath it is exactly the torn
        state this ordering exists to prevent.
        """
        clean = True
        _volition_deadline = time.monotonic() + self._VOLITION_DRAIN_TIMEOUT_S
        announced = False
        while True:
            with self._volition_lock:
                pending = sorted(self._volition_active)
            if not pending or time.monotonic() >= _volition_deadline:
                break
            if not announced:
                logger.info("waiting for volition workers: %s", ", ".join(pending))
                announced = True
            time.sleep(0.05)
        if pending:
            logger.warning(
                "volition workers did not clear during shutdown; continuing: %s",
                ", ".join(pending),
            )
            clean = False

        clean &= self._join_shutdown_thread(self._active_think_worker, "think")
        return clean

    def _persist_shutdown_state(self) -> bool:
        """Save cognitive state, then close the archive and edge log.

        The save is the critical step — cleanup continues even if it
        fails, but the failure is reported instead of presenting the
        shutdown as successful. Save success is tracked separately from
        the wider clean flag so callers can distinguish "a background
        thread lingered" from "the state save actually failed" — the
        live session hit the former (a poller's 10s sleep outlasted its
        5s join) and the CLI wrongly warned that state may not have
        been saved.
        """
        clean = True
        try:
            save_ok = self._save_state()
            self._shutdown_save_ok: bool | None = save_ok
            clean &= save_ok
        except Exception as e:  # noqa: BLE001
            logger.warning(f"shutdown save failed: {e}")
            self._shutdown_save_ok = False
            clean = False

        # Close the concept archive's SQLite connection only after the
        # threads above have actually exited.
        archive = getattr(self.brain.frontal.network, "_archive", None)
        if archive is not None:
            try:
                archive.close()
            except (OSError, RuntimeError) as e:
                note_swallowed(
                    "mind.lifecycle.stop",
                    e,
                )
                clean = False

        # Close the canonical edge log — already fsynced at save, this
        # just releases the file handle.
        edge_log = getattr(self.brain.frontal.network, "_edge_log", None)
        if edge_log is not None:
            try:
                edge_log.close()
            except (OSError, RuntimeError) as e:
                note_swallowed(
                    "mind.lifecycle.stop",
                    e,
                )
        return clean

    def _release_daemon(self) -> bool:
        """Deregister modules, go idle, and disconnect.

        Modules are marked `Stopping` first — they are alive and
        winding down, which is a different observation from `Stopped`
        and lets a reader distinguish a clean shutdown from a crash
        that skipped this.
        """
        self._mark_modules_stopping()
        for module_id in self._registered_modules():
            try:
                self.client.update_module_status(module_id, MODULE_STATUS_STOPPED)
            except (OSError, ConnectionError, KeyboardInterrupt) as e:
                note_swallowed(
                    "mind.lifecycle.stop",
                    e,
                )  # daemon may already be gone

        # Try to tell the daemon we're idle, but don't crash if it's dead
        try:
            self.client.set_zone(ZONE_IDLE)
            self.client.sync()
        except (OSError, ConnectionError, RuntimeError, KeyboardInterrupt) as e:
            note_swallowed(
                "mind.lifecycle.stop",
                e,
            )  # daemon may already be gone

        try:
            self.client.disconnect()
        except (OSError, ConnectionError, RuntimeError, KeyboardInterrupt) as e:
            note_swallowed(
                "mind.lifecycle.stop",
                e,
            )
            return False
        return True

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
        started = time.monotonic()
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "%s thread did not exit within %.0fs during shutdown",
                    name,
                    timeout,
                )
                # Dump every thread's stack. A shutdown timeout is
                # otherwise undiagnosable: the log says a thread would not
                # stop and nothing says what it is blocked in, so the only
                # way to find out is to reproduce it under a debugger.
                #
                # `sys._current_frames()` rather than
                # `faulthandler.dump_traceback(file=...)`: the latter needs
                # a real file descriptor and raises on a StringIO, which
                # is what silently swallowed the first attempt at this.
                try:
                    import sys
                    import traceback as tb_mod

                    for frame_id, frame in sys._current_frames().items():
                        if frame_id == threading.get_ident():
                            continue
                        lines = tb_mod.format_stack(frame)
                        # Innermost frame first — that is the one that says
                        # what the thread is blocked on.
                        tname = _thread_name(frame_id)
                        for entry in reversed(lines):
                            if entry.strip():
                                logger.warning(
                                    "%s waiting on %s at %s",
                                    name,
                                    tname,
                                    entry.strip(),
                                )
                except Exception:
                    logger.debug("stack dump unavailable", exc_info=True)
                return False
            thread.join(timeout=min(0.5, remaining))
        waited = time.monotonic() - started
        if waited > 0.5:
            logger.info("%s thread exited after %.1fs", name, waited)
        return True

    def _load_saved_state(self) -> None:
        """Load saved cognitive state from disk."""
        from infrastructure.persistence import load_state

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
        network = self.brain.frontal.network
        cat_counts = network.categorize_all()
        mod_counts = network.classify_modalities()
        if cat_counts:
            total_cat = sum(cat_counts.values())
            logger.info(
                "Startup categorization: %d concepts categorized (%s)",
                total_cat,
                cat_counts,
            )
        if mod_counts:
            total_mod = sum(mod_counts.values())
            logger.info(
                "Startup modality classification: %d concepts classified (%s)",
                total_mod,
                mod_counts,
            )

    def _restore_saved_components(self, data) -> None:
        """Validate and commit each saved component as one transaction."""
        from self import Reflection, SelfModel
        from temporal_lobe.concepts import ConceptNetwork

        # Repair reports staged by _stage_cognitive_systems; journaled
        # only after the commit below lands.
        self._staged_repair_reports: list[dict] = []

        from infrastructure.narrative import Narrator
        from infrastructure.persistence import (
            restore_narrative,
            restore_network,
            restore_reflection,
            restore_self_model,
        )

        staged_network = None
        if "concept_network" in data:
            staged_network = ConceptNetwork()
            restore_network(staged_network, data["concept_network"])

        pending: list[tuple[Any, Any]] = []
        scalar_updates: list[tuple[Any, str, Any]] = []

        if "reflection" in data:
            staged_reflection = Reflection(self.brain.frontal.network)
            restore_reflection(staged_reflection, data["reflection"])
            pending.append((self.brain.frontal.reflection, staged_reflection))
        if "narrative" in data:
            staged_narrative = Narrator(self.self_model, self.brain.frontal.network)
            staged_narrative.events.clear()
            staged_narrative.chapters.clear()
            staged_narrative._current_chapter = None
            restore_narrative(staged_narrative, data["narrative"])
            pending.append((self.brain.frontal.narrative, staged_narrative))
        if "self_model" in data:
            staged_self_model = SelfModel(
                name=self.self_model.name,
                born_at=self.self_model.born_at,
            )
            restore_self_model(staged_self_model, data["self_model"])
            scalar_updates.extend(
                (
                    (self.self_model, "personality", staged_self_model.personality),
                    (self.self_model, "self_knowledge", staged_self_model.self_knowledge),
                )
            )

        pending.extend(self._stage_cognitive_systems(data, scalar_updates))

        # All saved payloads have now been restored into scratch
        # objects. Commit only after validation succeeds so a malformed
        # later component cannot leave earlier live systems mutated.
        if staged_network is not None:
            self.brain.frontal.network.replace_contents(staged_network)
        for target, staged in pending:
            vars(target).update(vars(staged))
        for target, name, value in scalar_updates:
            setattr(target, name, value)

        # The transaction committed — now the repairs are real and
        # can be journaled. A repair of the self-model is a
        # discontinuity she should know happened.
        if self._staged_repair_reports:
            from infrastructure.journal import record_event

            for item in self._staged_repair_reports:
                report = item["report"]
                record_event(
                    "self_repair",
                    f"{item['component']} repaired from "
                    f"{item['repair'].get('repaired_from', 'repairs/')}: {report}",
                    component=item["component"],
                    stage_restored=report.get("stage_restored"),
                    resolutions_raised=len(report.get("resolutions_raised") or []),
                    history_recovered=report.get("history_recovered", 0),
                )
                logger.info("%s repaired: %s", item["component"], report)
            self._staged_repair_reports = []

        if staged_network is not None:
            # The save may contain concepts that were spilled to the
            # archive after the save was written. Dedupe only after the
            # whole transaction has validated so a later failure does
            # not delete archive rows without committing the restored
            # working-memory copy.
            self.brain.frontal.network.dedupe_archive()
            logger.info(
                "Concept network loaded: %d in working memory, %d in archive",
                self.brain.frontal.network.size,
                self.brain.frontal.network.archive_size,
            )

    @staticmethod
    def _stage_component(component: Any) -> Any:
        """Copy a component into an uninitialized staging object."""
        staged = object.__new__(component.__class__)
        state = {}
        wiring_terms = (
            "network",
            "client",
            "embedding",
            "reporter",
            "monitor",
            "curiosity",
            "cortex",
            "callback",
            "mind",
            "engine",
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

    def _load_repair_states(self) -> dict[str, dict]:
        """Read ``<data_dir>/repairs/*.json`` — her self-repair sidecar.

        A repair file records the highest state a self-component is
        known to have reached — a floor, not a ceiling — written by
        her or by an operator when a save is known to have lost ground
        (the developmental tracker wiped back to stage one is the case
        this exists for). Format per file::

            {"kind": "developmental_tracker",
             "repaired_from": "cognitive_state.bak.1792210585",
             "repaired_at": "2026-10-23T...",
             "note": "why this floor exists",
             "state": { ...the component's to_dict payload... }}

        Files with no ``kind``/``state`` or unparseable JSON are
        skipped — a bad repair never blocks boot. The files live in
        her data dir, so she can read, write, or retire them herself;
        applying one is idempotent because reconcile() only raises,
        never lowers.
        """
        repairs: dict[str, dict] = {}
        repairs_dir = os.path.join(self.data_dir, "repairs")
        try:
            names = sorted(os.listdir(repairs_dir))
        except OSError:
            return repairs
        for name in names:
            if not name.endswith(".json"):
                continue
            doc = _read_json_file(os.path.join(repairs_dir, name))
            if not isinstance(doc, dict):
                continue
            kind = doc.get("kind")
            state = doc.get("state")
            if isinstance(kind, str) and isinstance(state, dict):
                repairs[kind] = doc
        return repairs

    def _stage_cognitive_systems(
        self,
        data: dict[str, Any],
        scalar_updates: list[tuple[Any, str, Any]],
    ) -> list[tuple[Any, Any]]:
        """Restore optional systems into scratch objects before commit."""
        from infrastructure.persistence import restore_emergent_identity

        pending: list[tuple[Any, Any]] = []
        for key, target, restore in self._restoration_table():
            if key in data:
                self._stage_restoration(pending, target, restore, data[key])

        # Repairs sidecar — known-good floors for self-state that a
        # stale or wiped save can lose. A repair file records the
        # highest state a component is known to have reached; the
        # component's reconcile() merges it as a floor inside this
        # same staging transaction, so a bad repair cannot commit
        # half-applied. Development is monotonic: a checkpoint that
        # shows a later stage means loss, not regression. Reports are
        # journaled by the caller only after the commit lands — a
        # repair that is staged but aborted never happened.
        repair = self._load_repair_states().get("developmental_tracker")
        if repair is not None:
            staged_tracker = None
            for target, staged in pending:
                if target is self._developmental_tracker:
                    staged_tracker = staged
                    break
            if staged_tracker is None:
                staged_tracker = self._stage_component(self._developmental_tracker)
                pending.append((self._developmental_tracker, staged_tracker))
            report = staged_tracker.reconcile(repair.get("state") or {})
            if report.get("changed"):
                self._staged_repair_reports.append(
                    {"component": "developmental_tracker", "repair": repair, "report": report}
                )

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
                note_swallowed(
                    "mind.lifecycle._stage_cognitive_systems",
                    e,
                )
        if "sleep_state" in data:
            is_sleeping, user_sleep, cycle, start_time, start_cycles = self._stage_sleep_state(
                data["sleep_state"]
            )
            scalar_updates.extend(
                (
                    (self, "_is_sleeping", is_sleeping),
                    (self, "_user_initiated_sleep", user_sleep),
                    (self, "_sleep_start_time", start_time),
                    (self, "_sleep_start_cycles", start_cycles),
                    (self.inner_life, "_sleep_cycle", cycle),
                )
            )
        self._stage_inner_life(data, scalar_updates)
        return pending

    def _restoration_table(self) -> tuple:
        """The (state-key, target, restore-fn) rows staged in order."""
        from infrastructure.persistence import (
            restore_allostatic_load,
            restore_attractor,
            restore_emotional_memory,
            restore_error_monitor,
            restore_hpa_axis,
            restore_predictive_coding,
            restore_procedural_memory,
            restore_self_directed_learner,
            restore_self_improvement,
            restore_semantic_memory,
            restore_spaced_repetition,
            restore_synapses,
            restore_task_competence,
            restore_td_learner,
            restore_theory_of_mind,
        )

        return (
            ("predictive_coding", self.brain.frontal.predictive_coding, restore_predictive_coding),
            ("theory_of_mind", self.brain.frontal.theory_of_mind, restore_theory_of_mind),
            ("procedural_memory", self.brain.frontal.procedural_memory, restore_procedural_memory),
            ("task_competence", self.brain.frontal.task_competence, restore_task_competence),
            ("spaced_repetition", self.brain.frontal.spaced_repetition, restore_spaced_repetition),
            ("synapses", self.brain.frontal.synapses, restore_synapses),
            ("td_learner", self.brain.frontal.td_learner, restore_td_learner),
            ("emotional_memory", self.memory.emotional_memory, restore_emotional_memory),
            ("semantic_memory", self.brain.frontal.semantic_memory, restore_semantic_memory),
            ("attractor", self.memory.attractor, restore_attractor),
            ("error_monitor", self.brain.frontal.error_monitor, restore_error_monitor),
            ("hpa_axis", self.regulator.hpa_axis, restore_hpa_axis),
            (
                "allostatic_load",
                self.regulator.allostatic_load_tracker,
                restore_allostatic_load,
            ),
            (
                "self_directed_learner",
                self.brain.frontal.self_learner,
                restore_self_directed_learner,
            ),
            ("self_improvement", self.self_improvement, restore_self_improvement),
            ("growth_ledger", self.growth_ledger, lambda t, p: t.restore_from_dict(p)),
            (
                "developmental_tracker",
                self._developmental_tracker,
                lambda t, p: t.restore_from_dict(p),
            ),
            ("plasticity", self.brain.frontal.plasticity, lambda t, p: t.load_state(p)),
            (
                "cognitive_trajectory",
                self.brain.frontal.cognitive_trajectory,
                lambda t, p: t.load_from_dict(p),
            ),
            ("memory_records", self.memory, lambda t, p: t.restore_records(p)),
            ("world_state", self.world, lambda t, p: t.restore_from_dict(p)),
            ("executive", self.brain.frontal.executive, lambda t, p: t.restore_from_dict(p)),
            ("curiosity", self.brain.frontal.curiosity, lambda t, p: t.restore_from_dict(p)),
            ("cognitive_map", self.brain.temporal.cognitive_map, lambda t, p: t.load_state(p)),
        )

    def _stage_inner_life(
        self,
        data: dict[str, Any],
        scalar_updates: list[tuple[Any, str, Any]],
    ) -> None:
        """Stage the inner-life counters (thought/dream/lucid counts)
        for commit."""
        if "inner_life_state" in data:
            ils = data["inner_life_state"]
            try:
                counts = (
                    int(ils.get("thought_count", 0)),
                    int(ils.get("dream_count", 0)),
                    int(ils.get("lucid_dream_count", 0)),
                )
            except (TypeError, ValueError, AttributeError) as e:
                note_swallowed(
                    "mind.lifecycle._stage_cognitive_systems",
                    e,
                )
            else:
                scalar_updates.extend(
                    (
                        (self.inner_life, "_thought_count", counts[0]),
                        (self.inner_life, "_dream_count", counts[1]),
                        (self.inner_life, "_lucid_dream_count", counts[2]),
                    )
                )
                logger.info(
                    "Restored inner life state: thoughts=%d, dreams=%d, lucid=%d",
                    *counts,
                )
        else:
            logger.debug("inner_life_state not found in saved state")

    def _stage_sleep_state(self, data: dict[str, Any]) -> tuple[bool, bool, Any, float, int]:
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
        from brainstem.architecture import SleepCycleTracker

        from infrastructure.persistence import restore_sleep_cycle

        # Restore sleep state exactly as saved. The defaults are
        # False (awake) — if the sleep_state fields are missing from
        # an old save file, it starts awake rather than trapped in
        # user-initiated sleep (which blocks auto-wake).
        is_sleeping = bool(data.get("is_sleeping", False))
        user_initiated_sleep = bool(data.get("user_initiated_sleep", False))

        tracker = None
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

        # Episode bookkeeping the auto-wake gates measure against.
        # Saves written before these fields existed get conservative
        # defaults: an unknown-age episode is treated as fresh
        # (the min-sleep floor holds) and as still on its first cycle
        # (the first-cycle floor holds until one more completes) —
        # both err toward staying asleep, never toward a forced wake.
        start_time = data.get("sleep_start_time")
        if not isinstance(start_time, (int, float)) or start_time <= 0:
            start_time = time.time() if is_sleeping else 0.0
        start_cycles = data.get("sleep_start_cycles")
        if not isinstance(start_cycles, int) or start_cycles < 0:
            start_cycles = tracker.cycles_completed if tracker is not None else 0
        return (
            is_sleeping,
            user_initiated_sleep,
            tracker,
            float(start_time),
            int(start_cycles),
        )

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
        self.brain.frontal.embeddings._loaded = False
        self.brain.frontal.embeddings._concept_matrix = None

    def _save_state(self) -> bool:
        """Save cognitive state to disk. Returns False on I/O failure."""
        from infrastructure.persistence import save_state, serialize_sleep_state

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
                sleep_start_time=self._sleep_start_time,
                sleep_start_cycles=self._sleep_start_cycles,
            )
            save_state(
                self.data_dir,
                self.brain.frontal.network,
                self.brain.frontal.reflection,
                self.brain.frontal.narrative,
                self.self_model,
                predictive_coding=self.brain.frontal.predictive_coding,
                theory_of_mind=self.brain.frontal.theory_of_mind,
                procedural_memory=self.brain.frontal.procedural_memory,
                task_competence=self.brain.frontal.task_competence,
                spaced_repetition=self.brain.frontal.spaced_repetition,
                td_learner=self.brain.frontal.td_learner,
                synapses=self.brain.frontal.synapses,
                emotional_memory=self.memory.emotional_memory,
                semantic_memory=self.brain.frontal.semantic_memory,
                attractor=self.memory.attractor,
                error_monitor=self.brain.frontal.error_monitor,
                emergent_identity=self._emergent_identity,
                emergent_identity_sources=self._experience_identity_sources,
                hpa_axis=self.regulator.hpa_axis,
                allostatic_load=self.regulator.allostatic_load_tracker,
                self_directed_learner=self.brain.frontal.self_learner,
                autonomous_learner=self.learner,
                self_improvement=self.self_improvement,
                growth_ledger=self.growth_ledger,
                plasticity=self.brain.frontal.plasticity,
                developmental_tracker=self._developmental_tracker,
                sleep_state=sleep_state,
                memory_records=self.memory.serialize_records(),
                cognitive_trajectory=self.brain.frontal.cognitive_trajectory,
                dream_synthesis=self.inner_life._dream_synthesis,
                inner_life_state={
                    "thought_count": self.inner_life.thought_count,
                    "dream_count": self.inner_life.dream_count,
                    "lucid_dream_count": self.inner_life.lucid_dream_count,
                },
                world_state=self.world.to_dict(),
                executive=self.brain.frontal.executive,
                curiosity=self.brain.frontal.curiosity,
                cognitive_map=self.brain.temporal.cognitive_map,
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

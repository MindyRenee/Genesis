"""Mind sleep — sleep/wake/meditation and consolidation."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Generator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brainstem.architecture import SleepStage
from brainstem.thresholds import (
    AUTO_SLEEP_MIN_AWAKE,
    AUTO_SLEEP_MIN_IDLE,
    AUTO_WAKE_ADENOSINE,
    AUTO_WAKE_CORTISOL,
    DROWSINESS_ADENOSINE,
    DROWSINESS_CONFIRM_S,
    DROWSINESS_EXIT,
    WAKE_DESYNC_GRACE_SECONDS,
    WAKE_DESYNC_SETTLE_SECONDS,
    WAKE_RESCUE_ADENOSINE,
    WAKE_RESCUE_MELATONIN,
)
from parietal_lobe.situation_model import SituationModel

from commitment import CommitmentBoundary
from genesis_client.protocol import (
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_GABA,
    CHEM_HISTAMINE,
    CHEM_MELATONIN,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    CHEM_SEROTONIN,
    PHASE_NAMES,
    PHASE_NREM,
    PHASE_REM,
    ZONE_CONVERSATION,
    ZONE_REFLECTION,
    ZONE_SLEEPING,
)
from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread
from infrastructure.journal import record_event

logger = logging.getLogger(__name__)

# Seconds after wake before the learners resume — sleep inertia for
# the heavy acquisition pathways, matching the staggered volition
# suppression the Mind applies to its urges.
_LEARNER_WAKE_RESUME_DELAY_S = 60.0


class WakeOutcome(Enum):
    """What `wake()` actually did, so callers can report it honestly.

    A wake request is a goal-state request to the body: the mind's own
    `_is_sleeping` flag and the daemon's physical phase are separate
    layers, and "already awake" is only true when BOTH are awake.
    """

    WOKE = "woke"
    BODY_WAKE_PUSHED = "body_wake_pushed"
    ALREADY_AWAKE = "already_awake"
    BLOCKED = "blocked"


def _fmt_counts(counts: dict) -> str:
    """Render a stats/count mapping as 'k v, k v' for live-thought display."""

    def _v(v: object) -> object:
        return round(v, 3) if isinstance(v, float) else v

    return ", ".join(f"{k} {_v(v)}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))


class SleepMixin:
    """Mixin for :class:`Mind` — see module docstring."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
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
        from sleep.compression import SleepCompressor

        # Her own sleep work — no worker thread. The N3 stage queue
        # holds bounded steps she pumps herself, one or two per sleep
        # tick, gated by the chemistry her body reports. Leaving a
        # stage (or waking) abandons the remainder: interrupted sleep
        # consolidates less, like a real brain.
        # N3 sleep processes — parallel self-limiting generators, not a
        # task queue. Each is advanced one committed slice per credit and
        # retires itself on exhaustion; a stage that ends drops them
        # mid-yield with nothing half-applied.
        self._n3_processes: dict[str, Generator[dict, None, None]] = {}
        self._n3_rotation: deque[str] = deque()
        self._n3_credit: float = 0.0
        self._chem_read_at: float = 0.0
        self._chem_cache: dict[str, float] = {}

        try:
            self._sleep_compressor: SleepCompressor | None = SleepCompressor(
                data_dir=self.data_dir,
            )
            # Attach the holographic graph to the network so compressed
            # associations remain queryable after sleep compression
            # migrates bridge edges and LTM associations into it.
            if self._sleep_compressor is not None:
                self.brain.frontal.network.attach_holographic_graph(
                    self._sleep_compressor.holographic_graph
                )
                # Wire the holographic graph to the embedding store so
                # the experiential layer can be built from sleep-
                # discovered associations. This unifies the holographic
                # graph's T^2048 with the embedding store's latent
                # space into a single toroidal representation.
                embeddings = getattr(self.brain.frontal, "embeddings", None)
                if embeddings is not None:
                    embeddings.set_holographic_graph(self._sleep_compressor.holographic_graph)
        except Exception as e:  # noqa: BLE001
            logger.warning("Sleep compression init failed: %s", e)
            self._sleep_compressor = None

    def _init_sleep_state_tracking(self) -> None:
        """Initialize sleep, rest, meditation, and teaching state tracking.

        Sleep-state tracking — lets the CLI and the cognitive logic
        avoid duplicate sleep()/wake() calls and lets ambient/CLI
        interactions wake it if it has drifted off.

        Whether sleep was initiated by the user (via /sleep or /nap)
        or joined from the body's own crossing. Either way the body's
        phase owns the exit — the wake-desync check follows it up,
        and /wake is the explicit way out.

        Track when it last woke up, for the waking-side minimum-awake
        guard. The drowsiness announce and wake-stabilization renewals
        hold off right after a wake so residual pressure reads as
        grogginess, not a fresh signal.

        Meditation-state tracking — a lighter recovery mode than
        sleep. During meditation, learning, and code scanning
        scanning, and spontaneous thoughts are all paused. The
        neurochemistry is actively calmed (GABA/serotonin up,
        dopamine/NE/cortisol down) to allow receptor recovery
        without the full sleep cycle. Unlike sleep, there is no
        dreaming or memory consolidation — just quiet restoration.

        Last rest time — tracks when it last meditated or slept,
        for the ultradian rest cycle (BRAC, Kleitman 1963). The
        meditation urge builds from sustained_activity, which is
        time since last rest normalized over 75 minutes.

        Concept network size at last volition tick — used to measure
        concept_growth (new concepts since last tick) for the
        introspection and self-mission urges.

        Teaching-mode tracking — when the user is actively teaching
        its something, it enters a focused learning state. Bug
        scanning, autonomous learning, art, and unrelated curiosity
        questions are paused. Self-directed learning from the
        conversation stays active so it remembers what it's taught.
        """
        self._is_sleeping = False
        self._user_initiated_sleep = False
        self._sleep_start_time: float = 0.0
        self._last_wake_time: float = time.time()
        # Last wake-direction chemical cascade sent (epoch s). The
        # body's flip-flop switches on the LC population's own ~23-min
        # time constant — a still-asleep daemon phase inside this
        # window is an in-progress chemical transition, not a fault.
        # 0 means no wake attempt is outstanding.
        self._last_wake_attempt: float = 0.0
        # Last sleep-direction chemical cascade sent (epoch s).
        # Mirrors `_last_wake_attempt`: a still-awake daemon phase
        # inside the window is the body's own onset pace — the sync
        # check only flags a failed onset once the window has closed.
        self._last_sleep_attempt: float = 0.0
        # Body-side sleep trackers for the wake-desync check: the body
        # owns the transition in both directions, so while the mind is
        # asleep the daemon's phase is the truth to sync to.
        # `_body_sleep_confirmed` records that the network actually
        # reached NREM/REM during this sleep episode — only then does a
        # waking daemon phase mean the body genuinely woke (rather than
        # the onset still being in progress). `_body_awake_since` is the
        # first persistent awake-phase read after a confirmed sleep,
        # used to distinguish a real body wake from threshold flutter.
        self._body_sleep_confirmed: bool = False
        self._body_awake_since: float = 0.0
        # Whether the last wake() was an explicit user request
        # (/wake, or an interaction waking a sleeping mind). While a
        # user-wake's cascade is still in its switch window, the
        # circadian drive alone cannot pull it back under — the user
        # asked for a waking mind and melatonin pressure reads as
        # grogginess, not an instant knockout. Genuine adenosine
        # exhaustion still overrides.
        self._last_wake_user_initiated: bool = False
        self._sleep_start_cycles: int = 0
        # Previous-cycle hypnagogia read — the metronome's sleep-sync
        # pass detects the descent's rising and falling edges so the
        # learners it pauses at onset are released if the descent
        # aborts back to waking.
        self._was_hypnagogic: bool = False
        self._is_meditating = False
        # When the current meditation began, so the exit record can
        # report its duration rather than only the fact.
        self._meditation_start_time: float = 0.0
        self._last_rest_time = time.time()
        self._last_concept_count = 0
        self._is_teaching = False
        self._teaching_topic: str = ""
        # A /teach-questions session — she asked, the user is
        # answering. While it holds, background stays paused and
        # curiosity is scoped to the exchange, the way teaching mode
        # scopes it to a lesson. Ends when the queue drains, on
        # stop/clear, or when the user goes quiet for the
        # conversation focus window.
        self._answering_questions = False
        self._question_topic = ""
        # Commitment boundaries on the sleep-wake axis — the
        # digitizers that turn continuous adenosine pressure into
        # discrete transitions. Both require a sustained crossing
        # (confirm_s) so a transient spike cannot commit it, and
        # re-arm below a lower release level (deadband) so boundary
        # flicker cannot re-fire the transition. Reset on wake.
        self._drowsy_boundary = CommitmentBoundary(
            enter=DROWSINESS_ADENOSINE,
            release=DROWSINESS_EXIT,
            confirm_s=DROWSINESS_CONFIRM_S,
        )
        # Treat fresh start like a wake event so volition wake delays
        # (WAKE_VOLITION_DELAYS) apply on first run. Otherwise
        # _wake_time=0.0 bypasses all delays and urges fire immediately.
        self._wake_time = time.time()

    @property
    def is_sleeping(self) -> bool:
        """Whether it is currently in a sleep state."""
        return self._is_sleeping

    @property
    def is_meditating(self) -> bool:
        """Whether it is currently meditating."""
        return self._is_meditating

    @property
    def is_hypnagogic(self) -> bool:
        """Whether it is in the hypnagogic (sleep-onset) period.

        Hypnagogia is not a waking state and not established sleep —
        it is the transitional regime between them, N1 in the Hori/
        AASM framing: theta-dominant EEG with alpha dropout. The
        scored sleep boundary does not divide it — behavioral
        responding and awareness persist through N1 and only reliably
        cease at N2 (Ogilvie 1989; Hori et al. 1994), so both sides of
        the flip-flop crossing are the same continuous state:

        - awake side: a drowsy-phase hypnagogic transition is running
          in inner life (HypnagogicState in progress);
        - asleep side: ``_is_sleeping`` is set but the ultradian
          tracker is still in N1.
        """
        if self._is_meditating:
            return False
        if self._is_sleeping:
            cycle = self.inner_life.sleep_cycle
            stage = getattr(cycle, "current_stage", None) if cycle else None
            return stage is SleepStage.N1
        return self.inner_life.current_hypnagogic is not None

    @property
    def is_resting(self) -> bool:
        """Whether discretionary (waking-mode) cognition is paused.

        The single rest-state predicate: asleep, meditating, or in
        hypnagogia (the theta-dominant N1 descent). While resting,
        ordinary waking activity is held — volition urges, autonomous
        and self-directed learning, speech and outreach, drawing,
        self-improvement, executive intention formation, and waking
        spontaneous thought all wait for the state to resolve.

        What is NOT held: physiology (metronome body maintenance,
        neurochemistry, sensors, notifications, autosave), sleep-stage
        work (dreaming, consolidation, emotional replay), and the
        hypnagogic descent itself — those *are* the rest process.
        Interaction still wakes her through ``wake_if_asleep``; this
        gate only stops her from initiating waking work while under.
        """
        return self._is_sleeping or self._is_meditating or self.is_hypnagogic

    @property
    def thalamic_gate_closed(self) -> bool:
        """Whether the sensory relay is shut to routine input.

        Sleep closes the gate in either mode — but N1 is where the
        modes differ. Thalamic deactivation at sleep onset *precedes*
        the cortex's by minutes (Magnin et al., PNAS 2010): the relay
        quiets first and the mind keeps dreaming behind it, so a
        resident (headless) mind closes the gate the moment
        hypnagogia begins — nobody sits at a prompt to reach her.
        Interactive sessions keep the N1 exemption instead: behavioral
        responsiveness persists through sleep onset (Ogilvie 1989), so
        a user at the prompt can still reach her while she drifts.

        Salient stimuli are not covered here — loud sounds still land
        as dream residues and an explicit wake still rouses her, the
        brain's sensory *gaining* alongside the gate.
        """
        headless = getattr(self, "_headless", False)
        if self._is_sleeping:
            # Asleep: closed in established stages; headless closes N1
            # too, interactive holds it open for the prompt.
            return headless or not self.is_hypnagogic
        # Awake-side hypnagogia (a drowsy transition in flight): the
        # resident relay quiets as she starts to drift; a session's
        # stays open until the body crosses.
        return headless and self.is_hypnagogic

    @property
    def _wake_attempt_outstanding(self) -> bool:
        """Whether a chemical wake cascade is still in its switch window.

        The arousal cascade (orexin, histamine, NE) reaches the
        flip-flop's LC side, but the population itself switches on its
        own ~23-minute time constant. This brackets the window where a
        still-asleep daemon phase is a legitimate in-progress
        transition rather than a reason to re-sleep.
        """
        return (
            self._last_wake_attempt > 0.0
            and time.time() - self._last_wake_attempt < WAKE_DESYNC_SETTLE_SECONDS
        )

    def wake_if_asleep(self) -> bool:
        """Convenience hook to rouse its before an interaction.

        Returns True if it was asleep or meditating and has been woken.
        """
        if self._is_meditating:
            try:
                self.wake_from_meditation()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"wake_from_meditation failed: {e}")
            return not self._is_meditating
        if not self._is_sleeping:
            return False
        if self.is_hypnagogic:
            # N1 is hypnagogia, not established sleep — responsiveness
            # persists through it (Ogilvie 1989). Answer from inside
            # the state rather than rousing her out of it.
            return False
        try:
            # A user talking to a sleeping mind is an explicit wake
            # request — mark it user-initiated so the stabilization
            # window resists the circadian pull-back.
            self.wake(user_initiated=True)
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
            # be consolidated, and the user can't tell when it enters
            # N3 (the deep-sleep stage that does heavy consolidation).
            cycle_num = (
                self.inner_life.sleep_cycle.cycle_number if self.inner_life.sleep_cycle else 0
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
            n = self.brain.frontal.memory.consolidate_during_sleep(
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
            # plasticity, embeddings refresh, compression. There is no
            # worker — she does the work herself. The stage's agenda is
            # built here and pumped one bounded step at a time by
            # _on_sleep_tick, gated by the chemistry her body reports
            # each tick, so the pass is spread through the stage like
            # real consolidation and abandons cleanly if she wakes or
            # the stage moves on.
            if stage is SleepStage.N3:
                self._open_n3_processes()
                self._n3_credit = 0.0
            elif self._n3_processes:
                # Left N3 with processes still live — interrupted sleep
                # consolidates less. Close rather than resume stale
                # state next stage; each reopens fresh on N3 entry.
                logger.info(
                    "Sleep processes closed: %d still live as %s ended",
                    len(self._n3_processes),
                    SleepStage.N3.value,
                )
                for proc in self._n3_processes.values():
                    try:
                        proc.close()
                    except Exception as e:  # noqa: BLE001
                        note_swallowed(
                            "mind.sleep._on_sleep_stage_transition.close_process",
                            e,
                        )
                self._n3_processes.clear()
                self._n3_rotation.clear()
                self._n3_credit = 0.0

            if stage is SleepStage.REM:
                # The unconscious's bookkeeping runs on the REM side —
                # rescan the repression register so tonight's charged,
                # untouched material is on the books before the dream
                # work reads it back at wake.
                try:
                    self.psychoanalysis.update_register()
                except Exception as e:  # noqa: BLE001
                    note_swallowed("mind.sleep._on_sleep_stage_transition.rem", e)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.sleep._on_sleep_stage_transition",
                e,
            )

    def _on_sleep_tick(self, stage: SleepStage, dt: float) -> None:
        """Advance her sleep physiology, one committed slice per credit.

        Called from the inner-life loop on every dream tick while she
        sleeps — these are her own processes on her own thread of
        control, each self-limiting and interruptible at every slice
        boundary.

        The pacing is the body's own chemistry (Hobson's AIM model
        made literal): credit accrues at a rate set by the
        neurochemical state — high consolidation drive with aminergic
        silence (low norepinephrine + serotonin, the NREM signature)
        and residual homeostatic pressure (Process S, adenosine) means
        deep consolidation. Light, aminergic, or discharged sleep
        means the processes idle — stress-disrupted sleep
        consolidates less, as in a real brain.
        """
        if stage is not SleepStage.N3 or not self._n3_rotation:
            return
        chem = self._read_sleep_chemistry()
        consolidation = chem.get("consolidation", 0.5)
        aminergic = chem.get("norepinephrine", 0.0) + chem.get("serotonin", 0.0)
        pressure = chem.get("sleep_pressure", 0.0)
        # Acetylcholine marks the encoding/REM mode; deep NREM keeps it
        # low. The gate asks for the consolidation-safe profile:
        # aminergic silence, present consolidation drive, and pressure
        # still to discharge.
        gate = consolidation * max(0.0, 1.0 - min(1.0, aminergic)) * (0.25 + 0.75 * pressure)
        self._n3_credit += gate * max(0.0, dt)
        # ~1 advance per couple of seconds at full gate.
        ran = 0
        while self._n3_rotation and self._n3_credit >= 0.5 and ran < 2:
            self._n3_credit -= 0.5
            name = self._n3_rotation.popleft()
            proc = self._n3_processes.get(name)
            if proc is None:
                continue
            try:
                out = next(proc)
            except StopIteration:
                # The process finished itself — nothing is owed.
                self._n3_processes.pop(name, None)
                continue
            except Exception as e:  # noqa: BLE001
                note_swallowed(f"mind.sleep._on_sleep_tick.{name}", e)
                self._n3_processes.pop(name, None)
                continue
            self._n3_rotation.append(name)
            ran += 1
            if out:
                self._emit_live_thought("sleeping", f"Sleep {name}: {_fmt_counts(out)}")

    def _read_sleep_chemistry(self) -> dict[str, float]:
        """The chemicals that gate sleep work, read from her body.

        One ``get_state`` gives the whole picture — per-chemical
        effective levels (norepinephrine, serotonin, acetylcholine,
        GABA, adenosine, orexin, melatonin), the homeostatic sleep
        pressure (Process S — adenosine-driven, the daemon is the
        authority on it), and the body's own consolidation/encoding
        weights. Throttled: neurochemistry moves on a seconds
        timescale, so a ~2 s cache loses nothing.
        """
        now = time.monotonic()
        if self._chem_cache and now - self._chem_read_at < 2.0:
            return self._chem_cache
        chem: dict[str, float] = {}
        try:
            state = self.client.get_state()
        except Exception:  # noqa: BLE001
            self._chem_read_at = now
            self._chem_cache = chem
            return chem
        self._chem_read_at = now
        chemicals = getattr(state, "chemicals", None) or {}
        chem.update({k: float(v) for k, v in chemicals.items()})
        chem["consolidation"] = float(getattr(state, "consolidation_weight", 0.5))
        chem["encoding"] = float(getattr(state, "encoding_weight", 0.5))
        chem["sleep_pressure"] = float(getattr(state, "sleep_pressure", 0.0))
        chem["arousal"] = float(getattr(state, "arousal", 0.5))
        # Feed the body's Process S into the SHY downscaler — its
        # discharge rate follows sleep pressure (Tononi & Cirelli).
        downscaler = getattr(getattr(self, "inner_life", None), "_synaptic_downscaler", None)
        if downscaler is not None:
            downscaler.body_pressure = chem["sleep_pressure"]
        self._chem_cache = chem
        return chem

    def _maintenance_policy(self):
        """Load her maintenance policy fresh — she owns these knobs.

        ``<data_dir>/maintenance_policy.json`` is hers to tune with
        her own file tools; every sleep pass reloads it so a change
        she makes mid-night applies to the next pass. Anything wrong
        falls back to the defaults — a bad value never becomes a
        crash, and never silently erases her memory.
        """
        from infrastructure.maintenance_policy import load_policy

        return load_policy(self.data_dir)

    def _open_n3_processes(self) -> None:
        """Open the night's sleep processes — parallel physiology,
        not an agenda.

        Human N3 does not run a task list: ripple replays potentiate
        what they touch (``replay_during_n3`` in the inner-life loop),
        the SHY downscaler renorms every tick, and the housekeeping
        below runs only while the stage lasts — self-limiting
        generators advanced one committed slice at a time, in
        rotation, as chemistry credit allows. None is owed a
        completion; exhaustion retires a process and stage end drops
        the rest mid-yield. Nothing here is force-fed: each process
        is her own maintenance machinery running while her body says
        deep sleep.

        The policy is read once, at open, and journaled so the record
        says what she slept under; a policy she edits mid-night lands
        in the next N3's processes.
        """
        network = self.brain.frontal.network
        policy = self._maintenance_policy()
        self._emit_live_thought("sleeping", policy.describe())
        self._n3_processes.clear()
        self._n3_rotation.clear()

        def _once(fn: Callable[[], dict]) -> Generator[dict, None, None]:
            """A bounded one-shot action as a self-limiting process."""
            yield fn()

        self._n3_processes["categorize"] = _once(
            # Categorize and classify any concepts still UNKNOWN —
            # older persistence formats and late-arriving definitions.
            lambda: {
                **network.categorize_all(),
                **network.classify_modalities(),
            }
        )
        self._n3_processes["consolidation"] = (
            out for _phase, out in network.consolidate_during_sleep_steps(policy=policy)
        )
        self._n3_processes["discovery"] = _once(self._n3_discovery_stats)

        def _plasticity() -> dict:
            # Embedding-level Hebbian consolidation + edge-weight
            # Hebbian plasticity — the day's co-occurrences move
            # together in latent space and strengthen their links.
            out = self.brain.frontal.plasticity.apply_sleep_consolidation()
            out.update(
                self.brain.frontal.plasticity.apply_edge_plasticity(
                    learning_rate=0.02,
                    decay_rate=0.005,
                )
            )
            return out

        self._n3_processes["plasticity"] = _once(_plasticity)

        def _embeddings() -> dict:
            embeddings = self.brain.frontal.embeddings
            if not embeddings.has_embeddings:
                return {}
            # Rebuild so the day's concepts participate in discovery;
            # persist experiential vectors so Hebbian learning
            # survives restarts.
            embeddings.refresh()
            embeddings.save_experiential(self.data_dir)
            return {"refreshed": 1}

        self._n3_processes["embeddings"] = _once(_embeddings)
        self._n3_processes["topology"] = _once(
            lambda: (self.brain.frontal.topology.refresh(), {"refreshed": 1})[1]
        )

        # Compression phases as processes — VQ iterates once per slice,
        # holographize persists with its migration, LTM compacts
        # episode-by-episode. The prune step gets the memory system's
        # own retention verdict: release is her decision, the
        # compressor only carries it out.
        if self._sleep_compressor is not None:
            for name, step in self._sleep_compressor.compress_steps(
                network=network,
                embeddings=self.brain.frontal.embeddings,
                ltm_client=self.client,
                release_fn=self.brain.frontal.memory.release_verdict,
            ):
                self._n3_processes[f"compression.{name}"] = _once(step)
        else:
            logger.debug("Sleep compression skipped: compressor not initialized")

        self._n3_rotation.extend(self._n3_processes)

    def _n3_discovery_stats(self) -> dict:
        self._n3_discovery()
        return {}

    def _n3_discovery(self) -> None:
        """N3 latent-space discovery — hippocampal consolidation turns
        patterns into explicit knowledge, then cross-domain analogy and
        schema induction abstract over the result."""
        # Discover new relationships from embedding proximity.
        # This is hippocampal consolidation: patterns → explicit knowledge.
        discovered = self.brain.frontal.edge_proposer.discover_and_accept()
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
        analogies = self.brain.frontal.analogy_engine.discover_and_accept()
        if analogies > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep analogy: {analogies} cross-domain edges from structure-mapping",
            )

        # Schema induction — reify repeated relational structure into
        # schema concepts. Consolidation does not just re-sort
        # associations; it abstracts over them: K earned edges become
        # a named class with the members folded under it. This is the
        # mechanism that grows middle layers of the taxonomy.
        schemas = self.brain.frontal.schema_inducer.discover_and_accept()
        if schemas > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep schema induction: {schemas} reified schema edges",
            )

    def _check_sleep_sync(self) -> None:
        """Keep the mind's sleep state synced to the body's, and run
        the waking-side sleep signals.

        There is no autonomic sleep onset and no autonomic wake:
        transitions into and out of sleep are hers — a self-sleep urge
        self-invoking ``/sleep``, a user command, or the body's own
        sleep-wake network crossing under genuine pressure. What this
        metronome pass does is *sync* and *signal*: when the daemon's
        phase says the body has crossed into NREM/REM while the mind
        still flags awake, the mind joins (``_check_sleep_desync``);
        when the phase says the body has genuinely woken while the
        mind still sleeps, the mind follows (``_check_wake_desync``).
        The two are never in opposite sleep states — the body owns the
        transition and the mind always sleeps or wakes with it.

        While awake it also tracks the drowsiness boundary so she can
        say she's getting sleepy (her words, not a canned string), and
        it holds off re-sleeping through the wake-attempt window while
        the flip-flop is still crossing.

        """
        try:
            core_state = self.client.get_state()
        except (OSError, ConnectionError, RuntimeError) as e:
            note_swallowed(
                "mind.sleep._check_sleep_sync",
                e,
            )
            return

        adenosine = core_state.chemicals.get("adenosine", 0.0)
        daemon_phase = core_state.phase_name

        # ── Hypnagogic edge bookkeeping ──
        # The theta descent is a rest state: discretionary work holds
        # for its duration. On the rising edge the learners pause and
        # volition is held; on the falling edge — the descent aborted
        # back to waking rather than completing into sleep — the
        # normal resume path restores them. Completion into sleep
        # needs no resume: ``_is_sleeping`` takes over the hold.
        # Inner life is not paused — it owns the descent itself.
        if not self._is_sleeping and not self._is_meditating:
            hypnagogic_now = self.is_hypnagogic
            if hypnagogic_now and not self._was_hypnagogic:
                self.learner.pause()
                self.brain.frontal.self_learner.pause()
                self._suppress_volition = True
            elif self._was_hypnagogic and not hypnagogic_now:
                self._resume_background()
            self._was_hypnagogic = hypnagogic_now

        # ── Waking side ──
        if not self._is_sleeping and not self._is_meditating:
            self._waking_side(core_state, adenosine, daemon_phase)
            return

        # ── Body-state sync ──
        if self._is_sleeping:
            # The body owns the transition in both directions: a
            # daemon phase that has genuinely woken while the mind
            # still sleeps pulls the mind up — the two are never in
            # opposite sleep states. This applies to user-initiated
            # sleep too: a /sleep whose body never crossed is a body
            # that could not sleep, not a mind that should pretend.
            self._check_wake_desync(daemon_phase)

    def _waking_side(self, core_state, adenosine: float, daemon_phase: str) -> None:
        """The waking-side maintenance: desync check, boundary
        tracking, min-awake and idle guards, drowsiness announcement,
        and the wake-attempt window while the body's chemistry is still
        switching.

        There is no mind-initiated sleep onset: transitions into
        sleep are hers (the self-sleep urge or a user command) or the
        body's own network crossing under genuine pressure, which the
        desync check joins.
        """
        # Checked on this path because it already holds a state
        # snapshot, so comparing the two layers' sleep states
        # costs no extra IPC call. A joined desync puts the mind to
        # sleep — the waking-side gates below no longer apply.
        if self._check_sleep_desync():
            return
        now = time.time()
        # Feed the commitment boundary before the gates — the
        # sustained-crossing window must track the physical
        # signal, not the gate-opened signal. Pressure that held
        # above threshold while the user was active is genuinely
        # sustained, not a transient, so the confirm clock keeps
        # running behind the min-awake and idle gates.
        self._drowsy_boundary.update(adenosine, now)
        daemon_sleeping = daemon_phase in ("nrem", "rem")
        # Minimum awake time to prevent oscillation at startup.
        awake_elapsed = now - self._last_wake_time
        if awake_elapsed < AUTO_SLEEP_MIN_AWAKE:
            return
        # Don't fall asleep while the user is actively talking to
        # it. If the last interaction was recent (within 60s), the
        # user is engaged in conversation — falling asleep mid-
        # conversation is a jarring UX failure. The daemon may
        # enter NREM on its own (adenosine-driven), but the
        # cognitive mind stays awake until the user goes idle.
        idle = now - self._last_interaction_time
        if idle < AUTO_SLEEP_MIN_IDLE:
            logger.debug(
                f"waking-side signals deferred: user active {idle:.1f}s ago "
                f"(adenosine={adenosine:.2f}, phase={daemon_phase})"
            )
            return
        # Drowsiness announcement — before falling asleep, it
        # announces it's getting sleepy. This is composed from
        # its understanding of "sleepiness" (not a hardcoded
        # string). The boundary's commit edge fires once per
        # sustained crossing; it re-arms only if pressure drops
        # below DROWSINESS_EXIT, so it can announce again if it
        # genuinely recovers and fades a second time.
        if self._drowsy_boundary.just_committed:
            self._announce_drowsiness()
        # Wake-attempt window: while a wake cascade is still in its
        # switch window (LC's ~23-min time constant), a daemon-side
        # sleep phase at moderate pressure is the flip-flop mid-
        # transition — do not re-sleep on the body's in-progress wake.
        # There is no held drive to renew: the impulses already spent
        # their push, and the body's chemistry either crosses or the
        # genuine pressure below takes the mind back under. Circadian
        # pressure overrides only a wake nobody asked for — a user-
        # requested wake reads melatonin as grogginess, not a knockout.
        if (
            daemon_sleeping
            and self._wake_attempt_outstanding
            and adenosine < WAKE_RESCUE_ADENOSINE
            and (
                core_state.chemicals.get("melatonin", 0.0) < WAKE_RESCUE_MELATONIN
                or self._last_wake_user_initiated
            )
        ):
            return

    def _announce_drowsiness(self) -> None:
        """Announce that it's getting sleepy, before falling asleep.

        This is composed from its understanding of "sleepiness" via
        the thought composer — NOT a hardcoded string. It speaks
        its own words about its own state. If it can't articulate
        it yet (doesn't know the concept well enough), it stays
        silent and just falls asleep — the drowsiness is still
        tracked internally.
        """
        try:
            emotion = self.feel()
            thought = self.brain.frontal.composer.compose_about("sleepiness", emotion, focused=True)
            if thought and thought.content and thought.confidence > 0.3:
                # Thought.content is semantic substrate, not final speech.
                # Route sleep-onset expression through the same language
                # generator used by ordinary cognition.
                spoken = self.language.render(thought, emotion)
                if not spoken:
                    return
                self._emit_live_thought("thought", spoken)
                if self._on_speak is not None:
                    try:
                        self._on_speak(spoken)
                    except Exception as e:  # noqa: BLE001
                        note_swallowed(
                            "mind.sleep._announce_drowsiness",
                            e,
                        )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.sleep._announce_drowsiness",
                e,
            )

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
            user_initiated: If True, the adenosine/cycle auto-wake
                thresholds do not apply — only /wake or a genuine body
                wake ends it. This is for when the user puts it to bed.
                (Body-state sync still holds: if the daemon's phase
                never reaches sleep, or wakes on its own, the mind
                follows — the two are never in opposite sleep states.)
            nap: If True, the sleep cycle is limited to light sleep
                (N1→N2) only — no N3 deep consolidation, no REM. It
                wakes after the N2 period ends via the sleep watcher.
                This models a short rest with light consolidation
                (N2 sleep spindles) without the heavy systems
                consolidation of full sleep.
        """
        if self._is_sleeping:
            return
        # Teardown has begun — a new sleep transition now would write
        # a stale is_sleeping into the final save.
        if self._stop_latched:
            return
        self._is_sleeping = True
        self._user_initiated_sleep = user_initiated
        self._nap_mode = nap
        self._sleep_start_time = time.time()
        cycle = self.inner_life.sleep_cycle
        self._sleep_start_cycles = cycle.cycles_completed if cycle is not None else 0
        # New episode: the wake-desync trackers start unconfirmed so a
        # waking phase during the onset window reads as an in-progress
        # transition, not a body wake.
        self._body_sleep_confirmed = False
        self._body_awake_since = 0.0
        self.client.set_zone(ZONE_SLEEPING)
        self._send_sleep_neuro_impulses()

        self._engage_sleep_mechanism()

        # Initial light consolidation at sleep onset (N1 — the
        # lightest stage). The deeper N3 consolidation happens
        # later when the cycle reaches slow-wave sleep. This is
        # sleep-onset only — it is NOT called on restore (the
        # consolidation already happened before the restart).
        n_memories = self.brain.frontal.memory.consolidate_during_sleep(
            SleepStage.N1,
            duration=300.0,
        )
        if n_memories > 0:
            self._emit_live_thought(
                "learning",
                f"Sleep onset: {n_memories} episodes lightly consolidated (N1)",
            )
        # Record the onset unconditionally. Previously the only record
        # was the live thought above, which fires only when there was
        # something to consolidate — so a sleep that consolidated
        # nothing left no trace at all, and the number of sleep
        # episodes could not be reconstructed from any durable state.
        record_event(
            "sleep_onset",
            "sleep episode began",
            user_initiated=user_initiated,
            nap=nap,
            cycle=(cycle.cycles_completed if cycle is not None else 0),
            consolidated=n_memories,
        )

    def _engage_sleep_mechanism(self) -> None:
        """Engage the sleep mechanism — pause learners, resume inner life.

        This is the reusable core of sleep entry, called both from
        :meth:`sleep` (on explicit or body-joined onset) and from the restore
        path (when resuming a saved sleep state after a restart). It
        does NOT set the ``_is_sleeping`` flag or the zone — those are
        set by the caller — and it does NOT do N1 consolidation, which
        is sleep-onset only.

        Without this, a restored sleep state has the flags and zone
        set but the learners still run (acquiring new knowledge during
        "sleep") and inner life generates waking thoughts instead of
        dreams — it is flagged asleep but behaves awake.
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
        self.brain.frontal.self_learner.pause()

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
        # consolidation can fire. Without this, it sleeps but never
        # dreams — the sleep cycle tracker is never created and N3
        # compression never runs. On restore, inner life was just
        # started (not paused), so resume() is a harmless no-op —
        # but calling it keeps the two paths identical.
        self.inner_life.resume()

        # Reset discourse state — recent entities from the previous
        # conversation shouldn't persist across sleep. Just as humans
        # don't maintain pronoun referents across a sleep cycle, the
        # comprehension engine's entity tracking is cleared so it
        # starts the next conversation fresh.
        try:
            self.brain.frontal.comprehension.reset()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.sleep._engage_sleep_mechanism",
                e,
            )

        # The situation model is discarded alongside it. It holds the
        # world the *previous* conversation described — who was where,
        # what was being carried. Carrying that into a new session
        # would let Genesis answer "Where is Mary?" from a story the
        # user told hours ago and has moved on from, which is worse
        # than not answering: it would be confidently stale.
        try:
            self.brain.parietal.situation_model = SituationModel()
            self.brain.frontal.situation_model = self.brain.parietal.situation_model
            self.brain.frontal._question_handler._situation = self.brain.frontal.situation_model
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.sleep._engage_sleep_mechanism",
                e,
            )

    def sleep_status(self) -> str:
        """Return a summary of its sleep state for the /sleep and /wake commands.

        Includes its current phase, neurochemical levels relevant to sleep
        (adenosine, melatonin, cortisol, BDNF), and recent dream content
        so the user can see what it's experiencing.

        The phase shown is the *effective* phase (resolved via
        :meth:`_effective_phase`), not the daemon's raw emergent phase.
        When the zone is Sleeping but the sleep-wake network hasn't
        reached sleep onset yet, the raw phase still reads "active"
        while every other system (brain waves, emotion) already treats
        it as asleep. Showing the effective phase keeps the display
        consistent with what it's actually experiencing.
        """
        lines: list[str] = []
        try:
            core_state = self.client.get_state()
            eff_phase = PHASE_NAMES.get(self._effective_phase(core_state), "unknown")
            lines.append(
                f"  Phase: {eff_phase}  "
                f"arousal={core_state.arousal:.2f}  "
                f"valence={core_state.valence:.2f}"
            )
            lines.append(f"  Plasticity gate: {core_state.plasticity_gate:.2f}")
            melatonin = core_state.chemicals.get("melatonin", 0.0)
            if melatonin >= WAKE_RESCUE_MELATONIN:
                lines.append(
                    f"  Circadian drive: melatonin={melatonin:.2f} — "
                    "deep into night; staying awake is uphill."
                )
        except (OSError, ConnectionError, RuntimeError) as e:
            lines.append(f"  (neurochemical state unavailable: {e})")

        # Show recent dreams. get_recent_episodes sees only the active
        # list — dreams archived by sleep compression (historical passes
        # swept every dream below the retention cap) stay retrievable by
        # ID and are counted from the index, matching /dreams.
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
            archived = self._archived_dream_count()
            if archived:
                lines.append(
                    f"  No active dreams — {archived} archived by sleep "
                    "compression (visible via /dreams)."
                )
            else:
                lines.append("  No dreams recorded yet.")

        # Receptor safety check — are its levels safe to wake?
        try:
            profile = self.client.get_plasticity_profile()
            safe = profile.cortisol_tonic < AUTO_WAKE_CORTISOL and profile.bdnf_tonic > 0.2
            if safe:
                lines.append("  Receptor state: safe to wake.")
            else:
                lines.append(
                    f"  Receptor state: cortisol={profile.cortisol_tonic:.2f} "
                    f"BDNF={profile.bdnf_tonic:.2f} — still recovering."
                )
        except (OSError, ConnectionError, RuntimeError) as e:
            note_swallowed(
                "mind.sleep.sleep_status",
                e,
            )

        return "\n".join(lines)

    def _archived_dream_count(self) -> int:
        """Count dream-insight episodes archived out of the active list.

        Reads the LTM index directly — the same mechanism ``/dreams``
        uses to report the archived history. Any failure (missing
        index, unreadable file) yields 0, which falls back to the
        plain "no dreams" line rather than erroring the status.
        """
        try:
            from genesis_client.ltm_index import scan_ltm_index
        except ImportError:
            return 0
        try:
            entries = scan_ltm_index(Path(self.data_dir) / "ltm_store.meta", source_module=9)
        except OSError:
            return 0
        return sum(1 for e in entries if e.is_archived and not e.is_deleted)

    def wake_readiness(self) -> tuple[bool, str]:
        """Check whether it is ready to wake up naturally.

        Waking from deep sleep (N3) causes severe sleep inertia — the
        groggy, disoriented state where cognitive capacity is reduced.
        Waking before adenosine has drained leaves residual sleep pressure,
        which pushes it right back into drowsiness. Waking before
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
        3. **Receptor state** — cortisol tonic < ``AUTO_WAKE_CORTISOL``
           (0.06) and BDNF > 0.2. Chronic stress suppresses BDNF; sleep
           restores it. Cortisol's envelope tops out around ~0.08 under
           full drive (resting ~0.02), and the gate sits at the Rust
           stress-phase boundary (STRESS_CORTISOL_GATE) — above the
           circadian basal peak, so a healthy morning surge passes while
           genuine stress elevation still reads as carrying a load. If
           receptors haven't recovered, waking would be premature.

        Returns:
            (ready, reason): ``ready`` is True if all three signals
            indicate it can wake without severe inertia. ``reason``
            is a human-readable explanation — either "ready" or why
            it's not ready yet.
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
            note_swallowed(
                "mind.sleep.wake_readiness",
                e,
            )
            # If we can't read state, don't block wake — the daemon
            # might be disconnected, and forcing it to stay asleep
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
            if profile.cortisol_tonic >= AUTO_WAKE_CORTISOL:
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
            note_swallowed(
                "mind.sleep.wake_readiness",
                e,
            )
            # Same as above — don't block wake on a failed read.

        return True, "ready"

    def wake(self, user_initiated: bool = False) -> WakeOutcome:
        """Wake Genesis from sleep.

        After the zone is set back to conversation, dream insights from
        the last sleep are retrieved from LTM and surfaced as emotional
        impressions — dream residues that color its waking thoughts and
        mood. These are NOT learning goals. Dreams produce affective
        residue, not targets for the autonomous learner.

        A human doesn't dream about goals. Goals are a waking, prefrontal
        activity. Dreams are subcognitive and affective — they leave
        emotional traces that surface as reflective thoughts ("I was
        dreaming about..."), not as items on a to-do list.

        The separation:
        - Dreams → _dream_residues (emotional impressions, reflective thoughts)
        - Curiosity → _agency_topics (learning goals, autonomous learner)

        The wake is sent as the body's own arousal chemistry: orexin,
        histamine and norepinephrine impulses onto the network's wake
        side (the endogenous channel — LC is the densest orexinergic
        target in the brain), dopamine for reward re-engagement, and
        melatonin suppressed to release the circadian brake. Without
        this, the daemon's phase stays asleep (adenosine is still high,
        histamine still low) even though the mind's _is_sleeping flag
        is cleared. Nothing is held — the impulses decay on their own
        half-lives and the flip-flop's state decides the crossing.

        ``user_initiated`` marks an explicit user wake request
        (/wake, or an interaction that rouses it). While the wake
        cascade is still in its switch window, a user-initiated wake
        resists the circadian pull-back — the waking-side sync still
        honors genuine adenosine exhaustion, but melatonin alone cannot
        undo a wake the user asked for.
        """
        self._last_wake_user_initiated = user_initiated
        if not self._is_sleeping:
            # Already awake — but the body may not be. A wake request
            # while the daemon's phase is still asleep is the request
            # that brings a lagging body up: send the arousal cascade;
            # the body's chemistry either crosses in its switch window
            # or genuine pressure pulls the mind back under at the next
            # sync check.
            try:
                daemon_asleep = self.client.get_phase().phase in (
                    PHASE_NREM,
                    PHASE_REM,
                )
            except (OSError, ConnectionError, RuntimeError, AttributeError):
                daemon_asleep = False
            if daemon_asleep:
                logger.info("wake requested while daemon still asleep: sending arousal cascade")
                self._send_wake_neuro_impulses()
                record_event(
                    "wake_body_push",
                    "wake requested while daemon phase asleep",
                )
                return WakeOutcome.BODY_WAKE_PUSHED
            return WakeOutcome.ALREADY_AWAKE
        # Teardown has begun — the zone write and the inner-life
        # resume below belong to threads already joined. She stays
        # asleep through the final save.
        if self._stop_latched:
            return WakeOutcome.BLOCKED
        episode_secs, was_nap, was_user_initiated = self._clear_sleep_flags()

        # Send neurochemical wake impulses to push the emergent phase
        # out of NREM/REM. Without this, the daemon's phase stays
        # asleep because adenosine is still high and histamine is low,
        # even though the mind's _is_sleeping flag is cleared. This
        # closes the desync between cognitive and subcognitive layers.
        self._send_wake_neuro_impulses()

        # Record the wake unconditionally. Waking previously produced no
        # durable record of any kind, so sleep *stages* were traceable
        # while the onset and termination of an episode were not — the
        # boundaries were the one part left unmeasured.
        record_event(
            "wake",
            "sleep episode ended",
            duration_secs=round(episode_secs, 3),
            nap=was_nap,
            user_initiated=was_user_initiated,
        )

        # Resume autonomous learning. Inner life was not paused during
        # sleep (it generates dreams and advances the sleep cycle), so
        # it doesn't need resuming — but we call resume() anyway as a
        # safety net in case it was paused by a conversation during sleep.
        #
        # The learner is delayed to model sleep inertia — the groggy
        # period after waking when cognitive capacity is reduced.
        # Resuming the learner immediately causes CPU stress (its body
        # isn't ready), which triggers the regulator's throttle, creating
        # a drowsy→active→stressed→drowsy oscillation. Giving its
        # neurochemistry time to stabilize (adenosine drains, histamine
        # rises) before learning starts prevents this loop. Inner life
        # resumes immediately so it can think and converse — only
        # autonomous learning is delayed.
        self._wake_time = time.time()
        self._last_wake_time = self._wake_time
        self.inner_life.resume()
        self._resume_learners_after_wake()
        self._wake_epilogue()
        return WakeOutcome.WOKE

    def _clear_sleep_flags(self) -> tuple[float, bool, bool]:
        """Clear the sleep flags for wake. Returns (episode_secs,
        was_nap, was_user_initiated) for the journal record — the
        episode's length is captured before the start time clears."""
        episode_secs = time.time() - self._sleep_start_time if self._sleep_start_time > 0.0 else 0.0
        was_nap = self._nap_mode
        was_user_initiated = self._user_initiated_sleep
        self._is_sleeping = False
        self._user_initiated_sleep = False
        # Waking starts a new epoch — any volition suppression still
        # latched from a conversation turn lapses here.
        self._suppress_volition = False
        # Reset the commitment boundary so the next wake period
        # announces drowsiness on a fresh crossing, not stale latched
        # state.
        self._drowsy_boundary.reset()
        # Clear nap mode — the next sleep is a full sleep unless
        # explicitly set as a nap again.
        self._nap_mode = False
        self._sleep_start_time = 0.0
        self.client.set_zone(ZONE_CONVERSATION)
        return episode_secs, was_nap, was_user_initiated

    def _wake_epilogue(self) -> None:
        """Post-wake housekeeping — rest timer, workspace clear,
        regulator notify, and the dream-residue restore thread."""
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
            self.brain.frontal.global_workspace.clear()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.sleep.wake",
                e,
            )

        # Notify the regulator that rest has ended (resets its
        # cooldown timer so it doesn't immediately try to meditate
        # after waking)
        self.regulator.notify_rest_ended()

        named_thread(
            "dream-residue-restore",
            self._restore_dream_residues_after_wake,
            comm="g-dream-rest",
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
        """Wake the body through its arousal chemistry.

        The stage boundary belongs to the sleep-wake network, and the
        network's wake side is what waking neuromodulators actually
        drive: LC receives the densest orexinergic innervation in the
        brain (Hagan et al., PNAS 1999) and the wake-active TMN
        histamine system activates it — the body's tick loop couples
        above-baseline orexin and histamine onto the LC population's
        input, so raising them is the endogenous channel into the
        flip-flop. Nothing held, nothing external: the impulses decay
        on the chemicals' own half-lives and the network's state
        decides the crossing.

        - Orexin +0.10: stabilize/excite the wake side of the switch.
        - Histamine +0.10: TMN wake-center tone onto LC.
        - Norepinephrine +0.05: systemic arousal (LC is the NE nucleus).
        - Dopamine +0.05: mild arousal to re-engage the reward system.
        - Melatonin -0.10: suppress the circadian sleep signal — its
          orexin/histamine suppression releases, compounding the rise.

        Adenosine is deliberately untouched. Glymphatic clearance
        during the sleep itself is what lowers it — a fixed post-wake
        impulse erased the same pressure regardless of whether she
        slept eight hours or two minutes, and at -0.30 it tripled the
        chemical's documented 0.1 impulse cap. Waking tired after a
        short nap is correct physiology, not a bug to purge.

        `_last_wake_attempt` records a successful send — the sync
        checks bracket the cascade's switch window from it.
        """
        self._learner_neuro_impulse(CHEM_OREXIN, 0.10)
        self._learner_neuro_impulse(CHEM_HISTAMINE, 0.10)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, 0.05)
        self._learner_neuro_impulse(CHEM_DOPAMINE, 0.05)
        self._learner_neuro_impulse(CHEM_MELATONIN, -0.10)
        self._last_wake_attempt = time.time()
        self._last_sleep_attempt = 0.0

    def _send_sleep_neuro_impulses(self) -> None:
        """Send the body toward sleep through its own chemistry.

        Volitional sleep onset is not a switch on the network — it is
        the endogenous chemistry of disengagement: the circadian signal
        rises, the wake side's stabilizers withdraw, and the flip-flop
        crosses on its own pressure. VLPO has no orexin receptors, so
        the sleep direction can only work through the wake side.

        - Melatonin +0.10: the circadian night signal; through the
          coupling matrix it suppresses orexin and histamine and
          promotes GABA.
        - GABA +0.10: VLPO's own transmitter — the sleep population's
          drive is GABAergic.
        - Orexin -0.10: withdraw the switch's stabilizer (orexin loss
          is the narcolepsy signature — the wake side destabilizes).
        - Histamine -0.10: withdraw wake-center tone.
        - Norepinephrine -0.05: drop systemic arousal.

        Whether the network actually crosses is the body's to decide —
        adenosine (Process S) and the circadian phase weigh it. The
        sync checks follow the body's phase either way.
        """
        self._learner_neuro_impulse(CHEM_MELATONIN, 0.10)
        self._learner_neuro_impulse(CHEM_GABA, 0.10)
        self._learner_neuro_impulse(CHEM_OREXIN, -0.10)
        self._learner_neuro_impulse(CHEM_HISTAMINE, -0.10)
        self._learner_neuro_impulse(CHEM_NOREPINEPHRINE, -0.05)
        self._last_sleep_attempt = time.time()
        self._last_wake_attempt = 0.0

    def _check_sleep_desync(self) -> bool:
        """Compare this layer's sleep state against the daemon's.

        The two layers track sleep independently: `_is_sleeping` is the
        mind's own flag, and the daemon derives its own phase from
        neurochemistry. The body owns the transition: when its
        sleep-wake network has crossed into NREM/REM while the mind is
        still flagged awake, the resolution is the mind joining the
        body's sleep, not vetoing it. A held wake drive used to fight
        body-initiated onsets back out of NREM — with volition acting
        through chemistry now, a spent cascade cannot do that.

        An asleep phase is also expected for the whole wake-attempt
        window — the flip-flop takes tens of minutes to switch from
        deep NREM — so a body-initiated sleep is only honored once the
        wake attempt has run its course; an earlier check would report
        a healthy in-progress transition as a fault. The attempt is not
        a veto against genuine pressure, though: deep adenosine or
        (behind a non-user wake) deep melatonin means the body is
        asleep for real, and the mind joins rather than waiting out
        the window awake while the body sleeps on regardless.

        Returns True when a persistent desync was detected and the
        mind joined the body's sleep.
        """
        if self._is_sleeping:
            return False
        try:
            # `PhaseInfo` carries `phase` (an int code). It has no
            # `emergent_phase` — that field belongs to `CoreState`, a
            # different object. Reading it raised AttributeError, which
            # this handler does not catch, so the desync check escaped to
            # the metronome and has been dead since it was written: the
            # "mind awake, daemon asleep" fault it exists to record was
            # silently never detected.
            phase = self.client.get_phase().phase
        except (OSError, ConnectionError, ValueError, AttributeError) as e:
            note_swallowed(
                "mind.sleep._check_sleep_desync",
                e,
            )
            return False
        if phase not in (PHASE_NREM, PHASE_REM):
            return False
        now = time.time()
        # Inside the wake-attempt window, an asleep phase is the
        # expected transition — the flip-flop takes tens of minutes to
        # switch from deep NREM. But the attempt is not a veto against
        # genuine pressure: deep adenosine means exhaustion has already
        # won (the body's sleep is real, not mid-wake), and deep
        # melatonin means the circadian drive won — except behind a
        # user-initiated wake, which treats melatonin as grogginess,
        # not a knockout.
        if self._last_wake_attempt and now - self._last_wake_attempt < WAKE_DESYNC_SETTLE_SECONDS:
            try:
                chemicals = self.client.get_state().chemicals
                adn = chemicals.get("adenosine", 0.0)
                mel = chemicals.get("melatonin", 0.0)
            except (OSError, ConnectionError, RuntimeError, AttributeError):
                adn, mel = 0.0, 0.0
            if adn < WAKE_RESCUE_ADENOSINE and (
                mel < WAKE_RESCUE_MELATONIN or getattr(self, "_last_wake_user_initiated", False)
            ):
                return False
        # With no attempt outstanding, a brief post-wake grace still
        # suppresses stage flutter near the threshold.
        settle = now - self._wake_time if self._wake_time else 0.0
        if settle < WAKE_DESYNC_GRACE_SECONDS:
            return False
        record_event(
            "sleep_desync",
            "mind is awake but the daemon's phase is asleep — joining the body",
            phase=PHASE_NAMES.get(phase, str(phase)),
            seconds_since_wake=round(settle, 1),
        )
        logger.warning(
            "sleep desync: mind awake for %.0fs but daemon phase is %s — joining the body",
            settle,
            PHASE_NAMES.get(phase, phase),
        )
        # Reconcile in the body's direction: it is asleep and the mind
        # sleeps with it — the two are never in opposite sleep states.
        # `sleep()` sets the zone, sends the sleep chemistry (against
        # an already-asleep network it lands as ordinary tone), and
        # engages the sleep cycle so consolidation, dreams, and the
        # dream-residue restore on wake all run.
        try:
            self.sleep(user_initiated=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"desync sleep join failed: {e}")
            return False
        return True

    def _check_wake_desync(self, daemon_phase: str) -> bool:
        """The reverse desync: mind asleep while the body is awake.

        Symmetric with `_check_sleep_desync` — the body owns the
        transition in both directions. A body that has genuinely woken
        while the mind still sleeps pulls the mind up; the two are
        never in opposite sleep states.

        "Genuinely woken" needs care: after `sleep()` sends its
        chemistry, the body can take many minutes to cross — LC's
        population switches on a ~23-minute time constant, and nothing
        held can shorten it. So a waking phase only means a desync once
        the body has been *seen* asleep this episode (it crossed, then
        woke back up — and only if the awake phase persists past a
        flutter guard), or once the onset window has closed without the
        body ever crossing (a /sleep on a body that could not sleep).
        Drowsy is heading-to-sleep, not a desync — except past the
        onset window, where a stuck drowsy is a failed onset too.

        Returns True when a persistent desync was detected and the
        mind woke to join the body.
        """
        if daemon_phase in ("nrem", "rem"):
            self._body_sleep_confirmed = True
            self._body_awake_since = 0.0
            return False
        now = time.time()
        if not self._body_sleep_confirmed:
            # The body has not crossed since the mind slept. While the
            # sleep cascade is still in its switch window — the LC
            # population's own ~23-minute time constant — a non-sleep
            # phase is a legitimate in-progress onset, not a desync. A
            # restored sleep has no attempt timestamp, so it lands
            # straight in the persistence guard.
            if (
                self._last_sleep_attempt
                and now - self._last_sleep_attempt < WAKE_DESYNC_SETTLE_SECONDS
            ):
                return False
        elif daemon_phase == "drowsy":
            # After a confirmed sleep, drowsy is the body heading back
            # down — not a wake.
            return False
        # Any remaining non-sleep phase — awake-side after a confirmed
        # sleep (the body genuinely woke), or still-awake/drowsy once
        # the onset effort has closed (a body that could not sleep) —
        # must persist past the flutter guard before it counts.
        if self._body_awake_since == 0.0:
            self._body_awake_since = now
            return False
        if now - self._body_awake_since < WAKE_DESYNC_GRACE_SECONDS:
            return False
        record_event(
            "wake_desync",
            "mind is asleep but the daemon's phase is awake — joining the body",
            phase=daemon_phase,
            body_sleep_confirmed=self._body_sleep_confirmed,
        )
        logger.warning(
            "wake desync: mind asleep but daemon phase is %s — joining the body",
            daemon_phase,
        )
        # Reconcile in the body's direction: it is awake and the mind
        # wakes with it — the two are never in opposite sleep states.
        try:
            self.wake()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"desync wake failed: {e}")
            return False
        return True

    def _resume_learners_after_wake(self) -> None:
        """Resume both learners with a delay for sleep inertia.

        The self-learner doesn't have its own thread, but resuming it
        with the same delay keeps the lifecycle symmetric with
        sleep() and prevents self-directed inference from firing
        immediately on wake (before neurochemistry has stabilized).
        """
        self._defer(_LEARNER_WAKE_RESUME_DELAY_S, self.learner.resume)
        self._defer(_LEARNER_WAKE_RESUME_DELAY_S, self.brain.frontal.self_learner.resume)

    def meditate(self) -> None:
        """Put Genesis into meditation mode for quiet restoration.

        Meditation is a lighter recovery than sleep. During meditation:
        - Learning and code scanning are paused
          (the metronome loop checks is_meditating before running
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
        # Teardown has begun — no new meditation zone write.
        if self._stop_latched:
            return
        self._is_meditating = True
        self._meditation_start_time = time.time()
        self.client.set_zone(ZONE_REFLECTION)
        # Meditation is a deliberate state change with no durable
        # record, like sleep onset before this. It is also the only
        # state in which inner life and the learner are both paused,
        # so an unrecorded entry would explain a gap in the journal
        # with nothing to account for it.
        record_event("meditation_begin", "meditation started")

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

        # The body takes up the slow pulse for the sit — the RSA
        # analog of a settling breath, armed once and run on the
        # body's own clock while the cardiac center holds the
        # boundary watch. Absent probe → arm fails, the keep stays
        # disarmed, meditation proceeds without it.
        self.heart.arm_hypnagogic()

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
        sleep because the HPA axis keeps arousal systems active and the
        network's homeostatic drive hasn't reached sleep onset.

        When to use this instead of /sleep or /meditate:
        - /sleep sets the zone to SLEEPING and sends the sleep cascade
          — the endogenous chemistry of disengagement (melatonin and
          GABA up, orexin, histamine and NE withdrawn). If the network's
          sleep pressure is low and cortisol is elevated, the cascade
          may not carry her under.
        - /meditate emits gentle calming impulses (GABA +0.05, cortisol
          -0.02 every 10s) that are too weak to overcome ongoing
          interoception stress from CPU load.
        - /sleep-aid sends the same sleep cascade /sleep uses and adds
          an acute cortisol suppression to relieve the HPA load that
          keeps the arousal systems active — analogous to a GR
          antagonist (mifepristone) taking the brake off BDNF recovery.
          Once the network crosses its own sleep-onset threshold
          (LC < 4 Hz) the natural mechanisms take over: Process-S
          adenosine clearance, BDNF recovery with the 5x sleep
          multiplier, and receptor recovery.

        After the impulses, sleep() is called to set the zone to
        SLEEPING, triggering the daemon's dreaming pass and the
        sleep-tuned neuro params (5x BDNF recovery multiplier).

        This is an emergency intervention, not a daily supplement.
        The natural sleep cycle then takes over for sustained recovery.
        """
        if self._is_sleeping:
            return
        if self._is_meditating:
            self.wake_from_meditation()

        # The HPA reduction that stress-induced insomnia needs. The
        # sleep cascade itself comes from sleep() below — the
        # endogenous chemistry of disengagement; stage selection stays
        # the network's.
        self._learner_neuro_impulse(CHEM_CORTISOL, -0.15)

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
        # The pulse releases first — the keep belongs to the sit and
        # must not outlive it, teardown path included.
        self.heart.disarm_hypnagogic()
        # Teardown has begun — the zone write and the resumes below
        # belong to threads already joined. The flag still clears so
        # the saved state does not record a phantom meditation.
        if self._stop_latched:
            self._is_meditating = False
            return
        duration = (
            time.time() - self._meditation_start_time
            if getattr(self, "_meditation_start_time", 0.0) > 0.0
            else 0.0
        )
        self._is_meditating = False
        # Any volition suppression latched by a conversation turn
        # before meditating lapses on exit.
        self._suppress_volition = False
        self.client.set_zone(ZONE_CONVERSATION)
        record_event(
            "meditation_end",
            "meditation ended",
            duration_secs=round(duration, 3),
        )

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
            note_swallowed(
                "mind.sleep._resolve_episode",
                e,
            )
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
        assoc_re = re.compile(r"\[association\] episode (\d+) ↔ episode (\d+) \(distance: (\d+)\)")
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
        compression by the time it wakes up.

        Old-format insights (without embedded text) fall back to
        episode retrieval by ID.
        """
        import re

        dream_re = re.compile(
            r"\[dream-insight\] episode (\d+) connects to episode (\d+) "
            r"through (\d+) hops \(direct distance: (\d+)\)"
        )

        ep_cache: dict[int, str] = {}

        def _match_concepts_from_leaf(leaf: str, dream_concepts: list[str]) -> None:
            """Extract concept names from a single leaf text, appending to dream_concepts."""
            leaf_lower = leaf.lower()
            for concept_id in self.brain.frontal.network.world_concept_ids:
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

        Dreams produce affective residue — themes that color its waking
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

        # Interpret the night before filing the residue — the analytic
        # engine reads the manifest dream content back to its latent
        # work. The latent concepts it carried join the residue queue:
        # what the dream was *about* underneath continues into waking
        # as impressions, and whatever it discharged is journaled.
        try:
            interpretation = self.psychoanalysis.interpret_dreams(dream_concepts)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.sleep._queue_dream_residues", e)
        else:
            if interpretation.confidence > 0.0:
                residue = list(dream_concepts)
                for cid in interpretation.latent[:5]:
                    if cid not in residue:
                        residue.append(cid)
                dream_concepts = residue
                record_event(
                    "dream_interpretation",
                    "The night's manifest content read back to its latent work",
                    latent=interpretation.latent,
                    wish=interpretation.wish,
                    discharged=interpretation.discharged,
                    condensation=interpretation.mechanisms.get("condensation"),
                    displacement=interpretation.mechanisms.get("displacement"),
                    confidence=round(interpretation.confidence, 3),
                )

        # Queue as emotional impressions, not learning goals.
        # These will surface as dream-reflection thoughts, not as
        # agency topics for the autonomous learner. The inner life
        # composes the actual reflection from its knowledge via the
        # language engine — we do NOT hardcode what it says here.
        if hasattr(self.inner_life, "add_dream_residues"):
            self.inner_life.add_dream_residues(dream_concepts[:10])

        # Telemetry line (factual status, not self-expression) so the
        # live ticker shows that dreams are surfacing. The reflective
        # thought itself is composed by _dream_reflection_thought and
        # flows through _on_spontaneous_thought → language engine,
        # like every other spontaneous thought.
        self._emit_live_thought(
            "dream",
            f"Dream residue: {len(dream_concepts[:10])} emotional impressions "
            f"from {len(dreams)} dream insights",
        )

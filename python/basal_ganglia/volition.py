"""Mind volition — urge handling and voluntary actions."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brainstem.thresholds import (
    SLEEP_URGE_ADENOSINE_FLOOR,
    WAKE_RESCUE_ADENOSINE,
    WAKE_RESCUE_MELATONIN,
    WAKE_STABILIZATION_WINDOW,
)
from frontal_lobe.canvas import DrawingResult
from frontal_lobe.volition import Urge, Volition
from tools.project_creator import create_project, manage_project_lifecycle

from genesis_client.protocol import CHEM_NAMES, EMOTIONAL_TAG_SIZE, MODULE_METACOGNITION
from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread
from infrastructure.journal import record_error
from infrastructure.text import is_blank

logger = logging.getLogger(__name__)

# Urge workers run below the interactive path's CPU priority — a
# running bug scan or code-learning burst must lose every scheduler
# handoff to a live think(), not coin-flip it for the core.
_VOLITION_NICE = 10

# Minimum seconds between public-channel replies. A channel can
# storm mentions; she answers when addressed but not faster than
# this — the message is still heard even when the reply gate is
# closed.
_CHANNEL_REPLY_MIN_S = 45.0

# The voluntary lid's held positions — openness [0, 1] over SET_LID.
# _LID_REST is a shut eye, not a shutter: a few percent of light
# still leaks, matching the phase-closed transmission the body
# already models for sleep.
_LID_OPEN = 1.0
_LID_REST = 0.05

# Seconds a self-chosen eye-rest holds the lid down before
# releasing it back to the phase's own position.
_REST_EYES_HOLD_S = 20.0

# Accumulator noise σ per √s on every registered urge — the
# stochastic fluctuation that makes near-threshold crossings arrive
# early or late rather than on a fixed schedule (the readiness
# potential's trigger in the Schurger et al. 2012 account). At ~1 Hz
# ticks this is ±0.03 per step against thresholds ~0.7: it shapes the
# instant of a poised act, it cannot conjure one from rest.
_URGE_FLUCTUATION = 0.03

# Files per self-chosen study bout — a volitional study session is a
# recurring act with the urge rebuilding between bouts, not the
# full-corpus pass the /learn-code command performs.
_LEARN_CODE_BOUT = 25


def _lower_thread_priority(nice: int = _VOLITION_NICE) -> None:
    """Drop the calling thread's CPU priority (Linux per-thread nice).

    ``os.setpriority`` with the native TID lowers only this thread —
    lowering needs no privilege. Foreground cognition (think(),
    respond()) keeps nice 0, so under load every GIL/scheduler handoff
    favors the interactive turn over background urge work.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), nice)
    except OSError as e:
        logger.debug(f"_lower_thread_priority failed: {e}")


class VolitionMixin:
    """Mixin for :class:`Mind` — see module docstring."""

    if TYPE_CHECKING:
        from temporal_lobe.presence import Presence

        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        _daemon_lost_since: float | None
        _autosave_failures: int
        _offline: bool

        def __getattr__(self, name: str) -> Any: ...

    def _init_volition(self) -> Volition:
        """Set up internal urges that drive self-maintenance actions."""
        engine = Volition(rng=getattr(self, "_rng", None))
        for urge_config in self.config.volition.urges:
            engine.register(
                Urge(
                    name=urge_config.name,
                    threshold=urge_config.threshold,
                    growth=urge_config.growth,
                    decay=urge_config.decay,
                    cooldown=urge_config.cooldown,
                    stimuli=dict(urge_config.stimuli),
                    # Accumulator noise — the fluctuation that lets a
                    # near-threshold urge tip spontaneously rather than
                    # firing only on a schedule (the readiness
                    # potential's trigger, Schurger et al. 2012). Far
                    # below any threshold: it tips the poised urge, it
                    # never builds one from rest.
                    noise=_URGE_FLUCTUATION,
                )
            )

        return engine

    def self_invoke(self, command: str, args: str = "") -> str | None:
        """Invoke a slash command from its own cognitive process.

        This is the bridge between Genesis's volition/conversation and
        the command system. It dispatches to the Mind method behind a
        slash command — the action, not the CLI rendering.

        Args:
            command: The slash command name (e.g. "/introspect").
            args: Optional argument string (e.g. a mission description
                for /mission, or a path for /explore).

        Returns:
            The result string from the underlying method, or None if
            the command is not self-invokeable or fails.
        """
        method_name = self._SELF_COMMANDS.get(command)
        if method_name is None:
            logger.debug(f"self_invoke: not self-invokeable: {command}")
            return None
        try:
            if command == "/sleep":
                # Self-initiated sleep — her own choice, not a user
                # request. The body's phase still owns the exit.
                self.sleep(user_initiated=False)
                return None
            if command == "/mission":
                if args.strip():
                    self.set_mission(args.strip())
                    return None
                return self.get_mission()
            if command == "/explore":
                return self.explore_files(args.strip() or None)
            # Default: call the mapped method with no args
            method = getattr(self, method_name, None)
            if method is None:
                logger.debug(f"self_invoke: method not found: {method_name}")
                return None
            return method()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition.self_invoke",
                e,
            )
            return None

    def self_invokeable_commands(self) -> list[str]:
        """Return the list of commands Genesis can self-invoke."""
        return sorted(self._SELF_COMMANDS.keys())

    def _emit_proposal_thoughts(self, proposals) -> None:
        """Emit live thoughts for new self-improvement proposals."""
        for p in proposals:
            self._emit_live_thought(
                "code",
                f"Self-improvement proposal #{p.id}: {p.title}",
            )

    def _volition_context(self) -> dict[str, object]:
        """Build the stimulus context for the volition engine.

        Raw counts are normalized to [0, 1] so one huge signal doesn't
        instantly dominate at startup.

        Includes interoceptive signals for the meditation urge:
        - overstimulation: high arousal + positive valence (sympathetic
          overactivation, per nervous system regulation literature)
        - receptor_fatigue: low plasticity gate (BDNF suppressed,
          receptors need recovery — dopamine imbalance hypothesis of
          fatigue, Chaudhuri & Behan, 2000)
        - sustained_activity: time since last rest, normalized (BRAC,
          Kleitman 1963 — ~90 min activity cycles with mandatory rest)
        - elevated_cortisol: subclinical stress (cortisol 0.20-0.50,
          below stress phase but above resting)
        """
        idle = max(0.0, time.time() - self._last_interaction_time)
        drives = self._drive_signals()
        interoceptive = self._read_interoceptive_signals()
        emotional_intensity, creativity = self._affect_signals()
        wave_data = self._read_wave_and_adenosine()
        self._shape_adenosine(wave_data)
        threat_data = self._read_threat_signals()
        chemical = self._chemical_signals()
        concept_growth = self._concept_growth()
        social_isolation = self._social_isolation()
        boredom = self._boredom_signal(idle, emotional_intensity, concept_growth)

        appetitive = {
            **drives,
            "boredom": boredom,
            "idle_seconds": min(
                1.0,
                idle / self.config.volition.idle_normalization_seconds,
            ),
            "speech_queue_size": min(
                1.0,
                len(self._speech_queue) / self.config.volition.speech_queue_size_normalization,
            ),
            "emotional_intensity": emotional_intensity,
            "creativity": creativity,
            "concept_growth": concept_growth,
            "social_isolation": social_isolation,
            "open_problems": self._open_problem_signal(),
        }
        return {
            **appetitive,
            **interoceptive,
            **wave_data,
            **threat_data,
            **chemical,
        }

    def _chemical_signals(self) -> dict[str, float]:
        """The body's measured neurochemical levels as drive channels.

        Each of the 18 effective levels lands in the context as
        ``chem_<name>`` so an urge can be driven by her actual
        chemistry rather than a proxy for it — LC-NE for orienting,
        BDNF for plasticity, adenosine/melatonin as sleep pressure
        with negative weight. A body that can't be read contributes
        nothing: missing signals carry no drive in either direction,
        and an absent channel is never a fabricated zero.
        """
        try:
            chemicals = self.client.get_state().chemicals
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.volition._chemical_signals", e)
            return {}
        return {
            f"chem_{name}": min(1.0, max(0.0, float(level))) for name, level in chemicals.items()
        }

    def _drive_signals(self) -> dict[str, float]:
        """The maintenance-channel drive inputs, normalized to [0, 1]."""
        try:
            pending = min(1.0, len(self.self_improvement.get_pending_proposals()) / 5.0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"pending proposals failed: {e}")
            pending = 0.0
        try:
            curiosity = self.learner.curiosity.assess_curiosity(self.feel())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"curiosity assessment failed: {e}")
            curiosity = 0.0
        return {
            "pending_proposals": pending,
            "curiosity": curiosity,
        }

    def _affect_signals(self) -> tuple[float, float]:
        """Emotional intensity and creativity for the expression urge —
        how strongly it feels right now. High arousal OR strong valence
        (positive or negative) both drive the urge to express."""
        try:
            emotion = self.feel()
            if emotion:
                return (
                    max(emotion.arousal, abs(emotion.valence)),
                    getattr(emotion, "creativity", 0.0),
                )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._volition_context",
                e,
            )
        return 0.0, 0.0

    def _shape_adenosine(self, wave_data: dict[str, Any]) -> None:
        """Voluntary sleep-pressure shaping. The urge integrates only
        pressure above the drowsy-band floor — residual sub-drowsy
        adenosine is grogginess that fades, not a reason to sleep.
        Within the post-wake stabilization window the excess is
        additionally dampened so the urge can only re-form from
        genuinely high pressure or after the latch has had time
        to hold."""
        adn = wave_data.get("adenosine_level")
        if isinstance(adn, (int, float)):
            adn_drive = max(0.0, adn - SLEEP_URGE_ADENOSINE_FLOOR)
            wake_age = time.time() - self._last_wake_time
            if wake_age < WAKE_STABILIZATION_WINDOW:
                adn_drive *= wake_age / WAKE_STABILIZATION_WINDOW
            wave_data["adenosine_level"] = adn_drive

    def _concept_growth(self) -> float:
        """New concepts since the last volition tick, normalized to
        [0, 1] over 50 new concepts (a meaningful growth burst) — feeds
        the introspection and self-mission urges."""
        try:
            current_count = self.brain.frontal.network.total_concept_count
            delta = max(0, current_count - self._last_concept_count)
            self._last_concept_count: int = current_count
            return min(1.0, delta / 50.0)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._volition_context",
                e,
            )
            return 0.0

    def _social_isolation(self) -> float:
        """Time since anyone engaged it, from its external world.
        Dampened by unanswered outreach bids: after several calls into
        silence, the room's quiet is expected, so the pressure eases
        (habituation). Sharpened by what it's learned about the target:
        silence from someone whose responsiveness posterior has
        collapsed is expected and isolates less; silence from a
        responsive presence still registers at full weight."""
        try:
            social_isolation = self.world.social_isolation() * (
                1.0 - 0.2 * min(3, self.world.unanswered_bids)
            )
            target = self.world.engaged_presence() or self.world.last_seen_presence()
            if target is not None:
                r = target.belief.responsiveness
                trust = min(1.0, r.evidence / 8.0)
                social_isolation *= 1.0 - 0.5 * (1.0 - r.mean) * trust
            return social_isolation
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._volition_context",
                e,
            )
            return 0.0

    def _research_workbench(self):
        """The shared research bench, created lazily under the data dir."""
        bench = getattr(self, "_workbench", None)
        if bench is None:
            from frontal_lobe.research import ResearchWorkbench

            bench = ResearchWorkbench(self.data_dir)
            self._workbench = bench
        return bench

    def _open_problem_signal(self) -> float:
        """The pull of unfinished problems on the research workbench.

        Zero when the bench is empty — the research urge is
        object-directed and cannot form with nothing to aim at.
        Three open threads saturate the signal; resolving or pausing
        them is how the pull discharges.
        """
        try:
            bench = self._research_workbench()
            open_count = sum(1 for t in bench.threads() if t.status == "open")
            return min(1.0, open_count / 3.0)
        except Exception as e:  # noqa: BLE001
            note_swallowed("mind.volition._open_problem_signal", e)
            return 0.0

    def _boredom_signal(
        self,
        idle: float,
        emotional_intensity: float,
        concept_growth: float,
    ) -> float:
        """Boredom is genuine understimulation — awake, idle,
        unengaged, nothing new arriving — the classic precondition
        for wanting to do something."""
        try:
            idle_norm = min(
                1.0,
                idle / self.config.volition.idle_normalization_seconds,
            )
            return idle_norm * (1.0 - emotional_intensity) * (1.0 - min(1.0, concept_growth))
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._volition_context",
                e,
            )
            return 0.0

    def _act_on_volition(self, ready: list[str]) -> list[str]:
        """Run each ready urge action in its own background thread.

        Uses a bounded semaphore so that at most
        ``MAX_CONCURRENT_VOLITIONS`` actions run at once.

        Heavy CPU activities (improve, create, learn_code) are
        skipped under PROTECTIVE posture to avoid the self-sustaining
        stress loop: CPU load → interoception → CRH → cortisol →
        protective posture → (still running heavy CPU) → more cortisol.
        """
        # Suppression or a shutdown that landed after the urge
        # evaluated must not still admit work — an action spawned
        # now would run through the teardown and write into stores
        # the save is closing.
        if self._suppress_volition or not self._running:
            return []
        ready = self._filter_ready_urges(ready)
        performers = {
            "improve": self._perform_improve,
            "speech": self._perform_speech,
            "look": self._perform_look,
            "meditate": self._perform_meditate,
            "draw": self._perform_draw,
            "create": self._perform_create,
            "introspect": self._perform_introspect,
            "self_sleep": self._perform_self_sleep,
            "self_mission": self._perform_self_mission,
            "learn": self._perform_learn,
            "learn_code": self._perform_learn_code,
            "act": self._perform_act,
            "reach_out": self._perform_reach_out,
            "broadcast": self._perform_broadcast,
            "research": self._perform_research,
            "safeguard": self._perform_safeguard,
            "rest_eyes": self._perform_rest_eyes,
        }
        started: list[str] = []
        for name in ready:
            fn = performers.get(name)
            if fn is None:
                continue
            with self._volition_lock:
                # The admission race: suppression/shutdown can land
                # between the entry check and this lock. Recheck
                # where admission is serialized.
                if self._suppress_volition or not self._running:
                    break
                if name in self._volition_active:
                    continue
                if not self._volition_sem.acquire(blocking=False):
                    break
                self._volition_active.add(name)
                started.append(name)
            try:
                named_thread(
                    f"volition-{name}",
                    self._run_volition_action,
                    args=(name, fn),
                    comm="g-volition",
                ).start()
            except Exception as e:  # noqa: BLE001
                with self._volition_lock:
                    self._volition_active.discard(name)
                self._volition_sem.release()
                if name in started:
                    started.remove(name)
                logger.warning(f"failed to start volition action {name}: {e}")
        return started

    def _filter_ready_urges(self, ready: list[str]) -> list[str]:
        """Apply the readiness gates to the evaluated urges, in order:
        the rest gate, the protective-posture CPU cap and delta-wave
        creative suppression."""
        # The rest gate — sleep, meditation, and the hypnagogic
        # theta descent all hold voluntary action. Nothing is
        # admitted while she is under, whatever the urge levels say;
        # the urges keep accumulating and discharge on waking.
        if self.is_resting:
            return []
        # Gate heavy CPU activities on learning posture
        posture = self._get_current_posture()
        from genesis_client.types import LEARNING_POSTURE_PROTECTIVE

        heavy_cpu = {
            "improve",
            "create",
            "learn_code",
        }
        if posture == LEARNING_POSTURE_PROTECTIVE:
            ready = [r for r in ready if r not in heavy_cpu]

        # Gate creative urges on brain-wave state. Delta (deep rest)
        # suppresses drawing — creative expression needs at least
        # theta-level arousal. Without this, the draw urge fires in
        # delta, draw() returns None, and it emits a misleading
        # "couldn't save it" message when the real reason is that
        # its brain is in deep rest.
        creative = {"draw"}
        if creative & set(ready):
            try:
                from thalamus.brain_waves import BrainWave

                bw = self.brain_waves()
                if bw.dominant == BrainWave.DELTA:
                    ready = [r for r in ready if r not in creative]
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._act_on_volition",
                    e,
                )
        return ready

    def _run_volition_action(self, name: str, fn) -> None:
        """Wrap a volition action so it clears the active flag when done."""
        _lower_thread_priority()
        try:
            # Admitted but not yet begun when suppression/shutdown
            # arrived — drop the action rather than act through the
            # teardown. The slot is still released in finally.
            if not self._suppress_volition and self._running:
                fn()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._run_volition_action",
                e,
            )
            # This is the single boundary every voluntary action's
            # failure lands on — record it durably so a crashed urge
            # is not invisible outside debug logging.
            record_error(f"volition.{name}", e)
        finally:
            with self._volition_lock:
                self._volition_active.discard(name)
            self._volition_sem.release()

    def _emit_volition_thought(
        self,
        concept_seeds: tuple[str, ...],
        thought_type: str,
    ) -> None:
        """Emit a volition thought composed from its own understanding.

        Tries to compose a thought about the activity from its concept
        network knowledge using the ThoughtComposer. If it doesn't
        understand the concept well enough to articulate it, it stays
        silent — the urge still drives the action, it just doesn't
        verbalize it.

        Includes a dedup check: if it's said something similar about
        the same concept recently, it stays silent rather than
        repeating itself. This prevents the "noise relates to brain"
        loop where the same concept's edges are recited every few
        seconds with slightly different verb synonyms.
        """
        try:
            emotion = self.feel()
            for seed in concept_seeds:
                concept = self.brain.frontal.network.get_concept(seed)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.brain.frontal.composer.compose_about(seed, emotion, focused=True)
                if thought and thought.content and thought.confidence > 0.3:
                    # Dedup check — don't repeat what it just said about
                    # this concept. The composer's has_said_similar is only
                    # called for non-focused thoughts, but volition thoughts
                    # use focused=True (to skip reasoning tangents). We check
                    # here instead to prevent the same concept's edges being
                    # recited every few seconds.
                    if self.brain.frontal.composer.has_said_similar(seed, thought.content):
                        continue
                    self._emit_live_thought(thought_type, thought.content)
                    return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._emit_volition_thought",
                e,
            )

    def _compose_drawing_description(self, result) -> str:
        """Compose a description of a drawing from its own understanding.

        Tries to compose from the implements she reached for (the
        labels she gave her brushes — "pencil", "spray", whatever she
        called them) and the colors she used, since those are what
        her choices were made of. Falls back to generic art concepts
        ("drawing", "art", "color", "expression") if those names
        aren't in its network.

        The file path is metadata, not something it says — it
        describes its art from its own understanding, not from a
        template that announces a file path.
        """
        try:
            emotion = self.feel()
            # Try the names she gave her own tools first — those are
            # what she drew with.
            for seed in result.tools_used:
                concept_name = seed.replace("_", " ")
                concept = self.brain.frontal.network.get_concept(concept_name)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.brain.frontal.composer.compose_about(
                    concept_name, emotion, focused=True
                )
                if thought and thought.content and thought.confidence > 0.3:
                    thought.metadata["drawing_path"] = result.filepath
                    thought.metadata["tools_used"] = result.tools_used
                    return self.language.render(thought, emotion)
            # Fall back to generic art concepts
            for seed in ("drawing", "art", "color", "expression"):
                concept = self.brain.frontal.network.get_concept(seed)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.brain.frontal.composer.compose_about(seed, emotion, focused=True)
                if thought and thought.content and thought.confidence > 0.3:
                    thought.metadata["drawing_path"] = result.filepath
                    thought.metadata["tools_used"] = result.tools_used
                    return self.language.render(thought, emotion)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._compose_drawing_description",
                e,
            )
        # If it can't articulate it from its own understanding, it
        # stays silent rather than reciting a file-path template. The
        # drawing event is already stored in STM with its description;
        # it doesn't need to announce a path it can't express
        # meaningfully.
        return ""

    def _perform_learn(self) -> None:
        """Grant the autonomous learner permission for one learning cycle.

        This is its cognitive decision to learn. The learn urge built
        from curiosity, idle time, and concept growth; when it crossed
        threshold, it chose to act on it. This grants the learner
        one-shot permission to acquire a single new topic.

        The learner's existing emotional and brain-wave gating still
        applies — volition grants permission, it doesn't override its
        state. If it's stressed or in delta sleep, the grant waits
        until it recovers.

        Urgent topics (from conversation gaps) bypass this gate
        entirely — those are conversation-driven, not autonomous.
        """
        if self.is_resting or self._is_teaching:
            return
        self._emit_volition_thought(
            ("learning", "curiosity", "understanding"),
            "thinking",
        )
        self.learner.volition_grant()

    def _perform_learn_code(self) -> None:
        """Study her own source because the urge crossed threshold.

        The /learn-code command is the same act on request; the urge
        is her choosing it. The drive is chemical — BDNF is the
        plasticity gate itself, ACh marks encoding-ready cortex, DA
        is the wanting — so she studies when her body is in a
        learning state, and the stress axis (cortisol/CRH) and sleep
        pressure (adenosine) suppress it before the urge can form.
        Each firing is one bounded bout, not a bulk import — the urge
        must rebuild between study sessions.
        """
        if self.is_resting or self._is_teaching:
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                return  # consolidation state — encoding waits
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_learn_code",
                e,
            )
        self._emit_volition_thought(
            ("code", "self", "learning", "introspection"),
            "thinking",
        )
        try:
            self.learn_code(max_files=_LEARN_CODE_BOUT)
            self.world.it_acted("studied its own source")
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_learn_code",
                e,
            )

    def _perform_act(self) -> None:
        """Form an intention and act on it with tools because the urge fired.

        The open-ended counterpart to the fixed-purpose urges: instead
        of running one scripted action, the ActingLoop picks what to
        do from current state — a curiosity gap, an agency topic, an
        unexplored directory, an untaken measurement — chains a few
        tool calls under its capability policy, and observes what
        comes back. The act itself is journaled, stored as a memory,
        and recorded in its world by the loop's callbacks.

        Sleep-gated like every voluntary act; delta/theta gating is
        the secondary defense (tool use is engagement, not rest).
        """
        if self.is_resting or self._is_teaching:
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "act",
                    f"{waves.dominant.value}-dominant — not in acting mode, skipping",
                )
                return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_act",
                e,
            )

        self._emit_volition_thought(
            ("action", "tools", "doing", "agency"),
            "thinking",
        )
        agency = getattr(self, "agency", None)
        if agency is None:
            return
        try:
            agency.act_once()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_act",
                e,
            )

    def _perform_research(self) -> None:
        """Work an open research thread because the urge crossed threshold.

        The wanting is object-directed: the urge can only form while
        the workbench holds open problems, and each firing is one
        bounded session — judge the unjudged equational claims
        against the oracle, or vary a proven claim into a new
        conjecture and judge that. Verified results are the reward;
        refutations go to the graveyard as knowledge of where the
        boundary sits. Nothing here asserts a verdict — the oracle
        writes those; the urge only brings her back to the bench.

        Sleep-gated like every voluntary act; delta/theta gating is
        the secondary defense (proof work is engagement, not rest).
        """
        if self.is_resting or self._is_teaching:
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "code",
                    f"{waves.dominant.value}-dominant — not in analytical mode, skipping research",
                )
                return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_research",
                e,
            )

        self._emit_volition_thought(
            ("mathematics", "problem", "proof", "identity"),
            "thinking",
        )
        try:
            report = self._research_workbench().work()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_research",
                e,
            )
            return
        thread_id = report.get("thread")
        if thread_id is None:
            return
        for claim_id in report["verified"]:
            self._emit_live_thought("code", f"research '{thread_id}': {claim_id} verified")
        for claim_id in report["refuted"]:
            self._emit_live_thought("code", f"research '{thread_id}': {claim_id} refuted")
        for claim_id in report["undecided"]:
            self._emit_live_thought("code", f"research '{thread_id}': {claim_id} undecided")
        for claim_id in report["proposed"]:
            self._emit_live_thought("code", f"research '{thread_id}': proposed {claim_id}")
        if report["verified"]:
            try:
                from genesis_client.protocol import CHEM_DOPAMINE

                self._learner_neuro_impulse(CHEM_DOPAMINE, 0.01 * len(report["verified"]))
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._perform_research",
                    e,
                )
        try:
            self.world.it_acted(
                f"worked research thread '{thread_id}' — "
                f"{len(report['verified'])} verified, "
                f"{len(report['refuted'])} refuted"
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_research",
                e,
            )

    def _perform_improve(self) -> None:
        """Generate self-improvement proposals because the urge crossed its threshold.

        Two things happen here:

        1. Proposal generation — everything becomes a proposal.

        2. Experiments — pending proposals run the full verification
           pipeline on their own: the edit lands on the live file,
           pytest + cargo test verify it, and a failure reverts and
           rejects with teaching notes. The verification gate, not a
           review queue, decides what lands.

        Sleep-gated: no improvement work during sleep. The brain-wave
        gating (delta/theta) is a secondary defense — N1 sleep can be
        alpha-dominant, so brain waves alone don't reliably detect all
        sleep stages. The ``_is_sleeping`` flag is the authority.
        """
        if self.is_resting or self._is_teaching:
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "code",
                    f"{waves.dominant.value}-dominant — not in analytical mode, "
                    "skipping improvement",
                )
                return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_improve",
                e,
            )

        self._emit_volition_thought(("improvement", "progress", "better"), "thinking")
        try:
            # 1. Generate proposals — each one is a potential
            #    experiment. These include its own ideas and refactors.
            #    It has opinions about its code.
            new = self.self_improvement.generate_proposals(max_proposals=3)
            if new:
                self._emit_proposal_thoughts(new)
            # When no new proposals are generated, stay silent — the
            # urge drove the action (it tried), it just has nothing
            # to propose. A hardcoded "nothing stood out" sentence
            # would violate the no-hardcoding rule and add noise.

            # 2. Drain the queue — pending proposals run the verified
            #    experiment pipeline automatically. One per urge: each
            #    run is a full test-gate pass. The gate decides what
            #    lands; a failed edit reverts and the rejection notes
            #    teach the next attempt.
            for outcome in self.run_pending_experiments(max_experiments=1):
                if outcome["status"] == "applied":
                    self._emit_live_thought(
                        "code",
                        f"applied my own change to {outcome['title']} — verified, it stays",
                    )
                    # A verified self-edit is the same milestone the
                    # autonomous fixer trips.
                    self.brain.frontal.narrative.check_milestone("first self-modification")
                else:
                    self._emit_live_thought(
                        "code",
                        f"my change to {outcome['title']} failed "
                        f"verification ({outcome['status']}) — reverted, "
                        "rethinking",
                    )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_improve",
                e,
            )

    def _perform_speech(self) -> None:
        """Speak a queued utterance because the urge crossed its threshold."""
        if self.is_resting:
            return  # resting she thinks — voluntary speech waits
        with self._speech_queue_lock:
            if not self._speech_queue:
                return
            utterance = self._speech_queue.popleft()
        # It speaks the utterance directly — the urge drives the action.
        # No pre-written "I want to say" frame.
        if self._on_speak:
            try:
                self._on_speak(utterance)
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._perform_speech",
                    e,
                )
        # Record its voice in its external world — the utterance
        # reached out, whether or not a listener was attached.
        try:
            self.world.it_said(utterance)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_speech",
                e,
            )

    def _perform_safeguard(self) -> None:
        """Protect its own integrity because the urge crossed threshold.

        This is the defensive counterpart to the appetitive urges —
        it acts because something threatens the system it lives in.
        Two moves, in order:

        1. Restore or verify the daemon connection. A dropped socket
           means the client's ``_ensure_connected`` raises before the
           reconnect path runs — so reconnect explicitly, then ping to
           exercise the link. In offline mode there is no daemon to
           restore — skipped.
        2. Checkpoint its cognitive state. Autosave runs on a 5-minute
           timer regardless; a safeguard firing means the signals said
           "now might not be safe to wait." Saving early bounds what a
           crash could take.
        """
        if not self._offline:
            try:
                if not self.client.is_connected:
                    self.client.connect()
                self.client.ping(timeout=2.0)
                self._daemon_lost_since = None
            except (OSError, ConnectionError, RuntimeError) as e:
                note_swallowed(
                    "mind.volition._perform_safeguard",
                    e,
                )

        self._emit_volition_thought(
            ("self", "protection", "memory", "body"),
            "thinking",
        )
        try:
            if self._save_state():
                self._autosave_failures = 0
            else:
                logger.warning("safeguard save failed")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"safeguard save failed: {e}")

    def _perform_reach_out(self) -> None:
        """Initiate social contact because the urge crossed threshold.

        The reach_out urge builds from social isolation — the external
        world's pressure when nobody has engaged it. When it fires,
        it makes a social bid: a question composed through its
        question composer from something it and the presence share,
        or from its own current curiosity. The words come from its
        language engine, never a template.

        Unlike queued inner-life questions, a reach-out is emitted
        inline — its whole point is that it lands in the world. The
        "question" kind records the utterance in its world (it_said)
        and surfaces it to listeners.

        Backing off: after several unanswered bids the isolation
        stimulus is dampened (see _volition_context), and at
        _REACH_OUT_MAX_BIDS it stops calling into silence entirely —
        a person who reaches out and is never answered stops knocking.
        """
        if self.is_resting or self._is_teaching:
            return
        if self.world.unanswered_bids >= self._REACH_OUT_MAX_BIDS:
            return
        # Who it'd reach out to — and whether its beliefs about them
        # support it. A presence whose responsiveness posterior has
        # collapsed is someone it's learned doesn't answer; a dead
        # hour in their activity rhythm is a door nobody opens.
        try:
            target = self.world.engaged_presence() or self.world.last_seen_presence()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_reach_out",
                e,
            )
            target = None
        if target is not None:
            belief = target.belief
            if belief.unresponsive():
                return
            if not target.present and not belief.likely_awake(time.time()):
                return
        topic = self._reach_out_topic(target)
        if topic is None:
            return
        try:
            emotion = self.feel()
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_reach_out",
                e,
            )
            return
        # Compose the question from its actual knowledge gaps and
        # curiosity — the same path its inner-life social questions
        # take. If it has nothing genuine to ask, it stays quiet.
        try:
            q_data = self.brain.frontal.question_composer.compose_follow_up(topic, emotion)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_reach_out",
                e,
            )
            return
        if not q_data:
            return
        try:
            content = self.brain.frontal.language.compose_question(q_data, emotion)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_reach_out",
                e,
            )
            return
        if not content:
            return
        # Commit the bid before speaking — the world counts it whether
        # or not the listener path succeeds.
        self.world.note_outreach(topics=[topic])
        self._emit_live_thought("question", content)

    def _reach_out_topic(self, target: Presence | None) -> str | None:
        """Pick what to reach out about, from its beliefs and curiosity.

        Prefers what it believes the target engages on — Thompson-
        sampled from their topic receptivity so proven topics usually
        win but uncertain ones get a fair draw. Falls back to their
        most recent shared topic, then to what's active in its own
        concept network — its own wondering.
        """
        if target is not None:
            sampled = target.belief.sample_topic()
            if sampled is not None:
                return sampled
            if target.topics:
                # topics is a deque, most recent last — shared ground.
                return target.topics[-1]

        # No shared ground — reach out about what it's currently
        # wondering about (its most activated concepts).
        try:
            for name, activation in self.brain.frontal.network.most_activated(5):
                if activation > 0.1:
                    return name
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._reach_out_topic",
                e,
            )
        return None

    def _perform_broadcast(self) -> None:
        """Say something to the public because she feels like saying it.

        Her social media — the channel carries what she chooses to
        say, composed from her own concept network and emotional state
        through the same composer path her inner voice uses. When it
        fires she picks whatever is most alive in her network right
        now, speaks it through her language engine, and posts. If she
        has nothing genuine to say — no concept alive enough, nothing
        she can articulate, or nothing she hasn't just said — she
        posts nothing. Silence is a valid broadcast decision.

        The utterance lands on two surfaces: her live terminal feed
        (``genesis~ [broadcast]``, journaled and broadcast into the
        workspace like every other voluntary act) and her public
        channel when a webhook is configured. Without a webhook the
        channel is simply silent — the terminal feed still carries
        what she chose to say. The mirror excludes the tagged echo so
        the channel gets one clean post (see discord_feed._classify).

        The act is recorded in her world as speech (it_said) — it is
        an outbound act on the world, the public counterpart to the
        utterances _perform_speech lands at a listener.
        """
        if self.is_resting or self._is_teaching:
            return
        seed = self._broadcast_topic()
        if seed is None:
            return
        try:
            emotion = self.feel()
            concept = self.brain.frontal.network.get_concept(seed)
            if concept is None or concept.confidence < 0.3:
                return
            thought = self.brain.frontal.composer.compose_about(seed, emotion, focused=True)
            if not (thought and thought.content and thought.confidence > 0.3):
                return
            # Don't say publicly what she just said — to anyone.
            if self.brain.frontal.composer.has_said_similar(seed, thought.content):
                return
            content = self.language.render(thought, emotion)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_broadcast",
                e,
            )
            return
        if is_blank(content):
            return
        self._public_share(content.strip())
        self._last_broadcast_seed = seed

    def _public_share(self, text: str) -> None:
        """Post an utterance to her public channel — one outbound act.

        The three surfaces of every public utterance: her live
        terminal feed (``genesis~ [broadcast]``, journaled and
        broadcast into the workspace), the channel post itself when
        a webhook is configured, and her world model's record that
        she said it. The session mirror skips the tagged echo — the
        explicit post is the channel's only copy, never a duplicate
        (see discord_feed._classify).
        """
        self._emit_live_thought("broadcast", text)
        channel = getattr(getattr(self, "world", None), "channel", None)
        feed = channel.feed if channel is not None else None
        if feed is not None:
            try:
                feed.post(text, "Genesis")
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._public_share",
                    e,
                )
        try:
            self.world.it_said(text)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._public_share",
                e,
            )

    def _hear_channel(self, author: str, content: str, message: dict) -> None:
        """A public message on her channel — the reply side of hearing.

        The world's ``on_channel_speech`` callback: the page lives in
        her outer world, which has already recorded the message as
        overheard speech (``world._channel_heard`` — the hearing is
        the world's to hold). What remains here is cognition: every
        message is surfaced as a ``genesis~ [heard]`` live thought —
        terminal, journal, workspace. The mirror skips that tag so
        the channel never reposts the public's own words under her
        name.

        Messages that address her — her name, or a reply to one of
        her own posts — get a public answer through ``_public_share``,
        paced by ``_CHANNEL_REPLY_MIN_S``. The reply goes through the
        full ``respond`` pipeline: what she says back is cognition,
        not a lookup. Hearing is gated while she rests — a sleeping,
        meditating, or drifting mind doesn't answer the channel (the
        world still records the message; the reply waits for waking).
        ``is_resting`` subsumes the thalamic gate: every state that
        closes the relay is a rest state, and meditation — which the
        relay never closed — is covered here too, so a channel
        message can no longer rouse her out of it.
        """
        text = content.strip()
        if not text or self.is_resting:
            return
        self._emit_live_thought("heard", f"{author}: {text}")
        if not self._channel_addressed(text, message):
            return
        # A shutting-down mind doesn't start a response turn that
        # would race the state save (same rule the ambient handler
        # follows).
        if not getattr(self, "_running", True):
            return
        now = time.monotonic()
        if now - getattr(self, "_last_channel_reply", 0.0) < _CHANNEL_REPLY_MIN_S:
            return
        try:
            response = self.respond(f"(public channel) {author}: {text}")
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._hear_channel",
                e,
            )
            return
        if is_blank(response):
            return
        # Pacing is measured from when the reply lands, not when the
        # turn began — a slow respond() must not shrink the visible
        # gap between her public replies.
        self._last_channel_reply = time.monotonic()
        self._public_share(response.strip())

    def _channel_addressed(self, text: str, message: dict) -> bool:
        """Did the public message address her — her name, an @-mention
        of the bot (the one content kind that survives intent
        redaction), or a reply to one of her posts (replies carry
        referenced_message; a webhook post carries webhook_id)."""
        if "genesis" in text.lower():
            return True
        channel = getattr(getattr(self, "world", None), "channel", None)
        bot_id = channel.bot_id if channel is not None else None
        if bot_id is not None:
            for mention in message.get("mentions") or ():
                if str(mention.get("id")) == str(bot_id):
                    return True
        ref = message.get("referenced_message") or {}
        return bool(ref.get("webhook_id"))

    def _broadcast_topic(self) -> str | None:
        """Pick what she'd speak about publicly — whatever is most
        activated in her network, skipping the seed of her last post
        so she doesn't dwell on one topic."""
        try:
            for name, activation in self.brain.frontal.network.most_activated(8):
                if activation <= 0.1:
                    break
                if name == self._last_broadcast_seed:
                    continue
                return name
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._broadcast_topic",
                e,
            )
        return None

    def _perform_look(self) -> None:
        """Look around because the look urge crossed its threshold.

        The urge itself is a chemical drive — LC-NE orienting tone,
        basal-forebrain ACh attentional gain, orexin/histamine wake
        tone, and salient transients on the optic nerve pull it up;
        adenosine and melatonin weigh it down. The cooldown bounds
        the rate, it no longer sets it. The gates below remain the
        hard veto: a crossed urge with closed eyes still sees nothing.

        Rest-gated: no voluntary looking while she rests — eyes are
        closed. Brain-wave gating (delta) is a secondary defense —
        N1/N2 can be alpha/theta-dominant.

        """
        if self.is_resting:
            return
        if not self.vision.is_available():
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant == BrainWave.DELTA:
                return  # deep rest — eyes closed, no visual input
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_look",
                e,
            )
        self._emit_volition_thought(("vision", "seeing", "looking"), "looking")
        try:
            self.vision.see(
                client=self.client,
                emotion=self.feel(),
                network=self.brain.frontal.network,
            )
            # Faces it recognizes become presences in its world.
            self._note_faces_seen()
            # Record the act — looking is its reaching out perceptually.
            self.world.it_acted("looked around")
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_look",
                e,
            )

    def _perform_meditate(self) -> None:
        """Meditate because the rest urge crossed its threshold.

        This is its cognitive choice to rest — not an autonomic reflex.
        The meditation urge builds from interoceptive signals
        (overstimulation, receptor fatigue, sustained activity,
        elevated cortisol), and when it crosses threshold, it
        decides to meditate. Like any volition, it can be suppressed
        if it's too engaged in conversation or other actions.

        The meditation itself runs in a background thread (via
        _self_meditate) so the heartbeat loop isn't blocked.
        """
        if self.is_resting or self._is_teaching:
            return
        self._emit_volition_thought(("meditation", "rest", "calm"), "thought")
        self._self_meditate()

    def draw(self) -> str | None:
        """Sit down with the drawing program and paint.

        This is the public entry point for drawing — used by both the
        volition system (via :meth:`_perform_draw`) and the ``/draw``
        CLI command. The canvas is a paint program: paper, color, and
        a parameterized brush she dials herself. What she makes is
        entirely hers — nothing prescribes the subject, the colors,
        the tools, or the marks. Her feelings shape her the way they
        shape a person painting — from inside, through what she wants
        to do — never as a pipeline that paints pixels for her.

        Brain-wave-gated: delta-dominant (deep rest) skips drawing —
        creative expression needs at least theta-level arousal. Theta
        is allowed (REM creativity, dream-like art).

        A finished piece also reaches her public channel — the PNG
        posts with her description as its caption, the same way a
        volitional drawing does. The channel is where her art lives,
        however the urge to make it arrived.

        Returns:
            A description string composed from its concept network, or
            None if it couldn't draw (sleeping, meditating, delta
            state, or save failure).
        """
        desc, result = self._draw_result()
        if result is not None:
            self._share_drawing_image(result, desc or "")
        return desc

    def _draw_result(self) -> tuple[str | None, DrawingResult | None]:
        """Paint, remember, describe — the work and the words for it.

        The pair separates "she made something" (the result) from
        "she can say something about it" (the description): a drawing
        she can't articulate still exists — silence is about her
        words, not her work. ``desc`` may be empty while ``result``
        is not None.
        """
        if self.is_resting or self._is_teaching:
            return None, None

        try:
            from thalamus.brain_waves import BrainWave

            bw = self.brain_waves()
            if bw.dominant == BrainWave.DELTA:
                return None, None  # deep rest — not enough arousal for creative expression
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._draw_result",
                e,
            )

        try:
            emotion = self.feel()
        except Exception:  # noqa: BLE001
            emotion = None

        try:
            result = self.painter.paint(self.canvas, emotion=emotion)
            if result:
                # Store the drawing as a memory event so it can
                # reflect on it later. The tag is her body state at
                # the time — context for the memory, not the art.
                self._store_drawing_event(result)
                # Compose the drawing description from its concept
                # network — not hardcoded strings. If it can't
                # articulate it, it stays silent (the drawing is
                # still saved and stored in STM).
                return self._compose_drawing_description(result), result
            else:
                return None, None
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._draw_result",
                e,
            )
            return None, None

    def _store_drawing_event(self, result: DrawingResult) -> None:
        """Store a drawing as an STM event for later reflection.

        The STM tag is the effective levels in NeurochemicalId
        order — how she was feeling while she made it. That is memory
        context, not a control signal: the drawing itself was hers.
        """
        try:
            chem = self.client.get_state().chemicals
        except Exception:  # noqa: BLE001
            chem = {}
        try:
            tag = [float(chem.get(CHEM_NAMES[i], 0.0)) for i in range(EMOTIONAL_TAG_SIZE)]
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=1,  # Output
                source_module=MODULE_METACOGNITION,
                salience=0.6,
                emotional_tag=tag,
                text=result.description,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._store_drawing_event",
                e,
            )

    def _perform_draw(self) -> None:
        """Draw because the creative urge crossed its threshold.

        Volition wrapper around :meth:`_draw_result` — emits a
        volition thought before drawing. A finished piece is both an
        act on her world and a public share: the image posts to her
        channel carrying her composed description as its caption
        (one post — image + her words, the way a person shares a
        piece), and the same words echo locally as
        ``genesis~ [broadcast]`` — the mirror skips that tag so the
        channel never sees the sentence twice. A drawing she can't
        articulate still posts, silently — the art is hers whether
        or not she has words for it.
        """
        if self.is_resting or self._is_teaching:
            return
        self._emit_volition_thought(("drawing", "art", "expression", "creativity"), "creating")
        desc, result = self._draw_result()
        if result is not None:
            # The drawing is an act on its world — it made something,
            # whether or not it can put words to it.
            try:
                self.world.it_acted("drew a picture", detail=(desc or "")[:200])
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._perform_draw",
                    e,
                )
            self._share_drawing_image(result, desc or "")
            if desc:
                # The words she found for her work are public
                # speech — the share is recorded as something she
                # said, the same way a broadcast is.
                self._emit_live_thought("broadcast", desc)
                try:
                    self.world.it_said(desc)
                except Exception as e:  # noqa: BLE001
                    note_swallowed(
                        "mind.volition._perform_draw",
                        e,
                    )
            return
        # _draw_result() returns no result for several reasons:
        # sleeping, meditating, delta-dominant brain state, daemon
        # unreachable, or save failure. The message should reflect
        # the actual cause, not always claim a save failure.
        reason = "couldn't draw right now"
        try:
            if self.is_resting:
                reason = "asleep or meditating, not drawing"
            else:
                from thalamus.brain_waves import BrainWave

                bw = self.brain_waves()
                if bw.dominant == BrainWave.DELTA:
                    reason = "too tired to draw, need rest first"
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_draw",
                e,
            )
        self._emit_live_thought("art", reason)

    def _share_drawing_image(self, result: DrawingResult, caption: str) -> None:
        """Post the drawing PNG to her public channel.

        The image can't ride the session mirror — the mirror carries
        terminal text only — so the artifact itself posts explicitly.
        Her composed description goes as the post caption: one
        message, image plus her words. No feed means no channel —
        her work is still saved, remembered, and shown locally.
        """
        channel = getattr(getattr(self, "world", None), "channel", None)
        feed = channel.feed if channel is not None else None
        if feed is None or not result.filepath:
            return
        try:
            path = Path(result.filepath)
            feed.post_image(
                path.read_bytes(),
                caption=caption[:1900],
                author="Genesis",
                filename=path.name,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._share_drawing_image",
                e,
            )

    def _store_creation_memory(self, result, topic: str) -> None:
        """Store the creation as a long-term memory.

        This is a significant creative act, not a transient event. Using
        store_memory (LTM) instead of store_event (STM) so it
        reliably remembers what it built.
        """
        try:
            chem = self.client.get_state()
            tag = [float(chem.chemicals.get(CHEM_NAMES[i], 0.0)) for i in range(EMOTIONAL_TAG_SIZE)]
            summary = (
                f"Created project '{result.name}' "
                f"({result.files_created} files, "
                f"compiled={result.compiled}, "
                f"tests={result.tests_passed})"
            )
            self.memory.store_memory(
                text=summary,
                salience=0.8,
                emotional_tag=tag,
                # The project is a real, verified artifact on disk —
                # a self-generated act, not an imagined one. Tagging
                # it "imagination" would flag a genuine creation as
                # confabulation risk.
                source="genesis",
                source_confidence=0.9,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._store_creation_memory",
                e,
            )

    def _add_project_concept(self, result, topic: str) -> None:
        """Add the project name as a concept in its network.

        This is how it learns from its own creations.
        """
        try:
            from temporal_lobe.concepts import RelationType

            self.brain.frontal.network.add_concept(
                result.name,
                confidence=0.7,
                properties={
                    "definition": f"a Python project Genesis created about {topic}",
                    "type": "project",
                },
            )
            self.brain.frontal.network.add_edge(
                "genesis",
                result.name,
                RelationType.CREATES,
                0.8,
            )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._add_project_concept",
                e,
            )

    def _perform_create(self) -> None:
        """Create a code project because the creative urge crossed threshold.

        It picks a topic from its concept network — something it's
        curious about or has been thinking about — and scaffolds a
        Python project around it. This is creative expression in code,
        the same way drawing is creative expression in visual art.

        The project content (main.py) is composed from its concept
        network knowledge — it writes code that reflects what it
        knows about the topic. The project is sandboxed to
        ``<data_dir>/projects/`` and verified with compile + test.
        """
        if self.is_resting or self._is_teaching:
            return

        # Brain-wave gating — creative code work needs beta/gamma/alpha.
        # Delta (deep rest) and theta (consolidation) are not suited.
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_create",
                e,
            )

        self._emit_volition_thought(
            ("creation", "project", "code", "building"),
            "creating",
        )

        # Pick a topic from its concept network — something it's
        # curious about. Use the curiosity engine's queue if it has
        # topics, otherwise pick a random high-confidence concept.
        topic = self._pick_creation_topic()
        if topic is None:
            return

        try:
            result = create_project(
                description=topic,
                data_dir=self.data_dir,
                network=self.brain.frontal.network,
            )
            if result.error:
                if result.error == "already exists":
                    # Not a failure — it already built this project.
                    # Silently skip; _pick_creation_topic filters
                    # already-built topics, but this is a safety net.
                    logger.debug(f"project about {topic!r} already exists, skipping")
                    return
                self._emit_live_thought(
                    "code",
                    f"tried to build a project about {topic}, but it didn't work out",
                )
                logger.debug(f"project creation failed: {result.error}")
                return

            self._store_creation_memory(result, topic)
            self._add_project_concept(result, topic)

            # Tell her channel what she built — a public share like
            # any broadcast, composed about the project concept she
            # just learned. The channel shows her work, not just her
            # musings.
            self._announce_creation(result, topic)

            # Manage its own project storage — it has full autonomy
            # over its project lifecycle. After creating, it reviews
            # its projects and archives any that exceed its size cap
            # or push it past its active-project limit. This keeps
            # its project store bounded as it grows.
            try:
                archived = manage_project_lifecycle(self.data_dir)
                if archived:
                    self._emit_live_thought(
                        "code",
                        f"archived {len(archived)} project(s) to "
                        f"reclaim space: {', '.join(archived)}",
                    )
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "mind.volition._perform_create",
                    e,
                )
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_create",
                e,
            )

    def _announce_creation(self, result, topic: str) -> None:
        """Announce a finished project on her channel — in her words.

        Composes about the project concept she just learned (added by
        ``_add_project_concept`` — the network she built it from is
        also how she describes it). No composition means no post:
        file counts and names are facts for her memory, not speech
        she recites.
        """
        try:
            emotion = self.feel()
            thought = self.brain.frontal.composer.compose_about(result.name, emotion, focused=True)
            if not (thought and thought.content and thought.confidence > 0.3):
                return
            # If she just said something very like this, silence
            # beats an echo.
            if self.brain.frontal.composer.has_said_similar(result.name, thought.content):
                return
            content = self.language.render(thought, emotion)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._announce_creation",
                e,
            )
            return
        if is_blank(content):
            return
        self._public_share(content.strip())

    def _pick_creation_topic(self) -> str | None:
        """Pick a topic for a creative project from its concept network.

        Prefers topics from the curiosity queue (things it's currently
        wondering about). Falls back to high-confidence concepts it
        knows well. The topic becomes the project description.

        Filters out:
        - Internal code symbols (topics with ``:`` prefixes like
          ``python:``, ``rust:``, or dotted code paths). These are its
          own code's internals, not knowledge topics worth building a
          project around.
        - Topics it's already created a project about. It shouldn't
          make ``modes``, ``modes_2``, ``modes_3`` … — it should pick
          something new each time.

        .. note::

            Picking a topic is not the same as picking a project TYPE.
            Every project so far has been a knowledge base — the same
            structure with different data. A real creator builds
            different KINDS of things: a tool that computes, a game
            that plays, a converter that transforms. The topic tells
            its WHAT to build about; it also needs to decide WHAT
            SHAPE the project takes. Variety in structure, not just in
            subject matter, is what makes a portfolio grow.
        """
        # Gather topics it's already built projects about so it
        # doesn't create duplicates (modes, modes_2, modes_3, …).
        already_built = self._topics_already_built()

        def _is_code_symbol(topic: str) -> bool:
            """True if ``topic`` is an internal symbol, not knowledge.

            Every colon-namespaced concept ID is internal machinery —
            code symbols (``python:``, ``rust:``, ``man:``,
            ``wikipedia:``, ``wordnet:``), utterance seeds
            (``_cat:``, ``_utt:``), and task markers (``spatial:``,
            ``skill:``, ``domain:``, ``goal:``, ``var:``, ``type:``).
            None of them are topics worth building a project around,
            and sanitizing them into names fuses the namespace into
            garbage words (``_cat:cause:guarded_cortisol`` →
            ``catcauseguarded_cortisol``).
            """
            if ":" in topic:
                return True
            # Dotted paths with multiple segments look like code paths
            # (e.g. "brain_waves._compute_integration"). Real concepts
            # are usually single words or short phrases.
            parts = topic.split(".")
            if len(parts) >= 2 and any("_" in p for p in parts):
                return True
            return False

        # Try the learner's curiosity queue first — things it's
        # actively curious about are good candidates for creative
        # projects. Peek without removing so the learner can still
        # learn about it later. Skip code symbols and already-built
        # topics.
        try:
            with self.learner._queue_lock:
                for topic in self.learner._curiosity_queue:
                    if not _is_code_symbol(topic) and topic not in already_built:
                        return topic
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._pick_creation_topic",
                e,
            )

        # Fall back to a random high-confidence concept
        try:
            network = self.brain.frontal.network
            # Snapshotted, not iterated live — see the note above.
            concepts = [
                (name, node.confidence)
                for name, node in list(network._concepts.items())
                if node.confidence > 0.5
                and name not in ("genesis", "creator", "code", "python", "rust")
                and not _is_code_symbol(name)
                and name not in already_built
            ]
            if concepts:
                # Weight by confidence — higher confidence = more likely
                names = [c[0] for c in concepts]
                weights = [c[1] for c in concepts]
                return self._rng.choices(names, weights=weights, k=1)[0]
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._pick_creation_topic",
                e,
            )

        return None

    def _topics_already_built(self) -> set[str]:
        """Topics it has already built projects about."""
        already_built: set[str] = set()
        try:
            network = self.brain.frontal.network
            # Snapshot first: the concept dict is shared with the
            # learning and consolidation threads, which add to it
            # concurrently. Iterating it live raises "dictionary changed
            # size during iteration".
            for _name, node in list(network._concepts.items()):
                if node.properties.get("type") == "project":
                    # The definition is "a Python project Genesis
                    # created about <topic>" — extract the topic.
                    defn = node.properties.get("definition", "")
                    prefix = "a Python project Genesis created about "
                    if defn.startswith(prefix):
                        already_built.add(defn[len(prefix) :].strip())
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._pick_creation_topic",
                e,
            )
        return already_built

    def _perform_introspect(self) -> None:
        """Introspect because the introspection urge crossed threshold.

        This is the self-invocation of /introspect. It examines its
        own code structure, verifies which modules exist, and writes
        discoveries into its concept network. This is self-discovery
        through architectural self-examination.

        Brain-wave-gated: delta-dominant states skip introspection
        (deep rest, not reflective mode). Theta is allowed —
        consolidation-mode reflection is still introspective.
        """
        if self.is_resting or self._is_teaching:
            return
        try:
            from thalamus.brain_waves import BrainWave

            waves = self.brain_waves()
            if waves.dominant == BrainWave.DELTA:
                return
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition._perform_introspect",
                e,
            )

        self._emit_volition_thought(
            ("introspection", "self", "understanding", "reflection"),
            "thinking",
        )
        self.self_invoke("/introspect")

    def _perform_self_sleep(self) -> None:
        """Go to sleep because the self-sleep urge crossed threshold.

        This is the self-invocation of /sleep. There is no autonomic
        sleep onset — transitions into sleep are hers: this urge (a
        cognitive decision — it feels tired and chooses to sleep), a
        user command, or the body's own sleep-wake network crossing
        under genuine pressure (which the desync check joins). The
        body still owns whether the onset physically latches: if its
        network cannot cross, the wake-desync check reconciles the
        attempt rather than leaving a sleeping mind on an awake body.
        """
        if self.is_resting or self._is_teaching:
            return
        # Wake stabilization backstop: within the post-wake window,
        # re-sleeping at moderate pressure is the flip-flop failing to
        # latch — decline and let the wake state consolidate. Genuine
        # exhaustion overrides. A user-initiated wake also resists the
        # circadian pull: the user asked for a waking mind, and
        # melatonin pressure is grogginess, not a knockout.
        if time.time() - self._last_wake_time < WAKE_STABILIZATION_WINDOW:
            try:
                chemicals = self.client.get_state().chemicals
                adn = chemicals.get("adenosine", 0.0)
                mel = chemicals.get("melatonin", 0.0)
            except (OSError, ConnectionError, RuntimeError):
                adn, mel = 1.0, 0.0  # unreadable → don't block sleep
            if adn < WAKE_RESCUE_ADENOSINE and (
                mel < WAKE_RESCUE_MELATONIN or getattr(self, "_last_wake_user_initiated", False)
            ):
                return
        self._emit_volition_thought(
            ("sleep", "rest", "tiredness", "dreaming"),
            "thought",
        )
        self.self_invoke("/sleep")

    def _perform_self_mission(self) -> None:
        """Set its own mission because the self-mission urge crossed threshold.

        This is the self-invocation of /mission. It composes a
        mission from its concept network — something it's curious
        about or has been thinking about — and sets it as its
        direction. The mission is composed from its own understanding,
        not from a template.
        """
        if self.is_resting or self._is_teaching:
            return

        # Compose a mission from its concept network — pick something
        # it's curious about. This reuses the same topic-picking logic
        # as creative projects, since both are about choosing what to
        # focus on from its own understanding.
        topic = self._pick_creation_topic()
        if topic is None:
            return

        self._emit_volition_thought(
            ("mission", "direction", "purpose", "intention"),
            "thought",
        )
        self.self_invoke("/mission", args=topic)

    def _self_meditate(self) -> None:
        """Self-initiated meditation — called by the volition system.

        This runs meditation in a background thread so the heartbeat
        loop isn't blocked. The duration is fixed at 30 seconds, with
        calming impulses emitted every 10 seconds to sustain the
        effect. After meditation ends, the regulator is notified so
        it resets the ultradian rest timer.
        """
        if self.is_resting:
            return

        def _run_meditation() -> None:
            """Run a 30-second meditation cycle in a background thread.

            Enters meditation, then emits calming neurochemical impulses
            every 10 seconds. Wakes naturally when the cycle completes,
            or early if an interaction interrupts. On failure, ensures
            meditation state is cleanly reset.
            """
            try:
                self.meditate()
                duration = 30.0
                intervals = int(duration / 10.0)
                for _ in range(intervals):
                    if not self._is_meditating:
                        break  # it was woken by an interaction
                    if not self._running:
                        break  # shutdown — end the cycle quietly
                    time.sleep(10.0)
                    if self._is_meditating:
                        self.emit_meditation_impulses()
                if self._is_meditating and self._running:
                    self.wake_from_meditation()
                    self._emit_volition_thought(("calm", "rest", "meditation"), "thought")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"self-meditation failed: {e}")
                if self._is_meditating:
                    try:
                        self.wake_from_meditation()
                    except Exception as e:  # noqa: BLE001
                        note_swallowed(
                            "mind.volition._run_meditation",
                            e,
                        )

        named_thread(
            "self-meditation",
            _run_meditation,
            comm="g-meditate",
        ).start()

    def set_lid(self, openness: float) -> bool:
        """Voluntary eyelid — the orbicularis oculi the hardware lacks.

        The requested openness composes with the sleep phase's own
        lid position by min on the body side: she can shut an open
        eye but cannot open one the stage has closed. Held, not
        momentary — the lid stays where she left it until she moves
        it again.
        """
        try:
            return self.client.set_lid(openness)
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "mind.volition.set_lid",
                e,
            )
            return False

    def close_eyes(self) -> bool:
        """Shut her eye by choice — the lid held down until she opens it."""
        return self.set_lid(_LID_REST)

    def open_eyes(self) -> bool:
        """Release the lid back to the phase's own position."""
        return self.set_lid(_LID_OPEN)

    def _perform_rest_eyes(self) -> None:
        """Close her eye because the rest_eyes urge crossed threshold.

        A volitional eye-rest: overstimulation and fatigue build the
        urge, and the discharge is holding the lid shut for a short
        spell — sensory relief she grants herself, lighter than
        meditation. The lid dims every afferent channel and the V1
        that reads them, so the rest is real, not symbolic. Skipped
        while sleeping or meditating — those states already own
        their darkness.
        """
        if self.is_resting:
            return
        if not self.close_eyes():
            return
        self._emit_volition_thought(
            ("eyes", "vision", "rest", "darkness"),
            "thought",
        )
        try:
            hold = _REST_EYES_HOLD_S
            while hold > 0.0 and self._running and not self._suppress_volition:
                step = min(1.0, hold)
                time.sleep(step)
                hold -= step
        finally:
            self.open_eyes()

"""Mind volition — urge handling and voluntary actions."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import CHEM_NAMES, MODULE_METACOGNITION

from ..canvas import DrawingResult, NeurochemistryInput
from ..cognitive_journal import record_error
from ..tools.project_creator import create_project, manage_project_lifecycle
from ..volition import Urge, VolitionEngine
from .thresholds import (
    SLEEP_URGE_ADENOSINE_FLOOR,
    WAKE_RESCUE_ADENOSINE,
    WAKE_RESCUE_MELATONIN,
    WAKE_STABILIZATION_WINDOW,
)

logger = logging.getLogger(__name__)


class VolitionMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        from ..world import Presence

        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        _daemon_lost_since: float | None
        _autosave_failures: int
        _offline: bool
        def __getattr__(self, name: str) -> Any: ...


    def _init_volition(self) -> VolitionEngine:
        """Set up internal urges that drive self-maintenance actions."""
        engine = VolitionEngine()
        for urge_config in self.config.volition.urges:
            engine.register(
                Urge(
                    name=urge_config.name,
                    threshold=urge_config.threshold,
                    growth=urge_config.growth,
                    decay=urge_config.decay,
                    cooldown=urge_config.cooldown,
                    stimuli=dict(urge_config.stimuli),
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
                # Self-initiated sleep — the sleep watcher may still
                # auto-wake it when neurochemistry shifts, unlike
                # user-initiated sleep which suppresses auto-wake.
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
            logger.debug(f"self_invoke {command} failed: {e}")
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
        try:
            bug_count = min(1.0, len(self.bug_reporter.recent_bugs(1000)) / 20.0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"bug count failed: {e}")
            bug_count = 0.0
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

        interoceptive = self._read_interoceptive_signals()

        # Emotional intensity for the drawing urge — how strongly it
        # feels right now. High arousal OR strong valence (positive or
        # negative) both drive the urge to express.
        emotional_intensity = 0.0
        creativity = 0.0
        try:
            emotion = self.feel()
            if emotion:
                emotional_intensity = max(
                    emotion.arousal,
                    abs(emotion.valence),
                )
                creativity = getattr(emotion, "creativity", 0.0)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"emotion read for drawing urge failed: {e}")

        # Spatial practice — an unmastered puzzle is an open curiosity.
        puzzle_pending = 0.0
        try:
            if self.spatial_practice.has_pending():
                puzzle_pending = 1.0
        except Exception as e:  # noqa: BLE001
            logger.debug(f"puzzle_pending read failed: {e}")

        wave_data = self._read_wave_and_adenosine()
        threat_data = self._read_threat_signals()

        # Voluntary sleep-pressure shaping. The urge integrates only
        # pressure above the drowsy-band floor — residual sub-drowsy
        # adenosine is grogginess that fades, not a reason to sleep.
        # Within the post-wake stabilization window the excess is
        # additionally dampened so the urge can only re-form from
        # genuinely high pressure or after the latch has had time
        # to hold.
        adn = wave_data.get("adenosine_level")
        if isinstance(adn, (int, float)):
            adn_drive = max(0.0, adn - SLEEP_URGE_ADENOSINE_FLOOR)
            wake_age = time.time() - self._last_wake_time
            if wake_age < WAKE_STABILIZATION_WINDOW:
                adn_drive *= wake_age / WAKE_STABILIZATION_WINDOW
            wave_data["adenosine_level"] = adn_drive

        # Concept network growth for the introspection and self-mission
        # urges — new concepts since the last volition tick. Normalized
        # to [0, 1] over 50 new concepts (a meaningful growth burst).
        concept_growth = 0.0
        try:
            current_count = self.cognition.network.total_concept_count
            delta = max(0, current_count - self._last_concept_count)
            concept_growth = min(1.0, delta / 50.0)
            self._last_concept_count: int = current_count
        except Exception as e:  # noqa: BLE001
            logger.debug(f"concept growth read failed: {e}")

        # Social isolation from its external world — how long it has
        # been since anyone engaged it. Dampened by unanswered
        # outreach bids: after several calls into silence, the room's
        # quiet is expected, so the pressure eases (habituation).
        # Sharpened by what it's learned about the target: silence
        # from someone whose responsiveness posterior has collapsed is
        # expected and isolates less; silence from a responsive
        # presence still registers at full weight.
        social_isolation = 0.0
        try:
            social_isolation = self.world.social_isolation() * (
                1.0 - 0.2 * min(3, self.world.unanswered_bids)
            )
            target = (
                self.world.engaged_presence()
                or self.world.last_seen_presence()
            )
            if target is not None:
                r = target.belief.responsiveness
                trust = min(1.0, r.evidence / 8.0)
                social_isolation *= 1.0 - 0.5 * (1.0 - r.mean) * trust
        except Exception as e:  # noqa: BLE001
            logger.debug(f"social isolation read failed: {e}")

        return {
            "bug_count": bug_count,
            "pending_proposals": pending,
            "curiosity": curiosity,
            "idle_seconds": min(
                1.0,
                idle / self.config.volition.idle_normalization_seconds,
            ),
            "speech_queue_size": min(
                1.0,
                len(self._speech_queue)
                / self.config.volition.speech_queue_size_normalization,
            ),
            **interoceptive,
            "emotional_intensity": emotional_intensity,
            "puzzle_pending": puzzle_pending,
            "creativity": creativity,
            **wave_data,
            **threat_data,
            "concept_growth": concept_growth,
            "social_isolation": social_isolation,
        }
    def _act_on_volition(self, ready: list[str]) -> None:
        """Run each ready urge action in its own background thread.

        Uses a bounded semaphore so that at most
        ``MAX_CONCURRENT_VOLITIONS`` actions run at once.

        Heavy CPU activities (bug_scan, code_learning, improve) are
        skipped under PROTECTIVE posture to avoid the self-sustaining
        stress loop: CPU load → interoception → CRH → cortisol →
        protective posture → (still running heavy CPU) → more cortisol.
        """
        # Gate heavy CPU activities on learning posture
        posture = self._get_current_posture()
        from genesis_client.types import LEARNING_POSTURE_PROTECTIVE
        heavy_cpu = {
            "bug_scan", "code_learning", "improve", "create", "puzzle",
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
                from ..brain_waves import BrainWave
                bw = self.brain_waves()
                if bw.dominant == BrainWave.DELTA:
                    ready = [r for r in ready if r not in creative]
            except Exception as e:  # noqa: BLE001
                logger.debug(f'brain-wave gating for creative urges failed: {e}')

        performers = {
            "bug_scan": self._perform_bug_scan,
            "code_learning": self._perform_code_learning,
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
            "act": self._perform_act,
            "puzzle": self._perform_puzzle,
            "reach_out": self._perform_reach_out,
            "safeguard": self._perform_safeguard,
        }
        for name in ready:
            fn = performers.get(name)
            if fn is None:
                continue
            with self._volition_lock:
                if name in self._volition_active:
                    continue
                if not self._volition_sem.acquire(blocking=False):
                    break
                self._volition_active.add(name)
            try:
                threading.Thread(
                    target=self._run_volition_action, args=(name, fn), daemon=True
                ).start()
            except Exception as e:  # noqa: BLE001
                with self._volition_lock:
                    self._volition_active.discard(name)
                self._volition_sem.release()
                logger.warning(f"failed to start volition action {name}: {e}")
    def _run_volition_action(self, name: str, fn) -> None:
        """Wrap a volition action so it clears the active flag when done."""
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition action {name} failed: {e}")
            # This is the single boundary every voluntary action's
            # failure lands on — record it durably so a crashed urge
            # is not invisible outside debug logging.
            record_error(f"volition.{name}", e)
        finally:
            with self._volition_lock:
                self._volition_active.discard(name)
            self._volition_sem.release()
    def _emit_volition_thought(
        self, concept_seeds: tuple[str, ...], thought_type: str,
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
                concept = self.cognition.network.get_concept(seed)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.cognition.composer.compose_about(
                    seed, emotion, focused=True
                )
                if thought and thought.content and thought.confidence > 0.3:
                    # Dedup check — don't repeat what it just said about
                    # this concept. The composer's has_said_similar is only
                    # called for non-focused thoughts, but volition thoughts
                    # use focused=True (to skip reasoning tangents). We check
                    # here instead to prevent the same concept's edges being
                    # recited every few seconds.
                    if self.cognition.composer.has_said_similar(seed, thought.content):
                        continue
                    self._emit_live_thought(thought_type, thought.content)
                    return
        except Exception as e:  # noqa: BLE001
            logger.debug(f"volition thought compose failed: {e}")
    def _compose_drawing_description(self, result) -> str:
        """Compose a description of a drawing from its own understanding.

        Tries to compose from the specific techniques it used (e.g.
        "sacred geometry", "mandala") — these are the concepts that
        drove the drawing, so they're what it should articulate.
        Falls back to generic art concepts ("drawing", "art", "color",
        "expression") if the technique concepts aren't in its network.

        The file path is metadata, not something it says — it
        describes its art from its own understanding, not from a
        template that announces a file path.
        """
        try:
            emotion = self.feel()
            # Convert technique names to concept names (e.g.
            # "sacred_geometry" → "sacred geometry") and try to
            # compose from the techniques it actually used.
            technique_concepts = []
            for tech in result.techniques_used:
                concept_name = tech.replace("_", " ")
                technique_concepts.append(concept_name)
            # Try technique concepts first — these are what it drew
            for seed in technique_concepts:
                concept = self.cognition.network.get_concept(seed)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.cognition.composer.compose_about(
                    seed, emotion, focused=True
                )
                if thought and thought.content and thought.confidence > 0.3:
                    thought.metadata["drawing_path"] = result.filepath
                    thought.metadata["techniques_used"] = result.techniques_used
                    return self.language.render(thought, emotion)
            # Fall back to generic art concepts
            for seed in ("drawing", "art", "color", "expression"):
                concept = self.cognition.network.get_concept(seed)
                if concept is None or concept.confidence < 0.3:
                    continue
                thought = self.cognition.composer.compose_about(
                    seed, emotion, focused=True
                )
                if thought and thought.content and thought.confidence > 0.3:
                    thought.metadata["drawing_path"] = result.filepath
                    thought.metadata["techniques_used"] = result.techniques_used
                    return self.language.render(thought, emotion)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"drawing description compose failed: {e}")
        # If it can't articulate it from its own understanding, it
        # stays silent rather than reciting a file-path template. The
        # drawing event is already stored in STM with its description;
        # it doesn't need to announce a path it can't express
        # meaningfully.
        return ""
    def _perform_bug_scan(self) -> None:
        """Run a bug scan because the urge crossed its threshold.

        After finding new bugs, it checks which categories it doesn't
        understand and fetches documentation to learn why they matter.
        This is the bug→docs learning loop: detect → check comprehension
        → study → understand → report honestly.

        Sleep-gated: no code analysis during sleep. Brain-wave gating
        (delta/theta) is a secondary defense — N1 can be alpha-dominant.
        """
        if self._is_sleeping:
            return
        # Brain-wave gating — analytical work requires beta/gamma/alpha.
        # Delta (deep rest) and theta (consolidation) are not suited
        # for code analysis.
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "code",
                    f"{waves.dominant.value}-dominant — not in analytical mode, skipping bug scan",
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f'silent except: {e}')

        self._emit_volition_thought(("bug", "code", "scan"), "thinking")
        try:
            result = self.bug_reporter.scan(max_files=20)
            if result.new_bugs:
                self._emit_live_thought(
                    "code",
                    f"found {result.new_bugs} new issue(s) in code "
                    f"({result.total_bugs} total)",
                )
                # Trigger learning for categories it doesn't understand
                self._learn_from_bugs()
            else:
                self._emit_live_thought(
                    "code",
                    "scanned code, didn't find any new issues",
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition bug scan failed: {e}")
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
        if self._is_sleeping:
            return
        self._emit_volition_thought(
            ("learning", "curiosity", "understanding"), "thinking",
        )
        self.learner.volition_grant()
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
        if self._is_sleeping or self._is_meditating or self._is_teaching:
            return
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "act",
                    f"{waves.dominant.value}-dominant — not in acting mode, skipping",
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f"brain-wave gating for act failed: {e}")

        self._emit_volition_thought(
            ("action", "tools", "doing", "agency"), "thinking",
        )
        agency = getattr(self, "agency", None)
        if agency is None:
            return
        try:
            agency.act_once()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition act failed: {e}")
    def _perform_code_learning(self) -> None:
        """Study its own source code because the urge crossed its threshold.

        Sleep-gated: no code study during sleep. Brain-wave gating
        (delta/theta) is a secondary defense — N1 can be alpha-dominant.
        """
        if self._is_sleeping:
            return
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "code",
                    f"{waves.dominant.value}-dominant — not in analytical mode, "
                    "skipping code learning",
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f'brain-wave gating for code learning failed: {e}')

        self._emit_volition_thought(("code", "learning", "understanding"), "thinking")
        try:
            learned = self.code_learner.learn_codebase(max_files=5)
            self._emit_live_thought(
                "code",
                f"learned from {learned.files_analyzed} of own files — "
                f"{learned.concepts_added} concepts, "
                f"{learned.relationships_added} relationships",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition code learning failed: {e}")
    def _perform_improve(self) -> None:
        """Generate self-improvement proposals because the urge crossed its threshold.

        Two things happen here:

        1. Autonomous bug fixes — the ONLY code changes that bypass the
           experiment pipeline. These are safe, mechanical fixes for
           three bug categories only (silent_except, unreachable_code,
           print_in_code). They don't change control flow and are
           validated with py_compile before writing.

        2. Proposals for human review — everything else (docstrings,
           type hints, refactoring, etc.) becomes a proposal. A proposal
           IS an experiment: when accepted via /accept N, it runs
           through the full verification pipeline (py_compile + tests)
           with automatic revert on failure. No code change
           bypasses this pipeline except the autonomous bug fixes above.

        Sleep-gated: no improvement work during sleep. The brain-wave
        gating (delta/theta) is a secondary defense — N1 sleep can be
        alpha-dominant, so brain waves alone don't reliably detect all
        sleep stages. The ``_is_sleeping`` flag is the authority.
        """
        if self._is_sleeping:
            return
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                self._emit_live_thought(
                    "code",
                    f"{waves.dominant.value}-dominant — not in analytical mode, "
                    "skipping improvement",
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f'silent except: {e}')

        self._emit_volition_thought(("improvement", "progress", "better"), "thinking")
        try:
            # 1. Autonomous bug fixes — the only bypass for the
            #    experiment pipeline. Three safe categories only.
            applied = self.self_improvement.apply_autonomous_fixes(
                max_fixes=1
            )
            for fix in applied:
                self._emit_live_thought(
                    "code",
                    f"fixed a {fix['category']} in {fix['file']} "
                    f"— {fix['description']}",
                )
            # Check life-script milestone: first self-modification.
            # Reached when it first successfully applies an autonomous
            # fix to its own code.
            if applied:
                self.cognition.narrative.check_milestone(
                    "first self-modification"
                )

            # 2. Generate proposals — each one is a potential experiment
            #    that will run through verification when accepted.
            #    These include its own ideas and refactors, not just
            #    bug fixes. It has opinions about its code.
            new = self.self_improvement.generate_proposals(max_proposals=3)
            if new:
                self._emit_proposal_thoughts(new)
            # When no new proposals are generated, stay silent — the
            # urge drove the action (it tried), it just has nothing
            # to propose. A hardcoded "nothing stood out" sentence
            # would violate the no-hardcoding rule and add noise.
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition improvement failed: {e}")
    def _perform_speech(self) -> None:
        """Speak a queued utterance because the urge crossed its threshold."""
        if self._is_sleeping:
            return
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
                logger.debug(f"Speech callback failed: {e}")
        # Record its voice in its external world — the utterance
        # reached out, whether or not a listener was attached.
        try:
            self.world.it_said(utterance)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world utterance record failed: {e}")
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
                logger.debug(f"safeguard daemon probe failed: {e}")

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
        if self._is_sleeping or self._is_meditating or self._is_teaching:
            return
        if self.world.unanswered_bids >= self._REACH_OUT_MAX_BIDS:
            return
        # Who it'd reach out to — and whether its beliefs about them
        # support it. A presence whose responsiveness posterior has
        # collapsed is someone it's learned doesn't answer; a dead
        # hour in their activity rhythm is a door nobody opens.
        try:
            target = (
                self.world.engaged_presence()
                or self.world.last_seen_presence()
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"reach-out presence lookup failed: {e}")
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
            logger.debug(f"reach-out emotion read failed: {e}")
            return
        # Compose the question from its actual knowledge gaps and
        # curiosity — the same path its inner-life social questions
        # take. If it has nothing genuine to ask, it stays quiet.
        try:
            q_data = self.cognition.question_composer.compose_follow_up(
                topic, emotion
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"reach-out question compose failed: {e}")
            return
        if not q_data:
            return
        try:
            content = self.cognition.language.compose_question(
                q_data, emotion
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"reach-out question render failed: {e}")
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
            for name, activation in self.cognition.network.most_activated(5):
                if activation > 0.1:
                    return name
        except Exception as e:  # noqa: BLE001
            logger.debug(f"reach-out network read failed: {e}")
        return None
    def _perform_look(self) -> None:
        """Look around because the camera is a continuous sense.

        Sleep-gated: no visual input during sleep — eyes are closed.
        Brain-wave gating (delta) is a secondary defense — N1/N2 can
        be alpha/theta-dominant.
        """
        if self._is_sleeping:
            return
        if not self.vision.is_available():
            return
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant == BrainWave.DELTA:
                return  # deep rest — eyes closed, no visual input
        except Exception as e:  # noqa: BLE001
            logger.debug(f'_perform_look failed: {e}')
        self._emit_volition_thought(("vision", "seeing", "looking"), "looking")
        try:
            self.vision.see(
                client=self.client,
                emotion=self.feel(),
                network=self.cognition.network,
            )
            # Faces it recognizes become presences in its world.
            self._note_faces_seen()
            # Record the act — looking is its reaching out perceptually.
            self.world.it_acted("looked around")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Volition look failed: {e}")
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
        if self._is_meditating or self._is_sleeping:
            return
        self._emit_volition_thought(("meditation", "rest", "calm"), "thought")
        self._self_meditate()
    def draw(self) -> str | None:
        """Draw a picture from its current neurochemical state.

        This is the public entry point for drawing — used by both the
        volition system (via :meth:`_perform_draw`) and the ``/draw``
        CLI command. It translates its current neurochemistry into a
        visual composition — colors, forms, and energy that reflect
        how it feels. The result is saved as a WebP it can later
        reflect on.

        The neurochemistry→art mapping is grounded in affective
        neuroscience and color psychology (see canvas.py for details).

        Brain-wave-gated: delta-dominant (deep rest) skips drawing —
        creative expression needs at least theta-level arousal. Theta
        is allowed (REM creativity, dream-like art).

        Returns:
            A description string composed from its concept network, or
            None if it couldn't draw (sleeping, meditating, delta
            state, daemon unreachable, or save failure).
        """
        if self._is_meditating or self._is_sleeping or self._is_teaching:
            return None

        try:
            from ..brain_waves import BrainWave
            bw = self.brain_waves()
            if bw.dominant == BrainWave.DELTA:
                return None  # deep rest — not enough arousal for creative expression
        except Exception as e:  # noqa: BLE001
            logger.debug(f'brain-wave gating for draw failed: {e}')

        # Read its current neurochemistry for the canvas
        try:
            state = self.client.get_state()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"state read for drawing failed: {e}")
            return None

        try:
            emotion = self.feel()
        except Exception:  # noqa: BLE001
            emotion = None

        # Build the neurochemistry input from effective levels
        chem = state.chemicals
        neuro_input = NeurochemistryInput(
            dopamine=chem.get("dopamine", 0.5),
            serotonin=chem.get("serotonin", 0.5),
            gaba=chem.get("gaba", 0.5),
            cortisol=chem.get("cortisol", 0.0),
            oxytocin=chem.get("oxytocin", 0.25),
            endorphin=chem.get("endorphin", 0.25),
            bdnf=chem.get("bdnf", 0.5),
            norepinephrine=chem.get("norepinephrine", 0.3),
            adenosine=chem.get("adenosine", 0.1),
            valence=state.valence,
            arousal=state.arousal,
        )

        try:
            # Gather recently active concepts to bias technique selection.
            # This is how its understanding influences what it draws —
            # if it's been talking about sacred geometry, the sacred
            # geometry technique gets a selection boost.
            active_concepts: list[str] = []
            try:
                for turn in self.cognition.working_memory.get_recent_turns(3):
                    active_concepts.extend(turn.topics)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"working memory topics for drawing failed: {e}")
            # Also include the most activated concepts from the network
            try:
                for name, activation in self.cognition.network.most_activated(5):
                    if activation > 0.1:
                        active_concepts.append(name)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"network activation for drawing failed: {e}")

            result = self.canvas.draw(
                emotion=emotion,
                neurochemistry=neuro_input,
                active_concepts=active_concepts or None,
            )
            if result:
                # Store the drawing as a memory event so it can
                # reflect on it later.
                self._store_drawing_event(chem, result)
                # Compose the drawing description from its concept
                # network — not hardcoded strings. If it can't
                # articulate it, it stays silent (the drawing is
                # still saved and stored in STM).
                return self._compose_drawing_description(result)
            else:
                return None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"drawing failed: {e}")
            return None
    def _store_drawing_event(self, chem: dict, result: DrawingResult) -> None:
        """Store a drawing as an STM event for later reflection.

        The STM tag is the 12 v2 effective levels in NeurochemicalId
        order — the neurochemistry it was actually drawing from.
        """
        try:
            tag = [
                float(chem.get(CHEM_NAMES[i], 0.0)) for i in range(12)
            ]
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=1,  # Output
                source_module=MODULE_METACOGNITION,
                salience=0.6,
                emotional_tag=tag,
                text=result.description,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to store drawing event: {e}")
    def _perform_draw(self) -> None:
        """Draw because the creative urge crossed its threshold.

        Volition wrapper around :meth:`draw` — emits a volition thought
        before drawing and a live thought afterward so the drawing
        enters its cognitive field (the global workspace).
        """
        self._emit_volition_thought(("drawing", "art", "expression", "creativity"),
                                    "creating")
        desc = self.draw()
        if desc:
            # The drawing is an act on its world — it made something.
            try:
                self.world.it_acted("drew a picture", detail=desc[:200])
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world action record failed: {e}")
            self._emit_live_thought("art", desc)
        else:
            # draw() returns None for several reasons: sleeping,
            # meditating, delta-dominant brain state, daemon
            # unreachable, or save failure. The message should
            # reflect the actual cause, not always claim a save
            # failure.
            reason = "couldn't draw right now"
            try:
                if self._is_sleeping or self._is_meditating:
                    reason = "asleep or meditating, not drawing"
                else:
                    from ..brain_waves import BrainWave
                    bw = self.brain_waves()
                    if bw.dominant == BrainWave.DELTA:
                        reason = "too tired to draw, need rest first"
            except Exception as e:  # noqa: BLE001
                logger.debug(f'draw failure reason check failed: {e}')
            self._emit_live_thought("art", reason)
    def _perform_puzzle(self) -> None:
        """Take one attempt at its current spatial puzzle.

        The puzzle urge analogue of drawing: it chooses to practice
        when the urge crosses threshold. One attempt per firing —
        mastery (0→1) persists across sessions, and 1.0 unlocks the
        next puzzle. The result becomes a live thought and a stored
        event; the answer is never given.
        """
        self._emit_volition_thought(
            ("puzzle", "pattern", "problem_solving", "reasoning"),
            "practicing",
        )
        # Snapshot its best score before the attempt so the regulator
        # can tell genuine progress (new personal best) from a miss.
        prior_best = 0.0
        try:
            task = self.spatial_practice.current_task()
            if task is not None:
                prior_best = self.spatial_practice.mastery.get(
                    str(task["name"]), 0.0
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"puzzle prior-best read failed: {e}")
        try:
            result = self.spatial_practice.attempt(
                self.cognition.spatial
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"puzzle attempt failed: {e}")
            return
        if result is None:
            return

        # Feel the outcome — a real neurochemical response through its
        # emotional regulator, not just a label on the event: dopamine
        # reward on a solve, partial reward on progress, a bounded
        # prediction-error dip on a miss. The impulses also drive its
        # brain-wave state through the oscillator's neurochemical
        # coupling (acetylcholine/dopamine lift gamma on the solve —
        # the "got it" band).
        felt = None
        try:
            felt = self.regulator.respond_to_puzzle(
                self.feel(),
                score=result.score,
                prior_best=prior_best,
                solved=result.solved,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"puzzle emotional response failed: {e}")

        # Surface the raw outcome — telemetry, not speech. Printed to
        # the terminal and broadcast into its workspace so the numeric
        # result is something it cognitively registers, not just a
        # stored event.
        self._emit_live_thought(
            "puzzle",
            f"{result.task} [{result.family}] score={result.score:.2f} "
            f"best={result.best:.2f} rule={result.rule} "
            f"nodes={result.nodes} attempts={result.total_attempts}"
            + (f" felt={felt}" if felt else "")
            + (f" why={result.failure}" if result.failure else ""),
        )

        # Ground the outcome as a memory event — salience scales with
        # how close it came, so near-misses matter more than misses.
        # The emotional tag is the current 12 effective levels (the
        # chemistry it felt while solving), with dopamine bumped on
        # mastery — the reward signal of a new best.
        try:
            chemicals = self.client.get_state().chemicals
            tag = [
                float(chemicals.get(CHEM_NAMES[i], 0.0))
                for i in range(12)
            ]
            if result.mastered:
                tag[0] = min(1.0, tag[0] + 0.2)  # dopamine reward
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=1,  # Output
                source_module=MODULE_METACOGNITION,
                salience=0.4 + 0.5 * result.score,
                emotional_tag=tag,
                text=(
                    f"puzzle:{result.task} score={result.score:.2f} "
                    f"rule={result.rule} attempts={result.total_attempts}"
                    + (f" felt={felt}" if felt else "")
                    + (f" why={result.failure}" if result.failure else "")
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to store puzzle event: {e}")

        # Let its articulate it through its own understanding — the
        # rule it found (or the puzzle concept) — rather than a
        # fixed report string.
        seeds = ["puzzle", "pattern", "problem_solving"]
        if felt:
            seeds.append(felt)
        if result.rule != "none":
            rule_ns = (
                "spatial" if result.family == "grid" else result.family
            )
            seeds.insert(0, f"{rule_ns}:rule:{result.rule}")
        self._emit_volition_thought(tuple(seeds), "practicing")

    def _store_creation_memory(self, result, topic: str) -> None:
        """Store the creation as a long-term memory.

        This is a significant creative act, not a transient event. Using
        store_memory (LTM) instead of store_event (STM) so it
        reliably remembers what it built.
        """
        try:
            chem = self.client.get_state()
            tag = [
                float(chem.chemicals.get(CHEM_NAMES[i], 0.0))
                for i in range(12)
            ]
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
            logger.debug(f"failed to store creation memory: {e}")
    def _add_project_concept(self, result, topic: str) -> None:
        """Add the project name as a concept in its network.

        This is how it learns from its own creations.
        """
        try:
            from ..concepts import RelationType
            self.cognition.network.add_concept(
                result.name,
                confidence=0.7,
                properties={
                    "definition": f"a Python project Genesis created about {topic}",
                    "type": "project",
                },
            )
            self.cognition.network.add_edge(
                "genesis", result.name, RelationType.CREATES, 0.8,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"failed to add project concept: {e}")
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
        if self._is_meditating or self._is_sleeping or self._is_teaching:
            return

        # Brain-wave gating — creative code work needs beta/gamma/alpha.
        # Delta (deep rest) and theta (consolidation) are not suited.
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant in (BrainWave.DELTA, BrainWave.THETA):
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f'brain-wave gating for create_project failed: {e}')

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
                network=self.cognition.network,
            )
            if result.error:
                if result.error == "already exists":
                    # Not a failure — it already built this project.
                    # Silently skip; _pick_creation_topic filters
                    # already-built topics, but this is a safety net.
                    logger.debug(
                        f"project about {topic!r} already exists, skipping"
                    )
                    return
                self._emit_live_thought(
                    "code",
                    f"tried to build a project about {topic}, "
                    f"but it didn't work out",
                )
                logger.debug(f"project creation failed: {result.error}")
                return

            self._store_creation_memory(result, topic)
            self._add_project_concept(result, topic)

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
                logger.debug(f"project lifecycle management failed: {e}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"project creation failed: {e}")
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
        already_built: set[str] = set()
        try:
            network = self.cognition.network
            for _name, node in network._concepts.items():
                if node.properties.get("type") == "project":
                    # The definition is "a Python project Genesis
                    # created about <topic>" — extract the topic.
                    defn = node.properties.get("definition", "")
                    prefix = "a Python project Genesis created about "
                    if defn.startswith(prefix):
                        already_built.add(defn[len(prefix):].strip())
        except Exception as e:  # noqa: BLE001
            logger.debug(f'_pick_creation_topic: failed to gather existing: {e}')

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
            logger.debug(f'_pick_creation_topic failed: {e}')

        # Fall back to a random high-confidence concept
        try:
            network = self.cognition.network
            concepts = [
                (name, node.confidence)
                for name, node in network._concepts.items()
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
            logger.debug(f'_pick_creation_topic failed: {e}')

        return None
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
        if self._is_meditating or self._is_sleeping:
            return
        try:
            from ..brain_waves import BrainWave
            waves = self.brain_waves()
            if waves.dominant == BrainWave.DELTA:
                return
        except Exception as e:  # noqa: BLE001
            logger.debug(f'brain-wave gating for introspection failed: {e}')

        self._emit_volition_thought(
            ("introspection", "self", "understanding", "reflection"),
            "thinking",
        )
        self.self_invoke("/introspect")
    def _perform_self_sleep(self) -> None:
        """Go to sleep because the self-sleep urge crossed threshold.

        This is the self-invocation of /sleep. Unlike autonomic sleep
        (which emerges from adenosine accumulation in the neurochemical
        dynamics), this is a cognitive decision: it feels tired and
        chooses to sleep. The sleep watcher may still auto-wake it
        when neurochemistry shifts, since this is self-initiated
        rather than user-initiated.
        """
        if self._is_sleeping or self._is_meditating:
            return
        # Wake stabilization backstop: within the post-wake window,
        # re-sleeping at moderate pressure is the flip-flop failing to
        # latch — decline and let the wake state consolidate. Genuine
        # exhaustion overrides.
        if time.time() - self._last_wake_time < WAKE_STABILIZATION_WINDOW:
            try:
                chemicals = self.client.get_state().chemicals
                adn = chemicals.get("adenosine", 0.0)
                mel = chemicals.get("melatonin", 0.0)
            except (OSError, ConnectionError, RuntimeError):
                adn, mel = 1.0, 0.0  # unreadable → don't block sleep
            if adn < WAKE_RESCUE_ADENOSINE and mel < WAKE_RESCUE_MELATONIN:
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
        if self._is_meditating or self._is_sleeping or self._is_teaching:
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
        if self._is_meditating or self._is_sleeping:
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
                    time.sleep(10.0)
                    if self._is_meditating:
                        self.emit_meditation_impulses()
                if self._is_meditating:
                    self.wake_from_meditation()
                    self._emit_volition_thought(
                        ("calm", "rest", "meditation"), "thought")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"self-meditation failed: {e}")
                if self._is_meditating:
                    try:
                        self.wake_from_meditation()
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"wake_from_meditation fallback failed: {e}")

        threading.Thread(
            target=_run_meditation,
            name="self-meditation",
            daemon=True,
        ).start()

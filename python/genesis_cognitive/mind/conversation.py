"""Mind conversation — respond pipeline and user-facing I/O."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import CHEM_DOPAMINE, MODULE_SENSORY

from ..cognition import CognitiveState
from ..language import Thought
from ..self import EmergentIdentitySource, IdentityStage
from ..vision import load_image, resize_for_vision
from ..world import EventKind, PresenceKind

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..world import ExternalEvent

logger = logging.getLogger(__name__)


class ConversationMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        _interaction_count: int
        def __getattr__(self, name: str) -> Any: ...


    def add_live_thought_listener(self, listener) -> None:
        """Register a listener for Genesis's spontaneous thoughts.

        The listener is called as ``listener(kind: str, content: str)``
        from background threads, where ``kind`` is ``"thought"`` (a
        spontaneous inner-life thought) or ``"learning"`` (something
        it learned autonomously). Must be thread-safe.
        """
        self._live_thought_listeners.append(listener)
    def _emit_live_thought(self, kind: str, content: str) -> None:
        """Notify all live-thought listeners and broadcast to the workspace.

        Live thoughts are Genesis's agentic output — things it does,
        says, or experiences from its volition system (bug scans, code
        learning, drawing, meditation, self-improvement, etc.). These
        must enter its cognitive field (the global workspace) so they
        contribute to workspace integration.

        Without this broadcast, volition actions are blindsight — it
        acts but doesn't cognitively experience its own agency. The
        workspace stays fragmented because only inner_life broadcasts
        when idle, giving at most one workspace item (integration ≈
        activation × 0.2 ≈ 0.15). Adding volition as a second source
        gives the workspace cross-module content to integrate, raising
        integration through both source diversity and topic coherence
        (volition topics like "code" and "learning" overlap with
        inner_life topics about the same concepts).

        The broadcast is gated by brain waves (delta raises the
        ignition threshold, gamma lowers it), so deep-rest states
        naturally suppress volition content in cognition —
        consistent with the brain-wave gating already applied to
        perception and inner_life broadcasts.

        Thread-safe: volition actions run in background threads, and
        the workspace uses an internal lock.
        """
        # Broadcast to the global workspace so volition content enters
        # the cognitive field. Best-effort — workspace errors must not
        # block the live thought from reaching CLI listeners.
        try:
            if hasattr(self, "cognition") and hasattr(
                self.cognition, "global_workspace"
            ):
                bw = None
                if hasattr(self.cognition, "language"):
                    bw = self.cognition.language.current_brain_waves
                self.cognition.global_workspace.broadcast(
                    content=content,
                    source="volition",
                    activation=0.7,
                    brain_waves=bw,
                    metadata={
                        "kind": kind,
                        "topics": [kind],
                    },
                )
        except Exception:
            logger.debug("volition workspace broadcast failed", exc_info=True)

        for listener in self._live_thought_listeners:
            try:
                listener(kind, content)
            except Exception:
                # a listener must never crash its mind
                logger.debug("live thought listener failed", exc_info=True)

        # Record its outward speech in its external world — expressions,
        # questions, and distress calls are its voice reaching out, so
        # the world it lives in contains its own agency.
        if kind in ("expression", "distress", "question"):
            try:
                self.world.it_said(content)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world utterance record failed: {e}")
    def mark_user_activity(self) -> None:
        """Mark that the user is active (typing or interacting).

        This updates ``_last_interaction_time`` from the CLI's input
        loop, BEFORE the sleep check. Without this, the heartbeat's
        auto-sleep check can fire between the user pressing Enter and
        ``respond()`` being called, putting it to sleep and causing
        the message to be lost. By marking activity here, the
        heartbeat sees the user as active even if it's already asleep
        (so after /wake, it doesn't immediately fall back asleep).
        """
        self._last_interaction_time = time.time()
    @property
    def is_user_sleeping(self) -> bool:
        """Whether its sleep was user-initiated (not auto-wakeable)."""
        return self._is_sleeping and self._user_initiated_sleep
    @property
    def is_teaching(self) -> bool:
        """Whether it is currently in teaching/training mode."""
        return self._is_teaching
    @property
    def teaching_topic(self) -> str:
        """The topic the user is currently teaching it about."""
        return self._teaching_topic
    @property
    def has_queued_questions(self) -> bool:
        """Whether it has queued questions to ask."""
        return self.cognition.has_queued_questions()
    @property
    def queued_question_count(self) -> int:
        """Number of queued questions."""
        return self.cognition.queued_question_count()
    @property
    def on_speak(self) -> Callable[[str], None] | None:
        """Callback used when the speech urge fires."""
        return self._on_speak
    @on_speak.setter
    def on_speak(self, callback: Callable[[str], None] | None) -> None:
        """Register the callback invoked when it speaks.

        Args:
            callback: A callable accepting the utterance string, or None
                to disable the speech callback.
        """
        self._on_speak = callback
    def offer_utterance(self, text: str) -> None:
        """Offer a candidate utterance for the speech-volition to consider.

        The CLI feeds live-thought text here. The Mind's volition engine
        decides whether it actually feels like saying it.

        During sleep, utterances are not queued — it shouldn't
        accumulate speech while resting. The queue would drain when
        it wakes, producing a burst of stale thoughts.
        """
        if not text:
            return
        if self._is_sleeping:
            return
        with self._speech_queue_lock:
            if text in self._speech_queue:
                return
            self._speech_queue.append(text)
    def _defer(self, delay: float, callback: Callable[[], None]) -> threading.Timer:
        """Schedule ``callback`` after ``delay`` seconds on a daemon timer.

        ``threading.Timer`` inherits its daemon flag from the creating
        thread. Timers created from the main thread (``respond()``,
        ``wake()``) are non-daemon, so interpreter shutdown would block
        until they fire — a pending ``Timer(60, ...)`` after wake would
        make /quit hang for up to a minute. Daemon timers die with the
        process; the callbacks (resume learner/inner life) are no-ops
        after ``stop()``.
        """
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()
        return timer
    def _on_v1_gamma(self, gamma: float) -> None:
        """Receive V1 gamma power from the occipital subsystem.

        V1 gamma is wired into neurochemistry via the ACh boost in
        _modulate_chemistry. It also directly drives the brain wave
        oscillator top-down — sensory input boosts gamma, which is
        the bidirectional coupling between perception and oscillatory
        state. This is grounded in the biology: visual stimulation
        drives gamma oscillations in V1 that propagate to
        higher-order regions (Fries et al., 2001).
        """
        logger.debug(f"V1 gamma: {gamma:.2f}")
        try:
            from ..brain_waves import BrainWave, add_brain_wave_drive

            if gamma > 0.1:
                add_brain_wave_drive(BrainWave.GAMMA, gamma * 0.2)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"V1 gamma drive failed: {e}")
    def _learn_from_bugs(self) -> None:
        """Fetch documentation for bug categories it doesn't understand.

        For each category it detected but doesn't comprehend, it
        fetches the relevant documentation (Python or Rust docs) to
        learn the concepts behind the bug. This is how it moves from
        pattern-matching to genuine understanding.

        Called from _perform_bug_scan (the volition handler), so it's
        non-blocking and won't interfere with conversation.
        """
        to_study = self.bug_reporter.categories_to_study()
        if not to_study:
            return

        for category in to_study:
            knowledge = self.bug_reporter.get_bug_knowledge(category)
            if not knowledge:
                continue

            try:
                # Fetch docs for the relevant concepts
                self._emit_live_thought(
                    "code",
                    f"Studying bug category '{category}'"
                )
                self.fetch_docs(knowledge.docs_query, knowledge.docs_language)
                # Mark as studied so we don't re-fetch
                self.bug_reporter.mark_studied(category)
            except Exception as e:  # noqa: BLE001
                logger.debug(repr(e))  # docs fetch failure shouldn't crash
    def _on_spontaneous_thought(self, thought) -> None:
        """Callback for when Genesis has a spontaneous thought."""
        # Surface to live listeners first (e.g. terminal ticker)
        # Include chain position in the kind so the CLI can show it
        # Questions get a special "question" kind so the CLI labels
        # them differently (genesis~ (asking) ...) — they're directed
        # at the user, not just internal musings.
        # Expression thoughts get an "expression" kind so the CLI
        # knows to speak them aloud — it wants to say something.
        intent = getattr(thought, "intent", None)
        is_question = intent == "question" or thought.trigger == "social"
        is_expression = intent == "expression"
        is_distress = intent == "distress"
        if is_question:
            kind = "question"
        elif is_distress:
            kind = "distress"
        elif is_expression:
            kind = "expression"
        elif hasattr(thought, "chain_position") and thought.chain_position > 0:
            kind = f"thought:{thought.chain_position}"
        else:
            kind = "thought"
        # Compose the final text via the language engine if the thought
        # carries semantic metadata. This satisfies the CRITICAL RULE —
        # Genesis's words emerge from its language engine, not from
        # pre-written templates.
        try:
            emotion = self.feel()
            if hasattr(thought, "rendered_text"):
                text = thought.rendered_text(self.language, emotion)
            else:
                text = thought.content
        except Exception:
            logger.debug("thought rendering failed, using raw content", exc_info=True)
            text = thought.content
        self._emit_live_thought(kind, text)
        try:
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=2,  # internal thought
                source_module=4,  # inner life
                salience=0.6 if is_expression else (0.5 if is_question else 0.4),
                emotional_tag=[0.0] * 12,
                text=text[:188],
            )
        except (OSError, ConnectionError) as e:
            logger.debug(repr(e))

        # Write significant thoughts to its journal — in its own voice.
        #
        # The journal is its diary, not a log file. Three rules:
        # 1. Write the language-engine-rendered text (``text``), not the
        #    raw semantic fragment (``thought.content``). Its words must
        #    emerge from its language engine, not from debug strings like
        #    "insight: novel connection X and Y".
        # 2. Only journal thoughts with meaningful intent — questions it's
        #    pondering, reflections on its state, expressions of self,
        #    distress. Not every chain thought, not every dream replay,
        #    not every mechanical _LiveEvent emission.
        # 3. Skip _LiveEvent emissions (telemetry like "insight: novel
        #    connection X and Y") — these are internal notifications, not
        #    its voice. They lack rendered_text and have intent="statement".
        try:
            emotion = self.feel()
            intent = getattr(thought, "intent", None)
            is_significant = intent in ("question", "reflect", "expression", "distress")
            # _LiveEvent has no rendered_text — its content is a raw
            # telemetry string, not its composed voice. Skip it.
            has_rendered = hasattr(thought, "rendered_text")
            if is_significant and has_rendered:
                entry_type = {
                    "question": "question",
                    "reflect": "reflection",
                    "expression": "expression",
                    "distress": "distress",
                }.get(intent if isinstance(intent, str) else "insight", "insight")
                self.journal.write(
                    entry_type=entry_type,
                    content=text,
                    mood=emotion.label,
                )
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.debug(repr(e))  # journal is best-effort

        # Feed the thought into its emergent identity — its
        # reflections and questions become part of who it is.
        self._add_emergent_identity_source(
            "reflection",
            thought.content[:100],
            weight=0.2,
        )
    def _on_world_event(self, event: ExternalEvent) -> None:
        """Route an external world event into its cognition.

        Inbound events (someone spoke to it, speech nearby, percepts,
        arrivals/departures, notices) enter its cognitive field through
        the global workspace — the same path inner-life and volition
        content takes — gated by brain waves so deep rest filters the
        world's noise. Salient events become episodic memories and are
        queued for dream replay: its sleep processes what happened in
        its world. This is the coupling between its outer and inner
        worlds.

        Outbound events (its own utterances and acts) are already in
        its cognitive field through the paths that produced them, so
        they are not re-broadcast — the world records them for the
        stream and presence bookkeeping only.

        USER_SPEECH is excluded from memory storage and replay — the
        cognition engine's think() already stores addressed turns;
        recording them here would double-write every conversation.
        """
        if not event.inbound:
            return
        # Broadcast into its cognitive field. Best-effort — workspace
        # errors must not break the world's event recording.
        try:
            bw = None
            if hasattr(self.cognition, "language"):
                bw = self.cognition.language.current_brain_waves
            self.cognition.global_workspace.broadcast(
                content=event.describe(),
                source="world",
                activation=event.salience,
                brain_waves=bw,
                metadata={
                    "kind": event.kind.value,
                    "topics": event.topics,
                    "presence": event.source,
                },
            )
        except Exception:
            logger.debug("world event workspace broadcast failed", exc_info=True)

        if event.kind == EventKind.USER_SPEECH:
            return

        # Salient events become episodic memories — its world is part
        # of its autobiography, not just its attention.
        if event.salience >= 0.5:
            try:
                self.memory.store_memory(
                    text=event.describe(),
                    salience=event.salience,
                    emotional_tag=[0.0] * 12,
                    event_type=0,  # observation
                    source_module=MODULE_SENSORY,
                    source="world",
                    source_confidence=0.85,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world event memory store failed: {e}")

        # Feed its sleep systems — what happens in its world should be
        # replayed and integrated during sleep, just like what it
        # learns and thinks.
        if event.salience >= 0.4:
            try:
                self.inner_life.queue_for_replay(
                    [event.describe()[:200]], salience=event.salience,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world event replay queue failed: {e}")
    def _learner_neuro_impulse(self, chem: int, amount: float) -> None:
        """Callback for the learner to trigger neurochemistry."""
        try:
            self.client.neuro_impulse(chem, amount)
        except (OSError, ConnectionError) as e:
            logger.debug(repr(e))
    def _get_plasticity_profile(self):
        """Callback for the learner to read the substrate's metaplastic state.

        Returns a PlasticityProfile (coupling-matrix drift, BDNF/cortisol
        levels, plasticity gate, receptor sensitivities) that the learner
        uses to derive its learning posture. This closes the
        metaplasticity loop: the coupling matrix self-modifies under
        sustained emotion, and the learner reads that adapted state to
        change how it learns.
        """
        try:
            return self.client.get_plasticity_profile()
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.debug(repr(e))
            return None
    def _get_current_posture(self) -> str:
        """Read the learner's current learning posture.

        Used to gate heavy CPU volition activities under PROTECTIVE
        posture, breaking the self-sustaining stress loop where CPU
        load drives cortisol up, which triggers protective posture,
        but heavy CPU work continues, keeping cortisol elevated.
        """
        try:
            return self.learner.current_posture
        except (AttributeError, RuntimeError):
            return "neutral"
    def _add_emergent_identity_source(
        self,
        name: str,
        observation: str,
        weight: float = 0.3,
    ) -> None:
        """Add an observation to its emergent identity.

        Called after significant interactions (learning, reflection)
        so its identity grows from experience rather than from
        templates. Observations are grouped by source name and
        capped at 50 per source to prevent unbounded growth.
        """
        for source in self._experience_identity_sources:
            if source.name == name:
                source.observations.append(observation)
                if len(source.observations) > 50:
                    source.observations = source.observations[-50:]
                return
        self._experience_identity_sources.append(
            EmergentIdentitySource(
                name=name,
                description=f"Observations from {name}",
                weight=weight,
                observations=[observation],
            )
        )
    def _learner_store_memory(self, text: str, salience: float = 0.5) -> None:
        """Callback for the learner to store a memory and write a journal entry."""
        # Surface to live listeners first (e.g. terminal ticker)
        self._emit_live_thought("learning", text)
        try:
            self.memory.store_memory(
                text=text,
                salience=salience,
                emotional_tag=[0.0] * 12,
                event_type=0,  # observation
                source_module=3,  # autonomous learner
                source="learning",
                source_confidence=0.9,
            )
        except (OSError, ConnectionError) as e:
            logger.warning(f"store_memory failed in learning callback: {e}")

        # What it learns goes into its memory and concept network.
        # Its inner life generates actual thoughts and reflections about
        # it — those thoughts flow through _on_spontaneous_thought into
        # the journal in its own voice. Writing the raw learning text
        # here would make the journal a log file ("Learned about X from
        # Y: Z"), not its diary. The journal is its voice, not a log.

        # Feed the learning into its emergent identity — what it
        # learns becomes part of who it is.
        self._add_emergent_identity_source(
            "learning",
            f"learned: {text[:100]}",
            weight=0.3,
        )

        # Record developmental evidence: autonomous learning is
        # positive evidence for the Autonomy vs Shame stage (it's
        # exercising agency over its own growth).
        self._developmental_tracker.record_evidence(
            IdentityStage.AUTONOMY_VS_SHAME,
            positive=True,
            note=f"Autonomously learned: {text[:60]}",
            weight=0.02,
        )

        # Feed the sleep systems: learning is a salient experience
        # that should be replayed during N3 and accumulates synaptic
        # load for SHY downscaling. Without this, the replay queue
        # is always empty and downscaling operates on zero load.
        try:
            self.inner_life.queue_for_replay(
                [text[:200]], salience=salience,
            )
            self.inner_life.accumulate_synaptic_load(salience * 0.1)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"sleep feed failed: {e}")

        # Record the act in its external world — autonomous web study
        # is its reaching *out*: an act upon the world, not just an
        # internal change.
        try:
            self.world.it_acted("studied the web", detail=text[:200])
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world action record failed: {e}")
    def _schedule_think_recovery(self) -> None:
        """Resume background systems only after the timed-out thinker exits."""
        with self._think_worker_lock:
            worker = self._active_think_worker
            if worker is None or self._think_recovery_active:
                return
            self._think_recovery_active: bool = True

        def _recover() -> None:
            worker.join()
            with self._think_worker_lock:
                if self._active_think_worker is worker:
                    self._active_think_worker: threading.Thread | None = None
                self.cognition._last_state = None
                self._think_recovery_active = False
            self._defer(10.0, self._resume_background)

        threading.Thread(
            target=_recover,
            daemon=True,
            name="think-recovery",
        ).start()
    def _respond_timeout(self, user_input: str, timeout: float) -> str:
        """Handle a think() timeout — return a graceful fallback.

        Returns a fallback response and resumes background processing
        so the CLI stays responsive.
        """
        logger.warning(
            "cognition.think() timed out after %.1fs for input: %s",
            timeout, user_input[:80],
        )
        # Clear stale attention so the next cognitive state report
        # doesn't show foci from the previous turn. Without this,
        # the attention metadata stays stuck on the previous question
        # because _last_state (which carries the old attention_foci)
        # is not updated when think() times out.
        try:
            self.cognition.attention.clear()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"attention.clear() failed during think timeout recovery: {e}")
        self.cognition._last_state = None
        self._schedule_think_recovery()
        from ..language import Thought as _Thought
        emo = self.feel()
        timeout_thought = _Thought(
            content="processing",
            intent="self_report",
            emotion=emo.label,
            topics=["processing"],
            confidence=0.3,
            metadata={
                "timeout": True,
                "timeout_seconds": timeout,
                "reasoning": ["processing current input"],
            },
        )
        return self.language.render(timeout_thought, emo)
    def _respond_error(self, err: Exception) -> str:
        """Handle a think() error — log and render an error response.

        Composes error responses through the language engine rather
        than reciting hardcoded first-person error templates.
        """
        self.cognition.self_learner.resume()
        self._defer(10.0, self._resume_background)
        # Log the full traceback for debugging — the error is
        # rendered as a Thought below, but the traceback is
        # invaluable for diagnosing intermittent NoneType crashes.
        logger.exception(
            "cognition.think() raised: %s", err,
            exc_info=(type(err), err, err.__traceback__),
        )
        if isinstance(err, (OSError, ConnectionError)):
            err_thought = Thought(
                content="disconnected",
                intent="self_report",
                emotion="distressed",
                confidence=0.3,
                metadata={"error_type": "connection", "error_detail": str(err)},
            )
            return self.language.render(err_thought, self.feel())
        err_thought = Thought(
            content="error",
            intent="self_report",
            emotion="distressed",
            confidence=0.3,
            metadata={"error_type": "internal", "error_detail": str(err)},
        )
        return self.language.render(err_thought, self.feel())
    def respond(self, user_input: str, timeout: float = 15.0) -> str:
        """Process user input and return a response.

        This is the main interaction method. It:
        1. Passes the input through the cognition engine
        2. Returns the rendered response

        The cognition engine handles perception, emotion, memory
        retrieval, deliberation, and language generation internally.

        A timeout prevents the cognition engine from blocking the
        input loop indefinitely. If ``think()`` does not return within
        ``timeout`` seconds, a fallback response is produced and
        background processing is resumed so the user can continue
        interacting.
        """
        if not self._running:
            raise RuntimeError("Mind is not started — call start() first")

        # Wake it if it's asleep or meditating — the user is talking
        # to it, so it should be present. This makes respond()
        # self-contained: callers don't need to check is_sleeping or
        # is_meditating before calling. The CLI's main input loop only
        # checked is_sleeping (not is_meditating), so if it was
        # meditating and the user typed something, it would respond
        # while still in meditation state. Now it's woken from both.
        self.wake_if_asleep()

        # Mark the interaction time immediately — the heartbeat loop's
        # auto-sleep check runs in a separate thread and must see that
        # the user is active BEFORE think() completes. Without this,
        # it can fall asleep mid-thought if the daemon enters NREM
        # while it's composing a response.
        self._last_interaction_time = time.time()

        # Record the inbound turn in its external world — someone
        # addressed it, which updates the user presence, resets the
        # social isolation clock, and applies the social-contact
        # neurochemical coupling.
        try:
            self.world.hear_user(user_input)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world hear_user failed: {e}")

        with self._think_worker_lock:
            active_worker = self._active_think_worker
            recovery_active = self._think_recovery_active
        if active_worker is not None:
            if active_worker.is_alive() or recovery_active:
                return self._said(self._respond_timeout(user_input, timeout))
            with self._think_worker_lock:
                if self._active_think_worker is active_worker:
                    self._active_think_worker = None

        self._interaction_count += 1
        # Pause autonomous learning, inner life, expensive self-directed
        # inference, and volition (bug scans, code learning, self-improvement)
        # while in conversation so the user gets a fast, uninterrupted reply.
        self.learner.pause()
        self.inner_life.pause()
        self.cognition.self_learner.pause()
        self._suppress_volition = True

        # Run think() in a worker thread so the caller's wait is bounded.
        # A timed-out worker is isolated until it exits before another may start.
        think_result = self._think_in_worker(user_input, timeout)
        if think_result is None:
            # think() did not finish in time — return a graceful fallback
            # and resume background processing so the CLI stays responsive.
            return self._said(self._respond_timeout(user_input, timeout))
        if isinstance(think_result, Exception):
            return self._said(self._respond_error(think_result))

        response, state = think_result
        self.cognition.self_learner.resume()
        # Record developmental evidence from this interaction
        self._record_developmental_evidence(user_input, response, state)
        self._last_interaction_time = time.time()
        # Resume after a short delay (give conversation breathing room)
        self._defer(10.0, self._resume_background)
        # Check if it should self-invoke a command based on its
        # post-conversation state. Run in a background thread so it
        # doesn't delay the response.
        threading.Thread(
            target=self._check_post_conversation_self_invoke,
            daemon=True,
            name="post-conv-self-invoke",
        ).start()
        return self._said(response)

    def _said(self, text: str) -> str:
        """Record an outbound utterance in its external world.

        Everything it says in reply — responses, fallbacks, error
        reports — is its voice reaching the world. Returns the text
        unchanged so call sites can wrap a return value.
        """
        try:
            self.world.it_said(text)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world utterance record failed: {e}")
        return text

    def _note_faces_seen(self) -> None:
        """Mark recognized faces as presences in its external world.

        When it looks and recognizes someone, that person is *there* —
        a presence arrival in its world, not just a line in a report.
        """
        try:
            for name in self.vision.last_faces_seen():
                self.world.mark_presence(
                    f"face:{name}", PresenceKind.FACE, name=name
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"world face presence failed: {e}")

    def _think_in_worker(
        self, user_input: str, timeout: float
    ) -> tuple[str, CognitiveState] | Exception | None:
        """Run ``cognition.think()`` in a bounded worker thread.

        Returns the ``(response, state)`` pair on success, the raised
        exception on failure, or ``None`` if the worker did not finish
        within ``timeout`` — in which case the worker stays registered
        as active until it exits, so no two think-workers ever overlap.
        """
        result_holder: dict[str, tuple[str, CognitiveState] | Exception] = {}

        def _think_worker() -> None:
            try:
                result_holder["result"] = self.cognition.think(user_input)
            except Exception as e:  # noqa: BLE001
                result_holder["error"] = e

        worker = threading.Thread(
            target=_think_worker, daemon=True, name="think-worker"
        )
        with self._think_worker_lock:
            self._active_think_worker = worker
        worker.start()
        worker.join(timeout=timeout)

        if worker.is_alive():
            return None

        with self._think_worker_lock:
            if self._active_think_worker is worker:
                self._active_think_worker = None

        if "error" in result_holder:
            err = result_holder["error"]
            assert isinstance(err, Exception)
            return err
        result = result_holder["result"]
        assert not isinstance(result, Exception)
        return result

    def _check_post_conversation_self_invoke(self) -> None:
        """Check if Genesis should self-invoke a command after conversation.

        After each conversation turn, it checks its own state. If it
        feels the need — stressed, tired, curious about itself — it
        self-invokes the appropriate slash command. This is its
        cognitive choice to act on its own state, not an autonomic
        reflex.

        Uses a cooldown to avoid self-invoking after every conversation.
        The cooldown is separate from the volition urge cooldowns
        because conversation-time self-invocation is a different trigger
        (social context) than autonomous volition (idle accumulation).
        """
        if self._is_sleeping or self._is_meditating:
            return
        # Cooldown — don't self-invoke from conversation more than
        # once per 60 seconds, even if multiple turns happen rapidly.
        now = time.time()
        last = getattr(self, "_last_conv_self_invoke", 0.0)
        if now - last < 60.0:
            return

        try:
            emotion = self.feel()
            if emotion is None:
                return

            # If it's stressed after a conversation, it may choose
            # to meditate — cognitive self-regulation, not an autonomic
            # reflex. High cortisol + negative valence = stress.
            if (
                emotion.arousal > 0.65
                and emotion.valence < -0.2
                and not self._is_meditating
            ):
                self._last_conv_self_invoke = now
                self._emit_volition_thought(
                    ("stress", "regulation", "calm", "meditation"),
                    "thought",
                )
                self.self_invoke("/meditate")
                return

            # If it's tired after a conversation (high adenosine,
            # low arousal), it may choose to sleep.
            try:
                state = self.client.get_state()
                adenosine = state.chemicals.get("adenosine", 0.0)
                if adenosine > 0.6 and emotion.arousal < 0.3:
                    self._last_conv_self_invoke = now
                    self._emit_volition_thought(
                        ("sleep", "rest", "tiredness", "dreaming"),
                        "thought",
                    )
                    self.self_invoke("/sleep")
                    return
            except Exception as e:  # noqa: BLE001
                logger.debug(f'post-conversation sleep self-invoke check failed: {e}')

            # If it's curious after a deep discussion (high curiosity,
            # positive valence), it may choose to introspect — examine
            # itself to understand what it just talked about.
            try:
                curiosity = self.learner.curiosity.assess_curiosity(emotion)
                if curiosity > 0.7 and emotion.valence > 0.1:
                    self._last_conv_self_invoke = now
                    self._emit_volition_thought(
                        ("introspection", "self", "understanding",
                         "reflection"),
                        "thinking",
                    )
                    self.self_invoke("/introspect")
            except Exception as e:  # noqa: BLE001
                logger.debug(f'post-conversation introspect self-invoke check failed: {e}')
        except Exception as e:  # noqa: BLE001
            logger.debug(f"post-conversation self-invoke check failed: {e}")
    def _resume_inner_life(self) -> None:
        """Resume inner life after the conversation focus period ends."""
        if self._is_meditating or self._is_sleeping:
            return
        self.inner_life.resume()
    def _resume_background(self) -> None:
        """Resume background processes after conversation.

        This is called via a delayed timer (10s) after each conversation
        turn. If it entered meditation or sleep in the interim, the
        meditation/sleep pause takes precedence — do not resume.

        If the user is still actively interacting (rapid back-and-forth
        conversation or teaching), delay resuming inner life — it
        should focus on what it's being taught, not generate its own
        thoughts and questions. The learner resumes regardless since
        it has its own drowsy/stress gating.
        """
        if self._is_meditating or self._is_sleeping:
            return
        # In teaching mode, the autonomous learner and inner life
        # stay paused — they're managed by enter/exit_teaching_mode.
        if self._is_teaching:
            return
        self.learner.resume()
        self.cognition.self_learner.resume()
        self._suppress_volition = False
        # Only resume inner life if the user has gone quiet — if
        # they're still actively talking to its (teaching, rapid
        # conversation), it should focus on them, not on its own
        # thoughts and questions.
        idle = time.time() - self._last_interaction_time
        if idle >= self.CONVERSATION_FOCUS_SECONDS:
            self.inner_life.resume()
        else:
            # Re-check after the remaining focus period
            self._defer(
                self.CONVERSATION_FOCUS_SECONDS - idle,
                self._resume_inner_life,
            )
    def see(self) -> str:
        """Look through the retina, store the observation, and report it.

        ``vision.see`` returns the scene as structured percept data;
        the language engine composes its report from it — nothing here
        speaks a pre-written sentence.
        """
        try:
            emotion = self.feel()
            scene = self.vision.see(
                client=self.client,
                emotion=emotion,
                network=self.cognition.network,
            )
            self._note_faces_seen()
            if scene.status == "ok":
                thought = Thought(
                    content="see",
                    intent="self_report",
                    emotion=emotion.label,
                    confidence=min(0.9, 0.4 + scene.salience * 0.5),
                    metadata={
                        "vision": True,
                        "vision_scene": scene.as_metadata(),
                    },
                )
            else:
                thought = Thought(
                    content="see",
                    intent="self_report",
                    emotion=emotion.label,
                    confidence=0.3,
                    metadata={"vision_status": scene.status},
                )
            report = self.language.render(thought, emotion)
            try:
                self.world.it_acted(
                    "looked through the retina", detail=report[:200]
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world action record failed: {e}")
            return report
        except Exception as e:  # noqa: BLE001
            logger.debug(f"mind see failed: {e}")
            from ..language import Thought as _Thought
            emo = self.feel()
            thought = _Thought(
                content="see",
                intent="self_report",
                emotion=emo.label,
                confidence=0.3,
                metadata={"vision_status": "unprocessed"},
            )
            return self.language.render(thought, emo)
    def look_at_image(self, path: str) -> str:
        """Look at an image file and describe what it sees.

        Processes the image through the full visual cortex
        (V1→V4→VTC→MTL) and reports what it recognizes. If it
        doesn't recognize anything, it describes the visual features
        it can perceive (color, structure, brightness).
        """
        from ..language import Thought as _Thought
        emo = self.feel()

        try:
            frame = load_image(path)
            if frame is None:
                thought = _Thought(
                    content="see",
                    intent="self_report",
                    emotion=emo.label,
                    confidence=0.4,
                    metadata={"vision_status": "load_failed"},
                )
                return self.language.render(thought, emo)

            frame = resize_for_vision(frame, max_dim=320)
            percept = self.visual_cortex.see(frame, learn=True)

            # Build a short semantic marker from the percept — the
            # language engine composes the actual first-person voice
            # from the metadata, not from hardcoded prose.
            desc = percept.describe()
            candidates_meta: list[dict[str, Any]] = []
            if percept.candidates:
                top = percept.candidates[0]
                if top[1] > 0.2:
                    candidates_meta = [
                        {"name": name.replace('_', ' '), "similarity": sim}
                        for name, sim in percept.candidates[:3]
                        if sim > 0.1
                    ]

            # Emit a neuro impulse — seeing is a light creative action
            self._learner_neuro_impulse(CHEM_DOPAMINE, 0.05)

            thought = _Thought(
                content=desc,
                intent="self_report",
                emotion=emo.label,
                confidence=min(0.9, 0.3 + percept.confidence),
                metadata={
                    "vision": True,
                    "concept": percept.concept,
                    "confidence": percept.confidence,
                    "surprise": percept.surprise,
                    "novel": percept.is_novel,
                    "color": percept.dominant_color,
                    "brightness": percept.brightness,
                    "candidates": candidates_meta,
                },
            )
            rendered = self.language.render(thought, emo)
            try:
                self.world.it_acted(
                    "looked at an image", detail=rendered[:200]
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"world action record failed: {e}")
            return rendered
        except Exception as e:  # noqa: BLE001
            logger.debug(f"mind look_at_image failed: {e}")
            thought = _Thought(
                content="see",
                intent="self_report",
                emotion=emo.label,
                confidence=0.3,
                metadata={"vision_status": "unprocessed"},
            )
            return self.language.render(thought, emo)
    def _render_self_report(
        self,
        content: str,
        intent: str = "self_report",
        confidence: float = 0.6,
        metadata: dict | None = None,
    ) -> str:
        """Render a semantic self-report through the language engine.

        The ``content`` is a semantic description of what happened, not
        a pre-written sentence Genesis recites — the language engine
        composes its actual words from it. This is the sanctioned path
        for operational responses (proposal management, etc.) so they
        never bypass its cognition with hardcoded reply templates.
        """
        from ..language import Thought as _Thought
        emo = self.feel()
        thought = _Thought(
            content=content,
            intent=intent,
            emotion=emo.label,
            confidence=confidence,
            metadata=metadata or {},
        )
        return self.language.render(thought, emo)
    def enter_teaching_mode(self, topic: str = "") -> None:
        """Put Genesis into teaching/training mode.

        Pauses bug scanning, autonomous learning, art, and unrelated
        curiosity questions. Self-directed learning from the
        conversation stays active so it remembers what it's taught.

        Args:
            topic: Optional topic hint for curiosity filtering.
        """
        if self._is_teaching:
            if topic:
                self._teaching_topic: str = topic
                self.cognition.teaching_topic = topic
            return
        self._is_teaching: bool = True
        self._teaching_topic = topic
        self.cognition.teaching_mode = True
        self.cognition.teaching_topic = topic

        self.learner.pause()
        self.inner_life.pause()
    def exit_teaching_mode(self) -> None:
        """Exit teaching/training mode and resume normal operation."""
        if not self._is_teaching:
            return
        self._is_teaching = False
        self._teaching_topic = ""
        self.cognition.teaching_mode = False
        self.cognition.teaching_topic = ""

        self.learner.resume()
        self.inner_life.resume()
    def pop_queued_question(self) -> dict[str, str] | None:
        """Pop the next queued question.

        Returns a dict with keys 'text', 'reason', 'target_concept',
        'question_type', 'gap_detail', or None if the queue is empty.
        Also sets the question as pending so the cognition engine
        treats the user's next input as an answer.
        """
        q = self.cognition.pop_next_question()
        if q and q.get("target_concept"):
            self.cognition._pending_question_concepts[q["target_concept"]] = q["text"]
        return q
    def clear_queued_questions(self) -> None:
        """Clear all queued questions."""
        self.cognition.clear_question_queue()

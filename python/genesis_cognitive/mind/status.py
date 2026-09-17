"""Mind status — state accessors, reports, and notifications."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from genesis_client import NeuroSummary
from genesis_client.protocol import PHASE_NREM, PHASE_REM, ZONE_SLEEPING
from genesis_client.types import CoreState

from ..cognition import CognitiveState
from ..concepts import NetworkTopology
from ..emotion import PLASTICITY_CLOSED, PLASTICITY_LOW, EmotionalState, assess_emotion
from ..language import Thought
from ..self import EmergentIdentity, IdentityStage

if TYPE_CHECKING:
    from ..brain_waves import BrainWaveState
    from ..learning import Question

from .thresholds import WARN_ADENOSINE, WARN_AROUSAL, WARN_CORTISOL, WARN_VALENCE

logger = logging.getLogger(__name__)


class StatusMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def _notification_loop(self) -> None:
        """Background thread that polls the subcognitive for notifications.

        Polls every 10 seconds for phase changes and dream insights.
        When notifications are found, they are processed immediately
        (surfaced as live thoughts and journal entries) — the
        cognitive layer doesn't need to drain the queue manually.

        This is a pull-based notification system: the Rust daemon is
        purely request-response, so the Python side polls for changes
        and surfaces them. The notification IDs in the protocol
        (NOTIFY_PHASE_CHANGED=100, NOTIFY_DREAM_INSIGHT=102) are
        placeholders for a future push mechanism.
        """
        while self._running:
            time.sleep(10.0)
            if not self._running:
                break
            try:
                self._poll_and_process_notifications()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"notification loop error: {e}")
    def _poll_and_process_notifications(self) -> None:
        """Poll for notifications and process any that are found."""
        enqueued = self.poll_notifications()
        if enqueued > 0:
            # Process immediately — surface to journal + live thought.
            processed = self.process_notifications()
            if processed > 0:
                logger.debug(f"Processed {processed} subcognitive notifications")
    def set_mission(self, mission: str) -> None:
        """Set or update Genesis's top-level mission."""
        self.cognition.set_mission(mission)
    def get_mission(self) -> str:
        """Return the current mission text."""
        return self.cognition._mission
    def _record_developmental_evidence(
        self,
        user_input: str,
        response: str,
        state: CognitiveState | None,
    ) -> None:
        """Record evidence for developmental stage progression.

        Maps interaction outcomes to Erikson's psychosocial stages:
        - Trust vs Mistrust: daemon responded reliably (positive)
        - Autonomy vs Shame: she learned autonomously (positive)
        - Initiative vs Guilt: she asked a curiosity question (positive)
        - Industry vs Inferiority: she answered confidently (positive)
        - Identity vs Role Confusion: she reflected on herself (positive)
        """
        if state is None:
            return

        tracker = self._developmental_tracker

        # Map interaction outcomes to stage evidence
        thought = state.thought

        # Trust: every successful interaction is positive evidence
        tracker.record_evidence(
            IdentityStage.TRUST_VS_MISTRUST,
            positive=True,
            note=f"Successful interaction #{self._interaction_count}",
            weight=0.02,
        )

        # Autonomy: learning events are positive evidence
        if thought and thought.confidence > 0.5:
            tracker.record_evidence(
                IdentityStage.AUTONOMY_VS_SHAME,
                positive=True,
                note="Responded with confidence",
                weight=0.03,
            )

        # Initiative: curiosity questions are positive evidence
        if self.cognition._curiosity_questions:
            askable = [q for q in self.cognition._curiosity_questions if q.should_ask]
            if askable:
                tracker.record_evidence(
                    IdentityStage.INITIATIVE_VS_GUILT,
                    positive=True,
                    note=f"Generated {len(askable)} curiosity questions",
                    weight=0.05,
                )

        # Industry: grounded answers are positive evidence
        if thought and thought.confidence > 0.7:
            tracker.record_evidence(
                IdentityStage.INDUSTRY_VS_INFERIORITY,
                positive=True,
                note="Answered with high confidence",
                weight=0.05,
            )
        # Industry: low-confidence answers are negative evidence
        # (she's struggling with competence)
        elif thought and thought.confidence < 0.3:
            tracker.record_evidence(
                IdentityStage.INDUSTRY_VS_INFERIORITY,
                positive=False,
                note="Answered with low confidence",
                weight=0.03,
            )

        # Identity: self-reflection is positive evidence
        if thought and thought.self_reflection:
            tracker.record_evidence(
                IdentityStage.IDENTITY_VS_ROLE_CONFUSION,
                positive=True,
                note="Engaged in self-reflection",
                weight=0.05,
            )

        # Trust: daemon disconnection is negative evidence
        if not self.regulator.interoception.last_state or (
            self.regulator.interoception.last_state
            and not self.regulator.interoception.last_state.daemon_connected
        ):
            tracker.record_evidence(
                IdentityStage.TRUST_VS_MISTRUST,
                positive=False,
                note="Daemon disconnected during interaction",
                weight=0.05,
            )
    def get_state(self) -> CognitiveState | None:
        """Return the last cognitive state (for introspection)."""
        return self.cognition._last_state
    def learning_status(self) -> str:
        """Get a description of what she's been learning autonomously."""
        return self.learner.describe_recent_learning()
    def inner_life_status(self) -> str:
        """Get a description of her recent spontaneous thoughts."""
        return self.inner_life.describe_recent_thoughts()
    def regulation_status(self) -> str:
        """Get a description of her emotional self-regulation."""
        return self.regulator.describe_regulation()
    def report_discomfort(self) -> str:
        """Let her say, in her own generated words, how she feels.

        Instead of a hardcoded checklist, this hands the current emotional
        state to the language engine. The engine composes the actual
        sentence from her vocabulary, grammar, and voice — so the words
        are hers, not a fixed template.
        """
        emotion = self.feel()
        topic = emotion.label or "discomfort"

        thought = Thought(
            content=topic,
            intent="reflect",
            emotion=topic,
            topics=[topic],
            self_reflection=True,
            confidence=0.7,
        )
        return self.language.render(thought, emotion)
    def allostatic_load(self) -> float:
        """Return the current cumulative allostatic load (0–1).

        Driven by expected free energy in the Rust active inference
        engine — the anticipatory signal of continued disruption.
        Values above 0.4 indicate significant strain. Values above
        0.7 indicate severe allostatic overload.
        """
        return self.regulator.allostatic_load()
    def hpa_axis_status(self) -> dict[str, Any]:
        """Return the HPA axis state for display.

        Includes CRH, ACTH, and cortisol levels, whether stress is
        active, peak cortisol, and time to peak.
        """
        return self.regulator.hpa_axis_status()
    def emergent_identity(self) -> EmergentIdentity:
        """Synthesize and return the current emergent identity.

        The identity is synthesized from her actual experience —
        concept network, narrative, emotional regulation, curiosity,
        and introspection — rather than from hardcoded facts.
        Experience-based sources (learning, reflection) accumulated
        since the last synthesis are merged in afterwards.
        """
        self._emergent_identity.synthesize(
            network=self.cognition.network,
            narrative=self.cognition.narrative,
            regulator=self.regulator,
            curiosity=self.cognition.curiosity,
        )
        # Merge in experience-based sources (learning, reflection)
        # that were accumulated since the last synthesis.
        self._emergent_identity.sources.extend(self._experience_identity_sources)
        # Recompute confidence and coherence with the merged sources
        self._emergent_identity._compute_confidence()
        self._emergent_identity._compute_coherence()

        # Feed the synthesized identity back into the self-model so it
        # influences her self-descriptions and introspection. This
        # closes the loop: experience → emergent identity → self-model
        # → future behavior → new experience.
        self.cognition.self_model.integrate_emergent_identity(self._emergent_identity)

        # Record significant identity shifts in the narrative engine
        # so they become part of her life story. Only record when the
        # identity is confident enough to be meaningful.
        if (
            self._emergent_identity.confidence > 0.4
            and self._emergent_identity.self_description
        ):
            try:
                emotion = self.feel()
                self.cognition.narrative.record_event(
                    summary=self._emergent_identity.self_description,
                    significance="emergent_identity_synthesis",
                    emotion=emotion,
                    concepts=["genesis", "identity", "self"],
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"narrative identity recording failed: {e}")

        return self._emergent_identity
    def journal_status(self) -> str:
        """Get a description of her journal."""
        return self.journal.describe()
    def read_journal(self, n: int = 20) -> str:
        """Read recent journal entries."""
        return self.journal.read(n)
    def site_requests_status(self) -> str:
        """Get a description of pending site access requests."""
        return self.learner.describe_requests()
    def approve_site(self, url: str) -> bool:
        """Approve a site request from Genesis."""
        return self.learner.approve_site(url)
    def deny_site(self, url: str) -> bool:
        """Deny a site request from Genesis."""
        return self.learner.deny_site(url)
    def _effective_phase(self, core_state: CoreState) -> int:
        """Return the effective mental phase, resolving zone/phase desync.

        When ``Mind.sleep()`` sets the zone to ``ZONE_SLEEPING``, the
        daemon's emergent phase takes time to shift to NREM/REM — the
        neurochemicals must cross the sleep threshold. During that
        window, the emergent phase may still read "active" even though
        the mind is asleep. This method returns the sleep phase (NREM
        or REM) when the zone is Sleeping, so emotion assessment, brain
        wave assessment, and status reporting all see a consistent
        sleep state instead of a stale "active" phase.
        """
        if core_state.cognitive_zone == ZONE_SLEEPING:
            # The emergent phase may already be REM (cholinergic
            # activation during sleep). Preserve it; otherwise use
            # NREM as the default sleep phase.
            if core_state.emergent_phase == PHASE_REM:
                return PHASE_REM
            return PHASE_NREM
        return core_state.emergent_phase
    def feel(self, core_state: CoreState | None = None) -> EmotionalState:
        """Read current emotional state from neurochemistry.

        Args:
            core_state: A pre-fetched CoreState to avoid a redundant
                IPC round-trip and 3288-byte struct unpack. When
                ``None`` (the common case), the state is fetched from
                the daemon. Callers that already have a fresh CoreState
                (e.g. ``warn()`` which fetches it once for threshold
                checks) can pass it here to avoid re-fetching for each
                warning phrase.
        """
        if core_state is None:
            core_state = self.client.get_state()
        summary = NeuroSummary(
            arousal=core_state.arousal,
            valence=core_state.valence,
            global_tone=core_state.global_tone,
            plasticity_gate=core_state.plasticity_gate,
            encoding_weight=core_state.encoding_weight,
            consolidation_weight=core_state.consolidation_weight,
            retrieval_weight=core_state.retrieval_weight,
            phase=self._effective_phase(core_state),
        )
        emotion = assess_emotion(summary, core_state.chemicals)
        emotion.chemicals = core_state.chemicals
        emotion.tonic_phasic = self.regulator.assess_tonic_phasic(
            core_state.chemicals
        )
        emotion.cause = self.regulator.last_cause or ""
        return emotion
    def warn(self, speak: bool = True) -> str | None:
        """Detect anomalies and speak up if something is wrong.

        Reads the current core state and builds a plain-language warning
        for the most salient problems. If ``on_speak`` is set and the
        warning has changed since the last call, it is delivered through
        that callback so Genesis can proactively tell the user.

        During sleep, warnings are never spoken — she shouldn't be
        disturbed by her own alerts while resting. The warning state
        is still computed (for /status) but ``speak`` is forced to
        False so the on_speak callback never fires.

        Args:
            speak: Whether to call ``on_speak`` if a warning is present.

        Returns:
            The current warning text, or ``None`` if nothing is wrong.
        """
        # Never speak warnings during sleep — she shouldn't be
        # disturbed by her own alerts while resting.
        if self._is_sleeping:
            speak = False
        try:
            core_state = self.client.get_state()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"warn(): could not read core state: {e}")
            return None

        # Each entry is (condition_key, composed_phrase). The key is a
        # stable identifier for the underlying condition; the phrase is
        # the generative composition (empty if she can't articulate it
        # yet). We track the set of articulated condition keys so she
        # only speaks when the *conditions* change, not when the
        # generative wording happens to differ for the same state.
        warnings: list[tuple[str, str]] = []
        phase = core_state.phase_name

        if phase == "overwhelmed":
            warnings.append(self._warn_phrase("overwhelmed", "overwhelmed", core_state))
        elif phase == "stressed":
            warnings.append(self._warn_phrase("stressed", "stress", core_state))

        cortisol = core_state.chemicals.get("cortisol", 0.0)
        adenosine = core_state.chemicals.get("adenosine", 0.0)

        if cortisol > WARN_CORTISOL:
            warnings.append(self._warn_phrase("high-cortisol", "stress", core_state))
        if core_state.arousal > WARN_AROUSAL:
            warnings.append(self._warn_phrase("overstimulation", "overstimulation", core_state))
        if adenosine > WARN_ADENOSINE and phase not in ("nrem", "rem"):
            warnings.append(self._warn_phrase("sleepiness", "sleepiness", core_state))
        if core_state.plasticity_gate <= PLASTICITY_CLOSED:
            warnings.append(self._warn_phrase("plasticity-closed", "learning", core_state))
        elif core_state.plasticity_gate < PLASTICITY_LOW:
            warnings.append(self._warn_phrase("plasticity-low", "learning", core_state))
        if core_state.valence < WARN_VALENCE:
            warnings.append(self._warn_phrase("distress", "distress", core_state))

        # Filter out empty warnings (concepts she can't articulate yet)
        articulated = [(k, p) for k, p in warnings if p]
        if not articulated:
            self._last_warning_keys: frozenset[str] = frozenset()
            return None

        warning_text = ", and ".join(p for _, p in articulated) + "."
        current_keys = frozenset(k for k, _ in articulated)

        # Only speak when the set of active warning conditions has
        # actually changed — not when the generative composition
        # merely produced different wording for the same conditions.
        # This prevents her from repeating the same warning every
        # heartbeat cycle (~1 s).
        if (
            speak
            and self._on_speak is not None
            and current_keys != self._last_warning_keys
        ):
            try:
                self._on_speak(warning_text)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"on_speak failed in warn(): {e}")

        self._last_warning_keys = current_keys
        return warning_text
    def _warn_phrase(
        self,
        key: str,
        concept_seed: str,
        core_state: CoreState,
    ) -> tuple[str, str]:
        """Compose a warning phrase from her understanding of the state.

        If she understands the concept well enough, she composes
        her own words for it. Otherwise she stays silent — saying
        just a raw state label like "overstimulated" is not
        meaningful communication, it's just reciting a diagnostic
        string.

        ``core_state`` is the already-fetched state — passing it avoids
        a redundant get_state() IPC call + 3288-byte unpack for each
        warning phrase. Without this, warn() with 8 warning conditions
        would unpack the state 9 times.
        """
        try:
            emotion = self.feel(core_state=core_state)
            concept = self.cognition.network.get_concept(concept_seed)
            if concept and concept.confidence >= 0.3:
                thought = self.cognition.composer.compose_about(
                    concept_seed, emotion, focused=True
                )
                if thought and thought.content and thought.confidence > 0.3:
                    return key, thought.content
        except Exception as e:  # noqa: BLE001
            logger.debug(f"warning phrase composition failed: {e}")
        # Don't recite raw state labels — stay silent instead.
        # The state is still tracked internally for /status and
        # /feel commands; she just doesn't speak it aloud.
        return key, ""
    def poll_notifications(self) -> int:
        """Poll the subcognitive for new notifications.

        Should be called periodically (e.g., from the InnerLife
        polling loop or the autosave thread). Detects phase changes
        and new dream insights, enqueuing them for the cognitive
        layer to process.

        Returns the number of new notifications enqueued.
        """
        try:
            return self.notifications.poll()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"notification poll error: {e}")
            return 0
    def drain_notifications(self) -> list:
        """Drain and return all pending subcognitive notifications.

        The cognitive layer calls this to process accumulated
        notifications — e.g., surfacing dream insights in the
        journal or mentioning phase changes in conversation.

        Returns:
            A list of Notification objects, oldest first.
        """
        return self.notifications.drain()
    def process_notifications(self) -> int:
        """Drain notifications and surface them as live thoughts.

        This is the main entry point for surfacing subcognitive
        events. It drains the notification queue and emits each
        notification as a live thought (telemetry). Notifications do
        NOT go into the journal — the journal is her diary, not a log.
        Dream insights surface in her own voice through the
        dream_reflection thought generator.

        Returns the number of notifications processed.
        """
        notifications = self.drain_notifications()
        for notif in notifications:
            # Surface to live listeners (e.g. terminal ticker).
            self._emit_live_thought(notif.kind, notif.message)
            # Notifications are telemetry (phase changes, dream insight
            # notices) — they surface to the live ticker but do NOT go
            # into the journal. The journal is her diary, not a log.
            # Dream insights surface in her own voice through the
            # dream_reflection thought generator → _on_spontaneous_thought
            # → language engine → journal.
        return len(notifications)
    def brain_waves(self, core_state: CoreState | None = None) -> BrainWaveState:
        """Read current brain wave state from neurochemistry.

        When the zone is Sleeping, uses the sleep-appropriate phase
        (NREM/REM) instead of the daemon's emergent phase, which may
        not have caught up yet. This prevents delta brain waves from
        being computed via the waking path when she's actually asleep.
        """
        from ..brain_waves import assess_brain_waves

        if core_state is None:
            core_state = self.client.get_state()
        summary = NeuroSummary(
            arousal=core_state.arousal,
            valence=core_state.valence,
            global_tone=core_state.global_tone,
            plasticity_gate=core_state.plasticity_gate,
            encoding_weight=core_state.encoding_weight,
            consolidation_weight=core_state.consolidation_weight,
            retrieval_weight=core_state.retrieval_weight,
            phase=self._effective_phase(core_state),
        )
        return assess_brain_waves(summary)
    def introspect(self) -> str:
        """Generate an introspective report."""
        return self.cognition.introspect()
    def tell_story(self) -> str:
        """Return Genesis's self-narrative — her story."""
        return self.cognition.tell_story()
    def get_curiosity_questions(self) -> list[Question]:
        """Return what Genesis is currently curious about."""
        return self.cognition.get_curiosity_questions()
    def developmental_summary(self) -> dict:
        """Return Genesis's developmental stage progress.

        Shows which Erikson-like psychosocial stages she has resolved
        and which she's currently working through.
        """
        return self._developmental_tracker.developmental_summary()
    def bug_report(self) -> str:
        """Return a description of bugs Genesis has noticed in her code."""
        return self.bug_reporter.describe_concerns()
    def recent_bugs(self, n: int = 10) -> list:
        """Return recent bug reports from the log."""
        return self.bug_reporter.recent_bugs(n)
    def scan_for_bugs(self) -> str:
        """Run a bug scan now and return the results."""
        result = self.bug_reporter.scan(max_files=50)
        return result.summary()
    def environment_status(self) -> str:
        """Return a description of Genesis's machine environment."""
        self.system_monitor.snapshot()
        return self.system_monitor.describe_environment()
    def environment_concerns(self) -> str:
        """Return environmental concerns Genesis has noticed."""
        if self.system_monitor.last_snapshot is None:
            self.system_monitor.snapshot()
        return self.system_monitor.describe_concerns()
    def system_snapshot(self) -> dict:
        """Return a raw system snapshot as a dict."""
        return self.system_monitor.snapshot().to_dict()
    def fetch_docs(self, query: str, language: str = "python") -> str:
        """Fetch documentation for a specific API or concept.

        Args:
            query: The API name or concept to look up.
            language: "python" or "rust".
        """
        result = self.learner.fetch_docs(query, language)
        emo = self.feel()
        from ..language import Thought as _Thought
        if result is None:
            thought = _Thought(
                content=f"couldn't fetch {language} docs for '{query}'",
                intent="self_report",
                emotion=emo.label,
                confidence=0.3,
                metadata={"docs_fetch_failed": True,
                          "language": language, "query": query},
            )
            return self.language.render(thought, emo)
        thought = _Thought(
            content=f"studied {language} docs for '{query}'",
            intent="inform",
            emotion=emo.label,
            confidence=0.7,
            metadata={"docs_studied": True,
                      "language": language,
                      "query": query,
                      "concepts_learned": len(result.concepts_learned),
                      "relationships_learned": len(result.relationships_learned)},
        )
        return self.language.render(thought, emo)
    def learn_code(self, max_files: int = 100) -> str:
        """Learn from her own source code.

        Reads Python and Rust source files, extracts functions,
        classes, structs, and relationships, and adds them to her
        concept network. This is how Genesis understands herself.

        Returns a human-readable summary of what she learned.
        """
        result = self.code_learner.learn_codebase(max_files=max_files)
        # Create bridges between the new code concepts and existing
        # dictionary/personal concepts — this is how she connects
        # what she learned from code to what she already knows.
        bridges = self.cognition.network._create_semantic_bridges()
        assoc = self.cognition.network._create_associative_bridges(max_new=500)
        hubs = self.cognition.network._attach_orphans_to_hubs(max_new=2000)
        # Refresh topology — the network changed
        self.cognition.topology = NetworkTopology(self.cognition.network)
        return (
            f"Analyzed {result.files_analyzed} source files. "
            f"Learned {result.concepts_added} code concepts and "
            f"{result.relationships_added} relationships. "
            f"Found {result.total_functions} functions and "
            f"{result.total_classes} classes/structs across "
            f"{result.total_lines} lines of code. "
            f"Formed {bridges + assoc + hubs} new connections."
        )
    def explore_files(self, path: str | None = None, max_files: int = 50) -> str:
        """Explore local files (docs, config, text) and learn from them.

        Args:
            path: Subdirectory to explore (relative to project root).
                  If None, explores the whole project.
            max_files: Maximum files to explore.

        Returns a human-readable summary.
        """
        if path:
            result = self.explorer.explore_path(path)
            self.cognition.topology = NetworkTopology(self.cognition.network)
            return result
        exploration = self.explorer.explore_directory(max_files=max_files)
        summary = self.explorer.get_exploration_summary()
        # Refresh topology — the network changed
        self.cognition.topology = NetworkTopology(self.cognition.network)
        return f"Explored {exploration.files_explored} files. {summary}"
    def code_summary(self) -> dict:
        """Return a summary of code self-knowledge."""
        return self.code_learner.get_code_summary()
    def _topology_status_summary(self) -> dict[str, Any]:
        """Return a compact summary of her knowledge network topology.

        Exposes graph-theoretic metrics (global clustering, average
        path length, small-world coefficient, community count, top
        hubs, top bridges) so the shape of her knowledge is visible
        through introspection. Computed lazily by the topology module
        and cached; refreshed after sleep consolidation.

        For large networks (>10k concepts), expensive metrics
        (clustering, betweenness, path length, communities) are skipped
        by the topology module, so those fields will be zero/empty.
        """
        try:
            topo = self.cognition.topology
            return {
                "global_clustering": topo.global_clustering(),
                "average_path_length": topo.average_path_length(),
                "is_small_world": topo.is_small_world(),
                "small_world_sigma": topo.small_world_sigma(),
                "community_count": len(topo.communities()),
                "hub_concepts": [
                    {"concept": name, "centrality": round(c, 4)}
                    for name, c in topo.hub_concepts(k=5)
                ],
                "bridge_concepts": [
                    {"concept": name, "betweenness": round(b, 4)}
                    for name, b in topo.bridge_concepts(k=5)
                ],
            }
        except Exception as e:  # noqa: BLE001
            logger.debug(f"_topology_status_summary failed: {e}")
            return {
                "global_clustering": 0.0,
                "average_path_length": 0.0,
                "is_small_world": False,
                "small_world_sigma": 0.0,
                "community_count": 0,
                "hub_concepts": [],
                "bridge_concepts": [],
            }
    def record_growth_snapshot(self) -> None:
        """Record a snapshot of current growth metrics to the ledger.

        Captures the current state of all tracked dimensions:
        knowledge (concepts, edges), self-improvement (proposals,
        experiments), dream synthesis (insights), and emotional
        development. Only records a milestone if the value has changed
        since the last snapshot.
        """
        # Knowledge dimension
        self.growth_ledger.record(
            "knowledge", "concept_count",
            self.cognition.network.total_concept_count,
            f"Concept network: {self.cognition.network.total_concept_count} concepts",
        )
        self.growth_ledger.record(
            "knowledge", "edge_count",
            self.cognition.network.edge_count,
            f"Relationships: {self.cognition.network.edge_count} edges",
        )
        # Quality metrics: non-monotonic, can go up or down.
        self.growth_ledger.record(
            "knowledge", "mean_edge_weight",
            round(self.cognition.network.mean_edge_weight, 4),
            f"Mean edge weight: {self.cognition.network.mean_edge_weight:.3f}",
        )
        self.growth_ledger.record(
            "knowledge", "mean_concept_confidence",
            round(self.cognition.network.mean_concept_confidence, 4),
            f"Mean concept confidence: {self.cognition.network.mean_concept_confidence:.3f}",
        )
        self.growth_ledger.record(
            "knowledge", "network_density",
            round(self.cognition.network.network_density, 4),
            f"Network density: {self.cognition.network.network_density:.4f}",
        )

        # Self-improvement dimension
        proposals = self.self_improvement.get_all_proposals()
        accepted = sum(1 for p in proposals if p.status.value == "accepted")
        self.growth_ledger.record(
            "self_improvement", "proposals_total",
            len(proposals),
            f"Total proposals: {len(proposals)}",
        )
        self.growth_ledger.record(
            "self_improvement", "proposals_accepted",
            accepted,
            f"Accepted proposals: {accepted}",
        )
        self.growth_ledger.record(
            "self_improvement", "experiments_applied",
            self.heuristic_experiment.applied_count,
            f"Applied experiments: {self.heuristic_experiment.applied_count}",
        )

        # Dream synthesis dimension
        dream_stats = self.inner_life.dream_synthesis_stats
        self.growth_ledger.record(
            "dreams", "insights_validated",
            dream_stats.get("validated", 0),
            f"Validated dream insights: {dream_stats.get('validated', 0)}",
        )
        self.growth_ledger.record(
            "dreams", "synthesis_proposed",
            dream_stats.get("proposed", 0),
            f"Dream synthesis proposals: {dream_stats.get('proposed', 0)}",
        )

        # Analogy dimension — cross-domain structure-mapping insights
        analogy_engine = self.cognition.analogy_engine
        self.growth_ledger.record(
            "analogy", "insights_validated",
            analogy_engine.insight_count,
            f"Analogy insights: {analogy_engine.insight_count}",
        )
        self.growth_ledger.record(
            "analogy", "pairs_found",
            analogy_engine.pairs_found,
            f"Analogous pairs found: {analogy_engine.pairs_found}",
        )

        # Inner life dimension
        self.growth_ledger.record(
            "inner_life", "thought_count",
            self.inner_life.thought_count,
            f"Spontaneous thoughts: {self.inner_life.thought_count}",
        )
        self.growth_ledger.record(
            "inner_life", "dream_count",
            self.inner_life.dream_count,
            f"Dream sequences: {self.inner_life.dream_count}",
        )
    def growth_narrative(self) -> str:
        """Return a first-person narrative of Genesis's growth."""
        self.record_growth_snapshot()
        return self.growth_ledger.generate_narrative()
    def growth_report(self) -> str:
        """Return a markdown-formatted growth report."""
        self.record_growth_snapshot()
        return self.growth_ledger.generate_markdown_report()
    def growth_summary(self) -> str:
        """Return a brief summary of the growth ledger."""
        return self.growth_ledger.summary()
    def growth_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of growth across all dimensions.

        Exposes the GrowthLedger's query APIs: milestone count,
        milestones per dimension, and latest values. This is for
        introspection — Genesis can query her own growth.
        """
        return {
            "milestone_count": self.growth_ledger.milestone_count,
            "milestones": [
                {
                    "dimension": m.dimension,
                    "metric": m.metric,
                    "value": m.value,
                    "note": m.note,
                }
                for m in self.growth_ledger.get_milestones()
            ],
        }
    def _growth_latest_values(self) -> dict[str, float]:
        """Return the latest recorded value for each (dimension, metric) pair.

        Uses the GrowthLedger's get_milestones API to find the most
        recent value per metric, exposing the get_value and get_latest
        query APIs for introspection.
        """
        latest: dict[str, float] = {}
        for m in self.growth_ledger.get_milestones():
            key = f"{m.dimension}/{m.metric}"
            # Milestones are in chronological order, so the last one
            # wins (most recent).
            latest[key] = m.value
        return latest
    def get_concept_summary(self) -> str:
        """Return a summary of Genesis's concept network."""
        return self.cognition.get_concept_summary()
    def get_reflection_summary(self) -> str:
        """Return a summary of Genesis's recent self-reflections."""
        return self.cognition.get_reflection_summary()
    def status(self) -> dict[str, Any]:
        """Return a status summary for display."""
        core_state = self.client.get_state()
        stats = self.client.get_memory_stats()
        summary = NeuroSummary(
            arousal=core_state.arousal,
            valence=core_state.valence,
            global_tone=core_state.global_tone,
            plasticity_gate=core_state.plasticity_gate,
            encoding_weight=core_state.encoding_weight,
            consolidation_weight=core_state.consolidation_weight,
            retrieval_weight=core_state.retrieval_weight,
            phase=self._effective_phase(core_state),
        )
        emotion = assess_emotion(summary, core_state.chemicals)

        status = {
            "running": self._running,
            "interactions": self._interaction_count,
            "phase": summary.phase_name,
            "emotion": emotion.label,
            "arousal": summary.arousal,
            "valence": summary.valence,
            "plasticity": summary.plasticity_gate,
            "stm_count": stats.stm_count,
            "ltm_count": stats.ltm_count,
            "language_engine": self.language.name,
            "allostatic_load": self.regulator.allostatic_load(),
            "hpa_active": self.regulator.hpa_axis.is_active,
            "primed_concepts": self.memory.priming.primed_count,
            "working_concepts": self.cognition.network.size,
            "archive_concepts": self.cognition.network.archive_size,
            "total_concepts": self.cognition.network.total_concept_count,
        }
        status.update(self._extended_status())
        return status
    def _extended_status(self) -> dict[str, Any]:
        """Extended status fields for introspection.

        These are the subsystem-exposing fields that go beyond the
        basic runtime/emotion/memory stats in :meth:`status`. Each
        field exposes a subsystem's introspection API so the shape of
        her mind is visible from the outside.
        """
        return {
            # Holographic graph — fixed-size associative memory stats.
            # Exposes the stats() API for introspection so the holographic
            # graph's state (edge count, storage, concepts/relations) is
            # visible alongside the explicit network stats.
            "holographic_graph": (
                self._sleep_compressor.holographic_graph.stats()
                if self._sleep_compressor is not None
                else {}
            ),
            # VQ codebook — vector quantization compression stats.
            # Exposes the stats() API for introspection so the codebook's
            # state (prototypes, compression ratio, reconstruction error)
            # is visible alongside the holographic graph stats.
            "vq_codebook": (
                self._sleep_compressor.vq_codebook.stats()
                if self._sleep_compressor is not None
                else {}
            ),
            # Growth ledger — how many milestones she's reached,
            # plus the latest values per dimension for introspection.
            "growth_milestones": self.growth_ledger.milestone_count,
            "growth_latest": self._growth_latest_values(),
            # Journal — how many entries she's written, plus a brief
            # description of her last entry (exposes the describe() API
            # for introspection).
            "journal_entries": self.journal.entry_count,
            "journal_status": self.journal.describe(),
            # User profile — what Genesis knows about the human. The
            # summarize() method returns a short, human-readable summary
            # of the user's name, preferences, and goals, exposing the
            # profile for introspection.
            "user_profile_summary": self.user_profile.summarize(),
            "user_interaction_count": self.user_profile.interaction_count,
            # Vision — whether the retina is attached and what she
            # last saw. Exposes last_faces_seen() and last_objects_seen()
            # for introspection so her visual state is visible alongside
            # her cognitive state.
            "vision_available": self.vision.is_available(),
            "vision_last_faces": self.vision.last_faces_seen(),
            "vision_last_objects": self.vision.last_objects_seen(),
            # Volition — her current urges and their strengths.
            "volition_state": self.volition.state(),
            # Developmental stages — Erikson-like psychosocial stage
            # progress. Exposes the developmental_summary() API for
            # introspection so her current stage, crisis, and
            # resolution scores are visible.
            "developmental_summary": self.developmental_summary(),
            # HPA axis — the full stress response state (CRH, ACTH,
            # cortisol, peak cortisol, time to peak). The status
            # already includes hpa_active (boolean); this exposes the
            # full hpa_axis_status() dict for detailed introspection.
            "hpa_axis_status": self.hpa_axis_status(),
            # Narrative — her self-story stats. Exposes the narrative
            # engine's event_count, chapter_count, temporal_link_count,
            # and life-script deviations for introspection.
            "narrative_events": self.cognition.narrative.event_count,
            "narrative_chapters": self.cognition.narrative.chapter_count,
            "narrative_temporal_links": (
                self.cognition.narrative.temporal_link_count
            ),
            "narrative_life_script_deviations": (
                self.cognition.narrative.deviation_from_script()
            ),
            # Subcognitive notifications — pending count from the
            # pull-based notification queue. Exposes peek() so the
            # queue depth is visible for introspection (e.g., checking
            # whether the cognitive layer is keeping up with the
            # subcognitive).
            "notifications_pending": self.notifications.peek(),
            # Projects — her creative output. Exposes counts so her
            # portfolio is visible for introspection (e.g., checking
            # how many projects she's built, how many are archived).
            "projects": self._projects_status_summary(),
            # Topology — structural self-awareness of her knowledge
            # network. Exposes graph-theoretic metrics (clustering,
            # small-world coefficient, communities, hubs, bridges)
            # so the shape of her knowledge is visible for
            # introspection. Computed lazily and cached by the
            # topology module; refreshed after sleep consolidation.
            "topology": self._topology_status_summary(),
        }

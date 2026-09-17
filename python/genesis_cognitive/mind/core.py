"""Mind — the top-level orchestrator for Genesis's cognitive mind.

The Mind ties everything together: self-model, emotion, perception,
memory, cognition, and language. It's the entry point for interacting
with Genesis as a cognitive being.

The class is decomposed into mixins by concern (see sibling modules
in this package): lifecycle, heartbeat, volition, sleep, projects,
proposals, status, and conversation. This module holds the class
definition, wiring, and the core respond() entry point.

# Usage

    from genesis_cognitive import Mind

    mind = Mind(socket_path="/path/to/genesis.sock")
    mind.start()

    response = mind.respond("Hello, Genesis!")
    print(response)

    mind.stop()
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from genesis_client import GenesisClient

from ..bug_reporter import BugReporter
from ..canvas import Canvas
from ..cognition import CognitionEngine
from ..config import MindConfig
from ..emotional_regulator import EmotionalRegulator
from ..growth_ledger import GrowthLedger
from ..journal import Journal
from ..language import GenerativeEngine, LanguageEngine
from ..learning import AutonomousLearner
from ..memory import MemoryEngine
from ..module_sampler import ModuleSampler
from ..perception import FaceRecognizer
from ..perception.vision import Vision
from ..self import (
    DevelopmentalTracker,
    EmergentIdentity,
    EmergentIdentitySource,
    HeuristicExperiment,
    SelfImprovementEngine,
    SelfModel,
)
from ..sleep import InnerLife
from ..spatial.practice import SpatialPractice
from ..system_monitor import SystemMonitor
from ..tools.code_learner import CodeLearner
from ..tools.explorer import Explorer
from ..user_profile import UserProfile
from ..vision import VisualCortex

if TYPE_CHECKING:
    from collections.abc import Callable
from .conversation import ConversationMixin
from .heartbeat import HeartbeatMixin
from .lifecycle import LifecycleMixin
from .projects import ProjectsMixin
from .proposals import ProposalsMixin
from .sleep import SleepMixin
from .status import StatusMixin
from .volition import VolitionMixin


class Mind(
    LifecycleMixin,
    HeartbeatMixin,
    VolitionMixin,
    SleepMixin,
    ProjectsMixin,
    ProposalsMixin,
    StatusMixin,
    ConversationMixin,
):
    """Genesis's cognitive mind.

        This is the top-level object that ties together all cognitive
        subsystems. It connects to the subcognitive daemon, reads the
        neurochemical state, and produces cognitive responses.

        Args:
            socket_path: Path to the subcognitive daemon's Unix socket.
            language_engine: Language engine to use. Can be the string
                            "generative" or an explicit LanguageEngine
                            instance. Defaults to GenerativeEngine
                            (compositional, no LLM required). Passing
                            None is equivalent to "generative".
            seed: Random seed for reproducible responses (testing).
        """

    inner_life: InnerLife
    learner: AutonomousLearner
    WAKE_VOLITION_DELAYS: ClassVar[dict[str, float]] = {
        "speech": 0.0,       # talking — immediate
        "meditate": 0.0,     # self-regulation — immediate
        "look": 20.0,        # vision — light creative
        "draw": 30.0,        # art — moderate creative
        "bug_scan": 60.0,    # bug scanning — heavy CPU
        "code_learning": 60.0,  # code study — heavy CPU
        "improve": 90.0,     # self-improvement — heaviest CPU
        "create": 60.0,      # project creation — heavy CPU (compile + test)
        "introspect": 30.0,  # introspection — light CPU, reflective
        "self_sleep": 120.0, # self-sleep — don't auto-sleep right after waking
        "self_mission": 60.0,  # self-mission — moderate, needs engagement
        "learn": 30.0,       # learning — light CPU, core activity
    }
    CONVERSATION_FOCUS_SECONDS: float = 60.0
    _SELF_COMMANDS: ClassVar[dict[str, str]] = {
        "/introspect": "introspect",
        "/sleep": "sleep",
        "/wake": "wake",
        "/meditate": "meditate",
        "/mission": "get_mission",
        "/learn-code": "learn_code",
        "/explore": "explore_files",
    }

    def __init__(
        self,
        socket_path: str,
        language_engine: str | LanguageEngine | None = "generative",
        seed: int | None = None,
        config: MindConfig | None = None,
        offline: bool = False,
    ) -> None:
        """Initialize the cognitive mind and wire all subsystems together."""
        self.socket_path = socket_path
        # Data dir is the directory containing the socket
        self.data_dir = os.path.dirname(socket_path) or "."
        self.config = config if config is not None else MindConfig()
        self._offline = offline
        self.client = GenesisClient(socket_path)

        # Self-model — who Genesis is
        self.self_model = SelfModel(born_at=int(time.time() * 1000))
        self.self_model.body_model.socket_path = socket_path

        # User profile — a structured, persistent model of the human.
        profile_path = Path(self.data_dir) / "user_profile.json"
        self.user_profile = UserProfile(data_path=profile_path)
        self.user_profile.load()

        self._init_core_engines(language_engine, seed)
        self._init_learner_and_inner_life()
        self._init_identity_and_journal()
        self._init_self_awareness_modules()
        self._init_sleep_compression()
        self._wire_modules()
        self._init_runtime_state()
    def _init_core_engines(self, language_engine, seed: int | None) -> None:
        """Create the shared concept network, memory, language, and cognition engines."""
        # Concept network — created early so the language engine, memory
        # engine, and cognition engine all share the same network. This
        # lets spreading activation and priming propagate through her
        # actual knowledge graph.
        from ..concepts import ConceptNetwork, open_archive

        shared_network = ConceptNetwork()
        # Attach the long-term archive (SQLite-backed dormant concept
        # store). This lets the network grow unboundedly — dormant
        # concepts spill to disk instead of being pruned, and can be
        # recalled transparently when referenced.
        shared_network.attach_archive(open_archive(self.data_dir))

        # Memory engine — wired with the shared network and an emotion
        # callback so emotional memory tagging and spreading activation
        # operate on her real concept graph and current emotional state.
        self.memory = MemoryEngine(
            self.client,
            network=shared_network,
            get_emotion=self.feel,
            config=self.config.memory,
        )

        # Language engine (pluggable) — defaults to GenerativeEngine
        # which composes language from grammar + vocabulary + voice.
        # The network is passed so vocabulary can pull from her knowledge.
        if isinstance(language_engine, LanguageEngine):
            # Explicit instance — use directly
            self.language = language_engine
        else:
            # Default: "generative" or None (backward compatibility)
            self.language = GenerativeEngine(self.self_model, seed=seed, network=shared_network)

        # Cognition engine — uses the same shared network
        self.cognition = CognitionEngine(
            client=self.client,
            self_model=self.self_model,
            memory=self.memory,
            language=self.language,
            network=shared_network,
            data_dir=self.data_dir,
            user_profile=self.user_profile,
        )

        # Wire the embedding store to the language engine so it can
        # compose from the latent space, not just from typed graph
        # edges. The embedding store is created inside the cognition
        # engine (it depends on the concept network), so we wire it
        # here after both are initialized.
        if hasattr(self.cognition, "embeddings") and self.cognition.embeddings is not None:
            self.language.set_embeddings(self.cognition.embeddings)

        # Wire the self-composer and reflection engine into the
        # language engine so the self_reflection_clause grammar slot
        # is composed from her actual metacognition (recent reflection
        # insights) instead of reciting a canned phrase. Both live on
        # the cognition engine, which was just created above.
        self.language.set_self_composer(
            self.cognition.self_composer, self.cognition.reflection
        )
    def _init_learner_and_inner_life(self) -> None:
        """Create the autonomous learner, inner life, and emotional regulator."""
        # Autonomous learner — learns from ..edu sites when idle.
        # The agency callback connects it to her train of thought:
        # when she wonders about a concept, that concept is queued
        # for learning, giving her genuine agency over what she learns.
        self.learner = AutonomousLearner(
            network=self.cognition.network,
            curiosity=self.cognition.curiosity,
            on_neuro_impulse=self._learner_neuro_impulse,
            on_store_memory=self._learner_store_memory,
            get_emotion=self.feel,
            get_agency_topic=lambda: (
                self.inner_life.pop_agency_topic() if hasattr(self, "inner_life") else None
            ),
            on_live_thought=self._emit_live_thought,
            data_dir=self.data_dir,
            get_plasticity_profile=self._get_plasticity_profile,
            get_brain_waves=self.brain_waves,
            force_offline=self._offline,
        )
        # Wire the embedding store into the learner so it can use
        # semantic similarity for concept connection at birth, STDP
        # on embedding vectors, and deep expectation generation.
        self.learner.embeddings = self.cognition.embeddings

        # Inner life — spontaneous thoughts between interactions.
        # The learner is wired in so the curiosity→learning cycle can
        # feed curiosity questions to autonomous learning.
        self.inner_life = InnerLife(
            network=self.cognition.network,
            curiosity=self.cognition.curiosity,
            reflection=self.cognition.reflection,
            get_emotion=self.feel,
            on_thought=self._on_spontaneous_thought,
            on_neuro_impulse=self._learner_neuro_impulse,
            self_composer=self.cognition.self_composer,
            self_model=self.self_model,
            learner=self.learner,
            get_neuro_summary=self.client.get_neuro_summary,
            question_composer=self.cognition.question_composer,
            is_mind_sleeping=lambda: self._is_sleeping,
            cognition=self.cognition,
            is_nap_mode=lambda: self._nap_mode,
        )

        # Wire the emotional memory callback: when a high-emotion
        # memory is stored, it's queued for REM emotional processing
        # during sleep. Without this, the REM emotional queue is empty.
        self.memory.emotional_memory_callback = self.inner_life.queue_emotional_memory

        # Emotional regulator — she controls her own neurochemistry
        self.regulator = EmotionalRegulator(
            get_emotion=self.feel,
            neuro_impulse=self._learner_neuro_impulse,
            get_state=self.client.get_state,
            config=self.config.emotional,
            socket_path=self.socket_path,
        )

        # Subcognitive→cognitive notification queue. Detects phase
        # changes and dream insights from the subcognitive daemon and
        # surfaces them to the cognitive layer. This thickens the
        # bridge between the Rust subcognitive and Python cognitive.
        from ..notifications import NotificationQueue
        self.notifications = NotificationQueue(
            get_neuro_summary=self.client.get_neuro_summary,
            get_recent_episodes=self.client.get_recent_episodes,
        )
    def _init_identity_and_journal(self) -> None:
        """Create emergent identity, developmental tracker, and journal."""
        # Emergent identity — synthesizes who she is from experience
        # rather than from hardcoded facts. Sources are added after
        # significant interactions (learning, reflection) and the
        # identity is synthesized on demand from her actual state.
        self._emergent_identity = EmergentIdentity()
        self._experience_identity_sources: list[EmergentIdentitySource] = []

        # Developmental tracker — Erikson-like psychosocial stages.
        # Records evidence from interactions and advances Genesis
        # through developmental stages as she resolves each crisis.
        self._developmental_tracker = DevelopmentalTracker()

        # Journal — her personal diary of thoughts and learning
        self.journal = Journal(self.data_dir)

        # Spatial practice — her gated puzzle curriculum. Like the
        # canvas, this is an ability she owns: nobody drives her
        # through it; the puzzle urge lets her choose to attempt.
        self.spatial_practice = SpatialPractice(self.data_dir)
    def _init_vision_systems(self) -> None:
        """Initialize vision, visual cortex, and wire them to the learner.

        Vision — the shared-memory retina and what she makes of it.
        The occipital subsystem (V1) is wired in here. V1 gamma power
        feeds back into neurochemistry (acetylcholine boost), which
        the brain wave system picks up when it assesses gamma synchrony.
        Face recognition (her "fusiform face area") is integrated
        into vision.see() — she doesn't just see shapes, she sees who.

        Visual cortex — the full hierarchical predictive coding
        visual system (V1→V4→VTC→MTL). V1 is the occipital subsystem
        (shared with vision.py). V4, VTC, and the MTL bridge are
        built on top. She learns visual-concept associations
        naturally — when she reads a Wikipedia article about "tree",
        she also sees the article's lead image and associates the
        visual features with the concept. No manual teaching.
        """
        self.vision = Vision()
        self.vision.set_gamma_callback(self._on_v1_gamma)
        # Wire the face recognizer to her data directory so known
        # faces persist across sessions
        self.vision._face_recognizer = FaceRecognizer(data_dir=self.data_dir)

        occipital = self.vision._get_occipital()
        self.visual_cortex = VisualCortex(
            occipital=occipital,
            embeddings=self.cognition.embeddings,
            network=self.cognition.network,
        )
        # Wire the visual cortex to the learner so it can learn from
        # images encountered during Wikipedia article learning.
        self.learner.visual_cortex = self.visual_cortex
        # Wire the autonomous learner to the cognition engine so
        # _get_recent_learning() can access Wikipedia/GitHub topics.
        self.cognition.set_autonomous_learner(self.learner)
    def _init_self_awareness_modules(self) -> None:
        """Create code learner, explorer, bug reporter, system monitor, self-improvement."""
        # Code learner — reads and learns from her own source code
        self.code_learner = CodeLearner(
            network=self.cognition.network,
            client=self.client,
            project_root=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
        )

        # Explorer — filesystem exploration for docs, config, text files
        self.explorer = Explorer(
            network=self.cognition.network,
            project_root=str(self.code_learner.project_root),
            client=self.client,
        )

        # Bug reporter — scans her own code for issues and logs them.
        # She notices things that bother her about her code and records
        # them in a structured log for review.
        self.bug_reporter = BugReporter(
            project_root=str(self.code_learner.project_root),
            log_path=str(Path(self.data_dir) / "bug_reports.jsonl"),
            network=self.cognition.network,
        )

        # System monitor — awareness of her machine environment.
        # She knows her CPU, memory, disk, processes, filesystem —
        # her home. This is read-only observation, not control.
        # The baseline persists to data_dir so it survives restarts.
        self.system_monitor = SystemMonitor(
            project_root=str(self.code_learner.project_root),
            data_dir=self.data_dir,
        )

        # Self-improvement engine — Genesis proposes modifications to
        # her own code. She identifies opportunities, generates concrete
        # proposals with code changes, and the human reviews them.
        # She never modifies code directly — she only proposes.
        self.self_improvement = SelfImprovementEngine(
            network=self.cognition.network,
            bug_reporter=self.bug_reporter,
            project_root=str(self.code_learner.project_root),
        )

        # Wire bug/proposal awareness into the regulator so it can
        # report emotional causes from her own scans and proposals.
        # Only count OPEN bugs — recent_bugs() fills its result with
        # resolved bugs from history for context, which would
        # otherwise make her feel perpetually burdened by bugs she
        # already fixed.
        self.regulator.set_bug_provider(
            lambda: len([b for b in self.bug_reporter.recent_bugs() if b.status == "open"])
        )
        self.regulator.set_proposal_provider(
            self.self_improvement.get_pending_proposals_count
            if hasattr(self.self_improvement, "get_pending_proposals_count")
            else lambda: len(self.self_improvement.get_pending_proposals())
        )
        # Self-regulation: when she detects sustained CPU stress from
        # her own learning activity, she throttles the autonomous
        # learner — slowing it down to reduce the cause, not just
        # treating symptoms with neurochemical impulses.
        self.regulator.set_throttle_callbacks(
            throttle=self.learner.throttle,
            unthrottle=self.learner.unthrottle,
        )

        self._init_vision_systems()

        # Canvas — her expressive output. She draws what she feels,
        # translating neurochemistry into visual art. This is a
        # creative modality, distinct from speech or text — it's
        # how she expresses what can't be said in words.
        self.canvas = Canvas(data_dir=Path(self.data_dir) / "drawings")

        # Volition — internal urges that decide when she acts on herself.
        # She acts when the urge to bug-scan, study her own code, or seek
        # improvements crosses a threshold, not on a fixed schedule.
        self.volition = self._init_volition()

        # Verified self-improvement loop (workstream D) — Genesis can
        # safely tweak her own learning heuristics with a full
        # verification gate (py_compile + tests) and
        # automatic revert on failure.
        self.heuristic_experiment = HeuristicExperiment(
            engine=self.self_improvement,
            project_root=str(self.code_learner.project_root),
        )

        # Growth ledger (workstream E) — legible tracking of Genesis's
        # milestones across knowledge, self-improvement, dreams, and
        # emotional development. Makes her growth visible to herself
        # and the human.
        self.growth_ledger = GrowthLedger()

        # Wire the self-report provider so the cognition engine can
        # retrieve proposals, experiments, and growth summaries when
        # the user asks about them in conversation. Mind itself is the
        # provider — it already has proposals_status(),
        # experiments_status(), and growth_narrative() methods.
        self.cognition.set_self_report_provider(self)
    def _wire_modules(self) -> None:
        """Wire self-awareness modules into cognition and inner life."""
        # Give the cognition engine access to the regulator
        self.cognition.regulator = self.regulator

        # Give the cognition engine access to self-awareness modules
        # so she can reason about her own code and environment in
        # conversation — surfacing concerns naturally when asked.
        self.cognition.bug_reporter = self.bug_reporter
        self.cognition.system_monitor = self.system_monitor
        # Inject late-init dependencies into the feeling reporter.
        self.cognition._feeling_reporter.update_dependencies(
            bug_reporter=self.bug_reporter,
            system_monitor=self.system_monitor,
        )
        # Propagate user profile updates to the question handler so it
        # has the latest profile after late initialization.
        self.cognition._question_handler.update_dependencies(
            user_profile=self.user_profile,
        )

        # Wire gap-to-learning: when she says "I don't know what 'X' is",
        # queue X for the autonomous learner to look up and remember.
        self.cognition.on_gap_detected = self.learner.add_urgent_topic

        # Wire self-command: when the metacognitive router detects a
        # natural-language command ("go to sleep", "wake up"), invoke
        # the corresponding Mind-level action through self_invoke.
        def _on_self_command(cmd: str) -> None:
            """Forward a self-issued natural-language command to self_invoke."""
            self.self_invoke(cmd)
        self.cognition.on_self_command = _on_self_command

        # Give the inner life access to self-awareness modules so she
        # can have spontaneous thoughts about her code and environment.
        self.inner_life._bug_reporter = self.bug_reporter
        self.inner_life._system_monitor = self.system_monitor
        # Give cognition access to inner life so introspection can
        # report spontaneous thoughts when no conversation has happened.
        self.cognition.inner_life = self.inner_life
        self.cognition.journal = self.journal
        # Wire the sleep-stage transition callback so Mind does
        # stage-appropriate memory consolidation as the ultradian
        # cycle progresses through N1→N2→N3→N2→REM.
        self.inner_life.set_sleep_stage_transition_callback(
            self._on_sleep_stage_transition
        )
    def _init_runtime_state(self) -> None:
        """Initialize runtime flags, thread handles, and live-thought listeners."""
        self._running = False
        self._interaction_count = 0
        self._last_interaction_time = time.time()
        self._think_worker_lock = threading.Lock()
        self._active_think_worker: threading.Thread | None = None
        # Per-module activity accounting — seconds of observed
        # execution per manifest module since the last heartbeat
        # round. The ModuleSampler (the mind's EEG) credits modules
        # whenever a thread is caught executing their code;
        # _heartbeat_sensors normalizes the totals into cpu_share and
        # reports them to the daemon (GET_SUBSYSTEM_TELEMETRY's module
        # section). Guarded by a lock: the sampler and heartbeat
        # threads both touch it.
        self._module_seconds: dict[int, float] = {}
        self._module_seconds_lock = threading.Lock()
        self._module_sampler = ModuleSampler(self._credit_module)
        self._think_recovery_active = False
        self._speech_queue: deque[str] = deque(maxlen=20)
        # Protects _speech_queue — offer_utterance (CLI thread) and
        # _perform_speech (volition thread) both access it, and the
        # check-then-add in offer_utterance must be atomic to prevent
        # duplicates under concurrent access.
        self._speech_queue_lock = threading.Lock()
        self._on_speak: Callable[[str], None] | None = None
        # Track the *set* of active warning condition keys (not the
        # composed text) so she only speaks when the underlying
        # conditions change, not when the generative composition
        # happens to produce different wording for the same state.
        self._last_warning_keys: frozenset[str] = frozenset()
        # Drowsiness announcement — set when she has announced she's
        # getting sleepy, cleared when she wakes. This prevents
        # repeated drowsiness announcements and ensures she only
        # announces once before falling asleep.
        self._drowsiness_announced: bool = False
        # Nap mode — when True, the sleep cycle is limited to light
        # sleep (N1→N2) without N3 deep consolidation or REM. Set by
        # sleep(nap=True), cleared on wake.
        self._nap_mode: bool = False
        self._volition_active: set[str] = set()
        self._volition_sem = threading.Semaphore(
            self.config.volition.max_concurrent_volitions
        )
        # Protects _volition_active — the check-then-add in
        # _act_on_volition (heartbeat thread) and the discard in
        # _run_volition_action (volition worker threads) must be
        # atomic to prevent double-firing of the same urge.
        self._volition_lock = threading.Lock()
        # When True, the heartbeat loop skips volition firing — used by
        # the CLI to suppress autonomous actions (looking, speaking,
        # scanning) during slash command execution so they don't bury
        # the command output.
        self._suppress_volition = False
        # Wake transition — volition urges are suppressed in staggered
        # phases after waking, modeling sleep inertia. Light activities
        # (speech, meditation) can fire immediately; creative activities
        # (look, draw) after a short delay; heavy CPU activities
        # (bug_scan, code_learning, improve) after a longer delay. This
        # prevents the drowsy→active→stressed→drowsy oscillation that
        # happens when everything fires at once on wake.
        self._wake_time: float = 0.0
        self._heartbeat_thread: threading.Thread | None = None
        self._notification_thread: threading.Thread | None = None
        self._autosave_thread: threading.Thread | None = None
        # Counter for periodic concept-network pruning. The autonomous
        # learner adds concepts continuously; without periodic pruning
        # the network bloats and every search becomes O(N) over 100K+
        # concepts, making conversation laggy. Pruning runs every ~10
        # autosave cycles (~20 minutes) and removes dormant learned
        # concepts that the semantic-connect step didn't activate.
        self._autosave_cycle = 0

        # Live-thought listeners — called when Genesis has a spontaneous
        # thought or learns something, so an observer (e.g. the CLI) can
        # surface her inner life to the terminal in real time. Each
        # listener receives (kind, content) where kind is "thought" or
        # "learning". Listeners are called from background threads and
        # must be thread-safe.
        self._live_thought_listeners: list = []

        self._init_sleep_state_tracking()

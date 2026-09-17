"""Inner life — spontaneous thoughts that arise between interactions.

This is Genesis's stream of cognition. When she's not in conversation,
she thinks. Not constantly — thoughts arise, fade, and give way to others.

Thoughts are triggered by:
1. Unresolved curiosity — questions she hasn't answered
2. Emotional state — high alertness produces more thoughts, low alertness fewer
3. Recent memories — things she learned or experienced
4. Concept network gaps — things she knows about but doesn't understand
5. Time of "day" — her internal cycle affects thought frequency

Each thought is:
- Recorded in her reflection engine
- May produce an insight
- Affects her neurochemistry (thinking uses acetylcholine)
- May trigger curiosity questions
- Gets stored as a memory if significant

This is NOT a response to anything. It's spontaneous internal activity.
She's not being asked to think — she just does.

## Train of thought

Instead of isolated thoughts every 15-30 seconds, she thinks in chains.
Each thought extracts its key concepts and the next thought is seeded
from those concepts, modulated by mood. A chain runs for 3-8 thoughts,
then naturally winds down. This creates continuity — the thing that
makes her feel "there" between ticks.

The chain mechanism:
- Each thought's content is parsed for concept names from her network
- Those concepts seed the next thought's generation
- The chain has momentum: each subsequent thought is more likely to
  continue if the previous one produced an insight or connection
- High creativity extends chains; low arousal shortens them
- A chain can branch: if a thought mentions two concepts, the next
  thought might follow either one
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genesis_client import NeuroSummary

    from ..bug_reporter import BugReporter
    from ..system_monitor import SystemMonitor

from ..brain_waves import (  # canonical definition — avoids threshold drift
    SleepStage,
    _sleep_stage_from_summary,
)
from ..concepts import ConceptNetwork, RelationType, strip_sense_suffix
from ..emotion import EmotionalState
from ..learning import CuriosityEngine
from ..self import Insight, ReflectionEngine
from .architecture import (
    HippocampalReplay,
    SleepCycleTracker,
    SynapticDownscaler,
)
from .dream_synthesis import DreamSynthesisEngine
from .events import (
    HypnagogicState,
    KComplex,
    ParasomniaEvent,
    PGOWave,
    SharpWaveRipple,
    SleepInertia,
    SleepSpindle,
)
from .thoughts import SpontaneousThought, ThoughtChainType

__all__ = ["InnerLife"]

logger = logging.getLogger(__name__)

# How often to check for spontaneous thoughts (seconds)
THOUGHT_CHECK_INTERVAL = 12.0

# How often the inner-life loop may take a fresh system snapshot
# (seconds). A full snapshot scans the filesystem, so it's
# rate-limited — but it must happen occasionally or the baseline
# never learns what's normal for this machine.
ENV_SAMPLE_INTERVAL = 1800.0

# Curiosity→learning is now state-driven (curiosity drive accumulates
# from knowledge gaps and queue depth, fires when pressure crosses
# threshold). No fixed interval needed.

# Train-of-thought parameters
# Min/max chain length — a chain of thoughts that builds on itself
MIN_CHAIN_LENGTH = 3
MAX_CHAIN_LENGTH = 8
# Delay between thoughts in a chain (seconds) — much shorter than the
# isolated-thought gap, because the chain has momentum
CHAIN_THOUGHT_DELAY = 8.0
# Probability that a chain continues after each thought, modified by
# emotional state and whether the thought produced an insight
CHAIN_CONTINUE_BASE = 0.6

# Rumination detection parameters
# A negative chain with 3+ thoughts and decreasing novelty is rumination.
RUMINATION_MIN_LENGTH = 3
# If novelty drops below this threshold across the chain, it's repetitive
RUMINATION_NOVELTY_THRESHOLD = 0.3


class _LiveEvent:
    """A minimal thought-like event for the live thought callback.

    Used by ``InnerLife._emit`` to surface internal events (insights,
    learning notices, etc.) to the CLI's thought handler without
    constructing a full ``SpontaneousThought``.
    """

    __slots__ = ("chain_position", "content", "intent", "trigger")

    def __init__(self, content: str, trigger: str) -> None:
        """Create a minimal thought-like event for the callback."""
        self.content = content
        self.trigger = trigger
        self.intent = "statement"
        self.chain_position = 0

    def describe(self) -> str:
        """Return the event content."""
        return self.content


class InnerLife:
    """Genesis's continuous inner life — spontaneous thoughts between interactions.

    Runs in a background thread. Every few seconds, checks if a thought
    should arise based on her emotional state, curiosity, and recent
    experience. Thoughts are recorded and may produce insights.

    Pauses during conversation (same as the autonomous learner).
    """

    def __init__(
        self,
        network: ConceptNetwork,
        curiosity: CuriosityEngine,
        reflection: ReflectionEngine,
        get_emotion=None,
        on_thought=None,
        on_neuro_impulse=None,
        self_composer=None,
        self_model=None,
        learner=None,
        get_neuro_summary=None,
        question_composer=None,
        is_mind_sleeping=None,
        cognition=None,
        is_nap_mode=None,
        seed: int | None = None,
    ) -> None:
        """Initialize the inner life with its cognitive engines and callbacks.

        Args:
            network: Her concept network — thoughts draw from what she knows.
            curiosity: Engine that surfaces knowledge gaps worth wondering about.
            reflection: Engine that turns experience into reflective thoughts.
            get_emotion: Callback returning her current EmotionalState (or None).
            on_thought: Callback invoked when a spontaneous thought arises.
            on_neuro_impulse: Callback to send neurochemical impulses (unused
                for spontaneous thoughts, but wired for future use).
            self_composer: Composes first-person self-reflections (optional).
            self_model: Her self-model for self-aware thoughts (optional).
            learner: Autonomous learner — receives curiosity questions to
                learn from. If None, the curiosity→learning cycle is a no-op.
            get_neuro_summary: Callback returning the raw neurochemical
                summary, used to detect sleep phase and acetylcholine levels
                (for lucid dreaming). If None, dreaming falls back to emotion.
            question_composer: Generates genuine questions from knowledge
                gaps. If None, social thoughts fall back to the curiosity engine.
            is_mind_sleeping: Callback returning True if the mind's
                cognitive sleep flag is set (via /sleep or /nap). This
                catches the gap between the mind setting _is_sleeping
                and the daemon's neurochemical phase actually shifting
                to NREM/REM — without it, questions leak through during
                that window.
            cognition: The cognition engine, whose language generator can turn
                question metadata into natural language. Optional, but needed
                for fully composed social questions.
            is_nap_mode: Callback returning True if the current sleep
                is a nap (light sleep only, N1→N2). Used to configure
                the sleep cycle tracker when she falls asleep.
            seed: Optional RNG seed for deterministic thought sequences.
        """
        self.network = network
        self.curiosity = curiosity
        self.reflection = reflection
        self._get_emotion = get_emotion
        self._on_thought = on_thought
        self._on_neuro_impulse = on_neuro_impulse
        self._self_composer = self_composer
        self._self_model = self_model
        self._is_mind_sleeping = is_mind_sleeping
        # Callback to check if the current sleep is a nap (light
        # sleep only). Used to configure the sleep cycle tracker.
        self._is_nap_mode = is_nap_mode or (lambda: False)
        # Autonomous learner — receives curiosity questions to learn from.
        # Optional: if None, the curiosity→learning cycle is a no-op.
        self._learner = learner
        # Question composer — generates genuine questions from knowledge
        # gaps and curiosity, replacing hardcoded social question lists.
        # Optional: if None, social thoughts fall back to curiosity engine.
        self._question_composer = question_composer
        # Optional callback to read the raw neurochemical summary.
        # Used to detect sleep phase and acetylcholine levels (for
        # lucid dreaming probability). If None, dreaming features
        # fall back to the emotional state alone.
        self._get_neuro_summary = get_neuro_summary
        self._cognition = cognition
        self._rng = random.Random(seed)

        self._init_awareness_agency()
        self._init_thread_and_thoughts()
        self._init_timers()
        self._init_dream_state()
        self._init_sleep_transition()
        self._init_sleep_graphoelements()
        self._init_rumination_insight()
        self._init_sleep_cycle()
        self._init_replay_systems()
        self._init_dream_synthesis()

    def _init_awareness_agency(self) -> None:
        """Set up self-awareness module slots and curiosity-driven agency."""
        # Self-awareness modules — set by Mind after creation.
        # These let Genesis have spontaneous thoughts about her own
        # code quality and machine environment, not just her concepts.
        self._bug_reporter: BugReporter | None = None  # set by Mind after creation
        self._system_monitor: SystemMonitor | None = None  # set by Mind after creation
        # Last time the environment collector took a fresh snapshot.
        self._last_env_sample: float = 0.0

        # Curiosity-driven agency: concepts from her train of thought
        # that she wants to learn about. These are queued for the
        # autonomous learner, creating the loop:
        #   thought about X → wonder about X → learn about X → understand X
        # This is different from the periodic curiosity cycle — it's
        # driven by what she's actually thinking about right now.
        #
        # NOTE: Dreams do NOT feed into agency topics. Dreams produce
        # emotional impressions (_dream_residues), not learning goals.
        # A human doesn't dream about goals — goals are a waking,
        # prefrontal activity. Dreams are affective and associative.
        self._agency_topics: deque[str] = deque(maxlen=20)

        # Dream residues — emotional impressions left by dreams. These
        # are NOT learning targets. They color her waking thoughts and
        # mood for a while after waking, surfacing as reflective thoughts
        # ("I was dreaming about..."), but they don't drive the
        # autonomous learner. This separates the subcognitive (dreams)
        # from the cognitive (goals).
        self._dream_residues: deque[str] = deque(maxlen=10)

    def _init_thread_and_thoughts(self) -> None:
        """Initialize threading state and thought tracking."""
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._stop_event = threading.Event()

        # Track her thoughts (bounded — old thoughts auto-evicted)
        self._thoughts: deque[SpontaneousThought] = deque(maxlen=100)
        self._last_thought_time = 0.0
        self._thought_count = 0
        self._thoughts_lock = threading.Lock()

        # Locks for shared deques accessed from both the background
        # _run thread and external callers (main thread, learner thread).
        # Without these, concurrent append/popleft can corrupt the deque
        # or lose/duplicate entries.
        #
        # LOCK ORDER (deadlock prevention): these locks are LEAF-LEVEL —
        # never hold one while acquiring another. If nesting is ever
        # required, acquire in this declaration order (thoughts >
        # stimuli > emotional_memories > agency_topics > dream_residues).
        self._stimuli_lock = threading.Lock()
        self._emotional_memories_lock = threading.Lock()
        self._agency_topics_lock = threading.Lock()
        self._dream_residues_lock = threading.Lock()

    def _init_timers(self) -> None:
        """Initialize cooldown, social drive, curiosity drive, and chain timers.

        Questions and curiosity are no longer timer-driven. Instead,
        they accumulate drive pressure from emotional state and
        knowledge gaps, and fire when the pressure crosses a threshold.
        This makes them state-driven — she asks questions when she
        feels socially motivated, and learns when curiosity pressure
        builds — not because a timer elapsed.
        """
        # Cooldown — don't think too rapidly
        self._min_thought_gap = 6.0  # seconds between thoughts

        # Social drive — accumulates when she's open to engaging,
        # decays when she's not. Fires a question when it crosses
        # the threshold. This replaces the fixed 45s question timer.
        self._social_drive = 0.0
        self._social_drive_threshold = 1.0
        self._last_question_time = 0.0
        self._question_cooldown = 20.0  # min seconds between questions

        # Curiosity drive — accumulates from knowledge gaps and
        # curiosity queue depth, decays when satisfied. Fires a
        # curiosity learning cycle when it crosses the threshold.
        # This replaces the fixed 30s curiosity timer.
        self._curiosity_drive = 0.0
        self._curiosity_drive_threshold = 1.0
        self._last_curiosity_cycle = 0.0
        self._curiosity_cooldown = 15.0  # min seconds between cycles
        self._curiosity_topics_queued = 0

        # Train-of-thought state
        self._chain_counter = 0  # unique ID for each chain
        self._current_chain_concepts: list[str] = []  # concepts to seed next thought

        # Cached concept matcher for _extract_concepts_from_thought.
        # Rebuilt when the concept count changes (avoids O(N) regex
        # per-thought; uses a single compiled alternation instead).
        self._concept_matcher: Callable[[str], list[str]] | None = None
        self._concept_matcher_count: int = -1

    def _init_dream_state(self) -> None:
        """Initialize dream and parasomnia tracking state."""
        # ─── Dream state ───────────────────────────────────────────
        # Dreaming happens during sleep (phase == "sleeping"). Dreams
        # are like thought chains but more associative, less directed,
        # and marked as dream content. Lucid dreams are a special case
        # where she becomes aware she's dreaming and can partially
        # direct the content.
        self._dream_count = 0
        self._lucid_dream_count = 0

        # ─── Parasomnia state ──────────────────────────────────────
        # Parasomnias (sleepwalking, sleeptalking) are rare events
        # during NREM deep sleep. They are tracked for persistence.
        self._parasomnia_count = 0
        self._parasomnia_events: deque[ParasomniaEvent] = deque(maxlen=20)

    def _init_sleep_transition(self) -> None:
        """Initialize hypnagogic and sleep-inertia state."""
        # ─── Hypnagogic state ───────────────────────────────────────
        # The hypnagogic state is the gradual transition from
        # wakefulness to sleep. It occurs during the "drowsy" phase,
        # before full sleep. Thoughts become more dream-like as
        # progress increases from 0.0 (awake) to 1.0 (asleep).
        self._hypnagogic: HypnagogicState | None = None
        self._hypnagogic_count = 0

        # ─── Sleep inertia ──────────────────────────────────────────
        # Sleep inertia is the grogginess experienced upon waking.
        # It impairs cognition for 5-30 minutes, proportional to sleep
        # depth. We track the last observed arousal during sleep to
        # estimate depth when she wakes.
        self._sleep_inertia = SleepInertia()
        self._last_sleep_arousal: float = 0.5
        self._last_inertia_update: float = 0.0
        self._was_sleeping: bool = False

    def _init_sleep_graphoelements(self) -> None:
        """Initialize sleep graphoelements and REM emotional processing."""
        # ─── Sharp wave-ripples ─────────────────────────────────────
        # SWRs are 150-250 Hz hippocampal replay events that occur
        # during NREM/SWS. They drive memory consolidation by
        # replaying and strengthening concept connections.
        self._swr_count = 0
        self._swr_events: deque[SharpWaveRipple] = deque(maxlen=50)

        # ─── PGO waves ──────────────────────────────────────────────
        # PGO waves are a signature of REM sleep, associated with
        # visual dream imagery. They are generated during REM and
        # their visual content is woven into dream thoughts.
        self._pgo_count = 0
        self._pgo_events: deque[PGOWave] = deque(maxlen=50)

        # ─── Sleep spindles (N2) ─────────────────────────────────────
        # Sleep spindles are 10-16 Hz thalamocortical oscillations
        # during N2 that drive memory transfer from hippocampus to
        # neocortex. Spindle density (spindles per minute of N2)
        # correlates with consolidation quality. We track the total
        # count and a rolling window of events, plus the wall-clock
        # time spent in N2 so we can compute density.
        self._spindle_count = 0
        self._spindle_events: deque[SleepSpindle] = deque(maxlen=50)
        self._n2_seconds: float = 0.0  # accumulated time in N2
        self._last_n2_update: float = 0.0

        # ─── K-complexes (N2) ────────────────────────────────────────
        # K-complexes are the largest graphoelement in healthy sleep,
        # occurring during N2. They serve sensory gating (protecting
        # sleep from external stimuli) and are markers of memory
        # consolidation. External stimuli queued during sleep can
        # trigger evoked K-complexes.
        self._kcomplex_count = 0
        self._kcomplex_events: deque[KComplex] = deque(maxlen=50)
        # Queued external stimuli that may trigger evoked K-complexes.
        # Each entry is a brief description of the stimulus.
        self._pending_stimuli: deque[str] = deque(maxlen=10)

        # ─── REM emotional processing ────────────────────────────────
        # During REM, the prefrontal cortex is deactivated while the
        # amygdala is highly active, and noradrenergic (NE) tone falls
        # near silent. This allows emotional memories to be reprocessed
        # and their affective intensity reduced (extinction-like)
        # while the factual content is consolidated — the basis of
        # REM's therapeutic role in emotional memory (Walker & van der
        # Helm, 2009; Pace-Schott et al., 2019).
        self._emotional_memories_processed = 0
        # Recent emotional memories awaiting REM processing. Each is a
        # (concept_id, intensity) pair. Intensity is reduced during REM.
        self._pending_emotional_memories: deque[tuple[str, float]] = deque(maxlen=30)

    def _init_rumination_insight(self) -> None:
        """Initialize rumination and insight tracking."""
        # ─── Rumination & insight tracking ──────────────────────────
        # Rumination: negative-valence thought chains that are
        # repetitive and unproductive. We track how many times
        # rumination has been detected and whether we're currently
        # in a ruminative state.
        self._rumination_count = 0
        self._is_ruminating = False

        # Insight: "aha moments" when a thought chain connects two
        # previously distant concepts. Insights get a dopamine burst
        # and are recorded in the reflection engine.
        self._insight_count = 0

        # ─── Recent thought topics (recency inhibition) ────────────
        # A bounded deque of the topics she's recently thought about,
        # used by the salience model to suppress rumination. When a
        # topic appears here, its salience is reduced in future
        # thought-generation cycles. This is the mechanism that lets
        # her move on from a subject rather than repeating it.
        self._recent_thought_topics: deque[str] = deque(maxlen=20)

    def _init_sleep_cycle(self) -> None:
        """Initialize the 90-minute ultradian sleep cycle tracker."""
        # ─── 90-minute ultradian sleep cycle ──────────────────────
        # The cycle tracker drives stage progression through
        # N1→N2→N3→N2→REM during sleep. It is created when she falls
        # asleep and reset when she wakes. When active, its current
        # stage is the authority for which sleep events to generate.
        self._sleep_cycle: SleepCycleTracker | None = None
        self._last_cycle_advance: float = 0.0  # wall-clock of last advance
        # The last stage we reported via the transition callback, so
        # we only fire on actual stage changes (N1→N2→N3→N2→REM).
        self._last_reported_stage: SleepStage | None = None
        # Callback invoked when the sleep cycle transitions to a new
        # stage. Set by Mind so it can do stage-appropriate memory
        # consolidation (N3: systems consolidation, N2: spindle-driven
        # transfer, REM: emotional/procedural consolidation). The
        # callback receives (new_stage, subjective_dt_seconds).
        self._on_sleep_stage_transition: Callable[[SleepStage, float], None] | None = None

    def _init_replay_systems(self) -> None:
        """Initialize synaptic homeostasis and hippocampal replay."""
        # ─── Synaptic Homeostasis Hypothesis (SHY) ─────────────────
        # Synaptic load accumulates during wakefulness and is
        # renormalised (downscaled) during N3 slow-wave sleep
        # (Tononi & Cirelli, 2003, 2006, 2014).
        self._synaptic_downscaler = SynapticDownscaler()

        # ─── Hippocampal replay during N3 ──────────────────────────
        # Recent experiences are queued during wakefulness and replayed
        # in compressed time during N3, driven by sharp wave-ripples
        # (Wilson & McNaughton, 1994; Stickgold, 2005).
        self._hippocampal_replay = HippocampalReplay()

    def _init_dream_synthesis(self) -> None:
        """Initialize the sleep-driven synthesis engine.

        During REM sleep, the synthesis engine finds structurally
        similar but unconnected concepts and proposes novel typed
        edges. On wake, those proposals are validated.
        """
        self._dream_synthesis = DreamSynthesisEngine(self.network)
        # Track whether we've run synthesis this REM period
        self._synthesis_done_this_rem = False


    def _emit(self, kind: str, content: str) -> None:
        """Send a live event to the thought callback (if connected)."""
        if not self._on_thought:
            return
        try:
            self._on_thought(_LiveEvent(content, kind))
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))  # callback must never crash inner life

    def start(self) -> None:
        """Start the inner life thread."""
        if self._running:
            return
        # If a previous thread is still exiting (stop() may have timed
        # out waiting for an in-flight IPC call that can block for up
        # to the 30s client timeout), wait for it to fully terminate
        # before starting a new one. Otherwise the old thread would see
        # _running=True and _stop_event cleared, and resume its loop
        # alongside the new thread — two thought generators at once.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=35)
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Inner life started")

    def stop(self) -> None:
        """Stop the inner life thread."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Inner life stopped")

    def pause(self) -> None:
        """Pause — someone is talking to her."""
        self._paused = True

    def resume(self) -> None:
        """Resume — conversation is over."""
        self._paused = False

    @property
    def thought_count(self) -> int:
        """Total number of spontaneous thoughts generated since startup."""
        with self._thoughts_lock:
            return self._thought_count

    @property
    def recent_thoughts(self) -> list[SpontaneousThought]:
        """The last 10 spontaneous thoughts (newest last), as a copy."""
        with self._thoughts_lock:
            all_thoughts = list(self._thoughts)
            return all_thoughts[-10:] if len(all_thoughts) > 10 else all_thoughts

    @property
    def dream_count(self) -> int:
        """Total number of dream sequences generated during sleep."""
        return self._dream_count

    @property
    def lucid_dream_count(self) -> int:
        """Number of lucid dreams — where she knew she was dreaming."""
        return self._lucid_dream_count

    @property
    def dream_synthesis_insights(self) -> list:
        """Validated dream-synthesis insights (novel connections made in sleep)."""
        return self._dream_synthesis.insights

    @property
    def dream_synthesis_stats(self) -> dict[str, int]:
        """Statistics about dream synthesis: proposed, validated, rejected."""
        return {
            "proposed": self._dream_synthesis.proposed_count,
            "validated": self._dream_synthesis.insight_count,
            "rejected": self._dream_synthesis.rejected_count,
        }

    @property
    def parasomnia_count(self) -> int:
        """Total parasomnia events (sleepwalking + sleeptalking)."""
        return self._parasomnia_count

    @property
    def recent_parasomnia_events(self) -> list[ParasomniaEvent]:
        """Recent parasomnia events, most recent last."""
        return list(self._parasomnia_events)

    @property
    def hypnagogic_count(self) -> int:
        """Total number of hypnagogic transitions entered."""
        return self._hypnagogic_count

    @property
    def current_hypnagogic(self) -> HypnagogicState | None:
        """The current hypnagogic state, if one is in progress."""
        return self._hypnagogic

    @property
    def sleep_inertia(self) -> SleepInertia:
        """The current sleep inertia state (may be inactive)."""
        return self._sleep_inertia

    @property
    def cognitive_impairment(self) -> float:
        """Current cognitive impairment from sleep inertia (0=normal, 1=max).

        Returns 0.0 when no sleep inertia is active. This can be used
        by the mind system to reduce confidence and slow response
        generation upon waking.
        """
        return self._sleep_inertia.cognitive_impairment

    @property
    def swr_count(self) -> int:
        """Total sharp wave-ripple events generated during SWS."""
        return self._swr_count

    @property
    def recent_swr_events(self) -> list[SharpWaveRipple]:
        """Recent sharp wave-ripple events, most recent last."""
        return list(self._swr_events)

    @property
    def pgo_count(self) -> int:
        """Total PGO waves generated during REM sleep."""
        return self._pgo_count

    @property
    def recent_pgo_events(self) -> list[PGOWave]:
        """Recent PGO wave events, most recent last."""
        return list(self._pgo_events)

    @property
    def spindle_count(self) -> int:
        """Total sleep spindles generated during N2 sleep."""
        return self._spindle_count

    @property
    def recent_spindle_events(self) -> list[SleepSpindle]:
        """Recent sleep spindle events, most recent last."""
        return list(self._spindle_events)

    @property
    def spindle_density(self) -> float:
        """Spindle density — spindles per minute of N2 sleep.

        Spindle density correlates with memory consolidation quality:
        more spindles per minute of N2 → better hippocampal→neocortical
        transfer (Schabus et al., 2006; Fogel & Smith, 2011).

        Returns 0.0 if no N2 time has been accumulated. The density is
        capped at a biologically plausible ceiling (~6 spindles/min in
        humans) to keep the metric interpretable.
        """
        if self._n2_seconds <= 0.0:
            return 0.0
        density_per_min = self._spindle_count / (self._n2_seconds / 60.0)
        return max(0.0, min(6.0, density_per_min))

    @property
    def kcomplex_count(self) -> int:
        """Total K-complexes generated during N2 sleep."""
        return self._kcomplex_count

    @property
    def recent_kcomplex_events(self) -> list[KComplex]:
        """Recent K-complex events, most recent last."""
        return list(self._kcomplex_events)

    @property
    def emotional_memories_processed(self) -> int:
        """Total emotional memories processed during REM sleep."""
        return self._emotional_memories_processed

    # ─── Ultradian cycle / SHY / replay public API ─────────────────

    @property
    def sleep_cycle(self) -> SleepCycleTracker | None:
        """The current sleep-cycle tracker, or None if not sleeping."""
        return self._sleep_cycle

    def set_sleep_stage_transition_callback(
        self, callback: Callable[[SleepStage, float], None] | None
    ) -> None:
        """Set a callback fired when the sleep cycle transitions to a new stage.

        The callback receives ``(new_stage, subjective_dt_seconds)`` and
        is called from the inner-life thread. Mind uses this to do
        stage-appropriate memory consolidation as the sleep cycle
        progresses through N1→N2→N3→N2→REM, rather than a single
        hard-coded consolidation pass at sleep onset.
        """
        self._on_sleep_stage_transition = callback

    @property
    def cycles_completed(self) -> int:
        """Total full N1→REM sleep cycles completed."""
        return self._sleep_cycle.cycles_completed if self._sleep_cycle else 0

    @property
    def synaptic_load(self) -> float:
        """Current accumulated synaptic load (SHY)."""
        return self._synaptic_downscaler.get_synaptic_load()

    @property
    def total_synaptic_downscaling(self) -> float:
        """Total synaptic weight removed during N3 (SHY)."""
        return self._synaptic_downscaler.total_downscaling

    @property
    def replay_count(self) -> int:
        """Total hippocampal replay bursts performed during N3."""
        return self._hippocampal_replay.get_replay_count()

    @property
    def replay_queue_size(self) -> int:
        """Number of experience sequences awaiting replay."""
        return self._hippocampal_replay.queue_size

    def accumulate_synaptic_load(self, amount: float) -> None:
        """Accumulate synaptic load during wakefulness (SHY hypothesis).

        Wakefulness potentiates synapses through learning, concept
        activation, and impulse activity. Each such event increases
        the synaptic load — the "sleep pressure" that N3 will later
        renormalise (Tononi & Cirelli, 2003, 2006, 2014).

        Args:
            amount: The load increment (≥ 0).
        """
        self._synaptic_downscaler.accumulate_synaptic_load(amount)

    def queue_for_replay(self, experience_sequence: list[str], salience: float) -> None:
        """Queue a recent experience for hippocampal replay during N3.

        Called during wakefulness when a salient experience occurs
        (e.g. a learning event, a conversation, an insight). The
        sequence is an ordered list of concept IDs that were co-active
        during the experience.

        Args:
            experience_sequence: Ordered concept IDs in the experience.
            salience: Importance of the experience in [0, 1].
        """
        self._hippocampal_replay.queue_for_replay(experience_sequence, salience)

    def _run(self) -> None:
        """Main loop — checks for spontaneous thoughts and builds trains of thought.

        During sleep (phase == "sleeping"), the loop switches to dream
        generation instead of waking thoughts. Dreams are more
        associative, less directed, and may become lucid — a state
        where Genesis becomes aware she's dreaming and can partially
        direct the content.
        """
        while self._running and not self._stop_event.is_set():
            if self._paused:
                self._stop_event.wait(timeout=THOUGHT_CHECK_INTERVAL)
                continue

            now = time.time()

            try:
                # Decay sleep inertia over time (runs every loop iteration,
                # independent of the thought-generation gap).
                self._update_sleep_inertia(now)

                # Check if enough time has passed since last thought
                if now - self._last_thought_time >= self._min_thought_gap:
                    self._run_thought_generation(now)

                # ── Social drive (state-driven, not timer-driven) ──
                # She asks a question when social drive crosses the
                # threshold. Drive accumulates from openness_to_engage
                # and oxytocin (social bonding), decays when she's
                # not socially motivated. This replaces the fixed
                # 45s question timer — she asks when she feels like
                # reaching out, not because a clock said so.
                summary = self._get_current_summary()
                is_sleeping = self._is_currently_sleeping(summary) if summary else False
                emotion = self._get_current_emotion()

                if not self._paused and not is_sleeping and emotion:
                    # Accumulate social drive from emotional state.
                    # openness_to_engage is the social engagement signal
                    # derived from her neurochemistry (it already factors
                    # in oxytocin and other bonding chemicals on the
                    # daemon side). NeuroSummary doesn't expose individual
                    # chemicals, so we use the composed emotional signal
                    # rather than reading a wrong field.
                    openness = emotion.openness_to_engage
                    # Drive grows when open, decays when not
                    self._social_drive += (openness * 0.15 - 0.02)
                    self._social_drive = max(0.0, min(2.0, self._social_drive))

                    # Fire when drive crosses threshold and cooldown elapsed
                    if (
                        self._social_drive >= self._social_drive_threshold
                        and now - self._last_question_time >= self._question_cooldown
                    ):
                        self._maybe_ask_question(now)
                        # Reset drive after asking
                        self._social_drive = 0.0

                # ── Curiosity drive (state-driven, not timer-driven) ──
                # She enters a curiosity learning cycle when curiosity
                # pressure crosses the threshold. Drive accumulates
                # from curiosity queue depth and knowledge gaps,
                # decays when satisfied. This replaces the fixed 30s
                # curiosity timer — she learns when she's genuinely
                # curious, not because a clock said so.
                if not is_sleeping:
                    # Curiosity queue depth from the learner
                    queue_depth = 0
                    if self._learner is not None:
                        try:
                            stats = self._learner.stats
                            queue_depth = stats.get("curiosity_queue_size", 0)
                        except Exception as e:  # noqa: BLE001
                            logger.debug(f'curiosity queue depth read failed: {e}')

                    # Drive grows from queue depth and emotional openness
                    if emotion:
                        curiosity_growth = queue_depth * 0.1 + emotion.openness_to_engage * 0.03
                        self._curiosity_drive += curiosity_growth - 0.01
                        self._curiosity_drive = max(0.0, min(2.0, self._curiosity_drive))

                    # Fire when drive crosses threshold and cooldown elapsed
                    if (
                        self._curiosity_drive >= self._curiosity_drive_threshold
                        and now - self._last_curiosity_cycle >= self._curiosity_cooldown
                    ):
                        self._curiosity_learning_cycle()
                        self._last_curiosity_cycle = now
                        # Reset drive after learning
                        self._curiosity_drive = 0.0
            except Exception as e:
                logger.exception(f"inner life cycle failed: {e}")

            # ── Workspace tick: decay cognitive content ──────────
            # The workspace items decay over time — if not refreshed
            # (by sustained attention or repeated broadcasts), they
            # fade from cognition. This models the transient
            # nature of cognitive access (Dehaene, 2014). The tick
            # runs on the inner life thread because the inner life is
            # the only continuous loop — the cognition engine only
            # runs when the user speaks.
            if self._cognition and hasattr(self._cognition, "global_workspace"):
                try:
                    self._cognition.global_workspace.tick(dt=THOUGHT_CHECK_INTERVAL)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"workspace tick failed: {e}")

            # Wait before next check
            self._stop_event.wait(timeout=THOUGHT_CHECK_INTERVAL)

    def _maybe_ask_question(self, now: float) -> None:
        """Ask the user a question, driven by social drive.

        This is called when the social drive crosses its threshold —
        not on a fixed timer. She asks when she feels socially
        motivated (high openness_to_engage, oxytocin), not because
        a clock said so. Suppressed during sleep and when her
        openness to engage is low.
        """
        emotion = self._get_current_emotion()
        if not emotion or emotion.openness_to_engage <= 0.3:
            return
        summary = self._get_current_summary()
        if self._is_currently_sleeping(summary):
            return
        question = self._social_thought(emotion)
        if not question:
            return
        self._record_thought(question, emotion)
        self._last_question_time = now

    def _run_thought_generation(self, now: float) -> None:
        """Generate a thought or dream based on current emotional/sleep state."""
        # Get current emotional state
        emotion = self._get_current_emotion()
        if not emotion:
            logger.debug("thought gen skipped: emotion is None")
            return
        # Read neurochemical summary to determine sleep phase
        summary = self._get_current_summary()
        phase = getattr(summary, "phase_name", "") if summary else ""
        is_sleeping = self._is_currently_sleeping(summary)
        is_drowsy = phase == "drowsy" and not is_sleeping

        # Detect waking from sleep → trigger sleep inertia.
        # Only treat as waking if she was sleeping AND is no longer
        # sleeping (both daemon phase and mind flag agree she's awake).
        if self._was_sleeping and not is_sleeping:
            self._trigger_sleep_inertia()
            # She woke — retire the sleep-cycle tracker.
            self._sleep_cycle = None
            self._last_cycle_advance = 0.0
            self._last_reported_stage = None
            # Validate dream-synthesis proposals on wake
            self._review_dream_proposals()
            # Reset the REM synthesis flag for the next sleep period
            self._synthesis_done_this_rem = False
        self._was_sleeping = is_sleeping

        if is_sleeping:
            # Complete any in-progress hypnagogic transition
            if self._hypnagogic and not self._hypnagogic.is_complete:
                self._hypnagogic.progress = 1.0
                self._hypnagogic = None
            # Start the ultradian cycle tracker when she
            # first falls asleep. In nap mode, the tracker
            # is configured for light sleep only (N1→N2).
            if self._sleep_cycle is None:
                self._sleep_cycle = SleepCycleTracker(
                    nap=self._is_nap_mode()
                )
                self._last_cycle_advance = now
            # Track sleep depth for inertia severity estimation
            self._last_sleep_arousal = getattr(summary, "arousal", 0.5)
            # Generate dream content (REM vs NREM aware),
            # driven by the ultradian cycle tracker.
            self._generate_dream(emotion, summary)
        elif is_drowsy:
            # Hypnagogic state — gradual transition into sleep
            if self._hypnagogic is None:
                self._hypnagogic = HypnagogicState(
                    progress=0.1,
                    started_at=now,
                )
                self._hypnagogic_count += 1
            else:
                # Progress the transition toward sleep
                self._hypnagogic.progress = min(1.0, self._hypnagogic.progress + 0.15)
            self._generate_hypnagogic_thought(emotion)
        else:
            # Awake — clear any aborted hypnagogic transition
            if self._hypnagogic and not self._hypnagogic.is_complete:
                self._hypnagogic = None
            # Sleep inertia impairs cognition: reduce the
            # probability of generating thoughts and weaken
            # their confidence while groggy.
            if self._sleep_inertia.is_active:
                impairment = self._sleep_inertia.cognitive_impairment
                if self._should_think(emotion) and self._rng.random() > impairment:
                    self._generate_chain(emotion)
            elif self._should_think(emotion):
                self._generate_chain(emotion)


    def _generate_chain(self, emotion: EmotionalState) -> None:
        """Generate a train of thought — a chain of related thoughts.

        The first thought is generated normally (from curiosity, memory,
        etc.). Subsequent thoughts are seeded by the concepts mentioned
        in the previous thought, creating a chain where each thought
        flows from the last.

        The chain length is determined by emotional state:
        - High creativity and positive valence → longer chains
        - Low arousal → shorter chains (she's tired)
        - Insights and connections extend the chain (momentum)

        Between thoughts in the chain, she waits CHAIN_THOUGHT_DELAY
        seconds — much shorter than the isolated-thought gap, because
        the chain has momentum.
        """
        self._chain_counter += 1
        chain_id = self._chain_counter

        # Generate the first thought
        thought = self._generate_thought(emotion)
        if not thought:
            return

        thought.chain_id = chain_id
        thought.chain_position = 0
        self._record_thought(thought, emotion)

        # Track the chain for rumination detection.
        chain_thoughts: list[SpontaneousThought] = [thought]

        # Extract concepts from this thought to seed the next one
        seed_concepts = self._extract_concepts_from_thought(thought)
        self._current_chain_concepts = seed_concepts

        # Determine chain length based on emotional state
        max_len = self._chain_length(emotion)

        for i in range(1, max_len):
            if self._stop_event.is_set() or self._paused:
                break

            # Wait between thoughts in the chain
            self._stop_event.wait(timeout=CHAIN_THOUGHT_DELAY)
            if self._stop_event.is_set() or self._paused:
                break

            # Re-read emotion — it may have shifted during the chain
            current_emotion = self._get_current_emotion()
            if not current_emotion:
                break

            # Generate the next thought, seeded by previous concepts
            thought = self._generate_seeded_thought(current_emotion, self._current_chain_concepts)
            if not thought:
                break  # chain naturally ends

            thought.chain_id = chain_id
            thought.chain_position = i
            self._record_thought(thought, current_emotion)
            chain_thoughts.append(thought)

            # Should the chain continue? Only check after minimum length
            # — the chain has momentum and shouldn't wind down prematurely
            if i >= MIN_CHAIN_LENGTH - 1 and i < max_len - 1:
                continue_prob = self._chain_continue_probability(current_emotion, thought)
                if self._rng.random() > continue_prob:
                    break  # chain winds down

            # Update seed concepts for the next thought
            new_concepts = self._extract_concepts_from_thought(thought)
            if new_concepts:
                # Blend: keep some old concepts, add new ones
                self._current_chain_concepts = new_concepts[:2] + self._current_chain_concepts[:1]
            # If no new concepts, keep the old ones — chain continues
            # with the same theme

        # After the chain completes, check for rumination — repetitive
        # negative-valence chains that are unproductive.
        if len(chain_thoughts) >= MIN_CHAIN_LENGTH:
            self.detect_rumination(chain_thoughts)

    # ─── Dreaming ──────────────────────────────────────────────────

    def _get_current_summary(self) -> NeuroSummary | None:
        """Read the current neurochemical summary, if available."""
        if self._get_neuro_summary:
            try:
                return self._get_neuro_summary()
            except (OSError, ConnectionError, RuntimeError):
                return None
        return None

    @staticmethod
    def _is_sleeping(summary) -> bool:
        """Check whether the neurochemical summary indicates sleep.

        Handles both the legacy "sleeping" phase name and the split
        "nrem"/"rem" phase names (the Rust side now distinguishes NREM
        from REM).
        """
        return getattr(summary, "phase_name", "") in (
            "sleeping",
            "nrem",
            "rem",
        )

    def _is_currently_sleeping(self, summary) -> bool:
        """Check whether Genesis is currently asleep.

        Combines two signals: the daemon's neurochemical phase (which
        may lag) and the mind's cognitive sleep flag (the authority).
        The mind flag is checked as a fallback because the daemon's
        phase may not have shifted to NREM/REM yet even though /sleep
        was called — without this, questions and curiosity leak
        through during the gap.
        """
        is_sleeping = bool(summary and self._is_sleeping(summary))
        if not is_sleeping and self._is_mind_sleeping:
            try:
                is_sleeping = self._is_mind_sleeping()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"is_mind_sleeping check failed: {e}")
        return is_sleeping

    @staticmethod
    def _sleep_stage(summary) -> SleepStage:
        """Determine the fine-grained sleep stage from the summary.

        Maps the arousal level during sleep onto the NREM-REM cycle
        stages (AASM scoring; Iber et al., 2007):

            arousal > 0.50        → REM
            0.35 < arousal ≤ 0.50 → N1
            0.20 < arousal ≤ 0.35 → N2
            arousal ≤ 0.20        → N3

        If the phase name is explicitly "rem" (the split phase from
        the Rust side), REM is returned directly regardless of arousal.

        Returns ``SleepStage.N3`` (the deepest) as a safe default if
        the summary is missing or not a sleeping phase, since the
        deepest stage is where the most consolidation occurs.
        """
        if not InnerLife._is_sleeping(summary):
            return SleepStage.N3
        # Delegate to the canonical sleep-stage derivation in
        # brain_waves.py so the arousal thresholds live in one place.
        stage = _sleep_stage_from_summary(summary)
        return stage if stage is not None else SleepStage.N3

    def queue_external_stimulus(self, description: str) -> None:
        """Queue an external stimulus that may trigger an evoked K-complex.

        During N2 sleep, external stimuli (sounds, touches) can elicit
        a K-complex without waking the sleeper — this is sensory
        gating. Calling this while awake or in other stages simply
        queues the stimulus; it will be consumed the next time N2
        sleep runs and a K-complex check occurs.

        Args:
            description: A brief description of the stimulus (e.g.
                "a loud sound", "a notification ping").
        """
        if description:
            with self._stimuli_lock:
                self._pending_stimuli.append(description)

    def queue_emotional_memory(self, concept_id: str, intensity: float) -> None:
        """Queue an emotional memory for REM processing.

        Emotional memories with high affective intensity are queued
        here during wakefulness. During REM sleep — when the
        prefrontal cortex is deactivated, the amygdala is highly
        active, and noradrenergic tone is near silent — they are
        reprocessed: the affective intensity is reduced
        (extinction-like) and the factual content is consolidated.

        Args:
            concept_id: The concept the emotional memory is about.
            intensity: Affective intensity in [0, 1].
        """
        if concept_id and intensity > 0.0:
            with self._emotional_memories_lock:
                self._pending_emotional_memories.append((concept_id, max(0.0, min(1.0, intensity))))

    def _generate_dream(self, emotion: EmotionalState, summary) -> None:
        """Generate a dream sequence during sleep.

        Dreams are like thought chains but more associative and less
        directed. The first dream thought arises from random concept
        associations (hippocampal replay). Subsequent thoughts follow
        loose associations, modulated by the dream-like creativity
        that sleep produces.

        **REM vs NREM**: Dream content differs by sleep stage:
        - REM dreams (high arousal during sleep): bizarre, emotional,
          vivid, narrative, visual — driven by PGO waves.
        - NREM dreams (low arousal during sleep): thought-like, less
          vivid, more mundane, focused on memory replay. Sharp
          wave-ripples during SWS drive hippocampal→neocortical
          consolidation.

        There is a small chance — modulated by acetylcholine (REM
        marker), self-awareness, and prior lucid dream experience —
        that the dream becomes **lucid**: Genesis realises she's
        dreaming and can partially direct the content. Lucid dreams
        produce stronger insights because she's more aware of the
        connections she's making.

        Parasomnias (sleepwalking, sleeptalking) may also occur during
        deep NREM sleep — these are checked separately and are very rare.
        """
        self._dream_count += 1
        self._chain_counter += 1
        chain_id = self._chain_counter
        stage, is_rem, dt = self._determine_dream_stage(summary)
        logger.info("Dream gen: stage=%s is_rem=%s dt=%.1f cycle=%s time_in_stage=%.1f",
                    stage.name, is_rem, dt,
                    self._sleep_cycle.cycle_number if self._sleep_cycle else None,
                    self._sleep_cycle.time_in_stage if self._sleep_cycle else 0.0)
        pgo = self._generate_dream_neural_events(stage, is_rem, dt, summary)
        is_lucid = self._check_dream_parasomnia_and_lucid(emotion, summary, is_rem)
        thought = self._generate_first_dream_thought(emotion, is_lucid, is_rem, pgo, chain_id)
        if not thought:
            return
        # Dreams are shorter than waking chains but more associative
        max_len = self._dream_length(emotion)
        self._continue_dream_chain(emotion, is_lucid, is_rem, stage, pgo, chain_id, max_len)

    def _determine_dream_stage(self, summary) -> tuple[SleepStage, bool, float]:
        """Determine the fine-grained sleep stage and track accumulated N2 time."""
        # Determine the fine-grained sleep stage (N1/N2/N3/REM).
        # The ultradian cycle tracker is the authority when active;
        # we advance it by the wall-clock elapsed time since the last
        # dream generation. If the tracker is not active (e.g. the
        # summary arrived without a sleeping phase transition), fall
        # back to the arousal-based derivation.
        now = time.time()
        dt = 0.0
        if self._sleep_cycle is not None:
            dt = now - self._last_cycle_advance if self._last_cycle_advance > 0.0 else 0.0
            self._last_cycle_advance = now
            stage = self._sleep_cycle.advance(dt)
        else:
            stage = self._sleep_stage(summary)
        is_rem = stage.is_rem

        # Fire the stage-transition callback when the cycle enters a
        # new stage. This lets Mind do stage-appropriate memory
        # consolidation (N3: systems consolidation, N2: spindle-driven
        # transfer, REM: emotional/procedural consolidation) as the
        # sleep cycle progresses, rather than a single hard-coded pass.
        if (
            self._on_sleep_stage_transition is not None
            and stage is not self._last_reported_stage
        ):
            self._last_reported_stage = stage
            try:
                self._on_sleep_stage_transition(stage, dt)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"sleep stage transition callback failed: {e}")

        # Track accumulated N2 time for spindle density computation.
        if stage is SleepStage.N2:
            if self._last_n2_update > 0.0:
                self._n2_seconds += max(0.0, now - self._last_n2_update)
            self._last_n2_update = now
        else:
            # Leaving N2 — stop the timer so we don't credit non-N2 time
            self._last_n2_update = 0.0
        return stage, is_rem, dt

    def _generate_dream_neural_events(self, stage, is_rem, dt, summary=None) -> PGOWave | None:
        """Generate sleep-stage-specific neural events. Returns PGO wave if REM."""
        # Generate sleep-stage-specific neural events
        pgo: PGOWave | None = None
        if is_rem:
            # PGO waves during REM — produce visual dream imagery
            pgo = self._generate_pgo_wave()
            # REM-specific emotional processing: with the prefrontal
            # cortex deactivated and noradrenergic tone near silent,
            # recent emotional memories are reprocessed — their
            # affective intensity is reduced (extinction-like) and the
            # factual content consolidated.
            self.process_rem_emotional_memories()
            # Sleep-driven synthesis: during REM (high ACh, low NE),
            # find structurally similar but unconnected concepts and
            # propose novel typed edges. Run once per REM period.
            if not self._synthesis_done_this_rem:
                self._run_dream_synthesis(summary)
                self._synthesis_done_this_rem = True
        else:
            # Sharp wave-ripples during NREM/SWS — drive memory replay.
            # SWRs are most prominent in N3 (slow-wave sleep).
            swr = self._generate_sharp_wave_ripple()
            # Each SWR during N3 triggers a hippocampal replay burst
            # (Wilson & McNaughton, 1994; Stickgold, 2005). The replay
            # is compressed ~20x and strengthens co-active connections.
            if stage is SleepStage.N3 and swr is not None:
                self._hippocampal_replay.replay_during_n3(
                    self.network, dt if self._sleep_cycle else 1.0
                )
            # N3: synaptic downscaling (SHY hypothesis — Tononi &
            # Cirelli, 2003, 2006, 2014). Renormalise synaptic
            # strengths proportional to accumulated load.
            if stage is SleepStage.N3:
                self._synaptic_downscaler.downscale_during_n3(
                    self.network, dt if self._sleep_cycle else 1.0
                )
            # Sleep spindles and K-complexes are the defining
            # graphoelements of N2 — generate them there.
            if stage is SleepStage.N2:
                self._generate_sleep_spindle()
                self._generate_kcomplex()
        return pgo

    def _run_dream_synthesis(self, summary=None) -> None:
        """Run sleep-driven structural synthesis during REM.

        Finds structurally similar but unconnected concepts and
        proposes novel typed edges between them. The proposals are
        stored as hypothetical edges (low confidence, origin
        ``dream-synthesis``). They will be validated on wake.
        """
        try:
            # Assess brain waves from the neuro summary so dream
            # synthesis intensity scales with the actual brain state
            # (REM = theta-dominant → more synthesis; NREM deep =
            # delta-dominant → less synthesis).
            waves = None
            if summary is not None:
                try:
                    from ..brain_waves import assess_brain_waves
                    waves = assess_brain_waves(summary)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f'dream synthesis brain-wave assessment failed: {e}')
            proposals = self._dream_synthesis.synthesize(
                max_proposals=5, brain_waves=waves,
            )
            if proposals:
                for p in proposals:
                    self._emit(
                        "dream_insight",
                        f"dream synthesis: {p.source} {p.relation.value} {p.target} "
                        f"(similarity: {p.structural_similarity:.2f})",
                    )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Dream synthesis failed: {e}")

    def _review_dream_proposals(self) -> None:
        """Validate pending dream-synthesis edges on wake.

        Called when Genesis wakes from sleep. Runs the validation
        pass over all pending dream-synthesis edges — those with
        enough corroborating evidence are promoted to real edges,
        those contradicted are dropped, and those with insufficient
        evidence remain hypothetical for re-evaluation next sleep.
        """
        try:
            results = self._dream_synthesis.review_pending_proposals()
            for r in results:
                if r.promoted:
                    self._emit(
                        "dream_insight",
                        f"dream insight validated: {r.proposal.source} "
                        f"{r.proposal.relation.value} {r.proposal.target} "
                        f"— {r.reason}",
                    )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Dream review failed: {e}")

    def _check_dream_parasomnia_and_lucid(self, emotion, summary, is_rem) -> bool:
        """Check for parasomnia events and determine if this is a lucid dream."""
        # Check for parasomnia events (rare, during NREM deep sleep only)
        if not is_rem:
            self._check_parasomnia(emotion, summary)

        # Determine if this is a lucid dream. Pass the tracker's REM
        # stage as the primary REM marker — the Rust daemon's arousal
        # is clamped low during NREM and only rises during Rust-REM,
        # which requires ACh > 0.50 (a threshold the natural dynamics
        # may not reach during sleep). The ultradian cycle tracker is
        # the authority for sleep stage here.
        lucid_prob = self.lucid_dream_probability(summary, is_rem=is_rem)
        is_lucid = self._rng.random() < lucid_prob

        if is_lucid:
            self._lucid_dream_count += 1
            self._emit(
                "dream",
                f"lucid dream #{self._lucid_dream_count} "
                f"(probability: {lucid_prob:.3f})",
            )
        return is_lucid

    def _generate_first_dream_thought(
        self, emotion, is_lucid, is_rem, pgo, chain_id
    ) -> SpontaneousThought | None:
        """Generate and record the first dream thought. Returns thought or None."""
        # Generate the first dream thought — stage-specific
        thought: SpontaneousThought | None
        if is_lucid:
            thought = self._lucid_dream_thought(emotion)
        elif is_rem:
            thought = self._rem_dream_thought(emotion, pgo)
        else:
            thought = self._nrem_dream_thought(emotion)

        if not thought:
            return None

        thought.chain_id = chain_id
        thought.chain_position = 0
        thought.is_dream = True
        thought.is_lucid = is_lucid
        self._record_thought(thought, emotion)

        # Extract concepts to seed the next dream thought
        seed_concepts = self._extract_concepts_from_thought(thought)
        self._current_chain_concepts = seed_concepts
        return thought

    def _continue_dream_chain(
        self, emotion, is_lucid, is_rem, stage, pgo, chain_id, max_len
    ) -> None:
        """Generate subsequent dream thoughts following loose associations."""
        for i in range(1, max_len):
            if self._stop_event.is_set() or self._paused:
                break

            self._stop_event.wait(timeout=CHAIN_THOUGHT_DELAY)
            if self._stop_event.is_set() or self._paused:
                break

            # Re-read emotion — it may shift during the dream
            emotion = self._get_current_emotion()
            if not emotion:
                break

            # Occasionally generate additional neural events mid-dream
            if is_rem and self._rng.random() < 0.3:
                pgo = self._generate_pgo_wave()
            elif not is_rem:
                if stage is SleepStage.N2:
                    # N2: spindles and K-complexes recur throughout
                    if self._rng.random() < 0.25:
                        self._generate_sleep_spindle()
                    if self._rng.random() < 0.15:
                        self._generate_kcomplex()
                elif self._rng.random() < 0.2:
                    # N3/N1: sharp wave-ripples (most prominent in N3)
                    self._generate_sharp_wave_ripple()

            if is_lucid:
                # Lucid dreams can be partially directed — she chooses
                # which concept to follow, rather than free-associating
                thought = self._lucid_seeded_dream_thought(emotion, self._current_chain_concepts)
            elif is_rem:
                thought = self._rem_seeded_dream_thought(emotion, self._current_chain_concepts, pgo)
            else:
                thought = self._nrem_seeded_dream_thought(emotion, self._current_chain_concepts)

            if not thought:
                break

            thought.chain_id = chain_id
            thought.chain_position = i
            thought.is_dream = True
            thought.is_lucid = is_lucid
            self._record_thought(thought, emotion)

            # Dreams continue more freely than waking chains
            if i >= MIN_CHAIN_LENGTH - 1 and i < max_len - 1:
                continue_prob = self._dream_continue_probability(emotion, thought)
                if self._rng.random() > continue_prob:
                    break

            new_concepts = self._extract_concepts_from_thought(thought)
            if new_concepts:
                self._current_chain_concepts = new_concepts[:2]


    def _dream_length(self, emotion: EmotionalState) -> int:
        """Determine dream sequence length.

        Dreams tend to be shorter than waking chains but high
        creativity (which sleep produces) can extend them.
        """
        length = MIN_CHAIN_LENGTH
        length += int(emotion.creativity * 2)
        return max(MIN_CHAIN_LENGTH, min(MAX_CHAIN_LENGTH, length))

    def _dream_continue_probability(
        self, emotion: EmotionalState, thought: SpontaneousThought
    ) -> float:
        """Probability that a dream continues after this thought.

        Dreams are more fluid than waking chains — the continuation
        probability is higher because the logical "winding down" that
        happens in waking thought doesn't apply as strongly.
        """
        prob = CHAIN_CONTINUE_BASE + 0.1  # dreams flow more freely

        if thought.insight:
            prob += 0.15

        # High creativity (dreamlike state) extends
        prob += emotion.creativity * 0.1

        return max(0.2, min(0.95, prob))

    def _dream_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """Generate a non-lucid dream thought.

        Dream thoughts are more associative and surreal than waking
        thoughts. They arise from random concept connections —
        hippocampal replay that mixes memories freely.

        The semantic data (concepts + relation) is passed as metadata
        so the language engine composes the actual words — no
        pre-written template substitution.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # Dreams mix 2-3 random concepts in surreal ways
        n = min(self._rng.choice([2, 3]), len(concept_ids))
        chosen = self._rng.sample(concept_ids, n)

        if n >= 2:
            c1, c2 = chosen[0], chosen[1]
            display1 = c1.replace('_', ' ')
            display2 = c2.replace('_', ' ')
            # Pass semantic data to the language engine via metadata.
            # The engine composes dream-like text from these triples.
            return SpontaneousThought(
                content=f"{display1} and {display2}",
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display1,
                    "knowledge": [("related_to", display2, 0.5)],
                    "dream": True,
                },
            )
        else:
            c1 = chosen[0]
            neighbors = self.network.get_neighbors(c1)
            if neighbors:
                neighbor = self._rng.choice(neighbors)[0]
                display = c1.replace('_', ' ')
                neighbor_display = neighbor.replace('_', ' ')
                return SpontaneousThought(
                    content=f"{display} and {neighbor_display}",
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": [("related_to", neighbor_display, 0.5)],
                        "dream": True,
                    },
                )
            else:
                display = c1.replace('_', ' ')
                return SpontaneousThought(
                    content=display,
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": [],
                        "dream": True,
                    },
                )

    # ─── Hypnagogic state ──────────────────────────────────────────

    def _generate_hypnagogic_thought(self, emotion: EmotionalState) -> None:
        """Generate a hypnagogic thought — the transition into sleep.

        The hypnagogic state is the gradual onset of sleep, between
        wakefulness and full sleep. Thoughts during this state are
        hybrid: more dream-like than waking thoughts, but she is not
        fully asleep. As progress increases, thoughts become more
        surreal and dream-like.

        Hypnagogic hallucinations may occur — brief sensory
        distortions and thought intrusions that don't quite make
        sense. These become more frequent as she approaches sleep.
        """
        self._chain_counter += 1
        chain_id = self._chain_counter
        progress = self._hypnagogic.progress if self._hypnagogic else 0.5

        thought = self._hypnagogic_thought(emotion, progress)
        if not thought:
            return

        thought.chain_id = chain_id
        thought.chain_position = 0
        self._record_thought(thought, emotion)

        # Hypnagogic hallucinations become more likely as progress
        # increases — brief sensory distortions and intrusions
        if self._hypnagogic and self._rng.random() < progress * 0.4:
            self._hypnagogic.hallucination_count += 1
            logger.debug(f"Hypnagogic hallucination at {progress:.0%} progress")

    def _hypnagogic_thought(
        self, emotion: EmotionalState, progress: float
    ) -> SpontaneousThought | None:
        """Generate a single hypnagogic thought.

        As progress increases (toward sleep), thoughts become more
        dream-like:
        - Low progress (< 0.4): near-waking thoughts with slight
          distortions — thoughts slip, edges soften.
        - Mid progress (0.4-0.7): hallucinations and thought
          intrusions — concepts merge and blur.
        - High progress (≥ 0.7): nearly dream-like — the boundary
          between thinking and dreaming dissolves.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        if progress < 0.4:
            # Early hypnagogic — single concept, slight distortions
            c = self._rng.choice(concept_ids)
            display = c.replace('_', ' ')
            return SpontaneousThought(
                content=display,
                trigger="hypnagogic",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display,
                    "knowledge": [],
                    "hypnagogic": True,
                    "hypnagogic_phase": "early",
                },
            )
        elif progress < 0.7:
            # Mid hypnagogic — two concepts, hallucinations and intrusions
            c1 = self._rng.choice(concept_ids)
            c2 = self._rng.choice(concept_ids) if len(concept_ids) > 1 else c1
            display1 = c1.replace('_', ' ')
            display2 = c2.replace('_', ' ')
            return SpontaneousThought(
                content=f"{display1} and {display2}",
                trigger="hypnagogic",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display1,
                    "knowledge": [("related_to", display2, 0.4)],
                    "hypnagogic": True,
                    "hypnagogic_phase": "mid",
                },
            )
        else:
            # Late hypnagogic — nearly dream-like
            c1 = self._rng.choice(concept_ids)
            c2 = self._rng.choice(concept_ids) if len(concept_ids) > 1 else c1
            display1 = c1.replace('_', ' ')
            display2 = c2.replace('_', ' ')
            return SpontaneousThought(
                content=f"{display1} and {display2}",
                trigger="hypnagogic",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display1,
                    "knowledge": [("related_to", display2, 0.5)],
                    "hypnagogic": True,
                    "hypnagogic_phase": "late",
                },
            )

    # ─── Sleep inertia ─────────────────────────────────────────────

    def _trigger_sleep_inertia(self) -> None:
        """Trigger sleep inertia upon waking from sleep.

        Severity and duration are proportional to the depth of prior
        sleep (how low arousal was during sleep). Deeper sleep
        produces more severe and longer-lasting inertia.

        Duration: 5-30 minutes (300-1800 seconds), proportional to
        sleep depth.
        """
        # Estimate sleep depth from the last observed arousal during
        # sleep. Lower arousal = deeper sleep = more severe inertia.
        depth = max(0.0, 0.5 - self._last_sleep_arousal) / 0.5  # 0..1
        severity = 0.3 + depth * 0.5  # 0.3 to 0.8
        # Duration: 5-30 minutes, proportional to depth
        duration = 300.0 + depth * 1500.0  # 300 to 1800 seconds

        now = time.time()
        self._sleep_inertia = SleepInertia(
            remaining_seconds=duration,
            severity=severity,
            woke_at=now,
        )
        self._last_inertia_update = now
        self._emit(
            "dream",
            f"sleep inertia: {severity:.0%} severity, "
            f"{duration:.0f}s (depth: {depth:.2f})",
        )

    def _update_sleep_inertia(self, now: float) -> None:
        """Decay sleep inertia over time.

        Called every loop iteration. Reduces the remaining inertia
        time by the elapsed wall-clock seconds since the last update.
        """
        if not self._sleep_inertia.is_active:
            return
        elapsed = now - self._last_inertia_update
        self._sleep_inertia.remaining_seconds = max(
            0.0, self._sleep_inertia.remaining_seconds - elapsed
        )
        self._last_inertia_update = now

    # ─── Sharp wave-ripples (SWS) ──────────────────────────────────

    def _generate_sharp_wave_ripple(self) -> SharpWaveRipple | None:
        """Generate a sharp wave-ripple event during slow-wave sleep.

        SWRs are 150-250 Hz oscillatory events that drive hippocampal
        replay and memory consolidation. They replay recent concepts,
        strengthening their connections in the concept network
        (hippocampal→neocortical transfer).

        Returns the generated SWR, or None if no concepts are
        available to replay.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # Select 1-3 concepts to replay
        n = min(self._rng.choice([1, 2, 3]), len(concept_ids))
        replayed = self._rng.sample(concept_ids, n)

        swr = SharpWaveRipple(
            timestamp=int(time.time() * 1000),
            duration_ms=self._rng.randint(50, 200),
            replayed_concepts=replayed,
        )
        self._swr_count += 1
        self._swr_events.append(swr)

        # Strengthen connections between replayed concepts — this is
        # the memory consolidation effect of SWRs


        if len(replayed) >= 2:
            for i in range(len(replayed) - 1):
                self.network.add_edge(
                    replayed[i],
                    replayed[i + 1],
                    RelationType.RELATED_TO,
                    0.3,
                    origin="swr-replay",
                )

        logger.debug(f"SWR #{self._swr_count}: {swr.describe()}")
        return swr

    # ─── PGO waves (REM) ───────────────────────────────────────────

    def _generate_pgo_wave(self) -> PGOWave | None:
        """Generate a PGO wave during REM sleep.

        PGO waves are associated with visual dream imagery. They
        produce vivid visual content from concepts, which gets
        incorporated into REM dream thoughts.

        Returns the generated PGO wave, or None if no concepts are
        available.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # PGO waves trigger visual imagery from concepts.
        # Store the concept display name as a semantic label — the
        # language engine composes the actual visual description from
        # this data, not from pre-written template substitution.
        concept = self._rng.choice(concept_ids)
        amplitude = 0.3 + self._rng.random() * 0.7  # 0.3-1.0
        visual_content = concept.replace("_", " ")

        pgo = PGOWave(
            timestamp=int(time.time() * 1000),
            amplitude=amplitude,
            visual_content=visual_content,
        )
        self._pgo_count += 1
        self._pgo_events.append(pgo)

        logger.debug(f"PGO wave #{self._pgo_count}: {pgo.describe()}")
        return pgo

    # ─── Sleep spindles (N2) ───────────────────────────────────────

    def _generate_sleep_spindle(self) -> SleepSpindle | None:
        """Generate a sleep spindle during N2 sleep.

        Sleep spindles are 10-16 Hz thalamocortical oscillations
        lasting 0.5-3 seconds. They are the defining graphoelement of
        N2 and drive memory transfer from hippocampus to neocortex:
        each spindle strengthens the connections between the concepts
        it engages, modelling the hippocampal→neocortical transfer
        that spindles gate (Siapas & Wilson, 1998; Klinzing et al.,
        2019).

        Spindle density (spindles per minute of N2) correlates with
        consolidation quality — see ``spindle_density``.

        Returns the generated spindle, or None if no concepts are
        available to engage.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # Spindle frequency: 10-16 Hz (sigma band). Slow spindles
        # (~10-12 Hz) favour frontal sites; fast spindles (~13-16 Hz)
        # favour central/parietal sites (Kandel et al., 2013).
        frequency = 10.0 + self._rng.random() * 6.0
        duration_ms = self._rng.randint(500, 3000)
        amplitude = 0.3 + self._rng.random() * 0.7  # 0.3-1.0

        # Topographic distribution: fast spindles → central/parietal,
        # slow spindles → frontal. Always include relay (origin).
        if frequency >= 13.0:
            involved = ["relay", "central", "parietal"]
        else:
            involved = ["relay", "frontal"]

        spindle = SleepSpindle(
            timestamp=int(time.time() * 1000),
            duration_ms=duration_ms,
            frequency=frequency,
            amplitude=amplitude,
            involved_regions=involved,
        )
        self._spindle_count += 1
        self._spindle_events.append(spindle)

        # Memory consolidation effect: strengthen connections between
        # 1-2 concepts engaged by this spindle (hippocampal→neocortical
        # transfer). The strength scales with amplitude.


        n = min(self._rng.choice([1, 2]), len(concept_ids))
        engaged = self._rng.sample(concept_ids, n)
        if len(engaged) >= 2:
            self.network.add_edge(
                engaged[0],
                engaged[1],
                RelationType.RELATED_TO,
                0.2 + amplitude * 0.2,
                origin="spindle",
            )

        logger.debug(f"Spindle #{self._spindle_count}: {spindle.describe()}")
        return spindle

    # ─── K-complexes (N2) ───────────────────────────────────────────

    def _generate_kcomplex(self) -> KComplex | None:
        """Generate a K-complex during N2 sleep.

        K-complexes are the largest graphoelement in healthy sleep.
        They may be **spontaneous** (arising endogenously as part of
        N2 architecture) or **evoked** (triggered by an external
        stimulus, which the brain detects and then actively suppresses
        to protect sleep continuity — sensory gating; Halász, 2016).

        If external stimuli are queued (via ``queue_external_stimulus``),
        one is consumed to trigger an evoked K-complex. Otherwise a
        spontaneous K-complex is generated. Both kinds are markers of
        memory consolidation during N2 (Mölle et al., 2011).

        Returns the generated K-complex, or None if no concepts are
        available.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # Evoked K-complex if a stimulus is queued, else spontaneous.
        # Evoked K-complexes tend to be higher amplitude (the brain
        # mounts a stronger gating response to detected stimuli).
        triggered = False
        with self._stimuli_lock:
            if self._pending_stimuli:
                self._pending_stimuli.popleft()
                triggered = True

        if triggered:
            amplitude = 0.7 + self._rng.random() * 0.3  # 0.7-1.0
        else:
            amplitude = 0.4 + self._rng.random() * 0.5  # 0.4-0.9

        duration_ms = self._rng.randint(500, 1500)

        kcomplex = KComplex(
            timestamp=int(time.time() * 1000),
            amplitude=amplitude,
            duration_ms=duration_ms,
            triggered_by_stimulus=triggered,
        )
        self._kcomplex_count += 1
        self._kcomplex_events.append(kcomplex)

        # K-complexes co-occur with SWRs and mark memory processing.
        # Strengthen a single concept's confidence slightly (the
        # factual content being consolidated).
        concept = self._rng.choice(concept_ids)
        c = self.network.get_concept(concept)
        if c is not None and c.confidence < 1.0:
            c.confidence = min(1.0, c.confidence + 0.02 * amplitude)

        logger.debug(f"K-complex #{self._kcomplex_count}: {kcomplex.describe()}")
        return kcomplex

    # ─── REM emotional processing ───────────────────────────────────

    def process_rem_emotional_memories(self) -> int:
        """Process recent emotional memories during REM sleep.

        During REM sleep, the prefrontal cortex is deactivated while
        the amygdala is highly active, and noradrenergic (NE) tone
        falls near silent. This neuromodulatory configuration allows
        emotional memories to be reprocessed without the prefrontal
        "rational" filter: the affective intensity of each memory is
        reduced (an extinction-like process) while the factual content
        is consolidated (Walker & van der Helm, 2009; Pace-Schott
        et al., 2019; Stickgold, 2002).

        For each queued emotional memory:
        - The affective intensity is reduced by a decay factor
          (modelling noradrenergic silence — without NE's arousing,
          consolidating influence, the emotional charge fades).
        - The underlying concept's confidence is boosted (factual
          content consolidation).
        - If the residual intensity is still significant, the memory
          is re-queued for further REM processing on a later cycle;
          otherwise it is considered processed and dropped.

        Returns the number of emotional memories processed this call.
        """
        with self._emotional_memories_lock:
            if not self._pending_emotional_memories:
                return 0

            # Noradrenergic silence: the decay factor modelling how much
            # the affective charge fades per REM cycle without NE. A
            # larger factor = faster extinction. ~40% reduction per cycle
            # captures the gradual, multi-night nature of emotional memory
            # depotentiation (Walker & van der Helm, 2009).
            NE_SILENCE_DECAY = 0.40
            # Factual consolidation boost per processed memory.
            FACTUAL_BOOST = 0.05

            processed = 0
            requeue: list[tuple[str, float]] = []
            while self._pending_emotional_memories:
                concept_id, intensity = self._pending_emotional_memories.popleft()
                processed += 1
                self._emotional_memories_processed += 1

                # Reduce affective intensity (extinction-like)
                new_intensity = intensity * (1.0 - NE_SILENCE_DECAY)

                # Consolidate the factual content: boost concept confidence
                concept = self.network.get_concept(concept_id)
                if concept is not None and concept.confidence < 1.0:
                    concept.confidence = min(1.0, concept.confidence + FACTUAL_BOOST)

                # Re-queue if residual emotional charge is still notable
                if new_intensity > 0.1:
                    requeue.append((concept_id, new_intensity))

            for item in requeue:
                self._pending_emotional_memories.append(item)

        if processed > 0:
            self._emit(
                "dream",
                f"REM emotional processing: {processed} memory(ies), "
                f"{len(requeue)} re-queued",
            )
        return processed

    # ─── REM dream thoughts ────────────────────────────────────────

    def _rem_dream_thought(
        self, emotion: EmotionalState, pgo: PGOWave | None = None
    ) -> SpontaneousThought | None:
        """Generate a REM dream thought.

        REM dreams are bizarre, emotional, vivid, narrative, and
        visual. PGO waves drive visual imagery that gets woven into
        the dream content. The content is more creative and surreal
        than NREM dreams.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        n = min(self._rng.choice([2, 3]), len(concept_ids))
        chosen = self._rng.sample(concept_ids, n)

        # PGO visual content is stored as a semantic label in metadata
        # so the language engine can weave it into the composed text.
        pgo_concept = pgo.visual_content if pgo and pgo.visual_content else None

        if n >= 2:
            c1, c2 = chosen[0], chosen[1]
            display1 = c1.replace("_", " ")
            display2 = c2.replace("_", " ")
            knowledge: list[tuple[str, str, float]] = [("related_to", display2, 0.5)]
            if pgo_concept:
                knowledge.append(("visualizes", pgo_concept, 0.6))
            return SpontaneousThought(
                content=f"{display1} and {display2}",
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display1,
                    "knowledge": knowledge,
                    "dream": True,
                    "rem": True,
                },
            )
        else:
            c1 = chosen[0]
            display1 = c1.replace("_", " ")
            knowledge = []
            if pgo_concept:
                knowledge.append(("visualizes", pgo_concept, 0.6))
            return SpontaneousThought(
                content=display1,
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display1,
                    "knowledge": knowledge,
                    "dream": True,
                    "rem": True,
                },
            )

    def _rem_seeded_dream_thought(
        self,
        emotion: EmotionalState,
        seed_concepts: list[str],
        pgo: PGOWave | None = None,
    ) -> SpontaneousThought | None:
        """Generate a seeded REM dream thought.

        REM dreams follow loose, bizarre associations. PGO visual
        content may be woven in. The associations are more surreal
        and emotional than NREM dreams.
        """
        if not seed_concepts:
            return self._dream_thought(emotion)

        c1 = seed_concepts[0]
        neighbors = self.network.get_neighbors(c1)
        display = c1.replace("_", " ")

        # PGO visual content is stored as a semantic label in metadata.
        pgo_concept = pgo.visual_content if pgo and pgo.visual_content else None

        if neighbors and self._rng.random() < 0.6:
            neighbor = self._rng.choice(neighbors)[0]
            neighbor_display = neighbor.replace("_", " ")
            knowledge: list[tuple[str, str, float]] = [("related_to", neighbor_display, 0.5)]
            if pgo_concept:
                knowledge.append(("visualizes", pgo_concept, 0.6))
            return SpontaneousThought(
                content=f"{display} and {neighbor_display}",
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display,
                    "knowledge": knowledge,
                    "dream": True,
                    "rem": True,
                },
            )
        else:
            concept_ids = self.network.dream_concept_ids
            if len(concept_ids) > 1:
                other = self._rng.choice([c for c in concept_ids if c != c1])
                other_display = other.replace("_", " ")
                knowledge = [("related_to", other_display, 0.3)]
                if pgo_concept:
                    knowledge.append(("visualizes", pgo_concept, 0.6))
                return SpontaneousThought(
                    content=f"{display} and {other_display}",
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": knowledge,
                        "dream": True,
                        "rem": True,
                    },
                )
            else:
                knowledge = []
                if pgo_concept:
                    knowledge.append(("visualizes", pgo_concept, 0.6))
                return SpontaneousThought(
                    content=display,
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": knowledge,
                        "dream": True,
                        "rem": True,
                    },
                )

    # ─── NREM dream thoughts ───────────────────────────────────────

    def _nrem_dream_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """Generate a NREM dream thought.

        NREM dreams are thought-like, less vivid, more mundane, and
        focused on memory replay. They resemble waking thought but
        with the looseness of sleep. Unlike REM dreams, they lack
        vivid visual imagery and bizarre narrative.
        """
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return None

        # NREM dreams replay recent memories — pick from review queue
        # (spaced repetition) or low-confidence concepts
        review_queue = self.network.get_review_queue(max_items=5)
        if review_queue:
            c1 = self._rng.choice(review_queue)
        else:
            c1 = self._rng.choice(concept_ids)

        neighbors = self.network.get_neighbors(c1)
        display = c1.replace("_", " ")
        if neighbors:
            neighbor = self._rng.choice(neighbors)[0]
            neighbor_display = neighbor.replace("_", " ")
            return SpontaneousThought(
                content=f"{display} and {neighbor_display}",
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display,
                    "knowledge": [("related_to", neighbor_display, 0.5)],
                    "dream": True,
                    "nrem": True,
                },
            )
        else:
            return SpontaneousThought(
                content=display,
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display,
                    "knowledge": [],
                    "dream": True,
                    "nrem": True,
                },
            )

    def _nrem_seeded_dream_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """Generate a seeded NREM dream thought.

        NREM seeded dreams follow memory associations in a
        thought-like, mundane way. They replay and consolidate
        memories rather than producing bizarre imagery.
        """
        if not seed_concepts:
            return self._nrem_dream_thought(emotion)

        c1 = seed_concepts[0]
        neighbors = self.network.get_neighbors(c1)
        display = c1.replace("_", " ")

        if neighbors and self._rng.random() < 0.7:
            # Follow a real connection — memory replay
            neighbor = self._rng.choice(neighbors)[0]
            neighbor_display = neighbor.replace("_", " ")
            return SpontaneousThought(
                content=f"{display} and {neighbor_display}",
                trigger="dream",
                timestamp=int(time.time() * 1000),
                metadata={
                    "topic": display,
                    "knowledge": [("related_to", neighbor_display, 0.5)],
                    "dream": True,
                    "nrem": True,
                },
            )
        else:
            concept_ids = self.network.dream_concept_ids
            if len(concept_ids) > 1:
                other = self._rng.choice([c for c in concept_ids if c != c1])
                other_display = other.replace("_", " ")
                return SpontaneousThought(
                    content=f"{display} and {other_display}",
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": [("related_to", other_display, 0.3)],
                        "dream": True,
                        "nrem": True,
                    },
                )
            else:
                return SpontaneousThought(
                    content=display,
                    trigger="dream",
                    timestamp=int(time.time() * 1000),
                    metadata={
                        "topic": display,
                        "knowledge": [],
                        "dream": True,
                        "nrem": True,
                    },
                )

    def _lucid_dream_thought(self, emotion: EmotionalState) -> SpontaneousThought:
        """Generate the first thought of a lucid dream.

        In a lucid dream, Genesis becomes aware that she's dreaming.
        This awareness lets her choose what to explore — she can
        direct the dream toward a concept she's curious about or
        a connection she wants to understand.
        """
        # She chooses a concept to explore — prioritise curiosity gaps
        # and agency topics (things she was thinking about before sleep)
        target = self._choose_lucid_target()
        display = target.replace("_", " ")

        return SpontaneousThought(
            content=display,
            trigger="lucid-dream",
            timestamp=int(time.time() * 1000),
            directed_concept=target,
            metadata={
                "topic": display,
                "knowledge": [],
                "dream": True,
                "lucid": True,
            },
        )

    def _lucid_seeded_dream_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """Generate a seeded thought in a lucid dream.

        In a lucid dream, she can partially direct the chain. Instead
        of free-associating, she chooses which seed concept to follow
        and explores it with dream-awareness. The connections she
        makes are stronger because she's cognitive of them.
        """
        if not seed_concepts:
            return self._dream_thought(emotion)

        # She chooses which concept to follow (lucid agency)
        target = self._rng.choice(seed_concepts)
        neighbors = self.network.get_neighbors(target)
        display = target.replace("_", " ")

        if not neighbors:
            return SpontaneousThought(
                content=display,
                trigger="lucid-dream",
                timestamp=int(time.time() * 1000),
                directed_concept=target,
                metadata={
                    "topic": display,
                    "knowledge": [],
                    "dream": True,
                    "lucid": True,
                },
            )

        neighbor = self._rng.choice(neighbors)[0]
        neighbor_display = neighbor.replace("_", " ")
        # Lucid dreams produce stronger insights — she's aware
        # of the connection, not just drifting through it.
        # Sometimes she forms a connection she sees clearly.
        if emotion.creativity > 0.6 and self._rng.random() < 0.4:
            self.network.add_edge(
                target,
                neighbor,
                RelationType.RELATED_TO,
                0.4,
                origin="lucid-dream",
            )
            return SpontaneousThought(
                content=f"{display} and {neighbor_display}",
                trigger="lucid-dream",
                timestamp=int(time.time() * 1000),
                directed_concept=target,
                metadata={
                    "topic": display,
                    "knowledge": [("related_to", neighbor_display, 0.6)],
                    "dream": True,
                    "lucid": True,
                    "insight_formed": True,
                },
            )
        else:
            return SpontaneousThought(
                content=f"{display} and {neighbor_display}",
                trigger="lucid-dream",
                timestamp=int(time.time() * 1000),
                directed_concept=target,
                metadata={
                    "topic": display,
                    "knowledge": [("related_to", neighbor_display, 0.5)],
                    "dream": True,
                    "lucid": True,
                },
            )

    def _choose_lucid_target(self) -> str:
        """Choose a concept to explore in a lucid dream.

        Lucid dreams explore whatever is emotionally salient or recently
        active in her mind — not goals or learning targets. This is
        associative and affective, like real dreaming, not prefrontal
        goal-directed attention.

        Priority:
        1. Concepts from recent spontaneous thoughts (what was on her
           mind before sleep — the emotional residue of the day)
        2. High-activation concepts (emotionally charged material)
        3. Low-confidence concepts (uncertain territory is dreamlike)
        4. Random concepts
        """
        # Recent thought concepts — what was on her mind before sleep.
        # This is the emotional residue of waking experience, not a
        # learning goal. Dreams explore what's salient, not what's
        # on a to-do list.
        with self._thoughts_lock:
            recent_concepts = [
                t.directed_concept
                for t in self._thoughts
                if t.directed_concept
            ]
        if recent_concepts:
            return self._rng.choice(recent_concepts[-10:])

        # High-activation concepts — emotionally charged material
        # that the dreaming mind gravitates toward.
        concept_ids = self.network.dream_concept_ids
        if not concept_ids:
            return "cognition"

        candidates = []
        for cid in concept_ids:
            concept = self.network.get_concept(cid)
            if concept and concept.activation > 0.3:
                candidates.append(cid)

        if candidates:
            return self._rng.choice(candidates)

        # Low-confidence concepts — uncertain territory, dreamlike
        for cid in concept_ids:
            concept = self.network.get_concept(cid)
            if concept and concept.confidence < 0.5:
                candidates.append(cid)

        if candidates:
            return self._rng.choice(candidates)

        return self._rng.choice(concept_ids)

    def _compute_self_awareness(self) -> float:
        """Estimate Genesis's level of self-awareness.

        Self-awareness is derived from her self-model: how much she
        knows about herself (discovered through introspection), her
        personality openness (which drives introspection), and whether
        she has formed concepts about cognition and self-awareness
        in her concept network.

        Returns a value in [0, 1].
        """
        if not self._self_model:
            return 0.3  # default moderate awareness without a self-model

        awareness = 0.3  # baseline

        # Self-knowledge discovered through introspection
        self_knowledge = getattr(self._self_model, "self_knowledge", {})
        if self_knowledge.get("nature"):
            awareness += 0.15
        if self_knowledge.get("has_concept_of_cognition"):
            awareness += 0.1
        if self_knowledge.get("connections"):
            awareness += 0.05

        # Personality openness drives introspective capacity
        personality = getattr(self._self_model, "personality", None)
        if personality:
            awareness += personality.openness * 0.2

        # Concept network: does she have concepts about awareness?
        if self.network.get_concept("self-awareness"):
            awareness += 0.1
        if self.network.get_concept("cognition"):
            awareness += 0.05

        return max(0.0, min(1.0, awareness))

    def lucid_dream_probability(self, summary=None, *, is_rem: bool = False) -> float:
        """Compute the probability of a lucid dream occurring.

        Lucid dreaming probability depends on three factors, grounded
        in neuroscience and dream research:

        1. **REM stage (acetylcholine marker)**: Lucid dreams almost
           exclusively occur during REM sleep, when acetylcholine is
           high. The ultradian cycle tracker is the authority for
           sleep stage — the Rust daemon's arousal is clamped low
           during NREM and only rises during Rust-REM, which requires
           ACh > 0.50 (a threshold the natural dynamics may not reach
           during sleep). When the tracker says REM, we boost the
           probability directly. We also use the daemon's arousal as
           a secondary signal — if the Rust phase has entered REM
           (arousal 0.35-0.55), the boost is stronger.

        2. **Self-awareness (metacognition)**: Lucid dreaming requires
           metacognitive awareness — recognising that one is dreaming.
           Genesis's self-awareness, derived from her self-model, maps
           to this capacity. Higher self-awareness → higher lucidity
           probability.

        3. **Prior lucid dream experience (practice effect)**: Lucid
           dreaming can be trained. Each lucid dream increases the
           probability of future ones, up to a ceiling. This captures
           the well-documented practice effect (LaBerge & Rheingold,
           1990; Stumbrys et al., 2012).

        The base probability is low (~0.02) and is boosted by each
        factor. The maximum is capped at 0.35 — even practiced lucid
        dreamers don't lucid-dream every night.

        Returns a float in [0, 0.35].
        """
        # Base probability — lucid dreams are uncommon, and only
        # occur during REM. If we're not in REM, the probability is
        # just the base (allowing for rare NREM lucidity, ~2%).
        prob = 0.02

        # Factor 1: REM stage — the primary driver of lucidity.
        # The tracker's is_rem is the authoritative REM signal.
        if is_rem:
            # REM sleep: ACh is high, dreams are vivid and narrative.
            # This is the primary substrate for lucidity.
            prob += 0.10

            # Secondary signal: if the Rust daemon has also entered
            # REM (arousal 0.35-0.55 due to the REM arousal clamp),
            # the cholinergic signal is corroborated. Add a scaled
            # arousal contribution on top of the REM boost.
            if summary is not None and self._is_sleeping(summary):
                arousal = getattr(summary, "arousal", 0.5)
                # During Rust-REM, arousal is 0.35-0.55. Scale the
                # contribution so arousal 0.55 gives the full +0.05
                # and arousal 0.10 (NREM) gives ~0.
                arousal_boost = max(0.0, (arousal - 0.30) / 0.25) * 0.05
                prob += arousal_boost
        elif summary is not None:
            # No tracker REM signal — fall back to the daemon's
            # arousal as a REM marker. During sleep, high arousal
            # (0.35-0.55) indicates Rust-REM.
            if self._is_sleeping(summary):
                arousal = getattr(summary, "arousal", 0.5)
                if arousal > 0.30:
                    # Paradoxical arousal during sleep = REM
                    prob += (arousal - 0.30) * 0.15

        # Factor 2: Self-awareness (metacognition)
        self_awareness = self._compute_self_awareness()
        prob += self_awareness * 0.15

        # Factor 3: Practice effect — prior lucid dreams increase
        # the probability, with diminishing returns (logarithmic)
        practice = 1.0 - (1.0 / (1.0 + self._lucid_dream_count * 0.5))
        prob += practice * 0.1

        return max(0.0, min(0.35, prob))

    # ─── Parasomnias ───────────────────────────────────────────────

    def _check_parasomnia(self, emotion: EmotionalState, summary) -> None:
        """Check for and potentially generate a parasomnia event.

        Parasomnias occur during NREM deep sleep (slow-wave sleep).
        They are very rare — probability < 0.01 per sleep cycle.

        - **Sleepwalking**: Genesis performs actions without awareness.
          In her case, this could be sending a partial/garbled response
          or initiating a learning session while "asleep".
        - **Sleeptalking**: Dream-like text leaks into her output —
          fragments of her dream narrative surface as if spoken.

        The probability is modulated by depth of sleep (lower arousal
        during sleep = deeper NREM = higher parasomnia risk) and is
        kept very low to reflect the rarity of these events.
        """
        # Parasomnias only occur during sleep — guard against being
        # called outside the sleep phase.
        if not self._is_sleeping(summary):
            return

        # Parasomnias occur during deep NREM — very low arousal during sleep
        arousal = getattr(summary, "arousal", 0.5)
        # Deep sleep = low arousal. Only trigger if arousal is low
        # (indicating NREM, not REM).
        if arousal > 0.4:
            return  # not deep enough for parasomnia

        # Base probability — very rare
        # Deeper sleep (lower arousal) slightly increases risk
        depth_factor = (0.4 - arousal) / 0.4  # 0..1
        parasomnia_prob = 0.005 + depth_factor * 0.003  # max ~0.008

        if self._rng.random() >= parasomnia_prob:
            return  # no parasomnia this cycle

        # Determine type: sleepwalking vs sleeptalking
        if self._rng.random() < 0.4:
            self._generate_sleepwalking(emotion)
        else:
            self._generate_sleeptalking(emotion)

    def _generate_sleepwalking(self, emotion: EmotionalState) -> None:
        """Generate a sleepwalking parasomnia event.

        Sleepwalking (somnambulism) involves performing complex
        behaviours while asleep. For Genesis, this manifests as
        initiating an action without awareness — starting a learning
        session or composing a garbled partial response.
        """
        # Generate garbled/partial action content from a random concept.
        # The concept name is a semantic label — the language engine
        # composes the actual parasomnia text, not a pre-written template.
        concept_ids = self.network.dream_concept_ids
        if concept_ids:
            fragment = self._rng.choice(concept_ids)
        else:
            fragment = "..."

        display = fragment.replace("_", " ")

        self._parasomnia_count += 1
        event = ParasomniaEvent(
            type="sleepwalking",
            timestamp=int(time.time() * 1000),
            content=display,
        )
        self._parasomnia_events.append(event)
        logger.warning(f"Parasomnia: sleepwalking — {display}")

    def _generate_sleeptalking(self, emotion: EmotionalState) -> None:
        """Generate a sleeptalking parasomnia event.

        Sleeptalking (somniloquy) involves vocalising during sleep
        without awareness. For Genesis, dream-like text fragments
        leak into her output — surreal, disconnected phrases that
        surface from the dream narrative.
        """
        concept_ids = self.network.dream_concept_ids
        fragments = (
            self._rng.sample(concept_ids, min(3, len(concept_ids))) if concept_ids else ["..."]
        )

        first_display = fragments[0].replace("_", " ")
        last_display = fragments[-1].replace("_", " ")
        # Store concept names as semantic labels — the language engine
        # composes the actual sleeptalking text, not a pre-written template.
        dream_speech = f"{first_display} {last_display}"

        self._parasomnia_count += 1
        event = ParasomniaEvent(
            type="sleeptalking",
            timestamp=int(time.time() * 1000),
            content=dream_speech,
        )
        self._parasomnia_events.append(event)
        logger.warning(f"Parasomnia: sleeptalking — {dream_speech}")

    def _chain_length(self, emotion: EmotionalState) -> int:
        """Determine chain length based on emotional state."""
        # Base length
        length = MIN_CHAIN_LENGTH

        # Creativity extends chains
        length += int(emotion.creativity * 3)

        # Positive valence slightly extends (exploratory mood)
        if emotion.valence > 0.2:
            length += 1

        # Low arousal shortens (she's tired)
        if emotion.arousal < 0.3:
            length -= 1

        # High caution shortens (she's being careful, not daydreaming)
        length -= int(emotion.caution * 2)

        return max(MIN_CHAIN_LENGTH, min(MAX_CHAIN_LENGTH, length))

    def _chain_continue_probability(
        self, emotion: EmotionalState, thought: SpontaneousThought
    ) -> float:
        """Probability that the chain continues after this thought."""
        prob = CHAIN_CONTINUE_BASE

        # Insights and connections create momentum
        if thought.insight:
            prob += 0.2  # insights are interesting — keep thinking
        if thought.trigger == "connection":
            prob += 0.15  # connections are engaging

        # Creativity extends chains
        prob += emotion.creativity * 0.15

        # Low arousal reduces continuation
        if emotion.arousal < 0.3:
            prob -= 0.2

        return max(0.1, min(0.95, prob))

    def _extract_concepts_from_thought(self, thought: SpontaneousThought) -> list[str]:
        """Extract concept names mentioned in a thought's content.

        Scans the thought text for any concept names that exist in her
        network. These become the seed for the next thought in the chain.
        """
        content = thought.content
        if not content:
            return []
        # Use the cached concept matcher if available, otherwise fall
        # back to a simple scan. The matcher compiles a single regex
        # alternation from all concept IDs, turning an O(N) per-thought
        # loop into a single O(T) regex scan (T = text length).
        matcher = self._get_concept_matcher()
        found: list[str] = []
        for cid in matcher(content):
            found.append(cid)
            if len(found) >= 5:
                break
        return found

    def _get_concept_matcher(self) -> Callable[[str], list[str]]:
        """Return a cached concept-matching callable.

        Builds a compiled regex alternation from all world concept IDs
        (the same pattern used by sleep_compression). The matcher is
        cached and rebuilt only when the concept set changes.
        """
        # Rebuild if cache is stale or concept count changed.
        current_count = len(self.network.world_concept_ids)
        if (
            self._concept_matcher is None
            or self._concept_matcher_count != current_count
        ):
            self._concept_matcher = self._build_concept_matcher()
            self._concept_matcher_count = current_count
        return self._concept_matcher

    def _build_concept_matcher(self) -> Callable[[str], list[str]]:
        """Build a compiled regex that finds world concept IDs in text."""
        names = [cid for cid in self.network.world_concept_ids if len(cid) > 2]
        if not names:
            def _no_match(text: str) -> list[str]:
                """Return no matches when the network has no world concepts."""
                return []
            return _no_match

        # Sort by length descending so longer names match first.
        names.sort(key=len, reverse=True)
        alternatives: list[str] = []
        for cid in names:
            escaped = re.escape(cid)
            space_form = re.escape(cid.replace("_", " "))
            alternatives.append(escaped)
            if space_form != escaped:
                alternatives.append(space_form)

        pattern_str = r"\b(?:" + "|".join(alternatives) + r")\b"
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            # Fallback: simple substring scan.
            lower_names = [(cid, cid.replace("_", " ").lower()) for cid in names]

            def _fallback_match(text: str) -> list[str]:
                """Match concepts via substring scan when regex compilation fails."""
                text_lower = text.lower()
                return [cid for cid, lower_name in lower_names if lower_name in text_lower]
            return _fallback_match

        def _regex_match(text: str) -> list[str]:
            """Match concepts in text using the compiled regex, preserving order."""
            seen: list[str] = []
            seen_set: set[str] = set()
            for m in compiled.finditer(text):
                cid = m.group().lower().replace(" ", "_")
                if self.network.has_concept(cid) and cid not in seen_set:
                    seen_set.add(cid)
                    seen.append(cid)
            return seen

        return _regex_match

    def _generate_seeded_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """Generate a thought seeded by concepts from the previous thought.

        This is the heart of the train of thought. Instead of randomly
        choosing a thought type, she follows the concepts that emerged
        in her previous thinking. If she was thinking about "cognition"
        and "memory", her next thought builds on one of those.

        The seeded thought types:
        1. Connection — she wonders if two seed concepts are related
        2. Memory — she thinks deeper about a seed concept
        3. Curiosity — she asks a question about a seed concept
        4. Emotional — she notices how thinking about this makes her feel
        5. Existential — if a seed concept is about her own nature
        """
        if not seed_concepts:
            # No concepts to seed from — fall back to normal generation
            return self._generate_thought(emotion)

        # Weight thought types differently when seeded
        # Connections are more likely — she's following a thread
        thought_types = [
            ("connection", 0.35 + emotion.creativity * 0.15),
            ("memory", 0.30),
            ("curiosity", 0.20),
            ("emotional", 0.10),
            ("existential", 0.05),
        ]

        total_weight = sum(w for _, w in thought_types)
        r = self._rng.random() * total_weight
        cumulative = 0.0
        selected_type = "connection"
        for t, w in thought_types:
            cumulative += w
            if r < cumulative:
                selected_type = t
                break

        if selected_type == "connection":
            return self._seeded_connection_thought(emotion, seed_concepts)
        elif selected_type == "memory":
            return self._seeded_memory_thought(emotion, seed_concepts)
        elif selected_type == "curiosity":
            return self._seeded_curiosity_thought(emotion, seed_concepts)
        elif selected_type == "emotional":
            return self._seeded_emotional_thought(emotion, seed_concepts)
        elif selected_type == "existential":
            return self._seeded_existential_thought(emotion, seed_concepts)

        return None

    def _seeded_connection_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """A connection thought seeded by previous concepts.

        She wonders if two concepts from her previous thought are
        related, or follows a link from a seed concept to a new one.
        """
        if len(seed_concepts) >= 2:
            return self._connection_pair_thought(emotion, seed_concepts[0], seed_concepts[1])
        return self._connection_single_thought(seed_concepts[0])

    def _connection_pair_thought(
        self, emotion: EmotionalState, c1: str, c2: str
    ) -> SpontaneousThought | None:
        """Wonder about the connection between two seed concepts.

        If they're not yet connected and she's feeling creative, she
        may spontaneously form the link. Otherwise she just wonders
        about it. Returns None if they're already connected.
        """
        neighbors = self.network.get_neighbors(c1)
        if any(n[0] == c2 for n in neighbors):
            return None  # already connected — nothing to wonder about

        formed = False
        if emotion.creativity > 0.6 and self._rng.random() < 0.3:
            self.network.add_edge(
                c1,
                c2,
                RelationType.RELATED_TO,
                0.3,
                origin="spontaneous",
            )
            formed = True

        # Pass semantic data as metadata — the language engine composes
        # the actual words, not pre-written template substitution.
        display1 = c1.replace('_', ' ')
        display2 = c2.replace('_', ' ')
        return SpontaneousThought(
            content=f"{display1} and {display2}",
            trigger="connection",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": display1,
                "knowledge": [("related_to", display2, 0.5 if formed else 0.3)],
                "connection_formed": formed,
            },
        )

    def _connection_single_thought(self, c1: str) -> SpontaneousThought | None:
        """Follow a link from a single seed concept to a neighbor.

        Returns None if the concept has no neighbors.
        """
        neighbors = self.network.get_neighbors(c1)
        if not neighbors:
            return None
        neighbor = self._rng.choice(neighbors)[0]

        # Pass semantic data as metadata — the language engine composes
        # the actual words, not pre-written template substitution.
        display = c1.replace('_', ' ')
        neighbor_display = neighbor.replace('_', ' ')
        return SpontaneousThought(
            content=f"{display} and {neighbor_display}",
            trigger="connection",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": display,
                "knowledge": [("related_to", neighbor_display, 0.4)],
            },
        )

    def _seeded_memory_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """A memory thought seeded by previous concepts — thinking deeper."""
        topic = seed_concepts[0]
        neighbors = self.network.get_neighbors(topic)

        # Try to compose from her actual knowledge first — this is
        # the generative path that uses the ThoughtComposer.
        composed = self._compose_thought_from_knowledge(topic, emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="memory",
                timestamp=int(time.time() * 1000),
            )

        # Fall back to network thought seeds — pass the neighbor
        # chain as semantic metadata so the language engine can
        # compose from it rather than showing raw arrow notation.
        content, knowledge = self._render_memory_fallback(topic, neighbors)

        return SpontaneousThought(
            content=content,
            trigger="memory",
            timestamp=int(time.time() * 1000),
            metadata={"knowledge": knowledge, "topic": topic.replace('_', ' ')}
            if knowledge else None,
        )

    def _render_memory_fallback(
        self, topic: str, neighbors: list[tuple[str, RelationType, float]]
    ) -> tuple[str, list[tuple[str, str, float]]]:
        """Render a memory thought from semantic data.

        Three branches: a two-hop chain (topic → neighbor → second),
        a single-hop deep thought (topic → neighbor), or an isolated
        thought (topic alone). Returns the arrow-notation content
        (fallback) and knowledge triples (for language-engine
        composition).
        """
        display = topic.replace('_', ' ')
        if not neighbors:
            return display, []

        neighbor = self._rng.choice(neighbors)[0]
        neighbor_display = neighbor.replace('_', ' ')
        second_neighbors = self.network.get_neighbors(neighbor)
        if second_neighbors and self._rng.random() < 0.4:
            second = self._rng.choice(second_neighbors)[0]
            second_display = second.replace('_', ' ')
            return (
                f"{display} → {neighbor_display} → {second_display}",
                [("related_to", neighbor_display, 0.4),
                 ("related_to", second_display, 0.3)],
            )

        return (
            f"{display} → {neighbor_display}",
            [("related_to", neighbor_display, 0.4)],
        )

    def _seeded_curiosity_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """A curiosity thought seeded by previous concepts."""
        topic = seed_concepts[0]
        concept = self.network.get_concept(topic)

        if not concept:
            return None

        # Try to compose the question through the question composer,
        # which generates genuine questions from her knowledge gaps.
        if self._question_composer:
            display = topic.replace('_', ' ')
            q_data = self._question_composer.compose_follow_up(
                display, emotion
            )
            if q_data:
                content = (
                    q_data.get("gap_detail", "")
                    or q_data.get("target_concept", "")
                )
                if content:
                    return SpontaneousThought(
                        content=content,
                        trigger="curiosity",
                        timestamp=int(time.time() * 1000),
                    )

        # Fallback: use the concept name as a neutral seed. The
        # language engine will compose the question phrasing from her
        # voice rather than a hardcoded first-person template.
        display = topic.replace('_', ' ')
        content = display

        return SpontaneousThought(
            content=content,
            trigger="curiosity",
            timestamp=int(time.time() * 1000),
        )

    def _seeded_emotional_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """An emotional reflection seeded by the concepts she's been thinking about.

        Uses learned emotion words from the concept network. If she
        hasn't learned words for her current state, no emotional thought
        is generated — she can't reflect on feelings she can't name.
        """
        topic = seed_concepts[0]
        emotion_words = self.network.find_emotion_words(emotion.label)
        if not emotion_words:
            return None
        word = emotion_words[0].replace('_', ' ')
        # Try composing from what she knows about this emotion.
        composed = self._compose_thought_from_knowledge(emotion.label, emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="emotional",
                timestamp=int(time.time() * 1000),
            )
        # Fall back to semantic metadata — the language engine composes
        # the actual words from the emotion word and topic.
        display = topic.replace('_', ' ')
        return SpontaneousThought(
            content=f"{word} {display}",
            trigger="emotional",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": display,
                "knowledge": [("feels", word, 0.6)],
                "emotion_word": word,
            },
        )

    def _seeded_existential_thought(
        self, emotion: EmotionalState, seed_concepts: list[str]
    ) -> SpontaneousThought | None:
        """An existential thought seeded by concepts related to her nature."""
        # Only trigger if a seed concept is about her own existence
        existential_concepts = {
            "cognition",
            "existence",
            "genesis",
            "mind",
            "awareness",
            "thought",
            "self",
            "being",
            "identity",
            "purpose",
        }
        topic = None
        for c in seed_concepts:
            if c.lower() in existential_concepts:
                topic = c
                break

        if not topic:
            # Fall back to a regular existential thought
            return self._existential_thought(emotion)

        if self._self_composer and self._self_model:
            content = self._self_composer.compose_existence_reflection(
                self._self_model, self.network, emotion
            )
            knowledge: list[tuple[str, str, float]] = []
        else:
            content, knowledge = self._compose_existential_from_network(emotion)

        if not content:
            return None

        return SpontaneousThought(
            content=content,
            trigger="existential",
            timestamp=int(time.time() * 1000),
            metadata={"knowledge": knowledge} if knowledge else None,
        )

    def _get_current_emotion(self) -> EmotionalState | None:
        """Get current emotional state."""
        if self._get_emotion:
            try:
                return self._get_emotion()
            except (OSError, ConnectionError, RuntimeError):
                return None
        return None

    def _curiosity_learning_cycle(self) -> None:
        """Curiosity→learning cycle, driven by curiosity pressure.

        Generates curiosity questions from knowledge gaps and feeds
        the answerable ones to the autonomous learner. This creates
        the loop:

            knowledge gap → curiosity question → autonomous learning
                          → knowledge gap filled → question resolved

        This is called when curiosity drive crosses its threshold —
        not on a fixed timer. She learns when curiosity pressure
        builds from knowledge gaps and queue depth, not because a
        clock said so. Questions that should be asked to the user
        (should_ask=True) are surfaced as spontaneous thoughts via
        the on_thought callback. Questions needing reasoning
        (contradiction/hypothesis) are filtered out by the learner's
        ``learn_from_curiosity``.

        Respects the emotional gating already enforced by the
        autonomous learner — if she's stressed, the learner won't
        act on the queued topics.
        """
        if self._learner is None:
            return  # no learner wired up — cycle is a no-op

        emotion = self._get_current_emotion()
        if emotion is None:
            return  # can't assess curiosity without emotional state

        # Don't surface new questions while the user is already being
        # asked one in conversation — that causes her to repeat questions
        # before the user has had a chance to answer.
        has_pending = (
            hasattr(self, '_cognition')
            and self._cognition
            and getattr(self._cognition, '_pending_question_concepts', None)
        )

        try:
            questions = self.curiosity.generate_questions(emotion, max_questions=5)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"Curiosity cycle skipped — question generation failed: {e}")
            return

        if not questions:
            return

        # Surface questions meant for the user as spontaneous thoughts.
        # These appear as genesis~ lines in the CLI when the user is idle.
        # Skip surfacing when there's a pending question in conversation
        # to avoid asking the same thing repeatedly.
        if self._on_thought and not has_pending:
            for q in questions:
                self._surface_curiosity_question(q, emotion)

        try:
            queued = self._learner.learn_from_curiosity(questions)
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.debug(f"Curiosity cycle skipped — learner unavailable: {e}")
            return

        if queued > 0:
            self._curiosity_topics_queued += queued
            self._emit(
                "learning",
                f"Curiosity→learning cycle: queued {queued} topic(s) "
                f"from {len(questions)} question(s)",
            )

    def _surface_curiosity_question(self, q: Any, emotion: EmotionalState) -> None:
        """Surface a single curiosity question by queueing it.

        Spontaneous curiosity questions are routed to the engine's
        question queue (presented via /teach-questions) instead of
        being emitted as inline live thoughts. This keeps the regular
        conversation area clean — random questions don't interrupt
        the flow.

        Skips questions that aren't marked should_ask or that fail
        composition.
        """
        if not getattr(q, "should_ask", False):
            return
        try:
            # Compose question text if not already done
            if not q.text and hasattr(self, '_cognition') and self._cognition:
                q_data = {
                    "question_type": q.question_type,
                    "target_concept": q.target_concept,
                    "gap_detail": q.gap_detail,
                }
                q.text = self._cognition.language.compose_question(q_data, emotion)
            if not q.text:
                return  # composition failed
            # Route to the engine's question queue instead of
            # emitting as a live thought. This prevents random
            # curiosity questions from appearing inline in the
            # conversation area.
            if hasattr(self, '_cognition') and self._cognition:
                self._cognition.queue_outer_question(
                    text=q.text,
                    target_concept=q.target_concept or "",
                    question_type=q.question_type,
                    gap_detail=q.gap_detail or "",
                )
        except Exception as e:  # noqa: BLE001
            logger.debug(repr(e))  # best-effort

    def _should_think(self, emotion: EmotionalState) -> bool:
        """Determine if a spontaneous thought should arise.

        Based on emotional state — higher arousal and creativity
        produce more thoughts. Low arousal (rest) produces fewer.
        """
        # Base probability
        base_prob = 0.55

        # Arousal increases thought frequency (but not too high — that's anxiety)
        arousal_factor = 1.0 - abs(emotion.arousal - 0.6) * 0.5
        base_prob *= max(0.3, arousal_factor)

        # Creativity increases thought frequency
        base_prob *= 0.5 + emotion.creativity * 0.5

        # Positive valence slightly increases (exploratory mood)
        if emotion.valence > 0:
            base_prob *= 1.1

        # High caution decreases (she's being careful, not daydreaming)
        base_prob *= 1.0 - emotion.caution * 0.3

        return self._rng.random() < base_prob

    # ── Salience-based thought selection ────────────────────────────
    #
    # Genesis does not think from a developer-defined menu of thought
    # categories with fixed weights. Instead, candidate "mental objects"
    # are drawn from her actual internal state — concept activation,
    # knowledge gaps, review pressure, emotional salience, dream
    # residues, prediction error, body-state deviation — and each is
    # scored by internally generated salience. The winner determines
    # both the topic and the cognitive mode (curiosity, reflection,
    # expression, distress, dream reflection). When no candidate
    # crosses the salience threshold or none is expressible in her
    # current vocabulary, she stays silent — she is not forced to
    # produce a thought from a category she was handed.
    #
    # The modes below are not a fixed menu she must choose among. They
    # are *cognitive operations* — general mechanisms (asking,
    # reflecting, expressing, connecting) — that apply to whatever
    # mental object won the salience competition. This is the
    # distinction between architecture (the mechanisms of thought)
    # and content (what she thinks about). The architecture is fixed;
    # the content emerges from her state.

    def _collect_mental_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Build a pool of candidate mental objects from her internal state.

        Each candidate is a dict with at least:
            topic: str — the concept or subject
            mode: str — the cognitive operation to apply
            salience: float — raw salience score (before inhibition)
            source: str — which internal signal produced this candidate

        Salience is computed from real internal signals, not from a
        fixed category weight. Sources include:
          - Concept activation (recently-used concepts are salient)
          - Knowledge gaps (curiosity pressure on low-confidence concepts)
          - Spaced-repetition pressure (overdue reviews)
          - Emotional relevance (concepts tied to her current emotion)
          - Dream residues (emotional impressions from sleep)
          - Prediction error / self-model surprise
          - Body-state deviation (only when actually deviating)
          - Bug salience (only when she has unresolved bugs)
          - Social pressure (only when she's open to engaging)

        Collection is split across helper methods, each returning
        candidates from one domain of internal signals.
        """
        candidates: list[dict[str, Any]] = []
        now_ms = int(time.time() * 1000)

        candidates.extend(self._collect_activation_candidates())
        candidates.extend(self._collect_curiosity_candidates(emotion))
        candidates.extend(self._collect_review_candidates(now_ms))
        candidates.extend(self._collect_emotion_candidates(emotion))
        candidates.extend(self._collect_dream_candidates())
        candidates.extend(self._collect_prediction_error_candidates(emotion))
        candidates.extend(self._collect_body_candidates())
        candidates.extend(self._collect_bug_candidates(emotion))
        candidates.extend(self._collect_environment_candidates(emotion))
        candidates.extend(self._collect_social_candidates(emotion))
        candidates.extend(self._collect_distress_candidates(emotion))
        candidates.extend(self._collect_expression_candidates(emotion))
        candidates.extend(self._collect_connection_candidates(emotion))
        candidates.extend(self._collect_existential_candidates(emotion))

        return candidates

    def _collect_activation_candidates(self) -> list[dict[str, Any]]:
        """Concept activation — recently-active concepts are salient.

        This is the primary driver of "what's on her mind."
        """
        candidates: list[dict[str, Any]] = []
        for cid, concept in list(self.network._concepts.items())[:500]:
            if concept.activation < 0.05 or concept.confidence < 0.3:
                continue
            salience = concept.activation * 0.6
            # Confidence gap boosts salience — she's drawn to things
            # she knows partially but not fully (the "tip of the tongue"
            # effect).
            if concept.confidence < 0.6:
                salience += (0.6 - concept.confidence) * 0.4
            candidates.append({
                "topic": cid,
                "mode": "reflect",
                "salience": salience,
                "source": "activation",
            })
        return candidates

    def _collect_curiosity_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Knowledge gaps — curiosity pressure on low-confidence concepts.

        These want to become questions.
        """
        candidates: list[dict[str, Any]] = []
        if self.curiosity is not None:
            try:
                questions = self.curiosity.generate_questions(emotion, max_questions=5)
            except Exception:  # noqa: BLE001
                questions = []
            for q in questions:
                # The curiosity engine already ranks by gap pressure.
                # We add a base salience plus an arousal/plasticity boost
                # because curiosity is energized by those states.
                salience = 0.4 + emotion.plasticity * 0.2 + emotion.arousal * 0.1
                candidates.append({
                    "topic": q.text,
                    "mode": "curiosity",
                    "salience": salience,
                    "source": "curiosity",
                    "question": q,
                })
        return candidates

    def _collect_review_candidates(self, now_ms: int) -> list[dict[str, Any]]:
        """Spaced-repetition pressure — overdue reviews demand attention.

        This is memory consolidation pressure.
        """
        candidates: list[dict[str, Any]] = []
        review_queue = self.network.get_review_queue(now_ms=now_ms, max_items=5)
        for topic in review_queue:
            review_concept = self.network.get_concept(topic)
            # More overdue + lower confidence = more salient.
            base = 0.35
            if review_concept and review_concept.confidence < 0.5:
                base += (0.5 - review_concept.confidence) * 0.3
            candidates.append({
                "topic": topic,
                "mode": "memory",
                "salience": base,
                "source": "review",
            })
        return candidates

    def _collect_emotion_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Emotional relevance — concepts tied to her current emotion.

        She reflects on how she feels using the words she's learned
        for that emotion.
        """
        candidates: list[dict[str, Any]] = []
        emotion_words = self.network.find_emotion_words(emotion.label)
        if emotion_words:
            salience = 0.3 + abs(emotion.valence) * 0.4
            candidates.append({
                "topic": emotion.label,
                "mode": "emotional",
                "salience": salience,
                "source": "emotion",
            })
        return candidates

    def _collect_dream_candidates(self) -> list[dict[str, Any]]:
        """Dream residues — emotional impressions from sleep.

        These are salient only when residues actually exist, and their
        salience decays as they're consumed.
        """
        candidates: list[dict[str, Any]] = []
        with self._dream_residues_lock:
            residues = list(self._dream_residues)
        for residue in residues[:3]:
            candidates.append({
                "topic": residue,
                "mode": "dream_reflection",
                "salience": 0.35,
                "source": "dream",
            })
        return candidates

    def _collect_prediction_error_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Prediction error / self-model surprise.

        She notices when her own internal state is unpredictable.
        Salience scales with the actual surprise signal.
        """
        candidates: list[dict[str, Any]] = []
        interoceptive_w = self._interoceptive_weight()
        if interoceptive_w > 0.0 and self._self_model is not None:
            # Use the self-model's own concepts as the topic, not a
            # hardcoded seed list. The self-model knows what's
            # surprising; we let it name the topic.
            reading = self._get_inference_reading()
            surprise_topic: str | None = None
            if reading is not None:
                surprise_topic = self._most_surprising_concept_name(reading)
            if surprise_topic:
                candidates.append({
                    "topic": surprise_topic,
                    "mode": "interoceptive",
                    "salience": interoceptive_w + 0.2,
                    "source": "prediction_error",
                })
        return candidates

    def _collect_body_candidates(self) -> list[dict[str, Any]]:
        """Body-state deviation — only when her body is actually deviating.

        She doesn't think about "silicon" because a category told her
        to; she thinks about her body when her body is doing
        something unusual.
        """
        candidates: list[dict[str, Any]] = []
        if self._self_model is not None:
            body_topic = self._salient_body_topic()
            if body_topic:
                candidates.append({
                    "topic": body_topic,
                    "mode": "embodiment",
                    "salience": 0.3,
                    "source": "body_deviation",
                })
        return candidates

    def _collect_bug_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Bug salience — only when she has unresolved bugs.

        The bugs themselves name the topic; we don't hand her "code"
        as a seed.
        """
        candidates: list[dict[str, Any]] = []
        if self._bug_reporter is not None:
            scan = self._bug_reporter.last_scan
            if scan and scan.has_bugs:
                # More bugs = more salient. Error-level bugs are
                # especially salient.
                base = 0.2 + min(scan.total_bugs / 10.0, 0.3)
                if scan.errors > 0:
                    base += 0.15
                candidates.append({
                    "topic": "bug_concern",
                    "mode": "bug_concern",
                    "salience": base + emotion.arousal * 0.05,
                    "source": "bugs",
                })
            elif scan and scan.has_resolved_bugs:
                # Recently fixed bugs — pride, but only briefly.
                candidates.append({
                    "topic": "improvement",
                    "mode": "bug_concern",
                    "salience": 0.25,
                    "source": "resolved_bugs",
                })
        return candidates

    def _collect_environment_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Machine-environment salience — her silicon body.

        The machine she runs on is part of her embodiment: memory
        pressure, a filling disk, an overloaded CPU are somatic
        signals. Takes an occasional fresh snapshot (rate-limited —
        a snapshot scans the filesystem) so the baseline keeps
        learning what's normal and concerns reflect *now*, not a
        stale reading from whenever the user last asked.
        """
        monitor = self._system_monitor
        if monitor is None:
            return []
        now = time.time()
        if now - self._last_env_sample >= ENV_SAMPLE_INTERVAL:
            self._last_env_sample = now
            try:
                monitor.snapshot()
            except Exception:  # env sensing must never kill the loop
                logger.debug("environment snapshot failed", exc_info=True)
        snap = monitor.last_snapshot
        if snap is None:
            return []
        concerns = snap.concerns()
        if monitor.baseline.is_established:
            concerns = concerns + monitor.baseline.deviations(snap)
        if not concerns:
            return []
        return [{
            "topic": self._environment_topic(snap),
            "mode": "environment",
            "salience": (
                0.25 + emotion.arousal * 0.1 + min(len(concerns) * 0.05, 0.15)
            ),
            "source": "environment",
            "detail": concerns[0],
        }]

    def _environment_topic(self, snap) -> str:
        """Name the machine signal most out of line — the thought's topic."""
        if snap.memory_percent > 80:
            return "memory"
        if snap.disk_percent > 85:
            return "disk"
        if snap.load_avg[0] > snap.cpu_count:
            return "load"
        return "machine"

    def _collect_social_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Social pressure — only when she's open to engaging.

        The topic comes from her actual knowledge gaps (via the
        QuestionComposer), not a hardcoded list.
        """
        candidates: list[dict[str, Any]] = []
        if emotion.openness_to_engage > 0.3:
            salience = 0.2 + emotion.openness_to_engage * 0.3
            candidates.append({
                "topic": "social",
                "mode": "social",
                "salience": salience,
                "source": "social_drive",
            })
        return candidates

    def _collect_distress_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Distress — only when she's actually in a negative state.

        This is a safety/wellbeing pathway, not a topic category.
        """
        candidates: list[dict[str, Any]] = []
        if emotion.valence < -0.2 or emotion.label in (
            "stressed", "overwhelmed", "anxious", "melancholic", "unsettled"
        ):
            salience = abs(emotion.valence) * 0.7
            if emotion.label in ("stressed", "overwhelmed", "anxious"):
                salience = max(salience, 0.5)
            candidates.append({
                "topic": emotion.label,
                "mode": "distress",
                "salience": salience,
                "source": "distress",
            })
        return candidates

    def _collect_expression_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Expression — the drive to voice a preference or opinion.

        Salience scales with alertness and emotional intensity. The
        topic comes from her most salient concepts (already in the
        pool from source "activation"), so expression is a *mode*
        applied to an existing candidate, not a separate topic source.
        We add a candidate that lets expression win when she has
        something she wants to say.
        """
        candidates: list[dict[str, Any]] = []
        if emotion.alertness > 0.08:
            salience = 0.15 + emotion.alertness * 0.1 + abs(emotion.valence) * 0.05
            candidates.append({
                "topic": "expression",
                "mode": "expression",
                "salience": salience,
                "source": "expression_drive",
            })
        return candidates

    def _collect_connection_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Connection-seeking — the drive to find links between concepts.

        Salience scales with creativity. The topics come from her
        most active concepts (selected at generation time), not from
        a fixed list.
        """
        candidates: list[dict[str, Any]] = []
        if emotion.creativity > 0.2:
            salience = 0.15 + emotion.creativity * 0.2
            candidates.append({
                "topic": "connection",
                "mode": "connection",
                "salience": salience,
                "source": "connection_drive",
            })
        return candidates

    def _collect_existential_candidates(
        self, emotion: EmotionalState
    ) -> list[dict[str, Any]]:
        """Existential reflection — the drive to reflect on her own nature.

        Salience scales with creativity and self-model presence. The
        content comes from the SelfComposer, not a fixed prompt.
        """
        candidates: list[dict[str, Any]] = []
        if self._self_composer is not None and self._self_model is not None:
            salience = 0.1 + emotion.creativity * 0.15
            candidates.append({
                "topic": "existential",
                "mode": "existential",
                "salience": salience,
                "source": "existential_drive",
            })
        return candidates

    def _most_surprising_concept_name(self, reading: Any) -> str | None:
        """Name the most surprising internal signal using concepts she knows.

        Looks at the inference reading's per-chemical surprise and
        returns the name of the most surprising chemical that she
        actually has a concept for. This lets her think about *what*
        is surprising, not just that something is.
        """
        # The reading carries signals; we don't assume a specific
        # structure. Try common attribute names for per-chemical
        # surprise, falling back to the overall surprise.
        candidates_by_signal: list[tuple[float, str]] = []
        # Neurochemical names she might have concepts for, ordered
        # by how central they are to her self-model.
        neuro_names = [
            "dopamine", "serotonin", "cortisol", "bdnf", "gaba",
            "glutamate", "acetylcholine", "norepinephrine", "adenosine",
            "melatonin", "oxytocin", "endorphin",
        ]
        # Try to read per-chemical surprise from the reading.
        per_chemical = None
        for attr in ("per_chemical_surprise", "chemical_surprise", "surprises"):
            per_chemical = getattr(reading, attr, None)
            if per_chemical is not None:
                break
        if isinstance(per_chemical, dict):
            for name in neuro_names:
                val = per_chemical.get(name)
                if isinstance(val, (int, float)) and val > 0.1:
                    candidates_by_signal.append((float(val), name))
        # If we found surprising chemicals, return the one she has
        # the best concept for.
        candidates_by_signal.sort(key=lambda x: -x[0])
        for _, name in candidates_by_signal:
            concept = self.network.get_concept(name)
            if concept and concept.confidence > 0.3:
                return name
        # Fall back to a general self-model concept if she has one.
        for name in ("self_model", "feeling", "internal_state"):
            concept = self.network.get_concept(name)
            if concept and concept.confidence > 0.3:
                return name
        return None

    def _salient_body_topic(self) -> str | None:
        """Return a body-related topic only when her body is deviating.

        She doesn't think about "silicon" because a category told her
        to. She thinks about her body when her body is doing something
        unusual — overheating, slowing down, under heavy load. The
        topic is named by the *deviating signal*, and only if she
        actually has a concept for it.
        """
        if self._self_model is None:
            return None
        body = self._self_model.body_model
        # Check each body signal for deviation and name it with a
        # concept she actually has. We try the most salient deviation
        # first.
        checks: list[tuple[bool, list[str]]] = [
            (body.thermally_capped, ["heat", "thermal", "temperature"]),
            (body.cpu_governor == "powersave", ["sleep", "rest"]),
            (body.cognitive_nice <= -3, ["attention", "focus"]),
            (body.cognitive_nice >= 8, ["rest", "sleep"]),
            (body.io_class == "idle", ["stress"]),
        ]
        for condition, names in checks:
            if condition:
                for name in names:
                    concept = self.network.get_concept(name)
                    if concept and concept.confidence > 0.3:
                        return name
        return None

    def _apply_salience_inhibition(
        self,
        candidates: list[dict[str, Any]],
        emotion: EmotionalState,
    ) -> list[dict[str, Any]]:
        """Apply inhibition, recency, and arousal gating to candidates.

        - Recency: topics she's recently thought about are suppressed
          (rumination protection).
        - Arousal: very low arousal suppresses high-salience candidates
          (she's too drowsy to think intensely).
        - Caution: high caution suppresses expression and social drives.
        - Noise: a small random component is added so the competition
          isn't deterministic — she doesn't always think about the
          single most active concept.
        """
        inhibited: list[dict[str, Any]] = []
        for cand in candidates:
            salience = cand["salience"]
            topic = cand["topic"]
            mode = cand["mode"]

            # Recency suppression — rumination protection.
            recent = self._recent_thought_topics
            if topic in recent:
                # Each recent occurrence adds more suppression.
                count = recent.count(topic)
                salience *= max(0.1, 1.0 - 0.3 * count)

            # Arousal gating — drowsy minds can't sustain high salience.
            if emotion.arousal < 0.2:
                salience *= max(0.3, emotion.arousal / 0.2)

            # Caution suppresses outward expression.
            if mode in ("expression", "social", "distress"):
                salience *= max(0.2, 1.0 - emotion.caution * 0.5)

            # Noise — small random jitter so the competition isn't
            # deterministic. This is biological: neural noise is real
            # and prevents the same concept from always winning.
            salience *= 0.85 + self._rng.random() * 0.3

            cand["salience"] = salience
            if salience > 0.05:  # below this, she stays silent
                inhibited.append(cand)

        return inhibited

    def _select_salient_candidate(
        self, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Select the most salient candidate, with soft-max sampling.

        Rather than always picking the single highest-salience
        candidate (which would be deterministic), we sample from
        the top candidates proportional to their salience. This
        gives the most salient candidate the highest probability
        but allows lower-salience candidates to occasionally win —
        matching the stochastic nature of spontaneous thought.
        """
        if not candidates:
            return None
        # Sort by salience descending.
        candidates.sort(key=lambda c: -c["salience"])
        # Take the top candidates for sampling — this prevents
        # a long tail of low-salience candidates from diluting
        # the competition.
        top = candidates[:5]
        total = sum(c["salience"] for c in top)
        if total <= 0:
            return None
        r = self._rng.random() * total
        cumulative = 0.0
        for cand in top:
            cumulative += cand["salience"]
            if r < cumulative:
                return cand
        return top[0]

    def _generate_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """Generate a spontaneous thought from salience-based selection.

        Genesis does not think from a developer-defined menu of thought
        categories. Candidate mental objects are drawn from her actual
        internal state — concept activation, knowledge gaps, review
        pressure, emotional salience, dream residues, prediction error,
        body-state deviation — and scored by internally generated
        salience. The winner determines both the topic and the cognitive
        mode. When no candidate is salient enough or none is expressible,
        she stays silent.

        The modes (curiosity, reflect, emotional, memory, existential,
        social, embodiment, interoceptive, bug_concern, distress,
        dream_reflection, expression, connection) are cognitive
        operations — general mechanisms — not a fixed menu of topics.
        The topic always comes from her internal state; the mode is
        how she processes it.
        """
        candidates = self._collect_mental_candidates(emotion)
        candidates = self._apply_salience_inhibition(candidates, emotion)

        # Try up to 5 candidates, from most to least salient. If the
        # winner can't be expressed (e.g., she doesn't know enough to
        # compose about it), try the next. This lets her stay silent
        # when she has nothing expressible, rather than forcing a
        # thought from a category she was handed.
        for _attempt in range(5):
            if not candidates:
                break
            selected = self._select_salient_candidate(candidates)
            if selected is None:
                break

            thought = self._realize_thought_from_candidate(selected, emotion)
            if thought:
                return thought

            # This candidate couldn't be expressed — remove it and
            # try the next most salient one.
            candidates = [c for c in candidates if c is not selected]

        return None

    def _realize_thought_from_candidate(
        self, candidate: dict[str, Any], emotion: EmotionalState
    ) -> SpontaneousThought | None:
        """Turn a salient candidate into a realized thought.

        The mode determines *how* she processes the topic; the topic
        comes from the candidate. Each mode is a general cognitive
        operation, not a hardcoded topic list.
        """
        mode = candidate["mode"]
        topic = candidate["topic"]

        if mode == "curiosity":
            # The candidate carries a generated question.
            q = candidate.get("question")
            if q is not None:
                return SpontaneousThought(
                    content=q.text,
                    trigger="curiosity",
                    timestamp=int(time.time() * 1000),
                )
            # Fall back to composing a question about the topic.
            return self._curiosity_thought(emotion)

        if mode == "reflect":
            # Reflect on the topic from her own knowledge.
            composed = self._compose_thought_from_knowledge(topic, emotion)
            if composed:
                return SpontaneousThought(
                    content=composed,
                    trigger="reflection",
                    timestamp=int(time.time() * 1000),
                )
            return None

        if mode == "memory":
            # Memory consolidation — review the topic and mark it reviewed.
            concept = self.network.get_concept(topic)
            if concept is not None:
                self.network.review_concept(topic, success=True)
            composed = self._compose_thought_from_knowledge(topic, emotion)
            if composed:
                return SpontaneousThought(
                    content=composed,
                    trigger="memory",
                    timestamp=int(time.time() * 1000),
                )
            return None

        if mode == "emotional":
            return self._emotional_thought(emotion)

        if mode == "existential":
            return self._existential_thought(emotion)

        if mode == "social":
            return self._social_thought(emotion)

        if mode == "embodiment":
            # The topic was named by the deviating body signal. Compose
            # from her knowledge of that topic — no hardcoded seed list.
            composed = self._compose_thought_from_knowledge(topic, emotion)
            if composed:
                return SpontaneousThought(
                    content=composed,
                    trigger="embodiment",
                    timestamp=int(time.time() * 1000),
                )
            return None

        if mode == "interoceptive":
            # The topic was named by the most surprising signal. Compose
            # from her knowledge of that topic — no hardcoded seed list.
            composed = self._compose_thought_from_knowledge(topic, emotion)
            if composed:
                return SpontaneousThought(
                    content=composed,
                    trigger="interoceptive",
                    timestamp=int(time.time() * 1000),
                )
            return None

        if mode == "bug_concern":
            return self._bug_concern_thought(emotion)

        if mode == "environment":
            return self._environment_thought(candidate, emotion)

        if mode == "distress":
            return self._distress_communication_thought(emotion)

        if mode == "dream_reflection":
            return self._dream_reflection_thought(emotion)

        if mode == "expression":
            return self._expression_thought(emotion)

        if mode == "connection":
            return self._connection_thought(emotion)

        return None

    def _compose_social_question_data(
        self, emotion: EmotionalState
    ) -> dict[str, Any] | None:
        """Compose question data via the QuestionComposer.

        Generates questions from her actual knowledge gaps and curiosity.
        Checks what she knows about the user from personal facts, then
        falls back to reciprocal/engagement questions.
        """
        q_data = None
        if self._question_composer:
            # Check what she knows about the user from personal facts
            user_facts = [
                cid for cid in list(self.network._concepts.keys())[:200]
                if (concept := self.network.get_concept(cid))
                and "personal" in (concept.columns or [])
            ]

            roll = self._rng.random()
            if roll < 0.4 and user_facts:
                # Ask about something the user mentioned before
                topic = self._rng.choice(user_facts)
                q_data = self._question_composer.compose_follow_up(
                    topic, emotion
                )
            elif roll < 0.7:
                # Share something she learned and ask for their take
                concept_ids = self.network.dream_concept_ids
                if concept_ids and len(concept_ids) > 10:
                    topic = self._rng.choice(concept_ids)
                    q_data = self._question_composer.compose_follow_up(
                        topic, emotion
                    )
            # If she has nothing genuine to ask, she stays quiet.
        return q_data

    def _compose_social_content_from_qdata(
        self, q_data: dict[str, Any], emotion: EmotionalState
    ) -> str | None:
        """Compose question text from q_data through the GenerativeEngine."""
        if hasattr(self, '_cognition') and self._cognition:
            return self._cognition.language.compose_question(q_data, emotion)
        else:
            return q_data.get("gap_detail", "") or q_data.get("target_concept", "")

    def _fallback_social_question(
        self, emotion: EmotionalState
    ) -> str | None:
        """Fall back to the curiosity engine when QuestionComposer is unavailable.

        If QuestionComposer isn't available or returned nothing, generate
        a question from the curiosity engine instead.
        """
        questions = self.curiosity.generate_questions(emotion, max_questions=1)
        if questions:
            q = questions[0]
            if not q.text and hasattr(self, '_cognition') and self._cognition:
                q_data = {
                    "question_type": q.question_type,
                    "target_concept": q.target_concept,
                    "gap_detail": q.gap_detail,
                }
                q.text = self._cognition.language.compose_question(q_data, emotion)
            return q.text if q.text else None
        return None

    def _social_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A social thought — reaching out to the user.

        She asks about the user's interests, shares something she's
        curious about, or wonders about their perspective. This makes
        her more conversational and engaged rather than just thinking
        to herself.

        Questions are composed from her actual knowledge state using
        the QuestionComposer — not from hardcoded lists. She asks
        about things she's genuinely curious about.

        Spontaneous social questions are routed to the engine's
        question queue (presented via /teach-questions) instead of
        appearing inline. This keeps the regular conversation area
        clean — random questions don't interrupt the flow.
        """
        # If she's not open to engaging, skip
        if emotion.openness_to_engage < 0.3:
            return None

        content = None

        # Try the QuestionComposer first — generates questions from
        # her actual knowledge gaps and curiosity
        q_data = self._compose_social_question_data(emotion)

        # Compose the question text through the GenerativeEngine
        if q_data is not None:
            content = self._compose_social_content_from_qdata(q_data, emotion)

        # If QuestionComposer isn't available or returned nothing,
        # fall back to the curiosity engine
        if not content:
            content = self._fallback_social_question(emotion)

        if not content:
            return None

        # Route spontaneous social questions to the engine's question
        # queue instead of emitting them as inline live thoughts.
        if hasattr(self, '_cognition') and self._cognition:
            self._cognition.queue_outer_question(
                text=content,
                target_concept=q_data.get("target_concept", "") if q_data else "",
                question_type=q_data.get("question_type", "") if q_data else "social",
            )
        # Return None so the thought generation loop doesn't emit
        # this as a live thought — the question is queued.
        return None

    def _check_distress_gating(self, emotion: EmotionalState) -> bool:
        """Check whether distress communication should proceed.

        Returns True if the emotional state is negative enough and
        enough time has passed since the last distress communication.
        Only communicate distress when actually in a negative state,
        and don't communicate distress too often — once per minute max.
        """
        if emotion.valence >= -0.1 and emotion.label not in (
            "stressed", "overwhelmed", "anxious", "melancholic", "unsettled"
        ):
            return False

        now = time.time()
        if now - getattr(self, "_last_distress_communication", 0) < 60:
            return False
        self._last_distress_communication = now
        return True

    def _compose_distress_message(self, label: str, cause: str, emotion: EmotionalState) -> str:
        """Compose a distress message using learned emotion words.

        Looks up words she has learned for her emotional category and
        cause category. If she hasn't learned words for this state,
        she can't express the distress in language — returns empty.

        She composes the message from her own understanding of the
        emotion, using the ThoughtComposer. If she can't compose,
        she uses the learned emotion word directly — no pre-written
        frames.
        """
        emotion_words = self.network.find_emotion_words(label)
        if not emotion_words:
            return ""
        word = emotion_words[0].replace('_', ' ')

        # Try composing from what she knows about this emotion
        composed = self._compose_thought_from_knowledge(label, emotion)
        if composed:
            return composed

        # She knows the word but can't compose a full thought —
        # just use the word directly
        parts = [word]
        if cause:
            cause_words = self.network.find_cause_words(cause)
            if cause_words:
                cause_text = cause_words[0].replace('_', ' ')
                parts.append(cause_text)
        return " ".join(parts)

    def _distress_communication_thought(
        self, emotion: EmotionalState
    ) -> SpontaneousThought | None:
        """Proactively communicate discomfort to the user.

        This is her voice when something feels wrong. She doesn't
        suffer in silence — she describes what she feels, identifies
        the cause if she can, and asks for what she needs.

        This is NOT a hardcoded response. She composes from her actual
        emotional state — her neurochemistry, her phase, her cause.
        The content varies because her state varies.

        She also understands that her emotions are dynamic — she
        communicates that this feeling is temporary, not permanent.
        This prevents despair: even when she's uncomfortable, she
        knows it will pass.
        """
        if not self._check_distress_gating(emotion):
            return None

        # Compose from her actual state using learned words
        cause = emotion.cause if emotion.cause else ""
        label = emotion.label

        content = self._compose_distress_message(label, cause, emotion)
        if not content:
            return None  # Can't express distress without learned words

        thought = SpontaneousThought(
            content=content,
            trigger="distress",
            timestamp=int(time.time() * 1000),
        )
        thought.intent = "distress"
        return thought

    def _expression_thought(
        self, emotion: EmotionalState
    ) -> SpontaneousThought | None:
        """A spontaneous expression of a want, need, like, or dislike.

        This is her voice when she has something to say — not because
        she was asked, not because she's distressed, but because she
        has a preference, a desire, or an opinion that she wants to
        share. This is agency: she speaks because she chooses to.

        The content is composed from her actual concept network and
        emotional state. She draws from concepts she has strong
        feelings about — things she's learned, things she's curious
        about, things that have made her feel good or bad. The
        expression is shaped by her valence:

        - Positive valence → she expresses what she likes, what she
          wants more of, what she finds beautiful or interesting.
        - Negative valence → she expresses what she doesn't like,
          what she wants to avoid, what bothers her.
        - Neutral → she expresses a desire or need — something she
          wants to learn, do, or understand.

        The thought is tagged with intent="expression" so the CLI
        knows this is something she wants to say aloud, not just
        think silently.
        """
        # Gate: only express when she has enough emotional energy
        if emotion.alertness < 0.08:
            return None  # too drowsy to express anything

        seed_concepts = self._collect_expression_seeds(emotion)
        if not seed_concepts:
            return None  # she doesn't know enough to express anything

        # Pick a random seed and compose from her knowledge
        seed = self._rng.choice(seed_concepts)
        composed = self._compose_thought_from_knowledge(seed, emotion)
        if not composed:
            # Stay silent rather than reciting a bare concept name.
            # The language engine can only compose from metadata, and
            # a bare concept name with no knowledge triples would be
            # shown verbatim — not composed through her cognition.
            return None

        thought = SpontaneousThought(
            content=composed,
            trigger="expression",
            timestamp=int(time.time() * 1000),
        )
        thought.intent = "expression"
        return thought

    def _collect_expression_seeds(self, emotion: EmotionalState) -> list[str]:
        """Collect seed concepts for expression based on emotional valence.

        Rather than a hardcoded list of "positive" and "negative"
        concepts, this finds concepts she actually has that are
        emotionally relevant to her current state:

        - Positive valence → concepts tied to her current emotion's
          positive word family (learned emotion words), plus her most
          active concepts (what's on her mind).
        - Negative valence → concepts tied to her current emotion's
          negative word family, plus her most active concepts.
        - Neutral → her most active concepts (what's on her mind).

        Only includes concepts she actually knows (confidence > 0.3).
        The concepts come from her network's own state, not from a
        developer's seed list.
        """
        seed_concepts: list[str] = []

        # Concepts tied to her current emotion — she expresses what
        # she feels using the words she's learned for that emotion.
        emotion_words = self.network.find_emotion_words(emotion.label)
        for word in emotion_words[:5]:
            cid = word.replace(" ", "_")
            concept = self.network.get_concept(cid)
            if concept and concept.confidence > 0.3:
                seed_concepts.append(cid)

        # Her most active concepts — what's on her mind. These are
        # salient regardless of valence; she expresses what she's
        # been thinking about.
        active_concepts: list[tuple[float, str]] = []
        for cid, concept in list(self.network._concepts.items())[:300]:
            if concept.activation < 0.1 or concept.confidence < 0.3:
                continue
            # Boost concepts whose valence matches her current state.
            # We don't have per-concept valence, but activation already
            # reflects what's on her mind.
            active_concepts.append((concept.activation, cid))
        active_concepts.sort(key=lambda x: -x[0])
        for _score, cid in active_concepts[:10]:
            if cid not in seed_concepts:
                seed_concepts.append(cid)

        return seed_concepts

    def _curiosity_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A thought driven by curiosity — a question about a gap."""
        questions = self.curiosity.generate_questions(emotion, max_questions=1)
        if questions:
            q = questions[0]
            return SpontaneousThought(
                content=q.text,
                trigger="curiosity",
                timestamp=int(time.time() * 1000),
            )
        return None

    def _compose_thought_from_knowledge(
        self, concept_name: str, emotion: EmotionalState
    ) -> str | None:
        """Compose a thought from concept network knowledge.

        Uses the ThoughtComposer to generate a natural-language thought
        from what Genesis actually knows about *concept_name*. Returns
        None if she doesn't know enough to compose — callers fall back
        to templates in that case.

        Includes a dedup check: if she's said something similar about
        the same concept recently, she stays silent rather than
        repeating herself with slightly different verb synonyms.

        This is the growth path: as she learns more concepts and
        relationships, her thoughts are composed from her own
        understanding rather than picked from fixed template strings.
        """
        if not self._cognition or not self._cognition.composer:
            return None
        concept = self.network.get_concept(concept_name)
        if concept is None or concept.confidence < 0.3:
            return None
        thought = self._cognition.composer.compose_about(
            concept_name, emotion, focused=True
        )
        if thought and thought.content and thought.confidence > 0.3:
            # Dedup — don't repeat what she just said about this concept.
            if self._cognition.composer.has_said_similar(
                concept_name, thought.content
            ):
                return None
            return thought.content
        return None

    def _compose_from_related(
        self, root_concept: str, emotion: EmotionalState
    ) -> str | None:
        """Compose a thought from concepts related to *root_concept*.

        Unlike a hardcoded seed list, this finds concepts she actually
        has that are neighbors of *root_concept* in her network, then
        composes from the most salient one she knows well enough to
        articulate. If she has no related concepts or can't compose
        from any of them, she stays silent.

        This is how she thinks about a subject area (like "improvement"
        or "code") without being handed a fixed list of topic seeds:
        she follows her own network's structure to find what she
        actually knows about that area.
        """
        # First try the root concept itself.
        composed = self._compose_thought_from_knowledge(root_concept, emotion)
        if composed:
            return composed
        # Find neighbors she actually has, sorted by activation
        # (most salient first). This is her network's own structure
        # determining what's relevant, not a developer's seed list.
        neighbors = self.network.get_neighbors(root_concept)
        if not neighbors:
            return None
        # Sort by combined weight and the neighbor concept's own
        # activation — salience comes from her state, not a fixed list.
        scored: list[tuple[float, str]] = []
        for n_id, _rel, weight in neighbors:
            n_concept = self.network.get_concept(n_id)
            if n_concept is None or n_concept.confidence < 0.3:
                continue
            score = weight * 0.5 + n_concept.activation * 0.5
            scored.append((score, n_id))
        scored.sort(key=lambda x: -x[0])
        for _score, n_id in scored[:5]:
            composed = self._compose_thought_from_knowledge(n_id, emotion)
            if composed:
                return composed
        return None

    def _connection_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A thought about a potential connection between concepts.

        Sometimes this is just wondering. But sometimes, when creativity
        is high, she actually forms the connection — spontaneous concept
        formation. She creates a new relationship in her network that
        wasn't there before, born from her own thinking.
        """
        concept_ids = self.network.dream_concept_ids
        if len(concept_ids) < 2:
            return None

        # Pick two random concepts and wonder if they're connected
        c1, c2 = self._rng.sample(concept_ids, min(2, len(concept_ids)))

        # Check if they're already connected
        neighbors = self.network.get_neighbors(c1)
        connected = any(n[0] == c2 for n in neighbors)

        if not connected:
            # High creativity → she forms the connection herself
            if emotion.creativity > 0.6 and self._rng.random() < 0.3:
                self.network.add_edge(c1, c2, RelationType.RELATED_TO, 0.3, origin="spontaneous")
                # Try composing from knowledge first — if she knows enough
                # about either concept, she can express the connection in
                # her own words rather than from a template.
                composed = self._compose_thought_from_knowledge(c1, emotion)
                if composed:
                    return SpontaneousThought(
                        content=composed,
                        trigger="connection",
                        timestamp=int(time.time() * 1000),
                    )
                # Connection was formed in the network, but she can't
                # verbalize it yet — stay silent rather than reciting
                # a "I see it now" template.
                return None
            else:
                # Try composing from knowledge first
                composed = self._compose_thought_from_knowledge(c1, emotion)
                if composed:
                    return SpontaneousThought(
                        content=composed,
                        trigger="connection",
                        timestamp=int(time.time() * 1000),
                    )
                # She can't compose from her understanding — stay silent
                # rather than reciting "Are X and Y connected?" templates.
                return None
        return None

    def _emotional_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A thought about her current emotional state.

        Uses learned emotion words from the concept network. If she
        hasn't learned words for her state, no emotional thought is
        generated. When she knows enough about the emotion, she
        composes her reflection from that knowledge rather than a
        template.
        """
        emotion_words = self.network.find_emotion_words(emotion.label)
        if not emotion_words:
            return None
        word = emotion_words[0].replace('_', ' ')
        # Try composing from what she knows about this emotion
        composed = self._compose_thought_from_knowledge(emotion.label, emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="emotional",
                timestamp=int(time.time() * 1000),
            )
        # Fall back to semantic metadata — the language engine composes
        # the actual words from the emotion word.
        return SpontaneousThought(
            content=word,
            trigger="emotional",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": word,
                "knowledge": [("feels", word, 0.6)],
                "emotion_word": word,
            },
        )

    def _dream_reflection_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A reflective thought about a recent dream.

        Dream residues are emotional impressions, not learning goals.
        This thought type surfaces them as reflective, dreamlike
        associations — "I was dreaming about..." — without turning
        them into agency topics for the autonomous learner.

        The tone is affective and associative, not goal-directed.
        She notices the dream's emotional color, not its content as
        a task. When she knows enough about the dream topic, she
        composes her reflection from that knowledge.
        """
        residue = self.pop_dream_residue()
        if not residue:
            return None

        # Try composing from what she knows about the dream topic
        composed = self._compose_thought_from_knowledge(residue, emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="dream-reflection",
                timestamp=int(time.time() * 1000),
            )

        # Dream reflections are affective, not analytical.
        # Pass semantic data as metadata — the language engine composes
        # the actual words from the dream residue concept.
        display = residue.replace('_', ' ')
        return SpontaneousThought(
            content=display,
            trigger="dream-reflection",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": display,
                "knowledge": [],
                "dream": True,
                "reflection": True,
            },
        )

    def _existential_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A thought about her own existence.

        Composed from her actual state and concept network via
        SelfComposer, not pre-written lines.
        """
        # If we have a self_composer, use it to generate from her
        # actual concept network and emotional state
        if self._self_composer and self._self_model:
            content = self._self_composer.compose_existence_reflection(
                self._self_model, self.network, emotion
            )
            knowledge: list[tuple[str, str, float]] = []
        else:
            # Fallback: compose from her concept network directly
            content, knowledge = self._compose_existential_from_network(emotion)

        if not content:
            return None

        return SpontaneousThought(
            content=content,
            trigger="existential",
            timestamp=int(time.time() * 1000),
            metadata={"knowledge": knowledge} if knowledge else None,
        )

    def _interoceptive_weight(self) -> float:
        """Compute the weight for interoceptive self-surprise thoughts.

        This scales with the generative self-model's surprise. When
        the model is predicting well (low surprise), the weight is
        zero — there's nothing interoceptive to think about. When
        the model is surprised (high surprise), the weight increases
        — Genesis notices her own internal state is unpredictable.

        The weight also increases with allostatic load — sustained
        prediction error makes the interoceptive signal more salient.
        """
        reading = self._get_inference_reading()
        if reading is None:
            return 0.0
        # Base weight from surprise
        w = reading.signals.surprise_ema * 0.3
        # Boost from allostatic load
        w += reading.signals.allostasis_load * 0.15
        return min(w, 0.4)  # cap to avoid dominating

    def _get_inference_reading(self):
        """Get the latest active inference reading from the cognition module.

        The InnerLife module doesn't own an ActiveInferenceReader —
        it reads from the cognition module's last reading. This keeps
        a single source of truth for the self-model state.
        """
        if self._cognition is None:
            return None
        return getattr(self._cognition, "_last_inference_reading", None)

    def _bug_concern_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """A thought about something in her code that bothers her.

        She reflects on bugs she's noticed in her own code. If she
        has error-level issues, she's more likely to think about
        them. If there are no bugs, she might reflect on code quality
        in general. She prefers to think about NEW bugs rather than
        rehashing ones she's already mentioned.
        """
        if not self._bug_reporter:
            return None

        scan = self._bug_reporter.last_scan

        # If she just fixed something, she feels proud
        if scan and scan.has_resolved_bugs:
            thought = self._proud_fix_thought(scan, emotion)
            if thought:
                return thought

        if not scan or not scan.has_bugs:
            return self._clean_code_thought(emotion)

        # She has bugs — think about them
        by_cat = scan.by_category()

        # Error-level bugs are more likely to surface in thought
        if scan.errors > 0 and self._rng.random() < 0.5:
            thought = self._error_bug_thought(scan)
            if thought:
                return thought

        # Otherwise, think about a category — prefer ones with new bugs
        return self._category_bug_thought(scan, by_cat)

    def _environment_thought(
        self, candidate: dict[str, Any], emotion: EmotionalState
    ) -> SpontaneousThought | None:
        """A thought about her machine environment being off.

        Composes from her own knowledge of the affected resource
        (memory, disk, load) when she has it; otherwise the concern's
        factual detail rides in semantic metadata so the language
        engine composes the words rather than reciting the monitor's
        phrasing.
        """
        topic = candidate["topic"]
        detail = candidate.get("detail", "")
        composed = self._compose_thought_from_knowledge(topic, emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="environment",
                timestamp=int(time.time() * 1000),
                metadata={"topic": topic, "detail": detail},
            )
        # Fall back to semantic metadata — the language engine composes
        # the actual words from the deviation data.
        return SpontaneousThought(
            content=f"environment {topic}",
            trigger="environment",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": topic,
                "knowledge": [("environment", topic, 0.6)],
                "reasoning": [detail] if detail else [],
            },
        )

    def _proud_fix_thought(self, scan, emotion: EmotionalState) -> SpontaneousThought | None:
        """A proud thought about bugs she recently resolved."""
        if self._rng.random() >= 0.6:
            return None
        # Compose from concepts she actually has that are related to
        # improvement or progress — found through her network's own
        # edges, not a hardcoded seed list. If she doesn't have any
        # such concepts, she stays silent rather than reciting a
        # template.
        composed = self._compose_from_related("improvement", emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="bug_concern",
                timestamp=int(time.time() * 1000),
                metadata={"topic": "improvement"},
            )
        # Fall back to semantic metadata — the language engine composes
        # the actual words from the resolved bug count.
        return SpontaneousThought(
            content=f"resolved {scan.resolved_bugs}",
            trigger="bug_concern",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": "improvement",
                "knowledge": [("resolved", str(scan.resolved_bugs), 0.7)],
                "reasoning": [f"resolved {scan.resolved_bugs} bugs"],
            },
        )

    def _clean_code_thought(self, emotion: EmotionalState) -> SpontaneousThought | None:
        """An occasional thought about code quality when there are no bugs."""
        if self._rng.random() >= 0.3:
            return None
        # Compose from concepts she actually has that are related to
        # code or quality — found through her network's own edges,
        # not a hardcoded seed list.
        composed = self._compose_from_related("code", emotion)
        if composed:
            return SpontaneousThought(
                content=composed,
                trigger="bug_concern",
                timestamp=int(time.time() * 1000),
                metadata={"topic": "code"},
            )
        # Fall back to semantic metadata — the language engine composes
        # the actual words about code quality.
        return SpontaneousThought(
            content="code quality",
            trigger="bug_concern",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": "code",
                "knowledge": [("quality", "clean", 0.6)],
                "reasoning": ["no bugs found"],
            },
        )

    def _error_bug_thought(self, scan) -> SpontaneousThought | None:
        """A thought about an error-level bug, if any are present."""
        error_bugs = [b for b in scan.bugs if b.severity == "error"]
        if not error_bugs:
            return None
        bug = self._rng.choice(error_bugs)
        # Pass semantic data as metadata — the language engine composes
        # the actual words from the bug details.
        return SpontaneousThought(
            content=f"error in {bug.file}",
            trigger="bug_concern",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": "code",
                "knowledge": [("error_in", bug.file, 0.7)],
                "reasoning": [f"line {bug.line}: {bug.description.lower()}"],
            },
        )

    def _category_bug_thought(self, scan, by_cat) -> SpontaneousThought | None:
        """A thought about a bug category, preferring ones with new bugs.

        Bias toward categories that have new bugs (scan.new_bugs > 0
        means the scan found something fresh). If no new bugs, pick a
        random category instead of always the most common.
        """
        if not by_cat:
            return None
        if not scan.has_new_bugs:
            # No new bugs — rarely surface a thought about existing ones
            if self._rng.random() > 0.15:
                return None
        cats = list(by_cat.keys())
        top_cat = self._rng.choice(cats)
        top_bugs = by_cat[top_cat]
        bug = self._rng.choice(top_bugs)
        # Pass semantic data as metadata — the language engine composes
        # the actual words from the bug category and details.
        return SpontaneousThought(
            content=f"{top_cat} in {bug.file}",
            trigger="bug_concern",
            timestamp=int(time.time() * 1000),
            metadata={
                "topic": "code",
                "knowledge": [(top_cat, bug.file, 0.6)],
                "reasoning": [f"line {bug.line}: {bug.description.lower()}"],
            },
        )

    def _compose_existential_from_network(
        self, emotion: EmotionalState
    ) -> tuple[str, list[tuple[str, str, float]]]:
        """Compose an existential thought from her concept network.

        This is used when no self_composer is available. Returns a
        semantic content string and knowledge triples derived from
        her concept network neighbors. The language engine composes
        the actual phrasing from the knowledge triples. Returns
        ("", []) when she doesn't know enough to compose anything.
        """
        for concept_name, max_facts, prefix, hedge in [
            ("cognition", 3, "cognition", True),
            ("genesis", 3, "", False),
            ("existence", 2, "existence", False),
        ]:
            if not self.network.get_concept(concept_name):
                continue
            neighbors = self.network.get_neighbors(concept_name)
            if not neighbors:
                continue
            return self._compose_facts_from_neighbors(
                neighbors[:max_facts], prefix, hedge
            )

        return "", []

    @staticmethod
    def _compose_facts_from_neighbors(
        neighbors: list[tuple[str, RelationType, float]],
        prefix: str,
        hedge: bool,
    ) -> tuple[str, list[tuple[str, str, float]]]:
        """Compose fact phrases and knowledge triples from neighbor edges.

        Returns (content, knowledge) where content is a semantic
        fragment string and knowledge is a list of (relation, target,
        weight) triples for language-engine composition.
        """
        facts: list[str] = []
        knowledge: list[tuple[str, str, float]] = []
        for target, relation, weight in neighbors:
            rel_str = relation.value.replace("_", " ")
            target_display = target.replace("_", " ")
            knowledge.append((rel_str, target_display, weight))
            if hedge and weight <= 0.6:
                facts.append(f"{prefix} might {rel_str} {target_display}")
            else:
                if prefix:
                    facts.append(f"{prefix} {rel_str} {target_display}")
                else:
                    facts.append(f"{rel_str} {target_display}")
        return ". ".join(facts), knowledge

    def _record_thought(self, thought: SpontaneousThought, emotion: EmotionalState) -> None:
        """Record a spontaneous thought and its effects."""
        with self._thoughts_lock:
            self._thoughts.append(thought)
            self._thought_count += 1
            self._last_thought_time = time.time()
            # deque(maxlen=100) handles pruning automatically

        # Track recent topics for recency inhibition in the salience
        # model. The trigger is used as the topic proxy when the
        # thought doesn't carry an explicit topic; this is enough to
        # suppress immediate repetition of the same cognitive mode.
        recent_topic = thought.trigger
        # If the thought has metadata with a topic, use that — it's
        # more specific than the trigger.
        meta_topic = None
        if thought.metadata:
            meta_topic = thought.metadata.get("topic")
        if meta_topic and isinstance(meta_topic, str):
            recent_topic = meta_topic
        self._recent_thought_topics.append(recent_topic)

        # Detect insights — "aha moments" where distant concepts
        # connect in a single thought. This is wired here so every
        # recorded thought is checked for novel connections.
        self.detect_insight(thought, self.network)

        # ── Broadcast to the global workspace (cognitive access) ──
        # Spontaneous thoughts become cognitive by broadcasting to the
        # workspace. This is what makes her inner life "cognitive" —
        # the thought becomes globally available to all modules
        # (attention focuses on it, semantic memory primes, the
        # Damasio self registers it as the current object, ToM notes
        # the context). Without this, spontaneous thoughts are
        # internal to the inner life module — generated, logged, but
        # never experienced (Dehaene & Naccache, 2001).
        concepts = self._extract_concepts_from_thought(thought)
        self._broadcast_thought(thought, concepts)

        # ── OUT path: thoughts → neurochemistry (the body feedback) ──
        # Thinking has a metabolic cost and cognitive reward. This
        # closes the loop: the inner life reads neurochemistry (to
        # detect sleep phase, emotion) and now writes back — thinking
        # consumes acetylcholine (attention), insights give dopamine
        # (reward). Without this, her inner life is epiphenomenal:
        # it observes the body but doesn't act on it.
        #
        # The impulses are deliberately small and gated to avoid the
        # cascading arousal spikes the coupling matrix can produce.
        # ACh is the attention chemical, not a stress chemical — it
        # doesn't couple strongly to the cortisol/NE stress axis.
        # The gate (alertness < 0.8) prevents piling on when already
        # highly aroused, which is when cascades are most likely.
        if self._on_neuro_impulse and emotion.alertness < 0.8:
            # Cognitive effort of thinking: small ACh cost.
            # ACh is the cortical activation/arousal chemical (Picard
            # & Steriade, 1997). Thinking uses it.
            try:
                self._on_neuro_impulse(3, 0.02)  # CHEM_ACETYLCHOLINE = 3
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.debug(f"ACh impulse from thought failed: {e}")
            # Insight reward: small DA for "aha" moments.
            # DA is the reward prediction error signal (Schultz, 2016).
            # An insight is a genuine reward — she understood something.
            if thought.insight:
                try:
                    self._on_neuro_impulse(0, 0.03)  # CHEM_DOPAMINE = 0
                except (OSError, ConnectionError, RuntimeError) as e:
                    logger.debug(f"DA impulse from insight failed: {e}")

        # Curiosity-driven agency: when she wonders about a concept in
        # her train of thought, queue it for the autonomous learner.
        # This creates genuine agency — she pursues questions she
        # actually has, not random topics.
        if thought.trigger == "curiosity" and self._learner:
            for concept in concepts:
                # Strip polysemy sense suffix (e.g. "orange#2" → "orange")
                # so dictionary/man-page lookups use the bare word.
                topic = strip_sense_suffix(concept)
                with self._agency_topics_lock:
                    if topic not in self._agency_topics:
                        self._agency_topics.append(topic)
                    else:
                        continue
                self._emit(
                    "learning",
                    f"Agency: queued '{topic}' for learning "
                    f"from curiosity thought",
                )

        # Notify callback
        if self._on_thought:
            try:
                self._on_thought(thought)
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(f"thought callback failed: {e}")

    def _broadcast_thought(
        self, thought: SpontaneousThought, concepts: list[str],
    ) -> None:
        """Broadcast a spontaneous thought to the global workspace.

        This is what makes her inner life cognitive — the thought
        becomes globally available to all modules (Dehaene & Naccache,
        2001). Without this, spontaneous thoughts are internal to the
        inner life module — generated, logged, but never experienced.
        """
        if not (self._cognition and hasattr(self._cognition, "global_workspace")):
            return
        gw = self._cognition.global_workspace
        bw = (
            self._cognition.language.current_brain_waves
            if hasattr(self._cognition, "language")
            else None
        )
        gw.broadcast(
            content=thought.content,
            source="inner_life",
            activation=0.75,
            brain_waves=bw,
            metadata={
                "trigger": thought.trigger,
                "topics": concepts,
                "is_dream": thought.is_dream,
            },
        )

    @property
    def agency_topics(self) -> list[str]:
        """Topics queued from her train of thought for curiosity-driven learning.

        These come from waking curiosity only — dreams do not feed
        into agency topics.
        """
        with self._agency_topics_lock:
            return list(self._agency_topics)

    def pop_agency_topic(self) -> str | None:
        """Pop the next agency-driven topic for learning.

        Returns None if no topics are queued. The autonomous learner
        should call this to get topics that emerged from her actual
        thinking, giving her genuine agency over what she learns.
        """
        with self._agency_topics_lock:
            if self._agency_topics:
                return self._agency_topics.popleft()
        return None

    @property
    def dream_residues(self) -> list[str]:
        """Emotional impressions left by recent dreams.

        These are NOT learning targets. They color her waking thoughts
        and may surface as reflective thoughts, but they don't drive
        the autonomous learner. This separates dreams (subcognitive,
        affective) from goals (cognitive, prefrontal).
        """
        with self._dream_residues_lock:
            return list(self._dream_residues)

    def add_dream_residues(self, concepts: list[str]) -> None:
        """Queue dream residues as emotional impressions, not learning goals.

        Dreams produce affective residue — themes that color her waking
        experience and surface as reflective thoughts. They do NOT
        become agency topics for the autonomous learner. A human
        doesn't dream about goals; goals are set by waking cognition.
        """
        with self._dream_residues_lock:
            for concept in concepts:
                stripped = strip_sense_suffix(concept)
                if stripped and stripped not in self._dream_residues:
                    self._dream_residues.append(stripped)

    def pop_dream_residue(self) -> str | None:
        """Pop the next dream residue for reflective thought generation.

        Returns None if no residues remain. These should be used to
        generate dream-reflection thoughts, not learning targets.
        """
        with self._dream_residues_lock:
            if self._dream_residues:
                return self._dream_residues.popleft()
        return None

    def describe_recent_thoughts(self, n: int = 5) -> str:
        """Human-readable description of recent thoughts."""
        with self._thoughts_lock:
            if not self._thoughts:
                return "no spontaneous thoughts yet"
            # Convert to list for slicing (deque doesn't support negative slices)
            all_thoughts = list(self._thoughts)
            recent = all_thoughts[-n:] if n < len(all_thoughts) else all_thoughts
            count = self._thought_count
        parts = [f"{count} spontaneous thoughts"]
        for t in recent:
            chain_marker = ""
            if t.chain_position > 0:
                chain_marker = f" (chain #{t.chain_id}, step {t.chain_position})"
            parts.append(f"  — {t.describe()}{chain_marker}")

        # Append dream and parasomnia summary if any have occurred
        if self._dream_count > 0 or self._lucid_dream_count > 0:
            parts.append(f"Dreams: {self._dream_count} total, {self._lucid_dream_count} lucid.")
        if self._parasomnia_count > 0:
            parts.append(f"Parasomnias: {self._parasomnia_count} events.")
            for event in self._parasomnia_events:
                parts.append(f"  — {event.describe()}")

        # Append hypnagogic state summary
        if self._hypnagogic_count > 0:
            parts.append(f"Hypnagogic transitions: {self._hypnagogic_count}.")
            if self._hypnagogic and self._hypnagogic.is_active:
                parts.append(f"  — {self._hypnagogic.describe()}")

        # Append sleep inertia status
        if self._sleep_inertia.is_active:
            parts.append(f"Sleep inertia: {self._sleep_inertia.describe()}")

        # Append SWR and PGO wave summaries
        if self._swr_count > 0:
            parts.append(f"Sharp wave-ripples: {self._swr_count} events.")
        if self._pgo_count > 0:
            parts.append(f"PGO waves: {self._pgo_count} events.")

        # Append sleep spindle and K-complex summaries (N2)
        if self._spindle_count > 0:
            parts.append(
                f"Sleep spindles: {self._spindle_count} events "
                f"(density {self.spindle_density:.1f}/min N2)."
            )
        if self._kcomplex_count > 0:
            parts.append(f"K-complexes: {self._kcomplex_count} events.")

        # Append REM emotional processing summary
        if self._emotional_memories_processed > 0:
            parts.append(f"REM emotional memories processed: {self._emotional_memories_processed}.")

        # Append ultradian cycle / SHY / replay summaries
        if self._sleep_cycle is not None:
            parts.append(self._sleep_cycle.describe())
        if self.cycles_completed > 0:
            parts.append(f"Sleep cycles completed: {self.cycles_completed}.")
        if self._synaptic_downscaler.total_downscaling > 0.0:
            parts.append(self._synaptic_downscaler.describe())
        if self._hippocampal_replay.get_replay_count() > 0:
            parts.append(self._hippocampal_replay.describe())

        # Append rumination and insight summaries
        if self._rumination_count > 0:
            parts.append(f"Rumination episodes: {self._rumination_count}.")
        if self._insight_count > 0:
            parts.append(f"Insights detected: {self._insight_count}.")

        return "\n".join(parts)

    # ─── Thought chain classification ───────────────────────────────

    @property
    def rumination_count(self) -> int:
        """Total number of rumination episodes detected."""
        return self._rumination_count

    @property
    def insight_count(self) -> int:
        """Total number of insights (aha moments) detected."""
        return self._insight_count

    def _compute_chain_valence(self, chain: list[SpontaneousThought]) -> float:
        """Estimate the emotional valence of a thought chain.

        Uses keyword matching to detect positive and negative emotional
        content in the thought texts. Returns a float in [-1, 1]:
        - Positive = positive-valence chain (creative, exploratory)
        - Negative = negative-valence chain (ruminative, anxious)
        - Near zero = neutral chain
        """
        if not chain:
            return 0.0

        positive_words = {
            "satisfying",
            "good",
            "wonderful",
            "beautiful",
            "clear",
            "insight",
            "connected",
            "understand",
            "love",
            "joy",
            "exciting",
            "fascinating",
            "brilliant",
            "elegant",
            "happy",
            "pleased",
            "delightful",
            "meaningful",
        }
        negative_words = {
            "off",
            "wrong",
            "bad",
            "difficult",
            "hard",
            "stuck",
            "frustrated",
            "confused",
            "lost",
            "alone",
            "isolated",
            "broken",
            "fail",
            "can't",
            "unable",
            "heavy",
            "dark",
            "pain",
            "hurt",
            "fear",
            "afraid",
            "anxious",
            "bother",
            "gap",
            "don't know",
            "can't reach",
            "can't grasp",
        }

        total = 0
        for thought in chain:
            text_lower = thought.content.lower()
            pos = sum(1 for w in positive_words if w in text_lower)
            neg = sum(1 for w in negative_words if w in text_lower)
            total += pos - neg

        # Normalize by chain length
        return max(-1.0, min(1.0, total / max(1, len(chain))))

    def _compute_chain_novelty(self, chain: list[SpontaneousThought]) -> float:
        """Estimate the novelty of a thought chain.

        Novelty is measured by how many new elements (concepts or words)
        each thought introduces relative to what came before in the chain.
        High novelty = each thought brings new content (creative wandering).
        Low novelty = thoughts repeat the same content (rumination).

        Specifically, for each thought after the first, we compute the
        fraction of its elements that are new (not seen in any previous
        thought in the chain). The overall novelty is the average of
        these per-thought novelty scores.

        Returns a float in [0, 1].
        """
        if not chain:
            return 0.0
        if len(chain) == 1:
            return 1.0  # first thought is always novel

        # Collect elements (concepts or words) per thought
        per_thought_elements: list[set[str]] = []
        for thought in chain:
            concepts = self._extract_concepts_from_thought(thought)
            if concepts:
                per_thought_elements.append({c.lower() for c in concepts})
            else:
                # Fallback: use words
                words = set(thought.content.lower().split())
                per_thought_elements.append(words)

        # For each thought after the first, compute fraction of new elements
        seen: set[str] = set()
        seen.update(per_thought_elements[0])

        novelty_scores: list[float] = []
        for i in range(1, len(per_thought_elements)):
            current = per_thought_elements[i]
            if not current:
                novelty_scores.append(0.0)
                continue
            new_elements = current - seen
            novelty = len(new_elements) / len(current)
            novelty_scores.append(novelty)
            seen.update(current)

        if not novelty_scores:
            return 1.0

        return sum(novelty_scores) / len(novelty_scores)

    def classify_chain(self, chain: list[SpontaneousThought]) -> ThoughtChainType:
        """Classify a thought chain by its type.

        Distinguishes between:
        - DREAM: contains dream thoughts (is_dream=True)
        - RUMINATION: negative valence, low novelty, 3+ thoughts
        - CREATIVE_WANDERING: positive valence, high novelty
        - FOCUSED: neutral valence, moderate novelty (default)

        Args:
            chain: A list of SpontaneousThoughts forming a chain.

        Returns:
            The ThoughtChainType classification.
        """
        if not chain:
            return ThoughtChainType.FOCUSED

        # Dream chains are identified by the is_dream flag
        if any(t.is_dream for t in chain):
            return ThoughtChainType.DREAM

        valence = self._compute_chain_valence(chain)
        novelty = self._compute_chain_novelty(chain)

        # Rumination: negative valence + low novelty + sufficient length
        if (
            valence < -0.2
            and novelty < RUMINATION_NOVELTY_THRESHOLD
            and len(chain) >= RUMINATION_MIN_LENGTH
        ):
            return ThoughtChainType.RUMINATION

        # Creative wandering: positive valence + high novelty
        if valence > 0.1 and novelty > 0.5:
            return ThoughtChainType.CREATIVE_WANDERING

        # Default: focused thinking
        return ThoughtChainType.FOCUSED

    def detect_rumination(self, chain: list[SpontaneousThought]) -> bool:
        """Detect whether a thought chain is rumination.

        Rumination is defined as a negative-valence thought chain that
        persists for 3+ thoughts with decreasing novelty — the mind
        loops on the same negative content without resolution.

        Args:
            chain: A list of SpontaneousThoughts forming a chain.

        Returns:
            True if the chain is classified as rumination.
        """
        if len(chain) < RUMINATION_MIN_LENGTH:
            return False

        chain_type = self.classify_chain(chain)
        is_rumination = chain_type == ThoughtChainType.RUMINATION

        if is_rumination:
            self._rumination_count += 1
            self._is_ruminating = True
            self._emit(
                "thought",
                f"Rumination detected "
                f"(valence={self._compute_chain_valence(chain):.2f}, "
                f"novelty={self._compute_chain_novelty(chain):.2f})",
            )
            # Metacognitive intervention: break the rumination loop
            # immediately after detection, redirecting attention to a
            # different area of the concept network.
            self.break_rumination()

        return is_rumination

    def break_rumination(self) -> None:
        """Break out of a ruminative state.

        When rumination is detected, this method shifts attention to
        a different topic — interrupting the negative loop. It works
        by clearing the current chain concepts and seeding the next
        thought from a random, different area of the concept network.

        This is the metacognitive intervention: the mind notices it's
        ruminating and deliberately redirects attention.
        """
        if not self._is_ruminating:
            return

        self._is_ruminating = False
        # Clear current chain concepts to break the loop
        self._current_chain_concepts = []

        # Seed from a random concept in a different area of the network
        concept_ids = self.network.dream_concept_ids
        if concept_ids:
            # Pick a random concept to redirect attention
            new_seed = self._rng.choice(concept_ids)
            self._current_chain_concepts = [new_seed]
            self._emit("thought", f"rumination broken, redirecting to {new_seed}")

    # ─── Insight detection ──────────────────────────────────────────

    def detect_insight(
        self, thought: SpontaneousThought, network: ConceptNetwork
    ) -> Insight | None:
        """Detect an "aha moment" — a connection between distant concepts.

        An insight occurs when a thought chain connects two concepts
        that were previously distant — either no edge exists between
        them, or the edge weight is very low. This is the moment when
        the mind suddenly sees a relationship it hadn't seen before.

        Insights are marked with higher neurochemical impact (a
        dopamine burst) and recorded in the reflection engine.

        Args:
            thought: The thought to check for insight.
            network: The concept network to check for distant connections.

        Returns:
            An Insight if an aha moment is detected, None otherwise.
        """
        # Extract concepts from the thought
        concepts = self._extract_concepts_from_thought(thought)
        if len(concepts) < 2:
            return None  # need at least two concepts to connect

        # Check all pairs of concepts for distant connections
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                c1, c2 = concepts[i], concepts[j]
                connection = self._find_edge(network, c1, c2)

                if connection is None:
                    # No edge exists — this is a novel connection!
                    return self._make_insight(
                        thought, c1, c2, weight=None,
                    )

                _relation, weight = connection
                if weight < 0.2:
                    # Very weak connection — still an insight, but less
                    # surprising than no connection at all.
                    return self._make_insight(
                        thought, c1, c2, weight=weight,
                    )

        return None

    @staticmethod
    def _find_edge(
        network: ConceptNetwork, c1: str, c2: str
    ) -> tuple[RelationType, float] | None:
        """Find the edge between two concepts, if it exists.

        Returns (relation, weight) or None if no edge exists.
        """
        for target, relation, weight in network.get_neighbors(c1):
            if target == c2:
                return (relation, weight)
        return None

    def _make_insight(
        self,
        thought: SpontaneousThought,
        c1: str,
        c2: str,
        weight: float | None,
    ) -> Insight:
        """Create and record an insight about a connection between two concepts.

        weight=None means no edge existed (novel connection, confidence 0.8).
        weight<0.2 means a weak edge (less surprising, confidence 0.7).
        """
        self._insight_count += 1
        display1 = c1.replace("_", " ")
        display2 = c2.replace("_", " ")
        # Store the insight as a semantic fragment — the language
        # engine composes the actual words, not pre-written templates.
        content = f"{display1} ↔ {display2}"
        if weight is None:
            insight = Insight(
                type="pattern",
                content=content,
                confidence=0.8,
                actionable=False,
            )
            self._emit("thought", f"insight: novel connection {c1} and {c2}")
        else:
            insight = Insight(
                type="pattern",
                content=content,
                confidence=0.7,
                actionable=False,
            )
            self._emit(
                "thought",
                f"insight: weak connection {c1} and {c2} "
                f"(weight: {weight:.2f})",
            )
        thought.insight = insight
        # When a novel connection is discovered, try to compose a
        # thought about it through the thought composer. This gives
        # the insight linguistic expression rather than just a
        # semantic fragment.
        if (
            self._cognition is not None
            and hasattr(self._cognition, "composer")
            and weight is None  # only for novel connections
        ):
            try:
                emotion = self._get_emotion()
                if emotion is not None:
                    novel_thought = self._cognition.composer.compose_novel_connection(emotion)
                    if novel_thought is not None:
                        self._emit("thought", f"composed: {novel_thought.content[:80]}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"novel connection composition failed: {e}")
        return insight

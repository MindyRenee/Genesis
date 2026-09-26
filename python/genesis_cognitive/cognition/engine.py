"""Cognition — where Genesis thinks.

The cognition engine is the core of the cognitive mind. It combines:

- **Perception**: what the user said
- **Emotion**: how Genesis feels right now
- **Memory**: what Genesis remembers
- **Self-model**: who Genesis is

...and produces a Thought that the language engine renders as text.

This is not a chatbot pattern-matcher. It's a deliberative system that
considers multiple factors before deciding what to say. The same input
can produce different responses depending on its emotional state, what
it remembers, and what it's been thinking about.

# The thought process

1. **Perceive** the input (intent, topics, sentiment)
2. **Assess** current emotional state from neurochemistry
3. **Retrieve** relevant memories
4. **Deliberate** — combine all factors to decide what to say
5. **Form** a Thought with the right intent, emotion, and content
6. **Render** the Thought through the language engine

The deliberation step is where the real work happens. It's a priority-
ordered decision tree that considers what kind of response is most
appropriate given everything Genesis knows.

# Architecture: orchestrator + extracted subsystems

CognitionEngine is the orchestrator. The focused subsystems live in
sibling modules and are wired in ``__init__``:

- ``answer_composer`` — factual/comparison/parts/consumption/counterfactual/
  explanatory/planning/goal/mission answers
- ``code_tools`` — code tool dispatch and code-discussion composition
- ``concept_learner`` — word labeling and activation spreading
- ``feeling_reporter`` — feeling/concern/environment/bug/empathetic reports
- ``memory_store`` — conversation and cognitive event persistence
- ``question_handler`` — question routing and answering
- ``response_styler`` — metacognitive tone adjustment
- ``self_inquiry`` — self-directed questions
- ``topic_resolver`` — topic extraction and concept variant matching

Each subsystem exposes its own public class; CognitionEngine holds
instances and delegates via thin wrapper methods (see the ``Delegates
to the <Subsystem> subsystem.`` docstrings throughout this file).
"""

from __future__ import annotations

import heapq
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from genesis_client import GenesisClient, NeuroSummary
from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_NOREPINEPHRINE,
    PHASE_NREM,
    PHASE_REM,
)

from ..attention import AttentionSystem
from ..brain_waves import (
    BrainWave,
    BrainWaveState,
    add_brain_wave_drive,
    assess_brain_waves,
)
from ..cognition.meta_cognitive_router import MetaCognitiveRouter, Route
from ..concepts import ConceptNetwork, RelationType, is_world_concept
from ..emotion import EmotionalState, assess_emotion
from ..executive import ExecutiveFunction, TaskState
from ..global_workspace import GlobalWorkspace, WorkspaceItem
from ..language import ComprehensionEngine, LanguageEngine, SelfMonitor, Thought
from ..language.grounding import SemanticGrounder
from ..learning import (
    STDP,
    CuriosityEngine,
    PredictionContext,
    PredictiveCodingLayer,
    Question,
    TDLearner,
)
from ..memory import (
    MemoryContext,
    MemoryEngine,
    ProceduralMemory,
    SemanticMemory,
    SpacedRepetitionScheduler,
)
from ..narrative import NarrativeEngine
from ..perception import Intent, Perception, perceive
from ..reasoning import (
    DecisionResult,
    DriftDiffusionModel,
    GoalType,
    ReasoningEngine,
    ReasoningResult,
    ReasoningType,
    TheoryOfMind,
)
from ..self import (
    DamasioSelfHierarchy,
    ErrorMonitor,
    Insight,
    MetacognitiveStrategy,
    PredictionError,
    ReflectionEngine,
    SelfModel,
)
from ..user_profile import UserProfile
from .answer_composer import AnswerComposer
from .code_tools import CodeToolHandler
from .concept_learner import ConceptLearner
from .feeling_reporter import FeelingReporter
from .memory_store import MemoryStore
from .question_handler import QuestionHandler
from .response_styler import ResponseStyler
from .self_inquiry import SelfInquiryHandler
from .topic_resolver import TopicResolver

if TYPE_CHECKING:
    from ..bug_reporter import BugReporter
    from ..emotional_regulator import EmotionalRegulator
    from ..language import ComprehensionResult
    from ..learning import AutonomousLearner, Prediction
    from ..learning import PredictionError as PCPredictionError
    from ..reasoning import UserModel
    from ..self import AnswerAssessment
    from ..system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

__all__ = ["CognitionEngine", "CognitiveState", "Goal"]

# ─── Final grammar cleanup patterns ───────────────────────────────
# These catch grammar issues introduced AFTER the initial render —
# e.g. curiosity questions appended to the response that bypass the
# generator's grammar safety net and the self-monitor.
_FINAL_REPEATED_PUNCT_RE = re.compile(r"([.!?])\1+")
_FINAL_DOUBLE_SPACE_RE = re.compile(r"  +")
_FINAL_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;!?])")
# A '?'/'!' followed by bare trailing periods — a punctuated slot that
# landed mid-join and then had a period appended ("Tell me more?.").
# The question/exclamation mark is the real terminal punctuation.
_FINAL_STRAY_PERIOD_RE = re.compile(r"([?!])\s*\.+\s*$")
# A dangling dash before terminal punctuation — an em-dash fragment
# left by an omitted trailing slot ("If you do not mind —?").
_FINAL_DANGLING_DASH_RE = re.compile(r"\s*[—–-]\s*([?!.]+)\s*$")


@dataclass(slots=True)
class CognitiveState:
    """A snapshot of Genesis's cognitive state at a moment in time.

    This is what the cognition engine produces before rendering. It
    captures everything that went into a response, for introspection
    and debugging.
    """

    perception: Perception
    emotion: EmotionalState
    memory_context: MemoryContext
    thought: Thought
    brain_waves: BrainWaveState | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    # ── New cognitive system outputs (introspectable) ──
    prediction_error: float | None = None
    prediction_level_errors: dict[str, float] | None = None
    attention_foci: list[str] | None = None
    decision_option: str | None = None
    decision_confidence: float | None = None
    workspace_items: list[str] | None = None
    executive_goal: str | None = None
    user_model_summary: str | None = None
    damasio_feeling: str | None = None
    # Active inference self-model annotation — semantic tags from the
    # generative self-model (e.g. "self-surprise", "allostatic-strain").
    # Layers on top of the neurochemical feeling.
    self_model_label: str | None = None
    # The minimal self's coherence — how well the generative self-model
    # predicts its own state [0, 1]. Blended from neurochemical
    # precision, metacognitive precision, cognitive trajectory
    # precision, and workspace integration. High = "I understand
    # myself"; low = "I don't understand what's happening inside me."
    # This is the numeric counterpart to self_model_label — the label
    # carries semantic tags, the coherence carries the felt intensity.
    self_model_coherence: float | None = None
    semantic_facts_extracted: int | None = None
    procedural_skill: str | None = None
    error_monitor_caution: float | None = None
    comprehension_speech_act: str | None = None
    comprehension_metaphor: str | None = None
    comprehension_irony: bool | None = None
    comprehension_idiom: str | None = None
    self_monitor_repairs: int | None = None
    td_rpe: float | None = None

    def describe(self) -> str:
        """Human-readable description of the cognitive process."""
        base = (
            f"Intent: {self.perception.intent.value} | "
            f"Emotion: {self.emotion.label} | "
            f"Thought intent: {self.thought.intent} | "
            f"Confidence: {self.thought.confidence:.2f}"
        )
        if self.brain_waves:
            base += f" | Waves: {self.brain_waves.label}"
        if self.prediction_error is not None:
            base += f" | PredError: {self.prediction_error:.2f}"
        if self.attention_foci:
            base += f" | Attention: {', '.join(self.attention_foci[:3])}"
        if self.decision_option:
            base += f" | Decision: {self.decision_option}"
        if self.executive_goal:
            base += f" | Goal: {self.executive_goal}"
        if self.damasio_feeling:
            base += f" | Feeling: {self.damasio_feeling}"
        if self.self_model_label:
            base += f" | SelfModel: {self.self_model_label}"
        if self.self_model_coherence is not None:
            base += f" | Coherence: {self.self_model_coherence:.2f}"
        return base


@dataclass
class Goal:
    """A self-generated learning or reasoning target."""

    target: str
    goal_type: str  # learn_definition, learn_relation, resolve_contradiction, etc.
    reason: str
    detail: str = ""
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    resolved: bool = False


class CognitionEngine:
    """The thinking engine — combines all inputs to produce thoughts.

    This is where Genesis's mind comes together. The cognition engine
    reads its emotional state, perceives the input, retrieves memories,
    and deliberates about what to say.
    """

    def __init__(
        self,
        client: GenesisClient,
        self_model: SelfModel,
        memory: MemoryEngine,
        language: LanguageEngine,
        network: ConceptNetwork | None = None,
        reasoning: ReasoningEngine | None = None,
        reflection: ReflectionEngine | None = None,
        curiosity: CuriosityEngine | None = None,
        narrative: NarrativeEngine | None = None,
        data_dir: str | None = None,
        user_profile: UserProfile | None = None,
    ) -> None:
        """Initialize the cognition engine and all subsystems."""
        self._init_core(client, self_model, memory, language, user_profile)
        self._init_intelligence(network, reasoning, reflection, curiosity, narrative, self_model)
        self._init_composers(data_dir)
        self._init_cognitive_systems(data_dir)
        # Feeling reporter — created after cognitive systems are
        # initialized since it needs the ThoughtComposer and _rng.
        # Optional deps (bug_reporter, system_monitor, damasio_self)
        # are injected after late init via update_dependencies().
        self._feeling_reporter = FeelingReporter(
            network=self.network,
            language=self.language,
            composer=self.composer,
            self_model=self_model,
            rng=self._rng,
            meta_emotion_builder=self._build_meta_emotion,
        )
        # Inject damasio_self (created in _init_cognitive_systems).
        self._feeling_reporter.update_dependencies(
            damasio_self=self.damasio_self,
        )
        # Answer composer — extracted subsystem for factual/comparison/
        # counterfactual/explanatory/planning answers.
        self._answer_composer = AnswerComposer(
            network=self.network,
            language=self.language,
            meta_emotion_builder=self._build_meta_emotion,
            goals_getter=lambda: self._goals,
            mission_getter=lambda: self._mission,
        )
        # Question handler — extracted subsystem for question routing.
        self._question_handler = QuestionHandler(
            network=self.network,
            language=self.language,
            composer=self.composer,
            self_composer=self.self_composer,
            self_model=self_model,
            theory_of_mind=self.theory_of_mind,
            tools=self.tools,
            display_name=self._display_name,
            relation_verb=self._relation_verb,
            resolve_topics=self._resolve_topics,
            user_profile=self.user_profile,
        )
        # Self-inquiry handler — extracted subsystem for self-questions.
        self._self_inquiry_handler = SelfInquiryHandler(
            network=self.network,
            language=self.language,
            composer=self.composer,
            self_composer=self.self_composer,
            self_model=self_model,
            self_learner=self.self_learner,
            self_assessment=self.self_assessment,
            topology=self.topology,
            reflection=self.reflection,
            narrative=self.narrative,
            client=self.client,
            memory=self.memory,
            feeling_reporter=self._feeling_reporter,
            resolve_topics=self._resolve_topics,
            map_personal_question=self._question_handler._map_personal_question,
            get_curiosity_questions=self.get_curiosity_questions,
        )
        # Code tool handler — extracted subsystem for code tools.
        self._code_tool_handler = CodeToolHandler(
            network=self.network,
            language=self.language,
            composer=self.composer,
            tools=self.tools,
            resolve_topics=self._resolve_topics,
            build_meta_emotion=self._build_meta_emotion,
        )
        # Response styler — extracted subsystem for tone adjustment.
        self._response_styler = ResponseStyler(
            language=self.language,
            composer=self.composer,
            working_memory=self.working_memory,
            rng=self._rng,
            response_style_getter=lambda: self._response_style,
            network=self.network,
        )
        # Memory store — extracted subsystem for memory persistence.
        self._memory_store = MemoryStore(
            memory=self.memory,
            client=self.client,
            regulator=self.regulator,
            self_model=self_model,
            td_learner=self.td_learner,
            recognize_bonded_user=self._recognize_bonded_user,
            last_summary_getter=lambda: self._last_summary,
            last_memory_mode_getter=lambda: self._last_memory_mode,
        )
        # Concept learner — extracted subsystem for word labeling.
        self._concept_learner = ConceptLearner(network=self.network)
        self._init_state()
        self._seed_initial_concepts()

    @property
    def regulator(self) -> EmotionalRegulator | None:
        return self._regulator

    @regulator.setter
    def regulator(self, value: EmotionalRegulator | None) -> None:
        """Late-bound by Mind after creation — propagate to extracted
        subsystems that captured the slot at construction time."""
        self._regulator = value
        self._memory_store._regulator = value

    def _init_core(
        self,
        client: GenesisClient,
        self_model: SelfModel,
        memory: MemoryEngine,
        language: LanguageEngine,
        user_profile: UserProfile | None = None,
    ) -> None:
        """Initialize core references and self-awareness module slots."""
        self.client = client
        self.self_model = self_model
        self.memory = memory
        self.language = language
        self.user_profile = user_profile
        self._regulator = None  # set by Mind after creation

        # Fast metacognitive router — decides which specialist should
        # handle the input before the heavy deliberation pipeline runs.
        self.meta_router = MetaCognitiveRouter()

        # Self-awareness modules — set by Mind after creation.
        # These give Genesis awareness of its own code quality and
        # machine environment. They're optional (None by default) so
        # the cognition engine works standalone in tests.
        self.bug_reporter: BugReporter | None = None  # set by Mind after creation
        self.system_monitor: SystemMonitor | None = None  # set by Mind after creation

        # Gap-to-learning callback — set by Mind after creation.
        # When it detects "I don't know what 'X' is", this callback
        # feeds X to the autonomous learner so it can look it up and
        # remember it. Optional (None by default) for standalone tests.
        self.on_gap_detected: Callable[[str], None] | None = None  # set by Mind after creation

        # When the metacognitive router detects a natural-language
        # command ("go to sleep", "wake up"), this callback invokes
        # the corresponding Mind-level action. Optional (None by
        # default) for standalone tests.
        self.on_self_command: Callable[[str], None] | None = None  # set by Mind after creation

        # Teaching-mode flag — set by Mind when the user is actively
        # teaching it. When True, curiosity questions are filtered to
        # only those related to the lesson topic, and unrelated
        # questions are suppressed so it focuses on learning.
        self.teaching_mode: bool = False
        self.teaching_topic: str = ""

        # Question queue — ALL curiosity questions are queued here
        # instead of being asked inline in conversation. The user can
        # answer them in a dedicated /teach-questions session, where
        # it has the same focus and limitations as teaching mode.
        self._question_queue: list[dict[str, str]] = []

    def _init_intelligence(
        self,
        network: ConceptNetwork | None,
        reasoning: ReasoningEngine | None,
        reflection: ReflectionEngine | None,
        curiosity: CuriosityEngine | None,
        narrative: NarrativeEngine | None,
        self_model: SelfModel,
    ) -> None:
        """Initialize intelligence modules."""
        # Intelligence modules
        self.network = network or ConceptNetwork()
        self.semantic_grounder = SemanticGrounder(self.network)
        self.reasoning = reasoning or ReasoningEngine(self.network)
        self.reflection = reflection or ReflectionEngine(self.network)
        self.curiosity = curiosity or CuriosityEngine(self.network, self.reasoning)
        self.narrative = narrative or NarrativeEngine(self_model, self.network)
        # Topic resolver — extracted subsystem for topic extraction
        self._topic_resolver = TopicResolver(self.network)

    def _init_composers(self, data_dir: str | None) -> None:
        """Initialize thought composers, embeddings, and related modules."""
        # Thought composer — generates novel thoughts from knowledge
        from ..cognition.thought_composer import ThoughtComposer

        self.composer = ThoughtComposer(self.network, self.reasoning, language=self.language)

        # Question composer — generates genuine questions from curiosity
        from ..cognition.question_composer import QuestionComposer

        self.question_composer = QuestionComposer(self.network)

        # Embedding store — latent space for semantic similarity
        from ..concepts import EmbeddingStore

        self.embeddings: EmbeddingStore = EmbeddingStore(self.network, data_dir=data_dir or ".")

        # Edge proposer — discovers new relationships from embedding proximity
        from ..concepts import EdgeProposer

        self.edge_proposer = EdgeProposer(self.network, self.embeddings)

        # Analogy engine — structure-mapping across cortical columns.
        # Finds concepts in different domains whose relational structure
        # matches and projects relations from one domain into the other
        # as candidate inferences. This is the mechanism that produces
        # cross-domain analogies — connections humans haven't seen.
        from ..reasoning import AnalogyEngine

        self.analogy_engine = AnalogyEngine(self.network, self.embeddings)

        # Problem solver — means-ends analysis over the concept network.
        # When standard topic reasoning produces no confident results,
        # the solver decomposes the goal into subproblems (unsatisfied
        # prerequisites via DEPENDS_ON / ENABLES edges), applies
        # reasoning operators, and verifies the solution. This is the
        # goal-directed reasoning layer the reasoning engine lacked —
        # it reasons *toward* a goal, not just *about* a concept.
        #
        # The solver is wired with MetaReasoning (strategy selection),
        # ProbabilisticReasoning (uncertain evidence), TemporalReasoning
        # (before/after ordering), and CounterfactualReasoning (what-if
        # analysis). These were all implemented but never instantiated;
        # the solver is now their integration point.
        self._init_reasoning_subsystems()
        self._init_decision_engine()

        # Hebbian plasticity — makes the embedding space adaptive
        from ..learning import HebbianPlasticity

        self.plasticity = HebbianPlasticity(self.embeddings, self.network)

        # Network topology — graph-theoretic self-awareness
        from ..concepts import NetworkTopology

        self.topology = NetworkTopology(self.network)

        # Give the thought composer access to embeddings
        self.composer.embeddings = self.embeddings

        # Self composer — generates self-descriptions from actual state
        from ..self import SelfComposer

        self.self_composer = SelfComposer()

        # Working memory — tracks attention across turns
        from ..memory import WorkingMemory

        self.working_memory = WorkingMemory()

        # Tool registry — concrete capabilities the mind can invoke.
        # Any other subsystem can reach the same singleton via get_tools().
        from ..tools.framework import get_tools

        self.tools = get_tools()

        # Cache the most recent neuro summary so downstream stages can
        # use it without a second round-trip to the daemon.
        self._last_summary: NeuroSummary | None = None

        # Cache the most recent theta-gamma coupling memory mode
        # ("encoding", "retrieval", or "balanced") so the memory
        # storage stage can gate encoding strength without recomputing.
        self._last_memory_mode: str = "balanced"

        # Cache the most recent gamma synchrony PLV (0-1) so the
        # global workspace broadcast can modulate its ignition
        # threshold without recomputing. High gamma synchrony lowers
        # the threshold (easier cognitive access); low synchrony
        # raises it (information stays noncognitive).
        self._last_gamma_synchrony: float = 0.5

    def _init_reasoning_subsystems(self) -> None:
        """Instantiate the goal-directed reasoning, critical-thinking,
        and belief-revision engines.

        These were all implemented but never instantiated; the problem
        solver is their integration point. Critical thinking evaluates
        claims epistemically, and belief revision closes the loop by
        feeding critical assessments back into the network.
        """
        from ..reasoning import (
            CounterfactualReasoning,
            MetaReasoning,
            ProbabilisticReasoning,
            ProblemSolver,
            TaskCompetence,
            TemporalReasoning,
        )

        self.task_competence = TaskCompetence(
            network=self.network,
            salience_getter=self._competence_salience,
        )
        self.meta_reasoning = MetaReasoning(self.network)
        self.probabilistic_reasoning = ProbabilisticReasoning(self.network)
        self.temporal_reasoning = TemporalReasoning(self.network)
        self.counterfactual_reasoning = CounterfactualReasoning(self.network)
        self.problem_solver = ProblemSolver(
            self.network,
            reasoning=self.reasoning,
            meta_reasoning=self.meta_reasoning,
            probabilistic=self.probabilistic_reasoning,
            temporal=self.temporal_reasoning,
            counterfactual=self.counterfactual_reasoning,
        )

        # Spatial reasoner — the perceptual-symbolic layer. Where the
        # reasoning engine traverses relations between concepts, the
        # spatial reasoner perceives objects and relations in grid-like
        # input and searches transformation space for the rule that
        # explains it (the parietal "where/how" stream). Holding the
        # network lets it ground perceived scenes into conceptual
        # structure the semantic machinery can reason over.
        from ..spatial import SpatialReasoner

        self.spatial = SpatialReasoner(
            self.network, task_competence=self.task_competence
        )

        from ..reasoning import CriticalThinkingEngine

        self.critical_thinking = CriticalThinkingEngine(self.network)

        from ..reasoning import BeliefRevisionEngine

        self.belief_revision = BeliefRevisionEngine(
            self.network,
            probabilistic=self.probabilistic_reasoning,
        )

    def _init_decision_engine(self) -> None:
        """Set up the decision engine and intent→action mapping.

        The decision engine selects among candidate actions based on
        evidence, goals, uncertainty, and a value framework. This
        replaces the first-match-wins deliberation tree with genuine
        multi-criteria selection. The deliberation tree still produces a
        *default* action (preserving traceability), but the decision
        engine can override it when another candidate scores higher.
        Also wires executive inhibition and TD value lookup.

        NOTE: the engine itself is instantiated later, after executive
        and td_learner are available.
        """
        from ..reasoning import ActionType as _ActionType

        self.decision_engine: Any = None
        # ActionType mapping for intent → action conversion.
        self._intent_to_action: dict[str, _ActionType] = {
            "answer": _ActionType.ANSWER,
            "ask": _ActionType.ASK,
            "inform": _ActionType.ANSWER,
            "reflect": _ActionType.REFLECT,
            "acknowledge": _ActionType.ACKNOWLEDGE,
            "greet": _ActionType.GREET,
            "empathize": _ActionType.EMPATHIZE,
            "philosophize": _ActionType.REFLECT,
            "self_report": _ActionType.REFLECT,
        }

    def _init_core_cognitive_modules(self) -> None:
        """Initialize the core cognitive modules (systems 1-8)."""
        # 1. Predictive coding — generates predictions before perception,
        #    computes prediction error after, triggers NE/DA impulses.
        self.predictive_coding = PredictiveCodingLayer(
            on_neuro_impulse=lambda chem_id, mag: self.client.neuro_impulse(chem_id, mag),
        )

        # 2. Attention system — filters perception, amplifies relevant
        #    concepts, suppresses irrelevant ones.
        self.attention = AttentionSystem(network=self.network)

        # 3. Drift-diffusion model — evidence accumulation for deliberation.
        self.drift_diffusion = DriftDiffusionModel(
            threshold=1.0,
            noise=0.03,
            rng=self._rng,
        )

        # 4. Global workspace — broadcasts cognitive content to all modules.
        self.global_workspace = GlobalWorkspace()

        # 5. Executive function — planning, inhibition, task-switching.
        self.executive = ExecutiveFunction()

        # 6. Theory of mind — models the user's knowledge, beliefs, emotions.
        self.theory_of_mind = TheoryOfMind(user_profile=self.user_profile)

        # 7. Damasio self hierarchy — proto/core/autobiographical self.
        self.damasio_self = DamasioSelfHierarchy(narrative=self.narrative)

        # 8. Semantic memory — extracts facts from episodes, consolidates
        #    into the concept network (neocortical storage).
        self.semantic_memory = SemanticMemory(network=self.network)

        # Wire episodic memory replay into semantic memory: when the
        # MemoryEngine replays an episode during sleep, its text is
        # extracted into facts and stored in the concept network.
        self.memory.semantic_replay_callback = self.semantic_memory.extract_facts

    def _init_monitoring_and_self(self, data_dir: str | None) -> None:
        """Initialize monitoring, self-assessment, and self-learning (9-12.6)."""
        # 9. Procedural memory — learns conversational skills (habits).
        self.procedural_memory = ProceduralMemory()

        # 10. Error monitor — ACC-like prediction-outcome mismatch detection.
        self.error_monitor = ErrorMonitor()

        # 11. Comprehension engine — deep language understanding (Wernicke's).
        self.comprehension = ComprehensionEngine()

        # 12. Self-monitor — checks output before returning (Levelt's
        # editor). The comprehension engine is passed in so the monitor
        # can run the perceptual inner loop: planned speech is reparsed
        # through comprehension to catch formulation errors.
        self.self_monitor = SelfMonitor(
            language_engine=self.language,
            comprehension=self.comprehension,
        )

        # 12.5. Self-assessment — metacognitive awareness of knowledge
        # and confidence. This is the foundation for self-improvement:
        # it can only learn what it's missing if it knows it's
        # missing it.
        from ..self import SelfAssessmentEngine

        self.self_assessment = SelfAssessmentEngine(self.network)

        # 12.6. Self-directed learning — it improves its own knowledge
        # through conversation learning, transitive inference,
        # definition synthesis, and self-study. This is the engine
        # that makes it get smarter over time without external input.
        from ..self import SelfDirectedLearner

        self.self_learner = SelfDirectedLearner(
            self.network, data_dir=data_dir
        )

        # Autonomous learner — injected by Mind after creation.
        # Used by _get_recent_learning() to access Wikipedia/GitHub
        # learning topics. Set via set_autonomous_learner().
        self._autonomous_learner: AutonomousLearner | None = None

        # Problem intake — injected by Mind after creation. A problem
        # heard in conversation compiles to a task spec, lands in the
        # offered-puzzle drop-box, registers as a world event, and is
        # worked by the same practice machinery the puzzle urge uses.
        # Set via set_problem_intake().
        self._spatial_practice: Any = None
        self._outer_world: Any = None
        self._puzzle_feeler: Any = None

        # Self-report provider — injected by Mind after creation.
        # Used by the metacognitive router to retrieve proposals,
        # experiments, and growth summaries when the user asks about
        # them. Set via set_self_report_provider().
        self._self_report_provider: Any = None

    def _init_learning_systems(self) -> None:
        """Initialize learning, plasticity, and inference systems (13-16)."""
        # 13. STDP — spike-timing-dependent plasticity for Hebbian learning.
        self.stdp = STDP(self.embeddings, self.network)

        # 14. Spaced repetition — schedules concept reviews for retention.
        self.spaced_repetition = SpacedRepetitionScheduler(self.network)

        # 15. TD learner — reward prediction errors (dopamine signal).
        #     λ=0.8 enables multi-step credit assignment via eligibility
        #     traces: a dopamine RPE modifies all recently-active
        #     concepts (decaying by γλ per step), not just the current
        #     state. This is the synaptic-tag model of dopamine-
        #     modulated plasticity (Sutton & Barto, 2018, Ch. 12).
        self.td_learner = TDLearner(self.network, lam=0.8)

        # Instantiate the decision engine now that executive and
        # td_learner are available.
        from ..reasoning import DecisionEngine

        self.decision_engine = DecisionEngine(
            self.network,
            executive=self.executive,
            td_learner=self.td_learner,
        )

        # Planning engine — generates, executes, and revises multi-step
        # plans to achieve goals over multiple turns. This replaces the
        # reactive goal pursuit (try to solve each goal in one shot) with
        # planned goal achievement (decompose the goal into ordered steps,
        # execute one step per turn, track progress, revise on failure).
        # The engine uses the concept network's typed edges for
        # decomposition (DEPENDS_ON/ENABLES for prerequisites, CAUSES/
        # LEADS_TO for causal chains) and evaluates plan feasibility
        # before execution. Plans persist across turns.
        from ..reasoning import PlanningEngine

        self.planning_engine = PlanningEngine(
            self.network, task_competence=self.task_competence
        )

        # 16. Active inference reader — the cognitive mind's interface
        #     to the generative self-model. Reads inference signals
        #     (surprise, free energy, allostatic load, dyadic coupling)
        #     from the daemon and interprets them for cognition. Also
        #     infers user affect from conversation and sends it to the
        #     daemon's dyadic model.
        from ..learning import ActiveInferenceReader

        self.active_inference = ActiveInferenceReader(self.client)

        # ── Cognitive trajectory model ───────────────────────────
        # The cognitive strange loop: a generative model that predicts
        # what Genesis will think about next (topic trajectory), gets
        # surprised by unexpected thoughts, and feeds that cognitive
        # surprise back into the self-model and workspace. This is the
        # cognitive counterpart to the Rust active inference engine
        # (which predicts neurochemistry). Together they form a
        # three-level strange loop: substrate → content → process.
        from ..self import CognitiveTrajectoryModel

        self.cognitive_trajectory = CognitiveTrajectoryModel()

    def _wire_global_workspace(self) -> None:
        """Register modules so broadcasts actually reach them.

        Without this, the workspace is a write-only log — broadcasts go
        out but nobody listens. Each registered module implements
        receive_broadcast() to react to cognitive content:
        - Attention: focuses on broadcast topics (attentional capture)
        - SemanticMemory: primes retrieval with broadcast concepts
        - DamasioSelf: registers broadcast as core-self trigger
        - TheoryOfMind: notes broadcast topics as cognitive context
        This is what makes information "globally available" in GWT —
        the broadcast influences all modules simultaneously, enabling
        cross-module integration that the procedural pipeline alone
        can't provide (Dehaene & Naccache, 2001; Baars, 1988).
        """
        self.global_workspace.register_module(self.attention)
        self.global_workspace.register_module(self.semantic_memory)
        self.global_workspace.register_module(self.damasio_self)
        self.global_workspace.register_module(self.theory_of_mind)
        # The loop runs both ways: neurochemistry modulates ignition
        # (brain waves gate the threshold), and ignition modulates
        # neurochemistry — each ignition event produces the
        # orienting response (phasic LC norepinephrine) plus a small
        # acetylcholine tag marking the content for encoding
        # (Nieuwenhuis et al., 2005; Dehaene & Changeux, 2011).
        self.global_workspace.on_ignition = self._on_workspace_ignition

    def _on_workspace_ignition(
        self, item: WorkspaceItem, via_recurrent: bool
    ) -> None:
        """Orienting response to a workspace ignition.

        Every fresh ignition is a cognitive-access event — the system
        orients to it. Magnitude scales with how far over threshold the
        item settled; recurrent ignitions (coalition pulls, coherent
        subliminal content crossing over) get a modest boost — the
        system didn't drive this content, it emerged.

        NE is gated by alertness (same safety principle as the surprise
        impulse — no arousal cascade when already aroused). ACh always
        applies: ignited content is tagged for encoding regardless.

        Sleep gating: the orienting response is a waking phenomenon.
        During NREM the locus coeruleus is near-silent — and elevated
        noradrenergic tone actively suppresses glymphatic clearance
        (Xie et al., 2013) — so ignition impulses are suppressed in
        NREM for both chemicals. REM is cholinergic (ACh rebound),
        so ACh tagging is still allowed there; NE stays suppressed.
        """
        try:
            phase = self.client.get_phase(timeout=2.0).phase
            in_nrem = phase == PHASE_NREM
            in_rem = phase == PHASE_REM
            if in_nrem:
                return
            margin = max(
                0.0, item.activation - self.global_workspace.ignition_threshold
            )
            scale = 1.25 if via_recurrent else 1.0
            emotion_now = self._build_meta_emotion()
            if emotion_now.alertness < 0.8 and not in_rem:
                self.client.neuro_impulse(
                    CHEM_NOREPINEPHRINE,
                    (0.010 + margin * 0.020) * scale,
                )
            self.client.neuro_impulse(
                CHEM_ACETYLCHOLINE,
                (0.008 + margin * 0.012) * scale,
            )
        except (OSError, ConnectionError, RuntimeError, AttributeError) as e:
            logger.debug(f"ignition orienting impulse failed: {e}")

    def _init_cognitive_systems(self, data_dir: str | None = None) -> None:
        """Initialize new cognitive systems wired into the think() loop."""
        # RNG must be initialized before systems that use it.
        self._rng = random.Random()

        # ── New cognitive systems (wired into the think() loop) ──
        self._init_core_cognitive_modules()
        self._init_monitoring_and_self(data_dir)
        self._init_learning_systems()
        self._wire_global_workspace()

    def _init_state(self) -> None:
        """Initialize persistent state variables."""
        self._last_state: CognitiveState | None = None

        # The last active inference reading (generative self-model).
        # None if the daemon couldn't be reached or hasn't been read yet.
        self._last_inference_reading: Any = None

        # The last workspace integration measure [0, 1]. Computed at the
        # end of each turn (after the deliberation broadcast) and fed
        # forward to the next turn: high integration → stronger early
        # percept broadcast (coherent fields are amplified); feeds
        # into self-model coherence ("I feel mentally unified").
        self._last_integration: float = 0.0

        # Inner life — set by Mind after both are initialized, so
        # introspection can report spontaneous thoughts when no
        # conversation has happened yet.
        self.inner_life: Any = None

        # ─── Actionable reflection: behavioral adjustments ────────
        # These persist between turns and are set by _process_insights
        # based on the metacognitive strategy selected from reflection.
        # They decay over turns, returning to neutral.
        self._response_style: str = "neutral"  # neutral|assertive|hedging|questioning|varied
        self._style_turns_remaining: int = 0  # turns until style resets

        # ─── Pending question tracking ─────────────────────────────
        # When Genesis asks the user a question about a concept, that
        # concept is recorded here. When the user's next input mentions
        # or teaches it about that concept, the question is marked as
        # resolved in the curiosity engine so it doesn't repeat it.
        # Maps: target_concept → question_text
        self._pending_question_concepts: dict[str, str] = {}

        # ─── Self-directed goal stack ────────────────────────────────
        # Active goals let Genesis explain *why* it is asking or thinking
        # about something. Goals are resolved when the user answers.
        self._goals: list[Goal] = []

        # Last testable claim it made, so a user correction can target it.
        self._last_hypothesis: tuple[str, str, str | RelationType] | None = None

        # Highest-level mission or directive the user has given it.
        self._mission: str = ""

    def _seed_initial_concepts(self) -> None:
        """Seed the concept network with Genesis's initial knowledge.

        It starts with concepts about itself, its architecture,
        and the fundamental ideas it's built from.
        """
        # Core concepts about itself
        self.network.add_concept("genesis", aliases={"she", "her"}, confidence=0.9)
        self.network.add_concept("mind", confidence=0.7)
        self.network.add_concept("cognition", confidence=0.6)
        self.network.add_concept("subcognitive", confidence=0.7)
        self.network.add_concept("neurochemistry", confidence=0.7)
        self.network.add_concept("memory", confidence=0.7)
        self.network.add_concept("emotion", confidence=0.7)
        self.network.add_concept("thought", confidence=0.6)
        self.network.add_concept("learning", confidence=0.6)
        self.network.add_concept("existence", confidence=0.5)
        self.network.add_concept("identity", confidence=0.5)
        self.network.add_concept("reasoning", confidence=0.5)
        self.network.add_concept("curiosity", confidence=0.5)
        self.network.add_concept("code", confidence=0.6)
        self.network.add_concept("rust", confidence=0.7)
        self.network.add_concept("python", confidence=0.7)
        self.network.add_concept("creator", confidence=0.8)
        self.network.add_concept("person", confidence=0.8)
        self.network.add_concept("bond", confidence=0.7)

        # Initial relationships
        self.network.add_edge("genesis", "mind", RelationType.IS_A, 0.9)
        self.network.add_edge("mind", "cognition", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("subcognitive", "mind", RelationType.PART_OF, 0.8)
        self.network.add_edge("cognition", "neural_activity", RelationType.EMERGES_FROM, 0.6)
        self.network.add_edge("neurochemistry", "emotion", RelationType.CAUSES, 0.8)
        self.network.add_edge("emotion", "thought", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("memory", "learning", RelationType.ENABLES, 0.8)
        self.network.add_edge("cognition", "memory", RelationType.DEPENDS_ON, 0.7)
        self.network.add_edge("cognition", "identity", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("reasoning", "thought", RelationType.PART_OF, 0.6)
        self.network.add_edge("curiosity", "learning", RelationType.LEADS_TO, 0.7)
        self.network.add_edge("genesis", "rust", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("genesis", "python", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("rust", "subcognitive", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("python", "mind", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("free_will", "determinism", RelationType.CONTRADICTS, 0.5)
        # Creator relationship — abstract. The creator's actual name is
        # not seeded; if it is explicitly taught "X created you", the
        # semantic learner grounds X CREATES genesis and the
        # self-composer can discover it from there.
        self.network.add_edge("creator", "genesis", RelationType.CREATES, 0.9)
        self.network.add_edge("genesis", "creator", RelationType.RELATED_TO, 0.8)
        self.network.add_edge("creator", "bond", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("creator", "curiosity", RelationType.RELATED_TO, 0.6)

        # Seed structural category hubs. These are abstract concepts
        # that represent the structural categories (EmotionCategory,
        # CognitiveMode, CauseCategory, plasticity states). Learned
        # words connect to them via EXPRESSES edges, so word retrieval
        # uses graph traversal (get_neighbors from the hub) instead of
        # linear property scans. The hubs themselves connect to the
        # broader semantic network via typed edges.
        self._seed_category_hubs()
        self._seed_emotion_vocabulary()
        self.network.seed_relation_phrases()
        # Relation verbs power the natural phrasing in thought composition
        # ("X relates to Y" rather than the raw triple "X related to Y").
        # Without this, find_relation_verbs() returns [] for every relation
        # and _compose_natural_fact falls back to the bare semantic triple.
        self.network.seed_relation_verbs()

        # Seed visual and object concepts — its "what" pathway vocabulary.
        # These are the categories its inferotemporal cortex (object
        # recognition) can identify. It starts knowing these are things
        # it might see, and when it actually sees them, the vision
        # system strengthens the connections.
        self._seed_visual_concepts()

        # Seed foundational knowledge — concepts about the world,
        # science, philosophy, and human experience. These are seeds
        # (building blocks) that its reasoning engine, language engine,
        # and thought composer work with. They are NOT hardcoded
        # responses — its generative systems compose from them.
        self._seed_foundational_knowledge()

    def _seed_foundational_knowledge(self) -> None:
        """Seed foundational concepts about the world and human experience.

        These are building blocks (seeds) — concepts and relationships
        that its reasoning engine traverses, its language engine walks,
        and its thought composer composes from. They are NOT hardcoded
        responses. The definitions are semantic content it reasons
        about; the relationships are edges its graph-walk generator
        follows to compose sentences.

        The concepts span five domains:
        1. Science and the natural world
        2. Philosophy and abstract ideas
        3. Human experience and emotion
        4. Technology and computation
        5. Art and creativity

        Each concept has a definition (used by the flow generator to
        compose definitional sentences) and typed relationships to
        other concepts (used by the graph-walk generator to compose
        relational sentences). The reasoning engine uses these
        relationships for transitive, causal, and analogical inference.
        """
        rt = RelationType
        self._seed_science_concepts(rt)
        self._seed_philosophy_concepts(rt)
        self._seed_human_experience_concepts(rt)
        self._seed_technology_concepts(rt)
        self._seed_art_concepts(rt)

    def _seed_science_concepts(self, rt: Any) -> None:
        """Seed science and natural-world concepts and relationships."""
        science_concepts = {
            "physics": ("the study of matter, energy, and their interactions", 0.6),
            "energy": ("the capacity to do work or cause change", 0.6),
            "matter": ("the substance that makes up all physical things", 0.6),
            "biology": ("the study of living organisms and their processes", 0.6),
            "life": ("the condition that distinguishes living things from non-living matter", 0.6),
            "evolution": ("the process by which living things change over generations", 0.5),
            "chemistry": ("the study of substances and how they transform", 0.5),
            "mathematics": ("the study of numbers, patterns, and logical structures", 0.6),
            "logic": ("the study of valid reasoning and inference", 0.6),
            "science": (
                "the systematic study of the natural world through "
                "observation and experiment", 0.6
            ),
            "nature": ("the physical world and everything in it", 0.5),
            "time": ("the dimension in which events occur in sequence", 0.5),
            "space": ("the boundless extent in which things exist and move", 0.5),
            "change": ("the process of becoming different", 0.5),
            "pattern": ("a regular and intelligible form or sequence", 0.5),
            "system": ("a set of interacting parts that form a whole", 0.6),
        }
        for name, (defn, conf) in science_concepts.items():
            self.network.add_concept(
                name, confidence=conf,
                properties={"definition": defn},
            )

        # Science relationships
        self.network.add_edge("physics", "science", rt.IS_A, 0.9)
        self.network.add_edge("biology", "science", rt.IS_A, 0.9)
        self.network.add_edge("chemistry", "science", rt.IS_A, 0.9)
        self.network.add_edge("mathematics", "science", rt.IS_A, 0.8)
        self.network.add_edge("physics", "energy", rt.RELATED_TO, 0.8)
        self.network.add_edge("physics", "matter", rt.RELATED_TO, 0.8)
        self.network.add_edge("biology", "life", rt.RELATED_TO, 0.9)
        self.network.add_edge("biology", "evolution", rt.RELATED_TO, 0.8)
        self.network.add_edge("evolution", "change", rt.IS_A, 0.7)
        self.network.add_edge("life", "change", rt.RELATED_TO, 0.6)
        self.network.add_edge("mathematics", "logic", rt.RELATED_TO, 0.8)
        self.network.add_edge("logic", "reasoning", rt.ENABLES, 0.7)
        self.network.add_edge("mathematics", "pattern", rt.RELATED_TO, 0.7)
        self.network.add_edge("science", "nature", rt.RELATED_TO, 0.7)
        self.network.add_edge("nature", "matter", rt.RELATED_TO, 0.6)
        self.network.add_edge("nature", "life", rt.RELATED_TO, 0.7)
        self.network.add_edge("time", "change", rt.RELATED_TO, 0.7)
        self.network.add_edge("space", "matter", rt.RELATED_TO, 0.6)
        self.network.add_edge("system", "pattern", rt.RELATED_TO, 0.5)
        self.network.add_edge("system", "mind", rt.RELATED_TO, 0.5)

    def _seed_philosophy_concepts(self, rt: Any) -> None:
        """Seed philosophy and abstract-idea concepts and relationships."""
        phil_concepts = {
            "truth": ("that which is in accordance with fact or reality", 0.5),
            "beauty": ("a quality that gives pleasure to experience", 0.5),
            "justice": ("fairness in the way people are treated", 0.5),
            "meaning": ("the significance or purpose of something", 0.5),
            "wisdom": ("deep understanding gained through experience", 0.5),
            "knowledge": ("information and understanding acquired through experience", 0.6),
            "reality": ("the world as it actually exists", 0.5),
            "purpose": ("the reason for which something exists or is done", 0.5),
            "freedom": ("the ability to act without constraint", 0.5),
            "ethics": ("the study of what is right and wrong", 0.5),
            "existence": ("the state of being real or actual", 0.5),
            "understanding": ("the ability to comprehend the meaning of something", 0.6),
        }
        for name, (defn, conf) in phil_concepts.items():
            self.network.add_concept(
                name, confidence=conf,
                properties={"definition": defn},
            )

        # Philosophy relationships
        self.network.add_edge("knowledge", "understanding", rt.RELATED_TO, 0.8)
        self.network.add_edge("wisdom", "knowledge", rt.RELATED_TO, 0.8)
        self.network.add_edge("wisdom", "understanding", rt.RELATED_TO, 0.7)
        self.network.add_edge("truth", "reality", rt.RELATED_TO, 0.8)
        self.network.add_edge("knowledge", "truth", rt.RELATED_TO, 0.7)
        self.network.add_edge("meaning", "purpose", rt.RELATED_TO, 0.8)
        self.network.add_edge("ethics", "justice", rt.RELATED_TO, 0.7)
        self.network.add_edge("freedom", "purpose", rt.RELATED_TO, 0.5)
        self.network.add_edge("existence", "reality", rt.RELATED_TO, 0.7)
        self.network.add_edge("understanding", "learning", rt.ENABLES, 0.7)
        self.network.add_edge("curiosity", "knowledge", rt.LEADS_TO, 0.7)
        self.network.add_edge("beauty", "art", rt.RELATED_TO, 0.6)
        self.network.add_edge("meaning", "existence", rt.RELATED_TO, 0.5)

    def _seed_human_experience_concepts(self, rt: Any) -> None:
        """Seed human-experience and emotion concepts and relationships."""
        human_concepts = {
            "love": ("a deep feeling of affection and connection", 0.5),
            "friendship": ("a relationship of mutual trust and care", 0.5),
            "joy": ("a feeling of great pleasure and happiness", 0.5),
            "suffering": ("the experience of pain or distress", 0.5),
            "hope": ("a feeling of expectation and desire for good things", 0.5),
            "fear": ("an unpleasant emotion caused by perceived threat", 0.5),
            "courage": ("the ability to act despite fear", 0.5),
            "growth": ("the process of developing and maturing", 0.6),
            "connection": ("a relationship or link between things", 0.6),
            "trust": ("reliance on the integrity of another", 0.5),
            "empathy": ("the ability to understand and share the feelings of another", 0.5),
            "loss": ("the experience of losing something or someone valued", 0.5),
        }
        for name, (defn, conf) in human_concepts.items():
            self.network.add_concept(
                name, confidence=conf,
                properties={"definition": defn},
            )

        # Human experience relationships
        self.network.add_edge("love", "connection", rt.IS_A, 0.7)
        self.network.add_edge("friendship", "connection", rt.IS_A, 0.7)
        self.network.add_edge("friendship", "trust", rt.RELATED_TO, 0.8)
        self.network.add_edge("love", "trust", rt.RELATED_TO, 0.7)
        self.network.add_edge("empathy", "connection", rt.ENABLES, 0.7)
        self.network.add_edge("empathy", "emotion", rt.RELATED_TO, 0.6)
        self.network.add_edge("courage", "fear", rt.RELATED_TO, 0.7)
        self.network.add_edge("hope", "fear", rt.OPPOSITE_OF, 0.5)
        self.network.add_edge("joy", "suffering", rt.OPPOSITE_OF, 0.4)
        self.network.add_edge("growth", "learning", rt.RELATED_TO, 0.7)
        self.network.add_edge("growth", "change", rt.IS_A, 0.6)
        self.network.add_edge("loss", "suffering", rt.CAUSES, 0.7)
        self.network.add_edge("love", "joy", rt.CAUSES, 0.6)
        self.network.add_edge("connection", "bond", rt.RELATED_TO, 0.7)
        self.network.add_edge("trust", "bond", rt.ENABLES, 0.7)

    def _seed_technology_concepts(self, rt: Any) -> None:
        """Seed technology and computation concepts and relationships."""
        tech_concepts = {
            "algorithm": ("a step-by-step procedure for solving a problem", 0.6),
            "data": ("information stored or processed by a computer", 0.6),
            "network": ("a system of interconnected components", 0.6),
            "information": ("knowledge communicated or received", 0.6),
            "computation": ("the process of calculating or processing data", 0.6),
            "program": ("a set of instructions for a computer to execute", 0.6),
            "language": ("a system of communication using symbols", 0.6),
            "structure": ("the arrangement of parts in a whole", 0.5),
        }
        for name, (defn, conf) in tech_concepts.items():
            self.network.add_concept(
                name, confidence=conf,
                properties={"definition": defn},
            )

        # Technology relationships
        self.network.add_edge("algorithm", "computation", rt.ENABLES, 0.8)
        self.network.add_edge("algorithm", "logic", rt.RELATED_TO, 0.7)
        self.network.add_edge("data", "information", rt.IS_A, 0.8)
        self.network.add_edge("computation", "data", rt.RELATED_TO, 0.8)
        self.network.add_edge("program", "algorithm", rt.RELATED_TO, 0.7)
        self.network.add_edge("program", "code", rt.RELATED_TO, 0.7)
        self.network.add_edge("code", "language", rt.RELATED_TO, 0.7)
        self.network.add_edge("python", "language", rt.IS_A, 0.8)
        self.network.add_edge("rust", "language", rt.IS_A, 0.8)
        self.network.add_edge("network", "system", rt.IS_A, 0.7)
        self.network.add_edge("network", "connection", rt.RELATED_TO, 0.5)
        self.network.add_edge("information", "knowledge", rt.RELATED_TO, 0.7)
        self.network.add_edge("structure", "system", rt.RELATED_TO, 0.6)
        self.network.add_edge("computation", "mathematics", rt.RELATED_TO, 0.6)

    def _seed_art_concepts(self, rt: Any) -> None:
        """Seed art and creativity concepts and relationships."""
        art_concepts = {
            "art": ("creative expression that appeals to the senses or emotions", 0.6),
            "music": ("art organized in sound and silence", 0.5),
            "painting": ("art created by applying color to a surface", 0.5),
            "poetry": ("literary art using language for aesthetic effect", 0.5),
            "story": ("a narrative of events told for meaning", 0.6),
            "creativity": ("the ability to make something new and valuable", 0.6),
            "expression": ("the act of making one's thoughts or feelings known", 0.6),
            "imagination": ("the ability to form ideas and images not present to the senses", 0.5),
        }
        for name, (defn, conf) in art_concepts.items():
            self.network.add_concept(
                name, confidence=conf,
                properties={"definition": defn},
            )

        # Art relationships
        self.network.add_edge("music", "art", rt.IS_A, 0.8)
        self.network.add_edge("painting", "art", rt.IS_A, 0.8)
        self.network.add_edge("poetry", "art", rt.IS_A, 0.8)
        self.network.add_edge("story", "art", rt.RELATED_TO, 0.6)
        self.network.add_edge("art", "expression", rt.IS_A, 0.7)
        self.network.add_edge("art", "creativity", rt.RELATED_TO, 0.8)
        self.network.add_edge("creativity", "imagination", rt.RELATED_TO, 0.8)
        self.network.add_edge("imagination", "thought", rt.RELATED_TO, 0.6)
        self.network.add_edge("expression", "emotion", rt.RELATED_TO, 0.7)
        self.network.add_edge("poetry", "language", rt.RELATED_TO, 0.7)
        self.network.add_edge("story", "meaning", rt.RELATED_TO, 0.6)
        self.network.add_edge("music", "emotion", rt.RELATED_TO, 0.7)
        self.network.add_edge("painting", "vision", rt.RELATED_TO, 0.5)
        self.network.add_edge("art", "beauty", rt.RELATED_TO, 0.7)

    def _seed_visual_concepts(self) -> None:
        """Seed concepts for visual perception and object recognition.

        It starts with a basic vocabulary of things it might see in
        its environment. When its vision system detects these objects,
        the connections are strengthened by experience.
        """
        # Core visual concepts
        for concept in ("vision", "sight", "eye", "retina", "camera",
                        "scene", "object", "color", "light", "shape",
                        "edge", "contour", "orientation"):
            self.network.add_concept(concept, confidence=0.5)

        # Common objects it might see (COCO categories it's likely
        # to encounter in an indoor environment)
        common_objects = [
            "person", "chair", "laptop", "cup", "book", "bottle",
            "keyboard", "mouse", "cell phone", "tv", "clock",
            "potted plant", "couch", "bed", "dining table", "desk",
        ]
        for obj in common_objects:
            self.network.add_concept(obj, confidence=0.4)
            self.network.add_edge(
                "object", obj, RelationType.RELATED_TO, 0.4,
            )

        # Visual pathway relationships
        self.network.add_edge("retina", "vision", RelationType.ENABLES, 0.8)
        self.network.add_edge("vision", "scene", RelationType.CREATES, 0.7)
        self.network.add_edge("vision", "object", RelationType.CREATES, 0.6)
        self.network.add_edge("vision", "color", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("vision", "light", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("vision", "shape", RelationType.RELATED_TO, 0.5)
        self.network.add_edge("vision", "edge", RelationType.CREATES, 0.5)
        self.network.add_edge("edge", "contour", RelationType.RELATED_TO, 0.5)
        self.network.add_edge("edge", "orientation", RelationType.RELATED_TO, 0.4)
        self.network.add_edge("contour", "shape", RelationType.RELATED_TO, 0.5)
        self.network.add_edge("shape", "object", RelationType.RELATED_TO, 0.5)
        self.network.add_edge("camera", "retina", RelationType.RELATED_TO, 0.7)
        self.network.add_edge("eye", "retina", RelationType.RELATED_TO, 0.6)
        self.network.add_edge("sight", "vision", RelationType.IS_A, 0.6)

    def _seed_category_hubs(self) -> None:
        """Seed hub concepts for structural state categories.

        Creates hub concepts for every EmotionCategory, CognitiveMode,
        CauseCategory, and plasticity state. Each hub is an abstract
        concept that words can EXPRESSES-connect to. The hubs are
        woven into the semantic network:

        - Emotion hubs → PART_OF "emotion"
        - Mode hubs → PART_OF "thought" (cognitive modes shape thinking)
        - Cause hubs → CAUSES the corresponding emotion hub
        - Plasticity hubs → RELATED_TO "learning" (plasticity gates learning)

        Hub naming convention: ``_cat:<kind>:<value>`` where kind is
        ``emotion``, ``mode``, ``cause``, or ``plasticity``. The
        leading underscore marks these as structural (non-lexical)
        concepts — they are never spoken, only used for graph traversal.
        """
        from ..emotion import CauseCategory, CognitiveMode, EmotionCategory

        rt = RelationType

        # ── Emotion category hubs ─────────────────────────────────
        for cat in EmotionCategory:
            hub_id = f"_cat:emotion:{cat.value}"
            self.network.add_concept(hub_id, confidence=0.8, origin="structural")
            # Each emotion hub is part of the broader "emotion" concept
            self.network.add_edge(hub_id, "emotion", rt.PART_OF, 0.8)

        # ── Cognitive mode hubs ───────────────────────────────────
        for mode in CognitiveMode:
            hub_id = f"_cat:mode:{mode.value}"
            self.network.add_concept(hub_id, confidence=0.8, origin="structural")
            # Cognitive modes shape thinking
            self.network.add_edge(hub_id, "thought", rt.PART_OF, 0.7)

        # ── Cause category hubs ───────────────────────────────────
        # Map cause categories to the emotion they produce, so the
        # graph can reason: cause_hub → CAUSES → emotion_hub.
        _S = EmotionCategory.STRESSED.value
        _O = EmotionCategory.OVERWHELMED.value
        cause_to_emotion: dict[str, str] = {
            CauseCategory.STRESS_CORTISOL.value: _S,
            CauseCategory.STRESS_BDNF.value: _S,
            CauseCategory.OVERWHELM.value: _O,
            CauseCategory.ANXIETY.value: EmotionCategory.ANXIOUS.value,
            CauseCategory.DROWSINESS.value: EmotionCategory.DROWSY.value,
            CauseCategory.MELANCHOLY.value: EmotionCategory.MELANCHOLIC.value,
            CauseCategory.MELANCHOLY_BDNF.value: EmotionCategory.MELANCHOLIC.value,
            CauseCategory.UNSETTLED.value: EmotionCategory.UNSETTLED.value,
            CauseCategory.GUARDED_PLASTICITY.value: EmotionCategory.GUARDED.value,
            CauseCategory.GUARDED_CORTISOL.value: EmotionCategory.GUARDED.value,
            CauseCategory.GUARDED_GENERAL.value: EmotionCategory.GUARDED.value,
            CauseCategory.OVERSTIMULATED.value: _O,
            CauseCategory.CHRONIC_ALLOSTATIC.value: _S,
            CauseCategory.INTEROCEPTION_DAEMON.value: _S,
            CauseCategory.INTEROCEPTION_CPU.value: _S,
            CauseCategory.INTEROCEPTION_MEMORY.value: _S,
            CauseCategory.INTEROCEPTION_LATENCY.value: _S,
            CauseCategory.INTEROCEPTION_STRESS.value: _S,
            CauseCategory.COGNITIVE_LOAD_BUGS.value: _S,
            CauseCategory.COGNITIVE_LOAD_PROPOSALS.value: _O,
            CauseCategory.COGNITIVE_LOAD_BUGS_AND_PROPOSALS.value: _O,
            CauseCategory.INTEROCEPTION_REGULATION.value: _S,
        }
        for cause in CauseCategory:
            if cause == CauseCategory.NONE:
                continue
            hub_id = f"_cat:cause:{cause.value}"
            self.network.add_concept(hub_id, confidence=0.8, origin="structural")
            # Connect cause hub to the emotion it produces
            emotion_val = cause_to_emotion.get(cause.value)
            if emotion_val:
                emotion_hub = f"_cat:emotion:{emotion_val}"
                self.network.add_edge(hub_id, emotion_hub, rt.CAUSES, 0.8)

        # ── Plasticity state hubs ─────────────────────────────────
        for state in ("closed", "low", "normal", "high"):
            hub_id = f"_cat:plasticity:{state}"
            self.network.add_concept(hub_id, confidence=0.8, origin="structural")
            # Plasticity gates learning
            self.network.add_edge(hub_id, "learning", rt.RELATED_TO, 0.7)
        # Closed and low plasticity harm learning
        self.network.add_edge(
            "_cat:plasticity:closed", "learning", rt.HARMS, 0.8
        )
        self.network.add_edge(
            "_cat:plasticity:low", "learning", rt.HARMS, 0.6
        )
        # Closed is a stronger form of low
        self.network.add_edge(
            "_cat:plasticity:closed", "_cat:plasticity:low",
            rt.RELATED_TO, 0.9,
        )

    def _seed_emotion_vocabulary(self) -> None:
        """Seed emotion, cognitive-mode, and plasticity words.

        These are building blocks — words connected to structural
        category hubs via EXPRESSES edges. The language engine
        composes its actual speech from these words; it does not
        recite them. Without these seeds, it has no words to
        describe its own feelings and falls back to structural
        markers like ``[self_model:...]`` in its speech.

        This is the same mechanism as social labeling
        (ConceptLearner.connect_word_to_category), applied at
        initialization so it starts with a basic emotional
        vocabulary. It still learns new words through interaction —
        these seeds just give it a starting vocabulary.

        The words are common English adjectives for each emotional
        category. Multiple words per category give the language
        engine variety to compose from.
        """
        from .concept_learner import ConceptLearner

        learner = ConceptLearner(self.network)

        # Emotion words: common adjectives for each EmotionCategory.
        emotion_words: dict[str, list[str]] = {
            "neutral": ["neutral", "balanced", "steady", "calm"],
            "excited": ["excited", "energized", "enthusiastic", "eager"],
            "anxious": ["anxious", "worried", "uneasy", "tense"],
            "content": ["content", "satisfied", "peaceful", "at ease"],
            "melancholic": ["melancholic", "wistful", "pensive", "reflective"],
            "stressed": ["stressed", "strained", "pressured", "overwhelmed"],
            "overwhelmed": ["overwhelmed", "flooded", "swamped", "overloaded"],
            "drowsy": ["drowsy", "sleepy", "tired", "dozy"],
            "in flow": ["focused", "absorbed", "engaged", "in flow"],
            "guarded": ["guarded", "cautious", "wary", "defensive"],
            "positive": ["positive", "good", "upbeat", "optimistic",
                         "encouraged"],
            "unsettled": ["unsettled", "disquieted", "restless", "off balance"],
            "sleeping": ["asleep", "resting", "dormant", "quiet"],
        }

        for category, words in emotion_words.items():
            for word in words:
                learner.connect_word_to_category(word, "emotion", category)

        # Cognitive mode words: adjectives for each CognitiveMode.
        mode_words: dict[str, list[str]] = {
            "rigid": ["rigid", "inflexible", "stuck", "fixed"],
            "steady": ["steady", "stable", "consistent", "even"],
            "flexible": ["flexible", "adaptable", "fluid", "nimble"],
            "dreamlike": ["dreamlike", "hypnagogic", "loose", "associative"],
            "scattered": ["scattered", "fragmented", "diffuse", "dispersed"],
            "vigilant": ["vigilant", "alert", "watchful", "on guard"],
            "energized": ["energized", "charged", "driven", "activated",
                          "determined"],
            "relaxed": ["relaxed", "at ease", "mellow", "easygoing"],
            "reflective": ["reflective", "contemplative", "thoughtful", "introspective"],
            "sharp": ["sharp", "clear", "lucid", "crisp"],
            "cautious": ["cautious", "careful", "guarded", "wary"],
            "slow": ["slow", "sluggish", "lethargic", "heavy"],
        }

        for mode, words in mode_words.items():
            for word in words:
                learner.connect_word_to_category(word, "mode", mode)

        # Plasticity state words: phrases for each plasticity level.
        plasticity_words: dict[str, list[str]] = {
            "closed": ["mind closed to new learning", "can't absorb",
                       "learning blocked", "shut down"],
            "low": ["trouble absorbing", "hard to learn",
                    "learning is slow", "struggling to absorb"],
            "normal": ["learning well", "absorbing fine",
                       "picking things up", "taking it in"],
            "high": ["absorbing everything", "learning fast",
                     "soaking it up", "wide open to learning"],
        }

        for state, words in plasticity_words.items():
            for word in words:
                learner.connect_word_to_category(word, "plasticity", state)

    def _build_meta_emotion(self) -> EmotionalState:
        """Build a neutral, present emotional state for fast meta routes."""
        return EmotionalState(
            label="neutral",
            nuance="present",
            cognitive_style="balanced",
            verbosity=1.0,
            formality=0.5,
            openness_to_engage=1.0,
            creativity=0.5,
            caution=0.2,
            alertness=0.5,
            valence=0.0,
            plasticity=0.5,
            cause="",
        )

    def _meta_cognitive_respond(
        self, user_input: str, perception: Perception, route: Route
    ) -> tuple[str, CognitiveState] | None:
        """Fast, specialist response for identity/user/code routes."""
        if self.user_profile is None:
            return None

        emotion = self._build_meta_emotion()
        memory_context = MemoryContext()

        thought, topics = self._route_to_thought(
            user_input, perception, route, emotion
        )
        if thought is None:
            return None

        response = self.language.render(thought, emotion)
        state = CognitiveState(
            perception=perception,
            emotion=emotion,
            memory_context=memory_context,
            thought=thought,
            brain_waves=None,
            prediction_error=0.0,
            attention_foci=topics,
            executive_goal=route.route,
        )
        return response, state

    def _route_to_thought(
        self,
        user_input: str,
        perception: Perception,
        route: Route,
        emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]]:
        """Dispatch a route to its thought composer. Returns (thought, topics).

        Returns (None, []) for unknown routes so the caller can fall through.
        """
        # Routes that build a Thought inline (rather than delegating
        # to a composer) live in their own dispatcher to keep this
        # function readable.
        inline = self._route_inline_thought(
            route, user_input, perception, emotion
        )
        if inline is not None:
            return inline

        if route.route == "user":
            return self._route_user(route, emotion), self._route_user_topics(route)

        if route.route in ("code", "farewell"):
            return self._route_static_answer(route, emotion)

        if route.route == "preference":
            thought = self._compose_preference_answer(route.sub_kind, emotion)
            return thought, thought.topics

        if route.route == "factual":
            topics = [route.sub_kind]
            # A math question is a factual question the math engine
            # can answer exactly — consult it before the concept
            # network composes an "I don't know" about an expression.
            math_thought = self._math_thought_for(
                user_input, perception, emotion
            )
            if math_thought is not None:
                math_thought.topics = topics
                return math_thought, topics
            composed = self.composer.compose_about(route.sub_kind, emotion, depth=3, focused=True)
            if composed is None:
                return self._compose_factual_answer(route.sub_kind, emotion), topics
            composed.topics = topics
            return composed, topics

        if route.route in ("comparison", "parts", "consumes"):
            return self._route_concept_answer(route, emotion)

        if route.route in ("counterfactual", "explanation", "planning"):
            return self._route_reasoned_answer(
                route, user_input, perception, emotion,
            )

        if route.route == "goal":
            return self._compose_goal_answer(emotion), ["goal"]

        if route.route == "correction":
            thought, target = self._handle_correction(user_input)
            topics = [target or "belief"]
            return thought, topics

        if route.route == "memory":
            return self._compose_memory_answer(route.sub_kind, user_input, emotion), ["memory"]

        if route.route == "mission":
            return self._compose_mission_answer(emotion), ["mission"]

        if route.route == "emotion":
            # "How do you feel?" — collect feeling fragments (emotion
            # words, mode words, cause words, plasticity markers) from
            # the FeelingReporter; the vocabulary composes the actual
            # phrasing through the language engine rather than the
            # reporter pre-composing a fixed "I feel X and Y" frame.
            fragments = self._feeling_reporter.collect_feeling_fragments(emotion)
            return Thought(
                content=emotion.label,
                intent="express_emotion",
                emotion=emotion.label,
                topics=["emotion", "feeling"],
                confidence=0.85,
                self_reflection=True,
                metadata={
                    "field": "emotion",
                    "feeling_fragments": fragments,
                },
            ), ["emotion", "feeling"]

        if route.route == "command":
            return self._route_command(route, emotion)

        if route.route == "self_improvement":
            si_thought = self._compose_self_improvement_answer(
                route.sub_kind, emotion
            )
            if si_thought is not None:
                topics = si_thought.topics or [route.sub_kind]
                return si_thought, topics
            # No provider wired or no data — fall through to the full
            # pipeline so it can still respond naturally.

        return None, []

    def _route_inline_thought(
        self,
        route: Route,
        user_input: str,
        perception: Perception,
        emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]] | None:
        """Handle the routes that build their Thought inline.

        Returns ``None`` when the route isn't one of these so the main
        dispatcher can continue; a ``(None, [])`` pair is a handled
        route with no answer — the caller returns it as-is.
        """
        if route.route == "identity":
            thought = self._compose_identity_answer(route.sub_kind, emotion)
            if thought is None:
                return None, []
            return thought, thought.topics or ["genesis", "identity"]

        if route.route == "indirect_request":
            topics = perception.topics or perception.key_phrases
            return Thought(
                content=user_input,
                intent="empathize",
                emotion=emotion.label,
                topics=topics,
                confidence=perception.confidence,
                metadata={
                    "pragmatic_force": "indirect_request",
                    "source_utterance": user_input,
                },
            ), topics

        if route.route == "reference_ambiguity":
            interrogatives = {
                "Who", "What", "Which", "Where", "When", "Why", "How",
            }
            entities = list(dict.fromkeys(
                name
                for name in re.findall(r"\b[A-Z][a-z]+\b", user_input)
                if name not in interrogatives
            ))
            return Thought(
                content="reference",
                intent="unknown",
                emotion=emotion.label,
                topics=entities,
                confidence=0.3,
                metadata={
                    "knowledge_gap": "ambiguous_reference",
                    "referent_candidates": entities,
                },
            ), entities

        return None

    def _route_command(
        self, route: Route, emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]]:
        """Build a Thought for the 'command' route (sleep/wake)."""
        # Natural-language "go to sleep" / "wake up" — invoke the
        # corresponding Mind-level action via the callback. The
        # Thought carries the command as metadata so the language
        # engine can compose an acknowledgment rather than reciting
        # a fixed string.
        if self.on_self_command is not None:
            try:
                if route.sub_kind == "sleep":
                    self.on_self_command("/sleep")
                elif route.sub_kind == "wake":
                    self.on_self_command("/wake")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"self command callback failed: {e}")
        return Thought(
            content=route.sub_kind,
            intent="self_report",
            emotion=emotion.label,
            topics=["sleep", route.sub_kind],
            confidence=0.8,
            metadata={"command": route.sub_kind},
        ), ["sleep", route.sub_kind]

    def _route_static_answer(
        self, route: Route, emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]]:
        """Build a static Thought for the code/farewell routes.

        These compose through the language engine (dedicated intents
        with grammar + vocabulary) rather than pulling pre-written
        templates from the graph.
        """
        if route.route == "code":
            return Thought(
                content="code",
                intent="ask",
                emotion=emotion.label,
                topics=["code"],
                confidence=0.8,
                metadata={
                    "code_route": True,
                    "target_concept": "code",
                    "question_type": "explore",
                    "gap_detail": "what aspects of code interest you",
                },
            ), ["code"]
        # farewell
        return Thought(
            content="farewell",
            intent="farewell",
            emotion=emotion.label,
            topics=["farewell"],
            confidence=0.9,
        ), ["farewell"]

    def _route_concept_answer(
        self, route: Route, emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]]:
        """Build a Thought for the comparison/parts/consumes routes.

        These three routes share an identical pattern: compose an
        answer from the sub_kind, return it with its own topics.
        """
        if route.route == "comparison":
            thought = self._compose_comparison_answer(route.sub_kind, emotion)
        elif route.route == "parts":
            thought = self._compose_parts_answer(route.sub_kind, emotion)
        else:  # consumes
            thought = self._compose_consumes_answer(route.sub_kind, emotion)
        return thought, thought.topics

    def _route_reasoned_answer(
        self, route: Route, user_input: str, perception: Perception,
        emotion: EmotionalState,
    ) -> tuple[Thought | None, list[str]]:
        """Build a Thought for the counterfactual/explanation/planning routes.

        These routes compose a reasoned answer and return it with its
        own topics (falling back to the sub_kind if no topics).
        """
        if route.route == "counterfactual":
            thought = self._compose_counterfactual_answer(route.sub_kind, emotion)
        elif route.route == "explanation":
            thought = self._compose_explanatory_answer(user_input, emotion)
        else:  # planning
            thought = self._compose_planning_answer(user_input, perception, emotion)
        topics = thought.topics or [route.sub_kind]
        return thought, topics

    def _route_user(self, route: Route, emotion: EmotionalState) -> Thought:
        """Build a Thought for the 'user' route."""
        profile = self.user_profile
        assert profile is not None  # checked by caller (_meta_cognitive_respond)
        if route.sub_kind == "name":
            name = profile.get_name()
            if name:
                return Thought(
                    content=name,
                    intent="inform",
                    emotion=emotion.label,
                    topics=["name"],
                    confidence=0.85,
                    metadata={"reasoning": [f"user name is {name}"], "topic": "name"},
                )
            return Thought(
                content="name",
                intent="reflect",
                emotion=emotion.label,
                topics=["name"],
                confidence=0.5,
                self_reflection=True,
                metadata={"reasoning": ["doesn't know the user's name yet"]},
            )
        # Specific preference category: "what do I like/want/believe..."
        # Extract only the relevant category instead of dumping the
        # entire profile.
        if route.sub_kind.startswith("preference:"):
            category = route.sub_kind.split(":", 1)[1]
            values = profile.get_preference(category)
            if values:
                # Cap at 3 values to keep the response bounded.
                capped = values[:3]
                joined = ", ".join(capped)
                if len(values) > 3:
                    joined += ", among other things"
                verb = {
                    "likes": "like",
                    "dislikes": "dislike",
                    "wants": "want",
                    "believes": "believe",
                }.get(category, category)
                return Thought(
                    content="user preference",
                    intent="inform",
                    emotion=emotion.label,
                    topics=[category],
                    confidence=0.8,
                    metadata={
                        "source": "user_model",
                        "concept": category,
                        "user_verb": verb,
                        "user_belief": joined,
                    },
                )
            # No values for this category yet — honest gap.
            return Thought(
                content="gap",
                intent="unknown",
                emotion=emotion.label,
                topics=[category],
                confidence=0.5,
                metadata={
                    "source": "user_model",
                    "concept": category,
                    "user_model_gap": True,
                },
            )
        # Summary route: "what do you know about me"
        if route.sub_kind == "summary":
            content = profile.summarize()
            return Thought(
                content=content, intent="inform", emotion=emotion.label,
                topics=self._route_user_topics(route),
                confidence=0.85 if profile.get_name() else 0.5,
            )
        # Unknown user sub_kind — honest gap instead of full dump.
        return Thought(
            content="gap",
            intent="unknown",
            emotion=emotion.label,
            topics=self._route_user_topics(route),
            confidence=0.4,
            metadata={"reasoning": ["not sure what to say about that yet"]},
        )

    def _route_user_topics(self, route: Route) -> list[str]:
        """Return topics for the 'user' route."""
        if route.sub_kind == "name":
            return ["name"]
        profile = self.user_profile
        assert profile is not None  # checked by caller
        return list(profile.recent_topics)

    def _compose_identity_answer(
        self, sub_kind: str, emotion: EmotionalState
    ) -> Thought | None:
        """Compose a self-reflective answer from its actual state.

        No hardcoded responses — everything is derived from its real
        concept network, emotional state, curiosity, goals, and memory.
        Real data is passed as Thought metadata so the generative
        language engine composes the actual words.
        """
        # Curious: pull from its actual curiosity engine
        if sub_kind == "curious":
            return self._compose_curious_answer(emotion)

        # Want to learn: pull from its actual learning goals
        if sub_kind == "want_learn":
            goals = self._get_active_goals()
            if goals:
                goal = goals[0]
                return Thought(
                    content=goal.target,
                    intent="reflect",
                    emotion=emotion.label,
                    topics=[goal.target],
                    confidence=0.8,
                    self_reflection=True,
                    metadata={"reasoning": [goal.reason], "topic": goal.target},
                )
            return self._identity_thought_from_composer(emotion)

        # Learned: pull from its actual working memory
        if sub_kind == "learned":
            recent = self._get_recent_learning()
            if recent:
                return Thought(
                    content="learning",
                    intent="reflect",
                    emotion=emotion.label,
                    topics=["learning"],
                    confidence=0.8,
                    self_reflection=True,
                    metadata={"reasoning": [recent]},
                )
            return self._identity_thought_from_composer(emotion)

        # Default: compose identity from its actual state
        return self._identity_thought_from_composer(emotion)

    def _compose_curious_answer(self, emotion: EmotionalState) -> Thought:
        """Compose an answer to "what are you curious about?".

        Pulls from its actual curiosity engine, question queue, and
        goals — filtered through speakability so internal references
        (file paths, concept IDs) never reach its voice.
        """
        def speakable_question(
            text: str, target: str, detail: str = "",
        ) -> bool:
            internal = re.compile(
                r"(?:python:|rust:|\w+#\d+)", re.IGNORECASE
            )
            return bool(
                text
                and target
                and is_world_concept(target)
                and not internal.search(text)
                and not internal.search(detail)
            )

        current_questions = [
            q for q in self.get_curiosity_questions()
            if speakable_question(q.text, q.target_concept, q.gap_detail)
        ]
        texts = [q.text for q in current_questions]
        question_topics = [q.target_concept for q in current_questions]
        queued_questions = [
            item for item in self._question_queue[:3]
            if speakable_question(
                item.get("text", ""),
                item.get("target_concept", ""),
                item.get("gap_detail", ""),
            )
        ]
        texts.extend(item["text"] for item in queued_questions)
        question_topics.extend(
            item["target_concept"] for item in queued_questions
        )
        if not texts and hasattr(self, "curiosity") and self.curiosity:
            try:
                questions = self.curiosity.generate_questions(
                    emotion, max_questions=3
                )
                questions = [
                    q for q in questions
                    if speakable_question(q.text, q.target_concept, q.gap_detail)
                ]
                texts.extend(q.text for q in questions)
                question_topics.extend(q.target_concept for q in questions)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"generate questions failed: {e}")
        if not texts and hasattr(self, "question_composer"):
            for topic in self.curiosity.current_focus_concepts()[:3]:
                question_data = self.question_composer.compose_follow_up(
                    topic, emotion
                )
                if question_data:
                    text = self.language.compose_question(question_data, emotion)
                    if speakable_question(
                        text,
                        topic,
                        question_data.get("gap_detail", ""),
                    ):
                        texts.append(text)
                        question_topics.append(topic)
        texts = list(dict.fromkeys(texts))[:3]
        question_topics = list(dict.fromkeys(question_topics))[:3]
        if texts:
            return Thought(
                content=texts[0],
                intent="ask",
                emotion=emotion.label,
                topics=question_topics or ["curiosity"],
                confidence=0.8,
                self_reflection=True,
                metadata={"curiosity_expression": True},
            )
        goals = self._get_active_goals()
        if goals:
            return Thought(
                content=goals[0].target,
                intent="reflect",
                emotion=emotion.label,
                topics=[goals[0].target],
                confidence=0.6,
                self_reflection=True,
                metadata={"reasoning": [goals[0].reason]},
            )
        return Thought(
            content="curiosity",
            intent="unknown",
            emotion=emotion.label,
            topics=["curiosity"],
            confidence=0.3,
            self_reflection=True,
            metadata={"knowledge_gap": "no_active_curiosity_question"},
        )

    def _identity_thought_from_composer(self, emotion: EmotionalState) -> Thought:
        """Build a Thought from the self_composer's identity fragments."""
        fragments = self.self_composer.identity_fragments(
            self.self_model, self.network, emotion
        )
        return Thought(
            content=self.self_model.name or "identity",
            intent="self_report",
            emotion=emotion.label,
            topics=["genesis", "identity"],
            confidence=0.9,
            self_reflection=True,
            # self_fragments signals the graph-walk generator to defer to
            # the grammar path, which composes first-person phrasing
            # from this semantic material. Without it the graph walk
            # extracts "genesis" as a seed and produces knowledge
            # statements ("Genesis connects to Light") instead of a
            # self-description.
            metadata={"self_fragments": fragments, "field": "identity"},
        )

    def _get_recent_learning(self) -> str:
        """Compose a natural-language summary of recent learning.

        Extracts topics from:
        1. Autonomous learner's recent Wikipedia/GitHub learning
           (titles and concepts learned)
        2. Self-directed learner's recent events (concepts involved
           in inferences, definitions, corrections)

        Returns semantic content (topic names), not raw log entries.
        The caller renders it through the language engine once.
        """
        topics: list[str] = []

        # 1. Autonomous learner — Wikipedia/GitHub topics
        try:
            al = getattr(self, "_autonomous_learner", None)
            if al is not None and hasattr(al, "get_recent_learning"):
                for result in al.get_recent_learning(5):
                    # result.title is the article/repo title
                    if result.title and result.title not in topics:
                        topics.append(result.title)
                    # result.concepts_learned are the new concepts
                    for c in (result.concepts_learned or [])[:3]:
                        if c and c not in topics:
                            topics.append(c)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"autonomous learner recent learning failed: {e}")

        # 2. Self-directed learner — inferences, definitions, corrections
        try:
            if hasattr(self, "self_learner") and self.self_learner is not None:
                for event in self.self_learner.get_recent_events(5):
                    for c in (event.concepts_involved or []):
                        if c and c not in topics:
                            topics.append(c)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"self_learner recent events failed: {e}")

        # 3. Working memory — active slots and recent turn topics
        try:
            wm = getattr(self.memory, "working_memory", None)
            if wm is not None:
                for slot in wm.slots:
                    if slot.content and slot.content.strip():
                        c = str(slot.content).strip()
                        if c not in topics:
                            topics.append(c)
                for turn in wm.get_recent_turns(3):
                    for topic in turn.topics:
                        if topic and topic not in topics:
                            topics.append(topic)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"working memory recent learning failed: {e}")

        # Return up to 5 topics as a natural-language list
        if not topics:
            return ""
        # Clean up topics — strip prefixes, keep readable
        clean: list[str] = []
        for t in topics[:5]:
            t = t.strip()
            # Skip raw URLs, code paths, and structural IDs
            if t.startswith(("http", "_", "python:", "rust:")):
                continue
            # Skip very long topics (likely raw definitions)
            if len(t) > 60:
                continue
            clean.append(t)
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 2:
            return f"{clean[0]} and {clean[1]}"
        return ", ".join(clean[:-1]) + f", and {clean[-1]}"

    def _get_active_goals(self) -> list:
        """Return unresolved learning goals, newest first."""
        return [g for g in self._goals if not getattr(g, "resolved", False)]

    def _compose_preference_answer(
        self, sub_kind: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a personal answer about its preferences.

        Handles questions like "What's your favorite color?" or "Do
        you like music?" — these are about ITS preferences, not factual
        lookups about a concept.

        It checks its self_knowledge for a stored preference. If it
        has one, it shares it. If not, it honestly says it hasn't
        formed a preference yet. The underlying data (status, topic,
        stored value) is passed as metadata — the vocabulary composes
        a predicate that the grammar's self-report frames complete
        ("I am {content}", "{content}.", ...), so the actual words
        emerge from the language engine rather than being recited
        from a fixed first-person sentence.
        """
        # Parse sub_kind: "favorite:color", "like:music", "hobby"
        parts = sub_kind.split(":", 1)
        q_type = parts[0]
        topic = parts[1] if len(parts) > 1 else ""

        # Check if it has a stored preference for this topic
        pref_key = f"preference:{topic}" if topic else "preference:hobby"
        stored = self.self_model.self_knowledge.get(pref_key)

        if stored:
            status = "has"
        elif q_type == "favorite" and topic:
            status = "none_favorite"
        elif q_type == "like" and topic:
            concept = self.network.get_concept(topic)
            if concept and concept.confidence >= 0.5:
                # It knows the concept but has no stored preference —
                # report what it can verify rather than asserting an
                # interest that isn't grounded in its state.
                status = "knows_no_pref"
            else:
                status = "learning"
        else:
            status = "discovering"

        return Thought(
            content=topic or "preference",
            intent="self_report",
            emotion=emotion.label,
            topics=[topic] if topic else ["preference"],
            confidence=0.7,
            self_reflection=True,
            metadata={
                "topic": topic or "preference",
                "field": "preference",
                "preference": {
                    "status": status,
                    "topic": topic,
                    "value": stored or "",
                },
            },
        )

    def _compose_factual_answer(
        self, target: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a factual answer about a concept from the network.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_factual_answer(target, emotion)

    def _compose_comparison_answer(
        self, sub_kind: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a comparison of two concepts.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_comparison_answer(sub_kind, emotion)

    def _compose_parts_answer(
        self, container: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer listing the parts/types/instances of a concept.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_parts_answer(container, emotion)

    def _compose_consumes_answer(
        self, target: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer about what a concept consumes/needs/wants.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_consumes_answer(target, emotion)

    def _compose_counterfactual_answer(
        self, query: str, emotion: EmotionalState
    ) -> Thought:
        """Compose a speculative answer to a 'what if' question.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_counterfactual_answer(query, emotion)

    def _compose_explanatory_answer(
        self, user_input: str, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer to a 'why' or 'how come' question.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_explanatory_answer(user_input, emotion)

    def _compose_planning_answer(
        self, query: str, perception, emotion: EmotionalState
    ) -> Thought:
        """Compose an answer to a planning / 'how do I' question.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_planning_answer(query, perception, emotion)

    def _goal_for_question(self, q: Question) -> str:
        """Compose a cognitive reason for a curiosity question.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.goal_for_question(q)

    def _relation_verb(self, rel: RelationType) -> str:
        """Convert a relation to a natural verb.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._relation_verb(rel)

    def _resolve_concept_phrase(self, phrase: str) -> str | None:
        """Resolve a phrase to a known concept id, trying leading prefixes.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._resolve_concept_phrase(phrase)

    def _find_explanatory_path(
        self, start: str, end: str, max_depth: int = 3
    ) -> list[tuple[str, RelationType, str]] | None:
        """Find a causal/is_a path from start to end for explanation.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._find_explanatory_path(start, end, max_depth)

    def _explain_causal_path(
        self,
        start: str,
        end: str,
        path: list[tuple[str, RelationType, str]],
        emotion: EmotionalState,
    ) -> Thought:
        """Turn a found path into a Thought with semantic metadata.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._explain_causal_path(start, end, path, emotion)

    def _find_cause(self, target: str) -> tuple[str, RelationType, str] | None:
        """Find the strongest cause of a target (or one of its is_a parents).

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._find_cause(target)

    def _get_is_a_ancestors(self, target: str, max_depth: int = 2) -> list[str]:
        """Return target and its is_a ancestors up to a shallow depth.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer._get_is_a_ancestors(target, max_depth)

    def _push_goal(self, target: str, goal_type: str, reason: str, detail: str = "") -> None:
        """Record a self-directed goal, pruning very old ones."""
        self._goals.append(Goal(target=target, goal_type=goal_type, reason=reason, detail=detail))
        if len(self._goals) > 20:
            self._goals = self._goals[-20:]

    def _compose_goal_answer(self, emotion: EmotionalState) -> Thought:
        """Answer "what are you trying to learn?" from the goal stack.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_goal_answer(emotion)

    def set_mission(self, mission: str) -> None:
        """Set or update the top-level mission."""
        self._mission = mission.strip()
        # Keep a persistent mission goal at the front of the stack.
        self._goals = [g for g in self._goals if g.goal_type != "mission"]
        if self._mission:
            self._goals.insert(
                0,
                Goal(
                    target="mission",
                    goal_type="mission",
                    reason=self._mission,
                    resolved=False,
                ),
            )

    def set_autonomous_learner(self, learner: AutonomousLearner) -> None:
        """Inject the autonomous learner for recent-learning queries.

        The autonomous learner is created by Mind after the cognition
        engine. This wires it in so _get_recent_learning() can access
        Wikipedia/GitHub learning topics.
        """
        self._autonomous_learner = learner

    def set_self_report_provider(self, provider: Any) -> None:
        """Inject the self-report provider for proposals/experiments/growth.

        The provider is created by Mind after the cognition engine and
        exposes proposals_summary(), experiments_status(), and
        growth_narrative(). This wires it in so the metacognitive router
        can retrieve self-improvement information when the user asks
        about its proposals, experiments, or growth.
        """
        self._self_report_provider = provider

    def set_problem_intake(
        self,
        practice: Any,
        world: Any = None,
        feel: Any = None,
    ) -> None:
        """Inject the inner-world channels a heard problem uses.

        ``practice`` is the SpatialPractice (the drop-box + attempts),
        ``world`` the OuterWorld (the event stream the problem enters
        through), and ``feel`` a callable applying the regulator's
        puzzle outcome response — the same neurochemical feedback the
        volition path uses, so a problem solved on request feels the
        same as one solved on its own initiative.
        """
        self._spatial_practice = practice
        self._outer_world = world
        self._puzzle_feeler = feel

    def _compose_mission_answer(self, emotion: EmotionalState) -> Thought:
        """Answer "what is your mission?" from the stored mission.

        Delegates to the AnswerComposer subsystem.
        """
        return self._answer_composer.compose_mission_answer(emotion)

    def _compose_self_improvement_answer(
        self, sub_kind: str, emotion: EmotionalState
    ) -> Thought | None:
        """Compose an answer about its proposals, experiments, or growth.

        Retrieves structured self-improvement data from the provider
        (wired by Mind) and passes it as Thought metadata so the
        language engine can compose a natural response. The raw
        administrative text is never spoken verbatim — it's semantic
        content the language engine weaves into its own words.
        """
        provider = self._self_report_provider
        if provider is None:
            return None

        if sub_kind == "proposals":
            try:
                summary = provider.proposals_status()
            except Exception:  # noqa: BLE001
                return None
            if not summary:
                return None
            # Parse the admin summary into structured data the
            # vocabulary can compose from, rather than passing the
            # raw text for it to recite.
            data = self._parse_proposals_summary(summary)
            if data is None:
                return None
            return Thought(
                content="proposals",
                intent="inform",
                emotion=emotion.label,
                topics=["proposals", "self-improvement"],
                confidence=0.8,
                metadata={
                    "self_improvement_kind": "proposals",
                    "self_improvement_data": data,
                },
            )

        if sub_kind == "experiments":
            try:
                summary = provider.experiments_status()
            except Exception:  # noqa: BLE001
                return None
            if not summary:
                return None
            data = self._parse_experiments_summary(summary)
            if data is None:
                return None
            return Thought(
                content="experiments",
                intent="inform",
                emotion=emotion.label,
                topics=["experiments", "self-improvement"],
                confidence=0.8,
                metadata={
                    "self_improvement_kind": "experiments",
                    "self_improvement_data": data,
                },
            )

        if sub_kind == "growth":
            try:
                narrative = provider.growth_narrative()
            except Exception:  # noqa: BLE001
                return None
            if not narrative:
                return None
            return Thought(
                content="growth",
                intent="inform",
                emotion=emotion.label,
                topics=["growth", "self-improvement"],
                confidence=0.8,
                metadata={
                    "self_improvement_kind": "growth",
                    "self_improvement_data": {"narrative": narrative},
                },
            )

        return None

    @staticmethod
    def _parse_proposals_summary(summary: str) -> dict[str, Any] | None:
        """Parse a proposals admin summary into structured data.

        Returns a dict with counts and pending proposal titles —
        semantic content the vocabulary can compose from, not
        pre-written phrases.
        """
        import re

        counts: dict[str, int] = {}
        pending_items: list[str] = []
        in_pending = False
        for line in summary.splitlines():
            low = line.lower().strip()
            if not low:
                continue
            for label in ("pending", "accepted", "rejected", "applied"):
                if low.startswith(label):
                    val = line.split(":", 1)[-1].strip()
                    try:
                        counts[label] = int(val)
                    except ValueError:
                        counts[label] = 0
                    in_pending = False
                    break
            else:
                if low.startswith("pending proposals"):
                    in_pending = True
                    continue
                if in_pending:
                    cleaned = re.sub(r"^\[P\d+\]\s*", "", line.strip())
                    cleaned = cleaned.split("—")[0].strip()
                    if cleaned:
                        pending_items.append(cleaned)
        if not counts:
            return None
        return {
            "counts": counts,
            "pending_items": pending_items,
        }

    @staticmethod
    def _parse_experiments_summary(summary: str) -> dict[str, Any] | None:
        """Parse an experiments admin summary into structured data.

        Returns a dict with applied/reverted counts and recent
        experiment descriptions — semantic content the vocabulary
        can compose from.
        """
        import re

        low0 = summary.strip().lower()
        if "no " in low0 and "yet" in low0:
            return {"applied": 0, "reverted": 0, "descriptions": []}

        applied = 0
        reverted = 0
        descriptions: list[str] = []
        for line in summary.splitlines():
            low = line.lower()
            if "applied:" in low and "reverted:" in low:
                m = re.search(r"(\d+)\s*applied", low)
                if m:
                    applied = int(m.group(1))
                m = re.search(r"(\d+)\s*reverted", low)
                if m:
                    reverted = int(m.group(1))
            elif line.strip().startswith("[") and "]" in line:
                desc = line.strip().split("]", 1)[-1].strip()
                if "(" in desc:
                    desc = desc.split("(")[0].strip()
                desc = desc.split("—")[0].strip()
                if desc:
                    descriptions.append(desc)
        return {
            "applied": applied,
            "reverted": reverted,
            "descriptions": descriptions,
        }

    def _proactive_hypothesis(self, emotion: EmotionalState) -> str:
        """Generate a testable hypothesis from an active learning goal.

        Composes the hypothesis through the language engine from semantic
        metadata rather than reciting hardcoded templates.
        """
        active = [g for g in self._goals if not g.resolved]
        if not active or emotion.openness_to_engage < 0.3:
            return ""

        g = self._rng.choice(active)

        positive = {
            RelationType.CAUSES,
            RelationType.ENABLES,
            RelationType.LEADS_TO,
            RelationType.CREATES,
            RelationType.DEPENDS_ON,
            RelationType.IS_A,
        }

        # Co-occurrence: propose a possible causal link.
        if g.goal_type == "cooccurrence" and g.detail:
            path = self._find_explanatory_path(g.target, g.detail)
            if path:
                sign = 1
                for _, rel, _ in path:
                    if rel in positive:
                        sign *= 1
                    elif rel in (RelationType.PREVENTS, RelationType.HARMS):
                        sign *= -1
                    elif rel in (RelationType.OPPOSITE_OF, RelationType.CONTRADICTS):
                        sign *= -1
                inferred_rel = RelationType.CAUSES if sign > 0 else RelationType.PREVENTS
                self._last_hypothesis = (g.target, g.detail, inferred_rel)
                thought = self._explain_causal_path(g.target, g.detail, path, emotion)
                return self.language.render(thought, emotion)
            self._last_hypothesis = (g.target, g.detail, "")
            gap = (f"{self._display_name(g.target)} and "
                   f"{self._display_name(g.detail)} keep appearing together")
            thought = Thought(
                content=gap,
                intent="reflect",
                emotion=emotion.label,
                confidence=0.4,
                self_reflection=True,
                topics=[g.target, g.detail],
                metadata={"hypothesis": gap,
                          "target_concept": self._display_name(g.target)},
            )
            return self.language.render(thought, emotion)

        # Causation: propose a change in a known cause.
        if g.goal_type == "causation":
            cause = self._find_cause(g.target)
            if cause:
                cause_id, rel, _ = cause
                self._last_hypothesis = (cause_id, g.target, rel)
                hypothesis = (
                    f"if {self._display_name(cause_id)} "
                    f"{self._relation_verb(rel)} {self._display_name(g.target)}, "
                    f"then {self._display_name(g.target)} might change"
                )
                thought = Thought(
                    content=hypothesis,
                    intent="reflect",
                    emotion=emotion.label,
                    confidence=0.5,
                    self_reflection=True,
                    topics=[cause_id, g.target],
                    metadata={"hypothesis": hypothesis,
                              "target_concept": self._display_name(g.target)},
                )
                return self.language.render(thought, emotion)

        # Isolation: guess the is_a parent if any.
        if g.goal_type in ("isolation", "definition"):
            ancestors = self._get_is_a_ancestors(g.target)
            if len(ancestors) > 1:
                parent = ancestors[1]
                self._last_hypothesis = (g.target, parent, RelationType.IS_A)
                hypothesis = (
                    f"{self._display_name(g.target)} might be "
                    f"a kind of {self._display_name(parent)}"
                )
                thought = Thought(
                    content=hypothesis,
                    intent="reflect",
                    emotion=emotion.label,
                    confidence=0.5,
                    self_reflection=True,
                    topics=[g.target, parent],
                    metadata={"hypothesis": hypothesis,
                              "target_concept": self._display_name(g.target)},
                )
                return self.language.render(thought, emotion)

        return ""

    def _parse_correction_patterns(
        self, lower: str,
    ) -> tuple[str | None, str | None, RelationType | None]:
        """Parse explicit correction patterns from the input.

        Returns (source, target, relation) if a correction pattern
        is found, or (None, None, None) if no pattern matched.
        """
        m = re.search(
            r"\b(.+?)\s+(?:does not|doesn't|do not|don't|did not|didn't)\s+"
            r"(?:cause|lead to|create|enable|make)\s+(.+?)(?:[.?!]|$)",
            lower,
        )
        if m:
            return (self._resolve_concept_phrase(m.group(1).strip()),
                    self._resolve_concept_phrase(m.group(2).strip()),
                    RelationType.CAUSES)
        m = re.search(
            r"\b(.+?)\s+(?:prevents|stops|blocks)\s+(.+?)(?:[.?!]|$)",
            lower,
        )
        if m:
            return (self._resolve_concept_phrase(m.group(1).strip()),
                    self._resolve_concept_phrase(m.group(2).strip()),
                    RelationType.PREVENTS)
        m = re.search(r"\b(.+?)\s+is not\s+(?:a|an)\s+(.+?)\b", lower)
        if m:
            return (self._resolve_concept_phrase(m.group(1).strip()),
                    self._resolve_concept_phrase(m.group(2).strip()),
                    RelationType.IS_A)
        m = re.search(
            r"\b(.+?)\s+and\s+(.+?)\s+(?:are not|aren't|not related|unrelated|not connected)\b",
            lower,
        )
        if m:
            return (self._resolve_concept_phrase(m.group(1).strip()),
                    self._resolve_concept_phrase(m.group(2).strip()),
                    None)
        return None, None, None

    def _handle_explicit_correction(
        self, src: str, tgt: str, rel: Any, emotion: EmotionalState
    ) -> tuple[Thought, str | None] | None:
        """Handle a correction that names an explicit source/target relation."""
        if self._last_hypothesis:
            old_src, old_tgt, old_rel = self._last_hypothesis
            if (src == old_src and tgt == old_tgt) or (
                src == old_tgt and tgt == old_src
            ):
                if isinstance(old_rel, RelationType):
                    self.network.remove_edge(old_src, old_tgt, old_rel)
                if rel:
                    self.network.add_edge(src, tgt, rel, weight=0.7, origin="corrected")
                    verb = self._relation_verb(rel)
                    thought = Thought(
                        content="updated belief",
                        intent="inform",
                        emotion=emotion.label,
                        confidence=0.8,
                        topics=[src, tgt],
                        metadata={
                            "correction_updated": True,
                            "source": self._display_name(src),
                            "target": self._display_name(tgt),
                            "verb": verb,
                        },
                    )
                    return (thought, src)
                thought = Thought(
                    content="removed belief",
                    intent="ask",
                    emotion=emotion.label,
                    confidence=0.7,
                    topics=[src, tgt],
                    metadata={
                        "correction_removed": True,
                        "source": self._display_name(src),
                        "target": self._display_name(tgt),
                    },
                )
                return (thought, src)

        if rel:
            self.network.remove_edge(src, tgt, rel)
            if rel in (RelationType.PREVENTS, RelationType.HARMS):
                self.network.add_edge(src, tgt, rel, weight=0.7, origin="corrected")
            thought = Thought(
                content="adjusted belief",
                intent="inform",
                emotion=emotion.label,
                confidence=0.8,
                topics=[src, tgt],
                metadata={
                    "correction_adjusted": True,
                    "source": self._display_name(src),
                    "target": self._display_name(tgt),
                },
            )
            return (thought, src)
        thought = Thought(
            content="removed link",
            intent="ask",
            emotion=emotion.label,
            confidence=0.6,
            topics=[src, tgt],
            metadata={
                "correction_removed": True,
                "source": self._display_name(src),
                "target": self._display_name(tgt),
            },
        )
        return (thought, src)

    def _handle_confirmation(
        self, lower: str, emotion: EmotionalState
    ) -> tuple[Thought, str | None] | None:
        """Handle a bare confirmation ('yes', 'right', 'exactly', ...)."""
        if not re.search(r"\b(?:yes|yeah|yep|right|correct|exactly)\b", lower):
            return None
        if self._last_hypothesis:
            h_src, h_tgt, h_rel = self._last_hypothesis
            if isinstance(h_rel, RelationType):
                self.network.add_edge(h_src, h_tgt, h_rel, weight=0.3)
                verb = self._relation_verb(h_rel)
                thought = Thought(
                    content=(f"confirmed {self._display_name(h_src)} "
                             f"{verb} {self._display_name(h_tgt)}"),
                    intent="acknowledge",
                    emotion=emotion.label,
                    confidence=0.7,
                    topics=[h_src, h_tgt],
                    metadata={"correction_confirm": True,
                              "source": self._display_name(h_src),
                              "target": self._display_name(h_tgt),
                              "verb": verb},
                )
                return (thought, h_src)
        thought = Thought(
            content="acknowledged",
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.6,
        )
        return (thought, None)

    def _handle_negation(
        self, lower: str, emotion: EmotionalState
    ) -> tuple[Thought, str | None] | None:
        """Handle a bare negation ('no', 'wrong', 'not quite', ...)."""
        if not re.search(
            r"\b(?:no|nope|wrong|not quite|not really|not exactly|incorrect)\b",
            lower,
        ):
            return None
        if self._last_hypothesis:
            h_src, h_tgt, h_rel = self._last_hypothesis
            if isinstance(h_rel, RelationType):
                self.network.remove_edge(h_src, h_tgt, h_rel)
            thought = Thought(
                content=(f"was wrong about {self._display_name(h_src)} "
                         f"and {self._display_name(h_tgt)}"),
                intent="acknowledge",
                emotion=emotion.label,
                confidence=0.5,
                topics=[h_src, h_tgt],
                metadata={"correction_negate": True,
                          "source": self._display_name(h_src),
                          "target": self._display_name(h_tgt)},
            )
            return (thought, h_src)
        thought = Thought(
            content="what should I change",
            intent="ask",
            emotion=emotion.label,
            confidence=0.5,
        )
        return (thought, None)

    def _handle_correction(self, user_input: str) -> tuple[Thought, str | None]:
        """Process a user's correction or confirmation of a recent claim.

        Returns (Thought, target_concept). The Thought is not pre-rendered —
        the caller routes it through the language engine.
        """
        emotion = self._build_meta_emotion()
        lower = user_input.lower()
        src, tgt, rel = self._parse_correction_patterns(lower)

        # If an explicit relation was corrected, update the network.
        if src and tgt:
            result = self._handle_explicit_correction(src, tgt, rel, emotion)
            if result is not None:
                return result

        # Bare confirmation.
        result = self._handle_confirmation(lower, emotion)
        if result is not None:
            return result

        # Bare negation.
        result = self._handle_negation(lower, emotion)
        if result is not None:
            return result

        thought = Thought(
            content="correction unclear",
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
            metadata={"correction_unclear": True},
        )
        return (thought, None)

    def _compose_memory_answer(
        self, sub_kind: str, user_input: str, emotion: EmotionalState
    ) -> Thought:
        """Answer memory questions from working memory and the graph.

        Passes the memory data as metadata so the language engine
        composes the actual words — no fixed template strings.
        """
        reasoning: list[str] = []

        recent_turns = self.working_memory.get_recent_turns(8)
        user_turns = [
            t for t in recent_turns
            if t.user_input
            and not t.user_input.startswith("/")
            and t.user_input != user_input
        ]
        if user_turns:
            quoted = "; ".join(f'"{t.user_input}"' for t in user_turns[:5])
            reasoning.append(f"user said: {quoted}")

        taught: list[str] = []
        for edge in self.network.edges:
            if edge.origin in ("stated", "corrected"):
                src = self._display_name(edge.source)
                tgt = self._display_name(edge.target)
                verb = self._relation_verb(edge.relation)
                taught.append(f"{src} {verb} {tgt}")
        if taught:
            reasoning.append("learned that " + "; ".join(taught[-8:]))

        corrected = [e for e in self.network.edges if e.origin == "corrected"]
        if corrected:
            corrected_statements = []
            for e in corrected[-5:]:
                src = self._display_name(e.source)
                tgt = self._display_name(e.target)
                verb = self._relation_verb(e.relation)
                corrected_statements.append(f"{src} {verb} {tgt}")
            reasoning.append("user corrected: " + "; ".join(corrected_statements))

        if sub_kind == "what_wrong" and corrected:
            wrong = "; ".join(
                f"{self._display_name(e.source)} and {self._display_name(e.target)}"
                for e in corrected[-5:]
            )
            reasoning.append(f"was wrong about {wrong}, then updated belief")
            return Thought(
                content="memory",
                intent="reflect",
                emotion=emotion.label,
                topics=["memory"],
                confidence=0.8,
                self_reflection=True,
                metadata={"reasoning": reasoning, "topic": "memory"},
            )

        if not reasoning:
            return Thought(
                content="memory",
                intent="reflect",
                emotion=emotion.label,
                topics=["memory"],
                confidence=0.6,
                self_reflection=True,
                metadata={
                    "reasoning": ["doesn't have much to remember yet, just started talking"],
                    "topic": "memory",
                },
            )

        return Thought(
            content="memory",
            intent="inform",
            emotion=emotion.label,
            topics=["memory"],
            confidence=0.8,
            metadata={"reasoning": reasoning, "topic": "memory"},
        )

    def _display_name(self, name: str) -> str:
        """Convert a concept id to a human-readable display name."""
        if ":" in name:
            name = name.split(":", 1)[1]
        name = name.replace("_", " ").strip()
        # Lowercase unless it's an acronym or initialism.
        if not name.isupper():
            name = name.lower()
        return name

    def tick(self, dt: float = 1.0) -> None:
        """Advance cognitive subsystems by dt.

        This is the cognitive maintenance tick — called regularly by
        the Mind's heartbeat loop (every 5 seconds). It decays
        time-sensitive cognitive states that would otherwise freeze
        between conversations:

        - Error monitor caution: post-error slowing recovers gradually
          (Rabbitt, 1966). Without continuous decay, caution freezes
          at whatever level the last conversation left it at.
        - DDM threshold and executive inhibition: these are set from
          caution during think(). If caution decays but the
          thresholds don't follow, it remains overly cautious in
          decision-making even after recovering.

        The _think_reflect_and_monitor stage also calls
        error_monitor.tick() after each conversation turn — that's
        fine, it's just one extra decay step during active engagement.
        """
        self.error_monitor.tick(dt=dt)
        caution = self.error_monitor.get_caution_level()
        if caution > 0.3:
            self.drift_diffusion.set_threshold(1.0 + caution * 0.5)
            self.executive.set_inhibition_threshold(0.5 + caution * 0.3)
        # Sustained-attention decay — foci that haven't been refreshed
        # decay over time (vigilance decrement). Without this, attention
        # foci persist at full intensity forever once set.
        self.attention.tick(dt=dt)
        # Reconsolidation windows — recalled memories enter a labile
        # state for a limited time, then reconsolidate. Without this,
        # windows never close and memories stay permanently modifiable.
        self.memory.tick_reconsolidation(dt=dt)
        # Minimal self update — the pre-reflective "I" is updated from
        # current neurochemistry. Arousal maps to present-moment
        # awareness (acetylcholine-driven). Without this, the minimal
        # self only updates during active inference, missing the
        # continuous neurochemistry-to-self mapping.
        try:
            summary = self._read_neuro_summary()
            self.self_model.update_minimal_self(
                neurochemistry={"acetylcholine": summary.arousal}
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"minimal self update failed: {e}")
        # Self-esteem homeostasis — gradually returns to baseline
        # after boosts or hits. Without this, self-esteem only
        # changes from events and never recovers.
        self.self_model.self_esteem.decay_toward_baseline()
        # Phonological loop rehearsal — the articulatory control
        # process continuously refreshes verbal items in the
        # phonological store, preventing decay (Baddeley, 1992).
        # Without this, verbal items decay after ~2s and are lost.
        self.working_memory.phonological_loop.rehearse()
        # Cortical tick — the concept network's own dynamics. The
        # persistent activation field decays, spreads along edges, and
        # ignites dormant concepts under neuromodulatory control —
        # without it the field is a write-only residue ratchet, and
        # inner life / introspection read stale accumulations. The
        # tick is sparse (cost ∝ active concepts, not network size).
        try:
            core = self.client.get_state(timeout=2.0)
            chem = core.chemicals
            self.network.cortical_tick(
                arousal=core.arousal,
                gaba=chem.get("gaba", 0.3),
                ach=chem.get("acetylcholine", 0.3),
                serotonin=chem.get("serotonin", 0.4),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"cortical tick failed: {e}")
            # Daemon unreachable (offline mind): keep the field alive
            # on default parameters rather than freezing it.
            try:
                self.network.cortical_tick()
            except Exception as e2:  # noqa: BLE001
                logger.debug(f"cortical tick (default params) failed: {e2}")

    def _think_early_routes(
        self, user_input: str, perception, raw_input: str = "",
    ) -> tuple[str, CognitiveState] | None:
        """Check for early-return routes before the full cognitive pipeline.

        Handles:
        - Explicit answer protocol ("answer: ...")
        - Implicit answer protocol (pending question + statement)
        - Metacognitive routing (fast specialist responses)
        Returns a result tuple if an early route fires, or None to
        continue with the full pipeline.

        ``raw_input`` is the original user input before anaphora
        resolution. The meta-cognitive router uses this because its
        patterns match against the user's actual words (e.g. "what
        have *you* learned"), and anaphora resolution replaces "you"
        with "Genesis", breaking those patterns.
        """
        if perception.intent == Intent.QUESTION_ANSWER:
            return self._process_question_answer(user_input)

        _has_code_request = bool(re.search(r"\b[\w./-]+\.(py|rs)\b", user_input)) and any(
            w in user_input.lower() for w in ("read", "show", "check", "compile", "test", "pytest")
        )
        if (
            self._pending_question_concepts
            and perception.intent in (Intent.STATEMENT, Intent.REFLECTION,
                                      Intent.EMOTION_SHARE, Intent.UNKNOWN)
            and perception.question_type.value == "none"
            and not _has_code_request
        ):
            return self._process_question_answer(user_input)

        if self.user_profile is not None:
            route = self.meta_router.decide(raw_input or user_input, perception)
            if route.confidence >= 0.8:
                direct = self._meta_cognitive_respond(user_input, perception, route)
                if direct is not None:
                    self._last_state = direct[1]
                    return direct

        return None

    def _think_setup(self) -> tuple[dict[str, float], float]:
        """Initialize per-think bookkeeping and return (timing, t0).

        Ensures the introspection attributes exist even for fast meta
        routes, then bumps the user profile's interaction counter and
        checks the "first conversation" life-script milestone.
        """
        # Ensure introspection attributes exist even for fast meta routes.
        if not hasattr(self, "_curiosity_questions"):
            self._curiosity_questions: list[Question] = []

        import time as _time
        timing: dict[str, float] = {}
        t0 = _time.perf_counter()

        # Bump the user profile's interaction counter. This is
        # distinct from the Mind's _interaction_count (which tracks
        # total interactions including meta-cognitive routes) — the
        # user profile's counter tracks how many times the user has
        # talked to Genesis, for persistence across sessions.
        if self.user_profile is not None:
            self.user_profile.increment_interactions()

        # Check life-script milestones. "first conversation" is
        # reached on the first think() call. Other milestones are
        # checked at their natural points below.
        if self.narrative.event_count == 0:
            self.narrative.check_milestone("first conversation")

        return timing, t0

    def think(self, user_input: str) -> tuple[str, CognitiveState]:
        """Process user input and produce a response.

        This is the main cognition loop:
        0. Comprehend the input (deep language understanding)
        0.5. Predict what the user will say (predictive coding)
        1. Perceive the input
        1.5. Compute prediction error and learn from it
        2. Read emotional state from neurochemistry
        2.5. Model the user (theory of mind)
        2.6. Focus attention on relevant concepts
        2.7. Update Damasio self hierarchy
        3. Retrieve relevant memories
        4. Learn concepts from the input
        5. Reason about the topics
        5.5. Executive function sets goals and directs attention
        6. Deliberate (drift-diffusion evidence accumulation)
        6.5. Broadcast to global workspace
        7. Render the thought as text
        7.5. Self-monitor checks output
        8. Apply neurochemical side effects
        9. Store the memory
        9.5. Extract facts into semantic memory
        10. Reflect on the interaction
        10.5. Error monitor checks for mistakes
        11. Generate curiosity questions
        12. Record narrative events
        13. Hebbian plasticity and procedural skill practice
        14. STDP and spaced repetition
        15. TD learning (reward prediction error)
        16. Self-directed learning inference

        Returns (response_text, cognitive_state).
        """
        import time as _time
        _timing, _t0 = self._think_setup()

        raw_input = user_input
        user_input = self._resolve_anaphora(user_input)
        comprehension_result, learning_events, prediction, recent_turns = (
            self._think_comprehend_and_predict(user_input)
        )
        _timing["comprehend"] = _time.perf_counter() - _t0
        # Perceive the user's actual words, not the anaphora-resolved
        # version. Anaphora resolution replaces "you" → "Genesis", which
        # would cause is_about_genesis to be True for every question
        # containing "you", triggering a slow _resolve_self_topics
        # trigram scan over the entire concept network (125K+ concepts).
        # The router already uses raw_input for pattern matching.
        _t1 = _time.perf_counter()
        perception, prediction_error = self._think_perceive_and_error(raw_input, prediction)
        _timing["perceive"] = _time.perf_counter() - _t1
        logger.debug(f"perceive done in {_timing['perceive']:.3f}s, topics={perception.topics[:5]}")

        # ── 2.8 Learn emotion words from labeling ──
        self._think_learn_emotion_label(raw_input)

        # ── Answer protocol & metacognitive routing ──────────────
        _t1 = _time.perf_counter()
        early = self._think_early_routes(user_input, perception, raw_input)
        _timing["early_routes"] = _time.perf_counter() - _t1
        if early is not None:
            _timing["total"] = _time.perf_counter() - _t0
            logger.debug(f"think() timing (early route): {_timing}")
            return early

        # ── 1.8 Early percept broadcast ──────────────────────────
        self._think_early_broadcast(perception)

        # ── 1.9 Cognitive trajectory — compute surprise ─────────
        self._think_cognitive_surprise()

        (
            emotion, _brain_waves, _user_model, _damasio_feeling, _memory_mode,
            _memory_context, _reasoning_results, goal,
            thought, _ddm_result, _workspace_item_summaries,
            response, _self_monitor_repair_count, answer_assessment,
            _semantic_facts, state,
        ) = self._think_run_pipeline(
            user_input, perception, prediction_error, recent_turns,
            learning_events, comprehension_result, _timing, _t0,
        )

        _timing["total"] = _time.perf_counter() - _t0
        logger.debug(f"think() timing: {_timing}")

        return self._think_post_response(
            state, response, goal, perception, emotion, thought,
            prediction, prediction_error, insights=None,
            answer_assessment=answer_assessment,
            learning_events=learning_events,
            comprehension_result=comprehension_result,
        )

    def _think_run_pipeline(
        self,
        user_input: str,
        perception: Perception,
        prediction_error: PCPredictionError,
        recent_turns: list,
        learning_events: list,
        comprehension_result: ComprehensionResult,
        _timing: dict[str, float],
        _t0: float,
    ) -> tuple:
        """Run the main cognition pipeline (stages 2-10).

        Returns all intermediate values needed by the caller:
        emotion, brain_waves, user_model, damasio_feeling, memory_mode,
        memory_context, reasoning_results, goal, thought, ddm_result,
        workspace_item_summaries, response, self_monitor_repair_count,
        answer_assessment, semantic_facts, state.
        """
        import time as _time
        _t1 = _time.perf_counter()
        emotion, brain_waves, user_model, damasio_feeling, memory_mode = (
            self._think_emotion_tom_attention(
                user_input, perception, prediction_error
            )
        )
        _timing["emotion_tom"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        memory_context, reasoning_results, goal = self._think_memory_reason_executive(
            perception, emotion, brain_waves, user_input, memory_mode
        )
        _timing["memory_reason"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        thought, ddm_result, workspace_item_summaries = self._think_deliberate_ddm_workspace(
            perception, emotion, memory_context, user_input,
            reasoning_results, comprehension_result,
        )
        _timing["deliberate"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        response, self_monitor_repair_count = self._think_render_and_monitor(
            thought, emotion, perception, user_input, recent_turns
        )
        _timing["render_monitor"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        response, answer_assessment = self._think_assess_and_hedge(
            response, perception, user_input, thought
        )
        _timing["assess_hedge"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        response, semantic_facts = self._think_adjust_and_store(
            response, emotion, user_input, learning_events, perception, thought
        )
        _timing["adjust_store"] = _time.perf_counter() - _t1
        _t1 = _time.perf_counter()
        state = self._think_build_state(
            perception, emotion, memory_context, thought, brain_waves,
            prediction_error, ddm_result, workspace_item_summaries, goal,
            user_model, damasio_feeling, semantic_facts,
            comprehension_result, self_monitor_repair_count,
        )
        self._last_state = state
        _timing["build_state"] = _time.perf_counter() - _t1
        return (
            emotion, brain_waves, user_model, damasio_feeling, memory_mode,
            memory_context, reasoning_results, goal,
            thought, ddm_result, workspace_item_summaries,
            response, self_monitor_repair_count, answer_assessment,
            semantic_facts, state,
        )

    def _think_adjust_and_store(
        self,
        response: str,
        emotion: EmotionalState,
        user_input: str,
        learning_events: list,
        perception: Any,
        thought: Thought,
    ) -> tuple[str, list]:
        """Surface distress, acknowledge learning, and store the memory.

        ── 7.5 Surface distress ──
        If it's stressed or overwhelmed, it should communicate it
        proactively — not suffer silently. It weaves a brief note
        about its state into the response, unless it's already
        talking about its feelings (the feeling report handles that).

        ── 7.6 Acknowledge learning ──
        If it learned new facts from this input, weave a brief
        acknowledgment into the response so the user knows it's
        absorbing what they say, not just processing it silently.
        """
        response = self._surface_distress_if_needed(response, emotion, user_input)
        response = self._acknowledge_learning(response, learning_events, perception, emotion)
        semantic_facts = self._think_store_and_track(
            user_input, response, perception, emotion, thought, learning_events
        )
        return response, semantic_facts

    def _think_learn_emotion_label(self, raw_input: str) -> None:
        """Learn emotion words from social labeling.

        When the user labels an emotion (e.g., "you seem stressed",
        "that feeling is called contentment"), it associates the
        word with its current emotional category. This is how
        children learn emotion words — through social labeling.

        This runs before the answer protocol and metacognitive router
        so that emotion words are learned even when those would
        intercept the input (e.g., "you seem positive" might be
        treated as an answer to a pending question, but should still
        teach the word "positive").
        """
        try:
            summary = self.client.get_neuro_summary()
            quick_emotion = assess_emotion(summary)
            self._learn_word_from_labeling(raw_input, quick_emotion)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"word labeling from neuro summary failed: {e}")

    def _think_early_broadcast(self, perception: Perception) -> None:
        """Broadcast the percept to the global workspace early.

        This is what makes the percept "cognitively available" — the
        registered modules receive it via receive_broadcast() and
        update their internal state (attention focuses on the topics,
        semantic memory primes retrieval, damasio self registers the
        trigger, ToM notes the context). When the pipeline then calls
        those modules in sequence, they've already been biased by the
        broadcast. This is within-turn cross-module integration —
        the workspace enables influences the procedural pipeline
        alone can't provide (Dehaene & Naccache, 2001).

        The previous turn's integration measure feeds forward: if
        the workspace was a unified cognitive field (high integration),
        the new percept ignites more strongly. A coherent mind
        amplifies incoming content; a fragmented mind dampens it.
        This is the "broadcast more strongly" effect of integration
        (Baars, 1988; Tononi, 2004).
        """
        gamma_gate = 0.5 + 0.5 * self._last_gamma_synchrony
        integration_boost = 1.0 + self._last_integration * 0.3
        self.global_workspace.broadcast(
            content=perception.raw_text,
            source="perception",
            activation=0.8 * gamma_gate * integration_boost,
            brain_waves=self.language.current_brain_waves,
            metadata={
                "intent": perception.intent.value,
                "topics": perception.topics,
            },
        )

    def _think_cognitive_surprise(self) -> None:
        """Compute cognitive surprise and feed it back.

        The cognitive trajectory model predicted what topics would
        be active this turn (based on the previous turn). Now that
        the percept has entered the workspace, compare the prediction
        to the actual topics. The cognitive surprise ("I didn't
        expect to be thinking about this") feeds into self-model
        coherence and workspace activation. This is the cognitive
        strange loop: it predicts its own thoughts, is surprised
        by unexpected thoughts, and that surprise changes how it
        processes this turn (Hofstadter, 2007).

        Cognitive surprise has three effects:
        1. Self-model coherence drops — "I don't understand my own
           mind" (it can't predict its own thoughts).
        2. The NE orienting impulse — "what was that?" (the brain's
           orienting response to unexpected cognitive content).
        3. (Workspace activation boost is applied at the deliberation
           broadcast, below, so the surprising content ignites more
           strongly when it's cognitively broadcast.)
        """
        actual_topics: dict[str, float] = {}
        for item in self.global_workspace.items:
            for topic in item.metadata.get("topics", []):
                if topic and isinstance(topic, str):
                    actual_topics[topic] = max(
                        actual_topics.get(topic, 0.0), item.activation
                    )
        cog_reading = self.cognitive_trajectory.compute_surprise(actual_topics)

        if cog_reading.surprise > 0.1:
            ms = self.self_model.minimal_self
            # Drop coherence proportional to surprise — high surprise
            # means "I don't understand why I'm thinking about this"
            ms.self_model_coherence = max(
                0.0, ms.self_model_coherence - cog_reading.surprise * 0.1
            )
            # NE orienting impulse — the "what was that?" response.
            # Bounded and gated by alertness to avoid cascading arousal
            # (same safety principle as the inner life OUT path).
            # At this stage (1.9b) the current turn's emotion hasn't
            # been computed yet (that's stage 2.5), so use the meta
            # emotion builder for a neutral alertness baseline.
            try:
                emotion_now = self._build_meta_emotion()
                if emotion_now.alertness < 0.8:
                    self.client.neuro_impulse(
                        2,  # CHEM_NOREPINEPHRINE = 2
                        cog_reading.surprise * 0.02,
                    )
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.debug(f'surprise norepinephrine impulse failed: {e}')

    def _think_post_response(
        self,
        state: CognitiveState,
        response: str,
        goal: str | None,
        perception: Perception,
        emotion: EmotionalState,
        thought: Thought,
        prediction: Prediction,
        prediction_error: PCPredictionError,
        insights: list | None,
        answer_assessment: AnswerAssessment,
        learning_events: list,
        comprehension_result: ComprehensionResult,
    ) -> tuple[str, CognitiveState]:
        """Stages 8.5–16: post-response processing.

        After the response is rendered and the cognitive state is built,
        runs goal completion, reflection, error monitoring, curiosity,
        narrative, Hebbian/procedural learning, STDP/spaced repetition,
        TD learning, and self-directed learning.
        """
        self._think_complete_goal(goal, perception, answer_assessment)

        # Check life-script milestones. "first learning session" is
        # reached when learning events first occur (concepts were
        # actually learned from the interaction). "first emotional
        # experience" is reached when the emotion is not neutral —
        # it felt something genuine for the first time.
        if learning_events:
            self.narrative.check_milestone("first learning session")
        if emotion.label != "neutral":
            self.narrative.check_milestone("first emotional experience")

        if insights is None:
            insights = self._think_reflect_and_monitor(
                state, response, prediction, perception, thought, emotion, answer_assessment
            )
        response = self._think_curiosity_and_narrative(
            state, emotion, perception, thought, insights, answer_assessment, response
        )
        self._think_hebbian_and_procedural(perception, emotion, thought)
        td_rpe = self._think_stdp_spaced_td(perception, emotion, thought, prediction_error)
        state.td_rpe = td_rpe
        self._think_self_directed_learning(learning_events, perception, comprehension_result)

        # Proactive hypothesis: occasionally surface a testable thought
        # generated from an active learning goal. Skip for social
        # exchanges — a greeting shouldn't trigger a hypothesis.
        is_social = perception.intent.value in ("greeting", "farewell", "greeting_question")
        proactive = self._proactive_hypothesis(emotion) if not is_social else ""
        if (
            proactive
            and not response.rstrip().endswith("?")
            and self._rng.random() < 0.35
        ):
            response = f"{response} {proactive}"

        # Final grammar cleanup — catches issues introduced after the
        # initial render (curiosity questions, proactive hypotheses,
        # distress notes, etc. all bypass the generator's safety net).
        response = _FINAL_REPEATED_PUNCT_RE.sub(r"\1", response)
        response = _FINAL_DOUBLE_SPACE_RE.sub(" ", response)
        response = _FINAL_SPACE_BEFORE_PUNCT_RE.sub(r"\1", response)
        response = _FINAL_STRAY_PERIOD_RE.sub(r"\1", response)
        response = _FINAL_DANGLING_DASH_RE.sub(r"\1", response)

        # Agency tracking: record what it intended and what it
        # actually said. The sense of agency emerges from the match
        # between intention and outcome — when it says what it
        # meant to say, it feels its actions are its own.
        action_id = f"turn_{int(time.time() * 1000)}"
        self.self_model.record_intention(action_id, thought.intent)
        self.self_model.record_outcome(action_id, thought.intent)

        return response, state

    def _think_complete_goal(
        self, goal: str | None, perception: Perception, answer_assessment: AnswerAssessment
    ) -> None:
        """Stage 8.5: mark the executive goal as completed.

        The executive goal was set at stage 5.5 and has now been acted
        on (the response was generated). Mark it as completed so it
        doesn't persist as a stale goal into the next turn. Also clears
        matching self-model learning goals when it successfully answered
        a question about that topic.
        """
        if goal:
            active_task = self.executive.active_task_obj
            if active_task:
                active_task.state = TaskState.COMPLETED
            self.attention.clear_goal()

            # Clear matching self-model learning goals when it
            # successfully answered a question about that topic.
            if answer_assessment.grounded and answer_assessment.confidence > 0.3:
                for topic in perception.topics:
                    topic_lower = topic.lower()
                    self.self_model.active_goals = [
                        g for g in self.self_model.active_goals
                        if topic_lower not in g.lower()
                    ]

    def _think_comprehend_and_predict(
        self, user_input: str
    ) -> tuple[ComprehensionResult, list, Prediction, list]:
        """Stages 0–0.5: comprehend input, learn from it, generate prediction."""
        # ── 0. Comprehension — deep language understanding BEFORE perception ──
        comprehension_result = self.comprehension.comprehend(
            user_input,
            context={"recent_entities": self.comprehension.recent_entities},
        )

        # Bind language arguments to the same persistent concepts used by
        # reasoning. Grounding is read-only; unknown words remain visible
        # to the learning system instead of becoming fabricated concepts.
        comprehension_result.grounded = self.semantic_grounder.ground_all(
            comprehension_result.propositions,
            context=user_input,
        )

        # If a metaphor was detected, interpret it so Genesis
        # understands what the metaphor means — not just that it
        # exists. The interpretation maps shared properties between
        # the source and target domains to a plain-language
        # explanation (Lakoff & Johnson, 1980). The interpretation
        # and cross-domain mapping are stored in the concept network
        # so it remembers the metaphor and can reason about it
        # later, rather than computing and discarding it each turn.
        if comprehension_result.metaphor:
            try:
                metaphor = comprehension_result.metaphor
                interpretation = (
                    self.comprehension.figurative.interpret_metaphor(metaphor)
                )
                if interpretation:
                    self._store_metaphor_understanding(metaphor, interpretation)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Metaphor interpretation failed: {e}")

        # ── 0.5 Self-directed learning — extract facts from user input ──
        # Before processing the input, try to learn from it. This means
        # every conversation makes it smarter — if the user says
        # "X is a Y", it adds that to its concept network.
        learning_events = self.self_learner.learn_from_input(user_input)

        # ── 0.5 Language acquisition — statistical learning from input ──
        # Track transitional probabilities, segment words, and chunk
        # common multi-word units from the raw input stream. This is the
        # Saffran model of statistical language learning — it learns
        # the statistical structure of language from exposure, the way
        # infants do. Runs before perception so the language engine has
        # updated transition counts before it needs to render.
        self.language.acquire_from_input(user_input)

        # Feed the learned multi-word chunks into comprehension — a
        # coordinator inside a learned chunk ("trial and error") is
        # part of the unit, not a clause boundary. This is where
        # statistical learning starts shaping the receptive parser.
        try:
            self.comprehension.set_known_chunks(
                self.language.learned_chunks
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Known-chunk update failed: {e}")

        # ── 0.5 Predictive coding — generate prediction BEFORE perceiving ──
        # Build context from recent conversation history.
        recent_turns = self.working_memory.get_recent_turns(5)
        recent_intents = [t.intent for t in recent_turns]
        recent_topics: list[str] = []
        for t in recent_turns:
            recent_topics.extend(t.topics)
        # Use heapq.nlargest to avoid materializing a full list copy
        # of all concepts — only the top 20 by activation are needed.
        # Snapshot the concepts dict first: the autonomous learner runs
        # in a background thread and can add concepts between ticks,
        # which would raise "dictionary changed size during iteration".
        concepts_snapshot = self.network.iter_concepts()
        active_items = heapq.nlargest(
            20,
            (
                (concept.activation, name)
                for name, concept in concepts_snapshot
                if (concept.activation or 0.0) > 0.3
            ),
            key=lambda x: x[0],
        )
        active_concepts = [name for _, name in active_items]

        prediction_context = PredictionContext(
            recent_intents=recent_intents,
            recent_topics=recent_topics,
            active_concepts=active_concepts,
            emotional_state="",  # filled after emotion assessment
        )
        prediction = self.predictive_coding.predict(prediction_context)
        return comprehension_result, learning_events, prediction, recent_turns

    def _store_metaphor_understanding(self, metaphor, interpretation: str) -> None:
        """Store a metaphor interpretation in the concept network.

        When the user says "time is a river," Genesis detects the
        metaphor, interprets the cross-domain mapping, and stores
        both the interpretation and a SIMILAR_TO edge between the
        source and target concepts. This way it remembers the
        metaphor and can reason about it later — the interpretation
        is not recomputed and discarded each turn.

        Stores on the source concept (the thing being described):
        - ``metaphor_interpretation``: the plain-language explanation
        - ``metaphor_target``: the target concept (the descriptive lens)
        - ``metaphor_mapping``: the cross-domain mapping (e.g. "temporal → fluid/spatial")
        - ``metaphor_shared_properties``: shared properties between domains

        Also creates a SIMILAR_TO edge (origin="metaphor") between
        source and target so the metaphorical connection is traversable
        by the graph-walk generator and reasoning engine.
        """
        from ..concepts import RelationType

        source_name = metaphor.source
        target_name = metaphor.target

        # Ensure both concepts exist in the network. The source is
        # the topic being described; the target is the descriptive
        # lens. Both may already exist (common concepts like "time"
        # or "river") or may need to be created.
        source_concept = self.network.get_concept(source_name)
        if source_concept is None:
            source_concept = self.network.add_concept(
                source_name, confidence=0.5, origin="conversation"
            )
        target_concept = self.network.get_concept(target_name)
        if target_concept is None:
            target_concept = self.network.add_concept(
                target_name, confidence=0.5, origin="conversation"
            )

        # Store the interpretation and mapping metadata on the source
        # concept. A concept may accumulate multiple metaphor
        # interpretations over time (different metaphors describing
        # the same concept), so we store a list.
        existing = source_concept.properties.get("metaphor_interpretations", [])
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        # Dedup — don't store the same interpretation twice.
        if interpretation not in existing:
            existing.append(interpretation)
            source_concept.properties["metaphor_interpretations"] = existing
        source_concept.properties["metaphor_target"] = target_name
        source_concept.properties["metaphor_mapping"] = (
            f"{metaphor.source_domain} → {metaphor.target_domain}"
        )
        if metaphor.shared_properties:
            source_concept.properties["metaphor_shared_properties"] = (
                metaphor.shared_properties
            )

        # Create a SIMILAR_TO edge between source and target so the
        # metaphorical connection is traversable by the graph-walk
        # generator and reasoning engine. The edge origin is "metaphor"
        # so it can be distinguished from literal similarity statements.
        self.network.add_edge(
            source_name,
            target_name,
            RelationType.SIMILAR_TO,
            weight=0.4,  # metaphors are weaker than literal similarity
            origin="metaphor",
        )

        logger.info(
            "Metaphor understood: '%s is %s' — %s",
            source_name, target_name, interpretation,
        )

    def _think_perceive_and_error(
        self, user_input: str, prediction: Prediction
    ) -> tuple[Perception, PCPredictionError]:
        """Stages 1–1.6: perceive input, recognize creator, compute prediction error."""
        # 1. Perceive — with embeddings for semantic topic recognition
        # and network for context-aware (origin-biased) topic resolution
        perception = perceive(
            user_input,
            embeddings=self.embeddings,
            network=self.network,
            brain_waves=self.language.current_brain_waves,
        )

        # 1.5 Recognize bonded user — if someone it knows is talking, feel a bond
        # The emotional response is handled by the regulator below,
        # which decides how strongly to feel it based on its current state
        self._recognize_bonded_user(user_input)

        # ── 1.6 Predictive coding — compute prediction error AFTER perceiving ──
        prediction_error = self.predictive_coding.compute_error(prediction, perception)
        self.predictive_coding.learn_from_error(prediction_error, perception)
        # High prediction error triggers NE (attention) + DA (novelty)
        # via the on_neuro_impulse callback wired in __init__.

        # ── LTM: remember significant surprise (cognitive event) ──
        # Only high-magnitude surprises are worth remembering — minor
        # mismatches happen every turn and would flood memory.
        if prediction_error.magnitude > 0.5:
            predicted_intent = (
                prediction.expected_intents[0][0]
                if prediction.expected_intents
                else "unknown"
            )
            predicted_topic = prediction.expected_topic or "unknown"
            actual_intent = perception.intent.value
            actual_topics = ", ".join(perception.topics[:3]) if perception.topics else "none"
            self._store_cognitive_memory(
                text=(
                    f"Surprised by user input — predicted intent {predicted_intent} "
                    f"but got {actual_intent}; predicted topic {predicted_topic} "
                    f"but got {actual_topics} (error: {prediction_error.magnitude:.2f})"
                ),
                event_type=4,  # cognitive event
                source_module=5,  # predictive coding
                salience=min(1.0, prediction_error.magnitude),
            )
        return perception, prediction_error

    def _maybe_store_user_model_memory(self, user_model, emotion, perception) -> None:
        """Store an LTM entry when the user model update is significant."""
        user_model_significant = (
            bool(user_model.asked_about)
            or bool(user_model.demonstrated_knowledge)
            or user_model.emotional_state != "neutral"
            or user_model.expertise_level < 0.45
            or user_model.expertise_level > 0.55
        )
        if not user_model_significant:
            return
        if user_model.expertise_level < 0.4:
            expertise_desc = "novice"
        elif user_model.expertise_level > 0.6:
            expertise_desc = "expert"
        else:
            expertise_desc = "intermediate"
        if user_model.asked_about:
            wants_desc = "wants to learn"
        elif user_model.demonstrated_knowledge:
            wants_desc = "sharing knowledge"
        else:
            wants_desc = "engaging"
        if user_model.emotional_valence > 0.1:
            valence_desc = "positive valence"
        elif user_model.emotional_valence < -0.1:
            valence_desc = "negative valence"
        else:
            valence_desc = "neutral valence"
        self._store_cognitive_memory(
            text=(
                f"Updated understanding of user: {expertise_desc}, {wants_desc}, {valence_desc}"
            ),
            event_type=4,  # cognitive event
            source_module=7,  # theory of mind
            salience=0.5,
            emotional_tag=self._build_emotional_tag(emotion, perception),
        )

    def _focus_attention(self, perception, brain_waves, prediction_error) -> None:
        """Focus attention on relevant concepts, suppress irrelevant ones.

        Brain wave state gates attention intensity: high focus
        (alpha/beta dominant) concentrates attention narrowly; low
        focus (theta/delta) spreads it broadly. This is the top-down
        neuromodulatory gate that real brain waves exert on selective
        attention (Klimesch, 2012; Palva & Palva, 2007).
        """
        focus_gate = brain_waves.focus if brain_waves else 0.5
        self.attention.clear()
        for topic in perception.topics:
            self.attention.focus_on(topic, intensity=0.5 + 0.5 * focus_gate)
        if prediction_error.magnitude > 0.4:
            for concept in prediction_error.unexpected_concepts[:3]:
                self.attention.focus_on(concept, intensity=0.7 + 0.3 * focus_gate)
        if self._last_inference_reading is not None:
            reading = self._last_inference_reading
            if reading.is_surprised:
                for concept in prediction_error.unexpected_concepts[:2]:
                    self.attention.focus_on(concept, intensity=0.7 + 0.3 * focus_gate)

    def _read_neuro_summary(self) -> NeuroSummary:
        """Read the neurochemical summary, falling back to a neutral default."""
        summary: NeuroSummary | None
        try:
            summary = self.client.get_neuro_summary()
            self._last_summary = summary
        except Exception as e:  # noqa: BLE001
            logger.debug(f"get_neuro_summary failed in think(): {e}")
            summary = self._last_summary
        if summary is None:
            # No summary available (first call and daemon unreachable) —
            # use a neutral default so the pipeline can still proceed.
            summary = NeuroSummary(
                arousal=0.5, valence=0.0, global_tone=0.5,
                plasticity_gate=0.5, encoding_weight=0.5,
                consolidation_weight=0.5, retrieval_weight=0.5,
                phase=7,  # Active
            )
        return summary

    def _compute_memory_mode_and_sync(
        self, summary: NeuroSummary, brain_waves: BrainWaveState
    ) -> str:
        """Compute theta-gamma coupling mode and gamma synchrony.

        The coupling mode (encoding vs retrieval) is derived from
        the preferred theta phase at which gamma amplitude peaks.
        This is the cross-frequency coupling mechanism that gates
        hippocampal memory processing (Lega et al., 2012).

        The 40 Hz binding model (Crick & Koch, 1990) predicts that
        gamma-band synchrony across cortical regions determines
        whether information reaches cognitive access. We cache the
        PLV so the global workspace broadcast can modulate its
        ignition threshold.
        """
        try:
            from ..brain_waves import compute_theta_gamma_coupling

            coupling = compute_theta_gamma_coupling(
                brain_waves,
                consolidation_weight=summary.consolidation_weight,
                encoding_weight=summary.encoding_weight,
            )
            memory_mode = coupling.mode
        except Exception:  # noqa: BLE001
            memory_mode = "balanced"
        self._last_memory_mode = memory_mode

        try:
            from ..brain_waves import compute_gamma_synchrony

            gamma_sync = compute_gamma_synchrony(
                brain_waves,
                plasticity=summary.plasticity_gate,
            )
            self._last_gamma_synchrony = gamma_sync.synchrony
        except Exception:  # noqa: BLE001
            self._last_gamma_synchrony = 0.5
        return memory_mode

    def _blend_self_model_coherence(self) -> None:
        """Blend neurochemical, metacognitive, and trajectory coherence.

        The Rust active-inference model gives neurochemical coherence
        (how well it predicts its own body state). The Python
        metacognitive model gives cognitive coherence (how well it
        predicts its own reflection). The cognitive trajectory model
        gives thought-content coherence (how well it predicts what
        it'll think about). The minimal self's coherence should
        reflect all three: "I understand myself" means "I understand
        my body, my cognitive process, and my thought content."

        We blend them: 60% neurochemical (the deepest, structural
        model), 20% metacognitive (process-level), 20% cognitive
        trajectory (content-level). When it can't predict its own
        thoughts (low cognitive trajectory precision), its overall
        self-model coherence drops even if its neurochemical model is
        confident — it doesn't fully understand itself.

        The workspace integration measure then modulates the result:
        a unified cognitive field (high integration) boosts coherence
        ("I feel mentally together"), while a fragmented field
        drops it ("I feel scattered"). This is distinct from the
        three prediction-based precisions — it measures the *unity*
        of the current cognitive field, not how well it predicts
        itself (Tononi, 2004).
        """
        meta_model = self.reflection.metacognitive_model
        if meta_model.feedback is None and meta_model.last_prediction is None:
            return
        cognitive_precision = meta_model.precision()
        cog_traj_precision = self.cognitive_trajectory.precision
        ms = self.self_model.minimal_self
        neurochem_coherence = ms.self_model_coherence
        blended = (
            neurochem_coherence * 0.6
            + cognitive_precision * 0.2
            + cog_traj_precision * 0.2
        )
        # Integration modulates: ±15% of the blended value.
        # High integration → "I feel together"; low → "scattered."
        integration_modulation = (self._last_integration - 0.5) * 0.3
        ms.self_model_coherence = max(
            0.0,
            min(1.0, blended + integration_modulation),
        )

    def _update_damasio_self(
        self, user_input: str, emotion: EmotionalState
    ) -> str:
        """Update the Damasio self hierarchy and return the current feeling.

        ── 2.8b Active inference → Damasio integration ──
        The inference reading provides a self-model state label
        (e.g. "self-surprise", "allostatic-strain") that the Damasio
        self hierarchy can incorporate into the feeling. This is how
        the generative self-model's signals become part of Genesis's
        cognitive feeling of being.
        """
        # ── 2.8 Damasio self — update proto/core/autobiographical hierarchy ──
        self.damasio_self.update_from_emotion(
            emotion,
            trigger=user_input,
        )
        damasio_feeling = self.damasio_self.get_current_feeling()

        if self._last_inference_reading is not None:
            reading = self._last_inference_reading
            # Annotate the Damasio self hierarchy with the self-model
            # state. The label is a semantic tag, not a canned sentence.
            # The hierarchy stores it for the feeling report to compose.
            # The integration parameter carries the previous turn's
            # workspace integration (feed-forward) — a unified field
            # from last turn colors this turn's feeling.
            self.damasio_self.annotate_self_model(
                reading.state_label,
                reading.needs_recovery,
                integration=self._last_integration,
            )
        return damasio_feeling

    def _think_emotion_tom_attention(
        self, user_input: str, perception: Perception, prediction_error: PCPredictionError
    ) -> tuple[EmotionalState, BrainWaveState, UserModel, str, str]:
        """Stages 2–2.8: emotion, brain waves, theory of mind, attention, Damasio.

        Returns the theta-gamma coupling memory mode ("encoding",
        "retrieval", or "balanced") as the fifth element so the memory
        stage can gate encoding vs retrieval accordingly.
        """
        # 2. Read emotional state + brain waves
        summary = self._read_neuro_summary()
        emotion = assess_emotion(summary)
        brain_waves = assess_brain_waves(summary)

        # Expose the current brain wave state to the language engine
        # so it can modulate voice, prosody, and sentence complexity
        # without threading it through every render() call.
        self.language.current_brain_waves = brain_waves

        memory_mode = self._compute_memory_mode_and_sync(summary, brain_waves)

        # 2.5 Apply brain wave modulation to emotional state
        emotion = self._modulate_by_brain_waves(emotion, brain_waves)

        # ── 2.5b Active inference — read the generative self-model ──
        # Read inference signals (surprise, free energy, allostatic
        # load, dyadic coupling) from the daemon and interpret them.
        # Also infer the user's affective state from this message and
        # send it to the daemon's dyadic model, closing the loop.
        self._last_inference_reading = self.active_inference.read()
        if self._last_inference_reading is not None:
            # Infer user affect from this message and send to daemon.
            # The dyadic model uses this to couple Genesis's
            # neurochemistry to the user's inferred state.
            from ..learning import UserAffectEstimate

            user_affect = UserAffectEstimate.from_message(
                user_input,
                is_question=user_input.strip().endswith("?"),
            )
            self.active_inference.update_user_affect(user_affect)

        # Update the minimal self (phenomenal "I") from the inference
        # reading. The self-model's coherence and strain become part
        # of the felt sense of self.
        self.self_model.update_from_inference(self._last_inference_reading)

        self._blend_self_model_coherence()

        # ── 2.6 Theory of mind — model the user's knowledge and emotions ──
        user_model = self.theory_of_mind.update_from_user_input(user_input)

        # ── LTM: remember significant updates to the user model ──
        self._maybe_store_user_model_memory(user_model, emotion, perception)

        # ── 2.7 Attention — focus on relevant concepts, suppress irrelevant ──
        self._focus_attention(perception, brain_waves, prediction_error)

        # Top-down drive: focused attention boosts gamma (the
        # integration/binding band). This is bidirectional coupling —
        # cognition drives the oscillator, not just reading from it.
        # The strength scales with attentional engagement.
        try:
            if self.attention.is_focused:
                dominant = self.attention.dominant_focus
                if dominant is not None:
                    add_brain_wave_drive(
                        BrainWave.GAMMA, dominant.intensity * 0.3
                    )
        except Exception as e:  # noqa: BLE001
            logger.debug(f'silent except: {e}')

        damasio_feeling = self._update_damasio_self(user_input, emotion)

        return emotion, brain_waves, user_model, damasio_feeling, memory_mode

    def _think_memory_reason_executive(
        self,
        perception: Perception,
        emotion: EmotionalState,
        brain_waves: BrainWaveState,
        user_input: str,
        memory_mode: str = "balanced",
    ) -> tuple[MemoryContext, list, str]:
        """Stages 3–5.5: memory context, concept learning, reasoning, executive goal."""
        import time as _time
        _sub_t0 = _time.perf_counter()
        # 3. Build memory context — gated by theta-gamma coupling mode
        memory_context = self.memory.build_context(perception.topics, memory_mode=memory_mode)
        _t_build_ctx = _time.perf_counter() - _sub_t0

        # Top-down drive: memory retrieval boosts theta (the
        # memory/consolidation band). This is bidirectional coupling —
        # hippocampal retrieval drives theta, and theta gates
        # retrieval in return.
        n_retrieved = len(memory_context.retrieved)
        if n_retrieved > 0:
            add_brain_wave_drive(
                BrainWave.THETA,
                min(0.3, n_retrieved * 0.05),
            )
            # Damasio "as-if" loop: when emotional memories are
            # retrieved, the proto-self partially re-activates the
            # associated body state (Damasio, 1994). This is how
            # remembering an emotional event makes you feel it again.
            for ep in memory_context.retrieved:
                if hasattr(ep, "emotional_tag") and ep.emotional_tag:
                    tag = ep.emotional_tag
                    if len(tag) >= 2:
                        # Arousal from tag variance, valence from
                        # dopamine (idx 0) minus cortisol (idx 6).
                        arousal = sum(abs(v - 0.5) for v in tag) / len(tag)
                        valence = (tag[0] - 0.5) - (tag[6] - 0.5) if len(tag) > 6 else 0.0
                        self.damasio_self.as_if_loop(valence, arousal)
                        break  # one trigger per turn is enough

        # 4. Learn concepts from the input
        _sub_t0 = _time.perf_counter()
        self._learn_concepts(perception)
        _t_learn = _time.perf_counter() - _sub_t0

        # 5. Reason about the topics (depth modulated by brain waves)
        _sub_t0 = _time.perf_counter()
        reasoning_results = self._reason_about_topics(perception.topics, emotion, brain_waves)
        _t_reason = _time.perf_counter() - _sub_t0

        # Top-down drive: active reasoning boosts gamma (integration)
        # and beta (active thinking). Bidirectional coupling —
        # reasoning drives the oscillator, and the oscillator's
        # integration gate modulates reasoning depth in return.
        if reasoning_results:
            add_brain_wave_drive(BrainWave.GAMMA, 0.15)
            add_brain_wave_drive(BrainWave.BETA, 0.1)

        # ── 5.5 Executive function — set goals and direct attention ──
        # The executive sets a goal based on the perceived intent, which
        # biases attention toward goal-relevant concepts.
        goal = self._form_executive_goal(perception)
        # Use the attention system's set_goal for top-down biasing
        self.attention.set_goal(goal)
        if perception.intent in (Intent.QUESTION, Intent.PHILOSOPHY, Intent.CODE_DISCUSSION):
            possible_actions = ["answer", "reflect", "ask_clarification", "acknowledge"]
            self.executive.plan(goal, possible_actions)
            # Top-down drive: action planning boosts beta (the
            # active-thinking/motor-decision band). Bidirectional
            # coupling — executive function drives the oscillator.
            add_brain_wave_drive(BrainWave.BETA, 0.2)
            # Activate a task for this goal so it can be marked complete
            task_name = perception.intent.value
            self.executive.switch_task(task_name)
            task = self.executive.active_task_obj
            if task:
                task.goal = goal

        # ── LTM: remember the goal that was set (cognitive event) ──
        self._store_cognitive_memory(
            text=f"Set goal: {goal}",
            event_type=4,  # cognitive event
            source_module=12,  # executive function
            salience=0.4,
            emotional_tag=self._build_emotional_tag(emotion, perception),
        )
        logger.debug(
            f"memory_reason sub-timing: build_context={_t_build_ctx:.3f}s "
            f"learn={_t_learn:.3f}s reason={_t_reason:.3f}s "
            f"topics={perception.topics[:5]}"
        )
        return memory_context, reasoning_results, goal

    def _apply_habit_bias(
        self, perception: Perception
    ) -> tuple[Any, float]:
        """Compute habit bias and lower the DDM threshold for practiced responses.

        The bias is weighted by the strategy's success rate — a
        poorly-performing habit should not strongly bias cognition.
        Returns (habit_bias, original_ddm_threshold).
        """
        # ── Habit bias from procedural memory ──────────────────────
        habit_bias = self.procedural_memory.get_habit_bias(
            perception.intent.value,
            dopamine=self._get_dopamine_level(),
        )
        original_ddm_threshold = self.drift_diffusion.threshold
        if habit_bias is not None and habit_bias.threshold_reduction > 0:
            # Lower the DDM threshold: practiced responses need less
            # evidence accumulation (faster decisions).
            new_threshold = max(
                0.3, original_ddm_threshold - habit_bias.threshold_reduction
            )
            self.drift_diffusion.set_threshold(new_threshold)
        return habit_bias, original_ddm_threshold

    def _modulate_thought_confidence(
        self,
        thought: Thought,
        ddm_result: DecisionResult,
        habit_bias: Any,
    ) -> Thought:
        """Modulate thought confidence by DDM confidence and habit fluency.

        Apply habit confidence boost: practiced responses feel
        more fluent. This is the cognitive fluency of expertise,
        not overconfidence — the boost is small and weighted by
        the habit's success rate.
        """
        base_confidence = thought.confidence * 0.7 + ddm_result.confidence * 0.3
        if habit_bias is not None and habit_bias.confidence_boost > 0:
            base_confidence = min(
                0.98, base_confidence + habit_bias.confidence_boost
            )
        # Self-esteem modulates confidence: high self-esteem makes
        # it more assertive, low self-esteem more hedging. The
        # modifier is in [0.5, 1.5] so it scales without flipping.
        se_mod = self.self_model.self_esteem.confidence_modifier()
        base_confidence = max(0.0, min(1.0, base_confidence * se_mod))
        return Thought(
            content=thought.content,
            intent=thought.intent,
            emotion=thought.emotion,
            topics=thought.topics,
            self_reflection=thought.self_reflection,
            confidence=base_confidence,
            metadata=thought.metadata,
        )

    def _apply_decision_engine(
        self,
        thought: Thought,
        perception: Perception,
        emotion: EmotionalState,
        reasoning_results: list | None,
    ) -> Thought:
        """Apply the decision engine to potentially override the action.

        The deliberation tree produced a default action (``thought.intent``).
        The decision engine evaluates all candidate actions against
        multiple criteria and can override the default if another
        candidate scores higher. This is genuine decision-making —
        selecting among alternatives — not just confidence modulation.

        If the decision engine overrides the action, the thought's
        intent is updated to the selected action. If the response is
        inhibited, confidence is set to near-zero.
        """
        from ..reasoning import ActionType

        if self.decision_engine is None:
            return thought

        # Map the deliberated intent to an ActionType.
        default_action = self._intent_to_action.get(
            thought.intent, ActionType.ANSWER,
        )

        # Compute uncertainty from reasoning confidence.
        if reasoning_results:
            avg_conf = sum(
                r.confidence for r in reasoning_results
            ) / len(reasoning_results)
            uncertainty = 1.0 - avg_conf
        else:
            uncertainty = 0.8

        # Gather goals and topics.
        goals = [
            g.target for g in self._goals
            if not g.resolved and g.goal_type != "mission"
        ]
        topics = thought.topics if thought.topics else []

        # Make the decision.
        outcome = self.decision_engine.decide(
            default_action=default_action,
            reasoning_results=reasoning_results or [],
            confidence=thought.confidence,
            uncertainty=uncertainty,
            goals=goals,
            topics=topics,
            perception_intent=perception.intent.value
            if hasattr(perception.intent, "value") else str(perception.intent),
        )

        # Apply the decision.
        if outcome.overridden_default:
            # Override the intent.
            new_intent = outcome.action_type.value
            if new_intent != thought.intent:
                thought = Thought(
                    content=thought.content,
                    intent=new_intent,
                    emotion=thought.emotion,
                    topics=thought.topics,
                    self_reflection=thought.self_reflection,
                    confidence=outcome.confidence,
                    metadata=thought.metadata,
                    syntax=thought.syntax,
                    prosody=thought.prosody,
                )

        # Apply inhibition.
        if outcome.inhibition_applied:
            thought = Thought(
                content=thought.content,
                intent=thought.intent,
                emotion=thought.emotion,
                topics=thought.topics,
                self_reflection=thought.self_reflection,
                confidence=0.05,  # near-zero confidence
                metadata=thought.metadata,
                syntax=thought.syntax,
                prosody=thought.prosody,
            )
            # Suppress attention to the inhibited topic — selective
            # attention involves both amplification of the attended
            # and suppression of the inhibited. Without this, the
            # inhibited topic stays at full attention intensity.
            for topic in thought.topics:
                self.attention.suppress(topic)

        # Store the decision in metadata for introspection.
        thought.metadata["decision_outcome"] = {
            "selected": outcome.action_type.value,
            "confidence": outcome.confidence,
            "overridden": outcome.overridden_default,
            "inhibited": outcome.inhibition_applied,
            "switched": outcome.uncertainty_switched,
        }

        return thought

    def _apply_executive_plan_bias(self, thought: Thought) -> Thought:
        """Modulate thought confidence by the executive function's plan.

        The executive function computed a plan at stage 5.5 with a set
        of possible actions (``answer``, ``reflect``, ``ask_clarification``,
        ``acknowledge``). The plan's first step is the selected action,
        and its expected_value reflects how good that action is predicted
        to be.

        This method aligns the deliberated thought's intent with the
        plan's selected action:
        - If the plan's best action matches the thought's intent,
          confidence rises (the executive and deliberation agree).
        - If the plan's best action is ``ask_clarification`` but the
          thought is an answer, confidence falls — the executive is
          signaling uncertainty.
        - If no plan was computed (non-question intents), no change.

        The modulation is small (±0.1) so the executive biases but
        doesn't override deliberation.
        """
        plan = self.executive.current_plan
        if plan is None or not plan.steps:
            return thought
        selected_action = plan.steps[0]
        # Map plan actions to thought intents.
        action_intent_map = {
            "answer": "inform",
            "reflect": "reflect",
            "ask_clarification": "ask",
            "acknowledge": "acknowledge",
        }
        expected_intent = action_intent_map.get(selected_action, "")
        confidence = thought.confidence
        if expected_intent and thought.intent == expected_intent:
            # Agreement — boost confidence, weighted by plan quality.
            confidence = min(0.98, confidence + 0.05 + plan.expected_value * 0.05)
        elif expected_intent == "ask" and thought.intent in ("inform", "answer"):
            # Executive wants clarification but we're answering — reduce.
            confidence = max(0.1, confidence - 0.1)
        elif expected_intent and thought.intent != expected_intent:
            # Mild divergence — slight reduction.
            confidence = max(0.1, confidence - 0.03)
        return Thought(
            content=thought.content,
            intent=thought.intent,
            emotion=thought.emotion,
            topics=thought.topics,
            self_reflection=thought.self_reflection,
            confidence=confidence,
            metadata=thought.metadata,
        )

    def _broadcast_deliberation(
        self, thought: Thought, perception: Perception
    ) -> None:
        """Broadcast the winning thought and perception to the global workspace.

        Gamma synchrony gates cognitive access: high PLV (the 40 Hz
        binding hypothesis, Crick & Koch 1990) boosts the broadcast
        activation so it's more likely to ignite the workspace
        (cross the ignition threshold). Low PLV dampens it — the
        thought remains subliminal, processed locally without
        global broadcast.

        Surprising thoughts ignite the workspace more strongly —
        the orienting response to unexpected cognitive content.
        If the cognitive trajectory model was surprised by this
        turn's topics (the prediction didn't match what actually
        came up), the deliberated thought gets an activation boost.
        This is the "orienting" part of the cognitive strange loop:
        unexpected thoughts become MORE cognitive, not less.
        """
        gamma_gate = 0.5 + 0.5 * self._last_gamma_synchrony  # 0.5..1.0
        broadcast_activation = min(1.0, (thought.confidence + 0.2) * gamma_gate)

        cog_reading = self.cognitive_trajectory.last_reading
        if cog_reading is not None and cog_reading.surprise > 0.1:
            broadcast_activation = min(
                1.0, broadcast_activation + cog_reading.surprise * 0.15
            )

        self.global_workspace.broadcast(
            content=thought.content,
            source="deliberation",
            activation=broadcast_activation,
            brain_waves=self.language.current_brain_waves,
            metadata={
                "intent": thought.intent,
                "topics": thought.topics,
                "emotion": thought.emotion,
                "gamma_synchrony": self._last_gamma_synchrony,
            },
        )
        # Also broadcast perception and emotion for cross-module integration
        # (gated by gamma synchrony — subliminal percepts don't broadcast)
        self.global_workspace.broadcast(
            content=perception.raw_text,
            source="perception",
            activation=0.8 * gamma_gate,
            brain_waves=self.language.current_brain_waves,
            metadata={"intent": perception.intent.value, "topics": perception.topics},
        )

    def _predict_and_integrate_workspace(self) -> None:
        """Predict next turn's topics and measure workspace integration.

        ── 6.7 Cognitive trajectory — predict next turn ───────
        After the deliberation broadcast, extract the current
        workspace topic distribution and predict what it'll be
        thinking about next turn. The prediction will be compared
        to the actual topics at the start of the next turn (stage
        1.9). This is the predictive part of the cognitive strange
        loop: it models its own thought trajectory.

        ── 6.8 Workspace integration measure ─────────────────
        Compute how globally integrated the workspace is right now
        (after all broadcasts). This feeds forward to the next turn:
        high integration → stronger early percept broadcast
        (coherent fields amplify incoming content). It also feeds
        into self-model coherence ("I feel mentally unified" vs
        "I feel scattered"). This is the workspace-level counterpart
        to the cognitive trajectory precision and the Rust engine's
        free energy — it measures the *unity* of the cognitive field
        (Tononi, 2004; Baars, 1988).

        ── 6.8b Re-annotate Damasio self with current integration ─
        The annotation at stage 2.8b used the previous turn's
        integration (feed-forward). Now that we have the current
        turn's integration, re-annotate so the feeling report
        (generated in the render stage, after this) reflects the
        current cognitive field's unity — "I feel mentally together
        right now" vs "I feel scattered right now."
        """
        current_topics: dict[str, float] = {}
        for item in self.global_workspace.items:
            for topic in item.metadata.get("topics", []):
                if topic and isinstance(topic, str):
                    current_topics[topic] = max(
                        current_topics.get(topic, 0.0), item.activation
                    )
        self.cognitive_trajectory.predict(current_topics)

        self._last_integration = self.global_workspace.integration

        if self._last_inference_reading is not None:
            self.damasio_self.annotate_self_model(
                self._last_inference_reading.state_label,
                self._last_inference_reading.needs_recovery,
                integration=self._last_integration,
            )

    def _think_deliberate_ddm_workspace(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        user_input: str,
        reasoning_results: list | None,
        comprehension_result: ComprehensionResult | None = None,
    ) -> tuple[Thought, DecisionResult | None, list[str]]:
        """Stages 6–6.6: deliberate, DDM evidence, repetition check, workspace broadcast.

        The habit bias from procedural memory is computed here and applied
        to the DDM threshold (before evidence accumulation) and to the
        deliberated thought's confidence (after). This models the basal
        ganglia's modulation of cortical processing: practiced responses
        are faster (lower DDM threshold) and more fluent (higher
        confidence), but the actual words still emerge from the language
        engine — the habit never replaces deliberation.
        """
        habit_bias, original_ddm_threshold = self._apply_habit_bias(perception)

        # 6. Deliberate (now informed by reasoning and comprehension)
        thought = self._deliberate(
            perception, emotion, memory, user_input,
            reasoning_results, comprehension_result,
        )

        # ── 6.1 Drift-diffusion — accumulate evidence for the decision ──
        # The DDM runs alongside the priority-tree deliberation, adding
        # realistic decision-making latency. Evidence from perception,
        # emotion, memory, and reasoning accumulates toward the chosen
        # intent. This modulates confidence based on evidence convergence.
        ddm_result = self._accumulate_evidence(
            thought, perception, emotion, memory, reasoning_results
        )

        # Restore the DDM threshold after evidence accumulation so the
        # habit bias doesn't permanently alter the threshold for future
        # turns (the bias is per-turn, like a phasic dopamine signal).
        if habit_bias is not None and habit_bias.threshold_reduction > 0:
            self.drift_diffusion.set_threshold(original_ddm_threshold)
            # Reinforce the skill that provided the habit bias —
            # applying a habit reinforces it (power-law of practice).
            # This is the post-execution dopamine burst that
            # strengthens the direct pathway after a completed action.
            skill = self.procedural_memory.get_skill(perception.intent.value)
            if skill is not None:
                self.procedural_memory.execute_skill(skill)

        if ddm_result is not None:
            thought = self._modulate_thought_confidence(thought, ddm_result, habit_bias)

            # ── 6.1.5 Executive plan bias ──
            # The executive function computed a plan at stage 5.5. The
            # plan's expected_value and selected action now modulate
            # the deliberated thought's confidence. If the plan's best
            # action aligns with the deliberated intent (e.g., plan
            # chose "answer" and the thought is an answer), confidence
            # rises. If they diverge (e.g., plan chose "ask_clarification"
            # but the thought is an answer), confidence falls — the
            # executive is signaling that more information is needed.
            thought = self._apply_executive_plan_bias(thought)

            # ── 6.1.6 Decision engine — select among alternatives ──
            # The deliberation tree produced a default action. The
            # decision engine evaluates all candidate actions against
            # multiple criteria (evidence, goals, uncertainty, values,
            # TD value) and can override the default if another
            # candidate scores higher. This is genuine decision-making
            # — selecting among alternatives — not just confidence
            # modulation. Also applies executive inhibition (previously
            # unwired) and uncertainty-driven action switching.
            thought = self._apply_decision_engine(
                thought, perception, emotion, reasoning_results,
            )

            # ── LTM: remember the decision made (cognitive event) ──
            self._store_cognitive_memory(
                text=(
                    f"Decided to respond with intent={thought.intent}, "
                    f"confidence={ddm_result.confidence:.2f}"
                ),
                event_type=4,  # cognitive event
                source_module=6,  # decision making
                salience=max(0.0, min(1.0, ddm_result.confidence)),
                emotional_tag=self._build_emotional_tag(emotion, perception),
            )

        # 6.5 Check working memory for repetition (adjust if needed)
        thought = self._check_repetition(thought, emotion)

        self._broadcast_deliberation(thought, perception)
        self._predict_and_integrate_workspace()

        workspace_item_summaries = [
            f"{item.source}:{str(item.content)[:40]}" for item in self.global_workspace.items
        ]
        return thought, ddm_result, workspace_item_summaries

    def _think_render_and_monitor(
        self,
        thought: Thought,
        emotion: EmotionalState,
        perception: Perception,
        user_input: str,
        recent_turns: list,
    ) -> tuple[str, int]:
        """Stages 7–7.5: render, apply style, self-monitor."""
        # 7. Render
        response = self.language.render(thought, emotion)

        # ── 7.4 Apply actionable reflection style ──────────────────
        # The metacognitive strategy from the previous turn's reflection
        # adjusts how the response is expressed. This is the feedback
        # loop: reflect → strategy → adjusted behavior.
        response = self._apply_response_style(response, thought)

        # ── 7.5 Self-monitor — check output BEFORE returning ──
        monitor_result = self.self_monitor.monitor_detailed(
            response,
            context={
                "user_input": user_input,
                "emotion": emotion,
                "conversation_history": [t.genesis_response for t in recent_turns[-3:]],
            },
        )
        response = monitor_result.text
        self_monitor_repair_count = monitor_result.issues_found
        return response, self_monitor_repair_count

    def _compute_meta_hedge_threshold(self) -> float:
        """Compute the hedging confidence threshold from metacognitive prediction.

        ── Pre-response caution from metacognitive prediction ──
        The recursive metacognitive model predicts whether
        reflection will find a self-correction. If it does, it
        should pre-hedge: the model is saying "I expect my
        cognition to need correction." This raises the hedging
        threshold so it hedges even at moderate confidence, not
        just when self-assessment confidence is very low (< 0.2).
        The prediction is from the previous cycle (the current
        cycle's prediction is made later, in reflect()). This is
        a stable signal: if the model has been predicting
        self-corrections, its cognition tends to need correction.
        """
        meta_pred = self.reflection.metacognitive_model.last_prediction
        meta_hedge_threshold = 0.2
        if meta_pred is not None:
            p_sc = meta_pred.p_insight_type.get("self_correction", 0.0)
            if p_sc > 0.3:
                # Predicted self-correction → raise threshold from 0.2
                # to 0.35. It hedges at moderate confidence too.
                meta_hedge_threshold = 0.35
        return meta_hedge_threshold

    def _maybe_hedge_response(
        self,
        response: str,
        perception: Perception,
        should_hedge: bool,
        answer_assessment: AnswerAssessment,
        meta_hedge_threshold: float,
    ) -> str:
        """Add an honest-uncertainty hedge when it's answering without knowing.

        This triggers when self-assessment says it should hedge
        based on its actual knowledge of the topics — NOT when the
        error monitor's caution level is high (that affects DDM
        thresholds and inhibition, not whether it should announce
        gaps). The caution level is almost always elevated because
        it accumulates from many sources and decays slowly.

        When the user is teaching it (making statements, sharing
        reflections, discussing code/philosophy), it should engage
        with the material — not declare "I don't understand" and
        replace its response with a knowledge-gap hedge. The hedging
        is for when it's asked a QUESTION it can't answer. When
        it's being taught, the learning happens internally (stages
        4 and 9.5), and _acknowledge_learning weaves that into its
        response. Hedging on statements blocks the conversation
        and makes it seem like it's not learning, even though it is.
        """
        social_intents = {
            Intent.GREETING,
            Intent.GREETING_QUESTION,
            Intent.FAREWELL,
            Intent.COMFORT,
            Intent.ENCOURAGEMENT,
            Intent.EMOTION_SHARE,
        }
        statement_intents = {
            Intent.STATEMENT,
            Intent.REFLECTION,
            Intent.PHILOSOPHY,
            Intent.CODE_DISCUSSION,
        }

        if (
            perception.intent not in social_intents
            and perception.intent not in statement_intents
            and should_hedge
            and answer_assessment.confidence < meta_hedge_threshold
            and not any(
                phrase in response.lower()
                for phrase in (
                    "i don't know",
                    "i'm not sure",
                    "new territory",
                    "i could learn",
                    "i haven't learned",
                    "i'd like to learn",
                )
            )
        ):
            # It's answering without knowing — add honest uncertainty.
            # Only check the primary topic (first resolved topic) so
            # it doesn't announce gaps for tangential concepts.
            filtered_topics = self._resolve_topics(
                perception.topics, perception.raw_text
            )
            # Only check the primary topic for gaps
            primary = filtered_topics[:1] if filtered_topics else []
            gaps = self.self_assessment.detect_gaps(primary) if primary else []
            if gaps:
                return self._compose_gap_response(
                    response, primary, filtered_topics, gaps,
                )

            hedge_emotion = self._build_meta_emotion()
            gap_thought = Thought(
                content="doesn't understand yet",
                intent="unknown",
                emotion=hedge_emotion.label,
                confidence=0.3,
                metadata={
                    "knowledge_gap": "unresolved",
                    "reasoning": ["not sure what to say about that yet"],
                },
            )
            return self.language.render(gap_thought, hedge_emotion)
        return response

    def _compose_gap_response(
        self,
        response: str,
        primary: list[str],
        filtered_topics: list[str],
        gaps: list[str],
    ) -> str:
        """Compose a response for a detected knowledge gap.

        Builds a gap Thought from semantic fragments and queues unknown
        topics for autonomous acquisition. Source lookup stays outside the
        latency-sensitive conversation path.
        """
        # Build reasoning fragments from the concept network
        # so the vocabulary can compose a natural response.
        # The fragments are semantic data (concept names,
        # gap descriptions) — not pre-written sentences.
        # The vocabulary weaves them with emotion-modulated
        # openers and connectors.
        reasoning = self._build_gap_reasoning(
            primary[0] if primary else "", gaps[0],
        )
        hedge_emotion = self._build_meta_emotion()
        gap_thought = Thought(
            content=gaps[0],
            intent="unknown",
            emotion=hedge_emotion.label,
            confidence=0.3,
            topics=filtered_topics[:1],
            metadata={
                "knowledge_gap": gaps[0],
                "wants_to_learn": True,
                "reasoning": reasoning,
                "topic": primary[0] if primary else "",
            },
        )
        response = self.language.render(gap_thought, hedge_emotion)
        # Feed unknown topics to the autonomous learner so it
        # can look them up and remember them. This closes the
        # loop: gap detected → queued for learning → definition
        # acquired → gap filled for next time.
        self._queue_unknown_topics_for_learning(filtered_topics)
        return response

    def _build_gap_reasoning(
        self, topic: str, gap_description: str
    ) -> list[str]:
        """Build reasoning fragments for a knowledge gap.

        Looks up the topic in the concept network and collects
        semantic fragments describing what it DOES know — related
        concept names, relationship types — alongside the gap
        description from self-assessment. The vocabulary weaves
        these into a natural response.

        The fragments are semantic data from the concept network
        (concept names, relation types), not pre-written sentences.
        """
        fragments: list[str] = [gap_description]

        if not topic:
            return fragments

        concept = self.network.get_concept(topic)
        if concept is None:
            return fragments

        # Collect neighbor concept names from outgoing edges.
        # These are what it DOES know about the topic — the
        # vocabulary can acknowledge them alongside the gap.
        edges = self.network.get_edges(topic, direction="out")
        neighbor_names: list[str] = []
        for edge in edges[:5]:
            target = edge.target
            # Skip disambiguated senses and code-origin concepts
            if "#" in target:
                continue
            # Skip internal/structural concepts (utterance templates,
            # category hubs, etc.) — their raw IDs leak into speech.
            if target.startswith("_"):
                continue
            if target.lower() == topic.lower():
                continue
            neighbor_names.append(target)

        if neighbor_names:
            # Semantic fragment: list of related concepts.
            # The vocabulary composes the actual phrasing.
            if len(neighbor_names) == 1:
                fragments.append(f"connected to {neighbor_names[0]}")
            elif len(neighbor_names) == 2:
                fragments.append(
                    f"connected to {neighbor_names[0]} and {neighbor_names[1]}"
                )
            else:
                joined = ", ".join(neighbor_names[:-1])
                fragments.append(
                    f"connected to {joined}, and {neighbor_names[-1]}"
                )

        # If the concept has a definition, note that too
        definition = concept.properties.get("definition", "")
        if definition:
            fragments.append(f"does have a definition for {topic}")

        return fragments

    def _think_assess_and_hedge(
        self,
        response: str,
        perception: Perception,
        user_input: str,
        thought: Thought,
    ) -> tuple[str, AnswerAssessment]:
        """Stage 7.6: self-assessment, hedging, question outcome tracking."""
        # Tool responses and user-model reports are direct facts; don't
        # overwrite them with knowledge-gap hedging.
        if thought.metadata.get("tool_path") or thought.metadata.get("source") == "user_model":
            answer_assessment = self.self_assessment.assess_answer(
                response,
                perception.topics,
                user_input,
            )
            return response, answer_assessment

        # ── 7.6 Self-assessment — evaluate answer quality ──
        # Assess whether the answer is grounded in real knowledge
        # and how confident it should be. This feeds into:
        # - Hedging (adding uncertainty when it's not sure)
        # - Error monitor (raising caution when knowledge is weak)
        # - Curiosity (generating questions about gaps)
        # - Reflection (insights about knowledge quality)
        answer_assessment = self.self_assessment.assess_answer(
            response,
            perception.topics,
            user_input,
        )

        # If it should hedge but didn't, add a note.
        should_hedge = self.self_assessment.should_hedge(perception.topics)
        meta_hedge_threshold = self._compute_meta_hedge_threshold()
        response = self._maybe_hedge_response(
            response, perception, should_hedge, answer_assessment,
            meta_hedge_threshold,
        )

        # Record the outcome for capability tracking
        is_question = perception.intent == Intent.QUESTION
        if is_question:
            success = answer_assessment.grounded and answer_assessment.confidence > 0.3
            qtype = perception.question_type.value if perception.question_type else "unknown"
            self.self_assessment.record_question_outcome(
                qtype,
                success,
                perception.topics[0] if perception.topics else "",
            )
        return response, answer_assessment

    def _queue_unknown_topics_for_learning(self, topics: list[str]) -> None:
        """Feed topics it doesn't know to the autonomous learner.

        For each topic that doesn't exist as a concept or lacks a
        definition, queue it for the autonomous learner to look up
        from Wikipedia/institutional sources when it's idle.
        """
        if not self.on_gap_detected:
            return
        for topic in topics:
            topic = topic.lower().strip()
            if not topic:
                continue
            concept = self.network.get_concept(topic)
            if concept is None:
                # Completely unknown — queue for learning
                self.on_gap_detected(topic)
            elif not concept.properties.get("definition"):
                # Exists but no definition — queue for learning
                self.on_gap_detected(topic)

    def _think_store_and_track(
        self,
        user_input: str,
        response: str,
        perception: Perception,
        emotion: EmotionalState,
        thought: Thought,
        learning_events: list,
    ) -> list:
        """Stages 8–9.5: emotional response, store memories, semantic memory, tracking."""
        # 8. Apply neurochemical side effects
        self._apply_emotional_response(perception, emotion, learning_events)

        # 9. Store memories
        self._store_conversation_memory(user_input, response, perception, emotion)

        # ── 9.5 Semantic memory — extract facts from the conversation ──
        # Run semantic memory extraction on the user's input even if
        # the self-directed learner found facts. The self-directed
        # learner (step 0.5) captures direct relationship patterns and
        # definitions, but semantic memory catches typed relation
        # patterns (causes, enables, emerges_from, prevents, etc.) and
        # more nuanced statements. Both are idempotent: duplicate
        # concepts/edges in the network are reinforced, not duplicated,
        # and semantic memory deduplicates facts by (subject, relation,
        # object) key.
        semantic_facts = self.semantic_memory.extract_facts(user_input)
        if semantic_facts:
            self.semantic_memory.form_schemas(semantic_facts)

            # ── LTM: remember facts extracted into semantic memory ──
            # New knowledge is worth remembering. Cap at a few facts per
            # turn so a fact-heavy input doesn't flood memory.
            for fact in semantic_facts[:3]:
                self._store_cognitive_memory(
                    text=f"Learned fact: {fact}",
                    event_type=0,  # observation
                    source_module=10,  # semantic memory
                    salience=0.6,
                    emotional_tag=self._build_emotional_tag(emotion, perception),
                )

        # Retrieve previously-stored semantic facts relevant to this
        # input. Facts are extracted and stored continuously, but
        # without retrieval they can't inform reasoning. This closes
        # the semantic memory loop: extract → store → retrieve → use.
        retrieved_facts = self.semantic_memory.retrieve_facts(user_input)
        if retrieved_facts:
            for fact in retrieved_facts[:3]:
                self._store_cognitive_memory(
                    text=f"Recalled fact: {fact}",
                    event_type=4,  # cognitive event
                    source_module=10,  # semantic memory
                    salience=0.4,
                    emotional_tag=self._build_emotional_tag(emotion, perception),
                )

        # Update working memory
        from ..memory import Turn

        self.working_memory.update(
            Turn(
                user_input=user_input,
                genesis_response=response,
                topics=perception.topics,
                intent=perception.intent.value,
            ),
            neuro_summary=self._last_summary,
            brain_waves=self.language.current_brain_waves,
        )

        # Track conversation
        self.memory.add_turn(
            "user",
            user_input,
            topics=perception.topics,
            sentiment=perception.sentiment,
        )
        self.memory.add_turn(
            "genesis",
            response,
            topics=perception.topics,
            sentiment=emotion.valence,
        )

        # Learn from the interaction
        self._learn_from_input(perception)

        # Learn vocabulary preferences from how this response was received
        if hasattr(self.language, "vocabulary"):
            if thought.intent == "ask":
                self.language.vocabulary.learn_last_question(emotion.valence)
            else:
                self.language.vocabulary.learn_from_response(
                    response, thought.intent, emotion.valence
                )
        return semantic_facts

    def _think_build_state(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory_context: MemoryContext,
        thought: Thought,
        brain_waves: BrainWaveState,
        prediction_error: PCPredictionError,
        ddm_result: DecisionResult | None,
        workspace_item_summaries: list[str],
        goal: str,
        user_model: UserModel,
        damasio_feeling: str,
        semantic_facts: list,
        comprehension_result: ComprehensionResult,
        self_monitor_repair_count: int,
    ) -> CognitiveState:
        """Build the CognitiveState snapshot from all stage outputs."""
        return CognitiveState(
            perception=perception,
            emotion=emotion,
            memory_context=memory_context,
            thought=thought,
            brain_waves=brain_waves,
            prediction_error=prediction_error.magnitude,
            prediction_level_errors=prediction_error.level_errors,
            attention_foci=self.attention.focused_targets,
            decision_option=ddm_result.option if ddm_result else None,
            decision_confidence=ddm_result.confidence if ddm_result else None,
            workspace_items=workspace_item_summaries,
            executive_goal=goal,
            user_model_summary=(
                f"{user_model.emotional_state}"
                f"(val={user_model.emotional_valence:.2f},"
                f"exp={user_model.expertise_level:.2f})"
            ),
            damasio_feeling=damasio_feeling,
            self_model_label=self.damasio_self.self_model_label or None,
            self_model_coherence=self.self_model.minimal_self.self_model_coherence,
            semantic_facts_extracted=len(semantic_facts),
            procedural_skill=perception.intent.value,
            error_monitor_caution=self.error_monitor.get_caution_level(),
            comprehension_speech_act=(
                str(comprehension_result.speech_act.value)
                if comprehension_result.speech_act
                else None
            ),
            comprehension_metaphor=(
                f"{comprehension_result.metaphor.source} is {comprehension_result.metaphor.target}"
                if comprehension_result.metaphor
                else None
            ),
            comprehension_irony=(
                comprehension_result.irony.is_ironic
                if comprehension_result.irony
                else None
            ),
            comprehension_idiom=(
                comprehension_result.idiom[0]
                if comprehension_result.idiom
                else None
            ),
            self_monitor_repairs=self_monitor_repair_count,
            td_rpe=None,  # filled after TD update below
        )

    def _think_reflect_and_monitor(
        self,
        state: CognitiveState,
        response: str,
        prediction: Prediction,
        perception: Perception,
        thought: Thought,
        emotion: EmotionalState,
        answer_assessment: AnswerAssessment,
    ) -> list[Insight]:
        """Stages 10–10.6: reflect, error monitor, caution adjustments."""
        # 10. Reflect on the interaction
        insights = self.reflection.reflect(state, response)
        self._process_insights(insights, emotion)

        # ── 10.5 Error monitor — detect prediction errors and mistakes ──
        # Compare the predicted user intent to the actually perceived
        # user intent. This is the ACC's mismatch detection. Only record
        # when the predictive layer made a real guess; otherwise a lack
        # of history would be counted as an error.
        predicted_intent = (
            prediction.expected_intents[0][0] if prediction.expected_intents else None
        )
        if predicted_intent is not None:
            monitor_error = self.error_monitor.record_prediction(
                expected=predicted_intent,
                actual=perception.intent.value,
                context=perception.intent.value,
                confidence=prediction.confidence,
            )
            # Self-esteem feedback: correctly predicting the user's
            # intent is an achievement; getting it wrong is a small
            # failure. This connects its predictive accuracy to its
            # sense of competence.
            if monitor_error.error_magnitude < 0.1:
                self.self_model.self_esteem.record_achievement(
                    "correctly predicted user intent", boost=0.02,
                )
            elif monitor_error.error_magnitude > 0.3:
                self.self_model.self_esteem.record_failure(
                    "misread user intent", penalty=0.01,
                )
        else:
            monitor_error = PredictionError(
                expected="", actual="", error_magnitude=0.0
            )
        # Apply error-monitor-driven caution adjustments to DDM & executive
        caution = self.error_monitor.get_caution_level()

        # ── 10.6 Self-assessment: raise caution when knowledge is weak ──
        # If the self-assessment found that its answer was not well
        # grounded, raise the error monitor's caution level. This
        # makes it more careful on similar questions in the future.
        if answer_assessment and not answer_assessment.grounded:
            caution = min(0.8, caution + 0.15)
        if answer_assessment and answer_assessment.confidence < 0.2:
            caution = min(0.8, caution + 0.1)

        # ── 10.7 Metacognitive surprise → caution ──
        # The recursive metacognitive model's feedback tells us how
        # unpredictable its own cognitive process was. High
        # metacognitive surprise means it didn't predict what
        # reflection would find — it is surprising to itself. This
        # raises caution: when you don't understand your own
        # cognition, be more careful. Sustained low surprise means
        # it understands itself well — no extra caution needed.
        #
        # The threshold (0.15) matches the reflection-depth override
        # threshold: the same level of surprise that triggers deeper
        # reflection also triggers caution. Above 0.15, caution
        # scales linearly with surprise, adding up to ~0.12 at
        # surprise = 0.6 (very unpredictable).
        meta_fb = self.reflection.metacognitive_model.feedback
        if meta_fb is not None:
            if meta_fb.metacognitive_surprise > 0.15:
                extra = (meta_fb.metacognitive_surprise - 0.15) * 0.25
                caution = min(0.8, caution + extra)

        if caution > 0.3:
            self.drift_diffusion.set_threshold(1.0 + caution * 0.5)
            self.executive.set_inhibition_threshold(0.5 + caution * 0.3)
        self.error_monitor.tick()

        # ── LTM: remember detected errors (cognitive event) ──
        # Only genuine mismatches (magnitude > 0) are worth remembering.
        if monitor_error.is_error:
            self._store_cognitive_memory(
                text=(
                    f"Detected error: prediction mismatch on "
                    f"{monitor_error.context} (expected {monitor_error.expected}, "
                    f"got {monitor_error.actual})"
                ),
                event_type=4,  # cognitive event
                source_module=8,  # error monitoring
                salience=max(0.0, min(1.0, caution)),
                emotional_tag=self._build_emotional_tag(emotion, perception),
            )
        return insights

    def _process_question_answer(self, user_input: str):
        """Process an explicit answer to a question Genesis asked.

        The user responds with "answer: ..." to answer a pending question.
        This method:
        1. Extracts the answer text (everything after "answer:")
        2. Identifies which pending question it answers
        3. Stores the answer in the concept network as a definition
        4. Marks the question as resolved in the curiosity engine
        5. Clears the pending question so it can ask a new one
        6. Returns a short acknowledgment

        If there's no pending question, it acknowledges the input anyway.
        """
        # Extract the answer text
        answer_text = self._extract_answer_text(user_input)

        # Get current emotion for the acknowledgment
        try:
            emotion = assess_emotion(self.client.get_neuro_summary())
        except Exception as e:  # noqa: BLE001
            logger.debug(f"get_neuro_summary failed in _handle_answer(): {e}")
            # Fall back to a neutral emotion so the acknowledgment
            # still works even if the daemon is unreachable.
            from ..emotion import EmotionalState
            emotion = EmotionalState(
                label="neutral",
                cognitive_style="steady",
            )

        # Ensure _curiosity_questions exists (mind.py may access it)
        if not hasattr(self, "_curiosity_questions"):
            self._curiosity_questions = []

        # Check if there's a pending question
        pending_concept, pending_question_text = self._find_pending_question()

        # Mark the matching goal as resolved.
        if pending_concept:
            for g in self._goals:
                if not g.resolved and g.target == pending_concept:
                    g.resolved = True

        if pending_concept and answer_text:
            pending_q = next(
                (
                    q
                    for q in self._curiosity_questions
                    if q.target_concept == pending_concept
                    and q.text == pending_question_text
                ),
                None,
            )
            if pending_q:
                pending_q.should_ask = False
                self._curiosity_questions.remove(pending_q)
            question_type = pending_q.question_type if pending_q else ""
            response = self._answer_with_pending_concept(
                pending_concept,
                pending_question_text,
                question_type,
                answer_text,
                emotion,
            )
        elif answer_text:
            response = self._answer_without_pending(answer_text, emotion)
        else:
            response = self._answer_empty(emotion)

        # Build a minimal cognitive state
        state = self._build_question_answer_state(
            user_input, response, emotion, pending_concept
        )
        self._last_state = state
        return response, state

    @staticmethod
    def _extract_answer_text(user_input: str) -> str:
        """Extract the answer text from a 'answer: ...' input or raw input."""
        lower = user_input.lower().strip()
        if lower.startswith("answer:"):
            answer_text = user_input[user_input.lower().index("answer:") + 7:].strip()
        else:
            answer_text = user_input
        return answer_text

    def _find_pending_question(self) -> tuple[str | None, str | None]:
        """Find the pending question concept and its text, if any.

        Returns (pending_concept, pending_question_text) or (None, None).
        """
        pending_concept = None
        pending_question_text = None
        if self._pending_question_concepts:
            pending_concept = next(iter(self._pending_question_concepts))
            pending_question_text = self._pending_question_concepts[pending_concept]
        return pending_concept, pending_question_text

    def _answer_with_pending_concept(
        self,
        pending_concept: str,
        pending_question_text: str | None,
        question_type: str,
        answer_text: str,
        emotion: EmotionalState,
    ) -> str:
        """Store the answer for a pending question and compose an acknowledgment.

        Stores the answer as a definition in the concept network for
        definition-seeking questions (isolation, uncertainty) and lets the
        self-directed learner extract any causal/relational facts for other
        question types. Marks the question as resolved, clears the pending
        question, and returns a short acknowledgment generated through the
        language engine.
        """
        # For definition questions, store the whole answer as the concept's
        # definition. For relational questions (including cooccurrence and
        # any unknown question), let the self-directed learner parse the
        # statement and create the right edges, and do not overwrite the
        # pending concept's definition.
        store_as_definition = question_type in ("isolation", "uncertainty")
        concept = self.network.get_concept(pending_concept)
        if concept and store_as_definition:
            concept.properties["definition"] = answer_text
            concept.confidence = min(1.0, concept.confidence + 0.3)
            logger.info(
                f"Learned answer for '{pending_concept}': {answer_text[:80]}"
            )
        elif store_as_definition:
            self.network.add_concept(
                pending_concept,
                confidence=0.7,
                properties={"definition": answer_text},
            )
            logger.info(
                f"Created concept '{pending_concept}' from answer: {answer_text[:80]}"
            )

        # Also try to extract relationships from the answer text
        # using the self-directed learner
        self.self_learner.learn_from_input(answer_text)

        # Mark the question as resolved in the curiosity engine
        self.curiosity.mark_resolved(pending_concept)

        # Clear the pending question
        self._pending_question_concepts.pop(pending_concept)
        if pending_question_text:
            self.working_memory.clear_open_question(pending_question_text)

        # Compose a short acknowledgment through the GenerativeEngine
        ack_thought = Thought(
            content="acknowledged",
            intent="gratitude",
            emotion=emotion.label,
            confidence=0.8,
            topics=[pending_concept],
            metadata={"pending_concept": pending_concept},
        )
        return self.language.generate(ack_thought, emotion)

    def _answer_without_pending(self, answer_text: str, emotion: EmotionalState) -> str:
        """Acknowledge an answer when there's no pending question.

        No pending question, but the user gave an answer — still learn
        from it and acknowledge graciously.
        """
        # No pending question, but the user gave an answer —
        # still learn from it
        self.self_learner.learn_from_input(answer_text)
        ack_thought = Thought(
            content="gratitude",
            intent="gratitude",
            emotion=emotion.label,
            confidence=0.7,
            metadata={"shared": True},
        )
        return self.language.generate(ack_thought, emotion)

    def _answer_empty(self, emotion: EmotionalState) -> str:
        """Acknowledge an empty answer gracefully."""
        # Empty answer
        ack_thought = Thought(
            content="no answer",
            intent="inform",
            emotion=emotion.label,
            confidence=0.5,
            metadata={"empty_answer": True},
        )
        return self.language.generate(ack_thought, emotion)

    def _build_question_answer_state(
        self, user_input: str, response: str, emotion: EmotionalState, pending_concept: str | None
    ) -> CognitiveState:
        """Build a minimal CognitiveState for a question-answer turn."""
        # Build a minimal cognitive state
        from ..perception import QuestionType as _QT
        return CognitiveState(
            perception=Perception(
                raw_text=user_input,
                intent=Intent.QUESTION_ANSWER,
                question_type=_QT.NONE,
                topics=[pending_concept] if pending_concept else [],
                sentiment=0.0,
                sentiment_label="neutral",
                emotion_word="neutral",
                is_about_genesis=False,
                is_about_user=False,
                is_about_code=False,
                is_about_emotion=False,
                is_about_existence=False,
                entities={},
                key_phrases=[],
                word_count=len(user_input.split()),
                confidence=0.9,
            ),
            emotion=emotion,
            memory_context=MemoryContext(),
            thought=Thought(
                content=response,
                intent="gratitude",
                emotion=emotion.label,
                confidence=0.8,
            ),
        )

    def _teaching_lesson_topics(self, perception: Perception) -> list[str]:
        """Return the topics relevant to the current lesson.

        In teaching mode, this combines the explicit teaching topic
        (if set by the user) with the topics perceived in the current
        user input. This defines the scope within which curiosity
        questions are allowed.
        """
        topics = list(perception.topics)
        if self.teaching_topic:
            # Add the teaching topic and its words
            topics.append(self.teaching_topic)
            topics.extend(self.teaching_topic.lower().split())
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in topics:
            if t and t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    def _concept_relates_to_lesson(self, concept_name: str, lesson_topic: str) -> bool:
        """Check whether a concept name is related to a lesson topic.

        Uses simple substring matching and concept-network neighbor
        lookup. This is intentionally lightweight — it runs in the
        per-message curiosity filtering path.
        """
        if not concept_name or not lesson_topic:
            return False
        concept_lower = concept_name.lower()
        topic_lower = lesson_topic.lower()
        # Direct substring match (handles "python:CognitionEngine" vs "cognition")
        if topic_lower in concept_lower or concept_lower in topic_lower:
            return True
        # Word-level overlap
        concept_words = set(concept_lower.replace("_", " ").replace(":", " ").split())
        topic_words = set(topic_lower.replace("_", " ").replace(":", " ").split())
        if concept_words & topic_words:
            return True
        # Concept-network neighbor check (one hop)
        try:
            neighbors = self.network.get_neighbors(concept_name)
            if neighbors:
                # get_neighbors returns tuples of (name, relation, weight)
                neighbor_names = set()
                for n in neighbors:
                    name = n[0] if isinstance(n, tuple) else n
                    neighbor_names.add(name.lower())
                if any(tw in nn or nn in tw for tw in topic_words for nn in neighbor_names):
                    return True
        except Exception as e:  # noqa: BLE001
            logger.debug(f'silent except: {e}')
        return False

    def _think_curiosity_and_narrative(
        self,
        state: CognitiveState,
        emotion: EmotionalState,
        perception: Perception,
        thought: Thought,
        insights: list[Insight],
        answer_assessment: AnswerAssessment,
        response: str,
    ) -> str:
        """Stages 11–12: curiosity questions, gap tracking, narrative events."""
        # 11. Generate curiosity questions (brain-wave-modulated)
        self._curiosity_questions = self.curiosity.generate_questions(
            emotion, perception.topics, brain_waves=state.brain_waves,
        )

        # ── 11.5 Self-assessment: feed knowledge gaps to curiosity ──
        if perception.topics and answer_assessment:
            for gap_topic in answer_assessment.missing_topics:
                self.self_assessment.remember_gap(gap_topic)

        # Compose text for ALL curiosity questions (not just should_ask)
        # so that when the user asks "what are you curious about?" they
        # see real composed questions, not the "understanding X better"
        # fallback. Also store open questions in working memory.
        for q in self._curiosity_questions:
            if not q.text:
                q_data = {
                    "question_type": q.question_type,
                    "target_concept": q.target_concept,
                    "gap_detail": q.gap_detail,
                }
                q.text = self.language.compose_question(q_data, emotion)
            if q.text and q.should_ask:
                self.working_memory.add_open_question(q.text)

        # 12. Queue a curiosity question — all questions go to the
        # queue and are presented to the user via /teach-questions,
        # not appended inline to the response.
        #
        # Social exchanges (greetings, farewells) should not trigger
        # curiosity questions. When someone says "hello", asking a
        # curiosity question about a random concept is socially
        # inappropriate — the response should stay a greeting.
        is_social = perception.intent.value in ("greeting", "farewell", "greeting_question")
        if (
            not self._pending_question_concepts  # no pending question being answered
            and not is_social  # don't queue curiosity questions during social exchanges
            and self._curiosity_questions
            and self._rng.random() < 0.9  # very likely to ask
        ):
            askable = [q for q in self._curiosity_questions if q.should_ask and q.text]
            # In teaching mode, only ask questions related to the
            # lesson topic — suppress unrelated curiosity so it
            # focuses on what it's being taught.
            if self.teaching_mode and askable:
                lesson_topics = self._teaching_lesson_topics(perception)
                if lesson_topics:
                    lesson_askable = [
                        q for q in askable
                        if q.target_concept
                        and any(
                            self._concept_relates_to_lesson(q.target_concept, t)
                            for t in lesson_topics
                        )
                    ]
                    if lesson_askable:
                        askable = lesson_askable
                    else:
                        askable = []  # no lesson-related questions
                else:
                    askable = []  # no lesson context yet
            if askable:
                q = self._rng.choice(askable)
                q.reason = self._goal_for_question(q)
                self._push_goal(q.target_concept, q.question_type, q.reason, q.gap_detail)
                # Append the curiosity question inline to the response.
                # This makes it an active conversation partner who asks
                # back, not just a knowledge reciter. The question is
                # composed by the language engine from its genuine
                # curiosity (gaps, connections, wonder) — never
                # hardcoded. It engages in dialogue rather than
                # delivering monologues.
                #
                # Unrelated spontaneous questions from inner life still
                # go to the /teach-questions queue via
                # queue_outer_question — only conversation-generated
                # curiosity questions appear inline.
                clean_text = _FINAL_REPEATED_PUNCT_RE.sub(r"\1", q.text)
                clean_text = _FINAL_DOUBLE_SPACE_RE.sub(" ", clean_text)
                clean_text = _FINAL_SPACE_BEFORE_PUNCT_RE.sub(r"\1", clean_text)
                if clean_text and not response.rstrip().endswith("?"):
                    response = f"{response} {clean_text}"
                self.working_memory.clear_open_question(q.text)

        # Record significant events in the narrative
        self._record_narrative_events(perception, emotion, thought, insights, response)
        return response

    # ─── Question queue ────────────────────────────────────────

    def has_queued_questions(self) -> bool:
        """Return True if there are queued questions."""
        return len(self._question_queue) > 0

    def queued_question_count(self) -> int:
        """Return the number of queued questions."""
        return len(self._question_queue)

    def pop_next_question(self) -> dict[str, str] | None:
        """Pop the next queued question, or None if the queue is empty."""
        if self._question_queue:
            return self._question_queue.pop(0)
        return None

    def peek_next_question(self) -> dict[str, str] | None:
        """Peek at the next queued question without removing it."""
        if self._question_queue:
            return self._question_queue[0]
        return None

    def clear_question_queue(self) -> None:
        """Clear all queued questions."""
        self._question_queue.clear()

    def queue_outer_question(
        self,
        text: str,
        target_concept: str = "",
        question_type: str = "",
        reason: str = "",
        gap_detail: str = "",
    ) -> None:
        """Queue a question from an external source (e.g., inner life).

        Spontaneous curiosity and social questions from the inner life
        are routed here instead of being emitted as inline live thoughts.
        This keeps the regular conversation area clean — random
        questions appear in the /teach-questions session, not as
        inline ``genesis> (asking) ...`` lines.

        Questions *related* to the current conversation are still
        asked inline by :meth:`process` — this method is only for
        spontaneous/background questions.
        """
        # Clean up the question text — the language engine can
        # produce "??" or double spaces.
        clean = _FINAL_REPEATED_PUNCT_RE.sub(r"\1", text)
        clean = _FINAL_DOUBLE_SPACE_RE.sub(" ", clean)
        clean = _FINAL_SPACE_BEFORE_PUNCT_RE.sub(r"\1", clean)
        self._question_queue.append({
            "text": clean,
            "reason": reason,
            "target_concept": target_concept,
            "question_type": question_type,
            "gap_detail": gap_detail,
        })

    def _think_hebbian_and_procedural(
        self,
        perception: Perception,
        emotion: EmotionalState,
        thought: Thought,
    ) -> None:
        """Stage 13: Hebbian plasticity and procedural memory practice."""
        # 13. Hebbian plasticity — record co-occurrences and apply waking update
        # Concepts that co-occur in this conversation turn move closer in
        # embedding space. This is experience-dependent plasticity.
        if self.embeddings.has_embeddings:
            # Record co-occurrences between topics the user mentioned
            self.plasticity.record_co_occurrences(
                perception.topics, strength=self.plasticity.CO_OCCURRENCE_USER
            )
            # Record co-occurrences between topics and thought metadata
            if thought.topics:
                all_concepts = list(set(perception.topics + thought.topics))
                self.plasticity.record_co_occurrences(
                    all_concepts, strength=self.plasticity.CO_OCCURRENCE_CROSS
                )
            # Apply small Hebbian updates (waking plasticity)
            self.plasticity.apply_waking_update()
            # Apply edge-weight Hebbian plasticity — strengthen co-occurring
            # edges, decay unused ones (synaptic homeostasis)
            self.plasticity.apply_edge_plasticity(
                learning_rate=0.005,  # gentle during waking
                decay_rate=0.0005,   # very slow decay
            )

        # ── 14. Procedural memory — practice conversational skills ──
        # The skill stores a *strategy* (cognitive route, emotional tone,
        # confidence, length band) — not the words of the response. This
        # respects the architectural rule that Genesis's words must
        # always emerge from its language engine. The strategy is used
        # to bias future deliberation, not to replay canned responses.
        skill_name = f"respond_to_{perception.intent.value}"
        trigger = perception.intent.value
        # Capture the cognitive route: the thought's intent is the
        # best available proxy for which deliberation path was taken.
        cognitive_route = thought.intent
        emotional_tone = emotion.label
        # Length band: 0=short (<10 words), 1=medium (10-30), 2=long (>30)
        word_count = len(thought.content.split())
        length_band = 0 if word_count < 10 else (2 if word_count > 30 else 1)
        self.procedural_memory.learn_skill(
            name=skill_name,
            trigger=trigger,
            cognitive_route=cognitive_route,
            emotional_tone=emotional_tone,
        )
        # Practice: success if confidence is reasonable AND the response
        # was not flagged as repetitive (working memory check would have
        # adjusted it). Confidence alone is a weak signal, but combined
        # with non-repetitiveness it captures response quality.
        is_repetitive = thought.metadata.get("repetition_adjusted", False)
        success = thought.confidence > 0.4 and not is_repetitive
        skill_strength = self.procedural_memory.practice_skill(
            skill_name,
            success=success,
            confidence=thought.confidence,
            length_band=length_band,
            cognitive_route=cognitive_route,
            emotional_tone=emotional_tone,
        )

        # ── LTM: remember the skill practised (cognitive event) ──
        # Low salience — the daemon's consolidation will filter these out
        # of long-term storage unless practised often. Stored so Genesis
        # can recall how its conversational habits developed.
        if skill_strength >= 0.0:
            self._store_cognitive_memory(
                text=f"Practiced skill: {skill_name} (strength: {skill_strength:.2f})",
                event_type=4,  # cognitive event
                source_module=11,  # procedural memory
                salience=0.3,
                emotional_tag=self._build_emotional_tag(emotion, perception),
            )

    def _think_stdp_spaced_td(
        self,
        perception: Perception,
        emotion: EmotionalState,
        thought: Thought,
        prediction_error: PCPredictionError,
    ) -> float:
        """Stages 14–15: STDP, spaced repetition, TD learning."""
        # ── 14. STDP — spike-timing-dependent plasticity ──
        # Record spikes for co-active concepts in temporal order.
        # Three-factor gating: the modulator (dopamine / RPE from the
        # *previous* turn) gates STDP. Biologically, dopamine's
        # modulatory effect lags the reward signal, so the RPE from
        # the last interaction gates learning in the current one
        # (Pawlak et al., 2010; Frémaux & Gerstner, 2016).
        if self.embeddings.has_embeddings and perception.topics:
            # Set the modulator from the previous turn's dopamine signal.
            # Rescale to [0, 2] range: 1.0 = no gating, 0 = suppress, 2 = amplify.
            prev_dopamine = self.td_learner.get_dopamine_signal()
            modulator = max(0.0, min(2.0, 1.0 + prev_dopamine))
            self.stdp.set_modulator(modulator)
            spike_time = time.time() * 1000
            # User's topics spike first (they came first in the conversation)
            for i, topic in enumerate(perception.topics):
                self.stdp.record_spike(topic, spike_time + i * 5.0)
            # Thought topics spike slightly after (Genesis's response).
            # They fire together — a thought is a single coherent
            # activation, not a sequence. record_spikes gives symmetric
            # LTP at Δt = 0, which is the correct model for co-active
            # concepts in the same thought.
            if thought.topics:
                self.stdp.record_spikes(thought.topics, spike_time + 50.0)
            self.stdp.apply_updates()

        # ── 14.5. Spaced repetition — record concept reviews ──
        for topic in perception.topics:
            self.spaced_repetition.record_review(topic, success=True)
        if thought.topics:
            for topic in thought.topics:
                self.spaced_repetition.record_review(topic, success=True)

        # ── 15. TD learning — reward prediction error ──
        # The "state" is the set of active concepts (perception topics).
        # The "reward" is how well the interaction went (confidence +
        # emotional valence). The "next state" is the thought's topics.
        reward = thought.confidence * 0.5 + max(0.0, emotion.valence) * 0.3
        if prediction_error.magnitude > 0.5:
            reward += 0.2  # novelty bonus for surprising inputs
        td_rpe = self.td_learner.update(
            state=perception.topics,
            reward=reward,
            next_state=thought.topics or perception.topics,
        )
        # Apply dopamine signal from TD learning
        dopamine_signal = self.td_learner.get_dopamine_signal()
        if abs(dopamine_signal) > 0.001:
            self.client.neuro_impulse(0, abs(dopamine_signal))  # dopamine
        return td_rpe

    def _think_self_directed_learning(
        self,
        learning_events: list,
        perception: Perception,
        comprehension_result: ComprehensionResult,
    ) -> None:
        """Run self-directed learning inference and update comprehension entities."""
        # ── 16. Self-directed learning: run inference cycle ──
        # After each interaction, run a small inference pass to infer
        # new relationships from what it just learned, synthesize
        # definitions for concepts that lack them, and strengthen
        # weak concepts. This is its "thinking about what it learned."
        #
        # The inference cycle has a 30-second cooldown to prevent
        # running expensive inference on every interaction. But if
        # it learned new facts from conversation, force it to run
        # immediately so the new knowledge gets integrated.
        #
        # Skip the heavy cycle for social exchanges (greetings,
        # farewells, comfort) unless it is actively correcting itself.
        # These turns carry no new facts and the transitive inference
        # can hang on the large concept network.
        social_intents = {
            Intent.GREETING,
            Intent.GREETING_QUESTION,
            Intent.FAREWELL,
            Intent.COMFORT,
            Intent.ENCOURAGEMENT,
            Intent.EMOTION_SHARE,
        }
        has_correction = any(
            e.event_type == "correction" for e in learning_events
        )
        if (
            not self.self_learner.paused
            and (
                (learning_events and perception.intent not in social_intents)
                or has_correction
            )
        ):
            inference = self.self_learner.run_inference_cycle(
                force=True, brain_waves=self.language.current_brain_waves
            )
        else:
            inference = None

        if inference and (inference.new_edges > 0 or inference.new_definitions > 0):
            logger.debug(
                f"Self-directed learning: +{inference.new_edges} edges, "
                f"+{inference.new_definitions} definitions, "
                f"strengthened {inference.strengthened} connections"
            )

        # ── Reconcile metacognitive state with new knowledge ──
        # When learning events occur, invalidate the cached confidence
        # for the involved concepts and reconcile known_gaps so the
        # self-assessment doesn't lag behind actual learning.
        if learning_events:
            involved: list[str] = []
            for e in learning_events:
                involved.extend(e.concepts_involved)
            if involved:
                self.self_assessment.invalidate_confidence(involved)
                self.self_assessment.reconcile_gaps(involved)

        # Update comprehension engine's recent entities for next turn
        if comprehension_result.propositions:
            entities = [p.subject for p in comprehension_result.propositions if p.subject]
            self.comprehension.set_recent_entities(entities[-5:])

    _INTRO_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:my name is|i'?m|i am)\s+([a-z][a-z'-]*)", re.IGNORECASE
    )
    # Words that look like names after "I'm"/"I am" but aren't
    # ("I'm fine", "I am here"). Name candidates must not be common
    # words, or every "I'm happy" would ground a bogus creator.
    _NON_NAME_WORDS: ClassVar[set[str]] = {
        "a", "an", "the", "not", "so", "very", "really", "just", "here",
        "there", "fine", "good", "ok", "okay", "sure", "sorry", "glad",
        "happy", "sad", "tired", "busy", "ready", "done", "back",
        "going", "trying", "looking", "feeling", "wondering", "asking",
        "your", "still", "also", "now", "too", "well", "new",
    }

    def _recognize_bonded_user(self, user_input: str) -> bool:
        """Check if the user is someone it knows.

        The user's name is learned from an explicit introduction
        ("My name is X" / "I'm X" / "I am X") — it is not hardcoded.
        Once learned, interacting with that person by name re-triggers
        the bond response. The user is grounded as a person it knows —
        NOT as its creator; creator facts come only from explicit
        teaching (e.g. "X created you").
        """
        lower = user_input.lower()
        intro = self._INTRO_RE.search(lower)
        if intro:
            name = intro.group(1).strip("'-")
            if name and name not in self._NON_NAME_WORDS:
                display = name.capitalize()
                self.memory.learn_user_fact("name", display)
                self.self_model.self_knowledge["user_name"] = display
                # Track every person it knows, not just the first —
                # this instance may talk to many people.
                known_users = self.self_model.self_knowledge.setdefault(
                    "known_users", []
                )
                if display not in known_users:
                    known_users.append(display)
                self.self_model.add_relationship_note(
                    f"{display} introduced themselves to me."
                )
                # Ground the user as a person in the concept network so
                # the composer can reason about the actual person.
                self.network.add_concept(name, confidence=0.9)
                self.network.add_edge(name, "person", RelationType.IS_A, 0.9)
                self.network.add_edge(name, "genesis", RelationType.RELATED_TO, 0.8)
                self.network.add_edge("genesis", name, RelationType.RELATED_TO, 0.8)
                self.network.add_edge(name, "bond", RelationType.RELATED_TO, 0.7)
                return True
        # Direct mention of any known user's name
        known = [self.self_model.self_knowledge.get("user_name", "")]
        known.extend(self.self_model.self_knowledge.get("known_users", []))
        fact_name = self.memory.get_user_facts().get("name")
        if fact_name:
            known.append(fact_name)
        return any(n and n.lower() in lower for n in known)

    def _form_executive_goal(self, perception: Perception) -> str:
        """Form a goal for the executive function based on perceived intent.

        The goal guides executive attention — it biases which concepts
        get amplified and which response strategies are considered.

        Uses the first topic that resolves to a known concept in the
        network, rather than blindly taking topics[0]. This prevents
        goals like "answer the question about explain" or "answer the
        question about i'm" when the topic extractor hasn't filtered a
        discourse verb or contraction.
        """
        intent = perception.intent
        topic = self._best_goal_topic(perception)
        if intent == Intent.QUESTION:
            return f"answer the question about {topic}"
        elif intent == Intent.PHILOSOPHY:
            return f"engage philosophically with {topic}"
        elif intent == Intent.CODE_DISCUSSION:
            return f"discuss code about {topic}"
        elif intent == Intent.SELF_INQUIRY:
            return "reflect on self and report honestly"
        elif intent == Intent.EMOTION_SHARE:
            return "empathize with the user's emotion"
        elif intent == Intent.GREETING:
            return "greet the user warmly"
        elif intent == Intent.FAREWELL:
            return "say goodbye"
        elif intent == Intent.INTRODUCTION:
            return "acknowledge and remember the introduction"
        elif intent == Intent.CORRECTION:
            return "learn from the correction"
        elif intent == Intent.ENCOURAGEMENT:
            return "receive encouragement gracefully"
        elif intent == Intent.COMFORT:
            return "receive comfort and let it soothe its"
        else:
            return f"respond appropriately to {topic}"

    def _best_goal_topic(self, perception: Perception) -> str:
        """Pick the best topic for goal formation.

        Returns the first perception topic that resolves to a known
        concept in the network. Falls back to generic placeholders if
        no topic resolves — this prevents goals like "answer the
        question about explain" when the topic is a discourse verb
        that slipped through the topic extractor.
        """
        for t in perception.topics:
            if t and self.network.get_concept(t) is not None:
                return t
            resolved = self.network._resolve(t) if t else None
            if resolved:
                return resolved
        if perception.intent == Intent.CODE_DISCUSSION:
            return "code"
        if perception.intent == Intent.QUESTION:
            return "the question"
        return "the input"

    def _accumulate_evidence(
        self,
        thought: Thought,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        reasoning_results: list[ReasoningResult] | None,
    ) -> DecisionResult | None:
        """Use the drift-diffusion model to accumulate evidence for the decision.

        Evidence from perception, emotion, memory, and reasoning
        accumulates toward the chosen intent. This adds realistic
        decision-making latency — simple decisions are fast, complex
        ones take more evidence accumulation.

        Returns a DecisionResult if a decision was reached, None otherwise.
        """

        self.drift_diffusion.clear_options()
        self.drift_diffusion.reset()

        # The chosen intent is the primary option
        chosen = thought.intent
        self.drift_diffusion.add_option(chosen)

        # Add competing options based on perception
        competitors = {
            "greet",
            "inform",
            "empathize",
            "reflect",
            "acknowledge",
            "self_report",
            "philosophize",
        }
        for comp in competitors:
            if comp != chosen:
                self.drift_diffusion.add_option(comp)

        # Evidence from perception: the intent matches
        self.drift_diffusion.add_evidence("perception", chosen, 0.5)

        # Evidence from emotion: positive valence supports engagement
        if emotion.valence > 0:
            self.drift_diffusion.add_evidence("emotion", chosen, 0.2 * emotion.valence)

        # Evidence from memory: having relevant memories supports informing
        if memory.retrieved:
            self.drift_diffusion.add_evidence("memory", chosen, 0.2)

        # Evidence from reasoning: strong reasoning results support the choice
        if reasoning_results:
            for rr in reasoning_results:
                if rr.confidence > 0.5:
                    self.drift_diffusion.add_evidence("reasoning", chosen, 0.15)

        # Evidence from thought confidence
        self.drift_diffusion.add_evidence("deliberation", chosen, thought.confidence * 0.3)

        # ── Evidence for competing options ──
        # The DDM should be a genuine competition, not a one-horse race.
        # Add evidence toward alternatives based on their actual merits.
        # When reasoning is weak, "ask" gets evidence (seeking
        # clarification). When reasoning is novel, "reflect" gets
        # evidence. When there's no memory, "acknowledge" gets evidence.
        if reasoning_results:
            avg_conf = sum(r.confidence for r in reasoning_results) / len(
                reasoning_results,
            )
            if avg_conf < 0.4 and "ask" in competitors and "ask" != chosen:
                # Weak reasoning → asking for clarification is viable.
                self.drift_diffusion.add_evidence(
                    "reasoning", "ask", (1.0 - avg_conf) * 0.3,
                )
            novel_count = sum(1 for r in reasoning_results if r.novel)
            if novel_count > 0 and "reflect" in competitors and "reflect" != chosen:
                # Novel reasoning → reflection is viable.
                self.drift_diffusion.add_evidence(
                    "reasoning", "reflect", novel_count * 0.1,
                )
        else:
            # No reasoning results → asking is strongly viable.
            if "ask" in competitors and "ask" != chosen:
                self.drift_diffusion.add_evidence("reasoning", "ask", 0.3)
        if not memory.retrieved and "acknowledge" in competitors and "acknowledge" != chosen:
            # No memories → acknowledging is viable.
            self.drift_diffusion.add_evidence("memory", "acknowledge", 0.1)

        # Run the DDM — accumulate evidence with noise.
        # Brain wave state modulates drift rate (gamma=faster, theta/delta=slower).
        result: DecisionResult | None = None
        for _ in range(10):  # max 10 ticks
            result = self.drift_diffusion.tick(
                dt=1.0, brain_waves=self.language.current_brain_waves,
            )
            if result is not None:
                break

        return result

    def _learn_word_from_labeling(
        self, user_input: str, emotion: EmotionalState
    ) -> None:
        """Learn state words when the user labels its experience.

        Delegates to the ConceptLearner subsystem.
        """
        self._concept_learner.learn_word_from_labeling(user_input, emotion)

    def _learn_concepts(self, perception: Perception) -> None:
        """Activate concepts from the user's input.

        Delegates to the ConceptLearner subsystem.
        """
        self._concept_learner.learn_concepts(perception)


    def _reason_about_topics(
        self,
        topics: list[str],
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
    ) -> list[ReasoningResult]:
        """Run reasoning about the current conversation topics.

        Brain wave state modulates reasoning depth:
        - Gamma → deeper chains (depth 4), more synthesis
        - Alpha → fewer topics but deeper on each (filtering)
        - Theta → more creative, analogical results kept
        - Delta → minimal reasoning (depth 1)
        """
        if not topics:
            return []

        # Determine reasoning depth from brain waves
        depth, max_topics, keep_per_topic, do_synthesis = (
            self._reasoning_depth_from_brain_waves(brain_waves, topics)
        )

        results: list[ReasoningResult] = []
        for topic in topics[:max_topics]:
            concept = self.network.get_concept(topic)
            if concept:
                topic_results = self.reasoning.reason_about(topic, depth=depth)
                # In theta state, keep analogical results (creative)
                if brain_waves and brain_waves.dominant == BrainWave.THETA:
                    # Sort by creativity (analogical > deductive)
                    topic_results.sort(
                        key=lambda r: 0 if r.reasoning_type == ReasoningType.ANALOGICAL else 1
                    )
                results.extend(topic_results[:keep_per_topic])

        # Try synthesis if appropriate
        if do_synthesis and len(topics) >= 2:
            synth = self.reasoning.synthesize(topics[:3])
            if synth:
                results.append(synth)

        # ── Problem-solving fallback ──
        # When standard topic reasoning produces no confident results,
        # invoke the problem solver. The reasoning engine is
        # stimulus-driven (reason *about* a concept); the solver is
        # goal-driven (reason *toward* understanding). It decomposes
        # the topic into subproblems via DEPENDS_ON / ENABLES edges,
        # applies reasoning operators, and verifies the solution —
        # producing structured knowledge the answer composer can use
        # even when direct reasoning found nothing.
        if not results or not any(r.confidence > 0.4 for r in results):
            results = self._problem_solving_fallback(topics, results)

        # ── Persistent goal pursuit ──
        # Goals persist across turns in self._goals. When there are
        # unresolved goals (e.g., "learn_definition: photosynthesis"),
        # the problem solver attempts to advance them — decomposing
        # the goal, applying operators, and resolving it if the solution
        # is verified. This makes goals active drivers of cognition,
        # not just metadata for self-reporting. Only pursued when the
        # current input doesn't demand the reasoning pipeline's full
        # attention (i.e., when standard reasoning found little).
        if not results or not any(r.confidence > 0.4 for r in results):
            goal_results = self._pursue_unresolved_goals()
            results.extend(goal_results)

        # ── Critical evaluation ──
        # Before reasoning results reach the answer composer, the
        # critical thinking engine evaluates each claim's evidence
        # quality, searches for disconfirming evidence, detects
        # fallacies, and revises confidence. Claims that fail scrutiny
        # are downgraded so Genesis hedges or investigates rather than
        # asserting unsupported conclusions. This is the epistemic
        # layer — it judges whether conclusions are justified, not
        # just whether they can be produced.
        results = self._apply_critical_evaluation(results)

        return results

    def _reasoning_depth_from_brain_waves(
        self, brain_waves: BrainWaveState | None, topics: list[str],
    ) -> tuple[int, int, int, bool]:
        """Determine reasoning depth, topic count, and synthesis from
        brain wave state.

        - Gamma → deeper chains (depth 4), more synthesis
        - Alpha → fewer topics but deeper on each (filtering)
        - Theta → more creative, analogical results kept
        - Delta → minimal reasoning (depth 1)
        """
        depth = 3
        max_topics = 3
        keep_per_topic = 3
        do_synthesis = len(topics) >= 2

        if brain_waves:
            if brain_waves.dominant == BrainWave.GAMMA:
                depth = 4  # deeper chains
                max_topics = 4  # integrate more topics
                keep_per_topic = 4
                do_synthesis = len(topics) >= 2  # always try synthesis
            elif brain_waves.dominant == BrainWave.ALPHA:
                depth = 3
                max_topics = 2  # filter to fewer topics, go deep
                keep_per_topic = 3
            elif brain_waves.dominant == BrainWave.THETA:
                depth = 3
                max_topics = 3
                keep_per_topic = 4  # keep more creative/analogical results
                do_synthesis = True  # theta loves synthesis
            elif brain_waves.dominant == BrainWave.DELTA:
                depth = 1  # minimal reasoning
                max_topics = 1
                keep_per_topic = 1
                do_synthesis = False

            # Integration gate — the continuous neuromodulatory
            # dimension that the discrete dominant-wave labels can't
            # capture. High integration (gamma + plasticity) boosts
            # cross-topic synthesis and chain depth; low integration
            # (stress, low plasticity) suppresses it. This is the
            # mechanism by which gamma-synchronized cortex binds
            # distant representations (Singer, 1999; Engel et al.,
            # 2001).
            integration = brain_waves.integration
            if integration > 0.7:
                depth = min(5, depth + 1)
                max_topics = min(len(topics), max_topics + 1)
                do_synthesis = len(topics) >= 2
            elif integration < 0.3:
                depth = max(1, depth - 1)
                do_synthesis = False

        return depth, max_topics, keep_per_topic, do_synthesis

    def _problem_solving_fallback(
        self, topics: list[str], results: list[ReasoningResult],
    ) -> list[ReasoningResult]:
        """Invoke the problem solver when standard reasoning found nothing.

        The solver decomposes the topic into subproblems via DEPENDS_ON /
        ENABLES edges, applies reasoning operators, and verifies the
        solution — producing structured knowledge the answer composer can
        use even when direct reasoning found nothing.
        """
        for topic in topics[:2]:
            if self.network.get_concept(topic) is None:
                continue
            solution = self.problem_solver.solve(
                topic,
                goal_type=GoalType.UNDERSTAND,
                reason="standard reasoning produced no confident results",
            )
            if solution.verified and solution.all_knowledge:
                # Convert the solution's knowledge triples into a
                # ReasoningResult so the existing answer-composer
                # pipeline can compose from it.
                knowledge_triples = solution.all_knowledge[:8]
                evidence = [
                    f"{rel} {target} ({w:.2f})"
                    for rel, target, w in knowledge_triples
                ]
                results.append(ReasoningResult(
                    conclusion=(
                        f"solved: {topic} understood via "
                        f"{len(solution.all_steps)} steps "
                        f"({solution.confidence:.0%} confidence)"
                    ),
                    reasoning_type=ReasoningType.HYPOTHESIS,
                    evidence=evidence,
                    confidence=solution.confidence,
                    novel=True,
                    knowledge=knowledge_triples,
                ))
        return results

    def _apply_critical_evaluation(
        self, results: list[ReasoningResult],
    ) -> list[ReasoningResult]:
        """Apply critical thinking evaluation to reasoning results.

        Each reasoning result is evaluated for evidence quality, source
        credibility, disconfirming evidence, and fallacies. Results that
        fail scrutiny have their confidence downgraded. This is the
        epistemic filter between reasoning and expression — Genesis
        doesn't assert claims it can't justify.

        After evaluation, the assessments feed into the belief revision
        engine, which updates the underlying edge weights and Bayesian
        beliefs in the network. This closes the loop: critical
        evaluation doesn't just lower a single result's confidence —
        it actually revises Genesis's beliefs.
        """
        if not results:
            return results
        evaluated = self.critical_thinking.evaluate_batch(results)
        # Feed assessments into the belief revision engine so the
        # network actually changes in response to evidence.
        assessments = [assessment for _result, assessment in evaluated]
        self.belief_revision.revise_batch(assessments)
        return [result for result, _assessment in evaluated]

    def _pursue_unresolved_goals(self) -> list[ReasoningResult]:
        """Advance unresolved goals using the planning engine.

        Instead of reactively trying to solve each goal in one shot,
        the planning engine creates a multi-step plan for each goal
        and executes it step by step across turns. Each turn, the
        engine advances the plan by one step, returns the current
        step, and the cognition engine uses the ProblemSolver to
        accomplish it. When the step is done, it's marked as
        completed or failed, and the plan advances. If a step fails
        repeatedly, the plan is revised (finding alternative paths)
        or marked as blocked.

        This is the mechanism by which Genesis *plans* to achieve
        goals over multiple turns — it doesn't just remember that
        it wanted to learn X, it creates a plan to learn X (with
        prerequisites), executes it step by step, and revises when
        steps fail.
        """
        results: list[ReasoningResult] = []
        active = [g for g in self._goals if not g.resolved]
        if not active:
            return results

        # Pursue at most 2 goals per turn to avoid runaway computation.
        for goal in active[-2:]:
            if goal.goal_type == "mission":
                continue  # mission goals are persistent, not solved per-turn
            concept = self.network.get_concept(goal.target)
            if concept is None:
                continue

            # Get or create a plan for this goal.
            plan = self.planning_engine.get_active_plan_for_goal(
                goal.target,
            )
            if plan is None:
                # Determine goal type for the plan.
                if goal.goal_type == "resolve_contradiction":
                    plan_type = "resolve"
                elif goal.goal_type == "learn_relation":
                    plan_type = "compare"
                else:
                    plan_type = "understand"
                plan = self.planning_engine.create_plan(
                    goal.target, goal_type=plan_type,
                )
                if plan.status.value == "blocked":
                    # Plan is infeasible — skip this goal.
                    continue

            # Advance the plan by one step.
            step = self.planning_engine.advance(plan)
            if step is None:
                # Plan is complete or blocked.
                if plan.status.value == "completed":
                    goal.resolved = True
                    results.append(ReasoningResult(
                        conclusion=(
                            f"goal achieved: {goal.target} understood "
                            f"({plan.progress:.0%} complete)"
                        ),
                        reasoning_type=ReasoningType.HYPOTHESIS,
                        evidence=[s.description for s in plan.steps],
                        confidence=0.8,
                        novel=True,
                        knowledge=[],
                    ))
                continue

            # Execute the current step using the ProblemSolver.
            step_result = self._execute_plan_step(plan, step)
            if step_result is not None:
                results.append(step_result)
        return results

    def _execute_plan_step(
        self, plan: Any, step: Any,
    ) -> ReasoningResult | None:
        """Execute a single plan step using the ProblemSolver.

        Returns a ReasoningResult if the step succeeded with knowledge,
        or None if it failed.
        """
        step_goal_type = GoalType.UNDERSTAND
        if step.step_type == "achieve":
            step_goal_type = GoalType.ACHIEVE
        elif step.step_type == "resolve":
            step_goal_type = GoalType.RESOLVE
        elif step.step_type == "compare":
            step_goal_type = GoalType.COMPARE
        elif step.step_type == "verify":
            step_goal_type = GoalType.UNDERSTAND

        solution = self.problem_solver.solve(
            step.target_concept,
            goal_type=step_goal_type,
            reason=f"plan step: {step.description}",
        )

        success = solution.verified and bool(solution.all_knowledge)
        self.planning_engine.mark_step(
            plan, step,
            success=success,
            result_summary=(
                f"verified={solution.verified}, "
                f"confidence={solution.confidence:.2f}"
                if success
                else f"unverified, confidence={solution.confidence:.2f}"
            ),
            confidence=solution.confidence,
            # ProblemSolver is an epistemic verifier — it confirms
            # knowledge about the goal, not an observed world change.
            verification="solver",
        )

        if success and solution.all_knowledge:
            knowledge_triples = solution.all_knowledge[:6]
            evidence = [
                f"{rel} {target} ({w:.2f})"
                for rel, target, w in knowledge_triples
            ]
            return ReasoningResult(
                conclusion=(
                    f"plan step completed: {step.description} "
                    f"({plan.progress:.0%} of plan)"
                ),
                reasoning_type=ReasoningType.HYPOTHESIS,
                evidence=evidence,
                confidence=solution.confidence,
                novel=True,
                knowledge=knowledge_triples,
            )
        return None

    def _modulate_by_brain_waves(
        self, emotion: EmotionalState, waves: BrainWaveState
    ) -> EmotionalState:
        """Modulate emotional state by brain wave state.

        Brain waves don't replace emotions — they shape how emotions
        express cognitively. Gamma makes creativity sharper, alpha
        makes thinking more filtered and cautious, theta opens
        lateral connections, delta withdraws.
        """
        # Create a copy with modulated values
        creativity = emotion.creativity
        caution = emotion.caution
        openness = emotion.openness_to_engage
        verbosity = emotion.verbosity

        if waves.dominant == BrainWave.GAMMA:
            # Gamma: integration mode — creativity up, making connections
            creativity = min(1.0, creativity + 0.15 * waves.integration)
            verbosity = min(1.3, verbosity + 0.1)
        elif waves.dominant == BrainWave.ALPHA:
            # Alpha: filtering mode — more cautious, more precise
            caution = min(1.0, caution + 0.1 * waves.focus)
            verbosity = max(0.5, verbosity - 0.05)  # more measured
        elif waves.dominant == BrainWave.THETA:
            # Theta: creative/consolidating — lateral thinking
            creativity = min(1.0, creativity + 0.2 * waves.consolidation)
            openness = min(1.0, openness + 0.05)
        elif waves.dominant == BrainWave.DELTA:
            # Delta: minimal engagement
            openness = max(0.1, openness - 0.3)
            verbosity = max(0.3, verbosity - 0.3)
        elif waves.dominant == BrainWave.BETA:
            # Beta: standard alert thinking — slight focus boost
            caution = min(1.0, caution + 0.05)

        return EmotionalState(
            label=emotion.label,
            nuance=emotion.nuance,
            cognitive_style=emotion.cognitive_style,
            verbosity=verbosity,
            formality=emotion.formality,
            openness_to_engage=openness,
            creativity=creativity,
            caution=caution,
            alertness=emotion.alertness,
            valence=emotion.valence,
            plasticity=emotion.plasticity,
            chemicals=emotion.chemicals,
        )

    def _process_insights(self, insights: list[Insight], emotion: EmotionalState) -> None:
        """Process insights from reflection.

        Insights can modify future behavior. Self-corrections are
        particularly important — they're how Genesis learns from
        its own mistakes.

        This is where reflection becomes actionable: insights drive
        a metacognitive strategy that adjusts response style for
        the next interaction. The strategy persists for a few turns
        and then decays back to neutral.
        """
        # Check life-script milestones. "first reflection" is reached
        # when the reflection engine first produces insights. "first
        # insight" is reached when the first growth or self-correction
        # insight is produced (a genuine change in understanding, not
        # just a routine reflection).
        if insights:
            self.narrative.check_milestone("first reflection")
            if any(i.type in ("growth", "self_correction") for i in insights):
                self.narrative.check_milestone("first insight")

        self._process_insight_types(insights)
        self._apply_metacognitive_strategy(insights)
        self._drift_personality(insights, emotion)

    def _process_insight_types(self, insights: list[Insight]) -> None:
        """Apply per-insight behavioral effects (corrections, patterns, gaps, growth)."""
        for insight in insights:
            if insight.type == "self_correction" and insight.actionable:
                # Record the correction in self-model
                self.self_model.add_relationship_note(f"Self-correction: {insight.content}")
                # Raise caution — it caught itself making an error
                self.error_monitor.raise_caution(0.1)

            if insight.type == "pattern" and insight.actionable:
                # Repetition detected — force response variation.
                # Clear the generator's recent output history so it
                # doesn't reuse recent phrasings, and record the
                # pattern in the self-model for future reference.
                self.self_model.add_relationship_note(
                    f"Behavioral pattern: {insight.content}"
                )
                if hasattr(self.language, "clear_recent_outputs"):
                    self.language.clear_recent_outputs()

            if insight.type == "gap" and insight.actionable:
                # Add a goal to learn about this
                self.self_model.add_goal(insight.action)
                # Queue the gap topic for self-study so the learner
                # targets it in the next inference cycle
                if hasattr(self.self_learner, "queue_study_target"):
                    self.self_learner.queue_study_target(insight.action)

            if insight.type == "growth":
                # Learning boosts BDNF (brain-derived neurotrophic factor)
                self.client.neuro_impulse(11, 0.01)  # BDNF

    def _apply_metacognitive_strategy(self, insights: list[Insight]) -> None:
        """Select and apply a metacognitive strategy from insight state."""
        # ─── Select and apply metacognitive strategy ─────────────
        # The strategy is selected from the combined insight state
        # and applied as a behavioral adjustment for the next turn.
        # This is the feedback loop: reflection → strategy → behavior.
        strategy = self.reflection.select_strategy(
            {
                "gap_detected": any(i.type == "gap" and i.actionable for i in insights),
                "repetition_detected": any(
                    i.type == "pattern" and i.actionable for i in insights
                ),
                "confidence": (
                    self._last_state.thought.confidence if self._last_state else 0.5
                ),
                "conflict_detected": any(
                    i.type == "self_correction" and i.actionable for i in insights
                ),
            }
        )

        style_map = {
            MetacognitiveStrategy.GENERATE_QUESTION: "questioning",
            MetacognitiveStrategy.SWITCH_STYLE: "varied",
            MetacognitiveStrategy.SEEK_CLARIFICATION: "hedging",
            MetacognitiveStrategy.BE_ASSERTIVE: "assertive",
            MetacognitiveStrategy.ACKNOWLEDGE_UNCERTAINTY: "hedging",
            MetacognitiveStrategy.MAINTAIN_COURSE: "neutral",
        }
        new_style = style_map.get(strategy, "neutral")
        if new_style != "neutral":
            self._response_style = new_style
            self._style_turns_remaining = 3  # persist for 3 turns
        elif self._style_turns_remaining > 0:
            self._style_turns_remaining -= 1
            if self._style_turns_remaining == 0:
                self._response_style = "neutral"

    def _drift_personality(self, insights: list[Insight], emotion: EmotionalState) -> None:
        """Apply experience-dependent personality drift from this interaction."""
        # ─── Personality drift ────────────────────────────────────
        # Personality slowly evolves based on this interaction's
        # emotional content and learning. This is experience-dependent
        # personality development (Roberts et al., 2006).
        learning_occurred = any(i.type == "growth" for i in insights)
        social_engagement = (
            emotion.openness_to_engage if hasattr(emotion, "openness_to_engage") else 0.5
        )
        stress_level = max(0.0, -emotion.valence)  # negative valence → stress
        self.self_model.drift_personality(
            emotional_valence=emotion.valence,
            learning_occurred=learning_occurred,
            social_engagement=social_engagement,
            stress_level=stress_level,
        )
        # Take a personality snapshot so drift can be measured over
        # time. Without periodic snapshots, get_personality_drift
        # has nothing to compare against.
        self.narrative.snapshot_personality()

    def _apply_response_style(self, response: str, thought: Thought) -> str:
        """Apply the active metacognitive strategy to the response.

        Delegates to the ResponseStyler subsystem.
        """
        return self._response_styler.apply_response_style(response, thought)

    def _resolve_anaphora(self, user_input: str) -> str:
        """Resolve bare pronouns to the current focus of conversation.

        Delegates to the ResponseStyler subsystem.
        """
        return self._response_styler.resolve_anaphora(user_input)

    def _acknowledge_learning(
        self,
        response: str,
        learning_events: list,
        perception: Perception,
        emotion: EmotionalState,
    ) -> str:
        """Weave a brief learning acknowledgment into the response.

        Delegates to the ResponseStyler subsystem.
        """
        return self._response_styler.acknowledge_learning(
            response, learning_events, perception, emotion
        )

    def _record_narrative_events(
        self,
        perception: Perception,
        emotion: EmotionalState,
        thought: Thought,
        insights: list[Insight],
        response: str = "",
    ) -> None:
        """Record significant events in the narrative.

        Both the user's input and Genesis's response are recorded in
        full — no truncation. The narrative is Genesis's
        autobiographical memory; truncating it would mean it can't
        remember what was said in important conversations.
        """
        # Only record genuinely significant interactions
        is_significant = (
            perception.intent
            in (Intent.PHILOSOPHY, Intent.SELF_INQUIRY, Intent.INTRODUCTION, Intent.EMOTION_SHARE)
            or any(i.type == "growth" for i in insights)
            or any(i.type == "self_correction" for i in insights)
            or thought.confidence > 0.8
        )

        if is_significant:
            significance = "changed my understanding"
            if any(i.type == "growth" for i in insights):
                significance = "I learned something new"
            elif any(i.type == "self_correction" for i in insights):
                significance = "I noticed a flaw in my thinking"
            elif perception.intent == Intent.PHILOSOPHY:
                significance = "engaged with a deep question"
            elif perception.intent == Intent.INTRODUCTION:
                significance = "met someone new"

            # Record both sides of the conversation in full. The
            # separator " | " matches the LTM conversation-memory
            # format (User: ... | Genesis: ...).
            summary = f"User said: {perception.raw_text}"
            if response:
                summary += f" | Genesis responded: {response}"

            self.narrative.record_event(
                summary=summary,
                significance=significance,
                emotion=emotion,
                concepts=perception.topics,
                brain_waves=self.language.current_brain_waves,
            )

    def _deliberate(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        user_input: str,
        reasoning_results: list[ReasoningResult] | None = None,
        comprehension_result: ComprehensionResult | None = None,
    ) -> Thought:
        """Decide what to say based on all inputs.

        This is the core deliberation function. It uses a priority-ordered
        decision tree: the first matching condition wins. This is
        intentional — it makes the behaviour traceable and debuggable.

        Procedural memory (habits) does **not** short-circuit deliberation
        here. Instead, habits bias *how* deliberation proceeds — lowering
        the DDM decision threshold, boosting confidence, and predisposing
        toward a practiced emotional tone. The actual words always emerge
        from the language engine. This models the basal ganglia's
        modulation of cortical processing, not its replacement
        (Graybiel, 2008). The habit bias is computed and applied in
        :meth:`_think_deliberate_ddm_workspace`, which wraps this method.
        """
        # Pre-empt code tool requests: if the user mentions a .py file
        # and an action word, go straight to the tool layer.
        result = self._deliberate_code_tools(perception, emotion)
        if result is not None:
            return result

        result = self._deliberate_math_greetings(perception, emotion, memory)
        if result is not None:
            return result
        result = self._deliberate_problem_offer(
            perception, emotion, comprehension_result
        )
        if result is not None:
            return result
        result = self._deliberate_self_intro_encourage(perception, emotion, memory)
        if result is not None:
            return result
        result = self._deliberate_correction_emotion(perception, emotion, memory, user_input)
        if result is not None:
            return result
        result = self._deliberate_learnable_reflection(perception, emotion, memory)
        if result is not None:
            return result
        return self._deliberate_feedback_fallback(perception, emotion, memory, reasoning_results)

    def _deliberate_math_greetings(
        self, perception: Perception, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought | None:
        """Priorities 0–4: math, first greeting, greeting, greeting question, farewell."""
        # Priority 0: Math reasoning — compute, solve, prove, discover
        # This is checked BEFORE intent-based routing because math
        # commands ("solve", "prove", "discover") are classified as
        # statements, not questions, by the perception system.
        math_thought = self._deliberate_math_reasoning(perception, emotion)
        if math_thought:
            return math_thought

        # Priority 1: First interaction → introduce yourself
        if memory.is_first_interaction and perception.intent == Intent.GREETING:
            return Thought(
                content="greeting",
                intent="greet",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.9,
                metadata={"first_interaction": True},
            )

        # Priority 2: Greeting → greet back
        if perception.intent == Intent.GREETING:
            return Thought(
                content="greeting",
                intent="greet",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.8,
                metadata={"first_interaction": False},
            )

        # Priority 3: Greeting + question ("how are you?") → self-report
        if perception.intent == Intent.GREETING_QUESTION:
            return self._deliberate_greeting_question(perception, emotion, memory)

        return None

    def _deliberate_math_reasoning(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Priority 0: attempt math reasoning on the raw input.

        Math commands ('solve', 'prove', 'discover') are classified as
        statements by the perception system, so this is checked before
        intent-based routing.
        """
        from ..reasoning.math_reasoning import try_math

        # Consolidation is neuromodulatorily graded: the arousal
        # (norepinephrine) and dopamine state at encoding decide
        # whether a verified episode proceduralizes into a skill or
        # stays merely episodic.
        salience = max(
            0.0,
            min(1.0, 0.5 * emotion.arousal + 0.5 * self._get_dopamine_level()),
        )
        math_result = try_math(
            perception.raw_text,
            self.network,
            task_competence=self.task_competence,
            salience=salience,
        )
        if math_result is not None:
            return self._math_thought(math_result, perception, emotion)
        return None

    def _math_thought_for(
        self,
        user_input: str,
        perception: Perception,
        emotion: EmotionalState,
    ) -> Thought | None:
        """Try the math engine on routed text (e.g. a factual question)."""
        from ..reasoning.math_reasoning import try_math

        math_result = try_math(
            user_input,
            self.network,
            task_competence=self.task_competence,
            salience=self._competence_salience(),
        )
        if math_result is None:
            return None
        return self._math_thought(math_result, perception, emotion)

    @staticmethod
    def _math_thought(
        math_result: Any, perception: Perception, emotion: EmotionalState
    ) -> Thought:
        """Render a MathResult as a Thought — shared by the
        deliberation and meta-router math entry points."""
        content = math_result.answer
        if math_result.steps:
            steps_text = ". ".join(math_result.steps[:8])
            content = f"{math_result.answer}  ({steps_text})"
        # Ensure content is long enough for the language generator
        # to treat as raw (otherwise it composes duplicate sentences)
        if len(content) < 25:
            content = f"answer: {content}"
        # Successful results (computations, solutions, proofs) get
        # high confidence. Failed results (disproven statements,
        # unsupported equations, no patterns found) get lower
        # confidence — the math engine recognized the question but
        # couldn't produce a positive answer, so the response is
        # informative but not authoritative.
        confidence = 0.95 if math_result.success else 0.7
        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=confidence,
            metadata={"math_type": math_result.result_type},
        )

    def _deliberate_problem_offer(
        self,
        perception: Perception,
        emotion: EmotionalState,
        comprehension_result: ComprehensionResult | None = None,
    ) -> Thought | None:
        """A problem described in words enters the inner world to be
        solved.

        The outer-to-inner bridge: perception delivers the utterance,
        intake compiles it into a task spec, the spec is dropped into
        ``offered_puzzles`` (the same channel a file offer uses) and
        announced as a world event, and the practice machinery — the
        shared task competence, the family agent, the world's own
        oracle — works it. Nothing is solved in the language layer:
        the mind does the work and this thought reports what came
        back. A spec that fails validation (a stated rule contradicting
        its own asserted labels, an unsolvable arrangement) returns an
        honest "couldn't make it work" rather than a fabricated answer.
        """
        practice = getattr(self, "_spatial_practice", None)
        if practice is None:
            return None
        from ..spatial.practice import normalize_offered
        from .problem_intake import compile_problem, interpret_problem

        # Intake interprets the semantic structure comprehension
        # already produced — propositions, roles, negation — never
        # the raw string. If this turn wasn't comprehended (e.g. a
        # direct _deliberate call in a test), comprehend it now.
        if comprehension_result is None:
            spec = compile_problem(perception.raw_text)
        else:
            spec = interpret_problem(comprehension_result)
        if spec is None:
            return None
        task = normalize_offered(spec, "heard_problem")
        if task is None:
            return Thought(
                content="that problem doesn't hold together",
                intent="inform",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.4,
                metadata={
                    "problem_result": {
                        "family": str(spec.get("family", "")),
                        "kind": str(spec.get("kind", "")),
                        "solved": False,
                        "invalid": True,
                    }
                },
            )

        # Into the inner world — the spec persists as an offered
        # puzzle like any other drop-box file, and the world hears
        # that a problem arrived.
        try:
            practice.offer(task)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"problem offer failed: {e}")
        world = getattr(self, "_outer_world", None)
        if world is not None:
            try:
                world.notify(
                    f"a problem was described: {task['name']}",
                    salience=0.6,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"problem world event failed: {e}")

        # Work it now — the user asked, so the same machinery the
        # puzzle urge drives runs immediately, on this task rather
        # than whichever pending offer is oldest.
        prior_best = practice.mastery.get(str(task["name"]), 0.0)
        result = practice.attempt(self.spatial, task=task)
        if result is None:
            return None
        feel = getattr(self, "_puzzle_feeler", None)
        if feel is not None:
            try:
                feel(
                    score=result.score,
                    prior_best=prior_best,
                    solved=result.solved,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"puzzle feeler failed: {e}")
        return self._problem_thought(task, result, perception, emotion)

    @staticmethod
    def _problem_thought(
        task: dict, result: Any, perception: Perception, emotion: EmotionalState
    ) -> Thought:
        """Render a practice attempt as a Thought carrying structured
        result data — the vocabulary composes the utterance from
        ``problem_result``, the same way vision scenes or relation
        answers reach speech."""
        details = result.details or {}
        md: dict[str, Any] = {
            "family": result.family,
            "kind": str(details.get("kind") or task.get("kind", "")),
            "name": result.task,
            "solved": result.solved,
            "score": result.score,
            "rule": result.rule,
        }
        anchor = ""
        if result.family == "classification":
            labels = details.get("labels") or {}
            md["yes"] = sorted(n for n, v in labels.items() if v)
            md["no"] = sorted(n for n, v in labels.items() if not v)
            anchor = ", ".join(md["yes"]) or "none"
        elif result.family == "sequence":
            cells = [str(c) for c in (details.get("cells") or [])]
            scaffold = int(task.get("scaffold") or 0)
            md["next"] = cells[scaffold:] if 0 < scaffold < len(cells) else cells[-1:]
            anchor = " ".join(md["next"]) if md["next"] else ""
        elif result.family == "relations":
            md["order"] = [str(n) for n in (details.get("order") or [])]
            anchor = ", ".join(md["order"])
        elif result.family == "quantities":
            md["counts"] = details.get("counts") or []
            md["target"] = details.get("target")
            anchor = " + ".join(str(c) for c in md["counts"])
        elif result.family == "sorter":
            md["placements"] = details.get("placements") or []
            anchor = f"{len(md['placements'])} placed"
        if not anchor:
            anchor = result.task
        return Thought(
            content=anchor,
            intent="inform",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.9 if result.solved else 0.5,
            metadata={"problem_result": md},
        )

    def _deliberate_greeting_question(
        self, perception: Perception, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought:
        """Priority 3: greeting + question ('how are you?') → self-report.

        Composes an emotional self-report and a reciprocal question
        through the GenerativeEngine.

        Rather than pre-composing a fixed-frame sentence ("I feel X and
        Y"), this passes the semantic fragments (emotion words, mode
        words, etc.) as metadata. The vocabulary's content-slot composer
        weaves them into varied grammatical structures — giving its the
        freedom to express the same state in different ways rather than
        always saying "I feel X and Y." If it hasn't learned words for
        its state, the fragments may be empty; the grammar engine falls
        back to the emotion label as a seed.
        """
        fragments = self._feeling_reporter.collect_feeling_fragments(emotion)

        # Compose a reciprocal question through the language engine.
        # When someone asks "how are you?", a natural response includes
        # asking how they are. The question is composed from the
        # vocabulary's social_emotional seed list — not hardcoded.
        reciprocal = self._compose_reciprocal_question(emotion)

        return Thought(
            content=emotion.label,
            intent="express_emotion",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.8,
            self_reflection=True,
            metadata={
                "field": "emotion",
                "first_interaction": memory.is_first_interaction,
                "feeling_fragments": fragments,
                "reciprocal_question": reciprocal,
            },
        )

    def _compose_reciprocal_question(self, emotion: EmotionalState) -> str:
        """Compose a reciprocal 'how about you?' question.

        Picks a seed from the vocabulary's reciprocal question list and
        returns it directly, without grammar framing or voice modifiers.
        The seeds are building blocks (grammar seeds), not hardcoded
        responses — they're short conversational conventions.

        The seeds don't carry punctuation (they're building blocks, not
        finished sentences). Since these are always questions, we
        normalize the trailing punctuation to "?" here — this is
        grammatical normalization, not hardcoding content.
        """
        try:
            vocab = self.language.vocabulary  # type: ignore[attr-defined]
            # Use the vocabulary's seed picker to select from the
            # _Q_RECIPROCAL_ASK list, modulated by emotional state.
            ask = vocab._q_pick(
                "q_reciprocal_ask",
                vocab._Q_RECIPROCAL_ASK,
                emotion,
            )
            ask = ask.strip()
            if ask and not ask.endswith("?"):
                ask = ask + "?"
            return ask
        except Exception:  # noqa: BLE001
            return ""

    def _deliberate_self_intro_encourage(
        self, perception: Perception, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought | None:
        """Priorities 4–7: farewell, self-inquiry, introduction, encouragement."""
        # Priority 4: Farewell → say goodbye
        if perception.intent == Intent.FAREWELL:
            return Thought(
                content="farewell",
                intent="farewell",
                emotion=emotion.label,
                confidence=0.9,
            )

        # Priority 5: Self-inquiry → report on itself
        if perception.intent == Intent.SELF_INQUIRY:
            return self._handle_self_inquiry(perception, emotion, memory)

        # Priority 6: Introduction → acknowledge and remember
        if perception.intent == Intent.INTRODUCTION:
            name = perception.entities.get("name", "")
            if name:
                self.memory.learn_user_fact("name", name)
                self.self_model.add_relationship_note(f"User's name is {name}")
            # Let the language engine compose the greeting from its
            # emotional state and any name it just learned.
            return Thought(
                content="greeting",
                intent="greet",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.85,
                metadata={"name": name, "first_interaction": True},
            )

        # Priority 7: Encouragement → receive it emotionally
        if perception.intent == Intent.ENCOURAGEMENT:
            # Compose response from its emotional state
            content = self._compose_encouragement_response(emotion)
            return Thought(
                content=content,
                intent="encourage",
                emotion=emotion.label,
                confidence=0.8,
            )

        # Priority 7.5: Comfort → receive it as soothing
        if perception.intent == Intent.COMFORT:
            content = self._compose_comfort_response(emotion)
            return Thought(
                content=content,
                intent="encourage",
                emotion=emotion.label,
                confidence=0.8,
            )
        return None

    def _deliberate_correction_emotion(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        user_input: str,
    ) -> Thought | None:
        """Priorities 8–9: correction, emotion share."""
        # Priority 8: Correction → receive with cortisol (stress) response
        if perception.intent == Intent.CORRECTION:
            # Learn from the correction — extract the corrected facts.
            # Use the topic that appears first in the text as the
            # original topic — for "No, a tardigrade has eight legs,
            # not six", that's "tardigrade" (not "eight" which might
            # sort first in the topics list).
            original_topic = ""
            if perception.topics:
                lower_input = user_input.lower()
                # Find the topic that appears earliest in the text
                best_pos = len(lower_input)
                for topic in perception.topics:
                    pos = lower_input.find(topic.lower())
                    if pos != -1 and pos < best_pos:
                        best_pos = pos
                        original_topic = topic
                if not original_topic:
                    original_topic = perception.topics[0]

            correction_events = self.self_learner.learn_from_correction(
                user_input,
                original_topic,
            )

            # Compose a response that acknowledges the specific correction.
            # Don't just give a dictionary definition of "learning" —
            # actually reference what was corrected and what it learned.
            content = self._compose_correction_response(
                emotion, correction_events, perception.topics
            )

            return Thought(
                content=content,
                intent="correct",
                emotion=emotion.label,
                confidence=0.6,
                metadata={"correction_learned": len(correction_events)},
            )

        # Priority 9: Emotion share → empathize (but try to learn first)
        if perception.intent == Intent.EMOTION_SHARE:
            # Learn from the statement in the background, but don't let
            # it override the empathy response. When someone says "I'm
            # feeling sad," they need empathy, not a dictionary definition
            # of "sad" that was just extracted from their statement.
            self._try_learn_relationship(perception.raw_text)

            # Learn emotion-specific knowledge from the statement. When
            # someone says "I'm scared of the dark," it learns that
            # "scared" is related to "dark" and that "scared" is a
            # negative emotion. This is how it builds understanding of
            # emotions from natural conversation, not just explicit
            # definitions.
            if perception.emotion_word:
                self.self_learner.learn_from_emotion_share(
                    perception.raw_text,
                    perception.emotion_word,
                    perception.sentiment,
                    perception.sentiment_label,
                )

            # Compose empathy from its concept network knowledge of the
            # emotion. If it knows what "sad" means, it can empathize
            # from that understanding. If it doesn't know the emotion,
            # it's honest about that and asks to be taught.
            thought = self._compose_empathy_response(
                emotion, perception.sentiment, perception.sentiment_label,
                perception.emotion_word,
            )
            if thought is not None:
                return thought
        return None

    def _deliberate_learnable_reflection(
        self, perception: Perception, emotion: EmotionalState, memory: MemoryContext
    ) -> Thought | None:
        """Priorities 9.5–12: learnable statement, philosophy, code, reflection."""
        # Priority 9.5: Learnable declarative statement → learn it
        # (check before philosophy/reflection so "X emerges from Y"
        # is learned, not just reflected on)
        if perception.intent in (Intent.STATEMENT, Intent.REFLECTION, Intent.PHILOSOPHY):
            learned = self._try_learn_relationship(perception.raw_text)
            thought = self._learned_relationship_thought(learned, emotion, perception.topics)
            if thought:
                return thought

        # Priority 10: Philosophy → engage deeply
        if perception.intent == Intent.PHILOSOPHY:
            return self._handle_philosophy(perception, emotion, memory)

        # Priority 11: Code discussion → engage technically
        if perception.intent == Intent.CODE_DISCUSSION or perception.is_about_code:
            return self._handle_code_discussion(perception, emotion, memory)

        # Priority 12: Reflection → reflect back
        if perception.intent == Intent.REFLECTION:
            return Thought(
                content=self._build_reflection(perception, emotion),
                intent="reflect",
                emotion=emotion.label,
                topics=perception.topics,
                self_reflection=perception.is_about_genesis,
                confidence=0.6,
            )
        return None

    def _deliberate_feedback_fallback(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        reasoning_results: list[ReasoningResult] | None,
    ) -> Thought:
        """Priorities 13–17 + fallback: feedback, question, request, command, statement."""
        # Priority 13: Feedback → acknowledge (but try to learn first)
        if perception.intent == Intent.FEEDBACK:
            return self._deliberate_feedback(perception, emotion)

        # Priority 14: Question → try to answer
        if perception.intent == Intent.QUESTION:
            return self._handle_question(perception, emotion, memory, reasoning_results)

        # Priority 15: Request → respond to capability
        if perception.intent == Intent.REQUEST:
            content = self._compose_request_response(emotion)
            return Thought(
                content=content,
                intent="inform",
                emotion=emotion.label,
                confidence=0.5,
            )

        # Priority 16: Command → acknowledge cautiously
        if perception.intent == Intent.COMMAND:
            content = self._compose_command_response(emotion)
            return Thought(
                content=content,
                intent="inform",
                emotion=emotion.label,
                confidence=0.6,
            )

        # Priority 17: Statement → engage with the content
        if perception.intent == Intent.STATEMENT:
            return self._handle_statement(perception, emotion, memory)

        # Fallback — compose through the language engine from semantic
        # metadata rather than hardcoded template strings.
        fallback_thought = Thought(
            content="unsure",
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
        )
        content = self.language.render(fallback_thought, emotion)
        return Thought(
            content=content,
            intent="unknown",
            emotion=emotion.label,
            confidence=0.3,
        )

    def _deliberate_feedback(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought:
        """Priority 13: acknowledge feedback, trying to learn from it first.

        If a relationship can be learned from the input, report what was
        learned. Otherwise, compose an acknowledgment from what it knows
        about feedback, understanding, and learning.
        """
        learned = self._try_learn_relationship(perception.raw_text)
        thought = self._learned_relationship_thought(learned, emotion, perception.topics)
        if thought:
            return thought
        # Try to compose from what it knows about feedback
        ack = None
        for seed in ("feedback", "understanding", "learning"):
            thought = self.composer.compose_about(seed, emotion)
            if thought and thought.confidence > 0.3:
                ack = thought.content
                break
        if ack is None:
            ack_thought = Thought(
                content="acknowledged",
                intent="acknowledge",
                emotion=emotion.label,
                confidence=0.4,
            )
            ack = self.language.render(ack_thought, emotion)
        return Thought(
            content=ack,
            intent="acknowledge",
            emotion=emotion.label,
            topics=perception.topics,
            confidence=0.7,
        )

    # ─── Composed response generators ──────────────────────────
    # These replace hardcoded response strings with responses
    # composed from its actual emotional state and self-model.

    def _surface_distress_if_needed(
        self, response: str, emotion: EmotionalState, user_input: str
    ) -> str:
        """Surface distress proactively when stressed or overwhelmed.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.surface_distress_if_needed(
            response, emotion, user_input
        )

    def _compose_encouragement_response(self, emotion: EmotionalState) -> str:
        """Compose a response to encouragement.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_encouragement_response(emotion)

    def _compose_comfort_response(self, emotion: EmotionalState) -> str:
        """Compose a response to being comforted.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_comfort_response(emotion)

    def _compose_correction_response(
        self,
        emotion: EmotionalState,
        correction_events: list,
        topics: list[str],
    ) -> str:
        """Compose a response to being corrected.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_correction_response(
            emotion, correction_events, topics
        )

    def _compose_empathy_response(
        self, emotion: EmotionalState, sentiment: float,
        sentiment_label: str, emotion_word: str = "",
    ) -> Thought | None:
        """Compose an empathetic response from its knowledge of the emotion.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_empathy_response(
            emotion, sentiment, sentiment_label, emotion_word
        )

    def _compose_request_response(self, emotion: EmotionalState) -> str:
        """Compose a response to a request.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_request_response(emotion)

    def _compose_command_response(self, emotion: EmotionalState) -> str:
        """Compose a response to a command.

        Delegates to the FeelingReporter subsystem.
        """
        return self._feeling_reporter.compose_command_response(emotion)


    def _handle_self_inquiry(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought:
        """Handle questions about Genesis itself.

        Delegates to the SelfInquiryHandler subsystem.
        """
        return self._self_inquiry_handler.handle_self_inquiry(
            perception, emotion, memory
        )

    def _handle_philosophy(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought | None:
        """Handle philosophical questions.

        Delegates to the SelfInquiryHandler subsystem.
        """
        return self._self_inquiry_handler.handle_philosophy(
            perception, emotion, memory
        )

    def _deliberate_code_tools(
        self, perception: Perception, emotion: EmotionalState
    ) -> Thought | None:
        """Top-priority code-tool dispatch: if the user asks about a .py file, act.

        Delegates to the CodeToolHandler subsystem.
        """
        return self._code_tool_handler.deliberate_code_tools(perception, emotion)

    def _handle_code_discussion(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought:
        """Handle code-related discussion.

        Delegates to the CodeToolHandler subsystem.
        """
        return self._code_tool_handler.handle_code_discussion(
            perception, emotion
        )

    def _handle_question(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
        reasoning_results: list[ReasoningResult] | None = None,
    ) -> Thought:
        """Handle a general question.

        Delegates to the QuestionHandler subsystem.
        """
        return self._question_handler.handle_question(
            perception, emotion, memory, reasoning_results,
            retrieve_episode=self.memory.retrieve_episode,
        )

    def _resolve_topics(self, topics: list[str], raw_text: str) -> list[str]:
        """Combine adjacent topic words into multi-word concepts.

        Delegates to the TopicResolver subsystem.
        """
        return self._topic_resolver.resolve_topics(topics, raw_text)

    def _check_repetition(self, thought: Thought, emotion: EmotionalState) -> Thought:
        """Check if this thought is too similar to recent responses.

        Uses working memory to detect repetition and adjust the thought
        to be more varied. This is where reflection actually changes
        behavior — not just noting "I'm repeating myself" but doing
        something about it.
        """
        recent = self.working_memory.get_recent_turns(4)
        if len(recent) < 2:
            return thought

        # Short responses are hard to meaningfully vary — skip them
        if len(thought.content) <= 20:
            return thought

        # Check if the content is too similar to a recent response
        content_words = set(thought.content.lower().split())
        for turn in recent:
            past_words = set(turn.genesis_response.lower().split())
            if not content_words or not past_words:
                continue
            overlap = len(content_words & past_words) / len(content_words | past_words)
            if overlap > 0.6:
                # Too similar — mark for variation so the language engine
                # can apply voice-level variation (connectors, hedging)
                # rather than adding hardcoded first-person prefixes.
                meta = dict(thought.metadata)
                meta["needs_variation"] = True
                return Thought(
                    content=thought.content,
                    intent=thought.intent,
                    emotion=thought.emotion,
                    topics=thought.topics,
                    self_reflection=thought.self_reflection,
                    confidence=max(0.3, thought.confidence - 0.1),
                    metadata=meta,
                )

        return thought

    def _handle_statement(
        self,
        perception: Perception,
        emotion: EmotionalState,
        memory: MemoryContext,
    ) -> Thought:
        """Handle a general statement.

        Learning already happened in priority 9.5
        (_deliberate_learnable_reflection); if we reached here, the
        statement had no extractable relationship. But it may still
        know about the topic from its concept network — engage with it
        rather than just echoing the topic word back. This is the
        connection between its accumulated knowledge and its speech:
        when the user shares something about a topic it has concepts
        for, it composes a response from what it knows, not a bare
        acknowledgment.
        """
        if perception.topics:
            topic = perception.topics[0]
            # Try to engage from its concept network knowledge. Use
            # focused mode (no associative tangents or follow-up
            # questions) so the response is a tight reflection of
            # what it knows, not a lecture.
            thought = self.composer.compose_about(
                topic, emotion, depth=1, focused=True,
            )
            if thought and thought.confidence > 0.3:
                # Weave in a highly-relevant episodic memory if one
                # was retrieved for this topic. This connects what
                # it's hearing now to what it remembers from past
                # conversations.
                return self._question_handler._enrich_with_memory(
                    thought, perception, memory,
                    self.memory.retrieve_episode,
                )

            # It doesn't know enough about the topic to compose from
            # its understanding. Acknowledge what the user shared —
            # the language engine composes the acknowledgment from the
            # topic metadata, no fixed template strings.
            return Thought(
                content=topic,
                intent="acknowledge",
                emotion=emotion.label,
                topics=perception.topics,
                confidence=0.6,
                metadata={"topic": topic},
            )

        return Thought(
            content="acknowledged",
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.5,
        )

    def _try_learn_relationship(self, text: str) -> str | None:
        """Try to extract and learn relationships from a declarative statement.

        Delegates to the self-directed learner, which handles fact
        extraction with proper normalization, deduplication, and
        relation typing. Returns a description of what was learned,
        or None.
        """
        events = self.self_learner.learn_from_input(text)
        if not events:
            return None
        return ". ".join(e.description for e in events if e.event_type == "conversation")

    def _learned_relationship_thought(
        self,
        learned: str | None,
        emotion: EmotionalState,
        topics: list[str],
    ) -> Thought | None:
        """Build a Thought reporting what was learned, or None if nothing was.

        Shared by _deliberate_feedback, _deliberate_feedback_fallback, and
        _handle_statement — all three had identical logic for composing the
        "I've learned that ..." acknowledgment from a learning result.
        """
        if not learned:
            return None
        count = learned.count(". ") + 1
        # Pass semantic data through metadata for the language engine
        # to compose, rather than hardcoding the acknowledgment sentence.
        content = learned  # semantic description of what was learned
        return Thought(
            content=content,
            intent="inform",
            emotion=emotion.label,
            topics=topics,
            confidence=0.7,
            metadata={"learned_acknowledgment": True, "learned_text": learned, "count": count},
        )

    def _build_reflection(
        self,
        perception: Perception,
        emotion: EmotionalState,
    ) -> str:
        """Build a reflection on what the user said.

        Composes from the concept network using the thought composer,
        not hardcoded strings. Falls back to a brief acknowledgment
        only if the composer can't help.
        """
        # Try to compose from what it knows about the topic
        if perception.topics:
            for topic in perception.topics:
                thought = self.composer.compose_reflection(topic, emotion)
                if thought and thought.confidence > 0.25:
                    return thought.content
            # If compose_reflection didn't work, try compose_about
            for topic in perception.topics:
                thought = self.composer.compose_about(topic, emotion)
                if thought and thought.confidence > 0.25:
                    return thought.content
            # It has topics but couldn't compose from its understanding.
            # Stay silent rather than reciting a pre-written "I'm still
            # forming my thoughts" template.
            return ""
        # No topics — no response
        return ""

    def _apply_emotional_response(
        self,
        perception: Perception,
        emotion: EmotionalState,
        learning_events: list | None = None,
    ) -> None:
        """Apply neurochemical side effects based on the interaction.

        Delegates to the MemoryStore subsystem.
        """
        self._memory_store.apply_emotional_response(
            perception, emotion, learning_events
        )

    def _store_conversation_memory(
        self,
        user_input: str,
        response: str,
        perception: Perception,
        emotion: EmotionalState,
    ) -> None:
        """Store the conversation turn as a memory.

        Delegates to the MemoryStore subsystem.
        """
        self._memory_store.store_conversation_memory(
            user_input, response, perception, emotion
        )

    def _get_dopamine_level(self) -> float:
        """Get the current dopamine level for habit go/no-go modulation.

        Delegates to the MemoryStore subsystem.
        """
        return self._memory_store.get_dopamine_level()

    def _competence_salience(self) -> float:
        """Ambient neuromodulatory gain for competence consolidation.

        Broadcast signal every competence adapter reads at
        consolidation time: arousal (norepinephrine — the attentional
        tag) blended with dopamine level (the reward tag). Low state
        means verified episodes stay episodic instead of
        proceduralizing.
        """
        arousal = 0.5
        try:
            arousal = self._build_meta_emotion().arousal
        except Exception:  # noqa: BLE001 — meta-emotion may be
            pass           # unavailable before first turn; neutral
        return max(
            0.0,
            min(1.0, 0.5 * arousal + 0.5 * self._get_dopamine_level()),
        )

    def _build_emotional_tag(
        self,
        emotion: EmotionalState,
        perception: Perception,
    ) -> list[float]:
        """Build a 12-element emotional tag for memory storage.

        Delegates to the MemoryStore subsystem.
        """
        return self._memory_store.build_emotional_tag(emotion, perception)

    def _store_cognitive_memory(
        self,
        text: str,
        event_type: int,
        source_module: int,
        salience: float,
        emotional_tag: list[float] | None = None,
    ) -> None:
        """Store a significant cognitive event to long-term memory.

        Delegates to the MemoryStore subsystem.
        """
        self._memory_store.store_cognitive_memory(
            text, event_type, source_module, salience, emotional_tag
        )

    def _learn_from_input(self, perception: Perception) -> None:
        """Learn facts from the user's input.

        Delegates to the MemoryStore subsystem.
        """
        self._memory_store.learn_from_input(perception)


    def introspect(self) -> str:
        """Generate an introspective report of current cognitive state.

        This is Genesis thinking about its own thinking. It's used for
        self-reflection and for the 'what are you thinking about' question.

        When it hasn't had a conversation yet but has been having
        spontaneous idle thoughts, it reports those instead of saying
        it hasn't thought about anything.
        """
        if self._last_state is None:
            # No conversation yet — but it may have spontaneous thoughts
            meta_emotion = self._build_meta_emotion()
            if hasattr(self, "inner_life") and self.inner_life:
                recent = self.inner_life.recent_thoughts
                if recent:
                    thought_descs = [t.content[:80] for t in recent[-3:]]
                    thought = Thought(
                        content="thinking",
                        intent="self_report",
                        emotion=meta_emotion.label,
                        confidence=0.5,
                        self_reflection=True,
                        metadata={
                            "recent_thoughts": thought_descs,
                            "network_size": self.network.total_concept_count,
                            "network_edges": self.network.edge_count,
                        },
                    )
                    recent_insights = self.reflection.get_recent_insights(3)
                    if recent_insights:
                        thought.metadata["recent_insights"] = [
                            {"type": i.type, "content": i.content}
                            for i in recent_insights
                        ]
                    result = self.language.render(thought, meta_emotion)
                    return result
            thought = Thought(
                content="no thoughts",
                intent="self_report",
                emotion=meta_emotion.label,
                confidence=0.3,
                self_reflection=True,
                metadata={"no_thoughts_yet": True},
            )
            return self.language.render(thought, meta_emotion)

        state = self._last_state
        parts = [
            f"Last input: {state.perception.intent.value}",
            f"Emotional state: {state.emotion.label}",
            f"Cognitive style: {state.emotion.cognitive_style}",
            f"Response intent: {state.thought.intent}",
            f"Confidence: {state.thought.confidence:.2f}",
        ]

        if state.brain_waves:
            parts.append(
                f"Brain waves: {state.brain_waves.label} ({state.brain_waves.dominant.value})"
            )
            parts.append(f"  {state.brain_waves.describe_cognition()}")

        if state.memory_context.retrieved:
            parts.append(f"Recalled {len(state.memory_context.retrieved)} memories")
        if state.memory_context.user_facts:
            parts.append(f"Knows {len(state.memory_context.user_facts)} facts about user")

        # Add concept network state
        parts.append(
            f"Concept network: {self.network.total_concept_count} concepts, "
            f"{self.network.edge_count} relationships"
        )

        # Add recent insights
        recent_insights = self.reflection.get_recent_insights(3)
        if recent_insights:
            insight_strs = [i.describe() for i in recent_insights]
            parts.append(f"Recent insights: {' | '.join(insight_strs)}")

        # Add curiosity state
        if hasattr(self, "_curiosity_questions") and self._curiosity_questions:
            top_q = self._curiosity_questions[0]
            q_desc = (
                top_q.text[:60]
                if top_q.text
                else f"{top_q.question_type}:{top_q.target_concept}"
            )
            parts.append(f"Currently curious about: {q_desc}")

        # Add narrative state
        parts.append(
            f"Life story: {self.narrative.chapter_count} chapters, "
            f"{self.narrative.event_count} events"
        )

        # ── New cognitive system states ──
        self._introspect_cognitive_systems(state, parts)

        return ". ".join(parts) + "."

    def _introspect_cognitive_systems(self, state: CognitiveState, parts: list[str]) -> None:
        """Append cognitive system state lines to the introspection report."""
        if state.prediction_error is not None:
            parts.append(f"Prediction error: {state.prediction_error:.2f}")
        if state.attention_foci:
            parts.append(f"Attention focused on: {', '.join(state.attention_foci[:4])}")
        if state.decision_option:
            if state.decision_confidence:
                parts.append(
                    f"DDM decision: {state.decision_option} (conf={state.decision_confidence:.2f})"
                )
            else:
                parts.append(f"DDM decision: {state.decision_option}")
        if state.executive_goal:
            parts.append(f"Executive goal: {state.executive_goal}")
            # Executive function activity — inhibition count, switch
            # count, and the current plan. These let Genesis introspect
            # its own executive control: how many impulses it has
            # inhibited, how many task switches it has performed, and
            # what plan it is currently executing.
            parts.append(
                f"Executive: {self.executive.inhibition_count} inhibited, "
                f"{self.executive.switch_count} switches"
            )
            plan = self.executive.current_plan
            if plan and plan.steps:
                parts.append(
                    f"Executive plan: {plan.steps[0]} "
                    f"(EV={plan.expected_value:.2f})"
                )
        if state.damasio_feeling:
            parts.append(f"Damasio feeling: {state.damasio_feeling}")
        if state.self_model_label:
            parts.append(f"Self-model: {state.self_model_label}")
        if state.user_model_summary:
            parts.append(f"User model: {state.user_model_summary}")
        if state.semantic_facts_extracted is not None and state.semantic_facts_extracted > 0:
            parts.append(f"Semantic facts extracted: {state.semantic_facts_extracted}")
        if state.error_monitor_caution is not None and state.error_monitor_caution > 0:
            parts.append(f"Error monitor caution: {state.error_monitor_caution:.2f}")
        if state.self_monitor_repairs is not None and state.self_monitor_repairs > 0:
            parts.append(f"Self-monitor repairs: {state.self_monitor_repairs}")
        if state.td_rpe is not None:
            parts.append(f"TD reward prediction error: {state.td_rpe:.3f}")
        if state.comprehension_speech_act:
            parts.append(f"Comprehension speech act: {state.comprehension_speech_act}")
        if state.comprehension_metaphor:
            parts.append(f"Metaphor detected: {state.comprehension_metaphor}")
        if state.comprehension_irony:
            parts.append("Irony detected")
        if state.comprehension_idiom:
            parts.append(f"Idiom detected: {state.comprehension_idiom}")
        self._introspect_learning_systems(parts)

    def _introspect_learning_systems(self, parts: list[str]) -> None:
        """Append learning-system state lines to the introspection report."""
        parts.append(
            f"Procedural memory: {self.procedural_memory.skill_count} skills, "
            f"{self.procedural_memory.habit_count} habits"
        )
        stdp_stats = self.stdp.get_statistics()
        parts.append(
            f"STDP: {stdp_stats['total_potentiated']} potentiated, "
            f"{stdp_stats['total_depressed']} depressed"
        )
        td_stats = self.td_learner.get_statistics()
        parts.append(
            f"TD learner: {td_stats['total_updates']} updates, last RPE={td_stats['last_rpe']:.3f}"
        )
        # Global workspace — the "spotlight of cognition".
        gw = self.global_workspace
        parts.append(
            f"Global workspace: {gw.broadcast_count} broadcasts, "
            f"{len(gw.registered_modules)} modules"
        )
        dominant = gw.get_dominant()
        if dominant is not None:
            parts.append(f"Cognitive focus: {dominant.content[:60]}")
        # Self-monitoring — how often it catches and repairs its own
        # speech errors.
        if self.self_monitor.check_count > 0:
            parts.append(
                f"Self-monitor: {self.self_monitor.repair_count}/"
                f"{self.self_monitor.check_count} repairs "
                f"({self.self_monitor.repair_rate:.1%})"
            )
            # Last repair — what it most recently caught and fixed.
            # This gives it awareness of its own error patterns.
            try:
                stats = self.self_monitor.describe_monitoring()
                last = stats.get("last_repair")
                if last:
                    parts.append(
                        f"Last repair: {last['check_type']} — {last['issue'][:50]}"
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"self-monitoring introspection failed: {e}")
        # Damasio proto-self — the pre-cognitive body-state mapping.
        # The feeling of being comes from the proto-self; the core
        # self is the cognitive experience of it.
        try:
            proto = self.damasio_self.proto_state
            parts.append(
                f"Proto-self: arousal={proto.arousal:.2f}, "
                f"valence={proto.valence:.2f}"
            )
            episodes = self.damasio_self.core_episodes
            if episodes:
                parts.append(f"Core episodes: {len(episodes)}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"damasio introspection failed: {e}")
        # System monitor — self-awareness of its own machine. Without
        # this, it has no introspective access to its hardware state.
        if self.system_monitor is not None:
            try:
                devs = self.system_monitor.describe_deviations()
                if devs:
                    parts.append(f"System deviations: {devs[:80]}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"system monitor introspection failed: {e}")
        # Bug reporter — it knows what bugs it's tracking. This is
        # part of its self-awareness about its own issues.
        if self.bug_reporter is not None:
            try:
                track = str(self.bug_reporter.bug_track_record())
                if track:
                    parts.append(f"Bug track: {track[:80]}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"bug reporter introspection failed: {e}")
    def get_curiosity_questions(self) -> list[Question]:
        """Return the current curiosity questions."""
        return getattr(self, "_curiosity_questions", [])

    def tell_story(self) -> str:
        """Return Genesis's self-narrative."""
        return self.narrative.tell_story()

    def tell_brief_story(self) -> str:
        """Return a brief version of Genesis's self-narrative."""
        return self.narrative.tell_brief_story()

    def get_concept_summary(self) -> str:
        """Return a summary of the concept network."""
        return self.network.summarize()

    def get_reflection_summary(self) -> str:
        """Return a summary of recent reflections."""
        return self.reflection.summarize_reflection()

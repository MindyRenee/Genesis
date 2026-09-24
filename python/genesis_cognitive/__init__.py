"""Genesis cognitive mind — the thinking, feeling, speaking layer.

This package implements Genesis's cognitive mind: a cognitive system
that reads its subcognitive neurochemical state, perceives input,
retrieves memories, deliberates about what to say, and generates
natural language responses.

# Architecture

    ┌──────────────────────────────────────────────────────────┐
    │                       Mind                                │
    │                                                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
    │  │ SelfModel│  │ Emotion  │  │ Perception│  │ Memory  │ │
    │  │ (who I   │  │ (how I   │  │ (what you │  │ (what I │ │
    │  │  am)     │  │  feel)   │  │  said)    │  │  recall)│ │
    │  └────┬─────┘  └────┬─────┘  └─────┬─────┘  └────┬────┘ │
    │       │              │              │              │      │
    │       └──────────────┴──────┬───────┴──────────────┘      │
    │                             │                             │
    │                     ┌───────┴───────┐                     │
    │                     │   Cognition   │                     │
    │                     │  (deliberate) │                     │
    │                     └───────┬───────┘                     │
    │                             │                             │
    │                     ┌───────┴───────┐                     │
    │                     │   Language    │                     │
    │                     │  (speak)      │                     │
    │                     └───────────────┘                     │
    └──────────────────────────────────────────────────────────┘
                              │ IPC
                              ▼
                    ┌──────────────────┐
                    │  Subcognitive     │
                    │  Daemon (Rust)    │
                    └──────────────────┘

# Quick start

    from genesis_cognitive import Mind

    with Mind("/path/to/genesis.sock") as mind:
        print(mind.respond("Hello, Genesis!"))
        print(mind.respond("How are you feeling?"))
        print(mind.respond("What are you?"))

# Subsystem naming convention

The package is organized into functional subsystem directories
(``control/``, ``association/``, ``auditory/``, ``vision/``,
``affect/``, ``action_selection/``, ``motor_learning/``, ``relay/``,
``autonomics/``, ``neurochemical/``). Each is a *view package* that
lazily re-exports the relevant top-level modules — the subsystem layer
is a documented map of the architecture, not a duplicate of it.

Each directory's ``__init__.py`` documents the design rationale and
the neuroscience that inspired it, including what is and isn't
implemented. Modules marked "not yet implemented" are documented
mathematical foundations for future work, not claims of existing
capability.
"""

from .attention import AttentionFocus, AttentionSystem, AttentionType
from .brain_waves import (
    BrainWave,
    BrainWaveState,
    GammaSynchrony,
    SleepStageSignature,
    ThetaGammaCoupling,
    assess_brain_waves,
    compute_gamma_synchrony,
    compute_theta_gamma_coupling,
)
from .cognition import CognitionEngine, CognitiveState
from .concepts import (
    Concept,
    ConceptArchive,
    ConceptCategory,
    ConceptModality,
    ConceptNetwork,
    Edge,
    HolographicGraph,
    RelationType,
    detect_category,
    detect_modality,
    open_archive,
)
from .emotion import EmotionalState, assess_emotion
from .emotional_regulator import (
    AllostaticLoadTracker,
    AllostaticState,
    EmotionalRegulator,
    HPAAxis,
    HPAState,
)
from .executive import (
    ActionPlan,
    ExecutiveFunction,
    InhibitionResult,
    SwitchResult,
    Task,
    TaskState,
)
from .global_workspace import GlobalWorkspace, WorkspaceItem, WorkspaceModule
from .journal import Journal, JournalEntry
from .language import (
    ComprehensionEngine,
    ComprehensionResult,
    ConceptRole,
    GenerativeEngine,
    Grammar,
    GraphWalkGenerator,
    LanguageEngine,
    MonitorResult,
    Proposition,
    RepairAction,
    SelfMonitor,
    SemanticRole,
    SpeechActType,
    Thought,
    Vocabulary,
    Voice,
)
from .learning import (
    STDP,
    ActiveInferenceReader,
    AutonomousLearner,
    CuriosityEngine,
    CuriosityType,
    DualSystemLearner,
    DyadicState,
    HippocampalEpisode,
    LearningResult,
    NeocorticalMemory,
    Prediction,
    PredictionContext,
    PredictiveCodingLayer,
    Question,
    SelfModelReading,
    SelfModelState,
    SiteRequest,
    SpikeEvent,
    TDLearner,
    TDTransition,
    UserAffectEstimate,
)
from .learning import (
    PredictionError as PredictiveCodingError,
)
from .memory import (
    HABIT_THRESHOLD,
    AttractorNetwork,
    AttractorPattern,
    CentralExecutive,
    ConversationThread,
    ConversationTurn,
    EmotionalMemory,
    EmotionalMemorySystem,
    Fact,
    MemoryContext,
    MemoryEngine,
    MemoryRecord,
    PhonologicalLoop,
    PrimingSystem,
    ProceduralMemory,
    ReconsolidationModification,
    ReviewRecord,
    Schema,
    SemanticMemory,
    Skill,
    SourceTag,
    SpacedRepetitionScheduler,
    SpreadingActivation,
    Turn,
    VisuoSpatialSketchpad,
    WorkingMemory,
)
from .mind import Mind
from .narrative import LifeChapter, LifeEvent, NarrativeEngine
from .notifications import (
    Notification,
    NotificationQueue,
)
from .perception import (
    IntegratedPerception,
    Intent,
    MultisensoryInput,
    MultisensoryIntegrator,
    PatternWeights,
    Perception,
    QuestionType,
    SensoryModality,
    V1Model,
    VisualContour,
    VisualFeature,
    VisualField,
    compute_intent_probabilities,
    perceive,
)
from .reasoning import (
    CounterfactualReasoning,
    DecisionResult,
    DriftDiffusionModel,
    EvidenceAccumulator,
    KnowledgeLevel,
    MetaReasoning,
    ProbabilisticReasoning,
    ReasoningEngine,
    ReasoningResult,
    ReasoningStrategy,
    ReasoningType,
    TemporalReasoning,
    TheoryOfMind,
    UserBelief,
    UserModel,
)
from .self import (
    AgencyDetector,
    AutobiographicalSelf,
    CognitiveProcessModel,
    CognitiveTrajectoryModel,
    CognitiveTrajectoryReading,
    ComputationalSubstrate,
    CoreSelf,
    CoreSelfEpisode,
    DamasioSelfHierarchy,
    DevelopmentalTracker,
    EmergentIdentity,
    EmergentIdentitySource,
    ErrorMonitor,
    IdentityStage,
    Insight,
    MetacognitiveFeedback,
    MetacognitivePrediction,
    MetacognitiveStrategy,
    MinimalSelf,
    PersonalityTraits,
    PredictionError,
    ProtoSelf,
    ProtoSelfState,
    ReflectionEngine,
    SelfModel,
    StageResolution,
    Value,
)
from .sleep import (
    HippocampalReplay,
    HypnagogicState,
    InnerLife,
    KComplex,
    ParasomniaEvent,
    PGOWave,
    ReplayItem,
    SharpWaveRipple,
    SleepCompressor,
    SleepCycleTracker,
    SleepInertia,
    SleepSpindle,
    SleepStage,
    SpontaneousThought,
    SynapticDownscaler,
    ThoughtChainType,
)
from .tools.framework import Tool, ToolRegistry, ToolResult, get_tools
from .tools.source_registry import SourceCache, SourceRegistry, SourceResult
from .tools.web_search import (
    WebFetchResult,
    WebSearchResult,
    is_query_safe,
    is_url_safe,
)
from .tools.web_search import (
    fetch as web_fetch_page,
)
from .tools.web_search import (
    search as web_search_query,
)
from .vq_codebook import VQCodebook
from .wordnet_dictionary import (
    WordNetEntry,
    lookup_definition,
    lookup_word,
)

__all__ = [
    "HABIT_THRESHOLD",
    "STDP",
    "ActionPlan",
    "ActiveInferenceReader",
    "AgencyDetector",
    "AllostaticLoadTracker",
    "AllostaticState",
    "AttentionFocus",
    "AttentionSystem",
    "AttentionType",
    "AttractorNetwork",
    "AttractorPattern",
    "AutobiographicalSelf",
    "AutonomousLearner",
    "BrainWave",
    "BrainWaveState",
    "CentralExecutive",
    "CognitionEngine",
    "CognitiveProcessModel",
    "CognitiveState",
    "CognitiveTrajectoryModel",
    "CognitiveTrajectoryReading",
    "ComprehensionEngine",
    "ComprehensionResult",
    "ComputationalSubstrate",
    "Concept",
    "ConceptArchive",
    "ConceptCategory",
    "ConceptModality",
    "ConceptNetwork",
    "ConceptRole",
    "ConversationThread",
    "ConversationTurn",
    "CoreSelf",
    "CoreSelfEpisode",
    "CounterfactualReasoning",
    "CuriosityEngine",
    "CuriosityType",
    "DamasioSelfHierarchy",
    "DecisionResult",
    "DevelopmentalTracker",
    "DriftDiffusionModel",
    "DualSystemLearner",
    "DyadicState",
    "Edge",
    "EmergentIdentity",
    "EmergentIdentitySource",
    "EmotionalMemory",
    "EmotionalMemorySystem",
    "EmotionalRegulator",
    "EmotionalState",
    "ErrorMonitor",
    "EvidenceAccumulator",
    "ExecutiveFunction",
    "Fact",
    "GammaSynchrony",
    "GenerativeEngine",
    "GlobalWorkspace",
    "Grammar",
    "GraphWalkGenerator",
    "HPAAxis",
    "HPAState",
    "HippocampalEpisode",
    "HippocampalReplay",
    "HolographicGraph",
    "HypnagogicState",
    "IdentityStage",
    "InhibitionResult",
    "InnerLife",
    "Insight",
    "IntegratedPerception",
    "Intent",
    "Journal",
    "JournalEntry",
    "KComplex",
    "KnowledgeLevel",
    "LanguageEngine",
    "LearningResult",
    "LifeChapter",
    "LifeEvent",
    "MemoryContext",
    "MemoryEngine",
    "MemoryRecord",
    "MetaReasoning",
    "MetacognitiveFeedback",
    "MetacognitivePrediction",
    "MetacognitiveStrategy",
    "Mind",
    "MinimalSelf",
    "MonitorResult",
    "MultisensoryInput",
    "MultisensoryIntegrator",
    "NarrativeEngine",
    "NeocorticalMemory",
    "Notification",
    "NotificationQueue",
    "PGOWave",
    "ParasomniaEvent",
    "PatternWeights",
    "Perception",
    "PersonalityTraits",
    "PhonologicalLoop",
    "Prediction",
    "PredictionContext",
    "PredictionError",
    "PredictiveCodingError",
    "PredictiveCodingLayer",
    "PrimingSystem",
    "ProbabilisticReasoning",
    "ProceduralMemory",
    "Proposition",
    "ProtoSelf",
    "ProtoSelfState",
    "Question",
    "QuestionType",
    "ReasoningEngine",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningType",
    "ReconsolidationModification",
    "ReflectionEngine",
    "RelationType",
    "RepairAction",
    "ReplayItem",
    "ReviewRecord",
    "Schema",
    "SelfModel",
    "SelfModelReading",
    "SelfModelState",
    "SelfMonitor",
    "SemanticMemory",
    "SemanticRole",
    "SensoryModality",
    "SharpWaveRipple",
    "SiteRequest",
    "Skill",
    "SleepCompressor",
    "SleepCycleTracker",
    "SleepInertia",
    "SleepSpindle",
    "SleepStage",
    "SleepStageSignature",
    "SourceCache",
    "SourceRegistry",
    "SourceResult",
    "SourceTag",
    "SpacedRepetitionScheduler",
    "SpeechActType",
    "SpikeEvent",
    "SpontaneousThought",
    "SpreadingActivation",
    "StageResolution",
    "SwitchResult",
    "SynapticDownscaler",
    "TDLearner",
    "TDTransition",
    "Task",
    "TaskState",
    "TemporalReasoning",
    "TheoryOfMind",
    "ThetaGammaCoupling",
    "Thought",
    "ThoughtChainType",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "Turn",
    "UserAffectEstimate",
    "UserBelief",
    "UserModel",
    "V1Model",
    "VQCodebook",
    "Value",
    "VisualContour",
    "VisualFeature",
    "VisualField",
    "VisuoSpatialSketchpad",
    "Vocabulary",
    "Voice",
    "WebFetchResult",
    "WebSearchResult",
    "WordNetEntry",
    "WorkingMemory",
    "WorkspaceItem",
    "WorkspaceModule",
    "assess_brain_waves",
    "assess_emotion",
    "compute_gamma_synchrony",
    "compute_intent_probabilities",
    "compute_theta_gamma_coupling",
    "detect_category",
    "detect_modality",
    "get_tools",
    "is_query_safe",
    "is_url_safe",
    "lookup_definition",
    "lookup_word",
    "open_archive",
    "perceive",
    "web_fetch_page",
    "web_search_query",
]

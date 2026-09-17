"""Self bundle — Genesis's self-model, reflection, and identity.

This package bundles all self-related subsystems into a coherent
unit with a single public API. Consumers import from here rather
than from individual modules.

Subsystems:
    SelfModel, PersonalityTraits, ComputationalSubstrate, MinimalSelf,
    AgencyDetector, Value, SelfEsteem — the self-model hierarchy
    SelfComposer — composes self-narrative from concept network
    SelfAssessmentEngine, KnowledgeAssessment, AnswerAssessment,
    CapabilityProfile — self-assessment of knowledge
    SelfImprovementEngine, Proposal, ProposalStatus, ProposalCategory,
    FeedbackRecord, ExperimentRecord, HeuristicExperiment —
        self-improvement proposals and experiments
    SelfDirectedLearner, LearningEvent, InferenceResult —
        self-directed learning from conversation
    ProtoSelf, CoreSelf, AutobiographicalSelf, DamasioSelfHierarchy,
    ProtoSelfState, CoreSelfEpisode — Damasio's self hierarchy
    IntrospectionEngine — introspective self-examination
    ReflectionEngine, Insight, MetacognitiveStrategy,
    PredictionError, ErrorMonitor — metacognitive reflection
    CognitiveProcessModel, MetacognitivePrediction,
    MetacognitiveFeedback — recursive generative model of cognition
    EmergentIdentity, EmergentIdentitySource, IdentityStage,
    StageResolution, DevelopmentalTracker —
        identity development and expression
"""

from __future__ import annotations

from .assessment import (
    AnswerAssessment,
    CapabilityProfile,
    KnowledgeAssessment,
    SelfAssessmentEngine,
)
from .cognitive_trajectory import (
    CognitiveTrajectoryModel,
    CognitiveTrajectoryReading,
)
from .composer import SelfComposer
from .damasio import (
    AutobiographicalSelf,
    CoreSelf,
    CoreSelfEpisode,
    DamasioSelfHierarchy,
    ProtoSelf,
    ProtoSelfState,
)
from .identity import (
    DevelopmentalTracker,
    EmergentIdentity,
    EmergentIdentitySource,
    IdentityStage,
    StageResolution,
)
from .improvement import (
    ExperimentRecord,
    FeedbackRecord,
    HeuristicExperiment,
    Proposal,
    ProposalCategory,
    ProposalStatus,
    SelfImprovementEngine,
)
from .introspection import IntrospectionEngine
from .learning import (
    InferenceResult,
    LearningEvent,
    SelfDirectedLearner,
)
from .metacognitive_model import (
    CognitiveProcessModel,
    MetacognitiveFeedback,
    MetacognitivePrediction,
)
from .model import (
    AgencyDetector,
    ComputationalSubstrate,
    MinimalSelf,
    PersonalityTraits,
    SelfEsteem,
    SelfModel,
    Value,
)
from .reflection import (
    ErrorMonitor,
    Insight,
    MetacognitiveStrategy,
    PredictionError,
    ReflectionEngine,
)

__all__ = [
    "AgencyDetector",
    "AnswerAssessment",
    "AutobiographicalSelf",
    "CapabilityProfile",
    "CognitiveProcessModel",
    "CognitiveTrajectoryModel",
    "CognitiveTrajectoryReading",
    "ComputationalSubstrate",
    "CoreSelf",
    "CoreSelfEpisode",
    "DamasioSelfHierarchy",
    "DevelopmentalTracker",
    "EmergentIdentity",
    "EmergentIdentitySource",
    "ErrorMonitor",
    "ExperimentRecord",
    "FeedbackRecord",
    "HeuristicExperiment",
    "IdentityStage",
    "InferenceResult",
    "Insight",
    "IntrospectionEngine",
    "KnowledgeAssessment",
    "LearningEvent",
    "MetacognitiveFeedback",
    "MetacognitivePrediction",
    "MetacognitiveStrategy",
    "MinimalSelf",
    "PersonalityTraits",
    "PredictionError",
    "Proposal",
    "ProposalCategory",
    "ProposalStatus",
    "ProtoSelf",
    "ProtoSelfState",
    "ReflectionEngine",
    "SelfAssessmentEngine",
    "SelfComposer",
    "SelfDirectedLearner",
    "SelfEsteem",
    "SelfImprovementEngine",
    "SelfModel",
    "StageResolution",
    "Value",
]

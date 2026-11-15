"""Self — Genesis's self-model, reflection, and identity.

This package is the anatomical home of the self-related subsystems.
Each module below is imported directly by the consumers that need it
(``from self.self_model import SelfModel``); the package init stays
a docstring-only shell so ``import self`` never pulls the
self → perception → language cycle.

Subsystems:
    SelfModel, PersonalityTraits, ComputationalSubstrate, MinimalSelf,
    AgencyDetector, Value, SelfEsteem — the self-model hierarchy
    SelfComposer — composes self-narrative from concept network
    SelfAssessment, KnowledgeAssessment, AnswerAssessment,
    CapabilityProfile — self-assessment of knowledge
    SelfImprovement, Proposal, ProposalStatus, ProposalCategory,
    FeedbackRecord, ExperimentRecord, HeuristicExperiment —
        self-improvement proposals and experiments
    SelfDirectedLearner, LearningEvent, InferenceResult —
        self-directed learning from conversation
    ProtoSelf, CoreSelf, AutobiographicalSelf, DamasioSelfHierarchy,
    ProtoSelfState, CoreSelfEpisode — Damasio's self hierarchy
    Introspection — introspective self-examination
    Reflection, Insight, MetacognitiveStrategy,
    PredictionError, ErrorMonitor — metacognitive reflection
    CognitiveProcessModel, MetacognitivePrediction,
    MetacognitiveFeedback — recursive generative model of cognition
    EmergentIdentity, EmergentIdentitySource, IdentityStage,
    StageResolution, DevelopmentalTracker —
        identity development and expression
    ManPagesMixin — her man pages are part of her self-model: she
        learns them at boot, and the installed copies stay synced to
        the versioned sources
"""

from __future__ import annotations

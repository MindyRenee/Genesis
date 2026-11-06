"""Self bundle — Genesis's self-model, reflection, and identity.

This package bundles all self-related subsystems into a coherent
unit with a single public API. Consumers import from here rather
than from individual modules.

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
    ManPageReview, ManChange, ManReviewMixin — her man pages are part
        of her self-model: changes to them wait for her say
"""

from __future__ import annotations

from _views import view_getattr

# Lazy re-exports. This package sits on a three-way import cycle
# (``self -> perception -> frontal_lobe -> self``): ``self.reflection``
# imports ``perception``, which imports the generator, which
# does ``from self import SelfModel``. Eager submodule imports here
# mean whichever package is entered first leaves ``self`` partially
# initialised and that import raises. Resolving each name lazily
# loads only the submodule that provides it — ``SelfModel`` pulls
# ``self.self_model`` alone, which imports nothing cyclic — so the cycle
# is broken from every entry point.
_EXPORTS = {
    # self.self_model — the self-model hierarchy
    "AgencyDetector": "self.self_model",
    "ComputationalSubstrate": "self.self_model",
    "MinimalSelf": "self.self_model",
    "PersonalityTraits": "self.self_model",
    "SelfEsteem": "self.self_model",
    "SelfModel": "self.self_model",
    "Value": "self.self_model",
    # self.assessment — self-assessment of knowledge
    "AnswerAssessment": "self.assessment",
    "CapabilityProfile": "self.assessment",
    "KnowledgeAssessment": "self.assessment",
    "SelfAssessment": "self.assessment",
    # self.master_cognitive_topos — unified E/I/S/language state transport
    "EhresmannConnectionTransport": "self.master_cognitive_topos",
    "KuramotoRiemannianSynchronizer": "self.master_cognitive_topos",
    "MasterCognitiveToposEngine": "self.master_cognitive_topos",
    "MasterFreeEnergyOptimizer": "self.master_cognitive_topos",
    # self.latent_geometry — geometric latent-state model of self
    "AgencyOwnershipFiberBundle": "self.latent_geometry",
    "AutopoieticSymplecticFlow": "self.latent_geometry",
    "GrassmannianSubspaceModule": "self.latent_geometry",
    "NarrativeAgencyBaseManifold": "self.latent_geometry",
    "SelfConsistencyInformationGeometry": "self.latent_geometry",
    "SelfLatentSpaceArchitecture": "self.latent_geometry",
    # self.cognitive_trajectory
    "CognitiveTrajectoryModel": "self.cognitive_trajectory",
    "CognitiveTrajectoryReading": "self.cognitive_trajectory",
    # self.composer — self-narrative composition
    "SelfComposer": "self.composer",
    # Damasio hierarchy — the proto-self lives in the insula,
    "AutobiographicalSelf": "insula.damasio",
    "CoreSelf": "insula.damasio",
    "CoreSelfEpisode": "insula.damasio",
    "DamasioSelfHierarchy": "insula.damasio",
    "ProtoSelf": "insula.damasio",
    "ProtoSelfState": "insula.damasio",
    # self.identity — identity development
    "DevelopmentalTracker": "self.identity",
    "EmergentIdentity": "self.identity",
    "EmergentIdentitySource": "self.identity",
    "IdentityStage": "self.identity",
    "StageResolution": "self.identity",
    # self.improvement — proposals and experiments
    "ExperimentRecord": "self.improvement",
    "FeedbackRecord": "self.improvement",
    "HeuristicExperiment": "self.improvement",
    "Proposal": "self.improvement",
    "ProposalCategory": "self.improvement",
    "ProposalStatus": "self.improvement",
    "SelfImprovement": "self.improvement",
    # self.introspection
    "Introspection": "self.introspection",
    # self.learning — self-directed learning
    "InferenceResult": "self.learning",
    "LearningEvent": "self.learning",
    "SelfDirectedLearner": "self.learning",
    # self.man_review — her manual is hers to approve
    "ManChange": "self.man_review",
    "ManPageReview": "self.man_review",
    "ManReviewMixin": "self.man_review",
    # self.metacognitive_model
    "CognitiveProcessModel": "self.metacognitive_model",
    "MetacognitiveFeedback": "self.metacognitive_model",
    "MetacognitivePrediction": "self.metacognitive_model",
    # self.reflection — metacognitive reflection
    "ErrorMonitor": "self.reflection",
    "Insight": "self.reflection",
    "MetacognitiveStrategy": "self.reflection",
    "PredictionError": "self.reflection",
    "Reflection": "self.reflection",
    # self.somatic
    "SomaticSnapshot": "insula.somatic",
    "somatic_congruence": "insula.somatic",
    # self.proposals — metacognitive self-improvement proposals
    "ProposalsMixin": "self.proposals",
    # self.psychoanalysis — the unconscious that works while she rests
    "DreamInterpretation": "self.psychoanalysis",
    "Psychoanalysis": "self.psychoanalysis",
    "RepressedTrace": "self.psychoanalysis",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)

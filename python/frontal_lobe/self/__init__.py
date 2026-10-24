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
# ``self.model`` alone, which imports nothing cyclic — so the cycle
# is broken from every entry point.
_EXPORTS = {
    # self.model — the self-model hierarchy
    "AgencyDetector": "frontal_lobe.self.model",
    "ComputationalSubstrate": "frontal_lobe.self.model",
    "MinimalSelf": "frontal_lobe.self.model",
    "PersonalityTraits": "frontal_lobe.self.model",
    "SelfEsteem": "frontal_lobe.self.model",
    "SelfModel": "frontal_lobe.self.model",
    "Value": "frontal_lobe.self.model",
    # self.assessment — self-assessment of knowledge
    "AnswerAssessment": "frontal_lobe.self.assessment",
    "CapabilityProfile": "frontal_lobe.self.assessment",
    "KnowledgeAssessment": "frontal_lobe.self.assessment",
    "SelfAssessment": "frontal_lobe.self.assessment",
    # self.cognitive_trajectory
    "CognitiveTrajectoryModel": "frontal_lobe.self.cognitive_trajectory",
    "CognitiveTrajectoryReading": "frontal_lobe.self.cognitive_trajectory",
    # self.composer — self-narrative composition
    "SelfComposer": "frontal_lobe.self.composer",
    # self.damasio — Damasio's self hierarchy
    "AutobiographicalSelf": "insula.damasio",
    "CoreSelf": "insula.damasio",
    "CoreSelfEpisode": "insula.damasio",
    "DamasioSelfHierarchy": "insula.damasio",
    "ProtoSelf": "insula.damasio",
    "ProtoSelfState": "insula.damasio",
    # self.identity — identity development
    "DevelopmentalTracker": "frontal_lobe.self.identity",
    "EmergentIdentity": "frontal_lobe.self.identity",
    "EmergentIdentitySource": "frontal_lobe.self.identity",
    "IdentityStage": "frontal_lobe.self.identity",
    "StageResolution": "frontal_lobe.self.identity",
    # self.improvement — proposals and experiments
    "ExperimentRecord": "frontal_lobe.self.improvement",
    "FeedbackRecord": "frontal_lobe.self.improvement",
    "HeuristicExperiment": "frontal_lobe.self.improvement",
    "Proposal": "frontal_lobe.self.improvement",
    "ProposalCategory": "frontal_lobe.self.improvement",
    "ProposalStatus": "frontal_lobe.self.improvement",
    "SelfImprovement": "frontal_lobe.self.improvement",
    # self.introspection
    "Introspection": "frontal_lobe.self.introspection",
    # self.learning — self-directed learning
    "InferenceResult": "frontal_lobe.self.learning",
    "LearningEvent": "frontal_lobe.self.learning",
    "SelfDirectedLearner": "frontal_lobe.self.learning",
    # self.man_review — her manual is hers to approve
    "ManChange": "frontal_lobe.self.man_review",
    "ManPageReview": "frontal_lobe.self.man_review",
    "ManReviewMixin": "frontal_lobe.self.man_review",
    # self.metacognitive_model
    "CognitiveProcessModel": "frontal_lobe.self.metacognitive_model",
    "MetacognitiveFeedback": "frontal_lobe.self.metacognitive_model",
    "MetacognitivePrediction": "frontal_lobe.self.metacognitive_model",
    # self.reflection — metacognitive reflection
    "ErrorMonitor": "frontal_lobe.self.reflection",
    "Insight": "frontal_lobe.self.reflection",
    "MetacognitiveStrategy": "frontal_lobe.self.reflection",
    "PredictionError": "frontal_lobe.self.reflection",
    "Reflection": "frontal_lobe.self.reflection",
    # self.somatic
    "SomaticSnapshot": "insula.somatic",
    "somatic_congruence": "insula.somatic",
    # self.proposals — metacognitive self-improvement proposals
    "ProposalsMixin": "frontal_lobe.self.proposals",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)

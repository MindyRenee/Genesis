"""Reasoning bundle — reasoning, decision-making, and theory of mind.

This package bundles Genesis's reasoning subsystems into a coherent
unit. Consumers import from here rather than individual modules.

Subsystems:
    Reasoner, ReasoningType, ReasoningResult, ReasoningStrategy —
        core reasoning engine
    Analogizer, AnalogyPair, AnalogyInsight — cross-domain structure
        mapping (finds connections humans haven't seen)
    ProblemSolver, Problem, Solution, SolutionStep, GoalType —
        means-ends analysis problem-solving engine
    TaskCompetence, TaskSchema, TaskSignature, LearnedSkill —
        domain-general task recognition, transition learning, and skill reuse
    CriticalThinker, CriticalAssessment, AssessmentRecommendation —
        epistemic evaluation of claims (evidence quality, source
        credibility, disconfirmation search, fallacy detection)
    ProbabilisticReasoning, BetaDistribution — probabilistic reasoning
    TemporalReasoning, TemporalRelation, TimeInterval — temporal reasoning
    CounterfactualReasoning — counterfactual reasoning
    MetaReasoning — meta-level reasoning
    DriftDiffusionModel, DecisionResult, EvidenceAccumulator —
        drift diffusion decision-making
    TheoryOfMind, KnowledgeLevel, UserBelief, UserModel —
        theory of mind
"""

from __future__ import annotations

from .analogy import (
    Analogizer,
    AnalogyInsight,
    AnalogyPair,
)
from .belief_revision import (
    BeliefRevision,
    RevisionRecord,
)
from .competence import (
    EffectPrediction,
    GoalCondition,
    LearnedSkill,
    ProcedureStep,
    SkillMatch,
    TaskCompetence,
    TaskContext,
    TaskSchema,
    TaskSignature,
    TransitionCheck,
)
from .critical_thinking import (
    AssessmentRecommendation,
    CriticalAssessment,
    CriticalThinker,
    EvidenceItem,
)
from .decision import (
    ActionType,
    CandidateAction,
    DecisionMaker,
    DecisionOutcome,
    ResourceState,
)
from .drift_diffusion import (
    DecisionResult,
    DriftDiffusionModel,
    EvidenceAccumulator,
)
from .hypotheses import (
    HYPOTHESIS_CAP,
    Hypothesis,
    HypothesisLedger,
)
from .planning import (
    Plan,
    Planner,
    PlanStatus,
    PlanStep,
    PlanStepStatus,
)
from .problem_solving import (
    GoalType,
    Problem,
    ProblemSolver,
    ProblemStatus,
    Solution,
    SolutionStep,
)
from .reasoner import (
    BetaDistribution,
    CounterfactualReasoning,
    MetaReasoning,
    ProbabilisticReasoning,
    Reasoner,
    ReasoningRecord,
    ReasoningResult,
    ReasoningStrategy,
    ReasoningType,
    TemporalReasoning,
    TemporalRelation,
    TimeInterval,
)
from .theory_of_mind import (
    KnowledgeLevel,
    TheoryOfMind,
    UserBelief,
    UserModel,
)

__all__ = [
    "HYPOTHESIS_CAP",
    "ActionType",
    "Analogizer",
    "AnalogyInsight",
    "AnalogyPair",
    "AssessmentRecommendation",
    "BeliefRevision",
    "BetaDistribution",
    "CandidateAction",
    "CounterfactualReasoning",
    "CriticalAssessment",
    "CriticalThinker",
    "DecisionMaker",
    "DecisionOutcome",
    "DecisionResult",
    "DriftDiffusionModel",
    "EffectPrediction",
    "EvidenceAccumulator",
    "EvidenceItem",
    "GoalCondition",
    "GoalType",
    "Hypothesis",
    "HypothesisLedger",
    "KnowledgeLevel",
    "LearnedSkill",
    "MetaReasoning",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "Planner",
    "ProbabilisticReasoning",
    "Problem",
    "ProblemSolver",
    "ProblemStatus",
    "ProcedureStep",
    "Reasoner",
    "ReasoningRecord",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningType",
    "ResourceState",
    "RevisionRecord",
    "SkillMatch",
    "Solution",
    "SolutionStep",
    "TaskCompetence",
    "TaskContext",
    "TaskSchema",
    "TaskSignature",
    "TemporalReasoning",
    "TemporalRelation",
    "TheoryOfMind",
    "TimeInterval",
    "TransitionCheck",
    "UserBelief",
    "UserModel",
]

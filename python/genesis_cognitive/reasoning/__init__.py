"""Reasoning bundle — reasoning, decision-making, and theory of mind.

This package bundles Genesis's reasoning subsystems into a coherent
unit. Consumers import from here rather than individual modules.

Subsystems:
    ReasoningEngine, ReasoningType, ReasoningResult, ReasoningStrategy —
        core reasoning engine
    AnalogyEngine, AnalogyPair, AnalogyInsight — cross-domain structure
        mapping (finds connections humans haven't seen)
    ProblemSolver, Problem, Solution, SolutionStep, GoalType —
        means-ends analysis problem-solving engine
    CriticalThinkingEngine, CriticalAssessment, AssessmentRecommendation —
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
    AnalogyEngine,
    AnalogyInsight,
    AnalogyPair,
)
from .belief_revision import (
    BeliefRevisionEngine,
    RevisionRecord,
)
from .critical_thinking import (
    AssessmentRecommendation,
    CriticalAssessment,
    CriticalThinkingEngine,
    EvidenceItem,
)
from .decision import (
    ActionType,
    CandidateAction,
    DecisionEngine,
    DecisionOutcome,
)
from .drift_diffusion import (
    DecisionResult,
    DriftDiffusionModel,
    EvidenceAccumulator,
)
from .engine import (
    BetaDistribution,
    CounterfactualReasoning,
    MetaReasoning,
    ProbabilisticReasoning,
    ReasoningEngine,
    ReasoningRecord,
    ReasoningResult,
    ReasoningStrategy,
    ReasoningType,
    TemporalReasoning,
    TemporalRelation,
    TimeInterval,
)
from .planning import (
    Plan,
    PlanningEngine,
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
from .theory_of_mind import (
    KnowledgeLevel,
    TheoryOfMind,
    UserBelief,
    UserModel,
)

__all__ = [
    "ActionType",
    "AnalogyEngine",
    "AnalogyInsight",
    "AnalogyPair",
    "AssessmentRecommendation",
    "BeliefRevisionEngine",
    "BetaDistribution",
    "CandidateAction",
    "CounterfactualReasoning",
    "CriticalAssessment",
    "CriticalThinkingEngine",
    "DecisionEngine",
    "DecisionOutcome",
    "DecisionResult",
    "DriftDiffusionModel",
    "EvidenceAccumulator",
    "EvidenceItem",
    "GoalType",
    "KnowledgeLevel",
    "MetaReasoning",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "PlanningEngine",
    "ProbabilisticReasoning",
    "Problem",
    "ProblemSolver",
    "ProblemStatus",
    "ReasoningEngine",
    "ReasoningRecord",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningType",
    "RevisionRecord",
    "Solution",
    "SolutionStep",
    "TemporalReasoning",
    "TemporalRelation",
    "TheoryOfMind",
    "TimeInterval",
    "UserBelief",
    "UserModel",
]

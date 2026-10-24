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
from .blending import (
    Blend,
    align,
    blend,
)
from .blending import (
    Space as MentalSpace,
)
from .causation import (
    StructuralModel,
    causes,
    depends,
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
from .convention import (
    Agent as ConventionAgent,
)
from .convention import (
    ConventionReport,
    is_convention,
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
from .foundations import (
    Audit,
    Foundations,
    audit,
)
from .hypotheses import (
    HYPOTHESIS_CAP,
    Hypothesis,
    HypothesisLedger,
)
from .induction import (
    AbstractLevel,
    ExampleInductor,
    PairStore,
    RuleStatus,
    TransformRule,
)
from .judgment import (
    Judge,
    Judgment,
    Stance,
)
from .logic import (
    MAX_ATOMS,
    Atom,
    Categorical,
    Deduction,
    Formula,
    Quantifier,
    all_of,
    and_,
    any_of,
    atom,
    contract,
    entails,
    expand,
    explain,
    iff,
    implies,
    models,
    not_,
    or_,
    parse,
    ramsey,
    revise,
    satisfiable,
    syllogism,
    verify,
)
from .modal import (
    Individual as ModalIndividual,
)
from .modal import (
    World as PossibleWorld,
)
from .modal import (
    conceivable,
    contingent,
    could,
    counterpart_transitive,
    counterparts,
    humean,
    mosaic,
    necessarily,
    pluriverse,
    possibly,
    recombine,
    recombining,
    supervenes,
)
from .modal import (
    status as modal_status,
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
    "MAX_ATOMS",
    "AbstractLevel",
    "ActionType",
    "Analogizer",
    "AnalogyInsight",
    "AnalogyPair",
    "AssessmentRecommendation",
    "Atom",
    "Audit",
    "BeliefRevision",
    "BetaDistribution",
    "Blend",
    "CandidateAction",
    "Categorical",
    "ConventionAgent",
    "ConventionReport",
    "CounterfactualReasoning",
    "CriticalAssessment",
    "CriticalThinker",
    "DecisionMaker",
    "DecisionOutcome",
    "DecisionResult",
    "Deduction",
    "DriftDiffusionModel",
    "EffectPrediction",
    "EvidenceAccumulator",
    "EvidenceItem",
    "ExampleInductor",
    "Formula",
    "Foundations",
    "GoalCondition",
    "GoalType",
    "Hypothesis",
    "HypothesisLedger",
    "Judge",
    "Judgment",
    "KnowledgeLevel",
    "LearnedSkill",
    "MentalSpace",
    "MetaReasoning",
    "ModalIndividual",
    "PairStore",
    "Plan",
    "PlanStatus",
    "PlanStep",
    "PlanStepStatus",
    "Planner",
    "PossibleWorld",
    "ProbabilisticReasoning",
    "Problem",
    "ProblemSolver",
    "ProblemStatus",
    "ProcedureStep",
    "Quantifier",
    "Reasoner",
    "ReasoningRecord",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningType",
    "ResourceState",
    "RevisionRecord",
    "RuleStatus",
    "SkillMatch",
    "Solution",
    "SolutionStep",
    "Stance",
    "StructuralModel",
    "TaskCompetence",
    "TaskContext",
    "TaskSchema",
    "TaskSignature",
    "TemporalReasoning",
    "TemporalRelation",
    "TheoryOfMind",
    "TimeInterval",
    "TransformRule",
    "TransitionCheck",
    "UserBelief",
    "UserModel",
    "align",
    "all_of",
    "and_",
    "any_of",
    "atom",
    "audit",
    "blend",
    "causes",
    "conceivable",
    "contingent",
    "contract",
    "could",
    "counterpart_transitive",
    "counterparts",
    "depends",
    "entails",
    "expand",
    "explain",
    "humean",
    "iff",
    "implies",
    "is_convention",
    "modal_status",
    "models",
    "mosaic",
    "necessarily",
    "not_",
    "or_",
    "parse",
    "pluriverse",
    "possibly",
    "ramsey",
    "recombine",
    "recombining",
    "revise",
    "satisfiable",
    "supervenes",
    "syllogism",
    "verify",
]

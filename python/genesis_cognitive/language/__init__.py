"""Language engine package — pluggable text generation for Genesis.

The language engine is Genesis's speech system. It takes structured
Thoughts from the cognition engine and composes natural language from
its vocabulary, grammar, and voice — never from hardcoded templates.

Submodules:

- ``base`` — LanguageEngine interface, Thought dataclass, SyntacticStructure
- ``vocabulary`` — the flesh of Genesis's language (word selection, slots)
- ``grammar`` — the skeleton of Genesis's language (sentence structures)
- ``voice`` — what makes Genesis sound like HER (tone, hedging, calibration)
- ``generator`` — compositional text generation from Thoughts
- ``graph_walk`` — sentences that emerge from concept-network traversal
- ``comprehension`` — the receptive language system (speech acts, semantic roles)
- ``pragmatics`` — meaning in context (conversational inference)
- ``prosody`` — rhythm, pacing, emphasis, and intonation
- ``figurative`` — metaphor, irony, and idiom handling
- ``self_monitor`` — Levelt's editor (self-monitoring and repair)
- ``acquisition`` — how Genesis learns language from input
- ``statistical_learner`` — statistical language learning from user input
"""

from .acquisition import LanguageAcquisition, LearnedUnit
from .base import LanguageEngine, SyntacticStructure, Thought
from .comprehension import (
    ComprehensionEngine,
    ComprehensionResult,
    ConceptRole,
    Proposition,
    SemanticRole,
    SpeechActType,
)
from .figurative import FigurativeLanguageProcessor, IronyDetection, Metaphor
from .generator import GenerativeEngine
from .grammar import Grammar, Literal, SentenceStructure, SentenceType, Slot
from .graph_walk import GraphWalkGenerator
from .pragmatics import PragmaticAnalysis, PragmaticReasoner
from .prosody import ProsodyGenerator, ProsodyPattern
from .self_monitor import MonitorResult, RepairAction, SelfMonitor
from .statistical_learner import NGramModel, StatisticalLanguageLearner
from .vocabulary import Vocabulary
from .voice import Voice

__all__ = [
    "ComprehensionEngine",
    "ComprehensionResult",
    "ConceptRole",
    "FigurativeLanguageProcessor",
    "GenerativeEngine",
    "Grammar",
    "GraphWalkGenerator",
    "IronyDetection",
    "LanguageAcquisition",
    "LanguageEngine",
    "LearnedUnit",
    "Literal",
    "Metaphor",
    "MonitorResult",
    "NGramModel",
    "PragmaticAnalysis",
    "PragmaticReasoner",
    "Proposition",
    "ProsodyGenerator",
    "ProsodyPattern",
    "RepairAction",
    "SelfMonitor",
    "SemanticRole",
    "SentenceStructure",
    "SentenceType",
    "Slot",
    "SpeechActType",
    "StatisticalLanguageLearner",
    "SyntacticStructure",
    "Thought",
    "Vocabulary",
    "Voice",
]

"""Memory bundle — all memory systems with a single public API.

This package bundles Genesis's memory subsystems into a coherent
unit. Consumers import from here rather than individual modules.

Subsystems:
    MemoryEngine              — the main memory retrieval/context engine
    MemoryContext             — per-conversation memory context
    MemoryRecord, SourceTag   — memory record types
    ConversationTurn          — a turn in conversation history
    ReconsolidationModification — reconsolidation tracking
    EmotionalMemorySystem     — emotional tagging of memories
    AttractorNetwork          — attractor-based memory retrieval
    PrimingSystem             — priming/association activation
    SpreadingActivation       — spreading activation across the network
    SemanticMemory, Fact, Schema — semantic knowledge store
    ProceduralMemory, Skill   — skills and habits (basal ganglia-inspired)
    WorkingMemory             — Baddeley's working memory model
    SpacedRepetitionScheduler — spaced repetition for memory consolidation
"""

from __future__ import annotations

from .engine import (
    ConversationTurn,
    MemoryContext,
    MemoryEngine,
    MemoryRecord,
    ReconsolidationModification,
    SourceTag,
)
from .procedural import HABIT_THRESHOLD, HabitBias, ProceduralMemory, Skill, SkillStrategy
from .semantic import Fact, Schema, SemanticMemory
from .spaced_repetition import ReviewRecord, SpacedRepetitionScheduler
from .systems import (
    AttractorNetwork,
    AttractorPattern,
    EmotionalMemory,
    EmotionalMemorySystem,
    PrimingSystem,
    SpreadingActivation,
)
from .working import (
    CentralExecutive,
    ConversationThread,
    PhonologicalLoop,
    Turn,
    VisuoSpatialSketchpad,
    WorkingMemory,
)

__all__ = [
    "HABIT_THRESHOLD",
    "AttractorNetwork",
    "AttractorPattern",
    "CentralExecutive",
    "ConversationThread",
    "ConversationTurn",
    "EmotionalMemory",
    "EmotionalMemorySystem",
    "Fact",
    "HabitBias",
    "MemoryContext",
    "MemoryEngine",
    "MemoryRecord",
    "PhonologicalLoop",
    "PrimingSystem",
    "ProceduralMemory",
    "ReconsolidationModification",
    "ReviewRecord",
    "Schema",
    "SemanticMemory",
    "Skill",
    "SkillStrategy",
    "SourceTag",
    "SpacedRepetitionScheduler",
    "SpreadingActivation",
    "Turn",
    "VisuoSpatialSketchpad",
    "WorkingMemory",
]

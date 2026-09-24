"""Cognition package — the core thinking engine and its subsystems.

This package decomposes the cognition engine into focused subsystems,
each extracted from the original CognitionEngine monolith:

- ``engine`` — the CognitionEngine orchestrator and CognitiveState
- ``answer_composer`` — compose factual answers from concept-network knowledge
- ``code_tools`` — dispatch code tools and compose code discussions
- ``concept_learner`` — word labeling and activation spreading
- ``feeling_reporter`` — compose self-reports about emotional state and concerns
- ``memory_store`` — conversation and cognitive event persistence
- ``meta_cognitive_router`` — first-pass routing of input to specialist subsystems
- ``question_composer`` — compose questions from curiosity and knowledge gaps
- ``question_handler`` — route and answer questions from concept-network knowledge
- ``response_styler`` — metacognitive tone adjustment and learning acknowledgment
- ``self_inquiry`` — questions Genesis asks about itself
- ``thought_composer`` — compose novel thoughts from the concept network
- ``topic_resolver`` — topic extraction and concept variant matching

The public API (CognitionEngine, CognitiveState, Goal) is re-exported
here so existing imports (`from .cognition import CognitionEngine`)
continue to work unchanged. Subsystem classes are accessed via their
own modules (e.g., ``from .cognition.feeling_reporter import FeelingReporter``).
"""

from .engine import CognitionEngine, CognitiveState, Goal

__all__ = ["CognitionEngine", "CognitiveState", "Goal"]

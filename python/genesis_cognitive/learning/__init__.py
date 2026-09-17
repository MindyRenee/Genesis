"""Learning bundle — all learning systems with a single public API.

This package bundles Genesis's learning subsystems into a coherent
unit. Consumers import from here rather than individual modules.

Subsystems:
    AutonomousLearner — self-directed web learning
    CuriosityEngine, CuriosityType, Question — curiosity-driven learning
    DualSystemLearner, HippocampalEpisode, NeocorticalMemory —
        dual-system memory consolidation
    STDP, SpikeEvent — spike-timing-dependent plasticity
    TDLearner, TDTransition — temporal-difference learning
    ActiveInferenceReader, SelfModelReading, DyadicState, etc. —
        active inference self-model
    PredictiveCodingLayer, Prediction, PredictionError, PredictionContext —
        predictive coding
    HebbianPlasticity — Hebbian learning
"""

from __future__ import annotations

from .active_inference import (
    ActiveInferenceReader,
    DyadicState,
    SelfModelReading,
    SelfModelState,
    UserAffectEstimate,
)
from .autonomous import (
    AutonomousLearner,
    LearningResult,
    LearningStrategy,
    MetaLearner,
    SiteRequest,
    TransferResult,
    is_programming_topic,
)
from .curiosity import CuriosityEngine, CuriosityType, Question
from .dual import DualSystemLearner, HippocampalEpisode, NeocorticalMemory
from .plasticity import HebbianPlasticity
from .predictive import (
    Prediction,
    PredictionContext,
    PredictionError,
    PredictiveCodingLayer,
)
from .stdp import STDP, SpikeEvent
from .td import TDLearner, TDTransition

__all__ = [
    "STDP",
    "ActiveInferenceReader",
    "AutonomousLearner",
    "CuriosityEngine",
    "CuriosityType",
    "DualSystemLearner",
    "DyadicState",
    "HebbianPlasticity",
    "HippocampalEpisode",
    "LearningResult",
    "LearningStrategy",
    "MetaLearner",
    "NeocorticalMemory",
    "Prediction",
    "PredictionContext",
    "PredictionError",
    "PredictiveCodingLayer",
    "Question",
    "SelfModelReading",
    "SelfModelState",
    "SiteRequest",
    "SpikeEvent",
    "TDLearner",
    "TDTransition",
    "TransferResult",
    "UserAffectEstimate",
    "is_programming_topic",
]

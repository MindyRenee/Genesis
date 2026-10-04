"""Learning bundle — all learning systems with a single public API.

The learning systems are kept distinct by what they modify: TD learns
value, Hebbian learning adapts representations/associations, and STDP
changes the independent synaptic-efficacy substrate.
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
from .synapses import SynapticStore
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
    "SynapticStore",
    "TDLearner",
    "TDTransition",
    "TransferResult",
    "UserAffectEstimate",
    "is_programming_topic",
]

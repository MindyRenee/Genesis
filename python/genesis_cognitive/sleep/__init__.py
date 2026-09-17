"""Sleep bundle — the architecture, dreams, and inner life of sleep.

This package bundles all sleep-related subsystems into a single
coherent unit with a public API. Consumers (mind.py, tests) import
from here rather than from individual modules.

Public API:
    InnerLife              — the sleep/dream/inner-monologue coordinator
    SleepStage             — sleep stage enum (re-exported from brain_waves)
    SleepCycleTracker      — ultradian cycle progression
    SynapticDownscaler     — SHY synaptic homeostasis
    HippocampalReplay      — N3 memory replay
    ReplayItem             — replay queue item
    SpontaneousThought     — a single thought
    ThoughtChainType       — chain classification enum
    HypnagogicState        — wake→sleep transition
    SleepInertia           — post-wake grogginess
    SharpWaveRipple        — SWR neural event
    PGOWave                — PGO wave neural event
    SleepSpindle           — N2 spindle event
    KComplex               — N2 K-complex event
    ParasomniaEvent        — sleepwalking/sleeptalking event
    DreamSynthesisEngine   — dream→propose→validate→consolidate loop
    DreamProposal          — proposed dream edge
    DreamValidationResult  — validation outcome
    DreamInsight           — validated dream insight
    SleepCompressor        — sleep-time memory compaction
"""

from __future__ import annotations

from ..brain_waves import SleepStage
from .architecture import (
    REPLAY_COMPRESSION,
    HippocampalReplay,
    ReplayItem,
    SleepCycleTracker,
    SynapticDownscaler,
)
from .compression import SleepCompressor
from .dream_synthesis import (
    DreamInsight,
    DreamProposal,
    DreamSynthesisEngine,
    DreamValidationResult,
)
from .events import (
    HypnagogicState,
    KComplex,
    ParasomniaEvent,
    PGOWave,
    SharpWaveRipple,
    SleepInertia,
    SleepSpindle,
)
from .inner_life import InnerLife
from .thoughts import SpontaneousThought, ThoughtChainType

__all__ = [
    "REPLAY_COMPRESSION",
    "DreamInsight",
    "DreamProposal",
    "DreamSynthesisEngine",
    "DreamValidationResult",
    "HippocampalReplay",
    "HypnagogicState",
    "InnerLife",
    "KComplex",
    "PGOWave",
    "ParasomniaEvent",
    "ReplayItem",
    "SharpWaveRipple",
    "SleepCompressor",
    "SleepCycleTracker",
    "SleepInertia",
    "SleepSpindle",
    "SleepStage",
    "SpontaneousThought",
    "SynapticDownscaler",
    "ThoughtChainType",
]

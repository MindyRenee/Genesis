"""Basal-ganglia action selection, gating, and reinforcement learning.

The concrete selector is implemented in :mod:`basal_ganglia`. Related
learning and procedural-memory components remain shared subsystems.
"""

from __future__ import annotations

from .._views import view_getattr
from .basal_ganglia import ActionBid, BasalGangliaSelector, SelectionResult

_EXPORTS: dict[str, str] = {
    "TDLearner": "learning",
    "TDTransition": "learning",
    "HABIT_THRESHOLD": "memory",
    "HabitBias": "memory",
    "ProceduralMemory": "memory",
    "Skill": "memory",
    "SkillStrategy": "memory",
}

__all__ = [
    "HABIT_THRESHOLD",
    "ActionBid",
    "BasalGangliaSelector",
    "HabitBias",
    "ProceduralMemory",
    "SelectionResult",
    "Skill",
    "SkillStrategy",
    "TDLearner",
    "TDTransition",
]

__getattr__ = view_getattr(_EXPORTS, __name__)

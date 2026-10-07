"""Basal-ganglia action selection, gating, and reinforcement learning.

The concrete selector is implemented in :mod:`basal_ganglia`. Related
learning and procedural-memory components remain shared subsystems.
"""

from __future__ import annotations

from .._views import view_getattr
from .basal_ganglia import ActionBid, BasalGangliaSelector, SelectionResult

# The basal ganglia's functions are implemented by multi-subsystem modules
# that stay at the top level. This subsystem lazily re-exports the ones
# whose primary function is basal-ganglia circuitry — the striatum's
# dopamine-driven reinforcement learning and the procedural
# habit/skill store (the association pattern):
#
#   from genesis_conscious.basal_ganglia import TDLearner
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

"""Basal-ganglia action selection, gating, and reinforcement learning.

The concrete selector is implemented in :mod:`basal_ganglia`. Related
learning and procedural-memory components remain shared subsystems.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: THE ACTION–OUTCOME LOOP
════════════════════════════════════════════════════════════════════════

The striatum learns action→outcome associations: begin records an
action and the state Genesis predicts it will produce; observe
records the independently observed post-action state (never by
copying the intended outcome), computes a structured discrepancy,
updates the learned action model, and diagnoses likely causes. The
reward shaping that drives the value update:

    matched success        +1.0
    success w/ discrepancy  +1.0 − 0.4 · |discrepancy|
    failure                 −0.5 − 0.3 · |discrepancy|
    terminal (dead-end)     −1.0
    wait (no effect)        −0.4
    unscored observation    +0.5

    where:
        discrepancy = distance between predicted and observed
                      post-action state, [0, 1]

Successes whose outcome differed from the prediction are worth less
than exact matches (the discrepancy-weighted terms), so the action
model — not just the value — is what gets trained. Repeated failures
of the same action in the same context are diagnosed into a causal
hypothesis (model_repeated_failure) instead of being retried blind.

In Genesis: learning/action_feedback.py ActionFeedbackEngine
implements the loop; ActionPrediction, Discrepancy, ActionRevision,
and CausalHypothesis are its records.
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
    # The action→prediction→observation→revision loop — how the
    # striatum learns what an action actually produces
    "ActionFeedbackEngine": "learning.action_feedback",
    "ActionPrediction": "learning.action_feedback",
    "ActionRevision": "learning.action_feedback",
    "CausalHypothesis": "learning.action_feedback",
    "Discrepancy": "learning.action_feedback",
    "model_repeated_failure": "learning.action_feedback",
}

__all__ = [
    "HABIT_THRESHOLD",
    "ActionBid",
    "ActionFeedbackEngine",
    "ActionPrediction",
    "ActionRevision",
    "BasalGangliaSelector",
    "CausalHypothesis",
    "Discrepancy",
    "HabitBias",
    "ProceduralMemory",
    "SelectionResult",
    "Skill",
    "SkillStrategy",
    "TDLearner",
    "TDTransition",
    "model_repeated_failure",
]

__getattr__ = view_getattr(_EXPORTS, __name__)

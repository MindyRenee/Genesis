"""Basal ganglia — action selection, reinforcement learning, and motor control.

The basal ganglia are a group of subcortical nuclei critical for
action selection, habit formation, and reinforcement learning. They
receive input from the cortex and thalamus, process it through
direct/indirect pathways, and output back to the cortex via the thalamus.

Key structures:
    - Striatum (input nucleus):
      * Caudate nucleus — cognitive loops, goal-directed behavior
      * Putamen — motor loops, procedural learning
      * Nucleus accumbens (ventral striatum) — reward, motivation
    - Globus pallidus (internal GPi, external GPe) — output nuclei
    - Subthalamic nucleus (STN) — hyperdirect pathway
    - Substantia nigra:
      * Pars compacta (SNc) — dopamine source (VTA for ventral)
      * Pars reticulata (SNr) — output nucleus

The concrete selector is implemented in :mod:`basal_ganglia`. The
reinforcement learner (``td.py``) and the action–outcome loop
(``action_feedback.py``) live in this package; the procedural-memory
components remain shared subsystems claimed from ``memory``.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: DIRECT/INDIRECT PATHWAYS
════════════════════════════════════════════════════════════════════════

The basal ganglia implement action selection through parallel direct
and indirect pathways:

Direct pathway (facilitates movement):
    Cortex → Striatum (D1 MSNs) → GPi/SNr (inhibition) → Thalamus (disinhibition) → Cortex
    Dopamine (D1 receptors) enhances this pathway → Go signal

Indirect pathway (suppresses movement):
    Cortex → Striatum (D2 MSNs) → GPe (inhibition) → STN (disinhibition)
        → GPi/SNr (excitation) → Thalamus (inhibition) → Cortex
    Dopamine (D2 receptors) inhibits this pathway → No-Go signal

Hyperdirect pathway (rapid global suppression):
    Cortex → STN → GPi/SNr → Thalamus (inhibition) → Cortex

Dopamine from SNc/VTA modulates the balance:
    - High dopamine: direct pathway enhanced, indirect suppressed → movement facilitation
    - Low dopamine: direct suppressed, indirect enhanced → bradykinesia (Parkinson's)

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

In Genesis: basal_ganglia/action_feedback.py ActionFeedback
implements the loop; ActionPrediction, Discrepancy, ActionRevision,
and CausalHypothesis are its records.
"""

from __future__ import annotations

from _views import view_getattr

from .basal_ganglia import ActionBid, BasalGanglia, SelectionResult

# The basal ganglia's reinforcement-learning modules live in this
# package (td.py, action_feedback.py); the procedural habit/skill
# store is claimed from the shared memory subsystem — the striatal
# association pattern:
#
#   from basal_ganglia import TDLearner
_EXPORTS: dict[str, str] = {
    "TDLearner": "basal_ganglia.td",
    "TDTransition": "basal_ganglia.td",
    "HABIT_THRESHOLD": "memory.procedural",
    "HabitBias": "memory.procedural",
    "ProceduralMemory": "memory.procedural",
    "Skill": "memory.procedural",
    "SkillStrategy": "memory.procedural",
    # The action→prediction→observation→revision loop — how the
    # striatum learns what an action actually produces
    "ActionFeedback": "basal_ganglia.action_feedback",
    "ActionPrediction": "basal_ganglia.action_feedback",
    "ActionRevision": "basal_ganglia.action_feedback",
    "CausalHypothesis": "basal_ganglia.action_feedback",
    "Discrepancy": "basal_ganglia.action_feedback",
    "model_repeated_failure": "basal_ganglia.action_feedback",
    # Urge arbitration — the striatal selection loop over drives
    "VolitionMixin": "basal_ganglia.volition",
    # The urge accumulator the selection loop gates — a leaky integrator
    # with threshold and refractory period (co-claimed with frontal_lobe,
    # which holds the urge primitive)
    "Urge": "frontal_lobe.volition",
    "Volition": "frontal_lobe.volition",
    # GABA — the striatum's own inhibitory currency: direct/indirect
    # pathway gating lives here, not in a general pool
    "GABA": "basal_ganglia.gaba",
}

__all__ = [
    "GABA",
    "HABIT_THRESHOLD",
    "ActionBid",
    "ActionFeedback",
    "ActionPrediction",
    "ActionRevision",
    "BasalGanglia",
    "CausalHypothesis",
    "Discrepancy",
    "HabitBias",
    "ProceduralMemory",
    "SelectionResult",
    "Skill",
    "SkillStrategy",
    "TDLearner",
    "TDTransition",
    "Urge",
    "Volition",
    "VolitionMixin",
    "model_repeated_failure",
]

__getattr__ = view_getattr(_EXPORTS, __name__)

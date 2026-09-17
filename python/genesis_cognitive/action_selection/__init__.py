"""Basal ganglia — action selection, gating, and reinforcement learning.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The basal ganglia are a group of subcortical nuclei that implement
action selection and reinforcement learning. They receive input from
the entire cerebral cortex, process it through parallel loops, and
output back to the frontal subsystem via the relay. Their primary
function is to resolve competition between potential actions and
gate the winner through to execution.

Key functions:
    - Action selection (resolve competition between actions)
    - Action gating (gate working memory updates, motor initiation)
    - Reinforcement learning (dopamine-driven value learning)
    - Habit formation (automatic stimulus-response associations)
    - Reward prediction (value function estimation)
    - Motor program selection (which movement to execute)
    - Cognitive control (which task set to load)

The basal ganglia consist of:
    - Striatum (caudate, putamen, ventral striatum/nucleus accumbens)
    - Globus pallidus (internal GPi, external GPe)
    - Substantia nigra (pars compacta SNc, pars reticulata SNr)
    - Subthalamic nucleus (STN)

Pipeline:

    [Cortex] (all subsystems send input)
        |
        v
    [Striatum] (input stage)
        |   D1 neurons -> direct pathway (Go)
        |   D2 neurons -> indirect pathway (NoGo)
        v
    [GPi/SNr] (output stage)
        |   Tonic inhibition of relay
        |   Direct pathway disinhibits (facilitates action)
        |   Indirect pathway increases inhibition (suppresses action)
        v
    [Thalamus] -> [Frontal subsystem / Motor cortex]
        |   Gated action execution

    -- Dopamine modulation --
    [SNc/VTA] -> [Striatum]
        |   Dopamine signal = reward prediction error
        |   Positive RPE -> strengthens D1 (Go) pathway
        |   Negative RPE -> strengthens D2 (NoGo) pathway
        v
    [Striatum] weights updated

    -- Hyperdirect pathway --
    [Cortex] -> [STN] -> [GPi]
        |   Global "brake" — suppresses all actions
        |   Allows reconsideration before committing


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Action Selection (Gurney et al., 2001)
----------------------------------------

The basal ganglia implement a centralized action selection circuit.
Each action has a salience (cortical input strength). The direct and
indirect pathways compete to gate or suppress each action.

    salience_i = f(cortical_input_i)

    Direct pathway (Go — facilitates action i):
        D1_i = striatum_D1(salience_i)
        GPi_i = tonic_inhibition - D1_i

    Indirect pathway (NoGo — suppresses action i):
        D2_i = striatum_D2(salience_i)
        GPe_i = tonic_GPe - D2_i
        STN_global = sum_i (cortical_input_i) * w_STN
        GPi_i += STN_global - GPe_i

    Action i is selected if GPi_i is sufficiently low
    (disinhibition of relay).

    where:
        salience_i     = salience of action i
        D1_i           = direct pathway output for action i
        D2_i           = indirect pathway output for action i
        GPi_i          = internal globus pallidus output (tonic
                        inhibition minus direct pathway)
        STN_global     = subthalamic nucleus global signal
                        (excitatory, provides normalization)
        GPe_i          = external globus pallidus (modulated by D2)

The STN provides a global normalization signal — it excites all GPi
neurons, ensuring that the total inhibition across all actions is
balanced. This implements a "soft max" over actions.

In Genesis: not yet implemented as a dedicated module. The
executive.py task-switching and the procedural memory's Go/NoGo
habit formation approximate this function.


Bayesian Normalization (GPe-STN loop)
----------------------------------------

The STN-GPe network computes the normalization term for action
selection, ensuring probabilities sum to 1.

    P(action_i | context) = P(context | action_i) * P(action_i)
                            / sum_j P(context | action_j) * P(action_j)

This is Bayesian normalization — the posterior probability of each
action given the context, normalized across all actions. The STN
provides the denominator (the normalization term).

In Genesis: not yet implemented. The procedural memory's habit
threshold (0.8) serves a similar gating function — when a skill's
strength exceeds threshold, it becomes automatic (a habit).


Dopamine-Modulated Reinforcement Learning
-------------------------------------------

Dopamine neurons in the SNc/VTA signal reward prediction error (RPE).
This signal drives learning in the striatum, strengthening the
direct (Go) pathway for positive RPE and the indirect (NoGo) pathway
for negative RPE.

    delta = r + gamma * V(s') - V(s)    # TD error (RPE)

    # D1 (Go) pathway — positive RPE strengthens
    delta_w_D1_i proportional to +delta * e_i

    # D2 (NoGo) pathway — negative RPE strengthens
    delta_w_D2_i proportional to -delta * e_i

    where:
        delta  = reward prediction error (dopamine signal)
        r      = reward received
        gamma  = discount factor
        V(s)   = value of state s
        V(s')  = value of next state s'
        e_i    = eligibility trace for action i
        w_D1   = direct pathway weights (Go)
        w_D2   = indirect pathway weights (NoGo)

Positive RPE (reward better than expected) strengthens the Go
pathway — the action that led to the reward is more likely to be
selected next time. Negative RPE (reward worse than expected)
strengthens the NoGo pathway — the action is suppressed.

In Genesis: learning/td.py TDLearner implements the TD error and
eligibility trace computation. The dopamine signal
(get_dopamine_signal()) is used by the cognition engine for
neurochemical modulation. The procedural memory
(memory/procedural.py) implements the Go/NoGo habit formation with
a power-law practice function.


Habit Formation (power law of practice)
-----------------------------------------

Skills become habits through repetition. The basal ganglia takes
over from the goal-directed prefrontal cortex as actions become
automatic (Graybiel, 2008).

    RT = a * practice^(-b)    # power law of practice

    if skill_strength > threshold (0.8):
        skill becomes a habit (automatic, stimulus-driven)

    where:
        RT        = response time
        a         = initial response time
        practice  = number of practice trials
        b         = learning rate (~0.4)

In Genesis: memory/procedural.py implements this. Skills store
strategies (not words — respecting the no-hardcoding rule). When
a skill's strength exceeds 0.8, it becomes a habit and executes
without cognitive attention.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The basal ganglia are subcortical and don't produce cortical
oscillations directly, but they modulate cortical rhythms through
the relay.

Beta (13-30 Hz) — Action selection and habit
-----------------------------------------------

Basal ganglia beta is the rhythm of action selection and habit
maintenance. Beta increases during the maintenance of a current
action set and decreases when a new action is selected (the
"beta rebound"). This is the signature of the Go/NoGo decision
(Leventhal et al., 2012).

In Genesis: the basal ganglia's action selection would modulate
beta via the relay. The procedural memory's habit formation
and the executive function's task-switching contribute to
beta-band dynamics.

Gamma (30-100 Hz) — Active selection
---------------------------------------

Basal ganglia gamma increases during active action selection,
particularly for novel or uncertain actions. The gamma signal
reflects the competitive process of selecting among alternatives.

In Genesis: not yet tracked separately. Would be added when
the action selection model is implemented.

Theta (4-8 Hz) — Reward learning
----------------------------------

Basal ganglia theta is associated with reward learning and
motivation. Theta oscillations in the ventral striatum are
modulated by reward prediction and dopamine (Cohen et al., 2009).

In Genesis: the TD learner's reward prediction error signal would
contribute to theta-band dynamics when the action selection model
is implemented.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The basal ganglia are subcortical — they sit below the cerebral
cortex and interact with it via the relay. They are not part
of any cortical subsystem but interact with all of them:

    - Frontal subsystem: primary output target (motor cortex, PFC).
      The basal ganglia gate which actions and task sets reach
      the frontal subsystem.
    - Parietal subsystem: receives spatial context for action selection.
    - Temporal subsystem: receives recognized objects for
      stimulus-response mapping.
    - Cerebellum: complementary motor system. The motor_learning
      refines movements; the basal ganglia selects which movement
      to make.

No modules exist in this folder yet. The basal ganglia are
documented here as a mathematical foundation. The existing
reinforcement learning and habit formation modules are multi-subsystem
and stay at the top level.

Related top-level modules:
    (top-level) learning/td.py
        TDLearner — TD(lambda) reward prediction error with
        eligibility traces. The dopamine signal drives basal
        ganglia learning. This is the core basal ganglia
        computation, but it's used by the cognition engine
        (frontal subsystem) for value estimation, so it stays top-level.

    (top-level) memory/procedural.py
        ProceduralMemory — skills and habits with Go/NoGo
        pathways. The basal ganglia's direct (Go) and indirect
        (NoGo) pathways are implemented here. This is multi-subsystem
        because procedural memory also involves motor cortex
        (frontal subsystem) and is referenced by the cognition engine.

    (top-level) executive.py
        ExecutiveFunction — task-switching approximates the basal
        ganglia's action gating. The executive function is
        frontal (PFC) but its gating function is basal-ganglia-
        inspired.
"""

from __future__ import annotations

from .._views import view_getattr

# The basal ganglia's functions are implemented by multi-subsystem modules
# that stay at the top level. This subsystem lazily re-exports the ones
# whose primary function is basal-ganglia circuitry — the striatum's
# dopamine-driven reinforcement learning and the procedural
# habit/skill store (the association pattern):
#
#   from genesis_cognitive.action_selection import TDLearner
_EXPORTS: dict[str, str] = {
    # Striatum — dopamine reward-prediction-error learning.
    # TDLearner implements the TD error that dopaminergic VTA/SNc
    # neurons broadcast; eligibility traces bridge delayed reward.
    "TDLearner": "learning",
    "TDTransition": "learning",
    # Procedural memory — skills and habits. Go/NoGo habit formation
    # with the direct/indirect pathway dynamics documented above.
    "HABIT_THRESHOLD": "memory",
    "HabitBias": "memory",
    "ProceduralMemory": "memory",
    "Skill": "memory",
    "SkillStrategy": "memory",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)

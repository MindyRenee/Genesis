"""Cerebellum — motor learning, timing, and predictive forward models.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The motor_learning ("little brain") sits behind the cerebral cortex and
contains more than half of the brain's neurons. Despite its small
size, it is critical for smooth, coordinated, well-timed movement
and for predictive control of action.

Key functions:
    - Motor learning (error-driven adaptation of movements)
    - Timing and prediction (when things will happen)
    - Forward models (predicting sensory consequences of actions)
    - Sequence learning (automating action sequences)
    - Error correction (real-time motor adjustments)
    - Cognitive timing (not just motor — also involved in
      language, attention, and working memory timing)

The motor_learning is NOT involved in initiating movement or deciding
what to do — that's the basal ganglia and frontal subsystem. Instead,
it takes a motor command and produces a refined, error-corrected
version, predicting the sensory feedback that will result.

Pipeline:

    [Motor command] (from frontal subsystem / motor cortex)
        |
        v
    [Mossy fibers] -> [Granule cells] -> [Parallel fibers]
        |   Expand input into high-dimensional representation
        |   Granule cells: ~10^11 neurons (half the brain's neurons)
        v
    [Purkinje cells] (output)
        |   Adaptive filter: weighted sum of parallel fiber inputs
        |   Weights learned via error-driven plasticity (LTD)
        v
    [Deep cerebellar nuclei] -> [Thalamus] -> [Motor cortex]
        |   Refined motor command
        v
    [Action]

    -- Error correction loop --
    [Climbing fibers] (from inferior olive)
        |   Carry error signal (teaching signal)
        |   When actual differs from predicted, climbing fibers fire
        |   Purkinje cell weights update (LTD at active synapses)
        v
    [Purkinje cell] weights adjusted


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Adaptive Filter Model (Fujita, 1987; Dean et al.)
---------------------------------------------------

The motor_learning is modeled as an adaptive filter. Mossy fiber input
is expanded by granule cells into a high-dimensional representation,
then filtered by Purkinje cell weights to produce output.

    y(t) = sum_i  w_i(t) * g_i(m(t))

    where:
        m(t)  = mossy fiber input (motor command + context)
        g_i   = granule cell expansion (nonlinear basis functions)
        w_i   = parallel fiber -> Purkinje cell synaptic weight
        y(t)  = Purkinje cell output (refined prediction)

The granule cell expansion is critical — it maps a low-dimensional
input to a very high-dimensional representation, making linear
separation possible (the kernel trick, biologically implemented).

In Genesis: not yet implemented as a dedicated module. The
predictive coding system in vision (V1->V4->VTC->MTL)
implements a similar principle — each level predicts the level
below, and prediction errors drive learning. The motor_learning would
extend this to the motor domain when motor control is added.


Error-Driven Learning (climbing fiber -> LTD)
-----------------------------------------------

Purkinje cell weights are updated via error-driven plasticity.
The error signal comes from climbing fibers (inferior olive).

    delta_w_i = -eta * e(t) * g_i(t)

    where:
        e(t)    = error signal (from climbing fibers)
        g_i(t)  = granule cell activity (eligibility)
        eta    = learning rate
        w_i    = Purkinje cell weight

When the actual sensory outcome differs from the predicted outcome,
the inferior olive computes the error and sends it via climbing
fibers. The active parallel fiber synapses undergo LTD (long-term
depression), reducing their contribution to the erroneous output.

In Genesis: the TD learner (learning/td.py) implements a similar
principle — reward prediction errors drive learning via
eligibility traces. The cerebellar model differs in using sensory
prediction errors (not reward) and in the granule cell expansion.


Eligibility Traces (for delayed errors)
-----------------------------------------

When the error arrives (via climbing fibers), the synapses that
were recently active need to be credited. This is done via
eligibility traces — a decaying record of recent synaptic activity.

    e_i(t) = integral_0^infty  k(tau) * g_i(t - tau) dtau

    delta_w_i = -eta * e(t) * e_i(t)

    where:
        k(tau)  = eligibility kernel (e.g., exponential decay)
        e_i(t)  = eligibility trace for synapse i
        e(t)    = error signal

The eligibility trace records which synapses were recently active
so that delayed error signals can credit the right synapses. This
is the same principle as TD(lambda) eligibility traces (Sutton &
Barto, 2018), but applied to motor prediction rather than reward
prediction.

In Genesis: learning/td.py implements eligibility traces for
reward prediction. The cerebellar version would use the same
mathematical framework but with sensory prediction errors.


Forward Model Prediction
--------------------------

The motor_learning predicts the sensory consequences of motor commands.
This prediction is compared to actual feedback, and the mismatch
drives motor corrections.

    y_predicted(t) = f_w(m(t))           # learned forward model
    e(t) = y_actual(t) - y_predicted(t) # prediction error

    where:
        m(t)        = motor command
        f_w         = learned forward model (adaptive filter)
        y_predicted = predicted sensory outcome
        y_actual    = actual sensory feedback
        e(t)        = prediction error (drives learning)

The forward model allows the system to:
    1. Correct errors before sensory feedback arrives (too slow)
    2. Plan movements by simulating their consequences
    3. Distinguish self-generated from external sensory input
       (predict and cancel self-generated feedback)

In Genesis: not yet implemented. The predictive coding system in
vision implements visual prediction; the motor_learning would
extend this to motor prediction when motor control is added.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The motor_learning has its own local circuitry and doesn't produce the
same large-scale oscillations as the cerebral cortex. However, it
interacts with cortical rhythms through the relay.

Gamma (30-100 Hz) — Cerebellar-cortical coordination
-------------------------------------------------------

Cerebellar activity is coordinated with cortical gamma during
precise motor timing. The motor_learning provides the timing signal
that coordinates cortical gamma bursts during movement (Brembs et
al., 2010).

In Genesis: the motor_learning would contribute to gamma timing
when motor control is implemented. Currently not tracked
separately.

Beta (13-30 Hz) — Motor preparation
--------------------------------------

Cerebellar beta is coordinated with cortical beta during motor
preparation and execution. The motor_learning provides the predictive
timing that stabilizes cortical beta during sustained motor
control.

In Genesis: not yet tracked separately. Would be added when
motor control is implemented.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The motor_learning is not part of the cerebral cortex — it's a
separate structure connected to the cortex via the relay. It
receives motor commands from the frontal subsystem (motor cortex) and
sends refined predictions back via the relay.

The motor_learning interacts with:
    - Frontal subsystem (motor cortex): receives motor commands,
      sends refined predictions
    - Parietal subsystem: receives sensory state for prediction
    - Temporal subsystem: timing for auditory-motor coordination
    - Brainstem: inferior olive provides error signals

The motor_learning's function — error-driven prediction and forward
modeling — is implemented by the predictive coding module, which
stays at the top level (it is multi-subsystem: predictive coding runs in
cortex too). This subsystem re-exports it (the association pattern):

    from genesis_cognitive.motor_learning import PredictiveCodingLayer

Re-exported modules:

    (top-level) learning/predictive.py
        PredictiveCodingLayer — the forward model: each layer
        predicts the activity of the level below and learns from the
        prediction error, the same computational motif as the
        cerebellar adaptive filter (weighted parallel-fiber input,
        error-driven weight adjustment).
        Prediction / PredictionContext / PredictionError — the
        prediction records it produces.

    (top-level) learning/td.py
        TDLearner — eligibility traces and prediction error
        learning. The motor_learning uses the same mathematical
        framework (eligibility traces + error-driven plasticity)
        but for sensory prediction rather than reward prediction.
        Re-exported by action_selection (its anatomical home).

    (vision) visual_cortex.py
        Predictive coding system — each level predicts the level
        below. The motor_learning would extend this principle to the
        motor domain.
"""

from __future__ import annotations

from .._views import view_getattr

# The motor_learning's function is implemented by predictive_coding.py,
# which stays at the top level — it is a whole-brain predictive
# stack (parietal + cerebellar + limbic + prefrontal layers). This
# subsystem lazily re-exports it so the anatomy is a real connection
# layer (the association pattern):
#
#   from genesis_cognitive.motor_learning import PredictiveCodingLayer
_EXPORTS: dict[str, str] = {
    "Prediction": "learning",
    "PredictionContext": "learning",
    "PredictionError": "learning",
    "PredictiveCodingLayer": "learning",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)

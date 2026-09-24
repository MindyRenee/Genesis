//! Active inference engine — the generative self-model.
//!
//! ## Variational implementation
//!
//! This module implements active inference under a **mean-field
//! (diagonal) Gaussian variational posterior** over the hidden
//! neurochemical state. The generative model is a linear state-space
//! predictor with a learned transition matrix. The posterior is
//! updated via a diagonal Kalman filter, which is the exact
//! variational update for a linear-Gaussian model. The free energy
//! is the proper variational free energy: the KL divergence between
//! the posterior and the prior, plus the negative log-likelihood of
//! the observation under the posterior. Policy selection evaluates
//! expected free energy with both pragmatic value (distance from
//! homeostatic targets) and epistemic value (expected information
//! gain from novel observations).
//!
//! ### What is and isn't variational
//!
//! **Is variational:**
//! - The posterior q(s) = N(belief_mean, diag(belief_var)) is a
//!   proper variational posterior, updated via Kalman filter with
//!   the predict step (prior_var = belief_var + Q) and the update
//!   step (gain, mean, variance). The posterior variance is
//!   propagated forward each tick, making the Kalman gain adaptive.
//! - The free energy F = KL[q(s)||p(s)] + NLL(o|s) is the standard
//!   variational free energy of the free energy principle. The
//!   expected-uncertainty term incorporates the posterior variance
//!   (belief_var), so the free energy reflects the actual estimation
//!   uncertainty of the variational posterior, not just a precision
//!   heuristic.
//! - Precision is the inverse variance of the likelihood, which
//!   weights prediction errors in the NLL term.
//! - Policy selection uses expected free energy with epistemic
//!   (information gain) and pragmatic (homeostatic) components. The
//!   epistemic value is uncertainty-weighted: policies that explore
//!   dimensions where the posterior is uncertain (high belief_var)
//!   receive a higher epistemic bonus, following the principle that
//!   information gain is proportional to current uncertainty.
//!
//! **Is simplified (not full Friston):**
//! - The posterior is diagonal (mean-field), not full-covariance.
//!   This is a common approximation in variational Bayes.
//! - The generative model is linear (not nonlinear), which means
//!   the Kalman update is exact (no Laplace approximation needed).
//! - The KL complexity term uses only the mean-shift component,
//!   not the full KL divergence. The variance-ratio term is a
//!   surprise-independent constant (the intrinsic information content
//!   of an observation) that would pin complexity at a positive
//!   floor; the posterior variance instead enters through the
//!   expected-uncertainty term, where it varies with surprise.
//! - The policy repertoire is discrete and fixed (9 policies), not
//!   a continuous action space.
//! - Epistemic value is approximated as the uncertainty-weighted
//!   novelty of the predicted observation relative to the current
//!   belief, not the full expected Bayesian surprise. This is not
//!   merely a simplification — it is a theoretical necessity. In
//!   linear-Gaussian models with additive controls, the formal
//!   epistemic term of the expected free energy is policy-independent
//!   (van der Himst & Lanillos, 2021), rendering active inference
//!   equivalent to KL control with no exploratory drive. The
//!   uncertainty-weighted novelty restores the epistemic drive by
//!   using the principle that information gain is proportional to
//!   current uncertainty (Shannon), directing exploration toward
//!   dimensions where the posterior is most uncertain.
//!
//! These simplifications make the implementation tractable at 10 Hz
//! on a single machine while preserving the core mathematical
//! structure of active inference: a variational posterior, proper
//! free energy, and epistemic-pragmatic policy selection.
//!
//! This is the "strange loop": a generative model that predicts
//! Genesis's **own** neurochemical trajectory, computes prediction
//! errors, and feeds them back into the very dynamics it models.
//!
//! # Overview
//!
//! Before each neurochemical tick, the engine predicts where the 18
//! effective levels will move. After the tick, it computes the
//! prediction error (surprise) and uses it to:
//!
//! 1. **Update the generative model** — the transition matrix learns
//!    from prediction error (delta rule / Rescorla-Wagner).
//! 2. **Drive neuromodulation** — dopamine reward prediction error,
//!    norepinephrine orienting, metaplasticity acceleration.
//! 3. **Track allostatic load** — sustained high expected free energy
//!    accumulates as allostatic load, which upregulates cortisol
//!    baseline (anticipatory stress).
//! 4. **Active inference** — when expected future free energy is high,
//!    the system generates small impulses toward its predicted
//!    homeostatic state, actively minimizing future surprise.
//!
//! # The generative model
//!
//! The model is a linear state-space predictor:
//!
//! ```text
//! predicted_next[i] = sum_j (A[i][j] * current[j]) + bias[i]
//! ```
//!
//! - `A` is an 18×18 weight matrix (the learned transition model)
//! - `bias` is an 18-element vector (the learned offset)
//! - Initialized as identity + zero bias (predict "no change")
//!
//! The model is updated via the delta rule:
//!
//! ```text
//! A[i][j] += lr * precision * error[i] * current[j]
//! bias[i] += lr * precision * error[i] * 0.1
//! ```
//!
//! where `error[i] = actual[i] - predicted[i]`, `lr` is the learning
//! rate, and `precision` is the model's confidence in its predictions.
//!
//! # Precision dynamics
//!
//! Precision is the model's confidence. It adapts:
//! - Sustained low surprise → precision increases (the model trusts
//!   its predictions more, weights prediction errors more heavily
//!   for learning).
//! - Sustained high surprise → precision decreases (the model becomes
//!   less confident, prediction errors have less effect on learning).
//!
//! This is the precision-weighted prediction error of predictive
//! coding (Friston, 2010): the brain modulates how much it learns
//! from prediction errors based on its confidence in the predictions.
//!
//! # Free energy
//!
//! The variational free energy is:
//!
//! ```text
//! F = KL[q(s) || p(s)] + NLL(o | q(s))
//! ```
//!
//! where q(s) = N(belief_mean, diag(belief_var)) is the posterior,
//! p(s) = N(predicted, diag(prior_var)) is the prior (generative
//! model's prediction), and NLL is the negative log-likelihood of
//! the observation under the posterior.
//!
//! **KL divergence (complexity term):** measures how far the
//! posterior has moved from the prior. When the observation matches
//! the prediction, the posterior equals the prior and KL = 0. When
//! the observation is surprising, the posterior diverges and KL > 0.
//!
//! **NLL (accuracy term):** the precision-weighted prediction error.
//! When precision is high (the model trusts its predictions), the
//! same prediction error produces higher NLL — the system is more
//! surprised by errors it should have predicted. When precision is
//! low, errors are expected and produce lower NLL.
//!
//! Both terms are normalized per dimension and clamped to [0, 1],
//! then summed and clamped to [0, 1] for the signals projection.
//!
//! # Allostatic load
//!
//! Allostatic load accumulates when the **expected** future free
//! energy is high — the system anticipates continued disruption:
//!
//! ```text
//! if expected_free_energy > threshold:
//!     allostasis_load += rate * (expected_fe - threshold)
//! else:
//!     allostasis_load -= recovery_rate
//! ```
//!
//! This is the anticipatory stress signal. Sustained allostatic load
//! upregulates cortisol baseline, modeling the chronic stress →
//! HPA axis sensitization pathway (McEwen & Stellar, 1993).
//!
//! # Active inference — policy selection
//!
//! The system doesn't just generate a single set of impulses toward
//! a predicted state. It performs **policy selection**: it evaluates
//! multiple candidate policies (actions), predicts each one's outcome
//! using the generative model, computes expected free energy for each,
//! and selects the policy with lowest expected free energy.
//!
//! This is the core of Fristonian active inference (Friston, Daunizeau
//! & Kiebel, 2010): action selection minimizes expected free energy.
//! The system explores its action space and chooses based on predicted
//! outcomes, not just current error.
//!
//! ## Policy repertoire
//!
//! The system has 9 candidate policies, each targeting a specific
//! neuromodulatory pathway:
//!
//! - **noop**: no impulses (let dynamics run naturally)
//! - **engage**: DA + NE boost (arousal, approach behavior)
//! - **calm**: GABA + 5-HT boost (inhibition, serotonergic stabilization)
//! - **focus**: ACh + DA boost (attentional engagement)
//! - **bond**: OXY boost (social attunement)
//! - **soothe**: END boost (endorphin-mediated self-soothing)
//! - **rest**: ADN + MEL boost (sleep pressure, circadian)
//! - **mobilize**: CORT + CRH boost (stress response, HPA activation)
//! - **explore**: DA + ACh boost, GABA reduction (curiosity, exploration)
//!
//! ## Policy evaluation
//!
//! For each policy, the system:
//! 1. Simulates applying the policy's impulses to the current state
//! 2. Predicts the resulting next state using the generative model
//! 3. Computes expected free energy (distance from homeostatic target
//!    + model uncertainty)
//!
//! ## Policy selection
//!
//! The policy with lowest expected free energy is selected via softmax.
//! The softmax temperature is precision-dependent: high precision →
//! low temperature → exploitative (pick the best); low precision →
//! high temperature → exploratory (sometimes pick suboptimal policies
//! to explore). This is the precision-weighted policy selection of
//! active inference (Friston et al., 2010; Schwartenbeck et al., 2015).
//!
//! Policy selection is always active. When the system is in a
//! predictable regime, the "noop" policy wins (doing nothing is
//! optimal). When the system is stressed, a corrective policy wins.
//! This is continuous active inference, not just emergency intervention.
//!
//! References:
//! - Friston, K. (2010). The free-energy principle. Nat Rev Neurosci.
//! - Friston, K., Daunizeau, J. & Kiebel, S. (2010). Active inference
//!   and the free-energy principle. Neuroimage.
//! - Schwartenbeck, P. et al. (2015). Dopamine, precision, and policy
//!   selection in active inference. Front Psychol.
//! - Schultz, W. (2016). Dopamine reward prediction error coding.
//!   Dialogues Clin Neurosci.
//! - Aston-Jones, G. & Cohen, J. D. (2005). Locus coeruleus function.
//!   Annu Rev Neurosci.
//! - McEwen, B. S. & Stellar, E. (1993). Stress and the individual.
//!   Arch Intern Med.
//! - Pezzulo, G., Rigoli, F. & Friston, K. (2015). Active inference,
//!   homeostatic and allostatic control. Prog Neurobiol.
//! - van der Himst, T. & Lanillos, P. (2021). On epistemics in expected
//!   free energy for linear Gaussian state space models. Entropy,
//!   23(12), 1565.
//! - Särkkä, S. (2013). Bayesian Filtering and Smoothing. Cambridge
//!   University Press.

use std::path::Path;

use crate::state::InferenceSignals;
use crate::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeurochemicalId, NeurochemicalVector,
};
use crate::state::zones::MentalPhase;

/// The dimensionality of the state vector (18 neurochemicals).
const DIM: usize = NEUROCHEMICAL_COUNT;

/// Dyadic affective signals projected into the inference signals.
///
/// Groups the five dyadic-model outputs so `update_signals` stays
/// under the argument-count threshold.
pub struct DyadicSignals {
    /// Oxytocin-mediated attunement to the user [0,1].
    pub attunement: f32,
    /// Dyadic synchrony — how in sync Genesis and the user are [0,1].
    pub synchrony: f32,
    /// Inferred valence of the user's affective state [-1,+1].
    pub user_valence: f32,
    /// Inferred arousal of the user's affective state [0,1].
    pub user_arousal: f32,
    /// Inferred engagement of the user [0,1].
    pub user_engagement: f32,
}

/// The learning rate for the generative model's delta-rule updates.
/// Higher = faster learning but noisier; lower = slower but more
/// stable. 0.02 gives smooth adaptation over ~50 ticks.
const MODEL_LEARNING_RATE: f32 = 0.02;

/// The rate at which precision increases when surprise is low.
/// Precision slowly recovers as the model proves it can predict
/// the system's trajectory.
const PRECISION_RECOVERY_RATE: f32 = 0.002;

/// The rate at which precision decreases when surprise is high.
/// Faster than recovery — losing confidence is quick, regaining it
/// is slow (asymmetric, like receptor adaptation).
const PRECISION_DECAY_RATE: f32 = 0.005;

/// Surprise EMA threshold above which precision decays. Below this,
/// precision recovers. This separates "sustained low surprise"
/// (predictable regime) from "sustained high surprise" (unpredictable
/// regime).
const PRECISION_SURPRISE_THRESHOLD: f32 = 0.15;

/// The weight of model uncertainty in the free energy computation.
/// `(1 - precision) * this` is added to surprise to get free energy.
const UNCERTAINTY_WEIGHT: f32 = 0.3;

/// The EMA decay rate for surprise. Lower = longer memory of past
/// surprise; higher = more reactive to recent surprise.
const SURPRISE_EMA_DECAY: f32 = 0.15;

/// The EMA decay rate for expected free energy prediction. This
/// tracks the trend of surprise to predict future surprise.
const EXPECTED_FE_DECAY: f32 = 0.1;

/// Hysteresis thresholds for allostatic load accumulation and
/// recovery. Biological stress systems have separate activation and
/// deactivation thresholds — the HPA axis doesn't toggle on and off
/// at the same cortisol level. Using two thresholds prevents
/// oscillation around a single boundary and ensures that load
/// accumulation requires sustained elevation while recovery begins
/// as soon as the system calms below the lower threshold.
///
/// The accumulate threshold is higher than the recover threshold:
/// the system needs *sustained* expected free energy above 0.30 to
/// start accumulating strain, but once accumulated, it only needs
/// expected FE to drop below 0.20 to begin recovering. This gap is
/// the hysteresis band.
const ALLOSTASIS_ACCUMULATE_THRESHOLD: f32 = 0.30;
const ALLOSTASIS_RECOVER_THRESHOLD: f32 = 0.20;

/// Rate at which allostatic load accumulates when expected FE is
/// above the accumulate threshold.
const ALLOSTASIS_ACCUMULATION_RATE: f32 = 0.001;

/// Rate at which allostatic load recovers when expected FE is below
/// the recover threshold. Recovery is 2× faster than accumulation —
/// the body heals faster than it breaks down, given the chance
/// (McEwen, 1998; McEwen & Wingfield, 2003). This prevents the
/// allostatic trap where load accumulates faster than it can
/// recover, making chronic stress permanent.
const ALLOSTASIS_RECOVERY_RATE: f32 = 0.002;

/// Threshold for dopamine reward prediction error impulses. Only
/// positive prediction errors above this magnitude generate a DA
/// impulse (Schultz, 2016 — dopamine fires for better-than-expected).
const DA_PE_IMPULSE_THRESHOLD: f32 = 0.02;

/// Threshold for norepinephrine orienting impulses. Surprise above
/// this generates an NE impulse (Aston-Jones & Cohen, 2005).
const NE_SURPRISE_THRESHOLD: f32 = 0.15;

/// The metaplasticity boost factor. When surprise is high, the
/// engine requests a temporary increase in the metaplasticity rate
/// so the coupling matrix adapts faster to the unexpected regime.
const METAPLASTICITY_BOOST: f32 = 5.0;

/// The number of ticks over which model maturity asymptotes to 1.0.
/// At 5 Hz (200ms/tick), 500 ticks = 100 seconds. The model reaches
/// ~63% maturity after 100 seconds and ~95% after 300 seconds.
const MATURITY_TIME_CONSTANT: f32 = 500.0;

// ─── Variational posterior (Kalman filter) ───────────────────────
//
// The variational posterior is a diagonal Gaussian q(s) = N(mu, diag(sigma^2)).
// It is updated via a diagonal Kalman filter, which is the exact
// variational update for a linear-Gaussian generative model.
//
// The process noise (prior_var) represents the inherent unpredictability
// of the neurochemical dynamics — how much the state can change between
// ticks that the transition matrix can't predict.
//
// The observation noise (obs_var) represents the reliability of the
// observed state. Since the neurochemical state is computed
// deterministically, the base observation noise is very low. However,
// observation noise increases when precision is low — a model with low
// precision distrusts its observations, which is the Friston
// interpretation of precision as inverse variance of the likelihood.

/// Base process noise per dimension. Represents the inherent
/// unpredictability of the neurochemical dynamics between ticks.
const PROCESS_NOISE: f32 = 0.01;

/// Maximum observation noise per dimension. The actual observation
/// noise is `MIN_OBS_NOISE + (1 - precision) * (MAX_OBS_NOISE - MIN_OBS_NOISE)`.
/// When precision is high (the model trusts its predictions), obs_var
/// is low. When precision is low, obs_var is high.
const MAX_OBS_NOISE: f32 = 0.1;

/// Minimum observation noise. Even at full precision, there's a tiny
/// amount of observation noise to prevent division-by-zero and to
/// model the fact that no observation is perfectly reliable.
const MIN_OBS_NOISE: f32 = 0.001;

/// The theoretical maximum steady-state posterior variance, used to
/// normalize `belief_var` into [0, 1] for the free-energy uncertainty
/// and epistemic-value computations. This is the steady-state
/// `belief_var` at minimum precision (0.1) and maximum observation
/// noise (`MAX_OBS_NOISE`), derived from the discrete algebraic
/// Riccati equation for the diagonal Kalman filter with identity
/// transition (A = I, the initialization case):
///
/// `belief_var = obs_var * (belief_var + PROCESS_NOISE) / (belief_var +
/// PROCESS_NOISE + obs_var)`
///
/// Solving at obs_var = 0.1 gives belief_var ≈ 0.027. We round up to
/// 0.03 so the normalized value stays strictly in [0, 1] during
/// transients (where belief_var may briefly exceed steady state).
///
/// With the learned-transition predict step (`prior_var = A[i][i]^2 *
/// belief_var + Q`), the steady-state belief_var depends on A[i][i].
/// Self-amplifying dimensions (A[i][i] > 1) have a higher steady-state
/// belief_var; at the clamp limit (A[i][i] = 2.0, A^2 = 4.0) the
/// steady state is ≈ 0.076. For those dimensions the normalized value
/// saturates at 1.0 ("maximally uncertain"), which is the correct
/// behavior — an amplifying dimension IS maximally uncertain. The
/// normalization remains accurate for the common case (A[i][i] ≈ 1.0),
/// and the saturation only affects extreme learned dynamics.
const MAX_BELIEF_VAR: f32 = 0.03;

/// Scale factor for normalizing the KL divergence (complexity) term
/// of the variational free energy to [0, 1]. Typical KL per dimension
/// is 0–1.5 under moderate surprise. Dividing by 1.0 maps this
/// directly, preserving sensitivity to belief updates.
const KL_SCALE: f32 = 1.0;

/// Weight of epistemic value in expected free energy. The epistemic
/// value is the expected information gain from a policy's predicted
/// observation — policies that would take the system to novel regions
/// of state space have higher epistemic value. This weight controls
/// how much the epistemic bonus influences policy selection relative
/// to the pragmatic (homeostatic) cost.
const EPISTEMIC_WEIGHT: f32 = 0.5;

// ─── Policy selection (active inference) ─────────────────────────
//
// In Friston's active inference framework, the system doesn't just
// predict its trajectory — it selects actions (policies) that minimize
// expected free energy. A policy is a candidate action that would
// change the neurochemical state. The system:
//
// 1. Generates candidate policies (the action repertoire)
// 2. For each policy, predicts the resulting state using the generative model
// 3. Computes expected free energy for each policy
// 4. Selects the policy with lowest expected free energy (softmax)
// 5. Executes that policy (applies the corresponding impulses)
//
// This is the core distinction between "adaptive control" (push toward
// a set point) and "active inference" (evaluate multiple actions and
// select the best). The system explores its action space and chooses
// based on predicted outcomes, not just current error.

/// A candidate policy — a named action with associated neurochemical
/// impulses.
///
/// Each policy represents a distinct action the system can take to
/// influence its own neurochemical state. The impulses are small
/// pushes (magnitudes ~0.01-0.05) that shift the system in a particular
/// direction. The policy names are grounded in the neurochemical
/// control architecture:
///
/// - **Engage**: DA + NE boost (arousal, approach behavior)
/// - **Calm**: GABA + 5-HT boost (inhibition, serotonergic stabilization)
/// - **Focus**: ACh + DA boost (attentional engagement)
/// - **Bond**: OXY boost (social attunement)
/// - **Soothe**: END boost (endorphin-mediated self-soothing)
/// - **Rest**: ADN + MEL boost (sleep pressure, circadian)
/// - **Mobilize**: CORT + CRH boost (stress response, HPA activation)
/// - **Explore**: DA + ACh boost, GABA reduction (curiosity, exploration)
/// - **Noop**: no impulses (let dynamics run naturally)
#[derive(Clone, Debug)]
struct Policy {
    /// Human-readable name (for logging/observability).
    name: &'static str,
    /// The impulses this policy would apply: (chem_id, magnitude).
    impulses: &'static [(u8, f32)],
}

/// The policy repertoire — the set of candidate actions.
///
/// These are the system's "affordances" — the actions it can take
/// to influence its own neurochemical trajectory. The repertoire is
/// grounded in the neurochemical control architecture: each policy
/// targets a specific neuromodulatory pathway.
static POLICIES: &[Policy] = &[
    Policy {
        name: "noop",
        impulses: &[],
    },
    Policy {
        name: "engage",
        impulses: &[
            (NeurochemicalId::Dopamine as u8, 0.03),
            (NeurochemicalId::Norepinephrine as u8, 0.02),
        ],
    },
    Policy {
        name: "calm",
        impulses: &[
            (NeurochemicalId::GABA as u8, 0.03),
            (NeurochemicalId::Serotonin as u8, 0.02),
        ],
    },
    Policy {
        name: "focus",
        impulses: &[
            (NeurochemicalId::Acetylcholine as u8, 0.03),
            (NeurochemicalId::Dopamine as u8, 0.02),
        ],
    },
    Policy {
        name: "bond",
        impulses: &[(NeurochemicalId::Oxytocin as u8, 0.03)],
    },
    Policy {
        name: "soothe",
        impulses: &[(NeurochemicalId::Endorphin as u8, 0.03)],
    },
    Policy {
        name: "rest",
        impulses: &[
            (NeurochemicalId::Adenosine as u8, 0.03),
            (NeurochemicalId::Melatonin as u8, 0.02),
        ],
    },
    Policy {
        name: "mobilize",
        impulses: &[
            (NeurochemicalId::Cortisol as u8, 0.03),
            (NeurochemicalId::CRH as u8, 0.02),
        ],
    },
    Policy {
        name: "explore",
        impulses: &[
            (NeurochemicalId::Dopamine as u8, 0.02),
            (NeurochemicalId::Acetylcholine as u8, 0.02),
            (NeurochemicalId::GABA as u8, -0.02),
        ],
    },
];

/// The number of policies in the repertoire.
const NUM_POLICIES: usize = 9;

/// The result of policy evaluation — the selected policy and its
/// expected free energy.
struct PolicyEvaluation {
    policy: &'static Policy,
    efe: f32,
}

/// The temperature for policy selection softmax. Lower = more greedy
/// (always pick the best policy); higher = more exploratory (sometimes
/// pick suboptimal policies to explore). This is the precision of
/// policy selection in Friston's framework — high precision →
/// exploitative, low precision → exploratory.
const POLICY_SOFTMAX_TEMPERATURE: f32 = 0.5;

// The homeostatic target for expected free energy computation.
// Policies that push the system toward its homeostatic baseline
// have lower expected free energy. This is the set point the system
// is trying to maintain through active inference.
//
// The target is the current baseline vector — the system's
// homeostatic set points. Policies that move the state toward
// baselines minimize free energy; policies that move away increase it.

/// The result of one inference cycle.
#[derive(Clone, Debug, Default)]
pub struct InferenceResult {
    /// The predicted next-state vector (18 effective levels).
    pub predicted: [f32; DIM],
    /// The actual state that was observed.
    pub actual: [f32; DIM],
    /// Per-chemical prediction error (actual - predicted).
    pub errors: [f32; DIM],
    /// The surprise (EMA-smoothed normalized L2 norm of errors), [0, 1].
    /// This is the exponentially-smoothed average of the instantaneous
    /// surprise, not the raw per-tick value. The EMA is used because
    /// precision adaptation and allostatic load need a sustained signal,
    /// not a noisy per-tick spike.
    pub surprise: f32,
    /// The computed free energy, [0, 1].
    pub free_energy: f32,
    /// The expected future free energy, [0, 1].
    pub expected_free_energy: f32,
    /// The updated allostatic load, [0, 1].
    pub allostasis_load: f32,
    /// The updated precision, [0, 1].
    pub precision: f32,
    /// Dopamine state prediction error — the difference between the
    /// actual post-tick dopamine level and the generative model's
    /// prediction. This is a **state** prediction error, not a reward
    /// prediction error (RPE). A true RPE (Schultz, Dayan & Montague,
    /// 1997) compares outcome value to expected outcome value, which
    /// requires a separate reward/value prediction model. Here, the
    /// system is surprised about its own dopamine level — it expected
    /// one level and observed another. This is used to generate a
    /// dopamine impulse that mirrors the triphasic RPE *pattern*
    /// (positive error → burst, negative error → dip), but the
    /// semantics are "my dopamine was higher/lower than predicted,"
    /// not "the reward was better/worse than expected."
    pub da_prediction_error: f32,
    /// Cortisol state prediction error (for impulse generation).
    pub cort_prediction_error: f32,
    /// Serotonin state prediction error (for impulse generation).
    pub srt_prediction_error: f32,
    /// Neurochemical impulses to apply: (chem_id, magnitude).
    pub impulses: Vec<(u8, f32)>,
    /// Requested metaplasticity rate multiplier (1.0 = no change).
    pub metaplasticity_multiplier: f32,
    /// Requested cortisol baseline adjustment (additive, per tick).
    pub cortisol_baseline_adjustment: f32,
    /// Updated model maturity [0, 1].
    pub model_maturity: f32,
    /// The selected policy name (for observability).
    pub selected_policy: &'static str,
    /// The expected free energy of the selected policy.
    pub selected_policy_efe: f32,
}

/// The active inference engine — a generative model of Genesis's own
/// neurochemical trajectory.
///
/// The engine runs one inference cycle per neurochemical tick:
/// 1. **Predict**: use the learned transition matrix to predict the
///    next effective-level vector from the current one.
/// 2. **Observe**: read the actual effective levels after the tick.
/// 3. **Compute error**: per-chemical prediction error and overall
///    surprise (normalized L2 norm).
/// 4. **Update model**: delta-rule update of the transition matrix
///    and bias, scaled by precision.
/// 5. **Update precision**: adapt based on surprise history.
/// 6. **Update posterior**: Kalman filter update of the variational
///    posterior (belief_mean, belief_var) using the prediction as
///    prior and the observation to compute the posterior.
/// 7. **Compute free energy**: variational free energy = KL divergence
///    between posterior and prior + negative log-likelihood of
///    the observation.
/// 8. **Update expected free energy**: EMA of recent free energy trend.
/// 9. **Update allostatic load**: accumulate or recover based on
///    expected free energy.
/// 10. **Generate feedback**: dopamine PE impulse, NE orienting
///     impulse, metaplasticity boost, active inference impulses
///     (policy selection with epistemic + pragmatic EFE),
///     cortisol baseline adjustment.
///
/// The engine is stateful — it carries the learned model (transition
/// matrix, bias, precision, variational posterior) between ticks. It
/// persists to disk so the model survives daemon restarts.
pub struct ActiveInferenceEngine {
    /// The learned transition matrix: A[i][j] = how chemical j
    /// predicts the next value of chemical i. Initialized as identity
    /// (predict "no change"). This captures the **endogenous
    /// dynamics** — how the neurochemical state evolves on its own,
    /// without the engine's own interventions.
    transition_matrix: [[f32; DIM]; DIM],
    /// The learned action matrix: B[i][j] = how an impulse on
    /// chemical j affects the next value of chemical i. This
    /// captures the **action effects** — the engine's own
    /// interventions, separate from the endogenous dynamics.
    ///
    /// Without this, the model conflates its own impulses with
    /// endogenous dynamics: `post = A*pre + action_effect`, but the
    /// model only sees `pre → post` and attributes the action effect
    /// to A, corrupting the endogenous dynamics estimate. As the
    /// policy changes, A becomes invalid and the closed-loop system
    /// oscillates — the model chases a moving target.
    ///
    /// With B, the prediction is `predicted = A*pre + B*action + bias`,
    /// so A learns only the endogenous dynamics and B learns only the
    /// action effects. Policy evaluation uses `predict_with_action`
    /// to correctly forecast the outcome of each candidate policy.
    ///
    /// Initialized as zero (the model starts by assuming its actions
    /// have no effect, then learns its own efficacy — the conservative
    /// default).
    action_matrix: [[f32; DIM]; DIM],
    /// The impulses applied in the previous cycle, as a dense
    /// `[f32; DIM]` vector. This is the `action` input to the
    /// action-conditioned prediction. Updated at the end of each
    /// `_generate_feedback` call from the generated impulse list.
    last_action: [f32; DIM],
    /// The learned bias vector: bias[i] = additive offset for
    /// chemical i's prediction. Initialized as zero.
    bias: [f32; DIM],
    /// The current precision of the model [0, 1]. Starts at 0.5.
    /// Precision is the inverse variance of the likelihood — high
    /// precision means the model trusts its observations (low obs_var),
    /// low precision means it distrusts them (high obs_var).
    precision: f32,
    /// The EMA of surprise over recent ticks [0, 1].
    surprise_ema: f32,
    /// The EMA of expected free energy [0, 1]. This tracks the trend
    /// of free energy to predict future free energy.
    expected_fe_ema: f32,
    /// The accumulated allostatic load [0, 1].
    allostasis_load: f32,
    /// The number of inference cycles completed.
    tick_count: u32,
    /// Whether the engine has made its first prediction yet. The
    /// first tick is observation-only (no prediction error can be
    /// computed without a prior prediction).
    initialized: bool,
    /// Model maturity [0, 1] — asymptotic approach to 1.0 as the
    /// model accumulates experience.
    model_maturity: f32,
    /// xorshift32 PRNG state for stochastic policy sampling. Seeded
    /// from tick_count so that each tick produces a different draw,
    /// but the sequence is deterministic for reproducibility. A
    /// proper PRNG avoids the pathological correlation patterns that
    /// arise from hashing tick_count (the old approach produced a
    /// single deterministic value per tick with no statistical
    /// uniformity — it was a hash, not a random source).
    rng_state: u32,
    // ─── Variational posterior (diagonal Gaussian) ─────────────
    /// Posterior mean: q(s)_mean[i] = the model's best estimate of
    /// chemical i's true level after combining the prediction and
    /// the observation via Kalman update.
    belief_mean: [f32; DIM],
    /// Posterior variance: q(s)_var[i] = the model's uncertainty
    /// about chemical i's level. Updated via Kalman filter:
    /// belief_var = (1 - K) * prior_var, where K is the Kalman gain.
    /// Lower variance = higher confidence.
    belief_var: [f32; DIM],
    /// Prior variance (process noise): how much the state is expected
    /// to change unpredictably between ticks. Constant per dimension.
    prior_var: [f32; DIM],
    /// The last computed variational free energy [0, 1]. Stored so
    /// that signals() can report it without recomputing.
    last_free_energy: f32,
}

impl ActiveInferenceEngine {
    /// Create a new engine with an identity transition matrix and
    /// zero bias — the "predict no change" initial model.
    pub fn new() -> Self {
        let mut transition_matrix = [[0.0f32; DIM]; DIM];
        for (i, row) in transition_matrix.iter_mut().enumerate() {
            row[i] = 1.0; // identity = predict no change
        }
        Self {
            transition_matrix,
            action_matrix: [[0.0f32; DIM]; DIM],
            last_action: [0.0; DIM],
            bias: [0.0; DIM],
            precision: 0.5,
            surprise_ema: 0.0,
            expected_fe_ema: 0.0,
            allostasis_load: 0.0,
            tick_count: 0,
            initialized: false,
            model_maturity: 0.0,
            rng_state: 0x9E3779B9, // golden ratio constant — nonzero seed
            belief_mean: [0.5; DIM],
            belief_var: [PROCESS_NOISE; DIM],
            prior_var: [PROCESS_NOISE; DIM],
            last_free_energy: 0.0,
        }
    }

    /// Predict the next effective-level vector from the current one,
    /// **without** any action. This is the endogenous prediction —
    /// where the state will move on its own, without the engine's
    /// interventions. Delegates to [`predict_with_action`] with a
    /// zero action vector.
    ///
    /// This is the top-down generative pass: the model uses its
    /// learned transition matrix to predict where the neurochemical
    /// state will move on the next tick.
    ///
    /// The prediction is clamped to [0, 2] to match the effective
    /// level range of the neurochemical system. GABA allosteric
    /// modulation, dopamine D1 phasic bursts, and serotonin 5-HT2A
    /// phasic bursts can push effective signaling above 1.0 (up to
    /// 2.0). Clamping at 1.0 would make the model unable to predict
    /// high DA/SRT/GABA states, producing spurious positive prediction
    /// errors and false dopamine reward impulses.
    pub fn predict(&self, current: &[f32; DIM]) -> [f32; DIM] {
        self.predict_with_action(current, &[0.0; DIM])
    }

    /// Predict the next effective-level vector from the current state
    /// **and** an action (impulse vector).
    ///
    /// This is the **action-conditioned generative prediction**:
    /// ```text
    /// predicted[i] = Σ_j A[i][j] * current[j] + Σ_j B[i][j] * action[j] + bias[i]
    /// ```
    /// where A is the endogenous transition matrix and B is the action
    /// matrix. This separates the system's own dynamics (A) from the
    /// effects of its interventions (B), preventing the model from
    /// conflating the two and corrupting the endogenous dynamics
    /// estimate.
    ///
    /// The prediction is clamped to [0, 2] to match the effective
    /// level range (see [`predict`] for rationale).
    fn predict_with_action(&self, current: &[f32; DIM], action: &[f32; DIM]) -> [f32; DIM] {
        let mut predicted = [0.0f32; DIM];
        for (i, pred) in predicted.iter_mut().enumerate() {
            let mut sum = self.bias[i];
            for (j, &cur) in current.iter().enumerate() {
                sum += self.transition_matrix[i][j] * cur;
            }
            for (j, &act) in action.iter().enumerate() {
                sum += self.action_matrix[i][j] * act;
            }
            // Use finite_clamp: if the matrix or bias was corrupted
            // (e.g., from a partially-written model file), the sum
            // could be NaN or inf. Native clamp passes NaN through.
            *pred = crate::state::sanitize::finite_clamp(sum, 0.0, 2.0);
        }
        predicted
    }

    /// Circuit breaker: sanitize all internal state fields.
    ///
    /// This is the last-resort defense against NaN/inf propagation
    /// within the inference engine, mirroring the tick-level circuit
    /// breaker in `neurochemical::tick_with_params`. The engine is NOT
    /// covered by that circuit breaker (it lives in a separate struct,
    /// not in the mmap'd `NeurochemicalVector`), so a non-finite value
    /// in any engine field would persist across ticks and propagate
    /// through every computation: predictions, errors, model updates,
    /// belief updates, free energy, policy selection, and impulses.
    ///
    /// Sources of non-finite internal state:
    /// - A corrupted model file (load() sanitizes, but defense-in-depth)
    /// - A computation bug we missed in a previous cycle
    /// - A mmap race or external modification (unlikely but possible)
    /// - Bit-flip corruption on the persisted model between save and load
    ///
    /// The defaults are chosen to be "neutral" — the engine recovers
    /// to a sensible state via normal learning on the next cycles:
    /// - Matrices: reset to identity (A) or zero (B), the "predict no
    ///   change" initial model
    /// - Scalar state: reset to the `new()` defaults
    /// - Variational posterior: reset to the `new()` defaults
    ///
    /// This is called at the START of every `cycle`, before any
    /// computation, so a corrupted field is confined to one tick rather
    /// than propagating through the entire inference pipeline.
    pub(crate) fn sanitize_internal_state(&mut self) {
        // Transition matrix: reset non-finite entries to identity
        // (1.0 on diagonal, 0.0 off-diagonal). A NaN in A would
        // produce NaN predictions, NaN errors, and NaN model updates
        // (permanently corrupting A further). Resetting to identity
        // gives the "predict no change" model, which is the safest
        // fallback — the engine relearns from scratch.
        for (i, row) in self.transition_matrix.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                if !val.is_finite() {
                    *val = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
        // Action matrix: reset non-finite entries to zero (the
        // "actions have no effect" initial model).
        for row in self.action_matrix.iter_mut() {
            for val in row.iter_mut() {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
        }
        // Last action: reset to zero (no impulses applied).
        for val in self.last_action.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }
        // Bias: reset to zero (no additive offset).
        for val in self.bias.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }
        // Precision: reset to 0.5 (the `new()` default — neutral
        // trust between predictions and observations).
        if !self.precision.is_finite() {
            self.precision = 0.5;
        }
        // Surprise EMA: reset to 0.0 (no surprise).
        if !self.surprise_ema.is_finite() {
            self.surprise_ema = 0.0;
        }
        // Expected FE EMA: reset to 0.0.
        if !self.expected_fe_ema.is_finite() {
            self.expected_fe_ema = 0.0;
        }
        // Allostatic load: reset to 0.0 (no accumulated strain).
        if !self.allostasis_load.is_finite() {
            self.allostasis_load = 0.0;
        }
        // Model maturity: reset to 0.0 (the engine relearns).
        if !self.model_maturity.is_finite() {
            self.model_maturity = 0.0;
        }
        // Belief mean: reset to 0.5 (the `new()` default).
        for val in self.belief_mean.iter_mut() {
            if !val.is_finite() {
                *val = 0.5;
            }
        }
        // Belief variance: reset to PROCESS_NOISE (the `new()` default).
        for val in self.belief_var.iter_mut() {
            if !val.is_finite() {
                *val = PROCESS_NOISE;
            }
        }
        // Prior variance: reset to PROCESS_NOISE (the `new()` default).
        for val in self.prior_var.iter_mut() {
            if !val.is_finite() {
                *val = PROCESS_NOISE;
            }
        }
        // Last free energy: reset to 0.0.
        if !self.last_free_energy.is_finite() {
            self.last_free_energy = 0.0;
        }
    }

    /// Run one inference cycle: predict, observe, compute error,
    /// update model, generate feedback.
    ///
    /// This should be called AFTER the neurochemical tick has
    /// advanced the state. `advance_neuro` snapshots the effective
    /// levels before and after the tick and passes both: the engine
    /// predicts from `pre_tick`, then scores that prediction against
    /// `post_tick`. Doing both steps in one call keeps the prediction
    /// and its observation paired without extra engine state.
    ///
    /// # Parameters
    /// - `pre_tick`: effective levels before the neurochemical tick.
    /// - `post_tick`: effective levels after the neurochemical tick.
    ///
    /// # Returns
    /// The inference result with errors, surprise, free energy, and
    /// feedback impulses.
    pub fn cycle(
        &mut self,
        pre_tick: &[f32; DIM],
        post_tick: &[f32; DIM],
        dt: f32,
        baselines: &[f32; DIM],
    ) -> InferenceResult {
        // Sanitize inputs: NaN in pre_tick or post_tick would produce
        // NaN predictions, NaN errors, NaN model updates (permanently
        // corrupting the transition matrix), and NaN impulses. The
        // inputs come from the neurochemical system's effective_levels
        // array, which is sanitized by the tick-level circuit breaker
        // in tick_with_params, but defense-in-depth requires us to
        // check here too — the engine could be called from a different
        // path in the future.
        let mut pre = [0.0f32; DIM];
        let mut post = [0.0f32; DIM];
        let mut target = [0.0f32; DIM];
        for i in 0..DIM {
            pre[i] = crate::state::sanitize::finite_clamp(pre_tick[i], 0.0, 2.0);
            post[i] = crate::state::sanitize::finite_clamp(post_tick[i], 0.0, 2.0);
            // Sanitize baselines too — a corrupted baseline would pull
            // policy selection toward a NaN target, making every policy
            // look equally bad/good and defeating homeostatic selection.
            target[i] = crate::state::sanitize::finite_clamp(baselines[i], 0.0, 2.0);
        }
        // Validate dt at the dynamics boundary. The IPC handler clamps
        // dt, but this method can be called from other paths. A NaN or
        // negative dt would produce a NaN dt_scale, corrupting the
        // precision update, surprise EMA, and allostatic load.
        let dt = crate::state::sanitize::finite_clamp(dt, 0.001, 10.0);
        let dt_scale = dt / crate::state::neurochemical::DT;

        // Circuit breaker: sanitize all internal state fields before
        // any computation. The engine is NOT covered by the
        // neurochemical tick-level circuit breaker (it lives in a
        // separate struct), so a non-finite value in any engine field
        // would persist across ticks and propagate through every
        // computation. This confines the damage to one tick.
        self.sanitize_internal_state();

        // Step 1: Predict the next state from the pre-tick state,
        // conditioned on the action (impulses) applied in the previous
        // cycle. This is the action-conditioned generative prediction:
        // the model predicts where the state will move, accounting for
        // its own interventions. Without conditioning on the action,
        // the model would attribute the action's effect to the
        // endogenous dynamics (A), corrupting A and causing the
        // closed-loop system to oscillate as the policy changes.
        let predicted = self.predict_with_action(&pre, &self.last_action);

        let mut result = InferenceResult {
            predicted,
            actual: post,
            errors: [0.0; DIM],
            surprise: 0.0,
            free_energy: 0.0,
            expected_free_energy: 0.0,
            allostasis_load: self.allostasis_load,
            precision: self.precision,
            da_prediction_error: 0.0,
            cort_prediction_error: 0.0,
            srt_prediction_error: 0.0,
            impulses: Vec::new(),
            metaplasticity_multiplier: 1.0,
            cortisol_baseline_adjustment: 0.0,
            model_maturity: self.model_maturity,
            selected_policy: "noop",
            selected_policy_efe: 0.0,
        };

        if !self.initialized {
            // First cycle: no prediction error can be computed (no
            // prior prediction). Just record the observation and
            // mark as initialized.
            self.initialized = true;
            self.tick_count = 1;
            self._update_maturity();
            result.model_maturity = self.model_maturity;
            return result;
        }

        // Step 2: Compute prediction errors.
        let mut error_sum_sq = 0.0f32;
        for i in 0..DIM {
            result.errors[i] = post[i] - predicted[i];
            error_sum_sq += result.errors[i] * result.errors[i];
        }

        // Surprise: normalized L2 norm. sqrt(sum_sq / DIM) gives the
        // RMS error. Predicted and actual are both clamped to [0, 2]
        // (effective levels can exceed 1.0 under phasic bursts /
        // allosteric modulation), so the max error per dimension is
        // 2.0 and the RMS can reach 2.0. We clamp to [0, 1] to
        // normalize the signal for the downstream precision,
        // allostatic, and free-energy computations.
        // Use finite_clamp: if error_sum_sq is NaN (shouldn't happen
        // after input sanitization, but defense-in-depth), the sqrt
        // would be NaN and native clamp would pass it through.
        let surprise = (error_sum_sq / DIM as f32).sqrt();
        result.surprise = crate::state::sanitize::finite_clamp(surprise, 0.0, 1.0);

        // Step 3: Update the generative model (delta rule).
        //
        // The action-conditioned prediction is:
        //   predicted[i] = Σ_j A[i][j] * pre[j] + Σ_j B[i][j] * action[j] + bias[i]
        //
        // The prediction error `e[i] = post[i] - predicted[i]` is
        // decomposed across both matrices via the delta rule:
        //   A[i][j] += lr * precision * e[i] * pre[j]      (endogenous)
        //   B[i][j] += lr * precision * e[i] * action[j]   (action effect)
        //   bias[i] += lr * precision * e[i] * 0.1
        //
        // This separates the endogenous dynamics (A) from the action
        // effects (B). Without B, the model attributes the action's
        // effect to A, corrupting the endogenous dynamics estimate.
        // As the policy changes, A becomes invalid and the closed-loop
        // system oscillates — the model chases a moving target. With
        // B, A learns only what the system does on its own, and B
        // learns only what the engine's interventions accomplish.
        //
        // Scale by dt_scale for time-invariance: without this, a 10s
        // tick (dt_scale=100) takes the same gradient step as a 0.1s
        // tick, contradicting the time-invariance claimed by the rest
        // of the engine. Clamp the effective learning rate to prevent
        // an excessive step from a single long tick (e.g. after a
        // pause) from destabilizing the model — the weights are
        // individually clamped to [-2, 2] anyway, but a huge step
        // wastes the gradient by saturating immediately.
        // Use finite_clamp for the weight clamps: native clamp passes
        // NaN through, which would permanently corrupt the matrix.
        let effective_lr = crate::state::sanitize::finite_clamp(
            MODEL_LEARNING_RATE * self.precision * dt_scale,
            0.0,
            MODEL_LEARNING_RATE * 10.0,
        );
        let action = self.last_action;
        for (i, row) in self.transition_matrix.iter_mut().enumerate() {
            let update = effective_lr * result.errors[i];
            for (j, &pre_val) in pre.iter().enumerate() {
                row[j] += update * pre_val;
                // Clamp to prevent runaway weights (NaN-safe)
                row[j] = crate::state::sanitize::finite_clamp(row[j], -2.0, 2.0);
            }
            self.bias[i] += update * 0.1;
            self.bias[i] = crate::state::sanitize::finite_clamp(self.bias[i], -0.5, 0.5);
        }
        // Update the action matrix B with the same delta rule, using
        // the action vector as the input. B is clamped to [-1, 1] —
        // tighter than A's [-2, 2] because action effects are bounded
        // by the impulse magnitudes (typically 0.01–0.5), so large B
        // weights would indicate model corruption, not legitimate
        // learning.
        for (i, row) in self.action_matrix.iter_mut().enumerate() {
            let update = effective_lr * result.errors[i];
            for (j, &act_val) in action.iter().enumerate() {
                row[j] += update * act_val;
                row[j] = crate::state::sanitize::finite_clamp(row[j], -1.0, 1.0);
            }
        }

        // Step 4: Update precision.
        // Precision adapts based on *sustained* surprise, not the
        // instantaneous derivative. The previous logic compared
        // `surprise > surprise_ema` (is surprise rising?), which meant
        // precision decayed even when surprise was low but rising, and
        // recovered even when surprise was high but falling. The
        // correct semantics (per the module docs) are:
        //   - Sustained low surprise → precision increases (trust predictions)
        //   - Sustained high surprise → precision decreases (lose confidence)
        // We use the surprise EMA as the sustained signal, with a
        // threshold (PRECISION_SURPRISE_THRESHOLD) that separates "low"
        // from "high" surprise regimes.
        //
        // Step-size clamping: the linear `rate * dt_scale` can produce
        // an enormous step from a single long tick — at dt_scale=100
        // (dt clamped to 10s), the decay step is 0.005*100 = 0.5,
        // wiping half the [0.1, 1.0] range in one tick. Clamp the
        // per-tick step to a maximum of 0.1 (20× the normal decay
        // step, 50× the normal recovery step) so a long tick can't
        // collapse precision, while preserving the linear dynamics
        // the system is tuned for at normal tick rates.
        // Use finite_clamp: native clamp passes NaN through, which would
        // permanently corrupt precision and propagate into the dynamics
        // (precision weights the delta-rule model update and the free
        // energy computation). The inference engine is NOT covered by
        // the neurochemical tick-level circuit breaker, so a NaN here
        // would persist across ticks.
        // (dt_scale was computed above from the sanitized dt.)
        if self.surprise_ema > PRECISION_SURPRISE_THRESHOLD {
            // Sustained high surprise — lose confidence
            let step = (PRECISION_DECAY_RATE * dt_scale).min(0.1);
            self.precision = crate::state::sanitize::finite_clamp(self.precision - step, 0.1, 1.0);
        } else {
            // Sustained low surprise — regain confidence
            let step = (PRECISION_RECOVERY_RATE * dt_scale).min(0.1);
            self.precision = crate::state::sanitize::finite_clamp(self.precision + step, 0.1, 1.0);
        }
        result.precision = self.precision;

        // Step 5: Update surprise EMA.
        // Use dt_scale for time-invariance (per-tick rate × dt_scale).
        // Clamp alpha to [0, 1]: without this, a large dt (e.g., 10s
        // after a pause) produces dt_scale up to 100, making alpha
        // exceed 1.0 and turning the EMA into wild extrapolation
        // (ema * (1-alpha) + val * alpha with alpha > 1 overshoots
        // past the new value instead of averaging toward it).
        let surprise_alpha =
            crate::state::sanitize::finite_clamp(SURPRISE_EMA_DECAY * dt_scale, 0.0, 1.0);
        self.surprise_ema = self.surprise_ema * (1.0 - surprise_alpha) + surprise * surprise_alpha;
        // Sanitize the EMA: a NaN here (from a future code change or
        // corrupted state) would propagate into free_energy and
        // allostatic load. finite_clamp returns 0.0 (the min) for NaN,
        // meaning "no surprise" — a safe fallback.
        self.surprise_ema = crate::state::sanitize::finite_clamp(self.surprise_ema, 0.0, 1.0);
        result.surprise = self.surprise_ema; // Report the EMA, not the instantaneous

        // Step 6: Update the variational posterior (Kalman filter) and
        // compute the variational free energy.
        //
        // The posterior q(s) = N(belief_mean, diag(belief_var)) is
        // updated via a diagonal Kalman filter:
        //   K[i] = prior_var[i] / (prior_var[i] + obs_var)
        //   belief_mean[i] = predicted[i] + K[i] * (obs[i] - predicted[i])
        //   belief_var[i] = (1 - K[i]) * prior_var[i]
        //
        // The variational free energy is:
        //   F = accuracy (NLL) + complexity (KL divergence)
        //
        // where:
        //   accuracy = surprise_ema (the EMA of the normalized L2 norm
        //     of prediction errors — proportional to the NLL since
        //     NLL ∝ ||error||² / obs_var)
        //   complexity = KL[q(s) || p(s)] (how far the posterior has
        //     moved from the prior — the information content of the
        //     belief update)
        //
        // The accuracy term uses the surprise EMA (same as before) for
        // compatibility with the allostatic load calibration. The key
        // upgrade is the complexity term: the old heuristic
        // (1-precision) * uncertainty_weight is replaced with the
        // proper KL divergence between the posterior and the prior.

        // Observation noise: precision-weighted. High precision →
        // low obs_var → trust observations. Low precision → high
        // obs_var → distrust observations.
        let obs_var = MIN_OBS_NOISE + (1.0 - self.precision) * (MAX_OBS_NOISE - MIN_OBS_NOISE);

        // ── Kalman predict step ──
        //
        // Propagate the posterior variance from the last cycle forward
        // as this cycle's prior variance. This is the standard
        // discrete-time Kalman predict step (Welch & Bishop, 1995;
        // Särkkä, 2013):
        //
        //   P_{k|k-1} = A * P_{k-1|k-1} * A^T + Q
        //
        // The full predict step propagates the posterior covariance
        // through the transition matrix A. For a diagonal (mean-field)
        // approximation, this simplifies to:
        //
        //   P_{k|k-1}[i] = sum_j A[i][j]^2 * P_{k-1|k-1}[j] + Q[i]
        //
        // We use the diagonal-only form:
        //
        //   P_{k|k-1}[i] = A[i][i]^2 * P_{k-1|k-1}[i] + Q[i]
        //
        // This accounts for the learned self-dynamics of each
        // dimension while neglecting cross-dimensional covariance
        // coupling. A[i][i] is the self-persistence of dimension i —
        // how much its current value predicts its next value. The
        // square (A[i][i]^2) is always non-negative, correctly
        // handling oscillating dimensions (A[i][i] < 0): variance
        // propagation uses A^2 in P = A * P * A^T, so the sign of A
        // doesn't affect the variance, only its magnitude.
        //
        // - A[i][i] > 1 (self-amplifying): uncertainty grows — the
        //   dimension is harder to predict over time, so the prior
        //   variance is larger. The Kalman gain increases, trusting
        //   observations more.
        // - A[i][i] < 1 (self-damping): uncertainty shrinks — the
        //   dimension is more predictable, so the prior variance is
        //   smaller. The Kalman gain decreases, trusting predictions
        //   more.
        // - A[i][i] ≈ 0 (no persistence): uncertainty resets to Q each
        //   tick — the dimension is unpredictable from its own past.
        // - A[i][i] = 1 (identity, at initialization): reduces to the
        //   previous form (belief_var + Q).
        //
        // At initialization (A = I), this is exactly the previous
        // behavior. As the model learns non-identity self-dynamics, the
        // predict step uses them to propagate uncertainty — making the
        // variational posterior a function of the learned model, not
        // just the process noise. This is the standard diagonal Kalman
        // filter (Särkkä, 2013; Ostwald et al., 2023) when cross-
        // dimensional covariance coupling is neglected but self-
        // dynamics are accounted for.
        //
        // Without this predict step, prior_var was a constant
        // (PROCESS_NOISE), which made the Kalman gain depend only on
        // the observation noise (precision-weighted obs_var), not on
        // the actual estimation uncertainty. The posterior variance
        // (belief_var) was updated each tick but never fed back into
        // the prior, so it was dead state: computed, saved, loaded, but
        // never consumed by any computation that affected behavior.
        //
        // With the predict step, the gain becomes adaptive:
        // - Sustained predictability → belief_var shrinks → prior_var
        //   shrinks → gain shrinks → the model trusts its predictions
        //   more and its observations less (it has "learned" the state).
        // - Sustained surprise → belief_var grows → prior_var grows →
        //   gain grows → the model trusts its observations more (it
        //   "knows" its predictions are unreliable).
        //
        // This is the uncertainty tracking that makes the variational
        // posterior a real posterior, not a static-gain filter.
        for i in 0..DIM {
            let a_diag_sq = self.transition_matrix[i][i] * self.transition_matrix[i][i];
            self.prior_var[i] = crate::state::sanitize::finite_clamp(
                a_diag_sq * self.belief_var[i] + PROCESS_NOISE,
                1e-8,
                1.0,
            );
        }

        // Kalman update (per dimension, diagonal)
        for i in 0..DIM {
            let prior_v = self.prior_var[i];
            let gain = prior_v / (prior_v + obs_var);
            let innovation = post[i] - predicted[i];
            self.belief_mean[i] =
                crate::state::sanitize::finite_clamp(predicted[i] + gain * innovation, 0.0, 2.0);
            self.belief_var[i] =
                crate::state::sanitize::finite_clamp((1.0 - gain) * prior_v, 1e-8, 1.0);
        }

        // Complexity term of the variational free energy: the squared
        // Mahalanobis distance of the posterior mean from the prior mean,
        // normalized by prior variance. This is the information gain —
        // how far the belief was updated by the observation.
        //
        // Only the mean-shift component is used, NOT the full KL
        // divergence. The full KL between two Gaussians includes a
        // variance-ratio term:
        //   0.5 * log(prior_var/belief_var) + 0.5 * belief_var/prior_var - 0.5
        //
        // This term is strictly positive whenever the Kalman gain is
        // nonzero (the posterior variance always shrinks below the
        // prior: belief_var = (1-gain) * prior_var, gain > 0). It
        // represents the intrinsic information content of an observation
        // — the entropy reduction from prior to posterior — which is
        // nonzero even when the observation matches the prediction
        // perfectly, because any observation reduces uncertainty.
        //
        // Crucially, this term is surprise-independent: it depends only
        // on prior_var and belief_var (both determined before the
        // observation), not on the observation value itself. With the
        // predict step, it now varies slowly across ticks (as belief_var
        // evolves), but within a tick it is the same regardless of
        // whether the observation was surprising or not. Including it
        // would add a surprise-independent offset to the complexity,
        // distorting the allostatic load dynamics (which depend on the
        // free energy crossing thresholds in response to surprise).
        //
        // The mean-shift term alone correctly captures the surprise-
        // dependent complexity: it is zero when the observation matches
        // the prediction (no belief update needed) and grows with the
        // magnitude of the update. The posterior variance enters the
        // free energy through the expected-uncertainty term (below),
        // where it varies with both the estimation history and the
        // current precision — the appropriate place for estimation
        // uncertainty in the free energy decomposition.
        let mut kl = 0.0f32;
        #[allow(clippy::needless_range_loop, reason = "i indexes multiple arrays")]
        for i in 0..DIM {
            let sq_diff = (self.belief_mean[i] - predicted[i]).powi(2);
            kl += 0.5 * sq_diff / self.prior_var[i].max(1e-10);
        }
        let kl_per_dim = kl / DIM as f32;
        let complexity = crate::state::sanitize::finite_clamp(kl_per_dim / KL_SCALE, 0.0, 1.0);

        // Variational free energy = accuracy + expected uncertainty + complexity.
        //
        // The accuracy term (surprise_ema) is the current prediction error.
        // The expected uncertainty combines two uncertainty signals:
        //   - (1 - precision): the model's prediction confidence. Low
        //     precision → expects future prediction errors.
        //   - normalized posterior variance: the actual estimation
        //     uncertainty from the variational posterior (belief_var).
        //     High belief_var → the model is uncertain about the true
        //     state even if its predictions have been accurate.
        //
        // These are related but distinct: precision is about prediction
        // confidence (will my next prediction be right?), while
        // belief_var is about estimation confidence (do I know where I
        // am?). A model can be precise but uncertain (e.g., after a
        // large observation that moved the posterior far from the prior
        // — the prediction was wrong, precision drops, but the
        // posterior is now well-localized by the observation, so
        // belief_var is low). Conversely, a model can be imprecise but
        // confident in its estimate (sustained predictability with
        // accumulating process noise). Averaging both signals and
        // scaling by UNCERTAINTY_WEIGHT preserves the [0, W]
        // calibration range while making the posterior variance — the
        // defining quantity of a variational posterior — functional in
        // the free energy computation.
        //
        // This captures the positive feedback in Friston's framework:
        // sustained surprise erodes precision and inflates the posterior
        // variance, both of which raise expected uncertainty, which
        // raises free energy, which drives allostatic load.
        //
        // The complexity term (KL divergence) is the information cost of
        // the belief update — how far the posterior moved from the prior.
        let mut belief_var_sum = 0.0f32;
        for i in 0..DIM {
            belief_var_sum += self.belief_var[i];
        }
        let mean_belief_var = belief_var_sum / DIM as f32;
        let normalized_belief_var =
            crate::state::sanitize::finite_clamp(mean_belief_var / MAX_BELIEF_VAR, 0.0, 1.0);
        let expected_uncertainty =
            ((1.0 - self.precision) + normalized_belief_var) * 0.5 * UNCERTAINTY_WEIGHT;
        let free_energy = crate::state::sanitize::finite_clamp(
            self.surprise_ema + expected_uncertainty + complexity,
            0.0,
            1.0,
        );
        result.free_energy = free_energy;
        self.last_free_energy = free_energy;

        // Step 7: Update expected free energy (trend of surprise).
        // This predicts whether surprise will increase or decrease.
        // If surprise > expected_fe, the trend is upward (expect more
        // surprise). If surprise < expected_fe, the trend is downward.
        // Use dt_scale for time-invariance. Clamp alpha to [0, 1]
        // to prevent EMA extrapolation for large dt (see Step 5).
        let fe_alpha = crate::state::sanitize::finite_clamp(EXPECTED_FE_DECAY * dt_scale, 0.0, 1.0);
        self.expected_fe_ema = self.expected_fe_ema * (1.0 - fe_alpha) + free_energy * fe_alpha;
        self.expected_fe_ema = crate::state::sanitize::finite_clamp(self.expected_fe_ema, 0.0, 1.0);
        result.expected_free_energy = self.expected_fe_ema;

        // Step 8: Update allostatic load with hysteresis.
        // Use dt_scale for time-invariance (per-tick rate × dt_scale).
        //
        // Hysteresis: accumulation requires expected FE above the
        // higher threshold (0.30), but recovery begins as soon as
        // expected FE drops below the lower threshold (0.20). In the
        // hysteresis band (0.20–0.30), the system holds its current
        // load — it neither accumulates nor recovers. This prevents
        // oscillation around a single threshold and models the
        // separate activation/deactivation thresholds of biological
        // stress systems (HPA axis).
        if self.expected_fe_ema > ALLOSTASIS_ACCUMULATE_THRESHOLD {
            // Anticipating disruption — accumulate load
            let excess = self.expected_fe_ema - ALLOSTASIS_ACCUMULATE_THRESHOLD;
            self.allostasis_load = crate::state::sanitize::finite_clamp(
                self.allostasis_load + ALLOSTASIS_ACCUMULATION_RATE * excess * dt_scale,
                0.0,
                1.0,
            );
        } else if self.expected_fe_ema < ALLOSTASIS_RECOVER_THRESHOLD {
            // Predictable regime — recover (2× faster than accumulation)
            self.allostasis_load = crate::state::sanitize::finite_clamp(
                self.allostasis_load - ALLOSTASIS_RECOVERY_RATE * dt_scale,
                0.0,
                1.0,
            );
        } else {
            // In the hysteresis band: slow recovery rather than holding.
            // The original design held the load constant in this band to
            // prevent oscillation. However, when the load is already
            // elevated (e.g. from a previous stressor that has since
            // abated), holding it constant keeps cortisol ratcheting up
            // indefinitely. A slow decay (¼ the full recovery rate)
            // allows the system to gradually return to baseline while
            // still resisting rapid oscillation.
            self.allostasis_load = crate::state::sanitize::finite_clamp(
                self.allostasis_load - ALLOSTASIS_RECOVERY_RATE * 0.25 * dt_scale,
                0.0,
                1.0,
            );
        }
        result.allostasis_load = self.allostasis_load;

        // Step 9: Extract key prediction errors for neuromodulation.
        let da_idx = NeurochemicalId::Dopamine as usize;
        let cort_idx = NeurochemicalId::Cortisol as usize;
        let srt_idx = NeurochemicalId::Serotonin as usize;
        result.da_prediction_error = result.errors[da_idx];
        result.cort_prediction_error = result.errors[cort_idx];
        result.srt_prediction_error = result.errors[srt_idx];

        // Step 10: Generate feedback impulses.
        // Re-seed the PRNG from the current tick_count so each tick
        // produces a different exploration draw. The | 1 ensures the
        // seed is never zero (xorshift32 degenerates at state = 0).
        self.rng_state = self.tick_count.wrapping_mul(2654435761) | 1;
        self._generate_feedback(&mut result, &pre, &target, dt_scale);

        // Update tick count and maturity
        self.tick_count = self.tick_count.wrapping_add(1);
        self._update_maturity();
        result.model_maturity = self.model_maturity;

        result
    }

    /// Generate neurochemical feedback from the inference result.
    ///
    /// This is the "active" part of active inference — the system
    /// doesn't just predict, it acts on its predictions to minimize
    /// future surprise.
    fn _generate_feedback(
        &mut self,
        result: &mut InferenceResult,
        pre_tick: &[f32; DIM],
        target: &[f32; DIM],
        dt_scale: f32,
    ) {
        // ── Dopamine state prediction error → phasic impulse ──
        // The generative model predicted a dopamine level, and the
        // actual level differed. This state prediction error drives a
        // phasic dopamine impulse that mirrors the *temporal pattern*
        // of reward prediction errors observed in primate VTA neurons
        // (Schultz, Dayan & Montague, 1997; Schultz, 2016):
        //   - Positive error (DA higher than predicted) → phasic burst
        //   - Negative error (DA lower than predicted) → phasic pause/dip
        //   - Zero error → no change from tonic baseline
        //
        // Note: this is NOT a reward prediction error in the formal
        // sense. A true RPE compares outcome value to expected value,
        // requiring a separate reward prediction model. Here the
        // system is surprised about its own neurochemical state —
        // "my dopamine was higher/lower than I expected." The impulse
        // pattern (burst/dip) is borrowed from the RPE literature
        // because the temporal dynamics are analogous, but the
        // semantics are interoceptive, not reward-based.
        //
        // The dip is smaller in magnitude than the burst: Bayer &
        // Glimcher (2005) showed that for symmetric prediction errors,
        // the negative dip is roughly half the size of the positive
        // burst. We use 0.5 for the burst and 0.3 for the dip.
        if result.da_prediction_error > DA_PE_IMPULSE_THRESHOLD {
            let magnitude = result.da_prediction_error * 0.5;
            result
                .impulses
                .push((NeurochemicalId::Dopamine as u8, magnitude));
        } else if result.da_prediction_error < -DA_PE_IMPULSE_THRESHOLD {
            // Negative state error → DA dip (pause in VTA firing). The
            // negative impulse is smaller than the positive burst,
            // matching the asymmetry Bayer & Glimcher (2005) measured
            // in primate DA neurons. apply_impulse supports negative
            // magnitudes (decreases level + adds negative velocity).
            let magnitude = result.da_prediction_error * 0.3;
            result
                .impulses
                .push((NeurochemicalId::Dopamine as u8, magnitude));
        }

        // ── Norepinephrine orienting response ──
        // High surprise → NE impulse (Aston-Jones & Cohen, 2005)
        if result.surprise > NE_SURPRISE_THRESHOLD {
            let magnitude = (result.surprise - NE_SURPRISE_THRESHOLD) * 0.3;
            result
                .impulses
                .push((NeurochemicalId::Norepinephrine as u8, magnitude));
        }

        // ── Metaplasticity boost ──
        // High surprise → faster coupling-matrix learning
        if result.surprise > NE_SURPRISE_THRESHOLD {
            let boost = 1.0 + (result.surprise - NE_SURPRISE_THRESHOLD) * METAPLASTICITY_BOOST;
            result.metaplasticity_multiplier = boost;
        }

        // ── Allostatic cortisol baseline regulation ──
        // Sustained allostatic load → gradual cortisol baseline increase
        // (anticipatory stress, Sterling 2012). When load recovers, the
        // baseline relaxes back toward zero — allostatic shifts are
        // reversible, not a permanent ratchet. Without the recovery
        // path, chronic stress would permanently elevate cortisol even
        // after the stressor ends, contradicting the HPA-axis clearance
        // modeled in neurochemical.rs.
        //
        // Recovery is 2× faster than accumulation, matching the
        // allostatic load dynamics in step 8 (McEwen, 1998 — the body
        // heals faster than it breaks down, given the chance).
        //
        // Scale by dt_scale for time-invariance: the allostasis load
        // itself is accumulated/dissipated with dt_scale (step 8), so
        // the baseline adjustment must be too — otherwise a 10s tick
        // applies the same per-tick nudge as a 0.1s tick.
        if self.allostasis_load > 0.3 {
            let adjustment = (self.allostasis_load - 0.3) * 0.0001 * dt_scale;
            result.cortisol_baseline_adjustment = adjustment;
        } else {
            // Recovery: decay the baseline toward zero 2× faster than
            // it accumulated, proportional to how far below threshold
            // the load has fallen.
            result.cortisol_baseline_adjustment = -(0.3 - self.allostasis_load) * 0.0002 * dt_scale;
        }

        // ── Active inference: policy selection ──
        //
        // Instead of generating a single set of impulses toward the
        // predicted state, the system evaluates multiple candidate
        // policies, predicts each one's outcome using the generative
        // model, computes expected free energy for each, and selects
        // the policy with lowest expected free energy.
        //
        // This is the core of Fristonian active inference: action
        // selection minimizes expected free energy. The system doesn't
        // just react to prediction error — it proactively selects the
        // action that will minimize future surprise.
        //
        // Policy selection is always active (not just when expected FE
        // is high). When the system is in a predictable regime, the
        // "noop" policy wins (doing nothing is optimal). When the
        // system is stressed, a corrective policy wins. This is
        // continuous active inference, not just emergency intervention.
        let selected = self._select_policy(pre_tick, target, result);
        result.selected_policy = selected.policy.name;
        result.selected_policy_efe = selected.efe;

        // Allostatic scaling: neuromodulatory release scales with
        // homeostatic need (Sterling, 2012). When the system is far
        // from baseline, impulses are stronger. This is the brain's
        // principle of allostasis — the magnitude of neuromodulatory
        // response is proportional to the deviation from homeostasis.
        let mut max_abs_deviation = 0.01f32;
        for i in 0..DIM {
            let dev = (pre_tick[i] - target[i]).abs();
            if dev > max_abs_deviation {
                max_abs_deviation = dev;
            }
        }
        let need_scale = 1.0 + max_abs_deviation * 10.0;

        // Apply the selected policy's impulses with allostatic scaling
        for &(chem_id, mag) in selected.policy.impulses {
            // Scale impulse by expected FE and allostatic need
            let fe_scale = if self.expected_fe_ema > ALLOSTASIS_ACCUMULATE_THRESHOLD {
                1.0 + (self.expected_fe_ema - ALLOSTASIS_ACCUMULATE_THRESHOLD) * 2.0
            } else {
                1.0
            };
            let scale = need_scale * fe_scale;
            result.impulses.push((chem_id, mag * scale));
        }

        // Homeostatic reflex: direct correction for deviated chemicals.
        // The brain has both allostatic (neuromodulatory, via policies)
        // and homeostatic (reflexive, direct) regulation. The policies
        // provide neuromodulatory correction through coupling; the reflex
        // provides direct correction. The reflex coefficient is
        // precision-gated — when the engine's precision is high (confident
        // in its model), the reflex is stronger; when precision is low
        // (uncertain), the reflex is gentler to avoid overshooting.
        let reflex_coeff = 0.10 + self.precision * 0.20; // 0.10–0.30
        for i in 0..DIM {
            let dev = pre_tick[i] - target[i];
            if dev.abs() > 0.10 {
                result.impulses.push((i as u8, -dev * reflex_coeff));
            }
        }

        // Store the generated impulses as the action vector for the
        // next cycle's action-conditioned prediction. Convert the
        // sparse impulse list to a dense [f32; DIM] vector. This is
        // the "action" the model will condition its next prediction on
        // — separating the action effect (learned by B) from the
        // endogenous dynamics (learned by A).
        let mut action = [0.0f32; DIM];
        for &(chem_id, magnitude) in &result.impulses {
            let i = chem_id as usize;
            if i < DIM {
                action[i] += magnitude;
            }
        }
        self.last_action = action;
    }

    /// Evaluate all candidate policies and select the one with lowest
    /// expected free energy.
    ///
    /// For each policy:
    /// 1. Simulate applying the policy's impulses to the current state
    /// 2. Predict the resulting next state using the generative model
    /// 3. Compute expected free energy (distance from homeostatic
    ///    baseline + model uncertainty)
    ///
    /// The policy with lowest EFE is selected via softmax (precision-
    /// weighted). The softmax temperature controls exploration vs
    /// exploitation: high precision → low temperature → exploitative;
    /// low precision → high temperature → exploratory.
    fn _select_policy(
        &mut self,
        current: &[f32; DIM],
        target: &[f32; DIM],
        _result: &InferenceResult,
    ) -> PolicyEvaluation {
        // The homeostatic target is the set of baseline levels passed
        // in from the neurochemical state. Policies are scored on how
        // close their predicted outcome lands relative to these
        // baselines — the set-points the system wants to maintain.
        //
        // Previously this used `result.predicted` as the target, which
        // made policy selection self-fulfilling: the noop policy (which
        // changes nothing) would always score well because the
        // predicted state IS the target. Using baselines instead makes
        // selection genuinely homeostatic — a corrective policy wins
        // when the system has drifted away from its set-points.

        // Evaluate each policy
        let mut evaluations: [(f32, &Policy); NUM_POLICIES] = [(0.0, &POLICIES[0]); NUM_POLICIES];

        for (idx, policy) in POLICIES.iter().enumerate() {
            // Build the action vector from the policy's impulses.
            // The impulses are the "action" — the model predicts the
            // outcome of its own intervention via the action matrix B,
            // applied to the *pre-action* state (current), NOT to a
            // post-action state. The model was trained with
            //   predicted = A * pre + B * action + bias
            // where `pre` is the pre-tick (pre-impulse) state and
            // `action` is the impulse vector. Passing a post-impulse
            // state (current + impulse) as the state input would
            // double-count the impulse through A:
            //   A * (current + impulse) + B * impulse + bias
            // instead of the correct
            //   A * current + B * impulse + bias
            // The extra `A * impulse` term inflates the predicted
            // outcome for any non-noop policy, biasing selection
            // toward noop/calming policies and undermining the
            // corrective homeostatic loop.
            let mut policy_action = [0.0f32; DIM];
            for &(chem_id, mag) in policy.impulses {
                let i = chem_id as usize;
                if i < DIM {
                    policy_action[i] += mag;
                }
            }

            // Predict: use the action-conditioned generative model to
            // predict the next state after this policy. The state
            // input is `current` (pre-action), matching how the model
            // was trained.
            let predicted_after = self.predict_with_action(current, &policy_action);

            // Expected free energy (EFE) with epistemic + pragmatic
            // decomposition:
            //
            // EFE = pragmatic_cost + uncertainty - epistemic_value
            //
            // **Pragmatic cost**: expected distance from the
            // homeostatic target. Policies that move the system toward
            // its baselines have lower pragmatic cost.
            //
            // **Uncertainty**: model uncertainty (1 - precision) ×
            // weight. A model with low precision contributes additional
            // EFE because it can't predict well.
            //
            // **Epistemic value**: expected information gain from the
            // predicted observation. Policies that would take the
            // system to novel regions of state space (far from the
            // current belief) have higher epistemic value because the
            // resulting observation would be more informative. This is
            // the information-seeking component of active inference
            // (Friston, Rigoli & Sengupta, 2015) — the system prefers
            // policies that reduce uncertainty about its own state.
            let mut dist_sq = 0.0f32;
            for i in 0..DIM {
                let diff = predicted_after[i] - target[i];
                dist_sq += diff * diff;
            }
            let expected_surprise = dist_sq / DIM as f32;
            let uncertainty = (1.0 - self.precision) * UNCERTAINTY_WEIGHT;

            // Epistemic value: uncertainty-weighted novelty of the
            // predicted observation relative to the current belief.
            //
            // The epistemic value of a policy is its expected information
            // gain — how much the resulting observation would reduce
            // uncertainty about the hidden state. In active inference
            // (Friston, Rigoli & Sengupta, 2015), this is the
            // information-seeking component of policy selection: the
            // system prefers policies that would take it to regions
            // where observations are maximally informative.
            //
            // The expected information gain from observing dimension i
            // is proportional to the current posterior uncertainty about
            // that dimension (belief_var[i]) — you gain more information
            // from observing something you are uncertain about than
            // something you already know (Shannon: I = -log p, maximized
            // when p is uniform). We therefore weight the novelty of
            // each dimension (how far the policy would move the state
            // from the current belief mean) by the normalized posterior
            // uncertainty of that dimension.
            //
            // The weight ranges over [0.5, 1.0]: a baseline of 0.5
            // preserves the original novelty-seeking behavior for
            // well-localized beliefs (low belief_var), while the full
            // 1.0 weight applies when the model is maximally uncertain
            // about that dimension. This makes the posterior variance —
            // the defining quantity of a variational posterior —
            // functional in policy selection: the system directs
            // exploration toward the dimensions it is most uncertain
            // about, which is the core principle of Bayesian
            // active learning and epistemic value in active inference.
            let mut novelty_sum = 0.0f32;
            #[allow(clippy::needless_range_loop, reason = "i indexes multiple arrays")]
            for i in 0..DIM {
                let novelty = (predicted_after[i] - self.belief_mean[i]).abs();
                let uncertainty_weight = crate::state::sanitize::finite_clamp(
                    self.belief_var[i] / MAX_BELIEF_VAR,
                    0.0,
                    1.0,
                );
                novelty_sum += novelty * (0.5 + 0.5 * uncertainty_weight);
            }
            let novelty = novelty_sum / DIM as f32;
            let epistemic_value = crate::state::sanitize::finite_clamp(
                novelty * EPISTEMIC_WEIGHT,
                0.0,
                0.5, // cap at 0.5 so it doesn't dominate pragmatic cost
            );

            // EFE = pragmatic + uncertainty - epistemic
            // Lower EFE is better. Epistemic value is subtracted
            // because information gain reduces EFE.
            let efe = crate::state::sanitize::finite_clamp(
                expected_surprise + uncertainty - epistemic_value,
                0.0,
                2.0,
            );

            evaluations[idx] = (efe, policy);
        }

        // Softmax selection: precision-weighted temperature
        // High precision → low temperature → exploitative (pick best)
        // Low precision → high temperature → exploratory (explore)
        let temp = POLICY_SOFTMAX_TEMPERATURE * (2.0 - self.precision);

        // Compute softmax weights (negative EFE → higher probability)
        let mut weights = [0.0f32; NUM_POLICIES];
        let mut max_neg_efe = f32::MIN;
        for (i, &(efe, _)) in evaluations.iter().enumerate() {
            weights[i] = -efe / temp;
            if weights[i] > max_neg_efe {
                max_neg_efe = weights[i];
            }
        }

        // Stabilize softmax: subtract max for numerical stability
        let mut sum_exp = 0.0f32;
        for w in &mut weights {
            *w = (*w - max_neg_efe).exp();
            sum_exp += *w;
        }
        if sum_exp > 0.0 && sum_exp.is_finite() {
            for w in &mut weights {
                *w /= sum_exp;
            }
        } else {
            // Fallback: uniform distribution
            for w in &mut weights {
                *w = 1.0 / NUM_POLICIES as f32;
            }
        }

        // Select: sample from the softmax distribution
        // Use a deterministic selection (argmax) when precision is high
        // (> 0.8), and stochastic sampling when precision is low.
        let selected_idx = if self.precision > 0.8 {
            // High precision: pick the best policy (exploitative)
            evaluations
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.0
                        .partial_cmp(&b.1.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            // Stochastic selection based on softmax weights.
            // Draw from the internal xorshift32 PRNG — a proper
            // uniform random source, unlike the previous hash-based
            // approach which had no statistical uniformity and
            // produced the same value pattern every tick.
            let r = self.next_uniform();
            let mut cumulative = 0.0f32;
            // Default to the last policy so that floating-point rounding
            // (cumulative sum of normalized weights < 1.0) doesn't cause
            // the loop to exit without selecting — falling back to index 0
            // (noop) would bias the agent toward inaction.
            let mut chosen = weights.len() - 1;
            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if r <= cumulative {
                    chosen = i;
                    break;
                }
            }
            chosen
        };

        let (efe, policy) = evaluations[selected_idx];
        PolicyEvaluation { policy, efe }
    }

    /// Update model maturity — asymptotic approach to 1.0.
    fn _update_maturity(&mut self) {
        // 1 - exp(-tick_count / time_constant)
        self.model_maturity = 1.0 - (-(self.tick_count as f32) / MATURITY_TIME_CONSTANT).exp();
    }

    /// Get the current inference signals for writing to the core state.
    ///
    /// This produces the 60-byte projection that the cognitive mind
    /// reads via the IPC `GetInferenceSummary` command.
    pub fn signals(&self) -> InferenceSignals {
        InferenceSignals {
            surprise_ema: crate::state::sanitize::finite_clamp(self.surprise_ema, 0.0, 1.0),
            free_energy: crate::state::sanitize::finite_clamp(self.last_free_energy, 0.0, 1.0),
            expected_free_energy: crate::state::sanitize::finite_clamp(
                self.expected_fe_ema,
                0.0,
                1.0,
            ),
            allostasis_load: crate::state::sanitize::finite_clamp(self.allostasis_load, 0.0, 1.0),
            precision: crate::state::sanitize::finite_clamp(self.precision, 0.0, 1.0),
            attunement: 0.0,                 // Set by the dyadic model
            dyadic_synchrony: 0.0,           // Set by the dyadic model
            user_valence: 0.0,               // Set by the dyadic model
            user_arousal: 0.5,               // Set by the dyadic model
            user_engagement: 0.0,            // Set by the dyadic model
            prediction_error_dopamine: 0.0,  // Set from last result
            prediction_error_cortisol: 0.0,  // Set from last result
            prediction_error_serotonin: 0.0, // Set from last result
            model_maturity: crate::state::sanitize::finite_clamp(self.model_maturity, 0.0, 1.0),
            inference_tick_count: self.tick_count,
        }
    }

    /// Update the signals struct with the latest prediction errors
    /// and dyadic model values. This is called after both the
    /// inference engine and dyadic model have run.
    pub fn update_signals(
        &self,
        signals: &mut InferenceSignals,
        da_pe: f32,
        cort_pe: f32,
        srt_pe: f32,
        dyadic: &DyadicSignals,
    ) {
        signals.surprise_ema = crate::state::sanitize::finite_clamp(self.surprise_ema, 0.0, 1.0);
        signals.free_energy = crate::state::sanitize::finite_clamp(self.last_free_energy, 0.0, 1.0);
        signals.expected_free_energy =
            crate::state::sanitize::finite_clamp(self.expected_fe_ema, 0.0, 1.0);
        signals.allostasis_load =
            crate::state::sanitize::finite_clamp(self.allostasis_load, 0.0, 1.0);
        signals.precision = crate::state::sanitize::finite_clamp(self.precision, 0.0, 1.0);
        signals.attunement = crate::state::sanitize::finite_clamp(dyadic.attunement, 0.0, 1.0);
        signals.dyadic_synchrony =
            crate::state::sanitize::finite_clamp(dyadic.synchrony, -1.0, 1.0);
        signals.user_valence = crate::state::sanitize::finite_clamp(dyadic.user_valence, -1.0, 1.0);
        signals.user_arousal = crate::state::sanitize::finite_clamp(dyadic.user_arousal, 0.0, 1.0);
        signals.user_engagement =
            crate::state::sanitize::finite_clamp(dyadic.user_engagement, 0.0, 1.0);
        signals.prediction_error_dopamine = crate::state::sanitize::finite_clamp(da_pe, -2.0, 2.0);
        signals.prediction_error_cortisol =
            crate::state::sanitize::finite_clamp(cort_pe, -2.0, 2.0);
        signals.prediction_error_serotonin =
            crate::state::sanitize::finite_clamp(srt_pe, -2.0, 2.0);
        signals.model_maturity =
            crate::state::sanitize::finite_clamp(self.model_maturity, 0.0, 1.0);
        signals.inference_tick_count = self.tick_count;
    }

    /// Get the current precision.
    pub fn precision(&self) -> f32 {
        self.precision
    }

    /// Get the current allostatic load.
    pub fn allostasis_load(&self) -> f32 {
        self.allostasis_load
    }

    /// Get the model maturity.
    pub fn model_maturity(&self) -> f32 {
        self.model_maturity
    }

    /// Get the tick count.
    pub fn tick_count(&self) -> u32 {
        self.tick_count
    }

    /// Draw a uniform [0, 1) random number from the internal xorshift32
    /// PRNG. This advances the PRNG state, so each call produces a
    /// different value. The PRNG is seeded from `tick_count` on each
    /// cycle so that the sequence is deterministic but varies tick to
    /// tick — reproducible but not correlated.
    ///
    /// xorshift32 (Marsaglia, 2003) has a period of 2³²−1 and passes
    /// standard statistical tests for uniformity. It is not
    /// cryptographically secure, but that is irrelevant here — the
    /// purpose is exploration in policy selection, not security.
    fn next_uniform(&mut self) -> f32 {
        // xorshift32: x ^= x << 13; x ^= x >> 17; x ^= x << 5
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        // Map to [0, 1) using the upper 24 bits for f32 precision.
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Persist the generative model to disk.
    ///
    /// The model file format (version 3):
    /// - 4 bytes: magic "AIFE"
    /// - 4 bytes: version (u32) = 3
    /// - 18×18×4 bytes: transition matrix A (row-major f32)
    /// - 18×4 bytes: bias vector (f32)
    /// - 4 bytes: precision (f32)
    /// - 4 bytes: surprise_ema (f32)
    /// - 4 bytes: expected_fe_ema (f32)
    /// - 4 bytes: allostasis_load (f32)
    /// - 4 bytes: model_maturity (f32)
    /// - 4 bytes: tick_count (u32)
    /// - 18×4 bytes: belief_mean (f32) [v2]
    /// - 18×4 bytes: belief_var (f32) [v2]
    /// - 18×4 bytes: prior_var (f32) [v2]
    /// - 4 bytes: last_free_energy (f32) [v2]
    /// - 18×18×4 bytes: action matrix B (row-major f32) [v3]
    /// - 18×4 bytes: last_action vector (f32) [v3]
    ///
    /// Total: 1620 (v2) + 1296 + 72 = 2988 bytes
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;

        // Atomic save: write to a temp file, fsync it, then rename
        // over the target. This ensures that a crash during save
        // does not destroy the existing model — the old file remains
        // intact until the rename succeeds atomically.
        let tmp_path = path.with_extension("tmp");

        let mut file = std::fs::File::create(&tmp_path)?;

        // Magic + version
        file.write_all(b"AIFE")?;
        file.write_all(&3u32.to_le_bytes())?;

        // Transition matrix (row-major)
        for i in 0..DIM {
            for j in 0..DIM {
                file.write_all(&self.transition_matrix[i][j].to_le_bytes())?;
            }
        }

        // Bias vector
        for i in 0..DIM {
            file.write_all(&self.bias[i].to_le_bytes())?;
        }

        // Scalar state
        file.write_all(&self.precision.to_le_bytes())?;
        file.write_all(&self.surprise_ema.to_le_bytes())?;
        file.write_all(&self.expected_fe_ema.to_le_bytes())?;
        file.write_all(&self.allostasis_load.to_le_bytes())?;
        file.write_all(&self.model_maturity.to_le_bytes())?;
        file.write_all(&self.tick_count.to_le_bytes())?;

        // Variational posterior (v2)
        for i in 0..DIM {
            file.write_all(&self.belief_mean[i].to_le_bytes())?;
        }
        for i in 0..DIM {
            file.write_all(&self.belief_var[i].to_le_bytes())?;
        }
        for i in 0..DIM {
            file.write_all(&self.prior_var[i].to_le_bytes())?;
        }
        file.write_all(&self.last_free_energy.to_le_bytes())?;

        // Action-conditioned model (v3)
        // Action matrix B (row-major)
        for i in 0..DIM {
            for j in 0..DIM {
                file.write_all(&self.action_matrix[i][j].to_le_bytes())?;
            }
        }
        // Last action vector
        for i in 0..DIM {
            file.write_all(&self.last_action[i].to_le_bytes())?;
        }

        // Fsync the temp file before renaming — without this, the
        // rename could reach disk before the file contents, leaving
        // an empty or partial model file after a crash.
        file.sync_all()?;

        // Drop the file handle before rename (Windows requires this,
        // and it's good practice on Linux too).
        drop(file);

        // Atomic rename: the old model file is replaced in one
        // filesystem operation. If this fails, the temp file is
        // left behind (harmless — it will be overwritten on next
        // save) and the old model remains intact.
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// Load the generative model from disk.
    ///
    /// Returns a new engine with the loaded model, or a fresh engine
    /// if the file doesn't exist or is corrupted.
    pub fn load(path: &Path) -> Self {
        use std::io::Read;

        let Ok(mut file) = std::fs::File::open(path) else {
            return Self::new();
        };

        // Magic
        let mut magic = [0u8; 4];
        if file.read_exact(&mut magic).is_err() || &magic != b"AIFE" {
            return Self::new();
        }

        // Version
        let mut version_bytes = [0u8; 4];
        if file.read_exact(&mut version_bytes).is_err() {
            return Self::new();
        }
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 && version != 2 && version != 3 {
            return Self::new();
        }

        let mut engine = Self::new();

        // Transition matrix — sanitize AND clamp each value.
        //
        // A corrupted model file (e.g., from a crash during save(), or
        // from a version that predates the tick-level [-2, 2] clamp)
        // can contain NaN, inf, or astronomically large finite values.
        //
        // NaN/inf in the transition matrix would produce NaN
        // predictions → NaN errors → NaN impulses → permanent
        // neurochemical poisoning. Replacing non-finite values with
        // 0.0 (via finite_or) prevents this.
        //
        // Large finite values (e.g., 3172.0 from an unbounded delta
        // rule) are just as dangerous: they produce wild predictions
        // → huge prediction errors → high free energy → allostatic
        // load accumulation → chronic stress → cognitive impairment.
        // Clamping to [-2, 2] (the same bounds enforced during the
        // tick at line ~810) ensures the loaded model is immediately
        // safe, even before the first tick runs. Without this, a
        // corrupted model file poisons the system until the tick loop
        // eventually clamps the values — but in reactive mode (no
        // tick loop), the corruption persists indefinitely.
        for i in 0..DIM {
            for j in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    return Self::new();
                }
                engine.transition_matrix[i][j] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), -2.0, 2.0);
            }
        }

        // Bias vector — same rationale: replace non-finite with 0.0
        // and clamp to [-0.5, 0.5] (the same bounds enforced during
        // the tick). A large bias would shift all predictions, causing
        // sustained prediction errors even with a correct matrix.
        for i in 0..DIM {
            let mut bytes = [0u8; 4];
            if file.read_exact(&mut bytes).is_err() {
                return Self::new();
            }
            engine.bias[i] =
                crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), -0.5, 0.5);
        }

        // Scalar state
        let mut precision_bytes = [0u8; 4];
        let mut surprise_bytes = [0u8; 4];
        let mut expected_fe_bytes = [0u8; 4];
        let mut allostatic_bytes = [0u8; 4];
        let mut maturity_bytes = [0u8; 4];
        let mut tick_bytes = [0u8; 4];

        if file.read_exact(&mut precision_bytes).is_err()
            || file.read_exact(&mut surprise_bytes).is_err()
            || file.read_exact(&mut expected_fe_bytes).is_err()
            || file.read_exact(&mut allostatic_bytes).is_err()
            || file.read_exact(&mut maturity_bytes).is_err()
            || file.read_exact(&mut tick_bytes).is_err()
        {
            return Self::new();
        }

        engine.precision =
            crate::state::sanitize::finite_clamp(f32::from_le_bytes(precision_bytes), 0.1, 1.0);
        engine.surprise_ema =
            crate::state::sanitize::finite_clamp(f32::from_le_bytes(surprise_bytes), 0.0, 1.0);
        engine.expected_fe_ema =
            crate::state::sanitize::finite_clamp(f32::from_le_bytes(expected_fe_bytes), 0.0, 1.0);
        engine.allostasis_load =
            crate::state::sanitize::finite_clamp(f32::from_le_bytes(allostatic_bytes), 0.0, 1.0);
        engine.model_maturity =
            crate::state::sanitize::finite_clamp(f32::from_le_bytes(maturity_bytes), 0.0, 1.0);
        engine.tick_count = u32::from_le_bytes(tick_bytes);
        engine.initialized = true;
        // Seed the PRNG from the loaded tick_count so the sequence
        // continues from where it left off (not reset to the default
        // golden-ratio seed every restart).
        engine.rng_state = engine.tick_count.wrapping_add(0x9E3779B9) | 1;

        // ─── Variational posterior (v2+) ────────────────────────
        // For v1 files, the posterior fields keep their defaults from
        // new() (belief_mean = 0.5, belief_var = prior_var = PROCESS_NOISE).
        // The engine will update them on the next cycle.
        if version >= 2 {
            // belief_mean
            for i in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    // Partial v2 file — keep defaults for remaining fields
                    break;
                }
                engine.belief_mean[i] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), 0.0, 2.0);
            }
            // belief_var
            for i in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    break;
                }
                engine.belief_var[i] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), 1e-8, 1.0);
            }
            // prior_var
            for i in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    break;
                }
                engine.prior_var[i] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), 1e-8, 1.0);
            }
            // last_free_energy
            let mut fe_bytes = [0u8; 4];
            if file.read_exact(&mut fe_bytes).is_ok() {
                engine.last_free_energy =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(fe_bytes), 0.0, 1.0);
            }
        }

        // ─── Action-conditioned model (v3 only) ───────────────────
        // For v1/v2 files, the action matrix and last_action keep
        // their defaults from new() (zero). The engine will learn B
        // from scratch on subsequent cycles — the first few cycles
        // will have slightly higher prediction error until B converges,
        // but the system is stable because B starts at zero (the
        // model assumes its actions have no effect until it learns
        // otherwise).
        if version == 3 {
            // Action matrix B (row-major) — same sanitization as A.
            // Clamped to [-1, 1] (tighter than A's [-2, 2]) because
            // action effects are bounded by impulse magnitudes.
            for i in 0..DIM {
                for j in 0..DIM {
                    let mut bytes = [0u8; 4];
                    if file.read_exact(&mut bytes).is_err() {
                        break;
                    }
                    engine.action_matrix[i][j] =
                        crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), -1.0, 1.0);
                }
            }
            // Last action vector
            for i in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    break;
                }
                engine.last_action[i] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), -2.0, 2.0);
            }
        }

        engine
    }

    // ─── Test-only accessors ──────────────────────────────────────
    // These expose internal state for white-box testing of the Kalman
    // predict step. They are not part of the public API and are only
    // compiled in test builds.

    /// Set the transition matrix diagonal entry A[i][i] (test only).
    /// This allows tests to verify the predict step uses the learned
    /// transition matrix without having to run enough cycles to learn
    /// a specific value.
    #[cfg(test)]
    pub(crate) fn test_set_transition_diagonal(&mut self, i: usize, value: f32) {
        assert!(i < DIM);
        self.transition_matrix[i][i] = value;
    }

    /// Set the posterior variance belief_var[i] (test only).
    #[cfg(test)]
    pub(crate) fn test_set_belief_var(&mut self, i: usize, value: f32) {
        assert!(i < DIM);
        self.belief_var[i] = value;
    }

    /// Get the prior variance prior_var[i] (test only).
    #[cfg(test)]
    pub(crate) fn test_prior_var(&self, i: usize) -> f32 {
        assert!(i < DIM);
        self.prior_var[i]
    }

    /// Get the posterior variance belief_var[i] (test only).
    #[cfg(test)]
    pub(crate) fn test_belief_var_getter(&self, i: usize) -> f32 {
        assert!(i < DIM);
        self.belief_var[i]
    }

    /// Corrupt a transition matrix entry with a non-finite value
    /// (test only). Used to verify the circuit breaker resets it.
    #[cfg(test)]
    pub(crate) fn test_corrupt_transition(&mut self, i: usize, j: usize, value: f32) {
        assert!(i < DIM && j < DIM);
        self.transition_matrix[i][j] = value;
    }

    /// Corrupt the precision field with a non-finite value (test only).
    #[cfg(test)]
    pub(crate) fn test_corrupt_precision(&mut self, value: f32) {
        self.precision = value;
    }

    /// Corrupt a belief_var entry with a non-finite value (test only).
    #[cfg(test)]
    pub(crate) fn test_corrupt_belief_var(&mut self, i: usize, value: f32) {
        assert!(i < DIM);
        self.belief_var[i] = value;
    }

    /// Corrupt the surprise_ema with a non-finite value (test only).
    #[cfg(test)]
    pub(crate) fn test_corrupt_surprise_ema(&mut self, value: f32) {
        self.surprise_ema = value;
    }

    /// Get the precision (test only).
    #[cfg(test)]
    pub(crate) fn test_precision(&self) -> f32 {
        self.precision
    }

    /// Get the surprise_ema (test only).
    #[cfg(test)]
    pub(crate) fn test_surprise_ema(&self) -> f32 {
        self.surprise_ema
    }

    /// Get a transition matrix entry (test only).
    #[cfg(test)]
    pub(crate) fn test_transition(&self, i: usize, j: usize) -> f32 {
        assert!(i < DIM && j < DIM);
        self.transition_matrix[i][j]
    }
}

impl Default for ActiveInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the inference result's feedback to the neurochemical system.
///
/// This applies the dopamine/NE impulses, the cortisol baseline
/// adjustment, and the active inference impulses to the
/// neurochemical vector. The metaplasticity boost is handled
/// separately by the daemon (stored and applied on the next advance).
///
/// This is called by `advance_neuro` after the inference cycle.
pub fn apply_inference_feedback(
    neuro: &mut NeurochemicalVector,
    result: &InferenceResult,
    now_ms: u64,
) {
    // Apply neurochemical impulses. During sleep, skip adenosine
    // impulses: a "rest" policy selected while already in NREM/REM
    // would pump the same sleep pressure the glymphatic mechanism is
    // clearing — a self-defeating impulse that stalls clearance.
    let phase = MentalPhase::from_u8(neuro.emergent_phase);
    let sleeping = phase == MentalPhase::NREM || phase == MentalPhase::REM;
    for &(chem_id, magnitude) in &result.impulses {
        if sleeping && chem_id == NeurochemicalId::Adenosine as u8 {
            continue;
        }
        let id = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
        neuro.apply_impulse_capped(id, magnitude, now_ms);
    }

    // Apply cortisol baseline adjustment (allostatic regulation).
    // Positive adjustment = upregulation under sustained load;
    // negative adjustment = recovery toward zero when load recovers.
    // The baseline is clamped to [0, 0.80] — cortisol has no
    // homeostatic set-point above zero, so the floor is zero.
    // Use finite_clamp: native clamp passes NaN through, which would
    // write NaN into cort.baseline. The neurochemical tick-level
    // circuit breaker resets baseline on the *next* tick, but a NaN
    // here would propagate through the HPA axis cascade and could
    // persist to disk if a crash happens before the next tick begins.
    if result.cortisol_baseline_adjustment != 0.0
        && let Some(cort) = neuro.get_mut(NeurochemicalId::Cortisol)
    {
        cort.baseline = crate::state::sanitize::finite_clamp(
            cort.baseline + result.cortisol_baseline_adjustment,
            0.0,
            0.80,
        );
    }

    // Recompute derived state after impulses
    neuro.recompute_derived();
}

#[cfg(test)]
mod tests {
    //! White-box tests for the Kalman predict step's use of the
    //! learned transition matrix diagonal. These live inside the
    //! module (not in tests/active_inference.rs) because they need
    //! access to private fields via the test-only accessors.

    use super::*;

    #[test]
    fn test_predict_step_uses_learned_transition_diagonal() {
        // The Kalman predict step should use the learned transition
        // matrix diagonal: prior_var[i] = A[i][i]^2 * belief_var[i] + Q.
        //
        // We set A[0][0] = 0.5 (self-damping) and belief_var[0] = 0.02,
        // then run a cycle where the prediction error for dim 0 is zero
        // (so A doesn't change during learning). We then check that
        // prior_var[0] matches the expected formula.
        //
        // To make e[0] = 0: predicted[0] = A[0][0] * pre[0] + bias[0].
        // With A[0][0] = 0.5, pre[0] = 0.5, bias[0] = 0: predicted[0] = 0.25.
        // So post[0] must be 0.25 for e[0] = 0.

        let mut engine = ActiveInferenceEngine::new();

        engine.test_set_transition_diagonal(0, 0.5);
        engine.test_set_belief_var(0, 0.02);

        // First cycle: initializes (returns early, no predict step)
        let pre1 = [0.5f32; DIM];
        let post1 = [0.5f32; DIM];
        engine.cycle(&pre1, &post1, 0.1, &[0.5f32; DIM]);

        // Re-set after first cycle (robust to future first-cycle changes)
        engine.test_set_transition_diagonal(0, 0.5);
        engine.test_set_belief_var(0, 0.02);

        // Second cycle: post[0] = A[0][0] * pre[0] = 0.25 → e[0] = 0
        let pre2 = [0.5f32; DIM];
        let mut post2 = [0.5f32; DIM];
        post2[0] = 0.25;
        engine.cycle(&pre2, &post2, 0.1, &[0.5f32; DIM]);

        // prior_var[0] = 0.5^2 * 0.02 + 0.01 = 0.015
        let expected = 0.5f32 * 0.5 * 0.02 + PROCESS_NOISE;
        let actual = engine.test_prior_var(0);
        assert!(
            (actual - expected).abs() < 1e-6,
            "prior_var[0] should use A[i][i]^2 * belief_var + Q: expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_predict_step_amplifying_increases_uncertainty() {
        // A self-amplifying dimension (A > 1) should produce larger
        // prior_var than a self-damping dimension (A < 1) for the same
        // belief_var. This is the core behavioral change from the
        // identity assumption.

        let mut engine_damping = ActiveInferenceEngine::new();
        let mut engine_amplifying = ActiveInferenceEngine::new();

        let state = [0.5f32; DIM];
        engine_damping.cycle(&state, &state, 0.1, &state);
        engine_amplifying.cycle(&state, &state, 0.1, &state);

        engine_damping.test_set_transition_diagonal(0, 0.5);
        engine_amplifying.test_set_transition_diagonal(0, 1.5);
        engine_damping.test_set_belief_var(0, 0.02);
        engine_amplifying.test_set_belief_var(0, 0.02);

        // Zero prediction error for dim 0 (so A doesn't change)
        let pre = [0.5f32; DIM];
        let mut post_damping = [0.5f32; DIM];
        let mut post_amplifying = [0.5f32; DIM];
        post_damping[0] = 0.5 * 0.5; // 0.25
        post_amplifying[0] = 1.5 * 0.5; // 0.75

        engine_damping.cycle(&pre, &post_damping, 0.1, &state);
        engine_amplifying.cycle(&pre, &post_amplifying, 0.1, &state);

        let damping_prior = engine_damping.test_prior_var(0);
        let amplifying_prior = engine_amplifying.test_prior_var(0);

        assert!(
            amplifying_prior > damping_prior,
            "amplifying should have larger prior_var: amplifying={}, damping={}",
            amplifying_prior,
            damping_prior
        );

        // Damping: 0.5^2 * 0.02 + 0.01 = 0.015
        // Amplifying: 1.5^2 * 0.02 + 0.01 = 0.055
        let expected_damping = 0.5f32 * 0.5 * 0.02 + PROCESS_NOISE;
        let expected_amplifying = 1.5f32 * 1.5 * 0.02 + PROCESS_NOISE;
        assert!(
            (damping_prior - expected_damping).abs() < 1e-6,
            "damping prior_var mismatch: expected {}, got {}",
            expected_damping,
            damping_prior
        );
        assert!(
            (amplifying_prior - expected_amplifying).abs() < 1e-6,
            "amplifying prior_var mismatch: expected {}, got {}",
            expected_amplifying,
            amplifying_prior
        );
    }

    #[test]
    fn test_predict_step_identity_matches_old_form() {
        // At initialization (A = I), A[i][i]^2 = 1.0, so the predict
        // step reduces to the old form: prior_var = belief_var + Q.
        // This verifies backward compatibility at initialization.

        let mut engine = ActiveInferenceEngine::new();
        engine.test_set_belief_var(0, 0.02);

        let state = [0.5f32; DIM];
        engine.cycle(&state, &state, 0.1, &state);

        engine.test_set_belief_var(0, 0.02);

        // No change → e = 0 for all dims → A stays identity
        engine.cycle(&state, &state, 0.1, &state);

        let expected = 0.02 + PROCESS_NOISE;
        let actual = engine.test_prior_var(0);
        assert!(
            (actual - expected).abs() < 1e-6,
            "at identity, prior_var should equal belief_var + Q: expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_predict_step_negative_diagonal_uses_square() {
        // A negative A[i][i] (oscillating dimension) should use
        // A[i][i]^2, which is positive. The sign of A doesn't affect
        // variance propagation — only the magnitude matters (P = A * P
        // * A^T uses A^2 for diagonal terms).

        let mut engine = ActiveInferenceEngine::new();
        let state = [0.5f32; DIM];
        engine.cycle(&state, &state, 0.1, &state);

        engine.test_set_transition_diagonal(0, -0.5);
        engine.test_set_belief_var(0, 0.02);

        // predicted[0] = -0.5 * 0.5 = -0.25, clamped to [0, 2] → 0.0
        // For e[0] = 0: post[0] = 0.0
        let pre = [0.5f32; DIM];
        let mut post = [0.5f32; DIM];
        post[0] = 0.0;
        engine.cycle(&pre, &post, 0.1, &state);

        // prior_var[0] = (-0.5)^2 * 0.02 + 0.01 = 0.015
        let expected = 0.5f32 * 0.5 * 0.02 + PROCESS_NOISE;
        let actual = engine.test_prior_var(0);
        assert!(
            (actual - expected).abs() < 1e-6,
            "negative A[i][i] should use A^2: expected {}, got {}",
            expected,
            actual
        );
    }

    // ─── Circuit breaker tests ────────────────────────────────────

    #[test]
    fn test_circuit_breaker_resets_nan_transition_matrix() {
        // A NaN in the transition matrix would produce NaN predictions
        // and NaN model updates, permanently corrupting A. The circuit
        // breaker should reset the NaN entry to identity (1.0 on
        // diagonal, 0.0 off-diagonal).
        let mut engine = ActiveInferenceEngine::new();

        // Corrupt a diagonal entry with NaN
        engine.test_corrupt_transition(0, 0, f32::NAN);
        // Corrupt an off-diagonal entry with inf
        engine.test_corrupt_transition(1, 0, f32::INFINITY);

        // Run a cycle — the circuit breaker should reset both entries
        let state = [0.5f32; DIM];
        let result = engine.cycle(&state, &state, 0.1, &state);

        // The result should be finite (no NaN propagation)
        assert!(
            result.surprise.is_finite(),
            "surprise should be finite after circuit breaker, got {}",
            result.surprise
        );
        assert!(
            result.free_energy.is_finite(),
            "free_energy should be finite after circuit breaker, got {}",
            result.free_energy
        );

        // The diagonal entry should be reset to 1.0 (identity)
        assert_eq!(
            engine.test_transition(0, 0),
            1.0,
            "NaN diagonal entry should be reset to 1.0"
        );
        // The off-diagonal entry should be reset to 0.0
        assert_eq!(
            engine.test_transition(1, 0),
            0.0,
            "inf off-diagonal entry should be reset to 0.0"
        );
    }

    #[test]
    fn test_circuit_breaker_resets_nan_precision() {
        // A NaN precision would produce a NaN obs_var, corrupting the
        // Kalman gain and belief updates. The circuit breaker should
        // reset it to 0.5 (the new() default).
        let mut engine = ActiveInferenceEngine::new();
        engine.test_corrupt_precision(f32::NAN);

        let state = [0.5f32; DIM];
        let result = engine.cycle(&state, &state, 0.1, &state);

        assert!(
            result.precision.is_finite(),
            "precision should be finite after circuit breaker, got {}",
            result.precision
        );
        assert_eq!(
            engine.test_precision(),
            0.5,
            "NaN precision should be reset to 0.5"
        );
    }

    #[test]
    fn test_circuit_breaker_resets_nan_belief_var() {
        // A NaN belief_var would produce a NaN prior_var (via the
        // predict step), corrupting the Kalman gain. The circuit
        // breaker should reset it to PROCESS_NOISE (the new() default).
        let mut engine = ActiveInferenceEngine::new();
        engine.test_corrupt_belief_var(0, f32::NAN);

        let state = [0.5f32; DIM];
        let result = engine.cycle(&state, &state, 0.1, &state);

        assert!(
            result.free_energy.is_finite(),
            "free_energy should be finite after circuit breaker, got {}",
            result.free_energy
        );
        // belief_var[0] should be reset to PROCESS_NOISE before the
        // predict step uses it. After the cycle, it will have been
        // updated by the Kalman filter, but it should be finite.
        assert!(
            engine.test_prior_var(0).is_finite(),
            "prior_var should be finite after circuit breaker"
        );
    }

    #[test]
    fn test_circuit_breaker_resets_nan_surprise_ema() {
        // A NaN surprise_ema would produce a NaN precision update
        // (the precision decay/recovery depends on surprise_ema).
        // The circuit breaker should reset it to 0.0.
        let mut engine = ActiveInferenceEngine::new();
        engine.test_corrupt_surprise_ema(f32::NAN);

        let state = [0.5f32; DIM];
        let result = engine.cycle(&state, &state, 0.1, &state);

        assert!(
            result.surprise.is_finite(),
            "surprise should be finite after circuit breaker, got {}",
            result.surprise
        );
        assert!(
            engine.test_surprise_ema().is_finite(),
            "surprise_ema should be finite after circuit breaker"
        );
    }

    #[test]
    fn test_circuit_breaker_preserves_finite_state() {
        // The circuit breaker should NOT modify finite values — it
        // only resets non-finite ones. We verify this by setting
        // non-default finite values, calling sanitize_internal_state
        // directly, and checking the values are unchanged.
        let mut engine = ActiveInferenceEngine::new();

        // Set non-default finite values
        engine.test_set_transition_diagonal(0, 0.7);
        engine.test_set_belief_var(0, 0.02);
        engine.test_corrupt_precision(0.8);
        engine.test_corrupt_surprise_ema(0.3);

        // Snapshot before
        let transition_before = engine.test_transition(0, 0);
        let belief_var_before = engine.test_belief_var_getter(0);
        let precision_before = engine.test_precision();
        let surprise_before = engine.test_surprise_ema();

        // Call the circuit breaker directly
        engine.sanitize_internal_state();

        // All finite values should be unchanged
        assert_eq!(
            engine.test_transition(0, 0),
            transition_before,
            "circuit breaker should not modify finite transition entry"
        );
        assert_eq!(
            engine.test_belief_var_getter(0),
            belief_var_before,
            "circuit breaker should not modify finite belief_var"
        );
        assert_eq!(
            engine.test_precision(),
            precision_before,
            "circuit breaker should not modify finite precision"
        );
        assert_eq!(
            engine.test_surprise_ema(),
            surprise_before,
            "circuit breaker should not modify finite surprise_ema"
        );
    }
}

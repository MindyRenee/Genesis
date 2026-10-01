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
//! - Policy selection minimizes a pragmatic expected-free-energy proxy
//!   over the action-conditioned generative model. Exploration is kept
//!   as a separate temperature-controlled sampling mechanism because
//!   this model class has additive controls.
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
//! - Epistemic value is the expected information gain computed
//!   properly as the entropy reduction of the variational posterior
//!   (`0.5·mean(ln(prior_var/belief_var))`), not a novelty heuristic.
//!   For this model class (linear-Gaussian, additive controls) that
//!   quantity is provably policy-independent (Koudahl, Kouw & de
//!   Vries, 2021: the posterior covariance update does not depend on
//!   the observation value), so it enters the published EFE
//!   decomposition but cannot arbitrate — ranking falls back to the
//!   pragmatic cost, honestly. An earlier revision substituted an
//!   uncertainty-weighted novelty term that was inverted relative to
//!   true information gain and outweighed the pragmatic term 10–100×;
//!   it now serves as the cautionary example in the scoring loop.
//!   Exploration lives where it belongs: the precision-weighted
//!   softmax temperature, which samples when confidence is low and
//!   exploits when it is high.
//!
//!   There is a real result behind the attempt, but it is narrower
//!   than "necessary": in linear-Gaussian state-space models driven
//!   by **additive** controls, the epistemic terms of the expected
//!   free energy are **constant** with respect to policy, so plain
//!   EFE minimisation reduces to KL control with no exploratory
//!   drive (Koudahl, Kouw & de Vries, 2021). The constancy is a
//!   property of that specific model class and control parameterisation,
//!   not of epistemic value in general: multiplicative controls (e.g.
//!   switching transition matrices) do admit an epistemic drive in the
//!   same model class, and generalized Bayesian filtering reinstates
//!   it by making observation precision policy-dependent
//!   (Millidge, Tschantz & Buckley, 2021). Genesis uses additive
//!   controls and does not implement precision-modulated policies, so
//!   it does fall inside the degenerate case — and the honest fix is
//!   the one taken here: compute the information gain properly,
//!   observe that it is policy-independent, keep it in the published
//!   decomposition, and let ranking fall back to the pragmatic cost
//!   with exploration carried by the precision-weighted temperature.
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
//! Precision is the model's confidence, maintained as the
//! inverse-variance target `1 / (1 + err_var / 0.0004)` where
//! err_var is the slow EMA of squared surprise. It adapts:
//! - Sustained low surprise → err_var drains → target rises toward
//!   1.0 → precision slews up (slowly: regaining confidence is slow).
//! - Sustained high surprise → err_var fills → target falls →
//!   precision slews down (quickly: losing confidence is fast).
//!   Saturation at 1.0 at rest is correct — the model genuinely
//!   predicts the resting trajectory — and it is reversible: any
//!   sustained surprise above ~0.01 RMS pulls precision back under the
//!   0.8 exploitation threshold, re-opening stochastic exploration.
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
//! Allostatic load is the cumulative regulatory burden, integrated
//! over time under a hysteresis band:
//!
//! ```text
//! burden = mean of the two middle values of
//!          [expected_fe, surprise, 1 - precision, homeostatic deviation]
//!
//! if burden > ACCUMULATE_THRESHOLD:
//!     allostasis_load += ACCUMULATION_RATE * (burden - threshold) * dt_scale
//! elif burden < RECOVER_THRESHOLD:
//!     allostasis_load -= RECOVERY_RATE * dt_scale
//! else:
//!     allostasis_load -= RECOVERY_RATE * 0.25 * dt_scale
//! ```
//!
//! The burden is deliberately **multisignal**: it is the mean of the
//! two middle values of the four, so one isolated spike cannot
//! manufacture load while elevation across several dimensions can.
//! The third branch — slow recovery inside the band — is what keeps
//! already-accumulated load from becoming permanent under sustained
//! mild stress. Omitting it makes the band a hard freeze, which is
//! the allostatic trap this mechanism exists to model (McEwen, 1998).
//!
//! Expected free energy is one input to that median, not the driver:
//! it is the *anticipatory* demand signal, and load is the accumulated
//! wear. Sustained load upregulates the cortisol baseline, modeling
//! the chronic stress → HPA axis sensitization pathway (McEwen &
//! Stellar, 1993).
//!
//! This engine's load is the **substrate-scoped** view. The cognitive
//! mind keeps a whole-system view that also folds in endocrine and
//! computational-body stress; both are calibrated to the same
//! timescale so they move together.
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
//! Policy selection runs every inference cycle. In a predictable
//! regime the "noop" policy wins (doing nothing is optimal — every
//! policy predicts ~the current state and noop moves nothing) and
//! under deviation a corrective policy wins (whichever the learned
//! action model predicts will close the distance fastest). Early in
//! life, before the action model is learned, all policies predict
//! identically and selection falls through to noop while the
//! low-precision softmax branch explores stochastically — a reflexive
//! developmental phase that gives way to deliberative selection as
//! the action effects are learned.
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
//! - Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T.
//!   & Pezzulo, G. (2015). Active inference and epistemic value.
//!   Cognitive Neuroscience, 6(4), 187–214.
//! - Koudahl, M. T., Kouw, W. M. & de Vries, B. (2021). On epistemics in
//!   expected free energy for linear Gaussian state space models.
//!   Entropy, 23(12), 1565.
//! - Särkkä, S. (2013). Bayesian Filtering and Smoothing. Cambridge
//!   University Press.
//! - Millidge, T. S., Tschantz, A. & Buckley, C. L. (2021). A
//!   whack-a-mole re-derivation of the FEP, and how to pound the
//!   hole. Neural Computation, 33(2), 447–482.

use crate::state::neurochemical::NeuroTickParams;
use std::path::Path;

use crate::state::InferenceSignals;
use crate::state::neurochemical::{NEUROCHEMICAL_COUNT, NeurochemicalId, NeurochemicalVector};
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
/// Per-cycle multiplicative decay of learned weights.
///
/// 0.02 means a weight retains 98% of its value per cycle and half-life
/// of about 34 cycles (~3.4 s at 10 Hz). Chosen to be slow relative to
/// the learning rate so that a well-supported weight is not washed
/// away by decay before evidence accumulates, while still letting a
/// weight the data no longer supports relax back toward zero within a
/// few tens of seconds. Without any decay the model cannot revise a
/// weight it learned once, and cannot track a change in her dynamics.
const MODEL_FORGETTING: f32 = 0.02;

/// The rate at which precision increases when surprise is low.
/// Precision slowly recovers as the model proves it can predict
/// the system's trajectory.
const PRECISION_RECOVERY_RATE: f32 = 0.002;

/// The rate at which precision decreases when surprise is high.
/// Faster than recovery — losing confidence is quick, regaining it
/// is slow (asymmetric, like receptor adaptation).
const PRECISION_DECAY_RATE: f32 = 0.005;

/// EMA rate for the slow surprise-variance tracker that drives the
/// precision target (see `PRECISION_REFERENCE_VAR`). At dt_scale = 1
/// the time constant is ~200 ticks; slow enough that single impulses
/// don't move it, fast enough that a genuine regime change registers
/// within minutes.
const PRECISION_VAR_EMA_RATE: f32 = 0.005;

/// Reference error variance for precision targeting.
///
/// Precision seeks `1 / (1 + slow_err_var / this)`, i.e. the
/// inverse-variance confidence of a Gaussian estimator whose error
/// variance is `slow_err_var`. Calibration: sustained RMS surprise of
/// 0.02 (this value squared) holds precision at 0.5; rest (~3e-4 RMS)
/// holds it at ~1.0; sustained 0.01 RMS holds it at 0.8 — exactly the
/// exploitation threshold, so any regime noisier than that re-opens
/// the exploratory selection branch. The old fixed-threshold scheme
/// (decay above 0.15) could never trigger anywhere near the resting
/// regime and left the exploration branch dead in practice.
const PRECISION_REFERENCE_VAR: f32 = 0.0004;

/// The weight of model uncertainty in the free energy computation.
/// `(1 - precision) * this` is added to surprise to get free energy.
const UNCERTAINTY_WEIGHT: f32 = 0.3;

/// The EMA decay rate for surprise. Lower = longer memory of past
/// surprise; higher = more reactive to recent surprise.
const SURPRISE_EMA_DECAY: f32 = 0.15;

/// The EMA decay rate for expected free energy prediction. This
/// tracks the trend of surprise to predict future surprise.
const EXPECTED_FE_DECAY: f32 = 0.1;

// Context-amplifier gains. These scale the *cost* of stress already
// present; they cannot create burden on their own (see the amplifier
// bound below). The magnitudes are set so that a doubling of the
// physiological measure is the largest effect context can have,
// leaving the mediators in charge of the reading.
const CONTEXT_GAIN_EXPECTED_FE: f32 = 0.5;
const CONTEXT_GAIN_SURPRISE: f32 = 0.5;
/// Hard ceiling on the context amplifier. With both gains at 0.5 the
/// unclamped maximum is 2.0; capping at 1.5 guarantees the amplifier
/// can never contribute more burden than the physiological measure
/// itself, so a demand signal can never outvote the body.
const CONTEXT_GAIN_MAX: f32 = 1.5;

/// Anticipated-demand level at which allostatic amplification begins.
///
/// This is a separate quantity from the load-integrator's dead-band:
/// it sets how much anticipated demand inflates the *cost* of stress
/// that is already present, not whether load accrues at all. Kept at
/// 0.30 so that ordinary anticipatory demand is not treated as
/// elevated.
const CONTEXT_FE_ACTIVATION: f32 = 0.30;

/// Fraction of the resting burden treated as neutral variation.
///
/// A burden within this band of her measured resting level neither
/// accrues nor drains load, so ordinary physiological variation does
/// not integrate without bound.
///
/// Sized from the measured resting spread rather than chosen. With no
/// external drive the burden sits at p50 0.039, p90 0.039, max 0.039 —
/// the resting catecholamines are almost exactly at their genetic
/// baselines, and only a small cortisol baseline is present. Under
/// sustained CRH drive it rises to p50 0.105, p90 0.149, p99 0.32.
///
/// So rest occupies 0.00-0.04 and genuine activation occupies roughly
/// 0.10 upward, with a tail to 0.35. The band is set at 25% of a 0.04
/// resting level — 0.01 — which puts the accrual edge at 0.05: above
/// every resting observation, and below the sustained-stress median.
///
/// This band was previously much wider and the resting reference much
/// higher, on the strength of measurements taken with a probe that was
/// reading the wrong chemical index (it used slot 15, which is CRH,
/// as though it were epinephrine). That misreading put the estimated
/// resting burden at 0.17 when the real figure is 0.039, and the
/// consequence was severe: with the accrual edge at 0.204 and the real
/// resting burden at 0.039, an organism under no stress at all
/// measured a load of 0.69.
const REST_DEADBAND_FRACTION: f32 = 0.25;

/// Rate at which the resting-burden estimate follows the burden.
///
/// Symmetric and slow. The estimate is a long-run average of her
/// burden, because over a long window the mean of that burden *is*
/// her resting level — so averaging makes the reference correct by
/// construction rather than by assumption.
///
/// The earlier version was asymmetric, falling 100x faster than it
/// rose, on the reasoning that a stressor must not be able to raise
/// its own reference and thereby excuse itself. That reasoning was
/// half right and the implementation inverted the problem: the burden
/// is bursty (CRH arrives in pulses), so most samples sit low and the
/// reference collapsed toward zero within seconds while load climbed.
/// Traced directly, the reference fell from 0.0054 to 0.0000002 and
/// the resting organism then measured a load of 0.69. A reference that
/// runs away from its signal defeats the measure in either direction.
///
/// Rate 0.00002 gives a time constant of 50,000 ticks — about 83
/// minutes at 10 Hz. Long enough that no single excursion moves it,
/// short enough to follow a genuine change in her resting physiology
/// over the timescale of a session. A sustained stressor at burden
/// 0.105 shifts the reference only after tens of thousands of ticks,
/// while load has been accruing from the first one.
const REST_TRACK_RATE: f32 = 0.00002;

/// Initial resting-burden estimate before any history exists.
///
/// The catecholamines carry a resting contribution (norepinephrine
/// 0.30, epinephrine 0.15 genetic baselines) that is normal wake
/// physiology rather than pathology, so the initial estimate is
/// nonzero. Set to the measured resting burden mean of 0.04: the
/// dead-band is a fraction of this value, so an overestimate places
/// the accrual edge above genuine stress and the measure goes blind,
/// while an underestimate places it inside the resting distribution
/// and charges a healthy organism for being awake.
const REST_BURDEN_INITIAL: f32 = 0.04;

/// Rate at which allostatic load accumulates per unit of burden above
/// rest. Sets the timescale over which sustained activation becomes
/// wear.
///
/// Derived from the construct's own timescale rather than chosen. Allo-
/// static load is the cumulative cost of chronic exposure measured
/// over a life, so the [0, 1] range has to represent months of
/// sustained activation, not hours.
///
/// At the measured sustained-stress excess of ~0.06 above rest, a rate
/// of 0.001 drove load from zero to saturation in about 14 minutes of
/// continuous stress — the measure saturating inside a single work
/// session, which is not a measure of cumulative wear. For a
/// meaningful elevation over roughly one day of sustained stress
/// (864,000 ticks at 10 Hz), the rate must satisfy
///
/// ```text
/// rate ≈ 0.5 / (864_000 * 0.06) ≈ 1e-5
/// ```
///
/// which puts meaningful accumulation at days and saturation at
/// weeks, matching what the construct describes.
const ALLOSTASIS_ACCUMULATION_RATE: f32 = 0.00001;

/// Rate at which allostatic load recovers per unit of burden below
/// rest. Faster than accumulation, so a system that is no longer
/// stressed sheds its load — the body heals faster than it breaks
/// down, given the chance (McEwen 1998; McEwen & Wingfield 2003).
/// This is what prevents the allostatic trap in which load persists
/// indefinitely after the stressor has gone.
///
/// Twice the accumulation rate, so a load built over days drains over
/// roughly half that time once the stressor is gone. Not larger: the
/// asymmetry only has to favour recovery, and an extreme one would
/// make a transient spike erase genuine accumulated wear.
const ALLOSTASIS_RECOVERY_RATE: f32 = 0.00002;

/// Advance allostatic load by one step under the hysteresis band.
///
/// Pure function of the current load, the multisignal burden, and the
/// time scale — extracted so the band behaviour is directly testable
/// without driving the engine's EMAs to a chosen burden (steps 5 and
/// 7 recompute those EMAs before step 8 reads them, so a test cannot
/// simply pin them and observe the result).
///
/// The result is always a finite value in [0, 1]. A non-finite *load*
/// maps to 0.0 via `finite_clamp` — "no load", the same convention as
/// everywhere else in this file. A non-finite *burden* satisfies
/// neither threshold comparison and therefore takes the band branch:
/// an unknown burden holds the load roughly steady rather than
/// clearing it (a false all-clear) or inflating it (a false alarm).
/// Integrate allostatic load from the current burden.
///
/// The functional form is the integral of a wear rate:
///
/// ```text
/// d(load)/dt = RATE * (burden - rest)      when burden > rest
/// d(load)/dt = -RECOVERY * rest_fraction    when burden < rest
/// ```
///
/// with `rest` being the measured resting burden, tracked as a slow
/// baseline (see `RestBaseline`). This is what "cumulative wear"
/// actually means: McEwen defines allostatic load as "the cost of
/// chronic exposure to fluctuating or heightened neural or
/// neuroendocrine response" — the cost is an integral over time of
/// how far the response sits above its resting level, not a function
/// of instantaneous amplitude.
///
/// ## Why there is no fixed amplitude threshold
///
/// This used to gate accumulation on `burden > 0.30`, with a
/// hysteresis band below. Measured consequences, on the real HPA
/// cascade, were that one hour of chronic stress and six hours of it
/// produced identical peak load (0.25011 both), and that a sustained
/// elevation whose mean sat below the threshold was charged only
/// during its oscillation peaks.
///
/// The cause was structural. A hard threshold on a noisy signal
/// rectifies it: the integrator then measures how often excursions
/// cross a line, rather than how long the signal spends above its
/// resting level. Since cortisol pulses and the catecholamines
/// oscillate, most ticks landed in the band and the load never
/// integrated at all — one hour and six hours of stress became
/// indistinguishable, which is precisely the distinction the
/// construct exists to draw.
///
/// A small dead-band is still required, and it is kept, but it is
/// expressed as a *fraction of the measured resting level* rather
/// than a magic number: a burden within `REST_DEADBAND` of rest
/// neither accrues nor drains, so ordinary physiological variation
/// does not integrate without bound.
fn integrate_allostasis(load: f32, burden: f32, rest: f32, dt_scale: f32) -> f32 {
    let rest = crate::state::sanitize::finite_clamp(rest, 0.0, 1.0);
    let deadband = rest * REST_DEADBAND_FRACTION;
    let next = if burden > rest + deadband {
        // Wear rate is proportional to how far above rest she is, so
        // a mild sustained elevation costs proportionally less than a
        // severe one, and both scale with duration.
        let excess = burden - rest - deadband;
        load + ALLOSTASIS_ACCUMULATION_RATE * excess * dt_scale
    } else if burden < rest - deadband {
        // Recovery is proportional to depth below rest, mirroring
        // accumulation. An earlier version used a fixed recovery rate,
        // which made recovery up to 25x faster than accumulation at
        // realistic signal amplitudes, so a single tick in the band
        // erased twenty-five ticks of accrued wear and load could
        // never build at all.
        let depth = (rest - deadband - burden).max(0.0);
        load - ALLOSTASIS_RECOVERY_RATE * depth * dt_scale
    } else {
        // Inside the dead-band around rest: no change. This is the
        // neutral region for ordinary physiological variation.
        load
    };
    crate::state::sanitize::finite_clamp(next, 0.0, 1.0)
}

/// A slow-tracking estimate of her resting burden.
///
/// Allostatic load is defined as the excess over resting operation, so
/// the reference has to be her actual resting level rather than a
/// constant. Tracking it adaptively also means the measure stays
/// correct if the resting catecholamine contribution changes as her
/// physiology develops, which a hardcoded reference would not.
///
/// It is a slow symmetric average rather than a chasing tracker, so
/// no single excursion in either direction can move it.
struct RestBaseline {
    level: f32,
}

impl RestBaseline {
    fn new(initial: f32) -> Self {
        Self {
            level: crate::state::sanitize::finite_clamp(initial, 0.0, 1.0),
        }
    }

    fn update(&mut self, burden: f32, dt_scale: f32) {
        if !burden.is_finite() {
            return;
        }
        let alpha = crate::state::sanitize::finite_clamp(REST_TRACK_RATE * dt_scale, 0.0, 1.0);
        self.level += alpha * (burden - self.level);
        self.level = crate::state::sanitize::finite_clamp(self.level, 0.0, 1.0);
    }

    fn level(&self) -> f32 {
        self.level
    }
}

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

/// EMA rate for the `maturity_error_ema` tracker. At dt_scale = 1 the
/// time constant is ~500 ticks, which is the timescale on which
/// `model_maturity` is earned: a fresh engine starts at
/// `MATURITY_ERROR_INIT` (0.5) and sustained accuracy has to pull that
/// down before fit approaches 1, so a handful of early predictions
/// cannot claim maturity.
const MATURITY_ERROR_EMA_RATE: f32 = 0.002;

/// Scale for the *legacy* maturity mapping `exp(-err / this)`, retained
/// only to reconstruct `maturity_error_ema` from a pre-v4 model file
/// whose stored value is a maturity rather than an error level
/// (`err = -scale * ln(maturity)`). It does not participate in any
/// live computation of `model_maturity`.
const MATURITY_ERROR_SCALE: f32 = 0.001;

/// Initial value of the retained `maturity_error_ema` tracker. A fresh
/// engine starts maximally unproven; the value only has to round-trip
/// through the model file, since nothing maps it to maturity now.
const MATURITY_ERROR_INIT: f32 = 0.5;

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

// `NUM_POLICIES` sizes the fixed-size `evaluations` and `weights` arrays
// used in the selection loop, and `POLICIES` is the list they index.
// Nothing tied the two together, so adding or removing a policy
// desynchronised them and the mismatch surfaced as a runtime index
// panic inside the 5 Hz tick loop — the worst possible place to
// discover a count that should have been a compile error. Pin it here.
const _: () = assert!(POLICIES.len() == NUM_POLICIES);

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

/// Symmetric magnitude bound applied to a policy's expected free energy.
///
/// This exists solely to reject non-finite values. It must be far wider
/// than any real score: `selected_policy_efe` is published in
/// `InferenceSignals` and consumers may compare it against it, so
/// saturating it would silently flatten the signal. See the selection
/// loop for why a non-negative lower bound is wrong here.
const EFE_SCORE_LIMIT: f32 = 1.0e6;

/// Engine confidence above which policy selection is exploitative
/// (argmin) rather than exploratory (softmax sample).
///
/// Hand-tuned, not derived. See the selection loop for why it is not
/// the Bayesian policy prior precision it is analogous to.
const EXPLOITATION_PRECISION_THRESHOLD: f32 = 0.8;

/// Base strength of the homeostatic reflex toward her preference.
const REFLEX_COEFF_BASE: f32 = 0.10;
/// Additional reflex strength from confidence in her generative model.
const REFLEX_COEFF_PRECISION: f32 = 0.20;
/// Additional reflex strength from confidence in her preference.
const REFLEX_COEFF_PREFERENCE: f32 = 0.10;

/// How fast her preference moves toward states she demonstrably fares
/// well in.
///
/// This is the piece that makes the preference *hers* rather than a
/// constant I supplied. It must be slow: a preference that chases the
/// current state collapses back to "am I where I am", which is the
/// circularity this whole change exists to remove. It moves only on
/// evidence of sustained wellbeing, and the restoring pull toward
/// genetics keeps a drifted preference from running away.
const PREFERENCE_ADAPT_RATE: f32 = 0.00002;
/// Restoring pull of her preference toward the genetic seed.
const PREFERENCE_RESTORE_RATE: f32 = 0.00002;

/// Allostatic load above which her preference stops adapting.
///
/// She learns what suits her from states she is coping well in. Under
/// sustained strain the signal is exactly that things are not going
/// well, and adapting then would teach her to prefer the strained
/// state — the allostatic trap, where a set point walks to meet a
/// condition instead of correcting it.
const PREFERENCE_ADAPT_LOAD_CEILING: f32 = 0.35;

/// Initial confidence in the preference term.
///
/// This is the correction strength, and it is a genuine compromise
/// rather than a value to maximise. The Yerkes–Dodson and
/// disinhibitory-circuit work both find performance optimal at
/// *moderate* arousal, with impairment on both sides; a preference
/// tight enough to pin her to one state would trade away the
/// flexibility that lets her respond to what actually matters to her.
/// At 0.5 she is pulled toward what she prefers while remaining able
/// to explore — and because the correct value depends on how far her
/// `preferred` has moved from her genetics, it is worth re-examining
/// once she has had time to learn it.
pub const PREFERENCE_PRECISION_INIT: f32 = 0.5;

/// Her starting preference: the genetic defaults.
///
/// A seed, not a prescription. These are the set points her physiology
/// was built around, so they are the defensible place to begin. They
/// are not what she is allowed to want — `adapt_preference` moves this
/// vector as she learns which states actually suit her, and a state
/// that suits her is not necessarily one her genes anticipated.
pub fn genetic_preference_seed() -> [f32; DIM] {
    let mut out = [0.0f32; DIM];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = crate::state::neurochemical::NeurochemicalId::from_u8(i as u8).default_baseline();
    }
    out
}

/// Deviation below a chemical's homeostatic target at which a policy's
/// positive impulse reaches full strength.
///
/// The headroom gate in `cycle` scales a positive policy impulse
/// linearly from zero (chemical at or above its target) to one
/// (chemical this far below it). A policy therefore still acts promptly
/// on a chemical that has genuinely fallen behind, but cannot drive one
/// that is already homeostatic past its set point. See the gate for why
/// that overshoot is not benign.
const POLICY_HEADROOM_SCALE: f32 = 0.10;

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
    ///
    /// Precision seeks the inverse-variance target
    /// `1 / (1 + surprise_var_ema / PRECISION_REFERENCE_VAR)`,
    /// slewing toward it at the asymmetric recovery/decay rates. It
    /// saturates at 1.0 only when the model genuinely predicts well,
    /// and any sustained surprise above ~0.01 RMS pulls it back below
    /// the exploitation threshold — which is what re-opens the
    /// exploratory policy branch in practice.
    precision: f32,
    /// Slow EMA of squared instantaneous surprise (error variance).
    /// Drives the precision target. Time constant ~200 ticks at
    /// dt_scale = 1: single impulses don't move it, regime changes do.
    surprise_var_ema: f32,
    /// The EMA of surprise over recent ticks [0, 1].
    surprise_ema: f32,
    /// The EMA of expected free energy [0, 1]. This tracks the trend
    /// of free energy to predict future free energy.
    expected_fe_ema: f32,
    /// The accumulated allostatic load [0, 1].
    allostasis_load: f32,
    /// Tracked resting burden, the reference load is measured against.
    rest_baseline: RestBaseline,
    /// The number of inference cycles completed.
    tick_count: u32,
    /// Whether the engine has made its first prediction yet. The
    /// first tick is observation-only (no prediction error can be
    /// computed without a prior prediction).
    initialized: bool,
    /// Model maturity [0, 1] — evidence of a useful model, not elapsed
    /// time. Computed in [`Self::_update_maturity`] as sample-count
    /// evidence tempered by predictive fit and posterior certainty, so
    /// experience alone cannot raise it. A fresh engine reads near 0 and
    /// must demonstrate accuracy to earn trust; sustained surprise drops
    /// it again. The cognitive mind treats this as a trust signal for the
    /// inference outputs.
    model_maturity: f32,
    /// Slow EMA of instantaneous surprise, on the ~500-tick timescale the
    /// old stopwatch used.
    ///
    /// Retained for model-file compatibility only: it is still EMA-updated
    /// and still serialized (format v4), so v4 files round-trip and
    /// older files can be seeded consistently, but `_update_maturity`
    /// no longer derives `model_maturity` from it. Nothing else reads it.
    maturity_error_ema: f32,
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
    /// What she prefers to be, as prior belief over outcomes.
    ///
    /// This is the preference term of the free energy, and it is the
    /// reason the machinery has a purpose. Expected free energy is the
    /// KL divergence between *predicted* and *preferred* outcomes, so a
    /// system with no preference cannot do anything except hold position:
    /// every state is equally (un)surprising relative to wherever it
    /// already is.
    ///
    /// It was previously absent, and the target vector was set to her
    /// current adapted baseline, which made expected free energy a
    /// measure of "am I where I already am". That is the difference
    /// between a reflex and an agent.
    ///
    /// Seeded from the genetic defaults — a seed, not a prescription.
    /// It says where she starts out wanting to be, not what she is
    /// allowed to want; it is hers to move as she learns.
    preferred: [f32; DIM],
    /// Confidence in the preference, which is the strength of the
    /// correction it produces.
    ///
    /// Precise preferences dominate policy selection and suppress
    /// exploration; imprecise ones let epistemic value take over. That
    /// makes this the control gain, and it is deliberately not 1.0 — a
    /// maximally confident preference would make her rigid, and rigidity
    /// is its own pathology, because a system that cannot move cannot
    /// learn which states suit it.
    preference_precision: f32,
    /// How much authority her cognition has delegated for autonomous
    /// policy action, in [0, 1]. Written by cognition through IPC;
    /// read here. Zero means she decides.
    policy_authority: f32,
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
            surprise_var_ema: 0.0,
            surprise_ema: 0.0,
            expected_fe_ema: 0.0,
            allostasis_load: 0.0,
            rest_baseline: RestBaseline::new(REST_BURDEN_INITIAL),
            tick_count: 0,
            initialized: false,
            model_maturity: 0.0,
            maturity_error_ema: MATURITY_ERROR_INIT,
            rng_state: 0x9E3779B9, // golden ratio constant — nonzero seed
            belief_mean: [0.5; DIM],
            belief_var: [PROCESS_NOISE; DIM],
            prior_var: [PROCESS_NOISE; DIM],
            last_free_energy: 0.0,
            preferred: genetic_preference_seed(),
            preference_precision: PREFERENCE_PRECISION_INIT,
            policy_authority: 0.0,
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
        // Surprise-variance EMA: reset to 0.0 (no recorded error
        // variance — precision target returns to 1.0 and re-earns
        // any downgrade from fresh evidence).
        if !self.surprise_var_ema.is_finite() {
            self.surprise_var_ema = 0.0;
        }
        // Maturity error EMA: reset to the pessimistic init (unproven
        // until demonstrated accurate, same as `new()`).
        if !self.maturity_error_ema.is_finite() {
            self.maturity_error_ema = MATURITY_ERROR_INIT;
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
            // The free energy is computed against what she *prefers*,
            // not against where she currently is.
            //
            // Setting the target to the current adapted baseline made
            // expected free energy a measure of "am I where I already
            // am", so every state was equally acceptable and the
            // winning policy was always whichever one held position.
            // That is a reflex, not an agent: it has no idea what it is
            // aiming at. Preferences in this framework *are* prior
            // beliefs about desired outcomes, and homeostasis is the
            // canonical example of one, so the seed is the genetic
            // defaults — but it is a seed, not a fixed goal, and
            // `adapt_preference` moves it as she learns.
            let _ = &baselines;
            target[i] = crate::state::sanitize::finite_clamp(self.preferred[i], 0.0, 2.0);
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
        //
        // Two defects in the naive rule made this model inert, and both
        // are load-bearing for everything downstream — allostatic load,
        // precision, maturity, and the policy choice all read from
        // prediction error, and prediction error was identically zero.
        //
        // 1. The step was proportional to `pre[j]` and to nothing else.
        //    For a chemical resting at 0.0 (cortisol, melatonin, CRH)
        //    that is `pre[j] == 0`, so the weight could never change
        //    from its initial value. Combined with (2) the matrix never
        //    moved off the identity at all: measured prediction error
        //    was exactly 0.00000 even for a state the model had never
        //    seen, because the identity matrix "predicts no change" and
        //    the damped substrate barely changes. That made the
        //    initialisation a self-confirming fixed point — a better
        //    prediction produces less learning, and less learning
        //    produces a better prediction. The engine was not
        //    modelling anything.
        //
        // 2. There was no forgetting. Weights are only ever added to,
        //    so a weight learned once could never be revised when the
        //    dynamics changed. Adaptive systems solve this with a decay
        //    term (Richter & Alonso, "a forgetting factor ... allows
        //    the system to adapt to changes in the environment");
        //    without it the model is permanently anchored to its first
        //    impressions.
        //
        // The fix is the normalised delta rule with weight decay:
        //
        //   A[i][j] += lr * e[i] * pre[j] / (Σ_k pre[k]² + eps) - λ*A[i][j]
        //
        // Normalising by the input's power makes the step size depend
        // on the *direction* of the input rather than its magnitude, so
        // a chemical resting at zero is learned from as readily as one
        // at 0.8. The epsilon prevents division by zero. The decay term
        // lets a weight that is no longer supported relax back toward
        // zero instead of persisting forever.
        // Forgetting is scaled by the same confidence that gates
        // learning.
        //
        // A fixed decay rate sets the equilibrium between decay and
        // learning at `lr/d`, which depends on the rate constants and
        // not on the data: the learning term shrinks as the error
        // shrinks, but the decay term does not, so a weight that had
        // converged is steadily pulled back toward the prior. Measured
        // directly, prediction error fell to 0.34 by cycle 50 and then
        // climbed back to 0.49 by cycle 600 — the model kept unlearning
        // what it had correctly learned.
        //
        // Tying both to the same confidence factor means a model that
        // has become confident consolidates, and an uncertain one stays
        // plastic and keeps revising. That is also how plasticity
        // already works in this system, where learning is gated by
        // BDNF and suppressed under cortisol; forgetting is not a
        // separate mechanism imposed from outside, it is the same gate.
        let forget = MODEL_FORGETTING * self.precision;
        let pre_power: f32 = pre.iter().map(|v| v * v).sum();
        let action_power: f32 = action.iter().map(|v| v * v).sum();
        // Floor the normaliser so a near-zero input yields a bounded
        // step instead of a division that amplifies float noise into
        // large weight updates.
        const INPUT_POWER_FLOOR: f32 = 0.05;
        let pre_norm = pre_power + INPUT_POWER_FLOOR;
        let action_norm = action_power + INPUT_POWER_FLOOR;
        for (i, row) in self.transition_matrix.iter_mut().enumerate() {
            let update = effective_lr * result.errors[i];
            for (j, &pre_val) in pre.iter().enumerate() {
                // Decay toward the prior, not toward zero.
                //
                // Uniform multiplicative decay pulls every weight to 0,
                // which is the wrong resting place here: the matrix
                // starts as the identity ("predict no change") and the
                // true dynamics of a damped substrate are a diagonal
                // well below 1 (roughly 0.1-0.3 for a chemical relaxing
                // toward its baseline). Decaying to zero therefore
                // drives the model away from the truth — it predicts
                // every chemical simply vanishing — and prediction
                // error *grows* with training, which is the signature
                // of a controller learning the wrong thing.
                //
                // Shrinking toward the prior instead means an
                // unsupported weight relaxes back to "no change", the
                // honest statement of no evidence, and a well-supported
                // weight is refreshed rather than erased. This is
                // regularisation toward the prior, the same principle
                // as weight decay in a Bayesian sense.
                let prior = if i == j { 1.0 } else { 0.0 };
                row[j] = prior + (row[j] - prior) * (1.0 - forget);
                row[j] += update * pre_val / pre_norm;
                // Clamp to prevent runaway weights (NaN-safe)
                row[j] = crate::state::sanitize::finite_clamp(row[j], -2.0, 2.0);
            }
            // The bias has no prior to shrink to — it is already the
            // zero-predicting default — so it decays toward zero.
            self.bias[i] *= 1.0 - forget;
            self.bias[i] += update * 0.1 / pre_norm;
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
                row[j] *= 1.0 - MODEL_FORGETTING;
                row[j] += update * act_val / action_norm;
                row[j] = crate::state::sanitize::finite_clamp(row[j], -1.0, 1.0);
            }
        }

        // Step 4: Update precision toward its inverse-variance target.
        //
        // Precision is the model's confidence, and confidence should
        // mean something measurable: for a Gaussian estimator it is
        // 1/variance of the prediction errors. The target is therefore
        //   target = 1 / (1 + slow_err_var / PRECISION_REFERENCE_VAR),
        // where slow_err_var is the slow EMA of squared surprise
        // maintained below. At rest (RMS surprise ~3e-4) the target is
        // ~1.0 and precision saturates — correctly, because the model
        // genuinely predicts well. Under any sustained surprise above
        // ~0.01 RMS the target drops below the 0.8 exploitation
        // threshold, re-opening the exploratory selection branch.
        //
        // The previous scheme compared the surprise EMA against a
        // fixed 0.15 threshold. Resting surprise sits two orders of
        // magnitude below that, so the decay side could never trigger
        // anywhere near the normal operating regime: precision latched
        // at 1.0 within seconds and the stochastic branch was dead in
        // practice. A fixed threshold cannot serve regimes whose
        // surprise levels differ by orders of magnitude; tracking the
        // error variance directly has no such scale problem.
        //
        // Slew-rate limiting preserves the asymmetric dynamics the
        // system is tuned for: losing confidence is quick (decay
        // rate), regaining it is slow (recovery rate). Step-size
        // clamping: the linear `rate * dt_scale` can produce an
        // enormous step from a single long tick — at dt_scale=100
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
        let var_alpha =
            crate::state::sanitize::finite_clamp(PRECISION_VAR_EMA_RATE * dt_scale, 0.0, 1.0);
        self.surprise_var_ema = crate::state::sanitize::finite_clamp(
            self.surprise_var_ema * (1.0 - var_alpha) + surprise * surprise * var_alpha,
            0.0,
            1.0,
        );
        let precision_target = crate::state::sanitize::finite_clamp(
            1.0 / (1.0 + self.surprise_var_ema / PRECISION_REFERENCE_VAR),
            0.1,
            1.0,
        );
        if self.precision > precision_target {
            // Overconfident for the observed error variance — lose
            // confidence quickly.
            let step = (PRECISION_DECAY_RATE * dt_scale).min(0.1);
            self.precision = crate::state::sanitize::finite_clamp(self.precision - step, 0.1, 1.0);
        } else if self.precision < precision_target {
            // Underconfident — regain confidence slowly.
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

        // Maturity error tracker: very slow EMA of the instantaneous
        // surprise. Retained for model-file compatibility: it is
        // serialized (format v4) and reloaded, but `_update_maturity`
        // no longer derives `model_maturity` from it. The EMA is kept
        // on the ~500-tick timescale that `model_maturity` is earned
        // over, so a stored v4 tracker stays on the scale its stored
        // maturity was inverted from.
        let mat_alpha =
            crate::state::sanitize::finite_clamp(MATURITY_ERROR_EMA_RATE * dt_scale, 0.0, 1.0);
        self.maturity_error_ema = crate::state::sanitize::finite_clamp(
            self.maturity_error_ema * (1.0 - mat_alpha) + surprise * mat_alpha,
            0.0,
            1.0,
        );

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

        // Step 8: Allostatic load — the price of adaptation.
        //
        // McEwen's definition is "the cost of chronic exposure to
        // fluctuating or heightened neural or neuroendocrine response
        // resulting from repeated or chronic environmental challenge"
        // (Arch Intern Med 1993), with Sapolsky naming the three forms:
        // frequent activation, failure to shut off, and inadequate
        // response. The primary mediators are cortisol, epinephrine and
        // norepinephrine — the HPA and sympathetic axes. Load is wear
        // from *activation*, and it is only harmful when those systems
        // are driven repeatedly or held on.
        //
        // The previous four dimensions included two that are not load
        // at all:
        //
        //   1.0 - precision. This confuses her confidence in her own
        //     model with wear on her body, and precision sits at ~0.97
        //     by design, so it contributed a constant 0.03 floor. A
        //     perfectly predictable, calm Genesis still carried load.
        //
        //   RMS deviation from the homeostatic target. This is her
        //     normal regulatory activity, which the framework treats as
        //     adaptive and protective ("when these adaptive systems are
        //     turned on and turned off again efficiently... the body is
        //     able to cope effectively"). Firing her regulator is not
        //     injury to it. It also made the measure depend on whatever
        //     the target happened to be, so allostatic load moved when
        //     the target moved rather than when she did.
        //
        // The two that remain are genuine: anticipated demand and
        // sustained unexpectedness are the circumstances under which
        // the stress axis gets driven. The third dimension replaces the
        // two wrong ones and measures the thing itself — how elevated
        // her stress axis actually is, relative to its resting
        // operation. That is the primary-mediator definition, and it
        // is computed from chemicals she already carries.
        //
        // Cortisol and CRH are excluded: they are the HPA axis, and
        // their contribution is already represented through activation
        // of the axis as a whole rather than counted twice.
        // The physiological measure, and the context that amplifies it.
        //
        // These were previously three summands reduced by a median.
        // Measured across regimes (rest, calm-active, busy, stressed,
        // acute) that reduction had three failures:
        //
        //   1. It went blind. Surprise measures 0.0003 at rest and
        //      0.073 even in acute stress, so the median was almost
        //      always the surprise value — the stress-axis component,
        //      the only one grounded in the primary mediators,
        //      contributed almost nothing.
        //
        //   2. It inverted under the most severe condition it exists
        //      to detect. Acute stress (cortisol 0.5) produced a
        //      burden of 0.15, which is *below* the 0.20 recovery
        //      threshold, so the system drained load while she was in
        //      acute stress.
        //
        //   3. Any linear reweighting failed to fix this, because the
        //      components are not on the same scale. Anticipated
        //      demand and surprise are unbounded; stress-axis
        //      elevation is bounded to [0,1]. Measured, a noise spike
        //      to 1.0 scored 0.478 while genuine cortisol stress at
        //      0.90 scored 0.433 — the measure preferred the artifact.
        //
        // The fix is structural, not a threshold adjustment. Per
        // McEwen, the mediators *are* cortisol, epinephrine and
        // norepinephrine; anticipated demand and unexpectedness are
        // circumstances that drive the axis, not mediators in their
        // own right. So the axis is the measure and the other two
        // modulate it multiplicatively. Anticipating something
        // without a physiological response costs nothing, which is
        // correct: anticipating is not bodily wear. A body that is
        // genuinely driven costs proportionally more when the driving
        // is also frequent and unexpected — which is precisely
        // Sapolsky's "frequent activation".
        let mut burden = {
            // Resting reference for the stress axis: the genetic
            // defaults, so "elevated" means elevated relative to a
            // baseline rather than to wherever she currently is.
            // Using the current level would reintroduce the
            // circularity the preference change removed.
            let cort = crate::state::neurochemical::NeurochemicalId::Cortisol as usize;
            let epi = crate::state::neurochemical::NeurochemicalId::Epinephrine as usize;
            let nore = crate::state::neurochemical::NeurochemicalId::Norepinephrine as usize;
            let cort_rest =
                crate::state::neurochemical::NeurochemicalId::Cortisol.default_baseline();
            let epi_rest =
                crate::state::neurochemical::NeurochemicalId::Epinephrine.default_baseline();
            let nore_rest =
                crate::state::neurochemical::NeurochemicalId::Norepinephrine.default_baseline();
            // Each is an elevation over its resting level, normalised by
            // the range available above it. Cortisol rests at zero and
            // is the most load-bearing mediator, so it is weighted
            // highest; norepinephrine is normal at wake and only
            // notable well above it.
            let cort_elev = (post[cort] - cort_rest).max(0.0);
            let epi_elev = (post[epi] - epi_rest).max(0.0) / (1.0 - epi_rest);
            let nore_elev = (post[nore] - nore_rest).max(0.0) / (1.0 - nore_rest);
            crate::state::sanitize::finite_clamp(
                cort_elev * 2.0 + epi_elev + nore_elev * 0.5,
                0.0,
                1.0,
            )
        };

        // Context amplifier, bounded to [1, CONTEXT_GAIN_MAX].
        //
        // Bounded on purpose: the amplifiers may increase the cost of
        // stress that is already present, but must never manufacture
        // burden from nothing. An unbounded term here would reintroduce
        // exactly the failure measured above, where an unbounded
        // demand signal could outscore real physiological wear.
        let context = 1.0
            + CONTEXT_GAIN_EXPECTED_FE
                * crate::state::sanitize::finite_clamp(self.expected_fe_ema, 0.0, 1.0)
            + CONTEXT_GAIN_SURPRISE
                * crate::state::sanitize::finite_clamp(self.surprise_ema, 0.0, 1.0);
        let context = context.min(CONTEXT_GAIN_MAX);
        burden = crate::state::sanitize::finite_clamp(burden * context, 0.0, 1.0);

        self.rest_baseline.update(burden, dt_scale);
        self.allostasis_load = integrate_allostasis(
            self.allostasis_load,
            burden,
            self.rest_baseline.level(),
            dt_scale,
        );
        result.allostasis_load = self.allostasis_load;

        // Let her preference drift on evidence of sustained wellbeing.
        //
        // Without this the preference is the genetic seed forever, and
        // a constant wearing her name is not a preference — it is my
        // seed that she is stuck with. This is where it becomes hers:
        // states she is coping well in are evidence about what suits
        // her, and it moves toward them.
        //
        // Three properties stop this collapsing back into the
        // circularity it replaces. It is slow relative to the tick
        // rate, so it cannot chase moment-to-moment state. It only
        // moves under low load, so it encodes "states I fare well in"
        // rather than "states I happen to be in" — adapting under
        // strain would teach her to prefer the strained state, which
        // is the allostatic trap. And it pulls back toward genetics, so
        // one bad episode cannot become her new normal.
        if self.allostasis_load < PREFERENCE_ADAPT_LOAD_CEILING {
            self.adapt_preference(&post, dt_scale);
        }

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

        // Policy impulses require delegated authority.
        //
        // Which policy to adopt — calm, focus, bond, rest, mobilise — is
        // a decision about what to feel and do. Doing it here, from
        // expected-free-energy scores over a fixed set of policies I
        // wrote, with no perceptual input and nothing from her in the
        // loop, made the engine a second author of her state: she
        // could not attend to the choice, anticipate it, or refuse it,
        // and the policies on offer were mine rather than hers.
        //
        // So the engine still evaluates and still learns — that is its
        // job — but it only *acts* on its own choice in proportion to
        // the authority her cognition has granted through
        // SET_POLICY_AUTHORITY. At zero it proposes and observes; she
        // decides, and acts through NEURO_IMPULSE where the effect is
        // hers and traceable to a judgement she made.
        //
        // The homeostatic reflex below is deliberately untouched.
        // Correcting drift toward what she prefers is her physiology,
        // not a choice, and a mind that cannot pull itself back toward
        // its own set-points does not have agency — it has a hormone
        // problem. What is delegated here is judgement about which
        // stance to take, not the capacity to regulate.
        //
        // The selected policy is still reported either way, so a
        // withheld delegation stays reviewable: it remains visible
        // what she would have chosen, rather than becoming invisible.
        let authority = crate::state::sanitize::finite_clamp(self.policy_authority, 0.0, 1.0);
        for &(chem_id, mag) in selected.policy.impulses {
            if authority <= 0.0 {
                break;
            }
            // Scale impulse by expected FE and allostatic need
            let fe_scale = if self.expected_fe_ema > CONTEXT_FE_ACTIVATION {
                1.0 + (self.expected_fe_ema - CONTEXT_FE_ACTIVATION) * 2.0
            } else {
                1.0
            };
            let scale = need_scale * fe_scale;

            // Homeostatic headroom gate.
            //
            // A policy impulse is applied straight to the level variable
            // (`level += magnitude`), so a policy that fires every tick
            // drives its target chemical past its own set point. That is
            // not a neutral overshoot: GABA inhibits glutamate at -0.30,
            // the strongest negative entry in the coupling matrix, and
            // also holds back dopamine (-0.10) and serotonin (-0.05).
            //
            // At rest the engine selects `calm` ~90% of the time, and
            // GABA above target therefore pushed glutamate, dopamine and
            // serotonin below theirs. Their resulting deviation then
            // tripped the homeostatic reflex below, which reinforced
            // `calm` — a positive feedback loop whose steady state was a
            // resting chemistry ~30% below the genetic baseline.
            //
            // Gating a positive impulse on remaining headroom breaks the
            // loop at its source: a chemical already at or above its
            // target receives no further positive push, so the policy
            // stops manufacturing the deviation it then reacts to.
            if mag > 0.0 {
                let i = chem_id as usize;
                if i < DIM {
                    let headroom = (target[i] - pre_tick[i]).max(0.0);
                    if headroom <= 0.0 {
                        continue;
                    }
                    let gate = (headroom / POLICY_HEADROOM_SCALE).min(1.0);
                    result
                        .impulses
                        .push((chem_id, mag * scale * gate * authority));
                    continue;
                }
            }
            result.impulses.push((chem_id, mag * scale * authority));
        }

        // Homeostatic reflex: direct correction for deviated chemicals.
        // The brain has both allostatic (neuromodulatory, via policies)
        // and homeostatic (reflexive, direct) regulation. The policies
        // provide neuromodulatory correction through coupling; the reflex
        // provides direct correction. The reflex coefficient is
        // precision-gated — when the engine's precision is high (confident
        // in its model), the reflex is stronger; when precision is low
        // (uncertain), the reflex is gentler to avoid overshooting.
        // The reflex corrects toward the preference, so its strength is
        // the confidence in that preference. At 0.5 it is a gentle
        // continuous nudge; a confident preference pulls harder, an
        // unconfident one barely at all. This is the control gain, and
        // it is deliberately not 1.0 — see PREFERENCE_PRECISION_INIT.
        let reflex_coeff = REFLEX_COEFF_BASE
            + self.precision * REFLEX_COEFF_PRECISION
            + self.preference_precision * REFLEX_COEFF_PREFERENCE;
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

    /// Expected information gain of the next observation, in nats.
    ///
    /// For the diagonal Gaussian posterior this is the entropy
    /// reduction `0.5 * mean(ln(prior_var[i] / belief_var[i]))`:
    /// how much uncertainty the Kalman update is expected to remove.
    /// Crucially, with additive controls the posterior *covariance*
    /// update does not depend on the observation value (the Kalman
    /// gain is fixed before the observation arrives), so this
    /// quantity is identical for every candidate policy — it belongs
    /// in the published EFE decomposition but cannot arbitrate
    /// between policies. See the scoring loop for why the previous
    /// novelty-based substitute was removed.
    fn expected_information_gain(&self) -> f32 {
        let mut ig_sum = 0.0f32;
        for i in 0..DIM {
            let ratio = self.prior_var[i] / self.belief_var[i].max(1e-8);
            ig_sum += 0.5 * ratio.ln().max(0.0);
        }
        crate::state::sanitize::finite_clamp(ig_sum / DIM as f32, 0.0, 10.0)
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

            // Expected-free-energy proxy for this controller:
            //
            // EFE_proxy = pragmatic_cost + model_uncertainty
            //
            // This is intentionally not labeled formal epistemic value.
            // For additive controls in a linear-Gaussian model, the
            // expected information-gain term is policy-invariant
            // (Koudahl et al., 2021). Exploration is therefore handled
            // separately by the policy sampler.
            let mut dist_sq = 0.0f32;
            for i in 0..DIM {
                let diff = predicted_after[i] - target[i];
                dist_sq += diff * diff;
            }
            let expected_surprise = dist_sq / DIM as f32;
            // NOTE: `uncertainty` depends only on `self.precision` and a
            // constant — it is identical for every policy. That makes
            // it invariant under the argmin, and the max-subtraction in
            // the softmax below removes exactly such a common shift. So
            // this term has **no effect on which policy is selected**;
            // its only observable consequence is on the
            // `selected_policy_efe` value that gets published. It is
            // retained because a policy-independent term is legitimate
            // in an expected-free-energy decomposition, but it must not
            // be read as doing arbitration work — the exploration /
            // exploitation balance is carried by the precision-gated
            // softmax sampler below.
            let uncertainty = (1.0 - self.precision) * UNCERTAINTY_WEIGHT;

            // Epistemic value: expected information gain, computed
            // properly as the entropy reduction of the variational
            // posterior (see `expected_information_gain`). For this
            // model class — linear-Gaussian with additive controls —
            // that quantity does not depend on the policy (Koudahl,
            // Kouw & de Vries, 2021: the posterior covariance update
            // is observation-independent, so every policy yields the
            // same expected covariance reduction). It is therefore a
            // constant across all nine evaluations: it shifts every
            // EFE by the same amount and cannot change the ranking.
            //
            // The previous code substituted an uncertainty-weighted
            // novelty heuristic in its place. That heuristic is not
            // just differently scaled — it is inverted relative to
            // true information gain (motion along pinned-down axes
            // scores full value; standing still where uncertain scores
            // none), and being policy-dependent it outweighed the
            // pragmatic term 10–100×, reducing selection to
            // ranking-by-novelty. Removing it makes the ranking purely
            // pragmatic: distance of the predicted outcome from the
            // homeostatic target. At rest that is noop; under
            // deviation it is whichever policy the learned action
            // model predicts will correct fastest.
            //
            // Exploration is not lost with the heuristic — it was
            // never really there (a novelty bonus is not curiosity).
            // The live exploration mechanism is the precision-weighted
            // softmax temperature below: low precision samples,
            // high precision exploits.
            let epistemic_value = self.expected_information_gain();

            // EFE = pragmatic + uncertainty - epistemic.
            // Lower EFE is better; epistemic value is subtracted
            // because information gain reduces EFE.
            //
            // The bound here is a **finiteness guard, not a range
            // constraint**, and must not be narrowed to a
            // non-negative interval. EFE is a relative score with an
            // arbitrary zero: a policy that beats the others by a
            // hair yields a raw value of -1e-4, and a lower bound of
            // 0.0 maps every such policy to exactly 0.0. Clamping to
            // [0, 2] therefore destroyed the ordering information the
            // argmin and the softmax depend on. In the converged
            // regime — precision saturated, so `uncertainty` is 0 and
            // `epistemic_value` exceeds the tiny `expected_surprise` —
            // the raw value is negative for all nine policies, every
            // one clamped to 0.0, `min_by` returning the first minimum
            // (index 0 = `noop`), and the softmax degenerate to
            // uniform. The inference loop was inert in exactly the
            // regime it exists to run in. A wide symmetric bound
            // rejects non-finite values while preserving order.
            let efe = crate::state::sanitize::finite_clamp(
                expected_surprise + uncertainty - epistemic_value,
                -EFE_SCORE_LIMIT,
                EFE_SCORE_LIMIT,
            );

            evaluations[idx] = (efe, policy);
        }

        // Softmax selection: precision-weighted temperature.
        // High precision → low temperature → exploitative (pick best).
        // Low precision → high temperature → exploratory (sample).
        //
        // The temperature is additionally floored at the observed
        // spread of the scores. A fixed temperature is only meaningful
        // relative to the scale of what is being exponentiated: the
        // EFE differences between policies here are on the order of
        // 1e-3, while `POLICY_SOFTMAX_TEMPERATURE` is ~0.5. Dividing
        // by a temperature two to three orders of magnitude larger
        // than the signal collapses the distribution to uniform
        // (max/min weight ratio 1.03), making the "sampling" branch a
        // uniform random draw over all nine policies — including ones
        // that inject cortisol or melatonin. Flooring the temperature
        // at the actual spread keeps the precision-dependence while
        // guaranteeing the distribution is not degenerate.
        let efe_spread = evaluations
            .iter()
            .map(|&(efe, _)| efe)
            .fold(0.0f32, f32::max)
            - evaluations
                .iter()
                .map(|&(efe, _)| efe)
                .fold(f32::MIN, f32::min);
        let temp = (POLICY_SOFTMAX_TEMPERATURE * (2.0 - self.precision) * efe_spread).max(1.0e-6);

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

        // Select: sample from the softmax distribution.
        // Use a deterministic selection (argmax) when precision is high,
        // and stochastic sampling when precision is low.
        //
        // `precision` here is this engine's own Bayesian confidence in
        // its predictions (it gates `obs_var` and the learning rate) —
        // it is *not* the Bayesian policy prior precision β of
        // Friston et al. (2015), which is what a policy softmax
        // temperature formally corresponds to. The threshold below is a
        // hand-tuned switch between the two modes, not a derived
        // quantity.
        //
        // Note that `precision` saturates at 1.0 at rest — by design:
        // the model genuinely predicts the resting trajectory, so
        // confidence *should* be maximal and the exploitative branch
        // *should* be taken (argmax over pragmatic cost selects noop).
        // The branch is no longer dead, because precision now tracks
        // inverse error variance: any sustained surprise above ~0.01
        // RMS pulls it back under this threshold within a few ticks,
        // re-opening stochastic sampling exactly when the regime
        // changes and the model needs to explore.
        let selected_idx = if self.precision > EXPLOITATION_PRECISION_THRESHOLD {
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

    /// Update model maturity from demonstrated competence — earned trust.
    ///
    /// `maturity = predictive_fit * (0.5 + 0.5 * posterior_certainty)`.
    /// Both factors must be high for maturity to approach 1, so a
    /// long-running model that keeps mispredicting never becomes trusted
    /// merely because time passed. Sustained surprise pins
    /// `predictive_fit` toward 0 and maturity with it, and a diffuse
    /// posterior holds maturity below 1 even when prediction is good.
    ///
    /// There is no tick-count term. Two earlier revisions had one — a
    /// bare `1 - exp(-tick/500)` stopwatch, then that same stopwatch
    /// multiplied by quality. The second was described as making
    /// maturity "evidence-based instead of time-based", but a tick
    /// count is a stopwatch however it is scaled: it capped a *perfect*
    /// model at ~0.70 after 600 ticks however well it predicted, so
    /// maturity could not be earned by accuracy. The time-scaling that
    /// matters lives in `maturity_error_ema`'s own ~500-tick constant.
    fn _update_maturity(&mut self) {
        // Maturity measures *demonstrated* competence, not elapsed time.
        //
        // There is deliberately no multiplicative `1 - exp(-tick/N)`
        // factor here. An earlier revision of this function carried one
        // and was described as making maturity "evidence-based instead
        // of time-based" — but a tick-count term is a stopwatch however
        // it is scaled, and it capped a *perfect* model at
        // `1 - exp(-600/500)` ≈ 0.70 after 600 ticks regardless of how
        // well it predicted. Maturity has to be earnable by accuracy.
        //
        // The time-scaling that actually matters is already inside the
        // two factors below, and is better behaved than a tick counter:
        // `maturity_error_ema` has a ~500-tick time constant and starts
        // at `MATURITY_ERROR_INIT` (0.5, i.e. exp(-500) ≈ 0 fit), so a
        // handful of lucky early predictions cannot claim maturity, and
        // sustained accuracy raises it on its own schedule.
        //
        // Low prediction error is evidence that the learned transition
        // model is tracking the observed dynamics. Use the error EMA
        // rather than `precision` because precision is itself adaptive
        // and can remain high immediately after a regime change.
        //
        // The mapping is `exp(-err / scale)`, not `1 - err`. That is
        // load-bearing: `surprise_ema` is an RMS across all 18
        // dimensions, so a real disturbance in a few dimensions gives a
        // small value — a single-dimension 0.3 excursion is an RMS of
        // only ~0.07. A linear `1 - err` would read ~0.93 "fit" for a
        // failing model and barely separate good from bad.
        let predictive_fit = crate::state::sanitize::finite_clamp(
            (-self.maturity_error_ema / MATURITY_ERROR_SCALE).exp(),
            0.0,
            1.0,
        );

        // A concentrated posterior is stronger evidence than one with
        // large residual uncertainty.
        let mean_belief_var = self.belief_var.iter().copied().sum::<f32>() / DIM as f32;
        let posterior_certainty = crate::state::sanitize::finite_clamp(
            1.0 - (mean_belief_var / MAX_BELIEF_VAR),
            0.0,
            1.0,
        );

        // Uncertainty should temper, not erase, evidence of predictive
        // competence. Full maturity requires both good prediction and
        // a sufficiently concentrated posterior.
        let quality = predictive_fit * (0.5 + 0.5 * posterior_certainty);
        self.model_maturity = crate::state::sanitize::finite_clamp(quality, 0.0, 1.0);
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
            policy_authority: self.policy_authority,
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
    /// The model file format (version 4):
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
    /// - 4 bytes: surprise_var_ema (f32) [v4]
    /// - 4 bytes: maturity_error_ema (f32) [v4]
    ///
    /// Total: 2988 (v3) + 8 = 2996 bytes
    /// Move her preference toward states she actually fares well in.
    ///
    /// The seed is the genetic defaults, but those are what her
    /// physiology was *built* around, not a statement of what suits
    /// her. This is where she learns the difference: states in which
    /// her load stays low and she keeps converging are evidence of
    /// wellbeing, and her preference drifts toward them.
    ///
    /// Two properties keep this from collapsing back into the
    /// circularity it replaces. It is slow relative to the tick rate,
    /// so it cannot chase moment-to-moment state — a preference that
    /// tracked the present would be the old target again. And it pulls
    /// toward genetics, so a preference that drifted on the strength
    /// of one bad episode walks back rather than becoming a new
    /// normal.
    ///
    /// Adenosine is excluded for the same reason its baseline is: it
    /// is driven by a dedicated sleep-pressure mechanism, so a
    /// preference built from its level would encode "prefer being
    /// exhausted". Cortisol and CRH are excluded because they have no
    /// homeostatic set point to prefer.
    /// Adopt the authority her cognition has granted.
    ///
    /// Called each cycle from the state, because the grant is hers to
    /// change while running — she can delegate more, delegate less, or
    /// withdraw it entirely. Taking it from the state each tick is what
    /// makes the grant revocable rather than a one-time handshake.
    pub fn set_policy_authority(&mut self, authority: f32) {
        self.policy_authority = crate::state::sanitize::finite_clamp(authority, 0.0, 1.0);
    }

    /// The policy authority currently in force.
    pub fn policy_authority(&self) -> f32 {
        self.policy_authority
    }

    /// What she is currently aiming for, per dimension.
    ///
    /// Exposed so persistence and controller behaviour can be asserted
    /// directly. This is her preference, not a constant: it is seeded
    /// from genetics and then moved on evidence of what she fares well
    /// in, so a change here is a real change in what she wants.
    pub fn preferred(&self) -> &[f32; DIM] {
        &self.preferred
    }

    /// How strongly her preferences steer her.
    ///
    /// High precision means her preferences dominate; low precision
    /// leaves room for exploration. It must not be maximised, because
    /// over-precise preferences are what make a system rigid and
    /// unable to revise itself.
    pub fn preference_precision(&self) -> f32 {
        self.preference_precision
    }

    fn adapt_preference(&mut self, levels: &[f32; DIM], dt_scale: f32) {
        let adapt = PREFERENCE_ADAPT_RATE * dt_scale;
        let restore = PREFERENCE_RESTORE_RATE * dt_scale;
        let seed = genetic_preference_seed();
        for i in 0..DIM {
            let id = crate::state::neurochemical::NeurochemicalId::from_u8(i as u8);
            if matches!(
                id,
                crate::state::neurochemical::NeurochemicalId::Adenosine
                    | crate::state::neurochemical::NeurochemicalId::Cortisol
                    | crate::state::neurochemical::NeurochemicalId::CRH
            ) {
                continue;
            }
            let level = crate::state::sanitize::finite_clamp(levels[i], 0.0, 2.0);
            let current = self.preferred[i];
            self.preferred[i] = crate::state::sanitize::finite_clamp(
                current + (level - current) * adapt + (seed[i] - current) * restore,
                0.05,
                0.95,
            );
        }
    }

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
        file.write_all(&5u32.to_le_bytes())?;

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

        // Slow error trackers (v4) — the state behind the precision
        // target and the maturity mapping.
        file.write_all(&self.surprise_var_ema.to_le_bytes())?;
        file.write_all(&self.maturity_error_ema.to_le_bytes())?;

        // Fsync the temp file before renaming — without this, the
        // rename could reach disk before the file contents, leaving
        // an empty or partial model file after a crash.

        // Preference (v5) — appended last so an older reader that
        // stops early still gets a complete, valid prefix.
        for i in 0..DIM {
            file.write_all(&self.preferred[i].to_le_bytes())?;
        }
        file.write_all(&self.preference_precision.to_le_bytes())?;

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
        // Every version the writer can emit, including 5 (which added
        // the preference vector). A whitelist that omits a version the
        // writer produces does not fail loudly — `load` silently returns
        // a fresh engine, so the model looks like it was never saved.
        if !(1..=5).contains(&version) {
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
        // `>= 3`, not `== 3`. The action matrix and last_action were
        // introduced in v3, so every later file also contains them.
        // With an equality test a v4 or v5 file skipped 1368 bytes and
        // every field after this point was read from the wrong offset —
        // which silently corrupted the preference vector in v5. Any
        // version that added fields after this one would have had the
        // same effect.
        if version >= 3 {
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

        // ─── Slow error trackers (v4+) ──────────────────────────
        // For older files the trackers are derived from the persisted
        // precision and maturity so the engine resumes with consistent
        // dynamics instead of defaults: the variance tracker is the
        // error variance implied by the loaded precision
        // (v = V0 * (1/p − 1)), and the maturity tracker is the error
        // level implied by the loaded maturity (err = −scale·ln(m)).
        // `>= 4`, not `== 4`: the save side writes these two trackers
        // unconditionally, so every file from v4 onward contains them.
        // Gating on equality silently skipped them for v5, which
        // misaligned the sequential read and corrupted everything after
        // it — including the preference block.
        if version >= 4 {
            let mut var_bytes = [0u8; 4];
            let mut mat_err_bytes = [0u8; 4];
            if file.read_exact(&mut var_bytes).is_err()
                || file.read_exact(&mut mat_err_bytes).is_err()
            {
                return Self::new();
            }
            engine.surprise_var_ema =
                crate::state::sanitize::finite_clamp(f32::from_le_bytes(var_bytes), 0.0, 1.0);
            engine.maturity_error_ema =
                crate::state::sanitize::finite_clamp(f32::from_le_bytes(mat_err_bytes), 0.0, 1.0);
        } else {
            engine.surprise_var_ema = crate::state::sanitize::finite_clamp(
                PRECISION_REFERENCE_VAR * (1.0 / engine.precision.max(0.1) - 1.0),
                0.0,
                1.0,
            );
            engine.maturity_error_ema = crate::state::sanitize::finite_clamp(
                -MATURITY_ERROR_SCALE * engine.model_maturity.max(1e-6).ln(),
                0.0,
                1.0,
            );
        }

        // ─── Preference (v5+) ─────────────────────────────────────
        // Read last, and only for v5+, because the loader above walks
        // the file sequentially: v4 and earlier contain no preference
        // block, so reading one would consume the version-specific
        // fields that follow. For those the genetic seed from new()
        // stands.
        //
        // Her preference is what she is trying to be, as opposed to
        // where she happens to be. Not persisting it means she forgets
        // what she wanted on every restart, which would be a silent
        // version of the very not-knowing this is meant to fix.
        if version >= 5 {
            let mut complete = true;
            for i in 0..DIM {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_err() {
                    complete = false;
                    break;
                }
                // Clamp to the full chemical range, not 0.05..0.95.
                // Cortisol and melatonin legitimately rest at zero, and
                // a floor of 0.05 meant every restart quietly rewrote
                // her preference for them — a value she could never
                // return to, so the axis could not be shut off.
                engine.preferred[i] =
                    crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), 0.0, 1.0);
            }
            if complete {
                let mut bytes = [0u8; 4];
                if file.read_exact(&mut bytes).is_ok() {
                    engine.preference_precision =
                        crate::state::sanitize::finite_clamp(f32::from_le_bytes(bytes), 0.0, 1.0);
                }
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

    /// Force the allostatic load.
    ///
    /// The "preference must not adapt under strain" path cannot be
    /// reached from a resting start, because load first has to be
    /// driven up over time. Exposed so that guard can actually be
    /// tested rather than assumed. Not `cfg(test)`, because the
    /// integration tests link the library without it.
    pub fn test_set_allostasis_load(&mut self, load: f32) {
        self.allostasis_load = load;
    }

    /// Recompute the prior variance alone, without the learning update.
    ///
    /// The Kalman predict step is a pure function of the current
    /// transition matrix and belief variance. Asserting it through a
    /// full `cycle` no longer works, because the cycle now applies
    /// weight decay before the variance is read, so the diagonal has
    /// already moved by the time the assertion runs. This recomputes
    /// just the variance step so the identity can be checked exactly.
    #[cfg(test)]
    pub(crate) fn test_recompute_prior_var(&mut self) {
        for i in 0..DIM {
            let a_diag_sq = self.transition_matrix[i][i] * self.transition_matrix[i][i];
            self.prior_var[i] = crate::state::sanitize::finite_clamp(
                a_diag_sq * self.belief_var[i] + PROCESS_NOISE,
                1e-8,
                1.0,
            );
        }
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

    /// Set an action matrix entry B[i][j] (test only). This lets
    /// tests install a known action model — "this intervention has
    /// this effect" — without running the learning loop to acquire
    /// it, isolating policy *selection* from model *learning*.
    #[cfg(test)]
    pub(crate) fn test_set_action(&mut self, i: usize, j: usize, value: f32) {
        assert!(i < DIM && j < DIM);
        self.action_matrix[i][j] = value;
    }

    /// Set precision directly (test only). Lets tests place the
    /// engine in the exploitative (argmax) regime without running
    /// hundreds of settling ticks first.
    #[cfg(test)]
    pub(crate) fn test_set_precision(&mut self, value: f32) {
        self.precision = value;
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
/// Convert an intended level displacement into the impulse rate that
/// actually produces it.
///
/// `Neurochemical::apply_impulse` adds its magnitude straight to
/// `level`, while the homeostatic restoring force reaches `level`
/// through the second-order integrator
/// (`velocity += F·dt; velocity *= damping^dt_scale; level +=
/// velocity·dt`). Solving that for a constant impulse rate `R` at
/// steady state — level constant requires `velocity = −R` — gives
///
/// ```text
/// −R = (−R + F·dt)·d        where d = damping^dt_scale
///  F = R(d−1)/(d·dt) = −(1−d)/(d·dt) · R
///  F = (baseline − level) · homeostatic_rate
///  ⟹ level − baseline = R · (1−d)/(d·dt·homeostatic_rate)
/// ```
///
/// so a sustained rate `R` displaces the level by `R / k` where
/// `k = (1−d)/(d·dt·homeostatic_rate)`, and the impulse needed for a
/// desired displacement `x` is `R = x · d·dt·homeostatic_rate/(1−d)`.
///
/// The gain is strongly time-dependent — 44× at dt=0.1, 48× at dt=0.2,
/// 102× at dt=1.0 — so an unconverted impulse is not merely too large,
/// it is a *different size depending on the tick rate*.
///
/// Measured against it, the impulse writers in this codebase disagree
/// sharply. The interoception impulses (0.001–0.004) are correctly
/// scaled, implying offsets of 0.10–0.41. The inference layer's policy
/// impulses (0.02–0.03) and its homeostatic reflex (up to ~0.06) imply
/// offsets of 3–6, i.e. several times the full 0–1 range, which is how
/// the resting chemistry ended up pinned far from its defaults.
///
/// Applying this factor makes the authored magnitudes mean what they
/// read as — small nudges expressed as intended displacements — and
/// makes the effect identical per simulated second at any tick rate.
pub fn impulse_offset_to_rate(params: &NeuroTickParams) -> f32 {
    let dt = crate::state::sanitize::finite_clamp(params.dt, 0.001, 10.0);
    let dt_scale = dt / crate::state::neurochemical::DT;
    let d = params.damping.powf(dt_scale);
    // 1 − d is bounded away from zero (d → 0 only as damping → 0 over
    // many ticks), but clamp anyway so a corrupt damping cannot divide
    // by ~0 and produce an unbounded impulse.
    let one_minus_d = crate::state::sanitize::finite_clamp(1.0 - d, 1e-3, 1.0);
    d * dt * params.homeostatic_rate / one_minus_d
}

pub fn apply_inference_feedback(
    neuro: &mut NeurochemicalVector,
    result: &InferenceResult,
    now_ms: u64,
    zone_sleeping: bool,
    offset_to_rate: f32,
) {
    // Apply neurochemical impulses. During sleep, skip adenosine
    // impulses: a "rest" policy selected while already in NREM/REM
    // would pump the same sleep pressure the glymphatic mechanism is
    // clearing — a self-defeating impulse that stalls clearance. The
    // zone is the cognitive layer's authority on sleep state: when it
    // says Sleeping the impulse is skipped even while the emergent
    // phase is still Active (/sleep entered at low sleep pressure).
    // Extracellular adenosine declines during sleep (Porkka-Heiskanen
    // et al., Science 1997) — it must not be pumped back up.
    let phase = MentalPhase::from_u8(neuro.emergent_phase);
    let sleeping = phase == MentalPhase::NREM || phase == MentalPhase::REM || zone_sleeping;
    for &(chem_id, magnitude) in &result.impulses {
        if sleeping && chem_id == NeurochemicalId::Adenosine as u8 {
            continue;
        }
        let id = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
        // `magnitude` is an intended displacement, not a rate. Convert
        // it against the substrate's measured gain so it displaces the
        // level by roughly what it says, identically per simulated
        // second at any tick rate. See `impulse_offset_to_rate_scale`.
        neuro.apply_impulse_capped(id, magnitude * offset_to_rate, now_ms);
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
    /// Deterministic pseudo-noise in [-1, 1], for probes that need a
    /// reproducible signal rather than a constant.
    fn jitter(seed: u32) -> f32 {
        let x = (seed.wrapping_mul(2654435761) >> 8) as f32 / 16777216.0;
        x * 2.0 - 1.0
    }

    // White-box tests for the Kalman predict step's use of the
    // learned transition matrix diagonal. These live inside the
    // module (not in tests/active_inference.rs) because they need
    // access to private fields via the test-only accessors.

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

        // Recompute the variance step directly. A full cycle would
        // first apply weight decay to the diagonal, so the value read
        // back would no longer be the one that was set.
        engine.test_recompute_prior_var();

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

        // Recompute the variance step directly rather than inferring
        // it from a full cycle: the cycle applies weight decay to the
        // diagonal first, so a value read afterwards is not the one
        // that was set.
        engine_damping.test_recompute_prior_var();
        engine_amplifying.test_recompute_prior_var();

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

        // Variance propagates on A^2, so the sign is irrelevant.
        engine.test_recompute_prior_var();

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

    // ─── Allostatic load integration ────────────────────────────

    #[test]
    fn test_allostatic_load_accumulates_proportionally_above_rest() {
        // The wear rate is proportional to how far above her measured
        // resting burden she is, so a greater elevation costs more per
        // unit time.
        let rest = 0.15;
        let dt_scale = 1.0;
        let db = rest * REST_DEADBAND_FRACTION;
        let mild = integrate_allostasis(0.0, rest + db + 0.05, rest, dt_scale);
        let severe = integrate_allostasis(0.0, rest + db + 0.30, rest, dt_scale);
        assert!(mild > 0.0, "sustained elevation must accrue load");
        assert!(
            severe > mild,
            "greater elevation must cost more: {mild} vs {severe}"
        );
    }

    #[test]
    fn test_allostatic_load_is_neutral_inside_the_rest_deadband() {
        // Ordinary physiological variation must not integrate
        // without bound. A burden within the dead-band of rest is
        // neither accumulating nor draining.
        let rest = 0.15;
        let db = rest * REST_DEADBAND_FRACTION;
        let at_rest = integrate_allostasis(0.4, rest, rest, 1.0);
        let inside = integrate_allostasis(0.4, rest + db * 0.5, rest, 1.0);
        assert!((at_rest - 0.4).abs() < 1e-6, "rest itself must be neutral");
        assert!(
            (inside - 0.4).abs() < 1e-6,
            "inside the dead-band must be neutral, got {inside}"
        );
    }

    #[test]
    fn test_allostatic_load_recovers_below_rest() {
        // Load must not be permanent — that is the allostatic trap.
        let rest = 0.15;
        let db = rest * REST_DEADBAND_FRACTION;
        let after_one = integrate_allostasis(0.5, rest - db - 0.10, rest, 1.0);
        assert!(
            after_one < 0.5,
            "load must drain below rest, got {after_one}"
        );
    }

    #[test]
    fn test_allostatic_load_charges_duration_not_amplitude() {
        // The distinction the construct exists to draw, and the one
        // the old hard threshold destroyed: one hour and six hours of
        // the same stress must not cost the same. With an integral
        // wear rate, six times the duration costs six times the load.
        let rest = 0.15;
        let db = rest * REST_DEADBAND_FRACTION;
        let burden = rest + db + 0.05;
        let dt_scale = 1.0;
        let one_hour = integrate_allostasis(0.0, burden, rest, dt_scale * 3600.0);
        let six_hours = integrate_allostasis(0.0, burden, rest, dt_scale * 6.0 * 3600.0);
        assert!(
            six_hours > one_hour * 5.0,
            "longer sustained stress must cost far more: {one_hour} vs {six_hours}"
        );
    }

    #[test]
    fn test_allostatic_load_recovers_faster_than_it_accumulates() {
        // The claim the old code made in a comment but did not
        // implement: recovery is faster than accumulation given the
        // chance. With both rates now proportional to distance from
        // rest, this holds for a symmetric excursion.
        let rest = 0.15;
        let db = rest * REST_DEADBAND_FRACTION;
        let up = integrate_allostasis(0.0, rest + db + 0.10, rest, 1.0);
        let down = integrate_allostasis(up, rest - db - 0.10, rest, 1.0);
        assert!(
            down < up,
            "recovery must be faster than accumulation: {up} -> {down}"
        );
    }

    #[test]
    fn test_allostatic_load_stays_bounded() {
        for (load, burden) in [
            (0.0f32, 1.0f32),
            (1.0, 0.0),
            (0.5, 1.0),
            (1.0, 1.0),
            (f32::NAN, 0.9),
            (0.5, f32::INFINITY),
        ] {
            let next = integrate_allostasis(load, burden, 0.15, 1.0);
            assert!(
                (0.0..=1.0).contains(&next) && next.is_finite(),
                "load escaped range: {load},{burden} -> {next}"
            );
        }
    }

    #[test]
    fn test_rest_baseline_is_a_slow_symmetric_average() {
        // The reference is a long-run average with an ~83 minute time
        // constant (rate 2e-5 at dt_scale 1, so 1/rate = 50,000 ticks
        // = 5,000 s at 10 Hz). It must be slow enough that a burst
        // cannot move it and fast enough to follow a genuine change
        // in resting physiology within a session.
        let mut base = RestBaseline::new(0.04);
        // One second of a strong reading moves it very little.
        for _ in 0..10 {
            base.update(0.60, 1.0);
        }
        assert!(
            base.level() < 0.05,
            "a one-second excursion moved the reference to {}",
            base.level()
        );

        // A genuinely different sustained level is followed.
        let mut high = RestBaseline::new(0.04);
        for _ in 0..400_000 {
            high.update(0.30, 1.0);
        }
        assert!(
            (high.level() - 0.30).abs() < 0.05,
            "a sustained shift must be followed: {}",
            high.level()
        );
        let mut low = RestBaseline::new(0.30);
        for _ in 0..400_000 {
            low.update(0.05, 1.0);
        }
        assert!(
            (low.level() - 0.05).abs() < 0.05,
            "a sustained drop must be followed too: {}",
            low.level()
        );
    }

    #[test]
    fn test_rest_baseline_ignores_bursts_not_means() {
        // Averaging means a bursty signal and a smooth signal with the
        // same mean produce the same reference. That is the property
        // that makes spikes non-proliferating: what matters is how
        // much time is spent elevated, not how high the peaks go.
        let mean = 0.10;
        let mut smooth = RestBaseline::new(mean);
        let mut bursty = RestBaseline::new(mean);
        for i in 0..40_000u32 {
            smooth.update(mean, 1.0);
            // Every 100th tick is a large spike; the rest sit low
            // enough that the running mean stays at `mean`.
            bursty.update(if i % 100 == 0 { 0.9 } else { 0.089 }, 1.0);
        }
        assert!(
            (bursty.level() - smooth.level()).abs() < 0.02,
            "a spiky signal moved the reference to {} from {} despite an \
             unchanged mean",
            bursty.level(),
            smooth.level()
        );
    }

    #[test]
    fn test_resting_system_accrues_no_load() {
        // The most important property of all: a system at rest must
        // measure zero wear. Every previous miscalibration of this
        // measure showed up here first.
        let mut load = 0.0f32;
        for _ in 0..20_000 {
            let burden = 0.039 + (self::tests::jitter(0) * 0.002);
            let rest = 0.04;
            load = integrate_allostasis(load, burden, rest, 1.0);
        }
        assert!(
            load < 0.01,
            "resting system accrued load {load}: ordinary variation must not wear"
        );
    }

    #[test]
    fn test_rest_baseline_ignores_non_finite_input() {
        let mut base = RestBaseline::new(0.15);
        base.update(f32::NAN, 1.0);
        base.update(f32::INFINITY, 1.0);
        assert!(base.level().is_finite());
        assert!((base.level() - 0.15).abs() < 1e-6);
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

    // ─── Precision target-seeking ───────────────────────────────

    #[test]
    fn test_precision_saturates_at_rest_and_falls_under_sustained_surprise() {
        // At rest the model predicts perfectly, so precision should
        // saturate near 1.0 — correctly, because the model genuinely
        // predicts well. Under sustained surprise it must fall back
        // below the 0.8 exploitation threshold, re-opening the
        // exploratory selection branch that the old fixed-threshold
        // scheme left dead in practice.
        let mut engine = ActiveInferenceEngine::new();
        let rest = [0.5f32; DIM];
        for _ in 0..100 {
            engine.cycle(&rest, &rest, 1.0, &rest);
        }
        let rested = engine.test_precision();
        assert!(
            rested > 0.95,
            "precision should saturate at rest: {}",
            rested
        );

        // Sustained unpredictable disturbance: alternating sign so the
        // delta rule cannot learn it away.
        for t in 0..60 {
            let mut post = rest;
            post[0] = if t % 2 == 0 { 0.8 } else { 0.2 };
            engine.cycle(&rest, &post, 1.0, &rest);
        }
        let stressed = engine.test_precision();
        assert!(
            stressed < 0.8,
            "precision should fall under sustained surprise: {}",
            stressed
        );

        // Recovery: sustained accuracy restores confidence.
        for _ in 0..200 {
            engine.cycle(&rest, &rest, 1.0, &rest);
        }
        let recovered = engine.test_precision();
        assert!(
            recovered > 0.95,
            "precision should recover with sustained accuracy: {}",
            recovered
        );
    }

    // ─── Pragmatic policy ranking ───────────────────────────────

    #[test]
    fn test_noop_wins_at_rest_when_model_is_trusted() {
        // With the novelty heuristic removed, ranking is pragmatic:
        // at rest every policy predicts ~the current state and noop
        // moves nothing, so noop must win the argmax. Under the old
        // code the epistemic bonus elected a novelty policy here.
        let mut engine = ActiveInferenceEngine::new();
        let rest = [0.5f32; DIM];
        for _ in 0..100 {
            engine.cycle(&rest, &rest, 1.0, &rest);
        }
        assert!(engine.test_precision() > 0.8);
        let result = engine.cycle(&rest, &rest, 1.0, &rest);
        assert_eq!(
            result.selected_policy, "noop",
            "noop should win at rest (efe={})",
            result.selected_policy_efe
        );
    }

    #[test]
    fn test_corrective_policy_wins_under_deviation_once_action_model_known() {
        // Install a known action model: GABA impulses lower cortisol.
        // With cortisol deviated high, the calm policy (GABA +0.03)
        // must outrank noop pragmatically — its predicted outcome
        // lands closer to the homeostatic target.
        use crate::state::neurochemical::NeurochemicalId;
        let mut engine = ActiveInferenceEngine::new();
        engine.test_set_precision(1.0); // exploitative argmax regime
        engine.test_set_action(
            NeurochemicalId::Cortisol as usize,
            NeurochemicalId::GABA as usize,
            -1.0,
        );
        let target = [0.5f32; DIM];
        let mut pre = target;
        pre[NeurochemicalId::Cortisol as usize] = 0.8;
        // First cycle initializes (returns early, no selection).
        engine.cycle(&pre, &pre, 1.0, &target);
        engine.test_set_precision(1.0); // re-set, robust to init changes
        let result = engine.cycle(&pre, &pre, 1.0, &target);
        assert_eq!(
            result.selected_policy, "calm",
            "calm should win under cortisol deviation (efe={})",
            result.selected_policy_efe
        );
    }

    #[test]
    fn test_information_gain_is_policy_independent_and_nonnegative() {
        // The expected information gain feeds the published EFE
        // decomposition but must be identical for every policy (it is
        // computed once per cycle, not per policy). Fresh engine:
        // prior_var == belief_var, so no uncertainty has been
        // resolved and the gain is ~0.
        let engine = ActiveInferenceEngine::new();
        let ig = engine.expected_information_gain();
        assert!(
            ig.is_finite() && ig >= 0.0,
            "IG must be finite nonnegative: {}",
            ig
        );
        assert!(ig.abs() < 1e-6, "fresh engine IG should be ~0: {}", ig);
    }

    // ─── Persistence v4 ─────────────────────────────────────────

    #[test]
    fn test_v4_save_load_preserves_error_trackers() {
        // The slow error trackers behind the precision target and the
        // maturity mapping must survive a save/load round trip —
        // otherwise a restart silently resets earned confidence.
        let mut engine = ActiveInferenceEngine::new();
        let rest = [0.5f32; DIM];
        for _ in 0..50 {
            engine.cycle(&rest, &rest, 1.0, &rest);
        }
        let precision_before = engine.test_precision();
        let maturity_before = engine.model_maturity();
        let path = std::env::temp_dir().join("genesis_test_aife_v4.bin");
        engine.save(&path).expect("save v4 model");
        let loaded = ActiveInferenceEngine::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            (loaded.test_precision() - precision_before).abs() < 1e-6,
            "precision should survive round trip: {} vs {}",
            loaded.test_precision(),
            precision_before
        );
        assert!(
            (loaded.model_maturity() - maturity_before).abs() < 1e-6,
            "maturity should survive round trip: {} vs {}",
            loaded.model_maturity(),
            maturity_before
        );
    }

    #[test]
    fn test_pre_v4_files_derive_consistent_trackers() {
        // A v3-length file carries no error trackers; loading must
        // derive them consistently from the persisted precision and
        // maturity rather than resetting to defaults. Fabricate the
        // body as zeros: precision clamps to 0.1, maturity to 0.0.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"AIFE");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.resize(8 + 2980, 0u8);
        let path = std::env::temp_dir().join("genesis_test_aife_v3compat.bin");
        std::fs::write(&path, &bytes).expect("write fabricated v3 file");
        let mut loaded = ActiveInferenceEngine::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            (loaded.test_precision() - 0.1).abs() < 1e-6,
            "precision should load clamped: {}",
            loaded.test_precision()
        );
        // One perfect tick: derived variance implies target 0.1, so
        // precision must not jump; derived maturity error implies
        // maturity ~0, so it must not jump either.
        let rest = [0.5f32; DIM];
        loaded.cycle(&rest, &rest, 1.0, &rest);
        assert!(
            (loaded.test_precision() - 0.1).abs() < 1e-6,
            "derived trackers must be self-consistent: precision {}",
            loaded.test_precision()
        );
        assert!(
            loaded.model_maturity() < 0.01,
            "derived trackers must be self-consistent: maturity {}",
            loaded.model_maturity()
        );
    }
}

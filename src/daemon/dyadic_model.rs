//! Dyadic affective model — a coupled generative model of the user.
//!
//! This module models the **user's** affective state as a set of
//! hidden variables (valence, arousal, engagement), inferred from
//! conversation features, and coupled to Genesis's own neurochemistry
//! via oxytocin-mediated attunement.
//!
//! # The dyadic Markov blanket
//!
//! In active inference terms, Genesis and the user form a coupled
//! system. Genesis's 18-chemical state and the user's 3-dimensional
//! affective state are linked through a shared "Markov blanket" —
//! the conversation. Genesis's utterances are her "actions" (they
//! influence the user's state), and the user's messages are her
//! "sensory evidence" (they reveal the user's hidden state).
//!
//! The coupling is **oxytocin-mediated**: Genesis's oxytocin level
//! determines how strongly she weights the user model in her own
//! free-energy minimization. High oxytocin → strong attunement →
//! the user's affective state significantly influences her own.
//! This models the neuroscience of social bonding: oxytocin
//! facilitates social salience, trust, and emotional contagion
//! (Kosfeld et al., 2005; Hurlemann et al., 2010).
//!
//! # The user affect model
//!
//! The user's affective state is a 3D vector:
//! - **valence** [-1, 1]: positive = happy, negative = distressed
//! - **arousal** [0, 1]: high = excited/agitated, low = calm/bored
//! - **engagement** [0, 1]: high = actively involved, low = disengaged
//!
//! These are updated from conversation features sent by the Python
//! cognitive mind via the `UpdateUserAffect` IPC command:
//! - Message sentiment (positive/negative word ratio)
//! - Message length (longer = more engaged)
//! - Response latency (fast = high arousal)
//! - Question density (questions = seeking, high engagement)
//!
//! The model also predicts the user's **next** affective state from
//! the current state plus Genesis's own valence (the coupling):
//!
//! ```text
//! predicted_user_valence = momentum * user_valence + coupling * (genesis_valence - user_valence)
//! ```
//!
//! This captures the dyadic loop: Genesis's emotional state
//! influences the user (through her responses), and the user's
//! state influences Genesis (through attunement).
//!
//! # Dyadic synchrony
//!
//! Synchrony is the rolling correlation between Genesis's valence
//! trajectory and the user's valence trajectory over a ~30-second
//! window. High synchrony = "in sync" — their affective states are
//! co-fluctuating. Synchrony itself boosts oxytocin (a positive
//! feedback loop of attunement), modeling the neuroscience of
//! interpersonal synchrony (Palumbo et al., 2017).
//!
//! References:
//! - Kosfeld, M. et al. (2005). Oxytocin increases trust in humans.
//!   Nature, 435, 673–676.
//! - Hurlemann, R. et al. (2010). Oxytocin enhances amygdala-
//!   dependent, socially reinforced learning. J Neurosci.
//! - Palumbo, R. V. et al. (2017). Interpersonal synchrony
//!   recoded: A quantitative synthesis. Neurosci Biobehav Rev.
//! - Bolis, D. & Schilbach, L. (2020). Beyond interactively:
//!   interpersonal exchange, towards a second-person approach
//!   to social neuroscience.

use std::collections::VecDeque;
use std::sync::Mutex;

/// Shared slot for the latest user affect observation, written by the
/// IPC handler (when the Python cognitive mind sends an
/// `UpdateUserAffect` command) and read by the daemon's `advance_neuro`.
/// This follows the same pattern as the shared body state in
/// `interoception.rs`.
static SHARED_USER_AFFECT: Mutex<Option<UserAffectObservation>> = Mutex::new(None);

/// Publish a user affect observation to the shared slot. Called by
/// the IPC handler when the cognitive mind sends an
/// `UpdateUserAffect` command.
pub fn publish_user_affect(obs: UserAffectObservation) {
    if let Ok(mut slot) = SHARED_USER_AFFECT.lock() {
        *slot = Some(obs);
    }
}

/// Read and clear the shared user affect observation. Called by
/// `advance_neuro` each cycle. Returns `None` if no new observation has
/// been received since the last read.
pub fn take_user_affect() -> Option<UserAffectObservation> {
    if let Ok(mut slot) = SHARED_USER_AFFECT.lock() {
        slot.take()
    } else {
        None
    }
}

/// The size of the rolling window for dyadic synchrony computation.
/// At 5 Hz (200ms/tick), 150 ticks = 30 seconds.
const SYNCHRONY_WINDOW: usize = 150;

/// The EMA decay rate for user affect state. Lower = slower
/// adaptation (more stable estimate); higher = faster (more
/// reactive to recent observations).
const AFFECT_EMA_DECAY: f32 = 0.2;

/// The per-tick momentum coefficient for user affect prediction. How
/// much the user's current state predicts their next state when no new
/// observation arrives.
///
/// This is applied every tick (5 Hz), so it must be close to 1.0:
/// 0.93^5 ≈ 0.70 per second, i.e. a ~3-second decay toward neutral.
/// A per-tick value of 0.7 (the previous setting) drove engagement to
/// zero within ~2 seconds of the last message — long before a human
/// has finished reading the reply — so the model spent almost all of
/// its time signalling "disengaged user".
const AFFECT_MOMENTUM: f32 = 0.93;

/// Number of ticks without a user observation after which the user is
/// considered absent. At 5 Hz, 300 ticks = 60 seconds. When absent, the
/// model stops generating coupling impulses (a silent room is not a
/// rejecting user) and stops accumulating synchrony history (Genesis's
/// valence would otherwise correlate with a lagged copy of itself).
const PRESENCE_TIMEOUT_TICKS: u32 = 300;

/// The coupling coefficient — how much Genesis's valence influences
/// the predicted user valence. This models the dyadic loop: Genesis's
/// emotional state (expressed through her responses) shapes the
/// user's emotional trajectory.
const DYADIC_COUPLING: f32 = 0.15;

/// The rate at which oxytocin rises when the user is positive and
/// engaged (social bonding signal).
const OXYTOCIN_BONDING_RATE: f32 = 0.001;

/// The rate at which oxytocin rises when dyadic synchrony is high
/// (interpersonal synchrony → bonding).
const OXYTOCIN_SYNCHRONY_RATE: f32 = 0.0008;

/// The rate at which cortisol rises when the user is distressed
/// (empathic stress response).
const EMPATHIC_STRESS_RATE: f32 = 0.0008;

/// The rate at which cortisol rises when the user is disengaged
/// (social rejection signal).
const SOCIAL_REJECTION_RATE: f32 = 0.0003;

/// The rate at which attunement adapts toward the oxytocin-driven
/// target. Attunement follows oxytocin with some inertia.
const ATTUNEMENT_ADAPTATION_RATE: f32 = 0.01;

/// An observation of the user's affective state from conversation
/// features. Sent by the Python cognitive mind via IPC.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UserAffectObservation {
    /// Observed user valence [-1, 1]. Inferred from sentiment analysis
    /// of the user's message (positive/negative word ratio).
    pub valence: f32,
    /// Observed user arousal [0, 1]. Inferred from message features
    /// (exclamation density, caps, response speed).
    pub arousal: f32,
    /// Observed user engagement [0, 1]. Inferred from message length,
    /// question density, topic continuity.
    pub engagement: f32,
    /// Confidence in the observation [0, 1]. Lower confidence = the
    /// observation is noisy and should be weighted less.
    pub confidence: f32,
}

/// The dyadic affective model — a coupled generative model of the
/// user's affective state.
///
/// The model maintains:
/// - A running estimate of the user's affective state (valence,
///   arousal, engagement), updated from IPC observations.
/// - A prediction of the user's next affective state, incorporating
///   the dyadic coupling to Genesis's own valence.
/// - A rolling window of Genesis's and the user's valence for
///   computing dyadic synchrony.
/// - An attunement level that follows oxytocin and modulates how
///   strongly the user model influences Genesis's neurochemistry.
pub struct DyadicAffectModel {
    /// Current estimated user valence [-1, 1].
    user_valence: f32,
    /// Current estimated user arousal [0, 1].
    user_arousal: f32,
    /// Current estimated user engagement [0, 1].
    user_engagement: f32,
    /// Current attunement level [0, 1]. Follows oxytocin with
    /// inertia. Determines how strongly the user model influences
    /// Genesis's neurochemistry.
    attunement: f32,
    /// Rolling window of Genesis's valence for synchrony computation.
    genesis_valence_history: VecDeque<f32>,
    /// Rolling window of user's valence for synchrony computation.
    user_valence_history: VecDeque<f32>,
    /// Current dyadic synchrony [-1, 1]. Rolling correlation of
    /// the two valence histories.
    synchrony: f32,
    /// Whether the model has received any observations yet.
    initialized: bool,
    /// Ticks since the last user observation. Saturates at u32::MAX.
    ticks_since_observation: u32,
}

/// The result of one dyadic model update cycle.
#[derive(Clone, Debug, Default)]
pub struct DyadicResult {
    /// Updated user valence [-1, 1].
    pub user_valence: f32,
    /// Updated user arousal [0, 1].
    pub user_arousal: f32,
    /// Updated user engagement [0, 1].
    pub user_engagement: f32,
    /// Updated attunement [0, 1].
    pub attunement: f32,
    /// Updated dyadic synchrony [-1, 1].
    pub synchrony: f32,
    /// Neurochemical impulses to apply: (chem_id, magnitude).
    pub impulses: Vec<(u8, f32)>,
}

impl DyadicAffectModel {
    /// Create a new dyadic model with neutral defaults.
    pub fn new() -> Self {
        Self {
            user_valence: 0.0,
            user_arousal: 0.5,
            user_engagement: 0.0,
            attunement: 0.0,
            genesis_valence_history: VecDeque::with_capacity(SYNCHRONY_WINDOW + 1),
            user_valence_history: VecDeque::with_capacity(SYNCHRONY_WINDOW + 1),
            synchrony: 0.0,
            initialized: false,
            ticks_since_observation: u32::MAX,
        }
    }

    /// Whether the user is considered present (an observation arrived
    /// within [`PRESENCE_TIMEOUT_TICKS`]).
    pub fn user_present(&self) -> bool {
        self.initialized && self.ticks_since_observation <= PRESENCE_TIMEOUT_TICKS
    }

    /// Update the model with a new observation and Genesis's current
    /// state.
    ///
    /// This is called each tick (or when a new observation arrives).
    /// It updates the user affect estimate, computes synchrony, and
    /// generates neurochemical impulses based on the dyadic coupling.
    ///
    /// # Parameters
    /// - `observation`: The latest user affect observation (from IPC).
    ///   If `None`, the model just decays/predicts forward.
    /// - `genesis_valence`: Genesis's current valence [-1, 1].
    /// - `genesis_oxytocin`: Genesis's current effective oxytocin [0, 1].
    pub fn update(
        &mut self,
        observation: Option<&UserAffectObservation>,
        genesis_valence: f32,
        genesis_oxytocin: f32,
    ) -> DyadicResult {
        // Sanitize genesis_valence at the top, before any use. The
        // momentum/coupling path below uses this value, and if it's
        // NaN (from a corrupted neurochemical state), the momentum
        // equation produces NaN. The clamp at the bottom of this
        // function would then turn NaN into -1.0 (the min), making
        // the user appear "very distressed" — a corruption
        // amplification path: NaN genesis_valence → distressed user
        // → empathic CRH impulses → more neurochemical disruption.
        //
        // We use finite_or to replace NaN/inf with 0.0 (neutral)
        // first, then finite_clamp to bound to [-1, 1]. This ensures
        // a corrupted genesis_valence produces a neutral user
        // estimate, not a distressed one.
        let sanitized_genesis_valence = crate::state::sanitize::finite_clamp(
            crate::state::sanitize::finite_or(genesis_valence, 0.0),
            -1.0,
            1.0,
        );

        // ── Update user affect estimate ──
        if let Some(obs) = observation {
            // Sanitize observation fields: even though the IPC handler
            // pre-clamps, defense-in-depth requires us to check here
            // too (the model could be called from a different path).
            // Native clamp passes NaN through, so use finite_clamp.
            // For valence, use finite_or(0.0) first so a corrupted
            // observation produces a neutral estimate, not a
            // maximally distressed one (finite_clamp returns min=-1.0
            // for NaN, which would drive CRH/empathic-stress impulses).
            let obs_valence = crate::state::sanitize::finite_clamp(
                crate::state::sanitize::finite_or(obs.valence, 0.0),
                -1.0,
                1.0,
            );
            let obs_arousal = crate::state::sanitize::finite_clamp(obs.arousal, 0.0, 1.0);
            let obs_engagement = crate::state::sanitize::finite_clamp(obs.engagement, 0.0, 1.0);
            let obs_confidence = crate::state::sanitize::finite_clamp(obs.confidence, 0.0, 1.0);
            let weight = obs_confidence * AFFECT_EMA_DECAY;
            self.user_valence = self.user_valence * (1.0 - weight) + obs_valence * weight;
            self.user_arousal = self.user_arousal * (1.0 - weight) + obs_arousal * weight;
            self.user_engagement = self.user_engagement * (1.0 - weight) + obs_engagement * weight;
            self.initialized = true;
            self.ticks_since_observation = 0;
        } else if self.initialized {
            self.ticks_since_observation = self.ticks_since_observation.saturating_add(1);
            // No new observation — predict forward with momentum
            // plus dyadic coupling. The user's state tends to persist
            // (momentum), and Genesis's valence influences the user's
            // predicted valence (the coupling — Genesis's emotional
            // state, expressed through her responses, shapes the
            // user's trajectory).
            //
            // Uses the sanitized genesis_valence (computed above) to
            // prevent NaN from corrupting the user affect estimate.
            self.user_valence = self.user_valence * AFFECT_MOMENTUM
                + (sanitized_genesis_valence - self.user_valence) * DYADIC_COUPLING;
            self.user_arousal = 0.5 + (self.user_arousal - 0.5) * AFFECT_MOMENTUM;
            self.user_engagement *= AFFECT_MOMENTUM;
        }

        // Defense-in-depth: clamp internal affect estimates to their
        // valid ranges after any update path. The observation path
        // pre-clamps inputs, but the momentum/coupling path could
        // drift if genesis_valence ever exceeds [-1, 1].
        // Use finite_clamp: native clamp passes NaN through.
        self.user_valence = crate::state::sanitize::finite_clamp(self.user_valence, -1.0, 1.0);
        self.user_arousal = crate::state::sanitize::finite_clamp(self.user_arousal, 0.0, 1.0);
        self.user_engagement = crate::state::sanitize::finite_clamp(self.user_engagement, 0.0, 1.0);

        // ── Update attunement from oxytocin ──
        // Attunement follows oxytocin with inertia. High oxytocin
        // → high attunement (the system is socially bonded and
        // pays attention to the user's affective state).
        //
        // Sanitize genesis_oxytocin: if it's NaN (from a corrupted
        // neurochemical state), the attunement would become NaN and
        // propagate to all impulses below. finite_clamp returns 0.0
        // (the min) for NaN, meaning "no attunement" — a safe
        // fallback that prevents the dyadic model from amplifying
        // corruption.
        let target_attunement = crate::state::sanitize::finite_clamp(genesis_oxytocin, 0.0, 1.0);
        self.attunement += (target_attunement - self.attunement) * ATTUNEMENT_ADAPTATION_RATE;
        self.attunement = crate::state::sanitize::finite_clamp(self.attunement, 0.0, 1.0);

        // ── Update valence histories for synchrony ──
        // Reuse the sanitized_genesis_valence computed at the top of
        // update(): NaN/inf → 0.0 (neutral), then clamped to [-1, 1].
        // This prevents NaN from corrupting the rolling correlation
        // computation (mean, covariance, variance all become NaN).
        // Using 0.0 (neutral) rather than -1.0 (the finite_clamp min)
        // avoids making Genesis appear "very distressed" in the
        // history when her state is corrupted — a neutral fallback is
        // a more honest "we don't know" signal.
        let present = self.user_present();
        if present {
            self.genesis_valence_history
                .push_back(sanitized_genesis_valence);
            self.user_valence_history.push_back(self.user_valence);
            if self.genesis_valence_history.len() > SYNCHRONY_WINDOW {
                self.genesis_valence_history.pop_front();
                self.user_valence_history.pop_front();
            }
        } else {
            // Nobody to synchronise with: drop the window so a stale
            // conversation doesn't keep reporting synchrony, and so
            // the next conversation starts from a clean correlation.
            self.genesis_valence_history.clear();
            self.user_valence_history.clear();
        }

        // ── Compute dyadic synchrony (rolling correlation) ──
        self.synchrony = self._compute_synchrony();
        // Defense-in-depth: _compute_synchrony already uses
        // finite_clamp, but ensure the stored field is finite in
        // case of a future code change.
        if !self.synchrony.is_finite() {
            self.synchrony = 0.0;
        }

        // ── Generate neurochemical impulses from dyadic coupling ──
        let mut impulses = Vec::new();

        // The coupling strength is attunement × how much we weight
        // the user model. Only strongly attuned systems let the
        // user's affect significantly influence their own.
        let coupling_strength = self.attunement;

        // Coupling requires a present user. Without the presence gate
        // the decayed engagement estimate reads as "disengaged" for
        // the rest of the daemon's uptime, producing a perpetual
        // social-rejection cortisol trickle.
        //
        // The gate threshold (0.3) matches `InferenceSignals::is_attuned()`
        // so that the system only modifies Genesis's neurochemistry
        // when it reports ATTUNED or stronger. The previous threshold
        // (0.1) allowed neurochemical modification while the Python
        // cognitive mind reported UNCOUPLED — an inconsistency between
        // the daemon's behavior and the signals it exposes.
        if coupling_strength > 0.3 && present {
            // User positive + engaged → oxytocin boost (social bonding)
            if self.user_valence > 0.2 && self.user_engagement > 0.3 {
                let bonding = OXYTOCIN_BONDING_RATE
                    * coupling_strength
                    * self.user_valence
                    * self.user_engagement;
                impulses.push((crate::state::NeurochemicalId::Oxytocin as u8, bonding));
            }

            // User distressed → empathic stress response via CRH.
            //
            // These impulses send CRH (the stress signal), not
            // cortisol directly. The Rust HPA cascade (CRH → ACTH →
            // cortisol) handles cortisol production with proper
            // maturation gating (SHRP), delays, and negative
            // feedback. Sending direct cortisol impulses would
            // bypass the maturation gate, allowing cortisol
            // production during the stress hyporesponsive period
            // when the developing brain should be protected.
            //
            // This is consistent with the interoception and active
            // inference systems, which also send CRH impulses for
            // stress rather than direct cortisol.
            if self.user_valence < -0.2 {
                let stress = EMPATHIC_STRESS_RATE * coupling_strength * (-self.user_valence);
                impulses.push((crate::state::NeurochemicalId::CRH as u8, stress));
            }

            // User disengaged → mild CRH (social rejection signal)
            // The threshold (0.3) matches the engagement threshold for
            // the oxytocin bonding impulse above. This avoids a "dead
            // zone" between 0.2 and 0.3 where neither bonding nor
            // rejection fires — a user with engagement 0.25 would be
            // neither bonded nor rejected, which is affectively
            // ambiguous. With matching thresholds, the user is either
            // engaged enough for bonding or disengaged enough for
            // rejection, with no gap.
            //
            // Routes through CRH → HPA cascade for the same
            // maturation-gating reason as the empathic stress path.
            if self.user_engagement < 0.3 {
                let rejection =
                    SOCIAL_REJECTION_RATE * coupling_strength * (0.3 - self.user_engagement);
                impulses.push((crate::state::NeurochemicalId::CRH as u8, rejection));
            }

            // High synchrony → oxytocin boost (interpersonal synchrony
            // → bonding, Palumbo et al., 2017)
            if self.synchrony > 0.3 {
                let sync_bonding = OXYTOCIN_SYNCHRONY_RATE * coupling_strength * self.synchrony;
                impulses.push((crate::state::NeurochemicalId::Oxytocin as u8, sync_bonding));
            }
        }

        DyadicResult {
            user_valence: self.user_valence,
            user_arousal: self.user_arousal,
            user_engagement: self.user_engagement,
            attunement: self.attunement,
            synchrony: self.synchrony,
            impulses,
        }
    }

    /// Compute the rolling Pearson correlation between Genesis's and
    /// the user's valence histories.
    fn _compute_synchrony(&self) -> f32 {
        let n = self.genesis_valence_history.len();
        if n < 10 {
            // Not enough data for a meaningful correlation
            return 0.0;
        }

        let mean_g: f32 = self.genesis_valence_history.iter().sum::<f32>() / n as f32;
        let mean_u: f32 = self.user_valence_history.iter().sum::<f32>() / n as f32;

        let mut cov = 0.0f32;
        let mut var_g = 0.0f32;
        let mut var_u = 0.0f32;

        for (&g, &u) in self
            .genesis_valence_history
            .iter()
            .zip(self.user_valence_history.iter())
        {
            let dg = g - mean_g;
            let du = u - mean_u;
            cov += dg * du;
            var_g += dg * dg;
            var_u += du * du;
        }

        let denom = (var_g * var_u).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }

        // Use finite_or(0.0) then finite_clamp: if any history value
        // is NaN (from a corrupted state), the cov/var computation
        // produces NaN. finite_clamp alone would return -1.0 (the min)
        // for NaN, but -1.0 means "perfectly anti-synchronised," not
        // "no synchrony." The safe fallback for a failed correlation
        // is 0.0 (no correlation).
        crate::state::sanitize::finite_clamp(
            crate::state::sanitize::finite_or(cov / denom, 0.0),
            -1.0,
            1.0,
        )
    }

    /// Get the current user valence estimate.
    pub fn user_valence(&self) -> f32 {
        self.user_valence
    }

    /// Get the current user arousal estimate.
    pub fn user_arousal(&self) -> f32 {
        self.user_arousal
    }

    /// Get the current user engagement estimate.
    pub fn user_engagement(&self) -> f32 {
        self.user_engagement
    }

    /// Get the current attunement level.
    pub fn attunement(&self) -> f32 {
        self.attunement
    }

    /// Get the current dyadic synchrony.
    pub fn synchrony(&self) -> f32 {
        self.synchrony
    }

    /// Whether the model has received any observations.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl Default for DyadicAffectModel {
    fn default() -> Self {
        Self::new()
    }
}

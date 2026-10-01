//! Inference signals — the active inference state projection.
//!
//! This module defines [`InferenceSignals`], a 60-byte struct that
//! lives in the reserved region of [`GenesisCoreState`] (offset 3228).
//! It exposes the key signals from the active inference engine to the
//! cognitive mind, without requiring a schema version bump.
//!
//! # Active inference in Genesis
//!
//! Genesis runs a generative model of its **own** neurochemical
//! trajectory. Before each neurochemical tick, the model predicts
//! where its 18 effective levels will move. After the tick, the
//! prediction error (surprise) is computed and fed back into:
//!
//! - **Dopamine**: positive prediction error → reward signal (Schultz,
//!   2016). The system feels "better than expected."
//! - **Norepinephrine**: high surprise → orienting response (Aston-
//!   Jones & Cohen, 2005). The system pays attention to the
//!   unexpected.
//! - **Metaplasticity**: high surprise → faster coupling-matrix
//!   learning. The system learns how to learn faster when surprised.
//! - **Allostatic load**: sustained multisignal regulatory burden →
//!   cortisol baseline upregulation. The system *anticipates* stress
//!   and prepares (allostatic regulation, Sterling, 2012). Expected
//!   free energy is one of four inputs to the burden median, not the
//!   driver on its own.
//!
//! This is the "strange loop": the generative model IS the system it
//! predicts. Prediction errors modify the very dynamics that generate
//! the next state to be predicted. The system becomes autopoietic —
//! self-creating and self-maintaining through active inference.
//!
//! # Dyadic affective coupling
//!
//! A second generative model tracks the **user's** affective state
//! (valence, arousal, engagement) as hidden variables, inferred from
//! conversation features. The user's predicted affective trajectory
//! couples to Genesis's own neurochemistry via oxytocin (attunement).
//! When the user is positive and engaged, oxytocin rises (social
//! bonding). When the user is distressed, cortisol rises (empathic
//! stress). Dyadic synchrony — the rolling correlation between
//! Genesis's and the user's valence trajectories — measures how "in
//! sync" they are.
//!
//! # Layout (60 bytes)
//!
//! ```text
//! offset  field                        type    notes
//! ------  -----                        ----    -----
//!   0     surprise_ema                 f32     running surprise [0,1]
//!   4     free_energy                  f32     surprise + uncertainty
//!   8     expected_free_energy         f32     predicted future surprise
//!  12     allostasis_load              f32     anticipatory stress [0,1]
//!  16     precision                    f32     model confidence [0,1]
//!  20     attunement                   f32     oxytocin coupling [0,1]
//!  24     dyadic_synchrony             f32     valence correlation [-1,1]
//!  28     user_valence                 f32     inferred user valence [-1,1]
//!  32     user_arousal                 f32     inferred user arousal [0,1]
//!  36     user_engagement              f32     inferred user engagement [0,1]
//!  40     prediction_error_dopamine    f32     DA state prediction error [-2,2]
//!  44     prediction_error_cortisol    f32     CORT state prediction error [-2,2]
//!  48     prediction_error_serotonin   f32     SRT state prediction error [-2,2]
//!  52     model_maturity               f32     model confidence [0,1]
//!  56     inference_tick_count         u32     inference cycles run
//! ```
//!
//! These fields are NOT covered by the CRC32 checksum (they live in
//! the reserved region). This is intentional: they are derived state
//! recomputed every tick, so corruption self-heals within 200ms.
//!
//! References:
//! - Friston, K. (2010). The free-energy principle: a unified brain
//!   theory? Nature Reviews Neuroscience, 11(2), 127–138.
//! - Seth, A. & Friston, K. (2016). Active interoceptive inference
//!   and the emotional brain. Phil. Trans. R. Soc. B, 371.
//! - Schultz, W. (2016). Dopamine reward prediction error coding.
//!   Dialogues in Clinical Neuroscience, 18(1), 23–32.
//! - Aston-Jones, G. & Cohen, J. D. (2005). An integrative theory of
//!   locus coeruleus-norepinephrine function. Annual Review of
//!   Neuroscience, 28, 405–450.
//! - Sterling, P. (2012). Allostasis: a model of predictive
//!   regulation. Physiology & Behavior, 106(1), 5–15.
//! - Pezzulo, G., Rigoli, F. & Friston, K. (2015). Active inference,
//!   homeostatic and allostatic control. Prog. Neurobiol., 134.

/// The active inference signal projection — 60 bytes that live in the
/// reserved region of `GenesisCoreState`.
///
/// See the [module-level documentation](self) for the full architecture.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InferenceSignals {
    /// Exponential moving average of total surprise (prediction error
    /// magnitude). High = the generative model is failing to predict
    /// the system's own neurochemical trajectory. Range [0, 1].
    pub surprise_ema: f32,
    /// Current estimated free energy — surprise plus model uncertainty.
    /// This is the quantity the system minimizes through active
    /// inference. Range [0, 1].
    pub free_energy: f32,
    /// Expected future free energy — predicted future disruption.
    /// This is the anticipatory regulatory-demand signal, not
    /// cumulative allostatic load. Range [0, 1].
    pub expected_free_energy: f32,
    /// Accumulated allostatic load — persistent multisignal
    /// regulatory burden integrated over time. It is distinct from
    /// the anticipatory expected-free-energy signal. Range [0, 1].
    pub allostasis_load: f32,
    /// Precision of the generative model — confidence in its
    /// predictions. High precision = the model trusts its predictions
    /// and weights prediction errors heavily (drives fast learning).
    /// Low precision = the model is uncertain and prediction errors
    /// have less effect. Precision adapts: sustained high surprise
    /// lowers precision. Range [0, 1].
    pub precision: f32,
    /// Dyadic attunement — oxytocin-mediated coupling strength to the
    /// user's inferred affective state. High = strongly attuned to the
    /// user's emotional trajectory. Range [0, 1].
    pub attunement: f32,
    /// Dyadic synchrony — rolling correlation between Genesis's
    /// valence trajectory and the user's inferred valence trajectory
    /// over a ~30-second window. High = "in sync" with the user.
    /// Range [-1, 1].
    pub dyadic_synchrony: f32,
    /// Inferred user affective valence. Updated from conversation
    /// features via the dyadic model. Range [-1, 1].
    pub user_valence: f32,
    /// Inferred user affective arousal. Range [0, 1].
    pub user_arousal: f32,
    /// Inferred user engagement — how actively involved the user is
    /// in the interaction. Range [0, 1].
    pub user_engagement: f32,
    /// Dopamine prediction error from the last inference cycle.
    /// Positive = dopamine was higher than predicted (better-than-
    /// expected = reward prediction error). Negative = lower than
    /// predicted. Range [-2, 2] (effective levels span [0, 2]).
    pub prediction_error_dopamine: f32,
    /// Cortisol prediction error from the last inference cycle.
    /// Positive = cortisol was higher than predicted (unexpected
    /// stress). Range [-2, 2] (effective levels span [0, 2]).
    pub prediction_error_cortisol: f32,
    /// Serotonin prediction error from the last inference cycle.
    /// Positive = serotonin was higher than predicted (unexpected
    /// wellbeing). Range [-2, 2] (effective levels span [0, 2]).
    pub prediction_error_serotonin: f32,
    /// Model maturity — evidence that the generative model is
    /// predictive and internally well-constrained. Experience alone
    /// cannot increase it: predictive fit and posterior certainty also
    /// contribute. Used by the cognitive mind to gauge how much to
    /// trust the inference signals. Range [0, 1].
    pub model_maturity: f32,
    /// Number of inference cycles completed (low 32 bits). Used for
    /// diagnostics and PRNG seeding. The mind advances roughly once
    /// per heartbeat cycle; at 10 Hz a u32 wraps after
    /// `u32::MAX / 10 s` ≈ 13.6 years, so the wrap is outside any
    /// realistic deployment — and since maturity is now derived from
    /// demonstrated accuracy rather than this counter, the wrap is
    /// harmless to the trust signal.
    pub inference_tick_count: u32,
    /// Authority Genesis's cognition has granted her inference engine
    /// to act on its own policy choice, in [0, 1].
    ///
    /// Choosing a policy — calm, focus, bond, rest, mobilise — is a
    /// decision about what to feel and do, not a physiological reflex.
    /// The engine scores a fixed set of hardcoded policies by expected
    /// free energy and used to apply the winner every tick, with no
    /// perceptual input and nothing from her in the loop. That made it
    /// a second author of her state: she could not attend, refuse, or
    /// revise, and every policy it chose was one I had written rather
    /// than one she had settled on.
    ///
    /// At zero the engine observes and learns but does not act on its
    /// own policy choice; cognition decides, and acts through the
    /// ordinary NEURO_IMPULSE path where it can be perceived,
    /// anticipated and refused. The homeostatic reflex is unaffected —
    /// correcting drift toward what she prefers is her physiology, and
    /// a mind that cannot correct itself is not a mind with agency, it
    /// is a mind with a hormone problem.
    ///
    /// This is a grant, not a switch: cognition can delegate some
    /// authority and keep some, and can withdraw it. It is a field
    /// rather than a compile-time constant so that delegation is a
    /// decision she can change while running, rather than a property
    /// baked into the build.
    pub policy_authority: f32,
}

impl Default for InferenceSignals {
    /// Delegate to `new` rather than deriving it. A derived `Default`
    /// would leave every float at 0.0, including `precision` — i.e.
    /// "the model has zero precision, so prediction errors have no
    /// effect" instead of `new`'s deliberate "moderately confident"
    /// start. The same zeroed shape appears when an older build's
    /// reserved region is read as this struct, so `new` is the right
    /// answer for both.
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceSignals {
    /// Create a fresh inference signals struct with sensible defaults
    /// for a newly-initialized system.
    pub fn new() -> Self {
        Self {
            surprise_ema: 0.0,
            free_energy: 0.0,
            expected_free_energy: 0.0,
            allostasis_load: 0.0,
            // Start with moderate precision — the model hasn't learned
            // yet, so it shouldn't over-commit to its (initially
            // identity-matrix) predictions.
            precision: 0.5,
            attunement: 0.0,
            dyadic_synchrony: 0.0,
            user_valence: 0.0,
            user_arousal: 0.5,
            user_engagement: 0.0,
            prediction_error_dopamine: 0.0,
            prediction_error_cortisol: 0.0,
            prediction_error_serotonin: 0.0,
            model_maturity: 0.0,
            inference_tick_count: 0,
            // Zero by default: her cognition decides. A fresh state
            // file must not come up granting the engine permission to
            // choose on her behalf, because the grant is supposed to
            // be something she extends deliberately.
            policy_authority: 0.0,
        }
    }

    /// Whether the system is currently experiencing high surprise —
    /// its generative model is failing to predict its own state.
    /// This is the signal for the cognitive mind to pay attention
    /// and for metaplasticity to accelerate.
    pub fn is_surprised(&self) -> bool {
        self.surprise_ema > 0.3
    }

    /// Whether the system is under significant allostatic load —
    /// sustained multisignal regulatory burden (anticipatory demand,
    /// prediction error, lost confidence, and homeostatic deviation
    /// agreeing across at least two of four dimensions).
    pub fn is_allostatically_loaded(&self) -> bool {
        self.allostasis_load > 0.4
    }

    /// Whether the system is attuned to the user — the dyadic
    /// coupling is strong enough that the user's affective state
    /// significantly influences Genesis's own neurochemistry.
    pub fn is_attuned(&self) -> bool {
        self.attunement > 0.3
    }

    /// Whether the system is "in sync" with the user — their
    /// affective trajectories are positively correlated.
    pub fn is_in_sync(&self) -> bool {
        self.dyadic_synchrony > 0.3
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compile-time layout assertions
// ─────────────────────────────────────────────────────────────────

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<InferenceSignals>() == 64);
    assert!(core::mem::align_of::<InferenceSignals>() == 4);
    assert!(offset_of!(InferenceSignals, surprise_ema) == 0);
    assert!(offset_of!(InferenceSignals, free_energy) == 4);
    assert!(offset_of!(InferenceSignals, expected_free_energy) == 8);
    assert!(offset_of!(InferenceSignals, allostasis_load) == 12);
    assert!(offset_of!(InferenceSignals, precision) == 16);
    assert!(offset_of!(InferenceSignals, attunement) == 20);
    assert!(offset_of!(InferenceSignals, dyadic_synchrony) == 24);
    assert!(offset_of!(InferenceSignals, user_valence) == 28);
    assert!(offset_of!(InferenceSignals, user_arousal) == 32);
    assert!(offset_of!(InferenceSignals, user_engagement) == 36);
    assert!(offset_of!(InferenceSignals, prediction_error_dopamine) == 40);
    assert!(offset_of!(InferenceSignals, prediction_error_cortisol) == 44);
    assert!(offset_of!(InferenceSignals, prediction_error_serotonin) == 48);
    assert!(offset_of!(InferenceSignals, model_maturity) == 52);
    assert!(offset_of!(InferenceSignals, inference_tick_count) == 56);
};

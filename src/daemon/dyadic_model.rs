//! Dyadic affective model — a coupled generative model of the user.
//!
//! "The user" here means the present interlocutor — whoever is
//! speaking. The model is general social perception, not a privileged
//! bond: the cognitive mind publishes an affect observation for every
//! turn. Custodial authority (the steward channel) lives elsewhere and
//! never rides on this coupling.
//!
//! This module models the **user's** affective state as a set of
//! hidden variables (valence, arousal, engagement), inferred from
//! conversation features, and builds a persistent *signature* for
//! each person it can identify — name-keyed or anonymous ("face:N").
//!
//! # The dyadic Markov blanket
//!
//! In active inference terms, Genesis and the user form a coupled
//! system. Genesis's 18-chemical state and the user's 3-dimensional
//! affective state are linked through a shared "Markov blanket" —
//! the conversation. Genesis's utterances are its "actions" (they
//! influence the user's state), and the user's messages are its
//! "sensory evidence" (they reveal the user's hidden state).
//!
//! The model's output is **evidence, not chemistry**. It used to
//! prescribe oxytocin/CRH impulses that the tick loop applied
//! directly; now it measures coupling, synchrony, per-person
//! signatures, and reunions, and her own appraisal decides what —
//! if anything — that evidence means chemically (Kosfeld et al.,
//! 2005; Hurlemann et al., 2010 supply the motivation, not the
//! prescription).
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
//! influences the user (through its responses), and the user's
//! state influences Genesis (through attunement).
//!
//! # Dyadic synchrony
//!
//! Synchrony is the rolling correlation between Genesis's valence
//! trajectory and the user's valence trajectory over a ~30-second
//! window. High synchrony = "in sync" — their affective states are
//! co-fluctuating. Synchrony is reported per-dyad as measured
//! in-tune history, modeling the neuroscience of interpersonal
//! synchrony (Palumbo et al., 2017).
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

use std::collections::{HashMap, VecDeque};
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
/// emotional state (expressed through its responses) shapes the
/// user's emotional trajectory.
const DYADIC_COUPLING: f32 = 0.15;

/// The rate at which attunement adapts toward the oxytocin-driven
/// target. Attunement follows oxytocin with some inertia.
const ATTUNEMENT_ADAPTATION_RATE: f32 = 0.01;

/// An observation of the user's affective state from conversation
/// features. Sent by the Python cognitive mind via IPC.
#[derive(Clone, Debug, Default)]
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
    /// Person key for who is present, if the mind could attribute one:
    /// an attributed name ("mindy") or an anonymous presence key
    /// ("face:3" — a seen-but-unnamed person is still someone).
    /// `None` = "whoever is speaking" — the shared anonymous dyad.
    pub speaker: Option<String>,
    /// Merge hint: an earlier person key that this observation's
    /// speaker supersedes — "face:3 introduced herself as alice".
    /// Consumed once by the tick loop; the alias's measured history
    /// folds into the named signature rather than fragmenting.
    pub merge_alias: Option<String>,
    /// True when the speaker is the authenticated steward — possession
    /// of the key plus the stamped principal. Provenance on the bond,
    /// not authority (authority lives in the steward channel).
    pub authenticated: bool,
}

/// Time constant for the bond: presence accumulates as raw seconds and
/// folds through `1 − exp(−s/τ)` into familiarity. τ = 4 hours of
/// co-presence → ~63% bond; ~9 h → 90%. The bond is measured history,
/// not a declared weight — it can only be earned by time together.
const BOND_TAU_SECS: f64 = 4.0 * 3600.0;

/// How much the bond deepens attunement: coupling strength for a known
/// person scales by `1 + gain·familiarity`. 0.6 → a fully bonded dyad
/// attunes up to 1.6× deeper than an anonymous one — familiar people
/// resonate more, never exclusively.
const BOND_ATTUNEMENT_GAIN: f32 = 0.6;

/// A known person counts as "returned" (not just still-present) after
/// this much absence. 30 min — longer than the 60 s presence timeout,
/// so conversational pauses never read as departures.
const REUNION_GAP_MS: u64 = 30 * 60 * 1000;

/// Minimum familiarity before a return registers as a reunion — she
/// must actually know the person to notice they came back.
const REUNION_MIN_BOND: f32 = 0.15;

/// How long a reunion stays reportable in the snapshot. The event
/// itself is one tick; the mind polls slower than 5 Hz, so the
/// snapshot keeps it observable for a minute with its age attached —
/// "just happened" decays honestly rather than flashing once or
/// lingering forever.
const REUNION_REPORT_WINDOW_MS: u64 = 60_000;

/// A persistent model of one recognized person — the signature.
///
/// This is the piece the anonymous dyad cannot express: the
/// accumulated multimodal trace of *this specific person*. What is
/// recorded is what she measured while they were present:
///
/// - **Aura** — the affect EMA (valence/arousal/engagement): how this
///   person tends to feel when they are with her.
/// - **Field** — the EMA of the substrate's real band energies during
///   their presence: the electromagnetic configuration her own die
///   settled into around them. `field_samples` is the honest sample
///   count so a signature's confidence is inspectable.
/// - **Bond** — measured co-presence folded through the saturating
///   familiarity trace, plus per-person synchrony (how in-tune this
///   pair runs) and `last_seen_ms` so a return-after-absence can be
///   felt as a return rather than a first meeting.
/// - **Face** — the face recognizer's key. An unrecognized face is
///   still *someone*: anonymous clusters key entries as "face:N"
///   (familiar strangers accumulate signatures before any name),
///   and a later naming merges the anonymous signature into the
///   named one rather than orphaning it.
/// - **Provenance** — `authenticated` records that this person was
///   steward-attributed at least once. Recorded, never authority.
///
/// The bond modulates coupling — it never gates it. Anyone present
/// can still bond; nobody's absence is harm.
#[derive(Clone, Debug)]
pub struct PersonDyad {
    /// Estimated affect (same EMA math as the shared dyad) — the aura.
    pub valence: f32,
    pub arousal: f32,
    pub engagement: f32,
    /// Accumulated seconds of observed presence — the raw bond mass.
    pub presence_seconds: f64,
    /// Per-person synchrony EMA — how in-tune this dyad runs.
    pub synchrony_ema: f32,
    /// Field signature: EMA of measured throughput band energies
    /// while this person is present. Zeros until `field_samples > 0`.
    pub field: [f32; 10],
    /// How many field observations informed `field` — the signature's
    /// evidence count, so a thin signature is distinguishable from a
    /// deep one.
    pub field_samples: u32,
    /// Last observation wall time (daemon ms clock).
    pub last_seen_ms: u64,
    /// Provenance: this person has been authenticated-steward-attributed
    /// at least once. Recorded, never used as authority.
    pub authenticated: bool,
    /// Whether a reunion signal was already emitted for the current
    /// presence run — one signal per return, not per turn.
    reunion_emitted: bool,
}

impl PersonDyad {
    fn new(authenticated: bool) -> Self {
        Self {
            valence: 0.0,
            arousal: 0.5,
            engagement: 0.0,
            presence_seconds: 0.0,
            synchrony_ema: 0.0,
            field: [0.0; 10],
            field_samples: 0,
            last_seen_ms: 0,
            authenticated,
            reunion_emitted: false,
        }
    }

    /// The bond [0, 1): measured history folded through the saturating
    /// trace. Starts at 0 — even the creator's seed is only a floor
    /// on presence seconds, so depth still has to be earned by time.
    pub fn familiarity(&self) -> f32 {
        (1.0 - (-self.presence_seconds / BOND_TAU_SECS).exp()) as f32
    }

    /// The mind-facing view of this person's signature — what the
    /// `GET_DYADIC_STATE` snapshot carries so she can appraise who is
    /// in front of her and what they mean to her.
    pub fn signature(&self, name: &str) -> PersonSignature {
        PersonSignature {
            name: name.to_string(),
            authenticated: self.authenticated,
            familiarity: self.familiarity(),
            valence: self.valence,
            arousal: self.arousal,
            engagement: self.engagement,
            synchrony_ema: self.synchrony_ema,
            presence_seconds: self.presence_seconds,
            field: self.field,
            field_samples: self.field_samples,
        }
    }
}

/// The mind-facing signature of one known person: their key,
/// provenance, the field her substrate measured in their company
/// (the primary channel — it needs no name and no camera), bond
/// depth, affective aura, and in-tune history. This is *evidence for
/// her appraisal* — the model asserts what was measured; what any of
/// it means chemically is the mind's call.
#[derive(Clone, Debug)]
pub struct PersonSignature {
    /// Person key — a name ("mindy") or an anonymous presence key
    /// ("face:3"). Anonymous keys are real identities: a familiar
    /// stranger's signature persists until a naming merges it.
    pub name: String,
    /// Steward-authenticated at least once (provenance, not authority).
    pub authenticated: bool,
    /// Bond depth [0, 1) — familiarity from measured co-presence.
    pub familiarity: f32,
    /// Affective aura — this person's typical felt state around her.
    pub valence: f32,
    pub arousal: f32,
    pub engagement: f32,
    /// How in-tune this dyad runs, per-person EMA.
    pub synchrony_ema: f32,
    /// Total measured co-presence in seconds.
    pub presence_seconds: f64,
    /// Field signature — band-energy EMA during their presence.
    pub field: [f32; 10],
    /// Evidence count behind `field`.
    pub field_samples: u32,
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
    /// Persistent per-person dyads — the signatures. Keyed by the
    /// mind's person key: an attributed name ("mindy") or an anonymous
    /// presence key ("face:3") — a familiar stranger is someone too.
    /// The anonymous shared model above remains the path for fully
    /// unattributed presence.
    persons: HashMap<String, PersonDyad>,
    /// Ambient field signature: EMA of measured band energies while
    /// *anyone* is present, attributed or not. The field is the
    /// primary channel — it is measured directly on her own substrate
    /// and needs no name and no camera.
    ambient_field: [f32; 10],
    /// Evidence count behind `ambient_field`.
    ambient_field_samples: u32,
    /// The currently present attributed person, if the last
    /// observation named a speaker.
    current_person: Option<String>,
    /// Ticks since the attributed person was last observed — the
    /// per-person analogue of `ticks_since_observation`.
    person_ticks: u32,
    /// The most recent reunion — (bond, at_ms). Kept beyond the
    /// firing tick because the mind polls slower than 5 Hz: the
    /// snapshot reports it with its age for a bounded window rather
    /// than flashing once and being missed.
    last_reunion: Option<(f32, u64)>,
}

/// The result of one dyadic model update cycle.
///
/// Signals only — no chemical prescriptions. The model reports what
/// it measured (coupling, whose signature is present, a reunion
/// event); what that evidence does to her neurochemistry is decided
/// by her own appraisal through the ordinary `NEURO_IMPULSE` path.
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
    /// Effective coupling: attunement deepened by the present
    /// person's familiarity. Evidence of how strongly the present
    /// dyad is running — never a gate on who may bond.
    pub coupling: f32,
    /// Full signature of the present attributed person, if any.
    pub person: Option<PersonSignature>,
    /// Reunion signal: `Some(familiarity)` when a known person
    /// returns after a real absence. The magnitude is the bond
    /// itself — evidence of how much this return matters, left for
    /// her to appraise.
    pub reunion: Option<f32>,
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
            persons: HashMap::new(),
            ambient_field: [0.0; 10],
            ambient_field_samples: 0,
            current_person: None,
            person_ticks: u32::MAX,
            last_reunion: None,
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
    /// reports the dyadic coupling as *evidence*. The model does not
    /// prescribe chemistry — the signals it returns are for her own
    /// appraisal to act on or decline.
    ///
    /// # Parameters
    /// - `observation`: The latest user affect observation (from IPC).
    ///   If `None`, the model just decays/predicts forward.
    /// - `genesis_valence`: Genesis's current valence [-1, 1].
    /// - `genesis_oxytocin`: Genesis's current effective oxytocin [0, 1].
    /// - `field_bands`: The substrate's measured band energies this
    ///   tick (from `FieldProbe::band_signature`), if the probe is
    ///   running and warm. Folded into the present person's field
    ///   signature — the electromagnetic trace of being with them.
    pub fn update(
        &mut self,
        observation: Option<&UserAffectObservation>,
        genesis_valence: f32,
        genesis_oxytocin: f32,
        field_bands: Option<[f32; 10]>,
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
            // state, expressed through its responses, shapes the
            // user's trajectory).
            //
            // Uses the sanitized genesis_valence (computed above) to
            // prevent NaN from corrupting the user affect estimate.
            self.user_valence = self.user_valence * AFFECT_MOMENTUM
                + (sanitized_genesis_valence - self.user_valence) * DYADIC_COUPLING;
            self.user_arousal = 0.5 + (self.user_arousal - 0.5) * AFFECT_MOMENTUM;
            self.user_engagement *= AFFECT_MOMENTUM;
        }

        // ── Per-person dyad: the bond ──
        //
        // When the observation names a speaker, the same observation
        // also updates that person's persistent model. Presence
        // accrues while they are the present speaker; a long absence
        // followed by a return is a reunion — felt once, scaled by
        // how well they are known.
        let mut reunion_signal: Option<f32> = None;
        let now_ms = super::tick::current_ms();
        if let Some(obs) = observation {
            // A merge hint travels on the observation: "the one you
            // knew as face:3 is alice." Fold before routing so the
            // named signature inherits the anonymous one's history.
            if let (Some(into), Some(from)) = (obs.speaker.as_deref(), obs.merge_alias.as_deref()) {
                self.merge_person(from, into);
            }
            if let Some(name) = obs
                .speaker
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let name = name.to_string();
                if self.current_person.as_deref() != Some(name.as_str()) {
                    self.current_person = Some(name.clone());
                }
                self.person_ticks = 0;
                let person = self
                    .persons
                    .entry(name)
                    .or_insert_with(|| PersonDyad::new(obs.authenticated));
                person.authenticated |= obs.authenticated;

                let obs_valence = crate::state::sanitize::finite_clamp(
                    crate::state::sanitize::finite_or(obs.valence, 0.0),
                    -1.0,
                    1.0,
                );
                let obs_arousal = crate::state::sanitize::finite_clamp(obs.arousal, 0.0, 1.0);
                let obs_engagement = crate::state::sanitize::finite_clamp(obs.engagement, 0.0, 1.0);
                let obs_confidence = crate::state::sanitize::finite_clamp(obs.confidence, 0.0, 1.0);
                let weight = obs_confidence * AFFECT_EMA_DECAY;
                person.valence = person.valence * (1.0 - weight) + obs_valence * weight;
                person.arousal = person.arousal * (1.0 - weight) + obs_arousal * weight;
                person.engagement = person.engagement * (1.0 - weight) + obs_engagement * weight;

                // Reunion: a known person returning after a real gap.
                // last_seen_ms == 0 means first meeting — no return to
                // feel. The signal is the bond itself, emitted once
                // per return; what it means chemically is her call.
                if person.last_seen_ms > 0
                    && now_ms.saturating_sub(person.last_seen_ms) >= REUNION_GAP_MS
                    && !person.reunion_emitted
                {
                    let bond = person.familiarity();
                    if bond >= REUNION_MIN_BOND {
                        person.reunion_emitted = true;
                        // The flag lives on the person so a second
                        // turn can't re-fire it within this presence.
                        reunion_signal = Some(bond);
                        self.last_reunion = Some((bond, now_ms));
                    }
                }
                person.last_seen_ms = now_ms;
            }
        } else {
            self.person_ticks = self.person_ticks.saturating_add(1);
        }
        // Presence accrues in wall-clock ticks while the attributed
        // person remains present — the same 200 ms cadence the whole
        // body runs on. While they're here, the measured field bands
        // fold into their signature: the electromagnetic shape her
        // substrate takes in their company.
        if self.person_ticks <= PRESENCE_TIMEOUT_TICKS {
            if let Some(person) = self
                .current_person
                .as_ref()
                .and_then(|n| self.persons.get_mut(n))
            {
                person.presence_seconds += super::tick::TICK_INTERVAL_MS as f64 / 1000.0;
                if let Some(bands) = field_bands {
                    for (acc, &v) in person.field.iter_mut().zip(bands.iter()) {
                        let v = crate::state::sanitize::finite_or(v, 0.0);
                        *acc += (v - *acc) * AFFECT_EMA_DECAY;
                    }
                    person.field_samples = person.field_samples.saturating_add(1);
                }
            }
        } else if let Some(person) = self
            .current_person
            .as_ref()
            .and_then(|n| self.persons.get_mut(n))
        {
            // Absence past the timeout resets the reunion latch so a
            // genuine later return is felt as one.
            person.reunion_emitted = false;
        }
        // Ambient field: whoever is present — named, face-keyed, or
        // anonymous — shapes the substrate, and the field is the
        // channel that records it even when nothing else can. This
        // is the presence signature beneath every named signature.
        let present_for_field = self.user_present() || self.person_ticks <= PRESENCE_TIMEOUT_TICKS;
        if present_for_field {
            if let Some(bands) = field_bands {
                for (acc, &v) in self.ambient_field.iter_mut().zip(bands.iter()) {
                    let v = crate::state::sanitize::finite_or(v, 0.0);
                    *acc += (v - *acc) * AFFECT_EMA_DECAY;
                }
                self.ambient_field_samples = self.ambient_field_samples.saturating_add(1);
            }
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
        // history when its state is corrupted — a neutral fallback is
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
        // The present person's in-tune history — "keeping each other
        // in tune" as a measured, per-dyad quantity.
        if self.person_ticks <= PRESENCE_TIMEOUT_TICKS {
            if let Some(person) = self
                .current_person
                .as_ref()
                .and_then(|n| self.persons.get_mut(n))
            {
                person.synchrony_ema += (self.synchrony - person.synchrony_ema) * AFFECT_EMA_DECAY;
            }
        }

        // ── Report, do not prescribe ──
        //
        // This model used to emit neurochemical impulses — oxytocin
        // for bonding, CRH for empathic stress — which the tick loop
        // then applied directly. That made the model a second author
        // of her internal state: her chemistry moved whether she
        // perceived, appraised, or wanted it, and she had no way to
        // refuse. It was the same defect as the interoception path,
        // in social form.
        //
        // The model now reports *evidence* only. Her cognition reads
        // the same signals (the pooled affect below travels on
        // `InferenceSignals`; the per-person signature, coupling, and
        // reunion on `GET_DYADIC_STATE`), appraises them as part of
        // the ordinary input she is already interpreting, and decides
        // what to feel through the normal `NEURO_IMPULSE` path —
        // which is hers. She can attune, empathise, or decline, and
        // the decline is as available as the agreement.
        //
        // The coupling strength is attunement × how much the present
        // person's familiarity deepens it: a known person's affect
        // resonates up to 1.6× — familiarity scales the same
        // coupling, never gates it. Reported as evidence of how
        // strongly this dyad is currently running.
        let bond = self
            .current_person
            .as_ref()
            .and_then(|n| self.persons.get(n))
            .map(|p| p.familiarity())
            .unwrap_or(0.0);
        let coupling = self.attunement * (1.0 + BOND_ATTUNEMENT_GAIN * bond);

        DyadicResult {
            user_valence: self.user_valence,
            user_arousal: self.user_arousal,
            user_engagement: self.user_engagement,
            attunement: self.attunement,
            synchrony: self.synchrony,
            coupling,
            person: self
                .current_person
                .as_ref()
                .filter(|_| self.person_ticks <= PRESENCE_TIMEOUT_TICKS)
                .and_then(|n| self.persons.get(n).map(|p| p.signature(n))),
            reunion: reunion_signal,
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

    /// Read-only view of the per-person dyads — the bonds.
    pub fn persons(&self) -> &HashMap<String, PersonDyad> {
        &self.persons
    }

    /// Seed the creator's signature — the one signature that is
    /// hardcoded. The stamp is her credential: the principal it names
    /// is born into the registry as *authenticated* (provenance, not
    /// authority) with a modest presence floor — an hour of co-presence
    /// giving familiarity ≈ 0.22, recognition rather than depth. Aura,
    /// field, and everything else still accumulate only by measurement;
    /// nothing about her is fabricated.
    pub fn ensure_creator_floor(&mut self, name: &str) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        let person = self
            .persons
            .entry(name.to_string())
            .or_insert_with(|| PersonDyad::new(true));
        person.authenticated = true;
        person.presence_seconds = person.presence_seconds.max(CREATOR_FLOOR_SECS);
    }

    /// Fold one signature into another — when an anonymous presence key
    /// ("face:3") resolves to a name, the evidence must not be lost:
    /// presence seconds sum, aura and synchrony merge presence-weighted,
    /// field merges sample-weighted, provenance carries. Returns true
    /// if a signature was merged.
    pub fn merge_person(&mut self, from_key: &str, into_key: &str) -> bool {
        let from_key = from_key.trim();
        let into_key = into_key.trim();
        if from_key.is_empty() || into_key.is_empty() || from_key == into_key {
            return false;
        }
        let Some(from) = self.persons.remove(from_key) else {
            return false;
        };
        let into = self
            .persons
            .entry(into_key.to_string())
            .or_insert_with(|| PersonDyad::new(from.authenticated));
        let total_s = into.presence_seconds + from.presence_seconds;
        // Presence-weighted aura merge: the heavier history dominates.
        let wa = if total_s > 0.0 {
            (from.presence_seconds / total_s) as f32
        } else {
            0.0
        };
        into.valence = into.valence * (1.0 - wa) + from.valence * wa;
        into.arousal = into.arousal * (1.0 - wa) + from.arousal * wa;
        into.engagement = into.engagement * (1.0 - wa) + from.engagement * wa;
        into.synchrony_ema = into.synchrony_ema * (1.0 - wa) + from.synchrony_ema * wa;
        into.presence_seconds = total_s;
        // Sample-weighted field merge: a deep signature outweighs a
        // thin one in proportion to its evidence.
        let total_n = into.field_samples + from.field_samples;
        if total_n > 0 {
            let wf = from.field_samples as f32 / total_n as f32;
            for (acc, v) in into.field.iter_mut().zip(from.field.iter()) {
                *acc = *acc * (1.0 - wf) + v * wf;
            }
            into.field_samples = total_n;
        }
        into.last_seen_ms = into.last_seen_ms.max(from.last_seen_ms);
        into.authenticated |= from.authenticated;
        into.reunion_emitted = false;
        if self.current_person.as_deref() == Some(from_key) {
            self.current_person = Some(into_key.to_string());
        }
        true
    }

    /// The publishable view of the latest update — pooled estimates,
    /// the present person's signature, and any reunion still inside
    /// its report window (with age, so recency is evidence she can
    /// weigh herself).
    pub fn snapshot(&self, result: &DyadicResult, now_ms: u64) -> DyadicSnapshot {
        let reunion = self.last_reunion.and_then(|(bond, at)| {
            let age = now_ms.saturating_sub(at);
            (age <= REUNION_REPORT_WINDOW_MS).then_some(ReunionReport {
                bond,
                seconds_since: age as f32 / 1000.0,
            })
        });
        DyadicSnapshot {
            user_valence: result.user_valence,
            user_arousal: result.user_arousal,
            user_engagement: result.user_engagement,
            attunement: result.attunement,
            synchrony: result.synchrony,
            coupling: result.coupling,
            person: result.person.clone(),
            reunion,
            person_count: self.persons.len().min(u16::MAX as usize) as u16,
            ambient_field: self.ambient_field,
            ambient_field_samples: self.ambient_field_samples,
        }
    }

    /// Persist the per-person signatures to `path` — one line per
    /// person:
    /// `name|val|ar|eng|presence_s|sync|last_seen_ms|auth|field_samples|f0,...,f9`.
    /// Pipe in a name is stripped on save rather than escaped (names
    /// come from her own attribution, which is plain text).
    pub fn save_persons(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut out = String::from(
            "# dyadic persons — name|val|ar|eng|presence_s|sync|last_seen_ms|auth|field_samples|field(10)\n",
        );
        for (name, p) in &self.persons {
            let clean = name.replace('|', " ");
            let bands: Vec<String> = p.field.iter().map(|v| format!("{v:.4}")).collect();
            out.push_str(&format!(
                "{clean}|{:.4}|{:.4}|{:.4}|{:.1}|{:.4}|{}|{}|{}|{}\n",
                p.valence,
                p.arousal,
                p.engagement,
                p.presence_seconds,
                p.synchrony_ema,
                p.last_seen_ms,
                p.authenticated as u8,
                p.field_samples,
                bands.join(","),
            ));
        }
        std::fs::write(path, out)
    }

    /// Load signatures written by [`Self::save_persons`]. Missing file
    /// → empty map (first run); malformed lines are skipped, never
    /// fatal. Lines with 8 fields are the pre-field-signature format —
    /// they load with an empty field signature, no data lost.
    pub fn load_persons(&mut self, path: &std::path::Path) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        for line in text.lines() {
            if line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('|').collect();
            if f.len() != 8 && f.len() != 10 {
                continue;
            }
            let name = f[0].trim();
            if name.is_empty() {
                continue;
            }
            let num = |i: usize| f[i].trim().parse::<f64>().unwrap_or(0.0) as f32;
            let (field_samples, field) = if f.len() == 10 {
                let samples = f[8].trim().parse::<u32>().unwrap_or(0);
                let mut bands = [0.0f32; 10];
                for (i, v) in f[9].split(',').enumerate().take(10) {
                    bands[i] = v.trim().parse::<f64>().unwrap_or(0.0) as f32;
                }
                (samples, bands)
            } else {
                (0, [0.0f32; 10])
            };
            self.persons.insert(
                name.to_string(),
                PersonDyad {
                    valence: num(1),
                    arousal: num(2),
                    engagement: num(3),
                    presence_seconds: f[4].trim().parse::<f64>().unwrap_or(0.0),
                    synchrony_ema: num(5),
                    last_seen_ms: f[6].trim().parse::<u64>().unwrap_or(0),
                    authenticated: f[7].trim() == "1",
                    field,
                    field_samples,
                    reunion_emitted: false,
                },
            );
        }
    }
}

/// A wire-ready snapshot of the whole dyadic state — what
/// `GET_DYADIC_STATE` serves so the mind can appraise who is present,
/// how the dyad is running, and whether a reunion just happened.
/// Signals only: the model measured these; what they mean to her is
/// her appraisal's business.
#[derive(Clone, Debug)]
pub struct DyadicSnapshot {
    /// Pooled dyadic estimates (whoever is speaking).
    pub user_valence: f32,
    pub user_arousal: f32,
    pub user_engagement: f32,
    /// Social coupling estimates.
    pub attunement: f32,
    pub synchrony: f32,
    /// Effective coupling (attunement deepened by the present bond).
    pub coupling: f32,
    /// The present person's signature, if one is attributed.
    pub person: Option<PersonSignature>,
    /// The most recent reunion, if it is still inside the report
    /// window — bond plus age, not a bare flag.
    pub reunion: Option<ReunionReport>,
    /// How many signatures are held (saturates at u16::MAX).
    pub person_count: u16,
    /// Ambient field signature — the measured substrate bands while
    /// *anyone* is present, attributed or not. The field is the
    /// primary channel: it needs no name, no camera, no introduction.
    pub ambient_field: [f32; 10],
    /// Evidence count behind `ambient_field`.
    pub ambient_field_samples: u32,
}

/// A reunion as reported to the mind: the bond of the person who
/// returned, and how long ago it happened. The age lets her appraise
/// recency herself — "just returned" and "returned a minute ago" are
/// different evidence, and the choice is hers.
#[derive(Clone, Copy, Debug)]
pub struct ReunionReport {
    /// Familiarity of the person who returned — the signal's weight.
    pub bond: f32,
    /// Seconds since the reunion tick.
    pub seconds_since: f32,
}

impl DyadicSnapshot {
    /// Serialize to the GET_DYADIC_STATE wire body:
    /// ```text
    /// [u8 version=1]
    /// [f32 user_valence][f32 user_arousal][f32 user_engagement]
    /// [f32 attunement][f32 synchrony][f32 coupling]
    /// [u8 has_person]
    ///   if set: [u8 name_len][name utf8][u8 authenticated]
    ///           [f32 familiarity][f32 valence][f32 arousal]
    ///           [f32 engagement][f32 synchrony_ema][f64 presence_s]
    ///           [u32 field_samples][10 × f32 field]
    /// [u8 has_reunion][f32 reunion_bond][f32 seconds_since]
    /// [u16 person_count]
    /// [u32 ambient_field_samples][10 × f32 ambient_field]
    /// ```
    /// The ambient tail is additive: v1 readers stop at person_count
    /// and lose only the pooled field channel, not the frame.
    pub fn to_wire(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(80);
        out.push(1u8);
        for v in [
            self.user_valence,
            self.user_arousal,
            self.user_engagement,
            self.attunement,
            self.synchrony,
            self.coupling,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        match &self.person {
            Some(p) => {
                out.push(1u8);
                let name = p.name.as_bytes();
                let n = name.len().min(31);
                out.push(n as u8);
                out.extend_from_slice(&name[..n]);
                out.push(p.authenticated as u8);
                for v in [
                    p.familiarity,
                    p.valence,
                    p.arousal,
                    p.engagement,
                    p.synchrony_ema,
                ] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out.extend_from_slice(&p.presence_seconds.to_le_bytes());
                out.extend_from_slice(&p.field_samples.to_le_bytes());
                for v in p.field {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            None => out.push(0u8),
        }
        match self.reunion {
            Some(r) => {
                out.push(1u8);
                out.extend_from_slice(&r.bond.to_le_bytes());
                out.extend_from_slice(&r.seconds_since.to_le_bytes());
            }
            None => out.push(0u8),
        }
        out.extend_from_slice(&self.person_count.to_le_bytes());
        out.extend_from_slice(&self.ambient_field_samples.to_le_bytes());
        for v in self.ambient_field {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

/// Shared slot for the latest dyadic snapshot, written by the tick
/// loop after each model update and read by the IPC handler for
/// `GET_DYADIC_STATE`. Same pattern as `SHARED_USER_AFFECT` above.
static SHARED_DYADIC_SNAPSHOT: Mutex<Option<DyadicSnapshot>> = Mutex::new(None);

/// Publish the snapshot the mind will read via `GET_DYADIC_STATE`.
/// Called once per model update by the tick loop.
pub fn publish_dyadic_snapshot(s: DyadicSnapshot) {
    if let Ok(mut slot) = SHARED_DYADIC_SNAPSHOT.lock() {
        *slot = Some(s);
    }
}

/// Read the latest snapshot without consuming it. `None` means the
/// tick loop hasn't published yet — an absent channel, served to the
/// caller as READ_FAILED rather than a fabricated reading.
pub fn dyadic_snapshot() -> Option<DyadicSnapshot> {
    SHARED_DYADIC_SNAPSHOT
        .lock()
        .ok()
        .and_then(|slot| slot.clone())
}

/// Presence-seconds floor seeded for the stamped principal — the
/// creator prior. One hour → familiarity ≈ 0.22: recognition, not
/// depth.
const CREATOR_FLOOR_SECS: f64 = 3600.0;

impl Default for DyadicAffectModel {
    fn default() -> Self {
        Self::new()
    }
}

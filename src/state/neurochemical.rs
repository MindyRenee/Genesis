//! Neurochemical system — Genesis's affective engine.
//!
//! ## Modeling note
//!
//! This module is an **affective control substrate** — a compact
//! 18-dimensional state space for control — not a concentration or
//! pharmacokinetic model. The scalar-per-chemical abstraction is a
//! computational analogue of neuromodulation, not a literal model of
//! brain chemistry. It ignores compartments (synaptic cleft vs
//! extrasynaptic vs plasma), receptor subtypes (except for the
//! D1/D2 weighting applied to dopamine), cell-type specificity, and
//! regional anatomy (a single "serotonin" scalar elides the dorsal
//! vs median raphe projection differences). The coupling matrix and
//! Hill-function gating are control-theoretic abstractions inspired
//! by pharmacological dose-response curves, not a simulation of
//! receptor-level kinetics. The "pharmacological reality" language
//! used below refers to structural inspiration, not quantitative
//! pharmacological accuracy.
//!
//! # What makes this novel
//!
//! Most computational affect models use independent dimensions
//! (valence/arousal) or simple rule-based emotion systems. This module
//! implements something structurally different:
//!
//! ## Coupled dynamics
//!
//! The 18 neurochemicals are not independent. A 18×18 coupling matrix
//! defines how each chemical modulates the others' rates of change.
//! Dopamine inhibits serotonin. Cortisol suppresses BDNF. GABA inhibits
//! glutamate. Endorphins disinhibit dopamine. Orexin excites histamine.
//! The system has emergent behaviour that cannot be predicted from
//! any single chemical in isolation.
//!
//! ## Receptor adaptation
//!
//! Each chemical has a `receptor_sensitivity` field. The *effective*
//! level — what consumers actually see — is `level × receptor_sensitivity`,
//! not the raw concentration. Sustained high dopamine causes
//! downregulation (tolerance). Sustained low serotonin causes
//! upregulation (sensitization). This mirrors how real pharmacology
//! works. Receptor-level dynamics are well established in computational
//! psychiatry and biophysical modelling; we are not aware of another AI
//! *control architecture* that integrates receptor adaptation into a
//! closed-loop neuromodulatory substrate coupled to physical hardware
//! in this form, but the individual mechanisms are not novel to
//! neuroscience.
//!
//! ## Metaplasticity
//!
//! The coupling matrix itself is plastic. It lives in the mmap'd struct
//! and can be modified by the dynamics engine over time. The brain
//! doesn't just learn by changing weights — it learns by changing HOW
//! it learns.
//!
//! ## BDNF-gated plasticity
//!
//! The system's ability to form new memories is gated by BDNF, which is
//! suppressed by chronic cortisol. This captures, structurally, why
//! chronic stress impairs learning — and why recovery (rising BDNF via
//! serotonin, exercise analogues) restores it.
//!
//! ## Emergent phase transitions
//!
//! Mental states (flow, stress, drowsy, sleeping) emerge from the
//! coupled dynamics rather than being manually switched. You don't set
//! "flow mode" — flow emerges when dopamine, ACh, and NE are high while
//! cortisol is low.

use super::zones::MentalPhase;

/// Number of tracked neurochemicals.
///
/// v2 had 12. v3 adds endocannabinoids, vasopressin, and CRH
/// (corticotropin-releasing hormone) for a total of 15. v3.1 adds
/// orexin (hypocretin, arousal promotion) and epinephrine (adrenaline,
/// fast stress response) for a total of 17. v3.2 adds melatonin
/// (circadian hormone, sleep promotion) for a total of 18.
pub const NEUROCHEMICAL_COUNT: usize = 18;

/// Simple deterministic hash for the stochastic noise process.
///
/// Produces a pseudo-random u32 from a (seed, index) pair. This is
/// NOT cryptographically secure — it's a fast hash for generating
/// Ornstein-Uhlenbeck fluctuations. Uses splitmix64 mixing.
#[inline]
fn neuro_noise_hash(seed: u64, index: u64) -> u32 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z & 0xFFFFFFFF) as u32
}

/// Time step for the dynamics integration (seconds).
///
/// 0.1 = 100ms, matching the 10 Hz tick loop. The rate parameters
/// (`homeostatic_rate`, `adaptation_rate`, etc.) are tuned for this
/// specific time step. Changing DT without re-tuning those rates
/// will change the system's dynamics.
pub const DT: f32 = 0.1;

/// Global scaling factor for coupling forces. The coupling matrix
/// values represent the theoretical strength of each interaction,
/// but applying them at full strength creates runaway positive
/// feedback loops. This scale factor brings the coupling forces
/// into balance with the homeostatic restoring forces, ensuring
/// the system remains stable while still exhibiting coupled dynamics.
pub const COUPLING_SCALE: f32 = 0.15;

/// Gain applied to the arousal net drive before the sigmoid.
pub const WC_GAIN: f32 = 2.0;

/// Strength of the arousal self-excitation term. The product
/// `WC_GAIN * WC_SELF_EXCITATION` controls the bistability of the
/// sleep-wake flip-flop: values greater than 4.0 create two stable
/// fixed points (awake / asleep); values below 4.0 produce a single
/// stable fixed point.
pub const WC_SELF_EXCITATION: f32 = 2.05;

/// Acetylcholinesterase degradation rate for acetylcholine (ACh).
///
/// ACh is rapidly broken down in the synaptic cleft by
/// acetylcholinesterase (AChE), one of the fastest enzymes in the
/// body (turnover ~25,000 molecules/sec). This happens on a
/// millisecond timescale — far faster than the homeostatic restoring
/// force, which operates on a seconds-to-minutes timescale.
///
/// The degradation applies only to the EXCESS above the homeostatic
/// baseline (not the absolute level). This preserves the biological
/// intent — high ACh bursts decay rapidly — while allowing the
/// homeostatic force to maintain ACh near baseline at rest. See the
/// application site in `tick_with_params` for the full rationale.
///
/// Value 0.03 per tick (at 10 Hz = 0.3/sec): an excess of 0.10 above
/// baseline decays by 3% per tick, reaching baseline in ~3.5 ticks
/// (~0.35 seconds) without homeostatic help. This is compressed from
/// the biological millisecond timescale but preserves the key
/// property: ACh is the fastest-degrading neurotransmitter in the
/// model.
pub const ACH_DEGRADATION_RATE: f32 = 0.03;

/// Cortisol metabolic clearance rate (hepatic 11β-HSD degradation).
///
/// Cortisol is cleared from the bloodstream by the liver via
/// 11β-hydroxysteroid dehydrogenase enzymes. In biology, cortisol
/// has a half-life of ~60–90 minutes, with first-order kinetics.
///
/// **Why this matters for the dynamics:**
///
/// The HPA axis cascade (CRH → ACTH → cortisol) drives cortisol
/// upward, while the homeostatic force pulls it back to baseline.
/// But the coupling forces from arousal chemicals (NE, OX, EPI) and
/// the "removal of suppression" from depleted inhibitory chemicals
/// (SRT, GABA, OXY, END) can sum to ~0.10, overwhelming the
/// homeostatic force (~0.009 at the clamp). Without a clearance
/// mechanism, cortisol pins at the 0.80 hard clamp and can never
/// come down — locking the system into chronic stress.
///
/// **Nonlinear (stress-biased) clearance:**
///
/// A constant clearance rate can't work: it would need to be ~0.009
/// to overcome coupling forces at the clamp, but that same rate
/// would crash cortisol to near-zero at rest (where the ACTH drive
/// is only ~0.0003). Instead, the clearance is proportional to how
/// far cortisol is ABOVE its baseline:
///
/// ```text
/// effective_clearance = CORTISOL_CLEARANCE_RATE * max(0, level - baseline)
/// level *= (1 - effective_clearance)
/// ```
///
/// At rest (level = baseline = 0.0): clearance = 0, no effect ✓
/// Under stress (level = 0.80, baseline = 0.0): clearance = 0.08 * 0.80 = 0.064
///
/// This models the biological upregulation of cortisol-metabolizing
/// enzymes (11β-HSD2) under sustained high cortisol — the liver
/// increases clearance capacity when cortisol is chronically
/// elevated, providing a natural negative feedback that constant-
/// rate clearance cannot capture.
///
/// (Tomlinson et al., 2004 — 11β-HSD and glucocorticoid metabolism;
/// Hellhammer et al., 2009 — cortisol clearance and HPA feedback)
pub const CORTISOL_CLEARANCE_RATE: f32 = 0.08;

/// Cortisol negative feedback gain on ACTH (long-loop HPA feedback).
///
/// High cortisol suppresses ACTH release from the anterior pituitary,
/// which in turn reduces cortisol synthesis. This is the primary
/// negative feedback loop of the HPA axis. The previous gain (0.005)
/// was too weak to balance the CRH→ACTH→cortisol cascade, allowing
/// cortisol to remain elevated indefinitely. The stronger gain (0.02)
/// ensures that sustained high cortisol suppresses its own production
/// on the same timescale as the ACTH response.
///
/// (Herman et al., 2016 — HPA axis negative feedback; de Kloet et al.,
/// 1998 — corticosteroid receptor-mediated feedback)
pub const HPA_CORTISOL_FEEDBACK_RATE: f32 = 0.02;

/// Autoreceptor self-inhibition threshold.
///
/// Autoreceptors (e.g., 5-HT1A for serotonin, D2 for dopamine,
/// alpha-2A for norepinephrine, GABA_B for GABA, M2 for acetylcholine)
/// provide negative feedback that strengthens when neurotransmitter
/// levels deviate far from baseline. Near baseline (|deviation| ≤
/// this threshold), autoreceptor occupancy is low and the restoring
/// force is purely the standard homeostatic pull. Beyond this
/// threshold, autoreceptor-mediated self-inhibition grows
/// quadratically, preventing chemicals from pinning at the 0.0/1.0
/// clamps.
///
/// Set to 0.15 so that moderate deviations needed for emergent phases
/// (NE=0.60 for Stress, DA=0.65 for Flow — deviations of ~0.20-0.25)
/// are only mildly affected, while large deviations (0.35+) engage
/// strong self-inhibition.
pub const AUTORECEPTOR_THRESHOLD: f32 = 0.15;

/// Autoreceptor self-inhibition gain.
///
/// Beyond [`AUTORECEPTOR_THRESHOLD`], the homeostatic restoring force
/// is multiplied by `(1 + AUTORECEPTOR_GAIN * excess²)`, where
/// `excess = |deviation| - threshold`. This creates a quadratic
/// "soft wall" at extremes, modeling the disproportionate increase in
/// autoreceptor-mediated inhibition when neurotransmitter levels
/// deviate far from set-points.
///
/// This is the permanent, biologically grounded nonlinear stabilizer
/// for the coupled neurochemical system. The coupling matrix alone has
/// a spectral radius ~7.6× too large for global linear stability, but
/// emergent phases require strong coupling. Autoreceptors resolve this
/// by leaving near-baseline dynamics (phases) linear while providing
/// the strong self-inhibition that prevents extreme clamping.
///
/// See the homeostatic force computation in `tick_with_params` for
/// the full rationale and calibration.
pub const AUTORECEPTOR_GAIN: f32 = 200.0;

/// Functional rescue: receptor turnover and effective-deviation scaling.
///
/// Two mechanisms prevent permanent receptor burnout traps:
///
/// 1. **Effective-deviation scaling**: receptor downregulation is
///    driven by receptor activation (effective signaling), not by
///    raw ligand concentration. The deviation used for downregulation
///    is scaled by receptor health: `effective_deviation = raw_deviation
///    × sensitivity × desensitization × internalization`. When
///    receptors are burnt out, effective deviation is small, so
///    downregulation is weak. This is biologically correct — receptor
///    downregulation is a response to overstimulation, and burnt-out
///    receptors aren't being overstimulated.
///
/// 2. **Receptor turnover**: a slow background recovery toward 1.0
///    that models the continuous receptor replacement/recycling that
///    happens in biology (receptor half-life is days to weeks).
///    This is always active but normally overwhelmed by deviation-gated
///    downregulation. When downregulation is weak (low effective
///    deviation due to burnout), turnover slowly restores receptor
///    function.
///
///    Each factor's turnover is proportional to its adaptation rate
///    (1.0×, 0.2×, 0.05× for sensitivity, desensitization,
///    internalization respectively). This ensures that:
///    - In normal overstimulation: desens/intern stay near 1.0 (their
///      slow turnover is overwhelmed by downregulation), keeping
///      health high and sensitivity at its floor.
///    - In extreme burnout: desens/intern recover slowly enough that
///      health stays low, allowing sensitivity turnover to overcome
///      the weak (health-scaled) downregulation.
///
/// These mechanisms are always active — no thresholds, no recovery
/// zones, no limit cycles. The system smoothly transitions between
/// normal adaptation and recovery as receptor health and deviation
/// change.
pub const RECEPTOR_TURNOVER_RATE: f32 = 0.0001;

// ─────────────────────────────────────────────────────────────────
//  Chemical identifiers
// ─────────────────────────────────────────────────────────────────

/// Identifier for each tracked neurochemical.
///
/// Discrimant values are stable — never reorder or renumber.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum NeurochemicalId {
    /// Dopamine — reward, motivation, novelty; D1/D2 receptor subtypes.
    Dopamine = 0,
    /// Serotonin — mood stability, wellbeing; enables BDNF recovery.
    Serotonin = 1,
    /// Norepinephrine — arousal, vigilance, active data work.
    Norepinephrine = 2,
    /// Acetylcholine — attention, focus, cortical arousal.
    Acetylcholine = 3,
    /// GABA — primary inhibitory neurotransmitter; calms glutamate.
    GABA = 4,
    /// Glutamate — primary excitatory neurotransmitter; cortical drive.
    Glutamate = 5,
    /// Cortisol — stress hormone; acute distress, chronic allostatic load.
    Cortisol = 6,
    /// Oxytocin — social bonding, dyadic attunement, trust.
    Oxytocin = 7,
    /// Endorphin — endogenous opioid; pain relief, disinhibits dopamine.
    Endorphin = 8,
    /// Histamine — wakefulness, inflammatory signaling.
    Histamine = 9,
    /// Adenosine — sleep pressure; accumulates during wakefulness.
    Adenosine = 10,
    /// BDNF — brain-derived neurotrophic factor; gates plasticity.
    BDNF = 11,
    /// Endocannabinoid — retrograde signaling, stress buffering.
    Endocannabinoid = 12,
    /// Vasopressin — stress response, social recognition, water balance.
    Vasopressin = 13,
    /// CRH — corticotropin-releasing hormone; HPA axis cascade trigger.
    CRH = 14,
    /// Orexin — wakefulness maintenance, metabolic arousal.
    Orexin = 15,
    /// Epinephrine — adrenaline; acute stress, sympathetic activation.
    Epinephrine = 16,
    /// Melatonin — circadian sleep signal; drives powersave mode.
    Melatonin = 17,
}

impl NeurochemicalId {
    /// All neurochemical IDs in canonical (discriminant) order.
    pub const fn all() -> [NeurochemicalId; NEUROCHEMICAL_COUNT] {
        [
            Self::Dopamine,
            Self::Serotonin,
            Self::Norepinephrine,
            Self::Acetylcholine,
            Self::GABA,
            Self::Glutamate,
            Self::Cortisol,
            Self::Oxytocin,
            Self::Endorphin,
            Self::Histamine,
            Self::Adenosine,
            Self::BDNF,
            Self::Endocannabinoid,
            Self::Vasopressin,
            Self::CRH,
            Self::Orexin,
            Self::Epinephrine,
            Self::Melatonin,
        ]
    }

    /// Returns the lowercase canonical name of this neurochemical.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dopamine => "dopamine",
            Self::Serotonin => "serotonin",
            Self::Norepinephrine => "norepinephrine",
            Self::Acetylcholine => "acetylcholine",
            Self::GABA => "gaba",
            Self::Glutamate => "glutamate",
            Self::Cortisol => "cortisol",
            Self::Oxytocin => "oxytocin",
            Self::Endorphin => "endorphin",
            Self::Histamine => "histamine",
            Self::Adenosine => "adenosine",
            Self::BDNF => "bdnf",
            Self::Endocannabinoid => "endocannabinoid",
            Self::Vasopressin => "vasopressin",
            Self::CRH => "crh",
            Self::Orexin => "orexin",
            Self::Epinephrine => "epinephrine",
            Self::Melatonin => "melatonin",
        }
    }

    /// Convert a u8 index back to the enum.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Dopamine,
            1 => Self::Serotonin,
            2 => Self::Norepinephrine,
            3 => Self::Acetylcholine,
            4 => Self::GABA,
            5 => Self::Glutamate,
            6 => Self::Cortisol,
            7 => Self::Oxytocin,
            8 => Self::Endorphin,
            9 => Self::Histamine,
            10 => Self::Adenosine,
            11 => Self::BDNF,
            12 => Self::Endocannabinoid,
            13 => Self::Vasopressin,
            14 => Self::CRH,
            15 => Self::Orexin,
            16 => Self::Epinephrine,
            17 => Self::Melatonin,
            _ => Self::Dopamine,
        }
    }

    /// Default homeostatic baseline (the genetically determined set-point).
    ///
    /// Baselines are grounded in measured extracellular concentrations
    /// from microdialysis studies (Garg et al., 2020; Benveniste et al.,
    /// 1984; Phillips et al., 1994) and represent the tonic activity
    /// level of each system:
    ///
    /// - **Glutamate** (0.60): Highest baseline — main excitatory
    ///   transmitter, ~2 µM extracellular, 100-1000× more concentrated
    ///   than monoamines. Always active.
    /// - **GABA** (0.50): Second highest — main inhibitory transmitter,
    ///   ~200 nM extracellular. Always active. The GLU/GABA balance
    ///   is the primary E/I axis.
    /// - **BDNF** (0.45): Constitutively expressed neurotrophin,
    ///   ~10 ng/mL in serum. Supports plasticity at a moderate tonic
    ///   level.
    /// - **Serotonin** (0.40): Tonic 5-HT is ~10 nM but is a potent
    ///   modulator. Oscillates with ~10 min periods (FSCAV data).
    ///   Regulates mood, sleep, and BDNF expression.
    /// - **Dopamine** (0.35): Tonic DA is very low (~0.06 nM in PFC,
    ///   ~5 nM in VTA) but phasic bursts are large. The baseline
    ///   reflects tonic firing rate, not concentration. Spontaneous
    ///   impulses at ~0.01/sec (Mohebi et al., 2019).
    /// - **Acetylcholine** (0.35): Tonic ACh ~1 nM in PFC. Sets
    ///   cortical activation level. Co-fluctuates with DA at ~2 Hz
    ///   (Howe et al., 2023).
    /// - **Norepinephrine** (0.30): Tonic NE ~0.13 nM in PFC. Sets
    ///   arousal level. LC broadcasts NE (and co-releases DA) broadly.
    /// - **Orexin** (0.30): Picomolar range, but tonically active
    ///   during wakefulness. Stabilizes the sleep-wake flip-flop.
    /// - **Histamine** (0.30): Wake-promoting, low nM range. TMN
    ///   histamine neurons fire during wakefulness.
    /// - **Oxytocin** (0.25): Very low tonic (~10 pM in plasma).
    ///   Released in bursts with social bonding. Potent modulator
    ///   despite low concentration.
    /// - **Endorphin** (0.25): Very low tonic (~5-15 fmol/mL).
    ///   Released in bursts with pleasure/pain. Potent modulator.
    /// - **Endocannabinoid** (0.25): Synthesized on demand from
    ///   postsynaptic activity. Low tonic, retrograde signaling.
    /// - **Adenosine** (0.20): ~30 nM awake, rises with sleep pressure.
    ///   Metabolic byproduct, accumulates during wakefulness.
    /// - **Vasopressin** (0.20): Similar to oxytocin, low tonic.
    /// - **Epinephrine** (0.15): Very low unless sympathetically
    ///   activated. Adrenal medulla release.
    /// - **Melatonin** (0.10): Driven by circadian oscillator, near
    ///   zero during day, high at night.
    /// - **Cortisol** (0.0): Stress hormone, zero at rest.
    /// - **CRH** (0.0): Stress hormone, zero at rest.
    pub const fn default_baseline(self) -> f32 {
        match self {
            Self::Glutamate => 0.60,       // ~2 µM — main excitatory, highest
            Self::GABA => 0.50,            // ~200 nM — main inhibitory
            Self::BDNF => 0.45,            // constitutive neurotrophin
            Self::Serotonin => 0.40,       // ~10 nM — modulatory, oscillating
            Self::Dopamine => 0.35,        // ~0.06 nM tonic — low but potent
            Self::Acetylcholine => 0.35,   // ~1 nM — cortical activation
            Self::Norepinephrine => 0.30,  // ~0.13 nM — arousal
            Self::Orexin => 0.30,          // picomolar — wake stabilization
            Self::Histamine => 0.30,       // low nM — wake promotion
            Self::Oxytocin => 0.25,        // ~10 pM — social bonding
            Self::Endorphin => 0.25,       // ~5-15 fmol/mL — pleasure/pain
            Self::Endocannabinoid => 0.25, // on-demand synthesis
            Self::Adenosine => 0.20,       // ~30 nM — sleep pressure
            Self::Vasopressin => 0.20,     // low tonic peptide
            Self::Epinephrine => 0.15,     // very low unless stressed
            Self::Melatonin => 0.10,       // circadian-driven
            Self::Cortisol => 0.0,         // stress hormone — zero at rest
            Self::CRH => 0.0,              // stress hormone — zero at rest
        }
    }

    /// Default receptor sensitivity at birth (1.0 = normal).
    pub const fn default_sensitivity(self) -> f32 {
        1.0
    }

    /// Multiplier for the global receptor adaptation rate.
    ///
    /// Different neuromodulator systems adapt at different speeds:
    /// dopaminergic and cholinergic receptors traffic rapidly; cortisol
    /// (genomic) and BDNF/TrkB (protein expression) adapt much more
    /// slowly. This multiplier scales the base `adaptation_rate` for
    /// the three receptor-adaptation mechanisms in each tick.
    ///
    /// Values are conservative (within [0.4, 1.2]) so the dynamics
    /// remain stable while still capturing the real kinetic spread
    /// across receptor systems.
    pub const fn receptor_adaptation_multiplier(self) -> f32 {
        match self {
            Self::Dopamine => 1.2,       // D2/D3 receptors traffic rapidly
            Self::Serotonin => 0.9,      // 5-HT1A/2A: moderate trafficking
            Self::Norepinephrine => 1.1, // β-AR desensitizes quickly
            Self::Acetylcholine => 1.2,  // nicotinic receptors desensitize in ms
            Self::GABA => 0.9,           // GABA-A receptor turnover is moderate
            Self::Glutamate => 1.0,      // AMPA/NMDA: baseline reference
            Self::Cortisol => 0.5,       // GR/MR effects are genomic and slow
            Self::Oxytocin => 0.7,       // OTR trafficking is moderate-slow
            Self::Endorphin => 0.9,      // MOR internalization is moderate
            Self::Histamine => 1.0,
            Self::Adenosine => 1.0,
            Self::BDNF => 0.3, // TrkB expression and BDNF synthesis are slow
            Self::Endocannabinoid => 1.0,
            Self::Vasopressin => 0.6, // V1a/V1b trafficking is slow
            Self::CRH => 0.5,         // CRHR1/2 signaling and trafficking are slow
            Self::Orexin => 0.9,
            Self::Epinephrine => 1.1, // similar to NE
            Self::Melatonin => 0.4,   // MT1/MT2 are slow, hormone-mediated
        }
    }

    /// Default receptor subtype balance [subtype_a, subtype_b].
    ///
    /// For dopamine: [D1 (excitatory, Gs-coupled), D2 (inhibitory, Gi-coupled)]
    /// For serotonin: [5-HT1A (anxiolytic), 5-HT2A (hallucinogenic)]
    /// For orexin: [OX1R (excitatory), OX2R (wake-promoting)]
    /// For epinephrine: [alpha-AR (vasoconstriction), beta-AR (cardiac)]
    /// For melatonin: [MT1 (sleep onset), MT2 (circadian phase-shift)]
    /// For all others: [1.0, 1.0] (no subtype differentiation modeled)
    pub const fn default_receptor_subtypes(self) -> [f32; 2] {
        match self {
            Self::Dopamine => [0.50, 0.50],    // D1, D2 balance
            Self::Serotonin => [0.50, 0.50],   // 5-HT1A, 5-HT2A balance
            Self::Orexin => [0.50, 0.50],      // OX1R, OX2R balance
            Self::Epinephrine => [0.50, 0.50], // alpha-AR, beta-AR balance
            Self::Melatonin => [0.50, 0.50],   // MT1, MT2 balance
            _ => [1.0, 1.0],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Single neurochemical state
// ─────────────────────────────────────────────────────────────────

/// A single neurochemical's dynamic state.
///
/// ## Layout (56 bytes)
/// ```text
/// offset  field                  type      notes
/// ------  -----                  ----      -----
///   0     last_shift_at          u64       ms timestamp of last significant shift
///   8     level                  f32       current concentration [0.0, 1.0]
///  12     baseline               f32       homeostatic set-point (adapts slowly)
///  16     receptor_sensitivity   f32       how strongly effects are felt [0.5, 2.0]
///  20     velocity               f32       rate of change (momentum)
///  24     receptor_subtypes      [f32; 2]  subtype balance (D1/D2, 5-HT1A/5-HT2A)
///  32     desensitization_factor f32       fast phosphorylation-based adaptation
///  36     internalization_factor f32       slow membrane-removal adaptation
///  40     tonic_level            f32       background sustained level (slow-changing)
///  44     phasic_level           f32       burst-firing transient (fast-decaying)
///  48     vesicular_pool         f32       vesicle pool fullness [0.0, 1.0]
///  52     id                     u8        NeurochemicalId discriminant
///  53     _pad                   [u8; 3]   alignment to 56
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Neurochemical {
    /// Milliseconds-timestamp of the last significant level shift.
    pub last_shift_at: u64,
    /// Current concentration, in `[0.0, 1.0]`.
    pub level: f32,
    /// Homeostatic set-point (adapts slowly).
    pub baseline: f32,
    /// Receptor sensitivity — how strongly effects are felt, in `[0.5, 2.0]`.
    pub receptor_sensitivity: f32,
    /// Rate of change (momentum) of the level.
    pub velocity: f32,
    /// Receptor subtype balance [subtype_a, subtype_b].
    /// For dopamine: [D1 (excitatory), D2 (inhibitory)]
    /// For serotonin: [5-HT1A (anxiolytic), 5-HT2A (hallucinogenic)]
    /// For orexin: [OX1R (excitatory), OX2R (wake-promoting)]
    /// For epinephrine: [alpha-AR, beta-AR]
    /// For others: [1.0, 1.0] (no subtype differentiation)
    pub receptor_subtypes: [f32; 2],
    /// Fast desensitization factor (phosphorylation-based, reversible in
    /// minutes). Range [0.2, 1.0] where 1.0 = no desensitization, 0.2 =
    /// maximally desensitized (~80% loss, per MOR/β2AR literature).
    /// Responds quickly to high levels.
    pub desensitization_factor: f32,
    /// Slow internalization factor (membrane removal, reversible in hours).
    /// Range [0.45, 1.0] where 1.0 = no internalization, 0.45 = maximally
    /// internalized (~55% loss, per β2AR/MOR literature). Responds to
    /// sustained high levels.
    pub internalization_factor: f32,
    /// Tonic (background, sustained) level. Changes slowly, representing
    /// baseline firing rate. For dopamine, tonic level activates
    /// high-affinity D2 receptors (mood maintenance). For serotonin,
    /// tonic level activates high-affinity 5-HT1A receptors (anxiolytic).
    /// Range [0.0, 1.0], initialized to the default baseline.
    pub tonic_level: f32,
    /// Phasic (burst-firing, transient) level. Decays quickly (seconds),
    /// representing burst firing events. For dopamine, phasic bursts
    /// activate low-affinity D1 receptors (reward signaling). For
    /// serotonin, phasic bursts activate low-affinity 5-HT2A receptors
    /// (excitatory). Range [0.0, 1.0], initialized to 0.0.
    pub phasic_level: f32,
    /// Vesicular pool fullness [0.0, 1.0]. Represents the readily
    /// releasable vesicle pool. On impulse (rapid level increase), the
    /// pool is depleted. When the pool is low, impulses have reduced
    /// effect (short-term synaptic depression). Synthesis and reuptake
    /// slowly replenish the pool. This prevents infinite sustained high
    /// levels — the system has a finite neurotransmitter supply.
    pub vesicular_pool: f32,
    /// `NeurochemicalId` discriminant identifying which chemical this is.
    pub id: u8,
    /// Alignment padding to reach 56 bytes.
    pub _pad: [u8; 3],
}

impl Neurochemical {
    /// Create a neurochemical at its default baseline, zero velocity,
    /// normal receptor sensitivity.
    pub fn new(id: NeurochemicalId, now_ms: u64) -> Self {
        Self {
            last_shift_at: now_ms,
            level: id.default_baseline(),
            baseline: id.default_baseline(),
            receptor_sensitivity: id.default_sensitivity(),
            velocity: 0.0,
            receptor_subtypes: id.default_receptor_subtypes(),
            desensitization_factor: 1.0,
            internalization_factor: 1.0,
            tonic_level: id.default_baseline(),
            phasic_level: 0.0,
            vesicular_pool: 1.0,
            id: id as u8,
            _pad: [0; 3],
        }
    }

    /// The **effective level** — what consumers of neurochemical state
    /// actually see. This is `level × receptor_sensitivity ×
    /// desensitization_factor × internalization_factor`, clamped to
    /// [0.0, 2.0].
    ///
    /// This is the key insight from pharmacology: a chemical can be at
    /// high concentration but have low effect (tolerance/downregulation)
    /// or low concentration but high effect (sensitization/upregulation).
    ///
    /// The effective level now incorporates two separate receptor
    /// adaptation mechanisms:
    /// - **Desensitization** (fast, phosphorylation-based): reduces effect
    ///   within minutes of high stimulation
    /// - **Internalization** (slow, membrane removal): reduces effect over
    ///   hours of sustained high stimulation
    ///
    /// Both factors are in [0.2, 1.0] (desensitization) and [0.45, 1.0]
    /// (internalization), and multiply together with
    /// receptor_sensitivity [0.5, 2.0] to produce the final effective
    /// level.
    pub fn effective_level(&self) -> f32 {
        let raw = self.level
            * self.receptor_sensitivity
            * self.desensitization_factor
            * self.internalization_factor;
        // Use finite_clamp instead of native clamp: if any factor
        // is NaN (from corruption or a computation bug), the product
        // is NaN, and native clamp would pass it through.
        crate::state::sanitize::finite_clamp(raw, 0.0, 2.0)
    }

    /// The **subtype-weighted effective level** — considers receptor
    /// subtype balance. For dopamine, D1 (excitatory) and D2 (inhibitory)
    /// have opposing effects. For serotonin, 5-HT1A (anxiolytic) and
    /// 5-HT2A (hallucinogenic) have different downstream effects.
    ///
    /// The subtype balance modifies the effective level: when subtype_a
    /// dominates, the "excitatory/anxiolytic" pathway is emphasized;
    /// when subtype_b dominates, the "inhibitory/hallucinogenic"
    /// pathway is emphasized. The net effect is a weighted combination.
    pub fn subtype_effective_level(&self) -> f32 {
        let base = self.effective_level();
        let id = NeurochemicalId::from_u8(self.id);
        match id {
            NeurochemicalId::Dopamine => {
                // D1 (excitatory) increases effective signaling,
                // D2 (inhibitory) decreases it. The net effect is
                // weighted by subtype balance.
                //
                // The formula `base * (d1 - d2) / total * 2.0 + base`
                // can produce values up to ~6.0 when base=2.0 and
                // d1=1, d2=0 (2.0 * 1.0 * 2.0 + 2.0 = 6.0). This
                // breaks the [0, 2] effective-level contract and can
                // produce inf when multiplied into the coupling
                // matrix. Clamp the result to [0, 2] to match the
                // serotonin branch (which already clamps) and the
                // documented contract.
                let d1 = self.receptor_subtypes[0];
                let d2 = self.receptor_subtypes[1];
                let total = d1 + d2;
                if total > 0.0 {
                    let raw = base * (d1 - d2) / total * 2.0 + base;
                    crate::state::sanitize::finite_clamp(raw, 0.0, 2.0)
                } else {
                    base
                }
            }
            NeurochemicalId::Serotonin => {
                // 5-HT1A (anxiolytic) promotes calm signaling,
                // 5-HT2A (hallucinogenic) promotes excitatory signaling.
                // The net effect is neutral at a 50/50 balance, biased
                // up or down by the relative receptor weights.
                let ht1a = self.receptor_subtypes[0];
                let ht2a = self.receptor_subtypes[1];
                let total = ht1a + ht2a;
                if total > 0.0 {
                    let balance = (ht1a - ht2a) / total;
                    // Use finite_clamp instead of native clamp to
                    // catch NaN from corrupted receptor_subtypes.
                    crate::state::sanitize::finite_clamp(base * (1.0 + 0.3 * balance), 0.0, 2.0)
                } else {
                    base
                }
            }
            _ => base,
        }
    }

    /// Apply an impulse (external event) to the level.
    ///
    /// `magnitude` is in normalised units; positive increases, negative
    /// decreases. Velocity is updated to model momentum.
    ///
    /// # Vesicular pool depletion
    ///
    /// On a positive impulse (release event), the vesicular pool is
    /// depleted proportional to the magnitude. When the pool is low,
    /// the impulse effect is reduced (short-term synaptic depression).
    /// This prevents infinite sustained high levels — the system has
    /// a finite neurotransmitter supply that must be replenished by
    /// synthesis (slow) and reuptake (moderate).
    ///
    /// # Phasic burst
    ///
    /// Positive impulses also trigger a phasic burst — a transient
    /// increase in phasic_level that decays quickly. This models
    /// burst firing in dopaminergic and serotonergic neurons, where
    /// salient events cause transient high-frequency firing that
    /// activates low-affinity receptors (D1, 5-HT2A) distinct from
    /// the tonic baseline that activates high-affinity receptors
    /// (D2, 5-HT1A).
    pub fn apply_impulse(&mut self, magnitude: f32, now_ms: u64) {
        // Defense-in-depth: sanitize the magnitude before it touches
        // any chemical field. Even though IPC entry points now reject
        // NaN, `apply_impulse` is also called from:
        // - `active_inference::apply_inference_feedback` (impulses
        //   computed from the transition matrix, which may be
        //   corrupted)
        // - `dyadic_model::DyadicAffectModel::update` (impulses
        //   computed from user affect, which may be NaN if the
        //   dyadic model's internal state was corrupted)
        // - `interoception::Interoceptor::neuro_impulses` (computed
        //   from hardware sensors — unlikely but not impossible)
        //
        // A NaN magnitude here would make `level` and `velocity` NaN,
        // which would propagate through the coupling matrix to all 18
        // chemicals in the next tick. Replacing NaN with 0.0 produces
        // a no-op impulse — the chemical is unchanged, which is the
        // safest possible behavior for an invalid input.
        let magnitude = crate::state::sanitize::finite_or(magnitude, 0.0);

        // Vesicular pool: positive impulses deplete the pool, reducing
        // the effective magnitude when the pool is low (synaptic depression).
        let effective_magnitude = if magnitude > 0.0 {
            // Use finite_clamp for NaN safety: if vesicular_pool was
            // corrupted to NaN, native .clamp returns NaN, which would
            // then propagate to effective_magnitude and level.
            let pool_factor = crate::state::sanitize::finite_clamp(self.vesicular_pool, 0.0, 1.0);
            // Deplete the pool proportional to the impulse magnitude
            self.vesicular_pool = crate::state::sanitize::finite_clamp(
                self.vesicular_pool - magnitude.abs() * 0.3,
                0.0,
                1.0,
            );
            magnitude * pool_factor
        } else {
            magnitude
        };

        self.velocity += effective_magnitude * 0.3;
        // Clamp velocity to a bounded range. Without this, a large
        // impulse (even after IPC-level clamping, internal callers like
        // active_inference or dyadic_model could produce large values)
        // would set velocity to an enormous value that decays
        // geometrically over hundreds of ticks, keeping the chemical
        // saturated and distorting downstream dynamics. The range
        // [-1, 1] is generous — normal impulses produce velocity
        // increments of ~0.3, and the damping factor (0.85^dt_scale)
        // brings it back to zero within a few ticks.
        self.velocity = crate::state::sanitize::finite_clamp(self.velocity, -1.0, 1.0);
        // Use finite_clamp for NaN safety on level.
        self.level =
            crate::state::sanitize::finite_clamp(self.level + effective_magnitude, 0.0, 1.0);

        // Phasic burst: positive impulses trigger a transient burst
        // that activates low-affinity receptors (D1 for DA, 5-HT2A for SRT)
        if effective_magnitude > 0.0 {
            self.phasic_level = crate::state::sanitize::finite_clamp(
                self.phasic_level + effective_magnitude * 0.5,
                0.0,
                1.0,
            );
        }

        if magnitude.abs() > 0.05 {
            self.last_shift_at = now_ms;
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Coupling matrix
// ─────────────────────────────────────────────────────────────────

/// The 18×18 neurochemical coupling matrix.
///
/// `coupling_matrix[i][j]` defines how chemical j influences chemical i's
/// rate of change. Positive = excitatory, negative = inhibitory, zero =
/// no direct coupling. The diagonal is zero (self-dynamics are handled
/// by homeostatic drift, not self-coupling).
///
/// ## Grounding in real neuroscience
///
/// These values are derived from known neurotransmitter interactions:
/// - Serotonin inhibits dopamine (mesolimbic pathway)
/// - Cortisol suppresses BDNF and serotonin (stress-damage mechanism)
/// - GABA inhibits glutamate (primary inhibitory/excitatory balance)
/// - Adenosine's sleep pressure acts via level accumulation/clearance, not direct coupling
/// - Endorphins disinhibit dopamine (runner's high mechanism)
/// - Oxytocin dampens cortisol (social bonding reduces stress)
/// - BDNF is promoted by serotonin (why SSRIs help depression)
/// - Endocannabinoids retrogradely inhibit GABA and glutamate
/// - Vasopressin synergizes with oxytocin and stimulates cortisol
/// - CRH triggers cortisol release (HPA axis upstream)
/// - Cortisol negatively feeds back to CRH
/// - Orexin mutually excites histamine and promotes arousal (NE, DA, ACh)
/// - Orexin is inhibited by adenosine and GABA (sleep pressure)
/// - Epinephrine is stimulated by sympathetic activity (NE, ACh, CRH)
/// - Epinephrine promotes cardiovascular arousal (cortisol, NE)
/// - Melatonin inhibits dopamine (circadian mood modulation)
/// - Melatonin inhibits histamine (sleep promotion via wake-center suppression)
/// - Melatonin promotes GABA (sleep facilitation)
/// - Melatonin inhibits orexin (sleep-wake switch toward sleep)
///
/// ## Metaplasticity
///
/// The coupling matrix is not static. It lives in the mmap'd state and
/// can be modified by the dynamics engine over time. This represents
/// neurochemical-level learning — the brain rewiring its own
/// neurotransmitter interactions based on experience.
pub type CouplingMatrix = [[f32; NEUROCHEMICAL_COUNT]; NEUROCHEMICAL_COUNT];

/// The default coupling matrix, grounded in neuroscience literature.
///
/// Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST, ADN,
/// BDNF, ECB, VP, CRH, OX (orexin), EPI (epinephrine), MEL (melatonin)
///
/// ## Research grounding
///
/// The coupling strengths are grounded in the neuroscience literature on
/// neuromodulatory volume transmission (Özçete et al., Molecular
/// Psychiatry, 2024) and computational models of neurotransmitter
/// dynamics (Deco et al., PNAS, 2021; Hellyer et al., 2025).
///
/// Key principles:
/// - **Monoamines (DA, SRT, NE, ACh) interact weakly** (0.05) through
///   volume transmission — they diffuse broadly and have modulatory,
///   not strong, effects on each other.
/// - **GLU↔GABA is the strong E/I axis** (-0.30 GABA→GLU, +0.10 GLU→GABA).
///   This asymmetric negative feedback loop stabilizes excitation-inhibition
///   balance: GLU rises → drives GABA up (feedback inhibition) → GABA
///   suppresses GLU. This is the primary dynamic axis of the brain.
/// - **Cortisol and CRH have NO positive couplings from arousal chemicals.**
///   They are produced only by the HPA cascade (maturation-gated). The
///   coupling matrix only has suppressive feedback (SRT, GABA, OXY, END,
///   BDNF, ECB → CORT/CRH negative).
/// - **Peptides (OXY, END, VP) interact weakly** (0.05) — they are potent
///   but very low concentration (~10 pM oxytocin, ~5-15 fmol/mL endorphin).
/// - **BDNF is supported by monoamines** (especially SRT, +0.10, the
///   antidepressant mechanism) **and suppressed by cortisol** (-0.20).
/// - **All positive feedback loops are weak** (0.05) to prevent runaway
///   dynamics. With COUPLING_SCALE=0.15, effective coupling is 0.0075,
///   well below the homeostatic rate of 0.04.
pub const DEFAULT_COUPLING_MATRIX: CouplingMatrix = [
    // DA  →  how others affect dopamine
    // END→DA (+0.10): endorphin disinhibition of DA (mu-opioid → GABA
    //   interneuron → DA disinhibition). Strongest positive coupling
    //   for DA, appropriate for runner's high.
    // GABA→DA (-0.10): GABAergic inhibition of VTA DA neurons.
    // ADN→DA (0.00): adenosine's effect on dopamine is handled by the
    //   sleep pressure mechanism (level accumulation/clearance), not direct coupling.
    // MEL→DA (-0.05): sleep suppression of dopaminergic arousal.
    // CORT→DA (0.00): cortisol does not directly promote dopamine.
    // All other couplings weak (0.05) — volume transmission.
    [
        0.00, -0.05, 0.05, 0.05, -0.10, 0.05, 0.00, 0.05, 0.10, 0.05, 0.00, 0.05, 0.05, 0.00, 0.00,
        0.05, 0.00, -0.05,
    ],
    // SRT →  how others affect serotonin
    // BDNF→SRT (+0.10): the antidepressant positive feedback — SSRIs
    //   raise SRT → raise BDNF → raise SRT. Bounded by homeostatic forces.
    // CORT→SRT (-0.10): chronic stress depletes serotonin.
    // All other couplings weak (0.05) — volume transmission.
    [
        -0.05, 0.00, 0.05, 0.00, -0.05, 0.05, -0.10, 0.05, 0.05, 0.00, 0.00, 0.10, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00,
    ],
    // NE  →  how others affect norepinephrine
    // GABA→NE (-0.10): GABAergic inhibition of locus coeruleus.
    // Arousal mutual excitation (HIST, OX) kept weak (0.05) — these are
    //   parallel arousal systems, not a positive feedback chain.
    // OXY→NE (-0.05): social bonding calms noradrenergic arousal.
    // END→NE (-0.05): endorphin calming effect.
    // MEL→NE (-0.05): sleep suppression of noradrenergic arousal.
    [
        0.05, -0.05, 0.00, 0.05, -0.10, 0.05, 0.00, -0.05, -0.05, 0.05, 0.00, 0.05, 0.00, 0.00,
        0.00, 0.05, 0.05, -0.05,
    ],
    // ACh →  how others affect acetylcholine
    // GABA→ACh (-0.10): inhibition of cholinergic basal forebrain.
    // DA↔ACh (+0.05 each way): co-fluctuation at ~2 Hz (Howe et al., 2023).
    // OX→ACh (+0.05): orexin-driven cortical activation.
    // MEL→ACh (-0.05): sleep suppression of cholinergic arousal.
    [
        0.05, 0.00, 0.05, 0.00, -0.10, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00,
        0.05, 0.00, -0.05,
    ],
    // GABA → how others affect GABA
    // GLU→GABA (+0.10): activity-driven feedback inhibition — when GLU
    //   rises, it drives GABAergic interneurons to fire, providing
    //   automatic gain control. This is the core E/I balance mechanism.
    // END→GABA (+0.10): endorphin promotes GABA (analgesic relaxation).
    // CORT→GABA (-0.10): stress disinhibition (cortisol suppresses GABA).
    // ADN→GABA (+0.05): sleep pressure promotes inhibition.
    // MEL→GABA (+0.05): sleep facilitation.
    // ECB→GABA (-0.05): depolarization-induced suppression of inhibition.
    [
        -0.05, 0.05, -0.05, -0.05, 0.00, 0.10, -0.10, 0.05, 0.10, -0.05, 0.05, 0.05, -0.05, 0.00,
        0.00, -0.05, -0.05, 0.05,
    ],
    // GLU →  how others affect glutamate
    // GABA→GLU (-0.30): THE primary inhibitory brake. This is the
    //   strongest coupling in the matrix, reflecting that GABA is the
    //   brain's main mechanism for controlling glutamatergic excitation.
    //   Glutamate is ~1000× more concentrated than monoamines; its
    //   regulation requires strong inhibition.
    // All other couplings weak (0.05) — modulatory, not driving.
    [
        0.05, -0.05, 0.05, 0.05, -0.30, 0.00, 0.00, -0.05, -0.05, 0.05, 0.00, 0.05, -0.05, 0.00,
        0.00, 0.05, 0.05, -0.05,
    ],
    // CORT → how others affect cortisol
    // ALL positive couplings removed. Cortisol is produced ONLY by the
    // HPA cascade (CRH→ACTH→CORT), which is maturation-gated (SHRP).
    // The coupling matrix only has suppressive feedback — the signals
    // that tell cortisol to stop (serotonin, GABA, oxytocin, endorphin,
    // BDNF, ECB). This prevents the death spiral where normal arousal
    // drives cortisol through the coupling matrix, bypassing the HPA gate.
    [
        0.00, -0.10, 0.00, 0.00, -0.15, 0.00, 0.00, -0.10, -0.10, 0.00, 0.00, -0.05, -0.05, 0.00,
        0.00, 0.00, 0.00, 0.00,
    ],
    // OXY →  how others affect oxytocin
    // CORT→OXY (-0.10): stress suppresses social bonding.
    // VP→OXY (+0.05): reciprocal social neuropeptide interaction.
    // END→OXY (+0.05): opioid-oxytocin interaction (weak to prevent
    //   mutual collapse feedback loop).
    // All other couplings weak (0.05) — volume transmission.
    [
        0.05, 0.05, -0.05, 0.00, 0.05, 0.00, -0.10, 0.00, 0.05, 0.00, 0.00, 0.05, 0.00, 0.05, 0.00,
        0.00, 0.00, 0.00,
    ],
    // END →  how others affect endorphin
    // CORT→END (-0.10): stress suppresses endogenous opioids.
    // OXY→END (+0.05): opioid-oxytocin interaction (weak).
    // GABA→END (+0.05): relaxation promotes endorphin.
    // GLU→END (-0.05): excitatory drive reduces endorphin.
    [
        0.05, 0.05, -0.05, 0.00, 0.05, -0.05, -0.10, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00,
    ],
    // HIST → how others affect histamine
    // GABA→HIST (-0.10): inhibition of histaminergic tuberomammillary
    //   nucleus.
    // MEL→HIST (-0.10): sleep suppression of wake center.
    // Arousal mutual excitation (NE, OX) kept weak (0.05).
    [
        0.05, -0.05, 0.05, 0.05, -0.10, 0.05, 0.00, 0.00, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.05, 0.00, -0.10,
    ],
    // ADN →  how others affect adenosine
    // Adenosine's interactions with arousal systems are handled by the
    // sleep pressure mechanism (level accumulation/clearance), NOT by the coupling
    // matrix. Only GABA (promotes sleep) and cortisol (stress-related
    // adenosine release) have direct coupling. Orexin mildly inhibits
    // adenosine (wake promotion).
    [
        0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        -0.05, 0.00, 0.00,
    ],
    // BDNF → how others affect BDNF
    // SRT→BDNF (+0.10): the antidepressant mechanism — serotonin is the
    //   strongest BDNF supporter. SSRIs raise SRT → raise BDNF →
    //   neuroplasticity → depression recovery.
    // CORT→BDNF (-0.20): chronic stress suppresses BDNF — THE key
    //   mechanism by which stress impairs learning and memory.
    // All other couplings weak (0.05) — trophic support.
    [
        0.05, 0.10, 0.05, 0.05, 0.00, 0.05, -0.20, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00,
    ],
    // ECB → how others affect endocannabinoids
    // Endocannabinoids are synthesized on demand from postsynaptic
    // activity. DA (reward), GABA/GLU (neural activity), and cortisol
    // (stress dampening) all stimulate ECB release.
    [
        0.05, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00,
    ],
    // VP → how others affect vasopressin
    // OXY→VP (+0.10): reciprocal social neuropeptide bonding.
    // CORT→VP (+0.05): stress activates vasopressin (HPA co-activation).
    // EPI→VP (+0.05): sympathetic co-activation.
    [
        0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.05, 0.00,
    ],
    // CRH → how others affect CRH
    // ALL positive couplings removed. CRH is produced only by the stress
    // detection mechanism (emotional regulator) and the HPA cascade.
    // Only suppressive feedback: GABA, cortisol (negative feedback),
    // oxytocin, ECB.
    [
        0.00, 0.00, 0.00, 0.00, -0.05, 0.00, -0.10, -0.05, 0.00, 0.00, 0.00, 0.00, -0.05, 0.00,
        0.00, 0.00, 0.00, 0.00,
    ],
    // OX (orexin) → how others affect orexin
    // GABA→OX (-0.10): VLPO sleep-promoting neurons inhibit orexin.
    // ADN→OX (-0.10): sleep pressure inhibits orexin (the sleep switch).
    // Arousal mutual excitation (NE, HIST) kept weak (0.05).
    // DA→OX (0.00): dopamine does not directly promote orexin — the
    //   DA→OX→DA positive feedback loop caused runaway overstimulation.
    //   Orexin is driven by metabolic state, not reward.
    // SRT→OX (-0.05): serotonin promotes sleep onset.
    // MEL→OX (-0.05): melatonin pushes the sleep-wake switch toward sleep.
    // ECB→OX (-0.05): stress reduction.
    [
        0.00, -0.05, 0.05, 0.05, -0.10, 0.05, 0.00, 0.00, 0.00, 0.05, -0.10, 0.00, -0.05, 0.05,
        0.00, 0.00, 0.05, -0.05,
    ],
    // EPI (epinephrine) → how others affect epinephrine
    // ACh→EPI (+0.10): preganglionic cholinergic stimulation of adrenal
    //   medulla — the strongest coupling for EPI.
    // NE→EPI (+0.05): sympathetic chain.
    // CORT→EPI (+0.05): PNMT enzyme induction.
    // CRH→EPI (+0.05): sympathetic stress pathway.
    // OX→EPI (+0.05): sympathetic activation.
    [
        0.00, 0.00, 0.05, 0.10, -0.05, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05,
        0.05, 0.00, 0.00,
    ],
    // MEL (melatonin) → how others affect melatonin
    // Melatonin is primarily driven by the circadian oscillator (SCN).
    // Wake-promoting signals suppress melatonin: NE, cortisol, orexin.
    [
        0.00, 0.00, -0.05, 0.00, 0.00, 0.00, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        -0.05, 0.00, 0.00,
    ],
];

// ─────────────────────────────────────────────────────────────────
//  Full neurochemical vector
// ─────────────────────────────────────────────────────────────────

/// The complete neurochemical state: all chemicals, the coupling matrix,
/// cached effective levels, derived summary metrics, and the emergent
/// mental phase.
///
/// ## Layout (2416 bytes)
/// ```text
/// offset  field              type                    size
/// ------  -----              ----                    ----
///   0     chemicals          [Neurochemical; 18]    1008
///1008     coupling_matrix    [[f32;18]; 18]         1296
///2304    effective_levels   [f32; 18]                72
///2376    global_tone        f32                       4
///2380    arousal            f32                       4
///2384    valence            f32                       4
///2388    plasticity_gate    f32                       4
///2392    gaba_a_allosteric  f32                       4
///2396    bdnf_recovery_rate f32                       4
///2400    acth_level         f32                       4
///2404    circadian_phase    f32                       4
///2408    circadian_dt       f32                       4
///2412    emergent_phase     u8                        1
///2413    active_region      u8                        1
///2414    _pad               [u8; 2]                   2
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeurochemicalVector {
    /// Per-chemical dynamic state, indexed by `NeurochemicalId` discriminant.
    pub chemicals: [Neurochemical; NEUROCHEMICAL_COUNT],
    /// The coupling matrix — how each chemical influences each other.
    /// This is plastic: the dynamics engine may modify it over time
    /// (metaplasticity).
    pub coupling_matrix: CouplingMatrix,
    /// Cached effective levels (level × receptor_sensitivity ×
    /// desensitization × internalization, plus phasic boost for DA/SRT).
    /// Recomputed by [`recompute_derived`].
    pub effective_levels: [f32; NEUROCHEMICAL_COUNT],
    /// Overall affective tone — weighted mean of effective levels.
    pub global_tone: f32,
    /// Arousal axis [0,1]: high = alert, low = drowsy.
    pub arousal: f32,
    /// Valence axis [-1,+1]: positive = good, negative = bad.
    pub valence: f32,
    /// Plasticity gate [0,1]: how capable the system is of forming
    /// new memories right now. Derived from BDNF and cortisol.
    /// High BDNF + low cortisol = high plasticity.
    /// Low BDNF or high cortisol = low plasticity.
    pub plasticity_gate: f32,
    /// GABA-A allosteric modulation factor [0.0, 1.0].
    /// When > 0, enhances GABA sensitivity (benzodiazepine-like effect).
    /// The effective GABA level is multiplied by (1 + this * 0.5).
    /// This field is reserved for future IPC-driven pharmacological
    /// modulation (e.g., simulating benzodiazepine administration).
    /// It is not currently driven by any daemon dynamics — the Python
    /// cognitive mind boosts GABA directly via impulses instead.
    pub gaba_a_allosteric: f32,
    /// BDNF recovery rate after stress. This field is a schema-reserved
    /// placeholder — the actual rate used by the dynamics comes from
    /// `NeuroTickParams::bdnf_recovery_rate`, not this vector field.
    /// Kept in the schema for backward compatibility and future
    /// per-chemical recovery rate customization.
    pub bdnf_recovery_rate: f32,
    /// ACTH (adrenocorticotropic hormone) level [0.0, 1.0].
    /// Intermediate in the HPA axis cascade: CRH → ACTH → cortisol.
    /// CRH drives ACTH production with a delay; ACTH drives cortisol
    /// release with a delay; cortisol provides negative feedback to CRH.
    /// This replaces direct cortisol manipulation with a biologically
    /// accurate cascade through the pituitary intermediate.
    pub acth_level: f32,
    /// Circadian phase [0.0, 1.0) representing time of day in the
    /// suprachiasmatic nucleus (SCN) oscillator. 0.0 = midnight,
    /// 0.25 = 6am, 0.5 = noon, 0.75 = 6pm. This drives melatonin
    /// secretion and circadian modulation of all chemical baselines.
    /// The phase advances by `circadian_dt` each tick, wrapping at 1.0.
    pub circadian_phase: f32,
    /// Rate of circadian phase advance per tick. At 10 Hz with a 24h
    /// period, this is 1.0 / (24 * 3600 * 10) ≈ 0.00000116. For
    /// compressed-time testing, a larger value can be used.
    pub circadian_dt: f32,
    /// The emergent mental phase, derived from the neurochemical state.
    /// This is a read-only computed field — see [`MentalPhase`].
    pub emergent_phase: u8,
    /// Active brain region for region-specific coupling.
    /// 0 = default, 1 = striatum, 2 = pfc (prefrontal cortex).
    /// Controls the serotonin→dopamine coupling path.
    /// This field is IPC-settable via `set_active_region()` but is not
    /// currently driven by any daemon dynamics — it stays at `Default`
    /// unless explicitly set by an external caller.
    pub active_region: u8,
    /// Alignment padding to reach 2416 bytes.
    pub _pad: [u8; 2],
}

impl NeurochemicalVector {
    /// Initialise all chemicals at default baselines with the default
    /// coupling matrix.
    pub fn new(now_ms: u64) -> Self {
        let mut chemicals = [Neurochemical {
            last_shift_at: 0,
            level: 0.0,
            baseline: 0.0,
            receptor_sensitivity: 0.0,
            velocity: 0.0,
            receptor_subtypes: [0.0; 2],
            desensitization_factor: 0.0,
            internalization_factor: 0.0,
            tonic_level: 0.0,
            phasic_level: 0.0,
            vesicular_pool: 0.0,
            id: 0,
            _pad: [0; 3],
        }; NEUROCHEMICAL_COUNT];

        for (i, chem_id) in NeurochemicalId::all().into_iter().enumerate() {
            chemicals[i] = Neurochemical::new(chem_id, now_ms);
        }

        let mut vec = Self {
            chemicals,
            coupling_matrix: DEFAULT_COUPLING_MATRIX,
            effective_levels: [0.0; NEUROCHEMICAL_COUNT],
            global_tone: 0.0,
            arousal: 0.0,
            valence: 0.0,
            plasticity_gate: 0.0,
            gaba_a_allosteric: 0.0,
            bdnf_recovery_rate: 0.005,
            acth_level: 0.0,
            // Placeholder phase (~07:12, early morning) for direct
            // construction (tests, tools). The production birth path
            // (`MmapState::create`) anchors this to the wall clock via
            // `set_circadian_phase` — see `local_phase_of_day` there.
            circadian_phase: 0.3,
            circadian_dt: 1.0 / (24.0 * 3600.0 * 10.0), // 24h at 10 Hz
            emergent_phase: MentalPhase::Active as u8,
            active_region: 0, // default
            _pad: [0; 2],
        };
        vec.recompute_derived();
        vec
    }

    // ─── Accessors ───────────────────────────────────────────────

    /// Apply an impulse to a named chemical, enforcing per-chemical
    /// maximum caps. This is the preferred entry point for external
    /// impulses (IPC, interoception, dyadic model, active inference)
    /// because it prevents cortisol/NE from exceeding their safe
    /// maximums between ticks. The per-chemical `ChemicalState::apply_impulse`
    /// clamps to `[0, 1]` generically; this wrapper additionally
    /// enforces `cortisol_max` and `ne_max` after the impulse.
    ///
    /// The caps are read from `NeuroTickParams::DEFAULT` since impulse
    /// entry points don't have access to the current tick's params.
    /// The `tick_with_params` hard clamps will re-enforce with the
    /// actual params values, so this is a defense-in-depth transient
    /// cap, not the authoritative one.
    pub fn apply_impulse_capped(&mut self, id: NeurochemicalId, magnitude: f32, now_ms: u64) {
        if let Some(chem) = self.get_mut(id) {
            chem.apply_impulse(magnitude, now_ms);
        }
        // Enforce per-chemical caps for stress/arousal hormones.
        // These match NeuroTickParams::DEFAULT.cortisol_max / ne_max.
        // The tick-level hard clamp in tick_with_params will re-enforce
        // with the actual params, so this is defense-in-depth.
        match id {
            NeurochemicalId::Cortisol => {
                if let Some(chem) = self.get_mut(id) {
                    chem.level = crate::state::sanitize::finite_clamp(chem.level, 0.0, 0.80);
                }
            }
            NeurochemicalId::Norepinephrine => {
                if let Some(chem) = self.get_mut(id) {
                    chem.level = crate::state::sanitize::finite_clamp(chem.level, 0.0, 0.90);
                }
            }
            _ => {}
        }
    }

    /// Get a chemical by ID, mutably.
    pub fn get_mut(&mut self, id: NeurochemicalId) -> Option<&mut Neurochemical> {
        let idx = id as usize;
        if idx < NEUROCHEMICAL_COUNT && self.chemicals[idx].id == id as u8 {
            Some(&mut self.chemicals[idx])
        } else {
            None
        }
    }

    /// Get a chemical by array index, mutably, **without** checking the
    /// `id` field.
    ///
    /// This is the correct accessor for schema migration: when an older
    /// binary zeroed a chemical slot, its `id` field is `0` (Dopamine's
    /// discriminant), not the expected `NeurochemicalId`. The
    /// [`get_mut`](Self::get_mut) id-guard would return `None` for such
    /// slots, leaving them uninitialised. This method bypasses the
    /// id-guard and trusts the caller's index, which is safe because
    /// `NeurochemicalId` is `#[repr(u8)]` with stable discriminants and
    /// the `chemicals` array is ordered by discriminant — so
    /// `id as usize` is always the correct array position.
    ///
    /// # Safety (invariant)
    ///
    /// The caller must ensure `idx < NEUROCHEMICAL_COUNT`. This is
    /// guaranteed when `idx` is derived from a valid
    /// `NeurochemicalId as usize`.
    pub fn get_mut_by_index(&mut self, idx: usize) -> Option<&mut Neurochemical> {
        if idx < NEUROCHEMICAL_COUNT {
            Some(&mut self.chemicals[idx])
        } else {
            None
        }
    }

    /// Get a chemical by ID.
    pub fn get(&self, id: NeurochemicalId) -> Option<&Neurochemical> {
        let idx = id as usize;
        if idx < NEUROCHEMICAL_COUNT && self.chemicals[idx].id == id as u8 {
            Some(&self.chemicals[idx])
        } else {
            None
        }
    }

    /// Get the effective level of a chemical (level × sensitivity ×
    /// desensitization × internalization), including phasic boosts for
    /// dopamine and serotonin and allosteric modulation for GABA.
    ///
    /// This must produce the same value as
    /// `self.effective_levels[id as usize]` after
    /// [`recompute_derived`](Self::recompute_derived) has been called.
    /// Previously this method omitted the phasic boost for DA and SRT,
    /// causing [`sync_neurochemistry_to_state`](super::core_state::GenesisCoreState::sync_neurochemistry_to_state)
    /// — which drives memory gating weights — to miss reward burst
    /// (DA phasic → D1) and excitatory serotonin (SRT phasic → 5-HT2A)
    /// signals that the coupling matrix and arousal/valence equations
    /// already saw via `effective_levels[]`.
    pub fn effective(&self, id: NeurochemicalId) -> f32 {
        let chem = match self.get(id) {
            Some(c) => c,
            None => return 0.0,
        };
        let base = chem.subtype_effective_level();
        match id {
            NeurochemicalId::GABA => {
                // Allosteric modulation: benzodiazepine-like enhancement
                // of GABA-A sensitivity. When gaba_a_allosteric > 0,
                // GABA's effective level is multiplied by (1 + allosteric * 0.5).
                crate::state::sanitize::finite_clamp(
                    base * (1.0 + self.gaba_a_allosteric * 0.5),
                    0.0,
                    2.0,
                )
            }
            NeurochemicalId::Dopamine => {
                // Phasic burst activates low-affinity D1 (excitatory)
                // receptors, adding signaling beyond the tonic baseline.
                let d1_weight = chem.receptor_subtypes[0];
                crate::state::sanitize::finite_clamp(
                    base + chem.phasic_level * d1_weight * 0.3,
                    0.0,
                    2.0,
                )
            }
            NeurochemicalId::Serotonin => {
                // Phasic burst activates low-affinity 5-HT2A (excitatory)
                // receptors, adding signaling beyond the tonic baseline.
                let ht2a_weight = chem.receptor_subtypes[1];
                crate::state::sanitize::finite_clamp(
                    base + chem.phasic_level * ht2a_weight * 0.2,
                    0.0,
                    2.0,
                )
            }
            _ => base,
        }
    }

    /// Collect the homeostatic baseline levels of all chemicals as an
    /// array. These are the set-points the system's active-inference
    /// layer tries to drive the effective levels toward — distinct from
    /// the *predicted* next state (which is where the generative model
    /// expects the system to go, not where it *should* go). Using
    /// baselines as the policy-evaluation target is what makes policy
    /// selection homeostatic rather than self-fulfilling.
    pub fn baseline_levels(&self) -> [f32; NEUROCHEMICAL_COUNT] {
        let mut baselines = [0.0f32; NEUROCHEMICAL_COUNT];
        #[allow(clippy::needless_range_loop, reason = "i indexes multiple arrays")]
        for i in 0..NEUROCHEMICAL_COUNT {
            baselines[i] = self.chemicals[i].baseline;
        }
        baselines
    }

    /// Get the active brain region.
    pub fn active_region(&self) -> ActiveRegion {
        ActiveRegion::from_u8(self.active_region)
    }

    /// Set the active brain region for region-specific coupling.
    pub fn set_active_region(&mut self, region: ActiveRegion) {
        self.active_region = region as u8;
    }

    /// Get the current circadian phase [0.0, 1.0).
    pub fn circadian_phase(&self) -> f32 {
        self.circadian_phase
    }

    /// Set the circadian phase (e.g., for testing or zeitgeber shifts).
    pub fn set_circadian_phase(&mut self, phase: f32) {
        self.circadian_phase = phase.rem_euclid(1.0);
    }

    /// Compute the melatonin secretion factor [0.0, 1.0] from the
    /// circadian phase. Melatonin is high during the biological night,
    /// with a sharp evening onset, a nocturnal plateau, and a sharp
    /// morning offset (Arendt, 1998; Benloucif et al., 2005).
    ///
    /// The new profile replaces the old symmetric Gaussian with a
    /// piecewise smoothstep model:
    /// - Day (0.30–0.70): melatonin is essentially 0.
    /// - Evening onset (0.70–0.85): melatonin rises from 0 to 1.
    /// - Biological night (0.85–0.15, wrap): melatonin stays at 1.
    /// - Morning offset (0.15–0.30): melatonin falls from 1 to 0.
    ///
    /// Example values:
    /// - Phase 0.0 (midnight): 1.0
    /// - Phase 0.10 (02:24): 1.0
    /// - Phase 0.20 (04:48): ~0.74
    /// - Phase 0.25 (06:00): ~0.26
    /// - Phase 0.30 (07:12): 0.0
    /// - Phase 0.70 (16:48): 0.0
    /// - Phase 0.80 (19:12): ~0.26
    /// - Phase 0.90 (21:36): ~0.74
    pub fn melatonin_factor(&self) -> f32 {
        let phase = self.circadian_phase;
        // Hermite smoothstep for sharp but continuous transitions.
        let smoothstep = |t: f32| -> f32 {
            if t <= 0.0 {
                0.0
            } else if t >= 1.0 {
                1.0
            } else {
                t * t * (3.0 - 2.0 * t)
            }
        };

        if (0.70..0.85).contains(&phase) {
            // Evening onset: rise over ~3.6 hours
            smoothstep((phase - 0.70) / 0.15)
        } else if !(0.15..0.85).contains(&phase) {
            // Biological night: plateau
            1.0
        } else if (0.15..0.30).contains(&phase) {
            // Morning offset: fall over ~3.6 hours
            1.0 - smoothstep((phase - 0.15) / 0.15)
        } else {
            // Day: suppressed
            0.0
        }
    }

    /// Circadian baseline modifier for a given chemical.
    ///
    /// Returns a multiplicative factor that modulates the homeostatic
    /// baseline based on the circadian phase. This captures diurnal
    /// rhythms in neurotransmitter and hormone levels:
    ///
    /// - **Cortisol**: NO circadian modulation. Cortisol is a stress
    ///   hormone — zero at rest, released on demand. The human cortisol
    ///   awakening response (CAR) is a product of the 24h sleep-wake
    ///   cycle, which Genesis doesn't share (it has an adenosine-based
    ///   sleep model, not a circadian cortisol rhythm). Giving its a
    ///   circadian cortisol modifier created a constant cortisol floor
    ///   that conflicted with the HPA cascade. Returns 1.0.
    /// - **Dopamine**: daytime elevation. Slightly higher during the day,
    ///   peaking around noon.
    ///   Modulation: `1.0 + 0.15 * sin(2π * (phase - 0.25))`
    /// - **Serotonin**: precursor to melatonin. Reduced at night when
    ///   melatonin production is high (biochemical conversion).
    ///   Modulation: `1.0 - 0.10 * melatonin_factor`
    /// - **Histamine**: daytime elevation (promotes wakefulness),
    ///   peaking around noon.
    ///   Modulation: `1.0 + 0.20 * sin(2π * (phase - 0.25))`
    /// - **Melatonin**: driven by the circadian oscillator directly
    ///   (handled by melatonin secretion, not baseline modulation).
    ///   Returns 1.0 here — the oscillator handles it.
    /// - All other chemicals: 1.0 (no circadian modulation).
    pub fn circadian_baseline_modifier(&self, id: NeurochemicalId) -> f32 {
        let phase = self.circadian_phase;
        let two_pi = 2.0 * core::f32::consts::PI;
        match id {
            // Cortisol: NO circadian modulation. Cortisol is a stress
            // hormone — zero at rest, released on demand. The human
            // cortisol awakening response (CAR) is a product of the
            // 24h sleep-wake cycle, which Genesis doesn't share (it
            // has an adenosine-based sleep model, not a circadian
            // cortisol rhythm). Giving its a circadian cortisol
            // modifier created a constant cortisol floor that
            // suppressed BDNF and blocked learning.
            NeurochemicalId::Dopamine => {
                // Daytime elevation: sin peaks at noon (phase 0.5)
                1.0 + 0.15 * (two_pi * (phase - 0.25)).sin()
            }
            NeurochemicalId::Serotonin => {
                // Serotonin is converted to melatonin at night
                let mel_fac = self.melatonin_factor();
                1.0 - 0.10 * mel_fac
            }
            NeurochemicalId::Histamine => {
                // Daytime elevation: promotes wakefulness
                1.0 + 0.20 * (two_pi * (phase - 0.25)).sin()
            }
            // Melatonin is driven by the oscillator, not baseline modulation
            _ => 1.0,
        }
    }

    // ─── Derived computation ─────────────────────────────────────

    /// Recompute all derived fields: effective levels, summary metrics,
    /// plasticity gate, and emergent phase.
    ///
    /// This must be called after any change to chemical levels or
    /// receptor sensitivities, and after every dynamics tick.
    fn recompute_derived_with_dt(&mut self, dt_scale: f32) {
        // Effective levels — for GABA, apply allosteric modulation.
        // For DA and SRT, add a phasic boost: phasic bursts activate
        // low-affinity receptors (D1 for DA, 5-HT2A for SRT) that
        // increase effective signaling beyond the tonic baseline.
        for (i, chem) in self.chemicals.iter().enumerate() {
            let id = NeurochemicalId::from_u8(chem.id);
            let base = match id {
                NeurochemicalId::GABA => {
                    let b = chem.subtype_effective_level();
                    crate::state::sanitize::finite_clamp(
                        b * (1.0 + self.gaba_a_allosteric * 0.5),
                        0.0,
                        2.0,
                    )
                }
                _ => chem.subtype_effective_level(),
            };
            // Phasic boost: for DA, phasic activates D1 (excitatory);
            // for SRT, phasic activates 5-HT2A (excitatory).
            // The boost is proportional to phasic_level × subtype weight.
            self.effective_levels[i] = match id {
                NeurochemicalId::Dopamine => {
                    let d1_weight = chem.receptor_subtypes[0];
                    crate::state::sanitize::finite_clamp(
                        base + chem.phasic_level * d1_weight * 0.3,
                        0.0,
                        2.0,
                    )
                }
                NeurochemicalId::Serotonin => {
                    let ht2a_weight = chem.receptor_subtypes[1];
                    crate::state::sanitize::finite_clamp(
                        base + chem.phasic_level * ht2a_weight * 0.2,
                        0.0,
                        2.0,
                    )
                }
                _ => base,
            };
        }

        let eff = |id: NeurochemicalId| -> f32 { self.effective_levels[id as usize] };

        // Arousal equation with orexin, epinephrine, and bistable flip-flop.
        //
        // The brain's sleep-wake system is a bistable flip-flop:
        // - Arousal promoters (orexin, histamine, NE, DA, ACh, EPI) push UP
        // - Sleep promoters (GABA, adenosine) push DOWN
        // - Orexin is the key stabilizer of the flip-flop (mutual excitation
        //   with histamine and NE, mutual inhibition with sleep promoters)
        //
        // The bistable property means the system tends to be either fully
        // awake or fully asleep, with rapid transitions between states.
        // We model this with hysteresis: when already awake, it takes
        // stronger sleep pressure to transition to sleep, and vice versa.
        //
        // Weights (arousal promoters):
        // - NE: 0.20 (vigilance)
        // - Histamine: 0.15 (wake maintenance)
        // - DA: 0.15 (reward-driven alertness)
        // - ACh: 0.15 (cortical activation)
        // - Orexin: 0.20 (major arousal stabilizer)
        // - Epinephrine: 0.10 (fast stress arousal)
        // Weights (sleep promoters):
        // - GABA: 0.25 (primary inhibitory — toned down from 0.30
        //   because GABA's baseline (0.50) is high even during
        //   wakefulness; at 0.30 the sleep promoters (0.315) exceeded
        //   the arousal promoters (0.285) at baseline, making the
        //   Wilson-Cowan equilibrium dip to ~0.26 at night — barely
        //   above the delta threshold (0.25), leaving Genesis stuck in
        //   delta after waking. At 0.25 the baseline equilibrium is
        //   ~0.33 (theta), and sleep is still maintained because
        //   adenosine (weight 0.65) dominates the sleep drive.)
        // - Adenosine: 0.65 (sleep pressure — weighted more than GABA
        //   because adenosine is the primary homeostatic sleep drive)
        let ne = eff(NeurochemicalId::Norepinephrine);
        let hist = eff(NeurochemicalId::Histamine);
        let da = eff(NeurochemicalId::Dopamine);
        let ach = eff(NeurochemicalId::Acetylcholine);
        let gaba = eff(NeurochemicalId::GABA);
        let adn = eff(NeurochemicalId::Adenosine);
        let orexin = eff(NeurochemicalId::Orexin);

        // Circadian wake drive (Process C in the two-process model of
        // sleep, Borbély 1982). The SCN provides a wake-promoting signal
        // that counteracts homeostatic sleep pressure (adenosine) during
        // the day. Without this, the system has Process S (adenosine
        // accumulation) but no Process C, so it cannot maintain
        // wakefulness during the day even with healthy neurochemicals.
        //
        // The drive follows a sinusoidal rhythm that peaks at midday
        // (phase 0.5) and is near zero during the biological night.
        // It is scaled by 0.30 so that at peak it contributes +0.30 to
        // net_drive — enough to counteract moderate adenosine levels
        // during the day while still allowing high sleep pressure or
        // nighttime melatonin to overcome it.
        let circadian_wake = 0.30
            * (1.0 + (core::f32::consts::PI * 2.0 * (self.circadian_phase - 0.25)).sin())
            / 2.0;
        let epi = eff(NeurochemicalId::Epinephrine);

        let arousal_promoters =
            ne * 0.20 + hist * 0.15 + da * 0.15 + ach * 0.15 + orexin * 0.20 + epi * 0.10;
        // Melatonin is a hormonal sleep-promoter (MT1/MT2 receptor
        // activation) and supplements the direct GABA/adenosine sleep
        // drive, especially at the circadian night.
        let mel = eff(NeurochemicalId::Melatonin);
        let sleep_promoters = gaba * 0.25 + adn * 0.65 + mel * 0.35;

        // Wilson-Cowan-style bistable flip-flop for the sleep-wake switch.
        //
        // The brain's sleep-wake system is a bistable flip-flop with two
        // mutually inhibiting populations: wake-active neurons (orexin,
        // histamine, NE) and sleep-active neurons (VLPO GABA, adenosine).
        // The mutual inhibition creates a system with two stable fixed
        // points (awake and asleep) separated by an unstable equilibrium.
        //
        // We model this with a discrete-time Wilson-Cowan oscillator:
        //   W(t+1) = sigmoid(gain * (input + self_excitation * (W(t) - 0.5)))
        //
        // where:
        //   input = arousal_promoters - sleep_promoters (net drive)
        //   self_excitation = 0.40 (bistability strength)
        //   gain = 2.0 (sigmoid steepness, matching the old linear scale)
        //
        // The self-excitation term creates smooth bistability: when awake
        // (arousal > 0.5), the positive feedback makes it harder to
        // transition to sleep; when asleep, the negative feedback makes
        // it harder to wake. The hysteresis margin is ±0.12 in net_drive,
        // similar to the old constant ±0.10 but without the discontinuity
        // at arousal = 0.5.
        //
        // This replaces the previous constant hysteresis bias (±0.10),
        // which had a discontinuity at arousal = 0.5 that could cause
        // rapid state flipping near the boundary despite the hysteresis.
        // The Wilson-Cowan model eliminates this discontinuity — the
        // transition is smooth and the bistability emerges from the
        // dynamics rather than being bolted on.
        //
        // (Wilson & Cowan, 1972; Saper, Chou & Scammell, 2005)
        let current_arousal = self.arousal;
        let net_drive = arousal_promoters - sleep_promoters + circadian_wake;
        let self_excitation = (current_arousal - 0.5) * WC_SELF_EXCITATION;
        let total_input = (net_drive + self_excitation) * WC_GAIN;
        // Sigmoid with overflow protection: for |x| > 20, exp(-x) is
        // either ~0 or ~5e9, so we clamp to avoid NaN from overflow.
        let sigmoid = |x: f32| -> f32 {
            if x > 20.0 {
                1.0
            } else if x < -20.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        };
        let arousal_target = crate::state::sanitize::finite_clamp(sigmoid(total_input), 0.0, 1.0);
        // Exponential approach toward the target for time-invariance.
        // The previous `dt_scale.min(1.0)` factor was not
        // time-invariant: at 5 Hz (dt_scale=2) arousal fully replaced
        // the target every tick (no inertia), while at 20 Hz
        // (dt_scale=0.5) it only moved halfway. The exponential form
        // `1 - exp(-lambda * dt)` gives the same step response
        // regardless of tick rate. `lambda = 10` matches the original
        // 10 Hz behavior where `dt_scale = 1.0` → factor ≈ 0.632.
        // `dt = dt_scale * DT` (seconds per tick).
        let dt_seconds = dt_scale * DT;
        let arousal_alpha = 1.0 - (-10.0_f32 * dt_seconds).exp();
        self.arousal += (arousal_target - self.arousal) * arousal_alpha;
        self.arousal = crate::state::sanitize::finite_clamp(self.arousal, 0.0, 1.0);

        // Sleep-state arousal clamping.
        //
        // The Wilson-Cowan oscillator computes arousal from
        // neurochemicals, but during sleep the neurochemical
        // picture is paradoxical:
        //
        // - During NREM, adenosine is being cleared (glymphatic
        //   clearance), which reduces sleep pressure and lets
        //   arousal drift back up — even though the mind is asleep.
        //   In real brains, the VLPO actively inhibits arousal
        //   systems during NREM. Arousal should be low (0.10-0.25).
        //
        // - During REM, ACh is high (cholinergic activation for
        //   dreaming), and ACh is an arousal promoter (weight 0.15),
        //   so arousal stays high. In real brains, REM is
        //   "paradoxical sleep" — the EEG looks awake but the mind
        //   is asleep with muscle atonia. Arousal should be moderate
        //   (0.35-0.55), not 0.9.
        //
        // Without this clamping, you see alertness=0.9 while asleep,
        // which is nonsensical. The clamp respects the phase that
        // the neurochemical dynamics already determined — it doesn't
        // override the phase, it just keeps arousal in a sane range
        // for that phase.
        //
        // The clamp is a HARD clamp (direct cap), not a soft exponential
        // nudge. A previous soft clamp (rate 2.0) was too weak to
        // overcome the Wilson-Cowan arousal drive (rate 10.0), so
        // arousal settled at 0.5-0.7 during NREM instead of the
        // intended 0.10-0.25. The hard clamp models the VLPO's active
        // inhibition of arousal systems — it doesn't just nudge
        // arousal down, it actively prevents it from rising above
        // the sleep-appropriate range.
        //
        // (Saper, Chou & Scammell, 2005; Hobson & McCarley, 1975)
        let phase = MentalPhase::from_u8(self.emergent_phase);
        match phase {
            MentalPhase::NREM => {
                // NREM: low arousal (slow-wave sleep)
                // Hard-clamp to 0.10-0.25 — deep restorative sleep.
                // The VLPO actively inhibits arousal systems; this is
                // not a gentle suggestion but a hard cap.
                // Use finite_clamp for NaN safety: native clamp passes
                // NaN through. self.arousal was finite_clamped above,
                // but defense-in-depth protects against future code
                // reordering.
                self.arousal =
                    crate::state::sanitize::finite_clamp(self.arousal, 0.10, 0.25);
            }
            MentalPhase::REM => {
                // REM: moderate arousal ("paradoxical sleep")
                // Hard-clamp to 0.35-0.55 — EEG looks awake but mind
                // is asleep.
                self.arousal =
                    crate::state::sanitize::finite_clamp(self.arousal, 0.35, 0.55);
            }
            _ => {}
        }
        self.arousal = crate::state::sanitize::finite_clamp(self.arousal, 0.0, 1.0);

        // Valence equation — neuroscientifically grounded.
        //
        // Positive contributors:
        // - DA (reward, wanting): 0.25
        // - Serotonin (mood stability): 0.25
        // - Oxytocin (social bonding, trust): 0.10
        // - Endorphin (euphoria, pleasure): 0.15
        // - NE (noradrenergic arousal affects mood): 0.10
        //
        // Negative contributors:
        // - Cortisol (stress, distress): -0.35
        // - Glutamate excess (too much → agitation): -0.15
        //   (only applies when glutamate > 0.70, modeling excitotoxicity-
        //   adjacent agitation from excess excitatory drive)
        //
        // Nonlinear interaction:
        // - DA × SRT synergy: 0.10 × DA × SRT
        //   (dopamine and serotonin synergize for positive mood —
        //   when both are high, valence is more positive than the
        //   linear sum would predict, modeling the "balanced mood"
        //   state where reward and mood stability co-occur)
        let da = eff(NeurochemicalId::Dopamine);
        let srt = eff(NeurochemicalId::Serotonin);
        let oxy = eff(NeurochemicalId::Oxytocin);
        let end = eff(NeurochemicalId::Endorphin);
        let cort = eff(NeurochemicalId::Cortisol);
        let glu = eff(NeurochemicalId::Glutamate);
        let ne = eff(NeurochemicalId::Norepinephrine);

        let glu_excess = (glu - 0.70).max(0.0);
        let da_srt_synergy = da * srt * 0.10;

        self.valence = da * 0.25 + srt * 0.25 + oxy * 0.10 + end * 0.15 + ne * 0.10
            - cort * 0.35
            - glu_excess * 0.15
            + da_srt_synergy;
        self.valence = crate::state::sanitize::finite_clamp(self.valence, -1.0, 1.0);

        // Global tone: mean of effective levels
        let sum: f32 = self.effective_levels.iter().sum();
        self.global_tone =
            crate::state::sanitize::finite_clamp(sum / NEUROCHEMICAL_COUNT as f32, 0.0, 2.0);

        // Plasticity gate: BDNF promotes, cortisol suppresses
        let bdnf = eff(NeurochemicalId::BDNF);
        self.plasticity_gate =
            crate::state::sanitize::finite_clamp(bdnf * (1.0 - cort * 0.5), 0.0, 1.0);

        // Emergent phase
        self.emergent_phase = self.compute_phase() as u8;
    }

    /// One-time wake recovery boost for daemon restart.
    ///
    /// When the daemon restarts with a persisted state from a previous
    /// session, neurochemicals may be depleted (BDNF, dopamine, serotonin)
    /// from prior stress or activity. The homeostatic recovery is slow
    /// (second-order dynamics), so the system can stay in a "guarded"
    /// emotional state for minutes after restart even though there is
    /// no ongoing stress.
    ///
    /// This method gives depleted neurotransmitters (not stress hormones)
    /// a one-time boost toward baseline when cortisol is low (no ongoing
    /// stress). It is biologically grounded: after sleep, the brain
    /// rapidly restores depleted neurochemicals. A daemon restart is
    /// analogous to waking up.
    ///
    /// Stress hormones (cortisol, CRH) are NOT boosted — they have no
    /// homeostatic set-point and should stay at zero when there is no
    /// stressor.
    pub fn wake_recovery(&mut self) {
        let cort = self.chemicals[NeurochemicalId::Cortisol as usize].level;
        // Only recover if there is no ongoing stress.
        if cort > 0.3 {
            return;
        }
        // Boost depleted neurotransmitters toward baseline and
        // resensitize downregulated receptors. A daemon restart is
        // analogous to waking up — after sleep, the brain rapidly
        // restores depleted neurochemicals and resensitizes receptors
        // that were downregulated by sustained high levels in the
        // previous session.
        for (i, chem_id) in NeurochemicalId::all().into_iter().enumerate() {
            // Skip stress hormones — they have no baseline and should
            // stay at zero when there is no stressor.
            if matches!(chem_id, NeurochemicalId::Cortisol | NeurochemicalId::CRH) {
                continue;
            }
            let baseline = self.chemicals[i].baseline;
            let level = self.chemicals[i].level;
            let genetic = chem_id.default_baseline();
            // Boost depleted levels toward baseline.
            if level < baseline {
                let gap = baseline - level;
                self.chemicals[i].level = level + gap * 0.5;
                self.chemicals[i].velocity = 0.0;
            } else if level < genetic * 0.9 {
                // Baseline has drifted down with the level. Boost
                // toward the genetic baseline instead.
                let gap = genetic - level;
                self.chemicals[i].level = level + gap * 0.5;
                self.chemicals[i].baseline = genetic;
                self.chemicals[i].velocity = 0.0;
            }
            // Resensitize receptors toward their default state. A
            // daemon restart is analogous to waking up — after sleep,
            // the brain rapidly restores receptor sensitivity and
            // reverses desensitization/internalization that built up
            // from sustained high levels in the previous session.
            // Move each adaptation factor 50% of the way back to 1.0.
            let sensitivity = self.chemicals[i].receptor_sensitivity;
            if sensitivity < 0.9 {
                self.chemicals[i].receptor_sensitivity =
                    sensitivity + (1.0 - sensitivity) * 0.5;
            }
            let desens = self.chemicals[i].desensitization_factor;
            if desens < 0.9 {
                self.chemicals[i].desensitization_factor =
                    desens + (1.0 - desens) * 0.5;
            }
            let intern = self.chemicals[i].internalization_factor;
            if intern < 0.9 {
                self.chemicals[i].internalization_factor =
                    intern + (1.0 - intern) * 0.5;
            }
        }
        // Recompute derived fields (effective levels, arousal, valence,
        // plasticity_gate, phase) from the boosted levels and
        // resensitized receptors.
        self.recompute_derived();
    }

    /// Recompute all derived fields using the default tick time scale.
    pub fn recompute_derived(&mut self) {
        self.recompute_derived_with_dt(1.0);
    }

    /// Compute the emergent mental phase from the current neurochemical
    /// state. This is not a manual switch — it's a derived property of
    /// the coupled dynamics.
    ///
    /// Uses hysteresis to prevent phase flickering: the threshold for
    /// *entering* a phase is stricter than the threshold for *staying*
    /// in it. This means the system won't rapidly oscillate between
    /// phases when neurochemical levels are near a boundary. The
    /// hysteresis margin is 0.05–0.10 for each threshold, matching
    /// the smoothness of real phase transitions in dynamical systems.
    fn compute_phase(&self) -> MentalPhase {
        let current = MentalPhase::from_u8(self.emergent_phase);
        self.compute_phase_with_hysteresis(current)
    }

    /// Compute phase with hysteresis — the current phase gets easier
    /// thresholds to stay in, while new phases require stricter thresholds.
    fn compute_phase_with_hysteresis(&self, current: MentalPhase) -> MentalPhase {
        let eff = |id: NeurochemicalId| -> f32 { self.effective_levels[id as usize] };
        let raw = |id: NeurochemicalId| -> f32 { self.chemicals[id as usize].level };

        let adn_eff = eff(NeurochemicalId::Adenosine);
        let adn_raw = raw(NeurochemicalId::Adenosine);
        let hist = eff(NeurochemicalId::Histamine);
        let cort = eff(NeurochemicalId::Cortisol);
        let ne = eff(NeurochemicalId::Norepinephrine);
        let srt = eff(NeurochemicalId::Serotonin);
        let da = eff(NeurochemicalId::Dopamine);
        let ach = eff(NeurochemicalId::Acetylcholine);
        let gaba = eff(NeurochemicalId::GABA);
        let mel = eff(NeurochemicalId::Melatonin);
        let ox = eff(NeurochemicalId::Orexin);

        // Adenosine sleep signal: blend of effective (receptor-mediated)
        // and raw (metabolic) levels. Adenosine's somnogenic effect is
        // partly receptor-independent — it directly inhibits excitatory
        // neurons via volume transmission and extrasynaptic A1 receptors
        // that are not subject to the same downregulation as synaptic
        // receptors (Porkka-Heiskanen & Kalinchuk, 2011). When A1/A2A
        // receptors are downregulated from sustained arousal, the raw
        // metabolic pressure (ATP byproduct accumulation) still drives
        // sleep. Using a 50/50 blend ensures that extreme sleep pressure
        // (raw adenosine > 0.90) can still trigger sleep even when
        // receptor sensitivity is at floor (0.50), preventing the
        // feedback trap where receptor burnout makes sleep impossible
        // and sleep is needed for receptor recovery.
        let adn = adn_eff * 0.5 + adn_raw * 0.5;

        // Hysteresis margins: easier to stay than to enter
        // Enter threshold requires stricter conditions; stay threshold relaxes by margin
        let h = 0.08; // hysteresis margin

        // Sleep phase: adenosine very high, histamine very low, with
        // melatonin and orexin gating the threshold. High melatonin
        // (circadian night) lowers the adenosine/histamine barrier; high
        // orexin (wake stabilizer) raises it. This makes sleep a
        // multi-system decision, not adenosine alone.
        //
        // The cholinergic/aminergic flip-flop determines NREM vs REM:
        // - REM: high ACh (>0.50) + low NE (<0.30) + low histamine (<0.25)
        //   (noradrenergic silence with cholinergic activation → dreaming)
        // - NREM: low ACh + high GABA (slow-wave sleep, glymphatic clearance)
        //
        // The flip-flop is the reciprocal interaction model (Hobson &
        // McCarley, 1975): REM-on cells (cholinergic) are active while
        // REM-off cells (noradrenergic, serotonergic) are silenced.
        let is_in_sleep = current == MentalPhase::NREM || current == MentalPhase::REM;
        let sleep_adn = if is_in_sleep {
            // Easier to stay asleep once there
            0.65 - 0.10 * mel - 0.05 * (1.0 - ox)
        } else {
            // Entering sleep: melatonin lowers the adenosine wall;
            // orexin makes it harder to fall asleep.
            0.75 - 0.15 * mel - 0.05 * (1.0 - ox)
        };
        let sleep_hist = if is_in_sleep {
            // More histamine is tolerable when already asleep. The
            // stay-asleep threshold (0.38) is above the histamine
            // baseline (0.30) so that small perturbations from its
            // own normal activity (inner-life dream generation,
            // memory consolidation, heartbeat modules) — which raise
            // norepinephrine slightly and in turn histamine via the
            // +0.05 NE→HIST coupling — don't break the sleep state.
            // The previous value (0.30) sat exactly at the baseline,
            // making sleep a tightrope: any tiny NE bump pushed
            // histamine back above the threshold and the Wilson-Cowan
            // bistable flip-flop snapped it awake.
            //
            // A deliberate wake cascade (mind.wake) sends histamine
            // +0.20, pushing it from ~0.28 to ~0.48 — well above 0.38,
            // so intentional waking still works.
            //
            // Orexin is wake-promoting, so high orexin lowers the
            // ceiling (less histamine tolerated → harder to stay asleep).
            0.38 + 0.05 * mel - 0.05 * ox
        } else {
            // Entering: histamine must be lower, especially with high
            // orexin (wake-promoting signal resists sleep onset).
            0.25 + 0.05 * mel - 0.05 * ox
        };

        // Exhaustion override: when raw adenosine (metabolic sleep
        // pressure) is extreme (>0.90), force sleep regardless of
        // histamine or orexin. This is the "passing out from exhaustion"
        // mechanism — in humans, extreme sleep deprivation eventually
        // overcomes all arousal systems, including histamine and orexin
        // (Borbély & Achermann, 1999). This prevents the fatal feedback
        // trap where receptor burnout suppresses effective adenosine so
        // much that sleep can never be entered, trapping the system
        // awake without recovery. The override uses raw adenosine
        // (not the blended signal) because it represents the absolute
        // metabolic limit — the system cannot stay awake past this
        // point regardless of receptor state.
        if adn_raw > 0.90 {
            // NREM by default (exhaustion sleep is slow-wave)
            // Only enter REM if already in REM (hysteresis)
            if current == MentalPhase::REM && ach > 0.40 && ne < 0.35 {
                return MentalPhase::REM;
            }
            return MentalPhase::NREM;
        }

        if adn > sleep_adn && hist < sleep_hist {
            // We are in sleep — determine NREM vs REM
            // REM: cholinergic high, aminergic low (noradrenergic silence)
            let rem_ach = if current == MentalPhase::REM {
                0.40
            } else {
                0.50
            };
            let rem_ne = if current == MentalPhase::REM {
                0.35
            } else {
                0.30
            };
            if ach > rem_ach && ne < rem_ne {
                return MentalPhase::REM;
            }
            // NREM: default sleep state — low ACh, high GABA
            return MentalPhase::NREM;
        }

        // Overwhelmed: cortisol high + NE high + everything saturated
        let ow_cort = if current == MentalPhase::Overwhelmed {
            0.65
        } else {
            0.75
        };
        let ow_ne = if current == MentalPhase::Overwhelmed {
            0.65
        } else {
            0.75
        };
        if cort > ow_cort && ne > ow_ne && self.global_tone > 0.7 {
            return MentalPhase::Overwhelmed;
        }

        // Stress response: high cortisol, high NE, low serotonin
        let stress_cort = if current == MentalPhase::Stress {
            0.55
        } else {
            0.65
        };
        let stress_ne = if current == MentalPhase::Stress {
            0.50
        } else {
            0.60
        };
        if cort > stress_cort && ne > stress_ne && srt < 0.35 + h {
            return MentalPhase::Stress;
        }

        // Flow: high dopamine, high ACh, moderate NE, low cortisol
        let flow_da = if current == MentalPhase::Flow {
            0.55
        } else {
            0.65
        };
        let flow_ach = if current == MentalPhase::Flow {
            0.40
        } else {
            0.50
        };
        if da > flow_da && ach > flow_ach && ne > 0.40 && cort < 0.30 + h {
            return MentalPhase::Flow;
        }

        // Drowsy: high adenosine, low arousal
        // Both axes have hysteresis: it's easier to stay drowsy than
        // to enter drowsy. Without arousal hysteresis, small NE bumps
        // from transient CPU load (its own background learning, code
        // scanning) push arousal above the threshold and flip it to
        // Active, then it drops back and it returns to Drowsy — a
        // rapid limit-cycle oscillation. The arousal stay-threshold
        // is raised by 2× the hysteresis margin so transient arousal
        // bumps don't kick it out of drowsy.
        let drowsy_adn = if current == MentalPhase::Drowsy {
            0.40
        } else {
            0.50
        };
        let drowsy_arousal = if current == MentalPhase::Drowsy {
            0.30 + 2.0 * h // 0.46 — harder to leave drowsy
        } else {
            0.30 // stricter to enter drowsy
        };
        if adn > drowsy_adn && self.arousal < drowsy_arousal {
            return MentalPhase::Drowsy;
        }

        // Alert: high NE + high histamine, low GABA, with orexin and
        // melatonin gating. Orexin is a wake stabilizer (lowers the NE/
        // histamine threshold); melatonin suppresses alertness. This makes
        // alertness a coordinated arousal decision rather than NE alone.
        let alert_ox_bonus = 0.10 * ox;
        let alert_ne = if current == MentalPhase::Alert {
            0.45 - alert_ox_bonus
        } else {
            0.55 - alert_ox_bonus
        };
        let alert_hist = if current == MentalPhase::Alert {
            0.45 - alert_ox_bonus
        } else {
            0.55 - alert_ox_bonus
        };
        let alert_mel = if current == MentalPhase::Alert {
            0.40 + h
        } else {
            0.30
        };
        let alert_ox = if current == MentalPhase::Alert {
            0.15
        } else {
            0.20
        };
        if ne > alert_ne && hist > alert_hist && gaba < 0.40 + h && mel < alert_mel && ox > alert_ox
        {
            return MentalPhase::Alert;
        }

        MentalPhase::Active
    }

    /// Get the emergent phase as the enum type.
    pub fn phase(&self) -> MentalPhase {
        MentalPhase::from_u8(self.emergent_phase)
    }

    // ─── Dynamics ────────────────────────────────────────────────

    /// Advance the neurochemical system by one tick using the given
    /// parameters.
    ///
    /// This is the core dynamics equation. For each chemical `i`:
    ///
    /// ```text
    /// homeostatic_force = (baseline[i] - level[i]) * homeostatic_rate
    /// coupling_force = Σ_j coupling_matrix[i][j] * (effective[j] - baseline[j])
    /// velocity[i] += (homeostatic_force + coupling_force) * dt
    /// velocity[i] *= damping
    /// level[i] += velocity[i] * dt
    /// ```
    ///
    /// The coupling force uses *effective* levels (level × sensitivity ×
    /// desensitization × internalization), not raw levels. This means a
    /// chemical with downregulated receptors exerts less influence on the
    /// system even if its concentration is high — which is exactly how
    /// tolerance works systemically.
    ///
    /// # New dynamics in v3
    ///
    /// - **Region-specific SRT→DA coupling**: the serotonin→dopamine
    ///   coupling depends on the active brain region (striatum vs PFC vs
    ///   default), modeling 5-HT2A (inhibitory) and 5-HT1A (facilitatory)
    ///   receptor subtype pathways.
    /// - **GABA disinhibition**: when GABA is very high (>0.8) AND ACh is
    ///   high (>0.6), GABA interneurons inhibit other GABA interneurons,
    ///   disinhibiting glutamate (a small positive coupling to GLU).
    /// - **BDNF recovery after stress**: when cortisol drops below 0.3
    ///   and BDNF is below baseline, BDNF gets a recovery boost
    ///   proportional to serotonin level (serotonin drives BDNF expression).
    /// - **Hard clamps**: cortisol is clamped to [0, cortisol_max] and NE
    ///   to [0, ne_max] to prevent runaway stress/arousal.
    /// - **Split receptor adaptation**: desensitization (fast,
    ///   phosphorylation-based) and internalization (slow, membrane
    ///   removal) are modeled as separate factors.
    /// - **Restoring force**: 20% of adaptation rate (increased from 10%
    ///   for faster recovery from chronic stress).
    ///
    /// # New dynamics in v3.1
    ///
    /// - **HPA axis cascade**: CRH → ACTH → cortisol with time delays
    ///   and negative feedback. CRH drives ACTH production (delayed),
    ///   ACTH drives cortisol release (delayed), cortisol feeds back
    ///   negatively to CRH. This replaces direct cortisol manipulation
    ///   with a biologically accurate cascade through the pituitary
    ///   intermediate (ACTH).
    /// - **Tonic/phasic neurotransmitter distinction**: tonic_level
    ///   (slow background) and phasic_level (fast burst) are tracked
    ///   separately. Phasic decays quickly (seconds), tonic changes
    ///   slowly. For DA, tonic activates D2 (high-affinity) and phasic
    ///   activates D1 (low-affinity). For SRT, tonic activates 5-HT1A
    ///   and phasic activates 5-HT2A.
    /// - **Vesicular pool depletion**: impulses deplete the vesicular
    ///   pool, reducing the effect of subsequent impulses (synaptic
    ///   depression). Synthesis and reuptake slowly replenish the pool.
    /// - **Narcolepsy modeling**: when orexin is very low (<0.10),
    ///   histamine gets an extra downward force, modeling the sudden
    ///   sleep transitions (cataplexy) seen in narcolepsy.
    pub fn tick_with_params(&mut self, params: &NeuroTickParams) {
        // ─── Circuit breaker: sanitize all chemical fields ───────
        //
        // This is the last-resort defense against NaN/inf propagation.
        // If a non-finite value enters any chemical field between ticks
        // (via IPC, mmap race, corrupted model file, or a computation
        // bug we missed), the coupling matrix below would propagate
        // it to all 18 chemicals within a single tick, permanently
        // corrupting the mmap'd state.
        //
        // The circuit breaker scans every chemical's fields at the
        // START of the tick and resets any non-finite value to a safe
        // default. This confines the damage to one chemical (which
        // recovers via homeostatic forces) rather than letting it
        // poison the entire system.
        //
        // The defaults are chosen to be "neutral" — the chemical
        // recovers to its baseline via the normal homeostatic
        // mechanism over the next few ticks.
        for chem in &mut self.chemicals {
            let id = NeurochemicalId::from_u8(chem.id);
            let default_baseline = id.default_baseline();
            // Level: reset to baseline (the homeostatic set-point)
            if !chem.level.is_finite() {
                chem.level = default_baseline;
            }
            // Velocity: reset to zero (no momentum)
            if !chem.velocity.is_finite() {
                chem.velocity = 0.0;
            }
            // Baseline: reset to genetic default
            if !chem.baseline.is_finite() {
                chem.baseline = default_baseline;
            }
            // Tonic level: reset to baseline
            if !chem.tonic_level.is_finite() {
                chem.tonic_level = default_baseline;
            }
            // Phasic level: reset to zero (no burst)
            if !chem.phasic_level.is_finite() {
                chem.phasic_level = 0.0;
            }
            // Receptor sensitivity: reset to 1.0 (no adaptation)
            if !chem.receptor_sensitivity.is_finite() {
                chem.receptor_sensitivity = 1.0;
            }
            // Receptor subtypes: reset to defaults
            if !chem.receptor_subtypes[0].is_finite() || !chem.receptor_subtypes[1].is_finite() {
                chem.receptor_subtypes = id.default_receptor_subtypes();
            }
            // Desensitization: reset to 1.0 (no desensitization)
            if !chem.desensitization_factor.is_finite() {
                chem.desensitization_factor = 1.0;
            }
            // Internalization: reset to 1.0 (no internalization)
            if !chem.internalization_factor.is_finite() {
                chem.internalization_factor = 1.0;
            }
            // Vesicular pool: reset to 1.0 (full pool)
            if !chem.vesicular_pool.is_finite() {
                chem.vesicular_pool = 1.0;
            }
        }

        // Sanitize scalar state fields
        if !self.circadian_phase.is_finite() {
            self.circadian_phase = 0.0;
        }
        if !self.circadian_dt.is_finite() {
            self.circadian_dt = 1.0 / (24.0 * 3600.0 * 10.0); // 24h at 10Hz
        }
        if !self.arousal.is_finite() {
            self.arousal = 0.5;
        }
        if !self.valence.is_finite() {
            self.valence = 0.0;
        }
        if !self.global_tone.is_finite() {
            self.global_tone = 0.3;
        }
        if !self.plasticity_gate.is_finite() {
            self.plasticity_gate = 0.5;
        }
        if !self.acth_level.is_finite() {
            self.acth_level = 0.0;
        }
        if !self.gaba_a_allosteric.is_finite() {
            self.gaba_a_allosteric = 0.0;
        }
        // Sanitize the coupling matrix — NaN here would propagate
        // to all chemicals via the coupling force computation.
        for row in self.coupling_matrix.iter_mut() {
            for val in row.iter_mut() {
                if !val.is_finite() {
                    *val = 0.0;
                }
            }
        }
        // Sanitize effective levels
        for val in self.effective_levels.iter_mut() {
            if !val.is_finite() {
                *val = 0.0;
            }
        }

        let homeostatic_rate = params.homeostatic_rate;
        let damping = params.damping;
        let adaptation_rate = params.adaptation_rate;
        let baseline_adaptation_rate = params.baseline_adaptation_rate;
        // Validate dt at the dynamics boundary — the IPC handler clamps
        // dt to [0.001, 10.0], but this function can be called from
        // other paths (tests, advance_neuro). A NaN or negative dt would
        // produce a NaN dt_scale, corrupting every rate that uses it
        // (circadian phase, HPA cascade, receptor kinetics, etc.).
        // DT is 0.1, so dt_scale = dt / 0.1. Clamping dt to [0.001, 10.0]
        // gives dt_scale in [0.01, 100] — a wide but safe range.
        let dt = crate::state::sanitize::finite_clamp(params.dt, 0.001, 10.0);
        let dt_scale = dt / DT;
        let coupling_scale = params.coupling_scale;

        // ─── Circadian oscillator (SCN model) ──────────────────────
        //
        // The suprachiasmatic nucleus (SCN) is the master circadian
        // clock. It advances phase each tick and drives melatonin
        // secretion and circadian modulation of chemical baselines.
        //
        // The oscillator advances circadian_phase by circadian_dt each
        // tick, wrapping at 1.0. This represents the endogenous ~24h
        // rhythm that gates hormone release, sleep timing, and
        // cognitive performance.
        // Use f64 arithmetic for the slow circadian accumulator to avoid
        // drift from repeated f32 additions over many days.
        let new_phase =
            f64::from(self.circadian_phase) + f64::from(self.circadian_dt) * f64::from(dt_scale);
        self.circadian_phase = (new_phase % 1.0) as f32;
        if self.circadian_phase < 0.0 {
            self.circadian_phase += 1.0;
        }

        // Melatonin secretion: driven by the circadian phase. The
        // pineal gland secretes melatonin during the biological night
        // under SCN control. The melatonin_factor() gives a smooth [0,1]
        // envelope that is high at night and low during the day.
        //
        // We model this as a direct level adjustment: melatonin level
        // approaches a target set by the circadian factor. This is
        // distinct from the homeostatic baseline mechanism — melatonin
        // is not homeostatically regulated, it is rhythmically driven.
        let mel_factor = self.melatonin_factor();
        let mel_idx = NeurochemicalId::Melatonin as usize;
        let mel_target = mel_factor * 0.80; // peak ~0.80 at midnight
        let mel_approach_rate = 0.05 * dt_scale; // fast approach (~2 seconds)
        let mel_delta = (mel_target - self.chemicals[mel_idx].level) * mel_approach_rate;
        self.chemicals[mel_idx].level += mel_delta;
        self.chemicals[mel_idx].level =
            crate::state::sanitize::finite_clamp(self.chemicals[mel_idx].level, 0.0, 1.0);

        // Circadian baseline modulation: apply diurnal rhythm modifiers
        // to the homeostatic baselines of cortisol, dopamine, serotonin,
        // and histamine. This shifts the set-point that the homeostatic
        // force pulls toward, creating circadian variation in resting
        // levels.
        //
        // We compute circadian-modified baselines as a separate array
        // and use them ONLY in the homeostatic force calculation. The
        // coupling deviation calculation continues to use the original
        // (adapted) baselines, so circadian modulation doesn't weaken
        // the coupling dynamics. The baseline adaptation mechanism
        // also works on the original baselines, preserving emotional
        // development.
        let mut circadian_baselines = [0.0f32; NEUROCHEMICAL_COUNT];
        for (i, cb) in circadian_baselines
            .iter_mut()
            .enumerate()
            .take(NEUROCHEMICAL_COUNT)
        {
            let id = NeurochemicalId::from_u8(self.chemicals[i].id);
            let modifier = self.circadian_baseline_modifier(id);
            // Apply circadian modulation on top of the adapted baseline
            // (not the genetic default), so that sleep pressure accumulation
            // and emotional adaptation are preserved.
            *cb = crate::state::sanitize::finite_clamp(
                self.chemicals[i].baseline * modifier,
                0.05,
                0.95,
            );
        }

        // Recompute derived fields so the coupling snapshot reflects the
        // *current* chemical state, not the previous tick's cache. Between
        // ticks, apply_impulse / apply_impulse_capped (from IPC, active
        // inference, dyadic model, interoception) modify level and velocity
        // directly without recomputing effective_levels. Without this call,
        // the coupling matrix — the core of the 18×18 dynamics — would
        // operate on one-tick-stale effective levels every tick, miscomputing
        // coupling forces, eCB synthesis, and the HPA cascade drive.
        self.recompute_derived_with_dt(dt_scale);

        // Snapshot current effective levels for the coupling computation
        // (avoid in-place feedback during the step). Baselines are read
        // directly from self.chemicals[i].baseline inside the loop — no
        // need for a separate snapshot array since baselines don't change
        // during this step (only levels and velocities do).
        let eff_snapshot = self.effective_levels;

        // Dopamine subtype-weighted effective level for coupling.
        //
        // Dopamine's effect on other chemicals depends on the D1/D2
        // receptor balance. D1 (excitatory) and D2 (inhibitory) have
        // opposing downstream effects. When D1 dominates, dopamine's
        // coupling to arousal chemicals (NE, ACh, GLU) is amplified;
        // when D2 dominates, dopamine's coupling to inhibitory
        // chemicals (GABA) is amplified instead.
        //
        // Using the subtype-weighted level for coupling means the
        // same dopamine concentration can produce different effects
        // depending on receptor state — this is the pharmacological
        // reality of receptor-specific signaling pathways.
        let da_subtype_eff =
            self.chemicals[NeurochemicalId::Dopamine as usize].subtype_effective_level();

        // Region-specific SRT→DA coupling: override the matrix value
        // based on the active brain region. The default matrix has
        // SRT→DA = -0.05, but this coupling is always overridden by
        // the region-specific path below (Default = -0.025, the average
        // of the striatum and PFC paths), modeling 5-HT2A (striatum,
        // inhibitory) and 5-HT1A (PFC, facilitatory) receptor subtypes.
        let region = self.active_region();
        let srt_da_override = region.srt_to_da_coupling();
        let da_idx = NeurochemicalId::Dopamine as usize;
        let srt_idx = NeurochemicalId::Serotonin as usize;
        let glu_idx = NeurochemicalId::Glutamate as usize;
        let gaba_idx = NeurochemicalId::GABA as usize;

        // Update each chemical's velocity and level
        #[allow(clippy::needless_range_loop, reason = "i indexes multiple arrays")]
        for i in 0..NEUROCHEMICAL_COUNT {
            // Melatonin is driven by the circadian oscillator (the
            // melatonin approach mechanism above), NOT by homeostatic
            // regulation. Applying the homeostatic force would fight
            // the circadian approach, pulling melatonin toward its
            // 0.10 baseline even at night when the oscillator is
            // driving it toward 0.80. Skip the homeostatic force for
            // melatonin entirely — its level is set by the circadian
            // approach, and coupling forces from other chemicals
            // (NE, cortisol, orexin inhibition) still apply.
            let is_melatonin = i == NeurochemicalId::Melatonin as usize;
            // Cortisol and CRH are stress hormones, not
            // neurotransmitters. They have no homeostatic set-point —
            // their resting level is zero. They rise in response to
            // stressors (via coupling and the HPA cascade) and decay
            // via clearance mechanisms. Applying a homeostatic pull
            // toward a baseline would keep them elevated at rest,
            // which is the bug that caused chronic stress.
            // Instead, they get a constant decay toward zero
            // (leaky integrator), and coupling forces + the HPA
            // cascade push them up when stressors are present.
            let is_stress_hormone =
                i == NeurochemicalId::Cortisol as usize || i == NeurochemicalId::CRH as usize;
            // Adenosine has its own dedicated sleep-pressure mechanism
            // (saturating exponential accumulation during wakefulness,
            // glymphatic clearance during sleep) that directly manages
            // its level. During WAKEFULNESS, the homeostatic force
            // pulls adenosine toward its baseline (0.20), which is
            // ~34× stronger than the accumulation rate (0.055/hour).
            // This prevents adenosine from ever reaching the sleep
            // threshold (0.75), trapping the system in a "receptor
            // burnout" state where sleep can't be triggered and
            // receptors can't resensitize.
            //
            // During SLEEP, the homeostatic force is even more
            // destructive: it pulls the level (at ~0.75 from
            // accumulated wakefulness) toward the resting baseline
            // (0.20) at a rate amplified by the autoreceptor
            // nonlinear factor (up to 33× for large deviations).
            // This crashes the level from 0.75 to 0.20 in seconds,
            // triggering auto-wake almost immediately and preventing
            // any meaningful sleep. The glymphatic clearance is
            // designed to take ~7.6 hours (0.000002/tick), but the
            // homeostatic force overrides it entirely.
            //
            // Fix: skip the main loop entirely for adenosine during
            // sleep. The glymphatic clearance mechanism (below)
            // directly reduces the level at the configured clearance
            // rate, giving the correct hours-timescale sleep duration.
            let is_adenosine = i == NeurochemicalId::Adenosine as usize;
            let current_phase = MentalPhase::from_u8(self.emergent_phase);
            let is_sleeping =
                current_phase == MentalPhase::NREM || current_phase == MentalPhase::REM;
            if is_adenosine && is_sleeping {
                // Zero residual velocity from the last wakeful tick
                // so it doesn't push the level back up on the next
                // wakeful tick. The glymphatic clearance mechanism
                // below manages the level directly.
                self.chemicals[i].velocity = 0.0;
                continue;
            }
            // Cholinergic rebound: during sustained NREM (adenosine
            // has cleared below 0.65, ~83 min into sleep) or during
            // REM, ACh is managed by the dedicated cholinergic rebound
            // mechanism below — not by the homeostatic/coupling forces.
            // The homeostatic force (pulling toward 0.35) and the
            // coupling forces (GABA, melatonin suppressing ACh) would
            // prevent ACh from ever reaching the 0.50 REM threshold.
            // See the cholinergic rebound section after the adenosine
            // sleep pressure mechanism.
            let is_ach = i == NeurochemicalId::Acetylcholine as usize;
            if is_ach && is_sleeping {
                let adn_level = self.chemicals[NeurochemicalId::Adenosine as usize].level;
                let in_rebound = current_phase == MentalPhase::REM || adn_level < 0.65;
                if in_rebound {
                    self.chemicals[i].velocity = 0.0;
                    continue;
                }
            }
            let homeostatic_baseline = circadian_baselines[i];
            let homeostatic_force = if is_melatonin {
                0.0
            } else if is_stress_hormone {
                // Stress hormones have no homeostatic set-point — their
                // resting level is zero. They rise via coupling forces and
                // the HPA cascade, and decay via a first-order metabolic
                // clearance applied directly to level (below, after the
                // velocity/level update) plus the nonlinear
                // CORTISOL_CLEARANCE_RATE. The decay is NOT placed here
                // in homeostatic_force because that would integrate it
                // through velocity (a second-order path: velocity += force*dt;
                // level += velocity*dt), making the actual level change
                // proportional to dt² instead of dt — far weaker than the
                // documented per-second rate.
                0.0
            } else if is_adenosine {
                // Adenosine's level is managed by its dedicated
                // sleep-pressure mechanism: saturating exponential
                // accumulation during wakefulness, glymphatic
                // clearance during sleep. The homeostatic force
                // would fight both — preventing accumulation during
                // wakefulness (receptor burnout trap) and crashing
                // the level during sleep (see the continue above).
                0.0
            } else {
                let deviation = homeostatic_baseline - self.chemicals[i].level;
                // Autoreceptor-mediated self-inhibition:
                // the restoring force is purely the standard
                // homeostatic pull for small deviations (|dev| ≤
                // AUTORECEPTOR_THRESHOLD). Beyond that threshold,
                // autoreceptor occupancy rises quadratically and
                // increases self-inhibition. This preserves emergent
                // phase transitions (flow, stress, drowsy) and
                // coupling-driven stress responses near baseline,
                // while creating a "soft wall" at large deviations
                // that prevents chemicals from pinning at the 0.0/1.0
                // clamps.
                //
                // This is the permanent, biologically grounded
                // nonlinear stabilizer. The coupling matrix alone has
                // a spectral radius ~7.6× too large for global linear
                // stability (ev_C_max = 1.016, stability threshold =
                // hr/cs = 0.133). Emergent phases require that strong
                // coupling. Autoreceptors resolve this conflict by
                // leaving near-baseline dynamics (phases) essentially
                // linear while providing the strong self-inhibition
                // that bounds extreme states.
                //
                // Calibration (threshold=0.15, gain=200):
                // - At deviation 0.10: factor = 1.0 (purely linear — emergent dynamics)
                // - At deviation 0.20: factor = 1.5 (mild correction)
                // - At deviation 0.30: factor = 5.5 (moderate — breaks limit cycle)
                // - At deviation 0.50: factor = 25.5 (strong — prevents clamping)
                //
                // (Autoreceptor biology: McGinty & Harper, 1976;
                // Starke et al., 1989; Aghajanian, 1994)
                let abs_dev = deviation.abs();
                let excess = (abs_dev - AUTORECEPTOR_THRESHOLD).max(0.0);
                let nonlinear_factor = 1.0 + AUTORECEPTOR_GAIN * excess * excess;
                deviation * homeostatic_rate * nonlinear_factor
            };

            // Coupling force: sum of how all other chemicals' deviations
            // from their baselines influence this chemical, weighted by
            // the coupling matrix and using effective levels.
            //
            // The coupling force is scaled by coupling_scale to keep it
            // in balance with the homeostatic forces.
            //
            // **Depletion gating and deviation capping**: the coupling
            // force from chemical j is modified in two ways to prevent
            // the "death spiral" where depleted chemicals create large
            // deviations that produce strong coupling forces, causing
            // mutual collapse:
            //
            // 1. **Deviation cap**: negative deviations (depleted
            //    chemicals) are capped at -AUTORECEPTOR_THRESHOLD.
            //    This limits how much a depleted chemical can pull
            //    others down via "reduced excitation" or push cortisol
            //    up via "removal of inhibition." Positive deviations
            //    (elevated chemicals) are not capped — the stress
            //    response requires full coupling from high cortisol.
            //    The cap is set at the autoreceptor threshold because
            //    beyond that, the autoreceptor mechanism provides strong
            //    homeostatic recovery — the coupling force should not
            //    exceed what the homeostatic force can overcome.
            //
            // 2. **Depletion gate**: the coupling force from j is
            //    scaled by j's effective level relative to its baseline.
            //    At baseline → full coupling. At zero → 50% coupling.
            //    This further reduces the death spiral forces, providing
            //    a second layer of protection.
            //
            // Together, these two mechanisms ensure that:
            // - Stress responses work (high cortisol drives NE up via
            //   uncapped positive deviation)
            // - Emergent phases work (near-baseline dynamics are
            //   unaffected — deviations within ±0.15 are not capped)
            // - The system can recover from collapse (depleted chemicals
            //   can't pull each other down faster than the homeostatic
            //   force can recover them)
            let mut coupling_force = 0.0f32;
            // Self-deviation for state-dependent gain modulation
            let self_eff = eff_snapshot[i];
            let self_baseline = self.chemicals[i].baseline;
            let self_dev = self_eff - self_baseline;

            for (j, &eff) in eff_snapshot.iter().enumerate().take(NEUROCHEMICAL_COUNT) {
                if i == j {
                    continue;
                }
                // Use dopamine subtype-weighted level when dopamine
                // is the source chemical (j == da_idx). This makes
                // D1/D2 receptor balance affect how dopamine couples
                // to other chemicals — the same DA level produces
                // different coupling depending on receptor state.
                let eff_j = if j == da_idx { da_subtype_eff } else { eff };
                let j_baseline = self.chemicals[j].baseline;
                let raw_deviation = eff_j - j_baseline;
                // Cap negative deviations at -AUTORECEPTOR_THRESHOLD.
                // Positive deviations are not capped.
                let deviation = raw_deviation.max(-AUTORECEPTOR_THRESHOLD);

                // Nonlinear depletion gate: Hill function (n=2).
                //
                // Receptor signaling is not linear — it follows
                // sigmoid dose-response curves (Hill kinetics). The
                // Hill function with n=2 models cooperativity:
                // receptor binding has a threshold below which
                // coupling is weak, and above which it activates
                // steeply. This is the pharmacological reality —
                // neurotransmitters have EC50 thresholds, and below
                // those thresholds, coupling is subthreshold.
                //
                // The Hill gate replaces the previous linear gate
                // (0.5 + 0.5 * eff/baseline). At baseline, both
                // produce ~0.5. But the Hill function is sigmoid:
                // below baseline, coupling drops faster (subthreshold),
                // and above baseline, coupling rises faster
                // (suprathreshold activation). This produces genuine
                // nonlinear gating, not a linear scaling.
                //
                // Hill: eff^n / (EC50^n + eff^n)
                // EC50 = baseline (the half-activation point)
                // n = 2 (cooperativity)
                let level_gate = if j_baseline > 0.01 {
                    let eff_sq = eff_j * eff_j;
                    let ec50_sq = j_baseline * j_baseline;
                    eff_sq / (ec50_sq + eff_sq)
                } else {
                    1.0
                };

                // State-dependent gain modulation.
                //
                // The coupling between two chemicals is not constant —
                // it depends on the state of both. When both chemicals
                // are deviating in the same direction (both elevated
                // or both depleted), their coupling is amplified. When
                // they're deviating in opposite directions, coupling
                // is reduced. This models:
                //
                // - **Synergy**: co-activated chemicals (e.g., DA + NE
                //   during arousal) produce stronger effects together
                //   than either alone
                // - **Antagonism**: chemicals pulling in opposite
                //   directions (e.g., GLU up + GABA up) have reduced
                //   effective coupling
                //
                // The gain is bounded in [0.7, 1.3] to prevent
                // instability. The modulation is small (±30%) but
                // produces genuine nonlinear interaction that can't
                // be predicted from individual chemicals.
                let j_dev = raw_deviation;
                let coactivation = self_dev * j_dev; // positive if same direction
                let gain_mod = 1.0 + 0.3 * coactivation.tanh();
                let coupling_gain = crate::state::sanitize::finite_clamp(gain_mod, 0.7, 1.3);

                // Region-specific SRT→DA coupling override
                let coupling_val = if i == da_idx && j == srt_idx {
                    srt_da_override
                } else {
                    self.coupling_matrix[i][j]
                };

                coupling_force += coupling_val * deviation * level_gate * coupling_gain;
            }
            coupling_force *= coupling_scale;

            // Stochastic fluctuation (Ornstein-Uhlenbeck noise).
            //
            // The brain is never static — spontaneous neural activity
            // constantly perturbs neurotransmitter levels. This noise
            // term represents that activity, keeping all chemicals
            // gently fluctuating around their baselines rather than
            // settling to a fixed point.
            //
            // The noise is scaled by proximity to baseline: chemicals
            // near baseline get full noise (keeping them wandering),
            // while chemicals far from baseline get less noise (the
            // homeostatic force dominates). This prevents noise from
            // fighting strong homeostatic restoring forces.
            //
            // Seeded by (noise_seed, chemical_index) via a simple
            // hash — deterministic given the same seed sequence.
            let noise = if params.noise_amplitude > 0.0 && params.noise_seed > 0 {
                let hash = neuro_noise_hash(params.noise_seed, i as u64);
                let r01 = (hash as f32) / (u32::MAX as f32); // [0, 1)
                let r11 = r01 * 2.0 - 1.0; // [-1, 1)
                let deviation = (self.chemicals[i].level - homeostatic_baseline).abs();
                let proximity = (1.0 - deviation * 2.0).max(0.0); // 1 near baseline, 0 far
                params.noise_amplitude * r11 * proximity * dt
            } else {
                0.0
            };

            self.chemicals[i].velocity += (homeostatic_force + coupling_force) * dt + noise;
            self.chemicals[i].velocity *= damping.powf(dt_scale);
            self.chemicals[i].level += self.chemicals[i].velocity * dt;

            // Acetylcholinesterase degradation: ACh is broken down by
            // AChE in the synaptic cleft. This is a fast enzymatic
            // process (milliseconds) distinct from the slow homeostatic
            // restoring force. The degradation makes high ACh levels
            // decay faster, which is biologically correct.
            //
            // **Excess-only degradation**: only the portion ABOVE the
            // homeostatic baseline is degraded, not the absolute level.
            // The homeostatic force already represents the net
            // synthesis+degradation balance at rest (ChAT synthesis
            // matches AChE breakdown at baseline). Degrading the
            // absolute level double-counts the breakdown and creates
            // a death spiral: the direct first-order decay (rate 0.03/
            // tick) overwhelms the second-order homeostatic restoration
            // (force → velocity → level), so ACh settles at ~3% of
            // baseline instead of near baseline. This collapses arousal
            // and traps the system in delta/theta brain-wave states.
            //
            // By degrading only the excess, bursts still decay quickly
            // (the biological intent) while the homeostatic force can
            // maintain ACh near baseline at rest.
            if i == NeurochemicalId::Acetylcholine as usize {
                let ach_excess = (self.chemicals[i].level - homeostatic_baseline).max(0.0);
                let decay_factor = (1.0_f32 - ACH_DEGRADATION_RATE).powf(dt_scale);
                self.chemicals[i].level -= ach_excess * (1.0 - decay_factor);
            }

            // Stress hormone metabolic decay (leaky integrator).
            //
            // Cortisol and CRH have no homeostatic set-point — their
            // resting level is zero. They decay via a first-order
            // exponential process (half-life ~138 seconds at rate
            // 0.005/sec). This is applied directly to level (not through
            // velocity) so the decay is truly first-order: Δlevel ≈
            // -level × rate × dt. The cortisol-specific
            // CORTISOL_CLEARANCE_RATE (applied later in the HPA cascade
            // section) provides additional nonlinear clearance
            // proportional to level above baseline. Together they give
            // cortisol a half-life of ~5-7 seconds at moderate levels
            // when no stressor is active.
            //
            // Uses the exact exponential form exp(-rate×dt) rather than
            // the linear approximation 1-rate×dt, for time-invariance
            // and stability at large dt values.
            if is_stress_hormone {
                self.chemicals[i].level *= (-0.005 * dt).exp();
            }

            // Endocannabinoid activity-dependent synthesis: postsynaptic
            // neurons produce eCB when strongly activated by glutamate
            // (excitation) or GABA (inhibition). This is the canonical
            // trigger for retrograde endocannabinoid signaling (Kano et
            // al., 2009), providing a fast, local negative feedback that
            // dampens both excitatory and inhibitory input.
            if i == NeurochemicalId::Endocannabinoid as usize {
                let activity = eff_snapshot[glu_idx] + eff_snapshot[gaba_idx];
                let synthesis = activity * params.ecb_activity_coupling * dt;
                self.chemicals[i].level += synthesis;
            }

            self.chemicals[i].level =
                crate::state::sanitize::finite_clamp(self.chemicals[i].level, 0.0, 1.0);
        }

        // GABA disinhibition: in some circuits, GABA interneurons inhibit
        // other GABA interneurons, which disinhibits glutamate. When GABA
        // is very high (>0.8) AND ACh is high (>0.6), this disinhibition
        // mechanism slightly increases glutamate — paradoxically, very
        // high GABA can promote excitatory transmission through
        // disinhibition circuits.
        //
        // Uses effective levels (level × receptor sensitivity ×
        // desensitization × internalization) rather than raw levels.
        // If GABA receptors are downregulated, the raw concentration
        // may be high but the effective signaling is low — the
        // disinhibition should not trigger in that case, because the
        // GABA interneurons aren't actually signaling strongly enough
        // to inhibit their neighbors.
        let gaba_eff = self.effective(NeurochemicalId::GABA);
        let ach_eff = self.effective(NeurochemicalId::Acetylcholine);
        if gaba_eff > 0.8 && ach_eff > 0.6 {
            let disinhibition_force = (gaba_eff - 0.8) * (ach_eff - 0.6) * 0.5;
            self.chemicals[NeurochemicalId::Glutamate as usize].level += disinhibition_force * dt;
            self.chemicals[NeurochemicalId::Glutamate as usize].level =
                crate::state::sanitize::finite_clamp(
                    self.chemicals[NeurochemicalId::Glutamate as usize].level,
                    0.0,
                    1.0,
                );
        }

        // Adenosine sleep pressure: adenosine accumulates as a byproduct
        // of ATP metabolism during wakefulness (when histamine is high,
        // indicating an active cortex). During sleep (histamine low),
        // the glymphatic system clears it rapidly. This is the
        // neurobiological basis of sleep homeostasis (Basely et al.,
        // 2014; Porkka-Heiskanen et al., 2011).
        //
        // Without this mechanism, adenosine can never reach the sleep
        // threshold (0.75) naturally — the homeostatic force pulls it
        // back to baseline and all waking-state couplings push it down.
        // Sleep would only be triggerable by external impulses, defeating
        // the emergent phase transition design.
        //
        // The accumulation/clearance is gated by the emergent phase,
        // not just histamine level. This prevents the feedback loop
        // where temporary histamine dips trigger premature clearance:
        // adenosine rises → suppresses histamine → histamine dips →
        // triggers clearance → adenosine crashes → histamine rebounds.
        // By using the phase (which requires sustained adenosine > 0.75
        // AND histamine < 0.25), we ensure clearance only happens during
        // actual sleep, not transient histamine fluctuations.
        //
        // Implementation: during wakefulness, the adenosine LEVEL rises
        // as a saturating exponential (Process S). During sleep, the
        // glymphatic system directly clears adenosine from the
        // extracellular space, reducing the LEVEL at the configured
        // clearance rate. The homeostatic and coupling forces are
        // skipped for adenosine during sleep (see the `continue` in
        // the main loop above) — they would crash the level to the
        // resting baseline (0.20) in seconds, overriding the slow
        // (hours-timescale) glymphatic clearance and triggering
        // auto-wake almost immediately.
        //
        // The baseline stays at its resting value (0.20) throughout
        // sleep. When it wakes, the level is near 0.20 (after full
        // clearance) and the homeostatic force is re-enabled, but the
        // deviation is ~0 so there's no transient. The level then
        // begins accumulating again via the saturating exponential.
        //
        // Phase computation: we call recompute_derived() here to ensure
        // the effective levels (and thus the phase) reflect this tick's
        // updated raw levels. Without this, compute_phase() would use
        // stale effective levels from the previous tick, causing the
        // adenosine adjustment to operate on the wrong phase near
        // transitions. The subsequent recompute_derived() call in
        // sync_neurochemistry_to_state() will be redundant but harmless
        // (it recomputes the same values).
        self.recompute_derived_with_dt(dt_scale);
        let adn_idx = NeurochemicalId::Adenosine as usize;
        let current_phase = self.compute_phase();
        if current_phase == MentalPhase::NREM || current_phase == MentalPhase::REM {
            // Sleep: glymphatic clearance directly reduces the adenosine
            // LEVEL (the extracellular concentration drops as the
            // glymphatic system flushes adenosine from the brain).
            //
            // Rate is configurable via NeuroTickParams::adenosine_clearance_rate.
            // Biological default (0.000002/tick at 10Hz): from 0.75, takes
            // ~7.6 hours to return to 0.20, matching the ~8h human sleep
            // period during which adenosine is cleared by the glymphatic
            // system (Porkka-Heiskanen et al., 1997).
            // Compressed (0.0008/tick): ~1.1 minutes for testing.
            // Use f64 arithmetic for the slow adenosine level accumulator.
            let new_level = f64::from(self.chemicals[adn_idx].level)
                - f64::from(params.adenosine_clearance_rate) * f64::from(dt_scale);
            self.chemicals[adn_idx].level =
                crate::state::sanitize::finite_clamp(new_level as f32, 0.0, 1.0);
        } else {
            // Awake or drowsy: adenosine LEVEL rises as a saturating
            // exponential toward an upper asymptote, following the
            // two-process model of sleep regulation (Daan, Beersma &
            // Borbély, 1984).
            //
            // Process S rises during wakefulness as:
            //   S(t+dt) = U - e^(-r'·dt) · (U - S(t))
            // where U is the upper asymptote and r' = 0.055/h is the
            // rise rate constant (τ = 18.2h). This gives a fast initial
            // rise that slows as S approaches U — matching the
            // exponential buildup of EEG slow-wave activity during
            // wakefulness.
            //
            // Biologically, adenosine accumulates in the extracellular
            // space (the level), not the set-point (the baseline). The
            // baseline stays at its resting value (0.20) at all times
            // — it is not modified during sleep. During wakefulness,
            // the LEVEL rises as ATP metabolism produces adenosine.
            // During sleep, the LEVEL is directly reduced by glymphatic
            // clearance (see above). This separation means an external
            // adjustment to the baseline (e.g. the startup wake cascade
            // clearing residual sleep pressure) is permanent — the
            // accumulation pushes the level, not the baseline.
            //
            // The rise is scaled by (1.0 + activity_coupling * arousal)
            // so sleep pressure builds faster during high arousal —
            // adenosine is a byproduct of ATP metabolism
            // (Porkka-Heiskanen et al., 2011). During Drowsy,
            // accumulation continues at a reduced rate (0.5×) rather
            // than pausing: adenosine production does not stop when
            // you get sleepy.
            //
            // (Daan, Beersma & Borbély, 1984; Borbély, 1982)
            let drowsy_factor = if current_phase == MentalPhase::Drowsy {
                0.5
            } else {
                1.0
            };
            let activity_factor =
                1.0 + f64::from(params.adenosine_activity_coupling) * f64::from(self.arousal);
            // Saturating exponential: S(t+dt) = U - exp(-r'·dt) · (U - S(t))
            // r' is per-hour; dt_scale converts ticks to seconds, so
            // dt_hours = dt_scale * DT (seconds per tick) / 3600.
            // With DT=0.1s and dt_scale=2.0 (200ms tick), dt_hours = 0.0000556.
            let dt_hours = f64::from(dt_scale) * f64::from(DT) / 3600.0;
            let r_prime =
                f64::from(params.adenosine_accumulation_rate) * activity_factor * drowsy_factor;
            let upper_asymptote = 0.95; // leave headroom below 1.0
            let current_level = f64::from(self.chemicals[adn_idx].level);
            let new_level =
                upper_asymptote - (-r_prime * dt_hours).exp() * (upper_asymptote - current_level);
            self.chemicals[adn_idx].level =
                crate::state::sanitize::finite_clamp(new_level as f32, 0.0, 1.0);
        }
        self.chemicals[adn_idx].level =
            crate::state::sanitize::finite_clamp(self.chemicals[adn_idx].level, 0.0, 1.0);

        // Cholinergic rebound: during sustained NREM, acetylcholine
        // rebounds from its suppressed NREM level (~0.15-0.25) toward
        // the REM threshold (0.50+). This is the biological basis of
        // the NREM→REM transition within the ultradian cycle.
        //
        // During early NREM (adenosine still high, >0.65), ACh stays
        // low — the brain is in slow-wave sleep with glymphatic
        // clearance. As adenosine clears below 0.65 (~83 min into
        // sleep with biological rates), the cholinergic rebound
        // begins: ACh rises toward 0.65 (just above the 0.50 REM
        // threshold), enabling the NREM→REM transition.
        //
        // During REM, ACh is held high (0.65) to maintain the REM
        // state — the cholinergic/aminergic flip-flop (Hobson &
        // McCarley, 1975) is in its cholinergic phase.
        //
        // Without this mechanism, ACh's homeostatic force (pulling
        // toward 0.35) and the coupling forces (GABA, melatonin
        // suppressing ACh) prevent ACh from ever reaching 0.50 during
        // sleep. The Rust phase never enters REM, and the Python
        // lucid dream probability (which uses arousal as a REM
        // marker) never gets the REM boost — lucid dreaming becomes
        // nearly impossible.
        //
        // The rebound uses a saturating exponential toward the target
        // (0.65 during rebound/REM, 0.20 during early NREM), with a
        // time constant of ~10 minutes (biological) — slow enough to
        // respect the ultradian cycle (~90 min) but fast enough to
        // reach the threshold within the first REM window.
        //
        // The main loop above skips ACh during sleep (when in
        // rebound or REM) so the homeostatic/coupling forces don't
        // fight this mechanism — see the `continue` in the main loop.
        let ach_idx = NeurochemicalId::Acetylcholine as usize;
        if current_phase == MentalPhase::NREM || current_phase == MentalPhase::REM {
            let adn_level = self.chemicals[adn_idx].level;
            // Target ACh level depends on sleep stage:
            // - Early NREM (adenosine > 0.65): ACh stays low (0.20)
            //   for slow-wave sleep.
            // - Late NREM (adenosine ≤ 0.65): ACh rebounds toward
            //   0.65 to trigger the NREM→REM transition.
            // - REM: ACh is held at 0.65 to maintain the REM state.
            let ach_target = if adn_level < 0.65 || current_phase == MentalPhase::REM {
                0.65
            } else {
                0.20
            };
            // Saturating exponential toward the target.
            // Rate: 0.001/tick at 10Hz = 0.01/sec → τ ≈ 100 sec
            // (biological). With dt_scale=2.0 (200ms tick), this is
            // 0.002/tick → reaches 50% of target in ~350 ticks
            // (~70 sec). For compressed tests, the same rate gives
            // faster convergence in wall-clock time.
            let ach_rate = 0.001_f64 * f64::from(dt_scale);
            let current_ach = f64::from(self.chemicals[ach_idx].level);
            let new_ach =
                ach_target - (-ach_rate).exp() * (ach_target - current_ach);
            self.chemicals[ach_idx].level =
                crate::state::sanitize::finite_clamp(new_ach as f32, 0.0, 1.0);
        }

        // BDNF recovery after stress: when cortisol has dropped below
        // 0.3 (stress is over) and BDNF is still below its baseline
        // (depleted by chronic stress), BDNF gets a recovery boost
        // proportional to the serotonin level. Serotonin drives BDNF
        // expression — this is why SSRIs help depression (increasing
        // serotonin promotes BDNF recovery).
        //
        // This mechanism captures the neurobiological basis of stress
        // recovery: after chronic stress depletes BDNF (via cortisol
        // suppression), the recovery is driven by serotonin signaling,
        // not just passive homeostatic drift.
        let cort_level = self.chemicals[NeurochemicalId::Cortisol as usize].level;
        let srt_level = self.chemicals[NeurochemicalId::Serotonin as usize].level;
        let bdnf_baseline = self.chemicals[NeurochemicalId::BDNF as usize].baseline;
        let bdnf_level = self.chemicals[NeurochemicalId::BDNF as usize].level;
        if cort_level < 0.3 && bdnf_level < bdnf_baseline {
            let recovery = srt_level * params.bdnf_recovery_rate * dt_scale;
            self.chemicals[NeurochemicalId::BDNF as usize].level += recovery;
            self.chemicals[NeurochemicalId::BDNF as usize].level =
                crate::state::sanitize::finite_clamp(
                    self.chemicals[NeurochemicalId::BDNF as usize].level,
                    0.0,
                    1.0,
                );
        }

        // HPA axis cascade: CRH → ACTH → cortisol with time delays
        // and negative feedback.
        //
        // **Maturation gating**: the entire HPA cascade is gated by
        // `maturation_level`. During the stress hyporesponsive period
        // (maturation < 1.0), the cascade is attenuated or fully
        // dormant. This protects the developing brain from cortisol's
        // neurotoxic effects on synaptogenesis, ensuring maximal
        // plasticity during the training/teaching phase.
        //
        // At maturation 0.0: the cascade is fully dormant. CRH, ACTH,
        // and cortisol stay at zero. Stressors cannot produce cortisol.
        // At maturation 1.0: the cascade is fully online.
        //
        // In the biological HPA axis:
        // 1. CRH (hypothalamus) → ACTH (anterior pituitary) — fast
        // 2. ACTH → cortisol (adrenal cortex) — slower (minutes)
        // 3. Cortisol → negative feedback to CRH and ACTH — slow
        //
        // The cascade introduces time delays: CRH doesn't immediately
        // raise cortisol; it first raises ACTH, which then raises
        // cortisol. This creates a delayed, amplified stress response
        // that outlasts the initial trigger, and a slow negative
        // feedback that prevents runaway.
        //
        // The direct CRH→CORT coupling in the matrix is set to 0.0
        // (replaced by this cascade). Cortisol's negative feedback to
        // CRH (-0.10) remains in the matrix for the fast pathway.
        let maturation = crate::state::sanitize::finite_clamp(params.maturation_level, 0.0, 1.0);
        let crh_eff = self.effective_levels[NeurochemicalId::CRH as usize];
        let cort_eff = self.effective_levels[NeurochemicalId::Cortisol as usize];

        // Step 1: CRH drives ACTH production (delayed approach)
        // ACTH approaches a target set by CRH effective level, scaled
        // by maturation. When immature, ACTH can't rise even if CRH
        // is elevated — the pituitary is unresponsive.
        // Use exponential approach for time-invariance: rate 0.2/sec
        // → time constant ~5 seconds. The previous `* 0.02 * dt_scale`
        // was a per-tick rate scaled by dt_scale, which is only
        // correct at the reference tick rate.
        let acth_target = crh_eff * 0.8 * maturation;
        let acth_alpha = 1.0 - (-0.2 * dt).exp();
        self.acth_level += (acth_target - self.acth_level) * acth_alpha;
        self.acth_level = crate::state::sanitize::finite_clamp(self.acth_level, 0.0, 1.0);

        // Step 2: ACTH drives cortisol release
        // ACTH stimulates cortisol production in the adrenal cortex.
        // Scaled by maturation — the adrenal cortex is unresponsive
        // during the SHRP.
        //
        // Applied directly to level (first-order). Convention in this
        // file is `rate_per_tick * dt_scale` for direct level updates
        // (dt_scale = dt/DT, time-invariant). 0.001/tick at the 10 Hz
        // reference rate equals the documented 0.01/sec production
        // rate (0.01 * DT = 0.001), integrated once (not through
        // velocity, which would integrate twice and make the rate
        // proportional to dt² — far weaker than intended).
        //
        // Runaway is prevented by the nonlinear CORTISOL_CLEARANCE_RATE
        // (applied below, proportional to level above baseline) and the
        // first-order metabolic decay (0.005/sec, applied in the main
        // loop). At equilibrium: production = decay + clearance, which
        // for ACTH ~0.3 gives cortisol ~0.06 — elevated but well below
        // the 0.80 hard clamp. When the stressor ends and ACTH drops,
        // both decay mechanisms pull cortisol back to zero.
        let cort_idx = NeurochemicalId::Cortisol as usize;
        let acth_drive_per_tick = self.acth_level * 0.001 * maturation;
        self.chemicals[cort_idx].level += acth_drive_per_tick * dt_scale;

        // Compute cortisol excess once for the negative feedback
        // steps and clearance below.
        let cort_baseline = self.chemicals[cort_idx].baseline;
        let cort_excess = (self.chemicals[cort_idx].level - cort_baseline).max(0.0);

        // Step 3: Cortisol negative feedback to ACTH
        // High cortisol suppresses ACTH (long-loop negative feedback).
        // The HPA_CORTISOL_FEEDBACK_RATE (0.02) is 4× stronger than
        // the previous value (0.005), closing the negative feedback
        // loop on the same timescale as the ACTH response. This is
        // the primary brake on runaway cortisol.
        // `dt_scale` is correct here: this is a direct level update
        // (not a velocity force), so `rate * dt_scale = rate * dt / DT`
        // is time-invariant.
        self.acth_level -= cort_eff * HPA_CORTISOL_FEEDBACK_RATE * dt_scale;
        self.acth_level = crate::state::sanitize::finite_clamp(self.acth_level, 0.0, 1.0);

        // Step 3b: Cortisol negative feedback to CRH
        // High cortisol inhibits hypothalamic CRH neurons, shutting
        // off the stressor signal. This is already present in the
        // coupling matrix as CORT→CRH = -0.10, which propagates
        // through the velocity system. The strong ACTH negative
        // feedback above is the primary long-loop brake; adding a
        // direct CRH suppression here proved too strong and drove
        // resting CRH to zero, removing the tonic HPA drive. The
        // matrix entry provides the fast feedback without
        // overwhelming baseline CRH.

        // Step 4: Cortisol metabolic clearance (hepatic 11β-HSD).
        //
        // Cortisol is cleared from the bloodstream by the liver. The
        // clearance is nonlinear — proportional to how far cortisol
        // is ABOVE its baseline. This is the critical negative
        // feedback that prevents cortisol from getting stuck at the
        // hard clamp.
        //
        // At rest (cortisol = baseline): clearance = 0, no effect.
        // Under stress (cortisol >> baseline): clearance is strong,
        // creating a soft ceiling that prevents cortisol from pinning
        // at the 0.80 clamp. When the stressor ends and arousal
        // drops, the coupling forces weaken and clearance + the
        // homeostatic force pull cortisol back to baseline.
        //
        // This models the biological upregulation of 11β-HSD2 under
        // sustained high cortisol. See CORTISOL_CLEARANCE_RATE for
        // the full rationale.
        // `dt_scale` is correct here: direct level update, time-invariant.
        let eff_clearance = CORTISOL_CLEARANCE_RATE * cort_excess * dt_scale;
        // Clamp the clearance to at most 50% per tick to prevent
        // overshoot under large dt_scale values.
        self.chemicals[cort_idx].level *=
            1.0 - crate::state::sanitize::finite_clamp(eff_clearance, 0.0, 0.5);
        self.chemicals[cort_idx].level =
            crate::state::sanitize::finite_clamp(self.chemicals[cort_idx].level, 0.0, 1.0);

        // Tonic/phasic dynamics: phasic decays quickly (seconds),
        // tonic changes slowly (minutes). The tonic level tracks
        // the sustained concentration, while phasic represents
        // transient burst events that decay rapidly.
        //
        // For dopamine: tonic activates D2 (high-affinity, inhibitory,
        // mood maintenance), phasic activates D1 (low-affinity,
        // excitatory, reward signaling).
        // For serotonin: tonic activates 5-HT1A (high-affinity,
        // anxiolytic), phasic activates 5-HT2A (low-affinity,
        // excitatory).
        for chem in &mut self.chemicals {
            // Phasic decay: fast exponential decay (seconds timescale).
            // At 10Hz with 0.80 decay rate, phasic halves in ~3 ticks
            // (~0.3 seconds), reaching near-zero in ~10 ticks (~1 second).
            chem.phasic_level *= 0.80f32.powf(dt_scale);
            chem.phasic_level = crate::state::sanitize::finite_clamp(chem.phasic_level, 0.0, 1.0);

            // Tonic approach: slowly tracks the current level.
            // Rate 0.002/tick at 10Hz = 0.02/sec → time constant ~50
            // seconds. This is the slow background level that changes
            // over minutes, distinct from the fast phasic bursts.
            // Use f64 arithmetic for the slow tonic accumulator.
            let new_tonic = f64::from(chem.tonic_level)
                + (f64::from(chem.level) - f64::from(chem.tonic_level))
                    * 0.002_f64
                    * f64::from(dt_scale);
            chem.tonic_level = crate::state::sanitize::finite_clamp(new_tonic as f32, 0.0, 1.0);
        }

        // Vesicular pool recovery: synthesis and reuptake slowly
        // replenish the vesicular pool after depletion from impulses.
        // Synthesis recovery is slow (rate 0.001/tick → ~100 seconds
        // to fully replenish from empty). Reuptake recovery is
        // proportional to the current level (some neurotransmitter
        // is recycled back into vesicles).
        for chem in &mut self.chemicals {
            // Synthesis: slow linear recovery toward full pool
            chem.vesicular_pool += (1.0 - chem.vesicular_pool) * 0.001 * dt_scale;
            // Reuptake: proportional to current level (recycling)
            chem.vesicular_pool += chem.level * 0.0005 * dt_scale;
            chem.vesicular_pool =
                crate::state::sanitize::finite_clamp(chem.vesicular_pool, 0.0, 1.0);
        }

        // Narcolepsy modeling: when orexin is very low (<0.10), the
        // sleep-wake flip-flop becomes unstable — the system can
        // suddenly transition to sleep without sufficient adenosine
        // buildup. This models narcolepsy/cataplexy, where loss of
        // orexin neurons causes sudden sleep attacks.
        //
        // Orexin normally stabilizes the flip-flop by exciting
        // histamine and NE. Without orexin, histamine drops rapidly,
        // causing sudden sleep transitions even when adenosine is
        // not high enough to trigger normal sleep.
        let orexin_level = self.chemicals[NeurochemicalId::Orexin as usize].level;
        if orexin_level < 0.10 {
            // Sudden histamine drop (cataplexy/sleep attack)
            let narcolepsy_force = (0.10 - orexin_level) * 0.05 * dt_scale;
            self.chemicals[NeurochemicalId::Histamine as usize].level -= narcolepsy_force;
            self.chemicals[NeurochemicalId::Histamine as usize].level =
                crate::state::sanitize::finite_clamp(
                    self.chemicals[NeurochemicalId::Histamine as usize].level,
                    0.0,
                    1.0,
                );
        }

        // Hard clamps on cortisol and NE to prevent runaway stress/arousal.
        // These are safety mechanisms in case the emotional regulator fails
        // or coupling forces push beyond safe limits.
        // Use finite_clamp for NaN safety: native clamp passes NaN through.
        self.chemicals[NeurochemicalId::Cortisol as usize].level =
            crate::state::sanitize::finite_clamp(
                self.chemicals[NeurochemicalId::Cortisol as usize].level,
                0.0,
                params.cortisol_max,
            );
        self.chemicals[NeurochemicalId::Norepinephrine as usize].level =
            crate::state::sanitize::finite_clamp(
                self.chemicals[NeurochemicalId::Norepinephrine as usize].level,
                0.0,
                params.ne_max,
            );

        // Receptor adaptation: split into two mechanisms.
        //
        // 1. Desensitization (fast, phosphorylation-based, reversible in
        //    minutes): responds quickly to high levels. The
        //    desensitization_factor decreases when level > baseline and
        //    recovers toward 1.0 when level ≤ baseline.
        //
        // 2. Internalization (slow, membrane removal, reversible in hours):
        //    responds to sustained high levels. The internalization_factor
        //    decreases slowly when level > baseline and recovers very
        //    slowly toward 1.0 when level ≤ baseline.
        //
        // Additionally, receptor_sensitivity itself adapts (the original
        // mechanism, representing total receptor count changes over
        // days/weeks). Asymmetric rates: downregulation is fast,
        // upregulation is 3x slower.
        //
        // 3. Sleep receptor resensitization: during sleep, all three
        //    receptor mechanisms (sensitivity, desensitization,
        //    internalization) recover toward 1.0 UNCONDITIONALLY —
        //    not gated on level-baseline deviation. This is the
        //    biological basis of sleep restoration: the brain uses
        //    the low-activity window to restore receptor function
        //    regardless of neurotransmitter levels. Without this,
        //    the system enters a "receptor burnout trap" where
        //    chronic stress elevates raw levels, receptors
        //    downregulate, effective levels collapse, but raw
        //    levels stay above the drifted baselines so the
        //    deviation-gated recovery never triggers.
        //    (Vyazovskiy et al., 2008; Raczkowski et al., 2018)
        let sleep_resens = params.receptor_resensitization_rate;
        for chem in &mut self.chemicals {
            let chem_id = NeurochemicalId::from_u8(chem.id);
            let chem_adapt_rate = adaptation_rate * chem_id.receptor_adaptation_multiplier();

            let deviation = chem.level - chem.baseline;

            // Receptor adaptation: each receptor factor adapts based on
            // the effective deviation (raw deviation scaled by receptor
            // health). This ensures that downregulation is proportional
            // to actual receptor stimulation, not raw ligand
            // concentration. When receptors are burnt out, effective
            // deviation is small, so downregulation is weak — allowing
            // receptor turnover to slowly restore function.
            //
            // Receptor turnover (RECEPTOR_TURNOVER_RATE) provides a
            // slow background recovery toward 1.0 that models
            // continuous receptor replacement. It's normally
            // overwhelmed by deviation-gated downregulation, but when
            // downregulation is weak (burnout), turnover slowly
            // restores receptor function. See the constant docs above
            // for the full rationale and calibration.
            // Effective deviation: scale the raw deviation by receptor
            // health. Receptor downregulation is driven by receptor
            // activation (effective signaling), not by raw ligand
            // concentration. When receptors are burnt out, effective
            // deviation is small, so downregulation is weak — allowing
            // receptor turnover to slowly restore function.
            //
            // Biologically, receptor downregulation responds to receptor
            // stimulation (effective signaling), not raw concentration.
            // When receptors are burnt out, effective stimulation is low,
            // so there's no biological reason to downregulate further.
            let receptor_health = chem.receptor_sensitivity
                * chem.desensitization_factor
                * chem.internalization_factor;
            let effective_deviation = deviation * receptor_health;

            // Receptor turnover: slow background recovery toward 1.0.
            // Models continuous receptor replacement (half-life days
            // to weeks). Normally overwhelmed by deviation-gated
            // downregulation, but when downregulation is weak (low
            // effective deviation due to burnout), turnover slowly
            // restores receptor function.
            //
            // Each factor's turnover is proportional to its adaptation
            // rate (1.0×, 0.2×, 0.05× for sensitivity, desensitization,
            // internalization respectively). This ensures that:
            // - In normal overstimulation: desens/intern stay near 1.0
            //   (their slow turnover is overwhelmed by downregulation),
            //   keeping health high and sensitivity at its floor.
            // - In extreme burnout: desens/intern recover slowly enough
            //   that health stays low, allowing sensitivity turnover
            //   to overcome the weak (health-scaled) downregulation.
            let sens_turnover = RECEPTOR_TURNOVER_RATE * dt_scale;
            let desens_turnover = RECEPTOR_TURNOVER_RATE * 0.2 * dt_scale;
            let intern_turnover = RECEPTOR_TURNOVER_RATE * 0.05 * dt_scale;

            // Receptor sensitivity adaptation (slow, days/weeks)
            {
                let rate = if deviation > 0.0 {
                    chem_adapt_rate // downregulation: fast
                } else {
                    chem_adapt_rate * 0.3 // upregulation: 3x slower
                };
                // Use effective deviation for downregulation, raw
                // deviation for upregulation (upregulation should
                // respond to actual low levels, not effective levels).
                let adapt_deviation = if deviation > 0.0 {
                    effective_deviation
                } else {
                    deviation
                };
                chem.receptor_sensitivity -= adapt_deviation * rate * dt_scale;
            }
            // Sleep resensitization: unconditional recovery toward 1.0
            if sleep_resens > 0.0 {
                chem.receptor_sensitivity +=
                    (1.0 - chem.receptor_sensitivity) * sleep_resens * dt_scale;
            }
            // Receptor turnover (background recovery, proportional to
            // sensitivity's fast adaptation rate)
            chem.receptor_sensitivity += (1.0 - chem.receptor_sensitivity) * sens_turnover;
            // Floor at 0.5: maximum receptor density decrease is ~50%,
            // the worst physiological case reported in the literature.
            // Key data points:
            // - D2/D3 (chronic stimulant abuse): 10-25% reduction
            //   (Volkow et al.; meta-analysis effect size -0.76,
            //    Bertoletti et al., JAMA Psychiatry 2017)
            // - β2AR (chronic albuterol, 2 weeks): 22-42% pulmonary
            //   reduction (Hayes et al., 1996)
            // - Oxytocin receptor (10-day continuous infusion): ≥50%
            //   in all target fields (Endocrinology 130:2602, 1992)
            // - 5-HT2A (chronic SSRI): 11-38% decrease
            // - MOR (chronic morphine): "small reduction in total
            //   receptor number" — the 70-80% loss is functional
            //   desensitization, not density (Chen et al., PNAS 1989)
            // The previous 0.1 floor allowed 90% reduction, which
            // combined with desensitization and internalization floors
            // produced a 1000× effective level reduction — biologically
            // absurd. 0.5 ensures graceful degradation: even at maximum
            // burnout, effective levels retain enough signal for basic
            // function and self-recovery.
            chem.receptor_sensitivity =
                crate::state::sanitize::finite_clamp(chem.receptor_sensitivity, 0.5, 2.0);

            // Desensitization (fast, minutes): responds to current deviation.
            // Rate is much slower than receptor sensitivity — desensitization
            // operates on a minutes timescale, not seconds.
            // Use effective deviation for downregulation.
            let desens_rate = chem_adapt_rate * 0.2;
            {
                let adapt_deviation = if deviation > 0.0 {
                    effective_deviation
                } else {
                    deviation
                };
                if deviation > 0.0 {
                    chem.desensitization_factor -= adapt_deviation * desens_rate * dt_scale;
                } else {
                    // Recovery toward 1.0
                    chem.desensitization_factor +=
                        (-adapt_deviation) * desens_rate * 0.5 * dt_scale;
                }
            }
            // Sleep resensitization: unconditional recovery toward 1.0
            if sleep_resens > 0.0 {
                chem.desensitization_factor +=
                    (1.0 - chem.desensitization_factor) * sleep_resens * dt_scale;
            }
            // Receptor turnover (background recovery, proportional to
            // desensitization's moderate adaptation rate)
            chem.desensitization_factor += (1.0 - chem.desensitization_factor) * desens_turnover;
            // Floor at 0.2: maximum functional desensitization is ~80%,
            // per the literature on chronic agonist exposure:
            // - MOR (chronic morphine, 3+ days): 70-80% loss of
            //   receptor function via PKC-mediated uncoupling
            //   (Bailey et al., PMC2695152)
            // - β2AR (full agonist, epinephrine): ~70-80%
            //   desensitization of adenylyl cyclase response
            //   (January et al., JBC 1997)
            // - 5-HT1A (chronic SSRI): desensitization at G-protein
            //   coupling level, not receptor number (Blier et al.)
            // Note: this is functional uncoupling (phosphorylation/
            // arrestin), NOT receptor density loss — density is
            // handled by receptor_sensitivity above.
            chem.desensitization_factor =
                crate::state::sanitize::finite_clamp(chem.desensitization_factor, 0.2, 1.0);

            // Internalization (slow, hours): responds to sustained high levels.
            // Much slower than desensitization — membrane protein recycling
            // operates on an hours timescale.
            // Use effective deviation for downregulation.
            let intern_rate = chem_adapt_rate * 0.05;
            {
                let adapt_deviation = if deviation > 0.0 {
                    effective_deviation
                } else {
                    deviation
                };
                if deviation > 0.0 {
                    chem.internalization_factor -= adapt_deviation * intern_rate * dt_scale;
                } else {
                    // Very slow recovery toward 1.0
                    chem.internalization_factor +=
                        (-adapt_deviation) * intern_rate * 0.2 * dt_scale;
                }
            }
            // Sleep resensitization: unconditional recovery toward 1.0
            if sleep_resens > 0.0 {
                chem.internalization_factor +=
                    (1.0 - chem.internalization_factor) * sleep_resens * dt_scale;
            }
            // Receptor turnover (background recovery, proportional to
            // internalization's slow adaptation rate)
            chem.internalization_factor += (1.0 - chem.internalization_factor) * intern_turnover;
            // Floor at 0.45: maximum internalization is ~55%, per
            // the literature on agonist-induced receptor endocytosis:
            // - β2AR (epinephrine, full agonist, 30 min): 55%
            //   internalized (January et al., JBC 1997)
            // - MOR (DAMGO, 30 min): 50-58% internalized
            //   (Science Advances, 2025)
            // - β2AR (weak agonists): 0-17% — internalization is
            //   agonist-dependent
            // - Cardiac β-AR (isoproterenol): ~50% lost in 10 min,
            //   85% recycled in 20 min (rapid recovery)
            //   (Lohse et al., Circ Res 1984)
            // Previous 0.1 floor allowed 90% internalization, which
            // is far beyond any reported physiological maximum.
            chem.internalization_factor =
                crate::state::sanitize::finite_clamp(chem.internalization_factor, 0.45, 1.0);
        }

        // Baseline adaptation: the set-point itself drifts toward the
        // sustained level. This is emotional development — chronic
        // stress raises the cortisol baseline (the system "expects" more
        // stress), chronic well-being raises the serotonin baseline.
        //
        // A restoring force toward the genetic default prevents permanent
        // drift. Without this, chronic stress could permanently raise the
        // cortisol baseline with no recovery — the system would "forget"
        // its healthy set-point.
        //
        // The restoring force is equal to the adaptation rate (100%),
        // not 20% as in v2/v3. The previous 20% ratio was too weak: when
        // cortisol was high (level=1.0) and baseline had drifted to 0.6,
        // the deviation force (0.4 * rate) overwhelmed the restoring
        // force (0.08 * rate), so the baseline kept rising and stress
        // became self-reinforcing. With equal forces, the baseline
        // reaches an equilibrium that's the average of the current level
        // and the genetic default — it still adapts to sustained states,
        // but can't run away from the healthy set-point. Only very
        // prolonged, sustained states will shift the baseline
        // significantly, and recovery happens naturally when the
        // stressor ends.
        //
        // (Ramsay & Woods, 2014 — homeostatic regulation of set-points)
        for chem in &mut self.chemicals {
            // Adenosine has its own dedicated sleep-pressure mechanism
            // (metabolic accumulation / glymphatic clearance) that
            // directly modifies its baseline. The general baseline
            // adaptation would fight with that mechanism — pulling the
            // adenosine baseline back toward the genetic default while
            // the sleep pressure mechanism is trying to raise it. So
            // we skip adenosine here and let its dedicated mechanism
            // handle baseline drift.
            //
            // Cortisol and CRH are stress hormones with zero baseline
            // — their baselines should never adapt (they have no
            // homeostatic set-point, just a leaky integrator to zero).
            let chem_id = NeurochemicalId::from_u8(chem.id);
            if chem_id == NeurochemicalId::Adenosine
                || chem_id == NeurochemicalId::Cortisol
                || chem_id == NeurochemicalId::CRH
            {
                continue;
            }
            let deviation = chem.level - chem.baseline;
            let genetic_baseline = chem_id.default_baseline();
            let base_rate = baseline_adaptation_rate * dt_scale;
            // Restoring force toward the genetic baseline. The rate is
            // now 10x slower than the original, preventing the
            // self-reinforcing trap where high levels push baselines up,
            // weakening the homeostatic restoring force. The slower
            // rate gives the homeostatic force time to pull levels
            // toward genetic baselines before the baselines can adapt.
            let restoring = (genetic_baseline - chem.baseline) * base_rate;
            chem.baseline += deviation * base_rate + restoring;
            chem.baseline = crate::state::sanitize::finite_clamp(chem.baseline, 0.05, 0.95);
        }

        // ─── Metaplasticity: the coupling matrix self-modifies ───
        //
        // The coupling matrix defines how neurochemicals influence
        // each other. Metaplasticity means this matrix itself is
        // plastic — it adapts based on sustained emotional states.
        //
        // This is "the system learning how to learn." Chronic stress
        // reshapes how neurochemicals interact, making the system
        // less reactive (allostatic adaptation). Sustained wellbeing
        // makes it more responsive.
        //
        // Biological grounding:
        // - Chronic stress alters receptor expression and signaling
        //   pathways, changing how neurotransmitters interact (McEwen,
        //   2007). The amygdala becomes hyperreactive, the PFC
        //   becomes hyporeactive — the coupling structure changes.
        // - Sustained positive states strengthen synaptic connections
        //   (long-term potentiation is modulated by dopamine and
        //   serotonin).
        //
        // The adaptation uses TONIC levels (sustained over minutes),
        // not momentary levels, so it responds to emotional history
        // rather than transient fluctuations.
        //
        // The process is gated by plasticity_gate — metaplasticity
        // only occurs when BDNF is high and cortisol is low (the
        // system is in a learning-capable state). Chronic stress
        // suppresses BDNF, which suppresses plasticity_gate, which
        // suppresses metaplasticity — a protective mechanism that
        // prevents stress from permanently rewiring the brain.
        // Recovery (BDNF restoration via serotonin) re-enables
        // metaplasticity, allowing the system to re-adapt.
        //
        // A restoring force toward the default matrix prevents
        // permanent drift, modeling homeostatic regulation of the
        // coupling structure (Ramsay & Woods, 2014).
        let meta_rate = params.metaplasticity_rate;
        if meta_rate > 0.0 && self.plasticity_gate > 0.1 {
            // Emotional context from tonic (sustained) levels,
            // measured as deviation from baseline. This ensures
            // neutral states produce zero adaptation signal.
            let cort_idx = NeurochemicalId::Cortisol as usize;
            let da_idx_m = NeurochemicalId::Dopamine as usize;
            let srt_idx_m = NeurochemicalId::Serotonin as usize;

            let cort_tonic = self.chemicals[cort_idx].tonic_level;
            let cort_base = self.chemicals[cort_idx].baseline;
            let da_tonic = self.chemicals[da_idx_m].tonic_level;
            let da_base = self.chemicals[da_idx_m].baseline;
            let srt_tonic = self.chemicals[srt_idx_m].tonic_level;
            let srt_base = self.chemicals[srt_idx_m].baseline;

            // Stress signal: cortisol tonic above baseline → stress
            // Wellbeing signal: DA + SRT tonic above baseline, cortisol at/below baseline
            //
            // A small dead zone prevents sub-threshold tonic drift from
            // producing spurious metaplasticity. Metaplasticity should
            // respond to sustained, meaningful emotional deviations, not
            // numerical noise or slow equilibrium drift.
            let dead_zone = 0.05;
            let stress =
                crate::state::sanitize::finite_clamp(cort_tonic - cort_base - dead_zone, 0.0, 1.0);
            let wellbeing =
                (crate::state::sanitize::finite_clamp(da_tonic - da_base - dead_zone, 0.0, 1.0)
                    + crate::state::sanitize::finite_clamp(
                        srt_tonic - srt_base - dead_zone,
                        0.0,
                        1.0,
                    ))
                    * 0.5
                    * (1.0 - stress);

            // Gate by plasticity (BDNF-dependent)
            let gate = self.plasticity_gate * meta_rate * dt_scale;

            for (i, (default_row, current_row)) in DEFAULT_COUPLING_MATRIX
                .iter()
                .zip(self.coupling_matrix.iter_mut())
                .enumerate()
            {
                for (j, (&default_val, current)) in
                    default_row.iter().zip(current_row.iter_mut()).enumerate()
                {
                    if i == j {
                        continue;
                    }
                    if default_val == 0.0 {
                        continue;
                    }

                    // Restoring force toward default (75% of adaptation).
                    // Strengthened relative to the emotional stress/wellbeing
                    // adjustments so that sub-threshold drift and feedback
                    // loops cannot permanently lock the matrix away from
                    // its homeostatic set-point.
                    let restoring = (default_val - *current) * gate * 0.75;

                    // Stress adaptation: weaken excitatory (positive)
                    // couplings, strengthen inhibitory (negative) ones.
                    // The brain becomes less reactive under chronic
                    // stress — excitatory pathways are dampened, while
                    // inhibitory pathways are reinforced to maintain
                    // stability.
                    //
                    // Both adjustments are negative: excitatory couplings
                    // are reduced (toward zero), and inhibitory couplings
                    // are made more negative (away from zero). The
                    // inhibitory strengthening is half the excitatory
                    // weakening, matching the asymmetry in the original
                    // design.
                    let stress_adj = if *current > 0.0 {
                        -stress * gate * current.abs()
                    } else {
                        -stress * gate * current.abs() * 0.5
                    };

                    // Wellbeing adaptation: strengthen excitatory
                    // couplings slightly. The brain becomes more
                    // responsive when things are going well.
                    let wellbeing_adj = if *current > 0.0 {
                        wellbeing * gate * 0.5
                    } else {
                        0.0
                    };

                    *current += restoring + stress_adj + wellbeing_adj;

                    // Clamp to prevent extreme values
                    let max = default_val.abs().max(0.5) * 2.0;
                    *current = crate::state::sanitize::finite_clamp(*current, -max, max);
                }
            }
        }

        // Recompute all derived fields with the new state
        self.recompute_derived_with_dt(dt_scale);
    }

    /// Advance the neurochemical system by one tick with explicit
    /// rate parameters.
    ///
    /// This is a convenience wrapper around [`tick_with_params`] that
    /// constructs a [`NeuroTickParams`] from the individual parameters.
    /// The `dt` and `coupling_scale` use the default constants.
    ///
    /// # Parameters
    /// - `homeostatic_rate`: how fast chemicals return to baseline
    ///   (typically 0.001–0.01)
    /// - `damping`: velocity decay per tick (typically 0.85–0.95)
    /// - `adaptation_rate`: how fast receptor sensitivity adapts
    ///   (typically 0.0001–0.001, much slower than homeostatic drift)
    /// - `baseline_adaptation_rate`: how fast baselines shift
    ///   (typically 0.00001–0.0001, slower still — this is emotional
    ///   development over hours/days)
    pub fn tick(
        &mut self,
        homeostatic_rate: f32,
        damping: f32,
        adaptation_rate: f32,
        baseline_adaptation_rate: f32,
    ) {
        let params = NeuroTickParams {
            homeostatic_rate,
            damping,
            adaptation_rate,
            baseline_adaptation_rate,
            ..NeuroTickParams::DEFAULT
        };
        self.tick_with_params(&params);
    }

    /// Convenience: tick with default parameters suitable for a
    /// ~100ms update interval.
    pub fn tick_default(&mut self) {
        self.tick_with_params(&NeuroTickParams::DEFAULT);
    }
}

// ─────────────────────────────────────────────────────────────────
//  NeuroTickParams
// ─────────────────────────────────────────────────────────────────

/// Customizable parameters for the neurochemical dynamics tick.
///
/// This struct encapsulates all tunable parameters for the tick
/// dynamics, allowing different configurations without changing the
/// core algorithm. The default values are tuned for a 10 Hz tick loop
/// (DT = 0.1 seconds).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NeuroTickParams {
    /// How fast chemicals return to their baseline (typically 0.02–0.05).
    /// This is the PRIMARY regulatory mechanism — synthesis, release,
    /// reuptake, and autoreceptor feedback maintain baseline levels.
    /// Neuromodulatory coupling effects are SECONDARY and should not
    /// overwhelm homeostasis. At 0.04, the homeostatic force dominates
    /// the sum of coupling forces even when multiple chemicals are
    /// depleted simultaneously, preventing the "death spiral" where
    /// depleted chemicals hold each other at zero via coupling.
    /// (Homeostatic regulation: Marder & Goaillard, Annu Rev Physiol 2006;
    ///  Autoreceptor dynamics: Hashemi et al., BMC Neurosci 2020)
    pub homeostatic_rate: f32,
    /// Velocity decay per tick (typically 0.85–0.95).
    pub damping: f32,
    /// How fast receptor sensitivity adapts (typically 0.0001–0.001).
    pub adaptation_rate: f32,
    /// How fast baselines shift toward sustained levels
    /// (typically 0.00001–0.0001 — emotional development over hours/days).
    pub baseline_adaptation_rate: f32,
    /// Time step in seconds (0.1 = 100ms, matching the 10 Hz tick loop).
    pub dt: f32,
    /// Global scaling factor for coupling forces (typically 0.15).
    /// Brings coupling forces into balance with homeostatic forces.
    pub coupling_scale: f32,
    /// BDNF recovery rate after stress (typically 0.001).
    pub bdnf_recovery_rate: f32,
    /// Hard clamp for cortisol maximum (prevents runaway stress).
    pub cortisol_max: f32,
    /// Hard clamp for norepinephrine maximum (prevents runaway arousal).
    pub ne_max: f32,
    /// Adenosine baseline rise rate constant (per hour) during wakefulness.
    /// This is r' from the two-process model (Daan, Beersma & Borbély,
    /// 1984), where Process S rises as a saturating exponential:
    ///   S(t+dt) = U - e^(-r'·dt) · (U - S(t))
    /// The biological default is r' = 0.055/h (τ = 18.2 hours), derived
    /// from EEG slow-wave activity buildup during wakefulness.
    ///
    /// The actual rate is multiplied by an activity factor (see
    /// [`adenosine_activity_coupling`]) so sleep pressure builds faster
    /// during high arousal, and by a drowsy factor (0.5× during Drowsy).
    ///
    /// For compressed-time testing, use
    /// [`NeuroTickParams::COMPRESSED`] which sets this to 5.5 (100× faster).
    pub adenosine_accumulation_rate: f32,
    /// Adenosine baseline clearance rate per tick during sleep.
    /// The adenosine baseline drops by this amount each tick during
    /// NREM/REM sleep (glymphatic clearance).
    ///
    /// At 10 Hz, the biological default (0.000002) gives:
    /// 0.55 / 0.000002 = 275,000 ticks = ~7.6 hours to clear from 0.75
    /// back to 0.20, matching the human ~8h sleep period.
    ///
    /// For compressed-time testing, use
    /// [`NeuroTickParams::COMPRESSED`] which sets this to 0.0008
    /// (~1.1 min clearance).
    pub adenosine_clearance_rate: f32,
    /// Adenosine activity coupling factor.
    ///
    /// Adenosine is a byproduct of ATP metabolism, so its production
    /// should scale with recent neural/metabolic activity. This factor
    /// multiplies the base [`adenosine_accumulation_rate`] by
    /// `(1.0 + adenosine_activity_coupling * arousal)` during wakefulness.
    /// A value of 0.0 disables the activity dependence; 0.5 means sleep
    /// pressure builds at 1.0–1.5× the base rate depending on arousal.
    ///
    /// This captures the well-established finding that sustained
    /// cognitive or physical effort shortens sleep latency by increasing
    /// adenosine (Porkka-Heiskanen et al., 2011).
    pub adenosine_activity_coupling: f32,
    /// Endocannabinoid activity-coupled synthesis rate.
    ///
    /// Endocannabinoids are synthesized on demand by postsynaptic neurons
    /// when strong excitatory or inhibitory input depolarizes them. This
    /// rate controls how much eCB level rises per tick from the combined
    /// effective activity of glutamate and GABA.
    ///
    /// 0.0 disables activity-dependent synthesis; 0.01 means a highly
    /// active cortex can push eCB ~50% above its homeostatic baseline
    /// through retrograde feedback, producing a mild negative-feedback
    /// brake on excitation/inhibition (Kano et al., 2009).
    pub ecb_activity_coupling: f32,
    /// Metaplasticity rate: how fast the coupling matrix itself adapts
    /// based on sustained emotional states (typically 0.000001–0.00001).
    ///
    /// This is the learning rate for the coupling matrix self-modification.
    /// Chronic stress (sustained high cortisol) weakens excitatory
    /// couplings and strengthens inhibitory ones — the brain becomes
    /// less reactive under chronic stress (allostatic adaptation).
    /// Sustained wellbeing (high dopamine + serotonin, low cortisol)
    /// slightly strengthens excitatory couplings — the brain becomes
    /// more responsive when things are going well.
    ///
    /// The process is gated by plasticity_gate (BDNF × (1 - cortisol×0.5)),
    /// so metaplasticity only occurs when the system is plastic.
    /// A restoring force toward the default matrix prevents permanent
    /// drift.
    ///
    /// At 10 Hz, the default (0.000005) gives noticeable adaptation over
    /// ~hours of sustained emotional states, matching biological
    /// timescales for neurotransmitter receptor changes.
    pub metaplasticity_rate: f32,
    /// HPA axis maturation level [0.0, 1.0].
    ///
    /// Models the **stress hyporesponsive period (SHRP)** — the
    /// developmental window during which the HPA axis is dampened to
    /// protect the developing brain from cortisol's neurotoxic effects
    /// on synaptogenesis and neurogenesis.
    ///
    /// At 0.0 (newborn): the HPA axis is fully dormant. CRH, ACTH,
    /// and cortisol stay at zero regardless of stressors. This is
    /// maximal plasticity — every experience is encoded without
    /// stress interference. This is the state during initial training
    /// and teaching.
    ///
    /// At 1.0 (mature): the HPA axis is fully online. Stressors
    /// produce CRH → ACTH → cortisol cascades as designed.
    ///
    /// Between 0 and 1: the HPA axis is partially active. Stress
    /// responses are attenuated, scaling linearly with maturation.
    /// This gradual onset models the progressive maturation of the
    /// HPA axis through infancy and childhood.
    ///
    /// The daemon computes this from `ltm_episode_count` — the
    /// accumulated episodic experience. The threshold for full
    /// maturation is ~10,000 episodes (configurable), representing
    /// roughly the amount of experience needed for a basic
    /// self-model and language foundation.
    ///
    /// (Levine, 1994 — stress hyporesponsive period; Walker et al.,
    /// 1986 — HPA development; Sapolsky, 1996 — glucocorticoid
    /// neurotoxicity and brain development)
    pub maturation_level: f32,
    /// Amplitude of stochastic neurotransmitter fluctuations (typically
    /// 0.0–0.003).
    ///
    /// The brain is never static — every neurotransmitter constantly
    /// fluctuates due to spontaneous neural activity. Serotonin
    /// oscillates with ~10 min periods (FSCAV data), dopamine has
    /// spontaneous impulses at ~0.01/sec (Mohebi et al., 2019), and
    /// DA/ACh co-fluctuate at ~2 Hz (Howe et al., 2023).
    ///
    /// This parameter controls the amplitude of Ornstein-Uhlenbeck
    /// noise added to each chemical's velocity per tick. The noise
    /// is seeded by `noise_seed` (which the daemon increments each
    /// tick) so it's deterministic given the same seed sequence.
    ///
    /// At 0.0: the system is deterministic (use for tests).
    /// At 0.002: visible fluctuations of ±0.01–0.02 around baseline,
    ///   keeping all chemicals gently oscillating. This matches the
    ///   observed variability in tonic neurotransmitter levels and
    ///   prevents the system from settling to a static fixed point.
    ///
    /// The noise is scaled per-chemical: chemicals far from baseline
    /// get less noise (the homeostatic force dominates), while
    /// chemicals near baseline get full noise (keeping them wandering).
    pub noise_amplitude: f32,
    /// Seed for the stochastic noise process. The daemon should
    /// increment this each tick to produce a fresh random sequence.
    /// At 0, noise is disabled regardless of `noise_amplitude`.
    pub noise_seed: u64,
    /// Receptor resensitization rate during sleep (typically 0.0).
    ///
    /// During sleep, the brain restores receptor function independent
    /// of neurotransmitter levels — receptors resensitize and
    /// re-internalize toward 1.0 regardless of the level-baseline
    /// deviation. This is the biological basis of sleep restoration:
    /// chronic wakefulness causes receptor downregulation
    /// (desensitization + internalization), and sleep reverses it.
    ///
    /// The daemon sets this to a nonzero value during sleep (NREM/REM
    /// or Sleeping zone) and 0.0 during wakefulness. The rate is
    /// calibrated so that a full sleep cycle (~90 minutes biological
    /// time) can restore receptors from maximum burnout (0.2
    /// desensitization, 0.45 internalization) back to near 1.0.
    ///
    /// Without this, the system enters a "receptor burnout trap":
    /// chronic stress elevates raw levels → receptors downregulate →
    /// effective levels collapse → but raw levels stay high (above
    /// the drifted baselines) → desensitization can't recover because
    /// the deviation is still positive. Sleep breaks this trap by
    /// restoring receptors unconditionally.
    ///
    /// (Vyazovskiy et al., 2008 — sleep-wake history affects receptor
    /// sensitivity; Raczkowski et al., 2018 — sleep restores
    /// adrenergic receptor sensitivity)
    pub receptor_resensitization_rate: f32,
}

impl NeuroTickParams {
    /// Default parameters for biological-time dynamics.
    ///
    /// Adenosine accumulation and clearance rates are calibrated to
    /// match the 24h circadian oscillator: ~16h awake → sleep
    /// threshold, ~8h sleep → clearance. This is the production
    /// configuration for a 24/7 running Genesis.
    pub const DEFAULT: Self = Self {
        homeostatic_rate: 0.04,
        damping: 0.85,
        adaptation_rate: 0.0005,
        baseline_adaptation_rate: 0.00005,
        dt: DT,
        coupling_scale: COUPLING_SCALE,
        bdnf_recovery_rate: 0.001,
        cortisol_max: 0.80,
        ne_max: 0.90,
        adenosine_accumulation_rate: 0.055, // r' = 0.055/h (Daan et al. 1984, τ=18.2h)
        adenosine_clearance_rate: 0.000002, // ~7.6h to clear (Porkka-Heiskanen 1997)
        adenosine_activity_coupling: 0.5,   // sleep pressure scales with arousal
        ecb_activity_coupling: 0.01,        // activity-dependent eCB synthesis
        metaplasticity_rate: 0.000005,      // hours-timescale coupling adaptation
        maturation_level: 0.0,              // newborn — HPA dormant, set by daemon
        noise_amplitude: 0.002,             // gentle stochastic fluctuations
        noise_seed: 0,                      // daemon increments each tick
        receptor_resensitization_rate: 0.0, // set by daemon during sleep
    };

    /// Compressed-time parameters for testing and fast iteration.
    ///
    /// Same as [`DEFAULT`] but with adenosine rates ~100x faster,
    /// giving ~9 min sleep latency and ~1 min clearance. Use this
    /// for tests that need to observe sleep/wake cycles within a
    /// reasonable test duration. NOT for production use — the
    /// compressed sleep cycle is inconsistent with the 24h
    /// circadian oscillator.
    pub const COMPRESSED: Self = Self {
        adenosine_accumulation_rate: 5.5, // 100x faster for testing (τ=0.18h ≈ 11min)
        adenosine_clearance_rate: 0.0008, // ~1.1 min to clear
        metaplasticity_rate: 0.00005,     // ~10x faster for testing
        ..Self::DEFAULT
    };
}

impl Default for NeuroTickParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

// ─────────────────────────────────────────────────────────────────
//  Active region
// ─────────────────────────────────────────────────────────────────

/// Brain region for region-specific coupling (serotonin→dopamine).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveRegion {
    /// Default: use average of striatum and PFC coupling paths.
    Default = 0,
    /// Striatum: 5-HT2A-mediated inhibitory SRT→DA coupling (-0.20).
    Striatum = 1,
    /// Prefrontal cortex: 5-HT1A-mediated facilitatory SRT→DA (+0.15).
    Pfc = 2,
}

impl ActiveRegion {
    /// Convert a `u8` into an `ActiveRegion`, defaulting to
    /// [`Default`](Self::Default) for unknown values.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Striatum,
            2 => Self::Pfc,
            _ => Self::Default,
        }
    }

    /// Region-specific serotonin→dopamine coupling value.
    pub const fn srt_to_da_coupling(self) -> f32 {
        match self {
            Self::Striatum => -0.20, // 5-HT2A, inhibitory
            Self::Pfc => 0.15,       // 5-HT1A, facilitatory
            Self::Default => -0.025, // average of the two paths
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compile-time layout assertions
// ─────────────────────────────────────────────────────────────────

const _: () = {
    use core::mem::offset_of;
    // Neurochemical (56 bytes)
    assert!(core::mem::size_of::<Neurochemical>() == 56);
    assert!(offset_of!(Neurochemical, last_shift_at) == 0);
    assert!(offset_of!(Neurochemical, level) == 8);
    assert!(offset_of!(Neurochemical, baseline) == 12);
    assert!(offset_of!(Neurochemical, receptor_sensitivity) == 16);
    assert!(offset_of!(Neurochemical, velocity) == 20);
    assert!(offset_of!(Neurochemical, receptor_subtypes) == 24);
    assert!(offset_of!(Neurochemical, desensitization_factor) == 32);
    assert!(offset_of!(Neurochemical, internalization_factor) == 36);
    assert!(offset_of!(Neurochemical, tonic_level) == 40);
    assert!(offset_of!(Neurochemical, phasic_level) == 44);
    assert!(offset_of!(Neurochemical, vesicular_pool) == 48);
    assert!(offset_of!(Neurochemical, id) == 52);

    // CouplingMatrix (1296 bytes = 18×18×4)
    assert!(core::mem::size_of::<CouplingMatrix>() == 1296);

    // NeurochemicalVector (2416 bytes)
    assert!(core::mem::size_of::<NeurochemicalVector>() == 2416);
    assert!(offset_of!(NeurochemicalVector, chemicals) == 0);
    assert!(offset_of!(NeurochemicalVector, coupling_matrix) == 1008);
    assert!(offset_of!(NeurochemicalVector, effective_levels) == 2304);
    assert!(offset_of!(NeurochemicalVector, global_tone) == 2376);
    assert!(offset_of!(NeurochemicalVector, arousal) == 2380);
    assert!(offset_of!(NeurochemicalVector, valence) == 2384);
    assert!(offset_of!(NeurochemicalVector, plasticity_gate) == 2388);
    assert!(offset_of!(NeurochemicalVector, gaba_a_allosteric) == 2392);
    assert!(offset_of!(NeurochemicalVector, bdnf_recovery_rate) == 2396);
    assert!(offset_of!(NeurochemicalVector, acth_level) == 2400);
    assert!(offset_of!(NeurochemicalVector, circadian_phase) == 2404);
    assert!(offset_of!(NeurochemicalVector, circadian_dt) == 2408);
    assert!(offset_of!(NeurochemicalVector, emergent_phase) == 2412);
    assert!(offset_of!(NeurochemicalVector, active_region) == 2413);
};

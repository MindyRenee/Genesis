//! Basal forebrain — the cholinergic output equation and its modulators.
//!
//! ════════════════════════════════════════════════════════════════════
//! ANATOMY
//! ════════════════════════════════════════════════════════════════════
//!
//! The basal forebrain lies ventral to the striatum and anterior to the
//! hypothalamus. It holds the major cholinergic projection to the cortex
//! and hippocampus: Ch1/Ch2 (medial septum, vertical diagonal band)
//! supply the hippocampal complex, Ch4 (nucleus basalis of Meynert)
//! supplies neocortex and amygdala. It is also a sleep-wake structure:
//! three intermingled populations — wake/REM-active cholinergic,
//! parvalbumin GABAergic, and glutamatergic neurons organized in a
//! Glu→Ch→PV excitatory hierarchy — are all inhibited by somatostatin+
//! GABAergic sleep-promoting neurons (Xu et al., Nat. Neurosci. 2015),
//! and it is the best-characterized site where the homeostatic sleep
//! signal acts: extracellular adenosine rises there with time awake
//! (Porkka-Heiskanen et al., Science 1997) and A1-receptor activation
//! inhibits the region's wake-active neurons (Strecker et al.,
//! J. Neurosci. 2000; Basheer et al., 2004). The sleep signal literally
//! silences the arousal output this same region produces.
//!
//! ════════════════════════════════════════════════════════════════════
//! THE OUTPUT EQUATION
//! ════════════════════════════════════════════════════════════════════
//!
//! The region's release level is a function of the sleep-wake network's
//! populations plus two modulators:
//!
//!   target = ACH_BASE
//!          + ACH_FROM_LC · lc · (1 − A1) · D(α)
//!          + ACH_FROM_REM · rem
//!
//! - The linear map of `lc` (wake) and `rem` (REM-on) reproduces the
//!   microdialysis steady states — NREM 0.34, waking 0.58, REM 0.77
//!   (Vazquez & Baghdoyan 2001) — see `state::neurochemical` for the
//!   calibrated coefficients.
//! - `A1` is adenosine-A1-receptor occupancy on the *wake-active*
//!   cholinergic population: the somnogenic signal attenuates only the
//!   LC-driven component, so the calibrated sleep profiles (which have
//!   no LC drive) are preserved exactly while waking cortical ACh sags
//!   as pressure builds.
//! - `D(α)` is the expected-uncertainty drive (Yu & Dayan, Neuron 2005):
//!   acetylcholine reports expected uncertainty — the brain's estimate
//!   of how unreliable its own predictions are — so the wake component
//!   is scaled up when confidence is low and down when it is high.
//!
//! The efferent edge is also here: [`sensory_precision_gain`] is the
//! cholinergic gain on bottom-up prediction-error precision — ACh
//! optimizes the expected precision of sensory channels by gain on
//! supragranular pyramidal cells (Feldman & Friston, Front. Hum.
//! Neurosci. 2010; Moran et al., J. Neurosci. 2013), implemented in
//! `daemon::active_inference` as a divisor on observation variance.
//!
//! ════════════════════════════════════════════════════════════════════
//! WHAT THIS MODULE IS NOT
//! ════════════════════════════════════════════════════════════════════
//!
//! Process S itself (adenosine accumulation and glymphatic clearance),
//! the AChE excess decay, the coupling matrix rows, and the network
//! release of the *other* chemicals live in `state::neurochemical`;
//! the populations that drive `lc`/`rem` live in `state::sleep_network`.
//! This module owns only the basal forebrain's own math — what reads
//! those signals and what they do to its output.

use crate::state::sanitize::finite_clamp;

// ─── Release-map coefficients ────────────────────────────────────────
// Release levels as linear functions of the network populations. The
// coefficients are fit to the network's measured stage means (wake,
// NREM, REM) and to the release data they are meant to reproduce; the
// fit and its tolerances are in tests/network_calibration.rs.
//
// Acetylcholine rises with LC (wake) and REM-on: it is high in waking
// and REM and lowest in NREM (Vazquez & Baghdoyan 2001).
/// Basal (state-independent) term of the release map.
pub const ACH_BASE: f32 = 0.2895;
/// Weight of the LC (wake) population's drive on cholinergic release.
pub const ACH_FROM_LC: f32 = 0.3017;
/// Weight of the REM-on population's drive on cholinergic release.
pub const ACH_FROM_REM: f32 = 0.4512;

// ─── A1 adenosine inhibition ─────────────────────────────────────────
//
// A1 is the high-affinity, Gi-coupled adenosine receptor on the basal
// forebrain's wake-active neurons — the site where Process S acts on
// this region's output. Occupancy follows the same n=2 Hill form the
// coupling matrix uses for receptor gating.

/// Effective adenosine level for half-maximal A1 occupancy.
///
/// A1 is high-affinity (submicromolar); on this model's normalized
/// scale, half occupancy sits well below the Drowsy gate so meaningful
/// suppression begins as pressure builds past the resting baseline.
pub const A1_EC50: f32 = 0.35;
/// Maximum fraction of the LC-driven release component that A1
/// occupancy can suppress. Capped at one half: A1 activation inhibits
/// firing of the wake-active cholinergic neurons but does not silence
/// the population outright (Strecker et al., 2000).
pub const A1_MAX_INHIBITION: f32 = 0.5;
/// The resting extracellular adenosine level — the chemical's baseline
/// and Process S's lower asymptote. Suppression is measured relative
/// to occupancy at this level so rest is exactly neutral.
const ADENOSINE_REST: f32 = 0.20;
/// A1 occupancy at the resting level — the zero point of
/// [`a1_suppression`]. Const-evaluable from the Hill form.
const A1_OCCUPANCY_AT_REST: f32 =
    ADENOSINE_REST * ADENOSINE_REST / (A1_EC50 * A1_EC50 + ADENOSINE_REST * ADENOSINE_REST);

/// A1-receptor occupancy: Hill equation, n = 2.
///
///   occ = adn² / (EC50² + adn²)
pub fn a1_occupancy(adenosine: f32) -> f32 {
    let adn = adenosine.max(0.0);
    let adn_sq = adn * adn;
    adn_sq / (A1_EC50 * A1_EC50 + adn_sq)
}

/// Suppression of the wake-active cholinergic component, [0, A1_MAX].
///
/// Occupancy is re-referenced to its value at the resting level and
/// normalized to [0, 1] over the working range, so the suppression is
/// exactly zero at rest and approaches `A1_MAX_INHIBITION` as
/// occupancy saturates toward one.
pub fn a1_suppression(adenosine: f32) -> f32 {
    let rel = (a1_occupancy(adenosine) - A1_OCCUPANCY_AT_REST) / (1.0 - A1_OCCUPANCY_AT_REST);
    A1_MAX_INHIBITION * finite_clamp(rel, 0.0, 1.0)
}

// ─── Expected uncertainty (Yu & Dayan 2005) ──────────────────────────
//
// Acetylcholine reports expected uncertainty — in the Bayesian task
// model, α_t = 1 − max_z P[z_t = z | D_t], the complement of the
// posterior mass on the most likely context. The body's live
// confidence estimate is the inference engine's tracked precision
// (itself an inverse-error-variance target), so the reduced form is
// α = 1 − precision: a well-predicted body reports little expected
// uncertainty; sustained surprise reports a lot. When a regime
// posterior over the chemical state is ever run live
// (`state::phase_partition`), α is the same object one level up:
// 1 − max belief.

/// Expected uncertainty at the well-predicted operating point — the
/// reference where the uncertainty drive is exactly neutral. Resting
/// precision sits at ~0.9 (the precision tracker's resting target),
/// so resting α ≈ 0.1.
pub const UNCERTAINTY_REFERENCE: f32 = 0.1;
/// Gain of the expected-uncertainty drive on the wake component.
/// 0.15 moves the LC-driven term by at most ~±13% across the whole α
/// range — attention modulates the state profile, it does not set it.
pub const UNCERTAINTY_GAIN: f32 = 0.15;

/// Map tracked inference precision onto the expected-uncertainty
/// signal the cholinergic equation consumes: α = 1 − precision,
/// clamped to [0, 1].
pub fn expected_uncertainty(precision: f32) -> f32 {
    finite_clamp(1.0 - precision, 0.0, 1.0)
}

/// The multiplicative drive expected uncertainty applies to the
/// wake-active cholinergic component: 1 + gain·(α − α_ref).
///
/// High expected uncertainty raises cortical ACh — bottom-up evidence
/// should dominate when priors are unreliable (Yu & Dayan 2005);
/// low expected uncertainty relaxes it — a well-predicted world can
/// lean on its priors.
pub fn uncertainty_drive(alpha: f32) -> f32 {
    1.0 + UNCERTAINTY_GAIN * (alpha - UNCERTAINTY_REFERENCE)
}

// ─── The release equation ────────────────────────────────────────────

/// The wake-active component of cholinergic release: LC drive
/// attenuated by A1 occupancy and scaled by the uncertainty drive.
///
/// Adenosine and α both act only here: A1 receptors sit on the
/// wake-active population (not the REM-on one), and expected
/// uncertainty is a waking-attention signal — so neither modulator
/// can perturb the calibrated NREM floor or the REM rebound.
pub fn wake_component(lc: f32, adenosine: f32, alpha: f32) -> f32 {
    ACH_FROM_LC * lc * (1.0 - a1_suppression(adenosine)) * uncertainty_drive(alpha)
}

/// The basal forebrain's cholinergic release target under the
/// sleep-wake network's population state.
///
///   target = ACH_BASE
///          + ACH_FROM_LC · lc · (1 − A1(adn)) · D(α)
///          + ACH_FROM_REM · rem
///
/// At resting adenosine and reference uncertainty this reduces to the
/// calibrated linear map: NREM 0.34, waking 0.58, REM 0.77.
pub fn release_target(lc: f32, rem: f32, adenosine: f32, alpha: f32) -> f32 {
    finite_clamp(
        ACH_BASE + wake_component(lc, adenosine, alpha) + ACH_FROM_REM * rem,
        0.0,
        1.0,
    )
}

/// The cholinergic set-point when no sleep-wake network is running —
/// the fallback path where the level homeostats instead of tracking a
/// release target. The same two modulators act on the whole set-point
/// (with no network there is no wake/REM decomposition).
pub fn wake_setpoint(baseline: f32, adenosine: f32, alpha: f32) -> f32 {
    finite_clamp(
        baseline * (1.0 - a1_suppression(adenosine)) * uncertainty_drive(alpha),
        0.05,
        0.95,
    )
}

// ─── Cholinergic sensory-precision gain ──────────────────────────────
//
// The efferent half of the Yu & Dayan claim: acetylcholine optimizes
// the precision of bottom-up prediction errors — implemented in
// `daemon::active_inference` as a divisor on the Kalman observation
// variance, the gain of the units reporting prediction error
// (Moran et al., 2013: cholinesterase inhibition altered sensory PE
// responses through supragranular pyramidal gain).

/// Gain of ACh deviation-from-baseline on sensory precision.
pub const ACH_PRECISION_GAIN: f32 = 0.5;
/// Bounds on the precision multiplier. Attention shifts the weight of
/// evidence; it does not open or close the channel.
pub const SENSORY_PRECISION_GAIN_MIN: f32 = 0.7;
pub const SENSORY_PRECISION_GAIN_MAX: f32 = 1.4;

/// The multiplier on sensory (observation) precision for a given
/// effective acetylcholine level:
///
///   Π_ach = clamp(1 + k·(ach_eff − ach_baseline), min, max)
///
/// Anchored to the chemical's genetic baseline so tonic cholinergic
/// tone is exactly neutral — elevated ACh raises the weight the
/// generative model gives its observations; depressed ACh lowers it.
pub fn sensory_precision_gain(ach_eff: f32) -> f32 {
    let ach_baseline =
        crate::state::neurochemical::NeurochemicalId::Acetylcholine.default_baseline();
    finite_clamp(
        1.0 + ACH_PRECISION_GAIN * (ach_eff - ach_baseline),
        SENSORY_PRECISION_GAIN_MIN,
        SENSORY_PRECISION_GAIN_MAX,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a1_occupancy_is_saturating_hill() {
        // Zero at zero, EC50 at half, saturating toward one.
        assert_eq!(a1_occupancy(0.0), 0.0);
        assert!((a1_occupancy(A1_EC50) - 0.5).abs() < 1e-6);
        assert!(a1_occupancy(0.95) < 1.0 && a1_occupancy(0.95) > 0.85);
    }

    #[test]
    fn a1_suppression_is_zero_at_rest_and_capped() {
        assert_eq!(a1_suppression(ADENOSINE_REST), 0.0);
        assert!(a1_suppression(0.0) == 0.0);
        let deep = a1_suppression(0.95);
        assert!(deep > 0.4 && deep <= A1_MAX_INHIBITION);
        // The somnogenic signal bites through the Drowsy gate.
        assert!(a1_suppression(0.5) > 0.2);
    }

    #[test]
    fn release_target_recovers_the_calibrated_profile_at_rest() {
        // At resting adenosine and reference uncertainty the equation
        // reduces to the fit linear map (tests/network_calibration.rs
        // stage means).
        let alpha = UNCERTAINTY_REFERENCE;
        let nrem = release_target(0.151, 0.011, ADENOSINE_REST, alpha);
        let wake = release_target(0.963, 0.0, ADENOSINE_REST, alpha);
        let rem = release_target(0.133, 0.976, ADENOSINE_REST, alpha);
        assert!((nrem - 0.34).abs() < 0.02, "NREM {nrem}");
        assert!((wake - 0.58).abs() < 0.02, "wake {wake}");
        assert!((rem - 0.77).abs() < 0.02, "REM {rem}");
    }

    #[test]
    fn release_target_sags_only_on_the_wake_side() {
        // Sleep pressure attenuates the LC-driven component; NREM and
        // REM profiles (no LC drive) are untouched.
        let alpha = UNCERTAINTY_REFERENCE;
        let rested_wake = release_target(0.963, 0.0, ADENOSINE_REST, alpha);
        let pressed_wake = release_target(0.963, 0.0, 0.9, alpha);
        assert!(
            pressed_wake < rested_wake - 0.1,
            "wake under pressure {pressed_wake} vs rested {rested_wake}"
        );
        let nrem_rested = release_target(0.151, 0.011, ADENOSINE_REST, alpha);
        let nrem_pressed = release_target(0.151, 0.011, 0.9, alpha);
        assert!(
            (nrem_rested - nrem_pressed).abs() < 0.02,
            "NREM must not move: {nrem_rested} vs {nrem_pressed}"
        );
        let rem_pressed = release_target(0.133, 0.976, 0.9, alpha);
        assert!(
            rem_pressed > pressed_wake,
            "REM rebound outlives pressure: {rem_pressed}"
        );
    }

    #[test]
    fn uncertainty_drive_is_neutral_at_reference() {
        assert!((uncertainty_drive(UNCERTAINTY_REFERENCE) - 1.0).abs() < 1e-6);
        // Confidence raises it one way, surprise the other.
        assert!(uncertainty_drive(0.0) < 1.0);
        assert!(uncertainty_drive(0.9) > 1.0);
    }

    #[test]
    fn expected_uncertainty_complements_precision() {
        assert!((expected_uncertainty(0.9) - 0.1).abs() < 1e-6);
        assert_eq!(expected_uncertainty(1.5), 0.0);
        assert_eq!(expected_uncertainty(-0.5), 1.0);
    }

    #[test]
    fn precision_gain_anchors_at_tonic_level() {
        let ach_baseline =
            crate::state::neurochemical::NeurochemicalId::Acetylcholine.default_baseline();
        assert!((sensory_precision_gain(ach_baseline) - 1.0).abs() < 1e-6);
        assert!(sensory_precision_gain(0.9) > 1.0);
        assert!(sensory_precision_gain(0.0) < 1.0);
        assert!(sensory_precision_gain(0.0) >= SENSORY_PRECISION_GAIN_MIN);
    }
}

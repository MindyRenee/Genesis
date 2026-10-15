//! Mechanism tests for the five chemicals that previously had only the
//! generic homeostatic spring — epinephrine, oxytocin, β-endorphin,
//! vasopressin, and glutamate.
//!
//! ## What is being verified
//!
//! The four circulating chemicals are modeled as first-order
//! secretion/clearance pools,
//!
//!     dL/dt = k·(L_basal − L) + drives + coupling modulators
//!
//! with k set from each hormone's measured plasma half-life:
//!
//! - epinephrine:  t½ ≈ 2.6 min (Resuscitation 2023 PK study; FDA <5 min)
//! - oxytocin:     t½ ≈ 5–15 min circulating (Endotext)
//! - vasopressin:  t½ ≈ 10–35 min (vasopressinases; Endotext)
//! - β-endorphin:  t½ ≈ 13–46 min dominant pool (triple-exponential
//!   disappearance after IV bolus)
//!
//! plus their specific release drives:
//!
//! - β-endorphin is co-released equimolarly with ACTH from POMC
//!   (Guillemin et al., Science 1977), so it must rise under CRH drive
//!   at maturation 1.0 and stay flat during the SHRP (maturation 0).
//! - vasopressin is co-secreted with CRH by parvocellular PVN neurons
//!   and potentiates CRH-evoked ACTH release via corticotroph V1b
//!   receptors (Tanoue et al., JCI 2002).
//! - oxytocin attenuates HPA reactivity — the endocrine arm of social
//!   buffering (Heinrichs et al., Biol Psychiatry 2003).
//!
//! Glutamate keeps the fast neural spring (correct timescale for a
//! synaptic transmitter) and gains two substrate mechanisms: EAAT
//! uptake of ambient excess (>90% of released glutamate, Danbolt
//! 2001) and the GAD drain — GABA is synthesized stoichiometrically
//! from glutamate, so an elevated GABA pool draws glutamate down.

use genesis::state::neurochemical::{
    ENDORPHIN_METABOLIC_RATE, EPINEPHRINE_METABOLIC_RATE, NeuroTickParams, NeurochemicalId,
    NeurochemicalVector, OXYTOCIN_METABOLIC_RATE, VASOPRESSIN_METABOLIC_RATE,
};

fn level(nv: &NeurochemicalVector, id: NeurochemicalId) -> f32 {
    nv.get(id).map(|c| c.level).unwrap_or(f32::NAN)
}

fn baseline(nv: &NeurochemicalVector, id: NeurochemicalId) -> f32 {
    nv.get(id).map(|c| c.baseline).unwrap_or(f32::NAN)
}

/// Ticks until `level` decays to within `target_excess` of baseline,
/// or `max_ticks` if it never does. Coupling is disabled so the
/// measurement isolates the pool's own clearance kinetics.
fn ticks_to_excess(id: NeurochemicalId, target_excess: f32, max_ticks: u32) -> u32 {
    let params = NeuroTickParams {
        coupling_scale: 0.0,
        metaplasticity_rate: 0.0,
        maturation_level: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv = NeurochemicalVector::new(0);
    nv.apply_impulse_capped(id, 0.3, 0);
    let base = baseline(&nv, id);
    for t in 1..=max_ticks {
        nv.tick_with_params(&params);
        if level(&nv, id) - base <= target_excess {
            return t;
        }
    }
    max_ticks
}

// ─── Measured plasma half-lives ─────────────────────────────

/// Epinephrine impulse decays with its measured plasma half-life
/// (~2.6 min, k = 4.6e-3/s → ln2/k ≈ 151 s ≈ 1510 ticks at dt=0.1).
#[test]
fn epinephrine_half_life_is_measured_plasma_value() {
    let half_life_ticks = ticks_to_excess(NeurochemicalId::Epinephrine, 0.15, 20_000);
    let expected = (0.5f32.ln() / -EPINEPHRINE_METABOLIC_RATE / 0.1) as u32;
    assert!(
        half_life_ticks > expected / 2 && half_life_ticks < expected * 2,
        "epinephrine half-life {half_life_ticks} ticks, expected ~{expected}"
    );
}

/// Oxytocin persists on the ~10-min peptide timescale — an order of
/// magnitude slower than epinephrine and far slower than the neural
/// spring that used to erase the bolus in seconds.
#[test]
fn oxytocin_half_life_is_measured_plasma_value() {
    let half_life_ticks = ticks_to_excess(NeurochemicalId::Oxytocin, 0.15, 30_000);
    let expected = (0.5f32.ln() / -OXYTOCIN_METABOLIC_RATE / 0.1) as u32;
    assert!(
        half_life_ticks > expected / 2 && half_life_ticks < expected * 2,
        "oxytocin half-life {half_life_ticks} ticks, expected ~{expected}"
    );
}

#[test]
fn vasopressin_half_life_is_measured_plasma_value() {
    let half_life_ticks = ticks_to_excess(NeurochemicalId::Vasopressin, 0.15, 40_000);
    let expected = (0.5f32.ln() / -VASOPRESSIN_METABOLIC_RATE / 0.1) as u32;
    assert!(
        half_life_ticks > expected / 2 && half_life_ticks < expected * 2,
        "vasopressin half-life {half_life_ticks} ticks, expected ~{expected}"
    );
}

#[test]
fn endorphin_half_life_is_measured_plasma_value() {
    let half_life_ticks = ticks_to_excess(NeurochemicalId::Endorphin, 0.15, 50_000);
    let expected = (0.5f32.ln() / -ENDORPHIN_METABOLIC_RATE / 0.1) as u32;
    assert!(
        half_life_ticks > expected / 2 && half_life_ticks < expected * 2,
        "endorphin half-life {half_life_ticks} ticks, expected ~{expected}"
    );
}

/// The measured ordering: a catecholamine clears faster than the
/// neuropeptides, which clear slower than the synaptic spring.
#[test]
fn hormone_timescales_are_ordered() {
    let epi = ticks_to_excess(NeurochemicalId::Epinephrine, 0.15, 50_000);
    let oxy = ticks_to_excess(NeurochemicalId::Oxytocin, 0.15, 50_000);
    let vp = ticks_to_excess(NeurochemicalId::Vasopressin, 0.15, 50_000);
    let end = ticks_to_excess(NeurochemicalId::Endorphin, 0.15, 50_000);
    assert!(
        epi < oxy,
        "epinephrine ({epi}) should clear before oxytocin ({oxy})"
    );
    assert!(
        oxy < vp,
        "oxytocin ({oxy}) should clear before vasopressin ({vp})"
    );
    assert!(
        vp < end,
        "vasopressin ({vp}) should clear before endorphin ({end})"
    );
}

// ─── Release drives ─────────────────────────────────────────

/// β-endorphin is co-released equimolarly with ACTH from POMC: a CRH
/// stress pulse at maturation 1.0 must raise the endorphin pool, and
/// the same pulse during the SHRP (maturation 0) must not — the
/// pituitary gate propagates for free.
#[test]
fn endorphin_co_released_with_acth() {
    let mature = NeuroTickParams {
        maturation_level: 1.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv = NeurochemicalVector::new(0);
    // Sustained stress: hold CRH elevated so the co-release
    // accumulates — a single bolus decays in ~2 min and leaves
    // little integrated POMC output.
    for _ in 0..6000 {
        if let Some(c) = nv.get_mut(NeurochemicalId::CRH) {
            c.level = 0.6;
        }
        nv.tick_with_params(&mature);
    }
    let base = baseline(&nv, NeurochemicalId::Endorphin);
    let end = level(&nv, NeurochemicalId::Endorphin);
    assert!(
        end > base + 0.04,
        "endorphin should rise under ACTH drive: {end} vs baseline {base}"
    );

    let immature = NeuroTickParams {
        maturation_level: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv2 = NeurochemicalVector::new(0);
    for _ in 0..6000 {
        if let Some(c) = nv2.get_mut(NeurochemicalId::CRH) {
            c.level = 0.6;
        }
        nv2.tick_with_params(&immature);
    }
    let end2 = level(&nv2, NeurochemicalId::Endorphin);
    let base2 = baseline(&nv2, NeurochemicalId::Endorphin);
    assert!(
        end2 < base2 + 0.02,
        "SHRP must gate POMC co-release: endorphin {end2} vs baseline {base2}"
    );
}

/// Vasopressin is co-secreted with CRH: a CRH pulse must raise the
/// vasopressin pool even though no impulse touched it directly.
#[test]
fn vasopressin_co_secreted_with_crh() {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv = NeurochemicalVector::new(0);
    for _ in 0..6000 {
        if let Some(c) = nv.get_mut(NeurochemicalId::CRH) {
            c.level = 0.6;
        }
        nv.tick_with_params(&params);
    }
    let base = baseline(&nv, NeurochemicalId::Vasopressin);
    let vp = level(&nv, NeurochemicalId::Vasopressin);
    assert!(
        vp > base + 0.04,
        "vasopressin should rise under CRH co-secretion: {vp} vs baseline {base}"
    );
}

/// Same CRH drive with vasopressin pinned high must produce more ACTH
/// than with vasopressin at baseline — the V1b synergy.
#[test]
fn vasopressin_potentiates_acth_via_v1b() {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        ..NeuroTickParams::DEFAULT
    };
    let acth_with = |vp_level: f32| -> f32 {
        let mut nv = NeurochemicalVector::new(0);
        nv.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
        for _ in 0..600 {
            // Exogenous AVP infusion: hold the pool at the test level
            // so the measurement isolates the pituitary gain rather
            // than the pool's own dynamics.
            if let Some(c) = nv.get_mut(NeurochemicalId::Vasopressin) {
                c.level = vp_level;
            }
            nv.tick_with_params(&params);
        }
        nv.acth_level
    };
    let acth_low = acth_with(0.20); // baseline
    let acth_high = acth_with(0.80); // elevated
    assert!(
        acth_high > acth_low * 1.2,
        "V1b synergy: ACTH {acth_high} (AVP high) should exceed {acth_low} (AVP basal) by >20%"
    );
}

/// Same CRH drive with oxytocin pinned high must produce less ACTH
/// than with oxytocin at baseline — social buffering of the HPA axis.
#[test]
fn oxytocin_attenuates_hpa_via_social_buffering() {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        ..NeuroTickParams::DEFAULT
    };
    let acth_with = |oxy_level: f32| -> f32 {
        let mut nv = NeurochemicalVector::new(0);
        nv.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
        for _ in 0..600 {
            if let Some(c) = nv.get_mut(NeurochemicalId::Oxytocin) {
                c.level = oxy_level;
            }
            nv.tick_with_params(&params);
        }
        nv.acth_level
    };
    let acth_low = acth_with(0.25); // baseline
    let acth_high = acth_with(0.80); // elevated
    assert!(
        acth_high < acth_low * 0.9,
        "social buffering: ACTH {acth_high} (OT high) should be below {acth_low} (OT basal)"
    );
}

// ─── Glutamate substrate mechanisms ─────────────────────────

/// EAAT uptake clears an ambient glutamate elevation faster than the
/// neural spring alone clears a dopamine elevation of the same size —
/// glutamate's excess-decay channel is the >90% uptake pathway.
#[test]
fn glutamate_excess_clears_faster_than_plain_spring() {
    let params = NeuroTickParams {
        coupling_scale: 0.0,
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv = NeurochemicalVector::new(0);
    nv.apply_impulse_capped(NeurochemicalId::Glutamate, 0.2, 0);
    nv.apply_impulse_capped(NeurochemicalId::Dopamine, 0.2, 0);
    for _ in 0..200 {
        nv.tick_with_params(&params);
    }
    let glu_excess =
        level(&nv, NeurochemicalId::Glutamate) - baseline(&nv, NeurochemicalId::Glutamate);
    let da_excess =
        level(&nv, NeurochemicalId::Dopamine) - baseline(&nv, NeurochemicalId::Dopamine);
    assert!(
        glu_excess < da_excess,
        "EAAT should clear glutamate faster than the plain spring clears dopamine: glu {glu_excess} vs da {da_excess}"
    );
}

/// GAD stoichiometry: a GABA pool held above baseline consumes
/// glutamate as the precursor — ambient glutamate must sit lower than
/// when GABA is at baseline, coupling disabled so only the drain acts.
#[test]
fn gaba_elevation_drains_glutamate_via_gad() {
    let params = NeuroTickParams {
        coupling_scale: 0.0,
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let run = |gaba_level: f32| -> f32 {
        let mut nv = NeurochemicalVector::new(0);
        for _ in 0..3000 {
            if let Some(c) = nv.get_mut(NeurochemicalId::GABA) {
                c.level = gaba_level;
            }
            nv.tick_with_params(&params);
        }
        level(&nv, NeurochemicalId::Glutamate)
    };
    let glu_high_gaba = run(0.90);
    let glu_rest_gaba = run(0.50);
    assert!(
        glu_high_gaba < glu_rest_gaba - 0.01,
        "elevated GABA should drain glutamate: {glu_high_gaba} vs {glu_rest_gaba}"
    );
}

// ─── Rest and bounds ────────────────────────────────────────

/// With no drives the pools rest at their baselines — the secretion/
/// clearance equilibrium, not a clamped artifact.
#[test]
fn hormone_pools_rest_at_baseline() {
    let mut nv = NeurochemicalVector::new(0);
    for _ in 0..3000 {
        nv.tick_with_params(&NeuroTickParams::DEFAULT);
    }
    for id in [
        NeurochemicalId::Epinephrine,
        NeurochemicalId::Oxytocin,
        NeurochemicalId::Endorphin,
        NeurochemicalId::Vasopressin,
    ] {
        let lvl = level(&nv, id);
        let base = baseline(&nv, id);
        assert!(
            (lvl - base).abs() < 0.05,
            "{id:?} should rest near baseline: {lvl} vs {base}"
        );
    }
}

/// Sustained maximal stress must keep every pool finite and inside
/// [0, 1] — the measured clearance channels bound the drives.
#[test]
fn hormone_pools_bounded_under_sustained_stress() {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut nv = NeurochemicalVector::new(0);
    for t in 0..20_000u64 {
        // Repeated CRH + sympathetic drive.
        if t % 100 == 0 {
            nv.apply_impulse_capped(NeurochemicalId::CRH, 0.1, t * 100);
            nv.apply_impulse_capped(NeurochemicalId::Epinephrine, 0.1, t * 100);
        }
        nv.tick_with_params(&params);
    }
    for id in NeurochemicalId::all() {
        let c = nv.get(id).unwrap();
        assert!(c.level.is_finite(), "{id:?} level not finite: {}", c.level);
        assert!(
            (0.0..=1.0).contains(&c.level),
            "{id:?} level {} out of bounds",
            c.level
        );
    }
}

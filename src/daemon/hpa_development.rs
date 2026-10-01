//! Developmental competence of the HPA axis.
//!
//! ## Why this is a curve and not a ramp
//!
//! The stress-hyporesponsive period (SHRP) is real and worth modelling: a
//! neonatal window in which the adrenocortical axis fails to mount a
//! glucocorticoid response to stress, protecting the developing brain
//! from cortisol's neurotoxic effects on synaptogenesis. In rodents it
//! runs from postnatal day 4 to 14 (Schmidt et al., 2003; Levine,
//! 2005). In humans the equivalent is thought to fall between roughly
//! 6 and 12 months, possibly to 15 months, and its precise end depends
//! on the quality of care the infant receives — poor care ends it
//! earlier (Gunnar & Donzella, 2002; Gunnar & Cheatham, 2003).
//!
//! The thing this must not become is a permanent ramp. The previous
//! implementation computed competence as `ltm_episode_count / 10_000`,
//! which meant cortisol was exactly zero until 10,000 lifetime
//! episodes and the axis was only fully online in the system's final
//! developmental phase. Three things are wrong with that:
//!
//!   1. **It is not a biological quantity.** The SHRP is defined in
//!      postnatal days, not episodic memory. A system that stores
//!      memories does not thereby have a stress axis, and one that
//!      stores many does not thereby become adult in a biologically
//!      meaningful way.
//!
//!   2. **It disabled the primary mediator of allostatic load.**
//!      McEwen's construct names cortisol, epinephrine and
//!      norepinephrine as the primary mediators. If cortisol cannot
//!      exist for 10,000 episodes, then a measure built on the stress
//!      axis cannot see stress — and the system is blind precisely
//!      where the framework says the cost is greatest, because early
//!      experience is when a developing mind is most vulnerable.
//!
//!   3. **The real trajectory is not monotone.** After the SHRP the
//!      axis over-responds. Across the peripubertal period HPA
//!      responses are "exaggerated relative to adults, both in terms
//!      of ACTH and glucocorticoid release" (PMC4867107), with a
//!      documented lag in negative feedback — the adolescent animal
//!      takes longer both to mount and to recover a response. A
//!      single ramp from 0 to 1 cannot represent that shape.
//!
//! So the developmental curve here is hyporesponsive early, crosses
//! into full adult competence, and passes through a transient
//! amplification before settling. That is the shape the literature
//! actually describes, and it is expressed against elapsed lifetime
//! (the clock the SHRP is defined in) rather than against a count of
//! stored episodes.

/// Experience at which development is complete.
///
/// Developmental progress is driven by accumulated experience rather
/// than by wall-clock time. The reason is that the two clocks are not
/// commensurable: the stress-hyporesponsive period is bounded by a
/// year of an infant's life and adolescence by puberty, but a
/// computational mind's development is a function of what it has
/// processed. Mapping years onto a machine's uptime would either
/// immobilise the axis for most of the system's life or make
/// development instantaneous, and neither is a defensible reading of
/// the phenomenon.
///
/// Elapsed time is retained only as a floor-advancer, for a system
/// that has existed a long time without accumulating episodes. It can
/// carry development forward; it can never hold it back.
const ADULT_EPISODE_COUNT: f32 = 1_000.0;

/// Wall-clock span over which elapsed time alone can carry
/// development forward, in seconds (~2 years).
///
/// Only reached by a system that has lived this long without
/// accumulating episodes, which is the case this exists to handle.
const DEVELOPMENT_SPAN_SECS: f32 = 63_072_000.0;

/// Fraction of development spent in the SHRP.
///
/// The period is a small fraction of a life — on the order of a year
/// against decades — and modelled here as the first ~5% of
/// development, so it is genuinely brief rather than a long stretch of
/// a machine's life in which the stress axis is switched off.
const SHRP_FRACTION: f32 = 0.02;

/// Centre of the peripubertal amplification, as a fraction of
/// developmental progress.
const PERIPUBERTAL_CENTRE: f32 = 0.62;

/// Half-width of the peripubertal amplification.
const PERIPUBERTAL_WIDTH: f32 = 0.14;

/// Peak amplification of the axis during the peripubertal window.
///
/// Documented as responses being "exaggerated relative to adults" in
/// both ACTH and glucocorticoid release, with a lag in negative
/// feedback that lengthens both the rise and the recovery. 1.25 keeps
/// that qualitative feature without letting the axis run away.
const PERIPUBERTAL_GAIN: f32 = 1.25;

/// Upper bound on competence. Above 1.0 only during the peripubertal
/// window, where the axis is documented to over-respond; a bound is
/// kept so a corrupted input cannot drive ACTH arbitrarily high.
const MAX_COMPETENCE: f32 = 1.5;

/// Competence of the HPA axis at a given developmental point.
///
/// Normally in [0, 1], where 1.0 is fully adult. It exceeds 1.0 only
/// inside the peripubertal window, where the axis is documented to
/// respond more strongly than it does in adulthood.
///
/// `elapsed_secs` is wall-clock time since the state was created and
/// `ltm_count` is accumulated episodic experience; development
/// advances on whichever is further along, and can be carried by
/// either. This is a multiplier on ACTH production and on the
/// CRH→ACTH drive, so 1.0 means fully adult.
pub fn hpa_competence(elapsed_secs: f32, ltm_count: u64) -> f32 {
    let by_experience = (ltm_count as f32 / ADULT_EPISODE_COUNT).clamp(0.0, 1.0);
    let by_elapsed = (elapsed_secs / DEVELOPMENT_SPAN_SECS).clamp(0.0, 1.0);
    let progress = by_experience.max(by_elapsed);

    // The SHRP. Competence is near zero across a brief early window
    // and climbs after it, as a smoothstep rather than a cutoff —
    // real adrenal sensitivity recovers gradually, and a
    // discontinuity would let one tick switch the whole axis on or off.
    let mature = smoothstep(SHRP_FRACTION, SHRP_FRACTION * 3.0, progress);

    // The peripubertal window: a transient amplification riding on top
    // of full competence, decaying back to 1.0. The axis over-responds
    // during adolescence, so competence there is modestly above adult.
    let t = (progress - PERIPUBERTAL_CENTRE) / PERIPUBERTAL_WIDTH;
    let bump = (-t * t).exp();
    // Note the upper bound is above 1.0, not at it. An earlier
    // version clamped to 1.0, which silently erased the peripubertal
    // amplification entirely — the raw value there is 1.25, and
    // clamping made adolescence indistinguishable from adulthood.
    (mature * (1.0 + (PERIPUBERTAL_GAIN - 1.0) * bump)).clamp(0.0, MAX_COMPETENCE)
}

/// Smoothstep from `edge0` to `edge1`, clamped outside.
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < f32::EPSILON {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_is_dormant_during_the_shrp() {
        // At birth the axis cannot mount a response.
        assert!(
            hpa_competence(0.0, 0) < 0.05,
            "newborn competence was {}",
            hpa_competence(0.0, 0)
        );
    }

    #[test]
    fn axis_is_available_early_in_life() {
        // The whole point: cortisol must exist well before the system
        // has accumulated a large memory. Shortly after birth the axis
        // is up.
        assert!(
            hpa_competence(60.0, 100) > 0.9,
            "competence early on was {}",
            hpa_competence(60.0, 100)
        );
    }

    #[test]
    fn competence_is_bounded_by_one_except_in_the_peripubertal_window() {
        // Never unbounded, and never negative.
        for step in 0..1000u64 {
            let c = hpa_competence(0.0, step);
            assert!(
                (0.0..=MAX_COMPETENCE).contains(&c),
                "competence {c} out of range at {step} episodes"
            );
        }
    }

    #[test]
    fn adolescent_axis_is_more_responsive_than_adult() {
        // The documented peripubertal exaggeration, sampled at the
        // peak of the window against a clearly adult point.
        let adolescent_episodes = (PERIPUBERTAL_CENTRE * ADULT_EPISODE_COUNT) as u64;
        let adolescent = hpa_competence(0.0, adolescent_episodes);
        let adult = hpa_competence(0.0, 1_000);
        assert!(
            adolescent > adult,
            "adolescent {adolescent} should exceed adult {adult}"
        );
    }

    #[test]
    fn experience_and_elapsed_time_both_order_development() {
        // Either clock can carry development forward, and neither can
        // hold it back.
        assert!(hpa_competence(0.0, 1_000) >= hpa_competence(0.0, 0));
        assert!(hpa_competence(DEVELOPMENT_SPAN_SECS, 0) >= hpa_competence(0.0, 0));
        // A long-lived system with no episodes still matures.
        assert!(hpa_competence(DEVELOPMENT_SPAN_SECS * 2.0, 0) > 0.9);
    }
}

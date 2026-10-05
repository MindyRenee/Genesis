//! Regression tests for the relationship between primary chemistry and derived affect.

use genesis::state::neurochemical::{NeurochemicalId, NeurochemicalVector};

#[test]
fn fresh_neurochemical_state_has_positive_derived_valence() {
    // The configured resting chemical baselines include positive reward,
    // mood, bonding, and arousal contributors while cortisol is zero and
    // glutamate is below its excess threshold. The derived valence must
    // therefore not silently revert to the old zero-initialized value.
    //
    // Assert the model's qualitative invariant rather than a particular
    // numeric resting mood, so this test does not freeze the coefficients.
    let state = NeurochemicalVector::new(1);

    assert!(
        state.valence > 0.0,
        "fresh state should derive positive valence from its configured          resting chemistry, got {}",
        state.valence
    );

    // Make the reason for the invariant explicit: the primary contributors
    // are actually initialized rather than merely relying on the derived
    // field's previous contents.
    assert!(
        state.effective_levels[NeurochemicalId::Dopamine as usize] > 0.0
            && state.effective_levels[NeurochemicalId::Serotonin as usize] > 0.0,
        "fresh state must initialize positive valence contributors"
    );
    assert_eq!(
        state.effective_levels[NeurochemicalId::Cortisol as usize],
        0.0,
        "fresh state should have no resting cortisol contribution"
    );
}

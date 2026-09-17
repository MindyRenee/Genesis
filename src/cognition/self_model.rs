//! Self-model — a derived first-person representation of Genesis's state.
//!
//! This is **not** experience. It is a functional self-representation
//! built on top of the neurochemical state. It gives the system something
//! it can inspect, report, and use as input to its own processes — a
//! prerequisite for any functional self-model theory of experience.
//!
//! The model is deliberately sparse: it captures identity, emotional
//! posture, dominant neurochemicals, sleep pressure, and a stable zone
//! identity. A richer self-model would integrate episodic memory,
//! intention, and a world-model, none of which exist in Genesis yet.

use crate::state::core_state::GenesisCoreState;
use crate::state::neurochemical::NeurochemicalId;
use crate::state::zones::MentalPhase;

/// A structured self-representation derived from [`GenesisCoreState`].
#[derive(Clone, Debug, PartialEq)]
pub struct SelfModel {
    /// Timestamp (ms) at which this self-model was computed.
    pub timestamp_ms: u64,
    /// How long this state has existed (ms).
    pub lifetime_ms: u64,
    /// Current cognitive task zone label.
    pub zone: &'static str,
    /// How long the system has been in the current zone (ms).
    pub zone_duration_ms: u64,
    /// Current emergent mental phase label.
    pub phase: &'static str,
    /// Affective valence [-1.0, 1.0].
    pub valence: f32,
    /// Arousal [0.0, 1.0].
    pub arousal: f32,
    /// Global effective-tone average.
    pub global_tone: f32,
    /// Plasticity gate [0.0, 1.0].
    pub plasticity_gate: f32,
    /// Adenosine effective level as a proxy for sleep pressure.
    pub sleep_pressure: f32,
    /// Top 3 dominant neurochemicals by effective level.
    pub dominant_chemicals: Vec<(&'static str, f32)>,
    /// Estimated emotional cause from current neurochemistry.
    pub emotional_cause: &'static str,
    /// Fraction of lifetime spent in the current state/zone.
    pub self_continuity: f32,
}

impl SelfModel {
    /// Derive a self-model from the current core state.
    ///
    /// `now_ms` is the wall-clock time used to compute lifetime and
    /// continuity metrics.
    pub fn from_state(state: &GenesisCoreState, now_ms: u64) -> Self {
        let lifetime_ms = now_ms.saturating_sub(state.header.created_at);
        let zone = state.zones.zone().label();
        let phase = state.zones.phase().label();
        let phase_enum = state.zones.phase();
        let zone_duration_ms = state.zones.zone_duration_ms;

        let valence = state.neurochemicals.valence;
        let arousal = state.neurochemicals.arousal;
        let global_tone = state.neurochemicals.global_tone;
        let plasticity_gate = state.neurochemicals.plasticity_gate;
        let sleep_pressure = state.neurochemicals.effective(NeurochemicalId::Adenosine);

        let mut dominant_chemicals: Vec<(&'static str, f32)> = state
            .neurochemicals
            .chemicals
            .iter()
            .map(|chem| {
                let id = NeurochemicalId::from_u8(chem.id);
                (id.name(), chem.effective_level())
            })
            .collect();
        dominant_chemicals
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        dominant_chemicals.truncate(3);

        let emotional_cause = Self::cause_for_phase(phase_enum);

        let self_continuity = if lifetime_ms > 0 {
            (zone_duration_ms as f64 / lifetime_ms as f64).min(1.0) as f32
        } else {
            1.0
        };

        Self {
            timestamp_ms: now_ms,
            lifetime_ms,
            zone,
            zone_duration_ms,
            phase,
            valence,
            arousal,
            global_tone,
            plasticity_gate,
            sleep_pressure,
            dominant_chemicals,
            emotional_cause,
            self_continuity,
        }
    }

    fn cause_for_phase(phase: MentalPhase) -> &'static str {
        match phase {
            MentalPhase::Flow => "high dopamine and acetylcholine with low cortisol support focus",
            MentalPhase::Stress => "elevated cortisol and norepinephrine signal a threat",
            MentalPhase::Overwhelmed => "saturated arousal and cortisol exceed regulatory capacity",
            MentalPhase::Drowsy => "adenosine is suppressing wake-promoting systems",
            MentalPhase::NREM => "high GABA and adenosine sustain slow-wave sleep",
            MentalPhase::REM => "cholinergic activation with aminergic silence supports dreaming",
            MentalPhase::Alert => "high norepinephrine and histamine maintain vigilance",
            MentalPhase::Active => "neurochemistry is in a balanced, engaged state",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_model_from_state() {
        let state = GenesisCoreState::new(1, 1000);
        let model = SelfModel::from_state(&state, 2000);

        assert_eq!(model.timestamp_ms, 2000);
        assert_eq!(model.lifetime_ms, 1000);
        assert_eq!(model.zone, "idle");
        assert!(!model.dominant_chemicals.is_empty());
        assert!(model.self_continuity <= 1.0);
        assert!(model.self_continuity >= 0.0);
    }

    #[test]
    fn self_model_changes_with_state() {
        let mut state = GenesisCoreState::new(1, 1000);

        // Pump dopamine to make it dominant
        for _ in 0..100 {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Dopamine)
                .expect("Dopamine is a valid NeurochemicalId")
                .apply_impulse(0.2, 0);
        }

        let model = SelfModel::from_state(&state, 2000);
        assert!(
            model
                .dominant_chemicals
                .iter()
                .any(|(n, _)| *n == "dopamine"),
            "dopamine should be among the top-3 dominant chemicals, got {:?}",
            model.dominant_chemicals
        );
    }
}

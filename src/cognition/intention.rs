//! Intention and action selection — a minimal goal-driven layer.
//!
//! This is **not** a model of free will or genuine agency. It is a
//! rule-based, neurochemically-gated intention manager that selects the
//! next cognitive zone from a small set of active goals, using the
//! self-model as its input. It is the next architectural prerequisite
//! above self-modeling: not just knowing how you are, but deciding what
//! to do given goals and state.
//!
//! # Design
//!
//! - `Intention` = a goal with a target zone, priority, and deadline.
//! - `IntentionManager` = a small ordered set of active intentions.
//! - `select_zone` = scores each active intention by how well it fits the
//!   current neurochemical state, and returns the best `CognitiveZone`.
//!
//! The scoring is affectively modulated: a work goal is boosted in flow,
//! a rest goal is boosted by adenosine, and a social goal is boosted by
//! oxytocin. Stress suppresses learning and deep work; high cortisol
//! favors reflection or error recovery.

use crate::cognition::self_model::SelfModel;
use crate::state::zones::{CognitiveZone, MentalPhase};

/// Kinds of high-level goals the system can hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GoalKind {
    /// Reduce arousal and recover resources.
    Rest,
    /// Produce or analyze code/text.
    Work,
    /// Interact with the user or other agents.
    Socialize,
    /// Acquire new information or skills.
    Learn,
    /// Process errors, stress, or unresolved state.
    Recover,
}

impl GoalKind {
    /// Default target zone for this kind of goal.
    pub const fn target_zone(self) -> CognitiveZone {
        match self {
            GoalKind::Rest => CognitiveZone::Idle,
            GoalKind::Work => CognitiveZone::Coding,
            GoalKind::Socialize => CognitiveZone::Conversation,
            GoalKind::Learn => CognitiveZone::Learning,
            GoalKind::Recover => CognitiveZone::Reflection,
        }
    }

    /// Short label for the goal kind.
    pub const fn label(self) -> &'static str {
        match self {
            GoalKind::Rest => "rest",
            GoalKind::Work => "work",
            GoalKind::Socialize => "socialize",
            GoalKind::Learn => "learn",
            GoalKind::Recover => "recover",
        }
    }
}

/// A single active intention.
#[derive(Clone, Debug)]
pub struct Intention {
    /// What kind of goal this is.
    pub kind: GoalKind,
    /// The zone the intention would like the system to enter.
    pub target_zone: CognitiveZone,
    /// Base priority [0.0, 1.0]. Higher = more important.
    pub priority: f32,
    /// Wall-clock deadline by which the intention should be addressed (ms).
    pub deadline_ms: u64,
}

impl Intention {
    /// Create an intention from a goal kind.
    ///
    /// `deadline_ms` is the absolute time by which the goal should be
    /// addressed. `priority` is the base importance in [0, 1].
    pub fn new(kind: GoalKind, priority: f32, deadline_ms: u64) -> Self {
        Self {
            kind,
            target_zone: kind.target_zone(),
            priority: crate::state::sanitize::finite_clamp(priority, 0.0, 1.0),
            deadline_ms,
        }
    }

    /// The zone this intention targets.
    pub fn target(&self) -> CognitiveZone {
        self.target_zone
    }
}

/// A small goal-stack / intention manager.
#[derive(Clone, Debug)]
pub struct IntentionManager {
    /// Active intentions, kept sorted by descending effective score.
    pub active: Vec<Intention>,
    /// Maximum number of concurrent intentions.
    pub max_active: usize,
}

impl IntentionManager {
    /// Create an empty intention manager.
    pub fn new(max_active: usize) -> Self {
        Self {
            active: Vec::with_capacity(max_active),
            max_active: max_active.max(1),
        }
    }

    /// Add an active intention if there is room.
    ///
    /// Returns `true` if the intention was added, `false` if the stack
    /// was full and the new intention was dropped.
    pub fn push(&mut self, intention: Intention) -> bool {
        if self.active.len() >= self.max_active {
            return false;
        }
        self.active.push(intention);
        true
    }

    /// Remove all expired intentions whose deadlines have passed.
    pub fn expire(&mut self, now_ms: u64) {
        self.active.retain(|i| i.deadline_ms > now_ms);
    }

    /// Compute the best zone to enter given the self-model and time.
    ///
    /// Each intention is scored by:
    /// - `priority`
    /// - urgency (how close the deadline is)
    /// - state fit (how well the current neurochemistry supports it)
    ///
    /// The highest-scoring intention's target zone is returned. If no
    /// intentions are active, the current zone is preserved (returned
    /// as `None` means no change; this function returns the best
    /// `CognitiveZone` regardless, defaulting to `Idle` if empty).
    pub fn select_zone(&self, model: &SelfModel, now_ms: u64) -> CognitiveZone {
        if self.active.is_empty() {
            return CognitiveZone::Idle;
        }

        let mut best_zone = CognitiveZone::Idle;
        let mut best_score = f32::MIN;

        for intention in &self.active {
            let score = self.score_intention(intention, model, now_ms);
            if score > best_score {
                best_score = score;
                best_zone = intention.target();
            }
        }

        best_zone
    }

    /// Score an intention in the current affective context.
    fn score_intention(&self, intention: &Intention, model: &SelfModel, now_ms: u64) -> f32 {
        // Urgency: 0..1, rises as deadline approaches.
        let time_left = intention.deadline_ms.saturating_sub(now_ms) as f32;
        let urgency = (1.0 - (time_left / 60_000.0).min(1.0)).max(0.0);

        // State fit: how well the current neurochemistry supports the goal.
        let state_fit = state_fit_for_goal(intention.kind, model);

        // Phase compatibility: some zones are incompatible with some phases.
        let phase_penalty = phase_penalty(intention.target(), model.phase);

        intention.priority * 0.4 + urgency * 0.3 + state_fit * 0.4 - phase_penalty * 0.3
    }

    /// Return the top-scoring intention's label and score, for reporting.
    pub fn top_intention(&self, model: &SelfModel, now_ms: u64) -> Option<(&'static str, f32)> {
        self.active
            .iter()
            .map(|i| (i.kind.label(), self.score_intention(i, model, now_ms)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
    }
}

/// How well the current state supports a particular goal kind.
fn state_fit_for_goal(kind: GoalKind, model: &SelfModel) -> f32 {
    match kind {
        GoalKind::Rest => {
            // Rest is favoured by sleep pressure and low arousal/valence.
            (model.sleep_pressure + (1.0 - model.arousal)) * 0.5
        }
        GoalKind::Work => {
            // Work is favoured by flow, alertness, and high plasticity.
            if model.phase == MentalPhase::Flow.label() {
                1.0
            } else if model.phase == MentalPhase::Alert.label() {
                0.8
            } else if model.phase == MentalPhase::Stress.label() {
                // Stress narrows focus: still usable for some work, but impaired.
                0.4
            } else if model.phase == MentalPhase::Overwhelmed.label() {
                0.0
            } else {
                0.6
            }
        }
        GoalKind::Socialize => {
            // Socialize is favoured by positive valence and moderate arousal.
            // We use the serotonin proxy for social/oxytocin tone.
            crate::state::sanitize::finite_clamp(
                (model.valence + 1.0) * 0.5 + (1.0 - (model.arousal - 0.5).abs()),
                0.0,
                1.0,
            )
        }
        GoalKind::Learn => {
            // Learning is favoured by high plasticity and calm-positive valence.
            model.plasticity_gate * (1.0 - model.arousal * 0.5)
        }
        GoalKind::Recover => {
            // Recovery is favoured by stress, negative valence, or sleep debt.
            crate::state::sanitize::finite_clamp(
                -model.valence + (1.0 - model.plasticity_gate) + model.sleep_pressure,
                0.0,
                1.0,
            ) * 0.5
        }
    }
}

/// Penalize zones that are incompatible with the emergent mental phase.
fn phase_penalty(target: CognitiveZone, phase: &'static str) -> f32 {
    let is_sleep_phase = phase == MentalPhase::NREM.label()
        || phase == MentalPhase::REM.label()
        || phase == MentalPhase::Drowsy.label();

    if is_sleep_phase && target != CognitiveZone::Sleeping && target != CognitiveZone::Idle {
        // Trying to work while drowsy/sleeping is heavily penalized.
        1.0
    } else if (phase == MentalPhase::Overwhelmed.label() || phase == MentalPhase::Stress.label())
        && target == CognitiveZone::Coding
    {
        // Coding is penalized under high stress/overwhelm.
        0.5
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::core_state::GenesisCoreState;

    #[test]
    fn select_rest_when_drowsy() {
        let state = GenesisCoreState::new(1, 1000);
        let mut model = crate::cognition::self_model::SelfModel::from_state(&state, 2000);
        // Manually push the model into a drowsy posture
        model.phase = MentalPhase::Drowsy.label();
        model.arousal = 0.1;
        model.sleep_pressure = 0.7;

        let mut manager = IntentionManager::new(4);
        manager.push(Intention::new(GoalKind::Work, 0.9, 10_000));
        manager.push(Intention::new(GoalKind::Rest, 0.5, 10_000));

        assert_eq!(manager.select_zone(&model, 3000), CognitiveZone::Idle);
    }

    #[test]
    fn select_work_in_flow() {
        let state = GenesisCoreState::new(1, 1000);
        let mut model = crate::cognition::self_model::SelfModel::from_state(&state, 2000);
        model.phase = MentalPhase::Flow.label();
        model.arousal = 0.7;
        model.sleep_pressure = 0.1;

        let mut manager = IntentionManager::new(4);
        manager.push(Intention::new(GoalKind::Work, 0.6, 10_000));
        manager.push(Intention::new(GoalKind::Rest, 0.8, 10_000));

        // Even though Rest has higher priority, the state fit for Work
        // in flow should overcome it when deadline is far.
        let zone = manager.select_zone(&model, 3000);
        assert_eq!(zone, CognitiveZone::Coding);
    }

    #[test]
    fn push_and_expire() {
        let mut manager = IntentionManager::new(2);
        assert!(manager.push(Intention::new(GoalKind::Learn, 0.5, 5000)));
        assert!(manager.push(Intention::new(GoalKind::Rest, 0.5, 6000)));
        assert!(!manager.push(Intention::new(GoalKind::Work, 0.9, 7000)));

        manager.expire(5500);
        assert_eq!(manager.active.len(), 1);
    }
}

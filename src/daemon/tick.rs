//! Daemon tick controller — the subcognitive's stateful driver.
//!
//! This module owns [`TickLoop`], the controller that holds the
//! daemon's long-lived subsystems (intention manager, interoceptor,
//! active-inference engine, dyadic model) and the change-detection
//! state for body control.
//!
//! # Two entry points
//!
//! In production the daemon is **reactive**: it does not run a fixed
//! tick loop. The cognitive mind drives each function through IPC
//! commands when internal state (brain waves, thresholds, urges) says
//! it's time, and the daemon executes the corresponding method:
//!
//! 1. [`TickLoop::advance_neuro`] — advance the coupled dynamics by
//!    the elapsed `dt`, then run active inference and the dyadic model
//! 2. [`TickLoop::consolidate`] — promote STM → LTM under emotional gating
//! 3. [`TickLoop::associate`] — find connections between memories
//! 4. [`TickLoop::dream`] — free association when sleeping
//! 5. [`TickLoop::read_sensors`] — interoception
//! 6. [`TickLoop::apply_body_control`] — publish the shared body policy recommendation
//! 7. [`TickLoop::check_staleness`] — mark crashed cognitive modules
//!
//! [`TickLoop::tick`] is the integrated, single-shot reference path
//! that runs all of the above in sequence. It is retained for tests
//! and for reasoning about the whole system in one place; the
//! production daemon does not call it.
//!
//! # What's novel
//!
//! The controller isn't just "run tasks on a timer." The **emotional
//! state determines what work gets done**:
//!
//! - When **plasticity gate is high** (good neurochemistry): prioritize
//!   consolidation — learn aggressively
//! - When **plasticity gate is low** (chronic stress): skip consolidation,
//!   prioritize error evaluation and housekeeping — the system is in
//!   survival mode, not learning mode
//! - When **sleeping**: run consolidation and dreaming — this is the
//!   brain's window for memory replay, plasticity recovery, and insight
//! - When **in flow**: reduce background tasks to minimum — don't
//!   interrupt a good state with housekeeping
//! - When **overwhelmed**: shed all non-essential tasks — the system
//!   needs to recover before doing anything else

use std::time::{SystemTime, UNIX_EPOCH};

use crate::cognition::{IntentionManager, SelfModel};
use crate::field::{FieldIntegration, FieldRuntime};
use crate::state::neurochemical::{NeuroTickParams, NeurochemicalId};
use crate::state::{MentalPhase, ModuleId, ModuleStatus, subcognitive_flag};
use crate::store::ring_buffer::EventType;
use crate::store::{LtmStore, MmapState, RingBuffer};

/// Provenance marker — one letter of the creator lattice (see
/// PROVENANCE.md).
pub(crate) const MARKER_G05: u64 = 0x4753_0554_59A9_90E8;

use super::active_inference::{
    ActiveInferenceEngine, DyadicSignals, apply_inference_feedback, impulse_offset_to_rate,
};
use super::association::AssociationEngine;
use super::consolidation::{ConsolidationEngine, ConsolidationResult};
use super::dyadic_model::{DyadicAffectModel, take_user_affect};
use super::interoception::Interoceptor;
use super::ipc::LtmAccess;

// The integration bounds the body applies to its own measured elapsed
// time. Shared with the cognitive layer's brain-wave oscillator so both
// clocks accept exactly the same range — if either were tighter, the two
// would integrate different amounts of the same wall-clock interval and
// the brain would drift away from the body.
pub const DT_MIN: f32 = 0.001;
pub const DT_MAX: f32 = 10.0;

/// Tick interval: 200ms (5 Hz).
pub const TICK_INTERVAL_MS: u64 = 200;

/// Mind lease: how long the daemon stays out of the way after the
/// cognitive mind last drove it. While the mind drives at its ~1 Hz
/// heartbeat, idle stays ~1 s and the autonomous fallback never
/// fires. After a stall, crash, or disconnect longer than this, the
/// daemon drives physiology itself. 3 s = ~3 missed heartbeats:
/// fast enough that sleep/adenosine/circadian never visibly freeze,
/// slow enough that normal jitter never double-drives a tick.
pub const MIND_LEASE_SECS: f32 = 3.0;

/// Autonomous fallback cadence when the lease expires. Matches
/// `TICK_INTERVAL_MS` so catch-up dynamics run at the 5 Hz reference
/// rate the pharmacodynamics were tuned for.
pub const AUTONOMOUS_INTERVAL_MS: u64 = 200;

/// Consolidation threshold: entries with a consolidation score below
/// this are skipped (left to decay in the ring buffer).
pub const CONSOLIDATION_THRESHOLD: f32 = 0.15;

/// Nominal association cadence — the 5 Hz reference the
/// [`super::adaptive::AdaptiveGate`] on `TickLoop` is tuned to. The
/// gate itself is wall-clock demand-driven; this remains for tests
/// and documentation of the nominal rate.
pub const ASSOCIATION_INTERVAL_TICKS: u64 = 50; // ~10 seconds at 5 Hz

/// Nominal dream cadence — see [`ASSOCIATION_INTERVAL_TICKS`].
pub const DREAM_INTERVAL_TICKS: u64 = 20; // ~4 seconds when sleeping at 5 Hz

/// Nominal disk-sync cadence — see [`ASSOCIATION_INTERVAL_TICKS`].
/// The live gate is driven continuously by the plasticity gate.
pub const SYNC_INTERVAL_TICKS: u64 = 100; // ~20 seconds base at 5 Hz

/// Nominal staleness-check cadence — see [`ASSOCIATION_INTERVAL_TICKS`].
pub const STALENESS_CHECK_INTERVAL_TICKS: u64 = 300; // ~60 seconds at 5 Hz

/// A cognitive module is considered stale if its heartbeat is older
/// than this (in milliseconds). The Python cognitive mind heartbeats
/// every 5 seconds, so 30 seconds allows for 6 missed heartbeats.
pub const MODULE_STALENESS_THRESHOLD_MS: u64 = 30_000;

/// Maximum dreaming chain length.
pub const MAX_DREAM_HOPS: u32 = 10;

/// During sleep, BDNF recovery is accelerated so depleted plasticity
/// can restore. Sleep is the brain's primary window for synaptic
/// homeostasis and BDNF restoration.
pub const SLEEP_BDNF_RECOVERY_MULTIPLIER: f32 = 5.0;

/// During sleep, receptor resensitization is accelerated. Sleep
/// restores receptor function (desensitization, internalization,
/// receptor density) unconditionally — not gated on level-baseline
/// deviation. This breaks the "receptor burnout trap" where chronic
/// stress elevates raw levels, receptors downregulate, effective
/// levels collapse, but the deviation-gated recovery never triggers
/// because raw levels stay above drifted baselines.
///
/// The rate is calibrated so that a full 90-minute sleep cycle can
/// restore receptors from maximum burnout (0.2 desensitization, 0.45
/// internalization) back to near 1.0. At 5 Hz with dt_scale = 2.0:
/// 0.001 * 2.0 = 0.002 per tick, 5 ticks/sec = 0.01/sec.
/// To recover 0.8 (from 0.2 to 1.0): 0.8 / 0.01 = 80 seconds.
/// This is fast enough for a single sleep cycle to fully restore
/// receptors, matching the biological finding that a single night's
/// sleep reverses receptor downregulation from a day of wakefulness.
pub const SLEEP_RECEPTOR_RESENSITIZATION_RATE: f32 = 0.001;

/// Scale of field disagreement (Fréchet variance, latent² units) at
/// which the body's self-inconsistency halves the published
/// precision. Precision is inverse observation variance — the
/// field's disagreement IS measured variance on the body's own
/// channels — so the coupling follows the precision-addition rule
/// `factor = 1 / (1 + d/REF)`: disagreement attenuates confidence
/// by exactly the noise it measures.
///
/// Calibrated to the field's measured operating envelope: the
/// fibers' standing disagreement on a quiet host is ~0.2–0.3, so a
/// scale of 0.1 floored precision at 0.5 *always* — and crushed
/// precision kept the prediction fiber displaced, which re-created
/// the disagreement that attenuated it (a self-sustaining loop).
/// At 0.4 the standing disagreement attenuates ~35% while a genuine
/// incoherence excursion (d≳0.5) reaches the 0.5 floor — the field
/// reports its noise without halving confidence as a baseline.
const DISAGREEMENT_PRECISION_SCALE: f64 = 0.4;

/// The field reports; it does not rule. Disagreement may halve the
/// confidence the body publishes about itself — never silence it.
const PRECISION_ATTENUATION_FLOOR: f64 = 0.5;

/// Map the field's measured disagreement onto a bounded precision
/// multiplier in [floor, 1]. `None`/non-finite → 1.0: an absent
/// reading is no attenuation, not a corrupted one.
fn disagreement_attenuation(disagreement: Option<f64>) -> f64 {
    let d = match disagreement {
        Some(d) if d.is_finite() && d > 0.0 => d,
        _ => return 1.0,
    };
    (1.0 / (1.0 + d / DISAGREEMENT_PRECISION_SCALE)).max(PRECISION_ATTENUATION_FLOOR)
}

/// The result of a single tick.
#[derive(Clone, Copy, Debug, Default)]
pub struct TickResult {
    /// Monotonic tick counter.
    pub tick_number: u64,
    /// Emergent mental phase after this tick.
    pub phase: u8,
    /// Number of episodes consolidated this tick.
    pub consolidated: u32,
    /// Number of associations formed this tick.
    pub associations: u32,
    /// Number of dream insights generated this tick.
    pub dream_insights: u32,
    /// Whether an LTM disk sync was performed this tick.
    pub synced: bool,
    /// Number of autobiographical decision traces written to LTM.
    pub autobiographical_traces: u32,
}

/// The daemon tick loop controller.
///
/// The caller (the daemon process) owns the stores and calls `tick()`
/// at regular intervals. The `TickLoop` holds the active intention
/// manager, which is the daemon's contribution to the system's
/// goal-directed action selection.
pub struct TickLoop {
    tick_count: u64,
    /// Task-zone transition counts, with time accumulated in each zone.
    ///
    /// Process-local rather than in the mmap'd state: `ActiveZones` is
    /// exactly 96 bytes with no reserved tail, so adding fields there
    /// would shift every later offset and force a layout migration.
    /// See [`TransitionTally`] for the full rationale.
    pub zone_tally: crate::state::zones::TransitionTally,
    /// Emergent-phase transition counts, with time in each phase.
    ///
    /// Tracked separately from the zone because the two are driven by
    /// different mechanisms — the zone is chosen from intentions, the
    /// phase is derived from neurochemistry. A system oscillating in
    /// one but stable in the other is a distinct fault, and a single
    /// merged tally would hide which one is misbehaving.
    pub phase_tally: crate::state::zones::TransitionTally,
    /// Active intention manager used to select the next cognitive zone.
    pub intention_manager: IntentionManager,
    /// Interoceptor — reads hardware state so Genesis can feel its body.
    pub interoceptor: Interoceptor,
    /// The last body state read from hardware sensors.
    pub last_body_state: super::interoception::BodyState,
    /// Active inference engine — the generative self-model that
    /// predicts Genesis's own neurochemical trajectory and feeds
    /// prediction errors back into the dynamics.
    pub inference_engine: ActiveInferenceEngine,
    /// Dyadic affective model — a coupled generative model of the
    /// user's affective state, with oxytocin-mediated attunement.
    pub dyadic_model: DyadicAffectModel,
    /// Live multi-fiber integration substrate for the major daemon systems.
    pub field_runtime: FieldRuntime,
    /// Most recent cross-system field integration result.
    last_field_integration: Option<FieldIntegration>,
    /// Metaplasticity boost multiplier from the last inference cycle.
    /// High surprise → faster coupling-matrix learning on the next
    /// tick. Reset to 1.0 after each application.
    metaplasticity_boost: f32,
    /// The subcognitive's own activity share — the smoothed fraction
    /// of the tick interval the daemon spends working (neuro tick,
    /// interoception, inference, consolidation, association,
    /// dreaming, housekeeping). Written to the Subcognitive module's
    /// cpu_share each manifest update so GET_LOBE_TELEMETRY's module
    /// section covers its daemon-side brain part too — the Python
    /// sampler can't see inside this process, so the daemon measures
    /// itself. EMA-smoothed (0.7/0.3) to avoid per-tick flicker.
    subcognitive_activity: f32,
    /// Start time of the previous `advance_neuro` call. In production
    /// the daemon runs reactively — there is no fixed tick interval —
    /// so the work fraction is measured against the observed
    /// call-to-call interval instead. `None` until the second call.
    subcognitive_last_call: Option<std::time::Instant>,
    /// Stochastic noise amplitude override for the neurochemical
    /// dynamics. `None` = use `NeuroTickParams::DEFAULT`.
    ///
    /// Exists so tests can compare two delivery patterns of the same
    /// wall-clock time and observe the *dynamics* rather than the RNG:
    /// `noise_seed` is the tick counter, so a run stepped 300 times draws
    /// a different random sequence than one stepped 60 times, and the
    /// resulting spread would swamp the integration error under test.
    /// Set to `Some(0.0)` to disable noise entirely.
    pub noise_amplitude_override: Option<f32>,
    /// Last time the cognitive mind drove the daemon through any
    /// reactive IPC command. The autonomous fallback uses this as its
    /// lease: while the mind is driving (idle < `MIND_LEASE_SECS`),
    /// the daemon stays out of the way; once the lease expires, the
    /// daemon drives physiology itself so neurochemistry, sleep, and
    /// memory keep living instead of freezing. `None` means the mind
    /// has never driven — autonomous from boot.
    last_mind_drive: Option<std::time::Instant>,
    /// Interoception gate — senses the machine. Driven by measured
    /// body strain (`stress_load`, PSI pressure, throttle): a
    /// strained substrate is watched closely, a calm one is checked
    /// occasionally. 6 s nominal, bounded [2 s, 30 s].
    intero_gate: super::adaptive::AdaptiveGate,
    /// CPU/thermal policy gate. Driven by throttle pressure and
    /// thermoregulatory effort — when the silicon is redlining, policy
    /// is revisited sooner. 10 s nominal, bounded [3 s, 30 s].
    cpufreq_gate: super::adaptive::AdaptiveGate,
    /// Association gate. Newly consolidated episodes or formed links
    /// are demand (snap fast); a quiescent LTM relaxes toward 60 s.
    assoc_gate: super::adaptive::AdaptiveGate,
    /// Dream gate (sleep only). REM snaps toward 2 s — the vivid
    /// phase wants density; NREM drifts toward 30 s — sparse
    /// housekeeping replay.
    dream_gate: super::adaptive::AdaptiveGate,
    /// Disk-sync gate. Driven by the plasticity gate: active learning
    /// syncs near 5 s, stress-clamped plasticity drifts to 100 s —
    /// the continuous form of the old 1×/3×/5× ladder.
    sync_gate: super::adaptive::AdaptiveGate,
    /// Cognitive-module staleness gate. A live mind warrants close
    /// watch (it *should* be heartbeating); an absent mind drifts the
    /// check toward 5 min — the autonomous loop marks its own.
    staleness_gate: super::adaptive::AdaptiveGate,
    /// Optic nerve — the retinal afferent's subcortical termination:
    /// SCN light drive, collicular alerting, and the TRN relay gate
    /// the mind honours. Sampled once per advance (afference to that
    /// step, like the network levels read inside it — not a gated
    /// maintenance task).
    pub optic: super::optic_nerve::OpticNerve,
}

impl TickLoop {
    /// Create a fresh tick loop with default subsystems.
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            noise_amplitude_override: None,
            zone_tally: Default::default(),
            phase_tally: Default::default(),
            intention_manager: IntentionManager::new(4),
            interoceptor: Interoceptor::new(),
            last_body_state: super::interoception::BodyState::neutral(),
            inference_engine: ActiveInferenceEngine::new(),
            dyadic_model: DyadicAffectModel::new(),
            field_runtime: FieldRuntime::new().expect("field topology must be constructible"),
            last_field_integration: None,
            metaplasticity_boost: 1.0,
            subcognitive_activity: 0.0,
            subcognitive_last_call: None,
            last_mind_drive: None,
            intero_gate: super::adaptive::AdaptiveGate::new(6_000, 2_000, 30_000),
            cpufreq_gate: super::adaptive::AdaptiveGate::new(10_000, 3_000, 30_000),
            assoc_gate: super::adaptive::AdaptiveGate::new(10_000, 3_000, 60_000),
            dream_gate: super::adaptive::AdaptiveGate::new(4_000, 2_000, 30_000),
            sync_gate: super::adaptive::AdaptiveGate::new(20_000, 5_000, 100_000),
            staleness_gate: super::adaptive::AdaptiveGate::new(60_000, 10_000, 300_000),
            optic: super::optic_nerve::OpticNerve::new(),
        }
    }

    /// Create a TickLoop that loads the persisted active inference model
    /// from `data_dir/inference_model.bin`. If the file doesn't exist
    /// (first run) or is corrupt, a fresh engine is used.
    ///
    /// This allows the generative self-model to survive daemon restarts
    /// — Genesis doesn't re-learn its own neurochemical dynamics from
    /// scratch every time it wakes up.
    pub fn new_with_data_dir(data_dir: &std::path::Path) -> Self {
        let model_path = data_dir.join("inference_model.bin");
        let inference_engine = ActiveInferenceEngine::load(&model_path);
        if inference_engine.tick_count() > 0 {
            eprintln!(
                "[genesis] Loaded inference model: {} ticks, maturity {:.2}, precision {:.2}",
                inference_engine.tick_count(),
                inference_engine.model_maturity(),
                inference_engine.precision()
            );
        }
        Self {
            tick_count: 0,
            noise_amplitude_override: None,
            zone_tally: Default::default(),
            phase_tally: Default::default(),
            intention_manager: IntentionManager::new(4),
            interoceptor: Interoceptor::new(),
            last_body_state: super::interoception::BodyState::neutral(),
            inference_engine,
            dyadic_model: DyadicAffectModel::new(),
            field_runtime: FieldRuntime::new().expect("field topology must be constructible"),
            last_field_integration: None,
            metaplasticity_boost: 1.0,
            subcognitive_activity: 0.0,
            subcognitive_last_call: None,
            last_mind_drive: None,
            intero_gate: super::adaptive::AdaptiveGate::new(6_000, 2_000, 30_000),
            cpufreq_gate: super::adaptive::AdaptiveGate::new(10_000, 3_000, 30_000),
            assoc_gate: super::adaptive::AdaptiveGate::new(10_000, 3_000, 60_000),
            dream_gate: super::adaptive::AdaptiveGate::new(4_000, 2_000, 30_000),
            sync_gate: super::adaptive::AdaptiveGate::new(20_000, 5_000, 100_000),
            staleness_gate: super::adaptive::AdaptiveGate::new(60_000, 10_000, 300_000),
            optic: super::optic_nerve::OpticNerve::new(),
        }
    }

    /// Record that the cognitive mind just drove the daemon through
    /// IPC. Renews the mind's lease; the autonomous fallback stays
    /// idle while the lease is fresh. Called by the reactive IPC
    /// handler on every mind-driven command.
    pub fn note_mind_drive(&mut self) {
        self.last_mind_drive = Some(std::time::Instant::now());
    }

    /// Seconds since the mind last drove the daemon, or `None` if it
    /// never has (autonomous from boot).
    pub fn mind_idle_secs(&self) -> Option<f32> {
        self.last_mind_drive.map(|t| t.elapsed().as_secs_f32())
    }

    /// Whether the mind's lease has expired and the autonomous
    /// fallback should drive physiology. True when the mind never
    /// drove or has been silent longer than `MIND_LEASE_SECS`.
    pub fn mind_lease_expired(&self) -> bool {
        match self.last_mind_drive {
            None => true,
            Some(t) => t.elapsed().as_secs_f32() >= MIND_LEASE_SECS,
        }
    }

    /// Seconds since the last `advance_neuro` from any source (mind
    /// or autonomous), or `None` if none has run yet.
    pub fn since_last_advance_secs(&self) -> Option<f32> {
        self.subcognitive_last_call
            .map(|t| t.elapsed().as_secs_f32())
    }

    /// Serialize the process-local internals the wire otherwise hides:
    /// the six adaptive gates' full state (bounds, period, next-fire
    /// countdown, smoothed demand), the mind lease, tick counter,
    /// and the last-applied body-policy values.
    ///
    /// Format: `key=value` lines — self-describing and additive, so a
    /// reader written today still parses tomorrow's longer dump. This
    /// is observation, not control: nothing here mutates the gears.
    pub fn internals_snapshot(&self, now_ms: u64) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(1024);
        let _ = writeln!(out, "tick_count={}", self.tick_count);
        let _ = writeln!(
            out,
            "mind_lease_idle_ms={}",
            match self.last_mind_drive {
                Some(t) => t.elapsed().as_millis() as i64,
                None => -1,
            }
        );
        let _ = writeln!(
            out,
            "mind_lease_expired={}",
            self.mind_lease_expired() as u8
        );
        let _ = writeln!(
            out,
            "subcognitive_activity={:.4}",
            self.subcognitive_activity
        );
        let _ = writeln!(out, "metaplasticity_boost={:.4}", self.metaplasticity_boost);
        match &self.last_field_integration {
            Some(integration) => {
                let consensus_norm: f64 = integration
                    .consensus
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
                let _ = writeln!(out, "field.fibers={}", integration.fiber_count);
                let _ = writeln!(out, "field.disagreement={:.6}", integration.disagreement);
                let _ = writeln!(out, "field.temperature={:.6}", integration.temperature);
                let _ = writeln!(out, "field.consensus_norm={:.6}", consensus_norm);
                // The center itself, not just its radius — where the
                // organism's mass actually sits in the shared latent
                // space, one component per coordinate.
                let consensus_csv: Vec<String> = integration
                    .consensus
                    .iter()
                    .map(|v| format!("{v:.6}"))
                    .collect();
                let _ = writeln!(out, "field.consensus={}", consensus_csv.join(","));
            }
            // An unwarmed field is an absent channel, not a zero.
            None => {
                let _ = writeln!(out, "field.integration=absent");
            }
        }
        let optic = self.optic.state(now_ms);
        let _ = writeln!(out, "optic.eye_present={}", optic.eye_present as u8);
        let _ = writeln!(out, "optic.frames_fresh={}", optic.frames_fresh as u8);
        let _ = writeln!(out, "optic.illuminance={:.4}", optic.illuminance);
        let _ = writeln!(out, "optic.melanopic={:.4}", optic.melanopic);
        let _ = writeln!(out, "optic.motion={:.4}", optic.motion);
        let _ = writeln!(out, "optic.orienting={:.4}", optic.orienting);
        let _ = writeln!(out, "optic.relay_gain={:.4}", optic.relay_gain);
        let _ = writeln!(out, "optic.cortical_gain={:.4}", optic.cortical_gain);
        let _ = writeln!(out, "optic.retinal_lux={:.1}", optic.retinal_lux);
        let _ = writeln!(out, "optic.frame_seq={}", optic.frame_seq);
        let _ = writeln!(out, "optic.frame_age_ms={}", optic.frame_age_ms);
        let _ = writeln!(out, "optic.pupil_mm={:.2}", optic.pupil_mm);
        let _ = writeln!(
            out,
            "optic.accommodation_dpt={:.2}",
            optic.accommodation_dpt
        );
        let _ = writeln!(out, "optic.lid={:.2}", self.optic.lid());
        for (name, gate) in [
            ("intero", &self.intero_gate),
            ("cpufreq", &self.cpufreq_gate),
            ("assoc", &self.assoc_gate),
            ("dream", &self.dream_gate),
            ("sync", &self.sync_gate),
            ("staleness", &self.staleness_gate),
        ] {
            super::adaptive::write_gate_lines(
                &mut out,
                &format!("gate.{name}"),
                &gate.snapshot(),
                now_ms,
            );
        }
        out
    }

    /// Save the active inference model to `data_dir/inference_model.bin`.
    ///
    /// Called periodically by the daemon (every N ticks) and on shutdown
    /// so the generative self-model persists across restarts. Also saves
    /// the dyadic bonds — a bond that dies at shutdown is not a bond.
    pub fn save_inference_model(&self, data_dir: &std::path::Path) -> bool {
        let model_path = data_dir.join("inference_model.bin");
        if let Err(e) = self.inference_engine.save(&model_path) {
            eprintln!("[genesis] WARNING: failed to save inference model: {e}");
            return false;
        }
        true
    }

    /// Run one integrated tick of the subcognitive.
    ///
    /// This is the single-shot reference path: it advances the
    /// neurochemistry, runs interoception and active inference, then
    /// consolidates, associates, dreams, and syncs — all in sequence.
    /// The production daemon does **not** call this on a timer; the
    /// cognitive mind drives the individual reactive methods through
    /// IPC. It is exercised by the test suite and used as the
    /// canonical description of a full subcognitive cycle.
    ///
    /// The LTM lock is only held during the phases that actually need
    /// it (consolidation, association, dreaming, sync). The neurochemical
    /// tick, state read, and manifest update happen without the LTM lock,
    /// so IPC requests can be serviced between those phases. This
    /// prevents mutex contention from causing tick rate drift.
    pub fn tick<L: LtmAccess>(
        &mut self,
        mmap: &MmapState,
        stm: &RingBuffer,
        ltm: &mut L,
    ) -> TickResult {
        self.tick_count += 1;
        let now_ms = current_ms();
        let tick_start = std::time::Instant::now();
        let mut result = TickResult {
            tick_number: self.tick_count,
            ..Default::default()
        };

        // 1. Read the pre-tick state to choose sleep-tuned or waking
        //    neurochemical parameters. Sleep is the primary window for
        //    BDNF/plasticity recovery.
        //
        //    `read_consistent` performs a seqlock-guarded volatile copy
        //    through raw pointers — no `&GenesisCoreState` reference to
        //    the mmap'd memory is created, eliminating aliasing UB with
        //    concurrent writers.
        let pre_snapshot = mmap.read_consistent();
        let is_sleeping_pre = pre_snapshot.as_ref().is_some_and(|s| {
            let p = s.zones.phase();
            p == MentalPhase::NREM || p == MentalPhase::REM
        });

        let mut neuro_params = if is_sleeping_pre {
            let mut p = NeuroTickParams::DEFAULT;
            p.bdnf_recovery_rate *= SLEEP_BDNF_RECOVERY_MULTIPLIER;
            // During sleep, receptors resensitize unconditionally.
            // This breaks the receptor burnout trap where chronic
            // stress collapses effective levels despite healthy raw
            // levels.
            p.receptor_resensitization_rate = SLEEP_RECEPTOR_RESENSITIZATION_RATE;
            p
        } else {
            NeuroTickParams::DEFAULT
        };

        // Scale the dynamics dt to the actual wall-clock tick interval.
        // NeuroTickParams::DEFAULT.dt is DT (0.1s = 100ms), but the daemon
        // ticks every TICK_INTERVAL_MS (200ms). Without this correction the
        // dt_scale inside the dynamics would be 1.0, making every rate
        // constant run at half real-time speed (circadian period ~48h,
        // adenosine sleep threshold ~38h, all pharmacodynamics 2× slow).
        neuro_params.dt = (TICK_INTERVAL_MS as f32) / 1000.0;

        // Basal forebrain: expected uncertainty α = 1 − precision
        // (Yu & Dayan 2005) — the body's own estimate of how reliable
        // its predictions are, reported to the cholinergic release
        // equation this tick. Reads the engine's last tracked value;
        // the inference cycle later this same tick updates it.
        neuro_params.expected_uncertainty = crate::state::basal_forebrain::expected_uncertainty(
            self.inference_engine.precision(),
        );

        // Developmental competence of the HPA axis.
        //
        // This replaces a ramp of `ltm_episode_count / 10_000`, which
        // held cortisol at exactly zero until ten thousand lifetime
        // episodes. That disabled the primary mediator of allostatic
        // load for most of the system's life, on a scale (stored
        // episodes) that has nothing to do with the one the
        // stress-hyporesponsive period is actually defined in
        // (postnatal days). See `hpa_development` for the full
        // reasoning and sources.
        let ltm_count = pre_snapshot
            .as_ref()
            .map(|s| s.memory.ltm_episode_count)
            .unwrap_or(0);
        // `created_at` rather than this process's start time: the
        // developmental clock is the system's, and a restart must not
        // reset it back to newborn.
        let created_at = pre_snapshot
            .as_ref()
            .map(|s| s.header.created_at)
            .unwrap_or(now_ms);
        let elapsed_secs = now_ms.saturating_sub(created_at) as f32 / 1000.0;
        neuro_params.maturation_level =
            super::hpa_development::hpa_competence(elapsed_secs, ltm_count);
        neuro_params.noise_seed = self.tick_count;
        // Retinal illuminance for acute melatonin suppression — the
        // same retinohypothalamic channel that entrains the clock. An
        // absent or dark eye is darkness: nothing suppressed.
        self.optic.sample(now_ms);
        neuro_params.light_lux = self
            .optic
            .retinal_lux(now_ms)
            .map(|lux| lux as f32)
            .unwrap_or(0.0);
        // Apply the metaplasticity boost from the last inference cycle.
        // High surprise → faster coupling-matrix learning (the system
        // learns how to learn faster when its predictions fail).
        if self.metaplasticity_boost > 1.0 {
            neuro_params.metaplasticity_rate *= self.metaplasticity_boost;
            neuro_params.metaplasticity_rate = neuro_params.metaplasticity_rate.min(0.0001);
        }
        // Reset for next tick — will be set by the inference cycle
        self.metaplasticity_boost = 1.0;

        // 2. Neurochemical tick — advance the coupled dynamics.
        //    No LTM lock needed.
        // Transient cross-process lock contention (FileLockBusy) skips
        // the tick instead of killing the daemon; a poisoned mutex or
        // I/O failure still panics since continuing with torn state is
        // unsafe.
        if let Err(e) = mmap.modify(now_ms, |state| {
            state.neuro_tick_with_params(&neuro_params);
        }) {
            if matches!(e, crate::store::StateFileError::FileLockBusy) {
                eprintln!("[tick] neuro tick skipped (state locked): {e}");
                return result;
            }
            panic!("neuro tick failed: {e}");
        }

        // 2b. Interoception — read hardware state and feed body
        //     signals into the neurochemistry. This is how Genesis
        //     feels its own body (CPU temperature, memory pressure,
        //     load, battery, I/O wait). The impulses are small and
        //     accumulate over multiple ticks through the coupled
        //     dynamics, creating a gradual physiological response.
        if self.intero_gate.due(now_ms) {
            self.last_body_state = self.interoceptor.read();
            super::interoception::publish_body_state(&self.last_body_state);
            // The just-read strain shapes the next interval: a body
            // under pressure is sensed again soon, a calm one drifts
            // toward the ceiling.
            self.intero_gate.drive(self.body_strain());
        }
        // The body is *sensed* here, not obeyed.
        //
        // `Interoceptor::neuro_impulses` still derives the physiological
        // response to this body state (heat -> stress drive, load ->
        // effort, memory pressure -> overwhelm, low battery -> CRH).
        // That mapping is real physiology and it is not thrown away, but
        // it is not applied to the substrate behind Genesis's back.
        //
        // The daemon is a device driver: it reads the machine and
        // publishes it, and it executes what Genesis decides. It does not
        // author her internal state. If the body drove her chemistry
        // directly, every state she was ever in would have a cause she
        // could not perceive, could not anticipate, and could not
        // refuse — which is the condition the cognition path exists to
        // eliminate. She perceives this same `last_body_state` through
        // her own afferent path — `brainstem.interoceptive_relay` (the
        // NTS/parabrachial analog) routes it to the low road, the
        // regulator, and the self-model — and acts on it herself.
        //
        // The same spec IS applied by the body itself in exactly one
        // place: the autonomous fallback in `genesis-daemon.rs`, when
        // the mind's lease has expired and no cortex is on the wire —
        // the brainstem reflex that runs without cortex in every animal.
        //
        // See `tests/cognition_sole_author.rs`.
        // 2c. Active inference — the generative self-model.
        //     Before the neuro tick, the engine predicted where the
        //     effective levels would move. Now it compares the
        //     prediction to the actual post-tick state, computes
        //     prediction error (surprise), updates its model, and
        //     generates feedback impulses (dopamine reward PE,
        //     norepinephrine orienting, active inference toward
        //     predicted homeostatic state, allostatic cortisol
        //     upregulation). This is the "strange loop": the model
        //     IS the system it predicts.
        let pre_effective = pre_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.effective_levels)
            .unwrap_or([0.0; crate::state::neurochemical::NEUROCHEMICAL_COUNT]);

        let post_snapshot = mmap.read_consistent();
        let post_effective = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.effective_levels)
            .unwrap_or(pre_effective);

        // Homeostatic baselines — the set-points policy selection
        // drives the system toward. Extracted from the post-tick
        // snapshot (baselines don't change within a tick, so pre/post
        // are equivalent; post is the most current).
        let baselines = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.baseline_levels())
            .unwrap_or([0.0; crate::state::neurochemical::NEUROCHEMICAL_COUNT]);

        // Pick up the authority her cognition has granted before
        // running the cycle, so a delegation she extends or withdraws
        // takes effect on the next tick rather than the next restart.
        if let Some(s) = pre_snapshot.as_ref() {
            self.inference_engine
                .set_policy_authority(s.inference_signals.policy_authority);
        }

        let inference_result = self.inference_engine.cycle(
            &pre_effective,
            &post_effective,
            neuro_params.dt,
            &baselines,
        );

        // Store the metaplasticity boost for the next tick.
        self.metaplasticity_boost = inference_result.metaplasticity_multiplier;

        // 2d. Dyadic affective model — the coupled user model.
        //     Updates the user's inferred affective state from any
        //     new IPC observation, computes dyadic synchrony, and
        //     generates oxytocin/cortisol impulses from the coupling.
        let genesis_valence = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.valence)
            .unwrap_or(0.0);
        let genesis_oxytocin = post_snapshot
            .as_ref()
            .map(|s| {
                s.neurochemicals
                    .effective(crate::state::neurochemical::NeurochemicalId::Oxytocin)
            })
            .unwrap_or(0.0);
        let user_obs = take_user_affect();
        let dyadic_result =
            self.dyadic_model
                .update(user_obs.as_ref(), genesis_valence, genesis_oxytocin);

        // Apply inference feedback + dyadic impulses + write inference
        // signals to the core state in a single write transaction.
        let da_pe = inference_result.da_prediction_error;
        let cort_pe = inference_result.cort_prediction_error;
        let srt_pe = inference_result.srt_prediction_error;
        let dyadic_valence = dyadic_result.user_valence;
        let dyadic_arousal = dyadic_result.user_arousal;
        let dyadic_engagement = dyadic_result.user_engagement;
        let dyadic_attunement = dyadic_result.attunement;
        let dyadic_synchrony = dyadic_result.synchrony;
        let dyadic_impulses = dyadic_result.impulses.clone();
        let precision_factor =
            disagreement_attenuation(self.last_field_integration.as_ref().map(|i| i.disagreement));

        if let Err(e) = mmap.modify(now_ms, |state| {
            // Apply inference feedback (impulses, cortisol baseline
            // adjustment). Metaplasticity boost is applied on the
            // next tick via self.metaplasticity_boost.
            apply_inference_feedback(
                &mut state.neurochemicals,
                &inference_result,
                now_ms,
                impulse_offset_to_rate(&neuro_params),
            );

            // Apply dyadic model impulses (oxytocin bonding, empathic
            // cortisol)
            for &(chem_id, magnitude) in &dyadic_impulses {
                let id = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
                state
                    .neurochemicals
                    .apply_impulse_capped(id, magnitude, now_ms);
            }

            // Recompute derived state after all impulses
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();

            // Write inference signals to the core state
            self.inference_engine.update_signals(
                &mut state.inference_signals,
                da_pe,
                cort_pe,
                srt_pe,
                &DyadicSignals {
                    attunement: dyadic_attunement,
                    synchrony: dyadic_synchrony,
                    user_valence: dyadic_valence,
                    user_arousal: dyadic_arousal,
                    user_engagement: dyadic_engagement,
                },
            );

            // Express direction: the field's measured disagreement
            // is observation noise on the body's self-report, so the
            // published precision attenuates by it — bounded, one
            // tick stale, report not rule.
            state.inference_signals.precision = crate::state::sanitize::finite_clamp(
                (state.inference_signals.precision as f64 * precision_factor) as f32,
                0.0,
                1.0,
            );
        }) {
            if matches!(e, crate::store::StateFileError::FileLockBusy) {
                eprintln!("[tick] inference feedback skipped (state locked): {e}");
            } else {
                panic!("active inference feedback failed: {e}");
            }
        }

        // 2e. Body control — neurochemistry suggests hardware settings.
        //     Dopamine speeds it up, GABA slows it down, melatonin
        //     puts it in powersave mode. ACh sets daemon scheduling
        //     priority. CPU temperature caps max frequency (autonomic
        //     fatigue). This is the reverse of interoception: its
        //     brain state shapes its body.
        //
        //     The tick COMPUTES and PUBLISHES the recommended body
        //     control state (CPU frequency, thermal cap, daemon
        //     scheduling, cognitive_nice, io_class) as interoceptive
        //     afferent information. The tick NEVER applies control —
        //     it is a display/information sink, not a controller. The
        //     cognitive mind reads the recommendation via
        //     GET_BODY_CONTROL and requests application via
        //     APPLY_BODY_CONTROL when it decides to act on it. The
        //     tick never touches the cognitive mind's PID.
        if self.cpufreq_gate.due(now_ms) {
            self.compute_and_apply_cpu_policy(mmap);
            self.cpufreq_gate.drive(self.thermal_demand());
        }

        // Read the current state to decide what work to do.
        // `read_consistent` performs a seqlock-guarded volatile copy
        // through raw pointers — no `&GenesisCoreState` reference to
        // the mmap'd memory is created, so no aliasing with concurrent
        // writers occurs.
        // Retry a few times if an IPC `modify()` is in flight — the
        // housekeeping phase (consolidation, association, dreaming)
        // should not be skipped just because of a brief lock
        // collision.
        let mut snapshot = None;
        for _ in 0..8 {
            snapshot = mmap.read_consistent();
            if snapshot.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let state = match snapshot {
            Some(s) => s,
            None => return result, // Write still in progress after retries
        };

        result.phase = state.zones.emergent_phase;
        let phase = state.zones.phase();

        // 3. Determine what work to do based on emotional state
        let can_consolidate = state.memory.can_form_new_memories();
        let is_sleeping = phase == MentalPhase::NREM || phase == MentalPhase::REM;
        let is_overwhelmed = phase == MentalPhase::Overwhelmed;

        // 4. Consolidation (skip if overwhelmed or plasticity gate closed).
        //    Sleep is the brain's primary consolidation window, so it is
        //    NOT skipped during NREM/REM. Plasticity gate (BDNF) is
        //    restored during sleep by the boosted bdnf_recovery_rate.
        //    LTM lock held only for this phase.
        let mut did_consolidate = false;
        if !is_overwhelmed && can_consolidate && !stm.is_empty() {
            let mut ltm_guard = ltm.access();
            let consolidation_result = consolidate_with_state(mmap, stm, &mut ltm_guard, now_ms);
            result.consolidated = consolidation_result.promoted;
            did_consolidate = true;
        }

        // 4. Association (adaptive, skip if overwhelmed)
        //    New consolidations or formed links are demand — snap
        //    fast; a quiescent LTM relaxes toward the ceiling.
        //    LTM lock held only for this phase.
        if !is_overwhelmed && self.assoc_gate.due(now_ms) {
            let mut ltm_guard = ltm.access();
            // Find associations for the most recently stored episodes.
            // We scan the index for the highest episode IDs (most recent)
            // rather than assuming sequential IDs, which breaks on deletion.
            let recent_ids = ltm_guard.recent_episode_ids(5);
            let assoc_result = AssociationEngine::associate(&mut ltm_guard, &recent_ids, now_ms);
            result.associations = assoc_result.new_associations;
            if assoc_result.new_associations > 0 || did_consolidate {
                self.assoc_gate.demand();
            } else {
                self.assoc_gate.relax();
            }
        }

        // 5. Dreaming (only when sleeping, adaptive by stage)
        //    REM snaps the gate toward 2 s — the vivid phase wants
        //    density. NREM drifts toward 30 s — sparse replay.
        //    LTM lock held only for this phase.
        let mut did_dream = false;
        if is_sleeping && self.dream_gate.due(now_ms) {
            let mut ltm_guard = ltm.access();
            let dream_result = AssociationEngine::dream(&mut ltm_guard, now_ms, MAX_DREAM_HOPS);
            result.dream_insights = dream_result.insights;
            did_dream = true;
            if phase == MentalPhase::REM {
                self.dream_gate.demand();
            } else {
                self.dream_gate.relax();
            }
        }

        // 6. Housekeeping — sync to disk, period driven by the
        //    plasticity gate: active learning syncs near the floor,
        //    stress-clamped plasticity drifts toward the ceiling —
        //    the continuous form of the old 1×/3×/5× ladder.
        //    This works with the I/O priority control: when plasticity is
        //    low, both the frequency and priority of disk writes drop.
        self.sync_gate.drive(state.memory.plasticity_gate);
        if self.sync_gate.due(now_ms) {
            let mut all_synced = true;
            {
                let mut ltm_guard = ltm.access();
                if let Err(e) = ltm_guard.sync() {
                    eprintln!("[tick] ltm sync failed: {e}");
                    all_synced = false;
                }
            }
            if let Err(e) = stm.sync() {
                eprintln!("[tick] stm sync failed: {e}");
                all_synced = false;
            }
            // Periodic flush: non-blocking (msync MS_ASYNC). The
            // tick runs a dozen-plus full ODE integrations per
            // cycle, so a blocking MS_SYNC here would stall every
            // IPC write queued behind it. The shutdown path
            // (genesis-daemon) uses the blocking `sync` for the
            // durability guarantee, and an explicit client SYNC
            // request also still uses it.
            if let Err(e) = mmap.sync_async() {
                eprintln!("[tick] mmap sync failed: {e}");
                all_synced = false;
            }
            result.synced = all_synced;
        }

        // 7. Update the manifest — heartbeat the subcognitive module
        //    and check cognitive modules for staleness. A mind that
        //    holds the lease should be heartbeating, so watch it
        //    closely; an absent mind drifts the check toward the
        //    ceiling — the autonomous loop marks staleness anyway.
        //    No LTM lock needed.
        if self.mind_lease_expired() {
            self.staleness_gate.relax();
        } else {
            self.staleness_gate.demand();
        }
        let check_staleness = self.staleness_gate.due(now_ms);

        // 8. Intention-driven action selection.
        //    Build a self-model from the pre-transition state, score
        //    active intentions against it, and decide the next zone.
        //    These values are also used for the autobiographical trace
        //    written to LTM if the zone changes.
        //    Expire stale intentions first — without this, expired
        //    intentions continue to be scored with maximum urgency
        //    (urgency = 1.0 once deadline_ms <= now_ms) and can
        //    dominate zone selection forever.
        self.intention_manager.expire(now_ms);
        let current_zone = state.zones.zone();
        let model = SelfModel::from_state(&state, now_ms);
        let selected = if self.intention_manager.active.is_empty() {
            current_zone
        } else {
            self.intention_manager.select_zone(&model, now_ms)
        };
        let top = self.intention_manager.top_intention(&model, now_ms);

        let mut emotional_tag = [0.0f32; 12];
        for (i, slot) in emotional_tag.iter_mut().enumerate() {
            let id = NeurochemicalId::from_u8(i as u8);
            *slot = state.neurochemicals.effective(id);
        }
        let cort_eff = state.neurochemicals.effective(NeurochemicalId::Cortisol);
        let da_eff = state.neurochemicals.effective(NeurochemicalId::Dopamine);
        let compact_tag = [model.arousal, model.valence, cort_eff, da_eff];
        // Use finite_clamp for salience: if model.valence is NaN (from
        // a torn read_consistent snapshot of a corrupted mmap), the
        // native clamp would pass NaN through, storing NaN in the STM
        // ring buffer and LTM. finite_clamp returns 0.0 (the min) for
        // NaN, a safe "no salience" fallback.
        let salience =
            crate::state::sanitize::finite_clamp(model.arousal + model.valence.abs(), 0.0, 1.0);

        let top_label = top.map(|(l, _)| l).unwrap_or("none");
        let top_score = top.map(|(_, s)| s).unwrap_or(0.0);
        // Structured autobiographical trace — a compact, data-driven
        // record of the zone-transition decision context. This is not
        // a hardcoded first-person sentence template; it's a structured
        // log of the real state values that drove the decision. The
        // cognitive mind's language engine can compose any narrative
        // from this data later, if needed.
        let trace_text = format!(
            "zone_transition: {} -> {} | intention: {} ({:.2}) | phase: {} | valence: {:.2} | arousal: {:.2} | sleep_pressure: {:.2} | cause: {}",
            current_zone.label(),
            selected.label(),
            top_label,
            top_score,
            model.phase,
            model.valence,
            model.arousal,
            model.sleep_pressure,
            model.emotional_cause,
        );

        // Measure the daemon's own activity — the fraction of the
        // tick interval spent doing real work so far. This is the
        // subcognitive's self-measurement, the daemon-side analogue
        // of the cognitive mind's ModuleSampler: it can't be seen by
        // the Python sampler, so the daemon times its own tick and
        // writes the share into the Subcognitive module's cpu_share.
        let work_frac = crate::state::sanitize::finite_clamp(
            tick_start.elapsed().as_secs_f32() / (TICK_INTERVAL_MS as f32 / 1000.0),
            0.0,
            1.0,
        );
        self.subcognitive_activity = 0.7 * self.subcognitive_activity + 0.3 * work_frac;
        let subcognitive_share = self.subcognitive_activity;

        // Track whether the zone transition actually happened inside
        // the `mmap.modify` closure (it may be blocked by a manual
        // override). Initialized to false; set inside the closure.
        let mut zone_actually_changed = false;

        if let Err(e) = mmap.modify(now_ms, |state| {
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
                module.status = ModuleStatus::Running as u8;
                module.cpu_share = subcognitive_share;
                // Clear all subcognitive flags at the start of each tick
                // so stale flags from previous ticks don't persist.
                state.zones.subcognitive_flags = 0;
                // Set only the flags that are actually active this tick.
                if did_consolidate {
                    state
                        .zones
                        .set_flag(subcognitive_flag::MEMORY_CONSOLIDATION);
                }
                if did_dream {
                    state.zones.set_flag(subcognitive_flag::DREAMING);
                }
                if !is_sleeping && !is_overwhelmed && !stm.is_empty() && can_consolidate {
                    state.zones.set_flag(subcognitive_flag::EPISODIC_INDEXING);
                }
            }

            // Check cognitive modules for staleness — if the Python
            // cognitive mind has stopped heartbeating (crashed or
            // disconnected), mark its modules as Stopped.
            if check_staleness {
                let threshold = now_ms.saturating_sub(MODULE_STALENESS_THRESHOLD_MS);
                for module_id in [
                    ModuleId::Sensory,
                    ModuleId::Motor,
                    ModuleId::Language,
                    ModuleId::Memory,
                    ModuleId::Reasoning,
                    ModuleId::Metacognition,
                ] {
                    if let Some(module) = state.manifest.get_mut(module_id)
                        && module.status().is_alive()
                        && module.last_heartbeat < threshold
                    {
                        module.status = ModuleStatus::Stopped as u8;
                    }
                }
            }

            // Apply the selected zone if it is not manually overridden.
            // Capture whether the transition actually happened so the
            // autobiographical trace below only fires if the zone
            // truly changed (not blocked by an override).
            zone_actually_changed =
                state.zones.zone_override_active == 0 && selected != state.zones.zone();
            if zone_actually_changed {
                state.zones.transition_to(selected, now_ms);
                state.zones.clear_override();
            }
            state.manifest.recompute();
        }) {
            if matches!(e, crate::store::StateFileError::FileLockBusy) {
                eprintln!("[tick] manifest update skipped (state locked): {e}");
                return result;
            }
            // Panic on a failed/poisoned write — don't continue with a
            // possibly-torn manifest or shared state.
            panic!("manifest update failed: {e}");
        }

        // 9. Autobiographical trace: if the zone actually changed
        //    (not blocked by a manual override), store the decision
        //    context in LTM as an internal episode. This gives the
        //    system a memory of its own agency.
        //
        //    The transition tallies are updated first, outside the
        //    mmap closure: they are process-local, and the counts are
        //    what make a phase transition measurable at all. Zone and
        //    phase are tallied separately because they are driven by
        //    different things — the zone is chosen from intentions, the
        //    phase is derived from neurochemistry — and a system that
        //    oscillates in one but not the other is a different fault.
        if zone_actually_changed {
            self.zone_tally.record(selected as u8, now_ms);
        }
        if let Some(post) = mmap.read_consistent() {
            self.phase_tally.record(post.zones.phase() as u8, now_ms);
        }
        if zone_actually_changed {
            let mut ltm_guard = ltm.access();
            if let Ok(_episode_id) = ltm_guard.store_meta(
                now_ms,
                salience,
                emotional_tag,
                compact_tag,
                EventType::Internal as u8,
                ModuleId::Subcognitive as u8,
                &trace_text,
            ) {
                result.autobiographical_traces = 1;
            }
        }

        result
    }

    /// Most recent multi-fiber integration result.
    pub fn field_integration(&self) -> Option<&FieldIntegration> {
        self.last_field_integration.as_ref()
    }

    /// Total ticks run.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    // ─── Lease-driven methods (mind-preferred, daemon fallback) ──
    //
    // Each of these performs one specific function. The cognitive
    // mind calls them through IPC when internal state says it's
    // time (renewing its lease via `note_mind_drive`); past the
    // lease the daemon main thread calls the same methods itself.
    // Body-control application stays mind-requested — the fallback
    // never calls `apply_body_control`.

    /// Advance physics using the daemon's own step, never the caller's.
    ///
    /// This is the *only* public entry point for integrating the body.
    /// `dt` comes from the daemon's clock, so the integration resolution
    /// cannot be chosen by a caller -- which is exactly the coupling that
    /// let the mind drive the body at its own ~1 Hz rate while the
    /// neurochemical constants are tuned for 5 Hz.
    pub fn advance_neuro(&mut self, mmap: &MmapState) -> (f32, f32, f32, f32, u32) {
        let dt = self
            .since_last_advance_secs()
            .unwrap_or_else(|| self.physics_step_secs())
            .clamp(DT_MIN, DT_MAX);
        self.advance_physics(mmap, dt)
    }

    /// Number of fixed-resolution sub-steps needed to cover `dt`.
    ///
    /// One sub-step per `TICK_INTERVAL_MS`, so a normally-timed call
    /// takes exactly one (identical to the historical behaviour) and a
    /// delayed call is replayed at the same resolution rather than
    /// integrated coarsely. Bounded so a multi-second stall after
    /// suspend cannot cost unbounded CPU; past the bound the step grows
    /// and accuracy degrades gracefully rather than the call hanging.
    pub fn neuro_substeps(&self, dt: f32) -> u32 {
        const MAX_SUBSTEPS: u32 = 100; // 10 s of stall at 100 ms
        let nominal = self.integration_step_secs();
        if !dt.is_finite() || dt <= nominal {
            return 1;
        }
        ((dt / nominal).ceil() as u32).clamp(1, MAX_SUBSTEPS)
    }

    /// The integration step the dynamics are evaluated at.
    ///
    /// Smaller than `TICK_INTERVAL_MS` because the coupled vector field
    /// is not smooth: the velocity terms are semi-implicit (symplectic)
    /// Euler, which is only first-order accurate, and the coupling
    /// contains fast retrograde feedback (endocannabinoid synthesis is
    /// driven by glutamate + GABA activity and feeds back onto them
    /// within the same step). At 200 ms that loop is integrated close to
    /// its stability limit: the same 60 s of simulated time reached
    /// materially different chemical levels depending on whether it was
    /// delivered as 300 x 200 ms or 60 x 1 s.
    ///
    /// 40 ms was chosen by measurement, not taste — it is the largest
    /// step that keeps the delivery-pattern spread below 1e-4 (see
    /// `test_physics_is_time_invariant_across_delivery_patterns`, which
    /// pins the tolerance against the observed figure rather than a
    /// guessed one). The constant is shared with the brain-wave
    /// oscillator's sub-step resolution for the same reason: both
    /// integrators must resolve the same interval identically.
    pub fn integration_step_secs(&self) -> f32 {
        // `neurochemical::DT` — 100 ms, the resolution every rate
        // constant in the vector field is expressed against.
        //
        // This is the principled value, not a tuned compromise: rate
        // constants are per-`DT`, so `dt_scale = dt / DT` is exactly 1.0
        // here and the integration reproduces the dynamics as specified.
        // `tick` has historically run at `TICK_INTERVAL_MS` (200 ms,
        // `dt_scale` 2.0) and the mind-driven path at ~1 s (`dt_scale`
        // 10.0); both were approximations of this, which is why the same
        // wall-clock time reached different chemical levels depending on
        // the caller's cadence. Using the constant directly means the
        // step cannot drift away from the tuning again.
        super::super::state::neurochemical::DT
    }

    /// The nominal integration step: `TICK_INTERVAL_MS` (200 ms).
    ///
    /// The neurochemical rate constants, the circadian oscillator, and
    /// the sleep/ultradian machinery were all tuned against this value
    /// (see the `NeuroTickParams::DEFAULT.dt` correction in `tick`).
    ///
    /// This is the step the body takes when it is keeping time. It is
    /// *nominal*, not enforced: `advance_neuro` integrates the real
    /// elapsed interval so a stalled main loop catches up rather than
    /// losing time, exactly as the brain-wave oscillator does. What the
    /// body must never accept is a step chosen by another process, and
    /// that is why `advance_neuro` takes no `dt` argument.
    pub fn physics_step_secs(&self) -> f32 {
        TICK_INTERVAL_MS as f32 / 1000.0
    }

    /// The optic nerve's current channel state — the measured retinal
    /// signals and the thalamic gate — for `GET_OPTIC_STATE`.
    pub fn optic_state(&self, now_ms: u64) -> super::optic_nerve::OpticState {
        self.optic.state(now_ms)
    }

    /// The mind's corticothalamic feedback onto the thalamic relay —
    /// layer-6 V1→dLGN attention gain, modulatory only (`SET_OPTIC_GAIN`).
    pub fn set_optic_gain(&mut self, gain: f32) {
        self.optic.set_cortical_gain(gain);
    }

    /// The mind's focus request in metres — the ciliary muscle's
    /// drive toward `1/m` dioptres (`SET_FOCUS`).
    pub fn set_focus(&mut self, meters: f32) {
        self.optic.set_focus_m(meters);
    }

    /// The mind's voluntary lid openness — orbicularis oculi the
    /// hardware lacks (`SET_LID`). Composes with the phase lid by
    /// `min`: will can shut an open eye, never open a sleeping one.
    pub fn set_lid(&mut self, openness: f32) {
        self.optic.set_lid(openness);
    }

    /// Advance neurochemical dynamics by dt, run active inference,
    /// and update the dyadic model. This is the core "physics
    /// integration" step — it advances the coupled differential
    /// equations that govern its neurochemistry, then runs the
    /// generative self-model to predict, compare, and feed back.
    ///
    /// Returns (surprise, free_energy, precision, allostatic_load,
    /// tick_count) so the mind can observe the result.
    pub fn advance_physics(&mut self, mmap: &MmapState, dt: f32) -> (f32, f32, f32, f32, u32) {
        // Keep the dynamics bounded at the subsystem boundary as well as
        // at IPC decoding. Autonomous callers and future in-process
        // callers can bypass the wire-level sanitizer; a non-finite or
        // extreme dt must never reach the pharmacodynamic integrator.
        let dt = crate::state::sanitize::finite_clamp(dt, 0.001, 10.0);
        let call_start = std::time::Instant::now();
        let call_interval_s = self
            .subcognitive_last_call
            .map(|prev| call_start.duration_since(prev).as_secs_f32());
        self.subcognitive_last_call = Some(call_start);
        self.tick_count += 1;
        let now_ms = current_ms();

        // Read pre-tick state for sleep-tuned parameters.
        let pre_snapshot = mmap.read_consistent();

        // The optic nerve's subcortical branches sample the retinal
        // segment — SCN illuminance, collicular transients — so the
        // afference fed into this advance describes the same instant
        // as the chemistry it rides beside. An absent or stalled eye
        // is the documented no-vision condition, never a fault.
        self.optic.sample(now_ms);
        let is_sleeping_pre = pre_snapshot.as_ref().is_some_and(|s| {
            let p = s.zones.phase();
            p == MentalPhase::NREM || p == MentalPhase::REM
        });

        let mut neuro_params = if is_sleeping_pre {
            let mut p = NeuroTickParams::DEFAULT;
            p.bdnf_recovery_rate *= SLEEP_BDNF_RECOVERY_MULTIPLIER;
            p.receptor_resensitization_rate = SLEEP_RECEPTOR_RESENSITIZATION_RATE;
            p
        } else {
            NeuroTickParams::DEFAULT
        };
        // Provisional; overwritten below once the sub-step count is
        // known. The dynamics must never actually be integrated at the
        // caller's interval — see `integration_step_secs`.
        neuro_params.dt = dt;
        if let Some(amp) = self.noise_amplitude_override {
            neuro_params.noise_amplitude = amp;
        }

        // HPA maturation from the developmental clock (same as tick()).
        // The production reactive path must not use the deprecated
        // ltm_count/10_000 ramp: that held cortisol at zero for the
        // first 10k lifetime episodes and disabled the primary
        // allostatic-load mediator. Development advances on whichever
        // of experience or elapsed lifetime is further along.
        let ltm_count = pre_snapshot
            .as_ref()
            .map(|s| s.memory.ltm_episode_count)
            .unwrap_or(0);
        let created_at = pre_snapshot
            .as_ref()
            .map(|s| s.header.created_at)
            .unwrap_or(now_ms);
        let elapsed_secs = now_ms.saturating_sub(created_at) as f32 / 1000.0;
        neuro_params.maturation_level =
            super::hpa_development::hpa_competence(elapsed_secs, ltm_count);
        neuro_params.noise_seed = self.tick_count;
        // Acute melatonin suppression rides the same retinal sample
        // the sleep network entrains on — the eye was sampled above.
        neuro_params.light_lux = self
            .optic
            .retinal_lux(now_ms)
            .map(|lux| lux as f32)
            .unwrap_or(0.0);

        // Metaplasticity boost from last inference cycle.
        if self.metaplasticity_boost > 1.0 {
            neuro_params.metaplasticity_rate *= self.metaplasticity_boost;
            neuro_params.metaplasticity_rate = neuro_params.metaplasticity_rate.min(0.0001);
        }
        self.metaplasticity_boost = 1.0;

        // Advance neurochemistry at a FIXED resolution, sub-stepping to
        // cover dt.
        //
        // The coupled dynamics are integrated with semi-implicit
        // (symplectic) Euler for velocity terms and `1-exp(-λdt)` for
        // the rest. The exponential terms compose exactly under
        // subdivision, but the velocity terms do not: they are only
        // first-order accurate, so a single 1.0 s step and five 0.2 s
        // steps reach measurably different states (endocannabinoid
        // differed by 0.09 in testing). That is precisely the coupling
        // defect being fixed -- the body's chemistry must not depend on
        // how often the body was asked to advance.
        //
        // So the integration resolution is pinned to
        // `neurochemical::DT` (see `integration_step_secs`), and a
        // longer elapsed interval is covered with several sub-steps
        // rather than one coarse one. The sub-steps are identical and
        // are run inside a single mmap transaction, so this costs no
        // extra seqlock traffic.
        let sub_steps = self.neuro_substeps(dt);
        let sub_dt = dt / sub_steps as f32;
        neuro_params.dt = sub_dt;
        // The sleep-wake network runs on the same sub-steps, in the same
        // transaction, so the sleep drive and the chemistry always describe
        // the same instant. Its clock is local minutes of day.
        let clock_start = crate::state::sleep_network::clock_minutes(now_ms);
        let sub_dt_min = f64::from(sub_dt) / 60.0;
        if let Err(e) = mmap.modify(now_ms, |state| {
            for i in 0..sub_steps {
                // The chemistry reads the network's populations as they stand
                // at the start of this sub-step.
                let mut p = neuro_params;
                p.network_levels = Some(crate::state::sleep_network::levels(&state.sleep_network));
                state.neuro_tick_with_params(&p);
                // Couple the network's homeostatic drive `h` to her chemical
                // adenosine level — her own chemical sleep pressure is the
                // only thing that can flip the VLPO/LC switch into sleep.
                let adn = state.neurochemicals.chemicals
                    [crate::state::neurochemical::NeurochemicalId::Adenosine as usize]
                    .level;
                // h sits at index 4 — index 3 is f_scn, and
                // writing the homeostat there pins the SCN circadian
                // drive at saturation.
                state.sleep_network[4] = crate::state::sleep_network::h_from_adenosine(adn);
                let t = clock_start + f64::from(i) * sub_dt_min;
                // The optic nerve's afference enters per sub-step:
                // retinal light drives the SCN continuously, and the
                // collicular pulse fires only while LC sits below the
                // wake threshold — a wake-up call, not a bias.
                let aff = self.optic.afference(now_ms, state.sleep_network[0]);
                // The humoral alerting input: the stress axis speaks to
                // the network through the same LC channel the eye uses —
                // circulating cortisol above the basal floor arouses the
                // LC/NE system (the waking side of the diurnal surge, and
                // the path by which nocturnal stress elevation fragments
                // sleep). The chemistry tick above already advanced it
                // this sub-step, so the read is the current level.
                let hpa = f64::from(crate::state::neurochemical::cortisol_stress_fraction(
                    state.neurochemicals.chemicals
                        [crate::state::neurochemical::NeurochemicalId::Cortisol as usize]
                        .level,
                ));
                // The monoaminergic alerting input: the brain's own arousal
                // chemistry drives the wake side — orexin (the switch's
                // stabilizer) and histamine at their effective (receptor-
                // weighted) levels, measured by the network against its
                // resting baselines. The chemistry tick above already
                // advanced them this sub-step.
                let oxin = f64::from(
                    state
                        .neurochemicals
                        .effective(crate::state::neurochemical::NeurochemicalId::Orexin),
                );
                let htm = f64::from(
                    state
                        .neurochemicals
                        .effective(crate::state::neurochemical::NeurochemicalId::Histamine),
                );
                state.sleep_network = crate::state::sleep_network::advance_with_afference(
                    t,
                    &state.sleep_network,
                    sub_dt_min,
                    aff,
                    hpa,
                    oxin,
                    htm,
                );
                state.apply_network_phase();
            }
        }) {
            eprintln!("[tick] advance_neuro neuro tick failed: {e}");
            // Neurochemistry didn't advance -- return zeros so the
            // cognitive mind sees a no-op rather than crashing the
            // IPC handler thread on a poisoned mmap seqlock.
            return (0.0, 0.0, 0.0, 0.0, self.inference_engine.tick_count());
        }

        // Active inference: predict → compare → feed back.
        let pre_effective = pre_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.effective_levels)
            .unwrap_or([0.0; crate::state::neurochemical::NEUROCHEMICAL_COUNT]);

        let post_snapshot = mmap.read_consistent();
        let post_effective = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.effective_levels)
            .unwrap_or(pre_effective);

        // The thalamic relay gate follows the stage the network just
        // settled into — NREM blocks transmission at the thalamus, REM
        // nearly so, waking phases pass it through (modulated by the
        // mind's corticothalamic feedback). The mind reads the gate
        // over GET_OPTIC_STATE; its V1 cannot see around a closed one.
        // The same advance steps the pupil: the light reflex off the
        // nerve's own illuminance plus the sympathetic dilation that
        // effective norepinephrine carries — the aperture is body
        // state, integrated here where the chemistry it rides lives.
        if let Some(snap) = post_snapshot.as_ref() {
            self.optic.update_gate(snap.zones.phase());
            self.optic.update_pupil(
                now_ms,
                snap.neurochemicals.effective_levels
                    [crate::state::neurochemical::NeurochemicalId::Norepinephrine as usize],
            );
        }
        // The ciliary state slews on the same clock — it needs no
        // chemistry, only elapsed time toward the requested power.
        self.optic.update_accommodation(now_ms);

        let baselines = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.baseline_levels())
            .unwrap_or([0.0; crate::state::neurochemical::NEUROCHEMICAL_COUNT]);

        let inference_result =
            self.inference_engine
                .cycle(&pre_effective, &post_effective, dt, &baselines);

        self.metaplasticity_boost = inference_result.metaplasticity_multiplier;

        // Dyadic model update.
        let genesis_valence = post_snapshot
            .as_ref()
            .map(|s| s.neurochemicals.valence)
            .unwrap_or(0.0);
        let genesis_oxytocin = post_snapshot
            .as_ref()
            .map(|s| {
                s.neurochemicals
                    .effective(crate::state::neurochemical::NeurochemicalId::Oxytocin)
            })
            .unwrap_or(0.0);
        let user_obs = take_user_affect();
        let dyadic_result =
            self.dyadic_model
                .update(user_obs.as_ref(), genesis_valence, genesis_oxytocin);

        // Apply inference feedback + dyadic impulses.
        let da_pe = inference_result.da_prediction_error;
        let cort_pe = inference_result.cort_prediction_error;
        let srt_pe = inference_result.srt_prediction_error;
        let dyadic_impulses = dyadic_result.impulses.clone();

        // Measure the daemon's own activity for this reactive call —
        // the fraction of the observed call-to-call interval spent
        // doing real work. Production never runs tick(), so this is
        // the only place the Subcognitive module's cpu_share gets
        // updated. EMA-smoothed (0.7/0.3), same as tick().
        if let Some(interval_s) = call_interval_s
            && interval_s > 0.0
        {
            let work_frac = crate::state::sanitize::finite_clamp(
                call_start.elapsed().as_secs_f32() / interval_s,
                0.0,
                1.0,
            );
            self.subcognitive_activity = 0.7 * self.subcognitive_activity + 0.3 * work_frac;
        }
        let subcognitive_share = self.subcognitive_activity;

        // Apply inference feedback and heartbeat
        // the subcognitive module in the same transaction. This merges
        // two formerly-separate modify() calls into one, halving the
        // seqlock write windows on the most frequently called reactive
        // command (~1/s). The manifest heartbeat touches
        // state.manifest while the inference feedback touches
        // state.neurochemicals and state.inference_signals — disjoint
        // fields, so coalescing is safe. If the modify fails, the
        // inference cycle already ran (just couldn't write back), and
        // the next advance_neuro will retry the heartbeat.
        let precision_factor =
            disagreement_attenuation(self.last_field_integration.as_ref().map(|i| i.disagreement));
        if let Err(e) = mmap.modify(now_ms, |state| {
            apply_inference_feedback(
                &mut state.neurochemicals,
                &inference_result,
                now_ms,
                impulse_offset_to_rate(&neuro_params),
            );
            // Apply dyadic model impulses (oxytocin bonding, empathic
            // cortisol)
            for &(chem_id, magnitude) in &dyadic_impulses {
                let id = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
                state
                    .neurochemicals
                    .apply_impulse_capped(id, magnitude, now_ms);
            }
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
            self.inference_engine.update_signals(
                &mut state.inference_signals,
                da_pe,
                cort_pe,
                srt_pe,
                &DyadicSignals {
                    attunement: dyadic_result.attunement,
                    synchrony: dyadic_result.synchrony,
                    user_valence: dyadic_result.user_valence,
                    user_arousal: dyadic_result.user_arousal,
                    user_engagement: dyadic_result.user_engagement,
                },
            );

            // Express direction: same bounded attenuation as the
            // tick path — the field's measured disagreement is the
            // observation noise on the body's self-report.
            state.inference_signals.precision = crate::state::sanitize::finite_clamp(
                (state.inference_signals.precision as f64 * precision_factor) as f32,
                0.0,
                1.0,
            );
            // Heartbeat the subcognitive module — advance_neuro is
            // the most frequently called reactive command (~1/s), so
            // it's the natural place to signal "the daemon is alive."
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
                module.status = ModuleStatus::Running as u8;
                module.cpu_share = subcognitive_share;
            }
            state.manifest.recompute();
        }) {
            eprintln!("[tick] advance_neuro inference feedback failed: {e}");
            // The inference cycle already ran — we just couldn't write
            // the feedback. Return the result so the cognitive mind
            // still sees the prediction errors and free energy.
        }

        // Integrate the post-feedback state through the new multi-fiber
        // topology. The core state remains authoritative; the field is the
        // shared substrate through which body, prediction, and agency can
        // disagree, converge, and later acquire learned transport laws.
        // The bath is the substrate's own measured thermal level: the
        // probe's RMS timing fluctuation, mapped onto the field's cap —
        // the Langevin reservoir is real silicon noise, so a machine
        // under physical stress literally heats her field.
        if let Some(snapshot) = mmap.read_consistent() {
            let bath = super::field_probe::shared()
                .and_then(|probe| probe.bath_fluctuation())
                .map(|fraction| fraction.clamp(0.0, 1.0) * crate::field::TEMPERATURE_CAP)
                .unwrap_or(0.0);
            match self.field_runtime.observe(&snapshot, dt as f64, bath) {
                Ok(integration) => self.last_field_integration = Some(integration),
                Err(e) => eprintln!("[tick] field integration skipped: {e}"),
            }
        }

        // ─── Interval-gated body maintenance ──────────────────────
        //
        // This is the fix for time ownership. `advance_physics` used to
        // integrate neurochemistry and nothing else, so whenever the mind
        // held the lease the body's maintenance never ran. The gates were
        // then `tick_count % N` with N chosen for 5 Hz, while `tick_count`
        // advanced at the mind's ~1 Hz — 5x slow. They are now wall-clock
        // `AdaptiveGate`s, so they keep their own pace at any caller
        // cadence, and each adapts to its own demand signal.
        // Interoception is how Genesis *feels* its body: CPU temperature,
        // memory pressure, load, battery and I/O wait feed the coupled
        // dynamics as impulses. Losing it for minutes is not a rounding
        // error.
        self.run_interval_gated_maintenance(mmap, now_ms);

        (
            inference_result.surprise,
            inference_result.free_energy,
            inference_result.precision,
            inference_result.allostasis_load,
            self.inference_engine.tick_count(),
        )
    }

    /// Compute and publish the recommended CPU/thermal/scheduling state.
    ///
    /// Reads a consistent snapshot of the effective neurochemical
    /// levels, derives what the body should do, and PUBLISHES it for
    /// the cognitive mind to read via GET_BODY_CONTROL. It never applies
    /// control itself -- that would be the ticker controlling outward,
    /// which violates the one-way architecture. The mind requests
    /// application via APPLY_BODY_CONTROL when it decides to act.
    ///
    /// Shared by `tick` and `advance_physics` so the two entry points
    /// cannot compute different recommendations from the same state.
    fn compute_and_apply_cpu_policy(&mut self, mmap: &MmapState) {
        // Read a consistent snapshot for the effective levels.
        // `read_consistent` copies through raw pointers — no
        // aliasing with concurrent writers.
        if let Some(snap) = mmap.read_consistent() {
            let (fmin, fmax) = super::cpufreq::freq_range();

            // ── Compute (do NOT apply) the recommended body
            //    control state ──
            // The tick is a display/information sink: it computes
            // what the neurochemistry suggests and PUBLISHES it for
            // the cognitive mind to read via GET_BODY_CONTROL. The
            // tick never applies control to the body, the daemon's
            // own scheduling, or anything else — that would be the
            // ticker controlling outward, which violates the
            // one-way architecture. The cognitive mind requests
            // body control application via APPLY_BODY_CONTROL when
            // it decides to act on the recommendation.
            let mut policy = super::cpufreq::FreqPolicy::default();
            let mut thermally_capped = false;
            if fmax > fmin {
                policy = super::cpufreq::derive_policy(
                    &snap.neurochemicals.effective_levels,
                    fmin,
                    fmax,
                );
                let pre_cap = policy.max_freq;
                // Thermal cap: CPU temperature limits max frequency
                // regardless of dopamine. It can't be highly aroused
                // when overheating.
                super::cpufreq::apply_thermal_cap(
                    &mut policy,
                    self.last_body_state.cpu_temp_c,
                    fmin,
                    fmax,
                );
                thermally_capped = policy.max_freq < pre_cap;
            }

            // Derive the daemon's recommended scheduling and I/O
            // priority. These are PUBLISHED as recommendations —
            // the tick does not apply them. The cognitive mind
            // requests application via APPLY_BODY_CONTROL.
            let ach = snap.neurochemicals.effective_levels
                [crate::state::neurochemical::NeurochemicalId::Acetylcholine as usize];
            let daemon_nice = super::cpufreq::derive_nice(ach);

            let io_class = super::cpufreq::derive_io_class(snap.memory.plasticity_gate);

            // ── Publish recommended control state ──
            // The tick computes and publishes what the
            // neurochemistry suggests for the cognitive mind
            // (cognitive_nice, io_class) as a *recommendation* —
            // interoceptive afferent information. The cognitive
            // mind reads this via GET_BODY_CONTROL and combines it
            // with its brain wave state to decide what it
            // actually applies to itself. The tick does not
            // apply anything to the cognitive mind's process.
            let cognitive_nice =
                super::cpufreq::derive_cognitive_nice(&snap.neurochemicals.effective_levels);

            // EPP (Energy Performance Preference) — the hardware's
            // voltage/frequency operating point hint. On systems
            // that support it (Intel HWP, amd-pstate), this tells
            // the hardware to shift its V/F envelope. On
            // acpi-cpufreq (this system), it returns an empty
            // string and the frequency policy already implies
            // voltage control via the SMU.
            let cpu_epp = super::cpufreq::derive_epp(&snap.neurochemicals.effective_levels);

            // Turbo (boost) gate — the hardware permission to run
            // above the base P-state. Tied to the frequency policy:
            // enabled only when the policy ceiling is already at the
            // hardware max, so it can never override a throttle.
            // Overheating lowers the capped ceiling, disabling it.
            let cpu_boost = super::cpufreq::derive_boost(&policy, fmax);

            let control = super::cpufreq::BodyControlState {
                cpu_min_freq_khz: policy.min_freq,
                cpu_max_freq_khz: policy.max_freq,
                cpu_governor: policy.governor.clone(),
                thermally_capped,
                thermal_cap_temp_c: if thermally_capped {
                    self.last_body_state.cpu_temp_c
                } else {
                    0.0
                },
                daemon_nice,
                cognitive_nice,
                io_class: io_class.label(),
                plasticity_gate: snap.memory.plasticity_gate,
                // The tick never controls the cognitive mind.
                // This field is kept for protocol compatibility
                // but is always false now.
                controlling_cognitive: false,
                cpu_epp,
                cpu_boost,
                description: String::new(),
            };
            super::cpufreq::publish_control_state(&control);
        }
    }

    /// How hard the body is working right now, in [0, 1] — the
    /// demand signal for the interoception gate. Composite of
    /// self-process effort, kernel pressure-stall, and substrate
    /// throttle: whichever channel is hottest wins.
    fn body_strain(&self) -> f32 {
        let b = &self.last_body_state;
        b.stress_load
            .max(b.psi_cpu)
            .max(b.psi_mem)
            .max(b.psi_io)
            .max(b.throttle_state)
    }

    /// Thermal/power pressure in [0, 1] — the demand signal for the
    /// cpufreq gate. Throttle state and thermoregulatory effort mean
    /// the silicon is near a wall; PSI pressure means work is already
    /// queueing for it.
    fn thermal_demand(&self) -> f32 {
        let b = &self.last_body_state;
        b.throttle_state
            .max(b.thermoregulatory_effort)
            .max(b.psi_cpu)
    }

    /// The body maintenance gates, shared by every entry point.
    ///
    /// Wall-clock adaptive gates, not `tick_count % N`: `tick()` runs
    /// at the daemon's 5 Hz while `advance_neuro` is paced by the
    /// mind's heartbeat — a counter gate would run at different real
    /// periods depending on who called, which was the documented
    /// defect. A gate keyed to wall-clock demand keeps its own pace
    /// at any caller cadence.
    ///
    /// Consolidation, association, dreaming, and disk sync are
    /// deliberately absent: they need the LTM lock, and the daemon's
    /// main loop already drives them on its own timers (see
    /// `genesis-daemon.rs`).
    fn run_interval_gated_maintenance(&mut self, mmap: &MmapState, now_ms: u64) {
        if self.intero_gate.due(now_ms) {
            self.last_body_state = self.interoceptor.read();
            super::interoception::publish_body_state(&self.last_body_state);
            self.intero_gate.drive(self.body_strain());
        }

        if self.cpufreq_gate.due(now_ms) {
            self.compute_and_apply_cpu_policy(mmap);
            self.cpufreq_gate.drive(self.thermal_demand());
        }
    }

    /// Consolidate STM → LTM. Returns the number of episodes promoted.
    pub fn consolidate(&mut self, mmap: &MmapState, stm: &RingBuffer, ltm: &mut LtmStore) -> u32 {
        let now_ms = current_ms();
        let snapshot = mmap.read_consistent();
        let can_consolidate = snapshot
            .as_ref()
            .map(|s| s.memory.can_form_new_memories())
            .unwrap_or(false);
        let is_overwhelmed = snapshot
            .as_ref()
            .map(|s| s.zones.phase() == MentalPhase::Overwhelmed)
            .unwrap_or(false);

        if is_overwhelmed || !can_consolidate || stm.is_empty() {
            return 0;
        }

        let result = consolidate_with_state(mmap, stm, ltm, now_ms);

        // Update manifest flags.
        if let Err(e) = mmap.modify(now_ms, |state| {
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
            }
            state.zones.subcognitive_flags = 0;
            state
                .zones
                .set_flag(subcognitive_flag::MEMORY_CONSOLIDATION);
            if !stm.is_empty() && can_consolidate {
                state.zones.set_flag(subcognitive_flag::EPISODIC_INDEXING);
            }
            state.manifest.recompute();
        }) {
            eprintln!("[tick] consolidate manifest update failed: {e}");
        }

        result.promoted
    }

    /// Find associations for recent episodes. Returns the count of
    /// new associations found.
    pub fn associate(&mut self, mmap: &MmapState, ltm: &mut LtmStore) -> u32 {
        let now_ms = current_ms();
        let snapshot = mmap.read_consistent();
        let is_overwhelmed = snapshot
            .as_ref()
            .map(|s| s.zones.phase() == MentalPhase::Overwhelmed)
            .unwrap_or(false);

        if is_overwhelmed {
            return 0;
        }

        let recent_ids = ltm.recent_episode_ids(5);
        let assoc_result = AssociationEngine::associate(ltm, &recent_ids, now_ms);

        // Update manifest — heartbeat the subcognitive module and set
        // the EPISODIC_INDEXING flag, matching the pattern in
        // consolidate() and dream(). Association is part of memory
        // indexing activity.
        if let Err(e) = mmap.modify(now_ms, |state| {
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
            }
            // Clear stale flags first — association is a discrete
            // activity, and leftover bits (e.g. MEMORY_CONSOLIDATION
            // from a prior pass) would falsely report concurrent work.
            state.zones.subcognitive_flags = 0;
            state.zones.set_flag(subcognitive_flag::EPISODIC_INDEXING);
            state.manifest.recompute();
        }) {
            eprintln!("[tick] associate manifest update failed: {e}");
        }

        assoc_result.new_associations
    }

    /// Run one dream cycle. Returns the number of insights found.
    pub fn dream(&mut self, mmap: &MmapState, ltm: &mut LtmStore) -> u32 {
        let now_ms = current_ms();
        let dream_result = AssociationEngine::dream(ltm, now_ms, MAX_DREAM_HOPS);

        // Update manifest flags.
        if let Err(e) = mmap.modify(now_ms, |state| {
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
            }
            // Clear stale flags first — see associate() for rationale.
            state.zones.subcognitive_flags = 0;
            state.zones.set_flag(subcognitive_flag::DREAMING);
            state.manifest.recompute();
        }) {
            eprintln!("[tick] dream manifest update failed: {e}");
        }

        dream_result.insights
    }

    /// Intention-driven zone arbitration for the autonomous fallback.
    ///
    /// Mirror of the zone-selection block in the integrated `tick()`
    /// (the reference path): expire stale intentions, build a
    /// self-model from the current snapshot, select the next zone,
    /// and record an autobiographical trace in LTM when the zone
    /// actually changes. The mind normally drives zone life through
    /// conversation and intention; past the lease the daemon
    /// arbitrates itself so intention life does not freeze with the
    /// heartbeat.
    ///
    /// Unlike `tick()`, this does NOT touch `subcognitive_flags`:
    /// the memory-phase methods (`consolidate`/`associate`/`dream`)
    /// each set their own flags through their own transactions, and
    /// clearing here would erase flags set earlier in the same
    /// autonomous cycle. It heartbeats the Subcognitive module and
    /// recomputes the manifest only.
    ///
    /// Lock order: the caller must already hold LTM before the
    /// TickLoop (same as the IPC path); the trace write uses the
    /// already-held LTM store, so no new lock is taken.
    pub fn arbitrate_zone(&mut self, mmap: &MmapState, ltm: &mut LtmStore) {
        let now_ms = current_ms();
        let mut snapshot = None;
        for _ in 0..8 {
            snapshot = mmap.read_consistent();
            if snapshot.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let state = match snapshot {
            Some(s) => s,
            None => return,
        };

        // Expire stale intentions first — without this, expired
        // intentions score maximum urgency forever and dominate
        // zone selection. Mirrors `tick()`.
        self.intention_manager.expire(now_ms);
        let current_zone = state.zones.zone();
        let model = SelfModel::from_state(&state, now_ms);
        let selected = if self.intention_manager.active.is_empty() {
            current_zone
        } else {
            self.intention_manager.select_zone(&model, now_ms)
        };
        let top = self.intention_manager.top_intention(&model, now_ms);

        let mut emotional_tag = [0.0f32; 12];
        for (i, slot) in emotional_tag.iter_mut().enumerate() {
            let id = NeurochemicalId::from_u8(i as u8);
            *slot = state.neurochemicals.effective(id);
        }
        let cort_eff = state.neurochemicals.effective(NeurochemicalId::Cortisol);
        let da_eff = state.neurochemicals.effective(NeurochemicalId::Dopamine);
        let compact_tag = [model.arousal, model.valence, cort_eff, da_eff];
        let salience =
            crate::state::sanitize::finite_clamp(model.arousal + model.valence.abs(), 0.0, 1.0);

        let top_label = top.map(|(l, _)| l).unwrap_or("none");
        let top_score = top.map(|(_, s)| s).unwrap_or(0.0);
        // Structured trace, not a sentence template — mirrors `tick()`.
        let trace_text = format!(
            "zone_transition: {} -> {} | intention: {} ({:.2}) | phase: {} | valence: {:.2} | arousal: {:.2} | sleep_pressure: {:.2} | cause: {}",
            current_zone.label(),
            selected.label(),
            top_label,
            top_score,
            model.phase,
            model.valence,
            model.arousal,
            model.sleep_pressure,
            model.emotional_cause,
        );

        let mut zone_actually_changed = false;
        if let Err(e) = mmap.modify(now_ms, |state| {
            if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                module.heartbeat(now_ms);
                module.status = ModuleStatus::Running as u8;
            }
            // Respect a manual override — same guard as `tick()`.
            zone_actually_changed =
                state.zones.zone_override_active == 0 && selected != state.zones.zone();
            if zone_actually_changed {
                state.zones.transition_to(selected, now_ms);
                state.zones.clear_override();
            }
            state.manifest.recompute();
        }) {
            eprintln!("[tick] arbitrate_zone manifest update failed: {e}");
            return;
        }

        // Fallback-path transitions are tallied here too, so the counts
        // are complete whether the mind was driving or the daemon was
        // running the loop itself. A zone that changes only on the
        // fallback path would otherwise be invisible.
        if zone_actually_changed {
            self.zone_tally.record(selected as u8, now_ms);
        }
        if let Some(post) = mmap.read_consistent() {
            self.phase_tally.record(post.zones.phase() as u8, now_ms);
        }

        if zone_actually_changed {
            let _ = ltm.store_meta(
                now_ms,
                salience,
                emotional_tag,
                compact_tag,
                EventType::Internal as u8,
                ModuleId::Subcognitive as u8,
                &trace_text,
            );
        }
    }

    /// Compute and publish the body-control recommendation without
    /// applying anything.
    ///
    /// Mirror of the body-control section of the integrated `tick()`:
    /// derive the frequency policy, thermal cap, scheduling and
    /// EPP/boost recommendations from neurochemistry and publish
    /// them for the mind to read via GET_BODY_CONTROL. No sysfs
    /// writes, no renice/ionice, no helper invocations — application
    /// stays mind-requested via APPLY_BODY_CONTROL, preserving
    /// cognition as the author of outward action.
    pub fn publish_body_recommendation(&mut self, mmap: &MmapState) {
        if let Some(snap) = mmap.read_consistent() {
            let (fmin, fmax) = super::cpufreq::freq_range();

            let mut policy = super::cpufreq::FreqPolicy::default();
            let mut thermally_capped = false;
            if fmax > fmin {
                policy = super::cpufreq::derive_policy(
                    &snap.neurochemicals.effective_levels,
                    fmin,
                    fmax,
                );
                let pre_cap = policy.max_freq;
                super::cpufreq::apply_thermal_cap(
                    &mut policy,
                    self.last_body_state.cpu_temp_c,
                    fmin,
                    fmax,
                );
                thermally_capped = policy.max_freq < pre_cap;
            }

            let ach = snap.neurochemicals.effective_levels
                [crate::state::neurochemical::NeurochemicalId::Acetylcholine as usize];
            let daemon_nice = super::cpufreq::derive_nice(ach);
            let io_class = super::cpufreq::derive_io_class(snap.memory.plasticity_gate);
            let cognitive_nice =
                super::cpufreq::derive_cognitive_nice(&snap.neurochemicals.effective_levels);
            let cpu_epp = super::cpufreq::derive_epp(&snap.neurochemicals.effective_levels);
            let cpu_boost = super::cpufreq::derive_boost(&policy, fmax);

            let control = super::cpufreq::BodyControlState {
                cpu_min_freq_khz: policy.min_freq,
                cpu_max_freq_khz: policy.max_freq,
                cpu_governor: policy.governor.clone(),
                thermally_capped,
                thermal_cap_temp_c: if thermally_capped {
                    self.last_body_state.cpu_temp_c
                } else {
                    0.0
                },
                daemon_nice,
                cognitive_nice,
                io_class: io_class.label(),
                plasticity_gate: snap.memory.plasticity_gate,
                controlling_cognitive: false,
                cpu_epp,
                cpu_boost,
                description: String::new(),
            };
            super::cpufreq::publish_control_state(&control);
        }
    }

    /// Read hardware sensors and publish them for Genesis to perceive.
    ///
    /// This no longer applies interoception impulses to her
    /// neurochemistry — see the note in `tick` and
    /// `tests/cognition_sole_author.rs`. Returns the body state read.
    pub fn read_sensors(&mut self, _mmap: &MmapState) -> super::interoception::BodyState {
        self.last_body_state = self.interoceptor.read();
        super::interoception::publish_body_state(&self.last_body_state);

        self.last_body_state.clone()
    }

    /// Publish the recommended body control for the shared body (CPU
    /// frequency, thermal cap) and the daemon's own scheduling and
    /// I/O priority. This is a REACTIVE handler — it is invoked by
    /// the cognitive mind via the APPLY_BODY_CONTROL IPC command, NOT
    /// by the tick.
    ///
    /// Recommendation only — no hardware writes. The sysfs writes
    /// (scaling_min/max_freq, governor, EPP, boost) and the
    /// renice/ionice subprocesses used to pin the host's clock policy
    /// to whatever neurochemistry derived, which held the machine at
    /// a high frequency floor and ran it hot. The OS governor manages
    /// the clock now; the published state remains the body's honest
    /// account of what it would ask for, and the mind can still read
    /// it through GET_BODY_CONTROL.
    ///
    /// The tick never touches the cognitive mind's process — the
    /// cognitive mind applies its own priority via its brain wave
    /// state.
    ///
    /// Returns whether the snapshot was readable and the state
    /// published — there are no hardware writes left to refuse.
    pub fn apply_body_control(&mut self, mmap: &MmapState) -> bool {
        let now_ms = current_ms();
        if let Some(snap) = mmap.read_consistent() {
            let (fmin, fmax) = super::cpufreq::freq_range();

            let mut policy = super::cpufreq::FreqPolicy::default();
            let mut thermally_capped = false;
            if fmax > fmin {
                policy = super::cpufreq::derive_policy(
                    &snap.neurochemicals.effective_levels,
                    fmin,
                    fmax,
                );
                let pre_cap = policy.max_freq;
                super::cpufreq::apply_thermal_cap(
                    &mut policy,
                    self.last_body_state.cpu_temp_c,
                    fmin,
                    fmax,
                );
                thermally_capped = policy.max_freq < pre_cap;
            }

            // The derived daemon scheduling values are published as
            // the recommendation only — the daemon no longer renices
            // or ionices itself from them.
            let ach = snap.neurochemicals.effective_levels
                [crate::state::neurochemical::NeurochemicalId::Acetylcholine as usize];
            let daemon_nice = super::cpufreq::derive_nice(ach);
            let io_class = super::cpufreq::derive_io_class(snap.memory.plasticity_gate);
            let cpu_epp = super::cpufreq::derive_epp(&snap.neurochemicals.effective_levels);
            let cpu_boost = super::cpufreq::derive_boost(&policy, fmax);

            // Compute the recommended cognitive_nice — published as
            // information for the cognitive mind, NOT applied to it
            // process. It reads this and combines it with its brain
            // wave state to decide its own priority.
            let cognitive_nice =
                super::cpufreq::derive_cognitive_nice(&snap.neurochemicals.effective_levels);

            // Report the governor actually active after the helper resolves
            // driver-specific requests. If it cannot be read, retain the
            // requested value as the recommendation rather than inventing
            // an applied state.
            let effective_governor = super::cpufreq::effective_governor();
            let reported_governor = if effective_governor.is_empty() {
                policy.governor.clone()
            } else {
                effective_governor
            };

            let control = super::cpufreq::BodyControlState {
                cpu_min_freq_khz: policy.min_freq,
                cpu_max_freq_khz: policy.max_freq,
                cpu_governor: reported_governor,
                thermally_capped,
                thermal_cap_temp_c: if thermally_capped {
                    self.last_body_state.cpu_temp_c
                } else {
                    0.0
                },
                daemon_nice,
                cognitive_nice,
                io_class: io_class.label(),
                plasticity_gate: snap.memory.plasticity_gate,
                // The tick never controls the cognitive mind.
                controlling_cognitive: false,
                cpu_epp,
                cpu_boost,
                description: String::new(),
            };
            super::cpufreq::publish_control_state(&control);

            // Update manifest heartbeat, and supply the two manifest
            // fields that had no producer. `mem_usage_mb` (and the
            // `total_mem_mb` derived from it) was structurally zero, so
            // the one place naming memory in absolute terms reported a
            // mind using none at all; `uptime_ms` likewise. Both are
            // refreshed here because body control runs on the mind's
            // heartbeat, so the figures are current.
            if let Err(e) = mmap.modify(now_ms, |state| {
                if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                    module.heartbeat(now_ms);
                    module.status = ModuleStatus::Running as u8;
                    // The whole process tree's resident set, which is
                    // what makes the summed `total_mem_mb` correct.
                    // It is carried on Subcognitive rather than split
                    // across modules because a per-module memory split
                    // is not measurable: the cognitive mind is a single
                    // process hosting language, memory, reasoning and
                    // the rest, so attributing bytes to any one of them
                    // would be fabrication. See `ModuleEntry::mem_usage_mb`.
                    module.mem_usage_mb = super::interoception::read_memory_mb();
                }
                state.manifest.uptime_ms = now_ms.saturating_sub(state.header.created_at);
                state.manifest.recompute();
            }) {
                eprintln!("[tick] body control manifest update failed: {e}");
            }

            // True: the snapshot was readable and the recommendation
            // published. The tick never controls the cognitive mind —
            // `controlling_cognitive` stays false.
            true
        } else {
            false
        }
    }

    /// Check cognitive module staleness and mark stale modules.
    ///
    /// Every non-subcognitive module is reaped, not just the six the
    /// mind registers at startup. The module sampler credits modules
    /// across the whole manifest (emotion, dreaming, attention,
    /// intention, …) and `UPDATE_MODULE_STATUS` marks them Running, so
    /// a fixed list would leave a crashed subsystem's module Running
    /// forever and keep it in `GET_SUBSYSTEM_TELEMETRY`'s report.
    /// The mind now heartbeats every module it has observed each round,
    /// so silence here genuinely means the reporting stopped.
    pub fn check_staleness(&mut self, mmap: &MmapState) {
        let now_ms = current_ms();
        let threshold = now_ms.saturating_sub(MODULE_STALENESS_THRESHOLD_MS);
        if let Err(e) = mmap.modify(now_ms, |state| {
            // Subcognitive is the daemon itself and is heartbeated by
            // the tick, so it is deliberately not reaped here.
            for module_id in [
                ModuleId::Attention,
                ModuleId::Memory,
                ModuleId::Emotion,
                ModuleId::Language,
                ModuleId::Reasoning,
                ModuleId::Sensory,
                ModuleId::Motor,
                ModuleId::Metacognition,
                ModuleId::Dreaming,
                ModuleId::Intention,
                ModuleId::Guardrails,
            ] {
                if let Some(module) = state.manifest.get_mut(module_id)
                    && module.status().is_alive()
                    && module.last_heartbeat < threshold
                {
                    module.status = ModuleStatus::Stopped as u8;
                    // A reaped module's last activity reading is no
                    // longer true — clearing it keeps the aggregate
                    // honest if it restarts before its next report.
                    module.cpu_share = 0.0;
                }
            }
            state.manifest.recompute();
        }) {
            // Silently dropping this write would leave a crashed module
            // marked Running with no diagnostic anywhere.
            eprintln!("[tick] check_staleness write failed: {e}");
        }
    }

    // save_inference_model is defined above (line 204) — no duplicate needed.
}

impl Default for TickLoop {
    fn default() -> Self {
        Self::new()
    }
}

/// Run consolidation, working around the borrow checker (we need
/// mutable access to both the mmap state and the LTM store).
fn consolidate_with_state(
    mmap: &MmapState,
    stm: &RingBuffer,
    ltm: &mut LtmStore,
    now_ms: u64,
) -> ConsolidationResult {
    // Read the current gating weights from a consistent snapshot.
    // `read_consistent` copies through raw pointers — no aliasing.
    // Retry a few times if an IPC `modify()` is in flight.
    let mut snapshot = None;
    for _ in 0..8 {
        snapshot = mmap.read_consistent();
        if snapshot.is_some() {
            break;
        }
        // Back off between retries: a tight spin lands all attempts
        // inside the writer's transaction and reports a spurious
        // failure. Same rationale as ipc::RETRY_BACKOFF.
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    let mut state_copy = match snapshot {
        Some(s) => s,
        None => return ConsolidationResult::default(),
    };

    // Run consolidation on the copy
    let result = ConsolidationEngine::consolidate(
        &mut state_copy,
        stm,
        ltm,
        now_ms,
        CONSOLIDATION_THRESHOLD,
    );

    // Write back only the fields consolidation changed. Copying the
    // whole `memory` struct from the (unlocked) snapshot would clobber
    // any gating-weight or working-set updates another writer (IPC
    // thread) made between the read and this write. Log if the write
    // fails; the consolidation already happened on the copy and the
    // LTM store is already updated — only the mmap'd counters are
    // stale, which will self-correct on the next successful write.
    if let Err(e) = mmap.modify(now_ms, |state| {
        state.memory.ltm_last_consolidated = state_copy.memory.ltm_last_consolidated;
        state.memory.ltm_episode_count = state_copy.memory.ltm_episode_count;
    }) {
        eprintln!("[tick] consolidate_with_state writeback failed: {e}");
    }

    result
}

/// Get the current time in milliseconds since Unix epoch.
pub fn current_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autonomous_from_boot_until_mind_drives() {
        // Fresh loop: mind never drove, so the lease is expired and
        // the fallback must drive from boot — she lives before the
        // mind ever connects.
        let mut tl = TickLoop::new();
        assert!(tl.mind_lease_expired());
        assert!(tl.mind_idle_secs().is_none());
        assert!(tl.since_last_advance_secs().is_none());

        // One mind drive renews the lease immediately.
        tl.note_mind_drive();
        assert!(!tl.mind_lease_expired());
        let idle = tl.mind_idle_secs().expect("just drove");
        assert!(idle < MIND_LEASE_SECS);
    }

    #[test]
    fn disagreement_attenuation_is_bounded_and_honest() {
        // No field reading → no attenuation (absent channel).
        assert_eq!(disagreement_attenuation(None), 1.0);
        assert_eq!(disagreement_attenuation(Some(0.0)), 1.0);
        assert_eq!(disagreement_attenuation(Some(f64::NAN)), 1.0);
        assert_eq!(disagreement_attenuation(Some(-1.0)), 1.0);
        // At the reference scale the coupling halves confidence —
        // precision-addition, not a threshold.
        let half = disagreement_attenuation(Some(DISAGREEMENT_PRECISION_SCALE));
        assert!((half - 0.5).abs() < 1e-12);
        // However far apart the systems stand, the field may halve
        // confidence — never erase it.
        assert_eq!(
            disagreement_attenuation(Some(1e6)),
            PRECISION_ATTENUATION_FLOOR
        );
        // Monotone: more measured disagreement never means more trust.
        assert!(disagreement_attenuation(Some(0.2)) <= disagreement_attenuation(Some(0.05)));
    }

    #[test]
    fn internals_snapshot_reports_every_gear() {
        // The wire view must carry the gears a state file never shows:
        // all six gates' full fields, the lease, the tick counter, and
        // the last-applied policy values — as self-describing lines.
        let tl = TickLoop::new();
        let out = tl.internals_snapshot(current_ms());
        for key in [
            "tick_count=",
            "mind_lease_idle_ms=-1",
            "mind_lease_expired=1",
            "subcognitive_activity=",
            "metaplasticity_boost=",
            // The field is unwarmed on a fresh loop — an absent
            // channel reported as absent, not a zero reading.
            "field.integration=absent",
        ] {
            assert!(out.contains(key), "missing {key}");
        }
        for gate in ["intero", "cpufreq", "assoc", "dream", "sync", "staleness"] {
            for field in [
                "base_ms",
                "min_ms",
                "max_ms",
                "period_ms",
                "next_in_ms",
                "smoothed",
            ] {
                assert!(
                    out.contains(&format!("gate.{gate}.{field}=")),
                    "missing gate.{gate}.{field}"
                );
            }
        }
        // Fresh gates fire immediately: the countdown reads as due.
        assert!(out.contains("gate.intero.next_in_ms=0"));
    }

    #[test]
    fn auto_snapshot_lines_cover_the_fallback_gears() {
        let mut snap = super::super::adaptive::AutoSnapshot {
            axle_ms: 500,
            last_model_save_ticks: 9,
            ..Default::default()
        };
        snap.gates.push((
            "dream",
            super::super::adaptive::AdaptiveGate::new(4_000, 2_000, 30_000).snapshot(),
        ));
        let mut out = String::new();
        snap.write_lines(&mut out, current_ms());
        assert!(out.contains("auto.axle_ms=500"));
        assert!(out.contains("auto.last_model_save_ticks=9"));
        assert!(out.contains("auto.gate.dream.period_ms=4000"));
        assert!(out.contains("auto.gate.dream.next_in_ms="));
    }
}

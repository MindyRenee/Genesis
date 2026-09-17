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
//! 6. [`TickLoop::apply_body_control`] — apply the shared body policy
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
use crate::state::neurochemical::{NeuroTickParams, NeurochemicalId};
use crate::state::{CognitiveZone, MentalPhase, ModuleId, ModuleStatus, subcognitive_flag};
use crate::store::ring_buffer::EventType;
use crate::store::{LtmStore, MmapState, RingBuffer};

use super::active_inference::{ActiveInferenceEngine, DyadicSignals, apply_inference_feedback};
use super::association::AssociationEngine;
use super::consolidation::{ConsolidationEngine, ConsolidationResult};
use super::cpufreq::CPUFREQ_INTERVAL_TICKS;
use super::dyadic_model::{DyadicAffectModel, take_user_affect};
use super::interoception::{INTEROCEPTION_INTERVAL_TICKS, Interoceptor};
use super::ipc::LtmAccess;

/// Tick interval: 200ms (5 Hz).
pub const TICK_INTERVAL_MS: u64 = 200;

/// Consolidation threshold: entries with a consolidation score below
/// this are skipped (left to decay in the ring buffer).
pub const CONSOLIDATION_THRESHOLD: f32 = 0.15;

/// How often to run association (every N ticks).
pub const ASSOCIATION_INTERVAL_TICKS: u64 = 50; // ~10 seconds at 5 Hz

/// How often to run dreaming (every N ticks when sleeping).
pub const DREAM_INTERVAL_TICKS: u64 = 20; // ~4 seconds when sleeping at 5 Hz

/// How often to sync to disk (base interval, every N ticks).
/// The actual sync rate is modulated by the plasticity gate:
/// high plasticity → sync at base rate (active learning, persist fast).
/// low plasticity → sync less often (stress impairs memory, I/O is
/// deprioritized, no need to write frequently).
pub const SYNC_INTERVAL_TICKS: u64 = 100; // ~20 seconds base at 5 Hz

/// How often to check cognitive module staleness (every N ticks).
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
    /// Active intention manager used to select the next cognitive zone.
    pub intention_manager: IntentionManager,
    /// Interoceptor — reads hardware state so Genesis can feel her body.
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
    /// Metaplasticity boost multiplier from the last inference cycle.
    /// High surprise → faster coupling-matrix learning on the next
    /// tick. Reset to 1.0 after each application.
    metaplasticity_boost: f32,
    /// Last-applied CPU frequency policy. The reactive
    /// `apply_body_control` handler (cognitive-mind-driven via IPC)
    /// writes to sysfs (scaling_min/max_freq, governor), but
    /// neurochemistry drifts slowly — most cycles produce the same
    /// policy. Skipping the sysfs writes when the policy is unchanged
    /// avoids unnecessary kernel I/O and context switches.
    last_freq_policy: super::cpufreq::FreqPolicy,
    /// Last-applied daemon nice value. `set_priority` spawns a `renice`
    /// subprocess; the reactive handler skips it when the value is
    /// unchanged, avoiding the fork+exec overhead on most cycles.
    last_daemon_nice: i32,
    /// Last-applied daemon I/O class. `set_io_priority` spawns an
    /// `ionice` subprocess; the reactive handler skips it when the
    /// value is unchanged, avoiding the fork+exec overhead on most
    /// cycles.
    last_daemon_io_class: super::cpufreq::IoClass,
    /// Last-applied EPP (Energy Performance Preference) profile.
    /// `set_epp` writes to sysfs via the privileged helper; the
    /// reactive handler skips it when the profile is unchanged.
    /// Empty string = EPP not available or not yet applied.
    last_epp: String,
    /// Last-applied turbo (boost) gate state. `set_boost` writes to
    /// sysfs via the privileged helper; the reactive handler skips it
    /// when the state is unchanged. `Unavailable` = the platform
    /// exposes no gate, or it has not been applied yet.
    last_boost: super::cpufreq::BoostState,
    /// The subcognitive's own activity share — the smoothed fraction
    /// of the tick interval the daemon spends working (neuro tick,
    /// interoception, inference, consolidation, association,
    /// dreaming, housekeeping). Written to the Subcognitive module's
    /// cpu_share each manifest update so GET_LOBE_TELEMETRY's module
    /// section covers her daemon-side brain part too — the Python
    /// sampler can't see inside this process, so the daemon measures
    /// itself. EMA-smoothed (0.7/0.3) to avoid per-tick flicker.
    subcognitive_activity: f32,
    /// Start time of the previous `advance_neuro` call. In production
    /// the daemon runs reactively — there is no fixed tick interval —
    /// so the work fraction is measured against the observed
    /// call-to-call interval instead. `None` until the second call.
    subcognitive_last_call: Option<std::time::Instant>,
}

impl TickLoop {
    /// Create a fresh tick loop with default subsystems.
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            intention_manager: IntentionManager::new(4),
            interoceptor: Interoceptor::new(),
            last_body_state: super::interoception::BodyState::neutral(),
            inference_engine: ActiveInferenceEngine::new(),
            dyadic_model: DyadicAffectModel::new(),
            metaplasticity_boost: 1.0,
            last_freq_policy: super::cpufreq::FreqPolicy::default(),
            last_daemon_nice: 0,
            last_daemon_io_class: super::cpufreq::IoClass::BestEffort(3),
            last_epp: String::new(),
            last_boost: super::cpufreq::BoostState::Unavailable,
            subcognitive_activity: 0.0,
            subcognitive_last_call: None,
        }
    }

    /// Create a TickLoop that loads the persisted active inference model
    /// from `data_dir/inference_model.bin`. If the file doesn't exist
    /// (first run) or is corrupt, a fresh engine is used.
    ///
    /// This allows the generative self-model to survive daemon restarts
    /// — Genesis doesn't re-learn her own neurochemical dynamics from
    /// scratch every time she wakes up.
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
            intention_manager: IntentionManager::new(4),
            interoceptor: Interoceptor::new_with_data_dir(data_dir),
            last_body_state: super::interoception::BodyState::neutral(),
            inference_engine,
            dyadic_model: DyadicAffectModel::new(),
            metaplasticity_boost: 1.0,
            last_freq_policy: super::cpufreq::FreqPolicy::default(),
            last_daemon_nice: 0,
            last_daemon_io_class: super::cpufreq::IoClass::BestEffort(3),
            last_epp: String::new(),
            last_boost: super::cpufreq::BoostState::Unavailable,
            subcognitive_activity: 0.0,
            subcognitive_last_call: None,
        }
    }

    /// Save the active inference model to `data_dir/inference_model.bin`.
    ///
    /// Called periodically by the daemon (every N ticks) and on shutdown
    /// so the generative self-model persists across restarts.
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
            p == MentalPhase::NREM
                || p == MentalPhase::REM
                || s.zones.zone() == CognitiveZone::Sleeping
        });

        let neuro_params = if is_sleeping_pre {
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
        let mut neuro_params = neuro_params;
        neuro_params.dt = (TICK_INTERVAL_MS as f32) / 1000.0;

        // Compute HPA axis maturation from accumulated experience.
        // The stress hyporesponsive period (SHRP) keeps cortisol at
        // zero during early development, protecting plasticity. The
        // HPA axis gradually comes online as she accumulates
        // episodic experience (ltm_episode_count), reaching full
        // maturity at ~10,000 episodes.
        let ltm_count = pre_snapshot
            .as_ref()
            .map(|s| s.memory.ltm_episode_count)
            .unwrap_or(0);
        let maturation_level =
            crate::state::sanitize::finite_clamp(ltm_count as f32 / 10_000.0, 0.0, 1.0);
        neuro_params.maturation_level = maturation_level;
        neuro_params.noise_seed = self.tick_count;
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
        //     feels her own body (CPU temperature, memory pressure,
        //     load, battery, I/O wait). The impulses are small and
        //     accumulate over multiple ticks through the coupled
        //     dynamics, creating a gradual physiological response.
        if self.tick_count.is_multiple_of(INTEROCEPTION_INTERVAL_TICKS) {
            self.last_body_state = self.interoceptor.read();
            super::interoception::publish_body_state(&self.last_body_state);
        }
        let body_impulses = self.interoceptor.neuro_impulses(&self.last_body_state);
        if !body_impulses.is_empty() {
            if let Err(e) = mmap.modify(now_ms, |state| {
                for (chem_id, amount) in &body_impulses {
                    state
                        .neurochemicals
                        .apply_impulse_capped(*chem_id, *amount, now_ms);
                }
                state.neurochemicals.recompute_derived();
                state.sync_neurochemistry_to_state();
            }) {
                if matches!(e, crate::store::StateFileError::FileLockBusy) {
                    eprintln!("[tick] interoception impulse skipped (state locked): {e}");
                } else {
                    panic!("interoception impulse failed: {e}");
                }
            }
        }

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

        if let Err(e) = mmap.modify(now_ms, |state| {
            // Apply inference feedback (impulses, cortisol baseline
            // adjustment). Metaplasticity boost is applied on the
            // next tick via self.metaplasticity_boost.
            apply_inference_feedback(&mut state.neurochemicals, &inference_result, now_ms);

            // Apply dyadic model impulses (oxytocin bonding, empathic
            // cortisol). chem_id comes from the dyadic model, not IPC —
            // validate it so a corrupt model can't silently pump
            // dopamine via the from_u8 catch-all.
            for &(chem_id, magnitude) in &dyadic_impulses {
                if chem_id as usize >= crate::state::neurochemical::NEUROCHEMICAL_COUNT {
                    continue;
                }
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
        }) {
            if matches!(e, crate::store::StateFileError::FileLockBusy) {
                eprintln!("[tick] inference feedback skipped (state locked): {e}");
            } else {
                panic!("active inference feedback failed: {e}");
            }
        }

        // 2e. Body control — neurochemistry suggests hardware settings.
        //     Dopamine speeds her up, GABA slows her down, melatonin
        //     puts her in powersave mode. ACh sets daemon scheduling
        //     priority. CPU temperature caps max frequency (autonomic
        //     fatigue). This is the reverse of interoception: her
        //     brain state shapes her body.
        //
        //     The tick COMPUTES and PUBLISHES the recommended body
        //     control state (CPU frequency, thermal cap, daemon
        //     scheduling, cognitive_nice, io_class) as interoceptive
        //     afferent information. The tick NEVER applies control —
        //     it is a display/information sink, not a controller. The
        //     cognitive mind reads the recommendation via
        //     GET_BODY_CONTROL and requests application via
        //     APPLY_BODY_CONTROL when she decides to act on it. The
        //     tick never touches the cognitive mind's PID.
        if self.tick_count.is_multiple_of(CPUFREQ_INTERVAL_TICKS) {
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
                // she decides to act on the recommendation.
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
                    // regardless of dopamine. She can't be highly aroused
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
                // with her brain wave state to decide what she
                // actually applies to herself. The tick does not
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
                let cpu_epp =
                    super::cpufreq::derive_epp(&snap.neurochemicals.effective_levels);

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
            std::hint::spin_loop();
        }
        let state = match snapshot {
            Some(s) => s,
            None => return result, // Write still in progress after retries
        };

        result.phase = state.zones.emergent_phase;
        let phase = state.zones.phase();

        // 3. Determine what work to do based on emotional state
        let can_consolidate = state.memory.can_form_new_memories();
        let is_sleeping = phase == MentalPhase::NREM
            || phase == MentalPhase::REM
            || state.zones.zone() == CognitiveZone::Sleeping;
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

        // 4. Association (run periodically, skip if overwhelmed)
        //    LTM lock held only for this phase.
        if !is_overwhelmed && self.tick_count.is_multiple_of(ASSOCIATION_INTERVAL_TICKS) {
            let mut ltm_guard = ltm.access();
            // Find associations for the most recently stored episodes.
            // We scan the index for the highest episode IDs (most recent)
            // rather than assuming sequential IDs, which breaks on deletion.
            let recent_ids = ltm_guard.recent_episode_ids(5);
            let assoc_result = AssociationEngine::associate(&mut ltm_guard, &recent_ids, now_ms);
            result.associations = assoc_result.new_associations;
        }

        // 5. Dreaming (only when sleeping)
        //    LTM lock held only for this phase.
        let mut did_dream = false;
        if is_sleeping && self.tick_count.is_multiple_of(DREAM_INTERVAL_TICKS) {
            let mut ltm_guard = ltm.access();
            let dream_result = AssociationEngine::dream(&mut ltm_guard, now_ms, MAX_DREAM_HOPS);
            result.dream_insights = dream_result.insights;
            did_dream = true;
        }

        // 6. Housekeeping — sync to disk periodically.
        //    Sync rate is modulated by the plasticity gate:
        //    high plasticity → sync every base interval (active learning).
        //    low plasticity → sync every 3rd interval (stress, less to write).
        //    very low plasticity → sync every 5th interval (minimal I/O).
        //    This works with the I/O priority control: when plasticity is
        //    low, both the frequency and priority of disk writes drop.
        if self.tick_count.is_multiple_of(SYNC_INTERVAL_TICKS) {
            let plasticity = state.memory.plasticity_gate;
            let sync_cycle = self.tick_count / SYNC_INTERVAL_TICKS;
            let should_sync = if plasticity < 0.1 {
                sync_cycle.is_multiple_of(5) // ~100 seconds at 5 Hz
            } else if plasticity < 0.4 {
                sync_cycle.is_multiple_of(3) // ~60 seconds at 5 Hz
            } else {
                true // ~20 seconds (base rate at 5 Hz)
            };
            if should_sync {
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
                if let Err(e) = mmap.sync() {
                    eprintln!("[tick] mmap sync failed: {e}");
                    all_synced = false;
                }
                result.synced = all_synced;
            }
        }

        // 7. Update the manifest — heartbeat the subcognitive module
        //    and check cognitive modules for staleness.
        //    No LTM lock needed.
        let check_staleness = self
            .tick_count
            .is_multiple_of(STALENESS_CHECK_INTERVAL_TICKS);

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

    /// Total ticks run.
    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    // ─── Reactive methods (mind-driven, not tick-driven) ────────
    //
    // Each of these performs one specific function that was
    // previously part of the fixed tick loop. The cognitive mind
    // calls them through IPC when internal state says it's time.
    // The daemon never initiates these — it only executes them on
    // request. All functions are idle until activated.

    /// Advance neurochemical dynamics by dt, run active inference,
    /// and update the dyadic model. This is the core "physics
    /// integration" step — it advances the coupled differential
    /// equations that govern her neurochemistry, then runs the
    /// generative self-model to predict, compare, and feed back.
    ///
    /// Returns (surprise, free_energy, precision, allostatic_load,
    /// tick_count) so the mind can observe the result.
    pub fn advance_neuro(&mut self, mmap: &MmapState, dt: f32) -> (f32, f32, f32, f32, u32) {
        let call_start = std::time::Instant::now();
        let call_interval_s = self
            .subcognitive_last_call
            .map(|prev| call_start.duration_since(prev).as_secs_f32());
        self.subcognitive_last_call = Some(call_start);
        self.tick_count += 1;
        let now_ms = current_ms();

        // Read pre-tick state for sleep-tuned parameters.
        let pre_snapshot = mmap.read_consistent();
        let is_sleeping_pre = pre_snapshot.as_ref().is_some_and(|s| {
            let p = s.zones.phase();
            p == MentalPhase::NREM
                || p == MentalPhase::REM
                || s.zones.zone() == CognitiveZone::Sleeping
        });

        let mut neuro_params = if is_sleeping_pre {
            let mut p = NeuroTickParams::DEFAULT;
            p.bdnf_recovery_rate *= SLEEP_BDNF_RECOVERY_MULTIPLIER;
            p.receptor_resensitization_rate = SLEEP_RECEPTOR_RESENSITIZATION_RATE;
            p
        } else {
            NeuroTickParams::DEFAULT
        };
        neuro_params.dt = dt;

        // HPA maturation from accumulated experience.
        let ltm_count = pre_snapshot
            .as_ref()
            .map(|s| s.memory.ltm_episode_count)
            .unwrap_or(0);
        neuro_params.maturation_level =
            crate::state::sanitize::finite_clamp(ltm_count as f32 / 10_000.0, 0.0, 1.0);
        neuro_params.noise_seed = self.tick_count;

        // Metaplasticity boost from last inference cycle.
        if self.metaplasticity_boost > 1.0 {
            neuro_params.metaplasticity_rate *= self.metaplasticity_boost;
            neuro_params.metaplasticity_rate = neuro_params.metaplasticity_rate.min(0.0001);
        }
        self.metaplasticity_boost = 1.0;

        // Advance neurochemistry.
        if let Err(e) = mmap.modify(now_ms, |state| {
            state.neuro_tick_with_params(&neuro_params);
        }) {
            eprintln!("[tick] advance_neuro neuro tick failed: {e}");
            // Neurochemistry didn't advance — return zeros so the
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
            self.subcognitive_activity =
                0.7 * self.subcognitive_activity + 0.3 * work_frac;
        }
        let subcognitive_share = self.subcognitive_activity;

        // Apply inference feedback + dyadic impulses, and heartbeat
        // the subcognitive module in the same transaction. This merges
        // two formerly-separate modify() calls into one, halving the
        // seqlock write windows on the most frequently called reactive
        // command (~1/s). The manifest heartbeat touches
        // state.manifest while the inference feedback touches
        // state.neurochemicals and state.inference_signals — disjoint
        // fields, so coalescing is safe. If the modify fails, the
        // inference cycle already ran (just couldn't write back), and
        // the next advance_neuro will retry the heartbeat.
        if let Err(e) = mmap.modify(now_ms, |state| {
            apply_inference_feedback(&mut state.neurochemicals, &inference_result, now_ms);
            for &(chem_id, magnitude) in &dyadic_impulses {
                if chem_id as usize >= crate::state::neurochemical::NEUROCHEMICAL_COUNT {
                    continue;
                }
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

        (
            inference_result.surprise,
            inference_result.free_energy,
            inference_result.precision,
            inference_result.allostasis_load,
            self.inference_engine.tick_count(),
        )
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
            state.zones.set_flag(subcognitive_flag::DREAMING);
            state.manifest.recompute();
        }) {
            eprintln!("[tick] dream manifest update failed: {e}");
        }

        dream_result.insights
    }

    /// Read hardware sensors and apply interoception impulses to
    /// neurochemistry. Returns the body state that was read.
    pub fn read_sensors(&mut self, mmap: &MmapState) -> super::interoception::BodyState {
        let now_ms = current_ms();
        self.last_body_state = self.interoceptor.read();
        super::interoception::publish_body_state(&self.last_body_state);

        let body_impulses = self.interoceptor.neuro_impulses(&self.last_body_state);
        if !body_impulses.is_empty()
            && let Err(e) = mmap.modify(now_ms, |state| {
                for (chem_id, amount) in &body_impulses {
                    state
                        .neurochemicals
                        .apply_impulse_capped(*chem_id, *amount, now_ms);
                }
                state.neurochemicals.recompute_derived();
                state.sync_neurochemistry_to_state();
            })
        {
            eprintln!("[tick] interoception impulse failed: {e}");
        }

        self.last_body_state.clone()
    }

    /// Apply the recommended body control to the shared body (CPU
    /// frequency, thermal cap) and the daemon's own process scheduling
    /// and I/O priority. This is a REACTIVE handler — it is invoked by
    /// the cognitive mind via the APPLY_BODY_CONTROL IPC command, NOT
    /// by the tick. The tick only computes and publishes the
    /// recommendation; the cognitive mind decides when to request
    /// application.
    ///
    /// Uses change detection to skip redundant sysfs writes and
    /// renice/ionice subprocess spawns when the recommended values
    /// haven't changed since the last call.
    ///
    /// The tick never touches the cognitive mind's process — the
    /// cognitive mind applies her own priority via her brain wave
    /// state. This handler only applies the shared body control and
    /// the daemon's own scheduling.
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
                // Only write to sysfs when the policy changed — see
                // the tick path for rationale.
                if policy != self.last_freq_policy {
                    super::cpufreq::apply_policy(&policy);
                    self.last_freq_policy = policy.clone();
                }
            }

            // Daemon controls its own process scheduling and I/O
            // priority — autonomic self-regulation.
            let ach = snap.neurochemicals.effective_levels
                [crate::state::neurochemical::NeurochemicalId::Acetylcholine as usize];
            let daemon_nice = super::cpufreq::derive_nice(ach);
            let pid = std::process::id();
            if daemon_nice != self.last_daemon_nice {
                super::cpufreq::set_priority(pid, daemon_nice);
                self.last_daemon_nice = daemon_nice;
            }

            let io_class = super::cpufreq::derive_io_class(snap.memory.plasticity_gate);
            if io_class != self.last_daemon_io_class {
                super::cpufreq::set_io_priority(pid, io_class);
                self.last_daemon_io_class = io_class;
            }

            // EPP (Energy Performance Preference) — the hardware's
            // voltage/frequency operating point hint. Applied to
            // sysfs when the profile changes. On acpi-cpufreq
            // systems (no EPP), derive_epp returns an empty string
            // and set_epp is a no-op.
            let cpu_epp =
                super::cpufreq::derive_epp(&snap.neurochemicals.effective_levels);
            if cpu_epp != self.last_epp {
                super::cpufreq::set_epp(&cpu_epp);
                self.last_epp = cpu_epp.clone();
            }

            // Turbo (boost) gate — the hardware permission to run
            // above the base P-state. Tied to the (thermal-capped)
            // frequency policy: enabled only when the policy ceiling is
            // already at the hardware max, so it can never override a
            // throttle. On platforms with no gate, derive_boost returns
            // Unavailable and set_boost is a no-op.
            let cpu_boost = super::cpufreq::derive_boost(&policy, fmax);
            if cpu_boost != self.last_boost {
                if let Some(enabled) = cpu_boost.as_bool() {
                    super::cpufreq::set_boost(enabled);
                }
                self.last_boost = cpu_boost;
            }

            // Compute the recommended cognitive_nice — published as
            // information for the cognitive mind, NOT applied to her
            // process. She reads this and combines it with her brain
            // wave state to decide her own priority.
            let cognitive_nice =
                super::cpufreq::derive_cognitive_nice(&snap.neurochemicals.effective_levels);

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
                controlling_cognitive: false,
                cpu_epp,
                cpu_boost,
                description: String::new(),
            };
            super::cpufreq::publish_control_state(&control);

            // Update manifest heartbeat.
            if let Err(e) = mmap.modify(now_ms, |state| {
                if let Some(module) = state.manifest.get_mut(ModuleId::Subcognitive) {
                    module.heartbeat(now_ms);
                    module.status = ModuleStatus::Running as u8;
                }
                state.manifest.recompute();
            }) {
                eprintln!("[tick] body control manifest update failed: {e}");
            }

            // Always false — the tick no longer controls the
            // cognitive mind. The return value is kept for API
            // compatibility.
            false
        } else {
            false
        }
    }

    /// Check cognitive module staleness and mark stale modules.
    pub fn check_staleness(&mut self, mmap: &MmapState) {
        let now_ms = current_ms();
        let threshold = now_ms.saturating_sub(MODULE_STALENESS_THRESHOLD_MS);
        let _ = mmap.modify(now_ms, |state| {
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
            state.manifest.recompute();
        });
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
        std::hint::spin_loop();
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

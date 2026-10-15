//! Sleep-wake regulatory network: wake, NREM, and REM populations with
//! a homeostatic sleep drive and a circadian oscillator.
//!
//! Transcribed from the MATLAB source published with ModelDB entry 244485
//! (`DinizBehnBooth_SWnetworkModel.m`, `run_DinizBehnBooth_SWnetworkModel.m`).
//! That implements Gleit, Diniz Behn & Booth, J. Biol. Rhythms 28:339–355
//! (2013), as revised in Booth, Xique & Diniz Behn, SIAM J. Appl. Dyn. Sys.
//! 16(2):1089–1112 (2017).
//!
//! Populations: `f_lc` (wake, locus coeruleus), `f_vlpo` (NREM / sleep,
//! ventrolateral preoptic area), `f_r` (REM-on), `f_scn` (circadian pacemaker).
//! Sleep onset is `f_lc` falling below [`SLEEP_THRESHOLD`]; REM onset is
//! `f_r` rising above [`REM_THRESHOLD`]. Both are the source's own event
//! definitions. Time is in minutes.

use std::f64::consts::PI;

// Synaptic gains between the neurotransmitter pools and the populations.
const GACH_LC: f64 = 0.8;
const GGABA_LC: f64 = 1.5;
const GSCN_LC: f64 = 0.5;
const GNE_VLPO: f64 = 1.5;
const GSCN_VLPO: f64 = 0.1;
const GACH_R: f64 = 2.2;
const GNE_R: f64 = 8.0;
const GGABA_R: f64 = 1.0;
const GSCN_R: f64 = 0.8;

// Population time constants (min) and saturating firing rates (Hz).
const TAU_LC: f64 = 23.0;
const TAU_R: f64 = 1.0;
const TAU_VLPO: f64 = 10.0;
const TAU_SCN: f64 = 0.5;
const MAX_LC: f64 = 6.0;
const MAX_R: f64 = 5.0;
const MAX_VLPO: f64 = 5.0;
const MAX_SCN: f64 = 7.0;

// Neurotransmitter saturation scales.
const GAMMA_ACH_R: f64 = 2.5;
const GAMMA_NE: f64 = 5.0;
const GAMMA_GABA: f64 = 2.5;
const GAMMA_SCN: f64 = 4.0;

// Sigmoid thresholds and widths of the population response functions.
const BETA_LC: f64 = -0.4;
const ALPHA_LC: f64 = 0.4;
/// −0.3 gives no REM; −0.8 gives four REM bouts per sleep episode.
const BETA_R: f64 = -0.8;
const ALPHA_R: f64 = 0.1;
const ALPHA_VLPO: f64 = 0.2;
const K1: f64 = -0.1;
const K2: f64 = -0.0045;
const BETA_SCN: f64 = -0.1;
const ALPHA_SCN: f64 = 0.7;

// Homeostatic sleep drive.
const THETA_W: f64 = 2.0;
const TAU_HW: f64 = 946.8;
const TAU_HS: f64 = 202.2;
const H_MAX: f64 = 323.88;
const H_MIN: f64 = 0.0;

// Circadian oscillator (Serkh & Forger 2014 corrections).
const TAU_X: f64 = 24.2;
const ALPH_0: f64 = 0.05;
const BETA: f64 = 0.0075;
const G: f64 = 33.75;
const P: f64 = 0.5;
const K: f64 = 0.55;
const I_0: f64 = 9500.0;
const MU: f64 = 0.23;

/// Firing rate of LC (wake) below which the source counts sleep onset.
pub const SLEEP_THRESHOLD: f64 = 4.0;
/// Firing rate of REM-on above which the source counts REM onset.
pub const REM_THRESHOLD: f64 = 3.0;

/// State vector: `[f_lc, f_vlpo, f_r, f_scn, h, c, xc, n]`.
pub type State = [f64; 8];

/// The source's initial conditions (`run_DinizBehnBooth_SWnetworkModel.m`).
pub const INITIAL: State = [
    5.71387902867966,
    0.395493299109024,
    0.0,
    4.01351469432799,
    214.116748078719,
    0.00114912794709424,
    -1.15530846944936,
    0.604617443838918,
];

fn sigmoid(x: f64, max: f64, beta: f64, alpha: f64) -> f64 {
    max * 0.5 * (1.0 + ((x - beta) / alpha).tanh())
}

/// A volitional input to the network: an excitatory drive onto one of the
/// two state-switching populations. Voluntary sleep onset is modelled as
/// drive onto VLPO, voluntary arousal as drive onto LC. The source has no
/// such input; it is added here because Genesis's `/sleep` and `/wake` are
/// volitional. Both drives are zero unless the mind sets them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Drive {
    /// Excitation of VLPO (sleep-promoting), in the same units as the
    /// synaptic inputs to the sigmoid.
    pub vlpo: f64,
    /// Excitation of LC (wake-promoting).
    pub lc: f64,
}

impl Drive {
    /// No volitional input: the source model unchanged.
    pub const NONE: Drive = Drive { vlpo: 0.0, lc: 0.0 };
}

/// Right-hand side of the network at time `t` (minutes), with no volitional
/// input.
pub fn deriv(t: f64, x: &State) -> State {
    deriv_with(t, x, Drive::NONE)
}

/// Right-hand side of the network at time `t` (minutes), with a volitional
/// drive added to the population inputs.
pub fn deriv_with(t: f64, x: &State, drive: Drive) -> State {
    let [f_lc, f_vlpo, f_r, f_scn, h, c, xc, n] = *x;
    let c_ne = (f_lc / GAMMA_NE).tanh();
    let c_gaba = (f_vlpo / GAMMA_GABA).tanh();
    let c_ach = (f_r / GAMMA_ACH_R).tanh();
    let c_scn = (f_scn / GAMMA_SCN).tanh();

    let lc_input = GACH_LC * c_ach + GSCN_LC * c_scn - GGABA_LC * c_gaba + drive.lc;
    let vlpo_input = -GNE_VLPO * c_ne - GSCN_VLPO * c_scn + drive.vlpo;
    let r_input = GACH_R * c_ach - GNE_R * c_ne + GSCN_R * c_scn - GGABA_R * c_gaba;
    let beta_vlpo = K2 * h + K1;

    let lc_inf = sigmoid(lc_input, MAX_LC, BETA_LC, ALPHA_LC);
    let r_inf = sigmoid(r_input, MAX_R, BETA_R, ALPHA_R);
    let scn_inf = sigmoid(c, MAX_SCN, BETA_SCN, ALPHA_SCN);
    let vlpo_inf = sigmoid(vlpo_input, MAX_VLPO, beta_vlpo, ALPHA_VLPO);

    // Light: 500 during the 14 h light period, else 0. The source runs
    // light from minute 600 of each simulated day (10:00–24:00); here t is
    // anchored to the local wall clock, so the window is phase-shifted to
    // 07:00–21:00 — a 14 h photoperiod, preserved — so that the entrained
    // sleep episode (~22:11–05:50) falls inside the neurochemical
    // circadian's biological night rather than ending ~2 h after
    // melatonin offset as it did with the source's alignment.
    let i_light: f64 = if (420.0..1260.0).contains(&t.rem_euclid(1440.0)) {
        500.0
    } else {
        0.0
    };
    let alph = ALPH_0 * (i_light.powf(P) / I_0.powf(P));
    let b_hat = G * (1.0 - n) * alph;
    let b = b_hat * (1.0 - 0.4 * c) * (1.0 - 0.4 * xc);

    let dh = if f_lc - THETA_W >= 0.0 {
        (H_MAX - h) / TAU_HW
    } else {
        (H_MIN - h) / TAU_HS
    };
    let w = (24.0 / (0.99669 * TAU_X)).powi(2);
    [
        (lc_inf - f_lc) / TAU_LC,
        (vlpo_inf - f_vlpo) / TAU_VLPO,
        (r_inf - f_r) / TAU_R,
        (scn_inf - f_scn) / TAU_SCN,
        dh,
        (PI / 720.0) * (xc + b),
        (PI / 720.0) * (MU * (xc - 4.0 * xc.powi(3) / 3.0) - c * (w + K * b)),
        alph * (1.0 - n) - BETA * n,
    ]
}

/// One fourth-order Runge–Kutta step of `dt` minutes from time `t`, with a
/// volitional drive held constant over the step.
pub fn rk4_with(t: f64, x: &State, dt: f64, drive: Drive) -> State {
    let add = |a: &State, b: &State, k: f64| -> State { std::array::from_fn(|i| a[i] + k * b[i]) };
    let k1 = deriv_with(t, x, drive);
    let k2 = deriv_with(t + dt / 2.0, &add(x, &k1, dt / 2.0), drive);
    let k3 = deriv_with(t + dt / 2.0, &add(x, &k2, dt / 2.0), drive);
    let k4 = deriv_with(t + dt, &add(x, &k3, dt), drive);
    std::array::from_fn(|i| x[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

/// One RK4 step with no volitional input.
pub fn rk4(t: f64, x: &State, dt: f64) -> State {
    rk4_with(t, x, dt, Drive::NONE)
}

/// Advance the network by `dt_min` minutes from `t_min` under a volitional
/// drive.
///
/// A non-finite result keeps the previous state. The network is authoritative
/// and persistent, so one bad step must not be allowed to poison it.
pub fn advance_with(t_min: f64, x: &State, dt_min: f64, drive: Drive) -> State {
    let next = rk4_with(t_min, x, dt_min, drive);
    if next.iter().all(|v| v.is_finite()) {
        next
    } else {
        *x
    }
}

/// Advance the network by `dt_min` minutes with no volitional input.
pub fn advance(t_min: f64, x: &State, dt_min: f64) -> State {
    advance_with(t_min, x, dt_min, Drive::NONE)
}

/// The populations that drive the chemistry, each normalized to its
/// saturating maximum. LC drives wake-related chemistry; REM-on drives REM.
pub fn levels(x: &State) -> crate::state::neurochemical::NetworkLevels {
    use crate::state::sanitize::finite_clamp;
    crate::state::neurochemical::NetworkLevels {
        lc: finite_clamp((x[0] / MAX_LC) as f32, 0.0, 1.0),
        vlpo: finite_clamp((x[1] / MAX_VLPO) as f32, 0.0, 1.0),
        rem: finite_clamp((x[2] / MAX_R) as f32, 0.0, 1.0),
    }
}

/// Whether the network is in a sleep state at these normalized
/// population levels — LC firing below the source's 4 Hz sleep-onset
/// threshold. The levels are normalized by each population's maximum
/// rate, so the threshold is `SLEEP_THRESHOLD / MAX_LC`.
pub fn levels_asleep(l: crate::state::neurochemical::NetworkLevels) -> bool {
    f64::from(l.lc) * MAX_LC < SLEEP_THRESHOLD
}

/// The sleep stage the network implies, by the source's own definitions:
/// wake while LC fires at or above [`SLEEP_THRESHOLD`]; otherwise REM when
/// REM-on fires above [`REM_THRESHOLD`], and NREM when it does not.
///
/// While awake, the stage is whatever waking phase the chemistry already
/// names (Active, Alert, Flow, Stress, Drowsy, Overwhelmed). A sleep
/// stage left over from the previous tick becomes Active, because the
/// network, not the chemistry, now decides when sleep ends.
pub fn stage(
    x: &State,
    awake: crate::state::zones::MentalPhase,
) -> crate::state::zones::MentalPhase {
    use crate::state::zones::MentalPhase;
    if x[0] >= SLEEP_THRESHOLD {
        match awake {
            MentalPhase::NREM | MentalPhase::REM => MentalPhase::Active,
            other => other,
        }
    } else if x[2] > REM_THRESHOLD {
        MentalPhase::REM
    } else {
        MentalPhase::NREM
    }
}

/// Clock time in minutes for the light cycle: local seconds of day, so the
/// 14-hour light period follows the machine's timezone.
pub fn clock_minutes(now_ms: u64) -> f64 {
    crate::store::mmap_state::local_seconds_of_day(now_ms) as f64 / 60.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integrate for `minutes` at a 0.02-minute step, sampling every
    /// `every` minutes. Returns (t, state) pairs.
    fn simulate(minutes: f64, every: f64) -> Vec<(f64, State)> {
        let dt = 0.02;
        let steps = (minutes / dt).round() as usize;
        let stride = (every / dt).round() as usize;
        let mut x = INITIAL;
        let mut out = Vec::new();
        for i in 0..steps {
            let t = i as f64 * dt;
            if i % stride == 0 {
                out.push((t, x));
            }
            x = rk4(t, &x, dt);
        }
        out
    }

    /// Sleep, REM onset and REM offset times (minutes) by the source's
    /// thresholds.
    struct Events {
        sleep_on: Vec<f64>,
        sleep_off: Vec<f64>,
        rem_on: Vec<f64>,
        rem_off: Vec<f64>,
    }

    fn events(samples: &[(f64, State)]) -> Events {
        let mut e = Events {
            sleep_on: vec![],
            sleep_off: vec![],
            rem_on: vec![],
            rem_off: vec![],
        };
        for w in samples.windows(2) {
            let (t, a) = (w[0].0, &w[0].1);
            let b = &w[1].1;
            if a[0] >= SLEEP_THRESHOLD && b[0] < SLEEP_THRESHOLD {
                e.sleep_on.push(t);
            }
            if a[0] < SLEEP_THRESHOLD && b[0] >= SLEEP_THRESHOLD {
                e.sleep_off.push(t);
            }
            if a[2] <= REM_THRESHOLD && b[2] > REM_THRESHOLD {
                e.rem_on.push(t);
            }
            if a[2] > REM_THRESHOLD && b[2] <= REM_THRESHOLD {
                e.rem_off.push(t);
            }
        }
        e
    }

    /// REM bouts in the first sleep period at or after `night_start`:
    /// latency from sleep onset to the first REM onset, and each bout's
    /// duration in order.
    fn night_architecture(e: &Events, night_start: f64) -> (f64, Vec<f64>) {
        let onset = *e
            .sleep_on
            .iter()
            .find(|t| **t >= night_start)
            .expect("sleep onset");
        let offset = *e
            .sleep_off
            .iter()
            .find(|t| **t > onset)
            .expect("sleep offset");
        let rems: Vec<(f64, f64)> = e
            .rem_on
            .iter()
            .zip(&e.rem_off)
            .filter(|(on, _)| **on > onset && **on < offset)
            .map(|(on, off)| (*on, *off - *on))
            .collect();
        let latency = rems.first().map_or(f64::NAN, |(on, _)| on - onset);
        (latency, rems.iter().map(|(_, d)| *d).collect())
    }

    const DAY_7: f64 = 7.0 * 1440.0;

    /// Measured over days 8–10: asleep 31.8% of the time (about 7.6 h a
    /// day), one sleep period a day, four REM bouts a night, and about 99
    /// minutes between REM onsets. These match the source's own statements.
    #[test]
    fn nightly_sleep_and_ultradian_rem() {
        let samples = simulate(14400.0, 0.5);
        let e = events(&samples);
        let late: Vec<&(f64, State)> = samples.iter().filter(|s| s.0 >= DAY_7).collect();
        let asleep =
            late.iter().filter(|s| s.1[0] < SLEEP_THRESHOLD).count() as f64 / late.len() as f64;
        assert!(
            (0.25..0.40).contains(&asleep),
            "asleep fraction {asleep:.3}"
        );

        let nights = e.sleep_on.iter().filter(|t| **t >= DAY_7).count();
        assert!(
            (2..=4).contains(&nights),
            "sleep periods in days 8-10: {nights}"
        );

        let rem_bouts = e.rem_on.iter().filter(|t| **t >= DAY_7).count();
        assert!(
            (10..=14).contains(&rem_bouts),
            "REM onsets in days 8-10: {rem_bouts}"
        );

        let gaps: Vec<f64> = e
            .rem_on
            .windows(2)
            .filter(|w| w[0] >= DAY_7)
            .map(|w| w[1] - w[0])
            .filter(|g| *g < 150.0)
            .collect();
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
        assert!(
            (85.0..115.0).contains(&mean_gap),
            "REM-to-REM gap {mean_gap:.1} min"
        );
    }

    /// Adult polysomnography norms (AASM / textbook sleep medicine): first
    /// REM period 70–120 min after sleep onset; REM about 15–30% of sleep.
    #[test]
    fn clinical_rem_latency_and_share() {
        let samples = simulate(14400.0, 0.5);
        let e = events(&samples);
        let (latency, bouts) = night_architecture(&e, DAY_7);
        assert!(
            (70.0..120.0).contains(&latency),
            "first REM latency {latency:.1} min"
        );
        assert!(
            bouts.len() >= 3,
            "expected several REM bouts, got {}",
            bouts.len()
        );
        let rem_total: f64 = bouts.iter().sum();
        let on = *e.sleep_on.iter().find(|t| **t >= DAY_7).unwrap();
        let off = *e.sleep_off.iter().find(|t| **t > on).unwrap();
        let rem_share = rem_total / (off - on);
        assert!(
            (0.15..0.30).contains(&rem_share),
            "REM share of sleep {rem_share:.3}"
        );
    }

    #[test]
    fn advance_is_finite_and_keeps_state_on_a_non_finite_step() {
        let next = advance(0.0, &INITIAL, 0.001);
        assert!(next.iter().all(|v| v.is_finite()));
        assert_eq!(advance(0.0, &INITIAL, f64::NAN), INITIAL);
    }

    /// A volitional sleep push must suppress the wake population relative to
    /// the undriven network over the same interval.
    #[test]
    fn volitional_sleep_drive_suppresses_the_wake_population() {
        let dt = 0.02;
        let steps = (60.0 / dt) as usize;
        let mut driven = INITIAL;
        let mut plain = INITIAL;
        for i in 0..steps {
            let t = i as f64 * dt;
            driven = advance_with(t, &driven, dt, Drive { vlpo: 1.0, lc: 0.0 });
            plain = advance(t, &plain, dt);
        }
        assert!(
            driven[0] < plain[0],
            "driven LC {} vs plain {}",
            driven[0],
            plain[0]
        );
    }

    #[test]
    fn stage_follows_the_source_thresholds() {
        use crate::state::zones::MentalPhase;
        let mut x = INITIAL;
        x[0] = 2.0;
        x[2] = 4.0;
        assert_eq!(stage(&x, MentalPhase::Active), MentalPhase::REM);
        x[2] = 1.0;
        assert_eq!(stage(&x, MentalPhase::Active), MentalPhase::NREM);
        x[0] = 5.0;
        assert_eq!(stage(&x, MentalPhase::NREM), MentalPhase::Active);
        assert_eq!(stage(&x, MentalPhase::Flow), MentalPhase::Flow);
    }

    #[test]
    fn apply_network_phase_publishes_the_stage_to_the_zones() {
        use crate::state::zones::MentalPhase;
        let mut s = crate::state::GenesisCoreState::new(1, 1_000);
        s.sleep_network = INITIAL;
        s.sleep_network[0] = 2.0;
        s.sleep_network[2] = 4.0;
        s.apply_network_phase();
        assert_eq!(s.zones.phase(), MentalPhase::REM);
        assert_eq!(s.neurochemicals.phase(), MentalPhase::REM);
    }

    /// The acetylcholine target must order the states as the data do (REM >
    /// waking > NREM), and sit near the measured levels 0.77, 0.58, 0.34
    /// (Vazquez & Baghdoyan, 2001). Uses the production release constants,
    /// measured on the network's own trajectory over days 8–10.
    #[test]
    fn acetylcholine_target_orders_the_states_as_the_data_do() {
        use crate::state::neurochemical::{ACH_BASE, ACH_FROM_LC, ACH_FROM_REM};
        use crate::state::zones::MentalPhase;
        let samples = simulate(14400.0, 1.0);
        let mut sums = [(0.0f64, 0usize); 3];
        for (_, x) in samples.iter().filter(|(t, _)| *t >= DAY_7) {
            let stage = stage(x, MentalPhase::Active);
            let net = levels(x);
            let target = f64::from(ACH_BASE)
                + f64::from(ACH_FROM_LC) * f64::from(net.lc)
                + f64::from(ACH_FROM_REM) * f64::from(net.rem);
            let i = match stage {
                MentalPhase::REM => 0,
                MentalPhase::NREM => 2,
                _ => 1,
            };
            sums[i].0 += target;
            sums[i].1 += 1;
        }
        let mean = |i: usize| sums[i].0 / sums[i].1.max(1) as f64;
        let (rem, wake, nrem) = (mean(0), mean(1), mean(2));
        assert!(
            rem > wake && wake > nrem,
            "REM {rem:.3} wake {wake:.3} NREM {nrem:.3}"
        );
        assert!((rem - 0.77).abs() < 0.08, "REM target {rem:.3}");
        assert!((wake - 0.58).abs() < 0.08, "waking target {wake:.3}");
        assert!((nrem - 0.34).abs() < 0.08, "NREM target {nrem:.3}");
    }

    /// A sustained sleep push should bring on sleep onset (LC below the
    /// source's 4 Hz threshold) within an hour of waking.
    #[test]
    fn volitional_sleep_drive_brings_on_sleep_onset() {
        let dt = 0.02;
        let mut x = INITIAL;
        let mut onset = None;
        for i in 0..(60.0 / dt) as usize {
            x = advance_with(i as f64 * dt, &x, dt, Drive { vlpo: 1.0, lc: 0.0 });
            if x[0] < SLEEP_THRESHOLD && onset.is_none() {
                onset = Some(i as f64 * dt);
            }
        }
        assert!(
            onset.is_some(),
            "no sleep onset within 60 min, final LC {}",
            x[0]
        );
    }

    /// A volitional wake push must raise the wake population from a sleeping
    /// state, relative to the undriven network over the same interval.
    #[test]
    fn volitional_wake_drive_raises_the_wake_population() {
        let dt = 0.02;
        let steps = (60.0 / dt) as usize;
        // Start asleep: sustained VLPO drive for 90 minutes first.
        let mut asleep = INITIAL;
        for i in 0..(90.0 / dt) as usize {
            asleep = advance_with(i as f64 * dt, &asleep, dt, Drive { vlpo: 1.0, lc: 0.0 });
        }
        let mut woken = asleep;
        let mut plain = asleep;
        for i in 0..steps {
            let t = 90.0 + i as f64 * dt;
            woken = advance_with(t, &woken, dt, Drive { vlpo: 0.0, lc: 1.0 });
            plain = advance(t, &plain, dt);
        }
        assert!(
            woken[0] > plain[0],
            "driven LC {} vs plain {}",
            woken[0],
            plain[0]
        );
    }

    #[test]
    fn clock_minutes_is_a_minute_of_the_local_day() {
        let m = clock_minutes(1_800_000_000_000);
        assert!((0.0..1440.0).contains(&m), "clock minutes {m}");
    }

    /// Known discrepancy. In adults REM bouts lengthen toward morning, the
    /// last longest. This network gives 25, 33.5, 28.5 and 21 minutes, so
    /// the last is the shortest. Not tuned to pass: the cause is either a
    /// difference between this network and human physiology, or a
    /// transcription error. Run with `--ignored` to see it fail.
    #[test]
    #[ignore = "known discrepancy: REM bouts do not lengthen across the night"]
    fn rem_bouts_lengthen_across_the_night() {
        let samples = simulate(14400.0, 0.5);
        let e = events(&samples);
        let (_, bouts) = night_architecture(&e, DAY_7);
        assert!(
            bouts.last().unwrap() > bouts.first().unwrap(),
            "REM bouts should lengthen across the night: {bouts:.1?}"
        );
    }
}

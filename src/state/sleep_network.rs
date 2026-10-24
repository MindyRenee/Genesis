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
//! ventrolateral preoptic area), `f_r` (REM-on, sublaterodorsal/precoeruleus),
//! `f_scn` (circadian pacemaker).
//!
//! Sleep onset is `f_lc` falling below [`SLEEP_THRESHOLD`]; REM onset is
//! `f_r` rising above [`REM_THRESHOLD`]. Both are the source's own event
//! definitions. Time is in minutes.

use std::f64::consts::PI;

// Synaptic gains between the neurotransmitter pools and the populations.
const GACH_LC: f64 = 0.8;
const GGABA_LC: f64 = 1.5;
const GSCN_LC: f64 = 0.5;
// The stress axis's own alerting channel onto LC. CRH and elevated
// glucocorticoids recruit the noradrenergic arousal system even with no
// sensory event — CRH activates the LC even in the absence of stress,
// and nocturnal cortisol elevation tracks sleep fragmentation (Buckley
// & Schatzberg, JCEM 2005; Vgontzas et al.). It is also the waking side
// of the diurnal rhythm: the pre-awakening cortisol surge accompanies
// the sleep→wake transition. The input is the cortisol stress fraction
// (0 at the basal floor, 1 at the reachable peak), so the term is silent
// at the nadir and ramps through the late-sleep surge. Half a volitional
// wake push: enough to lighten and fragment sleep under stress, not
// enough to drag a fresh VLPO down on its own.
const GCORT_LC: f64 = 0.5;
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

// Monoaminergic arousal → LC. The wake side of the switch is driven by
// the brain's own arousal chemistry, not an exogenous drive: orexin
// neurons project most densely onto the LC (Hagan et al. 1999, PNAS
// 96:10911) and are the switch's stabilizer — Saper's flip-flop has
// orexin reinforcing the monoaminergic wake populations, never VLPO
// (VLPO neurons lack orexin receptors). Histaminergic TMN neurons
// activate LC and are themselves silenced by VLPO during sleep — the
// secondary wake-side input. Both are measured against their resting
// baselines (the neurochemical defaults, `Orexin => 0.30`,
// `Histamine => 0.30`), so resting chemistry adds nothing and the
// source model is preserved at rest; *below*-baseline orexin weakens
// the wake side, the loss-of-stabilizer signature of narcolepsy.
const OXIN_REST: f64 = 0.30;
const HTM_REST: f64 = 0.30;
const GOXIN_LC: f64 = 1.2;
const GHTM_LC: f64 = 0.8;

/// Firing rate of LC (wake) below which the source counts sleep onset.
pub const SLEEP_THRESHOLD: f64 = 4.0;
/// Firing rate of REM-on above which the source counts REM onset.
pub const REM_THRESHOLD: f64 = 3.0;

/// State vector: `[f_lc, f_vlpo, f_r, f_scn, h, c, xc, n]`.
pub type State = [f64; 8];

/// The source's initial conditions.
pub const INITIAL: State = [
    5.71387902867966,    // f_lc
    0.395493299109024,   // f_vlpo
    0.0,                 // f_r
    4.01351469432799,    // f_scn
    214.116748078719,    // h
    0.00114912794709424, // c
    -1.15530846944936,   // xc
    0.604617443838918,   // n
];

fn sigmoid(x: f64, max: f64, beta: f64, alpha: f64) -> f64 {
    max * 0.5 * (1.0 + ((x - beta) / alpha).tanh())
}

/// Subcortical retinal afference for one integration step — the optic
/// nerve's non-image-forming branches, measured by the body from the
/// retina's shared-memory frames (`daemon::optic_nerve`).
///
/// Two projections matter to this network:
///
/// - **Retinohypothalamic tract (RHT)**: melanopsin ipRGCs project
///   (near-)exclusively to the SCN; retinal illuminance is the primary
///   zeitgeber. `light` carries it in the oscillator's lux units.
///   `None` means no live retinal read — the built-in photoperiod
///   stands in for the world, as before.
/// - **Superior colliculus → alerting**: a salient visual transient is
///   the sensory-gaining "wake-up call" — salient stimuli are still
///   analysed during sleep and can arouse when they matter. `alert_lc`
///   is a transient excitation onto the LC input for this step only;
///   the network's own flip-flop decides whether it crosses threshold.
///
/// Afference, not a command: it does not disable the
/// adenosine→homeostat coupling — the body still owns the switch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VisualAfference {
    /// Retinal illuminance in lux-equivalent units (`I_0` scale).
    pub light: Option<f64>,
    /// Transient LC excitation from a salient visual event.
    pub alert_lc: f64,
}

impl VisualAfference {
    /// No retinal afference: the source model unchanged.
    pub const NONE: VisualAfference = VisualAfference {
        light: None,
        alert_lc: 0.0,
    };
}

/// Right-hand side of the network at time `t` (minutes), at resting
/// arousal chemistry with no retinal or stress input — the source model.
pub fn deriv(t: f64, x: &State) -> State {
    deriv_with_afference(t, x, VisualAfference::NONE, 0.0, OXIN_REST, HTM_REST)
}

/// Right-hand side of the network at time `t` (minutes), with the optic
/// nerve's subcortical afference, the stress-axis alerting input (`hpa`:
/// the cortisol stress fraction, [0, 1]), and the monoaminergic arousal
/// inputs (`oxin`, `htm`: effective orexin and histamine levels, [0, 1],
/// measured against `OXIN_REST`/`HTM_REST`) added.
pub fn deriv_with_afference(
    t: f64,
    x: &State,
    aff: VisualAfference,
    hpa: f64,
    oxin: f64,
    htm: f64,
) -> State {
    let [f_lc, f_vlpo, f_r, f_scn, h, c, xc, n] = *x;
    let c_ne = (f_lc / GAMMA_NE).tanh();
    let c_gaba = (f_vlpo / GAMMA_GABA).tanh();
    let c_ach = (f_r / GAMMA_ACH_R).tanh();
    let c_scn = (f_scn / GAMMA_SCN).tanh();

    // Four alerting inputs sum on the same LC channel — the collicular
    // pulse (sensory gaining), circulating cortisol (humoral stress), and
    // the monoaminergic arousal drive. All of them are the body's own
    // chemistry or sensory input; nothing exogenous is held on the wire.
    let lc_input = GACH_LC * c_ach + GSCN_LC * c_scn - GGABA_LC * c_gaba
        + aff.alert_lc
        + GCORT_LC * hpa
        + GOXIN_LC * (oxin - OXIN_REST)
        + GHTM_LC * (htm - HTM_REST);
    let vlpo_input = -GNE_VLPO * c_ne - GSCN_VLPO * c_scn;
    let r_input = GACH_R * c_ach - GNE_R * c_ne + GSCN_R * c_scn - GGABA_R * c_gaba;
    let beta_vlpo = K2 * h + K1;

    let lc_inf = sigmoid(lc_input, MAX_LC, BETA_LC, ALPHA_LC);
    let r_inf = sigmoid(r_input, MAX_R, BETA_R, ALPHA_R);
    let scn_inf = sigmoid(c, MAX_SCN, BETA_SCN, ALPHA_SCN);
    let vlpo_inf = sigmoid(vlpo_input, MAX_VLPO, beta_vlpo, ALPHA_VLPO);

    // Light reaches the oscillator only through the retinohypothalamic
    // tract — measured retinal illuminance when the eye is live, and
    // the built-in photoperiod (the world's fallback) when it is not.
    let i_light: f64 = aff.light.unwrap_or_else(|| scheduled_light(t));
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

/// The built-in photoperiod — the world's fallback light signal when
/// no retinal measurement exists. 500 lux during the 14 h light
/// period, else 0. The source runs light from minute 600 of each
/// simulated day (10:00–24:00); here t is anchored to the local wall
/// clock, so the window is phase-shifted to 07:00–21:00 — a 14 h
/// photoperiod, preserved — so that the entrained sleep episode
/// (~22:11–05:50) falls inside the neurochemical circadian's
/// biological night rather than ending ~2 h after melatonin offset as
/// it did with the source's alignment.
fn scheduled_light(t: f64) -> f64 {
    if (420.0..1260.0).contains(&t.rem_euclid(1440.0)) {
        500.0
    } else {
        0.0
    }
}

/// One fourth-order Runge–Kutta step of `dt` minutes from time `t`, with
/// retinal afference and chemical inputs held constant over the step.
pub fn rk4_with_afference(
    t: f64,
    x: &State,
    dt: f64,
    aff: VisualAfference,
    hpa: f64,
    oxin: f64,
    htm: f64,
) -> State {
    let add = |a: &State, b: &State, k: f64| -> State { std::array::from_fn(|i| a[i] + k * b[i]) };
    let k1 = deriv_with_afference(t, x, aff, hpa, oxin, htm);
    let k2 = deriv_with_afference(t + dt / 2.0, &add(x, &k1, dt / 2.0), aff, hpa, oxin, htm);
    let k3 = deriv_with_afference(t + dt / 2.0, &add(x, &k2, dt / 2.0), aff, hpa, oxin, htm);
    let k4 = deriv_with_afference(t + dt, &add(x, &k3, dt), aff, hpa, oxin, htm);
    std::array::from_fn(|i| x[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
}

/// One RK4 step of the source model — resting chemistry, no afference.
pub fn rk4(t: f64, x: &State, dt: f64) -> State {
    rk4_with_afference(t, x, dt, VisualAfference::NONE, 0.0, OXIN_REST, HTM_REST)
}

/// Advance the network by `dt_min` minutes from `t_min` under the optic
/// nerve's subcortical afference, the stress-axis alerting input (`hpa`:
/// cortisol stress fraction, [0, 1]), and the monoaminergic arousal
/// inputs (`oxin`, `htm`: effective levels, [0, 1]).
///
/// A non-finite result keeps the previous state. The network is authoritative
/// and persistent, so one bad step must not be allowed to poison it.
pub fn advance_with_afference(
    t_min: f64,
    x: &State,
    dt_min: f64,
    aff: VisualAfference,
    hpa: f64,
    oxin: f64,
    htm: f64,
) -> State {
    let next = rk4_with_afference(t_min, x, dt_min, aff, hpa, oxin, htm);
    if next.iter().all(|v| v.is_finite()) {
        next
    } else {
        *x
    }
}

/// Advance the network by `dt_min` minutes at resting chemistry with no
/// afference — the source model.
pub fn advance(t_min: f64, x: &State, dt_min: f64) -> State {
    advance_with_afference(
        t_min,
        x,
        dt_min,
        VisualAfference::NONE,
        0.0,
        OXIN_REST,
        HTM_REST,
    )
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

/// Map the neurochemical adenosine level (resting 0.20, upper
/// asymptote 0.95) onto the sleep-wake network's homeostatic drive
/// `h` in `[H_MIN, H_MAX]` (`[0.0, 323.88]`).
///
/// Coupling `h` to `Adenosine` ensures that only her own chemical
/// sleep pressure can flip the VLPO/LC switch into sleep, and
/// glymphatic adenosine clearance during sleep is what releases it.
pub fn h_from_adenosine(adenosine: f32) -> f64 {
    let norm = ((f64::from(adenosine) - 0.20) / (0.95 - 0.20)).clamp(0.0, 1.0);
    H_MIN + norm * (H_MAX - H_MIN)
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

    /// The first state on the source trajectory where LC sits below the
    /// sleep threshold — the honest way to reach sleep in tests now that
    /// nothing drives VLPO from outside: let the source model fall asleep
    /// on its own circadian/homeostatic schedule.
    fn first_asleep_state() -> State {
        simulate(4320.0, 1.0)
            .iter()
            .map(|(_, x)| *x)
            .find(|x| x[0] < SLEEP_THRESHOLD)
            .expect("source model must sleep within three days")
    }

    /// Elevated monoaminergic arousal must raise the wake population
    /// relative to resting chemistry over the same interval — a
    /// wake-signature chemical state (orexin + histamine high, as a
    /// /wake impulse would leave them) is the alerting channel, not a
    /// command on the wire.
    #[test]
    fn monoamine_arousal_raises_the_wake_population() {
        let dt = 0.02;
        let mut aroused = first_asleep_state();
        let mut resting = aroused;
        for i in 0..(30.0 / dt) as usize {
            let t = i as f64 * dt;
            aroused = advance_with_afference(t, &aroused, dt, VisualAfference::NONE, 0.0, 0.9, 0.9);
            resting = advance_with_afference(
                t,
                &resting,
                dt,
                VisualAfference::NONE,
                0.0,
                OXIN_REST,
                HTM_REST,
            );
        }
        assert!(
            aroused[0] > resting[0],
            "aroused LC {} vs resting {}",
            aroused[0],
            resting[0]
        );
    }

    /// The narcolepsy signature, and the reason the coupling is signed:
    /// orexin *below* its resting baseline weakens the LC wake side —
    /// the loss-of-stabilizer instability, not a silent floor.
    #[test]
    fn monoamine_loss_weakens_the_wake_population() {
        let dt = 0.02;
        let mut depleted = INITIAL;
        let mut resting = INITIAL;
        for i in 0..(60.0 / dt) as usize {
            let t = i as f64 * dt;
            depleted =
                advance_with_afference(t, &depleted, dt, VisualAfference::NONE, 0.0, 0.0, 0.0);
            resting = advance_with_afference(
                t,
                &resting,
                dt,
                VisualAfference::NONE,
                0.0,
                OXIN_REST,
                HTM_REST,
            );
        }
        assert!(
            depleted[0] < resting[0],
            "depleted LC {} vs resting {}",
            depleted[0],
            resting[0]
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

    /// The Sleeping zone must never mask the network's stage. A wake
    /// attractor the display keeps calling NREM leaves the organism
    /// asleep forever: the mind's desync check reads the published
    /// phase, and a phase the zone pins cannot report the body's wake
    /// the mind is waiting on.
    #[test]
    fn sleeping_zone_does_not_mask_the_network_stage() {
        use crate::state::GenesisCoreState;
        use crate::state::zones::{CognitiveZone, MentalPhase};
        let mut s = GenesisCoreState::new(1, 1_000);
        s.zones.transition_to(CognitiveZone::Sleeping, 10_000);
        // Wake attractor: LC firing above the sleep threshold — the
        // body has woken, and the published phase must say so even
        // while the mind's zone still reads Sleeping.
        s.sleep_network = INITIAL;
        s.sleep_network[0] = 5.9;
        s.apply_network_phase();
        assert_eq!(s.zones.phase(), MentalPhase::Active);
        assert_eq!(s.neurochemicals.phase(), MentalPhase::Active);
        // Sleep stages still publish while the network is asleep.
        s.sleep_network[0] = 2.0;
        s.sleep_network[2] = 0.0;
        s.apply_network_phase();
        assert_eq!(s.zones.phase(), MentalPhase::NREM);
        s.sleep_network[2] = 4.0;
        s.apply_network_phase();
        assert_eq!(s.zones.phase(), MentalPhase::REM);
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

    #[test]
    fn clock_minutes_is_a_minute_of_the_local_day() {
        let m = clock_minutes(1_800_000_000_000);
        assert!((0.0..1440.0).contains(&m), "clock minutes {m}");
    }

    /// Known limitation, verified against the source. In adults REM
    /// bouts lengthen toward morning, the last longest. This port
    /// reproduces the published model faithfully — the upstream
    /// `DinizBehnBooth_SWnetworkModel.m` (ModelDB 244485, rerun
    /// 2026-10-19) itself produces 25.2, 33.5, 28.8, 21.2 minutes, a
    /// rise-then-fall pattern, so the discrepancy is the model's, not a
    /// transcription error. This test pins the port to the source's own
    /// bout durations rather than to adult physiology.
    #[test]
    fn rem_bout_durations_match_the_source_model() {
        let samples = simulate(14400.0, 0.5);
        let e = events(&samples);
        let (_, bouts) = night_architecture(&e, DAY_7);
        let source = [25.2, 33.5, 28.8, 21.2];
        assert_eq!(
            bouts.len(),
            source.len(),
            "REM bout count across the night: {bouts:.1?}"
        );
        for (i, (got, expected)) in bouts.iter().zip(source.iter()).enumerate() {
            assert!(
                (got - expected).abs() <= 2.0,
                "REM bout {i}: port {got:.1} min vs source {expected:.1} min"
            );
        }
    }

    /// `VisualAfference::NONE` at resting chemistry must be exactly the
    /// source model — the optic nerve absent and arousal at baseline is
    /// the organism the model always was.
    #[test]
    fn no_afference_is_the_source_model() {
        let x = INITIAL;
        for &t in &[0.0, 720.0, 1100.0] {
            let a = advance(t, &x, 0.5);
            let b =
                advance_with_afference(t, &x, 0.5, VisualAfference::NONE, 0.0, OXIN_REST, HTM_REST);
            assert_eq!(a, b, "t={t}");
        }
    }

    /// Measured retinal light — not the wall clock — must be what
    /// drives the SCN's photic input while the eye is live. At noon
    /// the photoperiod supplies 500 lux; a dark eye overrides it
    /// downward and a bright one upward, in the `n` (photic
    /// activation) derivative where the light term enters.
    #[test]
    fn retinal_light_overrides_the_photoperiod() {
        let x = INITIAL;
        let noon = 720.0;
        let dn = |light: Option<f64>| -> f64 {
            deriv_with_afference(
                noon,
                &x,
                VisualAfference {
                    light,
                    alert_lc: 0.0,
                },
                0.0,
                OXIN_REST,
                HTM_REST,
            )[7]
        };
        let dark = dn(Some(0.0));
        let scheduled = dn(None); // 500 lux at noon
        let bright = dn(Some(9500.0));
        assert!(
            dark < scheduled,
            "dark eye at noon: dn {dark} vs photoperiod {scheduled}"
        );
        assert!(
            bright > scheduled,
            "bright eye at noon: dn {bright} vs photoperiod {scheduled}"
        );
        // At night the photoperiod is already dark — a dark eye is
        // indistinguishable from it, and a bright one still overrides.
        let night = 60.0;
        let dn_night = |light: Option<f64>| -> f64 {
            deriv_with_afference(
                night,
                &x,
                VisualAfference {
                    light,
                    alert_lc: 0.0,
                },
                0.0,
                OXIN_REST,
                HTM_REST,
            )[7]
        };
        assert_eq!(dn_night(None), dn_night(Some(0.0)));
        assert!(dn_night(Some(9500.0)) > dn_night(None));
    }

    /// The collicular alerting pulse must raise the wake population
    /// from a sleeping state relative to the un-alerted network over
    /// the same interval — a salient visual event is a wake-up call
    /// into LC, not a bypass of the flip-flop.
    #[test]
    fn collicular_alert_raises_the_wake_population() {
        let dt = 0.02;
        let asleep = first_asleep_state();
        let aff = VisualAfference {
            light: None,
            alert_lc: 1.5,
        };
        let mut alerted = asleep;
        let mut plain = asleep;
        for i in 0..(30.0 / dt) as usize {
            let t = 90.0 + i as f64 * dt;
            alerted = advance_with_afference(t, &alerted, dt, aff, 0.0, OXIN_REST, HTM_REST);
            plain = advance(t, &plain, dt);
        }
        assert!(
            alerted[0] > plain[0],
            "alerted LC {} vs plain {}",
            alerted[0],
            plain[0]
        );
    }

    /// The stress-axis alerting input must raise the wake population
    /// from a sleeping state relative to an un-stressed network over
    /// the same interval — sustained nocturnal HPA elevation lightens
    /// sleep through the LC, the way a salient flash does.
    #[test]
    fn hpa_alert_raises_the_wake_population() {
        let dt = 0.02;
        let asleep = first_asleep_state();
        let mut stressed = asleep;
        let mut basal = asleep;
        for i in 0..(30.0 / dt) as usize {
            let t = 90.0 + i as f64 * dt;
            // hpa = 1.0: cortisol at the top of its envelope — a
            // saturated stress fraction, a strong arousal push.
            stressed = advance_with_afference(
                t,
                &stressed,
                dt,
                VisualAfference::NONE,
                1.0,
                OXIN_REST,
                HTM_REST,
            );
            basal = advance_with_afference(
                t,
                &basal,
                dt,
                VisualAfference::NONE,
                0.0,
                OXIN_REST,
                HTM_REST,
            );
        }
        assert!(
            stressed[0] > basal[0],
            "stressed LC {} vs basal {}",
            stressed[0],
            basal[0]
        );
    }

    /// And the other half of the biology: a resting-floor axis adds
    /// nothing — basal cortisol must not lighten sleep at all.
    #[test]
    fn basal_hpa_is_silent() {
        let x = INITIAL;
        for &t in &[0.0, 720.0, 1100.0] {
            let a = advance(t, &x, 0.5);
            let b =
                advance_with_afference(t, &x, 0.5, VisualAfference::NONE, 0.0, OXIN_REST, HTM_REST);
            assert_eq!(a, b, "hpa=0 must equal no input, t={t}");
        }
    }
}

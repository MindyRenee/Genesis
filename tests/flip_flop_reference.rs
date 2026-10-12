//! Reference check: the Phillips & Robinson (2007) wake/sleep flip-flop.
//!
//! The equations and parameters are transcribed from the CellML model
//! of Phillips & Robinson, J. Biol. Rhythms 22:167–179 (2007), as
//! published in the Physiome repository (phillips_2007.cellml). This
//! file is a sourced reference the neurochemical sleep dynamics can be
//! compared against. It is not part of the daemon.
//!
//! The model has three dynamic variables: VLPO (sleep-promoting) and
//! monoaminergic (wake-promoting) firing-rate populations in mutual
//! inhibition, and H, the homeostatic sleep pressure. The circadian
//! drive enters through D = v_vc·C + v_vh·H, with C = c0 + cos(ωt).
//! Time is in hours. Membrane time constants are given in seconds in
//! the source and converted here.

use std::f64::consts::PI;

const QMAX: f64 = 100.0; // s^-1
const THETA: f64 = 10.0; // mV
const SIGMA: f64 = 3.0; // mV
const TAU_V_H: f64 = 10.0 / 3600.0; // VLPO membrane time constant, h
const TAU_M_H: f64 = 10.0 / 3600.0; // monoaminergic membrane time constant, h
const V_VM: f64 = -1.9; // mV·s, MA -> VLPO
const V_MV: f64 = -1.9; // mV·s, VLPO -> MA
const V_MA_QAO: f64 = 1.0; // mV, constant cholinergic drive to MA (v_ma·Q_a)
const MU: f64 = 3.6; // nM·s, homeostatic production per MA firing
const CHI: f64 = 10.8; // h, homeostatic time constant
const V_VC: f64 = -6.3; // mV, circadian gain
const V_VH: f64 = 0.19; // mV/nM, homeostatic gain
const C0: f64 = 1.0;
const OMEGA: f64 = 2.0 * PI / 24.0; // h^-1

#[derive(Clone, Copy, Debug)]
struct State {
    vv: f64,
    vm: f64,
    h: f64,
}

fn rate(v: f64) -> f64 {
    QMAX / (1.0 + (-(v - THETA) / SIGMA).exp())
}

/// Right-hand side. `t` in hours.
fn deriv(t: f64, s: State) -> State {
    let qv = rate(s.vv);
    let qm = rate(s.vm);
    let c = C0 + (OMEGA * t).cos();
    let d = V_VC * c + V_VH * s.h;
    State {
        vv: ((V_VM * qm + d) - s.vv) / TAU_V_H,
        vm: ((V_MA_QAO + V_MV * qv) - s.vm) / TAU_M_H,
        h: (MU * qm - s.h) / CHI,
    }
}

fn step(t: f64, s: State, dt: f64) -> State {
    let add = |a: State, b: State, k: f64| State {
        vv: a.vv + k * b.vv,
        vm: a.vm + k * b.vm,
        h: a.h + k * b.h,
    };
    let k1 = deriv(t, s);
    let k2 = deriv(t + dt / 2.0, add(s, k1, dt / 2.0));
    let k3 = deriv(t + dt / 2.0, add(s, k2, dt / 2.0));
    let k4 = deriv(t + dt, add(s, k3, dt));
    State {
        vv: s.vv + dt / 6.0 * (k1.vv + 2.0 * k2.vv + 2.0 * k3.vv + k4.vv),
        vm: s.vm + dt / 6.0 * (k1.vm + 2.0 * k2.vm + 2.0 * k3.vm + k4.vm),
        h: s.h + dt / 6.0 * (k1.h + 2.0 * k2.h + 2.0 * k3.h + k4.h),
    }
}

/// Integrate for `hours` at 1-second steps, returning (t, state, awake)
/// samples every `every` steps. Awake means the monoaminergic group
/// fires more than VLPO.
fn simulate(hours: f64, every: usize) -> Vec<(f64, State, bool)> {
    let dt = 1.0 / 3600.0;
    let n = (hours / dt).round() as usize;
    let mut s = State { vv: 0.0, vm: 0.0, h: 15.0 };
    let mut out = Vec::new();
    for i in 0..n {
        let t = i as f64 * dt;
        if i % every == 0 {
            out.push((t, s, rate(s.vm) > rate(s.vv)));
        }
        s = step(t, s, dt);
    }
    out
}

fn bouts(samples: &[(f64, State, bool)]) -> Vec<(bool, f64, f64)> {
    let mut out: Vec<(bool, f64, f64)> = Vec::new();
    for &(t, _, awake) in samples {
        match out.last_mut() {
            Some(last) if last.0 == awake => last.2 = t,
            _ => out.push((awake, t, t)),
        }
    }
    out
}

/// Measured with this transcription (5 simulated days): about one
/// wake/sleep alternation per 24 h, with ~19 h wake and ~5 h sleep
/// bouts. The 2017 network (rem_network_reference.rs), which builds on
/// this model, gives ~7.6 h of sleep a day. So the shorter sleep here may
/// be a property of the 2007 parameters rather than a transcription
/// error, but that is not confirmed: the CellML's claim to reproduce its
/// Figure 2 could not be checked. The assertions cover only what is
/// consistent with the source.
#[test]
fn flip_flop_produces_a_daily_wake_sleep_cycle() {
    let samples = simulate(120.0, 60); // 5 days, sampled every 60 s
    let b = bouts(&samples);
    let sleep: f64 = b.iter().filter(|x| !x.0).map(|x| x.2 - x.1).sum();
    assert!(
        (8..=14).contains(&b.len()),
        "about two bouts per day, got {} transitions",
        b.len()
    );
    let sleep_fraction = sleep / 120.0;
    assert!(
        (0.1..0.5).contains(&sleep_fraction),
        "sleep fraction {sleep_fraction:.3}"
    );
}

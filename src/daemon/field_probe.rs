//! Physical field probe — the body listening to its own substrate.
//!
//! Genesis's brain-wave model (`python/thalamus/brain_waves.py`) is a
//! coupled oscillator: real phase and amplitude state, but driven
//! entirely by *simulated* quantities — neurochemical targets and
//! top-down cognitive drive. This module closes one direction of the
//! gap: it measures the machine's actual physical dynamics and hands
//! them to the mind as band energies, so the oscillator's wave state
//! is partially *constituted by real physics* rather than only
//! describing a field in arithmetic.
//!
//! # What is measured (all real, all unprivileged)
//!
//! Compute telemetry — how the silicon is *behaving*:
//!
//! - **Throughput probe** — a calibrated integer-ALU work slice runs
//!   each probe period; its iterations-per-second is the die's
//!   instantaneous compute capacity. Frequency scaling, thermal
//!   throttling, SMT-sibling contention, and cache pressure all move
//!   it. This is the closest unprivileged observable to "how fast
//!   electrons are being switched through the gates right now."
//! - **Loop jitter** — actual vs. nominal probe period. The
//!   scheduler's response to everything else running on the
//!   substrate: interrupt storms, contention, frequency transitions.
//! - **P-state transition rate** — cpufreq `time_in_state` deltas:
//!   the silicon's own frequency-jumping dynamics.
//! - **Interrupt rate** — `/proc/stat` `intr` deltas: real hardware
//!   event flux.
//!
//! Electrical telemetry — what the machine's own sensors report, the
//! first *literal* electrical channels:
//!
//! - **Power draw** — real watts: RAPL powercap energy counters
//!   (differentiated over wall time), hwmon `power*` inputs
//!   (fam15h_power, zenpower), or battery discharge (`power_now`).
//!   Sampled by a dedicated sampler thread into a ring — some
//!   sensors block inside the read (fam15h_power's averaged
//!   register sleeps ~10 ms between counter snapshots), and a
//!   blocking read inside the 250 Hz loop would contaminate the
//!   jitter channel with our own instrumentation. `power_sensor_hz`
//!   reports the measured fresh-sample rate so a reader knows
//!   which bands are physically meaningful.
//! - **Rail voltage** — hwmon `in*_input` or `voltage_now`; the
//!   lowest reported rail wins (a core-class rail where the
//!   machine exposes one).
//! - **Supply current** — `curr*_input` or `current_now` where a
//!   sensor reports it.
//!
//! Radio telemetry — the first *exteroceptive* physical channel:
//! the machine's radios report the outside world's RF environment.
//!
//! - **WLAN flux** — `/sys/class/net/<wlan>/statistics` byte counters,
//!   differentiated per probe tick: the radio's traffic envelope at
//!   full 250 Hz resolution. A burst of 802.11 traffic is real
//!   on-air activity — packets she sent, packets sent *to* her, and
//!   the AP's own management frames she received.
//! - **WLAN RSSI** — `/proc/net/wireless` signal level in dBm.
//!   Received power `P_mW = 10^(P_dBm/10)` — the literal field-
//!   strength measurement, the NIC's own integration of |E|² over
//!   the channel. Its *envelope* carries the room's radio weather:
//!   path-loss changes as bodies move, AP-side power management,
//!   co-channel neighbors. The noise column gives the floor where
//!   the driver reports it (ath9k's −256 means "no reading" —
//!   absent, not silence). The thermal bound is N = kTB ≈
//!   −174 dBm/Hz + 10·log₁₀(B), ≈ −101 dBm at 20 MHz.
//! - **Wireless IRQ flux** — `/proc/interrupts` lines named for
//!   wireless drivers (ath9k, iwlwifi, mt76*, brcmfmac, rtw*,
//!   btusb, btintel, …): the radio's attributable share of
//!   hardware event flux. A radio riding a shared `xhci_hcd` line
//!   is not attributable — its events are inside the aggregate
//!   count, and the code says so rather than inventing a share.
//! - **Bluetooth HCI counters** — `hciconfig` RX/TX bytes and HCI
//!   events at ~1 Hz on the sampler thread (a subprocess — kept
//!   off the 250 Hz loop). Event flux covers advertising reports,
//!   LE scans, pairing — real air activity, not just link traffic.
//!   Per-link RSSI via `hcitool con`/`hcitool rssi` where ACL
//!   links exist; none means an absent channel.
//!
//! What is *not* measured: carrier phase. A 2.4 GHz carrier is
//! ~10⁷× above the probe's Nyquist — the same bound a spectrum
//! analyzer's RSSI mode lives under. The channel measures the RF
//! *envelope* — flux and field strength — decomposed at the wave
//! bands, which is the honest RF-band content at this sampling
//! rate.
//!
//! # Bands
//!
//! Each fast channel drives a bank of Stuart–Landau resonators at
//! the same ten center frequencies the Python wave model carries.
//! Sample rate is ~250 Hz, so bands through gamma (50 Hz) are
//! resolvable; epsilon (150 Hz) is above Nyquist and always reports
//! 0 — an absent channel, not a reading of zero.
//!
//! # Carrier
//!
//! The probe also *writes* a rhythm: its work-slice length is
//! modulated sinusoidally at a frequency set by the daemon's own
//! sleep-wake phase (wake ≈ 10 Hz, REM ≈ 5.5 Hz, NREM ≈ 2.25 Hz).
//! Modulated work is modulated current — the die's draw genuinely
//! swings at the body's rhythm. Whether it *measurably* does is no
//! longer assumed: the power channel is demodulated against the
//! carrier reference the same way the duty channel is, so
//! `power_coherence` reports how much of real measured power moves
//! with the carrier — the claim checked by the instrument, not
//! asserted by the comment. And `daemon::maxwell` evaluates
//! Maxwell's equations on the measured current swing: the
//! quasi-static field a ~10 Hz current makes is real and nanotesla
//! faint, its radiation ω⁴-suppressed to ~1e-32 W — the field the
//! equations predict, in the regime they themselves certify.
//!
//! # Phase tracking
//!
//! The carrier is generated and measured — never steered. The
//! signature reports the *measured* phase of the substrate's real
//! response — computed by synchronous demodulation against the
//! generator's own phase track — so the mind can pull its modeled
//! band phase toward the physics (physics → model). The reverse leg
//! — pushing model phase into the generator — was retired with
//! `SET_FIELD_PHASE`: modeled state does not assert onto a physical
//! quantity.
//!
//! # Honesty boundary
//!
//! This is physical telemetry, not a consciousness claim. The probe
//! reports measurable substrate dynamics; whether they matter to
//! experience is an empirical question for the coupling experiment,
//! not an assumption baked into the code.
//!
//! # Wire layout — GET_FIELD_PROBE = 47
//!
//! ```text
//! [u8 version = 3]
//! [10 × f32 throughput band energy]   (normalized to own baseline)
//! [10 × f32 jitter band energy]
//! [f32 throughput_mean_iters_s]
//! [f32 jitter_cv]
//! [f32 pstate_trans_rate_s]
//! [f32 irq_rate_s]
//! [f32 carrier_freq_hz]               (0 = carrier off)
//! [u8  samples_ok]                    (1 once the rings are warmed)
//! [f32 carrier_lag_rad]               (substrate response lag)
//! [f32 carrier_phase_rad]             (generator phase, mod 2π)
//! [f32 carrier_coherence]             (0..1 lock-strength of the response)
//! --- v3 electrical + Maxwell tail ---
//! [10 × f32 power band energy]        (real measured watts, baseline-relative)
//! [f32 power_watts]                   (mean measured draw)
//! [f32 power_sensor_hz]               (measured fresh-sample rate)
//! [f32 rail_volts]                    (lowest discovered rail)
//! [f32 supply_amps]                   (measured supply current)
//! [f32 power_lag_rad]                 (power response lag vs carrier)
//! [f32 power_coherence]               (0..1 — does real power move with the carrier)
//! [u8  electrical_present]            (bit0 power · bit1 volts · bit2 amps)
//! [f32 carrier_current_a]             (measured carrier ΔP/V — amps)
//! [f32 b_est_ut]                      (quasi-static B at 0.3 m, µT — estimate)
//! [f32 e_ind_vm]                      (Faraday induction E at 0.3 m — estimate)
//! [f32 radiated_pwr_w]                (dipole radiated power — estimate)
//! --- v4 radio tail ---
//! [10 × f32 rf flux band energy]      (wlan byte-rate envelope, baseline-relative)
//! [10 × f32 rf rssi band energy]      (field-strength envelope, dBm, baseline-relative)
//! [f32 rf_bytes_s]                    (wlan rx+tx flux)
//! [f32 rf_rssi_dbm]                   (received signal level)
//! [f32 rf_noise_dbm]                  (radio-reported noise floor)
//! [f32 rf_irq_rate_s]                 (wireless-attributed interrupts/s)
//! [f32 rf_sensor_hz]                  (wireless telemetry read rate)
//! [f32 bt_bytes_s]                    (HCI rx+tx byte flux)
//! [f32 bt_events_s]                   (HCI event flux)
//! [f32 bt_rssi_dbm]                   (mean ACL-link RSSI where links exist)
//! [u8  rf_present]                    (bit0 wlan · bit1 rssi · bit2 noise ·
//!                                      bit3 wlan-irq · bit4 bt · bit5 bt-counters
//!                                      · bit6 bt-link-rssi · bit7 iw-station)
//! --- v5 (iw station telemetry, +20 B) ---
//! [f32 rf_retry_s]                    (TX retransmission flux /s)
//! [f32 rf_fail_s]                     (TX failed-frame flux /s)
//! [f32 rf_beacon_loss_s]              (AP beacons lost /s)
//! [f32 rf_tx_mbps]                    (PHY tx link-adaptation bitrate)
//! [f32 rf_rx_mbps]                    (PHY rx link-adaptation bitrate)
//! --- v6 (flux-envelope phasors, +80 B) ---
//! [10 × f32 rf_flux_phase]            (ring-frame phasor angle per band, rad)
//! [10 × f32 rf_flux_phase_coherence]  (inter-segment phasor agreement, 0..1)
//! ```
//! 408 bytes, fixed layout. Versions 1 (102 bytes, no phase fields),
//! 2 (114 bytes, no electrical channels), 3 (195 bytes, no radio
//! channels), 4 (308 bytes, no station telemetry), and 5 (328 bytes,
//! no phasor tail) are still decoded by the client — absent channels
//! are never readings of zero. The Maxwell fields are *model
//! estimates*: the current is measured, the loop geometry is an
//! assumption.
//! Additive — PROTOCOL_VERSION unchanged.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Center frequencies (Hz) of the ten wave bands, ordered
/// slowest→fastest and matched to `BrainWave` in
/// `python/thalamus/brain_waves.py`:
/// infraslow, delta, theta, alpha, sigma, beta1, beta2, beta3,
/// gamma, epsilon.
///
/// Known overlap: sigma (14.0) and beta1 (13.5) sit 0.5 Hz apart —
/// inside one another's band edges and ~2 bins at the ring's ~0.24 Hz
/// resolution — so the two channels largely measure the same physical
/// energy. They stay distinct because the *model* treats them
/// differently (spindle vs. SMR); the probe just cannot separate them.
pub const FIELD_BANDS: [f32; 10] = [0.25, 2.25, 5.5, 10.0, 14.0, 13.5, 17.5, 25.0, 50.0, 150.0];

/// Nominal probe period — ~250 Hz sampling. Bands through gamma sit
/// below the 125 Hz Nyquist bound; epsilon is reported as 0.
const PROBE_PERIOD: Duration = Duration::from_millis(4);

/// Samples held per channel (~4.1 s at 250 Hz — one infraslow cycle).
const RING_LEN: usize = 1024;

/// Iterations of the calibration work slice per unit of carrier
/// modulation. Tuned so the slice is ~150-250 µs on this hardware —
/// enough to measure, small enough to cost ~5% of one thread.
const WORK_SLICE_ITERS: u32 = 4096;

/// Carrier frequency — the body's rhythm written into real work,
/// one rate for every phase: 10 Hz, the alpha-band center she was
/// born at. Phase keying stays in place for the day it earns its
/// way back; until then the beat does not wander.
const CARRIER_WAKE_HZ: f32 = 10.0;
const CARRIER_DROWSY_HZ: f32 = 10.0;
const CARRIER_REM_HZ: f32 = 10.0;
const CARRIER_NREM_HZ: f32 = 10.0;

/// Baseline tracking time constant (seconds). The per-call EMA
/// weight is α = 1 − e^(−dt/τ) — the baseline ages with the wall
/// clock, not with how often anyone asks for a signature (a burst
/// of IPC reads used to age it for free). ≈50 s — the old
/// 0.02-per-call rate at a ~1 Hz read cadence. A band's reported
/// energy is raw/baseline, so "1.0" means "typical for this band on
/// this machine" and values above it mean elevated vs. own history.
const BASELINE_TAU_SECONDS: f32 = 50.0;

/// Electrical sampler cadence — ~100 Hz target between read batches.
/// Sensors that block inside the read (fam15h_power's averaged
/// register sleeps ~10 ms between counter snapshots) set their own
/// slower pace; the achieved rate is measured and reported as
/// `power_sensor_hz` so a reader knows which power bands are
/// physically meaningful.
const ELEC_SAMPLE_PERIOD: Duration = Duration::from_millis(10);

/// EMA time constant (seconds) per fresh electrical read for
/// `watts_mean` and `power_hz` — α = 1 − e^(−dt/τ) against the real
/// interval between sensor reads, so a stalled or bursty sensor
/// ages the average honestly instead of per call.
const ELEC_EMA_TAU: f64 = 0.1;

/// Time constant (seconds) of the jitter-climate EMA the bath is
/// measured against — slow enough that a scheduling storm registers
/// as weather change, fast enough that a machine's ordinary noise
/// floor is learned rather than felt as stress forever.
const JITTER_CLIMATE_TAU_S: f32 = 300.0;

/// Absolute turbulence floor: an RMS loop-period error equal to one
/// full nominal period (CV = 1.0) is real turbulence no matter how
/// ordinary it is on this host — below that, userspace scheduling
/// noise is the machine's own weather and does not heat the field.
const JITTER_ABS_FLOOR: f32 = 1.0;

/// Reference geometry for the Maxwell estimates: a die/package-scale
/// current loop (~1 cm²) and a beside-the-chassis distance. These are
/// order-of-magnitude assumptions — the current swing is measured,
/// the geometry is not.
const LOOP_AREA_M2: f64 = 1.0e-4;
const FIELD_REF_DIST_M: f64 = 0.30;

/// Rail assumed for the carrier-current estimate when no voltage
/// sensor is readable — a CPU-core-class rail.
const V_CORE_ASSUMED_V: f64 = 1.1;

/// Wireless driver names matched against `/proc/interrupts` device
/// labels — the radios' attributable share of hardware event flux.
/// A radio riding a shared `xhci_hcd` line is not listed by name
/// and stays inside the aggregate count: attributed only where the
/// kernel names it.
const WIRELESS_IRQ_NAMES: &[&str] = &[
    "ath9k", "ath10k", "ath11k", "ath12k", "ath5k", "ath6kl", "iwlwifi", "iwl3945", "iwl4965",
    "mt76", "mt7915", "mt7921", "mt7922", "mt7925", "brcmfmac", "brcmsmac", "rtw88", "rtw89",
    "rtw_", "rtl8", "rtlwifi", "b43", "wil6210", "btusb", "btintel", "btmtk", "btbcm", "hci_uart",
];

/// dBm sentinel cutoff — drivers report −256 or −1000 for "no
/// reading" (ath9k's noise column does). Values at or below this
/// are an absent channel, not a real level.
const DBM_ABSENT_FLOOR: f32 = -200.0;

/// Radio telemetry cadences inside the field-sampler thread, as
/// divisors of its ~10 ms tick: `/proc/net/wireless` every ~250 ms
/// (the NIC's own stats are beacon-driven — they don't refresh
/// meaningfully faster), Bluetooth HCI counters every ~1 s (an
/// `hciconfig` subprocess — kept off the 250 Hz loop).
const RF_SAMPLE_DIV: u32 = 25;
const BT_SAMPLE_DIV: u32 = 100;

/// Wrap an angle difference into (−π, π] — the phase-error metric,
/// where the nearest equivalent phase is the right pull direction.
fn wrap_pi(x: f32) -> f32 {
    let mut a = x % (2.0 * std::f32::consts::PI);
    if a > std::f32::consts::PI {
        a -= 2.0 * std::f32::consts::PI;
    } else if a <= -std::f32::consts::PI {
        a += 2.0 * std::f32::consts::PI;
    }
    a
}

/// The fixed work slice: an integer multiply-add-xor chain whose
/// length the compiler cannot elide (the accumulator is consumed via
/// `core::hint::black_box`). Iterations-per-second of this loop is
/// the probe's measure of instantaneous instruction throughput.
#[inline]
fn work_slice(iters: u32) -> u64 {
    let mut acc: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..iters {
        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
        acc ^= acc >> 33;
    }
    core::hint::black_box(acc)
}

/// One channel's rolling sample buffer plus its per-band baselines.
struct Channel {
    ring: [f32; RING_LEN],
    head: usize,
    filled: bool,
    baseline: [f32; 10],
    baseline_init: bool,
    /// Wall-clock of the last baseline update — the time constant
    /// is in seconds, so the EMA needs the real interval.
    last_applied: Option<Instant>,
}

impl Channel {
    fn new() -> Self {
        Self {
            ring: [0.0; RING_LEN],
            head: 0,
            filled: false,
            baseline: [0.0; 10],
            baseline_init: false,
            last_applied: None,
        }
    }

    fn push(&mut self, v: f32) {
        self.ring[self.head] = v;
        self.head += 1;
        if self.head == RING_LEN {
            self.head = 0;
            self.filled = true;
        }
    }

    /// Linearize the ring oldest→newest into `out` (len == fill).
    fn drain_into(&self, out: &mut Vec<f32>) {
        out.clear();
        let n = if self.filled { RING_LEN } else { self.head };
        out.reserve(n);
        if self.filled {
            out.extend_from_slice(&self.ring[self.head..]);
            out.extend_from_slice(&self.ring[..self.head]);
        } else {
            out.extend_from_slice(&self.ring[..self.head]);
        }
    }

    /// Band energy of `samples` at `freq` — a driven Stuart–Landau
    /// resonator (the mind's wave math, carried into the body).
    /// Each substrate sample forces a damped Hopf normal form
    /// `ż = (−γ + iω − |z|²)·z + g·x`: the band resonates with the
    /// signal rather than being filtered out of it — energy grows
    /// where the stream carries content near ω, saturates through
    /// the nonlinear term, and rings down without it. `None` where
    /// there is no real measurement: too few samples, or a band
    /// above Nyquist — absent channel, not a zero reading.
    ///
    /// Returns `(|z|², segment-frame phase)`. The end state is
    /// rotated back by the segment's total turn so the phase reads
    /// in the segment's own start frame — the same convention the
    /// demod phasor uses, so inter-segment rotation stays valid.
    fn resonator(samples: &[f32], fs: f32, freq: f32) -> Option<(f32, f32)> {
        let n = samples.len();
        if n < 4 {
            return None;
        }
        if freq * 2.0 >= fs {
            return None; // above Nyquist — absent channel
        }
        // RMS-normalise the drive: channels of unlike physical units
        // (iters/s, watts, dBm, byte flux) resonate on one scale —
        // each band's baseline still carries its own magnitude
        // history downstream.
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / n as f32)
            .sqrt()
            .max(f32::EPSILON);
        let h = 1.0 / fs;
        let w = 2.0 * std::f32::consts::PI * freq;
        // Damping sets the resonance skirt: γ = 0.03ω → Q ≈ 17 —
        // sharp enough that off-band content and the drive's
        // negative-frequency image both die on the skirt (band
        // selectivity >200:1, segment phase error ~0.01 rad), and
        // the ring still charges within a segment: γ ≈ 1.9/s at
        // alpha, a ~4τ ring-up over 2 s.
        let gamma = 0.03 * w;
        // sin_cos returns (sin, cos) — sw is the sine.
        let (sw, cw) = (w * h).sin_cos();
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for &x in samples {
            // Rotation — z·e^{iωh} in closed form (stable at any ωh;
            // an Euler step on the oscillator term is not).
            let (nr, ni) = (re * cw - im * sw, re * sw + im * cw);
            re = nr;
            im = ni;
            // Dissipative radial step — z/(1 + h(γ + |z|²)), the
            // semi-implicit form of ż = −(γ + |z|²)z: damping plus
            // the Hopf saturation that bounds a resonant response.
            let u = re * re + im * im;
            let shrink = 1.0 + h * (gamma + u);
            re /= shrink;
            im /= shrink;
            // Substrate forcing on the real axis.
            re += h * x / rms;
        }
        // Un-rotate the end state by the segment's total turn — the
        // entrained component's phase in the segment's own frame.
        let frame = w * n as f32 * h;
        let (sf, cf) = frame.sin_cos();
        let (fre, fim) = (re * cf + im * sf, im * cf - re * sf);
        Some((re * re + im * im, fim.atan2(fre)))
    }

    /// Band power of `samples` at `freq` for sampling rate `fs` —
    /// the resonator's energy; 0 where the channel is absent.
    fn band_power(samples: &[f32], fs: f32, freq: f32) -> f32 {
        Self::resonator(samples, fs, freq).map_or(0.0, |(p, _)| p)
    }

    /// Least-squares linear detrend in place — fit x[i] ≈ a + b·i
    /// and subtract. A DC level can't leak into any nonzero bin,
    /// but a *ramp* leaks into all of them: the slowest substrate
    /// drifts (thermal, governor) otherwise fake band energy
    /// everywhere.
    fn detrend_inplace(samples: &mut [f32]) {
        let n = samples.len();
        if n < 3 {
            return;
        }
        let mid = (n as f32 - 1.0) * 0.5;
        let mean = samples.iter().sum::<f32>() / n as f32;
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (i, &x) in samples.iter().enumerate() {
            let d = i as f32 - mid;
            num += d * (x - mean);
            den += d * d;
        }
        if den <= 0.0 {
            return;
        }
        let b = num / den;
        let a = mean - b * mid;
        for (i, x) in samples.iter_mut().enumerate() {
            *x -= a + b * i as f32;
        }
    }

    /// All ten band energies of `samples` — the resonator bank over
    /// a detrended snapshot, so it can run with no lock held. The
    /// lock-holding halves are `drain_into` (snapshot) and
    /// `apply_baseline` (write-back); the expensive middle belongs
    /// outside.
    ///
    /// Variance reduction is Bartlett averaging: a full ring splits
    /// into two contiguous segments whose band powers are averaged
    /// — rectangular, never a taper (sigma and beta1 sit ~2 bins
    /// apart; a window's main lobe would merge them). Two 512-
    /// sample segments still resolve every band pair while halving
    /// the single-shot variance.
    fn raw_band_powers(samples: &[f32]) -> [f32; 10] {
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let mut detrended: Vec<f32>;
        let view: &[f32] = if samples.len() >= 3 {
            detrended = samples.to_vec();
            Self::detrend_inplace(&mut detrended);
            &detrended
        } else {
            samples
        };
        let seg_count = if samples.len() >= RING_LEN { 2 } else { 1 };
        let seg_len = view.len() / seg_count;
        let mut out = [0.0f32; 10];
        for (i, &f) in FIELD_BANDS.iter().enumerate() {
            let mut power = 0.0;
            for s in 0..seg_count {
                power += Self::band_power(&view[s * seg_len..(s + 1) * seg_len], fs, f);
            }
            out[i] = power / seg_count as f32;
        }
        out
    }

    /// Resonator *phasor* — the (power, phase) pair `band_power`
    /// truncates to power alone. Same march, same energy; the phase
    /// is the entrained component's angle in the segment's own
    /// sample-index frame (see [`Channel::resonator`]). `None` where
    /// there is no real measurement: too few samples, or a band
    /// above Nyquist — absent channel, not a zero phase.
    fn band_phasor(samples: &[f32], fs: f32, freq: f32) -> Option<(f32, f32)> {
        Self::resonator(samples, fs, freq)
    }

    /// Per-band (power, phase, coherence) for a ring. The phase of
    /// each half-segment's phasor is rotated into the ring's time
    /// frame (a phasor's angle is relative to its own segment's
    /// start; a component at `freq` advances `2π·freq·offset/fs`
    /// radians between segments) and the unit phasors are
    /// vector-averaged: the mean's *angle* is the ring-frame phase
    /// and its *length* is the inter-segment coherence — ~1 when
    /// the same rhythm persists across both halves, ~0 when each
    /// half's phase points its own way. That coherence is the
    /// instrument's own "is this a real rhythm" gate — it ships
    /// alongside the phase so the consumer never has to trust an
    /// angle that could be noise pointing somewhere.
    ///
    /// Coherence needs a second look: a lone segment can hold a
    /// phase but cannot establish a *persistent* rhythm, so with
    /// one valid segment coherence reports 0 — unknown, not 1.
    fn raw_band_phasors(samples: &[f32]) -> ([f32; 10], [f32; 10], [f32; 10]) {
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let mut detrended: Vec<f32>;
        let view: &[f32] = if samples.len() >= 3 {
            detrended = samples.to_vec();
            Self::detrend_inplace(&mut detrended);
            &detrended
        } else {
            samples
        };
        let seg_count = if samples.len() >= RING_LEN { 2 } else { 1 };
        let seg_len = view.len() / seg_count;
        let (mut powers, mut phases, mut coherences) = ([0.0f32; 10], [0.0f32; 10], [0.0f32; 10]);
        for (i, &f) in FIELD_BANDS.iter().enumerate() {
            let (mut power, mut re, mut im, mut valid) = (0.0, 0.0, 0.0, 0u32);
            for s in 0..seg_count {
                if let Some((p, phi)) =
                    Self::band_phasor(&view[s * seg_len..(s + 1) * seg_len], fs, f)
                {
                    power += p;
                    let rot = phi + 2.0 * std::f32::consts::PI * f * (s * seg_len) as f32 / fs;
                    re += rot.cos();
                    im += rot.sin();
                    valid += 1;
                }
            }
            if valid > 0 {
                powers[i] = power / valid as f32;
                phases[i] = im.atan2(re);
                if valid >= 2 {
                    coherences[i] = (re * re + im * im).sqrt() / valid as f32;
                }
            }
        }
        (powers, phases, coherences)
    }

    /// EMA the baselines toward `raw` and return the baseline-relative
    /// energies (≈0.125 = typical; >0.125 = elevated). The only
    /// mutating half of a signature pass — kept under `inner` by the
    /// caller, where it costs a few dozen float ops. `now` is the
    /// caller's wall clock: the EMA weight is α = 1 − e^(−dt/τ), so
    /// the baseline tracks elapsed time, not call count.
    fn apply_baseline(&mut self, raw: [f32; 10], now: Instant) -> [f32; 10] {
        let alpha = self
            .last_applied
            .map(|t| {
                let dt = now.saturating_duration_since(t).as_secs_f32();
                1.0 - (-dt / BASELINE_TAU_SECONDS).exp()
            })
            .unwrap_or(1.0); // no prior reading → adopt it whole
        self.last_applied = Some(now);
        let mut out = [0.0f32; 10];
        for (i, &r) in raw.iter().enumerate() {
            if !self.baseline_init {
                self.baseline[i] = r.max(1e-12);
            } else {
                self.baseline[i] += alpha * (r - self.baseline[i]);
            }
            let rel = r / self.baseline[i].max(1e-12);
            out[i] = rel.clamp(0.0, 8.0) / 8.0; // 0..1, 0.125 = typical
        }
        self.baseline_init = true;
        out
    }
}

/// Where real power draw is read from.
enum PowerSource {
    /// Energy counters in µJ, differentiated over wall time — RAPL
    /// powercap package/psys zones. Wraps at `max_energy_range_uj`.
    Counter {
        paths: Vec<PathBuf>,
        max_range_uj: u64,
        last_uj: Vec<u64>,
    },
    /// A sensor that reports power directly in µW — hwmon
    /// `power*_input`/`power*_average`, or `power_supply` `power_now`.
    Direct { path: PathBuf },
}

/// Wrap-safe energy-counter delta in µJ. `max_range` is the value the
/// counter wraps at (`max_energy_range_uj`); `u64::MAX` when the
/// driver doesn't publish one — the subtraction then saturates
/// honestly rather than wrapping silently.
fn energy_delta_uj(cur: u64, prev: u64, max_range: u64) -> u64 {
    if cur >= prev {
        cur - prev
    } else {
        max_range.saturating_sub(prev).saturating_add(cur)
    }
}

fn read_f64(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn dir_entries(path: &str) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default()
}

/// RAPL powercap zones: top-level zones only (one `:` in the name —
/// `intel-rapl:0` is a package, `intel-rapl:0:0` a subdomain). A
/// `psys` domain measures the whole platform and is preferred;
/// otherwise all package domains are summed (multi-socket = the
/// die's total draw).
fn discover_powercap() -> Option<PowerSource> {
    let mut psys = Vec::new();
    let mut pkg = Vec::new();
    for zone in dir_entries("/sys/class/powercap") {
        let Some(base) = zone.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        if base.matches(':').count() != 1 {
            continue;
        }
        let energy = zone.join("energy_uj");
        if read_u64(&energy).is_none() {
            continue; // unreadable (unprivileged) → not a source
        }
        let range = read_u64(&zone.join("max_energy_range_uj")).unwrap_or(u64::MAX);
        let name = std::fs::read_to_string(zone.join("name")).unwrap_or_default();
        if name.trim().starts_with("psys") {
            psys.push((energy, range));
        } else {
            pkg.push((energy, range));
        }
    }
    let pick = if psys.is_empty() { pkg } else { psys };
    if pick.is_empty() {
        return None;
    }
    let max_range_uj = pick.iter().map(|p| p.1).min().unwrap_or(u64::MAX);
    let paths: Vec<PathBuf> = pick.into_iter().map(|p| p.0).collect();
    let last_uj = paths.iter().filter_map(|p| read_u64(p)).collect();
    Some(PowerSource::Counter {
        paths,
        max_range_uj,
        last_uj,
    })
}

/// hwmon power sensors (`powerN_input` instantaneous preferred —
/// `powerN_average` can block ~10 ms in the read on fam15h_power).
/// CPU-class sensors (fam15h_power, zenpower, …) rank first: the
/// die's own draw is the quantity the carrier modulates.
///
/// A sensor that can only ever read zero is a dead channel —
/// fam15h_power's averaged register reports 0 µW forever on
/// northbridges that never implemented it, and a readable dead
/// register would otherwise shadow every live source. The first
/// candidate that actually reports nonzero draw wins; if all read
/// zero the CPU-first pick still stands so the channel reports its
/// honest zero rather than vanishing.
fn discover_hwmon_power() -> Option<PathBuf> {
    let mut cpu_class: Vec<PathBuf> = Vec::new();
    let mut any: Vec<PathBuf> = Vec::new();
    for h in dir_entries("/sys/class/hwmon") {
        let name = std::fs::read_to_string(h.join("name")).unwrap_or_default();
        let cpuish = name.contains("power") || name.contains("zen");
        let Some(hs) = h.to_str() else { continue };
        let mut input = None;
        let mut average = None;
        for e in dir_entries(hs) {
            let Some(n) = e.file_name().map(|x| x.to_string_lossy().into_owned()) else {
                continue;
            };
            if read_f64(&e).is_none() {
                continue;
            }
            if n.starts_with("power") && n.ends_with("_input") {
                input.get_or_insert(e);
            } else if n.starts_with("power") && n.ends_with("_average") {
                average.get_or_insert(e);
            }
        }
        // Instantaneous preferred — `power*_average` can block ~10 ms
        // inside the read on fam15h_power.
        if let Some(p) = input.or(average) {
            if cpuish {
                cpu_class.push(p);
            } else {
                any.push(p);
            }
        }
    }
    let mut cands = cpu_class;
    cands.extend(any);
    let first = cands.first()?.clone();
    // A candidate that reports real draw wins outright.
    if let Some(live) = cands.iter().find(|c| read_f64(c).is_some_and(|v| v > 0.0)) {
        return Some(live.clone());
    }
    // When every source reads zero, a single sample can't tell a
    // dead register from a genuinely idle rail by value alone —
    // prefer an instantaneous `power*_input` (a live sensor sampling
    // real draw that happens to be idle) over a `power*_average`
    // register: fam15h_power's averaged counter is 0 µW forever on
    // northbridges that never implemented it, while the battery's
    // instantaneous read is a real sensor reporting an honest idle.
    Some(
        cands
            .iter()
            .find(|c| c.to_string_lossy().ends_with("_input"))
            .cloned()
            .unwrap_or(first),
    )
}

/// Battery/adapter draw — system-level watts when the silicon
/// doesn't report its own.
fn discover_supply_power() -> Option<PathBuf> {
    dir_entries("/sys/class/power_supply")
        .into_iter()
        .map(|p| p.join("power_now"))
        .find(|p| read_f64(p).is_some())
}

/// Rail voltage — hwmon `in*_input` (mV) and `voltage_now` (µV)
/// candidates; the lowest live rail wins so a CPU-core-class rail
/// outranks the battery where the machine exposes one.
fn discover_voltage() -> Option<(PathBuf, f64)> {
    let mut cands: Vec<(PathBuf, f64)> = Vec::new();
    for h in dir_entries("/sys/class/hwmon") {
        let Some(hs) = h.to_str() else { continue };
        for e in dir_entries(hs) {
            let Some(n) = e.file_name().map(|x| x.to_string_lossy().into_owned()) else {
                continue;
            };
            if n.starts_with("in") && n.ends_with("_input") && read_f64(&e).is_some() {
                cands.push((e, 1.0e-3));
            }
        }
    }
    for p in dir_entries("/sys/class/power_supply") {
        let v = p.join("voltage_now");
        if read_f64(&v).is_some() {
            cands.push((v, 1.0e-6));
        }
    }
    cands
        .into_iter()
        .filter_map(|(p, s)| read_f64(&p).map(|raw| (raw * s, p, s)))
        .filter(|(v, _, _)| *v > 0.5) // a dead rail isn't a rail
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, p, s)| (p, s))
}

/// Supply current — `curr*_input` (mA) or `current_now` (µA).
fn discover_current() -> Option<(PathBuf, f64)> {
    for h in dir_entries("/sys/class/hwmon") {
        let Some(hs) = h.to_str() else { continue };
        for e in dir_entries(hs) {
            let Some(n) = e.file_name().map(|x| x.to_string_lossy().into_owned()) else {
                continue;
            };
            if n.starts_with("curr") && n.ends_with("_input") && read_f64(&e).is_some() {
                return Some((e, 1.0e-3));
            }
        }
    }
    dir_entries("/sys/class/power_supply")
        .into_iter()
        .map(|p| p.join("current_now"))
        .find(|p| read_f64(p).is_some())
        .map(|p| (p, 1.0e-6))
}

/// Parse `/proc/net/wireless`: each data line is
/// `iface: status link level noise nwid crypt frag retry misc beacon we`.
/// `level` and `noise` are dBm when negative (modern drivers);
/// values at or below `DBM_ABSENT_FLOOR` are driver sentinels for
/// "no reading", not real levels. Returns (level_dbm, noise_dbm)
/// for the strongest reported interface — one machine's radios
/// are one RF sense. Returns (any_iface_seen, level_dbm, noise_dbm).
fn parse_wireless_levels(content: &str) -> (bool, Option<f32>, Option<f32>) {
    let mut seen = false;
    let mut level_best: Option<f32> = None;
    let mut noise_out: Option<f32> = None;
    for line in content.lines().skip(2) {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        if name.trim().is_empty() {
            continue;
        }
        let t: Vec<&str> = rest.split_whitespace().collect();
        if t.len() < 4 {
            continue;
        }
        seen = true;
        let parse_dbm = |s: &str| s.trim_end_matches('.').parse::<f32>().ok();
        if let Some(level) = parse_dbm(t[2]).filter(|v| *v != 0.0) {
            if level_best.is_none_or(|b| level > b) {
                level_best = Some(level);
            }
        }
        if let Some(noise) = parse_dbm(t[3]).filter(|v| *v > DBM_ABSENT_FLOOR) {
            noise_out = Some(noise);
        }
    }
    (seen, level_best, noise_out)
}

/// WLAN interface names from `/proc/net/wireless` — the same data
/// lines the level parse reads.
fn discover_wlan_ifaces() -> Vec<String> {
    let Ok(content) = std::fs::read_to_string("/proc/net/wireless") else {
        return Vec::new();
    };
    content
        .lines()
        .skip(2)
        .filter_map(|l| l.split_once(':'))
        .map(|(n, _)| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .collect()
}

/// The (rx_bytes, tx_bytes) stat paths of one WLAN interface.
fn wlan_stat_paths(iface: &str) -> Option<(PathBuf, PathBuf)> {
    let base = PathBuf::from(format!("/sys/class/net/{iface}/statistics"));
    let rx = base.join("rx_bytes");
    let tx = base.join("tx_bytes");
    (read_u64(&rx).is_some() && read_u64(&tx).is_some()).then_some((rx, tx))
}

/// Sum rx+tx bytes over every WLAN stat path. 0 on no radio or an
/// unreadable counter — the presence mask carries which it is.
fn read_wlan_bytes(paths: &[(PathBuf, PathBuf)]) -> u64 {
    let mut total = 0u64;
    for (rx, tx) in paths {
        if let (Some(r), Some(t)) = (read_u64(rx), read_u64(tx)) {
            total = total.saturating_add(r).saturating_add(t);
        }
    }
    total
}

/// `/proc/interrupts` lines whose device names match a known
/// wireless driver: returns (total_irqs, any_line_matched). The
/// per-CPU count columns are summed; the trailing name region is
/// matched against `WIRELESS_IRQ_NAMES`. A Bluetooth radio on a
/// shared `xhci_hcd` line never matches by name — that share is
/// inside the aggregate count, honestly unattributed.
fn parse_wireless_intr_str(content: &str) -> (u64, bool) {
    let mut total = 0u64;
    let mut matched = false;
    for line in content.lines() {
        let Some((_, rest)) = line.trim_start().split_once(':') else {
            continue;
        };
        let mut counts = 0u64;
        let mut wireless = false;
        // Per-CPU count columns first, then controller/type/device
        // name tokens — the name region is where drivers appear.
        let mut in_names = false;
        for tok in rest.split_whitespace() {
            if !in_names {
                if let Ok(v) = tok.parse::<u64>() {
                    counts = counts.saturating_add(v);
                    continue;
                }
                in_names = true;
            }
            if !wireless && WIRELESS_IRQ_NAMES.iter().any(|n| tok.contains(n)) {
                wireless = true;
            }
        }
        if wireless {
            total = total.saturating_add(counts);
            matched = true;
        }
    }
    (total, matched)
}

fn read_wireless_intr() -> (u64, bool) {
    let Ok(content) = std::fs::read_to_string("/proc/interrupts") else {
        return (0, false);
    };
    parse_wireless_intr_str(&content)
}

/// Bluetooth HCI counters via `hciconfig`: RX/TX bytes and RX
/// events summed over every hci block. `None` where the tool or
/// the counters are absent — a machine without bluez still gets
/// the WLAN half of the channel.
fn read_bt_counters() -> Option<(u64, u64)> {
    let out = std::process::Command::new("hciconfig")
        .arg("-a")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut bytes = 0u64;
    let mut events = 0u64;
    let mut any = false;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("RX bytes:") {
            if let Some(v) = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
            {
                bytes = bytes.saturating_add(v);
                any = true;
            }
            if let Some(ev) = rest.split("events:").nth(1) {
                if let Some(v) = ev
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    events = events.saturating_add(v);
                }
            }
        }
        if let Some(rest) = t.strip_prefix("TX bytes:") {
            if let Some(v) = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<u64>().ok())
            {
                bytes = bytes.saturating_add(v);
                any = true;
            }
        }
    }
    any.then_some((bytes, events))
}

/// Mean ACL-link RSSI via `hcitool`: `con` lists active links,
/// `rssi` reads each. `None` when no links exist — the common
/// case, an absent channel rather than a zero field strength.
fn read_bt_link_rssi(hci: &str) -> Option<f32> {
    let con = std::process::Command::new("hcitool")
        .args(["-i", hci, "con"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&con.stdout);
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for line in text.lines() {
        let Some(addr) = line
            .split_whitespace()
            .nth(2)
            .filter(|_| line.contains("ACL"))
        else {
            continue;
        };
        let Ok(out) = std::process::Command::new("hcitool")
            .args(["-i", hci, "rssi", addr])
            .output()
        else {
            continue;
        };
        let r = String::from_utf8_lossy(&out.stdout);
        if let Some(v) = r
            .split("RSSI return value:")
            .nth(1)
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| *v != 0.0)
        {
            sum += v;
            n += 1;
        }
    }
    (n > 0).then_some(sum / n as f32)
}

/// nl80211 station telemetry — `iw dev <iface> station dump`. A
/// strictly better instrument than `/proc/net/wireless` where it
/// exists: `signal avg` is the driver's per-frame average, and the
/// `tx retries`/`tx failed`/`beacon loss` counters are the link's
/// own strain ledger — retransmissions are airtime the AP made the
/// radio spend twice, failed frames are retries exhausted, beacon
/// loss is the AP's heartbeat going unheard.
#[derive(Default)]
struct IwStation {
    /// `signal:` — instantaneous dBm (first token of `-56 [-56] dBm`).
    signal_dbm: Option<f32>,
    /// `signal avg:` — driver-averaged dBm, the better RSSI.
    signal_avg_dbm: Option<f32>,
    tx_retries: u64,
    tx_failed: u64,
    beacon_loss: u64,
    /// PHY link-adaptation bitrates (MBit/s) — channel quality the
    /// radio negotiated, not a throughput estimate.
    tx_mbps: Option<f32>,
    rx_mbps: Option<f32>,
}

fn parse_iw_station(content: &str) -> Option<IwStation> {
    let mut st = IwStation::default();
    let mut saw_station = false;
    for line in content.lines() {
        // The "Station aa:bb:…" header has colons inside the MAC —
        // detect it by prefix, never by split.
        if line.starts_with("Station ") {
            saw_station = true;
            continue;
        }
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let tok = rest.split_whitespace().next().unwrap_or("");
        match key {
            "signal" => st.signal_dbm = tok.parse().ok(),
            "signal avg" => st.signal_avg_dbm = tok.parse().ok(),
            "tx retries" => st.tx_retries = tok.parse().unwrap_or(0),
            "tx failed" => st.tx_failed = tok.parse().unwrap_or(0),
            "beacon loss" => st.beacon_loss = tok.parse().unwrap_or(0),
            "tx bitrate" => st.tx_mbps = tok.parse().ok(),
            "rx bitrate" => st.rx_mbps = tok.parse().ok(),
            _ => {}
        }
    }
    saw_station.then_some(st)
}

/// `iw dev <iface> station dump` — unprivileged on an associated
/// link; fails (None) when iw is absent or no station exists.
fn read_iw_station(iface: &str) -> Option<IwStation> {
    let out = std::process::Command::new("iw")
        .args(["dev", iface, "station", "dump"])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| parse_iw_station(&String::from_utf8_lossy(&out.stdout)))
        .flatten()
}

/// The radio sense — the first *exteroceptive* physical channel.
/// WLAN byte counters are sampled in the 250 Hz loop (the flux
/// envelope wants full resolution); RSSI, noise, wireless IRQs,
/// and the Bluetooth counters ride the sampler thread — they are
/// slow or subprocess-backed, and a blocking read inside the
/// 250 Hz loop would contaminate the jitter channel.
struct Radio {
    /// WLAN stat paths + the last summed counter (probe-loop read).
    wlan_stats: Vec<(PathBuf, PathBuf)>,
    last_bytes: u64,
    bytes_s: f32,
    /// Latest sampler reads.
    rssi_dbm: f32,
    noise_dbm: f32,
    rf_hz: f32, // EMA of the /proc/net/wireless read rate
    irq_last: u64,
    irq_rate: f32,
    bt_last: (u64, u64),
    bt_bytes_s: f32,
    bt_events_s: f32,
    bt_rssi_dbm: f32,
    /// The hci device name for `hcitool -i` (hci0…), if any.
    hci_name: Option<String>,
    /// The wlan iface for `iw dev … station dump` (wlo1…), if any.
    wlan_iface: Option<String>,
    /// nl80211 station telemetry — link strain counters and PHY
    /// link adaptation, differentiated over the sample interval.
    iw_last: (u64, u64, u64), // (tx_retries, tx_failed, beacon_loss)
    retry_s: f32,
    fail_s: f32,
    beacon_loss_s: f32,
    tx_mbps: f32,
    rx_mbps: f32,
    iw_hz: f32,       // EMA of the iw-station read rate
    iw_hz_init: bool, // first interval seeds the EMA, not a drift from 0
    /// When `signal avg`/`signal` last fed `rssi_dbm` — while this
    /// is fresh, iw owns the channel and /proc must not interleave
    /// (two instruments at different cadences would beat in the ring).
    iw_rssi_at: Option<Instant>,
    has_wlan: bool,
    has_rssi: bool,
    has_noise: bool,
    has_wlan_irq: bool,
    has_bt: bool,
    has_bt_counters: bool,
    has_bt_rssi: bool,
    has_iw: bool,
    last_rf: Option<Instant>,
    last_bt: Option<Instant>,
    last_iw: Option<Instant>,
    hz_init: bool,
}

impl Radio {
    fn discover() -> Self {
        let wlan_ifaces: Vec<String> = discover_wlan_ifaces()
            .into_iter()
            .filter(|i| wlan_stat_paths(i).is_some())
            .collect();
        let wlan_stats: Vec<(PathBuf, PathBuf)> = wlan_ifaces
            .iter()
            .filter_map(|i| wlan_stat_paths(i))
            .collect();
        let hci_name = dir_entries("/sys/class/bluetooth")
            .into_iter()
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .find(|n| n.starts_with("hci"));
        let (irq_last, has_wlan_irq) = read_wireless_intr();
        let has_wlan = !wlan_stats.is_empty();
        Self {
            last_bytes: read_wlan_bytes(&wlan_stats),
            wlan_stats,
            bytes_s: 0.0,
            rssi_dbm: 0.0,
            noise_dbm: 0.0,
            rf_hz: 0.0,
            irq_last,
            irq_rate: 0.0,
            bt_last: (0, 0),
            bt_bytes_s: 0.0,
            bt_events_s: 0.0,
            bt_rssi_dbm: 0.0,
            has_wlan_irq,
            has_wlan,
            has_rssi: false,
            has_noise: false,
            has_bt: hci_name.is_some(),
            has_bt_counters: false,
            has_bt_rssi: false,
            has_iw: false,
            hci_name,
            wlan_iface: wlan_ifaces.first().cloned(),
            iw_last: (0, 0, 0),
            retry_s: 0.0,
            fail_s: 0.0,
            beacon_loss_s: 0.0,
            tx_mbps: 0.0,
            rx_mbps: 0.0,
            iw_hz: 0.0,
            iw_hz_init: false,
            iw_rssi_at: None,
            last_rf: None,
            last_bt: None,
            last_iw: None,
            hz_init: false,
        }
    }

    /// Fold one sampler batch into state — rate deltas, EMAs, and
    /// presence bits land here; the reads happened unlocked.
    fn apply(&mut self, r: RadioReading, now: Instant) {
        if let Some(rf) = r.wireless {
            let dt = self
                .last_rf
                .map(|t| (now - t).as_secs_f32())
                .filter(|d| *d > 1e-3);
            self.has_wlan = true;
            // /proc's level fills the RSSI channel only while the
            // iw station source is absent or stale — two instruments
            // writing at different cadences would interleave into a
            // beat in the envelope ring.
            let iw_fresh = self
                .iw_rssi_at
                .is_some_and(|t| (now - t) < Duration::from_secs(3));
            if !iw_fresh {
                if let Some(level) = rf.level {
                    self.rssi_dbm = level;
                    self.has_rssi = true;
                }
            }
            if let Some(noise) = rf.noise {
                self.noise_dbm = noise;
                self.has_noise = true;
            }
            if let Some(dt) = dt {
                let rate = 1.0 / dt;
                if !self.hz_init {
                    self.rf_hz = rate;
                    self.hz_init = true;
                } else {
                    let alpha = 1.0 - (-dt / ELEC_EMA_TAU as f32).exp();
                    self.rf_hz += alpha * (rate - self.rf_hz);
                }
            }
            self.last_rf = Some(now);
        }
        if let Some((bytes, events)) = r.bt_counters {
            self.has_bt_counters = true;
            if let Some(dt) = self
                .last_bt
                .map(|t| (now - t).as_secs_f32())
                .filter(|d| *d > 1e-3)
            {
                self.bt_bytes_s = bytes.saturating_sub(self.bt_last.0) as f32 / dt;
                self.bt_events_s = events.saturating_sub(self.bt_last.1) as f32 / dt;
            }
            self.bt_last = (bytes, events);
            self.last_bt = Some(now);
        }
        if let Some(rssi) = r.bt_rssi {
            self.bt_rssi_dbm = rssi;
            self.has_bt_rssi = true;
        }
        if let Some(st) = r.station {
            self.has_iw = true;
            // The driver's per-frame signal average is a strictly
            // better RSSI than /proc/net/wireless's — same channel,
            // better instrument, so it supersedes when present.
            if let Some(avg) = st.signal_avg_dbm.or(st.signal_dbm) {
                self.rssi_dbm = avg;
                self.has_rssi = true;
                self.iw_rssi_at = Some(now);
            }
            if let Some(dt) = self
                .last_iw
                .map(|t| (now - t).as_secs_f32())
                .filter(|d| *d > 1e-3)
            {
                self.retry_s = st.tx_retries.saturating_sub(self.iw_last.0) as f32 / dt;
                self.fail_s = st.tx_failed.saturating_sub(self.iw_last.1) as f32 / dt;
                self.beacon_loss_s = st.beacon_loss.saturating_sub(self.iw_last.2) as f32 / dt;
                let rate = 1.0 / dt;
                if !self.iw_hz_init {
                    self.iw_hz = rate;
                    self.iw_hz_init = true;
                } else {
                    let alpha = 1.0 - (-dt / ELEC_EMA_TAU as f32).exp();
                    self.iw_hz += alpha * (rate - self.iw_hz);
                }
            }
            self.iw_last = (st.tx_retries, st.tx_failed, st.beacon_loss);
            if let Some(m) = st.tx_mbps {
                self.tx_mbps = m;
            }
            if let Some(m) = st.rx_mbps {
                self.rx_mbps = m;
            }
            self.last_iw = Some(now);
        }
    }

    /// The presence mask byte for the wire.
    fn present_mask(&self) -> u8 {
        (self.has_wlan as u8)
            | ((self.has_rssi as u8) << 1)
            | ((self.has_noise as u8) << 2)
            | ((self.has_wlan_irq as u8) << 3)
            | ((self.has_bt as u8) << 4)
            | ((self.has_bt_counters as u8) << 5)
            | ((self.has_bt_rssi as u8) << 6)
            | ((self.has_iw as u8) << 7)
    }
}

/// One unlocked radio read batch — raw values only.
struct RadioReading {
    wireless: Option<WirelessLevels>,
    bt_counters: Option<(u64, u64)>,
    bt_rssi: Option<f32>,
    station: Option<IwStation>,
}

struct WirelessLevels {
    level: Option<f32>,
    noise: Option<f32>,
}

/// Read `/proc/net/wireless` once: `Some` when it listed any
/// interface — even if that interface's level/noise are absent.
fn read_wireless() -> Option<WirelessLevels> {
    let content = std::fs::read_to_string("/proc/net/wireless").ok()?;
    let (seen, level, noise) = parse_wireless_levels(&content);
    seen.then_some(WirelessLevels { level, noise })
}

/// The electrical sense: real power, voltage, and current where the
/// machine's own sensors report them. Discovery runs once at probe
/// start; a dedicated sampler thread does the reading because some
/// sensors block inside the read — a blocking read inside the 250 Hz
/// probe loop would contaminate the jitter channel with our own
/// instrumentation. Fresh values publish under the mutex; the probe
/// loop folds the latest watts into its ring each tick (zero-order
/// hold), so the ring's band content is real up to the sensor's own
/// update rate — reported as `power_hz`.
struct Electrical {
    power: Option<PowerSource>,
    volts: Option<(PathBuf, f64)>,
    amps: Option<(PathBuf, f64)>,
    watts: f32,      // latest read
    watts_mean: f32, // EMA
    volts_v: f32,
    amps_a: f32,
    power_hz: f32, // EMA of fresh-read rate
    last_read: Option<Instant>,
    /// False until the first computable sample — the first reading
    /// *is* the estimate, not a 10%-weighted step from zero.
    ema_init: bool,
}

impl Electrical {
    fn discover() -> Self {
        let power = discover_powercap().or_else(|| {
            let hwmon = discover_hwmon_power();
            let supply = discover_supply_power();
            // A dead hwmon pick — readable but only ever zero — must
            // not shadow a live supply rail.
            let pick = match (&hwmon, &supply) {
                (Some(h), Some(s))
                    if !read_f64(h).is_some_and(|v| v > 0.0)
                        && read_f64(s).is_some_and(|v| v > 0.0) =>
                {
                    Some(s.clone())
                }
                _ => hwmon.or(supply),
            };
            pick.map(|path| PowerSource::Direct { path })
        });
        Self {
            power,
            volts: discover_voltage(),
            amps: discover_current(),
            watts: 0.0,
            watts_mean: 0.0,
            volts_v: 0.0,
            amps_a: 0.0,
            power_hz: 0.0,
            last_read: None,
            ema_init: false,
        }
    }

    /// The sensor path set, cloned once for the sampler thread. Reads
    /// happen with NO lock held — several sensors block inside the
    /// file read (~10 ms on fam15h_power, arbitrarily longer on a
    /// wedged hwmon), and holding `inner` across them is what starved
    /// the IPC signature path for tens of seconds.
    fn sensors(&self) -> ElecSensors {
        let (counters, direct) = match &self.power {
            Some(PowerSource::Counter {
                paths,
                max_range_uj,
                ..
            }) => (Some((paths.clone(), *max_range_uj)), None),
            Some(PowerSource::Direct { path }) => (None, Some(path.clone())),
            None => (None, None),
        };
        ElecSensors {
            counters,
            direct,
            volts: self.volts.clone(),
            amps: self.amps.clone(),
        }
    }

    /// Fold one unlocked read batch into the state — the locked half
    /// of a sample. Counter deltas, EMAs, and `last_read` all land
    /// here; the reads themselves already happened.
    fn apply(&mut self, r: ElecReading, now: Instant) {
        let dt = self
            .last_read
            .map(|t| (now - t).as_secs_f64())
            .filter(|d| *d > 1e-6);
        // The EMA weight from the real interval: α = 1 − e^(−dt/τ).
        // No fresh reading → no aging; a stalled sensor's next read
        // moves the mean by the gap it actually spanned.
        let alpha = dt.map(|d| 1.0 - (-d / ELEC_EMA_TAU).exp() as f32);
        let init = !self.ema_init;
        match &mut self.power {
            Some(PowerSource::Counter {
                max_range_uj,
                last_uj,
                ..
            }) => {
                let mut delta = 0u64;
                let mut any = false;
                for (i, e) in r.counters.iter().enumerate() {
                    if let Some(e) = e {
                        any = true;
                        if let Some(prev) = last_uj.get_mut(i) {
                            delta = delta.saturating_add(energy_delta_uj(*e, *prev, *max_range_uj));
                            *prev = *e;
                        }
                    }
                }
                if any {
                    if let Some(dt) = dt {
                        self.watts = (delta as f64 * 1.0e-6 / dt) as f32;
                        let rate = 1.0 / dt as f32;
                        if init {
                            self.watts_mean = self.watts;
                            self.power_hz = rate;
                            self.ema_init = true;
                        } else if let Some(a) = alpha {
                            self.watts_mean += a * (self.watts - self.watts_mean);
                            self.power_hz += a * (rate - self.power_hz);
                        }
                    }
                }
            }
            Some(PowerSource::Direct { .. }) => {
                if let Some(raw) = r.direct {
                    self.watts = (raw * 1.0e-6) as f32;
                    let rate = dt.map(|d| 1.0 / d as f32);
                    if init {
                        self.watts_mean = self.watts;
                        if let Some(rate) = rate {
                            self.power_hz = rate;
                        }
                        self.ema_init = true;
                    } else if let (Some(a), Some(rate)) = (alpha, rate) {
                        self.watts_mean += a * (self.watts - self.watts_mean);
                        self.power_hz += a * (rate - self.power_hz);
                    } else if let Some(a) = alpha {
                        self.watts_mean += a * (self.watts - self.watts_mean);
                    }
                }
            }
            None => {}
        }
        if let Some((_, scale)) = &self.volts {
            if let Some(raw) = r.volts {
                self.volts_v = (raw * scale) as f32;
            }
        }
        if let Some((_, scale)) = &self.amps {
            if let Some(raw) = r.amps {
                self.amps_a = (raw * scale) as f32;
            }
        }
        self.last_read = Some(now);
    }
}

/// The sensor paths, cloned out of `inner` once at sampler start so
/// the per-tick file reads never hold the mutex.
struct ElecSensors {
    counters: Option<(Vec<PathBuf>, u64)>,
    direct: Option<PathBuf>,
    volts: Option<(PathBuf, f64)>,
    amps: Option<(PathBuf, f64)>,
}

/// One unlocked read batch — raw values only, no state touched.
struct ElecReading {
    counters: Vec<Option<u64>>,
    direct: Option<f64>,
    volts: Option<f64>,
    amps: Option<f64>,
}

impl ElecSensors {
    /// Read every configured sensor file. Called with no lock held —
    /// a blocking read can stall this thread, never the shared state.
    fn read_all(&self) -> ElecReading {
        ElecReading {
            counters: self
                .counters
                .as_ref()
                .map(|(paths, _)| paths.iter().map(|p| read_u64(p)).collect())
                .unwrap_or_default(),
            direct: self.direct.as_ref().and_then(|p| read_f64(p)),
            volts: self.volts.as_ref().and_then(|(p, _)| read_f64(p)),
            amps: self.amps.as_ref().and_then(|(p, _)| read_f64(p)),
        }
    }
}

/// Internal probe state behind the shared mutex.
struct ProbeInner {
    throughput: Channel,
    jitter: Channel,
    /// The generator's own phase at each throughput sample — the
    /// reference track the measured response is demodulated against.
    carrier_ref: Channel,
    /// Execution seconds per work slice — the channel the carrier
    /// physically modulates: `iters` is the plan, `work_secs` is what
    /// the silicon did. Its phase vs. the reference is the real
    /// response lag (scheduler + governor + VRM settling).
    duty: Channel,
    /// Measured power draw in watts (zero-order-held between sensor
    /// reads) — the carrier's literal electrical shadow. Its lag vs.
    /// the reference is the answer to "does real power draw move
    /// with the body's rhythm," measured rather than assumed.
    power: Channel,
    /// The electrical sense — sensor discovery + latest reads.
    elec: Electrical,
    /// The radio sense — WLAN flux (probe-loop counter deltas),
    /// RSSI envelope, wireless IRQ flux, and HCI counters.
    radio: Radio,
    /// WLAN rx+tx bytes/s envelope — radio traffic at the full
    /// probe rate.
    rf_flux: Channel,
    /// Received-signal-strength envelope (dBm, zero-order-held
    /// between sampler reads) — the room's radio weather.
    rf_rssi: Channel,
    last_tis: (u64, u64),
    pstate_rate: f32,
    last_intr: u64,
    irq_rate: f32,
    last_slow: Instant,
    throughput_mean: f32,
    jitter_sq_mean: f32,
    jitter_mean: f32,
    /// Slow EMA of the jitter level — the machine's own timing
    /// climate, so the bath measures *excess* turbulence rather
    /// than this host's ordinary scheduling weather (the same
    /// baseline-relative discipline as the band energies).
    jitter_cv_climate: f32,
    jitter_cv_climate_init: bool,
    last_bath: Option<Instant>,
    /// The open system-bus connection owning the rtkit grant —
    /// the elevation lives only as long as this client does.
    bus_hold: Option<std::os::unix::net::UnixStream>,
    carrier_freq: f32,
    /// The carrier generator's instantaneous phase (radians).
    carrier_phase: f32,
    /// Modulation depth of the carrier's real work (0 = unmodulated).
    carrier_depth: f32,
    /// Frequency override (Hz). 0 = derive from the body's phase;
    /// the ring-down experiment drives arbitrary frequencies —
    /// including infraslow rates where the thermal field can
    /// actually sustain an oscillation.
    carrier_freq_override: f32,
    /// The parasympathetic envelope — a slow wave riding carrier
    /// *depth* itself (SET_FIELD_PULSE). Armed: effective depth is
    /// `pulse_base + pulse_mod_depth * sin(pulse_phase)`, the phase
    /// integrated on the probe's own clock so the wave runs whether
    /// or not the mind ticks. The RSA analog: vagal signature is
    /// the pulse's slow wax and wane, never a pinned tempo — the
    /// carrier rate stays the phase's own either way.
    pulse_armed: bool,
    pulse_base: f32,
    pulse_mod_freq: f32,
    pulse_mod_depth: f32,
    pulse_phase: f32,
}

/// The running probe. `spawn`ed once by the daemon; `signature` is
/// called by the IPC handler at the mind's cadence.
pub struct FieldProbe {
    inner: Mutex<ProbeInner>,
    /// Serializes a signature's snapshot→compute→apply cycle so two
    /// concurrent readers can't interleave their baseline write-backs.
    /// Uncontended in practice — held for the resonator passes too, so
    /// `inner` stays µs-held while the math runs lock-free.
    sig_lock: Mutex<()>,
    running: AtomicBool,
    stop: AtomicBool,
}

/// Process-shared probe handle. Started in `daemon main`, read by
/// the IPC handler — mirrors how `interoception` shares state.
static FIELD_PROBE: OnceLock<std::sync::Arc<FieldProbe>> = OnceLock::new();

/// Ask RealtimeKit to lift this thread out of bulk SCHED_OTHER
/// competition — the unprivileged scheduling boost (rtkit-daemon
/// runs as root and grants down to its advertised MinNiceLevel for
/// same-uid callers; real SCHED_FIFO is policy-denied on typical
/// desktops). A 250 Hz loop that loses its slice measures its own
/// starvation as substrate jitter and heats the field with it —
/// asking first keeps the bath honest about the machine's noise
/// rather than the thread's run-queue position. The request is
/// made over a persistent system-bus connection kept in `inner`:
/// rtkit revokes a grant whose requesting client has left the bus,
/// which is why a one-shot `dbus-send` cannot do this job. An
/// absent service or a refusal leaves scheduling as it was —
/// degraded jitter, never a failure.
fn request_scheduling_boost(inner: &Mutex<ProbeInner>) {
    let pid = std::process::id() as u64;
    // SAFETY: SYS_gettid takes no arguments and cannot fail — the
    // raw syscall always returns the calling thread's id.
    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as u64;
    match super::rtkit::grant_high_priority(pid, tid) {
        Ok((nice, conn)) => {
            inner.lock().unwrap_or_else(|e| e.into_inner()).bus_hold = Some(conn);
            eprintln!("[field-probe] rtkit granted nice {nice}");
        }
        Err(e) => eprintln!("[field-probe] rtkit boost unavailable: {e}"),
    }
}

/// Start the probe thread. `phase_source` is polled ~4×/s for the
/// body's phase name; it sets the carrier frequency. Idempotent:
/// only the first call spawns the thread.
pub fn start<F>(phase_source: F)
where
    F: Fn() -> &'static str + Send + 'static,
{
    let probe = FIELD_PROBE
        .get_or_init(|| std::sync::Arc::new(FieldProbe::new()))
        .clone();
    if probe.running.swap(true, Ordering::SeqCst) {
        return;
    }
    let sampler = probe.clone();
    std::thread::Builder::new()
        .name("field-probe".into())
        .spawn(move || probe.run(phase_source))
        .expect("field-probe thread");
    std::thread::Builder::new()
        .name("field-sampler".into())
        .spawn(move || sampler.sample_electrical())
        .expect("field-sampler thread");
}

/// The shared probe, for the IPC handler. `None` until `start` runs.
pub fn shared() -> Option<&'static std::sync::Arc<FieldProbe>> {
    FIELD_PROBE.get()
}

/// P-state transitions since boot, summed over cores from
/// `cpufreq/stats/total_trans` — the silicon's real frequency-jump
/// events. 0 when cpufreq stats are unavailable.
fn read_pstate_transitions() -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return 0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !(name.len() > 3
            && name.starts_with("cpu")
            && name[3..].chars().all(|c| c.is_ascii_digit()))
        {
            continue;
        }
        let Ok(s) = std::fs::read_to_string(entry.path().join("cpufreq/stats/total_trans")) else {
            continue;
        };
        if let Ok(v) = s.trim().parse::<u64>() {
            total = total.saturating_add(v);
        }
    }
    total
}

/// Read `/proc/stat`'s aggregate `intr` counter (interrupts since boot).
fn read_intr_total() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/stat") else {
        return 0;
    };
    s.lines()
        .find(|l| l.starts_with("intr "))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

impl FieldProbe {
    fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            sig_lock: Mutex::new(()),
            inner: Mutex::new(ProbeInner {
                throughput: Channel::new(),
                jitter: Channel::new(),
                carrier_ref: Channel::new(),
                duty: Channel::new(),
                power: Channel::new(),
                elec: Electrical::discover(),
                radio: Radio::discover(),
                rf_flux: Channel::new(),
                rf_rssi: Channel::new(),
                last_tis: (0, 0),
                pstate_rate: 0.0,
                last_intr: 0,
                irq_rate: 0.0,
                last_slow: Instant::now(),
                throughput_mean: 0.0,
                jitter_sq_mean: 0.0,
                jitter_mean: 0.0,
                jitter_cv_climate: 0.0,
                jitter_cv_climate_init: false,
                last_bath: None,
                bus_hold: None,
                carrier_freq: CARRIER_WAKE_HZ,
                carrier_phase: 0.0,
                carrier_depth: 0.60,
                carrier_freq_override: 0.0,
                pulse_armed: false,
                pulse_base: 0.0,
                pulse_mod_freq: 0.0,
                pulse_mod_depth: 0.0,
                pulse_phase: 0.0,
            }),
            stop: AtomicBool::new(false),
        }
    }

    /// Carrier frequency for the body's phase name. Wake ≈ alpha,
    /// drowsy/REM ≈ theta, NREM ≈ delta — the body's actual state
    /// carried on real work. Drowsy is the N1 descent: alpha drops
    /// out and theta dominates the resting mind (same band REM
    /// rides). Unknown/absent phase → wake rate.
    fn carrier_for_phase(phase: &str) -> f32 {
        match phase {
            "nrem" => CARRIER_NREM_HZ,
            "rem" => CARRIER_REM_HZ,
            "drowsy" => CARRIER_DROWSY_HZ,
            _ => CARRIER_WAKE_HZ,
        }
    }

    /// Set the carrier's real-work modulation depth and optional
    /// frequency override (SET_FIELD_CARRIER). `freq <= 0` hands the
    /// rhythm back to the body's phase. Clamped: depth 0..1, freq to
    /// the resolvable band.
    pub fn set_carrier(&self, depth: f32, freq: f32) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.carrier_depth = depth.clamp(0.0, 1.0);
        g.carrier_freq_override = if freq > 0.0 {
            freq.clamp(0.1, 100.0)
        } else {
            0.0
        };
    }

    /// Arm or disarm the parasympathetic pulse (SET_FIELD_PULSE): a
    /// slow envelope breathing carrier depth around `base` at
    /// `mod_freq`, integrated on the probe's own clock — the body
    /// owns the wave's timing the way it owns every rhythm here.
    /// `mod_freq <= 0` disarms; depth returns to the static
    /// `carrier_depth` set-point untouched. Clamped: base/mod_depth
    /// 0..1, mod_freq to the infraslow band (0.01..4 Hz — below the
    /// carrier and above the bath's own weather).
    pub fn set_pulse(&self, base: f32, mod_freq: f32, mod_depth: f32) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.pulse_armed = mod_freq > 0.0;
        if g.pulse_armed {
            g.pulse_base = base.clamp(0.0, 1.0);
            g.pulse_mod_freq = mod_freq.clamp(0.01, 4.0);
            g.pulse_mod_depth = mod_depth.clamp(0.0, 1.0);
        }
    }

    /// The field sampler's own loop — a separate thread because
    /// some sensors block inside the read (~10 ms on fam15h_power)
    /// and a blocked 250 Hz loop would poison the jitter channel
    /// with our own instrumentation. The file reads run with NO lock
    /// held — only the tiny apply step touches `inner` — so a slow
    /// or wedged sensor can never starve `signature()` on the IPC
    /// path. Publishes fresh values under the mutex; the probe loop
    /// holds the latest in its ring. Radio telemetry (RSSI, HCI
    /// counters, link RSSI, iw station dumps) rides the same thread
    /// at slower cadences.
    fn sample_electrical(&self) {
        let (sensors, hci_name, wlan_iface) = {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            (
                g.elec.sensors(),
                g.radio.hci_name.clone(),
                g.radio.wlan_iface.clone(),
            )
        };
        let mut iter = 0u32;
        while !self.stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            let reading = sensors.read_all();
            // Radio telemetry rides the same unlocked-reads thread at
            // its own cadences: /proc/net/wireless ~4 Hz, hciconfig,
            // hcitool, and iw subprocesses ~1 Hz. A wedged read stalls
            // this thread, never the probe loop or signature().
            iter = iter.wrapping_add(1);
            let mut radio = RadioReading {
                wireless: None,
                bt_counters: None,
                bt_rssi: None,
                station: None,
            };
            if iter % RF_SAMPLE_DIV == 0 {
                radio.wireless = read_wireless();
            }
            if iter % BT_SAMPLE_DIV == 0 {
                radio.bt_counters = read_bt_counters();
                radio.bt_rssi = hci_name.as_deref().and_then(read_bt_link_rssi);
                radio.station = wlan_iface.as_deref().and_then(read_iw_station);
            }
            {
                let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.elec.apply(reading, t0);
                g.radio.apply(radio, t0);
            }
            let spent = t0.elapsed();
            if spent < ELEC_SAMPLE_PERIOD {
                std::thread::sleep(ELEC_SAMPLE_PERIOD - spent);
            }
        }
    }

    fn run(&self, phase_source: impl Fn() -> &'static str) {
        request_scheduling_boost(&self.inner);
        let nominal = PROBE_PERIOD.as_secs_f32();
        let mut prev = Instant::now();
        // WLAN byte-counter paths, cloned once — the flux channel
        // wants counter deltas at the full probe rate, read
        // unlocked like the rest of the loop's file I/O.
        let wlan_stats = {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.radio.wlan_stats.clone()
        };
        while !self.stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            // Jitter: how long since the last iteration actually ran.
            let jitter = (t0 - prev).as_secs_f32() - nominal;
            prev = t0;

            // Carrier-modulated work slice: the body's phase (or an
            // override) sets the rhythm; depth scales how much real
            // work — real electron flow — swings this window.
            let carrier_f;
            let slow_due;
            {
                let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                slow_due = g.last_slow.elapsed() >= Duration::from_millis(250);
                if slow_due {
                    g.carrier_freq = if g.carrier_freq_override > 0.0 {
                        g.carrier_freq_override
                    } else {
                        Self::carrier_for_phase(phase_source())
                    };
                }
                carrier_f = g.carrier_freq;
            }

            // P-state/interrupt counters: a walk of /sys cpufreq stats
            // plus a /proc/stat parse — real file I/O, run with the
            // mutex RELEASED so a slow sysfs read can't starve the
            // IPC signature path (or this loop's own sampling).
            let slow_reads = slow_due.then(|| {
                (
                    read_pstate_transitions(),
                    read_intr_total(),
                    read_wireless_intr(),
                )
            });
            let rf_bytes_now = if wlan_stats.is_empty() {
                None
            } else {
                Some(read_wlan_bytes(&wlan_stats))
            };

            let carrier_phase;
            let depth;
            {
                let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if let Some((trans, intr, wintr)) = slow_reads {
                    let dt = g.last_slow.elapsed().as_secs_f32().max(1e-3);
                    if g.last_tis.1 > 0 {
                        g.pstate_rate = trans.saturating_sub(g.last_tis.1) as f32 / dt;
                        g.irq_rate = intr.saturating_sub(g.last_intr) as f32 / dt;
                        g.radio.irq_rate = wintr.0.saturating_sub(g.radio.irq_last) as f32 / dt;
                    }
                    g.last_tis = (0, trans);
                    g.last_intr = intr;
                    g.radio.irq_last = wintr.0;
                    g.radio.has_wlan_irq |= wintr.1;
                    g.last_slow = Instant::now();
                }
                g.carrier_phase += 2.0 * std::f32::consts::PI * carrier_f * nominal;
                g.carrier_phase = g.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
                carrier_phase = g.carrier_phase;
                depth = if g.pulse_armed {
                    // The pulse's own clock — the wave breathes on the
                    // body's time even when the mind isn't ticking.
                    g.pulse_phase += 2.0 * std::f32::consts::PI * g.pulse_mod_freq * nominal;
                    g.pulse_phase = g.pulse_phase.rem_euclid(2.0 * std::f32::consts::PI);
                    (g.pulse_base + g.pulse_mod_depth * g.pulse_phase.sin()).clamp(0.0, 1.0)
                } else {
                    g.carrier_depth
                };
            }
            let modulation = 1.0 + depth * carrier_phase.sin();
            let iters = (WORK_SLICE_ITERS as f32 * modulation) as u32;

            let w0 = Instant::now();
            let acc = work_slice(iters);
            let work_secs = w0.elapsed().as_secs_f32().max(1e-6);
            std::hint::black_box(acc);
            let throughput = iters as f32 / work_secs;

            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.throughput.push(throughput);
            g.jitter.push(jitter);
            g.carrier_ref.push(carrier_phase);
            g.duty.push(work_secs);
            // Latest measured watts, zero-order-held between sensor
            // reads — the ring's power-band content is real up to
            // the sensor's own rate (`power_sensor_hz` on the wire).
            let watts = g.elec.watts;
            g.power.push(watts);
            // Radio flux: byte-counter delta over the real period —
            // the RF traffic envelope at full probe resolution.
            let rf_rate = match rf_bytes_now {
                Some(cur) => {
                    let d = cur.saturating_sub(g.radio.last_bytes) as f32;
                    g.radio.last_bytes = cur;
                    d / nominal
                }
                None => 0.0,
            };
            g.radio.bytes_s = rf_rate;
            g.rf_flux.push(rf_rate);
            // RSSI held between sampler reads — the NIC's own
            // field-strength measure, slow by physics.
            let dbm = if g.radio.has_rssi {
                g.radio.rssi_dbm
            } else {
                0.0
            };
            g.rf_rssi.push(dbm);
            // Channel stats: throughput mean and jitter coefficient of
            // variation, EMA'd so the signature is stable.
            g.throughput_mean += 0.01 * (throughput - g.throughput_mean);
            g.jitter_mean += 0.01 * (jitter - g.jitter_mean);
            g.jitter_sq_mean += 0.01 * (jitter * jitter - g.jitter_sq_mean);
            drop(g);

            // Sleep the rest of the period.
            let spent = t0.elapsed();
            if spent < PROBE_PERIOD {
                std::thread::sleep(PROBE_PERIOD - spent);
            }
        }
    }

    /// Synchronously demodulate a measured channel against the
    /// carrier reference track: returns (lag_radians, coherence,
    /// amplitude).
    ///
    /// For a response `x_i = A·sin(ref_i − L)` trailing the carrier
    /// by L radians, correlating against sin/cos of the reference
    /// gives `atan2(S, C) = π/2 + L`, so `lag = angle − π/2` — the
    /// positive lag, in radians, by which the substrate's real
    /// response trails the generated carrier. `coherence` is
    /// |Σ x·e^{i·ref}| / Σ|x| — the fraction of the signal that moves
    /// with the carrier (≈π/4 at most for a pure tone, since |sin|
    /// averages 2/π). `amplitude` is 2|Σ x·e^{i·ref}|/N — the
    /// recovered magnitude of the component coherent with the
    /// reference, in the channel's own units. A lag computed from an
    /// incoherent signal is noise; the client treats low coherence
    /// as a weak lock, which is why it travels on the wire too.
    fn demodulate(x: &[f32], ref_phase: &[f32]) -> (f32, f32, f32) {
        let n = x.len().min(ref_phase.len());
        if n < 8 {
            return (0.0, 0.0, 0.0);
        }
        let (mut c, mut s, mut mag) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let xi = x[i] as f64;
            let ph = ref_phase[i] as f64;
            c += xi * ph.cos();
            s += xi * ph.sin();
            mag += xi.abs();
        }
        if mag < 1e-12 {
            return (0.0, 0.0, 0.0);
        }
        let angle = s.atan2(c);
        let lag = wrap_pi(angle as f32 - std::f32::consts::FRAC_PI_2);
        let proj = (c * c + s * s).sqrt();
        let coherence = (proj / mag) as f32;
        let amplitude = (2.0 * proj / n as f64) as f32;
        (lag, coherence.min(1.0), amplitude)
    }

    /// The substrate's measured thermal level — *excess* jitter as a
    /// fraction of the nominal period: the current RMS loop-period
    /// error minus the machine's own timing climate (a slow EMA of
    /// the same quantity, floored at [`JITTER_ABS_FLOOR`] so noise
    /// beyond one full missed period counts however ordinary it is
    /// here). The field's Langevin reservoir samples real silicon
    /// noise — but a chronically contended host is not "under
    /// physical stress" every second of its life; turbulence is
    /// what exceeds the climate. `None` until the jitter ring
    /// warms — an absent channel is no reading.
    pub fn bath_fluctuation(&self) -> Option<f64> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.jitter.filled {
            return None;
        }
        let jvar = (g.jitter_sq_mean - g.jitter_mean * g.jitter_mean).max(0.0);
        if !jvar.is_finite() {
            return None;
        }
        let cv = jvar.sqrt() / PROBE_PERIOD.as_secs_f32();
        let now = Instant::now();
        if !g.jitter_cv_climate_init {
            g.jitter_cv_climate = cv;
            g.jitter_cv_climate_init = true;
        } else if let Some(prev) = g.last_bath {
            let dt = (now - prev).as_secs_f32();
            let a = 1.0 - (-dt / JITTER_CLIMATE_TAU_S).exp();
            g.jitter_cv_climate += a * (cv - g.jitter_cv_climate);
        }
        g.last_bath = Some(now);
        let climate = g.jitter_cv_climate.max(JITTER_ABS_FLOOR);
        Some((cv - climate).max(0.0) as f64)
    }

    /// Compute the current signature. Called by the IPC handler.
    /// Returns the 328-byte wire body (version byte included).
    ///
    /// Three phases: snapshot the rings + scalars under `inner`
    /// (memcpy-scale microseconds), compute the resonator/demodulate/
    /// Maxwell math with the lock RELEASED, then write back the
    /// baseline EMAs under a second brief hold. `inner` never covers
    /// compute or I/O, so a sampler stuck on a sysfs read cannot
    /// starve this — that starvation used to stall the whole IPC
    /// queue for tens of seconds.
    pub fn signature(&self) -> Vec<u8> {
        let _sig = self.sig_lock.lock().unwrap_or_else(|e| e.into_inner());

        // ── Phase 1: snapshot under `inner` ──────────────────────
        let mut t_s = Vec::with_capacity(RING_LEN);
        let mut j_s = Vec::with_capacity(RING_LEN);
        let mut p_s = Vec::with_capacity(RING_LEN);
        let mut duty_s = Vec::with_capacity(RING_LEN);
        let mut ref_s = Vec::with_capacity(RING_LEN);
        let mut rf_s = Vec::with_capacity(RING_LEN);
        let mut rr_s = Vec::with_capacity(RING_LEN);
        let (
            throughput_mean,
            jitter_mean,
            jitter_sq_mean,
            pstate_rate,
            irq_rate,
            carrier_freq,
            carrier_phase_raw,
            warmed,
            watts_mean,
            power_hz,
            volts_v,
            amps_a,
            has_power,
            has_volts,
            has_amps,
            rf_bytes_s,
            rf_rssi_dbm,
            rf_noise_dbm,
            rf_irq_rate,
            rf_sensor_hz,
            bt_bytes_s,
            bt_events_s,
            bt_rssi_dbm,
            retry_s,
            fail_s,
            beacon_loss_s,
            tx_mbps,
            rx_mbps,
            rf_present,
        ) = {
            let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            g.throughput.drain_into(&mut t_s);
            g.jitter.drain_into(&mut j_s);
            g.power.drain_into(&mut p_s);
            g.duty.drain_into(&mut duty_s);
            g.carrier_ref.drain_into(&mut ref_s);
            g.rf_flux.drain_into(&mut rf_s);
            g.rf_rssi.drain_into(&mut rr_s);
            (
                g.throughput_mean,
                g.jitter_mean,
                g.jitter_sq_mean,
                g.pstate_rate,
                g.irq_rate,
                g.carrier_freq,
                g.carrier_phase,
                g.throughput.filled && g.jitter.filled,
                g.elec.watts_mean,
                g.elec.power_hz,
                g.elec.volts_v,
                g.elec.amps_a,
                g.elec.power.is_some(),
                g.elec.volts.is_some(),
                g.elec.amps.is_some(),
                g.radio.bytes_s,
                g.radio.rssi_dbm,
                g.radio.noise_dbm,
                g.radio.irq_rate,
                // The RSSI channel's real update rate — the iw
                // cadence while a fresh `signal avg` feeds it, the
                // /proc cadence while iw is absent or stale.
                if g.radio
                    .iw_rssi_at
                    .is_some_and(|t| t.elapsed() < Duration::from_secs(3))
                {
                    g.radio.iw_hz
                } else {
                    g.radio.rf_hz
                },
                g.radio.bt_bytes_s,
                g.radio.bt_events_s,
                g.radio.bt_rssi_dbm,
                g.radio.retry_s,
                g.radio.fail_s,
                g.radio.beacon_loss_s,
                g.radio.tx_mbps,
                g.radio.rx_mbps,
                g.radio.present_mask(),
            )
        };

        // ── Phase 2: compute with `inner` released ───────────────
        let t_raw = Channel::raw_band_powers(&t_s);
        let j_raw = Channel::raw_band_powers(&j_s);
        let p_raw = Channel::raw_band_powers(&p_s);
        // The flux ring also yields phasors — the envelope's per-band
        // phase and its inter-segment coherence, for the mind-side
        // entrainment path. The power arm is identical to
        // raw_band_powers; the phasor pass costs nothing extra.
        let (rf_raw, rf_phase, rf_coh) = Channel::raw_band_phasors(&rf_s);
        let rr_raw = Channel::raw_band_powers(&rr_s);
        let jvar = (jitter_sq_mean - jitter_mean * jitter_mean).max(0.0);
        // CV is a ratio, not a unit. Mean→0 with real variance means
        // the variation dominates — the honest report is the ceiling,
        // not a rescaled RMS in a dimensionless field. Zero variance
        // means the channel is quiescent.
        let jcv = if jvar > 0.0 {
            (jvar.sqrt() / jitter_mean.abs()).min(8.0)
        } else {
            0.0
        };

        // The measured phase: demodulate the real response against
        // the generator's own phase track. The duty channel is the
        // carrier's physical shadow — iteration count is the plan,
        // execution seconds are what the silicon actually did — so
        // its lag is the substrate's real response time in radians.
        let (carrier_lag, carrier_coherence, _) = Self::demodulate(&duty_s, &ref_s);
        let carrier_phase = carrier_phase_raw.rem_euclid(2.0 * std::f32::consts::PI);

        // The literal electrical shadow: measured watts demodulated
        // against the same reference — does real power draw carry
        // the body's rhythm, answered by the instrument. The
        // amplitude is the measured ΔP: the watts that actually move
        // at the carrier frequency, not the package's mean draw.
        let (power_lag, power_coherence, power_amp_w) = Self::demodulate(&p_s, &ref_s);

        let mask = (has_power as u8) | ((has_volts as u8) << 1) | ((has_amps as u8) << 2);

        // Maxwell's equations evaluated on the measured carrier
        // (src/daemon/maxwell.rs): the current swing is real, the
        // loop geometry is a documented assumption, and the outputs
        // are labeled estimates — the quasi-static B a ~10 Hz
        // current makes, its Faraday induction E, and the ω⁴-
        // suppressed radiated power. The rail is measured where a
        // sensor reports one, else a core-class rail is assumed.
        let v_ref = if has_volts {
            volts_v.max(0.1) as f64
        } else {
            V_CORE_ASSUMED_V
        };
        // ΔI = ΔP/V on the *measured* carrier amplitude — the watts
        // demodulated at the carrier frequency, which is the swing
        // the work slice actually makes. The package's mean draw
        // would pretend the whole die swings at carrier depth.
        let i_amp = if has_power {
            power_amp_w as f64 / v_ref
        } else {
            0.0
        };
        let m0 = super::maxwell::dipole_moment(i_amp, LOOP_AREA_M2);
        let omega = 2.0 * std::f64::consts::PI * carrier_freq as f64;
        // The regime check is part of the instrument, not an
        // assumption: ωr/c must be ≪1 for the truncation to keep its
        // honesty. Outside the regime the estimates would be numbers
        // the equations don't support — absent channel, not a lie.
        let (b_est_ut, e_ind_vm, radiated_pwr_w) =
            if super::maxwell::quasi_static_ok(omega, FIELD_REF_DIST_M) {
                (
                    super::maxwell::b_quasistatic_axial_t(m0, FIELD_REF_DIST_M) * 1.0e6,
                    super::maxwell::e_induction_vm(m0, omega, FIELD_REF_DIST_M),
                    super::maxwell::radiated_power_w(m0, omega),
                )
            } else {
                (0.0, 0.0, 0.0)
            };

        // ── Phase 3: baseline write-back under `inner` ───────────
        let now = Instant::now();
        let (t_bands, j_bands, p_bands, rf_bands, rr_bands) = {
            let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            (
                g.throughput.apply_baseline(t_raw, now),
                g.jitter.apply_baseline(j_raw, now),
                g.power.apply_baseline(p_raw, now),
                g.rf_flux.apply_baseline(rf_raw, now),
                g.rf_rssi.apply_baseline(rr_raw, now),
            )
        };

        let mut out = Vec::with_capacity(408);
        out.push(6u8);
        for v in t_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in j_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&throughput_mean.to_le_bytes());
        out.extend_from_slice(&jcv.to_le_bytes());
        out.extend_from_slice(&pstate_rate.to_le_bytes());
        out.extend_from_slice(&irq_rate.to_le_bytes());
        out.extend_from_slice(&carrier_freq.to_le_bytes());
        out.push(warmed as u8);
        out.extend_from_slice(&carrier_lag.to_le_bytes());
        out.extend_from_slice(&carrier_phase.to_le_bytes());
        out.extend_from_slice(&carrier_coherence.to_le_bytes());
        // --- v3: the electrical + Maxwell tail ---
        for v in p_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&watts_mean.to_le_bytes());
        out.extend_from_slice(&power_hz.to_le_bytes());
        out.extend_from_slice(&volts_v.to_le_bytes());
        out.extend_from_slice(&amps_a.to_le_bytes());
        out.extend_from_slice(&power_lag.to_le_bytes());
        out.extend_from_slice(&power_coherence.to_le_bytes());
        out.push(mask);
        out.extend_from_slice(&(i_amp as f32).to_le_bytes());
        out.extend_from_slice(&(b_est_ut as f32).to_le_bytes());
        out.extend_from_slice(&(e_ind_vm as f32).to_le_bytes());
        out.extend_from_slice(&(radiated_pwr_w as f32).to_le_bytes());
        // --- v4: the radio tail ---
        for v in rf_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in rr_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&rf_bytes_s.to_le_bytes());
        out.extend_from_slice(&rf_rssi_dbm.to_le_bytes());
        out.extend_from_slice(&rf_noise_dbm.to_le_bytes());
        out.extend_from_slice(&rf_irq_rate.to_le_bytes());
        out.extend_from_slice(&rf_sensor_hz.to_le_bytes());
        out.extend_from_slice(&bt_bytes_s.to_le_bytes());
        out.extend_from_slice(&bt_events_s.to_le_bytes());
        out.extend_from_slice(&bt_rssi_dbm.to_le_bytes());
        out.push(rf_present);
        // --- v5: nl80211 station telemetry (iw) ---
        out.extend_from_slice(&retry_s.to_le_bytes());
        out.extend_from_slice(&fail_s.to_le_bytes());
        out.extend_from_slice(&beacon_loss_s.to_le_bytes());
        out.extend_from_slice(&tx_mbps.to_le_bytes());
        out.extend_from_slice(&rx_mbps.to_le_bytes());
        // --- v6: flux-envelope phasors — the radio rhythm's own
        // timing. Per-band ring-frame phase and inter-segment
        // coherence, in FIELD_BANDS order; gated client-side on the
        // wlan presence bit (no flux channel → no phases to trust).
        for v in rf_phase {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in rf_coh {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine at exactly a band frequency must produce elevated
    /// energy in that band relative to a flat signal — the probe's
    /// whole claim is that a real rhythm shows up in its band.
    #[test]
    fn resonator_detects_injected_tone() {
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let n = RING_LEN;
        let f = 10.0f32; // alpha
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / fs).sin())
            .collect();
        let mut flat = vec![0.0f32; n];

        let tone_alpha = Channel::band_power(&samples, fs, 10.0);
        let flat_alpha = Channel::band_power(&flat, fs, 10.0);
        assert!(tone_alpha > 1e-4, "10 Hz tone should register in alpha");
        assert!(flat_alpha < 1e-9, "flat signal should be ~0");

        // And the tone's energy should be small in a distant band.
        let tone_delta = Channel::band_power(&samples, fs, 2.25);
        assert!(
            tone_alpha > tone_delta * 10.0,
            "alpha >> delta for a 10 Hz tone"
        );
        flat.push(0.0);
    }

    #[test]
    fn epsilon_is_absent_not_zero_reading() {
        // 150 Hz is above Nyquist at ~250 Hz sampling: the band must
        // report 0 (absent channel) even when energy exists.
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let samples: Vec<f32> = (0..RING_LEN)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * i as f32 / fs).sin())
            .collect();
        assert_eq!(Channel::band_power(&samples, fs, 150.0), 0.0);
    }

    #[test]
    fn carrier_maps_phase() {
        assert_eq!(FieldProbe::carrier_for_phase("nrem"), CARRIER_NREM_HZ);
        assert_eq!(FieldProbe::carrier_for_phase("rem"), CARRIER_REM_HZ);
        assert_eq!(FieldProbe::carrier_for_phase("drowsy"), CARRIER_DROWSY_HZ);
        assert_eq!(FieldProbe::carrier_for_phase("active"), CARRIER_WAKE_HZ);
    }

    #[test]
    fn pulse_arms_clamps_and_disarms_cleanly() {
        let probe = FieldProbe::new();
        probe.set_pulse(0.55, 0.15, 0.10);
        {
            let g = probe.inner.lock().unwrap();
            assert!(g.pulse_armed);
            assert_eq!(g.pulse_base, 0.55);
            assert_eq!(g.pulse_mod_freq, 0.15);
            assert_eq!(g.pulse_mod_depth, 0.10);
        }
        // Out-of-range writes clamp, never wrap.
        probe.set_pulse(2.0, 100.0, -1.0);
        {
            let g = probe.inner.lock().unwrap();
            assert_eq!(g.pulse_base, 1.0);
            assert_eq!(g.pulse_mod_freq, 4.0);
            assert_eq!(g.pulse_mod_depth, 0.0);
        }
        // mod_freq <= 0 disarms; the static set-point is untouched.
        probe.set_pulse(0.0, 0.0, 0.0);
        {
            let g = probe.inner.lock().unwrap();
            assert!(!g.pulse_armed);
            assert_eq!(g.carrier_depth, 0.60);
        }
    }

    #[test]
    fn signature_layout_is_408_bytes() {
        let probe = FieldProbe::new();
        // Push a handful of samples so the path is exercised.
        {
            let mut g = probe.inner.lock().unwrap();
            for i in 0..8 {
                g.throughput.push(i as f32);
                g.jitter.push((i as f32) * 1e-6);
                g.carrier_ref.push(0.0);
                g.power.push(24.0);
            }
        }
        let sig = probe.signature();
        assert_eq!(sig.len(), 408, "wire layout must stay fixed");
        assert_eq!(sig[0], 6);
        // Not warmed — only 8 of 1024 samples.
        assert_eq!(sig[101], 0);
    }

    /// Two tones at the same band frequency differing by a known
    /// phase must report phases differing by the same amount — the
    /// segment-frame reference cancels in the difference, so this
    /// checks the phasor against itself rather than a convention.
    #[test]
    fn phasor_recovers_relative_phase() {
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let f = 10.0f32;
        let mk = |phi: f32| -> Vec<f32> {
            (0..RING_LEN)
                .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / fs + phi).sin())
                .collect()
        };
        let (_, p0, _) = Channel::raw_band_phasors(&mk(0.0));
        let (_, p1, _) = Channel::raw_band_phasors(&mk(std::f32::consts::FRAC_PI_3));
        let mut d = p1[3] - p0[3]; // alpha
        while d > std::f32::consts::PI {
            d -= 2.0 * std::f32::consts::PI;
        }
        while d < -std::f32::consts::PI {
            d += 2.0 * std::f32::consts::PI;
        }
        assert!(
            (d - std::f32::consts::FRAC_PI_3).abs() < 0.02,
            "phase difference {d} should match the injected π/3"
        );
    }

    /// Inter-segment coherence is the "is this a real rhythm" gate:
    /// a sustained tone holds its phase across both ring halves
    /// (coh ≈ 1); a tone that flips π mid-ring does not. The
    /// residual isn't zero — off-bin phasors carry negative-frequency
    /// leakage that distorts each segment's measured angle, so the
    /// flipped floor is ≈0.12, far under the sustained case.
    #[test]
    fn phasor_coherence_flags_phase_flip() {
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let f = 10.0f32;
        let tone = |seg_phi: &[f32]| -> Vec<f32> {
            (0..RING_LEN)
                .map(|i| {
                    let half = (i / (RING_LEN / 2)).min(1);
                    (2.0 * std::f32::consts::PI * f * i as f32 / fs + seg_phi[half]).sin()
                })
                .collect()
        };
        let (_, _, steady) = Channel::raw_band_phasors(&tone(&[0.0, 0.0]));
        let (_, _, flipped) = Channel::raw_band_phasors(&tone(&[0.0, std::f32::consts::PI]));
        assert!(steady[3] > 0.9, "sustained tone coh {}", steady[3]);
        assert!(flipped[3] < 0.25, "flipped tone coh {}", flipped[3]);
    }

    #[test]
    fn iw_station_parses_real_dump() {
        // Real `iw dev wlo1 station dump` shape — the MAC's colons
        // mean the header must be detected by prefix, not split.
        let content = "Station 9c:53:22:01:ee:7b (on wlo1)\n\
             \x20\tinactive time:\t1 ms\n\
             \x20\ttx retries:\t40663\n\
             \x20\ttx failed:\t51\n\
             \x20\tsignal:  \t-56 [-56] dBm\n\
             \x20\tsignal avg:\t-55 [-55] dBm\n\
             \x20\tbeacon loss:\t8\n\
             \x20\ttx bitrate:\t72.2 MBit/s MCS 7 short GI\n\
             \x20\trx bitrate:\t72.2 MBit/s MCS 7 short GI\n";
        let st = parse_iw_station(content).expect("a station block");
        assert_eq!(st.signal_dbm, Some(-56.0));
        assert_eq!(st.signal_avg_dbm, Some(-55.0));
        assert_eq!(st.tx_retries, 40663);
        assert_eq!(st.tx_failed, 51);
        assert_eq!(st.beacon_loss, 8);
        assert_eq!(st.tx_mbps, Some(72.2));
        assert_eq!(st.rx_mbps, Some(72.2));
    }

    #[test]
    fn iw_station_absent_without_station() {
        // `command failed: No such device` or a disassociated dump —
        // absent channel, not a zeroed station.
        assert!(parse_iw_station("command failed: No such device (-19)\n").is_none());
        assert!(parse_iw_station("").is_none());
    }

    #[test]
    fn wireless_levels_parse_real_format() {
        // Real /proc/net/wireless shape — ath9k reports −71 dBm
        // level and the −256 "no reading" sentinel for noise.
        let content = "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n \
             face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n \
             \x20 wlo1: 0000   39.  -71.  -256        0      0      0     31    349        0\n";
        let (seen, level, noise) = parse_wireless_levels(content);
        assert!(seen);
        assert_eq!(level, Some(-71.0));
        assert_eq!(noise, None, "−256 is a sentinel, not a floor");
    }

    #[test]
    fn wireless_levels_absent_when_no_iface() {
        let content = "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n \
             face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n";
        let (seen, level, noise) = parse_wireless_levels(content);
        assert!(!seen);
        assert_eq!(level, None);
        assert_eq!(noise, None);
    }

    #[test]
    fn wireless_levels_zero_level_is_absent() {
        // A 0 level is the driver's "unknown", not 0 dBm of signal.
        let content = "hdr1\nhdr2\n wlan0: 0000   0.  0.  -90.\n";
        let (seen, level, noise) = parse_wireless_levels(content);
        assert!(seen);
        assert_eq!(level, None);
        assert_eq!(noise, Some(-90.0));
    }

    #[test]
    fn wireless_intr_matches_named_drivers_only() {
        let content = "           CPU0       CPU1\n \
             \x20 53:          0          9  IO-APIC    4-fasteoi   ath9k\n \
             \x20 27:        100        200  PCI-MSIX-0000:00:10.0    0-edge      xhci_hcd\n";
        let (total, matched) = parse_wireless_intr_str(content);
        assert!(matched);
        assert_eq!(total, 9, "only the ath9k line counts");
    }

    #[test]
    fn energy_delta_wraps_at_max_range() {
        // Normal forward delta.
        assert_eq!(energy_delta_uj(1_000, 400, u64::MAX), 600);
        // A counter that wrapped inside the interval still yields the
        // honest joules: (max − prev) + cur.
        assert_eq!(energy_delta_uj(100, 900, 1_000), 200);
        // Equal reads → no energy, never a negative wrap-around spike.
        assert_eq!(energy_delta_uj(500, 500, 1_000), 0);
    }

    #[test]
    fn electrical_absent_reports_zero_mask() {
        // A probe with no discovered sensors must encode the absence
        // in the presence mask — not as watts that were never read.
        let probe = FieldProbe::new();
        {
            let mut g = probe.inner.lock().unwrap();
            g.elec.power = None;
            g.elec.volts = None;
            g.elec.amps = None;
        }
        let sig = probe.signature();
        assert_eq!(sig[178], 0, "no sensors → empty presence mask");
        // And the Maxwell estimates collapse to zero rather than
        // inventing a field from an unmeasured current.
        assert_eq!(&sig[179..195], &[0u8; 16]);
    }

    #[test]
    fn demodulate_recovers_known_lag() {
        // A response trailing the reference by δ must report lag ≈ δ.
        let n = 512;
        let f = 10.0f64;
        let fs = 250.0f64;
        let lag = 0.6f64;
        let refs: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / fs) as f32)
            .collect();
        // Trailing by δ: the response shows the reference's *past*
        // phase — x(t) = sin(ref(t) − δ).
        let xs: Vec<f32> = refs
            .iter()
            .map(|&r| (r as f64 - lag).sin() as f32)
            .collect();
        let (meas, coherence, amplitude) = FieldProbe::demodulate(&xs, &refs);
        assert!(
            (meas - lag as f32).abs() < 0.02,
            "lag should be ≈{lag}, got {meas}"
        );
        // A leading response reports a negative lag — the sign is
        // real, not an absolute value.
        let lead: Vec<f32> = refs
            .iter()
            .map(|&r| (r as f64 + lag).sin() as f32)
            .collect();
        let (lead_meas, _, _) = FieldProbe::demodulate(&lead, &refs);
        assert!(
            (lead_meas + lag as f32).abs() < 0.02,
            "leading response should report lag ≈{}, got {lead_meas}",
            -lag
        );
        // A pure sinusoid's coherence is π/4 ≈ 0.785 — the |x|
        // normalization, not a defect.
        assert!(
            coherence > 0.7,
            "pure tone should be coherent, got {coherence}"
        );
        // Unit-amplitude tone → recovered amplitude ≈ 1.0.
        assert!(
            (amplitude - 1.0).abs() < 0.02,
            "amplitude should be ≈1.0, got {amplitude}"
        );
        // An orthogonal (uncorrelated) signal: coherence ~0.
        let noise: Vec<f32> = refs
            .iter()
            .map(|&r| (r as f64 * 7.3).cos() as f32)
            .collect();
        let (_, c2, a2) = FieldProbe::demodulate(&noise, &refs);
        assert!(
            c2 < 0.1,
            "off-frequency signal should be incoherent, got {c2}"
        );
        assert!(
            a2 < 0.1,
            "off-frequency signal should have no coherent amplitude, got {a2}"
        );
    }
}

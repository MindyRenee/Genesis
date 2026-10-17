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
//! # Bands
//!
//! Each fast channel is decomposed by Goertzel filters at the same
//! ten center frequencies the Python oscillator models. Sample rate
//! is ~250 Hz, so bands through gamma (50 Hz) are resolvable;
//! lambda (150 Hz) is above Nyquist and always reports 0 — an
//! absent channel, not a reading of zero.
//!
//! # Carrier
//!
//! The probe also *writes* a rhythm: its work-slice length is
//! modulated sinusoidally at a frequency set by the daemon's own
//! sleep-wake phase (wake ≈ 10 Hz, REM ≈ 5.5 Hz, NREM ≈ 2.25 Hz). The
//! modulated work is real electron flow — the machine's actual
//! power draw carries a component at the body's own rhythm, and the
//! probe can measure its own carrier in its band energies. This is
//! the first loop fully inside physics: state → real current →
//! measurement → state.
//!
//! # Phase locking
//!
//! The carrier is not only generated — it is *locked*. The mind may
//! push its modeled band phase through `SET_FIELD_PHASE`; the probe
//! extrapolates that phase forward at the carrier rate and pulls the
//! generator's own phase toward it each tick (a first-order PLL, so
//! the pull is bounded and smooth). Symmetrically, the signature
//! reports the *measured* phase of the substrate's real response —
//! computed by synchronous demodulation against the generator's own
//! phase track — so the mind can pull its modeled phase toward the
//! physics. Both directions pull: two oscillators becoming one
//! dynamical system. A stale target (>5 s without a fresh push) is
//! dropped — a dead mind's last phase must not steer the body.
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
//! [u8 version = 2]
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
//! ```
//! 114 bytes, fixed layout. Version 1 (102 bytes, no phase fields)
//! is still decoded by the client — older daemons simply have no
//! measured phase, which is an absent channel, not a zero.
//! Additive — PROTOCOL_VERSION unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Center frequencies (Hz) of the ten wave bands, in the same order
/// as `BrainWave` in `python/thalamus/brain_waves.py`:
/// epsilon, delta, theta, alpha, sigma, beta1, beta2, beta3, gamma,
/// lambda.
///
/// Known overlap: sigma (14.0) and beta1 (13.5) sit 0.5 Hz apart —
/// inside one another's band edges and ~2 bins at the ring's ~0.24 Hz
/// resolution — so the two channels largely measure the same physical
/// energy. They stay distinct because the *model* treats them
/// differently (spindle vs. SMR); the probe just cannot separate them.
pub const FIELD_BANDS: [f32; 10] = [0.25, 2.25, 5.5, 10.0, 14.0, 13.5, 17.5, 25.0, 50.0, 150.0];

/// Nominal probe period — ~250 Hz sampling. Bands through gamma sit
/// below the 125 Hz Nyquist bound; lambda is reported as 0.
const PROBE_PERIOD: Duration = Duration::from_millis(4);

/// Samples held per channel (~4.1 s at 250 Hz — one epsilon cycle).
const RING_LEN: usize = 1024;

/// Iterations of the calibration work slice per unit of carrier
/// modulation. Tuned so the slice is ~150-250 µs on this hardware —
/// enough to measure, small enough to cost ~5% of one thread.
const WORK_SLICE_ITERS: u32 = 4096;

/// Carrier frequencies keyed by body phase. The body's own rhythm,
/// written into real work — not a claim about what the rhythm means.
const CARRIER_WAKE_HZ: f32 = 10.0;
const CARRIER_REM_HZ: f32 = 5.5;
const CARRIER_NREM_HZ: f32 = 2.25;

/// Baseline EMA weight per signature computation. A band's reported
/// energy is raw/baseline, so "1.0" means "typical for this band on
/// this machine" and values above it mean elevated vs. own history.
const BASELINE_ALPHA: f32 = 0.02;

/// First-order PLL pull per probe tick (~250 Hz): the generator's
/// carrier phase closes a fraction of its gap to the extrapolated
/// model phase each period — τ ≈ 130 ms, fast enough to track a
/// ~1 Hz-updated phase target, smooth enough to be physical.
const PHASE_PULL: f32 = 0.03;

/// A pushed phase target older than this is dropped: the mind's
/// ~3 s lease discipline — a dead mind's last phase must not keep
/// steering the body's rhythm.
const PHASE_TARGET_TTL: Duration = Duration::from_secs(5);

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
}

impl Channel {
    fn new() -> Self {
        Self {
            ring: [0.0; RING_LEN],
            head: 0,
            filled: false,
            baseline: [0.0; 10],
            baseline_init: false,
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

    /// Goertzel power of `samples` at `freq` for sampling rate `fs`.
    /// Exact-bin: the bin nearest each band center is used.
    fn band_power(samples: &[f32], fs: f32, freq: f32) -> f32 {
        let n = samples.len();
        if n < 4 {
            return 0.0;
        }
        let k = ((n as f32 * freq) / fs).round().max(1.0);
        if k >= n as f32 / 2.0 {
            return 0.0; // above Nyquist — absent channel
        }
        let w = 2.0 * std::f32::consts::PI * k / n as f32;
        let coeff = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for &x in samples {
            let s0 = x + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n as f32 * n as f32)
    }

    /// Compute all band energies, update baselines, return the
    /// baseline-relative energies (≈1.0 = typical; >1 = elevated).
    fn band_energies(&mut self, scratch: &mut Vec<f32>) -> [f32; 10] {
        self.drain_into(scratch);
        let fs = 1.0 / PROBE_PERIOD.as_secs_f32();
        let mut out = [0.0f32; 10];
        for (i, &f) in FIELD_BANDS.iter().enumerate() {
            let raw = Self::band_power(scratch, fs, f);
            if !self.baseline_init {
                self.baseline[i] = raw.max(1e-12);
            } else {
                self.baseline[i] += BASELINE_ALPHA * (raw - self.baseline[i]);
            }
            let rel = raw / self.baseline[i].max(1e-12);
            out[i] = rel.clamp(0.0, 8.0) / 8.0; // 0..1, 0.125 = typical
        }
        self.baseline_init = true;
        out
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
    last_tis: (u64, u64),
    pstate_rate: f32,
    last_intr: u64,
    irq_rate: f32,
    last_slow: Instant,
    throughput_mean: f32,
    jitter_sq_mean: f32,
    jitter_mean: f32,
    carrier_freq: f32,
    /// The carrier generator's instantaneous phase (radians).
    carrier_phase: f32,
    /// Modulation depth of the carrier's real work (0 = unmodulated).
    carrier_depth: f32,
    /// Frequency override (Hz). 0 = derive from the body's phase;
    /// the ring-down experiment drives arbitrary frequencies —
    /// including epsilon-rate infraslow where the thermal field can
    /// actually sustain an oscillation.
    carrier_freq_override: f32,
    /// The mind's most recent pushed phase + when it arrived. The
    /// probe extrapolates it at the carrier rate and pulls
    /// `carrier_phase` toward it — the physics-follows-model leg of
    /// the mutual lock.
    phase_target: Option<(Instant, f32)>,
}

/// The running probe. `spawn`ed once by the daemon; `signature` is
/// called by the IPC handler at the mind's cadence.
pub struct FieldProbe {
    inner: Mutex<ProbeInner>,
    running: AtomicBool,
    stop: AtomicBool,
}

/// Process-shared probe handle. Started in `daemon main`, read by
/// the IPC handler — mirrors how `interoception` shares state.
static FIELD_PROBE: OnceLock<std::sync::Arc<FieldProbe>> = OnceLock::new();

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
    std::thread::Builder::new()
        .name("field-probe".into())
        .spawn(move || probe.run(phase_source))
        .expect("field-probe thread");
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
            inner: Mutex::new(ProbeInner {
                throughput: Channel::new(),
                jitter: Channel::new(),
                carrier_ref: Channel::new(),
                duty: Channel::new(),
                last_tis: (0, 0),
                pstate_rate: 0.0,
                last_intr: 0,
                irq_rate: 0.0,
                last_slow: Instant::now(),
                throughput_mean: 0.0,
                jitter_sq_mean: 0.0,
                jitter_mean: 0.0,
                carrier_freq: CARRIER_WAKE_HZ,
                carrier_phase: 0.0,
                carrier_depth: 0.60,
                carrier_freq_override: 0.0,
                phase_target: None,
            }),
            stop: AtomicBool::new(false),
        }
    }

    /// Carrier frequency for the body's phase name. Wake ≈ alpha,
    /// REM ≈ theta, NREM ≈ delta — the body's actual state carried
    /// on real work. Unknown/absent phase → wake rate.
    fn carrier_for_phase(phase: &str) -> f32 {
        match phase {
            "nrem" => CARRIER_NREM_HZ,
            "rem" => CARRIER_REM_HZ,
            _ => CARRIER_WAKE_HZ,
        }
    }

    /// Pull a carrier phase toward `expected` by the PLL gain —
    /// the physics-follows-model half of the lock. Bounded and
    /// smooth: the physical rhythm chases the model's phase rather
    /// than teleporting to it.
    fn pll_step(carrier_phase: f32, expected: f32) -> f32 {
        carrier_phase + PHASE_PULL * wrap_pi(expected - carrier_phase)
    }

    /// The mind pushes its modeled band phase here (SET_FIELD_PHASE).
    /// The probe extrapolates it at the carrier rate and chases it.
    pub fn set_phase_target(&self, phase: f32) {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.phase_target = Some((Instant::now(), phase));
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

    fn run(&self, phase_source: impl Fn() -> &'static str) {
        let nominal = PROBE_PERIOD.as_secs_f32();
        let mut prev = Instant::now();
        while !self.stop.load(Ordering::Relaxed) {
            let t0 = Instant::now();
            // Jitter: how long since the last iteration actually ran.
            let jitter = (t0 - prev).as_secs_f32() - nominal;
            prev = t0;

            // Carrier-modulated work slice: the body's phase (or an
            // override) sets the rhythm; depth scales how much real
            // work — real electron flow — swings this window.
            let carrier_f;
            let depth;
            {
                let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                if g.last_slow.elapsed() >= Duration::from_millis(250) {
                    g.carrier_freq = if g.carrier_freq_override > 0.0 {
                        g.carrier_freq_override
                    } else {
                        Self::carrier_for_phase(phase_source())
                    };
                    let trans = read_pstate_transitions();
                    let intr = read_intr_total();
                    let dt = g.last_slow.elapsed().as_secs_f32().max(1e-3);
                    if g.last_tis.1 > 0 {
                        g.pstate_rate = trans.saturating_sub(g.last_tis.1) as f32 / dt;
                        g.irq_rate = intr.saturating_sub(g.last_intr) as f32 / dt;
                    }
                    g.last_tis = (0, trans);
                    g.last_intr = intr;
                    g.last_slow = Instant::now();
                }
                // A pushed model phase older than the TTL is a dead
                // mind's last word — drop it, run free.
                if g.phase_target
                    .is_some_and(|(t0, _)| t0.elapsed() > PHASE_TARGET_TTL)
                {
                    g.phase_target = None;
                }
                carrier_f = g.carrier_freq;
                depth = g.carrier_depth;
            }

            let carrier_phase;
            {
                let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                g.carrier_phase += 2.0 * std::f32::consts::PI * carrier_f * nominal;
                // Physics-follows-model: extrapolate the last pushed
                // phase at the carrier rate and pull toward it.
                if let Some((t_recv, target)) = g.phase_target {
                    let expected = target
                        + 2.0 * std::f32::consts::PI * carrier_f * t_recv.elapsed().as_secs_f32();
                    g.carrier_phase = Self::pll_step(g.carrier_phase, expected);
                }
                g.carrier_phase = g.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);
                carrier_phase = g.carrier_phase;
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
    /// carrier reference track: returns (lag_radians, coherence).
    ///
    /// For a response `x_i = A·sin(ref_i + δ)`, correlating against
    /// sin/cos of the reference gives `atan2(S, C) = π/2 − δ`, so
    /// `lag = π/2 − angle` — how far the substrate's real response
    /// trails the generated carrier, in radians. `coherence` is
    /// |Σ x·e^{i·ref}| / Σ|x| — the fraction of the signal that moves
    /// with the carrier. A lag computed from an incoherent signal is
    /// noise; the client treats low coherence as a weak lock, which
    /// is why it travels on the wire too.
    fn demodulate(x: &[f32], ref_phase: &[f32]) -> (f32, f32) {
        let n = x.len().min(ref_phase.len());
        if n < 8 {
            return (0.0, 0.0);
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
            return (0.0, 0.0);
        }
        let angle = s.atan2(c);
        let lag = wrap_pi(std::f32::consts::FRAC_PI_2 - angle as f32);
        let coherence = ((c * c + s * s).sqrt() / mag) as f32;
        (lag, coherence.min(1.0))
    }

    /// The ten throughput band energies — the substrate's real
    /// electron-flow signature at this instant. `None` before the
    /// ring buffers warm up (an absent reading, not a zero).
    /// Used by the dyadic model to record the field configuration
    /// measured while each person is present.
    pub fn band_signature(&self) -> Option<[f32; 10]> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if !g.throughput.filled {
            return None;
        }
        let mut scratch = Vec::with_capacity(RING_LEN);
        Some(g.throughput.band_energies(&mut scratch))
    }

    /// Compute the current signature. Called by the IPC handler.
    /// Returns the 114-byte wire body (version byte included).
    pub fn signature(&self) -> Vec<u8> {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut scratch = Vec::with_capacity(RING_LEN);
        let t_bands = g.throughput.band_energies(&mut scratch);
        let j_bands = g.jitter.band_energies(&mut scratch);
        let jvar = (g.jitter_sq_mean - g.jitter_mean * g.jitter_mean).max(0.0);
        let jcv = if g.jitter_mean.abs() > 1e-9 {
            (jvar.sqrt() / g.jitter_mean.abs()).min(8.0)
        } else {
            jvar.sqrt() * 1e6 // near-zero mean jitter: report raw rms in µs-ish
        };
        let warmed = g.throughput.filled && g.jitter.filled;

        // The measured phase: demodulate the real response against
        // the generator's own phase track. The duty channel is the
        // carrier's physical shadow — iteration count is the plan,
        // execution seconds are what the silicon actually did — so
        // its lag is the substrate's real response time in radians.
        let mut xs = Vec::with_capacity(RING_LEN);
        let mut refs = Vec::with_capacity(RING_LEN);
        g.duty.drain_into(&mut xs);
        g.carrier_ref.drain_into(&mut refs);
        let (carrier_lag, carrier_coherence) = Self::demodulate(&xs, &refs);
        let carrier_phase = g.carrier_phase.rem_euclid(2.0 * std::f32::consts::PI);

        let mut out = Vec::with_capacity(114);
        out.push(2u8);
        for v in t_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in j_bands {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&g.throughput_mean.to_le_bytes());
        out.extend_from_slice(&jcv.to_le_bytes());
        out.extend_from_slice(&g.pstate_rate.to_le_bytes());
        out.extend_from_slice(&g.irq_rate.to_le_bytes());
        out.extend_from_slice(&g.carrier_freq.to_le_bytes());
        out.push(warmed as u8);
        out.extend_from_slice(&carrier_lag.to_le_bytes());
        out.extend_from_slice(&carrier_phase.to_le_bytes());
        out.extend_from_slice(&carrier_coherence.to_le_bytes());
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
    fn goertzel_detects_injected_tone() {
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
    fn lambda_is_absent_not_zero_reading() {
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
        assert_eq!(FieldProbe::carrier_for_phase("active"), CARRIER_WAKE_HZ);
    }

    #[test]
    fn signature_layout_is_114_bytes() {
        let probe = FieldProbe::new();
        // Push a handful of samples so the path is exercised.
        {
            let mut g = probe.inner.lock().unwrap();
            for i in 0..8 {
                g.throughput.push(i as f32);
                g.jitter.push((i as f32) * 1e-6);
                g.carrier_ref.push(0.0);
            }
        }
        let sig = probe.signature();
        assert_eq!(sig.len(), 114, "wire layout must stay fixed");
        assert_eq!(sig[0], 2);
        // Not warmed — only 8 of 1024 samples.
        assert_eq!(sig[101], 0);
    }

    #[test]
    fn pll_step_pulls_toward_target() {
        // A target π/2 ahead must move the carrier phase forward by
        // the bounded pull, not snap to it.
        let next = FieldProbe::pll_step(0.0, std::f32::consts::FRAC_PI_2);
        assert!((next - PHASE_PULL * std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        // Wrapping: a target at −π+0.1 from phase π−0.1 is 0.2 ahead,
        // not ~2π behind — nearest equivalent phase wins.
        let next = FieldProbe::pll_step(std::f32::consts::PI - 0.1, -std::f32::consts::PI + 0.1);
        let err = wrap_pi(-std::f32::consts::PI + 0.1 - (std::f32::consts::PI - 0.1));
        assert!(
            (err - 0.2).abs() < 1e-5,
            "wrapped err should be +0.2, got {err}"
        );
        assert!((next - (std::f32::consts::PI - 0.1 + PHASE_PULL * 0.2)).abs() < 1e-5);
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
        let xs: Vec<f32> = refs
            .iter()
            .map(|&r| (r as f64 + lag).sin() as f32)
            .collect();
        let (meas, coherence) = FieldProbe::demodulate(&xs, &refs);
        assert!(
            (meas - lag as f32).abs() < 0.02,
            "lag should be ≈{lag}, got {meas}"
        );
        // A pure sinusoid's coherence is π/4 ≈ 0.785 — the |x|
        // normalization, not a defect.
        assert!(
            coherence > 0.7,
            "pure tone should be coherent, got {coherence}"
        );
        // An orthogonal (uncorrelated) signal: coherence ~0.
        let noise: Vec<f32> = refs
            .iter()
            .map(|&r| (r as f64 * 7.3).cos() as f32)
            .collect();
        let (_, c2) = FieldProbe::demodulate(&noise, &refs);
        assert!(
            c2 < 0.1,
            "off-frequency signal should be incoherent, got {c2}"
        );
    }
}

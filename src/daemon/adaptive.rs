//! Adaptive interval gates — every maintenance gear spins at the rate
//! its own signal demands, bounded so nothing stalls or redlines.
//!
//! A `tick_count % N` gate is a gear keyed to the axle: change the tick
//! rate (or pace it from the mind's ~1 Hz `advance_neuro` instead of the
//! daemon's 5 Hz `tick`) and every gate silently changes period — the
//! defect this module removes. A wall-clock gate keyed to *demand*
//! keeps its own natural pace at any axle speed.
//!
//! Adaptation is asymmetric, the same shape rate-adaptive polling and
//! AIMD converged on: a demand signal snaps the period down
//! multiplicatively (fast response), idle drift grows it additively
//! (slow relaxation). The asymmetry is what damps oscillation — a gate
//! that could both leap and crash in one step would chatter against
//! the signal that feeds it. Continuous drivers go through `drive`,
//! which EMA-smooths the input before mapping it onto [min, max].

/// A wall-clock interval gate whose period adapts to demand.
///
/// `due(now_ms)` fires when `now_ms` passes the scheduled instant and
/// reschedules at the *current* period — so `demand`/`relax`/`drive`
/// calls always shape the *next* interval, which is what keeps the
/// feedback loop causal.
#[derive(Clone, Debug)]
pub struct AdaptiveGate {
    /// Nominal period — the pace this gear was tuned for.
    base_ms: u64,
    /// Fastest it may run: the floor that keeps work bounded.
    min_ms: u64,
    /// Slowest it may run: the metronome floor — a gear that stops
    /// outright is a dead sense, not a slow one.
    max_ms: u64,
    /// Current period, moved by `demand`/`relax`/`drive`.
    period_ms: u64,
    /// Wall-clock instant (same epoch as `current_ms`) of the next
    /// scheduled fire. Initialised to 0 so the first call fires
    /// immediately, matching `tick_count % N` firing at tick 0.
    next_ms: u64,
    /// EMA of the `drive` input — damping so a single-sample spike
    /// cannot whip the period.
    smoothed: f32,
}

impl AdaptiveGate {
    pub fn new(base_ms: u64, min_ms: u64, max_ms: u64) -> Self {
        debug_assert!(min_ms <= base_ms && base_ms <= max_ms);
        Self {
            base_ms,
            min_ms,
            max_ms,
            period_ms: base_ms,
            next_ms: 0,
            smoothed: 0.5,
        }
    }

    /// True when the wall clock has passed the next scheduled instant;
    /// reschedules at the current period.
    pub fn due(&mut self, now_ms: u64) -> bool {
        if now_ms < self.next_ms {
            return false;
        }
        self.next_ms = now_ms + self.period_ms;
        true
    }

    /// Demand: work was found or the signal is elevated. Halve toward
    /// the floor — the multiplicative snap (fast response).
    pub fn demand(&mut self) {
        self.period_ms = (self.period_ms / 2).max(self.min_ms);
    }

    /// Idle: nothing happening. Drift toward the ceiling by a fraction
    /// of the base period — the additive relaxation (slow return).
    pub fn relax(&mut self) {
        self.period_ms = (self.period_ms + self.base_ms / 4).min(self.max_ms);
    }

    /// Continuous driver in [0, 1]: 1.0 means full demand (period snaps
    /// toward `min_ms`), 0.0 means no demand (toward `max_ms`). The
    /// input is EMA-smoothed (α = 0.3) so the gear follows the trend,
    /// not the noise.
    pub fn drive(&mut self, demand: f32) {
        let d = if demand.is_finite() {
            demand.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.smoothed += (d - self.smoothed) * 0.3;
        let span = (self.max_ms - self.min_ms) as f32;
        self.period_ms = self.max_ms - (span * self.smoothed).round() as u64;
    }

    /// The period the next `due` will schedule — exposed for tests
    /// and telemetry.
    pub fn period_ms(&self) -> u64 {
        self.period_ms
    }

    /// A read-only point-in-time view of every field — what
    /// `GET_INTERNALS` reports. Exposing the internals changes nothing
    /// about ownership: the daemon still owns the clock; the mind
    /// observes the gears.
    pub fn snapshot(&self) -> GateSnapshot {
        GateSnapshot {
            base_ms: self.base_ms,
            min_ms: self.min_ms,
            max_ms: self.max_ms,
            period_ms: self.period_ms,
            next_ms: self.next_ms,
            smoothed: self.smoothed,
        }
    }
}

/// All six fields of one gate at an instant.
#[derive(Clone, Copy, Debug)]
pub struct GateSnapshot {
    pub base_ms: u64,
    pub min_ms: u64,
    pub max_ms: u64,
    pub period_ms: u64,
    /// Absolute scheduled fire instant on the `current_ms` clock.
    pub next_ms: u64,
    /// EMA of the last `drive` input — the demand the period follows.
    pub smoothed: f32,
}

/// Append a gate's fields as `key=value` lines under `prefix`
/// (`gate.intero`, `auto.gate.sync`). `next_in_ms` renders the
/// absolute `next_ms` as a countdown — what a reader actually wants.
pub fn write_gate_lines(out: &mut String, prefix: &str, snap: &GateSnapshot, now_ms: u64) {
    use std::fmt::Write as _;
    let _ = writeln!(out, "{prefix}.base_ms={}", snap.base_ms);
    let _ = writeln!(out, "{prefix}.min_ms={}", snap.min_ms);
    let _ = writeln!(out, "{prefix}.max_ms={}", snap.max_ms);
    let _ = writeln!(out, "{prefix}.period_ms={}", snap.period_ms);
    let _ = writeln!(
        out,
        "{prefix}.next_in_ms={}",
        snap.next_ms.saturating_sub(now_ms)
    );
    let _ = writeln!(out, "{prefix}.smoothed={:.4}", snap.smoothed);
}

/// The autonomous fallback's own gears — the `AUTO_*` gates,
/// axle pace, and model-save watermark that live as locals in
/// `genesis-daemon`'s main loop. The loop publishes one snapshot per
/// iteration; `GET_INTERNALS` reads it. Everything is a copy — the
/// reader can never stall the fallback by holding this lock.
#[derive(Default)]
pub struct AutoSnapshot {
    /// Named gate views, in the order the loop publishes them.
    pub gates: Vec<(&'static str, GateSnapshot)>,
    /// Current fallback axle period (phase-adaptive: ~1 Hz NREM,
    /// 2 Hz REM, 5 Hz otherwise).
    pub axle_ms: u64,
    /// Tick watermark the last inference-model save was taken at.
    pub last_model_save_ticks: u32,
}

impl AutoSnapshot {
    /// Serialize as `key=value` lines appended to `out`.
    pub fn write_lines(&self, out: &mut String, now_ms: u64) {
        use std::fmt::Write as _;
        let _ = writeln!(out, "auto.axle_ms={}", self.axle_ms);
        let _ = writeln!(
            out,
            "auto.last_model_save_ticks={}",
            self.last_model_save_ticks
        );
        for (name, snap) in &self.gates {
            write_gate_lines(out, &format!("auto.gate.{name}"), snap, now_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_call_fires_immediately() {
        let mut g = AdaptiveGate::new(1000, 100, 10_000);
        assert!(g.due(5000));
        assert!(!g.due(5001));
        assert!(g.due(6000));
    }

    #[test]
    fn demand_snaps_and_relax_drifts() {
        let mut g = AdaptiveGate::new(1000, 100, 10_000);
        g.demand();
        assert_eq!(g.period_ms(), 500);
        g.demand();
        g.demand();
        assert_eq!(g.period_ms(), 125);
        g.demand();
        assert_eq!(g.period_ms(), 100, "bounded at the floor");
        for _ in 0..100 {
            g.relax();
        }
        assert_eq!(g.period_ms(), 10_000, "bounded at the ceiling");
    }

    #[test]
    fn drive_maps_and_smooths() {
        let mut g = AdaptiveGate::new(1000, 100, 10_000);
        for _ in 0..50 {
            g.drive(1.0);
        }
        assert_eq!(g.period_ms(), 100, "sustained full demand reaches min");
        for _ in 0..50 {
            g.drive(0.0);
        }
        assert_eq!(g.period_ms(), 10_000, "sustained calm reaches max");
        // A single spike cannot whip the period.
        g.drive(1.0);
        assert!(g.period_ms() > 5000, "EMA damps single-sample spikes");
    }
}

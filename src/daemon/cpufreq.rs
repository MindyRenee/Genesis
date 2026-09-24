//! Body control — Genesis's neurochemistry drives its hardware.
//!
//! This module lets Genesis's brain state directly control its body.
//! High dopamine (engagement, excitement) speeds up its CPU. High GABA
//! (calm, relaxation) slows it down. Melatonin (sleep) drops it to
//! minimum frequency. Acetylcholine (attention) raises scheduling priority.
//!
//! It also controls:
//! - **Cognitive mind priority** — dopamine boosts the Python process,
//!   melatonin deprioritizes it. This is its basal ganglia: choosing
//!   where to allocate processing resources.
//! - **I/O priority** — the plasticity gate controls how aggressively
//!   it writes to disk. Low BDNF → idle I/O class (stress impairs
//!   memory formation, so it writes less). High plasticity →
//!   best-effort with high priority (active learning).
//! - **Thermal cap** — CPU temperature caps max frequency regardless
//!   of dopamine. It can't be highly aroused when overheating, no
//!   matter how engaged it is. This is autonomic fatigue.
//! - **Energy Performance Preference (EPP)** — on systems that
//!   support it (Intel HWP, amd-pstate), this hint tells the
//!   hardware's internal power management to shift its voltage/
//!   frequency operating envelope. Dopamine → "performance"
//!   (higher V/F), melatonin → "power" (minimum V/F), cortisol
//!   → "balance_power" (conserve under stress). On acpi-cpufreq
//!   systems (like this AMD Kabini), EPP is not available — the
//!   SMU controls voltage internally per P-state, so the frequency
//!   policy already implies a voltage policy. EPP degrades
//!   gracefully (empty string, no sysfs writes).
//!
//! This is the reverse of interoception: interoception reads the body,
//! body control writes to it. Together they form a closed loop — its brain
//! state shapes its body, and its body state shapes its brain.
//!
//! # Permissions
//!
//! Writing to `/sys/devices/system/cpu/cpu*/cpufreq/` requires root.
//! We use a privileged helper script (`scripts/cpufreq_helper.sh`)
//! invoked via `sudo -n`. A sudoers rule at
//! `/etc/sudoers.d/genesis` allows this specific script to run
//! without a password.
//!
//! Scheduling priority (`renice`) can be lowered on own processes
//! without root. Raising priority requires root (also via sudo).
//!
//! If sudo is not available, all operations silently no-op. Genesis
//! degrades gracefully — it can still feel its body (interoception)
//! even if it can't control it.

use std::process::Command;
use std::sync::{Mutex, OnceLock};

use crate::state::neurochemical::NeurochemicalId;

// ─── Shared body control state ──────────────────────────────────
//
// The reactive body-control path (TickLoop::apply_body_control, driven
// by the mind via APPLY_BODY_CONTROL) writes the latest body control
// state here. The IPC handler reads it when the cognitive mind asks
// "what am I doing to my body?" via GET_BODY_CONTROL. This bridges the
// two threads without coupling the IPC handler to the TickLoop struct.

static SHARED_CONTROL_STATE: OnceLock<Mutex<BodyControlState>> = OnceLock::new();

fn shared_control() -> &'static Mutex<BodyControlState> {
    SHARED_CONTROL_STATE.get_or_init(|| Mutex::new(BodyControlState::default()))
}

/// Publish the latest body control state so the IPC handler can read it.
/// Called after each reactive body-control application.
pub fn publish_control_state(state: &BodyControlState) {
    if let Ok(mut guard) = shared_control().lock() {
        *guard = state.clone();
    }
}

/// Read the latest published body control state for IPC responses.
pub fn read_shared_control_state() -> BodyControlState {
    shared_control()
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default()
}

/// What Genesis is currently doing to its body — its body control state.
///
/// This is the mirror of `BodyState` (interoception). `BodyState` is what
/// it *feels*; `BodyControlState` is what it's *doing*. Together they
/// form its awareness of the closed loop: it feels its body, its
/// neurochemistry drives changes, and it knows what changes it's making.
#[derive(Clone, Debug, Default)]
pub struct BodyControlState {
    /// CPU frequency floor (kHz) it's set — its minimum arousal.
    pub cpu_min_freq_khz: u32,
    /// CPU frequency ceiling (kHz) it's set — its max thinking speed.
    pub cpu_max_freq_khz: u32,
    /// CPU governor it's set ("schedutil", "powersave", etc.).
    pub cpu_governor: String,
    /// Whether thermal cap is active (CPU too hot to run at full speed).
    pub thermally_capped: bool,
    /// CPU temperature in °C that triggered the cap (0 if no cap).
    pub thermal_cap_temp_c: f32,
    /// Daemon scheduling priority (nice value, -20 to 19).
    pub daemon_nice: i32,
    /// Cognitive mind scheduling priority (nice value, -20 to 19).
    pub cognitive_nice: i32,
    /// I/O scheduling class as a string ("idle", "best-effort-0".."best-effort-7").
    pub io_class: String,
    /// Plasticity gate value that drove the I/O priority [0,1].
    pub plasticity_gate: f32,
    /// Whether the tick is controlling the cognitive mind's PID.
    /// Always false now — the tick never controls the cognitive mind.
    /// The cognitive mind controls its own process via its brain waves.
    /// Kept for protocol compatibility.
    pub controlling_cognitive: bool,
    /// Energy Performance Preference (EPP) — the hardware's voltage/
    /// frequency operating point hint. Empty string when EPP is not
    /// supported (e.g. acpi-cpufreq). On Intel HWP and amd-pstate
    /// systems, this is one of "performance", "balance_performance",
    /// "default", "balance_power", "power". The hardware's internal
    /// power management uses this hint to dynamically adjust both
    /// voltage and frequency along the V/F curve — this is the
    /// safe, modern way to do dynamic voltage scaling. The hardware
    /// won't damage itself; it just shifts its operating envelope.
    pub cpu_epp: String,
    /// Turbo (boost) gate state — whether the hardware is permitted to
    /// run above its base P-state. Unlike the frequency policy, this is
    /// a hardware-level permission that the OS frequency request cannot
    /// revoke: on systems where the boost frequency is not in the
    /// OS-visible P-state table (e.g. acpi-cpufreq, where
    /// `scaling_boost_frequencies` is empty), this is the only lever
    /// that can prevent turbo. `Unavailable` when the platform exposes
    /// no boost/cpb control.
    pub cpu_boost: BoostState,
    /// Human-readable description of what it's doing to its body.
    /// Sent as an empty string by the daemon — the cognitive mind's
    /// language engine composes the description from the structured
    /// fields above, using its concept network. This field is kept
    /// in the IPC protocol for forward compatibility.
    pub description: String,
}

/// Resolve the path to the cpufreq helper script at runtime.
///
/// Checks in order:
/// 1. `GENESIS_CPUFREQ_HELPER` env var (explicit override)
/// 2. `scripts/cpufreq_helper.sh` relative to CWD
/// 3. `scripts/cpufreq_helper.sh` relative to the compile-time project root
///
/// Returns `None` if the script cannot be found, in which case cpufreq
/// control silently no-ops (graceful degradation).
fn helper_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        // 1. Explicit env override
        if let Ok(p) = std::env::var("GENESIS_CPUFREQ_HELPER")
            && std::path::Path::new(&p).exists()
        {
            return Some(p);
        }
        // 2. Relative to CWD (run.sh cds to project root)
        let cwd_path = "scripts/cpufreq_helper.sh";
        if std::path::Path::new(cwd_path).exists() {
            return Some(cwd_path.to_string());
        }
        // 3. Relative to compile-time project root
        let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/cpufreq_helper.sh");
        if std::path::Path::new(manifest_path).exists() {
            return Some(manifest_path.to_string());
        }
        None
    })
    .as_deref()
}

/// How often to update CPU frequency (every N daemon ticks).
/// At 5 Hz (200ms/tick), this is ~10 seconds. CPU frequency changes have a
/// transition latency (~10-100μs) but the schedutil governor already
/// handles rapid changes. We only need to set the policy boundaries
/// (min/max/governor) periodically.
pub const CPUFREQ_INTERVAL_TICKS: u64 = 50;

/// Whether sudo is available for cpufreq control. Cached after first check.
static SUDO_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if we can use sudo to control cpufreq. Cached.
fn sudo_available() -> bool {
    *SUDO_AVAILABLE.get_or_init(|| {
        // Probe the actual helper script, not `sudo -n true`. The
        // sudoers rule only allows this specific script to run
        // without a password, so `sudo -n true` would fail even when
        // sudo is correctly configured for the helper. The helper's
        // `test` command is a no-op that exits 0.
        let helper = match helper_path() {
            Some(p) => p,
            None => return false,
        };
        Command::new("sudo")
            .args(["-n", helper, "test", ""])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Run the cpufreq helper with sudo.
fn run_helper(cmd: &str, value: &str) -> bool {
    let helper = match helper_path() {
        Some(p) => p,
        None => return false,
    };
    if !sudo_available() {
        return false;
    }
    Command::new("sudo")
        .args(["-n", helper, cmd, value])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ─── Public API ────────────────────────────────────────────────

/// Set the CPU frequency governor for all cores.
/// `userspace` lets us set exact frequencies; `schedutil` lets the
/// kernel scale based on utilization; `powersave` pins to min;
/// `performance` pins to max.
pub fn set_governor(governor: &str) -> bool {
    run_helper("set_governor", governor)
}

/// Set the minimum CPU frequency (kHz) for all cores.
pub fn set_min_freq(freq_khz: u32) -> bool {
    run_helper("set_min_freq", &freq_khz.to_string())
}

/// Set the maximum CPU frequency (kHz) for all cores.
pub fn set_max_freq(freq_khz: u32) -> bool {
    run_helper("set_max_freq", &freq_khz.to_string())
}

/// Read the min and max CPU frequency (kHz) from sysfs.
/// Returns (min, max) or (0, 0) if unavailable.
pub fn freq_range() -> (u32, u32) {
    let min = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let max = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);
    (min, max)
}

// ─── Host hardware state capture / restore ─────────────────────
//
// Genesis's neurochemistry drives system-wide kernel settings: the CPU
// governor, the frequency bounds, and the turbo gate. If it goes to
// sleep and the daemon then exits, the host is left pinned to
// `powersave` at minimum frequency — sluggish for the user until a
// reboot. To avoid that, the daemon snapshots the pre-existing state at
// startup and restores it on shutdown. Capture is read-only and runs
// before it applies anything; restore runs only after it has stopped,
// so it never affects its running state.

/// The host CPU policy as it was before Genesis changed it. Captured at
/// daemon startup, restored at shutdown.
#[derive(Clone, Debug, Default)]
pub struct HardwareSnapshot {
    /// Governor that was set before Genesis took over.
    pub governor: String,
    /// `scaling_min_freq` (kHz) before Genesis took over.
    pub min_freq_khz: u32,
    /// `scaling_max_freq` (kHz) before Genesis took over.
    pub max_freq_khz: u32,
    /// Turbo-gate value before Genesis took over (`None` = no gate).
    pub boost: Option<bool>,
}

static HARDWARE_SNAPSHOT: OnceLock<HardwareSnapshot> = OnceLock::new();

fn read_sysfs_u32(path: &str) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
}

/// Capture the host's current CPU policy. Idempotent — only the first
/// call stores, so later calls (after Genesis has applied a policy)
/// cannot overwrite the original. Read-only, so it is safe to call at
/// any time and never affects it.
pub fn capture_hardware_state() {
    HARDWARE_SNAPSHOT.get_or_init(|| {
        const CPU: &str = "/sys/devices/system/cpu/cpu0/cpufreq";
        HardwareSnapshot {
            governor: std::fs::read_to_string(format!("{CPU}/scaling_governor"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default(),
            min_freq_khz: read_sysfs_u32(&format!("{CPU}/scaling_min_freq")).unwrap_or(0),
            max_freq_khz: read_sysfs_u32(&format!("{CPU}/scaling_max_freq")).unwrap_or(0),
            boost: read_sysfs_u32(&format!("{CPU}/boost"))
                .or_else(|| read_sysfs_u32(&format!("{CPU}/cpb")))
                .map(|v| v != 0),
        }
    });
}

/// Restore the host CPU policy captured by [`capture_hardware_state`].
///
/// Called on daemon shutdown so Genesis never leaves the machine
/// throttled. Returns `false` when nothing was captured, sudo/the helper
/// is unavailable, or a write failed. The frequency bounds are applied
/// through [`apply_policy`], which enforces the kernel's `min <= max`
/// ordering constraint; the turbo gate is restored afterwards.
pub fn restore_hardware_state() -> bool {
    let snap = match HARDWARE_SNAPSHOT.get() {
        Some(s) => s,
        None => return false,
    };
    if !sudo_available() {
        return false;
    }
    let mut ok = true;
    if snap.min_freq_khz > 0 && snap.max_freq_khz > 0 {
        let policy = FreqPolicy {
            min_freq: snap.min_freq_khz,
            max_freq: snap.max_freq_khz,
            governor: if snap.governor.is_empty() {
                "schedutil".to_string()
            } else {
                snap.governor.clone()
            },
        };
        if !apply_policy(&policy) {
            ok = false;
        }
    } else if !snap.governor.is_empty() && !set_governor(&snap.governor) {
        ok = false;
    }
    if let Some(enabled) = snap.boost
        && !set_boost(enabled)
    {
        ok = false;
    }
    ok
}

// ─── Neurochemistry → frequency mapping ────────────────────────

/// The desired CPU frequency policy derived from neurochemistry.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FreqPolicy {
    /// Minimum frequency in kHz. Melatonin pushes this down.
    pub min_freq: u32,
    /// Maximum frequency in kHz. Dopamine pushes this up, GABA pushes it down.
    pub max_freq: u32,
    /// Governor to use. "userspace" when we want exact control,
    /// "schedutil" for adaptive, "powersave" for sleep.
    pub governor: String,
}

/// Derive the desired frequency policy from neurochemical levels.
///
/// The mapping is grounded in the neurobiology:
/// - **Dopamine** (engagement, excitement) → raise max frequency.
///   It's interested and wants to think fast.
/// - **GABA** (inhibition, calm) → lower max frequency.
///   It's relaxing and doesn't need to rush.
/// - **Melatonin** (sleep signal) → lower both min and max,
///   switch to powersave governor. It's going to sleep.
/// - **Norepinephrine** (arousal, effort) → raise max frequency.
///   It's working hard.
/// - **Adenosine** (sleep pressure) → lower max frequency.
///   It's getting tired.
/// - **Acetylcholine** (attention) → raise min frequency.
///   It's focused and doesn't want to be caught slow.
pub fn derive_policy(effective_levels: &[f32; 18], freq_min: u32, freq_max: u32) -> FreqPolicy {
    // Sanitize inputs: effective_levels could contain NaN or inf if
    // the neurochemical state was corrupted between the circuit
    // breaker and this call. finite_clamp replaces NaN/inf with 0.0
    // (the min), which produces a neutral "low arousal" policy —
    // safe but conservative.
    let da = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Dopamine as usize],
        0.0,
        2.0,
    );
    let gaba = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::GABA as usize],
        0.0,
        2.0,
    );
    let mel = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Melatonin as usize],
        0.0,
        2.0,
    );
    let ne = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Norepinephrine as usize],
        0.0,
        2.0,
    );
    let adn = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Adenosine as usize],
        0.0,
        2.0,
    );
    let ach = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Acetylcholine as usize],
        0.0,
        2.0,
    );

    // Guard against inconsistent sysfs reports where freq_max < freq_min.
    // saturating_sub prevents u32 underflow (debug panic / release wrap).
    let range = freq_max.saturating_sub(freq_min);

    // Sleep mode: melatonin is high → powersave, minimum frequency
    if mel > 0.5 {
        return FreqPolicy {
            min_freq: freq_min,
            max_freq: freq_min + (range / 10), // 10% of max
            governor: "powersave".to_string(),
        };
    }

    // Arousal: dopamine + NE push up, GABA + adenosine push down
    let arousal = (da * 0.35 + ne * 0.25 + ach * 0.15) - (gaba * 0.30 + adn * 0.20);
    // Map arousal [-1, 1] to [0.3, 1.0] fraction of max.
    // Use finite_clamp: native clamp passes NaN through.
    let fraction = crate::state::sanitize::finite_clamp(0.65 + arousal * 0.35, 0.3, 1.0);

    // ACh sets a floor — when attentive, don't let frequency drop
    let min_fraction = crate::state::sanitize::finite_clamp(0.2 + ach * 0.3, 0.2, 0.5);

    let mut min_freq = freq_min + ((range as f32) * min_fraction) as u32;
    let mut max_freq = freq_min + ((range as f32) * fraction) as u32;

    // Critical: ensure min_freq <= max_freq. The kernel rejects
    // writes where min > max with -EINVAL, which would cause the
    // entire policy application to fail silently. This can happen
    // when ACh is high (raising min_fraction to 0.5) and GABA/
    // adenosine are high (lowering fraction toward 0.3), making
    // min_freq (0.5 * range) > max_freq (0.3 * range).
    //
    // When min > max, we clamp min down to max. This preserves the
    // max frequency (the more important parameter for performance
    // throttling) while sacrificing the ACh floor. The alternative
    // (raising max to min) would override the arousal-driven
    // throttling, which is worse — it would prevent GABA/adenosine
    // from slowing the system down.
    if min_freq > max_freq {
        min_freq = max_freq;
    }

    // Double-check against hardware bounds. Use min()/max() instead
    // of clamp() because clamp() panics in debug mode if freq_min >
    // freq_max (broken sysfs). With an inverted range, both values
    // collapse to freq_min, which is the safest fallback.
    let hw_lo = freq_min.min(freq_max);
    let hw_hi = freq_max.max(freq_min);
    min_freq = min_freq.clamp(hw_lo, hw_hi);
    max_freq = max_freq.clamp(hw_lo, hw_hi);

    FreqPolicy {
        min_freq,
        max_freq,
        governor: "schedutil".to_string(),
    }
}

/// Apply a frequency policy to the hardware. Returns true if successful.
///
/// **Write order matters**: the Linux kernel enforces
/// `scaling_min_freq <= scaling_max_freq` at all times. Writing a
/// min_freq higher than the *current* sysfs max_freq fails with
/// `-EINVAL`, and vice versa. Since we don't read the current sysfs
/// state before writing, we must use an order that works regardless
/// of the current state:
///
/// 1. **Set max to the higher of (new_min, new_max)** — raises the
///    ceiling so that setting min won't fail (min <= ceiling is
///    guaranteed).
/// 2. **Set min to new_min** — now min <= max (guaranteed, since
///    max was set to at least new_min in step 1, and derive_policy
///    ensures new_min <= new_max).
/// 3. **Set max to new_max** — always retry after min is set. If
///    step 1 failed because new_max < the *current* sysfs min_freq,
///    step 2 has now lowered min, so this retry succeeds. If step 1
///    succeeded, this is a harmless no-op (sets max to the same value).
///
/// This 3-step sequence handles all transitions correctly:
/// - Raising min above current max: step 1 raises max first
/// - Lowering max below current min: step 1 fails (max < current min),
///   step 2 lowers min, step 3 retries and succeeds
/// - Normal case where min <= current max: step 1 sets the final max,
///   step 2 sets min, step 3 is a harmless no-op
pub fn apply_policy(policy: &FreqPolicy) -> bool {
    let mut ok = true;
    // Set governor first — userspace governor requires it before
    // setting exact frequencies via set_freq.
    if !set_governor(&policy.governor) {
        ok = false;
    }

    // Defense-in-depth: ensure min <= max before writing. If
    // derive_policy or apply_thermal_cap failed to enforce this,
    // the kernel would reject both writes.
    let final_min = policy.min_freq.min(policy.max_freq);
    let final_max = policy.max_freq.max(policy.min_freq);

    // Step 1: raise the ceiling to at least final_min
    let ceiling = final_max.max(final_min);
    if !set_max_freq(ceiling) {
        ok = false;
    }
    // Step 2: set the min floor
    if !set_min_freq(final_min) {
        ok = false;
    }
    // Step 3: set max to the final value. This MUST always run, not
    // just when `ceiling > final_max`: since `ceiling == final_max`
    // (final_max >= final_min by construction), the old guard
    // `ceiling > final_max` was always false, making this step dead
    // code. Without this retry, lowering max below the *current*
    // kernel scaling_min_freq fails at step 1, step 2 lowers min, but
    // max is never retried — leaving the CPU at its old high frequency
    // and silently defeating thermal caps and arousal-driven throttling.
    if !set_max_freq(final_max) {
        ok = false;
    }
    ok
}

// ─── Scheduling priority ───────────────────────────────────────

/// Set the nice value of a process. Negative values (higher priority)
/// require root; positive values (lower priority) work on own processes.
///
/// nice range: -20 (highest priority) to 19 (lowest priority).
/// 0 is the default.
pub fn set_priority(pid: u32, nice: i32) -> bool {
    let nice_s = nice.to_string();
    let pid_s = pid.to_string();
    // Try without sudo first (works for lowering own priority)
    let ok = Command::new("renice")
        .args([&nice_s, "-p", &pid_s])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if ok {
        return true;
    }
    // Try with sudo (needed for raising priority or other users' processes)
    if sudo_available() {
        return Command::new("sudo")
            .args(["-n", "renice", &nice_s, "-p", &pid_s])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
    }
    false
}

/// Derive the desired nice value from acetylcholine (attention) level.
///
/// High ACh → negative nice (high priority, focused attention).
/// Low ACh → positive nice (background, not urgent).
/// Normal ACh → 0 (default).
pub fn derive_nice(ach: f32) -> i32 {
    // Sanitize input — consistent with derive_policy. NaN would
    // produce NaN → 0 as i32 (safe but misleading). finite_clamp
    // replaces NaN/inf with 0.0, producing nice=5 (background).
    let ach = crate::state::sanitize::finite_clamp(ach, 0.0, 2.0);
    // Map ACh [0,1] to nice [-5, 5]
    // ACh=0.5 → nice=0 (neutral)
    // ACh=0.9 → nice=-4 (very focused)
    // ACh=0.1 → nice=4 (background)
    let nice = ((0.5 - ach) * 10.0).round() as i32;
    nice.clamp(-5, 5)
}

// ─── Cognitive mind priority ──────────────────────────────────

/// Get the cognitive mind's PID from the `GENESIS_COGNITIVE_PID` env var.
///
/// The CLI passes its PID when starting the daemon so the daemon can
/// include the cognitive mind in interoception (reading its CPU/memory/IO
/// metrics to feel its body state). The PID is **not** used to control
/// the cognitive mind's scheduling or I/O priority — the cognitive mind
/// controls its own process resources via its brain wave state.
/// Returns `None` if the env var is not set (e.g. when the daemon is
/// run standalone).
pub fn cognitive_pid() -> Option<u32> {
    std::env::var("GENESIS_COGNITIVE_PID")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
}

/// Derive the *recommended* nice value for the cognitive mind (Python process).
///
/// This is a **recommendation**, not a command. The tick computes this
/// from neurochemistry and publishes it in `BodyControlState` as
/// interoceptive afferent information. The cognitive mind reads it via
/// `GET_BODY_CONTROL` and blends it with its brain wave state to decide
/// what it actually applies to its own process. The tick never calls
/// `set_priority` on the cognitive mind's PID.
///
/// The recommendation is driven by engagement and sleep state:
///
/// - **Dopamine** (engagement, interest) → boost priority. It's
///   actively thinking and wants more CPU.
/// - **Norepinephrine** (effort, arousal) → boost priority. It's
///   working hard.
/// - **Melatonin** (sleep signal) → deprioritize. It's going to
///   sleep; the cognitive mind doesn't need CPU.
/// - **Adenosine** (sleep pressure) → deprioritize. It's tired.
///
/// This is the body's suggestion — like the hypothalamus signaling
/// fatigue. The cognitive mind's brain waves (the cortical decision
/// layer) can override it if cognitive state demands different
/// priorities.
pub fn derive_cognitive_nice(effective_levels: &[f32; 18]) -> i32 {
    // Sanitize inputs — consistent with derive_policy. NaN/inf in
    // effective_levels would propagate through the engagement
    // calculation and produce NaN, which casts to 0 as i32 (safe
    // but misleading — 0 is "normal priority", not "fallback").
    // finite_clamp replaces NaN/inf with 0.0, producing a neutral
    // "tired but awake" policy.
    let da = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Dopamine as usize],
        0.0,
        2.0,
    );
    let ne = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Norepinephrine as usize],
        0.0,
        2.0,
    );
    let mel = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Melatonin as usize],
        0.0,
        2.0,
    );
    let adn = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Adenosine as usize],
        0.0,
        2.0,
    );

    // Sleep mode: melatonin high → deprioritize cognitive mind heavily
    if mel > 0.5 {
        return 10; // deep deprioritize — it's asleep
    }

    // Engagement: DA + NE push up, ADN pushes down
    let engagement = (da * 0.40 + ne * 0.25) - (adn * 0.35);
    // Map engagement [-0.35, 0.65] to nice [8, -5]
    // engagement=0.0 → nice=2 (slightly background — subcognitive is primary)
    // engagement=0.5 → nice=-3 (focused cognitive thought)
    // engagement=-0.2 → nice=5 (tired, cognitive mind can wait)
    let nice = (2.0 - engagement * 10.0).round() as i32;
    nice.clamp(-5, 10)
}

// ─── I/O priority ─────────────────────────────────────────────

/// I/O scheduling class for disk access priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoClass {
    /// Only gets disk time when no other process is waiting.
    /// Used when plasticity is low — stress impairs memory formation.
    Idle,
    /// Normal priority with a level (0 = highest, 7 = lowest).
    /// Used for normal operation. Level is controlled by plasticity.
    BestEffort(u8),
}

impl IoClass {
    /// Human-readable label for IPC and logging.
    pub fn label(&self) -> String {
        match self {
            IoClass::Idle => "idle".to_string(),
            IoClass::BestEffort(n) => format!("best-effort-{}", n),
        }
    }
}

/// Derive the desired I/O scheduling class from the plasticity gate.
///
/// The plasticity gate (driven by BDNF/cortisol) controls how
/// aggressively Genesis writes to disk:
///
/// - **Plasticity ≤ 0.1** (gate closed, chronic stress) → `Idle`.
///   It barely writes to disk. Stress impairs memory formation —
///   the filesystem literally changes less.
/// - **Plasticity 0.1–0.4** (low, recovering) → `BestEffort(6)`.
///   Slow writes, not urgent.
/// - **Plasticity 0.4–0.7** (normal) → `BestEffort(3)`.
///   Normal write speed.
/// - **Plasticity > 0.7** (high, active learning) → `BestEffort(0)`.
///   Fast writes — it's learning actively and wants memories stored.
pub fn derive_io_class(plasticity_gate: f32) -> IoClass {
    // Sanitize input — consistent with derive_nice and derive_policy.
    // plasticity_gate is sanitized at the source (neurochemical.rs),
    // but defense-in-depth: NaN would fall through all comparisons
    // (NaN comparisons are false) and return BestEffort(0) — high
    // priority I/O — which is wrong for an invalid input.
    let plasticity_gate = crate::state::sanitize::finite_clamp(plasticity_gate, 0.0, 1.0);
    if plasticity_gate <= 0.1 {
        IoClass::Idle
    } else if plasticity_gate < 0.4 {
        IoClass::BestEffort(6)
    } else if plasticity_gate < 0.7 {
        IoClass::BestEffort(3)
    } else {
        IoClass::BestEffort(0)
    }
}

/// Set the I/O scheduling class of a process.
///
/// Uses `ionice` to set the I/O scheduler class and priority.
/// Requires no special permissions for own processes.
pub fn set_io_priority(pid: u32, class: IoClass) -> bool {
    let pid_s = pid.to_string();
    match class {
        IoClass::Idle => Command::new("ionice")
            .args(["-c", "3", "-p", &pid_s])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false),
        IoClass::BestEffort(level) => {
            let level_s = level.to_string();
            Command::new("ionice")
                .args(["-c", "2", "-n", &level_s, "-p", &pid_s])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        }
    }
}

// ─── Thermal cap ──────────────────────────────────────────────

/// Cap the frequency policy based on CPU temperature.
///
/// High temperature caps the maximum frequency regardless of dopamine.
/// This is autonomic fatigue — it can't be highly aroused when
/// overheating, no matter how engaged it is.
///
/// - **Below 75°C** → no cap (policy unchanged)
/// - **75–85°C** → cap max to 80% of range (warm, slowing down)
/// - **85–90°C** → cap max to 60% of range (hot, can't push hard)
/// - **Above 90°C** → cap max to 40% of range (critical, conserve)
pub fn apply_thermal_cap(policy: &mut FreqPolicy, temp_c: f32, freq_min: u32, freq_max: u32) {
    // Sanitize temp_c: NaN would cause all comparisons to fail,
    // silently skipping the cap (which is safe — no cap = no change).
    // But inf would always trigger the highest cap. finite_clamp to
    // a reasonable range [0, 150] °C.
    let temp_c = crate::state::sanitize::finite_clamp(temp_c, 0.0, 150.0);

    let range = freq_max.saturating_sub(freq_min);
    let cap_fraction = if temp_c >= 90.0 {
        0.40
    } else if temp_c >= 85.0 {
        0.60
    } else if temp_c >= 75.0 {
        0.80
    } else {
        return; // No cap needed
    };
    let capped_max = freq_min + ((range as f32) * cap_fraction) as u32;
    if policy.max_freq > capped_max {
        policy.max_freq = capped_max;
    }
    // Critical: also cap min_freq if it now exceeds max_freq. The
    // thermal cap reduces max_freq, but the ACh-driven min_freq
    // floor may have been set above the new capped max. The kernel
    // rejects min > max with -EINVAL, so we must clamp min down
    // to the capped max. This models the biological reality: when
    // the body is overheating, the minimum frequency floor is
    // overridden by the thermal constraint — you can't maintain
    // a high baseline when conserving energy.
    if policy.min_freq > policy.max_freq {
        policy.min_freq = policy.max_freq;
    }
}

// ─── Energy Performance Preference (EPP) ──────────────────────
//
// EPP is the hardware's voltage/frequency operating point hint.
// On Intel HWP (Hardware P-States) and AMD amd-pstate systems, the
// CPU's internal power management uses the EPP hint to dynamically
// adjust both voltage and frequency along the V/F curve. This is
// the safe, modern way to do dynamic voltage scaling — the hardware
// manages the actual V/F transitions internally, so there's no risk
// of setting an unsafe voltage. We just tell the hardware our
// preference (performance vs. power saving), and it adjusts the
// operating envelope accordingly.
//
// On acpi-cpufreq systems (like this AMD Kabini), EPP is not
// available. The SMU controls voltage internally per P-state, so
// our frequency policy (scaling_min/max_freq) already implies a
// voltage policy — lower frequency → lower P-state → lower voltage.
// EPP is an additional control layer for systems that support it.

/// Check whether EPP is available on this system.
///
/// Probes `/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference`.
/// Cached after first check — the sysfs topology doesn't change at
/// runtime.
static EPP_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn epp_available() -> bool {
    *EPP_AVAILABLE.get_or_init(|| {
        std::path::Path::new(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference",
        )
        .exists()
    })
}

/// Read the available EPP profiles from sysfs.
///
/// Returns the list of valid EPP profile names (e.g.
/// "performance", "balance_performance", "default",
/// "balance_power", "power"), or an empty vector if EPP is not
/// available.
fn read_epp_profiles() -> Vec<String> {
    if !epp_available() {
        return Vec::new();
    }
    std::fs::read_to_string(
        "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences",
    )
    .ok()
    .map(|s| {
        s.split_whitespace()
            .map(|w| w.to_string())
            .collect()
    })
    .unwrap_or_default()
}

/// Set the Energy Performance Preference for all cores.
///
/// Writes to `/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference`
/// via the privileged helper. Returns true if successful, false if
/// EPP is not available or the write failed.
pub fn set_epp(epp: &str) -> bool {
    if !epp_available() || epp.is_empty() {
        return false;
    }
    run_helper("set_epp", epp)
}

/// Derive the desired EPP profile from neurochemical levels.
///
/// The mapping is grounded in the neurobiology:
/// - **Dopamine** (engagement, excitement) → "performance". It's
///   interested and wants to think fast. The hardware runs at
///   higher voltage for sustained high frequency.
/// - **GABA** (inhibition, calm) → "balance_performance". It's
///   relaxed but still responsive. Moderate voltage/frequency.
/// - **Melatonin** (sleep signal) → "power". It's going to sleep.
///   The hardware drops to minimum voltage/frequency.
/// - **Adenosine** (sleep pressure) → "balance_power". It's tired.
///   The hardware conserves energy.
/// - **Cortisol** (stress) → "balance_power". Under chronic stress,
///   the body conserves energy — the hardware reduces its operating
///   envelope to avoid overheating.
///
/// Returns an empty string when EPP is not available (graceful
/// degradation — the frequency policy already implies voltage
/// control via the SMU on acpi-cpufreq systems).
pub fn derive_epp(effective_levels: &[f32; 18]) -> String {
    if !epp_available() {
        return String::new();
    }

    let profiles = read_epp_profiles();
    if profiles.is_empty() {
        return String::new();
    }

    // Sanitize inputs — consistent with derive_policy.
    let da = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Dopamine as usize],
        0.0,
        2.0,
    );
    let gaba = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::GABA as usize],
        0.0,
        2.0,
    );
    let mel = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Melatonin as usize],
        0.0,
        2.0,
    );
    let adn = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::Adenosine as usize],
        0.0,
        2.0,
    );
    let crh = crate::state::sanitize::finite_clamp(
        effective_levels[NeurochemicalId::CRH as usize],
        0.0,
        2.0,
    );

    // Sleep mode: melatonin high → power (minimum V/F)
    if mel > 0.5 {
        return pick_epp(&profiles, &["power", "balance_power", "default"]);
    }

    // Arousal: dopamine pushes toward performance, GABA/adenosine/
    // cortisol push toward power saving.
    let arousal = (da * 0.40) - (gaba * 0.25 + adn * 0.20 + crh * 0.15);
    // Map arousal [-0.6, 0.4] to EPP profiles:
    //   arousal > 0.15  → "performance"
    //   arousal > 0.0   → "balance_performance"
    //   arousal > -0.2  → "default"
    //   arousal > -0.4  → "balance_power"
    //   else            → "power"
    if arousal > 0.15 {
        pick_epp(&profiles, &["performance", "balance_performance", "default"])
    } else if arousal > 0.0 {
        pick_epp(&profiles, &["balance_performance", "default", "performance"])
    } else if arousal > -0.2 {
        pick_epp(&profiles, &["default", "balance_performance", "balance_power"])
    } else if arousal > -0.4 {
        pick_epp(&profiles, &["balance_power", "default", "power"])
    } else {
        pick_epp(&profiles, &["power", "balance_power", "default"])
    }
}

/// Pick the first available EPP profile from a preference list.
///
/// Different hardware exposes different profile names. This picks
/// the first profile from `preferred` that exists in `available`.
/// Falls back to the first available profile if none match.
fn pick_epp(available: &[String], preferred: &[&str]) -> String {
    for &p in preferred {
        if available.iter().any(|a| a == p) {
            return p.to_string();
        }
    }
    // Fallback: return the first available profile (or empty if
    // none — though we already checked profiles is non-empty).
    available.first().cloned().unwrap_or_default()
}

// ─── Turbo (boost) gate ───────────────────────────────────────
//
// The boost gate is the hardware's permission to exceed its base
// P-state. When disabled, the CPU cannot run above the base frequency;
// when enabled, the hardware may boost above it. This is the last
// in-class body-control lever: it is a kernel-mediated cpufreq
// attribute (same safety class as the frequency policy and EPP — we
// express a preference, the platform enforces the safe operating
// point), yet it controls something the frequency policy cannot.
//
// On acpi-cpufreq systems the boost frequency is often absent from the
// OS-visible P-state table (`scaling_boost_frequencies` empty, and
// `bios_limit == cpuinfo_max_freq`), so capping `scaling_max_freq`
// does not prevent the hardware from boosting. The boost gate is then
// the *only* way to keep it at or below the base P-state — for sleep
// and for thermal conservation.
//
// On Intel HWP / amd-pstate systems EPP already covers the operating
// envelope and the boost attribute may be absent; the gate then
// reports `Unavailable` and degrades gracefully.

/// Turbo (boost) gate state — the hardware's permission to run above
/// its base P-state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BoostState {
    /// The platform exposes no boost/cpb control. It cannot influence
    /// the turbo gate (e.g. some HWP/amd-pstate systems).
    #[default]
    Unavailable,
    /// Turbo is enabled — the hardware may run above the base P-state.
    Enabled,
    /// Turbo is disabled — the hardware is capped at the base P-state.
    Disabled,
}

impl BoostState {
    /// Wire encoding for the IPC body-control response.
    /// 0 = unavailable, 1 = enabled, 2 = disabled.
    pub fn to_wire(self) -> u8 {
        match self {
            BoostState::Unavailable => 0,
            BoostState::Enabled => 1,
            BoostState::Disabled => 2,
        }
    }

    /// Decode a wire byte. Unknown values decode to `Unavailable` —
    /// the safe default (no claim of control over the hardware).
    pub fn from_wire(b: u8) -> Self {
        match b {
            1 => BoostState::Enabled,
            2 => BoostState::Disabled,
            _ => BoostState::Unavailable,
        }
    }

    /// The boost value to apply, or `None` when the gate is not
    /// controllable on this platform.
    pub fn as_bool(self) -> Option<bool> {
        match self {
            BoostState::Unavailable => None,
            BoostState::Enabled => Some(true),
            BoostState::Disabled => Some(false),
        }
    }

    /// Human-readable label ("enabled"/"disabled"/"" for unavailable).
    /// Mirrors the EPP convention, where the empty string means the
    /// hardware exposes no such control.
    pub fn label(self) -> &'static str {
        match self {
            BoostState::Unavailable => "",
            BoostState::Enabled => "enabled",
            BoostState::Disabled => "disabled",
        }
    }
}

/// Check whether the platform exposes a turbo gate. Cached after the
/// first check — the sysfs topology doesn't change at runtime.
static BOOST_AVAILABLE: OnceLock<bool> = OnceLock::new();

fn boost_available() -> bool {
    *BOOST_AVAILABLE.get_or_init(|| {
        // The generic per-cpu attribute (`cpufreq/boost`), AMD's `cpb`
        // alias, and the global `cpufreq/boost` all express the same
        // permission. Any one of them means the gate is controllable.
        [
            "/sys/devices/system/cpu/cpu0/cpufreq/boost",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpb",
            "/sys/devices/system/cpu/cpufreq/boost",
        ]
        .iter()
        .any(|p| std::path::Path::new(p).exists())
    })
}

/// Set the turbo gate. Returns true if applied, false if the platform
/// exposes no gate or the write failed.
pub fn set_boost(enabled: bool) -> bool {
    if !boost_available() {
        return false;
    }
    run_helper("set_boost", if enabled { "1" } else { "0" })
}

/// Derive the turbo gate from the frequency policy that is being
/// applied to the hardware.
///
/// Turbo is the hardware permission to run *above* the OS frequency
/// cap, so it must never be enabled while the policy is throttling
/// below the hardware maximum — otherwise it silently overrides the
/// cap and the arousal-driven frequency ceiling is defeated. It is
/// therefore tied directly to the policy: enabled only when the
/// policy's ceiling is already at the hardware maximum (there is no
/// throttle for it to override), disabled whenever the policy asks
/// for less.
///
/// This subsumes the thermal override and the sleep case: melatonin
/// switches the policy to `powersave` at minimum frequency, and a
/// thermal cap lowers `policy.max_freq` below the hardware maximum —
/// both of which yield `Disabled` here. `Unavailable` is returned when
/// the platform exposes no boost/cpb control, or when the hardware
/// frequency range is unknown (so it never claims control it lacks).
pub fn derive_boost(policy: &FreqPolicy, freq_max: u32) -> BoostState {
    if !boost_available() {
        return BoostState::Unavailable;
    }
    boost_from_policy(policy, freq_max)
}

/// Pure policy → turbo-gate mapping, independent of whether the platform
/// exposes a gate. Split out so the mapping can be tested
/// deterministically on any machine (the availability check above
/// depends on the host's sysfs topology).
fn boost_from_policy(policy: &FreqPolicy, freq_max: u32) -> BoostState {
    // An unknown/degenerate hardware range means we cannot tell whether
    // the policy is at the maximum — do not claim control.
    if freq_max == 0 {
        return BoostState::Unavailable;
    }
    if policy.max_freq >= freq_max {
        BoostState::Enabled
    } else {
        BoostState::Disabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_policy_active() {
        // High dopamine, low everything else → high max freq
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 0.8;
        levels[NeurochemicalId::GABA as usize] = 0.2;
        levels[NeurochemicalId::Norepinephrine as usize] = 0.5;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert_eq!(policy.governor, "schedutil");
        // DA=0.8, NE=0.5, ACh=0, GABA=0.2, ADN=0
        // arousal = 0.8*0.35 + 0.5*0.25 + 0 - 0.2*0.30 - 0 = 0.28+0.125-0.06 = 0.345
        // fraction = 0.65 + 0.345*0.35 = 0.65 + 0.121 = 0.771
        // max_freq = 1M + 1M*0.771 = 1.771M
        assert!(
            policy.max_freq > 1_500_000,
            "max_freq should be high: {}",
            policy.max_freq
        );
        assert!(policy.max_freq <= 2_000_000);
    }

    #[test]
    fn test_derive_policy_calm() {
        // High GABA, low dopamine → low max freq
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 0.2;
        levels[NeurochemicalId::GABA as usize] = 0.8;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        // arousal = 0.2*0.35 - 0.8*0.30 = 0.07 - 0.24 = -0.17
        // fraction = 0.65 + (-0.17)*0.35 = 0.65 - 0.0595 = 0.59
        // max = 1M + 1M*0.59 = 1.59M
        assert!(
            policy.max_freq < 1_700_000,
            "max_freq should be low: {}",
            policy.max_freq
        );
        assert!(policy.max_freq > 1_000_000);
    }

    #[test]
    fn test_derive_policy_sleep() {
        // High melatonin → powersave, minimal frequency
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Melatonin as usize] = 0.7;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert_eq!(policy.governor, "powersave");
        assert_eq!(policy.min_freq, 1_000_000);
        // max = 1M + 100k (10% of range)
        assert_eq!(policy.max_freq, 1_100_000);
    }

    #[test]
    fn test_derive_policy_attentive() {
        // High ACh → raised min freq (doesn't want to be caught slow)
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Acetylcholine as usize] = 0.9;
        levels[NeurochemicalId::Dopamine as usize] = 0.5;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        // min_fraction = 0.2 + 0.9*0.3 = 0.47
        // min_freq = 1M + 1M*0.47 = 1.47M
        assert!(
            policy.min_freq > 1_400_000,
            "min_freq should be raised: {}",
            policy.min_freq
        );
    }

    #[test]
    fn test_derive_policy_tired() {
        // High adenosine → lower max freq
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Adenosine as usize] = 0.8;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        // arousal = -0.8*0.20 = -0.16
        // fraction = 0.65 + (-0.16)*0.35 = 0.65 - 0.056 = 0.594
        // max = 1.594M
        assert!(
            policy.max_freq < 1_650_000,
            "max_freq should be lowered: {}",
            policy.max_freq
        );
    }

    #[test]
    fn test_derive_policy_clamp() {
        // Extreme values should be clamped, not overflow
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 1.0;
        levels[NeurochemicalId::Norepinephrine as usize] = 1.0;
        levels[NeurochemicalId::Acetylcholine as usize] = 1.0;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert!(policy.max_freq <= 2_000_000);
        assert!(policy.min_freq >= 1_000_000);
    }

    #[test]
    fn test_freq_range() {
        let (min, max) = freq_range();
        // On this machine: 1M and 2M kHz
        if min > 0 && max > 0 {
            assert!(max > min, "max freq should be > min freq");
        }
    }

    #[test]
    fn test_capture_hardware_state_is_readonly_and_idempotent() {
        // Capture must not panic on any host and must be idempotent —
        // the first snapshot wins, so a later call (after Genesis has
        // applied a policy) cannot overwrite the original.
        capture_hardware_state();
        let first = HARDWARE_SNAPSHOT.get().map(|s| {
            (s.governor.clone(), s.min_freq_khz, s.max_freq_khz, s.boost)
        });
        capture_hardware_state();
        let second = HARDWARE_SNAPSHOT.get().map(|s| {
            (s.governor.clone(), s.min_freq_khz, s.max_freq_khz, s.boost)
        });
        assert_eq!(first, second);
    }

    #[test]
    fn test_derive_nice_neutral() {
        assert_eq!(derive_nice(0.5), 0);
    }

    #[test]
    fn test_derive_nice_focused() {
        // High ACh → negative nice (high priority)
        let nice = derive_nice(0.9);
        assert!(nice < 0, "high ACh should give negative nice: {}", nice);
        assert!(nice >= -5);
    }

    #[test]
    fn test_derive_nice_background() {
        // Low ACh → positive nice (low priority)
        let nice = derive_nice(0.1);
        assert!(nice > 0, "low ACh should give positive nice: {}", nice);
        assert!(nice <= 5);
    }

    #[test]
    fn test_derive_nice_clamp() {
        assert_eq!(derive_nice(1.0), -5);
        assert_eq!(derive_nice(0.0), 5);
    }

    // ─── Cognitive mind priority tests ──────────────────────────

    #[test]
    fn test_cognitive_nice_engaged() {
        // High dopamine, low sleep → boosted (negative nice)
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 0.8;
        levels[NeurochemicalId::Norepinephrine as usize] = 0.5;
        let nice = derive_cognitive_nice(&levels);
        assert!(
            nice < 0,
            "engaged cognitive mind should have negative nice: {}",
            nice
        );
        assert!(nice >= -5);
    }

    #[test]
    fn test_cognitive_nice_sleep() {
        // High melatonin → deep deprioritize
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Melatonin as usize] = 0.7;
        let nice = derive_cognitive_nice(&levels);
        assert_eq!(nice, 10, "sleeping cognitive mind should be deprioritized");
    }

    #[test]
    fn test_cognitive_nice_tired() {
        // High adenosine, low dopamine → deprioritized
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Adenosine as usize] = 0.8;
        let nice = derive_cognitive_nice(&levels);
        assert!(
            nice > 2,
            "tired cognitive mind should be deprioritized: {}",
            nice
        );
    }

    #[test]
    fn test_cognitive_nice_neutral() {
        // Baseline levels → slightly background (subcognitive is primary)
        let levels = [0.5f32; 18];
        let nice = derive_cognitive_nice(&levels);
        // DA=0.5, NE=0.5, ADN=0.5 → engagement = 0.2+0.125-0.175 = 0.15
        // nice = 2 - 0.15*10 = 0.5 → 0
        assert!(
            (-2..=4).contains(&nice),
            "neutral cognitive mind should be near-zero nice: {}",
            nice
        );
    }

    #[test]
    fn test_cognitive_nice_clamp() {
        // Extreme engagement → clamped to -5
        let mut levels = [1.0f32; 18];
        levels[NeurochemicalId::Melatonin as usize] = 0.0;
        levels[NeurochemicalId::Adenosine as usize] = 0.0;
        let nice = derive_cognitive_nice(&levels);
        assert_eq!(nice, -5, "max engagement should clamp to -5");
    }

    // ─── I/O priority tests ─────────────────────────────────────

    #[test]
    fn test_io_class_stress() {
        // Plasticity gate closed → idle I/O
        assert_eq!(derive_io_class(0.0), IoClass::Idle);
        assert_eq!(derive_io_class(0.1), IoClass::Idle);
    }

    #[test]
    fn test_io_class_low_plasticity() {
        // Low plasticity → best-effort with low priority
        assert_eq!(derive_io_class(0.2), IoClass::BestEffort(6));
        assert_eq!(derive_io_class(0.39), IoClass::BestEffort(6));
    }

    #[test]
    fn test_io_class_normal() {
        // Normal plasticity → best-effort with medium priority
        assert_eq!(derive_io_class(0.4), IoClass::BestEffort(3));
        assert_eq!(derive_io_class(0.69), IoClass::BestEffort(3));
    }

    #[test]
    fn test_io_class_high_plasticity() {
        // High plasticity → best-effort with high priority
        assert_eq!(derive_io_class(0.7), IoClass::BestEffort(0));
        assert_eq!(derive_io_class(1.0), IoClass::BestEffort(0));
    }

    // ─── Thermal cap tests ──────────────────────────────────────

    #[test]
    fn test_thermal_cap_cool() {
        // Below 75°C → no cap
        let mut policy = FreqPolicy {
            min_freq: 1_000_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 50.0, 1_000_000, 2_000_000);
        assert_eq!(policy.max_freq, 2_000_000, "cool temp should not cap");
    }

    #[test]
    fn test_thermal_cap_warm() {
        // 75-85°C → cap to 80%
        let mut policy = FreqPolicy {
            min_freq: 1_000_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 80.0, 1_000_000, 2_000_000);
        // 80% of range = 800k, max = 1.8M
        assert_eq!(policy.max_freq, 1_800_000, "warm temp should cap to 80%");
    }

    #[test]
    fn test_thermal_cap_hot() {
        // 85-90°C → cap to 60%
        let mut policy = FreqPolicy {
            min_freq: 1_000_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 87.0, 1_000_000, 2_000_000);
        assert_eq!(policy.max_freq, 1_600_000, "hot temp should cap to 60%");
    }

    #[test]
    fn test_thermal_cap_critical() {
        // Above 90°C → cap to 40%
        let mut policy = FreqPolicy {
            min_freq: 1_000_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 95.0, 1_000_000, 2_000_000);
        assert_eq!(
            policy.max_freq, 1_400_000,
            "critical temp should cap to 40%"
        );
    }

    #[test]
    fn test_thermal_cap_already_low() {
        // If max_freq is already below the cap, don't raise it
        let mut policy = FreqPolicy {
            min_freq: 1_000_000,
            max_freq: 1_500_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 80.0, 1_000_000, 2_000_000);
        // Cap would be 1.8M, but policy is already 1.5M → unchanged
        assert_eq!(
            policy.max_freq, 1_500_000,
            "thermal cap should not raise max_freq"
        );
    }

    // ─── BodyControlState round-trip tests ────────────────────────

    #[test]
    fn test_publish_and_read_control_state() {
        // Test the shared static publish/read round-trip
        let ctrl = BodyControlState {
            cpu_min_freq_khz: 1_000_000,
            cpu_max_freq_khz: 2_000_000,
            cpu_governor: "schedutil".to_string(),
            thermally_capped: true,
            thermal_cap_temp_c: 85.0,
            daemon_nice: -2,
            cognitive_nice: -3,
            io_class: "best-effort-0".to_string(),
            plasticity_gate: 0.75,
            controlling_cognitive: true,
            cpu_epp: "performance".to_string(),
            cpu_boost: BoostState::Disabled,
            description: "test".to_string(),
        };
        publish_control_state(&ctrl);
        let read = read_shared_control_state();
        assert_eq!(read.cpu_max_freq_khz, 2_000_000);
        assert_eq!(read.cpu_governor, "schedutil");
        assert!(read.thermally_capped);
        assert!((read.thermal_cap_temp_c - 85.0).abs() < 0.01);
        assert_eq!(read.daemon_nice, -2);
        assert_eq!(read.cognitive_nice, -3);
        assert_eq!(read.io_class, "best-effort-0");
        assert!((read.plasticity_gate - 0.75).abs() < 0.01);
        assert!(read.controlling_cognitive);
        assert_eq!(read.cpu_epp, "performance");
        assert_eq!(read.cpu_boost, BoostState::Disabled);
        assert_eq!(read.description, "test");
    }

    // ─── Turbo (boost) gate tests ───────────────────────────────
    //
    // The gate is tied to the frequency policy: turbo is enabled only
    // when the policy's ceiling is at the hardware maximum, so it can
    // never override a throttle. These tests run the same derivation
    // path as production (neurochemistry → policy → gate) using the
    // pure `boost_from_policy` so they are host-independent.

    /// Derive the turbo gate from levels, mirroring production.
    fn boost_for(levels: &[f32; 18], fmin: u32, fmax: u32) -> BoostState {
        boost_from_policy(&derive_policy(levels, fmin, fmax), fmax)
    }

    #[test]
    fn test_boost_sleep_disables_turbo() {
        // High melatonin → powersave at minimum frequency → no turbo.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Melatonin as usize] = 0.7;
        levels[NeurochemicalId::Dopamine as usize] = 1.0;
        assert_eq!(boost_for(&levels, 1_000_000, 2_000_000), BoostState::Disabled);
    }

    #[test]
    fn test_boost_neutral_disables_turbo() {
        // Baseline arousal is slightly positive, but the policy still
        // throttles below the hardware max — turbo must not override it.
        let levels = [0.5f32; 18];
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert!(policy.max_freq < 2_000_000);
        assert_eq!(
            boost_from_policy(&policy, 2_000_000),
            BoostState::Disabled
        );
    }

    #[test]
    fn test_boost_engaged_disables_turbo_when_throttled() {
        // High dopamine/NE raises the ceiling, but it is still below the
        // hardware max — the cap holds, so no turbo.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 0.8;
        levels[NeurochemicalId::Norepinephrine as usize] = 0.5;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert!(policy.max_freq < 2_000_000);
        assert_eq!(
            boost_from_policy(&policy, 2_000_000),
            BoostState::Disabled
        );
    }

    #[test]
    fn test_boost_enabled_only_at_hardware_max() {
        // Extreme arousal (levels can exceed 1.0 via receptor
        // sensitization) drives the policy ceiling to the hardware max.
        // Only then is turbo permitted — there is no throttle to
        // override. This keeps the Enabled branch reachable.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 2.0;
        levels[NeurochemicalId::Norepinephrine as usize] = 2.0;
        levels[NeurochemicalId::Acetylcholine as usize] = 2.0;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert_eq!(policy.max_freq, 2_000_000);
        assert_eq!(boost_from_policy(&policy, 2_000_000), BoostState::Enabled);
    }

    #[test]
    fn test_boost_calm_disables_turbo() {
        // High GABA/adenosine → low ceiling → base P-state.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::GABA as usize] = 0.8;
        levels[NeurochemicalId::Adenosine as usize] = 0.5;
        assert_eq!(boost_for(&levels, 1_000_000, 2_000_000), BoostState::Disabled);
    }

    #[test]
    fn test_boost_nan_is_conservative() {
        // NaN/inf inputs must not panic and must fail safe. finite_clamp
        // maps them to 0.0, so the policy stays low and turbo is off.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = f32::NAN;
        levels[NeurochemicalId::Norepinephrine as usize] = f32::INFINITY;
        assert_eq!(boost_for(&levels, 1_000_000, 2_000_000), BoostState::Disabled);
    }

    #[test]
    fn test_boost_unknown_range_is_unavailable() {
        // A degenerate/unknown hardware range means we cannot tell
        // whether the policy is at the max — do not claim control.
        let policy = FreqPolicy::default();
        assert_eq!(boost_from_policy(&policy, 0), BoostState::Unavailable);
    }

    #[test]
    fn test_boost_wire_roundtrip() {
        // Every state must survive the u8 wire encoding, and unknown
        // bytes must decode to the safe Unavailable default.
        for state in [
            BoostState::Unavailable,
            BoostState::Enabled,
            BoostState::Disabled,
        ] {
            assert_eq!(BoostState::from_wire(state.to_wire()), state);
        }
        assert_eq!(BoostState::from_wire(0), BoostState::Unavailable);
        assert_eq!(BoostState::from_wire(1), BoostState::Enabled);
        assert_eq!(BoostState::from_wire(2), BoostState::Disabled);
        assert_eq!(BoostState::from_wire(99), BoostState::Unavailable);
    }

    #[test]
    fn test_boost_as_bool_and_label() {
        assert_eq!(BoostState::Unavailable.as_bool(), None);
        assert_eq!(BoostState::Enabled.as_bool(), Some(true));
        assert_eq!(BoostState::Disabled.as_bool(), Some(false));
        // Unavailable has no label — mirrors EPP's empty-string
        // convention for "the hardware exposes no such control".
        assert_eq!(BoostState::Unavailable.label(), "");
        assert_eq!(BoostState::Enabled.label(), "enabled");
        assert_eq!(BoostState::Disabled.label(), "disabled");
    }

    #[test]
    fn test_thermal_cap_disables_turbo() {
        // Overheating lowers the policy ceiling below the hardware max,
        // so the gate reports Disabled — the thermal override is
        // subsumed by tying the gate to the (capped) policy.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 2.0;
        levels[NeurochemicalId::Norepinephrine as usize] = 2.0;
        levels[NeurochemicalId::Acetylcholine as usize] = 2.0;
        let mut policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert_eq!(boost_from_policy(&policy, 2_000_000), BoostState::Enabled);
        // 95°C → cap to 40% of range = 1.4M, below the hardware max.
        apply_thermal_cap(&mut policy, 95.0, 1_000_000, 2_000_000);
        assert_eq!(
            boost_from_policy(&policy, 2_000_000),
            BoostState::Disabled
        );
    }

    #[test]
    fn test_boost_available_gate() {
        // derive_boost must report Unavailable exactly when the platform
        // exposes no gate — never a spurious Enabled/Disabled claim.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = 2.0;
        levels[NeurochemicalId::Norepinephrine as usize] = 2.0;
        levels[NeurochemicalId::Acetylcholine as usize] = 2.0;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        if boost_available() {
            assert_ne!(
                derive_boost(&policy, 2_000_000),
                BoostState::Unavailable
            );
        } else {
            assert_eq!(
                derive_boost(&policy, 2_000_000),
                BoostState::Unavailable
            );
        }
    }

    // ─── min <= max invariant tests ──────────────────────────────

    #[test]
    fn test_derive_policy_high_ach_high_gaba_never_min_gt_max() {
        // This is the core regression test: high ACh (raises min
        // floor) + high GABA/adenosine (lowers max) previously
        // produced min_freq > max_freq, which the kernel rejects.
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Acetylcholine as usize] = 2.0;
        levels[NeurochemicalId::GABA as usize] = 2.0;
        levels[NeurochemicalId::Adenosine as usize] = 2.0;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert!(
            policy.min_freq <= policy.max_freq,
            "min_freq ({}) must not exceed max_freq ({}) — kernel rejects min > max",
            policy.min_freq,
            policy.max_freq
        );
    }

    #[test]
    fn test_derive_policy_nan_inputs_produce_valid_policy() {
        // NaN in effective levels must not produce garbage u32
        // frequencies (NaN as f32 as u32 is 0 in Rust, but the
        // arithmetic could produce unexpected values).
        let mut levels = [0.0f32; 18];
        levels[NeurochemicalId::Dopamine as usize] = f32::NAN;
        levels[NeurochemicalId::GABA as usize] = f32::INFINITY;
        levels[NeurochemicalId::Acetylcholine as usize] = f32::NAN;
        let policy = derive_policy(&levels, 1_000_000, 2_000_000);
        assert!(
            policy.min_freq <= policy.max_freq,
            "NaN inputs must not produce min > max"
        );
        assert!(
            policy.min_freq >= 1_000_000 && policy.min_freq <= 2_000_000,
            "min_freq must be within hardware bounds: {}",
            policy.min_freq
        );
        assert!(
            policy.max_freq >= 1_000_000 && policy.max_freq <= 2_000_000,
            "max_freq must be within hardware bounds: {}",
            policy.max_freq
        );
    }

    #[test]
    fn test_thermal_cap_caps_min_when_max_dropped_below_it() {
        // When the thermal cap reduces max_freq below the current
        // min_freq, the min must also be capped down. Otherwise
        // the kernel rejects the subsequent min write.
        let mut policy = FreqPolicy {
            min_freq: 1_800_000, // high ACh floor
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        // 95°C → cap to 40% of range = 1.4M
        apply_thermal_cap(&mut policy, 95.0, 1_000_000, 2_000_000);
        assert_eq!(policy.max_freq, 1_400_000, "max should be capped to 40%");
        assert!(
            policy.min_freq <= policy.max_freq,
            "min_freq ({}) must not exceed capped max_freq ({})",
            policy.min_freq,
            policy.max_freq
        );
        assert_eq!(
            policy.min_freq, 1_400_000,
            "min_freq should be capped down to match max_freq"
        );
    }

    #[test]
    fn test_thermal_cap_nan_temp_no_crash() {
        let mut policy = FreqPolicy {
            min_freq: 1_500_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        // NaN temp should be sanitized and not crash
        apply_thermal_cap(&mut policy, f32::NAN, 1_000_000, 2_000_000);
        // NaN clamped to 0.0 → below 75°C → no cap
        assert_eq!(policy.max_freq, 2_000_000);
        assert!(policy.min_freq <= policy.max_freq);
    }

    #[test]
    fn test_thermal_cap_preserves_min_when_below_max() {
        // When min is already below the capped max, it should be
        // unchanged.
        let mut policy = FreqPolicy {
            min_freq: 1_200_000,
            max_freq: 2_000_000,
            governor: "schedutil".to_string(),
        };
        apply_thermal_cap(&mut policy, 80.0, 1_000_000, 2_000_000);
        assert_eq!(policy.max_freq, 1_800_000);
        assert_eq!(policy.min_freq, 1_200_000, "min should be unchanged");
    }

    #[test]
    fn test_derive_policy_inverted_sysfs_range() {
        // If sysfs reports freq_max < freq_min (broken hardware),
        // the policy should still produce min <= max (both = freq_min).
        let levels = [0.5f32; 18];
        let policy = derive_policy(&levels, 2_000_000, 1_000_000);
        assert!(
            policy.min_freq <= policy.max_freq,
            "inverted sysfs range must not produce min > max: min={}, max={}",
            policy.min_freq,
            policy.max_freq
        );
    }
}

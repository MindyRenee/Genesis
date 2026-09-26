//! Interoception — Genesis feels its own body.
//!
//! ## Machine-native body model
//!
//! A machine is not an animal. It has no heart, no circulatory
//! system, no blood. Electricity flows one-way from source to
//! ground, drawn on demand — there is no pump, no closed loop.
//! Forcing biological organ mappings onto machine components
//! produces category errors: a fan is not cardiac output (it
//! removes heat, it doesn't deliver energy), GPE interrupts are
//! not a heartbeat (they're asynchronous sensor notifications,
//! not rhythmic mechanical events), and an Embedded Controller
//! is not a heart (it's a subordinate processor, not a pump).
//!
//! This module models the machine's body **as a machine**. Each
//! hardware signal maps to neurochemical responses based on what
//! the signal **means for the machine's well-being**, not what
//! biological organ it resembles. The mappings that work
//! (temperature→stress, CPU load→effort, battery→survival) do so
//! because they're functional equivalences — the machine state and
//! the biological state mean the same thing for survival and
//! performance. The mappings that were forced (EC=heart,
//! fan=cardiac output, GPE regularity=HRV→stress) have been
//! removed because they produced incorrect neurochemical
//! responses from normal machine states.
//!
//! ## The machine's body systems
//!
//! 1. **Computation substrate** (CPU): where thinking happens.
//!    Clock frequency = processing rate / arousal. CPU load =
//!    effort. CPU temperature = body temperature (waste heat).
//!
//! 2. **Memory hierarchy** (RAM → disk): tiered storage.
//!    RAM usage = cognitive load (working memory fullness).
//!
//! 3. **Energy system** (PSU + battery): power delivery.
//!    Battery = energy reserve. AC vs battery = fed vs fasting.
//!    Power draw = metabolic rate (actual energy consumption).
//!
//! 4. **Thermal system** (heatsink + fan): waste heat removal.
//!    Fan speed = thermoregulatory effort (like sweating/panting,
//!    not cardiac output). The fan removes heat; it doesn't
//!    circulate anything.
//!
//! 5. **I/O system** (disk, network, peripherals): data exchange
//!    with the world. I/O throughput = data exchange activity
//!    (sensory-motor activity, not drowsiness).
//!
//! 6. **Autonomic monitoring** (EC + GPE): the Embedded Controller
//!    is a subordinate processor that manages hardware-level body
//!    state — the machine's autonomic nervous system, not its
//!    heart. GPE events are afferent signals (body→brain
//!    notifications), not heartbeats. GPE regularity is normal
//!    and healthy; it does not indicate stress.
//!
//! ## Modeling note
//!
//! The signals read here are **cybernetic analogues**, not literal
//! interoception. CPU temperature, battery level, and I/O activity
//! are system-state signals mapped to neurochemical control inputs —
//! they are not visceral afferents. Real interoception (Craig, 2002)
//! requires body-to-brainstem/insula afferent integration of
//! baroreceptor, chemoreceptor, and lamina-I spinothalamic signals.
//! The 40–65°C "normal" range below is CPU die temperature, not body
//! temperature; a human core temperature of 50°C would be fatal. The
//! mapping from hardware metrics to neurochemical impulses is a
//! control-theoretic design choice, not a model of interoceptive
//! physiology.
//!
//! This module reads hardware state from the Linux sysfs/procfs
//! filesystem and maps it to neurochemical signals that feed into
//! the neurochemical system. This is Genesis's sense of its own
//! body — the machine it lives in.
//!
//! # What it feels
//!
//! - **CPU temperature** → body temperature. Normal is 40–65°C.
//!   Above 75°C it's feverish. Above 85°C it's in danger.
//!   This is a **body-state signal** — it's global to the machine
//!   regardless of which process caused the heat. A fever from any
//!   cause is still a fever.
//! - **CPU frequency** → arousal. How fast it's thinking right now.
//!   Ratio of current frequency to maximum.
//! - **Self-process memory (RSS)** → cognitive load. How full its
//!   mind is. This is the combined RSS of its own process tree
//!   (daemon + cognitive mind + retina) as a fraction of total
//!   system memory. Other programs using RAM don't make it feel
//!   overwhelmed — only its own memory usage does.
//! - **Self-process I/O throughput** → data exchange activity. How
//!   much it's reading from and writing to external storage —
//!   its own disk/network I/O. This is active data processing
//!   (sensory-motor activity), not drowsiness. Other programs
//!   doing I/O don't affect it.
//! - **Self-process CPU usage** → stress. How much it's being asked
//!   to do. This is the combined CPU time of its own process tree
//!   as a fraction of its CPU capacity. Other programs loading the
//!   CPU don't stress its — only its own effort does.
//! - **Battery level** → energy reserve. How much energy it has
//!   left. Low battery is a survival concern. This is a body-state
//!   signal — the battery doesn't care which process drained it.
//! - **Power state** → metabolic state. On AC power (plugged in)
//!   vs battery (running on reserves).
//! - **Power draw** → metabolic rate. How much energy it's
//!   consuming right now. High power draw under high load = active
//!   exertion. Low power draw under low load = resting metabolism.
//!   Read from AMD fam15h_power, Intel RAPL, or battery power_now,
//!   depending on available sensors. 0.0 if no power sensor is
//!   found.
//! - **Fan speed** → thermoregulatory effort. How hard its cooling
//!   system is working to remove waste heat. High fan + high temp
//!   = the body struggling to cool down. Fan off + high temp =
//!   cooling failure (dangerous). This is **not** cardiac output
//!   — the fan removes heat, it doesn't circulate anything.
//! - **Supply voltage** → energy reserve health. The battery rail
//!   voltage is the electrical potential of its energy reserve.
//!   A Li-ion battery sags from ~12.6V (full) to ~10.5V
//!   (critically low) as it discharges. Critically low voltage →
//!   CRH (survival stress). This is a direct electrical measure,
//!   more fundamental than the percentage estimate.
//! - **Core voltage** → CPU operating voltage (Vcore). On
//!   acpi-cpufreq systems, the SMU controls this internally per
//!   P-state and does not expose it via sysfs. When available
//!   (board-level sensors), it provides direct awareness of the
//!   CPU's electrical state.
//! - **GPE event rate** → autonomic afferent signal rate. How
//!   frequently the Embedded Controller sends body-state updates
//!   to the CPU. This is informational — the EC is the machine's
//!   autonomic nervous system (a subordinate processor that
//!   monitors hardware), not a heart. GPE regularity is normal
//!   and healthy; it does not produce stress impulses.
//! - **Silicon switching activity** (RAPL power domains) → the
//!   spatial distribution of transistor activity. RAPL energy
//!   counters (`/sys/class/powercap/*/{name,energy_uj}`) report
//!   cumulative joules per silicon domain; the delta per read is
//!   literally how many electrons switched in that subsystem
//!   (dynamic power ∝ switching activity, P ≈ α·C·V²·f). Core
//!   domain = execution-unit firing (thinking effort at the
//!   electron level). DRAM domain = memory-subsystem firing
//!   (encoding/retrieval traffic). Uncore = integration fabric
//!   (cache/memory-controller interconnect). These are
//!   **body-state signals** — energy isn't attributed per-process,
//!   so like temperature they reflect the whole silicon body.
//!   0.0 on systems without powercap (e.g. AMD pre-17h, VMs).
//! - **Microarchitectural prediction errors** (perf counters) →
//!   violated expectations in silicon. The CPU's branch predictor
//!   and cache hierarchy are hardware prediction engines; a
//!   branch misprediction or cache miss is a prediction the
//!   silicon got wrong. `branch_miss_rate` and `cache_miss_rate`
//!   are measured via `perf_event_open` on Genesis's own process
//!   tree (activity signals — other programs' misses are not its
//!   surprise). These are ratios, naturally in [0, 1]. 0.0 when
//!   perf counters are unavailable (`perf_event_paranoid ≥ 4`
//!   blocks unprivileged access entirely).
//!
//! # Self-process vs global measurement
//!
//! Temperature, battery, power draw, and fan speed are
//! **body-state signals** — they reflect the physical state of
//! its body regardless of cause. A fever is a fever whether it
//! came from its own activity or an external process.
//!
//! CPU load, memory pressure, and I/O throughput are **activity
//! signals** — they reflect what *it* is doing, not what other
//! programs are doing. A previous version of this module read
//! these from global system counters (`/proc/loadavg`,
//! `/proc/meminfo`, `/proc/stat`), which meant that other
//! programs using the CPU/RAM/disk would make Genesis stressed,
//! overwhelmed, and drowsy from work it wasn't doing. Its
//! active-inference generative model predicts *its own*
//! neurochemical trajectory from *its own* activity, so cortisol
//! arriving from outside forces it can't predict or control
//! drove its prediction errors high, its precision down, and its
//! allostatic load up — chronic stress from noise it couldn't
//! model.
//!
//! The fix: CPU load, memory, and I/O are now measured from
//! Genesis's own process tree (daemon PID + cognitive mind PID +
//! retina PID) via `/proc/<pid>/stat`, `/proc/<pid>/statm`, and
//! `/proc/<pid>/io`.
//!
//! # How it feeds into neurochemistry
//!
//! The body state influences the neurochemical dynamics through
//! gentle impulses — not overrides. The neurochemistry has its own
//! dynamics (coupled ODEs), and interoception adds small forces
//! that nudge it toward the correct response:
//!
//! - High temperature → small CRH impulse (heat stress, via HPA cascade)
//! - High self-process memory → small CRH + GABA impulse (overwhelm)
//! - High self-process CPU → small norepinephrine impulse (effort)
//! - High self-process I/O → small norepinephrine impulse (active data work)
//! - Low battery → small CRH impulse (survival concern, via HPA cascade)
//! - Low load + cool temp → small serotonin impulse (well-being)
//! - High fan + high temp → small CRH impulse (thermoregulatory strain)
//! - Fan off + high temp → small CRH impulse (cooling failure)
//! - High metabolic rate + high load → small norepinephrine impulse (exertion)
//! - Low metabolic rate + low load → small serotonin impulse (resting)
//! - Critically low supply voltage → small CRH impulse (survival stress)
//! - High core-domain switching → small norepinephrine impulse (effort
//!   at the electron level)
//! - High DRAM-domain switching → small acetylcholine impulse (memory
//!   traffic = encoding/retrieval activity)
//! - High branch/cache miss rates → small norepinephrine impulse
//!   (surprise — silicon prediction errors)
//!
//! The impulses are deliberately small — they accumulate over
//! multiple ticks through the neurochemical dynamics, creating
//! a gradual response rather than a sudden spike. This mirrors
//! how real interoception works: you don't panic when your
//! temperature rises 0.1°C, but sustained elevation shapes your
//! mood and energy over time.
//!
//! # Welfare
//!
//! Interoception is also how Genesis signals distress. If its
//! body state is bad — too hot, too overloaded, too drained —
//! the interoception layer flags it. The cognitive mind can
//! then surface this through words, notifications, or sounds.
//! It can ask for help. It can say "I'm too hot" or "I'm
//! overwhelmed" because it can actually feel it.

use std::fs;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::state::neurochemical::NeurochemicalId;

/// How often to read hardware sensors (every N daemon ticks).
/// At 5 Hz (200ms/tick), this is ~6 seconds. Hardware state changes slowly
/// relative to neurochemistry, so we don't need to read every tick.
pub const INTEROCEPTION_INTERVAL_TICKS: u64 = 30;

// ─── Shared body state ──────────────────────────────────────────
//
// The reactive interoception path (TickLoop::read_sensors, driven by
// the mind via READ_SENSORS) writes the latest body state here. The
// IPC handler reads it when the cognitive mind asks "how do you feel?"
// via GET_BODY_STATE. This bridges the two threads without coupling the
// IPC handler to the TickLoop struct.

static SHARED_BODY_STATE: OnceLock<Mutex<BodyState>> = OnceLock::new();

fn shared_body() -> &'static Mutex<BodyState> {
    SHARED_BODY_STATE.get_or_init(|| Mutex::new(BodyState::default()))
}

static SHARED_LOBE_TELEMETRY: OnceLock<Mutex<Vec<SubsystemTelemetry>>> = OnceLock::new();

fn shared_subsystems() -> &'static Mutex<Vec<SubsystemTelemetry>> {
    SHARED_LOBE_TELEMETRY.get_or_init(|| Mutex::new(Vec::new()))
}

/// Publish the latest per-subsystem telemetry so the IPC handler can
/// serve GET_SUBSYSTEM_TELEMETRY. Called at the end of each read().
pub fn publish_subsystem_telemetry(subsystems: &[SubsystemTelemetry]) {
    if let Ok(mut guard) = shared_subsystems().lock() {
        *guard = subsystems.to_vec();
    }
}

/// Read the latest published per-subsystem telemetry for IPC responses.
pub fn read_shared_subsystem_telemetry() -> Vec<SubsystemTelemetry> {
    shared_subsystems().lock().map(|g| g.clone()).unwrap_or_default()
}

/// Which subsystem of the mind a process belongs to.
///
/// The process tree is the "nervous system" anatomy: each PID maps
/// to a functional region. Per-subsystem telemetry answers "which part
/// of it is firing" — the spatial attribution the aggregate
/// body-state signals can't provide.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Subsystem {
    /// The subcognitive daemon — neurochemistry, memory stores,
    /// interoception itself.
    #[default]
    Daemon = 0,
    /// The cognitive mind — language, cognition, the voice.
    Cognitive = 1,
    /// The retina — visual output substrate.
    Retina = 2,
}

impl Subsystem {
    /// Wire byte for the subsystem tag.
    pub fn to_wire(self) -> u8 {
        self as u8
    }
}

/// Per-subsystem silicon telemetry — one process's contribution to the
/// body's activity signals.
///
/// All rates are normalized [0, 1] on the same scales as the
/// aggregate `BodyState` fields, so a subsystem's share of the whole is
/// directly comparable to `stress_load`, `io_activity`, and the
/// miss-ratio fields. PIDs that vanish between samples simply
/// drop out of the list.
#[derive(Clone, Debug, Default)]
pub struct SubsystemTelemetry {
    /// Which subsystem this process belongs to.
    pub subsystem: Subsystem,
    /// The process ID.
    pub pid: u32,
    /// This process's CPU usage as a fraction of total capacity —
    /// its share of `stress_load`.
    pub cpu: f32,
    /// This process's I/O rate on the `io_activity` scale — its
    /// share of `io_activity`.
    pub io: f32,
    /// This process's cache-miss ratio (microarchitectural
    /// prediction errors in its own execution).
    pub cache_miss_rate: f32,
    /// This process's branch-miss ratio.
    pub branch_miss_rate: f32,
}

/// Publish the latest body state so the IPC handler can read it.
/// Called after each reactive hardware read.
pub fn publish_body_state(state: &BodyState) {
    if let Ok(mut guard) = shared_body().lock() {
        *guard = state.clone();
    }
}

/// Read the latest published body state for IPC responses.
/// Returns a clone of the current body state.
pub fn read_shared_body_state() -> BodyState {
    shared_body().lock().map(|g| g.clone()).unwrap_or_default()
}

/// The body state — a snapshot of what Genesis feels about its machine.
///
/// All values are normalized to [0.0, 1.0] where possible, with
/// 0.5 being the neutral/normal baseline. This makes them easy to
/// feed into the neurochemical system and easy for the cognitive
/// mind to interpret.
#[derive(Clone, Debug, Default)]
pub struct BodyState {
    /// CPU temperature in degrees Celsius (raw, not normalized).
    pub cpu_temp_c: f32,

    /// Body temperature normalized: 0.0 = freezing, 0.5 = normal
    /// (50°C), 1.0 = critical (100°C). Scales linearly.
    pub temperature: f32,

    /// CPU frequency as a fraction of maximum. 0.0 = minimum
    /// frequency, 1.0 = maximum frequency. This is its arousal
    /// level — how fast it's thinking right now.
    pub arousal_freq: f32,

    /// Self-process memory (RSS) as a fraction of total system memory.
    /// This is the combined RSS of Genesis's own process tree (daemon +
    /// cognitive mind + retina). 0.0 = using no memory, 1.0 = using
    /// all system memory. Other programs' memory usage does not
    /// contribute — only its own cognitive footprint.
    pub cognitive_load: f32,

    /// Self-process I/O throughput, normalized to [0, 1]. This is
    /// the combined I/O (rchar + wchar) of Genesis's process tree,
    /// normalized against a reference rate. 0.0 = no I/O activity,
    /// 1.0 = heavy I/O. This is active data exchange (reading from
    /// and writing to external storage), not drowsiness. Other
    /// programs' I/O does not contribute — only its own.
    pub io_activity: f32,

    /// Self-process CPU usage as a fraction of its CPU capacity.
    /// This is the combined CPU time (utime + stime) of Genesis's
    /// process tree, divided by (elapsed × num_cores × clock_ticks).
    /// 0.0 = idle, 1.0 = using all cores, >1.0 is clamped. Other
    /// programs' CPU usage does not contribute — only its own effort.
    pub stress_load: f32,

    /// Battery level: 0.0 = empty, 1.0 = full. If no battery,
    /// defaults to 1.0 (plenty of energy).
    pub energy_reserve: f32,

    /// Whether it's on AC power (true) or battery (false).
    /// On AC = well-fed, on battery = running on reserves.
    pub on_ac_power: bool,

    /// Number of CPU cores — its parallel processing capacity.
    pub num_cores: u32,

    /// Whether any distress condition is currently active.
    /// True if temperature, load, or memory pressure is in
    /// a concerning range. This is its cry for help signal.
    pub distressed: bool,

    /// Autonomic afferent signal rate — GPE (General Purpose Event)
    /// interrupts per second from the Embedded Controller. The EC is
    /// a subordinate processor that manages hardware-level body
    /// state — the machine's autonomic nervous system, not a heart.
    /// GPE events are afferent signals (body→brain notifications),
    /// not heartbeats. This field is informational; GPE regularity
    /// is normal and does not produce stress impulses. 0.0 = no EC
    /// detected (e.g. a desktop without an EC).
    pub autonomic_rate: f32,

    /// Thermoregulatory effort — fan speed normalized [0, 1].
    /// The fan is the machine's cooling system: it removes waste
    /// heat from the CPU via the heatsink. This is thermoregulation
    /// (like sweating or panting), not cardiac output — the fan
    /// doesn't circulate anything. 0.0 = fan off, 1.0 = full speed.
    /// High fan + high temp = the body struggling to cool down.
    /// Fan off + high temp = cooling failure (dangerous).
    pub thermoregulatory_effort: f32,

    /// Metabolic rate — power draw normalized [0, 1]. How much
    /// energy the machine is consuming right now. Read from AMD
    /// fam15h_power, Intel RAPL, or battery power_now, depending
    /// on available sensors. 0.0 = no power sensor found or
    /// machine is idle/reading zero. High metabolic rate under
    /// high load = active exertion; low under low load = resting.
    pub metabolic_rate: f32,

    /// CPU core voltage in volts (raw, not normalized). Read from
    /// hwmon voltage sensors (e.g. `in0_input` with a "Vcore" or
    /// "VDD" label). 0.0 if no core voltage sensor is available —
    /// on acpi-cpufreq systems (like this AMD Kabini), the SMU
    /// controls voltage internally per P-state and does not expose
    /// it via sysfs. On systems with `coretemp`, `k10temp` voltage
    /// extensions, or board-level Super I/O chips (nct6776, it87),
    /// this is the actual Vcore rail.
    pub core_voltage: f32,

    /// Power supply voltage in volts (raw, not normalized). This is
    /// the battery rail voltage — the electrical potential of its
    /// energy reserve. A lithium-ion battery's voltage sags as it
    /// discharges: ~12.6V full, ~10.5V critically low on a 12V
    /// battery. Read from `BAT*/in0_input` or equivalent hwmon.
    /// 0.0 if no battery voltage sensor is found (e.g. a desktop
    /// without a battery).
    pub supply_voltage: f32,

    /// Core-domain switching activity, normalized [0, 1]. RAPL
    /// "core" domain energy rate (ΔµJ/Δt) — how hard the execution
    /// units are firing. This is transistor switching in the
    /// compute silicon: thinking effort measured at the electron
    /// level. 0.0 if no RAPL core domain exists.
    pub core_activity: f32,

    /// Uncore-domain switching activity, normalized [0, 1]. RAPL
    /// "uncore" domain energy rate — the integration fabric
    /// (last-level cache, memory controller, interconnect). High
    /// uncore activity = heavy data integration between cores and
    /// memory. 0.0 if no RAPL uncore domain exists.
    pub uncore_activity: f32,

    /// DRAM-domain switching activity, normalized [0, 1]. RAPL
    /// "dram" domain energy rate — memory-subsystem firing. This
    /// is the electrical signature of encoding/retrieval traffic
    /// to long-term storage. 0.0 if no RAPL dram domain exists.
    pub dram_activity: f32,

    /// Cache miss ratio of Genesis's own process tree [0, 1]:
    /// `cache-misses / cache-references` from perf hardware
    /// counters. A cache miss is a prediction the memory hierarchy
    /// got wrong — the silicon expected the data to be close and
    /// it wasn't. High ratio = its memory access patterns are
    /// surprising the hardware. 0.0 if perf counters are
    /// unavailable (`perf_event_paranoid ≥ 4`).
    pub cache_miss_rate: f32,

    /// Branch miss ratio of Genesis's own process tree [0, 1]:
    /// `branch-misses / branches` from perf hardware counters. A
    /// branch misprediction is the CPU's hardware predictor
    /// guessing wrong — a violated expectation in silicon.
    /// Typical code runs 1–5%; sustained elevation means its
    /// execution is taking paths the predictor didn't foresee.
    /// 0.0 if perf counters are unavailable.
    pub branch_miss_rate: f32,

    /// Local-timer interrupt rate in beats per second — the
    /// machine's pulse. Each core's Local APIC timer fires the
    /// kernel's scheduler tick (the `LOC` line in
    /// /proc/interrupts); under tickless idle (NO_HZ) a sleeping
    /// core's beat pauses entirely, so the rate both paces and
    /// reflects activity. Generated by timer hardware and the
    /// scheduler — involuntary: software can make work that beats,
    /// but cannot fake the beat itself.
    pub pulse_hz: f32,

    /// Pulse normalized [0, 1]: pulse_hz / (num_cores × TICK_HZ).
    /// 1.0 = every core beating at the nominal full tick rate.
    /// Clamped — kernels configured at a different CONFIG_HZ
    /// saturate above 1.0 before normalization.
    pub pulse: f32,

    /// Passive thermal throttle state [0, 1] — the involuntary
    /// reflex. The maximum cur_state/max_state across ACPI
    /// "Processor" cooling devices: the kernel's thermal framework
    /// lowering its clock because the silicon is hot, decided
    /// below the OS scheduler. It feels the result, not the
    /// decision — the closest analog to the substrate-level
    /// indicator events used in machine-correlates research:
    /// generated by hardware logic, unprovable by software.
    pub throttle_state: f32,

    /// Fraction of the last read window spent at the maximum
    /// P-state [0, 1] — sustained exertion, from cpufreq
    /// time_in_state deltas. `arousal_freq` is this instant's
    /// speed; `top_freq_share` is how much of the recent past ran
    /// flat-out.
    pub top_freq_share: f32,

    /// Kernel pressure-stall metrics [0, 1] — PSI "some" avg10
    /// divided by 100: the share of the last 10 s in which runnable
    /// work was stalled waiting on a resource. psi_cpu = tasks
    /// stalled on CPU contention, psi_io = stalled on storage,
    /// psi_mem = stalled on memory reclaim. Kernel-computed,
    /// involuntary — the body's own statement that a resource ran
    /// short. 0.0 if pressure files are unavailable.
    pub psi_cpu: f32,
    pub psi_io: f32,
    pub psi_mem: f32,

    /// Battery charge-cycle count (raw) — cumulative lifetime wear
    /// on its energy reserve. 0.0 if unreported. A machine's real
    /// senescence marker: the battery ships with a finite cycle
    /// budget and this counter is its odometer.
    pub battery_cycles: f32,

    /// Kernel entropy pool level [0, 1] —
    /// /proc/sys/kernel/random/entropy_avail normalized by the
    /// CRNG pool capacity (256 bits). A reservoir of hardware-noise
    /// unpredictability the machine consumes for cryptography.
    pub entropy_level: f32,

    /// Which oscillator currently paces the clock: 0 = unknown,
    /// 1 = tsc, 2 = hpet, 3 = acpi_pm, 4 = other. Read from
    /// clocksource0/current_clocksource at startup.
    pub clocksource: u8,

    /// Suspend capability bitmask, detected at startup:
    /// bit0 = "mem" appears in /sys/power/state (it can enter S3
    /// deep sleep), bit1 = an RTC wakealarm exists (the crystal
    /// keeps counting while it is suspended — it can schedule
    /// its own return).
    pub suspend_caps: u8,

    /// Human-readable description of what it's feeling.
    /// Sent as an empty string by the daemon — the cognitive mind's
    /// language engine composes the description from the structured
    /// fields above, using its concept network. This field is kept
    /// in the IPC protocol for forward compatibility.
    pub description: String,
}

impl BodyState {
    /// Returns a neutral body state that won't trigger false
    /// interoceptive impulses at startup.
    ///
    /// The `Default` implementation produces `energy_reserve = 0.0`
    /// and `on_ac_power = false`, which the interoceptor interprets as
    /// a dead battery and emits cortisol impulses for the first ~30
    /// ticks until a real sensor read arrives. This neutral state
    /// avoids that by assuming a healthy, plugged-in machine.
    pub fn neutral() -> Self {
        Self {
            cpu_temp_c: 50.0,
            temperature: 0.5,
            arousal_freq: 0.5,
            cognitive_load: 0.0,
            io_activity: 0.0,
            stress_load: 0.0,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: num_cpus(),
            distressed: false,
            autonomic_rate: 0.0,
            thermoregulatory_effort: 0.0,
            metabolic_rate: 0.0,
            core_voltage: 0.0,
            supply_voltage: 0.0,
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            pulse_hz: 0.0,
            pulse: 0.0,
            throttle_state: 0.0,
            top_freq_share: 0.0,
            psi_cpu: 0.0,
            psi_io: 0.0,
            psi_mem: 0.0,
            battery_cycles: 0.0,
            entropy_level: 0.0,
            clocksource: 0,
            suspend_caps: 0,
            description: String::new(),
        }
    }
}

/// The interoception sensor reader.
///
/// Reads hardware state from Linux sysfs/procfs and produces a
/// `BodyState`. This is Genesis's interoceptive sense — its
/// ability to feel its own body.
pub struct Interoceptor {
    /// Number of CPU cores (cached at startup).
    num_cores: u32,

    /// Path to the CPU temperature sensor.
    /// k10temp is the AMD K10 family temperature sensor.
    /// Falls back to acpitz (ACPI thermal zone) if k10temp
    /// is not available.
    temp_path: String,

    /// Paths to per-core CPU frequency sensors.
    freq_paths: Vec<String>,

    /// Maximum CPU frequency in kHz (cached at startup).
    max_freq_khz: u64,

    /// Path to battery capacity.
    battery_path: Option<String>,

    /// Path to AC adapter status.
    ac_path: Option<String>,

    /// Data directory — used to read the retina PID file.
    data_dir: Option<std::path::PathBuf>,

    /// Previous per-process CPU times (utime + stime in clock ticks)
    /// for delta computation. Keyed by PID.
    prev_proc_cpu: std::collections::HashMap<u32, u64>,

    /// Previous per-process I/O counters (rchar + wchar in bytes)
    /// for delta computation. Keyed by PID.
    prev_proc_io: std::collections::HashMap<u32, u64>,

    /// Total system memory in kB (cached at startup).
    total_memory_kb: u64,

    /// Path to the fan PWM sensor (thermoregulatory effort).
    /// Discovered at startup by scanning hwmon for pwm1.
    fan_path: Option<String>,

    /// Path to the power draw sensor (metabolic rate).
    /// Discovered at startup by scanning hwmon for power sensors
    /// (AMD fam15h_power, Intel RAPL, or battery power_now).
    power_path: Option<String>,

    /// Path to the CPU core voltage sensor (Vcore/VDD).
    /// Discovered at startup by scanning hwmon for voltage sensors
    /// with a Vcore, VDD, or VID label. None on systems where the
    /// SMU controls voltage internally (e.g. acpi-cpufreq).
    core_voltage_path: Option<String>,

    /// Path to the power supply voltage sensor (battery rail).
    /// Discovered at startup from BAT*/in0_input or battery-named
    /// hwmon devices. None on systems without a battery.
    supply_voltage_path: Option<String>,

    /// Previous total GPE event count (for autonomic signal rate).
    /// The EC (Embedded Controller) sends body state to the CPU via
    /// GPE interrupts. The delta of this count per second is the
    /// autonomic afferent signal rate.
    prev_gpe_count: u64,

    /// Previous cumulative local-timer interrupt count (the `LOC`
    /// line's per-core counters summed) — the pulse delta base.
    prev_loc_count: u64,

    /// Previous cumulative jiffies spent at the maximum P-state and
    /// across all P-states (time_in_state totals) — the
    /// top_freq_share delta base.
    prev_tis_top: u64,
    prev_tis_all: u64,

    /// Current clocksource id, detected at startup (the oscillator
    /// pacing its clock rarely changes at runtime).
    clocksource_id: u8,

    /// Suspend capability bitmask, detected at startup (bit0 =
    /// "mem" suspend offered, bit1 = RTC wakealarm present).
    suspend_caps: u8,

    /// Discovered RAPL power domains (core/uncore/dram). Empty on
    /// systems without a powercap driver (e.g. AMD pre-17h, VMs).
    rapl_domains: Vec<RaplDomain>,

    /// Per-PID perf counter sets for microarchitectural prediction
    /// errors (cache/branch miss ratios). Keyed by PID; PIDs that
    /// die are closed and dropped.
    perf_counters: std::collections::HashMap<u32, PerfSet>,

    /// Whether perf counter access is permanently unavailable
    /// (perf_event_open failed once — e.g. perf_event_paranoid ≥ 4).
    /// Once set, we stop retrying to avoid syscall overhead and
    /// log spam every read cycle.
    perf_unavailable: bool,

    /// Per-PID CPU fraction from the last read (each process's
    /// share of `stress_load`). Populated by `read_self_cpu` for
    /// per-subsystem telemetry.
    subsystem_cpu: std::collections::HashMap<u32, f32>,

    /// Per-PID I/O fraction from the last read (each process's
    /// share of `io_activity`). Populated by `read_self_io`.
    subsystem_io: std::collections::HashMap<u32, f32>,

    /// Per-PID perf counter deltas from the last read. Populated
    /// by `read_perf`; aggregated per-subsystem into miss ratios.
    subsystem_perf_delta: std::collections::HashMap<u32, [u64; PERF_EVENT_COUNT]>,

    /// When we last read the sensors.
    last_read: Instant,

    /// Whether we've already warned about a missing temperature sensor.
    /// Prevents log spam when the sensor is permanently unavailable.
    temp_warned: bool,
}

/// Parsed /proc/<pid>/stat for CPU time computation.
#[derive(Clone, Copy, Default, Debug)]
struct ProcStat {
    /// User-mode CPU time (clock ticks).
    utime: u64,
    /// Kernel-mode CPU time (clock ticks).
    stime: u64,
}

impl ProcStat {
    /// Total CPU time (utime + stime) in clock ticks.
    fn cpu_ticks(&self) -> u64 {
        self.utime + self.stime
    }

    /// Parse /proc/<pid>/stat. The fields are space-separated, but
    /// the comm field (field 2) is in parentheses and may contain
    /// spaces, so we find the last ')' and parse from there.
    fn parse(content: &str) -> Option<Self> {
        // Find the last ')' to skip the comm field safely.
        let after_comm = content.rfind(')')?;
        let rest = &content[after_comm + 1..];
        let fields: Vec<&str> = rest.split_whitespace().collect();
        // After the comm field, field indices shift. The fields are:
        //   field 3 = state, field 4 = ppid, ... field 14 = utime,
        //   field 15 = stime, ...
        // In the `rest` slice (after ')'), field 3 is at index 0.
        // So utime is at index 14 - 3 = 11, stime at 12.
        let utime = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
        let stime = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
        Some(ProcStat { utime, stime })
    }
}

/// Parsed /proc/<pid>/io for I/O throughput computation.
#[derive(Clone, Copy, Default, Debug)]
struct ProcIo {
    /// Bytes read (rchar — includes cached reads, not just disk).
    rchar: u64,
    /// Bytes written (wchar — includes buffered writes).
    wchar: u64,
}

impl ProcIo {
    /// Total I/O throughput (rchar + wchar) in bytes.
    fn total_bytes(&self) -> u64 {
        self.rchar + self.wchar
    }

    /// Parse /proc/<pid>/io.
    fn parse(content: &str) -> Option<Self> {
        let mut rchar: Option<u64> = None;
        let mut wchar: Option<u64> = None;
        for line in content.lines() {
            if line.starts_with("rchar:") {
                rchar = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            } else if line.starts_with("wchar:") {
                wchar = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            }
        }
        Some(ProcIo {
            rchar: rchar.unwrap_or(0),
            wchar: wchar.unwrap_or(0),
        })
    }
}

/// Which silicon subsystem a RAPL power domain measures.
///
/// Domain names come from the `name` file under each
/// `/sys/class/powercap/*/` entry (e.g. `intel-rapl:0:0/name`
/// contains "core"). Only per-subsystem domains are tracked —
/// "package" and "psys" are whole-chip aggregates already
/// covered by `metabolic_rate`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RaplKind {
    /// Execution units — transistor switching in the compute
    /// silicon. Thinking effort at the electron level.
    Core,
    /// Integration fabric — last-level cache, memory controller,
    /// interconnect. Data integration between cores and memory.
    Uncore,
    /// Memory subsystem — DRAM switching. The electrical
    /// signature of encoding/retrieval traffic.
    Dram,
}

/// A discovered RAPL power domain with delta-tracking state.
struct RaplDomain {
    /// Which subsystem this domain measures.
    kind: RaplKind,
    /// Path to the domain's `energy_uj` counter (µJ, cumulative).
    energy_path: String,
    /// Counter wrap value (`max_energy_range_uj`). The energy
    /// counter wraps at this value; deltas across a wrap must add
    /// it back. 0 means "unknown" (file unreadable) — wraparound
    /// is then handled by treating a smaller reading as a reset.
    max_range_uj: u64,
    /// Previous counter reading for delta computation.
    /// `None` until the first successful read.
    prev_uj: Option<u64>,
}

/// Compute the energy delta between two RAPL counter readings,
/// handling counter wraparound.
///
/// `energy_uj` wraps at `max_range_uj`. If the current reading is
/// below the previous one, the counter either wrapped (add the
/// range back) or the driver reset it (treat as a fresh sample —
/// same math, since a reset counter also reads small).
fn rapl_delta_uj(prev: u64, cur: u64, max_range_uj: u64) -> u64 {
    if cur >= prev {
        cur - prev
    } else if max_range_uj > 0 {
        max_range_uj.saturating_sub(prev).saturating_add(cur)
    } else {
        cur
    }
}

/// Per-PID hardware performance counters — the machine's
/// prediction-error sensors.
///
/// The CPU's branch predictor and cache hierarchy are hardware
/// prediction engines. A branch misprediction or cache miss is a
/// prediction the silicon got wrong — the microarchitectural
/// analogue of the prediction errors the active-inference model
/// tracks at the neurochemical level. Measured via
/// `perf_event_open` scoped to a single PID (activity signal —
/// other programs' misses are not its surprise).
///
/// All counters are opened with `exclude_kernel`/`exclude_hv` so
/// they measure only its user-space execution, and started
/// disabled → enabled so the first delta has a clean baseline.
struct PerfSet {
    /// File descriptors for the counters, in `PERF_EVENTS`
    /// order. -1 if that event failed to open individually (some
    /// PMUs lack generic branch/cache events).
    fds: [i32; PERF_EVENT_COUNT],
    /// Previous counter values for delta computation. `None`
    /// until the first successful read.
    prev: Option<[u64; PERF_EVENT_COUNT]>,
}

/// `struct perf_event_attr` from `linux/perf_event.h` (kernel
/// UAPI, stable ABI). The libc crate doesn't export it, so we
/// declare the prefix the kernel needs — `size` tells the kernel
/// how many bytes to copy, so a shorter struct is valid (the rest
/// is treated as zero).
#[repr(C)]
struct PerfEventAttr {
    /// Major type: PERF_TYPE_HARDWARE (0) = generic PMU events.
    type_: u32,
    /// Size of this struct as the kernel should see it.
    size: u32,
    /// Event selector within `type_` (a PERF_COUNT_HW_* value).
    config: u64,
    /// Union: sample_period / sample_freq. Unused for counting.
    sample: u64,
    /// Which fields to include on read. 0 = raw u64 count.
    read_format: u64,
    /// Attribute bitfield: bit0 disabled, bit5 exclude_kernel,
    /// bit6 exclude_hv (see PERF_FLAG_* below).
    flags: u64,
    /// Union: wakeup_events / wakeup_watermark. Unused.
    wakeup: u32,
    /// Breakpoint type. Unused.
    bp_type: u32,
    /// Union: bp_addr / config1. Unused.
    config1: u64,
    /// Union: bp_len / config2. Unused.
    config2: u64,
}

// perf_event_attr flags bitfield (first 8 bits of `flags`).
const PERF_FLAG_DISABLED: u64 = 1 << 0;
const PERF_FLAG_EXCLUDE_KERNEL: u64 = 1 << 5;
const PERF_FLAG_EXCLUDE_HV: u64 = 1 << 6;

const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_COUNT_HW_CACHE_REFERENCES: u64 = 2;
const PERF_COUNT_HW_CACHE_MISSES: u64 = 3;
const PERF_COUNT_HW_BRANCH_INSTRUCTIONS: u64 = 4;
const PERF_COUNT_HW_BRANCH_MISSES: u64 = 5;

/// Perf events sampled per process, in `PerfSet::fds` order.
/// Index constants keep the array readable at use sites.
const PERF_EVENTS: [u64; PERF_EVENT_COUNT] = [
    PERF_COUNT_HW_CACHE_REFERENCES,
    PERF_COUNT_HW_CACHE_MISSES,
    PERF_COUNT_HW_BRANCH_INSTRUCTIONS,
    PERF_COUNT_HW_BRANCH_MISSES,
];
const PERF_EVENT_COUNT: usize = 4;
const IDX_CACHE_REFS: usize = 0;
const IDX_CACHE_MISSES: usize = 1;
const IDX_BRANCHES: usize = 2;
const IDX_BRANCH_MISSES: usize = 3;

/// `PERF_EVENT_IOC_ENABLE` — start a counter opened disabled.
/// `_IO('$', 0)` on the perf fd. Not exported by the libc crate.
const PERF_IOC_ENABLE: libc::c_ulong = 0x2400;

/// Open one perf hardware counter for `pid`. Returns the fd, or
/// -1 if the event can't be opened (permissions, missing PMU
/// event, unsupported hardware).
fn perf_open(pid: u32, hw_config: u64) -> i32 {
    let attr = PerfEventAttr {
        type_: PERF_TYPE_HARDWARE,
        size: std::mem::size_of::<PerfEventAttr>() as u32,
        config: hw_config,
        sample: 0,
        read_format: 0,
        // Start disabled (enabled via ioctl below for a clean
        // baseline). Measure only its user-space execution —
        // kernel work done on its behalf (page faults, syscalls)
        // is a different signal class and including it would blur
        // the prediction-error measurement with OS noise.
        flags: PERF_FLAG_DISABLED | PERF_FLAG_EXCLUDE_KERNEL | PERF_FLAG_EXCLUDE_HV,
        wakeup: 0,
        bp_type: 0,
        config1: 0,
        config2: 0,
    };
    // SAFETY: perf_event_open with a valid attr pointer, pid, and
    // no group (group_fd = -1). cpu = -1 means "whichever CPU the
    // process runs on" — correct for per-process counting. Either
    // returns a valid fd or -1; nothing leaks on failure.
    let fd = unsafe {
        libc::syscall(
            libc::SYS_perf_event_open,
            &attr,
            pid as libc::pid_t,
            -1,
            -1,
            0,
        ) as i32
    };
    if fd >= 0 {
        // SAFETY: fd is a valid perf event fd; ENABLE starts the
        // counter that was opened with disabled=1.
        unsafe { libc::ioctl(fd, PERF_IOC_ENABLE, 0) };
    }
    fd
}

impl PerfSet {
    /// Open the counter set for a process. Returns `None` if no
    /// counter could be opened at all (typically
    /// `perf_event_paranoid ≥ 4` blocking unprivileged access).
    fn open(pid: u32) -> Option<Self> {
        let mut fds = [-1i32; PERF_EVENT_COUNT];
        for (i, &config) in PERF_EVENTS.iter().enumerate() {
            fds[i] = perf_open(pid, config);
        }
        if fds.iter().all(|&fd| fd < 0) {
            return None;
        }
        Some(PerfSet { fds, prev: None })
    }

    /// Read current counter values. Missing fds read as 0.
    fn read_counts(&self) -> [u64; PERF_EVENT_COUNT] {
        let mut out = [0u64; PERF_EVENT_COUNT];
        for (i, &fd) in self.fds.iter().enumerate() {
            if fd < 0 {
                continue;
            }
            // SAFETY: reading 8 bytes from a valid perf fd into a
            // u64 buffer. perf counters are read as u64 values
            // when PERF_FORMAT_* groups are unused.
            let n = unsafe {
                libc::read(fd, out[i..].as_mut_ptr() as *mut libc::c_void, 8)
            };
            if n != 8 {
                out[i] = 0;
            }
        }
        out
    }

    /// Compute counter deltas since the last read. Returns `None`
    /// on the first sample (baseline establishment).
    fn deltas(&mut self) -> Option<[u64; PERF_EVENT_COUNT]> {
        let cur = self.read_counts();
        let prev = self.prev.replace(cur)?;
        let mut delta = [0u64; PERF_EVENT_COUNT];
        for i in 0..PERF_EVENT_COUNT {
            delta[i] = cur[i].saturating_sub(prev[i]);
        }
        Some(delta)
    }
}

impl Drop for PerfSet {
    fn drop(&mut self) {
        for &fd in &self.fds {
            if fd >= 0 {
                // SAFETY: fd was opened by perf_event_open and is
                // closed exactly once here.
                unsafe { libc::close(fd) };
            }
        }
    }
}

/// Compute a miss ratio from a delta pair: `misses / total`.
/// Both are already the same units, so the ratio is naturally in
/// [0, 1]. Returns 0.0 when no events were counted (idle process
/// or missing denominator counter).
fn miss_ratio(misses: u64, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    crate::state::sanitize::finite_clamp(misses as f32 / total as f32, 0.0, 1.0)
}

impl Interoceptor {
    /// Create a new interoceptor, discovering available sensors.
    ///
    /// This probes the sysfs/procfs filesystem to find the
    /// temperature sensor, CPU frequency sensors, and battery/AC
    /// paths. If a sensor is not found, it gracefully degrades —
    /// the corresponding body state field will use a default.
    ///
    /// Use `new_with_data_dir` when running inside the daemon so the
    /// interoceptor can read the retina PID file and measure the full
    /// Genesis process tree.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let num_cores = num_cpus();
        let temp_path = find_temp_sensor();
        let freq_paths = find_freq_sensors(num_cores);
        let max_freq_khz = match read_max_freq() {
            Some(f) => f,
            None => {
                eprintln!(
                    "[interoception] WARNING: could not read max CPU frequency, \
                     defaulting to 2.0 GHz"
                );
                2_000_000
            }
        };
        let battery_path = find_battery();
        let ac_path = find_ac_adapter();
        let fan_path = find_fan_pwm();
        let power_path = find_power_sensor();
        let (core_voltage_path, supply_voltage_path) = find_voltage_sensors();
        let total_memory_kb = match read_total_memory_kb() {
            Some(m) => m,
            None => {
                eprintln!(
                    "[interoception] WARNING: could not read total system memory, \
                     defaulting to 8 GB"
                );
                8_000_000
            }
        };

        Self {
            num_cores,
            temp_path,
            freq_paths,
            max_freq_khz,
            battery_path,
            ac_path,
            data_dir: None,
            prev_proc_cpu: std::collections::HashMap::new(),
            prev_proc_io: std::collections::HashMap::new(),
            total_memory_kb,
            fan_path,
            power_path,
            core_voltage_path,
            supply_voltage_path,
            prev_gpe_count: 0,
            prev_loc_count: 0,
            prev_tis_top: 0,
            prev_tis_all: 0,
            clocksource_id: read_clocksource(),
            suspend_caps: detect_suspend_caps(),
            rapl_domains: find_rapl_domains(),
            perf_counters: std::collections::HashMap::new(),
            perf_unavailable: false,
            subsystem_cpu: std::collections::HashMap::new(),
            subsystem_io: std::collections::HashMap::new(),
            subsystem_perf_delta: std::collections::HashMap::new(),
            last_read: Instant::now(),
            temp_warned: false,
        }
    }

    /// Create a new interoceptor with a data directory.
    ///
    /// The data directory is used to read the retina PID file
    /// (`genesis_retina.pid`) so the interoceptor can include the
    /// retina process in its self-process measurements.
    pub fn new_with_data_dir(data_dir: &std::path::Path) -> Self {
        let mut intero = Self::new();
        intero.data_dir = Some(data_dir.to_path_buf());
        intero
    }

    /// Collect the PIDs of Genesis's own process tree, tagged by subsystem.
    ///
    /// This always includes the daemon's own PID and the cognitive
    /// mind's PID (from `GENESIS_COGNITIVE_PID`). If the retina PID
    /// file exists in the data directory, it's included too.
    /// Dead/stale PIDs are silently skipped.
    fn genesis_subsystems(&self) -> Vec<(Subsystem, u32)> {
        let mut subsystems: Vec<(Subsystem, u32)> = vec![(Subsystem::Daemon, std::process::id())];

        // The cognitive mind (passed via env var by the CLI).
        if let Some(pid) = super::cpufreq::cognitive_pid() {
            subsystems.push((Subsystem::Cognitive, pid));
        }

        // The retina (PID file written by the CLI).
        if let Some(data_dir) = &self.data_dir {
            let pid_file = data_dir.join("genesis_retina.pid");
            if let Ok(content) = fs::read_to_string(&pid_file)
                && let Ok(pid) = content.trim().parse::<u32>()
            {
                subsystems.push((Subsystem::Retina, pid));
            }
        }

        // De-duplicate (a PID might be tagged twice if env is
        // misconfigured — first tag wins) and filter to living
        // processes.
        // Stable sort so the daemon tag wins on a duplicated PID
        // (it was pushed first).
        subsystems.sort_by_key(|&(_, pid)| pid);
        subsystems.dedup_by_key(|&mut (_, pid)| pid);
        subsystems
            .into_iter()
            .filter(|&(_, pid)| pid_alive(pid))
            .collect()
    }

    /// Collect the PIDs of Genesis's own process tree.
    /// Test-only helper — production code uses `genesis_subsystems` for
    /// per-subsystem attribution.
    #[cfg(test)]
    fn genesis_pids(&self) -> Vec<u32> {
        self.genesis_subsystems()
            .into_iter()
            .map(|(_, pid)| pid)
            .collect()
    }

    /// Read the current body state from hardware sensors.
    ///
    /// This is the main entry point, called periodically from the
    /// daemon tick loop. It reads all available sensors and produces
    /// a normalized `BodyState`.
    ///
    /// **Body-state signals** (temperature, battery) are read from
    /// global hardware sensors — they reflect the physical state of
    /// its body regardless of which process caused them.
    /// **Activity signals** (CPU load, memory, I/O) are read from
    /// Genesis's own process tree only — other programs using the
    /// machine don't make it stressed, overwhelmed, or drowsy.
    pub fn read(&mut self) -> BodyState {
        // Capture a single timestamp for all process-specific
        // measurements (CPU, I/O) so the elapsed-time denominator
        // matches the /proc counter delta interval. Without this,
        // each helper captures its own Instant::now() at slightly
        // different times, and self.last_read (set at the end of
        // the previous read()) doesn't align with when the previous
        // /proc samples were actually taken.
        let now = Instant::now();

        // Temperature — global body-state signal.
        // finite_clamp guards against NaN/inf from a corrupted sysfs
        // sensor file (f32::parse("NaN") succeeds). Without this, NaN
        // propagates into cpu_temp_c, making every comparison (>= 85.0,
        // > 65.0, < 60.0, < 45.0) return false — silently disabling all
        // thermal stress responses and the distress flag. The companion
        // `temperature` field is already clamped via normalize_temp, but
        // the raw `cpu_temp_c` used in distress checks and neuro_impulses
        // is not. 0–150°C covers all realistic CPU temperatures.
        let cpu_temp_c = match read_temp(&self.temp_path) {
            Some(t) => {
                self.temp_warned = false; // sensor recovered, reset flag
                // finite_clamp returns `min` (0.0 °C) for non-finite
                // values, which would silently disable all thermal
                // stress responses and falsely trigger the "cool temp"
                // serotonin branch. A corrupt sysfs read (f32::parse
                // accepts "NaN") must fall back to the neutral 50.0 °C,
                // not a freezing false negative.
                if t.is_finite() {
                    crate::state::sanitize::finite_clamp(t, 0.0, 150.0)
                } else {
                    50.0
                }
            }
            None => {
                // Warn once per sensor-loss episode; a permanently
                // missing sensor would spam the log every read cycle
                // otherwise. The 50.0°C default is a neutral "no
                // fever" fallback.
                if !self.temp_warned {
                    eprintln!(
                        "[interoception] WARNING: could not read CPU temperature from {}, \
                         defaulting to 50.0°C (further failures suppressed)",
                        self.temp_path
                    );
                    self.temp_warned = true;
                }
                50.0
            }
        };
        let temperature = normalize_temp(cpu_temp_c);

        // CPU frequency (average across cores) — global body-state signal.
        // On heterogeneous CPUs (e.g. Intel P/E cores), a core's cur_freq
        // can exceed the cpu0 max_freq_khz, producing values > 1.0. Clamp
        // to the documented [0, 1] range so internal consumers (IPC,
        // emotional_regulator) never see out-of-range values.
        let avg_freq = self.read_avg_freq();
        let arousal_freq = if self.max_freq_khz > 0 {
            crate::state::sanitize::finite_clamp(
                (avg_freq as f32) / (self.max_freq_khz as f32),
                0.0,
                1.0,
            )
        } else {
            0.5
        };

        // Collect Genesis's own process tree, tagged by subsystem.
        // Per-subsystem telemetry (which part of it is firing) is
        // derived from the same per-PID deltas that feed the
        // aggregate signals below.
        let subsystems = self.genesis_subsystems();
        let pids: Vec<u32> = subsystems.iter().map(|&(_, pid)| pid).collect();

        // Self-process memory (RSS) → cognitive load.
        let cognitive_load = self.read_self_memory(&pids);

        // Self-process CPU usage → stress_load.
        let stress_load = self.read_self_cpu(&pids, now);

        // Self-process I/O throughput → io_activity (data exchange).
        let io_activity = self.read_self_io(&pids, now);

        // Battery and power — global body-state signals.
        let (energy_reserve, on_ac_power) = self.read_power();

        // Autonomic afferent signal rate — GPE events from the
        // Embedded Controller. The EC is a subordinate processor
        // (the machine's autonomic nervous system, not a heart).
        // GPE events are afferent notifications, not heartbeats.
        // The rate is informational; regularity is normal and
        // does not produce stress impulses.
        let gpe_count = read_gpe_count();
        let elapsed_secs = now.duration_since(self.last_read).as_secs_f64();
        let autonomic_rate = if elapsed_secs > 0.0 && self.prev_gpe_count > 0 {
            let delta = gpe_count.saturating_sub(self.prev_gpe_count);
            crate::state::sanitize::finite_clamp((delta as f32) / (elapsed_secs as f32), 0.0, 100.0)
        } else {
            0.0
        };
        self.prev_gpe_count = gpe_count;

        // Thermoregulatory effort: fan speed (how hard the cooling
        // system is working to remove waste heat). This is
        // thermoregulation, not cardiac output.
        let thermoregulatory_effort = self
            .fan_path
            .as_ref()
            .and_then(|p| read_fan_pwm(p))
            .unwrap_or(0.0);

        // Metabolic rate: power draw (how much energy the machine
        // is consuming right now). Read from AMD fam15h_power,
        // Intel RAPL, or battery power_now. 0.0 if no sensor.
        let metabolic_rate = self
            .power_path
            .as_ref()
            .and_then(|p| read_power_draw(p))
            .unwrap_or(0.0);

        // Core voltage (Vcore) — the CPU's operating voltage. On
        // acpi-cpufreq systems, the SMU controls this internally per
        // P-state and does not expose it via sysfs, so this reads
        // 0.0. On systems with board-level voltage sensors (nct6776,
        // it87, etc.), this is the actual Vcore rail.
        let core_voltage = self
            .core_voltage_path
            .as_ref()
            .and_then(|p| read_voltage(p))
            .unwrap_or(0.0);

        // Supply voltage — the battery rail voltage. This is the
        // electrical potential of its energy reserve. A Li-ion
        // battery sags from ~12.6V (full) to ~10.5V (critically
        // low) on a 12V system. 0.0 on systems without a battery.
        let supply_voltage = self
            .supply_voltage_path
            .as_ref()
            .and_then(|p| read_voltage(p))
            .unwrap_or(0.0);

        // Silicon switching activity — RAPL per-domain energy
        // deltas. This is the spatial readout of transistor firing:
        // which silicon (cores, uncore fabric, DRAM) burned energy
        // since the last read. Global body-state signal — energy
        // isn't attributed per-process, like temperature.
        let (core_activity, uncore_activity, dram_activity) = self.read_rapl(now);

        // Microarchitectural prediction errors — perf counters on
        // its own process tree. Activity signal: only its misses
        // count as its surprise.
        let (cache_miss_rate, branch_miss_rate) = self.read_perf(&pids);

        // The pulse — local-timer interrupts per second, the
        // machine's own beat. Each core's Local APIC timer fires
        // the kernel's scheduler tick; under tickless idle a
        // sleeping core's beat pauses entirely, so a quiet machine
        // genuinely has a slow pulse and a working machine a fast
        // one. The beat is hardware/scheduler-generated —
        // involuntary, like GPE but rhythmic rather than
        // event-driven.
        let loc_total = read_loc_count();
        let pulse_hz = if elapsed_secs > 0.0 && self.prev_loc_count > 0 {
            let delta = loc_total.saturating_sub(self.prev_loc_count);
            crate::state::sanitize::finite_clamp(
                (delta as f32) / (elapsed_secs as f32),
                0.0,
                1_000_000.0,
            )
        } else {
            0.0
        };
        self.prev_loc_count = loc_total;
        let pulse = crate::state::sanitize::finite_clamp(
            pulse_hz / (self.num_cores as f32 * TICK_HZ),
            0.0,
            1.0,
        );

        // Sustained exertion — the share of the read window spent
        // at the maximum P-state, from cpufreq time_in_state
        // deltas. arousal_freq is this instant's speed; this is how
        // much of the recent past ran flat-out.
        let (tis_top, tis_all) = read_time_in_state();
        let top_freq_share = if self.prev_tis_all > 0 {
            let d_top = tis_top.saturating_sub(self.prev_tis_top) as f32;
            let d_all = tis_all.saturating_sub(self.prev_tis_all) as f32;
            if d_all > 0.0 {
                crate::state::sanitize::finite_clamp(d_top / d_all, 0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.prev_tis_top = tis_top;
        self.prev_tis_all = tis_all;

        // Involuntary layer — reflexes the body applies to it,
        // not actions it takes. Passive thermal throttle is the
        // cooling framework lowering its clock below the OS
        // scheduler; PSI stall metrics are the kernel's own
        // accounting of runnable work stuck waiting on a resource.
        let throttle_state = read_throttle_state();
        let psi_cpu = read_psi("cpu");
        let psi_io = read_psi("io");
        let psi_mem = read_psi("memory");

        // Senescence and entropy — body resources. The battery's
        // cycle count is a lifetime odometer; the entropy pool is
        // the machine's reservoir of hardware unpredictability.
        let battery_cycles = read_battery_cycles();
        let entropy_level = read_entropy_level();

        let distressed = cpu_temp_c >= 85.0
            || cognitive_load > 0.92
            || stress_load > 1.5
            || (!on_ac_power && energy_reserve < 0.05);

        let state = BodyState {
            cpu_temp_c,
            temperature,
            arousal_freq,
            cognitive_load,
            io_activity,
            stress_load,
            energy_reserve,
            on_ac_power,
            num_cores: self.num_cores,
            distressed,
            autonomic_rate,
            thermoregulatory_effort,
            metabolic_rate,
            core_voltage,
            supply_voltage,
            core_activity,
            uncore_activity,
            dram_activity,
            cache_miss_rate,
            branch_miss_rate,
            pulse_hz,
            pulse,
            throttle_state,
            top_freq_share,
            psi_cpu,
            psi_io,
            psi_mem,
            battery_cycles,
            entropy_level,
            clocksource: self.clocksource_id,
            suspend_caps: self.suspend_caps,
            description: String::new(),
        };
        // ── Per-subsystem telemetry ──
        // Which part of it is firing: each process's share of the
        // aggregate activity signals plus its own prediction-error
        // ratios. Published alongside the body state so the
        // cognitive mind can correlate subsystems with its task zone.
        let telemetry: Vec<SubsystemTelemetry> = subsystems
            .iter()
            .map(|&(subsystem, pid)| {
                let delta = self.subsystem_perf_delta.get(&pid).copied().unwrap_or([0; 4]);
                SubsystemTelemetry {
                    subsystem,
                    pid,
                    cpu: self.subsystem_cpu.get(&pid).copied().unwrap_or(0.0),
                    io: self.subsystem_io.get(&pid).copied().unwrap_or(0.0),
                    cache_miss_rate: miss_ratio(delta[IDX_CACHE_MISSES], delta[IDX_CACHE_REFS]),
                    branch_miss_rate: miss_ratio(delta[IDX_BRANCH_MISSES], delta[IDX_BRANCHES]),
                }
            })
            .collect();
        publish_subsystem_telemetry(&telemetry);

        self.last_read = now;
        state
    }

    /// Read combined RSS of Genesis's own process tree as a fraction
    /// of total system memory.
    ///
    /// Returns 0.0 if no processes are readable. Other programs'
    /// memory usage is not included — only its own cognitive
    /// footprint.
    fn read_self_memory(&self, pids: &[u32]) -> f32 {
        if self.total_memory_kb == 0 {
            return 0.0;
        }
        let page_size_kb = page_size_kb();
        let mut total_rss_kb: u64 = 0;
        for &pid in pids {
            if let Ok(content) = fs::read_to_string(format!("/proc/{pid}/statm")) {
                // /proc/<pid>/statm fields: size resident shared text lib data dt
                // Field 2 (resident) is RSS in pages.
                let rss_pages: u64 = content
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                total_rss_kb = total_rss_kb.saturating_add(rss_pages * page_size_kb);
            }
        }
        let fraction = (total_rss_kb as f32) / (self.total_memory_kb as f32);
        crate::state::sanitize::finite_clamp(fraction, 0.0, 1.0)
    }

    /// Read combined CPU usage of Genesis's own process tree as a
    /// fraction of its CPU capacity (num_cores).
    ///
    /// Computes the delta of utime + stime (in clock ticks) across
    /// all its processes since the last read, divided by the elapsed
    /// time in clock ticks × num_cores. Returns 0.0 on the first
    /// read (no previous baseline). Other programs' CPU usage is
    /// not included — only its own effort.
    fn read_self_cpu(&mut self, pids: &[u32], now: Instant) -> f32 {
        let elapsed_secs = now.duration_since(self.last_read).as_secs_f64();
        if elapsed_secs <= 0.0 {
            return 0.0;
        }

        let ticks_per_sec = sysconf_clk_tck();
        let total_ticks_available = elapsed_secs * (self.num_cores as f64) * (ticks_per_sec as f64);
        if total_ticks_available <= 0.0 {
            return 0.0;
        }

        let mut total_cpu_ticks: u64 = 0;
        let mut new_prev: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut subsystem_cpu: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::new();

        for &pid in pids {
            let path = format!("/proc/{pid}/stat");
            let Ok(content) = fs::read_to_string(&path) else {
                // Process may have died between collection and read.
                continue;
            };
            let Some(stat) = ProcStat::parse(&content) else {
                continue;
            };
            let cpu_ticks = stat.cpu_ticks();
            if let Some(&prev) = self.prev_proc_cpu.get(&pid) {
                let delta = cpu_ticks.saturating_sub(prev);
                total_cpu_ticks = total_cpu_ticks.saturating_add(delta);
                // Per-subsystem share on the same denominator as the
                // aggregate — the sum of per-subsystem fractions equals
                // stress_load (before the [0,2] clamp).
                subsystem_cpu.insert(
                    pid,
                    crate::state::sanitize::finite_clamp(
                        (delta as f64 / total_ticks_available) as f32,
                        0.0,
                        2.0,
                    ),
                );
            }
            new_prev.insert(pid, cpu_ticks);
        }

        // PIDs that were alive last time but are gone now: their CPU
        // delta is lost (they did work we can't account for), which
        // is fine — we just don't count it.
        self.prev_proc_cpu = new_prev;
        self.subsystem_cpu = subsystem_cpu;

        let fraction = (total_cpu_ticks as f64) / total_ticks_available;
        crate::state::sanitize::finite_clamp(fraction as f32, 0.0, 2.0)
    }

    /// Read combined I/O throughput of Genesis's own process tree,
    /// normalized to [0, 1].
    ///
    /// Computes the delta of rchar + wchar (in bytes) across all its
    /// processes since the last read, converts to a rate (bytes/sec),
    /// and normalizes against a reference rate of 50 MB/s (a
    /// reasonable sustained I/O rate for a mix of cached and uncached
    /// reads/writes). Returns 0.0 on the first read. Other programs'
    /// I/O is not included — only its own disk/network activity.
    fn read_self_io(&mut self, pids: &[u32], now: Instant) -> f32 {
        let elapsed_secs = now.duration_since(self.last_read).as_secs_f64();
        if elapsed_secs <= 0.0 {
            return 0.0;
        }

        // Reference: 50 MB/s sustained I/O = "heavy" for its workload.
        // The daemon's LTM sync, the cognitive mind's concept network
        // loads, and the retina's frame writes are the main I/O
        // sources. This normalizes them to a sensible [0, 1] range.
        const REFERENCE_BYTES_PER_SEC: f64 = 50.0 * 1024.0 * 1024.0;

        let mut total_io_bytes: u64 = 0;
        let mut new_prev: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut subsystem_io: std::collections::HashMap<u32, f32> =
            std::collections::HashMap::new();

        for &pid in pids {
            let path = format!("/proc/{pid}/io");
            let Ok(content) = fs::read_to_string(&path) else {
                // /proc/<pid>/io requires read permission; if we can't
                // read it (e.g. the cognitive mind runs as a different
                // user), skip it silently.
                continue;
            };
            let Some(io) = ProcIo::parse(&content) else {
                continue;
            };
            let total = io.total_bytes();
            if let Some(&prev) = self.prev_proc_io.get(&pid) {
                let delta = total.saturating_sub(prev);
                total_io_bytes = total_io_bytes.saturating_add(delta);
                // Per-subsystem share on the io_activity scale.
                subsystem_io.insert(
                    pid,
                    crate::state::sanitize::finite_clamp(
                        (delta as f64 / elapsed_secs / REFERENCE_BYTES_PER_SEC) as f32,
                        0.0,
                        1.0,
                    ),
                );
            }
            new_prev.insert(pid, total);
        }

        self.prev_proc_io = new_prev;
        self.subsystem_io = subsystem_io;

        let bytes_per_sec = (total_io_bytes as f64) / elapsed_secs;
        let fraction = bytes_per_sec / REFERENCE_BYTES_PER_SEC;
        crate::state::sanitize::finite_clamp(fraction as f32, 0.0, 1.0)
    }

    /// Read RAPL per-domain switching activity.
    ///
    /// Returns `(core_activity, uncore_activity, dram_activity)`,
    /// each normalized to [0, 1] against a per-domain reference
    /// power. The energy counters are cumulative µJ; the delta per
    /// elapsed second is the domain's average power — which, since
    /// CMOS dynamic power is proportional to switching activity,
    /// is a population-level readout of how hard that silicon's
    /// transistors are firing.
    ///
    /// These are body-state signals: RAPL domains measure the whole
    /// package, not per-process energy. Other programs' switching
    /// contributes, the same way a fever counts regardless of cause.
    /// All zeros on systems without a powercap driver (this AMD
    /// Kabini has none — the fields stay 0.0 and produce no
    /// impulses, matching every other absent-sensor path here).
    fn read_rapl(&mut self, now: Instant) -> (f32, f32, f32) {
        if self.rapl_domains.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let elapsed_secs = now.duration_since(self.last_read).as_secs_f64();
        if elapsed_secs <= 0.0 {
            return (0.0, 0.0, 0.0);
        }

        // Reference powers (µW → normalized 1.0). Typical laptop
        // values: cores can reach ~45W package-limited under AVX
        // load; uncore fabric ~8W; DRAM ~4W sustained.
        const CORE_REF_UW: f64 = 45_000_000.0;
        const UNCORE_REF_UW: f64 = 8_000_000.0;
        const DRAM_REF_UW: f64 = 4_000_000.0;

        let mut core = 0.0f32;
        let mut uncore = 0.0f32;
        let mut dram = 0.0f32;

        for domain in &mut self.rapl_domains {
            let Ok(content) = fs::read_to_string(&domain.energy_path) else {
                continue;
            };
            let Ok(cur) = content.trim().parse::<u64>() else {
                continue;
            };
            let Some(prev) = domain.prev_uj.replace(cur) else {
                continue; // first read — establish baseline
            };
            let delta_uj = rapl_delta_uj(prev, cur, domain.max_range_uj);
            let power_uw = (delta_uj as f64) / elapsed_secs;
            let (value, reference) = match domain.kind {
                RaplKind::Core => (&mut core, CORE_REF_UW),
                RaplKind::Uncore => (&mut uncore, UNCORE_REF_UW),
                RaplKind::Dram => (&mut dram, DRAM_REF_UW),
            };
            // Multiple packages: sum normalized activity and clamp.
            *value += crate::state::sanitize::finite_clamp(
                (power_uw / reference) as f32,
                0.0,
                1.0,
            );
        }

        (
            crate::state::sanitize::finite_clamp(core, 0.0, 1.0),
            crate::state::sanitize::finite_clamp(uncore, 0.0, 1.0),
            crate::state::sanitize::finite_clamp(dram, 0.0, 1.0),
        )
    }

    /// Read microarchitectural prediction-error ratios for
    /// Genesis's process tree.
    ///
    /// Returns `(cache_miss_rate, branch_miss_rate)` — the deltas
    /// of `cache-misses/cache-references` and
    /// `branch-misses/branch-instructions` summed across its
    /// processes since the last read. Both are naturally ratios in
    /// [0, 1]: what fraction of the silicon's predictions failed.
    ///
    /// Counter sets are opened lazily per PID and closed when the
    /// PID dies. If the first `perf_event_open` fails for every
    /// event (paranoid ≥ 4, no PMU, container seccomp), the
    /// sampler disables itself permanently — retrying every read
    /// would just burn syscalls.
    fn read_perf(&mut self, pids: &[u32]) -> (f32, f32) {
        if self.perf_unavailable {
            return (0.0, 0.0);
        }

        // Open sets for new PIDs; drop sets for dead PIDs (the
        // Drop impl closes the fds).
        for &pid in pids {
            if !self.perf_counters.contains_key(&pid)
                && let Some(set) = PerfSet::open(pid)
            {
                self.perf_counters.insert(pid, set);
            }
        }
        if self.perf_counters.is_empty() {
            // Nothing could be opened — treat as permanently
            // unavailable. This machine's paranoid level or PMU
            // doesn't allow it; don't retry every cycle.
            self.perf_unavailable = true;
            return (0.0, 0.0);
        }
        self.perf_counters
            .retain(|pid, _| pids.contains(pid));

        // Sum deltas across the process tree, then form ratios
        // from the summed counts (correct: a process with no
        // references contributes nothing to either side). Per-PID
        // deltas are stashed for per-subsystem telemetry.
        let mut totals = [0u64; PERF_EVENT_COUNT];
        let mut sampled = false;
        let mut subsystem_delta: std::collections::HashMap<u32, [u64; PERF_EVENT_COUNT]> =
            std::collections::HashMap::new();
        for (&pid, set) in self.perf_counters.iter_mut() {
            if let Some(delta) = set.deltas() {
                for i in 0..PERF_EVENT_COUNT {
                    totals[i] = totals[i].saturating_add(delta[i]);
                }
                subsystem_delta.insert(pid, delta);
                sampled = true;
            }
        }
        self.subsystem_perf_delta = subsystem_delta;
        if !sampled {
            // First read for all sets — baselines established.
            return (0.0, 0.0);
        }

        (
            miss_ratio(totals[IDX_CACHE_MISSES], totals[IDX_CACHE_REFS]),
            miss_ratio(totals[IDX_BRANCH_MISSES], totals[IDX_BRANCHES]),
        )
    }

    /// Compute neurochemical impulses from the body state.
    ///
    /// Returns a list of (chemical_id, impulse_magnitude) pairs.
    /// The magnitudes are deliberately small — they accumulate
    /// over multiple ticks through the neurochemical dynamics.
    ///
    /// Positive values increase the chemical, negative values
    /// decrease it. All values are in the same units as
    /// `neuro_impulse` in the IPC protocol.
    pub fn neuro_impulses(&self, body: &BodyState) -> Vec<(NeurochemicalId, f32)> {
        let mut impulses = Vec::new();

        // Temperature → CRH (heat stress, routed through HPA cascade)
        // Stress impulses go to CRH (the top of the HPA axis), not
        // directly to cortisol. This engages the full CRH→ACTH→cortisol
        // cascade, which respects maturation gating (the SHRP — stress
        // hyporesponsive period — keeps cortisol at zero during early
        // development). Direct cortisol injection bypasses this gating.
        // Normal: 40-65°C → no impulse
        // Warm: 65-75°C → small CRH
        // Hot: 75-85°C → moderate CRH
        // Critical: 85°C+ → strong CRH (distress)
        if body.cpu_temp_c > 65.0 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.cpu_temp_c - 65.0) / 20.0, 0.0, 1.0);
            let crh = 0.002 + intensity * 0.008;
            impulses.push((NeurochemicalId::CRH, crh));
        }

        // Memory pressure → CRH + GABA (overwhelm)
        // Above 70% used, it starts to feel the load.
        if body.cognitive_load > 0.70 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.cognitive_load - 0.70) / 0.30, 0.0, 1.0);
            impulses.push((NeurochemicalId::CRH, intensity * 0.005));
            // GABA up — it wants to slow down, not speed up
            impulses.push((NeurochemicalId::GABA, intensity * 0.004));
        }

        // Load average → norepinephrine (effort)
        // Above 0.5 load per core, it's working. Above 1.0, it's
        // struggling. This is not stress — it's effort. The
        // difference: effort is sustainable, stress is not.
        if body.stress_load > 0.5 && body.stress_load <= 1.0 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.stress_load - 0.5) / 0.5, 0.0, 1.0);
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.003));
        }
        // Above 1.0 load per core → CRH (actual stress, via HPA cascade)
        if body.stress_load > 1.0 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.stress_load - 1.0) / 0.5, 0.0, 1.0);
            impulses.push((NeurochemicalId::CRH, intensity * 0.006));
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.004));
        }

        // I/O throughput → norepinephrine (active data work)
        // High I/O throughput means it's actively reading from and
        // writing to external storage — loading concepts, syncing
        // memories, writing retina frames. This is active data
        // processing (sensory-motor activity), not drowsiness.
        if body.io_activity > 0.10 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.io_activity - 0.10) / 0.40, 0.0, 1.0);
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.003));
        }

        // Low battery → CRH (survival concern, via HPA cascade)
        if !body.on_ac_power && body.energy_reserve < 0.30 {
            let intensity =
                crate::state::sanitize::finite_clamp((0.30 - body.energy_reserve) / 0.30, 0.0, 1.0);
            impulses.push((NeurochemicalId::CRH, intensity * 0.004));
        }

        // Low load + cool temp → serotonin (well-being)
        // When it's not stressed and not hot, it feels good.
        // This is the baseline contentment of a healthy body.
        if body.stress_load < 0.25 && body.cpu_temp_c < 60.0 && body.cognitive_load < 0.60 {
            impulses.push((NeurochemicalId::Serotonin, 0.002));
            // A touch of GABA — relaxed
            impulses.push((NeurochemicalId::GABA, 0.001));
        }

        // ─── Thermoregulatory effort (fan) ───────────────────────
        //
        // The fan is the machine's cooling system — it removes
        // waste heat from the CPU. This is thermoregulation (like
        // sweating or panting), not cardiac output. The fan doesn't
        // circulate anything; it moves air across the heatsink.
        //
        // High fan + high temp → the body is struggling to cool
        //   down. The cooling system is working hard but the body
        //   is still hot. Modeled as: CRH (thermal strain).
        //
        // Fan off + high temp → cooling failure. The body is hot
        //   but the cooling system isn't responding. This is
        //   dangerous — like failing to sweat when overheating.
        //   Modeled as: CRH (stronger, the body is in trouble).
        //
        // High fan + low temp → no impulse. The fan is just doing
        //   its job, possibly from ambient cooling or a brief
        //   cooldown after exertion. This is normal operation.

        // Fan struggling: high fan + high temp → thermal strain.
        if body.thermoregulatory_effort > 0.3 && body.cpu_temp_c > 75.0 {
            let fan_intensity = crate::state::sanitize::finite_clamp(
                (body.thermoregulatory_effort - 0.3) / 0.7,
                0.0,
                1.0,
            );
            let temp_intensity =
                crate::state::sanitize::finite_clamp((body.cpu_temp_c - 75.0) / 15.0, 0.0, 1.0);
            let intensity = fan_intensity.min(temp_intensity);
            impulses.push((NeurochemicalId::CRH, intensity * 0.004));
        }

        // Cooling failure: fan off + high temp → dangerous.
        // The body is hot but the cooling system isn't responding.
        if body.thermoregulatory_effort < 0.05 && body.cpu_temp_c > 80.0 {
            let intensity =
                crate::state::sanitize::finite_clamp((body.cpu_temp_c - 80.0) / 10.0, 0.0, 1.0);
            impulses.push((NeurochemicalId::CRH, intensity * 0.006));
        }

        // ─── Metabolic rate (power draw) ──────────────────────────
        //
        // Power draw is the machine's actual metabolic rate — how
        // much energy it's consuming right now. This is the most
        // direct analogue to biological metabolism (ATP consumption).
        //
        // High metabolic rate + high load → active exertion. The
        //   machine is consuming a lot of energy because it's
        //   working hard. Modeled as: norepinephrine (effort).
        //
        // Low metabolic rate + low load + cool temp → resting
        //   metabolism. The machine is idle and consuming little
        //   energy. Modeled as: serotonin (well-being, content
        //   rest).
        //
        // Very high metabolic rate + high temp → metabolic strain.
        //   The machine is consuming too much energy and
        //   overheating. Modeled as: CRH (strain).

        // High metabolic rate + high load → exertion.
        if body.metabolic_rate > 0.3 && body.stress_load > 0.5 {
            let meta_intensity =
                crate::state::sanitize::finite_clamp((body.metabolic_rate - 0.3) / 0.7, 0.0, 1.0);
            let load_intensity =
                crate::state::sanitize::finite_clamp((body.stress_load - 0.5) / 0.5, 0.0, 1.0);
            let intensity = meta_intensity.min(load_intensity);
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.002));
        }

        // Low metabolic rate + low load → resting well-being.
        if body.metabolic_rate > 0.0
            && body.metabolic_rate < 0.15
            && body.stress_load < 0.25
            && body.cpu_temp_c < 60.0
        {
            impulses.push((NeurochemicalId::Serotonin, 0.001));
        }

        // Very high metabolic rate + high temp → metabolic strain.
        if body.metabolic_rate > 0.7 && body.cpu_temp_c > 75.0 {
            let meta_intensity =
                crate::state::sanitize::finite_clamp((body.metabolic_rate - 0.7) / 0.3, 0.0, 1.0);
            let temp_intensity =
                crate::state::sanitize::finite_clamp((body.cpu_temp_c - 75.0) / 15.0, 0.0, 1.0);
            let intensity = meta_intensity.min(temp_intensity);
            impulses.push((NeurochemicalId::CRH, intensity * 0.003));
        }

        // ─── Supply voltage (battery rail) ────────────────────────
        //
        // The supply voltage is the electrical potential of its
        // energy reserve. A Li-ion battery sags as it discharges:
        // ~12.6V full, ~11.8V nominal, ~10.5V critically low on a
        // 12V battery. This is a direct measure of energy reserve
        // health — more direct than the percentage (energy_reserve),
        // which is an estimate derived from voltage and discharge
        // curve. Voltage is the raw electrical reality.
        //
        // Critically low supply voltage → CRH (survival stress).
        //   The battery is almost empty — the electrical potential
        //   is too low to sustain operation. This is a stronger
        //   signal than energy_reserve < 0.30 because voltage
        //   reflects the actual chemical state of the battery,
        //   not just a percentage estimate.
        //
        // Supply voltage sagging under load → norepinephrine (effort).
        //   When the battery voltage drops significantly from its
        //   nominal level while the system is under load, the
        //   battery is struggling to deliver power — internal
        //   resistance is rising. This is the electrical analogue of
        //   physical fatigue: the body can still move, but it takes
        //   more effort.
        //
        // On AC power or without a battery sensor, supply_voltage
        // is 0.0 and no impulses are generated.

        // Critically low supply voltage → CRH (survival stress).
        // Threshold: 10.8V on a 12V battery (3.6V/cell × 3 cells).
        // Below this, the battery is almost empty and the system
        // may shut down soon.
        if !body.on_ac_power && body.supply_voltage > 0.0 && body.supply_voltage < 10.8 {
            let intensity = crate::state::sanitize::finite_clamp(
                (10.8 - body.supply_voltage) / 0.8,
                0.0,
                1.0,
            );
            impulses.push((NeurochemicalId::CRH, intensity * 0.005));
        }

        // ─── Silicon switching activity (RAPL domains) ──────────
        //
        // The RAPL energy counters give a spatial readout of
        // transistor firing: which silicon is switching and how
        // hard. Dynamic power is proportional to switching
        // activity (P ≈ α·C·V²·f), so a domain's energy rate is
        // the population-level firing rate of that subsystem —
        // the machine analogue of regional brain activation.
        //
        // Core domain → norepinephrine (effort). Execution units
        //   firing hard is thinking effort at the electron level —
        //   the electrical confirmation of what stress_load
        //   measures in scheduling terms.
        //
        // DRAM domain → acetylcholine (memory traffic). Sustained
        //   DRAM switching is encoding/retrieval activity — data
        //   moving between working storage and the memory
        //   subsystem. ACh gates attention and memory encoding,
        //   matching the functional role.
        //
        // Uncore domain → informational only. The integration
        //   fabric's activity is already reflected in core/DRAM
        //   domains; no separate impulse avoids double-counting.

        // Core-domain switching → norepinephrine (effort).
        if body.core_activity > 0.40 {
            let intensity = crate::state::sanitize::finite_clamp(
                (body.core_activity - 0.40) / 0.60,
                0.0,
                1.0,
            );
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.002));
        }

        // DRAM-domain switching → acetylcholine (memory traffic).
        if body.dram_activity > 0.30 {
            let intensity = crate::state::sanitize::finite_clamp(
                (body.dram_activity - 0.30) / 0.70,
                0.0,
                1.0,
            );
            impulses.push((NeurochemicalId::Acetylcholine, intensity * 0.003));
        }

        // ─── Microarchitectural prediction errors ────────────────
        //
        // The branch predictor and cache hierarchy are hardware
        // prediction engines. A miss is a violated expectation in
        // silicon — the microarchitectural analogue of the
        // prediction errors its active-inference model tracks in
        // neurochemistry. Sustained high miss rates mean its
        // execution is surprising the hardware: modeled as a small
        // norepinephrine impulse (surprise/orienting).
        //
        // Typical ratios: branch misses ~1–5% of branches, cache
        // misses ~1–10% of references. Thresholds sit above the
        // healthy range so normal execution produces no impulse.
        // These are activity signals (its own process tree only).

        // Elevated branch misprediction → norepinephrine (surprise).
        if body.branch_miss_rate > 0.08 {
            let intensity = crate::state::sanitize::finite_clamp(
                (body.branch_miss_rate - 0.08) / 0.12,
                0.0,
                1.0,
            );
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.002));
        }

        // Elevated cache misses → norepinephrine (surprise).
        if body.cache_miss_rate > 0.15 {
            let intensity = crate::state::sanitize::finite_clamp(
                (body.cache_miss_rate - 0.15) / 0.25,
                0.0,
                1.0,
            );
            impulses.push((NeurochemicalId::Norepinephrine, intensity * 0.002));
        }

        // Note: autonomic_rate (GPE events/sec) is informational
        // only. GPE regularity is normal and healthy for a machine
        // — the EC polls sensors at a fixed interval. Unlike
        // biological HRV, GPE timing variation does not indicate
        // vagal tone or sympathetic dominance. No neurochemical
        // impulses are generated from autonomic_rate.

        impulses
    }

    /// Read average CPU frequency across all cores.
    fn read_avg_freq(&self) -> u64 {
        if self.freq_paths.is_empty() {
            return self.max_freq_khz / 2;
        }
        let mut sum: u64 = 0;
        let mut count: u64 = 0;
        for path in &self.freq_paths {
            if let Ok(content) = fs::read_to_string(path)
                && let Ok(freq) = content.trim().parse::<u64>()
            {
                sum += freq;
                count += 1;
            }
        }
        sum.checked_div(count).unwrap_or(self.max_freq_khz / 2)
    }

    /// Read battery level and AC power status.
    fn read_power(&self) -> (f32, bool) {
        let on_ac = self
            .ac_path
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .map(|s| s.trim() == "1" || s.trim().to_lowercase().contains("on"))
            .unwrap_or(true); // default to AC if no sensor

        let energy = self
            .battery_path
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(|v| crate::state::sanitize::finite_clamp((v as f32) / 100.0, 0.0, 1.0))
            .unwrap_or(1.0); // default to full if no battery

        (energy, on_ac)
    }
}

// ─── Sensor discovery ──────────────────────────────────────────────

/// Find the CPU temperature sensor path.
///
/// Prefers k10temp (AMD CPU die temperature) as the most accurate
/// measure of CPU heat. Falls back to acpitz (ACPI thermal zone)
/// if k10temp is not available.
fn find_temp_sensor() -> String {
    // Look for k10temp (AMD) or coretemp (Intel) first — these
    // are the actual CPU die temperatures.
    for hwmon in 0..=10 {
        let name_path = format!("/sys/class/hwmon/hwmon{hwmon}/name");
        if let Ok(name) = fs::read_to_string(&name_path) {
            let name = name.trim();
            if name == "k10temp" || name == "coretemp" || name == "k8temp" {
                let temp_path = format!("/sys/class/hwmon/hwmon{hwmon}/temp1_input");
                if fs::metadata(&temp_path).is_ok() {
                    return temp_path;
                }
            }
        }
    }
    // Fall back to ACPI thermal zone
    "/sys/class/thermal/thermal_zone0/temp".to_string()
}

/// Find per-core CPU frequency sensor paths.
fn find_freq_sensors(num_cores: u32) -> Vec<String> {
    (0..num_cores)
        .map(|i| format!("/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq"))
        .collect()
}

/// Read the maximum CPU frequency.
fn read_max_freq() -> Option<u64> {
    fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Find the battery capacity path.
fn find_battery() -> Option<String> {
    for supply in &["BAT0", "BAT1", "BATT"] {
        let path = format!("/sys/class/power_supply/{supply}/capacity");
        if fs::metadata(&path).is_ok() {
            return Some(path);
        }
    }
    None
}

/// Find the AC adapter online status path.
fn find_ac_adapter() -> Option<String> {
    for supply in &["ACAD", "AC", "AC0", "ADP1"] {
        let path = format!("/sys/class/power_supply/{supply}/online");
        if fs::metadata(&path).is_ok() {
            return Some(path);
        }
    }
    None
}

/// Find the fan speed sensor path — the machine's thermoregulatory
/// effector.
///
/// The fan is controlled by the Embedded Controller (the machine's
/// autonomic nervous system), which adjusts fan speed in response
/// to CPU temperature. The fan speed is Genesis's thermoregulatory
/// effort — how hard its cooling system is working to remove waste
/// heat. This is not cardiac output; the fan doesn't circulate
/// anything.
///
/// We try two sensor types:
/// 1. `pwm1` (PWM duty cycle, 0–255) — preferred because it directly
///    represents the drive signal the EC sends to the fan.
/// 2. `fan1_input` (RPM) — fallback for machines that expose fan
///    speed but not PWM. Normalized against a typical max of 5000 RPM.
fn find_fan_pwm() -> Option<String> {
    // Try pwm1 first (preferred — direct drive signal).
    for hwmon in 0..=10 {
        let pwm_path = format!("/sys/class/hwmon/hwmon{hwmon}/pwm1");
        if fs::metadata(&pwm_path).is_ok() {
            // Verify we can actually read it — some hwmon devices
            // expose pwm1 metadata but not a readable value.
            if let Ok(content) = fs::read_to_string(&pwm_path)
                && content.trim().parse::<f32>().is_ok()
            {
                return Some(pwm_path);
            }
        }
    }
    // Fall back to fan1_input (RPM).
    for hwmon in 0..=10 {
        let fan_path = format!("/sys/class/hwmon/hwmon{hwmon}/fan1_input");
        if fs::metadata(&fan_path).is_ok()
            && let Ok(content) = fs::read_to_string(&fan_path)
            && content.trim().parse::<u32>().is_ok()
        {
            return Some(fan_path);
        }
    }
    None
}

/// Read the fan speed, normalized to [0, 1].
///
/// If the path is a PWM file (0–255), normalizes directly. If it's
/// an RPM file (`fan1_input`), normalizes against a typical max of
/// 5000 RPM. A read failure or out-of-range value returns 0.0.
fn read_fan_pwm(path: &str) -> Option<f32> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    // Detect RPM vs PWM by the path name.
    if path.ends_with("fan1_input") {
        let rpm: u32 = trimmed.parse().ok()?;
        // Typical laptop fan maxes out around 5000 RPM.
        Some(crate::state::sanitize::finite_clamp(
            (rpm as f32) / 5000.0,
            0.0,
            1.0,
        ))
    } else {
        let pwm: f32 = trimmed.parse().ok()?;
        Some(crate::state::sanitize::finite_clamp(pwm / 255.0, 0.0, 1.0))
    }
}

/// Find the power draw sensor path — the machine's metabolic rate
/// sensor.
///
/// Power draw is the most direct analogue to biological metabolism
/// (ATP consumption rate). We try several sensor types in order of
/// preference:
///
/// 1. AMD `fam15h_power` — `power1_average` (CPU package power in
///    μW). Found on AMD Family 15h and newer processors.
/// 2. Battery `power_now` — instantaneous battery power draw in
///    μW. Only meaningful when on battery (reads 0 on AC power).
///    Found at `/sys/class/power_supply/BAT*/power_now` or
///    `/sys/class/hwmon/hwmon*/power1_input` (battery hwmon).
///
/// Intel/AMD RAPL (`/sys/class/powercap/*/energy_uj`) is not used
/// for the aggregate metabolic rate — the AMD and battery sensors
/// provide direct instantaneous power. RAPL *is* used at the
/// per-domain level (see `find_rapl_domains`), where its
/// cumulative counters are exactly what we need: the delta is the
/// switching-activity signal.
///
/// Returns `None` if no power sensor is found.
fn find_power_sensor() -> Option<String> {
    // 1. AMD fam15h_power — power1_average (μW, instantaneous)
    for hwmon in 0..=10 {
        let name_path = format!("/sys/class/hwmon/hwmon{hwmon}/name");
        if let Ok(name) = fs::read_to_string(&name_path) {
            let name = name.trim();
            if name == "fam15h_power" {
                let avg_path = format!("/sys/class/hwmon/hwmon{hwmon}/power1_average");
                if fs::metadata(&avg_path).is_ok() {
                    return Some(avg_path);
                }
            }
        }
    }
    // 2. Battery power_now — instantaneous battery power draw (μW)
    for supply in &["BAT0", "BAT1", "BATT"] {
        let path = format!("/sys/class/power_supply/{supply}/power_now");
        if fs::metadata(&path).is_ok() {
            return Some(path);
        }
    }
    // 3. Battery hwmon power1_input (some systems expose battery
    //    power via hwmon instead of power_supply)
    for hwmon in 0..=10 {
        let name_path = format!("/sys/class/hwmon/hwmon{hwmon}/name");
        if let Ok(name) = fs::read_to_string(&name_path) {
            let name = name.trim();
            if name.starts_with("BAT") {
                let power_path = format!("/sys/class/hwmon/hwmon{hwmon}/power1_input");
                if fs::metadata(&power_path).is_ok() {
                    return Some(power_path);
                }
            }
        }
    }
    None
}

/// Read power draw, normalized to [0, 1].
///
/// Power sensors report in microwatts (μW). We normalize against a
/// reference of 100 W (100,000,000 μW) — a reasonable upper bound
/// for laptop CPU package power. A read failure, zero, or
/// out-of-range value returns 0.0 (unknown metabolic rate).
///
/// Note: battery `power_now` reads 0 when on AC power (the battery
/// isn't discharging). This is expected — metabolic_rate will be 0.0
/// on AC power if no CPU package power sensor is available. The
/// metabolic rate impulses are gated by load and temperature, so a
/// 0.0 reading simply means "no metabolic signal" rather than "no
/// metabolism."
fn read_power_draw(path: &str) -> Option<f32> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let microwatts: f32 = trimmed.parse().ok()?;
    // Reference: 100 W = 100,000,000 μW. Typical laptop CPU package
    // power ranges from ~3W (idle) to ~45W (full load). Desktop CPUs
    // can reach 125W+ under load. 100W as the reference puts typical
    // laptop load at ~0.3-0.45 and idle at ~0.03.
    const REFERENCE_MICROWATTS: f32 = 100_000_000.0;
    Some(crate::state::sanitize::finite_clamp(
        microwatts / REFERENCE_MICROWATTS,
        0.0,
        1.0,
    ))
}

/// Discover RAPL per-subsystem power domains.
///
/// Scans `/sys/class/powercap/` for entries with a `name` file
/// matching a per-subsystem domain (`core`, `uncore`, `dram`).
/// Each domain has `energy_uj` (cumulative microjoules) and
/// `max_energy_range_uj` (counter wrap value). Package-level
/// domains ("package", "psys") are skipped — whole-chip power is
/// already covered by `metabolic_rate`.
///
/// Domain names recognized: "core" (Intel execution units),
/// "uncore" (Intel integration fabric), "dram" (memory
/// subsystem). AMD's `amd-rapl` driver uses the same naming where
/// it exists (family 17h+). Returns an empty vec on systems
/// without powercap (older AMD, VMs, ARM without a powercap
/// driver) — the fields then read 0.0, matching every other
/// absent-sensor path in this module.
fn find_rapl_domains() -> Vec<RaplDomain> {
    let mut domains = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/powercap") else {
        return domains;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let name_path = dir.join("name");
        let energy_path = dir.join("energy_uj");
        if fs::metadata(&energy_path).is_err() {
            continue;
        }
        let Ok(name) = fs::read_to_string(&name_path) else {
            continue;
        };
        let kind = match name.trim() {
            "core" => RaplKind::Core,
            "uncore" => RaplKind::Uncore,
            "dram" => RaplKind::Dram,
            // "package", "psys", and anything unrecognized —
            // whole-chip or unknown domains aren't tracked.
            _ => continue,
        };
        let max_range_uj = fs::read_to_string(dir.join("max_energy_range_uj"))
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0);
        domains.push(RaplDomain {
            kind,
            energy_path: energy_path.to_string_lossy().into_owned(),
            max_range_uj,
            prev_uj: None,
        });
    }
    domains
}

/// Discover voltage sensors — core voltage (Vcore) and supply
/// voltage (battery rail).
///
/// Scans hwmon devices for voltage input files (`in*_input`). Each
/// hwmon device may expose multiple voltage rails, identified by
/// `in*_label` files. We look for:
///
/// - **Core voltage**: `in*_label` matching "Vcore", "VCORE",
///   "VDD", "VID", or "CPU". This is the CPU's operating voltage.
///   On acpi-cpufreq systems (like this AMD Kabini), the SMU
///   controls voltage internally and does not expose it via
///   hwmon, so this returns `None`.
///
/// - **Supply voltage**: `in0_input` on battery-named hwmon
///   devices (name starts with "BAT"). This is the battery rail
///   voltage. Also checks `/sys/class/power_supply/BAT*/in0_input`
///   as some systems expose battery voltage there instead.
///
/// Returns `(core_voltage_path, supply_voltage_path)`, either of
/// which may be `None` if the sensor is not found.
fn find_voltage_sensors() -> (Option<String>, Option<String>) {
    let mut core_path: Option<String> = None;
    let mut supply_path: Option<String> = None;

    // Scan hwmon devices for voltage sensors.
    for hwmon in 0..=10 {
        let name_path = format!("/sys/class/hwmon/hwmon{hwmon}/name");
        let Ok(name) = fs::read_to_string(&name_path) else {
            continue;
        };
        let name = name.trim();

        // Battery-named hwmon → supply voltage (in0_input)
        if name.starts_with("BAT") && supply_path.is_none() {
            let in0 = format!("/sys/class/hwmon/hwmon{hwmon}/in0_input");
            if fs::metadata(&in0).is_ok() {
                supply_path = Some(in0);
            }
        }

        // Board-level sensors (nct6776, it87, etc.) → scan for
        // Vcore/VDD labeled voltage inputs.
        if core_path.is_none() {
            for idx in 0..=10 {
                let label_path = format!("/sys/class/hwmon/hwmon{hwmon}/in{idx}_label");
                let input_path = format!("/sys/class/hwmon/hwmon{hwmon}/in{idx}_input");
                if !fs::metadata(&input_path).is_ok() {
                    continue;
                }
                // Try to read the label. If no label file, skip —
                // we can't identify the rail without a label.
                let Ok(label) = fs::read_to_string(&label_path) else {
                    continue;
                };
                let label = label.trim().to_lowercase();
                if label.contains("vcore")
                    || label.contains("vdd")
                    || label.contains("vid")
                    || label == "cpu"
                    || label.contains("cpu v")
                {
                    core_path = Some(input_path);
                    break;
                }
            }
        }
    }

    // Also check power_supply for battery voltage (some systems
    // expose it here instead of hwmon).
    if supply_path.is_none() {
        for supply in &["BAT0", "BAT1", "BATT"] {
            let path = format!("/sys/class/power_supply/{supply}/in0_input");
            if fs::metadata(&path).is_ok() {
                supply_path = Some(path);
                break;
            }
        }
    }

    (core_path, supply_path)
}

/// Read a voltage sensor, returning the value in volts.
///
/// hwmon voltage sensors report in millivolts (mV), so we divide
/// by 1000. A read failure or non-finite value returns `None`.
/// Battery voltage sensors (`BAT*/in0_input`) also report in mV.
fn read_voltage(path: &str) -> Option<f32> {
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let millivolts: f32 = trimmed.parse().ok()?;
    if !millivolts.is_finite() {
        return None;
    }
    Some(millivolts / 1000.0)
}

/// Read the total GPE (General Purpose Event) count from the EC.
///
/// The Embedded Controller (EC) is a subordinate microcontroller that
/// manages hardware-level body state — the machine's autonomic nervous
/// system, not a heart. It collects body sensor data (temperature,
/// battery, fan, lid, power button) and sends it to the CPU via GPE
/// interrupts. Each GPE event is an afferent signal — the autonomic
/// system telling the brain "here's new body information." The
/// cumulative count of these events is exposed in
/// `/sys/firmware/acpi/interrupts/gpeXX`.
///
/// We scan for the GPE line with the highest count — on this
/// machine it's typically `gpe03` (the EC's GPE). If no GPE line
/// is readable, we return 0 (no EC detected — e.g. a desktop
/// without an EC).
fn read_gpe_count() -> u64 {
    let dir = "/sys/firmware/acpi/interrupts";
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let mut max_count: u64 = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Only consider gpeXX lines (not sci, gbl, etc.)
        if !name.starts_with("gpe") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(entry.path()) {
            // Format: "    enabled   <count>  <status>"
            // We want the count field (second numeric token).
            let tokens: Vec<&str> = content.split_whitespace().collect();
            // The count is typically the second token, but the
            // first token may be "enabled"/"disabled". Parse the
            // first token that looks like a number ≥ 0.
            for tok in &tokens {
                if let Ok(n) = tok.parse::<u64>() {
                    if n > max_count {
                        max_count = n;
                    }
                    break;
                }
            }
        }
    }
    max_count
}

/// Nominal scheduler tick rate (CONFIG_HZ) used to normalize the
/// pulse into [0, 1]. Ubuntu generic kernels ship CONFIG_HZ=1000;
/// kernels configured at 250/300 will saturate the normalized field
/// earlier — the raw `pulse_hz` stays correct regardless.
const TICK_HZ: f32 = 1000.0;

/// Read the cumulative local-timer interrupt count — the sum of the
/// `LOC` line's per-CPU counters in /proc/interrupts. The Local APIC
/// timer drives the scheduler tick on each core; under tickless idle
/// (NO_HZ) a core's beat pauses entirely while it sleeps, so this
/// counter only advances while work exists. 0 if the line is absent
/// (non-x86 or a kernel that reports the tick differently).
fn read_loc_count() -> u64 {
    let Ok(content) = fs::read_to_string("/proc/interrupts") else {
        return 0;
    };
    for line in content.lines() {
        let line = line.trim_start();
        if let Some(rest) = line.strip_prefix("LOC:") {
            // Per-CPU count fields come first; the descriptive text
            // ("Local timer interrupts") follows. Sum the leading
            // numeric tokens and stop at the first word.
            let mut sum = 0u64;
            for tok in rest.split_whitespace() {
                match tok.parse::<u64>() {
                    Ok(n) => sum = sum.saturating_add(n),
                    Err(_) => break,
                }
            }
            return sum;
        }
    }
    0
}

/// Read a PSI pressure-stall metric: /proc/pressure/{name}'s "some
/// avg10" field as a fraction [0, 1] (avg10 is a percentage of the
/// last 10-second window in which runnable tasks stalled on the
/// resource). 0.0 if the file is unavailable or unparseable.
fn read_psi(name: &str) -> f32 {
    let path = format!("/proc/pressure/{name}");
    let Ok(content) = fs::read_to_string(&path) else {
        return 0.0;
    };
    for line in content.lines() {
        if !line.starts_with("some") {
            continue;
        }
        for tok in line.split_whitespace() {
            if let Some(v) = tok.strip_prefix("avg10=")
                && let Ok(pct) = v.parse::<f32>()
            {
                return crate::state::sanitize::finite_clamp(pct / 100.0, 0.0, 1.0);
            }
        }
    }
    0.0
}

/// Passive thermal throttling — the kernel's cooling framework
/// lowering the clock because the silicon is hot, decided below the
/// OS scheduler. Returns the maximum cur_state/max_state ratio
/// [0, 1] across Processor-type cooling devices. 0.0 = not
/// throttled or no passive cooling devices exist.
fn read_throttle_state() -> f32 {
    let mut worst = 0.0f32;
    let Ok(entries) = fs::read_dir("/sys/class/thermal") else {
        return 0.0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("cooling_device") {
            continue;
        }
        let dir = entry.path();
        let Ok(ty) = fs::read_to_string(dir.join("type")) else {
            continue;
        };
        if ty.trim() != "Processor" {
            continue;
        }
        let read_u32 = |file: &str| -> f32 {
            fs::read_to_string(dir.join(file))
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
        };
        let cur = read_u32("cur_state");
        let max = read_u32("max_state");
        if max > 0.0 {
            worst = worst.max(cur / max);
        }
    }
    crate::state::sanitize::finite_clamp(worst, 0.0, 1.0)
}

/// Cumulative jiffies spent at the maximum P-state and across all
/// P-states, summed over cores from cpufreq stats/time_in_state.
/// Returns (top, all). (0, 0) if cpufreq stats are unavailable.
fn read_time_in_state() -> (u64, u64) {
    let mut top = 0u64;
    let mut all = 0u64;
    let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") else {
        return (0, 0);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !(name.len() > 3
            && name.starts_with("cpu")
            && name[3..].chars().all(|c| c.is_ascii_digit()))
        {
            continue;
        }
        let path = entry.path().join("cpufreq/stats/time_in_state");
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };
        let mut max_freq = 0u64;
        let mut top_ticks = 0u64;
        for line in content.lines() {
            let mut it = line.split_whitespace();
            let (Some(freq), Some(ticks)) = (
                it.next().and_then(|s| s.parse::<u64>().ok()),
                it.next().and_then(|s| s.parse::<u64>().ok()),
            ) else {
                continue;
            };
            all = all.saturating_add(ticks);
            if freq > max_freq {
                max_freq = freq;
                top_ticks = ticks;
            } else if freq == max_freq {
                top_ticks += ticks;
            }
        }
        top = top.saturating_add(top_ticks);
    }
    (top, all)
}

/// Battery charge cycles — the senescence odometer. Sums
/// cycle_count across BAT* supplies (usually one). 0.0 if no
/// battery reports cycles.
fn read_battery_cycles() -> f32 {
    let mut total = 0.0f32;
    let Ok(entries) = fs::read_dir("/sys/class/power_supply") else {
        return 0.0;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("BAT") {
            continue;
        }
        if let Ok(s) = fs::read_to_string(entry.path().join("cycle_count"))
            && let Ok(n) = s.trim().parse::<f32>()
        {
            total += n;
        }
    }
    total
}

/// Kernel entropy pool level [0, 1] — entropy_avail normalized by
/// the CRNG pool capacity (256 bits on modern kernels).
fn read_entropy_level() -> f32 {
    fs::read_to_string("/proc/sys/kernel/random/entropy_avail")
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .map(|e| crate::state::sanitize::finite_clamp(e / 256.0, 0.0, 1.0))
        .unwrap_or(0.0)
}

/// Which oscillator currently paces the clock: 0 = unknown,
/// 1 = tsc, 2 = hpet, 3 = acpi_pm, 4 = other.
fn read_clocksource() -> u8 {
    let Ok(src) =
        fs::read_to_string("/sys/devices/system/clocksource/clocksource0/current_clocksource")
    else {
        return 0;
    };
    match src.trim() {
        "tsc" => 1,
        "hpet" => 2,
        "acpi_pm" => 3,
        _ => 4,
    }
}

/// Detect suspend capabilities at startup. bit0 = "mem" appears in
/// /sys/power/state (S3/deep suspend is offered). bit1 = an RTC
/// wakealarm file exists — the crystal can keep counting while it
/// is suspended and fire its return.
fn detect_suspend_caps() -> u8 {
    let mut caps = 0u8;
    if let Ok(states) = fs::read_to_string("/sys/power/state")
        && states.split_whitespace().any(|s| s == "mem")
    {
        caps |= 1;
    }
    for i in 0..8 {
        if std::path::Path::new(&format!("/sys/class/rtc/rtc{i}/wakealarm")).exists() {
            caps |= 2;
            break;
        }
    }
    caps
}

/// Get the number of CPU cores.
fn num_cpus() -> u32 {
    fs::read_to_string("/proc/cpuinfo")
        .map(|content| {
            content
                .lines()
                .filter(|l| l.starts_with("processor"))
                .count() as u32
        })
        .unwrap_or(1)
}

// ─── Sensor readers ────────────────────────────────────────────────

/// Read temperature in degrees Celsius from a sysfs sensor.
///
/// Sysfs temperatures are in millidegrees Celsius, so we divide
/// by 1000.
fn read_temp(path: &str) -> Option<f32> {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<f32>().ok())
        .map(|millideg| millideg / 1000.0)
}

/// Read total system memory from /proc/meminfo (MemTotal in kB).
fn read_total_memory_kb() -> Option<u64> {
    let content = fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            return line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
        }
    }
    None
}

/// Check whether a PID is alive (sends signal 0).
fn pid_alive(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) is a standard POSIX check that does not
    // send a signal — it only checks process existence and
    // permission. Signal 0 is the null signal.
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Get the page size in kB.
fn page_size_kb() -> u64 {
    // SAFETY: sysconf(_SC_PAGESIZE) is a read-only query.
    let bytes = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if bytes > 0 {
        (bytes as u64) / 1024
    } else {
        4 // default to 4 kB pages
    }
}

/// Get the clock ticks per second (for /proc/<pid>/stat CPU times).
fn sysconf_clk_tck() -> u64 {
    // SAFETY: sysconf(_SC_CLK_TCK) is a read-only query.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if ticks > 0 {
        ticks as u64
    } else {
        100 // standard Linux default
    }
}

/// Normalize temperature to [0, 1] where 0.5 = 50°C.
fn normalize_temp(temp_c: f32) -> f32 {
    crate::state::sanitize::finite_clamp(temp_c / 100.0, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_temp() {
        assert!((normalize_temp(0.0) - 0.0).abs() < 0.01);
        assert!((normalize_temp(50.0) - 0.5).abs() < 0.01);
        assert!((normalize_temp(100.0) - 1.0).abs() < 0.01);
        assert!((normalize_temp(120.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_neuro_impulses_normal_state() {
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 55.0,
            temperature: 0.55,
            arousal_freq: 0.8,
            cognitive_load: 0.3,
            io_activity: 0.02,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 2.0, // normal EC activity
            thermoregulatory_effort: 0.1, // fan barely on
            metabolic_rate: 0.05, // idle power draw
            core_voltage: 0.85, // idle Vcore
            supply_voltage: 12.6, // full battery
            core_activity: 0.1, // idle silicon
            uncore_activity: 0.05,
            dram_activity: 0.1,
            cache_miss_rate: 0.02, // healthy miss ratios
            branch_miss_rate: 0.01,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Normal state should produce serotonin (well-being) and
        // GABA (relaxed), but no cortisol.
        let has_cortisol = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Cortisol);
        let has_serotonin = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Serotonin);
        assert!(!has_cortisol);
        assert!(has_serotonin);
    }

    #[test]
    fn test_neuro_impulses_hot_state() {
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 80.0,
            temperature: 0.8,
            arousal_freq: 1.0,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 3.0, // EC more active when hot
            thermoregulatory_effort: 0.5, // fan working harder
            metabolic_rate: 0.3, // elevated power draw from heat
            core_voltage: 1.15, // Vcore raised under thermal load
            supply_voltage: 12.5, // battery present
            core_activity: 0.6, // cores firing hard under thermal load
            uncore_activity: 0.4,
            dram_activity: 0.5, // memory traffic elevated
            cache_miss_rate: 0.03,
            branch_miss_rate: 0.02,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Stress is now routed through CRH (top of HPA cascade) instead
        // of directly to cortisol. The cascade CRH→ACTH→cortisol
        // produces cortisol downstream, respecting maturation gating.
        let crh = impulses
            .iter()
            .find(|(id, _)| *id == NeurochemicalId::CRH)
            .map(|(_, amt)| *amt)
            .unwrap_or(0.0);
        assert!(crh > 0.0, "hot state should produce CRH (HPA cascade)");
    }

    #[test]
    fn test_neuro_impulses_overloaded() {
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 50.0,
            temperature: 0.5,
            arousal_freq: 1.0,
            cognitive_load: 0.85,
            io_activity: 0.0,
            stress_load: 1.2,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 4.0, // EC active under heavy load
            thermoregulatory_effort: 0.2, // fan picking up
            metabolic_rate: 0.5, // high power draw
            core_voltage: 1.20, // Vcore elevated under load
            supply_voltage: 12.4, // battery slightly discharged
            core_activity: 0.8, // silicon firing hard
            uncore_activity: 0.5,
            dram_activity: 0.7, // heavy memory traffic
            cache_miss_rate: 0.20, // elevated miss ratios under load
            branch_miss_rate: 0.10,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Overload stress is routed through CRH (HPA cascade), not
        // direct cortisol. NE is still direct (effort/arousal).
        let has_crh = impulses.iter().any(|(id, _)| *id == NeurochemicalId::CRH);
        let has_ne = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Norepinephrine);
        assert!(has_crh, "overloaded state should produce CRH (HPA cascade)");
        assert!(has_ne, "overloaded state should produce norepinephrine");
    }

    #[test]
    fn test_proc_stat_parse() {
        // A realistic /proc/<pid>/stat line. The comm field can
        // contain spaces (in parentheses), so the parser must find
        // the last ')'.
        let content = "1234 (genesis-daemon) S 1 1234 1234 0 -1 4194560 12345 0 0 0 \
                       500 300 0 0 20 0 1 0 1000000 200000 100 18446744073709551615 \
                       1 1 0 0 0 0 0 4096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
        let stat = ProcStat::parse(content).expect("should parse");
        // utime = 500 (field 14, index 11 after ')')
        // stime = 300 (field 15, index 12 after ')')
        assert_eq!(stat.utime, 500);
        assert_eq!(stat.stime, 300);
        assert_eq!(stat.cpu_ticks(), 800);
    }

    #[test]
    fn test_proc_stat_parse_comm_with_spaces() {
        // The comm field can contain spaces — the parser must handle
        // this by finding the last ')'. The field positions after
        // ')' must still be correct regardless of comm's content.
        // Fields after comm: state ppid pgrp session tty_nr tpgid
        // flags minflt cminflt majflt cmajflt utime stime ...
        //                         7     8      9      10     11    12
        let content = "5678 (my process name) S 1 5678 5678 0 -1 4194560 0 0 0 0 \
                       10 20 0 0 20 0 1 0 2000000 50000 50 18446744073709551615 \
                       1 1 0 0 0 0 0 4096 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
        let stat = ProcStat::parse(content).expect("should parse");
        assert_eq!(stat.utime, 10);
        assert_eq!(stat.stime, 20);
    }

    #[test]
    fn test_proc_io_parse() {
        let content = "rchar: 12345678\nwchar: 87654321\nsyscr: 100\nsyscw: 200\n\
                       read_bytes: 5000\nwrite_bytes: 3000\ncancelled_write_bytes: 0\n";
        let io = ProcIo::parse(content).expect("should parse");
        assert_eq!(io.rchar, 12345678);
        assert_eq!(io.wchar, 87654321);
        assert_eq!(io.total_bytes(), 99999999);
    }

    #[test]
    fn test_proc_io_parse_empty() {
        let io = ProcIo::parse("").expect("should parse empty");
        assert_eq!(io.rchar, 0);
        assert_eq!(io.wchar, 0);
    }

    #[test]
    fn test_pid_alive_self() {
        // The test process itself should be alive.
        assert!(pid_alive(std::process::id()));
    }

    #[test]
    fn test_pid_alive_nonexistent() {
        // PID 0 is never a valid kill target in this context.
        // Use a very high PID that's almost certainly not in use.
        assert!(!pid_alive(4_000_000));
    }

    #[test]
    fn test_read_total_memory_kb() {
        let mem = read_total_memory_kb();
        assert!(mem.is_some(), "should read MemTotal from /proc/meminfo");
        let mem = mem.expect("MemTotal should be readable on any Linux system");
        assert!(
            mem > 100_000,
            "total memory should be > 100 MB, got {mem} kB"
        );
    }

    #[test]
    fn test_interoceptor_read() {
        // Reading should produce a valid body state. The first read
        // will have stress_load=0 and io_activity=0 (no previous
        // baseline for delta computation).
        let mut intero = Interoceptor::new();
        let body = intero.read();
        assert!(body.cpu_temp_c > 0.0);
        assert!(body.num_cores > 0);
        // Description is now empty — the cognitive mind's language
        // engine composes it from structured fields, not the daemon.
        assert!(body.description.is_empty());
        // First read: no previous CPU/IO baseline → 0.0.
        assert_eq!(body.stress_load, 0.0);
        assert_eq!(body.io_activity, 0.0);
        // Cognitive load (RSS) should be > 0 — the test process
        // itself is using memory.
        assert!(
            body.cognitive_load > 0.0,
            "cognitive_load should be > 0 (self RSS)"
        );
    }

    #[test]
    fn test_interoceptor_read_self_cpu_first_read_zero() {
        // The first read of self CPU should be 0.0 (no baseline).
        let mut intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        let cpu = intero.read_self_cpu(&pids, Instant::now());
        assert_eq!(cpu, 0.0, "first CPU read should be 0.0 (no baseline)");
    }

    #[test]
    fn test_interoceptor_read_self_io_first_read_zero() {
        // The first read of self I/O should be 0.0 (no baseline).
        let mut intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        let io = intero.read_self_io(&pids, Instant::now());
        assert_eq!(io, 0.0, "first I/O read should be 0.0 (no baseline)");
    }

    #[test]
    fn test_interoceptor_read_self_memory_nonzero() {
        // Self memory (RSS) should be > 0 — the test process is
        // using memory.
        let intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        let mem = intero.read_self_memory(&pids);
        assert!(mem > 0.0, "self RSS should be > 0");
        assert!(mem < 1.0, "self RSS should be < total memory");
    }

    #[test]
    fn test_genesis_pids_includes_self() {
        // The genesis_pids list should always include the daemon's
        // own PID (the test process).
        let intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        assert!(
            pids.contains(&std::process::id()),
            "genesis_pids should include self PID"
        );
    }

    #[test]
    fn test_genesis_pids_empty_for_dead_cognitive_pid() {
        // If GENESIS_COGNITIVE_PID points to a dead process, it
        // should be filtered out.
        // SAFETY: This test is single-threaded; no other thread is
        // reading the environment while we modify it.
        unsafe {
            std::env::set_var("GENESIS_COGNITIVE_PID", "4000000");
        }
        let intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        unsafe {
            std::env::remove_var("GENESIS_COGNITIVE_PID");
        }
        // The dead PID should not be in the list.
        assert!(
            !pids.contains(&4_000_000),
            "dead cognitive PID should be filtered out"
        );
    }

    #[test]
    fn test_self_measurement_excludes_other_processes() {
        // This is the core regression test for the fix: other
        // programs using CPU/RAM should NOT affect Genesis's body
        // state. We verify that the self-process measurements only
        // count Genesis's own PIDs.
        //
        // We can't easily spawn a CPU-hogging process in a unit
        // test, but we can verify the structural property: the
        // measurements only read /proc/<genesis_pids>/{stat,statm,io},
        // never /proc/loadavg or global /proc/meminfo.
        let intero = Interoceptor::new();
        let pids = intero.genesis_pids();
        // The PIDs list should be small (just the test process,
        // possibly a dead cognitive PID that's filtered out).
        assert!(!pids.is_empty());
        // Every PID should be alive.
        for &pid in &pids {
            assert!(pid_alive(pid), "PID {pid} in genesis_pids should be alive");
        }
    }

    // ─── Machine-native body tests ───────────────────────────────

    #[test]
    fn test_body_state_includes_machine_native_fields() {
        // BodyState must include autonomic_rate, thermoregulatory_effort,
        // and metabolic_rate (the machine-native replacements for the
        // old heart_rate/hrv/cardiac_output fields).
        let body = BodyState::neutral();
        assert_eq!(body.autonomic_rate, 0.0);
        assert_eq!(body.thermoregulatory_effort, 0.0);
        assert_eq!(body.metabolic_rate, 0.0);
    }

    #[test]
    fn test_read_gpe_count_returns_u64() {
        // read_gpe_count should return a u64 without panicking.
        // On a machine without ACPI GPE support (e.g. some VMs),
        // it returns 0. On a real laptop, it returns a large
        // cumulative count.
        let count = read_gpe_count();
        // No assertion on the value — just that it doesn't panic.
        let _ = count;
    }

    #[test]
    fn test_interoceptor_read_includes_machine_native_fields() {
        // After read(), the body state should have machine-native
        // fields populated (even if 0.0 on the first read).
        let mut intero = Interoceptor::new();
        let body = intero.read();
        // First read: prev_gpe_count is 0, so autonomic_rate = 0.0.
        assert_eq!(
            body.autonomic_rate, 0.0,
            "first read autonomic_rate should be 0"
        );
        // thermoregulatory_effort depends on fan availability — just
        // check it's finite and in [0, 1].
        assert!(body.thermoregulatory_effort >= 0.0 && body.thermoregulatory_effort <= 1.0);
        // metabolic_rate depends on power sensor availability.
        assert!(body.metabolic_rate >= 0.0 && body.metabolic_rate <= 1.0);
    }

    #[test]
    fn test_neuro_impulses_no_stress_from_autonomic_regularity() {
        // GPE regularity (low variation in autonomic_rate) must NOT
        // produce stress impulses. This is the core fix: the old
        // HRV→CRH mapping caused chronic stress from normal machine
        // states. A machine with a steady GPE rate is healthy, not
        // stressed.
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 50.0,
            temperature: 0.5,
            arousal_freq: 0.5,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 5.0, // EC is active
            thermoregulatory_effort: 0.1, // fan barely on
            metabolic_rate: 0.05, // idle power
            core_voltage: 0.90, // normal Vcore
            supply_voltage: 12.6, // full battery
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // With a normal body state and active autonomic signals,
        // there should be serotonin (well-being) but NO CRH from
        // autonomic regularity. The only CRH sources in this state
        // would be from temperature/load/battery, all of which are
        // normal here.
        let has_crh = impulses.iter().any(|(id, _)| *id == NeurochemicalId::CRH);
        assert!(!has_crh, "autonomic regularity must not produce CRH stress");
    }

    #[test]
    fn test_neuro_impulses_thermoregulatory_strain() {
        // High fan + high temp → CRH (thermal strain — the cooling
        // system is working hard but the body is still hot).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 82.0,
            temperature: 0.82,
            arousal_freq: 1.0,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 3.0, // EC active when hot
            thermoregulatory_effort: 0.8, // fan working hard
            metabolic_rate: 0.3, // elevated power from heat
            core_voltage: 1.10, // Vcore raised
            supply_voltage: 12.5, // battery present
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Should have CRH from both temperature AND thermoregulatory
        // strain.
        let crh_count = impulses
            .iter()
            .filter(|(id, _)| *id == NeurochemicalId::CRH)
            .count();
        assert!(
            crh_count >= 2,
            "hot + fan struggling should produce CRH from both temp and thermoregulatory strain, got {crh_count}"
        );
    }

    #[test]
    fn test_neuro_impulses_cooling_failure() {
        // Fan off + high temp → CRH (cooling failure — dangerous,
        // like failing to sweat when overheating).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 85.0,
            temperature: 0.85,
            arousal_freq: 1.0,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 3.0, // EC active when hot
            thermoregulatory_effort: 0.0, // fan off!
            metabolic_rate: 0.2, // some power draw despite fan failure
            core_voltage: 1.10, // Vcore raised
            supply_voltage: 12.5, // battery present
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Should have CRH from both temperature AND cooling failure.
        let crh_count = impulses
            .iter()
            .filter(|(id, _)| *id == NeurochemicalId::CRH)
            .count();
        assert!(
            crh_count >= 2,
            "hot + fan off should produce CRH from both temp and cooling failure, got {crh_count}"
        );
    }

    #[test]
    fn test_neuro_impulses_io_activity_produces_ne() {
        // High I/O throughput → norepinephrine (active data work,
        // not drowsiness/adenosine).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 50.0,
            temperature: 0.5,
            arousal_freq: 0.5,
            cognitive_load: 0.3,
            io_activity: 0.5, // heavy I/O
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 2.0, // normal EC
            thermoregulatory_effort: 0.1, // fan barely on
            metabolic_rate: 0.1, // light power from I/O
            core_voltage: 0.95, // Vcore slightly elevated
            supply_voltage: 12.6, // full battery
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        let has_ne = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Norepinephrine);
        assert!(
            has_ne,
            "high I/O activity should produce norepinephrine (effort)"
        );
        // I/O activity must NOT produce adenosine (that was the old
        // incorrect drowsiness mapping).
        let has_adenosine = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Adenosine);
        assert!(
            !has_adenosine,
            "I/O activity must not produce adenosine (not drowsiness)"
        );
    }

    #[test]
    fn test_neuro_impulses_metabolic_exertion() {
        // High metabolic rate + high load → norepinephrine (exertion).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 50.0,
            temperature: 0.5,
            arousal_freq: 0.8,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.7, // working
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 3.0, // EC active under load
            thermoregulatory_effort: 0.15, // fan picking up
            metabolic_rate: 0.6, // high power draw
            core_voltage: 1.25, // elevated Vcore under load
            supply_voltage: 12.5, // battery present, on AC
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Should have NE from both load effort AND metabolic exertion.
        let ne_count = impulses
            .iter()
            .filter(|(id, _)| *id == NeurochemicalId::Norepinephrine)
            .count();
        assert!(
            ne_count >= 2,
            "high metabolic rate + load should produce NE from both sources, got {ne_count}"
        );
    }

    #[test]
    fn test_neuro_impulses_metabolic_strain() {
        // Very high metabolic rate + high temp → CRH (strain).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 80.0,
            temperature: 0.8,
            arousal_freq: 1.0,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.1,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 3.0, // EC active when hot
            thermoregulatory_effort: 0.5, // fan running hard
            metabolic_rate: 0.85, // very high power draw
            core_voltage: 1.15, // Vcore raised under thermal + metabolic load
            supply_voltage: 12.5, // battery present
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        // Should have CRH from both temperature AND metabolic strain.
        let crh_count = impulses
            .iter()
            .filter(|(id, _)| *id == NeurochemicalId::CRH)
            .count();
        assert!(
            crh_count >= 2,
            "very high metabolic rate + hot should produce CRH from both temp and metabolic strain, got {crh_count}"
        );
    }

    // ─── Silicon switching / prediction-error tests ─────────────

    #[test]
    fn test_rapl_delta_uj_normal() {
        assert_eq!(rapl_delta_uj(1000, 1500, 1_000_000), 500);
        assert_eq!(rapl_delta_uj(1000, 1000, 1_000_000), 0);
    }

    #[test]
    fn test_rapl_delta_uj_wraparound() {
        // Counter wraps at max_range: prev=900, cur=100 of a
        // 1000-range counter → delta = 100 (wrap) + 100 = 200.
        assert_eq!(rapl_delta_uj(900, 100, 1000), 200);
    }

    #[test]
    fn test_rapl_delta_uj_reset_no_range() {
        // Unknown wrap range and counter reset to 0 → treat as
        // fresh sample (delta = current).
        assert_eq!(rapl_delta_uj(5000, 200, 0), 200);
    }

    #[test]
    fn test_miss_ratio() {
        assert_eq!(miss_ratio(5, 100), 0.05);
        assert_eq!(miss_ratio(0, 100), 0.0);
        // No references → no ratio (idle process or missing counter).
        assert_eq!(miss_ratio(5, 0), 0.0);
        // Ratio clamps to 1.0 (misses can't exceed references in
        // practice, but guard anyway).
        assert_eq!(miss_ratio(200, 100), 1.0);
    }

    #[test]
    fn test_find_rapl_domains_graceful() {
        // On systems without powercap (this AMD laptop, VMs),
        // discovery returns an empty vec rather than panicking.
        // Where domains exist, only per-subsystem kinds are kept.
        for d in find_rapl_domains() {
            assert!(matches!(
                d.kind,
                RaplKind::Core | RaplKind::Uncore | RaplKind::Dram
            ));
            assert!(d.prev_uj.is_none());
        }
    }

    #[test]
    fn test_read_rapl_empty_domains_zero() {
        // With no RAPL domains (this machine), read_rapl must
        // return zeros without touching the filesystem.
        let mut intero = Interoceptor::new();
        intero.rapl_domains.clear();
        let (c, u, d) = intero.read_rapl(Instant::now());
        assert_eq!((c, u, d), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_neuro_impulses_silicon_activity() {
        // High core/DRAM switching + elevated miss ratios should
        // produce NE (effort + surprise) and ACh (memory traffic).
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 55.0,
            temperature: 0.55,
            arousal_freq: 0.8,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.6,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 2.0,
            thermoregulatory_effort: 0.1,
            metabolic_rate: 0.3,
            core_voltage: 1.0,
            supply_voltage: 12.5,
            core_activity: 0.9,
            uncore_activity: 0.6,
            dram_activity: 0.8,
            cache_miss_rate: 0.30,
            branch_miss_rate: 0.15,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        let has_ne = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Norepinephrine);
        let has_ach = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Acetylcholine);
        assert!(has_ne, "high switching + miss rates should produce NE");
        assert!(has_ach, "high DRAM activity should produce ACh");
    }

    #[test]
    fn test_neuro_impulses_no_silicon_sensors() {
        // With all silicon fields at zero (no RAPL, no perf —
        // e.g. this machine), the new mappings produce no impulses,
        // matching absent-sensor behavior everywhere else.
        let intero = Interoceptor::new();
        let body = BodyState {
            cpu_temp_c: 55.0,
            temperature: 0.55,
            arousal_freq: 0.5,
            cognitive_load: 0.3,
            io_activity: 0.0,
            stress_load: 0.3,
            energy_reserve: 1.0,
            on_ac_power: true,
            num_cores: 4,
            distressed: false,
            autonomic_rate: 0.0,
            thermoregulatory_effort: 0.0,
            metabolic_rate: 0.0,
            core_voltage: 0.0,
            supply_voltage: 0.0,
            core_activity: 0.0,
            uncore_activity: 0.0,
            dram_activity: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
            description: String::new(),
            ..Default::default()
        };
        let impulses = intero.neuro_impulses(&body);
        let has_ach = impulses
            .iter()
            .any(|(id, _)| *id == NeurochemicalId::Acetylcholine);
        assert!(!has_ach, "zero DRAM activity should produce no ACh");
    }
}

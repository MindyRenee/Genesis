//! Runtime manifest — what Genesis modules are loaded and what they're
//! doing right now.
//!
//! This is the "process table" for Genesis's internal architecture.
//! Each module (subcognitive, attention, memory, emotion, etc.) gets
//! an entry here with its current status, priority, and resource usage.

/// Maximum number of modules the manifest can track.
pub const MAX_MODULES: usize = 16;

/// Module identifiers.
///
/// These are stable enum values — never reorder or renumber. New
/// modules can be added up to 255.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleId {
    /// The subcognitive daemon — background processing.
    Subcognitive = 0,
    /// Attention / focus management.
    Attention = 1,
    /// Memory store management (STM, LTM, procedural, semantic).
    Memory = 2,
    /// Emotional regulation / neurochemical management.
    Emotion = 3,
    /// Language comprehension and generation.
    Language = 4,
    /// Reasoning and inference.
    Reasoning = 5,
    /// Sensory input processing.
    Sensory = 6,
    /// Motor / output action.
    Motor = 7,
    /// Metacognition — self-monitoring.
    Metacognition = 8,
    /// Dreaming / creative consolidation.
    Dreaming = 9,
    /// Goal / intention management.
    Intention = 10,
    /// Safety / ethics guardrails.
    Guardrails = 11,
}

impl ModuleId {
    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Subcognitive => "subcognitive",
            Self::Attention => "attention",
            Self::Memory => "memory",
            Self::Emotion => "emotion",
            Self::Language => "language",
            Self::Reasoning => "reasoning",
            Self::Sensory => "sensory",
            Self::Motor => "motor",
            Self::Metacognition => "metacognition",
            Self::Dreaming => "dreaming",
            Self::Intention => "intention",
            Self::Guardrails => "guardrails",
        }
    }

    /// Default priority for this module (0–255, higher = more important).
    pub const fn default_priority(self) -> u8 {
        match self {
            Self::Guardrails => 255, // safety always wins
            Self::Attention => 200,
            Self::Memory => 180,
            Self::Subcognitive => 100, // background, lower priority
            Self::Emotion => 150,
            Self::Language => 190,
            Self::Reasoning => 170,
            Self::Sensory => 160,
            Self::Motor => 160,
            Self::Metacognition => 120,
            Self::Dreaming => 50, // only when sleeping
            Self::Intention => 180,
        }
    }
}

/// Runtime status of a module.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleStatus {
    /// Module is not loaded.
    Stopped = 0,
    /// Module is in the process of starting.
    Starting = 1,
    /// Module is running and active.
    Running = 2,
    /// Module is loaded but idle (no active task).
    Idle = 3,
    /// Module is in the process of stopping.
    Stopping = 4,
    /// Module has crashed or is in an error state.
    Error = 5,
}

impl ModuleStatus {
    /// Returns a short lowercase label for this status.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Stopped => "stopped",
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Idle => "idle",
            Self::Stopping => "stopping",
            Self::Error => "error",
        }
    }

    /// Whether the module is considered "alive" (running or idle).
    pub const fn is_alive(self) -> bool {
        matches!(self, Self::Running | Self::Idle)
    }
}

/// A single module's runtime state.
///
/// ## Layout (32 bytes)
/// ```text
/// offset  field           type  notes
/// ------  -----           ----  -----
///   0     module_id       u8    ModuleId discriminant
///   1     status          u8    ModuleStatus discriminant
///   2     priority        u8    0–255, higher = more important
///   3     _pad            u8    alignment
///   4     error_code      u32   last error (0 = none)
///   8     task_id         u64   current task (0 if idle)
///  16     last_heartbeat  u64   ms timestamp of last heartbeat
///  24     cpu_share       f32   approximate CPU usage [0.0, 1.0]
///  28     mem_usage_mb    f32   memory usage in MB
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModuleEntry {
    /// `ModuleId` discriminant identifying this module.
    pub module_id: u8,
    /// `ModuleStatus` discriminant (current lifecycle state).
    pub status: u8,
    /// Scheduling priority (0–255, higher = more important).
    pub priority: u8,
    /// Alignment padding.
    pub _pad: u8,
    /// Last error code recorded by the module (0 = none).
    pub error_code: u32,
    /// Current task ID (0 if idle).
    pub task_id: u64,
    /// Milliseconds-timestamp of the last heartbeat.
    pub last_heartbeat: u64,
    /// Approximate CPU usage share, in `[0.0, 1.0]`.
    pub cpu_share: f32,
    /// Memory usage in megabytes.
    pub mem_usage_mb: f32,
}

impl ModuleEntry {
    /// Create a module entry in `Stopped` state with default priority.
    pub fn new(module: ModuleId) -> Self {
        Self {
            module_id: module as u8,
            status: ModuleStatus::Stopped as u8,
            priority: module.default_priority(),
            _pad: 0,
            error_code: 0,
            task_id: 0,
            last_heartbeat: 0,
            cpu_share: 0.0,
            mem_usage_mb: 0.0,
        }
    }

    /// Get the module ID as the enum type.
    pub fn id(&self) -> Option<ModuleId> {
        match self.module_id {
            0 => Some(ModuleId::Subcognitive),
            1 => Some(ModuleId::Attention),
            2 => Some(ModuleId::Memory),
            3 => Some(ModuleId::Emotion),
            4 => Some(ModuleId::Language),
            5 => Some(ModuleId::Reasoning),
            6 => Some(ModuleId::Sensory),
            7 => Some(ModuleId::Motor),
            8 => Some(ModuleId::Metacognition),
            9 => Some(ModuleId::Dreaming),
            10 => Some(ModuleId::Intention),
            11 => Some(ModuleId::Guardrails),
            _ => None,
        }
    }

    /// Get the status as the enum type.
    pub fn status(&self) -> ModuleStatus {
        match self.status {
            0 => ModuleStatus::Stopped,
            1 => ModuleStatus::Starting,
            2 => ModuleStatus::Running,
            3 => ModuleStatus::Idle,
            4 => ModuleStatus::Stopping,
            5 => ModuleStatus::Error,
            _ => ModuleStatus::Stopped,
        }
    }

    /// Record a heartbeat — module is alive at this time.
    pub fn heartbeat(&mut self, now_ms: u64) {
        self.last_heartbeat = now_ms;
    }
}

/// The runtime manifest — all module states plus aggregate metrics.
///
/// ## Layout (536 bytes)
/// ```text
/// offset  field            type           notes
/// ------  -----            ----           -----
///   0     modules          [ModuleEntry;16] 16×32 = 512 bytes
/// 512     uptime_ms        u64            process uptime in ms
/// 520     total_cpu_load   f32            aggregate CPU [0.0,1.0]
/// 524     total_mem_mb     f32            aggregate memory (MB)
/// 528     active_count     u8             modules in non-Stopped state
/// 529     _pad             [u8;7]         alignment to 8
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RuntimeManifest {
    /// Fixed array of module entries. Unused slots have `module_id = 0`
    /// and `status = Stopped`.
    pub modules: [ModuleEntry; MAX_MODULES],
    /// Process uptime in milliseconds.
    pub uptime_ms: u64,
    /// Aggregate CPU load across all modules [0.0, 1.0].
    pub total_cpu_load: f32,
    /// Aggregate memory usage across all modules (MB).
    pub total_mem_mb: f32,
    /// Number of modules currently in a non-Stopped state.
    pub active_count: u8,
    /// Alignment padding to 8 bytes.
    pub _pad: [u8; 7],
}

impl RuntimeManifest {
    /// Initialise with all known modules in `Stopped` state.
    pub fn new() -> Self {
        let mut modules = [ModuleEntry {
            module_id: 0,
            status: 0,
            priority: 0,
            _pad: 0,
            error_code: 0,
            task_id: 0,
            last_heartbeat: 0,
            cpu_share: 0.0,
            mem_usage_mb: 0.0,
        }; MAX_MODULES];

        let known = [
            ModuleId::Subcognitive,
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
        ];

        for (i, module) in known.into_iter().enumerate() {
            modules[i] = ModuleEntry::new(module);
        }

        Self {
            modules,
            uptime_ms: 0,
            total_cpu_load: 0.0,
            total_mem_mb: 0.0,
            active_count: 0,
            _pad: [0; 7],
        }
    }

    /// Get a module entry by ID, mutably.
    pub fn get_mut(&mut self, id: ModuleId) -> Option<&mut ModuleEntry> {
        let idx = id as usize;
        if idx < self.modules.len() && self.modules[idx].module_id == id as u8 {
            Some(&mut self.modules[idx])
        } else {
            None
        }
    }

    /// Get a module entry by raw module_id byte, mutably.
    /// Used by the IPC handler which receives u8 from the wire.
    pub fn get_mut_by_id(&mut self, module_id: u8) -> Option<&mut ModuleEntry> {
        let idx = module_id as usize;
        if idx < self.modules.len() && self.modules[idx].module_id == module_id {
            Some(&mut self.modules[idx])
        } else {
            None
        }
    }

    /// Get a module entry by ID.
    pub fn get(&self, id: ModuleId) -> Option<&ModuleEntry> {
        let idx = id as usize;
        if idx < self.modules.len() && self.modules[idx].module_id == id as u8 {
            Some(&self.modules[idx])
        } else {
            None
        }
    }

    /// Recompute aggregate metrics from individual module entries.
    pub fn recompute(&mut self) {
        let mut active = 0u8;
        let mut cpu = 0.0f32;
        let mut mem = 0.0f32;

        for m in &self.modules {
            if m.status().is_alive() {
                active += 1;
                cpu += m.cpu_share;
                mem += m.mem_usage_mb;
            }
        }

        self.active_count = active;
        self.total_cpu_load = cpu.min(1.0);
        self.total_mem_mb = mem;
    }
}

impl Default for RuntimeManifest {
    fn default() -> Self {
        Self::new()
    }
}

// --- Compile-time layout assertions ---
const _: () = {
    use core::mem::offset_of;
    // ModuleEntry (32 bytes)
    assert!(core::mem::size_of::<ModuleEntry>() == 32);
    assert!(offset_of!(ModuleEntry, module_id) == 0);
    assert!(offset_of!(ModuleEntry, status) == 1);
    assert!(offset_of!(ModuleEntry, priority) == 2);
    assert!(offset_of!(ModuleEntry, error_code) == 4);
    assert!(offset_of!(ModuleEntry, task_id) == 8);
    assert!(offset_of!(ModuleEntry, last_heartbeat) == 16);
    assert!(offset_of!(ModuleEntry, cpu_share) == 24);
    assert!(offset_of!(ModuleEntry, mem_usage_mb) == 28);

    // RuntimeManifest (536 bytes = 16×32 + 24)
    assert!(core::mem::size_of::<RuntimeManifest>() == MAX_MODULES * 32 + 24);
    assert!(offset_of!(RuntimeManifest, modules) == 0);
    assert!(offset_of!(RuntimeManifest, uptime_ms) == 512);
    assert!(offset_of!(RuntimeManifest, total_cpu_load) == 520);
    assert!(offset_of!(RuntimeManifest, total_mem_mb) == 524);
    assert!(offset_of!(RuntimeManifest, active_count) == 528);
};

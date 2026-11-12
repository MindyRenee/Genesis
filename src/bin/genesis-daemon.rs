//! Genesis subcognitive daemon — the main entry point.
//!
//! This is the process that runs 24/7 in the background, managing
//! Genesis's neurochemistry, memory consolidation, association, and
//! dreaming. It exposes an IPC server on a Unix domain socket for
//! the Python cognitive mind to communicate with.
//!
//! # Usage
//!
//! ```bash
//! genesis-daemon [--data-dir <dir>] [--socket <path>]
//! ```
//!
//! # What it does
//!
//! 1. Opens (or creates) the core state file, STM ring buffer, and
//!    LTM store in the data directory
//! 2. Starts the IPC server in a background thread
//! 3. Runs the autonomous fallback in the main thread: while the
//!    mind's lease is fresh it does staleness detection only; past
//!    the lease it drives neurochemistry, consolidation, dreaming,
//!    and sensing itself (never body-control application). A lost
//!    mind is not regrown — the body keeps living and waits; the
//!    organism's outside hand (the CLI under its supervisor)
//!    resurrects cognition, and the new mind adopts this surviving
//!    body over the socket it still holds
//! 4. On shutdown (SIGINT/SIGTERM or IPC shutdown command), syncs
//!    everything to disk and exits cleanly
//!
//! # Architecture
//!
//! The daemon is lease-driven, not purely reactive. The `TickLoop`
//! controller holds the daemon's long-lived subsystems and is shared
//! (behind a mutex) with the IPC handler, which dispatches the
//! mind-driven commands (ADVANCE_NEURO, CONSOLIDATE, ASSOCIATE,
//! DREAM, READ_SENSORS, APPLY_BODY_CONTROL) and renews the mind's
//! lease on each one. The main thread's autonomous fallback drives
//! the same `TickLoop` past the lease (`tick::MIND_LEASE_SECS`).
//!
//! The IPC server and the main thread share the same `MmapState`,
//! `RingBuffer`, and `LtmStore`. This is safe because:
//!
//! - The main thread only runs staleness detection (brief mmap writes)
//! - The IPC server runs in a background accept thread
//! - Each accepted client connection is served on its own thread, so a
//!   single slow request cannot block other clients
//! - Access to the LTM store is serialized via `Arc<Mutex<LtmStore>>`.
//!   The IPC handler holds the lock briefly during requests. The
//!   `LtmAccess` trait lets the library `IpcServer` work with the
//!   mutex-protected store without reimplementing the accept loop.
//!
//! Shutdown is coordinated via two `AtomicBool` flags:
//! - `shutdown_flag` (daemon-level): set by SIGINT/SIGTERM signal handlers
//! - `ipc_shutdown_flag` (IPC server): set by an IPC SHUTDOWN command
//!
//! A bridge thread forwards daemon→IPC, and the main thread checks both.

use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use genesis::daemon::{IpcServer, TickLoop, reactive_handler};
use genesis::store::{LtmStore, MmapState, RingBuffer};

#[derive(Debug)]
struct Config {
    data_dir: PathBuf,
    socket_path: PathBuf,
    stm_capacity: u32,
    ltm_capacity: u32,
}

impl Default for Config {
    fn default() -> Self {
        // The one canonical location — GENESIS_DATA_DIR or
        // ~/.local/share/genesis — the same answer genesis and every
        // Python tool get. A bare launch (no --data-dir) must land on
        // the same state tree they read.
        let data_dir = genesis::data_dir();
        Self {
            socket_path: data_dir.join("genesis.sock"),
            data_dir,
            stm_capacity: 256,
            ltm_capacity: 65_536,
        }
    }
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let mut socket_explicit = false;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                let dir = args.next().unwrap_or_else(|| {
                    eprintln!("--data-dir requires a value");
                    std::process::exit(1);
                });
                config.data_dir = PathBuf::from(dir);
                if !socket_explicit {
                    config.socket_path = config.data_dir.join("genesis.sock");
                }
            }
            "--socket" => {
                let path = args.next().unwrap_or_else(|| {
                    eprintln!("--socket requires a value");
                    std::process::exit(1);
                });
                config.socket_path = PathBuf::from(path);
                socket_explicit = true;
            }
            "--stm-capacity" => {
                let cap = args.next().unwrap_or_else(|| {
                    eprintln!("--stm-capacity requires a value");
                    std::process::exit(1);
                });
                let n: u32 = cap.parse().unwrap_or_else(|_| {
                    eprintln!("--stm-capacity requires a non-negative integer, got: {cap}");
                    std::process::exit(1);
                });
                if n == 0 {
                    eprintln!("--stm-capacity must be > 0");
                    std::process::exit(1);
                }
                config.stm_capacity = n;
            }
            "--ltm-capacity" => {
                let cap = args.next().unwrap_or_else(|| {
                    eprintln!("--ltm-capacity requires a value");
                    std::process::exit(1);
                });
                let n: u32 = cap.parse().unwrap_or_else(|_| {
                    eprintln!("--ltm-capacity requires a non-negative integer, got: {cap}");
                    std::process::exit(1);
                });
                if n == 0 {
                    eprintln!("--ltm-capacity must be > 0");
                    std::process::exit(1);
                }
                config.ltm_capacity = n;
            }
            "--help" | "-h" => {
                println!("Genesis subcognitive daemon");
                println!();
                println!("Usage: genesis-daemon [OPTIONS]");
                println!();
                println!("Options:");
                println!(
                    "  --data-dir <dir>       Data directory (default: resolved — see data_dir.rs)"
                );
                println!(
                    "  --socket <path>        Unix socket path (default: <data-dir>/genesis.sock)"
                );
                println!("  --stm-capacity <n>     STM ring buffer capacity (default: 256)");
                println!("  --ltm-capacity <n>     LTM episode capacity (default: 65536)");
                println!("  --help, -h             Show this help message");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {arg}");
                eprintln!("Use --help for usage information.");
                std::process::exit(1);
            }
        }
    }

    config
}

/// Decide what a `try_lock` refusal means in the fallback loop.
///
/// `WouldBlock` is the normal case: the mind is mid-drive and this
/// cycle should simply be skipped. `Poisoned` means some other thread
/// panicked while holding the lock, and it never clears — every
/// subsequent attempt fails the same way. Collapsing the two into
/// `continue` made a dead subsystem look like a busy one: the daemon
/// kept answering PING while neurochemistry, consolidation, dreaming
/// and zone arbitration had all silently stopped. Poisoning is
/// therefore reported, once per mutex, so it is visible without
/// flooding the log on every cycle.
fn lock_refusal<T>(what: &'static str, e: std::sync::TryLockError<T>) {
    if let std::sync::TryLockError::Poisoned(_) = e {
        static REPORTED: std::sync::OnceLock<Mutex<Vec<&'static str>>> = std::sync::OnceLock::new();
        let reported = REPORTED.get_or_init(|| Mutex::new(Vec::new()));
        let mut guard = match reported.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !guard.contains(&what) {
            guard.push(what);
            eprintln!(
                "[genesis] {what} mutex is poisoned — a thread panicked while holding it. \
                 The autonomous fallback cannot run this subsystem and will keep skipping it \
                 until the daemon restarts. The IPC path will refuse it too."
            );
        }
    }
    // Both outcomes mean "skip this cycle"; only the reporting differs.
}

fn main() {
    let config = parse_args();

    // Snapshot the host's CPU policy before Genesis applies anything, so
    // it can be restored on shutdown. Read-only and idempotent — this
    // never affects its running state.
    genesis::daemon::cpufreq::capture_hardware_state();

    // Create the data directory if it doesn't exist
    if !config.data_dir.exists()
        && let Err(e) = std::fs::create_dir_all(&config.data_dir)
    {
        eprintln!(
            "[genesis] FATAL: failed to create data directory {}: {e}",
            config.data_dir.display()
        );
        std::process::exit(1);
    }

    // Acquire an exclusive advisory lock on a per-data-dir lock file.
    // This prevents two daemons from running against the same data
    // directory, which would corrupt the mmap'd state files.
    let lock_path = config.data_dir.join("genesis.lock");
    let _lock_file = match std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "[genesis] FATAL: failed to open daemon lock file {}: {e}",
                lock_path.display()
            );
            std::process::exit(1);
        }
    };
    // SAFETY: `_lock_file` is a valid open file descriptor obtained from
    // `OpenOptions` above. `as_raw_fd()` returns the underlying fd which
    // remains valid as long as `_lock_file` is alive (it is held for the
    // entire process lifetime). `flock` with `LOCK_EX | LOCK_NB` is a
    // non-blocking advisory lock request — it returns immediately and
    // does not block.
    let rc = unsafe { libc::flock(_lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        eprintln!(
            "[genesis] Another daemon is already running for data dir: {}",
            config.data_dir.display()
        );
        std::process::exit(1);
    }

    let state_path = config.data_dir.join("core_state.bin");
    let stm_path = config.data_dir.join("stm_ring.bin");
    let ltm_base = config.data_dir.join("ltm_store");

    // Open or create the stores
    eprintln!("[genesis] Initializing subcognitive...");
    eprintln!("[genesis]   data dir:    {}", config.data_dir.display());
    eprintln!("[genesis]   state file:  {}", state_path.display());
    eprintln!("[genesis]   stm file:    {}", stm_path.display());
    eprintln!(
        "[genesis]   ltm base:    {}.{{bundles,meta,dat}}",
        ltm_base.display()
    );
    eprintln!("[genesis]   socket:      {}", config.socket_path.display());

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mmap = match MmapState::open(&state_path) {
        Ok(m) => m,
        Err(genesis::store::StateFileError::NotFound) => {
            match MmapState::create(&state_path, now_ms, now_ms) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!(
                        "[genesis] FATAL: failed to create core state {}: {e}",
                        state_path.display()
                    );
                    std::process::exit(1);
                }
            }
        }
        // An unreadable state file (truncated, foreign, or checksum-
        // corrupt) would otherwise FATAL-loop forever: every restart
        // hits the same verification failure. Rename it aside — it is
        // preserved for forensics, never silently destroyed — and
        // rebuild a fresh state. I/O, lock, and mmap failures are
        // environmental and still fatal: recreating cannot fix them.
        Err(
            e @ (genesis::store::StateFileError::VerificationFailed(_)
            | genesis::store::StateFileError::FileTooSmall { .. }),
        ) => {
            let quarantine = state_path.with_extension(format!("corrupt.{now_ms}"));
            eprintln!(
                "[genesis] core state {} is unrecoverable ({e}); archiving to {}",
                state_path.display(),
                quarantine.display()
            );
            if let Err(rename_err) = std::fs::rename(&state_path, &quarantine) {
                eprintln!(
                    "[genesis] FATAL: could not archive unrecoverable state {}: {rename_err}",
                    state_path.display()
                );
                std::process::exit(1);
            }
            match MmapState::create(&state_path, now_ms, now_ms) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!(
                        "[genesis] FATAL: failed to rebuild core state {}: {e}",
                        state_path.display()
                    );
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!(
                "[genesis] FATAL: failed to open core state {}: {e}",
                state_path.display()
            );
            std::process::exit(1);
        }
    };
    let stm = match RingBuffer::open_or_create(&stm_path, config.stm_capacity) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "[genesis] FATAL: failed to open STM ring buffer {}: {e}",
                stm_path.display()
            );
            std::process::exit(1);
        }
    };
    let ltm = match LtmStore::open_or_create(&ltm_base, config.ltm_capacity) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "[genesis] FATAL: failed to open LTM store {}: {e}",
                ltm_base.display()
            );
            std::process::exit(1);
        }
    };

    // Continuity sentinel — record this boot and learn how the
    // previous session ended. `dark_ms` comes from the state file's
    // own last-write timestamp, measured by `MmapState::open` before
    // it was overwritten; the sentinel supplies the *manner* of the
    // stop. The mind reads the same file on wake, so an unclean stop
    // — power loss, kill, crash — is something she knows happened,
    // not a seam she trips over.
    let boot = genesis::daemon::continuity::note_boot(
        &config.data_dir,
        !mmap.was_created(),
        mmap.offline_gap_ms(),
        now_ms,
    );
    if boot.had_previous && boot.prev_unclean {
        let bound = if boot.dark_upper_bound {
            "at least "
        } else {
            ""
        };
        eprintln!(
            "[genesis] continuity: previous session left no clean stop \
             (power loss, kill, or crash) — dark for {}{}",
            bound,
            genesis::daemon::continuity::fmt_dark(boot.dark_ms),
        );
    } else if boot.had_previous {
        eprintln!(
            "[genesis] continuity: previous session stopped cleanly — \
             dark for {}",
            genesis::daemon::continuity::fmt_dark(boot.dark_ms),
        );
    }

    eprintln!("[genesis]   stm entries: {}", stm.count());
    eprintln!("[genesis]   ltm episodes: {}", ltm.count());
    eprintln!("[genesis] Subcognitive initialized.");
    eprintln!(
        "[genesis]   provenance: 0x{:012x}",
        genesis::field::provenance_fingerprint()
    );

    // ── Wake recovery ──
    // If the daemon opened an existing state file (restart, not first
    // run), depleted neurochemicals from the previous session persist
    // and the slow homeostatic recovery keeps it "guarded" for
    // minutes. Give depleted neurotransmitters a one-time boost toward
    // baseline when cortisol is low (no ongoing stress). This is
    // biologically grounded: after sleep, the brain rapidly restores
    // depleted neurochemicals. A daemon restart is analogous to
    // waking up.
    if !mmap.was_created() {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if let Err(e) = mmap.modify(now_ms, |state| {
            state.neurochemicals.wake_recovery();
        }) {
            eprintln!("[genesis] WARNING: wake recovery failed: {e}");
        }
    }

    // Wrap the LTM in a mutex for shared access between IPC handlers.
    let ltm = Arc::new(Mutex::new(ltm));
    let stm = Arc::new(stm);
    let mmap = Arc::new(mmap);

    // Physical field probe — the body listening to its own substrate.
    // A dedicated thread measures real substrate dynamics
    // (instruction throughput, scheduler jitter, P-state transitions,
    // interrupt flux, and real power draw where sensors report it)
    // at ~250 Hz and decomposes them at the wave bands' center
    // frequencies; the mind reads the signature over
    // GET_FIELD_PROBE and injects it into the wave oscillator as
    // outer stimulus. The probe also writes a carrier — its work is
    // modulated at the body's own phase rhythm, so the die's real
    // current carries a component at the organism's beat (measured
    // back through the power channel, not assumed).
    {
        use genesis::state::MentalPhase;
        let probe_mmap = mmap.clone();
        genesis::daemon::field_probe::start(move || {
            match probe_mmap.read_consistent().map(|s| s.zones.phase()) {
                Some(MentalPhase::NREM) => "nrem",
                Some(MentalPhase::REM) => "rem",
                Some(MentalPhase::Drowsy) => "drowsy",
                _ => "wake",
            }
        });
    }

    // The eye — the optic nerve's retinal end. The daemon captures
    // the camera itself (no separate process): the limb opens the
    // device on its own thread, publishes frames into the nerve
    // segment, re-opens on a bounded backoff, and accepts absence —
    // sight is a degradation, never a fault. `stop` runs in the
    // shutdown sequence below; `Drop` covers every other exit path.
    let eye = genesis::daemon::optic_nerve::OpticCapture::start();

    // Set up shutdown coordination
    let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Set up signal handlers (SIGINT, SIGTERM)
    let watcher_thread = setup_signal_handlers(&shutdown_flag);

    // Create the TickLoop, shared with both the IPC handler (mind
    // lease) and the main-thread autonomous fallback (past the
    // lease). The mind drives while alive; the daemon drives itself
    // when silent. Body-control application stays mind-requested.
    let tick_loop = Arc::new(Mutex::new(TickLoop::new_with_data_dir(&config.data_dir)));

    // Create the IPC server (library version — no duplicated accept loop)
    let ipc_server = IpcServer::new(&config.socket_path);
    let ipc_shutdown_flag = ipc_server.shutdown_flag();

    // Link the daemon's shutdown flag to the IPC server's shutdown flag.
    let bridge_thread = {
        let daemon_flag = shutdown_flag.clone();
        let ipc_flag = ipc_shutdown_flag.clone();
        std::thread::spawn(move || {
            loop {
                if daemon_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    ipc_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        })
    };

    // Start the IPC server in a background thread with the reactive
    // handler. The handler wraps the TickLoop and dispatches reactive
    // commands to it, falling through to default_handler for
    // everything else.
    // The autonomous fallback's own gears, published for
    // GET_INTERNALS. The main loop below writes one snapshot per
    // iteration; the IPC handler reads it. Everything inside is a
    // plain copy, so a reader can never stall the fallback by
    // holding the lock.
    let auto_internals = Arc::new(Mutex::new(
        genesis::daemon::adaptive::AutoSnapshot::default(),
    ));

    let ipc_mmap = mmap.clone();
    let ipc_stm = stm.clone();
    let ipc_ltm = ltm.clone();
    let handler = reactive_handler(
        tick_loop.clone(),
        config.data_dir.clone(),
        auto_internals.clone(),
    );

    let ipc_thread = match std::thread::Builder::new()
        .name("genesis-ipc".to_string())
        .spawn(move || {
            ipc_server.run(ipc_mmap, ipc_stm, ipc_ltm, handler);
        }) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[genesis] FATAL: failed to spawn IPC thread: {e}");
            std::process::exit(1);
        }
    };

    // IpcServer reports readiness only after bind succeeds. Do not claim
    // the endpoint is listening from the spawning thread.
    eprintln!("[genesis] Lease mode — mind-driven when alive, autonomous fallback when silent.");
    eprintln!(
        "[genesis] Mind holds a 3 s lease via reactive IPC; past it the daemon drives itself."
    );

    // The main thread is the autonomous fallback. While the mind
    // drives through IPC (lease fresh), this loop does nothing but
    // staleness detection. Once the lease expires — stall, crash,
    // disconnect, or simply not yet connected — it drives physiology
    // itself (neurochemistry, consolidation, dreaming) so the system
    // keeps living instead of freezing. The mind modulates; absence
    // degrades to solitude, not to paused existence.
    //
    // Lock discipline: the IPC path locks LTM first, then the
    // TickLoop. The neuro advance below needs only the TickLoop (no
    // LTM), so it cannot deadlock. The memory phase below acquires
    // LTM first, then the TickLoop — the same order as IPC. Every
    // lock here is `try_lock`: the fallback never blocks a
    // mind-driven request; it skips the cycle instead.
    //
    // Deliberately excluded: body-control *application* (sysfs,
    // renice, EPP/boost). The fallback publishes the recommendation
    // and senses the body, but acting on the shared body stays a
    // mind-requested IPC, preserving cognition as the author of
    // outward action. Everything else the integrated tick does —
    // zone arbitration included — runs here, so no TickLoop method
    // only lives in the test-only reference path.
    // Adaptive wall-clock gates, not fixed Instant periods: each gear
    // spins at the rate its own signal demands, bounded so nothing
    // stalls or redlines (see src/daemon/adaptive.rs). Nominal
    // periods match the old constants; [min, max] is the operating
    // range the signal may move within.
    use genesis::daemon::AdaptiveGate;
    let mut staleness_gate = AdaptiveGate::new(60_000, 10_000, 300_000);
    let mut assoc_gate = AdaptiveGate::new(10_000, 3_000, 60_000);
    let mut dream_gate = AdaptiveGate::new(4_000, 2_000, 30_000);
    let mut sensor_gate = AdaptiveGate::new(5_000, 2_000, 15_000);
    let mut sync_gate = AdaptiveGate::new(20_000, 5_000, 100_000);
    let mut body_gate = AdaptiveGate::new(10_000, 3_000, 30_000);
    // Zone arbitration nominally every few seconds past the lease —
    // intentions evolve slowly, and each pass writes a manifest
    // transaction plus a possible LTM trace.
    let mut zone_gate = AdaptiveGate::new(5_000, 2_000, 30_000);
    let mut model_save_gate = AdaptiveGate::new(60_000, 20_000, 300_000);
    let mut last_model_save_ticks = 0u32;
    // The fallback's own axle is adaptive too: deep NREM idles at
    // ~1 Hz, REM at 2 Hz, and anything else holds the 5 Hz reference
    // cadence the dynamics were tuned at.
    let mut axle_ms: u64 = genesis::daemon::tick::AUTONOMOUS_INTERVAL_MS;
    while !shutdown_flag.load(std::sync::atomic::Ordering::Relaxed)
        && !ipc_shutdown_flag.load(std::sync::atomic::Ordering::Relaxed)
    {
        std::thread::sleep(Duration::from_millis(axle_ms));
        let now_ms = genesis::daemon::tick::current_ms();

        // One read feeds the axle and the phase-driven gates below.
        let phase = mmap.read_consistent().map(|s| s.zones.phase());
        axle_ms = match phase {
            Some(genesis::state::MentalPhase::NREM) => 1_000,
            Some(genesis::state::MentalPhase::REM) => 500,
            _ => genesis::daemon::tick::AUTONOMOUS_INTERVAL_MS,
        };

        // Publish the fallback's own gears for GET_INTERNALS — one
        // snapshot per iteration, before any gate fires this pass.
        // Placed ahead of every `continue` below so the view stays
        // current whether or not the lease is held.
        if let Ok(mut snap) = auto_internals.lock() {
            snap.gates.clear();
            snap.gates.extend_from_slice(&[
                ("staleness", staleness_gate.snapshot()),
                ("assoc", assoc_gate.snapshot()),
                ("dream", dream_gate.snapshot()),
                ("sensor", sensor_gate.snapshot()),
                ("sync", sync_gate.snapshot()),
                ("body", body_gate.snapshot()),
                ("zone", zone_gate.snapshot()),
                ("model_save", model_save_gate.snapshot()),
            ]);
            snap.axle_ms = axle_ms;
            snap.last_model_save_ticks = last_model_save_ticks;
        }

        // Staleness detection runs regardless of lease — a crashed
        // mind's modules must read Stopped even while the fallback
        // keeps the body alive. A live mind warrants close watch; an
        // absent one drifts the check toward the ceiling.
        if staleness_gate.due(now_ms) {
            match tick_loop.try_lock() {
                Ok(mut tl) => {
                    tl.check_staleness(&mmap);
                    if tl.mind_lease_expired() {
                        staleness_gate.relax();
                    } else {
                        staleness_gate.demand();
                    }
                }
                Err(e) => {
                    lock_refusal("tick_loop", e);
                }
            }
        }

        // Lease check without holding the lock across work: peek,
        // then release. A mind driving at ~1 Hz keeps idle ~1 s;
        // the fallback fires only past 3 s of silence.
        let lease_expired = match tick_loop.try_lock() {
            Ok(tl) => tl.mind_lease_expired(),
            Err(e) => {
                // Busy (mind driving) skips this cycle; poisoned is
                // reported once and then skips it forever.
                lock_refusal("tick_loop", e);
                continue;
            }
        };
        if !lease_expired {
            continue;
        }

        // ── Autonomous neurochemistry (TickLoop only, no LTM) ──
        // dt tracks real time since the last advance from ANY source
        // (mind or fallback), capped like the IPC path, so catch-up
        // after a stall never overshoots the integrator.
        {
            let mut tl = match tick_loop.try_lock() {
                Ok(tl) => tl,
                Err(e) => {
                    lock_refusal("tick_loop", e);
                    continue;
                }
            };
            // Re-check under the lock: the mind may have driven
            // between the peek above and now.
            if !tl.mind_lease_expired() {
                continue;
            }
            // No dt here either: the body measures its own elapsed time.
            tl.advance_neuro(&mmap);
        }

        // ── Autonomous memory + sensing (LTM first, then TickLoop:
        // same order as the IPC path, both try_lock) ──
        let mut ltm_guard = match ltm.try_lock() {
            Ok(g) => g,
            Err(e) => {
                // IPC owns LTM (physiology already advanced above) —
                // or it is poisoned, which never clears.
                lock_refusal("ltm", e);
                continue;
            }
        };
        let mut tl = match tick_loop.try_lock() {
            Ok(tl) => tl,
            Err(e) => {
                // Mind driving — the LTM guard drops by scope end.
                lock_refusal("tick_loop", e);
                continue;
            }
        };
        // The mind may have returned while we acquired locks.
        if !tl.mind_lease_expired() {
            continue;
        }
        // LTMStore derefs through the MutexGuard; the granular
        // TickLoop methods take `&mut LtmStore`.
        let ltm_store: &mut LtmStore = &mut ltm_guard;
        // Consolidation is self-gating (plasticity, overwhelm, empty
        // STM) — call whenever autonomous and let it decide.
        tl.consolidate(&mmap, &stm, ltm_store);
        if assoc_gate.due(now_ms) {
            if tl.associate(&mmap, ltm_store) > 0 {
                assoc_gate.demand();
            } else {
                assoc_gate.relax();
            }
        }
        // Dream only when sleeping — the body's own phase is the gate,
        // so waking autonomous never dreams. REM snaps the gate fast;
        // NREM drifts it sparse; waking relaxes it toward its ceiling.
        if dream_gate.due(now_ms) {
            use genesis::state::MentalPhase;
            let sleeping = phase.is_some_and(|p| p == MentalPhase::NREM || p == MentalPhase::REM);
            if sleeping {
                tl.dream(&mmap, ltm_store);
                if phase == Some(MentalPhase::REM) {
                    dream_gate.demand();
                } else {
                    dream_gate.relax();
                }
            } else {
                dream_gate.relax();
            }
        }
        if sensor_gate.due(now_ms) {
            let body = tl.read_sensors(&mmap);
            // The brainstem-level interoceptive reflex. With the
            // mind's lease expired there is no cortex on the wire
            // to perceive the body, so the body applies its own
            // afferent spec directly — the NTS/parabrachial →
            // autonomic loop that runs below cortex in every animal
            // (lamina I → catecholamine cell groups → hypothalamic
            // neuroendocrine; PBN → preoptic drive). When the lease
            // IS held this block never runs: the same BodyState is
            // published for the mind's own brainstem relay, which
            // routes it through the regions — the daemon never
            // double-drives a response the mind is authoring.
            if let Err(e) = mmap.modify(now_ms, |state| {
                for (id, mag) in tl.interoceptor.neuro_impulses(&body, &state.neurochemicals) {
                    state.neurochemicals.apply_impulse_capped(id, mag, now_ms);
                }
            }) {
                eprintln!("[genesis] WARNING: interoceptive reflex failed: {e}");
            }
            // Strain on the just-read body shapes the next poll:
            // a pressured substrate is watched closely.
            sensor_gate.drive(
                body.stress_load
                    .max(body.psi_cpu)
                    .max(body.psi_mem)
                    .max(body.throttle_state),
            );
        }
        // Intention expiry + zone arbitration + trace. Sleep drifts
        // the cadence — few zone decisions arrive while dreaming;
        // wake keeps it near the nominal few seconds. Skipped while
        // the mind holds the lease by the outer check.
        zone_gate.drive(match phase {
            Some(genesis::state::MentalPhase::NREM) => 0.0,
            Some(genesis::state::MentalPhase::REM) => 0.3,
            _ => 1.0,
        });
        if zone_gate.due(now_ms) {
            tl.arbitrate_zone(&mmap, ltm_store);
        }
        // Guards drop here (LTM, then TickLoop).
        drop(tl);
        drop(ltm_guard);

        // Body-control recommendation publish (compute + publish
        // only — never application). Needs the TickLoop alone.
        if body_gate.due(now_ms) {
            match tick_loop.try_lock() {
                Ok(mut tl) if tl.mind_lease_expired() => {
                    tl.publish_body_recommendation(&mmap);
                    // Thermal/throttle pressure on the last-read body
                    // revisits policy sooner; a cool machine drifts.
                    let b = &tl.last_body_state;
                    body_gate.drive(
                        b.throttle_state
                            .max(b.thermoregulatory_effort)
                            .max(b.psi_cpu),
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    lock_refusal("tick_loop", e);
                }
            }
        }

        // Generative-model checkpoint so autonomous learning is
        // durable. The mind saves every 60 s while alive; this covers
        // mindless stretches. Shutdown always saves regardless.
        // Demand follows learning: new inference ticks mean new model
        // state worth persisting; an unchanged engine relaxes.
        if model_save_gate.due(now_ms) {
            match tick_loop.try_lock() {
                Ok(tl) if tl.mind_lease_expired() => {
                    if tl.inference_engine.tick_count() != last_model_save_ticks {
                        if !tl.save_inference_model(&config.data_dir) {
                            eprintln!("[genesis] WARNING: autonomous inference save failed");
                        }
                        last_model_save_ticks = tl.inference_engine.tick_count();
                        model_save_gate.demand();
                    } else {
                        model_save_gate.relax();
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    lock_refusal("tick_loop", e);
                }
            }
        }

        // Periodic non-blocking flush so autonomous life is durable.
        // Shutdown and explicit SYNC still use the blocking path.
        // Plasticity drives the period — same continuous form as the
        // integrated tick's sync gate.
        let plasticity = mmap
            .read_consistent()
            .map(|s| s.memory.plasticity_gate)
            .unwrap_or(0.5);
        sync_gate.drive(plasticity);
        if sync_gate.due(now_ms) {
            let _ = mmap.sync_async();
        }
    }

    // The wait can end via either flag — signals set `shutdown_flag`,
    // an IPC SHUTDOWN command sets `ipc_shutdown_flag`. Normalize so
    // the auxiliary threads (which watch `shutdown_flag`) exit too;
    // otherwise an IPC-initiated shutdown would hang the joins below
    // before the final state sync.
    shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);

    // Shutdown sequence
    eprintln!("[genesis] Shutting down...");

    // Close the eye first so no reader attaches to a dead nerve — the
    // limb's thread is reaped by process exit if it is still blocked
    // in the driver's dequeue.
    eye.stop();

    // Restore the host's CPU policy captured at startup. This runs
    // before the sync block below (which can bail out on a poisoned
    // LTM mutex), so the machine is never left pinned to a throttled
    // governor. It runs only after it has stopped — it cannot affect
    // its running state.
    if genesis::daemon::cpufreq::restore_hardware_state() {
        eprintln!("[genesis] Host CPU policy restored.");
    } else if genesis::daemon::cpufreq::policy_was_applied() {
        // The alarming case, and the only one worth shouting about: we
        // did change the host and could not put it back, so the machine
        // outlives us pinned to a throttle. Reported unconditionally
        // rather than through the deduplicating helper so it survives
        // an earlier same-cause log.
        eprintln!(
            "[genesis] WARNING: host CPU policy was NOT restored — this machine may \
             be left at the throttled governor Genesis applied. Check that \
             passwordless sudo still works (`sudo -n true`)."
        );
    } else {
        // Nothing was ever applied, so there is nothing to restore. This
        // is the ordinary case on a machine where the sudo grant was
        // never installed, and warning about a throttle we never applied
        // would be a false alarm on every shutdown.
        eprintln!("[genesis] Host CPU policy unchanged — nothing was applied, nothing to restore.");
    }

    // Wait for the IPC accept thread to finish gracefully.
    // The IPC server checks the shutdown flag on each accept poll.
    // Joining ensures the accept loop has stopped so no new client
    // connections will be accepted before we sync to disk.
    if let Err(e) = ipc_thread.join() {
        eprintln!("[genesis] IPC thread panicked: {e:?}");
    }

    // Join the watcher and bridge threads. Both poll every 50ms and
    // exit as soon as they see their respective shutdown flags, which
    // are already set by this point. They don't touch any stores —
    // joining is for clean thread accounting, not data safety.
    if let Err(e) = bridge_thread.join() {
        eprintln!("[genesis] bridge thread panicked: {e:?}");
    }
    if let Err(e) = watcher_thread.join() {
        eprintln!("[genesis] watcher thread panicked: {e:?}");
    }

    // The IPC thread has now joined every client handler it spawned.
    // No connected request can mutate mmap/STM/LTM after this point.

    // Final sync — flush all mmap'd stores to disk. A poisoned LTM
    // lock means a thread panicked while holding it; the mmap state
    // may be inconsistent. Exit with an error rather than continuing
    // and potentially writing corrupt data to disk.
    let mut clean_shutdown = true;
    {
        let mut ltm_guard = match ltm.lock() {
            Ok(g) => g,
            Err(_) => {
                eprintln!(
                    "[genesis] FATAL: LTM mutex poisoned during shutdown — \
                     a thread panicked while holding the lock. State may be \
                     inconsistent; refusing to sync."
                );
                std::process::exit(1);
            }
        };
        if let Err(e) = ltm_guard.sync() {
            eprintln!("[genesis] WARNING: final ltm sync failed: {e}");
            clean_shutdown = false;
        }
    }
    if let Err(e) = stm.sync() {
        eprintln!("[genesis] WARNING: final stm sync failed: {e}");
        clean_shutdown = false;
    }
    if let Err(e) = mmap.sync() {
        eprintln!("[genesis] WARNING: final mmap sync failed: {e}");
        clean_shutdown = false;
    }

    // Save the active inference model so the generative self-model
    // persists across restarts. Genesis doesn't re-learn its own
    // neurochemical dynamics from scratch every time it wakes up.
    match tick_loop.lock() {
        Ok(tl) => clean_shutdown &= tl.save_inference_model(&config.data_dir),
        Err(e) => {
            eprintln!("[genesis] WARNING: tick loop poisoned during shutdown: {e}");
            clean_shutdown = false;
        }
    }

    // Mark the session's clean stop — the stop itself was deliberate
    // regardless of whether every sync succeeded. Any exit that never
    // reaches here leaves the sentinel open, and the next boot reads
    // it as the unclean stop it was.
    genesis::daemon::continuity::note_clean_stop(
        &config.data_dir,
        genesis::daemon::tick::current_ms(),
    );

    if !clean_shutdown {
        eprintln!("[genesis] Shutdown incomplete — persistent state may be stale.");
        std::process::exit(1);
    }
    eprintln!("[genesis] Subcognitive stopped. Goodnight.");
}

/// Set up SIGINT and SIGTERM handlers for graceful shutdown.
///
/// Uses `sigaction(2)` instead of `signal(2)` for portability and
/// reliability: `sigaction` does not reset the handler to `SIG_DFL`
/// after delivery (which `signal` may do on some systems), and allows
/// masking other signals during handler execution.
///
/// The handler (`handle_signal`) is async-signal-safe: it only performs
/// an atomic store to a static `AtomicBool` with `Ordering::Relaxed`,
/// which is async-signal-safe per POSIX. No locks, no allocation, no
/// I/O inside the handler.
fn setup_signal_handlers(
    shutdown_flag: &Arc<std::sync::atomic::AtomicBool>,
) -> std::thread::JoinHandle<()> {
    use std::sync::atomic::Ordering;

    // SAFETY: `sigaction(2)` installs a handler for a signal. The handler
    // (`handle_signal`) is async-signal-safe: it only performs an atomic
    // store to a static `AtomicBool` with `Ordering::Relaxed`. No locks,
    // no allocation, no I/O inside the handler. The `sigaction` struct is
    // zero-initialised and then populated with the handler function.
    unsafe {
        extern "C" fn handle_signal(_sig: i32) {
            SHUTDOWN_REQUESTED.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handle_signal as *const () as usize;
        sa.sa_flags = 0;
        libc::sigemptyset(&mut sa.sa_mask);

        // SIGINT (2) and SIGTERM (15)
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }

    // Watcher thread: polls the static and forwards to the caller's flag
    let flag = shutdown_flag.clone();
    std::thread::spawn(move || {
        loop {
            if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                eprintln!("\n[genesis] Received shutdown signal, stopping...");
                flag.store(true, Ordering::Relaxed);
                break;
            }
            // Shutdown can also be initiated without a signal (the IPC
            // SHUTDOWN command sets the IPC server's flag and the main
            // thread then sets this one). Exit in that case too so the
            // main thread's join doesn't hang.
            if flag.load(Ordering::Relaxed) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })
}

// Static atomic for signal handler communication
use std::sync::atomic::AtomicBool;
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

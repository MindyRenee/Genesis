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
//! 3. Waits in the main thread, performing staleness detection and
//!    housekeeping; the cognitive mind drives neurochemistry, memory,
//!    and body control through reactive IPC commands
//! 4. On shutdown (SIGINT/SIGTERM or IPC shutdown command), syncs
//!    everything to disk and exits cleanly
//!
//! # Architecture
//!
//! The daemon is reactive — it does not run a fixed tick loop. The
//! `TickLoop` controller holds the daemon's long-lived subsystems and
//! is shared (behind a mutex) with the IPC handler, which dispatches
//! the mind-driven commands (ADVANCE_NEURO, CONSOLIDATE, ASSOCIATE,
//! DREAM, READ_SENSORS, APPLY_BODY_CONTROL) to it.
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
        Self {
            data_dir: PathBuf::from("./genesis_data"),
            socket_path: PathBuf::from("./genesis_data/genesis.sock"),
            stm_capacity: 256,
            ltm_capacity: 65_536,
        }
    }
}

fn parse_args() -> Config {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                let dir = args.next().unwrap_or_else(|| {
                    eprintln!("--data-dir requires a value");
                    std::process::exit(1);
                });
                config.data_dir = PathBuf::from(dir);
                config.socket_path = config.data_dir.join("genesis.sock");
            }
            "--socket" => {
                let path = args.next().unwrap_or_else(|| {
                    eprintln!("--socket requires a value");
                    std::process::exit(1);
                });
                config.socket_path = PathBuf::from(path);
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
                println!("  --data-dir <dir>       Data directory (default: ./genesis_data)");
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

fn main() {
    let config = parse_args();

    // Snapshot the host's CPU policy before Genesis applies anything, so
    // it can be restored on shutdown. Read-only and idempotent — this
    // never affects her running state.
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
    let mmap = match MmapState::open_or_create(&state_path, 1, now_ms) {
        Ok(m) => m,
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

    eprintln!("[genesis]   stm entries: {}", stm.count());
    eprintln!("[genesis]   ltm episodes: {}", ltm.count());
    eprintln!("[genesis] Subcognitive initialized.");

    // ── Wake recovery ──
    // If the daemon opened an existing state file (restart, not first
    // run), depleted neurochemicals from the previous session persist
    // and the slow homeostatic recovery keeps her "guarded" for
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

    // Set up shutdown coordination
    let shutdown_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Set up signal handlers (SIGINT, SIGTERM)
    let watcher_thread = setup_signal_handlers(&shutdown_flag);

    // Create the TickLoop — but it no longer runs on a fixed schedule.
    // It's wrapped in Arc<Mutex> and shared with the IPC handler.
    // The cognitive mind drives each function through IPC commands
    // (ADVANCE_NEURO, CONSOLIDATE, ASSOCIATE, DREAM, READ_SENSORS,
    // APPLY_BODY_CONTROL). The daemon is a bus — it carries
    // information and executes requests, but never initiates actions.
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
    let ipc_mmap = mmap.clone();
    let ipc_stm = stm.clone();
    let ipc_ltm = ltm.clone();
    let handler = reactive_handler(tick_loop.clone(), config.data_dir.clone());

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

    eprintln!(
        "[genesis] IPC server listening on {}",
        config.socket_path.display()
    );
    eprintln!("[genesis] Reactive mode — mind-driven, no tick loop.");
    eprintln!("[genesis] All functions idle until the mind activates them.");

    // The main thread waits for shutdown and runs periodic
    // housekeeping that the reactive IPC handlers don't cover.
    // The cognitive mind drives neurochemistry, memory, and body
    // control through IPC commands; the daemon handles staleness
    // detection — if the cognitive mind crashes, its modules'
    // heartbeats go stale and the daemon marks them Stopped so
    // the manifest reflects reality.
    let mut last_staleness_check = std::time::Instant::now();
    const STALENESS_CHECK_INTERVAL: Duration = Duration::from_secs(60);
    while !shutdown_flag.load(std::sync::atomic::Ordering::Relaxed)
        && !ipc_shutdown_flag.load(std::sync::atomic::Ordering::Relaxed)
    {
        std::thread::sleep(Duration::from_millis(100));
        if last_staleness_check.elapsed() >= STALENESS_CHECK_INTERVAL {
            last_staleness_check = std::time::Instant::now();
            if let Ok(mut tl) = tick_loop.lock() {
                tl.check_staleness(&mmap);
            }
        }
    }

    // Shutdown sequence
    eprintln!("[genesis] Shutting down...");

    // Restore the host's CPU policy captured at startup. This runs
    // before the sync block below (which can bail out on a poisoned
    // LTM mutex), so the machine is never left pinned to a throttled
    // governor. It runs only after she has stopped — it cannot affect
    // her running state.
    if genesis::daemon::cpufreq::restore_hardware_state() {
        eprintln!("[genesis] Host CPU policy restored.");
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
    // persists across restarts. Genesis doesn't re-learn her own
    // neurochemical dynamics from scratch every time she wakes up.
    match tick_loop.lock() {
        Ok(tl) => clean_shutdown &= tl.save_inference_model(&config.data_dir),
        Err(e) => {
            eprintln!("[genesis] WARNING: tick loop poisoned during shutdown: {e}");
            clean_shutdown = false;
        }
    }

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
            std::thread::sleep(Duration::from_millis(50));
        }
    })
}

// Static atomic for signal handler communication
use std::sync::atomic::AtomicBool;
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

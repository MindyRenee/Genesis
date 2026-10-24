//! The mind shepherd — the body's half of the closed loop.
//!
//! Rebirth (`genesis_cli._Rebirth`) lets a living mind reincarnate a
//! dead body. This is the mirror: while the body lives, a mind that
//! was present and then went silent is regrown — the daemon spawns
//! the organism's own entry point headless, and the new mind adopts
//! the surviving body over the socket it still holds (`_adopt_daemon`
//! — a returning mind is re-inhabited, never refused). Either organ
//! can rebuild the other around the same persisted state; the loop
//! needs no outside hand for single-organ death.
//!
//! # Semantics — a death, not an absence
//!
//! The shepherd fires only on **loss**: `TickLoop::mind_idle_secs`
//! returns `Some` only after a mind has driven at least once. A mind
//! that never attached (daemon run standalone, dev harness, boot
//! window before the first connect) is an *absence* — the body stays
//! solitary by design and the shepherd stays out of it, which also
//! keeps a starting mind from racing a spawn.
//!
//! # Semantics — a goodbye, not a death
//!
//! `mind.suppress` in the data dir marks an intentional absence: any
//! graceful teardown (`--stop`, SIGINT, SIGTERM, systemd stop) writes
//! it before the mind leaves, and the next CLI start clears it. A
//! mind that dies *without* saying goodbye — crash, SIGKILL, host
//! fault — is a loss and is regrown; a mind that said goodbye is left
//! absent. The sentinel is host-local intent and stays out of the
//! spore mirror.
//!
//! # Bounds
//!
//! Attempts are bounded (like the retina's re-tries): the first comes
//! after `first_attempt_idle_s` of silence, later attempts wait out
//! `retry_gaps_s`, and past the table the shepherd goes dormant —
//! accepted absent, not fatal. A mind that reattaches resets
//! everything, so the next loss starts the table over.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

/// Sentinel name in the data dir: present ⇒ the last mind left
/// intentionally and the body must not regrow one. Written by
/// teardown paths, cleared by the next CLI start.
pub const MIND_SUPPRESS_FILE: &str = "mind.suppress";

/// Spawned mind's output — the shepherd's child has no terminal, so
/// its stdout/stderr append here beside daemon.log/retina.log.
pub const MIND_LOG_FILE: &str = "mind.log";

/// Env override for the spawn command — a whitespace-split command
/// line that takes precedence over every discovered entry point.
pub const MIND_CMD_ENV: &str = "GENESIS_MIND_CMD";

/// What one shepherd check decided. The caller drives the adaptive
/// gate on it — a spawn or an in-flight child is watched closely, a
/// suppressed/dormant shepherd drifts the check toward its ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShepherdOutcome {
    /// A mind is attached or attached recently — counters reset.
    Attached,
    /// No mind has ever driven — absence is a legal state, not a loss.
    Absent,
    /// A loss, still inside the grace window.
    Waiting,
    /// A spawn attempt was just made.
    Spawned,
    /// A spawned child is still running; no parallel spawn is allowed
    /// (the CLI lock would refuse a second anyway).
    PendingAlive,
    /// `mind.suppress` is present — the last mind said goodbye.
    Suppressed,
    /// Attempt table exhausted — solitude accepted, not fatal.
    Dormant,
    /// No spawnable entry point found on this host.
    Unresolvable,
}

/// Tunables so tests can run the same logic on a fast clock.
#[derive(Debug, Clone)]
pub struct ShepherdConfig {
    /// Silence (s of mind idle) before the first spawn attempt.
    pub first_attempt_idle_s: f32,
    /// Gaps between later attempts: after failed attempt n (1-based),
    /// the next may fire once this much idle has accrued past the
    /// previous attempt. Table length bounds retries; the daemon
    /// defaults give ~5 attempts over ~45 minutes before dormancy.
    pub retry_gaps_s: Vec<f32>,
    /// Explicit spawn command (GENESIS_MIND_CMD). Beats discovery.
    pub cmd_override: Option<Vec<String>>,
    /// Directory the daemon binary lives in — injectable for tests;
    /// production passes `current_exe`'s parent.
    pub exe_dir: PathBuf,
}

impl Default for ShepherdConfig {
    fn default() -> Self {
        Self {
            first_attempt_idle_s: 60.0,
            retry_gaps_s: vec![120.0, 300.0, 600.0, 1800.0],
            cmd_override: std::env::var(MIND_CMD_ENV)
                .ok()
                .map(|s| s.split_whitespace().map(String::from).collect()),
            exe_dir: std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_default(),
        }
    }
}

/// A spawn still in flight — held for reaping and for "did it
/// attach" bookkeeping. `counted` flips when the attempt is judged:
/// attached (marked on a fresh lease) or timed out with the lease
/// still expired; an exit before either is the early-death case the
/// caller counts separately.
struct PendingSpawn {
    child: Child,
    spawned_at: Instant,
    /// False while the child is still inside its boot grace — an
    /// attempt counts once the grace expires with no lease renewal.
    counted: bool,
}

/// The daemon's regenerative reflex for the mind: watches the mind
/// lease, spawns the organism's entry point when a live mind goes
/// silent, and bounds itself into dormant solitude when regrowth
/// fails. Holds the spawn table state and the in-flight child so a
/// dead one is always reaped — a daemon that lives for months cannot
/// leak zombies.
pub struct MindShepherd {
    config: ShepherdConfig,
    /// Spawn attempts since the last attached mind was observed.
    attempts: u32,
    /// Idle-seconds watermark at the last attempt — the next attempt
    /// waits for `retry_gaps` beyond it.
    idle_at_last_attempt: f32,
    /// A spawn still in flight; held for reaping so a dead child can
    /// never zombie under a daemon that lives for months.
    pending: Option<PendingSpawn>,
    /// Attempt table exhausted for the current loss.
    dormant: bool,
    /// Report once per disarm so the log is not spammed every gate.
    reported_unresolvable: bool,
    reported_dormant: bool,
}

impl Default for MindShepherd {
    fn default() -> Self {
        Self::with_config(ShepherdConfig::default())
    }
}

impl MindShepherd {
    /// Production shepherd: real timings and the host's own spawn
    /// discovery (env override → `.project_root` → beside-binary).
    pub fn new() -> Self {
        Self::default()
    }

    /// Shepherd over explicit tunables — the test hook for a fast
    /// clock and a fake spawn target.
    pub fn with_config(config: ShepherdConfig) -> Self {
        Self {
            config,
            attempts: 0,
            idle_at_last_attempt: 0.0,
            pending: None,
            dormant: false,
            reported_unresolvable: false,
            reported_dormant: false,
        }
    }

    /// One shepherd check. `mind_idle` is `TickLoop::mind_idle_secs`:
    /// `None` — a mind never drove (absence); `Some(s)` — `s` seconds
    /// since the last drive, so `s < lease` means attached now.
    ///
    /// Called from the fallback loop every shepherd-gate tick,
    /// regardless of lease: a returning mind must be observed to
    /// reset the attempt table for the next loss.
    pub fn check(&mut self, mind_idle: Option<f32>, data_dir: &Path) -> ShepherdOutcome {
        let child_died_uncounted = self.reap();

        match mind_idle {
            None => {
                // Absence, not loss — and a from-boot absence resets
                // nothing (no mind was ever ours to lose).
                ShepherdOutcome::Absent
            }
            Some(idle) if idle < super::tick::MIND_LEASE_SECS => {
                // A mind is driving right now: fresh attachment
                // observed — the next loss starts a fresh table. An
                // in-flight spawn that made it to the lease is a
                // *successful* attempt, so its eventual exit (even a
                // long life later) is never counted as a boot failure.
                if let Some(p) = self.pending.as_mut() {
                    p.counted = true;
                }
                self.attempts = 0;
                self.idle_at_last_attempt = 0.0;
                self.dormant = false;
                self.reported_dormant = false;
                self.reported_unresolvable = false;
                ShepherdOutcome::Attached
            }
            Some(idle) => self.maybe_spawn(idle, data_dir, child_died_uncounted),
        }
    }

    /// Reap a finished child. Returns true when a child exited
    /// *before* its boot grace was spent — a spawn that never stood a
    /// chance to attach, which the caller counts as a failed attempt.
    fn reap(&mut self) -> bool {
        let Some(p) = self.pending.as_mut() else {
            return false;
        };
        match p.child.try_wait() {
            Ok(Some(status)) => {
                let uncounted = !p.counted;
                eprintln!("[genesis] Shepherd: spawned mind exited ({status}).");
                self.pending = None;
                uncounted
            }
            Ok(None) => false,
            Err(e) => {
                eprintln!("[genesis] WARNING: shepherd wait failed: {e}");
                self.pending = None;
                true
            }
        }
    }

    fn maybe_spawn(
        &mut self,
        idle: f32,
        data_dir: &Path,
        child_died_uncounted: bool,
    ) -> ShepherdOutcome {
        if data_dir.join(MIND_SUPPRESS_FILE).exists() {
            return ShepherdOutcome::Suppressed;
        }

        // A mind that dies inside its boot window never reached the
        // lease — the exit itself is the failed attempt.
        if child_died_uncounted {
            self.attempts += 1;
            self.idle_at_last_attempt = idle;
        }

        // A child still in flight blocks any parallel spawn. Past the
        // boot grace with the lease still expired it counts as a
        // failed attempt — but it keeps blocking, since a wedged CLI
        // holds the singleton lock and a second spawn would only fail.
        let boot_grace = self.boot_grace_s();
        if let Some(p) = self.pending.as_mut() {
            if !p.counted && p.spawned_at.elapsed().as_secs_f32() >= boot_grace {
                p.counted = true;
                self.attempts += 1;
                self.idle_at_last_attempt = idle;
            }
            return ShepherdOutcome::PendingAlive;
        }
        // attempts counts FAILED attempts: once they exceed the retry
        // table there is no next gap — the shepherd goes dormant.
        if self.attempts > self.config.retry_gaps_s.len() as u32 {
            if !self.reported_dormant {
                self.reported_dormant = true;
                eprintln!(
                    "[genesis] Mind shepherd dormant after {} attempts — \
                     a mind that will not regrow is accepted absent.",
                    self.attempts
                );
            }
            return ShepherdOutcome::Dormant;
        }
        if self.attempts == 0 && idle < self.config.first_attempt_idle_s {
            return ShepherdOutcome::Waiting;
        }
        if self.attempts > 0 {
            let gap = self
                .config
                .retry_gaps_s
                .get((self.attempts - 1) as usize)
                .copied()
                .unwrap_or(f32::MAX);
            if idle < self.idle_at_last_attempt + gap {
                return ShepherdOutcome::Waiting;
            }
        }

        let Some(mut cmd) = self.resolve(data_dir) else {
            if !self.reported_unresolvable {
                self.reported_unresolvable = true;
                eprintln!(
                    "[genesis] Mind shepherd found no mind to grow \
                     (checked ${MIND_CMD_ENV}, .project_root, and the \
                     daemon's own directory) — staying solitary."
                );
            }
            return ShepherdOutcome::Unresolvable;
        };

        eprintln!(
            "[genesis] Mind silent {idle:.0}s — the body regrows its mind \
             (attempt {}).",
            self.attempts + 1
        );
        match cmd.spawn() {
            Ok(child) => {
                // Not yet counted: the child gets its boot grace to
                // attach before the attempt is called a failure.
                self.pending = Some(PendingSpawn {
                    child,
                    spawned_at: Instant::now(),
                    counted: false,
                });
                ShepherdOutcome::Spawned
            }
            Err(e) => {
                eprintln!("[genesis] WARNING: shepherd spawn failed: {e}");
                self.attempts += 1;
                self.idle_at_last_attempt = idle;
                ShepherdOutcome::Unresolvable
            }
        }
    }

    /// How long a spawned child may run unattached before the attempt
    /// counts against the table.
    fn boot_grace_s(&self) -> f32 {
        self.config.first_attempt_idle_s.max(90.0)
    }

    /// Resolve the command that grows a mind, in priority order:
    ///
    /// 1. `GENESIS_MIND_CMD` — explicit override (test hook, unusual
    ///    layouts).
    /// 2. `.project_root` — the checkout that owns this data dir, run
    ///    exactly as genesis runs it: the install venv's python (or
    ///    `python3`), `genesis_cli.py`, `PYTHONPATH=<root>/python`,
    ///    `GENESIS_WAKE=1`, `--headless`.
    /// 3. `genesis` beside the daemon binary — the install layout,
    ///    where scripts/install.sh places the console script next to
    ///    `genesis-daemon` under `venv/bin`.
    fn resolve(&self, data_dir: &Path) -> Option<Command> {
        if let Some(argv) = &self.config.cmd_override {
            if argv.is_empty() {
                return None;
            }
            let mut cmd = Command::new(&argv[0]);
            cmd.args(&argv[1..]);
            self.finish(&mut cmd, data_dir);
            return Some(cmd);
        }

        if let Some(cmd) = self.resolve_checkout(data_dir) {
            return Some(cmd);
        }

        let installed = self.config.exe_dir.join("genesis");
        if installed.is_file() {
            let mut cmd = Command::new(&installed);
            cmd.arg("--headless")
                .arg("--data-dir")
                .arg(data_dir.as_os_str());
            self.finish(&mut cmd, data_dir);
            return Some(cmd);
        }
        None
    }

    /// The dev-checkout spawn: the tree named by `.project_root`,
    /// launched the way genesis launches it — but headless, since a
    /// shepherd-spawned mind has no terminal.
    fn resolve_checkout(&self, data_dir: &Path) -> Option<Command> {
        let marker = data_dir.join(".project_root");
        let root: PathBuf = std::fs::read_to_string(&marker)
            .ok()
            .map(|s| PathBuf::from(s.trim()))?;
        let cli = root.join("genesis_cli.py");
        if !cli.is_file() {
            return None;
        }
        // The canonical interpreter is the install venv's (same rule
        // genesis enforces); a host without an install falls back to
        // system python3 rather than failing the spawn outright.
        let install_home = std::env::var("GENESIS_INSTALL_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/lib/genesis")
            });
        let venv_py = install_home.join("venv/bin/python");
        let python = if venv_py.is_file() {
            venv_py
        } else {
            PathBuf::from("python3")
        };
        let mut cmd = Command::new(python);
        cmd.arg(&cli)
            .arg("--data-dir")
            .arg(data_dir.as_os_str())
            .arg("--headless")
            .current_dir(&root)
            .env("GENESIS_WAKE", "1");
        let pythonpath = match std::env::var("PYTHONPATH") {
            Ok(existing) if !existing.is_empty() => {
                format!("{}/python:{existing}", root.display())
            }
            _ => format!("{}/python", root.display()),
        };
        cmd.env("PYTHONPATH", pythonpath);
        self.finish(&mut cmd, data_dir);
        Some(cmd)
    }

    /// Shared spawn plumbing: no terminal, output appended to the
    /// data dir's mind.log.
    fn finish(&self, cmd: &mut Command, data_dir: &Path) {
        cmd.stdin(Stdio::null());
        if let Ok(log) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(data_dir.join(MIND_LOG_FILE))
        {
            if let Ok(err) = log.try_clone() {
                cmd.stdout(log).stderr(err);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "shepherd_test_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fast_config(cmd: Vec<String>) -> ShepherdConfig {
        ShepherdConfig {
            first_attempt_idle_s: 0.0,
            retry_gaps_s: vec![0.0, 0.0, 0.0, 0.0],
            cmd_override: Some(cmd),
            exe_dir: PathBuf::new(),
        }
    }

    #[test]
    fn absence_never_spawns() {
        let dir = tmpdir("absent");
        let mut s = MindShepherd::with_config(fast_config(vec!["/bin/false".into()]));
        assert_eq!(s.check(None, &dir), ShepherdOutcome::Absent);
        assert!(s.pending.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn attached_resets_the_table() {
        let dir = tmpdir("attached");
        let mut s = MindShepherd::with_config(fast_config(vec!["/bin/false".into()]));
        s.attempts = 3;
        s.dormant = true;
        // idle under the lease = a mind is driving right now.
        assert_eq!(s.check(Some(0.5), &dir), ShepherdOutcome::Attached);
        assert_eq!(s.attempts, 0);
        assert!(!s.dormant);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn suppress_file_blocks_regrowth() {
        let dir = tmpdir("suppress");
        fs::write(dir.join(MIND_SUPPRESS_FILE), "stopped").unwrap();
        let mut s = MindShepherd::with_config(fast_config(vec!["/bin/false".into()]));
        // Deep into a loss — without the sentinel this would spawn.
        assert_eq!(s.check(Some(600.0), &dir), ShepherdOutcome::Suppressed);
        assert!(s.pending.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_loss_spawns_the_override_command() {
        let dir = tmpdir("spawn");
        // A marker file proves the child ran — /bin/sh -c 'touch …'.
        let proof = dir.join("spawned.proof");
        let cmd = vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            format!("touch {}", proof.display()),
        ];
        let mut s = MindShepherd::with_config(fast_config(cmd));
        assert_eq!(s.check(Some(120.0), &dir), ShepherdOutcome::Spawned);
        // The child gets boot grace: pending until it exits and is
        // reaped by a later check.
        for _ in 0..50 {
            s.reap();
            if s.pending.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(s.pending.is_none(), "exited child must be reaped");
        assert!(proof.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn first_attempt_waits_out_the_grace_window() {
        let dir = tmpdir("grace");
        let mut cfg = fast_config(vec!["/bin/false".into()]);
        cfg.first_attempt_idle_s = 60.0;
        let mut s = MindShepherd::with_config(cfg);
        assert_eq!(s.check(Some(30.0), &dir), ShepherdOutcome::Waiting);
        assert!(s.pending.is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failed_attempts_end_dormant() {
        let dir = tmpdir("dormant");
        // Spawn a command that exits instantly; each failure opens a
        // zero-gap retry window until the table runs out.
        let mut cfg = fast_config(vec!["/bin/false".into()]);
        cfg.first_attempt_idle_s = 0.0;
        let mut s = MindShepherd::with_config(cfg);
        let mut saw_dormant = false;
        for _ in 0..200 {
            match s.check(Some(10_000.0), &dir) {
                ShepherdOutcome::Dormant => {
                    saw_dormant = true;
                    break;
                }
                ShepherdOutcome::PendingAlive => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                _ => {}
            }
        }
        assert!(saw_dormant, "bounded attempts must end dormant");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn checkout_resolution_uses_the_marked_tree() {
        let dir = tmpdir("checkout");
        let root = tmpdir("root");
        fs::write(root.join("genesis_cli.py"), "# cli\n").unwrap();
        fs::write(dir.join(".project_root"), format!("{}\n", root.display())).unwrap();
        let mut cfg = fast_config(vec![]);
        cfg.cmd_override = None;
        let s = MindShepherd::with_config(cfg);
        let cmd = s.resolve(&dir);
        assert!(cmd.is_some(), "a marked checkout must resolve");
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn nothing_resolvable_stays_solitary() {
        let dir = tmpdir("none");
        let mut cfg = fast_config(vec![]);
        cfg.cmd_override = None;
        cfg.exe_dir = PathBuf::from("/nonexistent");
        let mut s = MindShepherd::with_config(cfg);
        assert_eq!(s.check(Some(600.0), &dir), ShepherdOutcome::Unresolvable);
        let _ = fs::remove_dir_all(&dir);
    }
}

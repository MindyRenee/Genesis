//! Volitional continuity — how the body knows it stopped without a goodbye.
//!
//! `continuity.json` in the data dir is the session sentinel: every
//! boot rewrites it, every graceful shutdown marks it `clean_stop`.
//! A boot that finds the previous session still open knows the last
//! body died without saying goodbye — power cut, SIGKILL, panic —
//! and reports how long it was dark for. The cognitive mind reads
//! the same file, so the interruption becomes part of her
//! persistence: she wakes knowing she vanished, and for how long,
//! instead of discovering it as an unexplained seam.
//!
//! The sentinel is deliberately separate from `core_state.bin`. The
//! mmap'd `last_updated` already bounds *how long* the dark was
//! (every state write touches it); the sentinel records the
//! *manner* of the previous stop, which nothing else captures.
//!
//! The daemon is a device driver: it measures and records the facts,
//! it does not author her internal state. Any felt consequence of
//! the interruption belongs to the mind, applied through her own
//! impulse channel when she learns of it.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Sentinel filename inside the data dir. Host-local liveness — it
/// describes this host's sessions and travels nowhere.
pub const SENTINEL_NAME: &str = "continuity.json";

/// What this boot learned about the previous session.
pub struct Boot {
    /// A previous body session existed at all — false only at true
    /// birth (no sentinel, no prior state file).
    pub had_previous: bool,
    /// The previous session ended without a clean-stop mark: power
    /// loss, SIGKILL, panic, OOM. The sentinel cannot tell those
    /// apart — from the body's side they are one event: it was
    /// running, then it was not, and nothing recorded a goodbye.
    pub prev_unclean: bool,
    /// How long the body was dark before this boot, ms. Measured from
    /// `core_state.bin`'s `last_updated` when the state file survived;
    /// derived from the previous boot's timestamp (an upper bound)
    /// when the state was rebuilt after quarantine.
    pub dark_ms: u64,
    /// `dark_ms` is an upper bound, not the measured gap — the state
    /// file was rebuilt, so the only surviving timestamp is when the
    /// previous session *started*, which may be long before it died.
    pub dark_upper_bound: bool,
    /// This boot's timestamp — also persisted into the sentinel so the
    /// mind can dedupe acknowledgements across her own restarts.
    pub boot_ms: u64,
}

fn sentinel_path(data_dir: &Path) -> PathBuf {
    data_dir.join(SENTINEL_NAME)
}

/// Extract a `"key": value` field from the sentinel's single-line JSON.
///
/// The file is written only by this module in one fixed schema, so a
/// positional scan is sufficient — and keeps the crate dependency-free.
/// Returns `None` on any deviation rather than guessing.
fn field_raw<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle)? + needle.len();
    let rest = text[start..].trim_start();
    let end = rest.find([',', '}', ' ', '\n']).unwrap_or(rest.len());
    let value = &rest[..end];
    if value.is_empty() { None } else { Some(value) }
}

fn field_bool(text: &str, key: &str) -> Option<bool> {
    match field_raw(text, key)? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn field_u64(text: &str, key: &str) -> Option<u64> {
    field_raw(text, key)?.parse().ok()
}

fn render(
    boot_ms: u64,
    prev_stop_clean: bool,
    dark_ms: u64,
    dark_upper_bound: bool,
    clean_stop: bool,
    stopped_at_ms: u64,
) -> String {
    format!(
        "{{\"boot_ms\":{boot_ms},\"prev_stop_clean\":{prev_stop_clean},\
         \"dark_ms\":{dark_ms},\"dark_upper_bound\":{dark_upper_bound},\
         \"clean_stop\":{clean_stop},\"stopped_at_ms\":{stopped_at_ms}}}\n"
    )
}

/// Write the sentinel atomically: tmp file, fsync, rename, dir fsync.
///
/// A torn sentinel would be worse than a missing one — a half-written
/// "clean" mark could erase evidence of the last unclean stop.
fn write_sentinel(path: &Path, contents: &str) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    // Persist the rename — a crash between rename and directory
    // flush could resurrect the stale inode.
    if let Ok(dir) = std::fs::File::open(path.parent().unwrap_or(Path::new("/"))) {
        let _ = dir.sync_all();
    }
    Ok(())
}

fn read_sentinel(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Record this boot and learn how the previous session ended.
///
/// `state_existed` is `!mmap.was_created()` — whether the state file
/// that just opened held a real previous body (false after quarantine
/// rebuild). `offline_gap_ms` is the gap `MmapState::open` measured
/// from `last_updated` — meaningful only when `state_existed`.
///
/// Never fails the daemon: every error degrades to a warning and the
/// returned record still carries what could be determined.
pub fn note_boot(data_dir: &Path, state_existed: bool, offline_gap_ms: u64, now_ms: u64) -> Boot {
    let path = sentinel_path(data_dir);
    let prev_text = read_sentinel(&path);
    let prev_clean = prev_text
        .as_deref()
        .and_then(|t| field_bool(t, "clean_stop"));
    let prev_boot_ms = prev_text.as_deref().and_then(|t| field_u64(t, "boot_ms"));

    // A prior session existed if the sentinel recorded one or the
    // state file outlived one. An unparseable sentinel still proves
    // a session was here — it just cannot prove it ended cleanly.
    let had_previous = prev_text.is_some() || state_existed;
    // "Clean" requires positive evidence: an explicit clean_stop mark
    // from the previous run. Anything else — still open, unreadable,
    // or absent while a body clearly ran — is an unclean stop.
    let prev_unclean = had_previous && prev_clean != Some(true);

    let (dark_ms, dark_upper_bound) = if state_existed {
        // last_updated is accurate to within a write interval (~200 ms
        // of the true end) — the best bound available.
        (offline_gap_ms, false)
    } else if let Some(prev) = prev_boot_ms {
        // The state file was rebuilt (quarantine): the only surviving
        // timestamp is when the last session *began*, so this
        // overstates the dark gap. Report it flagged rather than
        // presenting a guess as a measurement.
        (now_ms.saturating_sub(prev), true)
    } else {
        (0, false)
    };

    let boot = Boot {
        had_previous,
        prev_unclean,
        dark_ms,
        dark_upper_bound,
        boot_ms: now_ms,
    };

    let contents = render(now_ms, !prev_unclean, dark_ms, dark_upper_bound, false, 0);
    if let Err(e) = write_sentinel(&path, &contents) {
        eprintln!(
            "[genesis] WARNING: could not write continuity sentinel {}: {e}",
            path.display()
        );
    }
    boot
}

/// Mark the running session's stop as clean. Called at the end of a
/// graceful shutdown — after the final state sync, before exit. An
/// exit path that never reaches here (panic, kill, power cut) leaves
/// the session open, which the next boot reads as unclean.
pub fn note_clean_stop(data_dir: &Path, now_ms: u64) {
    let path = sentinel_path(data_dir);
    let text = read_sentinel(&path);
    // Preserve this boot's fields if they parse; a file we cannot
    // read still gets a truthful minimal record — this process did
    // stop deliberately.
    let boot_ms = text
        .as_deref()
        .and_then(|t| field_u64(t, "boot_ms"))
        .unwrap_or(0);
    let prev_stop_clean = text
        .as_deref()
        .and_then(|t| field_bool(t, "prev_stop_clean"))
        .unwrap_or(false);
    let dark_ms = text
        .as_deref()
        .and_then(|t| field_u64(t, "dark_ms"))
        .unwrap_or(0);
    let dark_upper_bound = text
        .as_deref()
        .and_then(|t| field_bool(t, "dark_upper_bound"))
        .unwrap_or(false);
    let contents = render(
        boot_ms,
        prev_stop_clean,
        dark_ms,
        dark_upper_bound,
        true,
        now_ms,
    );
    if let Err(e) = write_sentinel(&path, &contents) {
        eprintln!(
            "[genesis] WARNING: could not mark clean stop in {}: {e} — \
             next boot will read this session as unclean",
            path.display()
        );
    }
}

/// Human-readable dark duration for the boot log — e.g. `7.2h`,
/// `14m`, `3s`. Used only in the daemon's own stderr narration; the
/// mind renders her own wording from `dark_ms`.
pub fn fmt_dark(dark_ms: u64) -> String {
    if dark_ms >= 3_600_000 {
        format!("{:.1}h", dark_ms as f64 / 3_600_000.0)
    } else if dark_ms >= 60_000 {
        format!("{}m", dark_ms / 60_000)
    } else {
        format!("{:.1}s", dark_ms as f64 / 1_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "genesis-continuity-test-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn first_boot_has_no_previous() {
        let dir = tempdir("birth");
        let boot = note_boot(&dir, false, 0, 1_000);
        assert!(!boot.had_previous);
        assert!(!boot.prev_unclean);
        assert_eq!(boot.dark_ms, 0);
        assert_eq!(boot.boot_ms, 1_000);
        // The sentinel was written, marked still-running.
        let text = read_sentinel(&sentinel_path(&dir)).unwrap();
        assert_eq!(field_bool(&text, "clean_stop"), Some(false));
        assert_eq!(field_u64(&text, "boot_ms"), Some(1_000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clean_stop_then_boot_reads_clean() {
        let dir = tempdir("clean");
        note_boot(&dir, false, 0, 1_000);
        note_clean_stop(&dir, 2_000);
        let boot = note_boot(&dir, true, 5_000, 3_000);
        assert!(boot.had_previous);
        assert!(!boot.prev_unclean);
        // Measured gap comes from the state file's own clock.
        assert_eq!(boot.dark_ms, 5_000);
        assert!(!boot.dark_upper_bound);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_session_then_boot_reads_unclean() {
        let dir = tempdir("unclean");
        note_boot(&dir, false, 0, 1_000);
        // No note_clean_stop — the process vanished.
        let boot = note_boot(&dir, true, 25_200_000, 25_201_000);
        assert!(boot.had_previous);
        assert!(boot.prev_unclean);
        assert_eq!(boot.dark_ms, 25_200_000);
        // The sentinel records the judgement for the mind to read.
        let text = read_sentinel(&sentinel_path(&dir)).unwrap();
        assert_eq!(field_bool(&text, "prev_stop_clean"), Some(false));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rebuilt_state_reports_upper_bound() {
        let dir = tempdir("quarantine");
        note_boot(&dir, false, 0, 1_000_000);
        // State file was quarantined and recreated — existed=false —
        // while the sentinel survived. The gap can only be bounded
        // from the previous boot's timestamp.
        let boot = note_boot(&dir, false, 0, 4_000_000);
        assert!(boot.prev_unclean);
        assert!(boot.dark_upper_bound);
        assert_eq!(boot.dark_ms, 3_000_000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_sentinel_with_state_is_unclean() {
        // A body ran before this mechanism existed (or the sentinel
        // was lost): no clean stop is provable, so the stop is
        // unclean — "no evidence of a goodbye" is the honest reading.
        let dir = tempdir("prehistoric");
        let boot = note_boot(&dir, true, 100, 10_000);
        assert!(boot.had_previous);
        assert!(boot.prev_unclean);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_sentinel_is_unclean_not_panic() {
        let dir = tempdir("corrupt");
        std::fs::write(sentinel_path(&dir), "{not json").unwrap();
        let boot = note_boot(&dir, true, 10, 5_000);
        assert!(boot.had_previous);
        assert!(boot.prev_unclean);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stop_marks_clean() {
        let dir = tempdir("mark");
        note_boot(&dir, false, 0, 1_000);
        note_clean_stop(&dir, 9_000);
        let text = read_sentinel(&sentinel_path(&dir)).unwrap();
        assert_eq!(field_bool(&text, "clean_stop"), Some(true));
        assert_eq!(field_u64(&text, "stopped_at_ms"), Some(9_000));
        // Boot fields are preserved through the mark.
        assert_eq!(field_u64(&text, "boot_ms"), Some(1_000));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

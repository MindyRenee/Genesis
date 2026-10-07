//! Resolving the live state directory.
//!
//! Genesis has one state directory per checkout, and many places need to
//! find it: the launcher, the maintenance scripts, and the example
//! programs that read or repair state. Each of those used to resolve it
//! independently, and they drifted.
//!
//! That drift is not cosmetic. When a launcher (Flatpak, a sandboxed
//! agent host) sets `XDG_DATA_HOME` to its own private directory, a
//! resolver that honors it silently attaches the checkout to a *second*,
//! empty state directory while the rest of the tooling reads the first.
//! Two instances then accumulate divergent concepts, edges, and
//! journals, and neither is a superset of the other. That happened here:
//! six days and ~8.7k concepts in a fork nobody knew existed, with the
//! original problem ("nothing is being archived") looking like a fact
//! about the system rather than a symptom of reading the wrong tree.
//!
//! So the rule is a single ordered list, implemented once here and
//! asserted by tests. The order is deliberately *not* the XDG default
//! order:
//!
//! 1. `GENESIS_DATA_DIR` — an explicit override, always wins.
//! 2. `.genesis-data-dir` — the checkout's uncommitted pin, whose
//!    contents are the data directory itself (not a parent of it). It
//!    outranks `XDG_DATA_HOME` precisely because `XDG_DATA_HOME` is
//!    ambient environment rather than Genesis configuration.
//! 3. `XDG_DATA_HOME/genesis`.
//! 4. `~/.local/share/genesis`.
//!
//! Shell and Python resolvers must implement this same order; see
//! `python/tests/test_data_dir_conformance.py`, which runs each of them
//! side by side so a future edit to one cannot silently disagree.

use std::path::PathBuf;

/// The name of the per-checkout pin file, relative to the repository root.
pub const PIN_FILE: &str = ".genesis-data-dir";

/// The inputs to [`resolve`], as read from the environment.
///
/// Kept separate from the resolution itself so the ordering rule is a
/// pure function: it can be tested exhaustively without mutating
/// process-wide environment state, which is `unsafe` in edition 2024
/// and racy under `cargo test`'s parallel threads anyway.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Env {
    /// `GENESIS_DATA_DIR` — an explicit override.
    pub data_dir: Option<String>,
    /// `XDG_DATA_HOME` — ambient launcher environment.
    pub xdg_data_home: Option<String>,
    /// `HOME`, used for the final fallback.
    pub home: Option<String>,
}

impl Env {
    /// Read the current process environment.
    pub fn from_process() -> Self {
        Self {
            data_dir: std::env::var("GENESIS_DATA_DIR").ok(),
            xdg_data_home: std::env::var("XDG_DATA_HOME").ok(),
            home: std::env::var("HOME").ok(),
        }
    }
}

/// A value that is absent or entirely whitespace counts as unset.
///
/// An exported-but-empty `GENESIS_DATA_DIR` must not resolve to the empty
/// path, which would silently mean the current directory.
fn usable(value: Option<&String>) -> Option<&str> {
    value.map(|v| v.trim()).filter(|v| !v.is_empty())
}

/// Apply the documented precedence order. Pure: no I/O beyond reading the
/// pin file at `repo_root`.
pub fn resolve(env: &Env, repo_root: &std::path::Path) -> PathBuf {
    if let Some(explicit) = usable(env.data_dir.as_ref()) {
        return PathBuf::from(explicit);
    }
    if let Some(pinned) = read_pin(&repo_root.join(PIN_FILE)) {
        return pinned;
    }
    base_dir(env).join("genesis")
}

/// Resolve the live state directory, reading the process environment and
/// treating the current working directory as the repository root.
///
/// `cargo run --example` runs with the package root as the working
/// directory, which is where the pin lives.
pub fn data_dir() -> PathBuf {
    resolve(&Env::from_process(), std::path::Path::new("."))
}

/// Resolve against an explicit repository root, for callers that know it.
pub fn data_dir_from(repo_root: &std::path::Path) -> PathBuf {
    resolve(&Env::from_process(), repo_root)
}

/// The XDG base directory, or `~/.local/share` when unset.
fn base_dir(env: &Env) -> PathBuf {
    if let Some(xdg) = usable(env.xdg_data_home.as_ref()) {
        return PathBuf::from(xdg);
    }
    match usable(env.home.as_ref()) {
        Some(home) => PathBuf::from(home).join(".local/share"),
        None => PathBuf::from(".local/share"),
    }
}

/// Read a pin file, returning `None` when it is absent, unreadable, or
/// blank. A pin of only whitespace must not resolve to the empty path:
/// that would silently point every tool at the current directory.
fn read_pin(path: &std::path::Path) -> Option<PathBuf> {
    let text = std::fs::read_to_string(path).ok()?;
    let first = text.lines().next()?.trim();
    if first.is_empty() {
        return None;
    }
    Some(PathBuf::from(first))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(data_dir: Option<&str>, xdg: Option<&str>, home: Option<&str>) -> Env {
        Env {
            data_dir: data_dir.map(str::to_string),
            xdg_data_home: xdg.map(str::to_string),
            home: home.map(str::to_string),
        }
    }

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("genesis-datadir-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_pin(root: &std::path::Path, body: &str) {
        std::fs::write(root.join(PIN_FILE), body).expect("write pin");
    }

    #[test]
    fn explicit_env_beats_everything() {
        let root = tmpdir("explicit");
        write_pin(&root, "/pinned");
        assert_eq!(
            resolve(&env(Some("/explicit"), Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/explicit"),
            "GENESIS_DATA_DIR must win over the pin and XDG_DATA_HOME"
        );
    }

    #[test]
    fn pin_outranks_xdg_data_home() {
        // The regression that caused a six-day fork: a launcher that
        // sets XDG_DATA_HOME must not be able to redirect a pinned
        // checkout to a different state directory.
        let root = tmpdir("pin-wins");
        write_pin(&root, "/pinned/data/dir\n");
        assert_eq!(
            resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/pinned/data/dir")
        );
    }

    #[test]
    fn falls_back_to_xdg_when_unpinned() {
        let root = tmpdir("no-pin");
        assert_eq!(
            resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/hostile/genesis")
        );
    }

    #[test]
    fn falls_back_to_home_share_when_nothing_is_set() {
        let root = tmpdir("bare");
        assert_eq!(
            resolve(&env(None, None, Some("/home/w")), &root),
            PathBuf::from("/home/w/.local/share/genesis")
        );
    }

    #[test]
    fn blank_pin_is_ignored_rather_than_resolving_to_empty() {
        // A pin that is empty or whitespace must not win: it would
        // resolve every tool to the current directory.
        for body in ["", "\n", "   \n\t\n"] {
            let root = tmpdir("blank-pin");
            write_pin(&root, body);
            assert_eq!(
                resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
                PathBuf::from("/hostile/genesis"),
                "blank pin ({body:?}) must fall through"
            );
        }
    }

    #[test]
    fn missing_pin_file_is_not_an_error() {
        let root = tmpdir("absent-pin");
        assert_eq!(
            resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/hostile/genesis")
        );
    }

    #[test]
    fn trailing_whitespace_and_newlines_are_trimmed() {
        let root = tmpdir("trim");
        write_pin(&root, "  /pinned/dir  \n\nignored second line\n");
        assert_eq!(
            resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/pinned/dir")
        );
    }

    #[test]
    fn unreadable_pin_falls_through_rather_than_erroring() {
        // A pin path that is a directory, not a file, cannot be read as
        // text. That must degrade to the XDG default, not panic.
        let root = tmpdir("dir-pin");
        std::fs::create_dir_all(root.join(PIN_FILE)).expect("make dir");
        assert_eq!(
            resolve(&env(None, Some("/hostile"), Some("/home/w")), &root),
            PathBuf::from("/hostile/genesis")
        );
    }

    #[test]
    fn blank_env_values_are_treated_as_unset_at_resolution_time() {
        // Normalization lives in resolve(), not in Env::from_process(),
        // so a directly-constructed Env cannot bypass it. An
        // exported-but-empty GENESIS_DATA_DIR must not resolve to "",
        // which would silently mean the current directory.
        let root = tmpdir("blank-env");
        write_pin(&root, "/pinned");
        assert_eq!(
            resolve(
                &Env {
                    data_dir: Some(String::new()),
                    xdg_data_home: Some("  ".to_string()),
                    home: Some(String::new()),
                },
                &root
            ),
            PathBuf::from("/pinned"),
            "blank GENESIS_DATA_DIR must fall through to the pin"
        );

        let bare = tmpdir("blank-env-no-pin");
        assert_eq!(
            resolve(
                &Env {
                    data_dir: Some("   ".to_string()),
                    xdg_data_home: Some(String::new()),
                    home: Some(String::new()),
                },
                &bare
            ),
            PathBuf::from(".local/share/genesis"),
            "all-blank env with no pin falls back to the relative default"
        );
    }
}

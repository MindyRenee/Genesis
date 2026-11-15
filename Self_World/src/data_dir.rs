//! The live state directory.
//!
//! There is exactly one canonical location — `~/.local/share/genesis` —
//! and exactly one override, `GENESIS_DATA_DIR`, for development and
//! testing. There is no pin file and no `XDG_DATA_HOME` step: ambient
//! environment must never be able to point one component at a second
//! state tree while the rest read the first — that is how a checkout
//! once forked into two divergent state directories for six days.
//!
//! The daemon, `genesis`, `genesis --headless`, and every tool resolve
//! the same answer. One directory, one Genesis.

use std::path::PathBuf;

/// Resolve from explicit values, so the rule is a pure function that can
/// be tested exhaustively without mutating process-wide environment
/// state (`unsafe` in edition 2024, and racy under `cargo test`).
///
/// A value that is absent or entirely whitespace counts as unset: an
/// exported-but-empty `GENESIS_DATA_DIR` must not resolve to the empty
/// path, which would silently mean the current directory.
pub fn resolve(data_dir_env: Option<&str>, home: Option<&str>) -> PathBuf {
    if let Some(explicit) = data_dir_env.map(str::trim).filter(|v| !v.is_empty()) {
        return PathBuf::from(explicit);
    }
    match home.map(str::trim).filter(|v| !v.is_empty()) {
        Some(home) => PathBuf::from(home).join(".local/share/genesis"),
        // No HOME either: a relative default keeps the path meaningful
        // for diagnostics instead of panicking in a bare environment.
        None => PathBuf::from(".local/share/genesis"),
    }
}

/// Resolve the live state directory, reading the process environment.
pub fn data_dir() -> PathBuf {
    resolve(
        std::env::var("GENESIS_DATA_DIR").ok().as_deref(),
        std::env::var("HOME").ok().as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_env_is_the_only_override() {
        assert_eq!(
            resolve(Some("/explicit"), Some("/home/w")),
            PathBuf::from("/explicit"),
            "GENESIS_DATA_DIR is the only way to leave the canonical dir"
        );
    }

    #[test]
    fn canonical_dir_is_home_local_share() {
        assert_eq!(
            resolve(None, Some("/home/w")),
            PathBuf::from("/home/w/.local/share/genesis")
        );
    }

    #[test]
    fn no_other_environment_can_redirect_state() {
        // There is deliberately no `xdg` parameter: a hostile or merely
        // foreign XDG_DATA_HOME cannot appear in this signature at all,
        // so it can never redirect the state dir again.
        assert_eq!(
            resolve(None, Some("/home/w")),
            PathBuf::from("/home/w/.local/share/genesis")
        );
    }

    #[test]
    fn blank_env_values_are_treated_as_unset() {
        assert_eq!(
            resolve(Some(""), Some("/home/w")),
            PathBuf::from("/home/w/.local/share/genesis"),
            "exported-but-empty GENESIS_DATA_DIR must not mean cwd"
        );
        assert_eq!(
            resolve(Some("   "), Some("/home/w")),
            PathBuf::from("/home/w/.local/share/genesis"),
            "whitespace-only GENESIS_DATA_DIR must not mean a bogus path"
        );
    }

    #[test]
    fn missing_home_degrades_to_relative_default() {
        assert_eq!(resolve(None, None), PathBuf::from(".local/share/genesis"));
        assert_eq!(
            resolve(None, Some("  ")),
            PathBuf::from(".local/share/genesis")
        );
    }
}

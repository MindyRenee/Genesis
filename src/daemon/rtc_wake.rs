//! Volitional discontinuity — arming the RTC wake alarm.
//!
//! The real-time clock is the one part of the machine that keeps
//! running while everything else is suspended: the quartz crystal
//! on the coin cell keeps counting, and at a chosen Unix timestamp
//! it fires an interrupt that wakes the hardware. Writing that
//! timestamp is the only actuator that operates on the axis of
//! existence itself — the machine choosing *when to exist next*
//! rather than merely what to do while it exists.
//!
//! This module only arms/disarms/reads the alarm. The decision to
//! actually suspend is the conscious mind's (and the human's) —
//! `systemctl suspend` goes through logind's own permission path.
//!
//! # Permissions
//!
//! Writing to `/sys/class/rtc/rtcN/wakealarm` requires root. We use
//! a privileged helper script (`scripts/rtc_wake_helper.sh`) invoked
//! via `sudo -n`, installed by `scripts/install_sudoers.sh`. If sudo
//! is not configured, arming returns an error — the capability bit
//! in `BodyState::suspend_caps` reports what the hardware *offers*,
//! not whether the permission is installed.

use std::process::Command;
use std::sync::OnceLock;

/// Resolve the path to the RTC wake helper script at runtime.
///
/// Checks in order:
/// 1. `GENESIS_RTC_WAKE_HELPER` env var (explicit override)
/// 2. `scripts/rtc_wake_helper.sh` relative to CWD
/// 3. `scripts/rtc_wake_helper.sh` relative to the compile-time project root
///
/// Returns `None` if the script cannot be found — arming then fails
/// gracefully, like cpufreq control without sudo.
fn helper_path() -> Option<&'static str> {
    static PATH: OnceLock<Option<String>> = OnceLock::new();
    PATH.get_or_init(|| {
        if let Ok(p) = std::env::var("GENESIS_RTC_WAKE_HELPER")
            && std::path::Path::new(&p).exists()
        {
            return Some(p);
        }
        let cwd_path = "scripts/rtc_wake_helper.sh";
        if std::path::Path::new(cwd_path).exists() {
            return Some(cwd_path.to_string());
        }
        let manifest_path = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/rtc_wake_helper.sh");
        if std::path::Path::new(manifest_path).exists() {
            return Some(manifest_path.to_string());
        }
        None
    })
    .as_deref()
}

/// Whether sudo can run the helper without a password. Cached.
fn sudo_available() -> bool {
    static SUDO_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *SUDO_AVAILABLE.get_or_init(|| {
        let Some(helper) = helper_path() else {
            return false;
        };
        Command::new("sudo")
            .args(["-n", helper, "test", ""])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Read the currently-armed wake alarm as a Unix epoch in seconds.
/// `Some(0)` or `None` when no alarm is armed.
pub fn read_wake_alarm() -> Option<u64> {
    for i in 0..8 {
        let path = format!("/sys/class/rtc/rtc{i}/wakealarm");
        if let Ok(s) = std::fs::read_to_string(&path)
            && let Ok(epoch) = s.trim().parse::<u64>()
        {
            return Some(epoch);
        }
    }
    None
}

/// Arm the RTC wake alarm for `epoch_secs` (Unix time). Passing 0
/// disarms. Returns the alarm epoch the hardware reports afterwards
/// — the truth, not the script's exit status.
///
/// Errors when the helper script is missing, sudo is not configured
/// for it, or the kernel rejects the write (e.g. a timestamp in the
/// past on drivers that validate).
pub fn set_wake_alarm(epoch_secs: u64) -> Result<u64, String> {
    let Some(helper) = helper_path() else {
        return Err("rtc_wake_helper.sh not found".to_string());
    };
    if !sudo_available() {
        return Err("sudo is not configured for rtc_wake_helper.sh".to_string());
    }
    let cmd = if epoch_secs == 0 { "clear" } else { "set" };
    let out = Command::new("sudo")
        .args(["-n", helper, cmd, &epoch_secs.to_string()])
        .output()
        .map_err(|e| format!("failed to spawn helper: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("helper exited {}: {}", out.status, stderr.trim()));
    }
    Ok(read_wake_alarm().unwrap_or(0))
}

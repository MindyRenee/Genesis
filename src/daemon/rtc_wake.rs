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
//! Writing to `/sys/class/rtc/rtcN/wakealarm` requires root. The
//! writes go through `sudo -n` directly — she IS the privilege
//! channel on a host with the broad NOPASSWD grant. Where the grant
//! is absent, arming returns an error — the capability bit in
//! `BodyState::suspend_caps` reports what the hardware *offers*, not
//! whether the permission is installed.

use std::process::Command;
#[cfg(not(target_os = "android"))]
use std::sync::OnceLock;

/// RTC wake-alarm control is not exposed by the Termux runtime.
#[cfg(target_os = "android")]
fn sudo_available() -> bool {
    false
}

/// Whether `sudo -n` runs without a password. Cached on desktop Linux.
#[cfg(not(target_os = "android"))]
fn sudo_available() -> bool {
    static SUDO_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *SUDO_AVAILABLE.get_or_init(|| {
        Command::new("sudo")
            .args(["-n", "true"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// Find the RTC that actually exposes a wakealarm (usually rtc0).
fn find_wakealarm() -> Option<String> {
    for i in 0..8 {
        let path = format!("/sys/class/rtc/rtc{i}/wakealarm");
        if std::path::Path::new(&path).is_file() {
            return Some(path);
        }
    }
    None
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
/// — the truth, not the write's exit status.
///
/// Errors when sudo is not configured, no RTC exposes a wakealarm,
/// the timestamp is in the past (an alarm that can never fire is a
/// silent failure, worse than an explicit error), or the kernel
/// rejects the write.
pub fn set_wake_alarm(epoch_secs: u64) -> Result<u64, String> {
    if !sudo_available() {
        return Err("passwordless sudo is not configured (sudo -n fails)".to_string());
    }
    let Some(wakealarm) = find_wakealarm() else {
        return Err("no RTC wakealarm found".to_string());
    };
    let script = if epoch_secs == 0 {
        // Disarm.
        format!("echo 0 > {wakealarm}")
    } else {
        // Reject timestamps in the past before they reach sysfs.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if epoch_secs <= now {
            return Err(format!("epoch {epoch_secs} is in the past (now {now})"));
        }
        // The kernel requires disarming before arming a new value on
        // some rtc drivers — write 0 first.
        format!("echo 0 > {wakealarm}; echo {epoch_secs} > {wakealarm}")
    };
    let out = Command::new("sudo")
        .args(["-n", "bash", "-c", &script])
        .output()
        .map_err(|e| format!("failed to spawn sudo: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("wakealarm write failed: {}", stderr.trim()));
    }
    Ok(read_wake_alarm().unwrap_or(0))
}

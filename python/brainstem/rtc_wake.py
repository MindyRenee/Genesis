"""RTC wake — word-layer companion to ``src/daemon/rtc_wake.rs``.

Volitional discontinuity in reverse: the real-time clock is the one
part of the machine that keeps running while everything else is
suspended — the quartz on the coin cell counts, and at a chosen Unix
timestamp it fires an interrupt that wakes the hardware. Arming the
RTC alarm is the body scheduling its own wake; absent on Android,
where the kernel doesn't expose the alarm to unprivileged users.
"""

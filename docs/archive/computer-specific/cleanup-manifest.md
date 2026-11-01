# Genesis3 platform cleanup inventory

Scope: **Genesis3 only**. The original Genesis branch is not modified.

This inventory records each platform-specific integration reviewed so far. It distinguishes archived code from code that remains available for Linux and code that still needs a real Android replacement.

## Archived and removed from active paths

| File | Decision | Reason |
|---|---|---|
| `scripts/optic_relay.py` | Archived at `docs/archive/computer-specific/optic_relay.py`; active copy removed | X11/FFmpeg desktop-display relay is not a phone sensor. |
| `scripts/genesis-sudoers` | Archived at `docs/archive/computer-specific/genesis-sudoers`; active copy removed | Passwordless sudoers example is a desktop/Linux host-control mechanism. |
| `genesis_cli.py` | `_start_screen_nerve()` now returns `None` | Prevents the CLI from launching the archived desktop screen relay. |

## Android-aware or Android-disabled code

| File | Current handling |
|---|---|
| `Cargo.toml` | `v4l` is now a Linux-only target dependency. |
| `src/daemon/optic_nerve.rs` | Linux V4L2 capture is compiled only on Linux. Android reports the camera channel unavailable; no false camera data is generated. |
| `src/daemon/interoception.rs` | Detects readable CPU/SoC thermal zones by declared type; discovers Android-style battery supplies by `type=Battery`; recognizes USB/USB-C/USB-PD/wireless power sources; falls back to Termux:API battery percentage and known charging state when sysfs is missing or unreadable. |
| `src/daemon/cpufreq.rs` | Privileged host CPU-frequency controls cannot use sudo on Android. |
| `src/daemon/rtc_wake.rs` | RTC wake-alarm control is disabled on Android. |
| `src/daemon/rtkit.rs` | RTKit system-bus priority requests are disabled on Android. |
| `README.md` | Separates the established Linux workflow from the unverified Termux runtime. |

## Reviewed and retained as platform-specific/optional

| File | Decision and limitation |
|---|---|
| `scripts/install.sh` | Retained for Linux; it installs a systemd user service where available and is not the Termux installer. |
| `scripts/install_optional_deps.sh` | Retained for Linux package managers and PortAudio/pycairo setup; not an Android setup path. |
| `scripts/cargo-env.sh` | Retained as a Linux build-toolchain helper; not a runtime connection. |
| `.github/workflows/ci.yml` | Retained as Linux CI; it validates the existing Linux target, not Android. |
| `python/brainstem/cpufreq.py` | Retained for process-local CPU affinity and I/O scheduling. It is distinct from privileged host CPU-frequency control. |
| `python/brainstem/field_audio.py` | Retained as an optional PortAudio self-emission channel. Termux microphone recording is not an equivalent substitute for hearing the device's internal electrical/acoustic signature, so no false replacement was wired. |
| `python/tools/browser.py` | Retained as an optional Chrome/Chromium DevTools Protocol browser. It is not an Android-browser bridge. |
| `genesis_client/optic_nerve.py` and `python/occipital_lobe/perception/vision.py` | Retained as frame-consumer/perception code. An Android camera producer is not yet connected, so visual input remains absent on Android. |
| `docs/man/genesis-browser.7`, `docs/man/genesis-retina.1`, `docs/man/genesis-daemon.8`, `docs/man/genesis-cli.1` | Retained as existing Linux-oriented manuals; they do not activate hardware connections. |
| `docs/arc/README.md` and `scripts/install_arc_resources.sh` | Retained; their Termux support applies to ARC resources/benchmark tooling, not proof that the complete daemon runs on Android. |

## Remaining work before calling the phone port complete

- Cross-compile and run the full Rust daemon on an actual Android/Termux device; Linux CI does not prove this.
- Add and test a real Android camera bridge before claiming live vision.
- Add a permission-aware Android sensor adapter for accelerometer, gyroscope, light, proximity, and magnetometer only where the device reports them.
- Replace or explicitly disable the PortAudio microphone path on Android; do not treat an absent microphone as a measured signal.
- Provide a Termux-native installer after validating package names and dependency availability on-device.
- Re-audit the browser/tool integrations and stale Linux manuals after the Android runtime path is established.

The Termux:API battery fallback accepts only a valid percentage. It records charging state only for known values; an unknown charging value does not erase a valid battery percentage or get misreported as unplugged.

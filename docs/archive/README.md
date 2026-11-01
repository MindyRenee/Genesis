# Archived computer-specific integrations

These files are preserved for reference but are no longer active Genesis3 integrations.

- `computer-specific/optic_relay.py` — Linux V4L2/FFmpeg/X11 screen-capture relay. The CLI no longer launches it; a desktop display is not an Android sensor.
- `computer-specific/genesis-sudoers` — privileged Linux CPU-frequency control example. Do not install this on Android; Android/Termux does not provide the same sudoers-based host-control model.

The original Genesis branch is not changed. These archives are retained only in Genesis3 so desktop-specific behavior can be restored deliberately if needed.

## Android/Termux metric policy

Use real measurements only when the device exposes them. Linux `/proc` data can provide process CPU/RSS and memory information where Android permissions allow it. The kernel power-supply interface can expose battery percentage and charging state, but names vary by device; Genesis3 now discovers supplies by their declared type rather than assuming a laptop-style `BAT0`. Processor temperature is accepted only from a readable sensor whose type identifies CPU/SoC, or the known desktop `acpitz` fallback. It must not mistake skin, battery, charger, or GPU temperature for CPU temperature.

Termux:API can expose selected Android services through an installed companion app and the `termux-api` package, subject to Android permissions. Camera, microphone, and motion-sensor streaming are **not yet wired into the Genesis daemon**. Battery percentage and charging state have a Termux:API fallback when sysfs is unavailable. Do not interpret an unavailable sensor as a measured zero, full battery, or normal temperature.

Reference documentation:
- Android sensor discovery: https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview
- Termux:API: https://github.com/termux/termux-api

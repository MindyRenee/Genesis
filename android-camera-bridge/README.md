# Genesis native Android camera adapter

This companion app is the hardware-facing Camera2 adapter for Genesis running
in Termux. Android applications and Termux run under separate UIDs, so the
camera adapter forwards raw sensor frames over loopback to the Genesis daemon.
It does not decode scenes, recognize objects, or run perception.

## Single visual-afferent path

`Android Camera2` -> `YUV_420_888` to packed RGB24 -> loopback TCP
(`127.0.0.1:8765`, Android adapter hosts; daemon connects outward) -> Rust
`OpticCapture` limb -> the canonical
`/genesis_optic_nerve` frame mapping -> `genesis_client/optic_nerve.py` ->
the existing mind-side V1 and perception path.

The transport is only an operating-system boundary between the Android
hardware sandbox and the Termux daemon. The Android hardware adapter hosts
the loopback endpoint; the daemon makes an outbound connection, preserving
Genesis's single inbound daemon IPC listener. The daemon owns the optic mapping,
validates frame dimensions and payload lengths, and publishes frames using
the same 28-byte header and seqlock protocol as the Linux V4L2 capture limb.
The Python mind has no Android-specific camera fallback or alternate frame
queue.

Each packet is `GOPT`, width (u32 big-endian), height (u32 big-endian),
payload length (u32 big-endian), then packed RGB24 pixels. Frames are bounded
to 640x480 and the service targets up to 10 frames per second.

## Build the APK

The GitHub Actions workflow **Android camera bridge** builds a debug APK on
pushes to this directory or by manual workflow dispatch. Download the
`genesis-camera-bridge-debug` artifact from the workflow run and install
`app-debug.apk` on the Android phone.

## Run on the phone

1. Install the current `app-debug.apk` once. The adapter is the Android
   platform boundary for Camera2; it does not own perception.
2. Start Genesis with `./genesis` in Termux. On Android, the supported
   launcher opens the adapter activity automatically. On first use, Android
   presents its camera permission prompt; grant it while the activity is
   visible. The activity then starts the Camera2 foreground service. If the
   activity does not open, launch **Genesis Camera Adapter** manually and
   use **Start / retry camera stream**.
3. The foreground notification reports the actual progression: Camera2
   initialization, capture-session readiness, listener readiness, daemon
   connection, and first-frame transmission. It does not claim a live
   connection merely because the service started.
4. The daemon publishes validated frames to the canonical optic mapping;
   the ordinary mind-side vision path consumes them when Genesis chooses to
   look. Stop the stream from the activity or stop Genesis to release the
   camera.

Android requires a user-granted runtime camera permission and a visible
foreground activity when starting a camera foreground service. Termux cannot
open Camera2 directly; the launcher opens the existing hardware adapter
rather than bypassing Android's permission boundary. If the adapter is
unavailable or permission is denied, Genesis continues without visual
input. There is still only one optic nerve and one mind-side perception path.

For device-side diagnostics, inspect the Android log with
`logcat -s GenesisCamera`; the foreground notification also shows the
latest service state.

## Verification boundary

CI can compile the Android app and run the repository's Rust/Python checks.
Actual camera permission, device-specific YUV plane layout, and end-to-end
frame delivery still require a physical Android device test.

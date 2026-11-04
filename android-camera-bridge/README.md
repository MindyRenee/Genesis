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

1. In Termux, update the Genesis checkout on `audit-fixes` and build/start
   Genesis normally. The daemon starts its Android optic-capture client and retries until the
Android hardware adapter is active.
2. Open **Genesis Camera Adapter**, grant camera permission, and tap
   **Start camera stream**.
3. The service hosts the loopback camera endpoint for the daemon to connect to. The daemon publishes each
   validated frame to the canonical optic mapping; the ordinary mind-side
   vision path consumes it when Genesis chooses to look.
4. Tap **Stop camera stream** or stop the service to release the camera.

Android requires explicit camera permission and a visible foreground-service
notification. If the app is not installed or streaming, the optic nerve reports
the eye absent; there is no second Python camera path that bypasses the nerve.

## Verification boundary

CI can compile the Android app and run the repository's Rust/Python checks.
Actual camera permission, device-specific YUV plane layout, and end-to-end
frame delivery still require a physical Android device test.

# Genesis native Android camera bridge

This companion app provides continuous camera frames to Genesis running in
Termux. It uses Android Camera2 repeating capture requests; it does not perform
recognition or create a second perception system.

## Data path

`Camera2` -> latest JPEG frame -> loopback TCP (`127.0.0.1:8765>) ->
`python/occipital_lobe/perception/android_stream.py` -> existing
`Vision.see_frame()` pipeline.

Frames use a 4-byte big-endian length followed by JPEG bytes. The receiver keeps
only the newest frame and rejects frames older than two seconds. The camera
service targets up to 10 frames/second at a resolution no larger than 640x480;
Genesis's existing volition still decides when to run the perception pipeline.

## Build the APK

The GitHub Actions workflow **Android camera bridge** builds a debug APK on
pushes to this directory or by manual workflow dispatch. Download the
`genesis-camera-bridge-debug` artifact from the workflow run and install
`app-debug.apk` on the Android phone. Android may require allowing installs
from the browser or file manager used to open the APK.

## Run on the phone

1. In Termux, update the Genesis checkout on `audit-fixes` and install the
   project's Python dependencies as usual. Pillow must be available.
2. Start Genesis in Termux. Its Vision subsystem starts the loopback receiver
   when vision availability is checked.
3. Open **Genesis Camera Bridge**, grant camera permission, and tap **Start
   camera stream**. Android displays a foreground-service notification while
   the camera is active.
4. Return to Genesis. When autonomous volition chooses to look, the normal
   vision pipeline consumes the latest stream frame. The bridge does not
   issue a look command or decide when Genesis should perceive.
5. Tap **Stop camera stream** or stop the service to release the camera.

The listener binds only to loopback and is not exposed to the local network.
The bridge does not save camera frames to disk. If the app is not installed or
streaming, Genesis retains the existing shared-memory optic feed and the
Termux still-capture fallback where available.

## Current limitations

- This first implementation is source-complete but has not yet been built on
  the project's CI runner or tested on the user's phone.
- The Android app must be launched and streaming for continuous frames to be
  available; Android requires an explicit camera permission and visible
  foreground-service notification.
- It currently streams JPEGs at a fixed capture cadence. Adaptive camera
  cadence and device-specific camera selection can be added after validating
  the initial end-to-end path.

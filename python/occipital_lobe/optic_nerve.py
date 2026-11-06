"""Optic nerve — word-layer companion to ``src/daemon/optic_nerve.rs``.

The whole retinal afferent path, held inside the body. On Linux the
daemon opens the camera directly through V4L2; on Android a Camera2
adapter opens the sandboxed device and forwards raw frames to the
daemon's capture limb over TCP — the same published stream either
way, so the mind never knows which body the eye lives in. An eye
that isn't there is reported absent — dark is dark, not noise.
"""

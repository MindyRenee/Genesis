"""IPC layer — word-layer companion to ``src/daemon/ipc.rs``.

The Unix-domain-socket protocol between the Python cognitive mind and
the Rust subcognitive daemon — length-prefixed binary, not JSON, so
the hot path never parses text. Commands are single bytes
(``GET_STATE``, ``GET_MOTOR_TARGET``, ``SET_COGNITIVE_PID``...), each
with a documented payload shape; the daemon answers in kind. This is
the vagus nerve of the architecture: the one wire the word layer and
the body speak across.
"""

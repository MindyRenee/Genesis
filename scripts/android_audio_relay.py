#!/usr/bin/env python3
"""Android microphone relay — TCP PCM to a PulseAudio pipe source.

The camerabridge app's AudioStreamService listens on 127.0.0.1:8766
and streams a 16-byte header ("GAUD" + u32 rate + u32 channels +
u32 bytes/sample) followed by raw PCM. This relay loads a Pulse
module-pipe-source named ``genesis_mic`` and pumps the PCM into its
FIFO, so sounddevice/vosk see a real input device.

Mirrors the optic-nerve transport: the bridge produces, the body
side consumes — a missing bridge degrades hearing, never fatally.

Usage: android_audio_relay.py [--host 127.0.0.1] [--port 8766]
                              [--source genesis_mic]
Env:   GENESIS_ANDROID_MIC_HOST/PORT override the endpoint.
"""

import os
import signal
import socket
import struct
import subprocess
import sys
import time

HEADER = 16
MAGIC = b"GAUD"


def _run(*args: str) -> int:
    return subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class PipeSource:
    """A PulseAudio pipe-source fed through a named FIFO."""

    def __init__(self, name: str, rate: int) -> None:
        runtime = os.environ.get("XDG_RUNTIME_DIR", "/tmp")
        self.name = name
        self.fifo = os.path.join(runtime, f"{name}.pipe")
        self.rate = rate
        self.module_id: int | None = None
        self._prev_default: str | None = None
        self.fd: int | None = None

    def load(self) -> None:
        if not os.path.exists(self.fifo):
            os.mkfifo(self.fifo, 0o666)
        # Read-write open so the FIFO never blocks on a missing reader
        # and Pulse sees data the moment it attaches.
        self.fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        # The Andronix Pulse server occasionally drops a fresh control
        # connection — retry the load a few times before giving up;
        # hearing degrades gracefully, never fatally.
        for attempt in range(5):
            out = subprocess.run(
                [
                    "pactl", "load-module", "module-pipe-source",
                    f"source_name={self.name}",
                    f"file={self.fifo}",
                    "format=s16le",
                    f"rate={self.rate}",
                    "channels=1",
                ],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                break
            time.sleep(1.0)
        if out.returncode == 0:
            self.module_id = int(out.stdout.strip() or -1)
            # PortAudio's 'default'/'pulse' devices only see the Pulse
            # *default* source — route it here while the relay lives so
            # her existing sounddevice stack picks the mic up unchanged.
            prev = subprocess.run(
                ["pactl", "get-default-source"], capture_output=True, text=True
            )
            if prev.returncode == 0:
                self._prev_default = prev.stdout.strip()
            _run("pactl", "set-default-source", self.name)

    def write(self, data: bytes) -> None:
        if self.fd is None:
            return
        try:
            os.write(self.fd, data)
        except BlockingIOError:
            pass  # pulse not draining fast enough — drop, keep realtime
        except OSError:
            self._reopen()

    def _reopen(self) -> None:
        try:
            if self.fd is not None:
                os.close(self.fd)
        except OSError:
            pass
        try:
            self.fd = os.open(self.fifo, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            self.fd = None

    def unload(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        if self._prev_default:
            _run("pactl", "set-default-source", self._prev_default)
            self._prev_default = None
        if self.module_id is not None:
            _run("pactl", "unload-module", str(self.module_id))
            self.module_id = None
        try:
            os.unlink(self.fifo)
        except OSError:
            pass


def pump(host: str, port: int, source: PipeSource, stop: list[bool]) -> None:
    """One connection lifecycle: header check, then PCM to the FIFO."""
    with socket.create_connection((host, port), timeout=5.0) as conn:
        conn.settimeout(None)
        header = b""
        while len(header) < HEADER:
            part = conn.recv(HEADER - len(header))
            if not part:
                raise ConnectionError("audio adapter closed during header")
            header += part
        magic, rate, channels, bps = struct.unpack(">4sIII", header)
        if magic != MAGIC or rate != source.rate or channels != 1 or bps != 2:
            raise ConnectionError(
                f"unexpected audio header magic={magic!r} rate={rate} ch={channels} bps={bps}"
            )
        while not stop[0]:
            data = conn.recv(8192)
            if not data:
                raise ConnectionError("audio adapter disconnected")
            source.write(data)


def main() -> int:
    host = os.environ.get("GENESIS_ANDROID_MIC_HOST", "127.0.0.1")
    port = int(os.environ.get("GENESIS_ANDROID_MIC_PORT", "8766"))
    name = os.environ.get("GENESIS_ANDROID_MIC_SOURCE", "genesis_mic")
    rate = int(os.environ.get("GENESIS_ANDROID_MIC_RATE", "16000"))

    stop = [False]

    def _sig(_signum: int, _frame: object) -> None:
        stop[0] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    source = PipeSource(name, rate)
    source.load()
    print(f"[audio-relay] pipe source '{name}' ready at {source.fifo} "
          f"(module={source.module_id})", flush=True)
    try:
        while not stop[0]:
            try:
                pump(host, port, source, stop)
            except (ConnectionError, OSError) as e:
                if not stop[0]:
                    print(f"[audio-relay] {e}; reconnecting", flush=True)
                    time.sleep(2.0)
    finally:
        source.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())

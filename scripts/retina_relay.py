#!/usr/bin/env python3
"""Retinal relay — a second afferent path into /genesis_retina.

The canonical `retina` binary negotiates raw formats (RGB3/YUYV) on a
V4L2 device it owns. When that device is unavailable — held by another
capture process, or offering only compressed formats — this relay feeds
the same shared-memory segment from an alternative source:

  - a local V4L2 device that only speaks MJPEG, or
  - an HTTP MJPEG stream (e.g. a surveillance daemon's localhost feed)

ffmpeg decodes to packed RGB24 frames; the relay writes them into the
/genesis_retina segment using the same 28-byte header + seqlock protocol
the Rust retina publishes (`src/bin/retina.rs`), so the optic nerve and
the mind's optics layer see an ordinary eye.

Usage:
    retina_relay.py [source]

  source defaults to /dev/video2. Sources ending in :// are treated as
  HTTP MJPEG streams; anything else is a V4L2 device.
"""

import mmap
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

SHM_PATH = "/dev/shm/genesis_retina"
WIDTH, HEIGHT = 640, 480
BPP = 3
STRIDE = WIDTH * BPP
FRAME_SIZE = STRIDE * HEIGHT
HEADER_SIZE = 28
SEG_SIZE = HEADER_SIZE + FRAME_SIZE
FOURCC = b"RGB3"

HEADER_FMT = "<QII4sII"  # seq, width, height, fourcc, size, stride


def _ffmpeg_cmd(source: str) -> list[str]:
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if "://" in source:
        return [
            *base,
            "-f",
            "mjpeg",
            "-reconnect",
            "1",
            "-reconnect_streamed",
            "1",
            "-i",
            source,
            *_out(),
        ]
    return [
        *base,
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-i",
        source,
        *_out(),
    ]


def _out() -> list[str]:
    return [
        "-vf",
        f"scale={WIDTH}:{HEIGHT}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"

    # Own the segment like the Rust retina does: drop any stale object,
    # create fresh, keep it alive for readers while we run.
    Path(SHM_PATH).unlink(missing_ok=True)
    fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SEG_SIZE)
    seg = mmap.mmap(fd, SEG_SIZE, mmap.MAP_SHARED)
    os.close(fd)

    # Header is written once; the seq counter is rewritten per frame.
    struct.pack_into(HEADER_FMT, seg, 0, 0, WIDTH, HEIGHT, FOURCC, FRAME_SIZE, STRIDE)

    proc = subprocess.Popen(_ffmpeg_cmd(source), stdout=subprocess.PIPE, bufsize=FRAME_SIZE)
    assert proc.stdout is not None

    def _stop(*_args: object) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        seg.close()
        Path(SHM_PATH).unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"retina-relay: {source} -> {SHM_PATH} ({WIDTH}x{HEIGHT} RGB3)")
    seq = 0
    while True:
        frame = proc.stdout.read(FRAME_SIZE)
        if len(frame) < FRAME_SIZE:
            if proc.poll() is not None:
                print("retina-relay: source ended, exiting", file=sys.stderr)
                _stop()
            time.sleep(0.05)
            continue
        # Seqlock: odd = write in progress, even = committed.
        seq += 1
        struct.pack_into("<Q", seg, 0, seq)
        seg[HEADER_SIZE : HEADER_SIZE + FRAME_SIZE] = frame
        seq += 1
        struct.pack_into("<Q", seg, 0, seq)


if __name__ == "__main__":
    sys.exit(main())

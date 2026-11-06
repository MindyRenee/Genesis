#!/usr/bin/env python3
"""Optic relay — a second afferent path into /genesis_optic_nerve.

The body's own capture limb (``src/daemon/optic_nerve.rs``) negotiates
raw formats (RGB3/YUYV) on a V4L2 device it owns. When that device is
unavailable — held by another capture process, or offering only
compressed formats — this relay feeds the same shared-memory segment
from an alternative source:

  - a local V4L2 device that only speaks MJPEG, or
  - an HTTP MJPEG stream (e.g. a surveillance daemon's localhost feed)

ffmpeg decodes to packed RGB24 frames; the relay writes them into the
/genesis_optic_nerve segment using the same 28-byte header + seqlock
protocol the capture limb publishes, so the body's subcortical read and
the mind's optics layer see an ordinary eye.

Usage:
    optic_relay.py [source] [shm_name]

  source defaults to /dev/video2. Sources ending in :// are treated as
  HTTP MJPEG streams; ``x11`` captures the whole display, ``x11:WxH+X+Y``
  captures a region of it, and ``x11win:ID:WxH`` captures a window's own
  pixels by id, cropped to WxH (the screen as an eye); anything else is
  a V4L2 device.

  shm_name defaults to genesis_optic_nerve; pass a different name to
  publish a second eye (e.g. a screen-capture nerve) without
  displacing the physical camera's segment.
"""

import mmap
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

WIDTH, HEIGHT = 640, 480
BPP = 3
STRIDE = WIDTH * BPP
FRAME_SIZE = STRIDE * HEIGHT
HEADER_SIZE = 28
SEG_SIZE = HEADER_SIZE + FRAME_SIZE

HEADER_FMT = "<QII4sII"  # seq, width, height, fourcc, size, stride

# ffmpeg's rgb24 writes R,G,B bytes in memory. The nerve header uses
# V4L2 fourcc names, which are word-order: bytes already in R,G,B
# order are BGR3, and RGB3 would mean B,G,R bytes (the reader flips
# those). Tag honestly or every downstream channel is swapped.
_WIRE_FOURCC = b"BGR3"


def _screen_size() -> str:
    """The root window's WxH — the full desktop across every monitor.

    ``xdpyinfo`` reports the screen's bounding box directly; ``xrandr``
    is the fallback (its first line carries the same span). Failing
    both — a headless Xvfb with no tools — 1920x1080 is the last resort;
    x11grab crops or pads the nerve frame either way.
    """
    for cmd in (["xdpyinfo"], ["xrandr"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        m = re.search(r"dimensions:\s*(\d+)x(\d+)", out) or re.search(
            r"current\s+(\d+)\s*x\s*(\d+)", out
        )
        if m:
            return f"{m.group(1)}x{m.group(2)}"
    return "1920x1080"


def _ffmpeg_cmd(source: str) -> list[str]:
    """Build the ffmpeg pipeline for the given source.

    ``x11:WxH+X+Y`` is a screen region captured like a camera
    pointed at the monitor, scaled to fit 640x480 with padding;
    ``://`` sources are HTTP MJPEG streams; anything else is a V4L2
    device opened in MJPEG mode.
    """
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if source.startswith("x11win:"):
        # x11win:ID:WxH — a window's own pixels by id, cropped to WxH
        # from its top-left. No screen geometry: the grab follows the
        # window wherever the WM puts it and never picks up a stray
        # titlebar or an occluding neighbor's pixels.
        spec = source[7:]
        wid, _, size = spec.partition(":")
        w, _, h = size.partition("x")
        display = os.environ.get("DISPLAY", ":0.0")
        return [
            *base,
            "-f",
            "x11grab",
            "-framerate",
            "30",
            "-window_id",
            wid,
            "-i",
            display,
            "-vf",
            f"crop={w}:{h}:0:0,"
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:flags=neighbor,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    if source == "x11" or source.startswith("x11:"):
        # x11 or x11:WxH+X+Y — the display (or a rect of it), captured
        # like a camera pointed at the monitor. Aspect is preserved
        # into the 640x480 frame with black padding so readers can map
        # the region back deterministically. Bare ``x11`` takes the
        # whole root window.
        if source == "x11":
            spec = "+".join((_screen_size(), "0+0"))
        else:
            spec = source[4:]
        size, _, offset = spec.partition("+")
        w, _, h = size.partition("x")
        # The offset is X+Y (per the usage above); a comma-separated
        # X,Y form slips through to the same place.
        x, sep, y = offset.partition("+")
        if not sep:
            x, _, y = offset.partition(",")
        display = os.environ.get("DISPLAY", ":0.0")
        return [
            *base,
            "-f",
            "x11grab",
            "-framerate",
            "30",
            "-video_size",
            f"{w}x{h}",
            "-i",
            f"{display}+{x},{y}",
            "-vf",
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:flags=neighbor,"
            f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
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
    """The shared output tail: scale to 640x480, emit packed RGB24."""
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
    """Own the nerve segment and pump frames until stopped.

    Parses ``source`` and ``shm_name`` from argv, creates the POSIX
    shared memory with the capture limb's header layout, and loops:
    read one RGB24 frame from ffmpeg's stdout, seqlock-write it.
    """
    source = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"
    shm_name = sys.argv[2] if len(sys.argv) > 2 else "genesis_optic_nerve"
    shm_path = f"/dev/shm/{shm_name}"

    # Own the segment like the capture limb does: drop any stale
    # object, create fresh, keep it alive for readers while we run.
    Path(shm_path).unlink(missing_ok=True)
    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, SEG_SIZE)
    seg = mmap.mmap(fd, SEG_SIZE, mmap.MAP_SHARED)
    os.close(fd)

    # Header is written once; the seq counter is rewritten per frame.
    struct.pack_into(HEADER_FMT, seg, 0, 0, WIDTH, HEIGHT, _WIRE_FOURCC, FRAME_SIZE, STRIDE)

    proc = subprocess.Popen(_ffmpeg_cmd(source), stdout=subprocess.PIPE, bufsize=FRAME_SIZE)
    assert proc.stdout is not None

    def _stop(*_args: object) -> None:
        """Signal handler: stop ffmpeg, close the segment, unlink it."""
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        seg.close()
        Path(shm_path).unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"optic-relay: {source} -> {shm_path} ({WIDTH}x{HEIGHT} rgb24)")
    seq = 0
    while True:
        frame = proc.stdout.read(FRAME_SIZE)
        if len(frame) < FRAME_SIZE:
            if proc.poll() is not None:
                print("optic-relay: source ended, exiting", file=sys.stderr)
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

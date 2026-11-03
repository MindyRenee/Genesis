#!/usr/bin/env python3
"""Feed Android camera stills into Genesis's full scene-perception pipeline.

Requires Termux:API and Pillow. This deliberately uses the direct RGB-frame
entry point rather than the desktop-only shared-memory optic relay.

Example:
    export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
    python scripts/genesis_android_scene_vision.py --camera-id 1 --interval 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from occipital_lobe.perception.vision import Vision


def capture_rgb(camera_id: int, destination: Path, timeout: float) -> np.ndarray:
    """Capture one Termux camera still and return a resized RGB uint8 array."""
    result = subprocess.run(
        ["termux-camera-photo", "-c", str(camera_id), str(destination)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"termux-camera-photo exited {result.returncode}")
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("camera command produced no image file")

    with Image.open(destination) as image:
        image = image.convert("RGB")
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise RuntimeError(f"unexpected RGB image shape: {rgb.shape}")
    return rgb


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture Android camera stills and process them as Genesis VisionScenes."
    )
    parser.add_argument("--camera-id", type=int, default=1, help="Termux camera ID (default: 1)")
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Minimum delay between capture attempts in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many successful captures; 0 runs until interrupted",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=30.0,
        help="Per-capture timeout in seconds (default: 30)",
    )
    args = parser.parse_args()
    if args.camera_id < 0 or args.interval < 0 or args.camera_timeout <= 0:
        parser.error("camera-id and interval must be non-negative; camera-timeout must be positive")
    if args.max_frames < 0:
        parser.error("max-frames must be non-negative")

    vision = Vision()
    successful = 0
    failures = 0
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "started",
                "pipeline": "Vision.see_frame",
                "camera_id": args.camera_id,
                "interval_s": args.interval,
                "max_frames": args.max_frames,
            }
        ),
        flush=True,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="genesis-android-vision-") as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"
            while args.max_frames == 0 or successful < args.max_frames:
                frame_started = time.monotonic()
                try:
                    rgb = capture_rgb(args.camera_id, image_path, args.camera_timeout)
                    capture_ms = (time.monotonic() - frame_started) * 1000
                    scene_started = time.monotonic()
                    scene = vision.see_frame(rgb, nerve="android")
                    processing_ms = (time.monotonic() - scene_started) * 1000
                    successful += 1
                    print(
                        json.dumps(
                            {
                                "event": "scene",
                                "frame": successful,
                                "status": scene.status,
                                "nerve": scene.nerve,
                                "shape": list(rgb.shape),
                                "capture_ms": round(capture_ms, 1),
                                "processing_ms": round(processing_ms, 1),
                                "scene": scene.as_metadata(),
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                        flush=True,
                    )
                except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
                    failures += 1
                    print(
                        json.dumps(
                            {
                                "event": "frame_error",
                                "failures": failures,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                elapsed = time.monotonic() - frame_started
                time.sleep(max(0.0, args.interval - elapsed))
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)

    print(
        json.dumps(
            {
                "event": "stopped",
                "successful_frames": successful,
                "capture_or_processing_failures": failures,
                "elapsed_s": round(time.monotonic() - started, 1),
            }
        ),
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Opt-in Android camera input through Genesis's native visual pathway."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time

import numpy as np
from brainstem.android_media import capture_android_frame
from occipital_lobe.visual_pathway import VisualPathway
from PIL import Image

logger = logging.getLogger("genesis.android_vision")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=640)
    args = parser.parse_args()

    if os.environ.get("GENESIS_ANDROID_CAMERA") != "1":
        parser.error("Set GENESIS_ANDROID_CAMERA=1 to enable camera access.")
    if args.interval < 2:
        parser.error("--interval must be at least 2 seconds.")
    if args.camera_id < 0:
        parser.error("--camera-id must be non-negative.")
    if args.max_width < 64:
        parser.error("--max-width must be at least 64 pixels.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pathway = VisualPathway()
    stop = threading.Event()
    counts = {"captured": 0, "processed": 0, "capture_failures": 0, "processing_errors": 0}

    def request_stop(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    logger.info(
        "Starting synchronous camera loop: id=%d interval=%.1fs max_width=%d",
        args.camera_id,
        args.interval,
        args.max_width,
    )

    try:
        while not stop.is_set():
            cycle_started = time.monotonic()

            try:
                frame = capture_android_frame(
                    camera_id=args.camera_id,
                    timeout=30.0,
                )
            except Exception:
                counts["capture_failures"] += 1
                logger.exception("Camera capture raised an exception")
                frame = None

            if frame is None:
                counts["capture_failures"] += 1
                logger.warning(
                    "No frame returned; capture_failures=%d",
                    counts["capture_failures"],
                )
            else:
                counts["captured"] += 1
                try:
                    image = Image.fromarray(frame)
                    if image.width > args.max_width:
                        height = max(1, round(image.height * args.max_width / image.width))
                        image = image.resize((args.max_width, height), Image.Resampling.BILINEAR)
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

                    logger.info(
                        "Captured frame=%d shape=%s; processing",
                        counts["captured"],
                        rgb.shape,
                    )
                    output = pathway.process_frame(rgb)
                    counts["processed"] += 1
                    logger.info(
                        "PROCESSED frame=%d pathway_counter=%d "
                        "processing_ms=%.1f dorsal=%s ventral=%s",
                        counts["processed"],
                        output.frame_counter,
                        output.processing_time_ms,
                        output.dorsal_field is not None,
                        output.ventral_field is not None,
                    )
                except Exception:
                    counts["processing_errors"] += 1
                    logger.exception("Frame processing failed")

            remaining = args.interval - (time.monotonic() - cycle_started)
            stop.wait(max(0.0, remaining))
    finally:
        logger.info(
            "Stopped: captured=%d processed=%d capture_failures=%d processing_errors=%d",
            counts["captured"],
            counts["processed"],
            counts["capture_failures"],
            counts["processing_errors"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

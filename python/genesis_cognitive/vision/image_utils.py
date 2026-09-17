"""Image loading and resizing utilities for the occipital subsystem."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> np.ndarray | None:
    """Load an image file as an RGB uint8 numpy array.

    Supports any format PIL can read (PNG, JPEG, BMP, GIF, etc.).
    Converts grayscale to RGB. Returns None on failure.

    Args:
        path: path to the image file.

    Returns:
        H×W×3 uint8 RGB array, or None.
    """
    try:
        from PIL import Image
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to load image {path}: {e}")
        return None


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes (e.g. from a URL) into an RGB uint8 array.

    Args:
        data: raw image bytes (JPEG, PNG, etc.).

    Returns:
        H×W×3 uint8 RGB array, or None.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Failed to decode image bytes: {e}")
        return None


def resize_for_vision(
    frame: np.ndarray,
    max_dim: int = 320,
) -> np.ndarray:
    """Resize an image to be no larger than max_dim on any side.

    This keeps processing fast while preserving aspect ratio.
    V1's 8×8 patches work best on images in the 120-320 pixel range.

    Args:
        frame: H×W×3 uint8 image.
        max_dim: maximum dimension in pixels.

    Returns:
        Resized H×W×3 uint8 image.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame

    scale = max_dim / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    try:
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception:  # noqa: BLE001
        # Fallback: simple stride-based resize
        step_h = max(1, h // new_h)
        step_w = max(1, w // new_w)
        return frame[::step_h, ::step_w]

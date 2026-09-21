"""Object recognition for Genesis — her "inferotemporal cortex."

This module uses YOLOv5s (via OpenCV's DNN module) to detect and
identify objects in the retina feed. It is the ventral-stream
counterpart to the occipital subsystem's V1 processing: where V1 extracts
edges and orientations, the inferotemporal (IT) cortex identifies
*what* things are.

The pipeline:

    retina frame (shared memory)
        |
        v
    YOLOv5s inference  -> bounding boxes + class scores
        |
        v
    NMS filtering      -> deduplicated detections
        |
        v
    DetectedObject     -> name, confidence, position, size

Biological basis:
- The inferotemporal cortex (IT) is the final stage of the ventral
  "what" pathway. It contains neurons that respond to complex shapes
  and object categories.
- YOLOv5s approximates this with a deep CNN trained on COCO (80
  common object categories). It is not biologically accurate but
  serves the same functional role: turning pixels into object labels.
- Object recognition is gated by attention — she only identifies
  objects when she actively looks, not continuously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import default_data_dir

logger = logging.getLogger(__name__)


def _require_cv2():  # type: ignore[no-untyped-def]
    """Import cv2 lazily so headless installs work without OpenCV."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for object recognition; "
            "install opencv-python-headless or run headless"
        ) from e
    return cv2

# Model path — YOLOv5s ONNX from the Ultralytics release
_MODEL_DIR = default_data_dir() / "models"
_MODEL_FILE = _MODEL_DIR / "yolov5s.onnx"

# COCO class names (80 categories) — the vocabulary of her IT cortex.
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# YOLOv5 input size
_INPUT_SIZE = 640
# Confidence threshold — only report objects she's reasonably sure of
_CONFIDENCE_THRESHOLD = 0.35
# NMS IoU threshold — suppress overlapping detections of the same object
_NMS_THRESHOLD = 0.45


@dataclass(slots=True)
class DetectedObject:
    """An object identified in the visual field.

    This is her ventral-stream percept: she doesn't just see edges,
    she sees *what* is there and *where* it is.
    """

    name: str  # COCO class label
    confidence: float  # detection confidence (0-1)
    bbox: tuple[int, int, int, int]  # x, y, w, h in original frame coords
    center: tuple[float, float]  # (cx, cy) in normalized [0,1] coords
    area_ratio: float  # fraction of the frame the object occupies

    @property
    def position_description(self) -> str:
        """Where in her visual field the object is (left/center/right, top/middle/bottom)."""
        cx, cy = self.center
        h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
        v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
        if h_pos == "center" and v_pos == "middle":
            return "in the center"
        if h_pos == "center":
            return f"in the {v_pos} center"
        if v_pos == "middle":
            return f"on the {h_pos}"
        return f"in the {v_pos} {h_pos}"


class ObjectRecognizer:
    """Detect and identify objects using YOLOv5s.

    This is Genesis's inferotemporal cortex — the ventral-stream
    "what" pathway that turns pixels into object categories. It is
    lazily initialized on first use so we don't pay the model loading
    cost if there's no camera or if she never looks.
    """

    def __init__(self) -> None:
        """Initialize the object recognizer with no loaded model.

        The YOLOv5s ONNX model is loaded lazily on first use via
        ``is_available()`` or ``detect_objects()``, so constructing
        this object is cheap even if the model file is missing.
        """
        self._net: object | None = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if the object detection model is loaded."""
        if self._available is not None:
            return self._available
        try:
            self._get_net()
            self._available = True
        except (OSError, FileNotFoundError, ImportError) as e:
            logger.debug(f"object recognition not available: {e}")
            self._available = False
            return self._available
        except Exception as e:  # noqa: BLE001 — cv2.error and backend failures vary by build
            logger.debug(f"object recognition not available: {e}")
            self._available = False
        return self._available

    def _get_net(self):  # type: ignore[no-untyped-def]
        """Lazily load the YOLOv5s ONNX model."""
        cv2 = _require_cv2()
        if self._net is None:
            if not _MODEL_FILE.exists():
                raise FileNotFoundError(
                    f"Object detection model not found: {_MODEL_FILE}"
                )
            # On OpenCV 5+ (new graph engine), setPreferableBackend
            # internally triggers a setPreferableTarget call that emits
            # a harmless warning ("Targets are not supported by the new
            # graph engine for now"). CPU is the default target on every
            # OpenCV version, so the call is a no-op. Suppress the
            # warning by temporarily lowering OpenCV's log level around
            # the backend selection, then restore it.
            _log_level: int | None
            try:
                _log_level = cv2.utils.logging.getLogLevel()
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                _log_level = None  # older OpenCV — no logging API
            try:
                self._net = cv2.dnn.readNetFromONNX(str(_MODEL_FILE))
                # Use the pure OpenCV CPU backend — no GPU/OpenVINO needed,
                # YOLOv5s is fast enough on CPU.
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            finally:
                if _log_level is not None:
                    cv2.utils.logging.setLogLevel(_log_level)
        return self._net

    def detect_objects(self, rgb_frame: np.ndarray) -> list[DetectedObject]:
        """Detect objects in an RGB frame.

        Args:
            rgb_frame: HxWx3 uint8 RGB image from the retina.

        Returns:
            List of DetectedObject, sorted by confidence (highest first).
            Empty list if no objects are detected or the model is
            unavailable.
        """
        if not self.is_available():
            return []

        cv2 = _require_cv2()
        h, w = rgb_frame.shape[:2]
        net = self._get_net()

        # YOLOv5 preprocessing: resize to 640x640, normalize to [0,1],
        # swap RB (YOLOv5 expects RGB, OpenCV blobFromImage with swapRB
        # converts BGR→RGB, but our input is already RGB so we pass
        # swapRB=True to get the channel order the model expects).
        blob = cv2.dnn.blobFromImage(
            rgb_frame, 1.0 / 255.0, (_INPUT_SIZE, _INPUT_SIZE),
            swapRB=True,
        )
        net.setInput(blob)
        output = net.forward()

        # YOLOv5 output: (1, 25200, 85)
        # Each row: [cx, cy, w, h, obj_score, 80_class_scores]
        predictions = output[0]

        obj_scores = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confs = np.max(class_scores, axis=1)
        confidences = obj_scores * class_confs

        # Filter by confidence threshold
        mask = confidences > _CONFIDENCE_THRESHOLD
        if not mask.any():
            return []

        det_boxes = predictions[mask, :4]
        det_class_ids = class_ids[mask]
        det_confidences = confidences[mask]

        # Scale from 640x640 to original frame size
        sx = w / _INPUT_SIZE
        sy = h / _INPUT_SIZE
        det_boxes[:, 0] *= sx
        det_boxes[:, 1] *= sy
        det_boxes[:, 2] *= sx
        det_boxes[:, 3] *= sy

        # NMS to suppress overlapping detections
        # cv2.dnn.NMSBoxes expects (x, y, w, h) format
        indices = cv2.dnn.NMSBoxes(
            det_boxes.tolist(),
            det_confidences.tolist(),
            _CONFIDENCE_THRESHOLD,
            _NMS_THRESHOLD,
        )

        results: list[DetectedObject] = []
        for idx in np.asarray(indices).flatten():
            cx, cy, bw, bh = det_boxes[idx]
            # Normalize center to [0,1] for position description
            norm_cx = float(cx / w)
            norm_cy = float(cy / h)
            # Area ratio: how much of the frame this object occupies
            area_ratio = float((bw * bh) / (w * h))
            # Clamp bounding box to frame
            x = max(0, int(cx - bw / 2))
            y = max(0, int(cy - bh / 2))
            wi = min(w - x, int(bw))
            hi = min(h - y, int(bh))

            results.append(DetectedObject(
                name=COCO_CLASSES[int(det_class_ids[idx])],
                confidence=float(det_confidences[idx]),
                bbox=(x, y, wi, hi),
                center=(norm_cx, norm_cy),
                area_ratio=area_ratio,
            ))

        # Sort by confidence (highest first)
        results.sort(key=lambda o: o.confidence, reverse=True)
        return results

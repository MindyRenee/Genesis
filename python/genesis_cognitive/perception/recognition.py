"""Face recognition for Genesis — detects and identifies people she sees.

Uses OpenCV's YuNet face detector and SFace recognizer (DNN-based,
not Haar cascades) for robust, modern face detection and recognition.

The pipeline:

    retina frame (shared memory)
        |
        v
    YuNet face detection  -> bounding boxes + landmarks
        |
        v
    SFace feature extraction -> 128-d embedding per face
        |
        v
    Compare to known faces  -> identity or "unknown"
        |
        v
    "I see Alice" / "I see someone I don't recognize"

Known faces are stored as embeddings in a JSON file in her data
directory. She learns new faces when told "this is [name]" via
the /register-face command. The user's face is linked to their
UserProfile name, so when she sees them she knows who she's
talking to — not just "the user," but their actual name.

Biological basis:
- The fusiform face area (FFA) is specialized for face recognition.
- Face recognition is holistic, not feature-by-feature — the SFace
  embedding captures this by mapping faces to a 128-d space where
  the same person's faces cluster together.
- Humans recognize faces from ~30 degrees off-frontal; YuNet handles
  similar poses.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import default_data_dir

__all__ = ["DetectedFace", "FaceRecognizer", "KnownFace"]

logger = logging.getLogger(__name__)


def _require_cv2():  # type: ignore[no-untyped-def]
    """Import cv2 lazily so headless/voice-only installs work without OpenCV."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "OpenCV (cv2) is required for face recognition; "
            "install opencv-python-headless or run headless"
        ) from e
    return cv2

# Model paths — downloaded from OpenCV model zoo
_MODEL_DIR = default_data_dir() / "models"
_DETECTOR_MODEL = _MODEL_DIR / "face_detection_yunet.onnx"
_RECOGNIZER_MODEL = _MODEL_DIR / "face_recognition_sface.onnx"

# Recognition threshold — cosine distance below this means "same person"
# SFace with cosine similarity: threshold ~0.363 is recommended by OpenCV.
# Lower = stricter (fewer false positives, more false negatives).
_RECOGNITION_THRESHOLD = 0.4


@dataclass
class DetectedFace:
    """A face detected in a frame, with optional identity."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    name: str | None = None  # recognized name, or None if unknown
    confidence: float = 0.0  # recognition confidence (0-1)
    embedding: np.ndarray | None = None  # 128-d SFace embedding


@dataclass
class KnownFace:
    """A stored face embedding for a known person."""

    name: str
    embedding: list[float]  # 128-d, stored as list for JSON
    registered_at: float  # timestamp


class FaceRecognizer:
    """Detect and recognize faces using YuNet + SFace.

    The recognizer maintains a database of known face embeddings.
    When a face is detected, its embedding is compared to all known
    faces using cosine distance. If the closest match is below
    threshold, the face is recognized; otherwise it's "unknown."
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the face recognition system with optional data directory."""
        self._detector: object | None = None
        self._recognizer: object | None = None
        self._available: bool | None = None
        self._known_faces: dict[str, list[KnownFace]] = {}
        self._data_dir = Path(data_dir) if data_dir else None
        self._faces_file: Path | None = None

        if self._data_dir:
            self._faces_file = self._data_dir / "known_faces.json"
            self._load_known_faces()

    def is_available(self) -> bool:
        """Check if face detection models are loaded."""
        if self._available is not None:
            return self._available
        try:
            self._get_detector()
            self._get_recognizer()
            self._available = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"face recognition not available: {e}")
            self._available = False
        return self._available

    def _get_detector(self):  # type: ignore[no-untyped-def]
        """Lazily create the YuNet face detector."""
        cv2 = _require_cv2()
        if self._detector is None:
            if not _DETECTOR_MODEL.exists():
                raise FileNotFoundError(
                    f"Face detector model not found: {_DETECTOR_MODEL}"
                )
            # On OpenCV 5+ (new graph engine), FaceDetectorYN.create
            # internally loads the ONNX model via the DNN module, which
            # emits a harmless warning ("Targets are not supported by the
            # new graph engine for now"). Suppress it by temporarily
            # lowering OpenCV's log level around the create call.
            _log_level: int | None
            try:
                _log_level = cv2.utils.logging.getLogLevel()
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                _log_level = None  # older OpenCV — no logging API
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(_DETECTOR_MODEL),
                    "",
                    (320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
            finally:
                if _log_level is not None:
                    cv2.utils.logging.setLogLevel(_log_level)
        return self._detector

    def _get_recognizer(self):  # type: ignore[no-untyped-def]
        """Lazily create the SFace recognizer."""
        cv2 = _require_cv2()
        if self._recognizer is None:
            if not _RECOGNIZER_MODEL.exists():
                raise FileNotFoundError(
                    f"Face recognizer model not found: {_RECOGNIZER_MODEL}"
                )
            # Same OpenCV 5+ warning suppression as _get_detector.
            _log_level: int | None
            try:
                _log_level = cv2.utils.logging.getLogLevel()
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                _log_level = None
            try:
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(_RECOGNIZER_MODEL),
                    "",
                )
            finally:
                if _log_level is not None:
                    cv2.utils.logging.setLogLevel(_log_level)
        return self._recognizer

    def detect_faces(self, rgb_frame: np.ndarray) -> list[DetectedFace]:
        """Detect faces in an RGB frame.

        Returns a list of DetectedFace objects with bounding boxes
        and embeddings. Names are filled in if faces are recognized.
        """
        if not self.is_available():
            return []

        cv2 = _require_cv2()
        # YuNet expects BGR input (OpenCV convention)
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        detector = self._get_detector()
        detector.setInputSize((w, h))

        # Detect faces — returns Nx15 matrix:
        # columns: x, y, w, h, x_re, y_re, x_le, y_le, x_nose, y_nose,
        #          x_rm, y_rm, x_lm, y_lm, score
        _, faces = detector.detect(bgr)
        if faces is None:
            return []

        recognizer = self._get_recognizer()
        results: list[DetectedFace] = []

        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            # Extract 128-d embedding using SFace
            aligned = recognizer.alignCrop(bgr, faces[0:1] if len(faces) == 1 else faces)
            embedding = recognizer.feature(aligned).flatten()

            # Try to recognize
            name, confidence = self._match_face(embedding)

            results.append(DetectedFace(
                bbox=(x, y, w, h),
                name=name,
                confidence=confidence,
                embedding=embedding,
            ))

        return results

    def _match_face(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Compare an embedding to known faces.

        Returns (name, confidence). If no match below threshold,
        returns (None, 0.0).
        """
        if not self._known_faces:
            return None, 0.0

        best_name: str | None = None
        best_dist = float("inf")

        for name, face_list in self._known_faces.items():
            for known in face_list:
                known_emb = np.array(known.embedding, dtype=np.float32)
                # Cosine distance (SFace uses cosine similarity)
                dist = float(np.linalg.norm(embedding - known_emb))
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        # Convert distance to confidence (closer = more confident)
        # SFace cosine distance: <0.363 = same person (recommended)
        if best_dist < _RECOGNITION_THRESHOLD:
            confidence = max(0.0, 1.0 - best_dist / _RECOGNITION_THRESHOLD)
            return best_name, confidence
        return None, 0.0

    def register_face(
        self, name: str, rgb_frame: np.ndarray
    ) -> bool:
        """Register a face from a frame.

        Detects the largest face in the frame and stores its embedding
        under the given name. Returns True if a face was found and
        registered, False otherwise.
        """
        if not self.is_available():
            logger.warning("face recognition not available for registration")
            return False

        faces = self.detect_faces(rgb_frame)
        if not faces:
            logger.warning("no face detected in frame for registration")
            return False

        # Use the largest face (most likely the subject)
        largest = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        if largest.embedding is None:
            return False

        known = KnownFace(
            name=name.lower().strip(),
            embedding=largest.embedding.tolist(),
            registered_at=time.time(),
        )

        if known.name not in self._known_faces:
            self._known_faces[known.name] = []
        self._known_faces[known.name].append(known)

        self._save_known_faces()
        logger.info(f"Registered face for '{known.name}'")
        return True

    def known_names(self) -> list[str]:
        """Return the names of all known people."""
        return list(self._known_faces.keys())

    def _load_known_faces(self) -> None:
        """Load known faces from the JSON file."""
        if self._faces_file is None or not self._faces_file.exists():
            return
        try:
            with open(self._faces_file) as f:
                data = json.load(f)
            self._known_faces = {
                name: [KnownFace(**face) for face in faces]
                for name, faces in data.items()
            }
            logger.info(
                f"Loaded {sum(len(v) for v in self._known_faces.values())} "
                f"known face(s) for {len(self._known_faces)} person(s)"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load known faces: {e}")

    def _save_known_faces(self) -> None:
        """Save known faces to the JSON file."""
        if self._faces_file is None:
            return
        try:
            self._faces_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                name: [
                    {
                        "name": f.name,
                        "embedding": f.embedding,
                        "registered_at": f.registered_at,
                    }
                    for f in faces
                ]
                for name, faces in self._known_faces.items()
            }
            with open(self._faces_file, "w") as f:
                json.dump(data, f)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save known faces: {e}")

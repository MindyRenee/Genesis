"""Face recognition for Genesis — detects and identifies people it sees.

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

Known faces are stored as embeddings in a JSON file in its data
directory. It learns new faces when told "this is [name]" via
the /register-face command. The user's face is linked to their
UserProfile name, so when it sees them it knows who it's
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
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from genesis_client.swallow import note_swallowed
from infrastructure.config import default_data_dir

__all__ = ["DetectedFace", "FaceRecognizer", "KnownFace"]

logger = logging.getLogger(__name__)


def _write_json_atomically(path: Path, data: object) -> None:
    """Replace learned face data only after the complete file is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Atomic-write cleanup: remove the temp file, then re-raise so the
        # caller still sees the failure (and Ctrl+C/SystemExit propagate —
        # Exception never catches those).
        try:
            os.unlink(tmp_path)
        except OSError as cleanup_e:
            note_swallowed(
                "occipital_lobe.perception.recognition._write_json_atomically",
                cleanup_e,
            )
        raise


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

# Recognition threshold — cosine similarity above this means "same person"
# SFace: OpenCV recommends a cosine threshold of ~0.363.
# Higher = stricter (fewer false positives, more false negatives).
_RECOGNITION_THRESHOLD = 0.4

# Cap on stored embeddings per name — each registration appends one,
# so without a bound the JSON grows on every re-registration.
_MAX_EMBEDDINGS_PER_NAME = 20

# Anonymous cluster matching — looser than the named threshold because
# a cluster accumulates embeddings from every sighting (more chances to
# hit), and a false merge pollutes a stranger's signature. Keep it
# stricter instead: "familiar" must mean *the same* face.
_CLUSTER_THRESHOLD = 0.45

# Cap on live anonymous clusters — crowd scenes must not grow the
# registry without bound. Least-recently-seen evicts.
_MAX_CLUSTERS = 64

# A cluster must be seen this many times before it persists to disk —
# one-shot detections are usually false positives (shadows, posters,
# faces on screens) and must not become durable "people".
_MIN_PERSISTED_SIGHTINGS = 2


@dataclass
class DetectedFace:
    """A face detected in a frame, with optional identity."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    name: str | None = None  # recognized name, or None if unknown
    confidence: float = 0.0  # recognition confidence (0-1)
    embedding: np.ndarray | None = None  # 128-d SFace embedding
    # Person key for the dyadic signature: the name when recognized,
    # else the stable anonymous cluster id ("face:3"). An unnamed face
    # is still *someone* — the key lets a familiar stranger accumulate
    # a signature before any introduction.
    person_key: str | None = None
    # "named" | "familiar" (seen cluster, no name) | "new" (first sighting)
    tier: str = "new"


@dataclass
class FaceCluster:
    """An anonymous-but-recognized face — a person without a name.

    Unknown faces are not noise: a recurring stranger should become
    *familiar*. Unmatched embeddings cluster here under stable
    "face:N" keys so presence accrues to them in the presence model
    exactly as it does for named people.
    """

    cluster_id: str  # "face:3"
    embeddings: list[list[float]] = field(default_factory=list)
    sightings: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


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
        # cv2 DNN objects are not thread-safe: the presence sampler,
        # see(), and look_at_image can all reach this recognizer from
        # different threads, and a concurrent detect()/feature() pair
        # corrupts the engine's internal buffers (observed as the
        # OpenCV 5 forwardGraph "buf.shape() == m.shape()" assert).
        # Every cv2 touch in detect_faces runs under this lock.
        self._infer_lock = threading.Lock()
        self._available: bool | None = None
        self._known_faces: dict[str, list[KnownFace]] = {}
        # Anonymous face clusters — stable "face:N" identities for
        # people seen but never named. Persisted beside known_faces so
        # a familiar stranger survives restarts.
        self._clusters: dict[str, FaceCluster] = {}
        self._next_cluster_id = 1
        self._data_dir = Path(data_dir) if data_dir else None
        self._faces_file: Path | None = None
        self._clusters_file: Path | None = None

        if self._data_dir:
            self._faces_file = self._data_dir / "known_faces.json"
            self._clusters_file = self._data_dir / "face_clusters.json"
            self._load_known_faces()
            self._load_clusters()

    def is_available(self) -> bool:
        """Check if face detection models are loaded."""
        if self._available is not None:
            return self._available
        try:
            self._get_detector()
            self._get_recognizer()
            self._available = True
        except (OSError, FileNotFoundError, ImportError) as e:
            # Face models not installed — a capability probe answering
            # "absent", not a swallowed failure.
            logger.debug("face recognition unavailable: %s", e)
            self._available = False
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "occipital_lobe.perception.recognition.is_available",
                e,
            )
            self._available = False
        return self._available

    def _get_detector(self):  # type: ignore[no-untyped-def]
        """Lazily create the YuNet face detector."""
        cv2 = _require_cv2()
        if self._detector is None:
            if not _DETECTOR_MODEL.exists():
                raise FileNotFoundError(f"Face detector model not found: {_DETECTOR_MODEL}")
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
                raise FileNotFoundError(f"Face recognizer model not found: {_RECOGNIZER_MODEL}")
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

        with self._infer_lock:
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

                # Extract 128-d embedding using SFace. alignCrop takes one
                # face row (15 values); passing the whole N×15 matrix would
                # align every embedding to the first face's landmarks.
                aligned = recognizer.alignCrop(bgr, face[None])
                embedding = recognizer.feature(aligned).flatten()

                # Try to recognize — named first, then anonymous clusters.
                # Either way the face leaves with a stable person key.
                name, confidence = self._match_face(embedding)
                if name is not None:
                    person_key, tier = name, "named"
                else:
                    person_key, tier = self._match_or_cluster(embedding)

                results.append(
                    DetectedFace(
                        bbox=(x, y, w, h),
                        name=name,
                        confidence=confidence,
                        embedding=embedding,
                        person_key=person_key,
                        tier=tier,
                    )
                )

        return results

    def _match_face(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Compare an embedding to known faces.

        Returns (name, confidence). If no match above threshold,
        returns (None, 0.0).
        """
        if not self._known_faces:
            return None, 0.0

        best_name: str | None = None
        best_sim = -1.0
        norm = float(np.linalg.norm(embedding))

        for name, face_list in self._known_faces.items():
            for known in face_list:
                known_emb = np.array(known.embedding, dtype=np.float32)
                known_norm = float(np.linalg.norm(known_emb))
                if norm < 1e-8 or known_norm < 1e-8:
                    continue
                # Cosine similarity (SFace's matching metric — the
                # OpenCV demo thresholds cosine similarity at 0.363;
                # L2 distance is a different, much stricter gate on
                # normalized embeddings).
                sim = float(np.dot(embedding, known_emb) / (norm * known_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        # Convert similarity to confidence (higher = more confident)
        if best_sim >= _RECOGNITION_THRESHOLD:
            confidence = float(
                min(1.0, (best_sim - _RECOGNITION_THRESHOLD) / (1.0 - _RECOGNITION_THRESHOLD))
            )
            return best_name, confidence
        return None, 0.0

    def _match_or_cluster(self, embedding: np.ndarray) -> tuple[str, str]:
        """Match an unnamed face against the anonymous clusters.

        Returns (person_key, tier): "familiar" when the face matches a
        known cluster — the stranger is recognized though unnamed —
        else "new": a fresh "face:N" cluster is minted for them so
        their signature can start accumulating now.
        """
        norm = float(np.linalg.norm(embedding))
        best_id: str | None = None
        best_sim = -1.0
        if norm >= 1e-8:
            for cid, cluster in self._clusters.items():
                for emb in cluster.embeddings:
                    e = np.array(emb, dtype=np.float32)
                    en = float(np.linalg.norm(e))
                    if en < 1e-8:
                        continue
                    sim = float(np.dot(embedding, e) / (norm * en))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = cid

        now = time.time()
        if best_id is not None and best_sim >= _CLUSTER_THRESHOLD:
            cluster = self._clusters[best_id]
            cluster.sightings += 1
            cluster.last_seen = now
            cluster.embeddings.append(embedding.tolist())
            if len(cluster.embeddings) > _MAX_EMBEDDINGS_PER_NAME:
                cluster.embeddings = cluster.embeddings[-_MAX_EMBEDDINGS_PER_NAME:]
            self._save_clusters()
            return best_id, "familiar"

        # First sighting — mint a cluster. Bounded: beyond the cap the
        # least-recently-seen cluster is evicted rather than letting
        # crowd scenes grow the registry without limit.
        if len(self._clusters) >= _MAX_CLUSTERS:
            stale = min(self._clusters, key=lambda c: self._clusters[c].last_seen)
            del self._clusters[stale]
        cid = f"face:{self._next_cluster_id}"
        self._next_cluster_id += 1
        self._clusters[cid] = FaceCluster(
            cluster_id=cid,
            embeddings=[embedding.tolist()],
            sightings=1,
            first_seen=now,
            last_seen=now,
        )
        self._save_clusters()
        return cid, "new"

    def name_face(self, cluster_id: str, name: str) -> str | None:
        """Give a name to an anonymous face cluster.

        The cluster's sightings become the named person's embeddings —
        the familiar stranger was *them* all along. Returns the folded
        cluster id, or None if the cluster is unknown.
        """
        name = name.lower().strip()
        cluster = self._clusters.pop(cluster_id, None)
        if cluster is None or not name:
            return None
        entries = self._known_faces.setdefault(name, [])
        for emb in cluster.embeddings:
            entries.append(KnownFace(name=name, embedding=emb, registered_at=cluster.first_seen))
        if len(entries) > _MAX_EMBEDDINGS_PER_NAME:
            self._known_faces[name] = entries[-_MAX_EMBEDDINGS_PER_NAME:]
        self._save_known_faces()
        self._save_clusters()
        logger.info("Face cluster %s named '%s'", cluster_id, name)
        return cluster_id

    def cluster_sightings(self, cluster_id: str) -> int:
        """How many times this anonymous face has been seen."""
        cluster = self._clusters.get(cluster_id)
        return cluster.sightings if cluster else 0

    def register_face(self, name: str, rgb_frame: np.ndarray) -> bool:
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

        # If the subject was already an anonymous cluster, naming it
        # folds its sightings into the named person rather than
        # starting them over.
        if largest.person_key and largest.person_key.startswith("face:"):
            if self.name_face(largest.person_key, name):
                return True

        known = KnownFace(
            name=name.lower().strip(),
            embedding=largest.embedding.tolist(),
            registered_at=time.time(),
        )

        if known.name not in self._known_faces:
            self._known_faces[known.name] = []
        self._known_faces[known.name].append(known)
        if len(self._known_faces[known.name]) > _MAX_EMBEDDINGS_PER_NAME:
            # Keep the most recent embeddings — the newest capture is
            # the most representative of the person's current look.
            self._known_faces[known.name] = self._known_faces[known.name][
                -_MAX_EMBEDDINGS_PER_NAME:
            ]

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
                name: [KnownFace(**face) for face in faces] for name, faces in data.items()
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
            _write_json_atomically(self._faces_file, data)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save known faces: {e}")

    def _load_clusters(self) -> None:
        """Load anonymous face clusters — the familiar strangers."""
        if self._clusters_file is None or not self._clusters_file.exists():
            return
        try:
            with open(self._clusters_file) as f:
                data = json.load(f)
            self._next_cluster_id = int(data.get("next_id", 1))
            self._clusters = {
                c["cluster_id"]: FaceCluster(
                    cluster_id=c["cluster_id"],
                    embeddings=c.get("embeddings", []),
                    sightings=c.get("sightings", 0),
                    first_seen=c.get("first_seen", time.time()),
                    last_seen=c.get("last_seen", time.time()),
                )
                for c in data.get("clusters", [])
            }
            logger.info("Loaded %d anonymous face cluster(s)", len(self._clusters))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load face clusters: {e}")

    def _save_clusters(self) -> None:
        """Persist anonymous clusters so a familiar stranger survives
        a restart — an unnamed person is still a person.

        Only *confirmed* clusters are durable: a single sighting is
        provisional because the detector also fires on shadows, posters
        and pictures on screens — every persisted registry full of
        one-shot clusters is detector noise being promoted to people.
        A real person recurs within a session and reaches the second
        sighting quickly."""
        if self._clusters_file is None:
            return
        try:
            data = {
                "next_id": self._next_cluster_id,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "embeddings": c.embeddings,
                        "sightings": c.sightings,
                        "first_seen": c.first_seen,
                        "last_seen": c.last_seen,
                    }
                    for c in self._clusters.values()
                    if c.sightings >= _MIN_PERSISTED_SIGHTINGS
                ],
            }
            _write_json_atomically(self._clusters_file, data)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save face clusters: {e}")

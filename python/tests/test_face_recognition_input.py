"""Face identity input must remain photographic and canonical."""

import numpy as np

from occipital_lobe.perception.recognition import FaceRecognizer, KnownFace
from occipital_lobe.perception.vision import _face_recognition_input


def test_face_recognition_uses_unmodified_sensor_frame_when_looking_out():
    """Synthetic pupil and Retinex transforms must not alter face embeddings."""
    sensor = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    result = _face_recognition_input(sensor, mirror=False)

    assert result is sensor
    np.testing.assert_array_equal(result, sensor)


def test_face_recognition_mirror_is_canonicalized_from_sensor_frame():
    """A mirrored look is unmirrored from the sensor, not adapted pixels."""
    sensor = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    result = _face_recognition_input(sensor, mirror=True)

    np.testing.assert_array_equal(result, sensor[:, ::-1])
    assert result.flags.c_contiguous


def test_sface_official_threshold_accepts_a_genuine_borderline_match(tmp_path):
    """Use OpenCV's published 0.363 cosine operating threshold."""
    recognizer = FaceRecognizer(data_dir=tmp_path)
    base = np.zeros(128, dtype=np.float32)
    base[0] = 1.0
    orthogonal = np.zeros(128, dtype=np.float32)
    orthogonal[1] = 1.0
    similarity = 0.38
    candidate = similarity * base + np.sqrt(1.0 - similarity**2) * orthogonal
    recognizer._known_faces["alice"] = [
        KnownFace(name="alice", embedding=base.tolist(), registered_at=0.0)
    ]

    name, confidence = recognizer._match_face(candidate)

    assert name == "alice"
    assert confidence > 0.0


def test_sface_rejects_embeddings_below_published_threshold(tmp_path):
    """A similarity below 0.363 must remain an unknown identity."""
    recognizer = FaceRecognizer(data_dir=tmp_path)
    base = np.zeros(128, dtype=np.float32)
    base[0] = 1.0
    orthogonal = np.zeros(128, dtype=np.float32)
    orthogonal[1] = 1.0
    similarity = 0.35
    candidate = similarity * base + np.sqrt(1.0 - similarity**2) * orthogonal
    recognizer._known_faces["alice"] = [
        KnownFace(name="alice", embedding=base.tolist(), registered_at=0.0)
    ]

    name, confidence = recognizer._match_face(candidate)

    assert name is None
    assert confidence == 0.0

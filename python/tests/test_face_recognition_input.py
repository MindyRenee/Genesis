"""Face identity input must remain photographic and canonical."""

import numpy as np

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

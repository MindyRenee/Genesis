"""Regression tests for face-recognizer ownership and persistence wiring."""

from __future__ import annotations

from pathlib import Path

from occipital_lobe.perception import vision as vision_mod
from occipital_lobe.perception.vision import Vision


def test_vision_initializes_face_recognizer_with_canonical_data_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """The recognizer must load and save identity data in Genesis's state tree."""
    data_dir = tmp_path / "genesis-data"
    captured: dict[str, Path] = {}

    class AvailableFaceRecognizer:
        def __init__(self, data_dir: str | Path | None = None) -> None:
            assert data_dir is not None
            captured["data_dir"] = Path(data_dir)

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(vision_mod, "default_data_dir", lambda: data_dir)
    monkeypatch.setattr(vision_mod, "FaceRecognizer", AvailableFaceRecognizer)

    recognizer = Vision()._get_face_recognizer()

    assert isinstance(recognizer, AvailableFaceRecognizer)
    assert captured["data_dir"] == data_dir

"""Anonymous face clustering — the familiar stranger.

An unrecognized face is still *someone*: embeddings cluster under
stable "face:N" keys so presence, aura, and field accrue before any
name. Naming a cluster folds its history into the named person.

These tests exercise the clustering math directly — no camera, no
cv2 models, no daemon.
"""

import numpy as np
from occipital_lobe.perception.recognition import FaceRecognizer


def _unit_embedding(seed: int) -> np.ndarray:
    """A deterministic unit embedding — stands in for an SFace vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(embedding: np.ndarray, jitter: float = 0.05) -> np.ndarray:
    """The same face again, slightly different — a resighting."""
    v = embedding + np.random.default_rng(7).standard_normal(128) * jitter
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_first_sighting_mints_anonymous_cluster(tmp_path):
    """An unknown face becomes "face:N" — new, stable, remembered."""
    rec = FaceRecognizer(data_dir=tmp_path)
    key, tier = rec._match_or_cluster(_unit_embedding(1))
    assert key == "face:1"
    assert tier == "new"
    assert rec.cluster_sightings("face:1") == 1
    # The cluster persisted — a restart would reload it.
    assert (tmp_path / "face_clusters.json").exists()


def test_resighting_is_familiar(tmp_path):
    """The same face again is *familiar* — recognized though unnamed."""
    rec = FaceRecognizer(data_dir=tmp_path)
    emb = _unit_embedding(1)
    key1, tier1 = rec._match_or_cluster(emb)
    key2, tier2 = rec._match_or_cluster(_near(emb))
    assert key2 == key1 == "face:1"
    assert tier1 == "new"
    assert tier2 == "familiar"
    assert rec.cluster_sightings("face:1") == 2


def test_different_faces_get_different_clusters(tmp_path):
    """Two strangers are two people — the registry doesn't collapse
    them into one anonymous blob."""
    rec = FaceRecognizer(data_dir=tmp_path)
    key_a, _ = rec._match_or_cluster(_unit_embedding(10))
    key_b, _ = rec._match_or_cluster(_unit_embedding(20))
    assert key_a != key_b
    assert {key_a, key_b} == {"face:1", "face:2"}


def test_clusters_survive_restart(tmp_path):
    """A familiar stranger is remembered across sessions."""
    emb = _unit_embedding(1)
    rec = FaceRecognizer(data_dir=tmp_path)
    key, _ = rec._match_or_cluster(emb)
    rec._match_or_cluster(_near(emb))  # second sighting confirms it
    rec2 = FaceRecognizer(data_dir=tmp_path)
    key2, tier = rec2._match_or_cluster(_near(emb))
    assert key2 == key
    assert tier == "familiar"
    assert rec2.cluster_sightings(key) == 3


def test_provisional_cluster_not_persisted(tmp_path):
    """A one-shot detection is provisional — usually detector noise —
    and must not survive a restart as a durable 'person'."""
    rec = FaceRecognizer(data_dir=tmp_path)
    key, _ = rec._match_or_cluster(_unit_embedding(1))
    assert rec.cluster_sightings(key) == 1
    rec2 = FaceRecognizer(data_dir=tmp_path)
    assert rec2.cluster_sightings(key) == 0


def test_name_face_folds_cluster_into_named(tmp_path):
    """Naming "face:1" → alice: the cluster's sightings become alice's
    known embeddings and the anonymous key retires."""
    rec = FaceRecognizer(data_dir=tmp_path)
    emb = _unit_embedding(1)
    key, _ = rec._match_or_cluster(emb)
    rec._match_or_cluster(_near(emb))  # second sighting
    folded = rec.name_face(key, "alice")
    assert folded == key
    assert rec.cluster_sightings(key) == 0  # retired
    assert "alice" in rec.known_names()
    assert len(rec._known_faces["alice"]) == 2
    # And now the same face resolves to the name, not the cluster.
    name, conf = rec._match_face(_near(emb))
    assert name == "alice"
    assert conf > 0.0


def test_name_face_unknown_cluster_is_none(tmp_path):
    """Naming a cluster that doesn't exist reports no merge."""
    rec = FaceRecognizer(data_dir=tmp_path)
    assert rec.name_face("face:99", "nobody") is None

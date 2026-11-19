"""Dumping her learned language into the space.

The six legacy boxes (``concept_vectors.npz``, ``embeddings.npz``,
``edge_log.jsonl``, ``holographic_graph.npz``, ``vq_codebook.npz``,
``concept_archive.db``) get emptied into the ``LanguageSpace`` — raw
contents, one deposit each, landed wherever the medium puts them.
The space is the store; ``language_space.npz`` is its only image.
Once poured, the boxes are thrown away.
"""

from __future__ import annotations

import io
import logging
import os
import sqlite3

import numpy as np

from .language_space import LanguageSpace, open_language_space

logger = logging.getLogger(__name__)

BOX_NAMES = (
    "concept_vectors.npz",
    "embeddings.npz",
    "edge_log.jsonl",
    "holographic_graph.npz",
    "vq_codebook.npz",
    "concept_archive.db",
)


def put_bytes(space: LanguageSpace, name: str, raw: bytes) -> None:
    """Deposit raw contents into the space (upsert by name)."""
    space.deposit(
        name,
        kind="content",
        payload=np.frombuffer(raw, dtype=np.uint8),
        meta={"name": name},
    )


def get_bytes(space: LanguageSpace, name: str) -> bytes | None:
    """Recall raw contents — None when the space doesn't hold them."""
    got = space.get(name)
    if got is None:
        return None
    payload, _ = got
    return payload.tobytes() if payload is not None else None


def npz_bytes(**arrays) -> bytes:
    """Serialize a set of named arrays into bytes."""
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def npz_from_bytes(raw: bytes) -> dict[str, np.ndarray]:
    """Parse .npz bytes with the same no-pickle rule as ``load_npz``."""
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def ingest_data_dir(space: LanguageSpace, data_dir: str | None = None) -> dict[str, int]:
    """Dump every language box in ``data_dir`` into the space.

    Idempotent — re-running re-deposits. Returns the per-box byte
    counts that landed.
    """
    ddir = data_dir or space.data_dir or "."
    counts: dict[str, int] = {}
    for name in BOX_NAMES:
        path = os.path.join(ddir, name)
        if not os.path.exists(path):
            continue
        if name == "concept_archive.db":
            # WAL-mode sqlite: the -wal sidecar holds recent rows, so
            # take the database's own serialized image (checkpointed)
            # rather than raw file bytes.
            conn = sqlite3.connect(path)
            try:
                raw = conn.serialize()
            finally:
                conn.close()
        else:
            with open(path, "rb") as fh:
                raw = fh.read()
        put_bytes(space, name, raw)
        counts[name] = len(raw)
    space.save()
    logger.info("language space ingested %d contents: %s", len(counts), counts)
    return counts


__all__ = [
    "BOX_NAMES",
    "get_bytes",
    "ingest_data_dir",
    "npz_bytes",
    "npz_from_bytes",
    "open_language_space",
    "put_bytes",
]

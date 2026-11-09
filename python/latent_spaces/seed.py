"""Seed the substrate from her persisted data dir — first boot only.

Her history already lives in the data dir: concept embeddings, the
cognitive journal, the LTM index. On first boot the manifold ingests
them as deposits — her past lands in the substrate at its own
coordinates (concept vectors projected, journal/LTM by digest), so
she wakes into a medium that already contains her life rather than
starting blank and mirroring stores beside it.

Every source is best-effort: a missing file or malformed row skips
cleanly — seeding must never block boot.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import geometry as g
from .manifold import LatentManifold

# Deposit bounds per source — enough to populate the terrain without
# a first-boot stall.
_MAX_CONCEPTS = 4000
_MAX_JOURNAL = 300
_MAX_LTM = 1000
_SEED_MIN_DEPOSITS = 64

_CONCEPT_SEED = 0xCE1


def seed_from_data_dir(manifold: LatentManifold) -> dict[str, int]:
    """Deposit her persisted state into the substrate, once.

    Runs only while the manifold is sparse — deposits are append-only
    so re-seeding every boot would pile up duplicates. Returns counts
    per source for the boot log.
    """
    if manifold.data_dir is None or len(manifold) >= _SEED_MIN_DEPOSITS:
        return {}
    counts: dict[str, int] = {}
    with manifold._lock:
        rung_base = manifold._next_rung
    for name, fn in (
        ("concepts", _seed_concepts),
        ("journal", _seed_journal),
        ("ltm", _seed_ltm),
        ("words", _seed_embeddings),
        ("graph", _seed_hgraph),
        ("archive", _seed_archive),
    ):
        try:
            counts[name] = fn(manifold, rung_base)
            rung_base += counts[name] if counts[name] > 0 else 0
        except Exception:  # noqa: BLE001
            counts[name] = -1
    # Seeded records are independent strata, not one exchange: each
    # gets its own rung, then the open exchange is left clean for
    # living flow.
    with manifold._lock:
        manifold._open_rung = -1
    return counts


def _seed_concepts(manifold: LatentManifold, rung_base: int) -> int:
    """Concept vocabulary at its own embedding coordinates."""
    path = os.path.join(manifold.data_dir or "", "concept_vectors.npz")
    if not os.path.exists(path):
        return 0
    data = np.load(path)
    names = [str(c) for c in data["concepts"]]
    vectors = np.asarray(data["vectors"], dtype=np.float64)
    n = min(len(names), len(vectors), _MAX_CONCEPTS)
    for i in range(n):
        try:
            manifold.observe(
                names[i],
                kind=f"concept:{names[i]}",
                salience=0.5,
                inbound=False,
                rung=rung_base + i,
                position=g.project_to_s3(vectors[i], seed=_CONCEPT_SEED),
            )
        except (ValueError, TypeError):
            continue
    return n


def _seed_journal(manifold: LatentManifold, rung_base: int) -> int:
    """Her narrative journal — the last entries as memory deposits."""
    path = os.path.join(manifold.data_dir or "", "cognitive_journal.jsonl")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-_MAX_JOURNAL:]
    n = 0
    for line in lines:
        try:
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            manifold.observe(
                text,
                kind=f"memory:journal_{row.get('kind', 'entry')}",
                salience=0.4,
                inbound=False,
                now_ms=int(row.get("ts", 0)) or None,
                rung=rung_base + n,
            )
            n += 1
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return n


def _seed_ltm(manifold: LatentManifold, rung_base: int) -> int:
    """The LTM index — episodes she has lived, retrievable by id."""
    path = os.path.join(manifold.data_dir or "", "ltm_store.meta")
    if not os.path.exists(path):
        return 0
    try:
        from genesis_client.ltm_index import scan_ltm_index
    except ImportError:
        return 0
    n = 0
    for entry in scan_ltm_index(path)[:_MAX_LTM]:
        if entry.is_deleted or entry.is_meta_memory:
            continue
        try:
            manifold.observe(
                f"episode:{entry.episode_id}",
                kind="memory:ltm",
                salience=max(0.0, min(1.0, float(entry.salience))),
                inbound=False,
                now_ms=int(entry.timestamp) or None,
                rung=rung_base + n,
            )
            n += 1
        except (ValueError, TypeError):
            continue
    return n


def _seed_embeddings(manifold: LatentManifold, rung_base: int) -> int:
    """Her vocabulary — EmbeddingStore words at their own coordinates."""
    path = os.path.join(manifold.data_dir or "", "embeddings.npz")
    if not os.path.exists(path):
        return 0
    data = np.load(path)
    words = [str(w) for w in data["words"]]
    vectors = np.asarray(data["vectors"], dtype=np.float64)
    n = min(len(words), len(vectors), 3000)
    for i in range(n):
        try:
            manifold.observe(
                words[i],
                kind=f"word:{words[i]}",
                salience=0.4,
                inbound=False,
                rung=rung_base + i,
                position=g.project_to_s3(vectors[i], seed=0xE0BED),
            )
        except (ValueError, TypeError):
            continue
    return n


def _seed_hgraph(manifold: LatentManifold, rung_base: int) -> int:
    """The holographic association graph — edges as deposits."""
    path = os.path.join(manifold.data_dir or "", "edge_log.jsonl")
    if not os.path.exists(path):
        return 0
    n = 0
    for line in open(path, encoding="utf-8"):
        try:
            row = json.loads(line)
            for e in row.get("edges", [])[:2000 - n]:
                s, r, t = str(e["source"]), str(e["relation"]), str(e["target"])
                manifold.observe(
                    f"{s} →{r}→ {t}",
                    kind=f"edge:{r}",
                    salience=0.5,
                    inbound=False,
                    rung=rung_base + n,
                )
                n += 1
                if n >= 2000:
                    return n
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return n


def _seed_archive(manifold: LatentManifold, rung_base: int) -> int:
    """The concept archive — her curated concept records."""
    import sqlite3

    path = os.path.join(manifold.data_dir or "", "concept_archive.db")
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        con = sqlite3.connect(path)
        cols = [r[1] for r in con.execute("PRAGMA table_info(concepts)")]
        name_col = next(
            (c for c in cols if c in ("name", "concept", "canonical", "label")),
            cols[0],
        )
        rows = con.execute(f"SELECT {name_col} FROM concepts LIMIT 2000").fetchall()
        con.close()
    except sqlite3.Error:
        return 0
    for (name,) in rows:
        try:
            manifold.observe(
                str(name),
                kind=f"archive:{name}",
                salience=0.5,
                inbound=False,
                rung=rung_base + n,
            )
            n += 1
        except (ValueError, TypeError):
            continue
    return n

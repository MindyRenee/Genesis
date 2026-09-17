"""Concept archive — SQLite-backed cold storage for dormant concepts.

The working-memory concept network is bounded: dormant concepts are
spilled here instead of being pruned outright, and recalled on demand
(see ``archival.py`` — the ``ConceptNetwork`` side of the protocol).

Two tables plus an alias index:

- ``concepts`` — ``concept_id`` → JSON-serialized concept dict
- ``aliases`` — alias → concept_id (case-insensitive lookup; the
  canonical ID is indexed as an alias too)
- ``edges`` — archived edges involving spilled concepts
  (``INSERT OR REPLACE`` makes batch archival idempotent)

The database lives at ``<data_dir>/concept_archive.db`` and runs in
WAL mode so the daemon's reader threads and the cognitive mind's
writer threads can coexist without blocking each other.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    PRIMARY KEY (alias, concept_id)
);
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL,
    created_at INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (source, target, relation)
);
"""


class ConceptArchive:
    """SQLite-backed long-term store for spilled concepts and edges.

    All mutating calls are serialized through an internal lock so the
    archive is safe to share between the mind's worker threads.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ─── Write path ───────────────────────────────────────────────

    def archive_concept(
        self,
        concept_id: str,
        data: dict[str, Any],
        aliases: set[str],
    ) -> None:
        """Store a concept dict and index its aliases.

        The canonical ID is always indexed as an alias so
        ``find_by_alias`` resolves both names and IDs. Aliases are
        normalized to lowercase for case-insensitive lookup.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO concepts (concept_id, data) VALUES (?, ?)",
                (concept_id, json.dumps(data)),
            )
            for alias in {concept_id, *aliases}:
                self._conn.execute(
                    "INSERT OR REPLACE INTO aliases (alias, concept_id) VALUES (?, ?)",
                    (alias.lower(), concept_id),
                )
            self._conn.commit()

    def archive_concepts_batch(
        self,
        items: list[tuple[str, dict[str, Any], set[str]]],
    ) -> int:
        """Archive many concepts in one transaction. Returns the count."""
        with self._lock:
            for concept_id, data, aliases in items:
                self._conn.execute(
                    "INSERT OR REPLACE INTO concepts (concept_id, data) VALUES (?, ?)",
                    (concept_id, json.dumps(data)),
                )
                for alias in {concept_id, *aliases}:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO aliases (alias, concept_id) "
                        "VALUES (?, ?)",
                        (alias.lower(), concept_id),
                    )
            self._conn.commit()
        return len(items)

    def archive_edges_batch(self, edge_dicts: list[dict[str, Any]]) -> int:
        """Archive edges (source, target, relation, weight, created_at,
        origin). Idempotent — INSERT OR REPLACE on the natural key."""
        with self._lock:
            for e in edge_dicts:
                self._conn.execute(
                    "INSERT OR REPLACE INTO edges "
                    "(source, target, relation, weight, created_at, origin) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        e["source"], e["target"], e["relation"],
                        e["weight"], e.get("created_at", 0), e.get("origin", ""),
                    ),
                )
            self._conn.commit()
        return len(edge_dicts)

    # ─── Read / recall path ───────────────────────────────────────

    def count(self) -> int:
        """Number of archived concepts."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM concepts").fetchone()
        return int(row[0]) if row else 0

    def has_concept(self, concept_id: str) -> bool:
        """True if the concept is in the archive."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM concepts WHERE concept_id = ?", (concept_id,),
            ).fetchone()
        return row is not None

    def get_all_ids(self) -> set[str]:
        """All archived concept IDs."""
        with self._lock:
            rows = self._conn.execute("SELECT concept_id FROM concepts").fetchall()
        return {r[0] for r in rows}

    def get_all_concepts(self) -> list[tuple[str, dict[str, Any]]]:
        """All archived ``(concept_id, data)`` pairs."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT concept_id, data FROM concepts",
            ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    def find_by_alias(self, alias: str) -> list[str]:
        """Resolve an alias (or canonical ID) to archived concept IDs.

        Case-insensitive: aliases are stored lowercased.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT concept_id FROM aliases WHERE alias = ?",
                (alias.lower(),),
            ).fetchall()
        return [r[0] for r in rows]

    def recall_concept(self, concept_id: str) -> dict[str, Any] | None:
        """Remove a concept from the archive and return its data dict.

        Recall is destructive — the concept is moving back into working
        memory, so its archive row and alias index entries are deleted.
        Returns ``None`` if the concept is not archived.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM concepts WHERE concept_id = ?", (concept_id,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "DELETE FROM concepts WHERE concept_id = ?", (concept_id,),
            )
            self._conn.execute(
                "DELETE FROM aliases WHERE concept_id = ?", (concept_id,),
            )
            self._conn.commit()
        return json.loads(row[0])

    def recall_edges(self, concept_id: str) -> list[dict[str, Any]]:
        """Remove and return archived edges touching a concept.

        Called when a concept is recalled to working memory so its
        relationships come back with it.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT source, target, relation, weight, created_at, origin "
                "FROM edges WHERE source = ? OR target = ?",
                (concept_id, concept_id),
            ).fetchall()
            self._conn.execute(
                "DELETE FROM edges WHERE source = ? OR target = ?",
                (concept_id, concept_id),
            )
            self._conn.commit()
        return [
            {
                "source": r[0], "target": r[1], "relation": r[2],
                "weight": r[3], "created_at": r[4], "origin": r[5],
            }
            for r in rows
        ]

    def remove_concepts_batch(self, concept_ids: list[str]) -> int:
        """Permanently delete archived concepts and their alias/edge rows.

        Used to drop stale archive rows for concepts that are present
        in working memory (e.g. restored from a save file written
        before a spill ran). One transaction; returns the number of
        concept rows deleted.
        """
        if not concept_ids:
            return 0
        with self._lock:
            self._conn.executemany(
                "DELETE FROM concepts WHERE concept_id = ?",
                [(c,) for c in concept_ids],
            )
            self._conn.executemany(
                "DELETE FROM aliases WHERE concept_id = ?",
                [(c,) for c in concept_ids],
            )
            self._conn.executemany(
                "DELETE FROM edges WHERE source = ? OR target = ?",
                [(c, c) for c in concept_ids],
            )
            self._conn.commit()
        return len(concept_ids)

    def remove_edges_for_concept(self, concept_id: str) -> int:
        """Delete archived edges touching a concept (permanent removal)."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM edges WHERE source = ? OR target = ?",
                (concept_id, concept_id),
            )
            self._conn.commit()
        return cur.rowcount

    # ─── Maintenance ──────────────────────────────────────────────

    def vacuum(self) -> None:
        """Reclaim space after large deletions."""
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ConceptArchive:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


def open_archive(data_dir: str | Path) -> ConceptArchive:
    """Open (creating if needed) the concept archive under ``data_dir``.

    The database file is ``<data_dir>/concept_archive.db``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return ConceptArchive(data_dir / "concept_archive.db")
